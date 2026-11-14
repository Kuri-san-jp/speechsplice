#!/usr/bin/env python3
"""複数エンティティ同時 splice の音声生成（BL / INJ_1 / INJ_2 / INJ_12）.

`scripts/projection/eval_utmos_speaker_fixed.py`（= 公開版
`table1_generate_wordrepl.py`）を土台にした**新規**スクリプト。
既存ファイルは 1 行も変更していない。再利用できる部品
（`get_encoder_output_with_prefix` / `voice_clone_synth` / `score_wav` /
`seed_all` / SEED / DEFAULT_REF）は import してそのまま使うので、
話者固定・decode config・duration 推定・Whisper prefix の扱いは既存と定義上同一。

1 文あたり 4 条件を**同一 seed** で生成する:

  BL     注入なし。target_text = text_a
  INJ_1  スロット 1 だけ splice。ref = text_refs[0] の encoder 出力
  INJ_2  スロット 2 だけ splice。ref = text_refs[1] の encoder 出力
  INJ_12 両方 splice。スロット i の ref は ref_i（**スパンごとに別の forward**）

INJ_12 で「両方置換した 1 本の文」から 2 スパンを取らないのが要点。
そうすると干渉の原因（splice の相互作用 vs. ref 側の文脈が既に汚れている）が
分離できなくなる。

encoder forward は 1 文あたり 3 回（src, ref_1, ref_2）。decoder 生成は 4 回。

前提と固定事項
--------------
* duration は `duration_source_text=text_a` で固定（Δduration=0.00s 維持）。
  外すと MCD が壊れる。
* スパンは `splice_multi.resolve_spans_with_cursor` で**全スパンまとめて**
  target_text 上で解決し、`target_start_in_full` を**全スパンに一度だけ**加算する。
* 解決後に `verify_span_roundtrip` で往復一致を確認し、落ちた文は**丸ごと skip**。
  「末尾 N トークンを entity とみなす」fallback は使わない。
* position は長さ正規化された進行度ランプなので T が変わると全 token が動く。
  総ずれ delta_tok を条件ごとに記録する。

使い方
------
  SLURM_TIME=06:00:00 slurm_run.sh \
      python scripts/analysis/generate_multientity.py \
          --pairs_module en_multientity_300 --lang en \
          --reference_wav eval_results/_natural_refs_norm/en_natural_norm.wav \
          --reference_text "<transcript>"

出力: eval_results/<pairs_module>_multi/{checkpoint.json,results.json,*.wav}
"""
import argparse
import json
import os
import sys
import time

# **スレッド上限。torch を読む前に置く必要がある。**
# 2026-09-18、この種のジョブを並べたらログインノードの CPU を埋めて
# 他の利用者に迷惑をかけた（note/2026-09-18_gpuoverrun_debug.md）。
# **上限は 128 スレッドの 1/4 = 32**（2026-09-18 ユーザ指示）。
# GPU を 3 枚まで同時に使うので、1 本あたり 10 にして合計 30 に収める。
# 実測では GPU 律速で 1 コアしか使わないが、上限として置く。
_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr


def _autoroot():
    env = os.environ.get("SPEECHSPLICE_ROOT")
    if env:
        return env
    if os.path.isdir("/mnt/scripts"):
        return "/mnt"
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        if os.path.isdir(os.path.join(cand, "data")) and \
           os.path.isdir(os.path.join(cand, "scripts")):
            return cand
    return "."


PROJECT_ROOT = _autoroot()
# splice_clone / splice_multi / eval_utmos_speaker_fixed はすべてここ
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))

# eval_utmos_speaker_fixed を import した時点で HF_HOME / offline / sys.path
# （vendor t5gemma, scripts, data）がすべて既存と同じ設定になる。
# transformers 等より先に import すること。
from eval_utmos_speaker_fixed import (  # noqa: E402
    DEFAULT_REF, SEED, get_encoder_output_with_prefix, score_wav, seed_all,
    voice_clone_synth,
)
from splice_multi import (  # noqa: E402
    prepare_clone_splice_multi, resolve_spans_with_cursor, verify_span_roundtrip,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)
import torchaudio  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from encoder_injection_t5gemma import seed_everything  # noqa: E402
from inference_tts_utils import save_audio, transcribe_audio  # noqa: E402
from data.tokenizer import AudioTokenizer  # noqa: E402

CONDITIONS = ["bl", "inj1", "inj2", "inj12"]
# 条件名 -> 使うスロット index のリスト（0-based）
COND_SLOTS = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}

# **tgt を 2026-09-19 に追加（--with_tgt のときだけ）。**
# 目標文 text_both を**注入なしで**合成する。bl と同じ経路で、渡す文だけが違う。
#
# なぜ要るか: 複数 entity の判定は ASR の書き起こしに目標語の表記が出るかを
# 見ているが、中国語ではこれが厳しすぎた。注入は届いているのに
# ASR が別の漢字を当てるため落ちる（王 -> 山田 が "Yamada" と書き起こされる、
# 呉 -> 加藤 が "加通" になる）。単一 entity の中国語セルは PER（音素の
# 誤り率）で「読みが目標に近づいたか」を測っており、**指標が違う。**
# tgt があれば複数 entity でも同じ PER を測れる。
# 詳細は note/2026-09-19_multientity_zh_report.md。
#
# **既存の 4 条件の生成は 1 ビットも変わらない。**--with_tgt を渡したときだけ
# 5 つ目が増える（seed は各条件の直前で戻すので、条件を足しても他に影響しない）。
TGT_COND = "tgt"
# **枠ごとの目標文（2026-09-20 追加）。**上の tgt は text_both（2 枠とも
# 差し替えた文）で、基線との差が 2 箇所になる。音素列の差分整列は
# 「間に挟まる一致が 3 個以下なら同じ塊」と結合するので、隣り合う 2 枠
# （300 対中 50 対は間の語数が 0）では 1 つの塊に融合して枠を分けられない。
# tgt1 / tgt2 はそれぞれ 1 枠だけ差し替えた文なので差が 1 箇所になり、
# 既存の diff_block がそのまま効く。**本体セルと同じ手続きで測れる。**
TGT_SLOT_CONDS = {"tgt1": 0, "tgt2": 1}


def _local_snapshot(hub_id):
    hub = os.environ.get("HF_HUB_CACHE") or \
        os.path.join(os.environ["HF_HOME"], "hub")
    repo_dir = os.path.join(hub, f"models--{hub_id.replace('/', '--')}",
                            "snapshots")
    if os.path.isdir(repo_dir):
        snaps = sorted(os.listdir(repo_dir))
        for s in reversed(snaps):
            if os.path.exists(os.path.join(repo_dir, s, "config.json")):
                return os.path.join(repo_dir, s)
        if snaps:
            return os.path.join(repo_dir, snaps[-1])
    return hub_id


def load_items(pairs_module, n):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
    mod = __import__(pairs_module)
    items = mod.build_pairs()
    for it in items:
        assert len(it["spans"]) == 2, \
            f"pair {it['pair_idx']}: this script expects exactly 2 spans"
        assert len(it["text_refs"]) == 2
    return items[:n] if n else items


def resolve_item_spans(text_tok, item, src_enc, ref_encs, lang):
    """src 側 2 スパンと ref 側 1 スパン×2 を解決し、絶対 index にして返す.

    返り値: (s_ranges, r_ranges, detail) / 失敗時 (None, None, reason)
    """
    surfaces_a = [sp["word_a"] for sp in item["spans"]]
    starts_a = [sp["char_start_a"] for sp in item["spans"]]

    # --- src: 2 スパンを「まとめて」1 回で解決する ---
    s_sub, s_info = resolve_spans_with_cursor(
        text_tok, src_enc["target_text"], surfaces_a, add_bos=0,
        char_starts=starts_a)
    if s_sub is None:
        return None, None, f"src span resolve failed: {s_info['reason']}"
    ok, bad = verify_span_roundtrip(text_tok, src_enc["target_text"], s_sub,
                                    surfaces_a, add_bos=0)
    if not ok:
        return None, None, f"src span roundtrip failed: {bad}"

    # --- ref: スパンごとに別の forward。それぞれ 1 スパンだけ解決 ---
    r_sub = []
    for i, sp in enumerate(item["spans"]):
        sub, info = resolve_spans_with_cursor(
            text_tok, ref_encs[i]["target_text"], [sp["word_b"]], add_bos=0,
            char_starts=[sp["char_start_b"]])
        if sub is None:
            return None, None, f"ref[{i}] span resolve failed: {info['reason']}"
        ok, bad = verify_span_roundtrip(text_tok, ref_encs[i]["target_text"],
                                        sub, [sp["word_b"]], add_bos=0)
        if not ok:
            return None, None, f"ref[{i}] span roundtrip failed: {bad}"
        r_sub.append(sub[0])

    # --- BOS/prefix/x_sep のオフセットを「全スパンに一度だけ」足す ---
    s_off = src_enc["target_start_in_full"]
    s_ranges = [(a + s_off, b + s_off) for (a, b) in s_sub]
    r_ranges = []
    for i, (a, b) in enumerate(r_sub):
        r_off = ref_encs[i]["target_start_in_full"]
        r_ranges.append((a + r_off, b + r_off))

    if s_ranges[0][1] > s_ranges[1][0]:
        return None, None, (f"src spans overlap: {s_ranges[0]} {s_ranges[1]} "
                            f"({surfaces_a})")

    detail = {
        "src": [list(x) for x in s_ranges],
        "ref": [list(x) for x in r_ranges],
        "n": [s_ranges[i][1] - s_ranges[i][0] for i in range(2)],
        "m": [r_ranges[i][1] - r_ranges[i][0] for i in range(2)],
        "gap_tokens": s_ranges[1][0] - s_ranges[0][1],
        "src_target_start": s_off,
        "ref_target_start": [ref_encs[i]["target_start_in_full"] for i in range(2)],
        "src_resolve_via": [p["via"] for p in s_info["per_span"]],
    }
    detail["delta_per_span"] = [detail["m"][i] - detail["n"][i] for i in range(2)]
    detail["len_imbalance"] = abs(detail["m"][0] - detail["m"][1])
    return s_ranges, r_ranges, detail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_module", required=True,
                   help="data/<name>.py の build_pairs()。2 スパン形式であること")
    # zh を 2026-09-18 に追加。根拠:
    #  * `normalize_text_with_lang(text, "zh")` は恒等写像
    #    (inference_tts_utils.py:130 が ja のときだけ正規化する)。
    #    char_start が target_text 上でずれない。
    #  * `scripts/analysis/verify_zh_multientity_cpu.py` が
    #    300 件 600 スパンを 3 トークン化器 (T5Gemma / umt5 / Llasa) で解決済み。
    #    「的」「当」の融合は最小対称拡張で救え、従来通っていたスパンは
    #    すべて plain のまま（既存結果は 1 ビットも変わらない）。
    p.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--reference_wav", default=DEFAULT_REF)
    p.add_argument("--reference_text", default="",
                   help="空なら Whisper で転写する（既存挙動と同じ）")
    p.add_argument("--reference_prompt_sec", type=float, default=100.0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ckpt_every", type=int, default=10)
    # **目標文だけを後から足す（2026-09-20 追加）。**既存 4 条件は 300 対とも
    # 生成済みで、再開は対単位なので通常経路では何も作られない。
    # --tgt_only は対単位の再開を無視し、s<i>_tgt1.wav / s<i>_tgt2.wav が
    # 揃っていない対にだけ目標文を合成する。既存の wav と results.json には
    # 一切触れない（書き出しは results_tgt.json、帳面は checkpoint_tgt.json）。
    p.add_argument("--tgt_only", action="store_true",
                   help="目標文だけを合成する。既存の bl/inj は作り直さない")
    p.add_argument("--with_tgt", action="store_true",
                   help="目標文 text_both を注入なしで合成し s<i>_tgt.wav に出す。"
                        "PER で測るために要る。**既存 4 条件の出力は変わらない。**")
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"eval_results/{args.pairs_module}_multi"
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(SEED)

    items = load_items(args.pairs_module, args.n)
    print(f"[Info] {args.pairs_module}: {len(items)} items, lang={args.lang}, "
          f"ref={args.reference_wav}", flush=True)

    REF_PREFIX = args.reference_text or ""
    if not REF_PREFIX:
        print("[Info] Transcribing reference audio via Whisper ...", flush=True)
        REF_PREFIX = transcribe_audio(args.reference_wav, device)
        print(f"[Info] Whisper transcript: {REF_PREFIX}", flush=True)

    print("[Info] Loading T5Gemma-TTS ...", flush=True)
    model_id = _local_snapshot("Aratako/T5Gemma-TTS-2b-2b")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="auto", local_files_only=True)
    model.eval()
    cfg = model.config
    tkn_hub_id = getattr(cfg, "text_tokenizer_name", None) or \
        getattr(cfg, "t5gemma_model_name", None)
    tkn_path = _local_snapshot(tkn_hub_id) if tkn_hub_id else tkn_hub_id
    text_tok = AutoTokenizer.from_pretrained(tkn_path, local_files_only=True)
    assert getattr(text_tok, "is_fast", False), (
        "a fast tokenizer is required: span resolution uses offset_mapping and "
        "there is deliberately no character-length fallback")
    audio_tok = AudioTokenizer(backend="xcodec2",
                               model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    print("[Info] Loading UTMOS22 ...", flush=True)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True).to(device).eval()
    print("[Info] Loading SQUIM-OBJECTIVE ...", flush=True)
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    squim = SQUIM_OBJECTIVE.get_model().to(device).eval()

    rows, skipped = [], []
    # **--tgt_only は別の帳面を使う。**同じ checkpoint.json に書くと、
    # 既存 4 条件の再開記録（300 対ぶん）が目標文だけの記録で上書きされる。
    ckpt = os.path.join(args.output_dir,
                        "checkpoint_tgt.json" if args.tgt_only
                        else "checkpoint.json")
    if os.path.exists(ckpt) and not args.tgt_only:
        st = json.load(open(ckpt))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != "splice_multi"]
        assert not bad, (f"checkpoint mode mismatch in {ckpt}: pairs {bad[:5]} "
                         f"were not produced by this script; use a separate "
                         f"--output_dir")
        print(f"[Info] resume: {len(rows)} rows, {len(skipped)} skipped",
              flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}
    if args.tgt_only:
        # 対単位の済み印を捨て、**目標文の wav がある対だけ**を済みとする。
        args.with_tgt = True
        rows, skipped = [], []
        done = {it["pair_idx"] for it in items
                if all(os.path.exists(os.path.join(
                    args.output_dir, "s%04d_%s.wav" % (it["pair_idx"], c)))
                    for c in TGT_SLOT_CONDS)}
        print("[Info] tgt_only: 目標文が既にある対 %d / 作る対 %d"
              % (len(done), len(items) - len(done)), flush=True)
    t0 = time.time()

    for it in items:
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        seed_everything(SEED)
        text_a = it["text_a"]
        refs = it["text_refs"]

        try:
            src_enc = get_encoder_output_with_prefix(
                model, text_tok, REF_PREFIX, text_a, device, lang=args.lang)
            ref_encs = [get_encoder_output_with_prefix(
                model, text_tok, REF_PREFIX, r, device, lang=args.lang)
                for r in refs]
        except Exception as e:
            print(f"  [{pidx}] enc fail: {str(e)[:160]}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"enc: {e}"})
            done.add(pidx)
            continue

        # **--tgt_only ではスパンを解決しない（2026-09-20 修正）。**
        # 目標文は注入せずそのまま読ませるだけなので、実体語の区間は要らない。
        # それなのに解決を走らせていたため、助詞が癒着する対
        # （'は大塚' != '大塚'、'金子さん' != '金子'）で落ち、
        # 日本語 15 対・中国語 30 対の目標音声が作られなかった。
        if args.tgt_only:
            s_ranges, r_ranges, detail = [], [], {"skipped_for_tgt_only": True}
        else:
            s_ranges, r_ranges, detail = resolve_item_spans(
                text_tok, it, src_enc, ref_encs, args.lang)
            if s_ranges is None:
                print(f"  [{pidx}] SKIP {detail}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": detail})
                done.add(pidx)
                continue

        row = {
            "pair_idx": pidx, "mode": "splice_multi",
            "template_id": it.get("template_id"), "lang": args.lang,
            "text_a": text_a, "text_refs": refs, "text_both": it["text_both"],
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "repeat_surface": it.get("repeat_surface", False),
            "words_a": [sp["word_a"] for sp in it["spans"]],
            "words_b": [sp["word_b"] for sp in it["spans"]],
            "span_tok": detail,
            "T_src": int(src_enc["last_hidden_state"].shape[1]),
            "conds": {},
        }

        # --- 4 条件を同一 seed で生成（--with_tgt なら目標文も）---
        ok = True
        if args.tgt_only:
            conds_to_run = list(TGT_SLOT_CONDS)
        else:
            conds_to_run = list(CONDITIONS)
            if args.with_tgt:
                conds_to_run.append(TGT_COND)
        for cond in conds_to_run:
            if cond in TGT_SLOT_CONDS:
                # 枠ごとの目標文。注入なしで、bl と同じ経路で文だけ差し替える。
                slots = []
                modified, inj_dur_text = None, None
                inj_target = refs[TGT_SLOT_CONDS[cond]]
                delta_tok, out_ranges = 0, []
            elif cond == TGT_COND:
                # 目標文を注入なしで合成する。bl と同じ経路で文だけ差し替える。
                slots = []
                modified, inj_target, inj_dur_text = None, it["text_both"], None
                delta_tok, out_ranges = 0, []
            else:
                slots = COND_SLOTS[cond]
                modified, inj_target, inj_dur_text = None, text_a, None
                delta_tok, out_ranges = 0, []
            if slots:
                sub_s = [s_ranges[i] for i in slots]
                sub_r = [r_ranges[i] for i in slots]
                sub_ref = [ref_encs[i] for i in slots]
                # 長さ一致ダミーの自然候補: 使うスロットだけ置換した文
                nat = (it["text_both"],) if len(slots) == 2 else (refs[slots[0]],)
                try:
                    (modified, _t, inj_dur_text, delta_tok,
                     out_ranges) = prepare_clone_splice_multi(
                        text_tok, src_enc, sub_ref, sub_s, sub_r, args.lang,
                        duration_source_text=text_a, natural_candidates=nat)
                except Exception as e:
                    print(f"  [{pidx}] {cond} splice prep fail: {str(e)[:160]}",
                          flush=True)
                    skipped.append({"pair_idx": pidx,
                                    "reason": f"{cond} prep: {e}"})
                    ok = False
                    break
                if _t is not None:
                    inj_target = _t

            try:
                seed_everything(SEED)
                audio = voice_clone_synth(
                    model, text_tok, audio_tok,
                    target_text=inj_target, lang=args.lang, device=device,
                    codec_audio_sr=codec_audio_sr, codec_sr=codec_sr,
                    ref_wav=args.reference_wav, ref_text=REF_PREFIX,
                    ref_prompt_sec=args.reference_prompt_sec,
                    modified_hidden=modified, duration_text=inj_dur_text)
            except Exception as e:
                print(f"  [{pidx}] {cond} synth fail: {str(e)[:160]}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} synth: {e}"})
                ok = False
                break

            wav_path = os.path.join(args.output_dir, f"s{pidx:04d}_{cond}.wav")
            try:
                save_audio(wav_path,
                           audio if audio.dim() > 1 else audio.unsqueeze(0),
                           codec_audio_sr)
                u, pe = score_wav(wav_path, codec_audio_sr, utmos, squim, device)
                info = torchaudio.info(wav_path)
                row["conds"][cond] = {
                    "wav": os.path.basename(wav_path),
                    "delta_tok": int(delta_tok),
                    "T_out": int(modified.shape[1]) if modified is not None
                    else row["T_src"],
                    "out_ranges": [list(x) for x in out_ranges],
                    "target_text": inj_target,
                    "dur_text_is_text_a": (inj_dur_text == text_a)
                    if inj_dur_text is not None else True,
                    "utmos": u, "pesq": pe,
                    "dur_sec": round(info.num_frames / info.sample_rate, 4),
                }
            except Exception as e:
                print(f"  [{pidx}] {cond} score fail: {str(e)[:160]}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} score: {e}"})
                ok = False
                break

        done.add(pidx)
        if not ok:
            continue
        # 総 position ずれ（INJ_12 の delta_tok）。長さ正規化ランプなので
        # この値が 0 でなければ全 token の position が動いている。
        # **--tgt_only では inj12 を作っていない。**position ずれは注入条件の
        # 指標なので、目標文だけのときは記録しない。
        if not args.tgt_only:
            row["total_position_drift"] = row["conds"]["inj12"]["delta_tok"]
        rows.append(row)

        if len(rows) % 5 == 0:
            c = row["conds"]
            if args.tgt_only:
                print(f"  [{len(rows)}/{len(items) - len(done)}] idx={pidx} "
                      f"tgt1 {c['tgt1']['dur_sec']:.2f}s "
                      f"tgt2 {c['tgt2']['dur_sec']:.2f}s", flush=True)
            else:
                print(f"  [{len(rows)}/{len(items)}] idx={pidx} "
                      f"gap_tok={detail['gap_tokens']} "
                      f"delta12={row['total_position_drift']} "
                      f"U(bl/1/2/12)="
                      f"{c['bl']['utmos']:.2f}/{c['inj1']['utmos']:.2f}/"
                      f"{c['inj2']['utmos']:.2f}/{c['inj12']['utmos']:.2f}",
                      flush=True)
        if len(rows) % args.ckpt_every == 0:
            with open(ckpt, "w") as f:
                json.dump({"rows": rows, "skipped": skipped}, f,
                          ensure_ascii=False, indent=1)

    with open(ckpt, "w") as f:
        json.dump({"rows": rows, "skipped": skipped}, f, ensure_ascii=False,
                  indent=1)

    def _mean(key):
        v = [r["conds"][key]["utmos"] for r in rows
             if r["conds"].get(key, {}).get("utmos") is not None]
        return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] if v \
            else [None, None]

    # **集計は実際に作った条件だけを見る。**--tgt_only では bl / inj* を
    # 作っていないので、CONDITIONS 固定で回すと KeyError で落ちる。
    summ_conds = list(TGT_SLOT_CONDS) if args.tgt_only else list(CONDITIONS)
    summary = {
        "pairs_module": args.pairs_module, "lang": args.lang,
        "tgt_only": bool(args.tgt_only),
        "conditions": summ_conds,
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "reference_wav": args.reference_wav,
        "utmos": {c: _mean(c) for c in summ_conds},
        "mean_dur_sec": {c: round(float(np.mean(
            [r["conds"][c]["dur_sec"] for r in rows])), 4) for c in summ_conds}
        if rows else {},
        # position ずれは注入条件の指標。目標文だけのときは測る対象が無い。
        "mean_abs_position_drift": (None if args.tgt_only else (
            round(float(np.mean(
                [abs(r["total_position_drift"]) for r in rows])), 3)
            if rows else None)),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    out_name = "results_tgt.json" if args.tgt_only else "results.json"
    with open(os.path.join(args.output_dir, out_name), "w") as f:
        json.dump({"summary": summary, "rows": rows, "skipped": skipped}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n=== multi-entity generation {args.pairs_module} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if skipped:
        print(f"[Warn] {len(skipped)} items skipped; see results.json['skipped']")


if __name__ == "__main__":
    main()
