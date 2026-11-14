#!/usr/bin/env python3
"""最小対称拡張で「落ちた対だけ」を作り直す投入係（複数 entity 版）.

**`scripts/analysis/generate_multientity.py` は 1 行も変更していない。**
生成の中身（話者固定 / decode config / duration / seed / スコアリング）は
すべて `generate_multientity` から import して使うので、既存 285 対と
定義上同一の条件で動く。このファイルが足すのは 2 点だけ。

1. スパン解決を `resolve_spans_with_cursor(..., allow_symmetric_extension=True)`
   で行う（規則の実装は `scripts/projection/span_extend.py` の 1 箇所だけ。
   ここでは呼ぶだけで書き直していない）。
2. `--only_pairs` で指定した対**だけ**を生成し、既存の checkpoint を
   引き継いで `results.json` を更新する。既存 wav は触らない。

なぜ既存 285 対が変わらないか
-----------------------------
`minimal_symmetric_surface` は語そのままで往復一致する対に `plain=True` を
返す。この場合 要求する表層が語そのままなので、解決されるトークン区間は
既定経路と同じになる。本スクリプトはそれを**実行時に assert で確認**する
（`_assert_plain_matches_default`）。加えて `--only_pairs` に既に row を
持つ対を渡すと `--force` なしでは拒否するので、285 対の wav は
上書きされない。

CPU での事前確認（GPU 不要）
---------------------------
    taskset -c 0-63 python3 scripts/projection/verify_multientity_ext_cpu.py
    taskset -c 0-63 python3 scripts/analysis/generate_multientity_ext.py \\
        --pairs_module ja_multientity_300 --lang ja \\
        --output_dir eval_results/multientity_ja_300 --dry_run

GPU 投入（依頼者が実行する）
----------------------------
    SLURM_TIME=02:00:00 slurm_run.sh \\
        python3 -u /mnt/scripts/analysis/generate_multientity_ext.py \\
            --pairs_module ja_multientity_300 --lang ja \\
            --output_dir /mnt/eval_results/multientity_ja_300 \\
            --reference_text "基礎川はしばしば日本のライン川と呼ばれている" \\
            --only_pairs 70,85,129,144,148,174,179,194,198,215,224,229,244,265,274

`--reference_text` は既存 285 対の run で Whisper が出した転写そのもの
（`logs/autoqueue/multient_ja.log`）。これを渡すと prefix が 10 token になり
`target_start_in_full == 11` が既存 285 行と一致する。一致しなければ
`--expect_target_start` の assert で止まるので、prefix が違ったまま
15 対だけ別条件で作ってしまう事故は起きない。
"""
import argparse
import json
import os
import shutil
import sys
import time

# HF_HOME は**代入**で上書きする（setdefault では slurm の -e HF_HOME に負ける）。
# 値は eval_utmos_speaker_fixed.py と同一の規則にしてある。
_HF_HOME = "/mnt/.cache/huggingface" if os.path.isdir("/mnt/scripts") \
    else os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 既存スクリプトから「生成の中身」をまるごと借りる。import した時点で
# HF_HOME / offline / sys.path（vendor t5gemma, scripts, data）は
# 既存 285 対の run と同じ状態になる。
from generate_multientity import (  # noqa: E402
    CONDITIONS, COND_SLOTS, PROJECT_ROOT, _local_snapshot, load_items,
    resolve_item_spans,
)
from generate_multientity import (  # noqa: E402
    DEFAULT_REF, SEED, get_encoder_output_with_prefix, score_wav, seed_all,
    voice_clone_synth,
)
from generate_multientity import (  # noqa: E402
    AudioTokenizer, np, save_audio, seed_everything, torch, torchaudio,
    transcribe_audio,
)
from generate_multientity import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from splice_multi import (  # noqa: E402
    prepare_clone_splice_multi, resolve_spans_with_cursor, verify_span_roundtrip,
)
from inference_tts_utils import normalize_text_with_lang  # noqa: E402

SPAN_MODE = "minimal_symmetric_extension"


# --- スパン解決（拡張あり） -------------------------------------------------

def _assert_plain_matches_default(text_tok, full_text, surfaces, char_starts,
                                  spans, ext, where):
    """拡張が plain だったスパンは、既定経路と同じ区間でなければならない.

    これが「今まで通っていた対の結果が 1 ビットも変わらない」ことの実行時の
    担保。plain なのに区間が動いたら配線が壊れているので即座に落とす。
    """
    if not all(e["plain"] for e in ext):
        return
    dflt, info = resolve_spans_with_cursor(text_tok, full_text, surfaces,
                                           add_bos=0, char_starts=char_starts)
    assert dflt == spans, (
        f"{where}: all extensions are plain but spans moved: default={dflt} "
        f"extended={spans} (info={info.get('reason')})")


def resolve_item_spans_ext(text_tok, item, src_enc, ref_encs, lang,
                           max_pre=3, max_post=3):
    """`generate_multientity.resolve_item_spans` の拡張版.

    違いは `allow_symmetric_extension=True` を渡すことと、往復一致の検査を
    **拡張後の表層**で行うことだけ。返り値の形は元の関数と同じ
    (s_ranges, r_ranges, detail) / 失敗時 (None, None, reason)。
    detail には "extension" が増える。
    """
    spans_meta = item["spans"]
    surfaces_a = [sp["word_a"] for sp in spans_meta]
    starts_a = [sp["char_start_a"] for sp in spans_meta]
    surfaces_b = [sp["word_b"] for sp in spans_meta]
    starts_b = [sp["char_start_b"] for sp in spans_meta]
    text_a = src_enc["target_text"]
    texts_b = [e["target_text"] for e in ref_encs]

    # --- src: 2 スパンを「まとめて」1 回で解決する（元と同じ流儀） ---
    s_sub, s_info = resolve_spans_with_cursor(
        text_tok, text_a, surfaces_a, add_bos=0, char_starts=starts_a,
        allow_symmetric_extension=True,
        peer_texts=texts_b, peer_surfaces=surfaces_b, peer_char_starts=starts_b,
        side="a", max_pre=max_pre, max_post=max_post)
    if s_sub is None:
        return None, None, f"src span resolve failed (ext): {s_info['reason']}"
    ext_a = s_info["extension"]
    eff_a = [e["surface"].strip() for e in ext_a]
    ok, bad = verify_span_roundtrip(text_tok, text_a, s_sub, eff_a, add_bos=0)
    if not ok:
        return None, None, f"src span roundtrip failed (ext): {bad}"
    _assert_plain_matches_default(text_tok, text_a, surfaces_a, starts_a,
                                  s_sub, ext_a, "src")

    # --- ref: スパンごとに別の forward。それぞれ 1 スパンだけ解決 ---
    r_sub, ext_b = [], []
    for i, sp in enumerate(spans_meta):
        sub, info = resolve_spans_with_cursor(
            text_tok, texts_b[i], [sp["word_b"]], add_bos=0,
            char_starts=[sp["char_start_b"]],
            allow_symmetric_extension=True,
            peer_texts=[text_a], peer_surfaces=[sp["word_a"]],
            peer_char_starts=[sp["char_start_a"]],
            side="b", max_pre=max_pre, max_post=max_post)
        if sub is None:
            return None, None, f"ref[{i}] span resolve failed (ext): {info['reason']}"
        e = info["extension"]
        ok, bad = verify_span_roundtrip(text_tok, texts_b[i], sub,
                                        [e[0]["surface"].strip()], add_bos=0)
        if not ok:
            return None, None, f"ref[{i}] span roundtrip failed (ext): {bad}"
        _assert_plain_matches_default(text_tok, texts_b[i], [sp["word_b"]],
                                      [sp["char_start_b"]], sub, e, f"ref[{i}]")
        # 両側で同じ (pre, post) と同じ余りの文字が選ばれていなければ、
        # 差し替えたときに機能語が二重になる。数値を合わせる補正はせず落とす。
        if (e[0]["pre"], e[0]["post"], e[0]["extra"]) != \
           (ext_a[i]["pre"], ext_a[i]["post"], ext_a[i]["extra"]):
            return None, None, (
                f"span {i}: asymmetric extension src="
                f"(pre={ext_a[i]['pre']}, post={ext_a[i]['post']}, "
                f"extra={ext_a[i]['extra']!r}) ref="
                f"(pre={e[0]['pre']}, post={e[0]['post']}, "
                f"extra={e[0]['extra']!r})")
        r_sub.append(sub[0])
        ext_b.append(e[0])

    # --- BOS/prefix/x_sep のオフセットを「全スパンに一度だけ」足す ---
    s_off = src_enc["target_start_in_full"]
    s_ranges = [(a + s_off, b + s_off) for (a, b) in s_sub]
    r_ranges = []
    for i, (a, b) in enumerate(r_sub):
        r_off = ref_encs[i]["target_start_in_full"]
        r_ranges.append((a + r_off, b + r_off))

    if s_ranges[0][1] > s_ranges[1][0]:
        return None, None, (f"src spans overlap after extension: {s_ranges[0]} "
                            f"{s_ranges[1]} ({eff_a})")

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
    detail["extension"] = [
        {"slot": i + 1, "word_a": surfaces_a[i], "word_b": surfaces_b[i],
         "surface_a": ext_a[i]["surface"], "surface_b": ext_b[i]["surface"],
         "pre": ext_a[i]["pre"], "post": ext_a[i]["post"],
         "extra": ext_a[i]["extra"], "plain": ext_a[i]["plain"]}
        for i in range(2)]
    detail["all_plain"] = all(x["plain"] for x in detail["extension"])
    return s_ranges, r_ranges, detail


# --- checkpoint の引き継ぎ ---------------------------------------------------

def load_checkpoint(ckpt):
    if not os.path.exists(ckpt):
        return [], []
    st = json.load(open(ckpt))
    rows = st.get("rows", [])
    skipped = st.get("skipped", [])
    bad = [r["pair_idx"] for r in rows if r.get("mode") != "splice_multi"]
    assert not bad, (f"checkpoint mode mismatch in {ckpt}: pairs {bad[:5]} "
                     f"were not produced by generate_multientity(_ext); "
                     f"use a separate --output_dir")
    return rows, skipped


def build_summary(args, items, rows, skipped, elapsed_min):
    """`generate_multientity.main` の summary をそのままの形で作り直す."""
    def _mean(key):
        v = [r["conds"][key]["utmos"] for r in rows
             if r["conds"].get(key, {}).get("utmos") is not None]
        return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] if v \
            else [None, None]

    ext_rows = [r for r in rows if r.get("span_mode") == SPAN_MODE]
    return {
        "pairs_module": args.pairs_module, "lang": args.lang,
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "reference_wav": args.reference_wav,
        "utmos": {c: _mean(c) for c in CONDITIONS},
        "mean_dur_sec": {c: round(float(np.mean(
            [r["conds"][c]["dur_sec"] for r in rows])), 4) for c in CONDITIONS}
        if rows else {},
        "mean_abs_position_drift": round(float(np.mean(
            [abs(r["total_position_drift"]) for r in rows])), 3) if rows else None,
        "elapsed_min": elapsed_min,
        "span_extension": {
            "rule": SPAN_MODE,
            "impl": "scripts/projection/span_extend.py:minimal_symmetric_surface",
            "n_rows_with_extension": len(ext_rows),
            "pairs": sorted(r["pair_idx"] for r in ext_rows),
            "extras": sorted({x["extra"] for r in ext_rows
                              for x in r["span_tok"]["extension"] if x["extra"]}),
        },
    }


# --- CPU だけの下見 ---------------------------------------------------------

def dry_run(args, items, targets):
    """モデルを読まずにスパン解決だけ確かめる（CPU 専用・GPU 不要）.

    `get_encoder_output_with_prefix` は target_text に
    `normalize_text_with_lang` を通すだけなので、同じ正規化を自分で当てれば
    スパン解決は GPU なしで完全に再現できる。
    """
    model_id = _local_snapshot("Aratako/T5Gemma-TTS-2b-2b")
    cfg = json.load(open(os.path.join(model_id, "config.json")))
    tkn = cfg.get("text_tokenizer_name") or cfg.get("t5gemma_model_name")
    text_tok = AutoTokenizer.from_pretrained(_local_snapshot(tkn),
                                             local_files_only=True)
    assert getattr(text_tok, "is_fast", False), "fast tokenizer required"

    def fake_enc(text):
        n, _ = normalize_text_with_lang(text, args.lang)
        return {"target_text": n,
                "target_start_in_full": args.expect_target_start or 0}

    n_ok = 0
    out = []
    for it in items:
        if it["pair_idx"] not in targets:
            continue
        src_enc = fake_enc(it["text_a"])
        ref_encs = [fake_enc(r) for r in it["text_refs"]]
        s_old, _, d_old = resolve_item_spans(text_tok, it, src_enc, ref_encs,
                                             args.lang)
        s, r, d = resolve_item_spans_ext(text_tok, it, src_enc, ref_encs,
                                         args.lang, args.max_pre, args.max_post)
        rec = {"pair_idx": it["pair_idx"], "default_ok": s_old is not None,
               "ext_ok": s is not None,
               "default_src": d_old["src"] if s_old is not None else None,
               "default_ref": d_old["ref"] if s_old is not None else None}
        if s is None:
            rec["reason"] = d
        else:
            n_ok += 1
            rec.update({"src": d["src"], "ref": d["ref"], "n": d["n"],
                        "m": d["m"], "delta_per_span": d["delta_per_span"],
                        "gap_tokens": d["gap_tokens"],
                        "all_plain": d["all_plain"],
                        "extension": [x for x in d["extension"]]})
        out.append(rec)
        print(f"  [{it['pair_idx']:3d}] default={'ok' if s_old else 'FAIL'} "
              f"ext={'ok' if s else 'FAIL'} "
              + ("" if s is None else
                 " ".join(f"slot{x['slot']}:{x['word_a']}->{x['word_b']}"
                          f"({x['surface_a']!r}/{x['surface_b']!r}"
                          f" pre={x['pre']} post={x['post']})"
                          for x in d["extension"] if not x["plain"])),
              flush=True)
    path = os.path.join(args.output_dir, "_dry_run_span_ext.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[dry_run] {n_ok}/{len(targets)} pairs resolve with extension")
    print(f"[dry_run] -> {path}")
    return 0 if n_ok == len(targets) else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_module", default="ja_multientity_300",
                   help="data/<name>.py の build_pairs()。2 スパン形式であること")
    # zh を 2026-09-18 に追加。中国語の本番で 31 対が生成時のスパン往復確認で
    # 落ちるため、**この補充 runner が無いと 300 に届かない**（日本語で
    # 285 -> 300 を埋めたのと同じ役割）。根拠:
    #  * `normalize_text_with_lang(text, "zh")` は恒等写像
    #    (inference_tts_utils.py:130 が ja のときだけ正規化する)。
    #    このファイルは 271 行で args.lang を渡すので、zh でも整合する。
    #  * `scripts/analysis/verify_zh_multientity_cpu.py` が 300 件 600 スパンを
    #    4 トークン化器 (T5Gemma / umt5 / Llasa / Qwen3) で解決済み。
    #    落ちている 31 対は最小対称拡張（`的` `当` 等）で救える。
    p.add_argument("--lang", default="ja", choices=["en", "ja", "zh"])
    p.add_argument("--output_dir", default="eval_results/multientity_ja_300",
                   help="既存 checkpoint.json / wav があるディレクトリ")
    p.add_argument("--only_pairs", default="",
                   help="作り直す pair_idx のカンマ区切り。空なら checkpoint の "
                        "skipped 全件")
    p.add_argument("--reference_wav", default=DEFAULT_REF)
    p.add_argument("--reference_text", default="",
                   help="空なら Whisper で転写する。既存 285 対と同じ prefix に "
                        "するため、元の run の転写を渡すこと")
    p.add_argument("--reference_prompt_sec", type=float, default=100.0)
    p.add_argument("--expect_target_start", type=int, default=11,
                   help="prefix が既存と同じかの検査。0 で無効化")
    p.add_argument("--max_pre", type=int, default=3)
    p.add_argument("--max_post", type=int, default=3)
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--dry_run", action="store_true",
                   help="スパン解決だけ CPU で確認する（モデルを読まない）")
    p.add_argument("--force", action="store_true",
                   help="既に row / wav がある対も作り直す（既定では拒否）")
    args = p.parse_args()

    os.chdir(PROJECT_ROOT)
    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"output_dir does not exist: {args.output_dir}\n"
                         f"このスクリプトは既存 run の続きだけを作る。"
                         f"最初から作る場合は generate_multientity.py を使う。")
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    rows, skipped = load_checkpoint(ckpt)
    have_row = {r["pair_idx"] for r in rows}
    skipped_idx = [s["pair_idx"] for s in skipped]

    if args.only_pairs.strip():
        targets = [int(x) for x in args.only_pairs.replace(" ", "").split(",") if x]
    else:
        targets = sorted(skipped_idx)
    targets = sorted(set(targets))
    if not targets:
        raise SystemExit("nothing to do: --only_pairs is empty and checkpoint "
                         "has no skipped pairs")

    items = load_items(args.pairs_module, 0)
    known = {it["pair_idx"] for it in items}
    unknown = [t for t in targets if t not in known]
    assert not unknown, f"--only_pairs has unknown pair_idx: {unknown}"

    print(f"[Info] {args.pairs_module}: {len(items)} items, lang={args.lang}",
          flush=True)
    print(f"[Info] checkpoint: {len(rows)} rows, {len(skipped)} skipped "
          f"({sorted(skipped_idx)})", flush=True)
    print(f"[Info] targets ({len(targets)}): {targets}", flush=True)

    # dry_run は wav を書かないので、既存 row を持つ対にも当てられる
    # （285 対で拡張が起きないことを確かめるのに使う）。
    if args.dry_run:
        return dry_run(args, items, set(targets))

    # --- 既存 285 対を守る ---
    clash = [t for t in targets if t in have_row]
    if clash and not args.force:
        raise SystemExit(f"refusing to regenerate pairs that already have rows: "
                         f"{clash}\n既存 wav を作り直さないのが合格条件。"
                         f"本当にやるなら --force。")
    wav_clash = [f"s{t:04d}_{c}.wav" for t in targets for c in CONDITIONS
                 if os.path.exists(os.path.join(args.output_dir,
                                                f"s{t:04d}_{c}.wav"))]
    if wav_clash and not args.force:
        raise SystemExit(f"refusing to overwrite existing wavs: {wav_clash[:8]}"
                         f"\n本当にやるなら --force。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(SEED)

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

    by_idx = {it["pair_idx"]: it for it in items}
    new_rows, still_bad = [], []
    t0 = time.time()

    for pidx in targets:
        it = by_idx[pidx]
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
            still_bad.append({"pair_idx": pidx, "reason": f"enc: {e}"})
            continue

        # prefix が既存 285 対と同じでなければ止める（別条件の混入防止）
        if args.expect_target_start:
            got = [src_enc["target_start_in_full"]] + \
                  [e["target_start_in_full"] for e in ref_encs]
            assert all(g == args.expect_target_start for g in got), (
                f"target_start_in_full={got} != --expect_target_start="
                f"{args.expect_target_start}: reference prefix differs from the "
                f"existing 285 rows. Pass the same --reference_text as the "
                f"original run.")

        s_ranges, r_ranges, detail = resolve_item_spans_ext(
            text_tok, it, src_enc, ref_encs, args.lang,
            args.max_pre, args.max_post)
        if s_ranges is None:
            print(f"  [{pidx}] SKIP {detail}", flush=True)
            still_bad.append({"pair_idx": pidx, "reason": detail})
            continue

        row = {
            "pair_idx": pidx, "mode": "splice_multi",
            "span_mode": SPAN_MODE if not detail["all_plain"] else "plain",
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

        # --- 4 条件を同一 seed で生成（generate_multientity と同じ流儀） ---
        ok = True
        for cond in CONDITIONS:
            slots = COND_SLOTS[cond]
            modified, inj_target, inj_dur_text = None, text_a, None
            delta_tok, out_ranges = 0, []
            if slots:
                sub_s = [s_ranges[i] for i in slots]
                sub_r = [r_ranges[i] for i in slots]
                sub_ref = [ref_encs[i] for i in slots]
                nat = (it["text_both"],) if len(slots) == 2 else (refs[slots[0]],)
                try:
                    (modified, _t, inj_dur_text, delta_tok,
                     out_ranges) = prepare_clone_splice_multi(
                        text_tok, src_enc, sub_ref, sub_s, sub_r, args.lang,
                        duration_source_text=text_a, natural_candidates=nat)
                except Exception as e:
                    print(f"  [{pidx}] {cond} splice prep fail: {str(e)[:160]}",
                          flush=True)
                    still_bad.append({"pair_idx": pidx,
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
                still_bad.append({"pair_idx": pidx, "reason": f"{cond} synth: {e}"})
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
                still_bad.append({"pair_idx": pidx, "reason": f"{cond} score: {e}"})
                ok = False
                break

        if not ok:
            continue
        row["total_position_drift"] = row["conds"]["inj12"]["delta_tok"]
        new_rows.append(row)
        ex = " ".join(f"slot{x['slot']}{x['extra']!r}"
                      for x in detail["extension"] if not x["plain"])
        c = row["conds"]
        print(f"  [{len(new_rows)}/{len(targets)}] idx={pidx} ext={ex or 'plain'} "
              f"gap_tok={detail['gap_tokens']} "
              f"delta12={row['total_position_drift']} "
              f"U(bl/1/2/12)={c['bl']['utmos']:.2f}/{c['inj1']['utmos']:.2f}/"
              f"{c['inj2']['utmos']:.2f}/{c['inj12']['utmos']:.2f}", flush=True)

        if len(new_rows) % args.ckpt_every == 0:
            _save(args, ckpt, items, rows, new_rows, skipped, still_bad, targets,
                  t0, final=False)

    _save(args, ckpt, items, rows, new_rows, skipped, still_bad, targets, t0,
          final=True)
    return 0


def _save(args, ckpt, items, old_rows, new_rows, old_skipped, still_bad,
          targets, t0, final):
    """checkpoint と results.json を更新する（既存 row はそのまま残す）."""
    done = {r["pair_idx"] for r in new_rows}
    rows = sorted(old_rows + new_rows, key=lambda r: r["pair_idx"])
    # 作り直せた対は skipped から外す。作り直せなかった対は理由を差し替える。
    bad_by_idx = {b["pair_idx"]: b for b in still_bad}
    skipped = []
    for s in old_skipped:
        i = s["pair_idx"]
        if i in done:
            continue
        skipped.append(bad_by_idx.get(i, s))
    for i, b in bad_by_idx.items():
        if i not in {s["pair_idx"] for s in skipped} and i not in done:
            skipped.append(b)
    skipped.sort(key=lambda s: s["pair_idx"])

    with open(ckpt, "w") as f:
        json.dump({"rows": rows, "skipped": skipped}, f, ensure_ascii=False,
                  indent=1)
    if not final:
        return

    res = os.path.join(args.output_dir, "results.json")
    bak = os.path.join(args.output_dir, "results_before_span_ext.json")
    if os.path.exists(res) and not os.path.exists(bak):
        shutil.copy2(res, bak)   # 拡張前の 285 件をそのまま残す
        print(f"[Info] backed up previous results.json -> {bak}", flush=True)

    summary = build_summary(args, items, rows, skipped,
                            round((time.time() - t0) / 60, 2))
    with open(res, "w") as f:
        json.dump({"summary": summary, "rows": rows, "skipped": skipped}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n=== multi-entity span-extension {args.pairs_module} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[Info] added {len(new_rows)}/{len(targets)} pairs; "
          f"rows {len(old_rows)} -> {len(rows)}; skipped {len(skipped)}")
    if skipped:
        print(f"[Warn] still skipped: {[s['pair_idx'] for s in skipped]}")


if __name__ == "__main__":
    sys.exit(main())
