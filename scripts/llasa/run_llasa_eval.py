#!/usr/bin/env python3
"""Llasa-3B に SpeechSplice を適用して cross-culture 300 ペアを生成・評価する (GPU).

生成するもの (既存 3 モデル + Qwen3-TTS 行と同じ命名規約):
    {wav_dir}/s{pair_idx:04d}_bl.wav  = TTS(text_a)
    {wav_dir}/s{pair_idx:04d}_inj.wav = SpeechSplice(text_a)
    {wav_dir}/s{pair_idx:04d}_tgt.wav = TTS(text_b)
3 本そろっていないペアはスコアラ側で available:false として捨てられるので、
TGT の生成を省略してはならない (mcd_pysptk.py:228-229, phoneme_per.py:166-167)。

指標は既存ファイルをそのまま使う (1 行も改変していない):
    MCD : speechsplice_release/src/mcd_pysptk.py --mode whole
    PIB : speechsplice_release/src/phoneme_per.py
--run_scorers を付けると生成後にサブプロセスで呼ぶ。後から単体で回してもよい。
MCD (--mode whole) は numpy/scipy だけなので CPU で回せる。PIB は
wav2vec2-xlsr-53-espeak の forced alignment を使うので GPU がある方が速い。

統制:
  * seed 42 を BL / TGT / INJ の generate 直前に必ずリセットする。
  * 3 条件で復号設定 (do_sample / temperature / top_p / top_k / max_new_tokens) と
    prompt 構築を完全に共有する。BL と INJ は prompt が 1 token も違わない
    (同じ text_a、同じ参照音声トークン)。差は prefill 中の splice だけ。
  * voice clone の参照音声は Table-3 の他アーキテクチャと同じ自然音声
    (eval_results/_natural_refs_norm/en_natural_norm.wav)。

--span_extension を付けると、語そのままでは往復一致しない対 (トークン化器が実体語と
隣の機能語を 1 トークンに融合する対。中国語の「的」など) に限り最小対称拡張
(scripts/projection/span_extend.py) を試す。既定は off で従来どおり落とす。
往復一致する対は on/off に関わらず同じスパン・同じ prompt・同じ結果になる。

実行例 (GPU):
  SLURM_TIME=04:00:00 slurm_run.sh \\
    python3 scripts/llasa/run_llasa_eval.py --lang en --n 300 --layer 26 \\
      --allow_download --run_scorers
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
# スコアラ (mcd_pysptk.py / phoneme_per.py) が読むのと同じ <PROJECT_ROOT>/data を先に見る。
# speechsplice_release/data/ の同名ファイルとは内容が一致していることを確認済み。
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "llasa"))

import llasa_splice as L  # noqa: E402  (HF_HOME はこの import で設定される)

import numpy as np  # noqa: E402
import torch  # noqa: E402

MCD_SCORER = os.path.join(PROJECT_ROOT, "src", "mcd_pysptk.py")
PIB_SCORER = os.path.join(PROJECT_ROOT, "src", "phoneme_per.py")

# Table-3 の全アーキテクチャで共有している自然音声の参照
# (speechsplice_release/configs/voice_prompts.yaml と同じ)
DEFAULT_REFS = {
    "en": (os.path.join(PROJECT_ROOT, "eval_results", "_natural_refs_norm", "en_natural_norm.wav"),
           "Let me see, I'll give them a new pair of boots every Christmas."),
    # 日本語 (Llasa-1B-Multilingual 用、2026-09-18 追加)。
    # JSUT BASIC5000_0079 の公式書き起こし。他アーキテクチャの JA セルと同じ参照。
    "ja": (os.path.join(PROJECT_ROOT, "eval_results", "_natural_refs_norm",
                        "ja_natural_norm.wav"),
           "太平洋は、世界でいちばん大きな大洋です。"),
    # 中国語は BZNSYP 側を既定にする (2026-09-18)。
    # zh_natural_norm.wav は書き起こしが「钙钾等元素属于金属银和金等元素」で
    # 句読点が無く元素名の羅列になっており、これを参照にすると生成が止まらない
    # (3 対で BL 66.7% / TGT 100% が max_new_tokens=1024 に張り付き、
    #  1 文に 15〜20 秒かかった)。BZNSYP 側の「印度素有嫁女儿陪送嫁妆的传统。」に
    # 変えると上限張り付きが 0/3 になり、生成長も 3.3〜4.4 秒に収まる。
    "zh": (os.path.join(PROJECT_ROOT, "eval_results", "_natural_refs_norm",
                        "zh_natural_bznsyp_norm.wav"),
           "印度素有嫁女儿陪送嫁妆的传统。"),
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_pairs(lang, n):
    """Table 3 (cross-architecture) で既存モデルが使っているデータそのもの.

    pair_idx は最初から入っている (0..299)。mcd_pysptk.py / phoneme_per.py が
    内部で同じ関数を読むので、無改造でスコアラに渡せる。
    """
    if lang == "en":
        from en_cross_culture_names import build_300_pairs
    elif lang == "zh":
        from zh_cross_culture_names import build_300_pairs
    elif lang == "ja":
        # 日本語は Llasa-3B では非対応。HKUSTAudio/Llasa-1B-Multilingual で使う
        # (2026-09-18 追加)。同 repo は README の language に ja を含み、学習データの
        # Emilia 側に日本語が入っている。**--model_id を明示すること。**
        # 3B のまま --lang ja を指定すると学習していない言語を測ることになる。
        from ja_cross_culture_names import build_300_pairs
    else:
        raise SystemExit(f"--lang {lang} は対象外 (en / ja / zh)。")
    pairs = build_300_pairs()[:n]
    for i, p in enumerate(pairs):
        p.setdefault("pair_idx", i)
    return pairs


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def save_wav(path, audio, sr):
    """PCM_16 で保存。途中で落ちても壊れた wav が残らないよう tmp -> rename."""
    import soundfile as sf
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        raise RuntimeError(f"empty audio for {path}")
    n_clip = int(np.sum(np.abs(a) > 1.0))
    a = np.clip(a, -1.0, 1.0)
    tmp = path + ".tmp"
    sf.write(tmp, a, sr, format="WAV", subtype="PCM_16")
    os.replace(tmp, path)
    return int(a.size), float(a.size / sr), float(n_clip / a.size)


def atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def wav_paths(wav_dir, pair_idx):
    return (os.path.join(wav_dir, f"s{pair_idx:04d}_bl.wav"),
            os.path.join(wav_dir, f"s{pair_idx:04d}_inj.wav"),
            os.path.join(wav_dir, f"s{pair_idx:04d}_tgt.wav"))


def condition_tag(args):
    tag = f"{args.lang}_L{args.layer}"
    tag += "_greedy" if not args.do_sample else f"_t{args.temperature:g}_p{args.top_p:g}"
    if args.ref_mode == "none":
        tag += "_noref"
    if args.seed != L.SEED:
        tag += f"_seed{args.seed}"
    return tag


def link_or_copy(src, dst):
    """BL / TGT は層に依存しないので、層掃引では 1 層目の結果を持ち込める.

    hard link を試し、駄目ならコピーする。中身は同一なのでスコアラから見て等価。
    """
    if os.path.exists(dst) or not os.path.exists(src):
        return False
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return True


# --------------------------------------------------------------------------
# kappa aggregation
# --------------------------------------------------------------------------
def summarize_kappa(rows, num_layers):
    """kappa の層ごと集計 (entity ごとの値を層ごとに単純平均).

    index 0 = embedding 出力、index l+1 = layers[l] の出力。
    splice の対象は layers[layer] の出力なので、その層の値は
    kappa_at_splice = by_layer[layer+1] に一致する。
    母数・題材は既存 Table 3 の kappa セルと揃っていないので直接比較はできない。
    """
    per_layer, at_splice, last = {}, [], []
    for r in rows:
        if r.get("kappa_at_splice") is not None:
            at_splice.append(r["kappa_at_splice"])
        if r.get("kappa_last_layer") is not None:
            last.append(r["kappa_last_layer"])
        for li, v in enumerate(r.get("kappa_by_layer") or []):
            per_layer.setdefault(li, []).append(v)

    def stat(vals):
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0:
            return None
        return {"n": int(arr.size), "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std(ddof=1)), 6) if arr.size > 1 else None,
                "median": round(float(np.median(arr)), 6)}

    return {
        "by_hidden_index": {str(li): stat(v) for li, v in sorted(per_layer.items())},
        "at_splice_layer": stat(at_splice),
        "last_layer": stat(last),
        "note": ("kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}). "
                 "hidden index 0 = embedding output, index l+1 = output of layers[l] "
                 f"(num_layers={num_layers}). The spliced tensor is the output of "
                 "layers[layer], i.e. hidden index layer+1. Corpus/N differ from the "
                 "existing Table 3 kappa cells; not directly comparable."),
    }


def summarize_gen_length(rows, max_new_tokens, frame_rate=L.CODEC_FRAME_RATE):
    """条件ごとの生成長 (speech token 数) と上限張り付き率をまとめる.

    Llasa は <|SPEECH_GENERATION_END|> が出るまで自己回帰する。参照音声の
    トークンが壊れていたり prompt が噛み合っていないと EOS が出ず、
    max_new_tokens まで走り切って 1 発話が max_new_tokens/frame_rate 秒
    (既定 1024/50 = 20.48 s) になる。壁時計がこれに比例するので、
    上限張り付き率を manifest に残して後から見えるようにする。
    """
    out = {"max_new_tokens": int(max_new_tokens),
           "max_sec_at_cap": round(max_new_tokens / float(frame_rate), 3)}
    for cond in ("bl", "tgt", "inj"):
        vals = [r[f"n_tok_{cond}"] for r in rows if r.get(f"n_tok_{cond}") is not None]
        hits = [bool(r.get(f"hit_max_{cond}")) for r in rows
                if r.get(f"n_tok_{cond}") is not None]
        if not vals:
            out[cond] = None
            continue
        arr = np.array(vals, dtype=np.float64)
        out[cond] = {
            "n": int(arr.size),
            "mean_tokens": round(float(arr.mean()), 1),
            "median_tokens": round(float(np.median(arr)), 1),
            "p95_tokens": round(float(np.percentile(arr, 95)), 1),
            "max_tokens": int(arr.max()),
            "mean_sec": round(float(arr.mean()) / frame_rate, 3),
            "n_hit_max": int(sum(hits)),
            "hit_max_rate": round(sum(hits) / float(len(hits)), 4),
        }
    return out


def report_gen_length(summary, warn_rate=0.2):
    """生成長の集計を print し、上限に張り付いていたら警告を出す."""
    cap = summary["max_new_tokens"]
    print(f"[genlen] max_new_tokens={cap} (= {summary['max_sec_at_cap']}s at "
          f"{L.CODEC_FRAME_RATE} frames/sec)")
    for cond in ("bl", "tgt", "inj"):
        s = summary.get(cond)
        if not s:
            continue
        print(f"[genlen] {cond.upper():3s} n={s['n']} mean={s['mean_tokens']} tok "
              f"({s['mean_sec']}s) median={s['median_tokens']} p95={s['p95_tokens']} "
              f"hit_max={s['n_hit_max']}/{s['n']} ({s['hit_max_rate'] * 100:.1f}%)")
        if s["hit_max_rate"] > warn_rate:
            print(f"[genlen][warn] {cond.upper()} の {s['hit_max_rate'] * 100:.1f}% が "
                  f"max_new_tokens={cap} に張り付いている。EOS が出ていない "
                  f"(参照トークン / prompt / 復号設定を疑うこと)。壁時計はこの分だけ伸びる。")


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def build_ref(bundle, args):
    if args.ref_mode == "none":
        return None
    wav, text = DEFAULT_REFS[args.lang]
    wav = args.ref_wav or wav
    text = args.ref_text if args.ref_text is not None else text
    if not text:
        raise SystemExit(
            f"--lang {args.lang} の参照文が未設定。--ref_text を与えること "
            f"(voice clone では参照音声の書き起こしが本文の前に入る)。")
    if not os.path.exists(wav):
        raise SystemExit(f"参照音声が見つからない: {wav}")
    enc = L.encode_reference_wav(bundle, wav, max_sec=args.ref_max_sec)
    enc["ref_text"] = text
    ids = enc["speech_ids"]
    print(f"[ref] {wav} ({enc['dur_sec']}s) -> {enc['n_tokens']} speech tokens "
          f"(unique={len(set(ids))}, id_min={min(ids)}, id_max={max(ids)})", flush=True)
    # ここは --no_codec_check でも必ず通す。参照トークンが縮退したまま完走すると
    # 生成音声は話者性を持たず、しかも EOS が出ずに max_new_tokens まで走る。
    if min(ids) == max(ids):
        raise SystemExit(
            f"参照音声の speech token が全部同じ id ({min(ids)})。XCodec2 の "
            f"エンコーダが壊れている (act.beta -> act.bias の改名を確認すること)。")
    if not args.no_codec_check:
        rep = L.codec_roundtrip_check(bundle, enc)
        print(f"[ref] XCodec2 roundtrip: {rep}", flush=True)
        enc["roundtrip"] = rep
    return enc


def load_bundle(args):
    """層掃引では 1 回だけ読んで使い回す (毎層読み直すのは時間の無駄なので)."""
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] {args.model_id} dtype={args.dtype} device={args.device} "
          f"HF_HOME={os.environ['HF_HOME']} allow_download={args.allow_download}", flush=True)
    t_load = time.time()
    bundle = L.load_llasa(model_id=args.model_id, codec_id=args.codec_id,
                          device=args.device, dtype=dtype,
                          allow_download=args.allow_download,
                          attn_implementation=args.attn_implementation)
    print(f"[load] done in {time.time() - t_load:.1f}s  model_dir={bundle.model_dir}", flush=True)
    print(f"[load] layers={bundle.num_layers} hidden={bundle.hidden_size} "
          f"vocab={bundle.info['vocab_size']} arch={bundle.info['architectures']}", flush=True)
    return bundle


def run_generation(args, pairs, bundle=None):
    os.makedirs(args.wav_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.json")

    rows = {}
    if os.path.exists(progress_path) and not args.overwrite:
        try:
            with open(progress_path) as f:
                for r in json.load(f).get("rows", []):
                    rows[str(r["pair_idx"])] = r
            print(f"[resume] loaded {len(rows)} rows from {progress_path}", flush=True)
        except Exception as e:
            print(f"[resume] could not read {progress_path}: {e}", flush=True)

    own_bundle = bundle is None
    if own_bundle:
        bundle = load_bundle(args)
    if not (0 <= args.layer < bundle.num_layers):
        raise SystemExit(f"--layer {args.layer} が範囲外 (0..{bundle.num_layers - 1})")

    gen = L.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      temperature=args.temperature, top_p=args.top_p,
                      top_k=args.top_k, max_new_tokens=args.max_new_tokens)
    print(f"[gen] layer={args.layer} {gen.as_dict()}", flush=True)

    ref = build_ref(bundle, args)

    todo = pairs
    if args.num_shards > 1:
        todo = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_index]
        print(f"[shard] {args.shard_index}/{args.num_shards}: {len(todo)} of {len(pairs)} pairs",
              flush=True)

    t0 = time.time()
    n_done = n_skip = n_err = 0
    for k, pair in enumerate(todo):
        pidx = pair["pair_idx"]
        bl_p, inj_p, tgt_p = wav_paths(args.wav_dir, pidx)
        if args.link_bl_tgt_from:
            sbl, _, stgt = wav_paths(args.link_bl_tgt_from, pidx)
            link_or_copy(sbl, bl_p)
            link_or_copy(stgt, tgt_p)
        need_bl = args.overwrite or not os.path.exists(bl_p)
        need_tgt = args.overwrite or not os.path.exists(tgt_p)
        need_inj = args.overwrite or not os.path.exists(inj_p)
        if not (need_bl or need_tgt or need_inj):
            n_skip += 1
            if str(pidx) not in rows:
                rows[str(pidx)] = {"pair_idx": pidx, "word_a": pair["word_a"],
                                   "word_b": pair["word_b"], "status": "resumed_wavs_only"}
            continue
        try:
            tp = time.time()
            # prompt / span / hidden state は 1 ペアにつき 1 回だけ作り、
            # BL と INJ で同じ input_ids を使い回す (差が splice だけであることの保証)。
            in_a = L.build_input_text(pair["text_a"], ref["ref_text"] if ref else None)
            in_b = L.build_input_text(pair["text_b"], ref["ref_text"] if ref else None)
            ref_ids = ref["speech_ids"] if ref else None
            # --span_extension を付けたときだけ、語そのままで往復一致しない対に
            # 最小対称拡張を試す (scripts/projection/span_extend.py)。
            # 往復一致する対は span_extension に関わらず同じスパンを通る。
            src, tgt, span_ext = L.prepare_context_pair(
                bundle, in_a, pair["word_a"], in_b, pair["word_b"],
                ref_speech_ids=ref_ids,
                region_text_a=pair["text_a"], region_text_b=pair["text_b"],
                need_hidden_a=not args.no_kappa, need_hidden_b=True,
                strict_span=True, span_extension=args.span_extension)

            row = {"pair_idx": pidx, "status": "ok",
                   "word_a": pair["word_a"], "word_b": pair["word_b"],
                   "text_a": pair["text_a"], "text_b": pair["text_b"],
                   "prompt_len_a": src["prompt_len"], "prompt_len_b": tgt["prompt_len"],
                   "s_span": [src["span"]["tok_start"], src["span"]["tok_end"]],
                   "t_span": [tgt["span"]["tok_start"], tgt["span"]["tok_end"]],
                   "s_tokens": src["span"]["entity_tokens"],
                   "t_tokens": tgt["span"]["entity_tokens"]}
            if span_ext is not None:
                # 原稿に「何対で拡張が使われたか」を書くための記録。
                row["span_extension"] = span_ext

            if need_bl:
                g = L.generate_speech_ids(bundle, src["input_ids"], gen=gen)
                a = L.decode_speech_ids(bundle, g["speech_ids"])
                _, d, c = save_wav(bl_p, a, bundle.sampling_rate)
                row.update(dur_bl_sec=round(d, 3), clip_bl=round(c, 6),
                           n_tok_bl=len(g["speech_ids"]),
                           hit_max_bl=g["hit_max_new_tokens"])
            if need_tgt:
                g = L.generate_speech_ids(bundle, tgt["input_ids"], gen=gen)
                a = L.decode_speech_ids(bundle, g["speech_ids"])
                _, d, c = save_wav(tgt_p, a, bundle.sampling_rate)
                row.update(dur_tgt_sec=round(d, 3), clip_tgt=round(c, 6),
                           n_tok_tgt=len(g["speech_ids"]),
                           hit_max_tgt=g["hit_max_new_tokens"])
            if need_inj:
                inj = L.splice_synthesize(
                    bundle, pair["text_a"], pair["word_a"],
                    pair["text_b"], pair["word_b"], layer=args.layer, ref=ref,
                    gen=gen, collect_kappa=not args.no_kappa, strict_span=True,
                    src_ctx=src, tgt_ctx=tgt,
                    span_extension=args.span_extension)
                _, d, c = save_wav(inj_p, inj["audio"], bundle.sampling_rate)
                row.update(dur_inj_sec=round(d, 3), clip_inj=round(c, 6),
                           n_tok_inj=inj["n_speech_tokens"],
                           hit_max_inj=inj["hit_max_new_tokens"],
                           layer=inj["layer"], hidden_index=inj["hidden_index"],
                           splice_mode=inj["splice_mode"],
                           m_src_entity_tokens=inj["m_src_entity_tokens"],
                           n_tgt_entity_tokens=inj["n_tgt_entity_tokens"],
                           equal_len=inj["equal_len"],
                           fired=inj["fired"], decode_calls=inj["decode_calls"],
                           skipped_shapes=inj["skipped_shapes"],
                           kappa_by_layer=inj.get("kappa_by_layer"),
                           kappa_at_splice=inj.get("kappa_at_splice"),
                           kappa_last_layer=inj.get("kappa_last_layer"))
                if inj["fired"] != 1:
                    raise RuntimeError(f"splice hook fired={inj['fired']} (1 であるべき)")
                if inj.get("span_extension") != span_ext:
                    raise RuntimeError(
                        "span extension が context 側と生成側で食い違った: "
                        f"{span_ext} vs {inj.get('span_extension')}")
            row["elapsed_sec"] = round(time.time() - tp, 2)
            rows[str(pidx)] = row
            n_done += 1
        except Exception as e:
            import traceback
            n_err += 1
            rows[str(pidx)] = {"pair_idx": pidx, "status": "error",
                               "word_a": pair.get("word_a"), "word_b": pair.get("word_b"),
                               "err": f"{type(e).__name__}: {e}"[:500]}
            print(f"[err] pair {pidx}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if args.stop_on_error:
                raise
        atomic_json_dump({"rows": [rows[k2] for k2 in sorted(rows, key=lambda x: int(x))]},
                         progress_path)
        if (k + 1) % args.log_every == 0 or k + 1 == len(todo):
            el = time.time() - t0
            rate = el / max(1, n_done) if n_done else float("nan")
            left = (len(todo) - (k + 1)) * rate
            print(f"  [{k + 1}/{len(todo)}] done={n_done} skip={n_skip} err={n_err} "
                  f"{el / 60:.1f}min elapsed, ~{left / 60:.1f}min left "
                  f"({rate:.1f}s/pair)", flush=True)

    all_rows = [rows[k2] for k2 in sorted(rows, key=lambda x: int(x))]
    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    manifest = {
        "meta": {
            "model_id": args.model_id, "model_dir": bundle.model_dir,
            "codec_id": args.codec_id, "codec_dir": bundle.codec_dir,
            "dataset": "cross_culture", "lang": args.lang, "n_target": len(pairs),
            "layer": args.layer, "hidden_index": L.hidden_index_for_layer(args.layer),
            "num_layers": bundle.num_layers, "hidden_size": bundle.hidden_size,
            "seed": args.seed, "decode": gen.as_dict(),
            "span_extension": bool(args.span_extension),
            "ref_mode": args.ref_mode,
            "ref": ({"wav": ref["wav_path"], "text": ref["ref_text"],
                     "n_tokens": ref["n_tokens"], "dur_sec": ref["dur_sec"],
                     "codec_roundtrip": ref.get("roundtrip")} if ref else None),
            "dtype": args.dtype, "device": args.device,
            "attn_implementation": args.attn_implementation,
            "sampling_rate": bundle.sampling_rate,
            "wav_dir": args.wav_dir, "condition_tag": condition_tag(args),
            "splice": ("replace the rows of the entity-word tokens in the output of "
                       "model.model.layers[layer] during PREFILL ONLY, with the same "
                       "layer's hidden states taken from the target-context prompt. "
                       "Sequence length is never changed (m != n is handled by mean-"
                       "pooling the target span and broadcasting over the m source "
                       "rows), so RoPE positions and the KV cache layout stay intact. "
                       "No weight update; inference only."),
            "model_info": bundle.info,
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "n_generated": n_done, "n_skipped": n_skip, "n_error": n_err,
            "wallclock_sec": round(time.time() - t0, 1),
        },
        "kappa": summarize_kappa(ok_rows, bundle.num_layers),
        "gen_length": summarize_gen_length(ok_rows, args.max_new_tokens),
        "rows": all_rows,
    }
    man_path = os.path.join(args.out_dir, f"llasa_{condition_tag(args)}_manifest.json")
    atomic_json_dump(manifest, man_path)
    print(f"\n[manifest] -> {man_path}")
    print(f"[manifest] generated={n_done} skipped={n_skip} error={n_err}")
    ks = manifest["kappa"].get("at_splice_layer")
    if ks:
        print(f"[kappa] layer {args.layer} (hidden index {L.hidden_index_for_layer(args.layer)}) "
              f"mean = {ks['mean']:.4f} (n={ks['n']})")
    kl = manifest["kappa"].get("last_layer")
    if kl:
        print(f"[kappa] last layer mean = {kl['mean']:.4f}")
    report_gen_length(manifest["gen_length"])
    if own_bundle:
        del bundle
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return man_path


# --------------------------------------------------------------------------
# scorers (既存ファイルをサブプロセスでそのまま呼ぶ)
# --------------------------------------------------------------------------
def run_scorers(args):
    env = dict(os.environ)
    env["HF_HOME"] = os.environ["HF_HOME"]
    env["HF_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    env["REPO_ROOT"] = PROJECT_ROOT
    tag = condition_tag(args)
    jobs = []
    mcd_json = args.mcd_json or os.path.join(
        PROJECT_ROOT, "eval_results", f"_caseSPTK_whole_llasa_{tag}.json")
    jobs.append(("MCD", [sys.executable, MCD_SCORER,
                         "--wav_dir", args.wav_dir, "--lang", args.lang,
                         "--mode", "whole", "--n", str(args.n),
                         "--output_json", mcd_json]))
    pib_json = args.pib_json or os.path.join(
        PROJECT_ROOT, "eval_results", f"_caseSPTK_phperPER_llasa_{tag}.json")
    jobs.append(("PIB", [sys.executable, PIB_SCORER,
                         "--wav_dir", args.wav_dir, "--lang", args.lang,
                         "--n", str(args.n), "--output_json", pib_json]))
    for name, cmd in jobs:
        print(f"\n[scorer:{name}] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        print(f"[scorer:{name}] exit={r.returncode}", flush=True)
        if r.returncode != 0:
            print(f"[scorer:{name}] FAILED", flush=True)
    return {"mcd_json": mcd_json, "pib_json": pib_json}


# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="en", choices=["en", "ja", "zh"],
                   help="Llasa-3B は英語と中国語のみ (日本語は非対応)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--layer", type=int, default=26,
                   help="model.model.layers[l] の出力を差し替える (Qwen3-TTS の最良層と同じ 26 が既定)")
    p.add_argument("--wav_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_id", default=L.DEFAULT_MODEL_ID)
    p.add_argument("--codec_id", default=L.DEFAULT_CODEC_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--allow_download", action="store_true",
                   help="ローカル cache に無いとき HF から取得することを明示的に許可する")
    p.add_argument("--seed", type=int, default=L.SEED)
    p.add_argument("--hf_home", default=None)
    # 復号 (BL / TGT / INJ で共有)
    p.add_argument("--greedy", dest="do_sample", action="store_false",
                   help="do_sample=False にする (既定は公式と同じ sampling)")
    p.set_defaults(do_sample=True)
    p.add_argument("--temperature", type=float, default=L.DEFAULT_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=L.DEFAULT_TOP_P)
    p.add_argument("--top_k", type=int, default=L.DEFAULT_TOP_K)
    p.add_argument("--max_new_tokens", type=int, default=L.DEFAULT_MAX_NEW_TOKENS)
    # 参照音声 (voice clone)
    p.add_argument("--ref_mode", default="clone", choices=["clone", "none"],
                   help="clone: Table-3 と同じ自然音声を参照にする (既定)")
    p.add_argument("--ref_wav", default=None)
    p.add_argument("--ref_text", default=None)
    p.add_argument("--ref_max_sec", type=float, default=15.0)
    p.add_argument("--no_codec_check", action="store_true",
                   help="参照音声の encode -> decode 往復チェックを省く (既定は実施)")
    p.add_argument("--span_extension", action="store_true",
                   help="語そのままで往復一致しない対 (中国語の「的」など) に限り "
                        "最小対称拡張 (scripts/projection/span_extend.py) を試す。"
                        "既定 off = 従来どおり ValueError で落とす。"
                        "往復一致する対の結果は on/off で 1 ビットも変わらない")
    # control
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--no_kappa", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--link_bl_tgt_from", default=None,
                   help="別の層で生成済みの BL / TGT を hard link で持ち込む (層掃引用)")
    # scoring
    p.add_argument("--run_scorers", action="store_true")
    p.add_argument("--mcd_json", default=None)
    p.add_argument("--pib_json", default=None)
    return p


def resolve_paths(args):
    tag = condition_tag(args)
    if args.wav_dir is None:
        args.wav_dir = os.path.join(PROJECT_ROOT, "eval_results", "_cc_wavs_a", f"llasa_{tag}")
    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, "eval_results", f"llasa_cross_culture_{tag}")
    if not os.path.isabs(args.wav_dir):
        args.wav_dir = os.path.join(PROJECT_ROOT, args.wav_dir)
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    if args.link_bl_tgt_from and not os.path.isabs(args.link_bl_tgt_from):
        args.link_bl_tgt_from = os.path.join(PROJECT_ROOT, args.link_bl_tgt_from)
    return args


def main():
    args = build_argparser().parse_args()
    if args.hf_home:
        L.setup_hf_home(args.hf_home)
    args = resolve_paths(args)
    os.chdir(PROJECT_ROOT)

    pairs = load_pairs(args.lang, args.n)
    print(f"[data] cross_culture lang={args.lang} n={len(pairs)} "
          f"first={pairs[0]['word_a']}->{pairs[0]['word_b']} "
          f"last={pairs[-1]['word_a']}->{pairs[-1]['word_b']}", flush=True)
    print(f"[data] condition={condition_tag(args)}", flush=True)
    print(f"[data] wav_dir={args.wav_dir}", flush=True)

    if not args.skip_generation:
        run_generation(args, pairs)
    if args.run_scorers:
        run_scorers(args)


if __name__ == "__main__":
    main()
