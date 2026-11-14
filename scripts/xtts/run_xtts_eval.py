#!/usr/bin/env python3
"""XTTS-v2 に SpeechSplice を適用して EN cross-culture 300 ペアを生成・評価する (GPU).

生成するもの (既存 4 モデル = T5Gemma-TTS / SpeechT5 / LongCat / Llasa-3B /
Qwen3-TTS と同じ命名規約):
    {wav_dir}/s{pair_idx:04d}_bl.wav  = TTS(text_a)
    {wav_dir}/s{pair_idx:04d}_inj.wav = SpeechSplice(text_a)
    {wav_dir}/s{pair_idx:04d}_tgt.wav = TTS(text_b)
3 本そろっていないペアはスコアラ側で available:false として捨てられるので
TGT の生成を省いてはならない
(mcd_pysptk.py:228-229 / phoneme_per.py:166-167)。

指標は既存ファイルをそのまま使う (1 行も改変していない):
    MCD : speechsplice_release/src/mcd_pysptk.py --mode whole
    PIB : speechsplice_release/src/phoneme_per.py   (inj_per < bl_per の厳密不等号)
--run_scorers を付けると生成後にサブプロセスで呼ぶ。後から単体で回してもよい。

統制:
  * seed 42 を BL / TGT / INJ の generate 直前に必ずリセットする。
  * 3 条件で復号設定 (do_sample / temperature / top_p / top_k /
    repetition_penalty / length_penalty) と prompt を完全に共有する。
    BL と INJ は text token が 1 個も違わない。差は splice だけ。
  * 参照音声は Table-3 の他アーキテクチャと同じ自然音声
    (eval_results/_natural_refs_norm/en_natural_norm.wav, 3.25 s)。
    cond latents と speaker embedding は評価開始時に 1 回だけ作り、
    3 条件で bitwise 共有する (毎回 assert_reference_unchanged で照合)。
  * XTTS-v2 は日本語を cutlet で romaji 化、中国語を pypinyin で拼音化する。
    読みを直接指定する入口になるので本手法の前提が壊れる。**英語のみ**。

実行例 (GPU):
  SLURM_TIME=04:00:00 slurm_run.sh \\
    python3 scripts/xtts/run_xtts_eval.py --lang en --n 300 --layer 28 \\
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
# スコアラ (mcd_pysptk.py / phoneme_per.py) が読むのと同じデータを先に見る。
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "xtts"))

import xtts_splice as X  # noqa: E402  (HF_HOME はこの import で設定される)

import numpy as np  # noqa: E402
import torch  # noqa: E402

MCD_SCORER = os.path.join(PROJECT_ROOT, "src", "mcd_pysptk.py")
PIB_SCORER = os.path.join(PROJECT_ROOT, "src", "phoneme_per.py")

DEFAULT_REF_WAV = os.path.join(PROJECT_ROOT, "eval_results", "_natural_refs_norm",
                               "en_natural_norm.wav")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_pairs(lang, n):
    """Table 3 (cross-architecture) で既存モデルが使っているデータそのもの."""
    if lang != "en":
        raise SystemExit(
            f"--lang {lang} は対象外。XTTS-v2 は日本語を romaji 化、中国語を拼音化するので "
            f"読みを直接与える入口になり、本手法の前提が壊れる。英語のみ。")
    from en_cross_culture_names import build_300_pairs
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
    if not args.splice_latent_pass:
        tag += "_arONLY"
    if args.seed != X.SEED:
        tag += f"_seed{args.seed}"
    return tag


def link_or_copy(src, dst):
    """BL / TGT は層に依存しないので、層掃引では 1 層目の結果を持ち込める."""
    if os.path.exists(dst) or not os.path.exists(src):
        return False
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return True


# --------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------
def summarize_kappa(rows, num_layers):
    """kappa の層ごと集計 (entity ごとの値を層ごとに単純平均).

    index 0 = 埋め込み出力、index l+1 = layers[l] の出力。splice の対象は
    layers[layer] の出力なので kappa_at_splice = by_layer[layer+1]。
    母数・題材が既存 Table 3 の kappa セルと揃っていないので直接比較はできない。
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
                 "hidden index 0 = embedding output, index l+1 = output of gpt.gpt.h[l] "
                 f"(num_layers={num_layers}); the last index is post-ln_f. The spliced "
                 "tensor is the output of gpt.gpt.h[layer], i.e. hidden index layer+1. "
                 "Corpus/N differ from the existing Table 3 kappa cells; not directly "
                 "comparable."),
    }


def summarize_span(rows):
    """スパンがトークン境界に乗ったか / 長さが一致したかの内訳."""
    out = {"n": 0, "src_aligned": 0, "tgt_aligned": 0, "src_misaligned_pairs": [],
           "tgt_misaligned_pairs": [], "equal_len": 0, "how": {}, "m": {}, "n_tgt": {}}
    for r in rows:
        if r.get("s_span") is None:
            continue
        out["n"] += 1
        if r.get("s_aligned"):
            out["src_aligned"] += 1
        else:
            out["src_misaligned_pairs"].append(r["pair_idx"])
        if r.get("t_aligned"):
            out["tgt_aligned"] += 1
        else:
            out["tgt_misaligned_pairs"].append(r["pair_idx"])
        if r.get("equal_len"):
            out["equal_len"] += 1
        for k in ("s_how", "t_how"):
            v = r.get(k)
            if v:
                out["how"][v] = out["how"].get(v, 0) + 1
        if r.get("m_src_entity_tokens") is not None:
            out["m"][str(r["m_src_entity_tokens"])] = out["m"].get(
                str(r["m_src_entity_tokens"]), 0) + 1
        if r.get("n_tgt_entity_tokens") is not None:
            out["n_tgt"][str(r["n_tgt_entity_tokens"])] = out["n_tgt"].get(
                str(r["n_tgt_entity_tokens"]), 0) + 1
    out["src_misaligned"] = out["n"] - out["src_aligned"]
    out["tgt_misaligned"] = out["n"] - out["tgt_aligned"]
    out["src_misaligned_pairs"] = out["src_misaligned_pairs"][:32]
    out["tgt_misaligned_pairs"] = out["tgt_misaligned_pairs"][:32]
    return out


def summarize_gen_length(rows, max_gen_mel_tokens, frame_rate=None):
    """条件ごとの生成長 (mel code 数) と上限張り付き率."""
    out = {"max_gen_mel_tokens": int(max_gen_mel_tokens)}
    for cond in ("bl", "tgt", "inj"):
        vals = [r[f"n_codes_{cond}"] for r in rows if r.get(f"n_codes_{cond}") is not None]
        hits = [bool(r.get(f"hit_max_{cond}")) for r in rows
                if r.get(f"n_codes_{cond}") is not None]
        durs = [r[f"dur_{cond}_sec"] for r in rows if r.get(f"dur_{cond}_sec") is not None]
        if not vals:
            out[cond] = None
            continue
        arr = np.array(vals, dtype=np.float64)
        out[cond] = {
            "n": int(arr.size),
            "mean_codes": round(float(arr.mean()), 1),
            "median_codes": round(float(np.median(arr)), 1),
            "p95_codes": round(float(np.percentile(arr, 95)), 1),
            "max_codes": int(arr.max()),
            "mean_sec": (round(float(np.mean(durs)), 3) if durs else None),
            "n_hit_max": int(sum(hits)),
            "hit_max_rate": round(sum(hits) / float(len(hits)), 4),
        }
    return out


def report_gen_length(summary, warn_rate=0.2):
    cap = summary["max_gen_mel_tokens"]
    print(f"[genlen] max_gen_mel_tokens={cap}")
    for cond in ("bl", "tgt", "inj"):
        s = summary.get(cond)
        if not s:
            continue
        print(f"[genlen] {cond.upper():3s} n={s['n']} mean={s['mean_codes']} codes "
              f"({s['mean_sec']}s) median={s['median_codes']} p95={s['p95_codes']} "
              f"hit_max={s['n_hit_max']}/{s['n']} ({s['hit_max_rate'] * 100:.1f}%)")
        if s["hit_max_rate"] > warn_rate:
            print(f"[genlen][warn] {cond.upper()} の {s['hit_max_rate'] * 100:.1f}% が "
                  f"上限 {cap} に張り付いている。stop_audio_token が出ていない "
                  f"(参照音声 / 復号設定を疑うこと)。")


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
def build_ref(bundle, args):
    wav = args.ref_wav or DEFAULT_REF_WAV
    ref = X.load_reference(bundle, wav, gpt_cond_len=args.gpt_cond_len,
                           gpt_cond_chunk_len=args.gpt_cond_chunk_len,
                           max_ref_length=args.max_ref_len,
                           sound_norm_refs=args.sound_norm_refs)
    print(f"[ref] {wav} -> cond latents {list(ref['gpt_cond_latent'].shape)} "
          f"speaker emb {list(ref['speaker_embedding'].shape)}", flush=True)
    print(f"[ref] cond は Perceiver (参照音声の mel のみ) の出力。テキストは通らない。"
          f" checksum={ref['cond_checksum']}", flush=True)
    return ref


def load_bundle(args):
    """層掃引では 1 回だけ読んで使い回す."""
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[load] {args.model_id} dtype={args.dtype} device={args.device} "
          f"HF_HOME={os.environ['HF_HOME']} allow_download={args.allow_download}", flush=True)
    t0 = time.time()
    bundle = X.load_xtts(model_id=args.model_id, device=args.device, dtype=dtype,
                         allow_download=args.allow_download)
    print(f"[load] done in {time.time() - t0:.1f}s  model_dir={bundle.model_dir}", flush=True)
    print(f"[load] layers={bundle.num_layers} hidden={bundle.hidden_size} "
          f"cond_len={bundle.cond_len} text_vocab={bundle.info['number_text_tokens']} "
          f"sr={bundle.sampling_rate}", flush=True)
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
    X.assert_splice_target(bundle, args.layer)

    gen = X.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                      repetition_penalty=args.repetition_penalty,
                      length_penalty=args.length_penalty, num_beams=args.num_beams)
    print(f"[gen] layer={args.layer} splice_latent_pass={args.splice_latent_pass} "
          f"{gen.as_dict()}", flush=True)

    ref = build_ref(bundle, args)

    # 行番号のずれをここで 1 回だけ機械的に潰す。自前の prefill 再現が
    # モデル自身の store_prefix_emb と一致しなければ先へ進まない。
    probe = X.prepare_text(bundle, pairs[0]["text_a"], args.lang,
                           word=pairs[0]["word_a"], strict_span=True)
    layout = X.verify_prefill_layout(bundle, ref["gpt_cond_latent"], probe["text_tokens"])
    print(f"[layout] {layout}", flush=True)
    print(f"[layout] probe span rows={probe['span']['row_start']}..{probe['span']['row_end']} "
          f"tokens={probe['span']['entity_tokens']} aligned={probe['span']['aligned']} "
          f"how={probe['span']['how']}", flush=True)

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
            # prompt / span は 1 ペアにつき 1 回だけ作り、BL と INJ で同じ
            # text_tokens を使い回す (差が splice だけであることの保証)。
            src = X.prepare_text(bundle, pair["text_a"], args.lang,
                                 word=pair["word_a"], strict_span=True)
            tgt = X.prepare_text(bundle, pair["text_b"], args.lang,
                                 word=pair["word_b"], strict_span=True)

            row = {"pair_idx": pidx, "status": "ok",
                   "word_a": pair["word_a"], "word_b": pair["word_b"],
                   "text_a": pair["text_a"], "text_b": pair["text_b"],
                   "n_text_tokens_a": src["n_text_tokens"],
                   "n_text_tokens_b": tgt["n_text_tokens"],
                   "s_span": [src["span"]["tok_start"], src["span"]["tok_end"]],
                   "t_span": [tgt["span"]["tok_start"], tgt["span"]["tok_end"]],
                   "s_rows": [src["span"]["row_start"], src["span"]["row_end"]],
                   "t_rows": [tgt["span"]["row_start"], tgt["span"]["row_end"]],
                   "s_tokens": src["span"]["entity_tokens"],
                   "t_tokens": tgt["span"]["entity_tokens"],
                   "s_aligned": src["span"]["aligned"], "t_aligned": tgt["span"]["aligned"],
                   "s_how": src["span"]["how"], "t_how": tgt["span"]["how"]}

            if need_bl:
                r = X.synthesize(bundle, pair["text_a"], ref, args.lang, gen=gen, ctx=src)
                _, d, c = save_wav(bl_p, r["audio"], bundle.sampling_rate)
                row.update(dur_bl_sec=round(d, 3), clip_bl=round(c, 6),
                           n_codes_bl=r["n_mel_codes"], hit_max_bl=r["hit_max_mel"])
            if need_tgt:
                r = X.synthesize(bundle, pair["text_b"], ref, args.lang, gen=gen, ctx=tgt)
                _, d, c = save_wav(tgt_p, r["audio"], bundle.sampling_rate)
                row.update(dur_tgt_sec=round(d, 3), clip_tgt=round(c, 6),
                           n_codes_tgt=r["n_mel_codes"], hit_max_tgt=r["hit_max_mel"])
            if need_inj:
                inj = X.splice_synthesize(
                    bundle, pair["text_a"], pair["word_a"],
                    pair["text_b"], pair["word_b"], layer=args.layer, ref=ref,
                    lang=args.lang, gen=gen, collect_kappa=not args.no_kappa,
                    strict_span=True, splice_latent_pass=args.splice_latent_pass,
                    src_ctx=src, tgt_ctx=tgt)
                _, d, c = save_wav(inj_p, inj["audio"], bundle.sampling_rate)
                row.update(dur_inj_sec=round(d, 3), clip_inj=round(c, 6),
                           n_codes_inj=inj["n_mel_codes"], hit_max_inj=inj["hit_max_mel"],
                           layer=inj["layer"], hidden_index=inj["hidden_index"],
                           splice_mode=inj["splice_mode"],
                           splice_latent_pass=inj["splice_latent_pass"],
                           m_src_entity_tokens=inj["m_src_entity_tokens"],
                           n_tgt_entity_tokens=inj["n_tgt_entity_tokens"],
                           equal_len=inj["equal_len"],
                           fired_prefill=inj["fired_prefill"],
                           fired_latent_pass=inj["fired_latent_pass"],
                           decode_calls=inj["decode_calls"],
                           skipped_shapes=inj["skipped_shapes"],
                           kappa_by_layer=inj.get("kappa_by_layer"),
                           kappa_at_splice=inj.get("kappa_at_splice"),
                           kappa_last_layer=inj.get("kappa_last_layer"))
                want2 = 1 if args.splice_latent_pass else 0
                if inj["fired_prefill"] != 1 or inj["fired_latent_pass"] != want2:
                    raise RuntimeError(
                        f"splice hook の発火回数が想定外 "
                        f"(prefill={inj['fired_prefill']} latent={inj['fired_latent_pass']})")
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
            "license": "Coqui Public Model License (non-commercial); HF license: other",
            "dataset": "cross_culture", "lang": args.lang, "n_target": len(pairs),
            "layer": args.layer, "hidden_index": X.hidden_index_for_layer(args.layer),
            "num_layers": bundle.num_layers, "hidden_size": bundle.hidden_size,
            "cond_len": bundle.cond_len, "row_offset": X.gpt_row_offset(bundle.cond_len),
            "seed": args.seed, "decode": gen.as_dict(),
            "splice_latent_pass": bool(args.splice_latent_pass),
            "ref": {"wav": ref["wav_path"], "cond_len": ref["cond_len"],
                    "gpt_cond_len": ref["gpt_cond_len"],
                    "gpt_cond_chunk_len": ref["gpt_cond_chunk_len"],
                    "max_ref_length": ref["max_ref_length"],
                    "sound_norm_refs": ref["sound_norm_refs"],
                    "checksum": ref["cond_checksum"]},
            "dtype": args.dtype, "device": args.device,
            "sampling_rate": bundle.sampling_rate,
            "wav_dir": args.wav_dir, "condition_tag": condition_tag(args),
            "splice": (
                "Replace the rows of the entity-word tokens in the output of "
                "gpt.gpt.h[layer] (a GPT-2 block of the XTTS-v2 trunk) with the same "
                "layer's hidden states taken from the target-context prompt. XTTS-v2 "
                "runs the trunk twice per utterance: (1) the AR prefill inside "
                "gpt.generate(), whose prefix embedding is built by "
                "GPT.compute_embeddings/store_prefix_emb, and (2) the teacher-forced "
                "pass gpt(..., return_latent=True) that produces the latents fed to "
                "HiFiGAN. Both carry the same [cond x32][START][text][STOP] layout, so "
                "the splice is applied to both (disable with --no_splice_latent_pass). "
                "Causal masking makes the text rows identical in the two passes, so a "
                "single target vector taken from the pass-1 prefill serves both. The "
                "sequence length is never changed (m != n is handled by mean-pooling "
                "the target span and broadcasting over the m source rows), so the "
                "learned text positions and the KV cache layout stay intact. Text is "
                "prefix-concatenated; there is no text cross-attention. The Perceiver "
                "resampler does have cross-attention, but only over the reference-audio "
                "latents, never over text; it is computed once and shared bitwise by "
                "BL/TGT/INJ. No weight update; inference only."),
            "prefill_layout_check": layout,
            "model_info": bundle.info,
            "tokenizer": {"text_vocab": bundle.info["number_text_tokens"],
                          "start_text_token": bundle.start_text_token,
                          "stop_text_token": bundle.stop_text_token,
                          "start_audio_token": bundle.start_audio_token,
                          "stop_audio_token": bundle.stop_audio_token},
            "span_rule": ("scripts/encoder_injection_t5gemma.py "
                          "find_entity_token_range (3-stage). Token boundaries are NOT "
                          "verified and the window is NOT widened; whether the span "
                          "landed on a token boundary is only recorded as `aligned`."),
            "coqui_tts": _pkg_version("coqui-tts"),
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "n_generated": n_done, "n_skipped": n_skip, "n_error": n_err,
            "wallclock_sec": round(time.time() - t0, 1),
        },
        "span": summarize_span(ok_rows),
        "kappa": summarize_kappa(ok_rows, bundle.num_layers),
        "gen_length": summarize_gen_length(ok_rows, bundle.model.gpt.max_gen_mel_tokens),
        "rows": all_rows,
    }
    man_path = os.path.join(args.out_dir, f"xtts_{condition_tag(args)}_manifest.json")
    atomic_json_dump(manifest, man_path)
    print(f"\n[manifest] -> {man_path}")
    print(f"[manifest] generated={n_done} skipped={n_skip} error={n_err}")
    sp = manifest["span"]
    print(f"[span] n={sp['n']} src_aligned={sp['src_aligned']} "
          f"(misaligned {sp['src_misaligned']}) tgt_aligned={sp['tgt_aligned']} "
          f"(misaligned {sp['tgt_misaligned']}) equal_len={sp['equal_len']} how={sp['how']}")
    ks = manifest["kappa"].get("at_splice_layer")
    if ks:
        print(f"[kappa] layer {args.layer} (hidden index "
              f"{X.hidden_index_for_layer(args.layer)}) mean = {ks['mean']:.4f} (n={ks['n']})")
    kl = manifest["kappa"].get("last_layer")
    if kl:
        print(f"[kappa] last layer mean = {kl['mean']:.4f}")
    report_gen_length(manifest["gen_length"])
    if own_bundle:
        del bundle
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return man_path


def _pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


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
        PROJECT_ROOT, "eval_results", f"_caseSPTK_whole_xtts_{tag}.json")
    jobs.append(("MCD", [sys.executable, MCD_SCORER,
                         "--wav_dir", args.wav_dir, "--lang", args.lang,
                         "--mode", "whole", "--n", str(args.n),
                         "--output_json", mcd_json]))
    pib_json = args.pib_json or os.path.join(
        PROJECT_ROOT, "eval_results", f"_caseSPTK_phperPER_xtts_{tag}.json")
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
    p.add_argument("--lang", default="en", choices=["en"],
                   help="英語のみ (JA は romaji 化、ZH は拼音化されるので前提が壊れる)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--layer", type=int, default=28,
                   help="gpt.gpt.h[l] の出力を差し替える (既定は 30 層中 L-2 = 28)")
    p.add_argument("--wav_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_id", default=X.DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"],
                   help="XTTS-v2 は float32 が既定 (HiFiGAN を含むので半精度は未検証)")
    p.add_argument("--allow_download", action="store_true",
                   help="ローカル cache に無いとき HF から取得する (約 1.75 GB)")
    p.add_argument("--seed", type=int, default=X.SEED)
    p.add_argument("--hf_home", default=None)
    # 復号 (BL / TGT / INJ で共有)
    p.add_argument("--greedy", dest="do_sample", action="store_false",
                   help="do_sample=False (既定は config.json と同じ sampling)")
    p.set_defaults(do_sample=True)
    p.add_argument("--temperature", type=float, default=X.DEFAULT_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=X.DEFAULT_TOP_P)
    p.add_argument("--top_k", type=int, default=X.DEFAULT_TOP_K)
    p.add_argument("--repetition_penalty", type=float, default=X.DEFAULT_REPETITION_PENALTY)
    p.add_argument("--length_penalty", type=float, default=X.DEFAULT_LENGTH_PENALTY)
    p.add_argument("--num_beams", type=int, default=1)
    # splice
    p.add_argument("--no_splice_latent_pass", dest="splice_latent_pass",
                   action="store_false",
                   help="pass 2 (return_latent=True の forward) には splice しない。"
                        "既定は 2 つの pass 両方に掛ける")
    p.set_defaults(splice_latent_pass=True)
    # 参照音声
    p.add_argument("--ref_wav", default=None, help=f"既定 {DEFAULT_REF_WAV}")
    p.add_argument("--gpt_cond_len", type=int, default=X.DEFAULT_GPT_COND_LEN)
    p.add_argument("--gpt_cond_chunk_len", type=int, default=X.DEFAULT_GPT_COND_CHUNK_LEN)
    p.add_argument("--max_ref_len", type=int, default=X.DEFAULT_MAX_REF_LEN)
    p.add_argument("--sound_norm_refs", action="store_true",
                   default=X.DEFAULT_SOUND_NORM_REFS)
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
        args.wav_dir = os.path.join(PROJECT_ROOT, "eval_results", "_cc_wavs_a", f"xtts_{tag}")
    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, "eval_results", f"xtts_cross_culture_{tag}")
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
        X.setup_hf_home(args.hf_home)
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
