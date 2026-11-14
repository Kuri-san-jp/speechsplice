#!/usr/bin/env python3
"""Kiseki-TTS に SpeechSplice を適用して日本語 300 ペアを生成する（GPU）.

生成するもの（既存 3 モデルと同じ命名規約。3 本そろっていないペアはスコアラ側で
available:false として捨てられるので TGT の生成を省略してはならない。
mcd_pysptk.py:228-229, phoneme_per.py:166-167）:

    {wav_dir}/s{pair_idx:04d}_bl.wav    = TTS(text_a)
    {wav_dir}/s{pair_idx:04d}_inj.wav   = SpeechSplice(text_a の encoder 出力に
                                          text_b の実体語スパンを移植)
    {wav_dir}/s{pair_idx:04d}_tgt.wav   = TTS(text_b)

評価データは speechsplice_release/data/ja_cross_culture_names.py の
build_300_pairs()。**Table 3 の JA 行が使っているのと同じデータ** で、
mcd_pysptk.py / phoneme_per.py が内部で同じものを読むので無改造で流用できる。

BL / TGT / INJ は復号設定（seed / max_new_frames / temperature / top_k / top_p /
repetition_penalty）を完全に共有する。差は splice だけ。

指標の計算は既存ファイルをそのまま使う（1 行も改変していない）:
    MCD : speechsplice_release/src/mcd_pysptk.py --mode whole
    PIB : speechsplice_release/src/phoneme_per.py --lang ja

**スコアラは既存イメージ（my-lab-image:latest）で回すこと。** Kiseki の生成側も
同じイメージで動く（transformers 4.57.3 / torch 2.6.0。qwen-tts のピンと一致）ので、
Irodori と違って別イメージを作る必要がない。足りない依存（qwen_tts / sox）は
scripts/kiseki/_vendor に --target で入れてあり、site-packages には触れていない。

実行例（GPU）:
  SLURM_TIME=04:00:00 slurm_run.sh \\
    python3 scripts/kiseki/run_kiseki_eval.py --n 300
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

# --------------------------------------------------------------------------
# HF cache: slurm_run.sh は -e HF_HOME=/tmp/.cache/huggingface を渡すが、その実体は
# ${HOME}/.cache/huggingface（空・root 所有）なので書けない。実際にモデルが置いて
# あるのは <PROJECT_ROOT>/.cache/huggingface（コンテナ内 /mnt/.cache/huggingface）。
# setdefault ではコンテナの環境変数に負けるので、ここで **上書き** する。
# scripts/irodori/run_irodori_eval.py:44-52 と同じ扱い。
# transformers を import する前に済ませること。
# --------------------------------------------------------------------------
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".cache", "huggingface")


def setup_hf_home(hf_home):
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_home, "modules")
    return hf_home


setup_hf_home(os.environ.get("KISEKI_HF_HOME", DEFAULT_HF_HOME))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "kiseki"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kiseki_splice import (  # noqa: E402
    DEFAULT_CODEC_ID,
    DEFAULT_LANG_TOKEN,
    DEFAULT_MAX_NEW_FRAMES,
    DEFAULT_MODEL_ID,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    SEED,
    load_kiseki,
    splice_synthesize,
    synthesize,
)

MCD_SCORER = os.path.join(PROJECT_ROOT, "src", "mcd_pysptk.py")
PIB_SCORER = os.path.join(PROJECT_ROOT, "src", "phoneme_per.py")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_pairs(lang, n):
    """Table 3 の JA 行と同じ 300 ペア（pair_idx が最初から入っている）。"""
    if lang != "ja":
        raise SystemExit(
            f"Kiseki-TTS は日本語専用（model card: Japanese only、"
            f"<|2ja|> が唯一の言語タグ）。--lang {lang} は対象外。")
    from ja_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()[:n]
    for i, p in enumerate(pairs):
        p.setdefault("pair_idx", i)
    return pairs


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def save_wav(path, audio, sr):
    """PCM_16 で保存。途中で落ちても壊れた wav が残らないよう tmp -> rename。

    戻り値: (n_samples, duration_sec, clipped_ratio)
    """
    import soundfile as sf

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        raise RuntimeError(f"empty audio for {path}")
    n_clip = int(np.sum(np.abs(a) > 1.0))
    a = np.clip(a, -1.0, 1.0)
    tmp = path + ".tmp"   # 拡張子が .wav でないので format を明示する
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


# --------------------------------------------------------------------------
# kappa aggregation
# --------------------------------------------------------------------------
def summarize_kappa(rows):
    """kappa の層ごと集計（entity ごとの値を層ごとに単純平均）。

    kappa_l(e) = cos( mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l} )
    層 index 0 = encoder の embedding 出力、1..12 が EncoderLayer 12 層の出力。
    これとは別に kappa_encoder_out = 最終 RMSNorm を通った
    **decoder が実際に cross-attend するテンソル** 上の値も出す
    （= splice する対象そのもの）。

    **既存 Table 3 の kappa 列とは母数も題材も揃っていない。**
    T5Gemma の値は別単語ペア N=1000、SpeechT5 / Qwen3 は heteronym N=8 で、
    aggregate_table3.py:29-33 は kappa を定数として印字するだけで再計算していない。
    ここで出るのは「この cross-culture 300 ペアでの Kiseki の値」。
    どの流儀に合わせるかは要判断（README の未解決点を参照）。
    """
    per_layer = {}
    last = []
    for r in rows:
        if r.get("kappa_encoder_out") is not None:
            last.append(r["kappa_encoder_out"])
        kl = r.get("kappa_by_layer")
        if not kl:
            continue
        for li, v in enumerate(kl):
            per_layer.setdefault(li, []).append(v)

    def _stat(vals):
        arr = np.array(vals, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)), 6) if arr.size > 1 else None,
            "median": round(float(np.median(arr)), 6),
        }

    summary = {"by_layer_encoder": {str(li): _stat(v)
                                    for li, v in sorted(per_layer.items())}}
    if last:
        summary["encoder_out"] = _stat(last)
    summary["note"] = (
        "kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}). "
        "by_layer_encoder: index 0 = encoder embedding output, 1..12 = the 12 "
        "EncoderLayer outputs. encoder_out: after the final RMSNorm, i.e. the "
        "tensor the Mamba2 decoder cross-attends to and the one SpeechSplice "
        "replaces. Corpus/N differ from the existing Table 3 kappa cells; "
        "not directly comparable.")
    return summary


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def run_generation(args, pairs):
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

    print(f"[load] {args.model_id} precision={args.precision} device={args.device} "
          f"HF_HOME={os.environ['HF_HOME']}", flush=True)
    t_load = time.time()
    bundle = load_kiseki(
        model_id=args.model_id,
        device=args.device,
        precision=args.precision,
        lang_token=args.lang_token,
        codec_id=args.codec_id,
        codec_device=args.codec_device or args.device,
    )
    print(f"[load] done in {time.time() - t_load:.1f}s "
          f"sr={bundle.codec_sr} d_model={bundle.meta['d_model']} "
          f"enc_layers={bundle.meta['enc_layers']} num_q={bundle.meta['num_q']}",
          flush=True)

    todo = pairs
    if args.num_shards > 1:
        todo = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_index]
        print(f"[shard] {args.shard_index}/{args.num_shards}: {len(todo)} of {len(pairs)} pairs",
              flush=True)

    kw = dict(max_new_frames=args.max_new_frames, temperature=args.temperature,
              top_k=args.top_k, top_p=args.top_p,
              repetition_penalty=args.repetition_penalty)

    t0 = time.time()
    n_done = n_skip = n_err = 0
    for k, pair in enumerate(todo):
        pidx = pair["pair_idx"]
        bl_p, inj_p, tgt_p = wav_paths(args.wav_dir, pidx)
        if (not args.overwrite) and all(os.path.exists(p) for p in (bl_p, inj_p, tgt_p)):
            n_skip += 1
            if str(pidx) not in rows:
                rows[str(pidx)] = {"pair_idx": pidx, "word_a": pair["word_a"],
                                   "word_b": pair["word_b"], "status": "resumed_wavs_only"}
            continue
        try:
            tp = time.time()
            bl = synthesize(bundle, pair["text_a"], seed=args.seed, **kw)
            tgt = synthesize(bundle, pair["text_b"], seed=args.seed, **kw)
            inj = splice_synthesize(
                bundle, pair["text_a"], pair["word_a"],
                pair["text_b"], pair["word_b"], seed=args.seed,
                collect_kappa=not args.no_kappa, **kw)

            n_bl, d_bl, c_bl = save_wav(bl_p, bl["audio"], bl["sample_rate"])
            n_tg, d_tg, c_tg = save_wav(tgt_p, tgt["audio"], tgt["sample_rate"])
            n_in, d_in, c_in = save_wav(inj_p, inj["audio"], inj["sample_rate"])
            row = {
                "pair_idx": pidx, "status": "ok",
                "word_a": pair["word_a"], "word_b": pair["word_b"],
                "text_a": pair["text_a"], "text_b": pair["text_b"],
                "spliced_surface_text": inj["spliced_surface_text"],
                "s_span": inj["s_span"], "t_span": inj["t_span"],
                "span_info": inj["span_info"],
                "span_aligned": inj["span_info"]["aligned"],
                "n_tokens_src": inj["n_tokens_src"], "n_tokens_tgt": inj["n_tokens_tgt"],
                "n_tokens_spliced": inj["n_tokens_spliced"],
                "span_len_src": inj["span_len_src"], "span_len_tgt": inj["span_len_tgt"],
                "delta_tok": inj["delta_tok"],
                "kappa_encoder_out": inj["kappa_encoder_out"],
                "kappa_by_layer": inj["kappa_by_layer"],
                "n_frames": [bl["n_frames"], tgt["n_frames"], inj["n_frames"]],
                "hit_frame_cap": [bl["hit_frame_cap"], tgt["hit_frame_cap"],
                                  inj["hit_frame_cap"]],
                "dur_bl_sec": round(d_bl, 3), "dur_tgt_sec": round(d_tg, 3),
                "dur_inj_sec": round(d_in, 3),
                "clip_ratio": [round(c_bl, 6), round(c_tg, 6), round(c_in, 6)],
                "used_seed": [bl["used_seed"], tgt["used_seed"], inj["used_seed"]],
                "elapsed_sec": round(time.time() - tp, 2),
            }
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
    import transformers  # noqa: E402
    manifest = {
        "meta": {
            "model_id": args.model_id,
            "codec_id": args.codec_id,
            "dataset": "ja_cross_culture_names.build_300_pairs",
            "lang": args.lang,
            "lang_token": args.lang_token,
            "n_target": len(pairs),
            "seed": args.seed,
            "precision": args.precision,
            "device": args.device,
            "decode": {
                "max_new_frames": args.max_new_frames,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
            },
            "frame_rate_hz": 12.5,
            "sampling_rate": bundle.codec_sr,
            "wav_dir": args.wav_dir,
            "model_config": bundle.meta,
            "splice": (
                "encoder-output span replacement fed to generate_audio via the "
                "encoder_outputs argument, with input_ids=None and "
                "attention_mask=None so _build_cross_mask returns ones(B,1,1,S) "
                "with S taken from the spliced tensor "
                "(modeling_mamba2_s2s.py:419-425, 550-557). Sequence length goes "
                "T -> T-m+n; CrossAttention has no RoPE so no position re-indexing "
                "is needed. Entity spans follow the same rule as the three "
                "existing models (speechsplice_release/src/injection.py:401 "
                "find_entity_token_range): token-id subsequence search first, then "
                "a character-position fallback that tokenizes the prefix on its "
                "own. No window expansion and no token-boundary check; rows record "
                "span_aligned so the misaligned cases can be counted."),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "n_generated": n_done, "n_skipped": n_skip, "n_error": n_err,
            "n_span_misaligned": sum(
                1 for r in all_rows
                if r.get("status") == "ok" and r.get("span_aligned") is False),
            "wallclock_sec": round(time.time() - t0, 1),
        },
        "kappa": summarize_kappa([r for r in all_rows if r.get("status") == "ok"]),
        "rows": all_rows,
    }
    man_path = os.path.join(args.out_dir, f"kiseki_{args.lang}_generation_manifest.json")
    atomic_json_dump(manifest, man_path)
    print(f"\n[manifest] -> {man_path}")
    print(f"[manifest] generated={n_done} skipped={n_skip} error={n_err} "
          f"span_misaligned={manifest['meta']['n_span_misaligned']}")
    if manifest["kappa"].get("encoder_out"):
        print(f"[kappa] encoder_out mean = {manifest['kappa']['encoder_out']['mean']:.4f} "
              f"(n={manifest['kappa']['encoder_out']['n']})")
    del bundle
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return man_path


# --------------------------------------------------------------------------
# scorers (既存ファイルをサブプロセスでそのまま呼ぶ)
# --------------------------------------------------------------------------
def run_scorers(args):
    """既存スコアラを呼ぶ。

    pysptk / pykakasi / phonemizer(espeak) / wav2vec2 は既存 9 セルを出したときと
    同じイメージ（my-lab-image:latest）で動かすこと。Kiseki の生成側も同じ
    イメージで動くので、生成と同じジョブの中で続けて回してよい。
    """
    env = dict(os.environ)
    env["HF_HOME"] = os.environ["HF_HOME"]
    env["HF_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    env["REPO_ROOT"] = PROJECT_ROOT
    mcd_json = args.mcd_json or os.path.join(
        PROJECT_ROOT, "eval_results", f"_caseSPTK_whole_kiseki_{args.lang}.json")
    pib_json = args.pib_json or os.path.join(
        PROJECT_ROOT, "eval_results", f"_caseSPTK_phperPER_kiseki_{args.lang}.json")
    jobs = [
        ("MCD", [sys.executable, MCD_SCORER,
                 "--wav_dir", args.wav_dir, "--lang", args.lang,
                 "--mode", "whole", "--n", str(args.n),
                 "--output_json", mcd_json]),
        ("PIB", [sys.executable, PIB_SCORER,
                 "--wav_dir", args.wav_dir, "--lang", args.lang,
                 "--n", str(args.n), "--output_json", pib_json]),
    ]
    for name, cmd in jobs:
        print(f"\n[scorer:{name}] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        print(f"[scorer:{name}] exit={r.returncode}", flush=True)
        if r.returncode != 0:
            print(f"[scorer:{name}] FAILED", flush=True)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="ja", help="Kiseki-TTS は日本語専用")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--wav_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--codec_id", default=DEFAULT_CODEC_ID)
    p.add_argument("--lang_token", default=DEFAULT_LANG_TOKEN)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--codec_device", default=None)
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"],
                   help="checkpoint は F32。既定は fp32")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--hf_home", default=None)
    # decoding: 3 条件で共有する（差は splice だけ）
    p.add_argument("--max_new_frames", type=int, default=DEFAULT_MAX_NEW_FRAMES,
                   help="12.5Hz。250 frame = 20 秒")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--repetition_penalty", type=float,
                   default=DEFAULT_REPETITION_PENALTY)
    # control
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--no_kappa", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    # scoring
    p.add_argument("--run_scorers", action="store_true",
                   help="生成後に mcd_pysptk.py と phoneme_per.py を呼ぶ")
    p.add_argument("--mcd_json", default=None)
    p.add_argument("--pib_json", default=None)
    args = p.parse_args()

    if args.hf_home:
        setup_hf_home(args.hf_home)
    if args.wav_dir is None:
        args.wav_dir = os.path.join(PROJECT_ROOT, "eval_results", "_cc_wavs_a",
                                    f"kiseki_{args.lang}")
    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, "eval_results",
                                    f"kiseki_cross_culture_{args.lang}")
    if not os.path.isabs(args.wav_dir):
        args.wav_dir = os.path.join(PROJECT_ROOT, args.wav_dir)
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.chdir(PROJECT_ROOT)

    pairs = load_pairs(args.lang, args.n)
    print(f"[data] ja_cross_culture_names n={len(pairs)} "
          f"first={pairs[0]['word_a']}->{pairs[0]['word_b']} "
          f"last={pairs[-1]['word_a']}->{pairs[-1]['word_b']}", flush=True)
    print(f"[data] wav_dir={args.wav_dir}", flush=True)

    if not args.skip_generation:
        run_generation(args, pairs)
    if args.run_scorers:
        run_scorers(args)


if __name__ == "__main__":
    main()
