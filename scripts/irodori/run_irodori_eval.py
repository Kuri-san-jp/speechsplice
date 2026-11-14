#!/usr/bin/env python3
"""Irodori-TTS v4.1-Small に SpeechSplice を適用して日本語 300 ペアを生成する（GPU）.

生成するもの（既存 3 モデルと同じ命名規約。3 本そろっていないペアはスコアラ側で
available:false として捨てられるので TGT の生成を省略してはならない。
mcd_pysptk.py:228-229, phoneme_per.py:166-167）:

    {wav_dir}/s{pair_idx:04d}_bl.wav    = TTS(text_a)
    {wav_dir}/s{pair_idx:04d}_inj.wav   = SpeechSplice(text_a の text 条件に
                                          text_b の実体語スパンを移植)
    {wav_dir}/s{pair_idx:04d}_tgt.wav   = TTS(text_b)

評価データは speechsplice_release/data/ja_cross_culture_names.py の
build_300_pairs()。**Table 3 の JA 行（T5Gemma / SpeechT5 / Qwen3）が使っている
のと同じデータ**で、mcd_pysptk.py / phoneme_per.py が内部で同じものを読むので
無改造で流用できる。

BL / TGT / INJ は テキスト以外（乱数種・参照音声なし・caption なし・CFG・
RF ステップ数）を完全に共有する。差は splice だけ。

指標の計算は既存ファイルをそのまま使う（1 行も改変していない）:
    MCD : speechsplice_release/src/mcd_pysptk.py --mode whole
    PIB : speechsplice_release/src/phoneme_per.py --lang ja

**スコアラは既存イメージ（my-lab-image:latest）で回すこと。** 本スクリプトの生成側は
transformers 5.x / torch 2.10 の新イメージが要るが、スコアラを新イメージで回すと
wav2vec2 や phonemizer の挙動が既存 9 セルと揃う保証が無くなる。詳細は README.md。

--span_extension を付けると、両側の span_aligned が揃わない対（トークン化器が実体語と
後続の「さん」を 1 トークンに融合する類）に限り、最小対称拡張
(scripts/projection/span_extend.py) を試す。既定は off で従来どおりずれたまま注入する。
両側 aligned の対は on/off でスパンが 1 ビットも変わらない。

実行例（GPU、生成のみ）:
  IMAGE_NAME=irodori-tts:latest SLURM_TIME=08:00:00 \\
    slurm_run.sh \\
      python3 scripts/irodori/run_irodori_eval.py --n 300
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
# scripts/analysis/table2_whisper_inj_lv3.py:24-28、scripts/dia/run_dia_eval.py:46-52
# と同じ扱い。transformers を import する前に済ませること。
# --------------------------------------------------------------------------
DEFAULT_HF_HOME = os.path.join(PROJECT_ROOT, ".cache", "huggingface")


def setup_hf_home(hf_home):
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_MODULES_CACHE"] = os.path.join(hf_home, "modules")
    return hf_home


setup_hf_home(os.environ.get("IRODORI_HF_HOME", DEFAULT_HF_HOME))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "irodori"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "irodori", "_repo"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# **transformers の版差を吸収する。**`irodori_splice` より前に読むこと。
# `irodori_tts/model.py:686` が `transformers.initialization` を要求するが、
# その場所に移ったのは transformers 4.57 より後。実行環境は 4.57.3 で
# 旧い場所にある。詳細は `scripts/irodori/_compat_transformers.py`。
import _compat_transformers  # noqa: F401,E402

from irodori_splice import (  # noqa: E402
    DEFAULT_CODEC_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_STEPS,
    SEED,
    frames_to_seconds,
    load_irodori,
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
            f"Irodori-TTS は日本語特化モデル（モデルカード language: ja）。"
            f"--lang {lang} は対象外。")
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
    層 index 0 = ModernBERT-Ja の embedding 出力、以降 1..25 が 25 層の出力。
    これとは別に kappa_text_state = 射影 (768->512) + text_norm を通った
    **DiT が実際に cross-attend するテンソル** 上の値も出す。

    **既存 Table 3 の kappa 列とは母数も題材も揃っていない。**
    T5Gemma の値は別単語ペア N=1000、SpeechT5 / Qwen3 は heteronym N=8 で、
    aggregate_table3.py:29-33 は kappa を定数として印字するだけで再計算していない。
    ここで出るのは「この cross-culture 300 ペアでの Irodori の値」。
    どの流儀に合わせるかは要判断（README の未解決点を参照）。
    """
    per_layer = {}
    last = []
    for r in rows:
        if r.get("kappa_text_state") is not None:
            last.append(r["kappa_text_state"])
        kl = r.get("kappa_by_layer")
        if not kl:
            continue
        for li, v in enumerate(kl):
            per_layer.setdefault(li, []).append(v)
    out = {}
    for li, vals in sorted(per_layer.items()):
        arr = np.array(vals, dtype=np.float64)
        out[str(li)] = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)), 6) if arr.size > 1 else None,
            "median": round(float(np.median(arr)), 6),
        }
    summary = {"by_layer_modernbert": out}
    if last:
        arr = np.array(last, dtype=np.float64)
        summary["text_state"] = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)), 6) if arr.size > 1 else None,
            "median": round(float(np.median(arr)), 6),
        }
    summary["note"] = (
        "kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}). "
        "by_layer_modernbert: index 0 = ModernBERT-Ja embedding output, 1..25 = its 25 "
        "layers. text_state: after the 768->512 projector and text_norm, i.e. the tensor "
        "the DiT actually cross-attends to. "
        "Corpus/N differ from the existing Table 3 kappa cells; not directly comparable.")
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
    bundle = load_irodori(
        model_id=args.model_id,
        codec_id=args.codec_id,
        device=args.device,
        precision=args.precision,
        codec_device=args.codec_device or args.device,
    )
    print(f"[load] done in {time.time() - t_load:.1f}s "
          f"sr={bundle.sample_rate} hop={bundle.hop_length} "
          f"max_text_len={bundle.max_text_len}", flush=True)

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
        if (not args.overwrite) and all(os.path.exists(p) for p in (bl_p, inj_p, tgt_p)):
            n_skip += 1
            if str(pidx) not in rows:
                rows[str(pidx)] = {"pair_idx": pidx, "word_a": pair["word_a"],
                                   "word_b": pair["word_b"], "status": "resumed_wavs_only"}
            continue
        try:
            tp = time.time()
            bl = synthesize(bundle, pair["text_a"], seed=args.seed,
                            num_steps=args.num_steps)
            tgt = synthesize(bundle, pair["text_b"], seed=args.seed,
                             num_steps=args.num_steps)

            # INJ の長さの決め方。既定 predict = duration predictor を
            # **spliced な text 条件の上で** 走らせる（BL/TGT のどちらにも
            # 寄せないので MCD(INJ,BL) と MCD(INJ,TGT) のどちらにも下駄を
            # 履かせない）。このとき encode_conditions は 2 回走るが、
            # armed なパッチが両方に同じ spliced 条件を返すので一貫している。
            inj_seconds = None
            if args.duration_mode == "tgt":
                if tgt["latent_frames"] is None:
                    raise RuntimeError("could not read predicted frames from TGT messages")
                inj_seconds = frames_to_seconds(bundle, tgt["latent_frames"])
            elif args.duration_mode == "bl":
                if bl["latent_frames"] is None:
                    raise RuntimeError("could not read predicted frames from BL messages")
                inj_seconds = frames_to_seconds(bundle, bl["latent_frames"])
            elif args.duration_mode == "fixed":
                inj_seconds = float(args.fixed_seconds)

            # --span_extension を付けたときだけ、両側の aligned が揃わない対に
            # 最小対称拡張を試す (scripts/projection/span_extend.py)。
            # 両側 aligned の対は span_extension に関わらず従来と同じスパンを
            # 通る。既定 off = 従来どおりずれたまま注入する。
            inj = splice_synthesize(
                bundle, pair["text_a"], pair["word_a"],
                pair["text_b"], pair["word_b"], seed=args.seed,
                seconds=inj_seconds, num_steps=args.num_steps,
                span_extension=args.span_extension,
                collect_kappa=not args.no_kappa)

            n_bl, d_bl, c_bl = save_wav(bl_p, bl["audio"].numpy(), bundle.sample_rate)
            n_tg, d_tg, c_tg = save_wav(tgt_p, tgt["audio"].numpy(), bundle.sample_rate)
            n_in, d_in, c_in = save_wav(inj_p, inj["audio"].numpy(), bundle.sample_rate)
            row = {
                "pair_idx": pidx, "status": "ok",
                "word_a": pair["word_a"], "word_b": pair["word_b"],
                "text_a": pair["text_a"], "text_b": pair["text_b"],
                "spliced_surface_text": inj["spliced_surface_text"],
                "s_span": inj["s_span"], "t_span": inj["t_span"],
                "span_info": inj["span_info"],
                "span_aligned": inj["span_info"]["aligned"],
                "span_extension": inj["span_extension"],
                "n_tokens_src": inj["n_tokens_src"], "n_tokens_tgt": inj["n_tokens_tgt"],
                "n_tokens_spliced": inj["n_tokens_spliced"],
                "delta_tok": inj["delta_tok"],
                "n_encode_conditions_calls": inj["n_encode_conditions_calls"],
                "n_spliced": inj["n_spliced"],
                "kappa_text_state": inj["kappa_text_state"],
                "kappa_by_layer": inj["kappa_by_layer"],
                "latent_frames": [bl["latent_frames"], tgt["latent_frames"],
                                  inj["latent_frames"]],
                "inj_seconds_requested": inj_seconds,
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
            "checkpoint_path": bundle.checkpoint_path,
            "tokenizer_path": bundle.tokenizer_path,
            "dataset": "ja_cross_culture_names.build_300_pairs",
            "lang": args.lang,
            "n_target": len(pairs),
            "seed": args.seed,
            "precision": args.precision,
            "device": args.device,
            "num_steps": args.num_steps if args.num_steps else DEFAULT_NUM_STEPS,
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 3.0,
            "cfg_scale_speaker": 5.0,
            "no_ref": True,
            "caption": None,
            "duration_mode": args.duration_mode,
            "span_extension": bool(args.span_extension),
            "sampling_rate": bundle.sample_rate,
            "hop_length": bundle.hop_length,
            "max_text_len": bundle.max_text_len,
            "wav_dir": args.wav_dir,
            "model_config": bundle.meta.get("model_config"),
            "splice": (
                "text_state span replacement at model.encode_conditions "
                "(ModernBERT-Ja-310m -> PretrainedConditionProjector 768->512 -> text_norm); "
                "text_mask rebuilt to the new valid length T -> T-m+n; padded positions "
                "zeroed; tensor shape stays (B, max_text_len, 512). "
                "Entity spans follow the same rule as the three existing models "
                "(speechsplice_release/src/injection.py:401 find_entity_token_range): "
                "token-id subsequence search first, then a character-position "
                "fallback that tokenizes the prefix on its own. No window expansion "
                "and no token-boundary check; rows record span_aligned so the "
                "misaligned cases can be counted."
                + ("" if not args.span_extension else
                   " --span_extension is ON: pairs whose src/tgt spans are not both "
                   "aligned get the minimal symmetric surface extension "
                   "(scripts/projection/span_extend.py, max_pre=3 max_post=3); the "
                   "extended surface is re-resolved with the same "
                   "find_entity_token_range rule and is only adopted when both sides "
                   "then come out aligned. Pairs that were already aligned take the "
                   "unchanged path and their spans are bit-identical to the "
                   "span_extension=off run.")),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "n_generated": n_done, "n_skipped": n_skip, "n_error": n_err,
            "n_span_misaligned": sum(
                1 for r in all_rows
                if r.get("status") == "ok" and r.get("span_aligned") is False),
            "n_span_extended": sum(
                1 for r in all_rows
                if r.get("status") == "ok" and r.get("span_extension")),
            "wallclock_sec": round(time.time() - t0, 1),
        },
        "kappa": summarize_kappa([r for r in all_rows if r.get("status") == "ok"]),
        "rows": all_rows,
    }
    man_path = os.path.join(args.out_dir, f"irodori_{args.lang}_generation_manifest.json")
    atomic_json_dump(manifest, man_path)
    print(f"\n[manifest] -> {man_path}")
    print(f"[manifest] generated={n_done} skipped={n_skip} error={n_err}")
    if manifest["kappa"].get("text_state"):
        print(f"[kappa] text_state mean = {manifest['kappa']['text_state']['mean']:.4f} "
              f"(n={manifest['kappa']['text_state']['n']})")
    del bundle
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return man_path


# --------------------------------------------------------------------------
# scorers (既存ファイルをサブプロセスでそのまま呼ぶ)
# --------------------------------------------------------------------------
def run_scorers(args):
    """既存スコアラを呼ぶ。

    **推奨は既存イメージ（my-lab-image:latest）で別途回すこと。** pysptk /
    pykakasi / phonemizer(espeak) / wav2vec2 は既存 9 セルを出したときと同じ
    バージョンで動かさないと定義上の同一性が担保できない。README を参照。
    """
    env = dict(os.environ)
    env["HF_HOME"] = os.environ["HF_HOME"]
    env["HF_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    env["REPO_ROOT"] = PROJECT_ROOT
    mcd_json = args.mcd_json or os.path.join(
        PROJECT_ROOT, "eval_results", f"_caseSPTK_whole_irodori_{args.lang}.json")
    pib_json = args.pib_json or os.path.join(
        PROJECT_ROOT, "eval_results", f"_caseSPTK_phperPER_irodori_{args.lang}.json")
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
    p.add_argument("--lang", default="ja", help="Irodori は日本語専用")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--wav_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--codec_id", default=DEFAULT_CODEC_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--codec_device", default=None)
    p.add_argument("--precision", default="fp32",
                   choices=["fp32", "bf16", "fp16"],
                   help="モデルカードのベンチマーク条件は FP32")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--hf_home", default=None)
    # decoding: 未指定なら runtime の既定（RF 40 steps / text CFG 3.0）
    p.add_argument("--num_steps", type=int, default=None,
                   help=f"未指定なら runtime の既定 {DEFAULT_NUM_STEPS}")
    p.add_argument("--duration_mode", default="predict",
                   choices=["predict", "tgt", "bl", "fixed"],
                   help="INJ の長さの決め方。predict=spliced 条件で duration "
                        "predictor を走らせる（既定。BL/TGT のどちらにも寄せない）。"
                        "tgt/bl=それぞれの予測長を流用（encode_conditions は 1 回）。"
                        "fixed=--fixed_seconds")
    p.add_argument("--fixed_seconds", type=float, default=6.0)
    p.add_argument("--span_extension", action="store_true",
                   help="両側の aligned が揃わない対 (日本語の「さん」など) に限り "
                        "最小対称拡張 (scripts/projection/span_extend.py) を試す。"
                        "既定 off = 従来どおりずれたまま注入する。"
                        "両側 aligned の対の結果は on/off で 1 ビットも変わらない")
    # control
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--no_kappa", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    # scoring
    p.add_argument("--run_scorers", action="store_true",
                   help="生成後に mcd_pysptk.py と phoneme_per.py を呼ぶ。"
                        "既存イメージで別途回すほうが安全（README 参照）")
    p.add_argument("--mcd_json", default=None)
    p.add_argument("--pib_json", default=None)
    args = p.parse_args()

    if args.hf_home:
        setup_hf_home(args.hf_home)
    if args.wav_dir is None:
        args.wav_dir = os.path.join(PROJECT_ROOT, "eval_results", "_cc_wavs_a",
                                    f"irodori_{args.lang}")
    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, "eval_results",
                                    f"irodori_cross_culture_{args.lang}")
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
