#!/usr/bin/env python3
"""LongCat の層別掃引（GPU 1 枚）。

## 何をするか

既存の `longcat_splice.py:478 build_spliced_embedding` は
`H = LN(h_L) + LN(h_0)` を組んだ**後で**差し替えるので、
**差し替えた状態は 1 層も通らない。**

この版は umt5 の**層 L の出力で差し替え、L+1 以降を通してから** H を組む。

    層 L の出力 -> 差し替え -> L+1, ..., 最終 -> LN(h_final)+LN(h_0) = H -> DiT

**層番号は通過層数で指定する**（2026-09-19 統一基準）。
LongCat は L=12、公表は l*=12。掃引は **1 / 6 / 12**。

## 最終層での一致（--inject_layer 12）

通過層数 12 では層を 1 つも回し直さないので、式は既存本番の
`splice_part="both"` と同じになる。よって既存との md5 比較が
そのまま実装の検算になる。**`--match_final` は要らない**（残してあるが効かない）。

## 長さについて

**umt5 は層ごとに相対位置バイアスを持つ**ので、途中の層から長さを変えて
残りの層を回し直せる。h_0 も同じ位置で継ぐので足し算の長さが揃う。

**2026-09-20 に方式を変えた。**それ以前は forward hook で層の出力だけを
差し替えていたが、呼び出しの入口で決まる注意マスクが元の長さのままなので、
語の刻み数が変わる対では必ず落ちた。詳細は longcat_layer_parts.py の末尾。
"""
import argparse
import hashlib
import json
import os
import sys
import time

_hf = "/mnt/.cache/huggingface" if os.path.isdir("/mnt/scripts") else \
    os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")

_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else \
    os.environ.get("REPO_ROOT", "/mnt")
for _p in (os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "scripts", "longcat"),
           os.path.join(PROJECT_ROOT, "scripts", "layer_ablation"),
           os.path.join(PROJECT_ROOT, "data")):
    sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

import longcat_splice as L  # noqa: E402
import run_longcat_eval as R  # noqa: E402
from longcat_layer_parts import (
    encode_with_splice_resume,  # noqa: E402
    encode_layer_out, encode_with_splice, find_text_layers,
)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--inject_layer", type=int, required=True,
                   help="**通過層数**（統一基準）。12 = 最終出力")
    p.add_argument("--match_final", action="store_true",
                   help="h_0 も目標側にする。**一致検査のときだけ。**")
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--compare_dir", default=None)
    p.add_argument("--steps", type=int, default=L.DEFAULT_STEPS)
    p.add_argument("--cfg_strength", type=float, default=L.DEFAULT_CFG)
    p.add_argument("--guidance_method", default=L.DEFAULT_GUIDANCE)
    p.add_argument("--seed", type=int, default=L.SEED)
    p.add_argument("--ckpt_every", type=int, default=10)
    # load_longcat に渡す引数（run_longcat_eval.py:182 と同じ既定）
    p.add_argument("--model_id", default=L.DEFAULT_MODEL_ID)
    p.add_argument("--tokenizer_id", default=L.DEFAULT_TOKENIZER_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--no_vae_half", action="store_true")
    p.add_argument("--load_mode", default="meta", choices=["meta", "cpu"])
    a = p.parse_args()

    os.makedirs(a.wav_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    print("[Info] threads=%s" % _thr, flush=True)

    pairs = R.load_pairs(a.lang, a.n)
    print("[data] cross_culture lang=%s n=%d" % (a.lang, len(pairs)), flush=True)

    # **run_longcat_eval に load_bundle は無い。**同 182 行と同じ呼び方をする。
    bundle = L.load_longcat(model_id=a.model_id, tokenizer_id=a.tokenizer_id,
                            device=a.device, dtype=getattr(torch, a.dtype),
                            vae_half=not a.no_vae_half, load_mode=a.load_mode)
    layers = find_text_layers(bundle.model)
    n_layers = len(layers)
    print("[Info] text_encoder の層数 = %d" % n_layers, flush=True)
    if not (1 <= a.inject_layer <= n_layers):
        raise SystemExit("--inject_layer は 1..%d（通過層数）" % n_layers)
    hook_index = a.inject_layer - 1
    is_final = a.inject_layer == n_layers
    print("[Info] 通過層数 %d -> block[%d]%s"
          % (a.inject_layer, hook_index, "（最終出力）" if is_final else ""),
          flush=True)

    rows = []
    ckpt = os.path.join(a.out_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
        print("[Info] resume: %d 行" % len(rows), flush=True)
    done = {r["pair_idx"] for r in rows}

    t0 = time.time()
    for i, pr in enumerate(pairs):
        if i in done:
            continue
        rec = {"pair_idx": i, "word_a": pr["word_a"], "word_b": pr["word_b"]}
        try:
            src = L.encode_for_text(bundle, pr["text_a"])
            tgt = L.encode_for_text(bundle, pr["text_b"])
            s_span, t_span, _ = L.entity_token_span_pair(
                bundle.tokenizer, pr["text_a"], pr["word_a"],
                pr["text_b"], pr["word_b"], span_extension=True)
            L.verify_entity_span(s_span, strict=True)
            L.verify_entity_span(t_span, strict=True)
        except Exception as e:
            rec["skip"] = "span: %s" % str(e)[:120]
            rows.append(rec)
            continue

        s0, s1 = s_span["tok_start"], s_span["tok_end"]
        u0, u1 = t_span["tok_start"], t_span["tok_end"]

        ids_a = src["input_ids"]
        ids_b = tgt["input_ids"]
        am_a = src["attention_mask"]
        am_b = tgt["attention_mask"]

        n_new = (u1 - u0)
        n_old = (s1 - s0)

        # **2026-09-20: hook を使う版をやめた。**
        # 語の刻み数が変わる対（John -> Yamada など）では、呼び出しの入口で
        # 決まる注意マスクが元の長さのままなので、後続の層で必ず落ちていた
        # （tensor a (10) vs b (9)）。継いでから残りの層を回し直す版に替える。
        # 詳細は longcat_layer_parts.py の末尾の覚書。
        parts = encode_with_splice_resume(
            bundle.model, ids_a, am_a, ids_b, am_b,
            a.inject_layer, (s0, s1), (u0, u1))

        dur, dur_sec = L.duration_frames(bundle, src["norm_text"])
        wav = L.generate_from_embedding(
            bundle, parts["H"], dur, seed=a.seed, steps=a.steps,
            cfg_strength=a.cfg_strength, guidance_method=a.guidance_method)
        path = os.path.join(a.wav_dir, "s%04d_inj.wav" % i)
        # **save_wav は run_longcat_eval 側にある**（longcat_splice には無い）
        # **LongCat は `sampling_rate`、Irodori は `sample_rate`。**
        # 名前が違うので実物で確認すること（2026-09-19 にここで落ちた）。
        R.save_wav(path, wav, bundle.sampling_rate)
        rec.update({"s_span": [s0, s1], "t_span": [u0, u1],
                    "delta_tok": n_new - n_old,
                    "T_src": parts["T_src"], "T_new": parts["T_new"],
                    "n_layers_rerun": parts["n_layers_rerun"],
                    "md5_inj": md5_file(path)})
        rows.append(rec)
        if (i + 1) % a.ckpt_every == 0:
            json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)
            print("  [%d/%d] %.1fmin" % (len(rows), len(pairs),
                                         (time.time() - t0) / 60), flush=True)

    json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)

    summary = {"model": "LongCat", "lang": a.lang,
               "inject_layer_passed": a.inject_layer,
               "hook_index": hook_index, "n_layers": n_layers,
               "is_final": is_final, "match_final": a.match_final,
               "seed": a.seed, "steps": a.steps,
               "cfg_strength": a.cfg_strength,
               "n_rows": len(rows),
               "n_skipped": sum(1 for r in rows if r.get("skip")),
               "elapsed_min": round((time.time() - t0) / 60, 2)}

    if a.compare_dir:
        n_match = n_cmp = 0
        for r in rows:
            if r.get("skip"):
                continue
            mine = os.path.join(a.wav_dir, "s%04d_inj.wav" % r["pair_idx"])
            theirs = os.path.join(a.compare_dir, "s%04d_inj.wav" % r["pair_idx"])
            if not (os.path.exists(mine) and os.path.exists(theirs)):
                continue
            n_cmp += 1
            n_match += md5_file(mine) == md5_file(theirs)
        summary["md5_vs_existing"] = {"n": n_cmp, "n_match": n_match,
                                      "compare_dir": a.compare_dir}
        print("\n=== 既存経路との md5: %d/%d 一致 ===" % (n_match, n_cmp))
        if is_final and a.match_final and n_cmp and n_match != n_cmp:
            print("**最終層 + match_final なのに一致しない。実装が違う可能性。**")

    out = os.path.join(a.out_dir, "results.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print("\n[Done] %s" % out, flush=True)
    for k, v in summary.items():
        print("  %-24s %s" % (k, v))


if __name__ == "__main__":
    main()
