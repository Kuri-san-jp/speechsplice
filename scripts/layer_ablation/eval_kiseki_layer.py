#!/usr/bin/env python3
"""Kiseki の層別掃引（GPU 1 枚）。

## 何をするか

既存の `kiseki_splice.py:570 build_splice` は符号化器の**最終出力**で継ぐ。
この版は通過層数 l の段で継ぎ、**残りの層を回し直してから**復号器へ渡す。

    層 l の出力 -> 継ぐ -> l+1, ..., 最終 -> RMSNorm -> 復号器

**層番号は通過層数で指定する**（2026-09-19 統一基準）。

## Mamba2 なので作り直すのはマスクと位置索引だけ

`modeling_mamba2_s2s.py:312-331` の符号化器は
`x = layer(x, enc_mask, pos)` を繰り返し、最後に RMSNorm をかけるだけ。
**注意機構が無い**ので相対位置バイアスの作り直しは不要。
長さが変わるぶん `enc_mask`（全 1）と `pos`（arange）だけ作り直す。
部品は `kiseki_layer_parts.py`。CPU で検算済み
（最終層で既存の式とビット一致、最大差 0）。

## 最終層での一致

通過層数 = 層数 のとき層を 1 つも回さないので、式は既存の
`build_splice`（最終出力で継ぐ）と同じになる。よって
`--compare_dir` に既存本番を渡せば md5 で実装を検算できる。

**一致しなければ本番を回さない。**

## 使い方

  # 一致検査（n=5）
  SLURM_TIME=01:30:00 IMAGE_NAME=... slurm_run.sh \\
    python3 -u /mnt/scripts/layer_ablation/eval_kiseki_layer.py \\
      --lang ja --n 5 --inject_layer <L> \\
      --wav_dir /mnt/eval_results/_layerabl/ki_equiv \\
      --out_dir /mnt/eval_results/_layerabl/meta_ki_equiv \\
      --compare_dir /mnt/eval_results/_cc_wavs_a/kiseki_ja
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
           os.path.join(PROJECT_ROOT, "scripts", "kiseki"),
           os.path.join(PROJECT_ROOT, "scripts", "layer_ablation"),
           os.path.join(PROJECT_ROOT, "data")):
    sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

import kiseki_splice as K  # noqa: E402
from kiseki_layer_parts import encode_with_splice_resume, encoder_of  # noqa: E402


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="ja", choices=["ja"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--inject_layer", type=int, required=True,
                   help="**通過層数**（統一基準）。層数と同じなら最終出力")
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--compare_dir", default=None)
    # **基線・目標を本体セルから持ち込む（2026-09-21 追加）。**
    # この台本は注入音声しか作らないが、原稿の採点器は bl/inj/tgt の 3 本組を
    # 読む。BL と TGT は注入層に依存しないので、本体セルの音声を hard link で
    # 持ち込めばよい（T5Gemma の層別台本 :321 と同じ作法）。
    p.add_argument("--link_bl_tgt_from",
                   default="/mnt/eval_results/_cc_wavs_a/kiseki_ja",
                   help="BL/TGT の持ち込み元。空文字なら持ち込まない")
    p.add_argument("--seed", type=int, default=K.SEED)
    p.add_argument("--ckpt_every", type=int, default=10)
    a = p.parse_args()

    os.makedirs(a.wav_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    import ja_cross_culture_names as D
    pairs = D.build_300_pairs()[:a.n]
    print("[data] cross_culture %s n=%d" % (a.lang, len(pairs)), flush=True)

    bundle = K.load_kiseki()
    n_layers = len(encoder_of(bundle.model).layers)
    print("[Info] encoder の層数 = %d" % n_layers, flush=True)
    if not (1 <= a.inject_layer <= n_layers):
        raise SystemExit("--inject_layer は 1..%d（通過層数）" % n_layers)
    is_final = a.inject_layer == n_layers
    print("[Info] 通過層数 %d%s"
          % (a.inject_layer, "（最終出力）" if is_final else ""), flush=True)

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
            # **スパンの決め方は既存とまったく同じ**（build_splice の中身を
            # 書き写さず、同じ関数を呼ぶ）。違うのは継ぐ段だけ。
            sp = K.build_splice(bundle, pr["text_a"], pr["word_a"],
                                pr["text_b"], pr["word_b"], collect_kappa=False)
        except Exception as e:
            rec["skip"] = "span: %s" % str(e)[:120]
            rows.append(rec)
            continue

        # **入力 ID は build_input_ids から取る。**build_splice は
        # spliced_enc とスパンしか返さないので、同じ関数で作り直す
        # （build_splice も内部で encode_text -> build_input_ids を通る）。
        # **built_ids は Python のリスト。**テンソルは "input_ids" の方
        # （(1, T)、既に device 上）。リストを取ると .dim() で落ちる。
        ids_a = K.build_input_ids(bundle, pr["text_a"])["input_ids"]
        ids_b = K.build_input_ids(bundle, pr["text_b"])["input_ids"]
        if ids_a.dim() == 1:
            ids_a = ids_a.unsqueeze(0)
        if ids_b.dim() == 1:
            ids_b = ids_b.unsqueeze(0)
        dev = next(bundle.model.parameters()).device
        parts = encode_with_splice_resume(
            bundle.model, ids_a.to(dev), ids_b.to(dev),
            a.inject_layer, tuple(sp["s_span"]), tuple(sp["t_span"]))

        # **生成は splice_synthesize と同じ呼び方にする**（写さず同じ形で呼ぶ）。
        # 差し替えるのは encoder_outputs の中身だけ。
        from transformers.modeling_outputs import BaseModelOutput
        kw = K._decode_kwargs(K.DEFAULT_MAX_NEW_FRAMES, K.DEFAULT_TEMPERATURE,
                              K.DEFAULT_TOP_K, K.DEFAULT_TOP_P,
                              K.DEFAULT_REPETITION_PENALTY)
        K.seed_all(a.seed)
        with torch.inference_mode():
            out = bundle.model.generate_audio(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=parts["enc_out"]),
                input_ids=None, attention_mask=None, **kw)
        codes = K._codes_from_output(out)
        wav, sr = K.codes_to_wav(bundle, codes)
        path = os.path.join(a.wav_dir, "s%04d_inj.wav" % i)
        import soundfile as sf
        import numpy as np
        w = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        sf.write(path, w.squeeze(), int(sr))
        rec.update({"s_span": list(sp["s_span"]), "t_span": list(sp["t_span"]),
                    "T_src": parts["T_src"], "T_new": parts["T_new"],
                    "n_layers_rerun": parts["n_layers_rerun"],
                    "md5_inj": md5_file(path)})
        rows.append(rec)
        if (i + 1) % a.ckpt_every == 0:
            json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)
            print("  [%d/%d] %.1fmin" % (len(rows), len(pairs),
                                         (time.time() - t0) / 60), flush=True)

    json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)

    # BL / TGT を持ち込む。注入音声ができた対だけ張る。
    n_link = n_link_miss = 0
    if a.link_bl_tgt_from:
        import shutil
        for r in rows:
            i = r["pair_idx"]
            if not os.path.exists(os.path.join(a.wav_dir, "s%04d_inj.wav" % i)):
                continue
            for cond in ("bl", "tgt"):
                src = os.path.join(a.link_bl_tgt_from, "s%04d_%s.wav" % (i, cond))
                dst = os.path.join(a.wav_dir, "s%04d_%s.wav" % (i, cond))
                if os.path.exists(dst):
                    continue
                if not os.path.exists(src):
                    n_link_miss += 1
                    continue
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                n_link += 1
        print("[link] BL/TGT を %d 本持ち込み（元が無い %d 本）"
              % (n_link, n_link_miss), flush=True)

    summary = {"model": "Kiseki", "lang": a.lang,
               "link_bl_tgt_from": a.link_bl_tgt_from,
               "n_linked": n_link, "n_link_missing": n_link_miss,
               "inject_layer_passed": a.inject_layer, "n_layers": n_layers,
               "is_final": is_final, "seed": a.seed,
               "n_rows": len(rows),
               "n_skipped": sum(1 for r in rows if "skip" in r),
               "elapsed_min": round((time.time() - t0) / 60, 2)}

    if a.compare_dir:
        n = ok = 0
        for r in rows:
            if "md5_inj" not in r:
                continue
            q = os.path.join(a.compare_dir, "s%04d_inj.wav" % r["pair_idx"])
            if not os.path.exists(q):
                continue
            n += 1
            ok += int(md5_file(q) == r["md5_inj"])
        summary["md5_vs_existing"] = {"n": n, "n_match": ok,
                                      "compare_dir": a.compare_dir}

    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(a.out_dir, "results.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
