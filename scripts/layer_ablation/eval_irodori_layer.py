#!/usr/bin/env python3
"""Irodori の層別掃引（GPU 1 枚）。

## 何をするか

既存の `irodori_splice.py:517 ConditionSplicer` は `encode_conditions` の
**戻り値**（ModernBERT 25 層 -> 射影 768->512 -> RMSNorm を通った後の
`text_state`）を差し替える。**差し替えた状態は backbone を 1 層も通らない。**

この版は ModernBERT の**層 L の出力で差し替え、L+1 以降と射影・RMSNorm を
通してから** `text_state` にする。

    ModernBERT 層 L -> 差し替え -> L+1..24 -> 射影 -> RMSNorm -> text_state -> DiT

**層番号は通過層数で指定する**（2026-09-19 統一基準）。
Irodori は L=25、公表は l*=25。掃引は **1 / 13 / 25**。

## 落とし穴 1: backbone は 2 回呼ばれる

`encode_conditions` は text 用と caption 用で backbone を呼ぶ
（`irodori_splice.py:22-23`）。層に素朴に hook を掛けると caption 側も
差し替わる。**この版は `text_encoder` を直接呼ぶ経路**
（`irodori_splice.py:737-739` と同じ）を使うので、その心配が無い。

## 落とし穴 2: text_mask を同時に作り直す

`text_state` は固定長 `(B, max_text_len, 512)` で、有効長は `text_mask` が
決める（同 8-9 行、25-26 行）。**長さが変わる差し替えでは mask も作り直す。**
既存の `build_splice` がやっていることをそのまま踏襲する。

## 最終層での一致（--inject_layer 25）

層 25（= 最終層）で差し替えたものは、既存の `splice_synthesize` と
**同じにはならない。**既存は射影・RMSNorm を**通った後**の 512 次元で
差し替えるが、この版は**通る前**の 768 次元で差し替えるため。
つまり「同じ位置」ではない。

そこで一致検査は `--inject_layer 25 --after_proj` で行う。
`--after_proj` を付けると既存と同じ「射影後の text_state を差し替える」
経路を通り、md5 が一致するはずである。**本番では付けない。**
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
# **`scripts/irodori/_repo` を通す。**`irodori_tts` はそこにある
# （run_irodori_eval.py:68-70 と同じ並び）。通さないと
# `ModuleNotFoundError: No module named 'irodori_tts'` で落ちる。
for _p in (os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "scripts", "irodori"),
           os.path.join(PROJECT_ROOT, "scripts", "irodori", "_repo"),
           os.path.join(PROJECT_ROOT, "data"),
           os.path.join(PROJECT_ROOT, "scripts", "layer_ablation"),
           os.path.join(PROJECT_ROOT, "data")):
    sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

# **transformers の版差を吸収する。**共通の入口を使う
# （`scripts/irodori/_compat_transformers.py`）。
import _compat_transformers  # noqa: F401,E402

import irodori_splice as I  # noqa: E402
from irodori_layer_parts import find_backbone_layers  # noqa: E402
from layer_hook import LayerSplicer  # noqa: E402


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def text_state_with_splice(bundle, norm_text, text_ids, text_mask,
                           layers, hook_index, ent, span):
    """層 `hook_index` で差し替えてから text_state を作る（**長さ不変のときだけ**）。

    `irodori_splice.py:736-739` と同じ経路（text_encoder -> text_norm）に
    hook を重ねるだけ。**backbone を直接呼ばないので caption 側に触れない。**
    """
    model = bundle.runtime.model
    sp = LayerSplicer(layers, hook_index).attach()
    sp.arm(ent, span)
    try:
        with torch.inference_mode():
            st = model.text_encoder(model.pretrained_text_backbone,
                                    text_ids, text_mask)
            st = model.text_norm(st)
    finally:
        sp.disarm().detach()
    if sp.n_spliced == 0:
        raise RuntimeError("差し替えが起きていない（hook が効いていない）")
    return st, sp.n_spliced


def text_state_with_splice_carrier(bundle, layers, hook_index,
                                   h_src, h_tgt, ids_src, ids_tgt,
                                   s_span, t_span):
    """**長さが変わってもよい版（2026-09-20 追加）。**

    ## なぜ要るか

    上の hook 方式は、呼び出しの入口で決まるマスクと位置が**元の長さのまま**
    残るので、語の刻み数が変わる対で後続の層と食い違う。旧版はそれを避けるため
    `length_change` として**54/300 を飛ばしていた**（田中 3 -> タナカ 1 が 50 件）。
    `n=300 は絶対条件`なので使えない。

    ## 運び手（carrier）方式

    T5Gemma の層別（eval_en_crossculture_t5gemma_layer.py:150-186）と同じ手。

      1. 出所側・目標側をそれぞれ通して層 l の状態を取る（呼び出し側で実施）
      2. その段で継ぐ    h = [h_src[:s0], h_tgt[u0:u1], h_src[s1:]]
      3. **同じ長さの ID を作って**本体を呼び直し、層 l の出力を h で丸ごと置換

    ModernBERT は `attention_mask=None` なら入力長から全 1 を作り、
    `position_ids` も `arange(seq_len)`、`sliding_window_mask` も
    `_update_attention_mask` が内部で作る（modeling_modernbert.py の
    ModernBertModel.forward）。**新しい長さの ID を渡すだけで 3 つとも揃う。**

    運び手の**中身は結果に影響しない**（層 l の出力が丸ごと置き換わるため）。
    長さを決めるためだけに使う。
    """
    model = bundle.runtime.model
    s0, s1 = s_span
    u0, u1 = t_span
    h = torch.cat([h_src[:, :s0], h_tgt[:, u0:u1], h_src[:, s1:]], dim=1)

    car = torch.cat([ids_src[:, :s0], ids_tgt[:, u0:u1], ids_src[:, s1:]], dim=1)
    if int(car.shape[1]) != int(h.shape[1]):
        raise AssertionError("運び手 %d と状態 %d の長さが違う"
                             % (int(car.shape[1]), int(h.shape[1])))
    mask = torch.ones(car.shape, dtype=text_mask_dtype(model), device=car.device)

    fired = {"n": 0}

    def _hook(_m, _i, out):
        fired["n"] += 1
        if fired["n"] > 1:
            return out
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        return h

    hd = layers[hook_index].register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            st = model.text_encoder(model.pretrained_text_backbone, car, mask)
            st = model.text_norm(st)
    finally:
        hd.remove()
    if fired["n"] != 1:
        raise RuntimeError("層 %d が %d 回呼ばれた（1 回のはず）"
                           % (hook_index, fired["n"]))
    return st, 1, int(h.shape[1])


def text_mask_dtype(model):
    """text_mask の dtype を既存経路に合わせる（bool か long か実物で決める）。"""
    return torch.bool


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="ja", choices=["ja"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--inject_layer", type=int, required=True,
                   help="**通過層数**（統一基準）。25 = 最終層")
    p.add_argument("--after_proj", action="store_true",
                   help="射影後の text_state を差し替える（既存と同じ位置）。"
                        "**一致検査のときだけ。**")
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--compare_dir", default=None)
    p.add_argument("--seed", type=int, default=I.SEED)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--link_bl_tgt_from",
                   default="/mnt/eval_results/_cc_wavs_a/irodori_ja",
                   help="BL/TGT の持ち込み元。空文字なら持ち込まない")
    a = p.parse_args()

    os.makedirs(a.wav_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    print("[Info] threads=%s" % _thr, flush=True)

    from ja_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()[:a.n]
    print("[data] cross_culture ja n=%d" % len(pairs), flush=True)

    bundle = I.load_irodori()
    layers = find_backbone_layers(bundle.runtime.model)
    n_layers = len(layers)
    print("[Info] ModernBERT の層数 = %d" % n_layers, flush=True)
    if not (1 <= a.inject_layer <= n_layers):
        raise SystemExit("--inject_layer は 1..%d（通過層数）" % n_layers)
    hook_index = a.inject_layer - 1
    is_final = a.inject_layer == n_layers
    print("[Info] 通過層数 %d -> layers[%d]%s%s"
          % (a.inject_layer, hook_index,
             "（最終層）" if is_final else "",
             " + 射影後で差し替え" if a.after_proj else ""), flush=True)

    from irodori_tts.inference_runtime import SamplingRequest

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
            if a.after_proj:
                # **既存と同じ位置で差し替える**（一致検査用）
                out = I.splice_synthesize(
                    bundle, pr["text_a"], pr["word_a"],
                    pr["text_b"], pr["word_b"], seed=a.seed,
                    seconds=a.seconds, num_steps=a.num_steps,
                    collect_kappa=False, span_extension=True)
                wav = out["audio"]
                rec.update({"s_span": out.get("s_span"),
                            "t_span": out.get("t_span")})
            else:
                info = I.build_splice(bundle, pr["text_a"], pr["word_a"],
                                      pr["text_b"], pr["word_b"],
                                      collect_kappa=False, span_extension=True)
                s0, s1 = info["s_span"]
                u0, u1 = info["t_span"]
                # **長さが変わる対も扱う（2026-09-20）。**旧版はここで
                # length_change として 54/300 を飛ばしていた。運び手方式に変えた。
                norm_b = I.normalize_for_runtime(pr["text_b"])
                tgt = I.encode_text_only(bundle, norm_b, output_hidden_states=True)
                h_tgt = tgt["backbone_hidden_states"][a.inject_layer]

                norm_a = I.normalize_for_runtime(pr["text_a"])
                src = I.encode_text_only(bundle, norm_a, output_hidden_states=True)
                h_src = src["backbone_hidden_states"][a.inject_layer]

                st, n_sp, t_new = text_state_with_splice_carrier(
                    bundle, layers, hook_index, h_src, h_tgt,
                    src["input_ids"], tgt["input_ids"], (s0, s1), (u0, u1))
                rec["T_src"] = int(h_src.shape[1])
                rec["T_new"] = int(t_new)

                # **固定長に詰め直す（2026-09-21）。**運び手は継いだ長さ
                # （T_new）のまま出てくるが、モデルの text_state は常に
                # (1, max_text_len, 512) で、ConditionSplicer は形が違うと
                # 弾く（irodori_splice.py:552-554）。語の刻み数が変わる
                # 54/300 がここで落ちていた。有効長の外は 0 で埋める
                # （上流も mask の外を 0 にしているので同じ扱い）。
                max_len = int(info["text_mask"].shape[1])
                if int(st.shape[1]) > max_len:
                    # 目標の語の方が刻み数が多いと 256 を超える。超えた分は
                    # 末尾の詰め物なので落とす（有効長が収まることは
                    # build_splice が既に検めている。irodori_splice.py:466-469）。
                    st = st[:, :max_len]
                if int(st.shape[1]) < max_len:
                    pad = torch.zeros(st.shape[0], max_len - int(st.shape[1]),
                                      st.shape[2], dtype=st.dtype,
                                      device=st.device)
                    st = torch.cat([st, pad], dim=1)

                req = SamplingRequest(
                    text=info["spliced_surface_text"],
                    **I._base_request_kwargs(a.seconds, a.seed, a.num_steps, None))
                I.seed_all(a.seed)
                with torch.inference_mode(), \
                        bundle.splicer.armed(st, info["text_mask"]):
                    result = bundle.runtime.synthesize(req)
                wav = result.audio if hasattr(result, "audio") else result
                rec.update({"s_span": [s0, s1], "t_span": [u0, u1],
                            "n_spliced": n_sp})
        except Exception as e:
            rec["skip"] = "%s: %s" % (type(e).__name__, str(e)[:110])
            rows.append(rec)
            continue

        path = os.path.join(a.wav_dir, "s%04d_inj.wav" % i)
        import run_irodori_eval as RI
        RI.save_wav(path, wav, bundle.sample_rate)
        rec["md5_inj"] = md5_file(path)
        rows.append(rec)
        if (i + 1) % a.ckpt_every == 0:
            json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)
            print("  [%d/%d] %.1fmin" % (len(rows), len(pairs),
                                         (time.time() - t0) / 60), flush=True)

    json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)

    # **BL / TGT を本体セルから持ち込む。**層を変えても基線と目標は変わらない
    # ので作り直さない。採点器は 3 本組を読むため、無いと測れない。
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

    summary = {"model": "Irodori-TTS", "lang": a.lang,
               "link_bl_tgt_from": a.link_bl_tgt_from,
               "n_linked": n_link, "n_link_missing": n_link_miss,
               "inject_layer_passed": a.inject_layer,
               "hook_index": hook_index, "n_layers": n_layers,
               "is_final": is_final, "after_proj": a.after_proj,
               "seed": a.seed, "n_rows": len(rows),
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
        if a.after_proj and n_cmp and n_match != n_cmp:
            print("**after_proj なのに一致しない。実装が違う可能性。**")

    out = os.path.join(a.out_dir, "results.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print("\n[Done] %s" % out, flush=True)
    for k, v in summary.items():
        print("  %-24s %s" % (k, v))


if __name__ == "__main__":
    main()
