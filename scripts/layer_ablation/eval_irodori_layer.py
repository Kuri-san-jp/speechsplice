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

層 25 の後にあるのは ModernBERT の final_norm（LayerNorm）、射影
（Linear と residual MLP）、`* mask`、RMSNorm で、**どれもトークンごとの計算**
である。だから層 25 の出力で差し替えても、射影後の `text_state` で差し替えても
（既存の `splice_synthesize`）、有効な位置の値は同じになる。

一致検査: `--inject_layer 25 --compare_dir <置き換え位置の実行>` で、INJ 音声の
md5 が置き換え位置の実行（`run_irodori_eval.py`）と全対で一致すること。
`--after_proj` は既存の経路そのものを通す（比較用）。

## 2026-11-15 の直し（旧版 `_v2` の 3 つの食い違い）

旧版（`_layerabl/ir_ja_L*_v2`、2026-09-20〜29）は、層 25 でも置き換え位置の
実行と音声が一致しなかった（`meta_ir_equiv_v5`: md5 0/5）。原因は位置の違い
ではなく次の 3 つだった。

1. **最終層で final_norm が 2 回かかっていた。**transformers 5 の
   `capture_outputs` は `hidden_states[-1]` を `last_hidden_state`
   （final_norm 後）に差し替える（`tie_last_hidden_states=True`）。旧版は
   それを層 25 の出力として戻したので、final_norm がもう一度かかっていた。
   → 層の出力は forward hook で取る（`layer_output`）。どの版でも正規化前。
2. **差し替え後の層が詰め物を見ていた。**旧版の運び手は `batch_encode` の
   256 トークン（詰め物込み）を全部 1 のマスクで流したので、層 L+1..25 の
   注意が詰め物にも向いていた。本番の合成では詰め物はマスクされる。
   → 有効長で継ぎ、継いだ後のマスク（`build_splice` の `text_mask`）で流す。
3. **実体語の範囲が置き換え位置の実行と違っていた。**旧版は
   `span_extension=True` で、26 対の範囲が置き換え位置の実行（拡張なし）と
   違った。→ 既定は拡張なし。`--span_extension` で旧版と同じにできる。
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


def layer_output(bundle, layers, hook_index, ids, mask):
    """層 `hook_index` の出力（final_norm の前）を取る。

    `backbone(..., output_hidden_states=True)` の `hidden_states[25]` は使わない。
    transformers 5 では最後の要素が `last_hidden_state`（final_norm 後）に
    差し替わる（`capture_outputs(tie_last_hidden_states=True)`）。forward hook
    なら、どの版でも層そのものの出力になる。

    呼び方は置き換え位置の実行（`encode_text_only` -> `text_encoder` ->
    `pretrained_text_backbone(ids, mask)`）と同じ。詰め物の位置は 0 にする。
    """
    model = bundle.runtime.model
    got = {}

    def _hook(_m, _i, out):
        got["h"] = out[0] if isinstance(out, tuple) else out

    hd = layers[hook_index].register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model.pretrained_text_backbone(ids, mask)
    finally:
        hd.remove()
    h = got["h"]
    if h.dim() != 3:
        raise RuntimeError("層の出力が (B, T, D) でない: %s" % (tuple(h.shape),))
    return h * mask.unsqueeze(-1).to(dtype=h.dtype)


def text_state_with_splice_at_layer(bundle, layers, hook_index, src, tgt,
                                    s_span, t_span, new_mask):
    """層 `hook_index` の出力で継いでから text_state を作る（長さが変わってもよい）。

      1. 出所側・目標側を置き換え位置の実行と同じ呼び方で通し、層の出力を取る
      2. 有効長の中で継ぐ   h = [h_src[:s0], h_tgt[u0:u1], h_src[s1:T_src]]
      3. 継いだ ID 列（後ろは詰め物）と継いだ後のマスク `new_mask`
         （`build_splice` の `text_mask`）で本体を通し直し、層の出力を h で置換する。
         詰め物はマスクされるので、後続の層の有効な位置には効かない。

    層 25 では、この後はトークンごとの計算（final_norm、射影、RMSNorm）だけなので、
    結果は射影後に継ぐ `splice_text_state` と同じになる。
    """
    from irodori_splice import mask_valid_length

    model = bundle.runtime.model
    ids_s, ids_t = src["input_ids"], tgt["input_ids"]
    h_src = layer_output(bundle, layers, hook_index, ids_s, src["text_mask"])
    h_tgt = layer_output(bundle, layers, hook_index, ids_t, tgt["text_mask"])
    n_s = int(src["n_tokens"])
    s0, s1 = s_span
    u0, u1 = t_span
    max_len = int(ids_s.shape[1])

    h = torch.cat([h_src[:, :s0], h_tgt[:, u0:u1], h_src[:, s1:n_s]], dim=1)
    car = torch.cat([ids_s[:, :s0], ids_t[:, u0:u1], ids_s[:, s1:n_s]], dim=1)
    t_new = int(h.shape[1])
    if t_new != mask_valid_length(new_mask[0]):
        raise AssertionError("継いだ長さ %d とマスクの有効長 %d が違う"
                             % (t_new, mask_valid_length(new_mask[0])))
    if t_new > max_len:
        raise AssertionError("継いだ長さ %d が max_text_len %d を超える"
                             % (t_new, max_len))
    if t_new < max_len:
        pad_id = int(bundle.runtime.tokenizer.pad_token_id)
        car = torch.cat([car, torch.full((1, max_len - t_new), pad_id,
                                         dtype=car.dtype, device=car.device)], dim=1)
        h = torch.cat([h, torch.zeros(1, max_len - t_new, h.shape[2],
                                      dtype=h.dtype, device=h.device)], dim=1)
    mask = new_mask.to(device=car.device)

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
    return st, n_s, t_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="ja", choices=["ja"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--inject_layer", type=int, required=True,
                   help="**通過層数**（統一基準）。25 = 最終層")
    p.add_argument("--after_proj", action="store_true",
                   help="射影後の text_state を差し替える（既存と同じ位置）。"
                        "**一致検査のときだけ。**")
    p.add_argument("--span_extension", action="store_true",
                   help="旧版（_v2）と同じく、両側の aligned が揃わない対の範囲を広げる。"
                        "置き換え位置の実行（run_irodori_eval.py）は広げていないので、既定は広げない")
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
                # **既存と同じ位置で差し替える**（比較用）
                out = I.splice_synthesize(
                    bundle, pr["text_a"], pr["word_a"],
                    pr["text_b"], pr["word_b"], seed=a.seed,
                    seconds=a.seconds, num_steps=a.num_steps,
                    collect_kappa=False, span_extension=a.span_extension)
                wav = out["audio"]
                rec.update({"s_span": out.get("s_span"),
                            "t_span": out.get("t_span")})
            else:
                # 範囲とマスクは置き換え位置の実行と同じ build_splice で決める
                info = I.build_splice(bundle, pr["text_a"], pr["word_a"],
                                      pr["text_b"], pr["word_b"],
                                      collect_kappa=False,
                                      span_extension=a.span_extension)
                s0, s1 = info["s_span"]
                u0, u1 = info["t_span"]
                src = I.encode_text_only(bundle, I.normalize_for_runtime(pr["text_a"]))
                tgt = I.encode_text_only(bundle, I.normalize_for_runtime(pr["text_b"]))
                st, t_src, t_new = text_state_with_splice_at_layer(
                    bundle, layers, hook_index, src, tgt,
                    (s0, s1), (u0, u1), info["text_mask"])
                rec["T_src"] = int(t_src)
                rec["T_new"] = int(t_new)

                # 合成は splice_synthesize と同じ呼び方（text_state だけが違う）
                req = SamplingRequest(
                    text=info["spliced_surface_text"],
                    **I._base_request_kwargs(a.seconds, a.seed, a.num_steps, None))
                I.seed_all(a.seed)
                with torch.inference_mode(), \
                        bundle.splicer.armed(st, info["text_mask"]):
                    result = bundle.runtime.synthesize(req)
                wav = result.audio.detach().cpu().float()
                rec.update({"s_span": [s0, s1], "t_span": [u0, u1]})
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
               "span_extension": a.span_extension,
               "version": "2026-11-15 (pre-norm layer output, masked padding)",
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
