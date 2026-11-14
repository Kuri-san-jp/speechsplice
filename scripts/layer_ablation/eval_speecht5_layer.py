#!/usr/bin/env python3
"""SpeechT5 の層別掃引（GPU 1 枚）。

## なぜ要るか

層別掃引は T5Gemma と Qwen3 でしか走っていない。**符号化復号器は T5Gemma
だけ**なので、5eZ1 に「提案手法固有では」と突かれる余地がある
（2026-09-19 ユーザ判断: 符号化復号器 4 手法すべてに広げる）。

既存の `eval_speecht5_crossculture_fixed_v2.py:135` は encoder **全体**に
forward hook を掛けて `last_hidden_state` を置換するので、**最終出力しか
差し替えられない。**中間層に入れるには層ごとの hook が要る。

## この版がやること

`speecht5.encoder.wrapped_encoder.layers[i]` の出力に hook を掛けて差し替える。
差し替えた状態は i+1 層目以降を通って最終出力になる。

**層番号は「通過層数」で指定する**（2026-09-19 統一基準、`_layer_numbers.json`）。

    --inject_layer 12   12 層通過後 = 最終出力 = 公表セルと同じ
    --inject_layer 1    1 層通過後（layers[0] の出力）
    --inject_layer 6    6 層通過後（layers[5] の出力）

SpeechT5 は L=12、公表は l*=12。掃引は **1 / 6 / 12**。

## 長さについて

既存実装は、元と目標のトークン数が違うとき `adaptive_avg_pool1d` で
**元の長さに合わせて**差し替える（同 130-133 行）。
つまり **SpeechSplice の長さ変更を行わない。**そのため中間層でも
mask と位置符号が保たれ、長さの問題が起きない。
**この版も同じ規約に従う。**公表セルと条件を揃えるため。

## 最終層での一致

`--inject_layer 12`（= 最終出力）は、既存スクリプトと**同じ音声**になる
はずである。実装が正しいかの検査になるので、本番前に n=5 で md5 を
突き合わせること（`--compare_dir` を渡す）。
"""
import argparse
import hashlib
import json
import os
import sys

_hf = "/mnt/.cache/huggingface" if os.path.isdir("/mnt/scripts") else \
    os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")

# **スレッド上限。torch より前に置く。**GPU 3 枚で 128 の 1/4 に収めるため 10。
_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else \
    os.environ.get("REPO_ROOT", "/mnt")
for _p in (os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "scripts", "layer_ablation"),
           os.path.join(PROJECT_ROOT, "data")):
    sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

from layer_hook import LayerSplicer, find_layers  # noqa: E402

SEED = 42
LAYERS_PATH = "speecht5.encoder.wrapped_encoder.layers"


def seed_all(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="en", choices=["en"],
                   help="SpeechT5 は英語のみ（JA/ZH は BL/INJ/TGT が同一に"
                        "なる退化セルなので Table 3 から外してある）")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--inject_layer", type=int, required=True,
                   help="**通過層数**で指定（統一基準）。12 = 最終出力")
    p.add_argument("--s5_xvector_idx", type=int, default=7306)
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--link_bl_tgt_from", default="",
                   help="BL/TGT の持ち込み元。今日の環境で作った置き場を"
                        "指すこと（本体セルは 5 月生成で版が違う）")
    p.add_argument("--compare_dir", default=None,
                   help="既存の単一 entity 本番と md5 を突き合わせる"
                        "（--inject_layer が最終のときだけ意味がある）")
    a = p.parse_args()

    os.makedirs(a.wav_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Info] device=%s threads=%s" % (device, _thr), flush=True)

    from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                              SpeechT5Processor)
    from datasets import load_dataset

    seed_all(SEED)
    proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan").to(device).eval()
    emb_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker = torch.tensor(emb_ds[a.s5_xvector_idx]["xvector"]).unsqueeze(0).to(device)
    print("[Info] speaker = cmu-arctic-xvectors[%d] sum=%.10f"
          % (a.s5_xvector_idx, float(speaker.sum())), flush=True)

    layers = find_layers(model, LAYERS_PATH)
    n_layers = len(layers)
    print("[Info] %s: %d 層" % (LAYERS_PATH, n_layers), flush=True)
    if not (1 <= a.inject_layer <= n_layers):
        raise SystemExit("--inject_layer は 1..%d（通過層数）" % n_layers)
    hook_index = a.inject_layer - 1
    is_final = a.inject_layer == n_layers
    print("[Info] 通過層数 %d -> layers[%d] に hook%s"
          % (a.inject_layer, hook_index, "（最終出力）" if is_final else ""),
          flush=True)

    def get_enc(text):
        inputs = proc(text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            enc = model.speecht5.encoder(
                input_values=inputs["input_ids"],
                attention_mask=inputs["attention_mask"])
        return enc.last_hidden_state, inputs["input_ids"]

    def get_layer_out(text, idx):
        """layers[idx] の出力を取る。**hook で拾う**（返り値に無いため）。"""
        got = {}

        def _h(mod, args_, out):
            got["h"] = (out[0] if isinstance(out, tuple) else out).detach()
        hd = layers[idx].register_forward_hook(_h)
        try:
            inputs = proc(text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                model.speecht5.encoder(
                    input_values=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"])
        finally:
            hd.remove()
        return got["h"], inputs["input_ids"]

    def find_ent(entity, ids):
        ent = proc.tokenizer(entity, add_special_tokens=False)["input_ids"]
        idl = ids[0].cpu().tolist()
        for i in range(len(idl) - len(ent) + 1):
            if idl[i:i + len(ent)] == ent:
                return i, i + len(ent)
        ent_sp = proc.tokenizer(" " + entity, add_special_tokens=False)["input_ids"]
        for i in range(len(idl) - len(ent_sp) + 1):
            if idl[i:i + len(ent_sp)] == ent_sp:
                return i, i + len(ent_sp)
        return None

    def synth_plain(text):
        """既存 eval_speecht5_crossculture_fixed_v2.py:108-113 と同じ並び。

        `seed_all` -> processor -> generate_speech。processor は乱数を引かない
        ので、この順で既存と一致する。
        """
        seed_all(SEED)
        with torch.no_grad():
            inputs = proc(text=text, return_tensors="pt").to(device)
            if inputs["input_ids"].shape[1] == 0:
                return None
            return model.generate_speech(inputs["input_ids"], speaker,
                                         vocoder=vocoder)

    def synth_inj(text_a, text_b, word_a, word_b):
        """層 `hook_index` の出力で差し替えて合成する。

        **長さは元に合わせる**（既存実装 130-133 行と同じ規約）。
        SpeechSplice の長さ変更を行わないので、中間層でも mask が保たれる。
        """
        # **準備をすべて済ませてから seed を戻す。**
        # SpeechT5 の `SpeechT5SpeechDecoderPrenet._consistent_dropout` は
        # eval でも torch.bernoulli を引く（modeling_speecht5.py:671-686）ので、
        # seed を戻した後に符号化を挟むと乱数の状態がずれて別の音声になる。
        # 既存実装（eval_speecht5_crossculture_fixed_v2.py:134-138）は
        # `seed_all(SEED)` -> hook 登録 -> `generate_speech` の間に
        # 乱数を引く処理を挟んでいない。**同じ並びにする。**
        # 2026-09-19、ここを守らずに一致検査が 0/5 で落ちた
        # （長さが 1.76s 対 2.34s と変わった）。
        h_a, ids_a = get_layer_out(text_a, hook_index)
        h_b, _ = get_layer_out(text_b, hook_index)
        _, ids_a_full = get_enc(text_a)
        _, ids_b_full = get_enc(text_b)
        sr = find_ent(word_a, ids_a_full)
        rr = find_ent(word_b, ids_b_full)
        if sr is None or rr is None:
            return None, None
        ss, se = sr
        rs, re_ = rr
        src_len, tgt_len = se - ss, re_ - rs
        ent = h_b[0, rs:re_]
        if src_len != tgt_len:
            x = ent.unsqueeze(0).transpose(1, 2).float()
            ent = F.adaptive_avg_pool1d(x, src_len).transpose(1, 2)[0].to(h_b.dtype)

        sp = LayerSplicer(layers, hook_index).attach()
        sp.arm(ent, (ss, se))
        try:
            seed_all(SEED)          # **ここから generate_speech まで乱数を引かない**
            with torch.no_grad():
                audio = model.generate_speech(ids_a, speaker, vocoder=vocoder)
        finally:
            sp.disarm().detach()
        if sp.n_spliced == 0:
            raise RuntimeError("差し替えが 1 度も起きていない（hook が効いていない）")
        return audio, {"s_span": [ss, se], "t_span": [rs, re_],
                       "n_spliced": sp.n_spliced, "n_calls": sp.n_calls}

    from en_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()[:a.n]
    print("[Info] %d 対" % len(pairs), flush=True)

    rows = []
    ckpt = os.path.join(a.out_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
        print("[Info] resume: %d 行" % len(rows), flush=True)
    done = {r["pair_idx"] for r in rows}

    import time
    t0 = time.time()
    for i, pr in enumerate(pairs):
        if i in done:
            continue
        rec = {"pair_idx": i, "word_a": pr["word_a"], "word_b": pr["word_b"]}
        for cond, text, word in (("bl", pr["text_a"], None),
                                 ("tgt", pr["text_b"], None)):
            wav = synth_plain(text)
            path = os.path.join(a.wav_dir, "s%04d_%s.wav" % (i, cond))
            sf.write(path, wav.cpu().numpy(), 16000)
        out = synth_inj(pr["text_a"], pr["text_b"], pr["word_a"], pr["word_b"])
        if out[0] is None:
            rec["skip"] = "span_not_found"
            rows.append(rec)
            continue
        wav, meta = out
        path = os.path.join(a.wav_dir, "s%04d_inj.wav" % i)
        sf.write(path, wav.cpu().numpy(), 16000)
        rec.update(meta)
        rec["md5_inj"] = md5_file(path)
        rows.append(rec)
        if (i + 1) % 10 == 0:
            json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)
            print("  [%d/%d] %.1fmin" % (len(rows), len(pairs),
                                         (time.time() - t0) / 60), flush=True)

    json.dump({"rows": rows}, open(ckpt, "w"), ensure_ascii=False)

    # **BL / TGT を持ち込む。**層を変えても基線と目標は変わらないので作り直さない。
    # ただし SpeechT5 の本体セルは 5 月生成で transformers / torch の版が違い、
    # 同じ文でも波形が一致しない。持ち込み元は**今日の環境で作り直した置き場**にする。
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

    summary = {"model": "SpeechT5", "lang": a.lang,
               "link_bl_tgt_from": a.link_bl_tgt_from,
               "n_linked": n_link, "n_link_missing": n_link_miss,
               "inject_layer_passed": a.inject_layer,
               "hook_index": hook_index, "n_layers": n_layers,
               "is_final": is_final, "seed": SEED,
               "xvector_idx": a.s5_xvector_idx,
               "length_policy": "adaptive_avg_pool1d to source length "
                                "(既存 eval_speecht5_crossculture_fixed_v2.py:130-133 と同じ)",
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
        if is_final and n_cmp and n_match != n_cmp:
            print("**最終層なのに一致しない。実装が違う可能性がある。**")

    out = os.path.join(a.out_dir, "results.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print("\n[Done] %s" % out, flush=True)
    for k, v in summary.items():
        print("  %-24s %s" % (k, v))


if __name__ == "__main__":
    main()
