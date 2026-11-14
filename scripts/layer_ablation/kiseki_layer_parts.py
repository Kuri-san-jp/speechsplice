#!/usr/bin/env python3
"""Kiseki の層別掃引用: **層 l で継いでから残りの層を回す**部品。

## 既存との違い

`kiseki_splice.py` は符号化器の**最終出力**で差し替える。
層別ではその手前の層で継ぎ、残りの層を通してから復号器へ渡す。

## Mamba2 なので UMT5 より簡単

`modeling_mamba2_s2s.py:312-331` の符号化器はこれだけ:

    x = inputs_embeds
    for layer in self.layers:
        x = layer(x, enc_mask, pos)
    return BaseModelOutput(last_hidden_state=self.layer_norm(x))

* **注意機構が無い**（状態空間모델）ので、相対位置バイアスの作り直しが要らない。
* 層は `(x, enc_mask, pos)` を取る。**長さが変わるので両方作り直す。**
  - `enc_mask`: 詰め物が無ければ全 1。形は (B, 1, 1, S)。
  - `pos`: `arange(S)` を B 個に広げたもの。
* 最後に `layer_norm`（RMSNorm）が 1 回かかる。

## 層ごとの隠れ状態の取り方

`kiseki_splice.py:533-553` が既に hook で集めている（`collect_hidden=True`）。
そこと同じ並びにする:

    hidden_states[0]       埋め込み
    hidden_states[i]       i-1 番目の層の出力（i = 1..enc_layers）

`hidden_states[enc_layers]` は**最後の層の出力そのもの**で、
`layer_norm` は通っていない（UMT5 と違う点。あちらは最後だけ正規化後だった）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402


def encoder_of(model):
    """符号化器（Mamba2Seq2SeqEncoder）を返す。"""
    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "layers"):
        raise AttributeError(
            "model.encoder.layers が無い。Kiseki の符号化器の場所を実物で確認すること")
    return enc


def _mask_and_pos(enc, x):
    """長さ S の全 1 マスクと位置索引を作る。

    詰め物が無い前提（1 文ずつ・batch 1）。形は符号化器の forward
    （modeling_mamba2_s2s.py:317-326）と同じにする。
    """
    B, S, _ = x.shape
    dev = x.device
    mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev)
    pos = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
    return mask, pos


def encode_with_splice_resume(model, ids_src, ids_tgt, layer_passed,
                              s_span, t_span):
    """通過層数 `layer_passed` の段で継ぎ、残りの層を回して符号化器出力を返す。

    layer_passed : 1 .. enc_layers。enc_layers なら層を 1 つも回さず、
                   最後の RMSNorm だけを通す（= 既存の最終出力での差し替えと同じ式）。
    s_span       : (s0, s1) 出所側の語の区間
    t_span       : (u0, u1) 目標側の語の区間
    """
    enc = encoder_of(model)
    n_layers = len(enc.layers)
    if not (1 <= layer_passed <= n_layers):
        raise ValueError("通過層数 %d が範囲外 (1..%d)" % (layer_passed, n_layers))

    def run_to(ids, upto):
        """埋め込みから `upto` 層まで通した状態を返す（正規化はしない）。"""
        x = enc.embed_tokens(ids)
        mask, pos = _mask_and_pos(enc, x)
        for layer in enc.layers[:upto]:
            x = layer(x, mask, pos)
        return x

    with torch.inference_mode():
        h_src = run_to(ids_src, layer_passed)
        h_tgt = run_to(ids_tgt, layer_passed)

        s0, s1 = s_span
        u0, u1 = t_span
        h = torch.cat([h_src[:, :s0], h_tgt[:, u0:u1], h_src[:, s1:]], dim=1)

        # **残りの層は新しい長さで回す。**マスクと位置索引を作り直すのが要点。
        n_run = 0
        for layer in enc.layers[layer_passed:]:
            mask, pos = _mask_and_pos(enc, h)
            h = layer(h, mask, pos)
            n_run += 1
        out = enc.layer_norm(h)

    return {"enc_out": out,
            "layer_passed": int(layer_passed), "n_layers": int(n_layers),
            "is_final": bool(layer_passed == n_layers),
            "n_layers_rerun": int(n_run),
            "T_src": int(h_src.shape[1]), "T_tgt": int(h_tgt.shape[1]),
            "T_new": int(out.shape[1])}
