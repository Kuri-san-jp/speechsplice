#!/usr/bin/env python3
"""LongCat の層別掃引用: **層 L で差し替えてから残りの層を通す**部品。

## 既存との違い

`longcat_splice.py:433 encode_text_parts` は umt5 の最終出力から
`H = LN(h_L) + LN(h_0)` を組み、`build_spliced_embedding` が **H の段階で**
差し替える。つまり**差し替えた状態は 1 層も通らない。**

層別掃引では、層 L の出力で差し替えて **L+1 以降を通したうえで** H を組む。

    層 L の出力 h_L -> 差し替え -> L+1, ..., 最終 -> LN(h_final) + LN(h_0) = H

**h_0（語埋め込み）は差し替えない。**既存の `splice_part="both"` は
H（= LN(h_L)+LN(h_0)）ごと差し替えるので h_0 も入れ替わるが、
層別では「層 L の状態を差し替えて下流の計算をやり直す」ことを見たいので、
h_0 は元のまま残す。**L = 最終層のときに既存と一致しないのはこのため。**
一致検査は「h_0 も差し替えた版」で行う（`match_final=True`）。

## 長さについて

**umt5 は相対位置埋め込みなので、系列長が変わっても位置符号は破綻しない。**
ただし attention_mask は長さを合わせる必要がある。長さが変わる差し替えでは
mask を作り直して渡す。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from layer_hook import LayerSplicer  # noqa: E402

# umt5 の層の並び。encoder.block[i] が i 層目。
LAYERS_PATH = "text_encoder.encoder.block"


def find_text_layers(model):
    cur = model
    for part in LAYERS_PATH.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                "%s に %r が無い（path=%r）。LongCat の umt5 の層の場所を"
                "実物で確認すること" % (type(cur).__name__, part, LAYERS_PATH))
        cur = getattr(cur, part)
    return cur


def encode_layer_out(model, input_ids, attention_mask, layer_index):
    """layers[layer_index] の出力を取る（hook で拾う）。"""
    layers = find_text_layers(model)
    got = {}

    def _h(mod, args_, out):
        got["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    hd = layers[layer_index].register_forward_hook(_h)
    try:
        with torch.no_grad():
            model.text_encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
    finally:
        hd.remove()
    if "h" not in got:
        raise RuntimeError("層 %d の出力が取れなかった" % layer_index)
    return got["h"]


def encode_with_splice(model, input_ids, attention_mask, layer_index,
                       replacement, span, h0_source=None,
                       allow_length_change=False, new_attention_mask=None):
    """層 `layer_index` の出力を差し替えて最後まで通し、H を組んで返す。

    h0_source : LN(h_0) に使う hidden_states[0]。None なら自分の h_0。
                **一致検査のときだけ目標側の h_0 を渡す**（既存の
                splice_part="both" と条件を揃えるため）。
    """
    layers = find_text_layers(model)
    if not (0 <= layer_index < len(layers)):
        raise ValueError("layer_index %d が範囲外 (0..%d)"
                         % (layer_index, len(layers) - 1))

    sp = LayerSplicer(layers, layer_index,
                      allow_length_change=allow_length_change).attach()
    sp.arm(replacement, span)
    try:
        with torch.no_grad():
            out = model.text_encoder(
                input_ids=input_ids,
                attention_mask=(new_attention_mask if new_attention_mask
                                is not None else attention_mask),
                output_hidden_states=True)
    finally:
        sp.disarm().detach()
    if sp.n_spliced == 0:
        raise RuntimeError("差し替えが起きていない（hook が効いていない）")

    cfg = model.config
    if not cfg.text_norm_feat or not cfg.text_add_embed:
        raise RuntimeError("text_norm_feat=True かつ text_add_embed=True 前提")
    d = model.text_encoder.config.d_model
    ln_hL = F.layer_norm(out.last_hidden_state, (d,), eps=1e-6)
    h0 = h0_source if h0_source is not None else out.hidden_states[0]
    ln_h0 = F.layer_norm(h0, (d,), eps=1e-6)
    if ln_h0.shape[1] != ln_hL.shape[1]:
        raise ValueError("h_0 と h_L の長さが違う: %d vs %d"
                         % (ln_h0.shape[1], ln_hL.shape[1]))
    return {"H": (ln_hL + ln_h0).float(),
            "ln_hL": ln_hL.float(), "ln_h0": ln_h0.float(),
            "n_spliced": sp.n_spliced, "n_calls": sp.n_calls}


# ---------------------------------------------------------------------------
# 2026-09-20: **層の途中で差し替える版を作り直した。**
#
# ## なぜ作り直したか
#
# 上の `encode_with_splice` は forward hook で層の出力を差し替え、
# その 1 回の `text_encoder(...)` 呼び出しの中で最後まで通していた。
# **語の長さが変わると必ず落ちる。**
#
#   RuntimeError: The size of tensor a (10) must match the size of tensor b (9)
#
# 呼び出しの入口で決まる注意マスクは元の長さ（9）のまま、
# hook が返す状態だけが新しい長さ（10）になるため、後続の層で食い違う。
# 「John」-> 「Yamada」のように語の刻み数が変わる対は普通にあるので、
# 本番の 300 対はほぼ全滅する。
#
# ## 作り直した中身
#
# 呼び出しの途中に割り込むのをやめ、**自分で 2 段に分けて回す。**
#
#   1. 出所側と目標側をそれぞれ素通しし、通過層数 l の状態を取り出す
#   2. その段で継ぎ合わせる    h = [h_src[:s0], h_tgt[u0:u1], h_src[s1:]]
#   3. **残りの層を新しい長さで回し直す**（マスクは作り直す）
#
# **umt5 は層ごとに相対位置バイアスを持つ**ので（modeling_umt5.py:361 で
# 各層が has_relative_attention_bias=True）、途中の層から長さを変えて
# 再開しても位置の扱いは破綻しない。T5 のように 1 層目だけが
# バイアスを持つ造りなら、この再開はできなかった。
#
# ## h_0 の扱いを変えた
#
# 元の版は h_0 を出所側のまま残していたが、**長さが変われば足し算が成立しない**
# （LN(h_L) は新しい長さ、LN(h_0) は元の長さ）。
# 既存本番の `splice_part="both"` と同じく、h_0 も同じ位置で継ぐ。
# 層ごとに変わるのは「h_L をどの段で継いだか」だけになり、比較が素直になる。
#
# 層正規化は位置ごとに独立なので LN(継いだもの) = 継いだ LN。
# よって **通過層数 = 最終層のとき、これは既存本番と同じ式**になり、
# 一致検査がそのまま実装の検算になる（`--match_final` は要らない）。

def _text_stack(model):
    """umt5 の層の入れ物（UMT5Stack）を返す。`block` の親。"""
    cur = model
    for part in LAYERS_PATH.split(".")[:-1]:
        cur = getattr(cur, part)
    return cur


def encode_with_splice_resume(model, ids_src, am_src, ids_tgt, am_tgt,
                              layer_passed, s_span, t_span):
    """通過層数 `layer_passed` の段で継ぎ、残りの層を回し直して H を返す。

    layer_passed : 1 .. L。L なら層を 1 つも回さず、既存本番と同じ式になる。
    s_span       : (s0, s1) 出所側の語の区間
    t_span       : (u0, u1) 目標側の語の区間
    """
    stack = _text_stack(model)
    n_layers = len(stack.block)
    if not (1 <= layer_passed <= n_layers):
        raise ValueError("通過層数 %d が範囲外 (1..%d)" % (layer_passed, n_layers))
    # **マスクを None で回すので、詰め物が無いことを確かめる。**
    # 詰め物があると、残りの層で無視すべき位置を見てしまう。
    for nm, am in (("出所側", am_src), ("目標側", am_tgt)):
        if am is not None and int(am.sum().item()) != am.numel():
            raise ValueError("%sの注意マスクに詰め物がある。"
                             "この関数は 1 文ずつ・詰め物無しを前提にしている" % nm)

    with torch.no_grad():
        o_src = model.text_encoder(input_ids=ids_src, attention_mask=am_src,
                                   output_hidden_states=True)
        o_tgt = model.text_encoder(input_ids=ids_tgt, attention_mask=am_tgt,
                                   output_hidden_states=True)

    s0, s1 = s_span
    u0, u1 = t_span
    is_final = (layer_passed == n_layers)

    # **hidden_states の添字に注意。**modeling_umt5.py:786-790 を読むこと。
    #   hidden_states[0]   埋め込み
    #   hidden_states[i]   i-1 番目の層の出力（そのまま、正規化前）  i < L
    #   hidden_states[L]   **最後の層正規化を通した後**（= last_hidden_state）
    # よって最終層のときだけ扱いが違う。
    if is_final:
        h_src = o_src.last_hidden_state
        h_tgt = o_tgt.last_hidden_state
    else:
        h_src = o_src.hidden_states[layer_passed]
        h_tgt = o_tgt.hidden_states[layer_passed]

    h = torch.cat([h_src[:, :s0], h_tgt[:, u0:u1], h_src[:, s1:]], dim=1)

    n_run = 0
    with torch.no_grad():
        for blk in stack.block[layer_passed:]:
            # マスクは渡さない（詰め物が無いので全位置が見える）。
            # 位置バイアスは層が自分で新しい長さから作り直す。
            h = blk(h, attention_mask=None)[0]
            n_run += 1
        if not is_final:
            h = stack.final_layer_norm(h)
            h = stack.dropout(h)

    z_src, z_tgt = o_src.hidden_states[0], o_tgt.hidden_states[0]
    h0 = torch.cat([z_src[:, :s0], z_tgt[:, u0:u1], z_src[:, s1:]], dim=1)

    cfg = model.config
    if not cfg.text_norm_feat or not cfg.text_add_embed:
        raise RuntimeError("text_norm_feat=True かつ text_add_embed=True 前提")
    d = model.text_encoder.config.d_model
    ln_hL = F.layer_norm(h, (d,), eps=1e-6)
    ln_h0 = F.layer_norm(h0, (d,), eps=1e-6)
    if ln_h0.shape[1] != ln_hL.shape[1]:
        raise ValueError("h_0 と h_L の長さが違う: %d vs %d"
                         % (ln_h0.shape[1], ln_hL.shape[1]))
    return {"H": (ln_hL + ln_h0).float(),
            "ln_hL": ln_hL.float(), "ln_h0": ln_h0.float(),
            "layer_passed": int(layer_passed), "n_layers": int(n_layers),
            "is_final": bool(is_final), "n_layers_rerun": int(n_run),
            "T_src": int(h_src.shape[1]), "T_tgt": int(h_tgt.shape[1]),
            "T_new": int(h.shape[1])}
