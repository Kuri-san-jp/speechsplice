#!/usr/bin/env python3
"""Irodori の層別掃引用: **ModernBERT の層 L で差し替える**部品。

## 既存との違い

`irodori_splice.py:517 ConditionSplicer` は `model.encode_conditions` の
**戻り値**（ModernBERT 25 層 -> 射影 768->512 -> RMSNorm を全部通った後の
`text_state`）を差し替える。つまり**差し替えた状態は backbone を 1 層も
通らない。**

層別掃引では ModernBERT の層 L の出力で差し替え、L+1 以降と射影・RMSNorm を
通してから `text_state` にする。

    ModernBERT 層 L -> 差し替え -> L+1..24 -> 射影 -> RMSNorm -> text_state -> DiT

## 落とし穴: backbone は 2 回呼ばれる

`encode_conditions` は **text 用と caption 用**で backbone を呼ぶ
（`irodori_splice.py:22-23` に記録がある）。層に素朴に hook を掛けると
caption 側も差し替えてしまう。

そこで **`encode_conditions` の呼び出しをラップして、その中の「最初の
backbone 呼び出し」だけを対象にする。**`n_backbone_calls` を数えて、
2 回目以降は素通しする。text が先に呼ばれることは
`irodori_tts/model.py` の実装順で決まっており、実行時に
`last_text_shape` と突き合わせて確認する。

## 長さについて

`text_state` は常に `(B, max_text_len, 512)` の固定長で、有効長は
`text_mask` が決める（同 8-9 行）。**ModernBERT の中でも入力は固定長**
なので、層 L で長さを変えるとこの前提が崩れる。
**長さが変わる差し替えは扱わない**（呼び出し側が mask を作り直す既存の
経路と条件が変わるため）。長さが同じ対だけを対象にする。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from layer_hook import LayerSplicer  # noqa: E402

LAYERS_PATH = "pretrained_text_backbone.backbone.layers"


def find_backbone_layers(model):
    cur = model
    for part in LAYERS_PATH.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                "%s に %r が無い（path=%r）。Irodori の ModernBERT の層の"
                "場所を実物で確認すること" % (type(cur).__name__, part, LAYERS_PATH))
        cur = getattr(cur, part)
    return cur


class LayerConditionSplicer:
    """`encode_conditions` をラップし、**その中の最初の backbone 呼び出しの
    層 L だけ**を差し替える。

    armed でないときは完全に素通し。
    """

    def __init__(self, model, layer_index):
        self.model = model
        self.layers = find_backbone_layers(model)
        if not (0 <= layer_index < len(self.layers)):
            raise ValueError("layer_index %d が範囲外 (0..%d)"
                             % (layer_index, len(self.layers) - 1))
        self.layer_index = layer_index
        self._had_instance_attr = "encode_conditions" in vars(model)
        self._original = model.encode_conditions
        self._repl = None
        self._span = None
        self.n_calls = 0            # encode_conditions が呼ばれた回数
        self.n_backbone_calls = 0   # 層 L が呼ばれた回数（text + caption）
        self.n_spliced = 0
        self._handle = None
        model.encode_conditions = self._wrapped

    # ---- 差し替える中身 ----
    def arm(self, replacement, span):
        r = replacement
        if r.dim() == 3:
            if r.shape[0] != 1:
                raise ValueError("batch は 1 だけ: %s" % (tuple(r.shape),))
            r = r[0]
        self._repl = r
        self._span = tuple(span)
        return self

    def disarm(self):
        self._repl = None
        self._span = None
        return self

    def restore(self):
        """`encode_conditions` を元に戻す。**必ず呼ぶこと。**"""
        if self._had_instance_attr:
            self.model.encode_conditions = self._original
        else:
            try:
                del self.model.encode_conditions
            except AttributeError:
                self.model.encode_conditions = self._original
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False

    # ---- 本体 ----
    def _layer_hook(self, module, args, output):
        self.n_backbone_calls += 1
        # **最初の呼び出し（= text 用）だけ差し替える。**
        # 2 回目以降は caption 用なので素通し。
        if self.n_backbone_calls != 1 or self._repl is None:
            return output

        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if h.dim() != 3:
            raise ValueError("(B, T, d) を期待: %s" % (tuple(h.shape),))
        s0, s1 = self._span
        if not (0 <= s0 <= s1 <= h.shape[1]):
            raise ValueError("区間 (%d, %d) が長さ %d に収まらない"
                             % (s0, s1, h.shape[1]))
        repl = self._repl.to(dtype=h.dtype, device=h.device)
        if repl.shape[0] != (s1 - s0):
            raise ValueError(
                "長さが変わる差し替え (%d -> %d)。**Irodori の text_state は"
                "固定長 (B, max_text_len, 512) で、有効長は mask が決める。"
                "層の中で長さを変えるとこの前提が崩れる。**"
                "長さの同じ対だけを対象にすること" % (s1 - s0, repl.shape[0]))
        if repl.shape[-1] != h.shape[-1]:
            raise ValueError("次元が違う: %d vs %d" % (repl.shape[-1], h.shape[-1]))

        new_h = h.clone()
        new_h[:, s0:s1] = repl.unsqueeze(0)
        self.n_spliced += 1
        if is_tuple:
            return (new_h,) + tuple(output[1:])
        return new_h

    def _wrapped(self, *args, **kwargs):
        self.n_calls += 1
        self.n_backbone_calls = 0     # **呼び出しごとに数え直す**
        hd = self.layers[self.layer_index].register_forward_hook(self._layer_hook)
        try:
            return self._original(*args, **kwargs)
        finally:
            hd.remove()


def capture_layer_output(model, layer_index, call_fn):
    """`call_fn()` の中で、層 `layer_index` の**最初の**出力を捕まえる。

    text 用の呼び出しが最初であることを前提にする（caption 用は 2 回目）。
    """
    layers = find_backbone_layers(model)
    got = {}
    n = {"k": 0}

    def _h(mod, args_, out):
        n["k"] += 1
        if n["k"] == 1:
            got["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    hd = layers[layer_index].register_forward_hook(_h)
    try:
        call_fn()
    finally:
        hd.remove()
    if "h" not in got:
        raise RuntimeError("層 %d の出力が取れなかった" % layer_index)
    return got["h"], n["k"]
