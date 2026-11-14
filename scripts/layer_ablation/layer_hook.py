#!/usr/bin/env python3
"""テキスト符号化器の**中間層**で隠れ状態を差し替えるための共通部品。

## なぜ要るか

層別掃引は T5Gemma と Qwen3 でしか走っていない。**符号化復号器は T5Gemma
だけ**で、5eZ1 に「提案手法固有では」と突かれる余地がある
（2026-09-19 ユーザ判断: 4 手法すべてに広げる）。

既存の注入コードはどれも**符号化器の最終出力**だけを差し替える作りで、
中間層への入口が無い。

    SpeechT5  eval_speecht5_crossculture_fixed_v2.py:135 が encoder 全体に
              forward hook を掛け、last_hidden_state を置換する
    LongCat   longcat_splice.py:478 build_spliced_embedding が
              LN(h_L)+LN(h_0) の合成後の H だけを受け取る
    Irodori   irodori_splice.py:517 ConditionSplicer が encode_conditions の
              戻り値（ModernBERT 全体を通った後）を置換する
    Kiseki    kiseki_splice.py が encoder の戻り値を置換する

## この部品がやること

**層 L の出力に forward hook を掛け、そこで差し替える。**差し替えた状態は
L+1 以降の層を通って最終出力になる。T5Gemma 版
（`eval_en_crossculture_t5gemma_layer.py`）と同じ考え方。

    層 L の出力  h_L  ->  差し替え  ->  L+1, L+2, ... -> 最終出力 -> 復号器

**L が最終層のときは既存経路と一致しなければならない。**これは呼び出し側が
md5 で確認する（T5Gemma では 5/5 一致を確認済み）。

## 使い方

    hook = LayerSplicer(layers, layer_index)     # layers は nn.ModuleList
    hook.arm(replacement, s_span, t_span)        # 差し替える中身と区間
    out = model(...)                             # 普通に走らせる
    hook.disarm()

`layers` は手法ごとに違う場所にある。呼び出し側が渡す。

    SpeechT5  model.speecht5.encoder.wrapped_encoder.layers
    LongCat   model.text_encoder.encoder.block        (umt5)
    Irodori   model.text_encoder.layers               (ModernBERT)
    Kiseki    model.encoder.layers                    (Mamba2Seq2SeqEncoder)

## 長さが変わる差し替えについて

SpeechSplice は T -> T-m+n で長さが変わる。**中間層で長さを変えると、
その層より後ろの attention mask と位置符号が合わなくなる。**
そこで既定では**長さが変わらない対だけ**を扱い、変わる対は呼び出し側が
除外するか、mask を作り直して渡す。`allow_length_change=True` を明示した
ときだけ長さ変更を許し、そのとき mask の作り直しは呼び出し側の責任。
"""
import torch


class LayerSplicer:
    """層 L の出力に forward hook を掛け、armed の間だけ差し替える。

    armed でないときは完全に素通し。呼ばれた回数を数えて検証に使う
    （`n_calls` / `n_spliced`）。
    """

    def __init__(self, layers, layer_index, allow_length_change=False):
        """
        layers       : nn.ModuleList（符号化器の層の並び）
        layer_index  : 0-based。**len(layers)-1 が最終層。**
                       len(layers) を渡すと「最終層の出力＝符号化器の出力」
                       を意味するが、その場合このフックは使わず既存経路を
                       通すこと（呼び出し側で分岐する）
        """
        if not (0 <= layer_index < len(layers)):
            raise ValueError("layer_index %d が範囲外 (0..%d)"
                             % (layer_index, len(layers) - 1))
        self.layers = layers
        self.layer_index = layer_index
        self.allow_length_change = allow_length_change
        self._handle = None
        self._repl = None          # 差し替える中身 (n, d)
        self._span = None          # 差し替える区間 (s0, s1)
        self.n_calls = 0
        self.n_spliced = 0

    # ---- hook の着脱 ----
    def attach(self):
        if self._handle is not None:
            return self
        self._handle = self.layers[self.layer_index].register_forward_hook(
            self._hook)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return self

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    # ---- 差し替える中身の出し入れ ----
    def arm(self, replacement, span):
        """replacement: (n, d) か (1, n, d)。span: (s0, s1) で置換先の区間。"""
        r = replacement
        if r.dim() == 3:
            if r.shape[0] != 1:
                raise ValueError("batch は 1 だけ対応: %s" % (tuple(r.shape),))
            r = r[0]
        if r.dim() != 2:
            raise ValueError("replacement は (n, d) か (1, n, d): %s"
                             % (tuple(r.shape),))
        self._repl = r
        self._span = tuple(span)
        return self

    def disarm(self):
        self._repl = None
        self._span = None
        return self

    # ---- 本体 ----
    def _hook(self, module, args, output):
        self.n_calls += 1
        if self._repl is None:
            return output

        # 層の戻り値は素の tensor のことも tuple のこともある
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if not torch.is_tensor(h):
            raise TypeError("層の出力が tensor でない: %s" % type(h))
        if h.dim() != 3 or h.shape[0] != 1:
            raise ValueError("(1, T, d) を期待: %s" % (tuple(h.shape),))

        s0, s1 = self._span
        T = h.shape[1]
        if not (0 <= s0 <= s1 <= T):
            raise ValueError("区間 (%d, %d) が長さ %d に収まらない" % (s0, s1, T))

        repl = self._repl.to(dtype=h.dtype, device=h.device)
        if repl.shape[-1] != h.shape[-1]:
            raise ValueError("次元が違う: 置換 %d vs 出力 %d"
                             % (repl.shape[-1], h.shape[-1]))
        n_new = repl.shape[0]
        if n_new != (s1 - s0) and not self.allow_length_change:
            raise ValueError(
                "長さが変わる差し替え (%d -> %d)。**中間層で長さを変えると "
                "後続の mask と位置符号が合わない。**扱うなら "
                "allow_length_change=True にして、mask の作り直しを"
                "呼び出し側で行うこと" % (s1 - s0, n_new))

        new_h = torch.cat([h[:, :s0], repl.unsqueeze(0), h[:, s1:]], dim=1)
        self.n_spliced += 1
        if is_tuple:
            return (new_h,) + tuple(output[1:])
        return new_h


def find_layers(model, path):
    """"a.b.c" 形式で nn.ModuleList を取り出す。無ければ分かりやすく落ちる。"""
    cur = model
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                "%s に %r が無い（path=%r）。手法ごとに層の場所が違うので、"
                "実物を見て指定すること" % (type(cur).__name__, part, path))
        cur = getattr(cur, part)
    try:
        n = len(cur)
    except TypeError:
        raise TypeError("%r は並びでない: %s" % (path, type(cur).__name__))
    if n == 0:
        raise ValueError("%r が空" % path)
    return cur
