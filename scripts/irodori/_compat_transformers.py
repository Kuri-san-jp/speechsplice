#!/usr/bin/env python3
"""Irodori を動かすための transformers 版差の吸収（import するだけで効く）。

## なぜ要るか

`scripts/irodori/_repo/irodori_tts/model.py:686` が

    from transformers.initialization import no_init_weights

を要求する。**この場所に移ったのは transformers 4.57 より後**で、
実行環境（4.57.3）では旧い場所 `transformers.modeling_utils` にある。

失敗すると `model.py:689` が

    RuntimeError: transformers is required when text_encoder_type='pretrained'

を投げる。**メッセージが実態を隠している**（transformers は入っている。
入っていないのは `transformers.initialization` という名前だけ）。
2026-09-19 にこれで 2 回落とした。

`_repo` は third_party の写しなので**書き換えない。**
このモジュールを import した側で入口だけ用意する。

## 使い方

Irodori を import する前に 1 行:

    import _compat_transformers  # noqa: F401

`irodori_splice` より前であればよい。二重に import しても安全。
"""
import sys
import types

_FIXED = "_irodori_compat_applied"


def apply():
    """`transformers.initialization` が無ければ作る。あれば何もしない。"""
    if getattr(sys.modules.get(__name__), _FIXED, False):
        return False
    try:
        import transformers.initialization  # noqa: F401
        setattr(sys.modules[__name__], _FIXED, True)
        return False
    except ImportError:
        pass

    try:
        from transformers.modeling_utils import no_init_weights
    except ImportError as exc:
        raise RuntimeError(
            "transformers.initialization も modeling_utils.no_init_weights も"
            "見つからない。transformers の版を確認すること"
        ) from exc

    shim = types.ModuleType("transformers.initialization")
    shim.no_init_weights = no_init_weights
    sys.modules["transformers.initialization"] = shim
    setattr(sys.modules[__name__], _FIXED, True)
    print("[compat] transformers.initialization の入口を作った"
          "（no_init_weights は modeling_utils から）", flush=True)
    return True


# **トークナイザの別名は入れない（2026-09-20）。**
# 一度 `transformers.TokenizersBackend = PreTrainedTokenizerFast` を足したが、
# **本番は専用の像 `irodori-tts:latest` で動いていた**ことが分かったので取り消した。
#   本番の環境: transformers 5.17.0 / torch 2.10.0+cu128
#   既定の像  : transformers 4.57.3 / torch 2.6.0
# `TokenizersBackend` は transformers 5.x の名前で、4.57.3 には無い。
# 別名で「読める」ようにはできるが、**本番と同じ環境ではない**ので、
# 生成条件を揃えるには像を合わせる方が正しい。
# 出どころ: eval_results/irodori_cross_culture_ja/
#           irodori_ja_generation_manifest.json の meta.transformers / meta.torch

apply()
