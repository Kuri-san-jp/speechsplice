#!/usr/bin/env python3
"""SpeechSplice for Irodori-TTS v4.1-Small (Aratako, MIT).

凍結した Irodori の text 条件付け系列（``model.encode_conditions`` が返す
``text_state``）に対して、実体語スパンの hidden state を「目標の読みを書いた
テキスト」の 2 回目 forward pass の対応スパンで差し替える。重み更新は一切しない。

    H~ = (h^src_{1:s1-1},  h^tgt_{u1:un},  h^src_{sm+1:T})
    有効長: T -> T - m + n   （テンソル形状は常に (B, max_text_len, 512) のまま）

Irodori 固有の事情（すべて Aratako/Irodori-TTS のソースで確認済み）
-------------------------------------------------------------------
1. フック点は ``model.encode_conditions``
   ``irodori_tts/model.py:1768`` の ``TextToLatentRFDiT.encode_conditions`` が
   ``(text_state, text_mask, ref_state, ref_mask, caption_state, caption_mask)``
   を返す。``text_state`` は
       backbone(ModernBERT-Ja-310m, 25層, hidden 768)
       -> text_encoder(PretrainedConditionProjector, 768->512)
       -> text_norm(RMSNorm)
   を通った (B, max_text_len, 512)。DiT 12 層の JointAttention はこれを
   key/value としてのみ参照する（text は query にならず更新もされない）。
   backbone を直接フックすると text 用と caption 用で 2 回呼ばれてしまうので、
   ``encode_conditions`` をラップするのが最も確実（落とし穴 1 の回避）。

2. text_mask を同時に作り直さないと差し替えた区間が 0 になる
   ``PretrainedTextBackbone.forward``(model.py:774-780) と
   ``PretrainedConditionProjector.forward``(model.py:855-882) が、どちらも
   最後に ``* mask.unsqueeze(-1)`` を掛けてマスク外を **ゼロで潰す**。
   さらに ``text_mask`` はモデルの外（``inference_runtime.py:1205``）で
   ``tokenizer.batch_encode`` から作られ、DiT の cross-attention にも
   duration predictor にもそのまま渡る。splice で有効長が T -> T-m+n に
   変わったら mask も作り直さなければならない（落とし穴 2）。
   本モジュールは text_state と text_mask を **必ず対で** 差し替える。

3. 系列長は変わらない
   ``batch_encode(..., max_length=max_text_len)`` が常に 256 にパディングするので、
   splice で動くのは「有効長の境界」だけでテンソル形状は変わらない。
   T5Gemma 側（``scripts/projection/splice_clone.py``）や Dia 側
   （``scripts/dia/dia_splice.py``）で必要だった「長さを合わせたダミーテキスト」の
   全探索は不要。パディング位置は上流の慣習どおりゼロで埋める
   （RMSNorm は bias を持たないのでゼロはゼロのまま通る。model.py:64-76）。

4. 実体語スパンの決め方は既存 3 モデルと同一
   tokenizer は ModernBERT-Ja の subword（vocab 102400）。生の日本語をそのまま
   入れるだけで G2P も形態素解析も無い（``text_normalization.normalize_text`` の
   NFKC 等のみ。pyopenjtalk は使っていない）。

   スパン探索は ``speechsplice_release/src/injection.py:401``
   ``find_entity_token_range`` と **同じ計算** にしてある
   （``find_entity_token_range`` 本モジュール版）。3 段構え:

     1 段目: 文全体のトークン列の中から、実体語を単独でトークナイズした
             id 列を部分列として探す。見つかればその位置。
     2 段目: ``full_text.find(entity_text)`` で文字位置を取り、
             **その手前の文字列を単独でトークナイズした長さ** を開始位置、
             開始 + 実体語単独のトークン数を終了位置とする。
     3 段目: 大文字小文字を無視した文字検索（日本語では発火しない）。

   2 段目は **トークン境界に乗っているかを検証しない**。ずれたまま進む。
   窓を共有文脈側へ広げることもしない。日本語では ``田中さん`` のように
   実体語が後続の ``さん`` と融合して 1 トークンになる例があり、そこでは
   スパンが境界からずれる。既存 3 モデルの数値もこのずれを含んだ条件で
   出ているので、Irodori 側だけ窓を広げると比較が揃わない。

   ずれた件数は解釈に要るので、``aligned`` フラグ（そのスパンの
   トークンが実体語をちょうど綴っているか）を manifest に記録する。
   **記録するだけで挙動は変えない。**

5. seconds と encode_conditions の呼ばれる回数
   ``seconds`` を省略すると ``encode_conditions`` は 2 回走る
   （duration predictor 用と生成用。inference_runtime.py:1292 と rf.py:220）。
   ``seconds`` を明示すれば duration 分岐を飛ばして 1 回で済む（落とし穴 3）。
   本モジュールのパッチは「armed の間はどの呼び出しにも同一の spliced テンソルを
   返す」ので 2 回走っても結果は一貫する。呼ばれた回数は毎回 assert する。

6. 乱数
   ``sample_euler_rf_cfg`` は ``torch.Generator(device).manual_seed(seed)``
   （rf.py:11-16, 158）で初期ノイズを引く。グローバル RNG に依存しないので
   ``req.seed`` を揃えれば BL / TGT / INJ の初期ノイズは厳密に同一。
   念のためグローバル側も毎回リセットする（既存 eval スクリプトと同じ作法）。

7. 話者・caption 条件
   チェックポイントは ``use_speaker_condition=true`` / ``use_caption_condition=true``。
   既存 Table 3 の 3 モデルと同じゼロショットに揃えるため、参照音声なし
   (``no_ref=True`` -> ref_latent は零・ref_mask は全 False。
   inference_runtime.py:867-881) 、caption は空文字（caption_mask を zero_()。
   inference_runtime.py:1225-1227）で統一する。
   モデルカードのベンチマーク条件（FP32 / 40 RF steps / text CFG 3.0 /
   参照音声なし・caption なし）と一致する。

このモジュール自体は GPU を要求しない。純粋なスパン計算・splice 計算は
モデル無しで CPU からテストできるように分離してある
（``smoke_test_irodori_splice.py`` を参照）。本番生成は
``run_irodori_eval.py`` から呼ぶこと。
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")

SEED = 42
DEFAULT_MODEL_ID = "Aratako/Irodori-TTS-v4.1-Small"
DEFAULT_CODEC_ID = "Aratako/Semantic-DACVAE-Japanese-32dim"

# ``irodori_tts/config.py`` の既定と checkpoint metadata の両方が 256。
DEFAULT_MAX_TEXT_LEN = 256
# ``inference_runtime.py:1040``: RF は num_steps=None のとき 40。
DEFAULT_NUM_STEPS = 40


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
def seed_all(seed: int = SEED) -> None:
    """既存 eval スクリプト（eval_en_crossculture_t5gemma_fixed.py:45-47）と同一。

    Irodori のサンプラは専用 Generator を使うのでこれ自体は結果を左右しないが、
    上流の作法に合わせて BL / TGT / INJ の各 synthesize 直前に必ず呼ぶ。
    """
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------
# text normalization（runtime と厳密に同じ前処理）
# --------------------------------------------------------------------------
def normalize_for_runtime(text: str) -> str:
    """``inference_runtime.py:1087`` と同一: normalize_text(raw).strip()。

    スパンを **正規化後の文字列** の上で計算するために必須。生のテキストの
    文字位置を使うと NFKC 等でずれる。
    """
    from irodori_tts.text_normalization import normalize_text

    return normalize_text(text).strip()


# --------------------------------------------------------------------------
# span utilities（モデル不要・純粋関数。CPU でテストできる）
# --------------------------------------------------------------------------
def encode_ids(raw_tokenizer, text: str) -> list[int]:
    """``text_tokenizer.encode(text, add_special_tokens=False)`` と同じもの。

    ``speechsplice_release/src/injection.py:404-406`` は HuggingFace の
    ``PreTrainedTokenizer.encode(..., add_special_tokens=False)`` を呼ぶ。
    Irodori に同梱されているのは素の ``tokenizers.Tokenizer`` なので、
    そちらの ``encode(..., add_special_tokens=False).ids`` で同じものを取る。
    """
    return [int(i) for i in raw_tokenizer.encode(text, add_special_tokens=False).ids]


def find_entity_token_range(raw_tokenizer, full_text: str, entity_text: str,
                            offset: int = 0) -> Optional[tuple[int, int, str]]:
    """``injection.py:401`` ``find_entity_token_range`` と同一の計算。

    既存 3 モデル（T5Gemma 系）と同じスパンを使うためのもの。
    戻り値は ``(start, end, stage)``。``start``/``end`` は
    **special token を含まないトークン座標** に ``offset`` を足したもの。
    見つからなければ ``None``（呼び出し側でエラーにする）。

    3 段構え。日本語で主に効くのは 2 段目（フォールバック 1）:

      1. 文全体のトークン id 列の中から実体語単独のトークン id 列を部分列検索
      2. ``full_text.find(entity_text)`` の手前を単独でトークナイズした長さを
         開始位置にする。**トークン境界に乗っているかは検証しない。**
      3. 大文字小文字を無視した文字検索（日本語では発火しない）

    **窓は広げない。** ずれたまま進むのが既存 3 モデルの挙動。
    """
    full_tokens = encode_ids(raw_tokenizer, full_text)
    entity_tokens = encode_ids(raw_tokenizer, entity_text)

    # 1 段目: トークン列の部分列検索
    for start in range(len(full_tokens) - len(entity_tokens) + 1):
        if full_tokens[start:start + len(entity_tokens)] == entity_tokens:
            return (start + offset, start + len(entity_tokens) + offset, "subsequence")

    # フォールバック 1: 文字位置から prefix をトークナイズして開始位置を出す
    entity_pos = full_text.find(entity_text)
    if entity_pos >= 0:
        prefix_tokens = encode_ids(raw_tokenizer, full_text[:entity_pos])
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(entity_tokens) + offset,
                "char_prefix")

    # フォールバック 2: 大文字小文字を無視した文字検索（injection.py:422-437）
    pos = full_text.lower().find(entity_text.lower())
    if pos >= 0:
        actual_surface = full_text[pos:pos + len(entity_text)]
        actual_tokens = encode_ids(raw_tokenizer, actual_surface)
        for start in range(len(full_tokens) - len(actual_tokens) + 1):
            if full_tokens[start:start + len(actual_tokens)] == actual_tokens:
                return (start + offset, start + len(actual_tokens) + offset,
                        "subsequence_ci")
        prefix_tokens = encode_ids(raw_tokenizer, full_text[:pos])
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(actual_tokens) + offset,
                "char_prefix_ci")

    return None


def describe_entity_span(raw_tokenizer, full_text: str, entity_text: str,
                         nospec_offsets, span: tuple[int, int],
                         stage: str) -> dict:
    """スパンが実際に何を覆っているかを記録する（**挙動は変えない**）。

    ``span`` は special token を含まないトークン座標。``nospec_offsets`` は
    同じ座標系の offset mapping。

    ``aligned`` = そのスパンのトークンが実体語を **ちょうど** 綴っているか。
    ``False`` なら実体語の境界とサブワード境界がずれている（``田中さん`` が
    1 トークンになる類）。ずれていても進む。
    """
    s, t = span
    out = {
        "stage": stage,
        "span": [int(s), int(t)],
        "n_tokens": int(t - s),
        "entity_text": entity_text,
        "entity_chars": None,
        "covered_chars": None,
        "covered_text": None,
        "aligned": False,
        "in_range": False,
    }
    pos = full_text.find(entity_text)
    if pos >= 0:
        out["entity_chars"] = [pos, pos + len(entity_text)]
    if not (0 <= s < t <= len(nospec_offsets)):
        return out
    out["in_range"] = True
    c0 = int(nospec_offsets[s][0])
    c1 = int(nospec_offsets[t - 1][1])
    out["covered_chars"] = [c0, c1]
    out["covered_text"] = full_text[c0:c1]
    out["aligned"] = bool(full_text[c0:c1] == entity_text)
    return out


# --------------------------------------------------------------------------
# 最小対称拡張への配線 (2026-09-18 ユーザ承認)
#
# 規則の実装は scripts/projection/span_extend.py の 1 箇所だけ。ここは配線で、
# **両側とも aligned なら従来の経路をそのまま通す**。拡張が起きるのは
# describe_entity_span の aligned が False だった対だけ (日本語の「さん」など、
# トークン化器が実体語と隣の機能語を 1 トークンに融合する対)。
#
# LongCat / Llasa との違いは「従来どう扱っていたか」だけ。あちらは往復一致
# しない対を **除外** していたので拡張は純粋な追加だったが、Irodori は
# **除外せずずれたまま注入** していたので、拡張が成立した対は値が変わる。
# それが正しい (note/2026-09-18_irodori_span_report.md)。
# aligned だった対は on/off でスパンが 1 ビットも変わらない。
# --------------------------------------------------------------------------
SPAN_EXTEND_DIR = os.path.join(PROJECT_ROOT, "scripts", "projection")


def _load_minimal_symmetric_surface():
    """規則の実装を遅延 import する (aligned でない対に当たったときだけ読む)."""
    if SPAN_EXTEND_DIR not in sys.path:
        sys.path.insert(0, SPAN_EXTEND_DIR)
    from span_extend import minimal_symmetric_surface

    return minimal_symmetric_surface


class _OffsetTokenizerAdapter:
    """``span_extend.covered_surface`` の呼び出し規約に橋を架けるだけのもの。

    あちらは HuggingFace の ``tokenizer(text, add_special_tokens=False,
    return_offsets_mapping=True)["offset_mapping"]`` を前提にしている。
    Irodori に同梱されているのは素の ``tokenizers.Tokenizer`` なので、
    ``encode(...).offsets`` を同じ形に詰め替える。**計算は 1 行も変えない。**
    """

    def __init__(self, raw_tokenizer):
        self.raw_tokenizer = raw_tokenizer

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        enc = self.raw_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": [int(i) for i in enc.ids],
            "offset_mapping": [(int(a), int(b)) for a, b in enc.offsets],
        }


def maybe_extend_spans(raw_tokenizer, norm_a: str, word_a: str, nospec_offsets_a,
                       s_found, s_report,
                       norm_b: str, word_b: str, nospec_offsets_b,
                       t_found, t_report,
                       span_extension: bool = False):
    """既に決まっているスパンを、必要なときだけ最小対称拡張で取り直す。

    入力の ``s_found`` / ``s_report`` は ``find_entity_token_range`` と
    ``describe_entity_span`` の結果をそのまま渡すこと。
    戻り値 ``(s_found, s_report, t_found, t_report, ext)``。
    ``ext`` は実際に使われた拡張
    ``{"extra", "pre", "post", "surface_a", "surface_b"}`` か ``None``。

      * 両側とも ``aligned`` が True
        -> 渡された値をそのまま返す。**従来と bit 単位で同一**
           (``span_extension`` の値に依らず、``minimal_symmetric_surface`` も
           import しない)。
      * どちらかが ``aligned`` でない かつ ``span_extension=True``
        -> 最小対称拡張を試し、得た表層を **Irodori の規則で**
           (``find_entity_token_range`` + ``describe_entity_span``) 引き直す。
           引き直したスパンが両側とも ``aligned`` にならなければ拡張は使わず、
           拡張前のスパンを返す (従来どおりずれたまま進む)。
      * ``span_extension=False`` -> 常に従来どおり。
    """
    if s_report["aligned"] and t_report["aligned"]:
        return s_found, s_report, t_found, t_report, None   # 従来の経路そのまま
    if not span_extension:
        return s_found, s_report, t_found, t_report, None

    adapter = _OffsetTokenizerAdapter(raw_tokenizer)
    r = _load_minimal_symmetric_surface()(adapter, norm_a, word_a, norm_b, word_b)
    if r is None or r["plain"]:
        # 候補が無い / 語そのままで足りる (= 拡張は不要) -> 従来どおり。
        return s_found, s_report, t_found, t_report, None

    sa, sb = r["surface_a"], r["surface_b"]
    s2 = find_entity_token_range(raw_tokenizer, norm_a, sa)
    t2 = find_entity_token_range(raw_tokenizer, norm_b, sb)
    if s2 is None or t2 is None:
        return s_found, s_report, t_found, t_report, None
    s2_report = describe_entity_span(raw_tokenizer, norm_a, sa,
                                     nospec_offsets_a, s2[:2], s2[2])
    t2_report = describe_entity_span(raw_tokenizer, norm_b, sb,
                                     nospec_offsets_b, t2[:2], t2[2])
    if not (s2_report["aligned"] and t2_report["aligned"]):
        return s_found, s_report, t_found, t_report, None

    ext = {"extra": r["extra"], "pre": int(r["pre"]), "post": int(r["post"]),
           "surface_a": sa, "surface_b": sb}
    # 拡張したときだけ鍵を足す（拡張しない対の記録は従来と同じ形のまま）。
    # entity_text は要求した表層に変わるので、元の実体語も別に残す。
    for rep, word in ((s2_report, word_a), (t2_report, word_b)):
        rep["entity_word"] = word
        rep["requested_surface"] = rep["entity_text"]
        rep["expand"] = [int(r["pre"]), int(r["post"])]
        rep["span_extension"] = dict(ext)
    return s2, s2_report, t2, t2_report, ext


def spliced_surface_text(text_src: str, word_src: str,
                         text_tgt: str, word_tgt: str) -> str:
    """duration predictor に渡す表層テキスト（``P + word_tgt + S``）。

    トークンスパンとは独立に **文字位置だけ** で作る。ja_cross_culture_names は
    同一テンプレートの実体語差し替えなので、これは ``text_tgt`` に一致する
    （一致しない入力が来たら ValueError）。``req.text`` は text 条件としては
    使われない（armed なパッチが spliced 条件を返す）が、
    ``build_duration_features``（文字数・かな/漢字比など）の入力になるので、
    BL / TGT / INJ の間で決め方を揃えておく。
    """
    i_src = text_src.find(word_src)
    i_tgt = text_tgt.find(word_tgt)
    if i_src < 0:
        raise ValueError(f"word {word_src!r} not in {text_src!r}")
    if i_tgt < 0:
        raise ValueError(f"word {word_tgt!r} not in {text_tgt!r}")
    prefix_src = text_src[:i_src]
    suffix_src = text_src[i_src + len(word_src):]
    prefix_tgt = text_tgt[:i_tgt]
    suffix_tgt = text_tgt[i_tgt + len(word_tgt):]
    if prefix_src != prefix_tgt or suffix_src != suffix_tgt:
        raise ValueError(
            f"source/target context differs: {prefix_src!r}|{suffix_src!r} vs "
            f"{prefix_tgt!r}|{suffix_tgt!r}")
    return prefix_src + word_tgt + suffix_src


def locate_subsequence(haystack: list[int], needle: list[int]) -> int:
    """haystack の中で needle が現れる唯一の開始位置を返す。

    ``tokenizer.batch_encode`` が返す id 列（パディング済み）と、offset mapping を
    取るために別途叩いた素の tokenizer の id 列を突き合わせるのに使う。
    これで「wrapper が BOS/CLS を何個足しているか」を実行時に確定できるので、
    特殊トークンの個数をハードコードしなくて済む。
    複数箇所に現れたら曖昧なので ValueError。
    """
    if not needle:
        raise ValueError("empty needle")
    n, m = len(haystack), len(needle)
    hits = [i for i in range(0, n - m + 1) if haystack[i:i + m] == needle]
    if not hits:
        raise ValueError(
            f"raw tokenizer ids are not a subsequence of batch_encode ids "
            f"(needle[:8]={needle[:8]} haystack[:16]={haystack[:16]})")
    if len(hits) > 1:
        raise ValueError(f"ambiguous alignment: needle occurs at {hits}")
    return hits[0]


# --------------------------------------------------------------------------
# splice（純粋テンソル演算。モデル不要）
# --------------------------------------------------------------------------
def mask_valid_length(mask_row) -> int:
    """mask が「先頭からの連続 True」であることを確認して有効長を返す。

    right padding を前提にしているので、そうでなければ即エラーにする
    （黙って別の解釈で進むのは禁止）。
    """
    import torch

    row = mask_row.to(dtype=torch.bool).reshape(-1)
    n = int(row.sum().item())
    if n <= 0:
        raise ValueError("text_mask has no valid position")
    if not bool(row[:n].all()) or bool(row[n:].any()):
        raise ValueError(
            f"text_mask is not a contiguous prefix (valid={n}, "
            f"row={row.to(dtype=torch.int8).tolist()[:32]}...)")
    return n


def splice_text_state(src_state, src_mask, s_span: tuple[int, int],
                      tgt_state, tgt_mask, t_span: tuple[int, int]):
    """H~ = (h^src[:s0], h^tgt[t0:t1], h^src[s1:T_src]) を作り、mask も作り直す。

    src_state / tgt_state: (1, L, D)（L = max_text_len。パディング込み）
    戻り値: (new_state (1, L, D), new_mask (1, L) bool, info dict)

    パディング位置は上流の慣習どおりゼロで埋める
    （PretrainedConditionProjector.forward が ``* mask`` で潰しているのと同じ状態）。
    """
    import torch

    if src_state.dim() != 3 or src_state.shape[0] != 1:
        raise ValueError(f"src_state must be (1, L, D), got {tuple(src_state.shape)}")
    if tgt_state.shape != src_state.shape:
        raise ValueError(
            f"src/tgt state shape mismatch: {tuple(src_state.shape)} vs "
            f"{tuple(tgt_state.shape)}")
    _, max_len, dim = src_state.shape

    t_src = mask_valid_length(src_mask[0])
    t_tgt = mask_valid_length(tgt_mask[0])
    s0, s1 = s_span
    t0, t1 = t_span
    if not 0 <= s0 < s1 <= t_src:
        raise ValueError(f"s_span={s_span} outside src valid length {t_src}")
    if not 0 <= t0 < t1 <= t_tgt:
        raise ValueError(f"t_span={t_span} outside tgt valid length {t_tgt}")

    m = s1 - s0
    n = t1 - t0
    t_out = t_src - m + n
    if t_out > max_len:
        raise ValueError(
            f"spliced length {t_out} exceeds max_text_len {max_len}; "
            f"raise max_text_len or shorten the text")

    body = torch.cat(
        [
            src_state[0, :s0],
            tgt_state[0, t0:t1].to(device=src_state.device, dtype=src_state.dtype),
            src_state[0, s1:t_src],
        ],
        dim=0,
    )
    if int(body.shape[0]) != t_out:
        raise AssertionError(f"spliced body length {int(body.shape[0])} != {t_out}")

    new_state = torch.zeros_like(src_state)
    new_state[0, :t_out] = body
    new_mask = torch.zeros_like(src_mask, dtype=torch.bool)
    new_mask[0, :t_out] = True

    info = {
        "n_tokens_src": t_src,
        "n_tokens_tgt": t_tgt,
        "n_tokens_spliced": t_out,
        "span_len_src": m,
        "span_len_tgt": n,
        "delta_tok": n - m,
        "max_text_len": int(max_len),
        "text_dim": int(dim),
    }
    return new_state, new_mask, info


def entity_cosine(src_h, tgt_h, s_span: tuple[int, int], t_span: tuple[int, int]) -> float:
    """kappa = cos( mean_j h^src_{s_j}, mean_k h^tgt_{u_k} )。

    既存実装（eval_layer_convergence_6lang.py:426-431、injection.py:508-512）と同式。
    スパン内 hidden state をトークン方向に単純平均したベクトル同士の cosine。
    """
    import torch

    a = src_h[0, s_span[0]:s_span[1]].float().mean(0)
    b = tgt_h[0, t_span[0]:t_span[1]].float().mean(0)
    return float(
        torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# --------------------------------------------------------------------------
# encode_conditions のラッパ
# --------------------------------------------------------------------------
class ConditionSplicer:
    """``model.encode_conditions`` を差し替え、armed の間 text 条件を置換する。

    - ``text_state`` と ``text_mask`` を **必ず対で** 差し替える
      （片方だけ直すと差し替えた区間がゼロで潰れる。落とし穴 2）
    - ``ref_state`` / ``caption_state`` は元の実装の戻り値をそのまま通す
      （テキストに依存しないので再計算の必要が無い）
    - armed でないときは完全に素通し。呼ばれた回数を数えて検証に使う
    """

    def __init__(self, model):
        self.model = model
        # torch.compile を使うと encode_conditions は **インスタンス属性** に
        # なる（_maybe_compile_inference_model, inference_runtime.py:272）。
        # 素の状態ではクラスのメソッドなので、restore の戻し方を変える必要がある。
        self._had_instance_attr = "encode_conditions" in vars(model)
        self._original = model.encode_conditions
        self.replacement = None            # (text_state, text_mask) or None
        self.n_calls = 0
        self.n_replaced = 0
        self.last_text_shape: Optional[tuple] = None
        model.encode_conditions = self._wrapped

    def _wrapped(self, *args, **kwargs):
        import torch

        out = self._original(*args, **kwargs)
        self.n_calls += 1
        text_state, text_mask = out[0], out[1]
        self.last_text_shape = tuple(text_state.shape)
        if self.replacement is None:
            return out

        new_state, new_mask = self.replacement
        if tuple(new_state.shape[1:]) != tuple(text_state.shape[1:]):
            raise AssertionError(
                f"spliced text_state shape {tuple(new_state.shape)} is incompatible "
                f"with the model's {tuple(text_state.shape)}")
        batch = int(text_state.shape[0])
        rep_state = new_state.to(device=text_state.device, dtype=text_state.dtype)
        rep_mask = new_mask.to(device=text_mask.device, dtype=text_mask.dtype)
        if int(rep_state.shape[0]) != batch:
            rep_state = rep_state.expand(batch, -1, -1).contiguous()
            rep_mask = rep_mask.expand(batch, -1).contiguous()
        # mask 外はゼロのまま渡す（上流の慣習に一致させる）。
        rep_state = rep_state * rep_mask.unsqueeze(-1).to(dtype=rep_state.dtype)
        self.n_replaced += 1
        return (rep_state, rep_mask) + tuple(out[2:])

    @contextlib.contextmanager
    def armed(self, text_state, text_mask):
        prev = self.replacement
        self.replacement = (text_state, text_mask)
        try:
            yield self
        finally:
            self.replacement = prev

    def restore(self) -> None:
        """パッチを外す。元がクラスのメソッドならインスタンス属性ごと消す。"""
        if self._had_instance_attr:
            self.model.encode_conditions = self._original
            return
        try:
            del self.model.encode_conditions
        except AttributeError:
            self.model.encode_conditions = self._original


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------
@dataclass
class IrodoriBundle:
    runtime: Any
    raw_tokenizer: Any                 # tokenizers.Tokenizer（offset mapping 用）
    splicer: ConditionSplicer
    max_text_len: int = DEFAULT_MAX_TEXT_LEN
    sample_rate: int = 44100
    hop_length: int = 512
    checkpoint_path: str = ""
    tokenizer_path: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def model(self):
        return self.runtime.model


def _find_tokenizer_json(checkpoint_path: str) -> Path:
    """checkpoint と同梱の tokenizer.json を探す。

    ``_resolve_tokenizer_source``(inference_runtime.py:581-591) と同じ探索順。
    """
    ckpt = Path(checkpoint_path)
    for base in (ckpt.parent / "tokenizer", ckpt.parent.parent / "tokenizer"):
        cand = base / "tokenizer.json"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"bundled tokenizer.json not found next to {checkpoint_path}")


def load_irodori(model_id: str = DEFAULT_MODEL_ID,
                 codec_id: str = DEFAULT_CODEC_ID,
                 device: Optional[str] = None,
                 precision: str = "fp32",
                 codec_device: Optional[str] = None) -> IrodoriBundle:
    """Irodori runtime を読み込み、encode_conditions にパッチを当てて返す。

    HF_HOME は呼び出し側（run_irodori_eval.py）が import 前に設定しておくこと。
    ``precision`` はモデルカードのベンチマーク条件に合わせて既定 fp32。
    """
    import torch
    from tokenizers import Tokenizer

    from irodori_tts.inference_runtime import (
        InferenceRuntime,
        RuntimeKey,
        download_hf_checkpoint,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if codec_device is None:
        codec_device = device

    checkpoint_path = download_hf_checkpoint(model_id)
    key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(device),
        codec_repo=codec_id,
        model_precision=precision,
        codec_device=str(codec_device),
        codec_precision=precision,
    )
    runtime = InferenceRuntime.from_key(key)

    if runtime.model.pretrained_text_backbone is None:
        raise RuntimeError(
            "this checkpoint does not use a pretrained text backbone; "
            "the splice hook assumes ModernBERT-Ja + PretrainedConditionProjector")

    tok_json = _find_tokenizer_json(checkpoint_path)
    raw_tokenizer = Tokenizer.from_file(str(tok_json))

    splicer = ConditionSplicer(runtime.model)
    return IrodoriBundle(
        runtime=runtime,
        raw_tokenizer=raw_tokenizer,
        splicer=splicer,
        max_text_len=int(runtime.default_text_max_len),
        sample_rate=int(runtime.codec.sample_rate),
        hop_length=int(runtime.codec.model.hop_length),
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tok_json),
        meta={
            "model_id": model_id,
            "codec_id": codec_id,
            "device": str(device),
            "precision": precision,
            "model_config": json.loads(json.dumps(
                {k: v for k, v in vars(runtime.model_cfg).items()
                 if isinstance(v, (int, float, str, bool, type(None)))})),
        },
    )


# --------------------------------------------------------------------------
# text 条件の計算
# --------------------------------------------------------------------------
def encode_text_only(bundle: IrodoriBundle, normalized_text: str,
                     output_hidden_states: bool = False) -> dict:
    """text 条件だけを 1 回 forward する（ref/caption は計算しない）。

    ``encode_conditions``(model.py:1825-1829) の text 経路そのもの:
        text_state = text_encoder(pretrained_text_backbone, ids, mask)
        text_state = text_norm(text_state)

    Returns dict: text_state (1, L, 512) / text_mask (1, L) / input_ids /
                  n_tokens / raw_ids / offsets / align_offset /
                  nospec_ids / nospec_offsets / entity_offset /
                  backbone_hidden_states（要求時のみ。ModernBERT の 26 テンソル）

    ``entity_offset`` は「special token を含まないトークン座標」を
    ``batch_encode`` 座標へ移すためのオフセット。``injection.py`` 側の
    ``offset = 1 if add_bos else 0`` に対応するが、特殊トークンの個数を
    ハードコードせず実行時に確定させる。
    """
    import torch

    runtime = bundle.runtime
    model = runtime.model
    device = runtime.model_device

    if bundle.splicer.replacement is not None:
        raise AssertionError("encode_text_only must run with the splicer disarmed")

    text_ids, text_mask = runtime.tokenizer.batch_encode(
        [normalized_text], max_length=bundle.max_text_len)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    # batch_encode は max_text_len まで右詰めでパディングするので有効長で切る。
    # さらに、素の tokenizer は BOS と EOS の両方を足すが batch_encode は
    # EOS を足さない。そのため特殊トークン抜きの列を基準に突き合わせる。
    n_tokens = mask_valid_length(text_mask[0])
    valid_ids = [int(x) for x in text_ids[0].tolist()[:n_tokens]]

    # special token 抜きの座標系（injection.py の find_entity_token_range が
    # 使うのはこちら）。何個の特殊トークンが前に付くかは実行時に確定させる。
    nospec = bundle.raw_tokenizer.encode(normalized_text, add_special_tokens=False)
    nospec_ids = [int(x) for x in nospec.ids]
    entity_offset = locate_subsequence(valid_ids, nospec_ids)
    if entity_offset + len(nospec_ids) > n_tokens:
        raise AssertionError(
            f"tokenizer disagreement: batch_encode valid length {n_tokens} but "
            f"body of {len(nospec_ids)} ids starts at offset {entity_offset}")

    with torch.inference_mode():
        text_state = model.text_encoder(
            model.pretrained_text_backbone, text_ids, text_mask)
        text_state = model.text_norm(text_state)

    backbone_hidden = None
    if output_hidden_states:
        with torch.inference_mode():
            out = model.pretrained_text_backbone.backbone(
                input_ids=text_ids,
                attention_mask=text_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            backbone_hidden = tuple(
                h * text_mask.unsqueeze(-1).to(dtype=h.dtype)
                for h in out.hidden_states
            )

    return {
        "text": normalized_text,
        "text_state": text_state,
        "text_mask": text_mask,
        "input_ids": text_ids,
        "n_tokens": n_tokens,
        "nospec_ids": nospec_ids,
        "nospec_offsets": list(nospec.offsets),
        "entity_offset": entity_offset,
        "backbone_hidden_states": backbone_hidden,
    }


def build_splice(bundle: IrodoriBundle, text_a: str, word_a: str,
                 text_b: str, word_b: str,
                 collect_kappa: bool = True,
                 span_extension: bool = False) -> dict:
    """text_a / text_b を 1 回ずつ encode し、spliced な text 条件を作る。

    スパンは ``find_entity_token_range``（= ``injection.py:401`` と同じ計算）で
    src / tgt それぞれ独立に決める。**窓は広げず、トークン境界の検証もしない。**
    境界に乗っていたかどうかは ``span_info`` に記録するだけ。

    ``span_extension=True`` のときだけ、両側の ``aligned`` が揃わない対に
    最小対称拡張（``scripts/projection/span_extend.py``）を試す。既定は
    ``False`` で従来と完全に同じ。両側 ``aligned`` の対は
    ``span_extension`` の値に関わらず従来と同じスパンを通る。

    生成は行わない（``splice_synthesize`` から呼ばれる）。GPU 無しでも
    モデルさえ読めれば動く。
    """
    norm_a = normalize_for_runtime(text_a)
    norm_b = normalize_for_runtime(text_b)

    src = encode_text_only(bundle, norm_a, output_hidden_states=collect_kappa)
    tgt = encode_text_only(bundle, norm_b, output_hidden_states=collect_kappa)

    s_found = find_entity_token_range(bundle.raw_tokenizer, norm_a, word_a)
    t_found = find_entity_token_range(bundle.raw_tokenizer, norm_b, word_b)
    if s_found is None:
        raise ValueError(f"entity {word_a!r} not found in {norm_a!r}")
    if t_found is None:
        raise ValueError(f"entity {word_b!r} not found in {norm_b!r}")

    s_report = describe_entity_span(bundle.raw_tokenizer, norm_a, word_a,
                                    src["nospec_offsets"], s_found[:2], s_found[2])
    t_report = describe_entity_span(bundle.raw_tokenizer, norm_b, word_b,
                                    tgt["nospec_offsets"], t_found[:2], t_found[2])

    # --span_extension のときだけ、両側の aligned が揃わない対を取り直す。
    # 両側 aligned なら渡した値がそのまま返る（従来の経路）。
    s_found, s_report, t_found, t_report, span_ext = maybe_extend_spans(
        bundle.raw_tokenizer, norm_a, word_a, src["nospec_offsets"], s_found, s_report,
        norm_b, word_b, tgt["nospec_offsets"], t_found, t_report,
        span_extension=span_extension)

    # special token 抜きの座標 -> batch_encode 座標へ移す。
    s_span = (s_found[0] + src["entity_offset"], s_found[1] + src["entity_offset"])
    t_span = (t_found[0] + tgt["entity_offset"], t_found[1] + tgt["entity_offset"])

    span_info = {
        "src": s_report,
        "tgt": t_report,
        "aligned": bool(s_report["aligned"] and t_report["aligned"]),
        "src_span_text": _decode_span(bundle, src["input_ids"][0], s_span),
        "tgt_span_text": _decode_span(bundle, tgt["input_ids"][0], t_span),
    }
    if span_ext is not None:
        span_info["span_extension"] = dict(span_ext)

    new_state, new_mask, info = splice_text_state(
        src["text_state"], src["text_mask"], s_span,
        tgt["text_state"], tgt["text_mask"], t_span)

    kappa_text_state = entity_cosine(
        src["text_state"], tgt["text_state"], s_span, t_span)
    kappa_by_layer = None
    if collect_kappa and src["backbone_hidden_states"] is not None:
        kappa_by_layer = [
            entity_cosine(sh, th, s_span, t_span)
            for sh, th in zip(src["backbone_hidden_states"],
                              tgt["backbone_hidden_states"])
        ]

    spliced_surface = spliced_surface_text(norm_a, word_a, norm_b, word_b)

    out = {
        "text_state": new_state,
        "text_mask": new_mask,
        "spliced_surface_text": spliced_surface,
        "normalized_text_a": norm_a,
        "normalized_text_b": norm_b,
        "span_info": span_info,
        "s_span": list(s_span),
        "t_span": list(t_span),
        "kappa_text_state": kappa_text_state,
        "kappa_by_layer": kappa_by_layer,
        "span_extension": span_ext,
    }
    out.update(info)
    return out


def _decode_span(bundle: IrodoriBundle, ids_row, span: tuple[int, int]) -> str:
    """batch_encode の id 列から span を切り出して decode する（記録用）。

    スパンがずれている（``aligned=False``）ときに **実際に何が差し替わったか**
    を manifest から読めるようにするためだけのもの。ここで一致検証はしない
    （既存 3 モデルと同じく、ずれたまま進むのが仕様）。
    """
    s, t = span
    ids = [int(x) for x in ids_row[s:t].tolist()]
    return bundle.raw_tokenizer.decode(ids, skip_special_tokens=False)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _base_request_kwargs(seconds: Optional[float], seed: int,
                         num_steps: Optional[int],
                         extra: Optional[dict] = None) -> dict:
    """BL / TGT / INJ で完全に共有する復号設定。

    既存 Table 3 の 3 モデルに揃えてゼロショット（参照音声なし・caption なし）。
    モデルカードのベンチマーク条件（40 RF steps / text CFG 3.0）と同じ。
    """
    kwargs = dict(
        caption=None,
        no_ref=True,
        num_candidates=1,
        decode_mode="sequential",
        seconds=seconds,
        num_steps=num_steps,
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0,
        cfg_scale_speaker=5.0,
        seed=seed,
    )
    if extra:
        kwargs.update({k: v for k, v in extra.items() if v is not None})
    return kwargs


_FRAMES_RE = re.compile(r"using_frames=(\d+)")


def predicted_latent_frames(result) -> Optional[int]:
    """runtime の messages から duration predictor が決めた latent frame 数を拾う。

    ``inference_runtime.py:1329-1337`` が
    ``info: predicted duration frames=..., scale=..., using_frames=N (X.XXXs).``
    を messages に積む。整数 N をそのまま読むので丸め誤差が入らない。
    """
    for msg in getattr(result, "messages", []) or []:
        m = _FRAMES_RE.search(str(msg))
        if m:
            return int(m.group(1))
    return None


def frames_to_seconds(bundle: IrodoriBundle, frames: int) -> float:
    """latent frame 数 -> seconds。

    ``seconds`` を明示したとき runtime は
    ``target_samples = max(1, int(seconds * sr)); latent_steps = ceil(target_samples / hop)``
    （inference_runtime.py:1265-1266）で戻すので、frames * hop / sr を渡せば
    ちょうど同じ frames に戻る。
    """
    return float(frames * bundle.hop_length) / float(bundle.sample_rate)


def synthesize(bundle: IrodoriBundle, text: str, seed: int = SEED,
               seconds: Optional[float] = None,
               num_steps: Optional[int] = None,
               extra: Optional[dict] = None) -> dict:
    """素の TTS（BL = TTS(text_a), TGT = TTS(text_b)）。splicer は disarm したまま。"""
    import torch

    from irodori_tts.inference_runtime import SamplingRequest

    if bundle.splicer.replacement is not None:
        raise AssertionError("synthesize must run with the splicer disarmed")

    req = SamplingRequest(
        text=text,
        **_base_request_kwargs(seconds, seed, num_steps, extra),
    )
    calls_before = bundle.splicer.n_calls
    repl_before = bundle.splicer.n_replaced
    seed_all(seed)
    with torch.inference_mode():
        result = bundle.runtime.synthesize(req)
    n_calls = bundle.splicer.n_calls - calls_before
    n_repl = bundle.splicer.n_replaced - repl_before
    if n_repl != 0:
        raise AssertionError(f"baseline synthesis was spliced {n_repl} times")

    audio = result.audio.detach().cpu().float()
    return {
        "audio": audio,
        "sampling_rate": int(result.sample_rate),
        "used_seed": int(result.used_seed),
        "n_encode_conditions_calls": n_calls,
        "latent_frames": predicted_latent_frames(result),
        "messages": list(result.messages),
        "duration_sec": float(audio.shape[-1]) / float(result.sample_rate),
    }


def splice_synthesize(bundle: IrodoriBundle, text_a: str, word_a: str,
                      text_b: str, word_b: str, seed: int = SEED,
                      seconds: Optional[float] = None,
                      num_steps: Optional[int] = None,
                      collect_kappa: bool = True,
                      extra: Optional[dict] = None,
                      span_extension: bool = False) -> dict:
    """INJ = SpeechSplice。

    手順
      1. text_a / text_b をそれぞれ 1 回 encode（splicer は disarm）
      2. src / tgt それぞれで ``find_entity_token_range``（既存 3 モデルと
         同じ計算）を走らせてスパンを決める。窓は広げない
         （``span_extension=True`` のときだけ、両側の ``aligned`` が揃わない
         対に最小対称拡張を試す。既定 False = 従来と完全に同じ）
      3. H~ を作り、text_mask も同時に作り直す（有効長 T -> T-m+n）
      4. ``req.text`` に **差し替え後の表層テキスト** を渡して synthesize。
         encode_conditions は armed なパッチで必ず spliced な条件を返すので、
         batch_encode が作る素の text_state は使われない。
         表層テキストを渡すのは duration predictor の補助特徴
         （``build_duration_features``。文字数・かな/漢字比など）を
         差し替え後の表層に合わせるため。

    ``seconds`` を明示すると encode_conditions は 1 回だけ走る。省略すると
    duration predictor 用と生成用で 2 回走るが、どちらも同じ spliced 条件を
    受け取るので結果は一貫する（呼ばれた回数は下で assert する）。
    """
    import torch

    from irodori_tts.inference_runtime import SamplingRequest

    info = build_splice(bundle, text_a, word_a, text_b, word_b,
                        span_extension=span_extension,
                        collect_kappa=collect_kappa)

    req = SamplingRequest(
        text=info["spliced_surface_text"],
        **_base_request_kwargs(seconds, seed, num_steps, extra),
    )
    calls_before = bundle.splicer.n_calls
    repl_before = bundle.splicer.n_replaced
    seed_all(seed)
    with torch.inference_mode(), bundle.splicer.armed(info["text_state"], info["text_mask"]):
        result = bundle.runtime.synthesize(req)
    n_calls = bundle.splicer.n_calls - calls_before
    n_repl = bundle.splicer.n_replaced - repl_before

    expected = 1 if seconds is not None else 2
    if n_calls != expected:
        raise AssertionError(
            f"expected {expected} encode_conditions call(s) during spliced "
            f"synthesis (seconds={seconds}), got {n_calls}")
    if n_repl != n_calls:
        raise AssertionError(
            f"{n_calls} encode_conditions call(s) but only {n_repl} were spliced")

    audio = result.audio.detach().cpu().float()
    out = {
        "audio": audio,
        "sampling_rate": int(result.sample_rate),
        "used_seed": int(result.used_seed),
        "n_encode_conditions_calls": n_calls,
        "n_spliced": n_repl,
        "latent_frames": predicted_latent_frames(result),
        "messages": list(result.messages),
        "duration_sec": float(audio.shape[-1]) / float(result.sample_rate),
    }
    for key in ("spliced_surface_text", "normalized_text_a", "normalized_text_b",
                "span_info", "s_span", "t_span", "kappa_text_state", "kappa_by_layer",
                "n_tokens_src", "n_tokens_tgt", "n_tokens_spliced",
                "span_len_src", "span_len_tgt", "delta_tok",
                "max_text_len", "text_dim", "span_extension"):
        out[key] = info[key]
    return out
