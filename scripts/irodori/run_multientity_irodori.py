#!/usr/bin/env python3
"""Irodori-TTS v4.1-Small で 1 文中 2 箇所の実体語を同時に差し替える (JA, GPU).

`scripts/analysis/generate_multientity.py`（T5Gemma 版）の構成をそのまま写した
**新規** スクリプト。既存ファイルは 1 行も変更していない。差し替えの本体は
`scripts/projection/splice_multi.py` の `build_spliced_hidden_multi` を
**そのまま import して使う**（再実装しない）。Irodori 側の部品
（`load_irodori` / `encode_text_only` / `find_entity_token_range` /
`spliced_surface_text` / `synthesize` / `_base_request_kwargs` / `seed_all` /
`save_wav`）も既存ファイルから import する。

差し替えるテンソル
------------------
`model.encode_conditions` が返す `text_state` (1, max_text_len=256, 512)。
  ModernBERT-Ja-310m (25 層, hidden 768)
    -> PretrainedConditionProjector (768 -> 512)
    -> text_norm (RMSNorm)
DiT 12 層の JointAttention がこれを key/value としてのみ読む。
`ConditionSplicer`（`irodori_splice.py:408`）が armed の間だけ
(text_state, text_mask) を対で置換する。

1 文あたり 4 条件を同一 seed で生成する（T5Gemma 版と同じ命名・同じ意味）:

    bl      注入なし。req.text = text_a
    inj1    枠 1 だけ差し替え。ref = text_refs[0] の text_state
    inj2    枠 2 だけ差し替え。ref = text_refs[1] の text_state
    inj12   両方差し替え。枠 i の ref は ref_i（**枠ごとに別の forward**）

「両方置換した 1 本の文」から 2 スパンを取らないのが要点
（splice_multi.py の docstring 4 と同じ理由）。

Irodori 固有の事情（すべて既存の README / irodori_splice.py で確認済み）
-----------------------------------------------------------------------
A. テンソル形状は常に (1, 256, 512) で固定、動くのは **有効長** だけ。
   `batch_encode(..., max_length=256)` が右パディングするため。
   したがって `build_spliced_hidden_multi` には **有効長で切った view**
   `text_state[:, :n_tokens]` を渡し、返ってきた [1, T_out, 512] を
   ゼロ埋めの (1, 256, 512) に書き戻す。これは
   `irodori_splice.splice_text_state:362-376` の後半とまったく同じ処理。

B. **text_mask を新しい有効長で作り直す。** 作り直さないと
   `PretrainedTextBackbone.forward`(model.py:774-780) と
   `PretrainedConditionProjector.forward`(model.py:855-882) が掛ける
   `* mask.unsqueeze(-1)` で差し替えた区間がゼロに潰れる。
   **2 区間では総ずれが 2 つ分になる**:
       T_out = T_src + sum_i (m_i - n_i) = T_src + delta_tok
   `delta_tok` は `build_spliced_hidden_multi` が返す総和をそのまま使う。
   各条件で `mask.sum() == T_out` と `state[0, T_out:] == 0` を assert する。

C. **系列 index を自分でずらさない。** 1 個目の差し替えで長さが変わるので、
   順に in-place 置換すると 2 個目の s_range がずれる。
   `build_spliced_hidden_multi` は src の「元の」index だけを読んで
   一括 concat するので構造的にずれない。このスクリプトは
   s_ranges / r_ranges に delta を足す処理を **一切持たない**。

D. スパンの決め方は既存の単一 entity 経路（= Table 3 の JA 行）と同じ
   `irodori_splice.find_entity_token_range`（`injection.py:401` と同じ計算）を
   既定にする。窓は広げず、トークン境界の検証もしない。日本語では
   `田中さん` が 1 トークンに merge するので 300 件中 32 件でスパンが
   実体語の境界からずれている（既存 manifest の n_span_misaligned=32）。
   Irodori 側だけ窓を広げると Table 3 の比較が揃わないので、ずれは
   **記録するだけ**（`span_aligned`）。
   枠 2 は「枠 1 の終端以降」から探す必要があるので、文字カーソルを足した
   `find_entity_token_range_from` を使う。**cursor=0 のときは既存関数へ
   完全委譲する**ので、区間が 1 つのときは既存経路と同じ関数・同じ引数になる。
   これとは別に `splice_multi.resolve_spans_with_cursor`
   （offset_mapping ベース）でも解決して両者の一致を毎件記録し、
   `verify_span_roundtrip` の結果も記録する（`--span_mode cursor` で
   そちらを採用に切り替えられる）。

E. `req.text` は text 条件としては使われない（armed なパッチが spliced 条件を
   返す）が、duration predictor の補助特徴
   （`build_duration_features`: 文字数・かな/漢字比など）の入力になるので、
   差し替え後の表層テキストを渡す。1 枠のときは既存の
   `irodori_splice.spliced_surface_text` を同じ引数で呼び、2 枠のときは
   文字位置から P0+b1+P1+b2+P2 を組む（1 枠の値は既存関数の戻り値と
   一致することを毎回 assert する）。

F. 乱数。`sample_euler_rf_cfg` は `torch.Generator(device).manual_seed(req.seed)`
   で初期ノイズを引く（rf.py:11-16, 158）のでグローバル RNG に依存しない。
   さらに 4 条件すべてで `runtime.synthesize` の **直前** に `seed_all(seed)` を
   呼ぶ（`irodori_splice.synthesize:819` と `splice_synthesize:876` と同じ位置）。
   条件によって encoder forward の回数が違う（bl:0, inj1/inj2:2, inj12:3）が、
   forward は `torch.inference_mode()` の決定的な計算でグローバル RNG を
   消費せず、かつ生成直前に必ず seed を戻すので消費順序の差は残らない。

使い方
------
  本番 (JA 300 文 × 4 条件):
    IMAGE_NAME=irodori-tts:latest SLURM_TIME=12:00:00 \
      slurm_run.sh \
        python3 -u scripts/irodori/run_multientity_irodori.py \
          --mode multi --pairs_module ja_multientity_300 --n 300

  単一区間の等価性確認 (既存 s%04d_{bl,tgt,inj}.wav と md5 比較):
    IMAGE_NAME=irodori-tts:latest SLURM_TIME=01:00:00 \
      slurm_run.sh \
        python3 -u scripts/irodori/run_multientity_irodori.py \
          --mode equiv --n 5 \
          --out_dir eval_results/_equiv_multientity_irodori_ja_k1 \
          --compare_dir eval_results/_cc_wavs_a/irodori_ja

  天井用の目標文だけを後から足す (--tgt_only、2026-09-22 追加):
    IMAGE_NAME=irodori-tts:latest SLURM_TIME=12:00:00 \
      slurm_run.sh \
        python3 -u scripts/irodori/run_multientity_irodori.py \
          --mode multi --tgt_only --pairs_module ja_multientity_300 --n 300 \
          --out_dir eval_results/multientity_irodori_ja_300

    注入しないので区間の解決も往復確認もしない。既にある wav は作り直さない
    ので、既存の bl/inj1/inj2/inj12 と results.json / checkpoint.json は動かない。
    帳面は checkpoint_tgt.json / results_tgt.json（別ファイル）。
    --dry_run を足すとモデルを読まずに、出すつもりの文とファイル名だけ並べる。

出力: <out_dir>/{results.json, checkpoint.json, s%04d_{bl,inj1,inj2,inj12}.wav}
      --tgt_only のときは <out_dir>/{results_tgt.json, checkpoint_tgt.json,
      s%04d_{tgt1,tgt2}.wav}
"""
import argparse
import hashlib
import json
import os
import sys
import time

# --------------------------------------------------------------------------
# HF cache。slurm_run.sh が渡す HF_HOME=/tmp/.cache/huggingface は空で書けない
# ので **代入で上書き** する（setdefault では効かない）。transformers を
# import する前に済ませること。run_irodori_eval.py:51-61 と同じ扱い。
# --------------------------------------------------------------------------
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "modules")

for _p in (os.path.join(ROOT, "scripts", "irodori"),
           os.path.join(ROOT, "scripts", "irodori", "_repo"),
           os.path.join(ROOT, "data"),
           os.path.join(ROOT, "data"),
           os.path.join(ROOT, "scripts", "projection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# 目標文だけを出す経路（--tgt_only）の共通部品。stdlib しか読まない。
# **末尾に足す**ので、既存の import の解決順は 1 つも動かない。
sys.path.append(os.path.join(ROOT, "scripts", "multientity"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# --- Irodori 側の既存部品（1 行も変更していない） ---------------------------
# **transformers の版差を吸収する。**`irodori_splice` より前に読むこと。
# `irodori_tts/model.py:686` が `transformers.initialization` を要求するが、
# その場所に移ったのは transformers 4.57 より後。実行環境は 4.57.3 で
# 旧い場所にある。詳細は `scripts/irodori/_compat_transformers.py`。
import _compat_transformers  # noqa: F401,E402

from irodori_splice import (  # noqa: E402
    DEFAULT_CODEC_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_STEPS,
    SEED,
    _base_request_kwargs,
    describe_entity_span,
    encode_ids,
    encode_text_only,
    entity_cosine,
    find_entity_token_range,
    load_irodori,
    mask_valid_length,
    normalize_for_runtime,
    predicted_latent_frames,
    seed_all,
    spliced_surface_text,
    splice_text_state,
    synthesize,
)
# --- 複数区間 splice の本体（再実装しない） ---------------------------------
from splice_multi import (  # noqa: E402
    build_spliced_hidden_multi,
    check_span_sets,
    resolve_spans_with_cursor,
    span_deltas,
    verify_span_roundtrip,
)
# --- wav の書き出しは既存の単一 entity 経路と同一の関数を使う ---------------
# （import した時点で run_irodori_eval.py も同じ HF_HOME を代入するだけ）
from run_irodori_eval import atomic_json_dump, save_wav  # noqa: E402
# --- 天井用の目標文だけを出す経路（--tgt_only、2026-09-22 追加） -------------
from tgt_only import dry_run_report, run_targets  # noqa: E402

CONDITIONS = ["bl", "inj1", "inj2", "inj12"]
COND_SLOTS = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}
# equiv では既存の単一 entity 経路（run_irodori_eval.py:241-267）と **同じ順序**
# bl -> tgt -> inj で生成する。3 本とも既存 wav と md5 を比べられる。
EQUIV_CONDITIONS = ["bl", "tgt", "inj"]
EQUIV_SLOTS = {"bl": [], "tgt": [], "inj": [0]}
# 注入しない条件が req.text に何を渡すか（"a" = text_a, "ref0" = text_refs[0]）
COND_PLAIN_TEXT = {"bl": "a", "tgt": "ref0"}


# ==========================================================================
# 1. スパン解決
# ==========================================================================
class FastTokShim:
    """`tokenizers.Tokenizer` を HF fast tokenizer の呼び出し形に見せる薄い層.

    `splice_multi.resolve_spans_with_cursor` / `verify_span_roundtrip` は
    `text_tok(text, add_special_tokens=False, return_offsets_mapping=True)` と
    いう HF `PreTrainedTokenizerFast` の呼び方をする。Irodori に同梱されて
    いるのは素の `tokenizers.Tokenizer`（callable ではない）なので、
    この 1 メソッドだけを橋渡しする。**id も offsets も加工しない。**
    """

    def __init__(self, raw_tokenizer):
        self.raw = raw_tokenizer
        self.is_fast = True

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, **kwargs):
        enc = self.raw.encode(text, add_special_tokens=bool(add_special_tokens))
        out = {"input_ids": [int(i) for i in enc.ids]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(int(a), int(b)) for a, b in enc.offsets]
        return out


def find_entity_token_range_from(raw_tokenizer, full_text, entity_text,
                                 char_from=0, offset=0):
    """`find_entity_token_range` に文字カーソルを足したもの.

    **char_from <= 0 のときは既存関数へ完全委譲する。** したがって区間が
    1 つのとき（枠 1 のみ / cross-culture の単一 entity）は既存の単一 entity
    経路（`irodori_splice.build_splice:677-678`）と同じ関数・同じ引数になる。

    char_from > 0 のとき（枠 2）だけ、3 段の意味を保ったまま「char_from 以降」
    に限定する。窓は広げず、トークン境界の検証もしない（既存と同じ流儀）。
    """
    if char_from <= 0:
        return find_entity_token_range(raw_tokenizer, full_text, entity_text, offset)

    full_tokens = encode_ids(raw_tokenizer, full_text)
    entity_tokens = encode_ids(raw_tokenizer, entity_text)
    tok_from = len(encode_ids(raw_tokenizer, full_text[:char_from]))

    # 1 段目: トークン列の部分列検索（開始位置を tok_from 以降に限る）
    for start in range(tok_from, len(full_tokens) - len(entity_tokens) + 1):
        if full_tokens[start:start + len(entity_tokens)] == entity_tokens:
            return (start + offset, start + len(entity_tokens) + offset,
                    "subsequence")
    # フォールバック 1: 文字位置から prefix をトークナイズして開始位置を出す
    pos = full_text.find(entity_text, char_from)
    if pos >= 0:
        prefix_tokens = encode_ids(raw_tokenizer, full_text[:pos])
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(entity_tokens) + offset, "char_prefix")
    # フォールバック 2: 大文字小文字を無視（日本語では発火しない）
    lo = full_text.lower().find(entity_text.lower(), char_from)
    if lo >= 0:
        actual = full_text[lo:lo + len(entity_text)]
        actual_tokens = encode_ids(raw_tokenizer, actual)
        for start in range(tok_from, len(full_tokens) - len(actual_tokens) + 1):
            if full_tokens[start:start + len(actual_tokens)] == actual_tokens:
                return (start + offset, start + len(actual_tokens) + offset,
                        "subsequence_ci")
        prefix_tokens = encode_ids(raw_tokenizer, full_text[:lo])
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(actual_tokens) + offset, "char_prefix_ci")
    return None


def char_spans_with_cursor(full_text, surfaces, start=0):
    """表層形を左から順に文字位置へ解決する（同じ語が 2 回出ても取り違えない）."""
    out, cur = [], int(start)
    for surf in surfaces:
        p = full_text.find(surf, cur)
        if p < 0:
            return None, f"surface {surf!r} not found at or after char {cur}"
        out.append((p, p + len(surf)))
        cur = p + len(surf)
    return out, None


def resolve_side(raw_tokenizer, shim, norm_text, surfaces, nospec_offsets,
                 entity_offset, char_hints=None, start_at=0):
    """片側（src または ref 1 本）の k 個のスパンを解決する.

    `start_at` は探索を始める文字位置。ref_i（枠 i だけ置換した文）では
    枠 i-1 の語の終端を渡す。ref_i は text_a と枠 i の中身しか違わないので、
    枠 i-1 の終端までの文字列は text_a と同一であり、そのまま使える。

    返り値 (res, err)。res は
      spans_legacy : find_entity_token_range_from の結果（nospec 座標）
      spans_cursor : splice_multi.resolve_spans_with_cursor の結果 / None
      agree / roundtrip_ok / stages / aligned / char_spans / entity_offset
    """
    ch, err = char_spans_with_cursor(norm_text, surfaces, start=start_at)
    if ch is None:
        return None, f"char resolve failed: {err}"

    # --- 採用側: 既存 3 モデルと同じ計算（枠 2 だけ文字カーソル付き） -------
    legacy, stages, reports = [], [], []
    for i, surf in enumerate(surfaces):
        char_from = int(start_at) if i == 0 else ch[i - 1][1]
        found = find_entity_token_range_from(raw_tokenizer, norm_text, surf,
                                             char_from=char_from)
        if found is None:
            return None, f"span {i}: entity {surf!r} not found in {norm_text!r}"
        legacy.append((int(found[0]), int(found[1])))
        stages.append(found[2])
        reports.append(describe_entity_span(raw_tokenizer, norm_text, surf,
                                            nospec_offsets, found[:2], found[2]))

    # --- 参照側: splice_multi の cursor 解決 + 往復確認（記録用） -----------
    cursor_spans, cinfo = resolve_spans_with_cursor(
        shim, norm_text, list(surfaces), add_bos=0, char_starts=char_hints)
    if cursor_spans is None:
        rt_ok, rt_bad = False, [f"unresolved: {cinfo.get('reason')}"]
    else:
        rt_ok, rt_bad = verify_span_roundtrip(shim, norm_text, cursor_spans,
                                              list(surfaces), add_bos=0)
    agree = (cursor_spans is not None
             and [tuple(x) for x in cursor_spans] == [tuple(x) for x in legacy])

    res = {
        "spans_legacy": legacy,
        "spans_cursor": ([list(x) for x in cursor_spans]
                         if cursor_spans is not None else None),
        "cursor_via": ([p["via"] for p in cinfo["per_span"]]
                       if cursor_spans is not None else None),
        "agree": bool(agree),
        "roundtrip_ok": bool(rt_ok),
        "roundtrip_bad": rt_bad,
        "stages": stages,
        "aligned": [bool(r["aligned"]) for r in reports],
        "char_spans": [list(x) for x in ch],
        "reports": reports,
        "entity_offset": int(entity_offset),
    }
    return res, None


def to_abs(spans_nospec, entity_offset):
    """nospec 座標 -> batch_encode 座標。**全スパンに同じ offset を一度だけ足す。**

    `irodori_splice.build_splice:690-691` と同じ変換。
    """
    return [(int(a) + int(entity_offset), int(b) + int(entity_offset))
            for a, b in spans_nospec]


# ==========================================================================
# 2. splice（本体は splice_multi、ここは有効長 <-> パディングの橋渡しだけ）
# ==========================================================================
def splice_multi_padded(src, ref_encs, s_ranges, r_ranges):
    """k 区間差し替え + text_mask の作り直し（Irodori のパディング規約に戻す）.

    `build_spliced_hidden_multi` には **有効長で切った view** を渡す。
    戻ってきた [1, T_out, 512] をゼロ埋めの (1, 256, 512) に書き戻し、
    mask を新しい有効長 T_out で作り直す。これは
    `irodori_splice.splice_text_state:362-376` の後半と同一の処理で、
    区間が 1 つなら bit 一致する
    （`verify_multientity_irodori_cpu.py` の test_k1_bit_identical が確認）。
    """
    src_state, src_mask = src["text_state"], src["text_mask"]
    if src_state.dim() != 3 or src_state.shape[0] != 1:
        raise ValueError(f"text_state must be (1, L, D), got {tuple(src_state.shape)}")
    max_len = int(src_state.shape[1])
    t_src = int(mask_valid_length(src_mask[0]))

    # (c) 重なり検査。build_spliced_hidden_multi の中でも assert されるが、
    #     skip 理由を明示するためここでも明示的に呼ぶ。
    ref_lens = [int(mask_valid_length(r["text_mask"][0])) for r in ref_encs]
    check_span_sets(t_src, ref_lens, list(s_ranges), list(r_ranges))

    src_view = {"last_hidden_state": src_state[:, :t_src]}
    ref_views = []
    for r, n in zip(ref_encs, ref_lens):
        if r["text_state"].shape[1:] != src_state.shape[1:]:
            raise ValueError("ref text_state shape differs from src")
        ref_views.append({"last_hidden_state": r["text_state"][:, :n]})

    modified, delta_tok, out_ranges = build_spliced_hidden_multi(
        src_view, ref_views, list(s_ranges), list(r_ranges))

    t_out = int(modified.shape[1])
    per_span, total = span_deltas(list(s_ranges), list(r_ranges))
    if delta_tok != total or t_out != t_src + total:
        raise AssertionError(
            f"length bookkeeping mismatch: T_src={t_src} delta={delta_tok} "
            f"sum(per_span)={total} T_out={t_out}")
    if t_out > max_len:
        raise ValueError(
            f"spliced length {t_out} exceeds max_text_len {max_len}")

    new_state = torch.zeros_like(src_state)
    new_state[0, :t_out] = modified[0]
    new_mask = torch.zeros_like(src_mask, dtype=torch.bool)
    new_mask[0, :t_out] = True

    # (B) mask の作り直しが正しいことを毎件確かめる。2 区間では総ずれが 2 つ分。
    if int(new_mask.sum().item()) != t_out:
        raise AssertionError(f"rebuilt text_mask has {int(new_mask.sum())} valid "
                             f"positions, expected {t_out}")
    if t_out < max_len and float(new_state[0, t_out:].abs().max().item()) != 0.0:
        raise AssertionError("padding positions of the spliced text_state are not zero")

    info = {
        "n_tokens_src": t_src,
        "n_tokens_ref": ref_lens,
        "n_tokens_spliced": t_out,
        "T_out": t_out,
        "span_len_src": [int(b - a) for a, b in s_ranges],
        "span_len_ref": [int(b - a) for a, b in r_ranges],
        "delta_per_span": [int(x) for x in per_span],
        "delta_tok": int(delta_tok),
        "out_ranges": [list(x) for x in out_ranges],
        "mask_valid": int(new_mask.sum().item()),
        "max_text_len": max_len,
        "text_dim": int(src_state.shape[2]),
    }
    return new_state, new_mask, info


# ==========================================================================
# 3. 表層テキスト（duration predictor の補助特徴用）
# ==========================================================================
def surface_from_char_spans(norm_src, char_spans, words_b):
    """P0 + b1 + P1 + b2 + ... を文字位置だけで組む（k>=1 共通）."""
    out, cur = [], 0
    for (c0, c1), wb in zip(char_spans, words_b):
        if c0 < cur:
            raise ValueError(f"char spans not left-to-right: {char_spans}")
        out.append(norm_src[cur:c0])
        out.append(wb)
        cur = c1
    out.append(norm_src[cur:])
    return "".join(out)


def surface_text_for(norm_src, char_spans, words_a, words_b, norm_tgts):
    """差し替え後の表層テキスト。

    区間が 1 つのときは既存の `irodori_splice.spliced_surface_text` を
    **同じ引数で** 呼び、その戻り値を採用する（既存の単一 entity 経路と
    同一の文字列になることを保証する）。区間が 2 つのときは文字位置から組み、
    1 つのときは両者が一致することを assert する。
    """
    multi_val = surface_from_char_spans(norm_src, char_spans, words_b)
    if len(char_spans) == 1:
        legacy_val = spliced_surface_text(norm_src, words_a[0],
                                          norm_tgts[0], words_b[0])
        if legacy_val != multi_val:
            raise AssertionError(
                f"surface text mismatch: spliced_surface_text -> {legacy_val!r} "
                f"but char-span splice -> {multi_val!r}")
        return legacy_val
    return multi_val


# ==========================================================================
# 4. 生成
# ==========================================================================
def splice_synthesize_multi(bundle, new_state, new_mask, surface_text,
                            seed=SEED, seconds=None, num_steps=None):
    """armed な splicer 下で 1 回生成する.

    `irodori_splice.splice_synthesize:870-901` と同じ手順・同じ assert。
    違いは「spliced な (text_state, text_mask) を外から受け取る」点だけ
    （splice_synthesize は 1 区間しか組めないので再利用できない）。
    """
    from irodori_tts.inference_runtime import SamplingRequest

    req = SamplingRequest(text=surface_text,
                          **_base_request_kwargs(seconds, seed, num_steps, None))
    calls_before = bundle.splicer.n_calls
    repl_before = bundle.splicer.n_replaced
    seed_all(seed)   # 生成の直前に必ず戻す（4 条件で消費順序をそろえる）
    with torch.inference_mode(), bundle.splicer.armed(new_state, new_mask):
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
    return {
        "audio": audio,
        "sampling_rate": int(result.sample_rate),
        "used_seed": int(result.used_seed),
        "n_encode_conditions_calls": n_calls,
        "n_spliced": n_repl,
        "latent_frames": predicted_latent_frames(result),
        "duration_sec": float(audio.shape[-1]) / float(result.sample_rate),
    }


# ==========================================================================
# 5. データ
# ==========================================================================
def load_items_multi(pairs_module, n):
    """data/<pairs_module>.py の build_pairs()（2 枠形式）."""
    mod = __import__(pairs_module)
    items = mod.build_pairs()
    for it in items:
        assert len(it["spans"]) == 2, \
            f"pair {it['pair_idx']}: this script expects exactly 2 spans"
        assert len(it["text_refs"]) == 2
    return items[:n] if n else items


def load_items_equiv(n):
    """cross-culture 300 対を「区間 1 つの複数 entity 入力」に変換する.

    ここで作るレコードは 2 枠形式と同じ鍵を持つが spans が 1 個。
    これを複数 entity の経路に流すと、区間が 1 つのときの挙動が
    既存の単一 entity 経路と一致するかを確かめられる。
    """
    from ja_cross_culture_names import build_300_pairs

    pairs = build_300_pairs()
    if n:
        pairs = pairs[:n]
    items = []
    for i, p in enumerate(pairs):
        pidx = p.get("pair_idx", i)
        items.append({
            "pair_idx": pidx,
            "template_id": None,
            "lang": "ja",
            "text_a": p["text_a"],
            "text_refs": [p["text_b"]],
            "text_both": p["text_b"],
            "gap_words": None,
            "slot1_at_start": p["text_a"].startswith(p["word_a"]),
            "spans": [{"slot": 1, "word_a": p["word_a"], "word_b": p["word_b"],
                       "char_start_a": None, "char_start_b": None}],
        })
    return items


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ==========================================================================
# 6. 本体
# ==========================================================================
def run(args):
    conds = EQUIV_CONDITIONS if args.mode == "equiv" else CONDITIONS
    slots_of = EQUIV_SLOTS if args.mode == "equiv" else COND_SLOTS
    n_load = 0 if args.pair_ids else args.n
    items = (load_items_equiv(n_load) if args.mode == "equiv"
             else load_items_multi(args.pairs_module, n_load))
    if args.pair_ids:
        want = [int(x) for x in args.pair_ids.split(",") if x.strip()]
        by = {it["pair_idx"]: it for it in items}
        missing = [w for w in want if w not in by]
        if missing:
            raise SystemExit(f"--pair_ids に無い pair_idx がある: {missing}")
        items = [by[w] for w in want]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[data] mode={args.mode} n={len(items)} conds={conds}", flush=True)
    print(f"[data] out_dir={args.out_dir}", flush=True)
    print(f"[env]  HF_HOME={os.environ['HF_HOME']}", flush=True)

    if args.tgt_only and args.mode != "multi":
        raise SystemExit("--tgt_only は 2 枠のデータ (--mode multi) 専用")
    if args.dry_run:
        # モデルを読む前に返す（CPU で名前と文を確かめるための経路）。
        dry_run_report(items, args.out_dir, tag="irodori")
        return

    print(f"[load] {args.model_id} precision={args.precision} device={args.device}",
          flush=True)
    t_load = time.time()
    bundle = load_irodori(model_id=args.model_id, codec_id=args.codec_id,
                          device=args.device, precision=args.precision,
                          codec_device=args.codec_device or args.device)
    shim = FastTokShim(bundle.raw_tokenizer)
    print(f"[load] done in {time.time() - t_load:.1f}s sr={bundle.sample_rate} "
          f"hop={bundle.hop_length} max_text_len={bundle.max_text_len}", flush=True)

    if args.tgt_only:
        # 注入しないので区間の解決も往復確認もしない。合成は bl 条件と同じ
        # synthesize で、渡す文だけが text_refs[slot] に変わる。
        _seconds = (float(args.fixed_seconds)
                    if args.duration_mode == "fixed" else None)

        def _synth_tgt(text, path):
            seed_all(args.seed)
            out = synthesize(bundle, text, seed=args.seed, seconds=_seconds,
                             num_steps=args.num_steps)
            _, dur, clip = save_wav(path, out["audio"].numpy(),
                                    bundle.sample_rate)
            return {"dur_sec": round(float(dur), 4),
                    "clip_ratio": round(float(clip), 6),
                    "used_seed": out.get("used_seed"),
                    "md5": md5_file(path)}

        run_targets(items, args.out_dir, _synth_tgt, tag="irodori",
                    ckpt_every=args.ckpt_every, log_every=args.log_every,
                    extra_summary={"method": "irodori", "lang": args.lang,
                                   "pairs_module": args.pairs_module,
                                   "model_id": args.model_id,
                                   "precision": args.precision,
                                   "seed": int(args.seed),
                                   "duration_mode": args.duration_mode,
                                   "num_steps": args.num_steps})
        return

    ckpt = os.path.join(args.out_dir, "checkpoint.json")
    rows, skipped = [], []
    if os.path.exists(ckpt) and not args.overwrite:
        st = json.load(open(ckpt))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != f"irodori_splice_multi_{args.mode}"]
        assert not bad, (f"checkpoint mode mismatch in {ckpt}: pairs {bad[:5]} were "
                         f"not produced by this script/mode; use another --out_dir")
        print(f"[resume] {len(rows)} rows, {len(skipped)} skipped", flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}
    t0 = time.time()

    for it in items:
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        k = len(it["spans"])
        seed_all(args.seed)
        norm_a = normalize_for_runtime(it["text_a"])
        norm_refs = [normalize_for_runtime(t) for t in it["text_refs"]]

        # --- encode（splicer は disarm のまま。条件間で使い回す） -----------
        try:
            src = encode_text_only(bundle, norm_a)
            ref_encs = [encode_text_only(bundle, r) for r in norm_refs]
        except Exception as e:
            print(f"  [{pidx}] enc fail: {str(e)[:160]}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"enc: {e}"})
            done.add(pidx)
            continue

        # --- 区間の解決 -> 往復確認 -> 重なり検査 ----------------------------
        words_a = [sp["word_a"] for sp in it["spans"]]
        words_b = [sp["word_b"] for sp in it["spans"]]
        hints_a = ([sp["char_start_a"] for sp in it["spans"]]
                   if all(sp["char_start_a"] is not None for sp in it["spans"])
                   else None)
        s_res, err = resolve_side(
            bundle.raw_tokenizer, shim, norm_a, words_a,
            src["nospec_offsets"], src["entity_offset"], char_hints=hints_a)
        if s_res is None:
            print(f"  [{pidx}] SKIP src: {err}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"src: {err}"})
            done.add(pidx)
            continue

        # ref_i は「枠 i だけ置換した文」。枠 i の語を、枠 i-1 の語の終端以降
        # から探す。ref_i は text_a と枠 i の中身しか違わないので、枠 i-1 の
        # 終端までの文字列は text_a と同一であり、src 側で解決した文字位置を
        # そのまま流用できる。
        r_res, r_err = [], None
        for i, sp in enumerate(it["spans"]):
            hint = ([sp["char_start_b"]] if sp["char_start_b"] is not None
                    else None)
            start_at = 0 if i == 0 else s_res["char_spans"][i - 1][1]
            one, e1 = resolve_side(
                bundle.raw_tokenizer, shim, norm_refs[i], [words_b[i]],
                ref_encs[i]["nospec_offsets"], ref_encs[i]["entity_offset"],
                char_hints=hint, start_at=start_at)
            if one is None:
                r_err = f"ref[{i}]: {e1}"
                break
            r_res.append(one)
        if r_err is not None:
            print(f"  [{pidx}] SKIP {r_err}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": r_err})
            done.add(pidx)
            continue

        pick = "spans_cursor" if args.span_mode == "cursor" else "spans_legacy"
        if s_res[pick] is None or any(r[pick] is None for r in r_res):
            skipped.append({"pair_idx": pidx,
                            "reason": f"span_mode={args.span_mode} unresolved"})
            done.add(pidx)
            continue
        s_abs = to_abs(s_res[pick], s_res["entity_offset"])
        r_abs = [to_abs(r[pick], r["entity_offset"])[0] for r in r_res]

        if args.roundtrip_policy == "skip" and not (
                s_res["roundtrip_ok"] and all(r["roundtrip_ok"] for r in r_res)):
            reason = {"src": s_res["roundtrip_bad"],
                      "ref": [r["roundtrip_bad"] for r in r_res]}
            print(f"  [{pidx}] SKIP roundtrip {reason}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"roundtrip: {reason}"})
            done.add(pidx)
            continue

        # (c) 2 区間が重ならないこと。隣接は許す（長さ 0 の gap）。
        if k >= 2 and s_abs[0][1] > s_abs[1][0]:
            print(f"  [{pidx}] SKIP overlap {s_abs} {words_a}", flush=True)
            skipped.append({"pair_idx": pidx,
                            "reason": f"src spans overlap: {s_abs} ({words_a})"})
            done.add(pidx)
            continue

        span_tok = {
            "src": [list(x) for x in s_abs],
            "ref": [list(x) for x in r_abs],
            "n": [b - a for a, b in s_abs],
            "m": [b - a for a, b in r_abs],
            "gap_tokens": (s_abs[1][0] - s_abs[0][1]) if k >= 2 else None,
            "src_entity_offset": s_res["entity_offset"],
            "ref_entity_offset": [r["entity_offset"] for r in r_res],
            "src_stage": s_res["stages"],
            "ref_stage": [r["stages"][0] for r in r_res],
            "src_aligned": s_res["aligned"],
            "ref_aligned": [r["aligned"][0] for r in r_res],
            "src_resolver_agree": s_res["agree"],
            "ref_resolver_agree": [r["agree"] for r in r_res],
            "src_roundtrip_ok": s_res["roundtrip_ok"],
            "ref_roundtrip_ok": [r["roundtrip_ok"] for r in r_res],
            "src_spans_cursor": s_res["spans_cursor"],
            "src_spans_legacy": [list(x) for x in s_res["spans_legacy"]],
            "span_mode": args.span_mode,
        }
        span_tok["delta_per_span"] = [span_tok["m"][i] - span_tok["n"][i]
                                     for i in range(k)]
        span_tok["len_imbalance"] = (abs(span_tok["m"][0] - span_tok["m"][1])
                                    if k >= 2 else None)

        row = {
            "pair_idx": pidx, "mode": f"irodori_splice_multi_{args.mode}",
            "method": "irodori", "lang": "ja",
            "template_id": it.get("template_id"),
            "text_a": it["text_a"], "text_refs": it["text_refs"],
            "text_both": it["text_both"],
            "normalized_text_a": norm_a, "normalized_text_refs": norm_refs,
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "words_a": words_a, "words_b": words_b,
            "span_tok": span_tok,
            "span_aligned": bool(all(s_res["aligned"])
                                 and all(r["aligned"][0] for r in r_res)),
            "T_src": int(src["n_tokens"]),
            "max_text_len": int(bundle.max_text_len),
            "conds": {},
        }
        if args.kappa:
            row["kappa_text_state"] = [
                entity_cosine(src["text_state"], ref_encs[i]["text_state"],
                              tuple(s_abs[i]), tuple(r_abs[i])) for i in range(k)]

        seconds = float(args.fixed_seconds) if args.duration_mode == "fixed" else None

        ok = True
        for cond in conds:
            use = slots_of[cond]
            delta_tok, out_ranges, sinfo = 0, [], None
            new_state = new_mask = None
            if COND_PLAIN_TEXT.get(cond) == "ref0":
                target_text = it["text_refs"][0]
                plain_T = int(ref_encs[0]["n_tokens"])
            else:
                target_text = it["text_a"]
                plain_T = int(src["n_tokens"])
            if use:
                try:
                    sub_s = [s_abs[i] for i in use]
                    sub_r = [r_abs[i] for i in use]
                    sub_ref = [ref_encs[i] for i in use]
                    new_state, new_mask, sinfo = splice_multi_padded(
                        src, sub_ref, sub_s, sub_r)
                    delta_tok = sinfo["delta_tok"]
                    out_ranges = sinfo["out_ranges"]
                    target_text = surface_text_for(
                        norm_a, [s_res["char_spans"][i] for i in use],
                        [words_a[i] for i in use], [words_b[i] for i in use],
                        [norm_refs[i] for i in use])
                except Exception as e:
                    print(f"  [{pidx}] {cond} splice prep fail: {str(e)[:200]}",
                          flush=True)
                    skipped.append({"pair_idx": pidx, "reason": f"{cond} prep: {e}"})
                    ok = False
                    break

            try:
                if use:
                    out = splice_synthesize_multi(
                        bundle, new_state, new_mask, target_text,
                        seed=args.seed, seconds=seconds, num_steps=args.num_steps)
                else:
                    out = synthesize(bundle, target_text, seed=args.seed,
                                     seconds=seconds, num_steps=args.num_steps)
            except Exception as e:
                print(f"  [{pidx}] {cond} synth fail: {str(e)[:200]}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} synth: {e}"})
                ok = False
                break

            wav_path = os.path.join(args.out_dir, f"s{pidx:04d}_{cond}.wav")
            try:
                _, dur, clip = save_wav(wav_path, out["audio"].numpy(),
                                        bundle.sample_rate)
            except Exception as e:
                print(f"  [{pidx}] {cond} save fail: {str(e)[:200]}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} save: {e}"})
                ok = False
                break

            row["conds"][cond] = {
                "wav": os.path.basename(wav_path),
                "delta_tok": int(delta_tok),
                "T_out": int(sinfo["T_out"]) if sinfo else int(plain_T),
                "out_ranges": [list(x) for x in out_ranges],
                "target_text": target_text,
                "dur_sec": round(float(dur), 4),
                "slots": list(use),
                "mask_valid": int(sinfo["mask_valid"]) if sinfo
                else int(plain_T),
                "delta_per_span": sinfo["delta_per_span"] if sinfo else [],
                "n_encode_conditions_calls": out.get("n_encode_conditions_calls"),
                "n_spliced": out.get("n_spliced", 0),
                "latent_frames": out.get("latent_frames"),
                "used_seed": out.get("used_seed"),
                "clip_ratio": round(float(clip), 6),
                "utmos": None, "pesq": None,
            }

        done.add(pidx)
        if not ok:
            continue
        drift_cond = "inj12" if "inj12" in row["conds"] else conds[-1]
        row["total_position_drift"] = row["conds"][drift_cond]["delta_tok"]
        rows.append(row)

        if len(rows) % args.log_every == 0 or len(rows) == 1:
            el = (time.time() - t0) / 60
            c = row["conds"]
            print(f"  [{len(rows)}/{len(items)}] idx={pidx} "
                  f"gap_tok={span_tok['gap_tokens']} "
                  f"T_src={row['T_src']} drift={row['total_position_drift']} "
                  f"dur=" + "/".join(f"{c[x]['dur_sec']:.2f}" for x in conds)
                  + f" aligned={row['span_aligned']} {el:.1f}min", flush=True)
        if len(rows) % args.ckpt_every == 0:
            atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt)

    atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt)

    def _mean(cond, key):
        v = [r["conds"][cond][key] for r in rows
             if r["conds"].get(cond, {}).get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    summary = {
        "method": "irodori", "lang": "ja", "mode": args.mode,
        "pairs_module": (args.pairs_module if args.mode == "multi"
                         else "ja_cross_culture_names.build_300_pairs"),
        "conds": conds,
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "model_id": args.model_id, "codec_id": args.codec_id,
        "checkpoint_path": bundle.checkpoint_path,
        "tokenizer_path": bundle.tokenizer_path,
        "precision": args.precision, "device": args.device, "seed": args.seed,
        "num_steps": args.num_steps if args.num_steps else DEFAULT_NUM_STEPS,
        "num_steps_arg": args.num_steps,
        "duration_mode": args.duration_mode,
        "span_mode": args.span_mode, "roundtrip_policy": args.roundtrip_policy,
        "sampling_rate": bundle.sample_rate, "hop_length": bundle.hop_length,
        "max_text_len": bundle.max_text_len,
        "cfg_scale_text": 3.0, "cfg_scale_caption": 3.0, "cfg_scale_speaker": 5.0,
        "no_ref": True, "caption": None,
        "mean_dur_sec": {c: _mean(c, "dur_sec") for c in conds},
        "mean_abs_position_drift": round(float(np.mean(
            [abs(r["total_position_drift"]) for r in rows])), 3) if rows else None,
        "n_span_misaligned": sum(1 for r in rows if r["span_aligned"] is False),
        "n_resolver_disagree": sum(
            1 for r in rows if not (r["span_tok"]["src_resolver_agree"]
                                    and all(r["span_tok"]["ref_resolver_agree"]))),
        "n_roundtrip_failed": sum(
            1 for r in rows if not (r["span_tok"]["src_roundtrip_ok"]
                                    and all(r["span_tok"]["ref_roundtrip_ok"]))),
        "splice": ("text_state span replacement at model.encode_conditions; "
                   "k spans via splice_multi.build_spliced_hidden_multi "
                   "(single torch.cat, src indices read from the original "
                   "sequence, no manual index shifting); text_mask rebuilt to "
                   "T_out = T_src + sum_i (m_i - n_i); padded positions zeroed; "
                   "tensor shape stays (1, max_text_len, 512)."),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    import transformers  # noqa: E402
    summary["transformers"] = transformers.__version__
    summary["torch"] = torch.__version__
    atomic_json_dump({"summary": summary, "rows": rows, "skipped": skipped},
                     os.path.join(args.out_dir, "results.json"))
    print("\n=== multi-entity generation (irodori / ja) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    # --- 単一区間の等価性確認 -------------------------------------------
    if args.compare_dir:
        cmp_dir = args.compare_dir if os.path.isabs(args.compare_dir) \
            else os.path.join(ROOT, args.compare_dir)
        print(f"\n=== md5 vs {cmp_dir} ===")
        n_ok = n_bad = n_miss = 0
        report = []
        for r in rows:
            for cond in conds:
                c = r["conds"].get(cond)
                if not c:
                    continue
                mine = os.path.join(args.out_dir, c["wav"])
                theirs = os.path.join(cmp_dir, c["wav"])
                if not os.path.exists(theirs):
                    print(f"  s{r['pair_idx']:04d}_{cond}: MISSING reference")
                    n_miss += 1
                    continue
                a, b = md5_file(mine), md5_file(theirs)
                same = (a == b)
                n_ok += int(same)
                n_bad += int(not same)
                print(f"  s{r['pair_idx']:04d}_{cond}: "
                      f"{'MATCH' if same else 'DIFF '} {a} {b}")
                report.append({"wav": c["wav"], "match": same,
                               "md5_new": a, "md5_ref": b})
        print(f"[md5] match={n_ok} diff={n_bad} missing_ref={n_miss}")
        print("[md5] " + ("PASS: 単一区間で既存の単一 entity 経路と bit 一致"
                          if n_bad == 0 and n_ok > 0 else
                          "FAIL: 一致しない wav がある（移植のどこかが違う）"))
        atomic_json_dump({"compare_dir": cmp_dir, "n_match": n_ok,
                          "n_diff": n_bad, "n_missing_ref": n_miss,
                          "rows": report},
                         os.path.join(args.out_dir, "md5_equivalence.json"))
    if skipped:
        print(f"[warn] {len(skipped)} items skipped; see results.json['skipped']")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="multi", choices=["multi", "equiv"],
                   help="multi=2 枠同時差し替え本番 / equiv=区間 1 つで既存 "
                        "単一 entity 経路との md5 等価性確認")
    p.add_argument("--pairs_module", default="ja_multientity_300",
                   help="data/<name>.py の build_pairs()（2 枠形式）")
    p.add_argument("--lang", default="ja", choices=["ja"],
                   help="Irodori は日本語専用（モデルカード language: ja）")
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--pair_ids", default=None,
                   help="pair_idx をカンマ区切りで指定（--n より優先）。"
                        "equiv の md5 比較では参照 wav がどの run で作られたかを "
                        "mtime で確かめてから選ぶこと "
                        "(eval_results/_cc_wavs_a/irodori_ja の s0000-s0004 は "
                        "2026-09-16 18:56 の別 run 由来で、manifest に記録されて "
                        "いる 2026-09-17 09:38 の run のものではない)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--compare_dir", default=None,
                   help="生成後に同名 wav の md5 を比較する参照ディレクトリ "
                        "(例 eval_results/_cc_wavs_a/irodori_ja)")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--codec_id", default=DEFAULT_CODEC_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--codec_device", default=None)
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"],
                   help="既存の単一 entity 経路と同じ fp32")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_steps", type=int, default=None,
                   help=f"未指定なら runtime の既定 {DEFAULT_NUM_STEPS}。"
                        f"equiv では既存 run と同じく未指定のままにすること")
    p.add_argument("--duration_mode", default="predict",
                   choices=["predict", "fixed"],
                   help="predict=spliced 条件で duration predictor を走らせる"
                        "（既存の単一 entity 経路の既定と同じ）")
    p.add_argument("--fixed_seconds", type=float, default=6.0)
    p.add_argument("--span_mode", default="legacy", choices=["legacy", "cursor"],
                   help="legacy=find_entity_token_range（既存 3 モデルと同じ計算、"
                        "枠 2 のみ文字カーソル付き。既定）/ "
                        "cursor=splice_multi.resolve_spans_with_cursor")
    p.add_argument("--roundtrip_policy", default="record",
                   choices=["record", "skip"],
                   help="record=往復不一致でも進む（Irodori 既存の流儀。"
                        "スパンずれは span_aligned に記録）/ skip=文ごと捨てる")
    p.add_argument("--kappa", action="store_true",
                   help="text_state 上の kappa を記録（生成には影響しない）")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--ckpt_every", type=int, default=5)
    # **後から目標文だけを足すための経路（2026-09-22）。**既定は off なので
    # 既存の呼び出しはこの 2 つを一切通らない。
    p.add_argument("--tgt_only", action="store_true",
                   help="枠ごとの目標文 tgt1 / tgt2 だけを合成する（注入なし）。"
                        "既存の bl/inj1/inj2/inj12 と results.json は触らない")
    p.add_argument("--dry_run", action="store_true",
                   help="モデルを読まずに、出すつもりの文とファイル名を並べる")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.out_dir is None:
        args.out_dir = (f"eval_results/multientity_irodori_ja"
                        if args.mode == "multi"
                        else "eval_results/_equiv_multientity_irodori_ja_k1")
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(ROOT, args.out_dir)
    if args.mode == "equiv":
        if args.duration_mode != "predict":
            raise SystemExit("equiv では duration_mode=predict でなければ "
                             "既存 run と条件が揃わない")
        if args.num_steps is not None:
            print("[warn] equiv では --num_steps を指定しないこと（既存 run は "
                  "num_steps=None でリクエストしている）", flush=True)
    os.chdir(ROOT)
    run(args)


if __name__ == "__main__":
    main()
