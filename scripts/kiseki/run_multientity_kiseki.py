#!/usr/bin/env python3
"""Kiseki-TTS で 1 文中の 2 箇所の実体語を同時に差し替える (BL / INJ_1 / INJ_2 / INJ_12).

`scripts/analysis/generate_multientity.py`(T5Gemma 版) の構成をそのまま写した
**新規** スクリプト。既存ファイルは 1 行も変更していない。差し替えの本体は
再実装せず `scripts/projection/splice_multi.py` の
`build_spliced_hidden_multi` を import して使う。生成・スパン探索・wav 書き出しは
`scripts/kiseki/kiseki_splice.py` と `scripts/kiseki/run_kiseki_eval.py` から
import した既存関数をそのまま呼ぶので、単一 entity の経路と定義上同一になる。

Kiseki の差し替え先
-------------------
`encoder_out` = text encoder の最終 RMSNorm 出力
(`Mamba2Seq2SeqEncoder.forward` が `self.layer_norm(x)` として返すテンソル)。
Mamba2 decoder が交差注意で読むのはこれ。`generate_audio` に
`encoder_outputs=BaseModelOutput(last_hidden_state=...)` として渡し、
`input_ids=None, attention_mask=None` にすると `_build_cross_mask`
(modeling_mamba2_s2s.py:419-425) が `ones(B,1,1,S)` を返し、その `S` は
`generate_audio`(同:550-557) 内で **差し替え後の enc から** 取られるので
長さが必ず一致する。`CrossAttention` に RoPE は無いので位置の振り直しも不要。

4 条件 (1 文あたり encoder forward 3 回 / decoder 生成 4 回)
------------------------------------------------------------
  bl     差し替えなし。`kiseki_splice.synthesize(text_a)` をそのまま呼ぶ
  inj1   枠 1 だけ差し替え。ref = text_refs[0] の encoder 出力
  inj2   枠 2 だけ差し替え。ref = text_refs[1] の encoder 出力
  inj12  両方差し替え。枠 i の ref は ref_i (**枠ごとに別の forward**)

inj12 で「両方置き換えた 1 本の文」から 2 区間を取らないのが要点。そうすると
干渉の原因 (splice の相互作用 vs. ref 側の文脈が既に汚れている) が分離できない。

合格条件 (単一区間での等価性)
-----------------------------
`--pairs_module ja_cross_culture_names` を渡すと 1 区間モードになり、
条件は bl / inj1 の 2 つだけになる。このとき

  * bl   : `kiseki_splice.synthesize(bundle, text_a, seed)` — 既存と同じ関数
  * inj1 : `build_spliced_hidden_multi` で 1 区間だけ差し替え
           -> `generate_audio(encoder_outputs=..., input_ids=None,
              attention_mask=None)` — 既存 `splice_synthesize` と同じ呼び出し

となり、`eval_results/_cc_wavs_a/kiseki_ja/s{i:04d}_inj.wav` と md5 が一致する
はず。`--compare_dir` を渡すとスクリプト自身が md5 を突き合わせて
`equivalence.json` を書く。

**区間の解決は Kiseki 自身の `find_entity_token_range` を使う (重要)**
----------------------------------------------------------------------
`splice_multi.resolve_spans_with_cursor` は offset_mapping ベースで、
Kiseki の既存 300 件が使った token-id 部分列探索
(`kiseki_splice.find_entity_token_range` = `injection.py:401` と同じ計算) とは
別のスパンを返すことがある (既存 300 件のうち 9 件が `char_prefix` 段に落ち、
11 件は `aligned=False` のまま生成されている)。単一区間で md5 一致させるには
既存と同じ関数を同じ引数で呼ぶ必要があるので、既定の resolver は
`kiseki` (= `find_entity_token_range`) にした。1 個目の区間 (cursor 0) は
`find_entity_token_range(tok, full_text, surface)` に**そのまま委譲**し、
2 個目以降だけ文字カーソル / トークンカーソル以降を探す。
`--span_resolver cursor` にすると `resolve_spans_with_cursor` を使う経路にも
切り替えられる。既定経路でも `resolve_spans_with_cursor` を診断として毎回走らせ、
一致したかどうかを `span_tok.cursor_resolver_agrees` に記録する。

また、既存の Kiseki 単一 entity 経路は往復不一致 (`aligned=False`) でも
**skip せずそのまま生成する** (kiseki_splice.py の設計方針 5)。ここでも
既定は同じで、`verify_span_roundtrip` の結果は記録するだけ。`--require_roundtrip`
を付けたときだけ T5Gemma 版と同じく文ごと skip する。

実行例 (GPU)
------------
  # 単一区間の md5 等価性 (n=5)
  SLURM_TIME=00:40:00 slurm_run.sh \\
    python3 -u /mnt/scripts/kiseki/run_multientity_kiseki.py \\
      --pairs_module ja_cross_culture_names --lang ja --n 5 \\
      --output_dir /mnt/eval_results/multientity_kiseki_ja_equiv_n5 \\
      --compare_dir /mnt/eval_results/_cc_wavs_a/kiseki_ja

  # 本番 (JA 300 件 / 2 区間)
  SLURM_TIME=08:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/kiseki/run_multientity_kiseki.py \\
      --pairs_module ja_multientity_300 --lang ja \\
      --output_dir /mnt/eval_results/multientity_kiseki_ja_300

  # 天井用の目標文だけを後から足す (--tgt_only、2026-09-22 追加)
  SLURM_TIME=08:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/kiseki/run_multientity_kiseki.py \\
      --pairs_module ja_multientity_300 --lang ja --tgt_only \\
      --output_dir /mnt/eval_results/multientity_kiseki_ja_300

  注入しないので区間の解決も往復確認もしない。既にある wav は作り直さないので、
  既存の bl/inj1/inj2/inj12 と results.json / checkpoint.json は動かない。
  帳面は checkpoint_tgt.json / results_tgt.json (別ファイル)。
  --dry_run を足すとモデルを読まずに、出すつもりの文とファイル名だけ並べる。

出力: <out_dir>/{results.json, checkpoint.json, s{pair_idx:04d}_{bl,inj1,inj2,inj12}.wav}
      --tgt_only のときは <out_dir>/{results_tgt.json, checkpoint_tgt.json,
      s{pair_idx:04d}_{tgt1,tgt2}.wav}
"""
import argparse
import hashlib
import json
import os
import sys
import time

# --------------------------------------------------------------------------
# HF cache: slurm_run.sh がコンテナに渡す HF_HOME は書けない場所を指すので
# **代入** で上書きする (setdefault では負ける)。transformers より先に済ませる。
# --------------------------------------------------------------------------
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "modules")
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, ".cache", "torch"))

sys.path.insert(0, os.path.join(ROOT, "scripts", "projection"))   # splice_multi/_clone
sys.path.insert(0, os.path.join(ROOT, "scripts", "kiseki"))       # kiseki_splice
sys.path.insert(0, os.path.join(ROOT, "data"))                    # ja_multientity_300
sys.path.insert(0, os.path.join(ROOT, "data"))
# 目標文だけを出す経路 (--tgt_only) の共通部品。stdlib しか読まない。
# **末尾に足す**ので、既存の import の解決順は 1 つも動かない。
sys.path.append(os.path.join(ROOT, "scripts", "multientity"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 既存 Kiseki 経路の部品。**再実装しない。**
from kiseki_splice import (  # noqa: E402
    DEFAULT_CODEC_ID,
    DEFAULT_LANG_TOKEN,
    DEFAULT_MAX_NEW_FRAMES,
    DEFAULT_MODEL_ID,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    SEED,
    _check_span_in_text,
    _codes_from_output,
    _decode_kwargs,
    codes_to_wav,
    describe_entity_span,
    encode_ids,
    encode_text,
    entity_cosine,
    find_entity_token_range,
    load_kiseki,
    seed_all,
    splice_synthesize,
    synthesize,
)
# wav の書き出しは既存 300 件を書いたのと同一関数 (PCM_16 / tmp -> rename)。
# ここを自作すると md5 が一致しない。
from run_kiseki_eval import atomic_json_dump, save_wav  # noqa: E402

# 複数区間 splice の本体。**再実装しない。**
from splice_multi import (  # noqa: E402
    build_spliced_hidden_multi,
    check_span_sets,
    resolve_spans_with_cursor,
    span_deltas,
    verify_span_roundtrip,
)

# --- 天井用の目標文だけを出す経路 (--tgt_only、2026-09-22 追加) -------------
from tgt_only import dry_run_report, run_targets  # noqa: E402

# 条件名 -> 使う枠 index (0-based)。T5Gemma 版 (generate_multientity.py:90-92) と同一。
COND_SLOTS = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}
CONDITIONS_BY_K = {1: ["bl", "inj1"], 2: ["bl", "inj1", "inj2", "inj12"]}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_items(pairs_module, n):
    """2 区間形式 (build_pairs) と 1 区間形式 (build_300_pairs) の両方を読む.

    1 区間形式 (`ja_cross_culture_names`) は md5 等価性試験の入力。
    `spans` を 1 個だけ持つ item に**そのまま**変換するだけで、
    複数 entity の経路に流し込む。
    """
    mod = __import__(pairs_module)
    if hasattr(mod, "build_pairs"):
        raw = mod.build_pairs()
        if raw and "spans" in raw[0]:
            for it in raw:
                assert len(it["spans"]) == 2, \
                    f"pair {it['pair_idx']}: expected exactly 2 spans"
                assert len(it["text_refs"]) == 2
            items = raw
            k = 2
        else:
            items, k = _as_single_span(raw), 1
    elif hasattr(mod, "build_300_pairs"):
        items, k = _as_single_span(mod.build_300_pairs()), 1
    else:
        raise SystemExit(f"data/{pairs_module}.py has neither build_pairs() nor "
                         f"build_300_pairs()")
    return (items[:n] if n else items), k


def _as_single_span(raw):
    """{text_a, text_b, word_a, word_b} を 1 区間の multientity item にする."""
    out = []
    for i, p in enumerate(raw):
        idx = p.get("pair_idx", i)
        out.append({
            "pair_idx": idx,
            "template_id": p.get("template_id"),
            "lang": p.get("lang"),
            "text_a": p["text_a"],
            "text_refs": [p["text_b"]],
            "text_both": p["text_b"],
            "gap_words": None,
            "slot1_at_start": p["text_a"].startswith(p["word_a"]),
            "spans": [{"slot": 1, "word_a": p["word_a"], "word_b": p["word_b"],
                       "char_start_a": p["text_a"].find(p["word_a"]),
                       "char_start_b": p["text_b"].find(p["word_b"])}],
        })
    return out


# --------------------------------------------------------------------------
# span resolution (Kiseki 自身の規則。1 個目は既存関数へ丸ごと委譲)
# --------------------------------------------------------------------------
def find_entity_token_range_after(tokenizer, full_text, entity_text,
                                  char_cursor=0, tok_cursor=0):
    """`find_entity_token_range` にカーソルを付けたもの.

    `char_cursor == 0 and tok_cursor == 0` のとき (= 1 個目の区間、および
    1 区間モードの唯一の区間) は `kiseki_splice.find_entity_token_range` に
    **同じ引数でそのまま委譲**する。だから 1 区間に縮退させたときの計算は
    既存の単一 entity 経路と文字通り同一になる。

    2 個目以降だけ、既存の 3 段構えを「カーソル以降」に限って繰り返す。
    段の順序・戻り値の形 (start, end, stage) は既存と同じ。**窓は広げない。**
    """
    if char_cursor == 0 and tok_cursor == 0:
        return find_entity_token_range(tokenizer, full_text, entity_text)

    full_tokens = encode_ids(tokenizer, full_text)
    entity_tokens = encode_ids(tokenizer, entity_text)

    # 1 段目: トークン id 列の部分列検索 (tok_cursor 以降)
    for start in range(max(0, tok_cursor), len(full_tokens) - len(entity_tokens) + 1):
        if full_tokens[start:start + len(entity_tokens)] == entity_tokens:
            return (start, start + len(entity_tokens), "subsequence")

    # 2 段目: 文字位置 (char_cursor 以降) から prefix をトークナイズ
    pos = full_text.find(entity_text, char_cursor)
    if pos >= 0:
        prefix_tokens = encode_ids(tokenizer, full_text[:pos])
        return (len(prefix_tokens), len(prefix_tokens) + len(entity_tokens),
                "char_prefix")

    # 3 段目: 大文字小文字を無視した文字検索 (日本語では発火しない)
    low = full_text.lower().find(entity_text.lower(), char_cursor)
    if low >= 0:
        actual = full_text[low:low + len(entity_text)]
        actual_tokens = encode_ids(tokenizer, actual)
        for start in range(max(0, tok_cursor),
                           len(full_tokens) - len(actual_tokens) + 1):
            if full_tokens[start:start + len(actual_tokens)] == actual_tokens:
                return (start, start + len(actual_tokens), "subsequence_ci")
        prefix_tokens = encode_ids(tokenizer, full_text[:low])
        return (len(prefix_tokens), len(prefix_tokens) + len(actual_tokens),
                "char_prefix_ci")
    return None


def resolve_nospec_spans(tokenizer, full_text, surfaces, offsets, resolver):
    """特殊トークン抜き座標で k 個の区間を左から順に解決する.

    返り値: (spans, stages, reason)。失敗時は (None, None, reason)。
    `resolver="kiseki"` が既定 (既存 300 件と同じ計算)。
    `resolver="cursor"` は `splice_multi.resolve_spans_with_cursor` 経路。
    """
    if resolver == "cursor":
        spans, info = resolve_spans_with_cursor(tokenizer, full_text, surfaces,
                                                add_bos=0)
        if spans is None:
            return None, None, f"cursor resolver failed: {info['reason']}"
        return [tuple(s) for s in spans], [p["via"] for p in info["per_span"]], None

    spans, stages = [], []
    char_cursor, tok_cursor = 0, 0
    for i, surf in enumerate(surfaces):
        if not surf:
            return None, None, f"span {i}: empty surface"
        found = find_entity_token_range_after(tokenizer, full_text, surf,
                                              char_cursor, tok_cursor)
        if found is None:
            return None, None, (f"span {i}: entity {surf!r} not found at or after "
                                f"char {char_cursor} in {full_text!r}")
        s, e, stage = int(found[0]), int(found[1]), found[2]
        if not (0 <= s < e):
            return None, None, f"span {i}: degenerate range ({s}, {e})"
        spans.append((s, e))
        stages.append(stage)
        tok_cursor = e
        # 次の区間の文字カーソル。offset_mapping が使えればそこから、
        # 使えなければ表層長ぶん進める (探索の下限にしか使わない)。
        if 0 <= s < e <= len(offsets):
            char_cursor = int(offsets[e - 1][1])
        else:
            char_cursor = char_cursor + len(surf)
    return spans, stages, None


def resolve_item_spans(bundle, item, src, ref_encs, resolver):
    """src 側 k 区間と ref 側 1 区間 x k を解決し、build_tts_input_ids 座標で返す.

    返り値: (s_ranges, r_ranges, detail) / 失敗時 (None, None, reason 文字列)
    """
    tok = bundle.tokenizer
    spans = item["spans"]
    k = len(spans)
    surfaces_a = [sp["word_a"] for sp in spans]

    # --- src: k 区間を「まとめて」左から順に解決 ---
    s_sub, s_stages, why = resolve_nospec_spans(tok, item["text_a"], surfaces_a,
                                                src["offsets"], resolver)
    if s_sub is None:
        return None, None, f"src span resolve failed: {why}"

    # --- ref: 枠ごとに別の forward。それぞれ 1 区間だけ解決 ---
    r_sub, r_stages = [], []
    for i, sp in enumerate(spans):
        sub, st, why = resolve_nospec_spans(tok, ref_encs[i]["text"],
                                            [sp["word_b"]],
                                            ref_encs[i]["offsets"], resolver)
        if sub is None:
            return None, None, f"ref[{i}] span resolve failed: {why}"
        r_sub.append(sub[0])
        r_stages.append(st[0])

    # --- 往復一致 (splice_multi.verify_span_roundtrip)。既定では記録のみ ---
    src_rt_ok, src_rt_bad = verify_span_roundtrip(tok, item["text_a"], s_sub,
                                                  surfaces_a, add_bos=0)
    ref_rt = [verify_span_roundtrip(tok, ref_encs[i]["text"], [r_sub[i]],
                                    [spans[i]["word_b"]], add_bos=0)
              for i in range(k)]
    ref_rt_ok = all(x[0] for x in ref_rt)

    # --- 診断: cursor resolver と一致するか (既定経路でも毎回測る) ---
    cur_spans, _ = resolve_spans_with_cursor(
        tok, item["text_a"], surfaces_a, add_bos=0,
        char_starts=[sp.get("char_start_a") for sp in spans])
    cursor_agrees = (cur_spans is not None
                     and [tuple(x) for x in cur_spans] == [tuple(x) for x in s_sub])

    # --- 特殊トークン抜き座標 -> build_tts_input_ids 座標 (offset を一度だけ足す) ---
    s_off = src["entity_offset"]
    s_ranges = [(a + s_off, b + s_off) for (a, b) in s_sub]
    r_ranges = []
    for i, (a, b) in enumerate(r_sub):
        r_off = ref_encs[i]["entity_offset"]
        r_ranges.append((a + r_off, b + r_off))

    # --- 区間が本文の範囲を出ていないか ([TTS]/lang/eos を潰さない) ---
    try:
        for i in range(k):
            _check_span_in_text(s_ranges[i], src, item["text_a"],
                                surfaces_a[i], f"src[{i}]")
            _check_span_in_text(r_ranges[i], ref_encs[i], ref_encs[i]["text"],
                                spans[i]["word_b"], f"ref[{i}]")
    except ValueError as e:
        return None, None, f"span leaves text region: {e}"

    # --- (c) 重なり検査。splice_multi.check_span_sets を生成前に一度呼ぶ ---
    if k >= 2 and s_ranges[0][1] > s_ranges[1][0]:
        return None, None, (f"src spans overlap: {s_ranges[0]} {s_ranges[1]} "
                            f"({surfaces_a})")
    try:
        check_span_sets(int(src["enc"].shape[1]),
                        [int(ref_encs[i]["enc"].shape[1]) for i in range(k)],
                        s_ranges, r_ranges)
    except AssertionError as e:
        return None, None, f"check_span_sets failed: {e}"

    per_span_delta, delta_all = span_deltas(s_ranges, r_ranges)
    detail = {
        "k": k,
        "resolver": resolver,
        "src": [list(x) for x in s_ranges],
        "ref": [list(x) for x in r_ranges],
        "src_nospec": [list(x) for x in s_sub],
        "ref_nospec": [list(x) for x in r_sub],
        "n": [s_ranges[i][1] - s_ranges[i][0] for i in range(k)],
        "m": [r_ranges[i][1] - r_ranges[i][0] for i in range(k)],
        "gap_tokens": (s_ranges[1][0] - s_ranges[0][1]) if k >= 2 else None,
        "src_target_start": int(s_off),
        "ref_target_start": [int(ref_encs[i]["entity_offset"]) for i in range(k)],
        "src_resolve_stage": s_stages,
        "ref_resolve_stage": r_stages,
        "delta_per_span": per_span_delta,
        "delta_all": int(delta_all),
        "len_imbalance": abs(detail_m_diff(s_ranges, r_ranges)) if k >= 2 else None,
        "src_roundtrip_ok": bool(src_rt_ok),
        "src_roundtrip_bad": src_rt_bad,
        "ref_roundtrip_ok": bool(ref_rt_ok),
        "ref_roundtrip_bad": [x[1] for x in ref_rt],
        "cursor_resolver_agrees": bool(cursor_agrees),
        "span_report_src": [
            describe_entity_span(tok, item["text_a"], surfaces_a[i],
                                 src["offsets"], s_sub[i], s_stages[i])
            for i in range(k)],
        "span_report_ref": [
            describe_entity_span(tok, ref_encs[i]["text"], spans[i]["word_b"],
                                 ref_encs[i]["offsets"], r_sub[i], r_stages[i])
            for i in range(k)],
    }
    detail["aligned"] = bool(
        all(x["aligned"] for x in detail["span_report_src"])
        and all(x["aligned"] for x in detail["span_report_ref"]))
    return s_ranges, r_ranges, detail


def detail_m_diff(s_ranges, r_ranges):
    m = [b - a for (a, b) in r_ranges]
    return m[0] - m[1]


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def generate_spliced(bundle, spliced_enc, seed, kw, to_wav=True):
    """差し替え済み encoder_out から生成する.

    `kiseki_splice.splice_synthesize`(kiseki_splice.py:735-740) の生成部分と
    **同じ呼び出し**。`input_ids` / `attention_mask` を両方 None にするのが必須。
    (b) seed は `generate_audio` の直前で毎回戻す。
    `to_wav=False` は CPU 検証用 (codec を読まずに codes だけ取る)。
    """
    from transformers.modeling_outputs import BaseModelOutput

    seed_all(seed)
    with torch.inference_mode():
        out = bundle.model.generate_audio(
            encoder_outputs=BaseModelOutput(last_hidden_state=spliced_enc),
            input_ids=None, attention_mask=None, **kw)
    codes = _codes_from_output(out)
    res = {
        "codes": codes,
        "n_frames": int(codes.shape[0]),
        "used_seed": int(seed),
        "decode_kwargs": kw,
        "hit_frame_cap": bool(int(out["audio_codes"].shape[1])
                              >= int(kw["max_new_frames"])),
    }
    if to_wav:
        wav, sr = codes_to_wav(bundle, codes)
        res["audio"] = wav
        res["sample_rate"] = sr
    return res


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")

    items, k = load_items(args.pairs_module, args.n)
    conditions = CONDITIONS_BY_K[k]
    print(f"[data] {args.pairs_module}: {len(items)} items, k={k} span(s), "
          f"conds={conditions}, lang={args.lang}", flush=True)
    print(f"[data] first: {items[0]['text_a']}", flush=True)
    print(f"[env ] HF_HOME={os.environ['HF_HOME']}", flush=True)

    if args.tgt_only and k != 2:
        raise SystemExit("--tgt_only は 2 枠のデータ専用 (今のデータは k=%d)" % k)
    if args.dry_run:
        # モデルを読む前に返す (CPU で名前と文を確かめるための経路)。
        dry_run_report(items, args.output_dir, tag="kiseki")
        return

    print(f"[load] {args.model_id} precision={args.precision} device={args.device}",
          flush=True)
    t_load = time.time()
    bundle = load_kiseki(model_id=args.model_id, device=args.device,
                         precision=args.precision, lang_token=args.lang_token,
                         codec_id=args.codec_id,
                         codec_device=args.codec_device or args.device)
    print(f"[load] done in {time.time() - t_load:.1f}s sr={bundle.codec_sr} "
          f"d_model={bundle.meta['d_model']} enc_layers={bundle.meta['enc_layers']}",
          flush=True)

    if args.tgt_only:
        # 注入しないので区間の解決も往復確認もしない。合成は bl 条件
        # (`synthesize`) と同じ関数・同じ decode 引数で、渡す文だけが
        # text_refs[slot] に変わる。
        kw_tgt = _decode_kwargs(args.max_new_frames, args.temperature,
                                args.top_k, args.top_p, args.repetition_penalty)

        def _synth_tgt(text, path):
            seed_all(args.seed)
            gen = synthesize(bundle, text, seed=args.seed, to_wav=True, **kw_tgt)
            _, dur, clip = save_wav(path, gen["audio"], gen["sample_rate"])
            return {"dur_sec": round(float(dur), 4),
                    "n_frames": int(gen["n_frames"]),
                    "hit_frame_cap": bool(gen["hit_frame_cap"]),
                    "clip_ratio": round(float(clip), 6),
                    "used_seed": int(gen["used_seed"]),
                    "md5": md5_file(path)}

        run_targets(items, args.output_dir, _synth_tgt, tag="kiseki",
                    ckpt_every=args.ckpt_every,
                    extra_summary={"method": "kiseki", "lang": args.lang,
                                   "pairs_module": args.pairs_module,
                                   "model_id": args.model_id,
                                   "precision": args.precision,
                                   "seed": int(args.seed),
                                   "decode_kwargs": kw_tgt})
        return

    utmos = None
    if args.utmos:
        try:
            utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                   trust_repo=True).to(args.device).eval()
            print("[load] UTMOS22 ok", flush=True)
        except Exception as e:
            print(f"[warn] UTMOS not loaded ({type(e).__name__}: {e}); "
                  f"utmos will be null", flush=True)

    rows, skipped = [], []
    if os.path.exists(ckpt_path) and not args.overwrite:
        st = json.load(open(ckpt_path))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != "splice_multi_kiseki"]
        assert not bad, (f"checkpoint mode mismatch in {ckpt_path}: pairs "
                         f"{bad[:5]} were not produced by this script; use a "
                         f"separate --output_dir")
        print(f"[resume] {len(rows)} rows, {len(skipped)} skipped", flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}

    todo = items
    if args.num_shards > 1:
        todo = [p for i, p in enumerate(items)
                if i % args.num_shards == args.shard_index]
        print(f"[shard] {args.shard_index}/{args.num_shards}: {len(todo)} items",
              flush=True)

    kw = _decode_kwargs(args.max_new_frames, args.temperature, args.top_k,
                        args.top_p, args.repetition_penalty)
    t0 = time.time()

    for it in todo:
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        text_a = it["text_a"]
        refs = it["text_refs"]

        # --- encoder forward: src 1 回 + ref k 回 (枠ごとに別の forward) ---
        try:
            src = encode_text(bundle, text_a, collect_hidden=not args.no_kappa)
            ref_encs = [encode_text(bundle, r, collect_hidden=not args.no_kappa)
                        for r in refs]
        except Exception as e:
            print(f"  [{pidx}] enc fail: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"enc: {e}"[:400]})
            done.add(pidx)
            continue

        s_ranges, r_ranges, detail = resolve_item_spans(bundle, it, src, ref_encs,
                                                        args.span_resolver)
        if s_ranges is None:
            print(f"  [{pidx}] SKIP {detail}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": detail})
            done.add(pidx)
            continue
        if args.require_roundtrip and not (detail["src_roundtrip_ok"]
                                           and detail["ref_roundtrip_ok"]):
            r = (f"roundtrip failed (src={detail['src_roundtrip_bad']}, "
                 f"ref={detail['ref_roundtrip_bad']})")
            print(f"  [{pidx}] SKIP {r}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": r})
            done.add(pidx)
            continue

        row = {
            "pair_idx": pidx, "mode": "splice_multi_kiseki",
            "template_id": it.get("template_id"), "lang": args.lang,
            "text_a": text_a, "text_refs": refs, "text_both": it["text_both"],
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "repeat_surface": it.get("repeat_surface", False),
            "words_a": [sp["word_a"] for sp in it["spans"]],
            "words_b": [sp["word_b"] for sp in it["spans"]],
            "span_tok": detail,
            "span_aligned": detail["aligned"],
            "T_src": int(src["enc"].shape[1]),
            "n_tokens_ref": [int(e["enc"].shape[1]) for e in ref_encs],
            "conds": {},
        }
        if not args.no_kappa:
            row["kappa_encoder_out"] = [
                entity_cosine(src["enc"], ref_encs[i]["enc"], s_ranges[i],
                              r_ranges[i]) for i in range(len(s_ranges))]
            if src["hidden_states"] is not None:
                row["kappa_by_layer"] = [
                    [entity_cosine(sh, th, s_ranges[i], r_ranges[i])
                     for sh, th in zip(src["hidden_states"],
                                       ref_encs[i]["hidden_states"])]
                    for i in range(len(s_ranges))]

        ok = True
        for cond in conditions:
            slots = COND_SLOTS[cond]
            try:
                if not slots:
                    # BL は既存単一 entity 経路と同一関数をそのまま呼ぶ
                    seed_all(args.seed)   # (b) synthesize 内でも直前に呼ばれる
                    gen = synthesize(bundle, text_a, seed=args.seed, to_wav=True,
                                     **kw)
                    delta_tok, out_ranges = 0, []
                    t_out = row["T_src"]
                    fed_text = text_a
                else:
                    sub_s = [s_ranges[i] for i in slots]
                    sub_r = [r_ranges[i] for i in slots]
                    sub_ref = [{"last_hidden_state": ref_encs[i]["enc"]}
                               for i in slots]
                    # ★ 差し替えの本体。自分でずらさない (splice_multi が一括
                    #    concat で 2 個目の index のずれを構造的に吸収する)。
                    spliced, delta_tok, out_ranges = build_spliced_hidden_multi(
                        {"last_hidden_state": src["enc"]}, sub_ref, sub_s, sub_r)
                    t_out = int(spliced.shape[1])
                    gen = generate_spliced(bundle, spliced, args.seed, kw)
                    fed_text = None
                    del spliced
            except Exception as e:
                import traceback
                print(f"  [{pidx}] {cond} fail: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                traceback.print_exc()
                skipped.append({"pair_idx": pidx, "reason": f"{cond}: {e}"[:400]})
                ok = False
                break

            wav_path = os.path.join(args.output_dir, f"s{pidx:04d}_{cond}.wav")
            try:
                n_s, dur, clip = save_wav(wav_path, gen["audio"],
                                          gen["sample_rate"])
                u = None
                if utmos is not None:
                    u = score_utmos(utmos, gen["audio"], gen["sample_rate"],
                                    args.device)
                row["conds"][cond] = {
                    "wav": os.path.basename(wav_path),
                    "delta_tok": int(delta_tok),
                    "T_out": int(t_out),
                    "out_ranges": [list(x) for x in out_ranges],
                    "target_text": surface_text_for(it, cond),
                    "fed_input_text": fed_text,
                    "input_ids_none": bool(slots),
                    "dur_sec": round(float(dur), 4),
                    "n_frames": int(gen["n_frames"]),
                    "hit_frame_cap": bool(gen["hit_frame_cap"]),
                    "clip_ratio": round(float(clip), 6),
                    "used_seed": int(gen["used_seed"]),
                    "utmos": u,
                    "md5": md5_file(wav_path),
                }
            except Exception as e:
                print(f"  [{pidx}] {cond} save/score fail: {str(e)[:200]}",
                      flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} save: {e}"[:400]})
                ok = False
                break

        # --- 同一ジョブ内での等価性 self-test (k=1 のときだけ) ---
        # 既存の単一 entity 経路 `kiseki_splice.splice_synthesize` を同じ GPU・
        # 同じプロセスで走らせて md5 を突き合わせる。別ジョブ / 別 GPU との
        # 突き合わせ (--compare_dir) に付きまとう浮動小数点の揺れの疑いを
        # 完全に外せるので、移植の正しさはこちらが決定的。
        if ok and args.selftest_existing and k == 1:
            try:
                sp = it["spans"][0]
                seed_all(args.seed)
                legacy = splice_synthesize(
                    bundle, text_a, sp["word_a"], refs[0], sp["word_b"],
                    seed=args.seed, collect_kappa=False, to_wav=True, **kw)
                lp = os.path.join(args.output_dir, f"s{pidx:04d}_inj_legacy.wav")
                save_wav(lp, legacy["audio"], legacy["sample_rate"])
                row["selftest_legacy"] = {
                    "wav": os.path.basename(lp),
                    "md5": md5_file(lp),
                    "matches_inj1": md5_file(lp) == row["conds"]["inj1"]["md5"],
                    "s_span": legacy["s_span"], "t_span": legacy["t_span"],
                    "delta_tok": legacy["delta_tok"],
                    "n_frames": legacy["n_frames"],
                }
                if not row["selftest_legacy"]["matches_inj1"]:
                    print(f"  [{pidx}] SELFTEST MISMATCH vs splice_synthesize "
                          f"(legacy s_span={legacy['s_span']} "
                          f"mine={detail['src']})", flush=True)
            except Exception as e:
                print(f"  [{pidx}] selftest fail: {type(e).__name__}: "
                      f"{str(e)[:200]}", flush=True)
                row["selftest_legacy"] = {"error": f"{type(e).__name__}: {e}"[:300]}

        done.add(pidx)
        del src, ref_encs
        if not ok:
            continue
        # Kiseki は長さ正規化の position ランプを持たない (CrossAttention に RoPE が
        # 無い) ので、この値は「交差注意の参照長 S の変化量」を意味する。
        last = conditions[-1]
        row["total_position_drift"] = row["conds"][last]["delta_tok"]
        rows.append(row)

        if len(rows) % 5 == 0:
            print(f"  [{len(rows)}/{len(todo)}] idx={pidx} "
                  f"gap_tok={detail['gap_tokens']} "
                  f"delta={row['total_position_drift']} "
                  f"dur=" + "/".join(f"{row['conds'][c]['dur_sec']:.2f}"
                                     for c in conditions), flush=True)
        if len(rows) % args.ckpt_every == 0:
            atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt_path)

    atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt_path)

    import transformers
    summary = {
        "pairs_module": args.pairs_module, "lang": args.lang,
        "k_spans": k, "conditions": conditions,
        "mode": "splice_multi_kiseki",
        "method": "kiseki",
        "splice_site": ("text encoder final RMSNorm output (encoder_out), fed to "
                        "generate_audio via encoder_outputs with input_ids=None "
                        "and attention_mask=None so _build_cross_mask returns "
                        "ones(B,1,1,S) with S taken from the spliced tensor "
                        "(modeling_mamba2_s2s.py:419-425, 550-557)"),
        "span_resolver": args.span_resolver,
        "require_roundtrip": bool(args.require_roundtrip),
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "n_span_misaligned": sum(1 for r in rows if r.get("span_aligned") is False),
        "n_cursor_resolver_disagree": sum(
            1 for r in rows if r["span_tok"].get("cursor_resolver_agrees") is False),
        "selftest_existing": _selftest_summary(rows) if args.selftest_existing
        else None,
        "model_id": args.model_id, "codec_id": args.codec_id,
        "precision": args.precision, "device": args.device,
        "seed": args.seed, "decode": kw,
        "sampling_rate": bundle.codec_sr,
        "gpu_name": (torch.cuda.get_device_name(0)
                     if str(args.device).startswith("cuda")
                     and torch.cuda.is_available() else None),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "mean_dur_sec": {c: round(float(np.mean(
            [r["conds"][c]["dur_sec"] for r in rows])), 4) for c in conditions}
        if rows else {},
        "mean_abs_position_drift": round(float(np.mean(
            [abs(r["total_position_drift"]) for r in rows])), 3) if rows else None,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    if utmos is not None and rows:
        summary["utmos"] = {}
        for c in conditions:
            v = [r["conds"][c]["utmos"] for r in rows
                 if r["conds"].get(c, {}).get("utmos") is not None]
            summary["utmos"][c] = [round(float(np.mean(v)), 4),
                                   round(float(np.std(v)), 4)] if v else [None, None]

    atomic_json_dump({"summary": summary, "rows": rows, "skipped": skipped},
                     os.path.join(args.output_dir, "results.json"))
    print(f"\n=== multi-entity generation (kiseki) {args.pairs_module} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if skipped:
        print(f"[warn] {len(skipped)} items skipped; see results.json['skipped']")

    if args.selftest_existing and summary.get("selftest_existing"):
        print("\n=== in-job self-test vs kiseki_splice.splice_synthesize ===")
        print(json.dumps(summary["selftest_existing"], ensure_ascii=False, indent=1))
    if args.compare_dir:
        compare_md5(args, rows, conditions)
    return rows


def _selftest_summary(rows):
    """同一ジョブ内 self-test の集計 (k=1 のときだけ意味がある)."""
    have = [r["selftest_legacy"] for r in rows if "selftest_legacy" in r]
    n_match = sum(1 for x in have if x.get("matches_inj1") is True)
    n_bad = sum(1 for x in have if x.get("matches_inj1") is False)
    n_err = sum(1 for x in have if "error" in x)
    return {"n": len(have), "n_match": n_match, "n_mismatch": n_bad,
            "n_error": n_err,
            "verdict": "PASS" if (have and n_bad == 0 and n_err == 0) else "FAIL",
            "note": ("s{i:04d}_inj1.wav (build_spliced_hidden_multi, k=1) vs "
                     "s{i:04d}_inj_legacy.wav (kiseki_splice.splice_synthesize) "
                     "generated in the same process on the same GPU")}


def surface_text_for(it, cond):
    """その条件が文字レベルで相当する文 (Kiseki は INJ で文字列を渡さない)."""
    if cond == "bl":
        return it["text_a"]
    if cond == "inj1":
        return it["text_refs"][0]
    if cond == "inj2":
        return it["text_refs"][1]
    return it["text_both"]


def score_utmos(utmos, audio, sr, device):
    try:
        import torchaudio
        w = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        with torch.inference_mode():
            return round(float(utmos(w.to(device), 16000).item()), 4)
    except Exception:
        return None


def compare_md5(args, rows, conditions):
    """単一区間モードでの md5 等価性を突き合わせる.

    <out_dir>/s{i:04d}_inj1.wav  vs  <compare_dir>/s{i:04d}_inj.wav
    <out_dir>/s{i:04d}_bl.wav    vs  <compare_dir>/s{i:04d}_bl.wav
    """
    pair_of = {"bl": "bl", "inj1": "inj"}
    recs, n_ok, n_bad, n_miss = [], 0, 0, 0
    for r in rows:
        pidx = r["pair_idx"]
        rec = {"pair_idx": pidx}
        for cond in conditions:
            if cond not in pair_of:
                continue
            mine = os.path.join(args.output_dir, f"s{pidx:04d}_{cond}.wav")
            theirs = os.path.join(args.compare_dir,
                                  f"s{pidx:04d}_{pair_of[cond]}.wav")
            if not (os.path.exists(mine) and os.path.exists(theirs)):
                rec[cond] = {"status": "missing", "mine": os.path.exists(mine),
                             "ref": os.path.exists(theirs)}
                n_miss += 1
                continue
            a, b = md5_file(mine), md5_file(theirs)
            rec[cond] = {"status": "match" if a == b else "MISMATCH",
                         "mine_md5": a, "ref_md5": b,
                         "mine_bytes": os.path.getsize(mine),
                         "ref_bytes": os.path.getsize(theirs)}
            if a == b:
                n_ok += 1
            else:
                n_bad += 1
        recs.append(rec)
    out = {"summary": {"compare_dir": args.compare_dir,
                       "output_dir": args.output_dir,
                       "n_match": n_ok, "n_mismatch": n_bad,
                       "n_missing": n_miss,
                       "verdict": "PASS" if (n_bad == 0 and n_miss == 0 and n_ok)
                       else "FAIL"},
           "rows": recs}
    atomic_json_dump(out, os.path.join(args.output_dir, "equivalence.json"))
    print("\n=== single-span md5 equivalence ===")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
    for rec in recs:
        for cond, v in rec.items():
            if cond == "pair_idx" or not isinstance(v, dict):
                continue
            if v["status"] != "match":
                print(f"  [{rec['pair_idx']}] {cond}: {v}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs_module", required=True,
                   help="data/<name>.py。build_pairs() の 2 区間形式、または "
                        "build_300_pairs() の 1 区間形式 (等価性試験用)")
    p.add_argument("--lang", default="ja", choices=["ja"],
                   help="Kiseki-TTS は日本語専用 (<|2ja|> が唯一の言語タグ)")
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--compare_dir", default=None,
                   help="単一区間モードで md5 を突き合わせる既存 wav ディレクトリ "
                        "(eval_results/_cc_wavs_a/kiseki_ja)")
    p.add_argument("--span_resolver", default="kiseki", choices=["kiseki", "cursor"],
                   help="kiseki=find_entity_token_range (既定、既存 300 件と同一)")
    p.add_argument("--require_roundtrip", action="store_true",
                   help="往復不一致の文を skip する (既定は Kiseki 流で記録のみ)")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--codec_id", default=DEFAULT_CODEC_ID)
    p.add_argument("--lang_token", default=DEFAULT_LANG_TOKEN)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--codec_device", default=None)
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"],
                   help="既存 300 件は fp32。等価性試験では変えないこと")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max_new_frames", type=int, default=DEFAULT_MAX_NEW_FRAMES)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--repetition_penalty", type=float,
                   default=DEFAULT_REPETITION_PENALTY)
    p.add_argument("--selftest_existing", action="store_true",
                   help="k=1 のとき既存 splice_synthesize も同じジョブで走らせて "
                        "md5 を突き合わせる (移植の正しさの決定的な試験)")
    p.add_argument("--no_kappa", action="store_true")
    p.add_argument("--utmos", action="store_true", help="UTMOS22 も測る")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ckpt_every", type=int, default=10)
    # **後から目標文だけを足すための経路 (2026-09-22)。**既定は off なので
    # 既存の呼び出しはこの 2 つを一切通らない。
    p.add_argument("--tgt_only", action="store_true",
                   help="枠ごとの目標文 tgt1 / tgt2 だけを合成する (注入なし)。"
                        "既存の bl/inj1/inj2/inj12 と results.json は触らない")
    p.add_argument("--dry_run", action="store_true",
                   help="モデルを読まずに、出すつもりの文とファイル名を並べる")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(ROOT, "eval_results",
                                       f"multientity_kiseki_{args.lang}_300")
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(ROOT, args.output_dir)
    if args.compare_dir and not os.path.isabs(args.compare_dir):
        args.compare_dir = os.path.join(ROOT, args.compare_dir)
    run(args)


if __name__ == "__main__":
    main()
