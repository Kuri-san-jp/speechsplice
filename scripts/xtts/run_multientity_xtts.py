#!/usr/bin/env python3
"""XTTS-v2: 1 文の中の 2 箇所を同時に差し替える (BL / INJ_1 / INJ_2 / INJ_12).

T5Gemma 版 `scripts/analysis/generate_multientity.py` の構成をそのまま写した
**新規**スクリプト。既存ファイル (`scripts/xtts/xtts_splice.py`,
`scripts/xtts/run_xtts_eval.py`, `scripts/projection/splice_multi.py`) は
1 行も変更していない。再利用できる部品はすべて import する:

    xtts_splice   : load_xtts / load_reference / prepare_text /
                    prefill_hidden_states / _generate_codes / _codes_to_wav /
                    synthesize / seed_all / prefill_length / latent_pass_length
    splice_multi  : build_spliced_hidden_multi / check_span_sets /
                    resolve_spans_with_cursor / verify_span_roundtrip / span_deltas

XTTS-v2 に固有の事情
--------------------
1. **系列長を変えない。** decoder-only なので置換で長さを変えると後続の位置と
   KV cache がずれる。既存の単一 entity 実装 (`xtts_splice.RowSplice`) は
     m == n : 行対行で置換
     m != n : 目標スパンをトークン方向に平均した 1 本を m 行に broadcast
   で長さを保つ。ここでも**同じ規則**で「長さ m_i に揃えた置換行」を先に作り、
   それを splice_multi.build_spliced_hidden_multi に ref として渡す。
   r_range の長さが s_range の長さと必ず等しいので delta_tok は常に 0 になり、
   out_ranges は s_ranges と一致する。**concat そのものは再実装しない。**

2. **2 番目の区間の index はずれない。** (1) の帰結として置換で長さが変わらず、
   さらに build_spliced_hidden_multi は src の「元の」index だけを読んで 1 回の
   concat で組み立てる。したがって呼び出し側は index を自分でずらさない
   (ずらしてはいけない)。

3. **GPT2 trunk を 2 回通る。** pass 1 = gpt.generate の prefill (mel code の
   自己回帰生成)、pass 2 = gpt(..., return_latent=True) (HiFiGAN 用 latent)。
   片方だけ置換すると介入が半分打ち消される (xtts_splice.py:50-58)。
   複数 entity でも**両方**に同じ置換を掛ける (既定)。
   どちらの pass を狙うかは系列長で名指しする (expect_len)。

4. **英語のみ。** 日本語は cutlet で romaji 化、中国語は pypinyin で拼音化され、
   読みを直接書き込む入口になるので前提が壊れる (xtts_splice.py:44-48)。

出力 (採点は scripts/analysis/score_multientity.py がそのまま読む)
------------------------------------------------------------------
    <out_dir>/results.json     summary + rows (conds: bl / inj1 / inj2 / inj12)
    <out_dir>/checkpoint.json  途中再開用
    <out_dir>/s{pair_idx:04d}_{bl,inj1,inj2,inj12}.wav
    <out_dir>/s{pair_idx:04d}_{tgt1,tgt2}.wav   (--tgt_only のとき。天井の採点は
                                                 score_multientity_ceiling.py)

単一区間の等価性 (合格条件)
---------------------------
`--pairs_module en_cross_culture_names` を渡すと 1 区間 (k=1) の item に変換し、
条件は bl / inj1 の 2 つになる。このとき既存の単一 entity 本番
(`run_xtts_eval.py`) と**呼び出し列**までそろえる:
  * 生成の順序を bl -> tgt -> inj にする (s{i:04d}_tgt.wav も出す)
  * inj の前に src 側 prefill を 1 回空回しする (既存は kappa 用に必ず通る)
`--md5_check_against` に既存の wav ディレクトリを渡すと、生成した
s{i:04d}_{bl,tgt,inj1}.wav と既存の s{i:04d}_{bl,tgt,inj}.wav の md5 を
照合して PASS/FAIL を出す。

使い方 (GPU)
------------
  # 単一区間の md5 等価性 (6 対。m!=n と m==n の両方の枝を踏む)
  IMAGE_NAME=xtts-tts:latest SLURM_TIME=01:00:00 \\
  slurm_run.sh \\
    python3 -u /mnt/scripts/xtts/run_multientity_xtts.py \\
      --pairs_module en_cross_culture_names --lang en --layer 28 \\
      --pair_ids 0,1,2,180,181,220 --stop_on_error \\
      --output_dir /mnt/eval_results/_me_xtts_k1check \\
      --md5_check_against /mnt/eval_results/_cc_wavs_a/xtts_en

  # 本番 (EN 300 件 x 4 条件)
  IMAGE_NAME=xtts-tts:latest SLURM_TIME=04:00:00 \\
  slurm_run.sh \\
    python3 -u /mnt/scripts/xtts/run_multientity_xtts.py \\
      --pairs_module en_multientity_300 --lang en --n 300 --layer 28 \\
      --output_dir /mnt/eval_results/multientity_xtts_en_300

  # 天井用の目標文だけ (--tgt_only、2026-09-22 追加)
  IMAGE_NAME=xtts-tts:latest SLURM_TIME=03:00:00 \\
  slurm_run.sh \\
    python3 -u /mnt/scripts/xtts/run_multientity_xtts.py \\
      --pairs_module en_multientity_300 --lang en --n 300 --layer 28 \\
      --tgt_only \\
      --output_dir /mnt/eval_results/multientity_xtts_en_300

  注入しないので区間の解決も往復確認もしない。既にある wav は作り直さないので、
  先に本番 (4 条件) を回してから足しても既存の音声と results.json は動かない。
  帳面は checkpoint_tgt.json / results_tgt.json (別ファイル)。
  --dry_run を足すとモデルを読まずに、出すつもりの文とファイル名だけ並べる。
"""
import argparse
import hashlib
import json
import os
import sys
import time

# HF_HOME は import より前に **代入** して上書きする (setdefault は効かない)。
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "modules")

PROJECT_ROOT = ROOT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "xtts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
# 目標文だけを出す経路 (--tgt_only) の共通部品。stdlib しか読まない。
# **末尾に足す**ので、既存の import の解決順は 1 つも動かない。
sys.path.append(os.path.join(PROJECT_ROOT, "scripts", "multientity"))

import xtts_splice as X  # noqa: E402
from splice_multi import (  # noqa: E402
    build_spliced_hidden_multi, check_span_sets, resolve_spans_with_cursor,
    span_deltas, verify_span_roundtrip,
)
# --- 天井用の目標文だけを出す経路 (--tgt_only、2026-09-22 追加) -------------
from tgt_only import dry_run_report, run_targets  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_REF_WAV = os.path.join(PROJECT_ROOT, "eval_results", "_natural_refs_norm",
                               "en_natural_norm.wav")
# 既存の単一 entity 本番と同じ層 (eval_results/xtts_cross_culture_en/
# xtts_en_L28_t0.75_p0.85_manifest.json: layer 28 / hidden_index 29)
DEFAULT_LAYER = 28

CONDITIONS_K2 = ["bl", "inj1", "inj2", "inj12"]
COND_SLOTS_K2 = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}
CONDITIONS_K1 = ["bl", "inj1"]
COND_SLOTS_K1 = {"bl": [], "inj1": [0]}


# --------------------------------------------------------------------------
# tokenizer アダプタ (splice_multi のスパン解決に必要な最小の面)
# --------------------------------------------------------------------------
class BpeOffsetAdapter:
    """VoiceBpeTokenizer を HF fast tokenizer 風の呼び出しに見せるだけのアダプタ.

    `splice_multi.resolve_spans_with_cursor` / `verify_span_roundtrip` が使うのは
    ``tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]``
    だけ。中身は既存の `xtts_splice.entity_token_span` (xtts_splice.py:470-472) が
    使っているものと同一 (``tokenizer.tokenizer.encode(s)`` の ids / offsets)。
    ここで文字オフセットを渡さない (char_starts=None) のは、XTTS の BPE 入力が
    ``[en]`` + 小文字化 + 空白 -> ``[SPACE]`` を経た別の文字列で、データ側の
    char_start_a / char_start_b が指す位置とずれるため。同じ語が 2 回出る場合は
    resolve_spans_with_cursor の文字カーソルが左から順に解決する。
    """

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        enc = self.tok.tokenizer.encode(text)
        return {"input_ids": list(enc.ids),
                "offset_mapping": [(int(a), int(b)) for a, b in enc.offsets]}


# --------------------------------------------------------------------------
# k 区間の splice hook
# --------------------------------------------------------------------------
class MultiRowSplice:
    """gpt.gpt.h[l] の出力の、k 個の行区間を同時に差し替える forward hook.

    発火条件は既存の単一区間版 `xtts_splice.RowSplice` (xtts_splice.py:873-967)
    と同一:
      1. h.shape[1] <= 1                     -> decode ステップ。何もしない。
      2. expect_len があり h.shape[1] != それ -> 別 pass。見送る。
      3. 既に発火済み                         -> 二度と発火しない。

    置換行の作り方も既存と同一 (RowSplice._replacement, xtts_splice.py:914-919):
      m_i == n_i : 目標スパンをそのまま
      m_i != n_i : 目標スパンのトークン方向平均を m_i 行に broadcast
    したがって置換行の長さは必ず m_i で、**系列長は変わらない**。

    組み立ては `splice_multi.build_spliced_hidden_multi` に任せる (一括 concat)。
    k=1 のとき、その出力は RowSplice の `h.clone(); h[0,r0:r1] = rep` と
    同じ値・同じ形・同じ contiguous になる (どちらも純粋なコピーで演算を挟まない)。
    """

    def __init__(self, layer_module, s_rows, tgt_rows_list, expect_len=None,
                 name="prefill"):
        assert len(s_rows) == len(tgt_rows_list) >= 1
        self.layer_module = layer_module
        self.s_rows = [(int(a), int(b)) for a, b in s_rows]
        self.tgt_rows_list = [t.detach() for t in tgt_rows_list]
        self.expect_len = int(expect_len) if expect_len is not None else None
        self.name = name
        self.handle = None
        self.fired = 0
        self.decode_calls = 0
        self.skipped_shapes = []
        self.out_ranges = []
        self.delta_tok = None
        self.observed_len = None
        self.m = [b - a for a, b in self.s_rows]
        self.n = [int(t.shape[0]) for t in self.tgt_rows_list]
        self.equal_len = [bool(m == n) for m, n in zip(self.m, self.n)]
        for i, (m, n) in enumerate(zip(self.m, self.n)):
            if m <= 0:
                raise ValueError(f"span {i}: empty source span {self.s_rows[i]}")
            if n <= 0:
                raise ValueError(f"span {i}: empty target hidden span")

    def _replacements(self, dtype, device):
        """長さを m_i に揃えた置換行 (RowSplice._replacement と同一規則)."""
        reps = []
        for t, m, eq in zip(self.tgt_rows_list, self.m, self.equal_len):
            rep = t if eq else t.mean(dim=0, keepdim=True).expand(m, -1)
            reps.append(rep.to(dtype=dtype, device=device))
        return reps

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            return output
        T = h.shape[1]
        if T <= 1:
            self.decode_calls += 1
            return output
        if self.expect_len is not None and T != self.expect_len:
            self.skipped_shapes.append(int(T))
            return output
        if self.fired:
            self.skipped_shapes.append(int(T))
            return output
        hi = max(b for _, b in self.s_rows)
        if hi > T:
            raise RuntimeError(f"entity span {self.s_rows} が系列長 {T} を超えている")
        if h.shape[0] != 1:
            raise RuntimeError(f"batch size {h.shape[0]} は未対応 (1 のみ)")

        reps = self._replacements(h.dtype, h.device)
        # 長さを揃えてあるので r_range の長さ == s_range の長さ。delta_tok は 0。
        r_ranges = [(0, int(r.shape[0])) for r in reps]
        h_mod, delta_tok, out_ranges = build_spliced_hidden_multi(
            {"last_hidden_state": h},
            [{"last_hidden_state": r.unsqueeze(0)} for r in reps],
            self.s_rows, r_ranges)
        if delta_tok != 0 or tuple(h_mod.shape) != tuple(h.shape):
            raise RuntimeError(
                f"XTTS では系列長を変えられない: delta_tok={delta_tok} "
                f"{tuple(h.shape)} -> {tuple(h_mod.shape)}")
        if list(out_ranges) != [tuple(x) for x in self.s_rows]:
            raise RuntimeError(f"out_ranges {out_ranges} != s_rows {self.s_rows}")
        self.out_ranges = [list(x) for x in out_ranges]
        self.delta_tok = int(delta_tok)
        self.observed_len = int(T)
        self.fired += 1
        if isinstance(output, tuple):
            return (h_mod,) + tuple(output[1:])
        return h_mod

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def assert_fired(self, want=1):
        if self.fired != want:
            raise RuntimeError(
                f"splice hook[{self.name}] の発火回数が {self.fired} ({want} であるべき)。"
                f" expect_len={self.expect_len} skipped_shapes={self.skipped_shapes[:8]}")

    def stats(self):
        return {"name": self.name, "fired": self.fired,
                "decode_calls": self.decode_calls,
                "skipped_shapes": self.skipped_shapes[:16],
                "expect_len": self.expect_len, "observed_len": self.observed_len,
                "rows": [list(x) for x in self.s_rows],
                "m_src_entity_tokens": self.m, "n_tgt_entity_tokens": self.n,
                "equal_len": self.equal_len, "delta_tok": self.delta_tok,
                "out_ranges": self.out_ranges,
                "splice_mode": ["token_wise" if e else "mean_pool_broadcast"
                                for e in self.equal_len]}


# --------------------------------------------------------------------------
# データ
# --------------------------------------------------------------------------
def load_items(pairs_module, n, lang):
    """2 区間モジュール (build_pairs) と cross-culture (build_300_pairs) の両対応.

    cross-culture は k=1 の item に変換する。**単一区間の等価性試験専用**で、
    ここを通っても経路は複数 entity 用のものと 1 行も変わらない。
    """
    mod = __import__(pairs_module)
    if hasattr(mod, "build_pairs"):
        items = mod.build_pairs()
        for it in items:
            assert len(it["spans"]) == 2, \
                f"pair {it['pair_idx']}: 2 区間の形式であること"
            assert len(it["text_refs"]) == 2
        k = 2
    elif hasattr(mod, "build_300_pairs"):
        raw = mod.build_300_pairs()
        items = []
        for i, p in enumerate(raw):
            items.append({
                "pair_idx": p.get("pair_idx", i),
                "template_id": "cross_culture",
                "lang": lang,
                "text_a": p["text_a"],
                "text_refs": [p["text_b"]],
                "text_both": p["text_b"],
                "gap_words": None,
                "slot1_at_start": p["text_a"].startswith(p["word_a"]),
                "spans": [{"slot": 1, "word_a": p["word_a"], "word_b": p["word_b"],
                           "char_start_a": None, "char_start_b": None}],
            })
        k = 1
    else:
        raise SystemExit(f"data/{pairs_module}.py に build_pairs も build_300_pairs も無い")
    return (items[:n] if n else items), k


# --------------------------------------------------------------------------
# スパン解決
# --------------------------------------------------------------------------
def resolve_item_spans(bundle, ad, item, lang, strict_legacy=True):
    """src 側 k 区間と ref 側 1 区間 x k を、GPT2 の行番号にして返す.

    返り値: (s_rows, r_spans, detail) / 失敗時 (None, None, 理由の文字列)

    * src は k 区間を**まとめて 1 回**で解決する (resolve_spans_with_cursor)。
    * ref は区間ごとに別の文 (text_refs[i]) で 1 区間だけ解決する。
      「両方置換した 1 本の ref」から 2 区間を取らない (干渉の原因が分離できない)。
    * `verify_span_roundtrip` で往復一致を見て、落ちたら文ごと skip。
    * 既存の単一 entity 解決 (`xtts_splice.entity_token_span`) と**同じ区間**に
      なることを全区間で照合する (strict_legacy)。ここが一致していないと
      単一区間の md5 等価性が成立しない。EN cross-culture 600 区間 /
      en_multientity_300 の src 600 区間・ref 600 区間すべてで一致することを
      CPU で確認済み (scripts/xtts/_mecheck/)。
    * 行番号への変換は `gpt_row_offset(cond_len)` を**全区間に一度だけ**足す。
    """
    tok = bundle.tokenizer
    off = X.gpt_row_offset(bundle.cond_len)
    surfaces = [sp["word_a"] for sp in item["spans"]]

    sent = X.normalize_sentence(item["text_a"])
    full_s = X.bpe_input_string(tok, sent, lang)
    surf_s = [X.bpe_word_string(tok, X.normalize_sentence(w), lang) for w in surfaces]

    s_sub, s_info = resolve_spans_with_cursor(ad, full_s, surf_s, add_bos=0,
                                              char_starts=None)
    if s_sub is None:
        return None, None, f"src span resolve failed: {s_info['reason']}"
    ok, bad = verify_span_roundtrip(ad, full_s, s_sub, surf_s, add_bos=0)
    if not ok:
        return None, None, f"src span roundtrip failed: {bad}"

    r_sub, r_full_lens = [], []
    for i, sp in enumerate(item["spans"]):
        rsent = X.normalize_sentence(item["text_refs"][i])
        rfull = X.bpe_input_string(tok, rsent, lang)
        rsurf = X.bpe_word_string(tok, X.normalize_sentence(sp["word_b"]), lang)
        sub, info = resolve_spans_with_cursor(ad, rfull, [rsurf], add_bos=0,
                                              char_starts=None)
        if sub is None:
            return None, None, f"ref[{i}] span resolve failed: {info['reason']}"
        ok, bad = verify_span_roundtrip(ad, rfull, sub, [rsurf], add_bos=0)
        if not ok:
            return None, None, f"ref[{i}] span roundtrip failed: {bad}"
        r_sub.append(sub[0])
        r_full_lens.append(len(ad(rfull)["input_ids"]))

    # --- 既存の単一 entity 解決との照合 ---
    legacy_cmp = []
    for i, sp in enumerate(item["spans"]):
        try:
            la = X.entity_token_span(tok, item["text_a"], sp["word_a"], lang)
            lb = X.entity_token_span(tok, item["text_refs"][i], sp["word_b"], lang)
        except Exception as e:
            return None, None, f"legacy span resolve raised: {type(e).__name__}: {e}"
        same_a = (la["tok_start"], la["tok_end"]) == tuple(s_sub[i])
        same_b = (lb["tok_start"], lb["tok_end"]) == tuple(r_sub[i])
        legacy_cmp.append({"slot": i + 1, "src_same": bool(same_a),
                           "ref_same": bool(same_b),
                           "src_how": la["how"], "ref_how": lb["how"],
                           "src_aligned": la["aligned"], "ref_aligned": lb["aligned"]})
        if strict_legacy and not (same_a and same_b):
            return None, None, (
                f"legacy/cursor span mismatch slot{i + 1}: "
                f"src legacy=({la['tok_start']},{la['tok_end']}) cursor={s_sub[i]} "
                f"ref legacy=({lb['tok_start']},{lb['tok_end']}) cursor={r_sub[i]}")

    # --- 重なりの検査 (splice_multi の関数そのまま) ---
    src_full_len = len(ad(full_s)["input_ids"])
    try:
        check_span_sets(src_full_len, r_full_lens, [tuple(x) for x in s_sub],
                        [tuple(x) for x in r_sub])
    except AssertionError as e:
        return None, None, f"check_span_sets failed: {e}"

    s_rows = [(a + off, b + off) for (a, b) in s_sub]
    d, tot = span_deltas([tuple(x) for x in s_sub], [tuple(x) for x in r_sub])
    detail = {
        "src_tok": [list(x) for x in s_sub],
        "ref_tok": [list(x) for x in r_sub],
        "src_rows": [list(x) for x in s_rows],
        "n": [b - a for a, b in s_sub],
        "m": [b - a for a, b in r_sub],
        "raw_delta_per_span": d,
        "raw_delta_total": tot,
        "gap_tokens": (s_sub[1][0] - s_sub[0][1]) if len(s_sub) > 1 else None,
        "row_offset": off,
        "src_bpe_input": full_s,
        "src_resolve_via": [p["via"] for p in s_info["per_span"]],
        "legacy_agreement": legacy_cmp,
        "n_text_tokens_a": src_full_len,
        "n_text_tokens_refs": r_full_lens,
    }
    return s_rows, r_sub, detail


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
def target_rows_for_refs(bundle, ref, item, r_sub, layer, lang):
    """区間ごとに **別の forward** で目標行を取る (単一 entity 経路と同じ手順).

    xtts_splice.splice_synthesize:1136-1138 と同一:
        prefill_hidden_states(bundle, cond, tgt["text_tokens"]) の taps[layer] から
        [t0:t1] を clone する。
    """
    off = X.gpt_row_offset(bundle.cond_len)
    rows, t_rows, n_text = [], [], []
    for i, txt in enumerate(item["text_refs"]):
        ctx = X.prepare_text(bundle, txt, lang)
        _, taps = X.prefill_hidden_states(bundle, ref["gpt_cond_latent"],
                                          ctx["text_tokens"])
        t0, t1 = int(r_sub[i][0] + off), int(r_sub[i][1] + off)
        rows.append(taps[int(layer)][0, t0:t1].clone())
        t_rows.append([t0, t1])
        n_text.append(ctx["n_text_tokens"])
    return rows, t_rows, n_text


def splice_generate_multi(bundle, ref, src_ctx, s_rows, tgt_rows_list, layer, gen,
                          splice_latent_pass=True, src_prefill=False):
    """k 区間を pass 1 / pass 2 の両方で差し替えて 1 本合成する.

    xtts_splice.splice_synthesize:1144-1155 の流れをそのまま踏み、
    RowSplice の代わりに MultiRowSplice を渡す。呼ぶ関数と引数は同一:
        X._generate_codes(bundle, cond, src["text_tokens"], gen, splice=sp1)
        X._codes_to_wav(bundle, cond, spk, src["text_tokens"], codes, splice=sp2)

    `src_prefill=True` のとき、生成の前に src 側の prefill を 1 回空回しする。
    既存の単一 entity 経路は kappa 用にこれを必ず通る
    (splice_synthesize:1140-1142, run_xtts_eval.py は既定で collect_kappa=True)。
    乱数は `_generate_codes` の先頭で必ず戻るので出力には影響しないが、
    単一区間の md5 照合では**呼び出し列そのもの**を合わせておく。
    """
    cond = ref["gpt_cond_latent"]
    spk = ref["speaker_embedding"]
    block = X.assert_splice_target(bundle, layer)
    p1 = X.prefill_length(bundle.cond_len, src_ctx["n_text_tokens"])
    with torch.inference_mode():
        if src_prefill:
            X.prefill_hidden_states(bundle, cond, src_ctx["text_tokens"])
        sp1 = MultiRowSplice(block, s_rows, tgt_rows_list, expect_len=p1,
                             name="prefill")
        codes = X._generate_codes(bundle, cond, src_ctx["text_tokens"], gen,
                                  splice=sp1)
        n_codes = int(codes.shape[-1])
        sp2 = None
        if splice_latent_pass:
            p2 = X.latent_pass_length(bundle.cond_len, src_ctx["n_text_tokens"],
                                      n_codes)
            sp2 = MultiRowSplice(block, s_rows, tgt_rows_list, expect_len=p2,
                                 name="latent_pass")
        wav, _ = X._codes_to_wav(bundle, cond, spk, src_ctx["text_tokens"], codes,
                                 splice=sp2)
    return {
        "audio": wav, "sampling_rate": bundle.sampling_rate,
        "n_mel_codes": n_codes,
        "dur_sec": float(len(wav) / bundle.sampling_rate),
        "hit_max_mel": bool(n_codes >= bundle.model.gpt.max_gen_mel_tokens),
        "T_out": int(sp1.observed_len), "delta_tok": int(sp1.delta_tok),
        "out_ranges": [list(x) for x in sp1.out_ranges],
        "fired_prefill": sp1.fired,
        "fired_latent_pass": (sp2.fired if sp2 is not None else 0),
        "hook_stats": [sp1.stats()] + ([sp2.stats()] if sp2 is not None else []),
    }


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def save_wav(path, audio, sr):
    """run_xtts_eval.py:80-91 と同じ保存 (PCM_16 / tmp -> rename)."""
    import soundfile as sf
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        raise RuntimeError(f"empty audio for {path}")
    n_clip = int(np.sum(np.abs(a) > 1.0))
    a = np.clip(a, -1.0, 1.0)
    tmp = path + ".tmp"
    sf.write(tmp, a, sr, format="WAV", subtype="PCM_16")
    os.replace(tmp, path)
    return int(a.size), float(a.size / sr), float(n_clip / a.size)


def atomic_json_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs_module", default="en_multientity_300",
                   help="data/<name>.py。build_pairs (2 区間) か build_300_pairs (1 区間)")
    p.add_argument("--lang", default="en", choices=["en"],
                   help="XTTS-v2 は日本語を romaji 化、中国語を拼音化するので英語のみ")
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--pair_ids", default=None,
                   help="pair_idx をカンマ区切りで指定して選ぶ (--n より優先)。"
                        "md5 照合で m==n と m!=n の両方の枝を少数で踏むために使う")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--model_id", default=X.DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--allow_download", action="store_true")
    p.add_argument("--seed", type=int, default=X.SEED)
    p.add_argument("--ref_wav", default=None, help=f"既定 {DEFAULT_REF_WAV}")
    p.add_argument("--gpt_cond_len", type=int, default=X.DEFAULT_GPT_COND_LEN)
    p.add_argument("--gpt_cond_chunk_len", type=int, default=X.DEFAULT_GPT_COND_CHUNK_LEN)
    p.add_argument("--max_ref_len", type=int, default=X.DEFAULT_MAX_REF_LEN)
    p.add_argument("--sound_norm_refs", action="store_true")
    p.add_argument("--greedy", dest="do_sample", action="store_false")
    p.set_defaults(do_sample=True)
    p.add_argument("--temperature", type=float, default=X.DEFAULT_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=X.DEFAULT_TOP_P)
    p.add_argument("--top_k", type=int, default=X.DEFAULT_TOP_K)
    p.add_argument("--repetition_penalty", type=float,
                   default=X.DEFAULT_REPETITION_PENALTY)
    p.add_argument("--length_penalty", type=float, default=X.DEFAULT_LENGTH_PENALTY)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--no_splice_latent_pass", dest="splice_latent_pass",
                   action="store_false",
                   help="pass 2 (HiFiGAN 用 latent) を差し替えない。既定は両 pass")
    p.set_defaults(splice_latent_pass=True)
    p.add_argument("--no_strict_legacy", dest="strict_legacy", action="store_false",
                   help="既存の単一 entity スパン解決との不一致で skip しない")
    p.set_defaults(strict_legacy=True)
    p.add_argument("--src_prefill", default="auto", choices=["auto", "on", "off"],
                   help="生成前に src 側 prefill を空回しする (既存の単一 entity 経路は "
                        "kappa 用に必ず通る)。auto = k==1 のときだけ on")
    p.add_argument("--no_k1_tgt", dest="k1_tgt", action="store_false",
                   help="k=1 のとき TTS(text_b) も出さない (既定は出す。既存の単一 "
                        "entity 本番と生成の順序 bl -> tgt -> inj を合わせるため)")
    p.set_defaults(k1_tgt=True)
    # **後から目標文だけを足すための経路 (2026-09-22)。**既定は off なので
    # 既存の呼び出しはこの 2 つを一切通らない。
    p.add_argument("--tgt_only", action="store_true",
                   help="枠ごとの目標文 tgt1 / tgt2 だけを合成する (注入なし)。"
                        "既存の bl/inj1/inj2/inj12 と results.json は触らない")
    p.add_argument("--dry_run", action="store_true",
                   help="モデルを読まずに、出すつもりの文とファイル名を並べる")
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--md5_check_against", default=None,
                   help="k=1 のときだけ。既存の単一 entity wav ディレクトリ "
                        "(例 eval_results/_cc_wavs_a/xtts_en) と md5 を照合する")
    return p


def main():
    args = build_argparser().parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(
            PROJECT_ROOT, "eval_results",
            f"multientity_xtts_{args.lang}_L{args.layer}")
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    items, k = load_items(args.pairs_module, 0 if args.pair_ids else args.n,
                          args.lang)
    if args.pair_ids:
        want = [int(x) for x in args.pair_ids.replace(" ", "").split(",") if x]
        by_idx = {it["pair_idx"]: it for it in items}
        missing = [i for i in want if i not in by_idx]
        if missing:
            raise SystemExit(f"--pair_ids に無い pair_idx: {missing}")
        items = [by_idx[i] for i in want]
    conds = CONDITIONS_K2 if k == 2 else CONDITIONS_K1
    cond_slots = COND_SLOTS_K2 if k == 2 else COND_SLOTS_K1
    if args.md5_check_against and k != 1:
        raise SystemExit("--md5_check_against は k=1 (cross-culture) 専用")
    # k=1 (等価性の確認) では、既存の単一 entity 本番と**呼び出し列**まで合わせる:
    #   生成の順序 bl -> tgt -> inj、および inj の前に src prefill を 1 回空回し。
    src_prefill = (k == 1) if args.src_prefill == "auto" else \
        (args.src_prefill == "on")
    emit_tgt = bool(k == 1 and args.k1_tgt)
    print(f"[data] {args.pairs_module}: {len(items)} items, k={k} spans/item, "
          f"conds={conds} src_prefill={src_prefill} emit_tgt={emit_tgt}", flush=True)
    print(f"[out ] {args.output_dir}", flush=True)

    if args.tgt_only and k != 2:
        raise SystemExit("--tgt_only は 2 枠のデータ専用 (今のデータは k=%d)" % k)
    if args.dry_run:
        # モデルを読む前に返す (CPU で名前と文を確かめるための経路)。
        dry_run_report(items, args.output_dir, tag="xtts")
        return

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[load] {args.model_id} dtype={args.dtype} device={args.device} "
          f"HF_HOME={os.environ['HF_HOME']}", flush=True)
    t_load = time.time()
    bundle = X.load_xtts(model_id=args.model_id, device=args.device, dtype=dtype,
                         allow_download=args.allow_download)
    print(f"[load] done in {time.time() - t_load:.1f}s dir={bundle.model_dir} "
          f"layers={bundle.num_layers} hidden={bundle.hidden_size} "
          f"cond_len={bundle.cond_len}", flush=True)
    if not (0 <= args.layer < bundle.num_layers):
        raise SystemExit(f"--layer {args.layer} が範囲外 (0..{bundle.num_layers - 1})")
    X.assert_splice_target(bundle, args.layer)

    ref_wav = args.ref_wav or DEFAULT_REF_WAV
    ref = X.load_reference(bundle, ref_wav, gpt_cond_len=args.gpt_cond_len,
                           gpt_cond_chunk_len=args.gpt_cond_chunk_len,
                           max_ref_length=args.max_ref_len,
                           sound_norm_refs=args.sound_norm_refs)
    print(f"[ref ] {ref_wav} cond={list(ref['gpt_cond_latent'].shape)} "
          f"spk={list(ref['speaker_embedding'].shape)} "
          f"checksum={ref['cond_checksum']}", flush=True)

    gen = X.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      temperature=args.temperature, top_p=args.top_p,
                      top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                      length_penalty=args.length_penalty, num_beams=args.num_beams)
    print(f"[gen ] layer={args.layer} splice_latent_pass={args.splice_latent_pass} "
          f"{gen.as_dict()}", flush=True)

    ad = BpeOffsetAdapter(bundle.tokenizer)

    # 行番号のずれを 1 回だけ機械的に潰す (run_xtts_eval.py:308-316 と同じ)
    probe = X.prepare_text(bundle, items[0]["text_a"], args.lang)
    layout = X.verify_prefill_layout(bundle, ref["gpt_cond_latent"],
                                     probe["text_tokens"])
    print(f"[layout] {layout}", flush=True)

    if args.tgt_only:
        # 注入しないので区間の解決も往復確認もしない。合成は bl 条件と同じ
        # X.synthesize で、渡す文だけが text_refs[slot] に変わる。seed も
        # 本体と同じく条件の入口で戻す (X._generate_codes と同じ位置)。
        def _synth_tgt(text, path):
            X.seed_all(gen.seed)
            X.assert_reference_unchanged(ref)
            r = X.synthesize(bundle, text, ref, args.lang, gen=gen)
            X.assert_reference_unchanged(ref)
            _, dur, clip = save_wav(path, r["audio"], bundle.sampling_rate)
            return {"dur_sec": round(float(dur), 4),
                    "n_mel_codes": int(r["n_mel_codes"]),
                    "clip_rate": round(float(clip), 6),
                    "md5": md5_file(path)}

        run_targets(items, args.output_dir, _synth_tgt, tag="xtts",
                    ckpt_every=args.ckpt_every, log_every=args.log_every,
                    stop_on_error=args.stop_on_error,
                    extra_summary={"method": "xtts-v2", "lang": args.lang,
                                   "pairs_module": args.pairs_module,
                                   "model_id": args.model_id,
                                   "dtype": args.dtype,
                                   "ref_wav": ref_wav,
                                   "cond_checksum": ref["cond_checksum"],
                                   "gen": gen.as_dict()})
        return

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    rows, skipped = [], []
    if os.path.exists(ckpt_path) and not args.overwrite:
        st = json.load(open(ckpt_path))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows
               if r.get("mode") != "splice_multi" or r.get("method") != "xtts-v2"]
        assert not bad, (f"checkpoint mode mismatch in {ckpt_path}: pairs {bad[:5]} "
                         f"は別のスクリプト (別手法) の出力。--output_dir を分けること")
        print(f"[resume] {len(rows)} rows, {len(skipped)} skipped", flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}

    md5_report = []
    t0 = time.time()
    for it in items:
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        try:
            src_ctx = X.prepare_text(bundle, it["text_a"], args.lang)
        except Exception as e:
            print(f"  [{pidx}] prepare_text fail: {str(e)[:160]}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"prepare_text: {e}"})
            done.add(pidx)
            continue

        s_rows, r_sub, detail = resolve_item_spans(bundle, ad, it, args.lang,
                                                   strict_legacy=args.strict_legacy)
        if s_rows is None:
            print(f"  [{pidx}] SKIP {detail}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": detail})
            done.add(pidx)
            continue

        T_src = X.prefill_length(bundle.cond_len, src_ctx["n_text_tokens"])
        row = {
            "pair_idx": pidx, "mode": "splice_multi", "method": "xtts-v2",
            "splice_style": "row_wise_length_preserving",
            "template_id": it.get("template_id"), "lang": args.lang,
            "text_a": it["text_a"], "text_refs": it["text_refs"],
            "text_both": it["text_both"],
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "repeat_surface": it.get("repeat_surface", False),
            "words_a": [sp["word_a"] for sp in it["spans"]],
            "words_b": [sp["word_b"] for sp in it["spans"]],
            "span_tok": detail,
            "layer": int(args.layer),
            "hidden_index": X.hidden_index_for_layer(args.layer),
            "T_src": int(T_src),
            "n_text_tokens_a": int(src_ctx["n_text_tokens"]),
            "conds": {},
        }

        # 目標行の forward は **最初の inj 条件の直前** まで遅らせる。
        # 既存の単一 entity 本番の順序 (bl -> tgt -> splice_synthesize の中で
        # ref prefill -> src prefill -> generate) と GPU 呼び出し列をそろえるため。
        cache = {}

        def tgt_rows_all():
            if "rows" not in cache:
                r, t_rows, ref_n_text = target_rows_for_refs(
                    bundle, ref, it, r_sub, args.layer, args.lang)
                cache["rows"] = r
                row["span_tok"]["ref_rows"] = t_rows
                row["span_tok"]["n_text_tokens_refs_prepared"] = ref_n_text
            return cache["rows"]

        # 生成の順序。k=1 では既存の単一 entity 本番 (run_xtts_eval.py:365-381)
        # と同じ bl -> tgt -> inj にそろえる。
        steps = list(conds)
        if emit_tgt:
            steps.insert(1, "__tgt__")

        ok = True
        for cond in steps:
            if cond == "__tgt__":
                tgt_path = os.path.join(args.output_dir, f"s{pidx:04d}_tgt.wav")
                try:
                    X.seed_all(gen.seed)
                    rt = X.synthesize(bundle, it["text_refs"][0], ref, args.lang,
                                      gen=gen)
                    _, dt, ct = save_wav(tgt_path, rt["audio"], bundle.sampling_rate)
                    row["tgt"] = {"wav": os.path.basename(tgt_path),
                                  "text": it["text_refs"][0],
                                  "dur_sec": round(float(dt), 4),
                                  "n_mel_codes": int(rt["n_mel_codes"]),
                                  "clip_rate": round(float(ct), 6)}
                except Exception as e:
                    print(f"  [{pidx}] tgt fail: {type(e).__name__}: {str(e)[:200]}",
                          flush=True)
                    skipped.append({"pair_idx": pidx, "reason": f"tgt: {e}"[:400]})
                    ok = False
                    if args.stop_on_error:
                        raise
                    break
                continue
            slots = cond_slots[cond]
            wav_path = os.path.join(args.output_dir, f"s{pidx:04d}_{cond}.wav")
            try:
                # (b) 乱数の消費順序をそろえる: 条件の入口で必ず seed を戻す。
                #     正本は X._generate_codes:1048 の seed_all(gen.seed) で、
                #     generate の直前に必ずもう一度戻る。
                X.seed_all(gen.seed)
                X.assert_reference_unchanged(ref)
                if not slots:
                    r = X.synthesize(bundle, it["text_a"], ref, args.lang, gen=gen,
                                     ctx=src_ctx)
                    r.update(T_out=int(T_src), delta_tok=0, out_ranges=[],
                             fired_prefill=0, fired_latent_pass=0, hook_stats=[])
                else:
                    tg = tgt_rows_all()
                    r = splice_generate_multi(
                        bundle, ref, src_ctx,
                        [s_rows[i] for i in slots],
                        [tg[i] for i in slots],
                        args.layer, gen,
                        splice_latent_pass=args.splice_latent_pass,
                        src_prefill=src_prefill)
                    want2 = 1 if args.splice_latent_pass else 0
                    if r["fired_prefill"] != 1 or r["fired_latent_pass"] != want2:
                        raise RuntimeError(
                            f"hook の発火回数が想定外 (prefill={r['fired_prefill']} "
                            f"latent={r['fired_latent_pass']})")
                X.assert_reference_unchanged(ref)
                _, dur, clip = save_wav(wav_path, r["audio"], bundle.sampling_rate)
            except Exception as e:
                import traceback
                print(f"  [{pidx}] {cond} fail: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                traceback.print_exc()
                skipped.append({"pair_idx": pidx, "reason": f"{cond}: {e}"[:400]})
                ok = False
                if args.stop_on_error:
                    raise
                break

            row["conds"][cond] = {
                "wav": os.path.basename(wav_path),
                "delta_tok": int(r["delta_tok"]),
                "T_out": int(r["T_out"]),
                "out_ranges": [list(x) for x in r["out_ranges"]],
                "target_text": it["text_a"],   # XTTS は入力文を一切変えない
                "dur_sec": round(float(dur), 4),
                "slots": slots,
                "n_mel_codes": int(r["n_mel_codes"]),
                "hit_max_mel": bool(r["hit_max_mel"]),
                "clip_rate": round(float(clip), 6),
                "splice_latent_pass": bool(args.splice_latent_pass) if slots else False,
                "fired_prefill": int(r["fired_prefill"]),
                "fired_latent_pass": int(r["fired_latent_pass"]),
                "hook_stats": r["hook_stats"],
            }

        done.add(pidx)
        if not ok:
            continue
        # XTTS は系列長を変えないので総 position ずれは定義上 0。
        # (T5Gemma は長さ正規化ランプなので 0 でない値が出る)
        row["total_position_drift"] = row["conds"][conds[-1]]["delta_tok"]
        rows.append(row)

        if args.md5_check_against:
            base = args.md5_check_against
            if not os.path.isabs(base):
                base = os.path.join(PROJECT_ROOT, base)
            rec = {"pair_idx": pidx}
            pl = [("bl", "bl"), ("inj1", "inj")]
            if emit_tgt:
                pl.insert(1, ("tgt", "tgt"))
            for mine, theirs in pl:
                a = os.path.join(args.output_dir, f"s{pidx:04d}_{mine}.wav")
                b = os.path.join(base, f"s{pidx:04d}_{theirs}.wav")
                if not os.path.exists(b):
                    rec[mine] = "reference_missing"
                    continue
                ha, hb = md5_file(a), md5_file(b)
                rec[mine] = "PASS" if ha == hb else f"FAIL {ha} != {hb}"
            md5_report.append(rec)
            print(f"  [md5 {pidx}] bl={rec.get('bl')} "
                  f"tgt={rec.get('tgt', '-')} inj1={rec.get('inj1')}", flush=True)

        if len(rows) % args.log_every == 0:
            c = row["conds"]
            el = (time.time() - t0) / 60
            print(f"  [{len(rows)}/{len(items)}] idx={pidx} "
                  f"gap_tok={detail['gap_tokens']} "
                  f"n={detail['n']} m={detail['m']} "
                  f"dur=" + "/".join(f"{c[x]['dur_sec']:.2f}" for x in conds)
                  + f" {el:.1f}min", flush=True)
        if len(rows) % args.ckpt_every == 0:
            atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt_path)

    atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt_path)

    def mean_of(key, cond):
        v = [r["conds"][cond][key] for r in rows if cond in r["conds"]]
        return round(float(np.mean(v)), 4) if v else None

    n_mean_pool = {}
    for cond in conds:
        cnt = 0
        for r in rows:
            for hs in r["conds"].get(cond, {}).get("hook_stats", [])[:1]:
                cnt += sum(1 for e in hs["equal_len"] if not e)
        n_mean_pool[cond] = cnt

    summary = {
        "pairs_module": args.pairs_module, "lang": args.lang, "k_spans": k,
        "conds": conds,
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "model_id": args.model_id, "model_dir": bundle.model_dir,
        "license": "Coqui Public Model License (non-commercial)",
        "layer": int(args.layer),
        "hidden_index": X.hidden_index_for_layer(args.layer),
        "num_layers": bundle.num_layers, "hidden_size": bundle.hidden_size,
        "cond_len": bundle.cond_len,
        "row_offset": X.gpt_row_offset(bundle.cond_len),
        "seed": args.seed, "decode": gen.as_dict(),
        "dtype": args.dtype, "device": args.device,
        "splice_latent_pass": bool(args.splice_latent_pass),
        "strict_legacy_span_agreement": bool(args.strict_legacy),
        "src_prefill_parity": bool(src_prefill),
        "emit_tgt": bool(emit_tgt),
        "ref": {"wav": ref["wav_path"], "cond_len": ref["cond_len"],
                "gpt_cond_len": ref["gpt_cond_len"],
                "gpt_cond_chunk_len": ref["gpt_cond_chunk_len"],
                "max_ref_length": ref["max_ref_length"],
                "sound_norm_refs": ref["sound_norm_refs"],
                "checksum": ref["cond_checksum"]},
        "prefill_layout_check": layout,
        "sampling_rate": bundle.sampling_rate,
        "mean_dur_sec": {c: mean_of("dur_sec", c) for c in conds},
        "mean_n_mel_codes": {c: mean_of("n_mel_codes", c) for c in conds},
        "mean_abs_position_drift": round(float(np.mean(
            [abs(r["total_position_drift"]) for r in rows])), 3) if rows else None,
        "n_mean_pool_broadcast_spans": n_mean_pool,
        "splice": (
            "Replace, in the output of gpt.gpt.h[layer] (a GPT-2 block of the "
            "XTTS-v2 trunk), the rows of k entity spans at once with the same "
            "layer's hidden states taken from k separate target-context prompts "
            "(one forward per span). The assembly is done by "
            "scripts/projection/splice_multi.py:build_spliced_hidden_multi in a "
            "single concat, with the replacement rows length-matched to the source "
            "spans beforehand (token-wise when m==n, mean-pooled and broadcast "
            "otherwise, exactly as the single-span xtts_splice.RowSplice does), so "
            "the sequence length never changes and the index of the second span is "
            "never shifted by the first replacement. XTTS-v2 runs the trunk twice "
            "per utterance (AR prefill inside gpt.generate, and the teacher-forced "
            "pass gpt(..., return_latent=True) that feeds HiFiGAN); the splice is "
            "applied to both, named by sequence length. No weight update."),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "coqui_tts": _pkg_version("coqui-tts"),
        "transformers": _pkg_version("transformers"),
        "torch": torch.__version__,
    }
    if args.md5_check_against:
        keys = ["bl", "inj1"] + (["tgt"] if emit_tgt else [])
        n_pass = sum(1 for r in md5_report
                     if all(r.get(x) == "PASS" for x in keys))
        summary["md5_check"] = {"against": args.md5_check_against, "keys": keys,
                                "n_checked": len(md5_report), "n_pass": n_pass,
                                "rows": md5_report}

    atomic_json_dump({"summary": summary, "rows": rows, "skipped": skipped},
                     os.path.join(args.output_dir, "results.json"))
    print(f"\n=== multi-entity XTTS-v2 {args.pairs_module} (k={k}) ===")
    print(json.dumps({x: summary[x] for x in (
        "n_items", "n_generated", "n_skipped", "layer", "mean_dur_sec",
        "mean_n_mel_codes", "mean_abs_position_drift",
        "n_mean_pool_broadcast_spans")}, ensure_ascii=False, indent=1))
    if args.md5_check_against:
        print("\n=== 単一区間の md5 等価性 ===")
        for r in md5_report:
            print(f"  s{r['pair_idx']:04d}  bl={r.get('bl')}  "
                  f"tgt={r.get('tgt', '-')}  inj1={r.get('inj1')}")
        print(f"  PASS {summary['md5_check']['n_pass']}/"
              f"{summary['md5_check']['n_checked']}  "
              f"(照合した条件: {summary['md5_check']['keys']})")
    if skipped:
        print(f"[Warn] {len(skipped)} items skipped; results.json['skipped'] を見ること")


def _pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


if __name__ == "__main__":
    main()
