#!/usr/bin/env python3
"""CSM-1B で 1 文の中の 2 箇所を同時に差し替える (BL / INJ_1 / INJ_2 / INJ_12)。

T5Gemma 版 `scripts/analysis/generate_multientity.py` の構成をそのまま写した
**新規**スクリプト。既存ファイルは 1 行も書き換えていない。音を作る部品は
`scripts/csm/csm_splice.py` から、区間の一括差し替えは
`scripts/projection/splice_multi.py` から import して使う (どちらも再実装しない)。
出所の台帳 (設定 / 実装 / 入力の 3 層照合) は `scripts/csm/run_csm_eval.py` の
実装をそのまま import して使う。

--------------------------------------------------------------------------
CSM で 2 箇所同時をやるときの一番大事な違い: **系列長が変わらない**
--------------------------------------------------------------------------
T5Gemma は encoder-decoder なので、encoder 出力の区間を目標区間 (長さ m) で
置き換えると系列長が T -> T + Σ(m_i - n_i) に変わる。splice_multi の
`build_spliced_hidden_multi` が一括 concat でこれを扱う。

CSM は decoder-only で、差し替え地点は `backbone_model.layers[14]` の**出力**
(prefill の実体語トークン行) である。ここで行数を変えると
  * RoPE の位置 (cache_position / position_ids) が後続とずれる
  * layers[0..14] が既に書いた KV cache の長さと合わなくなる
ので、既存の単一 entity 実装 `csm_splice.PrefillSplice` は
**行数を変えない**方針を採っている (csm_splice.py:710-716):

    m = 元テキストの実体語トークン数、n = 目標テキストのそれ
    m == n : 行対行でそのまま置換            (splice_mode = token_wise)
    m != n : 目標区間をトークン方向に平均した 1 本を m 行に broadcast
                                              (splice_mode = mean_pool_broadcast)

複数区間版でもこの方針を 1 ビットも変えない。したがって

    * 各区間の置換行は必ず m_i 行 = 元の区間長  ->  Δtok_i = 0
    * 総ずれ delta_tok = 0、T_out == T_src、out_ranges == s_ranges
    * **2 番目の区間の index は 1 番目の差し替えでずれない**

置換行の作り方 (token_wise / mean_pool_broadcast の選択) は自分では書かず、
区間ごとに `csm_splice.PrefillSplice` を 1 個作って
`PrefillSplice._replacement(dtype)` に作らせる。そのうえで
`splice_multi.build_spliced_hidden_multi` に
「長さを合わせ済みの置換行」を ref として渡し、r_range = (0, m_i) で一括 concat
させる (下の MultiPrefillSplice)。こうすると

  * 長さ不一致の扱いは単一 entity 版と同じ関数で決まる
  * 区間の重なり / 範囲外 / 個数不一致の検査は splice_multi.check_span_sets
    (build_spliced_hidden_multi の中で必ず呼ばれる) が行う
  * k = 1 に縮退させたときの出力は
        h.clone(); h[0, s0:s1] = repl        (PrefillSplice.__call__)
    と
        cat([h[0,:s0], repl, h[0,s1:]])      (build_spliced_hidden_multi)
    の比較になり、どちらも値の移動しかしていないので **ビット一致**する
    (CPU 検証: scripts/csm/verify_multientity_csm_cpu.py の test_k1_bitwise)

--------------------------------------------------------------------------
区間の解決 (BOS / 話者タグのオフセット)
--------------------------------------------------------------------------
CSM は `<|begin_of_text|>[0]本文<|end_of_text|>` を 1 本の prefix として
prefill するので、トークン index = hidden state の行番号になる。
参照音声ターンを前置すると本文の開始行が動くため、**最後の BOS を起点**に
最終メッセージだけを tokenize して局所 index を取り、`bos_pos` を
**全区間に一度だけ**足す (csm_splice.entity_token_span:518-528 と同じ流儀)。

複数区間は `splice_multi.resolve_spans_with_cursor` で**まとめて**解決する
(同じ語が 2 回出る場合に左から順に解決できるのはこちら)。そのうえで
1 区間ずつ `csm_splice.entity_token_span` でも解決して**必ず突き合わせる**。
食い違ったらその文は丸ごと捨てる。「末尾 N トークンを実体語とみなす」種類の
fallback は一切作らない。

--------------------------------------------------------------------------
4 条件と乱数
--------------------------------------------------------------------------
    bl     差し替えなし。prompt = text_a
    inj1   枠 1 だけ差し替え。ref = text_refs[0] の prefill
    inj2   枠 2 だけ差し替え。ref = text_refs[1] の prefill
    inj12  両方差し替え。枠 i の ref は ref_i (**区間ごとに別の prefill**)

`text_both` (両方置換した 1 本) から 2 区間を取らない。取ると干渉の原因
(splice の相互作用なのか ref 側の文脈が既に汚れているのか) が分離できない。

4 条件は prompt が 1 トークンも違わない (どれも text_a)。差は prefill 中の
差し替えだけ。seed は `csm_splice._generate` が generate の直前に
`seed_all(seed)` で毎回戻すので、4 条件の乱数の消費順序は揃う。
prefill (`prefill_forward`) は eval + no_grad + サンプリング無しなので
乱数を 1 つも消費しない。

--------------------------------------------------------------------------
出所の台帳 (run_csm_eval.py の 3 層照合をそのまま引き継ぐ)
--------------------------------------------------------------------------
  a. 設定 (audio_fp_static_hash) 食い違えば **停止**
  b. 実装 (code_fp_hash)         停止しない。その指紋で書かれた wav を
                                 信用せず退避して作り直す
  c. 入力 (input_fp)             wav 1 本ごと。文・語・層・話者・参照音声

run_csm_eval.py は条件名が ("bl", "inj", "tgt") 固定なので、その台帳の実装を
使うために **import してモジュール属性を差し替える** (ファイルは触らない)。
差し替えるのは条件集合とファイル名の正規表現と台帳の schema だけで、
照合・退避・突き合わせの中身は 1 行も書き写していない。

出力 (採点器 scripts/analysis/score_multientity.py がそのまま読む形):
    <out_dir>/results.json       summary + rows。conds に bl/inj1/inj2/inj12
    <out_dir>/checkpoint.json    途中再開用
    <out_dir>/s{pair_idx:04d}_{bl,inj1,inj2,inj12}.wav
    <out_dir>/_wav_manifest.json 音声の出所の台帳

実行例 (GPU):
  SLURM_TIME=10:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/csm/run_multientity_csm.py \\
      --pairs_module en_multientity_300 --lang en --n 300 --layer 14 \\
      --out_dir /mnt/eval_results/csm_multientity_en_300

単一区間の等価性 (合格条件):
  SLURM_TIME=01:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/csm/run_multientity_csm.py \\
      --pairs_module en_cross_culture_names --lang en --n 5 --layer 14 \\
      --out_dir /mnt/eval_results/csm_multientity_equiv_k1_n5 \\
      --equiv_check_dir /mnt/eval_results/_cc_wavs_a/csm_en

CPU 検証:
  taskset -c 0-63 python3 scripts/csm/verify_multientity_csm_cpu.py
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid

# --------------------------------------------------------------------------
# HF cache。slurm_run.sh が渡す HF_HOME=/tmp/.cache/huggingface は空で書けない。
# **setdefault では負けるので代入で上書きする。** transformers より先に済ませる。
# --------------------------------------------------------------------------
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "modules")

PROJECT_ROOT = ROOT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "csm"))

import csm_splice as C  # noqa: E402  (HF_HOME をもう一度代入する。同じ値)
import run_csm_eval as R  # noqa: E402  (台帳・指紋の実装をそのまま使う)
from splice_multi import (  # noqa: E402  **再実装しない**
    build_spliced_hidden_multi, check_span_sets, resolve_spans_with_cursor,
    span_deltas, verify_span_roundtrip,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_REFS = R.DEFAULT_REFS


# --------------------------------------------------------------------------
# 条件集合
# --------------------------------------------------------------------------
def conditions_for(k):
    """区間数 k に対する条件名と、条件 -> 使う枠 index の対応。

    k = 2 : bl / inj1 / inj2 / inj12   (本番)
    k = 1 : bl / inj1                  (単一区間の等価性試験)
    """
    if k == 1:
        conds = ["bl", "inj1"]
        slots = {"bl": [], "inj1": [0]}
        return conds, slots
    if k == 2:
        conds = ["bl", "inj1", "inj2", "inj12"]
        slots = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}
        return conds, slots
    raise SystemExit(f"区間数 k={k} は未対応 (1 か 2)。データ側を確かめること。")


def patch_run_csm_eval(conds):
    """run_csm_eval の台帳実装を **条件集合だけ差し替えて**使う。

    run_csm_eval.py は条件名が ("bl", "inj", "tgt") 固定で、wav 名の正規表現と
    台帳の schema もそれ前提。照合・退避・突き合わせのコードは 300 行あり、
    書き写すと本物との乖離が起きるので、モジュール属性を差し替えて使う。
    **ファイルは書き換えていない。** 差し替えたのは以下だけ。
    """
    R.COND = tuple(conds)
    R._WAV_RE = re.compile(r"^s(\d{4})_("
                           + "|".join(sorted(conds, key=len, reverse=True))
                           + r")\.wav$")
    R.MANIFEST_SCHEMA = "csm_multientity_wav_manifest_v1"
    # 条件ごとの「wav が無ければ意味を失う判定」。本スクリプトは ASR 判定を
    # 持たない (採点は score_multientity.py) ので空。KeyError を出さないために要る。
    R.COND_JUDGEMENTS = {c: () for c in conds}
    # 音のビットを決める鍵に、複数 entity 固有のものを足す。
    extra = ("pairs_module", "n_spans", "conds", "splice_multi_policy")
    R.AUDIO_FP_STATIC_KEYS = tuple(
        list(R.AUDIO_FP_STATIC_KEYS) + [k for k in extra
                                        if k not in R.AUDIO_FP_STATIC_KEYS])
    # 差し替えが効いていることをその場で確かめる (効いていないと単一 entity の
    # wav を掴んだまま走る)。
    assert sorted(R.wav_paths(".", 0)) == sorted(conds), R.wav_paths(".", 0)
    assert R._WAV_RE.match(f"s0000_{conds[-1]}.wav"), conds
    assert not R._WAV_RE.match("s0000_tgt.wav"), "単一 entity の wav 名に当たっている"


# --------------------------------------------------------------------------
# 複数区間の prefill splice hook
# --------------------------------------------------------------------------
class MultiPrefillSplice:
    """`backbone_model.layers[l]` の出力の k 区間を、prefill のときだけ一括で置換する。

    単一区間版 `csm_splice.PrefillSplice` との関係:

      * prefill 判定 (T <= 1 は decode / prompt_len 不一致は見送り / 1 回だけ発火 /
        3 次元 Tensor でなければ raise / 別モジュールには触らない) は
        PrefillSplice と同じ条件をそのまま並べてある。
      * **置換行の作り方は自分で書かない。** 区間ごとに PrefillSplice を 1 個
        持ち、`PrefillSplice._replacement(dtype)` に作らせる
        (m == n なら行対行、m != n なら平均を m 行に broadcast)。
      * 置換そのものは `splice_multi.build_spliced_hidden_multi` に任せる。
        区間の重なり・範囲外・個数不一致は同関数が呼ぶ
        `splice_multi.check_span_sets` が AssertionError で落とす。
      * 置換行は必ず m_i 行 (= 元の区間長) なので delta_tok は必ず 0。
        したがって out_ranges == s_ranges で、**2 番目の区間の index を
        自分でずらす処理は 1 行も無い**。0 でなければ raise する。
    """

    def __init__(self, layer_module, s_ranges, tgt_hiddens, prompt_len=None,
                 expect_module=None):
        if len(s_ranges) != len(tgt_hiddens):
            raise ValueError(f"区間数 {len(s_ranges)} と目標 hidden の数 "
                             f"{len(tgt_hiddens)} が違う")
        self.layer_module = layer_module
        self.expect_module = expect_module if expect_module is not None else layer_module
        self.prompt_len = int(prompt_len) if prompt_len is not None else None
        self.s_ranges = [(int(a), int(b)) for a, b in s_ranges]
        # 長さ不一致の扱いを単一 entity 版と共有するための委譲先。
        # hook として登録はしない (register_forward_hook は呼ばない)。
        self.subs = [
            C.PrefillSplice(layer_module, {"tok_start": int(a), "tok_end": int(b)},
                            th, prompt_len=prompt_len, expect_module=self.expect_module)
            for (a, b), th in zip(self.s_ranges, tgt_hiddens)]
        # ref 側は「長さを合わせ済みの置換行」なので必ず (0, m_i)。
        self.r_ranges = [(0, s.m) for s in self.subs]
        self.handle = None
        self.fired = 0
        self.decode_calls = 0
        self.foreign_calls = 0
        self.skipped_shapes = []
        self.delta_tok = None
        self.out_ranges = None
        self.T_in = None
        self.T_out = None
        # 生成前に区間集合を検査しておく (発火してから落ちると原因が読みにくい)。
        # ref 側の長さは「置換行の数」= m_i。__call__ が渡すものと同一にする。
        check_span_sets(self.prompt_len if self.prompt_len is not None
                        else max(b for _a, b in self.s_ranges),
                        [int(s.m) for s in self.subs], self.s_ranges, self.r_ranges)
        _, d = span_deltas(self.s_ranges, self.r_ranges)
        if d != 0:
            raise RuntimeError(f"置換行の総ずれが {d} (0 でなければ decoder-only では"
                               f"位置と KV cache がずれる)")

    # 置換行。区間ごとに単一 entity 版の実装をそのまま呼ぶ。
    def _replacement_rows(self, dtype):
        return [s._replacement(dtype) for s in self.subs]

    def __call__(self, module, args, output):
        if module is not self.expect_module:
            self.foreign_calls += 1
            return output
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            raise RuntimeError(
                f"CsmDecoderLayer の出力が 3 次元の Tensor でない: "
                f"type={type(output)} shape={getattr(h, 'shape', None)}。"
                f"素通しすると INJ が BL と同一になるので中断する。")
        T = h.shape[1]
        if T <= 1:
            self.decode_calls += 1
            return output
        if self.prompt_len is not None and T != self.prompt_len:
            self.skipped_shapes.append(int(T))
            return output
        if self.fired:
            self.skipped_shapes.append(int(T))
            return output
        if h.shape[0] != 1:
            raise RuntimeError(f"batch size {h.shape[0]} は未対応 (1 のみ)")
        for s0, s1 in self.s_ranges:
            if s1 > T:
                raise RuntimeError(f"区間 [{s0},{s1}) が prefill 長 {T} を超えている")

        reps = self._replacement_rows(h.dtype)
        # PrefillSplice.__call__ と同じ順で dtype -> device を合わせる。
        ref_encs = [{"last_hidden_state": r.to(h.device).unsqueeze(0)} for r in reps]
        modified, delta_tok, out_ranges = build_spliced_hidden_multi(
            {"last_hidden_state": h}, ref_encs, self.s_ranges, self.r_ranges)
        if int(delta_tok) != 0:
            raise RuntimeError(f"delta_tok={delta_tok}。CSM は行数を変えられない。")
        if modified.shape[1] != T:
            raise RuntimeError(f"T_out={modified.shape[1]} != T_in={T}")
        got = [(int(a), int(b)) for a, b in out_ranges]
        if got != self.s_ranges:
            # 一括 concat が index を動かしていない = 自分でずらす必要が無いことの
            # 実行時検査。ここが落ちるのは長さが変わったときだけ。
            raise RuntimeError(f"out_ranges {got} != s_ranges {self.s_ranges}")
        self.fired += 1
        self.delta_tok = int(delta_tok)
        self.out_ranges = got
        self.T_in = int(T)
        self.T_out = int(modified.shape[1])
        if isinstance(output, tuple):
            return (modified,) + tuple(output[1:])
        return modified

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def assert_fired(self):
        if self.fired != 1:
            raise RuntimeError(
                f"splice hook の発火回数が {self.fired} (1 であるべき)。"
                f"prompt_len={self.prompt_len} skipped_shapes={self.skipped_shapes[:8]} "
                f"decode_calls={self.decode_calls}")

    def stats(self):
        return {
            "fired": self.fired, "decode_calls": self.decode_calls,
            "foreign_calls": self.foreign_calls,
            "skipped_shapes": self.skipped_shapes[:16],
            "delta_tok": self.delta_tok, "T_in": self.T_in, "T_out": self.T_out,
            "out_ranges": [list(x) for x in (self.out_ranges or [])],
            "m_src_entity_tokens": [int(s.m) for s in self.subs],
            "n_tgt_entity_tokens": [int(s.n) for s in self.subs],
            "equal_len": [bool(s.equal_len) for s in self.subs],
            "splice_mode": ["token_wise" if s.equal_len else "mean_pool_broadcast"
                            for s in self.subs],
        }


# --------------------------------------------------------------------------
# 区間の解決
# --------------------------------------------------------------------------
def _body_offset(bundle, text, speaker):
    """最終メッセージを rendered した文字列と、その中の本文の開始文字位置。

    csm_splice.entity_token_span:549-553 と同じ検査 (本文がちょうど 1 回出る)。
    """
    rendered = C.render_message(bundle.tokenizer, text, speaker=speaker)
    n_body = rendered.count(text)
    if n_body != 1:
        raise ValueError(f"本文が rendered に {n_body} 回出現する (1 回であるべき): {text!r}")
    return rendered, rendered.index(text)


def _bos_pos_from_ids(bundle, text, ids, speaker):
    """最後の BOS の位置。entity_token_span が使えないときの退路。

    entity_token_span:536-547 と同じ検査をする (最終メッセージの tokenize が
    ids[bos_pos:] と一致すること)。ここが合わないと行番号が信用できない。
    """
    bos = [i for i, t in enumerate(ids) if t == C.BOS_TOKEN_ID]
    if not bos:
        raise RuntimeError("input_ids に BOS (128000) が無い")
    bos_pos = bos[-1]
    rendered = C.render_message(bundle.tokenizer, text, speaker=speaker)
    local = list(bundle.tokenizer(rendered, add_special_tokens=False)["input_ids"])
    if local != ids[bos_pos:]:
        raise RuntimeError(
            f"最終メッセージの tokenize が input_ids[last_bos:] と一致しない "
            f"(len {len(local)} vs {len(ids) - bos_pos})")
    return int(bos_pos)


def resolve_spans_in_text(bundle, text, input_ids, surfaces, char_starts, speaker):
    """1 本の文の k 区間を **まとめて** 絶対トークン index に解決する。

    手順:
      1. `splice_multi.resolve_spans_with_cursor` で rendered 文字列上の
         局所 index を k 個まとめて取る (同じ語が 2 回出ても左から順に解ける)。
      2. `splice_multi.verify_span_roundtrip` で往復一致を見る。
      3. 局所 index に `bos_pos` を **全区間に一度だけ**足す。
      4. 1 区間ずつ `csm_splice.entity_token_span` でも解決して突き合わせる。
         これが単一 entity 経路との同一性の担保になる。表層形が本文に 2 回以上
         出る場合だけ entity_token_span が使えないので、その区間は
         xcheck="skipped" と記録する (黙って通さない)。

    返り値: (spans, detail)。解決できなければ (None, 理由の文字列)
    """
    tok = bundle.tokenizer
    ids = input_ids[0].tolist() if torch.is_tensor(input_ids) else list(input_ids)
    try:
        rendered, b0 = _body_offset(bundle, text, speaker)
    except Exception as e:
        return None, f"rendered 失敗: {type(e).__name__}: {e}"

    starts = None
    if char_starts is not None:
        starts = [None if cs is None else int(b0) + int(cs) for cs in char_starts]
    local, info = resolve_spans_with_cursor(tok, rendered, surfaces, add_bos=0,
                                            char_starts=starts)
    if local is None:
        return None, f"span resolve 失敗: {info['reason']}"
    ok, bad = verify_span_roundtrip(tok, rendered, local, surfaces, add_bos=0)
    if not ok:
        return None, f"span roundtrip 失敗: {bad}"

    # --- 単一 entity 経路との突き合わせ ---
    xspans, xstat, bos_pos = [], [], None
    for surf in surfaces:
        try:
            sp = C.entity_token_span(bundle, text, surf, input_ids=input_ids,
                                     speaker=speaker)
        except Exception as e:
            xspans.append(None)
            xstat.append(f"skipped:{type(e).__name__}:{str(e)[:80]}")
            continue
        okx, msg = C.verify_entity_span(sp, strict=False)
        if not okx:
            return None, f"entity_token_span の往復に失敗: {msg}"
        xspans.append((int(sp["tok_start"]), int(sp["tok_end"])))
        xstat.append("ok")
        if bos_pos is None:
            bos_pos = int(sp["bos_pos"])
        elif bos_pos != int(sp["bos_pos"]):
            return None, f"bos_pos が区間ごとに違う ({bos_pos} vs {sp['bos_pos']})"
    if bos_pos is None:
        try:
            bos_pos = _bos_pos_from_ids(bundle, text, ids, speaker)
        except Exception as e:
            return None, f"bos_pos を決められない: {type(e).__name__}: {e}"

    # --- BOS / 話者タグのオフセットを「全区間に一度だけ」足す ---
    spans = [(int(a) + bos_pos, int(b) + bos_pos) for a, b in local]
    for i, xs in enumerate(xspans):
        if xs is not None and tuple(spans[i]) != xs:
            return None, (f"区間 {i} が単一 entity 経路と食い違う: "
                          f"cursor={tuple(spans[i])} entity_token_span={xs}")
    for a, b in spans:
        if not (0 <= a < b <= len(ids)):
            return None, f"区間 ({a}, {b}) が系列長 {len(ids)} の外"
    detail = {
        "spans": [list(x) for x in spans],
        "local": [list(x) for x in local],
        "bos_pos": int(bos_pos),
        "n_tokens": [b - a for a, b in spans],
        "resolve_via": [p["via"] for p in info["per_span"]],
        "covered": [p["covered"] for p in info["per_span"]],
        "xcheck": xstat,
        "n_tokens_total": len(ids),
    }
    return spans, detail


def resolve_item_spans(bundle, item, in_a, in_refs, speaker):
    """src 側 k 区間と、枠ごとの ref 側 1 区間を解決する。

    返り値: (s_ranges, r_ranges, tgt_span_detail) / 失敗時 (None, None, 理由)
    """
    surfaces_a = [sp["word_a"] for sp in item["spans"]]
    starts_a = [sp.get("char_start_a") for sp in item["spans"]]
    s_ranges, s_detail = resolve_spans_in_text(
        bundle, item["text_a"], in_a["input_ids"], surfaces_a, starts_a, speaker)
    if s_ranges is None:
        return None, None, f"src {s_detail}"

    r_ranges, r_details = [], []
    for i, sp in enumerate(item["spans"]):
        one, d = resolve_spans_in_text(
            bundle, item["text_refs"][i], in_refs[i]["input_ids"],
            [sp["word_b"]], [sp.get("char_start_b")], speaker)
        if one is None:
            return None, None, f"ref[{i}] {d}"
        r_ranges.append(one[0])
        r_details.append(d)

    # 重なりの検査。build_spliced_hidden_multi の中で check_span_sets が必ず
    # 落とすが、生成に入る前にここで捨てて理由を記録する。
    try:
        check_span_sets(int(in_a["input_ids"].shape[1]),
                        [int(x["input_ids"].shape[1]) for x in in_refs],
                        s_ranges, r_ranges)
    except AssertionError as e:
        return None, None, f"区間集合が不正: {e}"
    if len(s_ranges) >= 2:
        for a, b in zip(s_ranges, s_ranges[1:]):
            if a[1] > b[0]:
                return None, None, f"src の 2 区間が重なっている: {a} {b}"

    detail = {
        "src": [list(x) for x in s_ranges],
        "ref": [list(x) for x in r_ranges],
        "n": [b - a for a, b in s_ranges],                   # 元の区間長 m_i
        "n_ref": [b - a for a, b in r_ranges],               # 目標の区間長
        "gap_tokens": (s_ranges[1][0] - s_ranges[0][1]) if len(s_ranges) > 1 else None,
        "src_bos_pos": s_detail["bos_pos"],
        "ref_bos_pos": [d["bos_pos"] for d in r_details],
        "src_resolve_via": s_detail["resolve_via"],
        "src_xcheck": s_detail["xcheck"],
        "ref_xcheck": [d["xcheck"] for d in r_details],
        "src_covered": s_detail["covered"],
        "ref_covered": [d["covered"] for d in r_details],
        "T_src": int(in_a["input_ids"].shape[1]),
        "T_ref": [int(x["input_ids"].shape[1]) for x in in_refs],
    }
    detail["len_imbalance"] = (abs(detail["n"][0] - detail["n"][1])
                              if len(detail["n"]) > 1 else 0)
    detail["equal_len_per_slot"] = [detail["n"][i] == detail["n_ref"][i]
                                    for i in range(len(s_ranges))]
    return s_ranges, r_ranges, detail


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
def splice_synthesize_multi(bundle, text_a, in_a, in_refs, s_ranges, r_ranges,
                            slots, layer, gen, collect_kappa=True):
    """k 区間同時の SpeechSplice。`csm_splice.splice_synthesize` の一般化。

    単一 entity 版 (csm_splice.py:989-1054) との対応。**同じ関数を同じ引数で呼ぶ**:

      splice_synthesize                       ここ
      ------------------------------------    ------------------------------------
      1004 C.assert_splice_target(b, layer)   同じ (呼び出し側 main で 1 回 + ここ)
      1006 in_a = prepare_inputs(text_a)      呼び出し側で作った in_a をそのまま使う
      1007 in_b = prepare_inputs(text_b)      in_refs[i] (k=1 なら text_refs[0]=text_b)
      1008 s_span = entity_token_span(...)    resolve_item_spans の中で突き合わせ済み
      1015 prefill_forward(in_b, layer=L)     枠ごとに同じ呼び出し (下の for)
      1022 tap と hidden_states[L+1] の照合   同じ
      1027 tgt_hidden = tap[0, t0:t1].clone() 同じ
      1031 prefill_forward(in_a, layer=None)  同じ (kappa 用)
      1033 PrefillSplice(layer_module, ...)   MultiPrefillSplice -> 内部で
                                              PrefillSplice を区間ごとに 1 個
      1036 C._generate(b, in_a, gen, splice)  **同じ関数・同じ引数**
    """
    C.assert_splice_target(bundle, layer)
    li = C.hidden_index_for_layer(layer)

    tgt_hiddens, ref_hs = [], []
    for i in slots:
        hs, tap = C.prefill_forward(bundle, in_refs[i], layer=layer,
                                    need_all_hidden=collect_kappa)
        if tap is None:
            raise RuntimeError(f"枠 {i} の目標文脈で layer tap が取れなかった")
        if collect_kappa and hs is not None and li < len(hs):
            if not torch.allclose(tap.float(), hs[li].float(), atol=1e-3, rtol=1e-3):
                raise RuntimeError(
                    f"layers[{layer}] の tap と hidden_states[{li}] が一致しない。"
                    f"index 規約が崩れているので中断する。")
        r0, r1 = r_ranges[i]
        tgt_hiddens.append(tap[0, r0:r1].clone())
        ref_hs.append(hs)

    src_hs = None
    if collect_kappa:
        src_hs, _ = C.prefill_forward(bundle, in_a, layer=None, need_all_hidden=True)

    splice = MultiPrefillSplice(
        bundle.layer_module(layer), [s_ranges[i] for i in slots], tgt_hiddens,
        prompt_len=int(in_a["input_ids"].shape[1]),
        expect_module=bundle.layer_module(layer))
    g = C._generate(bundle, in_a, gen=gen, splice=splice)

    g.update({
        "layer": int(layer), "hidden_index": int(li),
        "prompt_len": int(in_a["input_ids"].shape[1]),
        "slots": list(slots),
        "s_ranges": [list(s_ranges[i]) for i in slots],
        "r_ranges": [list(r_ranges[i]) for i in slots],
    })
    g.update(splice.stats())
    if collect_kappa and src_hs is not None:
        ks = []
        for j, i in enumerate(slots):
            if ref_hs[j] is None:
                continue
            ks.append(C.kappa_by_layer(
                src_hs, ref_hs[j],
                {"tok_start": s_ranges[i][0], "tok_end": s_ranges[i][1]},
                {"tok_start": r_ranges[i][0], "tok_end": r_ranges[i][1]}))
        if ks:
            g["kappa_by_layer"] = [[round(v, 6) for v in k] for k in ks]
            g["kappa_at_splice"] = [round(k[li], 6) for k in ks]
    return g


# --------------------------------------------------------------------------
# 指紋
# --------------------------------------------------------------------------
AUDIO_IMPL_FUNCS = (
    "synthesize", "save_wav", "build_ref", "load_csm", "load_ref_audio",
    "expected_audio_frames", "seed_all", "codes_fingerprint",
    "assert_splice_target", "prefill_forward", "entity_token_span",
    "render_message", "prepare_inputs", "_generate", "splice_synthesize",
)
AUDIO_IMPL_METHODS = (("PrefillSplice", "__init__"), ("PrefillSplice", "__call__"),
                      ("PrefillSplice", "_replacement"),
                      ("GenConfig", "generate_kwargs"))


def code_fingerprint(pairs_module):
    """音のビットを決めるコードの指紋 (ファイルの AST + 呼び出し実体)。

    run_csm_eval.code_fingerprint と同じ作りだが、複数 entity 経路の実体
    (splice_multi と本ファイル) も入れる。整形やコメントでは動かない (AST)。
    """
    d = {
        "csm_splice_ast": R._ast_sha256(getattr(C, "__file__", None)),
        "splice_multi_ast": R._ast_sha256(sys.modules["splice_multi"].__file__),
        "driver_ast": R._ast_sha256(os.path.abspath(__file__)),
        "run_csm_eval_ast": R._ast_sha256(getattr(R, "__file__", None)),
    }
    try:
        mod = __import__(pairs_module)
        d["pairs_data_ast"] = R._ast_sha256(getattr(mod, "__file__", None))
    except Exception as e:
        d["pairs_data_ast"] = f"<import失敗:{type(e).__name__}>"
    for nm in AUDIO_IMPL_FUNCS:
        d[f"fn:{nm}"] = R._callable_id(getattr(C, nm, None))
    for cls, meth in AUDIO_IMPL_METHODS:
        d[f"fn:{cls}.{meth}"] = R._callable_id(
            getattr(getattr(C, cls, None), meth, None))
    # 複数区間側の実体
    d["fn:build_spliced_hidden_multi"] = R._callable_id(build_spliced_hidden_multi)
    d["fn:check_span_sets"] = R._callable_id(check_span_sets)
    d["fn:resolve_spans_with_cursor"] = R._callable_id(resolve_spans_with_cursor)
    d["fn:verify_span_roundtrip"] = R._callable_id(verify_span_roundtrip)
    d["fn:MultiPrefillSplice.__call__"] = R._callable_id(MultiPrefillSplice.__call__)
    d["fn:MultiPrefillSplice._replacement_rows"] = R._callable_id(
        MultiPrefillSplice._replacement_rows)
    d["fn:splice_synthesize_multi"] = R._callable_id(splice_synthesize_multi)
    d["fn:resolve_item_spans"] = R._callable_id(resolve_item_spans)
    return d


def static_fingerprint(args, conds, n_spans):
    """モデルを読む前に決まる分の指紋 (run_csm_eval.static_fingerprint と同じ鍵 +
    複数 entity 固有の鍵)。CORE_FP_KEYS を欠かさないこと。"""
    rw, rt = DEFAULT_REFS[args.lang]
    ref_sha = ref_text = None
    if args.ref_mode == "clone":
        rw = args.ref_wav or rw
        rt = args.ref_text if args.ref_text is not None else rt
        ref_sha = R._sha256_file(rw) if os.path.isfile(rw) else f"<missing:{rw}>"
        ref_text = rt
    cf = code_fingerprint(args.pairs_module)
    return {
        "lang": args.lang,
        "layer": int(args.layer),
        "seed": int(args.seed),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "depth_temperature": float(args.depth_temperature),
        "depth_top_k": int(args.depth_top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "min_sec": float(args.min_sec),
        "fail_on_cap": bool(args.fail_on_cap),
        "model_id": str(args.model_id),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
        "device_type": str(args.device).split(":")[0],
        "speaker": str(args.speaker),
        "ref_mode": str(args.ref_mode),
        "ref_sha256": ref_sha,
        "ref_text_sha256": R._sha256_text(ref_text),
        "ref_max_sec": float(args.ref_max_sec),
        "splice_target": f"backbone_model.layers[{int(args.layer)}].output",
        "splice_len_policy": "token_wise|mean_pool_broadcast",
        # 複数 entity 固有。長さを変えない一括 concat であることを鍵に残す。
        "pairs_module": str(args.pairs_module),
        "n_spans": int(n_spans),
        "conds": list(conds),
        "splice_multi_policy": "length_preserving_concat|delta_tok=0",
        "splice_impl_sha256": R._sha256_obj(
            {k: v for k, v in cf.items() if "splice" in k or k.endswith("_ast")}),
        "code_fp_hash": R._sha256_obj(cf),
        "code_fp": cf,
        "driver_ast": R._ast_sha256(os.path.abspath(__file__)),
        # 本スクリプトは書き起こしをしない (採点は score_multientity.py)。
        # CORE_FP_KEYS に含まれる鍵なので値は置く。
        "asr_model": None,
        "run_asr": False,
        "wav_dir": os.path.abspath(args.wav_dir) if args.wav_dir else None,
        "transformers_version": R._tf_version(),
    }


def cond_input_fp(item, cond, slots, fp):
    """その wav を作った入力そのもののハッシュ (wav 1 本ごとに台帳へ書く)。"""
    payload = [cond, item["text_a"]]
    for i in slots:
        sp = item["spans"][i]
        payload += [int(i), sp["word_a"], sp["word_b"], item["text_refs"][i]]
    if slots:
        payload.append(int(fp.get("layer", -1)))
    return R._sha256_obj({
        "pair_idx": int(item["pair_idx"]), "cond": cond, "payload": payload,
        "speaker": fp.get("speaker"), "ref_mode": fp.get("ref_mode"),
        "ref_sha256": fp.get("ref_sha256"),
        "ref_text_sha256": fp.get("ref_text_sha256"),
        "ref_max_sec": fp.get("ref_max_sec"),
    })


def want_input_fps(items, cond_slots, fp):
    return {(int(it["pair_idx"]), c): cond_input_fp(it, c, cond_slots[c], fp)
            for it in items for c in cond_slots}


# --------------------------------------------------------------------------
# データ
# --------------------------------------------------------------------------
def load_items(pairs_module, n, lang):
    """2 枠の multientity データ、または単一 entity の cross-culture データを読む。

    cross-culture (`data/<lang>_cross_culture_names.py`) は **合格条件の
    単一区間試験**のために 1 枠の形に落として読む。文字オフセットは
    `csm_splice._entity_char_span` (語境界つき・一意) で取るので、
    単一 entity 経路と同じ位置になる。
    """
    mod = __import__(pairs_module)
    if hasattr(mod, "build_pairs"):
        raw = mod.build_pairs()
        items = []
        for it in raw:
            if len(it["spans"]) != 2 or len(it["text_refs"]) != 2:
                raise SystemExit(f"pair {it['pair_idx']}: 2 枠のデータでない")
            items.append(it)
        k = 2
    elif hasattr(mod, "build_300_pairs"):
        raw = mod.build_300_pairs()
        items = []
        for i, p in enumerate(raw):
            ca = C._entity_char_span(p["text_a"], p["word_a"])[0]
            cb = C._entity_char_span(p["text_b"], p["word_b"])[0]
            items.append({
                "pair_idx": int(p.get("pair_idx", i)),
                "template_id": None, "lang": lang,
                "text_a": p["text_a"], "text_refs": [p["text_b"]],
                "text_both": p["text_b"], "gap_words": None,
                "slot1_at_start": p["text_a"].startswith(p["word_a"]),
                "spans": [{"slot": 1, "word_a": p["word_a"], "word_b": p["word_b"],
                           "char_start_a": int(ca), "char_start_b": int(cb)}],
            })
        k = 1
    else:
        raise SystemExit(f"data/{pairs_module}.py に build_pairs も "
                         f"build_300_pairs も無い")
    if lang != "en":
        raise SystemExit(
            f"--lang {lang} は対象外。sesame/csm-1b のモデルカードの言語タグは en のみ。")
    for it in items:
        if it.get("lang") not in (None, lang):
            raise SystemExit(f"pair {it['pair_idx']}: データの lang が {it['lang']}")
    return (items[:n] if n else items), k


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ref_results_path(ref_dir, given=None):
    """比較先の wav を書いた run の results.json を探す (台帳の producers から)。"""
    if given:
        return given if os.path.isfile(given) else None
    man = os.path.join(ref_dir, "_wav_manifest.json")
    if not os.path.isfile(man):
        return None
    try:
        with open(man) as f:
            prod = json.load(f).get("producers") or []
    except Exception:
        return None
    for p in reversed(prod):
        d = str(p.get("out_dir") or "")
        for cand in (d, d.replace("/mnt", PROJECT_ROOT, 1)):
            q = os.path.join(cand, "results.json")
            if os.path.isfile(q):
                return q
    return None


def equiv_check(out_dir, ref_dir, items, conds, ref_results=None):
    """単一区間のときに、既存の単一 entity 経路の出力と突き合わせる。

    比較表 (単一区間版 -> 既存):
        s####_bl.wav   -> s####_bl.wav    差し替えなしの経路が同じか
        s####_inj1.wav -> s####_inj.wav   **合格条件の本体**

    3 段で見る。どれが食い違うかで原因の切り分けができる:
      1. wav の md5 と sha256 … 最終成果物。これが一致すれば完全一致。
      2. codes_fp            … 生成した codebook 列の指紋 (results.json に記録)。
                               md5 だけ違うなら差は wav 書き出し側、
                               codes_fp から違うなら差し替え経路そのもの。
      3. 区間と splice_mode  … s_span / t_span / m / n / prompt_len。
                               ここが違えば区間の解決が別物。
    """
    pairs = {"bl": "bl", "inj1": "inj"}
    rows, n_same, n_diff, n_miss = [], 0, 0, 0
    for it in items:
        pidx = int(it["pair_idx"])
        for c, oc in pairs.items():
            if c not in conds:
                continue
            new = os.path.join(out_dir, f"s{pidx:04d}_{c}.wav")
            old = os.path.join(ref_dir, f"s{pidx:04d}_{oc}.wav")
            rec = {"pair_idx": pidx, "cond": c, "ref_cond": oc,
                   "new": os.path.basename(new), "old": old}
            if not (os.path.isfile(new) and os.path.isfile(old)):
                rec["result"] = "missing"
                n_miss += 1
            else:
                a, b = _md5(new), _md5(old)
                rec["md5_new"], rec["md5_old"] = a, b
                rec["sha256_new"] = R._sha256_file(new)
                rec["sha256_old"] = R._sha256_file(old)
                rec["bytes_new"] = os.path.getsize(new)
                rec["bytes_old"] = os.path.getsize(old)
                rec["result"] = "same" if a == b else "DIFF"
                n_same += int(a == b)
                n_diff += int(a != b)
            rows.append(rec)
            print(f"  [{pidx:04d}] {c} vs {oc}: {rec['result']} "
                  f"md5 {rec.get('md5_new', '-')} / {rec.get('md5_old', '-')}",
                  flush=True)

    out = {"ref_dir": ref_dir, "n_same": n_same, "n_diff": n_diff,
           "n_missing": n_miss, "rows": rows}

    # --- codes_fp と区間の突き合わせ (results.json があるときだけ) ---
    rp = _ref_results_path(ref_dir, ref_results)
    out["ref_results"] = rp
    if rp is None:
        print("[equiv] 比較先の results.json が見つからない。wav の md5 だけ見た。",
              flush=True)
        return out
    try:
        with open(rp) as f:
            ref_rows = {int(r["pair_idx"]): r for r in json.load(f)["rows"]}
    except Exception as e:
        out["ref_results_error"] = f"{type(e).__name__}: {e}"
        return out
    try:
        with open(os.path.join(out_dir, "results.json")) as f:
            my_rows = {int(r["pair_idx"]): r for r in json.load(f)["rows"]}
    except Exception as e:
        out["own_results_error"] = f"{type(e).__name__}: {e}"
        return out

    det, n_cf_same, n_cf_diff = [], 0, 0
    for it in items:
        pidx = int(it["pair_idx"])
        a, b = my_rows.get(pidx), ref_rows.get(pidx)
        if not a or not b:
            continue
        d = {"pair_idx": pidx,
             "codes_fp": {}, "span": {}}
        for c, oc in pairs.items():
            if c not in a.get("conds", {}):
                continue
            mine = a["conds"][c].get("codes_fp")
            theirs = b.get(f"{oc}_codes_fp")
            d["codes_fp"][c] = {"new": mine, "old": theirs,
                                "same": bool(mine == theirs and mine is not None)}
            n_cf_same += int(d["codes_fp"][c]["same"])
            n_cf_diff += int(not d["codes_fp"][c]["same"])
        d["span"] = {
            "s_new": a["span_tok"]["src"][0], "s_old": b.get("s_span"),
            "t_new": a["span_tok"]["ref"][0], "t_old": b.get("t_span"),
            "prompt_len_new": a.get("prompt_len"),
            "prompt_len_old": b.get("prompt_len"),
            "m_new": a["span_tok"]["n"][0], "m_old": b.get("m_src_entity_tokens"),
            "n_new": a["span_tok"]["n_ref"][0], "n_old": b.get("n_tgt_entity_tokens"),
            "mode_new": (a["conds"].get("inj1") or {}).get("splice_mode"),
            "mode_old": b.get("splice_mode"),
        }
        mn = d["span"]["mode_new"]
        mn = mn[0] if isinstance(mn, list) and len(mn) == 1 else mn
        d["span"]["mode_same"] = bool(mn == d["span"]["mode_old"])
        d["span"]["same"] = bool(
            d["span"]["s_new"] == d["span"]["s_old"]
            and d["span"]["t_new"] == d["span"]["t_old"]
            and d["span"]["prompt_len_new"] == d["span"]["prompt_len_old"]
            and d["span"]["m_new"] == d["span"]["m_old"]
            and d["span"]["n_new"] == d["span"]["n_old"]
            and d["span"]["mode_same"])
        det.append(d)
        print(f"  [{pidx:04d}] codes_fp "
              + " ".join(f"{c}:{'same' if v['same'] else 'DIFF'}"
                         for c, v in d["codes_fp"].items())
              + f" / span:{'same' if d['span']['same'] else 'DIFF'} "
              f"s={d['span']['s_new']}/{d['span']['s_old']} "
              f"t={d['span']['t_new']}/{d['span']['t_old']}", flush=True)
    out["codes_fp_same"] = n_cf_same
    out["codes_fp_diff"] = n_cf_diff
    out["n_span_diff"] = sum(1 for d in det if not d["span"]["same"])
    out["detail"] = det
    print(f"[equiv] wav 一致 {n_same} / 不一致 {n_diff} / 欠落 {n_miss} ; "
          f"codes_fp 一致 {n_cf_same} / 不一致 {n_cf_diff} ; "
          f"区間の食い違い {out['n_span_diff']}", flush=True)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="CSM-1B 複数 entity 同時 splice (BL/INJ_1/INJ_2/INJ_12)")
    p.add_argument("--pairs_module", default="en_multientity_300",
                   help="data/<name>.py。2 枠なら build_pairs、"
                        "単一区間試験なら en_cross_culture_names")
    p.add_argument("--lang", default="en", choices=["en"],
                   help="CSM-1B は英語のみ (モデルカードの言語タグが en だけ)")
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--layer", type=int, default=C.DEFAULT_LAYER,
                   help=f"backbone_model.layers[L] の出力を差し替える。既定 "
                        f"{C.DEFAULT_LAYER} (= L-2)")
    p.add_argument("--out_dir", default=None,
                   help="wav も results.json もここ (採点器がこの 1 箇所を読む)")
    p.add_argument("--allow_download", action="store_true")
    p.add_argument("--seed", type=int, default=C.SEED)

    p.add_argument("--model_id", default=C.DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # **float32 が既定。** bfloat16 だと Mimi の decode で
    # "Input type (float) and bias type (c10::BFloat16) should be the same" で落ちる。
    # 既存の単一 entity 300 対も float32 で作ってある (_wav_manifest.json)。
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--no_strict_weights", dest="strict_weights", action="store_false")
    p.set_defaults(strict_weights=True)

    p.add_argument("--speaker", default=C.DEFAULT_SPEAKER)
    p.add_argument("--ref_mode", default="clone", choices=["clone", "none"])
    p.add_argument("--ref_wav", default=None)
    p.add_argument("--ref_text", default=None)
    p.add_argument("--ref_max_sec", type=float, default=10.0)

    p.add_argument("--temperature", type=float, default=C.DEFAULT_TEMPERATURE)
    p.add_argument("--depth_temperature", type=float, default=C.DEFAULT_DEPTH_TEMPERATURE)
    p.add_argument("--depth_top_k", type=int, default=C.DEFAULT_DEPTH_TOP_K)
    p.add_argument("--max_new_tokens", type=int, default=C.DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--greedy", dest="do_sample", action="store_false")
    p.set_defaults(do_sample=True)
    p.add_argument("--min_sec", type=float, default=0.5)
    p.add_argument("--fail_on_cap", action="store_true")
    p.add_argument("--no_kappa", dest="collect_kappa", action="store_false")
    p.set_defaults(collect_kappa=True)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force_resume_mismatch", action="store_true")
    p.add_argument("--adopt_unmanifested_wavs", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--equiv_check_dir", default=None,
                   help="単一区間のとき、ここの s####_{bl,inj}.wav と md5 / sha256 を"
                        "突き合わせる (読むだけ)")
    p.add_argument("--equiv_ref_results", default=None,
                   help="比較先の results.json (codes_fp と区間の突き合わせに使う)。"
                        "省略すると台帳の producers から探す")
    return p


def main():
    args = build_argparser().parse_args()
    os.chdir(PROJECT_ROOT)
    args.asr_only = False           # run_csm_eval の台帳関数が見る
    args.run_asr = False

    items, k = load_items(args.pairs_module, args.n, args.lang)
    conds, cond_slots = conditions_for(k)
    patch_run_csm_eval(conds)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            "eval_results", f"csm_multientity_{args.lang}_{len(items)}_L{args.layer}")
    args.out_dir = os.path.abspath(args.out_dir)
    # wav と results.json は同じディレクトリ (score_multientity.py が --dir 1 つを読む)
    args.wav_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # 単一 entity の完成品 (eval_results/_cc_wavs_a/*) を指していないか。
    # あちらの wav 名 (s####_{inj,tgt}.wav) がある所には絶対に書かない。
    stray = [f for f in os.listdir(args.out_dir)
             if re.match(r"^s\d{4}_(inj|tgt)\.wav$", f)]
    if stray:
        raise SystemExit(
            f"[out_dir] {args.out_dir} に単一 entity 経路の wav が {len(stray)} 本ある "
            f"({sorted(stray)[:5]})。複数 entity の出力先を別に取ること。"
            f"既存の成果物を退避してしまうので続行しない。")

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ckpt_path = os.path.join(args.out_dir, "checkpoint.json")
    results_path = os.path.join(args.out_dir, "results.json")

    print(f"[info] {args.pairs_module}: {len(items)} items / 区間数 k={k} / "
          f"条件 {conds}", flush=True)
    print(f"[info] run_id={run_id} out_dir={args.out_dir}", flush=True)
    print(f"[info] HF_HOME={os.environ['HF_HOME']}", flush=True)

    fp = static_fingerprint(args, conds, k)
    st_hash, full_hash = R.audio_fp_hashes(fp)
    code_hash = fp["code_fp_hash"]
    print(f"[info] 設定の指紋={st_hash} 実装の指紋={code_hash} "
          f"driver={fp['driver_ast']}", flush=True)

    resume_warnings, prov_warnings = [], []
    rows_by_idx, skipped_by_idx = {}, {}
    if os.path.isfile(ckpt_path) and not args.overwrite:
        try:
            with open(ckpt_path) as f:
                prev = json.load(f)
        except Exception as e:
            raise SystemExit(f"[resume] {ckpt_path} を読めなかった: {e}\n"
                             f"  作り直すなら --overwrite。")
        prev_rows = prev.get("rows", [])
        resume_warnings = R.check_resume_fingerprint(
            prev, fp, len(prev_rows), args.force_resume_mismatch)
        for r in prev_rows:
            rows_by_idx[int(r["pair_idx"])] = r
        for s in prev.get("skipped", []):
            skipped_by_idx[int(s["pair_idx"])] = s
        print(f"[resume] {ckpt_path} から {len(rows_by_idx)} 行 / "
              f"{len(skipped_by_idx)} skip を読んだ", flush=True)

    attempted = [int(it["pair_idx"]) for it in items]
    scope = set(attempted)
    item_by_idx = {int(it["pair_idx"]): it for it in items}
    want_input = want_input_fps(items, cond_slots, fp)

    man = R.load_manifest(args.wav_dir)
    n_wavs_on_disk = len(R.scan_wav_dir(args.wav_dir))
    prov_warnings += R.check_manifest_header(man, fp, st_hash, None, args,
                                             n_wavs_on_disk, phase="static")

    # ---------------- モデル ----------------
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.model_id} (device={args.device} "
          f"dtype={args.dtype}) ...", flush=True)
    t0 = time.time()
    bundle = C.load_csm(args.model_id, device=args.device, dtype=dtype,
                        allow_download=args.allow_download,
                        attn_implementation=args.attn_implementation,
                        strict_weights=args.strict_weights)
    print(f"[info] loaded in {time.time() - t0:.1f}s from {bundle.source}", flush=True)
    C.assert_splice_target(bundle, args.layer)
    if not getattr(bundle.tokenizer, "is_fast", False):
        raise SystemExit("fast tokenizer が要る (区間の解決に offset_mapping を使う。"
                         "文字数で数える fallback は意図的に作っていない)")

    fp["model_revision"] = os.path.basename(str(bundle.source).rstrip("/"))
    fp["num_layers"] = int(bundle.num_layers)
    fp["sampling_rate"] = int(bundle.sampling_rate)
    fp["num_codebooks"] = int(bundle.num_codebooks)
    st2, full_hash = R.audio_fp_hashes(fp)
    assert st2 == st_hash
    if rows_by_idx and os.path.isfile(ckpt_path) and not args.overwrite:
        try:
            with open(ckpt_path) as f:
                _prev = json.load(f)
        except Exception:
            _prev = {}
        resume_warnings += R.check_resume_fingerprint(
            _prev, fp, len(rows_by_idx), args.force_resume_mismatch)
    prov_warnings += R.check_manifest_header(man, fp, st_hash, full_hash, args,
                                             n_wavs_on_disk, phase="final")

    trusted, quarantined, w = R.audit_wav_dir(
        args.wav_dir, man, st_hash, full_hash, scope, args, run_id,
        quarantine=True, code_hash=code_hash, want_input=want_input)
    prov_warnings += w
    if man is None:
        man = R.new_manifest(args.wav_dir, fp, st_hash, full_hash)
        if args.adopt_unmanifested_wavs:
            for pidx, cond, path in R.scan_wav_dir(args.wav_dir):
                R.register_wav(man, pidx, cond, path, st_hash, full_hash,
                               args.out_dir, run_id, {}, code_hash=code_hash,
                               input_fp=want_input.get((pidx, cond)))
                trusted.add((pidx, cond))
    man["schema"] = R.MANIFEST_SCHEMA
    man["wav_dir"] = args.wav_dir
    man["audio_fp_static_hash"] = st_hash
    man["audio_fp_full_hash"] = full_hash
    man["audio_fingerprint"] = R.audio_fp_view(fp, dynamic=True)
    man["code_fp_hash"] = code_hash
    man["code_fp"] = fp.get("code_fp")
    man["driver_ast"] = fp.get("driver_ast")
    man.setdefault("wavs", {})
    for src, _dst, _why in quarantined:
        man["wavs"].pop(os.path.basename(src)[:-4], None)
    for kk, e in list(man["wavs"].items()):
        fn = e.get("file", kk + ".wav") if isinstance(e, dict) else kk + ".wav"
        if not os.path.isfile(os.path.join(args.wav_dir, fn)):
            man["wavs"].pop(kk, None)
    man.setdefault("producers", []).append(
        {"run_id": run_id, "out_dir": args.out_dir,
         "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
         "code_fp_hash": code_hash, "driver_ast": fp.get("driver_ast"),
         "n": len(items), "layer": args.layer, "mode": "generate_multientity"})
    man["producers"] = man["producers"][-20:]
    R.save_manifest(man, args.wav_dir)
    print(f"[wav_dir] 台帳 {R.manifest_path(args.wav_dir)} / 設定 {st_hash}/{full_hash}"
          f" / 実装 {code_hash} / 照合済み {len(trusted)} 本 / 退避 "
          f"{len(quarantined)} 本", flush=True)

    ref = None
    meta = {"run_id": run_id, "pairs_module": args.pairs_module,
            "n_spans": k, "conds": conds, "ref_mode": args.ref_mode}
    if args.ref_mode == "clone":
        rw, rt = DEFAULT_REFS[args.lang]
        rw = args.ref_wav or rw
        rt = args.ref_text if args.ref_text is not None else rt
        if not os.path.isfile(rw):
            raise SystemExit(f"参照 wav が無い: {rw}")
        ref = C.build_ref(bundle, rw, rt, speaker=args.speaker,
                          max_sec=args.ref_max_sec)
        print(f"[info] ref={rw} ({ref.orig_sr} Hz -> {bundle.sampling_rate} Hz, "
              f"{ref.n_samples} samples, {ref.n_frames} audio frames)", flush=True)
        meta.update({"ref_wav": rw, "ref_text": rt, "ref_n_frames": ref.n_frames})
    else:
        print("[warn] --ref_mode none。CSM は文脈が無いと話者が毎回変わる。", flush=True)

    gen = C.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      temperature=args.temperature,
                      depth_temperature=args.depth_temperature,
                      depth_top_k=args.depth_top_k,
                      max_new_tokens=args.max_new_tokens, min_sec=args.min_sec,
                      fail_on_cap=args.fail_on_cap)
    meta["gen"] = gen.as_dict()
    meta["model_source"] = bundle.source
    meta["splice_target"] = fp["splice_target"]

    def flush(final=False, equiv=None):
        rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]
        skipped = [skipped_by_idx[i] for i in sorted(skipped_by_idx)]
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"rows": rows, "skipped": skipped, "args": vars(args),
                       "fingerprint": fp, "run_id": run_id}, f,
                      ensure_ascii=False, indent=1)
        os.replace(tmp, ckpt_path)
        R.save_manifest(man, args.wav_dir)
        if not final:
            return None
        summary = summarize(rows, skipped, items, conds, args, meta, fp,
                            st_hash, full_hash, code_hash, run_id,
                            trusted, quarantined,
                            R._dedupe(list(resume_warnings) + list(prov_warnings)),
                            equiv=equiv)
        tmp = results_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"summary": summary, "rows": rows, "skipped": skipped}, f,
                      ensure_ascii=False, indent=1)
        os.replace(tmp, results_path)
        return summary

    # ---------------- 本体 ----------------
    import soundfile as sf
    n_done = n_skip = n_err = 0
    t_start = time.time()
    interrupted = None
    try:
        for i, it in enumerate(items):
            pidx = int(it["pair_idx"])
            if (not args.overwrite and pidx in rows_by_idx
                    and rows_by_idx[pidx].get("status") == "ok"
                    and all((pidx, c) in trusted for c in conds)):
                n_skip += 1
                continue
            # 前回 skip した対は毎回やり直す (区間の解決は生成を伴わないので安い。
            # 実装を直したのに前回の skip が残り続ける経路を作らない)。
            skipped_by_idx.pop(pidx, None)

            # ---- 入力と区間 ----
            try:
                in_a = C.prepare_inputs(bundle, it["text_a"], ref=ref,
                                        speaker=args.speaker)
                in_refs = [C.prepare_inputs(bundle, t, ref=ref, speaker=args.speaker)
                           for t in it["text_refs"]]
            except Exception as e:
                print(f"  [{pidx}] prepare_inputs fail: {str(e)[:160]}", flush=True)
                skipped_by_idx[pidx] = {"pair_idx": pidx,
                                        "reason": f"prepare_inputs: {e}"[:300]}
                continue
            s_ranges, r_ranges, detail = resolve_item_spans(
                bundle, it, in_a, in_refs, args.speaker)
            if s_ranges is None:
                print(f"  [{pidx}] SKIP {detail}", flush=True)
                skipped_by_idx[pidx] = {"pair_idx": pidx, "reason": detail}
                continue

            row = {
                "pair_idx": pidx, "mode": "splice_multi_csm",
                "template_id": it.get("template_id"), "lang": args.lang,
                "layer": int(args.layer), "seed": int(args.seed),
                "run_id": run_id,
                "text_a": it["text_a"], "text_refs": it["text_refs"],
                "text_both": it.get("text_both"),
                "gap_words": it.get("gap_words"),
                "slot1_at_start": it.get("slot1_at_start"),
                "repeat_surface": it.get("repeat_surface", False),
                "words_a": [sp["word_a"] for sp in it["spans"]],
                "words_b": [sp["word_b"] for sp in it["spans"]],
                "span_tok": detail,
                "T_src": int(in_a["input_ids"].shape[1]),
                "prompt_len": int(in_a["input_ids"].shape[1]),
                "conds": {},
            }

            gens, fail_conds = {}, []
            for cond in conds:
                slots = cond_slots[cond]
                try:
                    if slots:
                        g = splice_synthesize_multi(
                            bundle, it["text_a"], in_a, in_refs, s_ranges, r_ranges,
                            slots, args.layer, gen,
                            collect_kappa=args.collect_kappa)
                    else:
                        # 差し替えなし。単一 entity 経路の BL と同じ関数・同じ引数。
                        g = C.synthesize(bundle, it["text_a"], ref=ref,
                                         speaker=args.speaker, gen=gen)
                except Exception as e:
                    import traceback
                    row[f"{cond}_status"] = "error"
                    row[f"{cond}_err"] = f"{type(e).__name__}: {e}"[:300]
                    row[f"{cond}_trace"] = traceback.format_exc()[-400:]
                    fail_conds.append(cond)
                    print(f"  [{pidx}] {cond} error: {row[f'{cond}_err']}", flush=True)
                    if args.stop_on_error:
                        row["status"] = "error"
                        row["fail_conds"] = sorted(set(fail_conds))
                        rows_by_idx[pidx] = row
                        flush()
                        raise
                    continue
                gens[cond] = g
                row[f"{cond}_status"] = "ok"

            # 4 条件で prompt が 1 トークンも違わないこと (差は splice だけ)。
            check_failed = None
            plens = {c: int(g["prompt_len"]) for c, g in gens.items()}
            if len(set(plens.values())) > 1:
                check_failed = "prompt_len_check"
                row["err"] = f"条件間で prompt 長が違う: {plens}"
                print(f"  [{pidx}] {row['err']}", flush=True)
            if check_failed is not None:
                for c in conds:
                    row[f"{c}_status"] = "error"
                    row.setdefault(f"{c}_err", f"{check_failed}: wav を保存していない")

            # ---- 保存 + 記録 ----
            if check_failed is None:
                for cond in list(gens):
                    g = gens[cond]
                    wav_path = os.path.join(args.out_dir, f"s{pidx:04d}_{cond}.wav")
                    try:
                        _, dur, qc = C.save_wav(wav_path, g["audio"],
                                                bundle.sampling_rate)
                        info = sf.info(wav_path)
                        rec = {
                            "wav": os.path.basename(wav_path),
                            "delta_tok": int(g.get("delta_tok") or 0),
                            "T_out": int(g.get("T_out") or row["T_src"]),
                            "out_ranges": [list(x) for x in (g.get("out_ranges") or [])],
                            "target_text": it["text_a"],
                            "dur_sec": round(info.frames / float(info.samplerate), 4),
                            "slots": list(cond_slots[cond]),
                            "s_ranges": g.get("s_ranges", []),
                            "r_ranges": g.get("r_ranges", []),
                            "splice_mode": g.get("splice_mode"),
                            "m_src_entity_tokens": g.get("m_src_entity_tokens"),
                            "n_tgt_entity_tokens": g.get("n_tgt_entity_tokens"),
                            "fired": g.get("fired", 0),
                            "n_frames": g["n_frames"],
                            "cutoff_frames": g["cutoff_frames"],
                            "eos_found": g["eos_found"],
                            "stop_reason": g["stop_reason"],
                            "truncated_at_cap": g["truncated_at_cap"],
                            "codes_fp": g["codes_fp"],
                            "peak": qc["peak"], "n_clipped": qc["n_clipped"],
                            "kappa_at_splice": g.get("kappa_at_splice"),
                        }
                        row["conds"][cond] = rec
                        row[f"{cond}_wav"] = rec["wav"]
                        row[f"{cond}_sec"] = g["sec"]
                        row[f"{cond}_codes_fp"] = g["codes_fp"]
                        row[f"{cond}_wav_sha256"] = R.register_wav(
                            man, pidx, cond, wav_path, st_hash, full_hash,
                            args.out_dir, run_id, g, code_hash=code_hash,
                            input_fp=want_input.get((pidx, cond)))
                        trusted.add((pidx, cond))
                    except Exception as e:
                        row[f"{cond}_status"] = "error"
                        row[f"{cond}_err"] = f"save: {type(e).__name__}: {e}"[:300]
                        row["conds"].pop(cond, None)
                        fail_conds.append(cond)
                        trusted.discard((pidx, cond))
                        gens.pop(cond, None)
                        print(f"  [{pidx}] {cond} save error: "
                              f"{row[f'{cond}_err']}", flush=True)
                        if args.stop_on_error:
                            rows_by_idx[pidx] = row
                            flush()
                            raise

            # 黙った失敗の検出: splice が生成を 1 ビットも変えていないか。
            if "bl" in row["conds"]:
                blfp = row["conds"]["bl"]["codes_fp"]
                for cond in conds:
                    if cond == "bl" or cond not in row["conds"]:
                        continue
                    row["conds"][cond]["codes_fp_equals_bl"] = bool(
                        row["conds"][cond]["codes_fp"] == blfp)
            if len(row["conds"]) == len(conds):
                row["total_position_drift"] = int(
                    row["conds"][conds[-1]]["delta_tok"])

            if fail_conds or check_failed:
                row["status"] = "error"
                row["fail_conds"] = sorted(set(fail_conds))
                row["fail_stage"] = (fail_conds[0] if fail_conds else check_failed)
                n_err += 1
            else:
                row["status"] = "ok"
                n_done += 1
            rows_by_idx[pidx] = row

            if (i + 1) % args.log_every == 0:
                el = time.time() - t_start
                per = el / max(1, n_done)
                c = row["conds"]
                durs = "/".join(f"{c[x]['dur_sec']:.2f}" for x in conds if x in c)
                same = sum(1 for x in conds if x != "bl" and x in c
                           and c[x].get("codes_fp_equals_bl"))
                print(f"  [{i + 1}/{len(items)}] ok={n_done} err={n_err} "
                      f"skip={n_skip} gap_tok={detail['gap_tokens']} "
                      f"drift={row.get('total_position_drift')} dur={durs} "
                      f"same_as_bl={same} ({per:.1f}s/pair, "
                      f"eta {per * (len(items) - i - 1) / 60:.0f}min)", flush=True)
            if (i + 1) % args.save_every == 0:
                flush()
    except BaseException as _e:
        interrupted = _e
        print(f"[warn] 実行が中断した: {type(_e).__name__}: {_e}", flush=True)

    recon_q, recon_w = R.reconcile_wav_dir(
        args.wav_dir, man, rows_by_idx, scope, args, run_id, trusted,
        interrupted=(interrupted is not None))
    quarantined += recon_q
    prov_warnings += recon_w
    if interrupted is not None:
        prov_warnings.append(
            f"実行は最後まで走っていない ({type(interrupted).__name__}: "
            f"{str(interrupted)[:150]})。results.json は途中までの記録。")

    meta.update({"n_generated_this_run": n_done, "n_resumed_skipped": n_skip,
                 "n_error_this_run": n_err,
                 "interrupted": (None if interrupted is None
                                 else f"{type(interrupted).__name__}: "
                                      f"{str(interrupted)[:150]}")})
    summary = flush(final=True)

    # --- 単一区間の等価性 (results.json を書いたあとに突き合わせる) ---
    equiv = None
    if args.equiv_check_dir:
        if k != 1:
            prov_warnings.append(
                f"--equiv_check_dir は単一区間 (k=1) のときだけ意味がある。"
                f"今回は k={k} なので突き合わせていない。")
        else:
            print(f"\n[equiv] 単一区間の突き合わせ: {args.equiv_check_dir}", flush=True)
            equiv = equiv_check(args.out_dir, args.equiv_check_dir, items, conds,
                                ref_results=args.equiv_ref_results)
            if equiv["n_diff"] or equiv["n_missing"] or equiv.get("codes_fp_diff"):
                prov_warnings.append(
                    f"単一区間が既存の単一 entity 経路と一致しない: wav 一致 "
                    f"{equiv['n_same']} / 不一致 {equiv['n_diff']} / 欠落 "
                    f"{equiv['n_missing']}、codes_fp 不一致 "
                    f"{equiv.get('codes_fp_diff')}、区間の食い違い "
                    f"{equiv.get('n_span_diff')}。移植がどこかで違っている。")
            summary = flush(final=True, equiv=equiv)
    print(f"\n[info] saved {results_path}", flush=True)
    print(json.dumps({k2: summary[k2] for k2 in
                      ("n_items", "n_generated", "n_skipped", "conds",
                       "mean_dur_sec", "delta_tok", "n_codes_fp_equal_to_bl",
                       "mean_abs_position_drift") if k2 in summary},
                     ensure_ascii=False, indent=1))
    if summary.get("warnings"):
        print("\n[warnings]")
        for s in summary["warnings"]:
            print(" -", s)
    print("\n次に採点する (既存の採点器をそのまま使う):", flush=True)
    print(f"  SLURM_TIME=03:00:00 slurm_run.sh \\\n"
          f"    python3 -u /mnt/scripts/analysis/score_multientity.py \\\n"
          f"      --dir {args.out_dir.replace(PROJECT_ROOT, '/mnt')} "
          f"--lang {args.lang}", flush=True)
    if interrupted is not None:
        raise interrupted


def summarize(rows, skipped, items, conds, args, meta, fp, st_hash, full_hash,
              code_hash, run_id, trusted, quarantined, warnings, equiv=None):
    def vals(cond, key):
        return [r["conds"][cond][key] for r in rows
                if cond in r.get("conds", {}) and r["conds"][cond].get(key) is not None]

    def mean(v):
        return round(float(np.mean(v)), 4) if v else None

    s = dict(meta)
    s.update({
        "pairs_module": args.pairs_module, "lang": args.lang,
        "layer": int(args.layer), "seed": int(args.seed),
        "n_items": len(items), "n_generated": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_error": sum(1 for r in rows if r.get("status") != "ok"),
        "n_skipped": len(skipped),
        "conds": list(conds),
        "n_ok_by_cond": {c: sum(1 for r in rows if c in r.get("conds", {}))
                         for c in conds},
        "mean_dur_sec": {c: mean(vals(c, "dur_sec")) for c in conds},
        "mean_frames": {c: mean(vals(c, "n_frames")) for c in conds},
        # CSM は行数を変えないので必ず 0。0 でなければ実装が壊れている。
        "delta_tok": {c: sorted(set(vals(c, "delta_tok"))) for c in conds},
        "mean_abs_position_drift": mean([abs(r.get("total_position_drift", 0))
                                         for r in rows]),
        "n_codes_fp_equal_to_bl": {
            c: sum(1 for r in rows
                   if r.get("conds", {}).get(c, {}).get("codes_fp_equals_bl"))
            for c in conds if c != "bl"},
        "n_truncated_at_cap": {c: sum(1 for v in vals(c, "truncated_at_cap") if v)
                               for c in conds},
        "splice_mode_counts": {},
        "gap_tokens": sorted(set(r["span_tok"]["gap_tokens"] for r in rows
                                 if r.get("span_tok", {}).get("gap_tokens")
                                 is not None)),
        "mean_len_imbalance": mean([r["span_tok"]["len_imbalance"] for r in rows
                                    if "span_tok" in r]),
        "n_mean_pool_slots": sum(
            sum(1 for x in (r["span_tok"].get("equal_len_per_slot") or []) if not x)
            for r in rows if "span_tok" in r),
        "skipped_reasons": [str(x.get("reason"))[:200] for x in skipped][:50],
        "fingerprint": fp,
    })
    for c in conds:
        modes = {}
        for r in rows:
            for m in (r.get("conds", {}).get(c, {}).get("splice_mode") or []):
                modes[m] = modes.get(m, 0) + 1
        if modes:
            s["splice_mode_counts"][c] = modes
    s["wav_provenance"] = {
        "wav_dir": args.wav_dir, "manifest": R.manifest_path(args.wav_dir),
        "audio_fp_static_hash": st_hash, "audio_fp_full_hash": full_hash,
        "code_fp_hash": code_hash, "run_id": run_id,
        "n_wavs_on_disk": len(R.scan_wav_dir(args.wav_dir)),
        "n_trusted": len(trusted), "n_quarantined": len(quarantined),
        "quarantined": [os.path.basename(d) for _s, d, _w in quarantined][:50],
    }
    if equiv is not None:
        s["single_span_md5_equivalence"] = equiv
    if warnings:
        s["warnings"] = warnings
    return s


if __name__ == "__main__":
    main()
