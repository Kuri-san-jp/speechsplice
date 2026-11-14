#!/usr/bin/env python3
"""Llasa-3B で「1 文の中の 2 箇所の実体語を同時に差し替える」実験 (GPU)。

T5Gemma 版 `scripts/analysis/generate_multientity.py` の構成をそのまま写した
**新規**スクリプト。既存ファイルは 1 行も変更していない。差し替えの中身は
`scripts/projection/splice_multi.py:build_spliced_hidden_multi` を **import して
使う** (再実装していない)。区間の置換行を作る規則 (m == n なら行対行、
m != n なら目標スパンの平均を m 行に broadcast) は既存の
`llasa_splice.PrefillSplice._replacement` を **そのまま呼ぶ**。

4 条件 (T5Gemma 版と同じ鍵名):

    bl     差し替えなし
    inj1   枠 1 だけ差し替え
    inj2   枠 2 だけ差し替え
    inj12  両方差し替え     <- 同時注入の本体

Llasa 固有の事情
----------------
1. **decoder-only なので encoder 出力が無い。** 差し替え地点は
   `model.model.layers[26]` の出力 (= output_hidden_states[27]) の、prefill
   系列の中の実体語トークン行。prefill のときだけ hook を発火させる
   (`llasa_splice.PrefillSplice` の prefill 判定をそのまま踏襲)。

2. **系列長を変えられない。** T を変えると RoPE の位置と、layers[0..l] が
   既に書いた KV cache がずれる。したがって参照側スパンは
   `PrefillSplice._replacement` で **あらかじめ m_i 行に整えて** から
   splice_multi に渡す。結果として

       delta_tok = sum_i (m_i - m_i) = 0
       T_out     = T_src
       out_ranges[i] = s_ranges[i]

   になる。つまり「2 番目の区間の index が 1 番目の差し替えでずれる」問題は
   Llasa では構造的に起きない。**それでも index を自分でずらさない**
   (splice_multi の一括 concat に任せる)。

3. **長さ一致ダミーが不要。** T5Gemma 版は delta_tok != 0 のとき
   `make_length_matched_target` でダミー本文を作るが、Llasa は delta_tok が
   常に 0 なので 4 条件すべて target_text == text_a、input_ids も 1 token も
   違わない。差は prefill 中に実体語行を置き換えたかどうかだけ。

4. **chat template が実行日を prompt に差し込む。**
   `--date_string` で固定する。既定は `17 Sep 2026` で、これは既存の
   単一 entity 英語 300 対 (eval_results/_cc_wavs_a/llasa_en、2026-09-17 実行)
   の prompt に入っていた日付。md5 等価性試験を通すには必ずこの値を使う。

5. **BL と INJ がビット単位で同一になる対がある。** 単一 entity の英語 300 対
   では 300 中 156 対で `s*_bl.wav` と `s*_inj.wav` がバイト一致していた
   (m == n の対は 30、生成トークン数が一致した対は 174)。複数 entity でも
   同様に起きるので、条件ごとに wav の md5 を記録し、bl と一致した対の数を
   summary に出す。

6. **日本語は非対応** (Llasa-3B が学習していない。`run_llasa_eval.py` と同じ
   理由で ja は Llasa-1B-Multilingual 側)。**中国語は 2026-09-22 に追加した。**
   データは `data/zh_multientity_300_v2.py` (EN / JA と同じ 50 語対 × 6 文型)、
   参照音声は `run_llasa_eval.DEFAULT_REFS["zh"]` の BZNSYP 側。

7. **中国語は最小対称拡張が要る** (2026-09-22)。トークン化器が実体語と隣の
   機能語を 1 トークンに融合する対があり (「北大」が「北大的」)、語そのままでは
   往復確認で落ちて 300 対のうち 2 対 (pair_idx 221 の ref1、238 の ref0) が
   skip され n=298 になる。`--span_extension` は **lang で既定が決まる**
   (en: off / zh: on)。規則の実装は `scripts/projection/span_extend.py` 1 箇所
   だけで、単一 entity の ZH セル (eval_results/llasa_cross_culture_zh の
   manifest: "span_extension": true) も同じ規則で作ってある。語そのままで
   往復一致する枠は on/off で同じ区間になる。

使い方
------
  # (1) 単一区間の md5 等価性試験 (n=5)。既存の単一 entity 経路とバイト一致するか
  SLURM_TIME=01:00:00 slurm_run.sh \
    python3 -u scripts/llasa/run_multientity_llasa.py --mode equiv --lang en --n 5

  # (2) 複数 entity 本番 (英語 300 文)
  SLURM_TIME=20:00:00 slurm_run.sh \
    python3 -u scripts/llasa/run_multientity_llasa.py --mode multi --lang en \
      --pairs_module en_multientity_300 --n 300

  # (3) 複数 entity 本番 (中国語 300 文)。--pairs_module と --span_extension は
  #     lang から決まるので省略できる
  SLURM_TIME=20:00:00 slurm_run.sh \
    python3 -u scripts/llasa/run_multientity_llasa.py --mode multi --lang zh \
      --n 300 --out_dir eval_results/multientity_llasa_zh_300

  # (4) GPU を使わない区間の事前確認 (300/300 通るか)
  CUDA_VISIBLE_DEVICES= taskset -c 0-12 python3 \
    scripts/llasa/verify_zh_multientity_llasa_cpu.py

出力
----
  <out_dir>/results.json     summary + rows (conds に bl / inj1 / inj2 / inj12)
  <out_dir>/checkpoint.json  途中再開用
  <out_dir>/s{pair_idx:04d}_{bl,inj1,inj2,inj12}.wav
"""
import argparse
import hashlib
import json
import os
import sys
import time
from argparse import Namespace

# --- HF cache。**代入**で上書きする (setdefault はコンテナの環境変数に負ける) ---
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "modules")

PROJECT_ROOT = ROOT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "llasa"))

import llasa_splice as L  # noqa: E402  (HF_HOME はここでも同じ値に設定される)

# **再実装しない。** k 区間を一括 concat で差し替える本体はこれを使う。
from splice_multi import (  # noqa: E402
    build_spliced_hidden_multi, check_span_sets, resolve_spans_with_cursor,
    span_deltas, verify_span_roundtrip,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 既存の投入係。参照音声の符号化・bundle の読み込み・wav 書き出し・生成長集計を
# そのまま使う (参照音声トークンと wav バイト列を既存と一致させるため)。
import run_llasa_eval as R  # noqa: E402

CONDITIONS_MULTI = ["bl", "inj1", "inj2", "inj12"]
# **枠ごとの目標文（2026-09-20 追加、--tgt_only のときだけ）。**
# 複数 entity を本体セルと同じ PIB で測るには目標音声が要る。tgt1 / tgt2 は
# それぞれ 1 枠だけ差し替えた文で、基線との差が 1 箇所なので本体セルの
# 採点器（diff_block）がそのまま効く。注入はしないので splice は絡まない。
TGT_SLOT_CONDS = {"tgt1": 0, "tgt2": 1}
COND_SLOTS = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}

# 等価性試験 (単一区間) の条件。既存 run_llasa_eval.py の生成順
#   src prefill -> tgt prefill -> BL generate -> TGT generate -> INJ generate
# をそのまま再現するため、wav 名も既存と同じ bl / tgt / inj にする。
CONDITIONS_EQUIV = ["bl", "tgt", "inj"]

ROW_MODE_MULTI = "splice_multi_llasa"
ROW_MODE_EQUIV = "splice_single_via_multi_llasa"

# 既存の単一 entity 英語 300 対を作ったときに prompt に入っていた日付。
DEFAULT_DATE_STRING = "17 Sep 2026"


# ===========================================================================
# 複数区間 prefill hook
# ===========================================================================
class PrefillSpliceMulti:
    """`model.model.layers[l]` の出力に、prefill のときだけ k 区間まとめて介入する.

    既存の 1 区間版 `llasa_splice.PrefillSplice` と違うのは「k 区間を一度に
    扱う」ところだけで、

      * prefill 判定 (T <= 1 / prompt_len 違い / fired flag)
      * 置換行の作り方 (m == n なら行対行、m != n なら平均を m 行に broadcast)

    はどちらも既存のものを流用する。後者は `PrefillSplice._replacement` を
    **そのまま呼ぶ** (区間ごとに PrefillSplice を 1 個作るが、hook としては
    登録しない。使うのは _replacement と m / n / equal_len だけ)。

    置換そのものは `splice_multi.build_spliced_hidden_multi` に渡す。参照側
    スパンを先に m_i 行へ整えてあるので

        delta_tok = 0,  T_out = T_src,  out_ranges[i] = s_ranges[i]

    となり、系列長は変わらない (RoPE と KV cache の整合が保たれる)。
    **2 番目の区間の index を自分でずらすことはしない。** 一括 concat が
    src の元の index だけを読んで 1 パスで組み立てる。
    """

    def __init__(self, layer_module, s_ranges, tgt_hiddens, prompt_len=None,
                 verify_inplace_equiv=False):
        assert len(s_ranges) == len(tgt_hiddens) >= 1
        self.layer_module = layer_module
        self.s_ranges = [(int(a), int(b)) for a, b in s_ranges]
        self.prompt_len = int(prompt_len) if prompt_len is not None else None
        self.verify_inplace_equiv = bool(verify_inplace_equiv)
        # 区間ごとの置換行ビルダ。既存クラスをそのまま使う (hook 登録はしない)。
        self.units = [
            L.PrefillSplice(layer_module,
                            {"tok_start": int(a), "tok_end": int(b)},
                            t, prompt_len=prompt_len)
            for (a, b), t in zip(self.s_ranges, tgt_hiddens)
        ]
        self.handle = None
        self.fired = 0
        self.decode_calls = 0
        self.skipped_shapes = []
        self.out_ranges = None
        self.delta_tok = None
        self.inplace_equiv = None

    # --- prefill 判定は PrefillSplice と同一 --------------------------------
    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            return output
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
        for u in self.units:
            if u.s1 > T:
                raise RuntimeError(
                    f"entity span [{u.s0},{u.s1}) が prefill 長 {T} を超えている")
        if h.shape[0] != 1:
            raise RuntimeError(f"batch size {h.shape[0]} は未対応 (1 のみ)")

        # 参照側スパンを m_i 行に整える。ここが唯一の Llasa 固有部分。
        reps = [u._replacement(h.dtype).to(h.device) for u in self.units]
        src_enc = {"last_hidden_state": h}
        ref_encs = [{"last_hidden_state": r.reshape(1, r.shape[0], r.shape[1])}
                    for r in reps]
        r_ranges = [(0, int(r.shape[0])) for r in reps]

        modified, delta_tok, out_ranges = build_spliced_hidden_multi(
            src_enc, ref_encs, list(self.s_ranges), r_ranges)

        # 長さが変わっていないこと / index をずらしていないことの実地確認
        if delta_tok != 0 or modified.shape[1] != T:
            raise RuntimeError(
                f"Llasa は系列長を変えられない (delta_tok={delta_tok} "
                f"T_out={modified.shape[1]} T_src={T})")
        if [tuple(x) for x in out_ranges] != self.s_ranges:
            raise RuntimeError(
                f"out_ranges {out_ranges} が s_ranges {self.s_ranges} と違う")

        # k=1 のときは既存 1 区間 hook の in-place 置換と**バイト一致**するはず。
        # 数値を合わせるための補正ではなく、一致の確認だけを行う。
        if self.verify_inplace_equiv and len(self.units) == 1:
            probe = L.PrefillSplice(
                self.layer_module,
                {"tok_start": self.units[0].s0, "tok_end": self.units[0].s1},
                self.units[0].tgt, prompt_len=self.prompt_len)
            ref_out = probe(module, args, h)
            ref_h = ref_out[0] if isinstance(ref_out, tuple) else ref_out
            self.inplace_equiv = bool(
                ref_h.shape == modified.shape
                and ref_h.dtype == modified.dtype
                and torch.equal(ref_h.view(torch.int16), modified.view(torch.int16)))
            if not self.inplace_equiv:
                raise RuntimeError(
                    "k=1 で splice_multi の結果が既存 PrefillSplice の in-place "
                    "置換とビット一致しない。移植がどこかで違う。")

        self.fired += 1
        self.delta_tok = int(delta_tok)
        self.out_ranges = [tuple(int(v) for v in x) for x in out_ranges]
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
                f"prompt_len={self.prompt_len} skipped_shapes={self.skipped_shapes[:8]}")

    def stats(self):
        return {
            "fired": self.fired, "decode_calls": self.decode_calls,
            "skipped_shapes": self.skipped_shapes[:16],
            "k_spans": len(self.units),
            "m_src_entity_tokens": [u.m for u in self.units],
            "n_tgt_entity_tokens": [u.n for u in self.units],
            "equal_len": [u.equal_len for u in self.units],
            "splice_mode": ["token_wise" if u.equal_len else "mean_pool_broadcast"
                            for u in self.units],
            "delta_tok": self.delta_tok,
            "out_ranges": [list(x) for x in (self.out_ranges or [])],
            "inplace_equiv_k1": self.inplace_equiv,
        }


# ===========================================================================
# prompt / 区間の解決
# ===========================================================================
def render_dated(tokenizer, text, ref_speech_ids, date_string):
    """chat template を日付固定で通す。既存の render_prompt をそのまま使う."""
    return L.render_prompt(tokenizer, L.build_chat(text, ref_speech_ids),
                           date_string=date_string)


def prepare_context_multi(bundle, text, ref_speech_ids, words=(), region_text=None,
                          need_hidden=False, date_string=None, expands=None):
    """1 本の prompt について ids / k 個の実体語スパン / hidden state を作る.

    既存 `llasa_splice.prepare_context` (run_llasa_eval.py:357-362 が呼ぶもの)
    と、呼ぶ関数と引数を揃えてある。違いは
      * `render_prompt` に date_string を渡す (日付固定のため)
      * word を 1 個ではなく k 個受ける (同じ rendered を共有して k 回呼ぶ)
    だけ。`entity_token_span` / `verify_entity_span` / `prefill_hidden_states`
    は既存のものをそのまま呼ぶ。

    expands (2026-09-22 追加): 枠ごとの (pre, post)。**既定 None のときは
    `entity_token_span` に `expand` を渡さない** = 従来と同じ呼び出し。
    最小対称拡張が要る枠 (中国語の「的」融合) でだけ値を入れる。ここで拡張を
    反映しないと、`resolve_item_spans_llasa` が返す区間と食い違って
    「区間の解決が 2 経路で食い違っている」で落ちる。
    """
    r = render_dated(bundle.tokenizer, text, ref_speech_ids, date_string)
    out = {"prompt": r["prompt"], "input_ids": r["input_ids"],
           "offsets": r["offsets"], "prompt_len": len(r["input_ids"]),
           "rendered": r}
    spans = []
    for i, w in enumerate(words):
        e = None if not expands else expands[i]
        kw = {} if e is None else {"expand": (int(e[0]), int(e[1]))}
        sp = L.entity_token_span(bundle.tokenizer, text, w, rendered=r,
                                 region_text=region_text, **kw)
        L.verify_entity_span(sp, strict=True)
        spans.append(sp)
    out["spans"] = spans
    if need_hidden:
        out["hidden_states"] = L.prefill_hidden_states(bundle, r["input_ids"])
    return out


def _spans_via_splice_multi(tokenizer, prompt, surfaces, char_starts,
                            allow_symmetric_extension=False, peer_prompts=None,
                            peer_surfaces=None, peer_char_starts=None, side="a"):
    """splice_multi の解決器でも同じ区間が出ることを確かめる (交差検証).

    load-bearing なのは既存の `entity_token_span` の方 (既存の単一 entity 結果
    とバイト一致させるため)。ここは別実装で同じ index が出ることの確認と、
    往復一致 (verify_span_roundtrip) を回すためだけに呼ぶ。

    最小対称拡張 (2026-09-22 追加)
    ------------------------------
    `allow_symmetric_extension=False` (既定) のときは追加の引数を一切使わず、
    従来と同じ 2 行を呼ぶだけ。**既定の挙動は 1 ビットも変わっていない。**

    True のときは `resolve_spans_with_cursor` に対の反対側 (peer_*) を渡して
    `span_extend.minimal_symmetric_surface` で要求する表層を取り直させる。
    往復確認は **拡張後の表層** で行う (元の語で見ると拡張した対は必ず落ちる。
    `splice_multi.resolve_spans_with_cursor` の docstring に明記されている)。
    語そのままで足りる対には plain=True が返るので表層が変わらず、返る区間は
    既定経路と同一になる。
    """
    spans, info = resolve_spans_with_cursor(
        tokenizer, prompt, list(surfaces), add_bos=0,
        char_starts=list(char_starts),
        allow_symmetric_extension=bool(allow_symmetric_extension),
        peer_texts=list(peer_prompts) if peer_prompts is not None else None,
        peer_surfaces=list(peer_surfaces) if peer_surfaces is not None else None,
        peer_char_starts=(list(peer_char_starts)
                          if peer_char_starts is not None else None),
        side=side)
    if spans is None:
        return None, f"resolve_spans_with_cursor failed: {info['reason']}"
    check = list(surfaces)
    for i, e in enumerate(info.get("extension") or []):
        if i < len(check) and e.get("surface"):
            check[i] = e["surface"]
    ok, bad = verify_span_roundtrip(tokenizer, prompt, spans, check,
                                    add_bos=0)
    if not ok:
        return None, f"verify_span_roundtrip failed: {bad}"
    return [(int(a), int(b)) for a, b in spans], None


def _plan_span_extension_llasa(tokenizer, item, ref_text, date_string,
                               ref_speech_ids, in_a, r_a):
    """枠ごとに最小対称拡張が要るかを**先に**決める (中国語の「的」対策、2026-09-22).

    語そのままで src / ref の両側が往復一致する枠は `ext=None` を返す。この枠は
    拡張を計画しないので、以降の解決は既定経路と 1 ビットも変わらない。
    片側でも一致しない枠にだけ `llasa_splice._try_span_extension` を呼ぶ
    (規則の実装は `scripts/projection/span_extend.py` の 1 箇所だけ。ここでは
    呼ぶだけで書き直していない)。救えない枠も `ext=None` で返し、呼び出し側が
    従来どおり `verify_entity_span` で落とす。

    prompt も input_ids も変えない。変わるのは差し替えるトークン区間だけ。

    戻り値: 枠ごとの dict の list
      {"ext": {"pre","post","extra","surface_a","surface_b"} か None,
       "char_start_a": 拡張**前**の語の prompt 上の文字位置 (src 側),
       "char_start_b": 同 (ref 側),
       "r_b": ref 側の rendered prompt}
    """
    out = []
    for i, sp in enumerate(item["spans"]):
        sa = L.entity_token_span(tokenizer, in_a, sp["word_a"], rendered=r_a,
                                 region_text=item["text_a"])
        in_b = L.build_input_text(item["text_refs"][i], ref_text)
        r_b = render_dated(tokenizer, in_b, ref_speech_ids, date_string)
        sb = L.entity_token_span(tokenizer, in_b, sp["word_b"], rendered=r_b,
                                 region_text=item["text_refs"][i])
        rec = {"ext": None, "char_start_a": int(sa["char_span"][0]),
               "char_start_b": int(sb["char_span"][0]), "r_b": r_b}
        if not (sa["roundtrip_ok"] and sb["roundtrip_ok"]):
            src = {"prompt": r_a["prompt"], "input_ids": r_a["input_ids"],
                   "offsets": r_a["offsets"], "span": sa}
            tgt = {"prompt": r_b["prompt"], "input_ids": r_b["input_ids"],
                   "offsets": r_b["offsets"], "span": sb}
            e = L._try_span_extension(
                tokenizer, src, sp["word_a"], item["text_a"],
                tgt, sp["word_b"], item["text_refs"][i])
            if e is not None:
                rec["ext"] = {"pre": int(e["pre"]), "post": int(e["post"]),
                              "extra": e["extra"], "surface_a": e["surface_a"],
                              "surface_b": e["surface_b"]}
        out.append(rec)
    return out


def resolve_item_spans_llasa(tokenizer, item, ref_text, date_string,
                             ref_speech_ids=(0, 1, 2), span_extension=False):
    """src 側 k 区間 と ref 側 1 区間 × k を解決して detail を返す (tokenizer だけで動く).

    T5Gemma 版 `generate_multientity.resolve_item_spans` (行 121-178) と同じ流れ:
      区間の解決 -> 往復確認 -> 重なり検査。
    ただし Llasa は prompt 全体を 1 回 tokenize して offsets を取るので、
    `target_start_in_full` の加算は不要 (加算すると二重にずれる)。

    span_extension (2026-09-22 追加)
    --------------------------------
    **既定は False で、そのときは追加の処理を 1 つも通らない** (計画表は
    全枠 None のまま、`entity_token_span` にも `expand` を渡さず、交差検証も
    従来の引数で呼ぶ)。英語 300 対の解決結果は 1 ビットも変わらない。

    True のときだけ、語そのままで往復一致しない枠に最小対称拡張を試す。
    中国語では「北大」が「北大的」と 1 トークンに融合する対があり、既定のまま
    では往復確認で落ちて n=300 を割る (zh_multientity_300_v2 で 2 対)。
    """
    k = len(item["spans"])
    surfaces_a = [sp["word_a"] for sp in item["spans"]]
    ref_speech_ids = list(ref_speech_ids) if ref_speech_ids else None

    in_a = L.build_input_text(item["text_a"], ref_text)
    r_a = render_dated(tokenizer, in_a, ref_speech_ids, date_string)

    # --- 拡張の計画 (既定 False では plans は全枠 None 相当) ---
    plans = [None] * k
    if span_extension:
        plans = _plan_span_extension_llasa(tokenizer, item, ref_text,
                                           date_string, ref_speech_ids, in_a, r_a)
    exts = [None if p is None else p["ext"] for p in plans]
    use_ext = any(e is not None for e in exts)

    # --- src: 既存 1 区間経路と同じ関数・同じ引数で k 回解決する ---
    s_spans = []
    for i, w in enumerate(surfaces_a):
        # expand は拡張が要る枠にだけ渡す。渡さなければ従来と同じ呼び出し。
        kw = {} if exts[i] is None else {"expand": (exts[i]["pre"], exts[i]["post"])}
        sp = L.entity_token_span(tokenizer, in_a, w, rendered=r_a,
                                 region_text=item["text_a"], **kw)
        L.verify_entity_span(sp, strict=True)
        s_spans.append(sp)
    s_ranges = [(sp["tok_start"], sp["tok_end"]) for sp in s_spans]

    # --- splice_multi の解決器でも同じ区間が出ることを確認 (往復一致つき) ---
    # 拡張した枠があるときは対の反対側も渡し、**splice_multi 側にも独立に
    # 同じ (pre, post) を選ばせる**。char_start は拡張前の語の位置を渡すこと
    # (拡張量は向こうで引くので、拡張後の位置を渡すと二重に引かれる)。
    cross, err = _spans_via_splice_multi(
        tokenizer, r_a["prompt"], surfaces_a,
        [(p["char_start_a"] if p is not None else sp["char_span"][0])
         for p, sp in zip(plans, s_spans)],
        allow_symmetric_extension=use_ext,
        peer_prompts=([p["r_b"]["prompt"] for p in plans] if use_ext else None),
        peer_surfaces=([sp["word_b"] for sp in item["spans"]] if use_ext else None),
        peer_char_starts=([p["char_start_b"] for p in plans] if use_ext else None),
        side="a")
    if cross is None:
        raise ValueError(f"src {err}")
    if cross != s_ranges:
        raise ValueError(f"src span mismatch: splice_multi={cross} "
                         f"entity_token_span={s_ranges}")

    # --- ref: 区間ごとに別の prompt。それぞれ 1 区間だけ解決 ---
    r_ranges, ref_prompt_lens = [], []
    for i, sp in enumerate(item["spans"]):
        in_b = L.build_input_text(item["text_refs"][i], ref_text)
        r_b = render_dated(tokenizer, in_b, ref_speech_ids, date_string)
        kw = {} if exts[i] is None else {"expand": (exts[i]["pre"], exts[i]["post"])}
        tb = L.entity_token_span(tokenizer, in_b, sp["word_b"], rendered=r_b,
                                 region_text=item["text_refs"][i], **kw)
        L.verify_entity_span(tb, strict=True)
        cross_b, err = _spans_via_splice_multi(
            tokenizer, r_b["prompt"], [sp["word_b"]],
            [(plans[i]["char_start_b"] if plans[i] is not None
              else tb["char_span"][0])],
            allow_symmetric_extension=(exts[i] is not None),
            peer_prompts=([r_a["prompt"]] if exts[i] is not None else None),
            peer_surfaces=([sp["word_a"]] if exts[i] is not None else None),
            peer_char_starts=([plans[i]["char_start_a"]]
                              if exts[i] is not None else None),
            side="b")
        if cross_b is None:
            raise ValueError(f"ref[{i}] {err}")
        if cross_b != [(tb["tok_start"], tb["tok_end"])]:
            raise ValueError(f"ref[{i}] span mismatch: {cross_b} vs "
                             f"{[(tb['tok_start'], tb['tok_end'])]}")
        r_ranges.append((tb["tok_start"], tb["tok_end"]))
        ref_prompt_lens.append(len(r_b["input_ids"]))

    # --- 重なり検査。splice_multi の検査器をそのまま使う ---
    # 参照側は後で m_i 行に整えるので、ここでは「src 区間が重ならないこと」だけ
    # が意味を持つ。r_ranges は各 ref prompt 内の index なので長さも渡す。
    check_span_sets(len(r_a["input_ids"]), ref_prompt_lens, s_ranges, r_ranges)
    for a in range(k):
        for b in range(a + 1, k):
            if not (s_ranges[a][1] <= s_ranges[b][0]
                    or s_ranges[b][1] <= s_ranges[a][0]):
                raise ValueError(f"src spans overlap: {s_ranges[a]} {s_ranges[b]} "
                                 f"({surfaces_a})")
    if k >= 2 and not all(s_ranges[i][1] <= s_ranges[i + 1][0] for i in range(k - 1)):
        raise ValueError(f"src spans are not in left-to-right order: {s_ranges}")

    n = [s_ranges[i][1] - s_ranges[i][0] for i in range(k)]
    m = [r_ranges[i][1] - r_ranges[i][0] for i in range(k)]
    detail = {
        "src": [list(x) for x in s_ranges],
        "ref": [list(x) for x in r_ranges],
        "n": n, "m": m,
        "gap_tokens": (s_ranges[1][0] - s_ranges[0][1]) if k >= 2 else None,
        "src_prompt_len": len(r_a["input_ids"]),
        "ref_prompt_len": ref_prompt_lens,
        "src_tokens": [sp["entity_tokens"] for sp in s_spans],
        "delta_per_span_raw": [m[i] - n[i] for i in range(k)],
        "len_imbalance": abs(m[0] - m[1]) if k >= 2 else None,
        # 参照側を m_i 行に整えた後の Δ。Llasa は必ず 0。
        "delta_per_span_after_match": [0] * k,
        "total_position_drift": 0,
    }
    # 拡張を試したときだけ鍵を足す (既定 False の記録は従来と同じ形のまま)。
    if span_extension:
        detail["span_extension"] = exts
        detail["n_span_extended"] = int(sum(1 for e in exts if e is not None))
    return detail


# ===========================================================================
# 生成
# ===========================================================================
def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_scorers(device, enable):
    if not enable:
        return None, None
    print("[Info] Loading UTMOS22 ...", flush=True)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True).to(device).eval()
    print("[Info] Loading SQUIM-OBJECTIVE ...", flush=True)
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    squim = SQUIM_OBJECTIVE.get_model().to(device).eval()
    return utmos, squim


def score_wav_local(path, utmos, squim, device, target_sr=16000):
    """eval_utmos_speaker_fixed.score_wav (行 300-318) と同じ呼び方の局所版.

    T5Gemma 側の import を引き込まないために書き写しているだけで、
    UTMOS / SQUIM の呼び出し引数は同一。
    """
    if utmos is None:
        return None, None
    import torchaudio
    wav, sr_in = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr_in != target_sr:
        wav = torchaudio.functional.resample(wav, sr_in, target_sr)
    with torch.no_grad():
        u = float(utmos(wav.to(device), target_sr).item())
        try:
            _, pesq, _ = squim(wav.to(device))
            p = float(pesq.item())
        except Exception:
            p = None
    return u, p


def gen_plain(bundle, input_ids, gen):
    """差し替えなしの生成。既存 generate_speech_ids をそのまま使う (seed は中でリセット)."""
    g = L.generate_speech_ids(bundle, input_ids, gen=gen)
    return g, None


def gen_spliced(bundle, input_ids, gen, layer, s_ranges, tgt_hiddens, prompt_len,
                verify_inplace_equiv):
    sp = PrefillSpliceMulti(bundle.model.model.layers[layer], s_ranges,
                            tgt_hiddens, prompt_len=prompt_len,
                            verify_inplace_equiv=verify_inplace_equiv)
    g = L.generate_speech_ids(bundle, input_ids, gen=gen, splice=sp)
    return g, sp


def build_ns_for_existing_helpers(args):
    """run_llasa_eval.load_bundle / build_ref が読む属性だけを持つ Namespace."""
    return Namespace(
        model_id=args.model_id, codec_id=args.codec_id, device=args.device,
        dtype=args.dtype, allow_download=args.allow_download,
        attn_implementation=args.attn_implementation,
        lang=args.lang, ref_mode=args.ref_mode, ref_wav=args.ref_wav,
        ref_text=args.ref_text, ref_max_sec=args.ref_max_sec,
        no_codec_check=args.no_codec_check,
    )


def load_items(args):
    """mode に応じて item を作る。どちらも spans / text_refs 形式に揃える."""
    if args.mode == "multi":
        mod = __import__(args.pairs_module)
        items = mod.build_pairs()
        for it in items:
            assert len(it["spans"]) == 2, \
                f"pair {it['pair_idx']}: 2 区間であること"
            assert len(it["text_refs"]) == 2
        return items[:args.n] if args.n else items
    # equiv: cross_culture 300 対を「区間 1 つの多 entity item」に写す
    if args.lang != "en":
        raise SystemExit("equiv mode は英語のみ (既存の単一 entity 結果が EN)")
    from en_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()
    items = []
    for i, p in enumerate(pairs):
        items.append({
            "pair_idx": p.get("pair_idx", i), "template_id": None, "lang": "en",
            "text_a": p["text_a"], "text_refs": [p["text_b"]],
            "text_both": p["text_b"], "gap_words": None, "slot1_at_start": None,
            "spans": [{"slot": 1, "word_a": p["word_a"], "word_b": p["word_b"]}],
        })
    return items[:args.n] if args.n else items


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # **目標文だけを後から足す（2026-09-20）。**既存の注入音声は作り直さない。
    ap.add_argument("--tgt_only", action="store_true",
                    help="枠ごとの目標文だけを合成する")
    ap.add_argument("--mode", default="multi", choices=["multi", "equiv"],
                    help="multi: 2 区間同時。equiv: 区間 1 つで既存の単一 entity と md5 比較")
    ap.add_argument("--lang", default="en", choices=["en", "zh"],
                    help="en / zh。日本語は Llasa-3B が非対応なので入れていない "
                         "(run_llasa_eval.py と同じ理由。ja は Llasa-1B-Multilingual)。"
                         "zh は 2026-09-22 追加: 参照音声は run_llasa_eval.DEFAULT_REFS "
                         "の BZNSYP 側、データは zh_multientity_300_v2")
    # 既定を言語で決める (下で解決する)。en に解決したときの値は従来と同じ
    # "en_multientity_300" なので、英語の既存の呼び出しは何も変わらない。
    ap.add_argument("--pairs_module", default=None,
                    help="既定は lang で決まる (en: en_multientity_300 / "
                         "zh: zh_multientity_300_v2)")
    ap.add_argument("--n", type=int, default=0, help="0 なら全件")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--date_string", default=DEFAULT_DATE_STRING,
                    help="chat template の Today Date を固定する。既定は既存の "
                         "単一 entity 英語 300 対を作った日 (17 Sep 2026)")
    ap.add_argument("--equiv_ref_dir", default=None,
                    help="equiv mode で md5 を比べる既存 wav の場所 "
                         "(既定 eval_results/_cc_wavs_a/llasa_en)")
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--utmos", dest="utmos", action="store_true", default=None)
    ap.add_argument("--no_utmos", dest="utmos", action="store_false")
    ap.add_argument("--verify_inplace_equiv", dest="verify_inplace_equiv",
                    action="store_true", default=None,
                    help="k=1 の区間で、splice_multi の結果が既存 PrefillSplice の "
                         "in-place 置換とビット一致することを実行時に確かめる。"
                         "既定は multi mode で on、equiv mode で off "
                         "(md5 試験では余計なテンソル確保を GPU に持ち込まない)")
    ap.add_argument("--no_verify_inplace_equiv", dest="verify_inplace_equiv",
                    action="store_false")
    # **最小対称拡張 (2026-09-22 追加)。**既定は None で、下で
    #   args.span_extension = (args.lang != "en")
    # に解決する。英語は False に解決するので、既存の呼び出しは
    # このフラグを付けても付けなくても従来と同じ経路を通る。中国語は
    # 「北大」が「北大的」に融合する対が 2 つあり、off だと n=298 になる。
    # 単一 entity の ZH セル (eval_results/llasa_cross_culture_zh の manifest:
    # "span_extension": true) も拡張ありで作ってあるので、列として揃う。
    ap.add_argument("--span_extension", dest="span_extension",
                    action="store_true", default=None,
                    help="語そのままで往復一致しない枠に最小対称拡張 "
                         "(scripts/projection/span_extend.py) を試す。"
                         "既定は lang で決まる (en: off / zh: on)。"
                         "往復一致する枠の区間は on/off で 1 ビットも変わらない")
    ap.add_argument("--no_span_extension", dest="span_extension",
                    action="store_false",
                    help="拡張を切る (従来どおり往復一致しない枠は文ごと skip)")
    ap.add_argument("--no_kappa", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    # モデル / 復号 / 参照音声。既定は run_llasa_eval.py と同じ値にそろえる。
    ap.add_argument("--model_id", default=L.DEFAULT_MODEL_ID)
    ap.add_argument("--codec_id", default=L.DEFAULT_CODEC_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--allow_download", action="store_true")
    ap.add_argument("--seed", type=int, default=L.SEED)
    ap.add_argument("--greedy", dest="do_sample", action="store_false")
    ap.set_defaults(do_sample=True)
    ap.add_argument("--temperature", type=float, default=L.DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=L.DEFAULT_TOP_P)
    ap.add_argument("--top_k", type=int, default=L.DEFAULT_TOP_K)
    ap.add_argument("--max_new_tokens", type=int, default=L.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--ref_mode", default="clone", choices=["clone", "none"])
    ap.add_argument("--ref_wav", default=None)
    ap.add_argument("--ref_text", default=None)
    ap.add_argument("--ref_max_sec", type=float, default=15.0)
    ap.add_argument("--no_codec_check", action="store_true")
    args = ap.parse_args()

    if args.utmos is None:
        # 本番は T5Gemma 版と同じく UTMOS を出す。等価性試験では余計な
        # モデルを同じ GPU に載せない (割り当ての違いが bf16 の結果に
        # 影響する余地を消す)。
        args.utmos = (args.mode == "multi")
    if args.verify_inplace_equiv is None:
        args.verify_inplace_equiv = (args.mode == "multi")
    if args.span_extension is None:
        # 英語は False に解決する = 既存の呼び出しと同じ経路。
        args.span_extension = (args.lang != "en")
    if args.pairs_module is None:
        # en は従来の既定値そのもの。zh は EN / JA と同じ格子で作り直した v2。
        args.pairs_module = {"en": "en_multientity_300",
                             "zh": "zh_multientity_300_v2"}[args.lang]
    if args.mode == "equiv":
        conds = CONDITIONS_EQUIV
        row_mode = ROW_MODE_EQUIV
        if args.out_dir is None:
            args.out_dir = os.path.join(
                PROJECT_ROOT, "eval_results",
                f"multientity_llasa_equivcheck_{args.lang}_L{args.layer}")
        if args.equiv_ref_dir is None:
            args.equiv_ref_dir = os.path.join(
                PROJECT_ROOT, "eval_results", "_cc_wavs_a", f"llasa_{args.lang}")
    else:
        conds = CONDITIONS_MULTI
        # **2026-09-22 修正。**--tgt_only を足したとき (2026-09-20) に
        # multi 本番の row_mode と out_dir の既定が落ちていて、
        # `--mode multi` を --tgt_only 無しで走らせると row_mode が NameError、
        # --out_dir 未指定なら os.path.isabs(None) が TypeError になっていた。
        # 既存の英語 300 対はこの patch より前に作られている (results.json の
        # row_mode は splice_multi_llasa)。値もその記録に合わせる。
        row_mode = ROW_MODE_MULTI
        if args.out_dir is None:
            args.out_dir = os.path.join(
                PROJECT_ROOT, "eval_results",
                f"multientity_llasa_{args.lang}_L{args.layer}")
    if args.tgt_only:
        conds = list(TGT_SLOT_CONDS)
        row_mode = ROW_MODE_MULTI
        if args.out_dir is None:
            args.out_dir = os.path.join(
                PROJECT_ROOT, "eval_results",
                f"multientity_llasa_{args.lang}_L{args.layer}")
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    items = load_items(args)
    print(f"[data] mode={args.mode} lang={args.lang} items={len(items)} "
          f"layer={args.layer} (hidden index {L.hidden_index_for_layer(args.layer)})",
          flush=True)
    print(f"[data] date_string={args.date_string!r} "
          f"(chat template が実行日を差し込むのでここで固定する)", flush=True)
    print(f"[data] pairs_module={args.pairs_module} "
          f"span_extension={args.span_extension}", flush=True)
    print(f"[data] out_dir={args.out_dir}", flush=True)
    if args.mode == "equiv":
        print(f"[data] md5 比較先: {args.equiv_ref_dir}", flush=True)

    ns = build_ns_for_existing_helpers(args)
    bundle = R.load_bundle(ns)
    if not (0 <= args.layer < bundle.num_layers):
        raise SystemExit(f"--layer {args.layer} が範囲外 (0..{bundle.num_layers - 1})")
    if not L.chat_template_date(bundle.tokenizer):
        print("[warn] chat template に日付が入らない。date_string の固定は無意味", flush=True)
    ref = R.build_ref(bundle, ns)
    ref_ids = ref["speech_ids"] if ref else None
    ref_text = ref["ref_text"] if ref else None

    gen = L.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      temperature=args.temperature, top_p=args.top_p,
                      top_k=args.top_k, max_new_tokens=args.max_new_tokens)
    print(f"[gen] {gen.as_dict()}", flush=True)

    utmos, squim = make_scorers(bundle.device, args.utmos)

    # --tgt_only は別の帳面（既存 4 条件の再開記録を壊さない）。
    ckpt = os.path.join(args.out_dir,
                        "checkpoint_tgt.json" if args.tgt_only
                        else "checkpoint.json")
    rows, skipped = [], []
    if os.path.exists(ckpt) and not args.overwrite:
        st = json.load(open(ckpt))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != row_mode]
        assert not bad, (f"checkpoint mode mismatch in {ckpt}: pairs {bad[:5]} "
                         f"は別のモードで作られている。--out_dir を分けること")
        print(f"[resume] rows={len(rows)} skipped={len(skipped)}", flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}
    if args.tgt_only:
        rows, skipped = [], []
        done = {it["pair_idx"] for it in items
                if all(os.path.exists(os.path.join(
                    args.out_dir, "s%04d_%s.wav" % (it["pair_idx"], c)))
                    for c in TGT_SLOT_CONDS)}
        print("[Info] tgt_only: 目標文が既にある対 %d / 作る対 %d"
              % (len(done), len(items) - len(done)), flush=True)

    li = L.hidden_index_for_layer(args.layer)
    t0 = time.time()

    for nth, it in enumerate(items):
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        k = len(it["spans"])
        try:
            in_a = L.build_input_text(it["text_a"], ref_text)
            # --- 枠ごとの拡張量を先に決める (2026-09-22)。**英語は
            # args.span_extension が False なのでこの節を 1 行も通らない。**
            # src 側と ref 側の区間を同じ拡張で取らないと、後段の
            # 「区間の解決が 2 経路で食い違っている」で全対が落ちる。
            # **--tgt_only でも計画する。**resolve_item_spans_llasa は
            # --tgt_only では呼ばれないが、prepare_context_multi の
            # verify_entity_span(strict=True) は走るので、融合する枠は
            # 拡張しないとここで落ちて目標音声が作られない (= 天井が n<300)。
            expands = [None] * k
            if args.span_extension:
                _plan = _plan_span_extension_llasa(
                    bundle.tokenizer, it, ref_text, args.date_string, ref_ids,
                    in_a, render_dated(bundle.tokenizer, in_a, ref_ids,
                                       args.date_string))
                expands = [None if p["ext"] is None
                           else (p["ext"]["pre"], p["ext"]["post"]) for p in _plan]
            # --- src (text_a): k 区間をまとめて解決。kappa 用に hidden も取る ---
            src = prepare_context_multi(
                bundle, in_a, ref_ids,
                words=[sp["word_a"] for sp in it["spans"]],
                region_text=it["text_a"], need_hidden=not args.no_kappa,
                date_string=args.date_string, expands=expands)
            # --- ref: 区間ごとに別の prompt / 別の prefill ---
            refs_ctx = []
            for i, sp in enumerate(it["spans"]):
                in_b = L.build_input_text(it["text_refs"][i], ref_text)
                refs_ctx.append(prepare_context_multi(
                    bundle, in_b, ref_ids, words=[sp["word_b"]],
                    region_text=it["text_refs"][i], need_hidden=True,
                    date_string=args.date_string, expands=[expands[i]]))
            # **--tgt_only では区間の検査をしない（2026-09-20 修正）。**
            # 目標文は注入せずそのまま読ませるだけなので実体語の区間は要らない。
            # 検査を走らせると、助詞や接尾辞が癒着する対で落ちて目標音声が
            # 作られない（T5Gemma 日本語で 15 対、中国語で 30 対が落ちた）。
            if args.tgt_only:
                detail = {"skipped_for_tgt_only": True, "gap_tokens": None}
                s_ranges = []
            else:
                detail = resolve_item_spans_llasa(
                    bundle.tokenizer, it, ref_text, args.date_string,
                    ref_speech_ids=ref_ids, span_extension=args.span_extension)
                s_ranges = [tuple(x) for x in detail["src"]]
                assert s_ranges == [(sp["tok_start"], sp["tok_end"])
                                    for sp in src["spans"]], \
                    "区間の解決が 2 経路で食い違っている"
        except Exception as e:
            print(f"  [{pidx}] SKIP {type(e).__name__}: {str(e)[:200]}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"{type(e).__name__}: {e}"[:400]})
            done.add(pidx)
            if args.stop_on_error:
                raise
            continue

        tgt_hiddens = [
            refs_ctx[i]["hidden_states"][li][
                0, refs_ctx[i]["spans"][0]["tok_start"]:
                   refs_ctx[i]["spans"][0]["tok_end"]].clone()
            for i in range(k)
        ]

        row = {
            "pair_idx": pidx, "mode": row_mode,
            "template_id": it.get("template_id"), "lang": args.lang,
            "layer": args.layer, "hidden_index": li,
            "text_a": it["text_a"], "text_refs": it["text_refs"],
            "text_both": it.get("text_both"),
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "repeat_surface": it.get("repeat_surface", False),
            "words_a": [sp["word_a"] for sp in it["spans"]],
            "words_b": [sp["word_b"] for sp in it["spans"]],
            "span_tok": detail,
            "T_src": int(src["prompt_len"]),
            "date_string": args.date_string,
            "conds": {},
        }
        ok = True
        for cond in conds:
            if cond in TGT_SLOT_CONDS:
                # 枠ごとの目標文。注入なしで、その枠を差し替えた文を読ませる。
                ids, slots = refs_ctx[TGT_SLOT_CONDS[cond]]["input_ids"], []
            elif cond == "bl":
                ids, slots = src["input_ids"], []
            elif cond == "tgt":
                ids, slots = refs_ctx[0]["input_ids"], None
            elif cond == "inj":
                ids, slots = src["input_ids"], [0]
            else:
                ids, slots = src["input_ids"], COND_SLOTS[cond]
            try:
                if slots:
                    g, sp = gen_spliced(
                        bundle, ids, gen, args.layer,
                        [s_ranges[i] for i in slots],
                        [tgt_hiddens[i] for i in slots],
                        prompt_len=src["prompt_len"],
                        verify_inplace_equiv=(args.verify_inplace_equiv
                                              and len(slots) == 1))
                else:
                    g, sp = gen_plain(bundle, ids, gen)
                audio = L.decode_speech_ids(bundle, g["speech_ids"])
            except Exception as e:
                print(f"  [{pidx}] {cond} synth fail: {type(e).__name__}: "
                      f"{str(e)[:200]}", flush=True)
                skipped.append({"pair_idx": pidx,
                                "reason": f"{cond} synth: {type(e).__name__}: {e}"[:400]})
                ok = False
                if args.stop_on_error:
                    raise
                break

            wav_path = os.path.join(args.out_dir, f"s{pidx:04d}_{cond}.wav")
            _, dur, clip = R.save_wav(wav_path, audio, bundle.sampling_rate)
            u, pe = score_wav_local(wav_path, utmos, squim, bundle.device)
            ent = {
                "wav": os.path.basename(wav_path),
                "delta_tok": int(sp.delta_tok) if sp is not None else 0,
                "T_out": (int(refs_ctx[TGT_SLOT_CONDS[cond]]["prompt_len"])
                          if cond in TGT_SLOT_CONDS
                          else (int(src["prompt_len"]) if cond != "tgt"
                                else int(refs_ctx[0]["prompt_len"]))),
                "out_ranges": [list(x) for x in (sp.out_ranges or [])]
                if sp is not None else [],
                # Llasa は delta_tok が常に 0 なので長さ一致ダミーが不要。
                # 4 条件すべて本文は text_a (tgt 条件だけ text_refs[0])。
                "target_text": (it["text_refs"][TGT_SLOT_CONDS[cond]]
                                if cond in TGT_SLOT_CONDS
                                else (it["text_refs"][0] if cond == "tgt"
                                      else it["text_a"])),
                "dur_sec": round(dur, 4),
                "md5": md5_of(wav_path),
                "n_speech_tokens": int(len(g["speech_ids"])),
                "n_generated": int(g["n_generated"]),
                "hit_max_new_tokens": bool(g["hit_max_new_tokens"]),
                "clip_ratio": round(clip, 6),
                "utmos": u, "pesq": pe,
            }
            if sp is not None:
                ent["splice"] = sp.stats()
                sp.assert_fired()
            row["conds"][cond] = ent

        done.add(pidx)
        if not ok:
            continue

        # kappa は **生成が全部終わってから** 測る。既存の単一 entity 経路も
        # splice_synthesize の中で INJ 生成の後に測っている
        # (llasa_splice.py:969 generate -> 985-988 kappa)。生成の前に GPU で
        # 余計な演算を挟まないことで、bf16 の結果に効く余地を消しておく。
        if not args.no_kappa and src.get("hidden_states") is not None:
            row["kappa_by_layer"] = [
                [round(v, 6) for v in L.kappa_by_layer(
                    src["hidden_states"], refs_ctx[i]["hidden_states"],
                    src["spans"][i], refs_ctx[i]["spans"][0])]
                for i in range(k)
            ]
            row["kappa_at_splice"] = [round(row["kappa_by_layer"][i][li], 6)
                                      for i in range(k)]

        # 系列長を変えないので位置のずれは常に 0。記録は T5Gemma 版と同じ鍵名。
        row["total_position_drift"] = row["conds"][conds[-1]]["delta_tok"]
        # --tgt_only では bl を作っていないので、bl との一致比較はしない。
        if not args.tgt_only:
            bl_md5 = row["conds"]["bl"]["md5"]
            for c in conds:
                row["conds"][c]["bit_identical_to_bl"] = bool(
                    row["conds"][c]["md5"] == bl_md5)
        if args.mode == "equiv":
            for c in conds:
                p = os.path.join(args.equiv_ref_dir, f"s{pidx:04d}_{c}.wav")
                row["conds"][c]["ref_wav"] = p
                row["conds"][c]["ref_md5"] = md5_of(p) if os.path.exists(p) else None
                row["conds"][c]["md5_match"] = bool(
                    row["conds"][c]["ref_md5"] == row["conds"][c]["md5"])
        rows.append(row)

        if len(rows) % args.log_every == 0 or nth + 1 == len(items):
            c = row["conds"]
            extra = ""
            if args.mode == "equiv":
                extra = " md5=" + ",".join(
                    f"{k2}:{'OK' if c[k2]['md5_match'] else 'NG'}" for k2 in conds)
            elif args.tgt_only:
                extra = ""
            else:
                extra = " ident_to_bl=" + ",".join(
                    f"{k2}:{int(c[k2]['bit_identical_to_bl'])}" for k2 in conds)
            print(f"  [{len(rows)}/{len(items)}] idx={pidx} "
                  f"gap_tok={detail['gap_tokens']} drift={row['total_position_drift']} "
                  f"dur=" + "/".join(f"{c[k2]['dur_sec']:.2f}" for k2 in conds)
                  + extra, flush=True)
        if len(rows) % args.ckpt_every == 0:
            R.atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt)

    R.atomic_json_dump({"rows": rows, "skipped": skipped}, ckpt)

    def _mean(cond, key):
        v = [r["conds"][cond][key] for r in rows
             if r["conds"].get(cond, {}).get(key) is not None]
        return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] if v \
            else [None, None]

    summary = {
        "mode": args.mode, "row_mode": row_mode,
        "pairs_module": args.pairs_module if args.mode == "multi" else "en_cross_culture_names",
        "lang": args.lang, "layer": args.layer, "hidden_index": li,
        "date_string": args.date_string,
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "conditions": conds,
        "model_id": args.model_id, "model_dir": bundle.model_dir,
        "codec_id": args.codec_id, "codec_dir": bundle.codec_dir,
        "num_layers": bundle.num_layers, "hidden_size": bundle.hidden_size,
        "decode": gen.as_dict(), "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "verify_inplace_equiv": bool(args.verify_inplace_equiv),
        "sampling_rate": bundle.sampling_rate,
        "ref": ({"wav": ref["wav_path"], "text": ref["ref_text"],
                 "n_tokens": ref["n_tokens"], "dur_sec": ref["dur_sec"],
                 "codec_roundtrip": ref.get("roundtrip")} if ref else None),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "utmos": {c: _mean(c, "utmos") for c in conds} if args.utmos else None,
        "mean_dur_sec": {c: round(float(np.mean(
            [r["conds"][c]["dur_sec"] for r in rows])), 4) for c in conds}
        if rows else {},
        "mean_abs_position_drift": round(float(np.mean(
            [abs(r["total_position_drift"]) for r in rows])), 3) if rows else None,
        # 「注入後の音声が基線とビット単位で同一だった対の数」。単一 entity の
        # 英語 300 対では 156/300 だった。
        # --tgt_only では bl を作っていないので、この集計は取らない。
        "n_bit_identical_to_bl": ({} if args.tgt_only else ({
            c: int(sum(1 for r in rows if r["conds"][c]["bit_identical_to_bl"]))
            for c in conds} if rows else {})),
        "n_equal_len_spans": {
            "token_wise": int(sum(
                1 for r in rows for c in conds
                if r["conds"][c].get("splice")
                for e in r["conds"][c]["splice"]["equal_len"] if e)),
            "mean_pool_broadcast": int(sum(
                1 for r in rows for c in conds
                if r["conds"][c].get("splice")
                for e in r["conds"][c]["splice"]["equal_len"] if not e)),
        },
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    # **拡張を使ったときだけ鍵を足す (2026-09-22)。**off のときは足さないので、
    # 英語の results.json は鍵構成まで従来と同一のまま。鍵名は T5Gemma の ZH v2
    # (eval_results/multientity_zh_v2_300/results.json) に合わせてある。
    if args.span_extension and args.tgt_only:
        # --tgt_only は区間を記録しない回なので、使ったことだけ残す。
        summary["span_extension"] = {
            "rule": "minimal_symmetric_extension",
            "impl": "scripts/projection/span_extend.py:minimal_symmetric_surface",
            "per_pair_record": None,
        }
    elif args.span_extension:
        summary["span_extension"] = {
            "rule": "minimal_symmetric_extension",
            "impl": "scripts/projection/span_extend.py:minimal_symmetric_surface",
            "n_rows_with_extension": int(sum(
                1 for r in rows if r["span_tok"].get("n_span_extended"))),
            "pairs": [r["pair_idx"] for r in rows
                      if r["span_tok"].get("n_span_extended")],
            "extras": sorted({e["extra"] for r in rows
                              for e in (r["span_tok"].get("span_extension") or [])
                              if e}),
        }
    if args.mode == "equiv":
        summary["equiv_ref_dir"] = args.equiv_ref_dir
        summary["md5_match"] = {
            c: {"k": int(sum(1 for r in rows if r["conds"][c]["md5_match"])),
                "n": len(rows)} for c in conds}
        summary["inplace_equiv_k1"] = {
            "n_checked": int(sum(
                1 for r in rows for c in conds
                if (r["conds"][c].get("splice") or {}).get("inplace_equiv_k1") is not None)),
            "n_true": int(sum(
                1 for r in rows for c in conds
                if (r["conds"][c].get("splice") or {}).get("inplace_equiv_k1") is True)),
        }
    if rows:
        summary["gen_length"] = {
            c: {"mean_tokens": round(float(np.mean(
                    [r["conds"][c]["n_speech_tokens"] for r in rows])), 1),
                "n_hit_max": int(sum(1 for r in rows
                                     if r["conds"][c]["hit_max_new_tokens"]))}
            for c in conds}

    R.atomic_json_dump({"summary": summary, "rows": rows, "skipped": skipped},
                       os.path.join(args.out_dir,
                                    "results_tgt.json" if args.tgt_only
                                    else "results.json"))
    print(f"\n=== multientity llasa ({args.mode}) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.mode == "equiv":
        allok = all(v["k"] == v["n"] for v in summary["md5_match"].values())
        print("\n=== md5 等価性 ===")
        for c in conds:
            v = summary["md5_match"][c]
            print(f"  {c:4s} {v['k']}/{v['n']} 一致")
        print("  結論: " + ("合格 (既存の単一 entity 経路とバイト一致)"
                            if allok else "不合格。移植がどこかで違う"))
        if not allok:
            for r in rows:
                for c in conds:
                    if not r["conds"][c]["md5_match"]:
                        print(f"    NG idx={r['pair_idx']} {c} "
                              f"got={r['conds'][c]['md5'][:12]} "
                              f"exp={str(r['conds'][c]['ref_md5'])[:12]}")
    if skipped:
        print(f"[warn] {len(skipped)} 件 skip。results.json['skipped'] を見ること")


if __name__ == "__main__":
    main()
