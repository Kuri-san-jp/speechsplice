#!/usr/bin/env python3
"""suno/bark で 1 文の中の 2 箇所の実体語を同時に差し替える (BL / INJ_1 / INJ_2 / INJ_12).

T5Gemma 版 `scripts/analysis/generate_multientity.py` の構成をそのまま写した
**新規**スクリプト。既存ファイルは 1 行も変更していない。
差し替えの本体は `scripts/projection/splice_multi.py:build_spliced_hidden_multi`
を import して使う (再実装しない)。置換行の作り方 (等長なら行対行、不等長なら
平均を broadcast) は `bark_splice.PrefillSplice._replacement` を**そのまま呼ぶ**。

4 条件 (1 文あたり同一 seed で 4 回生成):

  bl     差し替えなし。TTS(text_a)
  inj1   枠 1 だけ差し替え。donor = text_refs[0] の prefill
  inj2   枠 2 だけ差し替え。donor = text_refs[1] の prefill
  inj12  両方差し替え。枠 i の donor は ref_i (**枠ごとに別の forward**)

inj12 で「両方置換した 1 本の文 (text_both)」から 2 枠を取らないのが要点。
そうすると干渉の原因 (splice の相互作用 vs. ref 側の文脈が既に汚れている) が
分離できない。text_both は使わない (Bark は系列長を変えないので、T5Gemma 版の
「長さ一致ダミー」も要らない)。

Bark 固有の事情
---------------
1. **prompt 長は 257 固定**。文にも preset にも依らない
   (= max_input_semantic_length 256 + infer token 1)。したがって
   * `delta_tok` は必ず 0、`T_out` は必ず 257
   * `out_ranges == s_ranges` になる (2 番目の枠の index は 1 番目の差し替えで
     ずれない)
   長さが変わる差し替えは Bark では**そもそも不可能**なので、不等長の枠は
   既存の単一 entity と同じ mean_pool_broadcast で m 行に潰す。
   これは新しい方策ではなく `PrefillSplice._replacement` の既定の挙動である。
2. 差し替え地点は `model.semantic.layers[22]` の出力 (prefill のみ)。
   **層 22 では差し替えが下流に届かない**。単一 entity では 300 対中 259 対が
   BL とビット単位で同一だった (層 22 の出力分布 KL = 1e-6)。複数 entity でも
   同じになる見込みなので、
   * 対ごとに BL と INJ の**意味トークン列の指紋**を突き合わせて
     `differs_from_bl` を条件別に記録する
   * 最初の `--null_check_pairs` 対で 3 つの INJ すべてが BL と一致したら
     **止まる**。続けるなら `--allow_null_layer` を付ける
     (本番は付ける前提。付けないと 12 対で止まる)
   * summary に `n_identical_to_bl` を条件別に必ず出す
   ようにしてある。「効果が無い」と「そもそも splice が音に出ていない」を
   取り違えないため。
3. 話者 preset は言語ごとに固定 (en = v2/en_speaker_6、ja = v2/ja_speaker_0)。
   外すと semantic_history が全部 pad になり、条件間で声が変わる。
4. 中国語のデータは存在しないので `--lang zh` は受け付けない。

単一区間の等価性 (移植が正しいことの合格条件)
--------------------------------------------
`data/<lang>_cross_culture_names.py` の 300 対 (1 枠) を入力にすると
本スクリプトは k=1 で走る。そのとき生成される
  s{i:04d}_bl.wav   は `bark_splice.synthesize(bundle, text_a)` そのもの
  s{i:04d}_inj1.wav は `bark_splice.splice_synthesize(...)` と**同じ hidden**
で、既存の `eval_results/_cc_wavs_a/bark_<lang>/s{i:04d}_{bl,inj}.wav` と
md5 が一致しなければならない。`--equiv_ref_dir` を渡すと対ごとに md5 を
突き合わせ、summary に判定を出す。

  * bl が不一致なら、原因は移植ではなく実行環境 (GPU 種別 / dtype / preset /
    復号設定) の違い。先にそこを直すこと。
  * bl が一致して inj1 だけ不一致なら、移植の側が違っている。

  **どの対を使うかが効く。** 層 22 では INJ が BL とビット単位で同一になる対が
  多いので (EN 259/300、JA 72/80)、その対の md5 一致は BL 経路しか試していない。
  差し替えが音に出た対を必ず含めること。既存結果での該当対は
    EN (eval_results/bark_cross_culture_en/results.json): 2, 5, 17, 19, 24, ...
       -> --n 5 なら対 2 が、--n 20 なら 2 / 5 (mean_pool) と 17 / 19 (token_wise)
          の両方式が入る
    JA (eval_results/bark_cross_culture_ja/progress.json): 5, 10, 20, 22, 23, ...
       -> --n 12 で対 5 / 10 が入る
  summary["equivalence"]["per_cond"]["n_nontrivial"] にその対数を出す。

実行例 (GPU。投入は依頼者が行う)
  # 単一区間の md5 等価性 (n=5)
  SLURM_TIME=00:40:00 slurm_run.sh \\
    python3 -u /mnt/scripts/bark/run_multientity_bark.py \\
      --pairs_module en_cross_culture_names --lang en --n 5 \\
      --out_dir /mnt/eval_results/bark_multientity_equiv_en_n5 \\
      --equiv_ref_dir /mnt/eval_results/_cc_wavs_a/bark_en \\
      --allow_null_layer

  # 本番 (EN 300 対)
  SLURM_TIME=10:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/bark/run_multientity_bark.py \\
      --pairs_module en_multientity_300 --lang en \\
      --out_dir /mnt/eval_results/bark_multientity_en_300 --allow_null_layer

  # 天井用の目標文だけ (--tgt_only、2026-09-22 追加)
  SLURM_TIME=10:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/bark/run_multientity_bark.py \\
      --pairs_module en_multientity_300 --lang en --tgt_only \\
      --out_dir /mnt/eval_results/bark_multientity_en_300 --allow_null_layer

  差し替えをしないので区間の解決も往復確認も層の検査も要らない。既にある wav は
  作り直さないので、先に本番 (4 条件) を回してから足しても既存の音声と
  results.json / checkpoint.json は動かない。帳面は checkpoint_tgt.json /
  results_tgt.json (別ファイル)。--dry_run を足すとモデルを読まずに、
  出すつもりの文とファイル名だけ並べる。

採点は既存の `scripts/analysis/score_multientity.py` が行う (新しい採点器は作らない)。
天井 (tgt1 / tgt2) の採点は `scripts/analysis/score_multientity_ceiling.py`。
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
# **setdefault は効かない** (slurm_run.sh がコンテナに HF_HOME=/tmp/... を固定で
# 渡すため)。代入で上書きする。transformers を import する前に済ませること。
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["TORCH_HOME"] = os.path.join(ROOT, ".cache", "torch")

sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "bark"))
# 目標文だけを出す経路 (--tgt_only) の共通部品。stdlib しか読まない。
# **末尾に足す**ので、既存の import の解決順は 1 つも動かない。
sys.path.append(os.path.join(ROOT, "scripts", "multientity"))

import bark_splice as C  # noqa: E402  (HF_HOME / NUMBA_CACHE_DIR を同じ値で代入し直す)

# **再実装しない。** k 区間の組み立てと検証はこの 4 つに任せる。
from splice_multi import (  # noqa: E402
    build_spliced_hidden_multi, check_span_sets, resolve_spans_with_cursor,
    span_deltas, verify_span_roundtrip,
)
# --- 天井用の目標文だけを出す経路 (--tgt_only、2026-09-22 追加) -------------
from tgt_only import dry_run_report, run_targets  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CONDITIONS_BY_K = {1: ("bl", "inj1"), 2: ("bl", "inj1", "inj2", "inj12")}
# 条件名 -> 使う枠 index (0-based)。T5Gemma 版 generate_multientity.py:92 と同じ。
COND_SLOTS = {"bl": (), "inj1": (0,), "inj2": (1,), "inj12": (0, 1)}
INJ_CONDS = ("inj1", "inj2", "inj12")
# 単一区間の md5 突き合わせ: 本スクリプトの条件名 -> 既存 run_bark_eval.py の条件名
EQUIV_COND_MAP = {"bl": "bl", "inj1": "inj"}
DEFAULT_PRESET = {"en": "v2/en_speaker_6", "ja": "v2/ja_speaker_0"}
MODE_TAG = "splice_multi_bark"


# --------------------------------------------------------------------------
# k 区間の prefill splice hook
# --------------------------------------------------------------------------
class MultiPrefillSplice:
    """`bark_splice.PrefillSplice` の k>=1 版.

    分担をはっきり分ける:
      * **置換行の中身**  -> `C.PrefillSplice._replacement`（枠ごとに 1 個持つ）。
        等長なら行対行、不等長なら目標スパンの平均を m 行に broadcast。
        単一 entity 経路と**同じメソッドを同じ引数で**呼ぶので、方策が
        枝分かれしない。
      * **系列の組み立て** -> `splice_multi.build_spliced_hidden_multi`。
        一括 concat なので、1 番目の枠を差し替えたせいで 2 番目の枠の index が
        ずれるという事故が構造的に起きない。**自分で index をずらさない。**
      * **区間集合の検証** -> `splice_multi.check_span_sets`（重なりを落とす。
        隣接は長さ 0 の gap として許す）。build_spliced_hidden_multi の中でも
        もう一度走る。

    prefill 判定は `C.PrefillSplice.__call__` (bark_splice.py:883-926) と
    1 行ずつ同じにしてある:
      1. module is expect_module でなければ絶対に書き換えない (coarse 段の
         1 窓目も長さ 257 になり得るので、長さの検査だけでは区別できない)
      2. h.shape[1] <= 1 は decode ステップ
      3. h.shape[1] != prompt_len (257) は見送って記録
      4. 既に 1 回発火していたら二度と発火しない

    Bark は prompt 長が 257 固定なので、Δtok != 0 は**必ず実装の誤り**である。
    そのまま音にすると KV cache と長さが合わないので例外で落とす
    (数値を合わせるための後付け補正はしない)。
    """

    def __init__(self, layer_module, s_spans, tgt_hiddens, prompt_len=C.PROMPT_LEN,
                 expect_module=None, verify_hidden=None, verify_atol=1e-3):
        if len(s_spans) != len(tgt_hiddens):
            raise ValueError(f"枠数が合わない: {len(s_spans)} vs {len(tgt_hiddens)}")
        if not s_spans:
            raise ValueError("枠が 0 個")
        self.layer_module = layer_module
        self.expect_module = expect_module if expect_module is not None else layer_module
        # 枠ごとの単一区間 splice。**hook は登録しない** (__enter__ を呼ばない)。
        # 置換行の作り方 (_replacement) と m / n / equal_len の帳簿だけ借りる。
        self.units = [
            C.PrefillSplice(layer_module, sp, th, prompt_len=prompt_len,
                            expect_module=self.expect_module)
            for sp, th in zip(s_spans, tgt_hiddens)]
        self.s_ranges = [(u.s0, u.s1) for u in self.units]
        self.prompt_len = int(prompt_len) if prompt_len is not None else None
        self.verify_hidden = None if verify_hidden is None else verify_hidden.detach()
        self.verify_atol = float(verify_atol)
        self.verify_max_abs_diff = None
        self.handle = None
        self.fired = 0
        self.skipped_shapes = []
        self.decode_calls = 0
        self.foreign_calls = 0
        self.delta_tok = None
        self.out_ranges = None
        self.T_out = None
        # 重なりは hook の中ではなく**ここで**落とす。generate の途中で例外を
        # 投げると片付けが面倒なので、生成に入る前に確定させる。
        check_span_sets(self.prompt_len, [u.m for u in self.units],
                        self.s_ranges, [(0, u.m) for u in self.units])

    # ------------------------------------------------------------------
    def _donors(self, dtype, device):
        """枠ごとの置換行 (m_i, hidden) を `_replacement` から作る.

        返り値は build_spliced_hidden_multi が期待する形
        (ref_encs = [{"last_hidden_state": [1, m_i, d]}], r_ranges = [(0, m_i)])。
        r_range の長さが s_range の長さと等しいので Δtok は枠ごとに 0 になる。
        """
        ref_encs, r_ranges = [], []
        for u in self.units:
            # bark_splice.py:878-881 と同じ呼び出し (等長 / 不等長の分岐もそこ)。
            rep = u._replacement(dtype).to(device)
            if rep.dim() != 2 or int(rep.shape[0]) != u.m:
                raise RuntimeError(
                    f"_replacement の形が {tuple(rep.shape)} (想定 ({u.m}, d))")
            ref_encs.append({"last_hidden_state": rep.unsqueeze(0)})
            r_ranges.append((0, int(rep.shape[0])))
        return ref_encs, r_ranges

    def __call__(self, module, args, output):
        if module is not self.expect_module:
            # 別の段 (coarse / fine) に掛かっている。絶対に書き換えない。
            self.foreign_calls += 1
            return output
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.dim() != 3:
            raise RuntimeError(
                f"BarkBlock の出力が 3 次元の Tensor でない: "
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
        if max(s1 for _, s1 in self.s_ranges) > T:
            raise RuntimeError(f"実体語スパン {self.s_ranges} が prefill 長 {T} を超えている")
        if h.shape[0] != 1:
            raise RuntimeError(f"batch size {h.shape[0]} は未対応 (1 のみ)")
        if self.verify_hidden is not None:
            if tuple(self.verify_hidden.shape) != tuple(h.shape):
                raise RuntimeError(
                    f"照合用 hidden の形が合わない: {tuple(self.verify_hidden.shape)} "
                    f"vs {tuple(h.shape)}")
            d = float((self.verify_hidden.float().to(h.device) - h.float()).abs().max())
            self.verify_max_abs_diff = d
            if not (d <= self.verify_atol):
                raise RuntimeError(
                    f"生成時の prefill hidden と、別経路で計算した prefill hidden が "
                    f"一致しない (max|Δ| = {d:.3e} > {self.verify_atol:.1e})。"
                    f"donor も同じ経路で計算しているので、このまま INJ を作ると "
                    f"別物の hidden を入れた音になる。中断する。")

        ref_encs, r_ranges = self._donors(h.dtype, h.device)
        modified, delta_tok, out_ranges = build_spliced_hidden_multi(
            {"last_hidden_state": h}, ref_encs, self.s_ranges, r_ranges)
        d_each, d_sum = span_deltas(self.s_ranges, r_ranges)
        if any(d != 0 for d in d_each) or int(d_sum) != 0 \
                or int(modified.shape[1]) != int(T):
            raise RuntimeError(
                f"Bark の prompt 長は {self.prompt_len} 固定なので Δtok は 0 で "
                f"なければならない: 枠別 Δ={d_each} 総和={d_sum} "
                f"T_in={T} T_out={modified.shape[1]}")
        got = [tuple(int(v) for v in x) for x in out_ranges]
        want = [tuple(int(v) for v in x) for x in self.s_ranges]
        if got != want:
            # Δtok=0 なら置換後の位置は元の位置と一致する。ここが食い違うのは
            # 「自分で index をずらした」ときだけ。
            raise RuntimeError(f"out_ranges {got} != s_ranges {want}")
        self.delta_tok = int(delta_tok)
        self.out_ranges = [list(x) for x in got]
        self.T_out = int(modified.shape[1])
        self.fired += 1
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
                f"decode_calls={self.decode_calls} foreign_calls={self.foreign_calls}")

    def stats(self):
        return {
            "fired": self.fired, "decode_calls": self.decode_calls,
            "skipped_shapes": self.skipped_shapes[:16],
            "foreign_calls": self.foreign_calls,
            "k_spans": len(self.units),
            "m_src_entity_tokens": [u.m for u in self.units],
            "n_tgt_entity_tokens": [u.n for u in self.units],
            "equal_len": [u.equal_len for u in self.units],
            "splice_mode": ["token_wise" if u.equal_len else "mean_pool_broadcast"
                            for u in self.units],
            "prefill_hidden_max_abs_diff": (None if self.verify_max_abs_diff is None
                                            else round(self.verify_max_abs_diff, 8)),
        }


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_items(pairs_module, lang, n=0):
    """2 枠形式 (multientity) と 1 枠形式 (cross_culture) の両方を受ける.

    1 枠形式は `build_300_pairs()` が返す {text_a, text_b, word_a, word_b} を
    2 枠形式と同じレコードに畳んで返す (text_refs = [text_b])。
    こうすると単一区間でも**同じ経路**が走るので、既存の単一 entity 結果との
    md5 突き合わせがそのまま等価性の試験になる。
    """
    mod = __import__(pairs_module)
    if hasattr(mod, "build_pairs"):
        items = mod.build_pairs()
    elif hasattr(mod, "build_300_pairs"):
        items = mod.build_300_pairs()
    else:
        raise SystemExit(f"data/{pairs_module}.py に build_pairs / build_300_pairs が無い")

    out = []
    for i, it in enumerate(items):
        it = dict(it)
        it.setdefault("pair_idx", i)
        if "spans" not in it:
            # 1 枠形式 -> 2 枠形式のレコードへ畳む
            it["spans"] = [{"slot": 1, "word_a": it["word_a"], "word_b": it["word_b"],
                            "char_start_a": None, "char_start_b": None}]
            it["text_refs"] = [it["text_b"]]
            it["text_both"] = it["text_b"]
            it.setdefault("template_id", None)
            it.setdefault("gap_words", None)
            it.setdefault("slot1_at_start", None)
        k = len(it["spans"])
        if k not in CONDITIONS_BY_K:
            raise SystemExit(f"pair {it['pair_idx']}: 枠が {k} 個 (1 か 2 のみ)")
        if len(it["text_refs"]) != k:
            raise SystemExit(f"pair {it['pair_idx']}: text_refs が {len(it['text_refs'])} 本")
        if it.get("lang") not in (None, lang):
            raise SystemExit(f"pair {it['pair_idx']}: lang={it['lang']} != --lang {lang}")
        out.append(it)
    ks = {len(it["spans"]) for it in out}
    if len(ks) != 1:
        raise SystemExit(f"枠数が文によって違う: {sorted(ks)}")
    return (out[:n] if n else out), ks.pop()


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# span 解決
# --------------------------------------------------------------------------
def resolve_item_spans(bundle, item, in_a, ref_inputs):
    """src 側 k 区間と ref 側 1 区間 x k を解決する.

    **区間の解決は既存の単一 entity 経路と同じ関数を同じ引数で呼ぶ**
    (`bark_splice.entity_token_span` / `verify_entity_span`。
     bark_splice.py:1196-1199 と対応)。Bark の prompt は
     「token index = hidden state の行番号」なので補正は一切足さない。

    そのうえで `splice_multi.resolve_spans_with_cursor` +
    `verify_span_roundtrip` で**独立に**もう一度解いて突き合わせる。
    食い違ったらその文は丸ごと skip する (「末尾 N トークンを実体語とみなす」
    種類の fallback は用意しない)。

    返り値: (s_spans, t_spans, detail) / 失敗時 (None, None, reason 文字列)
    """
    spans = item["spans"]
    k = len(spans)
    surfaces_a = [sp["word_a"] for sp in spans]

    # --- 区間の解決と往復確認。
    # **呼ぶ順序も単一 entity 経路に合わせる** (bark_splice.py:1196-1199 は
    # s_span -> t_span -> verify(s) -> verify(t) の順)。k=1 に縮退させると
    # 呼び出し列が splice_synthesize と 1 個ずつ一致する
    # (scripts/bark/_me_verify_cpu.py:test_a3_trace_equivalence で検査)。
    try:
        s_spans = [C.entity_token_span(bundle, item["text_a"], sp["word_a"],
                                       input_ids=in_a["input_ids"])
                   for sp in spans]
    except Exception as e:
        return None, None, f"src span: {type(e).__name__}: {e}"[:300]
    try:
        # ref は枠ごとに**別の文** (text_refs[i])。両方置換した 1 本は使わない。
        t_spans = [C.entity_token_span(bundle, item["text_refs"][i], sp["word_b"],
                                       input_ids=ref_inputs[i]["input_ids"])
                   for i, sp in enumerate(spans)]
    except Exception as e:
        return None, None, f"ref span: {type(e).__name__}: {e}"[:300]
    try:
        for sp in s_spans:
            C.verify_entity_span(sp, strict=True)
    except Exception as e:
        return None, None, f"src span roundtrip: {type(e).__name__}: {e}"[:300]
    try:
        for sp in t_spans:
            C.verify_entity_span(sp, strict=True)
    except Exception as e:
        return None, None, f"ref span roundtrip: {type(e).__name__}: {e}"[:300]

    s_ranges = [(s["tok_start"], s["tok_end"]) for s in s_spans]
    t_ranges = [(t["tok_start"], t["tok_end"]) for t in t_spans]

    # --- 独立解 (左から順のカーソル探索) と突き合わせる ---
    cur, info = resolve_spans_with_cursor(
        bundle.tokenizer, item["text_a"], surfaces_a, add_bos=0,
        char_starts=[sp.get("char_start_a") for sp in spans])
    if cur is None:
        return None, None, f"cursor resolve: {info['reason']}"[:300]
    ok, bad = verify_span_roundtrip(bundle.tokenizer, item["text_a"], cur,
                                    surfaces_a, add_bos=0)
    if not ok:
        return None, None, f"cursor roundtrip: {bad}"[:300]
    if [tuple(int(v) for v in x) for x in cur] != [tuple(x) for x in s_ranges]:
        return None, None, (f"span 不一致 entity_token_span={s_ranges} "
                            f"cursor={[tuple(x) for x in cur]}")

    # --- 区間集合の健全性 (重なりはここで落とす。隣接は許す) ---
    try:
        check_span_sets(C.PROMPT_LEN, [C.PROMPT_LEN] * k, s_ranges, t_ranges)
    except AssertionError as e:
        return None, None, f"span set: {e}"[:300]

    n = [b - a for a, b in s_ranges]
    m = [b - a for a, b in t_ranges]
    detail = {
        "src": [list(x) for x in s_ranges],
        "ref": [list(x) for x in t_ranges],
        "n": n, "m": m,
        "delta_per_span": [m[i] - n[i] for i in range(k)],
        # Bark は長さを変えないので、実際に入る行数は常に n_i。
        "rows_written_per_span": n,
        "gap_tokens": (s_ranges[1][0] - s_ranges[0][1]) if k == 2 else None,
        "len_imbalance": abs(m[0] - m[1]) if k == 2 else None,
        "prompt_len": int(C.PROMPT_LEN),
        "n_real_tokens_src": int(in_a["n_real_tokens"]),
        "src_resolve_via": [p["via"] for p in info["per_span"]],
        "src_tokens": [s["entity_tokens"] for s in s_spans],
        "ref_tokens": [t["entity_tokens"] for t in t_spans],
        "splice_mode": ["token_wise" if m[i] == n[i] else "mean_pool_broadcast"
                        for i in range(k)],
    }
    return s_spans, t_spans, detail


# --------------------------------------------------------------------------
# 1 条件の生成
# --------------------------------------------------------------------------
def gen_condition(bundle, cond, item, in_a, ref_inputs, s_spans, t_spans,
                  layer, gen, collect_kappa=True, verify_prefill=True):
    """1 条件ぶんを生成する。呼び出しの順序は単一 entity 経路と同じ.

    `bark_splice.splice_synthesize` (bark_splice.py:1192-1242) の並びを写した:
      目標側 prefill -> 元文脈 prefill (照合用) -> hook -> _generate
    seed は `_generate` の中で generate の直前にリセットされる
    (bark_splice.py:1087)。prefill は乱数を消費しないので、枠の数が条件ごとに
    違っても 4 条件の生成開始時の RNG 状態は同じになる。
    """
    slots = COND_SLOTS[cond]
    if not slots:
        # BL は素の TTS。splice_synthesize を通さない (単一経路と同じ)。
        g = C.synthesize(bundle, item["text_a"], gen=gen)
        g.update({"delta_tok": 0, "T_out": int(C.PROMPT_LEN), "out_ranges": [],
                  "k_spans": 0, "splice_mode": [], "slots": []})
        return g

    li = C.hidden_index_for_layer(layer)
    tgt_hiddens, tgt_hs_by_slot = [], {}
    for i in slots:
        tgt_hs, tgt_tap = C.prefill_forward(bundle, ref_inputs[i], layer=layer,
                                            need_all_hidden=collect_kappa)
        if tgt_tap is None:
            raise RuntimeError(f"枠 {i} の目標文脈 layer tap が取れなかった")
        if collect_kappa and tgt_hs is not None and li < len(tgt_hs):
            # tap と hidden_states[l+1] が一致することの自己検査 (index 規約)。
            if not torch.allclose(tgt_tap.float(), tgt_hs[li].float(),
                                  atol=1e-3, rtol=1e-3):
                raise RuntimeError(
                    f"layers[{layer}] の tap と hidden_states[{li}] が一致しない。"
                    f"index 規約が崩れているので中断する。")
        t0, t1 = t_spans[i]["tok_start"], t_spans[i]["tok_end"]
        tgt_hiddens.append(tgt_tap[0, t0:t1].clone())
        tgt_hs_by_slot[i] = tgt_hs

    src_hs, src_tap = C.prefill_forward(bundle, in_a, layer=layer,
                                        need_all_hidden=collect_kappa)
    splice = MultiPrefillSplice(
        bundle.layer_module(layer), [s_spans[i] for i in slots], tgt_hiddens,
        prompt_len=C.PROMPT_LEN, expect_module=bundle.layer_module(layer),
        verify_hidden=(src_tap if verify_prefill else None))
    g = C._generate(bundle, in_a, gen=gen, splice=splice)  # 発火回数はここで検査される
    g.update(splice.stats())
    g.update({"delta_tok": splice.delta_tok, "T_out": splice.T_out,
              "out_ranges": splice.out_ranges, "slots": list(slots),
              "layer": int(layer), "hidden_index": int(li)})
    if collect_kappa and src_hs is not None:
        kap = {}
        for i in slots:
            hs = tgt_hs_by_slot.get(i)
            if hs is None:
                continue
            ks = C.kappa_by_layer(src_hs, hs, s_spans[i], t_spans[i])
            kap[f"slot{i + 1}"] = {"at_splice": round(ks[li], 6),
                                   "last_layer": round(ks[-1], 6),
                                   "by_layer": [round(v, 6) for v in ks]}
        if kap:
            g["kappa"] = kap
    return g


# --------------------------------------------------------------------------
# 指紋 (再開の照合用)
# --------------------------------------------------------------------------
def run_fingerprint(args, bundle=None):
    fp = {
        "script": os.path.basename(__file__), "mode": MODE_TAG,
        "pairs_module": args.pairs_module, "lang": args.lang,
        "layer": int(args.layer), "seed": int(args.seed),
        "do_sample": bool(args.do_sample),
        "semantic_temperature": args.semantic_temperature,
        "semantic_top_k": args.semantic_top_k,
        "coarse_temperature": args.coarse_temperature,
        "coarse_top_k": args.coarse_top_k,
        "fine_temperature": args.fine_temperature,
        "semantic_max_new_tokens": args.semantic_max_new_tokens,
        "min_eos_p": args.min_eos_p, "min_sec": args.min_sec,
        "model_id": args.model_id, "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "voice_preset": args.voice_preset,
        "device_type": ("cuda" if str(args.device).startswith("cuda") else str(args.device)),
        "splice_target": f"model.semantic.layers[{args.layer}].output",
        "splice_len_policy": "token_wise|mean_pool_broadcast",
        "splice_builder": "splice_multi.build_spliced_hidden_multi",
        "verify_prefill": bool(args.verify_prefill),
    }
    if bundle is not None:
        fp["model_revision"] = os.path.basename(str(bundle.source).rstrip("/"))
        fp["preset_sha256"] = bundle.preset_sha256
        fp["num_layers"] = int(bundle.num_layers)
        fp["sampling_rate"] = int(bundle.sampling_rate)
    return fp


def fp_hash(fp):
    return hashlib.sha256(
        json.dumps(fp, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs_module", required=True,
                   help="data/<name>.py。2 枠 (en_multientity_300 / ja_multientity_300) "
                        "か 1 枠 (en_cross_culture_names / ja_cross_culture_names)")
    p.add_argument("--lang", required=True, choices=["en", "ja"],
                   help="中国語のデータは存在しないので zh は受け付けない")
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--layer", type=int, default=C.DEFAULT_LAYER,
                   help=f"差し替える semantic 層 (既定 {C.DEFAULT_LAYER} = L-2)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--voice_preset", default=None,
                   help="既定は言語ごとに固定 (en=v2/en_speaker_6, ja=v2/ja_speaker_0)")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--model_id", default=C.DEFAULT_MODEL_ID)
    p.add_argument("--allow_download", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn_implementation", default="eager", choices=["eager"])
    p.add_argument("--no_strict_weights", dest="strict_weights", action="store_false")
    p.add_argument("--semantic_temperature", type=float,
                   default=C.DEFAULT_SEMANTIC_TEMPERATURE)
    p.add_argument("--semantic_top_k", type=int, default=C.DEFAULT_SEMANTIC_TOP_K)
    p.add_argument("--coarse_temperature", type=float,
                   default=C.DEFAULT_COARSE_TEMPERATURE)
    p.add_argument("--coarse_top_k", type=int, default=C.DEFAULT_COARSE_TOP_K)
    p.add_argument("--fine_temperature", type=float, default=C.DEFAULT_FINE_TEMPERATURE)
    p.add_argument("--semantic_max_new_tokens", type=int,
                   default=C.DEFAULT_SEMANTIC_MAX_NEW_TOKENS)
    p.add_argument("--min_eos_p", type=float, default=C.DEFAULT_MIN_EOS_P)
    p.add_argument("--no_min_eos_p", dest="min_eos_p", action="store_const", const=None)
    p.add_argument("--greedy", dest="do_sample", action="store_false")
    p.add_argument("--min_sec", type=float, default=0.5)
    p.add_argument("--fail_on_cap", action="store_true")
    p.add_argument("--no_kappa", dest="collect_kappa", action="store_false")
    p.add_argument("--no_verify_prefill", dest="verify_prefill", action="store_false",
                   help="生成時 prefill hidden と別経路 prefill の突き合わせを切る "
                        "(既定は入り。切らないこと)")
    p.add_argument("--utmos", action="store_true",
                   help="UTMOS22 を測る (TORCH_HOME のキャッシュを使う)")
    p.add_argument("--null_check_pairs", type=int, default=12,
                   help="最初の何対で null 判定するか。0 で無効")
    p.add_argument("--allow_null_layer", action="store_true",
                   help="INJ の意味トークン列が BL と全対一致でも続ける。"
                        "層 22 は単一 entity でも 259/300 が同一だったので本番は必須")
    p.add_argument("--equiv_ref_dir", default=None,
                   help="単一区間 (k=1) のとき、既存の単一 entity wav と md5 を"
                        "突き合わせる先。例 eval_results/_cc_wavs_a/bark_en")
    # **後から目標文だけを足すための経路 (2026-09-22)。**既定は off なので
    # 既存の呼び出しはこの 2 つを一切通らない。
    p.add_argument("--tgt_only", action="store_true",
                   help="枠ごとの目標文 tgt1 / tgt2 だけを合成する (差し替えなし)。"
                        "既存の bl/inj1/inj2/inj12 と results.json は触らない")
    p.add_argument("--dry_run", action="store_true",
                   help="モデルを読まずに、出すつもりの文とファイル名を並べる")
    p.add_argument("--overwrite", action="store_true", help="再開せず作り直す")
    p.add_argument("--force_resume_mismatch", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=5)
    return p


def main():
    args = build_argparser().parse_args()
    os.chdir(ROOT)
    if args.voice_preset is None:
        args.voice_preset = DEFAULT_PRESET[args.lang]
    if args.out_dir is None:
        args.out_dir = os.path.join(
            "eval_results", f"bark_multientity_{args.lang}_L{args.layer}")
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "checkpoint.json")
    results_path = os.path.join(args.out_dir, "results.json")

    items, k = load_items(args.pairs_module, args.lang, args.n)
    conds = CONDITIONS_BY_K[k]
    print(f"[info] {args.pairs_module}: {len(items)} 文 / 枠 {k} 個 / lang={args.lang}",
          flush=True)
    print(f"[info] 条件 {conds} / layer={args.layer} / preset={args.voice_preset}",
          flush=True)
    print(f"[info] out_dir={args.out_dir}", flush=True)
    print(f"[info] HF_HOME={os.environ['HF_HOME']} TORCH_HOME={os.environ['TORCH_HOME']}",
          flush=True)
    if args.equiv_ref_dir and k != 1:
        raise SystemExit("--equiv_ref_dir は 1 枠のデータ (cross_culture) 専用")
    if args.equiv_ref_dir:
        args.equiv_ref_dir = os.path.abspath(args.equiv_ref_dir)
        if not os.path.isdir(args.equiv_ref_dir):
            raise SystemExit(f"--equiv_ref_dir が無い: {args.equiv_ref_dir}")
        print(f"[info] 単一区間の md5 等価性を {args.equiv_ref_dir} と突き合わせる "
              f"({EQUIV_COND_MAP})", flush=True)

    if args.tgt_only and k != 2:
        raise SystemExit("--tgt_only は 2 枠のデータ専用 (今のデータは 枠 %d 個)" % k)
    if args.dry_run:
        # モデルを読む前に返す (CPU で名前と文を確かめるための経路)。
        # checkpoint も読まない (指紋の食い違いで止まらないように)。
        dry_run_report(items, args.out_dir, tag="bark")
        return

    fp = run_fingerprint(args)
    print(f"[info] 設定の指紋 (モデル読み込み前) = {fp_hash(fp)}", flush=True)

    # ---------------- 再開 ----------------
    rows, skipped = [], []
    if os.path.isfile(ckpt_path) and not args.overwrite:
        st = json.load(open(ckpt_path))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != MODE_TAG]
        if bad:
            raise SystemExit(
                f"checkpoint の mode が違う ({ckpt_path} の対 {bad[:5]})。"
                f"別の --out_dir を使うこと。")
        prev_fp = st.get("fingerprint") or {}
        diff = [kk for kk in set(prev_fp) | set(fp)
                if prev_fp.get(kk) != fp.get(kk)
                and kk not in ("model_revision", "preset_sha256", "num_layers",
                               "sampling_rate")]
        if diff and rows:
            msg = (f"checkpoint の設定が今回と食い違う: {sorted(diff)[:8]}\n"
                   f"  そのまま混ぜると別設定の wav が 1 つの結果に混ざる。")
            if not args.force_resume_mismatch:
                raise SystemExit("[resume] " + msg +
                                 " --overwrite か --force_resume_mismatch を付けること。")
            print("[warn] " + msg, flush=True)
        print(f"[resume] {len(rows)} 行 / skip {len(skipped)} 行を読んだ", flush=True)
    done = {int(r["pair_idx"]) for r in rows} | {int(s["pair_idx"]) for s in skipped}

    # ---------------- モデル ----------------
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[info] loading {args.model_id} (dtype={args.dtype}, device={args.device}) ...",
          flush=True)
    t0 = time.time()
    bundle = C.load_bark(args.model_id, device=args.device, dtype=dtype,
                         allow_download=args.allow_download,
                         strict_weights=args.strict_weights,
                         voice_preset=args.voice_preset,
                         attn_implementation=args.attn_implementation)
    print(f"[info] loaded in {time.time() - t0:.1f}s from {bundle.source}", flush=True)
    print(f"[info] semantic {bundle.num_layers} 層 / hidden {bundle.hidden_size} / "
          f"sr {bundle.sampling_rate} / preset sha {bundle.preset_sha256}", flush=True)
    C.assert_splice_target(bundle, args.layer)
    fp = run_fingerprint(args, bundle)
    print(f"[info] 設定の指紋 = {fp_hash(fp)}", flush=True)

    utmos = None
    if args.utmos:
        print("[info] loading UTMOS22 ...", flush=True)
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(args.device).eval()

    gen = C.GenConfig(seed=args.seed, do_sample=args.do_sample,
                      semantic_temperature=args.semantic_temperature,
                      semantic_top_k=args.semantic_top_k,
                      coarse_temperature=args.coarse_temperature,
                      coarse_top_k=args.coarse_top_k,
                      fine_temperature=args.fine_temperature,
                      semantic_max_new_tokens=args.semantic_max_new_tokens,
                      min_eos_p=args.min_eos_p, min_sec=args.min_sec,
                      fail_on_cap=args.fail_on_cap)

    import soundfile as sf

    if args.tgt_only:
        # 差し替えをしないので、合成は bl 条件 (gen_condition の slots 無し) と
        # 同じ C.synthesize。渡す文だけが text_refs[slot] に変わる。
        def _synth_tgt(text, path):
            g = C.synthesize(bundle, text, gen=gen)
            _, _dur, qc = C.save_wav(path, g["audio"], bundle.sampling_rate)
            info = sf.info(path)
            return {"dur_sec": round(info.frames / float(info.samplerate), 4),
                    "sem_fp": g["sem_fp"], "n_semantic": int(g["n_semantic"]),
                    "eos_found": bool(g["eos_found"]),
                    "hit_cap": bool(g["hit_cap"]),
                    # C.save_wav の 3 番目は {"peak", "n_clipped"} の辞書
                    # (bark_splice.py:1266)。本体の記録 (861-862 行) と同じ鍵で残す。
                    "peak": qc["peak"], "n_clipped": qc["n_clipped"],
                    "md5": _md5(path)}

        run_targets(items, args.out_dir, _synth_tgt, tag="bark",
                    ckpt_every=args.ckpt_every, log_every=args.log_every,
                    stop_on_error=args.stop_on_error,
                    extra_summary={"method": "bark", "lang": args.lang,
                                   "pairs_module": args.pairs_module,
                                   "model_id": args.model_id,
                                   "dtype": args.dtype,
                                   "voice_preset": args.voice_preset,
                                   "preset_sha256": bundle.preset_sha256,
                                   "seed": int(args.seed)})
        return

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"rows": rows, "skipped": skipped, "fingerprint": fp,
                       "args": vars(args)}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, ckpt_path)

    run_warnings = []
    n_cmp = n_all_same = 0   # BL と全 INJ の意味トークン列が一致した対
    t_start = time.time()
    interrupted = None
    try:
        for i, it in enumerate(items):
            pidx = int(it["pair_idx"])
            if pidx in done:
                continue

            # --- 入力と区間 ---
            try:
                in_a = C.prepare_inputs(bundle, it["text_a"])
                ref_inputs = [C.prepare_inputs(bundle, t) for t in it["text_refs"]]
            except Exception as e:
                r = f"prepare_inputs: {type(e).__name__}: {e}"[:300]
                print(f"  [{pidx}] SKIP {r}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": r})
                done.add(pidx)
                continue
            s_spans, t_spans, detail = resolve_item_spans(bundle, it, in_a, ref_inputs)
            if s_spans is None:
                print(f"  [{pidx}] SKIP {detail}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": detail})
                done.add(pidx)
                continue

            row = {
                "pair_idx": pidx, "mode": MODE_TAG,
                "template_id": it.get("template_id"), "lang": args.lang,
                "layer": int(args.layer), "seed": int(args.seed),
                "text_a": it["text_a"], "text_refs": it["text_refs"],
                "text_both": it["text_both"],
                "gap_words": it.get("gap_words"),
                "slot1_at_start": it.get("slot1_at_start"),
                "repeat_surface": it.get("repeat_surface", False),
                "words_a": [sp["word_a"] for sp in it["spans"]],
                "words_b": [sp["word_b"] for sp in it["spans"]],
                "span_tok": detail,
                "T_src": int(C.PROMPT_LEN),
                "conds": {},
            }

            ok = True
            for cond in conds:
                try:
                    g = gen_condition(bundle, cond, it, in_a, ref_inputs,
                                      s_spans, t_spans, args.layer, gen,
                                      collect_kappa=args.collect_kappa,
                                      verify_prefill=args.verify_prefill)
                except Exception as e:
                    import traceback
                    r = f"{cond} gen: {type(e).__name__}: {e}"[:300]
                    print(f"  [{pidx}] {r}", flush=True)
                    print(traceback.format_exc()[-600:], flush=True)
                    skipped.append({"pair_idx": pidx, "reason": r})
                    ok = False
                    break

                wav_path = os.path.join(args.out_dir, f"s{pidx:04d}_{cond}.wav")
                try:
                    _, _dur, qc = C.save_wav(wav_path, g["audio"], bundle.sampling_rate)
                    info = sf.info(wav_path)
                    rec = {
                        "wav": os.path.basename(wav_path),
                        "delta_tok": int(g["delta_tok"]),
                        "T_out": int(g["T_out"]),
                        "out_ranges": [list(x) for x in g["out_ranges"]],
                        "target_text": it["text_a"],
                        "dur_sec": round(info.frames / float(info.samplerate), 4),
                        # Bark は入力テキストを条件間で変えない (長さ不変の
                        # 差し替えなので長さ一致ダミーが要らない)。
                        "dur_text_is_text_a": True,
                        "md5": _md5(wav_path),
                        "sem_fp": g["sem_fp"], "n_semantic": int(g["n_semantic"]),
                        "eos_found": bool(g["eos_found"]),
                        "hit_cap": bool(g["hit_cap"]),
                        "truncated_at_cap": bool(g["truncated_at_cap"]),
                        "stop_reason": g["stop_reason"],
                        "peak": qc["peak"], "n_clipped": qc["n_clipped"],
                        "splice_mode": g.get("splice_mode"),
                        "slots": g.get("slots"),
                        "fired": g.get("fired"),
                        "prefill_hidden_max_abs_diff":
                            g.get("prefill_hidden_max_abs_diff"),
                    }
                    if "kappa" in g:
                        rec["kappa"] = g["kappa"]
                    if utmos is not None:
                        a, sr = sf.read(wav_path, dtype="float32")
                        with torch.no_grad():
                            rec["utmos"] = float(utmos(
                                torch.from_numpy(a).unsqueeze(0).to(args.device), sr))
                    row["conds"][cond] = rec
                except Exception as e:
                    r = f"{cond} save/score: {type(e).__name__}: {e}"[:300]
                    print(f"  [{pidx}] {r}", flush=True)
                    skipped.append({"pair_idx": pidx, "reason": r})
                    ok = False
                    break

            done.add(pidx)
            if not ok:
                if args.stop_on_error:
                    save_ckpt()
                    raise SystemExit("--stop_on_error")
                continue

            # --- splice が音に出たか (Bark で最も重要な検査) ---
            bl_fp = row["conds"]["bl"]["sem_fp"]
            same_flags = []
            for c in conds:
                if c == "bl":
                    continue
                same = (row["conds"][c]["sem_fp"] == bl_fp)
                row["conds"][c]["differs_from_bl"] = bool(not same)
                same_flags.append(same)
            row["all_inj_identical_to_bl"] = bool(same_flags and all(same_flags))
            n_cmp += 1
            n_all_same += int(row["all_inj_identical_to_bl"])

            # Bark は長さを変えないので総ずれは常に 0。T5Gemma 版と同じ鍵で記録する。
            row["total_position_drift"] = int(row["conds"][conds[-1]]["delta_tok"])

            # --- 単一区間の md5 等価性 ---
            if args.equiv_ref_dir:
                eq = {}
                for c, ref_c in EQUIV_COND_MAP.items():
                    if c not in row["conds"]:
                        continue
                    ref = os.path.join(args.equiv_ref_dir, f"s{pidx:04d}_{ref_c}.wav")
                    if not os.path.isfile(ref):
                        eq[c] = {"ref": ref, "status": "ref_missing"}
                        continue
                    rm = _md5(ref)
                    eq[c] = {"ref": ref, "ref_md5": rm,
                             "md5": row["conds"][c]["md5"],
                             "match": bool(rm == row["conds"][c]["md5"])}
                row["equiv"] = eq
                tag = " ".join(f"{c}={'一致' if v.get('match') else v.get('status') or '不一致'}"
                               for c, v in eq.items())
                print(f"  [{pidx}] md5 {tag}", flush=True)

            rows.append(row)

            if len(rows) % args.log_every == 0:
                el = time.time() - t_start
                per = el / max(1, len(rows))
                c0 = row["conds"]
                print(f"  [{len(rows)}/{len(items)}] idx={pidx} "
                      f"gap_tok={detail['gap_tokens']} "
                      f"Δ={row['total_position_drift']} "
                      f"同一={n_all_same}/{n_cmp} "
                      "sec(" + "/".join(f"{c0[c]['dur_sec']:.2f}" for c in conds) + ") "
                      f"{per:.1f}s/対 残り {per * (len(items) - i - 1) / 60:.1f} 分",
                      flush=True)
            if len(rows) % args.ckpt_every == 0:
                save_ckpt()

            # --- null 層の早期検出 ---
            if (args.null_check_pairs and not args.allow_null_layer
                    and n_cmp >= int(args.null_check_pairs) and n_all_same == n_cmp):
                save_ckpt()
                raise SystemExit(
                    f"[null] 最初の {n_cmp} 対すべてで、3 つの INJ の意味トークン列が\n"
                    f"  BL と完全に一致した。層 {args.layer} では差し替えが下流に届いて\n"
                    f"  いない。その音声は BL とビット単位で同じになる。\n"
                    f"  Bark は prompt 257 行のうち 244 行がパディングで、テキストの\n"
                    f"  情報が浅い層でパディング行へ移るため、L5 以降は prompt を\n"
                    f"  ほとんど読まない (単一 entity では 300 対中 259 対が同一)。\n"
                    f"  対処: --layer を {C.DIAG_LAYERS} から選んで掃くか、\n"
                    f"        この層の null を結果として記録するなら\n"
                    f"        --allow_null_layer を付けて回し直すこと\n"
                    f"        (そのとき summary の n_identical_to_bl が {n_cmp} 以上になる)。\n"
                    f"  checkpoint.json は書いた。results.json は書いていない。")
    except SystemExit:
        save_ckpt()
        raise
    except KeyboardInterrupt as e:
        interrupted = e
        print("\n[warn] 中断された。ここまでを書き出す。", flush=True)
    except Exception as e:
        interrupted = e
        print(f"\n[warn] 例外で中断: {type(e).__name__}: {e}", flush=True)

    save_ckpt()

    # ---------------- summary ----------------
    def _mean(vals):
        v = [x for x in vals if x is not None]
        return round(float(np.mean(v)), 4) if v else None

    summary = {
        "pairs_module": args.pairs_module, "lang": args.lang,
        "mode": MODE_TAG, "k_spans": k, "conds": list(conds),
        "layer": int(args.layer), "seed": int(args.seed),
        "voice_preset": args.voice_preset, "preset_sha256": bundle.preset_sha256,
        "model_source": bundle.source, "prompt_len": int(C.PROMPT_LEN),
        "n_items": len(items), "n_generated": len(rows), "n_skipped": len(skipped),
        "gen": gen.as_dict(), "fingerprint": fp, "fingerprint_hash": fp_hash(fp),
        "mean_dur_sec": {c: _mean([r["conds"][c]["dur_sec"] for r in rows
                                   if c in r["conds"]]) for c in conds},
        "mean_n_semantic": {c: _mean([r["conds"][c]["n_semantic"] for r in rows
                                      if c in r["conds"]]) for c in conds},
        "n_hit_cap": {c: sum(1 for r in rows
                             if r["conds"].get(c, {}).get("hit_cap")) for c in conds},
        # Bark は系列長を変えないので 0 になるはず。0 でなければ実装の誤り。
        "mean_abs_position_drift": _mean([abs(r["total_position_drift"]) for r in rows]),
        "max_abs_delta_tok": max([abs(r["conds"][c]["delta_tok"])
                                  for r in rows for c in r["conds"]] or [0]),
        "n_compared_bl_inj": n_cmp,
        "n_all_inj_identical_to_bl": n_all_same,
        "n_identical_to_bl": {
            c: sum(1 for r in rows
                   if r["conds"].get(c, {}).get("differs_from_bl") is False)
            for c in conds if c != "bl"},
        "splice_mode_counts": {},
        "elapsed_min": round((time.time() - t_start) / 60, 2),
    }
    mode_count = {}
    for r in rows:
        for mm in (r["span_tok"].get("splice_mode") or []):
            mode_count[mm] = mode_count.get(mm, 0) + 1
    summary["splice_mode_counts"] = mode_count
    if utmos is not None:
        summary["utmos"] = {c: _mean([r["conds"][c].get("utmos") for r in rows
                                      if c in r["conds"]]) for c in conds}
    if args.equiv_ref_dir:
        eq_sum = {}
        for c in EQUIV_COND_MAP:
            vals = [r["equiv"][c] for r in rows if c in (r.get("equiv") or {})]
            eq_sum[c] = {"n": len(vals),
                         "n_match": sum(1 for v in vals if v.get("match")),
                         "n_ref_missing": sum(1 for v in vals
                                              if v.get("status") == "ref_missing"),
                         "mismatched_pairs": [r["pair_idx"] for r in rows
                                              if (r.get("equiv") or {}).get(c, {})
                                              .get("match") is False][:20]}
        bl_ok = eq_sum.get("bl", {}).get("n_match") == eq_sum.get("bl", {}).get("n")
        inj_ok = eq_sum.get("inj1", {}).get("n_match") == eq_sum.get("inj1", {}).get("n")
        # 層 22 では多くの対で INJ が BL とビット単位で同一になる。その対の
        # md5 一致は「BL 経路が合っている」ことしか示さない。差し替えが音に
        # 出た対が何対あったかを必ず併記する。
        n_nontrivial = sum(1 for r in rows
                           if r["conds"].get("inj1", {}).get("differs_from_bl") is True)
        eq_sum["n_nontrivial"] = n_nontrivial
        if bl_ok and inj_ok:
            verdict = (f"PASS: 単一区間で既存の単一 entity 経路とビット一致 "
                       f"(うち差し替えが音に出た対 = 経路を実際に試せた対は "
                       f"{n_nontrivial} 対。0 なら BL 経路しか試せていないので "
                       f"--n を増やすこと)")
        elif not bl_ok:
            verdict = ("FAIL(環境): BL からして不一致。移植ではなく実行環境の差 "
                       "(GPU 種別 / dtype / preset / 復号設定 / 乱数) を先に疑うこと")
        else:
            verdict = ("FAIL(移植): BL は一致したが INJ が不一致。複数 entity 経路の "
                       "区間解決か置換行の作り方が単一経路と違う")
        summary["equivalence"] = {"ref_dir": args.equiv_ref_dir,
                                  "cond_map": EQUIV_COND_MAP,
                                  "per_cond": eq_sum, "verdict": verdict}
    if n_cmp and n_all_same:
        run_warnings.append(
            f"INJ の意味トークン列が BL と一致した対が {n_all_same}/{n_cmp}。"
            f"その対の音声は BL とビット単位で同じで、注入成功率の差は必ず 0 になる。"
            f"層 {args.layer} の効きを疑うこと。")
    if skipped:
        run_warnings.append(f"{len(skipped)} 文を skip した (results.json['skipped'])")
    if run_warnings:
        summary["warnings"] = run_warnings

    tmp = results_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "rows": rows, "skipped": skipped}, f,
                  ensure_ascii=False, indent=1)
    os.replace(tmp, results_path)

    print(f"\n=== bark 複数 entity {args.pairs_module} (layer {args.layer}) ===")
    print(json.dumps({kk: vv for kk, vv in summary.items()
                      if kk not in ("fingerprint",)}, ensure_ascii=False, indent=1))
    print(f"[done] {results_path}")
    print("[next] 採点は既存の採点器で行う (新しい採点器は作らない):")
    print(f"  SLURM_TIME=02:00:00 slurm_run.sh \\")
    print(f"    python3 -u /mnt/scripts/analysis/score_multientity.py \\")
    print(f"      --dir {args.out_dir} --lang {args.lang}")
    if interrupted is not None:
        raise SystemExit(f"中断したまま終了: {type(interrupted).__name__}: {interrupted}")


if __name__ == "__main__":
    main()
