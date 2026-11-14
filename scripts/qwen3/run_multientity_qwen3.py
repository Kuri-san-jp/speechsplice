#!/usr/bin/env python3
"""Qwen3-TTS: 1 文中の 2 箇所を同時に差し替える生成 (BL / INJ_1 / INJ_2 / INJ_12).

`scripts/analysis/generate_multientity.py`（T5Gemma 版）の構成をそのまま写した
**新規**スクリプト。既存ファイルは 1 行も変更していない。差し替えの中身は
`scripts/projection/splice_multi.py` の `build_spliced_hidden_multi` を import して
使う（再実装していない）。単一 entity の本番経路
`scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py` から
再利用する部品は import する:

    eval_qwen3_injection.build_text_only_inputs      ドナー側 forward の入力組み立て
    eval_qwen3_injection.get_layer_hidden_states     層 26 出力の取り出し
    （スパン解決は splice_multi.resolve_spans_with_cursor に統一。k=1 で
      eval_qwen3_injection.find_entity_token_positions と同一の値を返すことを
      scripts/qwen3/_cpu_verify_multientity_qwen3.py が全 600 文で確認する）

Qwen3 固有の事情（T5Gemma と違う点）
------------------------------------
1. **差し替えは長さ不変。** T5Gemma は encoder 出力を可変長 splice できるが、
   Qwen3 は decoder-only で、層 26 の forward hook が書き込めるのは
   prefill 済み系列の「その位置」だけである。系列長を変えると KV cache と
   position が壊れるので、ドナー側のスパン長 m_i を受け側の長さ n_i に
   合わせてから splice する:

       m_i == n_i : ドナーのスパンをそのまま
       m_i != n_i : ドナーのスパンを時間平均して n_i 個に broadcast

   これは単一 entity 版（eval_qwen3_crossculture_acoustic_fixed.py:187-191）と
   同じ規則である。長さを合わせてから `build_spliced_hidden_multi` に渡すので
   `delta_tok` は常に 0、`T_out == T_src` になる。**index を自分でずらす処理は
   一切書いていない**（複数スパンの順序・重なり・out_ranges は splice_multi 側）。

2. **text body の開始位置（text_body_offset）は layout ごとに違う。**
   ドナー側 `build_text_only_inputs` は
   `[0..2]=role, [3..6]=codec tag/pad, [7]=tts_bos, [8..]=本文` なので 8。
   受け側（generate の prefill）は
   `core/models/modeling_qwen3_tts.py:2135-2186` の組み立てにより

       codec_prefill = 4 (think, think_bos, language_id, think_eos)
                       + 1 (x_vector speaker embed)
                       + 2 (codec_pad, codec_bos)          = 7
       role 3 + (7 - 1) = 9                                → 本文は index 9

   となり、言語に依らず 9 である（"auto" 指定で language_id が無いときだけ 8）。
   既存の単一 entity 本番スクリプトは受け側に
   `9 if lang == "ja" else 8`（同 eval_...fixed.py:170-171）を使っており、
   **英語だけ 1 つ手前（tts_bos の位置）に書き込んでいる**。
   本スクリプトは `--compat fixed`（既定）で両側とも
   「その layout の本文開始位置」という同じ規約に統一し、
   `--compat legacy` で既存の非対称をそのまま再現する（md5 等価性試験用）。

3. **prefill に本文が何トークン載るか。** `generate_voice_clone` の
   `non_streaming_mode` が False（既存スクリプトは未指定＝False）だと
   prefill に載る本文は**先頭 1 トークンだけ**で、残りは復号ステップごとに
   1 トークンずつ供給される（modeling_qwen3_tts.py:2196-2233）。
   hook は `seq_len <= 1` を素通しするので、この設定では
   **枠 2 には原理的に届かない**。`--compat fixed` は
   `non_streaming_mode=True` を渡して本文全体を prefill に載せる。
   `--compat legacy` は False のままにする（既存と同一）。

使い方
------
  # 本番（複数 entity, 2 枠）
  SLURM_TIME=06:00:00 slurm_run.sh \\
      python3 -u /mnt/scripts/qwen3/run_multientity_qwen3.py \\
          --pairs_module en_multientity_300 --lang en

  # 単一区間の md5 等価性試験（cross-culture 300 対の先頭 n 件）
  SLURM_TIME=01:00:00 slurm_run.sh \\
      python3 -u /mnt/scripts/qwen3/run_multientity_qwen3.py \\
          --mode cc_single --lang en --n 5 --compat legacy \\
          --output_dir eval_results/_md5_qwen3_en_k1

出力
----
  <out_dir>/results.json     summary + rows（conds: bl/inj1/inj2/inj12）
  <out_dir>/checkpoint.json  途中再開用
  <out_dir>/s{pair_idx:04d}_{bl,inj1,inj2,inj12}.wav
  cc_single モードでは conds は bl/inj/tgt、ファイル名も既存と同じ
  s{pair_idx:04d}_{bl,inj,tgt}.wav になる（md5 をそのまま比較できる）。
"""
import os

# HF_HOME は setdefault では効かない（slurm_run.sh が実体の無い値を渡す）ので代入する。
ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "modules")
os.environ["TORCH_HOME"] = os.path.join(ROOT, ".cache", "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(ROOT, ".cache")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

PROJECT_ROOT = ROOT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 複数スパンの splice は再実装せずここから import する。
from splice_multi import (  # noqa: E402
    build_spliced_hidden_multi, check_span_sets, resolve_spans_with_cursor,
    span_deltas, verify_span_roundtrip,
)

# --- 既存の単一 entity 本番経路と同じ定数 -----------------------------------
# eval_qwen3_crossculture_acoustic_fixed.py:45-46
# zh を 2026-09-18 に追加。**公表セルを作った
# `scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py:45` と同じ表記**
# にそろえる（LANG_CODE = {"en": "English", "ja": "Japanese", "zh": "Chinese"}）。
# text_body_offset も同 171 行と同じく「ja だけ 9、他は 8」なので zh は 8 のまま。
LANG_CODE = {"en": "English", "ja": "Japanese", "zh": "Chinese"}
SEED = 42
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_NEW_TOKENS = 512          # 同 eval_...fixed.py:140
DEFAULT_LAYER = 26            # talker の層 26 出力

# eval_qwen3_crossculture_acoustic_fixed.py:112-119（Table 3 共通の自然音声 ref）
REFS = {
    "en": ("eval_results/_natural_refs_norm/en_natural_norm.wav",
           "Let me see, I'll give them a new pair of boots every Christmas."),
    "ja": ("eval_results/_natural_refs_norm/ja_natural_norm.wav",
           "太平洋は、世界でいちばん大きな大洋です。"),
    # zh を 2026-09-18 に追加。**公表セルを作った
    # `scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py:117-118`
    # と同じ音声・同じ書き起こし**にそろえる。ここを足し忘れると
    # 539 行の `REFS[args.lang]` が KeyError で落ちる。
    "zh": ("eval_results/_natural_refs_norm/zh_natural_norm.wav",
           "印度素有嫁女儿陪送嫁妆的传统。"),
}

# processor.tokenizer が見ているのは常にこの接頭辞付きの文字列
# （eval_qwen3_injection.py:302 の _HEAD と同一）。
HEAD = "<|im_start|>assistant\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"

# ドナー側 layout の本文開始位置。eval_qwen3_injection.py:383-387
#   input_text_ids[0, :3] = role / [3..6] = codec tag と pad / [7] = tts_bos /
#   [8 : 8+text_len-3] = 本文
DONOR_TEXT_BODY_OFFSET = 8

CONDITIONS_MULTI = ["bl", "inj1", "inj2", "inj12"]
# **枠ごとの目標文（2026-09-20 追加、--tgt_only のときだけ）。**
# 複数 entity を本体セルと同じ PIB で測るには、目標を基準にした実体語 PER が
# 要り、そのための目標音声が要る。tgt1 / tgt2 はそれぞれ 1 枠だけ差し替えた文で、
# 基線との差が 1 箇所なので本体セルの採点器（diff_block）がそのまま効く。
# 2 枠とも差し替えた 1 本だけを作る案は却下した（隣り合う枠で差分の塊が融合する）。
TGT_SLOT_CONDS = {"tgt1": 0, "tgt2": 1}
COND_SLOTS_MULTI = {"bl": [], "inj1": [0], "inj2": [1], "inj12": [0, 1]}
# cc_single（単一区間の md5 試験）は既存と同じ 3 本
CONDITIONS_CC = ["bl", "tgt", "inj"]
COND_SLOTS_CC = {"bl": [], "tgt": [], "inj": [0]}


def seed_all(s):
    """eval_qwen3_crossculture_acoustic_fixed.py:49-51 と同一。"""
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)


def rng_fingerprint():
    """RNG 状態の指紋（**状態を進めない。CUDA にも触らない**）.

    4 条件でこれが一致することが「乱数の消費順序が揃っている」ことの証拠。
    `get_rng_state` は読み出しだけで消費しないので生成結果に影響しない。
    CUDA 側をわざと見ないのは 2 つの理由から:
      * `torch.cuda.get_rng_state()` は CUDA を初期化するので、CPU 検証で
        ログインノードの GPU を掴んでしまう
      * `seed_all` は CPU/CUDA/numpy/random を同じ seed から設定するので、
        CPU 側 3 つが一致すれば seed が戻されたことの証拠になる
    既存スクリプトに無い処理なので、生成に影響しない読み出しだけに限る。
    """
    h = hashlib.md5()
    h.update(torch.random.get_rng_state().numpy().tobytes())
    h.update(repr(random.getstate()).encode())
    st = np.random.get_state()
    h.update(repr((st[0], st[1].tobytes(), st[2], st[3], st[4])).encode())
    return h.hexdigest()[:16]


# --- layout ----------------------------------------------------------------

def recv_text_body_offset(config, language, x_vector_only=True):
    """generate の prefill における本文開始 index を config から導く.

    根拠は core/models/modeling_qwen3_tts.py:
      2135-2148  language_id が無ければ codec tag 3 個、あれば 4 個
      2163-2170  x_vector / icl なら speaker embed が 1 個入る
      2154-2161  末尾に codec_pad と codec_bos の 2 個
      2182-2184  _talker_input_embed = tts_pad*(len-2) + tts_bos  (= len-1 個)
      2186       role 3 個を前に付ける
    したがって本文開始 = 3 + (len(codec_input_embedding) - 1)。
    """
    tcfg = config.talker_config
    lang_ids = getattr(tcfg, "codec_language_id", None) or {}
    n_tag = 4 if str(language).lower() in lang_ids else 3
    n_codec = n_tag + (1 if x_vector_only else 0) + 2
    return 3 + (n_codec - 1)


def legacy_recv_text_body_offset(lang):
    """既存の単一 entity 本番スクリプトが使っている値（非対称）.

    eval_qwen3_crossculture_acoustic_fixed.py:170-171
        text_body_offset=9 if args.lang == "ja" else 8
    """
    return 9 if lang == "ja" else 8


def text_body_token_len(tokenizer, text):
    """generate に渡る本文トークン数 L_text.

    `_build_assistant_text`（qwen3_tts_model.py:269-270）の書式を符号化して
    `input_id[:, 3:-5]` の長さを数える（modeling_qwen3_tts.py:2197, 2215）。
    """
    ids = tokenizer(HEAD + text + TAIL, add_special_tokens=False)["input_ids"]
    return len(ids) - 3 - 5


def expected_prefill_len(tokenizer, text, derived_off, non_streaming):
    """prefill の系列長（hook が最初に見る seq_len）の期待値.

    **モデル側の layout で決まる量**なので、必ず config から導いた
    `derived_off` を渡す（`--compat legacy` で使う受け側 offset ではない）。

    non_streaming=True : role 3 + prefix + 本文 L_text + tts_eos + codec_bos
                         = derived_off + L_text + 2   (modeling:2216-2233)
    non_streaming=False: 本文は先頭 1 トークンだけ        (modeling:2195)
                         = derived_off + 1
    """
    if non_streaming:
        return derived_off + text_body_token_len(tokenizer, text) + 2
    return derived_off + 1


# --- スパン解決（splice_multi の関数を使う） --------------------------------

def resolve_qwen3_spans(tokenizer, text, surfaces, char_starts, text_body_offset,
                        roundtrip="strict",
                        span_extension=False, peer_texts=None,
                        peer_surfaces=None, peer_char_starts=None, side="a"):
    """k 個の表層形を「その layout の絶対 index（半開区間）」に解決する.

    `resolve_spans_with_cursor` に **HEAD 付きの文字列**を渡すのが要点。
    既存の `find_entity_token_positions`（eval_qwen3_injection.py:290-341）も
    `_HEAD + text` を符号化して offset_mapping で覆うトークンを拾っているので、
    同じトークン空間・同じ判定式（`a < end and b > start`）になる。
    そのあと `tok_to_seq`（同 335-341: `offset + tok - 3`）を両端に当てる。
    アフィン変換なので半開区間はそのまま半開区間で保たれ、
    legacy の閉区間 [s, e] は本関数の [s, e+1) に一致する。

    `roundtrip="strict"`（既定）は往復一致しない文を丸ごと落とす
    （T5Gemma 版と同じ方針。「末尾 N トークンを entity とみなす」種類の
    fallback は用意しない）。`roundtrip="warn"` は落とさず
    `info["roundtrip_bad"]` に理由を残すだけにする。既存の単一 entity 経路は
    往復一致を見ていないので、md5 等価性試験では "warn" を使う。

    返り値: (spans, info) / 失敗時 (None, {"reason": ...})
    """
    full = HEAD + text
    base = len(HEAD)
    cs = None
    if char_starts is not None:
        cs = [None if c is None else base + int(c) for c in char_starts]
    # **最小対称拡張（2026-09-20 追加）。**中国語で助詞「的」が entity に癒着し、
    # 往復確認が `covered='田中的' != surface='田中'` で落ちる対が 16/300 あった。
    # 規則の実装は scripts/projection/span_extend.py の 1 箇所だけで、
    # T5Gemma（generate_multientity_ext.py）と LongCat（longcat_splice.py）が
    # 既に同じものを使っている。ここでは呼ぶだけで書き直していない。
    # **既定は False。**語そのままで往復一致する対には plain=True が返り、
    # 解決されるトークン区間は既定経路と同一になる（既存 284 対は変わらない）。
    pcs = None
    if peer_char_starts is not None:
        pcs = [None if c is None else base + int(c) for c in peer_char_starts]
    pt = None
    if peer_texts is not None:
        pt = [HEAD + t for t in peer_texts]
    spans, info = resolve_spans_with_cursor(
        tokenizer, full, surfaces, add_bos=0, char_starts=cs,
        allow_symmetric_extension=bool(span_extension),
        peer_texts=pt, peer_surfaces=peer_surfaces,
        peer_char_starts=pcs, side=side)
    if spans is None:
        return None, {"reason": f"resolve: {info['reason']}"}
    # **往復確認は拡張後の表層で行う。**拡張が起きた対では要求する表層が
    # 変わっているので、元の語で見ると必ず落ちる
    # （splice_multi.resolve_spans_with_cursor の docstring に明記）。
    # 拡張が起きていない（plain）対では surface は元の語と同じなので、
    # 既定経路と判定が一致する。
    check = list(surfaces)
    for i, e in enumerate(info.get("extension") or []):
        if i < len(check) and e.get("surface"):
            check[i] = e["surface"]
    ok, bad = verify_span_roundtrip(tokenizer, full, spans, check, add_bos=0)
    if not ok:
        if roundtrip == "strict":
            return None, {"reason": f"roundtrip: {bad}"}
        info["roundtrip_bad"] = bad
    out = []
    for i, (t_s, t_e) in enumerate(spans):
        if t_s < 3:
            return None, {"reason": f"span {i} starts inside the role header "
                                    f"(tok {t_s} < 3)"}
        out.append((text_body_offset + t_s - 3, text_body_offset + t_e - 3))
    info["seq_spans"] = [list(x) for x in out]
    return out, info


def adapt_donor_len(donor, n):
    """ドナーのスパン [m, d] を受け側の長さ n に合わせる（長さ不変 splice 用）.

    単一 entity 版（eval_qwen3_crossculture_acoustic_fixed.py:187-191）と
    同じ規則をそのまま書いたもの:
        m == n : そのまま
        m != n : 時間平均して n 個に broadcast
    `.to(h.dtype)` は hook 側で当てる（legacy と同じ順序）。
    """
    if donor.shape[0] == n:
        return donor, "exact"
    return donor.mean(dim=0).unsqueeze(0).expand(n, -1), "mean_broadcast"


def make_splice_hook(s_ranges, donors, probe):
    """層 26 出力に k スパンを一括で差し替える forward hook.

    ガードは単一 entity 版（eval_...fixed.py:179-193）と同一:
      * 出力が 3 階テンソルでない / seq_len <= 1 は素通し（復号ステップ）
      * スパンが prefill の外に出ていたら素通し（legacy の
        `wa_e_inf >= h.shape[1]` を半開区間で書くと `max_end > T`）
      * 1 回だけ発火する（fired ラッチ）
    差し替え自体は `build_spliced_hidden_multi` に任せる。
    """
    max_end = max(e for (_s, e) in s_ranges)

    def hook(module, inp, out):
        if probe["fired"]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.dim() < 3 or h.shape[1] <= 1:
            return out
        if probe["T_prefill"] is None:
            probe["T_prefill"] = int(h.shape[1])
        if max_end > h.shape[1]:
            probe["unreached"] = True
            return out
        src_enc = {"last_hidden_state": h}
        ref_encs = [{"last_hidden_state": d.to(h.dtype).unsqueeze(0)}
                    for d in donors]
        r_ranges = [(0, int(d.shape[0])) for d in donors]
        modified, delta_tok, out_ranges = build_spliced_hidden_multi(
            src_enc, ref_encs, s_ranges, r_ranges)
        probe["fired"] = True
        probe["delta_tok"] = int(delta_tok)
        probe["out_ranges"] = [list(x) for x in out_ranges]
        probe["T_out"] = int(modified.shape[1])
        if isinstance(out, tuple):
            return (modified,) + out[1:]
        return modified

    return hook


# --- dry_run 用のスタブ（CPU 検証専用。本番経路には一切入らない） ----------

_DRY_HID = 8


def local_snapshot(hub_id):
    """HF cache 内の snapshot ディレクトリ（tokenizer を local から読むため）。"""
    hub = os.environ.get("HF_HUB_CACHE") or \
        os.path.join(os.environ["HF_HOME"], "hub")
    d = os.path.join(hub, f"models--{hub_id.replace('/', '--')}", "snapshots")
    if os.path.isdir(d):
        for sn in sorted(os.listdir(d), reverse=True):
            if os.path.exists(os.path.join(d, sn, "config.json")):
                return os.path.join(d, sn)
    return hub_id


class _DryStub:
    """talker / processor の代わり（dry_run では中身を使わない）。"""
    text_projection = None


class _DryConfig:
    """recv_text_body_offset が読む talker_config だけを本物の config.json から。"""

    def __init__(self, snap):
        cfg = json.load(open(os.path.join(snap, "config.json")))
        self.talker_config = type("T", (), cfg.get("talker_config", {}))


def _dry_donor_fns(tokenizer):
    """ドナー側 forward の代わり。層 26 出力を乱数で作る（形だけ本物と同じ）。"""

    def build_text_only_inputs(text, *a, **k):
        return ("dry", text), None

    def get_layer_hidden_states(talker_model, emb, mask, layer):
        text = emb[1]
        # eval_qwen3_injection.build_text_only_inputs の seq_len:
        #   8 + text_len - 3 + 1 + 1 = DONOR_OFFSET + L_text + 2
        T = DONOR_TEXT_BODY_OFFSET + text_body_token_len(tokenizer, text) + 2
        g = torch.Generator().manual_seed(
            int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
        return torch.randn(1, T, _DRY_HID, generator=g)

    return build_text_only_inputs, get_layer_hidden_states


# --- データ -----------------------------------------------------------------

def load_multi_items(pairs_module, n):
    mod = __import__(pairs_module)
    items = mod.build_pairs()
    for it in items:
        assert len(it["spans"]) == 2, \
            f"pair {it['pair_idx']}: this script expects exactly 2 spans"
        assert len(it["text_refs"]) == 2
    return items[:n] if n else items


def load_cc_items(lang, n):
    """単一区間 md5 試験用。既存の単一 entity 本番と同じデータ・同じ順序。"""
    if lang == "ja":
        from ja_cross_culture_names import build_300_pairs
    elif lang == "zh":
        from zh_cross_culture_names import build_300_pairs
    else:
        from en_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()
    items = []
    for i, p in enumerate(pairs[:n] if n else pairs):
        # 既存は text.find(word) で文字位置を決める
        # （eval_qwen3_injection.py:303）。同じ値を渡す。
        items.append({
            "pair_idx": p.get("pair_idx", i),
            "template_id": None,
            "lang": lang,
            "text_a": p["text_a"],
            "text_refs": [p["text_b"]],
            "text_both": p["text_b"],
            "gap_words": None,
            "slot1_at_start": p["text_a"].startswith(p["word_a"]),
            "spans": [{"slot": 1, "word_a": p["word_a"], "word_b": p["word_b"],
                       "char_start_a": p["text_a"].find(p["word_a"]),
                       "char_start_b": p["text_b"].find(p["word_b"])}],
        })
    return items


# --- 本体 -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # **目標文だけを後から足す（2026-09-20）。**既存 4 条件は生成済みで、
    # 再開は対単位なので通常経路では何も作られない。--tgt_only は対単位の
    # 再開を無視し、s<i>_tgt1.wav / s<i>_tgt2.wav が揃っていない対にだけ
    # 目標文を合成する。既存の wav・results.json・checkpoint.json には触れない。
    ap.add_argument("--tgt_only", action="store_true",
                   help="枠ごとの目標文だけを合成する。既存の注入音声は作り直さない")
    ap.add_argument("--mode", default="multi", choices=["multi", "cc_single"],
                    help="multi=2 枠の本番 / cc_single=1 区間の md5 等価性試験")
    ap.add_argument("--pairs_module", default=None,
                    help="multi モードのデータ（既定: <lang>_multientity_300）")
    # zh を 2026-09-18 に追加。根拠:
    #  * LANG_CODE と text_body_offset を公表セル側の実装にそろえた（上記）。
    #  * `scripts/analysis/verify_zh_multientity_cpu.py` に Qwen3 の
    #    トークン化器を足して検証し、**300 件 600 スパンすべて解決**
    #    （584 が plain、16 を最小対称拡張で救出、2 スパンの重なり 0、
    #    無回帰違反 0）。従来通っていたスパンは 1 つも壊れていない。
    ap.add_argument("--lang", required=True, choices=["en", "ja", "zh"],
                    help="中国語のデータは存在しないので対象外")
    ap.add_argument("--n", type=int, default=0, help="0 なら全件")
    ap.add_argument("--injection_layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--compat", default="fixed", choices=["fixed", "legacy"],
                    help="legacy: 既存単一 entity 本番と bit 一致させる"
                         "（受け側 offset=9(ja)/8(en), non_streaming_mode=False）"
                         " / fixed: 両側で同じ規約 + 本文全体を prefill に載せる")
    ap.add_argument("--recv_offset", type=int, default=None,
                    help="受け側 text_body_offset の明示指定（既定は --compat 由来）")
    ap.add_argument("--non_streaming", type=int, default=None, choices=[0, 1],
                    help="generate_voice_clone の non_streaming_mode（既定は --compat 由来）")
    ap.add_argument("--roundtrip", default=None, choices=["strict", "warn"],
                    help="スパンの往復一致が落ちた文の扱い。strict=文ごと skip"
                         "（T5Gemma 版と同じ）/ warn=記録だけして続行（既存の"
                         "単一 entity 経路と同じ）。既定は --compat 由来"
                         "（fixed→strict, legacy→warn）")
    # **中国語の助詞「的」対策（2026-09-20）。**既定は False で、
    # 付けなければ既存の経路とトークン区間が完全に一致する。
    ap.add_argument("--span_extension", action="store_true",
                    help="往復一致しない対に最小対称拡張を試す "
                         "(scripts/projection/span_extend.py の規則。"
                         "語そのままで一致する対は plain=True で既定経路と同一)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--utmos", action="store_true",
                    help="UTMOS22 も測る（既定 off。指示された条件以外は回さない）")
    ap.add_argument("--preload_scorers", action="store_true",
                    help="既存スクリプトと同じ順序で採点器を先に load する"
                         "（md5 が合わないときの切り分け用）")
    ap.add_argument("--dry_run", action="store_true",
                    help="GPU を使わずスパン解決・hook・出力形式だけ通す"
                         "（tokenizer だけ local から読み、hidden は乱数）")
    args = ap.parse_args()

    if args.pairs_module is None:
        args.pairs_module = f"{args.lang}_multientity_300"
    if args.output_dir is None:
        args.output_dir = (f"eval_results/multientity_qwen3_{args.lang}_300"
                           if args.mode == "multi"
                           else f"eval_results/_md5_qwen3_{args.lang}_k1")
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu" if args.dry_run else (
        "cuda" if torch.cuda.is_available() else "cpu")
    seed_all(SEED)

    non_streaming = bool(args.non_streaming) if args.non_streaming is not None \
        else (args.compat == "fixed")
    roundtrip = args.roundtrip or ("strict" if args.compat == "fixed" else "warn")

    conds = CONDITIONS_MULTI if args.mode == "multi" else CONDITIONS_CC
    cond_slots = COND_SLOTS_MULTI if args.mode == "multi" else COND_SLOTS_CC
    if args.tgt_only:
        # 目標文だけを後から足す。注入はしないので slots は空。
        conds = list(TGT_SLOT_CONDS)
        cond_slots = {c: [] for c in conds}
    items = (load_multi_items(args.pairs_module, args.n) if args.mode == "multi"
             else load_cc_items(args.lang, args.n))
    n_span = 2 if args.mode == "multi" else 1
    print(f"[Info] mode={args.mode} lang={args.lang} n_items={len(items)} "
          f"k={n_span} layer={args.injection_layer} compat={args.compat} "
          f"non_streaming={non_streaming} roundtrip={roundtrip}", flush=True)

    if args.preload_scorers:
        # eval_qwen3_crossculture_acoustic_fixed.py:82-83 と同じ順序
        from eval_table3_acoustic_common import AcousticFeatureExtractor
        from forced_alignment_utils import WhisperAligner
        _feat = AcousticFeatureExtractor(device)
        _aligner = WhisperAligner(device=device)
        print("[Info] scorers preloaded (md5 compat)", flush=True)

    if args.dry_run:
        # CPU 検証用。tokenizer だけ本物を local から読み、hidden は乱数にする。
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            local_snapshot(MODEL_ID), local_files_only=True,
            fix_mistral_regex=True)   # qwen3_tts_model.py:120 と同じ引数
        config = _DryConfig(local_snapshot(MODEL_ID))
        wrapper = None
        talker = talker_model = processor = _DryStub()
    else:
        print("[Info] Loading Qwen3-TTS ...", flush=True)
        from qwen_tts import Qwen3TTSModel
        wrapper = Qwen3TTSModel.from_pretrained(
            MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa")
        wrapper.eval() if hasattr(wrapper, "eval") else wrapper.model.eval()
        talker = wrapper.model.talker
        talker_model = talker.model
        processor = wrapper.processor
        config = wrapper.model.config
        tokenizer = processor.tokenizer
    assert getattr(tokenizer, "is_fast", False), (
        "a fast tokenizer is required: span resolution uses offset_mapping and "
        "there is deliberately no character-length fallback")

    derived_off = recv_text_body_offset(config, LANG_CODE[args.lang],
                                        x_vector_only=True)
    legacy_off = legacy_recv_text_body_offset(args.lang)
    recv_off = args.recv_offset if args.recv_offset is not None else (
        legacy_off if args.compat == "legacy" else derived_off)
    print(f"[Info] text_body_offset: donor={DONOR_TEXT_BODY_OFFSET} "
          f"recv={recv_off} (derived={derived_off}, legacy={legacy_off})",
          flush=True)
    if recv_off != derived_off:
        print(f"[Warn] 受け側 offset {recv_off} は layout から導いた "
              f"{derived_off} と違う。既存の非対称を再現する設定である。",
              flush=True)

    # eval_qwen3_crossculture_acoustic_fixed.py:120-126 と同一
    ref_audio, ref_text = REFS[args.lang]
    ref_audio = os.path.join(PROJECT_ROOT, ref_audio)
    prompt_items = None
    if not args.dry_run:
        print("[Info] Building voice clone prompt (x_vector_only_mode=True) ...",
              flush=True)
        prompt_items = wrapper.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=True)

    if args.dry_run:
        build_text_only_inputs, get_layer_hidden_states = _dry_donor_fns(
            tokenizer)
    else:
        # eval_qwen3_injection から再利用（HF_HOME を /mnt 固定で上書きするので戻す）
        from eval_qwen3_injection import (build_text_only_inputs,
                                          get_layer_hidden_states)
        os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
        os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface",
                                                  "hub")

    utmos = None
    if args.utmos:
        print("[Info] Loading UTMOS22 ...", flush=True)
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(device).eval()

    import torchaudio

    def gen_audio_dry(text, hook=None):
        """GPU 無しで hook 呼び出しまでを通す（波形は無音の固定長）。

        hook の呼び方は register_forward_hook が呼ぶのと同じ
        `hook(module, input, output)`。出力が tuple ではない層もあるので
        テンソル直渡しで呼ぶ。
        """
        seed_all(SEED)
        fp = rng_fingerprint()
        T = expected_prefill_len(tokenizer, text, derived_off, non_streaming)
        h = torch.randn(1, T, _DRY_HID)
        if hook is not None:
            got = hook(None, None, h)
            assert got.shape[1] == T, f"dry_run: T changed {got.shape[1]} != {T}"
        return torch.zeros(int(0.5 * 24000)), 24000, fp

    def gen_audio(text, hook=None):
        """eval_qwen3_crossculture_acoustic_fixed.py:130-147 と同一の生成.

        seed は **generate の直前に毎回戻す**。したがって 4 条件は同じ RNG 状態
        から始まり、条件ごとの乱数消費数の違いが他の条件に漏れない。
        """
        seed_all(SEED)
        fp = rng_fingerprint()
        h_handle = None
        if hook is not None:
            h_handle = talker_model.layers[args.injection_layer] \
                .register_forward_hook(hook)
        try:
            kw = {}
            if non_streaming:
                kw["non_streaming_mode"] = True
            with torch.no_grad():
                wavs, sr = wrapper.generate_voice_clone(
                    text=text, language=LANG_CODE[args.lang],
                    voice_clone_prompt=prompt_items,
                    max_new_tokens=MAX_NEW_TOKENS, **kw)
        finally:
            if h_handle is not None:
                h_handle.remove()
        wav = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(wav, np.ndarray):
            a = torch.tensor(wav).float()
        elif isinstance(wav, torch.Tensor):
            a = wav.cpu().float()
        else:
            a = torch.tensor(wav).float()
        if a.dim() > 1:
            a = a.squeeze()
        return a, sr, fp

    if args.dry_run:
        gen_audio = gen_audio_dry

    def save_wav(a, sr, path):
        """eval_qwen3_crossculture_acoustic_fixed.py:201-203 と同一。"""
        aa = a.unsqueeze(0) if a.dim() == 1 else a
        torchaudio.save(path, aa.cpu(), sr, encoding="PCM_S", bits_per_sample=16)

    def score_utmos(a, sr):
        if utmos is None:
            return None
        w = a.unsqueeze(0) if a.dim() == 1 else a
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        with torch.no_grad():
            return round(float(utmos(w.to(device), 16000).mean()), 4)

    rows, skipped = [], []
    # --tgt_only は別の帳面を使う（既存 4 条件の再開記録を壊さないため）。
    ckpt = os.path.join(args.output_dir,
                        "checkpoint_tgt.json" if args.tgt_only
                        else "checkpoint.json")
    if os.path.exists(ckpt):
        st = json.load(open(ckpt))
        rows = st.get("rows", [])
        skipped = st.get("skipped", [])
        bad = [r["pair_idx"] for r in rows if r.get("mode") != "splice_multi_qwen3"]
        assert not bad, (f"checkpoint mode mismatch in {ckpt}: pairs {bad[:5]} "
                         f"were not produced by this script; use a separate "
                         f"--output_dir")
        print(f"[Info] resume: {len(rows)} rows, {len(skipped)} skipped",
              flush=True)
    done = {r["pair_idx"] for r in rows} | {s["pair_idx"] for s in skipped}
    if args.tgt_only:
        # 対単位の済み印を捨て、**目標音声が両方そろった対だけ**を済みとする。
        rows, skipped = [], []
        done = {it["pair_idx"] for it in items
                if all(os.path.exists(os.path.join(
                    args.output_dir, "s%04d_%s.wav" % (it["pair_idx"], c)))
                    for c in TGT_SLOT_CONDS)}
        print("[Info] tgt_only: 目標文が既にある対 %d / 作る対 %d"
              % (len(done), len(items) - len(done)), flush=True)
    t0 = time.time()

    for it in items:
        pidx = it["pair_idx"]
        if pidx in done:
            continue
        text_a = it["text_a"]
        refs = it["text_refs"]
        spans = it["spans"]

        # **--tgt_only ではスパンを解決しない（2026-09-20 修正）。**
        # 目標文は注入せずそのまま読ませるだけなので実体語の区間は要らない。
        # 解決を走らせていたため、助詞が癒着する対で落ち、日本語 8 対の
        # 目標音声が作られなかった。重なり検査や長さの検算も同じ理由で飛ばす。
        if args.tgt_only:
            s_ranges, d_ranges = [], []
            adapted = []
            row = {"pair_idx": pidx, "mode": "tgt_only", "lang": args.lang,
                   "text_a": text_a, "text_refs": list(refs),
                   "text_both": it.get("text_both"),
                   "words_a": [sp["word_a"] for sp in spans],
                   "words_b": [sp["word_b"] for sp in spans],
                   "T_src": None, "conds": {}}
            ok = True
            for cond in conds:
                gen_text = refs[TGT_SLOT_CONDS[cond]]
                wav_path = os.path.join(args.output_dir,
                                        f"s{pidx:04d}_{cond}.wav")
                try:
                    a_, sr_, fp_ = gen_audio(gen_text, hook=None)
                    save_wav(a_, sr_, wav_path)
                except Exception as e:
                    print(f"  [{pidx}] {cond} synth fail: {str(e)[:200]}",
                          flush=True)
                    skipped.append({"pair_idx": pidx,
                                    "reason": f"{cond} synth: {e}"})
                    ok = False
                    break
                import soundfile as _sf
                _i = _sf.info(wav_path)
                row["conds"][cond] = {
                    "wav": os.path.basename(wav_path),
                    "target_text": gen_text,
                    "dur_sec": round(_i.frames / float(_i.samplerate), 4),
                    "rng_state_md5": fp_, "fired": False, "delta_tok": 0,
                    "T_out": None, "out_ranges": [], "utmos": None,
                    # 集計が参照する項目。目標文には注入が無いので常に偽/0。
                    "unreached": False, "delta_tok_abs": 0, "pesq": None,
                }
            done.add(pidx)
            if not ok:
                continue
            row["total_position_drift"] = None
            rows.append(row)
            if len(rows) % 5 == 0:
                print(f"  [{len(rows)}] idx={pidx} tgt1/tgt2 作成", flush=True)
            if len(rows) % args.ckpt_every == 0:
                with open(ckpt, "w") as f:
                    json.dump({"rows": rows, "skipped": skipped}, f,
                              ensure_ascii=False, indent=1)
            continue

        # --- 1. スパンの解決（src 側は全スパンまとめて 1 回） -------------
        s_ranges, s_info = resolve_qwen3_spans(
            tokenizer, text_a, [sp["word_a"] for sp in spans],
            [sp.get("char_start_a") for sp in spans], recv_off,
            roundtrip=roundtrip,
            span_extension=args.span_extension,
            peer_texts=refs, peer_surfaces=[sp["word_b"] for sp in spans],
            peer_char_starts=[sp.get("char_start_b") for sp in spans],
            side="a")
        if s_ranges is None:
            print(f"  [{pidx}] SKIP src {s_info['reason']}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"src {s_info['reason']}"})
            done.add(pidx)
            continue

        # --- 2. ドナー側はスパンごとに別の forward -------------------------
        d_ranges, d_fail = [], None
        for i, sp in enumerate(spans):
            r, r_info = resolve_qwen3_spans(
                tokenizer, refs[i], [sp["word_b"]], [sp.get("char_start_b")],
                DONOR_TEXT_BODY_OFFSET, roundtrip=roundtrip,
                span_extension=args.span_extension,
                peer_texts=[text_a], peer_surfaces=[sp["word_a"]],
                peer_char_starts=[sp.get("char_start_a")], side="b")
            if r is None:
                d_fail = f"ref[{i}] {r_info['reason']}"
                break
            d_ranges.append(r[0])
        if d_fail is not None:
            print(f"  [{pidx}] SKIP {d_fail}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": d_fail})
            done.add(pidx)
            continue

        # --- 3. 重なり検査（生成前） ---------------------------------------
        n_list = [e - s for (s, e) in s_ranges]
        m_list = [e - s for (s, e) in d_ranges]
        if n_span == 2 and s_ranges[0][1] > s_ranges[1][0]:
            reason = (f"src spans overlap: {s_ranges[0]} {s_ranges[1]} "
                      f"({[sp['word_a'] for sp in spans]})")
            print(f"  [{pidx}] SKIP {reason}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": reason})
            done.add(pidx)
            continue
        T_exp = expected_prefill_len(tokenizer, text_a, derived_off,
                                     non_streaming)
        try:
            # splice_multi 側の検証をそのまま使う（個数・空・重なり）。
            # `--compat legacy` では prefill が短く、スパンが prefill の外に
            # 出ることが正常にありうる（既存も skip せず「発火しないだけ」）。
            # ここでは重なりと空区間を見たいので、src_len には
            # 「スパンが収まる長さ」を渡して範囲検査だけ無効化する。
            # prefill 外かどうかは hook 側の `max_end > h.shape[1]` が見る。
            src_len_for_check = max(T_exp, max(e for (_s, e) in s_ranges))
            check_span_sets(src_len_for_check, n_list, s_ranges,
                            [(0, n) for n in n_list])
        except AssertionError as e:
            print(f"  [{pidx}] SKIP check_span_sets: {str(e)[:160]}", flush=True)
            skipped.append({"pair_idx": pidx,
                            "reason": f"check_span_sets: {e}"})
            done.add(pidx)
            continue

        # --- 4. ドナー hidden の取り出し（層 26 出力） ---------------------
        try:
            donors_full = []
            for i in range(n_span):
                emb, mask = build_text_only_inputs(
                    refs[i], talker, talker.text_projection, processor, config,
                    device)
                hid = get_layer_hidden_states(talker_model, emb, mask,
                                              args.injection_layer)
                d_s, d_e = d_ranges[i]
                assert d_e <= hid.shape[1], (
                    f"donor span {d_ranges[i]} outside donor sequence "
                    f"{hid.shape[1]}")
                donors_full.append(hid[0, d_s:d_e].clone())
                del hid
        except Exception as e:
            print(f"  [{pidx}] donor fail: {str(e)[:160]}", flush=True)
            skipped.append({"pair_idx": pidx, "reason": f"donor: {e}"})
            done.add(pidx)
            continue

        adapted, how = [], []
        for i in range(n_span):
            d, mode_i = adapt_donor_len(donors_full[i], n_list[i])
            adapted.append(d)
            how.append(mode_i)

        span_tok = {
            "src": [list(x) for x in s_ranges],
            "donor": [list(x) for x in d_ranges],
            "n": n_list, "m": m_list,
            "delta_per_span_raw": span_deltas(s_ranges, d_ranges)[0],
            "len_adapt": how,
            "gap_tokens": (s_ranges[1][0] - s_ranges[0][1]) if n_span == 2 else None,
            "recv_text_body_offset": recv_off,
            "donor_text_body_offset": DONOR_TEXT_BODY_OFFSET,
            "src_resolve_via": [p["via"] for p in s_info["per_span"]],
            "src_roundtrip_bad": s_info.get("roundtrip_bad"),
            "T_src_expected": T_exp,
        }

        row = {
            "pair_idx": pidx, "mode": "splice_multi_qwen3",
            "method": "qwen3", "compat": args.compat,
            "non_streaming_mode": non_streaming,
            "injection_layer": args.injection_layer,
            "template_id": it.get("template_id"), "lang": args.lang,
            "text_a": text_a, "text_refs": refs, "text_both": it.get("text_both"),
            "gap_words": it.get("gap_words"),
            "slot1_at_start": it.get("slot1_at_start"),
            "repeat_surface": it.get("repeat_surface", False),
            "words_a": [sp["word_a"] for sp in spans],
            "words_b": [sp["word_b"] for sp in spans],
            "span_tok": span_tok,
            "T_src": T_exp,
            "conds": {},
        }

        ok = True
        for cond in conds:
            slots = cond_slots[cond]
            probe = {"fired": False, "T_prefill": None, "T_out": None,
                     "delta_tok": 0, "out_ranges": [], "unreached": False}
            hook = None
            if slots:
                hook = make_splice_hook([s_ranges[i] for i in slots],
                                        [adapted[i] for i in slots], probe)
            # tgt 条件（cc_single）だけは text_b をそのまま読ませる。
            # tgt1 / tgt2 は枠ごとの目標文（--tgt_only）。
            if cond in TGT_SLOT_CONDS:
                gen_text = refs[TGT_SLOT_CONDS[cond]]
            else:
                gen_text = refs[0] if cond == "tgt" else text_a
            try:
                a, sr, fp = gen_audio(gen_text, hook=hook)
            except Exception as e:
                print(f"  [{pidx}] {cond} synth fail: {str(e)[:200]}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} synth: {e}"})
                ok = False
                break
            wav_path = os.path.join(args.output_dir, f"s{pidx:04d}_{cond}.wav")
            try:
                save_wav(a, sr, wav_path)
            except Exception as e:
                print(f"  [{pidx}] {cond} save fail: {str(e)[:200]}", flush=True)
                skipped.append({"pair_idx": pidx, "reason": f"{cond} save: {e}"})
                ok = False
                break
            T_out = probe["T_out"] if probe["T_out"] is not None else T_exp
            if probe["T_prefill"] is not None and probe["T_prefill"] != T_exp:
                # layout の導出（recv_text_body_offset / expected_prefill_len）が
                # 実際の prefill 長と合っていない。後付け補正はせず記録する。
                print(f"  [{pidx}] {cond} [Warn] prefill 長が想定と違う: "
                      f"observed={probe['T_prefill']} expected={T_exp}",
                      flush=True)
            row["conds"][cond] = {
                "wav": os.path.basename(wav_path),
                "delta_tok": int(probe["delta_tok"] or 0),
                "T_out": int(T_out),
                "out_ranges": probe["out_ranges"] if probe["out_ranges"]
                else [list(s_ranges[i]) for i in slots],
                "target_text": gen_text,
                "dur_sec": round(float(a.shape[-1]) / sr, 4),
                "sr": int(sr),
                "fired": bool(probe["fired"]),
                "unreached": bool(probe["unreached"]),
                "T_prefill": probe["T_prefill"],
                "T_prefill_matches_expected": (
                    None if probe["T_prefill"] is None
                    else bool(probe["T_prefill"] == T_exp)),
                "rng_state_md5": fp,
                "utmos": score_utmos(a, sr),
            }

        done.add(pidx)
        if not ok:
            continue
        # 長さ不変 splice なので総ずれは常に 0。記録して確認できるようにする。
        # --tgt_only では注入条件を作っていないので、総ずれは測る対象が無い。
        if args.tgt_only:
            row["total_position_drift"] = None
        else:
            key12 = "inj12" if args.mode == "multi" else "inj"
            row["total_position_drift"] = row["conds"][key12]["delta_tok"]
        fps = {c["rng_state_md5"] for c in row["conds"].values()}
        row["rng_state_identical"] = (len(fps) == 1)
        rows.append(row)

        if len(rows) % 5 == 0 or len(rows) == 1:
            c = row["conds"]
            fired = "/".join("1" if c[k]["fired"] else "0" for k in conds)
            print(f"  [{len(rows)}/{len(items)}] idx={pidx} "
                  f"n={n_list} m={m_list} gap={span_tok['gap_tokens']} "
                  f"T={row['T_src']} fired({'/'.join(conds)})={fired} "
                  f"drift={row['total_position_drift']} "
                  f"rng_same={row['rng_state_identical']}", flush=True)
        if len(rows) % args.ckpt_every == 0:
            with open(ckpt, "w") as f:
                json.dump({"rows": rows, "skipped": skipped}, f,
                          ensure_ascii=False, indent=1)

    with open(ckpt, "w") as f:
        json.dump({"rows": rows, "skipped": skipped}, f, ensure_ascii=False,
                  indent=1)

    def _mean(key, field):
        v = [r["conds"][key][field] for r in rows
             if r["conds"].get(key, {}).get(field) is not None]
        return round(float(np.mean(v)), 4) if v else None

    summary = {
        "method": "qwen3", "mode": args.mode, "lang": args.lang,
        "pairs_module": args.pairs_module if args.mode == "multi"
        else f"{args.lang}_cross_culture_names",
        "compat": args.compat, "non_streaming_mode": non_streaming,
        "roundtrip": roundtrip,
        "injection_layer": args.injection_layer,
        "recv_text_body_offset": recv_off,
        "recv_text_body_offset_derived": derived_off,
        "recv_text_body_offset_legacy": legacy_off,
        "donor_text_body_offset": DONOR_TEXT_BODY_OFFSET,
        "n_items": len(items), "n_generated": len(rows),
        "n_skipped": len(skipped),
        "reference_wav": ref_audio,
        "fired_rate": {c: round(sum(1 for r in rows if r["conds"][c]["fired"])
                                / max(1, len(rows)), 4) for c in conds},
        "unreached_rate": {
            c: round(sum(1 for r in rows if r["conds"][c]["unreached"])
                     / max(1, len(rows)), 4) for c in conds},
        "len_adapt_mean_broadcast": sum(
            1 for r in rows if "mean_broadcast" in r["span_tok"]["len_adapt"]),
        "mean_dur_sec": {c: _mean(c, "dur_sec") for c in conds},
        "utmos": {c: _mean(c, "utmos") for c in conds},
        "mean_abs_position_drift": (None if args.tgt_only else (
            round(float(np.mean(
                [abs(r["total_position_drift"]) for r in rows])), 3)
            if rows else None)),
        "rng_state_identical_all": all(r["rng_state_identical"] for r in rows)
        if rows else None,
        "n_prefill_len_mismatch": sum(
            1 for r in rows for c in r["conds"].values()
            if c.get("T_prefill_matches_expected") is False),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    out_name = "results_tgt.json" if args.tgt_only else "results.json"
    with open(os.path.join(args.output_dir, out_name), "w") as f:
        json.dump({"summary": summary, "rows": rows, "skipped": skipped}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n=== multi-entity qwen3 {args.mode} {args.lang} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if skipped:
        print(f"[Warn] {len(skipped)} items skipped; see results.json['skipped']")


if __name__ == "__main__":
    main()
