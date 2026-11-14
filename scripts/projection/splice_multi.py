#!/usr/bin/env python3
"""複数エンティティ同時 splice（k>=1 スパンの一括 concat）.

`splice_clone.py` の 1 スパン版を k スパンに一般化したもの。**既存ファイルは
1 行も変更していない。** k=1 に縮退させると `build_spliced_hidden` と
bit 一致する（`tests/test_splice_multi.py::test_k1_bit_identical_to_single`）。

設計の要点
----------
1. **index のずれを一括 concat で回避する。**
   既存配列を書き換えず、src の「元の」index だけを読んで新しい配列を 1 パスで
   組み立てる。torch.cat は 1 回だけ。

       out = cat([ src[:s1_s], ref1[r1_s:r1_e],
                   src[s1_e:s2_s], ref2[r2_s:r2_e],
                   src[s2_e:] ])

   スパンを順に in-place 置換していくと、1 個目で長さが変わった時点で
   2 個目の s_range が指す位置がずれる。一括 concat はこの問題を構造的に
   持たない。

2. **out_ranges を返す。** 置換後系列のどこにスパン i があるかは
   kappa の再測定とスパン別 PIB に必要になる。返り値は「入力で与えた順」で、
   内部のソート順ではない。

3. **重なりは assert で落とす。** 隣接（s1_e == s2_s）は長さ 0 の gap として
   自然に処理されるので許す。重なりは黙って壊れるので必ず落とす。

4. **ref はスパンごとに別の forward。** `ref_encs[i]` は
   「対象語 i **だけ**を目標形に置換した文」の encoder 出力である。
   両方置換した 1 本の ref から 2 スパンを取ると、干渉の原因
   （splice そのもの vs. ref 側の文脈が既に汚れている）が分離できない。

5. **position の総ずれ。** `models/t5gemma.py:_build_position_ids` は
   `pos / (T-1) * progress_scale` という長さ正規化された進行度ランプなので、
   T が変わると全 token の position 値が動く。総ずれ
   `delta_tok = sum_i (m_i - n_i)`（n_i=src スパン長, m_i=ref スパン長）を
   返すので、呼び出し側は必ず記録すること。

6. **中間層 splice は非対応。** `scripts/projection/eval_layer_sweep.py`
   の `run_injection_at_layer` は長さ固定（db_len > kanji_len なら先頭を切り、
   短ければ最終 token を繰り返して埋める）で、長さ可変 splice を中間層では
   行っていない。k>=2 を中間層でやる設計は未検証なので
   `assert_single_span_for_midlayer()` で明示的に NotImplementedError にする。

関連: scripts/projection/splice_clone.py（1 スパン版・正式実装）
      scripts/multientity/README.md
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from splice_clone import make_length_matched_target  # noqa: E402


# --- スパン集合の検証 -------------------------------------------------------

def check_span_sets(src_len, ref_lens, s_ranges, r_ranges):
    """スパン集合の健全性を検証し、s_s 昇順の並び替え index を返す.

    Raises AssertionError on: 個数不一致 / 空スパン / 範囲外 / **重なり**。
    隣接（s_ranges[i][1] == s_ranges[j][0]）は長さ 0 の gap として許す。
    """
    k = len(s_ranges)
    assert k >= 1, "s_ranges is empty"
    assert len(r_ranges) == k, f"len(r_ranges)={len(r_ranges)} != len(s_ranges)={k}"
    assert len(ref_lens) == k, f"len(ref_encs)={len(ref_lens)} != len(s_ranges)={k}"

    for i, (s_s, s_e) in enumerate(s_ranges):
        assert isinstance(s_s, int) and isinstance(s_e, int), \
            f"span {i}: s_range must be int, got {(s_s, s_e)}"
        assert 0 <= s_s < s_e <= src_len, \
            f"span {i}: src range {(s_s, s_e)} out of [0, {src_len}] or empty"
    for i, (r_s, r_e) in enumerate(r_ranges):
        assert isinstance(r_s, int) and isinstance(r_e, int), \
            f"span {i}: r_range must be int, got {(r_s, r_e)}"
        assert 0 <= r_s < r_e <= ref_lens[i], \
            f"span {i}: ref range {(r_s, r_e)} out of [0, {ref_lens[i]}] or empty"

    order = sorted(range(k), key=lambda i: s_ranges[i][0])
    for a, b in zip(order, order[1:]):
        assert s_ranges[a][1] <= s_ranges[b][0], (
            f"src spans overlap: span {a}={tuple(s_ranges[a])} and "
            f"span {b}={tuple(s_ranges[b])} (adjacent is allowed, overlap is not)")
    return order


def span_deltas(s_ranges, r_ranges):
    """各スパンの Δtok = m_i - n_i（入力順）と総和を返す."""
    d = [(r_e - r_s) - (s_e - s_s)
         for (s_s, s_e), (r_s, r_e) in zip(s_ranges, r_ranges)]
    return d, sum(d)


# --- 本体 -------------------------------------------------------------------

def build_spliced_hidden_multi(src_enc, ref_encs, s_ranges, r_ranges):
    """k スパンを一括 concat で差し替えた encoder hidden state を返す.

    引数
    ----
    src_enc   : dict。`last_hidden_state` [1, T_src, d] を持つ。
    ref_encs  : dict の list（長さ k）。`ref_encs[i]` はスパン i 専用の
                「対象語 i だけを置換した文」の encoder 出力。
    s_ranges  : [(s_s, s_e)] × k。フル系列（bos? + prefix + x_sep + target + eos?）
                内の**絶対** index。呼び出し側が target_start_in_full を加算済み
                であること。
    r_ranges  : [(r_s, r_e)] × k。`ref_encs[i]` のフル系列内の絶対 index。

    返り値
    ------
    (modified, delta_tok, out_ranges)
      modified   : [1, T_out, d]、T_out = T_src + delta_tok
      delta_tok  : sum_i (m_i - n_i)。position ランプの総ずれ。
      out_ranges : [(o_s, o_e)] × k。**入力で与えた順**。置換後系列における
                   スパン i の位置。o_e - o_s == m_i。
    """
    src_h = src_enc["last_hidden_state"]
    ref_hs = [e["last_hidden_state"] for e in ref_encs]
    assert src_h.dim() == 3 and src_h.shape[0] == 1, \
        f"src last_hidden_state must be [1, T, d], got {tuple(src_h.shape)}"
    for i, rh in enumerate(ref_hs):
        assert rh.dim() == 3 and rh.shape[0] == 1, \
            f"ref[{i}] last_hidden_state must be [1, T, d], got {tuple(rh.shape)}"
        assert rh.shape[2] == src_h.shape[2], \
            f"ref[{i}] hidden dim {rh.shape[2]} != src {src_h.shape[2]}"

    src_len = src_h.shape[1]
    order = check_span_sets(src_len, [rh.shape[1] for rh in ref_hs],
                            s_ranges, r_ranges)

    pieces = []
    out_ranges = [None] * len(s_ranges)
    cur = 0      # src 側の読み出しカーソル
    out_pos = 0  # 出力側の書き込み位置
    for i in order:
        s_s, s_e = s_ranges[i]
        r_s, r_e = r_ranges[i]
        pieces.append(src_h[0, cur:s_s])       # gap（先頭スパンでは prefix）
        out_pos += s_s - cur
        pieces.append(ref_hs[i][0, r_s:r_e])   # 置換スパン
        out_ranges[i] = (out_pos, out_pos + (r_e - r_s))
        out_pos += r_e - r_s
        cur = s_e
    pieces.append(src_h[0, cur:])              # suffix
    out_pos += src_len - cur

    modified = torch.cat(pieces, dim=0).unsqueeze(0)

    _, delta_tok = span_deltas(s_ranges, r_ranges)
    assert modified.shape[1] == src_len + delta_tok, (
        f"splice length mismatch: T_out={modified.shape[1]} T_src={src_len} "
        f"delta={delta_tok}")
    assert out_pos == modified.shape[1], (
        f"out cursor {out_pos} != T_out {modified.shape[1]}")
    for i, (o_s, o_e) in enumerate(out_ranges):
        assert 0 <= o_s < o_e <= modified.shape[1], \
            f"out_range[{i}]={(o_s, o_e)} outside [0, {modified.shape[1]}]"
        assert o_e - o_s == r_ranges[i][1] - r_ranges[i][0], \
            f"out_range[{i}] length != ref span length"
    return modified, delta_tok, out_ranges


def prepare_clone_splice_multi(text_tok, src_enc, ref_encs, s_ranges, r_ranges,
                               lang, duration_source_text, natural_candidates=()):
    """k スパン版の clone-splice 準備（`prepare_clone_splice` の一般化）.

    返り値: (modified, inj_target, inj_dur_text, delta_tok, out_ranges)
      - delta_tok == 0: inj_target=None / inj_dur_text=None
        （呼び出し側は従来どおり text_a で合成する。k=1 の既存挙動と同一。
        k>=2 では個々のスパンが不等長でも総和が 0 になることがあり、その場合も
        input_ids 長は変わらないのでダミーは不要。）
      - delta_tok != 0: inj_target=厳密長一致ダミー、
        inj_dur_text=duration_source_text（Δduration=0.00s 維持用）

    `duration_source_text` は必ず text_a を渡すこと。外すと MCD が壊れる
    （2026-07-22 plan Q1 決定）。
    """
    modified, delta_tok, out_ranges = build_spliced_hidden_multi(
        src_enc, ref_encs, s_ranges, r_ranges)
    if delta_tok == 0:
        return modified, None, None, 0, out_ranges
    needed = len(src_enc["target_tokens"]) + delta_tok
    inj_target = make_length_matched_target(text_tok, needed, lang,
                                            natural_candidates)
    return modified, inj_target, duration_source_text, delta_tok, out_ranges


def assert_single_span_for_midlayer(s_ranges, where="mid-layer injection"):
    """中間層 injection 経路のガード.

    `scripts/projection/eval_layer_sweep.py:run_injection_at_layer` は
    長さ固定（切り詰め / 最終 token 繰り返し）で、長さ可変 splice を
    中間層では行っていない。複数スパンを中間層でやる設計は未検証なので、
    k>=2 は明示的に落とす。黙って 1 スパンだけ処理する fallback は禁止。
    """
    if len(s_ranges) > 1:
        raise NotImplementedError(
            f"{where}: multi-span splice is not implemented for intermediate "
            f"layers (got k={len(s_ranges)} spans). The mid-layer path "
            f"(eval_layer_sweep.py:run_injection_at_layer) is length-preserving "
            f"(truncate / last-token repeat), so variable-length multi-span "
            f"splice there is unverified. Use the encoder-output path "
            f"(prepare_clone_splice_multi) instead.")


# --- 最小対称拡張（既定 OFF・追加のみ） -----------------------------------
#
# 2026-09-18 ユーザ承認の「最小対称拡張」を `resolve_spans_with_cursor` から
# 使えるようにするための配線。**既定は現状維持**
# （`allow_symmetric_extension=False`）。False のときこの節の関数は 1 度も
# 呼ばれないので、既存の呼び出し側（longcat / llasa / kiseki / irodori /
# csm / xtts / bark / qwen3 / validate_multientity / generate_multientity）の
# 挙動は 1 ビットも変わらない。
#
# 規則の実装は `scripts/projection/span_extend.py` に 1 箇所だけある。
# ここでは**その関数を呼ぶだけ**で、規則を書き直していない。
# import も関数の中の遅延 import にしてある（span_extend が壊れていても
# 既定経路の import に影響しないように）。


def _occurrence_index(text, surface, char_start):
    """`char_start` が `surface` の何番目の出現かを返す.

    `span_extend.minimal_symmetric_surface` / `covered_surface` の occurrence は
    「`find` を 1 文字ずつ進めた回数」なので、その数え方に合わせる
    （`str.count` は重なりを数えないので使わない）。
    `char_start` が None、またはその位置が `surface` でない場合は 0
    （= 最初の出現）を返す。
    """
    if char_start is None:
        return 0
    cs = int(char_start)
    if cs < 0 or text[cs:cs + len(surface)] != surface:
        return 0
    occ = 0
    p = text.find(surface)
    while 0 <= p < cs:
        occ += 1
        p = text.find(surface, p + 1)
    return occ


def _symmetric_surfaces(text_tok, full_text, surfaces, char_starts,
                        peer_texts, peer_surfaces, peer_char_starts, side,
                        max_pre, max_post):
    """`minimal_symmetric_surface` で要求する表層を取り直す.

    `side` は `full_text` が対のどちら側か（"a" = 元の文 / "b" = 目標の文）。
    引数は両側で同じなので、src 側と ref 側から別々に呼んでも
    **同じ (pre, post)** が選ばれる（`minimal_symmetric_surface` の探索順は固定）。

    返り値 (eff_surfaces, eff_char_starts, ext) / 失敗時 (None, None, reason:str)
    """
    from span_extend import minimal_symmetric_surface  # 遅延 import

    if side not in ("a", "b"):
        return None, None, f"side must be 'a' or 'b', got {side!r}"
    k = len(surfaces)
    if peer_texts is None or peer_surfaces is None:
        return None, None, ("allow_symmetric_extension=True requires peer_texts "
                            "and peer_surfaces (the other side of the pair)")
    if len(peer_texts) != k or len(peer_surfaces) != k:
        return None, None, (f"peer length mismatch: peer_texts={len(peer_texts)} "
                            f"peer_surfaces={len(peer_surfaces)} != k={k}")

    eff, eff_starts, ext = [], [], []
    for i, surf in enumerate(surfaces):
        if not surf or not peer_surfaces[i]:
            return None, None, f"span {i}: empty surface"
        my_start = char_starts[i] if char_starts is not None else None
        peer_start = peer_char_starts[i] if peer_char_starts is not None else None
        occ_self = _occurrence_index(full_text, surf, my_start)
        occ_peer = _occurrence_index(peer_texts[i], peer_surfaces[i], peer_start)
        if side == "a":
            ta, wa, oa = full_text, surf, occ_self
            tb, wb, ob = peer_texts[i], peer_surfaces[i], occ_peer
        else:
            ta, wa, oa = peer_texts[i], peer_surfaces[i], occ_peer
            tb, wb, ob = full_text, surf, occ_self
        r = minimal_symmetric_surface(text_tok, ta, wa, tb, wb,
                                      max_pre=max_pre, max_post=max_post,
                                      occurrence_a=oa, occurrence_b=ob)
        if r is None:
            return None, None, (f"span {i}: no minimal symmetric surface for "
                                f"{wa!r} / {wb!r} within pre<={max_pre}, "
                                f"post<={max_post}")
        s_eff = r["surface_a"] if side == "a" else r["surface_b"]
        eff.append(s_eff)
        if my_start is None:
            eff_starts.append(None)
        else:
            p = int(my_start) - r["pre"]
            eff_starts.append(p if p >= 0 else None)
        ext.append({
            "requested": surf, "surface": s_eff, "side": side,
            "pre": r["pre"], "post": r["post"], "extra": r["extra"],
            "plain": bool(r["plain"]), "occurrence": occ_self,
            "peer_surface": r["surface_b"] if side == "a" else r["surface_a"],
            "span_from_extend": list(r["span_a"] if side == "a" else r["span_b"]),
        })
    return eff, eff_starts, ext


# --- スパン探索（cursor 付き） ---------------------------------------------

def resolve_spans_with_cursor(text_tok, full_text, surfaces, add_bos=0,
                              char_starts=None,
                              allow_symmetric_extension=False,
                              peer_texts=None, peer_surfaces=None,
                              peer_char_starts=None, side="a",
                              max_pre=3, max_post=3):
    """k 個の表層形を**左から順に**トークン範囲へ解決する.

    既存の `find_entity_token_range` は常に文頭から探すので、同じ語が 2 回出ると
    両方の呼び出しが同じ範囲を返す。ここでは文字カーソルを持ち、i 番目の表層形は
    「i-1 番目の終端以降」から探す。

    `char_starts[i]` が与えられ、かつ `full_text[p:p+len(surf)] == surf` を
    満たす場合はその位置を使う（データ生成側が文字オフセットを知っている場合の
    経路）。満たさない場合はカーソル探索へ落とす（JA の
    `normalize_text_with_lang` がテキストを書き換えてオフセットが動く場合に備える）。

    **BOS/EOS のオフセットについて**: `full_text` は必ず「target 部分の
    正規化済みテキスト」（`get_encoder_output_with_prefix` の返す
    `target_text`）を渡すこと。返る index は
    `encode(full_text, add_special_tokens=False)` のトークン空間 + `add_bos`。
    prefix / x_sep / BOS の絶対オフセットは呼び出し側が
    `target_start_in_full` を**全スパンに一度だけ**加算する。
    同じ target_text に対して全スパンをまとめてここで解決し、同じ offset を
    一度だけ足すこと（流儀が混ざるとスパンごとに 1 token ずれる）。

    最小対称拡張（既定 OFF）
    --------------------------
    `allow_symmetric_extension=False`（既定）のときは従来と完全に同じ経路を
    通る。**既定の挙動は 1 ビットも変わっていない。**

    True のときだけ、探索を始める前に
    `span_extend.minimal_symmetric_surface` で要求する表層を取り直す。
    語そのままで往復一致する対は `plain=True` なので表層が変わらず、
    解決結果も変わらない。拡張が起きるのは従来なら除外されていた対だけ。

      peer_texts      : 対の反対側の文（len k）。src 側なら ref の target_text、
                        ref 側なら src の target_text。
      peer_surfaces   : その文で要求する語（len k）。
      peer_char_starts: 反対側の文字オフセット（任意。occurrence の特定に使う）。
      side            : `full_text` が "a"（元の文）か "b"（目標の文）か。
      max_pre/max_post: 広げられる最大文字数。

    src 側と ref 側で `minimal_symmetric_surface` に渡す引数は同一なので、
    別々に呼んでも同じ (pre, post) が選ばれる（探索順は固定）。
    拡張後は呼び出し側も `info["extension"][i]["surface"]` を
    `verify_span_roundtrip` に渡すこと（要求した表層が変わっているので、
    元の語で往復を見ると必ず落ちる）。

    返り値
    ------
    (spans, info)
      spans : [(t_s, t_e)] × k、または解決できなければ None
      info  : dict（診断用）。`ok`, `reason`, `per_span` を持つ。
              per_span[i] = {"surface", "char_start", "char_end",
                             "tok_start", "tok_end", "covered", "via"}
              `allow_symmetric_extension=True` のときは `extension` も付く。
              extension[i] = {"requested", "surface", "side", "pre", "post",
                              "extra", "plain", "occurrence", "peer_surface",
                              "span_from_extend"}
              （per_span[i]["surface"] は**拡張後**の表層。元の語は
                extension[i]["requested"]。既定 OFF のとき `extension` は付かない。）
    """
    offset = 1 if add_bos else 0
    info = {"ok": False, "reason": None, "per_span": []}

    if allow_symmetric_extension:
        eff, eff_starts, ext = _symmetric_surfaces(
            text_tok, full_text, surfaces, char_starts,
            peer_texts, peer_surfaces, peer_char_starts, side,
            max_pre, max_post)
        if eff is None:
            info["reason"] = f"symmetric extension failed: {ext}"
            return None, info
        surfaces, char_starts = eff, eff_starts
        info["extension"] = ext

    try:
        enc = text_tok(full_text, add_special_tokens=False,
                       return_offsets_mapping=True)
        offsets = list(enc["offset_mapping"])
    except Exception as e:  # fast tokenizer が無い環境
        info["reason"] = f"offset_mapping unavailable: {type(e).__name__}: {e}"
        return None, info
    if not offsets:
        info["reason"] = "empty offset_mapping"
        return None, info

    spans = []
    cursor = 0
    for i, surf in enumerate(surfaces):
        if not surf:
            info["reason"] = f"span {i}: empty surface"
            return None, info
        via = "cursor"
        pos = -1
        if char_starts is not None and char_starts[i] is not None:
            p = int(char_starts[i])
            if 0 <= p and full_text[p:p + len(surf)] == surf:
                pos, via = p, "char_start"
        if pos < 0:
            pos = full_text.find(surf, cursor)
            if pos < 0:
                low = full_text.lower().find(surf.lower(), cursor)
                pos, via = low, "cursor_lower"
        if pos < 0:
            info["reason"] = (f"span {i}: surface {surf!r} not found at or after "
                              f"char {cursor} in {full_text!r}")
            return None, info
        if pos < cursor:
            info["reason"] = (f"span {i}: surface {surf!r} at char {pos} is before "
                              f"the cursor {cursor} (spans must be given in "
                              f"left-to-right order)")
            return None, info

        end_char = pos + len(surf)
        idx = [j for j, (s, e) in enumerate(offsets) if s < end_char and e > pos]
        if not idx:
            info["reason"] = f"span {i}: no token covers chars [{pos}, {end_char})"
            return None, info
        t_s, t_e = min(idx), max(idx) + 1
        covered = full_text[offsets[t_s][0]:offsets[t_e - 1][1]]
        info["per_span"].append({
            "surface": surf, "char_start": pos, "char_end": end_char,
            "tok_start": t_s + offset, "tok_end": t_e + offset,
            "covered": covered, "via": via,
        })
        spans.append((t_s + offset, t_e + offset))
        cursor = end_char

    info["ok"] = True
    return spans, info


def verify_span_roundtrip(text_tok, full_text, spans, surfaces, add_bos=0,
                          case_sensitive=True):
    """スパンの往復一致を検証する（不一致なら文ごと skip させるための関数）.

    `full_tokens[t_s:t_e]` が覆う文字範囲を strip したものが表層形と一致するか
    を見る。トークン id の直接比較（`full_tokens[t_s:t_e] == encode(surface)`）
    は SentencePiece では成立しない: 文中の語は "▁" 付きの token になるが
    `encode(surface)` 単体には "▁" が付かないため、正しいスパンでも不一致に
    なる（`find_entity_token_range` の docstring が警告している 2026-07-11 の
    バグと同じ罠）。したがって**文字範囲での往復**を正とする。

    「末尾 3 token を entity とみなす」種類の fallback は一切用意しない。
    False が返ったらその文は捨てること。
    """
    offset = 1 if add_bos else 0
    try:
        enc = text_tok(full_text, add_special_tokens=False,
                       return_offsets_mapping=True)
        offsets = list(enc["offset_mapping"])
    except Exception:
        return False, ["offset_mapping unavailable"]
    bad = []
    for i, ((t_s, t_e), surf) in enumerate(zip(spans, surfaces)):
        a, b = t_s - offset, t_e - offset
        if not (0 <= a < b <= len(offsets)):
            bad.append(f"span {i}: token range ({t_s}, {t_e}) outside "
                       f"[{offset}, {len(offsets) + offset}]")
            continue
        cov = full_text[offsets[a][0]:offsets[b - 1][1]].strip()
        ok = (cov == surf) if case_sensitive else (cov.lower() == surf.lower())
        if not ok:
            bad.append(f"span {i}: covered {cov!r} != surface {surf!r}")
    return (not bad), bad
