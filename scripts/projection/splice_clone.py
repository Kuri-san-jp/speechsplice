#!/usr/bin/env python3
"""voice cloning経路用 splice 注入の部品（clone-splice）.

pooling（長さ不変圧縮）に代わり、prefix + 置換span + suffix を系列長ごと
差し替える splice を voice cloning 経路（eval_utmos_speaker_fixed.voice_clone_synth）
で使うための共有モジュール。モデル本体・vendor（TTS36/HF cache）は改造しない。

原理: encoder attention mask と pm-RoPE position ids はいずれも input_ids 長
のみから導出される（batch=1 で mask=all-ones、position=長さ正規化 uniform
progress）。したがって inference_one_sample に渡す target_text を
「splice後のtargetトークン数と厳密一致するダミーテキスト」にすれば、
既存の encoder forward hook（last_hidden_state 全置換）のまま系列長可変の
splice 注入が成立する。ダミーの内容は hidden 全置換により出力へ一切
寄与しない。duration は text_a 基準で固定する（Δduration=0.00s 維持、
2026-07-22 plan Q1 決定）。

背景・検証計画: note/2026-07-22_spliceclone_research.md / _plan.md
"""
import torch

# 単一文字の反復・結合でダミーターゲットを構成する候補（JA向け中心。
# tokenizer 依存の merge で特定トークン数が単一文字反復で作れない場合に
# 備えて複数用意する）
FILLERS = ["あ", "ん", "ア", "ま", "か", "や"]


def build_spliced_hidden(src_enc, ref_enc, s_range, r_range):
    """splice済み encoder hidden state と Δtok を返す.

    s_range / r_range はフル系列（prefix + x_sep + target + eos）内の
    絶対インデックス（eval_utmos_speaker_fixed.py が target_start_in_full を
    加算した後の値）。等長時の結果は in-place 行コピーと torch.equal
    （tests/test_splice_clone.py で保証）。

    Returns: (modified [1, T_out, d], delta_tok)  T_out = T_src + delta_tok
    """
    s_s, s_e = s_range
    r_s, r_e = r_range
    src_h = src_enc["last_hidden_state"]
    ref_h = ref_enc["last_hidden_state"]
    modified = torch.cat(
        [src_h[0, :s_s], ref_h[0, r_s:r_e], src_h[0, s_e:]], dim=0
    ).unsqueeze(0)
    delta_tok = (r_e - r_s) - (s_e - s_s)
    assert modified.shape[1] == src_h.shape[1] + delta_tok, (
        f"splice length mismatch: T_out={modified.shape[1]} "
        f"T_src={src_h.shape[1]} delta={delta_tok}")
    return modified, delta_tok


def prepare_clone_splice(text_tok, src_enc, ref_enc, s_range, r_range, lang,
                         duration_source_text, natural_candidates=()):
    """eval/bench 共用の clone-splice 準備（重複ロジックの一元化）.

    Returns: (modified, inj_target, inj_dur_text, delta_tok)
      - delta_tok == 0: inj_target=None / inj_dur_text=None
        （呼び出し側は従来どおり text_a で合成する。pool経路と同一挙動）
      - delta_tok != 0: inj_target=厳密長一致ダミー、
        inj_dur_text=duration_source_text（Δduration=0.00s 維持用）
    """
    modified, delta_tok = build_spliced_hidden(src_enc, ref_enc, s_range, r_range)
    if delta_tok == 0:
        return modified, None, None, 0
    needed = len(src_enc["target_tokens"]) + delta_tok
    inj_target = make_length_matched_target(text_tok, needed, lang, natural_candidates)
    return modified, inj_target, duration_source_text, delta_tok


def target_token_len(text_tok, text, lang):
    """inference_one_sample / get_encoder_output_with_prefix と同一の
    正規化（normalize_text_with_lang）→ strip → encode で target 部の
    トークン数を返す（完全ミラー。独自正規化は書かない）."""
    from inference_tts_utils import normalize_text_with_lang
    n, _ = normalize_text_with_lang(text, lang)
    return len(text_tok.encode(n.strip(), add_special_tokens=False))


def make_length_matched_target(text_tok, needed_len, lang, natural_candidates=()):
    """encode後トークン数が needed_len に厳密一致するターゲット文字列を返す.

    優先順: natural_candidates（text_b 等）→ 単一フィラー反復 →
    2種フィラー結合の全探索。見つからなければ AssertionError
    （黙って別の長さで進む silent fallback は禁止）。
    """
    assert needed_len >= 1, f"needed_len={needed_len}"
    for cand in natural_candidates:
        if cand and target_token_len(text_tok, cand, lang) == needed_len:
            return cand
    for ch in FILLERS:
        for k in range(1, needed_len * 3 + 8):
            s = ch * k
            n = target_token_len(text_tok, s, lang)
            if n == needed_len:
                return s
            if n > needed_len + 8:
                break
    for ch1 in FILLERS:
        for ch2 in FILLERS:
            if ch1 == ch2:
                continue
            for k1 in range(1, needed_len * 2 + 4):
                base = ch1 * k1
                n1 = target_token_len(text_tok, base, lang)
                if n1 >= needed_len:
                    break
                for k2 in range(1, (needed_len - n1) * 2 + 4):
                    s = base + ch2 * k2
                    n = target_token_len(text_tok, s, lang)
                    if n == needed_len:
                        return s
                    if n > needed_len + 4:
                        break
    raise AssertionError(
        f"length-matched dummy target not found: needed_len={needed_len} lang={lang}")
