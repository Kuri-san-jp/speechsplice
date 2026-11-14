#!/usr/bin/env python3
"""ZH多音字 n300 の pinyin レベル読み判定（CPU）.

Table 1 の文字含有判定は多音字（同表記）の読み変化を測れないため、
転写（whisper large-v3 の漢字）を pypinyin（文脈考慮・声調つき TONE3）で
読み列に変換し、対象文字位置の読みが目標読みに変わったかを窓照合で判定する。

判定の定義:
- src_syl / tgt_syl: text_a / text_b 文脈中の対象文字の読み（TONE3）
- 隣接文字（text_a 中の直後、なければ直前）の読みを文脈錨とし、
  転写読み列に [tgt_syl, 錨(声調不問)] が隣接して現れれば INJ 成功
- 錨が取れない場合は tgt_syl 単独含有
- BL は src_syl で対称に判定
- 変種: 声調なし（toneless）判定も併記
- 部分群: 声調のみ対（src/tgt の基底音節が同一）、退化対（src_syl==tgt_syl、
  pypinyin が両文脈で同読みを返す対）を分けて報告。N=300 は全て保持する

Usage: python3 scripts/analysis/cjk_zh_pinyin_judge.py --checkpoint <path> [--out <path>]
"""
import argparse
import json
import os  # (release) for REPO_ROOT
import math
import re

from pypinyin import pinyin, Style

HAN = re.compile(r"[一-鿿]")


def ctx_reading(text, ch, style=Style.TONE3):
    idx = text.find(ch)
    if idx < 0:
        return None, None
    py = pinyin(text, style=style, heteronym=False)
    syl = py[idx][0] if idx < len(py) else None
    # 文脈錨: 直後の漢字、なければ直前の漢字
    anchor = None
    for j in (idx + 1, idx - 1):
        if 0 <= j < len(text) and HAN.match(text[j]):
            anchor = py[j][0] if j < len(py) else None
            break
    return syl, anchor


def strip_tone(s):
    return re.sub(r"\d", "", s) if s else s


def tr_pinyin(tr):
    chars = [c for c in tr if HAN.match(c)]
    if not chars:
        return []
    return [p[0] for p in pinyin("".join(chars), style=Style.TONE3, heteronym=False)]


def unit_hit(tr_py, syl, anchor, toneless=False):
    if not syl:
        return False
    tgt = strip_tone(syl) if toneless else syl
    seq = [strip_tone(s) for s in tr_py] if toneless else tr_py
    anc = strip_tone(anchor) if anchor else None
    seq_nt = [strip_tone(s) for s in tr_py]
    for i, s in enumerate(seq):
        if s != tgt:
            continue
        if anc is None:
            return True
        if (i + 1 < len(seq_nt) and seq_nt[i + 1] == anc) or (i - 1 >= 0 and seq_nt[i - 1] == anc):
            return True
    return False


def wilson_half(k, n, z=1.96):
    p = k / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) * 100


GOLD_PATH = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/zh_polyphone_gold_readings.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = json.load(open(GOLD_PATH))
    rows = json.load(open(args.checkpoint))["results"]
    ok = [r for r in rows if r.get("status") == "success"]
    out_rows = []
    agg = {"n": len(ok), "inj_tone": 0, "inj_toneless": 0, "bl_tone": 0,
           "trm_char": 0, "tone_only_pairs": 0, "degenerate_pairs": 0,
           "inj_tone_toneonly": 0, "inj_tone_segmental": 0}
    n_toneonly = n_seg = 0
    for r in ok:
        ch = r["word_a"]
        # 読みは文単位のgold表（pypinyin+辞書照合の上書き29文）を正とする
        _, anc_a = ctx_reading(r["text_a"], ch)
        src = gold[r["text_a"]]["reading"]
        tgt = gold[r["text_b"]]["reading"]
        def han_only(t):
            return "".join(c for c in t if HAN.match(c))
        # 転写が原文と同一（漢字列一致）の場合、対象文字の読みは転写から判別不能。
        # gold表の原文読み（=src）を転写側にも適用する（pypinyin誤りの二重計上を防ぎ、
        # INJ側の見かけ成功（辞書誤りの偶然一致）も排除する）
        bl_verbatim = han_only(r.get("bl_transcript", "")) == han_only(r["text_a"])
        inj_verbatim = han_only(r.get("oeoi_transcript", "")) == han_only(r["text_a"])
        tr_bl = tr_pinyin(r.get("bl_transcript", ""))
        tr_inj = tr_pinyin(r.get("oeoi_transcript", ""))
        tone_only = bool(src and tgt and src != tgt and strip_tone(src) == strip_tone(tgt))
        degenerate = bool(src and tgt and src == tgt)
        if inj_verbatim:
            # 原文どおりの転写: 読み＝原文読み(src)とみなす（読み変化の証拠なし）
            inj_tone = (tgt == src)
            inj_nt = (strip_tone(tgt) == strip_tone(src))
        else:
            inj_tone = unit_hit(tr_inj, tgt, anc_a)
            inj_nt = unit_hit(tr_inj, tgt, anc_a, toneless=True)
        bl_tone = True if bl_verbatim else unit_hit(tr_bl, src, anc_a)
        agg.setdefault("inj_verbatim", 0)
        agg["inj_verbatim"] += inj_verbatim
        agg["inj_tone"] += inj_tone
        agg["inj_toneless"] += inj_nt
        agg["bl_tone"] += bl_tone
        agg["trm_char"] += bool(r.get("oeoi_correct"))
        agg["tone_only_pairs"] += tone_only
        agg["degenerate_pairs"] += degenerate
        if tone_only:
            n_toneonly += 1
            agg["inj_tone_toneonly"] += inj_tone
        elif not degenerate:
            n_seg += 1
            agg["inj_tone_segmental"] += inj_tone
        out_rows.append({"pair_idx": r["pair_idx"], "char": ch, "src": src, "tgt": tgt,
                         "anchor": anc_a, "tone_only": tone_only, "degenerate": degenerate,
                         "inj_tone": bool(inj_tone), "inj_toneless": bool(inj_nt),
                         "bl_tone": bool(bl_tone), "inj_verbatim": bool(inj_verbatim),
                         "trm_char": bool(r.get("oeoi_correct"))})
    n = agg["n"]
    print(f"n={n} (声調のみ対 {agg['tone_only_pairs']} / 退化対 {agg['degenerate_pairs']} / "
          f"INJ転写が原文どおり(読み判別不能) {agg.get('inj_verbatim', 0)})")
    for k, label in [("bl_tone", "BL 読み一致(声調あり)"), ("inj_tone", "INJ 読み一致(声調あり)"),
                     ("inj_toneless", "INJ 読み一致(声調なし)"), ("trm_char", "参考: 文字含有(論文判定)")]:
        v = agg[k]
        print(f"  {label}: {v}/{n} = {v/n*100:.2f}% ±{wilson_half(v, n):.2f}")
    if n_toneonly:
        print(f"  内訳 INJ(声調あり): 声調のみ対 {agg['inj_tone_toneonly']}/{n_toneonly}"
              f" = {agg['inj_tone_toneonly']/n_toneonly*100:.1f}% / "
              f"分節が異なる対 {agg['inj_tone_segmental']}/{n_seg}"
              f" = {agg['inj_tone_segmental']/n_seg*100:.1f}%")
    if args.out:
        json.dump({"summary": agg, "rows": out_rows}, open(args.out, "w"),
                  ensure_ascii=False, indent=1)
        print("->", args.out)


if __name__ == "__main__":
    main()
