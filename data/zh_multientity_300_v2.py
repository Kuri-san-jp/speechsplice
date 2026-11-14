#!/usr/bin/env python3
"""ZH 2 エンティティ同時置換 v2: 300 文（50 地名ペア × 6 文型）.

## なぜ作り直したか（2026-09-19）

旧版 `data/zh_multientity_300.py` は **中国語だけ語彙の設計が EN / JA と違った。**

    EN  London -> Paris   同種内（英語地名 -> 英語地名）
    JA  田中   -> 鈴木    同種内（日本姓 -> 日本姓）
    ZH  王     -> 山田    **異種間**（中国姓 1 字 -> 日本姓 2 字）

旧版は `zh_cross_culture_names` の語彙をそのまま流用していた。単一 entity の
cross-culture 条件と格子を揃える意図はあったが、**EN / JA が同種内なので
3 言語を並べられない。**実害も出た: 中国語の文脈で日本姓を発話させ、その
**表記**が Whisper の書き起こしに出るかで判定していたため、注入が届いていても
ASR が別の漢字を当てて落ちた（王 -> 山田 が "Yamada" と書き起こされる、
呉 -> 加藤 が "加通" になる）。差し替えた枠が読めた率が
EN 95.3% / JA 91.3% に対し **ZH 54.5%** と、言語差ではなく設計差で下がっていた。
詳細は `note/2026-09-19_multientity_zh_report.md`。

v2 は **EN / JA と同じ作り**にする。

    語彙  `data/zh_wordrepl300_pairs.py:_PAIRS`（中国地名 -> 中国地名、50 対、
          MOS 検証済み）。EN の `en_wordrepl_500:EN_PAIRS`、
          JA の `ja_wordrepl300_pairs:_PAIRS` と同じ系列
    格子  50 対 × 6 文型 = 300（EN / JA と同一）
    形式  レコードの鍵も EN / JA と同一

**旧版は 1 行も変更していない。**既存の結果（`multientity_zh_300/`）はその
ファイルで作られたものとして残る。

## スロット割り当て

文 i のスロット 1 は `_PAIRS[i]`、スロット 2 は `_PAIRS[(i + OFFSET) % 50]`。
gcd(OFFSET, 50) = 1 なら 50 巡回になり、どの対もちょうど 1 回ずつ
スロット 1 / スロット 2 に出る（スロット位置と語の交絡が無い）。

**OFFSET は「4 語の衝突が 0 になるもの」から選ぶ。**ZH の判定は
`scripts/analysis/cjk_zh_pinyin_judge.py` と同じ流儀で pypinyin の読み列の
窓照合なので、衝突は**表層と読み列の両方**で見る必要がある。
このファイルの末尾に検査を置き、`python3 data/zh_multientity_300_v2.py` で
全 OFFSET 候補の衝突数を出せるようにした。選んだ値の根拠は下の定数のコメント。

## 正規化

`inference_tts_utils.normalize_text_with_lang(text, "zh")` は lang が "ja" で
始まらないとき恒等写像なので、`char_start_*` は target_text 上でそのまま使える。
句読点は既存の ZH データと同じ半角 "," を使う。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zh_wordrepl300_pairs import _PAIRS as ZH_PAIRS  # noqa: E402

# **OFFSET は下の衝突検査で 0 になったものから選ぶ。**
# `python3 data/zh_multientity_300_v2.py --check` が全候補を数え直す。
# 実測（2026-09-19）: gcd(offset,50)==1 の候補のうち衝突 0 は
#   1 / 17 / 19 / 21 / 23 / 27 / 29 / 31 / 33 / 49
# **offset 1 は隣接対になり、地名の並びの偏りを拾う恐れがあるので避ける。**
# EN 版が 17、JA 版が 9 を使っているので、**衝突 0 の中で EN と同じ 17** を採る。
SLOT_OFFSET = 17

# 2 語の間の文字数が単調に増える 6 文型（JA 版と同じ考え方）。
# 中国語は分かち書きが無いので gap は**文字数**（JA 版と同じ規約）。
TEMPLATES = [
    ("Z0_g3", "{e1}和{e2}都是有名的城市。"),
    ("Z1_g5", "昨天他先去了{e1}，然后又去了{e2}。"),
    ("Z2_g7", "这次的会议在{e1}举行，下次的会议在{e2}举行。"),
    ("Z3_g9", "那份报告不是在{e1}写的，实际上是在{e2}写的。"),
    ("Z4_g12", "请先联系{e1}的负责人，接着再联系{e2}的负责人。"),
    ("Z5_g18", "我本来以为最后要去的是{e1}，但实际上要去的是{e2}。"),
]
assert len(TEMPLATES) == 6


def _fill(template, w1, w2):
    off1 = template.index("{e1}")
    s = template.replace("{e1}", w1, 1)
    off2 = s.index("{e2}")
    s = s.replace("{e2}", w2, 1)
    assert s[off1:off1 + len(w1)] == w1
    assert s[off2:off2 + len(w2)] == w2
    assert off1 + len(w1) <= off2
    return s, off1, off2


def _gap_chars(template):
    """{e1} の直後から {e2} の直前までの文字数（設計上の距離の目安）."""
    a = template.index("{e1}") + len("{e1}")
    b = template.index("{e2}")
    return b - a


def build_pairs():
    items = []
    idx = 0
    for tmpl_id, tmpl in TEMPLATES:
        gap = _gap_chars(tmpl)
        for i in range(len(ZH_PAIRS)):
            a1, b1 = ZH_PAIRS[i]
            a2, b2 = ZH_PAIRS[(i + SLOT_OFFSET) % len(ZH_PAIRS)]
            text_a, ca1, ca2 = _fill(tmpl, a1, a2)
            ref1, cb1, _ = _fill(tmpl, b1, a2)
            ref2, _, cb2 = _fill(tmpl, a1, b2)
            text_both, _, _ = _fill(tmpl, b1, b2)
            items.append({
                "pair_idx": idx,
                "template_id": tmpl_id,
                "lang": "zh",
                "text_a": text_a,
                "text_refs": [ref1, ref2],
                "text_both": text_both,
                "gap_words": gap,          # ZH は文字数（JA 版と同じ規約）
                "slot1_at_start": tmpl.startswith("{e1}"),
                "spans": [
                    {"slot": 1, "word_a": a1, "word_b": b1,
                     "char_start_a": ca1, "char_start_b": cb1},
                    {"slot": 2, "word_a": a2, "word_b": b2,
                     "char_start_a": ca2, "char_start_b": cb2},
                ],
            })
            idx += 1
    assert len(items) == 300, len(items)
    return items


# --------------------------------------------------------------------------
# 衝突検査（OFFSET を選ぶ根拠。`--check` で全候補を数える）
# --------------------------------------------------------------------------
def _collisions(offset):
    """4 語（a1,b1,a2,b2）の表層と読み列の衝突を数える。

    衝突 = 重複、または部分文字列の包含。表層と、pypinyin の声調なし読み列の
    両方で見る。`cjk_zh_pinyin_judge.py` が読み列の窓照合で判定するため、
    表層だけ見ても足りない。
    """
    from pypinyin import lazy_pinyin
    n = 0
    for i in range(len(ZH_PAIRS)):
        a1, b1 = ZH_PAIRS[i]
        a2, b2 = ZH_PAIRS[(i + offset) % len(ZH_PAIRS)]
        words = [a1, b1, a2, b2]
        if len(set(words)) != 4:
            n += 1
            continue
        for x in words:
            for y in words:
                if x is not y and x in y:
                    n += 1
                    break
            else:
                continue
            break
        else:
            pys = ["".join(lazy_pinyin(w)) for w in words]
            if len(set(pys)) != 4 or any(
                    p in q for p in pys for q in pys if p is not q):
                n += 1
    return n


if __name__ == "__main__":
    if "--check" in sys.argv:
        from math import gcd
        print("OFFSET ごとの衝突数（gcd(offset,50)==1 のものだけ）")
        for off in range(1, 50):
            if gcd(off, len(ZH_PAIRS)) != 1:
                continue
            c = _collisions(off)
            mark = "  <- 採用" if off == SLOT_OFFSET else ""
            if c == 0 or off == SLOT_OFFSET:
                print("  offset %2d  衝突 %d%s" % (off, c, mark))
        raise SystemExit(0)

    ps = build_pairs()
    print("%d items" % len(ps))
    for j in (0, 50, 100, 150, 200, 250):
        p = ps[j]
        print("[%d] %s gap_chars=%s" % (p["pair_idx"], p["template_id"],
                                        p["gap_words"]))
        print("   a    : %s" % p["text_a"])
        print("   ref1 : %s" % p["text_refs"][0])
        print("   ref2 : %s" % p["text_refs"][1])
        print("   both : %s" % p["text_both"])
        print("   spans: %s" % [(s["word_a"], s["word_b"]) for s in p["spans"]])
