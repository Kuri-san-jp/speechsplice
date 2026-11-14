#!/usr/bin/env python3
"""EN 2 エンティティ同時置換: 300 文（50 語ペア組 × 6 文型）.

**既存ファイルは 1 行も変更していない。** 語ペアは
`data/en_wordrepl_500.py:EN_PAIRS`（50 対）をそのまま import して使う。

構造
----
Table 1 の EN 単独条件（`data/en_wordrepl300_pairs.py` = 50 語ペア × 6 文型 =
300 文）と**同じ 50 × 6 の格子**にした。したがって単独成功率 p1 / p2 は
公開済みの単独条件と同じ語・同じ規模で測られ、直接比較できる。

スロット割り当て
----------------
文 i のスロット 1 は `EN_PAIRS[i]`、スロット 2 は `EN_PAIRS[(i + 17) % 50]`。
gcd(17, 50) = 1 なので i -> i+17 は長さ 50 の巡回になり、

  * 50 通りの組み合わせがすべて相異なる
  * どの語ペアもちょうど 1 回スロット 1、ちょうど 1 回スロット 2 に出る
    （スロット位置と語の交絡が無い）

`scripts/multientity/validate_multientity.py` が、1 文に現れる 4 語
（word_a1, word_b1, word_a2, word_b2）について正規化後の重複と
部分文字列包含が無いことを全文で検査する。これが無いと
「転写に word_b が含まれるか」という判定が別スロットの語で誤爆する。

距離条件
--------
2 語の間の語数を 0 / 2 / 4 / 7 / 11 / 15 と変えた 6 文型。実際のトークン間隔は
tokenizer 依存なので、生成スクリプトが解決後の
`gap_tokens = s2_start - s1_end` を各文に記録する。`gap_words` はあくまで
設計上の目安。

落とし穴の被覆
--------------
  * 2 語が近接         : M0（間に語が無い。カンマだけ）
  * 片方が文頭         : M0, M2（スロット 1 が文頭。300 文中 100 文）
  * 語長が大きく違う   : 語ペアの組は固定なので、生成側が n1, n2, m1, m2 を
                         記録し、|m1 - m2| で層別できるようにする
  * 同じ語が 2 回      : 本ファイルには入れない。
                         `data/en_multientity_repeat_40.py` の専用プローブで扱う

レコード
--------
  pair_idx      : 0..299
  template_id   : "M0_g0" 等
  lang          : "en"
  text_a        : 2 語とも元の形の文（BL と src encoder forward に使う）
  spans         : [{slot, word_a, word_b, char_start_a, char_start_b}] × 2
                  char_start_a は text_a 内、char_start_b は text_refs[slot-1] 内
  text_refs     : [スロット 1 だけ置換した文, スロット 2 だけ置換した文]
                  ref はスパンごとに別の forward を使う（両方置換した 1 本の ref
                  から 2 スパンを取ると干渉の原因が分離できない）
  text_both     : 2 語とも置換した文。長さ一致ダミーの自然候補にのみ使い、
                  encoder hidden の供給源には**しない**
  gap_words     : 設計上の語間隔
  slot1_at_start: スロット 1 が文頭か
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from en_wordrepl_500 import EN_PAIRS  # noqa: E402

SLOT_OFFSET = 17

# 間の語数が単調に増える 6 文型。{e1} が {e2} より必ず先に出る。
TEMPLATES = [
    ("M0_g0", 0,
     "{e1}, {e2}, and many others are famous."),
    ("M1_g2", 2,
     "Yesterday, {e1} and then {e2} became famous."),
    ("M2_g4", 4,
     "{e1} is better known than {e2} in Europe."),
    ("M3_g7", 7,
     "Many people say that {e1} is a good deal more famous than {e2} today."),
    ("M4_g11", 11,
     "According to the report, {e1} was mentioned far more often in the news "
     "last year than {e2} was."),
    ("M5_g15", 15,
     "The committee finally decided that {e1} should be the first name written "
     "right at the top of the list and that {e2} should be the second."),
]
assert len(TEMPLATES) == 6


def _fill(template, w1, w2):
    """{e1}/{e2} を埋め、埋めた語の文字オフセットも返す.

    `str.replace(..., 1)` を順に当てるので、オフセットは置換後の文字列に対して
    厳密に決まる（後から find し直さないので、同じ語が別の場所にあっても誤らない）。
    """
    off1 = template.index("{e1}")
    s = template.replace("{e1}", w1, 1)
    off2 = s.index("{e2}")
    s = s.replace("{e2}", w2, 1)
    assert s[off1:off1 + len(w1)] == w1, (s, off1, w1)
    assert s[off2:off2 + len(w2)] == w2, (s, off2, w2)
    assert off1 + len(w1) <= off2, "slot 1 must precede slot 2"
    return s, off1, off2


def build_pairs():
    items = []
    idx = 0
    for tmpl_id, gap_words, tmpl in TEMPLATES:
        for i in range(len(EN_PAIRS)):
            a1, b1 = EN_PAIRS[i]
            a2, b2 = EN_PAIRS[(i + SLOT_OFFSET) % len(EN_PAIRS)]

            text_a, ca1, ca2 = _fill(tmpl, a1, a2)
            ref1, cb1, _ = _fill(tmpl, b1, a2)      # スロット 1 だけ置換
            ref2, _, cb2 = _fill(tmpl, a1, b2)      # スロット 2 だけ置換
            text_both, _, _ = _fill(tmpl, b1, b2)

            items.append({
                "pair_idx": idx,
                "template_id": tmpl_id,
                "lang": "en",
                "text_a": text_a,
                "text_refs": [ref1, ref2],
                "text_both": text_both,
                "gap_words": gap_words,
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


if __name__ == "__main__":
    ps = build_pairs()
    print(f"{len(ps)} items")
    for j in (0, 50, 100, 150, 200, 250, 299):
        p = ps[j]
        print(f"[{p['pair_idx']}] {p['template_id']} gap_words={p['gap_words']}")
        print(f"   a   : {p['text_a']}")
        print(f"   ref1: {p['text_refs'][0]}")
        print(f"   ref2: {p['text_refs'][1]}")
        print(f"   both: {p['text_both']}")
