#!/usr/bin/env python3
"""JA 2 エンティティ同時置換: 300 文（50 姓ペア組 × 6 文型）.

**既存ファイルは 1 行も変更していない。** 姓ペアは
`data/ja_wordrepl300_pairs.py:_PAIRS`（50 対、MOS 検証済み）をそのまま import する。

EN 版（`data/en_multientity_300.py`）と同じ 50 × 6 の格子・同じレコード形式。
スロット割り当ては `(i + 9) % 50`。gcd(9, 50) = 1 なので 50 巡回になり、
どの姓ペアもちょうど 1 回ずつスロット 1 / スロット 2 に出る。
オフセットは **pykakasi の読み**で衝突（重複・部分文字列包含）が 0 になる
ものから選んだ。JA の判定は表層ではなく読みの部分一致なので、
「森」(もり) と「森田」(もりた) のような組は判定を誤爆させる
（EN 版で使った 17 は JA ではこの衝突が出るので使わない）。

JA 固有の注意
------------
1. `inference_tts_utils.normalize_text_with_lang(text, "ja")` が
   `_normalize_japanese_text` を通す。本ファイルの文は全角英数も
   長い三点リーダも含まないので恒等写像になるはずだが、
   `scripts/multientity/validate_multientity.py` が全文で
   `normalize(text) == text` を実際に確認する。恒等でない文が 1 つでもあれば
   `char_start_*` は当てにならないので、生成側は cursor 探索に落ちる
   （`resolve_spans_with_cursor` がその場合を自動で処理する）。
2. 姓 + 「さん」が 1 トークンに merge されるとスパンの往復一致が落ちる。
   validator が tokenizer で全件確認し、落ちた文は生成対象から外す。
   「末尾 N トークンを entity とみなす」種類の fallback は使わない。
3. 判定は `scripts/analysis/ja_wordrepl_judge.py` と同じ流儀
   （furigana Whisper 転写 + pykakasi の読み + かな正規化）。pyopenjtalk は使わない。

距離条件: 2 語の間の文字数が単調に増える 6 文型。実トークン間隔は生成側が記録する。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ja_wordrepl300_pairs import _PAIRS as JA_PAIRS  # noqa: E402

SLOT_OFFSET = 9

TEMPLATES = [
    ("J0_g3", "{e1}さんと{e2}さんが会議の資料を準備してくれた。"),
    ("J1_g5", "昨日、{e1}さんが駅で{e2}さんに偶然会った。"),
    ("J2_g7", "今度の企画は{e1}さんに任せるか{e2}さんに任せるかで迷っている。"),
    ("J3_g9", "その提案は{e1}さんではなく、実は{e2}さんが考えたものです。"),
    ("J4_g12", "受付で{e1}さんを呼んだあと、続けて{e2}さんを呼んでください。"),
    ("J5_g18",
     "最後に発表するのは{e1}さんだと聞いていたのですが、実際には{e2}さんでした。"),
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
        for i in range(len(JA_PAIRS)):
            a1, b1 = JA_PAIRS[i]
            a2, b2 = JA_PAIRS[(i + SLOT_OFFSET) % len(JA_PAIRS)]
            text_a, ca1, ca2 = _fill(tmpl, a1, a2)
            ref1, cb1, _ = _fill(tmpl, b1, a2)
            ref2, _, cb2 = _fill(tmpl, a1, b2)
            text_both, _, _ = _fill(tmpl, b1, b2)
            items.append({
                "pair_idx": idx,
                "template_id": tmpl_id,
                "lang": "ja",
                "text_a": text_a,
                "text_refs": [ref1, ref2],
                "text_both": text_both,
                "gap_words": gap,          # JA は文字数（目安）
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
    for j in (0, 50, 100, 150, 200, 250):
        p = ps[j]
        print(f"[{p['pair_idx']}] {p['template_id']} gap_chars={p['gap_words']}")
        print(f"   a   : {p['text_a']}")
        print(f"   ref1: {p['text_refs'][0]}")
        print(f"   ref2: {p['text_refs'][1]}")
