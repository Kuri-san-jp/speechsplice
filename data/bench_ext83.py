#!/usr/bin/env python3
"""ja-tts-g2p-bench の残り6カテゴリ = bench160拡張分（B1対象）のペア定義.

bench_hetnum77.py（heteronym43 + context_numeral34）と同一の構造・判定系を、
残りカテゴリ drama_specific(24) / okurigana(35) / jukujikun(6) /
verb_conjugation(6) / juubako_yuto(6) / onkun_domain(6) = 83項目に適用する。
numerals_verify_generic.py がこの定義から BL / ref_c / ref_w を生成し、
furigana whisper 判定（bench_hetnum77_furigana.py --module ext83）で
BL がどちらの読みに近いかを相対判定する。
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "external", "ja_tts_g2p_bench_v1.jsonl")
_CATS = ("drama_specific", "okurigana", "jukujikun",
         "verb_conjugation", "juubako_yuto", "onkun_domain")


def _kata(s):
    """ひらがな→カタカナ（U+3041-3096 を +0x60 シフト、他はそのまま）."""
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c
                   for c in s)


def build_pairs():
    pairs = []
    idx = 0
    with open(_SRC) as f:
        items = [json.loads(l) for l in f]
    for it in items:
        if it.get("category") not in _CATS:
            continue
        tgt = it["target"]
        sent = it["surface"]
        if tgt not in sent:
            print(f"[bench_ext83] skip {it['id']}: target '{tgt}' not in sentence")
            continue
        kc = _kata(it["reading_expected"])
        kw = _kata(it["reading_alt"])
        if kc == kw:
            print(f"[bench_ext83] skip {it['id']}: readings identical")
            continue
        pairs.append({
            "pair_idx": idx,
            "bench_id": it["id"],
            "category": it["category"],
            "note": it.get("note", ""),
            "word_a": tgt,
            "word_b": kc,
            "text_a": sent,
            "text_b": sent.replace(tgt, kc, 1),
            "correct_kana": kc,
            "wrong_kana": [kw],
        })
        idx += 1
    return pairs


if __name__ == "__main__":
    ps = build_pairs()
    import collections
    print(len(ps), dict(collections.Counter(p["category"] for p in ps)))
    for p in ps[:3]:
        print(p["bench_id"], p["word_a"], p["word_b"], "/", p["wrong_kana"][0], "|", p["text_a"][:30])
