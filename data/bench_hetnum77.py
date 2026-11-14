#!/usr/bin/env python3
"""ja-tts-g2p-bench の heteronym(43) + context_numeral(34) = 77項目の
T5Gemma-TTS 誤読スクリーニング用ペア.

各項目: word_a=対象語（漢字表記）、word_b=文脈上の正読（カタカナ）、
wrong_kana=[対立読み（カタカナ）]。numerals_verify_generic.py がこの定義から
BL / INJ（正読カナ注入）/ ref_c（正読参照合成）/ ref_w（誤読参照合成）を生成し、
BLがどちらの読みに近いかを相対判定する。
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "external", "ja_tts_g2p_bench_v1.jsonl")
_CATS = ("heteronym", "context_numeral")


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
            print(f"[bench_hetnum77] skip {it['id']}: target '{tgt}' not in sentence")
            continue
        kc = _kata(it["reading_expected"])
        kw = _kata(it["reading_alt"])
        if kc == kw:
            print(f"[bench_hetnum77] skip {it['id']}: readings identical")
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
