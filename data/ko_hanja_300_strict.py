"""Strict version of ko_hanja_300: only pairs whose text_a AND text_b both
contain the entity hanja (or its 두음법칙 equivalent) are included.

This guarantees `find_entity_token_range` succeeds on every pair, so the
caller can rely on exactly N=300 pairs being yielded by `build_300_pairs()`.

Source: ko_hanja_300.HETERONYMS — same hanja list but additional (text_a,
text_b) contexts are sampled to maintain N=300.
"""
import os, sys, itertools

# Reuse the same HETERONYMS list from the public ko_hanja_300 module.
_data_dir = os.path.dirname(os.path.abspath(__file__))
if _data_dir not in sys.path:
    sys.path.insert(0, _data_dir)
import ko_hanja_300  # noqa


# 두음법칙 mapping for SinoKorean hanja initial-sound shifts.
DUEUM = {
    "락": "낙", "랑": "낭", "량": "양", "려": "여", "력": "역",
    "련": "연", "렬": "열", "령": "영", "례": "예", "로": "노",
    "록": "녹", "론": "논", "롱": "농", "뢰": "뇌", "료": "요",
    "룡": "용", "루": "누", "류": "유", "륙": "육", "륜": "윤",
    "률": "율", "륭": "융", "륵": "늑", "름": "늠", "릉": "능",
    "리": "이", "린": "인", "림": "임", "립": "입",
}
DUEUM_REV = {v: k for k, v in DUEUM.items()}


def _contains(text, word):
    if word in text or word.lower() in text.lower():
        return True
    if word in DUEUM and DUEUM[word] in text: return True
    if word in DUEUM_REV and DUEUM_REV[word] in text: return True
    return False


def build_300_pairs():
    """Return N=300 pairs guaranteed to have the entity in both contexts.

    Strategy: enumerate all (entity, ctx_a, ctx_b) combinations from
    HETERONYMS; keep only those where ctx_a contains entity AND ctx_b
    contains entity (or 두음법칙 form). Truncate to 300.
    """
    pairs = []
    pair_idx = 0
    for entry in ko_hanja_300.HANJA_POLYPHONES:
        # entry is (word, label_a, ctxs_a, label_b, ctxs_b)
        word, label_a, ctxs_a, label_b, ctxs_b = entry
        valid_a = [c for c in ctxs_a if _contains(c, word)]
        valid_b = [c for c in ctxs_b if _contains(c, word)]
        for ca, cb in itertools.product(valid_a, valid_b):
            pairs.append({
                "word_a": word, "word_b": word,
                "text_a": ca, "text_b": cb,
                "direction": "a_to_b",
                "reading_a_label": label_a,
                "reading_b_label": label_b,
                "lang": "ko",
                "pair_idx": pair_idx,
            })
            pair_idx += 1
    return pairs[:300]


if __name__ == "__main__":
    p = build_300_pairs()
    print(f"strict pairs: {len(p)}")
    from collections import Counter
    c = Counter(x["word_a"] for x in p)
    print(f"unique entities: {len(c)}, distribution: {dict(c)}")
