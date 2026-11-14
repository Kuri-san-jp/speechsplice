#!/usr/bin/env python3
"""500 unique EN word-replacement pairs for cross-arch injection benchmark.

Structure: 50 proper-noun pairs (word_a != word_b) × 10 syntactic templates = 500 pairs.

Each pair (word_a, word_b, template) evaluates whether encoder-output injection
of word_b's hidden state at word_a's position changes the ASR transcript.

Same design principle as eval_en_midend_precise.py (which used 30 pairs × 12 templates)
but expanded to the canonical N=500 used throughout the cross-arch analysis.
"""

TEMPLATES = {
    "P0_subject_start":    "{word} is famous.",
    "P1_temporal_prefix":  "Yesterday, {word} became famous.",
    "P2_adverbial_prefix": "Around the world, {word} is known.",
    "P3_object_middle":    "Many people know {word} very well.",
    "P4_object_end":       "They respect {word}.",
    "P5_ni_middle":        "Everyone looks up to {word} today.",
    "P6_copula_predicate": "The most famous today is {word}.",
    "P7_apposition_end":   "The famous person is {word}.",
    "P8_relative_middle":  "This is a book {word} wrote.",
    "P9_short_end":        "Today is {word}.",
}
assert len(TEMPLATES) == 10

EN_PAIRS = [
    # First 30: same as eval_en_midend_precise.py
    ("London", "Paris"), ("Michael", "John"), ("Einstein", "Newton"),
    ("Mozart", "Beethoven"), ("Amazon", "Google"), ("Harvard", "Oxford"),
    ("Ferrari", "Porsche"), ("Saturn", "Jupiter"), ("Diamond", "Emerald"),
    ("Christmas", "Halloween"), ("Hollywood", "Broadway"), ("Victoria", "Elizabeth"),
    ("Leonardo", "Picasso"), ("Everest", "Kilimanjaro"), ("Atlantic", "Pacific"),
    ("Shakespeare", "Dickens"), ("Samsung", "Apple"), ("Tokyo", "Berlin"),
    ("Sydney", "Toronto"), ("Stanford", "Berkeley"), ("Napoleon", "Caesar"),
    ("Darwin", "Galileo"), ("Tesla", "Edison"), ("Boeing", "Airbus"),
    ("Disney", "Warner"), ("Intel", "AMD"), ("Chicago", "Boston"),
    ("Dallas", "Houston"), ("Denver", "Seattle"), ("Miami", "Atlanta"),
    # Additional 20 pairs
    ("Nike", "Adidas"), ("Toyota", "Honda"), ("Columbus", "Magellan"),
    ("Madrid", "Rome"), ("Vienna", "Prague"), ("Moscow", "Beijing"),
    ("Lincoln", "Washington"), ("Kennedy", "Roosevelt"), ("Mars", "Venus"),
    ("Mercury", "Neptune"), ("Sahara", "Gobi"), ("Brazil", "Argentina"),
    ("Nissan", "Mazda"), ("Audi", "BMW"), ("Netflix", "Spotify"),
    ("Facebook", "Twitter"), ("Cairo", "Lagos"), ("Mumbai", "Delhi"),
    ("Seoul", "Bangkok"), ("Milan", "Barcelona"),
]
assert len(EN_PAIRS) == 50


def build_500_pairs():
    """Build 500 unique word-replacement pairs."""
    pairs = []
    for tmpl_id, template in TEMPLATES.items():
        for i, (wa, wb) in enumerate(EN_PAIRS):
            pairs.append({
                "word_a": wa, "word_b": wb,
                "text_a": template.format(word=wa),
                "text_b": template.format(word=wb),
                "template_id": tmpl_id,
                "template": template,
                "pair_idx_within_template": i,
            })
    assert len(pairs) == 500
    return pairs


if __name__ == "__main__":
    pairs = build_500_pairs()
    print(f"Generated {len(pairs)} pairs")
    print(f"Templates: {len(TEMPLATES)}")
    print(f"Pair base: {len(EN_PAIRS)}")
    print(f"First: {pairs[0]}")
    print(f"Last: {pairs[-1]}")
    # Uniqueness check
    unique = set((p["word_a"], p["word_b"], p["text_a"]) for p in pairs)
    print(f"Unique (word_a, word_b, text_a): {len(unique)}")
