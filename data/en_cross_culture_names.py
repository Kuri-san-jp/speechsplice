"""EN cross-cultural name pairs for SpeechSplice (iii) decoder sensitivity test.

Each pair is a (source, target) name where:
- source: English/Western-origin first name (Whisper recognises in EN)
- target: Cross-cultural (Japanese-style) family name (very different surface
  and pronunciation, so Whisper transcript is unambiguous)

Used by eval_en_crossculture_acoustic.py to test whether each TTS model's
decoder can be steered toward an arbitrary target reading via encoder
hidden-state injection.

Builds 30 (source, target) name pairs x 10 sentence templates = 300 pairs.
"""

SOURCE_NAMES = [
    "John", "Mary", "David", "Sarah", "Michael", "Emily",
    "James", "Jessica", "Robert", "Ashley", "William", "Amanda",
    "Christopher", "Jennifer", "Joseph", "Stephanie", "Daniel", "Nicole",
    "Matthew", "Elizabeth", "Andrew", "Heather", "Joshua", "Tiffany",
    "Anthony", "Michelle", "Steven", "Amy", "Mark", "Melissa",
]

TARGET_NAMES = [
    "Yamada", "Tanaka", "Suzuki", "Watanabe", "Saito", "Sato",
    "Takahashi", "Ito", "Yamamoto", "Nakamura", "Kobayashi", "Kato",
    "Yoshida", "Yamaguchi", "Matsumoto", "Inoue", "Kimura", "Hayashi",
    "Shimizu", "Yamazaki", "Mori", "Abe", "Ikeda", "Hashimoto",
    "Yamashita", "Ishikawa", "Nakajima", "Maeda", "Fujita", "Goto",
]

TEMPLATES = [
    "{name} arrived at the airport this morning.",
    "I just met {name} for the first time last week.",
    "The book recommended by {name} was excellent.",
    "{name} said the meeting will be at three o'clock.",
    "Have you seen {name} recently?",
    "{name} is going to give a presentation tomorrow.",
    "We had lunch with {name} at the new cafe.",
    "{name} called me about the project deadline.",
    "I think {name} would enjoy this article.",
    "{name} just finished writing a new book.",
]


def build_300_pairs():
    pairs = []
    idx = 0
    for src, tgt in zip(SOURCE_NAMES, TARGET_NAMES):
        for tmpl in TEMPLATES:
            text_a = tmpl.format(name=src)
            text_b = tmpl.format(name=tgt)
            pairs.append({
                "pair_idx": idx,
                "text_a": text_a,
                "text_b": text_b,
                "word_a": src,
                "word_b": tgt,
            })
            idx += 1
    return pairs


if __name__ == "__main__":
    pairs = build_300_pairs()
    print(f"Total pairs: {len(pairs)}")
    print("First 3 pairs:")
    for p in pairs[:3]:
        print(f"  [{p['pair_idx']}] {p['word_a']} -> {p['word_b']}")
        print(f"    text_a: {p['text_a']}")
        print(f"    text_b: {p['text_b']}")
