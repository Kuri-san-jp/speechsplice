"""JA cross-culture name pair dataset for SpeechSplice direction-of-effect.

Source: Chinese family names written in Hanzi, embedded in Japanese sentences.
Target: Japanese family names written in Hanzi, embedded in identical
Japanese templates. Both are Hanzi tokens whose Japanese pronunciations
(on/kun yomi) are entirely distinct, so a Whisper transcript can
distinguish source vs target reading.

We instantiate 30 source x target pairs in 10 short Japanese sentence
templates, giving 30 x 10 = 300 pairs.
"""

# Chinese single-character family names (used as SOURCE in JA sentences).
# These are recognisable as Chinese surnames and have a Japanese on-yomi
# reading distinct from the target Japanese family names.
SOURCE_NAMES = [
    "王", "李", "張", "劉", "陳",
    "楊", "黄", "趙", "周", "呉",
    "徐", "孫", "馬", "朱", "胡",
    "郭", "何", "高", "林", "羅",
    "鄭", "梁", "謝", "宋", "唐",
    "許", "韓", "馮", "鄧", "曹",
]

# Japanese family names in Hanzi (different surface form from any SOURCE).
TARGET_NAMES = [
    "山田", "田中", "佐藤", "鈴木", "高橋",
    "渡辺", "伊藤", "中村", "小林", "加藤",
    "吉田", "山本", "佐々木", "松本", "井上",
    "木村", "斎藤", "清水", "山口", "池田",
    "橋本", "石川", "前田", "藤田", "小川",
    "岡田", "後藤", "近藤", "石井", "村上",
]

assert len(SOURCE_NAMES) == 30 and len(TARGET_NAMES) == 30

# Natural Japanese sentence templates. {name} is filled with either a
# Chinese surname (text_a / SOURCE) or a Japanese surname (text_b / TARGET).
TEMPLATES = [
    "{name}さんは今日空港に到着しました。",
    "私は先週{name}さんに会いました。",
    "{name}さんが書いた本はとても素晴らしいです。",
    "会議で{name}さんは重要な発言をしました。",
    "部長は{name}さんが参加できるかを尋ねました。",
    "{name}さんの研究は昨年賞を受賞しました。",
    "彼は{name}さんと一緒にレストランへ行きました。",
    "私たちは{name}さんを講演者として招待しました。",
    "先月{name}さんは大阪へ引っ越しました。",
    "この書類を{name}さんに渡してください。",
]


def build_300_pairs():
    """Return list of 300 dicts: {pair_idx, text_a, text_b, word_a, word_b}.

    For each (source, target) pair (matched by index 0..29), instantiate the
    pair in all 10 templates. text_a holds the Chinese surname (source);
    text_b holds the Japanese surname (target). word_a/word_b are the bare
    name strings used for forced-alignment and entity-token lookup.
    """
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
    for p in pairs[:6]:
        print(p)
