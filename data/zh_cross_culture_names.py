"""ZH cross-culture name pair dataset for SpeechSplice direction-of-effect.

Source: common Chinese single-character family names (Hanzi)
Target: Japanese family names written in Hanzi (also pronounceable in Mandarin)

Both are valid Hanzi sequences whose Mandarin pinyin pronunciations are
entirely distinct, so a Whisper transcript or wav2vec2 phoneme decode can
distinguish source vs target reading without the polyphone caveat.

We instantiate 30 source x target pairs in 10 short Mandarin sentence
templates (entity at sentence-initial, mid, or final positions), giving
30 x 10 = 300 pairs.
"""

# Single-character or short Chinese family names
SOURCE_NAMES = [
    "王", "李", "张", "刘", "陈",
    "杨", "黄", "赵", "周", "吴",
    "徐", "孙", "马", "朱", "胡",
    "郭", "何", "高", "林", "罗",
    "郑", "梁", "谢", "宋", "唐",
    "许", "韩", "冯", "邓", "曹",
]

# Distinct Japanese family names (Hanzi), pronounced in Mandarin pinyin.
# Two-character names so that pronunciation is clearly different from any
# single-character SOURCE entry. Each TARGET pair is selected to have no
# Hanzi overlap with the corresponding SOURCE.
TARGET_NAMES = [
    "山田", "田中", "佐藤", "铃木", "高桥",
    "渡边", "伊藤", "中村", "小林", "加藤",
    "吉田", "山本", "佐佐木", "松本", "井上",
    "木村", "斋藤", "清水", "山口", "池田",
    "桥本", "石川", "前田", "藤田", "小川",
    "冈田", "后藤", "近藤", "石井", "村上",
]

assert len(SOURCE_NAMES) == 30 and len(TARGET_NAMES) == 30

# Sentence templates with {name} placeholder. Templates use natural Chinese
# (Mandarin) prose. Position varies (initial/mid/final) to mirror the EN
# cross-culture template set.
TEMPLATES = [
    "{name}今天到达了机场。",
    "我刚刚遇到了{name}。",
    "{name}写的那本书非常精彩。",
    "在会议上,{name}做了重要发言。",
    "经理询问{name}是否能参加。",
    "{name}的研究在去年获得了奖项。",
    "他和{name}一起去了餐厅。",
    "我们邀请{name}担任演讲嘉宾。",
    "上个月{name}搬到了上海。",
    "请把这份文件交给{name}。",
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
