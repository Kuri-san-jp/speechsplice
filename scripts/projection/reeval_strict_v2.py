#!/usr/bin/env python3
"""Re-evaluate all Phase 2-4 data with Strict v2 metric and output augmented JSON."""
import json
import os
import re
import pykakasi
pyopenjtalk = None  # (release) not used: the path that needs it fails and is skipped

kks = pykakasi.kakasi()


def normalize_kana(text):
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x3041 <= cp <= 0x3096:
            result.append(chr(cp + 0x60))
        else:
            result.append(ch)
    text = "".join(result)
    text = re.sub(r"[^ァ-ヶー]", "", text)
    text = text.replace("ヅ", "ズ").replace("ヂ", "ジ")
    vowel_map = {
        "ア": "ア", "カ": "ア", "サ": "ア", "タ": "ア", "ナ": "ア", "ハ": "ア", "マ": "ア", "ヤ": "ア", "ラ": "ア", "ワ": "ア", "ガ": "ア", "ザ": "ア", "ダ": "ア", "バ": "ア", "パ": "ア",
        "イ": "イ", "キ": "イ", "シ": "イ", "チ": "イ", "ニ": "イ", "ヒ": "イ", "ミ": "イ", "リ": "イ", "ギ": "イ", "ジ": "イ", "ビ": "イ", "ピ": "イ",
        "ウ": "ウ", "ク": "ウ", "ス": "ウ", "ツ": "ウ", "ヌ": "ウ", "フ": "ウ", "ム": "ウ", "ユ": "ウ", "ル": "ウ", "グ": "ウ", "ズ": "ウ", "ブ": "ウ", "プ": "ウ",
        "エ": "エ", "ケ": "エ", "セ": "エ", "テ": "エ", "ネ": "エ", "ヘ": "エ", "メ": "エ", "レ": "エ", "ゲ": "エ", "ゼ": "エ", "デ": "エ", "ベ": "エ", "ペ": "エ",
        "オ": "オ", "コ": "オ", "ソ": "オ", "ト": "オ", "ノ": "オ", "ホ": "オ", "モ": "オ", "ヨ": "オ", "ロ": "オ", "ヲ": "オ", "ゴ": "オ", "ゾ": "オ", "ド": "オ", "ボ": "オ", "ポ": "オ",
    }
    normalized = []
    for i, ch in enumerate(text):
        if ch == "ー" and i > 0:
            prev = text[i - 1]
            if prev in vowel_map:
                normalized.append(vowel_map[prev])
            else:
                normalized.append(ch)
        else:
            normalized.append(ch)
    return "".join(normalized)


def pykakasi_to_kana(text):
    result = kks.convert(text)
    return "".join([item["kana"] for item in result])


def katakana_to_hiragana(text):
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text)


def check_strict_v2(transcript, expected_reading):
    # Path 1: pyopenjtalk g2p
    try:
        trans_kana = pyopenjtalk.g2p(transcript[:200], kana=True)
        trans_hira = katakana_to_hiragana(trans_kana)
        expected_hira = katakana_to_hiragana(expected_reading)
        if expected_hira in trans_hira:
            return True, "g2p"
    except Exception:
        pass
    # Path 2: pykakasi
    try:
        trans_pykakasi = pykakasi_to_kana(transcript[:200])
        trans_norm = normalize_kana(trans_pykakasi)
        expected_norm = normalize_kana(expected_reading)
        if expected_norm in trans_norm:
            return True, "pykakasi"
    except Exception:
        pass
    # Path 3: direct katakana
    trans_norm = normalize_kana(transcript[:200])
    expected_norm = normalize_kana(expected_reading)
    if expected_norm in trans_norm:
        return True, "direct"
    return False, "no_match"


def process_file(input_path, output_path, condition_keys):
    """Process a single eval JSON file and add strict_v2 fields."""
    with open(input_path, "r") as f:
        results = json.load(f)

    total = 0
    v2_matches = 0

    for r in results:
        reading_katakana = r.get("reading_katakana", "")
        if not reading_katakana:
            continue

        for seed_key, conds in r["conditions"].items():
            for cond_name in condition_keys:
                if cond_name not in conds:
                    continue
                result = conds[cond_name]
                transcript = result.get("transcript", "")
                if not transcript:
                    result["match_strict_v2"] = False
                    result["v2_method"] = "no_transcript"
                    continue

                v2, method = check_strict_v2(transcript, reading_katakana)
                result["match_strict_v2"] = v2
                result["v2_method"] = method
                total += 1
                if v2:
                    v2_matches += 1

    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  {os.path.basename(output_path)}: {v2_matches}/{total} ({100*v2_matches/total:.1f}%) strict v2 matches")


def main():
    files = [
        {
            "input": "eval_results/vector_db_eval/vector_db_eval.json",
            "output": "eval_results/vector_db_eval/vector_db_eval_v2.json",
            "keys": ["baseline", "db_injection", "oracle_2pass"],
        },
        {
            "input": "eval_results/phase3_eval/phase3_eval.json",
            "output": "eval_results/phase3_eval/phase3_eval_v2.json",
            "keys": ["baseline", "phase3"],
        },
        {
            "input": "eval_results/phase4_eval/phase4_eval.json",
            "output": "eval_results/phase4_eval/phase4_eval_v2.json",
            "keys": ["baseline", "phase3", "phase4"],
        },
    ]

    print("=== Strict v2 Re-evaluation ===")
    for f in files:
        if not os.path.exists(f["input"]):
            print(f"  SKIP: {f['input']} not found")
            continue
        print(f"\nProcessing: {f['input']}")
        process_file(f["input"], f["output"], f["keys"])

    # Summary per condition for Phase 2 (DB-only entities)
    print("\n=== Phase 2 Summary (DB-only entities, n=135) ===")
    with open("eval_results/vector_db_eval/vector_db_eval_v2.json", "r") as f:
        results = json.load(f)

    for cond in ["baseline", "db_injection", "oracle_2pass"]:
        v1_count = 0
        v2_count = 0
        lenient_count = 0
        total = 0
        methods = {}
        for r in results:
            if not r["in_db"]:
                continue
            for seed_key, conds in r["conditions"].items():
                if cond not in conds:
                    continue
                result = conds[cond]
                if not result.get("transcript"):
                    continue
                total += 1
                if result.get("match_strict", False):
                    v1_count += 1
                if result.get("match_strict_v2", False):
                    v2_count += 1
                    m = result.get("v2_method", "unknown")
                    methods[m] = methods.get(m, 0) + 1
                if result.get("match_lenient", False):
                    lenient_count += 1

        print(f"  {cond}: v1={v1_count}/{total} ({100*v1_count/total:.1f}%), "
              f"v2={v2_count}/{total} ({100*v2_count/total:.1f}%), "
              f"lenient={lenient_count}/{total} ({100*lenient_count/total:.1f}%)")
        print(f"    v2 methods: {methods}")


if __name__ == "__main__":
    main()
