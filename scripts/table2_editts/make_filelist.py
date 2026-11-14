#!/usr/bin/env python3
"""Build the filelist for EdiTTS content replacement (runs on CPU).

Format: "text_a with | word_a | ... #text_b with | word_b | ..."
The first 100 cross-culture pairs (the same samples as the vc100 experiment).

Usage:
  python3 scripts/editts/make_filelist.py
"""
import os
import sys

PROJECT_ROOT = "."
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
from en_cross_culture_names import build_300_pairs

OUT = os.path.join(PROJECT_ROOT, "eval_results", "editts_edit_en_n100", "filelist.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

pairs = build_300_pairs()[:100]
lines = []
for p in pairs:
    assert p["text_a"].count(p["word_a"]) == 1, p
    assert p["text_b"].count(p["word_b"]) == 1, p
    a = p["text_a"].replace(p["word_a"], f"| {p['word_a']} |")
    b = p["text_b"].replace(p["word_b"], f"| {p['word_b']} |")
    lines.append(f"{a} #{b}")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[Done] {len(lines)} lines -> {OUT}")
print("example:", lines[0])
