#!/usr/bin/env python3
"""EN word-replacement（英語固有名詞）の先頭200ペア.

en_wordrepl_500 の P0〜P3 テンプレート × 50語ペア。先頭100は
en_wordrepl100_pairs と同一（生成はresumeで再利用可能）。
"""
from en_wordrepl_500 import build_500_pairs


def build_pairs():
    return build_500_pairs()[:200]
