#!/usr/bin/env python3
"""EN word-replacement の先頭300ペア（P0〜P5 × 50語ペア）.

先頭200は en_wordrepl200_pairs と同一（生成はresumeで再利用）。
N=300 比較実験（SS / VoiceCraft / EdiTTS）用。
"""
from en_wordrepl_500 import build_500_pairs


def build_pairs():
    return build_500_pairs()[:300]
