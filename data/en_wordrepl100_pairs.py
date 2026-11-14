#!/usr/bin/env python3
"""EN word-replacement（英語固有名詞: London→Paris 等）の先頭100ペア.

en_wordrepl_500 の P0/P1 テンプレート × 50語ペア。日本語姓を含まないため
OOV名の音素化問題を排除した SpeechSplice vs VoiceCraft 比較用。
eval_utmos_speaker_fixed.py --pairs_module / 各採点スクリプト用ラッパー。
"""
from en_wordrepl_500 import build_500_pairs


def build_pairs():
    return build_500_pairs()[:100]
