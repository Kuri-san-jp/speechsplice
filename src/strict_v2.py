#!/usr/bin/env python3
"""pyopenjtalk-free TRM (strict v2) helpers.

Drop-in replacement for ``strict_v2.py`` (== TTS26 ``scripts/projection/
reeval_strict_v2.py``) that never imports pyopenjtalk.  Created 2026-09-27
for the Appendix G (pronunciation-baselines) redo.

What changed relative to strict_v2.py
-------------------------------------
* ``import pyopenjtalk`` (strict_v2.py L7) is removed.  Importing this module
  installs ``sys.modules["pyopenjtalk"] = None`` when pyopenjtalk has not been
  imported yet, so any later ``import pyopenjtalk`` in the same process raises
  ImportError instead of silently running it.
* Path 1 of ``check_strict_v2`` (strict_v2.py L53-61) used
  ``pyopenjtalk.g2p(transcript[:200], kana=True)`` and compared hiragana
  strings without normalisation.  It is replaced by the pykakasi hiragana
  reading of the transcript (``g2p_hira``) compared the same way (label
  ``"pykakasi_hira"``).  Paths 2 (pykakasi + normalize_kana) and 3 (raw
  transcript + normalize_kana) are unchanged, byte for byte.

Kana conversion used here
-------------------------
``g2p_hira(text) = "".join(item["hira"] for item in pykakasi.kakasi().convert(text))``

* kanji -> dictionary (orthographic) reading in hiragana, katakana ->
  hiragana, hiragana unchanged, digits / Latin letters / punctuation are passed
  through unchanged (they are later dropped by ``normalize_kana``).
* ``normalize_kana`` (unchanged): hiragana -> katakana, keep only [ァ-ヶー],
  ヅ->ズ, ヂ->ジ, and a long-vowel mark ー becomes the vowel of the preceding
  mora (トーキョー -> トオキョオ).

Known differences from pyopenjtalk.g2p(kana=True)
-------------------------------------------------
pyopenjtalk returns the *pronunciation* form after MeCab (NAIST-jdic)
analysis; pykakasi returns the *orthographic* reading from its own
dictionary with longest-match segmentation.  Consequences:

* particles: は -> pyopenjtalk ワ / pykakasi ハ, へ -> エ / ヘ, を -> オ / ヲ.
* long vowels: pyopenjtalk writes ー (トーキョー, サトー); pykakasi spells the
  vowel (とうきょう, さとう).  After normalize_kana these become トオキョオ vs
  トウキョウ, i.e. still different for おう/えい spellings.
* digits and Latin letters: pyopenjtalk reads them (3 -> サン, A -> エー);
  pykakasi leaves them as-is and normalize_kana deletes them.
* context-dependent readings (rendaku, 数詞+助数詞, 名/姓 disambiguation) follow
  MeCab in pyopenjtalk and the pykakasi dictionary here; unseen kanji
  sequences may be read character by character by pykakasi.

The target readings of Appendix G were already pykakasi katakana
(``to_katakana(word_b)`` in the scorer), so only the transcript side of the
comparison changes.
"""
import re
import sys

if "pyopenjtalk" not in sys.modules:
    sys.modules["pyopenjtalk"] = None  # block any later `import pyopenjtalk`

import pykakasi

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


def g2p_hira(text):
    """pykakasi hiragana reading (replacement for pyopenjtalk.g2p(kana=True))."""
    return "".join(item["hira"] for item in kks.convert(text))


def katakana_to_hiragana(text):
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text)


def strict_v2_paths(transcript, expected_reading):
    """Evaluate every path (no short-circuit). Returns {path: bool}."""
    t = transcript[:200]
    out = {}
    # Path 1' (replaces pyopenjtalk): pykakasi hiragana, raw hiragana compare
    try:
        out["pykakasi_hira"] = katakana_to_hiragana(expected_reading) in g2p_hira(t)
    except Exception:
        out["pykakasi_hira"] = False
    # Path 2 (unchanged): pykakasi katakana + normalize_kana
    try:
        out["pykakasi"] = normalize_kana(expected_reading) in normalize_kana(pykakasi_to_kana(t))
    except Exception:
        out["pykakasi"] = False
    # Path 3 (unchanged): raw transcript + normalize_kana
    out["direct"] = normalize_kana(expected_reading) in normalize_kana(t)
    return out


def check_strict_v2(transcript, expected_reading):
    # Path 1': pykakasi hiragana (was: pyopenjtalk g2p)
    try:
        trans_hira = g2p_hira(transcript[:200])
        expected_hira = katakana_to_hiragana(expected_reading)
        if expected_hira in trans_hira:
            return True, "pykakasi_hira"
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
