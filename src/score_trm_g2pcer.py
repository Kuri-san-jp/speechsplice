"""pyopenjtalk-free TRM and G2P-CER Match scoring (Appendix G, JA N=300).

Same CLI, inputs, conditions, aggregation and output schema as
``score_trm_g2pcer.py`` (== TTS26 ``scripts/projection/score_unified_trm_g2pcer.py``,
which produced ``eval_results/_baselines_cc_n300_unified/final_results.json``),
with the two pyopenjtalk calls replaced by pykakasi (created 2026-09-27):

  score_trm_g2pcer.py L26-27  ``from reeval_strict_v2 import ...`` / ``import pyopenjtalk``
      -> ``from strict_v2_nopyojt import ...`` (no pyopenjtalk; import is blocked)
  strict_v2.py L53-61         TRM path 1: pyopenjtalk.g2p(kana=True) hiragana match
      -> pykakasi hiragana match (see strict_v2_nopyojt.py)
  score_trm_g2pcer.py L69-72  G2P-CER: ``pyopenjtalk.g2p(transcript[:200], kana=True)``
      -> ``g2p_hira(transcript[:200])`` (pykakasi hiragana), then the same
         normalize_kana + sliding-window CER <= 0.30

Everything else is unchanged:
  * target reading r* = pykakasi katakana of word_b (``to_katakana``), as before
  * transcripts: all whitespace (incl. U+3000) removed before scoring
  * TRM = strict v2 match (paths: pykakasi_hira / pykakasi / direct)
  * G2P-CER Match = TRM OR min_window CER(norm(G2P(y)), norm(r*)) <= 0.30
    (the OR with strict v2 is the original code's definition, kept as is)
  * Wilson 95% half-width in percentage points, N = rows with a transcript

Extra fields (added, do not change the summary): per-row ``trm_paths`` (every
path evaluated without short-circuit), ``g2p_kana_norm`` and ``best_cer``;
top-level ``provenance`` (pykakasi version, input md5s).

CPU only.  Never imports pyopenjtalk.
"""
import sys

if "pyopenjtalk" not in sys.modules:
    sys.modules["pyopenjtalk"] = None  # block pyopenjtalk for this process

import argparse, hashlib, json, os, re
from math import sqrt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pykakasi
from strict_v2_nopyojt import (check_strict_v2, normalize_kana, g2p_hira,
                               strict_v2_paths)

kks = pykakasi.kakasi()


def to_katakana(text):
    """Convert kanji/hiragana to katakana via pykakasi."""
    return "".join(r["kana"] for r in kks.convert(text))


def normalize_transcript(text):
    """Strip whitespace (incl. full-width) before G2P.

    Some ASR outputs (e.g., character-spaced Whisper word chunks) split kanji
    with spaces, which makes a G2P read name characters one by one.
    Removing all whitespace avoids this artifact without changing the
    semantic content.
    """
    if not text: return text
    return re.sub(r"\s+", "", text).replace("　", "")


def compute_cer(ref, hyp):
    """Char-level edit distance / |ref|."""
    if not ref: return 1.0
    a = list(ref); b = list(hyp)
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if x == y else 1))
        prev = cur
    return prev[-1] / max(1, len(ref))


def sliding_cer(transcript, expected_reading):
    """(best window CER, normalised G2P string) with pykakasi G2P."""
    try:
        trans_kana = g2p_hira(transcript[:200])
    except Exception:
        trans_kana = transcript
    expected_norm = normalize_kana(expected_reading)
    trans_norm = normalize_kana(trans_kana)
    if not expected_norm or not trans_norm:
        return None, trans_norm
    exp_len = len(expected_norm)
    best = 1.0
    for start in range(max(1, len(trans_norm) - exp_len + 1)):
        window = trans_norm[start:start + exp_len]
        best = min(best, compute_cer(expected_norm, window))
    return best, trans_norm


def check_lenient(transcript, expected_reading):
    """Lenient: strict_v2 OR sliding-window CER <= 0.30 (pykakasi G2P)."""
    sv2, _ = check_strict_v2(transcript, expected_reading)
    if sv2: return True, "strict"
    best, trans_norm = sliding_cer(transcript, expected_reading)
    if best is None:
        return False, "empty"
    return (best <= 0.30, "lenient" if best <= 0.30 else "no_match")


def wilson_ci_half(n, k, z=1.96):
    """Wilson 95% CI half-width in percentage points."""
    if n == 0: return 0
    p = k / n
    return z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) * 100


def _md5(path):
    if not path or not os.path.exists(path): return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl_b1_b7_json", required=True,
                    help="baseline_text_edit_ja_n300/results.json")
    ap.add_argument("--inj_json", required=True,
                    help="t5gemma_cc_wavs_ja_n300/checkpoint.json (SpeechSplice INJ transcripts)")
    ap.add_argument("--rome_json", default=None,
                    help="Optional ROME transcripts JSON (pair_idx-keyed)")
    ap.add_argument("--g2p_json", default=None,
                    help="Optional G2P Baseline transcripts JSON (pair_idx-keyed)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bl_data = json.load(open(args.bl_b1_b7_json))
    bl_rows = {r["pair_idx"]: r for r in bl_data["rows"]}
    inj_data = json.load(open(args.inj_json))
    inj_rows = {r["pair_idx"]: r for r in inj_data["rows"]}
    rome_rows = {}
    if args.rome_json and os.path.exists(args.rome_json):
        rome_data = json.load(open(args.rome_json))
        rome_rows = {r["pair_idx"]: r for r in rome_data.get("rows", [])}
    g2p_rows = {}
    if args.g2p_json and os.path.exists(args.g2p_json):
        g2p_data = json.load(open(args.g2p_json))
        g2p_rows = {r["pair_idx"]: r for r in g2p_data.get("rows", [])}

    cond_keys = {
        "baseline": ("bl_transcript", bl_rows),
        "g2p":      ("g2p_transcript", g2p_rows),
        "b1_surface_edit": ("b1_transcript", bl_rows),
        "text_hint": ("b7_transcript", bl_rows),
        "rome": ("rome_transcript", rome_rows),
        "speechsplice": ("t5_inj_transcript", inj_rows),
    }

    out_rows = []
    pair_ids = sorted(bl_rows.keys())
    for pidx in pair_ids:
        bl = bl_rows[pidx]
        word_a = bl["word_a"]; word_b = bl["word_b"]
        target_kata = to_katakana(word_b)
        row = {"pair_idx": pidx, "word_a": word_a, "word_b": word_b,
               "target_kata": target_kata}
        for cname, (key, src) in cond_keys.items():
            r = src.get(pidx)
            if r is None or key not in r or not r[key]:
                row[cname] = {"transcript": None, "trm": None, "g2p_cer": None}
                continue
            tr = normalize_transcript(r[key])
            sv2, sv2_m = check_strict_v2(tr, target_kata)
            lenient, lenient_m = check_lenient(tr, target_kata)
            best, g2p_norm = sliding_cer(tr, target_kata)
            row[cname] = {
                "transcript": tr,
                "trm": bool(sv2),
                "trm_method": sv2_m,
                "g2p_cer_match": bool(lenient),
                "g2p_cer_type": lenient_m,
                "trm_paths": strict_v2_paths(tr, target_kata),
                "g2p_kana_norm": g2p_norm,
                "best_cer": None if best is None else round(best, 4),
            }
        out_rows.append(row)

    summary = {}
    for cname in cond_keys:
        rows_with = [r[cname] for r in out_rows if r[cname]["transcript"] is not None]
        n = len(rows_with)
        if n == 0:
            summary[cname] = {"n": 0, "trm_pct": None, "trm_ci_half": None,
                              "g2p_cer_match_pct": None, "g2p_cer_match_ci_half": None}
            continue
        n_trm = sum(1 for r in rows_with if r["trm"])
        n_lenient = sum(1 for r in rows_with if r["g2p_cer_match"])
        summary[cname] = {
            "n": n,
            "trm_n": n_trm,
            "trm_pct": round(100 * n_trm / n, 2),
            "trm_ci_half": round(wilson_ci_half(n, n_trm), 2),
            "g2p_cer_match_n": n_lenient,
            "g2p_cer_match_pct": round(100 * n_lenient / n, 2),
            "g2p_cer_match_ci_half": round(wilson_ci_half(n, n_lenient), 2),
        }

    try:
        import importlib.metadata as _md
        pk_ver = _md.version("pykakasi")
    except Exception:
        pk_ver = getattr(pykakasi, "__version__", None)
    provenance = {
        "scorer": os.path.abspath(__file__),
        "scorer_md5": _md5(os.path.abspath(__file__)),
        "strict_v2_module_md5": _md5(os.path.join(_HERE, "strict_v2_nopyojt.py")),
        "pykakasi_version": pk_ver,
        "pyopenjtalk_blocked": sys.modules.get("pyopenjtalk", "absent") is None,
        "inputs": {k: {"path": v, "md5": _md5(v)} for k, v in {
            "bl_b1_b7_json": args.bl_b1_b7_json, "inj_json": args.inj_json,
            "rome_json": args.rome_json, "g2p_json": args.g2p_json}.items()},
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "provenance": provenance, "rows": out_rows},
                  f, ensure_ascii=False, indent=2)

    print(f"\n=== Unified scoring (pyopenjtalk-free) on cross-culture JA (N={len(out_rows)}) ===")
    print(f"{'Condition':<20}{'N':>5}{'TRM%':>10}{'CI±':>9}{'G2P-CER%':>14}{'CI±':>9}")
    print("-" * 70)
    for cname in cond_keys:
        s = summary[cname]
        if s["n"] == 0:
            print(f"{cname:<20}{0:>5}    {'MISSING':>12}{'':>23}")
        else:
            print(f"{cname:<20}{s['n']:>5}{s['trm_pct']:>9.2f}%{s['trm_ci_half']:>9.2f}{s['g2p_cer_match_pct']:>12.2f}%{s['g2p_cer_match_ci_half']:>9.2f}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
