"""Aggregate the automatic-evaluation half of Table 1 (cross-lingual control).

Reproduces, from the pre-computed JSONs shipped in eval_results/:

  Lang | Whisper-INJ (%) | IPA change (%) | UTMOS (INJ)

Run:
  REPO_ROOT=$PWD python3 scripts/aggregate_table1.py

Cells that the shipped JSONs do not contain are printed as "n/a" and listed in
the footer. They require re-running the synthesis + scoring pipeline on a GPU
(UTMOS on the baseline waveforms, the absolute WavLM-SV cosines, and the
|dF0| ratio are computed from audio that is not redistributed here).

The human-evaluation half of Table 1 (INJ-selection rate, GEE, MOS) comes from
scripts/aggregate_human_eval.py.
"""
import json, math, os

REPO = os.environ.get("REPO_ROOT", os.getcwd())
T1 = os.path.join(REPO, "eval_results/table1_cross_lingual")
T2 = os.path.join(REPO, "eval_results/table2_system_comparison")
T3 = os.path.join(REPO, "eval_results/table3_cross_arch")


def wilson_half(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    den = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den


def jload(path):
    with open(path) as f:
        return json.load(f)


def whisper_inj(lang):
    """Whisper-INJ = % of injected samples whose transcript contains the target reading."""
    if lang == "en":
        # The EN cell of Table 1 is the same run as the SpeechSplice row of Table 2,
        # scored by Whisper large-v3 with target containment. Note that
        # {en}_judge.json carries a separate g2p-containment pass (291/300) and is
        # NOT the source of the manuscript's EN Whisper-INJ value.
        s = jload(os.path.join(T2, "whisper_inj_large_v3.json"))["systems"]["speechsplice"]
        return s["k"], s["n"]
    s = jload(os.path.join(T1, f"{lang}_judge.json"))["summary"]["whisper_inj"]
    n = jload(os.path.join(T1, f"{lang}_judge.json"))["summary"]["n"]
    return s["k"], n


def ipa_change(lang):
    """IPA change rate from the wav2vec2 phoneme recognizer."""
    f = os.path.join(T3, f"phoneme_per_{lang}.json")
    if not os.path.exists(f):
        return None
    return jload(f).get("summary", {}).get("ipa_change_rate_pct")


def utmos_inj(lang):
    f = os.path.join(T1, f"{lang}_judge.json")
    return jload(f)["summary"].get("utmos_mean")


print(f"{'Lang':<6}{'Whisper-INJ (%)':>20}{'IPA change (%)':>18}{'UTMOS (INJ)':>14}")
print("-" * 58)
missing = []
for lang in ["en", "ja", "zh"]:
    k, n = whisper_inj(lang)
    w = f"{100*k/n:6.2f} ± {wilson_half(k, n)*100:.2f}"
    ipa = ipa_change(lang)
    i_s = f"{ipa:6.2f}" if ipa is not None else "n/a"
    u = utmos_inj(lang)
    u_s = f"{u:6.4f}" if u is not None else "n/a"
    if u is None:
        missing.append(f"UTMOS (INJ) for {lang.upper()}")
    print(f"{lang.upper():<6}{w:>20}{i_s:>18}{u_s:>14}")

print()
print("Not derivable from the shipped JSONs (need the GPU synthesis/scoring run):")
for m in ["UTMOS on the baseline waveforms (all languages)"] + missing + [
        "absolute WavLM-SV cosines (BL--INJ and INJ--voice prompt)",
        "|dF0| ratio"]:
    print(f"  - {m}")
print()
print("Shipped BL--INJ speaker-similarity tests (used in Section 5.2):")
ss = jload(os.path.join(T1, "speaker_similarity_tests.json"))
for key in ["en_ss", "ja_ss", "zh_ss"]:
    r = ss[key]
    print(f"  {key:<8} n={r['n']:<4} delta={r['delta_mean']:+.4f} "
          f"±{r['ci95']:.4f}  paired Wilcoxon p={r['wilcoxon_p']:.4f}")
