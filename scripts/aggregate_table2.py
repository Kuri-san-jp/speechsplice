"""Aggregate Table 2: comparison with VoiceCraft and EdiTTS (English split).

Reproduces the Whisper-INJ column of Table 2 and the speaker-identity
statistics quoted in Section 5.2, from:

  eval_results/table2_system_comparison/whisper_inj_large_v3.json
  eval_results/table1_cross_lingual/speaker_similarity_tests.json

All three systems are scored by the same judge (Whisper large-v3, greedy,
target-word containment) on the same 300-pair English split.

Run:
  REPO_ROOT=$PWD python3 scripts/aggregate_table2.py

The UTMOS, |dF0| and |d duration| columns of Table 2 are computed from the
synthesized waveforms, which are not redistributed; they require re-running
the generation scripts under scripts/table2_voicecraft/ and
scripts/table2_editts/ on a GPU.
"""
import json, math, os

REPO = os.environ.get("REPO_ROOT", os.getcwd())
F = os.path.join(REPO, "eval_results/table2_system_comparison/whisper_inj_large_v3.json")
SS = os.path.join(REPO, "eval_results/table1_cross_lingual/speaker_similarity_tests.json")


def wilson_half(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den


with open(F) as f:
    d = json.load(f)

NICE = {"speechsplice": "SpeechSplice (ours)",
        "vc_tts": "VoiceCraft zero-shot TTS",
        "editts": "EdiTTS"}
SSKEY = {"speechsplice": "en_ss", "vc_tts": "vc_tts", "editts": "editts"}

with open(SS) as f:
    ss = json.load(f)

print(f"judge: {d['judge']}")
print()
print(f"{'System':<26}{'N':>5}{'Whisper-INJ (%)':>20}"
      f"{'SS delta (INJ-prompt)':>24}{'Wilcoxon p':>13}")
print("-" * 88)
for key in ["speechsplice", "vc_tts", "editts"]:
    s = d["systems"][key]
    k, n = s["k"], s["n"]
    w = f"{100*k/n:6.2f} ± {wilson_half(k, n)*100:.2f}"
    r = ss.get(SSKEY[key])
    if r is None:
        sd, pv = "n/a", "n/a"
    else:
        sd = f"{r['delta_mean']:+.4f} ± {r['ci95']:.4f}"
        pv = f"{r['wilcoxon_p']:.4f}"
    print(f"{NICE[key]:<26}{n:>5}{w:>20}{sd:>24}{pv:>13}")

print()
print("McNemar tests on the paired Whisper-INJ outcomes:")
for k, label in [("mcnemar_speechsplice_vs_vc_tts", "SpeechSplice vs VoiceCraft"),
                 ("mcnemar_speechsplice_vs_editts", "SpeechSplice vs EdiTTS")]:
    m = d[k]
    print(f"  {label:<30} b={m['b']:<4} c={m['c']:<4} p={m['p']:.3e}")

print()
print("Columns not derivable from the shipped JSONs (need the GPU generation run):")
for m in ["UTMOS (BL) and UTMOS (injected/edited) with their Wilcoxon p",
          "|dF0| ratio", "|d duration| (s)",
          "absolute WavLM-SV cosines (the shipped file holds BL--INJ deltas)"]:
    print(f"  - {m}")
