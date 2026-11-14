"""Aggregate Table 2: comparison with VoiceCraft and EdiTTS (English split).

Reproduces every column of Table 2 from eval_results/table2_system_comparison/
(and, for the SpeechSplice row, eval_results/table1_cross_lingual/):

  * Whisper-INJ: Whisper large-v3, greedy, target-word containment, Wilson 95%
    half-width (whisper_inj_large_v3.json); McNemar tests on the paired verdicts.
  * UTMOS: mean +- 1.96 * SD / sqrt(n) of the per-clip scores, and the two-sided
    paired Wilcoxon signed-rank p value between BL and the injected/edited clip.
    SpeechSplice: en_results.json (BL) and en_judge.json (INJ); VoiceCraft:
    voicecraft_metrics.json rows[].bl/inj.utmos; EdiTTS: editts_utmos.json
    rows[]["gradtts-1"] (BL, direct Grad-TTS synthesis) and rows[]["EdiTTS"].
  * Speaker similarity: WavLM-SV [mean, 1.96*SE, SD] summaries
    (table1_cross_lingual/speaker_sim_en.json, speaker_sim_systems.json).
  * |dF0| ratio: abs_diff / anchor_abs_diff (table1_cross_lingual/f0_summary.json).
  * |d duration|: mean +- 1.96 * SE of |dur(INJ or edited) - dur(BL)| from the
    per-pair wav-header durations in duration.json.

Run:
  REPO_ROOT=$PWD python3 scripts/aggregate_table2.py
"""
import json, math, os

from scipy.stats import wilcoxon

REPO = os.environ.get("REPO_ROOT", os.getcwd())
T1 = os.path.join(REPO, "eval_results/table1_cross_lingual")
T2 = os.path.join(REPO, "eval_results/table2_system_comparison")


def jload(path):
    with open(path) as f:
        return json.load(f)


def wilson_half(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den


def mean_ci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def utmos_pairs(system):
    if system == "speechsplice":
        bl = [r["bl_utmos"] for r in jload(os.path.join(T1, "en_results.json"))["rows"]]
        inj = [r["utmos"] for r in jload(os.path.join(T1, "en_judge.json"))["rows"]]
    elif system == "vc_tts":
        rows = jload(os.path.join(T2, "voicecraft_metrics.json"))["rows"]
        bl = [r["bl"]["utmos"] for r in rows]
        inj = [r["inj"]["utmos"] for r in rows]
    else:
        rows = jload(os.path.join(T2, "editts_utmos.json"))["rows"]
        bl = [r["gradtts-1"]["utmos"] for r in rows]
        inj = [r["EdiTTS"]["utmos"] for r in rows]
    return bl, inj


wi = jload(os.path.join(T2, "whisper_inj_large_v3.json"))
f0 = jload(os.path.join(T1, "f0_summary.json"))
ss_en = jload(os.path.join(T1, "speaker_sim_en.json"))
ss_sys = jload(os.path.join(T2, "speaker_sim_systems.json"))
dur = jload(os.path.join(T2, "duration.json"))["systems"]

NICE = {"speechsplice": "SpeechSplice (ours)", "vc_tts": "VoiceCraft zero-shot TTS", "editts": "EdiTTS"}
DURKEY = {"speechsplice": "SpeechSplice", "vc_tts": "VoiceCraft", "editts": "EdiTTS"}

print(f"judge: {wi['judge']}")
print()
print(f"{'System':<26}{'Whisper-INJ (%)':>17}{'UTMOS BL':>13}{'UTMOS INJ/edit':>16}{'p':>7}"
      f"{'SS BL--INJ':>13}{'SS INJ--prompt':>16}{'|dF0| ratio':>13}{'|d duration| (s)':>18}")
print("-" * 139)
for key in ["speechsplice", "vc_tts", "editts"]:
    s = wi["systems"][key]
    k, n = s["k"], s["n"]
    bl, inj = utmos_pairs(key)
    bm, bh = mean_ci(bl)
    im, ih = mean_ci(inj)
    p = wilcoxon(bl, inj).pvalue
    if key == "speechsplice":
        pair, prm = ss_en["bl_inj_pair"], ss_en["sim_prompt_inj"]
    elif key == "vc_tts":
        pair, prm = ss_sys["vc_tts"]["bl_inj_pair"], ss_sys["vc_tts"]["sim_prompt_inj"]
    else:
        pair, prm = ss_sys["editts"]["raw_edit_pair"], None
    f = f0[key]
    d = [r["abs_delta_dur_s"] for r in dur[DURKEY[key]]["rows"]]
    dm, dh = mean_ci(d) if any(d) else (0.0, 0.0)
    prm_s = f"{prm[0]:.2f} ± {prm[1]:.2f}" if prm else "--"
    print(f"{NICE[key]:<26}{100*k/n:9.2f} ± {100*wilson_half(k, n):.2f}"
          f"{bm:8.2f} ± {bh:.2f}{im:11.2f} ± {ih:.2f}{p:7.2f}"
          f"{pair[0]:8.2f} ± {pair[1]:.2f}{prm_s:>16}"
          f"{f['abs_diff'] / f['anchor_abs_diff']:13.2f}{dm:13.2f} ± {dh:.2f}")

print()
print("McNemar tests on the paired Whisper-INJ outcomes:")
for k, label in [("mcnemar_speechsplice_vs_vc_tts", "SpeechSplice vs VoiceCraft"),
                 ("mcnemar_speechsplice_vs_editts", "SpeechSplice vs EdiTTS")]:
    m = wi[k]
    print(f"  {label:<30} b={m['b']:<4} c={m['c']:<4} p={m['p']:.3e}")
print()
print("Paired change of the INJ--prompt similarity (Section 5.2): see scripts/aggregate_table1.py")
