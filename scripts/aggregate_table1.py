"""Aggregate the automatic-evaluation columns of Table 1 (cross-lingual control).

Reproduces, from the files shipped in eval_results/table1_cross_lingual/ and
eval_results/table2_system_comparison/:

  Lang | Whisper-INJ (%) | UTMOS BL | UTMOS INJ | Speaker similarity BL--INJ |
       INJ--prompt | |dF0| ratio

and the paired speaker-similarity tests quoted in Sections 5.1 and 5.2.

  * Whisper-INJ: % of INJ clips whose normalized transcript contains the target
    reading, with the Wilson 95% half-width. EN is scored by Whisper large-v3
    (whisper_inj_large_v3.json, the same run as the SpeechSplice row of
    Table 2), JA by Furigana Whisper (ja_judge.json), and ZH by Whisper large-v3
    after pinyin normalization (zh_judge.json).
  * UTMOS: mean +- 1.96 * SD / sqrt(n) over the 300 per-pair scores. EN BL comes
    from en_results.json and EN INJ from en_judge.json (the final set: 284
    equal-length pairs plus the 16 unequal-length pairs regenerated with the
    variable-length splice); JA and ZH come from {ja,zh}_results.json.
  * Speaker similarity: WavLM-SV cosine, stored as [mean, 1.96*SE, SD] in
    speaker_sim_{en,ja,zh}.json (bl_inj_pair = BL--INJ, sim_prompt_inj =
    INJ--prompt). Per-pair cosines were not stored.
  * |dF0| ratio: abs_diff / anchor_abs_diff in f0_summary.json, i.e. the mean
    |F0(INJ) - F0(BL)| over the mean |F0(BL_i) - F0(BL_{i+7})| between two
    baseline utterances of different sentences (pyworld Harvest, voiced
    frames). Per-pair F0 values were not stored.

Run:
  REPO_ROOT=$PWD python3 scripts/aggregate_table1.py

The human-evaluation columns of Table 1 (INJ selected, GEE z, MOS) come from
scripts/aggregate_human_eval.py.
"""
import json, math, os

REPO = os.environ.get("REPO_ROOT", os.getcwd())
T1 = os.path.join(REPO, "eval_results/table1_cross_lingual")
T2 = os.path.join(REPO, "eval_results/table2_system_comparison")


def wilson_half(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den


def jload(path):
    with open(path) as f:
        return json.load(f)


def mean_ci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n), n


def whisper_inj(lang):
    if lang == "en":
        s = jload(os.path.join(T2, "whisper_inj_large_v3.json"))["systems"]["speechsplice"]
        return s["k"], s["n"]
    d = jload(os.path.join(T1, f"{lang}_judge.json"))
    return d["summary"]["whisper_inj"]["k"], d["summary"]["n"]


def utmos(lang):
    if lang == "en":
        bl = [r["bl_utmos"] for r in jload(os.path.join(T1, "en_results.json"))["rows"]]
        inj = [r["utmos"] for r in jload(os.path.join(T1, "en_judge.json"))["rows"]]
    else:
        rows = jload(os.path.join(T1, f"{lang}_results.json"))["rows"]
        bl = [r["bl_utmos"] for r in rows]
        inj = [r["inj_utmos"] for r in rows]
    return mean_ci(bl), mean_ci(inj)


F0KEY = {"en": "speechsplice", "ja": "ja_wordrepl", "zh": "zh_wordrepl"}
f0 = jload(os.path.join(T1, "f0_summary.json"))

print(f"{'Lang':<5}{'Whisper-INJ (%)':>17}{'UTMOS BL':>14}{'UTMOS INJ':>14}"
      f"{'SS BL--INJ':>14}{'SS INJ--prompt':>16}{'|dF0| ratio':>13}")
print("-" * 93)
for lang in ["en", "ja", "zh"]:
    k, n = whisper_inj(lang)
    (bm, bh, _), (im, ih, _) = utmos(lang)
    ss = jload(os.path.join(T1, f"speaker_sim_{lang}.json"))
    pair, prm = ss["bl_inj_pair"], ss["sim_prompt_inj"]
    f = f0[F0KEY[lang]]
    print(f"{lang.upper():<5}{100*k/n:9.2f} ± {100*wilson_half(k, n):.2f}"
          f"{bm:9.2f} ± {bh:.2f}{im:9.2f} ± {ih:.2f}"
          f"{pair[0]:9.2f} ± {pair[1]:.2f}{prm[0]:11.2f} ± {prm[1]:.2f}"
          f"{f['abs_diff'] / f['anchor_abs_diff']:13.2f}")

print()
print("Paired change of the INJ--prompt similarity (INJ minus BL), Wilcoxon signed-rank:")
st = jload(os.path.join(T1, "speaker_similarity_tests.json"))
for key in ["en_ss", "ja_ss", "zh_ss", "vc_tts"]:
    r = st[key]
    print(f"  {key:<7} n={r['n']:<4} delta={r['delta_mean']:+.4f} ± {r['ci95']:.4f}  p={r['wilcoxon_p']:.4f}")
