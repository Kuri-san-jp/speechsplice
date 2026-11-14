"""Aggregate Table 3: cross-architecture results for 17 model-language conditions.

Reads eval_results/table3_cross_arch/manifest.json and the per-pair files it
lists (300 pairs per condition, or 300 two-entity sentences):

  point_mcd/{key}.json      MCD between INJ and BL at our replacement point
  point_pib/{key}.json      phoneme PER of BL and INJ against TGT at that point
  sweep_pib/{key}_lNN.json  the same PER file at each swept depth l* = NN
  two_entity/{key}.json     per-slot judge verdicts on the two-entity sentences

Every column of Table 3 is recomputed here from per-pair rows:

  * MCD (dB): mean of mcd_inj_vs_bl over the pairs with all three MCDs present;
    the interval is the Student-t 95% half-width t(0.975, n-1) * sd / sqrt(n)
    with sd using ddof=1.
  * PIB (%): 100 * #(available and inj_better) / #available, where
    inj_better means PER(INJ, TGT) < PER(BL, TGT) on the entity span. The
    interval is the Wilson 95% half-width (z = 1.96) around the plain
    proportion.
  * Layer sweep: the PIB of each sweep file, and its min-max over the five
    depths.
  * Two-entity sentences (%): BL = (#bl_slot1_ok + #bl_slot2_ok) / 600,
    INJ (one) = (#inj1_slot1_ok + #inj2_slot2_ok) / 600,
    INJ (both) = (#inj12_slot1_ok + #inj12_slot2_ok) / 600. The verdicts are
    Whisper large-v3 surface matches (EN), Furigana Whisper reading matches
    (JA), and toneless pinyin matches (ZH, the *_ok_pinyin flags).

Irodori-TTS replaces after the projection at our replacement point and before
it in the sweep (the dagger in the paper), so its point and sweep values come
from different runs by design.
"""
import json, os
from math import sqrt

from scipy import stats

REPO = os.environ.get("REPO_ROOT", os.getcwd())
DIR = os.path.join(REPO, "eval_results/table3_cross_arch")


def load(rel):
    return json.load(open(os.path.join(DIR, rel)))


def mcd_cell(rel):
    rows = load(rel)["rows"]
    v = [r["mcd_inj_vs_bl"] for r in rows
         if r.get("mcd_inj_vs_bl") is not None and r.get("mcd_inj_vs_tgt") is not None
         and r.get("mcd_bl_vs_tgt") is not None]
    n = len(v)
    m = sum(v) / n
    sd = sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, float(stats.t.ppf(0.975, n - 1)) * sd / sqrt(n), n


def wilson_half(k, n, z=1.96):
    p = k / n
    return z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def pib_cell(rel):
    rows = [r for r in load(rel)["rows"] if r.get("available", True)]
    k = sum(1 for r in rows if r["inj_better"])
    n = len(rows)
    return 100 * k / n, 100 * wilson_half(k, n), k, n


def two_entity(rel, judge):
    rows = load(rel)["rows"]
    suf = "_pinyin" if judge == "pinyin" else ""

    def cnt(f):
        return sum(1 for r in rows if r.get(f + suf))

    d = 2 * len(rows)
    return (100 * (cnt("bl_slot1_ok") + cnt("bl_slot2_ok")) / d,
            100 * (cnt("inj1_slot1_ok") + cnt("inj2_slot2_ok")) / d,
            100 * (cnt("inj12_slot1_ok") + cnt("inj12_slot2_ok")) / d)


def main():
    man = json.load(open(os.path.join(DIR, "manifest.json")))
    hdr = ("Model", "Lang", "L", "MCD (dB)", "PIB (%)", "l*", "sweep PIB min-max",
           "BL", "INJ one", "INJ both")
    print("%-17s %-4s %3s  %-14s %-15s %-15s %-13s %6s %8s %8s" % hdr)
    print("-" * 113)
    fam = None
    for e in man["rows"]:
        if e["family"] != fam:
            fam = e["family"]
            print("[%s]" % fam)
        m, mh, _ = mcd_cell(e["point_mcd"]["file"])
        p, ph, _, _ = pib_cell(e["point_pib"]["file"])
        sweep = [(s["lstar"], pib_cell(s["file"])[0]) for s in e["sweep"]]
        bl, one, both = two_entity(e["two_entity"]["file"], e["two_entity"]["judge"])
        dag = "+" if e["model"] == "Irodori-TTS" else " "
        print("%-17s %-4s %3d  %5.2f +- %4.2f  %5.2f +- %4.2f%s %-15s %5.2f-%5.2f  %6.2f %8.2f %8.2f" % (
            e["model"], e["lang"], e["L"], m, mh, p, ph, dag,
            "/".join(str(l) for l, _ in sweep),
            min(v for _, v in sweep), max(v for _, v in sweep), bl, one, both))
    print()
    print("Per-depth PIB of the sweep (%):")
    for e in man["rows"]:
        print("  %-17s %-3s " % (e["model"], e["lang"]) +
              "  ".join("l*=%d: %.2f" % (s["lstar"], pib_cell(s["file"])[0]) for s in e["sweep"]))
    print()
    print("+ Irodori-TTS replaces after the projection at our replacement point and before it in the sweep.")


if __name__ == "__main__":
    main()
