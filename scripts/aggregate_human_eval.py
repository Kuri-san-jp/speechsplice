"""Aggregate the human-evaluation results from the shipped item-level counts.

Reproduces, with no participant-level data and no GPU:

  * the INJ-selection rate and its item-bootstrap 95% CI for the three
    name sets (EN/JA/ZH) -- the "INJ selected (A/B, %)" column of Table 1;
  * the same quantities for the Japanese same-surface set -- Table
    "Human evaluation of the Japanese same-surface correction set";
  * the item-level majority counts and the response-level breakdown.

Inputs (both shipped):
  human_eval/results/sheet2_main_results.csv   per-set totals
  human_eval/results/sheet3_item_level.csv     per-item judgment counts

The estimators follow the pre-specified plan implemented in
human_eval/analysis/analyze_preference_primary.py: the headline is the
unconditional INJ-selection rate (ties count as failures and stay in the
denominator), the primary interval is an item bootstrap (seed 42, 2000
resamples), and the item-majority rate carries a Wilson interval.

Run:
  REPO_ROOT=$PWD python3 scripts/aggregate_human_eval.py

The GEE z statistics and the MOS columns of Table 1 need the participant-level
response database, which is withheld; they are shipped as text in
human_eval/results/primary_analysis.txt and human_eval/results/mos_final.txt.
The tail of this script echoes the MOS block from that file for convenience.
"""
import csv, math, os
from collections import defaultdict

import numpy as np

REPO = os.environ.get("REPO_ROOT", os.getcwd())
RES = os.path.join(REPO, "human_eval/results")
SEED, N_BOOT = 42, 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def pct(t):
    p, lo, hi = t
    return f"{p*100:5.1f}% [{lo*100:5.1f}, {hi*100:5.1f}]"


rows = list(csv.DictReader(open(os.path.join(RES, "sheet3_item_level.csv"))))
by_set = defaultdict(list)
for r in rows:
    if r["item_type"] != "main":
        continue
    by_set[(r["set"], r["language"])].append(r)

SETS = [("proper_name_steering", "en"), ("proper_name_steering", "ja"),
        ("proper_name_steering", "zh"), ("same_surface_correction", "ja_het")]

print("INJ-selection rate from the shipped item-level counts")
print("=" * 78)
for key in SETS:
    rs = sorted(by_set[key], key=lambda r: r["item_id"])
    n_item = len(rs)
    per_item = np.array([int(r["inj_n"]) / int(r["n"]) for r in rs])
    maj = int(sum(1 for x in per_item if x > 0.5))

    rng = np.random.default_rng(SEED)
    boots = [np.mean(per_item[rng.integers(0, n_item, n_item)]) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    bl_k = sum(int(r["bl_n"]) for r in rs)
    np_k = sum(int(r["nopref_n"]) for r in rs)
    inj_k = sum(int(r["inj_n"]) for r in rs)
    tot = sum(int(r["n"]) for r in rs)

    print(f"\n[{key[0]} / {key[1]}]  items={n_item}  judgments={tot}")
    print(f"  INJ selected            : {inj_k} ({100*inj_k/tot:.1f}%)")
    print(f"  BL selected             : {bl_k} ({100*bl_k/tot:.1f}%)")
    print(f"  Neither selected        : {np_k} ({100*np_k/tot:.1f}%)")
    print(f"  [E2] per-item rate      : {per_item.mean()*100:5.1f}% "
          f"[{lo*100:5.1f}, {hi*100:5.1f}]  (item bootstrap, primary)")
    print(f"  [E1] item-majority rate : {pct(wilson(maj, n_item))}  (Wilson, n={n_item})")
    print(f"  items reaching target   : {maj}/{n_item}")
    print(f"  (ref.) pooled Wilson    : {pct(wilson(inj_k, tot))}")

print()
print("=" * 78)
print("Per-set totals as shipped (human_eval/results/sheet2_main_results.csv)")
print("=" * 78)
with open(os.path.join(RES, "sheet2_main_results.csv")) as f:
    for line in f:
        print("  " + line.rstrip())

mos = os.path.join(RES, "mos_final.txt")
if os.path.exists(mos):
    print()
    print("=" * 78)
    print("MOS results as shipped (human_eval/results/mos_final.txt)")
    print("=" * 78)
    with open(mos) as f:
        for line in f:
            print("  " + line.rstrip())
