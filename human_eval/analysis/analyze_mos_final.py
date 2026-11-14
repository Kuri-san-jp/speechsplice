#!/usr/bin/env python3
"""Aggregation for the MOS-50 main study (CPU only; pre-specified analysis).

NOT RUNNABLE FROM THE DATA IN THIS REPOSITORY.
  This script consumes participant-level raw responses, which are withheld
  (listeners did not consent to individual-level release). It needs:
    - deploy_mos50/filemap_neutral.json  (clip -> (sid, condition) de-blinding map)
    - mos50_ratings_dump.csv             (one row per listener x clip rating)
  It is included so that the pre-specified analysis can be audited as code.
  Its output is shipped verbatim in ../results/mos_final.txt, and the MOS
  columns of Table 1 can be read from that file.

Pre-specified rules:
- A valid listener completed all 50 distinct main-test clips.
- For duplicate ratings, only the first (listener x clip) rating is used.
- One ZH listener who gave 5 to every item is dropped from the primary
  analysis (a sensitivity analysis including them is also reported).
- Tests: paired Wilcoxon by listener (per-listener BL mean vs INJ mean),
         paired Wilcoxon by item (50 pairs, BL mean vs INJ mean),
         item-resampling bootstrap 95% CI for the difference.
"""
import csv, json, statistics, random
from collections import defaultdict
from scipy.stats import wilcoxon

APP = "."
FM = json.load(open(f"{APP}/deploy_mos50/filemap_neutral.json"))
# One over-recruited ZH participant was removed from the source data (2026-08-02);
# EXCLUDE now contains only the pre-specified straight-liner (SHA-256 hashed).
EXCLUDE = {"sha256:cdee53ff2d51"}  # ZH, gave 5 to every item (pre-specified exclusion)

rows = list(csv.DictReader(open(f"{APP}/mos50_ratings_dump.csv")))
main = [r for r in rows if r["phase"] == "main"]

# De-blind the clip, and keep only the first rating per (listener, clip)
seen = set()
data = []  # (base_lang, block, pid, sid, cond, score)
for r in main:
    key = (r["pid"], r["lang"], r["audio_file"])
    if key in seen:
        continue
    seen.add(key)
    m = FM[r["lang"]].get(f"main/{r['audio_file']}")
    if m is None:
        raise KeyError(r["lang"], r["audio_file"])
    data.append((r["lang"].split("_")[0], r["lang"], r["pid"], m["sid"], m["cond"], int(r["score"])))

def analyze(lang, exclude=frozenset()):
    d = [x for x in data if x[0] == lang and x[2] not in exclude]
    # A valid listener completed all 50 clips
    per_p = defaultdict(set)
    for _, blk, pid, sid, cond, s in d:
        per_p[pid].add((sid, cond))
    valid = {p for p, clips in per_p.items() if len(clips) >= 50}
    d = [x for x in d if x[2] in valid]
    scores = {"baseline": [], "oeoi": []}
    p_cond = defaultdict(lambda: defaultdict(list))   # pid -> cond -> scores
    item_cond = defaultdict(lambda: defaultdict(list))  # sid -> cond -> scores
    for _, blk, pid, sid, cond, s in d:
        scores[cond].append(s)
        p_cond[pid][cond].append(s)
        item_cond[sid][cond].append(s)
    bl = statistics.mean(scores["baseline"]); inj = statistics.mean(scores["oeoi"])
    # Wilcoxon by listener
    pb = [statistics.mean(p_cond[p]["baseline"]) for p in sorted(p_cond)]
    pi = [statistics.mean(p_cond[p]["oeoi"]) for p in sorted(p_cond)]
    try: w_p = wilcoxon(pb, pi).pvalue
    except ValueError: w_p = float("nan")
    # Wilcoxon by item
    ib = [statistics.mean(item_cond[s]["baseline"]) for s in sorted(item_cond)]
    ii = [statistics.mean(item_cond[s]["oeoi"]) for s in sorted(item_cond)]
    w_i = wilcoxon(ib, ii).pvalue
    # Item-resampling bootstrap
    random.seed(2026)
    sids = sorted(item_cond)
    deltas = []
    for _ in range(10000):
        sample = [random.choice(sids) for _ in sids]
        db = statistics.mean(statistics.mean(item_cond[s]["baseline"]) for s in sample)
        di = statistics.mean(statistics.mean(item_cond[s]["oeoi"]) for s in sample)
        deltas.append(di - db)
    deltas.sort()
    lo, hi = deltas[249], deltas[9749]
    n_ratings = len(scores["baseline"]) + len(scores["oeoi"])
    per_clip = statistics.mean(len(v["baseline"]) for v in item_cond.values())
    return dict(lang=lang, n_valid=len(valid), n_ratings=n_ratings, per_clip=per_clip,
                bl=bl, inj=inj, delta=inj - bl, ci=(lo, hi), w_part=w_p, w_item=w_i,
                item_cond=item_cond)

KIND = {"en": {"ei": "sentence-initial person name",
               "em": "sentence-medial person name",
               "ee": "sentence-medial/final person name"},
        "zh": {"zc": "place name",
               "zd": "place name, additional",
               "zp": "person name"},
        "ja": {"rm": "sentence-medial surname",
               "rs": "surname replacement",
               "rx": "surname replacement, new",
               "re": "surname replacement, additional"}}

print("=" * 80)
for lang in ["en", "zh", "ja"]:
    r = analyze(lang, EXCLUDE)
    print(f"[{lang.upper()}] {r['n_valid']} valid listeners, {r['n_ratings']} accepted "
          f"ratings (mean {r['per_clip']:.1f} per clip)")
    print(f"  BL {r['bl']:.3f}  INJ {r['inj']:.3f}  Δ={r['delta']:+.3f}  "
          f"95%CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")
    print(f"  Wilcoxon by listener p={r['w_part']:.4f} / by item p={r['w_item']:.4f}")
    # Breakdown by item category
    for pref, name in KIND[lang].items():
        ic = {s: v for s, v in r["item_cond"].items() if s.startswith(pref)}
        if not ic: continue
        b = statistics.mean(statistics.mean(v["baseline"]) for v in ic.values())
        i = statistics.mean(statistics.mean(v["oeoi"]) for v in ic.values())
        print(f"    {name} ({pref}, {len(ic)} pairs): BL {b:.3f} INJ {i:.3f} Δ{i-b:+.3f}")
    if lang == "zh":
        r2 = analyze(lang, frozenset())
        print(f"  [sensitivity] including the one all-5 straight-liner: "
              f"BL {r2['bl']:.3f} INJ {r2['inj']:.3f} Δ={r2['delta']:+.3f} "
              f"(item p={r2['w_item']:.4f})")
    print()
