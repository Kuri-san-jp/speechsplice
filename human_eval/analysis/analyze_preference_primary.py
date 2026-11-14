#!/usr/bin/env python3
"""Primary analysis of the A/B pronunciation-match test (pre-specified plan, 2026-08-01 revision).

Participant-level responses are withheld (listeners did not consent to
individual-level release), so the *real* analysis cannot be re-run here. Its
output is shipped verbatim in ../results/primary_analysis.txt. What IS runnable
from this repository is --simulate, which builds a synthetic response DB from
the shipped stimulus manifests and checks the estimators against planted truth:

  python3 human_eval/analysis/analyze_preference_primary.py --simulate

To recompute the published A/B numbers directly from the shipped item-level
counts (no DB needed), use scripts/aggregate_human_eval.py instead.

CPU only. Usage:
  python3 analyze_preference_primary.py [--db PATH] [--simulate]

Implementation of the pre-specified statistical analysis plan:
 1. headline = unconditional INJ-selection rate (ties count as failures and stay in the denominator)
 2. CI: the primary report is item-level (Wilson on the per-item majority rate, and an
    item bootstrap on the mean per-item rate). The response level is reported alongside
    with a two-way listener x item cluster bootstrap. The pooled Wilson interval is shown
    for reference only.
 3. Primary test: GEE (Binomial, items as clusters, exchangeable, cluster-robust SE)
    against H0: p=0.5. If it fails to converge, fall back to an exact binomial test on
    the item-level majorities.
 4. Duplicates: keep only the earliest created_at per (participant, phase, pair_id).
 5. Exclusions: catch pass rate below 2/3, straight-liners (same on-screen position for
    every item), and completion below 90%. **Both before and after exclusions are reported.**
 6. Multiplicity: estimation-oriented. Tests against 50% use Bonferroni alpha=0.0125.
 7. Human x Whisper agreement: item-level 2x2 (human = majority), raw and specific
    agreement, and a listing of the mismatched items.
 8. No cross-language comparison is made (values are only reported side by side).
 9. In ab_main, value=2 is treated as invalid: excluded, with its count reported.
10. no-harm (ja_corr control) uses an item-level Wilson interval with n=12.

--simulate: dry run on synthetic data, with known ground-truth rates planted so the
            estimators can be checked. This mode needs no participant data and is the
            way to verify this script from the files shipped in this repository.
"""
import argparse
import json
import math
import random
import sqlite3
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

MOS_APP = Path(__file__).resolve().parent
STIMULI = MOS_APP.parent / "stimuli"
ALPHA_BONF = 0.0125  # 3 languages of ab_main + ja_corr = 4 comparisons
N_BOOT = 2000
SEED = 42


# ---------------------------------------------------------------- utilities
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, center - half, center + half


def load_items():
    """Read (lang, phase, pair_id) -> item_type and the target definition from the item manifests."""
    meta, blockmap = {}, {}
    for src, lang, phase in [("en", "en", "ab_main"), ("ja", "ja", "ab_main"),
                             ("zh", "zh", "ab_main"), ("ja_corr", "ja", "ab_corr"),
                             ("ja_het", "ja", "ab_het")]:
        manifest = STIMULI / f"{src}_items.json"
        if not manifest.exists():
            # ja_corr is not shipped: that phase collected 0 responses in the study.
            continue
        for it in json.load(open(manifest)):
            ph = (f"{phase}_practice" if phase == "ab_het" else "ab_practice") \
                if it["item_type"] == "practice" else phase
            meta[(lang, ph, it["pair_id"])] = it["item_type"]
            blockmap[(lang, ph, it["pair_id"])] = it.get("block", "both")
    return meta, blockmap


def load_whisper_item_results():
    """Read the item-level Whisper verdict (did the INJ audio realize the target word?)."""
    root = MOS_APP.parent.parent
    res = {}
    try:
        d = json.load(open(root / "eval_results/table2_system_comparison/whisper_inj_large_v3.json"))
        for r in d["systems"]["speechsplice"]["rows"]:
            res[("en", f"{r['idx']:04d}")] = bool(r["hit"])
    except Exception as e:
        print(f"[warn] could not load the EN Whisper verdicts: {e}")
    for lang, path in [("ja", "eval_results/table1_cross_lingual/ja_judge.json"),
                       ("zh", "eval_results/table1_cross_lingual/zh_judge.json")]:
        try:
            for r in json.load(open(root / path))["rows"]:
                res[(lang, f"{r['pair_idx']:04d}")] = bool(r["inj_hit"])
        except Exception as e:
            print(f"[warn] could not load the {lang} Whisper verdicts: {e}")
    return res


# ---------------------------------------------------------------- data prep
def fetch_rows(db_path, phase):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    # Use id as the secondary sort key so ties on created_at resolve deterministically
    rows = con.execute(
        """SELECT c.participant_id AS pdb, p.participant_id AS pid, p.lang,
                  c.pair_id, c.phase, c.system_a, c.system_b, c.is_swapped,
                  c.selected_value, c.created_at
           FROM preference_comparisons c
           JOIN preference_participants p ON p.id = c.participant_id
           WHERE c.phase = ? ORDER BY c.created_at, c.id""", (phase,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def drop_invalid(rows):
    """Rule 9: selected_value=2 in ab_main is invalid (dropped after dedupe, before QC)."""
    valid, n_inv = [], 0
    for r in rows:
        if r["phase"] == "ab_main" and r["selected_value"] == 2:
            n_inv += 1
            continue
        valid.append(r)
    return valid, n_inv


def dedupe(rows):
    """Keep only the earliest created_at per (participant, phase, pair_id) (rule 4)."""
    seen, out, ndup = set(), [], 0
    for r in rows:
        key = (r["pdb"], r["phase"], r["pair_id"])
        if key in seen:
            ndup += 1
            continue
        seen.add(key)
        out.append(r)
    return out, ndup


def classify(r, meta):
    lang = r["lang"].split("_")[0]
    itype = meta.get((lang, r["phase"], r["pair_id"]), "unknown")
    v = r["selected_value"]
    chosen = None
    if v == 1:
        chosen = r["system_a"]
    elif v == -1:
        chosen = r["system_b"]
    return lang, itype, v, chosen


def participant_qc(rows, meta, phase, blockmap=None):
    """Rule 5: apply the catch (>=2/3), straight-liner, and 90%-completion criteria.

    Block design (25 pairs x 2 blocks): a listener's block is inferred by majority over
    the main items they answered, and the completion denominator is the number of items
    in that block (25 + 3 catch = 28).
    """
    per = defaultdict(lambda: {"catch_ok": 0, "catch_n": 0, "labels": set(),
                               "n": 0, "lang": None, "blk_a": 0, "blk_b": 0})
    expected_by_block = defaultdict(lambda: defaultdict(set))
    for (lang, ph, pid), t in meta.items():
        if ph == phase:
            blk = (blockmap or {}).get((lang, ph, pid), "both")
            expected_by_block[lang][blk].add(pid)
    unknown = 0
    for r in rows:
        lang, itype, v, _ = classify(r, meta)
        if itype == "unknown":
            unknown += 1
            continue
        s = per[r["pdb"]]
        s["lang"] = lang
        s["n"] += 1
        s["labels"].add("A" if v == 1 else "B" if v == -1 else f"mid{v}")
        if itype == "catch":
            s["catch_n"] += 1
            s["catch_ok"] += (v == 0)
        elif itype == "main" and blockmap:
            blk = blockmap.get((lang, r["phase"], r["pair_id"]), "both")
            if blk == "a":
                s["blk_a"] += 1
            elif blk == "b":
                s["blk_b"] += 1
    if unknown:
        raise SystemExit(f"[fatal] {unknown} responses carry a pair_id absent from the item "
                         "manifests; resolve the stimulus-metadata mismatch first")
    excluded = {}
    for pdb, s in per.items():
        reasons = []
        # Catch pass rate below 2/3 (judged as a ratio even when catch_n<3 on partial completion)
        if s["catch_n"] >= 2 and s["catch_ok"] / s["catch_n"] < 2 / 3:
            reasons.append(f"catch {s['catch_ok']}/{s['catch_n']}")
        # straight-liner: pressed the same option (A, B, or a middle one) on every item.
        # Listeners with fewer than 10 answers are caught by the completion criterion instead.
        if s["n"] >= 10 and len(s["labels"]) == 1:
            reasons.append("straight-liner")
        by_blk = expected_by_block.get(s["lang"], {})
        both_n = len(by_blk.get("both", set()))
        if by_blk.get("a") or by_blk.get("b"):
            # Block design: infer the listener's block as the side they answered more of
            blk = "a" if s["blk_a"] >= s["blk_b"] else "b"
            n_exp = len(by_blk.get(blk, set())) + both_n
        else:
            n_exp = sum(len(v) for v in by_blk.values()) or 1
        if s["n"] < 0.9 * n_exp:
            reasons.append(f"completion {s['n']}/{n_exp}")
        if reasons:
            excluded[pdb] = reasons
    return excluded


# ---------------------------------------------------------------- analyses
def headline(rows, meta, lang, target_itype="main"):
    """Rules 1-2: item-level and response-level aggregation of the unconditional INJ rate."""
    by_item = defaultdict(list)
    n_invalid = 0
    for r in rows:
        rl, itype, v, chosen = classify(r, meta)
        if rl != lang or itype != target_itype:
            continue
        if r["phase"] == "ab_main" and v == 2:
            n_invalid += 1  # Rule 9 (normally 0, since the server rejects these with 422)
            continue
        by_item[r["pair_id"]].append(1 if chosen == "inj" else 0)
    if not by_item:
        return None
    items = sorted(by_item)
    per_item = np.array([np.mean(by_item[i]) for i in items])
    maj = np.array([np.mean(by_item[i]) > 0.5 for i in items])
    n_item = len(items)
    p_maj, lo_maj, hi_maj = wilson(int(maj.sum()), n_item)
    rng = np.random.default_rng(SEED)
    boots = [np.mean(per_item[rng.integers(0, n_item, n_item)])
             for _ in range(N_BOOT)]
    resp = [x for i in items for x in by_item[i]]
    k, n = int(np.sum(resp)), len(resp)
    return {
        "n_items": n_item, "n_resp": n, "n_invalid": n_invalid,
        "item_majority": (p_maj, lo_maj, hi_maj),
        "item_mean": (float(per_item.mean()),
                      float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5))),
        "pooled_ref": wilson(k, n),
        "by_item": by_item,
    }


def two_way_cluster_boot(rows, meta, lang, target_itype="main"):
    """Rule 2: two-way listener x item cluster bootstrap (response-level rate)."""
    data = defaultdict(dict)
    for r in rows:
        rl, itype, v, chosen = classify(r, meta)
        if rl != lang or itype != target_itype or (r["phase"] == "ab_main" and v == 2):
            continue
        data[r["pdb"]][r["pair_id"]] = 1 if chosen == "inj" else 0
    listeners = sorted(data)
    items = sorted({i for d in data.values() for i in d})
    if not listeners or not items:
        return None
    rng = np.random.default_rng(SEED + 1)
    stats = []
    for _ in range(N_BOOT):
        ls = rng.choice(listeners, len(listeners))
        its = rng.choice(items, len(items))  # keep multiplicity (do not deduplicate)
        vals = [data[l][i] for l in ls for i in its if i in data[l]]
        if vals:
            stats.append(np.mean(vals))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def gee_primary(rows, meta, lang, target_itype="main"):
    """Rule 3: GEE (Binomial, items as clusters, exchangeable, robust SE) against H0 p=0.5."""
    ys, groups = [], []
    for r in rows:
        rl, itype, v, chosen = classify(r, meta)
        if rl != lang or itype != target_itype or (r["phase"] == "ab_main" and v == 2):
            continue
        ys.append(1 if chosen == "inj" else 0)
        groups.append(r["pair_id"])
    if not ys:
        return {"method": "none", "note": "no data"}
    y = np.array(ys, dtype=float)
    try:
        import pandas as pd
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
        df = pd.DataFrame({"y": y, "g": groups, "const": 1.0})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = GEE(df["y"], df[["const"]], groups=df["g"],
                      family=Binomial(), cov_struct=Exchangeable()).fit(
                          cov_type="robust")
        if not getattr(res, "converged", True):
            raise RuntimeError("GEE did not converge")
        b, se = float(res.params.iloc[0]), float(res.bse.iloc[0])
        if not (np.isfinite(b) and np.isfinite(se) and se < 50):
            raise RuntimeError(f"GEE unstable (b={b:.2f}, se={se:.2f})")
        from scipy.stats import norm
        p_hat = 1 / (1 + math.exp(-b))
        lo = 1 / (1 + math.exp(-(b - 1.96 * se)))
        hi = 1 / (1 + math.exp(-(b + 1.96 * se)))
        zstat = b / se  # H0: logit(p)=0 <=> p=0.5
        pval = 2 * (1 - norm.cdf(abs(zstat)))
        return {"method": "GEE", "p_hat": p_hat, "ci": (lo, hi),
                "z": zstat, "pval": pval,
                "sig_bonf": bool(pval < ALPHA_BONF)}
    except Exception as e:
        # Fallback (rule 3): exact binomial on the item-level majorities
        from scipy.stats import binomtest
        by_item = defaultdict(list)
        for yi, g in zip(y, groups):
            by_item[g].append(yi)
        maj = [np.mean(v) > 0.5 for v in by_item.values()]
        # Report two-sided to match the direction of the GEE test
        bt = binomtest(int(np.sum(maj)), len(maj), 0.5, alternative="two-sided")
        return {"method": f"exact-binomial (GEE fallback: {e})",
                "k": int(np.sum(maj)), "n": len(maj),
                "pval": bt.pvalue, "sig_bonf": bool(bt.pvalue < ALPHA_BONF)}


def agreement(head, whisper, lang):
    """Rule 7: 2x2 of human (item majority) x Whisper, specific agreement, mismatch listing."""
    tt = tf = ft = ff = 0
    mism = []
    for pid, votes in head["by_item"].items():
        w = whisper.get((lang, pid))
        if w is None:
            continue
        h = np.mean(votes) > 0.5
        if h and w:
            tt += 1
        elif h and not w:
            tf += 1
        elif not h and w:
            ft += 1
        else:
            ff += 1
        if h != w:
            mism.append((pid, "human+/W-" if h else "human-/W+"))
    n = tt + tf + ft + ff
    if n == 0:
        return None
    raw = (tt + ff) / n
    ppa = 2 * tt / (2 * tt + tf + ft) if (2 * tt + tf + ft) else float("nan")
    npa = 2 * ff / (2 * ff + tf + ft) if (2 * ff + tf + ft) else float("nan")
    return {"n": n, "table": (tt, tf, ft, ff), "raw": raw,
            "ppa": ppa, "npa": npa, "mismatch": mism}


# ---------------------------------------------------------------- reporting
def fmt_ci(t):
    p, lo, hi = t
    return f"{p*100:5.1f}% [{lo*100:5.1f}, {hi*100:5.1f}]"


def report_phase(db, meta, whisper, phase, langs, blockmap=None):
    rows_all, ndup = dedupe(fetch_rows(db, phase))
    rows_all, ninv = drop_invalid(rows_all)
    print(f"\n{'='*74}\nPHASE {phase}: responses {len(rows_all)}"
          f" (duplicates removed {ndup}, invalid(v=2) {ninv})")
    excluded = participant_qc(rows_all, meta, phase, blockmap)
    if excluded:
        for pdb, why in excluded.items():
            print(f"  excluded listener pdb={pdb}: {', '.join(why)}")
    for tag, rows in [("Before exclusions", rows_all),
                      ("After exclusions", [r for r in rows_all if r["pdb"] not in excluded])]:
        n_listener = len({r["pdb"] for r in rows})
        print(f"\n--- {tag} (rule 5: report both) {n_listener} listeners ---")
        for lang in langs:
            h = headline(rows, meta, lang)
            if h is None:
                print(f"  {lang}: no data")
                continue
            tw = two_way_cluster_boot(rows, meta, lang)
            g = gee_primary(rows, meta, lang)
            print(f"  [{lang}] items={h['n_items']} resp={h['n_resp']}"
                  f" invalid(v=2)={h['n_invalid']}")
            print(f"    [E1] majority rate      : {fmt_ci(h['item_majority'])} (Wilson, n={h['n_items']}"
                  f"; estimand=proportion of items succeeding by majority)")
            print(f"    [E2] per-item rate      : {fmt_ci(h['item_mean'])} (item bootstrap"
                  f"; estimand=headline, unconditional INJ-selection rate)")
            if tw:
                print(f"    response-level rate     : {h['pooled_ref'][0]*100:5.1f}%"
                      f" [{tw[0]*100:5.1f}, {tw[1]*100:5.1f}] (two-way cluster boot)")
            print(f"    (ref.) pooled Wilson    : {fmt_ci(h['pooled_ref'])} (assumes independence; not the primary report)")
            if g["method"] == "GEE":
                pstr = f"{g['pval']:.2g}" if g['pval'] > 1e-16 else "<1e-16"
                print(f"    primary analysis GEE    : p_hat={g['p_hat']*100:.1f}%"
                      f" [{g['ci'][0]*100:.1f}, {g['ci'][1]*100:.1f}]"
                      f" z={g['z']:.2f} p={pstr}"
                      f" {'(significant at Bonferroni alpha=0.0125)' if g['sig_bonf'] else '(n.s.)'}")
            else:
                print(f"    primary analysis fb     : {g['method'][:60]}"
                      f" k/n={g.get('k')}/{g.get('n')} p={g.get('pval'):.2g}")
            if phase == "ab_main":
                a = agreement(headline(rows, meta, lang), whisper, lang)
                if a:
                    tt, tf, ft, ff = a["table"]
                    print(f"    human x Whisper (item)  : raw={a['raw']*100:.0f}%"
                          f" [both+ {tt}, human-only {tf}, W-only {ft}, both- {ff}]"
                          f" PPA={a['ppa']:.2f} NPA={a['npa']:.2f}")
                    for pid, kind in a["mismatch"]:
                        print(f"      mismatch item {pid}: {kind}")
            # Catch pass rate (QC indicator)
            catch = defaultdict(list)
            for r in rows:
                rl, itype, v, _ = classify(r, meta)
                if rl == lang and itype == "catch":
                    catch[r["pair_id"]].append(v == 0)
            if catch:
                allv = [x for v in catch.values() for x in v]
                print(f"    catch pass rate (QC)    : {fmt_ci(wilson(int(np.sum(allv)), len(allv)))}")
            if phase == "ab_corr":
                ctrl = defaultdict(list)
                for r in rows:
                    rl, itype, v, chosen = classify(r, meta)
                    if rl == lang and itype == "control":
                        ctrl[r["pair_id"]].append(v == 2 or chosen == "inj")
                if ctrl:
                    maj = [np.mean(v) > 0.5 for v in ctrl.values()]
                    print(f"    no-harm (item, n={len(maj)}) : "
                          f"{fmt_ci(wilson(int(np.sum(maj)), len(maj)))} (rule 10)")


# ---------------------------------------------------------------- simulate
def simulate(db_path):
    """Synthetic DB for the dry run: plant known ground-truth rates so the estimators can be checked."""
    rng = random.Random(SEED)
    con = sqlite3.connect(db_path)
    con.executescript("""
      CREATE TABLE preference_participants (
        id INTEGER PRIMARY KEY, participant_id TEXT, lang TEXT);
      CREATE TABLE preference_comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT, participant_id INT, pair_id TEXT,
        phase TEXT, system_a TEXT, system_b TEXT, is_swapped INT,
        selected_value INT, created_at TEXT);
    """)
    meta, blockmap = load_items()
    truth = {"en": 0.95, "ja": 0.90, "zh": 0.93}
    pdb = 0
    for lang in ["en", "ja", "zh"]:
        all_items = [(pid, t) for (l, ph, pid), t in meta.items()
                     if l == lang and ph == "ab_main"]
        for li in range(30):
            # Block design: first 15 listeners = block a, last 15 = block b (catch items shared)
            myblk = "a" if li < 15 else "b"
            items = [(pid, t) for pid, t in all_items
                     if t == "catch"
                     or blockmap.get((lang, "ab_main", pid)) in (myblk, "both")]
            pdb += 1
            con.execute("INSERT INTO preference_participants VALUES (?,?,?)",
                        (pdb, f"SIM_{lang}_{li}", lang))
            bad = (li == 0)          # 1 listener: fails every catch (should be excluded)
            liner = (li == 1)        # 1 listener: straight-liner
            partial = (li == 2)      # 1 listener: 50% completion (should be excluded)
            for k, (pid, t) in enumerate(items):
                if partial and k > len(items) // 2:
                    continue
                swapped = rng.random() < 0.5
                sa, sb = ("inj", "bl") if swapped else ("bl", "inj")
                if liner:
                    v = 1
                elif t == "catch":
                    v = 0 if (not bad and rng.random() < 0.95) else \
                        rng.choice([1, -1])
                else:
                    if rng.random() < truth[lang]:
                        v = 1 if sa == "inj" else -1
                    else:
                        v = 0 if rng.random() < 0.6 else (1 if sa == "bl" else -1)
                con.execute(
                    "INSERT INTO preference_comparisons"
                    "(participant_id,pair_id,phase,system_a,system_b,"
                    "is_swapped,selected_value,created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (pdb, pid, "ab_main", sa, sb, swapped, v,
                     f"2026-08-02T00:{li:02d}:{k:02d}"))
                if li == 3 and k < 5:  # 5 duplicates (to verify dedupe)
                    con.execute(
                        "INSERT INTO preference_comparisons"
                        "(participant_id,pair_id,phase,system_a,system_b,"
                        "is_swapped,selected_value,created_at)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (pdb, pid, "ab_main", sa, sb, swapped,
                         1 if v != 1 else -1, f"2026-08-02T09:{li:02d}:{k:02d}"))
    # ja_corr: main 90% inj / control 95% both
    items = [(pid, t) for (l, ph, pid), t in meta.items()
             if l == "ja" and ph == "ab_corr"]
    for li in range(30):
        pdb += 1
        con.execute("INSERT INTO preference_participants VALUES (?,?,?)",
                    (pdb, f"SIM_corr_{li}", "ja"))
        for k, (pid, t) in enumerate(items):
            swapped = rng.random() < 0.5
            sa, sb = ("inj", "bl") if swapped else ("bl", "inj")
            if t == "control":
                v = 2 if rng.random() < 0.95 else (1 if sa == "inj" else -1)
            else:
                v = (1 if sa == "inj" else -1) if rng.random() < 0.90 else \
                    rng.choice([0, 2])
            con.execute(
                "INSERT INTO preference_comparisons"
                "(participant_id,pair_id,phase,system_a,system_b,"
                "is_swapped,selected_value,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (pdb, pid, "ab_corr", sa, sb, swapped, v,
                 f"2026-08-02T01:{li:02d}:{k:02d}"))
    con.commit()
    con.close()
    print(f"synthetic DB written: {db_path}")
    print("planted truth: EN 95% / JA 90% / ZH 93%, corr main 90%, no-harm 95%")
    print("planted per language: 1 catch-failing listener, 1 straight-liner, 1 at 50% completion, 5 duplicates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(MOS_APP / "backend" / "mos_evaluation.db"),
                    help="participant-level response DB (withheld; not in this repository)")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    meta, blockmap = load_items()
    whisper = load_whisper_item_results()
    db = args.db
    if args.simulate:
        db = str(MOS_APP / "sim_ab.db")
        Path(db).unlink(missing_ok=True)
        simulate(db)
    report_phase(db, meta, whisper, "ab_main", ["en", "ja", "zh"], blockmap)
    report_phase(db, meta, whisper, "ab_het", ["ja"], blockmap)
    report_phase(db, meta, whisper, "ab_corr", ["ja"])


if __name__ == "__main__":
    main()
