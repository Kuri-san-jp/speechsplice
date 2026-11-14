"""Aggregate the pronunciation-baselines table (Japanese split).

Reads eval_results/appendix_f_baselines/results.json and prints:

  Condition           | TRM (%)         | G2P-CER Match (%)

This script aggregates a GPU rerun: `bash scripts/run_appendix_f.sh` writes the
file it reads. The per-pair results behind Table A5 of the paper are shipped in
eval_results/table_a5_baselines/ja.json and aggregated by
scripts/aggregate_appendix.py, which needs no GPU.
"""
import json, os

REPO = os.environ.get("REPO_ROOT", os.getcwd())
F = os.path.join(REPO, "eval_results/appendix_f_baselines/results.json")
if not os.path.exists(F):
    raise SystemExit(
        f"{F} not found.\n"
        "Run `bash scripts/run_appendix_f.sh` (GPU) to produce it, or run\n"
        "scripts/aggregate_appendix.py to aggregate the shipped Table A5 results.")
d = json.load(open(F))
s = d["summary"]

ROW_ORDER = ["baseline", "g2p", "text_hint", "rome", "speechsplice"]
NICE = {
    "baseline": "Baseline",
    "g2p": "G2P Baseline",
    "text_hint": "Text Hint",
    "rome": "ROME/MEMIT",
    "speechsplice": "SpeechSplice (ours)",
}

print(f"{'Condition':<24}{'N':>5}{'TRM (%)':>18}{'G2P-CER Match (%)':>22}")
print("-" * 69)
for k in ROW_ORDER:
    r = s.get(k)
    if not r or r.get("n", 0) == 0:
        print(f"{NICE[k]:<24}{0:>5}{'MISSING':>18}{'MISSING':>22}")
        continue
    trm = f"{r['trm_pct']:.2f} ± {r['trm_ci_half']:.2f}"
    gcr = f"{r['g2p_cer_match_pct']:.2f} ± {r['g2p_cer_match_ci_half']:.2f}"
    print(f"{NICE[k]:<24}{r['n']:>5}{trm:>18}{gcr:>22}")
