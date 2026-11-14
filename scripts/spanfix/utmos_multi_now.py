"""Re-score UTMOS22 for the two-entity block of Table 3 on the CURRENT audio (2026-10-10).

Why: scripts/analysis/utmos_table3.py (2026-09-27) caches scores by item number, so
pairs regenerated after 9/27 (e.g. the Table 3 two-entity EN pairs redone with seed 43
and the 'Thank you.' clips) would keep their old scores if it were simply re-run.
This wrapper calls the same scorer (same model, 16 kHz, same groups) but writes to a
fresh folder, so every file is scored from the audio on disk now. The 9/27 results in
eval_results/_utmos_table3/ are left untouched.

GPU job (from $REPO_ROOT; the project is /mnt in the container):
  SLURM_TIME=00:40:00 slurm_run.sh python3 -u /mnt/scripts/spanfix/gpu_guard.py \
      --min_free_gb 30 -- /mnt/scripts/spanfix/utmos_multi_now.py
CPU listing only:  python3 scripts/spanfix/utmos_multi_now.py --list
Output: eval_results/_utmos_multi_20261010/multi__<cell>.json and _summary.json
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))
import utmos_table3 as base  # noqa: E402

base.OUT = os.path.join(base.EV, "_utmos_multi_20261010")
_all_groups = base.build_groups


def multi_only():
    groups, missing = _all_groups()
    return ([g for g in groups if g[0] == "multi"],
            [m for m in missing if m[0] == "multi"])


base.build_groups = multi_only

if __name__ == "__main__":
    if "--list" in sys.argv:
        g, m = multi_only()
        for x in g:
            print(x[0], x[1], x[2])
        print("groups", len(g), "missing", len(m), "out", base.OUT)
    else:
        if os.path.isdir(base.OUT) and os.listdir(base.OUT):
            sys.exit("output folder is not empty: %s (refusing to reuse cached scores)" % base.OUT)
        base.main()
