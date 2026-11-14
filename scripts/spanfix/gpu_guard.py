#!/usr/bin/env python3
"""Run a GPU script only if the assigned GPU has enough free memory.

Some processes on this node use GPUs outside Slurm, so the GPU that Slurm
assigns can already be partly full (2026-10-01: an outside process held 20 GB
of GPU 2 and every Whisper call of appendix C part 0 ran out of memory).
This guard reads the free memory of the visible GPU with nvidia-smi, without
importing torch, and exits with code 75 before the target script starts (so
the target writes no record) when less than --min_free_gb is free.

  python3 -u /mnt/scripts/spanfix/gpu_guard.py --min_free_gb 30 -- /mnt/scripts/spanfix/appc_en_gen.py --seed 42 --part 1
"""
import argparse
import runpy
import subprocess
import sys


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    rows = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
    if len(rows) != 1:
        raise SystemExit("[gpu_guard] expected exactly one visible GPU, got %d: %s" % (len(rows), rows))
    idx, used, total = rows[0]
    return idx, (float(total) - float(used)) / 1024.0, float(used) / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_free_gb", type=float, required=True)
    ap.add_argument("script")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    idx, free, used = free_gb()
    print("[gpu_guard] visible GPU %s: used %.1f GB, free %.1f GB (need %.1f GB)" % (idx, used, free, a.min_free_gb), flush=True)
    if free < a.min_free_gb:
        print("[gpu_guard] NOT ENOUGH FREE MEMORY: the GPU is in use outside this job. Exiting 75 without running the target.", flush=True)
        raise SystemExit(75)
    sys.argv = [a.script] + [x for x in a.args if x != "--"]
    runpy.run_path(a.script, run_name="__main__")


if __name__ == "__main__":
    main()
