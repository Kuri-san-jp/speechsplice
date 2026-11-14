#!/usr/bin/env python3
"""Provenance for the |Δduration| column of Table 2 (naacl2027/main.tex, tab:system_comparison).

Audit item C5 (note/2026-10-02_gpuneeds2_report.md): the printed values match a recomputation
from the wav files, but neither the script nor a result JSON was kept. This script recomputes the
column from the wav headers (frames / sample rate) and stores per-pair values and the summary in
eval_results/_provenance/table2_duration_<timestamp>.json. It reads wavs only and changes nothing.

Definition (Section 4.2 of the paper): |Δduration| is the mean absolute change in utterance duration
from the baseline, |dur(INJ or edited) - dur(BL)| in seconds, averaged over the 300 EN pairs.
± is 1.96 × standard error of the mean (the convention of the Table 1 caption, which the Table 2
caption points to); both the n-1 and the n denominators are stored.

Wav sets (the same files as scripts/analysis/f0_systems.py and speaker_sim_systems.py):
  SpeechSplice  BL  eval_results/en_wordrepl100_pairs_gen/s{i:04d}_bl.wav
                INJ eval_results/en_wordrepl300_splice_inj/s{i:04d}_inj.wav if present (16 splice pairs),
                    else eval_results/en_wordrepl100_pairs_gen/s{i:04d}_inj.wav
  VoiceCraft    eval_results/voicecraft_tts_wordrepl100/s{i:04d}_{bl,inj}.wav
  EdiTTS        BL  eval_results/editts_edit_wordrepl300/wavs/gen_{i}_gradtts-1.wav
                INJ eval_results/editts_edit_wordrepl300/wavs/gen_{i}_EdiTTS.wav

Usage (CPU only):
  taskset -c 0-12 python3 scripts/analysis/provenance_table2_duration.py
"""
import os
import sys

for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "13"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.modules["pyopenjtalk"] = None

import hashlib
import json
import math
import time

import numpy as np
import soundfile as sf

ROOT = os.environ.get("REPO_ROOT", "/mnt")
E = os.path.join(ROOT, "eval_results")
OUT_DIR = os.path.join(E, "_provenance")
N = 300
PRINTED = {"SpeechSplice": "0.00 +- 0.00", "VoiceCraft": "0.20 +- 0.02", "EdiTTS": "0.13 +- 0.01"}


def ss_paths(i):
    bl = os.path.join(E, f"en_wordrepl100_pairs_gen/s{i:04d}_bl.wav")
    alt = os.path.join(E, f"en_wordrepl300_splice_inj/s{i:04d}_inj.wav")
    inj = alt if os.path.exists(alt) else os.path.join(E, f"en_wordrepl100_pairs_gen/s{i:04d}_inj.wav")
    return bl, inj


SYSTEMS = {
    "SpeechSplice": ss_paths,
    "VoiceCraft": lambda i: (os.path.join(E, f"voicecraft_tts_wordrepl100/s{i:04d}_bl.wav"),
                             os.path.join(E, f"voicecraft_tts_wordrepl100/s{i:04d}_inj.wav")),
    "EdiTTS": lambda i: (os.path.join(E, f"editts_edit_wordrepl300/wavs/gen_{i}_gradtts-1.wav"),
                         os.path.join(E, f"editts_edit_wordrepl300/wavs/gen_{i}_EdiTTS.wav")),
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def info(p):
    s = sf.info(p)
    return {"path": os.path.relpath(os.path.realpath(p), ROOT), "frames": int(s.frames),
            "sr": int(s.samplerate), "dur_s": s.frames / s.samplerate, "md5": md5(p)}


def summary(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return {"n": n, "mean": float(x.mean()), "sd_ddof1": float(x.std(ddof=1)),
            "ci_half_1p96se_ddof1": float(1.96 * x.std(ddof=1) / math.sqrt(n)),
            "ci_half_1p96se_ddof0": float(1.96 * x.std(ddof=0) / math.sqrt(n)),
            "min": float(x.min()), "max": float(x.max()), "n_zero": int((x == 0).sum())}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"script": os.path.abspath(__file__), "created": time.strftime("%Y%m%d_%H%M%S"),
           "definition": "|dur(INJ or edited) - dur(BL)| in seconds from wav headers; "
                         "± = 1.96 x SE of the mean over 300 pairs",
           "printed_main_tex_20261002": PRINTED, "systems": {}}
    for name, fn in SYSTEMS.items():
        rows, absd, missing = [], [], []
        for i in range(N):
            pb, pi = fn(i)
            if not (os.path.exists(pb) and os.path.exists(pi)):
                missing.append(i)
                continue
            b, j = info(pb), info(pi)
            d = abs(j["dur_s"] - b["dur_s"])
            rows.append({"pair_idx": i, "bl": b, "inj": j, "abs_delta_dur_s": d,
                         "same_sr": b["sr"] == j["sr"]})
            absd.append(d)
        s = summary(absd)
        s["n_missing"] = len(missing)
        s["missing"] = missing
        s["all_same_sr"] = all(r["same_sr"] for r in rows)
        s["printed"] = PRINTED[name]
        s["formatted_2dp"] = f"{s['mean']:.2f} +- {s['ci_half_1p96se_ddof1']:.2f}"
        out["systems"][name] = {"summary": s, "rows": rows}
        print(f"{name:12s} n={s['n']} missing={s['n_missing']} mean={s['mean']:.4f} "
              f"ci(ddof1)={s['ci_half_1p96se_ddof1']:.4f} ci(ddof0)={s['ci_half_1p96se_ddof0']:.4f} "
              f"zero={s['n_zero']} -> {s['formatted_2dp']} (printed {PRINTED[name]})")
    path = os.path.join(OUT_DIR, f"table2_duration_{out['created']}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
