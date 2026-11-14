#!/usr/bin/env python3
"""3システム（SS/VC-TTS/EdiTTS）のF0保存性（pyworld harvest、CPU、N=300）.

各システムで BL(または素生成) vs INJ(または編集後) の F0平均差と|Δf0|、
anchor(BL_i vs BL_{i+7})の|Δf0| を同一プロトコルで計測。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "32")  # 強制ルール: CPU最大64コア
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
import glob, json
import numpy as np
import pyworld
import soundfile as sf

ROOT = os.environ.get("REPO_ROOT", "/mnt")
E = f"{ROOT}/eval_results"

SYSTEMS = {
    "speechsplice": {
        "bl": lambda i: f"{E}/en_wordrepl100_pairs_gen/s{i:04d}_bl.wav",
        "inj": lambda i: (f"{E}/en_wordrepl300_splice_inj/s{i:04d}_inj.wav"
                          if os.path.exists(f"{E}/en_wordrepl300_splice_inj/s{i:04d}_inj.wav")
                          else f"{E}/en_wordrepl100_pairs_gen/s{i:04d}_inj.wav")},
    "vc_tts": {
        "bl": lambda i: f"{E}/voicecraft_tts_wordrepl100/s{i:04d}_bl.wav",
        "inj": lambda i: f"{E}/voicecraft_tts_wordrepl100/s{i:04d}_inj.wav"},
    "ja_wordrepl": {
        "bl": lambda i: f"{E}/ja_wordrepl300_gen/s{i:04d}_bl.wav",
        "inj": lambda i: f"{E}/ja_wordrepl300_gen/s{i:04d}_inj.wav"},
    "zh_wordrepl": {
        "bl": lambda i: f"{E}/zh_wordrepl300_gen/s{i:04d}_bl.wav",
        "inj": lambda i: f"{E}/zh_wordrepl300_gen/s{i:04d}_inj.wav"},
    "editts": {
        "bl": lambda i: f"{E}/editts_edit_wordrepl300/wavs/gen_{i}_gradtts-1.wav",
        "inj": lambda i: f"{E}/editts_edit_wordrepl300/wavs/gen_{i}_EdiTTS.wav"},
}


def f0_mean(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    x = x.astype(np.float64)
    f0, t = pyworld.harvest(x, sr, f0_floor=60.0, f0_ceil=500.0)
    v = f0[f0 > 0]
    return float(v.mean()) if len(v) > 20 else None


def main():
    out = {}
    for name, cfg in SYSTEMS.items():
        bl_m, inj_m = {}, {}
        for i in range(300):
            pb, pi = cfg["bl"](i), cfg["inj"](i)
            if not (os.path.exists(pb) and os.path.exists(pi)):
                continue
            mb, mi = f0_mean(pb), f0_mean(pi)
            if mb is None or mi is None:
                continue
            bl_m[i], inj_m[i] = mb, mi
            if len(bl_m) % 100 == 0:
                print(f"  [{name}] {len(bl_m)}", flush=True)
        ks = sorted(bl_m)
        n = len(ks)
        d = [inj_m[k] - bl_m[k] for k in ks]
        ad = [abs(x) for x in d]
        anc = [abs(bl_m[ks[i]] - bl_m[ks[(i + 7) % n]]) for i in range(n)]
        out[name] = {"n": n, "f0_mean_diff": round(float(np.mean(d)), 2),
                     "abs_diff": round(float(np.mean(ad)), 2),
                     "anchor_abs_diff": round(float(np.mean(anc)), 2)}
        print(name, out[name], flush=True)
    json.dump(out, open(f"{E}/f0_systems_v2.json", "w"), indent=1)
    print("-> f0_systems.json")


if __name__ == "__main__":
    main()
