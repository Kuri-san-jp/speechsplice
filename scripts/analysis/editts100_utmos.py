#!/usr/bin/env python3
"""EdiTTS N=100 の UTMOS22 + whisper転写（GPU）.

対象: eval_results/editts_edit_en_n100/wavs/gen_{i}_{gradtts-1,gradtts-2,EdiTTS,baseline}.wav
  - gradtts-1: text_a の Grad-TTS 素の生成（EdiTTS自身のBL相当）
  - gradtts-2: text_b の Grad-TTS 素の生成
  - EdiTTS:    content-edited 出力（本命）
  - baseline:  EdiTTS論文の切り貼りbaseline（参考）
UTMOS22 (utmos22_strong, 16kHz入力) を4条件で、whisper medium.en 転写を
EdiTTS/baseline 条件で計測し target語含有も判定する。

Usage:
  SLURM_TIME=01:00:00 slurm_run.sh \
      python scripts/analysis/editts100_utmos.py
"""
import json
import math
import os
import re
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import argparse

import numpy as np
import soundfile as sf
import torch
import torchaudio

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=100)
_ap.add_argument("--pairs_module", default=None,
                 help="data/<name>.py の build_pairs()（既定: crossculture 300）")
_ap.add_argument("--wav_dir", default=None,
                 help="gen_{i}_*.wav のディレクトリ（既定: editts_edit_en_n100/wavs）")
_args = _ap.parse_args()

N_PAIRS = _args.n
RMS_MIN = 1e-3
_default_dir = os.path.join(PROJECT_ROOT, "eval_results", "editts_edit_en_n100", "wavs")
WAV_DIR = (os.path.join(PROJECT_ROOT, _args.wav_dir) if _args.wav_dir else _default_dir)
OUT = os.path.join(os.path.dirname(WAV_DIR), "utmos_results.json")
CONDS = ["gradtts-1", "gradtts-2", "EdiTTS", "baseline"]


def load16k(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = wav.astype(np.float32)
    if np.abs(wav).max() > 1.5:  # int16スケールで保存されている場合
        wav = wav / 32768.0
    rms = float(np.sqrt((wav ** 2).mean())) if len(wav) else 0.0
    t = torch.from_numpy(wav).float()
    if sr != 16000:
        t = torchaudio.functional.resample(t.unsqueeze(0), sr, 16000).squeeze(0)
    return t, rms


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return None, None
    return float(xs.mean()), 1.96 * float(xs.std()) / math.sqrt(len(xs))


def main():
    if _args.pairs_module:
        _mod = __import__(_args.pairs_module)
        pairs = _mod.build_pairs()[:N_PAIRS]
    else:
        from en_cross_culture_names import build_300_pairs
        pairs = build_300_pairs()[:N_PAIRS]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True).to(device).eval()
    import whisper as openai_whisper
    wmodel = openai_whisper.load_model(
        "medium.en", device=device,
        download_root=os.path.join(PROJECT_ROOT, ".cache", "whisper"))

    def norm(s):
        return re.sub(r"\W", "", s).lower()

    rows = []
    for i in range(N_PAIRS):
        p = pairs[i]
        row = {"idx": i, "word_a": p["word_a"], "word_b": p["word_b"], "text_a": p["text_a"]}
        for c in CONDS:
            fp = os.path.join(WAV_DIR, f"gen_{i}_{c}.wav")
            if not os.path.exists(fp):
                continue
            t16, rms = load16k(fp)
            if rms < RMS_MIN:
                row[c] = {"rms": rms}
                continue
            with torch.no_grad():
                u = float(utmos(t16.unsqueeze(0).to(device), 16000).item())
            ent = {"rms": rms, "utmos": u}
            if c in ("EdiTTS", "baseline"):
                tr = wmodel.transcribe(fp, language="en")["text"].strip()
                ent["whisper"] = tr
                ent["target_in"] = norm(p["word_b"]) in norm(tr)
            row[c] = ent
        rows.append(row)
        if i % 10 == 0:
            print(f"[{i}] {row.get('EdiTTS', {}).get('utmos')}", flush=True)

    summary = {"n": N_PAIRS, "utmos": {}, "target_in": {}}
    for c in CONDS:
        us = [r[c]["utmos"] for r in rows if c in r and "utmos" in r[c]]
        summary["utmos"][c] = {"n": len(us), "mean_ci95": mean_ci(us)}
    for c in ("EdiTTS", "baseline"):
        xs = [r[c]["target_in"] for r in rows if c in r and "target_in" in r[c]]
        summary["target_in"][c] = {"n": len(xs), "k": int(sum(xs)),
                                   "pct": round(100 * sum(xs) / len(xs), 2) if xs else None}
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"[Done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
