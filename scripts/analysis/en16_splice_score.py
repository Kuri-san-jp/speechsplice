#!/usr/bin/env python3
"""EN単語入替の splice 再生成16ペアの採点（GPU: whisper medium.en + UTMOS22）.

出力: eval_results/en_wordrepl300_splice_inj/scores.json
  rows: {pair_idx, word_b, whisper, target_in, utmos}

Usage:
  SLURM_TIME=00:30:00 slurm_run.sh \
      python scripts/analysis/en16_splice_score.py
"""
import json
import os
import re
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import numpy as np
import soundfile as sf
import torch
import torchaudio

DIR = os.path.join(PROJECT_ROOT, "eval_results", "en_wordrepl300_splice_inj")
OUT = os.path.join(DIR, "scores.json")


def main():
    from en_wordrepl300_pairs import build_pairs
    pairs = build_pairs()[:300]
    meta = json.load(open(os.path.join(DIR, "meta.json")))["rows"]
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
    for m in meta:
        pidx = m["pair_idx"]
        p = pairs[pidx]
        fp = os.path.join(DIR, f"s{pidx:04d}_inj.wav")
        wav, sr = sf.read(fp)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        t = torch.from_numpy(wav.astype(np.float32))
        if sr != 16000:
            t = torchaudio.functional.resample(t.unsqueeze(0), sr, 16000).squeeze(0)
        with torch.no_grad():
            u = float(utmos(t.unsqueeze(0).to(device), 16000).item())
        tr = wmodel.transcribe(fp, language="en")["text"].strip()
        rows.append({"pair_idx": pidx, "word_a": p["word_a"], "word_b": p["word_b"],
                     "whisper": tr, "target_in": norm(p["word_b"]) in norm(tr),
                     "utmos": u, "delta_tok": m["delta_tok"]})
        print(f"[{pidx}] {p['word_b']} in={rows[-1]['target_in']} u={u:.3f} tr='{tr[:40]}'",
              flush=True)

    with open(OUT, "w") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)
    k = sum(r["target_in"] for r in rows)
    print(f"[Done] target_in {k}/{len(rows)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
