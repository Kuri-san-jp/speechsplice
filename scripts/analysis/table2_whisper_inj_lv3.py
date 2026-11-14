#!/usr/bin/env python3
"""Table 2（システム比較）を Table 1 と同一の Whisper-Inj で再採点（GPU）.

Whisper-Inj = whisper large-v3（多言語、greedy、language=en）の転写に
目標語 word_b が含まれるか（正規化: 英数字以外除去+小文字化。従来の
medium.en 版と同じ含有定義、モデルだけ Table 1 に統一）。

対象:
- SpeechSplice: en_wordrepl100_pairs_gen の INJ 300本（不等長16は splice 差し替え）
- VoiceCraft zero-shot TTS: voicecraft_tts_wordrepl100 の INJ 300本
- EdiTTS: editts_edit_wordrepl300/wavs の EdiTTS 300本

出力: eval_results/table2_whisper_inj_lv3.json（転写・個票・McNemar込み）
Usage: SLURM_TIME=02:00:00 slurm_run.sh python scripts/analysis/table2_whisper_inj_lv3.py
"""
import json
import math
import os
import re
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
_hf = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(_hf, "modules")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch
import torchaudio

E = os.path.join(PROJECT_ROOT, "eval_results")
SYSTEMS = {
    "speechsplice": {
        "path": lambda i: os.path.join(E, "en_wordrepl100_pairs_gen", f"s{i:04d}_inj.wav"),
        "override": lambda i: os.path.join(E, "en_wordrepl300_splice_inj", f"s{i:04d}_inj.wav"),
    },
    "vc_tts": {
        "path": lambda i: os.path.join(E, "voicecraft_tts_wordrepl100", f"s{i:04d}_inj.wav"),
        "override": None,
    },
    "editts": {
        "path": lambda i: os.path.join(E, "editts_edit_wordrepl300", "wavs", f"gen_{i}_EdiTTS.wav"),
        "override": None,
    },
}


def local_snapshot(hub_id):
    d = os.path.join(os.environ["HF_HUB_CACHE"],
                     f"models--{hub_id.replace('/', '--')}", "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def wilson_half(k, n, z=1.96):
    p = k / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) * 100


def mcnemar(hits_a, hits_b):
    from scipy.stats import binomtest
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
    if b + c == 0:
        return b, c, 1.0
    return b, c, binomtest(b, b + c, 0.5).pvalue


def main():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from en_wordrepl300_pairs import build_pairs
    pairs = build_pairs()
    assert len(pairs) == 300

    device = "cuda" if torch.cuda.is_available() else "cpu"
    snap = local_snapshot("openai/whisper-large-v3")
    proc = WhisperProcessor.from_pretrained(snap)
    model = WhisperForConditionalGeneration.from_pretrained(
        snap, torch_dtype=torch.float16).to(device).eval()

    out = {"judge": "whisper-large-v3 greedy, target-word containment (Table1準拠)",
           "systems": {}}
    hits_all = {}
    for name, cfg in SYSTEMS.items():
        rows, hits = [], []
        for i, p in enumerate(pairs):
            path = cfg["path"](i)
            if cfg["override"]:
                alt = cfg["override"](i)
                if os.path.exists(alt):
                    path = alt
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            feats = proc(wav.mean(0).numpy(), sampling_rate=16000,
                         return_tensors="pt").input_features.to(device, torch.float16)
            with torch.no_grad():
                ids = model.generate(feats, language="en", task="transcribe",
                                     max_new_tokens=64)
            tr = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            hit = norm(p["word_b"]) in norm(tr)
            hits.append(hit)
            rows.append({"idx": i, "word_b": p["word_b"], "transcript": tr, "hit": bool(hit)})
            if (i + 1) % 100 == 0:
                print(f"  [{name}] {i+1}/300", flush=True)
        k = sum(hits)
        out["systems"][name] = {
            "k": k, "n": 300, "pct": round(k / 3, 2),
            "wilson_half": round(wilson_half(k, 300), 2), "rows": rows}
        hits_all[name] = hits
        print(f"[{name}] Whisper-Inj(lv3) = {k}/300 = {k/3:.2f}%", flush=True)

    for a, b in [("speechsplice", "vc_tts"), ("speechsplice", "editts")]:
        bb, cc, pv = mcnemar(hits_all[a], hits_all[b])
        out[f"mcnemar_{a}_vs_{b}"] = {"b": bb, "c": cc, "p": pv}
        print(f"McNemar {a} vs {b}: b={bb}/c={cc}, p={pv:.3g}")

    json.dump(out, open(os.path.join(E, "table2_whisper_inj_lv3.json"), "w"),
              ensure_ascii=False, indent=1)
    print("-> table2_whisper_inj_lv3.json")


if __name__ == "__main__":
    main()
