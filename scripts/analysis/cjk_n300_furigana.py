#!/usr/bin/env python3
"""実験2: Table 1 日本語300語の再現音声への furigana whisper 読み判定（GPU）.

対象: eval_results/cjk_n300_rewav/s{idx:04d}_{bl,oeoi}.wav（300ペア、
checkpoint_ja.json と同一条件の再現生成、TRM再現済: BL 35.00% / OEOI 62.67%）。
判定: 転写（カナ）に word_b（正しい読み、カタカナ）が含まれるか（ひらがな正規化）。
出力: furigana 判定の BL/OEOI 正解率 + McNemar + TRM（pykakasi判定）との一致率。

Usage:
  SLURM_TIME=01:00:00 slurm_run.sh \
      python scripts/analysis/cjk_n300_furigana.py
"""
import json
import math
import os

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")

import soundfile as sf
import torch
import torchaudio

MODEL_ID = "Parakeet-Inc/furigana_whisper_small_jsut"
GEN_DIR = os.path.join(PROJECT_ROOT, "eval_results", "cjk_n300_rewav")
OUT = os.path.join(GEN_DIR, "furigana_verify.json")


def norm_hira(s):
    out = []
    for c in s:
        o = ord(c)
        if 0x30A1 <= o <= 0x30F6:
            c = chr(o - 0x60)
            o = ord(c)
        if 0x3041 <= o <= 0x3096 or c == "ー":
            out.append(c)
    return "".join(out)


def wilson_half(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return 100 * z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def mcnemar(b, c):
    from math import comb
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def main():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()

    @torch.no_grad()
    def transcribe(path):
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        t = torch.from_numpy(wav).float()
        if sr != 16000:
            t = torchaudio.functional.resample(t.unsqueeze(0), sr, 16000).squeeze(0)
        feats = proc(t.numpy(), sampling_rate=16000, return_tensors="pt").input_features
        ids = model.generate(feats.to(device), max_new_tokens=200)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    src = json.load(open(os.path.join(GEN_DIR, "checkpoint_ja.json")))["results"]
    # TRM（論文判定: pykakasi正規化転写に word_b 含有）を再現音声の転写で計算
    import pykakasi
    kks = pykakasi.kakasi()

    def to_kata(text):
        return "".join(item["kana"] for item in kks.convert(text))

    rows = []
    if os.path.exists(OUT):
        rows = json.load(open(OUT)).get("rows", [])
    done = {r["pair_idx"] for r in rows}

    for r in src:
        pidx = r["pair_idx"]
        if pidx in done or r.get("status") == "error":
            continue
        exp = norm_hira(r["word_b"])
        row = {"pair_idx": pidx, "word_a": r["word_a"], "word_b": r["word_b"]}
        skip = False
        for cond in ("bl", "oeoi"):
            fp = os.path.join(GEN_DIR, f"s{pidx:04d}_{cond}.wav")
            if not os.path.exists(fp):
                skip = True
                break
            tr = transcribe(fp)
            row[f"{cond}_tr"] = tr
            row[f"{cond}_furi"] = exp in norm_hira(tr)
            row[f"{cond}_trm"] = r["word_b"] in to_kata(r[f"{cond}_transcript"])
        if skip:
            continue
        rows.append(row)
        if len(rows) % 25 == 0:
            with open(OUT, "w") as f:
                json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)
            print(f"  [{len(rows)}/300]", flush=True)

    n = len(rows)
    summary = {"n": n, "judge": MODEL_ID}
    for cond in ("bl", "oeoi"):
        kf = sum(r[f"{cond}_furi"] for r in rows)
        kt = sum(r[f"{cond}_trm"] for r in rows)
        agree = sum(1 for r in rows if r[f"{cond}_furi"] == r[f"{cond}_trm"])
        summary[cond] = {
            "furigana": {"k": kf, "pct": round(100 * kf / n, 2),
                         "wilson_half": round(wilson_half(kf, n), 2)},
            "trm_whisper_pykakasi": {"k": kt, "pct": round(100 * kt / n, 2)},
            "agreement_pct": round(100 * agree / n, 2),
        }
    b = sum(1 for r in rows if r["bl_furi"] and not r["oeoi_furi"])
    c = sum(1 for r in rows if not r["bl_furi"] and r["oeoi_furi"])
    summary["mcnemar_bl_vs_oeoi_furigana"] = {"b": b, "c": c, "p": mcnemar(b, c)}
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"[Done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
