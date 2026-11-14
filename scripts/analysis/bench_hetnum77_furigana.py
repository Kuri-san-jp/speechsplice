#!/usr/bin/env python3
"""bench 77項目のBL音声を furigana Whisper でひらがな転写し、
参照データの読み（reading_expected / reading_alt）と照合する.

モデル: Parakeet-Inc/furigana_whisper_small_jsut（whisper-small のJSUT
ふりがな fine-tune、音声→ひらがな読みを直接出力）

判定（項目ごと）:
  1. 転写に正読（reading_expected）が含まれ誤読が含まれない → correct
  2. 転写に誤読（reading_alt）が含まれ正読が含まれない → wrong
  3. どちらとも決まらない場合 → 全文ひらがな刺激（stim_expected / stim_alt）
     とのCERが近い方に分類（マージン併記）、差が僅少なら ambiguous

Usage:
  SLURM_TIME=00:40:00 slurm_run.sh \
      python scripts/analysis/bench_hetnum77_furigana.py
"""
import json
import os
import re
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")

import numpy as np
import soundfile as sf
import torch
import torchaudio

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--cond", default="bl",
                 choices=["bl", "inj", "injh", "injs", "injsc", "injhs", "injhsc",
                          "injnn", "ref_c", "ref_w"],
                 help="判定対象の条件（s%%04d_<cond>.wav。sc=splice_clone, nn=NN写像。"
                      "ref_c/ref_w=参照合成の判定器妥当性検証、ファイル名 <cond>_s%%04d.wav）")
_ap.add_argument("--module", default="hetnum77", choices=["hetnum77", "ext83"],
                 help="対象セット: hetnum77=heteronym+context_numeral 77項目（既定・従来動作） / "
                      "ext83=残り6カテゴリ83項目（bench_ext83_gen）")
_args = _ap.parse_args()

MODEL_ID = "Parakeet-Inc/furigana_whisper_small_jsut"
if _args.module == "hetnum77":
    GEN_DIR = os.path.join(PROJECT_ROOT, "eval_results", "bench_hetnum77_gen")
    CATS = ("heteronym", "context_numeral")
else:  # ext83（bench_ext83.py と同一カテゴリ・同一順序）
    GEN_DIR = os.path.join(PROJECT_ROOT, "eval_results", "bench_ext83_gen")
    CATS = ("drama_specific", "okurigana", "jukujikun",
            "verb_conjugation", "juubako_yuto", "onkun_domain")
SRC = os.path.join(PROJECT_ROOT, "data", "external", "ja_tts_g2p_bench_v1.jsonl")
COND = _args.cond
OUT = os.path.join(GEN_DIR, "furigana_verify.json" if COND == "bl"
                   else f"furigana_verify_{COND}.json")
AMBIG_MARGIN = 0.02  # 全文CER差がこれ未満なら ambiguous


def norm_hira(s):
    """ひらがな・ーのみ残す（句読点・空白・カタカナ→ひらがな変換込み）."""
    out = []
    for c in s:
        o = ord(c)
        if 0x30A1 <= o <= 0x30F6:  # カタカナ→ひらがな
            c = chr(o - 0x60)
            o = ord(c)
        if 0x3041 <= o <= 0x3096 or c == "ー":
            out.append(c)
    return "".join(out)


def cer(ref, hyp):
    a, b = list(ref), list(hyp)
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if x == y else 1))
        prev = cur
    return prev[-1] / len(a)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()
    print(f"[Info] model loaded: {MODEL_ID}", flush=True)

    items = [json.loads(l) for l in open(SRC)]
    items = [it for it in items if it.get("category") in CATS]

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

    rows = []
    for i, it in enumerate(items):
        if COND in ("ref_c", "ref_w"):
            fn = os.path.join(GEN_DIR, f"{COND}_s{i:04d}.wav")
        else:
            fn = os.path.join(GEN_DIR, f"s{i:04d}_{COND}.wav")
        if not os.path.exists(fn) or os.path.getsize(fn) <= 1000:
            if COND == "bl":
                print(f"[{i}] missing/empty {os.path.basename(fn)}, skip", flush=True)
            continue
        tr_raw = transcribe(fn)
        tr = norm_hira(tr_raw)
        exp = norm_hira(it["reading_expected"])
        alt = norm_hira(it["reading_alt"])
        has_exp = exp in tr
        has_alt = alt in tr
        cer_e = cer(norm_hira(it["stim_expected"]), tr)
        cer_a = cer(norm_hira(it["stim_alt"]), tr)
        if has_exp and not has_alt:
            lab, how = "correct", "containment"
        elif has_alt and not has_exp:
            lab, how = "wrong", "containment"
        else:
            if abs(cer_e - cer_a) < AMBIG_MARGIN:
                lab, how = "ambiguous", "cer"
            else:
                lab, how = ("correct" if cer_e < cer_a else "wrong"), "cer"
        rows.append({
            "idx": i, "bench_id": it["id"], "category": it["category"],
            "target": it["target"], "expected": exp, "alt": alt,
            "transcript": tr_raw, "label": lab, "method": how,
            "cer_expected": round(cer_e, 3), "cer_alt": round(cer_a, 3),
        })
        print(f"[{i:2d}] {it['category'][:4]} {it['target']} exp={exp}/alt={alt} "
              f"-> {lab}({how}) cer_e={cer_e:.2f} cer_a={cer_a:.2f} tr='{tr_raw[:40]}'",
              flush=True)

    summary = {"model": MODEL_ID, "n": len(rows)}
    for cat in CATS:
        cr = [r for r in rows if r["category"] == cat]
        summary[cat] = {
            "n": len(cr),
            "correct": sum(1 for r in cr if r["label"] == "correct"),
            "wrong": sum(1 for r in cr if r["label"] == "wrong"),
            "ambiguous": sum(1 for r in cr if r["label"] == "ambiguous"),
        }
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"[Done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
