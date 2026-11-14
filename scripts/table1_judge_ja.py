#!/usr/bin/env python3
"""JA wordrepl300: reading judgment with Furigana Whisper + IPA change (GPU).

- Whisper-INJ: does the Furigana Whisper transcript contain the reading of the
  target surname (word_b)? Kana normalization unifies kana forms, long-vowel
  spelling variants, and alternative kana, following the same convention as the
  existing JA judgment for Table 1.
- BL check: does the BL transcript contain the reading of the source surname (word_a)?
- IPA change: does the wav2vec2-xlsr-53-espeak-cv-ft IPA transcript differ between BL and INJ?

Output: judge.json in the generation directory (shipped as
        eval_results/table1_cross_lingual/ja_judge.json).
Usage: python python scripts/analysis/ja_wordrepl_judge.py
"""
import json
import math
import os
import re
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else "."
_hf = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(_hf, "modules")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch
import torchaudio

GEN = os.path.join(PROJECT_ROOT, "eval_results", "ja_wordrepl300_gen")

VOWEL = {}
for row_, v in [("あかがさざただなはばぱまやらわゃぁっ", "あ"), ("いきぎしじちぢにひびぴみりゐぃ", "い"),
                ("うくぐすずつづぬふぶぷむゆるゔゅぅ", "う"), ("えけげせぜてでねへべぺめれゑぇ", "え"),
                ("おこごそぞとどのほぼぽもよろをょぉ", "お"), ("ん", "ん")]:
    for c in row_:
        VOWEL[c] = v


def norm_kana(s):
    out = []
    for c in s:
        o = ord(c)
        if 0x30A1 <= o <= 0x30F6:
            c = chr(o - 0x60)
            o = ord(c)
        if 0x3041 <= o <= 0x3096 or c == "ー":
            out.append(c)
    t = []
    for c in out:
        if c == "ー" and t:
            t.append(VOWEL.get(t[-1], ""))
        else:
            t.append(c)
    s = "".join(t).replace("おう", "おお").replace("えい", "ええ")
    for a, b in [("ぢ", "じ"), ("づ", "ず"), ("ゔ", "ぶ"), ("ゐ", "い"), ("ゑ", "え"), ("を", "お")]:
        s = s.replace(a, b)
    return s


def local_snapshot(hub_id):
    d = os.path.join(os.environ["HF_HUB_CACHE"],
                     f"models--{hub_id.replace('/', '--')}", "snapshots")
    # If several snapshots exist, prefer the complete one that contains config.json
    cands = sorted(os.listdir(d), reverse=True)
    for c in cands:
        if os.path.exists(os.path.join(d, c, "config.json")):
            return os.path.join(d, c)
    return os.path.join(d, cands[0])


def wilson_half(k, n, z=1.96):
    p = k / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) * 100


def main():
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              Wav2Vec2ForCTC, Wav2Vec2Processor)
    import pykakasi
    from ja_wordrepl300_pairs import build_pairs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kks = pykakasi.kakasi()

    def reading(word):
        return norm_kana("".join(x["hira"] for x in kks.convert(word)))

    fsnap = local_snapshot("Parakeet-Inc/furigana_whisper_small_jsut")
    fproc = WhisperProcessor.from_pretrained(fsnap)
    fmodel = WhisperForConditionalGeneration.from_pretrained(
        fsnap, torch_dtype=torch.float16).to(device).eval()
    psnap = local_snapshot("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    pproc = Wav2Vec2Processor.from_pretrained(psnap)
    pmodel = Wav2Vec2ForCTC.from_pretrained(psnap).to(device).eval()

    def furigana(path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        feats = fproc(wav.mean(0).numpy(), sampling_rate=16000,
                      return_tensors="pt").input_features.to(device, torch.float16)
        with torch.no_grad():
            ids = fmodel.generate(feats, max_new_tokens=64)
        return fproc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def ipa(path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            logits = pmodel(pproc(wav.mean(0).numpy(), sampling_rate=16000,
                                  return_tensors="pt").input_values.to(device)).logits
        return pproc.batch_decode(logits.argmax(-1))[0]

    pairs = build_pairs()
    rows = []
    k_inj = k_bl = k_ipa = 0
    for p in pairs:
        i = p["pair_idx"]
        fb = os.path.join(GEN, f"s{i:04d}_bl.wav")
        fi = os.path.join(GEN, f"s{i:04d}_inj.wav")
        tb, ti = furigana(fb), furigana(fi)
        ra, rb = reading(p["word_a"]), reading(p["word_b"])
        hit_bl = ra in norm_kana(tb)
        hit_inj = rb in norm_kana(ti)
        ib, ii = ipa(fb), ipa(fi)
        ch = ib != ii
        k_bl += hit_bl
        k_inj += hit_inj
        k_ipa += ch
        rows.append({"pair_idx": i, "word_a": p["word_a"], "word_b": p["word_b"],
                     "read_a": ra, "read_b": rb, "bl_tr": tb, "inj_tr": ti,
                     "bl_hit": hit_bl, "inj_hit": hit_inj, "ipa_changed": ch})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/300", flush=True)
    n = len(rows)
    summary = {
        "n": n,
        "whisper_inj": {"k": k_inj, "pct": round(k_inj / n * 100, 2),
                        "wilson_half": round(wilson_half(k_inj, n), 2)},
        "bl_source_reading": {"k": k_bl, "pct": round(k_bl / n * 100, 2)},
        "ipa_change": {"k": k_ipa, "pct": round(k_ipa / n * 100, 2),
                       "wilson_half": round(wilson_half(k_ipa, n), 2)},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(GEN, "judge.json"), "w"), ensure_ascii=False, indent=1)
    print("-> judge.json")


if __name__ == "__main__":
    main()
