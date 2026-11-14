#!/usr/bin/env python3
"""N=100 EN word-replacement 比較実験のメトリクス計算（Table 1 プロトコル準拠）.

対象: eval_results/_cc_wavs_a/t5gemma_en s0000〜s0099 (BL/INJ) と
      eval_results/voicecraft_edit_en_n100 (EDIT, 存在するもののみ)

メトリクス（Table 1 と同一のモデル・定義）:
  - Whisper-Inj: whisper medium.en 転写に target語 (word_b) が含まれる率
    （BL の source語含有率 source_in_bl も併記）
  - IPA Change: wav2vec2-xlsr-53-espeak-cv-ft の発話全体IPA転写が BL と
    異なる率（gen_en_phoneme_n300.py と同一定義）+ 編集距離
  - UTMOS22 (tarepan/SpeechMOS utmos22_strong, 16kHz入力): BL / INJ / EDIT
  - WavLM-SV SIM: sim(BL,INJ), sim(BL,EDIT), 発話間アンカー sim(BL_i, BL_{i+7})

ファイル単位の計算結果は cache.json に保存し、再実行時はスキップする
（VoiceCraft 完了後の2回目実行で SpeechSplice 側を再計算しない）。

Usage:
  SLURM_TIME=03:00:00 slurm_run.sh \
      python scripts/analysis/vc100_metrics.py
"""
import glob
import json
import math
import os
import re
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "analysis"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ja_numerals_v2_verify import local_snapshot

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--wav_root", default="eval_results/_cc_wavs_a/t5gemma_en",
                 help="s????_bl.wav / s????_inj.wav のあるディレクトリ（PROJECT_ROOT相対）")
_ap.add_argument("--edit_root", default="eval_results/voicecraft_edit_en_n100",
                 help="s????_edit.wav のあるディレクトリ。'none' で edit 側をスキップ")
_ap.add_argument("--out_dir", default="eval_results/vc100_en_comparison")
_ap.add_argument("--n", type=int, default=100)
_ap.add_argument("--pairs_module", default=None,
                 help="data/<name>.py の build_pairs() を使用（既定: crossculture 300）")
_args = _ap.parse_args()

N_PAIRS = _args.n
RMS_MIN = 1e-3
ANCHOR_SHIFT = 7
WAV_ROOT = os.path.join(PROJECT_ROOT, _args.wav_root)
EDIT_ROOT = os.path.join(PROJECT_ROOT, _args.edit_root)
SKIP_EDIT = _args.edit_root == "none"
OUT_DIR = os.path.join(PROJECT_ROOT, _args.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_PATH = os.path.join(OUT_DIR, "cache.json")


def load16k(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    rms = float(np.sqrt((wav ** 2).mean())) if len(wav) else 0.0
    t = torch.from_numpy(wav).float()
    if sr != 16000:
        t = torchaudio.functional.resample(t.unsqueeze(0), sr, 16000).squeeze(0)
    return t, rms


def wilson_ci(k, n):
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * center, 100 * half


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return None, None
    return float(xs.mean()), 1.96 * float(xs.std()) / math.sqrt(len(xs))


def main():
    from scipy import stats as scipy_stats
    if _args.pairs_module:
        mod = __import__(_args.pairs_module)
        pairs = mod.build_pairs()[:N_PAIRS]
    else:
        from en_cross_culture_names import build_300_pairs
        pairs = build_300_pairs()[:N_PAIRS]

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device={device}", flush=True)

    # --- モデル群 ---
    import whisper as openai_whisper
    wmodel = openai_whisper.load_model(
        "medium.en", device=device,
        download_root=os.path.join(PROJECT_ROOT, ".cache", "whisper"))

    from transformers import (AutoFeatureExtractor, AutoModelForAudioXVector,
                              Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC)
    ph_path = local_snapshot("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    ph_feat = Wav2Vec2FeatureExtractor.from_pretrained(ph_path, local_files_only=True)
    ph_model = Wav2Vec2ForCTC.from_pretrained(ph_path, local_files_only=True).to(device).eval()
    ph_vocab = json.load(open(os.path.join(ph_path, "vocab.json")))
    ph_id2sym = {v: k for k, v in ph_vocab.items()}
    ph_pad = ph_vocab.get("<pad>", 0)

    wl_path = local_snapshot("microsoft/wavlm-base-plus-sv")
    wl_feat = AutoFeatureExtractor.from_pretrained(wl_path, local_files_only=True)
    wl_model = AutoModelForAudioXVector.from_pretrained(wl_path, local_files_only=True).to(device).eval()

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True).to(device).eval()
    print("[Info] models loaded", flush=True)

    @torch.no_grad()
    def whisper_tr(path):
        return wmodel.transcribe(path, language="en")["text"].strip()

    @torch.no_grad()
    def ipa_tr(t16):
        # gen_en_phoneme_n300.py と同一のCTC collapse decode
        x = ph_feat(t16.numpy(), sampling_rate=16000,
                    return_tensors="pt").input_values.to(device)
        pred = torch.argmax(ph_model(x).logits, dim=-1)[0].cpu().tolist()
        tokens, prev = [], None
        for tid in pred:
            if tid == ph_pad:
                prev = tid
                continue
            if tid == prev:
                continue
            tokens.append(ph_id2sym.get(tid, ""))
            prev = tid
        return " ".join(tokens).replace("|", " ").strip()

    @torch.no_grad()
    def wavlm_emb(t16):
        inputs = wl_feat(t16.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        e = wl_model(**inputs).embeddings[0].cpu().numpy()
        return e / np.linalg.norm(e)

    @torch.no_grad()
    def utmos_score(t16):
        return float(utmos(t16.unsqueeze(0).to(device), 16000).item())

    def norm_word(w):
        return re.sub(r"\W", "", w).lower()

    def analyze_file(path):
        """1ファイルの transcript / IPA / UTMOS / RMS を計算（キャッシュ利用）."""
        key = os.path.relpath(path, PROJECT_ROOT)
        if key in cache:
            return cache[key]
        t16, rms = load16k(path)
        ent = {"rms": rms}
        if rms >= RMS_MIN:
            ent["whisper"] = whisper_tr(path)
            ent["ipa"] = ipa_tr(t16)
            ent["utmos"] = utmos_score(t16)
        cache[key] = ent
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
        return ent

    def edit_distance(a, b):
        # gen_en_phoneme_n300.py と同一の簡易距離（位置単位比較 + 長さ差）
        return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))

    rows = []
    emb_bl, emb_inj, emb_edit = {}, {}, {}
    for i in range(N_PAIRS):
        p = pairs[i]
        blf = os.path.join(WAV_ROOT, f"s{i:04d}_bl.wav")
        injf = os.path.join(WAV_ROOT, f"s{i:04d}_inj.wav")
        editf = os.path.join(EDIT_ROOT, f"s{i:04d}_edit.wav")
        if not (os.path.exists(blf) and os.path.exists(injf)):
            continue
        bl = analyze_file(blf)
        inj = analyze_file(injf)
        row = {"sid": f"s{i:04d}", "word_a": p["word_a"], "word_b": p["word_b"],
               "text_a": p["text_a"],
               "bl": bl, "inj": inj}
        if "whisper" in bl and "whisper" in inj:
            row["source_in_bl"] = norm_word(p["word_a"]) in norm_word(bl["whisper"])
            row["target_in_inj"] = norm_word(p["word_b"]) in norm_word(inj["whisper"])
            row["ipa_changed_inj"] = bl["ipa"] != inj["ipa"]
            row["ipa_dist_inj"] = edit_distance(bl["ipa"], inj["ipa"])
        if not SKIP_EDIT and os.path.exists(editf):
            # VoiceCraft ジョブと並走する場合、書き込み途中のwavは読めないことがある
            try:
                ed = analyze_file(editf)
            except Exception as e:
                print(f"[{i}] edit read failed (likely mid-write): {str(e)[:60]}", flush=True)
                ed = None
            if ed is not None:
                row["edit"] = ed
                if "whisper" in ed and "whisper" in bl:
                    row["target_in_edit"] = norm_word(p["word_b"]) in norm_word(ed["whisper"])
                    row["ipa_changed_edit"] = bl["ipa"] != ed["ipa"]
                    row["ipa_dist_edit"] = edit_distance(bl["ipa"], ed["ipa"])
        rows.append(row)
        if i % 10 == 0:
            print(f"[{i}] done (edit={'yes' if 'edit' in row else 'no'})", flush=True)

    # --- 話者埋め込み（無音除外後の有効ペアのみ） ---
    valid = [r for r in rows if "whisper" in r["bl"] and "whisper" in r["inj"]]
    for r in valid:
        i = int(r["sid"][1:])
        tb, _ = load16k(os.path.join(WAV_ROOT, f"{r['sid']}_bl.wav"))
        ti, _ = load16k(os.path.join(WAV_ROOT, f"{r['sid']}_inj.wav"))
        emb_bl[r["sid"]] = wavlm_emb(tb)
        emb_inj[r["sid"]] = wavlm_emb(ti)
        ef = os.path.join(EDIT_ROOT, f"{r['sid']}_edit.wav")
        if "edit" in r and "whisper" in r["edit"] and os.path.exists(ef):
            te, _ = load16k(ef)
            emb_edit[r["sid"]] = wavlm_emb(te)

    sids = sorted(emb_bl.keys())
    n = len(sids)
    sim_bi = np.array([float(emb_bl[s] @ emb_inj[s]) for s in sids])
    anchor = np.array([float(emb_bl[sids[k]] @ emb_bl[sids[(k + ANCHOR_SHIFT) % n]])
                       for k in range(n)])
    sim_be = np.array([float(emb_bl[s] @ emb_edit[s]) for s in sids if s in emb_edit])

    # --- 集計 ---
    def rate(key):
        xs = [r[key] for r in valid if key in r]
        k = sum(xs)
        m, h = wilson_ci(k, len(xs))
        return {"k": int(k), "n": len(xs), "pct": round(100 * k / len(xs), 2) if xs else None,
                "wilson_ci_center": round(m, 2), "wilson_ci_half": round(h, 2)}

    ub = [r["bl"]["utmos"] for r in valid]
    ui = [r["inj"]["utmos"] for r in valid]
    ue = [r["edit"]["utmos"] for r in valid if "edit" in r and "utmos" in r["edit"]]
    # paired BL vs INJ / BL vs EDIT
    pair_be = [(r["bl"]["utmos"], r["edit"]["utmos"]) for r in valid
               if "edit" in r and "utmos" in r["edit"]]
    w_bi = scipy_stats.wilcoxon(ub, ui).pvalue if len(ub) > 10 else None
    w_be = (scipy_stats.wilcoxon([a for a, _ in pair_be], [b for _, b in pair_be]).pvalue
            if len(pair_be) > 10 else None)

    summary = {
        "n_pairs": N_PAIRS, "n_valid": len(valid),
        "excluded_silent": N_PAIRS - len(valid),
        "whisper_model": "medium.en",
        "speechsplice": {
            "source_in_bl": rate("source_in_bl"),
            "whisper_inj": rate("target_in_inj"),
            "ipa_change": rate("ipa_changed_inj"),
            "utmos_bl": mean_ci(ub), "utmos_inj": mean_ci(ui),
            "utmos_wilcoxon_p": w_bi,
        },
        "voicecraft": {
            "n_edited": len([r for r in valid if "edit" in r]),
            "whisper_inj": rate("target_in_edit"),
            "ipa_change": rate("ipa_changed_edit"),
            "utmos_edit": mean_ci(ue),
            "utmos_bl_vs_edit_wilcoxon_p": w_be,
        },
        "wavlm_sim": {
            "bl_inj": mean_ci(sim_bi), "anchor": mean_ci(anchor),
            "bl_edit": mean_ci(sim_be) if len(sim_be) else None,
        },
    }
    out = {"summary": summary, "rows": rows}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(f"[Done] -> {OUT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
