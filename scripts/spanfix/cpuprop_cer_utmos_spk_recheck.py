#!/usr/bin/env python3
"""CPU recheck for audit items C1, C2, C3, T7 (2026-10-02, note/2026-10-02_gpuneeds2_report.md).

Reads only existing result files (and, with --wavlm, the existing wav files). Writes one new JSON
under eval_results/_provenance/. Never writes into existing result files or the manuscript.

  C1  Appendix H tab:cer, EN INJ rows. INJ transcripts of the final Table 1 EN set
      (eval_results/en_wordrepl300_splice_inj/final_merged_300.json, Whisper medium.en,
      284 equal-length originals + 16 splice-regenerated pairs) instead of the 7/22
      transcripts in eval_results/en_wordrepl100_metrics/results.json.
      BL transcripts: en_wordrepl100_metrics/results.json (medium.en, BL wavs unchanged since 7/14).
  C2  Appendix H tab:cer, ZH BL rows from the current eval_results/zh_wordrepl300_gen/judge.json
      (after the 8/4 regeneration of BL 67 and 76).
      JA column is recomputed too (must reproduce the printed values unchanged).
  C3  Table 2 SpeechSplice UTMOS p: two-sided paired Wilcoxon over per-clip UTMOS,
      BL = en_wordrepl100_pairs_gen/results.json bl_utmos,
      INJ = en_wordrepl300_splice_inj/final_merged_300.json utmos (final set).
      VoiceCraft and EdiTTS p are recomputed from their files as a check.
  T7  Table 2 EdiTTS BL--INJ speaker similarity: eval_results/speaker_sim_systems_wavlm.json
      editts.raw_edit_pair. With --wavlm the 300 cosines are recomputed on CPU from the wavs
      (microsoft/wavlm-base-plus-sv, local cache, CUDA hidden).

CER definition and normalization are copied from arr_resubmission/compute_cer_full.py
(that script is not run because it writes into arr_resubmission/).

Usage (CPU only):
  taskset -c 0-12 python3 scripts/spanfix/cpuprop_cer_utmos_spk_recheck.py [--wavlm]
"""
import os
import sys

# ---- environment: CPU only, 13 threads, offline caches, no pyopenjtalk ----
N_THREADS = "13"
for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_k] = N_THREADS
ROOT = os.environ.get("REPO_ROOT", "/mnt")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(ROOT, ".cache", "numba")
sys.modules["pyopenjtalk"] = None  # block: import pyopenjtalk raises ImportError

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
import unicodedata

import numpy as np
from scipy.stats import wilcoxon

E = os.path.join(ROOT, "eval_results")
OUT_DIR = os.path.join(E, "_provenance")

PRINTED = {  # naacl2027/main.tex as of 2026-10-02 10:21
    "tab:cer": {
        "EN bl_src": 0.008, "EN bl_tgt": 0.250, "EN inj_tgt": 0.025, "EN inj_src": 0.254,
        "JA bl_src": 0.047, "JA bl_tgt": 0.209, "JA inj_tgt": 0.050, "JA inj_src": 0.215,
        "ZH bl_src": 0.048, "ZH bl_tgt": 0.250, "ZH inj_tgt": 0.045, "ZH inj_src": 0.246,
        "caption_ci_half_range": [0.005, 0.020],
    },
    "tab:system_comparison": {
        "SS utmos p": 0.10, "VC utmos p": 0.54, "EdiTTS utmos p": 0.09,
        "SS utmos BL": "3.86 +- 0.05", "SS utmos INJ": "3.82 +- 0.05",
        "EdiTTS BL--INJ": "--",
    },
}


def load_mod(p):
    s = importlib.util.spec_from_file_location("m_" + os.path.basename(p)[:-3], p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mtime(p):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))


def ed(a, b):  # same as compute_cer_full.py
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def summ(v):  # same as compute_cer_full.py: mean, 1.96 * sqrt(var_{n-1} / n)
    n = len(v)
    m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    return {"n": n, "cer": m, "ci_half": 1.96 * (var / n) ** 0.5}


def norm_en(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s)).strip()


def norm_zh(s):
    return re.sub(r"[^一-鿿]", "", unicodedata.normalize("NFKC", s))


def norm_kata(s):
    return re.sub(r"[^ァ-ヶー]", "", unicodedata.normalize("NFKC", s))


def cer_en():
    en = load_mod(os.path.join(ROOT, "data/en_wordrepl_500.py"))
    tmpl = list(en.TEMPLATES.values())[:6]
    met = json.load(open(os.path.join(E, "en_wordrepl100_metrics/results.json")))["rows"]
    fin = json.load(open(os.path.join(E, "en_wordrepl300_splice_inj/final_merged_300.json")))["rows"]
    gen = json.load(open(os.path.join(E, "en_wordrepl100_pairs_gen/results.json")))["rows"]
    meta16 = [r["pair_idx"] for r in
              json.load(open(os.path.join(E, "en_wordrepl300_splice_inj/meta.json")))["rows"]]
    assert len(met) == len(fin) == len(gen) == 300
    old = {k: [] for k in ["bl_src", "bl_tgt", "inj_tgt", "inj_src"]}
    new = {k: [] for k in ["bl_src", "bl_tgt", "inj_tgt", "inj_src"]}
    diff_rows, text_b_mismatch = [], []
    for i in range(300):
        r, f, g = met[i], fin[i], gen[i]
        assert f["pair_idx"] == i and g["pair_idx"] == i and r["sid"] == f"s{i:04d}"
        assert r["word_b"] == f["word_b"] == g["word_b"]
        src = norm_en(r["text_a"])
        tgt = norm_en(tmpl[i // 50].format(word=r["word_b"]))
        if tgt != norm_en(g["text_b"]):
            text_b_mismatch.append(i)
        bl = norm_en(r["bl"]["whisper"])
        inj_old = norm_en(r["inj"]["whisper"])
        inj_new = norm_en(f["whisper"])
        if inj_old != inj_new:
            diff_rows.append({"pair_idx": i, "in_splice16": i in meta16,
                              "old": r["inj"]["whisper"], "new": f["whisper"],
                              "source": f.get("source")})
        for d, inj in ((old, inj_old), (new, inj_new)):
            d["bl_src"].append(ed(src, bl) / len(src))
            d["bl_tgt"].append(ed(tgt, bl) / len(tgt))
            d["inj_tgt"].append(ed(tgt, inj) / len(tgt))
            d["inj_src"].append(ed(src, inj) / len(src))
    splice16_wavs = {i: os.path.exists(os.path.join(E, f"en_wordrepl300_splice_inj/s{i:04d}_inj.wav"))
                     for i in meta16}
    return ({k: summ(v) for k, v in old.items()}, {k: summ(v) for k, v in new.items()},
            {"inj_transcript_changed": diff_rows, "n_changed": len(diff_rows),
             "splice16_pairs": meta16, "splice16_wav_exists": splice16_wavs,
             "target_text_mismatch_vs_pairs_gen_text_b": text_b_mismatch})


def cer_ja():
    import pykakasi
    kk = pykakasi.kakasi()
    kana = lambda s: "".join(x["kana"] for x in kk.convert(s))
    ja = load_mod(os.path.join(ROOT, "data/ja_wordrepl300_pairs.py"))
    pairs = {p["pair_idx"]: p for p in ja.build_pairs()}
    rows = json.load(open(os.path.join(E, "ja_wordrepl300_gen/judge.json")))["rows"]
    c = {k: [] for k in ["bl_src", "bl_tgt", "inj_tgt", "inj_src"]}
    for r in rows:
        p = pairs[int(r["pair_idx"])]
        src = norm_kata(kana(p["text_a"].replace(p["word_a"], r["read_a"])))
        tgt = norm_kata(kana(p["text_b"].replace(p["word_b"], r["read_b"])))
        bl, inj = norm_kata(r["bl_tr"]), norm_kata(r["inj_tr"])
        c["bl_src"].append(ed(src, bl) / len(src))
        c["bl_tgt"].append(ed(tgt, bl) / len(tgt))
        c["inj_tgt"].append(ed(tgt, inj) / len(tgt))
        c["inj_src"].append(ed(src, inj) / len(src))
    return {k: summ(v) for k, v in c.items()}


def cer_zh():
    zh = load_mod(os.path.join(ROOT, "data/zh_wordrepl300_pairs.py"))
    pairs = {p["pair_idx"]: p for p in zh.build_pairs()}
    rows = json.load(open(os.path.join(E, "zh_wordrepl300_gen/judge.json")))["rows"]
    c = {k: [] for k in ["bl_src", "bl_tgt", "inj_tgt", "inj_src"]}
    per_pair = {}
    for r in rows:
        i = int(r["pair_idx"])
        p = pairs[i]
        src, tgt = norm_zh(p["text_a"]), norm_zh(p["text_b"])
        bl, inj = norm_zh(r["bl_tr"]), norm_zh(r["inj_tr"])
        c["bl_src"].append(ed(src, bl) / len(src))
        c["bl_tgt"].append(ed(tgt, bl) / len(tgt))
        c["inj_tgt"].append(ed(tgt, inj) / len(tgt))
        c["inj_src"].append(ed(src, inj) / len(src))
        if i in (67, 76):
            per_pair[i] = {"bl_tr": r["bl_tr"], "bl_src": c["bl_src"][-1], "bl_tgt": c["bl_tgt"][-1]}
    return {k: summ(v) for k, v in c.items()}, per_pair


def utmos_p():
    gen = json.load(open(os.path.join(E, "en_wordrepl100_pairs_gen/results.json")))
    fin = json.load(open(os.path.join(E, "en_wordrepl300_splice_inj/final_merged_300.json")))["rows"]
    meta16 = [r["pair_idx"] for r in
              json.load(open(os.path.join(E, "en_wordrepl300_splice_inj/meta.json")))["rows"]]
    rows = gen["rows"]
    bl = np.array([r["bl_utmos"] for r in rows])
    inj_old = np.array([r["inj_utmos"] for r in rows])
    inj_new = np.array([f["utmos"] for f in fin])
    changed = [i for i in range(300) if inj_old[i] != inj_new[i]]

    def mci(x):
        return [float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x))),
                float(1.96 * x.std(ddof=0) / math.sqrt(len(x)))]

    out = {"ss": {
        "n": int(len(bl)),
        "p_old_recomputed": float(wilcoxon(bl, inj_old).pvalue),
        "p_old_stored": gen["summary"]["wilcoxon_pval"],
        "p_new": float(wilcoxon(bl, inj_new).pvalue),
        "bl_mean_ci_ddof1_ddof0": mci(bl),
        "inj_new_mean_ci_ddof1_ddof0": mci(inj_new),
        "inj_changed_pairs": changed,
        "inj_changed_subset_of_splice16": set(changed) <= set(meta16),
    }}
    vc = json.load(open(os.path.join(E, "voicecraft_tts_wordrepl100_metrics/results.json")))["rows"]
    vb = np.array([r["bl"]["utmos"] for r in vc])
    vi = np.array([r["inj"]["utmos"] for r in vc])
    out["voicecraft"] = {"n": int(len(vb)), "p": float(wilcoxon(vb, vi).pvalue),
                         "bl": mci(vb), "inj": mci(vi)}
    et = json.load(open(os.path.join(E, "editts_edit_wordrepl300/utmos_results.json")))["rows"]
    eb = np.array([r["gradtts-1"]["utmos"] for r in et])
    ei = np.array([r["EdiTTS"]["utmos"] for r in et])
    out["editts"] = {"n": int(len(eb)), "p": float(wilcoxon(eb, ei).pvalue),
                     "bl": mci(eb), "inj": mci(ei)}
    return out


def spk_editts(do_wavlm):
    p = os.path.join(E, "speaker_sim_systems_wavlm.json")
    d = json.load(open(p))["editts"]
    wavs = []
    for i in range(300):
        a = os.path.join(E, f"editts_edit_wordrepl300/wavs/gen_{i}_gradtts-1.wav")
        b = os.path.join(E, f"editts_edit_wordrepl300/wavs/gen_{i}_EdiTTS.wav")
        wavs.append((a, b))
    newest = max(max(os.path.getmtime(a), os.path.getmtime(b)) for a, b in wavs)
    out = {"file": p, "file_mtime": mtime(p), "n": d["n"],
           "raw_edit_pair_mean_ci_sd": d["raw_edit_pair"],
           "all_300_wav_pairs_exist": all(os.path.exists(a) and os.path.exists(b) for a, b in wavs),
           "newest_wav_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)),
           "json_newer_than_all_wavs": os.path.getmtime(p) > newest}
    if do_wavlm:
        import torch
        torch.set_num_threads(int(N_THREADS))
        import torchaudio
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
        name = "microsoft/wavlm-base-plus-sv"
        fe = AutoFeatureExtractor.from_pretrained(name, local_files_only=True)
        model = AutoModelForAudioXVector.from_pretrained(name, local_files_only=True).eval()

        def embed(path):  # same as scripts/analysis/speaker_sim_splice_rerun.py embed()
            wav, sr = torchaudio.load(path)
            wav = wav.mean(0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if float((wav ** 2).mean()) ** 0.5 < 1e-3:
                return None
            with torch.no_grad():
                ins = fe(wav.numpy(), sampling_rate=16000, return_tensors="pt")
                v = model(**ins).embeddings[0].numpy()
            return v / np.linalg.norm(v)

        cos, skipped = [], []
        for i, (a, b) in enumerate(wavs):
            ea, eb = embed(a), embed(b)
            if ea is None or eb is None:
                skipped.append(i)
                continue
            cos.append(float(ea @ eb))
        x = np.array(cos)
        out["wavlm_cpu_recompute"] = {
            "n": int(len(x)), "skipped_low_rms": skipped, "mean": float(x.mean()),
            "ci_half_1p96se_ddof1": float(1.96 * x.std(ddof=1) / math.sqrt(len(x))),
            "sd": float(x.std(ddof=1)), "torch": torch.__version__}
    return out


def fmt3(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavlm", action="store_true", help="recompute the EdiTTS cosines on CPU")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    en_old, en_new, en_info = cer_en()
    ja = cer_ja()
    zh, zh_pairs = cer_zh()
    new_cells = {}
    for lang, d in (("EN", en_new), ("JA", ja), ("ZH", zh)):
        for k, v in d.items():
            new_cells[f"{lang} {k}"] = v
    old_cells_recomputed_en = {f"EN {k}": v for k, v in en_old.items()}
    halfs = [v["ci_half"] for v in new_cells.values()]

    print("== C1/C2 tab:cer (printed -> recomputed)")
    for k, v in new_cells.items():
        pr = PRINTED["tab:cer"][k]
        flag = "" if fmt3(v["cer"]) == f"{pr:.3f}" else "  <-- changes"
        print(f"{k:12s} printed {pr:.3f}  new {v['cer']:.4f} ({fmt3(v['cer'])}) +- {v['ci_half']:.4f}{flag}")
    print("EN with the old 7/22 INJ transcripts (reproduction check):",
          {k: round(v["cer"], 4) for k, v in old_cells_recomputed_en.items()})
    print(f"EN INJ transcripts changed: {en_info['n_changed']} pairs ->",
          [r["pair_idx"] for r in en_info["inj_transcript_changed"]],
          "all in splice16:", all(r["in_splice16"] for r in en_info["inj_transcript_changed"]))
    print("EN target text mismatches vs pairs_gen text_b:", en_info["target_text_mismatch_vs_pairs_gen_text_b"])
    print("ZH pairs 67/76 now:", json.dumps(zh_pairs, ensure_ascii=False))
    print(f"CI half-width range over the 12 new cells: {min(halfs):.4f} .. {max(halfs):.4f} "
          f"-> {min(halfs):.3f} to {max(halfs):.3f}")

    up = utmos_p()
    print("== C3 Table 2 UTMOS p")
    print(json.dumps(up, indent=1, default=str))

    sp = spk_editts(args.wavlm)
    print("== T7 EdiTTS BL--INJ speaker similarity")
    print(json.dumps(sp, indent=1))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "script": os.path.abspath(__file__), "created": ts,
        "inputs": {p: {"md5": md5(os.path.join(E, p)), "mtime": mtime(os.path.join(E, p))} for p in [
            "en_wordrepl100_metrics/results.json",
            "en_wordrepl300_splice_inj/final_merged_300.json",
            "en_wordrepl300_splice_inj/meta.json",
            "en_wordrepl100_pairs_gen/results.json",
            "ja_wordrepl300_gen/judge.json",
            "zh_wordrepl300_gen/judge.json",
            "voicecraft_tts_wordrepl100_metrics/results.json",
            "editts_edit_wordrepl300/utmos_results.json",
            "speaker_sim_systems_wavlm.json"]},
        "printed": PRINTED,
        "C1_C2_tab_cer": {"new_cells": new_cells,
                          "en_old_inj_transcripts_recomputed": old_cells_recomputed_en,
                          "en_info": en_info, "zh_pairs_67_76": zh_pairs,
                          "ci_half_range": [min(halfs), max(halfs)]},
        "C3_utmos_p": up,
        "T7_editts_speaker_similarity": sp,
    }
    path = os.path.join(OUT_DIR, f"cpuprop_cer_utmos_spk_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("->", path)


if __name__ == "__main__":
    main()
