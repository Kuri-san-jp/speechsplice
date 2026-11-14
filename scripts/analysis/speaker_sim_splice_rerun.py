#!/usr/bin/env python3
"""話者性のsplice正式実装での再計測（WavLM、CPU可）.

計測（旧プロトコルと同一定義）:
- SIM-to-prompt: cos(prompt, BL) / cos(prompt, INJ)（±は1.96·SE、対応Wilcoxon）
- BL-INJ: 同一ペアのcos
- anchor: cos(BL_i, BL_{(i+7) mod n})（発話間変動）

Usage: python3 scripts/analysis/speaker_sim_splice_rerun.py --config ja_n300 | wordrepl
"""
import argparse
import json
import math
import os

_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
_hf = os.path.join(_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")

import numpy as np
import torch
torch.set_num_threads(32)  # 強制ルール: CPU最大64コア（余裕を見て32）
import torchaudio

ROOT = _ROOT

CONFIGS = {}
for _l in ["en", "zh", "ko", "fr", "de"]:
    CONFIGS[f"{_l}_n300"] = {
        "gen": f"{ROOT}/eval_results/utmos_speaker_fixed_{_l}_n300_splice",
        "override": None,
        "prompt": f"{ROOT}/eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav",
        "n": 300,
        "out": f"{ROOT}/eval_results/speaker_sim_{_l}_n300_splice_wavlm.json",
    }
CONFIGS.update({
    "ja_wordrepl": {
        "gen": f"{ROOT}/eval_results/ja_wordrepl300_gen",
        "override": None,
        "prompt": f"{ROOT}/eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav",
        "n": 300,
        "out": f"{ROOT}/eval_results/speaker_sim_ja_wordrepl_splice_wavlm.json",
    },
    "zh_wordrepl": {
        "gen": f"{ROOT}/eval_results/zh_wordrepl300_gen",
        "override": None,
        "prompt": f"{ROOT}/eval_results/_natural_refs_norm/zh_natural_bznsyp_norm.wav",
        "n": 300,
        "out": f"{ROOT}/eval_results/speaker_sim_zh_wordrepl_splice_wavlm.json",
    },
    "ja_n300": {
        "gen": f"{ROOT}/eval_results/utmos_speaker_fixed_ja_n300_splice",
        "override": None,
        "prompt": f"{ROOT}/eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav",
        "n": 300,
        "out": f"{ROOT}/eval_results/speaker_sim_ja_n300_splice_wavlm.json",
    },
    "wordrepl": {
        "gen": f"{ROOT}/eval_results/en_wordrepl100_pairs_gen",
        "override": f"{ROOT}/eval_results/en_wordrepl300_splice_inj",  # 不等長16ペアのINJ差し替え
        "prompt": f"{ROOT}/eval_results/_natural_refs_norm/en_natural_norm.wav",
        "n": 300,
        "out": f"{ROOT}/eval_results/speaker_sim_wordrepl_splice_wavlm.json",
    },
})


def emb_model():
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
    name = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioXVector.from_pretrained(name).to(device).eval()
    return fe, model


def embed(path, fe, model):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if float((wav ** 2).mean()) ** 0.5 < 1e-3:
        return None
    with torch.no_grad():
        ins = fe(wav.numpy(), sampling_rate=16000, return_tensors="pt")
        ins = {k: v.to(next(model.parameters()).device) for k, v in ins.items()}
        out = model(**ins).embeddings
    v = out[0].cpu().numpy()
    return v / np.linalg.norm(v)


def stats(xs):
    xs = np.array(xs)
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs))), float(xs.std(ddof=1))


def main():
    from scipy.stats import wilcoxon
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    fe, model = emb_model()
    prompt = embed(cfg["prompt"], fe, model)
    bl, inj, kept = [], [], []
    for i in range(cfg["n"]):
        pb = os.path.join(cfg["gen"], f"s{i:04d}_bl.wav")
        pi = os.path.join(cfg["gen"], f"s{i:04d}_inj.wav")
        if cfg["override"]:
            alt = os.path.join(cfg["override"], f"s{i:04d}_inj.wav")
            if os.path.exists(alt):
                pi = alt
        if not (os.path.exists(pb) and os.path.exists(pi)):
            continue
        eb, ei = embed(pb, fe, model), embed(pi, fe, model)
        if eb is None or ei is None:
            continue
        bl.append(eb)
        inj.append(ei)
        kept.append(i)
        if len(kept) % 50 == 0:
            print(f"  {len(kept)}", flush=True)
    n = len(kept)
    sp_bl = [float(prompt @ b) for b in bl]
    sp_inj = [float(prompt @ x) for x in inj]
    pair = [float(b @ x) for b, x in zip(bl, inj)]
    anchor = [float(bl[i] @ bl[(i + 7) % n]) for i in range(n)]
    res = {
        "n": n, "align": "splice", "backend": "wavlm-base-plus-sv",
        "sim_prompt_bl": stats(sp_bl), "sim_prompt_inj": stats(sp_inj),
        "wilcoxon_prompt_p": float(wilcoxon(sp_bl, sp_inj).pvalue),
        "bl_inj_pair": stats(pair), "anchor": stats(anchor),
        "wilcoxon_pair_vs_anchor_p": float(wilcoxon(pair, anchor).pvalue),
    }
    # 直接比較用: 先頭200のみの集計も併記（旧N=200と同じ土俵）
    if n >= 200:
        res["first200"] = {
            "sim_prompt_bl": stats(sp_bl[:200]), "sim_prompt_inj": stats(sp_inj[:200]),
            "wilcoxon_prompt_p": float(wilcoxon(sp_bl[:200], sp_inj[:200]).pvalue),
            "bl_inj_pair": stats(pair[:200]),
            "anchor": stats([float(bl[i] @ bl[(i + 7) % 200]) for i in range(200)]),
        }
    print(json.dumps(res, indent=1))
    json.dump(res, open(cfg["out"], "w"), indent=1)
    print("->", cfg["out"])


if __name__ == "__main__":
    main()
