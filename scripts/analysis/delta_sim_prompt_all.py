#!/usr/bin/env python3
"""ΔSIM-to-prompt（BL→INJの参照話者忠実度の変化）を5系統で算出（GPU）.

Δ = sim(prompt, INJ_i) − sim(prompt, BL_i) の対平均、95%CI（1.96·SE）、対応Wilcoxon。
EdiTTSは声まね不可のため疑似プロンプト（同一話者の独立生成 gradtts-2/gen_0）での参考値。
出力: eval_results/delta_sim_prompt_all.json
"""
import json, math, os, sys
PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
_hf = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "analysis"))
import torch
torch.set_num_threads(32)  # 強制ルール: CPU最大64コア
import numpy as np
from speaker_sim_splice_rerun import emb_model, embed

E = os.path.join(PROJECT_ROOT, "eval_results")
CONFIGS = {
    "en_ss": {"prompt": f"{E}/_natural_refs_norm/en_natural_norm.wav",
              "bl": lambda i: f"{E}/en_wordrepl100_pairs_gen/s{i:04d}_bl.wav",
              "inj": lambda i: (f"{E}/en_wordrepl300_splice_inj/s{i:04d}_inj.wav"
                                if os.path.exists(f"{E}/en_wordrepl300_splice_inj/s{i:04d}_inj.wav")
                                else f"{E}/en_wordrepl100_pairs_gen/s{i:04d}_inj.wav")},
    "ja_ss": {"prompt": f"{E}/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav",
              "bl": lambda i: f"{E}/ja_wordrepl300_gen/s{i:04d}_bl.wav",
              "inj": lambda i: f"{E}/ja_wordrepl300_gen/s{i:04d}_inj.wav"},
    "zh_ss": {"prompt": f"{E}/_natural_refs_norm/zh_natural_bznsyp_norm.wav",
              "bl": lambda i: f"{E}/zh_wordrepl300_gen/s{i:04d}_bl.wav",
              "inj": lambda i: f"{E}/zh_wordrepl300_gen/s{i:04d}_inj.wav"},
    "vc_tts": {"prompt": f"{E}/_natural_refs_norm/en_natural_norm.wav",
               "bl": lambda i: f"{E}/voicecraft_tts_wordrepl100/s{i:04d}_bl.wav",
               "inj": lambda i: f"{E}/voicecraft_tts_wordrepl100/s{i:04d}_inj.wav"},
    "editts": {"prompt": f"{E}/editts_edit_wordrepl300/wavs/gen_0_gradtts-2.wav",
               "bl": lambda i: f"{E}/editts_edit_wordrepl300/wavs/gen_{i}_gradtts-1.wav",
               "inj": lambda i: f"{E}/editts_edit_wordrepl300/wavs/gen_{i}_EdiTTS.wav"},
}


def main():
    from scipy.stats import wilcoxon
    fe, model = emb_model()
    out = {}
    for name, cfg in CONFIGS.items():
        pp = embed(cfg["prompt"], fe, model)
        deltas = []
        start = 1 if name == "editts" else 0  # 疑似プロンプトに使ったitem 0は除外
        for i in range(start, 300):
            pb, pi = cfg["bl"](i), cfg["inj"](i)
            if not (os.path.exists(pb) and os.path.exists(pi)):
                continue
            eb, ei = embed(pb, fe, model), embed(pi, fe, model)
            if eb is None or ei is None:
                continue
            deltas.append(float(pp @ ei) - float(pp @ eb))
        d = np.array(deltas)
        out[name] = {"n": len(d), "delta_mean": round(float(d.mean()), 4),
                     "ci95": round(1.96 * float(d.std(ddof=1)) / math.sqrt(len(d)), 4),
                     "wilcoxon_p": float(wilcoxon(d).pvalue)}
        print(name, out[name], flush=True)
    json.dump(out, open(f"{E}/delta_sim_prompt_all.json", "w"), indent=1)
    print("-> delta_sim_prompt_all.json")


if __name__ == "__main__":
    main()
