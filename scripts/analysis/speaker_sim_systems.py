#!/usr/bin/env python3
"""他システム（VC zero-shot TTS / EdiTTS）の話者性計測（WavLM、CPU可、N=300）.

- VC-TTS: SIM-to-prompt(BL/INJ) + BL-INJ対 + anchor（SSと同一プロトコル）
- EdiTTS: 素生成(gradtts-1) vs 編集後(EdiTTS) の対類似度 + 素生成同士のanchor
  （声まね不可のためSIM-to-promptは対象外＝編集の話者自己保存性のみ）
"""
import json, math, os
import numpy as np
import torch
torch.set_num_threads(32)  # 強制ルール: CPU最大64コア（余裕を見て32）, torchaudio

ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
import sys
sys.path.insert(0, f"{ROOT}/scripts/analysis")
from speaker_sim_splice_rerun import emb_model, embed, stats

def main():
    from scipy.stats import wilcoxon
    fe, model = emb_model()
    E = f"{ROOT}/eval_results"
    out = {}

    # VC zero-shot TTS
    prompt = embed(f"{E}/_natural_refs_norm/en_natural_norm.wav", fe, model)
    bl, inj = [], []
    for i in range(300):
        eb = embed(f"{E}/voicecraft_tts_wordrepl100/s{i:04d}_bl.wav", fe, model)
        ei = embed(f"{E}/voicecraft_tts_wordrepl100/s{i:04d}_inj.wav", fe, model)
        if eb is None or ei is None:
            continue
        bl.append(eb); inj.append(ei)
        if len(bl) % 100 == 0:
            print(f"  vc {len(bl)}", flush=True)
    n = len(bl)
    sp_bl = [float(prompt @ b) for b in bl]
    sp_inj = [float(prompt @ x) for x in inj]
    pair = [float(b @ x) for b, x in zip(bl, inj)]
    anchor = [float(bl[i] @ bl[(i + 7) % n]) for i in range(n)]
    out["vc_tts"] = {"n": n, "sim_prompt_bl": stats(sp_bl), "sim_prompt_inj": stats(sp_inj),
                     "wilcoxon_prompt_p": float(wilcoxon(sp_bl, sp_inj).pvalue),
                     "bl_inj_pair": stats(pair), "anchor": stats(anchor)}

    # EdiTTS（自己保存性のみ）
    src, ed = [], []
    for i in range(300):
        es = embed(f"{E}/editts_edit_wordrepl300/wavs/gen_{i}_gradtts-1.wav", fe, model)
        ee = embed(f"{E}/editts_edit_wordrepl300/wavs/gen_{i}_EdiTTS.wav", fe, model)
        if es is None or ee is None:
            continue
        src.append(es); ed.append(ee)
        if len(src) % 100 == 0:
            print(f"  editts {len(src)}", flush=True)
    m = len(src)
    pair_e = [float(a @ b) for a, b in zip(src, ed)]
    anchor_e = [float(src[i] @ src[(i + 7) % m]) for i in range(m)]
    out["editts"] = {"n": m, "raw_edit_pair": stats(pair_e), "anchor": stats(anchor_e),
                     "wilcoxon_pair_vs_anchor_p": float(wilcoxon(pair_e, anchor_e).pvalue)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{E}/speaker_sim_systems_wavlm.json", "w"), indent=1)
    print("-> speaker_sim_systems_wavlm.json")

if __name__ == "__main__":
    main()
