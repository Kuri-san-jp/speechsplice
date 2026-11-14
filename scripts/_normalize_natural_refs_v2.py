"""Normalize natural-human references *without* zero-padding.
Only sampling-rate (44100) and channel (mono) are aligned. Native duration
is preserved so the voice-cloning prompt contains real speech throughout
(no trailing silence that the model can learn to mimic).
"""
import os, torch, torchaudio
import torchaudio.functional as F

TARGET_SR = 44100
out_dir = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/_natural_refs_norm")
os.makedirs(out_dir, exist_ok=True)

natref = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/_natural_refs")
jsut = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "eval_results/subjective_eval_jsut_audio")

# JA: BASIC5000_0079 (3.73 s, UTMOS 4.197 -- highest among JSUT candidates).
#     Voice cloning has ~1 UTMOS-point loss vs the reference itself, so
#     starting from the highest-UTMOS natural source yields the best BL.
# EN: LibriTTS sample (3.25 s, UTMOS 4.16).
# ZH: FLEURS cmn_hans_cn (8.28 s, UTMOS 3.09 -- ceiling is ~2.1 for cloned BL).
selected = {
    "ja": f"{jsut}/001_GT_BASIC5000_0079.wav",
    "en": f"{natref}/en_libritts.wav",
    "zh": f"{natref}/zh_fleurs.wav",
}

for lang, src in selected.items():
    w, sr = torchaudio.load(src)
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if sr != TARGET_SR:
        w = F.resample(w.float(), sr, TARGET_SR)
        sr = TARGET_SR
    # No padding, no truncation (within reason): keep native duration
    out = f"{out_dir}/{lang}_natural_norm.wav"
    torchaudio.save(out, w, sr, encoding="PCM_S", bits_per_sample=16)
    info = torchaudio.info(out)
    print(f"  {lang}: src={src}")
    print(f"      -> {out}  sr={info.sample_rate} ch={info.num_channels} bits={info.bits_per_sample} dur={info.num_frames/info.sample_rate:.2f}s")
