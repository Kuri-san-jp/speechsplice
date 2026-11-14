"""Extract several BZNSYP samples (high-quality studio recording of a single
Mandarin female speaker), score them with UTMOS, and pick the highest-UTMOS
candidate as the ZH natural reference for speaker-fixed Table 2."""
import os, io, torch, torchaudio
import torchaudio.functional as F
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

_hf_home = "/mnt/.cache/huggingface" if os.path.isdir("/mnt/scripts") else os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")

print("[Info] Loading UTMOS22...")
device = "cuda" if torch.cuda.is_available() else "cpu"
mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                     trust_repo=True).to(device).eval()


def utmos(wav, sr):
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = F.resample(wav.float(), sr, 16000)
        sr = 16000
    with torch.no_grad():
        return float(mos(wav.to(device), sr))


print("[Info] Downloading BZNSYP shard 0...")
p = hf_hub_download("yvonne66/BZNSYP_new", "data/train-00000-of-00008.parquet",
                     repo_type="dataset")
tbl = pq.read_table(p)
# Try several rows with varying lengths; pick the highest-UTMOS one.
_proj = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
out_dir = f"{_proj}/eval_results/_natural_refs"
os.makedirs(out_dir, exist_ok=True)

candidates = []
# Sample a diverse set of rows: 0..N at varied indices
indices = [0, 100, 250, 500, 800, 1100, 1200]
for idx in indices:
    if idx >= tbl.num_rows: continue
    row = tbl.slice(idx, 1).to_pylist()[0]
    audio_bytes = row["audio"]["bytes"]
    text = row["text"]
    fname = row["audio"]["path"]
    buf = io.BytesIO(audio_bytes)
    w, sr = torchaudio.load(buf)
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    dur = w.shape[-1] / sr
    u = utmos(w, sr)
    candidates.append((u, idx, fname, dur, text, w, sr))
    print(f"  idx={idx} ({fname}) dur={dur:.2f}s UTMOS={u:.3f}  text={text[:30]}")

# Sort by UTMOS, pick the highest-quality one >=3s (JA/EN refs are 3.25-3.73 s
# and clone to BL 3.6-4.0). The 3-15 s window matches the JA/EN reference.
candidates.sort(key=lambda x: -x[0])
best = None
for u, idx, fname, dur, text, w, sr in candidates:
    if 3.0 <= dur <= 15.0:
        best = (u, idx, fname, dur, text, w, sr)
        break
if best is None:
    best = candidates[0]
u, idx, fname, dur, text, w, sr = best
print(f"\n[Info] Best: idx={idx} ({fname}) dur={dur:.2f}s UTMOS={u:.3f}")
print(f"  text: {text}")

# Save in native sr first
out_native = f"{out_dir}/zh_bznsyp.wav"
torchaudio.save(out_native, w, sr, encoding="PCM_S", bits_per_sample=16)
print(f"  saved: {out_native}")

# Save normalized version
TARGET_SR = 44100
if sr != TARGET_SR:
    w_n = F.resample(w.float(), sr, TARGET_SR)
else:
    w_n = w
norm_dir = f"{_proj}/eval_results/_natural_refs_norm"
os.makedirs(norm_dir, exist_ok=True)
# write to a candidate name, do NOT overwrite the FLEURS one until tested
out_norm = f"{norm_dir}/zh_natural_bznsyp_norm.wav"
torchaudio.save(out_norm, w_n, TARGET_SR, encoding="PCM_S", bits_per_sample=16)
print(f"  saved: {out_norm}")
