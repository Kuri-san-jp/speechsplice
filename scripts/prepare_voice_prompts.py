"""Rebuild the voice-cloning prompts from their public corpora and check them against the files used.

The prompts are not redistributed here. Each is a corpus recording converted to 44.1 kHz
mono 16-bit PCM by torchaudio (functional.resample with default arguments), exactly as
scripts/_normalize_natural_refs_v2.py and scripts/_extract_bznsyp_refs.py did:

  eval_results/_natural_refs_norm/en_natural_norm.wav        LibriTTS test-clean 260_123440_000002_000004.wav
  eval_results/_natural_refs_norm/zh_natural_norm.wav        FLEURS cmn_hans_cn test 10040380210557600780.wav
  eval_results/_natural_refs_norm/ja_natural_norm.wav        JSUT ver1.1 basic5000/wav/BASIC5000_0079.wav
  eval_results/_natural_refs_norm/zh_natural_bznsyp_norm.wav Databaker BZNSYP (Hugging Face yvonne66/BZNSYP_new,
                                                             data/train-00000-of-00008.parquet, row 800 = 000801)
  eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav
                                                             JSUT BASIC5000_0011, unchanged (also docs/audio/prompt/)

Usage (inside my-lab-image, repository at /mnt):
  python3 scripts/prepare_voice_prompts.py --libritts_wav <...>/260_123440_000002_000004.wav \
      --fleurs_wav <...>/10040380210557600780.wav --jsut_dir <...>/jsut_ver1.1
  (BZNSYP is read from the Hugging Face cache; download it with scripts/download_models.py --only prompts)
Each output is compared with the MD5 of the file the experiments used; a mismatch is reported.
"""
import argparse
import hashlib
import io
import os
import shutil

ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get(
    "REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NORM = os.path.join(ROOT, "eval_results", "_natural_refs_norm")
JSUT_DIR = os.path.join(ROOT, "eval_results", "subjective_eval_jsut_audio")
TARGET_SR = 44100
BZNSYP_ROW = 800  # the utterance 000801 (identified by matching the decoded samples)
EXPECTED = {
    "en_natural_norm.wav": "ea7e3416d4984961a25ba231ca136374",
    "zh_natural_norm.wav": "440a3eb1a3bc8968401cdc13395d92ee",
    "ja_natural_norm.wav": "3a924426f00a7206fa8e557fa2cafe28",
    "zh_natural_bznsyp_norm.wav": "d3395c36d22484c47116925e8bb6b2eb",
    "000_GT_BASIC5000_0011.wav": "129902fbafbe7437c411a5c40028095e",
    "001_GT_BASIC5000_0079.wav": "a51cc52c7e7b5c3f355e72d1e6c15fc6",
}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def check(path):
    got, want = md5(path), EXPECTED[os.path.basename(path)]
    print(f"  {os.path.relpath(path, ROOT)}  md5 {got}  {'OK' if got == want else 'MISMATCH (expected ' + want + ')'}")
    return got == want


def normalize(src_wave, sr, out):
    """mono + 44.1 kHz + PCM_S16, as in scripts/_normalize_natural_refs_v2.py"""
    import torchaudio
    import torchaudio.functional as F
    w = src_wave
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    if sr != TARGET_SR:
        w = F.resample(w.float(), sr, TARGET_SR)
    torchaudio.save(out, w, TARGET_SR, encoding="PCM_S", bits_per_sample=16)


def from_file(src, out):
    import torchaudio
    w, sr = torchaudio.load(src)
    normalize(w, sr, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libritts_wav")
    ap.add_argument("--fleurs_wav")
    ap.add_argument("--jsut_dir")
    ap.add_argument("--bznsyp_wav", help="the BZNSYP utterance as a wav file (instead of the parquet row)")
    a = ap.parse_args()
    os.makedirs(NORM, exist_ok=True)
    os.makedirs(JSUT_DIR, exist_ok=True)
    ok = True
    out = os.path.join(JSUT_DIR, "000_GT_BASIC5000_0011.wav")
    shutil.copyfile(os.path.join(ROOT, "docs", "audio", "prompt", "jsut-basic5000-0011.wav"), out)
    ok &= check(out)
    if a.jsut_dir:
        src = os.path.join(a.jsut_dir, "basic5000", "wav", "BASIC5000_0079.wav")
        gt = os.path.join(JSUT_DIR, "001_GT_BASIC5000_0079.wav")
        shutil.copyfile(src, gt)
        ok &= check(gt)
        out = os.path.join(NORM, "ja_natural_norm.wav")
        from_file(gt, out)
        ok &= check(out)
    if a.libritts_wav:
        out = os.path.join(NORM, "en_natural_norm.wav")
        from_file(a.libritts_wav, out)
        ok &= check(out)
    if a.fleurs_wav:
        out = os.path.join(NORM, "zh_natural_norm.wav")
        from_file(a.fleurs_wav, out)
        ok &= check(out)
    out = os.path.join(NORM, "zh_natural_bznsyp_norm.wav")
    if a.bznsyp_wav:
        from_file(a.bznsyp_wav, out)
    else:
        import pyarrow.parquet as pq
        import torchaudio
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("yvonne66/BZNSYP_new", "data/train-00000-of-00008.parquet", repo_type="dataset",
                            revision="1a97fc04e129ead20f42fdd14d7c61a40e7c8d0b")
        row = pq.read_table(p).slice(BZNSYP_ROW, 1).to_pylist()[0]
        w, sr = torchaudio.load(io.BytesIO(row["audio"]["bytes"]))
        normalize(w, sr, out)
    ok &= check(out)
    print("all prompts match the files used" if ok else "some prompts differ from the files used")


if __name__ == "__main__":
    main()
