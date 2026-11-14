"""Score wav2vec2-xlsr-53-espeak-cv-ft IPA ASR on an existing directory
of paired `s{idx:04d}_bl.wav` / `s{idx:04d}_inj.wav` files.

Mirrors the metric computed by `gen_en_phoneme_n300.py` (ipa_change_rate_pct,
mean_ipa_edit_distance) so the output JSON is compatible with the existing
`eval_results/phoneme_asr/{lang}_n300.json` schema and Table 1 in the paper.
"""
import argparse, json, os, sys
import numpy as np

for _p in ("/tmp/.cache/huggingface/token", os.path.expanduser("~/.cache/huggingface/token")):
    if os.path.exists(_p):
        try:
            with open(_p) as _f: os.environ.setdefault("HF_TOKEN", _f.read().strip())
            break
        except Exception: pass

# In-container path mapping
_proj = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
_hf_home = "/mnt/.cache/huggingface" if os.path.isdir("/mnt/scripts") else \
            os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")

import torch
import torchaudio


def edit_distance(a, b):
    a, b = list(a), list(b)
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if x == y else 1))
        prev = cur
    return prev[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True,
                   help="Directory containing s{idx:04d}_bl.wav and "
                        "s{idx:04d}_inj.wav files.")
    p.add_argument("--lang", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--phoneme_model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    p.add_argument("--max_pairs", type=int, default=None)
    args = p.parse_args()

    # Resolve paths to container form when running inside slurm
    def _resolve(p):
        if p.startswith(os.environ.get("REPO_ROOT", "/mnt")) and os.path.isdir("/mnt/scripts"):
            return p.replace(os.environ.get("REPO_ROOT", "/mnt"), "/mnt")
        return p
    wav_dir = _resolve(args.wav_dir)
    out_json = _resolve(args.output_json)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    print(f"[Info] wav_dir = {wav_dir}")
    print(f"[Info] output = {out_json}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading phoneme ASR: {args.phoneme_model} on {device}")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    from huggingface_hub import hf_hub_download
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(args.phoneme_model)
    asr_model = Wav2Vec2ForCTC.from_pretrained(args.phoneme_model).to(device).eval()
    vocab_path = hf_hub_download(args.phoneme_model, "vocab.json")
    with open(vocab_path) as _f:
        vocab = json.load(_f)
    id2sym = {v: k for k, v in vocab.items()}
    pad_id = vocab.get("<pad>", 0)
    print(f"[Info] Phoneme ASR vocab={len(vocab)}")

    def transcribe(wav_path):
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != feat_ext.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, feat_ext.sampling_rate)
        audio = wav.squeeze().numpy()
        # Skip near-silent wavs
        if np.sqrt((audio ** 2).mean()) < 0.001:
            return ""
        with torch.no_grad():
            x = feat_ext(audio, sampling_rate=feat_ext.sampling_rate,
                         return_tensors="pt").input_values.to(device)
            pred = torch.argmax(asr_model(x).logits, dim=-1)[0].cpu().tolist()
        tokens, prev = [], None
        for i in pred:
            if i == pad_id:
                prev = i; continue
            if i == prev: continue
            tokens.append(id2sym.get(i, ""))
            prev = i
        return " ".join(tokens).replace("|", " ").strip()

    # Find all paired wavs in directory
    files = sorted([f for f in os.listdir(wav_dir) if f.endswith("_bl.wav")])
    if args.max_pairs:
        files = files[: args.max_pairs]
    print(f"[Info] {len(files)} BL wavs found")

    results = []
    n_changed = 0
    ed_list = []
    for i, bl_name in enumerate(files):
        idx = bl_name.split("_")[0]  # e.g. "s0000"
        bl_path = os.path.join(wav_dir, bl_name)
        inj_path = os.path.join(wav_dir, f"{idx}_inj.wav")
        if not os.path.exists(inj_path):
            continue
        try:
            bl_ipa = transcribe(bl_path)
            inj_ipa = transcribe(inj_path)
        except Exception as e:
            print(f"  [{idx}] transcribe fail: {e}")
            continue
        bl_toks = bl_ipa.split()
        inj_toks = inj_ipa.split()
        ed = edit_distance(bl_toks, inj_toks)
        changed = bool(bl_ipa != inj_ipa)
        if changed: n_changed += 1
        ed_list.append(ed)
        results.append({
            "idx": i, "key": f"{wav_dir}/{idx}",
            "bl_wav": bl_path, "inj_wav": inj_path,
            "bl_ipa": bl_ipa, "inj_ipa": inj_ipa,
            "bl_len": len(bl_toks), "inj_len": len(inj_toks),
            "ipa_changed": changed, "ipa_edit_distance": ed,
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(files)}] changed_so_far={n_changed} ed_avg={np.mean(ed_list):.2f}")

    n = len(results)
    rate = 100.0 * n_changed / n if n else 0.0
    summary = {
        "n": n,
        "model": args.phoneme_model,
        "lang": args.lang,
        "ipa_change_rate_pct": round(rate, 2),
        "mean_ipa_edit_distance": round(float(np.mean(ed_list)), 2) if ed_list else 0.0,
    }
    out = {"summary": summary, "per_pair": results}
    with open(out_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n=== {args.lang} N={n} ===")
    print(f"  IPA change rate: {rate:.2f}%")
    print(f"  Mean IPA edit distance: {summary['mean_ipa_edit_distance']}")
    print(f"  Output: {out_json}")


if __name__ == "__main__":
    main()
