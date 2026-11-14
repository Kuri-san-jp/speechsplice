"""Score UTMOS22 on an existing directory of paired BL/INJ wav files.

Looks for `s{XXX}_bl.wav` and matching `s{XXX}_inj.wav` (or _bl_mask /
_inj_mask) and writes a summary JSON with means / paired statistics.
"""
import argparse, json, os
import numpy as np
import torch
import torchaudio
import torchaudio.functional as Faud
from scipy import stats

_proj = os.environ.get("REPO_ROOT", os.getcwd())
_hf_home = os.environ.get("HF_HOME", os.path.join(_proj, ".cache/huggingface"))
os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--lang", required=True)
    p.add_argument("--bl_suffix", default="_bl.wav",
                   help="Suffix for baseline wavs (default _bl.wav).")
    p.add_argument("--inj_suffix", default="_inj.wav",
                   help="Suffix for injected wavs (default _inj.wav).")
    p.add_argument("--max_pairs", type=int, default=None)
    args = p.parse_args()

    def _resolve(p):
        if p.startswith(os.environ.get("REPO_ROOT", os.getcwd())) and os.path.isdir("/mnt/scripts"):
            return p.replace(os.environ.get("REPO_ROOT", os.getcwd()), "/mnt")
        return p
    wav_dir = _resolve(args.wav_dir)
    out_json = _resolve(args.output_json)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading UTMOS22 on {device}")
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                         trust_repo=True).to(device).eval()

    def utmos(wav_path):
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = Faud.resample(wav.float(), sr, 16000); sr = 16000
        rms = (wav ** 2).mean().sqrt().item()
        if rms < 0.001:
            return None, rms
        with torch.no_grad():
            return float(mos(wav.to(device), sr)), rms

    files = sorted([f for f in os.listdir(wav_dir) if f.endswith(args.bl_suffix)])
    if args.max_pairs:
        files = files[: args.max_pairs]
    print(f"[Info] {len(files)} BL wavs in {wav_dir}")

    rows = []
    for i, bl_name in enumerate(files):
        prefix = bl_name[: -len(args.bl_suffix)]
        bl_path = os.path.join(wav_dir, bl_name)
        inj_path = os.path.join(wav_dir, f"{prefix}{args.inj_suffix}")
        if not os.path.exists(inj_path):
            continue
        try:
            bl_u, bl_rms = utmos(bl_path)
            inj_u, inj_rms = utmos(inj_path)
        except Exception as e:
            print(f"  [{prefix}] score fail: {e}")
            continue
        rows.append({
            "id": prefix, "bl_utmos": bl_u, "inj_utmos": inj_u,
            "bl_rms": bl_rms, "inj_rms": inj_rms,
            "bl_wav": bl_path, "inj_wav": inj_path,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}]")

    # Aggregate
    bl_us = np.array([r["bl_utmos"] for r in rows if r["bl_utmos"] is not None])
    inj_us = np.array([r["inj_utmos"] for r in rows if r["inj_utmos"] is not None])
    valid = [r for r in rows if r["bl_utmos"] is not None and r["inj_utmos"] is not None]
    bl_p = np.array([r["bl_utmos"] for r in valid])
    inj_p = np.array([r["inj_utmos"] for r in valid])
    n_paired = len(valid)

    if n_paired >= 2:
        t_stat, t_pval = stats.ttest_rel(bl_p, inj_p)
        w_stat, w_pval = stats.wilcoxon(bl_p, inj_p)
    else:
        t_stat = t_pval = w_stat = w_pval = None

    summary = {
        "lang": args.lang,
        "wav_dir": wav_dir,
        "n": len(rows),
        "n_paired": n_paired,
        "bl_utmos_mean": round(float(bl_us.mean()), 4) if len(bl_us) else None,
        "bl_utmos_std": round(float(bl_us.std()), 4) if len(bl_us) else None,
        "inj_utmos_mean": round(float(inj_us.mean()), 4) if len(inj_us) else None,
        "inj_utmos_std": round(float(inj_us.std()), 4) if len(inj_us) else None,
        "mean_diff_inj_minus_bl": round(float((inj_p - bl_p).mean()), 4) if n_paired else None,
        "paired_t_pval": round(float(t_pval), 6) if t_pval is not None else None,
        "wilcoxon_pval": round(float(w_pval), 6) if w_pval is not None else None,
    }
    print(f"\n=== {args.lang} N={summary['n']} ===")
    for k, v in summary.items():
        if k not in ("wav_dir",): print(f"  {k}: {v}")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
