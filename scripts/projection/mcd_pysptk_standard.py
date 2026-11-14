"""Standard MCD (dB) via pysptk mel-cepstrum.

Replaces the librosa MFCC-based proxy in cases A/B/C with proper mel-cepstral
coefficients extracted via pysptk.mcep. Output MCD values are in canonical
dB units suitable for paper reporting.

Modes:
  --mode whole       : whole-utterance MCD via DTW (case A re-do)
  --mode char_pos    : char-position fixed-window slice MCD (case B re-do)
  --mode phoneme     : phoneme-aligned slice via wav2vec2-CTC (case C re-do)

Standard MCD formula (per dimension d=1..order):
  MCD_db = (10 / ln10) * sqrt(2 * sum((c_x - c_y)^2))
Averaged over aligned frame pairs (DTW path).
"""
import argparse, json, math, os, sys
import numpy as np

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/projection"))

os.environ.setdefault("HF_HOME", "/mnt/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/mnt/.cache/torch")

# torchao monkey-patch for transformers (needed for case-C)
import torch.utils._pytree as _pt
if not hasattr(_pt, "register_constant"):
    _pt.register_constant = lambda *a, **kw: None


def load_pairs(lang, n=300):
    if lang == "en":
        from en_cross_culture_names import build_300_pairs
    elif lang == "ja":
        from ja_cross_culture_names import build_300_pairs
    elif lang == "zh":
        from zh_cross_culture_names import build_300_pairs
    else: raise ValueError(lang)
    return build_300_pairs()[:n]


def load_wav(path, target_sr=22050):
    import soundfile as sf, scipy.signal
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != target_sr:
        a = scipy.signal.resample(a, int(len(a) * target_sr / sr)).astype(np.float32)
        sr = target_sr
    return a, sr


def mel_cepstrum(audio, sr=22050, order=24, frame_len=1024, hop=256, alpha=None):
    """Extract mel-cepstral coefficients via pysptk. Returns (T, order+1)."""
    import pysptk
    if alpha is None:
        # Standard alpha values: 0.42 for 22kHz/24kHz, 0.41 for 16kHz, 0.46 for 44k
        alpha = {16000: 0.41, 22050: 0.42, 24000: 0.42, 44100: 0.46, 48000: 0.48}.get(sr, 0.42)
    # Frame the signal
    n = len(audio)
    if n < frame_len:
        audio = np.pad(audio, (0, frame_len - n))
        n = frame_len
    nframes = 1 + (n - frame_len) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(nframes, frame_len),
        strides=(audio.strides[0] * hop, audio.strides[0]),
        writeable=False,
    ).copy()
    # Apply window
    window = np.hanning(frame_len).astype(np.float32)
    frames *= window
    # mcep accepts time-domain frames
    # etype=1: bias added to periodogram before log (proper flooring)
    mcep = np.stack([pysptk.mcep(f.astype(np.float64), order=order, alpha=alpha,
                                  etype=1, eps=1e-6)
                     for f in frames])
    return mcep  # (T, order+1), c[0] is energy


def dtw_path(a, b):
    """Pure-NumPy DTW on cost matrix; returns list of (i, j) pairs."""
    from scipy.spatial.distance import cdist
    cost = cdist(a, b, metric="euclidean")
    Ta, Tb = cost.shape
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    i, j = Ta, Tb; path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [(D[i - 1, j - 1], i - 1, j - 1),
                   (D[i - 1, j],     i - 1, j),
                   (D[i, j - 1],     i,     j - 1)]
        _, i, j = min(choices)
    return list(reversed(path))


def mcd_db_dtw(a_mcep, b_mcep):
    """Canonical MCD in dB. Skip c0 (energy)."""
    if a_mcep is None or b_mcep is None or a_mcep.size == 0 or b_mcep.size == 0:
        return None
    a = a_mcep[:, 1:]
    b = b_mcep[:, 1:]
    path = dtw_path(a, b)
    if not path: return None
    # MCD per frame pair: (10/ln10) * sqrt(2 * sum((c_x - c_y)^2))
    coef = 10.0 / math.log(10) * math.sqrt(2.0)
    diffs = [coef * np.linalg.norm(a[i] - b[j]) for i, j in path]
    return float(np.mean(diffs))


def fixed_window_slice(audio, sr, text_full, entity, pad_sec=0.10):
    pos = text_full.find(entity)
    if pos < 0:
        return None
    n_chars = len(text_full)
    if n_chars == 0: return None
    dur = len(audio) / sr
    s_t = max(0.0, dur * pos / n_chars - pad_sec)
    e_t = min(dur, dur * (pos + len(entity)) / n_chars + pad_sec)
    s_i = int(s_t * sr); e_i = int(e_t * sr)
    if e_i <= s_i: return None
    return audio[s_i:e_i]


# ---- Phoneme alignment helpers (for --mode phoneme) ----
_PHONEMIZER_LANG = {"en": "en-us", "ja": "ja", "zh": "cmn"}
_KKS = None


def _to_hira(text):
    global _KKS
    if _KKS is None:
        import pykakasi; _KKS = pykakasi.kakasi()
    return "".join(r["hira"] for r in _KKS.convert(text))


def g2p_phonemes(word, lang):
    try:
        from phonemizer import phonemize
        if lang == "ja":
            word_h = _to_hira(word)
            ipa = phonemize(word_h, language="ja", backend="espeak",
                            separator=None, preserve_punctuation=False,
                            with_stress=False).strip()
        else:
            ipa = phonemize(word, language=_PHONEMIZER_LANG[lang], backend="espeak",
                            separator=None, preserve_punctuation=False,
                            with_stress=False).strip()
        return [p for p in ipa.replace(" ", "") if p]
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    p.add_argument("--output_json", required=True)
    p.add_argument("--mode", required=True, choices=["whole", "char_pos", "phoneme"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--order", type=int, default=24, help="mel-cepstrum order")
    p.add_argument("--max_dur_sec", type=float, default=15.0)
    p.add_argument("--pad_sec", type=float, default=0.10)
    args = p.parse_args()

    pairs = load_pairs(args.lang, args.n)

    # Phoneme-aligned mode needs wav2vec2-CTC (uses 16k internally)
    aligner = None
    if args.mode == "phoneme":
        import torch
        import torchaudio
        import torchaudio.functional as Fa
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        from huggingface_hub import hf_hub_download
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        feat = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        asr  = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
        vocab = json.load(open(hf_hub_download(model_name, "vocab.json")))
        sym2id = vocab; blank_id = vocab.get("<pad>", 0)
        AR_SR = 16000
        def emissions_at_16k(audio_16k):
            with torch.no_grad():
                iv = feat(audio_16k.numpy(), sampling_rate=AR_SR, return_tensors="pt").input_values.to(device)
                log_probs = torch.log_softmax(asr(iv).logits, dim=-1)
            return log_probs[0]
        def align_entity_at_22k(audio_22k_np, entity_phonemes, target_sr=22050):
            # Resample to 16k for ASR
            import scipy.signal
            audio_16k = scipy.signal.resample(audio_22k_np, int(len(audio_22k_np) * AR_SR / target_sr)).astype(np.float32)
            audio_16k_t = __import__("torch").tensor(audio_16k)
            audio_16k_t = audio_16k_t[:int(args.max_dur_sec * AR_SR)]
            emi = emissions_at_16k(audio_16k_t)
            tokens = [sym2id[p] for p in entity_phonemes if p in sym2id]
            if not tokens: return None
            try:
                tokens_t = __import__("torch").LongTensor([tokens]).to(device)
                alignment, _ = Fa.forced_align(emi.unsqueeze(0), tokens_t, blank=blank_id)
                ali = alignment[0].cpu().tolist()
                non_blank = [i for i, t in enumerate(ali) if t != blank_id]
                if not non_blank: return None
                f_start, f_end = non_blank[0], non_blank[-1] + 1
                # Convert frame to 22k audio sample
                stride_16k = audio_16k_t.shape[0] / emi.shape[0]
                s16 = int(f_start * stride_16k); e16 = int(f_end * stride_16k)
                # Map 16k samples back to 22k indices via ratio
                ratio = target_sr / AR_SR
                s22 = max(0, int(s16 * ratio)); e22 = min(len(audio_22k_np), int(e16 * ratio))
                return audio_22k_np[s22:e22] if e22 > s22 else None
            except Exception:
                return None

    rows = []
    for pidx, pair in enumerate(pairs):
        pair_idx = pair.get("pair_idx", pidx)
        word_a = pair["word_a"]; word_b = pair["word_b"]
        text_a = pair["text_a"]; text_b = pair["text_b"]
        bl_p  = os.path.join(args.wav_dir, f"s{pair_idx:04d}_bl.wav")
        inj_p = os.path.join(args.wav_dir, f"s{pair_idx:04d}_inj.wav")
        tgt_p = os.path.join(args.wav_dir, f"s{pair_idx:04d}_tgt.wav")
        if not (os.path.exists(bl_p) and os.path.exists(inj_p) and os.path.exists(tgt_p)):
            rows.append({"pair_idx": pair_idx, "available": False}); continue
        try:
            bl_a,  _ = load_wav(bl_p,  args.sr)
            inj_a, _ = load_wav(inj_p, args.sr)
            tgt_a, _ = load_wav(tgt_p, args.sr)
            max_s = int(args.max_dur_sec * args.sr)
            bl_a = bl_a[:max_s]; inj_a = inj_a[:max_s]; tgt_a = tgt_a[:max_s]
            # Slice based on mode
            if args.mode == "whole":
                bl_slice, inj_slice, tgt_slice = bl_a, inj_a, tgt_a
            elif args.mode == "char_pos":
                bl_slice  = fixed_window_slice(bl_a,  args.sr, text_a, word_a, args.pad_sec)
                inj_slice = fixed_window_slice(inj_a, args.sr, text_a, word_a, args.pad_sec)
                tgt_slice = fixed_window_slice(tgt_a, args.sr, text_b, word_b, args.pad_sec)
                if any(x is None for x in (bl_slice, inj_slice, tgt_slice)):
                    rows.append({"pair_idx": pair_idx, "skip": "char_pos_miss"}); continue
            else:  # phoneme
                ph_a = g2p_phonemes(word_a, args.lang)
                ph_b = g2p_phonemes(word_b, args.lang)
                bl_slice  = align_entity_at_22k(bl_a,  ph_a, args.sr)
                inj_slice = align_entity_at_22k(inj_a, ph_a, args.sr)
                tgt_slice = align_entity_at_22k(tgt_a, ph_b, args.sr)
                if any(x is None for x in (bl_slice, inj_slice, tgt_slice)):
                    rows.append({"pair_idx": pair_idx, "skip": "align_failed"}); continue
            bl_mc  = mel_cepstrum(bl_slice,  args.sr, order=args.order)
            inj_mc = mel_cepstrum(inj_slice, args.sr, order=args.order)
            tgt_mc = mel_cepstrum(tgt_slice, args.sr, order=args.order)
            mcd_inj_bl  = mcd_db_dtw(inj_mc, bl_mc)
            mcd_inj_tgt = mcd_db_dtw(inj_mc, tgt_mc)
            mcd_bl_tgt  = mcd_db_dtw(bl_mc,  tgt_mc)
            rows.append({
                "pair_idx": pair_idx, "available": True,
                "mcd_inj_vs_bl":  round(mcd_inj_bl, 4)  if mcd_inj_bl  is not None else None,
                "mcd_inj_vs_tgt": round(mcd_inj_tgt, 4) if mcd_inj_tgt is not None else None,
                "mcd_bl_vs_tgt":  round(mcd_bl_tgt, 4)  if mcd_bl_tgt  is not None else None,
            })
            if (pair_idx + 1) % 25 == 0:
                print(f"  [{pair_idx + 1}/{args.n}] processed", flush=True)
        except Exception as e:
            rows.append({"pair_idx": pair_idx, "err": str(e)[:200]})

    valid = [r for r in rows if r.get("mcd_inj_vs_bl") is not None and r.get("mcd_inj_vs_tgt") is not None and r.get("mcd_bl_vs_tgt") is not None]

    def stat(key):
        arr = np.array([r[key] for r in valid])
        if arr.size == 0: return None
        return {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std(ddof=1)), 4) if arr.size > 1 else None,
            "median": round(float(np.median(arr)), 4),
        }

    shift_pval = shift_d = shift_mean = None
    if valid:
        bl_tgt  = np.array([r["mcd_bl_vs_tgt"]  for r in valid])
        inj_tgt = np.array([r["mcd_inj_vs_tgt"] for r in valid])
        shift = bl_tgt - inj_tgt
        try:
            from scipy import stats
            t_, p_ = stats.ttest_1samp(shift, 0)
            shift_pval = float(p_)
            shift_d = float(shift.mean() / shift.std(ddof=1)) if shift.std(ddof=1) > 0 else None
            shift_mean = float(shift.mean())
        except Exception: pass

    summary = {
        "wav_dir": args.wav_dir, "lang": args.lang, "mode": args.mode, "order": args.order,
        "n_target": args.n, "n_valid": len(valid),
        "unit": "MCD (dB) via pysptk mel-cepstrum",
        "mcd_inj_vs_bl":  stat("mcd_inj_vs_bl"),
        "mcd_inj_vs_tgt": stat("mcd_inj_vs_tgt"),
        "mcd_bl_vs_tgt":  stat("mcd_bl_vs_tgt"),
        "shift_bl_tgt_minus_inj_tgt": shift_mean,
        "shift_pval": shift_pval,
        "shift_cohens_d": shift_d,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== Standard MCD dB ({args.mode}, {args.lang}) ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
