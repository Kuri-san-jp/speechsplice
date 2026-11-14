"""Cross-culture phoneme PER + inj_better_pct via wav2vec2-CTC.

For each pair {pair_idx, word_a, word_b, text_a, text_b}:
  1. Read BL, INJ, TGT wavs from {wav_dir}/s{pidx:04d}_{bl,inj,tgt}.wav
  2. Use wav2vec2-xlsr-53-espeak-cv-ft to get full IPA transcript per audio
  3. Use phonemizer (+pykakasi for JA) to get expected entity phoneme sequences
     for word_a (BL slot) and word_b (TGT slot).
  4. Use torchaudio forced_align to locate the entity phoneme range in each
     audio's CTC emission grid → IPA slice corresponding to entity region.
  5. PER(slice_X_IPA vs TGT_entity_IPA) computed for X in {BL, INJ}.
  6. inj_better = PER(INJ vs TGT) < PER(BL vs TGT) per pair.

Outputs per-cell summary including inj_better_pct.
"""

# 2026-09-27: CPU スレッド上限（1 本あたり 13）。torch などを読む前に設定する。
import os as _os_cap
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os_cap.environ[_v] = "13"
try:
    import torch as _torch_cap
    _torch_cap.set_num_threads(13)
except Exception:
    pass
import argparse, json, math, os, sys
import numpy as np

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

os.environ.setdefault("HF_HOME", "/mnt/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/mnt/.cache/torch")

# torchao monkey-patch
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


def char_edit_distance(a, b):
    if len(a) < len(b): a, b = b, a
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(cur[j-1]+1, prev[j]+1, prev[j-1] + (0 if x == y else 1))
        prev = cur
    return prev[-1]


def per_score(hyp_ipa: str, ref_ipa: str) -> float:
    """Char-level edit distance / len(ref). Returns float in [0, 1+] (can exceed 1)."""
    if not ref_ipa: return 1.0
    return char_edit_distance(hyp_ipa, ref_ipa) / max(1, len(ref_ipa))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--lang", required=True, choices=["en","ja","zh"])
    p.add_argument("--output_json", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--max_dur_sec", type=float, default=15.0)
    args = p.parse_args()

    import torch
    import torchaudio
    import torchaudio.functional as Fa
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    print(f"[Info] Loading {model_name} on {device}", flush=True)
    feat = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    asr  = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    vocab = json.load(open(hf_hub_download(model_name, "vocab.json")))
    sym2id = vocab; id2sym = {v: k for k, v in vocab.items()}; blank_id = vocab.get("<pad>", 0)
    AR_SR = 16000

    def load_audio(path):
        a, sr = torchaudio.load(path)
        if a.shape[0] > 1: a = a.mean(0, keepdim=True)
        if sr != AR_SR: a = Fa.resample(a, sr, AR_SR)
        a = a.squeeze(0)[:int(args.max_dur_sec * AR_SR)]
        return a

    def emissions(audio):
        with torch.no_grad():
            iv = feat(audio.numpy(), sampling_rate=AR_SR, return_tensors="pt").input_values.to(device)
            log_probs = torch.log_softmax(asr(iv).logits, dim=-1)
        return log_probs[0]  # (T, V)

    def decode_full_ipa(emi):
        """CTC greedy decode of emission to IPA string."""
        pred = emi.argmax(dim=-1).cpu().tolist()
        toks = []; prev = None
        for i in pred:
            if i == blank_id: prev = i; continue
            if i == prev: continue
            toks.append(id2sym.get(i, ""))
            prev = i
        return "".join(toks)

    def entity_ipa_via_forced_align(emi, audio, entity_phonemes):
        """Forced-align entity phonemes to emission; return IPA slice."""
        tokens = [sym2id[p] for p in entity_phonemes if p in sym2id]
        if not tokens: return None
        try:
            tokens_t = torch.LongTensor([tokens]).to(device)
            alignment, _ = Fa.forced_align(emi.unsqueeze(0), tokens_t, blank=blank_id)
            ali = alignment[0].cpu().tolist()
            non_blank = [i for i,t in enumerate(ali) if t != blank_id]
            if not non_blank: return None
            f_start, f_end = non_blank[0], non_blank[-1]+1
            # Decode CTC over the entity frame range
            sub_pred = emi[f_start:f_end].argmax(dim=-1).cpu().tolist()
            toks = []; prev = None
            for i in sub_pred:
                if i == blank_id: prev = i; continue
                if i == prev: continue
                toks.append(id2sym.get(i, ""))
                prev = i
            return "".join(toks)
        except Exception:
            return None

    pairs = load_pairs(args.lang, args.n)
    rows = []
    for pair in pairs:
        pidx = pair.get("pair_idx")
        word_a = pair["word_a"]; word_b = pair["word_b"]
        bl_p  = os.path.join(args.wav_dir, f"s{pidx:04d}_bl.wav")
        inj_p = os.path.join(args.wav_dir, f"s{pidx:04d}_inj.wav")
        tgt_p = os.path.join(args.wav_dir, f"s{pidx:04d}_tgt.wav")
        if not (os.path.exists(bl_p) and os.path.exists(inj_p) and os.path.exists(tgt_p)):
            rows.append({"pair_idx": pidx, "available": False}); continue
        try:
            bl_a  = load_audio(bl_p)
            inj_a = load_audio(inj_p)
            tgt_a = load_audio(tgt_p)
            # If silent (model lang collapse), mark and skip
            if bl_a.abs().max() < 1e-4 or tgt_a.abs().max() < 1e-4:
                rows.append({"pair_idx": pidx, "skip": "silent_audio"}); continue
            ph_a = g2p_phonemes(word_a, args.lang)
            ph_b = g2p_phonemes(word_b, args.lang)
            emi_bl  = emissions(bl_a)
            emi_inj = emissions(inj_a)
            emi_tgt = emissions(tgt_a)
            # Entity-IPA slices via forced alignment
            bl_ent_ipa  = entity_ipa_via_forced_align(emi_bl,  bl_a,  ph_a)
            inj_ent_ipa = entity_ipa_via_forced_align(emi_inj, inj_a, ph_a)
            tgt_ent_ipa = entity_ipa_via_forced_align(emi_tgt, tgt_a, ph_b)
            if any(x is None for x in (bl_ent_ipa, inj_ent_ipa, tgt_ent_ipa)):
                rows.append({"pair_idx": pidx, "skip": "align_failed"}); continue
            # PER computed against the TGT entity IPA (ground truth = word_b pronunciation
            # actually produced by TGT synthesis).
            bl_per  = per_score(bl_ent_ipa,  tgt_ent_ipa)
            inj_per = per_score(inj_ent_ipa, tgt_ent_ipa)
            inj_better = inj_per < bl_per
            rows.append({
                "pair_idx": pidx, "available": True,
                "bl_ent_ipa": bl_ent_ipa,
                "inj_ent_ipa": inj_ent_ipa,
                "tgt_ent_ipa": tgt_ent_ipa,
                "bl_per_vs_tgt":  round(bl_per, 4),
                "inj_per_vs_tgt": round(inj_per, 4),
                "delta_per": round(bl_per - inj_per, 4),
                "inj_better": bool(inj_better),
            })
            if (pidx + 1) % 25 == 0:
                n_ib = sum(1 for r in rows if r.get("inj_better"))
                n_valid = sum(1 for r in rows if r.get("inj_per_vs_tgt") is not None)
                print(f"  [{pidx+1}/{args.n}] valid={n_valid} inj_better={n_ib}", flush=True)
        except Exception as e:
            rows.append({"pair_idx": pidx, "err": str(e)[:200]})

    valid = [r for r in rows if r.get("inj_per_vs_tgt") is not None]
    nv = len(valid)
    if nv > 0:
        bl_per_arr  = np.array([r["bl_per_vs_tgt"]  for r in valid])
        inj_per_arr = np.array([r["inj_per_vs_tgt"] for r in valid])
        delta = bl_per_arr - inj_per_arr
        ib_n = sum(1 for r in valid if r["inj_better"])
        # Wilson 95% CI for inj_better_pct
        from math import sqrt
        z = 1.96; p = ib_n / nv
        ci_half = z * sqrt(p*(1-p)/nv + z*z/(4*nv*nv)) / (1 + z*z/nv)
        # Paired t-test
        from scipy import stats
        try:
            t_, t_p = stats.ttest_rel(inj_per_arr, bl_per_arr)
            w_, w_p = stats.wilcoxon(inj_per_arr, bl_per_arr, zero_method="zsplit")
        except Exception:
            t_p = w_p = float('nan')
        cd = float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else None
    else:
        bl_per_arr = inj_per_arr = delta = np.array([])
        ib_n = 0; ci_half = 0; t_p = w_p = float('nan'); cd = None

    summary = {
        "wav_dir": args.wav_dir, "lang": args.lang,
        "n_target": args.n, "n_valid": nv,
        "bl_per_vs_tgt_mean":  round(float(bl_per_arr.mean()), 4)  if nv else None,
        "bl_per_vs_tgt_std":   round(float(bl_per_arr.std(ddof=1)), 4)  if nv > 1 else None,
        "inj_per_vs_tgt_mean": round(float(inj_per_arr.mean()), 4) if nv else None,
        "inj_per_vs_tgt_std":  round(float(inj_per_arr.std(ddof=1)), 4) if nv > 1 else None,
        "delta_per_mean": round(float(delta.mean()), 4) if nv else None,
        "delta_per_std":  round(float(delta.std(ddof=1)), 4) if nv > 1 else None,
        "inj_better_n": ib_n,
        "inj_better_pct": round(100 * ib_n / nv, 2) if nv else 0,
        "inj_better_ci95_half_pp": round(100 * ci_half, 2) if nv else 0,
        "paired_t_pval": float(t_p),
        "wilcoxon_pval": float(w_p),
        "cohens_d_paired": round(cd, 4) if cd is not None else None,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== Phoneme PER inj_better ({args.lang}) ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
