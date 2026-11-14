#!/usr/bin/env python3
"""Crossattn-mask 4-condition entity-level similarity (multi-metric, DTW-based).

Implements all proposed improvements in one script:
  (1) DTW path-mean cos sim (frame-by-frame aligned, no mean-pooling smoothing)
  (2) Frame-max cos distance (most-different frame, captures local divergence)
  (3) Mel-spec 2D Frobenius distance (length-normalized, time-axis preserved)
  (4) Vowel-center frame cos sim (entity central 20% frames, heteronym focus)
  (5) Time-mean cos sim (legacy baseline for comparison)

All metrics computed pair-wise within the same text_a synthesis (same context):
  cos(BL, INJ)         - injection's direct effect (Standard cross-attention)
  cos(BL_mask, INJ_mask) - injection effect under mask (should be ~1)
  cos(BL, BL_mask)     - mask's effect alone
  cos(INJ, INJ_mask)   - mask's effect under injection
  cos(BL, TGT)         - baseline entity-difference (legacy)
  cos(INJ, TGT)        - injection toward TGT (legacy)

Three feature backbones: mel-spec / wav2vec2 hidden / wav2vec2 phoneme posterior.

Usage:
  SLURM_TIME=08:00:00 ./scripts/run.sh \\
    python scripts/projection/eval_crossattn_entity_dtw.py --lang ja --n 300
"""
import argparse, json, os, sys, time, random

# --- Anonymous release path setup ---
import os as _os, sys as _sys
_REPO_ROOT_FOR_IMPORTS = _os.environ.get('REPO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_REPO_ROOT_FOR_IMPORTS, 'src'))
_sys.path.insert(0, _os.path.join(_REPO_ROOT_FOR_IMPORTS, 'data'))
# ---

for _p in ("/tmp/.cache/huggingface/token", os.path.expanduser("~/.cache/huggingface/token")):
    if os.path.exists(_p):
        try:
            with open(_p) as _f:
                os.environ.setdefault("HF_TOKEN", _f.read().strip())
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
            break
        except Exception: pass

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", os.getcwd())
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", "")
sys.path.insert(0, T5GEMMA_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch, torchaudio, numpy as np

# torchao compat shim (before any transformers/xcodec2 imports)
import torch.utils._pytree as _pt
if not hasattr(_pt, "register_constant"):
    _pt.register_constant = lambda *a, **kw: None

from alignment import WhisperAligner, find_entity_chunks, slice_entity_audio


SEED = 42


def seed_all(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


def safe_norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return n + 1e-8


def time_mean_cos(A, B):
    """Mean-pool over time then cos sim."""
    if A is None or B is None or A.shape[0] == 0 or B.shape[0] == 0: return None
    a = A.mean(axis=0); b = B.mean(axis=0)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return None
    return float(np.dot(a, b) / (na * nb))


def dtw_path_mean_cos(A, B):
    """Build cosine sim matrix between frames of A (Ta,D) and B (Tb,D),
    find DP path maximizing sum of similarities, return mean sim along path."""
    if A is None or B is None or A.shape[0] == 0 or B.shape[0] == 0: return None
    A = A / safe_norm(A); B = B / safe_norm(B)
    sim = A @ B.T  # (Ta, Tb)
    Ta, Tb = sim.shape
    dp = np.full((Ta, Tb), -np.inf, dtype=np.float32)
    dp[0, 0] = sim[0, 0]
    for i in range(Ta):
        for j in range(Tb):
            if i == 0 and j == 0: continue
            best = -np.inf
            if i > 0: best = max(best, dp[i-1, j])
            if j > 0: best = max(best, dp[i, j-1])
            if i > 0 and j > 0: best = max(best, dp[i-1, j-1])
            dp[i, j] = best + sim[i, j]
    # Backtrack
    i, j = Ta - 1, Tb - 1
    path_sims = [sim[i, j]]
    while i > 0 or j > 0:
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            cands = [(dp[i-1, j], i-1, j), (dp[i, j-1], i, j-1), (dp[i-1, j-1], i-1, j-1)]
            _, i, j = max(cands)
        path_sims.append(sim[i, j])
    return float(np.mean(path_sims))


def dtw_frame_max_dist(A, B):
    """Find max cos distance (1 - sim) along DTW path."""
    if A is None or B is None or A.shape[0] == 0 or B.shape[0] == 0: return None
    A = A / safe_norm(A); B = B / safe_norm(B)
    sim = A @ B.T
    Ta, Tb = sim.shape
    dp = np.full((Ta, Tb), -np.inf, dtype=np.float32)
    dp[0, 0] = sim[0, 0]
    for i in range(Ta):
        for j in range(Tb):
            if i == 0 and j == 0: continue
            best = -np.inf
            if i > 0: best = max(best, dp[i-1, j])
            if j > 0: best = max(best, dp[i, j-1])
            if i > 0 and j > 0: best = max(best, dp[i-1, j-1])
            dp[i, j] = best + sim[i, j]
    i, j = Ta - 1, Tb - 1
    path_sims = [sim[i, j]]
    while i > 0 or j > 0:
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            cands = [(dp[i-1, j], i-1, j), (dp[i, j-1], i, j-1), (dp[i-1, j-1], i-1, j-1)]
            _, i, j = max(cands)
        path_sims.append(sim[i, j])
    return float(1 - np.min(path_sims))  # max distance = 1 - min sim


def vowel_center_cos(A, B, frac=0.4):
    """Cos sim on the central frac of frames (heuristic: vowel core)."""
    if A is None or B is None: return None
    Ta, Tb = A.shape[0], B.shape[0]
    if Ta < 3 or Tb < 3: return None
    s_a, e_a = int(Ta * (0.5 - frac/2)), int(Ta * (0.5 + frac/2))
    s_b, e_b = int(Tb * (0.5 - frac/2)), int(Tb * (0.5 + frac/2))
    a = A[s_a:e_a].mean(axis=0); b = B[s_b:e_b].mean(axis=0)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return None
    return float(np.dot(a, b) / (na * nb))


def frobenius_dist(A, B):
    """Length-normalized 2D Frobenius distance for 2D feature grids
    (mel-spec primarily). A and B may differ in time length: truncate or pad
    to the shorter length."""
    if A is None or B is None: return None
    T = min(A.shape[0], B.shape[0])
    if T == 0: return None
    diff = A[:T] - B[:T]
    return float(np.linalg.norm(diff) / np.sqrt(T))


def dtw_path_euclidean(A, B, band_frac=0.25):
    """Standard distance-minimizing DTW with Sakoe-Chiba band.
    A: (Ta, D), B: (Tb, D). Returns mean Euclidean distance along path."""
    if A is None or B is None or A.shape[0] == 0 or B.shape[0] == 0:
        return None
    Ta, Tb = A.shape[0], B.shape[0]
    band = max(2, int(band_frac * max(Ta, Tb)))
    # Pre-compute pairwise Euclidean distance matrix
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).astype(np.float32)
    INF = np.float32(1e9)
    dp = np.full((Ta, Tb), INF, dtype=np.float32)
    dp[0, 0] = cost[0, 0]
    for i in range(Ta):
        # Sakoe-Chiba band: |i - j*Ta/Tb| < band
        j_lo = max(0, int(i * Tb / Ta) - band)
        j_hi = min(Tb, int(i * Tb / Ta) + band + 1)
        for j in range(j_lo, j_hi):
            if i == 0 and j == 0: continue
            best = INF
            if i > 0: best = min(best, dp[i-1, j])
            if j > 0: best = min(best, dp[i, j-1])
            if i > 0 and j > 0: best = min(best, dp[i-1, j-1])
            dp[i, j] = best + cost[i, j]
    # Backtrack
    i, j = Ta - 1, Tb - 1
    if dp[i, j] >= INF: return None
    path_costs = [cost[i, j]]
    while i > 0 or j > 0:
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            cands = [(dp[i-1, j], i-1, j), (dp[i, j-1], i, j-1), (dp[i-1, j-1], i-1, j-1)]
            _, i, j = min(cands)
        path_costs.append(cost[i, j])
    return float(np.mean(path_costs))


def cmvn(mfcc):
    """Cepstral Mean and Variance Normalization."""
    if mfcc is None or mfcc.shape[0] == 0: return mfcc
    mean = mfcc.mean(axis=0, keepdims=True)
    std = mfcc.std(axis=0, keepdims=True) + 1e-8
    return (mfcc - mean) / std


def mcd_from_mfcc(mfcc_a, mfcc_b):
    """MCD (dB) over DTW Euclidean of CMVN-normalized MFCC frames (excluding C0)."""
    a = cmvn(mfcc_a); b = cmvn(mfcc_b)
    d = dtw_path_euclidean(a, b)
    if d is None: return None
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--lang", default="ja", choices=["en", "ja", "zh"])
    p.add_argument("--ja_source", default="heteronym", choices=["heteronym", "cross_culture"],
                   help="JA data source: heteronym (kanji↔katakana) or cross_culture (word_a/word_b both kanji)")
    p.add_argument("--cross_culture", action="store_true",
                   help="Use cross-culture pairs for EN/ZH (and implies JA cross_culture).")
    p.add_argument("--inject_mode", default="standard",
                   choices=["standard", "length_preserve"],
                   help="standard = original SpeechSplice (may change seq length); "
                        "length_preserve = mean-pool word_b hidden to fit word_a token count "
                        "(preserves encoder seq length for clean mask ablation).")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--feat_model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    p.add_argument("--whisper_model", default="openai/whisper-large-v3")
    p.add_argument("--target_valid", type=int, default=None,
                   help="Stop once this many aligned rows accumulate")
    args = p.parse_args()
    # cross_culture flag implies ja_source=cross_culture too
    if args.cross_culture and args.lang == "ja":
        args.ja_source = "cross_culture"
    use_cc = args.cross_culture or (args.lang == "ja" and args.ja_source == "cross_culture")
    suffix_lp = "_lp" if args.inject_mode == "length_preserve" else ""
    if args.output_dir is None:
        if use_cc:
            args.output_dir = f"eval_results/crossattn_entity_dtw_{args.lang}cc{suffix_lp}_n{args.n}"
        else:
            args.output_dir = f"eval_results/crossattn_entity_dtw_{args.lang}{suffix_lp}_n{args.n}"
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"
    seed_all(SEED)

    if args.lang == "en":
        if use_cc:
            from en_cross_culture_names import build_300_pairs as build_encc
            cc_pairs = build_encc()[: args.n]
            pairs = [{"text_a": p["text_a"], "text_b": p["text_b"],
                      "word_a": p["word_a"], "word_b": p["word_b"],
                      "pair_idx": p["pair_idx"]} for p in cc_pairs]
        else:
            from en_heteronyms_500 import build_500_pairs
            pairs = build_500_pairs()[: args.n]
    elif args.lang == "zh":
        if use_cc:
            from zh_cross_culture_names import build_300_pairs as build_zhcc
            cc_pairs = build_zhcc()[: args.n]
            pairs = [{"text_a": p["text_a"], "text_b": p["text_b"],
                      "word_a": p["word_a"], "word_b": p["word_b"],
                      "pair_idx": p["pair_idx"]} for p in cc_pairs]
        else:
            from zh_polyphones_300 import build_300_pairs
            pairs = build_300_pairs()[: args.n]
    elif args.lang == "ja":
        if args.ja_source == "cross_culture":
            from ja_cross_culture_names import build_300_pairs as build_jacc
            cc_pairs = build_jacc()[: args.n]
            pairs = [{"text_a": p["text_a"], "text_b": p["text_b"],
                      "word_a": p["word_a"], "word_b": p["word_b"],
                      "pair_idx": p["pair_idx"]} for p in cc_pairs]
        else:
            prev = json.load(open("eval_results/crossattn_mask_n300_filtered_v2/results.json"))["results"]
            pairs = []
            for r in prev[: args.n]:
                text_a = r["text"]; kanji = r["kanji"]; kata = r["kata"]
                pairs.append({
                    "text_a": text_a, "text_b": text_a.replace(kanji, kata),
                    "word_a": kanji, "word_b": kata, "pair_idx": r["idx"],
                })
    print(f"[Info] Loaded {len(pairs)} {args.lang} pairs")

    print(f"[Info] Loading wav2vec2-xlsr ...")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(args.feat_model)
    ctc_model = Wav2Vec2ForCTC.from_pretrained(args.feat_model).to(device).eval()
    w2v_base = ctc_model.wav2vec2
    target_sr = feat_ext.sampling_rate
    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=400, hop_length=160, n_mels=80
    ).to(device)
    # MFCC for MCD (TTS-standard cepstral coefficients)
    mfcc_extractor = torchaudio.transforms.MFCC(
        sample_rate=target_sr, n_mfcc=24,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 80}
    ).to(device)

    def compute_frame_features(audio_np):
        """Return per-frame features (T, D) for mel/w2v/pps/mfcc. None on failure."""
        if audio_np is None or audio_np.size < int(0.05 * target_sr):
            return None, None, None, None
        wav = torch.from_numpy(audio_np.astype(np.float32))
        with torch.no_grad():
            mel = mel_extractor(wav.to(device))         # (80, T_mel)
            mel = torch.log(mel + 1e-6).float().cpu().numpy().T   # (T, 80)
            mfcc = mfcc_extractor(wav.to(device))       # (24, T_mfcc)
            mfcc = mfcc.float().cpu().numpy().T          # (T, 24); we drop C0 in MCD calc
            x = feat_ext(wav.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt").input_values.to(device)
            base_out = w2v_base(x, output_hidden_states=False)
            w2v = base_out.last_hidden_state.squeeze(0).float().cpu().numpy()  # (T, 1024)
            logits = ctc_model.lm_head(base_out.last_hidden_state)
            pps = torch.softmax(logits, dim=-1).squeeze(0).float().cpu().numpy()  # (T, vocab)
        return mel, w2v, pps, mfcc

    print(f"[Info] Loading Whisper aligner ...")
    aligner = WhisperAligner(device=device, model_id=args.whisper_model)

    def entity_three_features(audio, sr_in, entity, lang, input_text):
        chunks, audio_np = aligner.transcribe_words(audio, sr_in, language=lang)
        if not chunks or audio_np is None: return None, None, None, None
        s_idx, e_idx = find_entity_chunks(chunks, entity, lang, input_text=input_text)
        if s_idx is None: return None, None, None, None
        ent_audio = slice_entity_audio(audio_np, aligner.target_sr, chunks, s_idx, e_idx)
        if ent_audio is None or ent_audio.size == 0: return None, None, None, None
        return compute_frame_features(ent_audio)

    print("[Info] Loading T5Gemma-TTS ...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # Importing `injection` first registers T5GEMMA_ROOT (=src/_vendor/t5gemma_tts)
    # on sys.path, so `data.tokenizer` then resolves to the vendor module rather
    # than the entity-pair package under `data/`.
    from injection import (
        seed_everything, get_encoder_output, generate_with_modified_encoder, find_entity_token_range,
    )
    from data.tokenizer import AudioTokenizer
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        "Aratako/T5Gemma-TTS-2b-2b", trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto")
    t5_model.eval()
    cfg = t5_model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    t5_tok = AutoTokenizer.from_pretrained(tok_name)
    audio_tok = AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(t5_model.args, "add_bos_to_text", 0)) else 0

    def run_condition(text_a, text_b, word_a, word_b, inject, mask_entity):
        seed_everything(SEED)
        src_enc = get_encoder_output(t5_model, t5_tok, text_a, device, lang=args.lang)
        if inject:
            ref_enc = get_encoder_output(t5_model, t5_tok, text_b, device, lang=args.lang)
            s_r = find_entity_token_range(t5_tok, src_enc["text"], word_a, add_bos=add_bos)
            r_r = find_entity_token_range(t5_tok, ref_enc["text"], word_b, add_bos=add_bos)
            if s_r is None or r_r is None: return None, None
            s_s, s_e = s_r; r_s, r_e = r_r
            modified = src_enc["last_hidden_state"].clone()
            if (s_e - s_s) == (r_e - r_s):
                modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
                entity_start, entity_end = s_s, s_e
            elif args.inject_mode == "length_preserve":
                # Mean-pool word_b hidden over its token range, broadcast to word_a's length.
                # This preserves encoder seq length (T_a unchanged), enabling clean mask ablation
                # without confounding position shifts in downstream context.
                ref_ent = ref_enc["last_hidden_state"][0, r_s:r_e]  # (len_b, D)
                pooled = ref_ent.mean(dim=0, keepdim=True)            # (1, D)
                len_a = s_e - s_s
                expanded = pooled.expand(len_a, -1).clone()           # (len_a, D)
                modified[0, s_s:s_e] = expanded
                entity_start, entity_end = s_s, s_e
            else:
                prefix = modified[0, :s_s]
                ref_ent = ref_enc["last_hidden_state"][0, r_s:r_e]
                suffix = modified[0, s_e:]
                modified = torch.cat([prefix, ref_ent, suffix], dim=0).unsqueeze(0)
                entity_start = s_s
                entity_end = s_s + (r_e - r_s)
            final_len = modified.shape[1]
        else:
            modified = src_enc["last_hidden_state"].clone()
            final_len = modified.shape[1]
            sr_range = find_entity_token_range(t5_tok, src_enc["text"], word_a, add_bos=add_bos)
            if sr_range is None: return None, None
            entity_start, entity_end = sr_range
        attn = torch.ones((1, final_len), dtype=torch.long, device=device)
        if mask_entity:
            attn[0, entity_start:entity_end] = 0
        pos_ids = None
        if int(getattr(t5_model.args, "use_pm_rope", 1)):
            x_lens = torch.LongTensor([final_len]).to(device)
            pos_ids = t5_model._build_position_ids(x_lens, final_len, device)
        seed_everything(SEED)
        out = generate_with_modified_encoder(
            t5_model, t5_tok, audio_tok, modified, attn, pos_ids,
            text_a, device, codec_audio_sr, codec_sr,
        )
        if out is None: return None, None
        return out[0], codec_audio_sr

    def synth_simple(text):
        seed_everything(SEED)
        enc = get_encoder_output(t5_model, t5_tok, text, device, lang=args.lang)
        h = enc["last_hidden_state"]
        attn = torch.ones((1, h.shape[1]), dtype=torch.long, device=device)
        pos_ids = None
        if int(getattr(t5_model.args, "use_pm_rope", 1)):
            x_lens = torch.LongTensor([h.shape[1]]).to(device)
            pos_ids = t5_model._build_position_ids(x_lens, h.shape[1], device)
        out = generate_with_modified_encoder(
            t5_model, t5_tok, audio_tok, h, attn, pos_ids,
            text, device, codec_audio_sr, codec_sr,
        )
        if out is None: return None, None
        return out[0], codec_audio_sr

    METRICS = ["mel", "w2v", "pps"]
    PAIRS_TO_COMPARE = [
        ("bl", "inj"),
        ("bl_mask", "inj_mask"),
        ("bl", "bl_mask"),
        ("inj", "inj_mask"),
        ("bl", "tgt"),
        ("inj", "tgt"),
    ]

    rows = []
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
    done = {r["pair_idx"] for r in rows}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        pair_idx = pair.get("pair_idx", i)
        if pair_idx in done: continue
        try:
            bl_a, bl_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"], False, False)
            inj_a, inj_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"], True, False)
            bl_m_a, bl_m_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"], False, True)
            inj_m_a, inj_m_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"], True, True)
            tgt_a, tgt_sr = synth_simple(pair["text_b"])
            if any(x is None for x in [bl_a, inj_a, bl_m_a, inj_m_a, tgt_a]):
                continue

            feats = {
                "bl":      entity_three_features(bl_a,    bl_sr,    pair["word_a"], args.lang, pair["text_a"]),
                "inj":     entity_three_features(inj_a,   inj_sr,   pair["word_a"], args.lang, pair["text_a"]),
                "bl_mask": entity_three_features(bl_m_a,  bl_m_sr,  pair["word_a"], args.lang, pair["text_a"]),
                "inj_mask":entity_three_features(inj_m_a, inj_m_sr, pair["word_a"], args.lang, pair["text_a"]),
                "tgt":     entity_three_features(tgt_a,   tgt_sr,   pair["word_b"], args.lang, pair["text_b"]),
            }
            row = {"pair_idx": pair_idx, "word_a": pair["word_a"], "word_b": pair["word_b"]}
            row["text_a"] = pair.get("text_a", "")
            row["text_b"] = pair.get("text_b", "")
            for cond in ["bl", "inj", "bl_mask", "inj_mask", "tgt"]:
                row[f"frames_{cond}"] = (int(feats[cond][0].shape[0]) if feats[cond][0] is not None else None)
            ok = all(all(x is not None for x in f) for f in feats.values())
            if not ok:
                row["aligned"] = False
                rows.append(row); done.add(pair_idx); continue
            for m_idx, m in enumerate(METRICS):
                for a, b in PAIRS_TO_COMPARE:
                    A, B = feats[a][m_idx], feats[b][m_idx]
                    row[f"tmean_{a}_{b}_{m}"] = round(time_mean_cos(A, B), 4) if (A is not None and B is not None) else None
                    row[f"dtw_{a}_{b}_{m}"] = round(dtw_path_mean_cos(A, B), 4) if (A is not None and B is not None) else None
                    row[f"fmax_{a}_{b}_{m}"] = round(dtw_frame_max_dist(A, B), 4) if (A is not None and B is not None) else None
                    row[f"vowel_{a}_{b}_{m}"] = round(vowel_center_cos(A, B), 4) if (A is not None and B is not None) else None
                    if m == "mel":
                        row[f"frob_{a}_{b}"] = round(frobenius_dist(A, B), 4) if (A is not None and B is not None) else None
            # MCD (TTS-standard, distance in dB) using MFCC, drop C0 (energy)
            for a, b in PAIRS_TO_COMPARE:
                Ma = feats[a][3]; Mb = feats[b][3]
                if Ma is None or Mb is None or Ma.shape[0] < 3 or Mb.shape[0] < 3:
                    row[f"mcd_{a}_{b}"] = None
                else:
                    row[f"mcd_{a}_{b}"] = round(mcd_from_mfcc(Ma[:, 1:], Mb[:, 1:]), 4)
            row["aligned"] = True
            rows.append(row); done.add(pair_idx)
        except Exception as e:
            print(f"  [{i}] err: {str(e)[:200]}"); continue

        if (i + 1) % 5 == 0:
            n_aligned = sum(1 for r in rows if r.get("aligned"))
            print(f"  [{len(rows)}/{args.n}] aligned={n_aligned} dtw(BL,INJ)_w2v={row.get('dtw_bl_inj_w2v','-')} dtw(BLm,INJm)_w2v={row.get('dtw_bl_mask_inj_mask_w2v','-')}")
        if (i + 1) % 25 == 0:
            with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)
        if args.target_valid is not None:
            n_aligned_now = sum(1 for r in rows if r.get("aligned"))
            if n_aligned_now >= args.target_valid:
                print(f"  [target_valid] reached n_aligned={n_aligned_now}, stopping.")
                break

    with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    aligned = [r for r in rows if r.get("aligned")]
    n_align = len(aligned)
    if n_align == 0:
        print("[Error] No aligned rows"); return

    def avg(key):
        vs = [r[key] for r in aligned if r.get(key) is not None]
        return [round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)] if vs else [None, None]

    summary = {"lang": args.lang, "n_total": len(rows), "n_aligned": n_align}
    for m in METRICS:
        for a, b in PAIRS_TO_COMPARE:
            for prefix in ["tmean", "dtw", "fmax", "vowel"]:
                summary[f"{prefix}_{a}_{b}_{m}"] = avg(f"{prefix}_{a}_{b}_{m}")
            if m == "mel":
                summary[f"frob_{a}_{b}"] = avg(f"frob_{a}_{b}")
    for a, b in PAIRS_TO_COMPARE:
        summary[f"mcd_{a}_{b}"] = avg(f"mcd_{a}_{b}")
    summary["elapsed_min"] = round((time.time() - t0)/60, 2)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== Crossattn entity-DTW {args.lang} N={n_align} ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
