"""Common acoustic metrics module for Table 3 (iii) decoder sensitivity.

Provides DTW path-mean cos sim and MCD computations over forced-aligned
entity audio slices. Imported by per-architecture eval scripts.
"""
import numpy as np
import torch
import torchaudio


def safe_norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return n + 1e-8


def time_mean_cos(A, B):
    if A is None or B is None or A.shape[0] == 0 or B.shape[0] == 0: return None
    a = A.mean(axis=0); b = B.mean(axis=0)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return None
    return float(np.dot(a, b) / (na * nb))


def dtw_path_mean_cos(A, B):
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
    return float(np.mean(path_sims))


def dtw_path_euclidean(A, B, band_frac=0.25):
    if A is None or B is None or A.shape[0] == 0 or B.shape[0] == 0: return None
    Ta, Tb = A.shape[0], B.shape[0]
    band = max(2, int(band_frac * max(Ta, Tb)))
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1).astype(np.float32)
    INF = np.float32(1e9)
    dp = np.full((Ta, Tb), INF, dtype=np.float32)
    dp[0, 0] = cost[0, 0]
    for i in range(Ta):
        j_lo = max(0, int(i * Tb / Ta) - band)
        j_hi = min(Tb, int(i * Tb / Ta) + band + 1)
        for j in range(j_lo, j_hi):
            if i == 0 and j == 0: continue
            best = INF
            if i > 0: best = min(best, dp[i-1, j])
            if j > 0: best = min(best, dp[i, j-1])
            if i > 0 and j > 0: best = min(best, dp[i-1, j-1])
            dp[i, j] = best + cost[i, j]
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
    """Cepstral Mean and Variance Normalization (CMVN):
    subtract per-utterance mean and divide by per-utterance std."""
    if mfcc is None or mfcc.shape[0] == 0: return mfcc
    mean = mfcc.mean(axis=0, keepdims=True)
    std = mfcc.std(axis=0, keepdims=True) + 1e-8
    return (mfcc - mean) / std


def mcd_from_mfcc(mfcc_a, mfcc_b):
    """MCD (dB) over DTW Euclidean of CMVN-normalized MFCC frames (excluding C0).
    CMVN brings the per-utterance distribution to zero-mean, unit-variance per
    coefficient, so the MCD value is in normalized-cepstral units."""
    a = cmvn(mfcc_a); b = cmvn(mfcc_b)
    d = dtw_path_euclidean(a, b)
    if d is None: return None
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * d)


class AcousticFeatureExtractor:
    """Compute mel-spec / wav2vec2 hidden / phoneme posterior / MFCC frame features
    on a single GPU, with weight-sharing between CTC and base wav2vec2."""

    def __init__(self, device, feat_model="facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        self.device = device
        self.feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(feat_model)
        self.ctc_model = Wav2Vec2ForCTC.from_pretrained(feat_model).to(device).eval()
        self.w2v_base = self.ctc_model.wav2vec2
        self.target_sr = self.feat_ext.sampling_rate
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr, n_fft=400, hop_length=160, n_mels=80
        ).to(device)
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=self.target_sr, n_mfcc=24,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 80}
        ).to(device)

    def compute(self, audio_np):
        """Return (mel, w2v, pps, mfcc) frame-level numpy arrays of shape (T, D)."""
        if audio_np is None or audio_np.size < int(0.05 * self.target_sr):
            return None, None, None, None
        wav = torch.from_numpy(audio_np.astype(np.float32))
        with torch.no_grad():
            mel = self.mel_extractor(wav.to(self.device))
            mel = torch.log(mel + 1e-6).float().cpu().numpy().T
            mfcc = self.mfcc_extractor(wav.to(self.device))
            mfcc = mfcc.float().cpu().numpy().T
            x = self.feat_ext(wav.cpu().numpy(), sampling_rate=self.target_sr,
                              return_tensors="pt").input_values.to(self.device)
            base_out = self.w2v_base(x, output_hidden_states=False)
            w2v = base_out.last_hidden_state.squeeze(0).float().cpu().numpy()
            logits = self.ctc_model.lm_head(base_out.last_hidden_state)
            pps = torch.softmax(logits, dim=-1).squeeze(0).float().cpu().numpy()
        return mel, w2v, pps, mfcc


def acoustic_pair_metrics(feats_a, feats_b):
    """Compute all metrics between two (mel, w2v, pps, mfcc) tuples.
    Returns dict with keys like dtw_mel, dtw_w2v, dtw_pps, mcd, tmean_mel, tmean_w2v, tmean_pps."""
    mel_a, w2v_a, pps_a, mfcc_a = feats_a
    mel_b, w2v_b, pps_b, mfcc_b = feats_b
    out = {}
    # DTW path mean cos for each feature
    out["dtw_mel"] = dtw_path_mean_cos(mel_a, mel_b)
    out["dtw_w2v"] = dtw_path_mean_cos(w2v_a, w2v_b)
    out["dtw_pps"] = dtw_path_mean_cos(pps_a, pps_b)
    # Time-mean cos for each feature (legacy)
    out["tmean_mel"] = time_mean_cos(mel_a, mel_b)
    out["tmean_w2v"] = time_mean_cos(w2v_a, w2v_b)
    out["tmean_pps"] = time_mean_cos(pps_a, pps_b)
    # MCD via MFCC (drop C0)
    if mfcc_a is not None and mfcc_b is not None and mfcc_a.shape[0] >= 3 and mfcc_b.shape[0] >= 3:
        out["mcd"] = mcd_from_mfcc(mfcc_a[:, 1:], mfcc_b[:, 1:])
    else:
        out["mcd"] = None
    return out
