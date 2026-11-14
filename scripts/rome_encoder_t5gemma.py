#!/usr/bin/env python3
"""ROME/MEMIT-style Encoder MLP Weight Editing for T5Gemma-TTS.

Permanently edits the encoder MLP weights to make the model generate
correct pronunciations for hard entities without runtime injection.

Theory (ROME):
  W_new = W + delta_W
  delta_W = (v* - W k*) @ (C^{-1} k*)^T / (k*^T C^{-1} k*)

where:
  k* = MLP input (pre_feedforward_layernorm output) at entity token, kanji text
  v* = MLP output at entity token, katakana text ("natural v*" on manifold)
  C  = covariance of MLP inputs over corpus

Target weight: down_proj.weight (2304, 9216)
Target layers: 2, 5, 8 (from causal tracing: MLP divergence peaks at Layer 2)

MEMIT extension: simultaneous editing of multiple entities by solving
the system equations jointly.
"""
import sys
import json
import os

T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)

import torch
import numpy as np
import random

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import (
    normalize_text_with_lang,
    save_audio,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, "/mnt/scripts")
from encoder_injection_t5gemma import (
    get_encoder_output,
    generate_with_modified_encoder,
    find_entity_token_range,
)


# ============================================================
# Target entities (BL correct rate <= 40%)
# ============================================================
TARGET_ENTITIES = [
    {
        "entity": "相川冬也",
        "reading": "アイカワトウヤ",
        "text": "相川冬也部長にご報告ください。",
        "text_katakana": "アイカワトウヤ部長にご報告ください。",
    },
    {
        "entity": "アイクル",
        "reading": "ナカノリョウコ",
        "text": "アイクルは横須賀市の施設です。",
        "text_katakana": "ナカノリョウコは横須賀市の施設です。",
    },
    {
        "entity": "会沢咲",
        "reading": "アイザワサキ",
        "text": "会沢咲は声優として活動しています。",
        "text_katakana": "アイザワサキは声優として活動しています。",
    },
    {
        "entity": "逢川奈々",
        "reading": "アイカワナナ",
        "text": "逢川奈々は最近注目の若手女優です。",
        "text_katakana": "アイカワナナは最近注目の若手女優です。",
    },
]

# Full eval set (same as multiseed eval)
EVAL_ENTITIES = [
    {"entity": "会沢咲", "reading": "アイザワサキ",
     "text": "会沢咲は声優として活動しています。",
     "text_katakana": "アイザワサキは声優として活動しています。"},
    {"entity": "藍川千尋", "reading": "アイカワチヒロ",
     "text": "藍川千尋は歌手としてデビューしました。",
     "text_katakana": "アイカワチヒロは歌手としてデビューしました。"},
    {"entity": "相川冬也", "reading": "アイカワトウヤ",
     "text": "相川冬也部長にご報告ください。",
     "text_katakana": "アイカワトウヤ部長にご報告ください。"},
    {"entity": "逢川奈々", "reading": "アイカワナナ",
     "text": "逢川奈々は最近注目の若手女優です。",
     "text_katakana": "アイカワナナは最近注目の若手女優です。"},
    {"entity": "相内沙英", "reading": "アイウチサエ",
     "text": "相内沙英アナウンサーが担当します。",
     "text_katakana": "アイウチサエアナウンサーが担当します。"},
    {"entity": "青木誠", "reading": "アオキマコト",
     "text": "青木誠が声優として出演しています。",
     "text_katakana": "アオキマコトが声優として出演しています。"},
    {"entity": "会沢紗弥", "reading": "アイザワサヤ",
     "text": "会沢紗弥は人気声優です。",
     "text_katakana": "アイザワサヤは人気声優です。"},
    {"entity": "愛ともえ", "reading": "アイトモエ",
     "text": "愛ともえは女優として活動中です。",
     "text_katakana": "アイトモエは女優として活動中です。"},
    {"entity": "あおい輝彦", "reading": "アオイテルヒコ",
     "text": "あおい輝彦は長年芸能界で活躍しています。",
     "text_katakana": "アオイテルヒコは長年芸能界で活躍しています。"},
    {"entity": "アイクル", "reading": "ナカノリョウコ",
     "text": "アイクルは横須賀市の施設です。",
     "text_katakana": "ナカノリョウコは横須賀市の施設です。"},
]

SEEDS = [42, 123, 456, 789, 2024]


# ============================================================
# Utility functions
# ============================================================

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_whisper(device):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    model_name = "openai/whisper-large-v3"
    print(f"[Info] Loading Whisper ({model_name})...")
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    whisper_model.eval()
    return whisper_model, processor


def transcribe_with_whisper(audio_path, whisper_model, processor, device):
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    input_features = processor(
        wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt",
    ).input_features.to(device, dtype=torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            input_features, language="ja", task="transcribe",
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def text_to_kana(text):
    raise ImportError('pyopenjtalk is not used in this release (authors\' rule)')
    import re
    text = re.sub(r'[\s\u3000]', '', text)
    text = re.sub(r'[\u3002\u3001\uff01\uff1f!?,.]', '', text)
    return pyopenjtalk.g2p(text, kana=True)


def text_to_phonemes(text):
    raise ImportError('pyopenjtalk is not used in this release (authors\' rule)')
    import re
    text = re.sub(r'[\s\u3000]', '', text)
    text = re.sub(r'[\u3002\u3001\uff01\uff1f!?,.]', '', text)
    return pyopenjtalk.g2p(text, kana=False).replace(' ', '')


def katakana_to_hiragana(text):
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            result.append(chr(cp - 0x60))
        else:
            result.append(ch)
    return ''.join(result)


def compute_cer(ref, hyp):
    ref = list(ref)
    hyp = list(hyp)
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(ref)][len(hyp)] / max(len(ref), 1)


def evaluate_pronunciation(text, reading_katakana, transcript):
    ref_kana = text_to_kana(text)
    trans_kana = text_to_kana(transcript)
    expected_kana = text_to_kana(reading_katakana)
    ref_phonemes = text_to_phonemes(text)
    trans_phonemes = text_to_phonemes(transcript)

    trans_hira = katakana_to_hiragana(trans_kana)
    expected_hira = katakana_to_hiragana(expected_kana)
    pronunciation_match = expected_hira in trans_hira

    phoneme_cer = compute_cer(ref_phonemes, trans_phonemes)

    return {
        'ref_kana': ref_kana,
        'trans_kana': trans_kana,
        'expected_kana': expected_kana,
        'pronunciation_match': pronunciation_match,
        'phoneme_cer': round(phoneme_cer, 4),
    }


# ============================================================
# Hook-based MLP input/output capture
# ============================================================

def capture_mlp_io(encoder, input_ids, attention_mask, layer_idx):
    """Capture MLP input (pre_feedforward_layernorm output) and MLP output at layer_idx."""
    captured = {}
    hooks = []

    def make_pre_ffn_hook(storage):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                storage['mlp_input'] = output[0].detach().clone()
            else:
                storage['mlp_input'] = output.detach().clone()
        return hook_fn

    def make_mlp_hook(storage):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                storage['mlp_output'] = output[0].detach().clone()
            else:
                storage['mlp_output'] = output.detach().clone()
        return hook_fn

    layer = encoder.layers[layer_idx]

    # Hook on pre_feedforward_layernorm (= MLP input before MLP)
    if hasattr(layer, 'pre_feedforward_layernorm'):
        hooks.append(
            layer.pre_feedforward_layernorm.register_forward_hook(make_pre_ffn_hook(captured))
        )
    else:
        # Fallback: capture input to MLP directly
        def make_mlp_input_hook(storage):
            def hook_fn(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    storage['mlp_input'] = input[0].detach().clone()
                else:
                    storage['mlp_input'] = input.detach().clone()
            return hook_fn
        hooks.append(
            layer.mlp.register_forward_pre_hook(make_mlp_input_hook(captured))
        )

    # Hook on MLP output
    hooks.append(
        layer.mlp.register_forward_hook(make_mlp_hook(captured))
    )

    with torch.no_grad():
        encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    for h in hooks:
        h.remove()

    return captured.get('mlp_input'), captured.get('mlp_output')


# ============================================================
# Covariance matrix computation
# ============================================================

def compute_mlp_input_covariance(encoder, text_tokenizer, corpus_texts, layer_idx, device,
                                 normalize_func=None):
    """Compute E[k k^T] over corpus texts at the given encoder layer.

    Returns C: (hidden_size, hidden_size) float32 tensor.
    """
    print(f"  [Cov] Computing covariance over {len(corpus_texts)} corpus samples, Layer {layer_idx}...")
    sum_outer = None
    count = 0

    for i, text in enumerate(corpus_texts):
        if normalize_func is not None:
            text_norm, _ = normalize_func(text, "ja")
        else:
            text_norm = text

        tokens = text_tokenizer.encode(text_norm, add_special_tokens=False)
        if len(tokens) == 0:
            continue

        input_ids = torch.LongTensor([tokens]).to(device)
        attention_mask = torch.ones_like(input_ids)

        mlp_input, _ = capture_mlp_io(encoder, input_ids, attention_mask, layer_idx)
        if mlp_input is None:
            continue

        # mlp_input: (1, T, hidden_size) -> collect all token positions
        h = mlp_input[0].float()  # (T, hidden_size)
        T = h.shape[0]

        # Accumulate outer products: sum over all token positions
        outer = h.T @ h  # (hidden_size, hidden_size)
        if sum_outer is None:
            sum_outer = outer
        else:
            sum_outer += outer
        count += T

        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(corpus_texts)} texts, {count} tokens total")

    if sum_outer is None or count == 0:
        raise RuntimeError("No valid corpus texts processed for covariance computation")

    C = sum_outer / count
    print(f"  [Cov] Done. C shape={C.shape}, count={count}")
    return C


# ============================================================
# ROME weight update for one entity
# ============================================================

def solve_Cinv_k(C, k, eps=1e-4):
    """Solve C x = k using Cholesky decomposition on CPU.

    C: (d, d) float32 tensor (may be on CPU)
    k: (d,) float32 tensor
    Returns: x (d,) float32 on same device as k
    """
    # Move to CPU for Cholesky to avoid potential GPU OOM on 9216x9216
    C_cpu = C.float().cpu()
    k_cpu = k.float().cpu()
    d = C_cpu.shape[0]
    C_reg = C_cpu + eps * torch.eye(d, dtype=torch.float32)

    try:
        # Cholesky: O(d^3/3), memory efficient
        L = torch.linalg.cholesky(C_reg)
        # Solve L y = k, then L^T x = y
        x = torch.cholesky_solve(k_cpu.unsqueeze(1), L).squeeze(1)
    except Exception:
        # Fallback: direct solve
        x = torch.linalg.solve(C_reg, k_cpu)

    return x.to(k.device)


def compute_rome_delta_W(k_star, v_star, v_current, C, eps=1e-4):
    """Compute ROME weight update delta_W for down_proj.

    k_star:    (9216,) down_proj input at entity last token (kanji text)
    v_star:    (2304,) target MLP output (katakana text)
    v_current: (2304,) current MLP output for k_star
    C:         (9216, 9216) covariance E[h h^T] on CPU

    Returns:
        delta_W: (2304, 9216)
    """
    k = k_star.float()
    delta_v = (v_star - v_current).float()  # (2304,)

    # C^{-1} k  (solved on CPU)
    C_inv_k = solve_Cinv_k(C, k, eps=eps)  # (9216,)

    # scalar = k^T C^{-1} k
    scalar = (k.cpu() * C_inv_k.cpu()).sum()
    if scalar.abs() < 1e-12:
        print(f"    [WARN] scalar near zero: {scalar.item():.2e}, skipping")
        return None

    # delta_W = (1/scalar) * delta_v outer C^{-1} k
    # (2304, 1) @ (1, 9216) = (2304, 9216)
    # Compute on CPU to avoid GPU memory pressure
    delta_v_cpu = delta_v.cpu()
    C_inv_k_cpu = C_inv_k.cpu()
    scalar_cpu = scalar.cpu() if hasattr(scalar, 'cpu') else scalar
    delta_W = (delta_v_cpu.unsqueeze(1) @ C_inv_k_cpu.unsqueeze(0)) / scalar_cpu  # on CPU

    return delta_W


def compute_memit_delta_W(k_stars, v_stars, v_currents, C, eps=1e-4):
    """MEMIT-style simultaneous editing of multiple entities.

    Simplified MEMIT formula:
      delta_W = (V* - V_current) (C^{-1} K)^T (K^T C^{-1} K)^{-1}

    where K = [k1 | k2 | ... | kN] stacked as columns.
    """
    N = len(k_stars)
    assert N == len(v_stars) == len(v_currents), "Mismatch in entity counts"

    K = torch.stack(k_stars, dim=1).float().cpu()    # (d_in, N)
    V_delta = torch.stack(
        [v_stars[i] - v_currents[i] for i in range(N)], dim=1
    ).float().cpu()  # (d_out, N)

    C_cpu = C.float().cpu()
    d = C_cpu.shape[0]
    C_reg = C_cpu + eps * torch.eye(d, dtype=torch.float32)

    try:
        L = torch.linalg.cholesky(C_reg)
        # C^{-1} K: solve L L^T X = K  =>  L Y = K, L^T X = Y
        C_inv_K = torch.cholesky_solve(K, L)  # (d_in, N)
    except Exception:
        C_inv_K = torch.linalg.solve(C_reg, K)

    # (K^T C^{-1} K)^{-1}: N x N matrix
    KtCinvK = K.T @ C_inv_K  # (N, N)
    KtCinvK_inv = torch.linalg.inv(KtCinvK + 1e-6 * torch.eye(N))  # (N, N)

    # delta_W = V_delta @ KtCinvK_inv @ C_inv_K^T
    # (d_out, N) @ (N, N) @ (N, d_in) = (d_out, d_in)
    delta_W = V_delta @ KtCinvK_inv @ C_inv_K.T  # (d_out, d_in) on CPU

    return delta_W


# ============================================================
# Capture down_proj input (MLP intermediate)
# ============================================================

def capture_down_proj_input(encoder, input_ids, attention_mask, layer_idx):
    """Capture the input to down_proj at the given layer.

    The MLP is: down_proj(Dropout(GELU(gate_proj(x)) * up_proj(x)))
    We need to capture the tensor just before down_proj.
    """
    captured = {}
    hooks = []
    layer = encoder.layers[layer_idx]

    def make_down_proj_pre_hook(storage):
        def hook_fn(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                storage['down_proj_input'] = input[0].detach().clone()
            elif not isinstance(input, tuple):
                storage['down_proj_input'] = input.detach().clone()
        return hook_fn

    def make_mlp_output_hook(storage):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                storage['mlp_output'] = output[0].detach().clone()
            else:
                storage['mlp_output'] = output.detach().clone()
        return hook_fn

    # Capture down_proj input via pre-hook
    if hasattr(layer.mlp, 'down_proj'):
        hooks.append(
            layer.mlp.down_proj.register_forward_pre_hook(make_down_proj_pre_hook(captured))
        )
    else:
        print(f"  [WARN] Layer {layer_idx} MLP has no down_proj attribute")

    # Also capture full MLP output
    hooks.append(
        layer.mlp.register_forward_hook(make_mlp_output_hook(captured))
    )

    with torch.no_grad():
        encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    for h in hooks:
        h.remove()

    return captured.get('down_proj_input'), captured.get('mlp_output')


def compute_down_proj_input_covariance(encoder, text_tokenizer, corpus_texts, layer_idx, device,
                                       normalize_func=None, max_samples=200):
    """Compute E[h h^T] where h is the down_proj input over corpus texts.

    For intermediate_size=9216, 9216x9216 float32 = ~320MB per matrix.
    We use online accumulation but limit to max_samples to avoid OOM.
    Falls back to CPU accumulation if needed.
    """
    print(f"  [Cov] Computing down_proj input covariance, Layer {layer_idx}...")
    texts_to_use = corpus_texts[:max_samples]
    sum_outer = None
    count = 0

    for i, text in enumerate(texts_to_use):
        if normalize_func is not None:
            text_norm, _ = normalize_func(text, "ja")
        else:
            text_norm = text

        tokens = text_tokenizer.encode(text_norm, add_special_tokens=False)
        if len(tokens) == 0:
            continue

        input_ids = torch.LongTensor([tokens]).to(device)
        attention_mask = torch.ones_like(input_ids)

        down_proj_input, _ = capture_down_proj_input(encoder, input_ids, attention_mask, layer_idx)
        if down_proj_input is None:
            continue

        # down_proj_input: (1, T, intermediate_size) -> (T, intermediate_size)
        h = down_proj_input[0].float().cpu()  # Accumulate on CPU to save GPU memory
        T = h.shape[0]

        outer = h.T @ h  # (intermediate_size, intermediate_size)
        if sum_outer is None:
            sum_outer = outer
        else:
            sum_outer += outer
        count += T

        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(texts_to_use)} texts, {count} tokens total")

    if sum_outer is None or count == 0:
        raise RuntimeError("No valid corpus texts for covariance computation")

    C = sum_outer / count  # kept on CPU (too large for GPU)
    print(f"  [Cov] Done. C shape={C.shape}, count={count}")
    return C  # float32, CPU


# ============================================================
# Main ROME editing function
# ============================================================

def apply_rome_edit(model, text_tokenizer, layer_idx, target_entities, corpus_texts, device,
                    edit_mode='memit', eps=1e-4):
    """Apply ROME/MEMIT weight edit to encoder layer layer_idx.

    Args:
        model: T5Gemma model
        text_tokenizer: tokenizer
        layer_idx: which encoder layer to edit
        target_entities: list of dicts with entity/reading/text/text_katakana
        corpus_texts: list of strings for covariance computation
        device: cuda/cpu
        edit_mode: 'rome' (sequential, one by one) or 'memit' (simultaneous)
        eps: regularization strength

    Returns:
        delta_W: applied weight delta (2304, 9216), or None on failure
    """
    encoder = model.encoder_module
    print(f"\n[ROME] Editing Layer {layer_idx}, mode={edit_mode}, {len(target_entities)} entities")

    # Step 1: Compute covariance of down_proj inputs over corpus (kept on CPU)
    C = compute_down_proj_input_covariance(
        encoder, text_tokenizer, corpus_texts, layer_idx, device,
        normalize_func=normalize_text_with_lang,
    )
    # Keep C on CPU (9216x9216 = ~320MB) - solve functions handle CPU internally

    add_bos = 1 if getattr(model.args, "add_bos_to_text", 0) else 0

    # Step 2: For each entity, capture k* (kanji) and v* (katakana)
    k_stars = []
    v_stars = []
    v_currents = []
    entity_infos = []

    for ent in target_entities:
        print(f"\n  Entity: {ent['entity']} ({ent['reading']})")

        # Tokenize kanji text
        kanji_text_norm, _ = normalize_text_with_lang(ent['text'], "ja")
        kata_text_norm, _ = normalize_text_with_lang(ent['text_katakana'], "ja")

        kanji_tokens = text_tokenizer.encode(kanji_text_norm, add_special_tokens=False)
        kata_tokens = text_tokenizer.encode(kata_text_norm, add_special_tokens=False)

        if len(kanji_tokens) == 0 or len(kata_tokens) == 0:
            print(f"    [WARN] Empty tokens, skipping")
            continue

        kanji_ids = torch.LongTensor([kanji_tokens]).to(device)
        kata_ids = torch.LongTensor([kata_tokens]).to(device)
        kanji_mask = torch.ones_like(kanji_ids)
        kata_mask = torch.ones_like(kata_ids)

        # Find entity token range in kanji text
        kanji_range = find_entity_token_range(
            text_tokenizer, kanji_text_norm, ent['entity'], add_bos=add_bos
        )
        kata_range = find_entity_token_range(
            text_tokenizer, kata_text_norm, ent['reading'], add_bos=add_bos
        )

        if kanji_range is None:
            print(f"    [WARN] Could not find entity '{ent['entity']}' in kanji text, skipping")
            continue
        if kata_range is None:
            print(f"    [WARN] Could not find reading '{ent['reading']}' in katakana text, skipping")
            continue

        kanji_start, kanji_end = kanji_range
        kata_start, kata_end = kata_range
        print(f"    Kanji range: {kanji_range}, Kata range: {kata_range}")

        # Capture down_proj input (k*) and MLP output (v*) for kanji text
        kanji_dp_input, kanji_mlp_out = capture_down_proj_input(
            encoder, kanji_ids, kanji_mask, layer_idx
        )
        # Capture MLP output (v*) for katakana text
        kata_dp_input, kata_mlp_out = capture_down_proj_input(
            encoder, kata_ids, kata_mask, layer_idx
        )

        if kanji_dp_input is None or kanji_mlp_out is None:
            print(f"    [WARN] Failed to capture kanji MLP I/O, skipping")
            continue
        if kata_mlp_out is None:
            print(f"    [WARN] Failed to capture katakana MLP output, skipping")
            continue

        # Extract entity position vectors
        # Use last entity token (most entity-specific)
        kanji_k_star = kanji_dp_input[0, kanji_end - 1].float()   # down_proj input, last entity token
        kanji_v_current = kanji_mlp_out[0, kanji_end - 1].float()  # current MLP output

        # For v*, use katakana text MLP output at last entity token
        kata_v_star = kata_mlp_out[0, kata_end - 1].float()

        print(f"    k_star shape: {kanji_k_star.shape}")
        print(f"    v* shape: {kata_v_star.shape}")
        print(f"    |delta_v| = {(kata_v_star - kanji_v_current).norm().item():.4f}")

        # Cosine sim between current and target
        cos_sim = torch.nn.functional.cosine_similarity(
            kanji_v_current.unsqueeze(0), kata_v_star.unsqueeze(0)
        ).item()
        print(f"    cos(v_current, v*) = {cos_sim:.6f}")

        k_stars.append(kanji_k_star)
        v_stars.append(kata_v_star)
        v_currents.append(kanji_v_current)
        entity_infos.append(ent)

    if not k_stars:
        print(f"  [ERROR] No valid entities found for editing at Layer {layer_idx}")
        return None

    print(f"\n  Applying {edit_mode} weight update for {len(k_stars)} entities...")

    # Move C and all vectors to the same device (down_proj weight device)
    down_proj = encoder.layers[layer_idx].mlp.down_proj
    original_dtype = down_proj.weight.dtype
    target_device = down_proj.weight.device

    C_dev = C.to(dtype=torch.float32, device=target_device)
    k_stars_dev = [k.to(device=target_device) for k in k_stars]
    v_stars_dev = [v.to(device=target_device) for v in v_stars]
    v_currents_dev = [v.to(device=target_device) for v in v_currents]

    # Step 3: Compute delta_W
    if edit_mode == 'memit' and len(k_stars_dev) > 1:
        delta_W = compute_memit_delta_W(k_stars_dev, v_stars_dev, v_currents_dev, C_dev, eps=eps)
    else:
        # ROME: sequential (apply one at a time)
        delta_W = None
        for i, (k, v_star, v_curr) in enumerate(zip(k_stars_dev, v_stars_dev, v_currents_dev)):
            dW = compute_rome_delta_W(k, v_star, v_curr, C_dev, eps=eps)
            if dW is not None:
                delta_W = dW if delta_W is None else delta_W + dW
                print(f"    Entity {i}: |delta_W| = {dW.norm().item():.6f}")

    if delta_W is None:
        print(f"  [ERROR] delta_W computation failed")
        return None

    print(f"  delta_W shape: {delta_W.shape}, |delta_W| = {delta_W.norm().item():.6f}")

    # Clip extreme values for numerical stability
    max_abs = delta_W.abs().max().item()
    if max_abs > 1e6:
        clip_val = 1e4
        print(f"  [WARN] delta_W max_abs={max_abs:.2e} is very large, clipping to +/-{clip_val:.0e}")
        delta_W = delta_W.clamp(-clip_val, clip_val)
        print(f"  After clipping: |delta_W| = {delta_W.norm().item():.6f}")

    # Step 4: Apply weight update to down_proj.weight
    with torch.no_grad():
        down_proj.weight.add_(delta_W.to(dtype=original_dtype, device=target_device))

    print(f"  Applied delta_W to down_proj.weight at Layer {layer_idx}")
    return delta_W


# ============================================================
# Generate audio with current model state
# ============================================================

def generate_audio_for_entity(model, text_tokenizer, audio_tokenizer,
                               ent, device, codec_audio_sr, codec_sr, output_path):
    """Generate audio for the entity text and save to output_path."""
    kanji_enc = get_encoder_output(model, text_tokenizer, ent['text'], device)

    gen_audio = generate_with_modified_encoder(
        model, text_tokenizer, audio_tokenizer,
        kanji_enc['last_hidden_state'],
        kanji_enc['attention_mask'],
        kanji_enc['position_ids'],
        ent['text'], device, codec_audio_sr, codec_sr,
    )

    if gen_audio is None:
        return None

    gen_wav = gen_audio[0].cpu()
    save_audio(output_path, gen_wav, codec_audio_sr)
    return output_path


# ============================================================
# Evaluation
# ============================================================

def run_evaluation(model, text_tokenizer, audio_tokenizer,
                   whisper_model, whisper_processor,
                   eval_entities, output_dir, label,
                   device, codec_audio_sr, codec_sr,
                   seeds=None):
    """Run multi-seed evaluation on all eval entities.

    Returns list of result dicts.
    """
    if seeds is None:
        seeds = SEEDS

    all_results = []
    total = len(seeds) * len(eval_entities)
    done = 0

    for seed in seeds:
        for idx, ent in enumerate(eval_entities):
            seed_everything(seed)
            done += 1
            print(f"  [{done}/{total}] {ent['entity']} seed={seed} [{label}]")

            audio_path = os.path.join(output_dir, f"ent{idx:02d}_s{seed}_{label}.wav")

            try:
                path = generate_audio_for_entity(
                    model, text_tokenizer, audio_tokenizer,
                    ent, device, codec_audio_sr, codec_sr, audio_path,
                )
                if path is None:
                    print(f"    ERROR: No audio generated")
                    all_results.append({
                        'entity': ent['entity'], 'seed': seed, 'label': label,
                        'status': 'error', 'error': 'No audio',
                    })
                    continue

                transcript = transcribe_with_whisper(path, whisper_model, whisper_processor, device)
                pron = evaluate_pronunciation(ent['text'], ent['reading'], transcript)
                mark = "OK" if pron['pronunciation_match'] else "NG"
                print(f"    [{mark}] '{transcript}' phCER={pron['phoneme_cer']:.4f}")

                all_results.append({
                    'entity': ent['entity'], 'seed': seed, 'label': label,
                    'transcript': transcript, **pron, 'status': 'success',
                })
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_results.append({
                    'entity': ent['entity'], 'seed': seed, 'label': label,
                    'status': 'error', 'error': str(e),
                })

    return all_results


def print_eval_summary(results, label, eval_entities):
    """Print evaluation summary table."""
    success = [r for r in results if r['status'] == 'success']
    correct = sum(1 for r in success if r['pronunciation_match'])
    total = len(success)
    phcers = [r['phoneme_cer'] for r in success]

    print(f"\n  [{label}] Overall: {correct}/{total} correct ({100*correct/max(total,1):.1f}%)")
    print(f"           Avg phCER: {np.mean(phcers):.4f} +/- {np.std(phcers):.4f}")

    print(f"\n  Per-entity:")
    print(f"  {'Entity':<12} | {'Correct':>7} | {'phCER':>8}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*8}")

    for ent in eval_entities:
        e = ent['entity']
        ent_results = [r for r in success if r['entity'] == e]
        ok = sum(1 for r in ent_results if r['pronunciation_match'])
        cer = np.mean([r['phoneme_cer'] for r in ent_results]) if ent_results else 0.0
        print(f"  {e:<12} | {ok:>3}/{len(ent_results)} | {cer:>8.4f}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 5, 8],
                        help='Encoder layers to edit (default: 2 5 8)')
    parser.add_argument('--edit_mode', choices=['rome', 'memit'], default='memit',
                        help='Editing mode: rome (sequential) or memit (simultaneous)')
    parser.add_argument('--eps', type=float, default=1.0,
                        help='Regularization strength for covariance matrix (default: 1.0 for numerical stability)')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/eval_results/t5gemma_rome',
                        help='Output directory for results')
    parser.add_argument('--corpus', choices=['val', 'train', 'both'], default='both',
                        help='Corpus for covariance computation')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Seeds for evaluation (default: 42 123 456)')
    parser.add_argument('--no_eval', action='store_true',
                        help='Skip evaluation, only apply weight edits')
    args = parser.parse_args()

    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Config] layers={args.layers}, mode={args.edit_mode}, eps={args.eps}")
    print(f"[Config] output_dir={args.output_dir}")
    print(f"[Config] seeds={args.seeds}")

    # Load model
    model_dir = "Aratako/T5Gemma-TTS-2b-2b"
    print(f"\n[Info] Loading T5Gemma model from {model_dir}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    cfg = model.config

    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", None),
    )
    codec_audio_sr = audio_tokenizer.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    # Load Whisper
    whisper_model, whisper_processor = load_whisper(device)

    # Load corpus texts
    data_dir = "/mnt/data"
    corpus_texts = []
    if args.corpus in ('val', 'both'):
        val_path = os.path.join(data_dir, "val_lora.jsonl")
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                if "text" in obj:
                    corpus_texts.append(obj["text"])
    if args.corpus in ('train', 'both'):
        train_path = os.path.join(data_dir, "train_lora.jsonl")
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                if "text" in obj:
                    corpus_texts.append(obj["text"])

    print(f"[Info] Corpus size: {len(corpus_texts)} texts")

    # =========================================================
    # Baseline evaluation (before any edits)
    # =========================================================
    if not args.no_eval:
        print(f"\n{'='*70}")
        print("BASELINE EVALUATION (before ROME edits)")
        print(f"{'='*70}")
        bl_results = run_evaluation(
            model, text_tokenizer, audio_tokenizer,
            whisper_model, whisper_processor,
            EVAL_ENTITIES, args.output_dir, 'baseline',
            device, codec_audio_sr, codec_sr,
            seeds=args.seeds,
        )
        print_eval_summary(bl_results, 'baseline', EVAL_ENTITIES)
    else:
        bl_results = []

    # =========================================================
    # ROME/MEMIT editing: try each layer configuration
    # =========================================================
    all_edit_results = {}

    for layer_config in [args.layers]:
        config_label = f"layers_{'_'.join(str(l) for l in layer_config)}_{args.edit_mode}"
        print(f"\n{'='*70}")
        print(f"EDITING CONFIG: {config_label}")
        print(f"{'='*70}")

        # Apply weight edits in-place (no deepcopy to save memory)
        delta_Ws = {}
        applied_deltas = {}  # store for revert

        for layer_idx in layer_config:
            delta_W = apply_rome_edit(
                model,
                text_tokenizer,
                layer_idx,
                TARGET_ENTITIES,
                corpus_texts,
                device,
                edit_mode=args.edit_mode,
                eps=args.eps,
            )
            if delta_W is not None:
                delta_Ws[layer_idx] = {
                    'norm': float(delta_W.norm().item()),
                    'max_abs': float(delta_W.abs().max().item()),
                    'shape': list(delta_W.shape),
                }
                applied_deltas[layer_idx] = delta_W

        print(f"\n  Weight edit summary:")
        for l, info in delta_Ws.items():
            print(f"    Layer {l}: |delta_W|={info['norm']:.6f}, max_abs={info['max_abs']:.6f}")

        # =========================================================
        # Post-edit evaluation
        # =========================================================
        if not args.no_eval:
            print(f"\n  Running post-edit evaluation...")
            edit_out_dir = os.path.join(args.output_dir, config_label)
            os.makedirs(edit_out_dir, exist_ok=True)

            edited_results = run_evaluation(
                model, text_tokenizer, audio_tokenizer,
                whisper_model, whisper_processor,
                EVAL_ENTITIES, edit_out_dir, config_label,
                device, codec_audio_sr, codec_sr,
                seeds=args.seeds,
            )
            print_eval_summary(edited_results, config_label, EVAL_ENTITIES)

            all_edit_results[config_label] = {
                'layers': layer_config,
                'edit_mode': args.edit_mode,
                'eps': args.eps,
                'delta_W_stats': delta_Ws,
                'results': edited_results,
            }

        # Revert weight edits for next config
        encoder = model.encoder_module
        for layer_idx, delta_W in applied_deltas.items():
            down_proj = encoder.layers[layer_idx].mlp.down_proj
            with torch.no_grad():
                down_proj.weight.data -= delta_W.to(down_proj.weight.dtype).to(down_proj.weight.device)
        print(f"  [Info] Reverted weight edits for layers {list(applied_deltas.keys())}")
        torch.cuda.empty_cache()

    # =========================================================
    # Summary comparison
    # =========================================================
    if not args.no_eval and bl_results:
        print(f"\n{'='*70}")
        print("SUMMARY: Baseline vs ROME Edits")
        print(f"{'='*70}")

        bl_success = [r for r in bl_results if r['status'] == 'success']
        bl_correct = sum(1 for r in bl_success if r['pronunciation_match'])
        bl_total = len(bl_success)
        bl_phcer = np.mean([r['phoneme_cer'] for r in bl_success]) if bl_success else 0

        print(f"  Baseline: {bl_correct}/{bl_total} ({100*bl_correct/max(bl_total,1):.1f}%)"
              f"  phCER={bl_phcer:.4f}")

        for config_label, data in all_edit_results.items():
            edit_success = [r for r in data['results'] if r['status'] == 'success']
            edit_correct = sum(1 for r in edit_success if r['pronunciation_match'])
            edit_total = len(edit_success)
            edit_phcer = np.mean([r['phoneme_cer'] for r in edit_success]) if edit_success else 0
            diff = (edit_correct / max(edit_total, 1)) - (bl_correct / max(bl_total, 1))

            print(f"  {config_label}: {edit_correct}/{edit_total}"
                  f" ({100*edit_correct/max(edit_total,1):.1f}%)"
                  f"  phCER={edit_phcer:.4f}"
                  f"  diff={diff*100:+.1f}%")

    # Save all results
    save_data = {
        'config': {
            'layers': args.layers,
            'edit_mode': args.edit_mode,
            'eps': args.eps,
            'corpus': args.corpus,
            'seeds': args.seeds,
        },
        'target_entities': [e['entity'] for e in TARGET_ENTITIES],
        'eval_entities': [e['entity'] for e in EVAL_ENTITIES],
        'baseline_results': bl_results,
        'edit_results': all_edit_results,
    }

    save_path = os.path.join(args.output_dir, "rome_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n[Info] Results saved to {save_path}")


if __name__ == "__main__":
    main()
