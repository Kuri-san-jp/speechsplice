#!/usr/bin/env python3
"""Qwen3-TTS-1.7B Layer 26 Injection Experiment (N=50 entity pairs).

Based on convergence analysis showing peak cos=0.974 at Layer 26 (96.4% depth),
3/8 pairs reaching cos=1.000, we test whether per-token hidden state replacement
at Layer 26 can redirect pronunciation in a decoder-only TTS.

Design:
  For each test pair (kanji entity vs katakana reading):
  1. Run forward pass with kanji text -> capture Layer 26 hidden states at entity positions
  2. Run forward pass with katakana text -> capture Layer 26 hidden states at entity positions
  3. During audio generation:
     - Hook Layer 26 to replace entity-position hidden states from kanji with katakana
     - Generate audio
  4. Transcribe with Whisper ASR
  5. Compare with baseline (no injection) transcription

Note: Qwen3-TTS is decoder-only, so injection during autoregressive generation
may not work the same way as in encoder-decoder models (T5Gemma). In decoder-only
models, each new token sees only previous tokens, so the injection at prefill
positions affects subsequent generation through KV cache.

Usage:
  GPU_PREFER_FREE=1 SLURM_TIME=02:00:00 slurm_run.sh \
    python scripts/projection/eval_qwen3_injection.py \
    --output_dir eval_results/qwen3_injection_n50 \
    --entity_store resources/entity_store.jsonl \
    --injection_layers 26
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Fix cache directories for Docker
os.environ["TORCH_HOME"] = "/mnt/.cache/torch"
os.environ["HF_HOME"] = "/mnt/.cache/huggingface"
os.environ["XDG_CACHE_HOME"] = "/mnt/.cache"
os.makedirs("/mnt/.cache/torch/hub", exist_ok=True)
os.makedirs("/mnt/.cache/huggingface", exist_ok=True)

# =============================================================================
# Default test pairs (8 entity pairs from original convergence analysis)
# Used as fallback if entity_store is not provided.
# =============================================================================
# Default reference audio for voice clone generation (x_vector_only_mode)
# (release) used only by this file's own main(), which no experiment of the paper runs; the paper's
# Qwen3-TTS runs import functions from here and pass the prompt of configs/voice_prompts.yaml
DEFAULT_REF_AUDIO = os.environ.get("QWEN3_DEFAULT_REF_AUDIO", "sample_0000.wav")

# Default template used when building pairs from entity_store
DEFAULT_TEMPLATE = "{}さんが会議に参加しました。"

def _load_heteronym500_pairs():
    """Load 500 EN heteronym pairs from en_heteronyms_500.py (unified with T5Gemma/SpeechT5)."""
    import sys as _sys
    _sys.path.insert(0, "/mnt/data" if os.path.isdir("/mnt/data") else os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "data"))
    from en_heteronyms_500 import build_500_pairs
    raw = build_500_pairs()
    out = []
    for p in raw:
        out.append({
            "kanji": p["word_a"],
            "katakana": p["word_b"],
            "reading_hira": p["word_a"],
            "template": p["text_a"],
            "template_b": p["text_b"],
        })
    return out

INJECTION_PAIRS_DEFAULT = _load_heteronym500_pairs()[:300]
EVAL_LANGUAGE = "en"


def hiragana_to_katakana(text):
    """Convert hiragana to katakana."""
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x3041 <= cp <= 0x3096:
            result.append(chr(cp + 0x60))
        else:
            result.append(ch)
    return ''.join(result)


def is_injection_suitable(surface):
    """Check if a surface form is suitable for kanji→katakana injection.

    Injection requires:
    - At least one kanji character in surface (so we can form kanji vs katakana pair)
    - Not purely katakana (those already have correct pronunciation)
    - Not contain ASCII/Latin characters or dots that complicate tokenization

    Returns True if suitable for injection test.
    """
    import unicodedata
    has_kanji = False
    has_hiragana = False
    has_ascii = False
    for ch in surface:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            has_kanji = True
        if 0x3041 <= cp <= 0x3096:
            has_hiragana = True
        if cp < 0x80 and ch not in ' ':
            has_ascii = True
        # Exclude middle dot (・) in purely katakana foreign names
        if ch == '・':
            has_ascii = True
    if has_ascii:
        return False
    # If surface has hiragana but no kanji (e.g., "あおい輝彦" has hiragana+kanji, OK)
    # Surface that is ONLY hiragana is skip (no kanji to inject from)
    if not has_kanji and has_hiragana:
        return False
    # Surface that is only katakana skip
    all_kata = all(
        (0x30A1 <= ord(ch) <= 0x30F6 or ch in 'ー')
        for ch in surface
    )
    if all_kata and len(surface) > 0:
        return False
    return True


def load_pairs_from_entity_store(entity_store_path, n_pairs=50):
    """Load entity pairs from entity_store.jsonl for injection testing.

    Args:
        entity_store_path: path to entity_store.jsonl
        n_pairs: maximum number of pairs to return

    Returns:
        list of dicts with keys: kanji, katakana, reading_hira, template
    """
    pairs = []
    seen_surfaces = set()

    with open(entity_store_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            surface = entry["surface"]
            reading_hira = entry.get("reading_hira", "")

            if not reading_hira:
                continue
            if surface in seen_surfaces:
                continue
            if not is_injection_suitable(surface):
                print(f"  [SKIP] {surface}: not suitable for injection")
                continue

            katakana = hiragana_to_katakana(reading_hira)
            pairs.append({
                "kanji": surface,
                "katakana": katakana,
                "reading_hira": reading_hira,
                "template": DEFAULT_TEMPLATE,
                "entity_id": entry.get("entity_id", surface),
            })
            seen_surfaces.add(surface)

            if len(pairs) >= n_pairs:
                break

    return pairs


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def katakana_to_hiragana(text):
    """Convert katakana to hiragana."""
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            result.append(chr(cp - 0x60))
        else:
            result.append(ch)
    return ''.join(result)


def load_whisper(device):
    """Load Whisper large-v3 for ASR evaluation."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    model_name = "openai/whisper-large-v3"
    print(f"[Info] Loading Whisper ({model_name})...")
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    whisper_model.eval()
    return whisper_model, processor


def transcribe_wav(wav_tensor, sr, whisper_model, whisper_processor, device):
    """Transcribe a waveform tensor with Whisper.

    Args:
        wav_tensor: (channels, samples) or (samples,) tensor
        sr: sample rate
        whisper_model: loaded Whisper model
        whisper_processor: loaded Whisper processor
        device: torch device

    Returns:
        transcription string
    """
    import torchaudio

    wav = wav_tensor
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav_mono = wav.squeeze()
    max_samples = 16000 * 30
    if wav_mono.shape[0] > max_samples:
        wav_mono = wav_mono[:max_samples]

    input_features = whisper_processor(
        wav_mono.numpy(), sampling_rate=16000, return_tensors="pt",
    ).input_features.to(device, dtype=torch.float16)

    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            input_features, language="en", task="transcribe",
        )
    return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def find_entity_token_positions(text, entity_surface, processor, text_body_offset=8):
    """Robust entity span finder using cumulative decode for BPE tokenizers."""
    entity_start_char = text.find(entity_surface)
    if entity_start_char == -1:
        return None, None
    entity_end_char = entity_start_char + len(entity_surface)

    full_text = f"<|im_start|>assistant\n{text}"
    full_tokens = processor.tokenizer.encode(full_text, add_special_tokens=False)

    prefix_len = len("<|im_start|>assistant\n")
    abs_start = entity_start_char + prefix_len
    abs_end = entity_end_char + prefix_len

    decoded_cum = [processor.tokenizer.decode(full_tokens[:i + 1]) for i in range(len(full_tokens))]
    start_tok_idx = None
    end_tok_idx = None
    for i, cum in enumerate(decoded_cum):
        if start_tok_idx is None and len(cum) > abs_start:
            start_tok_idx = i
        if len(cum) >= abs_end:
            end_tok_idx = i
            break

    if start_tok_idx is None or end_tok_idx is None:
        return None, None
    if end_tok_idx < start_tok_idx:
        end_tok_idx = start_tok_idx

    def tok_to_seq(tok_idx):
        if tok_idx < 3:
            return tok_idx
        else:
            return text_body_offset + tok_idx - 3

    return tok_to_seq(start_tok_idx), tok_to_seq(end_tok_idx)


def build_text_only_inputs(text, talker, text_projection, processor, config, device):
    """Build dual-channel text-only inputs_embeds for Qwen3-TTS.

    Reused from eval_qwen3_convergence.py.
    """
    talker_model = talker.model
    talker_config = talker.config
    model_dtype = next(talker.parameters()).dtype

    tts_bos = config.tts_bos_token_id
    tts_eos = config.tts_eos_token_id
    tts_pad = config.tts_pad_token_id
    codec_nothink = talker_config.codec_nothink_id
    codec_think_bos = talker_config.codec_think_bos_id
    codec_think_eos = talker_config.codec_think_eos_id
    codec_pad = talker_config.codec_pad_id
    codec_bos = talker_config.codec_bos_id

    formatted = (
        f"<|im_start|>assistant\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    text_ids = processor.tokenizer.encode(formatted, add_special_tokens=False)
    text_ids = text_ids[:-5] if len(text_ids) > 5 else text_ids

    text_len = len(text_ids)
    if text_len < 3:
        text_ids = text_ids + [tts_pad] * (3 - text_len)
        text_len = 3

    seq_len = 8 + text_len - 3 + 1 + 1

    input_text_ids = torch.full((1, seq_len), tts_pad, dtype=torch.long, device=device)
    input_codec_ids = torch.full((1, seq_len), codec_pad, dtype=torch.long, device=device)
    text_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    codec_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)

    input_text_ids[0, :3] = torch.tensor(text_ids[:3], dtype=torch.long)
    input_text_ids[0, 3:7] = tts_pad
    input_text_ids[0, 7] = tts_bos
    if text_len > 3:
        remaining = torch.tensor(text_ids[3:], dtype=torch.long)
        input_text_ids[0, 8:8 + text_len - 3] = remaining
    input_text_ids[0, 8 + text_len - 3] = tts_eos
    input_text_ids[0, 8 + text_len - 2] = tts_pad

    input_codec_ids[0, 3] = codec_nothink
    input_codec_ids[0, 4] = codec_think_bos
    input_codec_ids[0, 5] = codec_think_eos
    input_codec_ids[0, 6] = codec_pad
    input_codec_ids[0, 8 + text_len - 2] = codec_bos

    text_mask[0, :] = True
    codec_mask[0, 3:] = True
    codec_mask[0, 6] = False

    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    with torch.no_grad():
        text_emb = talker.text_projection(
            talker_model.text_embedding(input_text_ids)
        )
        codec_emb = talker_model.codec_embedding(input_codec_ids)

    text_emb = text_emb * text_mask.unsqueeze(-1)
    codec_emb = codec_emb * codec_mask.unsqueeze(-1)
    inputs_embeds = (text_emb + codec_emb).to(model_dtype)

    return inputs_embeds, attention_mask


def get_layer_hidden_states(talker_model, inputs_embeds, attention_mask, target_layer):
    """Get hidden states at a specific layer via forward hook.

    Returns:
        hidden_states: (1, seq_len, hidden_size) tensor at the target layer output
    """
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured["hidden"] = h.detach().clone()

    hook = talker_model.layers[target_layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        talker_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )

    hook.remove()
    return captured["hidden"]


def check_pronunciation(transcription, reading_hira, reading_kata):
    """Check if transcription contains the expected reading.

    Uses multiple strategies:
    1. Direct katakana match in transcription
    2. Hiragana match after conversion
    3. pyopenjtalk g2p match (strict)

    Returns:
        dict with match results
    """
    trans_lower = transcription.strip()

    # Strategy 1: katakana in transcription
    kata_match = reading_kata in trans_lower

    # Strategy 2: hiragana match
    hira_trans = katakana_to_hiragana(trans_lower)
    hira_match = reading_hira in hira_trans

    # Strategy 3: g2p (lenient CER-based)
    try:
        raise ImportError('pyopenjtalk is not used in this release (authors\' rule)')
        trans_clean = re.sub(r'[\s\u3000\u3002\u3001\uff01\uff1f!?,.]', '', trans_lower)
        trans_kana = pyopenjtalk.g2p(trans_clean, kana=True)
        trans_kana_hira = katakana_to_hiragana(trans_kana)
        g2p_match = reading_hira in trans_kana_hira
    except Exception:
        g2p_match = False

    # Lenient: any of the strategies
    lenient_match = kata_match or hira_match or g2p_match

    return {
        "kata_match": kata_match,
        "hira_match": hira_match,
        "g2p_match": g2p_match,
        "lenient_match": lenient_match,
    }


def run_injection_experiment(wrapper, whisper_model, whisper_processor, test_pairs,
                              injection_layer, device, output_dir, seed=42,
                              greedy=False, identity_test=False):
    """Run injection experiment at a specific layer.

    For each pair:
    1. Generate baseline audio (kanji text, no injection) -> transcribe
    2. Generate katakana baseline (katakana text, no injection) -> transcribe
    3. Generate injected audio (kanji text, Layer N injection) -> transcribe
    4. Compare transcriptions

    Args:
        wrapper: Qwen3TTSModel wrapper
        whisper_model: Whisper model
        whisper_processor: Whisper processor
        test_pairs: list of test pair dicts
        injection_layer: layer index for injection
        device: device string
        output_dir: output directory
        seed: random seed
    """
    base_model = wrapper.model
    processor = wrapper.processor
    config = base_model.config
    talker = base_model.talker
    talker_model = talker.model

    num_layers = len(talker_model.layers)
    print(f"\n{'='*70}")
    print(f"INJECTION EXPERIMENT: Layer {injection_layer}/{num_layers}")
    print(f"{'='*70}")

    results = []
    wav_dir = os.path.join(output_dir, f"wav_L{injection_layer}")
    os.makedirs(wav_dir, exist_ok=True)

    for pair_idx, pair in enumerate(test_pairs):
        if "template_b" in pair:
            kanji_text = pair["template"]
            katakana_text = pair["template_b"]
        else:
            kanji_text = pair["template"].format(pair["kanji"])
            katakana_text = pair["template"].format(pair["katakana"])

        print(f"\n--- Pair {pair_idx}: '{pair['kanji']}' ({pair['katakana']}) ---")
        print(f"  Kanji text:    {kanji_text}")
        print(f"  Katakana text: {katakana_text}")

        # Find entity positions in two layouts:
        # 1. Convergence layout (offset=8): for extracting katakana reference hidden states
        # 2. Inference layout (offset=9 for Japanese): for injection during generation
        kanji_start_conv, kanji_end_conv = find_entity_token_positions(
            kanji_text, pair["kanji"], processor, text_body_offset=8
        )
        katakana_start_conv, katakana_end_conv = find_entity_token_positions(
            katakana_text, pair["katakana"], processor, text_body_offset=8
        )
        kanji_start_infer, kanji_end_infer = find_entity_token_positions(
            kanji_text, pair["kanji"], processor, text_body_offset=9
        )

        if kanji_start_conv is None or katakana_start_conv is None:
            print(f"  [SKIP] Could not find entity positions")
            results.append({
                "pair_idx": pair_idx,
                "entity_kanji": pair["kanji"],
                "entity_katakana": pair["katakana"],
                "status": "skip_no_entity_pos",
            })
            continue

        kanji_entity_len = kanji_end_conv - kanji_start_conv + 1
        katakana_entity_len = katakana_end_conv - katakana_start_conv + 1

        print(f"  Kanji entity tokens (conv): [{kanji_start_conv},{kanji_end_conv}] ({kanji_entity_len} tokens)")
        print(f"  Katakana entity tokens (conv): [{katakana_start_conv},{katakana_end_conv}] ({katakana_entity_len} tokens)")
        print(f"  Kanji entity tokens (infer): [{kanji_start_infer},{kanji_end_infer}]")

        # Get katakana hidden states at injection layer (reference vectors)
        # Use convergence layout (offset=8) since build_text_only_inputs uses that layout
        katakana_embeds, katakana_mask = build_text_only_inputs(
            katakana_text, talker, talker.text_projection, processor, config, device
        )
        katakana_hidden = get_layer_hidden_states(
            talker_model, katakana_embeds, katakana_mask, injection_layer
        )
        # Extract entity hidden states using convergence layout positions
        katakana_entity_hidden = katakana_hidden[0, katakana_start_conv:katakana_end_conv + 1].clone()
        print(f"  Katakana entity hidden shape: {katakana_entity_hidden.shape}")

        # Sanity check: identity injection (kanji -> kanji at kanji positions).
        # If identity_test is enabled, replace katakana_entity_hidden with kanji_entity_hidden.
        # Expected: changed=False for all pairs (verifies hook does not introduce distortion).
        if identity_test:
            kanji_embeds, kanji_mask = build_text_only_inputs(
                kanji_text, talker, talker.text_projection, processor, config, device
            )
            kanji_hidden_l = get_layer_hidden_states(
                talker_model, kanji_embeds, kanji_mask, injection_layer
            )
            katakana_entity_hidden = kanji_hidden_l[0, kanji_start_conv:kanji_end_conv + 1].clone()
            print(f"  [IDENTITY TEST] Using kanji->kanji identity injection (expected no change)")

        # =====================================================
        # 1. Baseline: generate with kanji text, no injection
        # =====================================================
        def extract_audio(result):
            """Extract audio tensor and sample rate from generate_voice_clone result.

            Returns (List[np.ndarray], sample_rate) or (wavs, sr).
            """
            if isinstance(result, tuple) and len(result) == 2:
                wavs, sr = result
            else:
                wavs = result
                sr = 24000

            if isinstance(wavs, list) and len(wavs) > 0:
                wav = wavs[0]
            else:
                wav = wavs

            if isinstance(wav, np.ndarray):
                audio_cpu = torch.tensor(wav).float()
            elif isinstance(wav, torch.Tensor):
                audio_cpu = wav.cpu().float()
            else:
                audio_cpu = torch.tensor(wav).float()

            if audio_cpu.dim() > 2:
                audio_cpu = audio_cpu.squeeze(0)
            if audio_cpu.dim() == 1:
                audio_cpu = audio_cpu.unsqueeze(0)

            return audio_cpu, sr

        import torchaudio

        print(f"  [BL] Generating baseline...")
        seed_everything(seed)
        try:
            with torch.no_grad():
                bl_result = wrapper.generate_voice_clone(
                    text=[kanji_text],
                    language=["japanese"],
                    max_new_tokens=2048,
                    non_streaming_mode=True,
                    do_sample=(not greedy),
                    temperature=(1.0 if greedy else 0.9),
                    top_k=(None if greedy else 50),
                    top_p=(None if greedy else 1.0),
                    ref_audio=DEFAULT_REF_AUDIO,
                    x_vector_only_mode=True,
                )
            bl_audio_cpu, bl_sr = extract_audio(bl_result)

            bl_wav_path = os.path.join(wav_dir, f"pair{pair_idx}_bl.wav")
            torchaudio.save(bl_wav_path, bl_audio_cpu, bl_sr)

            bl_trans = transcribe_wav(bl_audio_cpu, bl_sr, whisper_model, whisper_processor, device)
            print(f"  [BL] Transcription: {bl_trans}")
        except Exception as e:
            print(f"  [BL] ERROR: {e}")
            import traceback
            traceback.print_exc()
            bl_trans = ""
            bl_sr = 24000

        # =====================================================
        # 2. Katakana baseline: generate with katakana text
        # =====================================================
        print(f"  [KT] Generating katakana baseline...")
        seed_everything(seed)
        try:
            with torch.no_grad():
                kt_result = wrapper.generate_voice_clone(
                    text=[katakana_text],
                    language=["japanese"],
                    max_new_tokens=2048,
                    non_streaming_mode=True,
                    do_sample=(not greedy),
                    temperature=(1.0 if greedy else 0.9),
                    top_k=(None if greedy else 50),
                    top_p=(None if greedy else 1.0),
                    ref_audio=DEFAULT_REF_AUDIO,
                    x_vector_only_mode=True,
                )
            kt_audio_cpu, kt_sr = extract_audio(kt_result)

            kt_wav_path = os.path.join(wav_dir, f"pair{pair_idx}_kt.wav")
            torchaudio.save(kt_wav_path, kt_audio_cpu, kt_sr)

            kt_trans = transcribe_wav(kt_audio_cpu, kt_sr, whisper_model, whisper_processor, device)
            print(f"  [KT] Transcription: {kt_trans}")
        except Exception as e:
            print(f"  [KT] ERROR: {e}")
            import traceback
            traceback.print_exc()
            kt_trans = ""

        # =====================================================
        # 3. Injection: hook Layer N during generation
        # =====================================================
        print(f"  [INJ] Generating with Layer {injection_layer} injection...")

        # We need to inject during the prefill phase of generation.
        # Strategy: install a forward hook on the target layer that replaces
        # entity position hidden states with the katakana reference vectors.
        #
        # Challenge: during autoregressive generation, the model processes one
        # token at a time after prefill. The injection only applies during
        # prefill (when seq_len > 1 and matches the text sequence).

        injection_done = [False]
        injection_count = [0]

        def make_injection_hook(layer_idx, kanji_s, kanji_e, kat_entity_hidden):
            """Create a hook that replaces entity positions at the target layer.

            For token-count mismatch (kanji tokens != katakana tokens),
            we mean-pool the katakana entity hidden and broadcast to kanji positions.
            """
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    rest = output[1:]
                else:
                    h = output
                    rest = None

                seq_len = h.shape[1]

                # Only inject during prefill (seq_len > 1)
                if seq_len <= 1:
                    return output

                # Check if this is the right sequence length (prefill)
                if kanji_e < seq_len:
                    kanji_len = kanji_e - kanji_s + 1
                    kat_len = kat_entity_hidden.shape[0]

                    if kanji_len == kat_len:
                        # Direct token-to-token replacement
                        h_modified = h.clone()
                        h_modified[0, kanji_s:kanji_e + 1] = kat_entity_hidden.to(h.dtype)
                    else:
                        # Token count mismatch: mean-pool katakana -> broadcast to kanji positions
                        kat_mean = kat_entity_hidden.mean(dim=0)
                        h_modified = h.clone()
                        h_modified[0, kanji_s:kanji_e + 1] = kat_mean.unsqueeze(0).expand(
                            kanji_len, -1
                        ).to(h.dtype)

                    injection_count[0] += 1
                    injection_done[0] = True

                    if rest is not None:
                        return (h_modified,) + rest
                    else:
                        return h_modified

                return output

            return hook_fn

        # Use inference layout positions for injection during generation
        hook = talker_model.layers[injection_layer].register_forward_hook(
            make_injection_hook(
                injection_layer, kanji_start_infer, kanji_end_infer, katakana_entity_hidden
            )
        )

        seed_everything(seed)
        try:
            with torch.no_grad():
                inj_result = wrapper.generate_voice_clone(
                    text=[kanji_text],
                    language=["japanese"],
                    max_new_tokens=2048,
                    non_streaming_mode=True,
                    do_sample=(not greedy),
                    temperature=(1.0 if greedy else 0.9),
                    top_k=(None if greedy else 50),
                    top_p=(None if greedy else 1.0),
                    ref_audio=DEFAULT_REF_AUDIO,
                    x_vector_only_mode=True,
                )
            inj_audio_cpu, inj_sr = extract_audio(inj_result)

            inj_wav_path = os.path.join(wav_dir, f"pair{pair_idx}_inj_L{injection_layer}.wav")
            torchaudio.save(inj_wav_path, inj_audio_cpu, inj_sr)

            inj_trans = transcribe_wav(inj_audio_cpu, inj_sr, whisper_model, whisper_processor, device)
            print(f"  [INJ] Transcription: {inj_trans}")
            print(f"  [INJ] Injection applied: {injection_done[0]} ({injection_count[0]} times)")
        except Exception as e:
            print(f"  [INJ] ERROR: {e}")
            import traceback
            traceback.print_exc()
            inj_trans = ""
        finally:
            hook.remove()

        # =====================================================
        # 4. Evaluate pronunciations
        # =====================================================
        bl_eval = check_pronunciation(bl_trans, pair["reading_hira"], pair["katakana"])
        kt_eval = check_pronunciation(kt_trans, pair["reading_hira"], pair["katakana"])
        inj_eval = check_pronunciation(inj_trans, pair["reading_hira"], pair["katakana"])

        pair_result = {
            "pair_idx": pair_idx,
            "entity_kanji": pair["kanji"],
            "entity_katakana": pair["katakana"],
            "reading_hira": pair["reading_hira"],
            "kanji_entity_tokens": kanji_entity_len,
            "katakana_entity_tokens": katakana_entity_len,
            "kanji_pos_conv": [kanji_start_conv, kanji_end_conv],
            "kanji_pos_infer": [kanji_start_infer, kanji_end_infer],
            "katakana_pos_conv": [katakana_start_conv, katakana_end_conv],
            "injection_layer": injection_layer,
            "injection_applied": injection_done[0],
            "injection_count": injection_count[0],
            "baseline": {
                "transcription": bl_trans,
                **bl_eval,
            },
            "katakana_baseline": {
                "transcription": kt_trans,
                **kt_eval,
            },
            "injection": {
                "transcription": inj_trans,
                **inj_eval,
            },
            "status": "ok",
        }

        # Did injection change the pronunciation?
        pair_result["pronunciation_changed"] = (bl_trans != inj_trans)
        pair_result["pronunciation_improved"] = (
            not bl_eval["lenient_match"] and inj_eval["lenient_match"]
        )
        pair_result["pronunciation_degraded"] = (
            bl_eval["lenient_match"] and not inj_eval["lenient_match"]
        )

        results.append(pair_result)

        print(f"  BL correct: {bl_eval['lenient_match']}, "
              f"INJ correct: {inj_eval['lenient_match']}, "
              f"Changed: {pair_result['pronunciation_changed']}, "
              f"Improved: {pair_result['pronunciation_improved']}")

        # Free GPU memory
        del katakana_hidden, katakana_entity_hidden
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS-1.7B Layer Injection Experiment (N=50)"
    )
    parser.add_argument(
        "--output_dir", default="eval_results/qwen3_injection",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model path"
    )
    parser.add_argument(
        "--injection_layers", default="26",
        help="Comma-separated layer indices to test injection (default: 26 only)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--entity_store", default=None,
        help="Path to entity_store.jsonl to load pairs from (default: use built-in 8 pairs)"
    )
    parser.add_argument(
        "--n_pairs", type=int, default=50,
        help="Number of entity pairs to use when loading from entity_store (default: 50)"
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use deterministic greedy decoding (do_sample=False) for layer-sweep ablation. "
             "Default is stochastic sampling (do_sample=True, temp=0.9)."
    )
    parser.add_argument(
        "--identity_test", action="store_true",
        help="Sanity check mode: inject kanji hidden state onto kanji (identity). "
             "Expected changed% == 0 if hook is bug-free."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    injection_layers = [int(x) for x in args.injection_layers.split(",")]

    # Build test pairs
    if args.entity_store is not None:
        print(f"\nLoading entity pairs from {args.entity_store} (n_pairs={args.n_pairs})...")
        test_pairs = load_pairs_from_entity_store(args.entity_store, n_pairs=args.n_pairs)
        print(f"  Loaded {len(test_pairs)} pairs")
        for i, p in enumerate(test_pairs):
            print(f"    [{i:02d}] {p['kanji']} -> {p['katakana']} ({p['reading_hira']})")
    else:
        test_pairs = INJECTION_PAIRS_DEFAULT
        print(f"\nUsing built-in {len(test_pairs)} test pairs (no --entity_store specified)")

    print(f"\nDevice: {device}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Injection layers: {injection_layers}")
    print(f"Seed: {args.seed}")
    print(f"N pairs: {len(test_pairs)}")
    t0 = time.time()

    # =========================================================
    # Load Qwen3-TTS
    # =========================================================
    print(f"\nLoading Qwen3-TTS from {args.model_path}...")
    from qwen_tts import Qwen3TTSModel

    wrapper = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
    )

    base_model = wrapper.model
    talker = base_model.talker
    talker_model = talker.model
    num_layers = len(talker_model.layers)

    load_time = time.time() - t0
    print(f"Qwen3-TTS loaded in {load_time:.1f}s")
    print(f"  Decoder layers: {num_layers}")
    print(f"  Hidden size: {talker.config.hidden_size}")

    # =========================================================
    # Load Whisper
    # =========================================================
    whisper_model, whisper_processor = load_whisper(device)

    # =========================================================
    # Run injection experiments
    # =========================================================
    all_results = {}

    for inj_layer in injection_layers:
        if inj_layer >= num_layers:
            print(f"\n[WARN] Layer {inj_layer} >= num_layers {num_layers}, skipping")
            continue

        layer_results = run_injection_experiment(
            wrapper=wrapper,
            whisper_model=whisper_model,
            whisper_processor=whisper_processor,
            test_pairs=test_pairs,
            injection_layer=inj_layer,
            device=device,
            output_dir=args.output_dir,
            seed=args.seed,
            greedy=args.greedy,
            identity_test=args.identity_test,
        )

        all_results[f"layer_{inj_layer}"] = layer_results

    # =========================================================
    # Aggregate and save
    # =========================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    summary = {}
    for layer_key, layer_results in all_results.items():
        ok_results = [r for r in layer_results if r.get("status") == "ok"]
        n = len(ok_results)
        if n == 0:
            continue

        bl_correct = sum(1 for r in ok_results if r["baseline"]["lenient_match"])
        inj_correct = sum(1 for r in ok_results if r["injection"]["lenient_match"])
        kt_correct = sum(1 for r in ok_results if r["katakana_baseline"]["lenient_match"])
        changed = sum(1 for r in ok_results if r["pronunciation_changed"])
        improved = sum(1 for r in ok_results if r["pronunciation_improved"])
        degraded = sum(1 for r in ok_results if r["pronunciation_degraded"])

        layer_summary = {
            "n_pairs": n,
            "bl_correct": bl_correct,
            "bl_accuracy": round(bl_correct / n * 100, 1),
            "inj_correct": inj_correct,
            "inj_accuracy": round(inj_correct / n * 100, 1),
            "kt_correct": kt_correct,
            "kt_accuracy": round(kt_correct / n * 100, 1),
            "pronunciation_changed": changed,
            "change_rate": round(changed / n * 100, 1),
            "improved": improved,
            "degraded": degraded,
            "delta_pp": round((inj_correct - bl_correct) / n * 100, 1),
        }
        summary[layer_key] = layer_summary

        print(f"\n{layer_key}:")
        print(f"  Baseline accuracy: {bl_correct}/{n} ({layer_summary['bl_accuracy']}%)")
        print(f"  Injection accuracy: {inj_correct}/{n} ({layer_summary['inj_accuracy']}%)")
        print(f"  Katakana BL accuracy: {kt_correct}/{n} ({layer_summary['kt_accuracy']}%)")
        print(f"  Change rate: {changed}/{n} ({layer_summary['change_rate']}%)")
        print(f"  Improved: {improved}, Degraded: {degraded}")
        print(f"  Delta: {layer_summary['delta_pp']:+.1f}pp")

    # Save results
    output = {
        "model": "Qwen3-TTS-1.7B",
        "num_layers": num_layers,
        "seed": args.seed,
        "injection_layers": injection_layers,
        "n_pairs": len(test_pairs),
        "entity_store": args.entity_store,
        "test_pairs_meta": [
            {"kanji": p["kanji"], "katakana": p["katakana"], "reading_hira": p["reading_hira"]}
            for p in test_pairs
        ],
        "summary": summary,
        "pair_results": all_results,
        "convergence_reference": {
            "peak_layer": 26,
            "peak_cos": 0.974,
            "output_cos": 0.905,
            "comparison": {
                "T5Gemma-TTS": {"max_cos": 1.0, "output_cos": 1.0, "injection_delta": "+27.5pp"},
                "Qwen2-0.5B": {"max_cos": 0.85, "output_cos": 0.681, "injection_delta": "N/A"},
            }
        },
    }

    json_path = os.path.join(args.output_dir, "qwen3_injection_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
