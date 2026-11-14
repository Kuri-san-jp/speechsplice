#!/usr/bin/env python3
"""T5Gemma-TTS Encoder Activation Injection Test.

Tests "controllability" of pronunciation by replacing encoder hidden states
at entity token positions:
1. Run encoder on kanji text -> encoder_output_kanji
2. Run encoder on katakana replacement text -> encoder_output_kana
3. Replace entity token positions in kanji output with kana output
4. Generate audio using the modified encoder output
5. Transcribe with Whisper to verify pronunciation change

Also tests intermediate layer injection (before Layer 9 convergence).
"""
import sys
import json
import os
import time
import re
import copy
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
T5GEMMA_ROOT = os.environ.get(
    "T5GEMMA_ROOT", os.path.join(_THIS_DIR, "_vendor", "t5gemma_tts")
)
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
    # Whisper max is 30 seconds at 16kHz = 480000 samples
    max_samples = 16000 * 30
    wav_mono = wav.squeeze()
    if wav_mono.shape[0] > max_samples:
        wav_mono = wav_mono[:max_samples]
    input_features = processor(
        wav_mono.numpy(), sampling_rate=16000, return_tensors="pt",
    ).input_features.to(device, dtype=torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            input_features, language="ja", task="transcribe",
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def text_to_kana(text):
    """Katakana reading via pykakasi (pyopenjtalk removed 2026-09-27)."""
    import pykakasi
    text = re.sub(r'[\s\u3000]', '', text)
    text = re.sub(r'[\u3002\u3001\uff01\uff1f!?,.]', '', text)
    return ''.join(x['kana'] for x in pykakasi.kakasi().convert(text))


def katakana_to_hiragana(text):
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            result.append(chr(cp - 0x60))
        else:
            result.append(ch)
    return ''.join(result)


def get_encoder_output(model, text_tokenizer, text, device, lang=None):
    """Get encoder output and token info for a given text."""
    target_text, lang_code = normalize_text_with_lang(text, lang)

    tokens = text_tokenizer.encode(target_text, add_special_tokens=False)
    add_eos = int(getattr(model.args, "add_eos_to_text", 0))
    add_bos = int(getattr(model.args, "add_bos_to_text", 0))
    if add_eos:
        tokens.append(add_eos)
    if add_bos:
        tokens = [add_bos] + tokens

    input_ids = torch.LongTensor([tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)

    encoder = model.encoder_module
    position_ids = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        x_lens = torch.LongTensor([len(tokens)]).to(device)
        position_ids = model._build_position_ids(x_lens, len(tokens), device)

    with torch.no_grad():
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    return {
        'last_hidden_state': encoder_outputs.last_hidden_state,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'tokens': tokens,
        'text': target_text,
    }


def get_encoder_layer_output(model, text_tokenizer, text, device, target_layer, lang=None):
    """Get intermediate encoder layer output and run remaining layers.

    Returns the hidden state at a specific layer for potential injection.
    """
    target_text, lang_code = normalize_text_with_lang(text, lang)

    tokens = text_tokenizer.encode(target_text, add_special_tokens=False)
    add_eos = int(getattr(model.args, "add_eos_to_text", 0))
    add_bos = int(getattr(model.args, "add_bos_to_text", 0))
    if add_eos:
        tokens.append(add_eos)
    if add_bos:
        tokens = [add_bos] + tokens

    input_ids = torch.LongTensor([tokens]).to(device)

    encoder = model.encoder_module
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['hidden'] = output[0].detach().clone()
        else:
            captured['hidden'] = output.detach().clone()

    layer = encoder.layers[target_layer]
    handle = layer.register_forward_hook(hook_fn)

    attention_mask = torch.ones_like(input_ids)
    position_ids = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        x_lens = torch.LongTensor([len(tokens)]).to(device)
        position_ids = model._build_position_ids(x_lens, len(tokens), device)

    with torch.no_grad():
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    handle.remove()

    return {
        'layer_hidden': captured['hidden'],
        'final_hidden': encoder_outputs.last_hidden_state,
        'tokens': tokens,
    }


@torch.no_grad()
def generate_with_modified_encoder(
    model, text_tokenizer, audio_tokenizer,
    modified_memory, encoder_attention_mask, encoder_position_ids,
    target_text, device, codec_audio_sr, codec_sr,
):
    """Run decoder generation with a modified encoder output (memory)."""
    import torch.nn as nn
    from functools import partial

    # Prepare decoder input (no reference audio: zero-length prompt)
    n_codebooks = 1
    y = torch.empty((1, n_codebooks, 0), dtype=torch.long, device=device)
    y_len = y.shape[-1]
    prompt_frames = y_len

    target_text_norm, lang_code = normalize_text_with_lang(target_text, None)
    target_duration = estimate_duration(target_text=target_text_norm, target_lang=lang_code)

    tgt_y_lens = torch.LongTensor(
        [int(y_len + codec_sr * target_duration)]
    ).to(device)

    # Use model's inference_tts-like logic but with injected memory
    eog_inference = (
        int(model.args.eos) if int(getattr(model.args, "eos", -1)) > 0 else int(model.args.eog)
    )

    batch_size = 1
    memory = modified_memory

    if int(getattr(model.args, "special_first", 0)):
        y = y + int(model.args.n_special)
    y = y.transpose(2, 1).contiguous()  # [1, K, T] -> [1, T, K]
    y_len = y.shape[1]  # T dimension after transpose

    target_total = int(tgt_y_lens[0].item())
    extra_cutoff = getattr(model.args, "extra_cutoff", 5.0)
    cutoff_limit = target_total + int(codec_sr * extra_cutoff)

    # For zero-shot (no reference audio), just start with BOS token
    bos_token_id = model.args.empty_token
    if y_len == 0:
        # No reference frames: cated_y is just BOS
        cated_y = torch.full(
            (batch_size, 1, 1),
            bos_token_id,
            dtype=torch.long, device=device,
        )
    else:
        bos = torch.full(
            (batch_size, 1, 1),
            bos_token_id,
            dtype=torch.long, device=device,
        )
        y_for_cat = y.transpose(1, 2).contiguous()  # back to [1, K, T]
        cated_y = torch.cat([bos, y_for_cat], dim=2)

    embedded_y = model.audio_embedding[0](cated_y[:, 0])
    embedded_y = model.audio_dropout(embedded_y)

    y_padding_mask = torch.full(
        (batch_size, embedded_y.shape[1]), False, device=device
    )
    current_length = embedded_y.shape[1]
    prompt_offset = prompt_frames + 1  # +BOS

    decoder_attention_mask = (~y_padding_mask).long()

    est_total = int(target_total) + 1
    max_gen_length = est_total + int(getattr(model.args, "encodec_sr", 50) * 10)
    full_dec_attention_mask = torch.ones(
        (batch_size, max_gen_length), dtype=torch.long, device=device
    )

    cur_len = embedded_y.shape[1]
    pm_kwargs = {}
    if int(getattr(model.args, "use_pm_rope", 1)):
        base = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(0)
        decoder_position_ids_full = (
            base / max(1, est_total - 1) * model.progress_scale
        )
        pm_kwargs["position_ids"] = decoder_position_ids_full
        pm_kwargs["pm_decoder_position_ids"] = decoder_position_ids_full
        pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
    else:
        pm_kwargs["position_ids"] = None

    decoder_outputs = model.decoder_module(
        inputs_embeds=embedded_y,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=memory,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=True,
        **pm_kwargs,
    )
    last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
    past_key_values = decoder_outputs.past_key_values

    # topk_sampling is defined at module level in the HF export
    # Import from the correct location
    try:
        from models.utils import topk_sampling
    except ImportError:
        # Fallback: use a simple implementation
        def topk_sampling(logits, top_k=10, top_p=1.0, min_p=0.0, temperature=1.0):
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            return token

    generated_tokens = []
    cur_num_gen = 0
    prev_token = -1
    consec_silence = 0
    silence_tokens = []

    while True:
        logits = model.predict_layer[0](last_hidden).squeeze(1)  # [1, V]

        effective_length = max(0, current_length - prompt_offset)
        if effective_length == 0:
            logits[:, eog_inference] = -1e9
        if cur_num_gen <= model.args.encodec_sr // 5:
            logits[:, eog_inference] = -10000.0

        tokens = topk_sampling(
            logits, top_k=30, top_p=0.9, min_p=0, temperature=0.8,
        )
        token = tokens[0, 0].item()

        if token == eog_inference:
            break

        generated_tokens.append(token)
        cur_num_gen += 1
        current_length += 1

        if cutoff_limit and current_length >= cutoff_limit:
            break

        # Prepare next step
        new_embed = model.audio_embedding[0](
            torch.LongTensor([[token]]).to(device)
        )
        new_embed = model.audio_dropout(new_embed)

        step_dec_mask = full_dec_attention_mask[:, :current_length]

        step_pm_kwargs = {}
        if int(getattr(model.args, "use_pm_rope", 1)):
            step_pos = torch.tensor(
                [[current_length - 1]], device=device, dtype=torch.float32
            )
            step_pos = step_pos / max(1, est_total - 1) * model.progress_scale
            step_pm_kwargs["position_ids"] = step_pos
            step_pm_kwargs["pm_decoder_position_ids"] = step_pos
            step_pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
        else:
            step_pm_kwargs["position_ids"] = None

        decoder_outputs = model.decoder_module(
            inputs_embeds=new_embed,
            attention_mask=step_dec_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            **step_pm_kwargs,
        )
        last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
        past_key_values = decoder_outputs.past_key_values
        prev_token = token

    if not generated_tokens:
        return None

    gen_frames = torch.LongTensor([generated_tokens]).unsqueeze(1).to(device)  # [1, 1, T]

    # Strip eos
    eos_token = eog_inference
    y_sep_token = getattr(model.args, "y_sep_token", None)
    mask = torch.ones_like(gen_frames, dtype=torch.bool)
    if y_sep_token is not None:
        mask &= gen_frames.ne(y_sep_token)
    if eos_token is not None:
        mask &= gen_frames.ne(eos_token)
    if not mask.all():
        valid_len = int(mask[0, 0].sum().item())
        gen_frames = gen_frames[0, 0][mask[0, 0]][:valid_len].view(1, 1, -1)

    gen_audio = audio_tokenizer.decode(gen_frames)
    return gen_audio


def find_entity_token_range(text_tokenizer, full_text, entity_text, add_bos=0, add_eos=0):
    """Find the token index range for the entity within the full text."""
    offset = 1 if add_bos else 0

    # Primary method: character offsets -> token range via offset mapping.
    # This is position-robust. The legacy subsequence search below fails for
    # mid-sentence entities in space-delimited languages: encode(entity) yields
    # tokens WITHOUT the SentencePiece "\u2581" space marker while the in-sentence
    # entity carries it, and the char-level fallback then mis-computes the
    # prefix length by one token (trailing-space token), silently replacing the
    # word AFTER the entity. (Bug found 2026-07-11.)
    try:
        enc = text_tokenizer(full_text, add_special_tokens=False,
                             return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        pos = full_text.find(entity_text)
        if pos < 0:
            pos = full_text.lower().find(entity_text.lower())
        if pos >= 0 and offsets:
            end_char = pos + len(entity_text)
            idx = [i for i, (s, e) in enumerate(offsets) if s < end_char and e > pos]
            if idx:
                return (min(idx) + offset, max(idx) + 1 + offset)
    except Exception:
        pass  # fast tokenizer が無い場合などは旧ロジックへ

    # Tokenize full text
    full_tokens = text_tokenizer.encode(full_text, add_special_tokens=False)
    # Tokenize entity alone
    entity_tokens = text_tokenizer.encode(entity_text, add_special_tokens=False)


    # Search for entity token subsequence
    for start in range(len(full_tokens) - len(entity_tokens) + 1):
        if full_tokens[start:start+len(entity_tokens)] == entity_tokens:
            return (start + offset, start + len(entity_tokens) + offset)

    # Fallback 1: character-level approach (case-sensitive).
    entity_pos = full_text.find(entity_text)
    if entity_pos >= 0:
        prefix = full_text[:entity_pos]
        prefix_tokens = text_tokenizer.encode(prefix, add_special_tokens=False)
        return (len(prefix_tokens) + offset, len(prefix_tokens) + len(entity_tokens) + offset)

    # Fallback 2: case-insensitive character-level search, then re-tokenize the
    # surface form actually present in full_text (handles e.g.\ "Perfect" in a
    # sentence-initial position when entity_text="perfect").
    low_full = full_text.lower()
    low_ent = entity_text.lower()
    pos = low_full.find(low_ent)
    if pos >= 0:
        actual_surface = full_text[pos:pos + len(entity_text)]
        actual_entity_tokens = text_tokenizer.encode(actual_surface, add_special_tokens=False)
        # Try subsequence match again with case-matched surface form
        for start in range(len(full_tokens) - len(actual_entity_tokens) + 1):
            if full_tokens[start:start + len(actual_entity_tokens)] == actual_entity_tokens:
                return (start + offset, start + len(actual_entity_tokens) + offset)
        prefix_tokens = text_tokenizer.encode(full_text[:pos], add_special_tokens=False)
        return (len(prefix_tokens) + offset,
                len(prefix_tokens) + len(actual_entity_tokens) + offset)

    return None


def run_encoder_injection_final_layer(
    model, text_tokenizer, audio_tokenizer,
    kanji_text, katakana_text, entity_kanji, entity_katakana,
    device, codec_audio_sr, codec_sr, output_path,
    whisper_model, whisper_processor,
):
    """Injection at the final encoder output level."""
    print(f"  [Final Layer Injection]")

    # Get encoder outputs for both texts
    kanji_enc = get_encoder_output(model, text_tokenizer, kanji_text, device)
    katakana_enc = get_encoder_output(model, text_tokenizer, katakana_text, device)

    print(f"    Kanji tokens ({len(kanji_enc['tokens'])}): {kanji_enc['tokens']}")
    print(f"    Katakana tokens ({len(katakana_enc['tokens'])}): {katakana_enc['tokens']}")

    # Find entity token ranges
    add_bos = 1 if int(getattr(model.args, "add_bos_to_text", 0)) else 0
    kanji_range = find_entity_token_range(
        text_tokenizer, kanji_enc['text'], entity_kanji, add_bos=add_bos
    )
    katakana_range = find_entity_token_range(
        text_tokenizer, katakana_enc['text'], entity_katakana, add_bos=add_bos
    )

    if kanji_range is None or katakana_range is None:
        print(f"    ERROR: Could not find entity token ranges")
        return None

    print(f"    Kanji entity range: {kanji_range}")
    print(f"    Katakana entity range: {katakana_range}")

    # Create modified memory: kanji encoder output with entity positions replaced
    modified_memory = kanji_enc['last_hidden_state'].clone()
    kanji_start, kanji_end = kanji_range
    kata_start, kata_end = katakana_range
    kanji_len = kanji_end - kanji_start
    kata_len = kata_end - kata_start

    # Strategy: direct replacement at entity positions
    # If token counts differ, we need to handle the size mismatch
    if kanji_len == kata_len:
        # Same length: direct replacement
        modified_memory[0, kanji_start:kanji_end] = \
            katakana_enc['last_hidden_state'][0, kata_start:kata_end]
        print(f"    Direct replacement: {kanji_len} tokens")
        modified_attention_mask = kanji_enc['attention_mask']
    else:
        # Different lengths: reconstruct the full sequence
        # [kanji_prefix] + [katakana_entity] + [kanji_suffix]
        kanji_prefix = modified_memory[0, :kanji_start]  # before entity
        katakana_entity = katakana_enc['last_hidden_state'][0, kata_start:kata_end]
        kanji_suffix = modified_memory[0, kanji_end:]  # after entity

        modified_memory = torch.cat(
            [kanji_prefix, katakana_entity, kanji_suffix], dim=0
        ).unsqueeze(0)
        print(f"    Reconstructed: prefix={kanji_start} + entity={kata_len} + suffix={modified_memory.shape[1]-kanji_start-kata_len}")

        # Update attention mask for new length
        new_len = modified_memory.shape[1]
        modified_attention_mask = torch.ones(
            (1, new_len), dtype=torch.long, device=device
        )

    # Cosine similarity check
    cos_sim = torch.nn.functional.cosine_similarity(
        kanji_enc['last_hidden_state'][0, kanji_start:kanji_end].mean(dim=0, keepdim=True),
        katakana_enc['last_hidden_state'][0, kata_start:kata_end].mean(dim=0, keepdim=True),
        dim=-1,
    ).item()
    print(f"    Entity hidden cos similarity (kanji vs katakana): {cos_sim:.6f}")

    # Rebuild position_ids for new length if needed
    encoder_position_ids = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        new_len = modified_memory.shape[1]
        x_lens = torch.LongTensor([new_len]).to(device)
        encoder_position_ids = model._build_position_ids(x_lens, new_len, device)

    # Generate audio with modified encoder output
    gen_audio = generate_with_modified_encoder(
        model, text_tokenizer, audio_tokenizer,
        modified_memory, modified_attention_mask, encoder_position_ids,
        kanji_text, device, codec_audio_sr, codec_sr,
    )

    if gen_audio is None:
        print(f"    ERROR: No audio generated")
        return None

    gen_wav = gen_audio[0].cpu()
    save_audio(output_path, gen_wav, codec_audio_sr)
    duration_sec = gen_wav.shape[-1] / codec_audio_sr
    print(f"    Generated {duration_sec:.2f}s audio")

    # Transcribe
    transcript = transcribe_with_whisper(output_path, whisper_model, whisper_processor, device)
    print(f"    Whisper transcript: '{transcript}'")

    return {
        'transcript': transcript,
        'duration_sec': round(duration_sec, 2),
        'entity_cos_sim': round(cos_sim, 6),
        'audio_path': output_path,
    }


def run_encoder_injection_intermediate(
    model, text_tokenizer, audio_tokenizer,
    kanji_text, katakana_text, entity_kanji, entity_katakana,
    device, codec_audio_sr, codec_sr, output_path,
    whisper_model, whisper_processor, inject_layer,
):
    """Injection at an intermediate encoder layer.

    Strategy:
    1. Run kanji through encoder, capture hidden state at inject_layer
    2. Run katakana through encoder, capture hidden state at inject_layer
    3. Replace entity positions at inject_layer level
    4. Re-run remaining encoder layers with modified hidden state
    5. Use the final output for decoder generation
    """
    print(f"  [Layer {inject_layer} Injection]")

    encoder = model.encoder_module
    add_bos = 1 if int(getattr(model.args, "add_bos_to_text", 0)) else 0

    # Tokenize both texts
    kanji_text_norm, _ = normalize_text_with_lang(kanji_text, "ja")
    kata_text_norm, _ = normalize_text_with_lang(katakana_text, "ja")

    kanji_tokens = text_tokenizer.encode(kanji_text_norm, add_special_tokens=False)
    kata_tokens = text_tokenizer.encode(kata_text_norm, add_special_tokens=False)

    add_eos = int(getattr(model.args, "add_eos_to_text", 0))
    add_bos_val = int(getattr(model.args, "add_bos_to_text", 0))
    if add_eos:
        kanji_tokens.append(add_eos)
        kata_tokens.append(add_eos)
    if add_bos_val:
        kanji_tokens = [add_bos_val] + kanji_tokens
        kata_tokens = [add_bos_val] + kata_tokens

    kanji_ids = torch.LongTensor([kanji_tokens]).to(device)
    kata_ids = torch.LongTensor([kata_tokens]).to(device)

    # Find entity ranges
    kanji_range = find_entity_token_range(
        text_tokenizer, kanji_text_norm, entity_kanji, add_bos=1 if add_bos_val else 0
    )
    kata_range = find_entity_token_range(
        text_tokenizer, kata_text_norm, entity_katakana, add_bos=1 if add_bos_val else 0
    )

    if kanji_range is None or kata_range is None:
        print(f"    ERROR: Could not find entity token ranges for layer injection")
        return None

    # Capture intermediate hidden states using hooks
    kanji_hidden_at_layer = {}
    kata_hidden_at_layer = {}

    def make_capture_hook(storage):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                storage['hidden'] = output[0].detach().clone()
            else:
                storage['hidden'] = output.detach().clone()
        return hook_fn

    # Run kanji encoder
    handle = encoder.layers[inject_layer].register_forward_hook(
        make_capture_hook(kanji_hidden_at_layer)
    )
    kanji_mask = torch.ones_like(kanji_ids)
    kanji_pos = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        x_lens = torch.LongTensor([len(kanji_tokens)]).to(device)
        kanji_pos = model._build_position_ids(x_lens, len(kanji_tokens), device)
    with torch.no_grad():
        encoder(input_ids=kanji_ids, attention_mask=kanji_mask, position_ids=kanji_pos)
    handle.remove()

    # Run katakana encoder
    handle = encoder.layers[inject_layer].register_forward_hook(
        make_capture_hook(kata_hidden_at_layer)
    )
    kata_mask = torch.ones_like(kata_ids)
    kata_pos = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        x_lens = torch.LongTensor([len(kata_tokens)]).to(device)
        kata_pos = model._build_position_ids(x_lens, len(kata_tokens), device)
    with torch.no_grad():
        encoder(input_ids=kata_ids, attention_mask=kata_mask, position_ids=kata_pos)
    handle.remove()

    kanji_h = kanji_hidden_at_layer['hidden']  # [1, T_kanji, D]
    kata_h = kata_hidden_at_layer['hidden']  # [1, T_kata, D]

    kanji_start, kanji_end = kanji_range
    kata_start, kata_end = kata_range
    kanji_len = kanji_end - kanji_start
    kata_len = kata_end - kata_start

    # Create modified hidden state at inject_layer
    if kanji_len == kata_len:
        modified_h = kanji_h.clone()
        modified_h[0, kanji_start:kanji_end] = kata_h[0, kata_start:kata_end]
        modified_mask = kanji_mask
        modified_pos = kanji_pos
    else:
        kanji_prefix = kanji_h[0, :kanji_start]
        kata_entity = kata_h[0, kata_start:kata_end]
        kanji_suffix = kanji_h[0, kanji_end:]
        modified_h = torch.cat([kanji_prefix, kata_entity, kanji_suffix], dim=0).unsqueeze(0)
        new_len = modified_h.shape[1]
        modified_mask = torch.ones((1, new_len), dtype=torch.long, device=device)
        if int(getattr(model.args, "use_pm_rope", 1)):
            x_lens = torch.LongTensor([new_len]).to(device)
            modified_pos = model._build_position_ids(x_lens, new_len, device)
        else:
            modified_pos = None

    # Now run the remaining encoder layers (inject_layer+1 to end) with modified hidden state
    # We need to hook into the encoder to replace the hidden state after inject_layer
    # Strategy: run full encoder with a hook that replaces the output of inject_layer

    replacement_done = [False]

    def replacement_hook(module, input, output):
        if replacement_done[0]:
            return output
        replacement_done[0] = True
        if isinstance(output, tuple):
            return (modified_h,) + output[1:]
        return modified_h

    # We need to run the encoder again but with the modified hidden state injected
    # The cleanest way is to hook into the inject_layer and replace its output
    handle = encoder.layers[inject_layer].register_forward_hook(replacement_hook)

    # Use kanji input (but the layer output will be replaced)
    # For position_ids, use the modified position_ids if lengths differ
    if kanji_len != kata_len:
        # Cannot directly inject different-length sequence via hook easily
        # Fallback: use the same-length approach with averaging for mismatched tokens
        handle.remove()

        # Alternative: average katakana entity hidden states to match kanji entity length
        avg_kata_entity = kata_h[0, kata_start:kata_end].mean(dim=0, keepdim=True)
        modified_h_same_len = kanji_h.clone()
        modified_h_same_len[0, kanji_start:kanji_end] = avg_kata_entity.expand(kanji_len, -1)

        replacement_done[0] = False
        modified_h = modified_h_same_len

        handle = encoder.layers[inject_layer].register_forward_hook(replacement_hook)

        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=kanji_ids, attention_mask=kanji_mask, position_ids=kanji_pos,
            )
        handle.remove()

        final_memory = encoder_outputs.last_hidden_state
        final_mask = kanji_mask
        final_pos = kanji_pos
    else:
        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=kanji_ids, attention_mask=kanji_mask, position_ids=kanji_pos,
            )
        handle.remove()

        final_memory = encoder_outputs.last_hidden_state
        final_mask = kanji_mask
        final_pos = kanji_pos

    # Generate audio with modified encoder output
    gen_audio = generate_with_modified_encoder(
        model, text_tokenizer, audio_tokenizer,
        final_memory, final_mask, final_pos,
        kanji_text, device, codec_audio_sr, codec_sr,
    )

    if gen_audio is None:
        print(f"    ERROR: No audio generated")
        return None

    gen_wav = gen_audio[0].cpu()
    save_audio(output_path, gen_wav, codec_audio_sr)
    duration_sec = gen_wav.shape[-1] / codec_audio_sr
    print(f"    Generated {duration_sec:.2f}s audio")

    transcript = transcribe_with_whisper(output_path, whisper_model, whisper_processor, device)
    print(f"    Whisper transcript: '{transcript}'")

    return {
        'transcript': transcript,
        'duration_sec': round(duration_sec, 2),
        'inject_layer': inject_layer,
        'audio_path': output_path,
    }


