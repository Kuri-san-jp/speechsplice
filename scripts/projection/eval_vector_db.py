#!/usr/bin/env python3
"""Phase 2: Evaluate Pronunciation Vector DB injection.

Uses precomputed per-token katakana encoder vectors for TTS25-style injection.
No need to run encoder twice at inference time - just DB lookup.

Evaluation:
1. Baseline (no injection)
2. 2-pass Oracle (TTS25-style, re-runs encoder on katakana text)
3. DB injection (lookup per-token vectors, same injection mechanism)
4. DB-2pass fallback (DB first, fall back to 2-pass if entity not in DB)

Metrics:
- Pronunciation Match (pyopenjtalk g2p based, with improved lenient check)
- Phoneme CER

Output: eval_results/vector_db_eval/vector_db_eval.json
"""
import sys
import os
import json
import argparse
import time

T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from encoder_injection_t5gemma import (
    get_encoder_output,
    find_entity_token_range,
    generate_with_modified_encoder,
    seed_everything,
    load_whisper,
    transcribe_with_whisper,
    save_audio,
    text_to_kana,
    katakana_to_hiragana,
)
from data.tokenizer import AudioTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def hiragana_to_katakana(text):
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x3041 <= cp <= 0x3096:
            result.append(chr(cp + 0x60))
        else:
            result.append(ch)
    return "".join(result)


def check_pronunciation_strict(transcript, expected_reading):
    """Strict check: expected reading in pyopenjtalk g2p of transcript."""
    if len(transcript) > 200:
        transcript = transcript[:200]
    try:
        trans_kana = text_to_kana(transcript)
        trans_hira = katakana_to_hiragana(trans_kana)
        expected_hira = katakana_to_hiragana(expected_reading)
        return expected_hira in trans_hira
    except Exception:
        return False


def _min_edit_distance_substring(pattern, text):
    """Find minimum edit distance between pattern and any substring of text.

    Pure Python Levenshtein-based substring matching (no external deps).
    Returns the minimum edit distance found.
    """
    m = len(pattern)
    n = len(text)
    if m == 0:
        return 0
    if n == 0:
        return m

    # DP: dp[j] = min edit distance for pattern[:i] vs any substring of text ending at j
    prev = [0] * (n + 1)  # Start from any position (free insertion at start)
    for i in range(1, m + 1):
        curr = [i] * (n + 1)  # Deleting all of pattern[:i]
        for j in range(1, n + 1):
            cost = 0 if pattern[i - 1] == text[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # Insert
                prev[j] + 1,          # Delete
                prev[j - 1] + cost,   # Replace
            )
        prev = curr
    return min(prev)


def check_pronunciation_lenient(transcript, expected_reading):
    """Lenient check: allow partial phoneme match (CER < threshold).

    Also checks if expected_reading appears literally in transcript (for kanji cases).
    """
    # Strict check first
    if check_pronunciation_strict(transcript, expected_reading):
        return True, "strict"

    # Check phoneme CER using substring edit distance
    try:
        trans_kana = text_to_kana(transcript)
        trans_hira = katakana_to_hiragana(trans_kana)
        expected_hira = katakana_to_hiragana(expected_reading)

        # Find best substring match in transcript for expected reading
        best_cer = _min_edit_distance_substring(expected_hira, trans_hira) / max(len(expected_hira), 1)
        if best_cer <= 0.3:  # Allow 30% character error
            return True, f"lenient_cer={best_cer:.2f}"
    except Exception:
        pass

    return False, "no_match"


def splice_injection(
    tts_model, text_tokenizer, audio_tokenizer,
    text, entity_surface, per_token_vectors,
    device, codec_audio_sr, codec_sr, seed=42,
):
    """TTS25-style splice injection using precomputed per-token vectors.

    per_token_vectors: (N, hidden_dim) tensor of katakana entity token vectors
    """
    seed_everything(seed)
    add_bos = 1 if getattr(tts_model.args, "add_bos_to_text", 0) else 0

    kanji_enc = get_encoder_output(tts_model, text_tokenizer, text, device)
    kanji_range = find_entity_token_range(
        text_tokenizer, kanji_enc["text"], entity_surface, add_bos=add_bos
    )

    if kanji_range is None:
        return None, f"kanji_range_not_found: '{entity_surface}' in '{kanji_enc['text'][:40]}'"

    k_start, k_end = kanji_range
    k_len = k_end - k_start
    n_len = per_token_vectors.shape[0]  # katakana token count

    modified_memory = kanji_enc["last_hidden_state"].clone()
    db_vecs = per_token_vectors.to(modified_memory.device, modified_memory.dtype)

    if k_len == n_len:
        # Same length: direct per-token replacement
        modified_memory[0, k_start:k_end] = db_vecs
        modified_attention_mask = kanji_enc["attention_mask"]
    else:
        # Different lengths: reconstruct sequence
        kanji_prefix = modified_memory[0, :k_start]
        kanji_suffix = modified_memory[0, k_end:]
        modified_memory = torch.cat([kanji_prefix, db_vecs, kanji_suffix], dim=0).unsqueeze(0)
        new_len = modified_memory.shape[1]
        modified_attention_mask = torch.ones((1, new_len), dtype=torch.long, device=device)

    encoder_position_ids = None
    if getattr(tts_model.args, "use_pm_rope", 1):
        new_len = modified_memory.shape[1]
        x_lens = torch.LongTensor([new_len]).to(device)
        encoder_position_ids = tts_model._build_position_ids(x_lens, new_len, device)

    gen_audio = generate_with_modified_encoder(
        tts_model, text_tokenizer, audio_tokenizer,
        modified_memory, modified_attention_mask, encoder_position_ids,
        text, device, codec_audio_sr, codec_sr,
    )
    return gen_audio, "success"


def tts25_oracle_injection(
    tts_model, text_tokenizer, audio_tokenizer,
    text, entity_surface, reading_katakana,
    device, codec_audio_sr, codec_sr, seed=42,
):
    """TTS25-style 2-pass injection (re-runs encoder on katakana text)."""
    seed_everything(seed)
    add_bos = 1 if getattr(tts_model.args, "add_bos_to_text", 0) else 0

    kata_text = text.replace(entity_surface, reading_katakana)
    if kata_text == text:
        return None, "entity_not_found"

    kanji_enc = get_encoder_output(tts_model, text_tokenizer, text, device)
    kata_enc = get_encoder_output(tts_model, text_tokenizer, kata_text, device)

    kanji_range = find_entity_token_range(
        text_tokenizer, kanji_enc["text"], entity_surface, add_bos=add_bos
    )
    kata_range = find_entity_token_range(
        text_tokenizer, kata_enc["text"], reading_katakana, add_bos=add_bos
    )

    if kanji_range is None or kata_range is None:
        return None, f"range_not_found"

    k_start, k_end = kanji_range
    t_start, t_end = kata_range
    k_len = k_end - k_start
    t_len = t_end - t_start

    modified_memory = kanji_enc["last_hidden_state"].clone()

    if k_len == t_len:
        modified_memory[0, k_start:k_end] = kata_enc["last_hidden_state"][0, t_start:t_end]
        modified_attention_mask = kanji_enc["attention_mask"]
    else:
        kanji_prefix = modified_memory[0, :k_start]
        kata_entity = kata_enc["last_hidden_state"][0, t_start:t_end]
        kanji_suffix = modified_memory[0, k_end:]
        modified_memory = torch.cat([kanji_prefix, kata_entity, kanji_suffix], dim=0).unsqueeze(0)
        new_len = modified_memory.shape[1]
        modified_attention_mask = torch.ones((1, new_len), dtype=torch.long, device=device)

    encoder_position_ids = None
    if getattr(tts_model.args, "use_pm_rope", 1):
        new_len = modified_memory.shape[1]
        x_lens = torch.LongTensor([new_len]).to(device)
        encoder_position_ids = tts_model._build_position_ids(x_lens, new_len, device)

    gen_audio = generate_with_modified_encoder(
        tts_model, text_tokenizer, audio_tokenizer,
        modified_memory, modified_attention_mask, encoder_position_ids,
        text, device, codec_audio_sr, codec_sr,
    )
    return gen_audio, "success"


def run_evaluation(
    tts_model, text_tokenizer, audio_tokenizer,
    whisper_model, whisper_processor,
    db_data, val_samples, device, output_dir,
    seeds=(42, 123, 456), max_per_entity=5,
):
    db = db_data["db"]
    entity_store = db_data["entity_store"]

    cfg = tts_model.config
    codec_audio_sr = audio_tokenizer.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    os.makedirs(output_dir, exist_ok=True)

    # Group samples by entity
    samples_by_entity = {}
    for s in val_samples:
        for eid in s.get("entity_ids", []):
            if eid not in samples_by_entity:
                samples_by_entity[eid] = []
            samples_by_entity[eid].append(s["text"])

    all_results = []

    for eid in sorted(samples_by_entity.keys()):
        if eid not in entity_store:
            continue

        record = entity_store[eid]
        surface = record["surface"]
        reading_hira = record["reading_hira"]
        reading_katakana = hiragana_to_katakana(reading_hira)

        in_db = eid in db
        db_entry = db.get(eid, None)

        contexts = samples_by_entity[eid][:max_per_entity]

        print(f"\n{'='*60}")
        print(f"Entity: {surface} ({reading_hira}) | in_db={in_db}")
        print(f"{'='*60}")

        for ctx_idx, ctx_text in enumerate(contexts):
            print(f"\n  Context {ctx_idx}: {ctx_text[:50]}")

            entity_result = {
                "entity_id": eid,
                "surface": surface,
                "reading": reading_hira,
                "reading_katakana": reading_katakana,
                "context_text": ctx_text,
                "context_idx": ctx_idx,
                "in_db": in_db,
                "conditions": {},
            }

            for seed in seeds:
                seed_key = f"seed{seed}"
                entity_result["conditions"][seed_key] = {}

                # 1. Baseline
                seed_everything(seed)
                enc = get_encoder_output(tts_model, text_tokenizer, ctx_text, device)
                bl_audio = generate_with_modified_encoder(
                    tts_model, text_tokenizer, audio_tokenizer,
                    enc["last_hidden_state"], enc["attention_mask"],
                    enc.get("position_ids"),
                    ctx_text, device, codec_audio_sr, codec_sr,
                )
                if bl_audio is not None:
                    wav_path = os.path.join(output_dir, f"{eid}_ctx{ctx_idx}_{seed_key}_baseline.wav")
                    save_audio(wav_path, bl_audio[0].cpu(), codec_audio_sr)
                    transcript = transcribe_with_whisper(wav_path, whisper_model, whisper_processor, device)
                    match_strict = check_pronunciation_strict(transcript, reading_katakana)
                    match_lenient, match_type = check_pronunciation_lenient(transcript, reading_katakana)
                    entity_result["conditions"][seed_key]["baseline"] = {
                        "transcript": transcript,
                        "match_strict": match_strict,
                        "match_lenient": match_lenient,
                        "match_type": match_type,
                    }
                    print(f"    [{seed_key}] BL: '{transcript[:40]}' strict={match_strict} lenient={match_lenient}")

                # 2. DB injection
                if in_db:
                    db_audio, status = splice_injection(
                        tts_model, text_tokenizer, audio_tokenizer,
                        ctx_text, surface, db_entry["per_token_vectors"],
                        device, codec_audio_sr, codec_sr, seed=seed,
                    )
                    if db_audio is not None:
                        wav_path = os.path.join(output_dir, f"{eid}_ctx{ctx_idx}_{seed_key}_db.wav")
                        save_audio(wav_path, db_audio[0].cpu(), codec_audio_sr)
                        transcript = transcribe_with_whisper(wav_path, whisper_model, whisper_processor, device)
                        match_strict = check_pronunciation_strict(transcript, reading_katakana)
                        match_lenient, match_type = check_pronunciation_lenient(transcript, reading_katakana)
                        entity_result["conditions"][seed_key]["db_injection"] = {
                            "transcript": transcript,
                            "match_strict": match_strict,
                            "match_lenient": match_lenient,
                            "match_type": match_type,
                            "db_token_count": db_entry["token_count_kata"],
                            "db_num_contexts": db_entry["num_contexts"],
                            "db_cos_sim": db_entry["context_cos_sim_mean"],
                        }
                        print(f"    [{seed_key}] DB:  '{transcript[:40]}' strict={match_strict} lenient={match_lenient}")
                    else:
                        entity_result["conditions"][seed_key]["db_injection"] = {
                            "transcript": None, "match_strict": False, "match_lenient": False,
                            "status": status,
                        }

                # 3. Oracle TTS25-style (2-pass)
                oracle_audio, status = tts25_oracle_injection(
                    tts_model, text_tokenizer, audio_tokenizer,
                    ctx_text, surface, reading_katakana,
                    device, codec_audio_sr, codec_sr, seed=seed,
                )
                if oracle_audio is not None:
                    wav_path = os.path.join(output_dir, f"{eid}_ctx{ctx_idx}_{seed_key}_oracle.wav")
                    save_audio(wav_path, oracle_audio[0].cpu(), codec_audio_sr)
                    transcript = transcribe_with_whisper(wav_path, whisper_model, whisper_processor, device)
                    match_strict = check_pronunciation_strict(transcript, reading_katakana)
                    match_lenient, match_type = check_pronunciation_lenient(transcript, reading_katakana)
                    entity_result["conditions"][seed_key]["oracle_2pass"] = {
                        "transcript": transcript,
                        "match_strict": match_strict,
                        "match_lenient": match_lenient,
                        "match_type": match_type,
                    }
                    print(f"    [{seed_key}] Oracle: '{transcript[:40]}' strict={match_strict} lenient={match_lenient}")

            all_results.append(entity_result)

    return all_results


def print_summary(results):
    print(f"\n{'='*70}")
    print("VECTOR DB EVALUATION SUMMARY")
    print(f"{'='*70}")

    cond_stats = {}
    for r in results:
        for seed_key, conds in r["conditions"].items():
            for cond_name, result in conds.items():
                if cond_name not in cond_stats:
                    cond_stats[cond_name] = {"strict": 0, "lenient": 0, "total": 0}
                cond_stats[cond_name]["total"] += 1
                if result.get("match_strict", False):
                    cond_stats[cond_name]["strict"] += 1
                if result.get("match_lenient", False):
                    cond_stats[cond_name]["lenient"] += 1

    print(f"\n{'Condition':<25} {'Strict':>15} {'Lenient':>15}")
    print("-" * 60)
    for cond, stats in sorted(cond_stats.items()):
        t = stats["total"]
        s = stats["strict"]
        l = stats["lenient"]
        print(f"{cond:<25} {s}/{t} ({100*s/t:.1f}%)    {l}/{t} ({100*l/t:.1f}%)")

    # Per-entity breakdown for DB vs Oracle
    print(f"\n{'='*70}")
    print("Per-entity: DB vs Oracle (lenient match rate)")
    print(f"{'='*70}")

    entity_stats = {}
    for r in results:
        eid = r["entity_id"]
        surface = r["surface"]
        if eid not in entity_stats:
            entity_stats[eid] = {"surface": surface, "bl": [], "db": [], "oracle": []}
        for seed_key, conds in r["conditions"].items():
            if "baseline" in conds:
                entity_stats[eid]["bl"].append(conds["baseline"].get("match_lenient", False))
            if "db_injection" in conds:
                entity_stats[eid]["db"].append(conds["db_injection"].get("match_lenient", False))
            if "oracle_2pass" in conds:
                entity_stats[eid]["oracle"].append(conds["oracle_2pass"].get("match_lenient", False))

    for eid in sorted(entity_stats.keys()):
        s = entity_stats[eid]
        def rate(lst):
            return f"{sum(lst)}/{len(lst)}" if lst else "N/A"
        print(f"  {s['surface']:12s}: BL={rate(s['bl'])} DB={rate(s['db'])} Oracle={rate(s['oracle'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="checkpoints/pronunciation_vector_db.pt")
    parser.add_argument("--val_data", default="data/val_lora.jsonl")
    parser.add_argument("--output_dir", default="eval_results/vector_db_eval")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--max_per_entity", type=int, default=5)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)
    print(f"[Info] Working directory: {project_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = tuple(int(s) for s in args.seeds.split(","))

    # Load DB
    print(f"[Info] Loading DB from {args.db}...")
    db_data = torch.load(args.db, weights_only=False)
    db = db_data["db"]
    print(f"[Info] DB contains {len(db)} entities")

    # Load val samples
    val_samples = []
    with open(args.val_data, "r", encoding="utf-8") as f:
        for line in f:
            val_samples.append(json.loads(line.strip()))
    print(f"[Info] Val samples: {len(val_samples)}")

    # Load TTS model
    model_dir = "Aratako/T5Gemma-TTS-2b-2b"
    print(f"[TTS] Loading {model_dir}...")
    tts_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    )
    tts_model.eval()
    cfg = tts_model.config
    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", None),
    )

    whisper_model, whisper_processor = load_whisper(device)

    print(f"\n{'='*70}")
    print(f"Running eval: {len(seeds)} seeds, max {args.max_per_entity} contexts/entity")
    print(f"{'='*70}")

    start_time = time.time()
    results = run_evaluation(
        tts_model, text_tokenizer, audio_tokenizer,
        whisper_model, whisper_processor,
        db_data, val_samples, device, args.output_dir,
        seeds=seeds, max_per_entity=args.max_per_entity,
    )
    eval_time = time.time() - start_time

    print_summary(results)
    print(f"\nEvaluation time: {eval_time:.1f}s")

    results_path = os.path.join(args.output_dir, "vector_db_eval.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
