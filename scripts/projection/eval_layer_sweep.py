#!/usr/bin/env python3
"""Layer Sweep Ablation: なぜLayer 9か実証実験.

T5Gemma-TTS encoder の各層（layer 0-11）でinjectionしたときの
発音精度（Strict v2 / Lenient）を比較する。

各layerごとに:
  1. DBのkatakanaコンテキストからlayer_hiddenを再抽出（layer-specific DB vectors）
  2. forward hookで対象layerの出力にentity位置を置換
  3. 残りのlayerが修正後のhidden stateを処理
  4. 最終outputをgenerate_with_modified_encoderに渡す
  5. Whisper文字起こし → Strict v2 / Lenient評価

実験設定:
  - Layers: 0, 1, 2, ..., 11 (全12層)
  - Entities: 最初の15 entities (速度のため)
  - Val data: val_lora_clean_eval.jsonl の各entityの最初3 contexts
  - Seeds: 42のみ (1 seed)
  - 合計: 15 entities × 3 contexts × 1 seed = 45 evals × 12 layers = 540 合成
  - Baseline: layer injectionなし (final output) も評価

出力:
  eval_results/layer_sweep/layer_sweep_results.json

Usage:
  SLURM_TIME=06:00:00 slurm_run.sh \\
    python scripts/projection/eval_layer_sweep.py \\
    --db checkpoints/pronunciation_vector_db_expanded.pt \\
    --val_data data/val_lora_clean_eval.jsonl \\
    --output_dir eval_results/layer_sweep
"""
import sys
import os
import json
import argparse
import time

T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from encoder_injection_t5gemma import (
    get_encoder_output,
    get_encoder_layer_output,
    find_entity_token_range,
    generate_with_modified_encoder,
    seed_everything,
    load_whisper,
)
from reeval_strict_v2 import check_strict_v2
from eval_vector_db import check_pronunciation_lenient

from data.tokenizer import AudioTokenizer
from inference_tts_utils import normalize_text_with_lang
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


def transcribe_in_memory(audio_tensor, src_sr, whisper_model, processor, device):
    """WAV保存なしでWhisper文字起こし。"""
    import torchaudio
    wav = audio_tensor.squeeze()
    if wav.dim() == 0:
        return ""
    if src_sr != 16000:
        wav = torchaudio.functional.resample(wav.float(), src_sr, 16000)
    wav_mono = wav.float()[:16000 * 30].cpu()
    input_features = processor(
        wav_mono.numpy(), sampling_rate=16000, return_tensors="pt",
    ).input_features.to(device, dtype=torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            input_features, language="ja", task="transcribe",
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def extract_layer_db_vectors(tts_model, text_tokenizer, db_data, target_layer, device,
                             entity_filter=None):
    """各layerのDB vectors（katakanaコンテキスト → layer_hidden）を構築する。

    DBのper_token_vectorsはfinal encoder output (layer 11後)のvectorsだが、
    layer sweep実験ではtarget_layerの出力空間に合わせたvectorsが必要。

    アプローチ:
    - train_prepared.jsonl/val_lora.jsonlからDB構築時に使ったコンテキスト取得は難しい
    - 代わりに val_lora_clean_eval.jsonl のコンテキストを使って
      katakana置換テキストのtarget_layer出力を抽出する
    - ただしこれは評価コンテキストと同じなのでデータリークの懸念がある
    - → 実際には別のアプローチを使う：
      val_lora.jsonlを使って各entityのDB構築コンテキストを再現する

    Args:
        entity_filter: 処理するentity_idのset。Noneの場合は全entity。
                       評価対象のみに限定することで大幅な高速化が可能。

    Returns:
        layer_db: {entity_id: tensor(N_tokens, 2304)} target layer のvectors
    """
    db = db_data["db"]

    # DB構築時のコンテキストソースを再ロード（リーク防止）
    # build_vector_db.py と同じ: train_prepared.jsonl + val_lora.jsonl
    contexts_by_entity = {}
    for data_path in ["data/train_prepared.jsonl", "data/val_lora.jsonl"]:
        if not os.path.exists(data_path):
            print(f"  [WARN] {data_path} not found, skipping")
            continue
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                except Exception:
                    continue
                for eid in sample.get("entity_ids", []):
                    if eid not in contexts_by_entity:
                        contexts_by_entity[eid] = []
                    contexts_by_entity[eid].append(sample["text"])

    add_bos = db_data.get("add_bos", 0)

    layer_db = {}
    for eid, entry in db.items():
        if entity_filter is not None and eid not in entity_filter:
            continue
        surface = entry["surface"]
        reading_katakana = entry["reading_katakana"]

        contexts = contexts_by_entity.get(eid, [])
        # 最大10コンテキスト（build_vector_db.pyと同じ上限）
        use_contexts = contexts[:10]

        if not use_contexts:
            print(f"  [SKIP] {eid}: no contexts for layer {target_layer}")
            continue

        all_token_vecs = []
        for ctx_text in use_contexts:
            kata_text = ctx_text.replace(surface, reading_katakana)
            if kata_text == ctx_text:
                continue  # entityが含まれていない

            try:
                out = get_encoder_layer_output(
                    tts_model, text_tokenizer, kata_text, device, target_layer
                )
                layer_hidden = out["layer_hidden"]  # (1, T, D)
                # katakanaエンティティのtoken範囲を取得
                target_text, _ = normalize_text_with_lang(kata_text, "ja")
                kata_range = find_entity_token_range(
                    text_tokenizer, target_text, reading_katakana, add_bos=add_bos
                )
                if kata_range is None:
                    continue
                t_start, t_end = kata_range
                token_vecs = layer_hidden[0, t_start:t_end].float().cpu()
                if token_vecs.shape[0] == 0:
                    continue
                all_token_vecs.append(token_vecs)
            except Exception as e:
                print(f"  [WARN] layer={target_layer} eid={eid} ctx error: {e}")
                continue

        if not all_token_vecs:
            print(f"  [SKIP] layer={target_layer} eid={eid}: no valid vectors extracted")
            continue

        # 最頻出token countのvectorsのみ使用（build_vector_db.pyと同様）
        token_counts = [v.shape[0] for v in all_token_vecs]
        most_common_count = max(set(token_counts), key=token_counts.count)
        filtered = [v for v in all_token_vecs if v.shape[0] == most_common_count]

        if not filtered:
            continue

        # コンテキスト平均
        stacked = torch.stack(filtered, dim=0)  # (num_ctx, N, D)
        avg_vecs = stacked.mean(dim=0)  # (N, D)
        layer_db[eid] = avg_vecs

    print(f"  Layer {target_layer} DB: {len(layer_db)}/{len(db)} entities extracted")
    return layer_db


def run_injection_at_layer(tts_model, text_tokenizer, kanji_text, surface,
                           db_vecs, target_layer, add_bos, device):
    """Forward hookを使ってtarget_layerでentity位置を置換し、最終encoder outputを返す。

    Args:
        tts_model: TTS model
        text_tokenizer: tokenizer
        kanji_text: 漢字入力テキスト
        surface: entity表層形（漢字）
        db_vecs: layer-specific DB vectors tensor (N_tokens, D)
        target_layer: injectionするlayer index (0-based)
        add_bos: BOS tokenオフセット
        device: torch device

    Returns:
        dict with 'last_hidden_state', 'attention_mask', 'position_ids'
        or None on failure
    """
    encoder = tts_model.encoder_module

    # 元テキストのtoken range取得
    target_text, _ = normalize_text_with_lang(kanji_text, "ja")
    kanji_range = find_entity_token_range(
        text_tokenizer, target_text, surface, add_bos=add_bos
    )
    if kanji_range is None:
        return None

    kanji_start, kanji_end = kanji_range
    kanji_len = kanji_end - kanji_start
    db_len = db_vecs.shape[0]

    # injection hookで使うmodified_h（同一長）の準備
    # 長さが異なる場合は平均化してkanji_lenに合わせる
    if kanji_len == db_len:
        inject_vecs = db_vecs.to(device)
    elif kanji_len > 0 and db_len > 0:
        # 長さが異なる場合: db_vecsをkanji_lenにresizeして使用
        # 単純に各tokenを引き伸ばし/縮め（phase3_pipelineのfallbackと同様の考え方）
        if db_len > kanji_len:
            # kanji_lenトークン分だけ使う
            inject_vecs = db_vecs[:kanji_len].to(device)
        else:
            # 繰り返してkanji_lenに合わせる（最後のtokenを繰り返す）
            pad = db_vecs[-1:].expand(kanji_len - db_len, -1)
            inject_vecs = torch.cat([db_vecs, pad], dim=0).to(device)
    else:
        return None

    # tokenize and prepare
    tokens_raw = text_tokenizer.encode(target_text, add_special_tokens=False)
    add_eos = getattr(tts_model.args, "add_eos_to_text", 0)
    add_bos_val = getattr(tts_model.args, "add_bos_to_text", 0)
    tokens = list(tokens_raw)
    if add_eos:
        tokens.append(add_eos)
    if add_bos_val:
        tokens = [add_bos_val] + tokens

    input_ids = torch.LongTensor([tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = None
    if getattr(tts_model.args, "use_pm_rope", 1):
        x_lens = torch.LongTensor([len(tokens)]).to(device)
        position_ids = tts_model._build_position_ids(x_lens, len(tokens), device)

    # forward hook: target_layerの出力をentity位置で置換
    replacement_triggered = [False]

    def replacement_hook(module, inp, output):
        if replacement_triggered[0]:
            return output
        replacement_triggered[0] = True
        if isinstance(output, tuple):
            hidden = output[0].clone()
        else:
            hidden = output.clone()
        # entity位置を置換（同一長を保証済み）
        n = inject_vecs.shape[0]
        hidden[0, kanji_start:kanji_start + n] = inject_vecs.to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = encoder.layers[target_layer].register_forward_hook(replacement_hook)
    try:
        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        final_hidden = encoder_outputs.last_hidden_state
    finally:
        handle.remove()

    return {
        "last_hidden_state": final_hidden,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def run_layer_sweep(
    tts_model, text_tokenizer, audio_tokenizer,
    db_data, val_samples, whisper_model, whisper_processor,
    device, layers, num_entities, max_contexts_per_entity, seeds,
    output_dir,
):
    """Layer sweepのメイン評価ループ。

    Args:
        layers: 評価するlayer indexのリスト (e.g. [0, 1, ..., 11])
        num_entities: 評価するentity数 (最初のN entities)
        max_contexts_per_entity: 各entityの最大コンテキスト数
        seeds: 評価シードのリスト

    Returns:
        results dict (JSON-serializable)
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "layer_sweep_checkpoint.json")

    db = db_data["db"]
    add_bos = db_data.get("add_bos", 0)
    codec_audio_sr = audio_tokenizer.sample_rate
    codec_sr = getattr(tts_model.args, "encodec_sr", 50)

    # val_samplesをentity_idでグループ化
    samples_by_entity = {}
    for s in val_samples:
        for eid in s.get("entity_ids", []):
            if eid not in samples_by_entity:
                samples_by_entity[eid] = []
            samples_by_entity[eid].append(s["text"])

    # 評価するentitiesを選択（DBにある最初のnum_entities件）
    db_eids = sorted(db.keys())
    eval_eids = [e for e in db_eids if e in samples_by_entity][:num_entities]
    print(f"\n{'='*70}")
    print(f"Evaluating {len(eval_eids)} entities × {len(layers)} layers × "
          f"{max_contexts_per_entity} ctx × {len(seeds)} seeds")
    total_synth = len(eval_eids) * len(layers) * max_contexts_per_entity * len(seeds)
    total_synth_with_bl = total_synth + len(eval_eids) * max_contexts_per_entity * len(seeds)
    print(f"Total synthesis calls: ~{total_synth_with_bl} (layer sweep + baseline)")
    print(f"Estimated time @ 10s/call: {total_synth_with_bl * 10 / 3600:.1f}h")
    print(f"{'='*70}\n")

    # checkpointからの再開
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            cp = json.load(f)
        print(f"[RESUME] Loaded checkpoint: {cp.get('completed_keys', []).__len__()} done")
        completed_keys = set(cp.get("completed_keys", []))
        raw_results = cp.get("raw_results", {})
    else:
        completed_keys = set()
        raw_results = {}  # {layer_key: {eid: {ctx_idx: {seed: {sv2, lenient}}}}}

    # layer-specific DB vectors を事前に全layerぶん抽出
    # （各layerで再抽出すると時間がかかるので一括で）
    # eval対象のentityのみDB vector抽出（45 entities全体ではなくeval_eidsのみ）
    entity_filter = set(eval_eids)
    print(f"Extracting layer-specific DB vectors for {len(eval_eids)} entities × {len(layers)} layers...")
    t_extract = time.time()
    layer_dbs = {}
    for layer in layers:
        layer_key = f"layer_{layer}"
        print(f"  Extracting layer {layer}...")
        layer_dbs[layer_key] = extract_layer_db_vectors(
            tts_model, text_tokenizer, db_data, layer, device,
            entity_filter=entity_filter
        )
    print(f"  Done in {time.time()-t_extract:.1f}s\n")

    total_evals = 0
    skipped = 0

    for eid in eval_eids:
        entry = db[eid]
        surface = entry["surface"]
        reading_katakana = entry["reading_katakana"]
        contexts = samples_by_entity.get(eid, [])[:max_contexts_per_entity]

        print(f"\n{'='*60}")
        print(f"Entity: {surface} ({reading_katakana}) | {len(contexts)} ctx")

        for ctx_idx, ctx_text in enumerate(contexts):
            for seed in seeds:
                # Baseline評価
                bl_key = f"baseline_{eid}_{ctx_idx}_{seed}"
                if bl_key not in completed_keys:
                    seed_everything(seed)
                    enc = get_encoder_output(tts_model, text_tokenizer, ctx_text, device)
                    bl_audio = generate_with_modified_encoder(
                        tts_model, text_tokenizer, audio_tokenizer,
                        enc["last_hidden_state"], enc["attention_mask"],
                        enc.get("position_ids"),
                        ctx_text, device, codec_audio_sr, codec_sr,
                    )
                    if bl_audio is not None:
                        bl_wav = bl_audio[0].cpu()
                        transcript_bl = transcribe_in_memory(
                            bl_wav, codec_audio_sr, whisper_model, whisper_processor, device
                        )
                        sv2_bl, _ = check_strict_v2(transcript_bl, reading_katakana)
                        lenient_bl, _ = check_pronunciation_lenient(transcript_bl, reading_katakana)
                    else:
                        transcript_bl = ""
                        sv2_bl = False
                        lenient_bl = False

                    if "baseline" not in raw_results:
                        raw_results["baseline"] = {}
                    if eid not in raw_results["baseline"]:
                        raw_results["baseline"][eid] = {}
                    if str(ctx_idx) not in raw_results["baseline"][eid]:
                        raw_results["baseline"][eid][str(ctx_idx)] = {}
                    raw_results["baseline"][eid][str(ctx_idx)][str(seed)] = {
                        "sv2": sv2_bl,
                        "lenient": lenient_bl,
                        "transcript": transcript_bl,
                    }
                    completed_keys.add(bl_key)
                    total_evals += 1
                    print(f"  [BL] {surface} ctx={ctx_idx} seed={seed}: "
                          f"sv2={sv2_bl} lenient={lenient_bl} '{transcript_bl[:30]}'")

                # Layer別injection評価
                for layer in layers:
                    layer_key = f"layer_{layer}"
                    eval_key = f"{layer_key}_{eid}_{ctx_idx}_{seed}"
                    if eval_key in completed_keys:
                        continue

                    layer_db = layer_dbs.get(layer_key, {})
                    if eid not in layer_db:
                        print(f"  [SKIP] layer={layer} eid={eid}: no layer DB vectors")
                        skipped += 1
                        completed_keys.add(eval_key)
                        continue

                    db_vecs = layer_db[eid]

                    seed_everything(seed)
                    enc_result = run_injection_at_layer(
                        tts_model, text_tokenizer, ctx_text, surface,
                        db_vecs, layer, add_bos, device
                    )
                    if enc_result is None:
                        print(f"  [SKIP] layer={layer} eid={eid}: injection failed (token range)")
                        skipped += 1
                        completed_keys.add(eval_key)
                        continue

                    gen_audio = generate_with_modified_encoder(
                        tts_model, text_tokenizer, audio_tokenizer,
                        enc_result["last_hidden_state"],
                        enc_result["attention_mask"],
                        enc_result["position_ids"],
                        ctx_text, device, codec_audio_sr, codec_sr,
                    )
                    if gen_audio is not None:
                        wav = gen_audio[0].cpu()
                        transcript = transcribe_in_memory(
                            wav, codec_audio_sr, whisper_model, whisper_processor, device
                        )
                        sv2, _ = check_strict_v2(transcript, reading_katakana)
                        lenient, _ = check_pronunciation_lenient(transcript, reading_katakana)
                    else:
                        transcript = ""
                        sv2 = False
                        lenient = False

                    if layer_key not in raw_results:
                        raw_results[layer_key] = {}
                    if eid not in raw_results[layer_key]:
                        raw_results[layer_key][eid] = {}
                    if str(ctx_idx) not in raw_results[layer_key][eid]:
                        raw_results[layer_key][eid][str(ctx_idx)] = {}
                    raw_results[layer_key][eid][str(ctx_idx)][str(seed)] = {
                        "sv2": sv2,
                        "lenient": lenient,
                        "transcript": transcript,
                    }
                    completed_keys.add(eval_key)
                    total_evals += 1
                    print(f"  [L{layer:2d}] {surface} ctx={ctx_idx} seed={seed}: "
                          f"sv2={sv2} lenient={lenient} '{transcript[:30]}'")

                    # 50 eval毎にチェックポイント保存
                    if total_evals % 50 == 0:
                        _save_checkpoint(checkpoint_path, completed_keys, raw_results)
                        print(f"  [CHECKPOINT] {total_evals} evals done")

    # 最終チェックポイント
    _save_checkpoint(checkpoint_path, completed_keys, raw_results)

    # 集計
    results = _aggregate_results(raw_results, layers, eval_eids, db)

    print(f"\n[Summary] Total evals: {total_evals}, Skipped: {skipped}")
    return results


def _save_checkpoint(path, completed_keys, raw_results):
    """チェックポイント保存。"""
    cp = {
        "completed_keys": list(completed_keys),
        "raw_results": raw_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


def _aggregate_results(raw_results, layers, eval_eids, db):
    """raw_resultsから集計結果を生成する。

    Returns:
        {
          "layers": [0, 1, ..., 11],
          "results": {
            "layer_0": {"strict_v2": float, "lenient": float, "n": int, "per_entity": {...}},
            ...
            "layer_11": ...,
          },
          "baseline": {"strict_v2": float, "lenient": float, "n": int},
        }
    """
    def _agg(condition_key):
        sv2_total = 0
        lenient_total = 0
        n_total = 0
        per_entity = {}

        cond_data = raw_results.get(condition_key, {})
        for eid in eval_eids:
            if eid not in cond_data:
                continue
            entity_sv2 = []
            entity_lenient = []
            for ctx_idx_str, seed_results in cond_data[eid].items():
                for seed_str, r in seed_results.items():
                    if r.get("sv2") is not None:
                        entity_sv2.append(r["sv2"])
                        entity_lenient.append(r["lenient"])
                        sv2_total += int(r["sv2"])
                        lenient_total += int(r["lenient"])
                        n_total += 1
            surface = db[eid]["surface"] if eid in db else eid
            per_entity[eid] = {
                "surface": surface,
                "n": len(entity_sv2),
                "strict_v2": sum(entity_sv2) / max(len(entity_sv2), 1),
                "lenient": sum(entity_lenient) / max(len(entity_lenient), 1),
            }

        return {
            "strict_v2": sv2_total / max(n_total, 1),
            "lenient": lenient_total / max(n_total, 1),
            "n": n_total,
            "per_entity": per_entity,
        }

    agg_results = {}
    for layer in layers:
        layer_key = f"layer_{layer}"
        agg_results[layer_key] = _agg(layer_key)

    baseline_agg = _agg("baseline")

    return {
        "layers": layers,
        "results": agg_results,
        "baseline": baseline_agg,
    }


def print_summary(results):
    """集計結果を表示する。"""
    print(f"\n{'='*70}")
    print("LAYER SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Condition':<15} {'Strict v2':>12} {'Lenient':>10} {'N':>6}")
    print("-" * 50)

    # Baseline
    bl = results["baseline"]
    print(f"{'baseline':<15} {bl['strict_v2']:>11.1%} {bl['lenient']:>10.1%} {bl['n']:>6}")

    # Layer results
    for layer in results["layers"]:
        layer_key = f"layer_{layer}"
        r = results["results"].get(layer_key, {})
        sv2 = r.get("strict_v2", 0.0)
        lenient = r.get("lenient", 0.0)
        n = r.get("n", 0)
        marker = " <-- Layer 9" if layer == 9 else ""
        print(f"  Layer {layer:2d}      {sv2:>11.1%} {lenient:>10.1%} {n:>6}{marker}")

    print(f"\n{'='*70}")
    print("BEST LAYER:")
    best_layer = max(
        results["layers"],
        key=lambda l: results["results"].get(f"layer_{l}", {}).get("strict_v2", 0.0)
    )
    best_r = results["results"].get(f"layer_{best_layer}", {})
    print(f"  Layer {best_layer}: Strict v2={best_r.get('strict_v2', 0):.1%}, "
          f"Lenient={best_r.get('lenient', 0):.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Layer Sweep Ablation: なぜLayer 9かを実証する実験"
    )
    parser.add_argument(
        "--db", default="checkpoints/pronunciation_vector_db_expanded.pt",
        help="Path to pronunciation vector DB"
    )
    parser.add_argument(
        "--val_data", default="data/val_lora_clean_eval.jsonl",
        help="Clean eval JSONL"
    )
    parser.add_argument(
        "--output_dir", default="eval_results/layer_sweep",
        help="Output directory"
    )
    parser.add_argument(
        "--layers", default="0,1,2,3,4,5,6,7,8,9,10,11",
        help="Comma-separated layer indices to sweep"
    )
    parser.add_argument(
        "--num_entities", type=int, default=15,
        help="Number of entities to evaluate (first N from DB)"
    )
    parser.add_argument(
        "--max_contexts", type=int, default=3,
        help="Max contexts per entity"
    )
    parser.add_argument(
        "--seeds", default="42",
        help="Comma-separated random seeds"
    )
    parser.add_argument(
        "--model_dir", default="Aratako/T5Gemma-TTS-2b-2b",
        help="TTS model directory"
    )
    args = parser.parse_args()

    # プロジェクトルートに移動
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(project_root)
    print(f"[Info] Working directory: {project_root}")

    layers = [int(l) for l in args.layers.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    # 空きGPUを自動選択（GPU 0が別プロセスで占有されている場合を考慮）
    if torch.cuda.is_available():
        best_gpu = 0
        max_free = 0
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"[Info] GPU {i}: free={free//1024**3}GB / total={total//1024**3}GB")
            if free > max_free:
                max_free = free
                best_gpu = i
        torch.cuda.set_device(best_gpu)
        device = f"cuda:{best_gpu}"
        print(f"[Info] Selected GPU {best_gpu} ({max_free//1024**3}GB free)")
    else:
        device = "cpu"
    print(f"[Info] Device: {device}")
    print(f"[Info] Layers: {layers}")
    print(f"[Info] Num entities: {args.num_entities}")
    print(f"[Info] Max contexts per entity: {args.max_contexts}")
    print(f"[Info] Seeds: {seeds}")

    # DB読み込み
    print(f"\nLoading DB from {args.db}...")
    db_data = torch.load(args.db, weights_only=False)
    print(f"  DB: {len(db_data['db'])} entities, add_bos={db_data.get('add_bos', 0)}")

    # TTS model読み込み
    print(f"\nLoading TTS model from {args.model_dir}...")
    t_load = time.time()
    tts_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    )
    tts_model.eval()
    cfg = tts_model.config
    tokenizer_name = (
        getattr(cfg, "text_tokenizer_name", None)
        or getattr(cfg, "t5gemma_model_name", None)
    )
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", None),
    )
    print(f"  Model loaded in {time.time()-t_load:.1f}s")

    # Encoderのlayer数確認
    num_encoder_layers = len(tts_model.encoder_module.layers)
    print(f"  Encoder layers: {num_encoder_layers}")
    # layersのバリデーション
    valid_layers = [l for l in layers if 0 <= l < num_encoder_layers]
    if len(valid_layers) != len(layers):
        invalid = [l for l in layers if l not in valid_layers]
        print(f"  [WARN] Invalid layers (ignored): {invalid}")
        layers = valid_layers

    # Whisper読み込み
    print("\nLoading Whisper...")
    whisper_model, whisper_processor = load_whisper(device)

    # val data読み込み
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Val samples: {len(val_samples)}")

    # 評価実行
    t_start = time.time()
    results = run_layer_sweep(
        tts_model=tts_model,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        db_data=db_data,
        val_samples=val_samples,
        whisper_model=whisper_model,
        whisper_processor=whisper_processor,
        device=device,
        layers=layers,
        num_entities=args.num_entities,
        max_contexts_per_entity=args.max_contexts,
        seeds=seeds,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed/3600:.2f}h")

    # サマリー表示
    print_summary(results)

    # 結果保存
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "layer_sweep_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
