"""ROME / MEMIT on JA cross-culture pairs — text_a only, all 300 pair_idx.

For unified baselines comparison with pair_idx 0..299 from
`data/ja_cross_culture_names.build_300_pairs()`. For each (word_a, word_b)
group (30 total), apply ONE ROME edit (L258 MEMIT), then synthesize
each of 10 text_a contexts → 300 wavs total. Outputs:

  {output_dir}/p{pair_idx:03d}_text_a_s{seed}.wav
  {output_dir}/rome_transcripts.json
"""
import argparse, json, os, sys, gc, time, unicodedata
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# torchao compat shim (must be BEFORE any transformers/xcodec2 imports that
# might transitively load torchao -- torchao expects torch>=2.11's
# torch.utils._pytree.register_constant, which is missing in torch 2.6).
import torch as _torch_pre
import torch.utils._pytree as _pt
if not hasattr(_pt, "register_constant"):
    _pt.register_constant = lambda *a, **kw: None

T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)

import numpy as np, torch
from data.tokenizer import AudioTokenizer
from inference_tts_utils import normalize_text_with_lang, save_audio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, "/mnt/scripts")
from encoder_injection_t5gemma import (
    get_encoder_output, generate_with_modified_encoder,
    find_entity_token_range, seed_everything,
)
from rome_encoder_t5gemma import (
    capture_down_proj_input, compute_down_proj_input_covariance,
    compute_rome_delta_W, compute_memit_delta_W, load_whisper,
    transcribe_with_whisper,
)

sys.path.insert(0, "/mnt/scripts/projection")
sys.path.insert(0, "/mnt/data")
from ja_cross_culture_names import build_300_pairs


def normalize_ja(text):
    return unicodedata.normalize("NFC", text).replace(" ", "").replace("\u3000", "")


def apply_rome_edit_jacc(model, text_tokenizer, layer_idx, target_entities,
                          corpus_texts, device, covariance_cache,
                          edit_mode="memit", eps=1.0):
    encoder = model.encoder_module
    if layer_idx in covariance_cache:
        C = covariance_cache[layer_idx]
    else:
        C = compute_down_proj_input_covariance(
            encoder, text_tokenizer, corpus_texts, layer_idx, device,
            normalize_func=normalize_text_with_lang,
        )
        covariance_cache[layer_idx] = C
    k_stars, v_stars, v_currents = [], [], []
    for ent in target_entities:
        word_a = ent["word_a"]; word_b = ent["word_b"]
        ref_text_norm,  _ = normalize_text_with_lang(ent["text_ref"], "ja")
        test_text_norm, _ = normalize_text_with_lang(ent["text_test"], "ja")
        ref_tokens  = text_tokenizer.encode(ref_text_norm,  add_special_tokens=False)
        test_tokens = text_tokenizer.encode(test_text_norm, add_special_tokens=False)
        if not ref_tokens or not test_tokens: continue
        ref_ids  = torch.LongTensor([ref_tokens]).to(device)
        test_ids = torch.LongTensor([test_tokens]).to(device)
        ref_range  = find_entity_token_range(text_tokenizer, ref_text_norm,  word_b, add_bos=0)
        test_range = find_entity_token_range(text_tokenizer, test_text_norm, word_a, add_bos=0)
        if ref_range is None or test_range is None: continue
        test_dp_input, test_mlp_out = capture_down_proj_input(
            encoder, test_ids, torch.ones_like(test_ids), layer_idx,
        )
        _, ref_mlp_out = capture_down_proj_input(
            encoder, ref_ids, torch.ones_like(ref_ids), layer_idx,
        )
        if test_dp_input is None or test_mlp_out is None or ref_mlp_out is None: continue
        ti = min(test_range[1] - 1, test_dp_input.shape[1] - 1, test_mlp_out.shape[1] - 1)
        ri = min(ref_range[1]  - 1, ref_mlp_out.shape[1]  - 1)
        if ti < 0 or ri < 0: continue
        k_stars.append(test_dp_input[0, ti].float())
        v_currents.append(test_mlp_out[0, ti].float())
        v_stars.append(ref_mlp_out[0, ri].float())
    if not k_stars: return None
    if edit_mode == "memit" and len(k_stars) > 1:
        delta_W = compute_memit_delta_W(k_stars, v_stars, v_currents, C, eps=eps)
    else:
        delta_W = None
        for k, vs, vc in zip(k_stars, v_stars, v_currents):
            dW = compute_rome_delta_W(k, vs, vc, C, eps=eps)
            if dW is not None: delta_W = dW if delta_W is None else delta_W + dW
    if delta_W is None: return None
    max_abs = delta_W.abs().max().item()
    if max_abs > 1e6: delta_W = delta_W.clamp(-1e4, 1e4)
    down_proj = encoder.layers[layer_idx].mlp.down_proj
    with torch.no_grad():
        down_proj.weight.add_(delta_W.to(dtype=down_proj.weight.dtype,
                                          device=down_proj.weight.device))
    print(f"  Applied L{layer_idx} |dW|={delta_W.norm().item():.4f}", flush=True)
    return delta_W


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8])
    p.add_argument("--edit_mode", default="memit")
    p.add_argument("--label", default="L258_memit")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = build_300_pairs()
    # Build unique (word_a, word_b) groups for ROME edit
    by_pair = defaultdict(lambda: {"a_ctx": [], "b_ctx": [], "word_a": None, "word_b": None,
                                    "pair_idx_list": []})
    for p_ in pairs:
        key = (p_["word_a"], p_["word_b"])
        by_pair[key]["a_ctx"].append(p_["text_a"])
        by_pair[key]["b_ctx"].append(p_["text_b"])
        by_pair[key]["word_a"] = p_["word_a"]
        by_pair[key]["word_b"] = p_["word_b"]
        by_pair[key]["pair_idx_list"].append(p_["pair_idx"])
    groups = list(by_pair.values())
    print(f"[Info] {len(groups)} unique (word_a, word_b) groups", flush=True)
    target_entities = [{"word_a": g["word_a"], "word_b": g["word_b"],
                        "text_ref":  g["b_ctx"][0], "text_test": g["a_ctx"][0]}
                       for g in groups]
    corpus_texts = [c for g in groups for c in g["a_ctx"] + g["b_ctx"]]

    print("[Info] Loading T5Gemma-TTS...", flush=True)
    import logging as _l; _l.getLogger("transformers").setLevel(_l.WARNING)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Aratako/T5Gemma-TTS-2b-2b", trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto")
    model.eval(); cfg = model.config
    tn = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tn)
    audio_tokenizer = AudioTokenizer(backend="xcodec2",
                                      model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tokenizer.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    whisper_model, whisper_processor = load_whisper(device)

    # Apply ROME edit
    print(f"\n[ROME] Applying edit: layers={args.layers}, mode={args.edit_mode}", flush=True)
    covariance_cache = {}
    saved = {}
    for li in args.layers:
        encoder = model.encoder_module
        saved[li] = encoder.layers[li].mlp.down_proj.weight.detach().clone()
        apply_rome_edit_jacc(
            model, text_tokenizer, li, target_entities, corpus_texts, device,
            covariance_cache, edit_mode=args.edit_mode, eps=1.0,
        )

    # Synthesize text_a for each pair_idx 0..299
    results = []
    pairs_sorted = sorted(pairs, key=lambda p: p["pair_idx"])
    total = len(pairs_sorted)
    t_start = time.time()
    for i, p_ in enumerate(pairs_sorted):
        seed_everything(args.seed)
        item = {"pair_idx": p_["pair_idx"], "word_a": p_["word_a"], "word_b": p_["word_b"],
                "text": p_["text_a"], "seed": args.seed}
        try:
            enc = get_encoder_output(model, text_tokenizer, p_["text_a"], device)
            gen = generate_with_modified_encoder(
                model, text_tokenizer, audio_tokenizer,
                enc["last_hidden_state"], enc["attention_mask"], enc.get("position_ids"),
                p_["text_a"], device, codec_audio_sr, codec_sr,
            )
            if gen is None:
                results.append({**item, "rome_transcript": "", "status": "error"}); continue
            wav_path = os.path.join(args.output_dir, f"p{p_['pair_idx']:03d}_text_a_s{args.seed}.wav")
            save_audio(wav_path, gen[0].cpu(), codec_audio_sr)
            transcript = transcribe_with_whisper(wav_path, whisper_model, whisper_processor, device)
            tn = normalize_ja(transcript)
            results.append({**item, "wav_path": wav_path,
                            "rome_transcript": transcript,
                            "source_in": p_["word_a"] in tn,
                            "target_in": p_["word_b"] in tn,
                            "status": "success"})
            del gen
        except Exception as e:
            results.append({**item, "rome_transcript": "", "status": "error", "error": str(e)[:200]})
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    # Restore weights (clean exit)
    for li, w in saved.items():
        model.encoder_module.layers[li].mlp.down_proj.weight.data.copy_(w)

    out_json = os.path.join(args.output_dir, "rome_transcripts.json")
    summary_n = sum(1 for r in results if r.get("target_in"))
    summary = {"label": args.label, "layers": args.layers, "edit_mode": args.edit_mode,
               "seed": args.seed, "n_total": len(results),
               "n_success": sum(1 for r in results if r.get("status") == "success"),
               "target_in_rate": summary_n / max(1, len(results))}
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] {summary['n_total']} synthesized, target_in={summary['target_in_rate']:.4f}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
