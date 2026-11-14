"""G2P Baseline on JA cross-culture pairs N=300.

For each pair (pair_idx, word_a, word_b, text_a) from
`build_300_pairs()`:
  1. G2P-predict word_a -> katakana via pykakasi (e.g. a Chinese surname is
     transliterated to its on-yomi katakana form)
  2. Replace word_a in text_a with the katakana prediction (modified_text)
  3. Synthesize modified_text with stock T5Gemma-TTS (no editing/injection)
  4. Transcribe with Whisper

This is the standard "G2P Baseline" control: the model receives the
G2P-predicted reading of word_a, NOT the desired word_b reading.
Output should still NOT contain word_b — confirming G2P is not a
mechanism for cross-cultural pronunciation transfer.

Naming: {output_dir}/p{pair_idx:03d}_g2p_s{seed}.wav
"""
import argparse, json, os, sys, gc, time, unicodedata

# --- Anonymous release path setup ---
import os as _os, sys as _sys
_REPO_ROOT_FOR_IMPORTS = _os.environ.get('REPO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_REPO_ROOT_FOR_IMPORTS, 'src'))
_sys.path.insert(0, _os.path.join(_REPO_ROOT_FOR_IMPORTS, 'data'))
# ---

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# torchao compat shim (before transformers / xcodec2)
import torch as _torch_pre
import torch.utils._pytree as _pt
if not hasattr(_pt, "register_constant"):
    _pt.register_constant = lambda *a, **kw: None

T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", "")
sys.path.insert(0, T5GEMMA_ROOT)

import numpy as np, torch
from data.tokenizer import AudioTokenizer
from inference_tts_utils import save_audio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, "/mnt/scripts")
from injection import (
    get_encoder_output, generate_with_modified_encoder, seed_everything,
)
from rome import load_whisper, transcribe_with_whisper

sys.path.insert(0, "/mnt/data")
from ja_cross_culture_names import build_300_pairs

import pykakasi
kks = pykakasi.kakasi()


def g2p_katakana(text):
    """Pykakasi-based katakana G2P prediction."""
    return "".join(r["kana"] for r in kks.convert(text))


def normalize_ja(text):
    return unicodedata.normalize("NFC", text).replace(" ", "").replace("\u3000", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = build_300_pairs()
    print(f"[Info] {len(pairs)} cross-culture pairs", flush=True)

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

    results = []
    pairs_sorted = sorted(pairs, key=lambda p: p["pair_idx"])
    total = len(pairs_sorted)
    t_start = time.time()
    for i, p_ in enumerate(pairs_sorted):
        seed_everything(args.seed)
        word_a = p_["word_a"]
        word_b = p_["word_b"]
        text_a = p_["text_a"]
        # G2P-predict word_a → katakana
        g2p_pred = g2p_katakana(word_a)
        # Replace in text
        modified_text = text_a.replace(word_a, g2p_pred, 1)
        item = {"pair_idx": p_["pair_idx"], "word_a": word_a, "word_b": word_b,
                "text_a": text_a, "g2p_pred_for_word_a": g2p_pred,
                "modified_text": modified_text, "seed": args.seed}
        try:
            enc = get_encoder_output(model, text_tokenizer, modified_text, device)
            gen = generate_with_modified_encoder(
                model, text_tokenizer, audio_tokenizer,
                enc["last_hidden_state"], enc["attention_mask"], enc.get("position_ids"),
                modified_text, device, codec_audio_sr, codec_sr,
            )
            if gen is None:
                results.append({**item, "g2p_transcript": "", "status": "error"}); continue
            wav_path = os.path.join(args.output_dir, f"p{p_['pair_idx']:03d}_g2p_s{args.seed}.wav")
            save_audio(wav_path, gen[0].cpu(), codec_audio_sr)
            transcript = transcribe_with_whisper(wav_path, whisper_model, whisper_processor, device)
            tn = normalize_ja(transcript)
            results.append({**item, "wav_path": wav_path,
                            "g2p_transcript": transcript,
                            "source_in": word_a in tn,
                            "target_in": word_b in tn,
                            "g2p_pred_in": g2p_pred in normalize_ja(transcript),
                            "status": "success"})
            del gen
        except Exception as e:
            results.append({**item, "g2p_transcript": "", "status": "error", "error": str(e)[:200]})
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    out_json = os.path.join(args.output_dir, "g2p_transcripts.json")
    summary = {"seed": args.seed, "n_total": len(results),
               "n_success": sum(1 for r in results if r.get("status") == "success"),
               "target_in_rate": sum(1 for r in results if r.get("target_in")) / max(1, len(results)),
               "g2p_pred_in_rate": sum(1 for r in results if r.get("g2p_pred_in")) / max(1, len(results))}
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] N={summary['n_total']} target_in={summary['target_in_rate']:.4f} "
          f"g2p_pred_in={summary['g2p_pred_in_rate']:.4f}")


if __name__ == "__main__":
    main()
