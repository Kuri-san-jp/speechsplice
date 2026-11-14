#!/usr/bin/env python3
"""VoiceCraft speech-editing baseline on EN cross-culture entity pairs.

Rebuttal ① (Jx95/Mxy8): compare SpeechSplice against a trained speech-editing
model on the entity-replacement task (VoiceCraft's native use case).

For each sampled pair:
  1. BL audio  = T5Gemma baseline synthesis of text_a (word_a)
     from eval_results/_cc_wavs_a/t5gemma_en/s{i:04d}_bl.wav
  2. Word span = whisper word timestamps of word_a in BL audio (MFA substitute)
  3. VoiceCraft edits the span with target_text = text_a.replace(word_a, word_b)
  4. Save edited wav + whisper transcript of the result
Scoring (UTMOS / WavLM-SV) runs post-hoc in the main image.

Usage:
  IMAGE_NAME=voicecraft-image SLURM_TIME=03:00:00 \
    python scripts/voicecraft/edit_entity_en.py
"""
import os, sys, json, random, re

os.environ["HF_HOME"] = "/mnt/.cache/huggingface"
sys.path.insert(0, "/opt/VoiceCraft")
sys.path.insert(0, "/mnt/data")

import numpy as np
import torch
import torchaudio

N_SELECT = 15
SEED = 2026
LEFT_MARGIN = 0.08
RIGHT_MARGIN = 0.08
CODEC_SR = 50
DECODE_CONFIG = {
    "top_k": -1, "top_p": 0.8, "temperature": 1.0,
    "stop_repetition": 2, "kvcache": 1,
    "codec_sr": CODEC_SR, "silence_tokens": [1388, 1898, 131],
}
OUT_DIR = "/mnt/eval_results/voicecraft_edit_en"
WAV_ROOT = "/mnt/eval_results/_cc_wavs_a/t5gemma_en"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    from en_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()
    idxs = sorted(random.sample(range(len(pairs)), N_SELECT))
    print(f"[Info] selected pair indices: {idxs}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import whisper as openai_whisper
    # /tmp/.cache can be a root-owned mount and unwritable; use the project cache instead
    wmodel = openai_whisper.load_model("medium.en", device=device,
                                       download_root="/mnt/.cache/whisper")

    from models import voicecraft
    from data.tokenizer import AudioTokenizer, TextTokenizer
    from inference_speech_editing_scale import inference_one_sample
    from huggingface_hub import hf_hub_download

    model = voicecraft.VoiceCraft.from_pretrained("pyp1/VoiceCraft_830M_TTSEnhanced")
    phn2num = model.args.phn2num
    model_args = model.args
    model.to(device).eval()

    enc_path = hf_hub_download(repo_id="pyp1/VoiceCraft", filename="encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=enc_path)
    text_tokenizer = TextTokenizer(backend="espeak")
    print("[Info] models loaded", flush=True)

    def find_word_span(audio_fn, word):
        """Locate the time span of word_a from Whisper word timestamps (stand-in for MFA)."""
        res = wmodel.transcribe(audio_fn, word_timestamps=True, language="en")
        target = re.sub(r"\W", "", word).lower()
        for seg in res["segments"]:
            for w in seg.get("words", []):
                if re.sub(r"\W", "", w["word"]).lower() == target:
                    return w["start"], w["end"], res["text"].strip()
        return None, None, res["text"].strip()

    results = []
    for i in idxs:
        p = pairs[i]
        audio_fn = f"{WAV_ROOT}/s{i:04d}_bl.wav"
        if not os.path.exists(audio_fn):
            print(f"[{i}] missing wav, skip", flush=True)
            continue
        t0, t1, bl_transcript = find_word_span(audio_fn, p["word_a"])
        row = {"pair_idx": i, "word_a": p["word_a"], "word_b": p["word_b"],
               "text_a": p["text_a"], "bl_transcript": bl_transcript,
               "span": [t0, t1]}
        if t0 is None:
            row["status"] = "align_fail"
            results.append(row)
            print(f"[{i}] align_fail ({p['word_a']}) BL='{bl_transcript[:50]}'", flush=True)
            continue

        info = torchaudio.info(audio_fn)
        dur = info.num_frames / info.sample_rate
        start = max(t0 - LEFT_MARGIN, 1 / CODEC_SR)
        end = min(t1 + RIGHT_MARGIN, dur - 1 / CODEC_SR)
        mask_interval = torch.LongTensor([[round(start * CODEC_SR), round(end * CODEC_SR)]])

        target_text = p["text_a"].replace(p["word_a"], p["word_b"])
        try:
            with torch.no_grad():
                orig, gen = inference_one_sample(
                    model, model_args, phn2num, text_tokenizer, audio_tokenizer,
                    audio_fn, target_text, mask_interval, device, DECODE_CONFIG)
            gen = gen[0].cpu()
            out_fn = f"{OUT_DIR}/s{i:04d}_edit.wav"
            torchaudio.save(out_fn, gen, 16000)
            tr = wmodel.transcribe(out_fn, language="en")["text"].strip()
            row["edit_transcript"] = tr
            row["target_in_edit"] = re.sub(r"\W", "", p["word_b"]).lower() in re.sub(r"\W", "", tr).lower()
            row["status"] = "success"
            print(f"[{i}] {p['word_a']}→{p['word_b']}  target_in_edit={row['target_in_edit']}  '{tr[:60]}'", flush=True)
        except Exception as e:
            row["status"] = f"error: {str(e)[:100]}"
            print(f"[{i}] ERROR {str(e)[:100]}", flush=True)
        results.append(row)

        with open(f"{OUT_DIR}/results.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

    succ = [r for r in results if r.get("status") == "success"]
    hit = sum(1 for r in succ if r.get("target_in_edit"))
    print(f"\n[Done] success={len(succ)}/{len(results)}, target_in_edit={hit}/{len(succ)}", flush=True)


if __name__ == "__main__":
    main()
