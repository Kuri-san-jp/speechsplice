#!/usr/bin/env python3
"""VoiceCraft zero-shot TTS baseline: SpeechSplice と同一話者プロンプトで全文生成 (N=100).

SpeechSplice 新世代 (en_cc100_pairs_gen) と同じ参照音声
eval_results/_natural_refs_norm/en_natural_norm.wav（LibriTTS spkr 260）を
voice prompt にして、crossculture 100ペアの text_a (BL) / text_b (INJ相当) を
zero-shot TTS で生成する。編集モードと違い区間外波形コピーがないため、
UTMOS・話者類似度の比較が全文生成同士で公平になる。

出力: eval_results/voicecraft_tts_en_n100/s{i:04d}_{bl,inj}.wav (16kHz, 生成部のみ)
      + results.json（whisper medium.en 転写、target語含有、duration）

Usage:
  IMAGE_NAME=voicecraft-image SLURM_TIME=05:00:00 \
    slurm_run.sh python scripts/voicecraft/tts_entity_en_n100.py
"""
import argparse
import json
import os
import re
import sys

os.environ["HF_HOME"] = "/mnt/.cache/huggingface"
sys.path.insert(0, "/opt/VoiceCraft")
sys.path.insert(0, "/mnt/data")

import torch
import torchaudio

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=100)
_ap.add_argument("--out_dir", default="/mnt/eval_results/voicecraft_tts_en_n100")
_ap.add_argument("--prompt_wav", default="/mnt/eval_results/_natural_refs_norm/en_natural_norm.wav")
_ap.add_argument("--pairs_module", default=None,
                 help="data/<name>.py の build_pairs() からペアを読む（既定: crossculture 300）")
_ap.add_argument("--sample_batch_size", type=int, default=4,
                 help="repo TTS demo 既定値4（候補から最短を採用）")
_cli = _ap.parse_args()

CODEC_SR = 50
DECODE_CONFIG = {
    # VoiceCraft リポジトリ TTS デモの既定値
    "top_k": 0, "top_p": 0.9, "temperature": 1.0,
    "stop_repetition": 3, "kvcache": 1,
    "codec_sr": CODEC_SR, "codec_audio_sr": 16000,
    "silence_tokens": [1388, 1898, 131],
    "sample_batch_size": _cli.sample_batch_size,
}


def main():
    os.makedirs(_cli.out_dir, exist_ok=True)
    torch.manual_seed(2026)

    if _cli.pairs_module:
        mod = __import__(_cli.pairs_module)
        pairs = mod.build_pairs()[: _cli.n]
    else:
        from en_cross_culture_names import build_300_pairs
        pairs = build_300_pairs()[: _cli.n]

    res_path = f"{_cli.out_dir}/results.json"
    results = []
    done = set()
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)
        done = {(r["pair_idx"], r["cond"]) for r in results if r.get("status") == "success"}
        print(f"[Resume] {len(done)} generations already done", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import whisper as openai_whisper
    wmodel = openai_whisper.load_model("medium.en", device=device,
                                       download_root="/mnt/.cache/whisper")

    # プロンプトを 16kHz mono に整形（VoiceCraft/EnCodec は 16kHz）
    pw, psr = torchaudio.load(_cli.prompt_wav)
    if pw.shape[0] > 1:
        pw = pw.mean(0, keepdim=True)
    if psr != 16000:
        pw = torchaudio.functional.resample(pw, psr, 16000)
    prompt16 = f"{_cli.out_dir}/_prompt16k.wav"
    torchaudio.save(prompt16, pw, 16000)
    prompt_sec = pw.shape[-1] / 16000
    prompt_end_frame = int(prompt_sec * 16000)
    prompt_transcript = wmodel.transcribe(prompt16, language="en")["text"].strip()
    print(f"[Info] prompt {prompt_sec:.2f}s transcript='{prompt_transcript}'", flush=True)

    from models import voicecraft
    from data.tokenizer import AudioTokenizer, TextTokenizer
    from inference_tts_scale import inference_one_sample
    from huggingface_hub import hf_hub_download

    model = voicecraft.VoiceCraft.from_pretrained("pyp1/VoiceCraft_830M_TTSEnhanced")
    phn2num = model.args.phn2num
    model_args = model.args
    model.to(device).eval()
    enc_path = hf_hub_download(repo_id="pyp1/VoiceCraft", filename="encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=enc_path)
    text_tokenizer = TextTokenizer(backend="espeak")
    print("[Info] models loaded", flush=True)

    def norm(s):
        return re.sub(r"\W", "", s).lower()

    for i, p in enumerate(pairs):
        for cond, text, word in (("bl", p["text_a"], p["word_a"]),
                                 ("inj", p["text_b"], p["word_b"])):
            if (i, cond) in done:
                continue
            row = {"pair_idx": i, "cond": cond, "text": text, "word": word}
            target_text = f"{prompt_transcript} {text}"
            try:
                with torch.no_grad():
                    concated, gen = inference_one_sample(
                        model, model_args, phn2num, text_tokenizer, audio_tokenizer,
                        prompt16, target_text, device, DECODE_CONFIG,
                        prompt_end_frame)
                gen = gen[0].cpu()
                out_fn = f"{_cli.out_dir}/s{i:04d}_{cond}.wav"
                torchaudio.save(out_fn, gen, 16000)
                row["dur"] = gen.shape[-1] / 16000
                tr = wmodel.transcribe(out_fn, language="en")["text"].strip()
                row["transcript"] = tr
                row["word_in"] = norm(word) in norm(tr)
                row["status"] = "success"
                print(f"[{i}:{cond}] {word}  word_in={row['word_in']}  '{tr[:60]}'", flush=True)
            except Exception as e:
                row["status"] = f"error: {str(e)[:120]}"
                print(f"[{i}:{cond}] ERROR {str(e)[:120]}", flush=True)
            results.append(row)
            with open(res_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    succ = [r for r in results if r.get("status") == "success"]
    inj = [r for r in succ if r["cond"] == "inj"]
    bl = [r for r in succ if r["cond"] == "bl"]
    print(f"\n[Done] success={len(succ)}/{len(results)}  "
          f"bl word_in={sum(r['word_in'] for r in bl)}/{len(bl)}  "
          f"inj word_in={sum(r['word_in'] for r in inj)}/{len(inj)}", flush=True)


if __name__ == "__main__":
    main()
