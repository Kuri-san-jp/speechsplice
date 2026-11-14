"""B1 (Surface Text Edit) + B7 (Text Hint) baseline evaluation.

For each (text_a, text_b, word_a, word_b) pair in cross-culture data:

  B0 (Baseline)     : synth(text_a) — faithful baseline with word_a reading
  B1 (Text Edit)    : synth(text_b) — surface-replaced text (word_b in place)
  B7 (Text Hint)    : synth(text_a + hint(word_a→word_b))
                       Hint format per language:
                         EN: "{a} (i.e., {b})"
                         JA: "{a}（{b}）"
                         ZH: "{a}（{b}）"

All three audio variants are saved and scored with:
  - Whisper transcription -> word_b in transcript? (orthographic)
  - wav2vec2-xlsr-53-espeak-cv-ft -> IPA -> entity-region CER (phonemic)

Uses the same natural-human voice-cloning reference per language as the
SpeechSplice and Qwen3 fixed cross-culture experiments, so all
architectures and baselines share an identical speaker.
"""
import argparse, json, os, sys, time, random
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

os.environ.setdefault("TORCH_HOME", "/mnt/.cache/torch")
os.environ.setdefault("HF_HOME", "/mnt/.cache/huggingface")
os.makedirs("/mnt/.cache/torch/hub", exist_ok=True)
os.makedirs("/mnt/.cache/huggingface", exist_ok=True)

for _p in ("/tmp/.cache/huggingface/token", os.path.expanduser("~/.cache/huggingface/token")):
    if os.path.exists(_p):
        try:
            with open(_p) as _f:
                os.environ.setdefault("HF_TOKEN", _f.read().strip())
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
            break
        except Exception: pass

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from encoder_injection_t5gemma import seed_everything
from inference_tts_utils import (
    inference_one_sample, save_audio,
    normalize_text_with_lang, transcribe_audio,
)
from duration_estimator import estimate_duration
from data.tokenizer import AudioTokenizer

SEED = 42

REFS = {
    "ja": (f"{PROJECT_ROOT}/eval_results/_natural_refs_norm/ja_natural_norm.wav",
           "太平洋は、世界でいちばん大きな大洋です。"),
    "en": (f"{PROJECT_ROOT}/eval_results/_natural_refs_norm/en_natural_norm.wav",
           "Let me see, I'll give them a new pair of boots every Christmas."),
    "zh": (f"{PROJECT_ROOT}/eval_results/_natural_refs_norm/zh_natural_norm.wav",
           "印度素有嫁女儿陪送嫁妆的传统。"),
}

HINT_TEMPLATES = {
    "en": lambda t, a, b: t.replace(a, f"{a} (i.e., {b})", 1),
    "ja": lambda t, a, b: t.replace(a, f"{a}（{b}）", 1),
    "zh": lambda t, a, b: t.replace(a, f"{a}（{b}）", 1),
}


def load_pairs(lang, n):
    """Load shared entity-replacement split per language."""
    if lang == "ja":
        from ja_cross_culture_names import build_300_pairs
        return build_300_pairs()[:n]
    if lang == "en":
        from en_cross_culture_names import build_300_pairs
        return build_300_pairs()[:n]
    if lang == "zh":
        from zh_cross_culture_names import build_300_pairs
        return build_300_pairs()[:n]
    raise ValueError(lang)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--phoneme_model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"eval_results/baseline_text_edit_{args.lang}_n{args.n}"
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = load_pairs(args.lang, args.n)
    print(f"[Info] {args.lang}: {len(pairs)} pairs", flush=True)

    def _local_snapshot(hub_id):
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
        repo_dir = os.path.join(hub, f"models--{hub_id.replace('/','--')}", "snapshots")
        if os.path.isdir(repo_dir):
            snaps = sorted(os.listdir(repo_dir))
            if snaps: return os.path.join(repo_dir, snaps[-1])
        return hub_id

    print("[Info] Loading T5Gemma-TTS...", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        _local_snapshot("Aratako/T5Gemma-TTS-2b-2b"),
        trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        local_files_only=False)
    model.eval()
    cfg = model.config
    tkn_hub_id = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tok = AutoTokenizer.from_pretrained(_local_snapshot(tkn_hub_id))
    audio_tok = AudioTokenizer(backend="xcodec2",
                                model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    ref_audio, ref_text_user = REFS[args.lang]
    print("[Info] Whisper-transcribing ref...", flush=True)
    ref_prefix = transcribe_audio(ref_audio, device)
    print(f"        prefix: {ref_prefix}", flush=True)

    def voice_clone(target_text):
        seed_everything(SEED)
        decode_config = {
            "top_k": 30, "top_p": 0.9, "min_p": 0.0,
            "temperature": 0.8, "stop_repetition": 3,
            "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
            "silence_tokens": [], "sample_batch_size": 1,
        }
        ref_info = torchaudio.info(ref_audio)
        prompt_end_frame = int(100 * ref_info.sample_rate)
        target_gen_sec = estimate_duration(
            target_text=target_text, reference_speech=ref_audio,
            reference_transcript=ref_prefix or None,
            target_lang=args.lang, reference_lang=None,
        )
        out = inference_one_sample(
            model=model, model_args=model.args,
            text_tokenizer=text_tok, audio_tokenizer=audio_tok,
            audio_fn=ref_audio, target_text=target_text,
            lang=args.lang, device=device, decode_config=decode_config,
            prompt_end_frame=prompt_end_frame,
            target_generation_length=target_gen_sec,
            prefix_transcript=ref_prefix, quiet=True,
        )
        if isinstance(out, tuple): _, gen_audio = out[0], out[1]
        else: gen_audio = out
        return gen_audio[0].cpu() if hasattr(gen_audio, "shape") else gen_audio

    # Phoneme ASR
    print(f"[Info] Loading phoneme ASR ({args.phoneme_model})...", flush=True)
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    from huggingface_hub import hf_hub_download
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(args.phoneme_model)
    asr = Wav2Vec2ForCTC.from_pretrained(args.phoneme_model).to(device).eval()
    vocab = json.load(open(hf_hub_download(args.phoneme_model, "vocab.json")))
    id2sym = {v: k for k, v in vocab.items()}
    pad_id = vocab.get("<pad>", 0)

    def transcribe_ipa(wav_t, sr=codec_audio_sr):
        if wav_t.dim() == 1: wav_t = wav_t.unsqueeze(0)
        if wav_t.shape[0] > 1: wav_t = wav_t.mean(0, keepdim=True)
        if sr != feat_ext.sampling_rate:
            wav_t = F.resample(wav_t.float(), sr, feat_ext.sampling_rate)
        a = wav_t.squeeze().numpy()
        if np.sqrt((a ** 2).mean()) < 0.001: return ""
        with torch.no_grad():
            x = feat_ext(a, sampling_rate=feat_ext.sampling_rate,
                          return_tensors="pt").input_values.to(device)
            pred = torch.argmax(asr(x).logits, dim=-1)[0].cpu().tolist()
        toks, prev = [], None
        for i in pred:
            if i == pad_id: prev = i; continue
            if i == prev: continue
            toks.append(id2sym.get(i, ""))
            prev = i
        return " ".join(toks).replace("|", " ").strip()

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    print(f"[Info] Loading Whisper (orthographic ASR)...", flush=True)
    wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    wm = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16).to(device).eval()

    def whisper_transcribe(wav_t, sr=codec_audio_sr, lang_code="en"):
        if wav_t.dim() > 1: wav_t = wav_t.squeeze()
        if sr != 16000:
            wav_t = F.resample(wav_t.unsqueeze(0).float(), sr, 16000).squeeze(0)
        feats = wp(wav_t.numpy(), sampling_rate=16000,
                    return_tensors="pt").input_features.to(device, dtype=torch.float16)
        forced = wp.get_decoder_prompt_ids(
            language={"en": "english", "ja": "japanese", "zh": "chinese"}[lang_code],
            task="transcribe")
        with torch.no_grad():
            ids = wm.generate(feats, forced_decoder_ids=forced, max_new_tokens=256)
        return wp.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def in_text(name, transcript):
        if args.lang in ("ja", "zh"):
            return name in transcript.replace(" ", "")
        return name.lower() in transcript.lower()

    hint_fn = HINT_TEMPLATES[args.lang]

    rows = []
    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
    done = {r["pair_idx"] for r in rows}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        pidx = pair.get("pair_idx", i)
        if pidx in done: continue
        try:
            text_a, text_b = pair["text_a"], pair["text_b"]
            word_a, word_b = pair["word_a"], pair["word_b"]

            # B0 baseline
            bl_audio = voice_clone(text_a)
            # B1 surface edit (synth text_b directly)
            b1_audio = voice_clone(text_b)
            # B7 text hint
            hinted_text = hint_fn(text_a, word_a, word_b)
            b7_audio = voice_clone(hinted_text)

            # Save wavs
            sp = lambda label, w: save_audio(
                os.path.join(wav_dir, f"s{pidx:04d}_{label}.wav"),
                w if w.dim() > 1 else w.unsqueeze(0), codec_audio_sr)
            sp("bl", bl_audio); sp("b1", b1_audio); sp("b7", b7_audio)

            # Whisper transcripts
            tr_bl = whisper_transcribe(bl_audio, lang_code=args.lang)
            tr_b1 = whisper_transcribe(b1_audio, lang_code=args.lang)
            tr_b7 = whisper_transcribe(b7_audio, lang_code=args.lang)
            # Phoneme IPA
            ipa_bl = transcribe_ipa(bl_audio)
            ipa_b1 = transcribe_ipa(b1_audio)
            ipa_b7 = transcribe_ipa(b7_audio)

            row = {
                "pair_idx": pidx,
                "word_a": word_a, "word_b": word_b,
                "text_a": text_a, "text_b": text_b, "text_b7": hinted_text,
                "bl_transcript": tr_bl,
                "b1_transcript": tr_b1,
                "b7_transcript": tr_b7,
                "bl_ipa": ipa_bl, "b1_ipa": ipa_b1, "b7_ipa": ipa_b7,
                "bl_target_in": in_text(word_b, tr_bl),
                "b1_target_in": in_text(word_b, tr_b1),
                "b7_target_in": in_text(word_b, tr_b7),
                "b1_source_in": in_text(word_a, tr_b1),
                "b7_source_in": in_text(word_a, tr_b7),
                "ipa_diff_b1": ipa_b1 != ipa_bl,
                "ipa_diff_b7": ipa_b7 != ipa_bl,
            }
            rows.append(row); done.add(pidx)
        except Exception as e:
            print(f"  [{i}] err: {str(e)[:200]}", flush=True)
            continue
        if (i + 1) % 5 == 0:
            n_b1 = sum(1 for r in rows if r.get("b1_target_in"))
            n_b7 = sum(1 for r in rows if r.get("b7_target_in"))
            print(f"  [{len(rows)}/{args.n}] B1_target_in={n_b1} B7_target_in={n_b7}", flush=True)
        if (i + 1) % 25 == 0:
            with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    n_total = len(rows)
    summary = {
        "lang": args.lang, "n_total": n_total,
        "b1_target_in_rate": round(sum(1 for r in rows if r["b1_target_in"]) / max(1, n_total), 4),
        "b7_target_in_rate": round(sum(1 for r in rows if r["b7_target_in"]) / max(1, n_total), 4),
        "bl_target_in_rate": round(sum(1 for r in rows if r["bl_target_in"]) / max(1, n_total), 4),
        "b1_source_in_rate": round(sum(1 for r in rows if r["b1_source_in"]) / max(1, n_total), 4),
        "b7_source_in_rate": round(sum(1 for r in rows if r["b7_source_in"]) / max(1, n_total), 4),
        "ipa_diff_b1_rate": round(sum(1 for r in rows if r["ipa_diff_b1"]) / max(1, n_total), 4),
        "ipa_diff_b7_rate": round(sum(1 for r in rows if r["ipa_diff_b7"]) / max(1, n_total), 4),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== B1/B7 {args.lang} N={n_total} ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
