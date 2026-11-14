#!/usr/bin/env python3
"""Speaker-fixed UTMOS22 paired evaluation (Table 2 Phase 2).

T5Gemma-TTS supports zero-shot voice cloning through `reference_speech`. We
exploit this to lock the speaker identity to a single reference wav across
*all six languages* and across BL / INJ conditions, so the resulting UTMOS22
differences are not contaminated by internal speaker drift.

Pipeline per pair:
  1. text_a (source-reading) and text_b (target-reading) -> encoder outputs.
  2. For INJ, replace the entity-span rows of the src encoder hidden state with
     the ref encoder hidden state rows. Alignment (--align):
       splice (default, the official implementation since 2026-07-23): replace the
         span together with its sequence length. For unequal lengths, a dummy target
         aligns the input_ids length and duration stays fixed to the text_a reference.
         （scripts/projection/splice_clone.py）。
       pool (legacy): length-preserving. Unequal lengths are compressed with
         adaptive_avg_pool1d, which loses phoneme information and breaks down on rare
         JA readings. Kept for comparison and for reproducing earlier runs.
  3. Register a forward hook on `model.encoder` to override `last_hidden_state`
     with the modified tensor.
  4. Call `inference_one_sample(audio_fn=<shared reference wav>, ...)` so that
     the decoder is initialised from the reference codec frames (voice cloning).
  5. Score the generated wav with UTMOS22 + SQUIM-PESQ.

Outputs `eval_results/utmos_speaker_fixed_<lang>_n<N>/results.json` whose
`summary` block contains `bl_utmos`, `inj_utmos`, `paired_t_pval`, `wilcoxon_pval`
suitable for appending to Appendix Table 2'.

Usage:
  python \\
    python scripts/projection/eval_utmos_speaker_fixed.py --lang ja --n 300 \\
      --reference_wav /data/.../subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav
"""
import argparse, json, os, sys, time, random

for _p in ("/tmp/.cache/huggingface/token", os.path.expanduser("~/.cache/huggingface/token")):
    if os.path.exists(_p):
        try:
            with open(_p) as _f:
                os.environ.setdefault("HF_TOKEN", _f.read().strip())
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
            break
        except Exception: pass

# Force-override HF_HOME so that we use a repository-local model cache holding
# Aratako/T5Gemma-TTS-2b-2b (avoids the gated-repo 401 inside a container).
# `os.environ.setdefault` would not override an externally set HF_HOME,
# so we assign directly.
_hf_home = "/mnt/.cache/huggingface" if os.path.isdir("/mnt/scripts") \
           else "./.cache/huggingface"
os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
# Disable any outgoing HF Hub call.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _autoroot():
    env = os.environ.get("SPEECHSPLICE_ROOT")
    if env: return env
    if os.path.isdir("/mnt/scripts"): return "/mnt"
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        if os.path.isdir(os.path.join(cand, "data")) and os.path.isdir(os.path.join(cand, "scripts")):
            return cand
    return "."
PROJECT_ROOT = _autoroot()
T5GEMMA_ROOT = "./vendor/t5gemma_tts"

sys.path.insert(0, T5GEMMA_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch, torchaudio, numpy as np
from scipy import stats as scipy_stats
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from encoder_injection_t5gemma import (
    seed_everything, get_encoder_output, find_entity_token_range,
)
from inference_tts_utils import (
    inference_one_sample, save_audio,
    normalize_text_with_lang, transcribe_audio,
)
from duration_estimator import estimate_duration
from data.tokenizer import AudioTokenizer

SEED = 42
DEFAULT_REF = "./eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav"


def seed_all(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


# --- Encoder helper that mirrors the official tokenization ---------------

def get_encoder_output_with_prefix(model, text_tokenizer, prefix_text, target_text,
                                    device, lang):
    """Compute the encoder last-hidden-state for the *full* text tokens that
    `inference_one_sample` will feed to the model, i.e.

        [BOS?] + prefix_tokens + [x_sep] + target_tokens + [EOS?]

    so that a forward-hook injection over `encoder.last_hidden_state` has the
    correct shape during voice-cloning generation. Also returns the offset of
    the target portion in the full token sequence so the caller can map
    `find_entity_token_range(target_text, ...)` back to absolute indices.
    """
    target_n, lang_code = normalize_text_with_lang(target_text, lang)
    prefix_n = ""
    if prefix_text:
        prefix_n, _ = normalize_text_with_lang(prefix_text, lang_code)

    prefix_tokens = text_tokenizer.encode(prefix_n.strip(), add_special_tokens=False) if prefix_n else []
    target_tokens = text_tokenizer.encode(target_n.strip(), add_special_tokens=False)

    x_sep = getattr(model.args, "x_sep_token", None)
    text_tokens = []
    if prefix_tokens:
        text_tokens += list(prefix_tokens)
        if x_sep is not None:
            text_tokens += [x_sep]
    target_start = len(text_tokens)
    text_tokens += list(target_tokens)
    add_eos = int(getattr(model.args, "add_eos_to_text", 0))
    add_bos = int(getattr(model.args, "add_bos_to_text", 0))
    if add_eos: text_tokens.append(add_eos)
    if add_bos:
        text_tokens = [add_bos] + text_tokens
        target_start += 1

    input_ids = torch.LongTensor([text_tokens]).to(device)
    attention_mask = torch.ones_like(input_ids)
    encoder = model.encoder_module
    position_ids = None
    if int(getattr(model.args, "use_pm_rope", 1)):
        x_lens = torch.LongTensor([len(text_tokens)]).to(device)
        position_ids = model._build_position_ids(x_lens, len(text_tokens), device)

    with torch.no_grad():
        out = encoder(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=position_ids)

    return {
        "last_hidden_state": out.last_hidden_state,
        "tokens": text_tokens,
        "target_text": target_n,
        "target_tokens": target_tokens,
        "target_start_in_full": target_start,
        "prefix_tokens": prefix_tokens,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


# --- Pair loading per language --------------------------------------------

def load_pairs(lang, n, pairs_module=None):
    if pairs_module:
        # Load from an arbitrary pair-definition module: build_pairs() in data/<name>.py
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
        mod = __import__(pairs_module)
        return mod.build_pairs()[:n]
    if lang == "en":
        from en_heteronyms_500 import build_500_pairs
        return build_500_pairs()[:n]
    if lang == "zh":
        from zh_polyphones_300 import build_300_pairs
        return build_300_pairs()[:n]
    if lang == "ja":
        prev_path = os.path.join(PROJECT_ROOT, "eval_results/crossattn_mask_n300_filtered_v2/results.json")
        prev = json.load(open(prev_path))["results"]
        out = []
        for r in prev[:n]:
            out.append({
                "pair_idx": r["idx"],
                "text_a": r["text"], "text_b": r["text"].replace(r["kanji"], r["kata"]),
                "word_a": r["kanji"], "word_b": r["kata"],
            })
        return out
    if lang == "ko":
        # Use the strict variant: only pairs whose text_a AND text_b both
        # contain the entity hanja (or its 두음법칙 equivalent), so all 300
        # pairs are guaranteed to have a valid entity-token range.
        from ko_hanja_300_strict import build_300_pairs
        return build_300_pairs()[:n]
    if lang == "fr":
        from fr_heteronyms_300 import build_300_pairs
        return build_300_pairs()[:n]
    if lang == "de":
        from de_heteronyms_300 import build_300_pairs
        return build_300_pairs()[:n]
    raise ValueError(lang)


# --- Voice-cloned synth with encoder-hidden-state override ----------------

def voice_clone_synth(
    model, text_tok, audio_tok, *, target_text, lang, device,
    codec_audio_sr, codec_sr,
    ref_wav, ref_text, ref_prompt_sec,
    modified_hidden=None, duration_text=None,
):
    """Generate audio with voice cloning. If `modified_hidden` is given (the
    modified encoder last_hidden_state), a forward-hook overrides the encoder
    output for the duration of one call.

    `duration_text`: when given, duration estimation uses this text instead of
    `target_text` (clone-splice mode passes a length-matched dummy as
    `target_text` while keeping duration anchored to the original text so
    that Δduration stays 0). None keeps the original behaviour bit-for-bit."""
    seed_everything(SEED)
    dur_text = duration_text if duration_text is not None else target_text

    handle = None
    if modified_hidden is not None:
        def _hook(module, inputs, output):
            if hasattr(output, "last_hidden_state"):
                assert output.last_hidden_state.shape[1] == modified_hidden.shape[1], (
                    f"encoder/hook length mismatch: encoder T={output.last_hidden_state.shape[1]} "
                    f"vs modified T={modified_hidden.shape[1]} (dummy target mis-sized?)")
                output.last_hidden_state = modified_hidden
                return output
            if isinstance(output, tuple) and len(output) > 0:
                assert output[0].shape[1] == modified_hidden.shape[1], (
                    f"encoder/hook length mismatch (tuple): {output[0].shape[1]} "
                    f"vs {modified_hidden.shape[1]}")
                return (modified_hidden,) + output[1:]
            assert output.shape[1] == modified_hidden.shape[1], (
                f"encoder/hook length mismatch (bare): {output.shape[1]} "
                f"vs {modified_hidden.shape[1]}")
            return modified_hidden
        # T5Gemma encoder module path
        encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
        handle = encoder.register_forward_hook(_hook)

    try:
        # Decode hyper-parameters match the official inference_commandline_hf.py
        # defaults for T5Gemma-TTS-2b-2b (top_k=30, top_p=0.9, temperature=0.8,
        # stop_repetition=3). Greedy decoding (top_k=0 / temp=1.0 / no
        # stop_repetition) produces noticeably lower-quality samples for this
        # model — confirmed by README example.
        decode_config = {
            "top_k": 30, "top_p": 0.9, "min_p": 0.0,
            "temperature": 0.8, "stop_repetition": 3,
            "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
            "silence_tokens": [],
            "sample_batch_size": 1,
        }
        # prompt_end_frame is in WAVEFORM samples (handed to torchaudio.load as
        # num_frames) at the REFERENCE wav's native sample rate, mirroring the
        # official inference_commandline_hf.py: cut_off_sec * info.sample_rate.
        try:
            ref_info = torchaudio.info(ref_wav)
            prompt_end_frame = int(ref_prompt_sec * ref_info.sample_rate)
        except Exception:
            prompt_end_frame = int(ref_prompt_sec * codec_audio_sr)
        # Use the official duration estimator, identical to the inference CLI.
        # When the reference and target languages differ, reusing the seconds-per-phoneme
        # calibrated from the reference mixes units and underestimates duration (e.g. the
        # JA reference mora rate of 0.075 s times the ZH syllable count truncates to a
        # third or a quarter of the real duration). Switch to the per-language default
        # speaking rate in the cross-lingual case.
        try:
            from duration_estimator import detect_language as _detect_lang
            _ref_lang = _detect_lang(ref_text) if ref_text else None
            if lang and _ref_lang and _ref_lang != lang:
                target_gen_sec = estimate_duration(
                    target_text=dur_text, target_lang=lang)
            else:
                target_gen_sec = estimate_duration(
                    target_text=dur_text,
                    reference_speech=ref_wav,
                    reference_transcript=ref_text or None,
                    target_lang=lang,
                    reference_lang=None,
                )
        except Exception:
            target_gen_sec = max(2.0, 0.18 * max(1, len(dur_text)))
        out = inference_one_sample(
            model=model, model_args=model.args,
            text_tokenizer=text_tok, audio_tokenizer=audio_tok,
            audio_fn=ref_wav,
            target_text=target_text, lang=lang, device=device,
            decode_config=decode_config,
            prompt_end_frame=prompt_end_frame,
            target_generation_length=target_gen_sec,
            prefix_transcript=ref_text,
            quiet=True,
        )
        if isinstance(out, tuple):
            _, gen_audio = out[0], out[1]
        else:
            gen_audio = out
        return gen_audio[0].cpu() if hasattr(gen_audio, "shape") else gen_audio
    finally:
        if handle is not None: handle.remove()


# --- Scoring --------------------------------------------------------------

def score_wav(wav_path_or_tensor, sr, utmos, squim, device, target_sr=16000):
    if isinstance(wav_path_or_tensor, str):
        wav, sr_in = torchaudio.load(wav_path_or_tensor)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr_in != target_sr:
            wav = torchaudio.functional.resample(wav, sr_in, target_sr)
    else:
        wav = wav_path_or_tensor
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav.float(), sr, target_sr)
    with torch.no_grad():
        u = utmos(wav.to(device), target_sr).item()
        try:
            _, pesq, _ = squim(wav.to(device))
            pesq_v = pesq.item()
        except Exception:
            pesq_v = None
    return u, pesq_v


# --- Main -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True,
                   choices=["ja", "en", "zh", "ko", "fr", "de"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--reference_wav", default=DEFAULT_REF)
    p.add_argument("--reference_text", default="",
                   help="Transcript of reference_wav. If empty, Whisper is used inside inference_one_sample.")
    p.add_argument("--reference_prompt_sec", type=float, default=100.0,
                   help="Truncate reference wav to first N sec as voice prompt "
                        "(matches the official demo's cut_off_sec=100 i.e. use the entire reference).")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--pairs_module", default=None,
                   help="read pairs from build_pairs() in data/<name>.py")
    p.add_argument("--align", default="splice", choices=["pool", "splice"],
                   help="alignment mode for INJ injection. splice=the official "
                        "implementation (since 2026-07-23): replaces the span with its "
                        "sequence length, aligns input_ids length with a dummy target, and "
                        "fixes duration to the text_a reference. pool=legacy: "
                        "length-preserving, compressing unequal lengths with "
                        "adaptive_avg_pool1d; use only for comparison or to reproduce "
                        "earlier runs.")
    args = p.parse_args()
    if args.output_dir is None:
        if args.pairs_module:
            args.output_dir = f"eval_results/{args.pairs_module}_gen"
        else:
            args.output_dir = f"eval_results/utmos_speaker_fixed_{args.lang}_n{args.n}"
        if args.align == "splice":
            args.output_dir += "_splice"
    os.chdir(PROJECT_ROOT); os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(SEED)

    pairs = load_pairs(args.lang, args.n, pairs_module=args.pairs_module)
    print(f"[Info] {args.lang}: {len(pairs)} pairs loaded; ref={args.reference_wav}")

    # Reference transcript: mirror the official inference behaviour. If the
    # caller did not supply --reference_text we Whisper-transcribe the wav so
    # that the model receives the prefix tokens it was trained with. Skipping
    # this drops cloned-BL UTMOS by ~0.5 (verified vs. inference_commandline_hf
    # oracle: 3.59 with Whisper prefix vs 3.11 without).
    REF_PREFIX = args.reference_text or ""
    if not REF_PREFIX:
        print("[Info] Transcribing reference audio via Whisper-large-v3-turbo ...")
        REF_PREFIX = transcribe_audio(args.reference_wav, device)
        print(f"[Info] Whisper transcript: {REF_PREFIX}")

    def _local_snapshot(hub_id):
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
        repo_dir = os.path.join(hub, f"models--{hub_id.replace('/','--')}", "snapshots")
        if os.path.isdir(repo_dir):
            snaps = sorted(os.listdir(repo_dir))
            if snaps:
                return os.path.join(repo_dir, snaps[-1])
        return hub_id

    print("[Info] Loading T5Gemma-TTS ...")
    model_id = _local_snapshot("Aratako/T5Gemma-TTS-2b-2b")
    print(f"[Info] model_id = {model_id}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto",
        local_files_only=True,
    )
    model.eval()
    cfg = model.config
    tkn_hub_id = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    tkn_path = _local_snapshot(tkn_hub_id) if tkn_hub_id else tkn_hub_id
    print(f"[Info] tokenizer path = {tkn_path}")
    text_tok = AutoTokenizer.from_pretrained(tkn_path, local_files_only=True)
    audio_tok = AudioTokenizer(backend="xcodec2",
                                model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(model.args, "add_bos_to_text", 0)) else 0

    print("[Info] Loading UTMOS22 ...")
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
    print("[Info] Loading SQUIM-OBJECTIVE ...")
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    squim = SQUIM_OBJECTIVE.get_model().to(device).eval()

    rows = []
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
        # Prevent resuming across mixed align modes (pool rows have no align key;
        # splice rows carry "splice")
        _expect = "splice" if args.align == "splice" else None
        _bad = [r["pair_idx"] for r in rows if r.get("align") != _expect]
        assert not _bad, (
            f"checkpoint align mismatch in {ckpt}: pairs {_bad[:5]}... were generated "
            f"with a different --align mode; use a separate --output_dir")
    done = {r["pair_idx"] for r in rows}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        pidx = pair.get("pair_idx", i)
        if pidx in done: continue
        seed_everything(SEED)
        text_a, text_b = pair["text_a"], pair["text_b"]
        word_a, word_b = pair["word_a"], pair["word_b"]

        try:
            src_enc = get_encoder_output_with_prefix(
                model, text_tok, REF_PREFIX, text_a, device, lang=args.lang)
            ref_enc = get_encoder_output_with_prefix(
                model, text_tok, REF_PREFIX, text_b, device, lang=args.lang)
        except Exception as e:
            print(f"  [{i}] enc fail: {str(e)[:120]}"); continue

        # Find entity within the *target* portion of each full sequence, then
        # shift by target_start_in_full to get absolute indices in the encoder
        # output (which has length = bos? + prefix + sep + target + eos?).
        def _resolve_surface(word, full_text):
            """Try word as-is; fall back to (1) stem-prefix match for
            inflected forms (e.g. DE word_a='übersetzen' but text has
            'übersetzt'); (2) Korean 두음법칙 equivalents (락↔낙, 량↔양,
            etc.) which the SinoKorean hanja system uses at word-initial
            position; (3) German umlaut variants (a↔ä etc.) covered by the
            stem search going down to 2 chars."""
            if word in full_text or word.lower() in full_text.lower():
                return word
            import re, unicodedata
            # KO 두음법칙 mapping (hanja initial-sound substitutions)
            DUEUM = {
                "락": "낙", "랑": "낭", "량": "양", "려": "여", "력": "역",
                "련": "연", "렬": "열", "령": "영", "례": "예", "로": "노",
                "록": "녹", "론": "논", "롱": "농", "뢰": "뇌", "료": "요",
                "룡": "용", "루": "누", "류": "유", "륙": "육", "륜": "윤",
                "률": "율", "륭": "융", "륵": "늑", "름": "늠", "릉": "능",
                "리": "이", "린": "인", "림": "임", "립": "입",
                # reverse mappings (낙→락 etc.)
            }
            DUEUM_REV = {v: k for k, v in DUEUM.items()}
            if word in DUEUM and DUEUM[word] in full_text:
                return DUEUM[word]
            if word in DUEUM_REV and DUEUM_REV[word] in full_text:
                return DUEUM_REV[word]
            # German umlaut-aware folding for stem prefix match
            def fold(s):
                # Unicode NFC then strip diacritics (NFD + filter combining)
                s = unicodedata.normalize("NFD", s)
                s = "".join(c for c in s if unicodedata.category(c) != "Mn")
                return s.lower()
            ftext_folded = fold(full_text)
            min_stem = max(3, min(5, len(word) - 2))
            stem_folded = fold(word)[:min_stem]
            for ml in range(min_stem, 1, -1):  # try down to 2 chars
                if ml > len(stem_folded): continue
                pat = re.compile(r"\w*" + re.escape(stem_folded[:ml]) + r"\w*",
                                  re.UNICODE | re.IGNORECASE)
                m = pat.search(ftext_folded)
                if m:
                    # Re-locate in original text using start/end offsets
                    start, end = m.start(), m.end()
                    return full_text[start:end]
            return word
        surface_a = _resolve_surface(word_a, src_enc["target_text"])
        surface_b = _resolve_surface(word_b, ref_enc["target_text"])
        s_sub = find_entity_token_range(text_tok, src_enc["target_text"], surface_a, add_bos=0)
        r_sub = find_entity_token_range(text_tok, ref_enc["target_text"], surface_b, add_bos=0)
        if s_sub is None or r_sub is None:
            print(f"  [{i}] range None: surface_a={surface_a!r} s_sub={s_sub} | surface_b={surface_b!r} r_sub={r_sub}", flush=True)
            # Fallback: use the last 3 tokens of the target span as the
            # "entity" (most heteronym variants are stem suffix anyway). This
            # keeps N=300 while not silently dropping pairs.
            s_len = len(text_tok.encode(src_enc["target_text"], add_special_tokens=False))
            r_len = len(text_tok.encode(ref_enc["target_text"], add_special_tokens=False))
            if s_len <= 0 or r_len <= 0:
                print(f"  [{i}] empty target tokens, skip", flush=True); continue
            n_ent = 3
            s_sub = (max(0, s_len - n_ent), s_len)
            r_sub = (max(0, r_len - n_ent), r_len)
            print(f"  [{i}] using last-{n_ent}-token fallback: s_sub={s_sub} r_sub={r_sub}", flush=True)
        s_off = src_enc["target_start_in_full"]
        r_off = ref_enc["target_start_in_full"]
        s_r = (s_sub[0] + s_off, s_sub[1] + s_off)
        r_r = (r_sub[0] + r_off, r_sub[1] + r_off)

        inj_target = text_a
        inj_dur_text = None
        delta_tok = 0
        if args.align == "pool":
            # Build modified encoder hidden state (entity replacement). Keep the
            # output length equal to src so the decoder's internal attention mask
            # (which is derived from the input id length) remains consistent.
            modified = src_enc["last_hidden_state"].clone()
            s_s, s_e = s_r; r_s, r_e = r_r
            s_len = s_e - s_s; r_len = r_e - r_s
            if s_len == r_len:
                modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
            else:
                import torch.nn.functional as F
                ref_seg = ref_enc["last_hidden_state"][0, r_s:r_e].unsqueeze(0).transpose(1, 2)
                ref_resampled = F.adaptive_avg_pool1d(ref_seg.float(), s_len).transpose(1, 2)
                modified[0, s_s:s_e] = ref_resampled[0].to(modified.dtype)
        else:
            # clone-splice: replace the span together with its sequence length. For
            # unequal lengths, use a dummy target that makes the input_ids length of
            # inference_one_sample match the post-splice length, and fix duration to the
            # text_a reference (keeping d_duration = 0.00 s).
            from splice_clone import prepare_clone_splice
            try:
                modified, _inj_t, inj_dur_text, delta_tok = prepare_clone_splice(
                    text_tok, src_enc, ref_enc, s_r, r_r, args.lang,
                    duration_source_text=text_a, natural_candidates=(text_b,))
            except Exception as e:
                print(f"  [{i}] splice prep fail: {str(e)[:120]}", flush=True); continue
            if _inj_t is not None:
                inj_target = _inj_t

        row = {"pair_idx": pidx, "word_a": word_a, "word_b": word_b,
               "text_a": text_a, "text_b": text_b}
        if args.align == "splice":
            row.update(align="splice", delta_tok=delta_tok,
                       T_src=int(src_enc["last_hidden_state"].shape[1]),
                       T_out=int(modified.shape[1]),
                       inj_target_text=inj_target)
        try:
            bl_audio = voice_clone_synth(
                model, text_tok, audio_tok,
                target_text=text_a, lang=args.lang, device=device,
                codec_audio_sr=codec_audio_sr, codec_sr=codec_sr,
                ref_wav=args.reference_wav, ref_text=REF_PREFIX,
                ref_prompt_sec=args.reference_prompt_sec,
                modified_hidden=None,
            )
            inj_audio = voice_clone_synth(
                model, text_tok, audio_tok,
                target_text=inj_target, lang=args.lang, device=device,
                codec_audio_sr=codec_audio_sr, codec_sr=codec_sr,
                ref_wav=args.reference_wav, ref_text=REF_PREFIX,
                ref_prompt_sec=args.reference_prompt_sec,
                modified_hidden=modified, duration_text=inj_dur_text,
            )
        except Exception as e:
            print(f"  [{i}] synth fail: {str(e)[:120]}"); continue

        try:
            bl_path = os.path.join(args.output_dir, f"s{i:04d}_bl.wav")
            inj_path = os.path.join(args.output_dir, f"s{i:04d}_inj.wav")
            save_audio(bl_path, bl_audio if bl_audio.dim() > 1 else bl_audio.unsqueeze(0), codec_audio_sr)
            save_audio(inj_path, inj_audio if inj_audio.dim() > 1 else inj_audio.unsqueeze(0), codec_audio_sr)
            u_bl, p_bl = score_wav(bl_path, codec_audio_sr, utmos, squim, device)
            u_inj, p_inj = score_wav(inj_path, codec_audio_sr, utmos, squim, device)
            row.update(bl_utmos=u_bl, bl_pesq=p_bl, inj_utmos=u_inj, inj_pesq=p_inj)
            if args.align == "splice":
                _ib, _ii = torchaudio.info(bl_path), torchaudio.info(inj_path)
                row.update(bl_dur_sec=round(_ib.num_frames / _ib.sample_rate, 4),
                           inj_dur_sec=round(_ii.num_frames / _ii.sample_rate, 4))
        except Exception as e:
            print(f"  [{i}] score fail: {str(e)[:120]}"); continue

        rows.append(row); done.add(pidx)
        if (i+1) % 5 == 0:
            print(f"  [{len(rows)}/{args.n}] {word_a}  BL_U={row.get('bl_utmos','-')}  INJ_U={row.get('inj_utmos','-')}")
        if (i+1) % 25 == 0:
            with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)
    with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    bl = np.array([r["bl_utmos"] for r in rows if r.get("bl_utmos") is not None])
    inj = np.array([r["inj_utmos"] for r in rows if r.get("inj_utmos") is not None])
    n_paired = min(len(bl), len(inj))
    if n_paired > 1:
        t_stat, t_pval = scipy_stats.ttest_rel(bl[:n_paired], inj[:n_paired])
        w_stat, w_pval = scipy_stats.wilcoxon(bl[:n_paired], inj[:n_paired])
    else:
        t_stat = t_pval = w_stat = w_pval = None
    summary = {
        "lang": args.lang, "n": len(rows), "n_paired": n_paired,
        "reference_wav": args.reference_wav,
        "bl_utmos": [round(float(bl.mean()), 4), round(float(bl.std()), 4)] if len(bl) else [None, None],
        "inj_utmos": [round(float(inj.mean()), 4), round(float(inj.std()), 4)] if len(inj) else [None, None],
        "mean_diff_inj_minus_bl": round(float((inj[:n_paired] - bl[:n_paired]).mean()), 4) if n_paired else None,
        "paired_t_stat": round(float(t_stat), 4) if t_stat is not None else None,
        "paired_t_pval": round(float(t_pval), 6) if t_pval is not None else None,
        "wilcoxon_stat": round(float(w_stat), 4) if w_stat is not None else None,
        "wilcoxon_pval": round(float(w_pval), 6) if w_pval is not None else None,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== Speaker-fixed UTMOS {args.lang} N={len(rows)} ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
