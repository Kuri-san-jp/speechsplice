#!/usr/bin/env python3
"""Crossattn-mask 4-condition entity-level phoneme PER on T5Gemma-TTS.

For each pair (text_a with entity_a, text_b with entity_b, same surface in
heteronym/polyphone, kanji vs katakana in JA), generate 5 audio:
  BL       = synth(text_a) with full cross-attention
  INJ      = synth(text_a) with entity-position memory replaced by entity_b's
  BL_mask  = synth(text_a) with cross-attention mask at entity zeroed
  INJ_mask = synth(text_a) with entity replaced AND mask zeroed
  TGT      = synth(text_b) with full cross-attention (target reference)

Forced-align each audio with Whisper word_timestamps + char subdivision,
slice the entity span, run wav2vec2-xlsr-53-espeak-cv-ft phoneme ASR on
the slice, and compute PER against TGT entity IPA. Target match rate is
the proportion of samples whose entity-PER < threshold (default 0.3).

Usage:
  SLURM_TIME=06:00:00 slurm_run.sh \\
    python scripts/projection/eval_crossattn_entity_per.py --lang ja --n 300
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

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch, torchaudio, numpy as np
from forced_alignment_utils import WhisperAligner, find_entity_chunks, slice_entity_audio


SEED = 42
PER_THRESHOLD = 0.3  # entity-level PER below this counts as target match


def seed_all(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


def levenshtein(a, b):
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def per_dist(hyp, ref):
    if not ref: return 1.0 if hyp else 0.0
    return levenshtein(hyp, ref) / max(1, len(ref))


def tokenize_ipa(s):
    if not s: return []
    s = s.strip()
    for ch in ("ˈ", "ˌ", "ː", "ʰ", "ʲ", "ʷ", " ", "|"):
        s = s.replace(ch, "")
    return list(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--lang", default="ja", choices=["en", "ja", "zh"])
    p.add_argument("--output_dir", default=None)
    p.add_argument("--phoneme_model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    p.add_argument("--whisper_model", default="openai/whisper-large-v3")
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"eval_results/crossattn_entity_per_{args.lang}_n{args.n}"

    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"
    seed_all(SEED)

    # === Load pairs ===
    if args.lang == "en":
        from en_heteronyms_500 import build_500_pairs
        pairs = build_500_pairs()[: args.n]
    elif args.lang == "zh":
        from zh_polyphones_300 import build_300_pairs
        pairs = build_300_pairs()[: args.n]
    elif args.lang == "ja":
        prev = json.load(open("eval_results/crossattn_mask_n300_filtered_v2/results.json"))["results"]
        pairs = []
        for r in prev[: args.n]:
            text_a = r["text"]; kanji = r["kanji"]; kata = r["kata"]
            pairs.append({
                "text_a": text_a, "text_b": text_a.replace(kanji, kata),
                "word_a": kanji, "word_b": kata, "pair_idx": r["idx"],
            })
    print(f"[Info] Loaded {len(pairs)} {args.lang} pairs")

    # === Phoneme ASR ===
    print(f"[Info] Loading phoneme ASR ...")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    from huggingface_hub import hf_hub_download
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(args.phoneme_model)
    asr_model = Wav2Vec2ForCTC.from_pretrained(args.phoneme_model).to(device).eval()
    vocab_path = hf_hub_download(args.phoneme_model, "vocab.json")
    with open(vocab_path) as _f: vocab = json.load(_f)
    id2sym = {v: k for k, v in vocab.items()}; pad_id = vocab.get("<pad>", 0)

    def transcribe_audio_to_ipa(audio_tensor, sr_in):
        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.from_numpy(audio_tensor).float()
        if audio_tensor.dim() > 1: audio_tensor = audio_tensor.squeeze()
        if sr_in != feat_ext.sampling_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor.float(), sr_in, feat_ext.sampling_rate)
        wav_np = audio_tensor.cpu().numpy()
        if wav_np.size < int(0.05 * feat_ext.sampling_rate):
            return ""
        with torch.no_grad():
            x = feat_ext(wav_np, sampling_rate=feat_ext.sampling_rate, return_tensors="pt").input_values.to(device)
            pred = torch.argmax(asr_model(x).logits, dim=-1)[0].cpu().tolist()
        tokens, prev = [], None
        for i in pred:
            if i == pad_id: prev = i; continue
            if i == prev: continue
            tokens.append(id2sym.get(i, "")); prev = i
        return " ".join(tokens).replace("|", " ").strip()

    # === Whisper aligner ===
    print(f"[Info] Loading Whisper aligner ...")
    aligner = WhisperAligner(device=device, model_id=args.whisper_model)

    def entity_ipa(audio, sr_in, entity, lang, input_text):
        chunks, audio_np = aligner.transcribe_words(audio, sr_in, language=lang)
        if not chunks or audio_np is None:
            return None
        s_idx, e_idx = find_entity_chunks(chunks, entity, lang, input_text=input_text)
        if s_idx is None: return None
        ent_audio = slice_entity_audio(audio_np, aligner.target_sr, chunks, s_idx, e_idx)
        if ent_audio is None or ent_audio.size == 0: return None
        return transcribe_audio_to_ipa(ent_audio, aligner.target_sr)

    # === T5Gemma synthesis ===
    print("[Info] Loading T5Gemma-TTS ...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from data.tokenizer import AudioTokenizer
    from encoder_injection_t5gemma import (
        seed_everything, get_encoder_output, generate_with_modified_encoder, find_entity_token_range,
    )
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        "Aratako/T5Gemma-TTS-2b-2b", trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto")
    t5_model.eval()
    cfg = t5_model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    t5_tok = AutoTokenizer.from_pretrained(tok_name)
    audio_tok = AudioTokenizer(backend="xcodec2",
                                model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(t5_model.args, "add_bos_to_text", 0)) else 0

    def run_condition(text_a, text_b, word_a, word_b, inject, mask_entity):
        """Generate audio under a single (inject, mask) condition.
        Returns (audio_tensor, sr) or (None, None) on failure."""
        seed_everything(SEED)
        src_enc = get_encoder_output(t5_model, t5_tok, text_a, device, lang=args.lang)
        if inject:
            ref_enc = get_encoder_output(t5_model, t5_tok, text_b, device, lang=args.lang)
            s_r = find_entity_token_range(t5_tok, src_enc["text"], word_a, add_bos=add_bos)
            r_r = find_entity_token_range(t5_tok, ref_enc["text"], word_b, add_bos=add_bos)
            if s_r is None or r_r is None: return None, None
            s_s, s_e = s_r; r_s, r_e = r_r
            modified = src_enc["last_hidden_state"].clone()
            if (s_e - s_s) == (r_e - r_s):
                modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
                entity_start, entity_end = s_s, s_e
            else:
                prefix = modified[0, :s_s]
                ref_ent = ref_enc["last_hidden_state"][0, r_s:r_e]
                suffix = modified[0, s_e:]
                modified = torch.cat([prefix, ref_ent, suffix], dim=0).unsqueeze(0)
                entity_start = s_s
                entity_end = s_s + (r_e - r_s)
            final_len = modified.shape[1]
        else:
            modified = src_enc["last_hidden_state"].clone()
            final_len = modified.shape[1]
            sr_range = find_entity_token_range(t5_tok, src_enc["text"], word_a, add_bos=add_bos)
            if sr_range is None: return None, None
            entity_start, entity_end = sr_range
        attn = torch.ones((1, final_len), dtype=torch.long, device=device)
        if mask_entity:
            attn[0, entity_start:entity_end] = 0
        pos_ids = None
        if int(getattr(t5_model.args, "use_pm_rope", 1)):
            x_lens = torch.LongTensor([final_len]).to(device)
            pos_ids = t5_model._build_position_ids(x_lens, final_len, device)
        seed_everything(SEED)
        out = generate_with_modified_encoder(
            t5_model, t5_tok, audio_tok, modified, attn, pos_ids,
            text_a, device, codec_audio_sr, codec_sr,
        )
        if out is None: return None, None
        return out[0], codec_audio_sr

    def synth_simple(text):
        """Plain synthesis (for TGT)."""
        seed_everything(SEED)
        enc = get_encoder_output(t5_model, t5_tok, text, device, lang=args.lang)
        h = enc["last_hidden_state"]
        attn = torch.ones((1, h.shape[1]), dtype=torch.long, device=device)
        pos_ids = None
        if int(getattr(t5_model.args, "use_pm_rope", 1)):
            x_lens = torch.LongTensor([h.shape[1]]).to(device)
            pos_ids = t5_model._build_position_ids(x_lens, h.shape[1], device)
        out = generate_with_modified_encoder(
            t5_model, t5_tok, audio_tok, h, attn, pos_ids,
            text, device, codec_audio_sr, codec_sr,
        )
        if out is None: return None, None
        return out[0], codec_audio_sr

    rows = []
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
    done = {r["pair_idx"] for r in rows}
    t0 = time.time()

    for i, pair in enumerate(pairs):
        pair_idx = pair.get("pair_idx", i)
        if pair_idx in done: continue

        try:
            # 5 audios: BL, INJ, BL_mask, INJ_mask, TGT
            bl_a, bl_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"],
                                         inject=False, mask_entity=False)
            inj_a, inj_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"],
                                           inject=True, mask_entity=False)
            bl_m_a, bl_m_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"],
                                             inject=False, mask_entity=True)
            inj_m_a, inj_m_sr = run_condition(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"],
                                               inject=True, mask_entity=True)
            tgt_a, tgt_sr = synth_simple(pair["text_b"])
            if any(x is None for x in [bl_a, inj_a, bl_m_a, inj_m_a, tgt_a]):
                continue

            # Entity-level IPAs (use word_b for TGT, word_a for the four text_a-based conditions)
            tgt_ipa = entity_ipa(tgt_a, tgt_sr, pair["word_b"], args.lang, pair["text_b"])
            bl_ipa = entity_ipa(bl_a, bl_sr, pair["word_a"], args.lang, pair["text_a"])
            inj_ipa = entity_ipa(inj_a, inj_sr, pair["word_a"], args.lang, pair["text_a"])
            bl_m_ipa = entity_ipa(bl_m_a, bl_m_sr, pair["word_a"], args.lang, pair["text_a"])
            inj_m_ipa = entity_ipa(inj_m_a, inj_m_sr, pair["word_a"], args.lang, pair["text_a"])

            row = {
                "pair_idx": pair_idx,
                "word_a": pair["word_a"], "word_b": pair["word_b"],
                "text_a": pair["text_a"], "text_b": pair["text_b"],
            }
            if tgt_ipa is None:
                row["aligned"] = False
                rows.append(row); done.add(pair_idx); continue

            tgt_t = tokenize_ipa(tgt_ipa)
            for cond, ipa in [("bl", bl_ipa), ("inj", inj_ipa), ("bl_mask", bl_m_ipa), ("inj_mask", inj_m_ipa)]:
                if ipa is None:
                    row[f"{cond}_per"] = None
                    row[f"{cond}_match"] = False
                else:
                    cond_t = tokenize_ipa(ipa)
                    pe = per_dist(cond_t, tgt_t)
                    row[f"{cond}_per"] = round(pe, 4)
                    row[f"{cond}_match"] = pe < PER_THRESHOLD
                    row[f"{cond}_ipa_len"] = len(cond_t)
            row["tgt_ipa_len"] = len(tgt_t)
            row["aligned"] = True
            rows.append(row); done.add(pair_idx)
        except Exception as e:
            print(f"  [{i}] err: {str(e)[:200]}"); continue

        if (i + 1) % 5 == 0:
            n_aligned = sum(1 for r in rows if r.get("aligned"))
            inj_match = sum(1 for r in rows if r.get("inj_match"))
            inj_m_match = sum(1 for r in rows if r.get("inj_mask_match"))
            print(f"  [{len(rows)}/{args.n}] aligned={n_aligned} inj_match={inj_match} inj_mask_match={inj_m_match}")
        if (i + 1) % 25 == 0:
            with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    n = len(rows)
    aligned = [r for r in rows if r.get("aligned")]
    n_align = len(aligned)
    if n_align == 0:
        print("[Error] No aligned rows"); return

    def cnt(key):
        return sum(1 for r in aligned if r.get(key))
    def avg_per(key):
        vs = [r[key] for r in aligned if r.get(key) is not None]
        return [round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)] if vs else [None, None]

    summary = {
        "lang": args.lang, "n_total": n, "n_aligned": n_align,
        "per_threshold_for_match": PER_THRESHOLD,
        "bl_match_count": cnt("bl_match"),
        "inj_match_count": cnt("inj_match"),
        "bl_mask_match_count": cnt("bl_mask_match"),
        "inj_mask_match_count": cnt("inj_mask_match"),
        "bl_match_pct": round(100*cnt("bl_match")/max(1, n_align), 2),
        "inj_match_pct": round(100*cnt("inj_match")/max(1, n_align), 2),
        "bl_mask_match_pct": round(100*cnt("bl_mask_match")/max(1, n_align), 2),
        "inj_mask_match_pct": round(100*cnt("inj_mask_match")/max(1, n_align), 2),
        "bl_per_mean_std": avg_per("bl_per"),
        "inj_per_mean_std": avg_per("inj_per"),
        "bl_mask_per_mean_std": avg_per("bl_mask_per"),
        "inj_mask_per_mean_std": avg_per("inj_mask_per"),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== Crossattn entity-PER {args.lang} N={n_align} ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
