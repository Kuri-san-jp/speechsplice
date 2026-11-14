#!/usr/bin/env python3
"""T5Gemma-only JA cross-culture name pair evaluation.

Mirrors eval_en_crossculture_t5gemma_only.py but for Japanese:
  - SOURCE: Chinese single-character family names embedded in Japanese sentences (王, 李, ...)
  - TARGET: Japanese family names in Hanzi (山田, 田中, ...)
  - 30 source x target x 10 templates = 300 pairs

Computes acoustic divergence (MCD, MEL DTW path-mean cos sim) on the entity
slice and Whisper-transcript direction-of-effect rate (target-name in INJ
transcript).

Usage:
  SLURM_TIME=04:00:00 slurm_run.sh \\
    python scripts/projection/eval_zh_crossculture_t5gemma_only.py \\
      --n 300 --target_valid 300 \\
      --output_dir eval_results/ja_crossculture_t5gemma_only_v1
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

import torch, numpy as np
from forced_alignment_utils import WhisperAligner, find_entity_chunks, slice_entity_audio
from eval_table3_acoustic_common import AcousticFeatureExtractor, acoustic_pair_metrics

SEED = 42
LANG = "ja"


def seed_all(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--target_valid", type=int, default=300)
    p.add_argument("--output_dir", default="eval_results/ja_crossculture_t5gemma_only_v1")
    p.add_argument("--save_wav_dir", default=None,
                   help="If set, save BL/INJ/TGT wavs as s{pidx:04d}_{bl,inj,tgt}.wav")
    args = p.parse_args()
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"
    seed_all(SEED)

    from ja_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()[: args.n + 50]
    print(f"[Info] Loaded {len(pairs)} JA cross-culture pairs (T5Gemma only)")

    feat = AcousticFeatureExtractor(device)
    aligner = WhisperAligner(device=device)

    def entity_features_and_text(audio, sr_in, entity, input_text):
        chunks, audio_np = aligner.transcribe_words(audio, sr_in, language=LANG)
        if not chunks or audio_np is None: return None, ""
        full_text = " ".join(c.get("word", "") for c in chunks).strip()
        s_idx, e_idx = find_entity_chunks(chunks, entity, LANG, input_text=input_text)
        if s_idx is None: return None, full_text
        ent_audio = slice_entity_audio(audio_np, aligner.target_sr, chunks, s_idx, e_idx)
        if ent_audio is None or ent_audio.size == 0: return None, full_text
        return feat.compute(ent_audio), full_text

    def in_text(name, transcript):
        # Hanzi substring match; transcripts may include extra whitespace
        return name in transcript.replace(" ", "")

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
    audio_tok = AudioTokenizer(backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    add_bos = 1 if int(getattr(t5_model.args, "add_bos_to_text", 0)) else 0

    def t5_synth(text, modified_hs=None, attn=None, pos_ids=None):
        seed_everything(SEED)
        if modified_hs is None:
            enc = get_encoder_output(t5_model, t5_tok, text, device, lang=LANG)
            modified_hs = enc["last_hidden_state"]
            attn = torch.ones((1, modified_hs.shape[1]), dtype=torch.long, device=device)
            if int(getattr(t5_model.args, "use_pm_rope", 1)):
                x_lens = torch.LongTensor([modified_hs.shape[1]]).to(device)
                pos_ids = t5_model._build_position_ids(x_lens, modified_hs.shape[1], device)
        return generate_with_modified_encoder(
            t5_model, t5_tok, audio_tok, modified_hs, attn, pos_ids,
            text, device, codec_audio_sr, codec_sr,
        )

    def t5_synth_inj(text_a, text_b, word_a, word_b):
        seed_everything(SEED)
        src_enc = get_encoder_output(t5_model, t5_tok, text_a, device, lang=LANG)
        ref_enc = get_encoder_output(t5_model, t5_tok, text_b, device, lang=LANG)
        s_r = find_entity_token_range(t5_tok, src_enc["text"], word_a, add_bos=add_bos)
        r_r = find_entity_token_range(t5_tok, ref_enc["text"], word_b, add_bos=add_bos)
        if s_r is None or r_r is None: return None
        s_s, s_e = s_r; r_s, r_e = r_r
        modified = src_enc["last_hidden_state"].clone()
        if (s_e - s_s) == (r_e - r_s):
            modified[0, s_s:s_e] = ref_enc["last_hidden_state"][0, r_s:r_e]
        else:
            prefix = modified[0, :s_s]
            ref_ent = ref_enc["last_hidden_state"][0, r_s:r_e]
            suffix = modified[0, s_e:]
            modified = torch.cat([prefix, ref_ent, suffix], dim=0).unsqueeze(0)
        attn = torch.ones((1, modified.shape[1]), dtype=torch.long, device=device)
        pos_ids = None
        if int(getattr(t5_model.args, "use_pm_rope", 1)):
            x_lens = torch.LongTensor([modified.shape[1]]).to(device)
            pos_ids = t5_model._build_position_ids(x_lens, modified.shape[1], device)
        return t5_synth(text_a, modified_hs=modified, attn=attn, pos_ids=pos_ids)

    rows = []
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
    done = {r["pair_idx"] for r in rows if r.get("t5_aligned") is True}
    t0 = time.time()

    print(f"[T5Gemma] target_valid={args.target_valid}")
    for i, pair in enumerate(pairs):
        pair_idx = pair.get("pair_idx", i)
        if pair_idx in done: continue
        n_aligned = sum(1 for r in rows if r.get("t5_aligned") is True)
        if n_aligned >= args.target_valid: break
        try:
            t5_bl  = t5_synth(pair["text_a"])
            t5_tgt = t5_synth(pair["text_b"])  # NEW: target audio (text_b synthesised directly)
            t5_inj = t5_synth_inj(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"])
            if t5_bl is None or t5_inj is None or t5_tgt is None: continue
            # Save raw wavs for downstream whole-utterance MCD/DTW analysis
            if args.save_wav_dir:
                import torchaudio
                os.makedirs(args.save_wav_dir, exist_ok=True)
                def _sv(a, sr, p):
                    a = a if a.dim() > 1 else a.unsqueeze(0)
                    torchaudio.save(p, a.cpu(), sr, encoding="PCM_S", bits_per_sample=16)
                _sv(t5_bl[0],  codec_audio_sr, os.path.join(args.save_wav_dir, f"s{pair_idx:04d}_bl.wav"))
                _sv(t5_tgt[0], codec_audio_sr, os.path.join(args.save_wav_dir, f"s{pair_idx:04d}_tgt.wav"))
                _sv(t5_inj[0], codec_audio_sr, os.path.join(args.save_wav_dir, f"s{pair_idx:04d}_inj.wav"))
            # FIXED slicing: BL@word_a, TGT@word_b, INJ@word_b (success) + INJ@word_a (failure)
            f_bl,  txt_bl  = entity_features_and_text(t5_bl[0],  codec_audio_sr, pair["word_a"], pair["text_a"])
            f_tgt, txt_tgt = entity_features_and_text(t5_tgt[0], codec_audio_sr, pair["word_b"], pair["text_b"])
            f_inj_b, txt_inj = entity_features_and_text(t5_inj[0], codec_audio_sr, pair["word_b"], pair["text_a"])
            f_inj_a, _       = entity_features_and_text(t5_inj[0], codec_audio_sr, pair["word_a"], pair["text_a"])
            row = {
                "pair_idx": pair_idx,
                "word_a": pair["word_a"], "word_b": pair["word_b"],
                "t5_bl_transcript": txt_bl,
                "t5_tgt_transcript": txt_tgt,
                "t5_inj_transcript": txt_inj,
                "t5_source_in_bl": in_text(pair["word_a"], txt_bl),
                "t5_target_in_tgt": in_text(pair["word_b"], txt_tgt),
                "t5_target_in_inj": in_text(pair["word_b"], txt_inj),
                "t5_source_in_inj": in_text(pair["word_a"], txt_inj),
            }
            # SUCCESS: INJ@word_b vs TGT@word_b (HIGH ≈ injection succeeded)
            if f_inj_b is not None and f_tgt is not None:
                m = acoustic_pair_metrics(f_inj_b, f_tgt)
                for k, v in m.items():
                    row[f"t5_inj_vs_tgt_{k}"] = round(v, 4) if v is not None else None
                row["t5_aligned_inj_vs_tgt"] = True
            else:
                row["t5_aligned_inj_vs_tgt"] = False
            # FAILURE: INJ@word_a vs BL@word_a (HIGH ≈ injection failed)
            if f_inj_a is not None and f_bl is not None:
                m = acoustic_pair_metrics(f_inj_a, f_bl)
                for k, v in m.items():
                    row[f"t5_inj_vs_bl_{k}"] = round(v, 4) if v is not None else None
                row["t5_aligned_inj_vs_bl"] = True
            else:
                row["t5_aligned_inj_vs_bl"] = False
            row["t5_aligned"] = row["t5_aligned_inj_vs_tgt"] or row["t5_aligned_inj_vs_bl"]
            rows.append(row); done.add(pair_idx)
        except Exception as e:
            print(f"  [{i}] T5 fail: {str(e)[:200]}"); continue
        if (i + 1) % 5 == 0:
            n_aligned = sum(1 for r in rows if r.get("t5_aligned") is True)
            n_tgt = sum(1 for r in rows if r.get("t5_target_in_inj"))
            n_has_text = sum(1 for r in rows if r.get("t5_inj_transcript", "").strip())
            print(f"  [{len(rows)}/{args.n}] aligned={n_aligned} target_in_inj={n_tgt} has_transcript={n_has_text}")
        if (i + 1) % 25 == 0:
            with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)
    with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    aligned = [r for r in rows if r.get("t5_aligned") is True]
    def avg(rs, key):
        vs = [r[key] for r in rs if r.get(key) is not None]
        return [round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)] if vs else [None, None]

    summary = {
        "lang": "ja_crossculture_t5only",
        "n_total": len(rows),
        "n_t5_aligned": len(aligned),
        "t5_target_in_inj_rate": round(sum(1 for r in aligned if r.get("t5_target_in_inj")) / max(1, len(aligned)), 4),
        "t5_target_in_inj_rate_full": round(sum(1 for r in rows if r.get("t5_target_in_inj") is True) / max(1, len(rows)), 4),
        "t5_source_in_bl_rate": round(sum(1 for r in aligned if r.get("t5_source_in_bl")) / max(1, len(aligned)), 4),
    }
    for prefix in ("t5_inj_vs_tgt", "t5_inj_vs_bl"):
        for k in ["dtw_mel","dtw_w2v","dtw_pps","tmean_mel","tmean_w2v","tmean_pps","mcd"]:
            summary[f"{prefix}_{k}"] = avg(aligned, f"{prefix}_{k}")
    summary["elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== T5Gemma-only ZH cross-culture N_aligned={len(aligned)} ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
