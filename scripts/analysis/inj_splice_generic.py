#!/usr/bin/env python3
"""指定ペア集合の INJ を splice 方式（声まね込み）で再生成する共通スクリプト（GPU）.

numerals_v3_inj_splice.py の一般化。data/<pairs_module>.py の build_pairs() を読み、
各ペアの INJ を splice 方式（scripts/projection/splice_clone.py、duration=text_a 固定）
で生成して out_dir に保存する。既存音声は上書きしない（新しい出力先を使うこと）。

--only_unequal を付けると、word_a と word_b のトークンスパン長が異なるペアのみ
生成する（等長ペアは splice と旧方式が同一処理のため再生成不要）。

Usage 例:
  SLURM_TIME=00:40:00 slurm_run.sh \
    python scripts/analysis/inj_splice_generic.py \
      --pairs_module en_wordrepl300_pairs --lang en --n 300 --only_unequal \
      --ref_wav eval_results/_natural_refs_norm/en_natural_norm.wav \
      --out_dir eval_results/en_wordrepl300_splice_inj
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
_hf = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "analysis"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch

_ap = argparse.ArgumentParser()
_ap.add_argument("--pairs_module", required=True)
_ap.add_argument("--lang", required=True)
_ap.add_argument("--n", type=int, default=None)
_ap.add_argument("--out_dir", required=True)
_ap.add_argument("--ref_wav", default=None,
                 help="参照音声（既定: eval_utmos_speaker_fixed.DEFAULT_REF = JSUT 0011）")
_ap.add_argument("--only_unequal", action="store_true",
                 help="トークンスパン長が異なるペアのみ生成")
_args = _ap.parse_args()


def main():
    import eval_utmos_speaker_fixed as ev
    from splice_clone import prepare_clone_splice
    from ja_numerals_v2_verify import local_snapshot
    from inference_tts_utils import save_audio, transcribe_audio
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from data.tokenizer import AudioTokenizer

    mod = __import__(_args.pairs_module)
    pairs = mod.build_pairs()
    if _args.n:
        pairs = pairs[:_args.n]

    out_dir = os.path.join(PROJECT_ROOT, _args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev.seed_all(ev.SEED)
    ref_wav = (os.path.join(PROJECT_ROOT, _args.ref_wav) if _args.ref_wav
               else ev.DEFAULT_REF)
    ref_prefix = transcribe_audio(ref_wav, device)
    print(f"[Info] ref={ref_wav}", flush=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_snapshot("Aratako/T5Gemma-TTS-2b-2b"), trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.eval()
    cfg = model.config
    tok_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tok = AutoTokenizer.from_pretrained(local_snapshot(tok_name), local_files_only=True)
    audio_tok = AudioTokenizer(backend="xcodec2",
                               model_name=getattr(cfg, "xcodec2_model_name", None))
    codec_audio_sr = audio_tok.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    meta_rows, done, skipped_eq = [], 0, 0
    for i, p in enumerate(pairs):
        pidx = int(p.get("pair_idx", i))
        out_fn = os.path.join(out_dir, f"s{pidx:04d}_inj.wav")
        if os.path.exists(out_fn) and os.path.getsize(out_fn) > 1000:
            done += 1
            continue
        ev.seed_all(ev.SEED)
        try:
            src_enc = ev.get_encoder_output_with_prefix(
                model, text_tok, ref_prefix, p["text_a"], device, lang=_args.lang)
            ref_enc = ev.get_encoder_output_with_prefix(
                model, text_tok, ref_prefix, p["text_b"], device, lang=_args.lang)
            s_sub = ev.find_entity_token_range(
                text_tok, src_enc["target_text"], p["word_a"], add_bos=0)
            r_sub = ev.find_entity_token_range(
                text_tok, ref_enc["target_text"], p["word_b"], add_bos=0)
            if s_sub is None or r_sub is None:
                print(f"  [{pidx}] range None ({p['word_a']}→{p['word_b']}), skip", flush=True)
                continue
            s_r = (s_sub[0] + src_enc["target_start_in_full"],
                   s_sub[1] + src_enc["target_start_in_full"])
            r_r = (r_sub[0] + ref_enc["target_start_in_full"],
                   r_sub[1] + ref_enc["target_start_in_full"])
            if _args.only_unequal and (s_r[1] - s_r[0]) == (r_r[1] - r_r[0]):
                skipped_eq += 1
                continue
            modified, inj_t, dur_t, delta = prepare_clone_splice(
                text_tok, src_enc, ref_enc, s_r, r_r, _args.lang,
                duration_source_text=p["text_a"],
                natural_candidates=(p["text_b"],))
            inj = ev.voice_clone_synth(
                model, text_tok, audio_tok,
                target_text=(inj_t if inj_t is not None else p["text_a"]),
                lang=_args.lang, device=device,
                codec_audio_sr=codec_audio_sr, codec_sr=codec_sr,
                ref_wav=ref_wav, ref_text=ref_prefix, ref_prompt_sec=100.0,
                modified_hidden=modified, duration_text=dur_t)
            save_audio(out_fn, inj if inj.dim() > 1 else inj.unsqueeze(0), codec_audio_sr)
            done += 1
            meta_rows.append({"pair_idx": pidx, "word_a": p["word_a"],
                              "word_b": p["word_b"], "delta_tok": int(delta)})
            print(f"  [{pidx}] {p['word_a']}→{p['word_b']} Δtok={delta:+d} saved", flush=True)
        except Exception as e:
            print(f"  [{pidx}] fail: {str(e)[:120]}", flush=True)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"align": "splice", "pairs_module": _args.pairs_module,
                   "lang": _args.lang, "ref_wav": ref_wav,
                   "only_unequal": _args.only_unequal, "rows": meta_rows},
                  f, ensure_ascii=False, indent=1)
    print(f"[Done] generated {len(meta_rows)} (既存 {done - len(meta_rows)}, "
          f"等長スキップ {skipped_eq})", flush=True)


if __name__ == "__main__":
    main()
