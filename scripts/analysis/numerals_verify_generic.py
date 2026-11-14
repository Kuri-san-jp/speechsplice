#!/usr/bin/env python3
"""数詞実験の音素レベル検証（汎用版: JA/ZH共用）.

ja_numerals_v2_verify.py の一般化。ペアモジュールに ref_c_text / ref_w_text が
あればそれを参照合成に使い、無ければ JA 方式（text_b / wrong_kana置換）に
フォールバックする。方式は Table A1 と同じ参照合成 + wav2vec2-XLSR IPA +
whole-sentence PER 分類。

Usage:
  SLURM_TIME=01:00:00 slurm_run.sh \\
    python scripts/analysis/numerals_verify_generic.py --module zh_numerals_v1 --lang zh
"""
import argparse, json, os, sys

for _p in ("/tmp/.cache/huggingface/token", os.path.expanduser("~/.cache/huggingface/token")):
    if os.path.exists(_p):
        try:
            with open(_p) as _f:
                os.environ.setdefault("HF_TOKEN", _f.read().strip())
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
            break
        except Exception:
            pass

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
T5GEMMA_ROOT = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
sys.path.insert(0, T5GEMMA_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "analysis"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch, torchaudio, numpy as np

from ja_numerals_v2_verify import local_snapshot, levenshtein  # 再利用

PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MARGIN = 0.02


def ref_texts(p):
    """ペアから (ref_c_text, ref_w_text or None) を得る."""
    rc = p.get("ref_c_text") or p["text_b"]
    rw = p.get("ref_w_text")
    if rw is None and p.get("wrong_kana"):
        rw = p["text_a"].replace(p["word_a"], p["wrong_kana"][0])
    return rc, rw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--gen_dir", default=None)
    ap.add_argument("--inj_dir", default=None,
                    help="INJ音声を別フォルダから読む（例: splice再生成分）")
    ap.add_argument("--inj_suffix", default="inj",
                    help="INJファイル名の条件部（例: injsc なら s%%04d_injsc.wav）")
    ap.add_argument("--out_name", default="phoneme_verify.json",
                    help="出力ファイル名（既存の結果を上書きしないため）")
    args = ap.parse_args()
    gen_dir = args.gen_dir or os.path.join("eval_results", f"{args.module}_gen")
    # コンテナ内では /data/... がread-onlyのため、必ずPROJECT_ROOT基準で解決する
    def _abspath(d):
        if not os.path.isabs(d):
            return os.path.join(PROJECT_ROOT, d)
        if not d.startswith(PROJECT_ROOT):
            rel = d.split("eval_results/", 1)[-1]
            return os.path.join(PROJECT_ROOT, "eval_results", rel)
        return d
    gen_dir = _abspath(gen_dir)
    inj_dir = _abspath(args.inj_dir) if args.inj_dir else gen_dir
    os.makedirs(gen_dir, exist_ok=True)
    out_json = os.path.join(gen_dir, args.out_name)

    mod = __import__(args.module)
    pairs = mod.build_pairs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(PROJECT_ROOT)

    # --- 1. 参照音声の合成（resumable） -----------------------------------
    need = []
    for p in pairs:
        i = p["pair_idx"]
        rc, rw = ref_texts(p)
        # BL（text_aの通常合成）も未生成なら合成対象に含める（既存ならスキップ）
        for tag, text in ([("bl", p["text_a"]), ("ref_c", rc)]
                          + ([("ref_w", rw)] if rw else [])):
            fn = (os.path.join(gen_dir, f"s{i:04d}_bl.wav") if tag == "bl"
                  else os.path.join(gen_dir, f"{tag}_s{i:04d}.wav"))
            if not os.path.exists(fn):
                need.append((fn, text))

    if need:
        print(f"[Info] Synthesizing {len(need)} reference clips ...")
        import eval_utmos_speaker_fixed as ev
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from data.tokenizer import AudioTokenizer
        from inference_tts_utils import save_audio, transcribe_audio

        ev.seed_all(ev.SEED)
        ref_wav = ev.DEFAULT_REF
        ref_prefix = transcribe_audio(ref_wav, device)
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
        for k, (fn, text) in enumerate(need):
            try:
                audio = ev.voice_clone_synth(
                    model, text_tok, audio_tok,
                    target_text=text, lang=args.lang, device=device,
                    codec_audio_sr=codec_audio_sr, codec_sr=codec_sr,
                    ref_wav=ref_wav, ref_text=ref_prefix,
                    ref_prompt_sec=100.0, modified_hidden=None)
                save_audio(fn, audio if audio.dim() > 1 else audio.unsqueeze(0), codec_audio_sr)
            except Exception as e:
                print(f"  [{k}] synth fail {os.path.basename(fn)}: {str(e)[:100]}", flush=True)
            if (k + 1) % 10 == 0:
                print(f"  [{k+1}/{len(need)}]", flush=True)
        del model, audio_tok
        torch.cuda.empty_cache()
    else:
        print("[Info] All reference clips already exist.")

    # --- 2. 音素ASR + 分類 --------------------------------------------------
    print(f"[Info] Loading phoneme ASR: {PHONEME_MODEL}")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    import soundfile as sf
    ph_path = local_snapshot(PHONEME_MODEL)
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(ph_path, local_files_only=True)
    asr_model = Wav2Vec2ForCTC.from_pretrained(ph_path, local_files_only=True).to(device).eval()
    vocab = json.load(open(os.path.join(ph_path, "vocab.json")))
    id2sym = {v: k for k, v in vocab.items()}
    pad_id = vocab.get("<pad>", 0)

    def to_ipa(path):
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        t = torch.from_numpy(wav).float()
        if sr != feat_ext.sampling_rate:
            t = torchaudio.functional.resample(t, sr, feat_ext.sampling_rate)
        with torch.no_grad():
            x = feat_ext(t.numpy(), sampling_rate=feat_ext.sampling_rate,
                         return_tensors="pt").input_values.to(device)
            pred = torch.argmax(asr_model(x).logits, dim=-1)[0].cpu().tolist()
        toks, prev = [], None
        for i in pred:
            if i == pad_id:
                prev = i; continue
            if i == prev:
                continue
            toks.append(id2sym.get(i, "")); prev = i
        return [s for s in " ".join(toks).replace("|", " ").split() if s]

    def classify(ipa_x, ipa_c, ipa_w):
        per_c = levenshtein(ipa_x, ipa_c) / max(1, len(ipa_c))
        if ipa_w is None:
            return ("correct" if per_c < 0.3 else "ambiguous"), per_c, None
        per_w = levenshtein(ipa_x, ipa_w) / max(1, len(ipa_w))
        if per_c + MARGIN < per_w:
            lab = "correct"
        elif per_w + MARGIN < per_c:
            lab = "wrong"
        else:
            lab = "ambiguous"
        return lab, per_c, per_w

    rows = []
    for p in pairs:
        i = p["pair_idx"]
        bl_fn = os.path.join(gen_dir, f"s{i:04d}_bl.wav")
        inj_fn = os.path.join(inj_dir, f"s{i:04d}_{args.inj_suffix}.wav")
        rc_fn = os.path.join(gen_dir, f"ref_c_s{i:04d}.wav")
        rw_fn = os.path.join(gen_dir, f"ref_w_s{i:04d}.wav")
        # INJ はオプション（BLのみのスクリーニング実行を許容）
        if not (os.path.exists(bl_fn) and os.path.exists(rc_fn)):
            print(f"[{i}] missing wav, skip"); continue
        ipa_c = to_ipa(rc_fn)
        ipa_w = to_ipa(rw_fn) if os.path.exists(rw_fn) else None
        bl_lab, bl_pc, bl_pw = classify(to_ipa(bl_fn), ipa_c, ipa_w)
        if os.path.exists(inj_fn):
            inj_lab, inj_pc, inj_pw = classify(to_ipa(inj_fn), ipa_c, ipa_w)
        else:
            inj_lab, inj_pc, inj_pw = None, None, None
        rows.append({
            "pair_idx": i, "set": p.get("set", "A"), "category": p.get("category", ""),
            "surface": p["word_a"], "inject": p["word_b"],
            "bl": bl_lab, "bl_per_c": round(bl_pc, 3),
            "bl_per_w": round(bl_pw, 3) if bl_pw is not None else None,
            "inj": inj_lab,
            "inj_per_c": round(inj_pc, 3) if inj_pc is not None else None,
            "inj_per_w": round(inj_pw, 3) if inj_pw is not None else None,
        })
        print(f"[{i:2d}] {p.get('set','A')} {p['word_a']}→{p['word_b']} ({p.get('category','')}) "
              f"BL={bl_lab}({bl_pc:.2f}/{'-' if bl_pw is None else round(bl_pw,2)}) "
              f"INJ={inj_lab}", flush=True)

    a = [r for r in rows if r["set"] == "A"]
    c = [r for r in rows if r["set"] == "C"]
    a_wrong = [r for r in a if r["bl"] == "wrong"]
    a_correct = [r for r in a if r["bl"] == "correct"]
    summary = {
        "module": args.module, "n_total": len(rows), "n_A": len(a), "n_C": len(c),
        "A_bl_wrong": len(a_wrong), "A_bl_correct": len(a_correct),
        "A_bl_ambiguous": len(a) - len(a_wrong) - len(a_correct),
        "A_corrected_of_wrong": sum(1 for r in a_wrong if r["inj"] == "correct"),
        "A_preserved_of_correct": sum(1 for r in a_correct if r["inj"] == "correct"),
        "C_bl_correct": sum(1 for r in c if r["bl"] == "correct"),
        "C_inj_overridden": sum(1 for r in c if r["inj"] == "wrong"),
    }
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[Done] -> {out_json}")


if __name__ == "__main__":
    main()
