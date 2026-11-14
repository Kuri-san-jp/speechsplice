#!/usr/bin/env python3
"""JA numerals v2 の音素レベル検証（誤読特定→修正率測定）.

方式（Table A1 と同じ参照合成方式）:
  1. 各ペアについて参照音声を T5Gemma で直接合成
       ref_c = text_b（正読みカナ文）の合成
       ref_w = text_a の数詞を誤読候補カナに置換した文の合成
  2. wav2vec2-XLSR-espeak で BL / INJ / ref_c / ref_w を IPA 転写
  3. whole-sentence PER で BL・INJ を correct / wrong / ambiguous に分類
       （文は数詞領域以外同一なので判別信号は数詞領域に集中する）
  4. Set A: BL誤読率、誤読サブセットでの INJ 修正率、正読サブセットでの保存率
     Set B: INJ が注入読みに追従した率

Usage:
  SLURM_TIME=01:00:00 slurm_run.sh \\
    python scripts/analysis/ja_numerals_v2_verify.py
"""
import json, os, sys

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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch, torchaudio, numpy as np

GEN_DIR = os.path.join(PROJECT_ROOT, "eval_results", "ja_numerals_v2_gen")
OUT_JSON = os.path.join(GEN_DIR, "phoneme_verify.json")
PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MARGIN = 0.02  # PER差がこの範囲内なら ambiguous


def local_snapshot(hub_id):
    """HF_HOME 配下のローカルスナップショットを解決（オフライン対応）.

    refs/main を優先し、無ければ config.json を含むスナップショットを選ぶ
    （safetensorsのみの不完全スナップショットを避ける）。
    """
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ["HF_HOME"], "hub")
    base = os.path.join(hub, f"models--{hub_id.replace('/', '--')}")
    ref = os.path.join(base, "refs", "main")
    snap_dir = os.path.join(base, "snapshots")
    if os.path.isfile(ref):
        p = os.path.join(snap_dir, open(ref).read().strip())
        if os.path.isdir(p):
            return p
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        for s in reversed(snaps):
            if os.path.exists(os.path.join(snap_dir, s, "config.json")):
                return os.path.join(snap_dir, s)
        if snaps:
            return os.path.join(snap_dir, snaps[-1])
    return hub_id


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def main():
    from ja_numerals_v2 import build_pairs
    pairs = build_pairs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(PROJECT_ROOT)

    # --- 1. 参照音声の合成（resumable） -----------------------------------
    need_synth = []
    for p in pairs:
        i = p["pair_idx"]
        refs = [("ref_c", p["text_b"])]
        if p["wrong_kana"]:
            refs.append(("ref_w", p["text_a"].replace(p["word_a"], p["wrong_kana"][0])))
        for tag, text in refs:
            fn = os.path.join(GEN_DIR, f"{tag}_s{i:04d}.wav")
            if not os.path.exists(fn):
                need_synth.append((fn, text))

    if need_synth:
        print(f"[Info] Synthesizing {len(need_synth)} reference clips ...")
        import eval_utmos_speaker_fixed as ev
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from data.tokenizer import AudioTokenizer
        from inference_tts_utils import save_audio, transcribe_audio

        ev.seed_all(ev.SEED)
        ref_wav = ev.DEFAULT_REF
        ref_prefix = transcribe_audio(ref_wav, device)
        print(f"[Info] REF_PREFIX: {ref_prefix}")

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

        for k, (fn, text) in enumerate(need_synth):
            try:
                audio = ev.voice_clone_synth(
                    model, text_tok, audio_tok,
                    target_text=text, lang="ja", device=device,
                    codec_audio_sr=codec_audio_sr, codec_sr=codec_sr,
                    ref_wav=ref_wav, ref_text=ref_prefix,
                    ref_prompt_sec=100.0, modified_hidden=None)
                save_audio(fn, audio if audio.dim() > 1 else audio.unsqueeze(0), codec_audio_sr)
            except Exception as e:
                print(f"  [{k}] synth fail {os.path.basename(fn)}: {str(e)[:100]}", flush=True)
            if (k + 1) % 10 == 0:
                print(f"  [{k+1}/{len(need_synth)}]", flush=True)
        del model, audio_tok
        torch.cuda.empty_cache()
    else:
        print("[Info] All reference clips already exist.")

    # --- 2. 音素ASR --------------------------------------------------------
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
        return [t for t in " ".join(toks).replace("|", " ").split() if t]

    def classify(ipa_x, ipa_c, ipa_w):
        """PER基準で correct / wrong / ambiguous を判定."""
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
        bl_fn = os.path.join(GEN_DIR, f"s{i:04d}_bl.wav")
        inj_fn = os.path.join(GEN_DIR, f"s{i:04d}_inj.wav")
        rc_fn = os.path.join(GEN_DIR, f"ref_c_s{i:04d}.wav")
        rw_fn = os.path.join(GEN_DIR, f"ref_w_s{i:04d}.wav")
        if not (os.path.exists(bl_fn) and os.path.exists(inj_fn) and os.path.exists(rc_fn)):
            print(f"[{i}] missing wav, skip"); continue
        ipa_c = to_ipa(rc_fn)
        ipa_w = to_ipa(rw_fn) if os.path.exists(rw_fn) else None
        bl_lab, bl_pc, bl_pw = classify(to_ipa(bl_fn), ipa_c, ipa_w)
        inj_lab, inj_pc, inj_pw = classify(to_ipa(inj_fn), ipa_c, ipa_w)
        rows.append({
            "pair_idx": i, "set": p["set"], "category": p["category"],
            "surface": p["word_a"], "kana": p["correct_kana"],
            "wrong_kana": p["wrong_kana"],
            "bl": bl_lab, "bl_per_c": round(bl_pc, 3),
            "bl_per_w": round(bl_pw, 3) if bl_pw is not None else None,
            "inj": inj_lab, "inj_per_c": round(inj_pc, 3),
            "inj_per_w": round(inj_pw, 3) if inj_pw is not None else None,
        })
        print(f"[{i:2d}] {p['set']} {p['word_a']:>6}({p['correct_kana']}) "
              f"BL={bl_lab}({bl_pc:.2f}/{bl_pw if bl_pw is None else round(bl_pw,2)}) "
              f"INJ={inj_lab}({inj_pc:.2f}/{inj_pw if inj_pw is None else round(inj_pw,2)})",
              flush=True)

    # --- 3. 集計 ------------------------------------------------------------
    a = [r for r in rows if r["set"] == "A"]
    b = [r for r in rows if r["set"] == "B"]
    a_wrong = [r for r in a if r["bl"] == "wrong"]
    a_correct = [r for r in a if r["bl"] == "correct"]
    summary = {
        "n_total": len(rows), "n_A": len(a), "n_B": len(b),
        "A_bl_wrong": len(a_wrong), "A_bl_correct": len(a_correct),
        "A_bl_ambiguous": len(a) - len(a_wrong) - len(a_correct),
        "A_corrected_of_wrong": sum(1 for r in a_wrong if r["inj"] == "correct"),
        "A_preserved_of_correct": sum(1 for r in a_correct if r["inj"] == "correct"),
        "B_follows_injected": sum(1 for r in b if r["inj"] == "correct"),
        "B_bl_pref_injected": sum(1 for r in b if r["bl"] == "correct"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[Done] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
