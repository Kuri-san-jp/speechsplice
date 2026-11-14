"""SpeechT5 cross-culture injection with BL/TGT/INJ wav saving for case A.

Extends the earlier SpeechT5-only EN cross-culture script:
  - Adds --lang {en, ja, zh} support (loads matching dataset).
  - Generates BL (text_a), TGT (text_b), and INJ (text_a w/ hooked encoder).
  - Saves all three wavs as s{pidx:04d}_{bl,tgt,inj}.wav under --save_wav_dir.

SpeechT5 is fundamentally an English TTS model; running it on JA/ZH text
produces gibberish (English-sounding) audio, which is the intended outcome
for the cross-architecture comparison (paper's "SpeechT5 JA/ZH collapse"
claim).
"""
import argparse, json, os, sys, time, random
import numpy as np
import torch
import torchaudio

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

os.environ.setdefault("HF_HOME", "/mnt/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/mnt/.cache/torch")

SEED = 42


def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_pairs(lang, n):
    if lang == "en":
        from en_cross_culture_names import build_300_pairs
    elif lang == "ja":
        from ja_cross_culture_names import build_300_pairs
    elif lang == "zh":
        from zh_cross_culture_names import build_300_pairs
    else:
        raise ValueError(lang)
    return build_300_pairs()[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--target_valid", type=int, default=300)
    p.add_argument("--s5_xvector_idx", type=int, default=7306)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--save_wav_dir", default=None,
                   help="Save BL/INJ/TGT wavs for downstream case A MCD.")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"eval_results/speecht5_cc_wavs_{args.lang}_n{args.n}"
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_wav_dir:
        os.makedirs(args.save_wav_dir, exist_ok=True)
    device = "cuda"
    seed_all(SEED)

    pairs = load_pairs(args.lang, args.n)
    print(f"[Info] Loaded {len(pairs)} {args.lang} cross-culture pairs (SpeechT5 xvec={args.s5_xvector_idx})")

    print("[Info] Loading SpeechT5 ...")
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    s5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    s5_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device).eval()
    s5_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    from datasets import load_dataset
    try:
        emb_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        s5_speaker = torch.tensor(emb_ds[args.s5_xvector_idx]["xvector"]).unsqueeze(0).to(device)
    except Exception:
        s5_speaker = torch.randn(1, 512).to(device) * 0.1

    SR = 16000

    def s5_get_enc(text):
        inputs = s5_processor(text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            enc = s5_model.speecht5.encoder(
                input_values=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return enc.last_hidden_state, inputs["input_ids"]

    def s5_find_ent(entity, ids):
        ent_ids = s5_processor.tokenizer(entity, add_special_tokens=False)["input_ids"]
        idl = ids[0].cpu().tolist()
        for i in range(len(idl) - len(ent_ids) + 1):
            if idl[i:i+len(ent_ids)] == ent_ids: return i, i + len(ent_ids)
        ent_ids_sp = s5_processor.tokenizer(" " + entity, add_special_tokens=False)["input_ids"]
        for i in range(len(idl) - len(ent_ids_sp) + 1):
            if idl[i:i+len(ent_ids_sp)] == ent_ids_sp: return i, i + len(ent_ids_sp)
        return None

    def make_hook(modified):
        def hook(module, input, output):
            if hasattr(output, "last_hidden_state"):
                output.last_hidden_state = modified
            elif isinstance(output, tuple) and output:
                output = (modified,) + output[1:]
            return output
        return hook

    def s5_synth(text):
        seed_all(SEED)
        with torch.no_grad():
            inputs = s5_processor(text=text, return_tensors="pt").to(device)
            if inputs["input_ids"].shape[1] == 0: return None
            audio = s5_model.generate_speech(inputs["input_ids"], s5_speaker, vocoder=s5_vocoder)
        return audio

    def s5_synth_inj(text_a, text_b, word_a, word_b):
        seed_all(SEED)
        h_a, ids_a = s5_get_enc(text_a)
        h_b, ids_b = s5_get_enc(text_b)
        sr = s5_find_ent(word_a, ids_a)
        rr = s5_find_ent(word_b, ids_b)
        if sr is None or rr is None: return None
        ss, se = sr; rs, re = rr
        src_len = se - ss
        tgt_len = re - rs
        modified = h_a.clone()
        if src_len == tgt_len:
            modified[0, ss:se] = h_b[0, rs:re]
        else:
            import torch.nn.functional as F
            ref_ent = h_b[0, rs:re].unsqueeze(0).transpose(1, 2)
            ref_ent_resampled = F.adaptive_avg_pool1d(ref_ent, src_len).transpose(1, 2)
            modified[0, ss:se] = ref_ent_resampled[0].to(modified.dtype)
        seed_all(SEED)
        handle = s5_model.speecht5.encoder.register_forward_hook(make_hook(modified))
        try:
            with torch.no_grad():
                audio = s5_model.generate_speech(ids_a, s5_speaker, vocoder=s5_vocoder)
        finally:
            handle.remove()
        return audio

    rows = []
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
    done = {r["pair_idx"] for r in rows if r.get("ok") is True}
    t0 = time.time()
    print(f"[Info] resume from {len(done)} done rows")

    def _sv(a, p):
        if a is None: return False
        if a.dim() < 2: a = a.unsqueeze(0)
        torchaudio.save(p, a.cpu(), SR, encoding="PCM_S", bits_per_sample=16)
        return True

    for i, pair in enumerate(pairs):
        pair_idx = pair.get("pair_idx", i)
        if pair_idx in done: continue
        if len([r for r in rows if r.get("ok")]) >= args.target_valid: break
        try:
            s5_bl  = s5_synth(pair["text_a"])
            s5_tgt = s5_synth(pair["text_b"])
            s5_inj = s5_synth_inj(pair["text_a"], pair["text_b"], pair["word_a"], pair["word_b"])
            ok = (s5_bl is not None) and (s5_tgt is not None) and (s5_inj is not None)
            row = {"pair_idx": pair_idx,
                   "word_a": pair["word_a"], "word_b": pair["word_b"],
                   "ok": bool(ok)}
            if ok and args.save_wav_dir:
                bl_p  = os.path.join(args.save_wav_dir, f"s{pair_idx:04d}_bl.wav")
                tgt_p = os.path.join(args.save_wav_dir, f"s{pair_idx:04d}_tgt.wav")
                inj_p = os.path.join(args.save_wav_dir, f"s{pair_idx:04d}_inj.wav")
                _sv(s5_bl, bl_p); _sv(s5_tgt, tgt_p); _sv(s5_inj, inj_p)
                row["saved"] = True
            rows.append(row); done.add(pair_idx)
        except Exception as e:
            rows.append({"pair_idx": pair_idx, "ok": False, "err": str(e)[:200]})
        if (i + 1) % 25 == 0:
            n_ok = sum(1 for r in rows if r.get("ok"))
            print(f"  [{len(rows)}/{args.n}] ok={n_ok}", flush=True)
            with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)

    with open(ckpt, "w") as f: json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)
    summary = {
        "lang": args.lang, "n_total": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_saved": sum(1 for r in rows if r.get("saved")),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=== SpeechT5 cross-culture {args.lang} N={summary['n_total']} ok={summary['n_ok']} saved={summary['n_saved']} ===")


if __name__ == "__main__":
    main()
