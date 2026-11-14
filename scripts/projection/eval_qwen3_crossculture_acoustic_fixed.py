"""FIXED Qwen3-TTS cross-culture acoustic eval.

Fixes for the bias bug in `eval_qwen3_crossculture_acoustic*.py`:
  - BL audio is sliced at word_a position (canonical original reading).
  - INJ audio is sliced at *word_b* position (the injected target reading
    that successful injection should produce).
  - TGT audio = synthesize(text_b) and sliced at word_b position.
  - Metrics computed on (TGT[word_b] vs INJ[word_b_slice]) so HIGH
    similarity means injection SUCCEEDED.

For Qwen3-TTS, voice cloning uses the same natural reference per language
as the rest of Table-3 evaluation, and `x_vector_only_mode=True` (the
working config from the ZH degeneration diagnose).
"""

# 2026-09-27: CPU スレッド上限（1 本あたり 13）。torch などを読む前に設定する。
import os as _os_cap
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os_cap.environ[_v] = "13"
try:
    import torch as _torch_cap
    _torch_cap.set_num_threads(13)
except Exception:
    pass
import os as _os
# slurm_run.sh は HF_HOME=/tmp/.cache/huggingface を渡すが実体が無く書き込めない。
# setdefault だとコンテナ側の値が残るので、明示的に上書きする。
_hf = "/mnt/.cache/huggingface" if _os.path.isdir("/mnt/scripts") else \
      os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
_os.environ["HF_HOME"] = _hf
_os.environ["HF_HUB_CACHE"] = _os.path.join(_hf, "hub")
_os.environ["HF_MODULES_CACHE"] = _os.path.join(_hf, "modules")
import argparse, json, os, sys, time, random
os.environ.setdefault("TORCH_HOME", "/mnt/.cache/torch")
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/projection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

import torch, torchaudio, numpy as np

from forced_alignment_utils import WhisperAligner, find_entity_chunks, slice_entity_audio
from eval_table3_acoustic_common import AcousticFeatureExtractor, acoustic_pair_metrics

LANG_CODE = {"en": "English", "ja": "Japanese", "zh": "Chinese"}
SEED = 42


def seed_all(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--injection_layer", type=int, default=26)
    p.add_argument("--target_valid", type=int, default=300)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--save_wav_dir", default=None,
                   help="If set, save BL/INJ/TGT wavs as s{pidx:04d}_{bl,inj,tgt}.wav")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"eval_results/qwen3_crossculture_acoustic_fixed_{args.lang}_n{args.n}"
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cross-culture pair source (word_a / word_b are different entities)
    if args.lang == "ja":
        from ja_cross_culture_names import build_300_pairs
    elif args.lang == "en":
        from en_cross_culture_names import build_300_pairs
    else:  # zh
        from zh_cross_culture_names import build_300_pairs
    pairs = build_300_pairs()[: args.n + 50]
    print(f"[Info] {args.lang} cross-culture: {len(pairs)} pairs", flush=True)

    feat = AcousticFeatureExtractor(device)
    aligner = WhisperAligner(device=device)

    def slice_at(audio, sr_in, entity, input_text):
        chunks, audio_np = aligner.transcribe_words(audio, sr_in, language=args.lang)
        if not chunks or audio_np is None: return None, "", None
        full_text = " ".join(c.get("word", "") for c in chunks).strip()
        s_idx, e_idx = find_entity_chunks(chunks, entity, args.lang, input_text=input_text)
        if s_idx is None: return None, full_text, None
        ent_audio = slice_entity_audio(audio_np, aligner.target_sr, chunks, s_idx, e_idx)
        if ent_audio is None or ent_audio.size == 0: return None, full_text, None
        return feat.compute(ent_audio), full_text, (s_idx, e_idx)

    def in_text(name, transcript):
        if args.lang in ("ja", "zh"):
            return name in transcript.replace(" ", "")
        return name.lower() in transcript.lower()

    print("[Info] Loading Qwen3-TTS ...", flush=True)
    from qwen_tts import Qwen3TTSModel
    wrapper = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    wrapper.eval() if hasattr(wrapper, "eval") else wrapper.model.eval()
    talker = wrapper.model.talker
    talker_model = talker.model
    processor = wrapper.processor
    config = wrapper.model.config

    # Per-language natural reference (shared across all Table-3 architectures)
    REFS = {
        "en": (f"{PROJECT_ROOT}/eval_results/_natural_refs_norm/en_natural_norm.wav",
               "Let me see, I'll give them a new pair of boots every Christmas."),
        "ja": (f"{PROJECT_ROOT}/eval_results/_natural_refs_norm/ja_natural_norm.wav",
               "太平洋は、世界でいちばん大きな大洋です。"),
        "zh": (f"{PROJECT_ROOT}/eval_results/_natural_refs_norm/zh_natural_norm.wav",
               "印度素有嫁女儿陪送嫁妆的传统。"),
    }
    ref_audio, ref_text = REFS[args.lang]

    print(f"[Info] Building voice clone prompt (x_vector_only_mode=True for stability)...", flush=True)
    prompt_items = wrapper.create_voice_clone_prompt(
        ref_audio=ref_audio, ref_text=ref_text,
        x_vector_only_mode=True,
    )

    from eval_qwen3_injection import build_text_only_inputs, get_layer_hidden_states, find_entity_token_positions

    def gen_audio(text, hook=None):
        seed_all(SEED)
        h_handle = None
        if hook is not None:
            h_handle = talker_model.layers[args.injection_layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                wavs, sr = wrapper.generate_voice_clone(
                    text=text, language=LANG_CODE[args.lang],
                    voice_clone_prompt=prompt_items,
                    max_new_tokens=512,
                )
        finally:
            if h_handle is not None: h_handle.remove()
        wav = wavs[0] if isinstance(wavs, list) else wavs
        a = torch.tensor(wav).float() if isinstance(wav, np.ndarray) else (wav.cpu().float() if isinstance(wav, torch.Tensor) else torch.tensor(wav).float())
        if a.dim() > 1: a = a.squeeze()
        return a, sr

    rows = []
    skip_set = set()
    ckpt = os.path.join(args.output_dir, "checkpoint.json")
    if os.path.exists(ckpt):
        st = json.load(open(ckpt))
        rows = st.get("rows", [])
        skip_set = set(st.get("skip_pairs", []))
    done = {r["pair_idx"] for r in rows}

    for i, pair in enumerate(pairs):
        pidx = pair.get("pair_idx", i)
        if pidx in done or pidx in skip_set: continue
        n_aligned = sum(1 for r in rows if r.get("aligned"))
        if n_aligned >= args.target_valid: break
        try:
            wa_s, wa_e = find_entity_token_positions(pair["text_a"], pair["word_a"], processor, text_body_offset=8)
            wb_s, wb_e = find_entity_token_positions(pair["text_b"], pair["word_b"], processor, text_body_offset=8)
            if None in (wa_s, wa_e, wb_s, wb_e): continue
            tb_emb, tb_mask = build_text_only_inputs(pair["text_b"], talker, talker.text_projection, processor, config, device)
            tb_hidden = get_layer_hidden_states(talker_model, tb_emb, tb_mask, args.injection_layer)
            wb_hidden = tb_hidden[0, wb_s:wb_e+1].clone()
            wa_s_inf, wa_e_inf = find_entity_token_positions(pair["text_a"], pair["word_a"], processor,
                                                               text_body_offset=9 if args.lang == "ja" else 8)

            # BL = text_a (faithful baseline)
            bl_a, bl_sr = gen_audio(pair["text_a"])
            # TGT = text_b (canonical target reading audio)
            tgt_a, tgt_sr = gen_audio(pair["text_b"])

            fired = [False]
            def hook(m, inp, out):
                if fired[0]: return out
                h = out[0] if isinstance(out, tuple) else out
                if h.dim() < 3 or h.shape[1] <= 1: return out
                if wa_e_inf >= h.shape[1]: return out
                wa_len = wa_e_inf - wa_s_inf + 1
                wb_len = wb_hidden.shape[0]
                h_mod = h.clone()
                if wa_len == wb_len:
                    h_mod[0, wa_s_inf:wa_e_inf+1] = wb_hidden.to(h.dtype)
                else:
                    wb_mean = wb_hidden.mean(dim=0)
                    h_mod[0, wa_s_inf:wa_e_inf+1] = wb_mean.unsqueeze(0).expand(wa_len, -1).to(h.dtype)
                fired[0] = True
                return (h_mod,) + out[1:] if isinstance(out, tuple) else h_mod

            inj_a, inj_sr = gen_audio(pair["text_a"], hook=hook)

            # Save raw wavs for downstream whole-utterance MCD/DTW analysis
            if args.save_wav_dir:
                import torchaudio
                os.makedirs(args.save_wav_dir, exist_ok=True)
                def _sv(a, sr, p):
                    a = a.unsqueeze(0) if a.dim() == 1 else a
                    torchaudio.save(p, a.cpu(), sr, encoding="PCM_S", bits_per_sample=16)
                _sv(bl_a,  bl_sr,  os.path.join(args.save_wav_dir, f"s{pidx:04d}_bl.wav"))
                _sv(tgt_a, tgt_sr, os.path.join(args.save_wav_dir, f"s{pidx:04d}_tgt.wav"))
                _sv(inj_a, inj_sr, os.path.join(args.save_wav_dir, f"s{pidx:04d}_inj.wav"))

            # CORRECTED slicing:
            #   BL  → word_a position (in text_a it should be word_a)
            #   TGT → word_b position (in text_b it should be word_b)
            #   INJ → word_b position (if injection succeeded, INJ contains word_b sound)
            f_bl,  txt_bl,  _ = slice_at(bl_a,  bl_sr,  pair["word_a"], pair["text_a"])
            f_tgt, txt_tgt, _ = slice_at(tgt_a, tgt_sr, pair["word_b"], pair["text_b"])
            f_inj, txt_inj, inj_range = slice_at(inj_a, inj_sr, pair["word_b"], pair["text_a"])
            # Also try slicing INJ at word_a position (legacy view)
            f_inj_at_a, _, _ = slice_at(inj_a, inj_sr, pair["word_a"], pair["text_a"])

            row = {
                "pair_idx": pidx,
                "word_a": pair.get("word_a"), "word_b": pair.get("word_b"),
                "fired": fired[0],
                "bl_transcript": txt_bl, "tgt_transcript": txt_tgt, "inj_transcript": txt_inj,
                "source_in_bl":  in_text(pair["word_a"], txt_bl),
                "target_in_tgt": in_text(pair["word_b"], txt_tgt),
                "target_in_inj": in_text(pair["word_b"], txt_inj),  # injection success indicator
                "source_in_inj": in_text(pair["word_a"], txt_inj),  # injection failure indicator
            }
            # SUCCESS metric: INJ at word_b vs TGT at word_b (HIGH = success)
            if f_inj is not None and f_tgt is not None:
                m = acoustic_pair_metrics(f_inj, f_tgt)
                for k, v in m.items():
                    row[f"inj_vs_tgt_{k}"] = round(v, 4) if v is not None else None
                row["aligned_inj_vs_tgt"] = True
            else:
                row["aligned_inj_vs_tgt"] = False
            # FAILURE metric: INJ at word_a vs BL at word_a (HIGH = injection didn't change anything)
            if f_inj_at_a is not None and f_bl is not None:
                m = acoustic_pair_metrics(f_inj_at_a, f_bl)
                for k, v in m.items():
                    row[f"inj_vs_bl_{k}"] = round(v, 4) if v is not None else None
                row["aligned_inj_vs_bl"] = True
            else:
                row["aligned_inj_vs_bl"] = False
            row["aligned"] = row["aligned_inj_vs_tgt"] or row["aligned_inj_vs_bl"]
            rows.append(row); done.add(pidx)
        except Exception as e:
            print(f"  [{i}] err: {str(e)[:200]}", flush=True)
            continue
        if (i + 1) % 5 == 0:
            n_aligned = sum(1 for r in rows if r.get("aligned"))
            print(f"  [{len(rows)}/{args.n}] aligned={n_aligned}", flush=True)
        if (i + 1) % 25 == 0:
            with open(ckpt, "w") as f:
                json.dump({"rows": rows, "skip_pairs": sorted(skip_set)}, f, ensure_ascii=False, indent=1)

    with open(ckpt, "w") as f:
        json.dump({"rows": rows, "skip_pairs": sorted(skip_set)}, f, ensure_ascii=False, indent=1)

    aligned = [r for r in rows if r.get("aligned")]
    def avg(key):
        vs = [r[key] for r in aligned if r.get(key) is not None]
        return [round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)] if vs else [None, None]

    summary = {
        "lang": args.lang, "n_total": len(rows), "n_aligned": len(aligned),
        "n_skip": len(skip_set),
        "target_in_inj_rate": round(sum(1 for r in aligned if r.get("target_in_inj")) / max(1, len(aligned)), 4),
        "source_in_inj_rate": round(sum(1 for r in aligned if r.get("source_in_inj")) / max(1, len(aligned)), 4),
        "source_in_bl_rate":  round(sum(1 for r in aligned if r.get("source_in_bl")) / max(1, len(aligned)), 4),
    }
    for prefix in ("inj_vs_tgt", "inj_vs_bl"):
        for k in ["dtw_mel", "dtw_w2v", "dtw_pps", "tmean_mel", "tmean_w2v", "tmean_pps", "mcd"]:
            summary[f"{prefix}_{k}"] = avg(f"{prefix}_{k}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n[Info] saved {args.output_dir}/results.json", flush=True)
    print(f"[Summary] {json.dumps(summary, indent=2, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
