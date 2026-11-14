#!/usr/bin/env python3
"""C7 (2026-10-02 audit): parameter counts for naacl2027/main.tex tab:models, from the weights (CPU only).

Counts are read from the cached checkpoints (safetensors headers, or torch.load with mmap).
Nothing is downloaded (HF_HUB_OFFLINE=1). Nothing in naacl2027/ or the caches is modified.

Counting rule (one rule for every row):
  * count floating-point tensors that are module parameters;
  * leave out non-parameter buffers (positional-encoding tables `.pe`, BatchNorm running statistics,
    the openai-whisper encoder `positional_embedding` buffer, iSTFT windows, causal-mask buffers, integer tensors);
  * count a tensor stored twice with identical bytes (tied weights) once;
  * for a TTS row, leave out the waveform codec / vocoder ("tts_only"); the full checkpoint is also reported
    ("checkpoint_total") so the author can pick either reading.
  For the HF-supported architectures the result is cross-checked against the model built on the meta device
  (`num_parameters()`), which needs no weights.

Output: eval_results/_c7_param_count/param_count_<timestamp>.json

Usage (CPU, 13 threads):
  taskset -c 0-12 python3 scripts/spanfix/c7_param_count.py
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "13"
PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(PROJECT_ROOT, ".cache", "numba")
sys.modules["pyopenjtalk"] = None  # never used here; block any indirect import

import glob
import hashlib
import json
import math
import pickle
import re
import struct
import time
import types
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
import torch  # noqa: E402

torch.set_num_threads(13)

HUB = os.path.join(PROJECT_ROOT, ".cache", "huggingface", "hub")
OUT_DIR = os.path.join(PROJECT_ROOT, "eval_results", "_c7_param_count")

BUFFER_RE = re.compile(
    r"(\.pe$|(^|\.)inv_freq$|(^|\.)position_ids$|running_mean$|running_var$|num_batches_tracked$"
    r"|(^|\.)masked_bias$|istft\.window$|(^|\.)window$|mel_filters$|(^|\.)attn\.bias$)")


def one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"[fatal] expected exactly one file for {pattern}, got {hits}")
    return hits[0]


def many(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"[fatal] no file for {pattern}")
    return hits


# ------------------------------------------------------------------ readers
class _SafeTensors:
    """Lazy view of one or more safetensors files: name -> (shape, dtype, file)."""

    def __init__(self, files):
        self.files = files
        self.meta = {}
        for f in files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                self.meta[k] = (tuple(v["shape"]), v["dtype"], f)

    def items(self):
        for k, (shape, dtype, _f) in self.meta.items():
            yield k, shape, dtype in ("F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2")

    def digest(self, name):
        from safetensors import safe_open
        shape, dtype, f = self.meta[name]
        with safe_open(f, framework="pt") as fh:
            t = fh.get_tensor(name)
        return hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class _TorchSD:
    def __init__(self, sd):
        self.sd = {k: v for k, v in _flatten(sd)}

    def items(self):
        for k, t in self.sd.items():
            yield k, tuple(t.shape), t.is_floating_point()

    def digest(self, name):
        t = self.sd[name]
        return hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _flatten(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _StubUnpickler(pickle.Unpickler):
    """Unpickle XTTS model.pth without the coqui TTS package (its config objects become stubs)."""

    def find_class(self, mod, name):
        if mod.split(".")[0] in ("TTS", "coqpit", "trainer", "omegaconf"):
            return type(name, (_Stub,), {})
        return super().find_class(mod, name)


_stubpickle = types.ModuleType("stubpickle")
_stubpickle.Unpickler = _StubUnpickler
_stubpickle.load = pickle.load


def tload(path, key=None, mmap=True, stub=False):
    kw = dict(map_location="cpu")
    if stub:
        d = torch.load(path, weights_only=False, pickle_module=_stubpickle, **kw)
    else:
        try:
            d = torch.load(path, mmap=mmap, weights_only=True, **kw)
        except Exception:
            d = torch.load(path, weights_only=True, **kw)
    if key is not None:
        for part in key.split("/"):
            d = d[part]
    return _TorchSD(d)


# ------------------------------------------------------------------ counting
def count(store, exclude_prefixes=(), extra_buffers=(), tie_candidates=None):
    """Return dict with params, excluded buffers, excluded prefixes, tied duplicates."""
    per_top = Counter()
    params = 0
    buffers = []
    excluded = Counter()
    nonfloat = 0
    names_by_shape = defaultdict(list)
    for name, shape, is_float in store.items():
        n = math.prod(shape) if shape else 1
        if not is_float:
            nonfloat += n
            continue
        if BUFFER_RE.search(name) or name in extra_buffers:
            buffers.append((name, n))
            continue
        pref = next((p for p in exclude_prefixes if name.startswith(p)), None)
        if pref:
            excluded[pref] += n
            continue
        params += n
        per_top[name.split(".")[0]] += n
        if tie_candidates and re.search(tie_candidates, name) and n >= 1_000_000:
            names_by_shape[shape].append(name)
    tied = []
    for shape, names in names_by_shape.items():
        if len(names) < 2:
            continue
        seen = {}
        for nm in names:
            dg = store.digest(nm)
            if dg in seen:
                tied.append({"dup": nm, "of": seen[dg], "numel": math.prod(shape)})
            else:
                seen[dg] = nm
    tied_n = sum(t["numel"] for t in tied)
    return {
        "params": params - tied_n,
        "params_before_tie_dedup": params,
        "tied_duplicates": tied,
        "by_top_level_prefix": dict(per_top.most_common()),
        "excluded_prefixes": dict(excluded),
        "excluded_buffers_numel": sum(n for _, n in buffers),
        "excluded_buffers_examples": [b[0] for b in buffers[:6]],
        "nonfloat_numel": nonfloat,
    }


def meta_num_parameters(builder, exclude_module=None):
    try:
        ctor = builder()  # imports and config reading happen outside the meta-device context
        with torch.device("meta"):
            m = ctor()
        total = sum(p.numel() for p in m.parameters())  # parameters() de-duplicates tied ones
        sub = 0
        if exclude_module is not None:
            sub = sum(p.numel() for p in getattr(m, exclude_module).parameters())
        return {"meta_num_parameters": total, "meta_excluded_module": exclude_module,
                "meta_excluded_module_params": sub, "meta_tts_only": total - sub}
    except Exception as e:  # report, do not fail
        return {"meta_error": f"{type(e).__name__}: {str(e)[:200]}"}


def fmt(n):
    return f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.0f}M" if n >= 1e8 else f"{n/1e6:.1f}M"


# ------------------------------------------------------------------ rows of tab:models
def rows():
    S = lambda repo, pat: os.path.join(HUB, f"models--{repo}", "snapshots", "*", pat)  # noqa: E731
    from transformers import AutoConfig

    def hfcfg(repo, pat="config.json"):
        return AutoConfig.from_pretrained(os.path.dirname(one(S(repo, pat))))

    R = []

    files = many(S("Aratako--T5Gemma-TTS-2b-2b", "model-0000*-of-00003.safetensors"))
    R.append(dict(row="T5Gemma-TTS-2b-2b", printed="4.0B", files=files,
                  store=lambda f=files: _SafeTensors(f), codec="NandemoGHS/Anime-XCodec2-44.1kHz-v2 (separate file; XCodec2 row)",
                  tie=r"(embed|predict_layer|audio_embedding|lm_head)"))

    f = one(S("microsoft--speecht5_tts", "model.safetensors"))
    voc = one(S("microsoft--speecht5_hifigan", "pytorch_model.bin"))

    def _st5():
        from transformers import SpeechT5Config, SpeechT5ForTextToSpeech
        cfg = SpeechT5Config.from_pretrained(os.path.dirname(one(S("microsoft--speecht5_tts", "config.json"))))
        return lambda: SpeechT5ForTextToSpeech(cfg)
    R.append(dict(row="SpeechT5 (TTS)", printed="144M", files=[f], store=lambda f=f: _SafeTensors([f]),
                  codec_files=[voc], codec_store=lambda v=voc: tload(v),
                  codec="microsoft/speecht5_hifigan (separate file)", meta=_st5))

    f = one(S("meituan-longcat--LongCat-AudioDiT-1B", "model.safetensors"))
    R.append(dict(row="LongCat-AudioDiT-1B", printed="--", files=[f], store=lambda f=f: _SafeTensors([f]),
                  exclude=("vae.",), codec="waveform VAE inside the same checkpoint (prefix vae.)"))

    f = one(S("Aratako--Irodori-TTS-v4.1-Small", "model.safetensors"))
    dac = one(S("Aratako--Semantic-DACVAE-Japanese-32dim", "weights.pth"))
    R.append(dict(row="Irodori-TTS-v4.1-Small", printed="--", files=[f], store=lambda f=f: _SafeTensors([f]),
                  codec_files=[dac], codec_store=lambda d=dac: tload(d, "state_dict"),
                  codec="Aratako/Semantic-DACVAE-Japanese-32dim (separate file)"))

    f = one(S("telecomadm1145--Kiseki-TTS", "model.safetensors"))
    R.append(dict(row="Kiseki-TTS", printed="0.41B", files=[f], store=lambda f=f: _SafeTensors([f]),
                  codec="not part of this checkpoint"))

    f = one(S("Qwen--Qwen3-TTS-12Hz-1.7B-Base", "model.safetensors"))
    tok = one(S("Qwen--Qwen3-TTS-12Hz-1.7B-Base", "speech_tokenizer/model.safetensors"))
    R.append(dict(row="Qwen3-TTS-12Hz-1.7B", printed="1.7B", files=[f], store=lambda f=f: _SafeTensors([f]),
                  codec_files=[tok], codec_store=lambda t=tok: _SafeTensors([t]),
                  codec="speech_tokenizer/ (same repository, separate file)",
                  tie=r"(embed|head|lm_head)"))

    f = one(S("coqui--XTTS-v2", "model.pth"))
    R.append(dict(row="XTTS-v2", printed="--", files=[f], store=lambda f=f: tload(f, "model", stub=True),
                  exclude=("hifigan_decoder.",), codec="HiFi-GAN decoder inside the same checkpoint (prefix hifigan_decoder.)",
                  tie=r"(embedding|head)"))

    files = many(S("HKUSTAudio--Llasa-3B", "model-0000*-of-00002.safetensors"))

    def _llasa():
        from transformers import AutoModelForCausalLM
        cfg = hfcfg("HKUSTAudio--Llasa-3B")
        return lambda: AutoModelForCausalLM.from_config(cfg)
    R.append(dict(row="Llasa-3B", printed="--", files=files, store=lambda f=files: _SafeTensors(f),
                  codec="HKUSTAudio/xcodec2 (separate repository; XCodec2 row)", meta=_llasa,
                  tie=r"(embed_tokens|lm_head)"))

    files = many(S("sesame--csm-1b", "transformers-0000*-of-00002.safetensors"))

    def _csm():
        from transformers import CsmForConditionalGeneration
        cfg = hfcfg("sesame--csm-1b")
        return lambda: CsmForConditionalGeneration(cfg)
    R.append(dict(row="CSM-1B", printed="--", files=files, store=lambda f=files: _SafeTensors(f),
                  exclude=("codec_model.",), codec="Mimi codec inside the transformers checkpoint (prefix codec_model.)",
                  meta=_csm, meta_exclude="codec_model", tie=r"(embed|head)"))

    f = one(S("suno--bark", "pytorch_model.bin"))

    def _bark():
        from transformers import BarkModel
        cfg = hfcfg("suno--bark")
        return lambda: BarkModel(cfg)
    R.append(dict(row="Bark", printed="--", files=[f], store=lambda f=f: tload(f),
                  exclude=("codec_model.",), codec="EnCodec inside the same checkpoint (prefix codec_model.)",
                  meta=_bark, meta_exclude="codec_model", tie=r"(embed|lm_head|lm_heads)"))

    fa = one(S("NandemoGHS--Anime-XCodec2-44.1kHz-v2", "model.safetensors"))
    R.append(dict(row="XCodec2 [as used by T5Gemma-TTS: NandemoGHS/Anime-XCodec2-44.1kHz-v2]", printed="0.5B",
                  files=[fa], store=lambda f=fa: _SafeTensors([f]), codec="(this row is the codec)"))
    fh = one(S("HKUSTAudio--xcodec2", "model.safetensors"))
    R.append(dict(row="XCodec2 [as used by Llasa-3B: HKUSTAudio/xcodec2]", printed="0.5B",
                  files=[fh], store=lambda f=fh: _SafeTensors([f]), codec="(this row is the codec)"))

    f = one(S("pyp1--VoiceCraft_830M_TTSEnhanced", "model.safetensors"))
    enc = one(S("pyp1--VoiceCraft", "encodec_4cb2048_giga.th"))
    R.append(dict(row="VoiceCraft", printed="830M", files=[f], store=lambda f=f: _SafeTensors([f]),
                  codec_files=[enc], codec_store=lambda e=enc: tload(e, "model", stub=True),
                  codec="pyp1/VoiceCraft encodec_4cb2048_giga.th (separate file)", tie=r"(embedding|predict_layer)"))

    g = os.path.join(PROJECT_ROOT, "third_party", "EdiTTS", "checkpts", "grad-tts-old.pt")
    h = os.path.join(PROJECT_ROOT, "third_party", "EdiTTS", "checkpts", "hifigan.pt")
    R.append(dict(row="EdiTTS", printed="--", files=[g], store=lambda g=g: tload(g),
                  codec_files=[h], codec_store=lambda h=h: tload(h, "generator", mmap=False),
                  codec="HiFi-GAN generator, third_party/EdiTTS/checkpts/hifigan.pt (separate file)"))

    f = one(S("openai--whisper-large-v3", "model.safetensors"))

    def _wl3():
        from transformers import WhisperForConditionalGeneration
        cfg = hfcfg("openai--whisper-large-v3")
        return lambda: WhisperForConditionalGeneration(cfg)
    R.append(dict(row="Whisper large-v3", printed="1.5B", files=[f], store=lambda f=f: _SafeTensors([f]),
                  meta=_wl3, tie=r"(embed_tokens|proj_out)"))

    f = os.path.join(PROJECT_ROOT, ".cache", "whisper", "medium.en.pt")
    R.append(dict(row="Whisper medium.en", printed="--", files=[f],
                  store=lambda f=f: tload(f, "model_state_dict"),
                  extra_buffers=("encoder.positional_embedding",)))

    f = one(S("facebook--wav2vec2-xlsr-53-espeak-cv-ft", "model.safetensors"))

    def _w2v():
        from transformers import Wav2Vec2ForCTC
        cfg = hfcfg("facebook--wav2vec2-xlsr-53-espeak-cv-ft")
        return lambda: Wav2Vec2ForCTC(cfg)
    R.append(dict(row="wav2vec2-xlsr-53-espeak-cv-ft", printed="0.3B", files=[f],
                  store=lambda f=f: _SafeTensors([f]), meta=_w2v))

    f = one(S("walston--whisper-pinyin", "checkpoint-epoch=0009.ckpt"))
    R.append(dict(row="whisper-pinyin", printed="--", files=[f], store=lambda f=f: tload(f, "state_dict"),
                  extra_buffers=("model.encoder.positional_embedding",)))

    f = os.path.join(PROJECT_ROOT, ".cache", "torch", "hub", "checkpoints", "utmos22_strong_step7459_v1.pt")
    R.append(dict(row="UTMOS22", printed="35M", files=[f], store=lambda f=f: tload(f)))
    return R


def main():
    t0 = time.time()
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "script": "scripts/spanfix/c7_param_count.py",
           "rule": __doc__.split("Counting rule (one rule for every row):")[1].split("Output:")[0].strip(),
           "rows": []}
    for spec in rows():
        st = spec["store"]()
        c = count(st, spec.get("exclude", ()), spec.get("extra_buffers", ()), spec.get("tie"))
        rec = {"row": spec["row"], "printed_in_main_tex": spec["printed"],
               "files": [os.path.relpath(os.path.realpath(p), PROJECT_ROOT) if os.path.realpath(p).startswith(PROJECT_ROOT)
                         else os.path.realpath(p) for p in spec["files"]],
               "file_links": [os.path.relpath(p, PROJECT_ROOT) for p in spec["files"]],
               "codec_or_vocoder": spec.get("codec", ""), **c}
        rec["tts_only"] = c["params"]
        rec["checkpoint_total"] = c["params"] + sum(c["excluded_prefixes"].values())
        if "codec_store" in spec:
            cc = count(spec["codec_store"]())
            rec["separate_codec_params"] = cc["params"]
            rec["separate_codec_files"] = [os.path.relpath(p, PROJECT_ROOT) for p in spec["codec_files"]]
        if "meta" in spec:
            rec.update(meta_num_parameters(spec["meta"], spec.get("meta_exclude")))
        rec["tts_only_fmt"] = fmt(rec["tts_only"])
        rec["checkpoint_total_fmt"] = fmt(rec["checkpoint_total"])
        out["rows"].append(rec)
        meta = rec.get("meta_tts_only", rec.get("meta_error", ""))
        print(f"{spec['row'][:60]:60s} printed={spec['printed']:>6s}  tts_only={rec['tts_only']:>14,d} ({rec['tts_only_fmt']})"
              f"  ckpt_total={rec['checkpoint_total']:>14,d}  tied={len(c['tied_duplicates'])}"
              f"  buffers={c['excluded_buffers_numel']:,}  meta={meta}", flush=True)
        del st
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"param_count_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(path):
        raise SystemExit(f"[fatal] {path} exists; not overwriting")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[written] {os.path.relpath(path, PROJECT_ROOT)}  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
