#!/usr/bin/env python3
"""Run the Table 3 T5Gemma cross-culture generation script on another pair set.

The three generation scripts

    scripts/projection/eval_{en,ja,zh}_crossculture_t5gemma_fixed.py

are used **unmodified** (not a byte changed).  Each of them reads its pairs with
one statement inside main():

    from {lang}_cross_culture_names import build_300_pairs

This wrapper imports that module first and replaces its ``build_300_pairs``
attribute with the one from ``--pairs_module`` (e.g. ``en_nontemplate_names``),
so the unmodified statement picks up the new pairs.  Everything else (seed 42,
sampling, no prompt audio, span search, splice, Whisper, wav names
s{pair_idx:04d}_{bl,tgt,inj}.wav) is the original code path.

Without ``--pairs_module`` (or with the default module name) nothing is
patched: the wrapper then runs the original script exactly as before, only
with the CPU thread cap below (and, under HF_HUB_OFFLINE=1, the tokenizer
path shim described next).

HF cache / offline (2026-09-27): the queue runs this wrapper with
  HF_HOME=/mnt/.cache/huggingface HF_HUB_CACHE=/mnt/.cache/huggingface/hub
  HF_HUB_OFFLINE=1 TORCH_HOME=/mnt/.cache/torch XDG_CACHE_HOME=/mnt/.cache
(slurm_run.sh itself passes HF_HOME=/tmp/.cache/huggingface, which is empty
and read-only; the JA / ZH generation scripts do not override it).  With
transformers 4.57.3, ``AutoTokenizer.from_pretrained(<repo id>)`` fails under
HF_HUB_OFFLINE=1 for tokenizers with more than 100k entries (T5Gemma,
google/t5gemma-2b-2b-ul2): ``_patch_mistral_regex`` calls the Hub API
(model_info).  So, **only when HF_HUB_OFFLINE is set**, the wrapper wraps
``transformers.AutoTokenizer.from_pretrained``: a repo id that is present in
the local HF cache is replaced by its local snapshot directory
(huggingface_hub.snapshot_download(local_files_only=True)), which is what
scripts/projection/eval_cjk_n300.py does explicitly.  The files are the same
snapshot, and for a non-Mistral tokenizer ``_patch_mistral_regex`` returns
the tokenizer unchanged on both paths (online: no "mistralai" tag; local:
config model_type t5gemma), so the tokenizer is identical.  Every
replacement is printed and stored in the provenance file.  Model weights
(``from_pretrained`` of models / pipelines) are untouched; they load from
the cache under HF_HUB_OFFLINE without a Hub call.

Pair-set guard (2026-09-27): if ``<output_dir>/pairs_module_provenance.json``
already exists (a resumed run) and records a different pair module or pair
digest, the wrapper stops, so a checkpoint made with other sentences is never
continued with new ones.

CPU threads: OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS are set to
``--threads`` (default 13) **before torch is imported**, and
``torch.set_num_threads(13)`` is called (slurm_run.sh cannot pass env vars).

Before generation the wrapper checks, from the generation script's source,
that the patched statement exists exactly once and executes the script's own
two pair-loading statements (extracted with ast) to confirm they return the
requested pairs.  ``--dry_run`` stops after these checks (CPU only; torch is
not imported).

Usage (GPU; normally submitted by scripts/per_fix/run_nontemplate3.sh):
  SLURM_TIME=06:00:00 slurm_run.sh bash -lc \\
    "env HF_HOME=/mnt/.cache/huggingface HF_HUB_CACHE=/mnt/.cache/huggingface/hub \\
         HF_HUB_OFFLINE=1 TORCH_HOME=/mnt/.cache/torch XDG_CACHE_HOME=/mnt/.cache \\
     python3 -u /mnt/scripts/nontemplate/gen_t5gemma_pairs_module.py \\
      --lang en --pairs_module en_nontemplate_names -- \\
      --n 300 --target_valid 300 \\
      --output_dir /mnt/eval_results/_nontemplate/meta_t5gemma_en \\
      --save_wav_dir /mnt/eval_results/_nontemplate/t5gemma_en"
CPU check:
  taskset -c 0-7 python3 scripts/nontemplate/gen_t5gemma_pairs_module.py \\
      --lang en --pairs_module en_nontemplate_names --dry_run -- --n 300
"""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIELDS = {"pair_idx", "text_a", "text_b", "word_a", "word_b"}


def gen_script(lang):
    return os.path.join(PROJECT_ROOT, "scripts", "projection",
                        "eval_%s_crossculture_t5gemma_fixed.py" % lang)


def default_module(lang):
    return "%s_cross_culture_names" % lang


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def pairs_digest(pairs):
    blob = json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def check_pairs(pairs, name, n_expected=300):
    """Light structural check (the full check is validate_nontemplate_pairs.py)."""
    if len(pairs) != n_expected:
        raise SystemExit("[pairs] %s: %d pairs (expected %d)" % (name, len(pairs), n_expected))
    for i, p in enumerate(pairs):
        if set(p) != FIELDS:
            raise SystemExit("[pairs] %s[%d]: fields %s" % (name, i, sorted(p)))
        if p["pair_idx"] != i:
            raise SystemExit("[pairs] %s[%d]: pair_idx=%s" % (name, i, p["pair_idx"]))
        if p["text_a"].count(p["word_a"]) != 1 or p["text_b"].count(p["word_b"]) != 1:
            raise SystemExit("[pairs] %s[%d]: entity not exactly once" % (name, i))


def pair_loading_statements(src_path, default_mod):
    """Return the generation script's own pair-loading statements.

    Finds, inside main(), `from <default_mod> import build_300_pairs` and the
    following `pairs = build_300_pairs(...)` assignment, and returns their
    exact source text.  Fails if the import is not present exactly once.
    """
    src = open(src_path, encoding="utf-8").read()
    tree = ast.parse(src)
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    if len(mains) != 1:
        raise SystemExit("[check] main() not found in %s" % src_path)
    imp, asg = [], []
    for node in ast.walk(mains[0]):
        if isinstance(node, ast.ImportFrom) and node.module == default_mod \
                and [a.name for a in node.names] == ["build_300_pairs"]:
            imp.append(node)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "pairs" \
                and isinstance(node.value, ast.Subscript) \
                and isinstance(node.value.value, ast.Call) \
                and getattr(node.value.value.func, "id", None) == "build_300_pairs":
            asg.append(node)
    # any other reference to a cross-culture module would escape the patch
    others = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
              and "cross_culture_names" in (getattr(n, "module", None) or
                                            ",".join(a.name for a in n.names))]
    if len(imp) != 1 or len(asg) != 1 or len(others) != 1:
        raise SystemExit("[check] %s: expected exactly one `from %s import build_300_pairs` "
                         "and one `pairs = build_300_pairs(...)[...]` (got %d / %d / %d)"
                         % (src_path, default_mod, len(imp), len(asg), len(others)))
    return ast.get_source_segment(src, imp[0]), ast.get_source_segment(src, asg[0])


def install_pairs_module(lang, pairs_module):
    """Patch <lang>_cross_culture_names.build_300_pairs; return info dict."""
    if DATA_DIR not in sys.path:
        sys.path.insert(0, DATA_DIR)
    dmod_name = default_module(lang)
    dmod = importlib.import_module(dmod_name)
    info = {"default_module": dmod_name,
            "default_module_md5": md5(dmod.__file__)}
    if not pairs_module or pairs_module == dmod_name:
        info.update({"pairs_module": dmod_name, "patched": False,
                     "pairs_module_md5": md5(dmod.__file__)})
        return info, dmod.build_300_pairs
    if not pairs_module.startswith(lang + "_"):
        raise SystemExit("[pairs] --pairs_module %s does not match --lang %s" % (pairs_module, lang))
    pmod = importlib.import_module(pairs_module)
    dmod.build_300_pairs = pmod.build_300_pairs
    info.update({"pairs_module": pairs_module, "patched": True,
                 "pairs_module_md5": md5(pmod.__file__)})
    return info, pmod.build_300_pairs


# ---- HF offline helpers (2026-09-27) ---------------------------------------
# Hub repos read by the generation path (eval_{lang}_crossculture_t5gemma_fixed.py
# and the modules it imports).  The two marked "cfg:" are read from the
# T5Gemma-TTS config.json, exactly as the script does.
GEN_REPOS = ["Aratako/T5Gemma-TTS-2b-2b", "cfg:text_tokenizer_name",
             "cfg:xcodec2_model_name", "openai/whisper-large-v3",
             "facebook/wav2vec2-xlsr-53-espeak-cv-ft"]


def hf_offline():
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")


def looks_like_repo_id(name):
    return (isinstance(name, str) and name.count("/") == 1
            and not name.startswith((".", "/", "~")) and not os.path.exists(name))


def local_snapshot(repo_id, revision=None, cache_dir=None):
    """Local snapshot dir of a cached Hub repo; raises if it is not cached."""
    from huggingface_hub import snapshot_download
    kw = {"repo_id": repo_id, "local_files_only": True}
    if revision:
        kw["revision"] = revision
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return snapshot_download(**kw)


def offline_preflight():
    """Resolve every repo of GEN_REPOS in the local cache (no network, no torch)."""
    out, cfg = {}, {}
    for r in GEN_REPOS:
        if r.startswith("cfg:"):
            key = r[4:]
            r = cfg.get(key) or (cfg.get("t5gemma_model_name") if key == "text_tokenizer_name" else None)
            if not r:
                out["cfg:" + key] = {"error": "not in T5Gemma-TTS config"}
                continue
        try:
            snap = local_snapshot(r)
            files = sorted(os.listdir(snap))
            out[r] = {"snapshot": snap,
                      "has_config": "config.json" in files,
                      "has_weights": any(f.endswith((".safetensors", ".bin", ".pt", ".ckpt"))
                                         for f in files)}
            if r == GEN_REPOS[0]:
                cfg = json.load(open(os.path.join(snap, "config.json")))
        except Exception as e:                      # noqa: BLE001
            out[r] = {"error": repr(e)[:300]}
    ok = all("error" not in v and v.get("has_config") for v in out.values())
    return ok, out


def install_offline_tokenizer_shim(log):
    """HF_HUB_OFFLINE only: AutoTokenizer.from_pretrained(<repo id>) -> local snapshot.

    Returns the original bound method (so tests can restore it).
    """
    from transformers import AutoTokenizer
    orig = AutoTokenizer.from_pretrained

    def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        name = pretrained_model_name_or_path
        if looks_like_repo_id(name):
            try:
                snap = local_snapshot(name, kwargs.get("revision"), kwargs.get("cache_dir"))
            except Exception as e:                  # noqa: BLE001 (keep original behaviour)
                log.append({"repo_id": name, "snapshot": None, "error": repr(e)[:200]})
                print("[offline] AutoTokenizer %s: not in the local cache (%s); unchanged"
                      % (name, type(e).__name__), flush=True)
            else:
                log.append({"repo_id": name, "snapshot": snap})
                print("[offline] AutoTokenizer %s -> %s" % (name, snap), flush=True)
                name = snap
        return orig(name, *args, **kwargs)

    AutoTokenizer.from_pretrained = staticmethod(from_pretrained)
    return orig


def guard_previous_provenance(path, info, allow_change):
    """Refuse to resume an output dir made with another pair set; return history."""
    if not os.path.exists(path):
        return []
    try:
        prev = json.load(open(path))
    except Exception as e:                          # noqa: BLE001
        raise SystemExit("[pairs] cannot read %s (%s); refusing to resume" % (path, e))
    same = (prev.get("pairs_module") == info["pairs_module"]
            and prev.get("pairs_sha1") == info["pairs_sha1"])
    if not same and not allow_change:
        raise SystemExit(
            "[pairs] %s was written for pairs_module=%s sha1=%s, but this run has %s sha1=%s. "
            "Refusing to continue that checkpoint with other sentences "
            "(use a new --output_dir / --save_wav_dir)." % (
                path, prev.get("pairs_module"), prev.get("pairs_sha1"),
                info["pairs_module"], info["pairs_sha1"]))
    hist = list(prev.get("previous_starts", []))
    hist.append({k: prev.get(k) for k in ("started", "pairs_module", "pairs_module_md5",
                                          "pairs_sha1", "finished")})
    return hist


def main():
    ap = argparse.ArgumentParser(
        description="Wrapper; arguments after `--` go to the generation script.")
    ap.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    ap.add_argument("--pairs_module", default=None,
                    help="module in data/ exposing build_300_pairs(); "
                         "default = <lang>_cross_culture_names (no patch)")
    ap.add_argument("--threads", type=int, default=13)
    ap.add_argument("--dry_run", action="store_true",
                    help="CPU only: check the patch, do not import torch or generate")
    ap.add_argument("--allow_pairs_change", action="store_true",
                    help="resume an output dir although its provenance records another pair set")
    argv = sys.argv[1:]
    if "--" in argv:
        k = argv.index("--")
        own, rest = argv[:k], argv[k + 1:]
    else:
        own, rest = argv, []
    args = ap.parse_args(own)

    # ---- thread cap: before torch is imported anywhere ----------------------
    t = str(args.threads)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = t
    if "torch" in sys.modules:
        raise SystemExit("[threads] torch was imported before the thread cap")

    script = gen_script(args.lang)
    info, expected_fn = install_pairs_module(args.lang, args.pairs_module)
    expected = expected_fn()
    check_pairs(expected, info["pairs_module"])

    # ---- execute the generation script's own pair-loading statements --------
    imp_src, asg_src = pair_loading_statements(script, info["default_module"])
    gp = argparse.ArgumentParser(add_help=False)
    gp.add_argument("--n", type=int, default=300)       # same default as the script
    gp.add_argument("--output_dir", default=None)
    gargs, _ = gp.parse_known_args(rest)
    ns = {"args": SimpleNamespace(n=gargs.n)}
    exec(imp_src + "\n" + asg_src, ns)
    got = ns["pairs"]
    if got != expected[: gargs.n + 50]:
        raise SystemExit("[check] the script's statements did not return %s" % info["pairs_module"])
    info.update({
        "generation_script": script, "generation_script_md5": md5(script),
        "statements": [imp_src, asg_src],
        "n_pairs_seen_by_script": len(got), "pairs_sha1": pairs_digest(got),
        "threads": args.threads, "script_args": rest,
    })
    print("[pairs] %s -> %d pairs (patched=%s) sha1=%s" % (
        info["pairs_module"], len(got), info["patched"], info["pairs_sha1"]), flush=True)
    for p in got[:3]:
        print("   ", p["pair_idx"], p["text_a"], "|", p["text_b"], flush=True)
    print("[script] %s md5=%s" % (script, info["generation_script_md5"]), flush=True)

    # ---- HF cache / offline (2026-09-27) --------------------------------------
    info["hf_env"] = {k: os.environ.get(k) for k in (
        "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TORCH_HOME", "XDG_CACHE_HOME")}
    info["hf_offline"] = hf_offline()
    if info["hf_offline"]:
        ok, pre = offline_preflight()
        info["offline_preflight"] = pre
        for r, v in pre.items():
            print("[offline] %-45s %s" % (r, v.get("snapshot") or v.get("error")), flush=True)
        if not ok:
            raise SystemExit("[offline] HF_HUB_OFFLINE=1 but a repo of the generation path "
                             "is not in the local cache (HF_HUB_CACHE=%s)"
                             % os.environ.get("HF_HUB_CACHE"))

    if args.dry_run:
        print("[dry_run] OK (no generation)")
        print(json.dumps({k: v for k, v in info.items() if k != "statements"},
                         ensure_ascii=False))
        return

    import torch
    torch.set_num_threads(args.threads)
    print("[threads] torch.get_num_threads()=%d  OMP_NUM_THREADS=%s"
          % (torch.get_num_threads(), os.environ["OMP_NUM_THREADS"]), flush=True)

    shim_log = []
    if info["hf_offline"]:
        install_offline_tokenizer_shim(shim_log)
        info["offline_tokenizer_shim"] = shim_log

    prov_path = None
    if gargs.output_dir:
        out = gargs.output_dir if os.path.isabs(gargs.output_dir) \
            else os.path.join(PROJECT_ROOT, gargs.output_dir)
        os.makedirs(out, exist_ok=True)
        prov_path = os.path.join(out, "pairs_module_provenance.json")
        info["previous_starts"] = guard_previous_provenance(
            prov_path, info, args.allow_pairs_change)
        info["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(prov_path, "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)

    spec = importlib.util.spec_from_file_location(
        "t5gemma_cc_fixed_%s" % args.lang, script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = [script] + rest
    print("[run] %s %s" % (script, " ".join(rest)), flush=True)
    mod.main()

    if prov_path:
        info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(prov_path, "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
