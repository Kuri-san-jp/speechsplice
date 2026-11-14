#!/usr/bin/env python3
"""付録 C（tab:ja_crossattn_phoneme）の日本語の行を、付録 K の日本語 300 対と直したスパン関数で作り直す
GPU の入り口（2026-10-02, 著者の決定 (1)）.

本番のスクリプト scripts/projection/eval_crossattn_entity_per.py を **改変せずに** main() を呼ぶ（md5 を固定）。
引数は元の実行と同じ `--lang ja --n 300` に `--output_dir`（新しい置き場）だけを足す。元と違うのは 2 つだけ:
  * 対: main() の JA の分岐が開く相対パス eval_results/crossattn_mask_n300_filtered_v2/results.json を、本番の
    module の open だけで、付録 K の 300 対を同じ形にしたファイル（appc_ja_span_check.py が作る
    eval_results/_spanfix_appC_ja/pairs/appk_ja_300_v2format.json）に読み替える。他の open はそのまま通す。
    読み替えは 1 回だけで、"[Info] Loaded 300 ja pairs" の時点で確かめる（違えば対を作る前に止まる）。
  * スパン: main() が import する encoder_injection_t5gemma.find_entity_token_range が 7/11 に直した版。

このラッパーがすること（本番の処理には手を入れない。出力を写し取るだけ）は英語の版（appc_en_gen.py）と同じ:
  * 1 回の起動で 1 つの batch（seed 42 は 50 対ずつの part00..part05）。対象以外は placeholder で「済み」にして
    本番の resume で対象の対だけを作る。時間切れでも同じコマンドで続きから（25 対ごとの checkpoint）。
  * モジュール変数 SEED を --seed に置き換える（対ごと、5 条件すべて同じ seed）。
  * generate_with_modified_encoder・find_entity_token_range・estimate_duration・WhisperAligner.transcribe_words・
    find_entity_chunks・slice_entity_audio を、呼び出しをそのまま通して記録だけを取る包みに替える。5 本の wav を
    float32 で保存し、使われたスパンと遮断した列、記憶（エンコーダ出力）の形と値の一致、長さ推定を記録する。
  * 無音は scripts/regen2/silence_criterion.py で保存した wav から判定する（使えるかの判定は appc_ja_scan.py）。
日本語で英語と違う点:
  * スパンは vendor の正規化（normalize_text_with_lang(text, "ja")）の後の文に対して呼ばれるので、正規化後の文で
    対を特定する（付録 K の 300 対は正規化で変わらないことを span_check で確かめ済み）。
  * a と b の範囲の長さが違う対は継ぎ合わせ（本番 run_condition）。INJ_mask で遮断する列は [a_s, a_s + (b_e - b_s))。
    記憶の確かめは、長さ（L - la + lb）、実体語の前後が BL と同じ、BL_mask = BL、INJ_mask = INJ。
  * pyopenjtalk: T5Gemma-TTS 同梱の長さ推定器（vendor duration_estimator._phoneme_count_ja）の中でだけ使う
    （5/11 の実行も同じ経路。ログの JPCommonLabel の警告）。GPU の本番では import の関所を置き、vendor の
    duration_estimator.py（と pyopenjtalk 自身の中）以外からの import を塞ぐ。読み込まれた後は
    duration_estimator.pyopenjtalk を g2p だけを通す包みに替え、呼び出し元が _phoneme_count_ja であることを
    毎回確かめて数える。長さ推定器が pyopenjtalk なし（len(text)）になっていれば、モデルを読む前に止まる。
    ホスト・--dry_run・--selftest・import されたときは sys.modules["pyopenjtalk"] = None で塞ぐ。

seed の決まり（著者の決定）: seed 42 で 300 対（--part 0..5）。seed 42 で使えない対（unaligned / silent /
generation_failed）だけ seed 43、さらに使えなければ seed 44。appc_ja_scan.py の状態ファイル（--status_file）から
対象を取る。環境の問題（run_error / alignment_exception / capture_error / no_row_unexplained）は同じ seed で
--retry N --pairs ...。同じ種類が 2 回、または retry 2 回でも続く対は scan が止める。選ぶ基準は有効性だけ。

環境変数（コマンドに env を付けない。このスクリプトが先頭で、numpy・torch・HF を読む前に自分で入れる）:
  スレッド上限 13（OMP/MKL/OPENBLAS/NUMEXPR/VECLIB、torch.set_num_threads(13)）、HF_HOME=<根>/.cache/huggingface,
  HF_HUB_CACHE, HF_HUB_OFFLINE=1, TORCH_HOME, XDG_CACHE_HOME, NUMBA_CACHE_DIR=<根>/.cache/numba（<根> はコンテナ
  の中なら /mnt）。ホストでは CUDA_VISIBLE_DEVICES=''。

GPU（決まった形。1 ジョブ = 1 batch = GPU 1 枚。投入の前に squeue で利用者全体で 2 枚以内を確かめる）:
  SLURM_TIME=02:00:00 slurm_run.sh python3 -u /mnt/scripts/spanfix/gpu_guard.py \
      --min_free_gb 30 -- /mnt/scripts/spanfix/appc_ja_gen.py --seed 42 --part 0
  （--part 1..5 は part00 の後の確かめ（appc_ja_scan.py）が済んでから。seed 43/44 とやり直しは scan が示す形で）
CPU（ホストで直接。モデルは読まない）:
  taskset -c 0-12 python3 scripts/spanfix/appc_ja_gen.py --seed 42 --part 0 --dry_run
  taskset -c 0-12 python3 scripts/spanfix/appc_ja_gen.py --selftest
"""
import builtins
import os
import sys

THREADS = 13
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TORCH_HOME", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR")
ENV_AT_LAUNCH = {k: os.environ.get(k) for k in THREAD_VARS + CACHE_VARS + ("CUDA_VISIBLE_DEVICES",
                                                                           "NVIDIA_VISIBLE_DEVICES")}
for _k in THREAD_VARS:
    os.environ[_k] = str(THREADS)
IN_CONTAINER = os.path.isdir("/mnt/scripts")
_R = "/mnt" if IN_CONTAINER else os.environ.get("REPO_ROOT", "/mnt")
CACHE_ENV = {"HF_HOME": _R + "/.cache/huggingface", "HF_HUB_CACHE": _R + "/.cache/huggingface/hub",
             "HF_HUB_OFFLINE": "1", "TORCH_HOME": _R + "/.cache/torch", "XDG_CACHE_HOME": _R + "/.cache",
             "NUMBA_CACHE_DIR": _R + "/.cache/numba"}
for _k, _v in CACHE_ENV.items():
    os.environ[_k] = _v  # setdefault ではなく上書き（slurm_run.sh は HF_HOME=/tmp/.cache/huggingface を渡す）
if not IN_CONTAINER:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ホストでは GPU に触れない
sys.dont_write_bytecode = True  # 既存の scripts/ 以下に .pyc を新しく作らない

# ---------------------------------------------------------------- pyopenjtalk の関所（numpy・torch より前）
VENDOR_DE_REAL = os.path.realpath("$T5GEMMA_ROOT/duration_estimator.py")
GPU_MODE = (IN_CONTAINER and __name__ == "__main__"
            and not any(a in sys.argv for a in ("--dry_run", "--selftest")))
PYOJT = {"mode": None, "allowed_imports": [], "blocked_imports": [], "g2p_calls": 0, "bad_callers": []}
_ORIG_IMPORT = builtins.__import__


def _abs_import_name(name, globals_, level):
    if not level:
        return name
    pkg = (globals_ or {}).get("__package__") or ""
    parts = pkg.split(".") if pkg else []
    if level > 1:
        parts = parts[:max(0, len(parts) - (level - 1))]
    base = ".".join(parts)
    return f"{base}.{name}" if name and base else (name or base)


def pyojt_import_guard(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002 (builtins の形)
    absname = _abs_import_name(name, globals, level)
    if absname == "pyopenjtalk" or absname.startswith("pyopenjtalk."):
        g = globals or {}
        f = g.get("__file__")
        mod = g.get("__name__", "") or ""
        allowed = bool((f and os.path.realpath(f) == VENDOR_DE_REAL) or mod == "pyopenjtalk"
                       or mod.startswith("pyopenjtalk."))
        rec = {"name": absname, "importer": mod, "file": f}
        if not allowed:
            PYOJT["blocked_imports"].append(rec)
            raise ImportError(f"pyopenjtalk は T5Gemma-TTS 同梱の長さ推定器の中だけで使う（{mod} からの import を塞いだ）")
        if len(PYOJT["allowed_imports"]) < 20:
            PYOJT["allowed_imports"].append(rec)
    return _ORIG_IMPORT(name, globals, locals, fromlist, level)


class G2PProxy:
    """vendor duration_estimator.pyopenjtalk の代わり。g2p だけを、_phoneme_count_ja から呼ばれたときだけ通す."""

    def __init__(self, real, allowed_code):
        self._real = real
        self._allowed = allowed_code

    def g2p(self, *a, **k):
        caller = sys._getframe(1).f_code
        if caller is not self._allowed:
            PYOJT["bad_callers"].append(f"{caller.co_filename}:{caller.co_name}")
            raise RuntimeError("pyopenjtalk.g2p は vendor の _phoneme_count_ja からだけ呼べる")
        PYOJT["g2p_calls"] += 1
        return self._real.g2p(*a, **k)

    def __getattr__(self, name):
        raise AttributeError(f"pyopenjtalk.{name} は使わない（g2p だけを通す）")


if GPU_MODE:
    builtins.__import__ = pyojt_import_guard
    PYOJT["mode"] = "gpu: import guard (vendor duration_estimator only)"
else:
    sys.modules["pyopenjtalk"] = None
    PYOJT["mode"] = "blocked (host / dry_run / selftest / imported)"

import argparse  # noqa: E402
import ast  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "regen2"))  # silence_criterion
import appc_ja_common as C  # noqa: E402
import silence_criterion as SC  # noqa: E402

ROOT = C.PROJECT_ROOT
assert _R == ROOT, (_R, ROOT)
assert VENDOR_DE_REAL == os.path.realpath(C.VENDOR_DE_FILE)
OWN_REL = "scripts/spanfix/appc_ja_gen.py"
OWN_CODE = (OWN_REL, "scripts/spanfix/appc_ja_common.py")
PAIR_ERR_RE = re.compile(r"^\s*\[(\d+)\] err: ")
WHISPER_ERR_RE = re.compile(r"^\s*\[Whisper\] transcribe failed: ")
LOADED_RE = re.compile(r"^\[Info\] Loaded (\d+) (\w+) pairs")
FA_WHISPER_FAIL_PRINT = 'print(f"  [Whisper] transcribe failed: {str(e)[:160]}")'


# ---------------------------------------------------------------- 小道具
def as_np(x, floating=False):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach()
        if floating:
            x = x.float()
        return x.cpu().numpy()
    return np.asarray(x, dtype=np.float32) if floating else np.asarray(x)


def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 事前の確かめ（CPU）
def static_checks():
    """本番スクリプトに、差し替えに使う箇所があるかを ast で確かめる（import しない）."""
    src = open(C.P(C.EVAL_REL), encoding="utf-8").read()
    tree = ast.parse(src)
    seed_assign = any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SEED" for t in n.targets)
                      for n in tree.body)
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    lazy = [n for n in ast.walk(main) if isinstance(n, ast.ImportFrom) and n.module == "encoder_injection_t5gemma"]
    lazy_names = sorted(a.name for n in lazy for a in n.names)
    top_inj = any(isinstance(n, (ast.ImportFrom, ast.Import)) and (
        getattr(n, "module", None) == "encoder_injection_t5gemma"
        or any(a.name == "encoder_injection_t5gemma" for a in getattr(n, "names", [])))
        for n in tree.body)
    fa_top = any(isinstance(n, ast.ImportFrom) and n.module == "forced_alignment_utils"
                 and sorted(a.name for a in n.names) == ["WhisperAligner", "find_entity_chunks", "slice_entity_audio"]
                 for n in tree.body)

    def shadowed(name):
        return any(isinstance(n, (ast.FunctionDef, ast.Assign, ast.Import, ast.ImportFrom)) and (
            getattr(n, "name", None) == name
            or any(getattr(t, "id", None) == name for t in getattr(n, "targets", []))
            or any((a.asname or a.name) == name for a in getattr(n, "names", [])))
            for n in tree.body)
    opens = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "open"]
    const_opens = [n.args[0].value for n in opens if n.args and isinstance(n.args[0], ast.Constant)]
    _, ja_src = C.production_ja_branch_source(ROOT)
    res = {
        "eval_md5": C.md5_file(C.P(C.EVAL_REL)),
        "module_SEED_assign": seed_assign,
        "main_lazy_imports": lazy_names,
        "no_top_level_import_of_injection": not top_inj,
        "fa_imported_at_top": fa_top,
        "module_print_not_shadowed": not shadowed("print"),
        "module_open_not_shadowed": not shadowed("open"),
        "json_imported_at_top": "import argparse, json, os, sys, time, random" in src,
        "n_open_calls_in_main": len(opens),
        "constant_open_paths_in_main": const_opens,
        "ja_branch_source": ja_src,
        "ja_branch_expected": all(s in ja_src for s in (
            'prev = json.load(open("eval_results/crossattn_mask_n300_filtered_v2/results.json"))["results"]',
            'for r in prev[: args.n]:', 'text_a = r["text"]; kanji = r["kanji"]; kata = r["kata"]',
            '"text_a": text_a, "text_b": text_a.replace(kanji, kata),',
            '"word_a": kanji, "word_b": kata, "pair_idx": r["idx"],')),
        "resume_skips_done": "if pair_idx in done: continue" in src,
        "pair_err_print": 'print(f"  [{i}] err: {str(e)[:200]}"); continue' in src,
        "loaded_print": 'print(f"[Info] Loaded {len(pairs)} {args.lang} pairs")' in src,
        "transcribe_call": "aligner.transcribe_words(audio, sr_in, language=lang)" in src,
        "find_call": "find_entity_chunks(chunks, entity, lang, input_text=input_text)" in src,
        "slice_call": "slice_entity_audio(audio_np, aligner.target_sr, chunks, s_idx, e_idx)" in src,
        "span_calls": src.count("find_entity_token_range(t5_tok, "),
        "splice_branch": "modified = torch.cat([prefix, ref_ent, suffix], dim=0).unsqueeze(0)" in src
                         and "entity_end = s_s + (r_e - r_s)" in src,
        "ckpt_every_25": "if (i + 1) % 25 == 0:" in src,
    }
    fa_src = open(C.P(C.FA_REL), encoding="utf-8").read()
    fa_print_lines = [ln.strip() for ln in fa_src.splitlines() if "print(" in ln]
    res["fa_only_print_is_whisper_fail"] = fa_print_lines == [FA_WHISPER_FAIL_PRINT]
    inj_src = open(C.P(C.INJ_REL), encoding="utf-8").read()
    gen_tree = ast.parse(inj_src)
    gen = [n for n in gen_tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_with_modified_encoder"]
    res["generate_params"] = [a.arg for a in gen[0].args.args] if gen else []
    res["inj_imports_estimate_duration"] = "from duration_estimator import estimate_duration" in inj_src
    res["inj_generate_calls_estimate_duration"] = (
        "target_duration = estimate_duration(target_text=target_text_norm, target_lang=lang_code)" in inj_src)
    de_src = open(C.VENDOR_DE_FILE, encoding="utf-8").read()
    res["vendor_de_pyopenjtalk_lines"] = [ln.strip() for ln in de_src.splitlines() if "pyopenjtalk" in ln]
    want_lazy = ["find_entity_token_range", "generate_with_modified_encoder", "get_encoder_output", "seed_everything"]
    ok = (seed_assign and lazy_names == want_lazy and res["no_top_level_import_of_injection"] and fa_top
          and res["module_print_not_shadowed"] and res["module_open_not_shadowed"] and res["json_imported_at_top"]
          and const_opens == [C.OLD_PAIR_SRC_REL] and res["ja_branch_expected"]
          and res["resume_skips_done"] and res["pair_err_print"] and res["loaded_print"]
          and res["transcribe_call"] and res["find_call"] and res["slice_call"] and res["span_calls"] == 3
          and res["splice_branch"] and res["fa_only_print_is_whisper_fail"]
          and res["inj_imports_estimate_duration"] and res["inj_generate_calls_estimate_duration"]
          and res["vendor_de_pyopenjtalk_lines"] == ["import pyopenjtalk", "pyopenjtalk = None",
                                                     "if pyopenjtalk is None:", "ph = pyopenjtalk.g2p(text)"]
          and res["generate_params"] == ["model", "text_tokenizer", "audio_tokenizer", "modified_memory",
                                         "encoder_attention_mask", "encoder_position_ids", "target_text",
                                         "device", "codec_audio_sr", "codec_sr"])
    res["ok"] = bool(ok)
    if not ok:
        raise SystemExit(f"[appc_ja] 本番スクリプトの形が想定と違う: {json.dumps(res, ensure_ascii=False)}")
    return res


def latest_span_check(pairs):
    d = C.P(C.SPAN_CHECK_DIR_REL)
    files = sorted(f for f in os.listdir(d) if f.startswith("span_check_")) if os.path.isdir(d) else []
    if not files:
        raise SystemExit("[appc_ja] CPU のスパンの確かめ（appc_ja_span_check.py）の結果が無い")
    rel = f"{C.SPAN_CHECK_DIR_REL}/{files[-1]}"
    sc = C.read_json(C.P(rel))
    errs = []
    if not sc.get("all_ok"):
        errs.append("all_ok でない")
    if sc.get("pairs_digest") != C.pairs_digest(pairs):
        errs.append("対の digest が違う")
    for k in (C.EVAL_REL, C.INJ_REL, C.APPK_JA_CKPT_REL):
        if sc.get("code_md5", {}).get(k) != C.PINNED_MD5[k]:
            errs.append(f"{k} の md5 が確かめの時点と違う")
    pf = C.P(C.PAIR_FILE_REL)
    if not os.path.exists(pf) or C.md5_file(pf) != sc.get("pair_file", {}).get("md5"):
        errs.append(f"{C.PAIR_FILE_REL} が無いか、確かめの時点と md5 が違う")
    if errs:
        raise SystemExit(f"[appc_ja] {rel}: " + " / ".join(errs))
    return rel, sc


def pairs_with_norm(pairs, sc):
    """Ctx が使う: 正規化後の文（スパンの呼び出しに渡る文）と、CPU で確かめたスパン."""
    fs = sc["fixed_spans"]
    out = []
    for p in pairs:
        f = fs[str(p["pair_idx"])]
        out.append(dict(p, norm_text_a=f["norm_text_a"], norm_text_b=f["norm_text_b"]))
    span_ref = {int(k): {"a": v["a"], "b": v["b"]} for k, v in fs.items()}
    return out, span_ref


def load_status(path, check_seeds):
    if not path:
        raise SystemExit("[appc_ja] --status_file（appc_ja_scan.py の出力）が要る")
    p = path if os.path.isabs(path) else C.P(path)
    st = C.read_json(p)
    for s in check_seeds:
        if st.get("fingerprint", {}).get(str(s)) != C.seed_fingerprint(s):
            raise SystemExit(f"[appc_ja] 状態ファイルが古い（seed {s} の記録が集計の後に変わった）。"
                             "appc_ja_scan.py を流し直す")
    return os.path.relpath(p, ROOT), st


def resolve_batch(args, pairs):
    """(batch 名, 対象の対, 種類, 根拠) を返す（appc_en_gen.resolve_batch と同じ決まり）."""
    if args.retry is not None:
        if not args.pairs:
            raise SystemExit("[appc_ja] --retry には --pairs が要る")
        targets = sorted({int(x) for x in args.pairs.split(",") if x.strip()})
        name = f"retry{args.retry:02d}"
        if os.path.exists(C.P(f"{C.batch_dir_rel(args.seed, name)}/manifest.json")):
            return name, targets, "retry", {"resume": True}
        st_rel, st = load_status(args.status_file, [s for s in C.SEEDS if s <= args.seed])
        fin = st.get("final", {}).get(str(args.seed), {})
        bad = [(p, fin.get(str(p), {}).get("status")) for p in targets
               if fin.get(str(p), {}).get("status") not in C.RETRY_SAME_SEED]
        if bad:
            raise SystemExit(f"[appc_ja] 同じ seed のやり直しは {C.RETRY_SAME_SEED} の対だけ: {bad}")
        return name, targets, "retry", {"status_file": st_rel}
    if args.part is None:
        raise SystemExit("[appc_ja] --part が要る")
    name = C.part_name(args.part)
    if args.seed == 42:
        return name, C.seed42_part_pairs(args.part), "part", {}
    if os.path.exists(C.P(f"{C.batch_dir_rel(args.seed, name)}/manifest.json")):
        man = C.read_json(C.P(f"{C.batch_dir_rel(args.seed, name)}/manifest.json"))
        return name, man["targets"], "part", {"resume": True}
    st_rel, st = load_status(args.status_file, [s for s in C.SEEDS if s < args.seed])
    allt = st.get("next_targets", {}).get(str(args.seed))
    base = st.get("next_part_base", {}).get(str(args.seed))
    if allt is None or base is None:
        raise SystemExit(f"[appc_ja] 状態ファイルに seed {args.seed} の対象が無い（前の seed が終わっていない）")
    k = args.part - base
    if k < 0:
        raise SystemExit(f"[appc_ja] part {args.part} は状態ファイルを作った時点で既にあった番号（{base} 以上を使う）")
    then = set(st.get("batches", {}).get(str(args.seed), {}))
    for n, man in C.list_batches(args.seed):
        if n in then:
            continue
        if not (n.startswith("part") and int(n[4:]) >= base and man.get("basis", {}).get("status_file") == st_rel):
            raise SystemExit(f"[appc_ja] 状態ファイルが古い（seed {args.seed} に {n} が後から作られた）。scan を流し直す")
    chunk = allt[k * C.PART_SIZE:(k + 1) * C.PART_SIZE]
    if not chunk:
        raise SystemExit(f"[appc_ja] seed {args.seed} の part {args.part} は空（残り {len(allt)} 対, 先頭の番号 {base}）")
    return name, chunk, "part", {"status_file": st_rel, "n_remaining_at_status": len(allt), "part_base": base}


def batch_state(batch_abs, targets):
    runs = C.list_runs(batch_abs)
    complete = any(e.get("ev") == "main_end" and e.get("error") is None for r in runs for e in C.read_events(r))
    ck = os.path.join(batch_abs, "checkpoint.json")
    try:
        rows = C.read_json(ck).get("rows", [])
    except Exception as ex:  # noqa: BLE001  壊れた checkpoint は消さず、著者の判断にする
        raise SystemExit(f"[appc_ja] {ck} が読めない（{type(ex).__name__}）。消さずに著者の判断を仰ぐ")
    done = {r["pair_idx"] for r in rows}
    real = {r["pair_idx"] for r in rows if not r.get("placeholder")}
    return {"runs": [os.path.basename(r) for r in runs], "complete": complete,
            "rows_real": sorted(real & set(targets)), "would_run": [t for t in targets if t not in done]}


def prepare_batch(args, name, targets, kind, basis, pairs, span_rel, create):
    rel = C.batch_dir_rel(args.seed, name)
    absd = C.P(rel)
    man_p = os.path.join(absd, "manifest.json")
    if os.path.exists(man_p):
        man = C.read_json(man_p)
        errs = []
        if man.get("seed") != args.seed:
            errs.append("seed")
        if man.get("targets") != targets:
            errs.append(f"targets（manifest {len(man.get('targets', []))} 対 / 今 {len(targets)} 対）")
        if man.get("pairs_digest") != C.pairs_digest(pairs):
            errs.append("pairs_digest")
        if man.get("pinned_md5") != C.PINNED_MD5:
            errs.append("pinned_md5")
        if man.get("pair_file_md5") != C.md5_file(C.P(C.PAIR_FILE_REL)):
            errs.append("pair_file_md5")
        if errs:
            raise SystemExit(f"[appc_ja] {rel} の manifest と合わない: {errs}")
        return rel, man, False
    if not create:
        return rel, None, True
    os.makedirs(absd, exist_ok=False)
    man = {
        "item": "Appendix C tab:ja_crossattn_phoneme, JA rows rerun on the Appendix K JA pairs (fixed span)",
        "seed": args.seed, "batch": name, "kind": kind, "targets": targets, "basis": basis,
        "created": C.stamp(), "created_order": int(time.time() * 1000),
        "pairs": f"{C.APPK_JA_CKPT_REL} results (Appendix K JA)", "pairs_digest": C.pairs_digest(pairs),
        "pair_file": C.PAIR_FILE_REL, "pair_file_md5": C.md5_file(C.P(C.PAIR_FILE_REL)),
        "pair_redirect": f"main() open('{C.OLD_PAIR_SRC_REL}') -> {C.PAIR_FILE_REL}",
        "production_script": C.EVAL_REL, "prod_args": C.PROD_ARGS, "output_dir_rel": rel,
        "span_check": span_rel, "pinned_md5": C.PINNED_MD5, "vendor_pinned_md5": C.VENDOR_PINNED_MD5,
        "seed_policy": "42 for all 300; 43 only for pairs unusable at 42 (unaligned/silent/generation_failed); "
                       "44 only for pairs unusable at 43; never by metric values",
    }
    C.write_json_new(man, man_p)
    rows = [{"pair_idx": i, "placeholder": True} for i in range(C.N_PAIRS) if i not in set(targets)]
    C.write_json_new({"rows": rows}, os.path.join(absd, "checkpoint.json"))
    return rel, man, True


# ---------------------------------------------------------------- 記録を取る包み
class Ctx:
    """本番の呼び出しから、対・条件ごとの記録と wav を作る（本番の値は変えない）."""

    def __init__(self, pairs, targets, run_rel, span_ref):
        self.pairs = {p["pair_idx"]: p for p in pairs}
        self.by_norm_text_a = {p["norm_text_a"]: p["pair_idx"] for p in pairs}
        assert len(self.by_norm_text_a) == len(pairs), "正規化後の text_a が対ごとに一意でない"
        self.span_ref = span_ref
        self.targets = set(targets)
        self.run_rel = run_rel
        os.makedirs(C.P(f"{run_rel}/wav"), exist_ok=True)
        self.f = open(C.P(f"{run_rel}/events.jsonl"), "a", encoding="utf-8")
        self.cur = None
        self.cur_labels = []
        self.mems = {}
        self.align_k = 0
        self.cur_align = None
        self.pending_spans = []
        self.pending_durations = []
        self.device_map = None
        self.device_offload = None
        self.n_gen = 0
        self.n_redirect = 0
        self.redirect_log = []

    def emit(self, rec):
        rec.setdefault("t", round(time.time(), 3))
        self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def start_pair(self, pidx):
        self.cur = pidx
        self.cur_labels = []
        self.mems = {}
        self.align_k = 0
        self.cur_align = None

    def side_of(self, pair, full_text, entity):
        if pair is None:
            return "other"
        if full_text == pair["norm_text_a"] and entity == pair["word_a"]:
            return "a"
        if full_text == pair["norm_text_b"] and entity == pair["word_b"]:
            return "b"
        return "other"

    # -- スパン・長さ推定
    def on_span(self, full_text, entity, add_bos, add_eos, result):
        self.pending_spans.append({"full_text": full_text, "entity": entity, "add_bos": add_bos,
                                   "add_eos": add_eos, "result": list(result) if result else None})

    def on_duration(self, target_text, target_lang, sec, g2p_delta):
        self.pending_durations.append({"target_text": target_text, "target_lang": target_lang,
                                       "sec": sec, "g2p_calls": g2p_delta})

    # -- 記憶の確かめ（長さが違えば継ぎ合わせ）
    def mem_check(self):
        import numpy as np

        def maxdiff(x, y):
            if x is None or y is None or x.shape != y.shape:
                return None
            return float(np.abs(x - y).max()) if x.size else 0.0
        out = {"ok": False}
        bl, inj, blm, injm = (self.mems.get(k) for k in ("bl", "inj", "bl_mask", "inj_mask"))
        ref = self.span_ref.get(self.cur) or {}
        if any(v is None for v in (bl, inj, blm, injm)) or not ref.get("a") or not ref.get("b"):
            out["reason"] = "memory or span missing"
            return out
        (a0, a1), (b0, b1) = ref["a"], ref["b"]
        la, lb = a1 - a0, b1 - b0
        L = int(bl.shape[1])
        out.update({"len_bl": L, "len_inj": int(inj.shape[1]), "la": la, "lb": lb,
                    "mode": "replace" if la == lb else "splice"})
        out["len_ok"] = bool(inj.shape[1] == L - la + lb and blm.shape == bl.shape and injm.shape == inj.shape)
        scale = float(np.abs(bl).max()) if bl.size else 1.0
        out["d_blmask_bl"] = maxdiff(blm, bl)
        out["d_injmask_inj"] = maxdiff(injm, inj)
        if out["len_ok"]:
            out["d_prefix"] = maxdiff(inj[:, :a0], bl[:, :a0])
            out["d_suffix"] = maxdiff(inj[:, a0 + lb:], bl[:, a1:])
            out["d_entity"] = maxdiff(inj[:, a0:a0 + lb], bl[:, a0:a1]) if la == lb else None
        tol = 1e-3 * max(scale, 1e-6)
        ok = out["len_ok"] and all(out.get(k) is not None and out[k] <= tol
                                   for k in ("d_blmask_bl", "d_injmask_inj", "d_prefix", "d_suffix"))
        if ok and la == lb:
            ok = out.get("d_entity") is not None and out["d_entity"] > tol
        out["scale"] = scale
        out["tol"] = tol
        out["ok"] = bool(ok)
        return out

    # -- 生成
    def on_gen(self, a, out, exception, elapsed):
        import numpy as np
        import soundfile as sf
        spans, self.pending_spans = self.pending_spans, []
        durs, self.pending_durations = self.pending_durations, []
        self.n_gen += 1
        anomalies = []
        text = a.get("target_text")
        masked, mcols = False, None
        try:
            m = (as_np(a.get("encoder_attention_mask")) == 0).reshape(-1)
            mcols = [int(i) for i in np.nonzero(m)[0]]
            masked = bool(mcols)
        except Exception as ex:  # noqa: BLE001
            anomalies.append(f"mask: {type(ex).__name__}")
        pidx = self.by_norm_text_a.get(spans[0]["full_text"]) if spans else self.cur
        pair = self.pairs.get(pidx) if pidx is not None else None
        sides = tuple(self.side_of(pair, s["full_text"], s["entity"]) for s in spans)
        for s, sd in zip(spans, sides):
            s["side"] = sd
        role = "a" if pair and text == pair["text_a"] else ("b" if pair and text == pair["text_b"] else "other")
        sig = (role, masked, sides)
        label = next((k for k, v in C.SLOT_SIGNATURE.items() if v == sig), None)
        if label is None:
            anomalies.append(f"signature={sig}")
        if pidx is None or pidx not in self.targets:
            anomalies.append(f"pair={pidx} is not a target of this batch")
        if len(durs) != 1:
            anomalies.append(f"duration calls={len(durs)}")
        if label == "bl" and pidx is not None:
            if pidx == self.cur:
                anomalies.append("BL twice for the same pair in one run")
            self.start_pair(pidx)
        elif pidx != self.cur:
            anomalies.append(f"pair {pidx} != current {self.cur} without BL")
        if label is not None:
            if label in self.cur_labels:
                anomalies.append(f"duplicate {label}")
            self.cur_labels.append(label)
        lab = label or f"unknown{self.n_gen:05d}"
        if self.device_map is None:
            dm = getattr(a.get("model"), "hf_device_map", None)
            self.device_map = {str(k): str(v) for k, v in dm.items()} if isinstance(dm, dict) else {}
            self.device_offload = any(v in ("cpu", "disk") for v in self.device_map.values())
            self.emit({"ev": "device_map", "device_map": self.device_map, "device_offload": self.device_offload})
        enc_len = None
        try:
            mem = as_np(a.get("modified_memory"), floating=True)
            enc_len = int(mem.shape[1])
            if label is not None:
                self.mems[label] = mem
        except Exception as ex:  # noqa: BLE001
            anomalies.append(f"memory: {type(ex).__name__}")
        mem_check = self.mem_check() if label == "inj_mask" else None
        rec = {"ev": "gen", "pair_idx": pidx, "label": lab, "signature": [role, masked, list(sides)],
               "signature_ok": not anomalies, "anomalies": anomalies, "spans": spans, "masked": masked,
               "masked_cols": mcols, "enc_len": enc_len, "returned_none": out is None and exception is None,
               "exception": exception, "elapsed_s": round(elapsed, 3), "device_offload": self.device_offload,
               "mem_check": mem_check, "target_text": text, "durations": durs,
               "pyojt_g2p_calls_total": PYOJT["g2p_calls"]}
        if out is not None:
            try:
                x = as_np(out[0], floating=True).reshape(-1)  # 本番と同じく先頭のバッチ
                sr = int(a.get("codec_audio_sr"))
                base = f"s{pidx:04d}_{lab}" if pidx is not None else f"unattributed_{self.n_gen:05d}"
                rel = f"{self.run_rel}/wav/{base}.wav"
                k = 1
                while os.path.exists(C.P(rel)):
                    k += 1
                    rel = f"{self.run_rel}/wav/{base}_dup{k}.wav"
                    rec["signature_ok"] = False
                    rec["anomalies"].append("wav name already used in this run")
                sf.write(C.P(rel), x, sr, subtype="FLOAT")  # 評価に使った値をそのまま残す（無損失）
                a_ = SC.assess_file(C.P(rel))
                rec.update({"wav": rel, "sr": sr, "n_samples": int(x.size), "md5": C.md5_file(C.P(rel)),
                            "silence": {k2: a_[k2] for k2 in ("rms", "peak", "active_frac", "dur", "silent",
                                                               "silent_reasons", "criterion_id")}})
            except Exception as ex:  # noqa: BLE001  保存の失敗で本番の処理を止めない（capture_error になる）
                rec["save_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        self.emit(rec)

    # -- 整列
    def next_align_label(self):
        return C.ALIGN_ORDER[self.align_k] if self.align_k < len(C.ALIGN_ORDER) else f"extra{self.align_k}"

    def on_transcribe(self, chunks, audio_np, sr):
        lab = self.next_align_label()
        self.align_k += 1
        self.cur_align = lab
        self.emit({"ev": "align_tr", "pair_idx": self.cur, "label": lab, "n_chunks": len(chunks or []),
                   "chunks": [{"w": c.get("word"), "s": c.get("start"), "e": c.get("end")} for c in (chunks or [])],
                   "audio_none": audio_np is None,
                   "dur": (round(len(audio_np) / sr, 4) if audio_np is not None else None)})

    def on_find(self, entity, input_text, result, exact):
        self.emit({"ev": "align_find", "pair_idx": self.cur, "label": self.cur_align, "entity": entity,
                   "input_text": input_text, "result": list(result) if result else None, "exact": exact})

    def on_slice(self, res, sr, s_idx, e_idx):
        self.emit({"ev": "align_slice", "pair_idx": self.cur, "label": self.cur_align, "chunk_range": [s_idx, e_idx],
                   "slice_sec": (round(len(res) / sr, 4) if res is not None else None)})

    # -- 例外の行
    def on_print_line(self, s):
        m = PAIR_ERR_RE.match(s)
        if m:
            self.emit({"ev": "err", "kind": "pair_loop", "pair_idx": int(m.group(1)), "line": s.strip()[:400]})
        elif WHISPER_ERR_RE.match(s):
            self.emit({"ev": "err", "kind": "whisper", "pair_idx": self.cur, "label": self.next_align_label(),
                       "line": s.strip()[:400]})


def safe(ctx, fn, *a):
    """記録の失敗で本番の処理を止めない（tap の err として残し、CPU 側で capture_error にする）."""
    try:
        fn(*a)
    except Exception as ex:  # noqa: BLE001
        builtins.print(f"[appc_ja] tap failed: {type(ex).__name__}: {ex}", flush=True)
        try:
            ctx.emit({"ev": "err", "kind": "tap", "pair_idx": ctx.cur, "line": f"{type(ex).__name__}: {str(ex)[:300]}"})
        except Exception:  # noqa: BLE001
            pass


def make_gen_tap(orig, ctx):
    sig = inspect.signature(orig)

    def gen_tap(*args, **kwargs):
        try:
            b = sig.bind(*args, **kwargs)
            b.apply_defaults()
            a = dict(b.arguments)
        except Exception:  # noqa: BLE001
            a = {}
        t0 = time.time()
        try:
            out = orig(*args, **kwargs)
        except BaseException as ex:
            safe(ctx, ctx.on_gen, a, None, f"{type(ex).__name__}: {str(ex)[:300]}", time.time() - t0)
            raise
        safe(ctx, ctx.on_gen, a, out, None, time.time() - t0)
        return out
    gen_tap.__wrapped__ = orig
    return gen_tap


def make_span_tap(orig, ctx):
    def span_tap(text_tokenizer, full_text, entity_text, add_bos=0, add_eos=0):
        res = orig(text_tokenizer, full_text, entity_text, add_bos=add_bos, add_eos=add_eos)
        safe(ctx, ctx.on_span, full_text, entity_text, add_bos, add_eos, res)
        return res
    span_tap.__wrapped__ = orig
    return span_tap


def make_duration_tap(orig, ctx):
    def duration_tap(*args, **kwargs):
        n0 = PYOJT["g2p_calls"]
        res = orig(*args, **kwargs)
        safe(ctx, ctx.on_duration, kwargs.get("target_text", args[0] if args else None),
             kwargs.get("target_lang"), res, PYOJT["g2p_calls"] - n0)
        return res
    duration_tap.__wrapped__ = orig
    return duration_tap


def make_open_redirect(ctx, target_abs):
    """本番 module の open: 5/11 の対のパス（相対、読み）だけを付録 K の対のファイルに向ける. 他はそのまま."""
    want_md5 = C.md5_file(target_abs)

    def open_redirect(file, mode="r", *a, **k):
        if file == C.OLD_PAIR_SRC_REL and mode in ("r", "rt"):
            ctx.n_redirect += 1
            md5 = C.md5_file(target_abs)
            ctx.redirect_log.append({"from": file, "to": os.path.relpath(target_abs, ROOT), "md5": md5})
            safe(ctx, ctx.emit, {"ev": "pairs_redirect", "from": file, "to": os.path.relpath(target_abs, ROOT),
                                 "md5": md5, "md5_ok": md5 == want_md5, "n": ctx.n_redirect})
            if md5 != want_md5:
                raise RuntimeError("対のファイルが起動の後に変わった")
            k.setdefault("encoding", "utf-8")
            return builtins.open(target_abs, mode, *a, **k)
        return builtins.open(file, mode, *a, **k)
    open_redirect.__wrapped__ = builtins.open
    return open_redirect


class PrintTap:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, *args, **kwargs):
        builtins.print(*args, **kwargs)
        s = " ".join(str(x) for x in args)
        try:
            self.ctx.on_print_line(s)
        except Exception:  # noqa: BLE001
            pass
        m = LOADED_RE.match(s)
        if m:  # 対を作る前に、読み替えが 1 回だけ効いて 300 対になったことを確かめる（違えば本番を止める）
            ok = int(m.group(1)) == C.N_PAIRS and m.group(2) == C.LANG and self.ctx.n_redirect == 1
            self.ctx.emit({"ev": "pairs_loaded", "line": s.strip(), "n_redirect": self.ctx.n_redirect, "ok": ok})
            if not ok:
                raise RuntimeError(f"[appc_ja] 対の読み替えが効いていない: {s.strip()} n_redirect={self.ctx.n_redirect}")


def install_taps(M, FA, INJ, ctx):
    INJ.generate_with_modified_encoder = make_gen_tap(INJ.generate_with_modified_encoder, ctx)
    INJ.find_entity_token_range = make_span_tap(INJ.find_entity_token_range, ctx)
    INJ.estimate_duration = make_duration_tap(INJ.estimate_duration, ctx)

    orig_tw = FA.WhisperAligner.transcribe_words

    def tw_tap(self_, audio, sr_in, language="en"):
        chunks, audio_np = orig_tw(self_, audio, sr_in, language=language)
        safe(ctx, ctx.on_transcribe, chunks, audio_np, getattr(self_, "target_sr", 16000))
        return chunks, audio_np
    FA.WhisperAligner.transcribe_words = tw_tap

    orig_find = M.find_entity_chunks

    def find_tap(chunks, entity, lang, input_text=None):
        res = orig_find(chunks, entity, lang, input_text=input_text)
        try:
            nc = "".join(FA.normalize_for_match(c["word"], lang) for c in (chunks or []))
            ne = FA.normalize_for_match(entity, lang)
            exact = bool(ne) and ne in nc
        except Exception:  # noqa: BLE001
            exact = None
        safe(ctx, ctx.on_find, entity, input_text, res, exact)
        return res
    M.find_entity_chunks = find_tap

    orig_slice = M.slice_entity_audio

    def slice_tap(audio_np, sr, chunks, start_idx, end_idx, *a, **k):
        res = orig_slice(audio_np, sr, chunks, start_idx, end_idx, *a, **k)
        safe(ctx, ctx.on_slice, res, sr, start_idx, end_idx)
        return res
    M.slice_entity_audio = slice_tap

    M.open = make_open_redirect(ctx, C.P(C.PAIR_FILE_REL))
    tap = PrintTap(ctx)
    M.print = tap
    FA.print = tap
    return {"generate_tapped": getattr(INJ.generate_with_modified_encoder, "__wrapped__", None) is not None,
            "span_tapped": getattr(INJ.find_entity_token_range, "__wrapped__", None) is not None,
            "duration_tapped": getattr(INJ.estimate_duration, "__wrapped__", None) is not None,
            "open_redirect_installed": getattr(M.open, "__wrapped__", None) is builtins.open,
            "print_tapped": M.print is tap and FA.print is tap}


# ---------------------------------------------------------------- GPU
def setup_offline(W):
    info = {"hf_offline": bool(W.hf_offline()), "offline_tokenizer_shim": []}
    if not info["hf_offline"]:
        raise SystemExit("[appc_ja] HF_HUB_OFFLINE=1 で流す")
    ok, pre = W.offline_preflight()
    info["offline_preflight_ok"] = bool(ok)
    info["offline_preflight"] = pre
    if not ok:
        raise SystemExit(f"[appc_ja] 生成の経路で読む repo が手元のキャッシュに無い: {pre}")
    W.install_offline_tokenizer_shim(info["offline_tokenizer_shim"])
    return info


def pyojt_refs(real):
    """pyopenjtalk の本物の module を属性に持つ module の名前（pyopenjtalk 自身を除く）. 期待: なし."""
    out = []
    for name, m in list(sys.modules.items()):
        if m is None or name == "pyopenjtalk" or name.startswith("pyopenjtalk."):
            continue
        try:
            if any(v is real for v in vars(m).values()):
                out.append(name)
        except Exception:  # noqa: BLE001
            continue
    return sorted(out)


def duration_preflight_ja(pairs):
    """vendor の長さ推定が日本語で pyopenjtalk を使う経路にあることを確かめ、g2p だけを通す包みに替える.

    pyopenjtalk なし（len(text) で数える）だと 5/11 の実行と長さが変わるので、モデルを読む前に止まる。
    """
    DE = sys.modules.get("duration_estimator")
    info = {"module_file": getattr(DE, "__file__", None)}
    if DE is None or os.path.realpath(DE.__file__) != VENDOR_DE_REAL:
        raise SystemExit(f"[appc_ja] duration_estimator が vendor のファイルでない: {info}")
    real = getattr(DE, "pyopenjtalk", None)
    info["pyopenjtalk_in_vendor"] = real is not None
    if real is None or not hasattr(real, "g2p"):
        raise SystemExit("[appc_ja] vendor の長さ推定が pyopenjtalk なし（len(text)）になっている。5/11 の条件とずれるので止める")
    info["pyopenjtalk_file"] = getattr(real, "__file__", None)
    info["pyopenjtalk_version"] = getattr(real, "__version__", None)
    DE.pyopenjtalk = G2PProxy(real, DE._phoneme_count_ja.__code__)
    info["refs_to_real_module_after_wrap"] = pyojt_refs(real)
    if info["refs_to_real_module_after_wrap"] or PYOJT["blocked_imports"]:
        raise SystemExit(f"[appc_ja] pyopenjtalk が vendor の長さ推定器の外で読み込まれている（モデルを読む前に止める）: "
                         f"{info['refs_to_real_module_after_wrap']} {PYOJT['blocked_imports']}")
    t = pairs[0]["text_a"]
    n0 = PYOJT["g2p_calls"]
    info["sample"] = {"text": t, "phonemes_vendor": DE._phoneme_count_ja(t), "len_text": len(t),
                      "estimate_sec": DE.estimate_duration(target_text=t, target_lang="ja"),
                      "g2p_calls": PYOJT["g2p_calls"] - n0}
    if info["sample"]["g2p_calls"] != 2:
        raise SystemExit(f"[appc_ja] 長さ推定の g2p の呼び出しが想定と違う: {info['sample']}")
    info["allowed_imports"] = list(PYOJT["allowed_imports"])
    info["blocked_imports"] = list(PYOJT["blocked_imports"])
    return info, real


def run_gpu(args, pairs, span_rel, span_ref):
    if not IN_CONTAINER or not GPU_MODE:
        raise SystemExit("[appc_ja] GPU の実行は slurm_run.sh の中で（ホストでは --dry_run / --selftest）")
    name, targets, kind, basis = resolve_batch(args, pairs)
    batch_rel, man, created = prepare_batch(args, name, targets, kind, basis, pairs, span_rel, create=True)
    batch_abs = C.P(batch_rel)
    st = batch_state(batch_abs, targets)
    print(f"[appc_ja] seed={args.seed} batch={name} targets={len(targets)} created={created} "
          f"runs={st['runs']} complete={st['complete']} would_run={len(st['would_run'])}", flush=True)
    if st["complete"]:
        print("[appc_ja] この batch は終わっている（main() が例外なく終わった起動がある）。何もしない", flush=True)
        return 0
    if not st["would_run"]:
        print("[appc_ja] 対象の行は全部 checkpoint にある。何もしない", flush=True)
        return 0
    if st["runs"] and not args.force_resume:
        last = os.path.join(batch_abs, "runs", st["runs"][-1])
        if not os.path.exists(os.path.join(last, "end.json")):
            ts = [os.path.getmtime(os.path.join(last, f)) for f in ("events.jsonl", "launch.json")
                  if os.path.exists(os.path.join(last, f))] or [os.path.getmtime(last)]
            age_min = (time.time() - max(ts)) / 60
            if age_min < 20:
                raise SystemExit(f"[appc_ja] 前の起動 {st['runs'][-1]} の記録が {age_min:.1f} 分前に更新されている"
                                 "（まだ動いている恐れ）。squeue で止まっていることを確かめ、20 分待つか --force_resume")
    r = len(st["runs"]) + 1
    run_rel = f"{batch_rel}/runs/run{r:02d}"
    os.makedirs(C.P(run_rel), exist_ok=False)
    with open(C.P(f"{batch_rel}/checkpoint.json"), "rb") as fi, open(C.P(f"{run_rel}/checkpoint_at_launch.json"), "xb") as fo:
        fo.write(fi.read())

    import torch
    torch.set_num_threads(THREADS)
    if not torch.cuda.is_available():
        raise SystemExit("[appc_ja] GPU が見えない")
    W = load_module_from_file("_appc_ja_offline_shim", C.P(C.SHIM_REL))
    off = setup_offline(W)
    M = load_module_from_file("eval_crossattn_entity_per", C.P(C.EVAL_REL))
    FA = sys.modules["forced_alignment_utils"]
    if os.path.realpath(FA.__file__) != os.path.realpath(C.P(C.FA_REL)):
        raise SystemExit(f"[appc_ja] forced_alignment_utils が本番のファイルでない: {FA.__file__}")
    import encoder_injection_t5gemma as INJ
    if os.path.realpath(INJ.__file__) != os.path.realpath(C.P(C.INJ_REL)):
        raise SystemExit(f"[appc_ja] encoder_injection_t5gemma が本番のファイルでない: {INJ.__file__}")
    dur, real_pyojt = duration_preflight_ja(pairs)
    # 本番の JA の分岐の文を、読み替えた open で実行して付録 K の 300 対になることを、この process でも確かめる
    def _open_pairs_only(f, mode="r"):
        if f != C.OLD_PAIR_SRC_REL:
            raise AssertionError(f"JA の分岐が想定外のファイルを開いた: {f}")
        return builtins.open(C.P(C.PAIR_FILE_REL), mode, encoding="utf-8")
    loaded = C.run_production_ja_loader(_open_pairs_only)
    if [tuple(p[k] for k in C.PAIR_KEYS) for p in loaded] != [tuple(p[k] for k in C.PAIR_KEYS) for p in pairs]:
        raise SystemExit("[appc_ja] 本番の JA の分岐で読んだ対が付録 K の 300 対と違う")
    ctx = Ctx(pairs, targets, run_rel, span_ref)
    taps = install_taps(M, FA, INJ, ctx)
    vend = {n: {"file": getattr(sys.modules.get(n), "__file__", None),
                "md5": C.md5_file(sys.modules[n].__file__) if getattr(sys.modules.get(n), "__file__", None) else None}
            for n in ("data.tokenizer", "duration_estimator", "inference_tts_utils")}
    import transformers
    launch = {
        "item": man["item"], "seed": args.seed, "batch": name, "run": r, "targets": targets,
        "would_run": st["would_run"], "rows_already": st["rows_real"],
        "argv_wrapper": sys.argv, "prod_argv": C.PROD_ARGS + ["--output_dir", batch_abs],
        "span_check": span_rel, "pair_file": C.PAIR_FILE_REL, "pair_file_md5": C.md5_file(C.P(C.PAIR_FILE_REL)),
        "code_md5": C.code_md5(ROOT, OWN_CODE), "vendor_modules": vend,
        "taps": taps, "offline": off, "duration_preflight": dur, "pyopenjtalk_policy": PYOJT["mode"],
        "env": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda_device": torch.cuda.get_device_name(0),
                "torch_num_threads": torch.get_num_threads(), "env_at_launch": ENV_AT_LAUNCH,
                "env_set_by_script": CACHE_ENV,
                "now": {k: os.environ.get(k) for k in THREAD_VARS + CACHE_VARS + ("CUDA_VISIBLE_DEVICES",
                                                                                 "NVIDIA_VISIBLE_DEVICES")}},
        "date": C.stamp(),
    }
    C.write_json_new(launch, C.P(f"{run_rel}/launch.json"))
    ctx.emit({"ev": "launch", "seed": args.seed, "batch": name, "run": r, "would_run": st["would_run"]})
    M.SEED = args.seed
    saved = sys.argv
    sys.argv = [C.P(C.EVAL_REL)] + C.PROD_ARGS + ["--output_dir", batch_abs]
    t0 = time.time()
    err = None
    try:
        M.main()
    except BaseException as ex:  # noqa: BLE001  例外でも、写し取った wav と記録は残す
        err = f"{type(ex).__name__}: {str(ex)[:300]}"
        builtins.print(f"[appc_ja] main() ended with {err}", flush=True)
        traceback.print_exc()
    finally:
        sys.argv = saved
    elapsed = round((time.time() - t0) / 60, 2)
    pyojt_end = {"g2p_calls": PYOJT["g2p_calls"], "blocked_imports": PYOJT["blocked_imports"],
                 "bad_callers": PYOJT["bad_callers"], "allowed_imports": PYOJT["allowed_imports"],
                 "refs_to_real_module": pyojt_refs(real_pyojt)}
    if pyojt_end["blocked_imports"] or pyojt_end["bad_callers"] or pyojt_end["refs_to_real_module"]:
        err = (err or "") + " | pyopenjtalk outside the vendor duration estimator: " + json.dumps(
            {k: pyojt_end[k] for k in ("blocked_imports", "bad_callers", "refs_to_real_module")}, ensure_ascii=False)
    ctx.emit({"ev": "main_end", "error": err, "elapsed_min": elapsed, "n_redirect": ctx.n_redirect,
              "pyopenjtalk": pyojt_end})
    st2 = batch_state(batch_abs, targets)
    end = {"error": err, "elapsed_min": elapsed, "n_gen_calls": ctx.n_gen, "rows_real": len(st2["rows_real"]),
           "would_run_after": st2["would_run"], "n_redirect": ctx.n_redirect, "pyopenjtalk": pyojt_end,
           "date": C.stamp()}
    C.write_json_new(end, C.P(f"{run_rel}/end.json"))
    builtins.print(f"[appc_ja] END seed={args.seed} batch={name} run={r} error={err} elapsed={elapsed}min "
                   f"gen_calls={ctx.n_gen} rows={len(st2['rows_real'])}/{len(targets)} "
                   f"g2p_calls={PYOJT['g2p_calls']}", flush=True)
    builtins.print("[appc_ja] NEXT: ホストで taskset -c 0-12 python3 scripts/spanfix/appc_ja_scan.py", flush=True)
    return 0 if err is None else 1


# ---------------------------------------------------------------- CPU の試運転
def dry_run(args, pairs, span_rel, static):
    name, targets, kind, basis = resolve_batch(args, pairs)
    batch_rel, man, would_create = prepare_batch(args, name, targets, kind, basis, pairs, span_rel, create=False)
    st = batch_state(C.P(batch_rel), targets) if not would_create else {"would_run": targets, "complete": False}
    W = load_module_from_file("_appc_ja_offline_shim", C.P(C.SHIM_REL))
    ok, pre = W.offline_preflight()
    SC.selftest()
    calls = []

    def open_redirect(f, mode="r"):
        calls.append(f)
        assert f == C.OLD_PAIR_SRC_REL, f
        return builtins.open(C.P(C.PAIR_FILE_REL), mode, encoding="utf-8")
    loaded = C.run_production_ja_loader(open_redirect)
    loader_ok = [tuple(p[k] for k in C.PAIR_KEYS) for p in loaded] == [tuple(p[k] for k in C.PAIR_KEYS) for p in pairs]
    n = len(st["would_run"])
    slurm = C.suggest_slurm_time(n) if n else None
    sel = (f"--seed {args.seed} --retry {args.retry} --pairs {args.pairs}" if args.retry is not None
           else f"--seed {args.seed} --part {args.part}")
    if args.status_file:
        st_abs = args.status_file if os.path.isabs(args.status_file) else C.P(args.status_file)
        sel += f" --status_file /mnt/{os.path.relpath(st_abs, ROOT)}"
    cmd = C.format_gpu_command(slurm, sel) if slurm else None
    rep = {"mode": "dry_run", "seed": args.seed, "batch": name, "kind": kind, "basis": basis,
           "n_targets": len(targets), "targets_head": targets[:10], "would_create_batch": would_create,
           "batch_state": st, "static_checks": static, "span_check": span_rel,
           "production_loader_on_pair_file_ok": loader_ok, "open_calls": calls,
           "offline_preflight_ok": ok, "offline_preflight": pre, "pinned_md5_ok": not C.pinned_errors(ROOT),
           "suggested_slurm_time": slurm, "expected_minutes": C.expected_minutes(n) if n else None,
           "gpu_command": cmd, "pyopenjtalk_policy_here": PYOJT["mode"],
           "code_md5": C.code_md5(ROOT, OWN_CODE), "date": C.stamp()}
    path = C.P(f"{C.DRYRUN_DIR_REL}/dryrun_{C.stamp()}_seed{args.seed}_{name}.json")
    C.write_json_new(rep, path)
    print(f"[appc_ja] DRY RUN seed={args.seed} batch={name} targets={len(targets)} (head {targets[:5]}) "
          f"create={would_create} would_run={n} offline_ok={ok} loader_ok={loader_ok} "
          f"SLURM_TIME={slurm} expected≈{rep['expected_minutes']} min")
    print(f"[appc_ja] GPU command: {cmd}")
    print(f"[appc_ja] wrote {os.path.relpath(path, ROOT)}")
    return 0 if (ok and loader_ok) else 1


def selftest_pyojt_guard():
    """import の関所と g2p の包みを、偽の pyopenjtalk で確かめる（ホスト。終わったら元に戻す）."""
    import types
    saved_mod = sys.modules.get("pyopenjtalk", "absent")
    saved_import = builtins.__import__
    saved_state = {k: (list(v) if isinstance(v, list) else v) for k, v in PYOJT.items()}
    fake = types.ModuleType("pyopenjtalk")
    fake.g2p = lambda text, kana=False: " ".join(["a"] * len(text))
    res = {}
    try:
        sys.modules["pyopenjtalk"] = fake
        builtins.__import__ = pyojt_import_guard
        g_vendor = {"__file__": VENDOR_DE_REAL, "__name__": "duration_estimator", "__package__": ""}
        ns = dict(g_vendor)
        exec("import pyopenjtalk as m", ns)
        res["vendor_import_allowed"] = ns["m"] is fake
        for who in ({"__file__": C.P(C.INJ_REL), "__name__": "encoder_injection_t5gemma", "__package__": ""},
                    {"__file__": os.path.abspath(__file__), "__name__": "appc_ja_gen", "__package__": ""}):
            try:
                exec("from pyopenjtalk import g2p", dict(who))
                res[f"blocked:{who['__name__']}"] = False
            except ImportError:
                res[f"blocked:{who['__name__']}"] = True
        g_inner = {"__file__": "/x/pyopenjtalk/__init__.py", "__name__": "pyopenjtalk", "__package__": "pyopenjtalk"}
        try:
            exec("import pyopenjtalk", dict(g_inner))
            res["inner_import_allowed"] = True
        except ImportError:
            res["inner_import_allowed"] = False
        # g2p の包み: vendor の _phoneme_count_ja を ast で取り出し、包みを持たせて呼ぶ
        src = open(C.VENDOR_DE_FILE, encoding="utf-8").read()
        node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_phoneme_count_ja"][0]
        dns = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), C.VENDOR_DE_FILE, "exec"), dns)
        proxy = G2PProxy(fake, dns["_phoneme_count_ja"].__code__)
        dns["pyopenjtalk"] = proxy
        n0 = PYOJT["g2p_calls"]
        res["phoneme_count_via_proxy"] = dns["_phoneme_count_ja"]("あいう")
        res["g2p_counted"] = PYOJT["g2p_calls"] - n0
        try:
            proxy.g2p("あ")
            res["direct_g2p_blocked"] = False
        except RuntimeError:
            res["direct_g2p_blocked"] = True
        try:
            proxy.extract_fullcontext  # noqa: B018
            res["other_attr_blocked"] = False
        except AttributeError:
            res["other_attr_blocked"] = True
    finally:
        builtins.__import__ = saved_import
        if saved_mod == "absent":
            sys.modules.pop("pyopenjtalk", None)
        else:
            sys.modules["pyopenjtalk"] = saved_mod
        for k, v in saved_state.items():
            PYOJT[k] = v
    want = {"vendor_import_allowed": True, "blocked:encoder_injection_t5gemma": True, "blocked:appc_ja_gen": True,
            "inner_import_allowed": True, "phoneme_count_via_proxy": 3, "g2p_counted": 1,
            "direct_g2p_blocked": True, "other_attr_blocked": True}
    assert res == want, res
    assert sys.modules.get("pyopenjtalk", "absent") is None and builtins.__import__ is _ORIG_IMPORT
    return res


def selftest(pairs, span_rel, sc):
    """偽の生成で、対の特定・ラベル・wav の保存・無音・記憶の確かめ（置換と継ぎ合わせ）・状態の判定・対の読み替え・
    表の書式（今の表の JA 8 セルの再現）・pyopenjtalk の関所を確かめる."""
    import numpy as np
    pairs_n, span_ref = pairs_with_norm(pairs, sc)
    P_ = {p["pair_idx"]: p for p in pairs_n}
    same = [p for p in range(C.N_PAIRS) if sc["per_pair"][p]["same_token_count"]]
    spl = [p for p in range(C.N_PAIRS) if not sc["per_pair"][p]["same_token_count"]][:2]
    assert same and len(spl) == 2, ("置換と継ぎ合わせの対が両方要る", len(same), len(spl))
    rep_ = same[0]
    targets = [rep_] + spl
    run_rel = f"{C.DRYRUN_DIR_REL}/selftest_{C.stamp()}"
    ctx = Ctx(pairs_n, targets, run_rel, span_ref)
    sr = 16000

    def fake_generate(model, text_tokenizer, audio_tokenizer, modified_memory, encoder_attention_mask,
                      encoder_position_ids, target_text, device, codec_audio_sr, codec_sr):
        fake_duration(target_text=target_text, target_lang=None)
        if target_text.endswith("RAISE"):
            raise RuntimeError("fake OOM")
        n = int(0.6 * codec_audio_sr)
        amp = 0.0 if getattr(fake_generate, "silent_next", False) else 0.3
        fake_generate.silent_next = False
        t = np.arange(n, dtype=np.float32) / codec_audio_sr
        return (amp * np.sin(2 * np.pi * 220 * t)).reshape(1, 1, -1)

    def fake_span(text_tokenizer, full_text, entity_text, add_bos=0, add_eos=0):
        for p in pairs_n:
            if full_text == p["norm_text_a"] and entity_text == p["word_a"]:
                return tuple(span_ref[p["pair_idx"]]["a"])
            if full_text == p["norm_text_b"] and entity_text == p["word_b"]:
                return tuple(span_ref[p["pair_idx"]]["b"])
        return None

    dur_tap = make_duration_tap(lambda target_text=None, target_lang=None: 2.0, ctx)

    def fake_duration(**k):
        return dur_tap(**k)

    gen = make_gen_tap(fake_generate, ctx)
    span = make_span_tap(fake_span, ctx)
    rng = np.random.RandomState(0)

    def run_pair(p, raise_at=None, silent_tgt=False):
        q = P_[p]
        (a0, a1), (b0, b1) = span_ref[p]["a"], span_ref[p]["b"]
        L = sc["per_pair"][p]["a"]["n_encoder_tokens"]
        src = rng.randn(1, L, 4).astype(np.float32)
        ent = rng.randn(1, b1 - b0, 4).astype(np.float32)
        inj = np.concatenate([src[:, :a0], ent, src[:, a1:]], axis=1)  # 本番と同じ継ぎ合わせ（同じ長さなら置換と同じ）
        ones_a = np.ones((1, L), dtype=np.int64)
        msk_a = ones_a.copy()
        msk_a[0, a0:a1] = 0
        ones_i = np.ones((1, inj.shape[1]), dtype=np.int64)
        msk_i = ones_i.copy()
        msk_i[0, a0:a0 + (b1 - b0)] = 0
        Lb = sc["per_pair"][p]["b"]["n_encoder_tokens"]
        seq = [("bl", [(q["norm_text_a"], q["word_a"])], src, ones_a, q["text_a"]),
               ("inj", [(q["norm_text_a"], q["word_a"]), (q["norm_text_b"], q["word_b"])], inj, ones_i, q["text_a"]),
               ("bl_mask", [(q["norm_text_a"], q["word_a"])], src, msk_a, q["text_a"]),
               ("inj_mask", [(q["norm_text_a"], q["word_a"]), (q["norm_text_b"], q["word_b"])], inj, msk_i,
                q["text_a"]),
               ("tgt", [], rng.randn(1, Lb, 4).astype(np.float32), np.ones((1, Lb), dtype=np.int64), q["text_b"])]
        for lab, stexts, mem, attn, text in seq:
            for t_, w_ in stexts:
                span(None, t_, w_, add_bos=0)
            if lab == "tgt" and silent_tgt:
                fake_generate.silent_next = True
            try:
                gen(None, None, None, mem, attn, None, text + ("RAISE" if lab == raise_at else ""), "cpu", sr, 50)
            except RuntimeError:
                return False
        return True

    assert run_pair(rep_) is True
    assert run_pair(spl[0], raise_at="inj") is False
    assert run_pair(spl[1], silent_tgt=True) is True
    ctx.f.close()
    ev = C.read_events(C.P(run_rel))
    gens = [e for e in ev if e.get("ev") == "gen"]
    got = [(e["pair_idx"], e["label"]) for e in gens]
    want = [(rep_, "bl"), (rep_, "inj"), (rep_, "bl_mask"), (rep_, "inj_mask"), (rep_, "tgt"),
            (spl[0], "bl"), (spl[0], "unknown00007"),
            (spl[1], "bl"), (spl[1], "inj"), (spl[1], "bl_mask"), (spl[1], "inj_mask"), (spl[1], "tgt")]
    assert got == want, got
    assert gens[6]["exception"] and not gens[6]["signature_ok"]
    mcs = {g["pair_idx"]: g["mem_check"] for g in gens if g["label"] == "inj_mask"}
    assert mcs[rep_]["ok"] and mcs[rep_]["mode"] == "replace", mcs[rep_]
    assert mcs[spl[1]]["ok"] and mcs[spl[1]]["mode"] == "splice", mcs[spl[1]]
    assert all(len(g["durations"]) == 1 for g in gens), [g["durations"] for g in gens]
    assert gens[-1]["silence"]["silent"] is True and gens[0]["silence"]["silent"] is False
    # 状態の判定（CPU 側と同じ関数）
    row_ok = {"aligned": True, **{f"{c}_per": 0.5 for c in C.PER_CONDS}}
    al = [{"ev": "align_tr", "label": lab} for lab in C.ALIGN_ORDER]

    def cls(p, row, errs=(), complete=True, ref=None):
        return C.classify_pair(row, [g for g in gens if g["pair_idx"] == p], al if row else [], list(errs), complete,
                               ROOT, ref or span_ref[p])[0]
    bad_ref = {"a": span_ref[rep_]["a"], "b": [span_ref[rep_]["b"][0], span_ref[rep_]["b"][1] + 1]}
    res = {"replace_pair": cls(rep_, dict(row_ok, pair_idx=rep_)),
           "splice_pair_silent_tgt": cls(spl[1], dict(row_ok, pair_idx=spl[1])),
           "exception": cls(spl[0], None), "incomplete_batch": cls(spl[0], None, complete=False),
           "span_mismatch": cls(rep_, dict(row_ok, pair_idx=rep_), ref=bad_ref),
           "cond_without_entity": cls(rep_, dict(row_ok, pair_idx=rep_, inj_per=None)),
           "whisper_exception": cls(rep_, dict(row_ok, pair_idx=rep_), errs=[{"kind": "whisper", "line": "x"}])}
    want_res = {"replace_pair": "valid", "splice_pair_silent_tgt": "silent", "exception": "run_error",
                "incomplete_batch": "pending", "span_mismatch": C.STATUS_STOP, "cond_without_entity": "unaligned",
                "whisper_exception": "alignment_exception"}
    assert res == want_res, res
    # 記憶の確かめが壊れたときに capture_error になること（INJ_mask の記憶が INJ と違う）
    broken = [dict(g) for g in gens if g["pair_idx"] == rep_]
    for g in broken:
        if g["label"] == "inj_mask":
            g["mem_check"] = dict(g["mem_check"], ok=False)
    assert C.classify_pair(dict(row_ok, pair_idx=rep_), broken, al, [], True, ROOT, span_ref[rep_])[0] == "capture_error"
    # 対の読み替え（本番の JA の分岐の文を、gen と同じ読み替え関数で実行）
    ctx2 = Ctx(pairs_n, targets, run_rel + "_redirect", span_ref)
    red = make_open_redirect(ctx2, C.P(C.PAIR_FILE_REL))
    loaded = C.run_production_ja_loader(red)
    ctx2.f.close()
    assert [tuple(p[k] for k in C.PAIR_KEYS) for p in loaded] == [tuple(p[k] for k in C.PAIR_KEYS) for p in pairs]
    assert ctx2.n_redirect == 1
    pt = PrintTap(ctx2)
    ctx2.f = open(C.P(f"{run_rel}_redirect/events.jsonl"), "a", encoding="utf-8")
    pt("[Info] Loaded 300 ja pairs")  # 読み替え 1 回・300 対 → 通る
    ctx2.n_redirect = 0
    try:
        pt("[Info] Loaded 300 ja pairs")
        raise AssertionError("読み替えなしで通ってしまった")
    except RuntimeError:
        pass
    ctx2.f.close()
    # 表の書式: 今の表の JA 8 セルを、今の results.json（5/11）から一字一句作り直せること
    old = C.read_json(C.P(f"{C.OLD_DIR_REL}/results.json"))
    cells = C.table_cells(old["rows"])
    paper = {k: C.strip_bold(v) for k, v in C.PAPER_JA_CELLS.items()}
    assert cells == paper, (cells, paper)
    via4 = C.table_cells_via4(old["rows"])
    bold = C.bold_ok(old["rows"])
    assert bold["inj_highest_match_strict"] and bold["inj_lowest_per_strict"], bold
    guard = selftest_pyojt_guard()
    print(f"[appc_ja] SELFTEST OK: {res}")
    print(f"[appc_ja]   events: {got}")
    print(f"[appc_ja]   記憶の確かめ: 置換 p{rep_} {mcs[rep_]['mode']} ok / 継ぎ合わせ p{spl[1]} la={mcs[spl[1]]['la']} "
          f"lb={mcs[spl[1]]['lb']} ok")
    print(f"[appc_ja]   対の読み替え: 本番の JA の分岐で 300 対、付録 K と同じ並び（読み替え 1 回）。読み替えなしなら止まる")
    print(f"[appc_ja]   今の表の JA 8 セルを 5/11 の results.json から再現: True（4 桁経由だと INJ の PER は "
          f"{via4['per_inj']} になり表の {paper['per_inj']} と違う）；太字の根拠 {bold['inj_highest_match_strict']}/"
          f"{bold['inj_lowest_per_strict']}")
    print(f"[appc_ja]   pyopenjtalk の関所: {guard}")
    print(f"[appc_ja]   wrote {run_rel}")
    return 0


# ---------------------------------------------------------------- 入り口
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, choices=C.SEEDS, default=None)
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--retry", type=int, default=None, help="同じ seed のやり直しの番号（retryNN）")
    ap.add_argument("--pairs", default=None, help="--retry の対（カンマ区切りの pair_idx）")
    ap.add_argument("--status_file", default=None, help="appc_ja_scan.py の出力（seed 43/44 と --retry）")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force_resume", action="store_true",
                    help="前の起動が止まっていることを squeue で確かめた後、20 分待たずに再開する")
    args = ap.parse_args()

    errs = C.pinned_errors(ROOT) + C.old_result_errors(ROOT)
    if errs:
        raise SystemExit("[appc_ja] " + " / ".join(errs))
    pairs = C.load_pairs(ROOT)
    span_rel, sc = latest_span_check(pairs)
    if args.selftest:
        return selftest(pairs, span_rel, sc)
    if args.seed is None:
        raise SystemExit("[appc_ja] --seed が要る")
    static = static_checks()
    if args.dry_run:
        return dry_run(args, pairs, span_rel, static)
    pairs_n, span_ref = pairs_with_norm(pairs, sc)
    return run_gpu(args, pairs_n, span_rel, span_ref)


if __name__ == "__main__":
    sys.exit(main())
