#!/usr/bin/env python3
"""付録 C（tab:ja_crossattn_phoneme）の英語 300 対を、直したスパン関数で作り直す GPU の入り口（2026-10-01）.

本番のスクリプト scripts/projection/eval_crossattn_entity_per.py を **改変せずに** main() を呼ぶ
（md5 を固定して確かめる）。引数は元の実行と同じ `--lang en --n 300` に `--output_dir`（新しい置き場）だけを足す。
元と違うのは、main() の中で import される scripts/encoder_injection_t5gemma.py の find_entity_token_range が
7/11 に直した版であること（ファイルはそのまま。差し替えていない）だけ。

このラッパーがすること（本番の処理には手を入れない。出力を写し取るだけ）:
  * 1 回の起動で 1 つの batch（seed 42 は 50 対ずつの part00..part05）だけを扱う。対象以外の対を「済み」
    （placeholder）とした checkpoint.json を新しい置き場に先に置くので、本番の resume の仕組みで対象の対だけが
    生成・採点される（G3 と同じ型）。時間切れで止まっても、同じコマンドをもう一度流せば checkpoint から続きを
    やる（本番の checkpoint は 25 対ごとに書かれるので、最大 24 対はやり直しになる）。
  * モジュール変数 SEED を --seed に置き換える（本番の seed_all / seed_everything が参照する）。
  * encoder_injection_t5gemma.generate_with_modified_encoder と find_entity_token_range を、呼び出しをそのまま通して
    記録だけを取る薄い包みに差し替える。5 本（BL, INJ, BL_mask, INJ_mask, TGT）の wav をその場で float32 で保存し、
    直前に使われたスパンの範囲と遮断した列を記録する（生成のたびに events.jsonl に追記。止まっても残る）。
  * forced_alignment_utils.WhisperAligner.transcribe_words と、本番 module の find_entity_chunks /
    slice_entity_audio も同じく記録だけを取る（Whisper の単語列、実体語が見つかった位置、切り出しの長さ）。
  * 本番の print を「そのまま出し、握りつぶされた例外の行だけを記録する」包みに替える（G3 と同じ）。
  * 無音は scripts/regen2/silence_criterion.py（regen2 共通の基準）で、保存した wav から判定する（記録のみ。
    使えるかどうかの判定は CPU 側の appc_en_scan.py が行う）。

seed の決まり（著者の決定 2026-10-01）:
  * seed 42: 300 対すべて（--part 0..5）。
  * seed 43: seed 42 で「対として使えない」（unaligned / silent / generation_failed）対だけ。
    appc_en_scan.py の状態ファイル（--status_file）から対象を取る。seed 42 の 6 part が全部終わってから。
  * seed 44: seed 43 でも使えない対だけ。45 以上は流さない（著者の判断）。
  * 環境の問題（run_error / alignment_exception / capture_error / no_row_unexplained）は対の良し悪しではないので、
    直してから同じ seed で --retry N --pairs ... として流す。同じ種類が同じ seed で 2 回出た対、retry が 2 回でも
    続く対は appc_en_scan.py が stop_author_decision にし、--retry は受け付けない（resolve_batch の確かめ）。
  * 選ぶ基準は有効性だけ。PER の値は見ない。

環境変数（コマンドに env を付けない。このスクリプトが先頭で、numpy・torch・HF を読む前に自分で入れる）:
  スレッド上限 13（OMP/MKL/OPENBLAS/NUMEXPR/VECLIB、torch.set_num_threads(13)）、
  HF_HOME=<根>/.cache/huggingface, HF_HUB_CACHE=<根>/.cache/huggingface/hub, HF_HUB_OFFLINE=1,
  TORCH_HOME=<根>/.cache/torch, XDG_CACHE_HOME=<根>/.cache, NUMBA_CACHE_DIR=<根>/.cache/numba
  （<根> はコンテナの中なら /mnt）。ホストでは CUDA_VISIBLE_DEVICES='' にする。
  pyopenjtalk は import できないように塞ぐ（英語では使われない。vendor の長さ推定は try/except で None にする）。
  HF_HUB_OFFLINE では T5Gemma の tokenizer を repo id で読むと落ちるので、9/28 の前例
  （scripts/nontemplate/gen_t5gemma_pairs_module.py）の読み替えを使う（G3 と同じ）。

GPU（プロジェクト直下から。1 ジョブ = 1 batch = GPU 1 枚。投入の前に squeue で利用者全体で 2 枚以内を確かめる）:
  SLURM_TIME=02:00:00 slurm_run.sh python3 -u /mnt/scripts/spanfix/appc_en_gen.py \
      --seed 42 --part 0 >> eval_results/_spanfix_appC_en/log_seed42_part00.log 2>&1
  （--part 1..5 も同じ。時間切れのときは同じコマンドをもう一度流すと続きからやる）
  seed 43 / 44（appc_en_scan.py が示したときだけ。part の数と SLURM_TIME も scan が示す）:
  SLURM_TIME=<scan の表示> slurm_run.sh python3 -u /mnt/scripts/spanfix/appc_en_gen.py \
      --seed 43 --part 0 --status_file /mnt/eval_results/_spanfix_appC_en/status/status_<日時>.json \
      >> eval_results/_spanfix_appC_en/log_seed43_part00.log 2>&1
CPU（ホストで直接。モデルは読まない）:
  taskset -c 0-12 python3 scripts/spanfix/appc_en_gen.py --seed 42 --part 0 --dry_run
  taskset -c 0-12 python3 scripts/spanfix/appc_en_gen.py --selftest
"""
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
sys.modules["pyopenjtalk"] = None  # 英語では使わない。間接の import も塞ぐ

import argparse  # noqa: E402
import ast  # noqa: E402
import builtins  # noqa: E402
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
import appc_en_common as C  # noqa: E402
import silence_criterion as SC  # noqa: E402

ROOT = C.PROJECT_ROOT
assert _R == ROOT, (_R, ROOT)
OWN_REL = "scripts/spanfix/appc_en_gen.py"
OWN_CODE = (OWN_REL, "scripts/spanfix/appc_en_common.py")
PAIR_ERR_RE = re.compile(r"^\s*\[(\d+)\] err: ")              # 本番 main() の対の処理中の例外
WHISPER_ERR_RE = re.compile(r"^\s*\[Whisper\] transcribe failed: ")  # forced_alignment_utils の例外
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
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    main = mains[0]
    lazy = [n for n in ast.walk(main) if isinstance(n, ast.ImportFrom) and n.module == "encoder_injection_t5gemma"]
    lazy_names = sorted(a.name for n in lazy for a in n.names)
    top_inj = any(isinstance(n, (ast.ImportFrom, ast.Import)) and (
        getattr(n, "module", None) == "encoder_injection_t5gemma"
        or any(a.name == "encoder_injection_t5gemma" for a in getattr(n, "names", [])))
        for n in tree.body)
    fa_top = any(isinstance(n, ast.ImportFrom) and n.module == "forced_alignment_utils"
                 and sorted(a.name for a in n.names) == ["WhisperAligner", "find_entity_chunks", "slice_entity_audio"]
                 for n in tree.body)
    print_shadowed = any(isinstance(n, (ast.FunctionDef, ast.Assign)) and (
        getattr(n, "name", None) == "print" or any(getattr(t, "id", None) == "print" for t in getattr(n, "targets", [])))
        for n in tree.body)
    res = {
        "eval_md5": C.md5_file(C.P(C.EVAL_REL)),
        "module_SEED_assign": seed_assign,
        "main_lazy_imports": lazy_names,
        "no_top_level_import_of_injection": not top_inj,
        "fa_imported_at_top": fa_top,
        "module_print_not_shadowed": not print_shadowed,
        "pairs_source": "from en_heteronyms_500 import build_500_pairs" in src
                        and "pairs = build_500_pairs()[: args.n]" in src,
        "resume_skips_done": "if pair_idx in done: continue" in src,
        "pair_err_print": 'print(f"  [{i}] err: {str(e)[:200]}"); continue' in src,
        "transcribe_call": "aligner.transcribe_words(audio, sr_in, language=lang)" in src,
        "find_call": "find_entity_chunks(chunks, entity, lang, input_text=input_text)" in src,
        "slice_call": "slice_entity_audio(audio_np, aligner.target_sr, chunks, s_idx, e_idx)" in src,
        "span_calls": src.count("find_entity_token_range(t5_tok, "),
        "tokenizer_by_repo_id": "t5_tok = AutoTokenizer.from_pretrained(tok_name)" in src,
        "ckpt_every_25": "if (i + 1) % 25 == 0:" in src,
    }
    fa_src = open(C.P(C.FA_REL), encoding="utf-8").read()
    fa_print_lines = [ln.strip() for ln in fa_src.splitlines() if "print(" in ln]
    res["fa_only_print_is_whisper_fail"] = fa_print_lines == [FA_WHISPER_FAIL_PRINT]
    gen_tree = ast.parse(open(C.P(C.INJ_REL), encoding="utf-8").read())
    gen = [n for n in gen_tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_with_modified_encoder"]
    res["generate_params"] = [a.arg for a in gen[0].args.args] if gen else []
    want_lazy = ["find_entity_token_range", "generate_with_modified_encoder", "get_encoder_output", "seed_everything"]
    ok = (seed_assign and lazy_names == want_lazy and res["no_top_level_import_of_injection"] and fa_top
          and res["module_print_not_shadowed"] and res["pairs_source"] and res["resume_skips_done"]
          and res["pair_err_print"] and res["transcribe_call"] and res["find_call"] and res["slice_call"]
          and res["span_calls"] == 3 and res["fa_only_print_is_whisper_fail"]
          and res["generate_params"] == ["model", "text_tokenizer", "audio_tokenizer", "modified_memory",
                                         "encoder_attention_mask", "encoder_position_ids", "target_text",
                                         "device", "codec_audio_sr", "codec_sr"])
    res["ok"] = bool(ok)
    if not ok:
        raise SystemExit(f"[appc] 本番スクリプトの形が想定と違う: {json.dumps(res, ensure_ascii=False)}")
    return res


def latest_span_check(pairs):
    files = sorted(f for f in os.listdir(C.P(C.SPAN_CHECK_DIR_REL)) if f.startswith("span_check_")) \
        if os.path.isdir(C.P(C.SPAN_CHECK_DIR_REL)) else []
    if not files:
        raise SystemExit("[appc] CPU のスパンの確かめ（appc_en_span_check.py）の結果が無い")
    rel = f"{C.SPAN_CHECK_DIR_REL}/{files[-1]}"
    sc = C.read_json(C.P(rel))
    errs = []
    if not sc.get("all_ok"):
        errs.append("all_ok でない")
    if sc.get("pairs_digest") != C.pairs_digest(pairs):
        errs.append("対の digest が違う")
    for k in (C.EVAL_REL, C.INJ_REL, C.PAIRS_REL):
        if sc.get("code_md5", {}).get(k) != C.PINNED_MD5[k]:
            errs.append(f"{k} の md5 が確かめの時点と違う")
    if errs:
        raise SystemExit(f"[appc] {rel}: " + " / ".join(errs))
    return rel, sc


def load_status(path, check_seeds):
    if not path:
        raise SystemExit("[appc] --status_file（appc_en_scan.py の出力）が要る")
    p = path if os.path.isabs(path) else C.P(path)
    st = C.read_json(p)
    for s in check_seeds:
        if st.get("fingerprint", {}).get(str(s)) != C.seed_fingerprint(s):
            raise SystemExit(f"[appc] 状態ファイルが古い（seed {s} の記録が集計の後に変わった）。"
                             "appc_en_scan.py を流し直す")
    return os.path.relpath(p, ROOT), st


def resolve_batch(args, pairs):
    """(batch 名, 対象の対, 種類, 根拠) を返す。既にある batch は manifest の対象と一致を確かめる."""
    if args.retry is not None:
        if not args.pairs:
            raise SystemExit("[appc] --retry には --pairs が要る")
        targets = sorted({int(x) for x in args.pairs.split(",") if x.strip()})
        name = f"retry{args.retry:02d}"
        if os.path.exists(C.P(f"{C.batch_dir_rel(args.seed, name)}/manifest.json")):
            return name, targets, "retry", {"resume": True}
        st_rel, st = load_status(args.status_file, [s for s in C.SEEDS if s <= args.seed])
        fin = st.get("final", {}).get(str(args.seed), {})
        bad = [(p, fin.get(str(p), {}).get("status")) for p in targets
               if fin.get(str(p), {}).get("status") not in C.RETRY_SAME_SEED]
        if bad:
            raise SystemExit(f"[appc] 同じ seed のやり直しは {C.RETRY_SAME_SEED} の対だけ: {bad}")
        return name, targets, "retry", {"status_file": st_rel}
    if args.part is None:
        raise SystemExit("[appc] --part が要る")
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
        raise SystemExit(f"[appc] 状態ファイルに seed {args.seed} の対象が無い（前の seed が終わっていない）")
    k = args.part - base
    if k < 0:
        raise SystemExit(f"[appc] part {args.part} は状態ファイルを作った時点で既にあった番号（{base} 以上を使う）")
    # 状態ファイルの後に、この seed で別の状態ファイルから batch が作られていないか（対象が重ならないように）
    then = set(st.get("batches", {}).get(str(args.seed), {}))
    for n, man in C.list_batches(args.seed):
        if n in then:
            continue
        if not (n.startswith("part") and int(n[4:]) >= base and man.get("basis", {}).get("status_file") == st_rel):
            raise SystemExit(f"[appc] 状態ファイルが古い（seed {args.seed} に {n} が後から作られた）。scan を流し直す")
    chunk = allt[k * C.PART_SIZE:(k + 1) * C.PART_SIZE]
    if not chunk:
        raise SystemExit(f"[appc] seed {args.seed} の part {args.part} は空（残り {len(allt)} 対, 先頭の番号 {base}）")
    return name, chunk, "part", {"status_file": st_rel, "n_remaining_at_status": len(allt), "part_base": base}


def batch_state(batch_abs, targets):
    runs = C.list_runs(batch_abs)
    complete = False
    for r in runs:
        for e in C.read_events(r):
            if e.get("ev") == "main_end" and e.get("error") is None:
                complete = True
    ck = os.path.join(batch_abs, "checkpoint.json")
    try:
        rows = C.read_json(ck).get("rows", [])
    except Exception as ex:  # noqa: BLE001  壊れた checkpoint は消さず、著者の判断にする
        raise SystemExit(f"[appc] {ck} が読めない（{type(ex).__name__}）。消さずに著者の判断を仰ぐ")
    done = {r["pair_idx"] for r in rows}
    real = {r["pair_idx"] for r in rows if not r.get("placeholder")}
    return {"runs": [os.path.basename(r) for r in runs], "complete": complete,
            "rows_real": sorted(real & set(targets)),
            "would_run": [t for t in targets if t not in done]}


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
        if errs:
            raise SystemExit(f"[appc] {rel} の manifest と合わない: {errs}")
        return rel, man, False
    if not create:
        return rel, None, True
    os.makedirs(absd, exist_ok=False)
    man = {
        "item": "Appendix C tab:ja_crossattn_phoneme, EN rows, span-fix rerun",
        "seed": args.seed, "batch": name, "kind": kind, "targets": targets, "basis": basis,
        "created": C.stamp(), "created_order": int(time.time() * 1000),
        "pairs": "data/en_heteronyms_500.py build_500_pairs()[:300]", "pairs_digest": C.pairs_digest(pairs),
        "production_script": C.EVAL_REL, "prod_args": C.PROD_ARGS, "output_dir_rel": rel,
        "span_check": span_rel, "pinned_md5": C.PINNED_MD5,
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

    def __init__(self, pairs, targets, run_rel):
        self.pairs = {p["pair_idx"]: p for p in pairs}
        self.by_text_a = {p["text_a"]: p["pair_idx"] for p in pairs}
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
        self.device_map = None
        self.device_offload = None
        self.n_gen = 0

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
        if full_text == pair["text_a"] and entity == pair["word_a"]:
            return "a"
        if full_text == pair["text_b"] and entity == pair["word_b"]:
            return "b"
        return "other"

    # -- スパン
    def on_span(self, full_text, entity, add_bos, add_eos, result):
        self.pending_spans.append({"full_text": full_text, "entity": entity, "add_bos": add_bos,
                                   "add_eos": add_eos, "result": list(result) if result else None})

    # -- 生成
    def mem_check(self):
        import numpy as np

        def maxdiff(x, y):
            if x is None or y is None or x.shape != y.shape:
                return None
            return float(np.abs(x - y).max())
        d1 = maxdiff(self.mems.get("inj"), self.mems.get("bl"))
        d2 = maxdiff(self.mems.get("bl_mask"), self.mems.get("bl"))
        d3 = maxdiff(self.mems.get("inj_mask"), self.mems.get("inj"))
        if d1 is None:
            ok = None  # 長さが違う（継ぎ合わせ）。この 300 対は全部同じ長さのはず（スパンの確かめ）
        else:
            ok = (d2 is not None and d3 is not None and d1 > 0 and d2 <= 0.01 * d1 and d3 <= 0.01 * d1)
        return {"maxdiff_inj_vs_bl": d1, "maxdiff_blmask_vs_bl": d2, "maxdiff_injmask_vs_inj": d3, "ok": ok}

    def on_gen(self, a, out, exception, elapsed):
        import numpy as np
        import soundfile as sf
        spans, self.pending_spans = self.pending_spans, []
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
        pidx = self.by_text_a.get(spans[0]["full_text"]) if spans else self.cur
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
               "mem_check": mem_check, "target_text": text}
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
    """記録の失敗で本番の処理を止めない（記録の失敗は tap の err として残し、CPU 側で capture_error にする）."""
    try:
        fn(*a)
    except Exception as ex:  # noqa: BLE001
        builtins.print(f"[appc] tap failed: {type(ex).__name__}: {ex}", flush=True)
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


class PrintTap:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, *args, **kwargs):
        builtins.print(*args, **kwargs)
        try:
            self.ctx.on_print_line(" ".join(str(x) for x in args))
        except Exception:  # noqa: BLE001
            pass


def install_taps(M, FA, INJ, ctx):
    INJ.generate_with_modified_encoder = make_gen_tap(INJ.generate_with_modified_encoder, ctx)
    INJ.find_entity_token_range = make_span_tap(INJ.find_entity_token_range, ctx)

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

    tap = PrintTap(ctx)
    M.print = tap
    FA.print = tap
    return {"generate_tapped": getattr(INJ.generate_with_modified_encoder, "__wrapped__", None) is not None,
            "span_tapped": getattr(INJ.find_entity_token_range, "__wrapped__", None) is not None,
            "print_tapped": M.print is tap and FA.print is tap}


# ---------------------------------------------------------------- GPU
def setup_offline(W):
    info = {"hf_offline": bool(W.hf_offline()), "offline_tokenizer_shim": []}
    if not info["hf_offline"]:
        raise SystemExit("[appc] HF_HUB_OFFLINE=1 で流す")
    ok, pre = W.offline_preflight()
    info["offline_preflight_ok"] = bool(ok)
    info["offline_preflight"] = pre
    if not ok:
        raise SystemExit(f"[appc] 生成の経路で読む repo が手元のキャッシュに無い: {pre}")
    W.install_offline_tokenizer_shim(info["offline_tokenizer_shim"])
    return info


def duration_preflight(pairs):
    """vendor の長さ推定が英語で g2p_en を使えることを確かめる（使えないと文字数で推定され、元の実行と変わる）."""
    DE = sys.modules.get("duration_estimator")
    info = {"module_file": getattr(DE, "__file__", None), "G2p_available": getattr(DE, "G2p", None) is not None,
            "pyopenjtalk_in_module": getattr(DE, "pyopenjtalk", "absent")}
    if DE is None or DE.G2p is None:
        raise SystemExit(f"[appc] duration_estimator の G2p が使えない: {info}")
    t = pairs[0]["text_a"]
    try:
        info["sample"] = {"text": t, "estimate_sec": DE.estimate_duration(target_text=t, target_lang="en"),
                          "phonemes": DE._phoneme_count_en(t)}
    except Exception as ex:  # noqa: BLE001  nltk のデータが取れない等。モデルを読む前に止まる
        raise SystemExit(f"[appc] 英語の長さ推定（g2p_en）が動かない: {type(ex).__name__}: {ex}")
    if DE.pyopenjtalk is not None:
        raise SystemExit("[appc] pyopenjtalk が読み込まれている（塞いだはず）")
    return info


def run_gpu(args, pairs, span_rel):
    if not IN_CONTAINER:
        raise SystemExit("[appc] GPU の実行は slurm_run.sh の中で（ホストでは --dry_run / --selftest）")
    name, targets, kind, basis = resolve_batch(args, pairs)
    batch_rel, man, created = prepare_batch(args, name, targets, kind, basis, pairs, span_rel, create=True)
    batch_abs = C.P(batch_rel)
    st = batch_state(batch_abs, targets)
    print(f"[appc] seed={args.seed} batch={name} targets={len(targets)} created={created} "
          f"runs={st['runs']} complete={st['complete']} would_run={len(st['would_run'])}", flush=True)
    if st["complete"]:
        print("[appc] この batch は終わっている（main() が例外なく終わった起動がある）。何もしない", flush=True)
        return 0
    if not st["would_run"]:
        print("[appc] 対象の行は全部 checkpoint にある。何もしない", flush=True)
        return 0
    # 同じ batch を 2 本同時に流さない: 前の起動が end.json なしで、記録が 20 分以内に更新されていれば止まる
    if st["runs"] and not args.force_resume:
        last = os.path.join(batch_abs, "runs", st["runs"][-1])
        if not os.path.exists(os.path.join(last, "end.json")):
            ts = [os.path.getmtime(os.path.join(last, f)) for f in ("events.jsonl", "launch.json")
                  if os.path.exists(os.path.join(last, f))] or [os.path.getmtime(last)]
            age_min = (time.time() - max(ts)) / 60
            if age_min < 20:
                raise SystemExit(f"[appc] 前の起動 {st['runs'][-1]} の記録が {age_min:.1f} 分前に更新されている"
                                 "（まだ動いている恐れ）。squeue で止まっていることを確かめ、20 分待つか --force_resume")
    r = len(st["runs"]) + 1
    run_rel = f"{batch_rel}/runs/run{r:02d}"
    os.makedirs(C.P(run_rel), exist_ok=False)
    with open(C.P(f"{batch_rel}/checkpoint.json"), "rb") as fi, open(C.P(f"{run_rel}/checkpoint_at_launch.json"), "xb") as fo:
        fo.write(fi.read())

    import torch
    torch.set_num_threads(THREADS)
    if not torch.cuda.is_available():
        raise SystemExit("[appc] GPU が見えない")
    W = load_module_from_file("_appc_offline_shim", C.P(C.SHIM_REL))
    off = setup_offline(W)
    M = load_module_from_file("eval_crossattn_entity_per", C.P(C.EVAL_REL))
    FA = sys.modules["forced_alignment_utils"]
    if os.path.realpath(FA.__file__) != os.path.realpath(C.P(C.FA_REL)):
        raise SystemExit(f"[appc] forced_alignment_utils が本番のファイルでない: {FA.__file__}")
    import encoder_injection_t5gemma as INJ
    if os.path.realpath(INJ.__file__) != os.path.realpath(C.P(C.INJ_REL)):
        raise SystemExit(f"[appc] encoder_injection_t5gemma が本番のファイルでない: {INJ.__file__}")
    dur = duration_preflight(pairs)
    ctx = Ctx(pairs, targets, run_rel)
    taps = install_taps(M, FA, INJ, ctx)
    vend = {n: {"file": getattr(sys.modules.get(n), "__file__", None),
                "md5": C.md5_file(sys.modules[n].__file__) if getattr(sys.modules.get(n), "__file__", None) else None}
            for n in ("data.tokenizer", "duration_estimator", "inference_tts_utils")}
    import transformers
    launch = {
        "item": man["item"], "seed": args.seed, "batch": name, "run": r, "targets": targets,
        "would_run": st["would_run"], "rows_already": st["rows_real"],
        "argv_wrapper": sys.argv, "prod_argv": C.PROD_ARGS + ["--output_dir", batch_abs],
        "span_check": span_rel, "code_md5": C.code_md5(ROOT, OWN_CODE), "vendor_modules": vend,
        "taps": taps, "offline": off, "duration_preflight": dur,
        "env": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "cuda_device": torch.cuda.get_device_name(0),
                "torch_num_threads": torch.get_num_threads(), "env_at_launch": ENV_AT_LAUNCH,
                "env_set_by_script": CACHE_ENV,
                "now": {k: os.environ.get(k) for k in THREAD_VARS + CACHE_VARS + ("CUDA_VISIBLE_DEVICES",
                                                                                 "NVIDIA_VISIBLE_DEVICES")},
                "pyopenjtalk_blocked": sys.modules.get("pyopenjtalk", "absent") is None},
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
        builtins.print(f"[appc] main() ended with {err}", flush=True)
        traceback.print_exc()
    finally:
        sys.argv = saved
    elapsed = round((time.time() - t0) / 60, 2)
    ctx.emit({"ev": "main_end", "error": err, "elapsed_min": elapsed})
    st2 = batch_state(batch_abs, targets)
    end = {"error": err, "elapsed_min": elapsed, "n_gen_calls": ctx.n_gen, "rows_real": len(st2["rows_real"]),
           "would_run_after": st2["would_run"], "date": C.stamp()}
    C.write_json_new(end, C.P(f"{run_rel}/end.json"))
    builtins.print(f"[appc] END seed={args.seed} batch={name} run={r} error={err} elapsed={elapsed}min "
                   f"gen_calls={ctx.n_gen} rows={len(st2['rows_real'])}/{len(targets)}", flush=True)
    builtins.print("[appc] NEXT: ホストで taskset -c 0-12 python3 scripts/spanfix/appc_en_scan.py", flush=True)
    return 0 if err is None else 1


# ---------------------------------------------------------------- CPU の試運転
def dry_run(args, pairs, span_rel, static):
    name, targets, kind, basis = resolve_batch(args, pairs)
    batch_rel, man, would_create = prepare_batch(args, name, targets, kind, basis, pairs, span_rel, create=False)
    st = batch_state(C.P(batch_rel), targets) if not would_create else {"would_run": targets, "complete": False}
    W = load_module_from_file("_appc_offline_shim", C.P(C.SHIM_REL))
    ok, pre = W.offline_preflight()
    SC.selftest()
    rep = {"mode": "dry_run", "seed": args.seed, "batch": name, "kind": kind, "basis": basis,
           "n_targets": len(targets), "targets_head": targets[:10], "would_create_batch": would_create,
           "batch_state": st, "static_checks": static, "span_check": span_rel,
           "offline_preflight_ok": ok, "offline_preflight": pre, "pinned_md5_ok": not C.pinned_errors(ROOT),
           "suggested_slurm_time": C.suggest_slurm_time(len(st["would_run"])) if st["would_run"] else None,
           "code_md5": C.code_md5(ROOT, OWN_CODE), "date": C.stamp()}
    path = C.P(f"{C.DRYRUN_DIR_REL}/dryrun_{C.stamp()}_seed{args.seed}_{name}.json")
    C.write_json_new(rep, path)
    print(f"[appc] DRY RUN seed={args.seed} batch={name} targets={len(targets)} (head {targets[:5]}) "
          f"create={would_create} would_run={len(st['would_run'])} offline_ok={ok} "
          f"SLURM_TIME={rep['suggested_slurm_time']}")
    print(f"[appc] wrote {path}")
    return 0 if ok else 1


def selftest(pairs):
    """偽の生成関数で、対の特定・ラベル付け・wav の保存・無音の判定・状態の判定が意図どおりかを確かめる."""
    import numpy as np
    run_rel = f"{C.DRYRUN_DIR_REL}/selftest_{C.stamp()}"
    targets = [0, 1, 2]
    ctx = Ctx(pairs, targets, run_rel)
    sr = 16000

    def fake_generate(model, text_tokenizer, audio_tokenizer, modified_memory, encoder_attention_mask,
                      encoder_position_ids, target_text, device, codec_audio_sr, codec_sr):
        if target_text.endswith("RAISE"):
            raise RuntimeError("fake OOM")
        n = int(0.6 * codec_audio_sr)
        amp = 0.0 if getattr(fake_generate, "silent_next", False) else 0.3
        fake_generate.silent_next = False
        t = np.arange(n, dtype=np.float32) / codec_audio_sr
        return (amp * np.sin(2 * np.pi * 220 * t)).reshape(1, 1, -1)

    def fake_span(text_tokenizer, full_text, entity_text, add_bos=0, add_eos=0):
        return (1, 2)

    gen = make_gen_tap(fake_generate, ctx)
    span = make_span_tap(fake_span, ctx)
    rng = np.random.RandomState(0)

    def run_pair(p, raise_at=None, silent_tgt=False):
        src = rng.randn(1, 6, 4).astype(np.float32)
        inj = src.copy()
        inj[0, 1:2] = rng.randn(1, 4)
        ones = np.ones((1, 6), dtype=np.int64)
        msk = ones.copy()
        msk[0, 1:2] = 0
        seq = [("bl", [p["text_a"]], src, ones, p["text_a"]),
               ("inj", [p["text_a"], p["text_b"]], inj, ones, p["text_a"]),
               ("bl_mask", [p["text_a"]], src, msk, p["text_a"]),
               ("inj_mask", [p["text_a"], p["text_b"]], inj, msk, p["text_a"]),
               ("tgt", [], rng.randn(1, 7, 4).astype(np.float32), np.ones((1, 7), dtype=np.int64), p["text_b"])]
        for lab, stexts, mem, attn, text in seq:
            for st_ in stexts:
                span(None, st_, p["word_a"], add_bos=0)
            if lab == "tgt" and silent_tgt:
                fake_generate.silent_next = True
            try:
                gen(None, None, None, mem, attn, None, text + ("RAISE" if lab == raise_at else ""), "cpu", sr, 50)
            except RuntimeError:
                return False
        return True

    P = {p["pair_idx"]: p for p in pairs}
    assert run_pair(P[0]) is True
    assert run_pair(P[1], raise_at="inj") is False   # 対 1 は INJ の例外で中断（本番は continue）
    assert run_pair(P[2], silent_tgt=True) is True   # 対 2 は TGT が無音
    ctx.f.close()
    ev = C.read_events(C.P(run_rel))
    gens = [e for e in ev if e.get("ev") == "gen"]
    got = [(e["pair_idx"], e["label"], e["signature_ok"]) for e in gens]
    want = [(0, "bl", True), (0, "inj", True), (0, "bl_mask", True), (0, "inj_mask", True), (0, "tgt", True),
            (1, "bl", True), (1, "unknown00007", False),
            (2, "bl", True), (2, "inj", True), (2, "bl_mask", True), (2, "inj_mask", True), (2, "tgt", True)]
    # 対 1 の INJ は target_text が "...RAISE" なので形が合わず unknown になる（例外は記録される）
    assert [g[:2] for g in got] == [w[:2] for w in want], got
    assert gens[6]["exception"] and not gens[6]["signature_ok"]
    assert all(g["mem_check"]["ok"] for g in gens if g["label"] == "inj_mask"), [g["mem_check"] for g in gens]
    assert gens[-1]["silence"]["silent"] is True and gens[0]["silence"]["silent"] is False
    # 状態の判定（CPU 側と同じ関数）
    span_ref = {"a": [1, 2], "b": [1, 2]}
    row_ok = {"pair_idx": 0, "aligned": True, **{f"{c}_per": 0.5 for c in C.PER_CONDS}}
    al = [{"ev": "align_tr", "label": lab} for lab in C.ALIGN_ORDER]
    s0, e0 = C.classify_pair(row_ok, [g for g in gens if g["pair_idx"] == 0], al, [], True, ROOT, span_ref)
    s2, e2 = C.classify_pair(dict(row_ok, pair_idx=2), [g for g in gens if g["pair_idx"] == 2], al, [], True,
                             ROOT, span_ref)
    s1, _ = C.classify_pair(None, [g for g in gens if g["pair_idx"] == 1], [], [], True, ROOT, span_ref)
    s1p, _ = C.classify_pair(None, [g for g in gens if g["pair_idx"] == 1], [], [], False, ROOT, span_ref)
    sbad, _ = C.classify_pair(row_ok, [g for g in gens if g["pair_idx"] == 0], al, [], True, ROOT,
                              {"a": [2, 3], "b": [1, 2]})
    s_un, _ = C.classify_pair(dict(row_ok, inj_per=None), [g for g in gens if g["pair_idx"] == 0], al, [], True,
                              ROOT, span_ref)
    s_wh, _ = C.classify_pair(row_ok, [g for g in gens if g["pair_idx"] == 0], al,
                              [{"kind": "whisper", "line": "x"}], True, ROOT, span_ref)
    res = {"pair0": s0, "pair2_silent_tgt": s2, "pair1_exception": s1, "pair1_incomplete_batch": s1p,
           "span_mismatch": sbad, "cond_without_entity": s_un, "whisper_exception": s_wh}
    want_res = {"pair0": "valid", "pair2_silent_tgt": "silent", "pair1_exception": "run_error",
                "pair1_incomplete_batch": "pending", "span_mismatch": C.STATUS_STOP,
                "cond_without_entity": "unaligned", "whisper_exception": "alignment_exception"}
    assert res == want_res, res
    # 表の書式: 今の表の EN 行を、今の results.json から作り直せること
    old = C.read_json(C.P(f"{C.OLD_DIR_REL}/results.json"))
    cells = C.table_cells(C.production_summary(old["rows"]))
    assert cells == C.PAPER_EN_CELLS, cells
    print(f"[appc] SELFTEST OK: {res}")
    print(f"[appc]   events: {got}")
    print(f"[appc]   今の表の EN 8 セルを今の results.json から再現: {cells == C.PAPER_EN_CELLS}")
    print(f"[appc]   wrote {run_rel}")
    return 0


# ---------------------------------------------------------------- 入り口
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, choices=C.SEEDS, default=None)
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--retry", type=int, default=None, help="同じ seed のやり直しの番号（retryNN）")
    ap.add_argument("--pairs", default=None, help="--retry の対（カンマ区切りの pair_idx）")
    ap.add_argument("--status_file", default=None, help="appc_en_scan.py の出力（seed 43/44 と --retry）")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force_resume", action="store_true",
                    help="前の起動が止まっていることを squeue で確かめた後、20 分待たずに再開する")
    args = ap.parse_args()

    errs = C.pinned_errors(ROOT) + C.old_result_errors(ROOT)
    if errs:
        raise SystemExit("[appc] " + " / ".join(errs))
    pairs = C.load_pairs(ROOT)
    if args.selftest:
        return selftest(pairs)
    if args.seed is None:
        raise SystemExit("[appc] --seed が要る")
    static = static_checks()
    span_rel, _ = latest_span_check(pairs)
    if args.dry_run:
        return dry_run(args, pairs, span_rel, static)
    return run_gpu(args, pairs, span_rel)


if __name__ == "__main__":
    sys.exit(main())
