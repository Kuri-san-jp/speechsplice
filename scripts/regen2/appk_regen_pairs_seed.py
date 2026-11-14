#!/usr/bin/env python3
"""G1: 付録 K の N=300 音声集合で使えない対を、seed 43（だめなら 44）で対ごとに作り直す（2026-09-29）.

著者の決定 (1): 使えない対（無音・欠け・読めない wav、生成の失敗）は、対ごと（BL と注入側）に seed 43 で作り直し、
43 でも使えなければ seed 44。選ぶ基準は使えるかどうかだけで、採点の結果では選ばない。
既定（--unit pair）は使えない対だけ（本番の計画では 39 対 = EN1/FR16/DE10/KO7/ZH5）。seed 44 に進むのも、
seed 43 で使えなかった対だけ。使える対は、同じ BL 文を共有していても作り直さない。
--unit bl_group（同じ BL 文の対をまとめて作り直し、まとめて seed 44 に進める。64 対になり、使える 25 対も
差し替わる）は、著者がはっきり認めたときだけ使う（appk_common.build_plan）。
無音の判定は regen2 共通の部品（scripts/regen2/silence_criterion.py。appk_common.wav_check 経由。基準は appk_common の説明を参照）。
seed 42 の確かめの対が作れなかった（行か wav がない）ときは、seed 43 に進まずに止める（status=run_error。
仕組みの不具合で seed 43/44 を使い切らない。直したあと同じコマンドで流し直せる）。

生成は本番と同じスクリプトをそのまま呼ぶ（1 バイトも変えない）:
  EN/FR/DE/KO: scripts/projection/eval_utmos_speaker_fixed.py の main()（--align splice、JSUT の声を見本にした声まね、
               本番 7/29 の utmos_speaker_fixed_{lang}_n300_splice と同じ引数）。
  ZH:          scripts/projection/eval_cjk_n300_zhv2.py の main()（--lang zh --save_wavs、本番 7/28 の cjk_n300_zh_v2）。
やり方は scripts/spkvar/regen_pairs_seed.py と同じ:
  * 出力先に、対象でない対を「済み」とした checkpoint を置く（本番の resume の仕組みで対象の対だけが作られる）。
  * モジュールの SEED を置き換える（E.SEED / Z.SEED。voice_clone_synth・main のループ・generate の直前の
    seed_everything(SEED) が実行時に読む）。
  * ZH は注意: load_zh_pairs が random.Random(SEED) で対を並べるので、SEED を変える前に seed 42 で対を作り、
    load_zh_pairs をその固定の並びを返す関数に置き換える（本番の checkpoint_zh.json の 300 行と一致を確かめる）。
  * 参照の書き起こし（EN/FR/DE/KO）は本番のログが無いので、seed 42 のまま対象外の 1 対（check_pair）を作り直し、
    本番と md5 が一致する候補を使う（appk_common.REF_TEXTS の順。どれも一致しなければ先頭の正解文を使い、
    selection に記録する。反映は --accept_check_mismatch がないと止まる）。
duration 推定（vendor の duration_estimator）は本番と同じ経路のまま（追加も削除もしない。このスクリプトは
pyopenjtalk を import しない。EN/FR/DE/KO は参照文が日本語なので言語別の既定話速、ZH は pypinyin の経路）。
言語ごとに子プロセスを 1 つ立てる（本番と同じく 1 言語 1 プロセス。言語どうしが干渉しない）。

置き場（本番は書き換えない。反映は appk_apply_regen_pairs.py）:
  eval_results/_appk_regen/plan.json                … 対象の対（--plan_only、CPU）
  eval_results/_appk_regen/check42/<lang>_<tag>/    … seed 42 の再現の確かめ
  eval_results/_appk_regen/seed43/<lang>/, seed44/<lang>/
  eval_results/_appk_regen/selection_<lang>.json    … 対ごとに採用した seed・wav の md5・行
HF: HF_HOME=/mnt/.cache/huggingface、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。tokenizer は
scripts/nontemplate/gen_t5gemma_pairs_module.py の offline shim を入れる（生成スクリプトは元から保存済みの
snapshot のパスを渡すので、shim は保険）。CPU スレッドは torch の import 前に 13。

使い方（CPU はホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定するので、env や nice は付けない。
相対パスの引数はプロジェクト直下を基準に解くので、どこから流しても同じ）:
  1. CPU 計画（既定は使えない対だけ = 39 対）:
       taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_regen_pairs_seed.py --plan_only --silent_scan eval_results/_appk_redo/silent_scan_n300.json
  2. CPU 予行:
       taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_regen_pairs_seed.py --cpu_dryrun
  3. GPU（$REPO_ROOT から投入。slurm_run.sh は投入したフォルダをコンテナの /mnt にする）:
       SLURM_TIME=02:00:00 slurm_run.sh python3 -u scripts/regen2/appk_regen_pairs_seed.py
     （言語ごとに selection_<lang>.json が "done" なら飛ばす。時間切れで止まっても同じコマンドで続きから）
終了コード: 0 = 全言語 done で未解決なし、2 = done だが seed 44 でも使えない対が残る、1 = 止まった言語がある。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_common as K  # noqa: E402  （torch を import しない）

K.cap_threads()

import argparse  # noqa: E402
import ast  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

PROJ_DIR = os.path.join(K.PROJ, "scripts", "projection")
GTPM = os.path.join(K.PROJ, "scripts", "nontemplate", "gen_t5gemma_pairs_module.py")
TEXT_KEYS = ("text_a", "text_b", "word_a", "word_b")
SEED_INDEP = {"E": ("align", "delta_tok", "T_src", "T_out", "inj_target_text", "bl_dur_sec", "inj_dur_sec"),
              "Z": ("cos_sim",)}
# 予行の報告と placeholder の置き場。検査（APPK_TEST_ROOT）では、その置き場の下に書く（本番の予行の報告を上書きしない）
DRY_BASE = K.TEST_ROOT or K.DRYRUN
DRY_OUT = os.path.join(DRY_BASE, "appk_regen_cpu_dryrun.json")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# 置き場の準備（placeholder の checkpoint と条件の記録）
# --------------------------------------------------------------------------
def ckpt_path(lang, out_dir):
    return os.path.join(out_dir, K.SETS[lang]["ckpt"] if K.SETS[lang]["gen"] == "Z" else "checkpoint.json")


def write_placeholders(lang, out_dir, targets):
    """対象でない対を「済み」とした checkpoint を置く（既にあれば中身を確かめるだけ）."""
    gen = K.SETS[lang]["gen"]
    p = ckpt_path(lang, out_dir)
    tg = set(targets)
    if os.path.isfile(p):
        d = K.read_json(p)
        rows = d["rows"] if gen == "E" else d["results"]
        ph = {r["pair_idx"] for r in rows if r.get("placeholder")}
        real = {r["pair_idx"] for r in rows if not r.get("placeholder")}
        if ph != set(range(K.N_EXPECTED)) - tg or not real <= tg:
            raise SystemExit(f"{p} の placeholder が今回の対象と合わない（別の条件の続きにしない）")
        return p
    os.makedirs(out_dir, exist_ok=True)
    if gen == "E":
        dummy = [{"pair_idx": i, "align": "splice", "bl_utmos": 0.0, "inj_utmos": 0.0, "placeholder": True}
                 for i in range(K.N_EXPECTED) if i not in tg]
        K.write_json_atomic({"rows": dummy}, p)
    else:
        dummy = [{"pair_idx": i, "lang": "zh", "status": "placeholder", "placeholder": True}
                 for i in range(K.N_EXPECTED) if i not in tg]
        K.write_json_atomic({"results": dummy}, p, indent=2)
    return p


def guard_config(out_dir, cfg):
    """同じ置き場を別の条件（seed・対象・参照文）で続けない."""
    K.assert_regen_path(out_dir)
    p = os.path.join(out_dir, "regen_config.json")
    if os.path.isfile(p):
        old = K.read_json(p)
        keys = ("lang", "seed", "targets", "ref_text", "generator")
        if any(old.get(k) != cfg.get(k) for k in keys):
            raise SystemExit(f"{out_dir} は別の条件で作られている（{ {k: old.get(k) for k in keys} }）。続けない。")
        return
    K.write_json_atomic(cfg, p)


def out_rows(lang, out_dir):
    p = ckpt_path(lang, out_dir)
    if not os.path.isfile(p):
        return {}
    d = K.read_json(p)
    rows = d["rows"] if K.SETS[lang]["gen"] == "E" else d["results"]
    return {r["pair_idx"]: r for r in rows if not r.get("placeholder")}


# --------------------------------------------------------------------------
# 使えるかの判定（使える対だけを採用する。採点の結果は見ない）
# --------------------------------------------------------------------------
def validate_pair(lang, out_dir, i, prod_row, rows):
    gen = K.SETS[lang]["gen"]
    reasons, files = [], {}
    row = rows.get(i)
    if row is None:
        reasons.append("no_row (generation failed or skipped)")
    else:
        if gen == "Z" and row.get("status") != "success":
            reasons.append(f"status={row.get('status')} error={row.get('error')}")
        bad_txt = [k for k in TEXT_KEYS if row.get(k) != prod_row.get(k)]
        if bad_txt:
            reasons.append(f"text_mismatch {bad_txt}")
    for side in K.SIDES:
        n = K.wav_name(lang, i, side)
        p = os.path.join(out_dir, n)
        st = K.wav_check(p)
        rec = {"src": p, "bad": st["bad"]}
        if not st.get("missing") and not st.get("unreadable"):
            rec.update({"md5": K.md5_file(p), "rms": st["rms"], "peak": st["peak"], "active_frac": st["active_frac"],
                        "dur_sec": st["dur_sec"], "silent_reason": st["silent_reason"]})
        if st["bad"]:
            reasons.append(f"{side}: {st['why']}")
        files[side] = rec
    diff = {}
    if row is not None:
        for k in SEED_INDEP[gen]:
            if row.get(k) != prod_row.get(k):
                diff[k] = [prod_row.get(k), row.get(k)]
    return {"valid": not reasons, "reasons": reasons, "files": files, "row": row,
            "seed_independent_diff_vs_prod": diff}


def wav_similarity(a, b):
    import numpy as np
    import soundfile as sf
    x, sx = sf.read(a, dtype="float64")
    y, sy = sf.read(b, dtype="float64")
    out = {"n_new": int(len(x)), "n_prod": int(len(y)), "sr_new": int(sx), "sr_prod": int(sy)}
    if len(x) == len(y) and len(x):
        d = x - y
        out["max_abs_diff"] = float(np.abs(d).max())
        pw = float((y ** 2).mean())
        dp = float((d ** 2).mean())
        out["snr_db"] = None if dp == 0 else round(10 * np.log10(pw / dp), 2) if pw > 0 else None
    return out


def compare_check(lang, out_dir, c, plan):
    L = plan["langs"][lang]
    res = {"pair": c, "files": {}}
    exact = True
    for side in K.SIDES:
        n = K.wav_name(lang, c, side)
        p = os.path.join(out_dir, n)
        prod = L["prod_md5"][n]
        if not os.path.isfile(p):
            res["files"][n] = {"missing": True, "prod_md5": prod}
            exact = False
            continue
        m = K.md5_file(p)
        rec = {"md5": m, "prod_md5": prod, "equal": m == prod}
        if m != prod:
            rec["similarity"] = wav_similarity(p, os.path.join(K.set_dir(lang), n))
            exact = False
        res["files"][n] = rec
    res["exact"] = exact
    return res


# --------------------------------------------------------------------------
# 1 言語ぶんの流れ（runner を差し替えて CPU で検査できる: appk_selftest_regen.py）
# --------------------------------------------------------------------------
def process_language(lang, plan, runner):
    L = plan["langs"][lang]
    gen = K.SETS[lang]["gen"]
    spath = K.selection_path(lang)
    sel = {"lang": lang, "status": "in_progress", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "plan": K.plan_path(), "unit": plan["unit"], "generator": K.GEN_E if gen == "E" else K.GEN_Z,
           "generator_md5": K.md5_file(K.GEN_E if gen == "E" else K.GEN_Z),
           "silence_criterion": K.SILENCE_CRITERION, "silence_helper": K.SILENCE_HELPER,
           "invalid_pairs": L["invalid_pairs"], "units_planned": L["units"], "targets": L["targets"],
           "seed_runs": {}, "units": [], "adopted": {}}

    def save():
        K.write_json_atomic(sel, spath)

    if os.path.isfile(spath):   # 前の回（止まった回）の記録を写してから書き直す（移動・削除はしない）
        bak = f"{spath}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(spath, bak)
        log(f"[{lang}] [backup] {spath} -> {bak}")

    def run_stage(seed, targets, ref_text, out_dir):
        cfg = {"lang": lang, "seed": seed, "targets": sorted(targets), "ref_text": ref_text,
               "generator": sel["generator"], "generator_md5": sel["generator_md5"]}
        guard_config(out_dir, cfg)
        t0 = time.time()
        info = runner(seed=seed, targets=sorted(targets), ref_text=ref_text, out_dir=out_dir) or {}
        rows = out_rows(lang, out_dir)
        missing = [i for i in targets if i not in rows]
        info.update({"dir": out_dir, "seed": seed, "targets": sorted(targets),
                     "elapsed_min": round((time.time() - t0) / 60, 2), "rows_missing": missing,
                     # 例外で止まり、行が欠けたとき（OOM など）は、無音と同じ扱いで次の seed に進めずに止める
                     "fatal": bool(info.get("exception")) and bool(missing)})
        return info, rows

    # 1) seed 42 のまま対象外の 1 対を作り直し、本番と同じ音声になるか（参照文の候補を順に）
    c = L["check_pair"]
    cands = list(K.REF_TEXTS) if gen == "E" else [("noref", None)]
    chk = {"pair": c, "candidates": []}
    chosen = None
    for tag, text in cands:
        out = K.check_dir(lang, tag)
        info, _ = run_stage(42, [c], text, out)
        res = compare_check(lang, out, c, plan)
        res.update({"tag": tag, "ref_text": text, "run": info})
        chk["candidates"].append(res)
        log(f"[{lang}] check pair {c} seed 42 ref={tag}: exact={res['exact']} "
            f"{ {n: v.get('equal') for n, v in res['files'].items()} }")
        if info["fatal"]:
            sel.update(status="run_error", check=chk, error=f"check run failed: {info.get('exception')}")
            save()
            return sel
        if res["exact"]:
            chosen = (tag, text)
            break
    # 確かめの対が作れなかった（行か wav がない）候補。一致した候補がなく、作れなかった候補があるときは、
    # 仕組みの不具合（参照の wav が読めない、など）とみなし、seed 43 に進まずに止める
    not_generated = [x["tag"] for x in chk["candidates"]
                     if x["run"]["rows_missing"] or any(v.get("missing") for v in x["files"].values())]
    chk["not_generated"] = not_generated
    if chosen is None and not_generated:
        chk["exact"] = False
        sel.update(status="run_error", check=chk,
                   error=(f"check pair {c} was not generated at seed 42 (no row or no wav; candidates {not_generated}). "
                          f"seed 43/44 were not run. Fix the pipeline and rerun the same command."))
        log(f"[{lang}] STOP: seed 42 の確かめの対 {c} が作れなかった（{not_generated}）。seed 43 に進まない。")
        save()
        return sel
    chk["exact"] = chosen is not None
    if chosen is None:
        chosen = cands[0]
        log(f"[{lang}] WARNING: seed 42 の作り直しが本番と一致しない（どの候補でも）。{chosen[0]} で続け、記録する。"
            f"反映には --accept_check_mismatch が要る。")
    chk["chosen_tag"] = chosen[0]
    sel.update(check=chk, ref_text_tag=chosen[0], ref_text=chosen[1])
    save()

    # 2) seed 43、使えない組だけ seed 44（組ごとに進める）
    pending = [list(u) for u in L["units"]]
    unit_recs = [{"pairs": list(u), "tried": {}, "adopted_seed": None} for u in pending]
    by_unit = {tuple(u["pairs"]): u for u in unit_recs}
    for seed in K.SEEDS:
        if not pending:
            break
        targets = sorted(i for u in pending for i in u)
        out = K.seed_dir(seed, lang)
        info, rows = run_stage(seed, targets, chosen[1], out)
        sel["seed_runs"][str(seed)] = info
        if info["fatal"]:
            sel.update(status="run_error", units=unit_recs, error=f"seed {seed} run failed: {info.get('exception')}")
            save()
            return sel
        val = {i: validate_pair(lang, out, i, L["prod_rows"][str(i)], rows) for i in targets}
        nxt = []
        for u in pending:
            rec = by_unit[tuple(u)]
            rec["tried"][str(seed)] = {str(i): {k: val[i][k] for k in ("valid", "reasons", "files",
                                                                      "seed_independent_diff_vs_prod")}
                                       for i in u}
            if all(val[i]["valid"] for i in u):
                rec["adopted_seed"] = seed
                for i in u:
                    sel["adopted"][str(i)] = {"seed": seed, "dir": out, "files": val[i]["files"],
                                              "row": val[i]["row"],
                                              "seed_independent_diff_vs_prod": val[i]["seed_independent_diff_vs_prod"]}
            else:
                nxt.append(u)
        log(f"[{lang}] seed {seed}: units ok {len(pending) - len(nxt)}/{len(pending)}; "
            f"still unusable {[u for u in nxt]}")
        pending = nxt
        sel["units"] = unit_recs
        save()
    sel["unresolved_units"] = pending
    sel["unresolved_pairs"] = sorted(i for u in pending for i in u)
    sel["n_adopted"] = len(sel["adopted"])
    sel["status"] = "done"
    sel["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log(f"[{lang}] done: adopted {sel['n_adopted']} pairs "
        f"(seed 43: {sum(1 for v in sel['adopted'].values() if v['seed'] == 43)}, "
        f"seed 44: {sum(1 for v in sel['adopted'].values() if v['seed'] == 44)}); unresolved {sel['unresolved_pairs']}")
    return sel


# --------------------------------------------------------------------------
# 生成スクリプトの対の読み方（CPU でも GPU でも、元のソースの関数そのものを使う）
# --------------------------------------------------------------------------
def _function_source(path, name):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(fns) == 1, (path, name)
    return ast.get_source_segment(src, fns[0]), src


def pairs_from_source_E(lang):
    """eval_utmos_speaker_fixed.load_pairs をソースから取り出して実行（torch なし）."""
    fsrc, _ = _function_source(K.GEN_E, "load_pairs")
    data = os.path.join(K.PROJ, "data")
    if data not in sys.path:
        sys.path.insert(0, data)
    ns = {"os": os, "sys": sys, "json": json, "PROJECT_ROOT": K.PROJ}
    exec(fsrc, ns)
    return ns["load_pairs"](lang, K.N_EXPECTED, pairs_module=None)


def pairs_from_source_Z(seed=42):
    fsrc, _ = _function_source(K.GEN_Z, "load_zh_pairs")
    ns = {"random": random, "SEED": seed,
          "ZH_POLYPHONES": K.read_json(os.path.join(K.PROJ, "data", "zh_polyphones_v2.json"))}
    exec(fsrc, ns)
    return ns["load_zh_pairs"](K.N_EXPECTED)


def compare_pairs(lang, pairs):
    prod = {r["pair_idx"]: r for r in K.load_rows(lang)}
    bad = []
    for i, p in enumerate(pairs):
        pidx = p.get("pair_idx", i)
        r = prod.get(pidx)
        if pidx != i or r is None or any(p.get(k) != r.get(k) for k in TEXT_KEYS):
            bad.append(i)
    return {"n_pairs": len(pairs), "n_mismatch": len(bad), "mismatch": bad[:10],
            "ok": len(pairs) == K.N_EXPECTED and not bad}


def verify_prod_unchanged(plan, lang):
    """計画の時点から、本番の対象・確かめの wav が変わっていないこと."""
    L = plan["langs"][lang]
    changed = [n for n, m in L["prod_md5"].items()
               if not os.path.isfile(os.path.join(K.set_dir(lang), n))
               or K.md5_file(os.path.join(K.set_dir(lang), n)) != m]
    return changed


def load_gtpm():
    spec = importlib.util.spec_from_file_location("appk_gtpm_shim", GTPM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_hf_env():
    hf = os.path.join(K.PROJ, ".cache", "huggingface")
    os.environ["HF_HOME"] = hf
    os.environ["HF_HUB_CACHE"] = os.path.join(hf, "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# --------------------------------------------------------------------------
# GPU: 1 言語の子プロセス
# --------------------------------------------------------------------------
def worker(lang, plan_file):
    plan = K.read_json(plan_file)
    if lang not in plan["langs"]:
        raise SystemExit(f"[{lang}] 計画にない")
    changed = verify_prod_unchanged(plan, lang)
    if changed:
        log(f"[{lang}] STOP: 計画のあとで本番の wav が変わっている {changed[:5]}。計画を作り直す。")
        sys.exit(2)
    set_hf_env()
    gtpm = load_gtpm()
    ok, pre = gtpm.offline_preflight()
    if not ok:
        log(f"[{lang}] STOP: HF の保存済み snapshot が足りない {pre}")
        sys.exit(2)
    import torch
    torch.set_num_threads(K.THREADS)
    shim_log = []
    gtpm.install_offline_tokenizer_shim(shim_log)
    log(f"[{lang}] torch threads={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} "
        f"HF_HOME={os.environ['HF_HOME']} offline=1 cuda={torch.cuda.is_available()}")
    # 生成スクリプトを直接実行したときと同じ sys.path（先頭が scripts/projection）
    sys.path[:] = [PROJ_DIR if p == _HERE else p for p in sys.path]
    if PROJ_DIR not in sys.path:
        sys.path.insert(0, PROJ_DIR)
    import gc
    gen = K.SETS[lang]["gen"]
    if gen == "E":
        import eval_utmos_speaker_fixed as M
        assert M.SEED == 42
        pairs = M.load_pairs(lang, K.N_EXPECTED, pairs_module=None)
        script = K.GEN_E
    else:
        import eval_cjk_n300_zhv2 as M
        assert M.SEED == 42
        pairs = M.load_zh_pairs(K.N_EXPECTED)            # seed 42 の並び（本番と同じ）
        pinned = [dict(p) for p in pairs]
        M.load_zh_pairs = lambda n: [dict(p) for p in pinned[:n]]   # SEED を変えても対の並びは変えない
        script = K.GEN_Z
    cmp = compare_pairs(lang, pairs)
    log(f"[{lang}] pairs vs production checkpoint: {cmp}")
    if not cmp["ok"]:
        sys.exit(2)

    def runner(seed, targets, ref_text, out_dir):
        write_placeholders(lang, out_dir, targets)
        if gen == "E":
            argv = ["--lang", lang, "--n", str(K.N_EXPECTED), "--align", "splice",
                    "--reference_wav", K.REF_WAV, "--reference_text", ref_text,
                    "--reference_prompt_sec", str(K.REF_PROMPT_SEC), "--output_dir", out_dir]
        else:
            argv = ["--lang", "zh", "--output_dir", out_dir, "--save_wavs"]
        log(f"[{lang}] run seed={seed} targets={targets} argv={argv}")
        saved = sys.argv
        exc = None
        M.SEED = seed
        sys.argv = [script] + argv
        try:
            M.main()
        except BaseException as ex:  # noqa: BLE001  集計（placeholder を含む）で落ちても wav と行は保存済み
            exc = f"{type(ex).__name__}: {ex}"
            traceback.print_exc()
            if isinstance(ex, KeyboardInterrupt):
                raise
        finally:
            sys.argv = saved
            M.SEED = 42
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"argv": argv, "exception": exc, "tokenizer_shim": list(shim_log)}

    sel = process_language(lang, plan, runner)
    sel["offline_preflight"] = pre
    sel["torch"] = torch.__version__
    K.write_json_atomic(sel, K.selection_path(lang))
    sys.exit(0 if sel["status"] == "done" else 1)


# --------------------------------------------------------------------------
# CPU: 計画と予行
# --------------------------------------------------------------------------
def plan_only(args):
    scan_path = args.silent_scan if os.path.isabs(args.silent_scan) else os.path.join(K.PROJ, args.silent_scan)
    scan = K.read_json(scan_path)
    missing = [l for l in K.LANGS if l not in scan["langs"]]
    if missing:
        raise SystemExit(f"無音の検査に {missing} がない（6 言語の検査結果を渡す）")
    if scan.get("er_root") and os.path.abspath(scan["er_root"]) != K.ER:
        raise SystemExit(f"無音の検査の置き場 {scan['er_root']} と今の置き場 {K.ER} が違う")
    langs = [l for l in args.langs.split(",") if l]
    plan = K.build_plan(scan, scan_path, langs=langs, unit=args.unit)
    plan["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if "ja_warning" in plan:
        log("WARNING:", plan["ja_warning"])
    if plan["unit"] != "pair":
        log("WARNING: --unit bl_group は同じ BL 文の使える対も作り直して差し替える（著者決定 (1) の外）。"
            "著者がはっきり認めたときだけ使う。")
    p = K.plan_path()
    if os.path.exists(p):
        old = K.read_json(p)
        same = all(old["langs"].get(l, {}).get("targets") == plan["langs"][l]["targets"] for l in langs) \
            and old.get("unit") == plan["unit"]
        started = [l for l in langs if os.path.exists(K.selection_path(l))]
        if started and not same:
            raise SystemExit(f"{p} は既にあり、作り直しが始まっている言語 {started} の対象が変わる。止める。")
        bak = f"{p}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(p, bak)   # 前の計画を写してから上書きする（移動・削除はしない）
        log(f"[backup] {p} -> {bak}")
    K.write_json_atomic(plan, p)
    for l in langs:
        L = plan["langs"][l]
        log(f"[{l}] invalid {L['n_invalid']} pairs {L['invalid_pairs']}")
        log(f"     units ({plan['unit']}) {L['units']} -> {L['n_targets']} pairs "
            f"(added as BL-sentence members: {L['added_as_bl_group_member']}); check pair {L['check_pair']}")
    log(f"[plan] total targets {sum(plan['langs'][l]['n_targets'] for l in langs)} "
        f"(invalid {sum(plan['langs'][l]['n_invalid'] for l in langs)}) -> {p}")


def cpu_dryrun(args):
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": K.plan_path(), "langs": {}}
    plan = K.read_json(K.plan_path())
    langs = [l for l in args.langs.split(",") if l in plan["langs"]]
    # 生成スクリプトの SEED の読み方（実行時にモジュールの SEED を読むこと）
    srcE = open(K.GEN_E, encoding="utf-8").read()
    srcZ = open(K.GEN_Z, encoding="utf-8").read()
    rep["static"] = {
        "E_md5": K.md5_file(K.GEN_E), "Z_md5": K.md5_file(K.GEN_Z),
        "E_seed_everything_SEED": srcE.count("seed_everything(SEED)"),
        "E_SEED_42": "\nSEED = 42\n" in srcE,
        "Z_seed_everything_SEED": srcZ.count("seed_everything(SEED)"),
        "Z_SEED_42": "\nSEED = 42\n" in srcZ,
        "Z_load_zh_pairs_uses_SEED": "rng = random.Random(SEED)" in _function_source(K.GEN_Z, "load_zh_pairs")[0],
        "g1_scripts_importing_pyopenjtalk": sorted(
            n for n in os.listdir(_HERE) if n.startswith("appk_") and n.endswith(".py")
            and any((isinstance(x, ast.Import) and any("pyopenjtalk" in a.name for a in x.names))
                    or (isinstance(x, ast.ImportFrom) and "pyopenjtalk" in (x.module or ""))
                    for x in ast.walk(ast.parse(open(os.path.join(_HERE, n), encoding="utf-8").read())))),
    }
    assert not rep["static"]["g1_scripts_importing_pyopenjtalk"], rep["static"]
    assert rep["static"]["E_seed_everything_SEED"] >= 2 and rep["static"]["E_SEED_42"], rep["static"]
    assert rep["static"]["Z_seed_everything_SEED"] == 3 and rep["static"]["Z_SEED_42"], rep["static"]
    assert rep["static"]["Z_load_zh_pairs_uses_SEED"], rep["static"]
    # HF の保存済み snapshot（ホストの cache で。コンテナでは同じフォルダが /mnt/.cache/huggingface）
    set_hf_env()
    gtpm = load_gtpm()
    ok, pre = gtpm.offline_preflight()
    rep["offline_preflight"] = {"ok": ok, "repos": pre}
    th = os.path.join(K.PROJ, ".torch_cache", "hub")
    rep["torch_hub_cache"] = {"SpeechMOS": os.path.isdir(os.path.join(th, "tarepan_SpeechMOS_v1.2.0")),
                              "SQUIM_objective": os.path.isfile(os.path.join(th, "torchaudio", "models",
                                                                             "squim_objective_dns2020.pth")),
                              "checkpoints": sorted(os.listdir(os.path.join(th, "checkpoints")))
                              if os.path.isdir(os.path.join(th, "checkpoints")) else None}
    dd = os.path.join(DRY_BASE, f"appk_regen_placeholder_check_{time.strftime('%Y%m%d_%H%M%S')}")
    for lang in langs:
        L = plan["langs"][lang]
        gen = K.SETS[lang]["gen"]
        r = {"prod_changed_since_plan": verify_prod_unchanged(plan, lang)}
        if gen == "E":
            r["pairs_vs_prod"] = compare_pairs(lang, pairs_from_source_E(lang))
        else:
            p42 = pairs_from_source_Z(42)
            r["pairs_vs_prod"] = compare_pairs(lang, p42)
            r["pairs_if_SEED_43_not_pinned_vs_prod"] = compare_pairs(lang, pairs_from_source_Z(43))
        # placeholder の checkpoint を試しに作り、生成スクリプトの resume の数え方で対象だけが残るか
        out = os.path.join(dd, lang)   # 毎回新しいフォルダ（消さない。TTS26 は削除禁止）
        tg = L["targets"]
        p = write_placeholders_to(lang, out, tg)
        d = K.read_json(p)
        rows = d["rows"] if gen == "E" else d["results"]
        done = {x["pair_idx"] for x in rows}
        if gen == "E":
            assert all(x.get("align") == "splice" for x in rows)
        else:
            assert all("status" in x for x in rows)
        would = [i for i in range(K.N_EXPECTED) if i not in done]
        r["placeholder_resume_generates_exactly_targets"] = would == tg
        if gen == "E":
            r["argv_seed43"] = ["--lang", lang, "--n", "300", "--align", "splice", "--reference_wav", K.REF_WAV,
                                "--reference_text", K.REF_TEXTS[0][1], "--reference_prompt_sec",
                                str(K.REF_PROMPT_SEC), "--output_dir", K.seed_dir(43, lang)]
        else:
            r["argv_seed43"] = ["--lang", "zh", "--output_dir", K.seed_dir(43, lang), "--save_wavs"]
        r["targets"] = tg
        r["check_pair"] = L["check_pair"]
        r["out_dirs_exist"] = {d_: os.path.exists(d_) for d_ in
                               [K.seed_dir(43, lang), K.seed_dir(44, lang), K.selection_path(lang)]}
        rep["langs"][lang] = r
        log(f"[{lang}] pairs ok={r['pairs_vs_prod']['ok']} prod_changed={r['prod_changed_since_plan'][:3]} "
            f"placeholder->targets {r['placeholder_resume_generates_exactly_targets']} targets={len(tg)} "
            f"check={L['check_pair']}"
            + (f" (unpinned SEED=43 mismatches {r['pairs_if_SEED_43_not_pinned_vs_prod']['n_mismatch']})"
               if gen == "Z" else ""))
    rep["ok"] = (ok and all(v["pairs_vs_prod"]["ok"] and not v["prod_changed_since_plan"]
                            and v["placeholder_resume_generates_exactly_targets"] for v in rep["langs"].values()))
    K.write_json_atomic(rep, DRY_OUT)
    log(f"[cpu_dryrun] offline_preflight ok={ok}; all ok={rep['ok']} -> {DRY_OUT}")
    return rep


def write_placeholders_to(lang, out_dir, targets):
    """予行用（_regen2_dryrun の下）に placeholder を書く."""
    if not os.path.abspath(out_dir).startswith(K.DRYRUN + "/"):
        raise SystemExit(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    gen = K.SETS[lang]["gen"]
    p = os.path.join(out_dir, "checkpoint.json" if gen == "E" else "checkpoint_zh.json")
    tg = set(targets)
    if gen == "E":
        K.write_json_atomic({"rows": [{"pair_idx": i, "align": "splice", "bl_utmos": 0.0, "inj_utmos": 0.0,
                                       "placeholder": True} for i in range(K.N_EXPECTED) if i not in tg]}, p)
    else:
        K.write_json_atomic({"results": [{"pair_idx": i, "lang": "zh", "status": "placeholder", "placeholder": True}
                                         for i in range(K.N_EXPECTED) if i not in tg]}, p, indent=2)
    return p


# --------------------------------------------------------------------------
# GPU: 親（言語ごとに子プロセスを順に立てる。torch は読まない）
# --------------------------------------------------------------------------
def parent(args):
    plan_file = K.plan_path()
    if not os.path.isfile(plan_file):
        raise SystemExit(f"計画がない: {plan_file}（先に --plan_only）")
    plan = K.read_json(plan_file)
    langs = [l for l in args.langs.split(",") if l in plan["langs"]]
    set_hf_env()
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": plan_file, "langs": {}}
    mpath = os.path.join(K.REGEN_ROOT, f"gpu_run_meta_{time.strftime('%Y%m%d_%H%M%S')}.json")
    rc_all = 0
    for lang in langs:
        sp = K.selection_path(lang)
        if os.path.isfile(sp) and K.read_json(sp).get("status") == "done":
            log(f"[{lang}] selection は done（飛ばす）: {sp}")
            meta["langs"][lang] = {"status": "skipped_done"}
            continue
        if not plan["langs"][lang]["targets"]:
            meta["langs"][lang] = {"status": "no_targets"}
            continue
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "--worker", "--lang", lang],
                             env=os.environ.copy())
        meta["langs"][lang] = {"returncode": rc, "elapsed_min": round((time.time() - t0) / 60, 2)}
        if os.path.isfile(sp):
            s = K.read_json(sp)
            meta["langs"][lang].update(status=s.get("status"), unresolved=s.get("unresolved_pairs"),
                                       check_exact=(s.get("check") or {}).get("exact"))
        if rc != 0:
            rc_all = 1
        K.write_json_atomic(meta, mpath)
    unresolved = {l: v.get("unresolved") for l, v in meta["langs"].items() if v.get("unresolved")}
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    K.write_json_atomic(meta, mpath)
    log(f"[run] {json.dumps(meta['langs'], ensure_ascii=False)} -> {mpath}")
    if rc_all == 0 and unresolved:
        rc_all = 2
    sys.exit(rc_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(K.REGEN_LANGS))
    ap.add_argument("--plan_only", action="store_true", help="CPU: 無音の検査から計画を作るだけ")
    ap.add_argument("--silent_scan", default="eval_results/_appk_redo/silent_scan_n300.json")
    ap.add_argument("--unit", default="pair", choices=["pair", "bl_group"],
                    help="pair（既定）= 使えない対だけ（39 対）。bl_group = 同じ BL 文の対をまとめる（64 対。"
                         "使える対も差し替わるので、著者がはっきり認めたときだけ）")
    ap.add_argument("--cpu_dryrun", action="store_true", help="CPU: 計画・対の並び・placeholder・HF cache を確かめる")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--lang", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.plan_only:
        plan_only(args)
    elif args.cpu_dryrun:
        rep = cpu_dryrun(args)
        sys.exit(0 if rep["ok"] else 1)
    elif args.worker:
        worker(args.lang, K.plan_path())
    else:
        parent(args)


if __name__ == "__main__":
    main()
