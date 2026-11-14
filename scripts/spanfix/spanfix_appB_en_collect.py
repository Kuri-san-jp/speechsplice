#!/usr/bin/env python3
"""spanfix (B): 付録 B 英語 300 対の作り直しの状態の確認・集計（CPU のみ, 2026-10-01）.

GPU の実行（spanfix_appB_en_gen.py）が eval_results/_spanfix_appB_en/seed<S>/ に残した記録を読み、
  --status         対ごとの状態と次にすること、次に流す GPU のコマンド（SLURM_TIME つき）を表示する（書かない）
  --check_current  今の表の英語の値が、今の出どころ（9/29 の入れ替え後の 300 対）から今の式で再現できるかを確かめる（書かない）
  --preview        今そろっている有効な対だけで表の値を計算して表示する（途中経過。書かない）
  --aggregate      300 対すべてに有効な記録がそろったときだけ、最終の表の値を計算し、新しい置き場
                   eval_results/_spanfix_appB_en/final_<日時>/ に書く（既存のファイルは上書きしない）
どの記録を使うかは seed の決まり（spanfix_common.next_action）が決める。人は選ばない。数値は選ぶのに使わない:
  seed 42 の最後の回が有効ならそれ。「対として使えない」なら seed 43、さらに使えなければ seed 44。
  取り込む前に、5 本の wav の md5 と無音の判定をファイルから計算し直して記録と一致することを確かめる。
表の値の式は今の表と同じ（spanfix_common.TABLE_KEYS と CI_METHOD）:
  MCD (dB): mcd_bl_inj（Intact）、mcd_bl_mask_inj_mask（Blocked at entity）
  MEL Similarity: tmean_bl_inj_mel、tmean_bl_mask_inj_mask_mel（実体語の区間で時間平均した log-mel の cosine 類似度）
  値 = 平均 ± 1.96 × 標準偏差（n で割る, ddof=0）/ √n
naacl2027/ ほか原稿の置き場には書かない（古い値と新しい値を表示するだけ）。

使い方（ホストで直接）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/spanfix_appB_en_collect.py --status
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/spanfix_appB_en_collect.py --check_current
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/spanfix_appB_en_collect.py --aggregate
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spanfix_common as C  # noqa: E402

LAUNCH_ENV = C.setup_env()  # numpy を読む前に

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, C.P("scripts/regen2"))
import silence_criterion as SC  # noqa: E402

ROOT = C.HOST_ROOT
# eval_crossattn_entity_dtw.py の main 末尾と同じ並び（summary を同じ式で作る）
METRICS = ["mel", "w2v", "pps"]
PAIRS_TO_COMPARE = [("bl", "inj"), ("bl_mask", "inj_mask"), ("bl", "bl_mask"), ("inj", "inj_mask"),
                    ("bl", "tgt"), ("inj", "tgt")]
GEN = "/mnt/scripts/spanfix/spanfix_appB_en_gen.py"
CHUNK = 75  # seed 43 / 44 で 1 つのジョブに入れる対の上限（SLURM_TIME=02:00:00 に収まる）


def summarize(rows):
    """本番 main() の末尾と同じ式（elapsed_min は無し）."""
    aligned = [r for r in rows if r.get("aligned")]

    def avg(key):
        vs = [r[key] for r in aligned if r.get(key) is not None]
        return [round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)] if vs else [None, None]

    s = {"lang": "en", "n_total": len(rows), "n_aligned": len(aligned)}
    for m in METRICS:
        for a, b in PAIRS_TO_COMPARE:
            for prefix in ["tmean", "dtw", "fmax", "vowel"]:
                s[f"{prefix}_{a}_{b}_{m}"] = avg(f"{prefix}_{a}_{b}_{m}")
            if m == "mel":
                s[f"frob_{a}_{b}"] = avg(f"frob_{a}_{b}")
    for a, b in PAIRS_TO_COMPARE:
        s[f"mcd_{a}_{b}"] = avg(f"mcd_{a}_{b}")
    return s


def ranges(ids):
    """[0,1,2,5] -> "0-2,5"（表示を短くするだけ）."""
    out, ids = [], sorted(ids)
    k = 0
    while k < len(ids):
        j = k
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(f"{ids[k]}-{ids[j]}" if j > k else f"{ids[k]}")
        k = j + 1
    return ",".join(out)


def gpu_cmd(seed, extra, n, log_name):
    st, est = C.slurm_time_for(n)
    return {"slurm_time": st, "expected_min": est, "n_pairs": n,
            "command": f"SLURM_TIME={st} slurm_run.sh python3 -u {GEN} --seed {seed} {extra} "
                       f">> {C.LOGS_REL}/{log_name}.log 2>&1"}


def status(data_root=ROOT, save=False):
    runs, warns = C.all_runs(data_root)
    acts = {i: C.next_action(runs, i, ROOT) for i in range(C.N_PAIRS)}
    by = {}
    for i, a in acts.items():
        by.setdefault(a["action"], []).append(i)
    unfinished = [{"run": r["rel"], "n_todo": len(r["todo"]), "started": r["record"].get("started")}
                  for s in C.SEEDS for r in runs[s] if r["end"] is None]
    cmds = []
    for k in sorted(C.PARTS):
        part = set(C.part_pairs(k))
        n = len([i for i in by.get("run_seed42", []) + by.get("rerun_seed42", []) if i in part])
        cs = sorted({acts[i]["unfinished_run"] for i in by.get("confirm_stopped_seed42", []) if i in part})
        if cs:
            n += len([i for i in by.get("confirm_stopped_seed42", []) if i in part])
            extra = f"--part {k} " + " ".join(f"--confirm_stopped {r}" for r in cs)
            cmds.append({**gpu_cmd(42, extra, n, f"seed42_part{k}"),
                         "before": "squeue -u $USER で、上の置き場の実行が動いていないことを確かめる"})
        elif n:
            cmds.append(gpu_cmd(42, f"--part {k}", n, f"seed42_part{k}"))
    for s in (43, 44):
        ids = sorted(by.get(f"run_seed{s}", []) + by.get(f"rerun_seed{s}", []))
        cs = sorted({acts[i]["unfinished_run"] for i in by.get(f"confirm_stopped_seed{s}", [])})
        ids_c = sorted(by.get(f"confirm_stopped_seed{s}", []))
        allids = ids + ids_c
        for j in range(0, len(allids), CHUNK):
            chunk = allids[j:j + CHUNK]
            extra = "--pairs " + ",".join(str(i) for i in chunk)
            cc = sorted({acts[i]["unfinished_run"] for i in chunk if i in ids_c})
            if cc:
                extra += " " + " ".join(f"--confirm_stopped {r}" for r in cc)
            cmds.append(gpu_cmd(s, extra, len(chunk), f"seed{s}_regen"))
    counts = {k: len(v) for k, v in sorted(by.items())}
    seed_counts = {}
    for s in C.SEEDS:
        c = {}
        for r in runs[s]:
            for i in r["todo"]:
                st, _ = C.pair_status(r, i)
                c[st] = c.get(st, 0) + 1
        seed_counts[str(s)] = c
    out = {"date": C.now(), "n_runs": {str(s): len(runs[s]) for s in C.SEEDS}, "action_counts": counts,
           "attempt_status_counts_by_seed": seed_counts,
           "pairs_by_action": {k: ranges(v) for k, v in sorted(by.items())},
           "stops": {str(i): {"action": acts[i]["action"], "note": acts[i].get("note"),
                              "errors": acts[i]["errors"][:5], "code_drift": acts[i].get("code_drift")}
                     for i in range(C.N_PAIRS) if acts[i]["action"].startswith("stop")},
           "unfinished_runs": unfinished, "record_warnings": warns, "next_gpu_commands": cmds,
           "ready_to_aggregate": all(a["action"].startswith("use_seed") for a in acts.values())}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if save:
        p = os.path.join(data_root, C.OUT_REL, "status", f"status_{time.strftime('%Y%m%d_%H%M%S')}.json")
        C.write_json_new({**out, "per_pair": {str(i): a for i, a in acts.items()}}, p)
        print(f"[spanfix-collect] status saved: {os.path.relpath(p, data_root)}")
    return out


def current_table():
    """今の表の英語の値を、今の出どころ（9/29 の入れ替え後の 300 対）から今の式で計算し直す."""
    res = C.read_json(C.P(C.CUR_TABLE_SRC_REL, ROOT))
    tv = C.table_values(res["rows"])
    repro = {m: (tv[m]["intact"]["tex"], tv[m]["blocked"]["tex"]) == C.CURRENT_TABLE_EN[m] for m in C.TABLE_KEYS}
    eor_ok = C.eor_tex(tv) == C.CURRENT_EOR_L279_EN_MCD
    return res, tv, repro, eor_ok


def check_current():
    res, tv, repro, eor_ok = current_table()
    prod = C.read_json(C.P(C.PROD_DIR_REL + "/results.json", ROOT))
    md5 = {n: C.md5_file(C.P(f"{C.PROD_DIR_REL}/{n}", ROOT)) for n in C.PROD_MD5}
    out = {"current_source": C.CUR_TABLE_SRC_REL, "current_source_md5": C.md5_file(C.P(C.CUR_TABLE_SRC_REL, ROOT)),
           "n_aligned": sum(1 for r in res["rows"] if r.get("aligned")),
           "values": {m: {c: {"tex": v["tex"], "n": v["n"], "mean": v["mean"], "ci95_half": v["ci95_half"]}
                          for c, v in cols.items()} for m, cols in tv.items()},
           "manuscript_values": C.CURRENT_TABLE_EN, "reproduced": repro,
           "eor_l279_en_mcd": {"manuscript": C.CURRENT_EOR_L279_EN_MCD, "recomputed": C.eor_tex(tv), "ok": eor_ok},
           "ci_method": C.CI_METHOD,
           "production_0520_md5_ok": md5 == C.PROD_MD5,
           "production_0520_n_aligned": sum(1 for r in prod["rows"] if r.get("aligned"))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    assert all(repro.values()) and eor_ok and out["production_0520_md5_ok"], "今の表が今の式で再現できない"
    print("[spanfix-collect] current table reproduced from the current source with the current formula: OK")


def collect_rows(data_root, preview, fake=False):
    """seed の決まりで使う記録を決め、wav を確かめ直して行を集める（fake: 試運転の偽の記録。出どころは見ない）."""
    runs, warns = C.all_runs(data_root)
    pairs = C.load_pairs(ROOT)
    prod = C.read_json(C.P(C.PROD_DIR_REL + "/results.json", ROOT))
    ref_keys = set(next(r for r in prod["rows"] if r.get("aligned")))
    acts = {i: C.next_action(runs, i, ROOT, check_provenance=not fake) for i in range(C.N_PAIRS)}
    by_rel = {r["rel"]: r for s in C.SEEDS for r in runs[s]}
    rows, sel, problems = [], {}, {}
    for i in range(C.N_PAIRS):
        a = acts[i]
        if not a["action"].startswith("use_seed") or a["errors"]:
            problems[i] = {"action": a["action"], "errors": a["errors"][:3], "note": a.get("note")}
            continue
        run = by_rel[a["use"]["run"]]
        row, cap = run["rows"][i], run["captures"].get(i) or {}
        errs = []
        for k in ("text_a", "text_b", "word_a", "word_b"):
            if row.get(k) != pairs[i][k]:
                errs.append(f"{k} が違う")
        if set(row) != ref_keys:
            errs.append("行のキーが本番の行と違う")
        wavs = cap.get("wavs") or {}
        if sorted(wavs) != sorted(C.CONDS):
            errs.append(f"wav が 5 本でない: {sorted(wavs)}")
        for c, w in wavs.items():
            p = os.path.join(data_root, w.get("path", ""))
            if not w.get("path", "").startswith(run["rel"] + "/wav/") or not os.path.isfile(p):
                errs.append(f"wav {c} が置き場に無い")
            elif C.md5_file(p) != w.get("md5"):
                errs.append(f"wav {c} の md5 が記録と違う")
            else:
                ass = SC.assess_file(p)
                if ass["silent"] or w.get("silent") is not False:
                    errs.append(f"wav {c} が無音（または記録と違う）")
        if errs:
            problems[i] = {"action": a["action"], "errors": errs}
            continue
        rows.append(row)
        sel[i] = {"seed": a["use"]["seed"], "run": run["rel"], "chain": a["chain"],
                  "wav_md5": {c: w["md5"] for c, w in wavs.items()}, "position": C.name_position(pairs[i])}
    if problems and not preview:
        raise SystemExit("[spanfix-collect] 300 対そろっていない（または確かめに通らない）ので集計しない:\n"
                         + json.dumps({str(k): v for k, v in list(problems.items())[:20]}, ensure_ascii=False,
                                      indent=1) + f"\n  （ほか {max(0, len(problems) - 20)} 対）")
    return rows, sel, problems, warns, prod, pairs


def repro_vs_0520(rows, sel, prod, pairs):
    """参考（選ぶのには使わない）: 範囲の変わらない文頭の対で seed 42 の行が 5/20 の行を再現するか。BL の frames は全対."""
    prod_rows = {r["pair_idx"]: r for r in prod["rows"]}
    keys = [v for cols in C.TABLE_KEYS.values() for v in cols.values()]
    out = {"note": "reference only; never used to choose records"}
    for pos in ("sentence_initial", "mid_sentence"):
        diffs = {k: [] for k in keys}
        same_frames = n = 0
        for r in rows:
            i = r["pair_idx"]
            if sel[i]["seed"] != 42 or sel[i]["position"] != pos or not prod_rows[i].get("aligned"):
                continue
            n += 1
            same_frames += all(r.get(f"frames_{c}") == prod_rows[i].get(f"frames_{c}") for c in C.CONDS)
            for k in keys:
                if r.get(k) is not None and prod_rows[i].get(k) is not None:
                    diffs[k].append(abs(r[k] - prod_rows[i][k]))
        out[pos] = {"n_seed42_and_0520_aligned": n, "all5_frames_equal": same_frames,
                    "max_abs_diff": {k: (max(v) if v else None) for k, v in diffs.items()}}
    bl_same = sum(1 for r in rows if sel[r["pair_idx"]]["seed"] == 42 and prod_rows[r["pair_idx"]].get("aligned")
                  and r.get("frames_bl") == prod_rows[r["pair_idx"]].get("frames_bl"))
    out["frames_bl_equal_all_positions"] = bl_same
    return out


def by_position(rows, pairs):
    out = {}
    for pos in ("sentence_initial", "mid_sentence"):
        rr = [r for r in rows if C.name_position(pairs[r["pair_idx"]]) == pos]
        if rr:
            tv = C.table_values(rr)
            out[pos] = {m: {c: v["tex"] + f" (n={v['n']})" for c, v in cols.items()} for m, cols in tv.items()}
    return out


def print_old_new(tv_new, tv_old, n_rows, label):
    print(f"\n=== {label}: Appendix B tab:crossattn_mask, EN rows (naacl2027/main.tex 894-895) ===")
    print(f"  {'cell':28s} {'old (manuscript)':>18s}   {'new (spanfix)':>18s}   n")
    for m, cols in C.TABLE_KEYS.items():
        for k, col in enumerate(("intact", "blocked")):
            old = C.CURRENT_TABLE_EN[m][k]
            new = tv_new[m][col]["tex"]
            print(f"  {m + ' / ' + col:28s} {old:>18s}   {new:>18s}   {tv_new[m][col]['n']}")
    print(f"  (old values recomputed from {C.CUR_TABLE_SRC_REL}: "
          + ", ".join(f"{m}/{c}={tv_old[m][c]['tex']}" for m in C.TABLE_KEYS for c in ("intact", "blocked")) + ")")
    print(f"=== explanation_of_revisions.tex L279 (EN entity-span MCD) ===")
    print(f"  old {C.CURRENT_EOR_L279_EN_MCD}  ->  new {C.eor_tex(tv_new)}   (rows used: {n_rows})")
    print(f"  CI: {C.CI_METHOD}")


def aggregate(data_root=ROOT, preview=False, fake=False):
    _, tv_old, repro, eor_ok = current_table()
    assert all(repro.values()) and eor_ok, "今の表が今の式で再現できない（式の確認が先）"
    if fake:  # 試運転の偽の記録は、試運転の置き場から、表示だけ（書かない）
        assert preview and os.path.realpath(data_root).startswith(os.path.realpath(C.P(C.DRYRUN_REL, ROOT)) + os.sep)
    rows, sel, problems, warns, prod, pairs = collect_rows(data_root, preview, fake)
    if not rows:
        print(f"[spanfix-collect] 有効な対がまだ無い（{len(problems)} 対が未決着）")
        return
    tv = C.table_values(rows)
    rep = repro_vs_0520(rows, sel, prod, pairs)
    pos = by_position(rows, pairs)
    label = f"PREVIEW (valid pairs so far: {len(rows)}/300, NOT FINAL)" if preview else "FINAL N=300"
    print_old_new(tv, tv_old, len(rows), label)
    print("\n  by name position (reference):", json.dumps(pos, ensure_ascii=False))
    print("  reproduction vs 5/20 (reference):", json.dumps(rep, ensure_ascii=False))
    seeds_used = {s: sum(1 for v in sel.values() if v["seed"] == s) for s in C.SEEDS}
    print(f"  seeds used: {seeds_used}")
    if preview:
        print(f"  not settled yet: {len(problems)} pairs")
        return
    assert len(rows) == C.N_PAIRS and all(v["n"] == C.N_PAIRS for cols in tv.values() for v in cols.values()), \
        "N が 300 でない"
    rows = sorted(rows, key=lambda r: r["pair_idx"])
    out_dir = os.path.join(data_root, C.OUT_REL, f"final_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=False)
    C.write_json_new({"summary": summarize(rows), "rows": rows}, os.path.join(out_dir, "results.json"), indent=2)
    tvj = {"lang": "en", "n": C.N_PAIRS, "table_keys": C.TABLE_KEYS, "ci_method": C.CI_METHOD,
           "new": tv, "old_recomputed_from_current_source": tv_old, "old_in_manuscript": C.CURRENT_TABLE_EN,
           "eor_l279_en_mcd": {"old": C.CURRENT_EOR_L279_EN_MCD, "new": C.eor_tex(tv)},
           "by_name_position": pos, "reproduction_vs_0520_reference": rep, "seeds_used": seeds_used,
           "silence_check_scope": "all 300 pairs: the 5 wavs of the used record were re-hashed and re-checked "
                                  "from the files with " + SC.CRITERION_TEXT}
    C.write_json_new(tvj, os.path.join(out_dir, "table_values.json"))
    C.write_json_new({"date": C.now(), "selection": {str(k): v for k, v in sorted(sel.items())},
                      "record_warnings": warns, "code_md5_now": C.current_code_md5(ROOT),
                      "seed_policy": "seed 42 if valid; else seed 43 if valid; else seed 44 if valid "
                                     "(chosen only by validity; numbers never used)"},
                     os.path.join(out_dir, "selection_record.json"))
    print(f"[spanfix-collect] written: {os.path.relpath(out_dir, data_root)}/"
          "{results.json, table_values.json, selection_record.json}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true")
    g.add_argument("--check_current", action="store_true")
    g.add_argument("--preview", action="store_true")
    g.add_argument("--aggregate", action="store_true")
    ap.add_argument("--save", action="store_true", help="--status の結果を新しいファイルにも残す")
    ap.add_argument("--fake_root", default=None,
                    help="--preview だけ: 試運転（spanfix_appB_en_gen.py --dry_run）の fake_root で集計の配管を確かめる")
    args = ap.parse_args()
    assert not C.IN_CONTAINER, "ホストで流す（CPU のみ）"
    if args.fake_root:
        assert args.preview, "--fake_root は --preview とだけ"
        aggregate(data_root=os.path.abspath(args.fake_root), preview=True, fake=True)
    elif args.status:
        status(save=args.save)
    elif args.check_current:
        check_current()
    elif args.preview:
        aggregate(preview=True)
    else:
        aggregate(preview=False)


if __name__ == "__main__":
    main()
