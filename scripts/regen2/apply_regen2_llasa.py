#!/usr/bin/env python3
"""G5a: regen_llasa_pairs_seed.py で作り直した Llasa-3B の対を本番に差し替える（CPU のみ）.

1. 事前確認（何も書かない。全セル分を先に済ませ、1 つでも問題があれば何も書かない）:
   _regen2_final/llasa/<cell>/regen_meta.json、採用した seed が 43 か 44（44 は 43 が使えなかった対だけ）、
   採用した音声が使えること（regen2_common.wav_check。無音・欠け・読めない）、採用した行が全層で
   status == "ok"、seed 42 の再現確認（一致しないときは --accept_check_mismatch が要る）、unresolved の対は
   差し替えない、本番の progress.json と manifest に対象の対の行があること、退避先と記録
   （regen2_seed43_pairs.json）がまだ無いこと、書き込み先が凍結した T5Gemma EN (a) や原稿でないこと。
   書き込む progress.json と manifest（kappa / gen_length の再計算を含む）は、ここでメモリの上に作っておく。
2. 退避: 本番の wav（対象の対）、meta（progress.json と manifest）、採点（PIB・MCD の JSON）を
   eval_results/_regen2_seed43/original_seed42_backup/llasa/<cell>/ に写す（既にあれば止める）。
3. 差し替え: 作り直した wav を本番のファイルに上書きで書く（削除も rename もしない）。
   掃引の BL/TGT は 5 層で同じ inode（hard link）なので、1 回上書きすれば 5 層とも新しい中身になる。
   書いた後に 5 層すべての md5 が新しい音声と一致し、nlink が変わっていないことを確かめる。
4. 1 で作った progress.json と manifest を書く（行に "regen2" を足し、manifest の meta に regen2 の記録を足す）。
5. 本番の out_dir に regen2_seed43_pairs.json（差し替えの記録）を書く。
採点（v1 PIB は GPU、whole MCD はホストの CPU）の取り込みは merge_regen2_scores.py で行う。

使い方（ホストで直接。スレッド 13・CPU 0-12・nice 10・GPU を見せない・HF オフラインはスクリプトの中で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/apply_regen2_llasa.py            # 予行（既定）
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/apply_regen2_llasa.py --apply    # 実行
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import regen2_common as C  # noqa: E402

C.host_cpu_setup()

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

KINDS = C.LLASA_KINDS
ALLOWED_SEEDS = (43, 44)
STALE_DERIVED = {
    "en_a": ["_utmos_table3/main__Llasa-3B_EN.json", "_v3_phperPER_llasa_en.json"],
    "zh_a": ["_utmos_table3/main__Llasa-3B_ZH.json", "_v3_phperPER_llasa_zh.json"],
    "en_sweep": ["_utmos_table3/sweep__Llasa-3B_EN_l%d.json" % (L + 1) for L in C.SWEEP_LAYERS],
    "zh_sweep": ["_utmos_table3/sweep__Llasa-3B_ZH_l%d.json" % (L + 1) for L in C.SWEEP_LAYERS]
                + ["_v1L_llasa_zh_L%02d.json (n=297, 旧)" % L for L in C.SWEEP_LAYERS],
}


def _R():
    """run_llasa_eval の集計関数（torch を読むので必要なときだけ）."""
    sys.path.insert(0, os.path.join(C.PROJECT_ROOT, "scripts", "llasa"))
    import run_llasa_eval as R
    return R


def manifest_path(cell, sub):
    d = os.path.join(C.EV, C.LLASA_CELLS[cell]["prod_out"], sub)
    ms = sorted(f for f in os.listdir(d) if f.endswith("_manifest.json") and "seed" not in f)
    if len(ms) != 1:
        raise SystemExit("manifest が 1 つでない: %s %s" % (d, ms))
    return os.path.join(d, ms[0])


def recompute_manifest(man):
    R = _R()
    ok_rows = [r for r in man["rows"] if r.get("status") == "ok"]
    k = R.summarize_kappa(ok_rows, man["meta"]["num_layers"])
    g = R.summarize_gen_length(ok_rows, man["meta"]["decode"]["max_new_tokens"])
    return k, g


def selftest_manifests(cells):
    """既存の manifest の kappa / gen_length を同じ関数で再計算し、1 桁も違わないかを確かめる."""
    out = {}
    for cell in cells:
        for layer, sub in C.llasa_layer_dirs(cell):
            man = C.read_json(manifest_path(cell, sub))
            k, g = recompute_manifest(man)
            out["%s/L%02d" % (cell, layer)] = {"kappa_same": k == man["kappa"], "gen_length_same": g == man["gen_length"]}
    return out


def check_seed_choice(cell, meta, problems):
    """採用した seed が 43 か 44 で、44 は seed 43 で使えなかった対だけか（結果で選んでいないこと）."""
    valid = meta.get("valid", {})
    for p, s in meta["chosen_seed"].items():
        if s is None:
            continue
        if s not in ALLOWED_SEEDS:
            problems.append("%s 対 %s: 採用した seed %s が 43/44 でない" % (cell, p, s))
        v = valid.get(str(p), {})
        if s == 44 and v.get("43") is not False:
            problems.append("%s 対 %s: seed 43 が使えないと記録されていないのに seed 44 を採用している" % (cell, p))
        if v.get(str(s)) is not True:
            problems.append("%s 対 %s: 採用した seed %s が使えると記録されていない" % (cell, p, s))


def plan_cell(cell, args, problems):
    c = C.LLASA_CELLS[cell]
    subs = C.llasa_layer_dirs(cell)
    first = subs[0][1]
    fin = os.path.join(C.llasa_final_root(), cell)
    meta_p = os.path.join(fin, "regen_meta.json")
    plan = {"cell": cell, "final_meta": meta_p, "wav_ops": [], "backup_files": [], "pairs": {}, "writes": []}
    bak = os.path.join(C.llasa_backup_root(), cell)
    plan["backup_dir"] = bak
    if not os.path.exists(meta_p):
        # 予行では、全対が seed 43 で通ったと仮定して書き込み先・hard link・退避先を点検する
        plan["status"] = "final not generated yet (planned as if every pair were valid at seed 43)"
        chosen = {p: (None if args.apply else 43) for p in c["pairs"]}
        meta = None
    else:
        meta = C.read_json(meta_p)
        chosen = {int(p): s for p, s in meta["chosen_seed"].items()}
        if sorted(chosen) != sorted(c["pairs"]):
            problems.append("%s: regen_meta の対 %s が計画 %s と違う" % (cell, sorted(chosen), c["pairs"]))
        check_seed_choice(cell, meta, problems)
        chk = meta.get("check_seed42")
        plan["check_seed42_match"] = chk.get("match_all") if chk else None
        if chk is None and not args.accept_check_mismatch:
            problems.append("%s: seed 42 の再現確認が記録されていない（--no_check で作った。"
                            "--accept_check_mismatch が要る）" % cell)
        elif not chk.get("match_all") and not args.accept_check_mismatch:
            problems.append("%s: seed 42 の再現確認で md5 が一致しない（--accept_check_mismatch が要る）" % cell)
        for p in meta.get("unresolved_pairs", []):
            problems.append("%s: 対 %d は seed 43/44 とも使えない（差し替えない。著者判断待ち）" % (cell, p))
    for pidx in c["pairs"]:
        seed = chosen.get(pidx)
        plan["pairs"][str(pidx)] = {"chosen_seed": seed}
        if seed is None:
            continue
        # BL / TGT（掃引は最初の層。5 層で同じ inode か確かめる）
        for k in ("bl", "tgt"):
            n = "s%04d_%s.wav" % (pidx, k)
            prod_first = os.path.join(C.EV, c["prod_wav"], first, n)
            group = [os.path.join(C.EV, c["prod_wav"], sub, n) for _, sub in subs]
            same = all(os.path.samefile(prod_first, g) for g in group)
            src = os.path.join(fin, "wav", first, n)
            targets = [prod_first] if same else group
            for t in targets:
                plan["wav_ops"].append({"src": src, "dst": t, "linked_group": group if same else [t],
                                        "nlink_before": os.stat(t).st_nlink})
        for layer, sub in subs:
            n = "s%04d_inj.wav" % pidx
            dst = os.path.join(C.EV, c["prod_wav"], sub, n)
            plan["wav_ops"].append({"src": os.path.join(fin, "wav", sub, n), "dst": dst, "linked_group": [dst],
                                    "nlink_before": os.stat(dst).st_nlink})
    for op in plan["wav_ops"]:
        C.assert_writable_target(op["dst"])
        # 上書きは同じ inode の全リンクに効く。リンクが対象の層の外にもあれば止める
        if op["nlink_before"] != len(op["linked_group"]):
            problems.append("hard link が対象の層の外にもある: %s (nlink=%d, group=%d)"
                            % (op["dst"], op["nlink_before"], len(op["linked_group"])))
        rel = os.path.relpath(op["dst"], os.path.join(C.EV, c["prod_wav"]))
        op["backup"] = os.path.join(bak, "wav", rel)
        plan["backup_files"].append((op["dst"], op["backup"]))
        if os.path.exists(op["backup"]):
            problems.append("退避先が既にある: %s" % op["backup"])
        if meta is not None:
            w = C.wav_check(op["src"])
            op["src_check"] = {k: w.get(k) for k in ("rms", "peak", "active_frac", "silent_reason", "why")}
            if w["bad"]:
                problems.append("作り直しの音声が使えない（%s）: %s" % (w.get("why"), op["src"]))
    for layer, sub in subs:
        for p in (os.path.join(C.EV, c["prod_out"], sub, "progress.json"), manifest_path(cell, sub)):
            C.assert_writable_target(p)
            b = os.path.join(bak, "meta", sub, os.path.basename(p))
            plan["backup_files"].append((p, b))
            if os.path.exists(b):
                problems.append("退避先が既にある: %s" % b)
    for f in list(c["pib_json"].values()) + list(c["mcd_json"].values()):
        p = os.path.join(C.EV, f)
        if os.path.exists(p):
            b = os.path.join(bak, "scores", f)
            plan["backup_files"].append((p, b))
            if os.path.exists(b):
                problems.append("退避先が既にある: %s" % b)
        else:
            problems.append("採点ファイルが無い: %s" % p)
    if c["kind"] == "sweep":
        p = os.path.join(C.EV, c["prod_out"], "layersweep_summary.json")
        if os.path.exists(p):
            b = os.path.join(bak, "meta", "layersweep_summary.json")
            plan["backup_files"].append((p, b))
            if os.path.exists(b):
                problems.append("退避先が既にある: %s" % b)
    rec = os.path.join(C.EV, c["prod_out"], "regen2_seed43_pairs.json")
    plan["record"] = C.assert_writable_target(rec)
    if os.path.exists(rec):
        problems.append("差し替えの記録が既にある（差し替え済み）: %s" % rec)
    plan["meta"] = meta
    if meta is not None:
        prepare_docs(plan, problems)
    return plan


def prepare_docs(plan, problems):
    """書き込む progress.json / manifest をメモリの上で作る。行が無い・status が ok でないときは problems に足す."""
    cell = plan["cell"]
    c = C.LLASA_CELLS[cell]
    meta = plan["meta"]
    chosen = {int(p): s for p, s in meta["chosen_seed"].items() if s is not None}
    plan["summaries"] = {}
    for layer, sub in C.llasa_layer_dirs(cell):
        key = "L%02d" % layer
        new_rows = {}
        for p, s in chosen.items():
            r = ((meta.get("final") or {}).get(str(p)) or {}).get("rows", {}).get(key)
            if r is None:
                problems.append("%s 対 %d: 作り直しの %s の行が regen_meta に無い" % (cell, p, key))
                continue
            if r.get("status") != "ok" or r.get("pair_idx") != p:
                problems.append("%s 対 %d: 作り直しの %s の行が ok でない（status=%s err=%s）"
                                % (cell, p, key, r.get("status"), str(r.get("err"))[:120]))
                continue
            r = dict(r)
            r["regen2"] = {"seed": s, "date_string": c["date_string"],
                           "reason": "unusable pair at seed 42 (G5a; %s)" % C.SILENCE_RULE,
                           "source": "eval_results/_regen2_seed%d/llasa/%s/meta/%s" % (s, cell, sub)}
            new_rows[p] = r
        prog_p = os.path.join(C.EV, c["prod_out"], sub, "progress.json")
        prog = C.read_json(prog_p)
        idx = {r["pair_idx"]: j for j, r in enumerate(prog["rows"])}
        man_p = manifest_path(cell, sub)
        man = C.read_json(man_p)
        midx = {r["pair_idx"]: j for j, r in enumerate(man["rows"])}
        for p in new_rows:
            if p not in idx:
                problems.append("%s: 本番の %s に対 %d の行が無い" % (cell, prog_p, p))
            if p not in midx:
                problems.append("%s: 本番の %s に対 %d の行が無い" % (cell, man_p, p))
        if any(p not in idx or p not in midx for p in new_rows):
            continue
        prog = copy.deepcopy(prog)
        for p, r in new_rows.items():
            prog["rows"][idx[p]] = r
        man = copy.deepcopy(man)
        for p, r in new_rows.items():
            man["rows"][midx[p]] = r
        before = {"kappa": man["kappa"], "gen_length": man["gen_length"]}
        man["kappa"], man["gen_length"] = recompute_manifest(man)
        man["meta"]["regen2_seed43_pairs"] = {"pairs": sorted(new_rows),
                                              "seeds": {str(p): chosen[p] for p in new_rows},
                                              "record": "regen2_seed43_pairs.json"}
        n_ok = sum(1 for r in man["rows"] if r.get("status") == "ok")
        if n_ok != 300 or len(man["rows"]) != 300:
            problems.append("%s %s: 差し替え後の manifest の ok の行が %d / %d（300 でない）"
                            % (cell, key, n_ok, len(man["rows"])))
        plan["writes"] += [(C.assert_writable_target(prog_p), prog), (C.assert_writable_target(man_p), man)]
        plan["summaries"][key] = {"before": before, "after": {"kappa": man["kappa"], "gen_length": man["gen_length"]}}


def do_cell(plan):
    """事前確認を通った plan だけを実行する（ここでは書くだけ。確認は plan_cell / prepare_docs で済ませてある）."""
    cell = plan["cell"]
    c = C.LLASA_CELLS[cell]
    meta = plan["meta"]
    # 2. 退避
    for src, dst in plan["backup_files"]:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            raise SystemExit("退避先が既にある: %s" % dst)
        shutil.copy2(src, dst)
    # 3. 差し替え（上書き。hard link はそのまま）
    files = []
    for op in plan["wav_ops"]:
        old_md5 = C.md5_file(op["backup"])
        shutil.copyfile(op["src"], op["dst"])
        new_md5 = C.md5_file(op["src"])
        for g in op["linked_group"]:
            if C.md5_file(g) != new_md5:
                raise SystemExit("差し替え後の md5 が合わない: %s" % g)
        if os.stat(op["dst"]).st_nlink != op["nlink_before"]:
            raise SystemExit("nlink が変わった: %s" % op["dst"])
        files.append({"file": os.path.relpath(op["dst"], C.EV), "linked_group": [os.path.relpath(g, C.EV)
                                                                               for g in op["linked_group"]],
                      "old_md5_seed42": old_md5, "new_md5": new_md5, "nlink": op["nlink_before"]})
    # 4. progress.json / manifest（事前に作った中身）
    for path, doc in plan["writes"]:
        C.write_json(doc, path, indent=2)
    # 5. 記録
    chosen = {int(p): s for p, s in meta["chosen_seed"].items() if s is not None}
    rec = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "item": "G5a", "cell": cell, "table3": c["table3"],
           "pairs": sorted(chosen), "chosen_seed": {str(p): s for p, s in chosen.items()},
           "reason": "unusable pair at seed 42 (silent wav; %s). Author decision 2026-09-28 extended to "
                     "Table 3 (G5a): regenerate the whole pair with seed 43 (44 if still unusable: silent, "
                     "missing or unreadable wav, or a generation row whose status is not ok)" % C.SILENCE_RULE,
           "date_string_pinned": c["date_string"], "span_extension": c["span_extension"],
           "valid": meta.get("valid"), "check_seed42": meta.get("check_seed42"),
           "row_compare_vs_prod": meta.get("row_compare_vs_prod"),
           "final_dir": os.path.join(C.llasa_final_root(), cell), "backup": plan["backup_dir"], "files": files,
           "manifest_summaries": plan["summaries"],
           "scores_to_update": {"pib": c["pib_json"], "mcd": c["mcd_json"]},
           "stale_derived_not_updated": STALE_DERIVED.get(cell, [])}
    if os.path.exists(plan["record"]):
        raise SystemExit("記録が既にある: %s" % plan["record"])
    C.write_json(rec, plan["record"])
    print("[apply] %s: %d files replaced, record -> %s" % (cell, len(files), plan["record"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(C.LLASA_CELL_ORDER))
    ap.add_argument("--apply", action="store_true", help="付けないときは予行（何も書かない）")
    ap.add_argument("--accept_check_mismatch", action="store_true",
                    help="seed 42 の再現確認で md5 が一致しなくても差し替える（GPU の違いなど。著者の了承が要る）")
    ap.add_argument("--skip_selftest", action="store_true")
    args = ap.parse_args()
    C.refuse_apply_in_test(args.apply)
    cells = [c for c in C.LLASA_CELL_ORDER if c in args.cells.split(",")]
    problems = []
    plans = [plan_cell(cell, args, problems) for cell in cells]
    rep = {"mode": "apply" if args.apply else "dry_run", "problems": problems,
           "criterion": C.SILENCE_CRITERION,
           "frozen_t5gemma_en_a_before": C.frozen_digest(),
           "plans": [dict({k: v for k, v in p.items() if k not in ("meta", "writes")},
                          writes=[w[0] for w in p["writes"]]) for p in plans]}
    if not args.skip_selftest:
        rep["selftest_manifest_recompute"] = st = selftest_manifests(cells)
        bad = [k for k, v in st.items() if not all(v.values())]
        if bad:
            problems.append("manifest の kappa/gen_length の再計算が既存と一致しない: %s" % bad)
    not_ready = [p["cell"] for p in plans if p["meta"] is None]
    if args.apply and not_ready:
        problems.append("作り直しがまだ無いセル: %s" % not_ready)
    print(json.dumps({"mode": rep["mode"], "problems": problems,
                      "selftest": rep.get("selftest_manifest_recompute"),
                      "n_wav_ops": {p["cell"]: len(p["wav_ops"]) for p in plans},
                      "n_json_writes": {p["cell"]: len(p["writes"]) for p in plans},
                      "not_ready": not_ready}, ensure_ascii=False, indent=1))
    out = os.path.join(C.DRYRUN_DIR, "apply_llasa_%s.json" % rep["mode"])
    C.write_json(rep, out)
    print("[plan] ->", out)
    if not args.apply:
        return
    if problems:
        raise SystemExit("問題があるので差し替えない（何も書いていない）")
    for p in plans:
        do_cell(p)
    after = C.frozen_digest()
    if after != rep["frozen_t5gemma_en_a_before"]:
        raise SystemExit("凍結した T5Gemma EN (a) が変わった（ありえない）")
    print("[apply] 完了。凍結した T5Gemma EN (a) は不変:", after)


if __name__ == "__main__":
    main()
