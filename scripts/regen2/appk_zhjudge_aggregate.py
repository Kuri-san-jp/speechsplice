#!/usr/bin/env python3
"""付録 K 中国語の注入成功率の判定し直し: 集計（2026-09-29、CPU のみ）.

順番（前の段で止まったら、新しい値は計算も表示もしない。結果を見てから条件を選ばないため）:
  1. 元の集計コード（.perm_cmds.txt の 1 行、ORIGINAL_CODE）が記録と一致することを md5 で確かめる。
  2. 7/28 の値の再現: 本番の ctc_lr_judge.json（md5 cb2dc8be…）に元の集計コードをそのまま流し、
     281/300 = 93.67 が llr_shift_summary.json（md5 6c13022e…）と一致することを確かめる
     （出力は stage の llr_shift_summary_stored_recheck.json。本番のファイルには書かない）。
  3. stage の確認: 600 本と checkpoint_zh.json が manifest の md5 のまま、GPU の実行記録が done で、
     判定器・gold・snapshot の md5 が 7/28 と同じ、出力の md5 が記録と一致、今の本番の音声が stage と同じ。
  4. 7/28 の判定の再現（作り直していない 295 対）と N=300 の確認:
       * 新しい行が 300、pair_idx 0..299、skip 0、degenerate 0（判定器の summary も同じ）
       * 300 対すべてで字・src・tgt・degenerate・tone_only が 7/28 と同じ
       * 295 対で主判定（d > 0 か）が 1 対も変わっていない  ← 変わったら必ず止まる
       * 295 対で尤度の値の差が --value_tol（既定 0.01）以内 ← 超えたら止まる（--accept_value_drift で著者が
         認めたときだけ続ける。その場合も記録に残す）
       * 副判定（*_choose_tgt*）の変化は、7/28 の余白が雑音の幅以内のときだけ許す
  5. 新しい値: stage の ctc_lr_judge.json に同じ元の集計コードを流す（出力 llr_shift_summary_rejudge.json、
     新しい名前）。別に書いた同じ式で検算し、7/28 の値と並べて Wilson 95% の半幅と区間を出す。
     作り直した 5 対の前後の d と成否も出す。
  6. 報告 <stage>/zh_injection_success_rejudge.json（前の版があれば .bak_<時刻> に写す）。

本番の llr_shift_summary.json・ctc_lr_judge.json・ctc_lr_v3_partial.json と原稿は書き換えない。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_zhjudge_aggregate.py --stage_dir eval_results/_appk_zhjudge/rejudge_ctc
終了コード: 0 = 新しい値を出した、1 = 止まった（理由は表示と報告の reasons）。
"""
import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_zhjudge_common as C  # noqa: E402

K = C.K
K.cap_threads()

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402


def stored_side(stage_dir):
    """手順 1–2: 元の集計コードの照合と 7/28 の値の再現."""
    out = {"problems": []}
    prov = C.verify_original_code()
    out["original_code"] = prov
    if not prov["ok"]:
        out["problems"].append("元の集計コードが記録（.perm_cmds.txt）と一致しない")
        return out
    for tag, path in (("stored_judge", C.STORED_JUDGE), ("stored_summary", C.STORED_SUMMARY)):
        ok, m = C.check_pinned(tag, path)
        out[f"{tag}_md5"] = m
        if not ok:
            out["problems"].append(f"{C.proj_rel(path)} の md5 {m} が 7/28 の {C.EXPECTED_MD5[tag]} でない")
    if out["problems"]:
        return out
    txt, summ = C.run_original_code(C.STORED_JUDGE, os.path.join(stage_dir, C.SUMMARY_STORED_RECHECK))
    stored = C.read_json(C.STORED_SUMMARY)
    rows = C.read_json(C.STORED_JUDGE)["rows"]
    k, n, _ = C.shift_count(rows)
    out.update(recheck_file=C.proj_rel(os.path.join(stage_dir, C.SUMMARY_STORED_RECHECK)), recheck=summ,
               stored_summary=stored, printed=txt, independent_count=[k, n], rows=rows)
    if not (summ == stored == C.STORED_SUMMARY_CONTENT and k == stored["shift_tone_pos"] and n == C.N):
        out["problems"].append(f"7/28 の値が再現しない: 元のコード {summ}、記録 {stored}、検算 {k}/{n}")
    return out


def load_stage(stage_dir, allow_selftest=False):
    """手順 3: stage・GPU の実行記録・今の本番との照合."""
    st = {"problems": [], "stage_dir": C.proj_rel(stage_dir)}
    mp = os.path.join(stage_dir, C.MANIFEST)
    if not os.path.isfile(mp):
        st["problems"].append(f"stage がない: {C.proj_rel(mp)}")
        return st
    man = C.read_json(mp)
    st.update(manifest=man, manifest_md5=C.md5_file(mp), mode=man.get("mode"))
    if (man.get("selftest") or man.get("allow_silent")) and not allow_selftest:
        st["problems"].append("自己検査の stage（または無音を許した stage）は集計しない")
    if man.get("mode") not in ("post_apply", "pre_apply"):
        st["problems"].append(f"stage の mode={man.get('mode')}")
    ch, un = man.get("changed_pairs", []), man.get("unchanged_pairs", [])
    if sorted(ch + un) != list(range(C.N)) or set(ch) & set(un):
        st["problems"].append("manifest の差し替えた対と差し替えていない対を合わせて 0..299 にならない")
    bad = C.verify_stage_files(stage_dir, man)
    if bad:
        st["problems"].append(f"stage のファイルが manifest と違う（{len(bad)} 件、例 {bad[:2]}）")
    jp = os.path.join(stage_dir, C.JUDGE_OUT)
    if not os.path.isfile(jp):
        st["problems"].append(f"判定器の出力がない: {C.proj_rel(jp)}（GPU の判定がまだ）")
        return st
    st["judge_output"] = jp
    st["judge_output_md5"] = C.md5_file(jp)
    metas = []
    for p in sorted(glob.glob(os.path.join(stage_dir, C.RUN_META_PREFIX + "*.json"))):
        m = C.read_json(p)
        if m.get("status") == "done":
            metas.append((p, m))
    if not metas:
        st["problems"].append("GPU の実行記録（status=done）がない")
        return st
    mpath, meta = metas[-1]
    st["run_meta"] = C.proj_rel(mpath)
    env = meta.get("environment", {})
    st["gpu_run"] = {k: meta.get(k) for k in ("started", "finished", "elapsed_min", "judge_argv", "judge_output_md5",
                                              "selftest")}
    st["gpu_run"]["environment"] = env
    st["gpu_run"]["text_side"] = (meta.get("preflight") or {}).get("text_side")
    checks = {
        "judge_md5": meta.get("judge_md5") == C.EXPECTED_MD5["judge"],
        "gold_md5": meta.get("gold_md5") == C.EXPECTED_MD5["gold_v2"],
        "snapshot": meta.get("snapshot") == C.SNAP_ID,
        "manifest_md5": meta.get("stage_manifest_md5") == st["manifest_md5"],
        "output_md5": meta.get("judge_output_md5") == st["judge_output_md5"],
        "n_rows_300": meta.get("n_rows") == C.N,
        "stage_unchanged_during_run": meta.get("stage_files_changed_during_run") == [],
        "pyopenjtalk_blocked": env.get("pyopenjtalk_blocked") is True,
        "ran_on_cuda": bool(env.get("cuda_available")) or bool(meta.get("selftest") and allow_selftest),
        "in_container": bool(env.get("in_container")) or bool(meta.get("selftest") and allow_selftest),
    }
    st["run_checks"] = checks
    for k, v in checks.items():
        if not v:
            st["problems"].append(f"GPU の実行記録の確認 {k} が合わない（{C.proj_rel(mpath)}）")
    # 今の本番が stage と同じか（反映のあとの本番を採点したことの確認）
    prod_diff = []
    for n, rec in man["files"].items():
        p = os.path.join(C.PROD_DIR, n)
        if (C.md5_file(p) if os.path.isfile(p) else None) != rec["md5"]:
            prod_diff.append(n)
    prod_rows_digest = C.md5_json(C.read_json(C.PROD_CKPT)["results"])
    st["production_now_equals_stage"] = {"wav_diff": prod_diff,
                                         "checkpoint_rows_equal": prod_rows_digest == man["checkpoint"]["rows_digest"]}
    same = not prod_diff and prod_rows_digest == man["checkpoint"]["rows_digest"]
    st["production_now_equals_stage"]["equal"] = same
    if not same and man.get("mode") == "post_apply" and not allow_selftest:
        st["problems"].append(f"stage を作ったあとで本番の音声か checkpoint が変わった（wav {len(prod_diff)} 本）。"
                              f"この判定は今の本番を表さない")
    return st


def evaluate(new_doc, stored_rows, man, value_tol=0.01, accept_value_drift=False):
    """手順 4: N=300 の形と、作り直していない 295 対の再現（新しい k は数えない）."""
    rows = new_doc.get("rows", [])
    shape = C.check_shape(rows, new_doc.get("summary"))
    new_by = {r.get("pair_idx"): r for r in rows}
    old_by = {r["pair_idx"]: r for r in stored_rows}
    cmp = C.compare_rows(new_by, old_by, list(range(C.N)), man["unchanged_pairs"], value_tol, accept_value_drift)
    reasons = list(shape)
    if cmp["missing"]:
        reasons.append(f"行がない対: {cmp['missing'][:10]}")
    if cmp["meta_mismatch"]:
        reasons.append(f"字・src・tgt などが 7/28 と違う対: {[x['pair_idx'] for x in cmp['meta_mismatch']][:10]}")
    if cmp["keyset_mismatch"]:
        reasons.append(f"欄の組が 7/28 と違う対: {[x['pair_idx'] for x in cmp['keyset_mismatch']][:10]}")
    if cmp["primary_flips"]:
        reasons.append(f"作り直していない対で主判定（d>0）が変わった: {cmp['primary_flips'][:10]}"
                       f"（環境が 7/28 と違う疑い。著者の判断まで止める）")
    if cmp["secondary_flips_beyond_noise"]:
        reasons.append(f"作り直していない対で副判定が余白を超えて変わった: {cmp['secondary_flips_beyond_noise'][:5]}")
    if not cmp["values_reproduced"] and not accept_value_drift:
        reasons.append(f"作り直していない対で尤度の値が {value_tol} を超えて違う（{len(cmp['values_over_tol'])} 件、"
                       f"最大 {cmp['max_abs_value_diff']}）。判定は同じでも環境の違いの疑い。著者が認めたら "
                       f"--accept_value_drift で続ける")
    return {"pass": not reasons, "reasons": reasons, "shape_problems": shape, "compare": cmp}


def _trim_compare(cmp, n=20):
    return {k: (v[:n] if isinstance(v, list) else v) for k, v in cmp.items()}


def changed_detail(new_by, old_by, man):
    seeds = (man.get("provenance", {}).get("apply_record") or {}).get("seeds") \
        or man.get("provenance", {}).get("seeds") or {}
    inv = man.get("provenance", {}).get("plan", {}).get("invalid_detail") or {}
    out = []
    for i in man["changed_pairs"]:
        o, n = old_by[i], new_by[i]
        do, dn = C.shift_of(o), C.shift_of(n)
        out.append({"pair_idx": i, "char": o["char"], "src": o["src"], "tgt": o["tgt"], "tone_only": o["tone_only"],
                    "silent_side_at_seed42": sorted(inv.get(str(i), {})), "new_seed": seeds.get(str(i)),
                    "new_wav_md5": {s: man["files"][C.wav_names(i)[s]]["md5"] for s in ("bl", "inj")},
                    "stored": {"d": round(do, 3), "success": do > 0,
                               **{f: o.get(f) for f in C.LP_FIELDS[:4]}},
                    "new": {"d": round(dn, 3), "success": dn > 0,
                            **{f: n.get(f) for f in C.LP_FIELDS[:4]}}})
    return out


def run(stage_dir, value_tol=0.01, accept_value_drift=False, out_path=None, allow_selftest=False):
    """手順 1–6。戻り値: (報告, 終了コード)."""
    stage_dir = os.path.abspath(stage_dir)
    out_path = out_path or os.path.join(stage_dir, C.REPORT)
    C.assert_write(out_path)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "status": None, "reasons": [],
           "what": ("Appendix K, Chinese injection success: the 7/28 judge (zh_ctc_lr_judge.py, whisper-pinyin CTC "
                    "log-likelihood, gold v2) and the 7/28 aggregation code, unchanged, re-run on the N=300 audio "
                    "after the 5 unusable pairs were regenerated (seed 43/44)"),
           "method": {"judge": C.JUDGE_REL, "judge_md5": C.EXPECTED_MD5["judge"], "gold": C.GOLD_V2_REL,
                      "gold_md5": C.EXPECTED_MD5["gold_v2"], "model": C.MODEL_ID, "snapshot": C.SNAP_ID,
                      "aggregation": ".perm_cmds.txt line (2026-07-28 18:14), embedded verbatim as ORIGINAL_CODE",
                      "success_rule": "d = (inj_lp_tgt - inj_lp_src) - (bl_lp_tgt - bl_lp_src) > 0, over all 300 pairs",
                      "interval": "Wilson 95% (z = 1.96); the paper prints pct ± half-width"},
           "gate_settings": {"value_tol": value_tol, "accept_value_drift": bool(accept_value_drift)},
           "not_written": [C.proj_rel(C.STORED_SUMMARY), C.proj_rel(C.STORED_JUDGE), C.proj_rel(C.STORED_PARTIAL),
                           "naacl2027/main.tex"]}

    def stop(reasons, code=1):
        rep["status"] = "stopped"
        rep["reasons"] += reasons
        rep["written_to"] = C.proj_rel(out_path)
        C.write_json(rep, out_path)
        return rep, code

    if not os.path.isfile(os.path.join(stage_dir, C.MANIFEST)):
        # stage が無いときは何も書かない（フォルダを作ると、あとで stage を作るときに止まる）
        rep["status"] = "stopped"
        rep["reasons"].append(f"stage がない: {C.proj_rel(stage_dir)}（appk_zhjudge_stage.py → GPU の判定のあとに流す）")
        return rep, 1
    s = stored_side(stage_dir)
    rep["original_code"] = s.get("original_code")
    if s["problems"]:
        return stop(s["problems"])
    old_k = s["stored_summary"]["shift_tone_pos"]
    rep["old"] = {**C.wilson(old_k, C.N), "source": C.proj_rel(C.STORED_SUMMARY), "stored_summary": s["stored_summary"],
                  "recheck_file": s["recheck_file"], "recheck_by_original_code": s["recheck"],
                  "original_code_printed": s["printed"]}
    st = load_stage(stage_dir, allow_selftest=allow_selftest)
    rep["stage"] = {k: st.get(k) for k in ("stage_dir", "mode", "manifest_md5", "judge_output_md5", "run_meta",
                                           "run_checks", "production_now_equals_stage", "gpu_run")}
    if st.get("manifest"):
        rep["stage"]["changed_pairs"] = st["manifest"]["changed_pairs"]
    if st["problems"]:
        return stop(st["problems"])
    man = st["manifest"]
    new_doc = C.read_json(st["judge_output"])
    ev = evaluate(new_doc, s["rows"], man, value_tol, accept_value_drift)
    rep["reproduction_of_unchanged_pairs"] = _trim_compare(ev["compare"])
    rep["n_check"] = {"shape_problems": ev["shape_problems"], "n_rows": len(new_doc.get("rows", [])),
                      "judge_summary": new_doc.get("summary")}
    if not ev["pass"]:
        return stop(ev["reasons"])

    # 手順 5: 新しい値（元の集計コードをそのまま）
    txt, summ = C.run_original_code(st["judge_output"], os.path.join(stage_dir, C.SUMMARY_NEW))
    k, n, _ = C.shift_count(new_doc["rows"])
    if not (n == C.N and summ == {"n": C.N, "shift_tone_pos": k, "shift_tone_pct": round(k / 3, 2)}
            and round(k / 3, 2) == round(100 * k / C.N, 2)):
        return stop([f"元の集計コード {summ} と検算 {k}/{n} が合わない"])
    new_by = {r["pair_idx"]: r for r in new_doc["rows"]}
    old_by = {r["pair_idx"]: r for r in s["rows"]}
    detail = changed_detail(new_by, old_by, man)
    pos_old = sum(1 for x in detail if x["stored"]["success"])
    pos_new = sum(1 for x in detail if x["new"]["success"])
    lo, hi = old_k - pos_old, old_k - pos_old + len(detail)
    if not (lo <= k <= hi and k == old_k - pos_old + pos_new):
        return stop([f"新しい k={k} が、作り直していない対の判定が同じときの値 {old_k - pos_old + pos_new}"
                     f"（幅 {lo}..{hi}）と合わない"])
    rep["status"] = "ok"
    rep["new"] = {**C.wilson(k, C.N), "summary_file": C.proj_rel(os.path.join(stage_dir, C.SUMMARY_NEW)),
                  "summary_by_original_code": summ, "original_code_printed": txt}
    rep["change_pct_points"] = round(rep["new"]["pct"] - rep["old"]["pct"], 2)
    rep["changed_pairs"] = detail
    rep["arithmetic"] = {"old_k": old_k, "changed_pairs_success_before": pos_old,
                         "changed_pairs_success_after": pos_new, "new_k": k,
                         "possible_range_k": [lo, hi], "possible_range_pct": [round(100 * lo / C.N, 2),
                                                                              round(100 * hi / C.N, 2)]}
    rep["n_valid_pairs"] = n
    rep["written_to"] = C.proj_rel(out_path)
    C.write_json(rep, out_path)
    return rep, 0


def print_report(rep, out_path):
    written = rep.get("written_to") == C.proj_rel(out_path)
    print(f"[zhjudge-aggregate] status={rep['status']} "
          + (f"-> {C.proj_rel(out_path)}" if written else "(nothing written)"))
    if rep.get("old"):
        o = rep["old"]
        print(f"  old (7/28, stored): {o['k']}/{o['n']} = {o['paper_style']}  "
              f"(Wilson 95% [{o['wilson95_lo_pct']}, {o['wilson95_hi_pct']}])  recheck by original code: "
              f"{o['recheck_by_original_code']}")
    for r in rep["reasons"]:
        print(f"  STOP: {r}")
    if rep["status"] != "ok":
        return
    rp = rep["reproduction_of_unchanged_pairs"]
    print(f"  295 unchanged pairs: decisions reproduced={rp['decisions_reproduced']}, values within "
          f"{rp['value_tol']}={rp['values_reproduced']} (max |diff| {rp['max_abs_value_diff']}, "
          f"{rp['n_values_equal']}/{rp['n_values_compared']} identical), secondary flips within noise "
          f"{len(rp['secondary_flips_within_noise'])}")
    nw = rep["new"]
    print(f"  new (re-judged):    {nw['k']}/{nw['n']} = {nw['paper_style']}  "
          f"(Wilson 95% [{nw['wilson95_lo_pct']}, {nw['wilson95_hi_pct']}]), change {rep['change_pct_points']:+.2f} points, "
          f"n_valid_pairs={rep['n_valid_pairs']}")
    for x in rep["changed_pairs"]:
        print(f"    pair {x['pair_idx']} {x['char']} {x['src']}->{x['tgt']} (tone_only={x['tone_only']}, "
              f"seed {x['new_seed']}): d {x['stored']['d']} ({'success' if x['stored']['success'] else 'fail'}) -> "
              f"{x['new']['d']} ({'success' if x['new']['success'] else 'fail'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_dir", default=C.proj_rel(C.DEFAULT_STAGE))
    ap.add_argument("--value_tol", type=float, default=0.01,
                    help="作り直していない対の尤度の値の差の許容（7/28 の記録は小数 3 桁）")
    ap.add_argument("--accept_value_drift", action="store_true",
                    help="値の差が許容を超えても、主判定が同じなら続ける（著者が認めたときだけ）")
    ap.add_argument("--out", default=None, help="報告の置き場（既定 <stage>/zh_injection_success_rejudge.json）")
    args = ap.parse_args()
    stage_dir = C.resolve(args.stage_dir)
    out = C.resolve(args.out) if args.out else os.path.join(stage_dir, C.REPORT)
    rep, code = run(stage_dir, args.value_tol, args.accept_value_drift, out)
    print_report(rep, out)
    sys.exit(code)


if __name__ == "__main__":
    main()
