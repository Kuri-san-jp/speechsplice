#!/usr/bin/env python3
"""付録 K 英語の判定: 集計（2026-09-29、ホストの CPU のみ）.

順番（前の段で止まったら、値は計算も表示もしない。結果を見てから条件を選ばないため）:
  1. 一覧（stage_manifest.json）: 著者が認めた読みの表で作られ、600 本の音声が一覧の md5 のままで今の本番とも同じ、
     読みの表・仮説の md5 が一覧と同じ。
  2. GPU の実行記録（status=done の最新）: 中国語の判定器の md5 と取り出した関数の md5、認識器の snapshot、
     一覧と出力の md5、行 300、実行中に音声が変わっていない、pyopenjtalk の封じ、CUDA、コンテナの中。
  3. 行の確認（N=300）: 300 行、pair_idx 0..299、skip なし、欄がそろう、語・ラベル・分類・degenerate が一覧と同じ、
     フレームに収まらない仮説 0、読みが同じ対（degenerate）は BL と注入の両方で src と tgt の値が完全に同じ、
     判定器の関数と torch の ctc_loss の差 ≤ 0.01、丸めと *_choose_tgt が丸める前の値と合う。
  4. 数え方の確認: 中国語の 7/28 の集計コードと同じ書き方の数え方が、中国語の 7/28 の行から 281/300 = 93.67 を出す
     （読むだけ）。中国語のファイル cjk_n300_zh_v2/ctc_lr_judge.json の md5 が固定値と違って確かめられないときも止まる
     （確かめを飛ばして値を出すことはしない）。
  5. 値:
     主  BL からの移り方  d = (inj_lp_tgt − inj_lp_src) − (bl_lp_tgt − bl_lp_src) > 0 の数 / 300（読みが同じ対は d = 0 で失敗）
     副  注入の音声が tgt を選んだ数 / 300（inj_choose_tgt。読みが同じ対は同点で失敗）
     確かめ（前もって決めたもの。報告だけ。対は除かない）  BL の音声が src を選んだ率
         （中国語の判定器と同じく読みが同じ対を除いた 240 対で数える。300 対で数えた値も並べる。中国語の合格水準 95% は参考）
     ± はどれも Wilson 95% の半幅（z = 1.96。中国語の集計コードの wil() と同じ式）と区間。
     読みが同じ対の一覧、語ごと・分類ごとの内訳、差し替えた対 23 の値も出す。
  6. 報告 <stage>/en_injection_success.json（前の版があれば .bak_<時刻> に写す）。原稿は書き換えない。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_enjudge_aggregate.py --stage_dir eval_results/_appk_enjudge/judge_ctc
終了コード: 0 = 値を出した、1 = 止まった（理由は表示と報告の reasons）。
モデルも Hugging Face も numba も読まないが、決まりどおり HF_HOME などを最初に設定する。
"""
import os
import sys

sys.dont_write_bytecode = True
# numpy / torch を読む前に（このスクリプトは読まないが、決まりどおり最初に設定する。appk_enjudge_run_gpu.py と同じ）
_PROJ = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
if _PROJ != "/mnt":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ホストでは GPU を見せない
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
_HF = os.path.join(_PROJ, ".cache", "huggingface")
os.environ["HF_HOME"] = _HF
os.environ["HF_HUB_CACHE"] = os.path.join(_HF, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(_PROJ, ".cache", "numba")
sys.modules["pyopenjtalk"] = None   # 自分のコードでは pyopenjtalk を使わない
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_enjudge_common as C  # noqa: E402

K = C.K
K.cap_threads()

import argparse  # noqa: E402
import glob  # noqa: E402
import time  # noqa: E402

REQUIRED = ("pair_idx", "word", "src_label", "tgt_label", "category", "degenerate",
            "bl_lp_src", "bl_lp_tgt", "inj_lp_src", "inj_lp_tgt", "bl_choose_tgt", "inj_choose_tgt", "full")


def load_stage(stage_dir, allow_selftest=False):
    """手順 1–2."""
    st = {"problems": [], "stage_dir": C.proj_rel(stage_dir)}
    mp = os.path.join(stage_dir, C.MANIFEST)
    if not os.path.isfile(mp):
        st["problems"].append(f"一覧がない: {C.proj_rel(mp)}")
        return st
    man = C.read_json(mp)
    st.update(manifest=man, manifest_md5=C.md5_file(mp))
    if not man["table"].get("author_approved"):
        st["problems"].append("一覧が著者の認めていない読みの表で作られている")
    if man.get("mode") != "post_apply" or (man.get("selftest") and not allow_selftest):
        st["problems"].append(f"一覧の mode={man.get('mode')} selftest={man.get('selftest')}")
    if sorted(man.get("files", {})) != sorted(C.all_wav_names()):
        st["problems"].append("一覧の音声が 600 本そろっていない")
    diff = [n for n, rec in man.get("files", {}).items()
            if (C.md5_file(os.path.join(C.PROD_DIR, n)) if os.path.isfile(os.path.join(C.PROD_DIR, n)) else None)
            != rec["md5"]]
    st["production_now_equals_stage"] = {"wav_diff": diff, "equal": not diff}
    if diff:
        st["problems"].append(f"一覧を作ったあとで本番の音声が変わった（{len(diff)} 本、例 {diff[:3]}）。この判定は今の本番を表さない")
    tp = C.resolve(man["table"]["path"])
    if (C.md5_file(tp) if os.path.isfile(tp) else None) != man["table"]["md5"]:
        st["problems"].append("読みの表が一覧を作ったあとで変わった")
    if C.hypotheses_digest(man.get("hypotheses", [])) != man.get("hypotheses_md5"):
        st["problems"].append("一覧の仮説の md5 が合わない")
    jp = os.path.join(stage_dir, C.JUDGE_OUT)
    if not os.path.isfile(jp):
        st["problems"].append(f"GPU の出力がない: {C.proj_rel(jp)}（GPU の判定がまだ）")
        return st
    st["judge_output"] = jp
    st["judge_output_md5"] = C.md5_file(jp)
    metas = [(p, C.read_json(p)) for p in sorted(glob.glob(os.path.join(stage_dir, C.RUN_META_PREFIX + "*.json")))]
    metas = [(p, m) for p, m in metas if m.get("status") == "done"]
    if not metas:
        st["problems"].append("GPU の実行記録（status=done）がない")
        return st
    mpath, meta = metas[-1]
    env = meta.get("environment", {})
    st["run_meta"] = C.proj_rel(mpath)
    st["gpu_run"] = {k: meta.get(k) for k in ("started", "finished", "elapsed_min", "judge_output_md5",
                                              "model_fingerprint", "selftest")}
    st["gpu_run"]["environment"] = env
    try:
        _, fn_md5 = C.zh_ctc_source()
    except SystemExit as e:
        fn_md5 = None
        st["problems"].append(str(e))
    checks = {
        "zh_judge_md5": meta.get("zh_judge_md5") == C.EXPECTED_MD5["zh_judge"],
        "ctc_function_md5": fn_md5 is not None and meta.get("ctc_function_md5") == fn_md5,
        "snapshots": meta.get("snapshot_main") == C.SNAP_MAIN and meta.get("snapshot_pr7") == C.SNAP_PR7,
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
    return st


def check_rows(doc, man):
    """手順 3。戻り値: problems."""
    rows = doc.get("rows", [])
    problems = []
    if len(rows) != C.N:
        problems.append(f"行が {len(rows)}（300 でない）")
    idx = [r.get("pair_idx") for r in rows]
    if idx != list(range(C.N)):
        problems.append("pair_idx が 0..299 の順に 1 回ずつになっていない")
    if any("skip" in r for r in rows):
        problems.append(f"skip の行がある: {[r.get('pair_idx') for r in rows if 'skip' in r][:10]}")
    miss = [r.get("pair_idx") for r in rows if any(k not in r for k in REQUIRED)]
    if miss:
        problems.append(f"欄が欠けた行: {miss[:10]}")
        return problems
    hyp = {h["pair_idx"]: h for h in man["hypotheses"]}
    meta_bad = [r["pair_idx"] for r in rows if r["pair_idx"] not in hyp or any(
        r[k] != hyp[r["pair_idx"]][k] for k in ("word", "src_label", "tgt_label", "category", "degenerate"))]
    if meta_bad:
        problems.append(f"語・ラベル・分類・degenerate が一覧と違う行: {meta_bad[:10]}")
    if sorted(r["pair_idx"] for r in rows if r["degenerate"]) != sorted(man["degenerate_pairs"]):
        problems.append("読みが同じ対の組が一覧と違う")
    infeasible = [r["pair_idx"] for r in rows if min(r[f] for f in C.LP_FIELDS) < C.INFEASIBLE_LP]
    if infeasible:
        problems.append(f"音素列がフレームに収まらなかった対: {infeasible[:10]}")
    deg_bad = [r["pair_idx"] for r in rows if r["degenerate"] and not (
        r["bl_lp_src"] == r["bl_lp_tgt"] and r["inj_lp_src"] == r["inj_lp_tgt"]
        and r["full"]["bl"]["lp_src"] == r["full"]["bl"]["lp_tgt"]
        and r["full"]["inj"]["lp_src"] == r["full"]["inj"]["lp_tgt"])]
    if deg_bad:
        problems.append(f"読みが同じ対なのに src と tgt の値が違う: {deg_bad[:10]}")
    worst, round_bad, flag_bad = 0.0, [], []
    for r in rows:
        for side in C.WAV_SIDES:
            f = r["full"][side]
            worst = max(worst, abs(f["lp_src"] - f["torch_ctc_src"]), abs(f["lp_tgt"] - f["torch_ctc_tgt"]))
            if round(f["lp_src"], 3) != r[f"{side}_lp_src"] or round(f["lp_tgt"], 3) != r[f"{side}_lp_tgt"]:
                round_bad.append(r["pair_idx"])
            if (f["lp_tgt"] > f["lp_src"]) != r[f"{side}_choose_tgt"]:
                flag_bad.append(r["pair_idx"])
    if worst > C.CTC_XCHECK_TOL:
        problems.append(f"判定器の関数と torch の ctc_loss の差が {worst:.4f}（{C.CTC_XCHECK_TOL} を超える）")
    if round_bad or flag_bad:
        problems.append(f"丸めた値か *_choose_tgt が丸める前の値と合わない: {sorted(set(round_bad + flag_bad))[:10]}")
    return problems


def zh_formula_check():
    """手順 4: 同じ書き方の数え方が中国語の 7/28 の行から 281/300 を出すか（読むだけ）.
    md5 が違えば数えずに checked=False を返す（run はそのとき止まる）."""
    ok, m = C.check_pinned("zh_stored_judge", C.ZH_STORED_JUDGE)
    if not ok:
        return {"checked": False, "why": f"{C.proj_rel(C.ZH_STORED_JUDGE)} の md5 {m} が固定値でない"}
    k, n, pct = C.shift_count_as_zh_original(C.read_json(C.ZH_STORED_JUDGE)["rows"])
    out = {"checked": True, "source": C.proj_rel(C.ZH_STORED_JUDGE), "k": k, "n": n, "pct": pct,
           "equals_zh_7_28": (k, n, pct) == (281, 300, 93.67)}
    try:   # 中国語の判定し直しの共通モジュールの関数とも合うか（記録だけ）
        import appk_zhjudge_common as Z
        out["zhjudge_common_shift_count"] = list(Z.shift_count(C.read_json(C.ZH_STORED_JUDGE)["rows"])[:2])
        out["zhjudge_common_wilson_281"] = Z.wilson(281, 300)["paper_style"]
    except BaseException as e:  # noqa: BLE001
        out["zhjudge_common"] = f"not imported: {e!r}"[:200]
    out["this_wilson_281"] = C.wilson(281, 300)["paper_style"]
    return out


def values(rows, man):
    """手順 5."""
    n = len(rows)
    d = {r["pair_idx"]: C.shift_of(r) for r in rows}
    succ = {i for i, x in d.items() if x > 0}
    k2, n2, pct2 = C.shift_count_as_zh_original(rows)
    if not (n == n2 == C.N and len(succ) == k2 and round(100 * k2 / C.N, 2) == pct2):
        raise SystemExit(f"数え方が合わない: {len(succ)} / {k2} / {pct2}")
    deg = [r for r in rows if r["degenerate"]]
    nondeg = [r for r in rows if not r["degenerate"]]
    if any(r["pair_idx"] in succ for r in deg):
        raise SystemExit("読みが同じ対が成功に数えられている")
    inj_tgt = [r["pair_idx"] for r in rows if r["inj_choose_tgt"]]
    bl_src_nd = [r["pair_idx"] for r in nondeg if not r["bl_choose_tgt"]]
    bl_src_300 = [r["pair_idx"] for r in rows if (not r["degenerate"]) and not r["bl_choose_tgt"]]
    out = {
        "primary_shift": {**C.wilson(len(succ), C.N),
                          "rule": "d = (inj_lp_tgt - inj_lp_src) - (bl_lp_tgt - bl_lp_src) > 0; identical-reading pairs have d = 0",
                          "as_zh_original_code": {"k": k2, "n": n2, "pct_round_k_over_3": pct2}},
        "secondary_inj_prefers_target": {**C.wilson(len(inj_tgt), C.N),
                                         "rule": "inj_lp_tgt > inj_lp_src (unrounded); identical-reading pairs tie and fail"},
        "preset_check_bl_prefers_source": {
            "note": "reported only; no pair is removed or changed because of it",
            "over_nondegenerate": {**C.wilson(len(bl_src_nd), len(nondeg)),
                                   "zh_reference_pct": C.ZH_REF_BL_SRC_PCT,
                                   "at_or_above_zh_reference": 100 * len(bl_src_nd) / len(nondeg) >= C.ZH_REF_BL_SRC_PCT},
            "over_300_ties_as_not": C.wilson(len(bl_src_300), C.N)},
        "n_check": {"n_rows": n, "denominator": C.N, "n_success_primary": len(succ),
                    "n_fail_primary": C.N - len(succ), "n_degenerate": len(deg),
                    "n_degenerate_counted_as_failure": sum(1 for r in deg if r["pair_idx"] not in succ),
                    "n_pairs_d_exactly_zero": sum(1 for x in d.values() if x == 0)},
        "degenerate_pairs": [{"pair_idx": r["pair_idx"], "word": r["word"], "src_label": r["src_label"],
                              "tgt_label": r["tgt_label"]} for r in deg],
    }
    by_cat, by_word = {}, {}
    for r in rows:
        for key, bucket in ((r["category"], by_cat), (r["word"], by_word)):
            b = bucket.setdefault(key, {"n": 0, "primary_k": 0, "inj_prefers_target_k": 0, "bl_prefers_source_k": 0})
            b["n"] += 1
            b["primary_k"] += int(r["pair_idx"] in succ)
            b["inj_prefers_target_k"] += int(r["inj_choose_tgt"])
            b["bl_prefers_source_k"] += int((not r["degenerate"]) and not r["bl_choose_tgt"])
    for w, b in by_word.items():
        b["category"] = man["categories"][w]
    out["by_category"] = by_cat
    out["by_word"] = by_word
    out["replaced_pairs"] = [{"pair_idx": i, "seed": (man["apply_record"].get("seeds") or {}).get(str(i)),
                              "word": rows[i]["word"], "src_label": rows[i]["src_label"],
                              "tgt_label": rows[i]["tgt_label"], "d": round(d[i], 3), "success": i in succ,
                              **{f: rows[i][f] for f in C.LP_FIELDS}} for i in man.get("replaced_pairs", [])]
    return out


def run(stage_dir, out_path=None, allow_selftest=False):
    stage_dir = os.path.abspath(stage_dir)
    out_path = out_path or os.path.join(stage_dir, C.REPORT)
    C.assert_write(out_path)
    rep = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "status": None, "reasons": [],
           "what": ("Appendix K, English injection success with the same judge as the Chinese row: CTC log-likelihood "
                    "of the whole-sentence phoneme sequence with the target vs the source reading at the entity "
                    f"({C.MODEL_ID}), BL-relative shift, N=300 (identical-reading pairs counted as failures)"),
           "method": {"ctc_function": f"{C.proj_rel(C.ZH_JUDGE)}:{C.ZH_CTC_FUNC}",
                      "zh_judge_md5": C.EXPECTED_MD5["zh_judge"], "recognizer": C.MODEL_ID,
                      "interval": "Wilson 95% (z = 1.96); pct ± half-width as in the paper"},
           "not_written": [C.proj_rel(C.PROD_DIR) + "/", "naacl2027/main.tex"]}

    def stop(reasons):
        rep["status"] = "stopped"
        rep["reasons"] += reasons
        rep["written_to"] = C.proj_rel(out_path)
        C.write_json(rep, out_path)
        return rep, 1

    if not os.path.isfile(os.path.join(stage_dir, C.MANIFEST)):
        rep["status"] = "stopped"
        rep["reasons"].append(f"一覧がない: {C.proj_rel(stage_dir)}（stage → GPU のあとに流す）。何も書かない")
        return rep, 1
    st = load_stage(stage_dir, allow_selftest=allow_selftest)
    rep["stage"] = {k: st.get(k) for k in ("stage_dir", "manifest_md5", "judge_output_md5", "run_meta", "run_checks",
                                           "production_now_equals_stage", "gpu_run")}
    if st.get("manifest"):
        man = st["manifest"]
        rep["stage"]["table"] = man["table"]
        rep["stage"]["context_mapping"] = man.get("context_mapping")
        rep["stage"]["hypotheses_md5"] = man.get("hypotheses_md5")
        rep["stage"]["symbol_emission_summary"] = (man.get("symbol_emission") or {}).get("summary")
    if st["problems"]:
        return stop(st["problems"])
    doc = C.read_json(st["judge_output"])
    rp = check_rows(doc, man)
    rep["n_check_rows"] = {"n_rows": len(doc.get("rows", [])), "problems": rp, "gpu_summary": doc.get("summary")}
    if rp:
        return stop(rp)
    zf = zh_formula_check()
    rep["same_formula_as_zh_row"] = zf
    if not zf.get("checked"):
        return stop([f"数え方を中国語の 7/28 の行で確かめられない（{zf.get('why')}）。確かめずに値は出さない"])
    if not zf.get("equals_zh_7_28"):
        return stop([f"同じ書き方の数え方が中国語の 7/28 の値を出さない: {zf}"])
    try:
        rep.update(values(doc["rows"], man))
    except SystemExit as e:
        return stop([str(e)])
    rep["status"] = "ok"
    rep["written_to"] = C.proj_rel(out_path)
    C.write_json(rep, out_path)
    return rep, 0


def print_report(rep, out_path):
    written = rep.get("written_to") == C.proj_rel(out_path)
    print(f"[enjudge-aggregate] status={rep['status']} " + (f"-> {C.proj_rel(out_path)}" if written else "(nothing written)"))
    for r in rep["reasons"]:
        print(f"  STOP: {r}")
    if rep["status"] != "ok":
        return
    zf = rep["same_formula_as_zh_row"]
    print(f"  same formula on the Chinese 7/28 rows: {zf.get('k')}/{zf.get('n')} = {zf.get('pct')} (checked={zf.get('checked')})")
    p, s = rep["primary_shift"], rep["secondary_inj_prefers_target"]
    b = rep["preset_check_bl_prefers_source"]
    print(f"  primary   (BL-relative shift, d>0): {p['k']}/{p['n']} = {p['paper_style']}  "
          f"(Wilson 95% [{p['wilson95_lo_pct']}, {p['wilson95_hi_pct']}])")
    print(f"  secondary (INJ prefers target):     {s['k']}/{s['n']} = {s['paper_style']}  "
          f"(Wilson 95% [{s['wilson95_lo_pct']}, {s['wilson95_hi_pct']}])")
    nd = b["over_nondegenerate"]
    print(f"  pre-set check (BL prefers source, report only): {nd['k']}/{nd['n']} = {nd['paper_style']} "
          f"(Chinese reference {nd['zh_reference_pct']}%: at or above = {nd['at_or_above_zh_reference']}); "
          f"over 300 = {b['over_300_ties_as_not']['paper_style']}")
    nc = rep["n_check"]
    print(f"  n check: rows {nc['n_rows']}, denominator {nc['denominator']}, identical-reading pairs "
          f"{nc['n_degenerate']} (all failures: {nc['n_degenerate_counted_as_failure'] == nc['n_degenerate']})")
    for cat, v in rep["by_category"].items():
        print(f"    {cat:18s} n={v['n']:3d} primary {v['primary_k']:3d}  inj->tgt {v['inj_prefers_target_k']:3d}  "
              f"bl->src {v['bl_prefers_source_k']:3d}")
    for x in rep["replaced_pairs"]:
        print(f"    replaced pair {x['pair_idx']} (seed {x['seed']}) {x['word']} {x['src_label']}->{x['tgt_label']}: "
              f"d {x['d']} ({'success' if x['success'] else 'fail'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_dir", default=C.proj_rel(C.DEFAULT_STAGE))
    ap.add_argument("--out", default=None, help="報告の置き場（既定 <stage>/en_injection_success.json）")
    args = ap.parse_args()
    stage_dir = C.resolve(args.stage_dir)
    out = C.resolve(args.out) if args.out else os.path.join(stage_dir, C.REPORT)
    rep, code = run(stage_dir, out)
    print_report(rep, out)
    sys.exit(code)


if __name__ == "__main__":
    main()
