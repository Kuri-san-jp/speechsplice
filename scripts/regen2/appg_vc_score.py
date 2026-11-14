#!/usr/bin/env python3
"""付録 G 声まね: 5 行を既存の pyopenjtalk なしの採点器で採点し、各行 n_valid_pairs == 300 を確かめる（2026-09-29 作成、CPU のみ）.

採点器 scripts/projection/score_unified_trm_g2pcer_nopyojt.py（中身は speechsplice_release/src/score_trm_g2pcer_nopyojt.py）は
変えない。1 つの書き起こしの JSON（appg_vc_transcribe.py）を 4 つの入力（--bl_b1_b7_json / --inj_json / --rome_json /
--g2p_json）すべてに渡す。b1_surface_edit は表に無いので n=0（MISSING）と出る（想定どおり）。

採点の前の確かめ（1 行でも 300 に届かなければ採点しない）: 行ごと・対ごとに
  wav がある、wav の md5 == 書き起こしの記録の md5 == selection.json の md5、無音でない（silence_criterion で測り直す）、
  書き起こしが空でない、書き起こしの失敗の印が無い。これを満たす対の数 = n_valid_pairs。5 行とも 300 であること。
採点のあとの確かめ: 採点器の summary の n が 5 行とも 300。
出力: 採点器の出力はまず一時ファイルに書かせ、読めたら os.replace（前の版は .prev_<run_id> に残す）。
  確かめの記録 <root>/score_check_<run_id>.json（表の書式の値つき。0 件の行は Wilson の上限を「+x.xx」で出す）。
原稿（naacl2027/main.tex）と凍結した値は変えない。ここでは数えて表示するだけ。

使い方（ホストで直接、CPU のみ）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_score.py
  --precheck_only で採点せずに n_valid_pairs だけ確かめる。
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
sys.modules["pyopenjtalk"] = None

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from math import sqrt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_vc_common as V  # noqa: E402

SCORER_REL = "scripts/projection/score_unified_trm_g2pcer_nopyojt.py"


def wilson(n, k, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"lower_pct": round(100 * (c - h), 2), "upper_pct": round(100 * (c + h), 2), "half_pct": round(100 * h, 2)}


def table_cell(n, k):
    if not n:
        return "n/a"
    w = wilson(n, k)
    pct = 100 * k / n
    if k == 0:
        return f"0.00 +{w['upper_pct']:.2f}"
    return f"{pct:.2f}±{w['half_pct']:.2f}"


def precheck(root, tr_path):
    doc = V.load_json(tr_path)
    if doc is None:
        return None, {"ok": False, "problems": [f"書き起こしの JSON が無い: {tr_path}"]}
    sel = V.load_json(os.path.join(root, "selection.json"))
    if not sel or not sel.get("complete"):
        return doc, {"ok": False, "problems": ["selection.json が無いか complete でない"]}
    rows = {r["pair_idx"]: r for r in doc.get("rows", [])}
    res = {"ok": sorted(rows) == list(range(V.N_PAIRS)), "per_row": {}, "problems": []}
    if not res["ok"]:
        res["problems"].append(f"pair_idx が 0..{V.N_PAIRS - 1} でない（{len(rows)} 行）")
    for row in V.ROW_ORDER:
        key = V.ROWS[row][2]
        selp = {e["pair_idx"]: e for e in sel["rows"][row]["pairs"]}
        bad = {"missing": [], "md5": [], "silent": [], "empty": [], "error": []}
        valid = 0
        for i in range(V.N_PAIRS):
            r = rows.get(i, {})
            w = r.get(f"wav_{row}") or {}
            p = V.proj_path(w.get("path") or selp[i]["path"])
            ok = True
            if not os.path.isfile(p):
                bad["missing"].append(i)
                continue
            m = V.md5_file(p)
            if not (m == w.get("md5") == selp[i]["md5"]):
                bad["md5"].append(i)
                ok = False
            st = V.AC.wav_stats(p)
            if st["silent"]:
                bad["silent"].append(i)
                ok = False
            if not (r.get(key) or "").strip():
                bad["empty"].append(i)
                ok = False
            if r.get(f"{key}_error"):
                bad["error"].append(i)
                ok = False
            valid += ok
        d = {"n_valid_pairs": valid, **{f"n_{k}": len(v) for k, v in bad.items()},
             **{f"{k}_pairs": v[:20] for k, v in bad.items() if v},
             "seed_counts": {str(sd): sum(1 for e in selp.values() if e["seed"] == sd) for sd in (42, 43, 44)}}
        d["ok"] = valid == V.N_PAIRS
        res["per_row"][row] = d
        res["ok"] &= d["ok"]
    return doc, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=V.OUT_ROOT_REL)
    ap.add_argument("--transcripts", default=None, help="既定 <root>/transcripts_5rows_vc.json")
    ap.add_argument("--output", default=None, help="既定 <root>/final_results_5rows_vc_nopyojt.json")
    ap.add_argument("--precheck_only", action="store_true")
    args = ap.parse_args()
    root = V.root_path(args.output_root)
    tr = V.proj_path(args.transcripts) if args.transcripts else os.path.join(root, "transcripts_5rows_vc.json")
    out = V.proj_path(args.output) if args.output else os.path.join(root, "final_results_5rows_vc_nopyojt.json")
    V.assert_writable_dir(os.path.dirname(out), args.output_root)
    run_id = V.new_run_id("score")
    _, pre = precheck(root, tr)
    chk = {"run_id": run_id, "date": V.now(), "transcripts": V.rel_to_proj(tr),
           "transcripts_md5": V.md5_file(tr) if os.path.isfile(tr) else None, "precheck": pre,
           "silence_criterion": V.AC.criterion_doc()}
    chk_path = os.path.join(root, f"score_check_{run_id}.json")
    for row, d in (pre.get("per_row") or {}).items():
        print(f"[precheck] {'OK' if d['ok'] else 'NG'} {row:<13} n_valid_pairs {d['n_valid_pairs']} "
              f"missing {d['n_missing']} md5 {d['n_md5']} silent {d['n_silent']} empty {d['n_empty']} "
              f"error {d['n_error']} seeds {d['seed_counts']}")
    for p in pre.get("problems", []):
        print(f"[precheck] PROBLEM: {p}")
    if not pre["ok"] or args.precheck_only:
        V.atomic_write_json(chk_path, chk)
        print(f"[score] saved {chk_path}")
        if not pre["ok"]:
            raise SystemExit("[score] n_valid_pairs が 300 に届かない行がある。採点しない")
        return

    scorer = V.proj_path(SCORER_REL)
    tmp = f"{out}.tmp_{run_id}"
    if os.path.lexists(tmp):
        raise SystemExit(f"[score] 一時ファイルの名前が既にある: {tmp}")
    cmd = [sys.executable, "-B", scorer, "--bl_b1_b7_json", tr, "--inj_json", tr, "--rome_json", tr,
           "--g2p_json", tr, "--output", tmp]
    print("[score] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=V.PROJ)
    if r.returncode != 0 or not os.path.isfile(tmp):
        raise SystemExit(f"[score] 採点器が失敗した（exit {r.returncode}）。前の出力は変えていない。一時ファイル: {tmp}")
    sc = json.load(open(tmp))
    if os.path.exists(out):
        shutil.copy2(out, f"{out}.prev_{run_id}")
    os.replace(tmp, out)
    table, bad = {}, []
    for row in V.ROW_ORDER:
        s = sc["summary"].get(row, {})
        n = s.get("n")
        if n != V.N_PAIRS:
            bad.append(row)
        table[row] = {"n": n, "trm_n": s.get("trm_n"), "trm_pct": s.get("trm_pct"), "trm_ci_half": s.get("trm_ci_half"),
                      "g2p_cer_match_n": s.get("g2p_cer_match_n"), "g2p_cer_match_pct": s.get("g2p_cer_match_pct"),
                      "g2p_cer_match_ci_half": s.get("g2p_cer_match_ci_half"),
                      "trm_cell": table_cell(n, s.get("trm_n") or 0),
                      "g2p_cer_cell": table_cell(n, s.get("g2p_cer_match_n") or 0),
                      "trm_wilson": wilson(n, s.get("trm_n") or 0) if n else None,
                      "g2p_cer_wilson": wilson(n, s.get("g2p_cer_match_n") or 0) if n else None}
        print(f"[score] {row:<13} n={n} TRM {table[row]['trm_cell']} ({s.get('trm_n')})  "
              f"G2P-CER {table[row]['g2p_cer_cell']} ({s.get('g2p_cer_match_n')})")
    chk.update({"scorer": SCORER_REL, "scorer_output": V.rel_to_proj(out), "scorer_output_md5": V.md5_file(out),
                "table": table, "all_rows_n_300": not bad, "rows_n_not_300": bad,
                "b1_surface_edit_n": sc["summary"].get("b1_surface_edit", {}).get("n"),
                "gated_rule_info": {"baseline_trm_n": table["baseline"]["trm_n"],
                                    "note": "main.tex L1060 gated rule: if the Baseline row TRM is 0/300 the gate "
                                            "cannot change the SpeechSplice value"}})
    V.atomic_write_json(chk_path, chk)
    print(f"[score] saved {out}\n[score] saved {chk_path}")
    if bad:
        raise SystemExit(f"[score] n が 300 でない行: {bad}")


if __name__ == "__main__":
    main()
