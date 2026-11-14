#!/usr/bin/env python3
"""付録 G 声まね: 全単位の有効・無効を今のファイルで判定し、作り直しの対象と次にすることを出す（2026-09-29 作成、CPU のみ）.

判定（appg_vc_common.assess_unit）: 単位の wav がすべてあり、記録（items/pNNN.json）の md5 と一致し、生成に成功し、
  1 本も無音でない（silence_criterion.py）なら有効。生成失敗か無音なら無効。記録と実物が食い違えば止めて調べる。
  本番の Baseline / Text Hint（textedit の seed 42）は記録が無いので wav だけで判定する（5/15 の回。900 本）。
作り直しの対象（D2。結果は見ない）: seed 42 で無効な単位 → check42（seed 42 の再現の確かめ。記録だけで選ばない）と seed 43、
  seed 43 でも無効な単位 → seed 44。seed 44 でも無効なら著者に聞く。
  Baseline / Text Hint は既定（--textedit_unit b7、著者の決定 2026-09-29）で b7 だけを作り直す（置き場 textedit/b7only/<段>）。
  本番の bl・b1 が無音の単位があれば、この決め方では直らないので problem として止まる。
書く物: <root>/scan/scan_latest.json（前の版は .prev_<時刻> に写して残す）。torch は読まない。

使い方（ホストで直接、CPU のみ）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_scan.py
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_scan.py --methods textedit
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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_vc_common as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=V.OUT_ROOT_REL)
    ap.add_argument("--methods", default=",".join(V.METHOD_ORDER))
    ap.add_argument("--out", default=None, help="既定 <root>/scan/scan_latest.json")
    ap.add_argument("--textedit_unit", default=V.TEXTEDIT_UNIT_DEFAULT, choices=sorted(V.TEXTEDIT_UNITS),
                    help="textedit の作り直しで作る wav（既定 b7 だけ）")
    args = ap.parse_args()
    V.set_textedit_unit(args.textedit_unit)
    root = V.root_path(args.output_root)
    methods = [m for m in args.methods.split(",") if m]
    assert all(m in V.METHODS for m in methods), methods
    rep = {"created": V.now(), "root": V.rel_to_proj(root), "silence_criterion": V.AC.criterion_doc(),
           "unit_rule": "a unit is invalid if generation failed or any wav in the unit is silent; "
                        "regenerate the unit with seed 43, then 44 (validity only). textedit: only the wavs in "
                        "textedit_regen_keys are regenerated/selected/replaced (b7 by default); check42 is a record only",
           "textedit_unit": V.textedit_unit(), "textedit_regen_keys": V.regen_keys("textedit"),
           "methods": {}}
    print(f"[scan] textedit の作り直しで作る wav: {V.regen_keys('textedit')}（--textedit_unit {V.textedit_unit()}）")
    ok = True
    for m in methods:
        s = V.scan_method(m, root)
        rep["methods"][m] = V.public_scan(s)
        ok &= s["done"]
        st42 = s["stages"]["seed42"]
        print(f"[scan] {m:<9} seed42: valid {st42['n_valid']} invalid {st42['n_invalid']} "
              f"not_attempted {st42['n_not_attempted']} inconsistent {st42['n_inconsistent']}"
              f" | invalid {st42['invalid'][:20]}")
        for tag in ("check42", "seed43", "seed44"):
            if tag in s["stages"]:
                t = s["stages"][tag]
                extra = ""
                if tag == "check42" and "reproduction" in t:
                    r = t["reproduction"]
                    extra = (f" | reproduction: all-md5-identical {r['n_all_keys_md5_identical']}, "
                             f"still invalid {r['n_still_invalid']}")
                print(f"[scan] {m:<9} {tag}: targets {len(t['targets'])} valid {t['n_valid']} invalid {t['n_invalid']} "
                      f"not_attempted {t['n_not_attempted']} inconsistent {t['n_inconsistent']}{extra}")
        print(f"[scan] {m:<9} seed_counts={s['seed_counts']} unresolved={s['unresolved']} -> {s['next_action']}")
        for p in s["problems"]:
            print(f"[scan] {m:<9} PROBLEM: {p}")
    rep["all_done"] = ok
    out = V.proj_path(args.out) if args.out else os.path.join(root, "scan", "scan_latest.json")
    V.assert_writable_dir(os.path.dirname(out), args.output_root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = V.write_json_keep_prev(out, rep)
    print(f"[scan] saved {out}" + (f"（前の版 {kept}）" if kept else ""))
    print(f"[scan] all_done={ok}")


if __name__ == "__main__":
    main()
