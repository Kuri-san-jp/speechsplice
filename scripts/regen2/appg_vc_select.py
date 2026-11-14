#!/usr/bin/env python3
"""付録 G 声まね: 各単位で使う seed を決め、5 行 × 300 の wav の一覧（selection.json）を作る（2026-09-29 作成、CPU のみ）.

決め方（D2。結果は見ない）: 単位ごとに seed 42 → 43 → 44 の順で最初の有効なもの（appg_vc_common.build_selection）。
  check42 は記録だけで選ばない。Baseline / Text Hint は既定（--textedit_unit b7、著者の決定 2026-09-29）で b7 だけを選び直す。
  bl（Baseline 行）と b1 は、どの対でも本番の seed 42 のまま。
  G2P / ROME / SpeechSplice は _appg_vc の置き場の wav をそのまま使う（本番のファイルは無いので差し替えは無い）。
  Baseline / Text Hint（textedit）で 43/44 を使う対は、本番 eval_results/baseline_text_edit_ja_n300/wavs への差し替え
  （appg_vc_apply_textedit.py、著者に見せたあとリードが流す）が済むまで ready_for_transcription = false。
すべての単位の作り直しが終わっていなければ（appg_vc_scan.py の next_action）、何も選ばずに止まる。
書く物: <root>/selection.json（前の版は .prev_<時刻> に写して残す）。torch は読まない。

使い方（ホストで直接、CPU のみ）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_select.py
  --check_only で書かずに表示だけ。
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
sys.modules["pyopenjtalk"] = None

import argparse  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_vc_common as V  # noqa: E402


def selection_path(root=None):
    return os.path.join(V.root_path(root), "selection.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=V.OUT_ROOT_REL)
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--textedit_unit", default=V.TEXTEDIT_UNIT_DEFAULT, choices=sorted(V.TEXTEDIT_UNITS),
                    help="textedit の作り直しで選ぶ wav（既定 b7 だけ）")
    args = ap.parse_args()
    V.set_textedit_unit(args.textedit_unit)
    root = V.root_path(args.output_root)
    print(f"[select] textedit の作り直しで選ぶ wav: {V.regen_keys('textedit')}（--textedit_unit {V.textedit_unit()}）")
    sel = V.build_selection(root)
    if not sel["complete"]:
        for p in sel["problems"]:
            print(f"[select] まだ選べない: {p}")
        if not args.check_only:
            raise SystemExit("[select] 作り直しが終わっていないので selection.json は書かない")
        return
    for m, c in sel["seed_counts"].items():
        print(f"[select] {m:<9} seeds used {c}")
    print(f"[select] Baseline / Text Hint で差し替えが要る対（apply 前）: {sel['textedit_needs_apply']}")
    print(f"[select] 本番から変わる行の対: {sel['changed_rows_vs_production']}")
    print(f"[select] ready_for_transcription={sel['ready_for_transcription']}")
    if args.check_only:
        print("[select] --check_only: 書いていない")
        return
    out = selection_path(args.output_root)
    kept = V.write_json_keep_prev(out, sel)
    print(f"[select] saved {out}" + (f"（前の版 {kept}）" if kept else ""))


if __name__ == "__main__":
    main()
