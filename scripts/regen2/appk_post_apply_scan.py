#!/usr/bin/env python3
"""G1: 作り直した対を本番に反映したあとの無音の検査と照合（2026-09-29、CPU のみ）.

1. appk_silent_scan.py と同じ検査（全 G 共通の無音の関数）を 6 言語 3,600 本に流し、--out に書く
   （既にあれば .bak_<時刻> に退避。この結果を IPA 採点 appk_run_ipa.py の --silent_scan に渡す）。
2. 計画（eval_results/_appk_regen/plan.json。作り直し前の検査の md5 を持つ）と反映の記録
   （本番の置き場の regen2_appk_pairs.json）と照らし合わせる:
   * md5 が変わった wav が、反映の記録にある wav とちょうど同じで、新しい md5 が記録と一致する
   * 各言語 300 行・pair_idx 0..299・(text_a, text_b) が作り直し前と同じ
   * 使えない対（無音・欠け・読めない）が 0（残れば言語と対を出す）
   * JA は 1 本も変わっていない
3. 照合の結果を <out>.post_apply_check.json に書く（既にあれば .bak_<時刻> に写してから上書きする）。
   全部よければ終了コード 0、そうでなければ 1。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定するので、env や nice は付けない。
相対パスの --out はプロジェクト直下を基準に解く）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_post_apply_scan.py --out eval_results/_appk_redo/silent_scan_n300.json
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_common as K  # noqa: E402

K.cap_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import appk_silent_scan as S  # noqa: E402


def check(res, plan, prev_text):
    """prev_text: 作り直し前の検査（計画の scan_path）の言語ごとの text_pairs_by_idx（無ければ None）."""
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": K.plan_path(), "langs": {}}
    all_ok = True
    for lang in K.LANGS:
        ls = res["langs"][lang]
        pre = plan["per_wav_md5"].get(lang, {})
        now = {n: v["md5"] for n, v in res["per_wav"][lang].items()}
        changed = sorted(n for n in set(pre) | set(now) if pre.get(n) != now.get(n))
        rec_path = os.path.join(K.set_dir(lang), K.PROD_RECORD_NAME)
        expected = {}
        if os.path.isfile(rec_path):
            rec = K.read_json(rec_path)
            expected = {f["file"]: f["new_md5"] for f in rec["files"]}
        unexpected = [n for n in changed if n not in expected]
        wrong_new = [n for n, m in expected.items() if now.get(n) != m]
        not_changed = [n for n in expected if n not in changed and expected[n] != pre.get(n)]
        # (text_a, text_b) が作り直し前と同じか（作り直し前の検査の text_pairs_by_idx と比べる）
        prev = (prev_text or {}).get(lang)
        text_same = None if prev is None else prev == ls["text_pairs_by_idx"]
        unusable = sorted({p["pair_idx"] for p in ls["silent_pairs"]} | set(ls["pairs_missing_or_unreadable"]))
        ok = (not unexpected and not wrong_new and not not_changed and not unusable
              and ls["n_ckpt_rows"] == K.N_EXPECTED and ls["pair_idx_is_0_to_299"]
              and ls["n_distinct_text_pairs"] == K.N_EXPECTED and text_same is not False
              and (lang != "ja" or not changed))
        all_ok &= ok
        out["langs"][lang] = {"ok": ok, "n_changed_since_plan": len(changed), "changed_not_in_record": unexpected,
                              "record_md5_mismatch": wrong_new, "record_files_unchanged": not_changed,
                              "record": rec_path if expected else None, "n_record_files": len(expected),
                              "unusable_pairs": unusable, "silent_pairs": [p["pair_idx"] for p in ls["silent_pairs"]],
                              "text_pairs_same_as_before": text_same,
                              "rows": ls["n_ckpt_rows"], "distinct_text_pairs": ls["n_distinct_text_pairs"]}
        print(f"[{lang}] ok={ok} changed={len(changed)} (record {len(expected)}) unexpected={unexpected[:4]} "
              f"wrong_new={wrong_new[:4]} unusable={unusable} text_same={text_same}")
    out["all_ok"] = all_ok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_results/_appk_redo/silent_scan_n300.json")
    args = ap.parse_args()
    plan = K.read_json(K.plan_path())
    out = S.resolve_out(args.out)
    # 作り直し前の検査を、同じ名前に上書きする前に読んでおく
    prev_text = None
    sp = plan.get("scan_path")
    if sp and os.path.isfile(sp):
        prev_scan = K.read_json(sp)
        prev_text = {l: v.get("text_pairs_by_idx") for l, v in prev_scan["langs"].items()}
        if prev_scan.get("date") != plan.get("scan_date"):
            print(f"WARNING: {sp} は計画を作ったときの検査ではない（{prev_scan.get('date')} != {plan.get('scan_date')}）。"
                  f"文の照合は飛ばす。")
            prev_text = None
    res = S.run_scan(list(K.LANGS), quiet=True)
    S.write_scan(res, out)
    print(f"[scan] {res['total']} -> {out}")
    chk = check(res, plan, prev_text)
    chk["scan"] = out
    cp = out + ".post_apply_check.json"
    if os.path.exists(cp):   # 前の照合結果を写してから上書きする（移動・削除はしない）
        bak = f"{cp}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(cp, bak)
        print(f"[backup] {cp} -> {bak}")
    with open(cp, "w") as f:
        json.dump(chk, f, ensure_ascii=False, indent=1)
    print(f"[post_apply] all_ok={chk['all_ok']} -> {cp}")
    sys.exit(0 if chk["all_ok"] else 1)


if __name__ == "__main__":
    main()
