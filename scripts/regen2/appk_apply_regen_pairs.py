#!/usr/bin/env python3
"""G1: appk_regen_pairs_seed.py で作り直した対を、付録 K の本番の音声集合に差し替える（2026-09-29、CPU のみ）.

手順は scripts/nontemplate/apply_regen_pairs.py と同じ（言語ごと）:
  0. 確かめる: selection_<lang>.json が "done"、計画の対象と一致、seed 42 の再現が一致（しなければ
     --accept_check_mismatch がないと止める）、作り直した wav の md5 が selection の記録と一致、本番の対象の wav が
     計画（無音の検査）の時点から変わっていない、退避先と本番の記録ファイルがまだない、生成スクリプトの集計式が
     本番の summary を再現する（EN/FR/DE/KO）。
  1. 退避: 本番の対象の wav（bl と注入側）と JSON（checkpoint.json・results.json、ZH は checkpoint_zh.json）を
     eval_results/_appk_regen/original_seed42_backup/<lang>/ へ写す。
  2. 差し替え: 採用した seed の wav を本番へ写し、md5 を確かめる。
  3. JSON: 対象の行を pair_idx で作り直しの行に入れ替える。EN/FR/DE/KO は results.json の summary を
     生成スクリプト（eval_utmos_speaker_fixed.py main 末尾）と同じ式で計算し直す（reference_wav と elapsed_min は元のまま）。
     ZH は本番に summary のファイルが無いので、eval_cjk_n300_zhv2.py が表示する値を記録にだけ残す。
  4. 記録: 本番の置き場に regen2_appk_pairs.json（古い md5・新しい md5・seed・前後の summary・古くなった既存の
     結果の一覧）を書き、同じものを eval_results/_appk_regen/apply_record_<lang>.json にも書く。
seed 44 でも使えなかった組（未解決）は差し替えない（元の対のまま。反映後の無音の検査で残りとして出る）。
注入成功の判定はやり直さない。ZH の 93.67（llr_shift_summary.json）への影響の幅だけを記録に書く。
古くなる既存の結果（話者類似度・F0、ZH の CTC 判定）は書き換えない（記録に一覧を載せる）。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定するので、env や nice は付けない）:
  予行（何も書かない）:
    taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_apply_regen_pairs.py
  反映（予行を著者が承認したあとだけ）:
    taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_apply_regen_pairs.py --apply
  そのあと無音の検査:
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


def dump_like(obj, path, indent):
    """本番の JSON を同じ字下げで上書きする（precedent apply_regen_pairs.py と同じく、その場で書く。元は退避済み）."""
    K.assert_prod_write(path)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def prepare(lang, plan, accept_mismatch):
    """反映の中身を決めて確かめる（書かない）。problems が空なら反映できる."""
    spec = K.SETS[lang]
    L = plan["langs"][lang]
    prod = K.set_dir(lang)
    sp = K.selection_path(lang)
    out = {"lang": lang, "prod_dir": prod, "selection": sp, "problems": [], "warnings": []}
    if not os.path.isfile(sp):
        out["problems"].append(f"selection がない: {sp}（GPU の作り直しが済んでいない）")
        return out
    sel = K.read_json(sp)
    if sel.get("status") != "done":
        out["problems"].append(f"selection の status={sel.get('status')}（done でない）")
        return out
    if sel.get("targets") != L["targets"]:
        out["problems"].append("selection の対象が今の計画と違う（計画が作り直された）")
    chk = sel.get("check") or {}
    out["check"] = {"exact": chk.get("exact"), "chosen_tag": chk.get("chosen_tag"),
                    "accepted_mismatch": bool(not chk.get("exact") and accept_mismatch)}
    if not chk.get("exact"):
        msg = "seed 42 で作り直した確かめの対が本番と一致しなかった（参照文か環境が本番と違う可能性）"
        (out["warnings"] if accept_mismatch else out["problems"]).append(
            msg + ("（--accept_check_mismatch で続ける）" if accept_mismatch else "。--accept_check_mismatch がないと反映しない"))
    adopted = {int(i): a for i, a in sel.get("adopted", {}).items()}
    unresolved = sorted(sel.get("unresolved_pairs", []))
    # 組ごとに丸ごと採用されていること（BL を共有する対を分けない）
    for u in L["units"]:
        got = [i in adopted for i in u]
        if any(got) and not all(got):
            out["problems"].append(f"組 {u} の一部だけが採用されている")
        if all(got) and len({adopted[i]["seed"] for i in u}) != 1:
            out["problems"].append(f"組 {u} の seed がそろっていない")
    pairs = sorted(adopted)
    if set(pairs) | set(unresolved) != set(L["targets"]) or set(pairs) & set(unresolved):
        out["problems"].append("採用 + 未解決 が計画の対象と一致しない")
    files = []
    seed_rows = {}
    for i in pairs:
        a = adopted[i]
        sdir = K.seed_dir(a["seed"], lang)
        if a["seed"] not in K.SEEDS:
            out["problems"].append(f"対 {i} の seed {a['seed']} が 43/44 でない")
        if sdir not in seed_rows:
            ck = K.read_json(os.path.join(sdir, spec["ckpt"] if spec["gen"] == "Z" else "checkpoint.json"))
            seed_rows[sdir] = {r["pair_idx"]: r for r in ck[spec["rows_key"]] if not r.get("placeholder")}
        row = seed_rows[sdir].get(i)
        if row is None or row != a["row"]:
            out["problems"].append(f"対 {i}: 作り直しの checkpoint の行が selection の記録と違う")
        for side in K.SIDES:
            n = K.wav_name(lang, i, side)
            src = os.path.join(sdir, n)
            dst = os.path.join(prod, n)
            rec_md5 = (a["files"].get(side) or {}).get("md5")
            new = K.md5_file(src) if os.path.isfile(src) else None
            if new is None or new != rec_md5:
                out["problems"].append(f"{src}: md5 が selection の記録と違う（{new} != {rec_md5}）")
            st = K.wav_check(src) if os.path.isfile(src) else {"bad": True, "why": "missing"}
            if st["bad"]:
                out["problems"].append(f"{src}: 使えない音声（{st.get('why')}）")
            old = K.md5_file(dst) if os.path.isfile(dst) else None
            planned_old = plan["per_wav_md5"][lang].get(n)
            if old != planned_old:
                out["problems"].append(f"{dst}: 本番の wav が計画の時点から変わっている（{old} != {planned_old}）")
            if os.path.islink(dst):
                out["problems"].append(f"{dst} が symlink")
            files.append({"file": n, "pair_idx": i, "side": side, "seed": a["seed"], "src": src, "dst": dst,
                          "old_md5_seed42": old, "new_md5": new})
    bdir = K.backup_dir(lang)
    rec_path = os.path.join(prod, K.PROD_RECORD_NAME)
    if os.path.exists(bdir):
        out["problems"].append(f"退避先が既にある（反映済み？）: {bdir}")
    if os.path.exists(rec_path):
        out["problems"].append(f"本番の記録が既にある（反映済み？）: {rec_path}")
    # JSON の入れ替え（メモリ上）
    new_docs, before, after = {}, None, None
    for jn in spec["json"]:
        p = os.path.join(prod, jn)
        doc = K.read_json(p)
        key = "rows" if "rows" in doc else "results"
        idx = {r["pair_idx"]: j for j, r in enumerate(doc[key])}
        if sorted(idx) != list(range(K.N_EXPECTED)):
            out["problems"].append(f"{p}: pair_idx が 0..299 でない")
            continue
        for i in pairs:
            row = seed_rows[K.seed_dir(adopted[i]["seed"], lang)].get(i)
            if row is not None:
                doc[key][idx[i]] = row
        if jn == "results.json":
            old_summary = doc["summary"]
            orig_rows = K.read_json(p)["rows"]
            if K.utmos_summary(orig_rows, old_summary) != old_summary:
                out["problems"].append(f"{p}: 集計の式が本番の summary を再現しない（書き換えない）")
            ck_rows = K.read_json(os.path.join(prod, "checkpoint.json"))["rows"]
            if orig_rows != ck_rows:
                out["warnings"].append("本番の results.json と checkpoint.json の行が元から違う")
            doc["summary"] = K.utmos_summary(doc["rows"], old_summary)
            before, after = old_summary, doc["summary"]
        if jn == "checkpoint_zh.json":
            before = K.zh_cjk_summary(K.read_json(p)["results"])
            after = K.zh_cjk_summary(doc["results"])
        new_docs[jn] = (doc, 2 if jn in ("results.json", "checkpoint_zh.json") else 1)
    out.update({"pairs_replaced": pairs, "unresolved_pairs_not_replaced": unresolved,
                "pairs_added_as_bl_group_member": [i for i in L["added_as_bl_group_member"] if i in pairs],
                "seeds": {str(i): adopted[i]["seed"] for i in pairs},
                "seed_independent_diff_vs_prod": {str(i): adopted[i].get("seed_independent_diff_vs_prod")
                                                  for i in pairs if adopted[i].get("seed_independent_diff_vs_prod")},
                "files": files, "backup": bdir, "record": rec_path,
                "summary_kind": "results.json summary (UTMOS)" if spec["gen"] == "E"
                else "eval_cjk_n300_zhv2 printed summary (not stored in production)",
                "summary_before": before, "summary_after": after,
                "stale_derived_not_recomputed": [os.path.join(K.REAL_ER, s) for s in K.STALE_DERIVED.get(lang, [])]})
    if lang == "zh":
        out["zh_injection_success_impact"] = K.zh_shift_impact(pairs)
        out["zh_injection_success_impact_all_planned"] = K.zh_shift_impact(L["targets"])
        out["after_apply"] = "appk_prepare_stage.py --stage_root <stage> を流し直すと、今の md5 を別名の manifest に残す"
    out["_new_docs"] = new_docs
    return out


def apply(lang, prep, plan):
    spec = K.SETS[lang]
    prod = prep["prod_dir"]
    bdir = prep["backup"]
    assert not os.path.exists(bdir), bdir
    os.makedirs(bdir)
    # 1) 退避
    for f in prep["files"]:
        shutil.copy2(f["dst"], os.path.join(bdir, f["file"]))
        assert K.md5_file(os.path.join(bdir, f["file"])) == f["old_md5_seed42"], f
    for jn in spec["json"]:
        shutil.copy2(os.path.join(prod, jn), os.path.join(bdir, jn))
    # 2) 差し替え
    for f in prep["files"]:
        K.assert_prod_write(f["dst"])
        shutil.copy2(f["src"], f["dst"])
        got = K.md5_file(f["dst"])
        assert got == f["new_md5"], (f, got)
    # 3) JSON
    for jn, (doc, indent) in prep["_new_docs"].items():
        dump_like(doc, os.path.join(prod, jn), indent)
    # 4) 記録
    rec = {k: v for k, v in prep.items() if not k.startswith("_") and k not in ("problems",)}
    rec.update({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": ("unusable pair at seed 42 (silent / missing / unreadable wav) in the appendix K N=300 set; "
                           "author decision (1). " + K.UNIT_DECISION[plan["unit"]]),
                "unit": plan["unit"], "plan": K.plan_path(), "silence_criterion": K.SILENCE_CRITERION,
                "restore": f"copy every file in {bdir} back to {prod}, then delete {prep['record']}"})
    K.assert_prod_write(prep["record"])
    with open(prep["record"], "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    K.write_json_atomic(rec, K.apply_record_copy(lang))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(K.REGEN_LANGS))
    ap.add_argument("--apply", action="store_true", help="付けないときは予行（何も書かない）")
    ap.add_argument("--accept_check_mismatch", action="store_true",
                    help="seed 42 の再現が本番と一致しなかった言語も反映する（著者が判断したときだけ）")
    args = ap.parse_args()
    plan = K.read_json(K.plan_path())
    langs = [l for l in args.langs.split(",") if l]
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "apply": args.apply, "er_root": K.ER, "langs": {}}
    rc = 0
    for lang in langs:
        if lang not in plan["langs"]:
            print(f"[{lang}] 計画にない（飛ばす）")
            continue
        if not plan["langs"][lang]["targets"]:
            print(f"[{lang}] 作り直しの対象なし")
            continue
        prep = prepare(lang, plan, args.accept_check_mismatch)
        view = {k: v for k, v in prep.items() if not k.startswith("_")}
        report["langs"][lang] = view
        print(f"[{lang}] replace {len(prep.get('pairs_replaced', []))} pairs "
              f"(seeds {prep.get('seeds')}), unresolved not replaced {prep.get('unresolved_pairs_not_replaced')}, "
              f"check {prep.get('check')}")
        for w in prep["warnings"]:
            print(f"    WARNING: {w}")
        for p in prep["problems"]:
            print(f"    PROBLEM: {p}")
        if prep.get("summary_before") is not None:
            print(f"    summary before: {json.dumps(prep['summary_before'], ensure_ascii=False)}")
            print(f"    summary after : {json.dumps(prep['summary_after'], ensure_ascii=False)}")
        if lang == "zh" and "zh_injection_success_impact" in prep:
            z = prep["zh_injection_success_impact"]
            print(f"    ZH 93.67 への影響（判定はやり直さない）: 今の成功に数えている対 {z['targets_counted_positive_now']}、"
                  f"判定し直すと {z['range_after_rejudge_pct'][0]}〜{z['range_after_rejudge_pct'][1]} %")
        if prep["problems"]:
            rc = 1
            continue
        if args.apply:
            rec = apply(lang, prep, plan)
            report["langs"][lang]["applied"] = True
            print(f"    APPLIED: {len(rec['files'])} wav, backup {rec['backup']}, record {rec['record']}")
    out = os.path.join(K.REGEN_ROOT, f"apply_{'run' if args.apply else 'dry_run'}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    K.write_json_atomic(report, out)
    print(f"-> {out}")
    if not args.apply:
        print("予行のみ（何も書き換えていない）。反映は --apply。")
    sys.exit(rc)


if __name__ == "__main__":
    main()
