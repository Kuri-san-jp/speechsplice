#!/usr/bin/env python3
"""付録 G 声まね: 本番の Baseline / Text Hint の wav（eval_results/baseline_text_edit_ja_n300/wavs）のうち、
無効だった単位を seed 43/44 で作り直した wav に差し替える（2026-09-29 作成、CPU のみ）.

D2: 本番のファイルの差し替えは、この apply 段だけで行う（著者に見せたあと、リードが流す）。
  既定は計画を出すだけ（何も書かない）。--commit を付けたときだけ書く。
差し替える wav: appg_vc_common.build_selection で seed 43/44 が選ばれた textedit の単位の、作り直した wav だけ。
  既定（--textedit_unit b7、著者の決定 2026-09-29）は b7（Text Hint）だけ。本番の bl（Baseline）・b1 は差し替えない
  （計画の中で bl・b1 に差し替えが要るとなっていたら、何も書かずに止まる）。
  --textedit_unit bl_b1_b7 なら bl・b1・b7 の 3 本ごと（以前の決め方）。
  seed 42 で有効な単位は差し替えない（有効かどうかだけで決まっている）。check42 は選ばないので差し替えにも使わない。
書く前に確かめること（1 つでも当たれば何も書かない）:
  作り直しがすべて終わっている（selection が complete）、本番の wav が走査のときの元の wav のまま（md5）、
  作り直しの wav が記録の md5 のまま、symlink でない、hard link でない（st_nlink == 1）、退避先に同じ名前が無い、
  一時ファイルの名前が無い、置き場に書ける、空き容量。
書き方:
  1. 退避: 差し替える本番の wav（既定は b7 だけ）を <root>/apply/backup_textedit/ へ写し、md5 を確かめる（元の中身はここに残る。消さない）。
  2. 作り直しの wav を本番の置き場の一時ファイル（.sNNNN_xx.wav.staging_<run_id>）へ写し、md5 を確かめる。
  3. ロックを取り、本番が計画のときから変わっていないことを確かめてから os.replace で置き換える。途中で失敗したら退避から戻す。
  4. 記録: <root>/apply/applied_textedit.json（累積。前の版は .prev_ に写して残す）、<root>/apply/apply_textedit_<run_id>.json、
     本番の置き場の一つ上に REPLACED_WAVS_appg_vc_<run_id>.json（どの wav を差し替えたか。results.json の書き起こしは
     差し替えた対では古いままであることも書く。results.json と checkpoint.json は変えない）。
  差し替えたあとの走査は、差し替えた対の seed 42 の判定を退避した元の wav で行う（appg_vc_common.assess_unit）。

使い方（ホストで直接、CPU のみ）:
  計画だけ: taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_apply_textedit.py
  書く:     taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_apply_textedit.py --commit
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
sys.modules["pyopenjtalk"] = None

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_vc_common as V  # noqa: E402

def keys_to_replace():
    """差し替える wav の鍵（作り直しの段で作った wav = regen_keys。既定は b7 だけ）。"""
    return V.regen_keys("textedit")


def die(msg):
    raise SystemExit(f"[apply][guard] {msg}（何も書いていない）")


def make_plan(root):
    keys = keys_to_replace()
    scans = {m: V.scan_method(m, root) for m in V.METHOD_ORDER}
    sel = V.build_selection(root, scans)
    if not sel["complete"]:
        die(f"作り直しが終わっていない: {sel['problems']}")
    per42 = scans["textedit"]["_per"]["seed42"]
    applied = V.load_applied(root)
    bak_dir = os.path.join(V.apply_dir(root), "backup_textedit")
    prod_dir = V.baseline_dir()
    items = []
    for i in sel["textedit_needs_apply"]:
        u = sel["units"]["textedit"][str(i)]
        if u["seed"] == V.MAIN_SEED:
            die(f"対 {i}: seed 42 が選ばれているのに差し替えが要る（本番が変わっている？）")
        if u["tag"] == "check42":
            die(f"対 {i}: check42 が選ばれている（check42 は記録だけ）")
        if i in applied:
            die(f"対 {i}: 既に差し替えの記録があるのに本番が選んだ wav と違う")
        other = [k for k in V.method_keys("textedit") if k not in keys and u["files"][k]["needs_apply"]]
        if other:
            die(f"対 {i}: 作り直していない wav {other} に差し替えが要るとなっている（本番の bl / b1 が変わっている？）")
        orig = per42[i]
        if orig["state"] != "invalid":
            die(f"対 {i}: 本番（seed 42）で無効でない単位は差し替えない: {orig['state']}")
        files = {}
        for k in keys:
            prod = V.file_path("textedit", "seed42", k, i, root)
            src = V.proj_path(u["files"][k]["source_path"])
            files[k] = {"production": prod, "source": src, "backup": os.path.join(bak_dir, V.file_name("textedit", k, i)),
                        "old_md5": orig["files"][k]["md5"], "new_md5": u["files"][k]["md5"],
                        "old_stats": {x: orig["files"][k].get(x) for x in ("rms", "peak", "active_frac", "silent", "silent_reasons")}}
        items.append({"pair_idx": i, "seed": u["seed"], "tag": u["tag"], "invalid_reason": orig.get("silent_keys"),
                      "files": files})
    return {"root": root, "prod_dir": prod_dir, "bak_dir": bak_dir, "items": items, "selection": sel,
            "textedit_unit": V.textedit_unit(), "keys": list(keys)}


def precheck(plan, run_id):
    problems, need = [], 0
    for it in plan["items"]:
        for k, f in it["files"].items():
            for p in (f["production"], f["source"]):
                if os.path.islink(p):
                    problems.append(f"symlink は扱わない: {p}")
            if not os.path.isfile(f["production"]):
                problems.append(f"本番の wav が無い: {f['production']}")
                continue
            if os.stat(f["production"]).st_nlink > 1:
                problems.append(f"hard link の wav は上書きしない（他の置き場も変わる）: {f['production']}")
            if V.md5_file(f["production"]) != f["old_md5"]:
                problems.append(f"本番の wav が走査のときと違う: {f['production']}")
            if V.md5_file(f["source"]) != f["new_md5"]:
                problems.append(f"作り直しの wav が記録と違う: {f['source']}")
            if os.path.lexists(f["backup"]):
                problems.append(f"退避先に同じ名前がある: {f['backup']}")
            st = os.path.join(plan["prod_dir"], f".{os.path.basename(f['production'])}.staging_{run_id}")
            if os.path.lexists(st):
                problems.append(f"一時ファイルの名前が既にある: {st}")
            need += os.path.getsize(f["source"]) + os.path.getsize(f["production"])
    if plan["items"]:
        if not os.access(plan["prod_dir"], os.W_OK):
            problems.append(f"本番の置き場に書けない: {plan['prod_dir']}")
        free = shutil.disk_usage(plan["prod_dir"]).free
        if free < 2 * need + (100 << 20):
            problems.append(f"空き容量が足りない: free={free} need~{need}")
    return problems


def printable(plan):
    keys = plan["keys"]
    if keys == ["b7"]:
        effect = ("Only the Text Hint (b7) wavs of these pairs are replaced. The Baseline (bl) and b1 wavs are NOT "
                  "changed (author decision 2026-09-29: b7-only regeneration)")
    else:
        effect = ("Baseline (bl) and Text Hint (b7) rows use the replaced wavs for these pairs; b1 is replaced too "
                  "(same unit) but is not in the table")
    return {"production_dir": V.rel_to_proj(plan["prod_dir"]), "backup_dir": V.rel_to_proj(plan["bak_dir"]),
            "textedit_unit": plan["textedit_unit"], "replaced_keys": keys,
            "n_units": len(plan["items"]), "n_files": sum(len(it["files"]) for it in plan["items"]),
            "units": [{"pair_idx": it["pair_idx"], "seed": it["seed"], "invalid_reason_at_seed42": it["invalid_reason"],
                       "files": {k: {"production": V.rel_to_proj(f["production"]), "source": V.rel_to_proj(f["source"]),
                                     "old_md5": f["old_md5"], "new_md5": f["new_md5"],
                                     "old_silent": f["old_stats"]["silent"]}
                                 for k, f in it["files"].items()}} for it in plan["items"]],
            "effect": effect}


def copy_verified(src, dst, md5):
    shutil.copy2(src, dst)
    got = V.md5_file(dst)
    if got != md5:
        raise RuntimeError(f"copy md5 mismatch: {src} -> {dst} ({got} != {md5})")


def commit(plan, run_id):
    root, prod_dir, bak_dir = plan["root"], plan["prod_dir"], plan["bak_dir"]
    ad = V.apply_dir(root)
    os.makedirs(ad, exist_ok=True)
    lock_f = open(os.path.join(ad, ".apply.lock"), "a")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        die("別の apply が動いている（.apply.lock）")
    for it in plan["items"]:
        for f in it["files"].values():
            if V.md5_file(f["production"]) != f["old_md5"]:
                die(f"計画のあとで本番の wav が変わった: {f['production']}")
    # 1. 退避
    os.makedirs(bak_dir, exist_ok=True)
    for it in plan["items"]:
        for f in it["files"].values():
            copy_verified(f["production"], f["backup"], f["old_md5"])
    # 2. 本番の置き場の一時ファイルへ
    staged = []
    for it in plan["items"]:
        for f in it["files"].values():
            st = os.path.join(prod_dir, f".{os.path.basename(f['production'])}.staging_{run_id}")
            copy_verified(f["source"], st, f["new_md5"])
            staged.append((st, f))
    # 3. 記録を一時ファイルへ
    old = V.load_json(V.applied_path(root)) or {"pairs": {}, "history": []}
    new = json.loads(json.dumps(old))
    for it in plan["items"]:
        new["pairs"][str(it["pair_idx"])] = {
            "seed": it["seed"], "tag": it["tag"], "apply_run_id": run_id, "date": V.now(),
            "textedit_unit": plan["textedit_unit"], "replaced_keys": sorted(it["files"]),
            "files": {k: {"old_md5": f["old_md5"], "new_md5": f["new_md5"], "backup": V.rel_to_proj(f["backup"]),
                          "source": V.rel_to_proj(f["source"]), "production": V.rel_to_proj(f["production"])}
                      for k, f in it["files"].items()}}
    new["history"].append({"apply_run_id": run_id, "date": V.now(), "pairs": [it["pair_idx"] for it in plan["items"]]})
    record = {"apply_run_id": run_id, "date": V.now(), "reason": "D2: invalid Baseline/Text Hint unit (silent wav or "
              "generation failure by silence_criterion) replaced by the first valid of seed 43 / 44 (validity only); "
              f"replaced wavs: {plan['keys']} (textedit_unit={plan['textedit_unit']})",
              "silence_criterion": V.AC.criterion_doc(), **printable(plan)}
    ap_path = V.applied_path(root)
    rec_path = os.path.join(ad, f"apply_textedit_{run_id}.json")
    side = os.path.join(os.path.dirname(prod_dir.rstrip("/")), f"REPLACED_WAVS_appg_vc_{run_id}.json")
    if os.path.lexists(side) or os.path.lexists(rec_path):
        die("記録の名前が既にある")
    tmp_ap = f"{ap_path}.tmp_{run_id}"
    with open(tmp_ap, "w") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=1)
    # 4. 置き換え（途中で落ちたら退避から戻す）
    done = []
    try:
        for st, f in staged:
            os.replace(st, f["production"])
            done.append(f)
    except Exception as e:  # noqa: BLE001
        for f in done:
            rb = f"{f['production']}.rollback_{run_id}"
            shutil.copy2(f["backup"], rb)
            os.replace(rb, f["production"])
        raise SystemExit(f"[apply] 置き換えの途中で失敗したので戻した（{len(done)} 本）: {e!r}。"
                         f"一時ファイルは消していない（本番の名前ではない）")
    if os.path.exists(ap_path):  # 前の版を残す（上書きで失わない）
        shutil.copy2(ap_path, f"{ap_path}.prev_{run_id}")
    os.replace(tmp_ap, ap_path)
    V.atomic_write_json(rec_path, record)
    V.atomic_write_json(side, {"note": "These wavs were replaced by scripts/regen2/appg_vc_apply_textedit.py "
                                       "(Appendix G voice-clone redo, author decision D2). The originals are in the backup. "
                                       "results.json / checkpoint.json in this folder were NOT changed, so their "
                                       "transcripts for these pairs describe the original (replaced) wavs.",
                               "record": V.rel_to_proj(rec_path), **printable(plan)})
    for it in plan["items"]:
        for f in it["files"].values():
            if V.md5_file(f["production"]) != f["new_md5"]:
                raise SystemExit(f"[apply] 差し替えたあとの md5 が違う: {f['production']}")
    fcntl.flock(lock_f, fcntl.LOCK_UN)
    lock_f.close()
    return rec_path, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=V.OUT_ROOT_REL)
    ap.add_argument("--commit", action="store_true", help="付けたときだけ書く（既定は計画を出すだけ）")
    ap.add_argument("--textedit_unit", default=V.TEXTEDIT_UNIT_DEFAULT, choices=sorted(V.TEXTEDIT_UNITS),
                    help="差し替える wav（既定 b7 だけ。本番の bl・b1 は変えない）")
    args = ap.parse_args()
    V.set_textedit_unit(args.textedit_unit)
    print(f"[apply] 差し替える wav: {keys_to_replace()}（--textedit_unit {V.textedit_unit()}）")
    root = V.root_path(args.output_root)
    run_id = V.new_run_id("apply")
    plan = make_plan(root)
    problems = precheck(plan, run_id)
    info = {"apply_run_id": run_id, "commit": args.commit, "precheck_problems": problems, **printable(plan)}
    print(json.dumps(info, ensure_ascii=False, indent=1))
    pd = os.path.join(V.apply_dir(root), f"plan_{run_id}.json")
    os.makedirs(os.path.dirname(pd), exist_ok=True)
    V.atomic_write_json(pd, info)
    print(f"[apply] 計画を保存: {pd}")
    if problems:
        die(f"書く前の確認で問題: {problems}")
    if not plan["items"]:
        print("[apply] 差し替える単位が無い。何も書いていない")
        return
    if not args.commit:
        print("[apply] 計画だけ（--commit なし）。本番は変えていない")
        return
    rec, side = commit(plan, run_id)
    n_files = sum(len(it["files"]) for it in plan["items"])
    print(f"[apply] done: {len(plan['items'])} 単位（{n_files} 本、{plan['keys']}）を差し替えた。記録 {rec} / {side}")
    sel = V.build_selection(root)
    print(f"[apply] 差し替えのあと: textedit_needs_apply={sel.get('textedit_needs_apply')} "
          f"ready_for_transcription={sel.get('ready_for_transcription')}（appg_vc_select.py をもう一度流す）")


if __name__ == "__main__":
    main()
