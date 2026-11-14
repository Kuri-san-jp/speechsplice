#!/usr/bin/env python3
"""G5a: regen_multientity_t5gemma_seed.py で作り直した 2 実体の対を multientity_en_300 に差し替える（CPU のみ）.

precedent: scripts/nontemplate/apply_regen_pairs.py と同じ流れ。
1. 事前確認（何も書かない。1 つでも問題があれば退避も書き込みもしない）:
   _regen2_final/multientity_en_300/regen_meta.json、採用した seed が 43 か 44（44 は 43 が使えなかった対だけ）、
   採用した bl/inj* が使えること・tgt1/tgt2 があって読めること（regen2_common.wav_check）、
   作り直しの results.json・results_tgt.json・checkpoint.json・checkpoint_tgt.json に対象の対の行があること
   （条件もそろっていること）、本番の 4 ファイルが読めること、unresolved の対は差し替えない、
   seed 42 の再現確認（一致しないときは --accept_check_mismatch）、退避先と記録（regen2_seed43_pairs.json）が
   まだ無いこと、書き込み先が凍結した T5Gemma EN (a) でないこと。
   generate_multientity の summary の式で、既存の results.json / results_tgt.json の summary を
   1 桁も違わず再現できることも確かめる（自己検査）。
   書き込む 4 つの JSON（行の入れ替えと summary の再計算）は、ここでメモリの上に作っておく。
2. 退避: 対象の対の wav 6 本（bl, inj1, inj2, inj12, tgt1, tgt2）、results.json・results_tgt.json・
   checkpoint.json・checkpoint_tgt.json、採点（multientity_asr_scores.json・ceiling_asr.json・pib_multi.json）を
   eval_results/_regen2_seed43/original_seed42_backup/multientity_en_300/ に写す。
3. 作り直した wav を本番のファイルに上書きで書く（削除も rename もしない）。_v1view_* の symlink は
   本番の wav を指しているので、自動で新しい中身を指す。
4. 1 で作った 4 つの JSON を書く。results_tgt.json に行の無い対（0..2 は別の回で作ったため行が無い。
   対象では 2）は行を足さない（記録だけ）。
5. multientity_en_300/regen2_seed43_pairs.json に記録を書く。
Whisper の書き起こし直しの取り込みは merge_regen2_scores.py --kind asr で行う。

使い方（ホストで直接。スレッド 13・CPU 0-12・nice 10・GPU を見せない・HF オフラインはスクリプトの中で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/apply_regen2_multientity.py          # 予行
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/apply_regen2_multientity.py --apply  # 実行

--campaign thankyou（著者判断 2026-09-30。regen2_common.THANKYOU。main_thankyou）:
  _regen2_thankyou/final/multientity_en_300/（regen2_thankyou_meta.json）の対を差し替える。流れは上と同じで、違いは:
  * 事前確認: campaign の id、対ごとの seed が THANKYOU["seed_plan"] に従うこと（44 は 43 が使えないと記録された
    対と 257 だけ）、unresolved の対があればどの対も差し替えない、採った bl/inj* が無音でなく、final の書き起こし
    （判定に使った行）で "thankyou" に当たらず失敗も無いこと、対 257 の今の本番が G5a の seed 43 の音声であること、
    凍結した T5Gemma EN (a) の digest が 0c9b56e5... であること。seed 42 の再現確認は求めない（G5a の記録を写す）。
  * 退避: 対象の対の wav 6 本（257 は G5a の seed 43 の音声）、4 つの JSON、採点 3 つを
    eval_results/_regen2_thankyou/backup_before_thankyou/multientity_en_300/ に写す（md5 を記録）。
  * 行の regen2 に campaign と前の regen2（257 は G5a の seed 43）を残す。
  * 記録は multientity_en_300/regen2_thankyou_pairs.json（G5a の regen2_seed43_pairs.json は触らない）。
  * 予行でも実行でも、Table 3 (c)（Baseline / One / Both、k/600。final の書き起こしを入れた見込み）と
    UTMOS の平均（bl / inj1 / inj12。explanation_of_revisions.tex ~216 行）の前後を表示する。
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/apply_regen2_multientity.py --campaign thankyou
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/apply_regen2_multientity.py --campaign thankyou --apply
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import regen2_common as C  # noqa: E402

C.job_env()
C.host_cpu_setup()

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

M = C.MULTI
ALL_CONDS = M["conds"] + M["tgt_conds"]
META_FILES = ("results.json", "results_tgt.json", "checkpoint.json", "checkpoint_tgt.json")
META_CONDS = {"results.json": M["conds"], "checkpoint.json": M["conds"],
              "results_tgt.json": M["tgt_conds"], "checkpoint_tgt.json": M["tgt_conds"]}
SCORE_FILES = ("multientity_asr_scores.json", "ceiling_asr.json", "pib_multi.json")
ALLOWED_SEEDS = (43, 44)


def summarize(rows, old, conds, tgt_only):
    """generate_multientity.py 490-515 行と同じ式。既存の summary にあるキーだけを書き換える."""
    def _mean(key):
        v = [r["conds"][key]["utmos"] for r in rows if r["conds"].get(key, {}).get("utmos") is not None]
        return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] if v else [None, None]
    s = dict(old)
    s["n_generated"] = len(rows)
    s["utmos"] = {c: _mean(c) for c in conds}
    s["mean_dur_sec"] = {c: round(float(np.mean([r["conds"][c]["dur_sec"] for r in rows])), 4)
                         for c in conds} if rows else {}
    if "mean_abs_position_drift" in old:
        s["mean_abs_position_drift"] = None if tgt_only else (
            round(float(np.mean([abs(r["total_position_drift"]) for r in rows])), 3) if rows else None)
    return s


def selftest(prod):
    out = {}
    r = C.read_json(os.path.join(prod, "results.json"))
    out["results.json"] = summarize(r["rows"], r["summary"], M["conds"], False) == r["summary"]
    t = C.read_json(os.path.join(prod, "results_tgt.json"))
    out["results_tgt.json"] = summarize(t["rows"], t["summary"], M["tgt_conds"], True) == t["summary"]
    return out


def check_seed_choice(meta, problems):
    valid = meta.get("valid", {})
    for p, s in meta["chosen_seed"].items():
        if s is None:
            continue
        if s not in ALLOWED_SEEDS:
            problems.append("対 %s: 採用した seed %s が 43/44 でない" % (p, s))
        v = valid.get(str(p), {})
        if s == 44 and v.get("43") is not False:
            problems.append("対 %s: seed 43 が使えないと記録されていないのに seed 44 を採用している" % p)
        if v.get(str(s)) is not True:
            problems.append("対 %s: 採用した seed %s が使えると記録されていない" % (p, s))


def prepare_docs(prod, fin, pairs, chosen, problems, annotate=None):
    """作り直しの行がそろっているかを確かめ、書き込む 4 つの JSON をメモリの上で作る（何も書かない）.
    annotate(pair, old_row) が行の regen2 の値を返す（None なら G5a の値）."""
    fin_docs = {}
    for n in META_FILES:
        p = os.path.join(fin, n)
        if not os.path.exists(p):
            problems.append("作り直しの %s が無い" % p)
            continue
        fin_docs[n] = {r["pair_idx"]: r for r in C.read_json(p)["rows"]}
        for pidx in pairs:
            r = fin_docs[n].get(pidx)
            if r is None:
                problems.append("作り直しの %s に対 %d の行が無い" % (n, pidx))
            elif not set(META_CONDS[n]) <= set(r.get("conds", {})):
                problems.append("作り直しの %s の対 %d の条件がそろわない: %s" % (n, pidx, sorted(r.get("conds", {}))))
    writes, summaries, not_added = [], {}, {}
    if len(fin_docs) != len(META_FILES):
        return writes, summaries, not_added
    for n in META_FILES:
        pth = C.assert_writable_target(os.path.join(prod, n))
        if not os.path.exists(pth):
            problems.append("本番の %s が無い" % pth)
            continue
        doc = copy.deepcopy(C.read_json(pth))
        idx = {r["pair_idx"]: j for j, r in enumerate(doc["rows"])}
        for p in pairs:
            new_row = fin_docs[n].get(p)
            if new_row is None:
                continue
            if annotate is None:
                new_row = dict(new_row, regen2={"seed": chosen[p], "reason": "unusable bl/inj at seed 42 (G5a)"})
            else:
                new_row = dict(new_row, regen2=annotate(p, doc["rows"][idx[p]] if p in idx else None))
            if p in idx:
                doc["rows"][idx[p]] = new_row
            else:
                not_added.setdefault(n, []).append(p)
        if "summary" in doc:
            tgt = n == "results_tgt.json"
            before = doc["summary"]
            doc["summary"] = summarize(doc["rows"], before, M["tgt_conds"] if tgt else M["conds"], tgt)
            summaries[n] = {"before": before, "after": doc["summary"]}
        writes.append((pth, doc))
    # results_tgt.json に行を足さないのは、本番にもともと行の無い対（0..2）だけ
    for n, ps in not_added.items():
        bad = [p for p in ps if not (n.endswith("_tgt.json") and p in (0, 1, 2))]
        if bad:
            problems.append("本番の %s に対 %s の行が無い（0..2 の tgt 以外は想定外）" % (n, bad))
    n_rows = {os.path.basename(p): len(d["rows"]) for p, d in writes}
    if n_rows.get("results.json") != 300 or n_rows.get("checkpoint.json") != 300:
        problems.append("差し替え後の results.json / checkpoint.json の行が 300 でない: %s" % n_rows)
    return writes, summaries, not_added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="付けないときは予行（何も書かない）")
    ap.add_argument("--accept_check_mismatch", action="store_true",
                    help="seed 42 の再現確認で md5 が一致しなくても差し替える（bf16 の非決定性など。著者の了承が要る）")
    ap.add_argument("--campaign", default="g5a", choices=["g5a", "thankyou"],
                    help="g5a: 9/29 の無音の対（既定。これまでと同じ）、thankyou: 9/30 の「Thank you.」の 6 対")
    args = ap.parse_args()
    C.refuse_apply_in_test(args.apply)
    if args.campaign == "thankyou":
        main_thankyou(args)
        return
    prod = os.path.join(C.EV, M["name"])
    fin = C.multi_final_dir()
    bak = C.multi_backup_dir()
    problems = []
    meta_p = os.path.join(fin, "regen_meta.json")
    meta = C.read_json(meta_p) if os.path.exists(meta_p) else None
    if meta is None:
        chosen = {p: (None if args.apply else 43) for p in M["pairs"]}
        if args.apply:
            problems.append("作り直しがまだ無い: %s" % meta_p)
    else:
        chosen = {int(p): s for p, s in meta["chosen_seed"].items()}
        if sorted(chosen) != sorted(M["pairs"]):
            problems.append("regen_meta の対 %s が計画 %s と違う" % (sorted(chosen), M["pairs"]))
        check_seed_choice(meta, problems)
        chk = meta.get("check_seed42")
        if chk is None and not args.accept_check_mismatch:
            problems.append("seed 42 の再現確認が記録されていない（--no_check で作った。--accept_check_mismatch が要る）")
        elif chk is not None and not chk.get("match_all") and not args.accept_check_mismatch:
            problems.append("seed 42 の再現確認で md5 が一致しない（--accept_check_mismatch が要る）")
        for p in meta.get("unresolved_pairs", []):
            problems.append("対 %d は seed 43/44 とも使えない（差し替えない。著者判断待ち）" % p)
    pairs = sorted(p for p, s in chosen.items() if s is not None)

    st = selftest(prod)
    if not all(st.values()):
        problems.append("summary の再計算が既存と一致しない: %s" % st)

    ops, backups = [], []
    for p in pairs:
        for k in ALL_CONDS:
            n = "s%04d_%s.wav" % (p, k)
            dst = C.assert_writable_target(os.path.join(prod, n))
            src = os.path.join(fin, n)
            ops.append({"src": src, "dst": dst, "backup": os.path.join(bak, n), "nlink": os.stat(dst).st_nlink})
            if os.stat(dst).st_nlink != 1:
                problems.append("hard link がある（想定外）: %s" % dst)
            if meta is not None:
                w = C.wav_check(src)
                ops[-1]["src_check"] = {kk: w.get(kk) for kk in ("rms", "peak", "active_frac", "silent_reason", "why")}
                if k in M["conds"] and w["bad"]:
                    problems.append("作り直しの音声が使えない（%s）: %s" % (w.get("why"), src))
                elif k in M["tgt_conds"] and (w.get("missing") or w.get("unreadable")):
                    problems.append("作り直しの tgt が無い・読めない（%s）: %s" % (w.get("why"), src))
    for n in META_FILES + SCORE_FILES:
        pth = os.path.join(prod, n)
        if os.path.exists(pth):
            backups.append((pth, os.path.join(bak, "meta_" + n if n in META_FILES else n)))
    for _, b in [(o["dst"], o["backup"]) for o in ops] + backups:
        if os.path.exists(b):
            problems.append("退避先が既にある: %s" % b)
    out_rec = C.assert_writable_target(os.path.join(prod, "regen2_seed43_pairs.json"))
    if os.path.exists(out_rec):
        problems.append("差し替えの記録が既にある（差し替え済み）: %s" % out_rec)

    writes, summaries, not_added = ([], {}, {})
    if meta is not None:
        writes, summaries, not_added = prepare_docs(prod, fin, pairs, chosen, problems)

    rep = {"mode": "apply" if args.apply else "dry_run", "pairs": pairs, "chosen_seed": chosen,
           "criterion": C.SILENCE_CRITERION,
           "selftest_summary_recompute": st, "n_wav_ops": len(ops), "backup_dir": bak,
           "json_writes": [w[0] for w in writes], "rows_not_added": not_added,
           "problems": problems, "frozen_t5gemma_en_a_before": C.frozen_digest()}
    print(json.dumps({k: v for k, v in rep.items() if k != "frozen_t5gemma_en_a_before"}, ensure_ascii=False,
                     indent=1, default=str))
    C.write_json(dict(rep, ops=ops, backups=backups, summaries=summaries),
                 os.path.join(C.DRYRUN_DIR, "apply_multi_%s.json" % rep["mode"]))
    if not args.apply:
        return
    if problems:
        raise SystemExit("問題があるので差し替えない（何も書いていない）")

    # 2. 退避
    for src, dst in [(o["dst"], o["backup"]) for o in ops] + backups:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            raise SystemExit("退避先が既にある: %s" % dst)
        shutil.copy2(src, dst)
    # 3. 差し替え（上書き）
    files = []
    for o in ops:
        old = C.md5_file(o["backup"])
        shutil.copyfile(o["src"], o["dst"])
        new = C.md5_file(o["dst"])
        if new != C.md5_file(o["src"]):
            raise SystemExit("差し替え後の md5 が合わない: %s" % o["dst"])
        files.append({"file": os.path.basename(o["dst"]), "old_md5_seed42": old, "new_md5": new})
    # 4. 行と summary（事前に作った中身）
    for pth, doc in writes:
        C.write_json(doc, pth)
    # 5. 記録
    rec = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "item": "G5a", "table3": M["table3"],
           "pairs": pairs, "chosen_seed": {str(p): chosen[p] for p in pairs},
           "reason": "unusable bl/inj1/inj2/inj12 at seed 42 (silent wav; %s). Author decision 2026-09-28 extended "
                     "to Table 3 (G5a): regenerate the whole pair (6 wavs) with seed 43 (44 if still unusable: "
                     "silent/missing/unreadable bl or inj, missing/unreadable tgt, or a failed generation row)"
                     % C.SILENCE_RULE,
           "excluded_tgt_only_silent_pairs": M["tgt_only_silent_excluded"],
           "valid": meta.get("valid"), "check_seed42": meta.get("check_seed42"),
           "row_compare_vs_prod": meta.get("row_compare_vs_prod"),
           "final_dir": fin, "backup": bak, "files": files, "rows_not_added": not_added, "summaries": summaries,
           "scores_to_update": {"asr": M["asr_json"]},
           "stale_derived_not_updated": M["stale_derived"]}
    if os.path.exists(out_rec):
        raise SystemExit("記録が既にある: %s" % out_rec)
    C.write_json(rec, out_rec)
    after = C.frozen_digest()
    if after != rep["frozen_t5gemma_en_a_before"]:
        raise SystemExit("凍結した T5Gemma EN (a) が変わった（ありえない）")
    print("[apply] %d files replaced; record -> %s; frozen T5Gemma EN (a) unchanged" % (len(files), out_rec))


# ---------------------------------------------------------------------------
# --campaign thankyou（著者判断 2026-09-30）
# ---------------------------------------------------------------------------
def check_seed_choice_thankyou(meta, plan, problems):
    """採った seed が THANKYOU["seed_plan"] に従うか（前の seed は使えないと記録され、採った seed は使えると記録）."""
    valid = meta.get("valid", {})
    for p, s in meta["chosen_seed"].items():
        p = int(p)
        if s is None:
            continue
        seq = plan.get(p)
        if not seq or s not in seq:
            problems.append("対 %d: 採った seed %s が計画 %s に無い" % (p, s, seq))
            continue
        v = valid.get(str(p), {})
        for e in seq[:seq.index(s)]:
            if v.get(str(e)) is not False:
                problems.append("対 %d: seed %d が使えないと記録されていないのに seed %d を採っている" % (p, e, s))
        if v.get(str(s)) is not True:
            problems.append("対 %d: 採った seed %s が使えると記録されていない" % (p, s))


def main_thankyou(args):
    T = C.THANKYOU
    prod = os.path.join(C.EV, M["name"])
    fin = C.thankyou_final_dir()
    bak = C.thankyou_backup_dir()
    plan = {int(p): list(s) for p, s in T["seed_plan"].items()}
    problems = []
    fz0 = C.frozen_digest()
    if not fz0["digest"].startswith(T["frozen_digest_prefix"]):
        problems.append("凍結した T5Gemma EN (a) の digest %s が %s... でない" % (fz0["digest"], T["frozen_digest_prefix"]))
    meta_p = os.path.join(fin, T["meta_name"])
    meta = C.read_json(meta_p) if os.path.exists(meta_p) else None
    if meta is None:
        chosen = {p: (None if args.apply else plan[p][0]) for p in T["pairs"]}
        if args.apply:
            problems.append("作り直しがまだ無い: %s" % meta_p)
    else:
        chosen = {int(p): s for p, s in meta["chosen_seed"].items()}
        if meta.get("campaign") != T["id"]:
            problems.append("meta の campaign %s が %s でない" % (meta.get("campaign"), T["id"]))
        if sorted(chosen) != sorted(T["pairs"]):
            problems.append("meta の対 %s が計画 %s と違う" % (sorted(chosen), T["pairs"]))
        check_seed_choice_thankyou(meta, plan, problems)
        for p in meta.get("unresolved_pairs", []):
            problems.append("対 %d は計画の seed %s ではどれも使えない（どの対も差し替えない。著者に報告）"
                            % (p, plan.get(p)))
        if meta.get("frozen_unchanged") is not True:
            problems.append("GPU の回で凍結した T5Gemma EN (a) の不変が記録されていない")
    pairs = sorted(p for p, s in chosen.items() if s is not None)

    st = selftest(prod)
    if not all(st.values()):
        problems.append("summary の再計算が既存と一致しない: %s" % st)

    # 今の本番（差し替え前）: 書き起こしと Table 3 (c)、対 257 が G5a の seed 43 の音声であること
    prod_asr = C.read_json(os.path.join(prod, M["asr_json"]))
    prod_scan = C.thankyou_scan(prod_asr["rows"], set(T["pairs"]))
    t3_before = C.table3c_cells(prod_asr["rows"])
    g257 = {}
    for k in ALL_CONDS:
        n = "s%04d_%s.wav" % (257, k)
        b = os.path.join(C.G5A_MULTI_FINAL_REAL, n)
        g257[k] = os.path.exists(b) and C.md5_file(os.path.join(prod, n)) == C.md5_file(b)
    if not all(g257.values()):
        problems.append("対 257 の今の本番が G5a の seed 43 の音声でない: %s" % g257)
    prev_rows = C.read_rows(os.path.join(prod, "results.json"))

    # final の書き起こし（判定に使った行）: 失敗が無く、6 枠の判定がそろい、"thankyou" に当たらないこと
    asr_p = os.path.join(fin, T["asr_final_name"])
    new_asr = {r["pair_idx"]: r for r in C.read_json(asr_p)["rows"]} if os.path.exists(asr_p) else None
    t3_after = None
    if meta is not None:
        if new_asr is None:
            problems.append("final の書き起こしが無い: %s" % asr_p)
        else:
            for p in pairs:
                r = new_asr.get(p)
                if r is None:
                    problems.append("final の書き起こしに対 %d の行が無い" % p)
                    continue
                errk = sorted(k for k in r if k.endswith("_err"))
                slot_keys = ("bl_slot1_ok", "bl_slot2_ok", "inj1_slot1_ok", "inj2_slot2_ok",
                             "inj12_slot1_ok", "inj12_slot2_ok")
                badk = [k for k in slot_keys if not isinstance(r.get(k), bool)]
                ty = [k for k in M["conds"] if C.is_thankyou(r.get(k + "_transcript"))]
                if errk or badk:
                    problems.append("final の書き起こしの対 %d が失敗している（%s）" % (p, errk + badk))
                if ty:
                    problems.append("final の書き起こしの対 %d の %s が 'thankyou'（採ってはいけない）" % (p, ty))
            if all(p in new_asr for p in pairs):
                rows = [new_asr[r["pair_idx"]] if r["pair_idx"] in pairs else r for r in prod_asr["rows"]]
                t3_after = C.table3c_cells(rows)

    ops, backups = [], []
    for p in pairs:
        for k in ALL_CONDS:
            n = "s%04d_%s.wav" % (p, k)
            dst = C.assert_writable_target(os.path.join(prod, n))
            src = os.path.join(fin, n)
            ops.append({"src": src, "dst": dst, "backup": os.path.join(bak, n), "nlink": os.stat(dst).st_nlink,
                        "md5_prod_before": C.md5_file(dst)})
            if os.stat(dst).st_nlink != 1:
                problems.append("hard link がある（想定外）: %s" % dst)
            if meta is not None:
                w = C.wav_check(src)
                ops[-1]["src_check"] = {kk: w.get(kk) for kk in ("rms", "peak", "active_frac", "silent_reason", "why")}
                if k in M["conds"] and w["bad"]:
                    problems.append("作り直しの音声が使えない（%s）: %s" % (w.get("why"), src))
                elif k in M["tgt_conds"] and (w.get("missing") or w.get("unreadable")):
                    problems.append("作り直しの tgt が無い・読めない（%s）: %s" % (w.get("why"), src))
    for n in META_FILES + SCORE_FILES:
        pth = os.path.join(prod, n)
        if os.path.exists(pth):
            backups.append((pth, os.path.join(bak, "meta_" + n if n in META_FILES else n)))
    for _, b in [(o["dst"], o["backup"]) for o in ops] + backups:
        if os.path.exists(b):
            problems.append("退避先が既にある: %s" % b)
    out_rec = C.assert_writable_target(os.path.join(prod, T["record_pairs"]))
    if os.path.exists(out_rec):
        problems.append("差し替えの記録が既にある（差し替え済み）: %s" % out_rec)

    def annotate(p, old_row):
        return {"seed": chosen[p], "campaign": T["id"],
                "reason": "Whisper transcript 'Thank you.' only on %s at the previous seed (author decision "
                          "2026-09-30: generation failure; whole pair regenerated)"
                          % "/".join(T["flagged_prod_clips"].get(p, [])),
                "previous": (old_row or {}).get("regen2")}

    writes, summaries, not_added = ([], {}, {})
    if meta is not None:
        writes, summaries, not_added = prepare_docs(prod, fin, pairs, chosen, problems, annotate=annotate)
    ut_before = C.utmos_means(C.read_json(os.path.join(prod, "results.json"))["summary"])
    ut_after = C.utmos_means(summaries["results.json"]["after"]) if "results.json" in summaries else None

    rep = {"mode": "apply" if args.apply else "dry_run", "campaign": T["id"], "pairs": pairs,
           "chosen_seed": chosen, "seed_plan": {str(p): v for p, v in plan.items()},
           "unresolved_pairs": (meta or {}).get("unresolved_pairs"),
           "rule": T["rule"], "criterion": C.SILENCE_CRITERION,
           "selftest_summary_recompute": st, "n_wav_ops": len(ops), "backup_dir": bak,
           "json_writes": [w[0] for w in writes], "rows_not_added": not_added,
           "prod_thankyou_clips_now": {str(p): v for p, v in sorted(prod_scan.items())},
           "pair257_prod_is_g5a_seed43": g257,
           "table3c_before": t3_before, "table3c_after_projected": t3_after,
           "utmos_before": ut_before, "utmos_after": ut_after,
           "problems": problems, "frozen_t5gemma_en_a_before": fz0}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("frozen_t5gemma_en_a_before", "rule", "criterion")},
                     ensure_ascii=False, indent=1, default=str))
    print("\n".join(C.before_after_lines(t3_before, t3_after, ut_before, ut_after)))
    if t3_after is None or ut_after is None:
        print("  (after: n/a until the GPU job has written %s)" % meta_p)
    C.write_json(dict(rep, ops=ops, backups=backups, summaries=summaries),
                 os.path.join(C.thankyou_dryrun_dir(), "apply_thankyou_%s.json" % rep["mode"]))
    if not args.apply:
        print("[dry_run] problems=%d（本番には何も書いていない）" % len(problems))
        return
    if problems:
        raise SystemExit("問題があるので差し替えない（何も書いていない）")

    # 2. 退避（md5 を記録）
    backed = []
    for src, dst in [(o["dst"], o["backup"]) for o in ops] + backups:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            raise SystemExit("退避先が既にある: %s" % dst)
        shutil.copy2(src, dst)
        if C.md5_file(dst) != C.md5_file(src):
            raise SystemExit("退避の md5 が合わない: %s" % dst)
        backed.append({"src": src, "backup": dst, "md5": C.md5_file(dst)})
    # 3. 差し替え（上書き）
    files = []
    for o in ops:
        old = C.md5_file(o["backup"])
        shutil.copyfile(o["src"], o["dst"])
        new = C.md5_file(o["dst"])
        if new != C.md5_file(o["src"]):
            raise SystemExit("差し替え後の md5 が合わない: %s" % o["dst"])
        files.append({"file": os.path.basename(o["dst"]), "old_md5": old, "new_md5": new})
    # 4. 行と summary（事前に作った中身）
    for pth, doc in writes:
        C.write_json(doc, pth)
    # 5. 記録
    rec = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "campaign": T["id"], "table3": M["table3"],
           "decision": T["decision"], "rule": T["rule"], "normalization": T["normalization"],
           "silence_criterion": C.SILENCE_CRITERION, "pairs": pairs,
           "chosen_seed": {str(p): chosen[p] for p in pairs}, "seed_plan": {str(p): v for p, v in plan.items()},
           "valid": meta.get("valid"), "asr": meta.get("asr"), "check_seed42_reference": meta.get("check_seed42_reference"),
           "row_compare_vs_prod": meta.get("row_compare_vs_prod"),
           "prod_thankyou_clips_before": {str(p): v for p, v in sorted(prod_scan.items())},
           "previous_regen2": {str(p): (prev_rows.get(p) or {}).get("regen2") for p in pairs},
           "final_dir": fin, "backup": bak, "backups": backed, "files": files, "rows_not_added": not_added,
           "summaries": summaries, "utmos_before": ut_before, "utmos_after": ut_after,
           "table3c_before": t3_before, "table3c_after_projected": t3_after,
           "scores_to_update": {"asr": M["asr_json"] + " (merge_regen2_scores.py --campaign thankyou --kind asr)"},
           "stale_derived_not_updated": M["stale_derived"]}
    if os.path.exists(out_rec):
        raise SystemExit("記録が既にある: %s" % out_rec)
    C.write_json(rec, out_rec)
    after = C.frozen_digest()
    if after != fz0:
        raise SystemExit("凍結した T5Gemma EN (a) が変わった（ありえない）")
    print("[apply] %d files replaced; record -> %s; frozen T5Gemma EN (a) unchanged (%s)"
          % (len(files), out_rec, after["digest"]))


if __name__ == "__main__":
    main()
