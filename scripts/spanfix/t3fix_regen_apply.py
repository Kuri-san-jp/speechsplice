#!/usr/bin/env python3
"""t3fix (ii) 差し替え（CPU のみ）: t3fix_regen_gpu.py が作り直した対を、本番の Table 3 のセルに差し替える.

対象（著者の決定 2026-10-02、(3)）:
  * EN 対 112,118,128,138,141,158,238: 掃引 5 点（_layerabl/t5gemma_en_L{0,7,14,21,25}）の BL・TGT・INJ。
    BL・TGT は今、_cc_wavs_b/t5gemma_en・掃引 5 点・t5gemma_en_prog_L*to25 の 10 か所で同じ inode（nlink 10）。
    掃引 5 点の名前だけを新しいファイルに付け替える（os.replace）。_cc_wavs_b と prog の 5 か所は古い中身のまま残る。
    凍結した (a)（_cc_wavs_a/t5gemma_en、45.33）は別の inode で、触らない（digest 0c9b56e5… を前後で確かめる）。
  * ZH 対 196: (a)（_cc_wavs_a/t5gemma_zh）と掃引 5 点の BL・TGT（6 か所で同じ inode、nlink 6）・INJ。
採点（Table 3 の値）:
  * v1 PIB: 作り直しのジョブが GPU で採点した行（phoneme_per.py、1 行も変えず）で、本番の採点ファイルの対象の行だけを
    入れ替える。EN は _v1L_t5gemma_en_L{0,7,14,21,25}.json、ZH は _v1L_t5gemma_zh_L*.json と (a) の _v1all_t5gemma_zh.json。
  * whole MCD（Table 3 では (a) だけ）: ZH 対 196 の (a) の BL・TGT・INJ の写しに mcd_pysptk.py --mode whole を
    ホストの CPU で回し、_caseSPTK_whole_t5gemma_zh.json の行を入れ替える。掃引の MCD は本番に無い（(b) は PIB だけ）。
  * summary は regen2_common.pib_summary / mcd_summary（元の式）。Table 3 の ± は t3fix_common.pib_cell / mcd_cell。
  ほかの対の行は変えない（各対の採点は他の対に依存しない）。

ZH の (a) の順序: B1・B2（t3fix_b12_apply.py。対 15,145,246,295）と対 196 は対が重ならない。どちらも「その時の本番の
  ファイルの、自分の対の行だけ」を入れ替えるので、最終の値は順序に依らない。勧める順は B2 → GPU → この差し替え。
  予行では、B2 を反映済みかどうかと、その状態での前後の値を表示する。逆の順（この差し替えの後に B2）でも、
  t3fix_b12_apply.py の式の自己確認は反映の段階を見るので止まらない（t3fix_common.expected_table3）。

予行（既定）: 何も書き換えない。eval_results/_t3fix/regen_apply/dryrun_<日時>/ に計画を書く。
  GPU の結果（_t3fix/regen/final/final_meta.json）がまだ無いときは、差し替えの一覧（inode・nlink）と、
  作り直した対の PIB が成功・失敗のどちらになった場合の範囲（上下限）を表示する。
--apply（承認の後）: 確認がすべて通ったときだけ、
  0. 入れ替え後の採点ファイル（行を入れ替え、summary を計算し直し、"t3fix_regen" の記録を足したもの）を
     apply_<日時>/prepared/ に書き、帳面 _t3fix/regen_apply/apply_journal.json（音声 123 と採点ファイル 12 の操作、
     書き換える前の md5・inode、退避の置き場、_cc_wavs_b と prog と凍結した EN (a) の今の状態）を書く
  1. 退避: 書き換える音声（パスごと）と採点ファイルを _t3fix/regen_apply/backup_before_apply/ に写す
  2. 差し替え: final の音声と入れ替え後の採点ファイルを新しいファイルとして _t3fix/regen_apply/tmp に作り、
     os.replace で本番の名前を付け替える（既存のファイルや hard link を通して書くことは決してない。1 件ごとに
     md5・新しい inode・nlink 1 と、_cc_wavs_b と prog の中身と inode が変わっていないことを確かめる）
  3. Table 3 の値を書いた後のファイルから計算し直して計画と一致すること、凍結した EN (a) の digest が変わらない
     ことを確かめ、apply_record.json を書く
途中で止まったとき（帳面があり記録が無い）: 予行は止まった状態（済み・残り）を示すだけ。同じ --apply で、帳面の
  とおりに残りの操作だけを行う（済んだ操作は md5 で見分ける。元の中身でも新しい中身でもないファイルが 1 つでも
  あれば何もせずに止まる。_cc_wavs_b・prog・凍結した EN (a) が帳面の状態と違っても止まる）。
元に戻したいとき（このスクリプトはしない。著者の判断）: 帳面の ops の各 dest について、backup（退避）を同じ
  置き場の一時ファイルに写してから os.replace で dest の名前を付け替える（t3fix_common.replace_with_copy と同じ
  やり方）。EN の BL・TGT は、戻すと掃引 5 点がそれぞれ別の inode になる（hard link は張り直さない。中身は元と同じ）。
  _t3fix/regen_apply/ の記録は残るので、反映の段階の判定は「反映済み」のままになる。その扱いも著者が決める。
確認（どれかが外れたら書かない）: unresolved が無い、seed 44 は seed 43 で使えなかった対だけ、seed 42 の再現が一致
  （外れたら --accept_check_mismatch が要る）、採点器の確認が一致（--accept_scorer_mismatch）、生成のコードの md5 が
  GPU の実行のときと同じ（--accept_code_drift）、本番の対象のファイルと行が GPU の実行の前の snapshot と同じ、
  final の音声が使える（無音でない）こと、まだ反映していないこと。

使い方（ホストで直接）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/t3fix_regen_apply.py            # 予行
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/t3fix_regen_apply.py --apply    # 反映
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import t3fix_common as T  # noqa: E402

LAUNCH_ENV = T.setup_env()

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402

import t3fix_regen_gpu as G  # noqa: E402  （prod_snapshot / snapshot_diff だけを使う。GPU は使わない）

LANGS = ("en", "zh")
GEN_CODE = ("scripts/spanfix/t3fix_common.py", "scripts/spanfix/t3fix_regen_gpu.py",
            "scripts/spanfix/spanfix_common.py", "scripts/regen2/regen2_common.py")
EXPECTED_NLINK = {("en", "bl"): 10, ("en", "tgt"): 10, ("en", "inj"): 1, ("zh", "bl"): 6, ("zh", "tgt"): 6, ("zh", "inj"): 1}


def apply_dir(ev):
    return os.path.join(T.out_root(ev), "regen_apply")


def wav_ops(ev, fm):
    ops = []
    for lang in LANGS:
        for p in T.REGEN_TARGETS[lang]:
            for cell in T.regen_cells(lang):
                for c in ("bl", "tgt", "inj"):
                    name = c if c != "inj" else "inj_%s" % cell["cell"]
                    dest = T.prod_wav(ev, cell, p, c)
                    st = T.stat_info(dest)
                    op = {"lang": lang, "pair": p, "cell": cell["cell"], "lstar": cell["lstar"], "cond": c,
                          "dest": dest, "dest_stat": st, "expected_nlink": EXPECTED_NLINK[(lang, c)],
                          "src_name": name, "src": None}
                    f = ((((fm or {}).get("wavs") or {}).get(lang) or {}).get(str(p)) or {}).get("files", {}).get(name)
                    if f is not None:
                        op["src"] = os.path.join(ev, f["file"])
                        op["src_md5"] = f["md5"]
                    ops.append(op)
    return ops


def protected_watch(ev):
    """変えてはいけない、対象の対の同じ inode の仲間（EN の _cc_wavs_b と prog）."""
    paths = []
    for p in T.REGEN_TARGETS["en"]:
        for c in ("bl", "tgt", "inj"):
            paths.append(os.path.join(ev, T.EN_B_DIR, "s%04d_%s.wav" % (p, c)))
        for d in T.EN_PROG_DIRS:
            for c in ("bl", "tgt"):
                q = os.path.join(ev, d, "s%04d_%s.wav" % (p, c))
                if os.path.exists(q):
                    paths.append(q)
    return {os.path.relpath(q, ev): T.stat_info(q) for q in paths}


def score_files():
    out = []
    for lang in LANGS:
        for cell in T.regen_cells(lang):
            out.append((lang, cell, "pib", cell["pib"]))
            if cell.get("mcd"):
                out.append((lang, cell, "mcd", cell["mcd"]))
    return out


def b2_state(ev):
    rec = os.path.join(T.out_root(ev), "b12", "apply_record.json")
    return {"applied": os.path.exists(rec), "record": rec}


def bounds_view(ev):
    """GPU の前: 作り直した対の PIB が全部成功・全部失敗の場合の範囲と、対象の対を除いた値（参考）."""
    out = {}
    for lang in LANGS:
        tg = set(T.REGEN_TARGETS[lang])
        for cell in T.regen_cells(lang):
            d = T.read_json(os.path.join(ev, cell["pib"]))
            s = d["summary"]
            rows = {r["pair_idx"]: r for r in d["rows"]}
            k, n = s["inj_better_n"], s["n_valid"]
            k_t = sum(1 for p in tg if rows[p].get("inj_better"))
            kmin, kmax = k - k_t, k - k_t + len(tg)
            excl = [r for p, r in rows.items() if p not in tg and T.pib_row_ok(r)]
            ke = sum(1 for r in excl if r.get("inj_better"))
            out["%s %s (l*=%d)" % (lang.upper(), cell["cell"], cell["lstar"])] = {
                "now": "%d/%d = %.2f" % (k, n, s["inj_better_pct"]),
                "targets_now_inj_better": k_t,
                "after_bounds": "%.2f..%.2f" % (100.0 * kmin / n, 100.0 * kmax / n),
                "excluding_targets": "%d/%d = %.2f" % (ke, len(excl), 100.0 * ke / len(excl))}
    return out


def plan(ev, args, problems, work):
    fin_p = os.path.join(G.regen_base(ev), "final", "final_meta.json")
    fm = T.read_json(fin_p) if os.path.exists(fin_p) else None
    pl = {"final_meta": fin_p, "final_exists": fm is not None, "ops": wav_ops(ev, fm), "after_docs": {},
          "planned_dests": [], "checks": {}}
    for op in pl["ops"]:
        pl["planned_dests"].append(os.path.abspath(op["dest"]))
        if op["dest_stat"]["nlink"] != op["expected_nlink"]:
            pl.setdefault("nlink_notes", []).append("%s: nlink %d（想定 %d。os.replace なのでほかの hard link は変わらない）"
                                                    % (os.path.relpath(op["dest"], ev), op["dest_stat"]["nlink"],
                                                       op["expected_nlink"]))
    for lang, cell, kind, rel in score_files():
        pl["planned_dests"].append(os.path.abspath(os.path.join(ev, rel)))
    if fm is None:
        pl["bounds"] = bounds_view(ev)
        return pl
    # ---- GPU の結果の確認
    if any(fm.get("unresolved", {}).values()):
        problems.append("seed 44 でも使えない対がある: %s（著者の判断）" % fm["unresolved"])
    for lang in LANGS:
        for p in T.REGEN_TARGETS[lang]:
            s = fm["chosen_seed"].get(lang, {}).get(str(p))
            v = fm["validity"].get(lang, {}).get(str(p), {})
            if s not in T.SEEDS:
                problems.append("%s 対 %d: 採用した seed が 43/44 でない（%s）" % (lang, p, s))
                continue
            if not (v.get(str(s)) or {}).get("valid"):
                problems.append("%s 対 %d: 採用した seed %s が使えると記録されていない" % (lang, p, s))
            if s == 44 and (v.get("43") or {}).get("valid") is not False:
                problems.append("%s 対 %d: seed 43 が使えないと記録されていないのに 44 を採用" % (lang, p))
    if problems:   # 採用した seed が決まっていない対があれば、差し替えの計画は作らない
        return pl
    chk = fm.get("check42") or {}
    pl["checks"]["check42_match_all"] = chk.get("match_all")
    if not chk.get("match_all") and not args.accept_check_mismatch:
        problems.append("seed 42 の再現が一致しない（%s/%s）。認めるなら --accept_check_mismatch" % (chk.get("n_match"),
                                                                                   chk.get("n_files")))
    sc = fm.get("scorer_check") or {}
    pl["checks"]["scorer_check"] = sc
    if not sc or not all(v.get("all_equal") for v in sc.values()):
        if not args.accept_scorer_mismatch:
            problems.append("採点器の確認（確認の対の行が本番と同じ）が通っていない: %s。認めるなら --accept_scorer_mismatch" % sc)
    drift = [rel for rel in GEN_CODE if (fm.get("code_md5") or {}).get(rel) != T.md5_file(T.P(rel))]
    pl["checks"]["code_drift"] = drift
    if drift and not args.accept_code_drift:
        problems.append("生成のコードの md5 が GPU の実行のときと違う: %s。認めるなら --accept_code_drift" % drift)
    perr = T.code_pin_errors()
    if perr:
        problems += perr
    if fm.get("mode") != "gpu" and not T.is_sandbox(ev):
        problems.append("本番の置き場に、GPU で作っていない結果（mode=%s）は入れない" % fm.get("mode"))
    snap_p = os.path.join(ev, fm["prod_snapshot"])
    diff = G.snapshot_diff(T.read_json(snap_p), G.prod_snapshot(ev))
    pl["checks"]["snapshot_diff"] = diff
    if diff:
        problems.append("本番の対象のファイル・行が GPU の実行の前と違う: %s" % diff[:10])
    for op in pl["ops"]:
        if not os.path.exists(op["src"]) or T.md5_file(op["src"]) != op["src_md5"]:
            problems.append("final の音声が記録と違う: %s" % op["src"])
        elif T.wav_check(op["src"]).get("bad"):
            problems.append("final の音声が使えない（無音など）: %s" % op["src"])
    # ---- 採点ファイル（メモリの上で入れ替える）
    stg = os.path.join(work, "mcd_staging", "zh")
    mcd_rows = {}
    for lang, cell, kind, rel in score_files():
        doc = T.read_json(os.path.join(ev, rel))
        pairs = list(T.REGEN_TARGETS[lang])
        if kind == "pib":
            if not T.summaries_equal(T.RC.pib_summary(doc["rows"], doc["summary"]), doc["summary"]):
                problems.append("%s: pib_summary が今の summary を再現しない" % rel)
            new = {}
            for p in pairs:
                r = fm["rows"][lang][cell["cell"]][str(p)]
                if not T.pib_row_ok(r):
                    problems.append("%s 対 %d: 新しい v1 の行が使えない: %s" % (rel, p, r))
                new[p] = copy.deepcopy(r)
            rows = T.merge_rows(doc, new, pairs)
            after = copy.deepcopy(doc)
            after["rows"], after["summary"] = rows, T.RC.pib_summary(rows, doc["summary"])
            if after["summary"]["n_valid"] != 300:
                problems.append("%s: 入れ替え後の n_valid=%s（300 でない）" % (rel, after["summary"]["n_valid"]))
        else:   # ZH (a) の whole MCD
            if not T.summaries_equal(T.RC.mcd_summary(doc["rows"], doc["summary"]), doc["summary"]):
                problems.append("%s: mcd_summary が今の summary を再現しない" % rel)
            for p in pairs:
                files = fm["wavs"][lang][str(p)]["files"]
                for name, dst in (("bl", "bl"), ("tgt", "tgt"), ("inj_%s" % cell["cell"], "inj")):
                    T.copy_new(os.path.join(ev, files[name]["file"]), os.path.join(stg, "s%04d_%s.wav" % (p, dst)))
            mcd_rows = T.run_mcd_cpu(stg, lang, os.path.join(work, "mcd_zh_A.json"), os.path.join(work, "mcd_zh_A.log"))
            for p in pairs:
                if not T.mcd_row_ok(mcd_rows.get(p)):
                    problems.append("%s 対 %d: MCD の行が出ない: %s" % (rel, p, mcd_rows.get(p)))
            rows = T.merge_rows(doc, {p: mcd_rows[p] for p in pairs if p in mcd_rows}, pairs) if all(
                p in mcd_rows for p in pairs) else doc["rows"]
            after = copy.deepcopy(doc)
            after["rows"], after["summary"] = rows, T.RC.mcd_summary(rows, doc["summary"])
            if after["summary"]["n_valid"] != 300:
                problems.append("%s: 入れ替え後の n_valid=%s（300 でない）" % (rel, after["summary"]["n_valid"]))
        pl["after_docs"][rel] = after
        old = {r["pair_idx"]: r for r in doc["rows"]}
        pl.setdefault("row_changes", {})[rel] = {str(p): {"old": old.get(p), "new": {x["pair_idx"]: x for x in
                                                                                    after["rows"]}.get(p)} for p in pairs}
    return pl


def protected_same(a, b):
    """_cc_wavs_b と prog の同じ inode の仲間の比較（中身と inode。nlink は差し替えで減るので見ない）."""
    return sorted(a) == sorted(b) and all(a[k]["md5"] == b[k]["md5"] and a[k]["ino"] == b[k]["ino"] for k in a)


def finish(args, ev, base, j, work, resumed):
    """帳面のとおりに退避と差し替えを行い（済んだものは飛ばす）、確かめて記録を書く."""
    jp = T.apply_paths(base)
    prot0, fz0 = j["protected_before"], j["frozen_before"]
    if not protected_same(prot0, protected_watch(ev)):
        raise SystemExit("[regen_apply] _cc_wavs_b か prog のファイルが帳面の状態と違う")
    if T.frozen_en_a_digest(ev) != fz0:
        raise SystemExit("[regen_apply] 凍結した EN (a) の digest が帳面と違う")
    if not T.is_sandbox(ev) and not fz0["digest"].startswith(T.FROZEN_DIGEST_PREFIX):
        raise SystemExit("[regen_apply] 凍結した EN (a) の digest が記録と違う: %s" % fz0)

    def after_op(o):   # 毎回確かめる（hard link を通して書いていないこと）
        prot = protected_watch(ev)
        if not protected_same(prot0, prot):
            raise SystemExit("[regen_apply] 変えてはいけないファイルが変わった: %s"
                             % [k for k in prot0 if prot0[k]["md5"] != prot[k]["md5"] or prot0[k]["ino"] != prot[k]["ino"]])
    results = T.run_journal(j, ev, base, after_op=after_op, stop_after=args.selftest_stop_after_ops or None)
    after_real = T.jsonable(T.table3_view(ev))
    fz1 = T.frozen_en_a_digest(ev)
    prot1 = protected_watch(ev)
    rec = {"date": T.now(), "step": "regen_apply", "ev": ev, "backup": j["backup"], "journal": jp["journal"],
           "resumed": resumed, "run_dir": work, "plan": j.get("plan"), "final_meta": j["final_meta"],
           "b2_state": j["b2_state"],
           "ops": [{k: o.get(k) for k in ("i", "kind", "lang", "pair", "cell", "cond", "rel", "dest", "src", "src_md5",
                                          "orig_md5", "orig_ino", "orig_nlink", "backup")} for o in j["ops"]],
           "results": results, "before": j["before"], "after_planned": j["after_planned"], "after": after_real,
           "frozen_before": fz0, "frozen_after": fz1, "protected_before": prot0, "protected_after": prot1,
           "after_equals_plan": after_real == j["after_planned"], "frozen_unchanged": fz0 == fz1,
           "protected_unchanged": protected_same(prot0, prot1), "host": T.host_state(), "code_md5": T.code_md5_record()}
    T.write_json_new(rec, jp["record"])
    print("\n".join(T.view_lines(j["before"], after_real, "applied: Table 3 T5Gemma-TTS regenerated pairs")))
    if not (rec["after_equals_plan"] and rec["frozen_unchanged"] and rec["protected_unchanged"]):
        raise SystemExit("[regen_apply] 反映の後の確認に失敗（記録 %s）" % jp["record"])
    print("[regen_apply] 反映した%s。記録 %s、退避 %s" % ("（途中から続けた）" if resumed else "", jp["record"], j["backup"]))


def resume(args, ev, base, work):
    """帳面があり記録が無い = 反映が途中で止まった。予行は状態を示すだけ、--apply は残りを行う."""
    jp = T.apply_paths(base)
    j = T.read_json(jp["journal"])
    states = T.journal_states(j)
    cnt = {s: states.count(s) for s in ("todo", "done", "unknown")}
    T.write_json_new({"mode": "interrupted_apply_" + ("resume" if args.apply else "dry_run"), "ev": ev,
                      "journal": jp["journal"], "states": cnt,
                      "per_op": [{"dest": o["dest"], "state": s} for o, s in zip(j["ops"], states)]},
                     os.path.join(work, "plan.json"))
    print("[regen_apply] 反映が途中で止まっている（帳面 %s）: 済み %d、残り %d、不明 %d"
          % (jp["journal"], cnt["done"], cnt["todo"], cnt["unknown"]))
    if not args.apply:
        print("[regen_apply] 予行。同じ --apply で、帳面のとおりに残りの操作だけを行う。")
        return
    finish(args, ev, base, j, work, resumed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sandbox", default=None)
    ap.add_argument("--accept_check_mismatch", action="store_true")
    ap.add_argument("--accept_scorer_mismatch", action="store_true")
    ap.add_argument("--accept_code_drift", action="store_true")
    ap.add_argument("--selftest_stop_after_ops", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    ev = T.ev_root(args.sandbox)
    if args.selftest_stop_after_ops and not T.is_sandbox(ev):
        raise SystemExit("[regen_apply] --selftest_stop_after_ops は試験の写しの中だけ")
    base = apply_dir(ev)
    jp = T.apply_paths(base)
    rec_p, bak = jp["record"], jp["backup"]
    work = os.path.join(base, "%s_%s_%d" % ("apply" if args.apply else "dryrun", T.stamp(), os.getpid()))
    os.makedirs(work)
    print("[regen_apply] ev=%s mode=%s" % (ev, "apply" if args.apply else "dry_run"), flush=True)
    if os.path.exists(rec_p):
        print("[regen_apply] 反映済み（%s）。何もしない。" % rec_p)
        T.write_json_new({"mode": "already_applied", "record": rec_p}, os.path.join(work, "plan.json"))
        return
    if os.path.exists(jp["journal"]):
        return resume(args, ev, base, work)
    problems = []
    b2 = b2_state(ev)
    before = T.table3_view(ev)
    pl = plan(ev, args, problems, work)
    view = {"mode": "apply" if args.apply else "dry_run", "ev": ev, "problems": problems, "b2_state": b2,
            "final_exists": pl["final_exists"], "checks": pl["checks"], "nlink_notes": pl.get("nlink_notes"),
            "ops": [{k: o[k] for k in ("lang", "pair", "cell", "lstar", "cond", "dest", "src", "dest_stat",
                                       "expected_nlink")} for o in pl["ops"]],
            "protected_watch": protected_watch(ev), "frozen_en_a": T.frozen_en_a_digest(ev),
            "manuscript_check": T.manuscript_check(), "before": before, "code_md5": T.code_md5_record(),
            "order_note": "ZH (a): B2（対 15,145,246,295）と対 196 は重ならない。勧める順は B2 → GPU → この差し替え。"
                          "B2 の反映: %s" % ("済み" if b2["applied"] else "まだ")}
    if not pl["final_exists"]:
        view["bounds"] = pl["bounds"]
        T.write_json_new(view, os.path.join(work, "plan.json"))
        print("[regen_apply] GPU の結果（%s）がまだ無い。差し替えの予定だけを示す。" % pl["final_meta"])
        print("[regen_apply] 書き換える音声 %d 本（EN 7 対 × 掃引 5 点 × 3、ZH 196 × 6 セル × 3）、採点ファイル %d 個"
              % (len(pl["ops"]), len(score_files())))
        for o in pl["ops"][:3] + pl["ops"][-3:]:
            print("   %s -> nlink %d (想定 %d)" % (os.path.relpath(o["dest"], ev), o["dest_stat"]["nlink"], o["expected_nlink"]))
        print("[regen_apply] B2 の反映: %s" % ("済み" if b2["applied"] else "まだ（勧める順: B2 → GPU → この差し替え）"))
        print("[regen_apply] 作り直した対の PIB が全部成功・全部失敗の場合の範囲（今 → 範囲、対象を除いた値は参考）:")
        for k, v in pl["bounds"].items():
            print("   %-14s now %-16s after in [%s]  (excluding targets: %s)" % (k, v["now"], v["after_bounds"],
                                                                              v["excluding_targets"]))
        if args.apply:
            raise SystemExit("[regen_apply] GPU の結果が無いので反映しない")
        print("[regen_apply] plan -> %s" % os.path.join(work, "plan.json"))
        return
    after_plan = T.table3_view(ev, docs=pl["after_docs"])
    view.update(after=after_plan, row_changes=pl.get("row_changes"))
    T.write_json_new(view, os.path.join(work, "plan.json"))
    lines = T.view_lines(before, after_plan, "Table 3 T5Gemma-TTS: regenerated EN sweep pairs %s and ZH pair %s (B2 %s)"
                         % (list(T.REGEN_TARGETS["en"]), list(T.REGEN_TARGETS["zh"]),
                            "already applied" if b2["applied"] else "not applied"))
    print("\n".join(lines))
    print("[regen_apply] problems: %s" % (json.dumps(problems, ensure_ascii=False) if problems else "none"))
    print("[regen_apply] plan -> %s" % os.path.join(work, "plan.json"))
    if not args.apply:
        print("[regen_apply] 予行。本番のファイルは何も変えていない。")
        return
    # ------------------------------------------------------------ 反映
    if problems:
        raise SystemExit("[regen_apply] 問題があるので反映しない")
    if os.path.exists(bak):
        raise SystemExit("[regen_apply] 退避の置き場が既にあるのに帳面が無い: %s（著者に報告する）" % bak)
    prot0 = protected_watch(ev)
    fz0 = T.frozen_en_a_digest(ev)
    if not T.is_sandbox(ev) and not fz0["digest"].startswith(T.FROZEN_DIGEST_PREFIX):
        raise SystemExit("[regen_apply] 凍結した EN (a) の digest が記録と違う: %s" % fz0)
    ops = [{"kind": "wav", "lang": o["lang"], "pair": o["pair"], "cell": o["cell"], "cond": o["cond"],
            "dest": o["dest"], "src": o["src"], "src_md5": o["src_md5"]} for o in pl["ops"]]
    seeds = T.read_json(pl["final_meta"])["chosen_seed"]
    for lang, cell, kind, rel in score_files():
        doc = pl["after_docs"][rel]
        doc["t3fix_regen"] = {"pairs": list(T.REGEN_TARGETS[lang]), "date": T.now(), "backup": bak,
                              "seeds": seeds[lang],
                              "source": ("v1 PIB rows scored on GPU by phoneme_per.py in t3fix_regen_gpu.py"
                                         if kind == "pib" else "mcd_pysptk.py --mode whole on host CPU"),
                              "note": "author decision 2026-10-02 (3): pairs regenerated as whole pairs with seed 43 "
                                      "then 44 (validity only); other rows unchanged; summary recomputed with the "
                                      "original formula"}
        prep = os.path.join(work, "prepared", rel)
        T.write_json_new(doc, prep, indent=2)
        ops.append({"kind": "doc", "lang": lang, "cell": cell["cell"], "rel": rel, "dest": os.path.join(ev, rel),
                    "src": prep})
    j = T.build_journal(ev, base, ops, {"step": "regen_apply", "before": T.jsonable(before),
                                        "after_planned": T.jsonable(after_plan), "frozen_before": fz0,
                                        "protected_before": prot0, "final_meta": pl["final_meta"], "b2_state": b2,
                                        "plan": os.path.join(work, "plan.json"), "code_md5": T.code_md5_record()})
    T.write_json_new(j, jp["journal"])
    finish(args, ev, base, j, work, resumed=False)


if __name__ == "__main__":
    main()
