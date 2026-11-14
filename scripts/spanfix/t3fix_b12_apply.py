#!/usr/bin/env python3
"""t3fix (i) B1・B2（CPU のみ）: Table 3 (a) T5Gemma-TTS JA・ZH の 4 対を l*=26 の音声に差し替える.

著者の決定（2026-10-02、(2b)）: (a) の JA・ZH は 5/17 に古い範囲の関数で作られ、JA 59,99,164,179 と
ZH 15,145,246,295 で差し替え・遮断の位置が誤っていた。この 4 対の INJ を、今の関数で作った l*=26 の掃引
（_layerabl/t5gemma_{ja,zh}_L25。BL・TGT は (a) と同じ inode）の INJ に差し替え、v1 PIB と whole MCD を
出し直す。ZH の対 30 も (a) と L25 で INJ が違うが、範囲とは関係ないので差し替えない（記録のみ）。

値の出し方（採点は元のスクリプトのまま。各対の採点は他の対に依存しない）:
  * v1 PIB: L25 の本番の採点 _v1L_t5gemma_{ja,zh}_L25.json の 4 対の行を使う。この行は同じ BL・TGT・INJ の
    3 本を同じ採点器（phoneme_per.py、GPU）で採点したもの。(a) と L25 の行で BL・TGT の IPA が 300 対すべて
    同じことを確かめる（採点がそろっている証拠）。ホストには espeak が無いので phoneme_per.py は回せない。
  * whole MCD: speechsplice_release/src/mcd_pysptk.py --mode whole（1 行も変えない）をホストの CPU で、
    4 対の BL・TGT・新しい INJ の写しに回す。監査（eval_results/_audit_spanB_20261002/mcd_whole_*_L25.json）の
    行と小数 4 桁まで一致することを求める。
  * summary は regen2_common.pib_summary / mcd_summary（元の式。今のファイルの summary を再現することを確かめる）。
  * Table 3 の ±: PIB は Wilson 95% の半幅、MCD は t_{0.975,n-1}·sd/√n（今の原稿の値を再現する式）。

予行（既定）: 何も書き換えない。eval_results/_t3fix/b12/dryrun_<日時>/ に MCD の写しと計画を書き、
  Table 3 の前後の値を表示する。
--apply: 予行と同じ確認をすべて通ったときだけ、
  0. 入れ替え後の採点ファイル 4 つ（_v1all_t5gemma_{ja,zh}.json、_caseSPTK_whole_t5gemma_{ja,zh}.json。4 対の行を
     入れ替え、summary を元の式で計算し直し、"t3fix_b12" の記録を足したもの）を apply_<日時>/prepared/ に書き、
     帳面 _t3fix/b12/apply_journal.json（12 の操作、書き換える前の md5・inode、退避の置き場）を書く
  1. 退避: (a) の INJ 8 本と採点ファイル 4 つを eval_results/_t3fix/b12/backup_before_apply/ に写す
  2. (a) の INJ 8 本と採点ファイル 4 つを差し替える: 新しいファイルを _t3fix/b12/tmp に作り os.replace
     （hard link を通して書かない。(a) の INJ は nlink 1。BL・TGT には触らない）
  3. 書いた後に、値・md5・inode・凍結した EN (a) の digest を確かめ、_t3fix/b12/apply_record.json を書く
  原稿（naacl2027/ ほか）には書かない。
途中で止まったとき（帳面があり記録が無い）: 予行は止まった状態（済み・残り）を示すだけ。同じ --apply で、帳面の
  とおりに残りの操作だけを行う（済んだ操作は md5 で見分ける。元の中身でも新しい中身でもないファイルが 1 つでも
  あれば何もせずに止まる）。計画は作り直さない（帳面の中身を使う）。
元に戻したいとき（このスクリプトはしない。著者の判断）: 帳面の ops の各 dest について、backup（退避）を同じ
  置き場の一時ファイルに写してから os.replace で dest の名前を付け替える（t3fix_common.replace_with_copy と同じ
  やり方。hard link を通して書かない）。戻した後も _t3fix/b12/ の記録は残るので、反映の段階の判定
  （t3fix_common.apply_state）は「反映済み」のままになる。その扱いも著者が決める。

ZH の (a) は、GPU で作り直す対 196（t3fix_regen_apply.py）でも変わる。2 つの作業は対が重ならず、どちらも
そのときの本番のファイルの自分の対の行だけを入れ替えるので、順序で最後の値は変わらない。勧める順は B2 → GPU →
対 196 の差し替え。逆の順でも動く: このスクリプトの式の自己確認は反映の段階を見て、作り直しの反映の後は、
その記録の値と照合する（t3fix_common.expected_table3）。

使い方（ホストで直接。スレッド 13・CPU 0-12・nice 10・GPU を見せない・HF オフラインはスクリプトの中で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/t3fix_b12_apply.py            # 予行
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/t3fix_b12_apply.py --apply    # 反映（承認の後）
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

LANGS = ("ja", "zh")
# 監査（eval_results/_audit_spanB_20261002/summary.json）の差し替え後の値。一致しなければ止める
AUDIT_EXPECT = {"ja": {"pib_k": 156, "mcd_mean": 8.2446, "mcd_half": 0.2053},
                "zh": {"pib_k": 164, "mcd_mean": 8.9528, "mcd_half": 0.2477}}


def b12_dir(ev):
    return os.path.join(T.out_root(ev), "b12")


def plan(ev, work, problems):
    """何も書き換えずに、差し替えの計画と差し替えた後の採点ファイル（メモリの上）を作る."""
    pl = {"langs": {}, "after_docs": {}, "planned_dests": []}
    for lang in LANGS:
        pairs = T.B12_PAIRS[lang]
        a_dir = os.path.join(ev, T.A_DIR[lang])
        l_dir = os.path.join(ev, T.sweep_dir(lang, 25))
        lp = {"pairs": list(pairs), "wav_ops": [], "states": {}}
        for p in pairs:
            ops = {"pair": p}
            for c in ("bl", "tgt"):
                a, l = os.path.join(a_dir, "s%04d_%s.wav" % (p, c)), os.path.join(l_dir, "s%04d_%s.wav" % (p, c))
                sa, sl = T.stat_info(a), T.stat_info(l)
                ops["%s_same_inode" % c] = (sa["ino"], sa["dev"]) == (sl["ino"], sl["dev"])
                ops["%s_same_md5" % c] = sa["md5"] == sl["md5"]
                ops["%s_a_stat" % c] = sa
                if not ops["%s_same_md5" % c]:
                    problems.append("%s 対 %d: (a) と L25 の %s が違う" % (lang, p, c))
                if not ops["%s_same_inode" % c]:   # 試験の写しも同じ形に作る（t3fix_selftest.build_sandbox）
                    problems.append("%s 対 %d: (a) と L25 の %s が同じ inode でない（本番では同じはず）" % (lang, p, c))
            dest, src = os.path.join(a_dir, "s%04d_inj.wav" % p), os.path.join(l_dir, "s%04d_inj.wav" % p)
            sd, ss = T.stat_info(dest), T.stat_info(src)
            pin_a, pin_l = T.B12_PINS[lang][p]
            state = "original" if sd["md5"] == pin_a else ("applied" if sd["md5"] == pin_l else "unknown")
            if ss["md5"] != pin_l:
                problems.append("%s 対 %d: L25 の INJ の md5 が記録と違う（%s）" % (lang, p, ss["md5"]))
            if state == "unknown":
                problems.append("%s 対 %d: (a) の INJ が、元の音声とも L25 とも違う（%s）" % (lang, p, sd["md5"]))
            if sd["nlink"] != 1 and state == "original":
                problems.append("%s 対 %d: (a) の INJ の nlink が 1 でない（%d）" % (lang, p, sd["nlink"]))
            ops.update({"dest": dest, "src": src, "dest_stat": sd, "src_stat": ss, "state": state})
            lp["wav_ops"].append(ops)
            lp["states"][p] = state
            pl["planned_dests"].append(os.path.abspath(dest))
        # ---- v1 PIB: L25 の本番の行
        a_pib = T.read_json(os.path.join(ev, T.A_PIB[lang]))
        l_pib = T.read_json(os.path.join(ev, T.sweep_pib(lang, 25)))
        ar = {r["pair_idx"]: r for r in a_pib["rows"]}
        lr = {r["pair_idx"]: r for r in l_pib["rows"]}
        same_bt = [p for p in range(300) if ar[p].get("bl_ent_ipa") == lr[p].get("bl_ent_ipa")
                   and ar[p].get("tgt_ent_ipa") == lr[p].get("tgt_ent_ipa") and ar[p].get("available") and lr[p].get("available")]
        inj_diff = [p for p in range(300) if ar[p].get("inj_ent_ipa") != lr[p].get("inj_ent_ipa")]
        lp["pib_bl_tgt_ipa_equal_pairs"] = len(same_bt)
        lp["pib_inj_ipa_differ_pairs"] = inj_diff
        if len(same_bt) != 300:
            problems.append("%s: (a) と L25 の v1 の行で BL・TGT の IPA が一致しない対がある（%d/300）" % (lang, len(same_bt)))
        new_pib = {}
        for p in pairs:
            if not T.pib_row_ok(lr[p]):
                problems.append("%s 対 %d: L25 の v1 の行が使えない: %s" % (lang, p, lr[p]))
            new_pib[p] = copy.deepcopy(lr[p])
        lp["pib_rows_old"] = {str(p): ar[p] for p in pairs}
        lp["pib_rows_new"] = {str(p): new_pib[p] for p in pairs}
        rows = T.merge_rows(a_pib, new_pib, pairs)
        summ = T.RC.pib_summary(rows, a_pib["summary"])
        if not T.summaries_equal(T.RC.pib_summary(a_pib["rows"], a_pib["summary"]), a_pib["summary"]):
            problems.append("%s: pib_summary が今の %s の summary を再現しない" % (lang, T.A_PIB[lang]))
        pdoc = copy.deepcopy(a_pib)
        pdoc["rows"], pdoc["summary"] = rows, summ
        # ---- whole MCD: ホストの CPU で 4 対を採点し直す
        stg = os.path.join(work, "mcd_staging", lang)
        for op in lp["wav_ops"]:
            p = op["pair"]
            T.copy_new(os.path.join(a_dir, "s%04d_bl.wav" % p), os.path.join(stg, "s%04d_bl.wav" % p))
            T.copy_new(os.path.join(a_dir, "s%04d_tgt.wav" % p), os.path.join(stg, "s%04d_tgt.wav" % p))
            T.copy_new(op["src"], os.path.join(stg, "s%04d_inj.wav" % p))
        mrows = T.run_mcd_cpu(stg, lang, os.path.join(work, "mcd_%s.json" % lang),
                              os.path.join(work, "mcd_%s.log" % lang))
        audit = {r["pair_idx"]: r for r in T.read_json(os.path.join(ev, T.AUDIT_MCD_L25[lang]))["rows"]}
        lp["mcd_vs_audit"] = {}
        for p in pairs:
            r = mrows.get(p)
            if not T.mcd_row_ok(r):
                problems.append("%s 対 %d: MCD の行が出ない: %s" % (lang, p, r))
                continue
            eq = all(r[k] == audit[p].get(k) for k in ("mcd_inj_vs_bl", "mcd_inj_vs_tgt", "mcd_bl_vs_tgt"))
            lp["mcd_vs_audit"][str(p)] = eq
            if not eq:
                problems.append("%s 対 %d: MCD が監査の行と違う（%s / %s）" % (lang, p, r, audit[p]))
        a_mcd = T.read_json(os.path.join(ev, T.A_MCD[lang]))
        if not T.summaries_equal(T.RC.mcd_summary(a_mcd["rows"], a_mcd["summary"]), a_mcd["summary"]):
            problems.append("%s: mcd_summary が今の %s の summary を再現しない" % (lang, T.A_MCD[lang]))
        old_m = {r["pair_idx"]: r for r in a_mcd["rows"]}
        for p in pairs:   # BL–TGT は差し替えで変わらないはず
            if p in mrows and mrows[p].get("mcd_bl_vs_tgt") != old_m[p].get("mcd_bl_vs_tgt"):
                problems.append("%s 対 %d: mcd_bl_vs_tgt が今の行と違う（BL・TGT が同じなら同じはず）" % (lang, p))
        lp["mcd_rows_old"] = {str(p): old_m[p] for p in pairs}
        lp["mcd_rows_new"] = {str(p): mrows.get(p) for p in pairs}
        mrows_m = T.merge_rows(a_mcd, {p: mrows[p] for p in pairs if p in mrows}, pairs) if all(
            p in mrows for p in pairs) else a_mcd["rows"]
        mdoc = copy.deepcopy(a_mcd)
        mdoc["rows"], mdoc["summary"] = mrows_m, T.RC.mcd_summary(mrows_m, a_mcd["summary"])
        pl["after_docs"][T.A_PIB[lang]] = pdoc
        pl["after_docs"][T.A_MCD[lang]] = mdoc
        pl["planned_dests"] += [os.path.abspath(os.path.join(ev, T.A_PIB[lang])),
                                os.path.abspath(os.path.join(ev, T.A_MCD[lang]))]
        # ---- 監査の値との一致（セル全体の値は、ほかの行が元のままのときだけ比べられる。ZH の対 196 を
        #      t3fix_regen_apply.py で先に差し替えた場合は、4 対の行が監査と同じことだけを確かめる（上の mcd_vs_audit）
        mc = T.mcd_cell(mrows_m)
        exp = AUDIT_EXPECT[lang]
        regen_applied = os.path.exists(os.path.join(T.out_root(ev), "regen_apply", "apply_record.json"))
        if lang == "zh" and regen_applied:
            lp["audit_match"] = "n/a: ZH pair 196 already replaced by t3fix_regen_apply.py; only the 4 rows are compared"
        else:
            lp["audit_match"] = {"pib_k": summ["inj_better_n"] == exp["pib_k"],
                                 "mcd_mean": round(mc["mean"], 4) == exp["mcd_mean"],
                                 "mcd_half": round(mc["half"], 4) == exp["mcd_half"]}
        if isinstance(lp["audit_match"], dict) and not all(lp["audit_match"].values()):
            problems.append("%s: 差し替え後の値が監査の値と違う %s（PIB k=%d、MCD %.4f ± %.4f）"
                            % (lang, lp["audit_match"], summ["inj_better_n"], mc["mean"], mc["half"]))
        pl["langs"][lang] = lp
    return pl


def keep_check(o):
    """差し替えた (a) の INJ と同じ対の BL・TGT（(a) と掃引 5 点で同じ inode）と、写した元の L25 の INJ が
    変わっていないこと."""
    if o["kind"] != "wav":
        return
    for c, k in o["keep"].items():
        st = T.stat_info(k["path"])
        if st["ino"] != k["ino"] or st["md5"] != k["md5"]:
            raise SystemExit("[b12] %s 対 %d の %s が変わった" % (o["lang"], o["pair"], c))
    if T.md5_file(o["src"]) != o["src_md5"]:
        raise SystemExit("[b12] L25 の INJ が変わった: %s" % o["src"])


def finish(args, ev, base, j, work, resumed):
    """帳面のとおりに退避と差し替えを行い（済んだものは飛ばす）、確かめて記録を書く."""
    jp = T.apply_paths(base)
    fz0 = j["frozen_before"]
    fz_now = T.frozen_en_a_digest(ev)
    if fz_now != fz0:
        raise SystemExit("[b12] 凍結した EN (a) の digest が帳面と違う: %s / %s" % (fz_now, fz0))
    if not T.is_sandbox(ev) and not fz0["digest"].startswith(T.FROZEN_DIGEST_PREFIX):
        raise SystemExit("[b12] 凍結した EN (a) の digest が記録と違う: %s" % fz0)
    results = T.run_journal(j, ev, base, after_op=keep_check, stop_after=args.selftest_stop_after_ops or None)
    for o in j["ops"]:   # 前の実行で済んだ操作も含めて、もう一度
        keep_check(o)
    after_real = T.jsonable(T.table3_view(ev))
    fz1 = T.frozen_en_a_digest(ev)
    rec = {"date": T.now(), "step": "b12", "ev": ev, "backup": j["backup"], "journal": jp["journal"],
           "resumed": resumed, "run_dir": work, "plan": j.get("plan"),
           "ops": [{k: o.get(k) for k in ("i", "kind", "lang", "pair", "rel", "dest", "src", "src_md5", "orig_md5",
                                          "orig_ino", "orig_nlink", "backup")} for o in j["ops"]],
           "results": results, "frozen_before": fz0, "frozen_after": fz1,
           "before": j["before"], "after_planned": j["after_planned"], "after": after_real,
           "after_equals_plan": after_real == j["after_planned"], "frozen_unchanged": fz0 == fz1,
           "code_md5": T.code_md5_record(), "host": T.host_state()}
    T.write_json_new(rec, jp["record"])
    print("\n".join(T.view_lines(j["before"], after_real, "applied: Table 3 (a) T5Gemma-TTS, B1/B2")))
    if not (rec["after_equals_plan"] and rec["frozen_unchanged"]):
        raise SystemExit("[b12] 反映の後の確認に失敗（記録 %s）" % jp["record"])
    print("[b12] 反映した%s。記録 %s、退避 %s" % ("（途中から続けた）" if resumed else "", jp["record"], j["backup"]))


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
    print("[b12] 反映が途中で止まっている（帳面 %s）: 済み %d、残り %d、不明 %d"
          % (jp["journal"], cnt["done"], cnt["todo"], cnt["unknown"]))
    if not args.apply:
        print("[b12] 予行。同じ --apply で、帳面のとおりに残りの操作だけを行う。")
        return
    finish(args, ev, base, j, work, resumed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="確認がすべて通ったときだけ本番に反映する（承認の後）")
    ap.add_argument("--sandbox", default=None, help="試験用の写し（eval_results/_t3fix/selftest/ の下）")
    ap.add_argument("--selftest_stop_after_ops", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    ev = T.ev_root(args.sandbox)
    if args.selftest_stop_after_ops and not T.is_sandbox(ev):
        raise SystemExit("[b12] --selftest_stop_after_ops は試験の写しの中だけ")
    base = b12_dir(ev)
    jp = T.apply_paths(base)
    rec_p, bak = jp["record"], jp["backup"]
    work = os.path.join(base, "%s_%s_%d" % ("apply" if args.apply else "dryrun", T.stamp(), os.getpid()))
    os.makedirs(work)
    problems = []
    print("[b12] ev=%s mode=%s host=%s" % (ev, "apply" if args.apply else "dry_run",
                                          json.dumps(T.host_state(), ensure_ascii=False)), flush=True)
    if not os.path.exists(rec_p) and os.path.exists(jp["journal"]):
        return resume(args, ev, base, work)
    code_errs = T.code_pin_errors()
    if code_errs:
        problems += code_errs
    # 式と表の値の自己確認（反映の段階を見る。試験の写しでも同じに行う）
    fst = T.formula_selftest(ev)
    if not fst["ok"]:
        problems.append("式・表の値の自己確認に失敗: %s" % T.formula_fail_reasons(fst))
    already = os.path.exists(rec_p)
    before_view = T.table3_view(ev)
    pl = plan(ev, work, problems)
    states = {lang: sorted(set(pl["langs"][lang]["states"].values())) for lang in LANGS}
    if already or any(s == ["applied"] for s in states.values()):
        if already and all(s == ["applied"] for s in states.values()):
            print("[b12] 反映済み（%s）。何もしない。" % rec_p)
            T.write_json_new({"mode": "already_applied", "record": rec_p, "states": states}, os.path.join(work, "plan.json"))
            return
        problems.append("反映の状態がそろっていない（記録 %s、音声の状態 %s）" % (already, states))
    after_view = T.table3_view(ev, docs=pl["after_docs"])
    lines = T.view_lines(before_view, after_view, "Table 3 (a) T5Gemma-TTS, B1/B2 (4 pairs each -> l*=26 INJ)")
    view = {"mode": "apply" if args.apply else "dry_run", "ev": ev, "problems": problems,
            "formula_selftest": fst, "manuscript_check": T.manuscript_check(), "states": states,
            "before": before_view, "after": after_view,
            "langs": {lang: {k: v for k, v in lp.items() if k not in ("wav_ops",)} for lang, lp in pl["langs"].items()},
            "wav_ops": {lang: [{k: o[k] for k in ("pair", "dest", "src", "state", "bl_same_inode", "tgt_same_inode")}
                               for o in pl["langs"][lang]["wav_ops"]] for lang in LANGS},
            "zh_pair30_note": "(a) と L25 で INJ が違うが範囲とは関係ない。差し替えない",
            "apply_state": T.apply_state(ev),
            "order_note": "ZH (a) は対 196 の作り直し（t3fix_regen_apply.py）でも変わる。対が重ならず、どちらも"
                          "そのときの本番の行のうち自分の対だけを入れ替えるので、順序で最後の値は変わらない。"
                          "勧める順は B2 → GPU → 196 の差し替え",
            "frozen_en_a": T.frozen_en_a_digest(ev), "code_md5": T.code_md5_record()}
    T.write_json_new(view, os.path.join(work, "plan.json"))
    print("\n".join(lines))
    print("[b12] problems: %s" % (json.dumps(problems, ensure_ascii=False) if problems else "none"))
    print("[b12] plan -> %s" % os.path.join(work, "plan.json"))
    if not args.apply:
        print("[b12] 予行。本番のファイルは何も変えていない。反映は --apply（承認の後）。")
        return
    # ------------------------------------------------------------ 反映
    if problems:
        raise SystemExit("[b12] 問題があるので反映しない")
    if os.path.exists(bak):
        raise SystemExit("[b12] 退避の置き場が既にあるのに帳面が無い: %s（著者に報告する）" % bak)
    fz0 = T.frozen_en_a_digest(ev)
    if not T.is_sandbox(ev) and not fz0["digest"].startswith(T.FROZEN_DIGEST_PREFIX):
        raise SystemExit("[b12] 凍結した EN (a) の digest が記録と違う: %s" % fz0)
    ops = []
    for lang in LANGS:
        a_dir = os.path.join(ev, T.A_DIR[lang])
        for op in pl["langs"][lang]["wav_ops"]:
            keep = {c: {"path": os.path.join(a_dir, "s%04d_%s.wav" % (op["pair"], c)),
                        "ino": op["%s_a_stat" % c]["ino"], "md5": op["%s_a_stat" % c]["md5"]} for c in ("bl", "tgt")}
            ops.append({"kind": "wav", "lang": lang, "pair": op["pair"], "dest": op["dest"], "src": op["src"],
                        "src_md5": op["src_stat"]["md5"], "keep": keep})
    for lang in LANGS:
        for rel in (T.A_PIB[lang], T.A_MCD[lang]):
            doc = pl["after_docs"][rel]
            doc["t3fix_b12"] = {"pairs": list(T.B12_PAIRS[lang]), "date": T.now(), "backup": bak,
                                "source": ("rows of _v1L_t5gemma_%s_L25.json (same BL/TGT/INJ wavs, same scorer)" % lang
                                           if "v1all" in rel else
                                           "mcd_pysptk.py --mode whole on host CPU for the 4 pairs (L25 INJ); "
                                           "equal to _audit_spanB_20261002/mcd_whole_t5gemma_%s_L25.json" % lang),
                                "note": "author decision 2026-10-02 (B1/B2): INJ of the span-affected pairs replaced by "
                                        "the l*=26 sweep INJ; other rows unchanged; summary recomputed with the "
                                        "original formula"}
            prep = os.path.join(work, "prepared", rel)
            T.write_json_new(doc, prep, indent=2)
            ops.append({"kind": "doc", "lang": lang, "rel": rel, "dest": os.path.join(ev, rel), "src": prep})
    j = T.build_journal(ev, base, ops, {"step": "b12", "before": T.jsonable(before_view),
                                        "after_planned": T.jsonable(after_view), "frozen_before": fz0,
                                        "plan": os.path.join(work, "plan.json"), "code_md5": T.code_md5_record()})
    T.write_json_new(j, jp["journal"])
    finish(args, ev, base, j, work, resumed=False)


if __name__ == "__main__":
    main()
