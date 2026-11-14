#!/usr/bin/env python3
"""G5a: 作り直した対の採点を、本番の採点ファイルの該当行に入れ替え、集計を元の式で計算し直す（CPU のみ）.

採点そのものは元のスクリプトを改変せずに、作り直した音声（_regen2_final）に対して回す:
  PIB  : speechsplice_release/src/phoneme_per.py（GPU、scripts/regen2/run_score_regen2_gpu.sh）
  MCD  : speechsplice_release/src/mcd_pysptk.py --mode whole（ホストの CPU、scripts/regen2/run_mcd_regen2_cpu.py）
  ASR  : scripts/analysis/score_multientity.py（GPU、Whisper large-v3。同じ .sh）
  天井 : scripts/analysis/score_multientity_ceiling.py（任意。Table 3 には載っていない）
対象外の 291〜299 対の行は本番のまま残す（書き起こしや GPU の非決定性で他の対が動かないように）。
各対の採点は他の対に依存しないので、対象の対の行を入れ替えて集計し直した値は、全対を採点し直した値と
（他の対の行が同じである限り）一致する。集計の式は regen2_common に元のスクリプトから写してあり、
--selftest で既存ファイルの summary を 1 桁も違わず再現できることを確かめる。

差し替え（apply_regen2_*.py --apply）が済んでいて、退避が取れているときだけ本番に書く。
N=300: 入れ替えた後の PIB と MCD の n_valid、2 実体の行数が 300 でなければ書かない。

使い方（ホストで直接。スレッド 13・CPU 0-12・nice 10・GPU を見せない・HF オフラインはスクリプトの中で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/merge_regen2_scores.py --selftest
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/merge_regen2_scores.py --selftest_mcd_view
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/merge_regen2_scores.py --kind all          # 予行
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/merge_regen2_scores.py --kind all --apply  # 実行

--campaign thankyou（著者判断 2026-09-30。regen2_common.THANKYOU。--kind asr だけ）:
  _regen2_thankyou/final/multientity_en_300/multientity_asr_scores_regen.json（seed の判定に使った書き起こしの行）で
  本番の multientity_asr_scores.json の対象の対の行を入れ替え、summary を元の式で計算し直す。
  書く条件: apply_regen2_multientity.py --campaign thankyou --apply が済んでいる（regen2_thankyou_pairs.json と
  退避 backup_before_thankyou/multientity_asr_scores.json がある）、unresolved が無い、書き起こしに失敗も
  "thankyou" も無い、行 300・枠 600、n_error が本番と同じ、まだ入れ替えていない。
  G5a の doc["regen2"] は残し、doc["regen2_thankyou"] を足す。記録は multientity_en_300/regen2_thankyou_scores.json。
  予行でも実行でも Table 3 (c)（k/600）と UTMOS の平均（bl / inj1 / inj12）の前後を表示する。
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/merge_regen2_scores.py --campaign thankyou --kind asr
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/merge_regen2_scores.py --campaign thankyou --kind asr --apply
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import regen2_common as C  # noqa: E402

C.job_env()
C.host_cpu_setup()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

M = C.MULTI
# Table 3 (c) の 6 枠（Baseline: bl の 2 枠、One: inj1 の枠 1 と inj2 の枠 2、Both: inj12 の 2 枠）
ASR_SLOT_KEYS = ("bl_slot1_ok", "bl_slot2_ok", "inj1_slot1_ok", "inj2_slot2_ok", "inj12_slot1_ok", "inj12_slot2_ok")


def _same(a, b):
    """summary の比較。NaN どうしは同じ。float は相対 1e-9 まで同じとみなす
    （p 値の最後の桁が scipy の版で 1e-16 程度ずれるため。表に出る丸めた値は完全一致を要求する）."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def regen_score_path(cell, kind, layer):
    return os.path.join(C.llasa_final_root(), cell, "scores", "%s_L%02d.json" % (kind, layer))


def chosen_pairs_llasa(cell):
    p = os.path.join(C.llasa_final_root(), cell, "regen_meta.json")
    if not os.path.exists(p):
        return None
    return sorted(int(k) for k, s in C.read_json(p)["chosen_seed"].items() if s is not None)


def chosen_pairs_multi():
    p = os.path.join(C.multi_final_dir(), "regen_meta.json")
    if not os.path.exists(p):
        return None
    return sorted(int(k) for k, s in C.read_json(p)["chosen_seed"].items() if s is not None)


def ceiling_summary(rows, old):
    k = n = 0
    per = {1: [0, 0], 2: [0, 0]}
    miss = 0
    for r in rows:
        for s, cond in ((1, "tgt1"), (2, "tgt2")):
            if r.get("%s_missing" % cond):
                miss += 1
                continue
            if "%s_ok" % cond in r:
                ok = bool(r["%s_ok" % cond])
                per[s][0] += int(ok)
                per[s][1] += 1
                k += int(ok)
                n += 1
    s = dict(old)
    s["n_attempted"] = len(rows)
    s["n_wav_missing"] = miss
    s["ceiling_ok"] = {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
                       "half": round(C._wilson(k, n)[1], 4) if n else None}
    s["by_slot"] = {str(sl): {"k": v[0], "n": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None}
                    for sl, v in per.items()}
    return s


# ---------------------------------------------------------------------------
# 自己検査
# ---------------------------------------------------------------------------
def selftest():
    out = {}
    for cell in C.LLASA_CELL_ORDER:
        c = C.LLASA_CELLS[cell]
        for layer, f in c["pib_json"].items():
            d = C.read_json(os.path.join(C.EV, f))
            out[f] = _same(C.pib_summary(d["rows"], d["summary"]), d["summary"])
        for layer, f in c["mcd_json"].items():
            d = C.read_json(os.path.join(C.EV, f))
            out[f] = _same(C.mcd_summary(d["rows"], d["summary"]), d["summary"])
    d = C.read_json(os.path.join(C.EV, M["name"], M["asr_json"]))
    out[M["asr_json"]] = _same(C.asr_summary(d["rows"], d["summary"]), d["summary"])
    # 本番の書き起こしに失敗が無く、どの行も Table 3 (c) の 6 枠の判定を持つこと（plan_multi が作り直した対に
    # 求める条件と同じ。本番がこれを満たすので、入れ替え後も n=600 のまま失敗 0 であることを求めてよい）
    out["asr_prod_n_error_is_0"] = (d["summary"].get("n_error") == 0)
    out["asr_prod_rows_have_6_slot_bools"] = all(isinstance(r.get(k), bool) for r in d["rows"] for k in ASR_SLOT_KEYS)
    cells = C.table3c_cells(d["rows"])
    out["table3c_reproduces_96.17/95.33/96.33"] = (
        [cells["Baseline"]["pct"], cells["One"]["pct"], cells["Both"]["pct"]] == [96.17, 95.33, 96.33])
    dc = C.read_json(os.path.join(C.EV, M["name"], "ceiling_asr.json"))
    out["ceiling_asr.json"] = _same(ceiling_summary(dc["rows"], dc["summary"]), dc["summary"])
    return out


def selftest_mcd_view():
    """本番の (a) の対象の対（と対象外 1 対）への symlink だけのフォルダで mcd_pysptk.py を回し、
    行が本番の MCD の JSON と一致するかを見る（CPU、数対だけ）。作り直した対の MCD を対象の行だけ
    差し替える方式が、全対を回し直したときと同じ値になることの確認."""
    res = {}
    for cell in ("en_a", "zh_a"):
        c = C.LLASA_CELLS[cell]
        view = C.assert_regen_path(os.path.join(C.DRYRUN_DIR, "mcd_view", cell))
        os.makedirs(view, exist_ok=True)
        pairs = list(c["pairs"]) + [c["check_pair"]]
        for p in pairs:
            for k in ("bl", "inj", "tgt"):
                n = "s%04d_%s.wav" % (p, k)
                dst = os.path.join(view, n)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(C.EV, c["prod_wav"], n), dst)
        out_json = os.path.join(view, "mcd_whole_view.json")
        env = dict(os.environ, REPO_ROOT=C.PROJECT_ROOT)
        cmd = [sys.executable, os.path.join(C.PROJECT_ROOT, "src", "mcd_pysptk.py"),
               "--wav_dir", view, "--lang", c["lang"], "--mode", "whole", "--n", "300", "--output_json", out_json]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=C.PROJECT_ROOT, env=env, capture_output=True, text=True)
        el = time.time() - t0
        if r.returncode != 0:
            res[cell] = {"error": r.stderr[-800:]}
            continue
        new = {x["pair_idx"]: x for x in C.read_json(out_json)["rows"]}
        old = {x["pair_idx"]: x for x in C.read_json(os.path.join(C.EV, c["mcd_json"][c["layer"]]))["rows"]}
        keys = ("mcd_inj_vs_bl", "mcd_inj_vs_tgt", "mcd_bl_vs_tgt")
        has_silent = {p: any(C.wav_stats(os.path.join(C.EV, c["prod_wav"], "s%04d_%s.wav" % (p, k)))["silent"]
                             for k in ("bl", "inj", "tgt")) for p in pairs}
        res[cell] = {"pairs": pairs, "rows_equal": {str(p): new[p] == old[p] for p in pairs},
                     "max_abs_diff_db": {str(p): max(abs(new[p][k] - old[p][k]) for k in keys) for p in pairs},
                     "pair_has_silent_wav": {str(p): has_silent[p] for p in pairs},
                     "nonsilent_pairs_equal": all(new[p] == old[p] for p in pairs if not has_silent[p]),
                     "elapsed_sec": round(el, 1), "sec_per_pair": round(el / len(pairs), 2),
                     "note": "production MCD was computed in the container; host numerics differ only in the "
                             "4th decimal for near-silent wavs"}
    return res


# ---------------------------------------------------------------------------
# 入れ替え
# ---------------------------------------------------------------------------
def merge_rows(prod_doc, new_rows, pairs, key="pair_idx"):
    rows = list(prod_doc["rows"])
    idx = {r[key]: j for j, r in enumerate(rows)}
    for p in pairs:
        if p not in new_rows:
            raise SystemExit("作り直しの採点に対 %d の行が無い" % p)
        rows[idx[p]] = new_rows[p]
    return rows


def plan_llasa(kind, problems):
    plans = []
    for cell in C.LLASA_CELL_ORDER:
        c = C.LLASA_CELLS[cell]
        files = c["pib_json"] if kind == "pib" else c["mcd_json"]
        pairs = chosen_pairs_llasa(cell)
        rec = os.path.join(C.EV, c["prod_out"], "regen2_seed43_pairs.json")
        for layer, f in files.items():
            prod_p = os.path.join(C.EV, f)
            new_p = regen_score_path(cell, kind, layer)
            bak = os.path.join(C.llasa_backup_root(), cell, "scores", f)
            pl = {"cell": cell, "layer": layer, "prod": prod_p, "new": new_p, "backup": bak, "pairs": pairs,
                  "ready": bool(pairs is not None and os.path.exists(new_p))}
            if pl["ready"]:
                prod = C.read_json(prod_p)
                new = {r["pair_idx"]: r for r in C.read_json(new_p)["rows"]}
                rows = merge_rows(prod, new, pairs)
                summ = C.pib_summary(rows, prod["summary"]) if kind == "pib" else C.mcd_summary(rows, prod["summary"])
                pl.update(rows=rows, before=prod["summary"], after=summ,
                          new_rows={str(p): new[p] for p in pairs},
                          old_rows={str(r["pair_idx"]): r for r in prod["rows"] if r["pair_idx"] in pairs})
                if len(rows) != 300 or summ["n_valid"] != 300:
                    problems.append("%s: 入れ替え後の n_valid=%s（300 でない）" % (f, summ["n_valid"]))
                if not os.path.exists(rec):
                    problems.append("%s: 音声の差し替え記録が無い（apply_regen2_llasa.py --apply が先）" % cell)
                if not os.path.exists(bak):
                    problems.append("%s: 退避が無い（%s）" % (f, bak))
            plans.append(pl)
    return plans


def plan_multi(kind, problems):
    pairs = chosen_pairs_multi()
    name = M["asr_json"] if kind == "asr" else "ceiling_asr.json"
    newname = "multientity_asr_scores_regen.json" if kind == "asr" else "ceiling_asr_regen.json"
    prod_p = os.path.join(C.EV, M["name"], name)
    new_p = os.path.join(C.multi_final_dir(), newname)
    bak = os.path.join(C.multi_backup_dir(), name)
    pl = {"kind": kind, "prod": prod_p, "new": new_p, "backup": bak, "pairs": pairs,
          "ready": bool(pairs is not None and os.path.exists(new_p))}
    if pl["ready"]:
        prod = C.read_json(prod_p)
        new = {r["pair_idx"]: r for r in C.read_json(new_p)["rows"]}
        if kind == "asr":
            # score_multientity.py は書き起こしに失敗した条件に <cond>_err だけを書き、slot*_ok を書かない。
            # table3c_cells は n = 2 × 行数で数えるので、失敗した枠は「読めなかった」として黙って k/600 に入る。
            # そこで作り直した対の行に失敗が無く、Table 3 (c) の 6 枠の判定がすべてそろうことを先に確かめる
            for p in (pairs or []):
                r = new.get(p)
                if r is None:
                    continue                      # 行が無いことは merge_rows が止める
                errk = sorted(k for k in r if k.endswith("_err"))
                badk = [k for k in ASR_SLOT_KEYS if not isinstance(r.get(k), bool)]
                if errk or badk:
                    problems.append("2 実体: 対 %d の書き起こしが失敗している（%s）" % (p, errk + badk))
        rows = merge_rows(prod, new, pairs)
        summ = C.asr_summary(rows, prod["summary"]) if kind == "asr" else ceiling_summary(rows, prod["summary"])
        pl.update(rows=rows, before=prod["summary"], after=summ,
                  new_rows={str(p): new[p] for p in pairs})
        if kind == "asr":
            pl["table3c_before"] = C.table3c_cells(prod["rows"])
            pl["table3c_after"] = C.table3c_cells(rows)
            if len(rows) != 300 or any(v["n"] != 600 for v in pl["table3c_after"].values()):
                problems.append("2 実体: 入れ替え後の行数/枠数が 300/600 でない")
            if summ["n_error"] != prod["summary"]["n_error"]:
                problems.append("2 実体: 入れ替え後の書き起こしの失敗の数 n_error=%s が本番の %s と違う"
                                % (summ["n_error"], prod["summary"]["n_error"]))
        if not os.path.exists(os.path.join(C.EV, M["name"], "regen2_seed43_pairs.json")):
            problems.append("2 実体: 音声の差し替え記録が無い（apply_regen2_multientity.py --apply が先）")
        if not os.path.exists(bak):
            problems.append("2 実体: 退避が無い（%s）" % bak)
    return [pl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=["pib", "mcd", "asr", "ceiling", "all"])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest_mcd_view", action="store_true")
    ap.add_argument("--campaign", default="g5a", choices=["g5a", "thankyou"],
                    help="g5a: 9/29 の無音の対（既定。これまでと同じ）、thankyou: 9/30 の「Thank you.」の 6 対（asr だけ）")
    args = ap.parse_args()
    C.refuse_apply_in_test(args.apply)
    if args.campaign == "thankyou":
        main_thankyou(args)
        return
    if args.selftest:
        st = selftest()
        C.write_json(st, os.path.join(C.DRYRUN_DIR, "merge_selftest.json"))
        print(json.dumps(st, ensure_ascii=False, indent=1))
    if args.selftest_mcd_view:
        st = selftest_mcd_view()
        C.write_json(st, os.path.join(C.DRYRUN_DIR, "merge_selftest_mcd_view.json"))
        print(json.dumps(st, ensure_ascii=False, indent=1))
    if not args.kind:
        return
    kinds = ["pib", "mcd", "asr"] if args.kind == "all" else [args.kind]
    problems, plans = [], []
    for k in kinds:
        plans += plan_llasa(k, problems) if k in ("pib", "mcd") else plan_multi(k, problems)
    view = []
    for pl in plans:
        v = {k: pl.get(k) for k in ("cell", "layer", "kind", "prod", "new", "ready", "pairs")}
        if pl.get("ready"):
            b, a = pl["before"], pl["after"]
            if "inj_better_pct" in a:
                v["PIB"] = [b["inj_better_pct"], a["inj_better_pct"], "n_valid", b["n_valid"], a["n_valid"]]
            if "mcd_inj_vs_bl" in a:
                v["MCD_inj_vs_bl"] = [b["mcd_inj_vs_bl"], a["mcd_inj_vs_bl"]]
            if "table3c_after" in pl:
                v["Table3c"] = {"before": pl["table3c_before"], "after": pl["table3c_after"]}
            if "ceiling_ok" in a:
                v["ceiling"] = [b["ceiling_ok"], a["ceiling_ok"]]
        view.append(v)
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "problems": problems, "plans": view},
                     ensure_ascii=False, indent=1, default=str))
    C.write_json({"problems": problems, "plans": view},
                 os.path.join(C.DRYRUN_DIR, "merge_%s_%s.json" % (args.kind, "apply" if args.apply else "dry_run")))
    if not args.apply:
        return
    not_ready = [(pl.get("cell") or pl.get("kind"), pl.get("layer")) for pl in plans if not pl["ready"]]
    if not_ready:
        problems.append("採点がまだ無い: %s" % not_ready)
    if problems:
        raise SystemExit("問題があるので書かない")
    records = {}
    for pl in plans:
        prod_p = C.assert_writable_target(pl["prod"])
        doc = C.read_json(prod_p)
        doc["rows"] = pl["rows"]
        doc["summary"] = pl["after"]
        doc.setdefault("regen2", {})
        doc["regen2"] = {"pairs": pl["pairs"], "source": pl["new"], "backup": pl["backup"],
                         "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "note": "rows of the regenerated pairs replaced; other rows unchanged; summary recomputed "
                                 "with the original formula"}
        indent = 2 if pl.get("cell") else (2 if pl.get("kind") == "asr" else 1)
        C.write_json(doc, prod_p, indent=indent)
        recdir = os.path.join(C.EV, C.LLASA_CELLS[pl["cell"]]["prod_out"]) if pl.get("cell") \
            else os.path.join(C.EV, M["name"])
        records.setdefault(recdir, []).append({k: pl.get(k) for k in ("cell", "layer", "kind", "prod", "new", "backup",
                                                                     "pairs", "before", "after", "table3c_before",
                                                                     "table3c_after", "old_rows", "new_rows")})
        print("[merge] %s <- %s" % (prod_p, pl["new"]), flush=True)
    for recdir, recs in records.items():
        out = os.path.join(recdir, "regen2_seed43_scores.json")
        prev = C.read_json(out)["merges"] if os.path.exists(out) else []
        C.write_json({"merges": prev + recs}, out)
        print("[merge] record ->", out)


# ---------------------------------------------------------------------------
# --campaign thankyou（著者判断 2026-09-30）
# ---------------------------------------------------------------------------
def selftest_thankyou():
    """今の本番の採点の summary を元の式で 1 桁も違わず再現でき、Table 3 (c) が G5a の記録の入れ替え後の値と同じか."""
    d = C.read_json(os.path.join(C.EV, M["name"], M["asr_json"]))
    rec = os.path.join(C.EV, M["name"], C.THANKYOU["prod_table3c_now_record"])
    last = C.read_json(rec)["merges"][-1].get("table3c_after") if os.path.exists(rec) else None
    return {"asr_summary_recompute": _same(C.asr_summary(d["rows"], d["summary"]), d["summary"]),
            "asr_prod_n_error_is_0": d["summary"].get("n_error") == 0,
            "asr_prod_rows_have_6_slot_bools": all(isinstance(r.get(k), bool) for r in d["rows"] for k in ASR_SLOT_KEYS),
            "table3c_now_equals_g5a_record_after": C.table3c_cells(d["rows"]) == last}


def plan_multi_thankyou(problems):
    T = C.THANKYOU
    fin = C.thankyou_final_dir()
    meta_p = os.path.join(fin, T["meta_name"])
    meta = C.read_json(meta_p) if os.path.exists(meta_p) else None
    pairs = sorted(int(k) for k, s in meta["chosen_seed"].items() if s is not None) if meta else None
    prod_p = os.path.join(C.EV, M["name"], M["asr_json"])
    new_p = os.path.join(fin, T["asr_final_name"])
    bak = os.path.join(C.thankyou_backup_dir(), M["asr_json"])
    rec_pairs = os.path.join(C.EV, M["name"], T["record_pairs"])
    applied = os.path.exists(rec_pairs)
    prod = C.read_json(prod_p)
    pl = {"kind": "asr", "campaign": T["id"], "prod": prod_p, "new": new_p, "backup": bak, "pairs": pairs,
          "ready": bool(pairs is not None and os.path.exists(new_p)), "audio_applied": applied,
          "table3c_before": C.table3c_cells(prod["rows"]), "table3c_after": None}
    # UTMOS（results.json は apply が書く）: apply 済みなら退避した差し替え前 → 今、まだなら今 → 未定
    res_now = C.read_json(os.path.join(C.EV, M["name"], "results.json"))["summary"]
    res_bak = os.path.join(C.thankyou_backup_dir(), "meta_results.json")
    if applied and os.path.exists(res_bak):
        pl["utmos_before"], pl["utmos_after"] = C.utmos_means(C.read_json(res_bak)["summary"]), C.utmos_means(res_now)
    else:
        pl["utmos_before"], pl["utmos_after"] = C.utmos_means(res_now), None
    if meta is not None:
        if meta.get("campaign") != T["id"]:
            problems.append("meta の campaign %s が %s でない" % (meta.get("campaign"), T["id"]))
        for p in meta.get("unresolved_pairs", []):
            problems.append("2 実体: 対 %d は計画の seed でどれも使えない（入れ替えない。著者に報告）" % p)
    if "regen2_thankyou" in prod:
        problems.append("2 実体: 本番の採点は既に入れ替え済み（regen2_thankyou がある）")
    if pl["ready"]:
        new = {r["pair_idx"]: r for r in C.read_json(new_p)["rows"]}
        for p in pairs:
            r = new.get(p)
            if r is None:
                continue                      # 行が無いことは merge_rows が止める
            errk = sorted(k for k in r if k.endswith("_err"))
            badk = [k for k in ASR_SLOT_KEYS if not isinstance(r.get(k), bool)]
            if errk or badk:
                problems.append("2 実体: 対 %d の書き起こしが失敗している（%s）" % (p, errk + badk))
            ty = [k for k in M["conds"] if C.is_thankyou(r.get(k + "_transcript"))]
            if ty:
                problems.append("2 実体: 対 %d の %s の書き起こしが 'thankyou'（入れ替えない）" % (p, ty))
        rows = merge_rows(prod, new, pairs)
        summ = C.asr_summary(rows, prod["summary"])
        pl.update(rows=rows, before=prod["summary"], after=summ, new_rows={str(p): new[p] for p in pairs},
                  old_rows={str(r["pair_idx"]): r for r in prod["rows"] if r["pair_idx"] in pairs})
        pl["table3c_after"] = C.table3c_cells(rows)
        if len(rows) != 300 or any(v["n"] != 600 for v in pl["table3c_after"].values()):
            problems.append("2 実体: 入れ替え後の行数/枠数が 300/600 でない")
        if summ["n_error"] != prod["summary"]["n_error"]:
            problems.append("2 実体: 入れ替え後の n_error=%s が本番の %s と違う" % (summ["n_error"], prod["summary"]["n_error"]))
    if not applied:
        problems.append("2 実体: 音声の差し替え記録が無い（apply_regen2_multientity.py --campaign thankyou --apply が先）: %s"
                        % rec_pairs)
    if not os.path.exists(bak):
        problems.append("2 実体: 退避が無い（%s）" % bak)
    return pl


def main_thankyou(args):
    T = C.THANKYOU
    if args.kind not in (None, "asr", "all"):
        raise SystemExit("--campaign thankyou は --kind asr だけ（Table 3 (c) は Whisper の書き起こしだけで決まる）")
    problems = []
    fz0 = C.frozen_digest()
    if not fz0["digest"].startswith(T["frozen_digest_prefix"]):
        problems.append("凍結した T5Gemma EN (a) の digest %s が %s... でない" % (fz0["digest"], T["frozen_digest_prefix"]))
    st = selftest_thankyou()
    if not all(st.values()):
        problems.append("本番の採点の自己検査が合わない: %s" % st)
    pl = plan_multi_thankyou(problems)
    view = {k: pl.get(k) for k in ("kind", "campaign", "prod", "new", "backup", "ready", "audio_applied", "pairs",
                                   "table3c_before", "table3c_after", "utmos_before", "utmos_after")}
    mode = "apply" if args.apply else "dry_run"
    print(json.dumps({"mode": mode, "selftest": st, "problems": problems, "plan": view},
                     ensure_ascii=False, indent=1, default=str))
    print("\n".join(C.before_after_lines(pl["table3c_before"], pl["table3c_after"], pl["utmos_before"],
                                         pl["utmos_after"])))
    if pl["table3c_after"] is None:
        print("  (Table 3 (c) after: n/a until %s exists)" % pl["new"])
    if pl["utmos_after"] is None:
        print("  (UTMOS after: n/a until apply_regen2_multientity.py --campaign thankyou --apply)")
    C.write_json({"selftest": st, "problems": problems, "plan": view, "frozen_t5gemma_en_a": fz0},
                 os.path.join(C.thankyou_dryrun_dir(), "merge_asr_%s.json" % mode))
    if not args.apply:
        print("[dry_run] problems=%d（本番には何も書いていない）" % len(problems))
        return
    if not pl["ready"]:
        problems.append("採点がまだ無い: %s" % pl["new"])
    if problems:
        raise SystemExit("問題があるので書かない")
    prod_p = C.assert_writable_target(pl["prod"])
    doc = C.read_json(prod_p)
    doc["rows"] = pl["rows"]
    doc["summary"] = pl["after"]
    doc["regen2_thankyou"] = {"campaign": T["id"], "pairs": pl["pairs"], "source": pl["new"], "backup": pl["backup"],
                              "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "note": "rows of the regenerated pairs replaced by the transcriptions that decided "
                                      "the seed; other rows unchanged; summary recomputed with the original formula; "
                                      "doc['regen2'] is the earlier G5a merge"}
    C.write_json(doc, prod_p, indent=2)
    print("[merge] %s <- %s" % (prod_p, pl["new"]), flush=True)
    out = os.path.join(C.EV, M["name"], T["record_scores"])
    prev = C.read_json(out)["merges"] if os.path.exists(out) else []
    rec = {k: pl.get(k) for k in ("kind", "campaign", "prod", "new", "backup", "pairs", "before", "after",
                                  "table3c_before", "table3c_after", "utmos_before", "utmos_after",
                                  "old_rows", "new_rows")}
    rec["date"] = doc["regen2_thankyou"]["date"]
    C.write_json({"merges": prev + [rec]}, out)
    after = C.frozen_digest()
    if after != fz0:
        raise SystemExit("凍結した T5Gemma EN (a) が変わった（ありえない）")
    print("[merge] record -> %s; frozen T5Gemma EN (a) unchanged (%s)" % (out, after["digest"]))


if __name__ == "__main__":
    main()
