#!/usr/bin/env python3
"""G5a: Table 3 の Llasa-3B (a)(b) と T5Gemma-TTS EN (c) の音声と行を、使えるかの基準で走査する（CPU のみ）.

基準（2026-09-29 に全 G で統一。regen2 共通の部品 silence_criterion.assess_file で判定し、
regen2_common.wav_check / wav_stats はその結果を G5a の鍵の名前に写すだけ）:
  無音 = (RMS < 0.005 かつ peak < 0.05) または 20 ms 区間のうち区間 RMS > 0.01 の割合 < 0.05
  使えない対 = 無音・欠けた・読めない音声を含む対、または生成やスパン合わせに失敗した対
    Llasa : progress.json の行が status == "ok" でない（"error" など）か行が無い
    2 実体: results.json に行が無い（generate_multientity は失敗した対を skipped に入れる）。
            tgt1/tgt2 は wav が無い・読めないときだけ使えないとし、無音は記録だけ（Table 3 の印字値に使わない）。
対は使えるかどうかだけで選ぶ。PIB や書き起こしは見ない。

  --stage pre   : 本番を走査し、使えない対が計画（regen2_common の pairs）と一致するかを確かめる。
  --stage final : _regen2_final の作り直した音声（と regen_meta.json の行）を走査する。
  --stage post  : 差し替え後の本番を走査する（対象の対が使えるようになったか）。
                  凍結した T5Gemma EN (a) の digest も前と同じかを見る。

使い方（ホストで直接。slurm_run.sh は使わない。スレッド 13・CPU 0-12・nice 10・GPU を見せない・HF オフラインは
スクリプトの中で設定する: regen2_common.host_cpu_setup）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/silence_scan_g5a.py --stage pre
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/silence_scan_g5a.py --stage final
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/silence_scan_g5a.py --stage post
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import regen2_common as C  # noqa: E402

C.host_cpu_setup()

import argparse  # noqa: E402
import datetime  # noqa: E402
import re  # noqa: E402

LLASA_KINDS = C.LLASA_KINDS


def scan_dir(d, kinds, pairs=None):
    """{pair_idx: {kind: stats}}。pairs を渡したらその対だけ（無い wav も missing として入れる）."""
    out = {}
    pat = re.compile(r"^s(\d{4})_(%s)\.wav$" % "|".join(kinds))
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        m = pat.match(n)
        if not m:
            continue
        i, k = int(m.group(1)), m.group(2)
        if pairs is not None and i not in pairs:
            continue
        p = os.path.join(d, n)
        st = C.wav_check(p)
        st["nlink"] = os.stat(p).st_nlink
        st["mtime_utc"] = datetime.datetime.utcfromtimestamp(os.stat(p).st_mtime).strftime("%Y-%m-%d %H:%M")
        out.setdefault(i, {})[k] = st
    for i in (pairs or []):
        for k in kinds:
            if k not in out.get(i, {}):
                out.setdefault(i, {})[k] = C.wav_check(os.path.join(d, "s%04d_%s.wav" % (i, k)))
    return out


def bad_list(scan):
    return sorted((i, k) for i, d in scan.items() for k, st in d.items() if st["bad"])


def llasa_dirs(stage):
    """(cell, layer, sub, 走査するディレクトリ)."""
    out = []
    for cell in C.LLASA_CELL_ORDER:
        c = C.LLASA_CELLS[cell]
        for layer, sub in C.llasa_layer_dirs(cell):
            if stage == "final":
                d = os.path.join(C.llasa_final_root(), cell, "wav", sub)
            else:
                d = os.path.join(C.EV, c["prod_wav"], sub)
            out.append((cell, layer, sub, d))
    return out


def llasa_rows_not_ok(stage, cell, layer, sub):
    """行が ok でない対（pre/post は本番の progress.json、final は regen_meta.json の採用した行）."""
    c = C.LLASA_CELLS[cell]
    if stage == "final":
        mp = os.path.join(C.llasa_final_root(), cell, "regen_meta.json")
        if not os.path.exists(mp):
            return None
        fin = C.read_json(mp).get("final", {})
        out = {}
        for p in c["pairs"]:
            r = (fin.get(str(p)) or {}).get("rows", {}).get("L%02d" % layer)
            if (r or {}).get("status") != "ok":
                out[p] = (r or {}).get("status") or "missing"
        return out
    rows = C.read_rows(os.path.join(C.EV, c["prod_out"], sub, "progress.json"))
    return {p: (rows.get(p) or {}).get("status") or "missing" for p in range(300)
            if (rows.get(p) or {}).get("status") != "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pre", "final", "post"], default="pre")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    out_json = args.out_json or os.path.join(C.DRYRUN_DIR, "silence_scan_g5a_%s.json" % args.stage)

    report = {"stage": args.stage, "criterion": C.SILENCE_CRITERION,
              "validity": "pair invalid iff it has a missing/unreadable/silent wav (2-entity tgt1/tgt2: "
                          "missing/unreadable only; silence recorded) or a failed generation row",
              "llasa": {}, "multi": {}, "problems": []}

    # ---------------- Llasa-3B ----------------
    for cell, layer, sub, d in llasa_dirs(args.stage):
        c = C.LLASA_CELLS[cell]
        if not os.path.isdir(d):
            report["problems"].append("missing dir %s" % d)
            continue
        pairs = set(c["pairs"]) if args.stage == "final" else None
        sc = scan_dir(d, LLASA_KINDS, pairs)
        if args.stage != "final":
            n_files = sum(len(v) for v in sc.values())
            if n_files != 900:
                report["problems"].append("%s L%02d: %d wav (900 のはず)" % (cell, layer, n_files))
            missing = sorted(set(range(300)) - set(sc))
            if missing:
                report["problems"].append("%s L%02d: wav の無い対 %s" % (cell, layer, missing[:20]))
        bad = bad_list(sc)
        rows_bad = llasa_rows_not_ok(args.stage, cell, layer, sub)
        ent = report["llasa"].setdefault(cell, {"planned_pairs": c["pairs"], "layers": {}})
        ent["layers"]["L%02d" % layer] = {
            "dir": d, "bad_files": ["s%04d_%s" % (i, k) for i, k in bad],
            "bad_detail": {"s%04d_%s" % (i, k): sc[i][k] for i, k in bad},
            "rows_not_ok": {str(p): s for p, s in (rows_bad or {}).items()},
        }
        if rows_bad is None:
            report["problems"].append("%s: regen_meta.json が無い（final の行を確かめられない）" % cell)
        if args.stage == "final":
            ent["layers"]["L%02d" % layer]["target_stats"] = {
                "s%04d_%s" % (i, k): sc[i][k] for i in sorted(sc) for k in sorted(sc[i])}
    for cell, ent in report["llasa"].items():
        found_wav = {int(f[1:5]) for L in ent["layers"].values() for f in L["bad_files"]}
        found_row = {int(p) for L in ent["layers"].values() for p in L["rows_not_ok"]}
        found = sorted(found_wav | found_row)
        ent["pairs_bad_wav"] = sorted(found_wav)
        ent["pairs_row_not_ok"] = sorted(found_row)
        ent["pairs_invalid"] = found
        planned = sorted(ent["planned_pairs"])
        if args.stage == "pre":
            ent["match_plan"] = (found == planned)
            if found != planned:
                report["problems"].append("%s: 使えない対 %s が計画 %s と違う" % (cell, found, planned))
        else:
            left = [p for p in found if p in planned]
            ent["planned_pairs_still_invalid"] = left
            if left:
                report["problems"].append("%s: 対象の対 %s がまだ使えない" % (cell, left))
            if args.stage == "post":
                other = [p for p in found if p not in planned]
                if other:
                    report["problems"].append("%s: 対象外の対 %s が使えない（差し替え前には無かったはず）"
                                              % (cell, other))

    # ---------------- T5Gemma-TTS EN 2 実体 ----------------
    M = C.MULTI
    kinds = tuple(M["conds"] + M["tgt_conds"])
    if args.stage == "final":
        d = C.multi_final_dir()
        pairs = set(M["pairs"])
    else:
        d = os.path.join(C.EV, M["name"])
        pairs = None
    if os.path.isdir(d):
        sc = scan_dir(d, kinds, pairs)
        if args.stage != "final":
            n_files = sum(len(v) for v in sc.values())
            if n_files != 1800:
                report["problems"].append("multi: %d wav (1800 のはず)" % n_files)
        rows = C.read_rows(os.path.join(d, "results.json"))
        rows_t = C.read_rows(os.path.join(d, "results_tgt.json"))
        universe = sorted(pairs) if pairs is not None else list(range(300))
        cond_bad = sorted({i for i in universe for k in M["conds"] if sc.get(i, {}).get(k, {"bad": True})["bad"]})
        tgt_missing = sorted({i for i in universe for k in M["tgt_conds"]
                              if (sc.get(i, {}).get(k) or {"missing": True}).get("missing")
                              or (sc.get(i, {}).get(k) or {}).get("unreadable")})
        row_missing = sorted(i for i in universe
                             if i not in rows or not set(M["conds"]) <= set(rows[i].get("conds", {})))
        invalid = sorted(set(cond_bad) | set(tgt_missing) | set(row_missing))
        tgt_silent = sorted({i for i in universe for k in M["tgt_conds"]
                             if (sc.get(i, {}).get(k) or {}).get("silent")
                             and not (sc.get(i, {}).get(k) or {}).get("missing")
                             and not (sc.get(i, {}).get(k) or {}).get("unreadable")})
        tgt_only = sorted(set(tgt_silent) - set(invalid))
        tgt_row_missing = sorted(i for i in universe
                                 if i not in rows_t or not set(M["tgt_conds"]) <= set(rows_t[i].get("conds", {})))
        bad = bad_list(sc)
        report["multi"] = {"dir": d, "bad_files": ["s%04d_%s" % (i, k) for i, k in bad],
                           "bad_detail": {"s%04d_%s" % (i, k): sc[i][k] for i, k in bad},
                           "pairs_bl_or_inj_bad": cond_bad, "pairs_tgt_wav_missing": tgt_missing,
                           "pairs_results_row_missing": row_missing,
                           "pairs_invalid": invalid,
                           "pairs_tgt_only_silent_recorded": tgt_only,
                           "pairs_results_tgt_row_missing": tgt_row_missing,
                           "planned_pairs": M["pairs"]}
        if args.stage == "final":
            report["multi"]["target_stats"] = {"s%04d_%s" % (i, k): sc[i][k]
                                               for i in sorted(sc) for k in sorted(sc[i])}
            report["multi"]["pair_validity"] = {str(p): C.multi_pair_validity(d, p)[1]["reasons"]
                                                for p in M["pairs"]}
            left = [p for p in M["pairs"] if not C.multi_pair_validity(d, p)[0]]
            report["multi"]["planned_pairs_still_invalid"] = left
            if left:
                report["problems"].append("multi: 対象の対 %s がまだ使えない" % left)
        elif args.stage == "pre":
            report["multi"]["match_plan"] = (invalid == sorted(M["pairs"]))
            if invalid != sorted(M["pairs"]):
                report["problems"].append("multi: 使えない対 %s が計画 %s と違う" % (invalid, M["pairs"]))
            if tgt_only != sorted(M["tgt_only_silent_excluded"]):
                report["problems"].append("multi: tgt だけ無音の対 %s が想定 %s と違う"
                                          % (tgt_only, M["tgt_only_silent_excluded"]))
            # 本番の results_tgt.json に行の無い 0..2 は別の回（2026-09-19 の --with_tgt）で作ったもの。
            # wav はあるので生成の失敗ではない（記録だけ）
            if [p for p in tgt_row_missing if p not in (0, 1, 2)]:
                report["problems"].append("multi: results_tgt.json に行の無い対 %s（0..2 以外）" % tgt_row_missing)
        else:
            left = [p for p in invalid if p in M["pairs"]]
            report["multi"]["planned_pairs_still_invalid"] = left
            if left:
                report["problems"].append("multi: 対象の対 %s がまだ使えない" % left)
            other = [p for p in invalid if p not in M["pairs"]]
            if other:
                report["problems"].append("multi: 対象外の対 %s が使えない" % other)
    else:
        report["problems"].append("missing dir %s" % d)

    # ---------------- 凍結した T5Gemma EN (a) ----------------
    dig = C.frozen_digest()
    report["frozen_t5gemma_en_a"] = dig
    ref_path = os.path.join(C.DRYRUN_DIR, "frozen_t5gemma_en_digest_pre.json")
    if args.stage == "pre" and not os.path.exists(ref_path):
        C.write_json(dig, ref_path)
    elif os.path.exists(ref_path):
        ref = C.read_json(ref_path)
        report["frozen_t5gemma_en_a"]["same_as_pre"] = (ref["digest"] == dig["digest"])
        if ref["digest"] != dig["digest"]:
            report["problems"].append("凍結した T5Gemma EN (a) の音声・採点が変わった")

    C.write_json(report, out_json)
    print("[scan] stage=%s -> %s" % (args.stage, out_json))
    print("[rule] %s" % C.SILENCE_RULE)
    for cell, ent in report["llasa"].items():
        print("[llasa] %-8s invalid pairs=%s (bad wav %s, row not ok %s)  %s" % (
            cell, ent.get("pairs_invalid"), ent.get("pairs_bad_wav"), ent.get("pairs_row_not_ok"),
            {L: v["bad_files"] for L, v in ent["layers"].items() if v["bad_files"]}))
    mm = report.get("multi", {})
    print("[multi] invalid pairs=%s (bl/inj bad %s, results row missing %s, tgt wav missing %s)  "
          "tgt-only silent (recorded) %s" % (
              mm.get("pairs_invalid"), mm.get("pairs_bl_or_inj_bad"), mm.get("pairs_results_row_missing"),
              mm.get("pairs_tgt_wav_missing"), mm.get("pairs_tgt_only_silent_recorded")))
    print("[frozen] T5Gemma EN (a) digest=%s" % report["frozen_t5gemma_en_a"])
    if report["problems"]:
        print("[problems]")
        for p in report["problems"]:
            print("  -", p)
    else:
        print("[ok] problems なし")


if __name__ == "__main__":
    main()
