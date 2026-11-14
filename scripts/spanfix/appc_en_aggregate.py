#!/usr/bin/env python3
"""付録 C（tab:ja_crossattn_phoneme）英語の行の新旧の値を出す（CPU のみ, 2026-10-01）.

ホストで直接:
  taskset -c 0-12 python3 scripts/spanfix/appc_en_aggregate.py            （300 対がそろったときだけ値を出す）
  taskset -c 0-12 python3 scripts/spanfix/appc_en_aggregate.py --partial  （途中経過。論文には使わない）

集計は今の表と同じ式:
  * 本番 main() の末尾と同じ summary（appc_en_common.production_summary: aligned の行で数え、
    一致率 = 100 * 件数 / n_aligned、PER は None を除いた np.mean と np.std（母標準偏差）を 4 桁に丸める）
  * 表の書式は今の表と同じ（一致率 % ± Wilson 95% 半幅、0 件は「0.00 + 上端」、PER は 4 桁の値を 3 桁に）
  * 「旧」は今の results.json（5/10）から同じ関数で作り直し、論文の 8 セルと一字一句同じになることを確かめる
対の選び方（appc_en_scan.py と同じ）: seed 42 が使えればそれ、使えなければ 43、さらに 44。値は見ない。
naacl2027/ には何も書かない（main.tex と explanation_of_revisions.tex は読むだけ）。
出力: eval_results/_spanfix_appC_en/aggregate/aggregate_<日時>.json（新規）
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import appc_en_scan as S  # noqa: E402  環境変数（スレッド上限・HF・GPU を見せない）は S の先頭で入る
import appc_en_common as C  # noqa: E402

import argparse  # noqa: E402
import collections  # noqa: E402

ROOT = C.PROJECT_ROOT
OWN_REL = "scripts/spanfix/appc_en_aggregate.py"
OLD_CELL_NUMBERS = ("1.00", "1.28", "3.33", "2.10", "0.885", "0.366", "0.915", "0.411", "0.955", "0.491")


def resolve(finals):
    rows, prov, unresolved = {}, {}, {}
    for p in range(C.N_PAIRS):
        chain = []
        for s in C.SEEDS:
            v = finals[s].get(p)
            if v is None:
                chain.append([s, "not_run"])
                break
            chain.append([s, v["status"]])
            if v["status"] == C.STATUS_VALID:
                rows[p] = v["row"]
                prov[p] = {"seed": s, "batch": v["batch"], "run": v["run"]}
                break
            if v["status"] not in C.PAIR_INVALID:
                break
        if p not in rows:
            unresolved[p] = chain
        else:
            prov[p]["chain"] = chain
    return rows, prov, unresolved


def read_lines(rel, lo, hi):
    with open(C.P(rel), encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [(i, lines[i - 1]) for i in range(lo, min(hi, len(lines)) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="300 対がそろっていなくても途中の値を出す（論文には使わない）")
    args = ap.parse_args()
    errs = C.pinned_errors(ROOT) + C.old_result_errors(ROOT)
    if errs:
        raise SystemExit("[agg] " + " / ".join(errs))
    pairs = C.load_pairs(ROOT)
    sc_rel, span_ref = S.latest_span_check(pairs)
    finals = {}
    for s in C.SEEDS:
        _, finals[s] = S.scan_seed(s, span_ref)
    stops = [(s, p) for s in C.SEEDS for p, v in finals[s].items() if v["status"] == C.STATUS_STOP]
    if stops:
        raise SystemExit(f"[agg] STOP の対がある（著者の判断）: {stops}")
    rows_by_p, prov, unresolved = resolve(finals)
    complete = len(rows_by_p) == C.N_PAIRS
    if not rows_by_p:
        print("[agg] 使える対がまだ 1 つも無い（生成の前）")
        return 2
    if not complete and not args.partial:
        print(f"[agg] まだ 300 対そろっていない（{len(rows_by_p)}/300）。appc_en_scan.py で次の手順を確かめる")
        print(f"[agg] 未解決の先頭: {dict(list(unresolved.items())[:10])}")
        return 2

    # ---- 旧（5/10）: 同じ関数で作り直し、保存された summary と論文の 8 セルに一致することを確かめる
    old = C.read_json(C.P(f"{C.OLD_DIR_REL}/results.json"))
    old_rows = old["rows"]
    old_sum = C.production_summary(old_rows)
    stored = {k: v for k, v in old["summary"].items() if k != "elapsed_min"}
    old_sum_matches_stored = (old_sum == stored)
    old_cells = C.table_cells(old_sum)
    old_cells_match_paper = (old_cells == C.PAPER_EN_CELLS)
    if not (old_sum_matches_stored and old_cells_match_paper):
        raise SystemExit(f"[agg] 旧の値の作り直しが論文と合わない: summary={old_sum_matches_stored} "
                         f"cells={old_cells_match_paper} {old_cells}")

    # ---- 新
    new_rows = [rows_by_p[p] for p in sorted(rows_by_p)]
    new_sum = C.production_summary(new_rows)
    new_cells = C.table_cells(new_sum)
    # 丸めの順の確かめ（4 桁に丸めてから 3 桁 / 直接 3 桁）
    import numpy as np
    aligned = [r for r in new_rows if r.get("aligned")]
    direct = {}
    for c in C.PER_CONDS:
        vs = [r[f"{c}_per"] for r in aligned if r.get(f"{c}_per") is not None]
        direct[c] = f"{np.mean(vs):.3f} \\pm {np.std(vs):.3f}" if vs else None
    rounding_differs = {c: (direct[c], new_cells[f"per_{c}"]) for c in C.PER_CONDS
                        if direct[c] != new_cells[f"per_{c}"]}

    # ---- 本文の主張の確かめ（main.tex L954 付近「In EN, INJ does not improve over BL, and the two blocked
    #      conditions are identical.」）。値は表示するだけで、本文は変えない
    n_mask_differ = sum(1 for r in aligned if (r.get("bl_mask_per"), r.get("bl_mask_match"))
                        != (r.get("inj_mask_per"), r.get("inj_mask_match")))
    claims = {
        "EN: INJ match count <= BL match count": new_sum["inj_match_count"] <= new_sum["bl_match_count"],
        "EN: INJ mean PER >= BL mean PER": (new_sum["inj_per_mean_std"][0] >= new_sum["bl_per_mean_std"][0]),
        "EN: BL_mask and INJ_mask identical in every row (PER and match)": n_mask_differ == 0,
        "n_rows_where_masked_conditions_differ": n_mask_differ,
    }
    win = read_lines("naacl2027/main.tex", 945, 965)
    hit = {i for i, ln in win if "In EN" in ln}
    text_lines = [(i, ln) for i, ln in win if i in hit or (i - 1) in hit]

    # ---- BL と TGT はスパンを使わない: seed 42 の対で、5/10 と同じ値が出たか（環境の違いの目安。選択には使わない）
    old_by_p = {r["pair_idx"]: r for r in old_rows}
    rep = collections.Counter()
    for p, r in rows_by_p.items():
        if prov[p]["seed"] != 42:
            continue
        o = old_by_p[p]
        rep["n"] += 1
        rep["bl_per_same"] += int(o.get("bl_per") == r.get("bl_per"))
        rep["bl_ipa_len_same"] += int(o.get("bl_ipa_len") == r.get("bl_ipa_len"))
        rep["tgt_ipa_len_same"] += int(o.get("tgt_ipa_len") == r.get("tgt_ipa_len"))

    # ---- explanation_of_revisions.tex L277-281（読むだけ）: L279 は付録 B の EN MCD。付録 C の英語の数値は無いはず
    eor = read_lines(C.EOR_REL, *C.EOR_LINES)
    eor_has_appc_en = [(i, n) for i, ln in eor for n in OLD_CELL_NUMBERS if n in ln]

    by_seed = collections.Counter(v["seed"] for v in prov.values())
    replaced = sorted(p for p, v in prov.items() if v["seed"] != 42)
    out = {"item": "Appendix C tab:ja_crossattn_phoneme, EN rows, span-fix rerun", "producer": OWN_REL,
           "date": C.stamp(), "complete_300": complete, "partial": not complete,
           "span_check": sc_rel, "n_final_rows": len(new_rows), "final_by_seed": dict(by_seed),
           "replaced_pairs": {str(p): prov[p] for p in replaced},
           "old": {"source": f"{C.OLD_DIR_REL}/results.json", "summary": old_sum, "cells": old_cells,
                   "summary_matches_stored": old_sum_matches_stored, "cells_match_paper": old_cells_match_paper},
           "new": {"summary": new_sum, "cells": new_cells, "rounding_direct_vs_via4": rounding_differs},
           "paper_cells": C.PAPER_EN_CELLS, "paper_source": C.PAPER_SOURCE,
           "claims_check": claims, "main_tex_lines": text_lines,
           "bl_tgt_reproduction_seed42_vs_0510": dict(rep),
           "eor_lines": eor, "eor_contains_appC_en_numbers": eor_has_appc_en,
           "final_rows": new_rows, "provenance": {str(p): v for p, v in prov.items()},
           "unresolved": {str(p): c for p, c in unresolved.items()}}
    path_rel = f"{C.AGG_DIR_REL}/aggregate_{C.stamp()}{'_PARTIAL' if not complete else ''}.json"
    C.write_json_new(out, C.P(path_rel))

    tag = "" if complete else "  ** PARTIAL（論文には使わない）**"
    print(f"[agg] 付録 C tab:ja_crossattn_phoneme の EN 行{tag}")
    print(f"[agg] 新: N={new_sum['n_total']} (aligned {new_sum['n_aligned']}), seed ごと {dict(by_seed)}, "
          f"作り直した対 {replaced}")
    print(f"{'cell':<16}{'論文（今）':<32}{'旧の作り直し':<32}{'新':<32}")
    for k in C.PAPER_EN_CELLS:
        print(f"{k:<16}{C.PAPER_EN_CELLS[k]:<32}{old_cells[k]:<32}{new_cells[k]:<32}")
    print(f"[agg] 件数 旧 {[old_sum[c + '_match_count'] for c in C.PER_CONDS]} -> 新 "
          f"{[new_sum[c + '_match_count'] for c in C.PER_CONDS]}（BL, INJ, BL_mask, INJ_mask）")
    print("[agg] LaTeX（EN 行, 表示のみ）:")
    print("  EN & $" + "$ & $".join(new_cells[f"match_{c}"] for c in C.PER_CONDS) + "$ \\\\")
    print("  EN & $" + "$ & $".join(new_cells[f"per_{c}"] for c in C.PER_CONDS) + "$ \\\\")
    if rounding_differs:
        print(f"[agg] 注意: 4 桁で丸めてから 3 桁にした値と直接 3 桁にした値が違うセル: {rounding_differs}")
    print(f"[agg] 本文の主張の確かめ: {claims}")
    for i, ln in text_lines:
        print(f"  main.tex L{i}: {ln.strip()[:200]}")
    print(f"[agg] BL/TGT の再現（seed 42 の対, 5/10 と同じ値の数）: {dict(rep)}")
    print("[agg] explanation_of_revisions.tex L277-281（読むだけ。L279 は付録 B の EN MCD）:")
    for i, ln in eor:
        print(f"  L{i}: {ln}")
    print(f"[agg]   付録 C の英語の旧い数値を含む行: {eor_has_appc_en or 'なし（付録 C の作り直しではこの行は変わらない）'}")
    print(f"[agg] wrote {path_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
