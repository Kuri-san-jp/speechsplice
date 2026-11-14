#!/usr/bin/env python3
"""付録 C（Table A2, tab:ja_crossattn_phoneme）日本語の行の新旧の値と、日本語の文の主張の確かめ（CPU のみ, 2026-10-02）.

ホストで直接:
  taskset -c 0-12 python3 scripts/spanfix/appc_ja_aggregate.py            （300 対がそろったときだけ値を出す）
  taskset -c 0-12 python3 scripts/spanfix/appc_ja_aggregate.py --partial  （途中経過。論文には使わない）

集計は今の表と同じ式:
  * 本番 main() の末尾と同じ summary（aligned の行で数え、一致率 = 100 * 件数 / n_aligned）
  * 書式: 一致率 % ± Wilson 95% 半幅（0 件は「0.00 + 上端」）、PER は np.mean ± np.std（n で割る）を生の値から
    3 桁に（今の表の JA 8 セルを 5/11 の results.json から一字一句作り直せる作り方。appc_ja_common.table_cells）
  * 「旧」は 5/11 の results.json から同じ関数で作り直し、論文の JA 8 セルと一字一句同じになることを確かめる
主張の確かめ（値を表示するだけ。原稿は変えない）:
  Results L972-976、Interpretation L981-983、表の説明 L966、太字（L956, L962 の INJ）、説明文書 L281-282。
対の選び方（appc_ja_scan.py と同じ）: seed 42 が使えればそれ、使えなければ 43、さらに 44。値は見ない。
naacl2027/ には何も書かない（main.tex と explanation_of_revisions.tex は読むだけ）。
出力: eval_results/_spanfix_appC_ja/aggregate/aggregate_<日時>.json（新規）
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import appc_ja_scan as S  # noqa: E402  環境変数（スレッド上限・HF・GPU を見せない・pyopenjtalk を塞ぐ）は S の先頭で入る
import appc_ja_common as C  # noqa: E402

import argparse  # noqa: E402
import collections  # noqa: E402
import math  # noqa: E402

ROOT = C.PROJECT_ROOT
OWN_REL = "scripts/spanfix/appc_ja_aggregate.py"


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
                prov[p] = {"seed": s, "batch": v["batch"], "run": v["run"], "bl_wav": v.get("bl_wav")}
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


def mcnemar_exact(b, c):
    """両側の正確な McNemar（二項, p=0.5）. 参考の値（原稿には無い）."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def claims(rows):
    """日本語の文の主張を、新しい値で確かめる（表示のためだけ。数値で原稿を変えない）."""
    import numpy as np
    s = C.production_summary(rows)
    al = [r for r in rows if r.get("aligned")]
    m = {c: s[f"{c}_match_count"] for c in C.PER_CONDS}
    per = {c: float(np.mean(C.per_values(rows, c))) for c in C.PER_CONDS}
    n = s["n_aligned"]
    w = {c: C.wilson(m[c], n) for c in C.PER_CONDS}
    b = sum(1 for r in al if r.get("bl_match") and not r.get("inj_match"))
    c_ = sum(1 for r in al if r.get("inj_match") and not r.get("bl_match"))
    d_int = per["bl"] - per["inj"]
    d_mask = per["bl_mask"] - per["inj_mask"]
    n_mask_differ = sum(1 for r in al if (r.get("bl_mask_per"), r.get("bl_mask_match"))
                        != (r.get("inj_mask_per"), r.get("inj_mask_match")))
    bold = C.bold_ok(rows)
    res = {
        "R1 (L972-974) JA intact: INJ improves over BL in target-match rate": m["inj"] > m["bl"],
        "R1 (L972-974) JA intact: INJ improves over BL in PER (lower)": per["inj"] < per["bl"],
        "R2 (L974-975) JA blocked: BL_mask and INJ_mask yield the same target-match rate": m["bl_mask"] == m["inj_mask"],
        "R3 (L975-976) JA blocked: nearly the same PER (|dPER blocked| < 0.5 x |dPER intact|; heuristic)":
            abs(d_mask) < 0.5 * abs(d_int),
        "I1 (L981-982) JA intact moves the output toward the target": m["inj"] > m["bl"] and per["inj"] < per["bl"],
        "I2 (L982-983) blocking removes the phoneme-level difference (R2 and R3)":
            m["bl_mask"] == m["inj_mask"] and abs(d_mask) < 0.5 * abs(d_int),
        "Caption (L966) INJ moves toward the target with intact but not with blocked cross-attention":
            m["inj"] > m["bl"] and per["inj"] < per["bl"] and m["inj_mask"] <= m["bl_mask"],
        "Bold (L956, L962) INJ is the best JA cell in both rows (strict)":
            bold["inj_highest_match_strict"] and bold["inj_lowest_per_strict"],
        "EoR (L281-282) the same blocking removes the target-directed gain": m["inj"] > m["bl"] and m["inj_mask"] <= m["bl_mask"],
    }
    info = {
        "match_counts": m, "n_aligned": n, "per_means": {k: round(v, 4) for k, v in per.items()},
        "wilson_95": {k: [round(100 * (cc - hh), 2), round(100 * (cc + hh), 2)] for k, (cc, hh) in w.items()},
        "intact_match_gain_pp": round(100 * (m["inj"] - m["bl"]) / n, 2),
        "dPER_intact_bl_minus_inj": round(d_int, 4), "dPER_blocked_blmask_minus_injmask": round(d_mask, 4),
        "paired_intact_match": {"bl_only": b, "inj_only": c_, "mcnemar_exact_p_two_sided": mcnemar_exact(b, c_),
                                "note": "reference only; the manuscript states no test for Table A2"},
        "n_rows_where_masked_conditions_differ": n_mask_differ,
        "bold": bold,
    }
    return res, info


def empty_ipa_counts(rows):
    out = {}
    for c in C.PER_CONDS:
        out[c] = sum(1 for r in rows if r.get("aligned") and r.get(f"{c}_per") is not None
                     and r.get(f"{c}_ipa_len") == 0)
    out["tgt"] = sum(1 for r in rows if r.get("aligned") and r.get("tgt_ipa_len") == 0)
    return out


def latex_rows(cells, bold_inj):
    def b(k):
        v = cells[k]
        return f"\\mathbf{{{v}}}" if bold_inj and k.endswith("_inj") else v
    top = "JA & $" + "$ & $".join(b(f"match_{c}") for c in C.PER_CONDS) + "$ \\\\"
    bot = "JA & $" + "$ & $".join(b(f"per_{c}") for c in C.PER_CONDS) + "$ \\\\"
    return top, bot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="300 対がそろっていなくても途中の値を出す（論文には使わない）")
    args = ap.parse_args()
    errs = C.pinned_errors(ROOT) + C.old_result_errors(ROOT)
    if errs:
        raise SystemExit("[agg] " + " / ".join(errs))
    pairs = C.load_pairs(ROOT)
    sc_rel, span_ref = S.latest_span_check(pairs)
    sc = C.read_json(C.P(sc_rel))
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
        print(f"[agg] まだ 300 対そろっていない（{len(rows_by_p)}/300）。appc_ja_scan.py で次の手順を確かめる")
        print(f"[agg] 未解決の先頭: {dict(list(unresolved.items())[:10])}")
        return 2

    # ---- 旧（5/11）: 同じ関数で作り直し、保存された summary と論文の JA 8 セルに一致することを確かめる
    old = C.read_json(C.P(f"{C.OLD_DIR_REL}/results.json"))
    old_rows = old["rows"]
    old_sum = C.production_summary(old_rows)
    stored = {k: v for k, v in old["summary"].items() if k != "elapsed_min"}
    paper = {k: C.strip_bold(v) for k, v in C.PAPER_JA_CELLS.items()}
    old_cells = C.table_cells(old_rows)
    if not (old_sum == stored and old_cells == paper):
        raise SystemExit(f"[agg] 旧の値の作り直しが論文と合わない: summary={old_sum == stored} cells={old_cells}")
    tex_now = {i: ln for i, ln in read_lines(C.MAIN_TEX_REL, 950, 966)}
    paper_cells_in_tex = all(any(v in tex_now.get(i, "") for i in C.PAPER_JA_LINES) for v in C.PAPER_JA_CELLS.values())
    old_claims, old_info = claims(old_rows)

    # ---- 新
    new_rows = [rows_by_p[p] for p in sorted(rows_by_p)]
    new_sum = C.production_summary(new_rows)
    new_cells = C.table_cells(new_rows)
    via4 = C.table_cells_via4(new_rows)
    rounding_differs = {k: (new_cells[k], v) for k, v in via4.items() if new_cells[k] != v}
    new_claims, new_info = claims(new_rows)
    pidx = [r["pair_idx"] for r in new_rows]
    distinct_rows = len(set(pidx)) == len(pidx) == len(new_rows) and (not complete or sorted(pidx) == list(range(C.N_PAIRS)))
    texts_match_pairs = all((r["text_a"], r["text_b"], r["word_a"], r["word_b"])
                            == tuple(pairs[r["pair_idx"]][k] for k in ("text_a", "text_b", "word_a", "word_b"))
                            for r in new_rows)

    # ---- 参考: 環境（seed 42 で使った BL が 7/23 の付録 K の BL と同じか）、IPA が空の数、端のトークン
    env = collections.Counter()
    for p, v in prov.items():
        if v["seed"] != 42 or not v.get("bl_wav"):
            continue
        r = C.bl_env_compare(C.P(v["bl_wav"]), p, ROOT)
        env["n"] += 1
        env["identical"] += int(bool(r.get("identical")))
    straddle_a = set(sc["straddle"]["a_side"])

    text_lines = {rel: [ln for lo, hi in rngs for ln in read_lines(rel, lo, hi)]
                  for rel, rngs in C.JA_TEXT_RANGES.items()}
    by_seed = collections.Counter(v["seed"] for v in prov.values())
    replaced = sorted(p for p, v in prov.items() if v["seed"] != 42)
    top, bot = latex_rows(new_cells, new_claims["Bold (L956, L962) INJ is the best JA cell in both rows (strict)"])
    out = {"item": "Appendix C tab:ja_crossattn_phoneme, JA rows rerun on the Appendix K JA pairs (fixed span)",
           "producer": OWN_REL, "date": C.stamp(), "complete_300": complete, "partial": not complete,
           "span_check": sc_rel, "pairs": C.APPK_JA_CKPT_REL, "pairs_digest": C.pairs_digest(pairs),
           "n_final_rows": len(new_rows), "final_rows_distinct_pair_idx": distinct_rows,
           "final_rows_texts_match_appk": texts_match_pairs, "final_by_seed": dict(by_seed),
           "replaced_pairs": {str(p): prov[p] for p in replaced},
           "old": {"source": f"{C.OLD_DIR_REL}/results.json", "pairs": C.OLD_PAIR_SRC_REL + " first 300 (289 distinct)",
                   "summary": old_sum, "cells": old_cells, "summary_matches_stored": old_sum == stored,
                   "cells_match_paper": old_cells == paper, "paper_cells_found_in_tex": paper_cells_in_tex,
                   "claims": old_claims, "claims_info": old_info},
           "new": {"summary": new_sum, "cells": new_cells, "rounding_direct_vs_via4": rounding_differs,
                   "claims": new_claims, "claims_info": new_info, "latex_display": [top, bot],
                   "empty_ipa_counts": empty_ipa_counts(new_rows),
                   "straddle_a_pairs_in_final": len(straddle_a & set(rows_by_p))},
           "paper_ja_cells": C.PAPER_JA_CELLS, "paper_en_cells_unchanged": C.PAPER_EN_CELLS_NOW,
           "bl_env_seed42_vs_appk_0723": dict(env),
           "text_lines": {k: v for k, v in text_lines.items()},
           "final_rows": new_rows, "provenance": {str(p): v for p, v in prov.items()},
           "unresolved": {str(p): c for p, c in unresolved.items()}}
    path_rel = f"{C.AGG_DIR_REL}/aggregate_{C.stamp()}{'_PARTIAL' if not complete else ''}.json"
    C.write_json_new(out, C.P(path_rel))

    tag = "" if complete else "  ** PARTIAL（論文には使わない）**"
    print(f"[agg] 付録 C Table A2（tab:ja_crossattn_phoneme）の JA 行{tag}")
    print(f"[agg] 新: 付録 K の JA 300 対、N={new_sum['n_total']} (aligned {new_sum['n_aligned']}), seed ごと {dict(by_seed)}, "
          f"作り直した対 {replaced}；行の pair_idx が重複なし={distinct_rows}、文が付録 K と同じ={texts_match_pairs}")
    print(f"{'cell':<16}{'論文（今, 5/11）':<34}{'旧の作り直し':<30}{'新':<30}")
    for k in C.PAPER_JA_CELLS:
        print(f"{k:<16}{C.PAPER_JA_CELLS[k]:<34}{old_cells[k]:<30}{new_cells[k]:<30}")
    print(f"[agg] 件数 旧 {[old_sum[c + '_match_count'] for c in C.PER_CONDS]} -> 新 "
          f"{[new_sum[c + '_match_count'] for c in C.PER_CONDS]}（BL, INJ, BL_mask, INJ_mask）")
    print("[agg] LaTeX（JA 行, 表示のみ。太字は新しい値でも INJ が両方の行で最良のときだけ）:")
    print("  " + top)
    print("  " + bot)
    print("[agg] EN 行（10/02 に反映済み。この作業では変えない）: "
          + " | ".join(f"{k}={v}" for k, v in C.PAPER_EN_CELLS_NOW.items()))
    if rounding_differs:
        print(f"[agg] 注意: 4 桁経由で 3 桁にした値と生の値から 3 桁にした値が違うセル: {rounding_differs}")
    print("[agg] 日本語の文の主張（旧 → 新）:")
    for k in new_claims:
        print(f"  {old_claims[k]!s:<5} → {new_claims[k]!s:<5}  {k}")
    i = new_info
    print(f"[agg]   参考（新）: 一致率の差 INJ-BL {i['intact_match_gain_pp']} pt、PER の差 BL-INJ {i['dPER_intact_bl_minus_inj']}、"
          f"遮断 BL_mask-INJ_mask {i['dPER_blocked_blmask_minus_injmask']}、遮断の 2 条件が違う行 "
          f"{i['n_rows_where_masked_conditions_differ']}；対ごと BL だけ一致 {i['paired_intact_match']['bl_only']}・"
          f"INJ だけ一致 {i['paired_intact_match']['inj_only']}（正確 McNemar p={i['paired_intact_match']['mcnemar_exact_p_two_sided']:.3g}、"
          f"原稿には無い値）")
    print(f"[agg]   参考（新）: IPA が空の数 {out['new']['empty_ipa_counts']}；a 側の端のトークンに助詞などを含む対 "
          f"{out['new']['straddle_a_pairs_in_final']}（§3.2 の定義どおり）；seed 42 の BL が 7/23 と同じ {dict(env)}")
    print("[agg] 日本語の文（読むだけ）:")
    for rel, lines in text_lines.items():
        for ln_no, ln in lines:
            if ln.strip():
                print(f"  {os.path.basename(rel)} L{ln_no}: {ln.strip()[:220]}")
    print(f"[agg] wrote {path_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
