#!/usr/bin/env python3
"""spanfix (2a): 付録 B（Table A1）JA・ZH の範囲ずれ 8 対の作り直しの確認・取り込み（CPU のみ, 2026-10-02）.

GPU の実行（appB_jazh_gen.py）が eval_results/_spanfix_appB_jazh/<lang>/seed<S>/ に残した記録を読み、
  --check_current  今の Table A1 の JA・ZH の値と説明文書 279-280 行の値が、今の出どころ（JA: 5/19-20 の本番、
                   ZH: 9/29 に対 42 を入れ替えたもの）から今の式で再現できるかを確かめる（書かない）
  --span_check     tokenizer だけで、300 対すべての範囲（修正後・5/19-20 の時点・別の求め方）を求め、範囲が変わる対が
                   作り直しの対象の 8 対とちょうど一致することを確かめて記録する（新しいファイルにだけ書く）
  --status         対ごとの状態と次にすること、次に流す GPU のコマンド（SLURM_TIME つき）を表示する（書かない）
  --preview        今そろっている有効な対だけを入れ替えた値を表示する（途中経過。書かない）
  --aggregate      対象の 4 対すべてに有効な記録がそろった言語だけ、新しい N=300 のファイルを
                   eval_results/_spanfix_appB_jazh/merged_<日時>/<lang>cc_lp_n300/ に書き、古い値と新しい値を表示する
  --selftest       取り込みの部品（入れ替え・式・拒否・置き場の決まり）を、本物の元のファイルと作った行で確かめる
どの記録を使うかは seed の決まり（appB_jazh_common.next_action）が決める。人は選ばない。数値は選ぶのに使わない。
取り込む前に、5 本の wav の md5 と無音の判定をファイルから計算し直して記録と一致することを確かめる。
表の値の式は今の表と同じ（spanfix_common.TABLE_KEYS / CI_METHOD。英語の 10/02 の集計と同じ）:
  MCD (dB): mcd_bl_inj（Intact）、mcd_bl_mask_inj_mask（Blocked at entity）
  MEL Similarity: tmean_bl_inj_mel、tmean_bl_mask_inj_mask_mel
  値 = 平均 ± 1.96 × 標準偏差（n で割る, ddof=0）/ √n
naacl2027/ ほか原稿の置き場と、既存の結果のファイルには書かない（古い値と新しい値を表示するだけ）。

使い方（ホストで直接。プロジェクト直下から）:
  taskset -c 0-12 python3 scripts/spanfix/appB_jazh_collect.py --check_current
  taskset -c 0-12 python3 scripts/spanfix/appB_jazh_collect.py --span_check
  taskset -c 0-12 python3 scripts/spanfix/appB_jazh_collect.py --status
  taskset -c 0-12 python3 scripts/spanfix/appB_jazh_collect.py --aggregate            （--lang ja / zh で片方だけ）
  taskset -c 0-12 python3 scripts/spanfix/appB_jazh_collect.py --selftest
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appB_jazh_common as J  # noqa: E402

LAUNCH_ENV = J.setup_env()  # numpy を読む前に
if not J.IN_CONTAINER:
    sys.modules["pyopenjtalk"] = None  # 集計では一切使わない

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, J.P("scripts/regen2"))
import silence_criterion as SC  # noqa: E402

C = J.C
ROOT = J.HOST_ROOT
METRICS = ["mel", "w2v", "pps"]  # eval_crossattn_entity_dtw.py の main 末尾と同じ並び
PAIRS_TO_COMPARE = [("bl", "inj"), ("bl_mask", "inj_mask"), ("bl", "bl_mask"), ("inj", "inj_mask"),
                    ("bl", "tgt"), ("inj", "tgt")]
OUT_NAMES = ("results.json", "checkpoint.json", "table_values.json", "merge_record.json")
SEED_POLICY_TEXT = ("Author decision (2a), 2026-10-02: the 8 Appendix B JA/ZH pairs whose entity span was wrong in the "
                    "2026-05-19/20 production run (legacy find_entity_token_range, before the 2026-07-11 fix) are "
                    "regenerated with the same script and arguments and the fixed span function, seed 42; an invalid "
                    "pair (silent audio under the 2026-09-29 criterion, generation failure or alignment failure) is "
                    "regenerated as a whole pair with seed 43, then seed 44, chosen only by validity, never by results.")


def silence_note(lang):
    other = ("the other 296 pairs come from the 2026-05-19/20 production run" if lang == "ja" else
             "the other 296 pairs are the rows of eval_results/_regen2_crossattn_merged/zhcc_lp_n300 (295 rows of "
             "the 2026-05-20 production run and pair 42 regenerated with seed 43 on 2026-09-29, whose 5 wavs were "
             "checked then)")
    return ("The unified silence criterion (" + SC.CRITERION_TEXT + ") was applied to the 4 regenerated pairs "
            "(their 5 wavs are saved and were re-hashed and re-checked from the files at merge time); " + other +
            "; the production run saved no audio, so the criterion cannot be applied to those rows (evidence of "
            "non-silence there: aligned=True in all 5 conditions).")


# ---------------------------------------------------------------- 式
def summarize(rows, old_summary):
    """本番 main() の末尾と同じ式（lang と elapsed_min は元の値のまま）."""
    aligned = [r for r in rows if r.get("aligned")]

    def avg(key):
        vs = [r[key] for r in aligned if r.get(key) is not None]
        return [round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)] if vs else [None, None]

    s = {"lang": old_summary["lang"], "n_total": len(rows), "n_aligned": len(aligned)}
    for m in METRICS:
        for a, b in PAIRS_TO_COMPARE:
            for prefix in ["tmean", "dtw", "fmax", "vowel"]:
                s[f"{prefix}_{a}_{b}_{m}"] = avg(f"{prefix}_{a}_{b}_{m}")
            if m == "mel":
                s[f"frob_{a}_{b}"] = avg(f"frob_{a}_{b}")
    for a, b in PAIRS_TO_COMPARE:
        s[f"mcd_{a}_{b}"] = avg(f"mcd_{a}_{b}")
    s["elapsed_min"] = old_summary.get("elapsed_min")
    return s


def cells(tv):
    return {m: (tv[m]["intact"]["tex"], tv[m]["blocked"]["tex"]) for m in J.TABLE_KEYS}


# ---------------------------------------------------------------- 取り込みの元
def load_base(lang):
    """今の Table A1 の出どころ（読むだけ）。md5・行の並び・式の再現・本番の行との一致を確かめる."""
    d = J.BASE_DIR_REL[lang]
    md5 = {n: J.md5_file(J.P(f"{d}/{n}", ROOT)) for n in J.BASE_MD5[lang]}
    if md5 != J.BASE_MD5[lang]:
        raise SystemExit(f"[appB_jazh-collect] 取り込みの元 {d} の md5 が違う: {md5}")
    res = J.read_json(J.P(f"{d}/results.json", ROOT))
    ck = J.read_json(J.P(f"{d}/checkpoint.json", ROOT))
    errs = []
    if ck["rows"] != res["rows"]:
        errs.append("checkpoint.json と results.json の行が違う")
    if [r["pair_idx"] for r in res["rows"]] != list(range(J.N_PAIRS)):
        errs.append("行の並びが 0..299 でない")
    if summarize(res["rows"], res["summary"]) != res["summary"]:
        errs.append("summary が今の式で再現できない")
    pd = J.PROD_DIR_REL[lang]
    pmd5 = {n: J.md5_file(J.P(f"{pd}/{n}", ROOT)) for n in J.PROD_MD5[lang]}
    if pmd5 != J.PROD_MD5[lang]:
        errs.append(f"本番 {pd} の md5 が違う: {pmd5}")
    prod = J.read_json(J.P(f"{pd}/results.json", ROOT))
    g3_new = None
    if J.BASE_NON_PROD_ROWS[lang]:
        mrp = J.P(J.G3_ZH_MERGE_RECORD_REL, ROOT)
        if J.md5_file(mrp) != J.G3_ZH_MERGE_RECORD_MD5:
            errs.append(f"{J.G3_ZH_MERGE_RECORD_REL} の md5 が違う")
        g3_new = J.read_json(mrp)["new_row"]
    for i in range(J.N_PAIRS):
        if i in J.BASE_NON_PROD_ROWS[lang]:
            if res["rows"][i] != g3_new:
                errs.append(f"対 {i} が 9/29 の取り込みの記録の行と違う")
        elif res["rows"][i] != prod["rows"][i]:
            errs.append(f"対 {i} が 5/19-20 の本番の行と違う")
    for i in J.TARGETS[lang]:
        if res["rows"][i].get("aligned") is not True:
            errs.append(f"対象の対 {i} が aligned=True でない")
    if errs:
        raise SystemExit("[appB_jazh-collect] 取り込みの元が確かめに通らない:\n  - " + "\n  - ".join(errs[:10]))
    return res, ck


# ---------------------------------------------------------------- 記録を選ぶ
def collect_rows(lang, data_root, base_res, fake=False):
    runs, warns = J.all_runs(data_root, lang)
    pairs = J.load_pairs(lang, ROOT)
    ref_keys = set(next(r for r in base_res["rows"] if r.get("aligned")))
    by_rel = {r["rel"]: r for s in J.SEEDS for r in runs[s]}
    new_rows, sel, problems = {}, {}, {}
    for i in J.TARGETS[lang]:
        a = J.next_action(runs, lang, i, ROOT, check_provenance=not fake)
        if not a["action"].startswith("use_seed") or a["errors"]:
            problems[i] = {"action": a["action"], "errors": a["errors"][:3], "note": a.get("note")}
            continue
        run = by_rel[a["use"]["run"]]
        row, cap = run["rows"][i], run["captures"].get(i) or {}
        errs = []
        for k in ("text_a", "text_b", "word_a", "word_b"):
            if row.get(k) != pairs[i][k]:
                errs.append(f"{k} が違う")
        if set(row) != ref_keys:
            errs.append("行のキーが本番の行と違う")
        if row.get("aligned") is not True or any(row.get(f"frames_{c}") is None for c in J.CONDS):
            errs.append("aligned=True で 5 本の frames がある行でない")
        if cap.get("span_match") is not True:
            errs.append("実行時の範囲が修正後の関数の範囲と一致した記録が無い")
        wavs = cap.get("wavs") or {}
        if sorted(wavs) != sorted(J.CONDS):
            errs.append(f"wav が 5 本でない: {sorted(wavs)}")
        for c, w in wavs.items():
            p = os.path.join(data_root, w.get("path", ""))
            if not w.get("path", "").startswith(run["rel"] + "/wav/") or not os.path.isfile(p):
                errs.append(f"wav {c} が置き場に無い")
            elif J.md5_file(p) != w.get("md5"):
                errs.append(f"wav {c} の md5 が記録と違う")
            elif SC.assess_file(p)["silent"] or w.get("silent") is not False:
                errs.append(f"wav {c} が無音（または記録と違う）")
        if errs:
            problems[i] = {"action": a["action"], "errors": errs}
            continue
        spf = (run["record"].get("span_preflight") or {}).get(str(i)) or {}
        new_rows[i] = row
        sel[i] = {"seed": a["use"]["seed"], "run": run["rel"], "chain": a["chain"],
                  "wav_md5": {c: w["md5"] for c, w in wavs.items()},
                  "span": {s: {"fixed": (spf.get(s) or {}).get("fixed"), "legacy_0520": (spf.get(s) or {}).get("legacy_0520"),
                               "fixed_pieces": ((spf.get(s) or {}).get("fixed_check") or {}).get("pieces"),
                               "legacy_pieces": ((spf.get(s) or {}).get("legacy_check") or {}).get("pieces")}
                           for s in ("a", "b")},
                  "src_span_changed_vs_0520": spf.get("src_span_changed_vs_0520"),
                  "pyopenjtalk_calls_from_vendor": ((run["end"] or {}).get("pyopenjtalk_audit") or {}).get("calls_from_vendor")}
    return new_rows, sel, problems, warns


def build_merged(lang, base_res, base_ck, new_rows):
    """取り込みの元を写し、対象の 4 対の行だけを入れ替える。summary は本番と同じ式で計算し直す."""
    targets = sorted(J.TARGETS[lang])
    if sorted(new_rows) != targets:
        raise SystemExit(f"[appB_jazh-collect] 入れ替える行が対象の 4 対とちょうど一致しない: {sorted(new_rows)}")
    ref_keys = set(next(r for r in base_res["rows"] if r.get("aligned")))
    res, ck = copy.deepcopy(base_res), copy.deepcopy(base_ck)
    for i in targets:
        new, old = new_rows[i], base_res["rows"][i]
        errs = []
        if new.get("pair_idx") != i or old.get("pair_idx") != i:
            errs.append("pair_idx が違う")
        if old.get("aligned") is not True or new.get("aligned") is not True:
            errs.append("aligned=True でない")
        if set(new) != ref_keys:
            errs.append("行のキーが元の行と違う")
        for k in ("text_a", "text_b", "word_a", "word_b"):
            if new.get(k) != old.get(k):
                errs.append(f"{k} が元の行と違う")
        if errs:
            raise SystemExit(f"[appB_jazh-collect] 対 {i} を入れ替えない: {errs}")
        for doc in (res, ck):
            assert doc["rows"][i]["pair_idx"] == i
            doc["rows"][i] = copy.deepcopy(new)
    res["summary"] = summarize(res["rows"], base_res["summary"])
    s = res["summary"]
    if not (s["n_total"] == J.N_PAIRS and s["n_aligned"] == J.N_PAIRS):
        raise SystemExit(f"[appB_jazh-collect] n_total={s['n_total']} n_aligned={s['n_aligned']}（300 のはず）")
    for i in range(J.N_PAIRS):
        if i not in targets:
            assert res["rows"][i] == base_res["rows"][i] and ck["rows"][i] == base_ck["rows"][i], i
    return res, ck


def repro_reference(lang, new_rows, sel, base_res):
    """参考（選ぶのには使わない）: seed 42 の行で、範囲を使わない BL・TGT（と、文 a の範囲が変わらない対の BL_mask）が
    5/19-20 と同じ長さになるか。BL・TGT の音声が同じなら mcd_bl_tgt も同じになる."""
    out = {"note": "reference only; never used to choose records"}
    for i, row in sorted(new_rows.items()):
        old = base_res["rows"][i]
        d = {"seed": sel[i]["seed"], "src_span_changed_vs_0520": sel[i]["src_span_changed_vs_0520"]}
        if sel[i]["seed"] == 42:
            d["frames_bl_equal"] = row.get("frames_bl") == old.get("frames_bl")
            d["frames_tgt_equal"] = row.get("frames_tgt") == old.get("frames_tgt")
            if sel[i]["src_span_changed_vs_0520"] is False:
                d["frames_bl_mask_equal"] = row.get("frames_bl_mask") == old.get("frames_bl_mask")
            for k in ("mcd_bl_tgt", "tmean_bl_tgt_mel"):
                if row.get(k) is not None and old.get(k) is not None:
                    d[f"abs_diff_{k}"] = round(abs(row[k] - old[k]), 6)
        out[str(i)] = d
    return out


# ---------------------------------------------------------------- 置き場の決まり
def location_guard(data_root, out_dir, fake):
    rd, ro = os.path.realpath(data_root), os.path.realpath(out_dir)
    for d in list(J.PROD_DIR_REL.values()) + list(J.BASE_DIR_REL.values()):
        pd = os.path.realpath(J.P(d, ROOT))
        if ro == pd or ro.startswith(pd + os.sep):
            raise SystemExit(f"[appB_jazh-collect] 本番・取り込みの元の置き場には書かない: {out_dir}")
    dry = os.path.realpath(J.P(J.DRYRUN_REL, ROOT))
    if rd == os.path.realpath(ROOT):
        if fake:
            raise SystemExit("[appB_jazh-collect] 本物の記録を試運転として取り込まない")
        want = os.path.realpath(J.P(J.OUT_REL, ROOT)) + os.sep + J.MERGED_PREFIX
        if not ro.startswith(want):
            raise SystemExit(f"[appB_jazh-collect] 本物の出力は {J.OUT_REL}/{J.MERGED_PREFIX}<日時>/ の下だけ: {out_dir}")
    else:
        if not fake or not rd.startswith(dry + os.sep) or not ro.startswith(dry + os.sep):
            raise SystemExit(f"[appB_jazh-collect] 試運転の記録（{data_root}）は試運転の置き場（{J.DRYRUN_REL}）の出力へだけ")


# ---------------------------------------------------------------- 表示
def print_old_new(lang, tv_old, tv_new, n_new, label):
    a, b = J.MAIN_TEX_ROWS[lang]
    print(f"\n=== {label}: Table A1 (tab:crossattn_mask) {lang.upper()} rows (naacl2027/main.tex {a}-{b}) ===")
    print(f"  {'cell':26s} {'manuscript':>16s} {'recomputed old':>16s}   {'new':>16s}   n")
    for m in J.TABLE_KEYS:
        for k, col in enumerate(("intact", "blocked")):
            print(f"  {m + ' / ' + col:26s} {J.CURRENT_TABLE[lang][m][k]:>16s} {tv_old[m][col]['tex']:>16s}   "
                  f"{tv_new[m][col]['tex']:>16s}   {tv_new[m][col]['n']}")
    print(f"  (regenerated pairs merged: {n_new}/4; CI: {J.CI_METHOD})")


def print_pair_detail(lang, new_rows, sel, base_res):
    print(f"  per regenerated pair ({lang.upper()}): old row (5/19-20, legacy span) -> new row (fixed span)")
    for i, row in sorted(new_rows.items()):
        old = base_res["rows"][i]
        sp = sel[i]["span"]
        print(f"   pair {i:3d} seed {sel[i]['seed']}  span a {sp['a']['legacy_pieces']}->{sp['a']['fixed_pieces']}  "
              f"b {sp['b']['legacy_pieces']}->{sp['b']['fixed_pieces']}")
        print(f"            mcd_bl_inj {old.get('mcd_bl_inj')} -> {row.get('mcd_bl_inj')}   "
              f"tmean_bl_inj_mel {old.get('tmean_bl_inj_mel')} -> {row.get('tmean_bl_inj_mel')}   "
              f"mcd_blocked {old.get('mcd_bl_mask_inj_mask')} -> {row.get('mcd_bl_mask_inj_mask')}   "
              f"frames {[old.get(f'frames_{c}') for c in J.CONDS]} -> {[row.get(f'frames_{c}') for c in J.CONDS]}")


def eor_strings(mcd):
    """explanation_of_revisions.tex 279-280 行の 3 つの値（EN, JA, ZH の Intact の MCD）."""
    return f"${mcd['en']}$, ${mcd['ja']}$ and ${mcd['zh']}$"


def print_eor(new_mcd_by_lang):
    en = J.read_json(J.P(J.EN_FINAL_TV_REL, ROOT))["new"]["MCD (dB)"]["intact"]
    en_tex = f"{en['mean']:.2f}\\pm{en['ci95_half']:.2f}"
    new = {"en": en_tex, **{lg: J.CURRENT_EOR[lg] for lg in J.LANGS}, **new_mcd_by_lang}
    print(f"\n=== naacl2027/explanation_of_revisions.tex L{J.EOR_LINES[0]}-{J.EOR_LINES[1]} (entity-span MCD, Intact) ===")
    print(f"  old: {eor_strings(J.CURRENT_EOR)}")
    print(f"  new: {eor_strings(new)}" + ("" if new_mcd_by_lang else "   (no JA/ZH merge yet: unchanged)"))
    print(f"  (EN from {J.EN_FINAL_TV_REL}; JA/ZH new only for the languages merged in this call)")
    return new


# ---------------------------------------------------------------- 取り込み
def merge_lang(lang, data_root=ROOT, out_dir=None, fake=False):
    location_guard(data_root, out_dir, fake)
    if os.path.lexists(out_dir):
        raise SystemExit(f"[appB_jazh-collect] 出力の置き場が既にある（上書きしない）: {out_dir}")
    base_res, base_ck = load_base(lang)
    tv_before = C.table_values(base_res["rows"])
    if cells(tv_before) != J.CURRENT_TABLE[lang]:
        raise SystemExit(f"[appB_jazh-collect] 今の表が今の出どころと式で再現できない: {cells(tv_before)}")
    new_rows, sel, problems, warns = collect_rows(lang, data_root, base_res, fake)
    if problems:
        raise SystemExit("[appB_jazh-collect] 対象の 4 対がそろっていない（または確かめに通らない）ので取り込まない:\n"
                         + json.dumps({str(k): v for k, v in problems.items()}, ensure_ascii=False, indent=1))
    res, ck = build_merged(lang, base_res, base_ck, new_rows)
    tv_after = C.table_values(res["rows"])
    assert all(v["n"] == J.N_PAIRS for cols in tv_after.values() for v in cols.values()), "N が 300 でない"
    ref = repro_reference(lang, new_rows, sel, base_res)
    blocked = {"mcd_bl_mask_inj_mask_max_over_300": max(r["mcd_bl_mask_inj_mask"] for r in res["rows"]),
               "tmean_bl_mask_inj_mask_mel_min_over_300": min(r["tmean_bl_mask_inj_mask_mel"] for r in res["rows"]),
               "note": "Appendix A (JA, Δtok=0 subset) and the Blocked column: if the max blocked MCD over all 300 "
                       "rows is 0, the statement holds for any subset"}
    rel_out = os.path.relpath(out_dir, ROOT)
    tv = {"lang": lang, "fake_selftest": fake, "n": J.N_PAIRS, "table_keys": J.TABLE_KEYS, "ci_method": J.CI_METHOD,
          "n300_source_note": f"Table A1 の {lang.upper()} 行の新しい N=300 の値の出どころはこのファイル（{rel_out}/table_values.json）。"
                              f"取り込みの元 {J.BASE_DIR_REL[lang]} と本番 {J.PROD_DIR_REL[lang]} は変えていない。",
          "before_n300": tv_before, "after_n300": tv_after,
          "old_in_manuscript": J.CURRENT_TABLE[lang], "manuscript_rows": J.MAIN_TEX_ROWS[lang],
          "eor_l279_280": {"old": J.CURRENT_EOR[lang],
                           "new": f"{tv_after['MCD (dB)']['intact']['mean']:.2f}\\pm{tv_after['MCD (dB)']['intact']['ci95_half']:.2f}"},
          "replaced_pairs": sorted(new_rows), "seeds_used": {str(i): sel[i]["seed"] for i in sorted(sel)},
          "silence_check_scope": silence_note(lang), "reproduction_reference": ref, "blocked_all_rows": blocked}
    rec = {"item": f"spanfix (2a) Appendix B tab:crossattn_mask {lang.upper()}", "date": J.now(), "fake_selftest": fake,
           "lang": lang, "replaced_pairs": sorted(new_rows), "reason": SEED_POLICY_TEXT,
           "seed_policy": SEED_POLICY_TEXT, "silence_check_scope": silence_note(lang),
           "base_dir": J.BASE_DIR_REL[lang], "base_md5": J.BASE_MD5[lang],
           "production_dir": J.PROD_DIR_REL[lang], "production_md5": J.PROD_MD5[lang],
           "selection": {str(i): sel[i] for i in sorted(sel)}, "record_warnings": warns,
           "old_rows": {str(i): base_res["rows"][i] for i in sorted(new_rows)},
           "new_rows": {str(i): new_rows[i] for i in sorted(new_rows)},
           "summary_before": base_res["summary"], "summary_after": res["summary"],
           "code_md5_now": J.current_code_md5(ROOT), "collect_md5": J.md5_file(os.path.abspath(__file__)),
           "reproduction_reference": ref}
    parent = os.path.dirname(os.path.abspath(out_dir))
    os.makedirs(parent, exist_ok=True)
    staging = os.path.join(parent, f"{os.path.basename(out_dir)}.staging_{time.strftime('%Y%m%d_%H%M%S')}_"
                                   f"{os.getpid()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(staging, exist_ok=False)
    J.write_json_new(res, os.path.join(staging, "results.json"), indent=2)
    J.write_json_new(ck, os.path.join(staging, "checkpoint.json"))
    J.write_json_new(tv, os.path.join(staging, "table_values.json"))
    J.write_json_new(rec, os.path.join(staging, "merge_record.json"))
    assert sorted(os.listdir(staging)) == sorted(OUT_NAMES)
    if os.path.lexists(out_dir):
        raise SystemExit(f"[appB_jazh-collect] 出力の置き場が途中でできた（上書きしない）: {out_dir}。作業用 {staging} は残る")
    os.rename(staging, out_dir)
    for n, want in J.BASE_MD5[lang].items():  # 元は変わっていない
        assert J.md5_file(J.P(f"{J.BASE_DIR_REL[lang]}/{n}", ROOT)) == want
    label = "FAKE SELFTEST (numbers are copies of the base rows)" if fake else "FINAL N=300"
    print_old_new(lang, tv_before, tv_after, len(new_rows), label)
    print_pair_detail(lang, new_rows, sel, base_res)
    print("  reproduction vs 5/19-20 (reference):", json.dumps(ref, ensure_ascii=False))
    print(f"  blocked condition over all 300 rows: max MCD {blocked['mcd_bl_mask_inj_mask_max_over_300']}, "
          f"min MEL {blocked['tmean_bl_mask_inj_mask_mel_min_over_300']} (Appendix A / Blocked column)")
    print(f"[appB_jazh-collect] written: {rel_out}/{{{', '.join(OUT_NAMES)}}}")
    return tv


def aggregate(langs, preview=False):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    new_mcd = {}
    for lang in langs:
        if preview:
            base_res, base_ck = load_base(lang)
            new_rows, sel, problems, _ = collect_rows(lang, ROOT, base_res)
            rows = [copy.deepcopy(r) for r in base_res["rows"]]
            for i, r in new_rows.items():
                rows[i] = r
            tv_old, tv_new = C.table_values(base_res["rows"]), C.table_values(rows)
            print_old_new(lang, tv_old, tv_new, len(new_rows), f"PREVIEW ({len(new_rows)}/4 regenerated pairs, NOT FINAL)")
            if new_rows:
                print_pair_detail(lang, new_rows, sel, base_res)
            print(f"  not settled yet: { {str(k): v['action'] for k, v in problems.items()} }")
            continue
        out = J.P(f"{J.OUT_REL}/{J.MERGED_PREFIX}{stamp}/{lang}cc_lp_n300", ROOT)
        tv = merge_lang(lang, ROOT, out, fake=False)
        new_mcd[lang] = tv["eor_l279_280"]["new"]
    print_eor(new_mcd)


# ---------------------------------------------------------------- 状態と次のコマンド
def status(langs):
    out = {"date": J.now(), "langs": {}}
    cmds = []
    for lang in langs:
        runs, warns = J.all_runs(ROOT, lang)
        acts = {i: J.next_action(runs, lang, i, ROOT) for i in J.TARGETS[lang]}
        by = {}
        for i, a in acts.items():
            by.setdefault(a["action"], []).append(i)
        for s in J.SEEDS:
            ids = sorted(by.get(f"run_seed{s}", []) + by.get(f"rerun_seed{s}", []))
            ids_c = sorted(by.get(f"confirm_stopped_seed{s}", []))
            allids = ids + ids_c
            for j in range(0, len(allids), J.CHUNK):
                chunk = allids[j:j + J.CHUNK]
                extra = []  # seed 42 は --pairs なし（流せる対象の対を gen が決まりで選ぶ）
                if s != 42:
                    extra.append("--pairs " + ",".join(str(i) for i in chunk))
                cc = sorted({acts[i]["unfinished_run"] for i in chunk if i in ids_c})
                extra += [f"--confirm_stopped {r}" for r in cc]
                c = J.gpu_command(lang, s, len(chunk), " ".join(extra))
                if cc:
                    c["before"] = "squeue -u $USER で、上の置き場の実行が動いていないことを確かめる"
                cmds.append(c)
        unfinished = [{"run": r["rel"], "n_todo": len(r["todo"]), "started": r["record"].get("started")}
                      for s in J.SEEDS for r in runs[s] if r["end"] is None]
        out["langs"][lang] = {
            "n_runs": {str(s): len(runs[s]) for s in J.SEEDS},
            "per_pair": {str(i): {"action": a["action"], "use": a.get("use"), "note": a.get("note"),
                                  "errors": a["errors"][:3], "code_drift": a.get("code_drift"), "chain": a["chain"]}
                         for i, a in acts.items()},
            "unfinished_runs": unfinished, "record_warnings": warns,
            "ready_to_aggregate": all(a["action"].startswith("use_seed") for a in acts.values())}
    out["next_gpu_commands"] = cmds
    out["gpu_rule"] = ("run from $REPO_ROOT; at most 2 GPUs for the user in total (squeue -u $USER "
                       "first); one job per command")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------------------------------------------------------------- 今の値の確かめ
def check_current():
    out, ok = {}, True
    main_lines = open(J.P(J.MAIN_TEX_REL, ROOT), encoding="utf-8").read().split("\n")
    eor_lines = open(J.P(J.EOR_REL, ROOT), encoding="utf-8").read().split("\n")
    for lang in J.LANGS:
        base_res, _ = load_base(lang)
        tv = C.table_values(base_res["rows"])
        a, b = J.MAIN_TEX_ROWS[lang]
        cur = J.CURRENT_TABLE[lang]
        tex_ok = (f"\\mathbf{{{cur['MCD (dB)'][0]}}}" in main_lines[a - 1] and cur["MCD (dB)"][1] in main_lines[a - 1]
                  and f"\\mathbf{{{cur['MEL Similarity'][0]}}}" in main_lines[b - 1]
                  and cur["MEL Similarity"][1] in main_lines[b - 1] and main_lines[a - 1].startswith(lang.upper()))
        out[lang] = {"base": J.BASE_DIR_REL[lang], "base_md5_ok": True, "recomputed": cells(tv),
                     "manuscript_constant": cur, "reproduced": cells(tv) == cur,
                     "main_tex_lines": {str(a): main_lines[a - 1].strip(), str(b): main_lines[b - 1].strip()},
                     "main_tex_matches_constant": tex_ok,
                     "n": {m: {c: v["n"] for c, v in cols.items()} for m, cols in tv.items()},
                     "blocked_mcd_max_over_300": max(r["mcd_bl_mask_inj_mask"] for r in base_res["rows"]),
                     "blocked_mel_min_over_300": min(r["tmean_bl_mask_inj_mask_mel"] for r in base_res["rows"])}
        ok = ok and out[lang]["reproduced"] and tex_ok
    eor = " ".join(eor_lines[J.EOR_LINES[0] - 1:J.EOR_LINES[1]])
    eor_ok = all(f"${J.CURRENT_EOR[k]}$" in eor for k in ("en", "ja", "zh"))
    en = J.read_json(J.P(J.EN_FINAL_TV_REL, ROOT))["new"]["MCD (dB)"]["intact"]
    en_ok = f"{en['mean']:.2f}\\pm{en['ci95_half']:.2f}" == J.CURRENT_EOR["en"]
    audit = J.read_json(J.P(J.AUDIT_SPANS_REL, ROOT))
    audit_ok = {lg: audit[lg]["summary"]["legacy_differs_idx"] == list(J.TARGETS[lg]) for lg in J.LANGS}
    out["eor"] = {"lines": eor.strip(), "values_present": eor_ok, "en_matches_spanfix_en_final": en_ok}
    out["audit_targets_match"] = audit_ok
    out["appendix_A_note"] = ("Appendix A (Δtok=0 JA subset) states MCD=0 for BL_mask vs INJ_mask; the blocked MCD is "
                              "0 for every one of the 300 JA rows (blocked_mcd_max_over_300), so it holds for any subset; "
                              "the new file is re-checked the same way at merge time")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    ok = ok and eor_ok and en_ok and all(audit_ok.values())
    assert ok, "今の値が今の出どころ・式・原稿の文字列と一致しない"
    print("[appB_jazh-collect] current Table A1 JA/ZH and EoR L279-280 reproduced from the current sources: OK")


# ---------------------------------------------------------------- 範囲の確かめ（tokenizer だけ）
def span_check():
    W = C.load_shim_module()
    off = C.setup_offline(W)
    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_ID)
    fixed_fn, legacy_fn = C.span_functions()
    norm = J.vendor_normalizer()
    rep = {"item": "spanfix (2a) Appendix B JA/ZH, CPU span check (tokenizer only)", "date": J.now(),
           "tokenizer": {"class": type(tok).__name__, "name_or_path": tok.name_or_path,
                         "transformers": transformers.__version__},
           "offline": {k: off[k] for k in ("hf_offline", "offline_preflight_ok", "offline_tokenizer_shim")},
           "env_at_launch": LAUNCH_ENV, "code_md5": J.current_code_md5(ROOT), "vendor_md5": J.vendor_md5_now(),
           "fixed_function": C.INJ_REL + " find_entity_token_range (offset mapping, 2026-07-11)",
           "legacy_function": C.LEGACY_SPAN_REL + " find_entity_token_range (before 2026-07-11)",
           "text": "vendor normalize_text_with_lang(text, lang) applied first (JA only changes characters), as in "
                   "get_encoder_output; add_bos=0 (the runtime adds the same offset to both)",
           "langs": {}}
    ok = True
    for lang in J.LANGS:
        pairs = J.load_pairs(lang, ROOT)
        t = J.span_table(tok, pairs, lang, fixed_fn, legacy_fn, norm)
        summ = J.span_summary(t, lang)
        agr = J.audit_agreement(t, lang, ROOT)
        rep["langs"][lang] = {"summary": summ, "audit_agreement": agr,
                              "normalization_changed_text": sorted(i for i, r in t.items()
                                                                   if r["a"]["normalization_changed_text"]
                                                                   or r["b"]["normalization_changed_text"]),
                              "pairs": {str(k): v for k, v in t.items()}}
        ok = ok and summ["equals_targets"] and all(t[i]["fixed_ok"] for i in J.TARGETS[lang]) and agr["n_disagree"] == 0
    out = J.P(f"{J.SPAN_CHECK_REL}/span_check_{time.strftime('%Y%m%d_%H%M%S')}.json", ROOT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(J.P(J.LOGS_REL, ROOT), exist_ok=True)
    J.write_json_new(rep, out)
    print(json.dumps({lg: {"targets": v["summary"]["targets"],
                           **{k: v["summary"][k] for k in ("pairs_whose_span_changes_vs_0520", "equals_targets",
                                                           "src_side_changed", "fixed_ok", "fixed_not_ok", "fixed_exact")},
                           "fixed_with_extra_chars_targets": {k: x for k, x in v["summary"]["fixed_with_extra_chars"].items()
                                                              if int(k) in J.TARGETS[lg]},
                           "n_fixed_with_extra_chars": len(v["summary"]["fixed_with_extra_chars"]),
                           "normalization_changed_text": v["normalization_changed_text"],
                           "audit_agreement": {k: v["audit_agreement"][k] for k in ("n_compared", "n_disagree")}}
                      for lg, v in rep["langs"].items()}, ensure_ascii=False, indent=1))
    print(f"[appB_jazh-collect] span check report: {os.path.relpath(out, ROOT)}")
    print(f"[appB_jazh-collect] spans change exactly at the 8 target pairs and the fixed span covers the name there: {ok}")
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------- 取り込みの部品の試運転
def selftest():
    base = J.P(f"{J.DRYRUN_REL}/collect_selftest_{time.strftime('%Y%m%d_%H%M%S')}", ROOT)
    os.makedirs(base, exist_ok=False)
    R = {}

    def refused(fn):
        try:
            fn()
        except SystemExit as e:
            return str(e)[:200]
        raise AssertionError("拒否されるはずが通った")

    for lang in J.LANGS:
        r = {}
        base_res, base_ck = load_base(lang)
        tg = list(J.TARGETS[lang])
        same = {i: copy.deepcopy(base_res["rows"][i]) for i in tg}
        res, ck = build_merged(lang, base_res, base_ck, same)
        assert res == base_res and ck == base_ck, "同じ行で入れ替えたのに元と違う"
        assert cells(C.table_values(res["rows"])) == J.CURRENT_TABLE[lang]
        r["identity_merge_equals_base"] = True
        bumped = {i: dict(copy.deepcopy(base_res["rows"][i]), mcd_bl_inj=base_res["rows"][i]["mcd_bl_inj"] + 1.0)
                  for i in tg}
        res2, _ = build_merged(lang, base_res, base_ck, bumped)
        diff = [i for i in range(J.N_PAIRS) if res2["rows"][i] != base_res["rows"][i]]
        m0 = C.table_values(base_res["rows"])["MCD (dB)"]["intact"]["mean"]
        m1 = C.table_values(res2["rows"])["MCD (dB)"]["intact"]["mean"]
        assert diff == tg and abs((m1 - m0) - len(tg) / J.N_PAIRS) < 1e-9, (diff, m1 - m0)
        assert res2["summary"]["mcd_bl_inj"][0] == round(float(np.mean([x["mcd_bl_inj"] for x in res2["rows"]])), 4)
        r["only_targets_change_and_mean_shift"] = {"changed": diff, "mean_shift": round(m1 - m0, 6)}
        if lang == "zh":
            assert res2["rows"][42] == base_res["rows"][42]
            r["zh_pair42_kept_from_0929"] = True
        r["refuse_missing_target"] = refused(lambda: build_merged(lang, base_res, base_ck,
                                                                  {i: same[i] for i in tg[:3]}))
        r["refuse_extra_pair"] = refused(lambda: build_merged(lang, base_res, base_ck,
                                                              {**same, 0: copy.deepcopy(base_res["rows"][0])}))
        r["refuse_text_mismatch"] = refused(lambda: build_merged(lang, base_res, base_ck,
                                                                 {**same, tg[0]: dict(same[tg[0]], text_a="X")}))
        r["refuse_unaligned_row"] = refused(lambda: build_merged(lang, base_res, base_ck,
                                                                 {**same, tg[0]: dict(same[tg[0]], aligned=False)}))
        r["refuse_extra_key"] = refused(lambda: build_merged(lang, base_res, base_ck,
                                                             {**same, tg[0]: dict(same[tg[0]], extra=1)}))
        dry = J.P(J.DRYRUN_REL, ROOT)
        r["guard_real_as_fake"] = refused(lambda: location_guard(ROOT, J.P(f"{J.OUT_REL}/{J.MERGED_PREFIX}x/y", ROOT), True))
        r["guard_real_out_elsewhere"] = refused(lambda: location_guard(ROOT, J.P(f"{J.OUT_REL}/other/y", ROOT), False))
        r["guard_into_base"] = refused(lambda: location_guard(ROOT, J.P(J.BASE_DIR_REL[lang] + "/x", ROOT), False))
        r["guard_fake_into_real"] = refused(lambda: location_guard(os.path.join(base, "fr"),
                                                                   J.P(f"{J.OUT_REL}/{J.MERGED_PREFIX}x/y", ROOT), True))
        location_guard(os.path.join(dry, "a", "b"), os.path.join(dry, "a", "out"), True)  # 通る
        r["guard_fake_within_dryrun_ok"] = True
        R[lang] = r
    J.write_json_new({"date": J.now(), "results": R}, os.path.join(base, "selftest_report.json"))
    print(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"[appB_jazh-collect] selftest OK; report {os.path.relpath(base, ROOT)}/selftest_report.json")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--status", "--check_current", "--span_check", "--preview", "--aggregate", "--selftest"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--lang", choices=list(J.LANGS), default=None, help="片方の言語だけ（既定は両方）")
    args = ap.parse_args()
    assert not J.IN_CONTAINER, "ホストで流す（CPU のみ）"
    langs = [args.lang] if args.lang else list(J.LANGS)
    if args.status:
        status(langs)
    elif args.check_current:
        check_current()
    elif args.span_check:
        span_check()
    elif args.preview:
        aggregate(langs, preview=True)
    elif args.aggregate:
        aggregate(langs, preview=False)
    else:
        selftest()


if __name__ == "__main__":
    main()
