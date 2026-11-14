#!/usr/bin/env python3
"""G3: 作り直した対の行を、付録 B（tab:crossattn_mask）の本番の結果に入れた「新しい」ファイルを作る（CPU のみ, 2026-09-29 改訂）.

どの記録を使うかは、このスクリプトが seed の決まり（g3_common.next_action）で自動で決める。人が選ばない:
  * seed 43 の最後の回が valid → seed 43 を使う（seed 44 の置き場があれば止まる）。
  * seed 43 の最後の回が対として使えない（unaligned / generation_failed / silent）→ seed 44 の最後の回が valid なら使う。
    seed 44 の記録は、今の seed 43 の記録（置き場と md5）を根拠にしていなければならない。
  * seed 42 の行（本番・再現の確認）は決して使わない。45 以上の置き場があれば止まる。
  * run_error / alignment_exception / capture_error / no_row_unexplained の回が最後なら止まる（同じ seed を
    --attempt でやり直す。alignment_exception が同じ seed で 2 回続いたら著者の判断）。
  * valid か使えない回の後に、同じ seed の別の回があれば止まる（結果で選ぶことになるので）。
出どころの確かめ（g3_common.provenance_errors。どれかに外れれば何も書かずに止まる）:
  producer と mode（regen）、置き場が決まりどおり、span_preflight が本物で一致、slurm_run.sh のコンテナ（/mnt）と GPU、
  HF_HOME=/mnt/.cache/huggingface と HF_HUB_OFFLINE=1、tokenizer を手元の snapshot から読んだ記録、本番の引数、
  main() と forced_alignment_utils の print を記録した証拠、例外なし（main_error・pair_loop_errors・whisper_errors が空）、
  本番のファイル 4 本の md5 が固定の値（g3_common.PINNED_MD5）、無音の基準の ID と数値が同じ、行が main() の書いた
  checkpoint.json の行と同じ、5 本の wav の md5 と無音の判定（共通の基準でファイルから計算し直す）。
  G3 自身のコード（ラッパーと g3_common.py）の md5 が GPU の実行の後に変わっていて、ほかは通るときは
  stop_author_decision（GPU を流し直しても解決しない。g3_common.py の説明を見る）。
無音の基準を当てられるのは作り直した対（wav がある）だけ。ほかの 299 対は 5/20 の本番で wav を残していないので、
  5 条件すべてで aligned=True（Whisper が実体語を見つけた）ことが無音でない証拠になる。これを merge_record.json の
  reason と table_values.json に書く。

入力:
  - 本番（読むだけ。md5 が 5/20 のものと違えば止まる）: eval_results/crossattn_entity_dtw_{en,zh}cc_lp_n300/
  - 作り直し: eval_results/_regen2_crossattn_seed{43,44}/<lang>_p<idx>[_a<k>]/regen_row.json
出力（新規。本番の置き場には書かない。既にあれば止まる。何も消さない）:
  eval_results/_regen2_crossattn_merged/<lang>cc_lp_n300/
  4 本は隣の作業用の置き場（<lang>cc_lp_n300.staging_<日時>_<pid>/）に書いてから、置き場ごと 1 回の rename で出す。
  途中で止まっても出力の置き場には何も現れず（4 本そろうか、何も無いか）、次の取り込みはそのまま流せる。
  - results.json      本番の results.json を写し、対象の行だけを入れ替え、summary を本番スクリプトと同じ式で計算し直す
                      （elapsed_min は元の値のまま）
  - checkpoint.json   同じ入れ替え
  - table_values.json 表の値（MCD = mcd_bl_inj / mcd_bl_mask_inj_mask、Mel similarity = dtw_bl_inj_mel /
                      dtw_bl_mask_inj_mask_mel）の平均と 1.96*SE。SE は np.std（ddof=0）/sqrt(n)（299 対の表と同じ）。
                      参考に ddof=1 の値も並べる。入れ替え前（299 対）と後（300 対）の両方。
                      ★ tab:crossattn_mask の N=300 の値の出どころはこのファイル（本番の置き場は 299 対のまま）。
  - merge_record.json 入れ替えの記録（元の行、新しい行、使った seed と回、seed 43/44 の記録の連なり、
                      seed 42 の再現の確認（参考）、wav の md5、本番ファイルの md5）
本番の置き場を書き換える手順（apply_regen_pairs.py の型）は、著者が求めたときにだけ作る。

使い方（ホストで直接。CPU のみ）:
  状態（読むだけ）: taskset -c 0-12 nice -n 10 python3 scripts/regen2/g3_merge_crossattn.py --status
  取り込み:         taskset -c 0-12 nice -n 10 python3 scripts/regen2/g3_merge_crossattn.py --lang en
  試運転（偽の記録で、受け入れ・拒否と生成側の門（seed_policy）を言語ごとに約 30 通り確かめる。出力は毎回新しい
         eval_results/_regen2_dryrun/g3_crossattn/merge_selftest_<日時>/）:
                    taskset -c 0-12 nice -n 10 python3 scripts/regen2/g3_merge_crossattn.py --selftest
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402

import g3_common as G  # noqa: E402
import silence_criterion as SC  # noqa: E402

ROOT = G.HOST_ROOT
N_PAIRS = G.N_PAIRS
CONDS = G.CONDS
# eval_crossattn_entity_dtw.py の main 末尾と同じ並び
METRICS = ["mel", "w2v", "pps"]
PAIRS_TO_COMPARE = [("bl", "inj"), ("bl_mask", "inj_mask"), ("bl", "bl_mask"), ("inj", "inj_mask"),
                    ("bl", "tgt"), ("inj", "tgt")]
TABLE_KEYS = {
    "MCD (dB)": {"intact": "mcd_bl_inj", "blocked": "mcd_bl_mask_inj_mask"},
    "Mel similarity": {"intact": "dtw_bl_inj_mel", "blocked": "dtw_bl_mask_inj_mask_mel"},
}
# 今の表（n=299 で計算された値）。入れ替え前の再計算がこれに一致するかを確かめるだけに使う（原稿は触らない）。
TABLE_299 = {
    "en": {"MCD (dB)": ("32.52 ± 0.59", "0.00 ± 0.00"), "Mel similarity": ("0.79 ± 0.01", "0.98 ± 0.00")},
    "zh": {"MCD (dB)": ("35.35 ± 0.39", "0.00 ± 0.00"), "Mel similarity": ("0.75 ± 0.01", "0.98 ± 0.00")},
}
SEED_POLICY_TEXT = ("Author decision (1), 2026-09-29: an invalid pair (silent audio, generation failure or alignment "
                    "failure) is regenerated as a whole pair with seed 43, then with seed 44 only if it is still "
                    "invalid at seed 43, chosen only by the validity criterion, never by results. Seed 42 reruns are "
                    "reproduction checks only and are never merged.")
DRY = os.path.join(ROOT, G.DRYRUN_ROOT_REL)


def summarize(rows, old_summary):
    """eval_crossattn_entity_dtw.py の main 末尾と同じ式（elapsed_min は元の値のまま）."""
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


def table_values(rows):
    aligned = [r for r in rows if r.get("aligned")]
    out = {}
    for metric, cols in TABLE_KEYS.items():
        out[metric] = {}
        for col, key in cols.items():
            vs = np.array([r[key] for r in aligned if r.get(key) is not None], dtype=np.float64)
            n = int(vs.size)
            mean = float(vs.mean())
            sd0 = float(vs.std())
            sd1 = float(vs.std(ddof=1)) if n > 1 else float("nan")
            ci0 = 1.96 * sd0 / math.sqrt(n)
            ci1 = 1.96 * sd1 / math.sqrt(n)
            out[metric][col] = {"key": key, "n": n, "mean": mean, "std_ddof0": sd0, "ci95_half_ddof0": ci0,
                                "std_ddof1": sd1, "ci95_half_ddof1": ci1,
                                "formatted": f"{mean:.2f} ± {ci0:.2f}",
                                "formatted_ddof1": f"{mean:.2f} ± {ci1:.2f}"}
    return out


def content_errors(lang, rec, prod_rows):
    """行の中身の確かめ（出どころの確かめは g3_common.provenance_errors）."""
    idx = rec.get("pair_idx")
    row = rec.get("row")
    errs = []
    prod = prod_rows.get(idx, {})
    if idx not in prod_rows:
        errs.append(f"本番に pair_idx {idx} が無い")
    if prod.get("aligned") is not False:
        errs.append(f"本番の対 {idx} は aligned={prod.get('aligned')}（入れ替えるのは aligned=False の行だけ）")
    if rec.get("seed") not in G.REGEN_SEEDS:
        errs.append(f"seed {rec.get('seed')} は取り込まない（43 か 44 だけ）")
    if rec.get("status") != G.STATUS_VALID or rec.get("valid") is not True or rec.get("merge_eligible") is not True:
        errs.append(f"作り直しの行が有効でない: status={rec.get('status')} valid={rec.get('valid')}")
    if not rec.get("capture_ok"):
        errs.append(f"wav の写し取りが確かめに通っていない: {rec.get('capture_labels')}")
    wavs = rec.get("wavs") or {}
    if sorted(wavs) != sorted(CONDS):
        errs.append(f"wav が 5 本そろっていない: {sorted(wavs)}")
    if any(w.get("silent") is not False for w in wavs.values()):
        errs.append("無音の wav がある（共通の基準）")
    if row is None or row.get("aligned") is not True:
        errs.append("row が無いか aligned=True でない")
    else:
        if any(row.get(f"frames_{c}") is None for c in CONDS):
            errs.append(f"frames に None: {[row.get(f'frames_{c}') for c in CONDS]}")
        if row.get("pair_idx") != idx:
            errs.append("row.pair_idx が記録と違う")
        for k in ("text_a", "text_b", "word_a", "word_b"):
            if row.get(k) != prod.get(k):
                errs.append(f"{k} が本番の行と違う: {row.get(k)!r} != {prod.get(k)!r}")
        ref = next((r for r in prod_rows.values() if r.get("aligned") is True), None)
        if ref is not None and set(row) != set(ref):
            errs.append(f"行のキーが本番の整列済みの行と違う: 余分 {sorted(set(row) - set(ref))[:5]} "
                        f"不足 {sorted(set(ref) - set(row))[:5]}")
    return errs


def location_guard(data_root, out_dir, fake):
    """本物の記録は本物の出力へ、試運転の記録は試運転の出力へだけ行く."""
    rd, ro = os.path.realpath(data_root), os.path.realpath(out_dir)
    for d in G.PROD_DIR.values():
        pd = os.path.realpath(os.path.join(ROOT, d))
        assert ro != pd and not ro.startswith(pd + os.sep), f"本番の置き場には書かない: {out_dir}"
    dry = os.path.realpath(DRY)
    if rd == os.path.realpath(ROOT):
        if fake:
            raise SystemExit("[g3-merge] 本物の記録を試運転として取り込まない")
        want = os.path.realpath(os.path.join(ROOT, G.MERGED_ROOT_REL))
        if not ro.startswith(want + os.sep):
            raise SystemExit(f"[g3-merge] 本物の出力は {G.MERGED_ROOT_REL}/ の下だけ: {out_dir}")
    else:
        if not fake or not rd.startswith(dry + os.sep) or not ro.startswith(dry + os.sep):
            raise SystemExit(f"[g3-merge] 試運転の記録（{data_root}）は試運転の置き場（{G.DRYRUN_ROOT_REL}）の出力へだけ")


def seed42_repro_info(data_root, lang, idx):
    rec, rel, errs, chain = G.final_record(data_root, 42, lang, idx)
    if rec is None:
        return None
    return {"regen_row": rel, "status": rec.get("status"), "frames": rec.get("frames"),
            "production_frames": rec.get("production_frames"),
            "frames_match_production": rec.get("frames_match_production"), "chain": chain, "chain_errors": errs,
            "note": "reproduction check only; never merged"}


OUT_NAMES = ("results.json", "checkpoint.json", "table_values.json", "merge_record.json")
NON_REGEN_SILENCE_NOTE = (
    "The unified silence criterion ({crit}) was applied only to the regenerated pair (its 5 wavs are saved and "
    "were re-checked from the files at merge time). The other 299 pairs come from the 2026-05-20 production run, "
    "which saved no audio, so the criterion cannot be applied to them; for them, the evidence of non-silence is "
    "aligned=True in all 5 conditions (Whisper located the entity word in each of the 5 wavs).")


def merge(lang, out_dir, data_root=ROOT, code_root=ROOT, fake=False):
    location_guard(data_root, out_dir, fake)
    if os.path.lexists(out_dir):
        raise SystemExit(f"[g3-merge] 出力の置き場が既にある（上書きしない）: {out_dir}")

    prod_dir = os.path.join(ROOT, G.PROD_DIR[lang])
    md5_before = {n: G.md5_file(os.path.join(prod_dir, n)) for n in ("results.json", "checkpoint.json")}
    if md5_before != G.PROD_MD5[lang]:
        raise SystemExit(f"[g3-merge] 本番のファイルが 5/20 のものと違う: {md5_before}")
    res = G.read_json(os.path.join(prod_dir, "results.json"))
    ck = G.read_json(os.path.join(prod_dir, "checkpoint.json"))
    prod_rows = {r["pair_idx"]: r for r in res["rows"]}
    unaligned = sorted(i for i, r in prod_rows.items() if not r.get("aligned"))
    idx = G.TARGETS[lang]
    assert unaligned == [idx], f"本番の aligned=False の対が想定と違う: {unaligned}"

    # 0) 式の確認: 本番の行から計算し直した summary が本番の summary と一致するか
    re_sum = summarize(res["rows"], res["summary"])
    mism = [k for k in res["summary"] if res["summary"][k] != re_sum.get(k)]
    assert not mism and set(re_sum) == set(res["summary"]), f"summary の再計算が本番と一致しない: {mism[:5]}"
    ck_rows = {r["pair_idx"]: r for r in ck["rows"]}
    assert ck_rows.keys() == prod_rows.keys() and all(ck_rows[i] == prod_rows[i] for i in prod_rows), \
        "本番の checkpoint.json と results.json の行が一致しない"

    # 1) どの記録を使うか（決まりで自動に決まる）と、出どころ・中身の確かめ
    na = G.next_action(data_root, code_root, lang, idx)
    if na["action"] != "merge" or na["errors"]:
        raise SystemExit("[g3-merge] 取り込まない: action=%s%s\n  - %s" % (
            na["action"], f" ({na.get('note')})" if na.get("note") else "",
            "\n  - ".join(na["errors"] + na.get("code_drift", [])) or "-"))
    seed, rel = na["use"]
    rec = G.read_json(os.path.join(data_root, rel))
    errs = content_errors(lang, rec, prod_rows)
    if not fake and rec.get("selftest_FAKE"):
        errs.append("試運転の偽の記録")
    if errs:
        raise SystemExit("[g3-merge] 入れ替えない:\n  - " + "\n  - ".join(errs))

    new_row = rec["row"]
    tv_before = table_values(res["rows"])
    old_summary = res["summary"]
    old_row = copy.deepcopy(prod_rows[idx])
    for doc in (res, ck):
        j = [k for k, r in enumerate(doc["rows"]) if r["pair_idx"] == idx]
        assert len(j) == 1
        doc["rows"][j[0]] = new_row
    res["summary"] = summarize(res["rows"], old_summary)
    s = res["summary"]
    assert s["n_total"] == N_PAIRS and s["n_aligned"] == N_PAIRS, f"n_total={s['n_total']} n_aligned={s['n_aligned']}"
    tv_after = table_values(res["rows"])
    for metric, cols in tv_after.items():
        for col, v in cols.items():
            assert v["n"] == N_PAIRS, f"{metric}/{col} の n={v['n']}（300 のはず）"
    reproduced = {m: (tv_before[m]["intact"]["formatted"], tv_before[m]["blocked"]["formatted"])
                  == TABLE_299[lang][m] for m in TABLE_KEYS}

    rel_out = os.path.relpath(out_dir, ROOT)
    silence_note = NON_REGEN_SILENCE_NOTE.format(crit=SC.CRITERION_TEXT)
    # 4 本は、隣の新しい作業用の置き場にすべて書いてから、置き場ごと 1 回の rename で出す（途中で止まっても
    # 出力の置き場には何も現れない。作業用の置き場は消さずに残るが、次の取り込みは別の名前を使う）
    parent = os.path.dirname(os.path.abspath(out_dir))
    os.makedirs(parent, exist_ok=True)
    staging = os.path.join(parent, f"{os.path.basename(out_dir)}.staging_{time.strftime('%Y%m%d_%H%M%S')}_"
                                   f"{os.getpid()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(staging, exist_ok=False)
    G.write_json_new(res, os.path.join(staging, "results.json"), indent=2)
    G.write_json_new(ck, os.path.join(staging, "checkpoint.json"))
    tv = {"lang": lang, "fake_selftest": fake,
          "n300_source_note": f"tab:crossattn_mask の N=300 の値の出どころはこのファイル（{rel_out}/table_values.json）。"
                              f"本番の置き場 {G.PROD_DIR[lang]} は 299 対のまま。",
          "silence_check_scope": silence_note,
          "se_convention": "1.96 * np.std(ddof=0) / sqrt(n)（299 対の表と同じ）。ddof=1 は参考",
          "mel_similarity_key": "dtw_*_mel（DTW path-mean cosine, mel-spectrogram）",
          "seed_used": seed, "regen_row": rel,
          "before_n299": tv_before, "after_n300": tv_after,
          "table_299_reproduced_from_production": reproduced,
          "table_299_in_manuscript": TABLE_299[lang]}
    G.write_json_new(tv, os.path.join(staging, "table_values.json"))
    rec_out = {
        "item": "G3 Appendix B tab:crossattn_mask", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fake_selftest": fake, "lang": lang, "pair_idx": idx, "seed_used": seed, "attempt_used": rec.get("attempt"),
        "reason": "At seed 42 (production, 2026-05-20) Whisper could not locate the entity in one or more of the "
                  "5 wavs (row aligned=False, n_aligned=299). " + SEED_POLICY_TEXT +
                  f" The N=300 source for tab:crossattn_mask is {rel_out}/table_values.json; the production folder "
                  f"{G.PROD_DIR[lang]} is left unchanged at n_aligned=299. " + silence_note,
        "silence_check_scope": silence_note,
        "seed_policy": SEED_POLICY_TEXT,
        "decision": na,
        "regen_row": rel, "regen_row_md5": G.md5_file(os.path.join(data_root, rel)),
        "regen_seed_policy_evidence": rec.get("seed_policy"),
        "seed42_repro_check": seed42_repro_info(data_root, lang, idx),
        "valid_criterion": rec.get("valid_criterion"),
        "silence_criterion": {"id": SC.CRITERION_ID, "text": SC.CRITERION_TEXT},
        "production_dir": G.PROD_DIR[lang], "production_md5": md5_before,
        "regen_wavs": rec.get("wavs"), "regen_env": rec.get("env"), "regen_code_md5": rec.get("code_md5"),
        "old_row": old_row, "new_row": new_row,
        "summary_before": old_summary, "summary_after": s,
    }
    G.write_json_new(rec_out, os.path.join(staging, "merge_record.json"))
    md5_after = {n: G.md5_file(os.path.join(prod_dir, n)) for n in ("results.json", "checkpoint.json")}
    assert md5_after == md5_before, "本番のファイルが変わった（起きてはならない）"
    assert sorted(os.listdir(staging)) == sorted(OUT_NAMES), os.listdir(staging)
    if os.path.lexists(out_dir):  # rename は空の置き場を黙って置き換えるので、直前にもう一度確かめる
        raise SystemExit(f"[g3-merge] 出力の置き場が途中でできた（上書きしない）: {out_dir}。作業用 {staging} は残る")
    os.rename(staging, out_dir)

    print(f"[g3-merge] {lang} pair {idx} seed {seed} (attempt {rec.get('attempt')}) -> {out_dir}")
    print(f"  n_total {old_summary['n_total']} -> {s['n_total']}, n_aligned {old_summary['n_aligned']} -> {s['n_aligned']}")
    for m in TABLE_KEYS:
        b, a = tv_before[m], tv_after[m]
        print(f"  {m:15s} intact  {b['intact']['formatted']:>14s} (n={b['intact']['n']}) -> "
              f"{a['intact']['formatted']:>14s} (n={a['intact']['n']})   "
              f"blocked {b['blocked']['formatted']:>14s} -> {a['blocked']['formatted']:>14s}")
    print(f"  299 対の値が今の表と一致: {reproduced}")
    return tv


def status(data_root=ROOT, code_root=ROOT):
    out = {}
    for lang, idx in G.TARGETS.items():
        na = G.next_action(data_root, code_root, lang, idx)
        na["seed42_repro_check"] = seed42_repro_info(data_root, lang, idx)
        out[f"{lang}:{idx}"] = na
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------------------------------------------------------------- 試運転（偽の記録）
def _write_wav(path, amp, sr=16000, dur=0.6):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(sr * dur)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="FLOAT")


def make_fake_record(data_root, lang, idx, seed, attempt, status_, *, silent_wav=None, seed43_ref=None,
                     tweak=None):
    """本物の GPU の記録と同じ形の偽物を、試運転の置き場に作る（数値は隣の対の写しで意味を持たない）."""
    prod = {r["pair_idx"]: r for r in G.read_json(os.path.join(ROOT, G.PROD_DIR[lang], "results.json"))["rows"]}
    rel_dir = G.pair_dir_rel(seed, lang, idx, attempt)
    tdir = os.path.join(data_root, rel_dir)
    os.makedirs(tdir, exist_ok=False)
    row = None
    if status_ in ("valid", "silent", "unaligned", "capture_error", G.STATUS_ALIGN_EXC):
        row = copy.deepcopy(prod[idx - 1])
        row.update({k: prod[idx][k] for k in ("pair_idx", "text_a", "text_b", "word_a", "word_b")})
        if status_ in ("unaligned", G.STATUS_ALIGN_EXC):
            row = copy.deepcopy(prod[idx])  # 本番の aligned=False の行と同じ形
    rows = [{"pair_idx": i, "placeholder": True} for i in range(N_PAIRS) if i != idx] + ([row] if row else [])
    with open(os.path.join(tdir, "checkpoint.json"), "x") as f:
        json.dump({"rows": rows}, f)
    wavs = {}
    for c in CONDS:
        rel = os.path.join(rel_dir, "wav", f"s{idx:04d}_{c}.wav")
        _write_wav(os.path.join(data_root, rel), 0.0 if (silent_wav == c or status_ == "silent" and c == "tgt") else 0.3)
        wavs[c] = {"path": rel, "sr": 16000, "md5": G.md5_file(os.path.join(data_root, rel)),
                   **SC.assess_file(os.path.join(data_root, rel))}
    policy = {"seed": seed, "attempt": attempt}
    if seed == 44 and seed43_ref:
        policy.update({"seed43_regen_row": seed43_ref, "seed43_md5": G.md5_file(os.path.join(data_root, seed43_ref))})
    rec = {
        "item": "G3 Appendix B tab:crossattn_mask", "selftest_FAKE": True,
        "producer": G.PRODUCER, "mode": G.MODE_REPRO if seed == 42 else G.MODE_REGEN,
        "lang": lang, "pair_idx": idx, "seed": seed, "attempt": attempt, "pair_dir": rel_dir,
        "status": status_, "valid": status_ == "valid", "pair_invalid": status_ in G.PAIR_INVALID,
        "retry_same_seed": status_ in G.RETRY_SAME_SEED,
        "merge_eligible": seed in G.REGEN_SEEDS and status_ == "valid",
        "valid_criterion": "FAKE selftest", "silence_criterion_id": SC.CRITERION_ID,
        "seed_policy": policy, "frames": G.frames_of(row), "row": row, "wavs": wavs,
        "capture_ok": status_ != "capture_error", "capture_labels": list(CONDS),
        "pair_loop_errors": [],
        "whisper_errors": (["[Whisper] transcribe failed: FAKE CUDA out of memory"]
                           if status_ == G.STATUS_ALIGN_EXC else []),
        "span_preflight": {f"{lang}:{idx}": {"all_match": True}},
        "main_argv": G.prod_args(lang) + ["--output_dir", f"{G.CONTAINER_ROOT}/{rel_dir}"], "main_error": None,
        "code_md5": G.current_code_md5(ROOT),
        "env": {"project_root": G.CONTAINER_ROOT, "cuda_device": "FAKE-GPU (selftest)",
                "HF_HOME": G.CONTAINER_HF_HOME, "HF_HUB_OFFLINE": "1", "offline_preflight_ok": True,
                "offline_tokenizer_shim": [{"repo_id": G.TOKENIZER_ID, "snapshot": "/FAKE/snapshot"}],
                "print_taps": {"eval_module_tapped": True, "forced_alignment_utils_tapped": True},
                "hf_device_map": {"": "0"}, "device_offload": False},
    }
    if tweak:
        tweak(rec, tdir)
    G.write_json_new(rec, os.path.join(tdir, "regen_row.json"))
    return os.path.join(rel_dir, "regen_row.json")


def make_crashed_attempt(data_root, lang, idx, seed, attempt, with_row):
    """regen_row.json を書く前に止まった回（checkpoint.json だけがある）を作る."""
    prod = {r["pair_idx"]: r for r in G.read_json(os.path.join(ROOT, G.PROD_DIR[lang], "results.json"))["rows"]}
    tdir = os.path.join(data_root, G.pair_dir_rel(seed, lang, idx, attempt))
    os.makedirs(tdir, exist_ok=False)
    rows = [{"pair_idx": i, "placeholder": True} for i in range(N_PAIRS) if i != idx]
    if with_row:
        rows.append(copy.deepcopy(prod[idx]))
    with open(os.path.join(tdir, "checkpoint.json"), "x") as f:
        json.dump({"rows": rows}, f)


def policy_selftest(lang, idx, root, mk):
    """生成側の門（seed_policy）が決まりどおりに流す・止めるかを、偽の記録で確かめる."""
    import g3_regen_crossattn_pairs as WR  # 同じ置き場。import しても torch は読まない
    out = {}

    def check(tag, build, seed, attempt, want_allowed, pair=idx):
        d = root("policy_" + tag)
        build(d)
        errs, _ = WR.seed_policy(lang, pair, seed, attempt, d, ROOT)
        assert (not errs) == want_allowed, (tag, errs)
        out[tag] = {"allowed": not errs, "why": errs[:2]}

    def none(d):
        return None

    check("s43_a1_empty", none, 43, 1, True)
    check("s43_a1_aligned_production_pair", none, 43, 1, False, pair=idx - 1)
    check("s42_a1_any_pair_repro", none, 42, 1, True, pair=idx - 1)
    check("s45_outside_policy", none, 45, 1, False)
    check("s43_a1_already_exists", lambda d: mk(d, 43, 1, "valid"), 43, 1, False)
    check("s43_a2_after_unaligned", lambda d: mk(d, 43, 1, "unaligned"), 43, 2, False)
    check("s43_a2_after_valid", lambda d: mk(d, 43, 1, "valid"), 43, 2, False)
    check("s43_a2_after_capture_error", lambda d: mk(d, 43, 1, "capture_error"), 43, 2, True)
    check("s43_a2_after_crashed_no_record",
          lambda d: make_crashed_attempt(d, lang, idx, 43, 1, with_row=False), 43, 2, True)
    check("s43_a2_after_crashed_after_row",
          lambda d: make_crashed_attempt(d, lang, idx, 43, 1, with_row=True), 43, 2, False)
    check("s43_a3_skipping_a2", lambda d: mk(d, 43, 1, "capture_error"), 43, 3, False)
    check("s43_a2_after_alignment_exception", lambda d: mk(d, 43, 1, G.STATUS_ALIGN_EXC), 43, 2, True)
    check("s43_a3_after_two_alignment_exceptions",
          lambda d: (mk(d, 43, 1, G.STATUS_ALIGN_EXC), mk(d, 43, 2, G.STATUS_ALIGN_EXC)), 43, 3, False)
    check("s43_a3_after_alignment_exception_then_run_error",
          lambda d: (mk(d, 43, 1, G.STATUS_ALIGN_EXC), mk(d, 43, 2, "run_error")), 43, 3, True)
    check("s44_after_43_alignment_exception", lambda d: mk(d, 43, 1, G.STATUS_ALIGN_EXC), 44, 1, False)
    check("s44_after_43_unaligned_but_code_drift",
          lambda d: mk(d, 43, 1, "unaligned", tweak=lambda r, t: r["code_md5"].update({G.COMMON_REL: "0" * 32})),
          44, 1, False)
    check("s44_without_seed43", none, 44, 1, False)
    check("s44_after_43_valid", lambda d: mk(d, 43, 1, "valid"), 44, 1, False)
    check("s44_after_43_unaligned", lambda d: mk(d, 43, 1, "unaligned"), 44, 1, True)
    check("s44_after_43_generation_failed", lambda d: mk(d, 43, 1, "generation_failed"), 44, 1, True)
    check("s44_after_43_silent", lambda d: mk(d, 43, 1, "silent"), 44, 1, True)
    check("s44_after_43_capture_error", lambda d: mk(d, 43, 1, "capture_error"), 44, 1, False)
    check("s44_after_43_fake_mode",
          lambda d: mk(d, 43, 1, "unaligned", tweak=lambda r, t: r.update(mode=G.MODE_FAKE)), 44, 1, False)
    return out


def selftest():
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(DRY, f"merge_selftest_{ts}")
    os.makedirs(base, exist_ok=False)
    report = {"FAKE": "pipeline selftest only; numbers are meaningless", "dir": os.path.relpath(base, ROOT),
              "results": {}}

    def expect_refused(name, fn):
        try:
            fn()
        except SystemExit as e:
            return {"refused": True, "why": str(e)[:300]}
        raise AssertionError(f"{name}: 拒否されるはずが通った")

    for lang, idx in G.TARGETS.items():
        R = {}
        n = [0]

        def root(tag):
            n[0] += 1
            d = os.path.join(base, lang, f"{n[0]:02d}_{tag}")
            os.makedirs(d)
            return d

        def out(d):
            return os.path.join(d, "merged", f"{lang}cc_lp_n300")

        def mk(d, seed, attempt, st, **kw):
            return make_fake_record(d, lang, idx, seed, attempt, st, **kw)

        # --- 受け入れ ---
        d = root("accept_seed43_valid")
        mk(d, 43, 1, "valid")
        tv = merge(lang, out(d), data_root=d, fake=True)
        mr = G.read_json(os.path.join(out(d), "merge_record.json"))
        assert mr["seed_used"] == 43 and tv["after_n300"]["MCD (dB)"]["intact"]["n"] == N_PAIRS
        R["accept_seed43_valid"] = {"seed_used": 43, "table_299_reproduced": tv["table_299_reproduced_from_production"]}

        d = root("accept_seed44_after_invalid43")
        p43 = mk(d, 43, 1, "unaligned")
        mk(d, 44, 1, "valid", seed43_ref=p43)
        merge(lang, out(d), data_root=d, fake=True)
        assert G.read_json(os.path.join(out(d), "merge_record.json"))["seed_used"] == 44
        R["accept_seed44_after_invalid43"] = {"seed_used": 44}

        d = root("accept_seed43_attempt2_after_capture_error")
        mk(d, 43, 1, "capture_error")
        mk(d, 43, 2, "valid")
        merge(lang, out(d), data_root=d, fake=True)
        assert G.read_json(os.path.join(out(d), "merge_record.json"))["attempt_used"] == 2
        R["accept_seed43_attempt2_after_capture_error"] = {"attempt_used": 2}

        d = root("accept_seed42_repro_is_recorded_not_used")
        mk(d, 42, 1, "valid")
        mk(d, 43, 1, "valid")
        merge(lang, out(d), data_root=d, fake=True)
        mr = G.read_json(os.path.join(out(d), "merge_record.json"))
        assert mr["seed_used"] == 43 and mr["seed42_repro_check"]["status"] == "valid"
        R["accept_seed42_repro_is_recorded_not_used"] = {"seed_used": 43}

        d = root("accept_seed43_attempt2_after_alignment_exception")
        mk(d, 43, 1, G.STATUS_ALIGN_EXC)
        mk(d, 43, 2, "valid")
        merge(lang, out(d), data_root=d, fake=True)
        assert G.read_json(os.path.join(out(d), "merge_record.json"))["attempt_used"] == 2
        R["accept_seed43_attempt2_after_alignment_exception"] = {"attempt_used": 2}

        # silence_criterion.py のファイルの md5 は一致を求めない（ID と数値と wav の判定で確かめる）
        d = root("accept_silence_criterion_file_md5_differs")
        mk(d, 43, 1, "valid", tweak=lambda r, t: r["code_md5"].update({G.SILENCE_REL: "0" * 32}))
        merge(lang, out(d), data_root=d, fake=True)
        R["accept_silence_criterion_file_md5_differs"] = {"merged": True}

        # merge_record.json と table_values.json に、無音の基準を当てた範囲が書かれている
        mr = G.read_json(os.path.join(out(d), "merge_record.json"))
        tvj = G.read_json(os.path.join(out(d), "table_values.json"))
        assert "299 pairs" in mr["reason"] and "aligned=True in all 5 conditions" in mr["reason"], mr["reason"]
        assert tvj["silence_check_scope"] == mr["silence_check_scope"] and "no audio" in tvj["silence_check_scope"]
        assert sorted(os.listdir(out(d))) == sorted(OUT_NAMES)
        R["silence_scope_note_written"] = {"merge_record_reason": True, "table_values": True}

        # --- 拒否（何も書かない） ---
        def refused(tag, build, want=None):
            d = root(tag)
            build(d)
            r = expect_refused(tag, lambda: merge(lang, out(d), data_root=d, fake=True))
            assert not os.path.exists(out(d)), f"{tag}: 拒否したのに出力がある"
            assert not any(".staging_" in x for x in os.listdir(os.path.dirname(out(d)))) \
                if os.path.isdir(os.path.dirname(out(d))) else True, f"{tag}: 拒否したのに作業用の置き場がある"
            if want:
                assert want in r["why"], (tag, want, r["why"])
                r["want"] = want
            R[tag] = r

        refused("refuse_no_records", lambda d: None)
        refused("refuse_seed42_only", lambda d: mk(d, 42, 1, "valid"))
        refused("refuse_seed44_while_seed43_valid",
                lambda d: (mk(d, 43, 1, "valid"), mk(d, 44, 1, "valid")))
        refused("refuse_seed44_after_seed43_capture_error",
                lambda d: mk(d, 44, 1, "valid", seed43_ref=mk(d, 43, 1, "capture_error")))
        refused("refuse_seed44_without_seed43_reference",
                lambda d: (mk(d, 43, 1, "unaligned"), mk(d, 44, 1, "valid")))
        refused("refuse_seed44_invalid_too_author_decision",
                lambda d: mk(d, 44, 1, "unaligned", seed43_ref=mk(d, 43, 1, "silent")))
        refused("refuse_seed43_retry_after_outcome",  # 使えなかった回の後に同じ seed をもう一度（結果で選ぶ）
                lambda d: (mk(d, 43, 1, "unaligned"), mk(d, 43, 2, "valid")))
        refused("refuse_seed45_dir_exists", lambda d: (mk(d, 43, 1, "valid"), os.makedirs(
            os.path.join(d, "eval_results", "_regen2_crossattn_seed45", f"{lang}_p{idx:04d}"))))
        refused("refuse_mode_fake_plumbing",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r.update(mode=G.MODE_FAKE)))
        refused("refuse_span_preflight_fake",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r.update(span_preflight={"FAKE": True})))
        refused("refuse_host_not_container",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r["env"].update(project_root=ROOT, cuda_device=None)))
        refused("refuse_wrong_argv",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r.update(main_argv=r["main_argv"][:-2])))
        # G3 自身のコードが GPU の後に変わった（ほかは通る）: 著者の判断（流し直しの道は無い）
        refused("refuse_wrapper_md5_changed_author_decision",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r["code_md5"].update({G.WRAPPER_REL: "0" * 32})),
                want="action=stop_author_decision")
        refused("refuse_common_md5_changed_author_decision",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r["code_md5"].update({G.COMMON_REL: "0" * 32})),
                want="action=stop_author_decision")
        # G3 のコードの違いに加えてほかの外れもある: stop_error
        refused("refuse_wrapper_md5_changed_and_fake_span",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: (
                    r["code_md5"].update({G.WRAPPER_REL: "0" * 32}), r.update(span_preflight={"FAKE": True}))),
                want="action=stop_error")
        # seed 43 が使えない（unaligned）が、その判定を作ったコードが今と違う: seed 44 に進むかは著者の判断
        refused("refuse_seed43_unaligned_code_drift_author_decision",
                lambda d: mk(d, 43, 1, "unaligned",
                             tweak=lambda r, t: r["code_md5"].update({G.COMMON_REL: "0" * 32})),
                want="action=stop_author_decision")
        # 本番のファイルの md5 が固定の値と違う: stop_error
        refused("refuse_pinned_production_md5_mismatch",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r["code_md5"].update({G.INJ_REL: "0" * 32})),
                want="action=stop_error")
        refused("refuse_pinned_fa_md5_mismatch",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r["code_md5"].update({G.FA_REL: "0" * 32})),
                want="action=stop_error")
        refused("refuse_no_offline_shim_record",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r["env"].update(offline_tokenizer_shim=[])))
        refused("refuse_no_print_tap_record",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r["env"].pop("print_taps")),
                want="action=stop_error")
        # status は valid でも whisper_errors があれば取り込まない
        refused("refuse_valid_with_whisper_errors",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r.update(
                    whisper_errors=["[Whisper] transcribe failed: FAKE"])), want="action=stop_error")
        refused("refuse_whisper_errors_key_missing",
                lambda d: mk(d, 43, 1, "valid", tweak=lambda r, t: r.pop("whisper_errors")), want="action=stop_error")
        # Whisper の例外は同じ seed のやり直し（次の seed へ進まない）。2 回続いたら著者の判断
        refused("refuse_alignment_exception_rerun_same_seed",
                lambda d: mk(d, 43, 1, G.STATUS_ALIGN_EXC), want="action=rerun_seed43")
        refused("refuse_two_alignment_exceptions_author_decision",
                lambda d: (mk(d, 43, 1, G.STATUS_ALIGN_EXC), mk(d, 43, 2, G.STATUS_ALIGN_EXC)),
                want="action=stop_author_decision")
        # run_error（例外の記録つき）は stop_error ではなく同じ seed のやり直し
        refused("refuse_run_error_rerun_same_seed",
                lambda d: mk(d, 43, 1, "run_error", tweak=lambda r, t: r.update(
                    pair_loop_errors=["[237] err: FAKE"], main_error=None)), want="action=rerun_seed43")
        refused("refuse_seed44_alignment_exception_rerun_seed44",
                lambda d: mk(d, 44, 1, G.STATUS_ALIGN_EXC, seed43_ref=mk(d, 43, 1, "unaligned")),
                want="action=rerun_seed44")
        # 出力の置き場が既にある（空でも上書きしない）
        d = root("refuse_output_dir_exists")
        mk(d, 43, 1, "valid")
        os.makedirs(out(d))
        R["refuse_output_dir_exists"] = expect_refused(
            "out_exists", lambda: merge(lang, out(d), data_root=d, fake=True))
        assert os.listdir(out(d)) == [] and not [x for x in os.listdir(os.path.dirname(out(d))) if ".staging_" in x]

        # 書いている途中で止まった（3 本目で落とす）: 出力の置き場は現れず、次の取り込みはそのまま流せる
        d = root("crash_midway_then_rerun")
        mk(d, 43, 1, "valid")
        orig_write = G.write_json_new
        n_calls = [0]

        def flaky(obj, path, indent=1):
            n_calls[0] += 1
            if n_calls[0] == 3:
                raise RuntimeError("FAKE crash while writing (selftest)")
            return orig_write(obj, path, indent=indent)

        G.write_json_new = flaky
        try:
            merge(lang, out(d), data_root=d, fake=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("crash_midway: 落ちるはずが通った")
        finally:
            G.write_json_new = orig_write
        stg = [x for x in os.listdir(os.path.dirname(out(d))) if ".staging_" in x]
        assert not os.path.exists(out(d)) and len(stg) == 1, (os.listdir(os.path.dirname(out(d))))
        left = sorted(os.listdir(os.path.join(os.path.dirname(out(d)), stg[0])))
        merge(lang, out(d), data_root=d, fake=True)
        assert sorted(os.listdir(out(d))) == sorted(OUT_NAMES)
        R["crash_midway_then_rerun"] = {"output_after_crash": False, "staging_left": left,
                                        "rerun_output": sorted(os.listdir(out(d)))}

        def _row_not_ckpt(r, t):
            r["row"] = dict(r["row"], mcd_bl_inj=0.0)
        refused("refuse_row_differs_from_checkpoint", lambda d: mk(d, 43, 1, "valid", tweak=_row_not_ckpt))

        def _hide_silence(r, t):
            r["wavs"]["inj"]["silent"] = False
        refused("refuse_silent_wav_recomputed",
                lambda d: mk(d, 43, 1, "valid", silent_wav="inj", tweak=_hide_silence))

        def _md5_bad(r, t):
            r["wavs"]["bl"]["md5"] = "0" * 32
        refused("refuse_wav_md5_mismatch", lambda d: mk(d, 43, 1, "valid", tweak=_md5_bad))
        refused("refuse_status_silent", lambda d: mk(d, 43, 1, "silent"))

        # 記録を書く前に止まった回（時間切れなど）
        d = root("accept_seed43_attempt2_after_crashed_no_record")
        make_crashed_attempt(d, lang, idx, 43, 1, with_row=False)
        mk(d, 43, 2, "valid")
        merge(lang, out(d), data_root=d, fake=True)
        assert G.read_json(os.path.join(out(d), "merge_record.json"))["attempt_used"] == 2
        R["accept_seed43_attempt2_after_crashed_no_record"] = {"attempt_used": 2}
        refused("refuse_last_attempt_crashed_rerun_needed",
                lambda d: make_crashed_attempt(d, lang, idx, 43, 1, with_row=False))
        refused("refuse_retry_after_crashed_after_row",
                lambda d: (make_crashed_attempt(d, lang, idx, 43, 1, with_row=True), mk(d, 43, 2, "valid")))

        # 生成側の門（g3_regen_crossattn_pairs.seed_policy）: 流してよいかの判定
        R["generator_seed_policy"] = policy_selftest(lang, idx, root, mk)

        # 本物の出力へ偽の記録を入れようとしたら止まる
        d = root("refuse_fake_into_real_output")
        mk(d, 43, 1, "valid")
        R["refuse_fake_into_real_output"] = expect_refused(
            "fake_into_real", lambda: merge(lang, os.path.join(ROOT, G.MERGED_ROOT_REL, f"{lang}cc_lp_n300"),
                                            data_root=d, fake=True))
        # 中身の確かめ: 本番で整列済みの行は入れ替えない
        prod = {r["pair_idx"]: r for r in G.read_json(os.path.join(ROOT, G.PROD_DIR[lang], "results.json"))["rows"]}
        fake_rec = {"pair_idx": idx - 1, "seed": 43, "status": "valid", "valid": True, "merge_eligible": True,
                    "capture_ok": True, "wavs": {c: {"silent": False} for c in CONDS}, "row": prod[idx - 1]}
        ce = content_errors(lang, fake_rec, prod)
        assert any("aligned=True" in e for e in ce), ce
        R["refuse_replacing_aligned_production_row"] = {"refused": True, "why": ce[0]}
        # 今の本物の置き場（まだ作り直していない）: 取り込みは止まり、何も書かない
        real_out = os.path.join(ROOT, G.MERGED_ROOT_REL, f"{lang}cc_lp_n300")
        existed = os.path.exists(real_out)
        R["real_data_now"] = expect_refused("real_now", lambda: merge(lang, real_out))
        assert os.path.exists(real_out) == existed
        report["results"][lang] = R

    G.write_json_new(report, os.path.join(base, "selftest_report.json"))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    n_ok = sum(len(v) for v in report["results"].values())
    print(f"[g3-merge] selftest OK: {n_ok} checks; report {os.path.relpath(base, ROOT)}/selftest_report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(G.PROD_DIR))
    ap.add_argument("--status", action="store_true", help="今の記録と次にすることを表示するだけ")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.status:
        status()
        return
    assert args.lang, "--lang が要る（または --status / --selftest）"
    merge(args.lang, os.path.join(ROOT, G.MERGED_ROOT_REL, f"{args.lang}cc_lp_n300"))


if __name__ == "__main__":
    main()
