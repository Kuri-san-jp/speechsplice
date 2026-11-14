#!/usr/bin/env python3
"""付録 K の IPA 変化率を 6 言語で集計する（2026-09-29、CPU のみ）.

入力:
  <ipa_dir>/ipa_{lang}_n300.json  … appk_run_ipa.py（採点器 score_phoneme_asr_from_dir.py）の出力
  --silent_scan                    … appk_silent_scan.py の出力（採点に使った音声と同じ状態で取ったもの）
言語ごとに出すもの:
  n_expected（300）、n_scored（採点器の per_pair の数）、n_wav_missing（今ディスクにない・読めない wav の本数、
  600 本中）、n_pairs_not_scored、n_silent_pairs、n_md5_mismatch（無音の検査のあとで変わった wav）、
  n_valid_pairs（採点済み・wav あり・無音でない・md5 が検査時と一致）、
  IPA 変化率（有効な対のうち BL と注入側の IPA 文字列が違う割合）と Wilson の 95% 区間、
  平均 IPA 編集距離、参考として採点器の生の値（全対で数えたもの）。
  n300_ok = (n_valid_pairs == 300 かつ n_wav_missing == 0 かつ 有効な対に空の IPA が 0)。False の言語は表に
  使えない（N=300 必須）。空の IPA（採点器は 16 kHz で RMS<0.001 の音声を空にする）が有効な対に残るときは
  status を NOT_N300_EMPTY_IPA にし、その対の番号を出す（作り直すかは著者が決める。自動では作り直さない）。
有効な対の選び方は、wav の有無と無音の基準（無音の検査の silent_pairs と pairs_missing_or_unreadable）だけで、
採点の結果（IPA）は使わない。
all_n300_ok は 6 言語すべてを集計して、すべて n300_ok のときだけ True（一部の言語だけの集計では False）。

出力: --out（既定: 6 言語なら <ipa_dir>/ipa_summary_n300.json、一部の言語なら <ipa_dir>/ipa_summary_n300_<言語>.json）。
既にあれば <名前>.bak_<時刻> に写してから上書きする。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定するので、env や nice は付けない。
相対パスの引数はプロジェクト直下を基準に解く）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_aggregate_ipa.py --ipa_dir eval_results/_appk_redo --silent_scan eval_results/_appk_redo/silent_scan_n300.json
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_common as K  # noqa: E402

K.cap_threads()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

from appk_silent_scan import SETS, N_EXPECTED, PROJ, ER  # noqa: E402

Z95 = 1.959963984540054
LANGS = ("en", "fr", "de", "ko", "ja", "zh")
STEM_RE = re.compile(r"^s(\d{4})_bl\.wav$")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k, n, z=Z95):
    if n == 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def host_path(p):
    return p.replace("/mnt/", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ""), 1) if p.startswith("/mnt/") and PROJ != "/mnt" else p


def agg_lang(lang, ipa_json, scan, verify_md5=True):
    spec = SETS[lang]
    src_dir = K.set_dir(lang)
    out = {"lang": lang, "source_dir": src_dir, "ipa_json": ipa_json, "n_expected": N_EXPECTED}
    # wav の有無（今のディスクの状態）
    missing = []
    for i in range(N_EXPECTED):
        for side in ("bl", "inj"):
            n = f"s{i:04d}_{spec[side]}.wav"
            if not os.path.isfile(os.path.join(src_dir, n)):
                missing.append(n)
    out["n_wav_missing"] = len(missing)
    out["missing_files"] = missing
    # 無音の対と検査時の md5
    ls = scan["langs"][lang]
    silent = {p["pair_idx"] for p in ls["silent_pairs"]} | set(ls.get("pairs_missing_or_unreadable", []))
    scan_md5 = {k: v["md5"] for k, v in scan["per_wav"][lang].items()}
    out["n_silent_pairs"] = len(silent)
    out["silent_pairs"] = sorted(silent)
    # 採点器の出力
    if not os.path.isfile(ipa_json):
        out.update({"status": "ipa_json_missing", "n_scored": 0, "n_valid_pairs": 0, "n300_ok": False})
        return out
    d = json.load(open(ipa_json))
    rows = d["per_pair"]
    out["scorer_summary_raw"] = d.get("summary")
    by_idx, bad_keys = {}, []
    for r in rows:
        m = STEM_RE.match(os.path.basename(r["bl_wav"]))
        if not m:
            bad_keys.append(r["bl_wav"])
            continue
        i = int(m.group(1))
        if i in by_idx:
            bad_keys.append(f"duplicate {i}")
        by_idx[i] = r
    out["n_scored"] = len(rows)
    out["n_unparsable_or_duplicate_rows"] = len(bad_keys)
    not_scored = [i for i in range(N_EXPECTED) if i not in by_idx]
    out["n_pairs_not_scored"] = len(not_scored)
    out["pairs_not_scored"] = not_scored
    # 採点に使った wav が元の集合の wav そのものか（ja/zh は symlink の実体）と、検査時からの変化
    md5_mismatch, wrong_target = [], []
    valid, changed, eds, empty_ipa = [], 0, [], []
    for i in range(N_EXPECTED):
        if i not in by_idx:
            continue
        r = by_idx[i]
        ok = True
        for side, key in (("bl", "bl_wav"), ("inj", "inj_wav")):
            name = f"s{i:04d}_{spec[side]}.wav"
            src = os.path.join(src_dir, name)
            used = os.path.realpath(host_path(r[key]))
            if used != os.path.realpath(src):
                wrong_target.append(f"{r[key]} -> {used} (expected {src})")
                ok = False
                continue
            if not os.path.isfile(src):
                ok = False
                continue
            if verify_md5 and scan_md5.get(name) != md5_file(src):
                md5_mismatch.append(name)
                ok = False
        if i in silent:
            ok = False
        if ok:
            valid.append(i)
            changed += int(bool(r["ipa_changed"]))
            eds.append(r["ipa_edit_distance"])
            if r["bl_ipa"] == "" or r["inj_ipa"] == "":
                empty_ipa.append(i)
    out["n_wrong_target"] = len(wrong_target)
    out["wrong_target"] = wrong_target[:10]
    out["n_md5_mismatch_since_scan"] = len(md5_mismatch)
    out["md5_mismatch_since_scan"] = md5_mismatch[:20]
    n = len(valid)
    rate, lo, hi = wilson(changed, n)
    n300 = (n == N_EXPECTED and out["n_wav_missing"] == 0)
    ok = n300 and not empty_ipa
    out.update({
        "n_valid_pairs": n,
        "n_ipa_changed": changed,
        "ipa_change_rate_pct": None if rate is None else round(rate, 2),
        "wilson95_pct": None if rate is None else [round(lo, 2), round(hi, 2)],
        "mean_ipa_edit_distance": round(sum(eds) / n, 2) if n else None,
        "n_valid_with_empty_ipa": len(empty_ipa),
        "valid_pairs_with_empty_ipa": empty_ipa,
        "n300_ok": ok,
        "status": "ok" if ok else ("NOT_N300_EMPTY_IPA" if n300 else "NOT_N300"),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipa_dir", default="eval_results/_appk_redo")
    ap.add_argument("--silent_scan", required=True)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_md5", action="store_true")
    args = ap.parse_args()
    ipa_dir = args.ipa_dir if os.path.isabs(args.ipa_dir) else os.path.join(PROJ, args.ipa_dir)
    scan_path = args.silent_scan if os.path.isabs(args.silent_scan) else os.path.join(PROJ, args.silent_scan)
    scan = json.load(open(scan_path))
    langs = [l for l in args.langs.split(",") if l]
    for lang in langs:
        assert lang in LANGS, lang
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "ipa_dir": ipa_dir, "silent_scan": scan_path,
           "silent_criteria": scan.get("criteria"),
           "validity": "scored, both wavs present, neither side silent/missing/unreadable, md5 equal to silent-scan time",
           "ci": "Wilson score interval, 95%", "langs_requested": langs, "langs": {}}
    for lang in langs:
        res["langs"][lang] = agg_lang(lang, os.path.join(ipa_dir, f"ipa_{lang}_n300.json"), scan,
                                      verify_md5=not args.no_md5)
    # 6 言語すべてを集計したときだけ True になりうる（一部の言語だけで 6 言語の合格にしない）
    res["all_n300_ok"] = set(res["langs"]) == set(LANGS) and all(v.get("n300_ok") for v in res["langs"].values())

    hdr = f"{'lang':4} {'status':16} {'n_valid':>7} {'wav_miss':>8} {'not_scored':>10} {'silent':>6} " \
          f"{'md5chg':>6} {'changed':>7} {'IPA chg %':>9} {'Wilson95':>16} {'meanED':>6} {'raw %':>6}"
    print(hdr)
    for lang, v in res["langs"].items():
        ci = v.get("wilson95_pct")
        raw = (v.get("scorer_summary_raw") or {}).get("ipa_change_rate_pct")
        print(f"{lang:4} {v['status']:16} {v.get('n_valid_pairs', 0):7d} {v['n_wav_missing']:8d} "
              f"{v.get('n_pairs_not_scored', N_EXPECTED):10d} {v['n_silent_pairs']:6d} "
              f"{v.get('n_md5_mismatch_since_scan', 0):6d} {v.get('n_ipa_changed', 0):7d} "
              f"{'' if v.get('ipa_change_rate_pct') is None else format(v['ipa_change_rate_pct'], '.2f'):>9} "
              f"{'' if not ci else f'[{ci[0]:.2f}, {ci[1]:.2f}]':>16} "
              f"{'' if v.get('mean_ipa_edit_distance') is None else v['mean_ipa_edit_distance']:>6} "
              f"{'' if raw is None else raw:>6}")
    print(f"all_n300_ok = {res['all_n300_ok']}")

    if args.out:
        out = args.out
    elif set(langs) == set(LANGS):
        out = os.path.join(ipa_dir, "ipa_summary_n300.json")
    else:  # 一部の言語だけのときは、6 言語の既定の名前には書かない
        out = os.path.join(ipa_dir, f"ipa_summary_n300_{'_'.join(langs)}.json")
    out = out if os.path.isabs(out) else os.path.join(PROJ, out)
    if os.path.exists(out):
        bak = f"{out}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(out, bak)   # 写してから上書きする（移動・削除はしない）
        print(f"[backup] {out} -> {bak}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
