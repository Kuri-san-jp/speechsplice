#!/usr/bin/env python3
"""付録 K（6 言語の表）の IPA 変化の採点し直しに使う N=300 の音声 3,600 本の無音を調べる（2026-09-29、CPU のみ）.

対象（各 300 対 = BL 300 本 + 注入側 300 本）:
  en/fr/de/ko : eval_results/utmos_speaker_fixed_{lang}_n300_splice/s{i:04d}_{bl,inj}.wav  （checkpoint.json の rows）
  ja          : eval_results/cjk_n300_rewav/s{i:04d}_{bl,oeoi}.wav                        （checkpoint_ja.json の results）
  zh          : eval_results/cjk_n300_zh_v2/s{i:04d}_{bl,oeoi}.wav                        （checkpoint_zh.json の results）

無音の判定（1 本ごと。regen2 共通の部品 silence_criterion.assess_file を appk_common 経由で使う。著者決定 2026-09-29）:
  (a) RMS < 0.005 かつ peak < 0.05、または
  (b) 20 ms 区間のうち区間 RMS > 0.01 の割合（active_frac）< 0.05
対のどちらか一方でも無音・欠け・読めないなら、その対は使えない（作り直しの対象。appk_regen_pairs_seed.py --plan_only）。
(b) だけに当たった音声は、境目の例として pairs_silent_by_active_frac_only に別に並べる（判定は変えない）。

あわせて確かめること: checkpoint の行が 300 で pair_idx が 0..299、(text_a, text_b) の異なり数、wav の欠け、
読み込みの失敗、hard link（nlink>1）、サンプリング周波数。md5 も記録する（採点・反映の時点で音声が変わって
いないかを照合するため）。

音声・結果ファイルは読むだけで書き換えない。出力は --out の JSON 1 つだけ（既にあれば .bak_<時刻> に写してから上書きする）。
--out に既にある結果が --langs より多くの言語を含むときは止める（一部の言語だけで 6 言語の結果を置き換えない）。

使い方（ホストで直接。スレッド 13 と優先度 10 はスクリプトが自分で設定するので、env や nice は付けない。
相対パスの --out はプロジェクト直下を基準に解く）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_silent_scan.py --out eval_results/_appk_redo/silent_scan_n300.json
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_common as K  # noqa: E402

K.cap_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

# 旧来の import 先（appk_run_ipa / appk_prepare_stage / appk_aggregate_ipa などが使う）
SETS, N_EXPECTED, PROJ, ER = K.SETS, K.N_EXPECTED, K.PROJ, K.ER
md5_file = K.md5_file


def scan_lang(lang, spec=None):
    spec = spec or SETS[lang]
    d = K.set_dir(lang)
    rows = K.load_rows(lang)
    pidx = [r.get("pair_idx") for r in rows]
    pairs_txt = [(r.get("text_a"), r.get("text_b")) for r in rows]
    words = [(r.get("word_a"), r.get("word_b")) for r in rows]
    txt_by_idx = {r.get("pair_idx"): r.get("text_a") for r in rows}
    rep = {
        "lang": lang, "dir": d, "checkpoint": os.path.join(d, spec["ckpt"]),
        "bl_suffix": spec["bl"], "inj_suffix": spec["inj"],
        "n_ckpt_rows": len(rows),
        "pair_idx_is_0_to_299": sorted(pidx) == list(range(N_EXPECTED)),
        "n_distinct_text_pairs": len(set(pairs_txt)),
        "n_distinct_text_a": len(set(t[0] for t in pairs_txt)),
        "n_word_a_eq_word_b": sum(1 for a, b in words if a is not None and a == b),
        "n_text_a_eq_text_b": sum(1 for a, b in pairs_txt if a == b),
        "text_pairs_by_idx": {str(r.get("pair_idx")): [r.get("text_a"), r.get("text_b")] for r in rows},
    }
    per_wav, pairs, missing, read_fail = {}, [], [], []
    srs, nlinks = {}, 0
    for i in range(N_EXPECTED):
        entry = {"pair_idx": i, "silent": False, "sides": {}}
        for side in K.SIDES:
            name = K.wav_name(lang, i, side)
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                missing.append(name)
                entry["sides"][side] = {"file": name, "missing": True}
                continue
            try:
                st = K.wav_stats(p)   # 全 G 共通の無音の関数
            except Exception as ex:  # 読めない wav は欠けと同じに扱う
                read_fail.append({"file": name, "error": f"{type(ex).__name__}: {ex}"})
                entry["sides"][side] = {"file": name, "read_fail": True}
                continue
            m = {"sr": st["sr"], "dur": st["dur_sec"], "rms": st["rms"], "peak": st["peak"],
                 "active_frac": st["active_frac"]}
            stt = os.stat(p)
            m["nlink"] = int(stt.st_nlink)
            m["md5"] = md5_file(p)
            m["silent_reasons"] = list(st["silent_reason"])
            srs[m["sr"]] = srs.get(m["sr"], 0) + 1
            if stt.st_nlink > 1:
                nlinks += 1
            per_wav[name] = m
            entry["sides"][side] = {"file": name, **{k: m[k] for k in ("rms", "peak", "active_frac", "dur")},
                                    "silent_reasons": m["silent_reasons"]}
            if m["silent_reasons"]:
                entry["silent"] = True
        entry["missing_or_unreadable"] = any(s.get("missing") or s.get("read_fail")
                                             for s in entry["sides"].values()) or len(entry["sides"]) < 2
        pairs.append(entry)
    silent_pairs = [e for e in pairs if e["silent"]]
    bad_io = [e["pair_idx"] for e in pairs if e["missing_or_unreadable"]]
    act_only = sorted(e["pair_idx"] for e in silent_pairs
                      if all(v.get("silent_reasons") in (None, [], ["active_frac"]) for v in e["sides"].values()))
    rep.update({
        "n_wav_expected": 2 * N_EXPECTED,
        "n_wav_found": len(per_wav) + len(read_fail),
        "n_wav_missing": len(missing), "missing_files": missing,
        "n_read_fail": len(read_fail), "read_fail": read_fail,
        "sample_rates": {str(k): v for k, v in srs.items()},
        "n_hardlinked_wavs": nlinks,
        "n_silent_pairs": len(silent_pairs),
        "silent_pairs": [{"pair_idx": e["pair_idx"],
                          "text_a": txt_by_idx.get(e["pair_idx"]),
                          "sides": {s: v for s, v in e["sides"].items() if v.get("silent_reasons")}}
                         for e in silent_pairs],
        "pairs_silent_by_active_frac_only": act_only,
        "n_pairs_missing_or_unreadable": len(bad_io),
        "pairs_missing_or_unreadable": bad_io,
        "n_unusable_pairs": len({e["pair_idx"] for e in silent_pairs} | set(bad_io)),
        "rms_min": min((m["rms"] for m in per_wav.values()), default=None),
        "peak_min": min((m["peak"] for m in per_wav.values()), default=None),
        "active_frac_min": min((m["active_frac"] for m in per_wav.values()), default=None),
        "rms_median": float(np.median([m["rms"] for m in per_wav.values()])) if per_wav else None,
        "lowest5_by_rms": sorted(({"file": k, "rms": v["rms"], "peak": v["peak"],
                                   "active_frac": v["active_frac"]} for k, v in per_wav.items()),
                                 key=lambda z: z["rms"])[:5],
        "lowest5_by_active_frac": sorted(({"file": k, "rms": v["rms"], "peak": v["peak"],
                                           "active_frac": v["active_frac"]} for k, v in per_wav.items()),
                                         key=lambda z: z["active_frac"])[:5],
    })
    return rep, per_wav


def run_scan(langs, quiet=False):
    t0 = time.time()
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "criteria": {"silent_if": K.SILENCE_CRITERION["rule"], "criterion": K.SILENCE_CRITERION,
                        "helper": K.SILENCE_HELPER,
                        "pair_silent_if": "BL or injected side is silent",
                        "pair_unusable_if": "silent, missing or unreadable on either side"},
           "er_root": K.ER, "langs": {}, "per_wav": {}}
    for lang in langs:
        rep, per_wav = scan_lang(lang)
        res["langs"][lang] = rep
        res["per_wav"][lang] = per_wav
        if not quiet:
            print(f"[{lang}] rows={rep['n_ckpt_rows']} idx_ok={rep['pair_idx_is_0_to_299']} "
                  f"distinct(text_a,text_b)={rep['n_distinct_text_pairs']} wav_found={rep['n_wav_found']} "
                  f"missing={rep['n_wav_missing']} read_fail={rep['n_read_fail']} sr={rep['sample_rates']} "
                  f"hardlinked={rep['n_hardlinked_wavs']} silent_pairs={rep['n_silent_pairs']} "
                  f"(active_frac only: {rep['pairs_silent_by_active_frac_only']}) "
                  f"rms_min={rep['rms_min']} peak_min={rep['peak_min']} act_min={rep['active_frac_min']}",
                  flush=True)
            for sp in rep["silent_pairs"]:
                print(f"    silent pair {sp['pair_idx']}: {json.dumps(sp['sides'], ensure_ascii=False)}", flush=True)
    res["total"] = {k: sum(r[k] for r in res["langs"].values())
                    for k in ("n_wav_expected", "n_wav_found", "n_wav_missing", "n_read_fail", "n_silent_pairs",
                              "n_unusable_pairs")}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def resolve_out(out):
    return out if os.path.isabs(out) else os.path.join(PROJ, out)


def write_scan(res, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):  # 作り直しの前後で同じ名前に書くときは、前の検査結果を残す
        try:
            old_langs = set(json.load(open(out)).get("langs", {}))
        except Exception:  # noqa: BLE001
            old_langs = set()
        extra = old_langs - set(res["langs"])
        if extra:
            raise SystemExit(f"{out} には {sorted(extra)} の結果もある。一部の言語だけで置き換えない"
                             f"（--langs を全言語にするか、別の --out にする）。")
        bak = f"{out}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(out, bak)   # 写してから上書きする（移動・削除はしない）
        print(f"[backup] {out} -> {bak}")
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default=",".join(K.LANGS))
    args = ap.parse_args()
    langs = args.langs.split(",")
    for lang in langs:
        assert lang in SETS, lang
    out = resolve_out(args.out)
    if os.path.exists(out):   # 走査の前に止める（時間を無駄にしない）
        old_langs = set(json.load(open(out)).get("langs", {}))
        if old_langs - set(langs):
            raise SystemExit(f"{out} には {sorted(old_langs - set(langs))} の結果もある。一部の言語だけで置き換えない。")
    res = run_scan(langs)
    write_scan(res, out)
    print(f"[total] {res['total']}  elapsed={res['elapsed_sec']}s  -> {out}")


if __name__ == "__main__":
    main()
