#!/usr/bin/env python3
"""付録 G 声まね: 新しい SpeechSplice の BL（声まね、seed 42、text_a）が、既存の Baseline の wav（5/15 の回）と
同じかを確かめる（2026-09-29 作成、CPU のみ）.

なぜ同じになるはずか: 両方とも同じ声まねの合成（参照・見本の文・100 秒・参照の話速による長さ・復号の定数・
  合成の直前の seed_everything(42)・vendor の save_audio）で text_a を読む。SpeechSplice の BL は注入の hook を付けない。
完全一致しないことがある理由: GPU の bf16 の計算は同じ入力でも毎回同じとは限らない（前例: 同じコードで JA を作り直して
  md5 一致 296/300。note/2026-09-18_silentwav_report.md）。コンテナの版の変化もありうる。
比べるもの（対ごと）: md5、標本数（長さ）、生成したトークン数（5/15 は _run.log の [Speed] 行 3i+1、新しい回は
  長さから (長さ秒 × 50) + 1 と逆算。5/15 の 900 本はこの関係が全部成り立つ）、新しい回の長さの推定値（target_gen_sec）。
Baseline 側が差し替え済み（apply）の対は、退避した元の wav と比べる（b7 だけを差し替えた対は本番の bl がそのまま元の wav）。

使い方（ホストで直接、CPU のみ）:
  確かめの小さな回: taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_bl_check.py --output_root eval_results/_appg_vc_smoke
  本番の 300 対:    taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_bl_check.py
書く物: <root>/bl_check/bl_check_<時刻>.json（上書きしない）。
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
sys.modules["pyopenjtalk"] = None

import argparse  # noqa: E402
import json  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_vc_common as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=V.OUT_ROOT_REL, help="splice/seed42 のある置き場（本番か確かめの小さな回）")
    ap.add_argument("--tag", default="seed42", choices=["seed42", "check42"])
    ap.add_argument("--apply_root", default=V.OUT_ROOT_REL, help="Baseline の差し替えの記録のある置き場（既定は本番の出力の置き場）")
    args = ap.parse_args()
    import soundfile as sf
    root = V.root_path(args.output_root)
    applied = V.load_applied(args.apply_root)  # 差し替え済みの対は退避した元の wav と比べる
    lg = V.parse_baseline_log()
    base_dir = V.baseline_dir()
    rows, same_md5, same_len, n = [], 0, 0, 0
    for i in range(V.N_PAIRS):
        ip = V.item_path("splice", args.tag, i, root)
        rec = V.load_json(ip)
        if not rec or rec.get("status") != "success":
            continue
        new = V.file_path("splice", args.tag, "bl", i, root)
        a = applied.get(i)
        if a and "bl" not in (a.get("files") or {}):
            a = None  # b7 だけを差し替えた対（著者の決定 2026-09-29）: 本番の bl は元のまま
        old = V.proj_path(a["files"]["bl"]["backup"]) if a else os.path.join(base_dir, V.file_name("textedit", "bl", i))
        if not (os.path.isfile(new) and os.path.isfile(old)):
            rows.append({"pair_idx": i, "error": "missing file", "new": V.rel_to_proj(new), "old": V.rel_to_proj(old)})
            continue
        n += 1
        mi, mo = sf.info(new), sf.info(old)
        m_new, m_old = V.md5_file(new), V.md5_file(old)
        dn, do = mi.frames / mi.samplerate, mo.frames / mo.samplerate
        tok_old = (lg["tokens"].get(i) or {}).get("bl")
        r = {"pair_idx": i, "md5_identical": m_new == m_old, "frames_new": mi.frames, "frames_old": mo.frames,
             "sr_new": mi.samplerate, "sr_old": mo.samplerate, "subtype_new": mi.subtype, "subtype_old": mo.subtype,
             "dur_new": round(dn, 4), "dur_old": round(do, 4),
             "tokens_old_log": tok_old, "tokens_old_from_len": round(do * 50 + 1, 3),
             "tokens_new_from_len": round(dn * 50 + 1, 3),
             "target_gen_sec_new": ((rec.get("files") or {}).get("bl") or {}).get("target_gen_sec"),
             "old_from_backup": bool(a)}
        same_md5 += r["md5_identical"]
        same_len += mi.frames == mo.frames
        rows.append(r)
    rep = {"created": V.now(), "root": V.rel_to_proj(root), "tag": args.tag, "baseline_dir": V.rel_to_proj(base_dir),
           "n_compared": n, "n_md5_identical": same_md5, "n_same_length": same_len,
           "md5_mismatch_pairs": [r["pair_idx"] for r in rows if r.get("md5_identical") is False],
           "length_mismatch_pairs": [r["pair_idx"] for r in rows if "frames_new" in r and r["frames_new"] != r["frames_old"]],
           "may_log_speed_lines": lg["n_speed_lines"],
           "note": "bit-identity is not guaranteed (bf16 GPU nondeterminism; container changes). "
                   "Report the counts; for mismatches compare length/tokens and, after transcription, the transcripts.",
           "rows": rows}
    d = os.path.join(root, "bl_check")
    V.assert_writable_dir(d, args.output_root)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"bl_check_{V.new_run_id(args.tag)}.json")
    V.atomic_write_json(out, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False, indent=1))
    print(f"[bl_check] saved {out}")


if __name__ == "__main__":
    main()
