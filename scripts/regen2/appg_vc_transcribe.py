#!/usr/bin/env python3
"""付録 G 声まね: 5 行 × 300 本を同じ関数で書き起こす（2026-09-29 作成）.

書き起こしの関数（D1）: scripts/rome_encoder_t5gemma.py の load_whisper / transcribe_with_whisper
  （openai/whisper-large-v3、fp16、generate(language="ja", task="transcribe")、ビーム・サンプリングの指定なし = greedy）。
  既存の Baseline / Text Hint の wav（5/15 の回の中で別の手順で書き起こしていた）も、この関数で書き起こし直す。
読む wav: <root>/selection.json（appg_vc_select.py）の 5 行 × 300。Baseline / Text Hint は本番の置き場
  （差し替えが要る対は apply が済んでいること = ready_for_transcription）。書き起こす前に、今の置き場から selection を
  作り直して selection.json と同じか、wav の md5 が同じか、無音でないかを確かめる（違えば GPU を使う前に止まる）。
出力: 採点器 score_trm_g2pcer_nopyojt.py が読める 1 つの JSON（rows[].pair_idx / word_a / word_b と
  bl_transcript / g2p_transcript / b7_transcript / rome_transcript / t5_inj_transcript。b1 は書かない = 表に無い）。
  行ごとに wav_<row>（パス・md5・seed・長さ・無音の判定・書き起こした回）。md5 が同じで書き起こし済みのものは飛ばす
  （止まったら同じ命令で続きから）。前の出力は .prev_<run_id> に写して残す。
pyopenjtalk: 書き起こしには使わないので import を塞ぐ（D4）。
textedit の決め方（--textedit_unit、既定 b7 = Text Hint だけを作り直す）は selection.json の記録と同じでなければ止まる。

GPU（プロジェクト直下から、GPU 1 枚）:
  SLURM_TIME=01:00:00 slurm_run.sh python3 -u scripts/regen2/appg_vc_transcribe.py
CPU の空運転（モデルを読まない）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appg_vc_transcribe.py --dry_run
"""
import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules:
    raise SystemExit("[threads] torch was imported before the thread cap")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
if os.path.isdir("/mnt/scripts"):
    os.environ["HF_HOME"] = "/mnt/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/mnt/.cache/huggingface/hub"
    os.environ["TORCH_HOME"] = "/mnt/.cache/torch"
    os.environ["XDG_CACHE_HOME"] = "/mnt/.cache"
    os.environ["NUMBA_CACHE_DIR"] = "/mnt/.cache/numba"
    os.makedirs("/mnt/.cache/numba", exist_ok=True)
else:
    os.environ["HF_HOME"] = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface/hub")
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/numba")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.modules["pyopenjtalk"] = None  # 書き起こしでは使わない（D4）

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appg_vc_common as V  # noqa: E402

AC = V.AC


def default_out(root):
    return os.path.join(root, "transcripts_5rows_vc.json")


def sel_rows_signature(sel):
    return {row: [(e["pair_idx"], e["path"], e["md5"], e["seed"]) for e in sel["rows"][row]["pairs"]]
            for row in V.ROW_ORDER}


def check_selection(root, rows_sel):
    """selection.json を読み、今の置き場から作り直した selection と同じか。戻り値 (sel, problems)。"""
    sp = os.path.join(root, "selection.json")
    sel = V.load_json(sp)
    if sel is None:
        return None, [f"selection.json が無い（appg_vc_select.py を先に流す）: {V.rel_to_proj(sp)}"]
    problems = []
    if not sel.get("complete"):
        problems.append(f"selection が complete でない: {sel.get('problems')}")
        return sel, problems
    if sel.get("textedit_unit", "bl_b1_b7") != V.textedit_unit():
        problems.append(f"selection.json の textedit_unit={sel.get('textedit_unit', 'bl_b1_b7')} が今の "
                        f"--textedit_unit {V.textedit_unit()} と違う（同じ決め方で select をもう一度流す）")
        return sel, problems
    live = V.build_selection(root)
    if not live.get("complete"):
        problems.append(f"今の置き場では作り直しが終わっていない: {live.get('problems')}")
        return sel, problems
    if not live.get("ready_for_transcription"):
        problems.append(f"Baseline / Text Hint の差し替え（apply）が済んでいない: {live.get('textedit_needs_apply')}")
    a, b = sel_rows_signature(sel), sel_rows_signature(live)
    diff = [row for row in rows_sel if a[row] != b[row]]
    if diff:
        problems.append(f"selection.json が今の置き場と違う行: {diff}（appg_vc_select.py をもう一度流す）")
    return live, problems


def base_rows(pairs, old):
    rows = {r["pair_idx"]: r for r in (old or {}).get("rows", [])}
    out = {}
    for p in pairs:
        i = p["pair_idx"]
        r = rows.get(i, {})
        r.update({"pair_idx": i, "word_a": p["word_a"], "word_b": p["word_b"], "text_a": p["text_a"], "text_b": p["text_b"]})
        out[i] = r
    return out


def plan_todo(sel, rows, rows_sel):
    """書き起こす wav の一覧と、守り（欠け・md5 の違い・無音）。"""
    todo, problems = [], []
    for row in rows_sel:
        key = V.ROWS[row][2]
        for e in sel["rows"][row]["pairs"]:
            i = e["pair_idx"]
            path = V.proj_path(e["path"])
            if not os.path.isfile(path):
                problems.append(f"{row} {i}: wav が無い {e['path']}")
                continue
            md5 = V.md5_file(path)
            if md5 != e["md5"]:
                problems.append(f"{row} {i}: md5 が selection と違う")
                continue
            st = AC.wav_stats(path)
            if st["silent"]:
                problems.append(f"{row} {i}: 無音 {st['silent_reasons']}")
                continue
            prev = rows[i].get(f"wav_{row}") or {}
            if prev.get("md5") == md5 and (rows[i].get(key) or "").strip() and not rows[i].get(f"{key}_error"):
                continue
            todo.append((row, i, path, md5, st, e["seed"]))
    return todo, problems


def record_result(row_d, row, path, md5, st, seed, run_id, text=None, error=None):
    key = V.ROWS[row][2]
    prev = (row_d.get(f"wav_{row}"), row_d.get(key), row_d.get(f"{key}_error"))
    if any(x is not None for x in prev):
        row_d.setdefault(f"{key}_history", []).append(
            {"wav": prev[0], "transcript": prev[1], "error": prev[2], "replaced_by_run": run_id})
    w = {"path": V.rel_to_proj(path), "md5": md5, "seed": seed, **V.pick_stats(st), "transcribed_run_id": run_id}
    if error is None:
        row_d[key] = text
        row_d.pop(f"{key}_error", None)
    else:
        row_d[key] = ""
        row_d[f"{key}_error"] = str(error)[:300]
        w["transcribe_error"] = True
    row_d[f"wav_{row}"] = w


def summarize(rows):
    s = {}
    for row in V.ROW_ORDER:
        key = V.ROWS[row][2]
        s[row] = {"transcript_key": key,
                  "n_wav_recorded": sum(1 for i in rows if rows[i].get(f"wav_{row}")),
                  "n_nonempty_transcript": sum(1 for i in rows if (rows[i].get(key) or "").strip()),
                  "n_error_keys": sum(1 for i in rows if rows[i].get(f"{key}_error")),
                  "seed_counts": {str(sd): sum(1 for i in rows if (rows[i].get(f"wav_{row}") or {}).get("seed") == sd)
                                  for sd in (42, 43, 44)}}
    return s


def run_transcription(todo, rows, doc, out_path, transcribe_fn, run_id, stop_after_min=None):
    t0 = time.time()
    n_err = 0
    for k, (row, i, path, md5, st, seed) in enumerate(todo):
        if stop_after_min and (time.time() - t0) / 60 > stop_after_min:
            print(f"[time] {stop_after_min} 分を超えたので止める（同じ命令で続きから）", flush=True)
            break
        try:
            text = transcribe_fn(path)
        except Exception as e:  # noqa: BLE001
            record_result(rows[i], row, path, md5, st, seed, run_id, error=e)
            n_err += 1
        else:
            record_result(rows[i], row, path, md5, st, seed, run_id, text=text)
        if (k + 1) % 50 == 0 or k + 1 == len(todo):
            doc["rows"] = [rows[j] for j in sorted(rows)]
            doc["summary"] = summarize(rows)
            V.atomic_write_json(out_path, doc)
            el = time.time() - t0
            print(f"  [{k + 1}/{len(todo)}] elapsed={el / 60:.1f}m errors={n_err}", flush=True)
    doc["rows"] = [rows[j] for j in sorted(rows)]
    doc["summary"] = summarize(rows)
    doc["provenance"]["n_errors_this_run"] = n_err
    doc["provenance"]["elapsed_min_this_run"] = round((time.time() - t0) / 60, 2)
    V.atomic_write_json(out_path, doc)
    return n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=V.OUT_ROOT_REL)
    ap.add_argument("--output", default=None, help="既定 <root>/transcripts_5rows_vc.json")
    ap.add_argument("--rows", default=",".join(V.ROW_ORDER))
    ap.add_argument("--stop_after_min", type=float, default=50.0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dryrun_out", default=None, help=f"既定 {V.DRYRUN_REL}/transcribe_dryrun.json")
    ap.add_argument("--textedit_unit", default=V.TEXTEDIT_UNIT_DEFAULT, choices=sorted(V.TEXTEDIT_UNITS),
                    help="selection を作ったときの textedit の決め方（既定 b7 だけ）")
    args = ap.parse_args()
    V.set_textedit_unit(args.textedit_unit)
    root = V.root_path(args.output_root)
    rows_sel = [x for x in args.rows.split(",") if x]
    assert all(x in V.ROWS for x in rows_sel), rows_sel
    out_path = V.proj_path(args.output) if args.output else default_out(root)
    V.assert_writable_dir(os.path.dirname(out_path), args.output_root)
    run_id = V.new_run_id("asr")

    sel, problems = check_selection(root, rows_sel)
    pairs = V.load_pairs()
    old = V.load_json(out_path)
    rows = base_rows(pairs, old)
    todo = []
    if not problems:
        todo, p2 = plan_todo(sel, rows, rows_sel)
        problems += p2

    if args.dry_run:
        pre_ok, pre = AC.offline_preflight(AC.ASR_REPOS)
        rep = {"mode": "dry_run", "root": V.rel_to_proj(root), "output": V.rel_to_proj(out_path),
               "ready": not problems, "problems": problems[:30], "n_to_transcribe": len(todo),
               "per_row": {r: sum(1 for t in todo if t[0] == r) for r in rows_sel},
               "offline_preflight_ok": pre_ok, "offline_preflight": pre,
               "function": "rome_encoder_t5gemma.transcribe_with_whisper (whisper-large-v3, fp16, language=ja, greedy)"}
        op = V.proj_path(args.dryrun_out) if args.dryrun_out else os.path.join(V.proj_path(V.DRYRUN_REL),
                                                                               "transcribe_dryrun.json")
        V.assert_writable_dir(os.path.dirname(op), args.output_root)
        rep["textedit_unit"] = V.textedit_unit()
        os.makedirs(os.path.dirname(op), exist_ok=True)
        V.atomic_write_json(op, rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "offline_preflight"}, ensure_ascii=False, indent=1))
        print(f"[dry_run] saved {op}")
        if not pre_ok:
            raise SystemExit("[dry_run] Whisper が手元のキャッシュに無い")
        return
    if problems:
        raise SystemExit(f"[guard] 書き起こしの前に: {problems[:20]}")
    pre_ok, pre = AC.offline_preflight(AC.ASR_REPOS)
    if not pre_ok:
        raise SystemExit(f"[offline] Whisper が手元のキャッシュに無い: {pre}")

    import torch as _torch_pre  # noqa: F401
    import torch.utils._pytree as _pt
    if not hasattr(_pt, "register_constant"):
        _pt.register_constant = lambda *a, **kw: None
    import torch
    torch.set_num_threads(13)
    sys.path.insert(0, os.path.join(V.PROJ, "scripts"))
    import hashlib
    import inspect
    import transformers
    import rome_encoder_t5gemma as R
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm, wp = R.load_whisper(device)
    gc_ = wm.generation_config
    prov = {"run_id": run_id, "date": V.now(), "script": os.path.abspath(__file__),
            "script_md5": V.md5_file(os.path.abspath(__file__)),
            "appg_vc_common_md5": V.md5_file(os.path.join(_HERE, "appg_vc_common.py")),
            "function": "rome_encoder_t5gemma.transcribe_with_whisper",
            "function_source_md5": hashlib.md5(inspect.getsource(R.transcribe_with_whisper).encode()).hexdigest(),
            "module_md5": V.md5_file(R.__file__), "whisper_model": "openai/whisper-large-v3", "dtype": "float16",
            "language": "ja", "task": "transcribe", "num_beams": getattr(gc_, "num_beams", None),
            "do_sample": getattr(gc_, "do_sample", None), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "pyopenjtalk_blocked": sys.modules.get("pyopenjtalk", "absent") is None,
            "selection_md5": V.md5_file(os.path.join(root, "selection.json")), "offline_preflight": pre,
            "n_to_transcribe": len(todo)}
    hist = list((old or {}).get("provenance_history", []))
    if old and old.get("provenance"):
        hist.append(old["provenance"])
    if old is not None:
        kept = f"{out_path}.prev_{run_id}"
        shutil.copy2(out_path, kept)
        prov["previous_output_kept_as"] = V.rel_to_proj(kept)
    doc = {"note": "Appendix G five rows, voice-clone condition of eval_baseline_text_edit.py (5/15), one ASR function. "
                   "Pass this file to all four scorer inputs.", "provenance": prov, "provenance_history": hist}
    print(f"[Info] {len(todo)} wavs to transcribe; num_beams={prov['num_beams']} do_sample={prov['do_sample']}", flush=True)
    n_err = run_transcription(todo, rows, doc, out_path,
                              lambda p: R.transcribe_with_whisper(p, wm, wp, device), run_id, args.stop_after_min)
    for r, v in doc["summary"].items():
        print(f"  {r:<13} recorded {v['n_wav_recorded']:>3} nonempty {v['n_nonempty_transcript']:>3} "
              f"errors {v['n_error_keys']} seeds {v['seed_counts']}")
    print(f"Saved: {out_path} (errors this run: {n_err})")


if __name__ == "__main__":
    main()
