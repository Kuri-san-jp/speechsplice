#!/usr/bin/env python3
"""付録 K 中国語の注入成功率の判定し直し: 7/28 の判定器をそのまま GPU で流すラッパー（2026-09-29）.

判定器 scripts/analysis/zh_ctc_lr_judge.py は 1 バイトも変えない（md5 7e823d42… を確かめてから runpy で呼ぶ）。
引数は 7/28 と同じ形（--gen_dir <stage> --gold eval_results/zh_polyphone_gold_readings_v2.json）で、
gen_dir だけを appk_zhjudge_stage.py が作った新しいフォルダ（途中保存なし）にする。判定器はそこに
ctc_lr_judge.json と ctc_lr_v3_partial.json を書く（本番の cjk_n300_zh_v2 には書かない）。

このラッパーがすること（判定器が自分ではしないことだけ）:
  * torch を読む前に: スレッド上限 13（OMP/MKL/OPENBLAS/NUMEXPR/VECLIB）、HF_HOME=/mnt/.cache/huggingface、
    HF_HUB_CACHE、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、NUMBA_CACHE_DIR=/mnt/.cache/numba、
    sys.modules['pyopenjtalk']=None（判定に pyopenjtalk は要らない。D4）。ZH_GEN_DIR / ZH_GOLD_PATH は消す。
  * torch.set_num_threads(13)。
  * 走らせる前の確認（1 つでも外れたら GPU を使う前に止まる）:
      判定器・gold v2 の md5、モデル snapshot が 7840ab2a… だけで各ファイルの md5 が同じ、whisper fork の md5、
      stage の 600 本と checkpoint_zh.json が manifest の md5 のまま、stage に判定器の出力（途中保存を含む）が無い、
      CUDA が使える（使えないと判定器は黙って CPU で動くので止める）、コンテナの中（/mnt）で動いている。
  * 判定器と同じ関数（sent_token_ids / tokenize_syllable）で 300 対の仮説を組み、tokenize の失敗 0、
    声調なしの欄が付くかどうかが 7/28 の行と 300/300 一致することを確かめる（CPU だけ。モデルは読まない）。
    仮説の md5 と pypinyin の版を記録する（コンテナの pypinyin が 7/28 と違うかの手がかり）。
  * 実行の記録 <stage>/zhjudge_run_meta_<時刻>.json（環境の版、GPU 名、各 md5、出力の md5、かかった時間）。

GPU（プロジェクト直下から。GPU 1 枚。7/28 は 600 本で約 7.5 分）:
  SLURM_TIME=00:20:00 slurm_run.sh python3 -u scripts/regen2/appk_zhjudge_run_gpu.py --stage_dir eval_results/_appk_zhjudge/rejudge_ctc
予行（ホストの CPU。モデルは読まない。何も書かない）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_zhjudge_run_gpu.py --dry_run
  （stage がまだ無いときは、本番の checkpoint_zh.json で仮説の確認だけをする）
集計は CPU で appk_zhjudge_aggregate.py。
"""
import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules or "numpy" in sys.modules:
    raise SystemExit("[threads] torch / numpy がスレッド上限の前に import されている")
_PROJ = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
if _PROJ != "/mnt":
    # ホスト（ログインノード）では GPU を見せない。予行と自己検査は GPU に一切触らない（CUDA の初期化もしない）
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "13"
_HF = os.path.join(_PROJ, ".cache", "huggingface")
os.environ["HF_HOME"] = _HF
os.environ["HF_HUB_CACHE"] = os.path.join(_HF, "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(_PROJ, ".cache", "numba")
if _PROJ == "/mnt":   # コンテナの中だけ作る（ホストの予行では何も書かない）
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
for _k in ("ZH_GEN_DIR", "ZH_GOLD_PATH"):   # 判定器はこれがあると既定の置き場を変える。引数だけで決める
    os.environ.pop(_k, None)
sys.modules["pyopenjtalk"] = None  # D4: 自分のコードでは pyopenjtalk を使わない（判定にも要らない）

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import appk_zhjudge_common as C  # noqa: E402

C.K.cap_threads()   # ホストでは優先度も 10 下げる（コンテナでは下げない）

import argparse  # noqa: E402
import importlib.metadata as _md  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(C.THREADS)


def _ver(pkg):
    try:
        return _md.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def environment():
    cuda = torch.cuda.is_available()   # ホストでは CUDA_VISIBLE_DEVICES="" なので常に False（初期化しない）
    env = {"python": platform.python_version(), "host": platform.node(), "proj": C.PROJ,
           "in_container": C.PROJ == "/mnt", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if cuda else None,
           "cuda_available": cuda,
           "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
           "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
           "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "torch_threads": torch.get_num_threads(),
           "pypinyin": _ver("pypinyin"), "transformers": _ver("transformers"), "numpy": _ver("numpy"),
           "numba": _ver("numba"), "openai_whisper_pip": _ver("openai-whisper"),
           "pyopenjtalk_blocked": sys.modules.get("pyopenjtalk", "absent") is None,
           "env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                  "NUMBA_CACHE_DIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    return env


def judge_namespace():
    """判定器のモジュールの最上位だけを実行して関数を得る（main は呼ばない。モデルは読まない）."""
    return runpy.run_path(C.JUDGE, run_name="appk_zhjudge_probe")


def hypotheses(J, rows, gold, sym2id):
    """判定器 main() 196–226 行と同じ手順で、各対の仮説（トークン id 列）を組む（音声は使わない）."""
    out, skips = [], []
    for r in rows:
        if r.get("status") != "success":
            continue
        ch = r["word_a"]
        src = gold[r["text_a"]]["reading"]
        tgt = gold[r["text_b"]]["reading"]
        ids_src = J["sent_token_ids"](r["text_a"], ch, src, sym2id)
        ids_tgt = J["sent_token_ids"](r["text_a"], ch, tgt, sym2id)
        if ids_src is None or ids_tgt is None:
            skips.append(r["pair_idx"])
            continue
        keys, ids = ["src", "tgt"], [ids_src, ids_tgt]
        src_base, tgt_base = re.sub(r"\d", "", src), re.sub(r"\d", "", tgt)
        for base, tag in [(src_base, "srcnt"), (tgt_base, "tgtnt")]:
            if tag == "tgtnt" and tgt_base == src_base:
                continue
            for t in "12345":
                x = J["sent_token_ids"](r["text_a"], ch, base + t, sym2id)
                if x is not None:
                    ids.append(x)
                    keys.append(f"{tag}{t}")
        out.append({"pair_idx": r["pair_idx"], "char": ch, "src": src, "tgt": tgt, "keys": keys, "ids": ids})
    return out, skips


def text_side_check(ckpt_rows):
    """判定器の文字の側（pypinyin → pinyin トークン）を CPU で確かめ、7/28 の行と形が合うかを見る."""
    J = judge_namespace()
    snap = J["local_snapshot"](C.MODEL_ID)
    if os.path.basename(snap) != C.SNAP_ID:
        return {"problems": [f"判定器が選ぶ snapshot が {os.path.basename(snap)}（{C.SNAP_ID} でない）"]}
    sym2id = J["load_tokens"](snap)
    gold = C.read_json(C.GOLD_V2)
    hyp, skips = hypotheses(J, ckpt_rows, gold, sym2id)
    stored = {e["pair_idx"]: e for e in C.read_json(C.STORED_JUDGE)["rows"]}
    nt_bad, meta_bad = [], []
    for h in hyp:
        e = stored.get(h["pair_idx"])
        has_src_nt = any(k.startswith("srcnt") for k in h["keys"])
        has_tgt_nt = any(k.startswith("tgtnt") for k in h["keys"])
        if e is None or ("bl_lp_src_nt" in e) != has_src_nt or ("bl_lp_tgt_nt" in e) != (has_src_nt and has_tgt_nt):
            nt_bad.append(h["pair_idx"])
        if e is None or (e["char"], e["src"], e["tgt"]) != (h["char"], h["src"], h["tgt"]):
            meta_bad.append(h["pair_idx"])
    res = {"n_rows": len(ckpt_rows), "n_hypothesis_sets": len(hyp), "tokenize_skips": skips,
           "n_hypotheses_total": sum(len(h["ids"]) for h in hyp),
           "hypothesis_md5": C.md5_json([[h["pair_idx"], h["keys"], h["ids"]] for h in hyp]),
           "nt_presence_mismatch_vs_stored": nt_bad, "char_src_tgt_mismatch_vs_stored": meta_bad,
           "pypinyin": _ver("pypinyin"), "problems": []}
    if len(ckpt_rows) != C.N or len(hyp) != C.N:
        res["problems"].append(f"仮説を組めた対が {len(hyp)}/{len(ckpt_rows)}（300 でない）")
    if skips:
        res["problems"].append(f"tokenize できない対: {skips[:10]}（7/28 は 0）")
    if nt_bad:
        res["problems"].append(f"声調なしの欄の有無が 7/28 の行と違う対: {nt_bad[:10]}（pypinyin の違いの疑い）")
    if meta_bad:
        res["problems"].append(f"字・src・tgt が 7/28 の行と違う対: {meta_bad[:10]}")
    return res


def preflight(stage_dir, dry_run, allow_selftest=False):
    """走らせる前の確認。戻り値: dict（problems / waits を含む）."""
    pf = {"stage_dir": stage_dir, "problems": [], "waits": [], "dry_run": dry_run}
    if not dry_run and C.PROJ != "/mnt":
        pf["problems"].append("GPU の実行はコンテナの中（slurm_run.sh 経由）だけ。ホストでは --dry_run のみ")
    if not dry_run and not torch.cuda.is_available():
        pf["problems"].append("CUDA が使えない（判定器は黙って CPU で動くので止める）")
    if sys.modules.get("pyopenjtalk", "absent") is not None:
        pf["problems"].append("pyopenjtalk が塞がれていない")
    for tag, path in (("judge", C.JUDGE), ("gold_v2", C.GOLD_V2), ("stored_judge", C.STORED_JUDGE)):
        ok, m = C.check_pinned(tag, path)
        pf[f"{tag}_md5"] = m
        if not ok:
            pf["problems"].append(f"{C.proj_rel(path)} の md5 {m} が {C.EXPECTED_MD5[tag]} でない")
    pf["problems"] += C.check_model_and_fork(full_ckpt_md5=True)
    mp = os.path.join(stage_dir, C.MANIFEST)
    ckpt_rows = None
    if not os.path.isfile(mp):
        msg = f"stage がまだない: {C.proj_rel(stage_dir)}（appk_zhjudge_stage.py を先に流す）"
        (pf["waits"] if dry_run else pf["problems"]).append(msg)
        if dry_run:   # 予行では本番の checkpoint_zh.json で文字の側だけ確かめる
            ckpt_rows = C.read_json(C.PROD_CKPT)["results"]
            pf["text_side_source"] = C.proj_rel(C.PROD_CKPT)
    else:
        man = C.read_json(mp)
        pf["manifest_md5"] = C.md5_file(mp)
        pf["manifest_mode"] = man.get("mode")
        pf["changed_pairs"] = man.get("changed_pairs")
        if man.get("selftest") and not allow_selftest:
            pf["problems"].append("自己検査の stage は判定に使わない")
        if man.get("allow_silent") and not allow_selftest:
            pf["problems"].append("無音を許して作った stage は判定に使わない")
        if man.get("wavs_are_symlinks") and not allow_selftest:
            pf["problems"].append("wav が symlink の stage（自己検査用）は判定に使わない")
        if len(man.get("files", {})) != 2 * C.N or sorted(man["files"]) != sorted(C.all_wav_names()):
            pf["problems"].append("manifest の wav が 600 本そろっていない")
        bad = C.verify_stage_files(stage_dir, man)
        if bad:
            pf["problems"].append(f"stage のファイルが manifest と違う（{len(bad)} 件、例 {bad[:2]}）")
        prior = C.prior_judge_outputs(stage_dir)
        if prior:
            pf["problems"].append(f"stage に判定器の出力が既にある {prior}（続きから始めてしまうので走らせない。"
                                  f"新しい --stage_dir で stage を作り直す）")
        ckpt_rows = C.read_json(os.path.join(stage_dir, C.STAGE_CKPT))["results"]
        pf["text_side_source"] = C.proj_rel(os.path.join(stage_dir, C.STAGE_CKPT))
    if ckpt_rows is not None:
        ts = text_side_check(ckpt_rows)
        pf["text_side"] = {k: v for k, v in ts.items() if k != "problems"}
        pf["problems"] += ts["problems"]
    return pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_dir", default=C.proj_rel(C.DEFAULT_STAGE))
    ap.add_argument("--dry_run", action="store_true", help="確認だけ（モデルを読まない。何も書かない）")
    args = ap.parse_args()
    stage_dir = C.resolve(args.stage_dir)
    t0 = time.time()
    env = environment()
    pf = preflight(stage_dir, args.dry_run)
    print(f"[zhjudge-gpu] stage={C.proj_rel(stage_dir)} dry_run={args.dry_run} proj={C.PROJ} "
          f"cuda={env['cuda_available']} gpu={env['gpu_name']} torch={env['torch']} pypinyin={env['pypinyin']}",
          flush=True)
    ts = pf.get("text_side") or {}
    print(f"  text side ({pf.get('text_side_source')}): hyp sets {ts.get('n_hypothesis_sets')}, "
          f"tokenize skips {ts.get('tokenize_skips')}, hyp md5 {ts.get('hypothesis_md5')}, "
          f"nt-presence mismatch {ts.get('nt_presence_mismatch_vs_stored')}", flush=True)
    for w in pf["waits"]:
        print(f"  WAIT: {w}", flush=True)
    for p in pf["problems"]:
        print(f"  PROBLEM: {p}", flush=True)
    if args.dry_run:
        print(f"  dry run: {'OK' if not pf['problems'] else 'PROBLEMS'} (nothing written)", flush=True)
        sys.exit(1 if pf["problems"] else 0)
    if pf["problems"]:
        sys.exit(1)

    started = time.strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(stage_dir, f"{C.RUN_META_PREFIX}{started}.json")
    judge_argv = [C.JUDGE_REL, "--gen_dir", C.proj_rel(stage_dir), "--gold", C.GOLD_V2_REL]
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running", "stage_dir": C.proj_rel(stage_dir),
            "judge": C.JUDGE_REL, "judge_md5": pf["judge_md5"], "judge_argv": judge_argv,
            "gold": C.GOLD_V2_REL, "gold_md5": pf["gold_v2_md5"], "model": C.MODEL_ID, "snapshot": C.SNAP_ID,
            "stage_manifest_md5": pf["manifest_md5"], "changed_pairs": pf.get("changed_pairs"),
            "environment": env, "preflight": pf}
    C.write_json(meta, meta_path, backup=False)   # この実行だけの記録（時刻つきの新しい名前）
    print(f"[zhjudge-gpu] run: {' '.join(judge_argv)}", flush=True)
    old_argv = sys.argv
    try:
        sys.argv = [C.JUDGE] + judge_argv[1:]
        runpy.run_path(C.JUDGE, run_name="__main__")
    except BaseException as e:  # noqa: BLE001  SystemExit も含めて記録する
        meta.update(status="failed", error=repr(e)[:500], traceback=traceback.format_exc()[-4000:],
                    elapsed_min=round((time.time() - t0) / 60, 2))
        C.write_json(meta, meta_path, backup=False)
        print(f"[zhjudge-gpu] FAILED: {e!r} (meta {meta_path})", flush=True)
        sys.exit(1)
    finally:
        sys.argv = old_argv
    out = os.path.join(stage_dir, C.JUDGE_OUT)
    ok = os.path.isfile(out)
    after = C.verify_stage_files(stage_dir, C.read_json(os.path.join(stage_dir, C.MANIFEST)))
    doc = C.read_json(out) if ok else {}
    meta.update(status="done" if ok and not after else "failed",
                finished=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_min=round((time.time() - t0) / 60, 2),
                judge_output=C.proj_rel(out), judge_output_md5=C.md5_file(out) if ok else None,
                partial_md5=C.md5_file(os.path.join(stage_dir, C.JUDGE_PARTIAL))
                if os.path.isfile(os.path.join(stage_dir, C.JUDGE_PARTIAL)) else None,
                n_rows=len(doc.get("rows", [])), judge_summary=doc.get("summary"),
                stage_files_changed_during_run=after)
    C.write_json(meta, meta_path, backup=False)
    print(f"[zhjudge-gpu] {meta['status']} in {meta['elapsed_min']} min, rows {meta['n_rows']}, "
          f"output md5 {meta['judge_output_md5']} -> meta {C.proj_rel(meta_path)}", flush=True)
    print("  next (CPU on the host): taskset -c 0-12 python3 "
          f"$REPO_ROOT/scripts/regen2/appk_zhjudge_aggregate.py --stage_dir {C.proj_rel(stage_dir)}",
          flush=True)
    sys.exit(0 if meta["status"] == "done" else 1)


if __name__ == "__main__":
    main()
