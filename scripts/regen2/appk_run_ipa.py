#!/usr/bin/env python3
"""付録 K の IPA 変化を、N=300 の同じ音声集合で 6 言語まとめて採点し直す（2026-09-29）.

GPU の処理は既存の採点器 scripts/projection/score_phoneme_asr_from_dir.py（wav2vec2-xlsr-53-espeak-cv-ft、
変更しない）を言語ごとに子プロセスで呼ぶだけ。このスクリプト自体は torch を import しない。
  en/fr/de/ko : eval_results/utmos_speaker_fixed_{lang}_n300_splice をそのまま渡す（s*_bl / s*_inj）
  ja/zh       : appk_prepare_stage.py で作った <out_dir>/stage_{ja,zh}/（s*_oeoi を s*_inj として並べた symlink）
出力は <out_dir>/ipa_{lang}_n300.json（新しい名前。既にあれば上書きせずにその言語を飛ばす）と
<out_dir>/ipa_{lang}_n300.log（前のジョブのログがあれば .bak_<時刻> に写してから書く）、
実行の記録 <out_dir>/ipa_run_meta_<開始時刻>.json（言語が 1 つ終わるごとに書き直す。
時間切れで止まっても、終わった言語の記録は残る）。

--silent_scan（appk_silent_scan.py か appk_post_apply_scan.py の出力）は必須。使えない対（無音・欠け・読めない）が
残る言語は採点しない（著者の決定: 対ごと seed 43/44 で作り直してから採点する）。検査のあとで音声が変わった言語も
採点しない（md5 で照合）。--allow_silent で強制できるが、その場合は出力を別の --out_dir に分けること。
採点器の子プロセスは HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1（保存済みの snapshot だけを読む）。
終了コード: 採点しなかった言語があれば 1（--dry_run では 0）。

GPU（$REPO_ROOT から投入する。slurm_run.sh は投入したフォルダをコンテナの /mnt にするので、
ほかのフォルダから出すとスクリプトが見つからない。GPU 1 枚。6 言語で SLURM_TIME=01:30:00、JA だけなら 00:20:00）:
  SLURM_TIME=01:30:00 slurm_run.sh python scripts/regen2/appk_run_ipa.py --langs en,fr,de,ko,ja,zh --out_dir eval_results/_appk_redo --silent_scan eval_results/_appk_redo/silent_scan_n300.json
入力の確認だけ（ホストの CPU、モデルは読まない、何も書かない。無音の検査と JA/ZH の symlink のあとに流す。
スレッド 13 と優先度 10 はスクリプトが自分で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/appk_run_ipa.py --dry_run --langs en,fr,de,ko,ja,zh --out_dir eval_results/_appk_redo --silent_scan eval_results/_appk_redo/silent_scan_n300.json
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
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

from appk_silent_scan import SETS, N_EXPECTED, PROJ, ER  # noqa: E402
from appk_prepare_stage import STAGED_LANGS, discover_like_scorer, stage_dir_of  # noqa: E402

SCORER = os.path.join(PROJ, "scripts", "projection", "score_phoneme_asr_from_dir.py")
MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
HF_REF = os.path.join(PROJ, ".cache", "huggingface", "hub",
                      "models--facebook--wav2vec2-xlsr-53-espeak-cv-ft", "refs", "main")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def abspath_proj(p):
    if p.startswith(os.environ.get("REPO_ROOT", "/mnt")) and PROJ == "/mnt":
        p = p.replace(os.environ.get("REPO_ROOT", "/mnt"), "/mnt", 1)
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(PROJ, p))


def wav_dir_for(lang, out_dir):
    if lang in STAGED_LANGS:
        return stage_dir_of(out_dir, lang)
    return K.set_dir(lang)


def read_ref():
    return open(HF_REF).read().strip() if os.path.isfile(HF_REF) else None


def scorer_env():
    """採点器の子プロセス: スレッド上限と、保存済みの snapshot だけを読む設定."""
    hf = os.path.join(PROJ, ".cache", "huggingface")
    return dict(os.environ, HF_HOME=hf, HF_HUB_CACHE=os.path.join(hf, "hub"),
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                **{v: str(K.THREADS) for v in K.THREAD_VARS})


def check_inputs(lang, wav_dir):
    if not os.path.isdir(wav_dir):
        return f"wav_dir がない: {wav_dir}（ja/zh は先に appk_prepare_stage.py --stage_root <out_dir> を実行）"
    files, found = discover_like_scorer(wav_dir)
    if len(found) != N_EXPECTED or found != [f"s{i:04d}" for i in range(N_EXPECTED)]:
        return f"採点器の探し方で見つかる対が {len(found)}（BL {len(files)}）で、s0000..s0299 の 300 対にならない"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,fr,de,ko,ja,zh")
    ap.add_argument("--out_dir", default="eval_results/_appk_redo")
    ap.add_argument("--silent_scan", default=None,
                    help="appk_silent_scan.py / appk_post_apply_scan.py の出力（必須）。使えない対が残る言語を飛ばす。")
    ap.add_argument("--allow_silent", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="入力を確かめてコマンドを表示するだけ（採点しない）")
    args = ap.parse_args()
    if not args.silent_scan and not args.allow_silent:
        sys.exit("--silent_scan が要る（無音の検査なしでは採点しない。強制は --allow_silent と別の --out_dir）")

    out_dir = abspath_proj(args.out_dir)
    assert out_dir.startswith(os.path.join(PROJ, "eval_results") + os.sep), out_dir
    langs = args.langs.split(",")
    for lang in langs:
        assert lang in SETS, lang
    assert not (args.allow_silent and os.path.basename(out_dir) == "_appk_redo"), \
        "--allow_silent の出力は本番の _appk_redo と分ける（例: eval_results/_appk_redo_prelim）"

    silent = {}
    scan_md5 = {}
    if args.silent_scan:
        sc = json.load(open(abspath_proj(args.silent_scan)))
        for lang in langs:
            if lang not in sc["langs"]:
                sys.exit(f"無音の検査に {lang} がない")
            ls = sc["langs"][lang]
            silent[lang] = sorted({p["pair_idx"] for p in ls["silent_pairs"]}
                                  | set(ls.get("pairs_missing_or_unreadable", [])))
            scan_md5[lang] = {k: v["md5"] for k, v in sc["per_wav"][lang].items()}

    started = time.strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(out_dir, f"ipa_run_meta_{started}.json")
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "scorer": SCORER,
            "scorer_md5": md5_file(SCORER), "model": MODEL,
            "hf_refs_main": read_ref(), "scorer_env_offline": True,
            "silent_scan": args.silent_scan, "allow_silent": args.allow_silent,
            "dry_run": args.dry_run, "langs": {}}

    def save_meta():
        if not args.dry_run:
            with open(meta_path, "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[run] PROJ={PROJ} out_dir={out_dir} scorer_md5={meta['scorer_md5']} "
          f"model_ref={meta['hf_refs_main']}", flush=True)

    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
    for lang in langs:
        wav_dir = wav_dir_for(lang, out_dir)
        out_json = os.path.join(out_dir, f"ipa_{lang}_n300.json")
        log_path = os.path.join(out_dir, f"ipa_{lang}_n300.log")
        rec = {"wav_dir": wav_dir, "output_json": out_json, "log": log_path}
        meta["langs"][lang] = rec
        err = check_inputs(lang, wav_dir)
        if err:
            rec["status"] = f"skip: {err}"
            print(f"[{lang}] SKIP {err}", flush=True)
            continue
        if lang in silent and silent[lang] and not args.allow_silent:
            rec["status"] = f"skip: unusable pairs remain {silent[lang]}"
            print(f"[{lang}] SKIP 使えない対（無音・欠け・読めない）が残っている（{len(silent[lang])} 対: {silent[lang]}）。"
                  f"作り直して無音を調べ直してから採点する。", flush=True)
            continue
        # 採点の直前に、音声が無音の検査の時点から変わっていないかを確かめる
        if lang in scan_md5:
            src_dir = K.set_dir(lang)
            changed = [n for n, m in scan_md5[lang].items()
                       if not os.path.isfile(os.path.join(src_dir, n)) or md5_file(os.path.join(src_dir, n)) != m]
            if changed:
                rec["status"] = f"skip: {len(changed)} wavs changed since silent scan"
                print(f"[{lang}] SKIP 無音の検査のあとに音声が {len(changed)} 本変わっている（例 {changed[:3]}）。"
                      f"検査をやり直すこと。", flush=True)
                continue
            rec["md5_checked_against_scan"] = len(scan_md5[lang])
        if os.path.exists(out_json):
            rec["status"] = "skip: output exists (not overwritten)"
            print(f"[{lang}] SKIP 出力が既にある（上書きしない）: {out_json}", flush=True)
            continue
        cmd = [sys.executable, SCORER, "--wav_dir", wav_dir, "--lang", lang,
               "--output_json", out_json, "--phoneme_model", MODEL]
        rec["cmd"] = cmd
        if args.dry_run:
            rec["status"] = "dry_run"
            print(f"[{lang}] OK（dry run）: {' '.join(cmd)}", flush=True)
            continue
        t0 = time.time()
        print(f"[{lang}] start: {' '.join(cmd)}", flush=True)
        if os.path.exists(log_path):   # 前のジョブ（失敗した回など）のログを写してから書く（移動・削除はしない）
            bak = f"{log_path}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(log_path, bak)
            rec["log_backup"] = bak
            print(f"[{lang}] [backup] {log_path} -> {bak}", flush=True)
        with open(log_path, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=scorer_env())
        rec.update({"returncode": rc, "elapsed_min": round((time.time() - t0) / 60, 2),
                    "hf_refs_main_after": read_ref()})
        if rec["hf_refs_main_after"] != meta["hf_refs_main"]:
            print(f"[{lang}] WARNING: 採点中にモデルの refs/main が変わった "
                  f"{meta['hf_refs_main']} -> {rec['hf_refs_main_after']}", flush=True)
        if rc == 0 and os.path.isfile(out_json):
            s = json.load(open(out_json))["summary"]
            rec["status"] = "done"
            rec["scorer_summary"] = s
            print(f"[{lang}] done in {rec['elapsed_min']} min: {s}", flush=True)
        else:
            rec["status"] = f"failed rc={rc}"
            print(f"[{lang}] FAILED rc={rc}（{log_path} を見る）", flush=True)
        save_meta()

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_meta()
    if not args.dry_run:
        print(f"[run] meta -> {meta_path}")
    # 既に出力がある言語（前のジョブで採点済み）は失敗に数えない
    bad = [l for l, r in meta["langs"].items()
           if r.get("status") not in ("done", "dry_run", "skip: output exists (not overwritten)")]
    print(f"[run] not scored in this run: {bad}" if bad else "[run] all requested languages OK")
    sys.exit(1 if (bad and not args.dry_run) else 0)


if __name__ == "__main__":
    main()
