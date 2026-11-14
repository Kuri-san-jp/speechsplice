#!/usr/bin/env python3
"""G5a: Table 3 (c) T5Gemma-TTS EN（multientity_en_300）で使えない対を seed 43（だめなら 44）で対ごとに作り直す（GPU）.

対象（scripts/regen2/silence_scan_g5a.py --stage pre で確認済み。bl か inj1/inj2/inj12 が無音の対）:
  2, 5, 30, 47, 141, 153, 241, 257, 276
tgt1/tgt2 だけが無音の対（36, 41, 91, 198, 209）は対象外（tgt は Table 3 の印字値に使われない）。
1 対あたり bl / inj1 / inj2 / inj12 と tgt1 / tgt2 の 6 本をすべて同じ seed で作る。

使えるかの基準（regen2_common.multi_pair_validity。対象の選び方と同じ）:
  * bl / inj1 / inj2 / inj12 が無音・欠け・読めないに当たらない
    （無音 = (RMS<0.005 かつ peak<0.05) または 20 ms 区間の active_frac<0.05）
  * tgt1 / tgt2 の wav があって読める（無音は記録だけ）
  * results.json に 4 条件の行、results_tgt.json に tgt1/tgt2 の行がある
    （generate_multientity は合成・採点・スパン合わせの失敗を skipped に入れ、行を作らない）

手順は本番と同じ scripts/analysis/generate_multientity.py の main() をそのまま呼ぶ
（本番: 4 条件は 2026-09-17、tgt1/tgt2 は 2026-09-20 の --tgt_only）。変えるのは次の 3 点だけ:
  1. SEED を置き換える。generate_multientity は `from eval_utmos_speaker_fixed import SEED` で値を
     取り込み、main 冒頭の seed_all(SEED) と対ごと・条件ごとの seed_everything(SEED) で使う。
     voice_clone_synth（eval_utmos_speaker_fixed）は自分のモジュールの SEED で seed_everything を呼ぶ。
     そこで両方のモジュールの SEED を置き換える（precedent: scripts/spkvar/regen_pairs_seed.py の E.SEED）。
  2. load_items を対象の対だけに絞る（各対・各条件の直前で seed を戻すので、他の対の有無は結果に影響しない）。
  3. --reference_text に本番で Whisper が返した参照文をそのまま渡す（本番は書き起こしで同じ文字列を得た。
     logs/autoqueue/multient_en.log 13 行）。書き起こしのぶれを入れないため。
duration 推定（vendor の duration_estimator）は本番と同じ経路のまま（追加も削除もしない。
著者は手法の構成要素として、コンテナ内の vendor の経路が pyopenjtalk を使うことを許している）。
このスクリプト自身は pyopenjtalk を import しない。
HF: generate_multientity / eval_utmos_speaker_fixed が import 時に HF_HOME=/mnt/.cache/huggingface と
HF_HUB_OFFLINE=1 を固定し、T5Gemma と tokenizer はローカルの snapshot のパスで読む（repo id では読まない）。

出し直し（時間切れ・ノード障害）で壊れないための作り（2026-09-29 追加）:
  generate_multientity の --tgt_only は checkpoint_tgt.json を読まず、tgt の wav がある対を済みとみなして
  results_tgt.json / checkpoint_tgt.json を「その回に作った行だけ」で上書きする。そこで main() は毎回
  新しい試行の置き場 <out_dir>/_attempts/aNN_{main,tgt}/ に書かせ、終わった対だけを本体（<out_dir>）へ写す:
  wav を写してから、本体の checkpoint*.json と results*.json の行を「前の行 + 今回の行」で書き直す（tmp → 置き換え）。
  本体で済み（行と wav がそろう）か失敗（skipped に記録）の対は、もう一度作らない。途中で止まった対は
  本体に残らず、次の実行で同じ seed のまま作り直される。失敗の振り分けは regen2_common.triage_attempt:
  GPU メモリ不足・通信などの環境の失敗は本体に書かずにジョブを止める（seed 44 に回さない）。CUDA の実行時の
  失敗（device-side assert など）は、その対と同じ試行でそれより後に処理した対を本体に書かずに止め、同じ対が
  同じ置き場・同じ段階で 2 回目なら生成の失敗として skipped に書く（→ seed 44。記録は
  <out_dir>/regen2_cuda_fault_strikes_{main,tgt}.json）。その他の例外は生成の失敗（→ seed 44）。

確かめること:
  * 先に seed 42 のまま、対象外の対 3 を作り直して本番と md5 を比べる（記録のみ。bf16 の非決定性があり得る。
    wav が無くても止めずに不一致として記録する）。
  * seed に依らない値（span_tok、T_src、条件ごとの target_text / delta_tok / T_out / out_ranges /
    dur_sec）が本番の行と一致するか。
  * seed 43 の対を使えるかの基準で調べ、使えない対は対ごと（6 本とも）seed 44 で作り直す。
    44 でも使えなければ unresolved として記録し、apply はその対を差し替えない。

出力（本番は書き換えない）:
  eval_results/_regen2_seed43/multientity_en_300/（results.json, results_tgt.json, checkpoint*.json, wav, _attempts/）
  eval_results/_regen2_seed44/multientity_en_300/（必要なときだけ）
  eval_results/_regen2_seed43/multientity_en_300/_check42/
  eval_results/_regen2_final/multientity_en_300/（採用した seed の wav と、対象の行だけの results*.json）

GPU（プロジェクト直下から）:
  SLURM_TIME=01:00:00 slurm_run.sh bash scripts/regen2/run_g5a_multi_gpu.sh
CPU の予行（モデルも generate_multientity も読まない。データ・本番の行・静的な SEED 参照を確かめる。ホストで直接。
スレッド 13・CPU 0-12・nice 10・GPU を見せない・HF オフラインは、--cpu_dryrun のときスクリプトの中で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/regen_multientity_t5gemma_seed.py --cpu_dryrun

--campaign thankyou（著者判断 2026-09-30。定義は regen2_common.THANKYOU）:
  「Thank you.」としか書き起こされない 6 本の対（21, 43, 111, 251, 257, 272）を対ごと（6 本とも）作り直す。
  対ごとの seed の順は THANKYOU["seed_plan"]（21/43/111/251/272: 43 → 44、257: 44 だけ）。
  seed ごとに: run_generate（上と同じ。試行の置き場 → 本体）→ 本番の採点 scripts/analysis/score_multientity.py を
  1 行も変えずにその seed の置き場に回す（<seed_dir>/asr/asr_rNN.json。出し直しでは、置き場の対をすべて失敗なく
  覆う既存のファイルを使い、書き起こし直さない）→ THANKYOU["rule"]（無音 または 書き起こしが "thankyou"）で判定 →
  使えない対だけ次の seed へ。採った seed の wav・行と、判定に使った書き起こしの行（書き起こし直さない）を
  eval_results/_regen2_thankyou/final/multientity_en_300/ にまとめ、regen2_thankyou_meta.json を書く。
  書き起こしが欠けて決められない対があれば seed を選ばずに止める（出し直すと新しい asr_rNN で書き起こす）。
  257 が seed 44 でも使えなければ unresolved として記録して止める（apply はどの対も差し替えない）。
  seed 42 の再現確認は回さない（9/29 の G5a で対 3 の 6 本が md5 まで一致。その記録を meta に写す）。
  出力はすべて eval_results/_regen2_thankyou/ の下（本番・G5a の置き場には書かない）。
  GPU（プロジェクト直下から）:
    SLURM_TIME=01:00:00 slurm_run.sh bash scripts/regen2/run_g5a_multi_gpu.sh thankyou
  CPU の予行:
    taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/regen_multientity_t5gemma_seed.py --campaign thankyou --cpu_dryrun
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import regen2_common as C  # noqa: E402

C.cap_threads()
C.job_env()          # HF_HOME=<project>/.cache/huggingface・HF_HUB_OFFLINE=1・NUMBA_CACHE_DIR・スレッド 13（torch を読む前）
if "--cpu_dryrun" in sys.argv:
    C.host_cpu_setup()   # ホストの CPU の予行だけ（GPU で動かすときは呼ばない）。torch を読む前
os.environ["GEN_THREADS"] = "13"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

M = C.MULTI
ANALYSIS = os.path.join(C.PROJECT_ROOT, "scripts", "analysis")
PROJECTION = os.path.join(C.PROJECT_ROOT, "scripts", "projection")
ALL_CONDS = M["conds"] + M["tgt_conds"]
SEED_INDEP_COND_KEYS = ("target_text", "delta_tok", "T_out", "out_ranges", "dur_text_is_text_a", "dur_sec")
# (phase, 本体の results, 帳面, 作る条件, --tgt_only)
PHASES = (("main", "results.json", "checkpoint.json", M["conds"], False),
          ("tgt", "results_tgt.json", "checkpoint_tgt.json", M["tgt_conds"], True))
# 生成モジュール（検査では偽物を入れる）
_MODS = {"G": None, "E": None}


def prod_dir():
    return os.path.join(C.EV, M["name"])


def load_rows(d, name):
    return C.read_rows(os.path.join(d, name))


def _modules():
    if _MODS["G"] is None:
        sys.path.insert(0, ANALYSIS)
        import generate_multientity as G   # eval_utmos_speaker_fixed もここで読まれる（HF_HOME 等は本番と同じ）
        _MODS["G"], _MODS["E"] = G, sys.modules["eval_utmos_speaker_fixed"]
    return _MODS["G"], _MODS["E"]


def _call_main(seed, pairs, out_dir, tgt_only):
    """本番の generate_multientity.main() を、SEED だけ置き換えて pairs だけで 1 回呼ぶ."""
    G, E = _modules()
    C.assert_regen_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tset = set(pairs)
    orig_load = G._orig_load_items if hasattr(G, "_orig_load_items") else G.load_items
    G._orig_load_items = orig_load

    def _load(pairs_module, n):
        items = orig_load(pairs_module, n)
        assert len(items) == 300 and all(it["pair_idx"] == i for i, it in enumerate(items))
        return [it for it in items if it["pair_idx"] in tset]

    base = [G.__file__, "--pairs_module", M["pairs_module"], "--lang", M["lang"],
            "--reference_wav", M["reference_wav"], "--reference_text", M["reference_text"],
            "--output_dir", out_dir] + (["--tgt_only"] if tgt_only else [])
    old = (G.SEED, E.SEED)
    saved = sys.argv
    G.load_items = _load
    G.SEED = seed
    E.SEED = seed
    print("[regen] SEED generate_multientity %s -> %d, eval_utmos_speaker_fixed %s -> %d; pairs=%s out=%s tgt_only=%s"
          % (old[0], seed, old[1], seed, sorted(tset), out_dir, tgt_only), flush=True)
    try:
        sys.argv = list(base)
        G.main()
    finally:
        sys.argv = saved
        G.load_items = orig_load
        G.SEED, E.SEED = old
        # main() はモデルを毎回読み直すので、前の回の分を解放しておく
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return base


def _new_attempt_dir(out_dir, phase):
    base = C.assert_regen_path(os.path.join(out_dir, "_attempts"))
    os.makedirs(base, exist_ok=True)
    k = 1
    while any(os.path.exists(os.path.join(base, "a%02d_%s" % (k, ph))) for ph in ("main", "tgt")):
        k += 1
    return os.path.join(base, "a%02d_%s" % (k, phase))


def run_generate(seed, targets, out_dir):
    """4 条件（main）→ tgt1/tgt2（tgt）の順に、本体で済みでも失敗でもない対だけを試行の置き場で作り、
    終わった対を本体へ写す。何度呼んでも、本体の行を消したり、済みの対を作り直したりしない."""
    C.assert_regen_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tset = sorted(set(targets))
    info = {"seed": seed, "pairs": tset, "out_dir": out_dir, "phases": []}
    for phase, name, ckname, conds, tgt_only in PHASES:
        res_p, ck_p = os.path.join(out_dir, name), os.path.join(out_dir, ckname)
        doc = C.read_json(res_p) if os.path.exists(res_p) else {"rows": [], "skipped": []}
        rows = {r["pair_idx"]: r for r in doc.get("rows", [])}
        skipped = list(doc.get("skipped", []))
        failed = {s["pair_idx"] for s in skipped}
        main_rows = load_rows(out_dir, "results.json")
        need, kept = [], {}
        for p in tset:
            if p in rows and all(os.path.exists(os.path.join(out_dir, "s%04d_%s.wav" % (p, k))) for k in conds):
                kept[p] = "complete"
            elif p in failed and p not in rows:
                kept[p] = "failed at this seed (skipped; final)"
            elif phase == "tgt" and p not in main_rows:
                kept[p] = "main phase not complete (pair already unusable at this seed)"
            else:
                need.append(p)
        ph = {"phase": phase, "kept": {str(k): v for k, v in kept.items()}, "need": need}
        info["phases"].append(ph)
        if not need:
            continue
        att = _new_attempt_dir(out_dir, phase)
        argv = _call_main(seed, need, att, tgt_only)
        att_res = os.path.join(att, name)
        if not os.path.exists(att_res):
            raise SystemExit("試行 %s が %s を書かずに終わった（main() の途中で止まった）" % (att, name))
        adoc = C.read_json(att_res)
        arows = {r["pair_idx"]: r for r in adoc.get("rows", [])}
        askipped = adoc.get("skipped", [])
        # 失敗の振り分け（regen2_common.triage_attempt）: 環境の失敗と、CUDA の失敗の後に処理した対は本体に
        # 書かない。CUDA の失敗は同じ置き場・同じ段階で 2 回目なら生成の失敗（→ seed 44）。
        # generate_multientity は load_items の順（pair_idx の昇順 = need の順）に処理する
        errs = {}
        for s in askipped:
            if s.get("pair_idx") in need and s["pair_idx"] not in arows:
                errs[s["pair_idx"]] = (errs.get(s["pair_idx"], "") + " | " + str(s.get("reason"))).strip(" |")
        strike_p = os.path.join(out_dir, "regen2_cuda_fault_strikes_%s.json" % phase)
        tri = C.triage_attempt(need, errs, C.read_strikes(strike_p))
        C.record_strikes(strike_p, tri["new_strikes"], os.path.basename(att))
        infra = tri["hold"]
        new_rows, new_skipped = {}, []
        for p in need:
            if p in infra:
                continue                                  # 本体に残さない（次の実行で同じ seed のまま作る）
            if p in arows:
                for k in conds:
                    src = os.path.join(att, "s%04d_%s.wav" % (p, k))
                    if not os.path.exists(src):
                        raise SystemExit("試行の行はあるのに wav が無い: %s" % src)
                    shutil.copyfile(src, os.path.join(out_dir, "s%04d_%s.wav" % (p, k)))
                new_rows[p] = arows[p]
            else:
                sk = [s for s in askipped if s.get("pair_idx") == p]
                if not sk:
                    raise SystemExit("試行 %s に対 %d の行も skipped も無い（想定外）" % (att, p))
                new_skipped += [dict(s, regen2_attempt=os.path.basename(att)) for s in sk]
        rows.update(new_rows)
        skipped = [s for s in skipped if s["pair_idx"] not in new_rows] + new_skipped
        merged_rows = [rows[k] for k in sorted(rows)]
        C.write_json_atomic_regen({"rows": merged_rows, "skipped": skipped}, ck_p)
        summ = dict(adoc.get("summary") or {})
        summ["regen2_merge"] = {"note": "rows merged over attempts (generate_multientity writes only the rows "
                                        "of one call); summary numbers are those of the last attempt",
                                "attempts": (doc.get("summary") or {}).get("regen2_merge", {}).get("attempts", [])
                                + [os.path.basename(att)],
                                "n_rows": len(merged_rows), "n_skipped": len(skipped)}
        C.write_json_atomic_regen({"summary": summ, "rows": merged_rows, "skipped": skipped}, res_p)
        ph.update({"attempt": att, "argv": argv, "merged_rows": sorted(new_rows),
                   "merged_skipped": sorted({s["pair_idx"] for s in new_skipped}),
                   "held": {str(k): v for k, v in infra.items()},
                   "generation_failures": {str(k): str(v)[:300] for k, v in tri["write_fail"].items()},
                   "cuda_strikes_added": {str(k): v for k, v in tri["new_strikes"].items()}})
        if tri["stop"]:
            C.write_json_atomic_regen(info, os.path.join(att, "run_info_stopped.json"))
            tag = "[cuda]" if tri["cut_after"] is not None else "[infra]"
            raise SystemExit("%s seed %d %s: 止める（対 %s）。本体に書かなかった対 %s は、同じジョブを出し直すと"
                             "同じ seed のまま作る。生成の失敗として書いた対 %s は seed 44 に回る。"
                             % (tag, seed, phase, tri["stop"], sorted(infra), sorted(tri["write_fail"])))
    return info


def pair_validity(d, pidx):
    return C.multi_pair_validity(d, pidx)


def compare_rows(d, pidx):
    """seed に依らない値を本番の行と比べる."""
    pr = load_rows(prod_dir(), "results.json").get(pidx, {})
    pt = load_rows(prod_dir(), "results_tgt.json").get(pidx, {})
    nr = load_rows(d, "results.json").get(pidx, {})
    nt = load_rows(d, "results_tgt.json").get(pidx, {})
    out = {"span_tok": nr.get("span_tok") == pr.get("span_tok"), "T_src": nr.get("T_src") == pr.get("T_src"),
           "total_position_drift": nr.get("total_position_drift") == pr.get("total_position_drift")}
    for k in M["conds"]:
        a, b = (nr.get("conds") or {}).get(k, {}), (pr.get("conds") or {}).get(k, {})
        out[k] = {kk: a.get(kk) == b.get(kk) for kk in SEED_INDEP_COND_KEYS}
    for k in M["tgt_conds"]:
        a, b = (nt.get("conds") or {}).get(k, {}), (pt.get("conds") or {}).get(k, {})
        out[k] = ({kk: a.get(kk) == b.get(kk) for kk in SEED_INDEP_COND_KEYS} if pt
                  else "production results_tgt.json has no row for this pair")
    return out


def run_check(root_check):
    """seed 42 のまま対 3 を作り直し、本番と md5 を比べる（記録のみ。wav が無くても止めない）."""
    cp = M["check_pair"]
    run_generate(42, [cp], root_check)
    res = {"pair": cp, "files": {}}
    for k in ALL_CONDS:
        n = "s%04d_%s.wav" % (cp, k)
        try:
            a, b = C.md5_file(os.path.join(root_check, n)), C.md5_file(os.path.join(prod_dir(), n))
            res["files"][k] = {"md5_new": a, "md5_prod": b, "match": a == b}
        except FileNotFoundError as e:
            res["files"][k] = {"error": "missing", "detail": str(e)[:300], "match": False}
    res["match_all"] = all(v["match"] for v in res["files"].values())
    ok, st = pair_validity(root_check, cp)
    res["valid"], res["invalid_reasons"] = ok, st["reasons"]
    res["rows"] = compare_rows(root_check, cp)
    print("[check42] pair %d md5 match=%s %s valid=%s" % (
        cp, res["match_all"], {k: v["match"] for k, v in res["files"].items()}, ok), flush=True)
    return res


def assemble_final(chosen, seed_dirs, fd=None, note="rows of the chosen seed only (G5a regen2)"):
    fd = C.assert_regen_path(fd or C.multi_final_dir())
    # 先に全部の元があるかを確かめる（途中まで写して止まらないように）
    for pidx, seed in sorted(chosen.items()):
        if seed is None:
            continue
        for k in ALL_CONDS:
            src = os.path.join(seed_dirs[seed], "s%04d_%s.wav" % (pidx, k))
            if not os.path.exists(src):
                raise SystemExit("採用した対 %d（seed %d）の %s が無い" % (pidx, seed, src))
        for name in ("results.json", "results_tgt.json", "checkpoint.json", "checkpoint_tgt.json"):
            if pidx not in load_rows(seed_dirs[seed], name):
                raise SystemExit("採用した対 %d（seed %d）の行が %s に無い" % (pidx, seed, name))
    os.makedirs(fd, exist_ok=True)
    docs = {}
    for name in ("results.json", "results_tgt.json", "checkpoint.json", "checkpoint_tgt.json"):
        rows = []
        summary = None
        for pidx, seed in sorted(chosen.items()):
            if seed is None:
                continue
            doc = C.read_json(os.path.join(seed_dirs[seed], name))
            summary = summary or doc.get("summary")
            rows += [r for r in doc["rows"] if r["pair_idx"] == pidx]
        docs[name] = {"rows": rows, "skipped": []}
        if name.startswith("results"):
            docs[name]["summary"] = dict(summary or {}, note=note)
    for pidx, seed in sorted(chosen.items()):
        if seed is None:
            continue
        for k in ALL_CONDS:
            n = "s%04d_%s.wav" % (pidx, k)
            src, dst = os.path.join(seed_dirs[seed], n), os.path.join(fd, n)
            if os.path.exists(dst):
                if C.md5_file(dst) != C.md5_file(src):
                    raise SystemExit("final に中身の違う %s が既にある（消さずに止める）" % dst)
                continue
            shutil.copy2(src, dst)
    for name, doc in docs.items():
        C.write_json(doc, os.path.join(fd, name))
    return fd


def gpu_main(args):
    import torch
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    meta = {"env": env}
    if not args.no_check:
        meta["check_seed42"] = run_check(os.path.join(C.multi_regen_dir(args.seed), "_check42"))
    env["hf_env"] = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                                                    "TRANSFORMERS_OFFLINE", "TORCH_HOME")}
    finalize(args, meta, run_generate)


def finalize(args, meta, gen_fn):
    """seed 43 で作る → 使えるかの基準で判定 → 使えない対だけ seed 44 で対ごと作る → 対ごとに seed を選び
    _regen2_final にまとめ、regen_meta.json を書く。gen_fn(seed, pairs, out_dir) が生成（検査では偽物）."""
    d43 = C.multi_regen_dir(args.seed)
    meta.update({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "planned_pairs": M["pairs"],
                 "seed": args.seed, "fallback_seed": args.fallback_seed, "reference_text": M["reference_text"],
                 "criterion": {"silence": C.SILENCE_CRITERION,
                               "pair_valid_iff": "bl/inj1/inj2/inj12 present, readable and not silent; tgt1/tgt2 "
                                                 "present and readable (silence recorded only); a results.json "
                                                 "row with the 4 conds and a results_tgt.json row with tgt1/tgt2"}})
    t0 = time.time()
    runs = {args.seed: gen_fn(args.seed, M["pairs"], d43)}
    seed_dirs = {args.seed: d43}
    valid, stats = {}, {}
    for p in M["pairs"]:
        ok, st = pair_validity(d43, p)
        valid[p] = {args.seed: ok}
        stats[p] = {args.seed: st}
    failing = [p for p in M["pairs"] if not valid[p][args.seed]]
    if failing and args.fallback_seed is not None:
        d44 = C.multi_regen_dir(args.fallback_seed)
        print("[regen] seed %d で使えない対 %s を seed %d で対ごと作り直す（理由 %s）" % (
            args.seed, failing, args.fallback_seed, {p: stats[p][args.seed]["reasons"] for p in failing}), flush=True)
        runs[args.fallback_seed] = gen_fn(args.fallback_seed, failing, d44)
        seed_dirs[args.fallback_seed] = d44
        for p in failing:
            ok, st = pair_validity(d44, p)
            valid[p][args.fallback_seed] = ok
            stats[p][args.fallback_seed] = st
    chosen = {p: next((s for s in (args.seed, args.fallback_seed) if s is not None and valid[p].get(s)), None)
              for p in M["pairs"]}
    fd = assemble_final(chosen, seed_dirs)
    meta.update({
        "runs": {str(s): v for s, v in runs.items()},
        "seed_dirs": {str(s): v for s, v in seed_dirs.items()},
        "valid": {str(p): {str(s): v for s, v in d.items()} for p, d in valid.items()},
        "wav_stats": {str(p): {str(s): v for s, v in d.items()} for p, d in stats.items()},
        "chosen_seed": {str(p): s for p, s in chosen.items()},
        "unresolved_pairs": [p for p, s in chosen.items() if s is None],
        "chosen_tgt_silent_recorded_only": {str(p): stats[p][s]["tgt_silent_recorded_only"]
                                            for p, s in chosen.items()
                                            if s is not None and stats[p][s]["tgt_silent_recorded_only"]},
        "row_compare_vs_prod": {str(p): compare_rows(seed_dirs[s], p) for p, s in chosen.items() if s is not None},
        "final_dir": fd, "elapsed_min": round((time.time() - t0) / 60, 2),
    })
    out = os.path.join(fd, "regen_meta.json")
    if os.path.exists(out):                      # 出し直し: 前の回の記録を残す
        old = C.read_json(out)
        meta["history"] = old.get("history", []) + [
            {k: old.get(k) for k in ("date", "chosen_seed", "unresolved_pairs", "valid", "runs", "elapsed_min")}]
    C.write_json(meta, out)
    print("[regen] chosen=%s unresolved=%s -> %s" % (chosen, meta["unresolved_pairs"], fd), flush=True)
    return meta


# ---------------------------------------------------------------------------
def _seed_refs(path):
    """モジュール内で SEED という名前を読む箇所（行番号）を静的に数える（import しない）."""
    import ast
    tree = ast.parse(open(path).read())
    return sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "SEED"})


def cpu_dryrun(args):
    rep = {"problems": []}
    _static_checks(rep, M["pairs"], M["check_pair"])
    ck = M["check_pair"]
    rep["plan"] = {"seed": args.seed, "fallback_seed": args.fallback_seed,
                   "out_seed43": C.multi_regen_dir(args.seed), "final": C.multi_final_dir(),
                   "check_pair": ck, "argv": ["--pairs_module", M["pairs_module"], "--lang", M["lang"],
                                              "--reference_wav", M["reference_wav"],
                                              "--reference_text", M["reference_text"],
                                              "--output_dir", "<out_dir>/_attempts/aNN_{main,tgt}"],
                   "n_wav_to_generate": 6 * len(M["pairs"]) + 6}
    out = os.path.join(C.DRYRUN_DIR, "multientity_cpu_dryrun.json")
    C.write_json(rep, out)
    print(json.dumps({k: v for k, v in rep.items() if k != "pairs"}, ensure_ascii=False, indent=1))
    print("[dryrun] ->", out)


def _static_checks(rep, pairs, check_pair):
    """予行の静的な確かめ（データと本番の行・参照音声と参照文・SEED の参照箇所・重みとコードの置き場）.
    check_pair が None なら seed 42 の再現確認の対を調べない（--campaign thankyou）."""
    sys.path.insert(0, os.path.join(C.PROJECT_ROOT, "data"))
    import importlib
    items = importlib.import_module(M["pairs_module"]).build_pairs()
    if len(items) != 300 or any(it["pair_idx"] != i for i, it in enumerate(items)):
        rep["problems"].append("pairs_module が 300 対・pair_idx 0..299 でない")
    pr = load_rows(prod_dir(), "results.json")
    pt = load_rows(prod_dir(), "results_tgt.json")
    rep["pairs"] = {}
    for p in list(pairs) + ([check_pair] if check_pair is not None else []):
        it = items[p]
        r = pr.get(p, {})
        same = (r.get("text_a") == it["text_a"] and r.get("text_refs") == it["text_refs"]
                and r.get("words_b") == [sp["word_b"] for sp in it["spans"]])
        rep["pairs"][str(p)] = {"role": "check" if p == check_pair else "target",
                                "text_a": it["text_a"], "text_refs": it["text_refs"],
                                "prod_row_matches_data": same, "prod_mode": r.get("mode"),
                                "prod_has_tgt_row": p in pt}
        if not same or r.get("mode") != "splice_multi":
            rep["problems"].append("pair %d: 本番の行とデータが一致しない" % p)
    if check_pair is not None:
        ck = check_pair
        ok, st = C.multi_pair_validity(prod_dir(), ck)
        rep["check_pair_validity"] = {"ok": ok, "reasons": st["reasons"],
                                      "tgt_silent_recorded_only": st["tgt_silent_recorded_only"]}
        if not ok or st["tgt_silent_recorded_only"]:
            rep["problems"].append("check_pair %d が使える対でない: %s %s" % (ck, st["reasons"],
                                                                         st["tgt_silent_recorded_only"]))
    # 参照音声と参照文
    if not os.path.exists(M["reference_wav"]):
        rep["problems"].append("参照音声が無い: %s" % M["reference_wav"])
    s = C.read_json(os.path.join(prod_dir(), "results.json"))["summary"]
    if s.get("reference_wav") != M["reference_wav"]:
        rep["problems"].append("本番の reference_wav が違う: %s" % s.get("reference_wav"))
    log = os.path.join(C.PROJECT_ROOT, "logs", "autoqueue", "multient_en.log")
    line = [ln for ln in open(log, encoding="utf-8") if "Whisper transcript:" in ln]
    got = line[0].split("Whisper transcript: ", 1)[1].rstrip("\n") if line else None
    rep["log_reference_text"] = got
    if got != M["reference_text"]:
        rep["problems"].append("参照文がログと一致しない: %r" % got)
    # SEED の参照箇所（置き換える 2 モジュール以外に SEED 定数を読む生成経路が無いこと）
    gm = os.path.join(ANALYSIS, "generate_multientity.py")
    eu = os.path.join(PROJECTION, "eval_utmos_speaker_fixed.py")
    sm = os.path.join(PROJECTION, "splice_multi.py")
    rep["SEED_refs"] = {"generate_multientity.py": _seed_refs(gm), "eval_utmos_speaker_fixed.py": _seed_refs(eu),
                        "splice_multi.py": _seed_refs(sm)}
    if rep["SEED_refs"]["splice_multi.py"]:
        rep["problems"].append("splice_multi.py も SEED を読む: %s" % rep["SEED_refs"]["splice_multi.py"])
    # 重み（読まない。置き場があるかだけ）
    hub = os.path.join(C.PROJECT_ROOT, ".cache", "huggingface", "hub")
    for m in ("models--Aratako--T5Gemma-TTS-2b-2b", "models--google--t5gemma-2b-2b-ul2",
              "models--NandemoGHS--Anime-XCodec2-44.1kHz-v2", "models--openai--whisper-large-v3",
              "models--facebook--wav2vec2-xlsr-53-espeak-cv-ft"):
        if not os.path.isdir(os.path.join(hub, m, "snapshots")):
            rep["problems"].append("HF cache に %s が無い" % m)
    tc = os.path.join(C.PROJECT_ROOT, ".torch_cache", "hub", "tarepan_SpeechMOS_v1.2.0")
    if not os.path.isdir(tc):
        rep["problems"].append("UTMOS の torch hub cache が無い: %s" % tc)
    vendor = os.environ.get("T5GEMMA_ROOT", os.path.join(os.environ.get("REPO_ROOT", "/mnt"), "src", "_vendor", "t5gemma_tts"))
    for f in ("inference_tts_utils.py", "duration_estimator.py", "data/tokenizer.py", "models/t5gemma.py"):
        if not os.path.isfile(os.path.join(vendor, f)):
            rep["problems"].append("vendor のコードが無い: %s" % f)


# ---------------------------------------------------------------------------
# --campaign thankyou（著者判断 2026-09-30。regen2_common.THANKYOU）
# ---------------------------------------------------------------------------
T = C.THANKYOU
SCORER = os.path.join(ANALYSIS, "score_multientity.py")
_ASR_RE = re.compile(r"^asr_r(\d{2})\.json$")


def thankyou_plan():
    return {int(p): list(s) for p, s in T["seed_plan"].items()}


def _asr_files(d):
    ad = os.path.join(d, "asr")
    if not os.path.isdir(ad):
        return []
    return sorted(os.path.join(ad, n) for n in os.listdir(ad) if _ASR_RE.match(n))


def asr_file_status(path, d):
    """書き起こしのファイルが、置き場 d の results.json の対をすべて、失敗なく覆うか（覆うなら使い回す）."""
    res = load_rows(d, "results.json")
    try:
        arows = {r["pair_idx"]: r for r in C.read_json(path)["rows"]}
    except Exception as e:  # noqa: BLE001
        return False, "unreadable: %r" % (e,)
    if sorted(arows) != sorted(res):
        return False, "pairs %s != results.json pairs %s" % (sorted(arows), sorted(res))
    for p, r in sorted(arows.items()):
        errs = sorted(k for k in r if k.endswith("_err"))
        if errs:
            return False, "pair %d has %s" % (p, errs)
        for k in M["conds"]:
            if k in (res[p].get("conds") or {}) and not isinstance(r.get(k + "_transcript"), str):
                return False, "pair %d lacks %s_transcript" % (p, k)
    return True, "ok"


def _subprocess_scorer(d, out_json):
    """本番の 2 実体の採点を 1 行も変えずに回す（run_score_regen2_gpu.sh の multi と同じ呼び方。asr_model は既定値
    openai/whisper-large-v3 = 本番の summary.asr_model）."""
    cmd = [sys.executable, "-u", SCORER, "--dir", d, "--lang", M["lang"], "--output_json", out_json]
    print("[thankyou] scorer: %s" % " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=C.PROJECT_ROOT, env=dict(os.environ))
    return r.returncode, cmd


def run_asr(d, scorer=None):
    """置き場 d の書き起こし（本番の採点の出力）を返す: (path, {pair: row}, info).
    使い回せるファイル（asr_file_status が ok）があればそれを使い（判定を出し直しで変えないため）、無ければ
    新しい asr_rNN.json に書き起こす（前のファイルは消さない）。書き起こしが欠けたら止める."""
    C.assert_regen_path(d)
    checked = []
    for f in _asr_files(d):
        ok, why = asr_file_status(f, d)
        checked.append({"file": f, "usable": ok, "why": why})
        if ok:
            print("[thankyou] reuse transcription %s" % f, flush=True)
            return f, {r["pair_idx"]: r for r in C.read_json(f)["rows"]}, {"reused": True, "checked": checked}
    if not load_rows(d, "results.json"):
        return None, {}, {"reused": False, "checked": checked, "note": "no results.json rows (nothing to transcribe)"}
    nums = [int(_ASR_RE.match(os.path.basename(f)).group(1)) for f in _asr_files(d)]
    out = os.path.join(d, "asr", "asr_r%02d.json" % (1 + max(nums or [0])))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rc, cmd = (scorer or _subprocess_scorer)(d, out)
    if rc != 0 or not os.path.exists(out):
        raise SystemExit("[thankyou] score_multientity.py が失敗した（rc=%s）: %s。同じジョブを出し直すと新しい "
                         "asr_rNN に書き起こす（生成は済んだ対を作り直さない）" % (rc, d))
    ok, why = asr_file_status(out, d)
    info = {"reused": False, "checked": checked, "cmd": cmd, "new_file": out, "usable": ok, "why": why}
    if not ok:
        raise SystemExit("[thankyou] 書き起こしが欠けた（%s）: %s。seed を選ばずに止める。同じジョブを出し直すと新しい "
                         "asr_rNN に書き起こす（%s は消さずに残る）" % (why, d, out))
    return out, {r["pair_idx"]: r for r in C.read_json(out)["rows"]}, info


def assemble_final_asr(chosen, asr, fd):
    """採った seed の置き場の書き起こしの行（seed の判定に使ったもの。書き起こし直さない）を final にまとめる.
    merge_regen2_scores.py --campaign thankyou はこの rows を本番の行と入れ替える."""
    rows, src = [], {}
    for p, s in sorted(chosen.items()):
        if s is None:
            continue
        f = asr[s]["file"]
        r = {x["pair_idx"]: x for x in C.read_json(f)["rows"]}.get(p)
        if r is None:
            raise SystemExit("採った対 %d（seed %d）の書き起こしの行が %s に無い" % (p, s, f))
        rows.append(r)
        src[str(p)] = f
    prod_s = C.read_json(os.path.join(prod_dir(), M["asr_json"]))["summary"]
    doc = {"summary": C.asr_summary(rows, dict(prod_s, dir=fd)), "rows": rows,
           "regen2_thankyou": {"campaign": T["id"], "sources": src,
                               "note": "rows copied from the seed-folder transcriptions that decided the seed "
                                       "(scripts/analysis/score_multientity.py unchanged); not re-transcribed"}}
    out = os.path.join(fd, T["asr_final_name"])
    if os.path.exists(out):
        if C.read_json(out).get("rows") != rows:
            raise SystemExit("final に中身の違う %s が既にある（消さずに止める）" % out)
        return out
    C.write_json(doc, out, indent=2)
    return out


def finalize_thankyou(args, meta, gen_fn, asr_fn):
    """対ごとに THANKYOU["seed_plan"] の順に作り、THANKYOU["rule"] だけで seed を選ぶ.
    gen_fn(seed, pairs, out_dir) が生成、asr_fn(out_dir) が (path, rows, info) を返す書き起こし（検査では偽物）."""
    pairs = list(T["pairs"])
    plan = thankyou_plan()
    seeds = sorted({s for v in plan.values() for s in v})
    fd = C.assert_regen_path(C.thankyou_final_dir())
    fz0 = C.frozen_digest()
    if not fz0["digest"].startswith(T["frozen_digest_prefix"]):
        raise SystemExit("凍結した T5Gemma EN (a) の digest %s が %s... でない（何も作らずに止める）"
                         % (fz0["digest"], T["frozen_digest_prefix"]))
    meta.update({"campaign": T["id"], "date": time.strftime("%Y-%m-%d %H:%M:%S"), "decision": T["decision"],
                 "rule": T["rule"], "normalization": T["normalization"], "silence_criterion": C.SILENCE_CRITERION,
                 "pairs": pairs, "seed_plan": {str(p): v for p, v in plan.items()},
                 "flagged_prod_clips": {str(p): v for p, v in T["flagged_prod_clips"].items()},
                 "reference_text": M["reference_text"], "scorer": os.path.relpath(SCORER, C.PROJECT_ROOT),
                 "frozen_t5gemma_en_a_before": fz0})
    t0 = time.time()
    runs, asr, seed_dirs = {}, {}, {}
    valid = {p: {} for p in pairs}
    stats = {p: {} for p in pairs}
    left = {p: list(plan[p]) for p in pairs}
    chosen = {}
    for seed in seeds:
        todo = [p for p in pairs if p not in chosen and left[p] and left[p][0] == seed]
        if not todo:
            continue
        d = C.assert_regen_path(C.thankyou_seed_dir(seed))
        seed_dirs[seed] = d
        print("[thankyou] seed %d: pairs %s -> %s" % (seed, todo, d), flush=True)
        runs[seed] = gen_fn(seed, todo, d)
        path, arows, info = asr_fn(d)
        asr[seed] = dict(info, file=path)
        undetermined = []
        for p in todo:
            ok, st = C.thankyou_pair_validity(d, p, arows.get(p))
            valid[p][seed], stats[p][seed] = ok, st
            if st["undetermined"]:
                undetermined.append(p)
        print("[thankyou] seed %d validity: %s" % (seed, {p: (valid[p][seed], stats[p][seed]["reasons"])
                                                          for p in todo}), flush=True)
        if undetermined:
            C.write_json({"seed": seed, "pairs": todo, "undetermined": undetermined, "asr": asr[seed],
                          "wav_stats": {str(p): stats[p][seed] for p in todo}},
                         os.path.join(d, "regen2_thankyou_undetermined_%s.json" % time.strftime("%Y%m%d_%H%M%S")))
            raise SystemExit("[thankyou] seed %d: 書き起こしが欠けて決められない対 %s。seed を選ばずに止める"
                             % (seed, undetermined))
        for p in todo:
            left[p].pop(0)
            if valid[p][seed]:
                chosen[p] = seed
    chosen = {p: chosen.get(p) for p in pairs}
    unresolved = [p for p in pairs if chosen[p] is None]
    assemble_final(chosen, seed_dirs, fd=fd, note="rows of the chosen seed only (regen2 thankyou 2026-09-30)")
    asr_final = assemble_final_asr(chosen, asr, fd)
    fz1 = C.frozen_digest()
    meta.update({
        "runs": {str(s): v for s, v in runs.items()},
        "seed_dirs": {str(s): v for s, v in seed_dirs.items()},
        "asr": {str(s): v for s, v in asr.items()},
        "valid": {str(p): {str(s): v for s, v in d_.items()} for p, d_ in valid.items()},
        "wav_stats": {str(p): {str(s): v for s, v in d_.items()} for p, d_ in stats.items()},
        "chosen_seed": {str(p): s for p, s in chosen.items()},
        "unresolved_pairs": unresolved,
        "chosen_tgt_silent_recorded_only": {str(p): stats[p][s]["tgt_silent_recorded_only"]
                                            for p, s in chosen.items()
                                            if s is not None and stats[p][s]["tgt_silent_recorded_only"]},
        "row_compare_vs_prod": {str(p): compare_rows(seed_dirs[s], p) for p, s in chosen.items() if s is not None},
        "final_dir": fd, "final_asr": asr_final, "frozen_t5gemma_en_a_after": fz1,
        "frozen_unchanged": fz1 == fz0, "elapsed_min": round((time.time() - t0) / 60, 2),
    })
    out = os.path.join(fd, T["meta_name"])
    if os.path.exists(out):                      # 出し直し: 前の回の記録を残す
        old = C.read_json(out)
        meta["history"] = old.get("history", []) + [
            {k: old.get(k) for k in ("date", "chosen_seed", "unresolved_pairs", "valid", "runs", "asr",
                                     "elapsed_min")}]
    C.write_json(meta, out)
    print("[thankyou] chosen=%s unresolved=%s -> %s" % (chosen, unresolved, out), flush=True)
    if fz1 != fz0:
        raise SystemExit("凍結した T5Gemma EN (a) が変わった（ありえない）")
    if unresolved:
        print("[thankyou] STOP: 対 %s は計画の seed %s でも使えない。著者に報告する（seed 45 は試さない。apply は "
              "どの対も差し替えない）" % (unresolved, {p: plan[p] for p in unresolved}), flush=True)
    return meta


def gpu_main_thankyou(args):
    import torch
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "hf_env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                                                     "TRANSFORMERS_OFFLINE", "TORCH_HOME", "NUMBA_CACHE_DIR",
                                                     "OMP_NUM_THREADS", "GEN_THREADS")}}
    g5a_meta = os.path.join(C.G5A_MULTI_FINAL_REAL, "regen_meta.json")
    chk = C.read_json(g5a_meta).get("check_seed42") if os.path.exists(g5a_meta) else None
    meta = {"env": env, "check_seed42_reference": {
        "source": g5a_meta, "pair": (chk or {}).get("pair"), "match_all": (chk or {}).get("match_all"),
        "note": "not re-run for this campaign; same generate_multientity path and container as G5a (2026-09-29)"}}
    finalize_thankyou(args, meta, run_generate, run_asr)


def cpu_dryrun_thankyou(args):
    rep = {"campaign": T["id"], "problems": [], "host_cpu": C.host_cpu_state()}
    pairs = list(T["pairs"])
    plan = thankyou_plan()
    _static_checks(rep, pairs, None)
    if sorted(plan) != sorted(pairs):
        rep["problems"].append("seed_plan の対 %s が pairs %s と違う" % (sorted(plan), pairs))
    # 基準の単体の確かめ（書き起こしの正規化）
    cases = {" Thank you.": True, "Thank you!": True, "THANK YOU": True, "thank-you": True, " Thank you… ": True,
             "Thank you very much.": False, "Thanks.": False, " Thank you. Thank you.": False, "": False,
             "Darwin, Mars, and many others are famous.": False}
    rep["rule_unit_cases"] = {k: C.is_thankyou(k) for k in cases}
    if rep["rule_unit_cases"] != cases or C.is_thankyou(None):
        rep["problems"].append("is_thankyou の単体の確かめが合わない: %s" % rep["rule_unit_cases"])
    # 本番の書き起こし（Table 3 (c) の入力）を基準で走査: 当たるのは著者が挙げた 6 本だけのはず
    prod_asr = C.read_json(os.path.join(prod_dir(), M["asr_json"]))
    hits = C.thankyou_scan(prod_asr["rows"])
    want = {p: v for p, v in T["flagged_prod_clips"].items()}
    rep["prod_scan_thankyou"] = {str(p): v for p, v in sorted(hits.items())}
    rep["prod_scan_matches_author_list"] = hits == want
    if hits != want:
        rep["problems"].append("本番の書き起こしで基準に当たる音声 %s が著者の 6 本 %s と違う" % (hits, want))
    rep["prod_asr_model"] = prod_asr["summary"].get("asr_model")
    src = open(SCORER, encoding="utf-8").read()
    if 'default="%s"' % rep["prod_asr_model"] not in src:
        rep["problems"].append("score_multientity.py の既定の asr_model が本番の %s でない" % rep["prod_asr_model"])
    # 著者の 6 本は無音の基準に通ること（通るから基準を足した）
    rep["flagged_clip_silence"] = {}
    for p, ks in sorted(want.items()):
        for k in ks:
            w = C.wav_check(os.path.join(prod_dir(), "s%04d_%s.wav" % (p, k)))
            rep["flagged_clip_silence"]["%d_%s" % (p, k)] = {kk: w.get(kk) for kk in ("rms", "peak", "active_frac",
                                                                                     "silent", "why")}
            if w["bad"]:
                rep["problems"].append("対 %d の %s が無音の基準に当たる（想定外）: %s" % (p, k, w.get("why")))
    # 今の本番の各対が使えるか（無音・行）と、今の本番の Table 3 (c)
    rep["prod_pair_validity_now"] = {}
    arows = {r["pair_idx"]: r for r in prod_asr["rows"]}
    for p in pairs:
        ok, st = C.thankyou_pair_validity(prod_dir(), p, arows.get(p))
        rep["prod_pair_validity_now"][str(p)] = {"valid": ok, "reasons": st["reasons"]}
        if ok:
            rep["problems"].append("対 %d は今の本番で基準に当たらない（作り直す理由が無い）" % p)
    rep["table3c_now"] = C.table3c_cells(prod_asr["rows"])
    rec43 = os.path.join(prod_dir(), T["prod_table3c_now_record"])
    last = C.read_json(rec43)["merges"][-1].get("table3c_after") if os.path.exists(rec43) else None
    rep["table3c_now_equals_g5a_record"] = rep["table3c_now"] == last
    if rep["table3c_now"] != last:
        rep["problems"].append("今の本番の Table 3 (c) %s が G5a の記録 %s と違う" % (rep["table3c_now"], last))
    rep["utmos_now"] = C.utmos_means(C.read_json(os.path.join(prod_dir(), "results.json"))["summary"])
    # 対 257 の今の本番は G5a の seed 43 の音声（退避で残す）
    g = {}
    for k in ALL_CONDS:
        n = "s%04d_%s.wav" % (257, k)
        a, b = os.path.join(prod_dir(), n), os.path.join(C.G5A_MULTI_FINAL_REAL, n)
        g[k] = os.path.exists(b) and C.md5_file(a) == C.md5_file(b)
    rep["pair257_prod_is_g5a_seed43"] = g
    row257 = load_rows(prod_dir(), "results.json").get(257, {}).get("regen2")
    rep["pair257_prod_row_regen2"] = row257
    if not all(g.values()) or (row257 or {}).get("seed") != 43:
        rep["problems"].append("対 257 の今の本番が G5a の seed 43 の音声・行でない: %s %s" % (g, row257))
    # 新しい置き場と記録がまだ無いこと（上書きしない）
    news = {"root": C.thankyou_root(), "final": C.thankyou_final_dir(), "backup": C.thankyou_backup_dir(),
            "record_pairs": os.path.join(prod_dir(), T["record_pairs"]),
            "record_scores": os.path.join(prod_dir(), T["record_scores"])}
    news.update({"seed%d" % s: C.thankyou_seed_dir(s) for s in sorted({s for v in plan.values() for s in v})})
    rep["new_paths"] = {k: {"path": v, "exists": os.path.exists(v)} for k, v in news.items()}
    for k, v in news.items():
        if k != "root" and os.path.exists(v):
            rep["problems"].append("新しい置き場・記録が既にある: %s" % v)
        if k not in ("record_pairs", "record_scores"):
            C.assert_regen_path(v)
    fz = C.frozen_digest()
    rep["frozen_t5gemma_en_a"] = fz
    if not fz["digest"].startswith(T["frozen_digest_prefix"]):
        rep["problems"].append("凍結した T5Gemma EN (a) の digest %s が %s... でない" % (fz["digest"],
                                                                                  T["frozen_digest_prefix"]))
    rep["env"] = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                                                 "NUMBA_CACHE_DIR", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                 "GEN_THREADS")}
    first = [p for p in pairs if plan[p][0] == 43]
    rep["plan"] = {"seed_plan": {str(p): v for p, v in plan.items()},
                   "seed43_pairs": first, "seed44_pairs_at_least": [p for p in pairs if plan[p][0] == 44],
                   "seed44_pairs_at_most": [p for p in pairs if 44 in plan[p]],
                   "n_wav_to_generate_min_max": [6 * len(pairs), 6 * (len(first) + len(pairs))],
                   "scorer_cmd": [os.path.relpath(SCORER, C.PROJECT_ROOT), "--dir", "<seed_dir>", "--lang", M["lang"],
                                  "--output_json", "<seed_dir>/asr/asr_rNN.json"],
                   "gpu_cmd": "SLURM_TIME=01:00:00 slurm_run.sh bash "
                              "scripts/regen2/run_g5a_multi_gpu.sh thankyou",
                   "check_seed42": "not re-run (G5a 2026-09-29 pair 3 md5 match recorded in "
                                   + os.path.join(C.G5A_MULTI_FINAL_REAL, "regen_meta.json") + ")"}
    out = os.path.join(C.thankyou_dryrun_dir(), "regen_cpu_dryrun.json")
    C.write_json(rep, out)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("pairs", "flagged_clip_silence")},
                     ensure_ascii=False, indent=1))
    print("\n".join(C.before_after_lines(rep["table3c_now"], None, rep["utmos_now"], None)))
    print("[dryrun] problems=%d -> %s" % (len(rep["problems"]), out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="g5a", choices=["g5a", "thankyou"],
                    help="g5a: 9/29 の無音の対（既定。これまでと同じ）、thankyou: 9/30 の「Thank you.」の 6 対")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--fallback_seed", type=int, default=44, help="負数で無効")
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--cpu_dryrun", action="store_true")
    args = ap.parse_args()
    if args.fallback_seed is not None and args.fallback_seed < 0:
        args.fallback_seed = None
    if args.campaign == "thankyou":
        if args.cpu_dryrun:
            cpu_dryrun_thankyou(args)
            return
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("GPU が無い。slurm_run.sh 経由で実行すること（--cpu_dryrun 以外）")
        gpu_main_thankyou(args)
        return
    if args.cpu_dryrun:
        cpu_dryrun(args)
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GPU が無い。slurm_run.sh 経由で実行すること（--cpu_dryrun 以外）")
    gpu_main(args)


if __name__ == "__main__":
    main()
