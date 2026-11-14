#!/usr/bin/env python3
"""G5a: Table 3 の Llasa-3B で使えない対を、seed 43（だめなら 44）で対ごとに作り直す（GPU）.

著者判断（2026-09-28 の方針を 2026-09-29 に Table 3 へ適用）:
  * 使えない対（無音の音声・生成の失敗）は対ごと別 seed で作り直す。対の中（BL・TGT・INJ）は同じ seed。
  * 対は使えるかどうかの基準だけで選ぶ。結果では選ばない。基準は regen2_common.llasa_pair_validity:
      - 対のどの wav も regen2_common.wav_check で無音・欠け・読めないに当たらない
        （無音 = (RMS<0.005 かつ peak<0.05) または 20 ms 区間の active_frac<0.05）
      - すべての層の progress.json に status == "ok" の行がある（"error" は生成の失敗。
        run_llasa_eval.run_generation は INJ の wav を保存した後で fired != 1 などを "error" にする）
  * Llasa-3B はサンプリング（temperature 0.8）なので、seed 43 で作り直すことそのものが判断。

対象（scripts/regen2/silence_scan_g5a.py --stage pre で確認済み、2026-09-29 の統一した基準）:
  en_a     _cc_wavs_a/llasa_en            対 197, 247            (a) EN
  zh_a     _cc_wavs_a/llasa_zh            対 145                 (a) ZH（--span_extension）
  en_sweep _cc_wavs_a/llasa_en_sweep_n300 対 97, 177, 257        (b) EN 5 点。BL/TGT は 5 点で共有
  zh_sweep _cc_wavs_a/llasa_zh_sweep_n300 対 85, 180, 199, 284   (b) ZH 5 点。BL/TGT は 5 点で共有

手順は本番と同じコード（scripts/llasa/run_llasa_eval.run_generation と
run_llasa_layersweep.layer_args）を改変せずに呼ぶ。変えるのは次の 3 点だけ:
  1. --seed 43（run_llasa_eval の既存の引数。GenConfig(seed=...) が generate 直前の seed_all に渡る）。
  2. 対を対象の対だけに絞る（run_generation に渡す pairs のリストを絞る。各対は generate 直前に
     seed を戻すので、他の対の有無は結果に影響しない）。
  3. chat template の日付を本番のその対の生成日に固定する。Llasa の chat template は実行日を
     prompt に入れる（llasa_splice.chat_template_date の説明）。本番の日付は wav の mtime（UTC、
     コンテナの時計は UTC）から決めた（regen2_common.LLASA_CELLS の date_string）。
     llasa_splice.render_prompt の date_string の既定値だけを差し替える（関数自体は元のまま）。
モデルは 1 回だけ読み、4 セルで使い回す（掃引の本番も 1 回読みで 5 層を回している）。

出し直し（時間切れ・ノード障害）で壊れないための作り（2026-09-29 追加）:
  * run_generation は毎回、新しい試行の置き場 <root>/<cell>/_attempts/aNN/ に書かせる。
    本体の置き場（<root>/<cell>/{wav,meta}）には、試行が終わった対だけを写す（wav を写してから
    progress.json に行を足す。帳面は tmp → 置き換え）。途中で止まった対は本体に残らず、次の実行で
    同じ seed のまま作り直される（同じ seed の再生であって、結果による選び直しではない）。
  * 本体に status == "error" の行がある対は、その seed での失敗として確定し、作り直さない（seed 44 へ）。
  * 失敗の振り分けは regen2_common.triage_attempt:
      - GPU メモリ不足・通信などの環境の失敗（is_infra_error）は本体に書かず、ジョブを止める（seed 44 に回さない）。
      - CUDA の実行時の失敗（device-side assert など）は、その対と、同じ試行でそれより後に処理した対を本体に
        書かずに止める。同じ対が同じ置き場（seed・層）で 2 回目の CUDA の失敗を出したら、決まった失敗として
        error の行を書く（→ seed 44）。記録は <cell>/meta/<Lxx>/regen2_cuda_fault_strikes.json。
      - その他の例外は生成の失敗として error の行を書く（→ seed 44）。
    同じジョブを出し直すと、本体に書かなかった対だけを同じ seed で作る。

確かめること:
  * 先に seed 42 のまま、同じ日付で、対象外の対 1 つ（check_pair）を作り直し、本番の wav と md5 が
    一致するかを記録する（日付とコード経路の再現確認。GPU の違いで一致しないこともあるので
    結果は記録のみ。apply は一致しないとき明示の許可を求める）。wav が無いときも止めずに不一致として記録する。
  * 作り直した行の kappa / prompt_len / スパンが本番の行と一致するか（seed に依らない値なので、
    日付とスパンが本番と同じであることの確認になる）。
  * seed 43 の対を使えるかの基準で調べ、使えない対は対ごと seed 44 で作り直す。44 でも使えなければ
    unresolved として記録し、apply はその対を差し替えない（著者判断待ち）。

出力（本番は書き換えない）:
  eval_results/_regen2_seed43/llasa/<cell>/{wav,meta}[/Lxx]（本体）と <cell>/_attempts/aNN/（試行ごと）
  eval_results/_regen2_seed44/llasa/<cell>/...（必要なときだけ）
  eval_results/_regen2_seed43/llasa/_check42/<cell>/...
  eval_results/_regen2_final/llasa/<cell>/wav[/Lxx]/s{i:04d}_{bl,inj,tgt}.wav（対ごとに採用した seed の音声）
  eval_results/_regen2_final/llasa/<cell>/regen_meta.json

GPU（プロジェクト直下から）:
  SLURM_TIME=02:00:00 slurm_run.sh bash scripts/regen2/run_g5a_llasa_gpu.sh
CPU の予行（モデルは読まない。tokenizer だけで prompt とスパンを本番の行と照合する。ホストで直接。
スレッド 13・CPU 0-12・nice 10・GPU を見せない・HF オフラインは、--cpu_dryrun のときスクリプトの中で設定する）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/regen2/regen_llasa_pairs_seed.py --cpu_dryrun
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import regen2_common as C  # noqa: E402

C.cap_threads()
if "--cpu_dryrun" in sys.argv:
    C.host_cpu_setup()   # ホストの CPU の予行だけ（GPU で動かすときは呼ばない）。llasa_splice / torch より前
# HF はオフライン（tokenizer・重みはローカルの snapshot のパスで読み、XCodec2 の中の facebook/w2v-bert-2.0 も
# キャッシュの refs/main から読む）。run_g5a_llasa_gpu.sh でも export している。transformers の import より前
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.join(C.PROJECT_ROOT, "scripts", "llasa"))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import llasa_splice as L  # noqa: E402  (HF_HOME の固定もこの import で入る)
import run_llasa_eval as R  # noqa: E402
import run_llasa_layersweep as LS  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(13)

# ---------------------------------------------------------------------------
# 日付の固定（render_prompt の date_string の既定値だけを差し替える）
# ---------------------------------------------------------------------------
_PIN = {"date": None, "calls": 0}
_orig_render_prompt = L.render_prompt


def _render_prompt_pinned(tokenizer, chat, date_string=None):
    if date_string is None:
        date_string = _PIN["date"]
    if date_string is None:
        raise RuntimeError("date_string が固定されていない")
    _PIN["calls"] += 1
    return _orig_render_prompt(tokenizer, chat, date_string=date_string)


L.render_prompt = _render_prompt_pinned

KINDS = C.LLASA_KINDS
# 生成の関数（検査では偽物に差し替える）
_GEN = {"run_generation": None}


def _run_generation(a, todo, bundle):
    fn = _GEN["run_generation"] or R.run_generation
    return fn(a, todo, bundle=bundle)


def prod_rows(cell, sub):
    c = C.LLASA_CELLS[cell]
    p = os.path.join(C.EV, c["prod_out"], sub, "progress.json")
    return {r["pair_idx"]: r for r in C.read_json(p)["rows"]}


def load_pairs_checked(lang):
    pairs = R.load_pairs(lang, 300)
    assert len(pairs) == 300 and all(p["pair_idx"] == i for i, p in enumerate(pairs)), "pair_idx が 0..299 でない"
    return pairs


def make_base_args(cell, seed, wav_dir, out_dir, layer):
    c = C.LLASA_CELLS[cell]
    argv = ["--lang", c["lang"], "--n", "300", "--layer", str(layer), "--seed", str(seed),
            "--wav_dir", wav_dir, "--out_dir", out_dir]
    if c["span_extension"]:
        argv.append("--span_extension")
    a = R.build_argparser().parse_args(argv)
    return R.resolve_paths(a), argv


def _new_attempt_dir(root, cell):
    base = C.assert_regen_path(os.path.join(root, cell, "_attempts"))
    os.makedirs(base, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(base, "a%02d" % k)):
        k += 1
    d = os.path.join(base, "a%02d" % k)
    os.makedirs(d)
    return d


def run_cell(cell, seed, targets, root, bundle, layers=None):
    """本番と同じ run_generation を対象の対だけで回す。掃引は layersweep と同じ hard link の作り.

    run_generation は新しい試行の置き場に書かせ、終わった対だけを本体（root/cell/{wav,meta}）に写す。
    本体で済み（ok の行と wav がそろう）か失敗（error の行）の対は、もう一度作らない."""
    c = C.LLASA_CELLS[cell]
    _PIN["date"] = c["date_string"]
    base_wav = C.assert_regen_path(os.path.join(root, cell, "wav"))
    base_out = C.assert_regen_path(os.path.join(root, cell, "meta"))
    by_idx = {p["pair_idx"]: p for p in load_pairs_checked(c["lang"])}
    targets = sorted(set(targets))
    assert all(p in by_idx for p in targets)
    if c["kind"] == "single":
        plan = [(c["layer"], "")]
    else:
        plan = [(L_, "L%02d" % L_) for L_ in (layers or c["layers"])]
    first_sub = plan[0][1]
    first_wav = os.path.join(base_wav, first_sub)
    first_prog = os.path.join(base_out, first_sub, "progress.json")
    info = {"cell": cell, "seed": seed, "pairs": targets, "date_string": c["date_string"],
            "span_extension": c["span_extension"], "runs": []}
    att = None
    for i, (layer, sub) in enumerate(plan):
        main_wav = os.path.join(base_wav, sub)
        main_meta = os.path.join(base_out, sub)
        os.makedirs(main_wav, exist_ok=True)
        os.makedirs(main_meta, exist_ok=True)
        prog_p = os.path.join(main_meta, "progress.json")
        rows = C.read_rows(prog_p)
        rows0 = C.read_rows(first_prog)
        kinds = KINDS if i == 0 else ("inj",)
        need, kept = [], {}
        for p in targets:
            st = (rows.get(p) or {}).get("status")
            if st == "error":
                kept[p] = "error row (generation failure at this seed; final)"
                continue
            if i > 0 and (rows0.get(p) or {}).get("status") != "ok":
                kept[p] = "first layer not ok (pair already unusable at this seed)"
                continue
            if st == "ok" and all(os.path.exists(os.path.join(main_wav, "s%04d_%s.wav" % (p, k))) for k in kinds):
                kept[p] = "complete"
                continue
            need.append(p)
        run = {"layer": layer, "sub": sub, "kept": {str(k): v for k, v in kept.items()}, "need": need}
        if not need:
            info["runs"].append(run)
            continue
        if att is None:
            att = _new_attempt_dir(root, cell)
        att_wav, att_meta = os.path.join(att, "wav"), os.path.join(att, "meta")
        if c["kind"] == "single":
            a, argv = make_base_args(cell, seed, att_wav, att_meta, layer)
        else:
            base, argv = make_base_args(cell, seed, att_wav, att_meta, plan[0][0])
            # 2 層目以降は本体の最初の層の BL/TGT を hard link で持ち込む（run_llasa_layersweep と同じ）
            a = LS.layer_args(base, layer, att_wav, att_meta, link_from=(None if i == 0 else first_wav))
            argv = argv + ["(layer %d)" % layer]
        print("[regen] %s L%02d seed=%d pairs=%s attempt=%s link_from=%s" % (
            cell, layer, seed, need, att, getattr(a, "link_bl_tgt_from", None)), flush=True)
        man = _run_generation(a, [by_idx[p] for p in need], bundle)
        arows = C.read_rows(os.path.join(a.out_dir, "progress.json"))
        # 失敗の振り分け（regen2_common.triage_attempt）: 環境の失敗と CUDA の失敗の後に処理した対は本体に
        # 書かない。CUDA の失敗は同じ置き場で 2 回目なら生成の失敗（→ seed 44）。run_generation は need の順に処理する
        errs = {p: (arows[p].get("err") or "error") for p in need if (arows.get(p) or {}).get("status") == "error"}
        strike_p = os.path.join(main_meta, "regen2_cuda_fault_strikes.json")
        tri = C.triage_attempt(need, errs, C.read_strikes(strike_p))
        C.record_strikes(strike_p, tri["new_strikes"], os.path.relpath(att, os.path.join(root, cell)))
        infra = tri["hold"]
        merged = {}
        for p in need:
            ar = arows.get(p)
            if ar is None or p in infra:
                continue                              # 本体に残さない（次の実行で同じ seed のまま作る）
            if ar.get("status") == "ok":
                for k in kinds:
                    src = os.path.join(a.wav_dir, "s%04d_%s.wav" % (p, k))
                    if not os.path.exists(src):
                        raise SystemExit("試行の行は ok なのに wav が無い: %s" % src)
                    shutil.copyfile(src, os.path.join(main_wav, "s%04d_%s.wav" % (p, k)))
                if i > 0:
                    for k in ("bl", "tgt"):
                        s0 = os.path.join(first_wav, "s%04d_%s.wav" % (p, k))
                        dst = os.path.join(main_wav, "s%04d_%s.wav" % (p, k))
                        if os.path.exists(dst):
                            if not os.path.samefile(s0, dst):
                                raise SystemExit("本体の %s が最初の層の BL/TGT と別のファイル" % dst)
                        else:
                            os.link(s0, dst)
            merged[p] = ar                            # ok の行、または生成の失敗（error）の行
        if merged:
            rows.update(merged)
            C.write_json_atomic_regen({"rows": [rows[k] for k in sorted(rows)]}, prog_p, indent=2)
        run.update({"attempt": att, "argv": argv, "manifest": man, "merged": sorted(merged),
                    "held": {str(k): v for k, v in infra.items()},
                    "generation_failures": {str(k): str(v)[:300] for k, v in tri["write_fail"].items()},
                    "cuda_strikes_added": {str(k): v for k, v in tri["new_strikes"].items()}})
        info["runs"].append(run)
        if tri["stop"]:
            C.write_json_atomic_regen(info, os.path.join(att, "run_info_stopped.json"))
            tag = "[cuda]" if tri["cut_after"] is not None else "[infra]"
            raise SystemExit("%s %s L%02d seed %d: 止める（対 %s）。本体に書かなかった対 %s は、同じジョブを出し直すと"
                             "同じ seed のまま作る。生成の失敗として書いた対 %s は seed 44 に回る。"
                             % (tag, cell, layer, seed, tri["stop"], sorted(infra), sorted(tri["write_fail"])))
    return info


def pair_validity(root, cell, pidx):
    return C.llasa_pair_validity(root, cell, pidx)


def rows_complete(root, cell, pidx):
    """ok の行が、その層で作った wav の記録を持つか（最初の層は BL/TGT/INJ、他の層は INJ）。
    試行の置き場から写すので必ずそろうはず。欠けたらコードの誤りなので止める."""
    missing = []
    for j, (layer, sub) in enumerate(C.llasa_layer_dirs(cell)):
        r = C.read_rows(os.path.join(root, cell, "meta", sub, "progress.json")).get(pidx) or {}
        need = ("n_tok_bl", "n_tok_tgt", "n_tok_inj") if j == 0 else ("n_tok_inj",)
        missing += ["L%02d:%s" % (layer, k) for k in need if r.get(k) is None]
    return missing


def compare_rows(cell, root, pidx, only_first=False):
    """seed に依らない値（prompt 長・スパン・kappa）を本番の行と比べる."""
    out = {}
    subs = C.llasa_layer_dirs(cell)
    for layer, sub in (subs[:1] if only_first else subs):
        new = C.read_rows(os.path.join(root, cell, "meta", sub, "progress.json"))
        old = prod_rows(cell, sub)
        n, o = new.get(pidx, {}), old.get(pidx, {})
        d = {}
        for k in ("prompt_len_a", "prompt_len_b", "s_span", "t_span", "s_tokens", "t_tokens", "span_extension"):
            d[k] = (n.get(k) == o.get(k))
        for k in ("kappa_at_splice", "kappa_last_layer"):
            if n.get(k) is not None and o.get(k) is not None:
                d[k + "_absdiff"] = round(abs(n[k] - o[k]), 6)
        d["status_new"] = n.get("status")
        out["L%02d" % layer] = d
    return out


def run_check(cell, bundle, root_check):
    """seed 42・同じ日付で対象外の 1 対を作り直し、本番と md5 を比べる（掃引は最初の層だけ）.
    記録のための確認なので、wav が無くても止めずに不一致として残す（apply は --accept_check_mismatch を求める）."""
    c = C.LLASA_CELLS[cell]
    cp = c["check_pair"]
    layers = None if c["kind"] == "single" else [c["layers"][0]]
    run_cell(cell, 42, [cp], root_check, bundle, layers=layers)
    layer, sub = C.llasa_layer_dirs(cell)[0]
    res = {"pair": cp, "layer": layer, "date_string": c["date_string"], "files": {}}
    for k in KINDS:
        new = os.path.join(root_check, cell, "wav", sub, "s%04d_%s.wav" % (cp, k))
        old = os.path.join(C.EV, c["prod_wav"], sub, "s%04d_%s.wav" % (cp, k))
        try:
            a, b = C.md5_file(new), C.md5_file(old)
            res["files"][k] = {"md5_new": a, "md5_prod": b, "match": a == b}
        except FileNotFoundError as e:
            res["files"][k] = {"error": "missing", "detail": str(e)[:300], "match": False}
    res["match_all"] = all(v["match"] for v in res["files"].values())
    row = C.read_rows(os.path.join(root_check, cell, "meta", sub, "progress.json")).get(cp) or {}
    res["row_status"] = row.get("status")
    res["row_err"] = row.get("err")
    res["rows"] = compare_rows(cell, root_check, cp, only_first=True)
    print("[check42] %s pair %d md5 match=%s status=%s rows=%s" % (
        cell, cp, res["match_all"], res["row_status"], res["rows"]), flush=True)
    return res


def assemble_final(cell, chosen, rows_by_seed):
    """対ごとに採用した seed の音声を _regen2_final にまとめる（掃引の BL/TGT は層間で hard link）."""
    final_root = C.llasa_final_root()
    out = {}
    subs = C.llasa_layer_dirs(cell)
    for pidx, seed in sorted(chosen.items()):
        if seed is None:
            continue
        src_root = C.llasa_regen_root(seed)
        first_sub = subs[0][1]
        for layer, sub in subs:
            dst_dir = C.assert_regen_path(os.path.join(final_root, cell, "wav", sub))
            os.makedirs(dst_dir, exist_ok=True)
            for k in KINDS:
                n = "s%04d_%s.wav" % (pidx, k)
                dst = os.path.join(dst_dir, n)
                if k in ("bl", "tgt") and sub != first_sub:
                    src = os.path.join(final_root, cell, "wav", first_sub, n)
                    if os.path.exists(dst):
                        if not os.path.samefile(src, dst):
                            raise SystemExit("final に別の %s が既にある" % dst)
                        continue
                    os.link(src, dst)
                    continue
                src = os.path.join(src_root, cell, "wav", sub, n)
                if os.path.exists(dst):
                    if C.md5_file(dst) != C.md5_file(src):
                        raise SystemExit("final に中身の違う %s が既にある（消さずに止める）" % dst)
                    continue
                shutil.copy2(src, dst)
        out[pidx] = {"seed": seed, "rows": rows_by_seed[seed][pidx]}
    return out


def gpu_main(args):
    cells = [c for c in C.LLASA_CELL_ORDER if c in args.cells.split(",")]
    print("[env] torch=%s cuda=%s gpu=%s transformers=%s" % (
        torch.__version__, torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        __import__("transformers").__version__), flush=True)
    # モデル読み込みの引数（model_id / dtype / attn / device）は本番の既定と同じ。ここでは何も書かない。
    a0, _ = make_base_args(cells[0], args.seed,
                           os.path.join(C.llasa_regen_root(args.seed), cells[0], "wav"),
                           os.path.join(C.llasa_regen_root(args.seed), cells[0], "meta"), 26)
    bundle = R.load_bundle(a0)
    if not L.chat_template_date(bundle.tokenizer):
        print("[warn] chat template に日付が入らない（日付の固定は無意味）", flush=True)
    env = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "chat_template_has_date": bool(L.chat_template_date(bundle.tokenizer)),
           "model_dir": bundle.model_dir, "codec_dir": bundle.codec_dir,
           "hf_env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                                                     "TRANSFORMERS_OFFLINE")}}

    root43 = C.llasa_regen_root(args.seed)
    root_check = os.path.join(root43, "_check42")
    for cell in cells:
        meta = {"env": env}
        if not args.no_check:
            meta["check_seed42"] = run_check(cell, bundle, root_check)

        def gen_fn(cell_, seed_, pairs_, root_):
            return run_cell(cell_, seed_, pairs_, root_, bundle)

        finalize_cell(cell, args, meta, gen_fn)


def finalize_cell(cell, args, meta, gen_fn):
    """seed 43 で作る → 使えるかの基準で判定 → 使えない対だけ seed 44 で対ごと作る → 対ごとに seed を選び
    _regen2_final にまとめ、regen_meta.json を書く。gen_fn(cell, seed, pairs, root) が生成（検査では偽物）."""
    c = C.LLASA_CELLS[cell]
    t0 = time.time()
    meta.update({"cell": cell, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "planned_pairs": c["pairs"], "seed": args.seed, "fallback_seed": args.fallback_seed,
                 "date_string": c["date_string"], "span_extension": c["span_extension"],
                 "orig_cmd": c["orig_cmd"],
                 "criterion": {"silence": C.SILENCE_CRITERION,
                               "pair_valid_iff": "every wav of the pair (BL/TGT and the INJ of every layer) is "
                                                 "present, readable and not silent, and every layer's "
                                                 "progress.json row has status == 'ok'"}})
    root43 = C.llasa_regen_root(args.seed)
    runs = {args.seed: gen_fn(cell, args.seed, c["pairs"], root43)}
    valid = {}
    stats = {}
    for p in c["pairs"]:
        ok, st = pair_validity(root43, cell, p)
        valid[p] = {args.seed: ok}
        stats.setdefault(p, {})[args.seed] = st
    failing = [p for p in c["pairs"] if not valid[p][args.seed]]
    if failing and args.fallback_seed is not None:
        print("[regen] %s: seed %d で使えない対 %s を seed %d で対ごと作り直す（理由 %s）" % (
            cell, args.seed, failing, args.fallback_seed,
            {p: stats[p][args.seed]["reasons"] for p in failing}), flush=True)
        root44 = C.llasa_regen_root(args.fallback_seed)
        runs[args.fallback_seed] = gen_fn(cell, args.fallback_seed, failing, root44)
        for p in failing:
            ok, st = pair_validity(root44, cell, p)
            valid[p][args.fallback_seed] = ok
            stats[p][args.fallback_seed] = st
    chosen = {}
    for p in c["pairs"]:
        chosen[p] = next((s for s in (args.seed, args.fallback_seed) if s is not None and valid[p].get(s)), None)
    for p, s in chosen.items():
        if s is not None:
            miss = rows_complete(C.llasa_regen_root(s), cell, p)
            if miss:
                raise SystemExit("%s 対 %d（seed %d）の ok の行に生成の記録が欠ける %s（コードの誤り）" % (cell, p, s, miss))
    rows_by_seed = {}
    for s in runs:
        rr = {}
        for layer, sub in C.llasa_layer_dirs(cell):
            for pidx, r in C.read_rows(os.path.join(C.llasa_regen_root(s), cell, "meta", sub,
                                                    "progress.json")).items():
                rr.setdefault(pidx, {})["L%02d" % layer] = r
        rows_by_seed[s] = rr
    final = assemble_final(cell, chosen, rows_by_seed)
    meta.update({
        "runs": {str(s): v for s, v in runs.items()},
        "valid": {str(p): {str(s): v for s, v in d.items()} for p, d in valid.items()},
        "wav_stats": {str(p): {str(s): v for s, v in d.items()} for p, d in stats.items()},
        "chosen_seed": {str(p): s for p, s in chosen.items()},
        "unresolved_pairs": [p for p, s in chosen.items() if s is None],
        "row_compare_vs_prod": {str(p): compare_rows(cell, C.llasa_regen_root(s), p)
                                for p, s in chosen.items() if s is not None},
        "final": {str(p): v for p, v in final.items()},
        "final_dir": os.path.join(C.llasa_final_root(), cell),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    })
    out = os.path.join(C.llasa_final_root(), cell, "regen_meta.json")
    C.assert_regen_path(out)
    if os.path.exists(out):                      # 出し直し: 前の回の記録を残す
        old = C.read_json(out)
        meta["history"] = old.get("history", []) + [
            {k: old.get(k) for k in ("date", "chosen_seed", "unresolved_pairs", "valid", "runs", "elapsed_min")}]
    C.write_json(meta, out)
    print("[regen] %s chosen=%s unresolved=%s -> %s" % (cell, chosen, meta["unresolved_pairs"], out), flush=True)
    return meta


# ---------------------------------------------------------------------------
# CPU の予行: tokenizer だけで prompt とスパンを本番の行と照合する（モデルは読まない）
# ---------------------------------------------------------------------------
class _TokOnly:
    def __init__(self, tok):
        self.tokenizer = tok


def cpu_dryrun(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(L.local_snapshot(L.DEFAULT_MODEL_ID))
    fb = _TokOnly(tok)
    rep = {"chat_template_has_date": bool(L.chat_template_date(tok)), "cells": {}, "problems": [],
           "hf_env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                                                     "TRANSFORMERS_OFFLINE")},
           "host_cpu": C.host_cpu_state(),
           "tokenizer_source": L.local_snapshot(L.DEFAULT_MODEL_ID)}
    for cell in C.LLASA_CELL_ORDER:
        c = C.LLASA_CELLS[cell]
        _PIN["date"] = c["date_string"]
        layer, sub = C.llasa_layer_dirs(cell)[0]
        man_dir = os.path.join(C.EV, c["prod_out"], sub)
        man = [f for f in os.listdir(man_dir) if f.endswith("_manifest.json")]
        mm = C.read_json(os.path.join(man_dir, man[0]))["meta"]
        n_ref = mm["ref"]["n_tokens"]
        ref_text = mm["ref"]["text"]
        rows = prod_rows(cell, sub)
        pairs = load_pairs_checked(c["lang"])
        ent = {"date_string": c["date_string"], "n_ref_tokens": n_ref, "ref_text": ref_text,
               "prod_manifest_meta": {k: mm.get(k) for k in ("seed", "span_extension", "layer", "dtype",
                                                             "attn_implementation", "decode")},
               "pairs": {}}
        if mm.get("seed") != 42 or mm["decode"]["temperature"] != 0.8:
            rep["problems"].append("%s: 本番 manifest の seed/temperature が想定外" % cell)
        if bool(mm.get("span_extension")) != c["span_extension"] and cell != "zh_sweep":
            rep["problems"].append("%s: span_extension が本番 manifest と違う" % cell)
        a, argv = make_base_args(cell, args.seed, os.path.join(C.llasa_regen_root(args.seed), cell, "wav"),
                                 os.path.join(C.llasa_regen_root(args.seed), cell, "meta"), layer)
        ent["argv_seed43"] = argv
        ent["condition_tag_seed43"] = R.condition_tag(a)
        dummy_ids = list(range(1000, 1000 + n_ref))
        for pidx in list(c["pairs"]) + [c["check_pair"]]:
            p = pairs[pidx]
            in_a = L.build_input_text(p["text_a"], ref_text)
            in_b = L.build_input_text(p["text_b"], ref_text)
            src, tgt, ext = L.prepare_context_pair(
                fb, in_a, p["word_a"], in_b, p["word_b"], ref_speech_ids=dummy_ids,
                region_text_a=p["text_a"], region_text_b=p["text_b"], need_hidden_a=False, need_hidden_b=False,
                strict_span=True, span_extension=c["span_extension"])
            r = rows[pidx]
            got = {"prompt_len_a": src["prompt_len"], "prompt_len_b": tgt["prompt_len"],
                   "s_span": [src["span"]["tok_start"], src["span"]["tok_end"]],
                   "t_span": [tgt["span"]["tok_start"], tgt["span"]["tok_end"]],
                   "s_tokens": src["span"]["entity_tokens"], "t_tokens": tgt["span"]["entity_tokens"],
                   "span_extension": ext}
            cmp = {k: (got[k] == r.get(k)) for k in got}
            date_in_prompt = c["date_string"] in src["prompt"]
            ent["pairs"][str(pidx)] = {"role": "check" if pidx == c["check_pair"] else "target",
                                       "text_a": p["text_a"], "text_b": p["text_b"],
                                       "word_a": p["word_a"], "word_b": p["word_b"],
                                       "prod_row_status": r.get("status"),
                                       "match_prod_row": cmp, "date_in_prompt": date_in_prompt}
            if not all(cmp.values()):
                rep["problems"].append("%s pair %d: prompt/スパンが本番の行と違う %s" % (cell, pidx, cmp))
            if not date_in_prompt:
                rep["problems"].append("%s pair %d: 固定した日付が prompt に無い" % (cell, pidx))
        # check_pair は対象外・使える対であること（統一した基準の wav_check と行の status）
        for k in KINDS:
            w = C.wav_check(os.path.join(C.EV, c["prod_wav"], sub, "s%04d_%s.wav" % (c["check_pair"], k)))
            if w["bad"]:
                rep["problems"].append("%s check_pair %d の %s が使えない（%s）" % (cell, c["check_pair"], k, w["why"]))
        if rows[c["check_pair"]].get("status") != "ok":
            rep["problems"].append("%s check_pair %d の本番の行が ok でない" % (cell, c["check_pair"]))
        rep["cells"][cell] = ent
    rep["render_prompt_calls"] = _PIN["calls"]
    out = os.path.join(C.DRYRUN_DIR, "llasa_cpu_dryrun.json")
    C.write_json(rep, out)
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, ensure_ascii=False, indent=1))
    for cell, ent in rep["cells"].items():
        print("[dryrun] %s date=%s tag=%s pairs=%s" % (
            cell, ent["date_string"], ent["condition_tag_seed43"],
            {k: all(v["match_prod_row"].values()) for k, v in ent["pairs"].items()}))
    print("[dryrun] ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(C.LLASA_CELL_ORDER))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--fallback_seed", type=int, default=44,
                    help="seed 43 で使えない対だけ、対ごとこの seed で作り直す（負数で無効）")
    ap.add_argument("--no_check", action="store_true", help="seed 42 の再現確認を省く")
    ap.add_argument("--cpu_dryrun", action="store_true")
    args = ap.parse_args()
    if args.fallback_seed is not None and args.fallback_seed < 0:
        args.fallback_seed = None
    if args.cpu_dryrun:
        cpu_dryrun(args)
    else:
        if not torch.cuda.is_available():
            raise SystemExit("GPU が無い。slurm_run.sh 経由で実行すること（--cpu_dryrun 以外）")
        gpu_main(args)


if __name__ == "__main__":
    main()
