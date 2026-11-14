#!/usr/bin/env python3
"""spanfix (2a): 付録 B（Table A1）の JA 4 対・ZH 4 対を、修正後の範囲の関数で作り直す（GPU, 2026-10-02）.

英語の spanfix_appB_en_gen.py（10/02 に英語 300 対を作り直したもの）と同じ手順を、JA/ZH の 8 対に当てる:
  * 本番スクリプト scripts/projection/eval_crossattn_entity_dtw.py を 1 バイトも変えずに main() を呼ぶ。
    引数は 5/19-20 の本番と同じ（--lang {ja,zh} --cross_culture --inject_mode length_preserve --n 300）で、
    --output_dir（新しい置き場）だけを足す。seed はモジュール変数 SEED を差し替える（42 = 本番と同じ。43 / 44 は作り直し）。
  * 範囲の関数は、本番が main() の中で from-import する scripts/encoder_injection_t5gemma.py の
    find_entity_token_range（7/11 に修正済み）がそのまま使われる。これが 5/19-20 との唯一の違い。
  * 1 回の起動で扱う対（todo）以外は、新しい置き場の checkpoint.json に「済み」（placeholder）と先に書いておき、
    本番の resume の仕組みで todo の対だけを生成・採点させる。
  * 5 本（BL, INJ, BL_mask, INJ_mask, TGT）の音声を写し取って wav（float32）に保存し、無音の基準
    （scripts/regen2/silence_criterion.py）で判定する。実行時の範囲と遮断した列が、事前に CPU で求めた修正後の関数の
    範囲（vendor の正規化を当てた文で求めたもの）と一致することを対ごとに確かめる。数値（MCD・類似度）は見ない。
    写し取り・範囲の記録・print の記録の部品は spanfix_appB_en_gen.py のものを読んで使う（英語のファイルは変えない）。
  * pyopenjtalk: vendor の duration_estimator.py の中でだけ使わせる（appB_jazh_common.PyopenjtalkGate）。
    JA では vendor の発話長推定が pyopenjtalk を使うこと（5/19-20 と同じ推定）をモデルを読む前に確かめる。

seed の決まり（appB_jazh_common.next_action / allowed_todo。英語と同じ）:
  * seed 42: 対象の 4 対（JA 59,99,164,179 / ZH 15,145,246,295）のうち、まだ流していない対と、最後の回が
    対の良し悪しでない状態（run_error / alignment_exception / capture_error / no_row_unexplained / crashed_no_record）の対。
  * seed 43: seed 42 の最後の回が「対として使えない」（unaligned / generation_failed / silent）の対だけ。
  * seed 44: seed 43 の最後の回が「対として使えない」の対だけ。45 以上は流さない（著者の判断）。
  * 最後まで終わっていない実行の対は、squeue で動いていないことを確かめてから --confirm_stopped <置き場> を付けたときだけ。

置き場（新規。既存の結果には書かない。何も消さない）:
  eval_results/_spanfix_appB_jazh/<lang>/seed<S>/<seed42|regen>_<日時>_<ジョブ番号>/
    checkpoint.json results.json（本番 main() が書く）、run_record.json、events.jsonl、capture/p<idx>.json、
    wav/s<idx>_<条件>.wav、run_end.json

GPU（プロジェクト直下 $REPO_ROOT から。1 ジョブ = GPU 1 枚。利用者全体で同時 2 枚まで）:
  SLURM_TIME=00:20:00 slurm_run.sh python3 -u /mnt/scripts/spanfix/gpu_guard.py --min_free_gb 30 -- /mnt/scripts/spanfix/appB_jazh_gen.py --lang ja --seed 42
  SLURM_TIME=00:20:00 slurm_run.sh python3 -u /mnt/scripts/spanfix/gpu_guard.py --min_free_gb 30 -- /mnt/scripts/spanfix/appB_jazh_gen.py --lang zh --seed 42
  （seed 43 / 44 と同じ seed のやり直しは appB_jazh_collect.py --status が示したときだけ）
CPU の試運転（モデルは読まない。tokenizer と偽の生成で配管・seed の決まり・取り込みを確かめる。ホストで直接）:
  taskset -c 0-12 python3 $REPO_ROOT/scripts/spanfix/appB_jazh_gen.py --lang ja --seed 42 --dry_run
"""
import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import appB_jazh_common as J  # noqa: E402  標準ライブラリだけを読む

LAUNCH_ENV = J.setup_env()  # numpy・torch・HF を読む前に（スレッド 13、HF_HOME、HF_HUB_OFFLINE=1、NUMBA_CACHE_DIR）
if not J.IN_CONTAINER:
    sys.modules["pyopenjtalk"] = None  # ホスト（試運転）では pyopenjtalk を一切読まない

import argparse  # noqa: E402
import builtins  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402

import spanfix_appB_en_gen as G  # noqa: E402  英語の部品（EventLog, SpanTap, MultiCapture, PrintTap, ...）。読むだけ

C = J.C
SC = G.SC
PROJECT_ROOT = J.PROJECT_ROOT
EVAL_SCRIPT = C.P(C.EVAL_REL)
INJ_SCRIPT = C.P(C.INJ_REL)
CONDS = J.CONDS


# ---------------------------------------------------------------- 写し取り（範囲の確かめだけ JA/ZH 用）
class JZCapture(G.MultiCapture):
    """英語の MultiCapture と同じ写し取り・順序の確かめ・wav の保存・無音の判定.

    範囲の確かめだけを変える: 実行時に範囲の関数へ渡された文は、本番の get_encoder_output が lang=ja/zh で
    正規化した文（ZH は元の文のまま）。期待する文と範囲は事前に CPU で求めた値（expected）。
    """

    def _span_checks(self, idx, calls, labels):
        exp = self.expected.get(idx)
        if exp is None:
            return False, ["事前の範囲が無い"]
        problems = []
        for c, lab in zip(calls, labels):
            if lab not in G.EXPECTED_SPAN_CALLS:
                continue
            want_sides = G.EXPECTED_SPAN_CALLS[lab]
            got = c["spans"]
            if len(got) != len(want_sides):
                problems.append(f"{lab}: 範囲の呼び出しが {len(got)} 回（{len(want_sides)} 回のはず）")
                continue
            for g, side in zip(got, want_sides):
                if "error" in g:
                    problems.append(f"{lab}/{side}: 範囲の記録に失敗 {g['error']}")
                    continue
                ab = g.get("add_bos", 0)
                want = [exp[side][0] + ab, exp[side][1] + ab]
                if g.get("text") != exp[f"text_{side}_norm"] or g.get("word") != exp[f"word_{side}"] \
                        or g.get("span") != want:
                    problems.append(f"{lab}/{side}: 実行時 {g} / 期待 text={exp[f'text_{side}_norm']!r} span={want}")
            if lab in ("bl_mask", "inj_mask"):
                ab = (got[0].get("add_bos", 0) if got else 0)
                want_cols = list(range(exp["a"][0] + ab, exp["a"][1] + ab))
                if c["masked_cols"] != want_cols:
                    problems.append(f"{lab}: 遮断した列 {c['masked_cols']} / 実体語の範囲 {want_cols}")
        return not problems, problems


def expected_spans(span_pre, todo):
    return {i: {"a": span_pre[i]["a"]["fixed"], "b": span_pre[i]["b"]["fixed"],
                "text_a_norm": span_pre[i]["a"]["text_norm"], "text_b_norm": span_pre[i]["b"]["text_norm"],
                "word_a": span_pre[i]["a"]["word"], "word_b": span_pre[i]["b"]["word"]} for i in todo}


# ---------------------------------------------------------------- 1 回の実行
class _FakeKill(BaseException):
    """試運転だけ: 時間切れで止まった実行を作る（run_end.json を書かない）."""


def run_one(lang, M, EI, FA, *, seed, tag, todo, pairs, span_pre, data_root, mode, env_info, policy, on_gpu, gate,
            uniq=None, require_real_fa=True):
    import torch
    stamp = time.strftime("%Y%m%d_%H%M%S")
    uniq = uniq or (os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}")
    rel = J.run_dir_rel(lang, seed, tag, stamp, uniq)
    d = os.path.join(data_root, rel)
    rp = os.path.realpath(d)
    assert "/_spanfix_appB_jazh/" in rp + "/" and "/crossattn_entity_dtw_" not in rp \
        and "/_regen2_crossattn_merged/" not in rp, d
    os.makedirs(d, exist_ok=False)
    rows = [{"pair_idx": i, "placeholder": True} for i in range(J.N_PAIRS) if i not in set(todo)]
    with open(os.path.join(d, "checkpoint.json"), "x") as f:
        json.dump({"rows": rows}, f)
    done = {r["pair_idx"] for r in rows}
    assert [q["pair_idx"] for q in pairs if q["pair_idx"] not in done] == sorted(todo), "todo だけが生成される設定でない"

    log = G.EventLog(os.path.join(d, "events.jsonl"))
    orig_gen, orig_span = EI.generate_with_modified_encoder, EI.find_entity_token_range
    span_tap = G.SpanTap(orig_span)
    cap = JZCapture(orig_gen, [pairs[i] for i in todo], expected_spans(span_pre, todo), d, rel, log, span_tap,
                    abort_on_offload=(mode == J.MODE_GPU))
    EI.generate_with_modified_encoder = cap
    EI.find_entity_token_range = span_tap
    tap = G.PrintTap(log, cap)
    taps = G.install_print_taps(M, FA, tap, require_real_fa)
    seed_before = M.SEED
    M.SEED = seed
    argv = J.prod_args(lang) + ["--output_dir", d]
    env = {"project_root": PROJECT_ROOT, "host": platform.node(), "python": platform.python_version(),
           "torch": torch.__version__,
           "cuda_device": torch.cuda.get_device_name(0) if on_gpu else None,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
           "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"), "torch_num_threads": torch.get_num_threads(),
           "thread_env": {k: os.environ.get(k) for k in C.THREAD_VARS + tuple(J.EXTRA_ENV)},
           **{k: os.environ.get(k) for k in C.CACHE_VARS},
           "env_at_launch": LAUNCH_ENV, "env_set_by_script": {**C.cache_env(), **J.EXTRA_ENV}}
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        env["transformers"] = None
    env.update(env_info)
    rec = {"item": f"spanfix (2a) Appendix B tab:crossattn_mask {lang.upper()}, fixed entity span, pairs {list(J.TARGETS[lang])}",
           "producer": J.PRODUCER, "mode": mode, "lang": lang, "seed": seed, "tag": tag, "run_dir": rel,
           "started": J.now(), "todo": sorted(todo), "n_todo": len(todo),
           "main_argv": argv,  # コンテナの中では --output_dir が /mnt/<rel>（出どころの確かめがこれを求める）
           "module_SEED": {"before": seed_before, "after": M.SEED},
           "span_preflight": {str(i): span_pre[i] for i in todo},
           "span_tap_installed": EI.find_entity_token_range is span_tap,
           "generate_tap_installed": EI.generate_with_modified_encoder is cap,
           "print_taps": taps, "seed_policy": policy, "code_md5": J.current_code_md5(PROJECT_ROOT),
           "pyopenjtalk": gate.report(),
           "silence_criterion": {"id": SC.CRITERION_ID, "text": SC.CRITERION_TEXT,
                                 "params": {k: getattr(SC, k) for k in G.SILENCE_PARAMS}},
           "valid_criterion": "row aligned (Whisper found the entity in all 5 wavs, no Whisper exception), 5 frames, "
                              "5 wavs captured in the checked order, runtime spans equal the fixed function's spans "
                              "on the vendor-normalized text, none silent under " + SC.CRITERION_TEXT,
           "env": env}
    if mode == J.MODE_GPU:
        assert d == f"{J.CONTAINER_ROOT}/{rel}", (d, rel)
    J.write_json_new(rec, os.path.join(d, "run_record.json"))
    log("run_start", pair_idx=None, n_todo=len(todo))
    print(f"[appB_jazh] run {rel}: lang={lang} seed={seed} todo={todo} argv={argv}", flush=True)
    saved_argv = sys.argv
    sys.argv = [EVAL_SCRIPT] + argv
    t0 = time.time()
    err = None
    killed = False
    try:
        M.main()
    except _FakeKill:
        killed = True
    except BaseException as ex:  # noqa: BLE001  例外でも、写し取った wav と checkpoint と記録は残す
        err = f"{type(ex).__name__}: {str(ex)[:500]}"
        print(f"[appB_jazh] main() ended with {err}", flush=True)
        traceback.print_exc()
    finally:
        sys.argv = saved_argv
        M.SEED = seed_before
        EI.generate_with_modified_encoder, EI.find_entity_token_range = orig_gen, orig_span
    cap.close()
    if killed:
        return rel, None
    elapsed = round((time.time() - t0) / 60, 2)
    end = {"finished": J.now(), "elapsed_min": elapsed, "main_error": err,
           "hf_device_map": cap.device_map, "device_offload": cap.device_offload(),
           "n_unmatched_generate_calls": cap.n_unmatched, "pyopenjtalk_audit": gate.report()}
    run = C.load_run(data_root, rel)
    run["end"] = end
    st = {i: J.pair_status(run, i)[0] for i in todo}
    end["statuses"] = {str(i): s for i, s in st.items()}
    end["status_counts"] = {s: sum(1 for v in st.values() if v == s) for s in sorted(set(st.values()))}
    end["load_errors"] = run["errors"]
    J.write_json_new(end, os.path.join(d, "run_end.json"))
    log("run_end", pair_idx=None, main_error=err, elapsed_min=elapsed)
    print(f"[appB_jazh] RESULT {rel}: elapsed={elapsed}min main_error={err} statuses={end['status_counts']} "
          f"pyopenjtalk calls_from_vendor={end['pyopenjtalk_audit'].get('calls_from_vendor')} "
          f"violations={len(end['pyopenjtalk_audit'].get('violations') or [])}", flush=True)
    print("[appB_jazh] NEXT: ホストで appB_jazh_collect.py --status を流す", flush=True)
    return rel, end


# ---------------------------------------------------------------- 流す対を決める（seed の決まり）
def select_todo(lang, seed, pairs_arg, confirm_stopped, data_root, check_provenance=True):
    runs, warns = J.all_runs(data_root, lang)
    targets = list(J.TARGETS[lang])
    if pairs_arg:
        cand = sorted(set(int(x) for x in pairs_arg.split(",")))
        extra = sorted(set(cand) - set(targets))
        if extra:
            raise SystemExit(f"[appB_jazh] --pairs に作り直しの対象（{targets}）でない対がある: {extra}")
    else:
        cand = targets
    tag = "seed42" if seed == 42 else "regen"
    todo, why = J.allowed_todo(runs, lang, seed, cand, confirm_stopped, PROJECT_ROOT, check_provenance)
    if pairs_arg:
        refused = sorted(set(cand) - set(todo))
        if refused:
            raise SystemExit(f"[appB_jazh] --pairs のうち seed {seed} で流せない対: { {i: why[i] for i in refused[:5]} }")
    if len(todo) > J.CHUNK:
        raise SystemExit(f"[appB_jazh] 1 回に {len(todo)} 対は多い（上限 {J.CHUNK}）。--pairs で分ける")
    policy = {"lang": lang, "seed": seed, "tag": tag, "candidates": cand, "todo": todo,
              "not_run_here": {str(i): v for i, v in why.items()}, "confirm_stopped": list(confirm_stopped),
              "record_warnings": warns}
    return todo, tag, policy


# ---------------------------------------------------------------- 試運転（CPU, モデルなし）
class FakeGate:
    """試運転だけ: 見張りの報告の形を真似る（violation=True で違反を 1 件持つ）."""

    def __init__(self, violation=False):
        self.violation = violation

    def report(self):
        return {"installed": True, "finalized": True, "vendor_duration_uses_pyopenjtalk": True, "FAKE": True,
                "violations": ([{"kind": "call_outside_vendor", "function": "g2p", "caller": "FAKE"}]
                               if self.violation else []), "calls_from_vendor": {}}


def make_fake_eval(lang, tok, fixed_fn, norm, pairs, base_rows):
    """本番 main() と同じ resume・呼び出し順・print を持つ偽物。数値の行は取り込みの元の行を写しただけ."""
    import numpy as np
    import torch
    sr = 16000

    def fake_generate(model, text_tokenizer, audio_tokenizer, modified_memory, encoder_attention_mask,
                      encoder_position_ids, target_text, device, codec_audio_sr, codec_sr):
        beh = FakeM.behavior.get((target_text, bool((encoder_attention_mask == 0).any().item())), "tone")
        if beh == "none":
            return None
        amp = 0.0 if beh == "zeros" else 0.3
        t = torch.arange(int(0.6 * codec_audio_sr), dtype=torch.float32) / codec_audio_sr
        return (amp * torch.sin(2 * np.pi * 220 * t)).view(1, 1, -1)

    EI = types.SimpleNamespace(generate_with_modified_encoder=fake_generate, find_entity_token_range=fixed_fn)
    FA = types.SimpleNamespace(print=builtins.print)

    class FakeM:
        SEED = 42
        print = builtins.print
        cases = {}
        behavior = {}
        kill_after = None

        @staticmethod
        def main():
            argv = sys.argv
            assert argv[1:-2] == J.prod_args(lang), argv
            out = argv[argv.index("--output_dir") + 1]
            ck = os.path.join(out, "checkpoint.json")
            rows = J.read_json(ck)["rows"]
            done = {r["pair_idx"] for r in rows}
            gen, span = EI.generate_with_modified_encoder, EI.find_entity_token_range  # main() と同じく呼び出し時に引く
            for i, pair in enumerate(pairs):
                if pair["pair_idx"] in done:
                    continue
                case = FakeM.cases.get(i, "valid")
                ta, tb = norm(pair["text_a"], lang), norm(pair["text_b"], lang)
                L = len(tok.encode(ta, add_special_tokens=False)) + 2
                src = torch.randn(1, L, 4)
                ones = torch.ones(1, L, dtype=torch.long)
                sa = fixed_fn(tok, ta, pair["word_a"], add_bos=0)  # 記録しない（差し替えの外で求める）
                inj = src.clone()
                inj[0, sa[0]:sa[1]] += 1.0
                seq = [(src, ones, pair["text_a"], ["a"]), (inj, ones, pair["text_a"], ["a", "b"]),
                       (src.clone(), "mask", pair["text_a"], ["a"]), (inj.clone(), "mask", pair["text_a"], ["a", "b"]),
                       (torch.randn(1, L, 4), ones, pair["text_b"], [])]
                if case == "capture_error":  # INJ と BL_mask の順が入れ替わる
                    seq[1], seq[2] = seq[2], seq[1]
                outs = []
                try:
                    for k, (mem, attn, text, sides) in enumerate(seq):
                        for side in sides:
                            if side == "a":
                                span(tok, ta, pair["word_a"], add_bos=0)
                            else:
                                span(tok, tb if case != "text_mismatch" else pair["text_b"] + "。", pair["word_b"],
                                     add_bos=0)
                        if isinstance(attn, str):
                            attn = ones.clone()
                            s0, e0 = sa
                            if case == "span_mismatch":
                                s0, e0 = s0 + 1, e0 + 1  # 隣の token を遮断（旧の不具合と同じ形）
                            attn[0, s0:e0] = 0
                        outs.append(gen(None, None, None, mem, attn, None, text, "cpu", sr, 50))
                        if case == "run_error" and k == 1:
                            raise RuntimeError("FAKE CUDA out of memory (selftest)")
                except Exception as e:  # noqa: BLE001  本番 main() と同じ握りつぶし
                    FakeM.print(f"  [{i}] err: {str(e)[:200]}")
                    continue
                if FakeM.kill_after is not None and i > FakeM.kill_after:
                    raise _FakeKill()
                if any(o is None for o in outs):
                    continue
                if case == "alignment_exception":
                    FA.print("  [Whisper] transcribe failed: FAKE CUDA out of memory")
                row = copy.deepcopy(base_rows[i])
                row.update({k2: pair[k2] for k2 in ("pair_idx", "text_a", "text_b", "word_a", "word_b")})
                row["aligned"] = case not in ("unaligned", "alignment_exception")
                rows.append(row)
                done.add(pair["pair_idx"])
                with open(ck, "w") as f:  # 本番 main() と同じく checkpoint を書き直す（今回作った置き場の中だけ）
                    json.dump({"rows": rows}, f, ensure_ascii=False)
            with open(os.path.join(out, "results.json"), "w") as f:
                json.dump({"summary": {"FAKE": True}, "rows": rows}, f, ensure_ascii=False)

    return FakeM, EI, FA


def fake_selftest(lang, tok, fixed_fn, norm, pairs, span_all, base):
    """偽の main() で、写し取り・状態・seed の決まり・見張りの違反の扱いを確かめる（英語の試運転と同じ型）."""
    base_rows = {r["pair_idx"]: r for r in J.read_json(J.P(J.BASE_DIR_REL[lang] + "/results.json"))["rows"]}
    data_root = os.path.join(base, "fake_root")
    os.makedirs(data_root, exist_ok=False)
    FakeM, EI, FA = make_fake_eval(lang, tok, fixed_fn, norm, pairs, base_rows)

    def go(seed, tag, todo, cases, behavior=None, kill_after=None, uniq=None, gate=None):
        FakeM.cases, FakeM.behavior, FakeM.kill_after = cases, behavior or {}, kill_after
        return run_one(lang, FakeM, EI, FA, seed=seed, tag=tag, todo=todo, pairs=pairs, span_pre=span_all,
                       data_root=data_root, mode=J.MODE_FAKE, env_info={"FAKE": True}, policy={"FAKE": True},
                       on_gpu=False, gate=gate or FakeGate(), uniq=uniq, require_real_fa=False)

    def beh(idx, cond):
        p = pairs[idx]
        return {"bl": (p["text_a"], False), "inj_mask": (p["text_a"], True), "bl_mask": (p["text_a"], True),
                "tgt": (p["text_b"], False)}[cond]

    def na_of(runs, idxs):
        return {i: J.next_action(runs, lang, i, PROJECT_ROOT, check_provenance=False)["action"] for i in idxs}

    R = {}
    cases1 = {2: "silent", 3: "generation_failed", 4: "run_error", 5: "alignment_exception", 6: "unaligned",
              7: "capture_error", 8: "span_mismatch", 9: "text_mismatch"}
    b1 = {beh(2, "bl_mask"): "zeros", beh(3, "bl_mask"): "none"}
    rel1, end1 = go(42, "seed42", list(range(0, 10)), cases1, b1, uniq="fake1")
    want1 = {0: "valid", 1: "valid", 2: "silent", 3: "generation_failed", 4: "run_error",
             5: "alignment_exception", 6: "unaligned", 7: "capture_error", 8: "span_mismatch", 9: "span_mismatch"}
    got1 = {int(k): v for k, v in end1["statuses"].items()}
    assert got1 == want1, (got1, want1)
    R["run1_statuses"] = got1
    rel2, end2 = go(42, "seed42", [10, 11, 12], {}, kill_after=10, uniq="fake2")
    assert end2 is None
    runs, _ = J.all_runs(data_root, lang)
    r2 = [r for r in runs[42] if r["rel"] == rel2][0]
    st2 = {i: J.pair_status(r2, i)[0] for i in (10, 11, 12)}
    assert st2 == {10: "valid", 11: "crashed_no_record", 12: "crashed_no_record"}, st2
    R["killed_run_statuses"] = st2
    na = na_of(runs, range(0, 14))
    want_na = {0: "use_seed42", 1: "use_seed42", 2: "run_seed43", 3: "run_seed43", 4: "rerun_seed42",
               5: "rerun_seed42", 6: "run_seed43", 7: "rerun_seed42", 8: "stop_author_decision",
               9: "stop_author_decision", 10: "use_seed42", 11: "confirm_stopped_seed42",
               12: "confirm_stopped_seed42", 13: "run_seed42"}
    assert na == want_na, (na, want_na)
    R["next_actions"] = na
    t42, _ = J.allowed_todo(runs, lang, 42, list(range(14)), (), PROJECT_ROOT, check_provenance=False)
    t42c, _ = J.allowed_todo(runs, lang, 42, list(range(14)), (rel2,), PROJECT_ROOT, check_provenance=False)
    t43, _ = J.allowed_todo(runs, lang, 43, list(range(14)), (), PROJECT_ROOT, check_provenance=False)
    assert t42 == [4, 5, 7, 13] and t42c == [4, 5, 7, 11, 12, 13] and t43 == [2, 3, 6], (t42, t42c, t43)
    R["allowed_todo"] = {"seed42": t42, "seed42_confirm_stopped": t42c, "seed43": t43}
    pe = J.run_provenance_errors([r for r in runs[42] if r["rel"] == rel1][0], lang, 42)
    assert any("mode" in e for e in pe) and any("コンテナ" in e for e in pe) and any("GPU" in e for e in pe) \
        and any("vendor" in e for e in pe) and any("pyopenjtalk の見張り" in e for e in pe) is False, pe
    # 偽の見張り（FakeGate）は installed/finalized を持つので「見張りが入っていない」は出ない。JA の pyopenjtalk の
    # 経路の確かめ（vendor_duration_uses_pyopenjtalk）は FakeGate で True なので出ない。出どころで拒まれる理由は mode 等
    R["fake_refused_by_provenance"] = len(pe)
    rel3, _ = go(43, "regen", [2, 3, 6], {6: "run_error"}, {beh(3, "tgt"): "zeros"}, uniq="fake3")
    runs, _ = J.all_runs(data_root, lang)
    na3 = na_of(runs, (2, 3, 6))
    assert na3 == {2: "use_seed43", 3: "run_seed44", 6: "rerun_seed43"}, na3
    R["after_seed43"] = na3
    go(42, "seed42", [5], {5: "alignment_exception"}, uniq="fake4")
    runs, _ = J.all_runs(data_root, lang)
    assert na_of(runs, (5,))[5] == "stop_author_decision"
    R["two_alignment_exceptions"] = "stop_author_decision"
    # pyopenjtalk の決まりの違反が記録された実行: 対は止まって著者の判断（有効な行でも使わない）
    _, end5 = go(42, "seed42", [13], {}, uniq="fake5", gate=FakeGate(violation=True))
    assert end5["statuses"] == {"13": J.STATUS_PYOJT}, end5["statuses"]
    runs, _ = J.all_runs(data_root, lang)
    assert na_of(runs, (13,))[13] == "stop_author_decision"
    R["pyopenjtalk_violation"] = "stop_author_decision"
    # 保存した wav が元と一致（無損失）、範囲が一致
    import numpy as np
    import soundfile as sf
    cap0 = J.read_json(os.path.join(data_root, rel1, "capture", "p0000.json"))
    x, fs = sf.read(os.path.join(data_root, cap0["wavs"]["bl"]["path"]), dtype="float32")
    assert fs == 16000 and abs(float(np.max(np.abs(x))) - 0.3) < 1e-2 and cap0["span_match"] is True, cap0
    assert J.md5_file(os.path.join(data_root, cap0["wavs"]["bl"]["path"])) == cap0["wavs"]["bl"]["md5"]
    cap9 = J.read_json(os.path.join(data_root, rel1, "capture", "p0009.json"))
    assert cap9["span_match"] is False and any("期待 text=" in s for s in cap9["span_problems"]), cap9["span_problems"]
    R["wav_roundtrip_and_span_match"] = True
    R["text_mismatch_detected"] = cap9["span_problems"][:1]
    R["fake_root"] = os.path.relpath(data_root, PROJECT_ROOT)
    return R


def fake_targets_and_merge(lang, tok, fixed_fn, norm, pairs, span_all, base):
    """本物の対象の 4 対で、seed 42（1 対だけ無音）→ seed 43 → 取り込み（偽の記録を試運転の置き場へ）を通す.

    偽の行は取り込みの元の行の写しなので、取り込んだ表の値は今の表と同じになるはず（配管の確かめ）。
    """
    import appB_jazh_collect as K  # 同じ置き場。numpy と無音の基準を読む
    base_rows = {r["pair_idx"]: r for r in J.read_json(J.P(J.BASE_DIR_REL[lang] + "/results.json"))["rows"]}
    data_root = os.path.join(base, "fake_targets_root")
    os.makedirs(data_root, exist_ok=False)
    FakeM, EI, FA = make_fake_eval(lang, tok, fixed_fn, norm, pairs, base_rows)
    t = list(J.TARGETS[lang])

    def go(seed, tag, todo, behavior=None, uniq=None):
        FakeM.cases, FakeM.behavior, FakeM.kill_after = {}, behavior or {}, None
        return run_one(lang, FakeM, EI, FA, seed=seed, tag=tag, todo=todo, pairs=pairs, span_pre=span_all,
                       data_root=data_root, mode=J.MODE_FAKE, env_info={"FAKE": True}, policy={"FAKE": True},
                       on_gpu=False, gate=FakeGate(), uniq=uniq, require_real_fa=False)

    R = {}
    _, e1 = go(42, "seed42", t, {(pairs[t[0]]["text_a"], True): "zeros"}, uniq="fakeT1")  # t[0] の遮断 2 本が無音
    assert e1["statuses"] == {str(t[0]): "silent", **{str(i): "valid" for i in t[1:]}}, e1["statuses"]
    out_parent = os.path.join(base, "merged_fake")
    out = os.path.join(out_parent, f"{lang}cc_lp_n300")

    def refused(fn):
        try:
            fn()
        except SystemExit as e:
            return str(e)[:300]
        raise AssertionError("拒否されるはずが通った")

    R["refuse_before_seed43"] = refused(lambda: K.merge_lang(lang, data_root=data_root, out_dir=out, fake=True))
    assert not os.path.exists(out)
    go(43, "regen", [t[0]], uniq="fakeT2")
    runs, _ = J.all_runs(data_root, lang)
    na = {i: J.next_action(runs, lang, i, PROJECT_ROOT, check_provenance=False)["action"] for i in t}
    assert na == {t[0]: "use_seed43", **{i: "use_seed42" for i in t[1:]}}, na
    R["next_actions"] = na
    tv = K.merge_lang(lang, data_root=data_root, out_dir=out, fake=True)
    res = J.read_json(os.path.join(out, "results.json"))
    base_res = J.read_json(J.P(J.BASE_DIR_REL[lang] + "/results.json"))
    assert res["rows"] == base_res["rows"] and res["summary"] == base_res["summary"], "偽の取り込みが元と一致しない"
    assert J.read_json(os.path.join(out, "checkpoint.json"))["rows"] == base_res["rows"]
    mr = J.read_json(os.path.join(out, "merge_record.json"))
    assert mr["selection"][str(t[0])]["seed"] == 43 and all(mr["selection"][str(i)]["seed"] == 42 for i in t[1:])
    for m, cols in J.CURRENT_TABLE[lang].items():
        assert (tv["after_n300"][m]["intact"]["tex"], tv["after_n300"][m]["blocked"]["tex"]) == cols, (m, tv)
    assert sorted(os.listdir(out)) == sorted(K.OUT_NAMES)
    R["merged_equals_base_and_current_table"] = True
    R["refuse_existing_out"] = refused(lambda: K.merge_lang(lang, data_root=data_root, out_dir=out, fake=True))
    real_out = J.P(f"{J.OUT_REL}/{J.MERGED_PREFIX}SELFTEST_should_not_exist/{lang}cc_lp_n300")
    R["refuse_fake_into_real"] = refused(lambda: K.merge_lang(lang, data_root=data_root, out_dir=real_out, fake=True))
    R["refuse_real_without_records"] = refused(lambda: K.merge_lang(lang, data_root=J.HOST_ROOT, out_dir=real_out))
    assert not os.path.exists(os.path.dirname(real_out)), "拒否したのに本物の置き場ができた"
    R["merged_dir"] = os.path.relpath(out, PROJECT_ROOT)
    return R


def dry_run(args):
    assert not J.IN_CONTAINER, "試運転はホストで（CPU のみ）"
    import torch
    torch.set_num_threads(J.THREADS)
    lang = args.lang
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = J.P(f"{J.DRYRUN_REL}/gen_{lang}_{ts}_seed{args.seed}")
    os.makedirs(base, exist_ok=False)
    rep = {"date": J.now(), "args": vars(args), "env_at_launch": LAUNCH_ENV,
           "env_now": {k: os.environ.get(k) for k in C.CACHE_VARS + C.THREAD_VARS + tuple(J.EXTRA_ENV)
                       + ("CUDA_VISIBLE_DEVICES",)},
           "pyopenjtalk_blocked_on_host": sys.modules.get("pyopenjtalk", "absent") is None}
    rep["pinned_errors"] = J.pinned_errors(lang)
    assert not rep["pinned_errors"], rep["pinned_errors"]
    rep["static_patch_checks"] = G.static_patch_checks()
    rep["silence"] = {"params_errors": G.silence_param_errors(), "selftest": SC.selftest()}
    assert not rep["silence"]["params_errors"]
    rep["pyopenjtalk_gate_selftest"] = J.gate_selftest(os.path.join(base, "gate_selftest"))
    W = C.load_shim_module()
    off = C.setup_offline(W)
    rep["offline"] = {k: off[k] for k in ("hf_offline", "offline_preflight_ok", "offline_tokenizer_shim")}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_ID)
    fixed_fn, legacy_fn = C.span_functions()
    norm = J.vendor_normalizer()
    pairs = J.load_pairs(lang)
    span_all = J.span_table(tok, pairs, lang, fixed_fn, legacy_fn, norm)
    rep["span_summary"] = J.span_summary(span_all, lang)
    rep["span_audit_agreement"] = J.audit_agreement(span_all, lang)
    assert rep["span_summary"]["equals_targets"], rep["span_summary"]["pairs_whose_span_changes_vs_0520"]
    assert all(span_all[i]["fixed_ok"] for i in J.TARGETS[lang]), "対象の対で修正後の範囲が実体語を覆わない"
    assert rep["span_audit_agreement"]["n_disagree"] == 0, rep["span_audit_agreement"]
    todo, tag, policy = select_todo(lang, args.seed, args.pairs, tuple(args.confirm_stopped or ()), PROJECT_ROOT)
    rep["policy_now"] = policy
    rep["would_run"] = {"tag": tag, "todo": todo, **(J.gpu_command(lang, args.seed, len(todo),
                                                                    f"--pairs {','.join(map(str, todo))}"
                                                                    if args.pairs else "") if todo else {}),
                        "container_output_dir": f"{J.CONTAINER_ROOT}/{J.OUT_REL}/{lang}/seed{args.seed}/{tag}_<日時>_<ジョブ番号>"}
    rep["fake_selftest"] = fake_selftest(lang, tok, fixed_fn, norm, pairs, span_all, base)
    rep["fake_targets_and_merge"] = fake_targets_and_merge(lang, tok, fixed_fn, norm, pairs, span_all, base)
    out = os.path.join(base, "dryrun_report.json")
    J.write_json_new(rep, out)
    show = {k: rep[k] for k in ("would_run", "fake_selftest", "fake_targets_and_merge")}
    show["span_targets"] = rep["span_summary"]["targets"]
    show["span_counts"] = {k: rep["span_summary"][k] for k in ("pairs_whose_span_changes_vs_0520", "src_side_changed",
                                                               "fixed_ok", "fixed_exact", "fixed_not_ok")}
    show["audit_agreement"] = {k: rep["span_audit_agreement"][k] for k in ("n_compared", "n_disagree")}
    show["gate"] = rep["pyopenjtalk_gate_selftest"]["violations_by_scenario"]
    print(json.dumps(show, ensure_ascii=False, indent=1))
    print(f"[appB_jazh] dry run OK ({lang}); report: {os.path.relpath(out, PROJECT_ROOT)}")


# ---------------------------------------------------------------- GPU
def gpu_main(args):
    lang = args.lang
    if not J.IN_CONTAINER:
        raise SystemExit("[appB_jazh] GPU の処理は slurm_run.sh のコンテナの中（/mnt）で流す（ホストは --dry_run）")
    if os.environ.get("HF_HOME") != C.CONTAINER_HF_HOME or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit(f"[appB_jazh] HF の環境が決まりと違う: {os.environ.get('HF_HOME')}")
    pe = J.pinned_errors(lang)
    if pe:
        raise SystemExit("[appB_jazh] 固定のファイルが md5 と違う:\n  - " + "\n  - ".join(pe))
    se = G.silence_param_errors()
    if se:
        raise SystemExit("[appB_jazh] 無音の基準の数値が決まりと違う:\n  - " + "\n  - ".join(se))
    todo, tag, policy = select_todo(lang, args.seed, args.pairs, tuple(args.confirm_stopped or ()), PROJECT_ROOT)
    print("[appB_jazh] seed policy:", json.dumps({k: policy[k] for k in ("lang", "seed", "tag", "todo", "confirm_stopped")},
                                                  ensure_ascii=False), flush=True)
    if not todo:
        print("[appB_jazh] 流す対が無い（すべて決着済み、または動いている／止まった実行の確認待ち）。終わる。", flush=True)
        return
    st, est = J.slurm_time_for(lang, len(todo))
    print(f"[appB_jazh] {len(todo)} pairs; estimate {est} min; recommended SLURM_TIME {st}", flush=True)

    gate = J.PyopenjtalkGate(J.VENDOR_DE)
    gate.install()  # これ以後、pyopenjtalk の import は vendor の duration_estimator.py からだけ
    import torch
    assert torch.cuda.is_available(), "GPU で実行する（slurm_run.sh 経由）"
    torch.set_num_threads(J.THREADS)
    W = C.load_shim_module()
    env_info = C.setup_offline(W)  # 事前確認と本番 main() の両方に効く tokenizer の読み替え
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_ID)
    fixed_fn, legacy_fn = C.span_functions()
    norm = J.vendor_normalizer()
    pairs = J.load_pairs(lang)
    span_pre = J.span_table(tok, [pairs[i] for i in todo], lang, fixed_fn, legacy_fn, norm)
    bad = [i for i in todo if not span_pre[i]["fixed_ok"] or not span_pre[i]["changed_vs_0520"]]
    if bad:
        raise SystemExit(f"[appB_jazh] 修正後の範囲が実体語を覆わない、または 5/19-20 と範囲が変わらない対: {bad}")

    # 本番スクリプトを module として読む（本番と同じ sys.path の並び）。このスクリプトの置き場と scripts/regen2 は外す
    drop = {os.path.realpath(_HERE), os.path.realpath(C.P("scripts/regen2"))}
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.getcwd()) not in drop]
    sys.path.insert(0, os.path.dirname(EVAL_SCRIPT))
    import eval_crossattn_entity_dtw as M
    import encoder_injection_t5gemma as EI  # vendor の duration_estimator（と pyopenjtalk）がここで読まれる
    assert os.path.realpath(M.__file__) == os.path.realpath(EVAL_SCRIPT), M.__file__
    assert os.path.realpath(EI.__file__) == os.path.realpath(INJ_SCRIPT), EI.__file__
    assert os.path.realpath(M.T5GEMMA_ROOT) == os.path.realpath(J.VENDOR_ROOT), M.T5GEMMA_ROOT
    DE = sys.modules["duration_estimator"]
    gate_rep = gate.finalize(DE)
    print(f"[appB_jazh] pyopenjtalk gate: uses={gate_rep['vendor_duration_uses_pyopenjtalk']} "
          f"file={gate_rep['pyopenjtalk_file']} patched={gate_rep['patched_functions']} "
          f"violations={gate_rep['violations']}", flush=True)
    if gate_rep["violations"]:
        raise SystemExit(f"[appB_jazh] pyopenjtalk の決まりの違反（モデルを読む前に止める）: {gate_rep['violations']}")
    dur = J.duration_preflight(DE, lang, span_pre[todo[0]]["a"]["text_norm"])
    vendor_md5 = {}
    for name, (path, want) in J.VENDOR_PINNED.items():
        f = os.path.realpath(sys.modules[name].__file__)
        if f != os.path.realpath(path) or C.md5_file(f) != want:
            raise SystemExit(f"[appB_jazh] vendor {name} が固定のファイルでない: {f}")
        vendor_md5[name] = {"file": f, "md5": C.md5_file(f)}
    FA = sys.modules["forced_alignment_utils"]
    assert M.WhisperAligner is FA.WhisperAligner
    env_info.update({"vendor_md5": vendor_md5, "duration_preflight": dur,
                     "pyopenjtalk_calls_in_preflight": dict(gate.calls_from_vendor),
                     "tokenizer_for_preflight": {"class": type(tok).__name__, "name_or_path": tok.name_or_path}})
    run_one(lang, M, EI, FA, seed=args.seed, tag=tag, todo=todo, pairs=pairs, span_pre=span_pre,
            data_root=PROJECT_ROOT, mode=J.MODE_GPU, env_info=env_info, policy=policy, on_gpu=True, gate=gate)
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(J.LANGS))
    ap.add_argument("--seed", type=int, required=True, choices=list(J.SEEDS))
    ap.add_argument("--pairs", default=None, help="対象の対の番号をカンマで（既定は流せる対象の対すべて）")
    ap.add_argument("--confirm_stopped", action="append",
                    help="最後まで終わっていない実行の置き場（squeue で動いていないことを確かめたもの）")
    ap.add_argument("--dry_run", action="store_true", help="CPU の試運転（モデルは読まない）")
    args = ap.parse_args(argv)
    if args.dry_run:
        dry_run(args)
    else:
        gpu_main(args)


if __name__ == "__main__":
    main()
