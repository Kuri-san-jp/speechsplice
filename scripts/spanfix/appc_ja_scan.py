#!/usr/bin/env python3
"""付録 C 日本語の作り直しの状態を集計し、次に流すもの（seed 43/44・やり直し）を決める（CPU のみ, 2026-10-02）.

ホストで直接（モデルも GPU も使わない。wav は soundfile で読むだけ。pyopenjtalk は塞ぐ）:
  taskset -c 0-12 python3 scripts/spanfix/appc_ja_scan.py

すること（英語の版 appc_en_scan.py と同じ決まり）:
  * eval_results/_spanfix_appC_ja/seed{42,43,44}/ の各 batch について、本番 main() の checkpoint.json の行と
    appc_ja_gen.py の記録（runs/run*/events.jsonl と wav）から、対ごとの状態を決める（appc_ja_common.classify_pair）。
      valid / unaligned / silent / generation_failed / run_error / alignment_exception / capture_error /
      no_row_unexplained / pending / stop_author_decision
    無音は scripts/regen2/silence_criterion.py の基準で wav から計算し直す。PER の値は見ない。
  * 同じ seed のやり直しの上限（same_seed_retry_verdict）。後で作った batch を使う。batch の中では、その対を最後に
    生成した起動の記録を使う。
  * pyopenjtalk が vendor の長さ推定器の外で使われた（塞がれた import・呼び出し元の違い・本物への参照）起動の対は
    stop_author_decision にする。対の読み替えが効いた記録（pairs_loaded ok）の無い起動で生成された対も止める。
  * seed 42 の BL が 7/23 の付録 K の BL（eval_results/cjk_n300_rewav/s####_bl.wav, 同じ文・同じ経路・seed 42）と
    標本まで同じだった件数を表示する（環境の目安。表の値は出さない。対の選択には使わない）。part00 の後、
    残りの part を流す前にここを見る（目安: 50 対のうち 45 対以上が同じなら続ける。少なければ著者に報告）。
  * seed 43 の対象 = seed 42 の 6 part がすべて終わった後、seed 42 で「対として使えない」対。seed 44 も同様。
  * 次に流す GPU コマンドを決まった形で表示する:
      SLURM_TIME=<許された値> slurm_run.sh python3 -u /mnt/scripts/spanfix/gpu_guard.py \
        --min_free_gb 30 -- /mnt/scripts/spanfix/appc_ja_gen.py --seed S ...
出力: eval_results/_spanfix_appC_ja/status/status_<日時>.json（新規。上書きしない）
"""
import os
import sys

THREADS = 13
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = str(THREADS)
_R = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
for _k, _v in {"HF_HOME": _R + "/.cache/huggingface", "HF_HUB_CACHE": _R + "/.cache/huggingface/hub",
               "HF_HUB_OFFLINE": "1", "NUMBA_CACHE_DIR": _R + "/.cache/numba"}.items():
    os.environ[_k] = _v
if not os.path.isdir("/mnt/scripts"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.modules["pyopenjtalk"] = None

import argparse  # noqa: E402
import collections  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "regen2"))
import appc_ja_common as C  # noqa: E402

ROOT = C.PROJECT_ROOT
OWN_REL = "scripts/spanfix/appc_ja_scan.py"
GATE_MIN_IDENTICAL = 45  # part00（50 対）で BL が 7/23 と同じ対の数の目安（英語の作り直しの「45/50 以上」と同じ）


def latest_span_check(pairs):
    d = C.P(C.SPAN_CHECK_DIR_REL)
    files = sorted(f for f in os.listdir(d) if f.startswith("span_check_")) if os.path.isdir(d) else []
    if not files:
        raise SystemExit("[scan] CPU のスパンの確かめの結果が無い（appc_ja_span_check.py を先に）")
    rel = f"{C.SPAN_CHECK_DIR_REL}/{files[-1]}"
    sc = C.read_json(C.P(rel))
    if not sc.get("all_ok") or sc.get("pairs_digest") != C.pairs_digest(pairs):
        raise SystemExit(f"[scan] {rel} が all_ok でないか、対が違う")
    return rel, {int(k): {"a": v["a"], "b": v["b"]} for k, v in sc["fixed_spans"].items()}


def run_flags(evs):
    """起動ごとの印: pyopenjtalk の決まりに反した記録、対の読み替えが効いた記録."""
    ends = [e for e in evs if e.get("ev") == "main_end"]
    pj = ends[-1].get("pyopenjtalk") if ends else None
    viol = bool(pj and (pj.get("blocked_imports") or pj.get("bad_callers") or pj.get("refs_to_real_module")))
    loaded = [e for e in evs if e.get("ev") == "pairs_loaded"]
    return {"pyojt_violation": viol, "pyojt": pj, "pairs_loaded_ok": bool(loaded) and all(e.get("ok") for e in loaded),
            "n_gen": sum(1 for e in evs if e.get("ev") == "gen"),
            "n_gen_g2p_not_1": sum(1 for e in evs if e.get("ev") == "gen"
                                   and [d.get("g2p_calls") for d in e.get("durations", [])] != [1])}


def scan_seed(seed, span_ref):
    batches = C.list_batches(seed, ROOT)
    binfo = collections.OrderedDict()
    source = {}
    cache = {}
    for name, man in batches:
        babs = C.P(C.batch_dir_rel(seed, name))
        runs = C.list_runs(babs)
        run_events = collections.OrderedDict((os.path.basename(r), C.read_events(r)) for r in runs)
        flags = {rn: run_flags(evs) for rn, evs in run_events.items()}
        complete = any(e.get("ev") == "main_end" and e.get("error") is None
                       for evs in run_events.values() for e in evs)
        try:
            rows = {r["pair_idx"]: r for r in C.read_json(os.path.join(babs, "checkpoint.json")).get("rows", [])
                    if not r.get("placeholder")}
            ck_ok = True
        except Exception:  # noqa: BLE001
            rows, ck_ok = {}, False
        unattributed = sum(1 for evs in run_events.values() for e in evs
                           if e.get("ev") == "gen" and (e.get("pair_idx") is None
                                                        or str(e.get("label", "")).startswith("unknown")))
        corrupt = sum(1 for evs in run_events.values() for e in evs if e.get("ev") == "corrupt_line")
        main_errors = [e.get("error") for evs in run_events.values() for e in evs
                       if e.get("ev") == "main_end" and e.get("error")]
        binfo[name] = {"kind": man.get("kind"), "n_targets": len(man["targets"]), "complete": complete,
                       "runs": list(run_events), "rows_real": len(set(rows) & set(man["targets"])),
                       "checkpoint_readable": ck_ok, "unattributed_gen_events": unattributed,
                       "corrupt_event_lines": corrupt, "main_errors": main_errors,
                       "run_flags": {rn: {k: v for k, v in f.items() if k != "pyojt"} for rn, f in flags.items()},
                       "created": man.get("created"), "basis": man.get("basis")}
        cache[name] = (man, rows, run_events, complete, flags)
        for p in man["targets"]:
            source[p] = name

    def classify_in(p, name):
        man, rows, run_events, complete, flags = cache[name]
        src_run = None
        for rn, evs in run_events.items():
            if any(e.get("ev") == "gen" and e.get("pair_idx") == p for e in evs):
                src_run = rn
        if src_run is not None:
            evs = run_events[src_run]
            gen = [e for e in evs if e.get("ev") == "gen" and e.get("pair_idx") == p]
            align = [e for e in evs if str(e.get("ev", "")).startswith("align_") and e.get("pair_idx") == p]
            errs = [e for e in evs if e.get("ev") == "err" and e.get("pair_idx") == p]
        else:
            gen, align, errs = [], [], []
        row = rows.get(p)
        if not binfo[name]["checkpoint_readable"]:
            status, ev = C.STATUS_STOP, {"reason": "checkpoint.json が読めない"}
        elif src_run is not None and flags[src_run]["pyojt_violation"]:
            status, ev = C.STATUS_STOP, {"reason": "pyopenjtalk が vendor の長さ推定器の外で使われた起動",
                                         "pyopenjtalk": flags[src_run]["pyojt"]}
        elif src_run is not None and not flags[src_run]["pairs_loaded_ok"]:
            status, ev = C.STATUS_STOP, {"reason": "対の読み替えが効いた記録（pairs_loaded ok）の無い起動"}
        else:
            status, ev = C.classify_pair(row, gen, align, errs, complete, ROOT, span_ref[p])
        return status, ev, src_run, row, gen

    in_batches = collections.defaultdict(list)
    for name, man in batches:
        for p in man["targets"]:
            in_batches[p].append(name)

    final = {}
    for p, name in sorted(source.items()):
        status, ev, src_run, row, gen = classify_in(p, name)
        bl_wav = next((g.get("wav") for g in gen if g.get("label") == "bl" and g.get("wav")), None)
        entry = {"status": status, "seed": seed, "batch": name, "run": src_run, "evidence": ev, "row": row,
                 "bl_wav": bl_wav}
        names = in_batches[p]
        if len(names) > 1:
            history = [[n, classify_in(p, n)[0]] for n in names[:-1]] + [[name, status]]
            entry["history"] = history
            reason = C.same_seed_retry_verdict(history)
            if reason:
                entry["status_before_retry_limit"] = status
                entry["status"] = C.STATUS_STOP
                entry["evidence"] = dict(ev, retry_limit=reason, history=history)
        final[p] = entry
    return binfo, final


def bl_env_report(final42):
    """seed 42 の BL の wav が 7/23 の付録 K の BL と標本まで同じか（件数だけ）."""
    by_batch = collections.defaultdict(collections.Counter)
    detail = {}
    for p, v in final42.items():
        if not v.get("bl_wav"):
            continue
        r = C.bl_env_compare(C.P(v["bl_wav"]), p, ROOT)
        detail[str(p)] = r
        b = by_batch[v["batch"]]
        b["n"] += 1
        if not r.get("compared"):
            b["not_compared"] += 1
        elif r.get("identical"):
            b["identical"] += 1
        elif r.get("same_len"):
            b["same_len_diff"] += 1
        else:
            b["diff_len"] += 1
    tot = collections.Counter()
    for b in by_batch.values():
        tot.update(b)
    return {"all": dict(tot), "by_batch": {k: dict(v) for k, v in by_batch.items()}, "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    errs = C.pinned_errors(ROOT) + C.old_result_errors(ROOT)
    if errs:
        raise SystemExit("[scan] " + " / ".join(errs))
    pairs = C.load_pairs(ROOT)
    sc_rel, span_ref = latest_span_check(pairs)

    status = {"item": "Appendix C tab:ja_crossattn_phoneme, JA rows rerun on the Appendix K JA pairs (fixed span)",
              "producer": OWN_REL, "date": C.stamp(), "span_check": sc_rel, "pairs_digest": C.pairs_digest(pairs),
              "fingerprint": {}, "batches": {}, "final": {}, "counts": {}, "next_targets": {},
              "retry_needed": {}, "stop": [], "author_decision": [], "commands": []}
    finals = {}
    for s in C.SEEDS:
        status["fingerprint"][str(s)] = C.seed_fingerprint(s, ROOT)
        binfo, final = scan_seed(s, span_ref)
        finals[s] = final
        status["batches"][str(s)] = binfo
        status["final"][str(s)] = {str(p): v for p, v in final.items()}
        status["counts"][str(s)] = dict(collections.Counter(v["status"] for v in final.values()))

    b42 = status["batches"]["42"]
    parts42 = [C.part_name(k) for k in range(C.N_PARTS_SEED42)]
    missing42 = [n for n in parts42 if n not in b42]
    incomplete = {s: [n for n, b in status["batches"][str(s)].items() if not b["complete"]] for s in C.SEEDS}
    # part00 の後の環境の目安（BL が 7/23 と同じ）。OK になるまで残りの part は「保留」の印を付けて出す
    env = bl_env_report(finals[42])
    status["bl_env_seed42_vs_appk_0723"] = env
    gate = {}
    for n, b in env["by_batch"].items():
        if b.get("n"):
            gate[n] = {"identical": b.get("identical", 0), "n": b["n"],
                       "ok": b.get("identical", 0) >= round(GATE_MIN_IDENTICAL * b["n"] / C.PART_SIZE)}
    status["env_gate"] = gate
    p00 = b42.get("part00")
    status["part00_gate_ok"] = bool(p00 and p00["complete"] and gate.get("part00", {}).get("ok"))
    for n in missing42:
        cmd = C.format_gpu_command(C.suggest_slurm_time(C.PART_SIZE), f"--seed 42 --part {int(n[4:])}")
        if n != "part00" and not status["part00_gate_ok"]:
            cmd = ("# 保留: part00 が終わり、BL が 7/23 と同じ対が 45/50 以上（env_gate.part00.ok）になってから。"
                   "NG なら著者に報告\n" + cmd)
        status["commands"].append(cmd)
    for s in C.SEEDS:
        for n in incomplete[s]:
            b = status["batches"][str(s)][n]
            left = max(b["n_targets"] - b["rows_real"], 1)
            if n.startswith("retry"):
                tg = C.read_json(C.P(C.batch_dir_rel(s, n) + "/manifest.json"))["targets"]
                sel = f"--seed {s} --retry {int(n[5:])} --pairs {','.join(str(p) for p in tg)}"
            else:
                sel = f"--seed {s} --part {int(n[4:])}"
            status["commands"].append(f"# 再開（{s}/{n}: 残り約 {left} 対。前の起動が止まっていることを squeue で確かめてから）\n"
                                      + C.format_gpu_command(C.suggest_slurm_time(min(left, C.PART_SIZE)), sel))

    for s in C.SEEDS:
        rt = sorted(p for p, v in finals[s].items() if v["status"] in C.RETRY_SAME_SEED)
        if rt:
            status["retry_needed"][str(s)] = rt
        status["stop"] += [{"seed": s, "pair_idx": p, "evidence": v["evidence"]}
                           for p, v in finals[s].items() if v["status"] == C.STATUS_STOP]

    def covered(seed):
        return set(finals[seed])

    if not missing42 and not incomplete[42]:
        inv42 = sorted(p for p, v in finals[42].items() if v["status"] in C.PAIR_INVALID)
        status["invalid"] = {"42": inv42}
        rem43 = [p for p in inv42 if p not in covered(43)]
        status["next_targets"]["43"] = rem43
        base43 = len([n for n in status["batches"]["43"] if n.startswith("part")])
        status["next_part_base"] = {"43": base43}
        for k in range((len(rem43) + C.PART_SIZE - 1) // C.PART_SIZE):
            n = len(rem43[k * C.PART_SIZE:(k + 1) * C.PART_SIZE])
            status["commands"].append(C.format_gpu_command(
                C.suggest_slurm_time(n), f"--seed 43 --part {base43 + k} --status_file /mnt/{{STATUS}}"))
        if inv42 and not rem43 and not incomplete[43]:
            inv43 = sorted(p for p in inv42 if finals[43].get(p, {}).get("status") in C.PAIR_INVALID)
            status["invalid"]["43"] = inv43
            rem44 = [p for p in inv43 if p not in covered(44)]
            status["next_targets"]["44"] = rem44
            base44 = len([n for n in status["batches"]["44"] if n.startswith("part")])
            status["next_part_base"]["44"] = base44
            for k in range((len(rem44) + C.PART_SIZE - 1) // C.PART_SIZE):
                n = len(rem44[k * C.PART_SIZE:(k + 1) * C.PART_SIZE])
                status["commands"].append(C.format_gpu_command(
                    C.suggest_slurm_time(n), f"--seed 44 --part {base44 + k} --status_file /mnt/{{STATUS}}"))
            if inv43 and not rem44 and not incomplete[44]:
                status["author_decision"] = sorted(
                    p for p in inv43 if finals[44].get(p, {}).get("status") in C.PAIR_INVALID)
    for s, rt in status["retry_needed"].items():
        nret = len([n for n in status["batches"][s] if n.startswith("retry")]) + 1
        status["commands"].append(
            f"# 環境の問題のやり直し（seed {s}, {len(rt)} 対。原因を直してから）\n" + C.format_gpu_command(
                C.suggest_slurm_time(min(len(rt), C.PART_SIZE)),
                f"--seed {s} --retry {nret} --pairs {','.join(map(str, rt[:C.PART_SIZE]))} --status_file /mnt/{{STATUS}}"))

    resolved, unresolved = {}, {}
    for p in range(C.N_PAIRS):
        chain = []
        for s in C.SEEDS:
            v = finals[s].get(p)
            if v is None:
                chain.append([s, "not_run"])
                break
            chain.append([s, v["status"]])
            if v["status"] == C.STATUS_VALID:
                resolved[p] = s
                break
            if v["status"] not in C.PAIR_INVALID:
                break
        if p not in resolved:
            unresolved[p] = chain
    status["resolved_by_seed"] = dict(collections.Counter(resolved.values()))
    status["n_resolved"] = len(resolved)
    status["unresolved_head"] = {str(p): c for p, c in list(unresolved.items())[:30]}
    status["n_unresolved"] = len(unresolved)
    status["ready_to_aggregate"] = len(resolved) == C.N_PAIRS and not status["stop"]

    path_rel = f"{C.STATUS_DIR_REL}/status_{C.stamp()}.json"
    status["commands"] = [c.replace("{STATUS}", path_rel) for c in status["commands"]]
    C.write_json_new(status, C.P(path_rel))

    print(f"[scan] span_check={sc_rel}")
    for s in C.SEEDS:
        bs = status["batches"][str(s)]
        if bs:
            print(f"[scan] seed {s}: batches " + ", ".join(
                f"{n}({'done' if b['complete'] else 'INCOMPLETE'} rows {b['rows_real']}/{b['n_targets']}"
                f"{' unattributed ' + str(b['unattributed_gen_events']) if b['unattributed_gen_events'] else ''})"
                for n, b in bs.items()))
            print(f"[scan]   status counts: {status['counts'][str(s)]}")
            g2p_odd = sum(f["n_gen_g2p_not_1"] for b in bs.values() for f in b["run_flags"].values())
            if g2p_odd:
                print(f"[scan]   参考: 長さ推定で pyopenjtalk.g2p を 1 回使わなかった生成 {g2p_odd} 本（言語判定が ja でない等）")
    if missing42:
        print(f"[scan] seed 42 の未作成の part: {missing42}")
    for k in ("invalid", "next_targets", "retry_needed"):
        if status.get(k):
            print(f"[scan] {k}: {status[k]}")
    if env["all"].get("n"):
        a = env["all"]
        print(f"[scan] seed 42 の BL が 7/23 の付録 K の BL と標本まで同じ件数（環境の目安。表の値ではない）: "
              f"{a.get('identical', 0)}/{a['n']}（同じ長さで違う {a.get('same_len_diff', 0)}、長さが違う "
              f"{a.get('diff_len', 0)}、比べられない {a.get('not_compared', 0)}）")
        for n, g in gate.items():
            print(f"[scan]   {n}: 同じ {g['identical']}/{g['n']} → 目安 {'OK' if g['ok'] else 'NG（著者に報告）'}")
    if status["stop"]:
        print(f"[scan] !!! STOP（著者の判断）: {[(x['seed'], x['pair_idx']) for x in status['stop']]}")
        for x in status["stop"][:10]:
            print(f"[scan]   seed {x['seed']} 対 {x['pair_idx']}: "
                  f"{x['evidence'].get('retry_limit') or x['evidence'].get('reason') or x['evidence'].get('span_mismatch')}")
    if status["author_decision"]:
        print(f"[scan] seed 44 でも使えない対（著者の判断）: {status['author_decision']}")
    print(f"[scan] 解決済み {status['n_resolved']}/{C.N_PAIRS}（seed ごと {status['resolved_by_seed']}）"
          f" ready_to_aggregate={status['ready_to_aggregate']}")
    if not args.quiet:
        for c in status["commands"]:
            print(c)
    print(f"[scan] wrote {path_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
