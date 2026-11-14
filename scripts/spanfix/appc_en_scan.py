#!/usr/bin/env python3
"""付録 C 英語の作り直しの状態を集計し、次に流すもの（seed 43/44・やり直し）を決める（CPU のみ, 2026-10-01）.

ホストで直接（モデルも GPU も使わない。wav は soundfile で読むだけ）:
  taskset -c 0-12 python3 scripts/spanfix/appc_en_scan.py

すること:
  * eval_results/_spanfix_appC_en/seed{42,43,44}/ の各 batch について、本番 main() の checkpoint.json の行と、
    appc_en_gen.py の記録（runs/run*/events.jsonl と wav）から、対ごとの状態を決める（appc_en_common.classify_pair）。
      valid              使える（Whisper が 5 本すべてで実体語を見つけ、5 本とも無音でなく、記録がそろっている）
      unaligned / silent / generation_failed     対として使えない → 次の seed（43、その次は 44）
      run_error / alignment_exception / capture_error / no_row_unexplained   環境の問題 → 同じ seed でやり直し
      pending            まだ（batch が途中で止まった。同じコマンドをもう一度流すと続きから）
      stop_author_decision  使われたスパンが CPU の確かめと違う等（流し直しでは直らない）
    無音は scripts/regen2/silence_criterion.py の基準で、wav ファイルから計算し直す。PER の値は見ない。
  * 同じ seed のやり直しの上限（appc_en_common.same_seed_retry_verdict）: 同じ seed の batch で環境の問題の
    同じ種類が 2 回出た対、または retry batch が 2 個になってなお環境の問題の対は stop_author_decision にする
    （付録 B の「Whisper の例外が同じ seed で 2 回続けば止める」と同じ考え方）。やり直しのコマンドは出さない。
  * seed 42 の BL・TGT が 5/10 と同じ値になった件数だけを表示する（環境の目安。表の値は出さない。選択に使わない）。
  * 1 つの対が同じ seed の複数の batch（part とやり直し）にあるときは、後で作った batch を使う。
    batch の中では、その対を最後に生成した起動の記録を使う（時間切れで checkpoint に残らなかった対は
    次の起動で同じ seed で作り直されるため。値を見て選んでいない）。
  * seed 43 の対象 = seed 42 の 6 part がすべて終わった後、seed 42 で「対として使えない」対。
    seed 44 の対象 = seed 43 の batch がすべて終わった後、seed 43 でも使えない対。44 でも使えなければ著者の判断。
  * 次に流すコマンドと SLURM_TIME（1 対約 0.9 分 + 読み込み約 3 分の 2 倍以上）を表示する。
出力: eval_results/_spanfix_appC_en/status/status_<日時>.json（新規。上書きしない）
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
import json  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "regen2"))
import appc_en_common as C  # noqa: E402

ROOT = C.PROJECT_ROOT
OWN_REL = "scripts/spanfix/appc_en_scan.py"
GEN = "/mnt/scripts/spanfix/appc_en_gen.py"


def latest_span_check(pairs):
    d = C.P(C.SPAN_CHECK_DIR_REL)
    files = sorted(f for f in os.listdir(d) if f.startswith("span_check_")) if os.path.isdir(d) else []
    if not files:
        raise SystemExit("[scan] CPU のスパンの確かめの結果が無い（appc_en_span_check.py を先に）")
    rel = f"{C.SPAN_CHECK_DIR_REL}/{files[-1]}"
    sc = C.read_json(C.P(rel))
    if not sc.get("all_ok") or sc.get("pairs_digest") != C.pairs_digest(pairs):
        raise SystemExit(f"[scan] {rel} が all_ok でないか、対が違う")
    return rel, {int(k): v for k, v in sc["fixed_spans"].items()}


def scan_seed(seed, span_ref):
    batches = C.list_batches(seed, ROOT)
    binfo = collections.OrderedDict()
    source = {}  # pair -> batch 名（後で作った batch が上書き）
    cache = {}
    for name, man in batches:
        babs = C.P(C.batch_dir_rel(seed, name))
        runs = C.list_runs(babs)
        run_events = collections.OrderedDict((os.path.basename(r), C.read_events(r)) for r in runs)
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
                       "created": man.get("created"), "basis": man.get("basis")}
        cache[name] = (man, rows, run_events, complete)
        for p in man["targets"]:
            source[p] = name
    def classify_in(p, name):
        man, rows, run_events, complete = cache[name]
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
        else:
            status, ev = C.classify_pair(row, gen, align, errs, complete, ROOT, span_ref[p])
        return status, ev, src_run, row

    # 対ごとに、同じ seed の中で入っていた batch の並び（作った順）
    in_batches = collections.defaultdict(list)
    for name, man in batches:
        for p in man["targets"]:
            in_batches[p].append(name)

    final = {}
    for p, name in sorted(source.items()):
        status, ev, src_run, row = classify_in(p, name)
        entry = {"status": status, "seed": seed, "batch": name, "run": src_run, "evidence": ev, "row": row}
        names = in_batches[p]
        if len(names) > 1:
            # 同じ seed のやり直しの上限（appc_en_common.same_seed_retry_verdict）。前の batch の状態も数える
            history = [[n, classify_in(p, n)[0]] for n in names[:-1]] + [[name, status]]
            entry["history"] = history
            reason = C.same_seed_retry_verdict(history)
            if reason:
                entry["status_before_retry_limit"] = status
                entry["status"] = C.STATUS_STOP
                entry["evidence"] = dict(ev, retry_limit=reason, history=history)
        final[p] = entry
    return binfo, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    errs = C.pinned_errors(ROOT) + C.old_result_errors(ROOT)
    if errs:
        raise SystemExit("[scan] " + " / ".join(errs))
    pairs = C.load_pairs(ROOT)
    sc_rel, span_ref = latest_span_check(pairs)

    status = {"item": "Appendix C tab:ja_crossattn_phoneme, EN rows, span-fix rerun", "producer": OWN_REL,
              "date": C.stamp(), "span_check": sc_rel, "pairs_digest": C.pairs_digest(pairs),
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

    # seed 42 の part がそろっているか
    b42 = status["batches"]["42"]
    parts42 = [C.part_name(k) for k in range(C.N_PARTS_SEED42)]
    missing42 = [n for n in parts42 if n not in b42]
    incomplete = {s: [n for n, b in status["batches"][str(s)].items() if not b["complete"]] for s in C.SEEDS}
    for n in missing42:
        status["commands"].append(
            f"SLURM_TIME=02:00:00 slurm_run.sh python3 -u {GEN} --seed 42 --part {int(n[4:])} "
            f">> eval_results/_spanfix_appC_en/log_seed42_{n}.log 2>&1")
    for s in C.SEEDS:
        for n in incomplete[s]:
            b = status["batches"][str(s)][n]
            left = b["n_targets"] - b["rows_real"]
            # 既にある batch は manifest の対象で再開する（seed 43/44 でも状態ファイルは要らない）
            if n.startswith("retry"):
                tg = C.read_json(C.P(C.batch_dir_rel(s, n) + "/manifest.json"))["targets"]
                sel = f" --retry {int(n[5:])} --pairs {','.join(str(p) for p in tg)}"
            else:
                sel = f" --part {int(n[4:])}"
            status["commands"].append(
                f"# 再開（{s}/{n}: 残り約 {left} 対）\nSLURM_TIME={C.suggest_slurm_time(min(max(left, 1), 50))} "
                f"slurm_run.sh python3 -u {GEN} --seed {s}{sel} "
                f">> eval_results/_spanfix_appC_en/log_seed{s}_{n}.log 2>&1")

    # 同じ seed のやり直しが要る対、止まる対
    for s in C.SEEDS:
        rt = sorted(p for p, v in finals[s].items() if v["status"] in C.RETRY_SAME_SEED)
        if rt:
            status["retry_needed"][str(s)] = rt
        status["stop"] += [{"seed": s, "pair_idx": p, "evidence": v["evidence"]}
                           for p, v in finals[s].items() if v["status"] == C.STATUS_STOP]

    # 次の seed の対象
    def covered(seed):
        return {p for p in finals[seed]}

    if not missing42 and not incomplete[42]:
        inv42 = sorted(p for p, v in finals[42].items() if v["status"] in C.PAIR_INVALID)
        status["invalid"] = {"42": inv42}
        rem43 = [p for p in inv42 if p not in covered(43)]
        status["next_targets"]["43"] = rem43
        base43 = len([n for n in status["batches"]["43"] if n.startswith("part")])
        status["next_part_base"] = {"43": base43}
        for k in range((len(rem43) + C.PART_SIZE - 1) // C.PART_SIZE):
            n = len(rem43[k * C.PART_SIZE:(k + 1) * C.PART_SIZE])
            status["commands"].append(
                f"SLURM_TIME={C.suggest_slurm_time(n)} slurm_run.sh python3 -u {GEN} "
                f"--seed 43 --part {base43 + k} --status_file /mnt/{{STATUS}} "
                f">> eval_results/_spanfix_appC_en/log_seed43_part{base43 + k:02d}.log 2>&1")
        if inv42 and not rem43 and not incomplete[43]:
            inv43 = sorted(p for p in inv42 if finals[43].get(p, {}).get("status") in C.PAIR_INVALID)
            status["invalid"]["43"] = inv43
            rem44 = [p for p in inv43 if p not in covered(44)]
            status["next_targets"]["44"] = rem44
            base44 = len([n for n in status["batches"]["44"] if n.startswith("part")])
            status["next_part_base"]["44"] = base44
            for k in range((len(rem44) + C.PART_SIZE - 1) // C.PART_SIZE):
                n = len(rem44[k * C.PART_SIZE:(k + 1) * C.PART_SIZE])
                status["commands"].append(
                    f"SLURM_TIME={C.suggest_slurm_time(n)} slurm_run.sh python3 -u {GEN} "
                    f"--seed 44 --part {base44 + k} --status_file /mnt/{{STATUS}} "
                    f">> eval_results/_spanfix_appC_en/log_seed44_part{base44 + k:02d}.log 2>&1")
            if inv43 and not rem44 and not incomplete[44]:
                status["author_decision"] = sorted(
                    p for p in inv43 if finals[44].get(p, {}).get("status") in C.PAIR_INVALID)
    for s, rt in status["retry_needed"].items():
        nret = len([n for n in status["batches"][s] if n.startswith("retry")]) + 1
        status["commands"].append(
            f"# 環境の問題のやり直し（seed {s}, {len(rt)} 対。原因を直してから）\n"
            f"SLURM_TIME={C.suggest_slurm_time(min(len(rt), 50))} slurm_run.sh python3 -u {GEN} "
            f"--seed {s} --retry {nret} --pairs {','.join(map(str, rt[:50]))} --status_file /mnt/{{STATUS}} "
            f">> eval_results/_spanfix_appC_en/log_seed{s}_retry{nret:02d}.log 2>&1")

    # 最終の解決（42 → 43 → 44 の順、有効性だけで）
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

    # BL と TGT はスパンを使わない: seed 42 で行のある対について、5/10 と同じ値が出た件数だけ（表の値は出さない）。
    # part00 の後、残りの part を流す前にここを見る（環境が 5/10 からどれだけ変わったかの目安。対の選択には使わない）
    old_rows = C.read_json(C.P(f"{C.OLD_DIR_REL}/results.json"))["rows"]
    rows42 = {p: v["row"] for p, v in finals[42].items() if v.get("row") is not None}
    rep = {"all": C.bl_tgt_reproduction(rows42, old_rows), "by_batch": {}}
    for n in status["batches"]["42"]:
        sub = {p: r for p, r in rows42.items() if finals[42][p]["batch"] == n}
        rep["by_batch"][n] = C.bl_tgt_reproduction(sub, old_rows)
    status["bl_tgt_reproduction_seed42_vs_0510"] = rep

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
    if missing42:
        print(f"[scan] seed 42 の未作成の part: {missing42}")
    for k in ("invalid", "next_targets", "retry_needed"):
        if status.get(k):
            print(f"[scan] {k}: {status[k]}")
    r42 = status["bl_tgt_reproduction_seed42_vs_0510"]
    if r42["all"]["n"]:
        a = r42["all"]
        print(f"[scan] seed 42 の BL・TGT が 5/10 と同じ件数（環境の目安。表の値ではない）: 行のある対 {a['n']} のうち "
              f"BL の PER 同じ {a['bl_per_same']}、BL の音素数 同じ {a['bl_ipa_len_same']}、"
              f"TGT の音素数 同じ {a['tgt_ipa_len_same']}、3 つとも同じ {a['all_three_same']}")
        for n, b in r42["by_batch"].items():
            if b["n"]:
                print(f"[scan]   {n}: 3 つとも同じ {b['all_three_same']}/{b['n']}")
    if status["stop"]:
        print(f"[scan] !!! STOP（著者の判断）: {[(x['seed'], x['pair_idx']) for x in status['stop']]}")
        for x in status["stop"]:
            if x["evidence"].get("retry_limit"):
                print(f"[scan]   seed {x['seed']} 対 {x['pair_idx']}: {x['evidence']['retry_limit']} "
                      f"{x['evidence'].get('history')}")
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
