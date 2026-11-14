#!/usr/bin/env python3
"""複数 entity の**目標文だけ**を合成する経路（tgt1 / tgt2）の共通部品。

## なぜ要るか

表 3 の `base` 列は「基線の音声に**元の語**が出るか」、`1 entity` / `2 entities`
列は「注入後の音声に**差し替え後の語**が出るか」を測っている。測る語が違うので
そのままでは比べられない。そこで**差し替え後の文をそのまま合成した音声**
（注入なし = 天井）で同じ語を測る。天井の判定は
`scripts/analysis/score_multientity_ceiling.py` が行い、そこが読む音声の名前は
同ファイルの `SLOTS = {1: "tgt1", 2: "tgt2"}`、つまり

    <dir>/s{pair_idx:04d}_tgt1.wav   text_refs[0] を合成したもの（枠 1 だけ差し替えた文）
    <dir>/s{pair_idx:04d}_tgt2.wav   text_refs[1] を合成したもの（枠 2 だけ差し替えた文）

である。**この名前は変えてはならない。**

枠ごとに 1 本ずつ作る（両方差し替えた文を 1 本だけ作らない）理由は
`scripts/csm/gen_multientity_targets_csm.py` の冒頭と同じ。基線との差を 1 箇所に
保っておくと、本体セルと同じ採点器がそのまま効く。

## 何を出すか

    <out_dir>/s{pair_idx:04d}_tgt1.wav
    <out_dir>/s{pair_idx:04d}_tgt2.wav
    <out_dir>/checkpoint_tgt.json   進み具合（**既存の checkpoint.json には触らない**）
    <out_dir>/results_tgt.json      まとめ（**既存の results.json には触らない**）

既にある wav は作り直さない。したがって既存の bl / inj1 / inj2 / inj12 と
results.json / checkpoint.json は 1 バイトも動かない。

## 使い方（呼び出し側の生成器）

    sys.path.insert(0, os.path.join(ROOT, "scripts", "multientity"))
    from tgt_only import TGT_SLOT_CONDS, dry_run_report, run_targets

    if args.dry_run:
        dry_run_report(items, out_dir, tag="speecht5")
        return
    if args.tgt_only:
        def synth(text, path):
            audio = synth_plain(..., text, ...)
            save_wav(audio, path)
        run_targets(items, out_dir, synth, tag="speecht5")
        return

`items` は build_pairs() の要素（`pair_idx` / `text_refs` / `spans` を持つ）。
注入しないので実体語の区間は要らない。**区間の解決も往復確認もしない。**
（区間の検査を通すと、助詞や接尾辞が癒着する対で落ちて目標音声が作られず、
n=300 を割る。T5Gemma 日本語で 15 対、中国語で 30 対が落ちた前例がある。）
"""
import json
import os
import re
import time

# 条件名 -> text_refs の添字。名前は score_multientity_ceiling.py:SLOTS と同じ。
TGT_SLOT_CONDS = {"tgt1": 0, "tgt2": 1}
# 生成順を固定する（辞書順に頼らない）。
CONDITIONS_TGT = ["tgt1", "tgt2"]
ROW_MODE_TGT = "tgt_only"


def wav_name(pair_idx, cond):
    """score_multientity_ceiling.py:127 と同じ綴り。"""
    return "s%04d_%s.wav" % (int(pair_idx), cond)


def _words_b(item):
    sp = sorted(item.get("spans", []), key=lambda x: x.get("slot", 0))
    return [x.get("word_b") for x in sp]


def plan(items, out_dir):
    """「どの文を、どの名前で出すつもりか」を並べる（モデルを読まない）。"""
    out = []
    wb_all = [_words_b(it) for it in items]
    for it, wb in zip(items, wb_all):
        for cond in CONDITIONS_TGT:
            slot = TGT_SLOT_CONDS[cond]
            refs = it.get("text_refs") or []
            name = wav_name(it["pair_idx"], cond)
            out.append({
                "pair_idx": int(it["pair_idx"]),
                "cond": cond,
                "slot": slot,
                "wav": name,
                "wav_path": os.path.join(out_dir, name),
                "text": refs[slot] if slot < len(refs) else None,
                "word_b": wb[slot] if slot < len(wb) else None,
                "exists": os.path.exists(os.path.join(out_dir, name)),
            })
    return out


def dry_run_report(items, out_dir, tag="", limit=10, verify_slots=True):
    """乾式。合成せずに、出すつもりの文とファイル名を並べて返す。"""
    p = plan(items, out_dir)
    missing_text = [r for r in p if not r["text"]]
    print("[dry] tag=%s out_dir=%s" % (tag, out_dir), flush=True)
    print("[dry] 文 %d 本 x 条件 %d = 音声 %d 本（既にある %d / 作る %d）"
          % (len(items), len(CONDITIONS_TGT), len(p),
             sum(1 for r in p if r["exists"]),
             sum(1 for r in p if not r["exists"])), flush=True)
    for r in p[:limit * len(CONDITIONS_TGT)]:
        print("  idx=%04d %-4s %-16s word_b=%-12s exists=%d text=%s"
              % (r["pair_idx"], r["cond"], r["wav"], str(r["word_b"]),
                 int(r["exists"]), (r["text"] or "")[:60]), flush=True)
    if len(p) > limit * len(CONDITIONS_TGT):
        print("  ... 以下 %d 本省略" % (len(p) - limit * len(CONDITIONS_TGT)),
              flush=True)
    if missing_text:
        print("[dry] **text_refs が足りない対がある**: %s"
              % [r["pair_idx"] for r in missing_text][:10], flush=True)
    if verify_slots:
        print("[dry] 名前の照合: %s" % check_scorer_slots(), flush=True)
    return p


def check_scorer_slots():
    """採点器 score_multientity_ceiling.py の SLOTS と綴りが合うかを見る。

    採点器は torch / transformers を読むので import はしない。ソースから
    `SLOTS = {...}` の行だけ読む（CPU で、モデル無しで確かめられるように）。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    src = os.path.join(root, "scripts", "analysis",
                       "score_multientity_ceiling.py")
    if not os.path.isfile(src):
        return "採点器が見つからない: %s" % src
    line = None
    with open(src) as f:
        for ln in f:
            if ln.startswith("SLOTS"):
                line = ln.strip()
                break
    if line is None:
        return "採点器に SLOTS の行が無い: %s" % src
    want = sorted(TGT_SLOT_CONDS)
    got = sorted(re.findall(r'["\']([^"\']+)["\']', line))
    return ("一致 (%s)" % line if got == want
            else "**食い違い** 採点器=%s こちら=%s" % (got, want))


def run_targets(items, out_dir, synth_to_wav, tag="", extra_summary=None,
                ckpt_every=10, log_every=25, stop_on_error=False):
    """目標文だけを合成する。既にある wav は作り直さない。

    synth_to_wav(text, path) は 1 本合成して path に書く呼び出し側の関数。
    注入は一切しないので、呼び出し側は素の合成（bl 条件と同じ関数）を使う。
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint_tgt.json")
    res_path = os.path.join(out_dir, "results_tgt.json")
    n_pre = sum(1 for r in plan(items, out_dir) if r["exists"])
    print("[tgt] tag=%s out_dir=%s" % (tag, out_dir), flush=True)
    print("[tgt] 文 %d 本 / 既にある音声 %d 本 / 作る音声 %d 本"
          % (len(items), n_pre, len(items) * len(CONDITIONS_TGT) - n_pre),
          flush=True)

    rows, fail = [], []
    made = 0
    t0 = time.time()
    for j, it in enumerate(items):
        pidx = int(it["pair_idx"])
        row = {"pair_idx": pidx, "mode": ROW_MODE_TGT,
               "template_id": it.get("template_id"),
               "text_a": it.get("text_a"), "text_refs": it.get("text_refs"),
               "words_b": _words_b(it), "conds": {}}
        for cond in CONDITIONS_TGT:
            slot = TGT_SLOT_CONDS[cond]
            name = wav_name(pidx, cond)
            path = os.path.join(out_dir, name)
            text = (it.get("text_refs") or [None, None])[slot]
            if os.path.exists(path):
                # **作り直さない。**既に配った音声のバイト列を動かさないため。
                row["conds"][cond] = {"wav": name, "target_text": text,
                                      "existed": True}
                continue
            if not text:
                fail.append({"pair_idx": pidx, "cond": cond,
                             "reason": "text_refs[%d] が無い" % slot})
                continue
            try:
                info = synth_to_wav(text, path)
                made += 1
            except Exception as e:
                msg = "%s %s: %s" % (cond, type(e).__name__, e)
                print("  [%d] FAIL %s" % (pidx, msg[:200]), flush=True)
                fail.append({"pair_idx": pidx, "cond": cond,
                             "reason": msg[:300]})
                if stop_on_error:
                    raise
                continue
            rec = {"wav": name, "target_text": text, "existed": False}
            if isinstance(info, dict):
                rec.update(info)
            row["conds"][cond] = rec
        rows.append(row)
        if (j + 1) % log_every == 0:
            el = (time.time() - t0) / 60
            print("  [%d/%d] 作った音声 %d  %.1fmin" % (j + 1, len(items), made, el),
                  flush=True)
        if (j + 1) % ckpt_every == 0:
            _atomic_json({"rows": rows, "fail": fail}, ckpt_path)
    _atomic_json({"rows": rows, "fail": fail}, ckpt_path)

    after = plan(items, out_dir)
    n_now = sum(1 for r in after if r["exists"])
    summary = {"tag": tag, "out_dir": out_dir, "mode": ROW_MODE_TGT,
               "conditions": list(CONDITIONS_TGT),
               "n_items": len(items), "n_made": made,
               "n_wav_present": n_now,
               "n_wav_expected": len(items) * len(CONDITIONS_TGT),
               "n_fail": len(fail), "fail": fail,
               "elapsed_min": round((time.time() - t0) / 60, 2)}
    if extra_summary:
        summary.update(extra_summary)
    _atomic_json({"summary": summary, "rows": rows}, res_path)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    if n_now != summary["n_wav_expected"]:
        # **欠けたまま先に進ませない。**天井は n=300 で出す約束になっている。
        raise SystemExit("目標音声が %d / %d 本しか無い。天井の母数が足りない"
                         % (n_now, summary["n_wav_expected"]))
    return summary


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
