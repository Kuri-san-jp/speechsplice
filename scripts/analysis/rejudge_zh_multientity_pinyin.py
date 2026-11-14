#!/usr/bin/env python3
"""中国語の複数 entity を、単一 entity と同じ pinyin 規約で判定し直す (CPU のみ)。

## なぜ要るか

2026-09-19、中国語の複数 entity で基線が 59.3% しかなかった
（英語 93.7 / 日本語 87.7）。書き起こしを見ると**同音異字**で落ちている。

    期待 杨 (yang)  書き起こし 羊和马是同一个项目组的成员   -> 表記不一致・読み一致
    期待 何 (he)    書き起こし 吴和和是同一个项目组的成员   -> 表記不一致・読み一致

`score_multientity.py:62-63` の `in_text` は中国語でも**表記の完全一致**で判定する。
一方、**単一 entity の中国語セルは pinyin 正規化で判定している**
（原稿 Table 1 の記述、実装は `scripts/analysis/cjk_zh_pinyin_judge.py`）。
**同じ論文の中で判定が揃っていない。**

## 何をするか

書き起こしは採点結果に保存済みなので、**ASR を回さずに判定だけやり直す。**
GPU も Whisper も要らない。

判定は `cjk_zh_pinyin_judge.py:49-71` と同じ規約:
  * pypinyin TONE3（文脈考慮、heteronym なし）で書き起こし全体の読み列を作る
  * 期待語の読み列が、その読み列の中に連続して現れるかを見る

**表記一致の結果も残す。**`_ok` を pinyin 判定に置き換えるのではなく、
`_ok_pinyin` として併記し、両方の集計を出す。どちらを採用するかは人間が決める。

日本語・英語には一切触れない。中国語のファイルだけを読む。

## 天井 (ceiling) も同じ規約で採れるようにした (2026-09-22 追記)

原稿 Table 3 の中国語の天井は ceiling_asr.json の書き起こしに pinyin 判定を
当てた値だが、その計算をしたスクリプトが残っていなかった。
新しい系の天井を足すときに「既存セルと同じ規約」であることを示せるよう、
同じ判定関数 (py_seq / contains) を ceiling_asr.json にも当てられるようにした。
ceiling_asr.json の 1 行は tgt1 / tgt2 の 2 枠ぶんの書き起こしを持つので、
枠の対応は tgt1 -> 枠 1、tgt2 -> 枠 2、分母は行数 x 2 になる。

Usage:
  # 従来どおり (bl / inj1 / inj2 / inj12 の採り直し)
  taskset -c 0-31 python3 scripts/analysis/rejudge_zh_multientity_pinyin.py \\
    --scores eval_results/multientity_zh_300/multientity_asr_scores.json

  # 天井の採り直し (--scores とは排他)
  taskset -c 0-12 python3 scripts/analysis/rejudge_zh_multientity_pinyin.py \\
    --ceiling eval_results/multientity_zh_v2_300/ceiling_asr.json
"""
import argparse
import json
import math
import os
import re

from pypinyin import Style, pinyin

HAN = re.compile(r"[一-鿿]")
CONDS = ("bl", "inj1", "inj2", "inj12")
# 条件ごとに、どちらのスロットが差し替わっているか（score_multientity.py と同じ）
COND_EXPECT = {"bl": ("a", "a"), "inj1": ("b", "a"),
               "inj2": ("a", "b"), "inj12": ("b", "b")}
# ceiling_asr.json 用。枠 1 は words_b[0]、枠 2 は words_b[1] を読み上げたか見る
CEILING_SLOTS = (("tgt1", 0), ("tgt2", 1))


def py_seq(text):
    """漢字だけ取り出して TONE3 の読み列にする。cjk_zh_pinyin_judge.py:49-53 と同じ。"""
    chars = [c for c in text if HAN.match(c)]
    if not chars:
        return []
    return [p[0] for p in pinyin("".join(chars), style=Style.TONE3,
                                 heteronym=False)]


def strip_tone(s):
    """声調の数字を落とす。同 45-47 行と同じ。"""
    return re.sub(r"\d", "", s)


def contains(seq, sub, toneless=True):
    """読み列 seq の中に sub が連続して現れるか。"""
    if not sub:
        return False
    a = [strip_tone(x) for x in seq] if toneless else list(seq)
    b = [strip_tone(x) for x in sub] if toneless else list(sub)
    n, m = len(a), len(b)
    return any(a[i:i + m] == b for i in range(n - m + 1)) if m and n >= m else False


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, h


def mcnemar_exact(b, c):
    """基線のみ正解 b 件、対照のみ正解 c 件。両側 p 値。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def rejudge_ceiling(path, out):
    """ceiling_asr.json の tgt1/tgt2 の書き起こしに pinyin 判定を当てる。

    表記一致の tgt1_ok / tgt2_ok はそのまま残し、pinyin 判定は
    tgt1_ok_pinyin / tgt2_ok_pinyin として併記する（本体と同じ方針）。
    """
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    print("[data] %d 行 / %s" % (len(rows), path))

    n_missing = 0
    k_s = {1: 0, 2: 0}
    k_p = {1: 0, 2: 0}
    n_slot = {1: 0, 2: 0}
    for r in rows:
        wb = r.get("words_b") or []
        for key, idx in CEILING_SLOTS:
            slot = idx + 1
            tr = r.get("%s_transcript" % key)
            if tr is None:
                n_missing += 1
                continue
            w = (list(wb) + [None, None])[idx]
            seq = py_seq(tr.replace(" ", ""))
            ok = contains(seq, py_seq(w)) if w else False
            r["%s_ok_pinyin" % key] = ok
            n_slot[slot] += 1
            k_p[slot] += 1 if ok else 0
            k_s[slot] += 1 if r.get("%s_ok" % key) else 0
    if n_missing:
        print("[warn] 書き起こしが無い枠 %d 件" % n_missing)

    n = n_slot[1] + n_slot[2]
    ks = k_s[1] + k_s[2]
    kp = k_p[1] + k_p[2]
    _, h_s = wilson(ks, n)
    _, h_p = wilson(kp, n)
    print()
    print("%-10s %-24s %-24s" % ("枠", "表記一致（元の tgt_ok）", "pinyin（単一 entity と同じ）"))
    for slot in (1, 2):
        m = n_slot[slot] or 1
        print("%-10s %6.2f%% (%3d/%d)            %6.2f%% (%3d/%d)"
              % ("枠 %d" % slot, 100 * k_s[slot] / m, k_s[slot], n_slot[slot],
                 100 * k_p[slot] / m, k_p[slot], n_slot[slot]))
    print("%-10s %6.2f%% +-%.2f (%3d/%d)   %6.2f%% +-%.2f (%3d/%d)"
          % ("合計", 100 * ks / n, 100 * h_s, ks, n,
             100 * kp / n, 100 * h_p, kp, n))

    out = out or path.replace(".json", "_pinyin.json")
    json.dump({"summary": {"source": path, "judge": "pinyin TONE3 toneless",
                           "note": "表記一致の結果も tgt1_ok / tgt2_ok として残してある",
                           "ceiling_ok_surface": {"k": ks, "n": n,
                                                  "rate": round(ks / n, 4),
                                                  "half": round(h_s, 4)},
                           "ceiling_ok_pinyin": {"k": kp, "n": n,
                                                 "rate": round(kp / n, 4),
                                                 "half": round(h_p, 4)},
                           "by_slot": {str(s): {"surface_k": k_s[s],
                                                "pinyin_k": k_p[s],
                                                "n": n_slot[s]}
                                       for s in (1, 2)},
                           "source_summary": d.get("summary")},
               "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n書き出し:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=None,
                    help="multientity_asr_scores.json（中国語）")
    # 天井用。--scores と排他にして、既存の呼び出し（--scores のみ）の挙動を変えない
    ap.add_argument("--ceiling", default=None,
                    help="ceiling_asr.json（中国語）。tgt1/tgt2 を枠 1/枠 2 として判定")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if bool(a.scores) == bool(a.ceiling):
        ap.error("--scores か --ceiling のどちらか一方を指定すること")
    if a.ceiling:
        return rejudge_ceiling(a.ceiling, a.out)

    d = json.load(open(a.scores, encoding="utf-8"))
    rows = d["rows"]
    print("[data] %d 行 / %s" % (len(rows), a.scores))

    n_missing = 0
    for r in rows:
        w = {"a": r.get("words_a") or [], "b": r.get("words_b") or []}
        for cond, (s1, s2) in COND_EXPECT.items():
            tr = r.get("%s_transcript" % cond)
            if tr is None:
                n_missing += 1
                continue
            seq = py_seq(tr.replace(" ", ""))
            w1 = (w[s1] + [None, None])[0]
            w2 = (w[s2] + [None, None])[1]
            ok1 = contains(seq, py_seq(w1)) if w1 else False
            ok2 = contains(seq, py_seq(w2)) if w2 else False
            r["%s_slot1_ok_pinyin" % cond] = ok1
            r["%s_slot2_ok_pinyin" % cond] = ok2
            r["%s_both_ok_pinyin" % cond] = bool(ok1 and ok2)
            # 元の語が残っていないか（差し替えた枠だけ）
            for slot, src, idx in ((1, s1, 0), (2, s2, 1)):
                if src != "b":
                    continue
                old = (w["a"] + [None, None])[idx]
                if old:
                    r["%s_slot%d_stale_pinyin" % (cond, slot)] = \
                        contains(seq, py_seq(old))
    if n_missing:
        print("[warn] 書き起こしが無い条件 %d 件" % n_missing)

    print()
    print("%-10s %-22s %-22s" % ("条件", "表記一致（今の判定）", "pinyin（単一 entity と同じ）"))
    summ = {}
    for cond in CONDS:
        k_s = sum(1 for r in rows if r.get("%s_both_ok" % cond))
        k_p = sum(1 for r in rows if r.get("%s_both_ok_pinyin" % cond))
        n = len(rows)
        _, h_s = wilson(k_s, n)
        _, h_p = wilson(k_p, n)
        print("%-10s %6.1f%% +-%.1f (%3d/%d)   %6.1f%% +-%.1f (%3d/%d)"
              % (cond, 100 * k_s / n, 100 * h_s, k_s, n,
                 100 * k_p / n, 100 * h_p, k_p, n))
        summ[cond] = {"surface_k": k_s, "pinyin_k": k_p, "n": n}

    print()
    base_s = {r["pair_idx"]: r.get("bl_both_ok") for r in rows}
    base_p = {r["pair_idx"]: r.get("bl_both_ok_pinyin") for r in rows}
    print("McNemar 厳密検定（基線 対 各条件、両側）")
    for cond in ("inj1", "inj2", "inj12"):
        line = []
        for lab, base, key in (("表記", base_s, "%s_both_ok" % cond),
                               ("pinyin", base_p, "%s_both_ok_pinyin" % cond)):
            pr = [(base[r["pair_idx"]], r.get(key)) for r in rows
                  if base.get(r["pair_idx"]) is not None and r.get(key) is not None]
            b = sum(1 for x, y in pr if x and not y)
            c = sum(1 for x, y in pr if y and not x)
            line.append("%s p=%.4f (b=%d c=%d)" % (lab, mcnemar_exact(b, c), b, c))
        print("  %-6s %s" % (cond, "   ".join(line)))
        summ[cond]["mcnemar"] = line

    st_s = sum(1 for r in rows for k, v in r.items()
               if k.endswith("_stale") and v)
    st_p = sum(1 for r in rows for k, v in r.items()
               if k.endswith("_stale_pinyin") and v)
    print()
    print("差し替えた枠に元の語が残った数: 表記 %d / pinyin %d" % (st_s, st_p))

    out = a.out or a.scores.replace(".json", "_pinyin.json")
    json.dump({"summary": {"source": a.scores, "judge": "pinyin TONE3 toneless",
                           "note": "表記一致の結果も _ok として残してある",
                           "counts": summ, "stale_surface": st_s,
                           "stale_pinyin": st_p},
               "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n書き出し:", out)


if __name__ == "__main__":
    main()
