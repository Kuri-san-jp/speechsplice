#!/usr/bin/env python3
"""複数 entity の同時注入が成立したかを、書き起こしで確かめる (GPU)。

生成は済んでいて品質と長さも測ってあるが、**注入した読みが実際に出たか**の
確認だけが無い。査読者 5eZ1 と GAi9 が求めているのはそこなので、4 条件の
音声を書き起こして、枠ごとに語が出ているかを数える。

4 条件と期待される語 (枠 1 の語を A1/B1、枠 2 の語を A2/B2 とする):

    bl     A1 と A2   どちらも元のまま
    inj1   B1 と A2   枠 1 だけ差し替え
    inj2   A1 と B2   枠 2 だけ差し替え
    inj12  B1 と B2   両方差し替え   ← 同時注入が成立したかの本体

語の出現判定は既存の規則をそのまま使う
(scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py:95-98)。
日本語と中国語は空白を除いて部分一致、英語は小文字化して部分一致。

実行例:
  SLURM_TIME=02:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/analysis/score_multientity.py \\
      --dir /mnt/eval_results/multientity_en_300 --lang en
"""
import argparse
import json
import math
import os
import sys

ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".cache", "huggingface", "hub")

# **スレッド上限。torch を読む前に置く必要がある。**
# 2026-09-18、この種のジョブを並べたらログインノードの CPU を埋めて
# 他の利用者に迷惑をかけた（note/2026-09-18_gpuoverrun_debug.md）。
# **上限は 128 スレッドの 1/4 = 32**（2026-09-18 ユーザ指示）。
# GPU を 3 枚まで同時に使うので、1 本あたり 10 にして合計 30 に収める。
# 実測では GPU 律速で 1 コアしか使わないが、上限として置く。
_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

COND_EXPECT = {
    # 条件 -> (枠1で期待する語の出どころ, 枠2で期待する語の出どころ)
    "bl": ("a", "a"),
    "inj1": ("b", "a"),
    "inj2": ("a", "b"),
    "inj12": ("b", "b"),
}


def in_text(name, transcript, lang):
    if not name or not transcript:
        return False
    if lang in ("ja", "zh"):
        return name in transcript.replace(" ", "")
    return name.lower() in transcript.lower()


def wilson(k, n, z=1.96):
    """二項比率の Wilson 区間の半幅 (近似でなく Wilson を使う)。"""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="multientity_*_300 のディレクトリ")
    ap.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    ap.add_argument("--asr_model", default="openai/whisper-large-v3")
    ap.add_argument("--n", type=int, default=0, help="0 なら全件")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    res = json.load(open(os.path.join(args.dir, "results.json")))
    rows = res["rows"]

    # **移植側の runner は words_a / words_b を行に書かないことがある。**
    # 2026-09-18、LongCat の本番 (multientity_longcat_en_300) がそれで、
    # 行のキーは conds / pair_idx / spans 由来の情報だけだった。
    # その場合だけ刺激モジュールの spans から補う。
    # T5Gemma の既存 3 セルは行に words_a/words_b を持つので**この経路を通らず、
    # 既存の採点結果は 1 文字も変わらない。**
    if rows and "words_a" not in rows[0]:
        mod = res.get("summary", {}).get("pairs_module") or \
            "%s_multientity_300" % args.lang
        sys.path.insert(0, os.path.join(ROOT, "data"))
        import importlib
        items = {it["pair_idx"]: it for it in importlib.import_module(mod).build_pairs()}
        n_fill = 0
        for r in rows:
            it = items.get(r["pair_idx"])
            if it is None:
                raise SystemExit("刺激に pair_idx %s が無い" % r["pair_idx"])
            sp = sorted(it["spans"], key=lambda x: x["slot"])
            r["words_a"] = [x["word_a"] for x in sp]
            r["words_b"] = [x["word_b"] for x in sp]
            r.setdefault("template_id", it.get("template_id"))
            n_fill += 1
        print("[data] words_a/words_b を刺激 %s から補った: %d 行"
              % (mod, n_fill), flush=True)

    if args.n:
        rows = rows[:args.n]
    print(f"[data] {len(rows)} 件 / lang={args.lang} / dir={args.dir}", flush=True)

    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    print(f"[asr] {args.asr_model} device={'cuda' if dev == 0 else 'cpu'}", flush=True)
    asr = pipeline("automatic-speech-recognition", model=args.asr_model,
                   device=dev, torch_dtype=torch.float16 if dev == 0 else torch.float32,
                   chunk_length_s=30)
    gen_kwargs = {"language": args.lang, "task": "transcribe"}

    out_rows = []
    n_err = 0
    for i, r in enumerate(rows):
        a1, a2 = (r["words_a"] + [None, None])[:2]
        b1, b2 = (r["words_b"] + [None, None])[:2]
        pick = {"a": (a1, a2), "b": (b1, b2)}
        rec = {"pair_idx": r["pair_idx"], "words_a": [a1, a2], "words_b": [b1, b2],
               "template_id": r.get("template_id"), "mode": r.get("mode")}
        for cond, (s1, s2) in COND_EXPECT.items():
            c = (r.get("conds") or {}).get(cond)
            if not c:
                continue
            wav = os.path.join(args.dir, c["wav"])
            if not os.path.exists(wav):
                rec[f"{cond}_err"] = "wav が無い"
                n_err += 1
                continue
            try:
                txt = asr(wav, generate_kwargs=gen_kwargs)["text"]
            except Exception as e:
                rec[f"{cond}_err"] = f"{type(e).__name__}: {e}"[:160]
                n_err += 1
                continue
            w1, w2 = pick[s1][0], pick[s2][1]
            rec[f"{cond}_transcript"] = txt
            rec[f"{cond}_slot1_ok"] = in_text(w1, txt, args.lang)
            rec[f"{cond}_slot2_ok"] = in_text(w2, txt, args.lang)
            rec[f"{cond}_both_ok"] = bool(rec[f"{cond}_slot1_ok"]
                                          and rec[f"{cond}_slot2_ok"])
            # 差し替えた枠に元の語が残っていないか (取り違えの検出)
            if s1 == "b":
                rec[f"{cond}_slot1_stale"] = in_text(a1, txt, args.lang)
            if s2 == "b":
                rec[f"{cond}_slot2_stale"] = in_text(a2, txt, args.lang)
        out_rows.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"  [{i + 1}/{len(rows)}] err={n_err}", flush=True)

    summary = {"dir": args.dir, "lang": args.lang, "asr_model": args.asr_model,
               "n_attempted": len(rows), "n_error": n_err}
    for cond in COND_EXPECT:
        for key in ("slot1_ok", "slot2_ok", "both_ok"):
            vals = [r.get(f"{cond}_{key}") for r in out_rows]
            vals = [v for v in vals if v is not None]
            k, n = sum(1 for v in vals if v), len(vals)
            if n:
                p, h = wilson(k, n)
                summary[f"{cond}_{key}"] = {"k": k, "n": n, "rate": round(k / n, 4),
                                            "wilson": round(p, 4), "half": round(h, 4)}
        for key in ("slot1_stale", "slot2_stale"):
            vals = [r.get(f"{cond}_{key}") for r in out_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                k = sum(1 for v in vals if v)
                summary[f"{cond}_{key}"] = {"k": k, "n": len(vals),
                                            "rate": round(k / len(vals), 4)}

    out = args.output_json or os.path.join(args.dir, "multientity_asr_scores.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "rows": out_rows}, f,
                  ensure_ascii=False, indent=2)
    os.replace(tmp, out)

    print("\n=== 枠ごとの成立率 (Wilson 95%) ===")
    print("%-6s %-24s %-24s %s" % ("条件", "枠1", "枠2", "両方"))
    for cond in ("bl", "inj1", "inj2", "inj12"):
        def fmt(key):
            s = summary.get(f"{cond}_{key}")
            return "—" if not s else "%.1f%% ±%.1f (%d/%d)" % (
                100 * s["rate"], 100 * s["half"], s["k"], s["n"])
        print("%-6s %-24s %-24s %s" % (cond, fmt("slot1_ok"), fmt("slot2_ok"),
                                       fmt("both_ok")))
    print("\n差し替えた枠に元の語が残った割合:")
    for cond in ("inj1", "inj2", "inj12"):
        for key in ("slot1_stale", "slot2_stale"):
            s = summary.get(f"{cond}_{key}")
            if s:
                print("  %-6s %-12s %.1f%% (%d/%d)"
                      % (cond, key, 100 * s["rate"], s["k"], s["n"]))
    print("\n書き出し:", out)


if __name__ == "__main__":
    main()
