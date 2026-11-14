#!/usr/bin/env python3
"""複数 entity の**天井**を測る（GPU 1 枚）。

## なぜ要るか

表 3 の `base` 列は「基線の音声に**元の語**が出るか」を測っていた。一方
`1 entity` / `2 entities` は「注入後の音声に**差し替え後の語**が出るか」である。
**測る語が違うので比較できない。**中国語では差し替え後の語の方が音声認識に
易しく、1 entity が base を上回る逆転が起きていた（77.0 -> 80.5）。

そこで**差し替え後の文をそのまま合成した音声**（注入なし、`tgt1` / `tgt2`）で
同じ語を測り、それを天井にする。3 列すべてが同じ語・同じ判定になる。

    天井      tgt1 の音声に words_b[0] が出るか / tgt2 の音声に words_b[1] が出るか
    1 entity  inj1 の音声に words_b[0] が出るか / inj2 の音声に words_b[1] が出るか
    2 entities inj12 の音声に words_b[0] と words_b[1] が出るか

判定は `score_multientity.py:60-64` の `in_text` と同一
（日中は空白を除いて部分一致、英語は小文字化して部分一致）。

## 使い方

  SLURM_TIME=02:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/analysis/score_multientity_ceiling.py \\
      --dir /mnt/eval_results/multientity_en_300 --lang en \\
      --output_json /mnt/eval_results/multientity_en_300/ceiling_asr.json
"""
import argparse
import json
import math
import os
import sys

ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else \
    os.environ.get("REPO_ROOT", "/mnt")
_hf = os.path.join(ROOT, ".cache", "huggingface")
os.environ.setdefault("HF_HOME", _hf)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_hf, "hub"))

_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

SLOTS = {1: "tgt1", 2: "tgt2"}


def in_text(name, transcript, lang):
    """score_multientity.py と同一の判定。"""
    if not name or not transcript:
        return False
    if lang in ("ja", "zh"):
        return name in transcript.replace(" ", "")
    return name.lower() in transcript.lower()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    ap.add_argument("--asr_model", default="openai/whisper-large-v3")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--output_json", required=True)
    a = ap.parse_args()

    res = json.load(open(os.path.join(a.dir, "results.json")))
    rows = res["rows"]

    # words_b を持たない runner があるので、刺激モジュールから補う
    # （score_multientity.py:93-113 と同じ手続き）。
    if rows and "words_b" not in rows[0]:
        mod = res.get("summary", {}).get("pairs_module") or \
            "%s_multientity_300" % a.lang
        sys.path.insert(0, os.path.join(ROOT, "data"))
        import importlib
        items = {it["pair_idx"]: it for it in importlib.import_module(mod).build_pairs()}
        for r in rows:
            it = items.get(r["pair_idx"])
            if it is None:
                raise SystemExit("刺激に pair_idx %s が無い" % r["pair_idx"])
            sp = sorted(it["spans"], key=lambda x: x["slot"])
            r["words_b"] = [x["word_b"] for x in sp]
        print("[data] words_b を刺激 %s から補った" % mod, flush=True)

    rows = rows[:a.n]
    print("[data] %d 件 / lang=%s / dir=%s" % (len(rows), a.lang, a.dir), flush=True)

    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=a.asr_model,
                   device=dev,
                   torch_dtype=torch.float16 if dev == 0 else torch.float32,
                   chunk_length_s=30)
    gen_kwargs = {"language": a.lang, "task": "transcribe"}

    out_rows = []
    k = n = 0
    n_missing = 0
    per_slot = {1: [0, 0], 2: [0, 0]}
    for i, r in enumerate(rows):
        wb = (r["words_b"] + [None, None])[:2]
        rec = {"pair_idx": r["pair_idx"], "words_b": wb}
        for s, cond in SLOTS.items():
            p = os.path.join(a.dir, "s%04d_%s.wav" % (r["pair_idx"], cond))
            if not os.path.isfile(p):
                n_missing += 1
                rec["%s_missing" % cond] = True
                continue
            t = asr(p, generate_kwargs=gen_kwargs)["text"].strip()
            ok = in_text(wb[s - 1], t, a.lang)
            rec["%s_transcript" % cond] = t
            rec["%s_ok" % cond] = bool(ok)
            per_slot[s][0] += int(ok)
            per_slot[s][1] += 1
            k += int(ok)
            n += 1
        out_rows.append(rec)
        if (i + 1) % 25 == 0:
            print("  [%d/%d] 天井 %d/%d" % (i + 1, len(rows), k, n), flush=True)

    # **欠けがあるときは書き出さない。**母数の足りない成果物を後続が拾うのを防ぐ
    # （2026-09-20 の不具合 1 と同じ手当て）。
    summary = {
        "dir": a.dir, "lang": a.lang, "asr_model": a.asr_model,
        "n_attempted": len(rows), "n_wav_missing": n_missing,
        "ceiling_ok": {"k": k, "n": n,
                       "rate": round(k / n, 4) if n else None,
                       "half": round(wilson(k, n)[1], 4) if n else None},
        "by_slot": {str(s): {"k": v[0], "n": v[1],
                             "rate": round(v[0] / v[1], 4) if v[1] else None}
                    for s, v in per_slot.items()},
        "note": "差し替え後の文をそのまま合成した音声で、差し替え後の語が読まれた率。"
                "1 entity / 2 entities と同じ語・同じ判定。",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    if n_missing:
        bad = a.output_json + ".incomplete"
        json.dump({"summary": summary, "rows": out_rows},
                  open(bad, "w"), ensure_ascii=False, indent=1)
        raise SystemExit("目標音声が欠けている: %d 本（診断は %s）" % (n_missing, bad))
    json.dump({"summary": summary, "rows": out_rows},
              open(a.output_json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
