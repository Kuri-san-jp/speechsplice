#!/usr/bin/env python3
"""日本語の複数 entity を、**原稿と同じ furigana 規約**で判定し直す（GPU 1 枚）。

## なぜ要るか

原稿の Whisper-Inj は言語ごとに判定が違う（原稿 Section 4）。

    EN  Whisper large-v3、表記の部分一致
    JA  **Furigana Whisper**（Parakeet-Inc/furigana_whisper_small_jsut）で
        カナの書き起こしを取り、**読み**が含まれるかを見る
    ZH  Whisper large-v3 ＋ **拼音正規化**後に照合

ところが複数 entity の採点器（`score_multientity.py`）は 3 言語とも
large-v3 の表記一致で判定していた。**同じ論文の中で判定が揃っていない。**
中国語は `rejudge_zh_multientity_pinyin.py` で揃えた。本スクリプトは日本語を揃える。

## 判定の出どころ

手続きは `ja_n300_furigana.py:60-100` と同一:

  * Furigana Whisper で書き起こす（出力はカナ）
  * `norm_hira` でひらがなに正規化し、期待する**読み**が含まれるかを見る

読みは原稿の日本語セルの成果物 `ja_wordrepl300_gen/judge.json` の
`read_a` / `read_b` から取る。**複数 entity と同じ 94 語をすべて含む**ことを
確認済みなので、読みを新たに起こす必要は無い（pyopenjtalk は使わない）。

## 何を出すか

条件ごと・枠ごとに、期待する語の読みが書き起こしに含まれた率。

    bl     枠1・枠2 とも**元の語**の読み
    inj1   枠1 は**差し替え後**の読み、枠2 は元の語の読み
    inj2   枠1 は元の語、枠2 は差し替え後
    inj12  枠1・枠2 とも差し替え後

## 天井（ceiling）も同じ規約で採れるようにした（2026-09-22 追記）

原稿 Table 3 の天井は「差し替え後の文をそのまま合成した音声」（注入なし、
`tgt1` / `tgt2`）に同じ語を当てて測る。既にある `ceiling_asr.json` は
large-v3 の**表記一致**で採ったもので、日本語の One / Both 列（読み一致）と
物差しが違う。`--ceiling` を渡すと、同じ Furigana Whisper・同じ `norm_hira`・
同じ読みの対応表で `tgt1` / `tgt2` の音声を採り直す。

枠の対応は `score_multientity_ceiling.py:SLOTS` と同じで、

    tgt1 の音声   枠 1 に words_b[0] の読みが出るか
    tgt2 の音声   枠 2 に words_b[1] の読みが出るか

分母は行数 x 2。**既存の `ceiling_asr.json` には触らない**（別名で書き出す）。

## 使い方

  # 従来どおり（bl / inj1 / inj2 / inj12 の採り直し）
  SLURM_TIME=03:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/analysis/rejudge_ja_multientity_furigana.py \\
      --dir /mnt/eval_results/multientity_ja_300 \\
      --output_json /mnt/eval_results/multientity_ja_300/asr_furigana.json

  # 天井の採り直し（tgt1 / tgt2 の音声が要る）
  SLURM_TIME=03:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/analysis/rejudge_ja_multientity_furigana.py \\
      --dir /mnt/eval_results/multientity_ja_300 --ceiling \\
      --output_json /mnt/eval_results/multientity_ja_300/ceiling_asr_furigana.json
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

MODEL_ID = "Parakeet-Inc/furigana_whisper_small_jsut"
READ_SRC = os.path.join(ROOT, "eval_results", "ja_wordrepl300_gen", "judge.json")
if not os.path.exists(READ_SRC):  # (release) the same file is shipped as the Table 1 JA judge output
    READ_SRC = os.path.join(ROOT, "eval_results", "table1_cross_lingual", "ja_judge.json")
# 条件 -> (枠1で期待する語の出どころ, 枠2で期待する語の出どころ)
# score_multientity.py:50-56 と同一。
COND_EXPECT = {"bl": ("a", "a"), "inj1": ("b", "a"),
               "inj2": ("a", "b"), "inj12": ("b", "b")}
# 天井用（--ceiling）。None を置いた枠は測らない。tgt1 の音声は枠 1 しか
# 差し替わっていないので枠 2 を見る意味が無い（tgt2 はその逆）。
CEILING_EXPECT = {"tgt1": ("b", None), "tgt2": (None, "b")}


def norm_hira(s):
    """ja_n300_furigana.py:29-38 と同一。カナをひらがなに寄せ、それ以外を捨てる。"""
    out = []
    for c in s:
        o = ord(c)
        if 0x30A1 <= o <= 0x30F6:
            c = chr(o - 0x60)
            o = ord(c)
        if 0x3041 <= o <= 0x3096 or c == "ー":
            out.append(c)
    return "".join(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def load_readings():
    d = json.load(open(READ_SRC))
    m = {}
    for r in d["rows"]:
        if r.get("word_a") and r.get("read_a"):
            m[r["word_a"]] = r["read_a"]
        if r.get("word_b") and r.get("read_b"):
            m[r["word_b"]] = r["read_b"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--output_json", required=True)
    # 天井経路。旗を渡さないかぎり従来どおりの条件集合で動く。
    ap.add_argument("--ceiling", action="store_true",
                    help="tgt1/tgt2 の音声を読み判定で採り、天井を出す")
    a = ap.parse_args()

    expect = CEILING_EXPECT if a.ceiling else COND_EXPECT
    read = load_readings()
    print("[data] 読みの対応表 %d 語" % len(read), flush=True)

    res = json.load(open(os.path.join(a.dir, "results.json")))
    rows = res["rows"]
    if rows and "words_a" not in rows[0]:
        mod = res.get("summary", {}).get("pairs_module") or "ja_multientity_300"
        sys.path.insert(0, os.path.join(ROOT, "data"))
        import importlib
        items = {it["pair_idx"]: it for it in importlib.import_module(mod).build_pairs()}
        for r in rows:
            sp = sorted(items[r["pair_idx"]]["spans"], key=lambda x: x["slot"])
            r["words_a"] = [x["word_a"] for x in sp]
            r["words_b"] = [x["word_b"] for x in sp]
        print("[data] words_a/words_b を刺激 %s から補った" % mod, flush=True)
    rows = rows[:a.n]

    miss = sorted({w for r in rows for w in (r["words_a"] + r["words_b"])
                   if w not in read})
    if miss:
        raise SystemExit("読みが無い語がある: %s" % miss[:20])

    import soundfile as sf
    import torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(dev).eval()
    print("[asr] %s device=%s" % (MODEL_ID, dev), flush=True)

    @torch.no_grad()
    def transcribe(path):
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        t = torch.from_numpy(wav).float()
        if sr != 16000:
            t = torchaudio.functional.resample(t.unsqueeze(0), sr, 16000).squeeze(0)
        feats = proc(t.numpy(), sampling_rate=16000,
                     return_tensors="pt").input_features
        ids = model.generate(feats.to(dev), max_new_tokens=200)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    cnt = {(c, s): [0, 0] for c in expect for s in (1, 2)
           if expect[c][s - 1] is not None}
    out_rows = []
    n_missing = 0
    for i, r in enumerate(rows):
        wa = (r["words_a"] + [None, None])[:2]
        wb = (r["words_b"] + [None, None])[:2]
        pick = {"a": wa, "b": wb}
        rec = {"pair_idx": r["pair_idx"], "words_a": wa, "words_b": wb}
        for cond, src in expect.items():
            p = os.path.join(a.dir, "s%04d_%s.wav" % (r["pair_idx"], cond))
            if not os.path.isfile(p):
                n_missing += 1
                rec["%s_missing" % cond] = True
                continue
            tr = transcribe(p)
            hira = norm_hira(tr)
            rec["%s_tr" % cond] = tr
            for s in (1, 2):
                if src[s - 1] is None:
                    continue
                w = pick[src[s - 1]][s - 1]
                exp = norm_hira(read.get(w, ""))
                ok = bool(exp) and exp in hira
                rec["%s_slot%d_ok" % (cond, s)] = bool(ok)
                cnt[(cond, s)][0] += int(ok)
                cnt[(cond, s)][1] += 1
        out_rows.append(rec)
        if (i + 1) % 25 == 0:
            print("  [%d/%d]" % (i + 1, len(rows)), flush=True)

    def rate(c, s):
        k, n = cnt[(c, s)]
        return {"k": k, "n": n, "rate": round(100.0 * k / n, 2) if n else None,
                "half_pp": round(100 * wilson(k, n), 2) if n else None}

    summary = {"dir": a.dir, "lang": "ja", "asr_model": MODEL_ID,
               "judge": "読み（ひらがな正規化）の部分一致。"
                        "ja_n300_furigana.py と同一の手続き",
               "reading_source": READ_SRC,
               "mode": "ceiling" if a.ceiling else "conditions",
               "n_attempted": len(rows), "n_wav_missing": n_missing}
    for c in expect:
        for s in (1, 2):
            if expect[c][s - 1] is None:
                continue
            summary["%s_slot%d_ok" % (c, s)] = rate(c, s)
    if a.ceiling:
        # 枠 1・枠 2 をまとめた 1 つの値。分母は行数 x 2 で、原稿 Table 3 の
        # Ceiling 列はこの合計を使う（中国語の ceiling_ok_pinyin と同じ流儀）。
        k = sum(cnt[(c, s)][0] for (c, s) in cnt)
        n = sum(cnt[(c, s)][1] for (c, s) in cnt)
        summary["ceiling_ok_reading"] = {
            "k": k, "n": n,
            "rate": round(100.0 * k / n, 2) if n else None,
            "half_pp": round(100 * wilson(k, n), 2) if n else None}
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    if n_missing:
        bad = a.output_json + ".incomplete"
        json.dump({"summary": summary, "rows": out_rows},
                  open(bad, "w"), ensure_ascii=False, indent=1)
        raise SystemExit("音声が欠けている: %d 本（診断は %s）" % (n_missing, bad))
    json.dump({"summary": summary, "rows": out_rows},
              open(a.output_json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
