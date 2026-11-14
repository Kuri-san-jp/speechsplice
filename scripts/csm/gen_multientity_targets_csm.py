#!/usr/bin/env python3
"""複数 entity の**目標文だけ**を合成する（CSM-1B、GPU 1 枚）。

## なぜ要るか

表 1・表 2 は PIB（目標の発音へどれだけ近づいたか）で測っているが、
複数 entity だけ音声認識の読み取り率で、**指標が違う**。揃えるには
目標を基準にした実体語 PER が要り、そのために目標文の音声が要る。
既存の複数 entity の出力は bl / inj1 / inj2 / inj12 の 4 本だけである。

## 何を作るか

    <out_dir>/s{idx:04d}_tgt1.wav = TTS(text_refs[0])   枠 1 だけ差し替えた文
    <out_dir>/s{idx:04d}_tgt2.wav = TTS(text_refs[1])   枠 2 だけ差し替えた文

**枠ごとに作る。**2 枠とも差し替えた文を 1 本だけ作る案は却下した。基線との
差が 2 箇所になり、音素列の差分整列の結合規則（間の一致が 3 個以下なら同じ塊）
のせいで、隣り合う 2 枠（300 対中 50 対は間の語数が 0）で塊が融合する。
1 枠だけ差し替えた文なら差が 1 箇所で、本体セルと同じ採点器がそのまま効く。

## 台帳には触れない

本番の `run_multientity_csm.py` は `run_csm_eval.py` の台帳（wav 名の正規表現・
schema・突き合わせ）を条件名 4 つ前提で流用している。そこへ条件を足すと
台帳側の検査に当たるので、**台帳を通さない別の経路**にした。
読むのは `<dir>/results.json` の `text_refs` だけで、既存の wav も
results.json も書き換えない。既にある目標音声は飛ばすので再開できる。

## 本体と同じ作り方に揃える

差し替えが無いので、本番の bl 条件（`run_multientity_csm.py:1236`）と
同じ関数・同じ引数で、渡す文だけが違う:

    C.synthesize(bundle, text, ref=ref, speaker=..., gen=gen)

参照音声・話者・seed は results.json の summary から読み、本番と揃える。

## 使い方

  SLURM_TIME=10:00:00 slurm_run.sh \\
    python3 -u /mnt/scripts/csm/gen_multientity_targets_csm.py \\
      --dir /mnt/eval_results/csm_multientity_en_300
"""
import argparse
import json
import os
import sys
import time

_hf = "/mnt/.cache/huggingface" if os.path.isdir("/mnt/scripts") else \
    os.path.join(os.environ.get("REPO_ROOT", "/mnt"), ".cache/huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")

_thr = os.environ.get("GEN_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else \
    os.environ.get("REPO_ROOT", "/mnt")
for _p in (os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "scripts", "csm"),
           os.path.join(PROJECT_ROOT, "data")):
    sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)

import csm_splice as C  # noqa: E402

CONDS = {"tgt1": 0, "tgt2": 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="複数 entity の出力先。results.json を読み、同じ所に書く")
    p.add_argument("--n", type=int, default=0, help="0 なら全件")
    p.add_argument("--device", default=None)
    # **本番と同じ float32 にする（2026-09-20 修正）。**csm_splice.load_csm の
    # 既定は bfloat16 だが、複数 entity の本番は --dtype float32 で回している
    # （run_multientity_csm.py:939 の既定）。bfloat16 で作ると
    #   RuntimeError: Input type (float) and bias type (c10::BFloat16)
    # で落ちるうえ、通っていたとしても**本番と違う精度の目標音声**になっていた。
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    a = p.parse_args()

    res = os.path.join(a.dir, "results.json")
    doc = json.load(open(res))
    summ = doc.get("summary", {})
    rows = doc["rows"]
    if a.n:
        rows = rows[:a.n]
    print("[data] %s rows=%d" % (res, len(rows)), flush=True)

    # **本番と同じ設定を結果ファイルから取る。**既定値を書き写すと、
    # 本番が別の値で回っていたときに気づけない。
    seed = int(summ.get("seed", 42))
    # 本番の既定は C.DEFAULT_SPEAKER。summary に speaker が無い版で作られた
    # ディレクトリもあるので、そのときは本番と同じ既定へ落とす。
    speaker = str(summ.get("speaker", C.DEFAULT_SPEAKER))
    ref_wav = summ.get("ref_wav")
    ref_text = summ.get("ref_text")
    for k, v in (("seed", seed), ("speaker", speaker),
                 ("ref_wav", ref_wav), ("ref_text", ref_text)):
        print("[cfg] %-9s %s" % (k, v), flush=True)
    if not ref_wav or not ref_text:
        raise SystemExit("results.json に ref_wav / ref_text が無い。"
                         "本番と同じ参照音声を特定できないので止める")
    if not os.path.isfile(ref_wav):
        # コンテナの内外でパスが変わる場合に備える
        alt = ref_wav.replace("/mnt/", PROJECT_ROOT + "/")
        if os.path.isfile(alt):
            ref_wav = alt
        else:
            raise SystemExit("参照音声が見つからない: %s" % ref_wav)

    todo = [r for r in rows
            if not all(os.path.exists(os.path.join(
                a.dir, "s%04d_%s.wav" % (r["pair_idx"], c))) for c in CONDS)]
    print("[data] 目標文が既にある対 %d / 作る対 %d"
          % (len(rows) - len(todo), len(todo)), flush=True)
    if not todo:
        print("[done] 作るものが無い", flush=True)
        return

    dtype = getattr(torch, a.dtype)
    bundle = C.load_csm(summ.get("model_id", "sesame/csm-1b"),
                        device=a.device, dtype=dtype)
    ref = C.build_ref(bundle, ref_wav, ref_text, speaker=speaker)
    gen = C.GenConfig(seed=seed)
    print("[gen] speaker=%s seed=%d" % (speaker, seed), flush=True)

    t0 = time.time()
    made, fail = 0, []
    for j, r in enumerate(todo):
        pidx = r["pair_idx"]
        for cond, slot in CONDS.items():
            path = os.path.join(a.dir, "s%04d_%s.wav" % (pidx, cond))
            if os.path.exists(path):
                continue
            try:
                # 保存は本番 run_multientity_csm.py:1271 と同じ形。
                g = C.synthesize(bundle, r["text_refs"][slot], ref=ref,
                                 speaker=speaker, gen=gen)
                C.save_wav(path, g["audio"], bundle.sampling_rate)
                made += 1
            except Exception as e:
                msg = "%s %s: %s" % (cond, type(e).__name__, e)
                print("  [%d] FAIL %s" % (pidx, msg[:200]), flush=True)
                fail.append({"pair_idx": pidx, "cond": cond,
                             "reason": msg[:300]})
        if (j + 1) % 25 == 0:
            el = (time.time() - t0) / 60
            print("  [%d/%d] 作った音声 %d  %.1fmin (%.1fs/文)"
                  % (j + 1, len(todo), made, el, el * 60 / max(made, 1)),
                  flush=True)

    summary = {"dir": a.dir, "made": made, "n_fail": len(fail),
               "n_rows": len(rows), "seed": seed, "speaker": speaker,
               "ref_wav": ref_wav, "ref_text": ref_text,
               "elapsed_min": round((time.time() - t0) / 60, 2),
               "fail": fail}
    with open(os.path.join(a.dir, "results_tgt.json"), "w") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    if fail:
        raise SystemExit("目標文の合成に失敗した対がある: %d" % len(fail))


if __name__ == "__main__":
    main()
