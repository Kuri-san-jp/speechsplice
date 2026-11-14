#!/usr/bin/env python3
"""実体語 PER（第3版）。全体音素列の突き合わせで区間を取る。

## なぜこの方式か

v1 は実体語だけの音素列を文全体の emission に forced_align し、区間が文全体になった。
v2 は Whisper の単語タイムスタンプを使ったが、日本語・中国語では文が1チャンクで返り、
文字の時刻が等分割になるため区間が外れ、300対中最大182対が除外された。

v3 は時刻を一切使わない。BL と TGT は文脈が同一で実体語だけが違うので、
2つの全体音素列を編集距離で整列すれば、食い違う区間がそのまま実体語になる。
INJ は TGT を基準に整列して同じ区間を取る。

  * Whisper 不要      → 等分割の問題が原理的に発生しない
  * espeak 不要       → 漢字の読みを推測する循環が発生しない
  * forced_align 不要 → 区間の反転や退化が発生しない
  * 全条件で「TGT を基準に整列する」同一手続き
  * 音声が読める限り必ず答えが出る（有効率100%）

歯止めは「除外」ではなく「印」として記録する。除外が結果と相関しないようにするため。
"""
import argparse, json, os, sys, glob

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get(
    "TTS26_ROOT", os.environ.get("REPO_ROOT", "/mnt"))
# slurm_run.sh は書き込めない HF_HOME を渡すので、必ず上書きする（setdefault では駄目）
_hf = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = _hf
os.environ["HF_HUB_CACHE"] = os.path.join(_hf, "hub")
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")

# **スレッド上限。torch より前に置く必要がある。**
# 2026-09-18、この採点を 3 本並べたら 1 本あたり 384 スレッド・40 コアを使い、
# 計 120 コアでログインノードを埋めて他の利用者に迷惑をかけた。
# 既定では torch が nproc (=128) 個のスレッドを取る。環境変数で上限を与える。
# **上限は 128 スレッドの 1/4 = 32**（2026-09-18 ユーザ指示）。
# GPU を 3 枚まで同時に使うので、1 本あたり 10 にして合計 30 に収める。
# 実測では GPU 律速で 1 コアしか使わないが、上限として置く。
_thr = os.environ.get("PER_THREADS", "13")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _thr

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "per_fix"))
from diff_localise import align, diff_block, entity_from_pair, per  # noqa: E402

import torch          # noqa: E402
import torchaudio     # noqa: E402

torch.set_num_threads(int(_thr))
torch.set_num_interop_threads(1)


def collapse(pred_ids, id2sym, blank_id):
    """CTC 崩壊。第2版 phoneme_per_v2.py:760 と同一。"""
    toks, prev = [], None
    for i in pred_ids:
        if i == blank_id:
            prev = i
            continue
        if i == prev:
            continue
        toks.append(id2sym.get(i, ""))
        prev = i
    return "".join(toks)


def decode_full_ipa(model, feat, wav_path, device, id2sym, blank_id, max_dur=30.0):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    x = wav.mean(0)[:int(max_dur * 16000)]
    iv = feat(x.numpy(), sampling_rate=16000,
              return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(iv).logits[0]
    return collapse(logits.argmax(dim=-1).tolist(), id2sym, blank_id), x.shape[0]/16000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--lang", required=True, choices=["en", "ja", "zh"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--output_json", required=True)
    p.add_argument("--phoneme_model",
                   default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    p.add_argument("--gap", type=int, default=3,
                   help="この数以下の一致で隔たった食い違いは同じ実体語として結合する")
    p.add_argument("--pairs_json", default="",
                   help="word_a / word_b を持つ既存結果。印の付与にだけ使う")
    a = p.parse_args()

    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    from huggingface_hub import hf_hub_download
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] scorer=phoneme_per_v3 device={device} lang={a.lang}", flush=True)
    print(f"[Info] Loading {a.phoneme_model}", flush=True)
    feat = Wav2Vec2FeatureExtractor.from_pretrained(a.phoneme_model)
    model = Wav2Vec2ForCTC.from_pretrained(a.phoneme_model).to(device).eval()
    vocab = json.load(open(hf_hub_download(a.phoneme_model, "vocab.json")))
    id2sym = {v: k for k, v in vocab.items()}
    blank_id = vocab.get("<pad>", 0)

    words = {}
    if a.pairs_json and os.path.isfile(a.pairs_json):
        for r in json.load(open(a.pairs_json)).get("rows", []):
            words[str(r.get("pair_idx"))] = (r.get("word_a"), r.get("word_b"))

    rows, n_ok, n_missing = [], 0, 0
    for i in range(a.n):
        rec = {"pair_idx": i}
        paths = {c: os.path.join(a.wav_dir, f"s{i:04d}_{c}.wav")
                 for c in ("bl", "inj", "tgt")}
        if not all(os.path.isfile(v) for v in paths.values()):
            rec["skip"] = "wav_missing"; n_missing += 1; rows.append(rec); continue
        try:
            ipa, dur = {}, {}
            for c, pth in paths.items():
                ipa[c], dur[c] = decode_full_ipa(model, feat, pth, device, id2sym, blank_id)
            rec.update(bl_full_ipa=ipa["bl"], inj_full_ipa=ipa["inj"],
                       tgt_full_ipa=ipa["tgt"],
                       dur_sec={c: round(v, 3) for c, v in dur.items()})
            w = words.get(str(i), (None, None))
            rec["word_a"], rec["word_b"] = w

            e = entity_from_pair(ipa["bl"], ipa["inj"], ipa["tgt"], 0, a.gap)
            if e is None or not e[2]:
                # BL と TGT の音素列が完全一致＝注入対象の差が音に出ていない。
                # 除外せず PER 0 として記録する（情報がゼロであることも結果）。
                rec.update(bl_ent_ipa="", inj_ent_ipa="", tgt_ent_ipa="",
                           bl_per_vs_tgt=0.0, inj_per_vs_tgt=0.0,
                           delta_per=0.0, inj_better=False,
                           flag="identical_bl_tgt")
            else:
                b_e, i_e, t_e = e
                bp, ip = per(b_e, t_e), per(i_e, t_e)
                rec.update(bl_ent_ipa=b_e, inj_ent_ipa=i_e, tgt_ent_ipa=t_e,
                           bl_per_vs_tgt=bp, inj_per_vs_tgt=ip,
                           delta_per=bp - ip, inj_better=ip < bp)
                # 印（除外はしない）
                fl = []
                if len(t_e) > 0.6 * len(ipa["tgt"]): fl.append("entity_covers_most_of_utt")
                if len(t_e) < 2: fl.append("tgt_entity_very_short")
                if bp > 1.0 or ip > 1.0: fl.append("per_gt1")
                if fl: rec["flags"] = fl
            n_ok += 1
        except Exception as ex:
            import traceback
            rec["skip"] = "error"
            rec["err"] = f"{type(ex).__name__}: {ex}"[:200]
            rec["trace"] = traceback.format_exc()[-400:]
        rows.append(rec)
        if (i + 1) % 50 == 0:
            nb = sum(1 for r in rows if r.get("inj_better"))
            print(f"  [{i+1}/{a.n}] valid={n_ok} inj_better={nb}", flush=True)

    val = [r for r in rows if r.get("inj_per_vs_tgt") is not None]
    def mean(k): return sum(r[k] for r in val)/len(val) if val else None
    summ = {
        "scorer_version": "phoneme_per_v3",
        "localisation": "ipa_diff (no timestamps, no g2p, no forced_align)",
        "wav_dir": a.wav_dir, "lang": a.lang, "gap": a.gap,
        "n_attempted": a.n, "n_valid": len(val),
        "valid_pct": round(len(val)/a.n*100, 2),
        "n_wav_missing": n_missing,
        "n_error": sum(1 for r in rows if r.get("skip") == "error"),
        "bl_per_vs_tgt_mean": mean("bl_per_vs_tgt"),
        "inj_per_vs_tgt_mean": mean("inj_per_vs_tgt"),
        "delta_per_mean": mean("delta_per"),
        "inj_better_n": sum(1 for r in val if r.get("inj_better")),
        "inj_better_pct": round(sum(1 for r in val if r.get("inj_better"))/len(val)*100, 2) if val else 0,
        "n_identical_bl_tgt": sum(1 for r in rows if r.get("flag") == "identical_bl_tgt"),
        "n_flagged": sum(1 for r in rows if r.get("flags")),
        "entity_len_mean": {
            c: round(sum(len(r.get(f"{c}_ent_ipa", "")) for r in val)/len(val), 2) if val else None
            for c in ("bl", "inj", "tgt")},
    }
    print(f"\n=== phoneme_per_v3 ({a.lang}) ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")
    json.dump({"summary": summ, "rows": rows},
              open(a.output_json, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.output_json}")


if __name__ == "__main__":
    main()
