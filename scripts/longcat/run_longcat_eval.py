#!/usr/bin/env python3
"""LongCat-AudioDiT-1B に SpeechSplice を適用して cross-culture 300 ペアを生成・評価する (GPU).

生成するもの (既存 3 モデルと同じ命名規約):
    {wav_dir}/s{pair_idx:04d}_bl.wav    = TTS(text_a)
    {wav_dir}/s{pair_idx:04d}_inj.wav   = SpeechSplice(text_a の encoder 出力に
                                          text_b の実体語スパンを移植)
    {wav_dir}/s{pair_idx:04d}_tgt.wav   = TTS(text_b)
3 本そろっていないペアはスコアラ側で available:false として捨てられるので、
TGT の生成を省略してはならない (mcd_pysptk.py:228-229, phoneme_per.py:166-167)。

指標は既存ファイルをそのまま使う (1 行も改変していない):
    MCD : speechsplice_release/src/mcd_pysptk.py --mode whole
    PIB : speechsplice_release/src/phoneme_per.py
--run_scorers を付けると生成後にサブプロセスで呼ぶ。後から単体で回してもよい。

統制条件 (既定ではすべて素の splice、フラグで切り替える):
  (a) duration は常に source テキスト (text_a) 由来。BL と INJ で完全に同一。
      LongCat の初期ノイズは randn(duration, latent_dim) なので、duration が
      同じであれば BL と INJ のノイズは bit 単位で同一になる。
  (b) --splice_part hL_only : h_0 (文脈非依存の語埋め込み) は source のまま残し、
      文脈依存側 LN(h_L) だけ差し替える。m == n のペアでは厳密。
  (c) --adaln_source : AdaLN に入る text 条件 (系列平均) を source に固定する。
      これを付けると BL と INJ の差は cross-attention の key/value だけになる。

--span_extension を付けると、語そのままでは往復一致しない対 (トークン化器が実体語と
隣の機能語を 1 トークンに融合する対。中国語の「的」など) に限り最小対称拡張
(scripts/projection/span_extend.py) を試す。既定は off で従来どおり落とす。
往復一致する対は on/off に関わらず同じスパン・同じ結果になる。

実行例 (GPU):
  SLURM_TIME=02:00:00 slurm_run.sh \\
    python3 scripts/longcat/run_longcat_eval.py --lang en --n 300 --run_scorers
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "longcat"))

import longcat_splice as L  # noqa: E402  (HF_HOME はこの import で設定される)

import numpy as np  # noqa: E402
import torch  # noqa: E402

MCD_SCORER = os.path.join(PROJECT_ROOT, "src", "mcd_pysptk.py")
PIB_SCORER = os.path.join(PROJECT_ROOT, "src", "phoneme_per.py")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_pairs(lang, n):
    """Table 3 (cross-architecture) で既存 3 モデルが使っているデータ.

    pair_idx は最初から入っている (0..299)。mcd_pysptk.py / phoneme_per.py が
    内部で同じ関数を読むので、無改造でスコアラに渡せる。
    """
    if lang == "en":
        from en_cross_culture_names import build_300_pairs
    elif lang == "zh":
        from zh_cross_culture_names import build_300_pairs
    else:
        raise SystemExit(
            f"--lang {lang} は対象外。LongCat-AudioDiT は中国語と英語のみ "
            "(日本語は DiT が学習していない)。")
    pairs = build_300_pairs()[:n]
    for i, p in enumerate(pairs):
        p.setdefault("pair_idx", i)
    return pairs


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def save_wav(path, audio, sr):
    """PCM_16 で保存。途中で落ちても壊れた wav が残らないよう tmp -> rename."""
    import soundfile as sf
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        raise RuntimeError(f"empty audio for {path}")
    n_clip = int(np.sum(np.abs(a) > 1.0))
    a = np.clip(a, -1.0, 1.0)
    tmp = path + ".tmp"
    sf.write(tmp, a, sr, format="WAV", subtype="PCM_16")
    os.replace(tmp, path)
    return int(a.size), float(a.size / sr), float(n_clip / a.size)


def atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def wav_paths(wav_dir, pair_idx):
    return (os.path.join(wav_dir, f"s{pair_idx:04d}_bl.wav"),
            os.path.join(wav_dir, f"s{pair_idx:04d}_inj.wav"),
            os.path.join(wav_dir, f"s{pair_idx:04d}_tgt.wav"))


def condition_tag(args):
    tag = f"{args.lang}_cfg{args.cfg_strength:g}_{args.guidance_method}_{args.splice_part}"
    if args.adaln_source:
        tag += "_adalnsrc"
    if args.tgt_duration == "source":
        tag += "_tgtdursrc"
    if args.steps != L.DEFAULT_STEPS:
        tag += f"_nfe{args.steps}"
    return tag


# --------------------------------------------------------------------------
# kappa aggregation
# --------------------------------------------------------------------------
def summarize_kappa(rows):
    """kappa の層ごと集計 (entity ごとの値を層ごとに単純平均).

    層 0 = umt5-base encoder の embedding 出力、層 12 = encoder 最終層。
    kappa_H は実際に cross-attention の key/value になる
    H = LN(h_L) + LN(h_0) 上の cosine。
    母数・題材は既存 3 モデルのセルと揃っていないので直接比較はできない。
    """
    per_layer, last, hh = {}, [], []
    for r in rows:
        if r.get("kappa_last_layer") is not None:
            last.append(r["kappa_last_layer"])
        if r.get("kappa_H") is not None:
            hh.append(r["kappa_H"])
        for li, v in enumerate(r.get("kappa_by_layer") or []):
            per_layer.setdefault(li, []).append(v)

    def stat(vals):
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0:
            return None
        return {"n": int(arr.size), "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std(ddof=1)), 6) if arr.size > 1 else None,
                "median": round(float(np.median(arr)), 6)}

    out = {str(li): stat(v) for li, v in sorted(per_layer.items())}
    summary = {"by_layer": out, "last_layer": stat(last), "H_cross_attn_input": stat(hh)}
    summary["note"] = (
        "kappa_l(e) = cos(mean_j h^src_{s_j,l}, mean_k h^tgt_{u_k,l}). "
        "layer 0 = umt5-base embedding output, layer 12 = encoder last layer. "
        "kappa_H is on H = LN(h_L) + LN(h_0), the tensor actually spliced and fed "
        "to cross-attention. Corpus/N differ from the existing Table 3 kappa cells; "
        "not directly comparable.")
    return summary


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def run_generation(args, pairs):
    os.makedirs(args.wav_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.json")

    rows = {}
    if os.path.exists(progress_path) and not args.overwrite:
        try:
            with open(progress_path) as f:
                for r in json.load(f).get("rows", []):
                    rows[str(r["pair_idx"])] = r
            print(f"[resume] loaded {len(rows)} rows from {progress_path}", flush=True)
        except Exception as e:
            print(f"[resume] could not read {progress_path}: {e}", flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] {args.model_id} dtype={args.dtype} device={args.device} "
          f"HF_HOME={os.environ['HF_HOME']}", flush=True)
    t_load = time.time()
    bundle = L.load_longcat(model_id=args.model_id, tokenizer_id=args.tokenizer_id,
                            device=args.device, dtype=dtype, vae_half=not args.no_vae_half,
                            load_mode=args.load_mode)
    print(f"[load] done in {time.time() - t_load:.1f}s  dir={bundle.model_dir}", flush=True)
    print(f"[load] state_dict: n={bundle.load_report['n_tensors']} "
          f"missing={bundle.load_report['missing_keys']} "
          f"unexpected={bundle.load_report['unexpected_keys']}", flush=True)
    print(f"[gen] steps={args.steps} cfg_strength={args.cfg_strength} "
          f"guidance={args.guidance_method} seed={args.seed} "
          f"splice_part={args.splice_part} adaln_source={args.adaln_source} "
          f"tgt_duration={args.tgt_duration}", flush=True)

    todo = pairs
    if args.num_shards > 1:
        todo = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_index]
        print(f"[shard] {args.shard_index}/{args.num_shards}: {len(todo)} of {len(pairs)} pairs",
              flush=True)

    t0 = time.time()
    n_done = n_skip = n_err = 0
    for k, pair in enumerate(todo):
        pidx = pair["pair_idx"]
        bl_p, inj_p, tgt_p = wav_paths(args.wav_dir, pidx)
        if (not args.overwrite) and all(os.path.exists(p) for p in (bl_p, inj_p, tgt_p)):
            n_skip += 1
            if str(pidx) not in rows:
                rows[str(pidx)] = {"pair_idx": pidx, "word_a": pair["word_a"],
                                   "word_b": pair["word_b"], "status": "resumed_wavs_only"}
            continue
        try:
            tp = time.time()
            # スパンの往復一致を先に確かめる。SentencePiece が実体語と後続の語
            # (中国語の「的」など) を 1 トークンに融合してしまうペアは、
            # 差し替えると実体語以外まで置き換わるので生成せずに落とす。
            # 落としたペアはスコアラ側で available:false として除外される。
            # --span_extension を付けたときだけ、往復一致しない対に最小対称拡張を
            # 試す (scripts/projection/span_extend.py)。往復一致する対は
            # span_extension に関わらず従来と同じスパンを通る。
            s_span, t_span, span_ext = L.entity_token_span_pair(
                bundle.tokenizer, pair["text_a"], pair["word_a"],
                pair["text_b"], pair["word_b"], span_extension=args.span_extension)
            ok_s, msg_s = L.verify_entity_span(s_span, strict=False)
            ok_t, msg_t = L.verify_entity_span(t_span, strict=False)
            if not (ok_s and ok_t):
                rows[str(pidx)] = {
                    "pair_idx": pidx, "status": "skipped_span_roundtrip",
                    "word_a": pair["word_a"], "word_b": pair["word_b"],
                    "reason": (msg_s or msg_t)[:300],
                    "s_covered": s_span["covered_text"], "t_covered": t_span["covered_text"],
                }
                n_skip += 1
                print(f"[skip] pair {pidx}: span roundtrip 失敗 ({msg_s or msg_t})", flush=True)
                continue
            # テキストは 1 ペアにつき 1 回だけ encode し、BL / TGT / INJ で
            # 同じテンソルを使い回す (BL と INJ の差が splice だけであることを保証する)。
            src = L.encode_for_text(bundle, pair["text_a"])
            tgt = L.encode_for_text(bundle, pair["text_b"])
            dur_src, sec_src = L.duration_frames(bundle, src["norm_text"])
            if args.tgt_duration == "source":
                dur_tgt, sec_tgt = dur_src, sec_src
            else:
                dur_tgt, sec_tgt = L.duration_frames(bundle, tgt["norm_text"])

            gen_kw = dict(seed=args.seed, steps=args.steps,
                          cfg_strength=args.cfg_strength,
                          guidance_method=args.guidance_method)
            bl_wav = L.generate_from_embedding(bundle, src["H"], dur_src, **gen_kw)
            tgt_wav = L.generate_from_embedding(bundle, tgt["H"], dur_tgt, **gen_kw)
            inj = L.splice_synthesize(
                bundle, pair["text_a"], pair["word_a"], pair["text_b"], pair["word_b"],
                splice_part=args.splice_part, adaln_source=args.adaln_source,
                collect_kappa=not args.no_kappa, strict_span=True,
                src_parts=src, tgt_parts=tgt,
                span_extension=args.span_extension, **gen_kw)

            if args.splice_part == "hL_only" and args.equal_len_only and not inj["equal_len"]:
                rows[str(pidx)] = {"pair_idx": pidx, "status": "skipped_unequal_len",
                                   "word_a": pair["word_a"], "word_b": pair["word_b"],
                                   "m_src_entity_tokens": inj["m_src_entity_tokens"],
                                   "n_tgt_entity_tokens": inj["n_tgt_entity_tokens"]}
                n_skip += 1
                continue

            n_bl, d_bl, c_bl = save_wav(bl_p, bl_wav, bundle.sampling_rate)
            n_tg, d_tg, c_tg = save_wav(tgt_p, tgt_wav, bundle.sampling_rate)
            n_in, d_in, c_in = save_wav(inj_p, inj["audio"], bundle.sampling_rate)
            row = {
                "pair_idx": pidx, "status": "ok",
                "word_a": pair["word_a"], "word_b": pair["word_b"],
                "text_a": pair["text_a"], "text_b": pair["text_b"],
                "s_span": inj["s_span"], "t_span": inj["t_span"],
                "s_tokens": inj["s_tokens"], "t_tokens": inj["t_tokens"],
                "m_src_entity_tokens": inj["m_src_entity_tokens"],
                "n_tgt_entity_tokens": inj["n_tgt_entity_tokens"],
                "equal_len": inj["equal_len"],
                "T_src": inj["T_src"], "T_tgt": inj["T_tgt"],
                "T_spliced": inj["T_spliced"], "delta_tok": inj["delta_tok"],
                "duration_frames": {"bl": dur_src, "tgt": dur_tgt, "inj": inj["duration_frames"]},
                "duration_sec_est": {"bl": round(sec_src, 3), "tgt": round(sec_tgt, 3)},
                "dur_bl_sec": round(d_bl, 3), "dur_tgt_sec": round(d_tg, 3),
                "dur_inj_sec": round(d_in, 3),
                "clip_ratio": [round(c_bl, 6), round(c_tg, 6), round(c_in, 6)],
                "adaln_source": inj["adaln_source"], "adaln_stats": inj["adaln_stats"],
                "kappa_last_layer": inj.get("kappa_last_layer"),
                "kappa_H": inj.get("kappa_H"),
                "kappa_by_layer": inj.get("kappa_by_layer"),
                "elapsed_sec": round(time.time() - tp, 2),
            }
            if span_ext is not None:
                # 原稿に「何対で拡張が使われたか」を書くための記録。
                row["span_extension"] = span_ext
                if inj.get("span_extension") != span_ext:
                    raise RuntimeError(
                        "span extension が事前チェックと生成で食い違った: "
                        f"{span_ext} vs {inj.get('span_extension')}")
            elif inj.get("span_extension") is not None:
                raise RuntimeError(
                    f"生成側だけ span extension が使われた: {inj['span_extension']}")
            if dur_src != inj["duration_frames"]:
                raise RuntimeError("統制 (a) 違反: BL と INJ の duration が違う")
            rows[str(pidx)] = row
            n_done += 1
        except Exception as e:
            import traceback
            n_err += 1
            rows[str(pidx)] = {"pair_idx": pidx, "status": "error",
                               "word_a": pair.get("word_a"), "word_b": pair.get("word_b"),
                               "err": f"{type(e).__name__}: {e}"[:500]}
            print(f"[err] pair {pidx}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if args.stop_on_error:
                raise
        atomic_json_dump({"rows": [rows[k2] for k2 in sorted(rows, key=lambda x: int(x))]},
                         progress_path)
        if (k + 1) % args.log_every == 0 or k + 1 == len(todo):
            el = time.time() - t0
            rate = el / max(1, n_done) if n_done else float("nan")
            left = (len(todo) - (k + 1)) * rate
            print(f"  [{k + 1}/{len(todo)}] done={n_done} skip={n_skip} err={n_err} "
                  f"{el / 60:.1f}min elapsed, ~{left / 60:.1f}min left "
                  f"({rate:.1f}s/pair)", flush=True)

    all_rows = [rows[k2] for k2 in sorted(rows, key=lambda x: int(x))]
    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    manifest = {
        "meta": {
            "model_id": args.model_id,
            "model_dir": bundle.model_dir,
            "tokenizer_id": args.tokenizer_id,
            "dataset": "cross_culture",
            "lang": args.lang,
            "n_target": len(pairs),
            "seed": args.seed,
            "steps_nfe": args.steps,
            "cfg_strength": args.cfg_strength,
            "guidance_method": args.guidance_method,
            "splice_part": args.splice_part,
            "adaln_source": args.adaln_source,
            "tgt_duration": args.tgt_duration,
            "equal_len_only": args.equal_len_only,
            "span_extension": bool(args.span_extension),
            "prompt_audio": None,
            "dtype": args.dtype,
            "vae_half": not args.no_vae_half,
            "load_mode": args.load_mode,
            "device": args.device,
            "sampling_rate": bundle.sampling_rate,
            "latent_hop": bundle.latent_hop,
            "wav_dir": args.wav_dir,
            "condition_tag": condition_tag(args),
            "splice": ("H = LN(h_L) + LN(h_0) of the frozen umt5-base encoder; "
                       "entity span replaced by the target text's span via torch.cat, "
                       "T -> T - m + n; passed to the official forward as text_embedding "
                       "(no hook, no code change). duration is always taken from the "
                       "source text so BL and INJ share the initial noise exactly."),
            "controls": {
                "a_duration_from_source": True,
                "b_splice_part": args.splice_part,
                "c_adaln_text_mean_source": args.adaln_source,
            },
            "state_dict_load": bundle.load_report,
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "n_generated": n_done, "n_skipped": n_skip, "n_error": n_err,
            "wallclock_sec": round(time.time() - t0, 1),
        },
        "kappa": summarize_kappa(ok_rows),
        "rows": all_rows,
    }
    man_path = os.path.join(args.out_dir, f"longcat_{condition_tag(args)}_manifest.json")
    atomic_json_dump(manifest, man_path)
    print(f"\n[manifest] -> {man_path}")
    print(f"[manifest] generated={n_done} skipped={n_skip} error={n_err}")
    if manifest["kappa"].get("last_layer"):
        print(f"[kappa] encoder last layer mean = "
              f"{manifest['kappa']['last_layer']['mean']:.4f} "
              f"(n={manifest['kappa']['last_layer']['n']})")
    if manifest["kappa"].get("H_cross_attn_input"):
        print(f"[kappa] H (cross-attn input) mean = "
              f"{manifest['kappa']['H_cross_attn_input']['mean']:.4f}")
    del bundle
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return man_path


# --------------------------------------------------------------------------
# scorers (既存ファイルをサブプロセスでそのまま呼ぶ)
# --------------------------------------------------------------------------
def run_scorers(args):
    env = dict(os.environ)
    env["HF_HOME"] = os.environ["HF_HOME"]
    env["HF_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    env["REPO_ROOT"] = PROJECT_ROOT
    tag = condition_tag(args)
    jobs = []
    if args.scorers in ("both", "mcd"):
        # mcd_pysptk.py --mode whole は GPU を一切使わない (pysptk + numpy DTW)。
        # GPU ジョブの中で回すと GPU を遊ばせるので、--scorers pib で分けて
        # MCD だけ後から CPU で直接叩いてもよい (README 参照)。
        mcd_json = args.mcd_json or os.path.join(
            PROJECT_ROOT, "eval_results", f"_caseSPTK_whole_longcat_{tag}.json")
        jobs.append(("MCD", [sys.executable, MCD_SCORER,
                             "--wav_dir", args.wav_dir, "--lang", args.lang,
                             "--mode", "whole", "--n", str(args.n),
                             "--output_json", mcd_json]))
    if args.scorers in ("both", "pib"):
        # phoneme_per.py は wav2vec2-CTC の forced alignment を使うので GPU 推奨。
        pib_json = args.pib_json or os.path.join(
            PROJECT_ROOT, "eval_results", f"_caseSPTK_phperPER_longcat_{tag}.json")
        jobs.append(("PIB", [sys.executable, PIB_SCORER,
                             "--wav_dir", args.wav_dir, "--lang", args.lang,
                             "--n", str(args.n), "--output_json", pib_json]))
    for name, cmd in jobs:
        print(f"\n[scorer:{name}] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        print(f"[scorer:{name}] exit={r.returncode}", flush=True)
        if r.returncode != 0:
            print(f"[scorer:{name}] FAILED", flush=True)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="en", choices=["en", "zh"],
                   help="LongCat-AudioDiT は中国語と英語のみ")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--wav_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model_id", default=L.DEFAULT_MODEL_ID)
    p.add_argument("--tokenizer_id", default=L.DEFAULT_TOKENIZER_ID)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["bfloat16", "float16", "float32"],
                   help="公式 inference.py は dtype を変えない (float32)")
    p.add_argument("--no_vae_half", action="store_true",
                   help="既定は公式と同じく Wav-VAE のみ fp16")
    p.add_argument("--load_mode", default="meta", choices=["meta", "cpu"])
    p.add_argument("--seed", type=int, default=L.SEED)
    p.add_argument("--hf_home", default=None)
    # decoding
    p.add_argument("--steps", type=int, default=L.DEFAULT_STEPS, help="Euler steps (NFE)")
    p.add_argument("--cfg_strength", type=float, default=L.DEFAULT_CFG,
                   help="1.0 / 2.0 / 4.0 を掃引する用")
    p.add_argument("--guidance_method", default=L.DEFAULT_GUIDANCE, choices=["cfg", "apg"])
    # 統制条件
    p.add_argument("--splice_part", default="both", choices=["both", "hL_only"],
                   help="hL_only: h_0 を source のまま残す統制 (b)")
    p.add_argument("--adaln_source", action="store_true",
                   help="AdaLN の text 条件を source に固定する統制 (c)")
    p.add_argument("--tgt_duration", default="own", choices=["own", "source"],
                   help="own: TGT は text_b 由来の duration (既定、素の TTS(text_b))。"
                        "source: TGT も text_a 由来にすると 3 条件で初期ノイズが完全に共有される")
    p.add_argument("--equal_len_only", action="store_true",
                   help="hL_only のとき m == n のペアだけ生成する (h_0 写像が厳密になる部分集合)")
    p.add_argument("--span_extension", action="store_true",
                   help="語そのままで往復一致しない対 (中国語の「的」など) に限り "
                        "最小対称拡張 (scripts/projection/span_extend.py) を試す。"
                        "既定 off = 従来どおり skipped_span_roundtrip で落とす。"
                        "往復一致する対の結果は on/off で 1 ビットも変わらない")
    # control
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--no_kappa", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    # scoring
    p.add_argument("--run_scorers", action="store_true",
                   help="生成後に mcd_pysptk.py / phoneme_per.py を呼ぶ")
    p.add_argument("--scorers", default="both", choices=["both", "mcd", "pib"],
                   help="どのスコアラを回すか。MCD は CPU 専用なので GPU ジョブでは "
                        "pib だけにして、MCD は別途 CPU で回す手もある")
    p.add_argument("--mcd_json", default=None)
    p.add_argument("--pib_json", default=None)
    args = p.parse_args()

    if args.hf_home:
        L.setup_hf_home(args.hf_home)
    tag = condition_tag(args)
    if args.wav_dir is None:
        args.wav_dir = os.path.join(PROJECT_ROOT, "eval_results", "_cc_wavs_a",
                                    f"longcat_{tag}")
    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, "eval_results", f"longcat_cross_culture_{tag}")
    if not os.path.isabs(args.wav_dir):
        args.wav_dir = os.path.join(PROJECT_ROOT, args.wav_dir)
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.chdir(PROJECT_ROOT)

    pairs = load_pairs(args.lang, args.n)
    print(f"[data] cross_culture lang={args.lang} n={len(pairs)} "
          f"first={pairs[0]['word_a']}->{pairs[0]['word_b']} "
          f"last={pairs[-1]['word_a']}->{pairs[-1]['word_b']}", flush=True)
    print(f"[data] condition={tag}", flush=True)
    print(f"[data] wav_dir={args.wav_dir}", flush=True)

    if not args.skip_generation:
        run_generation(args, pairs)
    if args.run_scorers:
        run_scorers(args)


if __name__ == "__main__":
    main()
