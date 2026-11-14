#!/usr/bin/env python3
"""Llasa-3B の splice 層掃引 (GPU). 格子は Qwen3-TTS talker と同じ 0 / 7 / 14 / 21 / 26.

Llasa-3B も Qwen3-TTS talker も 28 層なので、既存の掃引格子をそのまま使える
(Qwen3-TTS では 26 層目が最良で +4.5pt だった)。

やること:
  1. モデルを **1 回だけ** 読み、層ごとに INJ を生成する。
     BL と TGT は層に依存しないので最初の層のディレクトリで 1 回だけ作り、
     以降の層へは hard link で持ち込む (run_llasa_eval.py --link_bl_tgt_from)。
     生成量は 2N + 5N ではなく 2N + 5N の INJ だけ、つまり N*(2+5) 回。
  2. 層ごとに PIB (phoneme_per.py) を回す。--with_mcd を付ければ MCD も回す。
  3. 層 x 指標の表を JSON と Markdown で書き出す。

既定の N は 50 (Qwen3-TTS の掃引と同じ規模感の探索用)。本番の 300 ペア評価は
最良層を決めてから run_llasa_eval.py で回す。

実行例 (GPU):
  SLURM_TIME=04:00:00 slurm_run.sh \\
    python3 scripts/llasa/run_llasa_layersweep.py --lang en --n 50 \\
      --allow_download --run_scorers
"""
import argparse
import copy
import json
import os
import sys
import time

PROJECT_ROOT = "/mnt" if os.path.isdir("/mnt/scripts") else os.environ.get("REPO_ROOT", "/mnt")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "llasa"))

import llasa_splice as L  # noqa: E402
import run_llasa_eval as R  # noqa: E402

import torch  # noqa: E402


def parse_layers(s, num_layers=None):
    if s is None or s.strip() == "":
        layers = list(L.SWEEP_LAYERS)
    else:
        layers = [int(x) for x in s.replace(",", " ").split()]
    if num_layers is not None:
        bad = [x for x in layers if not (0 <= x < num_layers)]
        if bad:
            raise SystemExit(f"層 {bad} が範囲外 (0..{num_layers - 1})")
    return layers


def layer_args(base, layer, base_wav_dir, base_out_dir, link_from=None):
    a = copy.deepcopy(base)
    a.layer = layer
    a.wav_dir = os.path.join(base_wav_dir, f"L{layer:02d}")
    a.out_dir = os.path.join(base_out_dir, f"L{layer:02d}")
    a.link_bl_tgt_from = link_from
    a.mcd_json = os.path.join(a.out_dir, "mcd_whole.json")
    a.pib_json = os.path.join(a.out_dir, "phoneme_per.json")
    return a


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def collect(args, layers, base_out_dir):
    """層ごとの PIB / MCD / kappa を 1 つの表にまとめる."""
    table = []
    for layer in layers:
        out_dir = os.path.join(base_out_dir, f"L{layer:02d}")
        row = {"layer": layer, "hidden_index": L.hidden_index_for_layer(layer)}
        man = None
        for fn in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
            if fn.endswith("_manifest.json"):
                man = read_json(os.path.join(out_dir, fn))
        if man:
            k = (man.get("kappa") or {}).get("at_splice_layer")
            row["kappa_at_splice"] = k["mean"] if k else None
            row["kappa_n"] = k["n"] if k else None
            row["n_generated"] = man["meta"].get("n_generated")
            row["n_error"] = man["meta"].get("n_error")
            eq = [r.get("equal_len") for r in man.get("rows", []) if r.get("equal_len") is not None]
            row["equal_len_pairs"] = f"{sum(1 for e in eq if e)}/{len(eq)}" if eq else None
        pib = read_json(os.path.join(out_dir, "phoneme_per.json"))
        if pib:
            s = pib["summary"]
            row["pib_pct"] = s.get("inj_better_pct")
            row["pib_ci95_half_pp"] = s.get("inj_better_ci95_half_pp")
            row["paired_t_pval"] = s.get("paired_t_pval")
            row["pib_n_valid"] = s.get("n_valid")
            row["bl_per"] = s.get("bl_per_vs_tgt_mean")
            row["inj_per"] = s.get("inj_per_vs_tgt_mean")
            row["delta_per"] = s.get("delta_per_mean")
        mcd = read_json(os.path.join(out_dir, "mcd_whole.json"))
        if mcd:
            s = mcd["summary"]
            row["mcd_inj_vs_bl"] = (s.get("mcd_inj_vs_bl") or {}).get("mean")
            row["mcd_inj_vs_tgt"] = (s.get("mcd_inj_vs_tgt") or {}).get("mean")
            row["mcd_bl_vs_tgt"] = (s.get("mcd_bl_vs_tgt") or {}).get("mean")
            row["mcd_shift"] = s.get("shift_bl_tgt_minus_inj_tgt")
        table.append(row)
    return table


def render_md(args, table):
    lines = [f"# Llasa-3B layer sweep ({args.lang}, N={args.n}, seed={args.seed})", ""]
    lines.append("splice 対象 = model.model.layers[layer] の出力 (prefill のみ)。"
                 "BL / TGT は層に依存しないので全層で同一の音声を使っている。")
    lines.append("")
    lines.append("| layer | kappa | PIB% | +/-95%CI | n_valid | BL PER | INJ PER | dPER | "
                 "MCD(inj,bl) | MCD(inj,tgt) | MCD(bl,tgt) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def f(v, nd=4):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    for r in table:
        lines.append(
            f"| {r['layer']} | {f(r.get('kappa_at_splice'))} | {f(r.get('pib_pct'), 1)} | "
            f"{f(r.get('pib_ci95_half_pp'), 1)} | "
            f"{f(r.get('pib_n_valid'))} | {f(r.get('bl_per'))} | {f(r.get('inj_per'))} | "
            f"{f(r.get('delta_per'))} | {f(r.get('mcd_inj_vs_bl'), 3)} | "
            f"{f(r.get('mcd_inj_vs_tgt'), 3)} | {f(r.get('mcd_bl_vs_tgt'), 3)} |")
    lines.append("")
    best = [r for r in table if r.get("pib_pct") is not None]
    if best:
        b = max(best, key=lambda r: r["pib_pct"])
        lines.append(f"PIB 最良: layer {b['layer']} ({b['pib_pct']}%, n_valid={b['pib_n_valid']})")
        lines.append("")
        lines.append("注: PIB は inj_per < bl_per の厳密不等号で数えた割合。"
                     "偶然でも 50% 近辺は出るので、層間の差が小さい場合は "
                     "「どの層でも効かない」と読むこと。")
    return "\n".join(lines) + "\n"


def main():
    p = R.build_argparser()
    p.add_argument("--layers", default=None,
                   help=f"既定は Qwen3-TTS と同じ格子 {','.join(str(x) for x in L.SWEEP_LAYERS)}")
    p.add_argument("--sweep_wav_dir", default=None)
    p.add_argument("--sweep_out_dir", default=None)
    p.add_argument("--with_mcd", action="store_true",
                   help="層ごとに MCD も回す (--mode whole は CPU で重いので既定は PIB のみ)")
    # 掃引の既定 N は 50
    p.set_defaults(n=50)
    args = p.parse_args()

    if args.hf_home:
        L.setup_hf_home(args.hf_home)
    tag = f"{args.lang}_sweep"
    if args.sweep_wav_dir is None:
        args.sweep_wav_dir = os.path.join(PROJECT_ROOT, "eval_results", "_cc_wavs_a",
                                          f"llasa_{tag}_n{args.n}")
    if args.sweep_out_dir is None:
        args.sweep_out_dir = os.path.join(PROJECT_ROOT, "eval_results",
                                          f"llasa_layersweep_{args.lang}_n{args.n}")
    os.makedirs(args.sweep_wav_dir, exist_ok=True)
    os.makedirs(args.sweep_out_dir, exist_ok=True)
    os.chdir(PROJECT_ROOT)

    pairs = R.load_pairs(args.lang, args.n)
    print(f"[data] cross_culture lang={args.lang} n={len(pairs)}", flush=True)

    bundle = None
    if not args.skip_generation:
        bundle = R.load_bundle(args)
    layers = parse_layers(args.layers, bundle.num_layers if bundle else None)
    print(f"[sweep] layers={layers}", flush=True)

    t0 = time.time()
    first_dir = None
    per_layer_args = {}
    for i, layer in enumerate(layers):
        a = layer_args(args, layer, args.sweep_wav_dir, args.sweep_out_dir,
                       link_from=(None if i == 0 else first_dir))
        per_layer_args[layer] = a
        if first_dir is None:
            first_dir = a.wav_dir
        print(f"\n{'=' * 70}\n[sweep] layer {layer}  wav_dir={a.wav_dir}\n{'=' * 70}", flush=True)
        if not args.skip_generation:
            R.run_generation(a, pairs, bundle=bundle)

    if bundle is not None:
        del bundle
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[sweep] generation done in {(time.time() - t0) / 60:.1f} min", flush=True)

    if args.run_scorers:
        for layer in layers:
            a = per_layer_args[layer]
            print(f"\n[sweep] scoring layer {layer}", flush=True)
            if not args.with_mcd:
                # PIB だけ回す (MCD whole は CPU で重い)
                import subprocess
                env = dict(os.environ)
                env["REPO_ROOT"] = PROJECT_ROOT
                cmd = [sys.executable, R.PIB_SCORER, "--wav_dir", a.wav_dir,
                       "--lang", a.lang, "--n", str(a.n), "--output_json", a.pib_json]
                print(f"[scorer:PIB] {' '.join(cmd)}", flush=True)
                r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
                print(f"[scorer:PIB] exit={r.returncode}", flush=True)
            else:
                R.run_scorers(a)

    table = collect(args, layers, args.sweep_out_dir)
    R.atomic_json_dump({"lang": args.lang, "n": args.n, "seed": args.seed,
                        "layers": layers, "table": table,
                        "sweep_wav_dir": args.sweep_wav_dir,
                        "decode": L.GenConfig(
                            seed=args.seed, do_sample=args.do_sample,
                            temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, max_new_tokens=args.max_new_tokens).as_dict()},
                       os.path.join(args.sweep_out_dir, "layersweep_summary.json"))
    md = render_md(args, table)
    with open(os.path.join(args.sweep_out_dir, "layersweep_summary.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"[sweep] -> {args.sweep_out_dir}/layersweep_summary.md")


if __name__ == "__main__":
    main()
