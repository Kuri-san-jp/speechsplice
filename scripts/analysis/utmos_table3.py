"""UTMOS22 for every audio sample that appears in Table 3 of the NAACL manuscript
(2026-09-26, user request: "Table3に出ている全ての試料について、utmosを計測して").

Three blocks of Table 3, all N=300:
  main   17 cells of "Our replacement point": _cc_wavs_a/<cell>/s*_{bl,inj,tgt}.wav
  sweep  every swept l* point: the wav_dir recorded in each _v1L_*.json that feeds
         the "PIB over swept l*" column (s*_{inj,bl,tgt}.wav when present)
  multi  17 two-entity cells: multientity dirs, s*_{bl,inj1,inj2,inj12,tgt1,tgt2}.wav
         (tgt1 / tgt2 are the direct-synthesis ceiling audio)

Scoring is the one used for Tables 1/2 (scripts/projection/eval_utmos_speaker_fixed.py
score_wav): mono, resample to 16 kHz, tarepan/SpeechMOS:v1.2.0 utmos22_strong.
Per-file scores are cached per group, so the job can be restarted and re-run after the
new l* grid points (scripts/per_fix/run_lstar_grid5.sh) finish; hard-linked files are
scored once. A sweep point is scored only when all 300 injected wavs exist.

GPU job (slurm_run.sh maps the project to /mnt):
  SLURM_TIME=08:00:00 slurm_run.sh python3 -u scripts/analysis/utmos_table3.py
Outputs: eval_results/_utmos_table3/<block>__<group>.json and _summary.json
"""
import glob
import json
import math
import os
import re
import sys
import time

THREADS = int(os.environ.get("UTMOS_THREADS", "13"))  # CPU cap: 13 per job (40 total)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))

EV = os.environ.get("EV_ROOT", "/mnt/eval_results")
OUT = os.path.join(EV, "_utmos_table3")
N = 300

MAIN = {
    "T5Gemma-TTS_EN": "_cc_wavs_a/t5gemma_en",
    "T5Gemma-TTS_JA": "_cc_wavs_a/t5gemma_ja",
    "T5Gemma-TTS_ZH": "_cc_wavs_a/t5gemma_zh",
    "SpeechT5_EN": "_cc_wavs_a/speecht5_en",
    "LongCat_EN": "_cc_wavs_a/longcat_en_cfg4_cfg_both",
    "LongCat_ZH": "_cc_wavs_a/longcat_zh_cfg4_cfg_both",
    "Irodori-TTS_JA": "_cc_wavs_a/irodori_ja",
    "Kiseki-TTS_JA": "_cc_wavs_a/kiseki_ja",
    "Qwen3-TTS_EN": "_cc_wavs_a/qwen3_en",
    "Qwen3-TTS_JA": "_cc_wavs_a/qwen3_ja",
    "Qwen3-TTS_ZH": "_cc_wavs_a/qwen3_zh",
    "XTTS-v2_EN": "_cc_wavs_a/xtts_en",
    "Llasa-3B_EN": "_cc_wavs_a/llasa_en",
    "Llasa-3B_ZH": "_cc_wavs_a/llasa_zh",
    "CSM-1B_EN": "_cc_wavs_a/csm_en",
    "Bark_EN": "_cc_wavs_a/bark_en",
    "Bark_JA": "_cc_wavs_a/bark_ja",
}

# (cell, score-json template, file indices, index -> manuscript l*)
SWEEP = [
    ("T5Gemma-TTS_EN", "_v1L_t5gemma_en_L{}.json", [0, 7, 14, 21, 25], 1),
    ("T5Gemma-TTS_JA", "_v1L_t5gemma_ja_L{}.json", [0, 7, 14, 21, 25], 1),
    ("T5Gemma-TTS_ZH", "_v1L_t5gemma_zh_L{}.json", [0, 7, 14, 21, 25], 1),
    ("SpeechT5_EN", "_v1L_s5_en_L{}.json", list(range(1, 12)), 0),
    ("LongCat_EN", "_v1L_lc_en_L{}.json", [1, 3, 6, 9, 12], 0),
    ("LongCat_ZH", "_v1L_lc_zh_L{}.json", [1, 3, 6, 9, 12], 0),
    ("Irodori-TTS_JA", "_v1L_ir_ja_L{}.json", list(range(1, 25)), 0),
    ("Kiseki-TTS_JA", "_v1L_ki_ja_L{}.json", list(range(1, 12)), 0),
    ("Qwen3-TTS_EN", "_v1L_qwen3_en_L{}.json", [0, 9, 13, 20, 26], 1),
    ("Qwen3-TTS_JA", "_v1L_qwen3_ja_L{}.json", [0, 9, 13, 20, 26], 1),
    ("Qwen3-TTS_ZH", "_v1L_qwen3_zh_L{}.json", [0, 9, 13, 20, 26], 1),
    ("XTTS-v2_EN", "_v1L_xtts_en_L{:02d}.json", [0, 8, 15, 23, 28], 1),
    ("Llasa-3B_EN", "_v1L_llasa_en_L{:02d}.json", [0, 7, 14, 21, 26], 1),
    ("Llasa-3B_ZH", "_v1L_llasa_zh_L{:02d}_n300.json", [0, 7, 14, 21, 26], 1),
    ("CSM-1B_EN", "_v1L_csm_en_L{}.json", [0, 3, 7, 10, 14], 1),
    ("Bark_EN", "_v1L_bark_en_L{}.json", [0, 5, 11, 16, 22], 1),
    ("Bark_JA", "_v1L_bark_ja_jasp_L{}.json", [0, 5, 11, 16, 22], 1),
]

MULTI = {
    "T5Gemma-TTS_EN": "multientity_en_300",
    "T5Gemma-TTS_JA": "multientity_ja_300",
    "T5Gemma-TTS_ZH": "multientity_zh_v2_300",
    "SpeechT5_EN": "multientity_speecht5_en_300",
    "LongCat_EN": "multientity_longcat_en_300",
    "LongCat_ZH": "multientity_longcat_zh_300_ext",
    "Irodori-TTS_JA": "multientity_irodori_ja_300",
    "Kiseki-TTS_JA": "multientity_kiseki_ja_300",
    "Qwen3-TTS_EN": "multientity_qwen3_en_300",
    "Qwen3-TTS_JA": "multientity_qwen3_ja_300",
    "Qwen3-TTS_ZH": "multientity_qwen3_zh_v2_300",
    "XTTS-v2_EN": "multientity_xtts_en_300",
    "Llasa-3B_EN": "multientity_llasa_en_300",
    "Llasa-3B_ZH": "multientity_llasa_zh_300",
    "CSM-1B_EN": "csm_multientity_en_300",
    "Bark_EN": "bark_multientity_en_300",
    "Bark_JA": "bark_multientity_ja_300",
}

WAV_RE = re.compile(r"^s(\d+)_([a-z0-9]+)\.wav$")


def host_path(p):
    """Score jsons record container paths (/mnt/eval_results/...)."""
    return p.replace("/mnt/eval_results", EV, 1) if p.startswith("/mnt/eval_results") else p


def list_wavs(d, kinds):
    out = {}
    for f in sorted(os.listdir(d)):
        m = WAV_RE.match(f)
        if m and m.group(2) in kinds:
            out.setdefault(m.group(2), []).append((int(m.group(1)), os.path.join(d, f)))
    return out


def build_groups():
    groups, missing = [], []
    for cell, rel in MAIN.items():
        d = os.path.join(EV, rel)
        if os.path.isdir(d):
            groups.append(("main", cell, d, ("bl", "inj", "tgt"), None))
        else:
            missing.append(("main", cell, d))
    for cell, tmpl, idxs, off in SWEEP:
        for k in idxs:
            j = os.path.join(EV, tmpl.format(k))
            if not os.path.exists(j):
                missing.append(("sweep", cell, j))
                continue
            d = host_path(json.load(open(j))["summary"]["wav_dir"])
            n_inj = len(glob.glob(os.path.join(d, "s*_inj.wav")))
            if n_inj < N:
                missing.append(("sweep", cell, "%s (inj=%d)" % (d, n_inj)))
                continue
            groups.append(("sweep", "%s_l%d" % (cell, k + off), d, ("inj", "bl", "tgt"), k + off))
    for cell, rel in MULTI.items():
        d = os.path.join(EV, rel)
        if os.path.isdir(d):
            groups.append(("multi", cell, d, ("bl", "inj1", "inj2", "inj12", "tgt1", "tgt2"), None))
        else:
            missing.append(("multi", cell, d))
    return groups, missing


def stats(v):
    v = [x for x in v if x is not None and not math.isnan(x)]
    n = len(v)
    if n == 0:
        return {"n": 0}
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
            "ci95_half": round(1.96 * sd / math.sqrt(n), 4) if n > 1 else None}


def paired(a, b):
    """mean of b - a over items present in both (dicts item -> score)."""
    keys = sorted(set(a) & set(b))
    return stats([b[k] - a[k] for k in keys])


def main():
    import torch
    import torchaudio
    torch.set_num_threads(THREADS)
    os.makedirs(OUT, exist_ok=True)
    groups, missing = build_groups()
    print("[Info] groups=%d missing=%d" % (len(groups), len(missing)), flush=True)
    for m in missing:
        print("  [skip]", *m, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Info] device", device, "- loading UTMOS22 strong", flush=True)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True).to(device).eval()

    inode_cache = {}   # (dev, ino) -> score, so hard links are scored once
    summary = {"model": "tarepan/SpeechMOS:v1.2.0 utmos22_strong", "sr": 16000,
               "groups": {}, "skipped": [" | ".join(map(str, m)) for m in missing]}
    t0 = time.time()
    done_files = 0
    for block, gid, d, kinds, lstar in groups:
        out_json = os.path.join(OUT, "%s__%s.json" % (block, gid))
        prev = json.load(open(out_json)) if os.path.exists(out_json) else {}
        scores = prev.get("scores", {})   # kind -> {item: score}
        wavs = list_wavs(d, kinds)
        n_new = 0
        for kind, items in wavs.items():
            ks = scores.setdefault(kind, {})
            for idx, path in items:
                key = "%04d" % idx
                if key in ks:
                    continue
                st = os.stat(path)
                ik = (st.st_dev, st.st_ino)
                if ik in inode_cache:
                    ks[key] = inode_cache[ik]
                    continue
                wav, sr = torchaudio.load(path)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                with torch.no_grad():
                    u = float(utmos(wav.to(device), 16000).item())
                ks[key] = inode_cache[ik] = u
                n_new += 1
                done_files += 1
        res = {"block": block, "group": gid, "wav_dir": d, "l_star": lstar,
               "scores": scores,
               "summary": {k: stats(list(v.values())) for k, v in scores.items()}}
        if "bl" in scores:
            for k in ("inj", "inj1", "inj2", "inj12", "tgt", "tgt1", "tgt2"):
                if k in scores:
                    res["summary"]["%s_minus_bl" % k] = paired(scores["bl"], scores[k])
        json.dump(res, open(out_json, "w"), indent=1)
        summary["groups"]["%s__%s" % (block, gid)] = res["summary"]
        el = time.time() - t0
        print("[%s] %-6s %-26s new=%5d  total_new=%6d  %.0fs  %s" % (
            time.strftime("%H:%M:%S"), block, gid, n_new, done_files, el,
            {k: v.get("mean") for k, v in res["summary"].items() if "_minus_" not in k}),
            flush=True)
    json.dump(summary, open(os.path.join(OUT, "_summary.json"), "w"), indent=1)
    print("[Done] files scored this run:", done_files, flush=True)


if __name__ == "__main__":
    if "--list" in sys.argv:          # CPU dry run: show what would be scored
        g, m = build_groups()
        for x in g:
            print(x[0], x[1], x[2])
        print("groups", len(g), "missing", len(m))
        for x in m:
            print("  [skip]", *x)
    else:
        main()
