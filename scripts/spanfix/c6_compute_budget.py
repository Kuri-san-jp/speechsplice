#!/usr/bin/env python3
"""C6 (2026-10-02 audit): GPU time behind the tables and appendices of naacl2027/main.tex (CPU only).

Sources
  * Slurm accounting: `sacct -u $USER -X` (read only).  Accounting for this user starts 2026-05-15 16:14,
    so runs before that come from the `summary.elapsed_min` of their results.json.
  * eval_results/_auditC_silence/auditC_scan_20261002_132335.json: the inventory of the speech sets
    behind the manuscript numbers (directories), used here to count the utterances.
  * results.json `summary.elapsed_min` for runs that did not go through Slurm accounting.

Attribution (curated, printed per item into the output JSON so that every job id can be audited)
  * A job counts for an item when its command runs that item's generation script AND names that item's
    final output directory (or it is listed by job id below).  Every attempt on the final output counts,
    including failed, cancelled and resumed attempts; runs whose output was superseded or is not in the
    paper (smoke tests, n=50 pilots, abandoned variants, the 5/20 App B EN run, earlier Bark JA sweep without
    the JA prompt, other layers of the Irodori/Kiseki sweeps) are not counted.
  * GPU time = ElapsedRaw x allocated GPUs (all jobs here used one GPU).
  * Utterances = wav files newly written for the item (unique inodes not already counted by an earlier
    item; BL/TGT reused through hard links are counted once), or the documented count for runs that kept
    no wav.
  * Scoring = other jobs whose script is a scorer and whose command names a paper speech directory.

Output: eval_results/_c6_compute/compute_budget_<timestamp>.json
Usage:  taskset -c 0-12 python3 scripts/spanfix/c6_compute_budget.py
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "13"
sys.modules["pyopenjtalk"] = None

import glob
import json
import re
import statistics
import subprocess
import time
from collections import defaultdict

ROOT = os.environ.get("REPO_ROOT", "/mnt")
ER = os.path.join(ROOT, "eval_results")
OUT_DIR = os.path.join(ER, "_c6_compute")
AUDIT = os.path.join(ER, "_auditC_silence", "auditC_scan_20261002_132335.json")
FIELDS = ["JobID", "JobName", "Submit", "Start", "End", "ElapsedRaw", "State", "AllocTRES",
          "NodeList", "WorkDir", "SubmitLine"]


# ------------------------------------------------------------------ sacct
def read_sacct():
    cmd = ["sacct", "-u", "kiyoshi.kurihara", "-S", "2026-01-01", "-E", "now", "-X", "-n", "--parsable2",
           "--format=" + ",".join(FIELDS)]
    txt = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    recs, cur = [], None
    for line in txt.split("\n"):
        if re.match(r"^\d+\|", line) and line.count("|") >= len(FIELDS) - 1:
            if cur is not None:
                recs.append(cur)
            cur = line
        elif cur is not None:
            cur += "\n" + line
    if cur is not None:
        recs.append(cur)
    jobs = []
    for rec in recs:
        d = dict(zip(FIELDS, rec.split("|", len(FIELDS) - 1)))
        if not d["WorkDir"].startswith(ROOT):
            continue
        sl = d.pop("SubmitLine")
        m = re.search(r"--gpus=(\d+)", sl)
        d["gpus"] = int(m.group(1)) if m else 0
        c = sl.split("\n -- ", 1)[1] if "\n -- " in sl else (sl.split(" -- ", 1)[1] if " -- " in sl else sl)
        d["cmd"] = re.sub(r"\s+", " ", c).strip()
        d["ElapsedRaw"] = int(d["ElapsedRaw"] or 0)
        d["gpu_s"] = d["ElapsedRaw"] * d["gpus"]
        jobs.append(d)
    return cmd, jobs


def D(path):
    """regex: an eval_results directory named exactly (not as a prefix of a longer name)."""
    return r"(?:^|[\s/=])(?:/mnt/)?(?:eval_results/)?" + re.escape(path) + r"(?=$|[\s/\"'])"


# ------------------------------------------------------------------ items
# (key, group, sweep, script regex, dir regex or None, explicit job ids, utterance dirs, fixed utts, elapsed extras)
def items():
    I = []

    def add(key, group, script=None, dirs=None, ids=(), wav_dirs=(), utts=None, extra=(), sweep=False, note=""):
        I.append(dict(key=key, group=group, sweep=sweep, script=script, dirs=dirs or [], ids=list(ids),
                      wav_dirs=list(wav_dirs), utts=utts, extra=list(extra), note=note))

    # Table 1 (word replacement sets) and the listening-test stimuli
    add("T1_EN", "Table 1/2", r"eval_utmos_speaker_fixed\.py", ["en_wordrepl100_pairs_gen"], ids=[21080],
        wav_dirs=["en_wordrepl100_pairs_gen"], note="21080 wrote the first 100 pairs to the default dir name")
    add("T1_EN_splice16", "Table 1/2", r"inj_splice_generic\.py", ["en_wordrepl300_splice_inj"],
        wav_dirs=["en_wordrepl300_splice_inj"])
    add("T1_JA", "Table 1/2", r"eval_utmos_speaker_fixed\.py", ["ja_wordrepl300_gen"], wav_dirs=["ja_wordrepl300_gen"])
    add("T1_ZH", "Table 1/2", r"eval_utmos_speaker_fixed\.py", ["zh_wordrepl300_gen", "_zh_regen_check"],
        wav_dirs=["zh_wordrepl300_gen"])
    add("T2_VoiceCraft", "Table 1/2", r"tts_entity_en_n100\.py", ["voicecraft_tts_wordrepl100"],
        wav_dirs=["voicecraft_tts_wordrepl100"])
    add("T2_EdiTTS", "Table 1/2", None, None, ids=[21391], utts=800,
        note="run_editts_wr_p1.sh wrote pairs 100-299 (4 wav per pair = 800 of the 1,200 files); the job that wrote "
             "pairs 0-99 of this directory is not identifiable and is not counted")
    add("HE_stimuli", "Listening-test stimuli", r"(mos50_|ja_mos_splice_regen|entity_redraw_failed|entity_pool_gate)", None,
        ids=[21811, 21813], utts=306,
        note="MOS 50-pair sets (3 x 100 clips) and the replacement AB item 7001; gate/redraw jobs attributed by script name")
    # Table 3 (a): replacement point
    t3a = [("t5gemma_en", [14878]), ("t5gemma_ja", [14880]), ("t5gemma_zh", [14881]), ("speecht5_en", [14882]),
           ("longcat_en_cfg4_cfg_both", [27030]), ("longcat_zh_cfg4_cfg_both", [27052, 27451]),
           ("irodori_ja", [27214]), ("kiseki_ja", [27257]), ("qwen3_en", [14875]), ("qwen3_ja", [14876]),
           ("qwen3_zh", [14877]), ("xtts_en", [27271, 27281]), ("csm_en", [27338, 27391, 27395]),
           ("bark_en", [27398]), ("bark_ja", [27413]), ("llasa_en", [27331, 27333]), ("llasa_zh", [27418, 27446])]
    for name, ids in t3a:
        add("T3a_" + name, "Table 3 replacement point", None, None, ids=ids, wav_dirs=["_cc_wavs_a/" + name])
    add("T3a_llasa_regen_G5a", "Table 3 replacement point", None, None, ids=[28498], utts=0,
        note="9/29 seed-43 regeneration of Llasa-3B pairs (a)+(b); counted here, utterances already in the dirs")
    # Table 3 (b): layer sweep
    sw = [("t5gemma_en", r"eval_en_crossculture_t5gemma_layer\.py", ["_layerabl/t5gemma_en_L%d" % l for l in (0, 7, 14, 21, 25)], [27230, 27228]),
          ("t5gemma_ja", r"eval_en_crossculture_t5gemma_layer\.py", ["_layerabl/t5gemma_ja_L%d" % l for l in (0, 7, 14, 21, 25)], []),
          ("t5gemma_zh", r"eval_en_crossculture_t5gemma_layer\.py", ["_layerabl/t5gemma_zh_L%d" % l for l in (0, 7, 14, 21, 25)], []),
          ("speecht5_en", r"(eval_speecht5_layer|eval_speecht5_crossculture_fixed_v2)\.py", ["_layerabl/s5_en_L%d" % l for l in (1, 3, 6, 9)] + ["_layerabl/s5_en_today"], []),
          ("longcat_en", r"eval_longcat_layer\.py", ["_layerabl/lc_en_L%d" % l for l in (1, 3, 6, 9, 12)], []),
          ("longcat_zh", r"eval_longcat_layer\.py", ["_layerabl/lc_zh_L%d" % l for l in (1, 3, 6, 9, 12)], []),
          ("irodori_ja", r"eval_irodori_layer\.py", ["_layerabl/ir_ja_L%d_v2" % l for l in (1, 7, 13, 19, 25)], []),
          ("kiseki_ja", r"eval_kiseki_layer\.py", ["_layerabl/ki_ja_L%d" % l for l in (1, 3, 6, 9, 12)], []),
          ("qwen3_en", r"eval_qwen3_crossculture_acoustic_fixed\.py", ["_layerabl/qwen3_en_L%d" % l for l in (0, 9, 13, 20, 26)], []),
          ("qwen3_ja", r"eval_qwen3_crossculture_acoustic_fixed\.py", ["_layerabl/qwen3_ja_L%d" % l for l in (0, 9, 13, 20, 26)], []),
          ("qwen3_zh", r"eval_qwen3_crossculture_acoustic_fixed\.py", ["_layerabl/qwen3_zh_L%d" % l for l in (0, 9, 13, 20, 26)], []),
          ("csm_en", r"run_csm_eval\.py", ["_layerabl/csm_en_L%d" % l for l in (0, 3, 7, 10, 14)], []),
          ("bark_en", r"run_bark_eval\.py", ["_layerabl/bark_en_L%d" % l for l in (0, 5, 11, 16, 22)], []),
          ("bark_ja", r"run_bark_eval\.py", ["_layerabl/bark_ja_jasp_L%d" % l for l in (0, 5, 11, 16, 22)], [])]
    for name, script, dirs, ids in sw:
        wd = [d for d in dirs]
        if name == "t5gemma_en":
            wd = ["_cc_wavs_b/t5gemma_en"] + wd
        add("T3b_" + name, "Table 3 layer sweep", script, dirs, ids=ids, wav_dirs=wd, sweep=True)
    add("T3b_xtts_en", "Table 3 layer sweep", None, None, ids=[27611, 27613], sweep=True,
        wav_dirs=["_cc_wavs_a/xtts_en_sweep_n300/L%02d" % l for l in (0, 8, 15, 23, 28)],
        note="run_xtts_layersweep.py --n 300 (default depths); the 9/19 run with depths 0,14,28 is not counted")
    add("T3b_llasa_en", "Table 3 layer sweep", None, None, ids=[27608], sweep=True,
        wav_dirs=["_cc_wavs_a/llasa_en_sweep_n300/L%02d" % l for l in (0, 7, 14, 21, 26)],
        note="run_llasa_layersweep.py --lang en --n 300 (default depths); the 9/19 run with depths 0,13,26 is not counted")
    add("T3b_llasa_zh", "Table 3 layer sweep", None, None, ids=[27609, 28103, 28109], sweep=True,
        wav_dirs=["_cc_wavs_a/llasa_zh_sweep_n300/L%02d" % l for l in (0, 7, 14, 21, 26)])
    # Table 3 (c): two-entity sentences
    me = [("multientity_en_300", r"generate_multientity(_ext)?\.py", ["multientity_en_300"], [28500, 28594]),
          ("multientity_ja_300", r"generate_multientity(_ext)?\.py", ["multientity_ja_300"], []),
          ("multientity_zh_v2_300", r"generate_multientity(_ext)?\.py", ["multientity_zh_v2_300"], []),
          ("multientity_speecht5_en_300", r"run_multientity_speecht5\.py", ["multientity_speecht5_en_300"], []),
          ("multientity_longcat_en_300", r"(run_multientity_longcat|gen_multientity_targets_longcat)\.py", ["multientity_longcat_en_300"], []),
          ("multientity_longcat_zh_300_ext", r"(run_multientity_longcat|gen_multientity_targets_longcat)\.py", ["multientity_longcat_zh_300_ext"], []),
          ("multientity_irodori_ja_300", r"run_multientity_irodori\.py", ["multientity_irodori_ja_300"], []),
          ("multientity_kiseki_ja_300", r"run_multientity_kiseki\.py", ["multientity_kiseki_ja_300"], []),
          ("multientity_qwen3_en_300", r"run_multientity_qwen3\.py", ["multientity_qwen3_en_300"], []),
          ("multientity_qwen3_ja_300", r"run_multientity_qwen3\.py", ["multientity_qwen3_ja_300"], []),
          ("multientity_qwen3_zh_v2_300", r"run_multientity_qwen3\.py", ["multientity_qwen3_zh_v2_300"], []),
          ("multientity_xtts_en_300", r"run_multientity_xtts\.py", ["multientity_xtts_en_300"], []),
          ("multientity_llasa_en_300", r"run_multientity_llasa\.py", ["multientity_llasa_en_300"], []),
          ("multientity_llasa_zh_300", r"run_multientity_llasa\.py", ["multientity_llasa_zh_300"], []),
          ("csm_multientity_en_300", r"(run_multientity_csm|gen_multientity_targets_csm)\.py", ["csm_multientity_en_300"], []),
          ("bark_multientity_en_300", r"run_multientity_bark\.py", ["bark_multientity_en_300"], []),
          ("bark_multientity_ja_300", r"run_multientity_bark\.py", ["bark_multientity_ja_300"], [])]
    for name, script, dirs, ids in me:
        add("T3c_" + name, "Table 3 two-entity", script, dirs, ids=ids, wav_dirs=dirs)
    # Appendices
    add("AppB_EN_spanfix", "App A-C (cross-attention)", r"spanfix_appB_en_gen\.py", None, wav_dirs=[],
        utts=None, note="utterances = wav under _spanfix_appB_en/seed4*/")
    add("AppB_JA_and_AppA", "App A-C (cross-attention)", None, None, ids=[15518], utts=1500,
        note="5/20 length-preserve run; no wav kept (300 pairs x 5 conditions)")
    add("AppB_ZH", "App A-C (cross-attention)", None, None, ids=[15520, 28504], utts=1500,
        note="5/20 length-preserve run + 9/29 pair 42 seed 43; no wav kept for the 5/20 run")
    add("AppC_EN_spanfix", "App A-C (cross-attention)", r"appc_en_gen\.py", None, utts=None,
        note="utterances = wav under _spanfix_appC_en/seed4*/")
    add("AppC_JA", "App A-C (cross-attention)", None, None, utts=1500,
        extra=[("eval_results/crossattn_entity_per_ja_n300/results.json", "5/11 run before Slurm accounting")])
    add("AppE_bench", "App E-F (same-surface)", None, None, ids=[21103, 21104, 21105, 21441, 21450, 21459],
        wav_dirs=["bench_hetnum77_gen", "bench_ext83_gen"],
        note="BL/reference generation (numerals_verify_generic) and splice_clone INJ; other INJ variants not counted")
    add("AppG", "App G (baselines)", r"appg_vc_gen\.py", None, ids=[],
        wav_dirs=["_appg_vc/splice/seed42", "_appg_vc/g2p/seed42", "_appg_vc/rome/seed42", "baseline_text_edit_ja_n300/wavs"],
        extra=[("eval_results/baseline_text_edit_ja_n300/results.json", "Text Hint 5/15 run, not in Slurm accounting")],
        note="smoke run (--output_root _appg_vc_smoke) excluded")
    add("AppK", "App K (six languages)", r"(eval_utmos_speaker_fixed|eval_cjk_n300|eval_cjk_n300_zhv2|appk_regen_pairs_seed)\.py",
        ["utmos_speaker_fixed_en_n300_splice", "utmos_speaker_fixed_fr_n300_splice", "utmos_speaker_fixed_de_n300_splice",
         "utmos_speaker_fixed_ko_n300_splice", "cjk_n300_rewav", "cjk_n300_zh_v2"], ids=[28497],
        wav_dirs=["utmos_speaker_fixed_en_n300_splice", "utmos_speaker_fixed_fr_n300_splice",
                  "utmos_speaker_fixed_de_n300_splice", "utmos_speaker_fixed_ko_n300_splice",
                  "cjk_n300_rewav", "cjk_n300_zh_v2"])
    return I


EXCLUDE_ALWAYS = {
    27040: "LongCat --adaln_source variant (not in the paper)",
    27535: "Llasa sweep with depths 0,13,26 (superseded)",
    27543: "XTTS sweep with depths 0,14,28 (superseded)",
    28515: "App G smoke run",
}

SCORE_RX = re.compile(r"(phoneme_per|mcd_|per_fix|score_|utmos_table|_judge|judge_|whisper|furigana|sim_to_prompt|"
                      r"speaker_sim|_metrics|metrics\.py|kappa|rejudge|transcribe|_check|verify|aggregate|ctc|"
                      r"pinyin_asr|run_v1_multientity|editts100_utmos|ipa|delta_sim)", re.I)


def wav_inodes(rel):
    p = os.path.join(ER, rel)
    out = set()
    for f in glob.glob(os.path.join(p, "**", "*.wav"), recursive=True):
        try:
            st = os.stat(f)  # follows symlinks, so a view link and its target count once
        except FileNotFoundError:  # broken view link
            continue
        out.add((st.st_dev, st.st_ino))
    return out


def main():
    t0 = time.time()
    query, jobs = read_sacct()
    gpu_jobs = [j for j in jobs if j["gpus"]]
    by_id = {int(j["JobID"]): j for j in gpu_jobs}
    assigned = {}
    seen_inodes = set()
    rows = []
    for it in items():
        matched = []
        for j in gpu_jobs:
            jid = int(j["JobID"])
            if jid in EXCLUDE_ALWAYS:
                continue
            ok = jid in it["ids"]
            if not ok and it["script"] and re.search(it["script"], j["cmd"]):
                if it["dirs"]:
                    ok = any(re.search(D(d), j["cmd"]) for d in it["dirs"])
                elif it["key"].startswith(("AppB_EN", "AppC_EN", "HE_", "AppG")):
                    ok = True
            if ok:
                matched.append(jid)
        dup = [jid for jid in matched if jid in assigned]
        for jid in matched:
            assigned.setdefault(jid, it["key"])
        gpu_s = sum(by_id[jid]["gpu_s"] for jid in matched if assigned[jid] == it["key"])
        extra_s = 0.0
        extra_src = []
        for path, why in it["extra"]:
            d = json.load(open(os.path.join(ROOT, path)))
            em = float(d["summary"]["elapsed_min"])
            extra_s += em * 60
            extra_src.append({"file": path, "elapsed_min": em, "why": why})
        if it["utts"] is not None:
            n_utt = it["utts"]
        elif it["key"] == "AppB_EN_spanfix":
            inos = set().union(*(wav_inodes(os.path.relpath(p, ER)) for p in glob.glob(os.path.join(ER, "_spanfix_appB_en", "seed4*"))))
            n_utt = len(inos - seen_inodes)
            seen_inodes |= inos
        elif it["key"] == "AppC_EN_spanfix":
            inos = set().union(*(wav_inodes(os.path.relpath(p, ER)) for p in glob.glob(os.path.join(ER, "_spanfix_appC_en", "seed4*"))))
            n_utt = len(inos - seen_inodes)
            seen_inodes |= inos
        else:
            inos = set()
            for w in it["wav_dirs"]:
                inos |= wav_inodes(w)
            n_utt = len(inos - seen_inodes)
            seen_inodes |= inos
        total_s = gpu_s + extra_s
        rows.append({"key": it["key"], "group": it["group"], "sweep": it["sweep"], "jobs": matched,
                     "jobs_also_matched_elsewhere": dup, "gpu_h_slurm": round(gpu_s / 3600, 3),
                     "gpu_h_from_elapsed_min": round(extra_s / 3600, 3), "elapsed_min_sources": extra_src,
                     "gpu_h": round(total_s / 3600, 3), "utterances": n_utt,
                     "sec_per_utt": round(total_s / n_utt, 2) if n_utt else None, "note": it["note"]})

    gen_ids = set(assigned)
    paper_dirs = set()
    for it in items():
        paper_dirs.update(it["dirs"])
        paper_dirs.update(w.split("/wavs")[0] for w in it["wav_dirs"])
    score_jobs = [j for j in gpu_jobs if int(j["JobID"]) not in gen_ids and SCORE_RX.search(j["cmd"])
                  and any(re.search(D(d), j["cmd"]) for d in paper_dirs)]
    total_gen_h = sum(r["gpu_h"] for r in rows)
    sweep_h = sum(r["gpu_h"] for r in rows if r["sweep"])
    groups = defaultdict(float)
    for r in rows:
        groups[r["group"]] += r["gpu_h"]
    utt_total = sum(r["utterances"] for r in rows)
    # utterance-weighted median of the per-run seconds per utterance
    per_utt = []
    for r in rows:
        if r["utterances"] and r["sec_per_utt"] is not None:
            per_utt.extend([r["sec_per_utt"]] * r["utterances"])
    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/spanfix/c6_compute_budget.py",
        "sacct_query": " ".join(query),
        "sacct_first_job_start_TTS26": min(j["Start"] for j in jobs),
        "sacct_last_job_start_TTS26": max(j["Start"] for j in jobs if j["Start"] != "Unknown"),
        "all_TTS26_gpu_jobs": {"n": len(gpu_jobs), "gpu_h": round(sum(j["gpu_s"] for j in gpu_jobs) / 3600, 2)},
        "generation_behind_paper": {
            "gpu_h": round(total_gen_h, 2), "n_jobs": len(gen_ids), "utterances": utt_total,
            "mean_sec_per_utt": round(total_gen_h * 3600 / utt_total, 2),
            "median_sec_per_utt_utt_weighted": round(statistics.median(per_utt), 2),
            "layer_sweep_gpu_h": round(sweep_h, 2), "layer_sweep_share": round(sweep_h / total_gen_h, 4),
            "by_group_gpu_h": {k: round(v, 2) for k, v in sorted(groups.items(), key=lambda x: -x[1])}},
        "scoring_jobs_on_paper_speech": {"n": len(score_jobs), "gpu_h": round(sum(j["gpu_s"] for j in score_jobs) / 3600, 2),
                                         "share_of_generation": round(sum(j["gpu_s"] for j in score_jobs) / 3600 / total_gen_h, 4)},
        "excluded_job_ids": {str(k): v for k, v in EXCLUDE_ALWAYS.items()},
        "items": rows,
        "scoring_job_ids": [int(j["JobID"]) for j in score_jobs],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"compute_budget_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if os.path.exists(path):
        raise SystemExit(f"[fatal] {path} exists")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    for r in rows:
        print(f"{r['key'][:34]:34s} {r['gpu_h']:7.2f} h  utt={r['utterances']:6d}  s/utt={r['sec_per_utt']}  jobs={len(r['jobs'])}"
              + (f"  DUP={r['jobs_also_matched_elsewhere']}" if r['jobs_also_matched_elsewhere'] else ""))
    print(json.dumps({k: out[k] for k in ("all_TTS26_gpu_jobs", "generation_behind_paper", "scoring_jobs_on_paper_speech",
                                          "sacct_first_job_start_TTS26")}, ensure_ascii=False, indent=1))
    print(f"[written] {os.path.relpath(path, ROOT)} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
