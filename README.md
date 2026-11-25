# SpeechSplice

Code, benchmark item lists, listening-test materials, and pre-computed
evaluation outputs for **SpeechSplice: Pronunciation Control in Encoder–Decoder
LLM-based TTS by Replacing Encoder Hidden States** (EACL 2027).

SpeechSplice controls pronunciation in an encoder-decoder LLM-based TTS system
without training and without touching model weights: it replaces the
entity-position hidden states at the encoder output with pronunciation-bearing
vectors taken from a target-reading forward pass.

**Authors.** Kiyoshi Kurihara, Jumpei Ukita, Yusuke Iwasawa, and Yutaka Matsuo
(Matsuo Institute, Inc. / The University of Tokyo).
Contact: `{kiyoshi.kurihara, jumpei.ukita, iwasawa, matsuo}@weblab.t.u-tokyo.ac.jp`

**Audio samples:** [`docs/`](docs/) holds a static page with 40 baseline /
SpeechSplice pairs that the listeners in the paper heard.

Table and appendix numbers below follow the camera-ready version of the paper.
Every experiment of the paper can be checked at two levels:

- **From the shipped per-pair outputs, on a CPU** (§4): every value of every table,
  in a few seconds.
- **From scratch, on a GPU** (`REPRODUCE.md`): the generation, scoring and aggregation
  code that produced those outputs, with the commands, containers, model revisions,
  voice prompts and random seeds of the original runs.

---

## 1. What is in this repository

| Path | Contents |
|---|---|
| `scripts/` | All code of the experiments, in the folder layout in which it ran (`projection/`, `analysis/`, `layer_ablation/`, one folder per TTS system, `spanfix/` and `regen2/` for the reruns and re-seeded pairs, ...), plus the table aggregators `scripts/aggregate_*.py` and the setup scripts (`env.sh`, `setup_third_party.sh`, `download_models.py`, `prepare_voice_prompts.py`) |
| `src/` | The scorers shared by many scripts: phoneme PER / PIB (`phoneme_per.py`), MCD (`mcd_pysptk.py`), and the TRM / G2P-CER scorer of Table A5; `src/_vendor/t5gemma_tts/` is the T5Gemma-TTS inference code (see `NOTICE`) |
| `third_party/` | Vendored model code: LongCat-AudioDiT and whisper-pinyin (see `NOTICE`) |
| `docker/` | The container definitions of the runs and the `pip freeze` of the main image |
| `data/` | **Benchmark item lists**: the 11 sets of 3,500 sentence pairs, the two-entity sentence sets, the Table A3 benchmark items (`data/external/`), and the six-language same-surface test item lists |
| `configs/` | Experiment configuration (`table_specs.yaml`, `voice_prompts.yaml`, `pair_seeds.csv`) |
| `eval_results/` | **Pre-computed per-pair evaluation outputs**, one directory per table (`table1_*`, `table2_*`, `table3_*`, `table_a*`, `appendix_i_compute`, and `supp_six_language` for the six-language same-surface test, which is not in the paper). Each `manifest.json` names the experiment file a copy came from and its MD5 |
| `human_eval/` | **Listening-test materials and results**: stimulus manifests, verbatim instructions in three languages, interface screenshots, item-level and aggregate results, and the pre-specified analysis code. See [`human_eval/README.md`](human_eval/README.md) |
| `docs/` | The audio sample page (`index.html`) and its clips, under the terms in `docs/LICENSE_AUDIO.md` |
| `REPRODUCE.md` | Commands to rerun every experiment on a GPU |
| `requirements-cpu.txt` | The packages needed for the GPU-free reproduction (§4) |
| `requirements.txt` | The main packages of the GPU environment; `docker/my_lab_image_freeze.txt` pins all of them |

## 2. Newly released resources

These artifacts are published here for the first time and are usable
independently of our method.

1. **A word-replacement pronunciation benchmark: 300 sentence pairs per
   language in English, Japanese, and Chinese** (900 pairs). Each language is
   50 hand-written source/target word pairs crossed with 6 sentence frames.
   Every pair carries `word_a`, `word_b`, `text_a`, `text_b`, and (for JA/ZH)
   the target reading. Built by `data/{en,ja,zh}_wordrepl300_pairs.py`.
   These are usable as a test set for any pronunciation- or
   entity-control method, not only for encoder injection.
2. **Cross-culture name sets: 300 pairs per language in English, Japanese, and
   Chinese**, built from 30 name pairs crossed with 10 sentence templates, with
   a Japanese surname as the target (`data/{en,ja,zh}_cross_culture_names.py`).
   Table 3, Table A1, and Table A5 use these.
3. **Two-entity sentence sets: 300 sentences per language** with two entity slots
   each (`data/{en,ja}_multientity_300.py`, `data/zh_multientity_300_v2.py`), used by the
   multi-entity test (Sections 4.2 and 5.5): the two-entity columns of Table 3 and Table A8.
4. **Six-language heteronym / polyphone sets**: EN heteronyms (500 pairs), ZH
   polyphones, KO hanja polyphones, FR and DE heteronyms (300 pairs each). With
   the first two sets above, 3,500 pairs in total across 11 sets, all
   materialized by one command (§4, step 1). The JA and ZH items of the six-language same-surface test come
   from `data/entities_10000.jsonl` and `data/zh_polyphones_v2.json`.
5. **Listening-test stimulus manifests, 200 items** across four sets
   (EN/JA/ZH name sets and the JA same-surface set, 50 main items each, plus 3
   catch and 2 practice items per set), with source/target words, the reading
   shown on screen, and the block assignment: `human_eval/stimuli/`.
6. **Item-level listening results for all 200 items**
   (`human_eval/results/sheet3_item_level.csv`): per-item counts of INJ / BL /
   neither, the number of judgments, and the automatic (Whisper) verdict for
   both conditions. This is enough to recompute the published preference rates
   and their confidence intervals without any participant-level data (§4,
   step 2).
7. **Verbatim instruction text in three languages** for both studies
   (`human_eval/instructions_verbatim.md`, with English glosses of the
   Japanese and Chinese text), and **32 screenshots** of the deployed
   experiment interface (`human_eval/screenshots/`). These let others
   replicate the exact listener-facing protocol rather than reconstruct it
   from prose.
8. **Pre-computed per-pair outputs behind every table** (`eval_results/`), so the
   published values can be recomputed and audited without rerunning synthesis:
   the per-pair UTMOS, durations and verdicts of Tables 1 and 2; for Table 3, all
   17 model-language conditions at our replacement point, at each of the five
   swept depths, and on the two-entity sentences; and the per-item results of
   Tables A1, A2, A3, A5, A6, A10 and A12.
9. **The pre-specified analysis code** for both listening studies
   (`human_eval/analysis/`), including a self-test mode that runs on synthetic
   data with planted ground truth (§4, step 4).

## 3. Setup

The reproduction in §4 needs Python 3.10+ and the packages in
`requirements-cpu.txt`:

```bash
pip install -r requirements-cpu.txt     # numpy, scipy, pandas, statsmodels, pykakasi
```

Rerunning the experiments themselves needs a GPU and the containers of `docker/`;
see `REPRODUCE.md`.

## 4. Reproduction without a GPU

Every command below runs on CPU from a clean checkout of this repository and
prints the values of the paper. Set `REPO_ROOT` once:

```bash
export REPO_ROOT="$PWD"
```

**Step 1 - build every benchmark item list (stdlib only).**

```bash
python3 scripts/verify_benchmark.py
```

Materializes all 11 sets and asserts their sizes; prints `TOTAL 3500`.
Add `--dump out/` to write one JSONL file per set.

```bash
python3 scripts/verify_item_lists.py
```

Rebuilds the items of the six-language same-surface test (JA and ZH), the two-entity sentences of Table 3,
and the Table A3 benchmark from `data/` and checks them against the items recorded
in the shipped outputs.

**Step 2 - recompute the human-evaluation results.**

```bash
python3 scripts/aggregate_human_eval.py
python3 scripts/aggregate_failure_cases.py
```

The first recomputes the INJ-selection rates and their item-bootstrap intervals
from the item-level counts, reproducing the human columns of Table 1 (EN
`99.4 [98.6, 100.0]`, JA `98.6 [95.8, 100.0]`, ZH `100.0 [100.0, 100.0]`) and
Table A4, the Japanese same-surface set (`98.6 [97.4, 99.6]`, 493/3/4
judgments, 50/50 items reaching target), and echoes the shipped MOS results.
The second joins the item-level counts with the automatic verdicts and prints
the eight listener–ASR disagreements of Appendix H.

**Step 3 - recompute the automatic-evaluation tables.**

```bash
python3 scripts/aggregate_table1.py      # Table 1, automatic columns (Whisper-INJ, UTMOS, speaker similarity, |dF0| ratio)
python3 scripts/aggregate_table2.py      # Table 2, every column, with the Wilcoxon and McNemar tests
python3 scripts/aggregate_table3.py      # Table 3: all 17 rows, every column
python3 scripts/aggregate_two_entity_paired.py   # Appendix J: paired difference INJ (both) - INJ (one), bootstrap over sentences
python3 scripts/aggregate_appendix.py    # Tables A1, A2, A3, A5, A6, A10, A11, A12 and the compute figures
```

The definitions behind every value (counts, denominators, interval formulas)
are written at the top of each script.

**Step 4 - self-test the pre-specified listening-test analysis (~30 s).**

```bash
python3 human_eval/analysis/analyze_preference_primary.py --simulate
```

Builds a synthetic response database from the shipped stimulus manifests with
known rates planted (EN 95% / JA 90% / ZH 93%) plus one catch-failing listener,
one straight-liner, and one incomplete listener per language, then runs the real
estimator over it. The recovered rates track the planted truth and all nine
quality-control exclusions fire, which exercises the analysis path end to end
without any participant data. It writes `human_eval/analysis/sim_ab.db`, which
is safe to delete.

## 5. Table-to-file map

| Paper | CPU command (§4) | Files read | Code that produced them (`REPRODUCE.md`) |
|---|---|---|---|
| Table 1, human columns; Table A4 | `aggregate_human_eval.py` | `human_eval/results/sheet3_item_level.csv`, `sheet2_main_results.csv` | `human_eval/analysis/` |
| Table 1, GEE `z` and MOS | — | `human_eval/results/primary_analysis.txt`, `mos_final.txt` | `human_eval/analysis/` (needs the withheld participant-level data) |
| Table 1, automatic columns | `aggregate_table1.py` | `eval_results/table1_cross_lingual/` | §1: `scripts/projection/eval_utmos_speaker_fixed.py`, judges and scorers in `scripts/analysis/` |
| Table 2 | `aggregate_table2.py` | `eval_results/table2_system_comparison/`, `table1_cross_lingual/` | §1: also `scripts/voicecraft/`, `scripts/editts/` |
| Table 3 | `aggregate_table3.py` | `eval_results/table3_cross_arch/` (`manifest.json` lists the file of every cell) | §2: one folder per system, `scripts/layer_ablation/`, `src/phoneme_per.py`, `src/mcd_pysptk.py` |
| Table A1 | `aggregate_appendix.py` | `eval_results/table_a1_crossattn_mask/` | §3: `scripts/projection/eval_crossattn_entity_dtw.py` |
| Table A2 | `aggregate_appendix.py` | `eval_results/table_a2_phoneme_mask/` | §3: `scripts/projection/eval_crossattn_entity_per.py` |
| Table A3 | `aggregate_appendix.py` | `eval_results/table_a3_ja_reading/` | §3: `scripts/analysis/numerals_verify_generic.py`, `bench_hetnum77_{inj,furigana}.py` |
| Table A5 | `aggregate_appendix.py` | `eval_results/table_a5_baselines/` | §3: `scripts/regen2/appg_vc_*.py`, `scripts/projection/eval_baseline_text_edit.py` |
| Table A6 | `aggregate_appendix.py` | `eval_results/table_a6_cer/`, `data/` (JA needs `pykakasi`) | §1: transcripts of the Table 1 runs |
| Listener demographics (not in the paper) | — | `human_eval/results/demographics_aggregate.txt` | — |
| Listener–ASR disagreements (Appendix H) | `aggregate_failure_cases.py` | `human_eval/results/sheet3_item_level.csv`, the Table 1/2 judge files | — |
| Six-language same-surface test (not in the paper; its JA and EN pairs feed Table A2) | `aggregate_appendix.py` | `eval_results/supp_six_language/` | §3: `scripts/projection/eval_cjk_n300*.py`, `scripts/regen2/appk_*.py` |
| Table A7, compute (Appendix I) | `aggregate_appendix.py` | `eval_results/appendix_i_compute/` | `scripts/spanfix/c7_param_count.py`, `c6_compute_budget.py` |
| Appendix J, paired accuracy difference | `aggregate_two_entity_paired.py` | `eval_results/table3_cross_arch/two_entity/`; writes `two_entity_paired.json` | — |
| Table A8 | `aggregate_appendix.py` | `eval_results/table_a8_two_entity_utmos/` | §2: `scripts/spanfix/utmos_multi_now.py` |

### Known caveats in the shipped files

- **Interval formulas differ by table**, as in the paper; each script's
  docstring states its formula. Table 3 MCD uses the Student-t 95% half-width
  with the sample standard deviation; Table 3 PIB and the rates of Tables A2,
  A5, and A10 use Wilson intervals; Table A1 uses 1.96 × the population
  standard deviation / √n.
- **Table 3, six MCD files were computed at camera-ready time.** For
  T5Gemma-TTS EN/JA/ZH, SpeechT5 EN, and Llasa-3B EN/ZH the value at our
  replacement point is taken from the layer-sweep run at that depth, whose MCD
  had not been computed. It was computed on 2026-11-14 with the same function as
  the other MCD files (`manifest.json` marks them).
- **Table 3, Irodori-TTS** was rerun on 2026-11-15 on an NVIDIA H100 80GB
  (point and sweep in one run; the two-entity columns are the original A100
  run). The original sweep script applied the final LayerNorm twice at
  l* = 25, let the layers after the replacement attend to padding, and widened
  26 spans, so its l* = 25 audio differed from the replacement point. The fixed
  script (`scripts/layer_ablation/eval_irodori_layer.py`) gives md5-identical
  audio at l* = 25 in all 300 pairs. The rerun of the replacement point matches
  the original A100 run closely (same verdict in 297/300 pairs; per-pair MCD
  within 0.03 dB); its PIB is 49.33 against 49.00 before.
- **Table 3, SpeechT5 EN sweep:** the l* = 12 file is also the file at our
  replacement point. Its baseline and target entity transcriptions differ from
  those of the other four depth files in 30 of 300 pairs.
- **Table 3 two-entity columns (ZH) and Table A5** are counted from stored
  per-row verdicts. Re-deriving the ZH verdicts from the transcripts needs
  `pypinyin`; re-deriving the Table A5 verdicts needs `pykakasi` and
  `src/score_trm_g2pcer_nopyojt.py`.
- **Table 1, ZH:** the voice-prompt transcript given to T5Gemma-TTS was not the
  transcript of the prompt for 298 pairs, and the Whisper large-v3-turbo transcript
  for pairs 67 and 76, which were regenerated (`REPRODUCE.md` §0.1).
- **Table 1, speaker similarity and |ΔF0|** are shipped as the summaries the scorers
  wrote ([mean, 1.96·SE, SD]); the per-pair values were not stored and need the audio.
- **Six-language same-surface test, Japanese injection success:** the per-row file stores the strict
  containment result (255/300). The paper's 256/300 applies the kana
  normalization `norm_kana` of `scripts/analysis/ja_wordrepl_judge.py`
  (ー expansion, おう = おお, えい = ええ, variant kana), which `scripts/aggregate_appendix.py`
  copies; the only changed item is pair 105.
- **Six-language same-surface test, IPA change:** pairs whose BL or INJ IPA transcript is empty are
  excluded (KO 10, FR 28, DE 30 pairs; listed in
  `ipa_excluded_empty.json`), as stated in the paper.
- **Appendix I, compute:** the job-record recount covers jobs up to 2026-10-02
  (272.86 GPU-hours); a few later regenerations (the Table A2 JA rerun, 8 pairs of
  Table A1, 8 pairs of Table 3) bring it to about 275, the value in the paper.
- `eval_results/table1_cross_lingual/en_judge.json` records Whisper medium.en
  containment (291/300). The paper's EN Whisper-INJ value (289/300 = 96.33%) comes
  from `eval_results/table2_system_comparison/whisper_inj_large_v3.json`, the same
  run scored by Whisper large-v3; `aggregate_table1.py` reads the latter.

## 6. Rerunning the experiments

`REPRODUCE.md` gives the command of every run behind the shipped outputs, with the
containers (`docker/`), the model revisions (`scripts/download_models.py`), the
third-party code (`scripts/setup_third_party.sh`), the voice prompts
(`scripts/prepare_voice_prompts.py`, which rebuilds them from their corpora and
checks their MD5) and the re-seeded pairs (`configs/pair_seeds.csv`).

- The base TTS model `Aratako/T5Gemma-TTS-2b-2b` and some others are gated
  Hugging Face repositories. Accept their licenses and export `HF_TOKEN` before
  downloading.
- The original runs used one NVIDIA A100-PCIE-40GB. Most systems sample during
  synthesis (and SpeechT5 applies dropout at inference), so on other GPUs the audio
  and the per-pair scores differ even with the same seeds; `REPRODUCE.md` reports a
  rerun on an H100.
- The scripts are the files that ran, and many of their comments and messages are
  in Japanese. Only the host paths (now `$REPO_ROOT`, default `/mnt`), the module
  locations of this repository, and the Japanese kana helpers outside the generation
  path (now pykakasi) were changed.

Producing the utterances behind all tables of the paper took about 275
GPU-hours on one NVIDIA A100-PCIE-40GB, at 8.7 seconds per utterance on average;
the layer sweep of Table 3 accounts for about 43% of that.

### Random seeds

`configs/table_specs.yaml` gives the seed of each table. The pairs listed in
`configs/pair_seeds.csv` are the exceptions: the clips named in `clips` were
synthesized with the seed in `seed`. `table` is the table number in the paper,
`cell` the block or row within it (`-` for every value of that language), and
`pair_idx` the 0-based position of the pair in the item list behind that table.

## 7. What is not included, and why

- **Model weights.** SpeechSplice trains nothing and edits no weights. All
  experiments use published frozen checkpoints, obtained from their original
  sources (`scripts/download_models.py`). There is no checkpoint of ours to release.
- **Voice prompts.** The corpus recordings used as voice-cloning prompts (LibriTTS,
  JSUT, the Databaker corpus, FLEURS) are not redistributed, apart from the JSUT
  recording on the sample page; `scripts/prepare_voice_prompts.py` rebuilds them.
- **Synthesized audio.** The generated waveforms clone a prompt speaker's
  voice, so redistributing them would redistribute that speaker identity.
  Apart from the 80 listening-test clips on the sample page (`docs/`), we
  release the item lists, the code, and the per-pair scores instead,
  so the numbers can be audited without the audio.
- **Participant-level listening responses.** Listeners consented to aggregate
  reporting, not to release of individual-level records. We release item-level
  counts (all 200 items), per-set totals, and aggregate demographics, which is
  enough to recompute every published preference figure (§4, step 2). The one
  participant identifier that the analysis code must name — the pre-specified
  straight-liner excluded from the MOS analysis — appears only as a truncated
  SHA-256 digest.
- **Crowdsourcing platform identifiers and completion codes.** Removed
  throughout, including from the screenshots, where participant identifiers and
  completion codes are masked.

## 8. Human-evaluation materials

See [`human_eval/README.md`](human_eval/README.md) for the contents of that
tree, the exclusion rules, and which analyses can and cannot be re-run from the
released files.

## 9. License

Our code, benchmark item lists, and listening-test materials are released under
the MIT License (`LICENSE`). Third-party code under `src/_vendor/` and
`third_party/` keeps its own license, and the ja-tts-g2p-bench items
(`data/external/`, `eval_results/table_a3_ja_reading/`) are CC BY-SA 4.0; see
`NOTICE`. The audio and sentence texts on the sample page (`docs/`) are not
covered by the MIT License; their terms are in `docs/LICENSE_AUDIO.md`.

## 10. Citation

Machine-readable metadata is in `CITATION.cff`. The paper will appear at
EACL 2027; until the proceedings are published, please cite it as:

```bibtex
@misc{kurihara2027speechsplice,
  title  = {{SpeechSplice}: Pronunciation Control in Encoder--Decoder
            {LLM}-based {TTS} by Replacing Encoder Hidden States},
  author = {Kurihara, Kiyoshi and Ukita, Jumpei and Iwasawa, Yusuke and
            Matsuo, Yutaka},
  year   = {2027},
  note   = {To appear at EACL 2027}
}
```
