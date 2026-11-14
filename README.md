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

---

## 1. What is in this repository

| Path | Contents |
|---|---|
| `src/` | Implementation modules: encoder-output injection (`injection.py`), ROME/MEMIT editor (`rome.py`), forced alignment (`alignment.py`), scorers (`mcd_pysptk.py`, `phoneme_per.py`, `score_utmos.py`, `score_trm_g2pcer.py`, `strict_v2.py`) |
| `src/_vendor/t5gemma_tts/` | Minimal T5Gemma-TTS inference utilities, vendored with their original MIT license (see `NOTICE`) |
| `data/` | **Benchmark item lists.** 11 sets, 3,500 sentence pairs in total, across 6 languages |
| `scripts/` | Generation and scoring scripts for T5Gemma-TTS and the comparison systems, and the table aggregators (`aggregate_*.py`) |
| `configs/` | Experiment configuration (`table_specs.yaml`, `voice_prompts.yaml`, `pair_seeds.csv`) |
| `eval_results/` | **Pre-computed per-pair evaluation outputs**, one directory per table (`table1_*`, `table2_*`, `table3_*`, `table_a*`). Each `manifest.json` names the experiment file a copy came from and its MD5 |
| `human_eval/` | **Listening-test materials and results**: stimulus manifests, verbatim instructions in three languages, interface screenshots, item-level and aggregate results, and the pre-specified analysis code. See [`human_eval/README.md`](human_eval/README.md) |
| `docs/` | The audio sample page (`index.html`) and its clips, under the terms in `docs/LICENSE_AUDIO.md` |
| `requirements.txt` | Full dependency set (needed only for the GPU synthesis/scoring pipeline) |
| `requirements-cpu.txt` | The packages needed for the GPU-free reproduction below |

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
3. **Six-language heteronym / polyphone sets**: EN heteronyms (500 pairs), ZH
   polyphones, KO hanja polyphones, FR and DE heteronyms (300 pairs each). With
   the two sets above, 3,500 pairs in total across all 11 sets, all
   materialized by one command (§4, step 1).
4. **Listening-test stimulus manifests, 200 items** across four sets
   (EN/JA/ZH name sets and the JA same-surface set, 50 main items each, plus 3
   catch and 2 practice items per set), with source/target words, the reading
   shown on screen, and the block assignment: `human_eval/stimuli/`.
5. **Item-level listening results for all 200 items**
   (`human_eval/results/sheet3_item_level.csv`): per-item counts of INJ / BL /
   neither, the number of judgments, and the automatic (Whisper) verdict for
   both conditions. This is enough to recompute the published preference rates
   and their confidence intervals without any participant-level data (§4,
   step 2).
6. **Verbatim instruction text in three languages** for both studies
   (`human_eval/instructions_verbatim.md`, with English glosses of the
   Japanese and Chinese text), and **32 screenshots** of the deployed
   experiment interface (`human_eval/screenshots/`). These let others
   replicate the exact listener-facing protocol rather than reconstruct it
   from prose.
7. **Pre-computed per-pair outputs behind the tables** (`eval_results/`), so
   the published values can be recomputed and audited without rerunning
   synthesis. For Table 3 this covers all 17 model-language conditions:
   the MCD and phoneme scores at our replacement point, the phoneme scores at
   each of the five swept depths, and the per-slot judge verdicts on the
   two-entity sentences (which also list each sentence's word pairs and
   template id).
8. **The pre-specified analysis code** for both listening studies
   (`human_eval/analysis/`), including a self-test mode that runs on synthetic
   data with planted ground truth (§4, step 4).

## 3. Setup

The reproduction in §4 needs Python 3.10+ and the packages in
`requirements-cpu.txt`:

```bash
pip install -r requirements-cpu.txt     # numpy, scipy, pandas, statsmodels, pykakasi
```

`requirements.txt` is the full environment, and is needed only to rerun the
synthesis and scoring pipeline on a GPU (§6).

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

**Step 2 - recompute the human-evaluation preference results.**

```bash
python3 scripts/aggregate_human_eval.py
```

Recomputes the INJ-selection rates and their item-bootstrap intervals from the
item-level counts, reproducing the human columns of Table 1 (EN
`99.4 [98.6, 100.0]`, JA `98.6 [95.8, 100.0]`, ZH `100.0 [100.0, 100.0]`) and
Table A4, the Japanese same-surface set (`98.6 [97.4, 99.6]`, 493/3/4
judgments, 50/50 items reaching target). Also echoes the shipped MOS results.

**Step 3 - recompute the automatic-evaluation tables.**

```bash
python3 scripts/aggregate_table1.py      # Whisper-INJ: EN 96.33±2.19, JA 92.33±3.04, ZH 96.00±2.28
python3 scripts/aggregate_table2.py      # SpeechSplice 96.33±2.19, VoiceCraft 90.00±3.41, EdiTTS 86.00±3.93
python3 scripts/aggregate_table3.py      # Table 3: all 17 rows, every column
python3 scripts/aggregate_appendix.py    # Tables A1, A2, A5, A6, A10, A12
```

`aggregate_table1.py` and `aggregate_table2.py` print, in a footer, which
columns they cannot derive from the shipped files and why. The definitions
behind every value (counts, denominators, interval formulas) are written at
the top of each script.

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

| Paper | Command | Files read | Reproduced here? |
|---|---|---|---|
| Table 1, human columns (INJ selected, A/B %) | `scripts/aggregate_human_eval.py` | `human_eval/results/sheet3_item_level.csv`, `sheet2_main_results.csv` | Yes |
| Table 1, GEE `z` and MOS columns | — | `human_eval/results/primary_analysis.txt`, `mos_final.txt` | Shipped as text; recomputation needs participant-level data (withheld) |
| Table 1, Whisper-INJ column | `scripts/aggregate_table1.py` | `eval_results/table1_cross_lingual/{ja,zh}_judge.json`, `eval_results/table2_system_comparison/whisper_inj_large_v3.json` | Yes, all three languages |
| Table 1, UTMOS / speaker similarity / \|ΔF0\| | `scripts/aggregate_table1.py` | as above + `speaker_similarity_tests.json` | Partly: UTMOS (INJ) for EN and the BL–INJ similarity tests only. The rest needs the GPU run |
| Table 2 (VoiceCraft / EdiTTS comparison) | `scripts/aggregate_table2.py` | `eval_results/table2_system_comparison/whisper_inj_large_v3.json`, `speaker_similarity_tests.json` | Whisper-INJ, McNemar, and speaker-identity statistics: yes. UTMOS / \|ΔF0\| / \|Δduration\|: needs the GPU run |
| Table 3 (cross-architecture, 17 conditions) | `scripts/aggregate_table3.py` | `eval_results/table3_cross_arch/manifest.json` and its `point_mcd/`, `point_pib/`, `sweep_pib/`, `two_entity/` | Yes: MCD, PIB, the five swept depths with their min–max, and the three two-entity columns |
| Table A1 (acoustic cross-attention mask ablation) | `scripts/aggregate_appendix.py` | `eval_results/table_a1_crossattn_mask/{en,ja,zh}.json` | Yes |
| Table A2 (phoneme-level mask ablation) | `scripts/aggregate_appendix.py` | `eval_results/table_a2_phoneme_mask/{en,ja}.json` | Yes |
| Table A3 (Japanese 160-item reading benchmark) | — | — | No. The items come from ja-tts-g2p-bench; per-item outputs are not shipped |
| Table A4 (same-surface listening test) | `scripts/aggregate_human_eval.py` | `human_eval/results/sheet3_item_level.csv` | Yes |
| Table A5 (text- and weight-level baselines) | `scripts/aggregate_appendix.py` | `eval_results/table_a5_baselines/ja.json` | Yes, from the stored per-pair verdicts (see the caveats) |
| Table A6 (full-sentence CER) | `scripts/aggregate_appendix.py` | `eval_results/table_a6_cer/*.json`, `data/` | Yes (JA needs `pykakasi`) |
| Table A8 (listener demographics) | — | `human_eval/results/demographics_aggregate.txt` | Shipped as text |
| Table A10 (six-language same-surface test) | `scripts/aggregate_appendix.py` | `eval_results/table_a10_six_language/` | Yes |
| Table A12 (UTMOS of the two-entity sentences) | `scripts/aggregate_appendix.py` | `eval_results/table_a12_two_entity_utmos/` | Yes |

### Known caveats in the shipped files

- **Interval formulas differ by table**, as in the paper; each script's
  docstring states its formula. Table 3 MCD uses the Student-t 95% half-width
  with the sample standard deviation; Table 3 PIB and the rates of Tables A2,
  A5, and A10 use Wilson intervals; Table A1 uses 1.96 × the population
  standard deviation / √n.
- **Table 3, four MCD files were computed at camera-ready time.** For
  T5Gemma-TTS EN, SpeechT5 EN, and Llasa-3B EN/ZH, the value at our
  replacement point is taken from the layer-sweep run at that depth. The MCD of
  those four sweep folders was computed on 2026-11-14 with the same function as
  the other MCD files (`manifest.json` marks them); their per-pair rows are
  unchanged from that computation.
- **Table 3, T5Gemma-TTS JA and ZH:** the value at our replacement point
  (`point_pib/`) and the sweep value at l* = 26 (`sweep_pib/..._l26.json`) come
  from runs whose audio differs in a few pairs (JA 59, 99, 164, 179; ZH 15, 30,
  145, 246, 295), so the two files give 51.33 vs 52.00 (JA) and 54.67 vs 55.00
  (ZH). Both values lie inside the printed sweep range.
- **Table 3, Irodori-TTS** replaces after the projection at our replacement
  point and before it in the sweep (the dagger in the paper), so its point and
  sweep files come from different runs by design.
- **Table 3, SpeechT5 EN sweep:** the l* = 12 file is also the file at our
  replacement point. Its baseline and target entity transcriptions differ from
  those of the other four depth files in 30 of 300 pairs.
- **Table 3 two-entity columns (ZH) and Table A5** are counted from stored
  per-row verdicts. Re-deriving the ZH verdicts from the transcripts needs
  `pypinyin`; re-deriving the Table A5 verdicts needs `pykakasi` and
  `src/score_trm_g2pcer_nopyojt.py`.
- **Table A10, Japanese injection success:** the per-row file stores the strict
  containment result (255/300). The paper's 256/300 applies a kana
  normalization (ー expansion, おう = おお, えい = ええ, variant kana) whose
  original code was not preserved; `scripts/aggregate_appendix.py`
  re-implements it from its documented rules and reproduces 256 (the only
  changed item is pair 105).
- **Table A10, IPA change:** pairs whose BL or INJ IPA transcript is empty are
  excluded (KO 10, FR 28, DE 30 pairs; listed in
  `ipa_excluded_empty.json`), as stated in the paper.
- `eval_results/table1_cross_lingual/en_judge.json` records a separate
  g2p-containment scoring pass (291/300). The paper's EN Whisper-INJ value
  (289/300 = 96.33%) comes from
  `eval_results/table2_system_comparison/whisper_inj_large_v3.json`, which is
  the same run scored by the Table 2 judge. `aggregate_table1.py` reads the
  latter and says so in a comment.

## 6. What needs a GPU

Nothing in §4 does. The synthesis and scoring pipeline does, and is not
exercised by the reproduction above.

- **Generation code included:** T5Gemma-TTS (`src/injection.py`, the Table 1
  and Table A1 scripts, and the Table A5 baselines including ROME/MEMIT),
  and the VoiceCraft and EdiTTS runs of Table 2
  (`scripts/table2_voicecraft/`, `scripts/table2_editts/`).
- **Not included:** the splice code for the other nine systems of Table 3
  (SpeechT5, LongCat-AudioDiT, Irodori-TTS, Kiseki-TTS, Qwen3-TTS, XTTS-v2,
  Llasa-3B, CSM-1B, Bark), the layer-sweep and two-entity generation, the
  phoneme-level scoring of Table A2, and the Table A3 runs. Except for Table
  A3, their per-pair outputs are shipped instead (§5).
- The base TTS model `Aratako/T5Gemma-TTS-2b-2b` is a gated Hugging Face
  repository. Accept its license and export `HF_TOKEN` before running any
  generation script.
- `espeak-ng` is required by the IPA phonemizer
  (`apt-get install -y espeak-ng`, or `brew install espeak`).

Producing the utterances behind all tables of the paper took about 290
GPU-hours on one NVIDIA A100-PCIE-40GB, at a median of 8.0 seconds per
utterance; the layer sweep of Table 3 accounts for about two thirds of that.

### Portability note for the GPU scripts

The generation and scoring scripts were run inside a container with this
repository mounted at `/mnt`. Most of them detect that (`os.path.isdir(
"/mnt/scripts")`) and otherwise fall back to `REPO_ROOT` or the working
directory, so they run unchanged outside the container. A few still hardcode
`/mnt` and need editing for another layout: `scripts/appendix_f_eval_rome.py`,
`scripts/appendix_f_eval_g2p_baseline.py`,
`scripts/appendix_f_eval_baseline_text_edit.py`,
`scripts/table2_voicecraft/edit_entity_en.py`, and the cache defaults in
`src/phoneme_per.py`. None of these are on the GPU-free path in section 4.
The `appendix_f_*` and `table4_*` file names predate the camera-ready
numbering; they belong to Table A5 and Table A1, respectively.

### Random seeds

`configs/table_specs.yaml` gives the seed of each table. The pairs listed in
`configs/pair_seeds.csv` are the exceptions: the clips named in `clips` were
synthesized with the seed in `seed`. `table` is the table number in the paper,
`cell` the block or row within it (`-` for every value of that language), and
`pair_idx` the 0-based position of the pair in the item list behind that table.

## 7. What is not included, and why

- **Model weights.** SpeechSplice trains nothing and edits no weights. All
  experiments use published frozen checkpoints, obtained from their original
  sources: the ten TTS systems of Table 3, VoiceCraft and EdiTTS, and the
  evaluation models (Whisper large-v3 and medium.en, Furigana Whisper,
  whisper-pinyin, wav2vec2-xlsr-53-espeak-cv-ft, UTMOS, WavLM-SV). Appendix J
  of the paper lists them with links. There is no checkpoint of ours to
  release.
- **Synthesized audio.** The generated waveforms clone a prompt speaker's
  voice, so redistributing them would redistribute that speaker identity.
  Apart from the 80 listening-test clips on the sample page (`docs/`), we
  release the item lists, the scoring code, and the per-pair scores instead,
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
the MIT License (`LICENSE`). The vendored T5Gemma-TTS utilities under
`src/_vendor/t5gemma_tts/` are third-party code under their own MIT license;
see `NOTICE` and `src/_vendor/t5gemma_tts/LICENSE`. The audio and sentence
texts on the sample page (`docs/`) are not covered by the MIT License; their
terms are in `docs/LICENSE_AUDIO.md`.

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
