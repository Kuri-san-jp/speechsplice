# Reproducing every experiment from scratch (GPU)

`README.md` §4 recomputes every value of the paper from the shipped per-pair outputs on a
CPU. This file is for re-running the experiments themselves: synthesis, scoring and
aggregation. The commands are the ones that produced the shipped outputs, taken from
the job queues of the original runs; only the host paths were replaced.

## 0. Setup

**Containers.** The experiments ran in Docker containers with this repository mounted
at `/mnt` (the scripts detect `/mnt/scripts` and use `/mnt` as the repository root; outside
a container they use `$REPO_ROOT`).

| Image | Definition | Used for |
|---|---|---|
| `my-lab-image` | `docker/Dockerfile.main` (+ `docker/my_lab_image_freeze.txt`) | everything not listed below |
| `irodori-tts` | `docker/Dockerfile.irodori` | Irodori-TTS (PyTorch 2.10, Transformers 5) |
| `xtts-tts` | `docker/Dockerfile.xtts` | XTTS-v2 (adds coqui-tts 0.27.5) |
| `voicecraft-image` | `docker/Dockerfile.voicecraft` | VoiceCraft (Table 2) |

```bash
docker build -f docker/Dockerfile.main -t my-lab-image:latest docker
docker run --gpus all -it -v "$PWD":/mnt -w /mnt my-lab-image:latest bash
# inside the container:
bash scripts/setup_third_party.sh           # Irodori-TTS, whisper-pinyin, EdiTTS, VoiceCraft, qwen-tts, coqui-tts at the used versions
HF_HUB_OFFLINE=0 python3 scripts/download_models.py   # every model at the used revision (~150 GB; gated repos need HF_TOKEN)
python3 scripts/prepare_voice_prompts.py    # rebuilds the voice-cloning prompts from their corpora (see §0.1)
source scripts/env.sh                       # HF_HOME, offline mode, TORCH_HOME, PYTHONPATH
```

**Hardware.** Every utterance of the paper was synthesized on one NVIDIA
A100-PCIE-40GB (about 275 GPU-hours; Appendix J). Most systems sample during synthesis,
and SpeechT5 applies dropout at inference, so on a different GPU the audio and the
per-pair scores change even with the same seeds. Rerunning on an H100 (2026-11-14) gave:

| Condition | Paper (A100) | Rerun (H100) | Audio vs. the original |
|---|---|---|---|
| LongCat-AudioDiT EN, our replacement point | MCD 10.00 ± 0.24, PIB 65.33 | PIB 65.00 (same verdict in 295/300 pairs) | nearly identical (SNR about 50 dB) |
| SpeechT5 EN, our replacement point | MCD 3.53 ± 0.06, PIB 64.00 | MCD 3.66 ± 0.07, PIB 55.33 | different (prenet dropout) |

The SpeechT5 value of the paper comes from a rerun on 2026-09-21; the first run on
2026-05-17 (A100, earlier library versions) gave MCD 3.66 ± 0.07 and PIB 55.67. Treat the
spread between runs as part of the uncertainty of the generated-speech metrics; the
intervals of the paper describe sampling over pairs only.

**What was run from a clean checkout (2026-11-14, H100, `my-lab-image`).** Every
generator and scorer below completed on two pairs or more, with the models of
`scripts/download_models.py` and the prompts of `scripts/prepare_voice_prompts.py`:
T5Gemma-TTS (Table 1 EN/JA, Table 3 point and sweep, two-entity EN/JA, Tables A1, A2, A3,
A5 baseline, A10 JA/ZH/FR), SpeechT5, LongCat-AudioDiT, Kiseki-TTS, Qwen3-TTS, XTTS-v2
(with coqui-tts 0.27.5 and its dependencies), Llasa-3B, CSM-1B, Bark (EN and JA),
EdiTTS, and the PIB, MCD, Whisper, Furigana-Whisper and IPA scorers. Irodori-TTS and
VoiceCraft need their own images and were not rerun.

**Seeds.** Every generator seeds with 42 before each synthesis. The pairs in
`configs/pair_seeds.csv` were regenerated as whole pairs with seed 43 (then 44) because the
seed-42 output was silent or unusable; the wrapper that did it is named in each section.

### 0.1 Voice-cloning prompts

The prompts are not redistributed (JSUT and Databaker terms); `scripts/prepare_voice_prompts.py`
rebuilds them from the original corpora and checks their MD5 against the files used.

| File | Source | Used by |
|---|---|---|
| `eval_results/_natural_refs_norm/en_natural_norm.wav` | LibriTTS test-clean `260_123440_000002_000004`, 44.1 kHz mono | Table 1/2 EN, VoiceCraft, Table 3 Qwen3/XTTS/Llasa/CSM EN |
| `eval_results/_natural_refs_norm/zh_natural_bznsyp_norm.wav` | Databaker (BZNSYP), 44.1 kHz mono | Table 1 ZH, Table 3 Llasa ZH |
| `eval_results/_natural_refs_norm/ja_natural_norm.wav` | JSUT `BASIC5000_0079`, 44.1 kHz mono | Table A5; Table 3 Qwen3 JA |
| `eval_results/_natural_refs_norm/zh_natural_norm.wav` | FLEURS `cmn_hans_cn`, 44.1 kHz mono | Table 3 Qwen3 ZH |
| `eval_results/subjective_eval_jsut_audio/000_GT_BASIC5000_0011.wav` | JSUT `BASIC5000_0011` | Table 1 JA, Table A3, Table A10 EN/FR/DE/KO, two-entity T5Gemma-TTS |

T5Gemma-TTS in Table 3 (single entity), Table A1, Table A2, and Table A10 JA/ZH is
zero-shot (no prompt). When a T5Gemma-TTS command gives no `--reference_text`, the prompt
transcript is produced by Whisper large-v3-turbo. Two details matter for an exact rerun:

- Table 1 ZH was run with `--reference_text 他二人皮划艇比赛获得银牌`, which is not the
  transcript of the BZNSYP prompt. Pairs 67 and 76 (silent baselines in that run) were
  regenerated on 2026-08-04 without `--reference_text`, i.e. with the Whisper
  large-v3-turbo transcript of the prompt, and copied into the run folder.
- The 300 Table 1 EN pairs are 284 equal-length pairs from a run before the
  variable-length splice existed (where pooling and splicing are the same operation) plus
  the 16 unequal-length pairs regenerated with the splice (`scripts/analysis/inj_splice_generic.py`).

## 1. Table 1 and Table 2 (T5Gemma-TTS word replacement; VoiceCraft; EdiTTS)

```bash
# SpeechSplice (T5Gemma-TTS), 300 pairs per language
python3 scripts/projection/eval_utmos_speaker_fixed.py --lang en --n 300 --pairs_module en_wordrepl300_pairs \
  --reference_wav eval_results/_natural_refs_norm/en_natural_norm.wav --output_dir eval_results/en_wordrepl100_pairs_gen
python3 scripts/analysis/inj_splice_generic.py --pairs_module en_wordrepl300_pairs --lang en --n 300 --only_unequal \
  --ref_wav eval_results/_natural_refs_norm/en_natural_norm.wav --out_dir eval_results/en_wordrepl300_splice_inj
python3 scripts/projection/eval_utmos_speaker_fixed.py --lang ja --align splice --pairs_module ja_wordrepl300_pairs \
  --reference_text 木曽川は、しばしば日本のライン川と呼ばれている。 --output_dir eval_results/ja_wordrepl300_gen
python3 scripts/projection/eval_utmos_speaker_fixed.py --lang zh --align splice --pairs_module zh_wordrepl300_pairs \
  --reference_wav eval_results/_natural_refs_norm/zh_natural_bznsyp_norm.wav \
  --reference_text 他二人皮划艇比赛获得银牌 --output_dir eval_results/zh_wordrepl300_gen
# VoiceCraft (voicecraft-image) and EdiTTS (my-lab-image; three batches, as run)
#   VoiceCraft seeds once per run (torch.manual_seed(2026)) and resumes, so keep the three runs of the original:
python3 scripts/voicecraft/tts_entity_en_n100.py --pairs_module en_wordrepl100_pairs --out_dir /mnt/eval_results/voicecraft_tts_wordrepl100
python3 scripts/voicecraft/tts_entity_en_n100.py --pairs_module en_wordrepl200_pairs --n 200 --out_dir /mnt/eval_results/voicecraft_tts_wordrepl100
python3 scripts/voicecraft/tts_entity_en_n100.py --pairs_module en_wordrepl300_pairs --n 300 --out_dir /mnt/eval_results/voicecraft_tts_wordrepl100
bash scripts/editts/run_editts.sh            # pairs 0-99, 100-199, 200-299 (EdiTTS also seeds once per run)
# Scoring
python3 scripts/analysis/en16_splice_score.py                 # Whisper medium.en + UTMOS of the 16 regenerated EN pairs
python3 scripts/analysis/vc100_metrics.py --wav_root eval_results/en_wordrepl100_pairs_gen --edit_root none \
  --out_dir eval_results/en_wordrepl100_metrics --pairs_module en_wordrepl300_pairs --n 300
python3 scripts/analysis/vc100_metrics.py --wav_root eval_results/voicecraft_tts_wordrepl100 --edit_root none \
  --out_dir eval_results/voicecraft_tts_wordrepl100_metrics --pairs_module en_wordrepl300_pairs --n 300
python3 scripts/analysis/editts100_utmos.py --n 300 --pairs_module en_wordrepl300_pairs --wav_dir eval_results/editts_edit_wordrepl300/wavs
python3 scripts/analysis/table2_whisper_inj_lv3.py            # Whisper-INJ of all three systems (Whisper large-v3)
python3 scripts/analysis/ja_wordrepl_judge.py                 # JA Whisper-INJ (Furigana Whisper)
python3 scripts/analysis/zh_wordrepl_judge.py                 # ZH Whisper-INJ (Whisper large-v3 + pinyin)
python3 scripts/analysis/speaker_sim_splice_rerun.py --config wordrepl      # and --config ja_wordrepl, zh_wordrepl
python3 scripts/analysis/speaker_sim_systems.py
python3 scripts/analysis/delta_sim_prompt_all.py
python3 scripts/analysis/f0_systems.py
python3 scripts/analysis/provenance_table2_duration.py
python3 scripts/spanfix/cpuprop_cer_utmos_spk_recheck.py      # UTMOS p values and the Table A6 CER cells
```

## 2. Table 3 (17 conditions) and Table A12

Depth arguments: T5Gemma-TTS `--inject_layer k` and the decoder-only `--layer k` /
`--injection_layer k` are 0-based (l* = k + 1); the SpeechT5, LongCat-AudioDiT,
Irodori-TTS and Kiseki-TTS sweep scripts take the number of layers passed (l* = k).
`AB=eval_results/_layerabl`.

**Scoring (every condition).** PIB: `python3 src/phoneme_per.py --wav_dir <dir> --lang <l> --n 300 --output_json <json>`;
MCD: `python3 src/mcd_pysptk.py --wav_dir <dir> --lang <l> --mode whole --n 300 --output_json <json>`.
`scripts/aggregate_table3.py` lists which file each cell comes from (`eval_results/table3_cross_arch/manifest.json`).

```bash
# T5Gemma-TTS (zero-shot). BL/TGT of the point and sweep folders come from these runs:
python3 scripts/projection/eval_en_crossculture_t5gemma_fixed.py --n 300 --target_valid 300 \
  --output_dir eval_results/t5gemma_cc_wavs_en_n300 --save_wav_dir eval_results/_cc_wavs_b/t5gemma_en
python3 scripts/projection/eval_ja_crossculture_t5gemma_fixed.py --n 300 --target_valid 300 \
  --output_dir eval_results/t5gemma_cc_wavs_ja_n300 --save_wav_dir eval_results/_cc_wavs_a/t5gemma_ja
python3 scripts/projection/eval_zh_crossculture_t5gemma_fixed.py --n 300 --target_valid 300 \
  --output_dir eval_results/t5gemma_cc_wavs_zh_n300 --save_wav_dir eval_results/_cc_wavs_a/t5gemma_zh
#   sweep, k = 0 7 14 21 25 (l* = 1 8 15 22 26; k = 25 is also our replacement point); LANG=en|ja|zh,
#   SRC=_cc_wavs_b/t5gemma_en for EN and _cc_wavs_a/t5gemma_<lang> for JA/ZH
python3 scripts/layer_ablation/eval_en_crossculture_t5gemma_layer.py --inject_layer $k --n 300 --lang $LANG --final_via_hook \
  --wav_dir $AB/t5gemma_${LANG}_L$k --out_dir $AB/meta_${LANG}_L$k --link_bl_tgt_from eval_results/$SRC
#   seed-43 regeneration of EN pairs 112 118 128 138 141 158 238 and ZH pair 196 (all five depths)
python3 scripts/spanfix/t3fix_regen_gpu.py && python3 scripts/spanfix/t3fix_regen_apply.py --apply

# SpeechT5 (point = l*=12 run; sweep k = 1 3 6 9)
python3 scripts/projection/eval_speecht5_crossculture_fixed_v2.py --lang en --n 300 --output_dir $AB/meta_s5_today --save_wav_dir $AB/s5_en_today
python3 scripts/layer_ablation/eval_speecht5_layer.py --lang en --n 300 --inject_layer $k --wav_dir $AB/s5_en_L$k --out_dir $AB/meta_s5_L$k --link_bl_tgt_from $AB/s5_en_today

# LongCat-AudioDiT (point; ZH also with --span_extension for the 6 pairs the first pass skipped; sweep k = 1 3 6 9 12)
python3 scripts/longcat/run_longcat_eval.py --lang en --n 300
python3 scripts/longcat/run_longcat_eval.py --lang zh --n 300 && python3 scripts/longcat/run_longcat_eval.py --lang zh --n 300 --span_extension
python3 scripts/layer_ablation/eval_longcat_layer.py --lang $LANG --n 300 --inject_layer $k --wav_dir $AB/lc_${LANG}_L$k --out_dir $AB/meta_lc_${LANG}_L$k

# Irodori-TTS (irodori-tts image; the point run replaces after the projection, the sweep before it; sweep k = 1 7 13 19 25)
python3 scripts/irodori/run_irodori_eval.py --lang ja --n 300
python3 scripts/layer_ablation/eval_irodori_layer.py --lang ja --n 300 --inject_layer $k --wav_dir $AB/ir_ja_L${k}_v2 --out_dir $AB/meta_ir_L${k}_v2

# Kiseki-TTS (point; sweep k = 1 3 6 9 12)
python3 scripts/kiseki/run_kiseki_eval.py --lang ja --n 300
python3 scripts/layer_ablation/eval_kiseki_layer.py --lang ja --n 300 --inject_layer $k --wav_dir $AB/ki_ja_L$k --out_dir $AB/meta_ki_L$k

# Qwen3-TTS (point = --injection_layer 26, l*=27; sweep k = 0 9 13 20 26)
python3 scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py --lang $LANG --n 300 --target_valid 300 \
  --output_dir eval_results/qwen3_cc_wavs_${LANG}_n300 --save_wav_dir eval_results/_cc_wavs_a/qwen3_$LANG
python3 scripts/projection/eval_qwen3_crossculture_acoustic_fixed.py --lang $LANG --n 300 --injection_layer $k \
  --output_dir $AB/meta_qwen3_${LANG}_L$k --save_wav_dir $AB/qwen3_${LANG}_L$k

# XTTS-v2 (xtts-tts image; point = --layer 28; the sweep script writes l* = 1 9 16 24 29)
python3 scripts/xtts/run_xtts_eval.py --lang en --n 300 --allow_download --wav_dir /mnt/eval_results/_cc_wavs_a/xtts_en --out_dir /mnt/eval_results/xtts_cross_culture_en
python3 scripts/xtts/run_xtts_layersweep.py --lang en --n 300

# Llasa-3B (the sweep at --layer 26 is also our replacement point; ZH: --span_extension for pairs 15 145 295)
python3 scripts/llasa/run_llasa_layersweep.py --lang en --n 300
python3 scripts/llasa/run_llasa_layersweep.py --lang zh --n 300 && python3 scripts/llasa/run_llasa_layersweep.py --lang zh --n 300 --span_extension
#   seed-43 regeneration of EN 97 177 257 and ZH 85 180 199 284 (all five depths)
python3 scripts/regen2/regen_llasa_pairs_seed.py && python3 scripts/regen2/apply_regen2_llasa.py --apply

# CSM-1B (point = --layer 14; sweep k = 0 3 7 10 14)
python3 scripts/csm/run_csm_eval.py --lang en --n 300 --dtype float32 --overwrite --wav_dir /mnt/eval_results/_cc_wavs_a/csm_en --out_dir /mnt/eval_results/csm_cross_culture_en
python3 scripts/csm/run_csm_eval.py --lang en --n 300 --layer $k --dtype float32 --wav_dir $AB/csm_en_L$k --out_dir $AB/meta_csm_en_L$k

# Bark (point = --layer 22; sweep k = 0 5 11 16 22; JA with --voice_preset v2/ja_speaker_0)
python3 scripts/bark/run_bark_eval.py --lang en --n 300 --wav_dir /mnt/eval_results/_cc_wavs_a/bark_en --out_dir /mnt/eval_results/bark_cross_culture_en
python3 scripts/bark/run_bark_eval.py --lang ja --n 300 --voice_preset v2/ja_speaker_0 --wav_dir /mnt/eval_results/_cc_wavs_a/bark_ja --out_dir /mnt/eval_results/bark_cross_culture_ja
python3 scripts/bark/run_bark_eval.py --lang $LANG --n 300 --layer $k [--voice_preset v2/ja_speaker_0] --wav_dir $AB/bark_${LANG}_L$k --out_dir $AB/meta_bark_${LANG}_L$k
```

Llasa-3B's chat template inserts the current date into the prompt; the shipped runs
carry "19 Sep 2026" (sweep) and "17 Sep 2026" (two-entity). Set the system date, or pass the
date to `render_prompt` in `scripts/llasa/llasa_splice.py`, to reproduce them exactly.

**Two-entity sentences** (`data/{en,ja}_multientity_300.py`, `data/zh_multientity_300_v2.py`):

```bash
python3 scripts/analysis/generate_multientity.py --pairs_module en_multientity_300 --lang en --output_dir /mnt/eval_results/multientity_en_300
#   (ja: ja_multientity_300 -> multientity_ja_300; zh: zh_multientity_300_v2 -> multientity_zh_v2_300)
python3 scripts/projection/run_multientity_speecht5.py --mode multi --lang en --n 300 --output_dir /mnt/eval_results/multientity_speecht5_en_300
python3 scripts/longcat/run_multientity_longcat.py --pairs_module en_multientity_300 --lang en
python3 scripts/longcat/run_multientity_longcat.py --pairs_module zh_multientity_300_v2 --lang zh --splice_part both --cfg_strength 4 --guidance_method cfg --span_extension
python3 scripts/irodori/run_multientity_irodori.py --mode multi --lang ja --n 300            # irodori-tts image
python3 scripts/kiseki/run_multientity_kiseki.py --pairs_module ja_multientity_300 --lang ja
python3 scripts/qwen3/run_multientity_qwen3.py --mode multi --lang en                          # ja likewise
python3 scripts/qwen3/run_multientity_qwen3.py --pairs_module zh_multientity_300_v2 --lang zh --span_extension
python3 scripts/xtts/run_multientity_xtts.py --pairs_module en_multientity_300 --lang en --n 300 --layer 28   # xtts-tts image
python3 scripts/llasa/run_multientity_llasa.py --mode multi --lang en --pairs_module en_multientity_300 --n 300   # zh: --lang zh
python3 scripts/csm/run_multientity_csm.py --pairs_module en_multientity_300 --lang en --n 300 --layer 14
python3 scripts/bark/run_multientity_bark.py --pairs_module en_multientity_300 --lang en --n 300 --allow_null_layer  # ja likewise
#   judges: EN Whisper large-v3; JA Furigana Whisper; ZH large-v3 transcripts + toneless pinyin
python3 scripts/analysis/score_multientity.py --dir /mnt/eval_results/<dir> --lang <l> --n 300
python3 scripts/analysis/rejudge_ja_multientity_furigana.py --dir /mnt/eval_results/<dir> --n 300 --output_json /mnt/eval_results/<dir>/asr_furigana.json
#     (the target readings come from the Table 1 JA judge output, eval_results/ja_wordrepl300_gen/judge.json,
#      or its shipped copy eval_results/table1_cross_lingual/ja_judge.json)
python3 scripts/analysis/rejudge_zh_multientity_pinyin.py --scores /mnt/eval_results/<dir>/multientity_asr_scores.json
#   seed-43/44 regeneration of the T5Gemma-TTS EN pairs listed in configs/pair_seeds.csv
python3 scripts/regen2/regen_multientity_t5gemma_seed.py && python3 scripts/regen2/apply_regen2_multientity.py --apply
python3 scripts/regen2/regen_multientity_t5gemma_seed.py --campaign thankyou && python3 scripts/regen2/apply_regen2_multientity.py --apply --campaign thankyou
#   Table A12: UTMOS22 of every two-entity clip
python3 scripts/spanfix/utmos_multi_now.py
```

## 3. Appendix experiments

```bash
# Table A1 (acoustic cross-attention mask; length-preserving mean-pooled injection)
python3 scripts/projection/eval_crossattn_entity_dtw.py --lang en --cross_culture --inject_mode length_preserve --n 300
python3 scripts/projection/eval_crossattn_entity_dtw.py --lang ja --cross_culture --inject_mode length_preserve --n 300
python3 scripts/projection/eval_crossattn_entity_dtw.py --lang zh --cross_culture --inject_mode length_preserve --n 300
#   the shipped EN file is a complete rerun in four parts with seed fallback, and JA/ZH have
#   8 pairs regenerated after the span fix: scripts/spanfix/spanfix_appB_en_{gen,collect}.py,
#   scripts/spanfix/appB_jazh_{gen,collect}.py, scripts/regen2/g3_{regen_crossattn_pairs,merge_crossattn}.py

# Table A2 (phoneme-level mask; splice of Eq. 1; JA pairs = the Table A10 JA pairs)
python3 scripts/spanfix/appc_en_gen.py --seed 42 --part 0   # parts 0-5, then seed 43/44 for the pairs in pair_seeds.csv
python3 scripts/spanfix/appc_en_aggregate.py
python3 scripts/spanfix/appc_ja_span_check.py               # builds the JA pair file from the Table A10 JA run
python3 scripts/spanfix/appc_ja_gen.py --seed 42 --part 0   # parts 0-5
python3 scripts/spanfix/appc_ja_aggregate.py
#   (both wrap scripts/projection/eval_crossattn_entity_per.py --lang {en,ja} --n 300)

# Table A3 (Japanese 160-item reading benchmark; data/external/ja_tts_g2p_bench_v1.jsonl)
python3 scripts/analysis/numerals_verify_generic.py --module bench_hetnum77 --lang ja
python3 scripts/analysis/numerals_verify_generic.py --module bench_ext83 --lang ja
python3 scripts/analysis/bench_hetnum77_furigana.py                                  # BL reading decision (hetnum77)
python3 scripts/analysis/bench_hetnum77_furigana.py --module ext83 --cond bl
python3 scripts/analysis/bench_hetnum77_inj.py --align splice_clone                  # INJ for the BL-wrong items
python3 scripts/analysis/bench_hetnum77_inj.py --module ext83 --align splice_clone
python3 scripts/analysis/bench_hetnum77_furigana.py --cond injsc                     # and --module ext83 --cond injsc
python3 scripts/analysis/bench_hetnum77_furigana.py --cond ref_c                     # validation: ref_c / ref_w, both modules

# Table A5 (text- and weight-level baselines, voice-cloned; run in this order, appg_vc_gen.py
# checks its inputs against the baseline run)
python3 scripts/projection/eval_baseline_text_edit.py --lang ja --n 300              # Baseline and Text Hint
python3 scripts/regen2/appg_vc_gen.py --method g2p --seed 42
python3 scripts/regen2/appg_vc_gen.py --method rome --seed 42                         # ROME/MEMIT, layers (2, 5, 8)
python3 scripts/regen2/appg_vc_gen.py --method splice --seed 42 --pairs 0-149         # and --pairs 150-299
python3 scripts/regen2/appg_vc_gen.py --method textedit --regen --textedit_unit b7 && python3 scripts/regen2/appg_vc_apply_textedit.py --commit
python3 scripts/regen2/appg_vc_transcribe.py && python3 scripts/regen2/appg_vc_score.py

# Table A6 (full-sentence CER): computed by scripts/spanfix/cpuprop_cer_utmos_spk_recheck.py (section 1)

# Table A10 (six-language same-surface test)
python3 scripts/projection/eval_utmos_speaker_fixed.py --lang en --align splice \
  --reference_text 木曽川は、しばしば日本のライン川と呼ばれている。 --output_dir eval_results/utmos_speaker_fixed_en_n300_splice   # fr, de, ko likewise
python3 scripts/projection/eval_cjk_n300.py --lang ja --output_dir eval_results/cjk_n300_rewav --save_wavs
python3 scripts/projection/eval_cjk_n300_zhv2.py --lang zh --output_dir eval_results/cjk_n300_zh_v2 --save_wavs
python3 scripts/regen2/appk_regen_pairs_seed.py && python3 scripts/regen2/appk_apply_regen_pairs.py --apply   # pairs in pair_seeds.csv
python3 scripts/analysis/cjk_n300_furigana.py                                        # JA judge (Furigana Whisper)
python3 scripts/regen2/appk_zhjudge_run_gpu.py --stage_dir eval_results/_appk_zhjudge/rejudge_ctc   # ZH: whisper-pinyin CTC
python3 scripts/regen2/appk_enjudge_run_gpu.py --stage_dir eval_results/_appk_enjudge/judge_ctc     # EN: CMUdict + wav2vec2 CTC
python3 scripts/regen2/appk_run_ipa.py --langs en,fr,de,ko,ja,zh --out_dir eval_results/_appk_redo \
  --silent_scan eval_results/_appk_redo/silent_scan_n300.json                         # IPA change

# Table A11 and the compute figures of Appendix J (CPU)
python3 scripts/spanfix/c7_param_count.py
python3 scripts/spanfix/c6_compute_budget.py    # needs the Slurm accounting of the runs
```

The JA kana normalization of Table A10 is `norm_kana` in
`scripts/analysis/ja_wordrepl_judge.py` (also used by `scripts/aggregate_appendix.py`).
