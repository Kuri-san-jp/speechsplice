#!/bin/bash
# Table 1: Cross-lingual injection (Whisper-Inj / IPA Change) + UTMOS22 quality.
# Six languages (EN, JA, ZH, FR, DE, KO) at N=300 per language.
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO_ROOT="$PWD"

OUT_DIR="eval_results/table1_cross_lingual"
mkdir -p "$OUT_DIR"

for LANG in en ja zh; do
  echo "[Table 1] language=$LANG"
  # 1. Synthesise BL and INJ audio for each (entity, template) pair
  python scripts/table1_synth.py --lang "$LANG" --n 300 \
    --output_dir "${OUT_DIR}/audio_${LANG}"
  # 2. Whisper-Inj / IPA Change metrics
  python scripts/table1_score_change.py \
    --wav_dir "${OUT_DIR}/audio_${LANG}" --lang "$LANG" \
    --output_json "${OUT_DIR}/change_${LANG}.json"
  # 3. UTMOS22 baseline / injected scoring
  python src/score_utmos.py \
    --wav_dir "${OUT_DIR}/audio_${LANG}" --lang "$LANG" \
    --output_json "${OUT_DIR}/utmos22_${LANG}.json"
done

# Aggregate into table-style output
python scripts/aggregate_table1.py
