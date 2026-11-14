#!/bin/bash
# Table A5 (Appendix E): text-side and weight-side pronunciation-control baselines (JA, N=300).
# Conditions: Baseline / G2P Baseline / Text Hint / ROME-MEMIT / SpeechSplice.
# Output: eval_results/appendix_f_baselines/results.json
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO_ROOT="$PWD"

OUT_DIR="eval_results/appendix_f_baselines"
mkdir -p "$OUT_DIR"

# Path to the SpeechSplice INJ transcripts (checkpoint.json from the
# generation run). Not shipped; override to match your own run.
INJ_JSON="${INJ_JSON:-${OUT_DIR}/speechsplice_ja_n300/checkpoint.json}"
if [ ! -f "$INJ_JSON" ]; then
  echo "error: INJ transcripts not found at $INJ_JSON" >&2
  echo "       Synthesise the SpeechSplice condition first, then re-run with" >&2
  echo "       INJ_JSON=/path/to/checkpoint.json bash scripts/run_appendix_f.sh" >&2
  exit 1
fi

# 1. SpeechSplice + Baseline + Surface Text Edit + Text Hint (synthesises BL/INJ/B1/B7)
python scripts/appendix_f_eval_baseline_text_edit.py \
  --lang ja --n 300 \
  --output_dir "${OUT_DIR}/baseline_text_edit"

# 2. G2P Baseline (replace source surface with G2P-predicted katakana)
python scripts/appendix_f_eval_g2p_baseline.py \
  --output_dir "${OUT_DIR}/g2p_baseline" --seed 42

# 3. ROME / MEMIT (L258_memit) on the same cross-culture split
python scripts/appendix_f_eval_rome.py \
  --output_dir "${OUT_DIR}/rome_L258_memit" \
  --seed 42 --layers 2 5 8 --edit_mode memit --label L258_memit

# 4. SpeechSplice INJ transcripts.
#    NOTE: these are NOT shipped in this repository. Step 5 reads them from
#    "${INJ_JSON}" below; synthesise the SpeechSplice condition first (see
#    scripts/table1_generate_wordrepl.py) and point INJ_JSON at the
#    checkpoint.json it writes.

# 5. Unified TRM (strict v2) + G2P-CER Match (CER<=0.30) scoring
python src/score_trm_g2pcer.py \
  --bl_b1_b7_json "${OUT_DIR}/baseline_text_edit/results.json" \
  --inj_json "${INJ_JSON}" \
  --rome_json "${OUT_DIR}/rome_L258_memit/rome_transcripts.json" \
  --g2p_json  "${OUT_DIR}/g2p_baseline/g2p_transcripts.json" \
  --output    "${OUT_DIR}/results.json"

python scripts/aggregate_appendix_f.py
