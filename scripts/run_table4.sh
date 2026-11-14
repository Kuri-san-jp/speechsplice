#!/bin/bash
# Table A1 (Appendix A): acoustic cross-attention mask ablation with length-preserving injection.
# Four conditions (BL / INJ / BL_mask / INJ_mask) on cross-culture pairs.
# Languages: EN / JA / ZH at N=300 each. ~5-6 hours per language on a 24GB GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO_ROOT="$PWD"

OUT_DIR="eval_results/table_a1_crossattn_mask/rerun"
mkdir -p "$OUT_DIR"

for LANG in en ja zh; do
  echo "[Table A1] language=${LANG} (length-preserving, cross-culture)"
  python scripts/table4_crossattn_mask.py \
    --lang "$LANG" \
    --cross_culture \
    --inject_mode length_preserve \
    --n 300 \
    --output_dir "${OUT_DIR}/${LANG}_lp"
  # results.json is written under each ${OUT_DIR}/${LANG}_lp/
done

echo "Done. Per-pair results: ${OUT_DIR}/{en,ja,zh}_lp/results.json"
echo "The shipped files behind the paper are eval_results/table_a1_crossattn_mask/{en,ja,zh}.json;"
echo "scripts/aggregate_appendix.py aggregates those (copy a rerun file there to aggregate it instead)."
