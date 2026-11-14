#!/bin/bash
# EdiTTS content replacement, extension batch (wordrepl pairs 100-299 -> gen_100..gen_299)
set -eu
ROOT=/mnt
EDITTS=$ROOT/third_party/EdiTTS
OUT=$ROOT/eval_results/editts_edit_wordrepl300
pip install --no-cache-dir "Cython<3" Unidecode 2>&1 | tail -1
if ! ls "$EDITTS"/model/monotonic_align/model/monotonic_align/core*.so >/dev/null 2>&1; then
  (cd "$EDITTS/model/monotonic_align" && mkdir -p model/monotonic_align \
    && python setup.py build_ext --inplace)
fi
cd "$EDITTS"
python edit_content.py -f "$OUT/filelist_p1.txt" -c checkpts/grad-tts-old.pt -t 1000 \
  -s "$OUT/wavs" --offset 0
echo "[run_editts_wr_p1] done"
