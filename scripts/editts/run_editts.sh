#!/bin/bash
# EdiTTS content replacement for Table 2 (my-lab-image, repository at /mnt).
# The original ran as three jobs, and EdiTTS seeds once per job (torch.manual_seed(1234) in
# edit_content.py), so the split is kept:
#   filelist_p1.txt  pairs   0-99   (run_editts_wr_p1.sh)
#   filelist_p2.txt  pairs 100-199  (run_editts_p2.sh; that job stopped after its first 100 lines,
#                                    which are the pairs used; the remaining lines were never run)
#   filelist_p3.txt  pairs 200-299  (run_editts_p3.sh; sentence-final punctuation guarded)
# The original wrote pairs 100-299 to eval_results/editts_edit_en_n100/wavs and linked them into
# eval_results/editts_edit_wordrepl300/wavs; here all 300 go to the latter directly.
# Needs third_party/EdiTTS with scripts/editts/edit_content.patch (scripts/setup_third_party.sh).
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
python edit_content.py -f "$ROOT/scripts/editts/filelist_p1.txt" -c checkpts/grad-tts-old.pt -t 1000 -s "$OUT/wavs" --offset 0
head -n 100 "$ROOT/scripts/editts/filelist_p2.txt" > /tmp/editts_filelist_p2_first100.txt
python edit_content.py -f /tmp/editts_filelist_p2_first100.txt -c checkpts/grad-tts-old.pt -t 1000 -s "$OUT/wavs" --offset 100
python edit_content.py -f "$ROOT/scripts/editts/filelist_p3.txt" -c checkpts/grad-tts-old.pt -t 1000 -s "$OUT/wavs" --offset 200
echo "[run_editts] done"
