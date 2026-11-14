"""TTS26-side entry point for the pyopenjtalk-free Appendix G scorer (2026-09-27).

pyopenjtalk-free counterpart of ``score_unified_trm_g2pcer.py`` (which is kept
unchanged so that ``eval_results/_baselines_cc_n300_unified/final_results.json``
stays reproducible).  The implementation lives in one place,
``speechsplice_release/src/score_trm_g2pcer_nopyojt.py``; this wrapper only
puts that directory on sys.path (host or /mnt container layout) and runs it
with the same CLI.

Example (CPU, host):
  taskset -c 0-7 env OMP_NUM_THREADS=1 python3 \
    scripts/projection/score_unified_trm_g2pcer_nopyojt.py \
    --bl_b1_b7_json eval_results/baseline_text_edit_ja_n300/results.json \
    --inj_json eval_results/t5gemma_cc_wavs_ja_n300/checkpoint.json \
    --rome_json eval_results/_baselines_cc_n300_unified/rome_L258_memit/rome_transcripts.json \
    --g2p_json eval_results/_baselines_cc_n300_unified/g2p_baseline/g2p_transcripts.json \
    --output eval_results/_appg_redo_nopyojt/final_results_nopyojt.json
"""
import os
import sys

if "pyopenjtalk" not in sys.modules:
    sys.modules["pyopenjtalk"] = None

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "src"))  # (release) the scorer lives in src/

from score_trm_g2pcer_nopyojt import main  # noqa: E402

if __name__ == "__main__":
    main()
