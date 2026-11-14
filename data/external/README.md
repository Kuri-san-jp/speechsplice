# ja-tts-g2p-bench (Table A3)

`ja_tts_g2p_bench_v1.jsonl` is the copy of ja-tts-g2p-bench used for Table A3 and for 38 of
the 50 items of the Japanese same-surface listening set: 160 items with the fields `id`,
`surface`, `target`, `reading_expected`, `reading_alt`, `stim_expected`, `stim_alt`,
`category`, `note`. It was obtained in July 2026; the upstream revision was not recorded.

- Source: https://huggingface.co/datasets/kawajiri-tellernovel/ja-tts-g2p-bench
- License: CC BY-SA 4.0. This file and the per-item results that contain its sentences
  (`eval_results/table_a3_ja_reading/`) are redistributed under the same license.

`data/bench_hetnum77.py` (heteronyms and contextual numerals, 77 items) and
`data/bench_ext83.py` (the other six categories, 83 items) turn the items into
source/target pairs for the scripts of `REPRODUCE.md` §3.
