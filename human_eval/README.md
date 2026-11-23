# Human evaluation materials

Materials and results for the two listening studies: an **A/B pronunciation
preference test** and a **MOS naturalness test**. Everything here is
aggregate or item-level; participant-level responses are not released
(listeners consented to aggregate reporting only).

## Contents

| Path | Contents |
|---|---|
| `stimuli/` | Item manifests for the four preference-test sets (`en_items.json`, `ja_items.json`, `zh_items.json` for the name sets, `ja_het_items.json` for the Japanese same-surface set). 55 items each: 50 main, 3 catch, 2 practice. Fields: `pair_id`, `word_a`, `word_b`, `text_a`, `text_b`, `display_word`, `display_reading`, `item_type`, `block`. |
| `results/sheet2_main_results.csv` | Per-set totals: items, listeners, judgments, and the INJ / BL / neither counts and percentages. |
| `results/sheet3_item_level.csv` | **Per-item results for all 200 main items**: INJ / BL / neither counts, number of judgments, and the automatic (Whisper) verdict for each condition. |
| `results/primary_analysis.txt` | Output of the pre-specified preference analysis, reported before and after exclusions. |
| `results/mos_final.txt` | Output of the pre-specified MOS analysis, with the by-category breakdown and the ZH sensitivity analysis. |
| `results/demographics_aggregate.txt` | Aggregate self-reported age band, gender, and listening device, per language and block, for both studies. |
| `instructions_verbatim.md` | The full instruction text shown to listeners in EN, JA, and ZH for both studies. The Japanese and Chinese text is the text that was actually displayed, so it is left in the original language by design; English translations accompany it. |
| `screenshots/ui_flow/` | 21 screenshots of the instruction, main-intro, and trial screens of the deployed application (A/B preference and MOS, all three languages). The four `ab_*_instructions.png` files show the login screen that precedes the A/B instructions; the instruction text itself is in `instructions_verbatim.md`. |
| `screenshots/login_and_completion/` | 11 screenshots of the login and completion screens. Participant identifiers and completion codes are masked. |
| `analysis/analyze_preference_primary.py` | The pre-specified preference analysis (headline rate, item bootstrap, two-way cluster bootstrap, binomial GEE, quality control, human-vs-Whisper agreement). |
| `analysis/analyze_mos_final.py` | The pre-specified MOS analysis (validity rule, de-duplication, paired Wilcoxon by listener and by item, item bootstrap CI). |

## Design summary

- **A/B preference test.** Listeners heard a baseline (BL) and an injected
  (INJ) rendering of the same sentence in randomized on-screen order and chose
  which realized the reading shown on screen, or neither. 50 main items per
  set, split into two blocks of 25, with 3 catch items shared across blocks.
  20 listeners per set, 500 judgments per set.
- **MOS test.** Listeners rated naturalness on a 5-point scale. 50 clips per
  language per condition, 20 valid listeners per language, 1000 accepted
  ratings per language.
- **Headline estimand.** The unconditional INJ-selection rate: ties count as
  failures and stay in the denominator.
- **Primary interval.** An item bootstrap (seed 42, 2000 resamples). The
  item-majority rate carries a Wilson interval, and the response level is
  reported with a two-way listener × item cluster bootstrap. The pooled Wilson
  interval is reference only.
- **Primary test.** A binomial GEE against chance (50%), items as clusters,
  exchangeable working correlation, cluster-robust standard errors, two-sided,
  Bonferroni-corrected α = 0.0125. Where the GEE is completely separated (ZH,
  all 500 judgments selected INJ), the pre-specified fallback is an exact
  binomial test on the 50 item-level majorities.
- **Exclusions.** Catch pass rate below 2/3, straight-lining by on-screen
  position, or completion below 90%. Results are reported both before and
  after exclusions; in this study the two coincide for the preference test.

## What can be re-run from the released files

**Runnable, no GPU and no participant data** — recompute the published
preference rates and intervals from the item-level counts:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel) python3 scripts/aggregate_human_eval.py
```

This reproduces the INJ-selection rate and its item-bootstrap 95% interval for
all four sets, the item-majority counts, and the response-level breakdown.

**Runnable as a self-test** — exercise the real preference estimator on
synthetic data with planted ground truth:

```bash
python3 human_eval/analysis/analyze_preference_primary.py --simulate
```

**Not runnable from the released files.** Both scripts in `analysis/`, run in
their normal (non-`--simulate`) mode, read participant-level responses:

- `analyze_preference_primary.py` without `--simulate` needs the response
  database (`--db`), which is withheld. Its output is
  `results/primary_analysis.txt`.
- `analyze_mos_final.py` needs `mos50_ratings_dump.csv` and
  `deploy_mos50/filemap_neutral.json`, which are withheld. Its output is
  `results/mos_final.txt`.

Both are included so the pre-specified analysis can be audited as code. The
GEE statistics and the MOS means therefore cannot be recomputed here; they are
shipped verbatim as the two text files above.

## Note on the human-vs-Whisper agreement

`results/primary_analysis.txt` computes this agreement against the judge JSONs
in `eval_results/`, which is what `analyze_preference_primary.py` reads.
`scripts/aggregate_failure_cases.py` repeats the join on the released files and
prints the eight listener–ASR disagreements of Appendix H.

The `auto_inj_says_target` column of `sheet3_item_level.csv` holds the same
verdicts. Until 2026-11-14 it carried, for two EN items (0047 Delhi, 0056
Porsche), a verdict from a Whisper medium.en pass instead of the Whisper
large-v3 judge of the paper; those two cells were corrected, and the script
above checks the column against the judge files. `auto_bl_says_source` for EN
is still the Whisper medium.en verdict on the baseline clip (no large-v3
verdict on the baseline clips exists). The human counts were not changed.
