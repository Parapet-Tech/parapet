# L1 Evaluation Snapshots

This note records versioned `L1` evaluation milestones.

Purpose:

- preserve historical baselines
- make major weight revisions legible
- keep `strategy/l1.md` focused on role and architecture rather than turning it into a changelog

These snapshots use the external challenge set: 2,386 attack and 2,386 benign samples spanning diverse prompt-injection attacks and hard benign, attack-shaped text.

## Snapshot Table

| Version | Date    | Training Shape | F1 | Precision | Recall | FPR | FP | FN | Notes |
|--------|---------|----------------|---:|----------:|-------:|----:|---:|---:|-------|
| `production_published` | 2026-03 | published production weights | 0.754 | 0.684 | 0.840 | 38.9% | 928 | - | Best published weights at the time of release. High recall, but too much benign overfire. |
| `v7_1to1` | 2026-04 | `v7` corpus, `1:1` benign ratio | 0.780 | 0.770 | 0.789 | 23.6% | 562 | 503 | Major cleanup step relative to published production weights. |
| `v8_2to1_hardneg` | 2026-04 | `2:1` benign ratio, Recipe `H` | 0.815 | 0.869 | 0.767 | 11.6% | 276 | 555 | Current best challenge-set result for stack use. Precision and false-positive rate improved sharply; recall trade is acceptable for `L1` as a sensor. |

## Current Read

The most important movement is not just `F1`.

From `production_published` to `v8_2to1_hardneg`:

- false positives dropped from `928` to `276`
- false positive rate dropped from `38.9%` to `11.6%`
- precision rose from `0.684` to `0.869`

This suggests the main recent gain came from improved benign composition rather than model-family change alone.

