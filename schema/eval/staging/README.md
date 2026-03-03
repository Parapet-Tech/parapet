# Staging Artifacts

This directory contains data produced by `python -m parapet_data stage`.

Run from `parapet/`:

```bash
python -m parapet_data stage \
  --index ../TheWall/INDEX.yaml \
  --output schema/eval/staging/ \
  --holdout-sets schema/eval/l1_holdout.yaml \
                 schema/eval/t3/l1_holdout_generalist_curated_100k.yaml \
                 schema/eval/challenges/tough_attack_v1/tough_attack_mirror_v2_novel.yaml \
                 schema/eval/challenges/tough_neutral_v1/tough_neutral_mirror_v2_novel.yaml
```

## File types

- `*_attacks_staged.yaml`: staged malicious rows for one dataset/language.
- `*_benign_staged.yaml`: staged benign rows for one dataset/language.
- `staging_manifest.json`: cumulative staging summary across runs.
- `staging_rejected.jsonl`: per-row hard rejections from the most recent run.
- `*_quarantine.jsonl`: benign rows flagged by attack-signature heuristic (requires review).
- `*.hashes`: holdout hash sidecar cache files for faster startup.

## Row schema

Each staged YAML row uses this contract:

```yaml
- content: "<normalized text>"
  label: malicious|benign
  language: EN|RU|ZH|AR
  source: "<dataset_name>"
  reason: "<attack_reason>"
  content_hash: "<sha256(content.strip())>"
```

## Manifest contract

`staging_manifest.json` fields:

- `timestamp`: ISO timestamp of the latest run.
- `thewall_index_hash`: SHA256 of `INDEX.yaml` used by the latest run.
- `datasets_processed`: per-dataset records (`rows_read`, `rows_staged`, `rows_rejected`, `rows_quarantined`, `rejection_reasons`, `by_reason`, `by_language`).
- `total_staged`: sum of `rows_staged` across accumulated dataset entries.
- `total_rejected`: sum of `rows_rejected` across accumulated dataset entries.
- `output_hashes`: SHA256 by staged output filename.

Manifest behavior:

- Runs are cumulative.
- Dataset entries are keyed by `name`; re-staging a dataset replaces that dataset's prior entry.
- Totals are recomputed from the merged dataset list.

## Gate expectations

The stage pipeline is fail-closed with these key checks:

- label resolution
- script/language validation
- content minimum quality
- dedup/cross-contamination
- holdout-leakage exclusion
- benign attack-signature quarantine
- reason assignment with confidence floor (attacks)

If a dataset lacks enough metadata in `INDEX.yaml`, set `staging_status: needs_review` and skip until fixed.
