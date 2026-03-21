# Adjudication

This directory is for local adjudication state used by the `parapet-data`
verification and curation pipeline.

Versioned here:
- directory structure and ignore rules
- optional example ledger files

Not versioned here:
- live adjudication ledgers
- review exports
- other locally derived cleanup data

Local review artifacts belong under:
- `review/exports/`
- `review/batches/`
- `review/classified/`
- `review/manifests/`

`work/` is retired. Keep local review and repair state here instead of under
the repository root.

Use a local `ledger.yaml` in this directory when running:

```powershell
python -m parapet_data curate --ledger adjudication\ledger.yaml ...
```
