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

Use a local `ledger.yaml` in this directory when running:

```powershell
python -m parapet_data curate --ledger adjudication\ledger.yaml ...
```
