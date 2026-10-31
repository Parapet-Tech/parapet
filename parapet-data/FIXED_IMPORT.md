# Fixed imports

`python -m parapet_data fixed-import --spec fixed.json --output out/` composes
pinned local JSONL inputs independently of MirrorSpec. JSON and YAML specs are
accepted. There is no sampling, text transformation, deduplication, signature
filtering, or sample weighting.

Example spec (pins must be replaced with actual SHA256 values):

```yaml
mode: fixed_import
name: example
seed: 42
inputs:
  - name: examples
    path: examples.jsonl
    sha256: <64 lowercase hex characters>
    text_column: text
    id_column: id
    label_column: label
    group_column: family
expected_counts: {benign: 10, malicious: 10}
splits: {train: 0.8, val: 0.1, holdout: 0.1}
```

Paths resolve relative to the spec. Each input needs a label column or
`fixed_label` (`benign` or `malicious`); supplying both asserts agreement.
Noncanonical labels and conflicting existing `label`/`content` fields fail
closed. IDs must be nonempty strings and globally unique, including unselected
rows. Text and present, non-null groups must be nonempty strings. Original fields
remain in each output row; `content` and `label` provide the runner contract.

Optional `selection: {path: arms.json, sha256: <pin>, arm: example}` reads
`{example: {examples: [id1, id2]}}`. Parts must exactly match input names.
Missing selected IDs, duplicate IDs, bad pins, and expected label count
mismatches abort before writing artifacts. A zero-total selected population also
aborts; individual selection parts may be empty. Expected counts may assert one or
both labels. Input order determines output row order.

Splits default to all train, with empty val and holdout artifacts. Otherwise,
SHA256 of the seed and group determines the split against cumulative fractions
in train/val/holdout order. Groups share a namespace across inputs, allowing
leader IDs to bind benign rows to their mirrors. Without a group column the
row ID is its group. With a configured column, absent or null values use a
tagged row identity, distinct from real groups and other ungrouped rows. Fractions are probabilistic targets, not exact quotas.

JSONL serialization escapes Unicode characters, including all `str.splitlines()`
boundaries, preserving decoded text and metadata exactly for the runner.

Outputs are `curated.jsonl`, `train.jsonl`, `val.jsonl`, `holdout.jsonl`,
`manifest.json`, and `composition.json`. The manifest follows CurationManifest
and adds a `fixed_import` receipt containing the full spec (input and selection
pins, arm, split policy, expected counts). Content hashes use the runner's
whitespace-stripped hashing convention without changing stored text. The
semantic hash uses the shared canonical function. The manifest timestamp is
a fixed epoch marker for byte reproducibility, not the wall-clock run time.
Use a fresh output directory for each import. Private specs and outputs belong
in ignored local directories.
