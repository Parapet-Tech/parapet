# Parapet v0.2.0 Release Notes

Parapet v0.2.0 is the first Parapet PyPI release delivered as five
platform-specific wheels with the `parapet` and `parapet-engine` executables
bundled. Its product source is the independently reviewed v0.1.0 snapshot plus
release mechanics and version metadata only.

This remains a research release. It is not stable or
production-ready.

## Install

Use an explicit compatible-version constraint:

```bash
python -m pip install "parapet>=0.2.0,<0.3"
```

Exact reproducibility uses `parapet==0.2.0`. The supported wheel platforms are
Linux x86_64, Linux arm64, macOS arm64, macOS x86_64, and Windows x86_64.

The existing public PyPI `parapet==0.1.0` is a legacy universal, pure-Python
SDK from February 2026. It remains immutable and installable. On an unsupported
platform, an unconstrained install can resolve that older universal package
because no compatible v0.2.0 wheel exists. The release verification therefore
covers both a clean v0.2.0 install and an actual `0.1.0 -> 0.2.0` upgrade.

## Reviewed assets

- five `parapet-0.2.0-py3-none-<platform>.whl` files from one successful
  attempt-1 tag workflow run;
- `parapet-v0.2.0-macos-arm64-evaluator.tar.gz`;
- `parapet-v0.2.0-release-evidence.tar.gz`;
- `parapet-v0.2.0-RELEASE_SHA256SUMS`.

The evaluator archive is a deterministic local witness package. The detached
manifest covers exactly the five wheels, evaluator archive, and evidence
archive.

## TypeScript/npm status

TypeScript source metadata is aligned to `0.2.0`, but this release does **not**
publish `@parapet-tech/parapet@0.2.0` to npm. The npm registry contained only
`0.1.0` and `0.1.1` at the 2026-11-10 preflight; that inventory is re-read
before draft staging. Do not infer npm 0.2.0 availability from the Git tag or
GitHub release.

## Source-line note

The default branch is a newer development product line and uses unpublished
`0.3.0.dev0` (Python) / `0.3.0-dev.0` (Cargo and TypeScript) metadata. Those
main-branch artifacts are not v0.2.0 release inputs. The v0.2.0 tag binds only
the separately reviewed `release/v0.2.0` source candidate descended from
`602738d7fabfe05a167cc010fd18975df3f32026`.

Publication uses GitHub OIDC Trusted Publishing. No PyPI API token or other
upload password is stored as a repository, organization, or environment
secret.

## Known issues and limitations

- This release is a research artifact, not a production support commitment.
- Only the five named wheel platforms are supported.
- The bundled evaluator uses deterministic local fixtures and no real model;
  it is not an independent usability validation.
