# Parapet v0.1.0 Release Notes

Parapet v0.1.0 is an early local-first research release. It is not stable or
production-ready.

## Supported first-use environment

- macOS 11 or newer on Apple silicon (arm64)
- Python 3.10 or newer
- the exact wheel
  `parapet-0.1.0-py3-none-macosx_11_0_arm64.whl`

The packaged 15-minute evaluator path is supported only on macOS arm64.

## Downloadable artifacts

- `parapet-v0.1.0-macos-arm64-evaluator.tar.gz` — deterministic local evaluator
- `parapet-v0.1.0-release-evidence.tar.gz` — release verification evidence
- `parapet-v0.1.0-RELEASE_SHA256SUMS` — detached checksums for release assets
- platform wheels named `parapet-0.1.0-py3-none-<platform>.whl`

The evaluator contains its own `EVALUATOR_SHA256SUMS`; verify it before use.

## Known issues and limitations

- The evaluator uses deterministic offline fixtures and makes no real model
  request.
- Its bounded `write_file` examples demonstrate local policy, receipt,
  verification, restoration, and cleanup behavior only.
- It does not establish general prompt-injection robustness, production
  readiness, crash safety, or support for every client or platform.
- Protected runtime and file-effect lifecycle behavior are POSIX-only in this
  release.
- Python dependency installation may use the configured package index unless
  the dependency is already present in an approved offline cache.
- npm publication is not part of v0.1.0.

See `START_HERE.md` for the supported evaluator journey and
`RESULT_TEMPLATE.md` for the result record.
