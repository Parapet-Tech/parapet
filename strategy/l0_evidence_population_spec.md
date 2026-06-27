# L0 evidence population spec

Date: 2026-06-27
Owner: routing
Status: accepted and implemented

## Goal

Wire `L0Evidence` population for the existing routing evidence context without
changing enforcement behavior. The first slice should make
`removed_invisible_count` observable to routing consumers while preserving the
current L0 sanitize semantics, including trust-span remapping.

## Current State

- `routing::L0Evidence` already exists and is deliberately content-free:
  numeric and boolean fields only, no offsets, tokens, ranges, or policy labels.
- `RoutingEvidenceContext::new` accepts `l0: Option<&[L0Evidence]>`, but
  `engine::process_request` currently passes `None`.
- `L0Normalizer::normalize` already performs:
  `NFKC -> strip_html -> remove_invisible_chars -> replace_mixed_script_confusables`.
- The recent Default_Ignorable work extended `is_invisible`; those display-path
  removals must count the same way as the older invisible set.
- `normalize_for_comparison` has an additional comparison-only strip for
  `U+034F`, `U+17B4`, and `U+17B5`; it emits no evidence and is not part of this
  slice.

## Contract

Add a metrics-producing L0 sanitize helper that reuses the same private stage
functions as `L0Normalizer::normalize`:

```rust
pub struct L0NormalizationEvidence {
    pub pre_char_len: usize,
    pub post_char_len: usize,
    pub pre_byte_len: usize,
    pub post_byte_len: usize,
    pub removed_invisible_count: usize,
    pub confusable_replacement_count: usize,
    pub html_stripped: bool,
}

pub fn normalize_with_evidence(input: &str) -> (String, L0NormalizationEvidence)
```

For whole-message normalization, a message helper may wrap this per-string
evidence with `message_index`; use a distinct wrapper name only if it adds that
message-local field.

The ordinary `Normalizer::normalize` hot path must not do evidence work. Use a
private normalization core for the stage order, and have both
`L0Normalizer::normalize` and `normalize_with_evidence` call that core. This is
important because `remap_trust_spans` calls `normalizer.normalize(prefix)` for
span-boundary prefixes; evidence allocation and counting should not run for
those internal prefix normalizations.

The evidence-producing message helper should also avoid a second full-message
L0 pass. It may reuse the normalized output returned by `normalize_with_evidence`
as the message content, then call `remap_trust_spans` with the injected
normalizer for span-prefix remapping. Add a debug-only agreement check between
the concrete L0 output and the injected normalizer so future normalizer
divergence is caught in tests/debug builds.

The helper should not introduce a second invisible-character table. It should
derive `removed_invisible_count` by applying the existing L0 stage order and
testing characters with the same `is_invisible` predicate used by
`remove_invisible_chars`.

Counting semantics:

- `pre_*` count the L0 input string before NFKC. This is after request parsing,
  trust-header parsing, and trust assignment, not raw request bytes.
- `post_*` count the final L0 normalized output after confusable replacement,
  before role-marker neutralization.
- `removed_invisible_count` counts characters removed by the display-path
  invisible strip after NFKC and HTML stripping.
- Invisible characters inside text swallowed by the HTML-strip stage are not
  counted in `removed_invisible_count`; they are removed as part of the earlier
  HTML-strip mutation, not by the invisible-strip stage.
- `html_stripped` is true when the HTML-strip stage mutates the NFKC string. It
  means the stage changed content; it does not prove well-formed HTML was
  present because the current stripper also mutates stray angle-bracket spans.
- `confusable_replacement_count` counts exact character positions changed by the
  final mixed-script confusable replacement stage. The current replacement stage
  emits one output char for each input char, so compare the pre-confusable and
  post-confusable strings position-wise by `char`.
- `role_marker_neutralized_count` remains populated from
  `neutralize_role_markers`. Role-marker neutralization uses equal-length space
  replacement and is represented only by this field, not by `post_*`.

Engine behavior:

- When L0 mode is `sanitize`, preserve the existing
  `normalize_messages_with_spans` behavior. The evidence-producing path must
  normalize message content in the same L0 stage order and still call
  `remap_trust_spans` so span offsets remain valid for
  `neutralize_role_markers` and downstream span-aware logic.
- Collect one `routing::L0Evidence` per message and pass
  `Some(&l0_evidence)` into `RoutingEvidenceContext::new`.
- When L0 is absent or not in `sanitize` mode, pass `None`.
- First integration must not change verdicts, blocking, thresholds, or default
  router behavior.

## Non-goals

- Do not count comparison-only removals from `normalize_for_comparison`.
- Do not expose raw content, spans, offsets, matched tokens, or codepoint lists
  through routing evidence.
- Do not add an enforcement dependency on L0 evidence.
- Do not replace `normalize_messages_with_spans` with a message-only normalize
  helper that drops span remapping.
- Do not expose private L0 stage functions to engine just to compose evidence
  there.

## Implementation Sketch

1. In `normalize/mod.rs`, add `L0NormalizationEvidence` and
   `normalize_with_evidence`.
2. Factor the existing stage order into private helpers so
   `L0Normalizer::normalize` stays cheap while `normalize_with_evidence`
   performs the extra counts. Do not make `normalize` delegate through the
   evidence-producing API.
3. Add a span-preserving message-slice helper, likely:
   `normalize_messages_with_spans_and_evidence(...) -> Vec<L0MessageEvidence>`.
   If `L0MessageEvidence` is introduced, it should be the per-string
   `L0NormalizationEvidence` plus `message_index`, not a second evidence
   vocabulary.
   This helper must keep the current `normalize_messages_with_spans` sequence:
   normalize the full message content once, reuse the normalized output from
   `normalize_with_evidence`, and call `remap_trust_spans` against the same
   normalizer. Return a normalize-local evidence type; engine maps it into
   `routing::L0Evidence` to avoid a `normalize -> routing` dependency.
4. In `engine::process_request`, replace the L0 sanitize call with the
   span-preserving evidence helper and merge in the role-marker neutralization
   count by `message_index`.
5. Bind the collected `Vec<routing::L0Evidence>` in a scope that outlives
   `RoutingEvidenceContext::new`, then pass `Some(&l0_evidence)` into the
   context.

## Tests

Targeted Rust tests are enough for this slice:

- `normalize_with_evidence_counts_invisible_removed`: includes at least one old
  invisible and one recent CHANGE 1 code point such as `U+E0080`; expected count
  reflects both.
- `normalize_with_evidence_does_not_count_comparison_only`: `U+034F` survives
  display normalization and does not increment `removed_invisible_count`.
- `normalize_with_evidence_reports_html_and_confusable`: HTML stage and mixed
  Cyrillic/Latin confusable replacement are reflected without exposing content.
- `normalize_with_evidence_html_stage_mutation_semantics`: stray angle-bracket
  stripping sets `html_stripped`, and an invisible inside a stripped tag is not
  counted by `removed_invisible_count`.
- Engine or routing-context test proving L0 sanitize passes `Some(&[L0Evidence])`
  to a test router, and non-sanitize/no-L0 passes `None`.
- Span regression: a trusted message with an untrusted span containing an
  invisible character still has remapped spans after evidence-producing
  normalization, and the same message produces L0 evidence.
- Role-marker regression: role-marker neutralization increments
  `role_marker_neutralized_count` but does not change the `post_*` L0 lengths.
- Existing normalize tests continue to pass.

Suggested validation:

```bash
cd parapet
cargo test normalize --lib
cargo test routing --lib
cargo test engine --lib
git diff --check
python3 scripts/check_no_data_commit.py --all
```

Full `cargo test` is preferred before merge if the local environment allows it.

## Review Questions

1. `removed_invisible_count` should count at the actual invisible-strip stage
   after NFKC and HTML stripping. Keep the name and document the stage semantics.
2. `confusable_replacement_count` should be exact. The current confusable stage
   is one output char per input char, so position-wise char comparison is enough.
3. The helper should live in `normalize`; engine should not compose private L0
   stages directly.
