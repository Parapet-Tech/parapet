# Current Detection Direction

This note summarizes the current public strategy direction for Parapet's
prompt-injection detection stack. It captures stable conclusions without
importing local experiment artifacts into canonical docs.

## Program Truth

- The inline semantic-transformer L2 path is closed for the hot path.
- The target inbound stack is `L0 -> L1 -> L2 -> L3 -> L4 -> upstream`.
- `L1` is the deterministic pattern gate.
- `L2` is the lightweight lexical classifier, currently the compiled char
  n-gram SVM still implemented under legacy `L1` names.
- `L3` is the typed orthogonal-evidence and router surface. The runtime default
  emits no signals; the current opt-in shadow router is attribution-only and
  cannot affect a verdict.
- There is no separate specialist/escalation layer in the current target
  architecture. L1-internal specialist weight tables remain an implementation
  option inside the lexical-classifier layer.
- Legacy `L2a` / Prompt Guard style payload analysis is not the strategic path.

## What Recent Research Established

### 1. L1/L2 naming changed, but the useful parts remain

The old stack used `L1` for the SVM and `L3_inbound` for pattern scanning.
The target taxonomy moves the cheaper deterministic pattern gate ahead of the
SVM:

1. pattern scanning is cheaper and more explainable
2. pattern evidence is useful input to later layers
3. the SVM is better understood as a lightweight lexical sensor, not the first
   logical gate

Implementation names will lag the target taxonomy until a deliberate rename
lands. `strategy/layers.md` is the mapping source of truth during that
transition.

### 2. Small inline transformers did not earn the hot-path budget

Residual testing showed that the latency-feasible transformer envelope depended
on aggressive truncation/threading assumptions and failed effectiveness on the
hard tail. Larger DeBERTa-class models exceeded the CPU latency budget by a
wide margin.

The conclusion is narrow but load-bearing: Parapet should not keep reopening
the inline transformer ladder as the default L2 fallback.

### 3. Orthogonal sensors remain a staged direction

The next useful layer proposed by this direction is not another semantic model.
It is a set of fast, mechanically different sensors over normalized text and
upstream layer evidence:

- entropy and compression shape
- structural/markup shape
- obfuscation indicators
- sizing and line/span features
- optional token-shape features only if their runtime cost is justified

This is direction, not current runtime capability. The intended router should be
deterministic policy code. Offline trees may help discover rules, but production
should prefer auditable rules and config-hashed thresholds over an opaque second
classifier. Config-hashed sensor thresholds are not implemented today.

### Sensor work status at v3

The structural sensor surface is stable at `v3`. These observation sensors run
offline through `python -m parapet_data scope-audit`. The audit baseline combines
the mechanical blob detector (`v2`), the zero-width signal split (`v3`), and the
documented validation caveats for Arabic public holdout coverage and
`escape_sequence_blob` promotion. Runtime `L3` carries typed evidence and a
no-op router by default; no observation sensor currently affects a runtime
verdict.

No further structural sensors are planned in the immediate term. New structural
sensor proposals must be gated on measurement evidence, such as real eval misses
involving the target pattern, not on pattern prevalence in benign corpora alone.
Local research trail may exist under gitignored `implement/research-findings/`
for parked ideas such as homoglyph recall gaps or fragmented visible text, but
those files are not published artifacts or active implementation plans.

No public `L1`/`L2`/`L3` stack-interaction measurement has landed. The existing
Phase 1 residual feature analysis is not a sensor simulation, so orthogonal
sensor promotion remains open rather than measured program progress.

### 4. Product fallback is simplification, not escalation

If `L3` sensors do not materially move the residual, the fallback is to simplify
around `L1 + L2 + policy/reporting`, not to add an inline specialist model.
Coverage gaps should be reported per language and per attack family rather than
hidden behind an unbounded model ladder.

## Current Near-Term Work

1. Keep the naming transition explicit in docs and configs.
2. Simulate orthogonal sensors against the Phase 1 residual.
3. Promote only sensors that clear both gates:
   - at least 30% targeted FN-family reduction at acceptable FP cost
   - about 1pp absolute stack improvement or comparable routed-volume reduction
   No sensor has yet been evaluated against either gate.
4. Mark each sensor's validation class explicitly. Synthetic-only fixtures are
   useful regression checks, but they do not support corpus precision/recall
   claims when the public corpus has no footprint for the target pattern. See
   `strategy/sensor_validation.md`.
5. Do not add a runtime `L3` sensor until the Python simulation earns it. The
   existing Rust observation sensors predate this rule and remain offline audit
   tooling.

The working question is:

"How much deployable coverage can deterministic patterns, the lexical SVM, and
orthogonal sensors provide before the product should simplify around reporting
and policy rather than add another inline model?"
