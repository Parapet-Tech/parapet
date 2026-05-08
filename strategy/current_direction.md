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
- `L3` is the orthogonal sensor/router layer. It is sensor-first, deterministic,
  and returns only `allow` or `block`.
- There is no specialist/escalation branch in the current target architecture.
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

### 3. Orthogonal sensors are the next deployable bet

The next useful layer is not another semantic model. It is a set of fast,
mechanically different sensors over normalized text and upstream layer evidence:

- entropy and compression shape
- structural/markup shape
- obfuscation indicators
- sizing and line/span features
- optional token-shape features only if their runtime cost is justified

The router should be deterministic policy code. Offline trees may help discover
rules, but production should prefer auditable rules and config-hashed
thresholds over an opaque second classifier.

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
4. Do not port a sensor to Rust until the Python simulation earns it.

The working question is:

"How much deployable coverage can deterministic patterns, the lexical SVM, and
orthogonal sensors provide before the product should simplify around reporting
and policy rather than add another inline model?"
