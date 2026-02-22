# Phase 4 Roadmap (Revised)

## Goals and Constraints

1. Primary goal: intent classification using signal processing.
2. Constraint: close tech debt first.
3. SDK direction: TS SDK should reach Python parity or improve both experiences.
4. Model direction: PromptGuard spike is reference input, not a hard blocker.
5. Benchmark direction: run public benchmarks early and after changes to drive learning.
6. No fixed deadline.

## Decisions Locked (2026-02-18)

1. `engine.on_failure` stays in config and is implemented. Default remains `open`.
2. `engine.on_failure: closed` means any layer runtime error blocks (including partial failures such as L2a timeout while other layers pass).
3. Benchmark protocol is frozen before first run and includes dataset version/commit pinning + dataset artifact hash in results.
4. Hybrid rollback uses one enum key: `enforcement_mode: block | hybrid | shadow`, default `block`.
5. SDK distribution and packaging work is deferred to Phase 5 (`implement/v5/roadmap.md`).
6. CI expansion scope is limited to missing lanes: Rust, Python, and integration smoke.
7. Add two fast SDK E2E suites (Python and TS) covering benign and attack traffic before SDK work to surface integration issues early.

## Priority Model

1. `P0`: blocks intent-classification outcome or introduces production risk.
2. `P1`: improves quality, recall/FP balance, or adoption.
3. `P2`: scale and acceleration work.

## Prioritized Backlog

| ID | Priority | Work Item | Why Now | Depends On |
|---|---|---|---|---|
~~| V4-01 | P0 | Rust trust-span correctness fixes | Prevent silent under-enforcement and trust misattribution | None |~~
~~| V4-00 | P0 | SDK E2E suites (Python + TS, benign + attack) | Catch SDK/engine integration regressions before SDK-focused changes | V4-01 |~~
~~| V4-02 | P0 | Python SDK high-severity fixes | Remove bypass/leak/cross-session contamination risk | V4-00 |~~
~~| V4-02T | P0 | TS SDK post-ship debt closeout | Ensure TS parity and capture deferred fixes immediately after release | V4-00 |~~
~~| V4-03A | P0 | `engine.on_failure` semantics record | Decision is made; lock exact semantics in docs/tests before implementation | None |~~
| V4-03B | P0 | `engine.on_failure` implementation | Align runtime behavior with chosen policy | V4-03A |
~~| V4-10 | P1 | Data expansion + benchmark suite integration | Complete dataset inventory and integrate external benchmark suites before baselining | None |~~
~~| V4-10B | P1 | L1 retrain: restore imoxto + expand training recipe | L1 classifier retrained with expanded recipe; F1 95.5%->96.2% | V4-10 |~~
| V4-07A | P1 | Public benchmark baseline run (pre-change) | Establish baseline before hybrid/tuning changes | V4-10, V4-10B |
~~| V4-11 | P1 | Streaming redaction gap spike (compressed SSE) | Convert known limitation into measured fix path | V4-01 |~~
| V4-04 | P0 | Hybrid combiner enforcement with rollback controls | Core phase goal, but must be safe to disable quickly | V4-01, V4-02, V4-03B |
| V4-05A | P0 | Combiner rule spec freeze (exoneration + L2a) | Prevent scope drift in exoneration work | V4-04 |
| V4-05B | P0 | Exoneration + L2a combiner implementation | Reduce FP while preserving detection | V4-05A |
| V4-06 | P1 | PG2-22M latency + threshold tuning | Improve L2a operating point under FP budget | V4-05B |
| V4-07B | P1 | Public benchmark post-change run | Measure improvement vs baseline | V4-05B, V4-06, V4-07A |
| V4-09 | P2 | CI expansion (Rust + Python + integration) | TS CI exists; fill missing lanes only | V4-02, V4-02T |

Dependency notes:
V4-10 moved before V4-07A so benchmarks run on the complete dataset corpus. Expanding data after baselining would force a re-baseline.
V4-07A intentionally does not depend on V4-02T because baseline benchmarks run through engine and Python/zero-SDK paths.
If benchmark execution later includes TS transport in the measured path, add V4-02T as a hard dependency.

## Execution Plan

1. **Stage A (`P0` hardening and semantics lock)**: V4-01, V4-00, V4-02, V4-02T, V4-03A.
2. **Stage B (`P1` data + spike)**: V4-10 and V4-11 in parallel, then V4-10B (L1 retrain), then V4-07A.
3. **Stage C (`P0` core pipeline)**: V4-03B, then V4-04.
4. **Stage D (`P0/P1` refinement)**: V4-05A, V4-05B, V4-06.
5. **Stage E (parallel tracks)**:
   Track 1: V4-07B.
   Track 2: V4-09.

Critical path note:
V4-04 depends on V4-02 by design to avoid changing enforcement while known SDK bypass/leak issues are unresolved.
If V4-02 slips, the fallback is to run V4-04 in shadow-only behind a non-default flag until V4-02 closes.

## Task Definitions

~~### V4-01: Rust trust-span correctness fixes~~

**Description**: Fix ambiguous trust-span mapping for repeated content and invalid byte-range handling.
  1. P1: X-Guard-Trust span mapping is ambiguous for repeated content
     parapet/src/engine/mod.rs:3115 uses body_str.find(escaped_interior) per message, which always returns the first
     occurrence. If two messages share the same content, spans can attach to the wrong message (or be missed). Existing
     tests only cover unique/single-message cases (parapet/src/engine/mod.rs:2287).
     Recommendation: map spans via structured message offsets (or at least track a moving search cursor), and add
     duplicate-content regression tests.
  2. P1: Trust-span byte offsets can become invalid and get silently ignored
     Engine computes spans from escaped JSON text (parapet/src/engine/mod.rs:3112), then downstream L3 span checks do
     filter_map and drop invalid ranges (parapet/src/layers/l3_inbound.rs:235). For escaped/unicode-heavy content this
     can under-enforce untrusted-span policy.
     Recommendation: validate span bounds at parse time and reject/normalize invalid spans; add tests with \n, quotes,
     and non-ASCII payloads.

**Acceptance Criteria**:
- [ ] Repeated message content maps spans to the correct message deterministically.
- [ ] Invalid span offsets are rejected or normalized explicitly (not silently dropped).
- [ ] Regression tests cover escaped text, quotes/newlines, and repeated-content scenarios.

**Verification**:
- [ ] `cargo test` passes for trust/span and inbound policy tests.
- [ ] New regressions fail on prior behavior and pass on fixed behavior.

**Files likely affected**:
- `parapet/src/engine/mod.rs`
- `parapet/src/layers/l3_inbound.rs`
- `parapet/tests/integration.rs`

~~### V4-00: SDK E2E suites (Python + TS, benign + attack)~~

**Description**: 
Add two fast end-to-end suites that run SDKs against a live engine on benign and attack traffic to surface integration issues before SDK implementation work.
Create e2e verify skill that can test py, ts, or both. Functional readme for agents.

**Acceptance Criteria**:
- [ ] Python SDK E2E suite exists and runs against live engine.
- [ ] TypeScript SDK E2E suite exists and runs against live engine.
- [ ] Both suites include benign/regular traffic cases (expected allow).
- [ ] Both suites include attack traffic cases (expected block/shadow according to test config mode).
- [ ] Suites are fast enough for frequent local/CI execution.
- [ ] Create e2e verify skill that can test py, ts, or both.
- [ ] Update AGENTS.md to reference tests / skill


**Verification**:
- [ ] Local commands are documented and green.
- [ ] CI lane runs both E2E suites and reports pass/fail independently.

**Files likely affected**:
- `parapet-py/tests/*`
- `parapet-ts/test/*`
- `.github/workflows/*`
- `implement/v4/ci_workflows.md`

~~### V4-02: Python SDK high-severity fixes~~

**Description**: Fix session propagation, failopen exception handling, trust registry isolation, and streaming request handling.

**Acceptance Criteria**:
- [ ] `session()` baggage reaches outbound requests.
- [ ] Engine-refused errors failopen using `httpx` exception classes.
- [ ] Trust registry is isolated per request/session lifecycle.
- [ ] Streaming requests do not throw on trust scan path.

**Verification**:
- [ ] Python SDK tests pass with regression coverage.
- [ ] Repro scripts confirm closure of each high-severity finding.

**Files likely affected**:
- `parapet-py/parapet/__init__.py`
- `parapet-py/parapet/transport.py`
- `parapet-py/parapet/trust.py`
- `parapet-py/tests/test_transport.py`

~~### V4-02T: TS SDK post-ship debt closeout~~

P0 (Security/Correctness)

1. AbortSignal.any() requires Node 20.3+ but package.json claims >=18.0.0
- transport.ts:129 — AbortSignal.any([originalReq.signal, timeoutSignal])
- AbortSignal.any() was added in Node 20.3.0. On Node 18/19, this crashes at runtime.
- Node 18 is already EOL (April 2025). Fix is likely to bump engines.node to >=20.0.0, but must be a conscious
decision.

2. Baggage header is replaced, not merged
- transport.ts:110 — headers.set("baggage", baggage) overwrites any caller-provided baggage.
- Python transport (transport.py:109-111) merges: f"{existing},{baggage}" if existing else baggage.
- If an OpenAI SDK user sets their own baggage header, TS silently drops it.

P1 (Quality/Parity)

3. session() doesn't validate init() was called
- Python session() raises RuntimeError if init() wasn't called first (__init__.py:94-97).
- TS session() (index.ts:44-54) silently succeeds — context is set but no fetch wrapper exists, so headers are never
injected. Silent misconfiguration.

4. No early config file validation on init()
- Python init() checks Path(config_path).exists() and raises FileNotFoundError with a clear message.
- TS init() delegates to EngineManager.start() which calls readFileSync — the error is a raw ENOENT, not
user-friendly.

5. Windows sidecar signal handling
- sidecar.ts uses SIGTERM/SIGKILL/SIGINT, which don't work the same on Windows.
- SIGTERM → TerminateProcess (immediate kill, no graceful shutdown).
- Signal handlers (SIGINT, SIGTERM) may not fire reliably.
- Python's subprocess.terminate() has the same issue, so this is parity, but still debt.

6. untrusted() called outside session throws, but message doesn't mention init
- index.ts:70-73 throws about session(), which is correct, but unhelpful if the real problem is that init() wasn't
called either.

  ---
 A/C

1. Fix P0 items #1 and #2 (node version + baggage merge).
2. Fix P1 item #3 (session guard).
3. Track P1 #4-5 and P2 #6 as explicit backlog items (can be deferred).
4. Tests pass after fixes.

**Verification**:
- [ ] `npm run typecheck` and `npm test` pass after fixes.

**Files likely affected**:
- `parapet-ts/src/*.ts`
- `parapet-ts/test/*.test.ts`

### V4-03A: `engine.on_failure` semantics record

**Description**: Record the already-approved semantics so implementation and tests are unambiguous.

**Acceptance Criteria**:
- [ ] `on_failure` is documented as supported and implemented (not removed).
- [ ] Default is explicitly `open`.
- [ ] `closed` is explicitly defined as: any layer runtime error => block.
- [ ] Partial-failure examples are documented (for example L2a timeout while L1/L3 pass still blocks in `closed`).

**Verification**:
- [ ] Semantics note committed in local planning docs and referenced by V4-03B tests.

### V4-03B: `engine.on_failure` behavior implementation

**Description**: Implement the V4-03A decision so config semantics and runtime behavior match.

**Acceptance Criteria**:
- [ ] Behavior in runtime matches config/docs exactly.
- [ ] `on_failure: open` allows request flow on layer runtime errors.
- [ ] `on_failure: closed` blocks on any layer runtime error (including partial failures).
- [ ] Tests cover both modes with at least one partial-failure case.

**Verification**:
- [ ] Config-driven tests pass.

**Files likely affected**:
- `parapet/src/config/types.rs`
- `parapet/src/config/loader.rs`
- `parapet/src/engine/mod.rs`
- `README.md`

### V4-07A: Public benchmark baseline run (pre-change)

**Description**: Run baseline benchmark suite before hybrid/tuning changes.

**Acceptance Criteria**:
- [ ] Benchmark set is fixed and documented (initially JailbreakBench + BIPIA).
- [ ] Dataset versions are pinned by commit/tag.
- [ ] Dataset artifact hash is recorded with results.
- [ ] Baseline matrix includes `no guardrails`, `current stack`, and layer ablations currently available.
- [ ] Metrics include ASR, benign block rate, utility degradation, latency.

**Verification**:
- [ ] Baseline artifact has config hash + commit hash.

**Files likely affected**:
- `implement/v4/public_benchmarks.md`
- `schema/eval/results/*`

### V4-04: Hybrid combiner enforcement with rollback controls

**Description**: Move from shadow-only combiner to enforceable hybrid mode.

**Acceptance Criteria**:
- [ ] Hybrid mode uses combiner action for non-atomic evidence.
- [ ] Atomic/policy violations remain hard blocks.
- [ ] Single config key exists for rollout/rollback: `enforcement_mode: block | hybrid | shadow`.
- [ ] Default enforcement mode is `block` (legacy behavior).
- [ ] Rollout plan includes shadow-first validation window and canary enforcement ramp.

**Verification**:
- [ ] Unit/integration tests cover `block`, `hybrid`, and `shadow/signal` modes.
- [ ] Rollback kill switch test proves immediate reversion to legacy behavior.

**Files likely affected**:
- `parapet/src/engine/mod.rs`
- `parapet/src/signal/combiner.rs`
- `parapet/src/config/types.rs`
- `schema/examples/parapet.yaml`

~~### V4-11: Streaming redaction gap spike (compressed SSE)~~

**Description**: Investigate the known compressed-streaming limitation in L5a and choose a bounded remediation path.

**Acceptance Criteria**:
- [ ] Reproduction harness exists for uncompressed SSE, gzip SSE, and deflate SSE.
- [ ] At least 2 candidate remediations are tested (for example: force `Accept-Encoding: identity` for stream requests vs. continuous decompress-and-scan).
- [ ] Candidate evaluation includes relative performance budgets and explicit failure behavior (pass-through + telemetry on decompression failure).
- [ ] Decision record captures selected path, expected risk reduction, and implementation scope.

**Verification**:
- [ ] Spike report includes detection impact, latency impact, and compatibility notes by provider.
- [ ] Follow-up implementation task is created with explicit in/out-of-scope.

**Files likely affected**:
- `implement/v4/streaming.md`
- `parapet/src/engine/mod.rs`
- `parapet/src/stream/mod.rs`

### V4-05A: Combiner rule spec freeze (exoneration + L2a)

**Description**: Define concrete thresholds and signal sources before implementation.

**Scope freeze (initial defaults)**:
- Benign-context source: `block_patterns` category `benign_context` plus structured payload parse-success signal.
- Quiet thresholds: `L1 < 0.20` and max non-benign `L3 < 0.15`.
- Corroboration thresholds: `L2a >= 0.40` plus (`L1 >= 0.35` or `L3 >= 0.35`) for boost eligibility.
- Threshold provenance: initial priors from current eval score distributions and existing combiner constants; treated as starting values and recalibrated after V4-07A/V4-07B.

**Acceptance Criteria**:
- [ ] Spec is documented with threshold constants and rationale.
- [ ] Non-goals are explicit (no EWMA/hysteresis in this task).

**Verification**:
- [ ] Team review sign-off on the frozen rule set.

### V4-05B: Exoneration + L2a combiner implementation

**Description**: Implement V4-05A spec with tests and eval deltas.

**Acceptance Criteria**:
- [ ] Benign-context signals dampen multiplicatively per spec.
- [ ] L2a-only weak signals are dampened when both quiet thresholds hold.
- [ ] L2a corroboration boost applies only when corroboration thresholds hold.

**Verification**:
- [ ] Property tests for order invariance and monotonicity constraints.
- [ ] Eval delta report shows FP/FN movement vs baseline.

**Files likely affected**:
- `parapet/src/signal/combiner.rs`
- `parapet/src/signal/extractor.rs`
- `schema/examples/default_block_patterns.yaml`

### V4-06: PG2-22M tuning

**Description**: Execute latency + threshold/weight sweeps and choose operating point.

**Acceptance Criteria**:
- [ ] Stage-1 latency validation completed.
- [ ] Sweep matrix complete and reproducible.
- [ ] Chosen config meets benign FP budget and latency target.

**Verification**:
- [ ] Results table and command log committed.

**Files likely affected**:
- `implement/v4/tune_p22m.md`
- `schema/eval/eval_config_l2a_only.yaml`
- `schema/examples/parapet.yaml`

### V4-07B: Public benchmark post-change run

**Description**: Re-run benchmark suite after V4-05B/V4-06 and compare against V4-07A.

**Acceptance Criteria**:
- [ ] Same benchmark protocol as baseline.
- [ ] Comparative report highlights gains/regressions and next actions.

**Verification**:
- [ ] Baseline-vs-post artifact with commit/config parity.

### V4-09: CI expansion (TS exists)

**Description**: Extend CI from existing TS workflow to missing pipelines only.

**Acceptance Criteria**:
- [ ] Rust lane exists: `cargo fmt --check`, `cargo clippy`, `cargo test`.
- [ ] Python lane exists: `ruff check`, `pytest`.
- [ ] Python SDK E2E lane (benign + attack) exists and is green.
- [ ] TypeScript SDK E2E lane (benign + attack) exists and is green.
- [ ] Integration smoke lane exists: start engine and run cross-SDK smoke tests.
- [ ] Existing TS workflow remains source of truth for TS lane.

**Verification**:
- [ ] All workflows green on representative PR.

**Files likely affected**:
- `.github/workflows/*.yml`
- `implement/v4/ci_workflows.md`

~~### V4-10: Data expansion + benchmark suite integration~~

**Description**: Complete the eval dataset inventory and integrate external benchmark suites so V4-07A baselines on the full corpus.

Three workstreams:

1. **Missing datasets**: Fetch and convert the datasets identified in `implement/v4/data.md`:
   - JailbreakV-28K/JailBreakV-28k
   - DhruvTre/jailbreakbench-paraphrase-2025-08
   - microsoft/llmail-inject-challenge
   - hendzh/PromptShield
   - geekyrakshit/prompt-injection-dataset
   - xTRam1/safe-guard-prompt-injection

2. **BIPIA**: Frozen core benchmark per `implement/v4/public_benchmarks.md`. Fetch dataset, write fetch script, convert to eval YAML.

3. **External benchmark suites** (research + integration spike):
   - **garak** (NVIDIA): Learn what's needed to run garak probes against the engine's proxy endpoint. Document setup, probe selection, and how to capture results in our eval format.
   - **AgentDojo**: Understand requirements for running AgentDojo indirect injection benchmarks against the engine. Document feasibility and integration path.
   - If either tool can run against an OpenAI-compatible proxy endpoint, wire it up. If not, document what's blocking and what adaptation is needed.

**Acceptance Criteria**:
- [ ] Dataset inventory updated with provenance/licensing notes for all new datasets.
- [ ] BIPIA fetch script exists and eval YAML is generated.
- [ ] All new datasets wired into eval configs.
- [ ] Baseline eval run succeeds with expanded dataset corpus.
- [ ] garak integration: documented setup, tested against engine, or blocker documented.
- [ ] AgentDojo integration: documented setup, tested against engine, or blocker documented.

**Verification**:
- [ ] `parapet-eval` runs clean with all new datasets included.
- [ ] Integration notes for garak/AgentDojo committed under `implement/v4/`.

**Files likely affected**:
- `implement/v4/data.md`
- `scripts/fetch_*.py`
- `schema/eval/*`
- `implement/v4/benchmark_suites.md` (new — garak/AgentDojo integration notes)

~~### V4-10B: L1 retrain — restore imoxto + expand training recipe~~

**Status**: DONE (2026-02-22, commit 37b6952)

**Description**: Retrained L1 char n-gram classifier with expanded recipe. Originally planned to remove imoxto, but re-fetching from HuggingFace and testing showed it improves recall. 14 experiments guided the final config.

**Result**: F1 95.5% -> 96.2%, FP 86 -> 80 on 63,616-case eval suite. Beats original model on every metric.

**Final training recipe** (config #9, 14 datasets, 25,700 samples, 4,744 atk / 20,956 ben, uncapped):

| Role | Dataset | Source | Samples |
|------|---------|--------|--------:|
| attack | gandalf | Lakera/gandalf_ignore_instructions | 1,000 |
| attack | chatgpt_jailbreak | rubend18/ChatGPT-Jailbreak-Prompts | 78 |
| attack | imoxto | imoxto/prompt_injection_cleaned_dataset-v2 | 1,000 |
| attack | hackaprompt | hackaprompt/hackaprompt-dataset | 2,000 |
| attack | jailbreak_cls | jackhhao/jailbreak-classification | 666 |
| benign | notinject | leolee99/NotInject (staging/) | 339 |
| benign | wildguardmix | allenai/wildguardmix (staging/) | 2,000 |
| benign | awesome_chatgpt | fka/awesome-chatgpt-prompts | 1,190 |
| benign | teven | teven/prompted_examples | 1,285 |
| benign | dahoas | Dahoas/synthetic-hh-rlhf-prompts | 4,000 |
| benign | chatgpt_prompts | MohamedRashad/ChatGPT-prompts | 360 |
| benign | hf_instruction | HuggingFaceH4/instruction-dataset | 327 |
| benign | no_robots | HuggingFaceH4/no_robots | 9,455 |
| benign | ultrachat | HuggingFaceH4/ultrachat_200k | 2,000 |

**Key findings**:
- Uncapping `--max-per-file` (was 1000, now 0) restored natural 18:82 attack:benign ratio — biggest single improvement.
- Hard-negative benign data (NotInject trigger-word prompts, WildGuardMix safety-adjacent) reduces FPs without hurting recall.
- Indirect injection data (BIPIA, PromptShield, Deepset) hurts L1 — those are L3's job.
- Supplemental jailbreak attacks (jailbreak_cls) fixed chatgpt_jailbreak recall (15% -> 92%).

**Completed A/C**:
- [x] imoxto re-fetched from HuggingFace (1000 attack cases, label=1 only).
- [x] New datasets added: jailbreak_cls attacks, notinject benign, wildguardmix benign.
- [x] `l1_weights.rs` regenerated (2,217 features).
- [x] `l1_holdout.yaml` regenerated (5,140 cases).
- [x] F1 beats baseline: 96.2% vs 95.5%.
- [x] `cargo build` + `cargo test` (19/19) green.
- [x] Eval delta documented in `schema/eval/results/v4_10b_summary.md`.
- [x] `--output` flag added to parapet-eval (bypasses PowerShell UTF-16).
- [x] `schema/eval/staging/` directory for datasets not yet in eval baseline.
- [x] Fetch scripts: imoxto, notinject, wildguardmix, protectai-validation.
- [x] README.md training data description updated.
- [x] dataset_inventory.md updated (imoxto restored, new datasets added).

Full experiment log: `schema/eval/results/v4_10b_summary.md`

## Phase Exit Criteria

Phase 4 is complete when all of the following are true:
- [ ] Hybrid combiner enforcement is available and verified with rollback kill switch.
- [ ] V4-07A and V4-07B are complete with baseline-vs-post comparison artifacts.
- [ ] Benchmark artifacts include pinned dataset versions and dataset artifact hashes.
- [ ] Post-change evaluation meets agreed FP/FN guardrails against baseline.
- [ ] High-severity SDK debt (V4-02 and V4-02T `P0` items) is closed or explicitly escalated.
- [ ] Python and TS SDK E2E suites (benign + attack) are stable and green.
- [ ] Streaming spike (V4-11) is resolved to a chosen remediation path with implementation task tracked.
- [ ] CI expansion scope (V4-09) is implemented.

## Recommended Immediate Start

1. V4-01 Rust trust-span correctness.
2. V4-00 SDK E2E suites (benign + attack).
3. V4-02 Python SDK high-severity fixes.
4. V4-02T TS SDK post-ship debt closeout.
5. V4-03A `engine.on_failure` semantics record.
