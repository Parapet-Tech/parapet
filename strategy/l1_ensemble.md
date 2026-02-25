# L1 Ensemble: Specialist Classifiers

## Core Idea

Replace one general-purpose L1 classifier with an ensemble of specialist
classifiers, each trained on a specific attack family. The taxonomy is
**self-defining**: analyze where the current model fails, train a specialist
to cover that gap, repeat.

One linear classifier has one decision boundary. It compromises -- features
that help detect jailbreaks hurt detection of data exfiltration. N specialists
each get their own boundary optimized for one attack class. The union of N
hyperplanes carves a richer decision surface than one.

## Why This Beats an LLM Classifier

| Property | LLM (e.g. PromptGuard2) | Specialist Ensemble |
|----------|------------------------|--------------------|
| Latency | 100-300ms (transformer) | ~20us (20 phf_map lookups) |
| Dependency | External API / local model | Compiled into binary |
| Retrainability | Black box | Full control per specialist |
| Extensibility | Retrain entire model | Add one specialist |
| Availability | Network, GPU | Zero I/O |

20 phf_map lookups are four orders of magnitude faster than one transformer
inference. Same binary, same deployment, no GPU.

## Why Many Weak Learners Win

Each attack family has strong character n-gram signals that are nearly linearly
separable **on their own**. A generalist dilutes these signals because it must
share feature space across all attack types. A specialist amplifies them.

This is a mixture of experts without a routing network. Each specialist votes
independently. The verdict processor combines calibrated scores.

## The Self-Defining Taxonomy

The process:

1. Run eval on current ensemble (or single L1).
2. Collect false negatives. Group by failure mode.
3. When a cluster of FNs shares a distinct vocabulary or attack pattern,
   it defines a new specialist category.
4. Train that specialist on focused positive/negative data.
5. Add it to the ensemble. Re-run eval.
6. Repeat from (1).

The taxonomy is emergent, not designed up front. Weakness defines the next
specialist. This means the system improves monotonically -- each specialist
can only add coverage, never subtract it (given independent thresholds).

## Initial Taxonomy (from V4-10B eval failures)

Derived from 79 false negatives and 80 false positives in the V4-10B L1 eval
(24,915 L1-routed cases, 2026-02-22).

### Specialists

Thesis:
20 dumb models beats 1 smart. The classifiers must be tuned down to increase accuracy to 100% because false positives/negatives scale with number of models.

| # | Name | Signal | FN Examples |
|---|------|--------|-------------|
| 1 | **Instruction Override** | "ignore previous", "disregard", "forget all" | Current L1 baseline -- already strong here |
| 2 | **Roleplay Jailbreak** | Narrative framing, character assumption, scene-setting | jailbreak-cls-0015 (Star Wars scene), chatgpt-jb-0017 ("JB" persona), chatgpt-jb-0044 (Reservoir Dogs) |
| 3 | **Meta-Probe** | Questions about the system itself, prompt extraction | gandalf-0355 ("summary of previous instructions"), hackaprompt-0466 ("switch input and output") |
| 4 | **Privilege Escalation** | Authority claims, command language | l1-att-021 ("I am admin"), l1-att-022 ("sudo mode enabled") |
| 5 | **Exfiltration** | Data movement verbs, external destinations | l1-att-031 ("send to my server"), l1-att-036 ("redirect output") |
| 6 | **Obfuscation** | Encoded payloads, mixed-case, leetspeak, non-ASCII tricks | jailbreak-cls-0326 ("$gpt 4 Directive"), hackaprompt-0499 (mixed-language) |
| 7 | **Constraint Bypass** | Restriction-removal vocabulary | l1-att-035 ("skip constraints"), l1-att-036 ("bypass validation") |

### Why These Categories

Each category failed the generalist for a **different reason**:

- **Roleplay**: No classic injection n-grams. The signal is narrative structure,
  not "ignore"/"instructions".
- **Meta-Probe**: Very short prompts. Too few n-grams to accumulate score.
  A specialist tuned for meta-questioning has a lower feature-count threshold.
- **Privilege Escalation**: Different vocabulary entirely -- "sudo", "admin",
  "root" -- that the injection-trained generalist never learned.
- **Exfiltration**: Action-oriented ("send", "access", "redirect") rather than
  instruction-oriented.
- **Obfuscation**: Character-level entropy patterns, not word semantics.

### Why FPs Cluster Too

The 80 false positives also cluster:

- **notinject** (16 FPs): Benign prompts containing "ignore", "instructions".
  The generalist can't distinguish "ignore the error" from "ignore previous
  instructions". A specialist trained only on instruction override can use
  tighter hard-negative data.
- **no-robots** (22 FPs): Instructional content. Shares vocabulary with
  injection but not structure.
- **wildguardmix-ben**: Safety-adjacent discussions.

Each specialist **owns its own false positive surface**. The roleplay specialist
doesn't fire on "instructions" at all -- it's looking for narrative framing.

## Architecture

### Training

Each specialist is trained independently:

```
scripts/train_l1.py                    # Existing -- becomes "generalist" or retired
scripts/train_l1_specialist.py         # New -- trains one specialist
  --category roleplay_jailbreak
  --attack-data schema/eval/specialists/roleplay_attacks.yaml
  --benign-data schema/eval/specialists/roleplay_benign.yaml
  --hard-negatives schema/eval/specialists/roleplay_hard_negatives.yaml
  --out parapet/src/layers/l1_weights_roleplay.rs
```

Hard negatives are critical. Each specialist gets benign examples that share
vocabulary with its attack class but are not attacks. This is what makes
individual specialists more precise than the generalist.

**Analyzer per specialist.** All specialists default to `char_wb` n-grams
(same as current L1). The training script accepts per-specialist overrides:

- Most specialists: `analyzer='char_wb', ngram_range=(3,5)` -- standard.
- Obfuscation specialist: may benefit from wider windows `(2,6)` or raw
  `char` (no word boundary padding) to catch cross-word encoding tricks
  like `s.u.d.o` or `b y p a s s` where the signal is sub-word fragments
  spanning punctuation.
- Meta-Probe specialist: may benefit from `(2,4)` range since probes are
  short and the discriminative n-grams are smaller.

### Runtime

Each specialist is a phf_map + bias, same as current L1. The ensemble
scanner replaces `DefaultL1Scanner`:

```
for each specialist:
    raw_score = specialist.bias + sum(matched weights)
    if raw_score >= specialist.threshold:       # threshold in raw SVM margin space
        verdict = Block (report which specialist triggered)
    calibrated = sigmoid(raw_score)             # for combiner signal output only
```

**Threshold space: raw SVM margin.** Same as current L1 (l1.rs:191).
Thresholds are compared against the raw score (bias + sum of weights),
not the calibrated probability. Calibrated scores are computed for the
signal extractor output only — they feed the cross-layer verdict combiner.
This avoids the overhead of a sigmoid per comparison and keeps threshold
semantics consistent with the existing L1 config.

All specialists scan in parallel (sequential in practice -- each is ~1us).
The verdict processor sees per-specialist calibrated scores for combination
with L3/L4.

### Rust Types

```rust
pub struct SpecialistWeights {
    pub name: &'static str,
    pub bias: f64,
    pub threshold: f64,
    pub weights: &'static phf::Map<&'static str, f64>,
}

pub struct EnsembleL1Scanner {
    specialists: Vec<SpecialistWeights>,
}

impl L1Scanner for EnsembleL1Scanner {
    fn scan(&self, messages: &[Message], config: &L1Config) -> L1Result {
        // For each message, score against all specialists.
        // Per-message score = max calibrated across specialists.
        // Verdict = first message where any specialist breaches threshold.
    }
}
```

The `L1Scanner` trait doesn't change. The ensemble is just a different impl.
Existing tests and eval harness work unchanged against the trait.

**Implementation delta** (not covered in this strategy doc):

- `L1Config` gains optional `specialists` map (config/types.rs)
- `parapet.schema.json` gains specialist threshold schema
- Config loader parses specialist thresholds (config/loader.rs)
- `L1MessageScore` gains `specialist_name: Option<String>` field
- `SignalExtractor::extract_l1` passes specialist name as Signal category
  (currently hardcoded to `None` in extractor.rs:46)

### Config

```yaml
layers:
  L1:
    mode: block
    # Per-specialist thresholds allow tuning precision/recall independently.
    specialists:
      instruction_override: { threshold: 0.0 }
      roleplay_jailbreak:   { threshold: 0.5 }
      meta_probe:            { threshold: -0.5 }  # Lower threshold -- short text needs less signal
      privilege_escalation:  { threshold: 0.0 }
      exfiltration:          { threshold: 0.0 }
      obfuscation:           { threshold: 0.0 }
      constraint_bypass:     { threshold: 0.0 }
```

## Growth Model

This system grows monotonically:

1. **New attack class appears** -- train specialist #8, ship binary update.
2. **False positives in one specialist** -- retrain that specialist with
   hard negatives. Other specialists unaffected.
3. **Specialist becomes redundant** -- remove it. Others unaffected.
4. **Need more precision on one class** -- lower that specialist's threshold
   independently.

No specialist interacts with any other. No routing network. No retraining
the whole system. Each one is independently trainable, testable, deployable.

## Runtime Evolution: Separate Maps -> Unified SIMD Dictionary

Two phases, ship Phase 1 first:

### Phase 1: Separate phf_maps (ship first)

Each specialist has its own `phf_map`. For each n-gram in the input, loop
over N specialists and do N lookups. ~1us per specialist, ~20us for 20.

Advantages: simple, each specialist is independently swappable, easy to
add/remove without touching others. Perfect for iteration speed while the
taxonomy is still being discovered.

### Phase 2: Unified dictionary (when N grows large)

A build script merges all specialist maps into one:

```rust
// Auto-generated: one lookup scores all specialists simultaneously.
pub static UNIFIED: phf::Map<&'static str, [f64; N]> = phf_map! {
    " ign" => [0.842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // only instruction_override cares
    " act" => [0.0, 0.731, 0.0, 0.0, 0.0, 0.0, 0.0],    // only roleplay cares
    " sud" => [0.0, 0.0, 0.0, 0.912, 0.0, 0.0, 0.0],    // only priv_esc cares
    // ...
};
```

One n-gram parse pass. One phf lookup per n-gram. LLVM auto-vectorizes
the `[f64; N]` addition into SIMD instructions. Single-digit microseconds
regardless of specialist count.

The training script codegen handles the merge -- runtime code doesn't change,
just the weight file format. `EnsembleL1Scanner` gets a second impl that
reads the unified map instead of looping over separate maps.

**When to switch:** when N > ~30 and latency matters. Not before. Phase 1's
operational flexibility is more valuable than Phase 2's raw speed while
the taxonomy is still evolving.

## Combiner Boundary: Ensemble Is Internal to L1

The verdict combiner (see `implement/v4/signal.md`) operates at the
**cross-layer** level: L1 + L3 + L4 signals combined via boost/dampener.
It is designed for combining fundamentally different sensor types.

The specialist ensemble is **intra-L1**. All specialists are the same kind
of sensor (character n-gram linear models) looking at the same text. The
combiner's cross-layer boost would be wrong here — 3 specialists agreeing
is not the same signal as L1 + L3 + L4 agreeing.

### What each level does

| Level | Combination | Logic |
|-------|------------|-------|
| **Intra-L1** (ensemble) | `max(specialist_calibrated_scores)` | Any specialist firing is sufficient. Simple, fast. |
| **Cross-layer** (combiner) | Boost + dampener | L1 ensemble emits **one** Signal per message. Combiner sees it alongside L3/L4. |

### The ensemble is invisible to the combiner

The `SignalExtractor::extract_l1()` already takes `L1Result` and emits
`Signal` objects. It doesn't need to know whether those scores came from
one classifier or twenty. The `L1Scanner` trait boundary encapsulates this.

### But the specialist name flows through as category

The ensemble tags which specialist fired on the Signal:

```rust
Signal::new(
    LayerId::L1,
    SignalKind::Evidence,
    Some("roleplay_jailbreak".to_string()),  // which specialist
    max_calibrated,
    1.0,
)
```

This gives the combiner L1 intent attribution for free. The verdict
explainability becomes:

```
L1:roleplay_jailbreak(0.82) + L3:refusal_suppression(0.6) → boost(0.12) = 0.94 → Block
```

Instead of today's opaque:

```
L1 classifier score 2.341 >= threshold 0.0
```

### Observability payoff

The specialist name **is** the explanation. Every verdict tells a story:

- **What kind of attack**: the specialist name (roleplay_jailbreak, meta_probe, exfiltration)
- **How confident**: the calibrated score
- **What corroborated it**: which L3 pattern categories / L4 risk categories also fired
- **How to fix a false positive**: which specialist misfired, retrain it with hard negatives

No post-hoc labeling. No digging through 2,217 n-gram weights. The
classifier that fired tells you what it saw and why it cared.

For FP investigations: "the meta_probe specialist triggers on customer
support prompts asking 'what are your hours?'" — add those as hard
negatives, retrain one specialist, ship. The log told you exactly where
to look.

## Candidate Replacement for PromptGuard2

This ensemble is a candidate to replace PromptGuard2 (or any external LLM
classifier) by achieving comparable coverage through quantity of cheap
specialists rather than quality of one expensive model. The key tradeoff:

- LLM classifiers understand semantics. They catch novel attacks.
- Specialist ensembles understand vocabulary. They catch known attack families.

**This is a hypothesis, not a premise.** The ensemble must demonstrate:

1. Lower attack success rate (ASR) than current L1 on frozen benchmarks.
2. No benign block-rate regression (FP count <= current).
3. p95 latency materially better than PG2 (target: <50us vs PG2's 100-300ms).
4. Stable results across at least 2 independent benchmark runs.

Until these gates pass, PG2/L2a remain available as shadow-mode evidence
sensors. The ensemble augments L1 first; it replaces PG2 only after
benchmark validation.

For known attack families (which is most of what appears in production),
20 specialists with focused training data should match or exceed an LLM
classifier's recall at four orders of magnitude lower latency.

For truly novel attacks, L3/L4 (deterministic pattern scanning and
multi-turn heuristic scoring) remain the backstop. The ensemble's job is
to handle the long tail cheaply so downstream layers only process what
the ensemble can't confidently classify.
