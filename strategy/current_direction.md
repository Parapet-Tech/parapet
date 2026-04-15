# Current Detection Direction

This note summarizes the current public strategy direction for Parapet's prompt-injection detection stack.

It exists to capture stable conclusions from recent research without importing local experiment notes into the canonical docs.

## Program Truth

- `L1` remains the fast lexical front door.
- `L2a` remains the optional heavier analysis slot.
- `L2b` is experimental, not a canonical public layer.
- The strategic direction is away from Prompt Guard 2 as the future of `L2a`.

## What Recent Research Established

### 1. L1 is useful, but should not be treated as a standalone final gate

Recent work reinforced that `L1` is best understood as a fast lexical sensor:

- it provides cheap coverage and strong routing signal
- it catches obvious prompt-injection language well
- harder benign families remain challenging when text is attack-shaped but not malicious

That means the right optimization target for `L1` is useful coverage and routing quality, not "solve the whole problem alone."

### 2. Lexical follow-on stages help, but do not fully solve the hard benign problem

Experimental lexical second stages improved some precision and routing behavior, but they did not fully eliminate the remaining hard benign families.

This keeps lexical helper stages in play for:

- precision recovery
- route-rate control
- latency shielding

But it also motivates the move toward a semantic `L2a`.

### 3. One prior CTF benign conclusion was invalidated by a data-labeling bug

Recent dataset debugging found that a previously used SaTML-derived "benign" challenge family was overwhelmingly attack traffic mislabeled as benign during staging.

That means:

- that family should not be treated as a valid benign quality gate
- earlier conclusions drawn from that family need to be interpreted carefully
- the remaining benign evaluation focus should stay on genuinely benign but difficult families

### 4. The main remaining benign problem is now clearer

The current benign focus is concentrated in families such as:

- `notinject`
- `atlas_neg`
- `wildguardmix`
- `bipia`

These represent different failure modes, including:

- trigger-word overfire
- quoted-attack use-vs-mention confusion
- indirect task-over-context confusion
- broad safety-topic confusion

## Current Near-Term Work

The active research path is:

1. improve `L1` data composition, especially benign ratio and benign family mix
2. preserve lexical helper stages only where they buy precision or latency
3. continue validating a Rust-first semantic `L2a`

The working question is no longer "should Parapet stay on PG2?" The answer there is effectively no.

The working question is:

"How far can improved `L1` composition plus an optional semantic `L2a` carry the stack before larger-model escalation becomes necessary?"
