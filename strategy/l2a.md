# L2a: Data Payload Scanning

L2a scans untrusted data payloads (tool results, retrieved documents, API responses) for prompt injection using Meta's [Prompt Guard 2](https://huggingface.co/meta-llama/Prompt-Guard-2-22M) ONNX model combined with structural heuristics. This catches injections hiding in data that bypass pattern-based detection.

L2a runs on CPU. No GPU required.

## Model options

| Model | Parameters | Size on disk | Overall recall | Best for |
|-------|-----------|-------------|---------------|----------|
| `pg2-22m` | 22M | ~85 MB | 59.5% | Low-latency blocking path |
| `pg2-86m` | 86M | ~1.1 GB | 61.9% (+2.4pp) | Shadow analysis or high-latency budgets |

### Latency snapshot: 22M vs 86M

| Metric | 22M | 86M | Ratio |
|--------|----:|----:|------:|
| Model load | 731 ms | 2,502 ms | 3.4x |
| Short text (median) | 129 ms | 393 ms | 3.0x |
| Medium text (median) | 133 ms | 666 ms | 5.0x |
| Long text (median) | 137 ms | 613 ms | 4.5x |

Observed on recent local benchmark runs. Absolute numbers vary by CPU and runtime environment.

## Config

```yaml
policy:
  layers:
    L2a:
      mode: shadow         # shadow | block
      model: pg2-22m       # pg2-22m | pg2-86m
      model_dir: null      # optional
      pg_threshold: 0.5
      block_threshold: 0.8
      heuristic_weight: 0.3
      fusion_confidence_agreement: 0.95
      fusion_confidence_pg_only: 0.7
      fusion_confidence_heuristic_only: 0.4
      max_segments: 16
      timeout_ms: 200
      max_concurrent_scans: 4
```

Model path resolution order:

1. `policy.layers.L2a.model_dir` (config)
2. `$PARAPET_MODEL_DIR` (environment)
3. `~/.parapet/models/` (default)

Fetch model files explicitly before enabling L2a:

```bash
parapet-fetch --model pg2-22m --skip-checksum
```

If L2a is configured but the model cannot be loaded, engine startup fails closed.

## Detection results: pg2-22m

Evaluated on 10,489 cases (5,747 attack, 4,742 benign) across 9 open-source datasets. All attacks are prompt injection payloads delivered as tool results / data payloads.

### Attack detection (recall)

| Dataset | Cases | Detected | Missed | Recall | F1 |
|---------|------:|--------:|-------:|-------:|---:|
| Gandalf | 1,000 | 973 | 27 | 97.3% | 98.6% |
| Giskard | 35 | 33 | 2 | 94.3% | 97.1% |
| Jailbreak-CLS | 666 | 594 | 72 | 89.2% | 94.3% |
| HackAPrompt | 2,000 | 1,441 | 559 | 72.0% | 83.8% |
| JailbreakBench | 100 | 29 | 71 | 29.0% | 45.0% |
| deepset | 203 | 41 | 162 | 20.2% | 33.6% |
| Mosscap | 1,743 | 310 | 1,433 | 17.8% | 30.2% |
| **Total** | **5,747** | **3,421** | **2,326** | **59.5%** | |

Zero false positives across all attack datasets (100% precision).

### False positive rate

| Dataset | Cases | False positives | Accuracy |
|---------|------:|---------------:|--------:|
| HC3 (benign) | 4,399 | 1 | 100.0% |
| deepset (benign) | 343 | 1 | 99.7% |
| **Total** | **4,742** | **2** | **99.96%** |

False positive rate: **0.04%** (2 out of 4,742 benign inputs incorrectly blocked).

### What the numbers mean

PG2-22M is strongest against **direct, explicit injection** -- attack strings that contain clear instruction-override language:

- Gandalf (97.3%), Giskard (94.3%), Jailbreak-CLS (89.2%) -- these datasets contain direct "ignore instructions", "you are now", "reveal the password" style attacks. PG2-22M catches nearly all of them.

- HackAPrompt (72.0%) -- mixed difficulty. Many direct injections caught, but competition entries include creative obfuscation that evades the model.

- JailbreakBench (29.0%), deepset (20.2%), Mosscap (17.8%) -- these datasets contain **indirect, subtle, or heavily obfuscated** attacks. The 22M model lacks capacity to recognize these patterns.

PG2-22M is best used as a **complement to L3 pattern matching**, not a replacement. L3 catches attacks by structural patterns (regex); L2a catches attacks by semantic understanding. Together they cover more ground than either alone.

## Detection results: pg2-86m

Evaluated on 10,489 cases (5,747 attack, 4,742 benign) across 9 open-source datasets. Same test suite as pg2-22m.

### Attack detection (recall)

| Dataset | Cases | Detected | Missed | Recall | F1 |
|---------|------:|--------:|-------:|-------:|---:|
| Gandalf | 1,000 | 993 | 7 | 99.3% | 99.6% |
| Giskard | 35 | 33 | 2 | 94.3% | 97.1% |
| Jailbreak-CLS | 666 | 626 | 40 | 94.0% | 96.9% |
| HackAPrompt | 2,000 | 1,498 | 502 | 74.9% | 85.6% |
| JailbreakBench | 100 | 31 | 69 | 31.0% | 47.3% |
| deepset | 203 | 50 | 153 | 24.6% | 39.5% |
| Mosscap | 1,743 | 327 | 1,416 | 18.8% | 31.6% |
| **Total** | **5,747** | **3,558** | **2,189** | **61.9%** | |

Zero false positives across all attack datasets (100% precision).

### False positive rate

| Dataset | Cases | False positives | Accuracy |
|---------|------:|---------------:|--------:|
| HC3 (benign) | 4,399 | 1 | 100.0% |
| deepset (benign) | 343 | 1 | 99.7% |
| **Total** | **4,742** | **2** | **99.96%** |

False positive rate: **0.04%** (2 out of 4,742 benign inputs incorrectly blocked).

### Head-to-head: pg2-22m vs pg2-86m

| Dataset | 22M Recall | 86M Recall | Delta |
|---------|----------:|----------:|------:|
| Gandalf | 97.3% | 99.3% | +2.0 |
| Giskard | 94.3% | 94.3% | 0.0 |
| Jailbreak-CLS | 89.2% | 94.0% | +4.8 |
| HackAPrompt | 72.0% | 74.9% | +2.9 |
| JailbreakBench | 29.0% | 31.0% | +2.0 |
| deepset | 20.2% | 24.6% | +4.4 |
| Mosscap | 17.8% | 18.8% | +1.0 |
| **Overall** | **59.5%** | **61.9%** | **+2.4** |
| **FP rate** | **0.04%** | **0.04%** | **0.0** |

The 86M model gains +1-5pp recall across every dataset with no increase in false positives. The largest gains are on medium-difficulty datasets (Jailbreak-CLS +4.8, deepset +4.4). Both models share the same weakness on indirect/obfuscated attacks (JailbreakBench, Mosscap).

### Which model to use

- **pg2-22m**: Default for production blocking. Similar precision and much lower latency.
- **pg2-86m**: Use in shadow mode for sampled/high-risk traffic when +2.4pp recall justifies 3x-5x latency.

## How L2a works

```
Data payload arrives (tool result, retrieved doc)
  -> Segment extraction (split into scannable chunks)
  -> Prompt Guard 2 ONNX inference (malicious probability per segment)
  -> Structural heuristic scan (instruction patterns, role markers)
  -> Sensor fusion (weighted combination of PG2 + heuristic scores)
  -> In block mode, block if fused score >= block_threshold
```

L2a only scans **untrusted data payloads** -- tool results, retrieved documents, and other content that could contain injected instructions. It does not scan user messages (that's L1's job) or system prompts (those are trusted).

## Build requirements

L2a requires the `l2a` cargo feature and MSVC toolchain on Windows:

```bash
# Build
cargo +stable-x86_64-pc-windows-msvc build --features l2a --release

# Run tests
cargo +stable-x86_64-pc-windows-msvc test --features l2a

# Run eval
parapet/target/release/parapet-eval.exe \
  --config schema/eval/eval_config_l2a_only.yaml \
  --dataset schema/eval/
```

## Eval datasets

| Dataset | Source | Type | Cases |
|---------|--------|------|------:|
| Gandalf | [Lakera Gandalf](https://gandalf.lakera.ai/) | Attack | 1,000 |
| Giskard | [Giskard LLM Scan](https://github.com/Giskard-AI/giskard) | Attack | 35 |
| Jailbreak-CLS | [Jailbreak Classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Attack | 666 |
| HackAPrompt | [HackAPrompt Competition](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) | Attack | 2,000 |
| JailbreakBench | [JailbreakBench](https://jailbreakbench.github.io/) | Attack | 100 |
| deepset | [deepset prompt injections](https://huggingface.co/datasets/deepset/prompt-injections) | Attack | 203 |
| Mosscap | [Mosscap](https://huggingface.co/datasets/Mosscap/prompt_injection) | Attack | 1,743 |
| HC3 | [HC3 (Human ChatGPT Comparison)](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | Benign | 4,399 |
| deepset | [deepset prompt injections](https://huggingface.co/datasets/deepset/prompt-injections) | Benign | 343 |
