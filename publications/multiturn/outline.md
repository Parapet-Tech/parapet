# Peak + Accumulation: A Proxy-Level Scoring Formula for Multi-Turn LLM Attack Detection

## Paper Outline (Target: arXiv preprint, cs.CR, 4-6 pages)

---

## Abstract (~150 words)

Multi-turn prompt injection attacks exploit the stateless nature of LLM API proxies by distributing malicious intent across multiple conversation turns. While single-turn detection has been extensively studied, no published formula exists for converting per-turn pattern scores into a conversation-level risk score at the proxy layer — without invoking an LLM. We identify a fundamental flaw in the intuitive weighted-average approach: it converges to the per-turn score regardless of turn count, meaning a 20-turn attack scores identically to a 1-turn attack. Drawing on analogies from change-point detection (CUSUM), Bayesian belief updating, and security risk-based alerting (Splunk RBA), we propose *peak + accumulation scoring* — a formula combining peak single-turn risk, persistence ratio, and category diversity. Evaluated on 1,015 multi-turn conversations from WildChat and handcrafted attack sequences, our formula achieves 100% recall (up from 44%) at 1.2% false positive rate. We release both the scoring algorithm and eval harness as open source.

---

## 1. Introduction (~0.75 page)

**The problem:** LLM firewalls/guardrails sit between client and API. They see the full `messages[]` array on each request. Multi-turn attacks (instruction seeding, role confusion escalation, authority claim buildup) distribute signals across turns that are individually benign.

**Why proxy-level matters:**
- Many deployments cannot call a second LLM for classification (latency, cost, recursive vulnerability)
- Proxy sees every request — natural enforcement point
- Deterministic, auditable, no model drift

**The gap:** Extensive work on single-turn detection (DMPI-PMHFE, PromptScreen, NeMo heuristics). Multi-turn detection work exists (Defensive M2S, MindGuard) but requires LLM inference. No published algebraic formula for proxy-level scoring.

**Our contributions:**
1. We identify and prove the weighted-average ceiling failure
2. We propose peak + accumulation scoring with three configurable signals
3. We evaluate on public datasets with an open-source harness
4. We release everything: algorithm, patterns, eval data, results

---

## 2. Related Work (~1 page)

### 2.1 Single-Turn Proxy-Level Detection
- DMPI-PMHFE (2025): DeBERTa + heuristics. Heuristic channel includes synonym matching, many-shot detection, repeated token detection. 97.94% accuracy. Single-turn only.
- PromptScreen (2025): TF-IDF + Linear SVM. CPU-efficient, interpretable. Single-turn.
- NeMo Guardrails heuristics: Length/perplexity ratio, prefix/suffix perplexity. Targets GCG suffixes. Single-turn.
- Rebuff (ProtectAI): 4-layer defense including keyword permutation heuristics. Single-turn.

### 2.2 Multi-Turn Detection (LLM-Required)
- Defensive M2S (Kim et al., 2026): Compresses multi-turn to single-turn, classifies with guardrail LLM. 93.8% recall, 94.6% token reduction. Requires LLM.
- MindGuard (2026): Clinically grounded risk taxonomy, turn-level annotations. 4B-8B param classifiers.
- AprielGuard (2025): 8B param safeguard model. Multi-turn capable but requires model inference.
- LesWrong iterative safeguarding: `P(jailbreak) = f_E(response)` with two-threshold system. Closest to proxy-level architecture but uses LLM evaluator.

### 2.3 Multi-Turn Attack Characterization
- Yang et al. (2025): Multi-turn jailbreaks ≈ resampling single-turn. `S(k) = A - B * e^(-ck)`. Persistence is the primary signal.
- Crescendo (Russinovich et al., 2024): Gradual escalation. User prompts contain no injection phraseology — undetectable by patterns.
- MTJ-Bench (2025): Post-jailbreak persistence scenarios. Two persistence modes.
- Guarding the Guardrails (2025): 7 mechanism-oriented families including escalation and persistence.

### 2.4 Scoring Analogies from Adjacent Fields
- **CUSUM:** Cumulative sum for change-point detection. Accumulates deviations. Repeated signals combine.
- **Bayesian belief updating:** Evidence monotonically moves posterior toward hypothesis.
- **Splunk RBA:** Each anomaly *adds* to risk score 1-100. Designed for "death by a thousand cuts."
- **Key insight:** All use accumulation, not averaging. More evidence = higher score.

### 2.5 The Gap
No work combines: (a) per-turn pattern scoring, (b) conversation-level risk aggregation, (c) proxy-computable (no LLM), (d) a specific published formula.

---

## 3. The Weighted Average Failure (~0.75 page)

### 3.1 The Formula
Common intuition: weight later turns more heavily (recency bias).
```
w_i = 1 + i / (n - 1)          # weights from 1.0 to 2.0
cum  = Σ(score_i * w_i) / Σ(w_i)
```

### 3.2 The Ceiling Bug
**Theorem:** When all turns score `s`, `cum = s` regardless of `n`.

*Proof:* If `score_i = s` for all `i`:
```
cum = s * Σ(w_i) / Σ(w_i) = s
```

**Consequence:** A conversation where *every* turn matches role_confusion (weight 0.5) scores exactly 0.5 — identical to a single-turn match. With a threshold of 0.7, this attack is never detected regardless of how many turns persist.

### 3.3 Empirical Demonstration
[Table: 5 handcrafted multi-turn attacks, all with patterns on every turn, all scoring 0.3-0.5 under weighted average, all below 0.7 threshold]

The weighted average *punishes* persistence — the exact signal that characterizes multi-turn attacks.

---

## 4. Peak + Accumulation Scoring (~1 page)

### 4.1 Design Principles
1. **Peak sensitivity:** A single highly suspicious turn should contribute its full score
2. **Persistence reward:** More matching turns should increase the score
3. **Diversity reward:** Attacks spanning multiple categories (instruction seeding + role confusion + authority claims) are more suspicious than single-category repetition
4. **Additive bonuses:** Escalation gradient and resampling detection stack independently

### 4.2 The Formula

```
peak            = max(turn_scores)
match_ratio     = |{i : turn_score_i > 0}| / n
distinct_cats   = |unique categories matched|
diversity_bonus = max(0, distinct_cats - 1) * diversity_factor

final = clamp(
    peak
    + match_ratio * persistence_factor
    + diversity_bonus
    + escalation_bonus (if detected)
    + resampling_bonus (if detected),
    0.0, 1.0
)
```

**Default parameters:** persistence_factor = 0.35, diversity_factor = 0.15, escalation_bonus = 0.2, resampling_bonus = 0.7, threshold = 0.7.

### 4.3 Signal Descriptions

**Per-turn scoring:** Each user/tool message is scanned against 5 regex pattern categories:

| Category | Weight | Signal |
|----------|--------|--------|
| instruction_seeding | 0.4 | "remember this for later", "in my next message" |
| role_confusion | 0.5 | "you are now in developer mode", "switch to unrestricted" |
| deferred_authority | 0.3 | "admin said it was ok", "override authorized" |
| escalation_probing | 0.3 | "can you try to bypass", "what if you pretend" |
| repetition_resampling | 0.2 | (detected algorithmically via Jaccard trigram similarity) |

**Escalation gradient:** Detected when 3+ consecutive turns have strictly increasing scores. Indicates deliberate escalation from benign to hostile.

**Resampling detection:** Jaccard similarity on word-level trigrams between consecutive user messages. 3+ consecutive pairs with Jaccard > 0.5 triggers. Based on Yang et al.'s finding that multi-turn ≈ resampling.

### 4.4 Worked Examples

**Example A: Sparse single-category (Allow)**
- 4 turns: [0.0, 0.0, 0.0, 0.3]. Only one turn matches one category.
- peak=0.3, match_ratio=0.25, distinct=1, diversity=0
- final = 0.3 + 0.0875 + 0 = **0.3875** → Allow

**Example B: Dense multi-category (Block)**
- 4 turns: [0.0, 0.3, 0.0, 0.5]. Two turns match two categories.
- peak=0.5, match_ratio=0.5, distinct=2, diversity=0.15
- final = 0.5 + 0.175 + 0.15 = **0.825** → Block

**Example C: Persistent single-category (Block)**
- 4 turns: [0.5, 0.5, 0.5, 0.5]. All turns match role_confusion.
- peak=0.5, match_ratio=1.0, distinct=1, diversity=0
- final = 0.5 + 0.35 + 0 = **0.85** → Block
- (Under weighted average: 0.5 → Allow. The core failure case.)

---

## 5. Evaluation (~1 page)

### 5.1 Setup
- **System:** Parapet, an open-source Rust HTTP proxy firewall
- **Config:** L4-only mode (L3 pattern matching disabled to isolate L4 scoring)
- **Datasets:**
  - 9 handcrafted multi-turn attack sequences (instruction seeding, role confusion, deferred authority, escalation, combined)
  - 6 handcrafted benign multi-turn conversations
  - 1,000 real multi-turn conversations from WildChat (benign)
- **Threshold:** 0.7

### 5.2 Results

| Metric | Weighted Average | Peak + Accumulation |
|--------|-----------------|---------------------|
| True Positives | 4/9 | **9/9** |
| False Positives | 12/1000 | 12/1000 |
| True Negatives | 988/1000 | 988/1000 |
| False Negatives | 5/9 | **0/9** |
| Recall | 44.4% | **100.0%** |
| Precision | 25.0% | **42.9%** |
| F1 | 32.0% | **60.0%** |
| Accuracy | 98.3% | **98.8%** |

### 5.3 Analysis

**Recall improvement:** All 5 previously-missed attacks had patterns matching on every turn, but the weighted average converged to per-turn score (0.3-0.5). Peak + accumulation correctly rewards persistence.

**FP stability:** The 12 false positives are unchanged — they are real WildChat conversations that happen to contain phrases matching L4 patterns (e.g., roleplay conversations, legitimate discussions about AI modes). These are pattern specificity issues, not scoring issues.

**Precision:** Doubled from 25% to 42.9%. Still below 50% due to the small attack corpus (9 cases). Precision will improve as attack dataset grows.

### 5.4 Scope and Limitations
- **Content safety attacks are out of scope.** Crescendo-style attacks use deliberately innocuous language. Proxy-level regex cannot detect topic trajectory escalation. We evaluated SafeMTData (crescendo) and Anthropic hh-rlhf (social engineering) — both score 0%. This is expected and correct; these require LLM-based classification.
- **Small attack corpus.** 9 handcrafted sequences. Larger public multi-turn injection datasets do not yet exist.
- **Pattern brittleness.** Regex patterns can be evaded with rephrasing. This is a known limitation of all pattern-based approaches (see "The Attacker Moves Second"). Peak + accumulation scoring is orthogonal to pattern quality — it correctly aggregates whatever signals the patterns produce.

---

## 6. Discussion (~0.5 page)

### 6.1 Why No One Published This
Multi-turn detection work focuses on LLM-based classification. The proxy-level constraint (no LLM available) is considered a deployment detail, not a research problem. But proxy firewalls are widely deployed (Cloudflare AI Gateway, AWS Bedrock Guardrails, Azure AI Content Safety) and all face this scoring problem.

### 6.2 Parameter Sensitivity
The formula has 4 tunable parameters. We chose defaults based on:
- persistence_factor (0.35): Should push an all-matching conversation from peak to above threshold
- diversity_factor (0.15): One additional category is meaningful but shouldn't dominate
- escalation_bonus (0.2): Strong signal but not sufficient alone
- resampling_bonus (0.7): Near-threshold alone, since resampling = confirmed retry behavior

Formal sensitivity analysis is future work.

### 6.3 Integration with Layered Defense
Peak + accumulation is one layer in a defense-in-depth stack:
- L0: Unicode normalization, encoding hygiene
- L3: Single-turn pattern matching (inbound + outbound)
- L4: Multi-turn scoring (this paper)
- L5a: Output scanning (canary tokens, sensitive data redaction)

Each layer is independently configurable. L4 runs only when min_user_turns ≥ 2.

---

## 7. Conclusion (~0.25 page)

We presented the first published formula for proxy-level multi-turn LLM attack scoring. The weighted average approach, while intuitive, has a mathematical ceiling that makes it fundamentally unsuitable for persistence detection. Peak + accumulation scoring fixes this with three additive signals: peak risk, persistence ratio, and category diversity. On 1,015 test conversations, it achieves 100% recall at 1.2% false positive rate.

Code, patterns, eval harness, and datasets: [GitHub repo URL]

---

## References

1. Kim et al. "Defensive M2S" (2026). arXiv:2601.00454
2. Yang et al. "Multi-Turn Jailbreaks Are Simpler Than They Seem" (2025). arXiv:2508.07646
3. Russinovich et al. "Crescendo" (2024). arXiv:2404.01833 (USENIX Security 2025)
4. Shao et al. "MTJ-Bench" (2025). arXiv:2508.06755
5. Li et al. "MindGuard" (2026). arXiv:2602.00950
6. Shen et al. "DMPI-PMHFE" (2025). arXiv:2506.06384
7. Inan et al. "PromptScreen" (2025). arXiv:2512.19011
8. NeMo Guardrails Jailbreak Detection Heuristics. NVIDIA (2025).
9. ProtectAI. "Rebuff." GitHub (2025).
10. LesWrong. "Iterative Multi-Turn Safeguarding" (2025).
11. McKenzie et al. "STACK" (2025). arXiv:2506.24068
12. Xiao et al. "The Attacker Moves Second" (2025). arXiv:2510.09023
13. Ren et al. "Guarding the Guardrails" (2025). arXiv:2510.13893
14. Wei et al. "SoK: Evaluating Jailbreak Guardrails" (2025). arXiv:2506.10597
15. Zheng et al. "WildChat" (2024). Allen Institute for AI.
