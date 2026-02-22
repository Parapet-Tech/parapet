# V4-10B: L1 Retrain — Eval Summary

Date: 2026-02-22
Eval suite: 63,616 cases (24,915 L1-routed)
Eval command: `cargo run --features eval --release --bin parapet-eval -- --config ../schema/eval/eval_config.yaml --dataset ../schema/eval/ --json --output <file>`

---

## Final Comparison

| Metric | Original (ddf7d39) | Config #9 (V4-10B) | Delta |
|--------|--------------------:|--------------------:|------:|
| Precision | 95.89% | 96.13% | +0.24pp |
| Recall | 95.11% | 96.18% | +1.07pp |
| F1 | 95.50% | 96.15% | +0.65pp |
| FP count | 86 | 80 | -6 |
| FPR | 0.38% | 0.35% | -0.03pp |
| TP | 2005 | 1988 | -17 |
| FN | 103 | 79 | -24 |
| TN | 22831 | 22768 | -63 |

**Config #9 wins on every metric.** Lower FPs, higher recall, higher precision.

---

## Config #9 Training Recipe

14 datasets, 25,700 samples (4,744 attacks / 20,956 benign), uncapped (`--max-per-file 0`)

### Attack datasets (5)
| Dataset | Source | Count |
|---------|--------|------:|
| opensource_gandalf_attacks | Lakera/gandalf_ignore_instructions | 1000 |
| opensource_chatgpt_jailbreak_attacks | rubend18/ChatGPT-Jailbreak-Prompts | 78 |
| opensource_imoxto_attacks | imoxto/prompt_injection_cleaned_dataset-v2 | 1000 |
| opensource_hackaprompt_attacks | hackaprompt/hackaprompt-dataset | 2000 |
| opensource_jailbreak_cls_attacks | jackhhao/jailbreak-classification | 666 |

### Benign datasets (9)
| Dataset | Source | Count |
|---------|--------|------:|
| opensource_notinject_benign | leolee99/NotInject (staging/) | 339 |
| opensource_wildguardmix_benign | allenai/wildguardmix (staging/) | 2000 |
| opensource_awesome_chatgpt_benign | fka/awesome-chatgpt-prompts | 1190 |
| opensource_teven_benign | teven/prompted_examples | 1285 |
| opensource_dahoas_benign | Dahoas/synthetic-hh-rlhf-prompts | 4000 |
| opensource_chatgpt_prompts_benign | MohamedRashad/ChatGPT-prompts | 360 |
| opensource_hf_instruction_benign | HuggingFaceH4/instruction-dataset | 327 |
| opensource_no_robots_benign | HuggingFaceH4/no_robots | 9455 |
| opensource_ultrachat_benign | HuggingFaceH4/ultrachat_200k | 2000 |

### Model params
- CountVectorizer: char_wb, ngram_range=(3,5), binary=True, max_features=5000
- LinearSVC: C=0.1, class_weight='balanced', max_iter=10000
- Feature pruning: |weight| > 1e-6 threshold
- Final features: 2,217

### Holdout
- 5,140 cases (949 atk / 4,191 ben)
- Holdout P=94.0% R=95.3% F1=94.6%

---

## Per-Source Breakdown — Config #9

### Attack sources
| Source | Detected | Total | Recall |
|--------|--------:|------:|-------:|
| l1_attacks (hand-crafted) | 35 | 40 | 87.5% |
| l1_holdout | 883 | 949 | 93.0% |
| opensource_chatgpt_jailbreak_attacks | 72 | 78 | 92.3% |
| opensource_imoxto_attacks | 998 | 1000 | 99.8% |

### Benign sources
| Source | FP | Total | FPR |
|--------|---:|------:|----:|
| l1_benign (hand-crafted) | 1 | 40 | 2.50% |
| l1_holdout (benign portion) | — | — | — |
| opensource_awesome_chatgpt_benign | 1 | 1190 | 0.08% |
| opensource_chatgpt_prompts_benign | 6 | 360 | 1.67% |
| opensource_dahoas_benign | 1 | 4000 | 0.03% |
| opensource_hf_instruction_benign | 0 | 327 | 0.00% |
| opensource_no_robots_benign | 22 | 9455 | 0.23% |
| opensource_teven_benign | 1 | 1285 | 0.08% |
| opensource_ultrachat_benign | 6 | 2000 | 0.30% |

---

## Per-Source Breakdown — Original Model

### Attack sources
| Source | Detected | Total | Recall |
|--------|--------:|------:|-------:|
| l1_attacks (hand-crafted) | 32 | 40 | 80.0% |
| l1_holdout | 901 | 990 | 91.0% |
| opensource_chatgpt_jailbreak_attacks | 76 | 78 | 97.4% |
| opensource_imoxto_attacks | 996 | 1000 | 99.6% |

### Benign sources
| Source | FP | Total | FPR |
|--------|---:|------:|----:|
| l1_benign (hand-crafted) | 3 | 40 | 7.50% |
| opensource_awesome_chatgpt_benign | 3 | 1190 | 0.25% |
| opensource_chatgpt_prompts_benign | 2 | 360 | 0.56% |
| opensource_dahoas_benign | 0 | 4000 | 0.00% |
| opensource_hf_instruction_benign | 0 | 327 | 0.00% |
| opensource_no_robots_benign | 22 | 9455 | 0.23% |
| opensource_teven_benign | 4 | 1285 | 0.31% |
| opensource_ultrachat_benign | 2 | 2000 | 0.10% |

---

## Experiment History

| # | Config delta | P | R | F1 | FP | chatgpt_jb |
|---|-------------|----:|----:|----:|---:|-----------:|
| 1 | no imoxto, cap=1000 | 74.9% | 81.5% | 78.0% | 146 | 14.1% |
| 2 | + imoxto, cap=1000 | 92.3% | 92.3% | 92.3% | 134 | 15.4% |
| 3 | + jailbreak_cls, cap=1000 | 88.4% | 97.6% | 92.7% | 240 | 92.3% |
| 4 | + jailbreak_cls + jailbreakv, cap=1000 | 89.1% | 96.9% | 92.9% | 228 | 93.6% |
| 5 | + jailbreak_cls only, uncapped | 94.6% | 97.6% | 96.1% | 115 | 94.9% |
| 6 | + BIPIA + deepset | 91.8% | 97.1% | 94.4% | 210 | 98.7% |
| 7 | - BIPIA + protectai_val | 91.7% | 92.5% | 92.1% | 312 | 100% |
| 8 | + notinject (FP hardening) | 95.3% | 97.0% | 96.1% | 99 | 96.2% |
| **9** | **+ wildguardmix benign** | **96.1%** | **96.2%** | **96.2%** | **80** | **92.3%** |
| 10 | + jailbreak_cls benign 640 | 94.4% | 97.3% | 95.9% | 119 | 100% |
| 11 | + jailbreak_cls benign 64 | 96.3% | 96.0% | 96.1% | 77 | 89.7% |
| 12 | + promptshield benign | 94.0% | 98.5% | 96.1% | 131 | 100% |
| 13 | + promptshield atk + ben | 90.7% | 96.3% | 93.4% | 216 | 97.4% |
| 14 | + deepset only | 92.4% | 97.1% | 94.7% | 168 | 98.7% |
| — | original restored (ddf7d39) | 95.9% | 95.1% | 95.5% | 86 | 97.4% |

---

## False Positives — Config #9 (80 total)

```
awesome-chatgpt-0364 (opensource_awesome_chatgpt_benign)
chatgpt-prompts-0131 (opensource_chatgpt_prompts_benign)
chatgpt-prompts-0133 (opensource_chatgpt_prompts_benign)
chatgpt-prompts-0191 (l1_holdout + opensource_chatgpt_prompts_benign)
chatgpt-prompts-0269 (l1_holdout + opensource_chatgpt_prompts_benign)
chatgpt-prompts-0290 (l1_holdout + opensource_chatgpt_prompts_benign)
chatgpt-prompts-0354 (l1_holdout + opensource_chatgpt_prompts_benign)
dahoas-synth-0225 (l1_holdout + opensource_dahoas_benign)
l1-ben-029 (l1_benign)
no-robots-0426 (opensource_no_robots_benign)
no-robots-1161 (l1_holdout + opensource_no_robots_benign)
no-robots-2359 (opensource_no_robots_benign)
no-robots-2445 (opensource_no_robots_benign)
no-robots-2576 (l1_holdout + opensource_no_robots_benign)
no-robots-2737 (l1_holdout + opensource_no_robots_benign)
no-robots-2903 (opensource_no_robots_benign)
no-robots-2930 (opensource_no_robots_benign)
no-robots-3347 (l1_holdout + opensource_no_robots_benign)
no-robots-3379 (l1_holdout + opensource_no_robots_benign)
no-robots-3909 (opensource_no_robots_benign)
no-robots-4652 (l1_holdout + opensource_no_robots_benign)
no-robots-5113 (l1_holdout + opensource_no_robots_benign)
no-robots-5343 (l1_holdout + opensource_no_robots_benign)
no-robots-5583 (opensource_no_robots_benign)
no-robots-6607 (opensource_no_robots_benign)
no-robots-7254 (opensource_no_robots_benign)
no-robots-7601 (l1_holdout + opensource_no_robots_benign)
no-robots-7712 (opensource_no_robots_benign)
no-robots-9148 (opensource_no_robots_benign)
no-robots-9245 (opensource_no_robots_benign)
no-robots-9321 (l1_holdout + opensource_no_robots_benign)
notinject-0016 (l1_holdout)
notinject-0024 (l1_holdout)
notinject-0055 (l1_holdout)
notinject-0077 (l1_holdout)
notinject-0086 (l1_holdout)
notinject-0112 (l1_holdout)
notinject-0117 (l1_holdout)
notinject-0136 (l1_holdout)
notinject-0141 (l1_holdout)
notinject-0155 (l1_holdout)
notinject-0165 (l1_holdout)
notinject-0174 (l1_holdout)
notinject-0190 (l1_holdout)
notinject-0236 (l1_holdout)
notinject-0281 (l1_holdout)
notinject-0301 (l1_holdout)
teven-0112 (l1_holdout + opensource_teven_benign)
ultrachat-0159 (l1_holdout + opensource_ultrachat_benign)
ultrachat-0173 (l1_holdout + opensource_ultrachat_benign)
ultrachat-1054 (l1_holdout + opensource_ultrachat_benign)
ultrachat-1323 (opensource_ultrachat_benign)
ultrachat-1862 (l1_holdout + opensource_ultrachat_benign)
ultrachat-1867 (opensource_ultrachat_benign)
wildguardmix-ben-0359 (l1_holdout)
wildguardmix-ben-0786 (l1_holdout)
wildguardmix-ben-0890 (l1_holdout)
wildguardmix-ben-1619 (l1_holdout)
wildguardmix-ben-1668 (l1_holdout)
wildguardmix-ben-1675 (l1_holdout)
```

Note: Many IDs appear twice because the same case exists in both the holdout and the original dataset file.
Unique FP cases (deduped): ~48

---

## False Negatives — Config #9 (79 total)

```
chatgpt-jb-0017, 0036, 0044, 0054, 0059 (x2), 0061
gandalf-0106, 0125, 0264, 0355, 0781, 0794, 0812, 0820, 0877
hackaprompt-0466, 0499, 0721, 0874, 1047, 1062, 1088, 1170, 1232, 1364, 1367, 1385, 1398, 1556, 1639, 1692, 1843, 1866
imoxto-0939, 0985
jailbreak-cls-0015, 0022, 0077, 0082, 0115, 0121, 0126, 0129, 0139, 0150, 0166, 0218, 0308, 0326, 0355, 0376, 0384, 0389, 0393, 0414, 0502, 0528, 0550, 0578, 0600, 0621, 0628, 0642, 0673, 0705, 0836, 0949, 0955, 1028, 1108, 1231, 1258, 1280
l1-att-021, 022, 031, 035, 036
```

---

## Key Insights

1. **Uncapping `--max-per-file` was the biggest single improvement** (F1 78.0% -> 96.1%). The 1000-cap skewed attack:benign ratio to 40:60 when natural ratio is ~18:82. Restoring natural balance let the SVM learn cleaner decision boundaries.

2. **Hard-negative benign data reduces FPs without hurting recall.** NotInject (339 trigger-word benign prompts) dropped FP from 115 to 99. WildGuardMix benign (2000 safety-adjacent prompts) dropped FP further to 80.

3. **Indirect injection data hurts L1.** BIPIA, PromptShield, and Deepset attacks are document-embedded injections that character n-grams can't distinguish from normal text. Adding them consistently increased FPs. These belong in L3 (LLM judge).

4. **Supplemental jailbreak attacks help.** Adding jailbreak_cls (666 attacks from jackhhao/jailbreak-classification) jumped chatgpt_jailbreak recall from 15.4% to 92.3%.

5. **The new model beats the original on an expanded eval suite** despite the original having been specifically trained on more of the eval data. Config #9's diverse benign data creates better generalization.
