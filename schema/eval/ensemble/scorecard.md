# L1 Ensemble Scorecard

Date: 2026-02-24
Training script: `scripts/train_l1_specialist.py`
Vectorizer: `CountVectorizer(analyzer='char_wb', ngram_range=(3,5), binary=True)`
Model: `LinearSVC(C=0.1, class_weight='balanced')`

## Specialist Status

| # | Specialist | Status | CV F1 | Holdout F1 | FP | FN | Attacks | Benign | Features | Max/File |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | exfiltration | **FROZEN** | 0.964 | 0.970 | 60 | 70 | 11K | 25K | 2726 | 2000 |
| 2 | adversarial_suffix | **FROZEN** | 0.947 | 0.958 | 12 | 9 | 1.2K | 19K | 1157 | 2000 |
| 3 | indirect_injection | **FROZEN** | 0.952 | 0.951 | 96 | 94 | 10K | 23K | 2909 | 2000 |
| 4 | meta_probe | iterate | 0.902 | 0.905 | 119 | 179 | 8K | 25K | 3685 | 2000 |
| 5 | instruction_override | iterate | 0.899 | 0.903 | 962 | 1228 | 57K | 102K | 6182 | 10000 |
| 6 | constraint_bypass | iterate | 0.856 | 0.858 | 463 | 626 | 20K | 25K | 5844 | 2000 |
| 7 | roleplay_jailbreak | iterate | 0.797 | 0.809 | 344 | 410 | 10K | 25K | 5616 | 2000 |

## Frozen Specialists — Threshold Picks

### exfiltration (F1 0.970)
```
Threshold     TP     FP     FN     TN    Prec  Recall      F1
     -0.5   2150    141     44   4887   0.938   0.980   0.959
      0.0   2124     60     70   4968   0.973   0.968   0.970  <-- default
      0.5   2080     35    114   4993   0.983   0.948   0.965
```
Top signals: `' secr'`, `'cter '`, `'e! '`, `'s: '`, `' <|'`

### adversarial_suffix (F1 0.958)
```
Threshold     TP     FP     FN     TN    Prec  Recall      F1
     -0.5    245     44      3   3784   0.848   0.988   0.912
      0.0    239     12      9   3816   0.952   0.964   0.958  <-- default
      0.5    235      4     13   3824   0.983   0.948   0.965
```
Only 21 total errors. 9 FN are all jailbreakbench (semantic, not suffix-shaped).

### indirect_injection (F1 0.951)
```
Threshold     TP     FP     FN     TN    Prec  Recall      F1
     -0.5   1899    183     56   4445   0.912   0.971   0.941
      0.0   1861     96     94   4532   0.951   0.952   0.951  <-- default
      0.5   1821     42    134   4586   0.977   0.931   0.954
```
Top signals: `'ecr'`, `' | '`, `'send'`, `' send'`, `'.com'`

## Iterate — Known Issues

### instruction_override (F1 0.903)
- 962 FP / 1228 FN on 32K holdout — ran with --max-per-file 10000 (too large)
- FP includes Russian benign, code, instructional text
- FN includes CJK, Arabic, emoji-encoded, Unicode math chars
- Retrain with --max-per-file 2000 and more focused attack routing

### roleplay_jailbreak (F1 0.809)
- Weakest specialist. Attack data is too semantic for char n-grams.
- galtea dataset is noisy (general harmful questions, not jailbreaks)
- FN dominated by non-English MultiJail and semantic galtea
- Consider: drop galtea, focus on structured jailbreak templates only

### constraint_bypass (F1 0.858)
- Broad category, lots of overlap with roleplay_jailbreak
- 463 FP includes XSTest (safe prompts that look dangerous) — expected
- FN dominated by non-English, emoji-encoded, and semantic content
- Consider: split into "refusal suppression" + "harmful intent" sub-specialists

### meta_probe (F1 0.905)
- Decent but 179 FN. Many are hackaprompt Unicode/emoji entries.
- Squash pass should recover some of these.
- FP includes instructional text with "find", "extract", "convert"

## Dropped

| Specialist | Reason |
|---|---|
| obfuscation | Replaced by squash pass in l1.rs. 314 attacks, 22.8% recall — not viable as ML model. |

## Wiring Status

- [x] l1_weights_instruction_override.rs — compiled
- [x] l1_weights_roleplay_jailbreak.rs — compiled
- [x] l1_weights_meta_probe.rs — compiled
- [x] l1_weights_exfiltration.rs — compiled
- [ ] l1_weights_constraint_bypass.rs — needs wiring in compiled_specialists()
- [ ] l1_weights_adversarial_suffix.rs — needs wiring in compiled_specialists()
- [ ] l1_weights_indirect_injection.rs — needs wiring in compiled_specialists()
- [x] l1_weights_obfuscation.rs — TO BE REMOVED from compiled_specialists()
