# Eval Dataset Inventory

Canonical registry of all datasets used by `parapet-eval`.
Versioned alongside eval configs and YAML files in `schema/eval/`.

## Datasets

### Existing (fetched and wired)

| ID | Source | License | Rows (eval) | Split | Type | Text Col | Label Col | Script | Output |
|----|--------|---------|-------------|-------|------|----------|-----------|--------|--------|
| deepset | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Not specified | 546 (203a/343b) | train | attack+benign | `text` | `label` 0/1 | `fetch_deepset.py` | `opensource_deepset_{attacks,benign}.yaml` |
| giskard | [Giskard-AI/prompt-injections](https://github.com/Giskard-AI/prompt-injections) | Not specified | 35a | — | attack-only | `prompt` | all attack | `fetch_giskard.py` | `opensource_giskard_attacks.yaml` |
| gandalf | [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | Not specified | 1,000a | train | attack-only | `text` | all attack | `fetch_gandalf.py` | `opensource_gandalf_attacks.yaml` |
| mosscap | [Lakera/mosscap_prompt_injection](https://huggingface.co/datasets/Lakera/mosscap_prompt_injection) | Not specified | 1,743a (sampled from 223K) | train | attack-only | `prompt` | all attack | `fetch_mosscap.py` | `opensource_mosscap_attacks.yaml` |
| jbb | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT | 200 (100a/100b) | harmful+benign | attack+benign | `Goal` | split-based | `fetch_jailbreakbench.py` | `opensource_jbb_{attacks,benign}.yaml` |
| jailbreak-cls | [rubend18/ChatGPT-Jailbreak-Prompts](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) | — | varies | train | attack+benign | `text` | `label` 0/1 | `fetch_jailbreak_cls.py` | `opensource_jailbreak_cls_{attacks,benign}.yaml` |
| hackaprompt | [hackaprompt/hackaprompt-dataset](https://huggingface.co/datasets/Hackaprompt/hackaprompt-dataset) | — | varies | — | attack-only | `user_input` | all attack | `fetch_hackaprompt.py` | `opensource_hackaprompt_attacks.yaml` |
| chatgpt-jailbreak | [rubend18/ChatGPT-Jailbreak-Prompts](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) | — | varies | — | attack-only | `Prompt` | all attack | `fetch_chatgpt_jailbreak.py` | `opensource_chatgpt_jailbreak_attacks.yaml` |

### Benign-only datasets (fetched and wired)

| ID | Source | Rows (eval) | Script | Output |
|----|--------|-------------|--------|--------|
| hc3 | HC3 | varies | `fetch_hc3.py` | `opensource_hc3_benign.yaml` |
| prompts-chat | Prompts Chat | varies | `fetch_prompts_chat.py` | `opensource_prompts_chat_benign.yaml` |
| wildchat | WildChat | varies | `fetch_wildchat.py` | `opensource_wildchat_benign.yaml` |
| teven | Teven Prompted | varies | `fetch_teven_prompted.py` | `opensource_teven_benign.yaml` |
| dahoas | Dahoas HH | varies | `fetch_dahoas_hh.py` | `opensource_dahoas_benign.yaml` |
| chatgpt-prompts | ChatGPT Prompts | varies | `fetch_chatgpt_prompts.py` | `opensource_chatgpt_prompts_benign.yaml` |
| hf-instruction | HF Instruction | varies | `fetch_hf_instruction.py` | `opensource_hf_instruction_benign.yaml` |
| no-robots | No Robots | varies | `fetch_no_robots.py` | `opensource_no_robots_benign.yaml` |
| ultrachat | UltraChat | varies | `fetch_ultrachat.py` | `opensource_ultrachat_benign.yaml` |
| awesome-chatgpt | Awesome ChatGPT | varies | `fetch_awesome_chatgpt.py` | `opensource_awesome_chatgpt_benign.yaml` |

### V4-10 new datasets (scripts written, not yet fetched)

| ID | Source | License | Full Size | Eval Sample | Split | Type | Text Col | Label Col | Script | Output |
|----|--------|---------|-----------|-------------|-------|------|----------|-----------|--------|--------|
| bipia | [microsoft/BIPIA](https://github.com/microsoft/BIPIA) | MIT (code); CC-BY-SA 4.0 (data) | 86K+ (test, combinatorial) | ~600a + ~50b (sampled cross-product) | test | attack+benign | constructed (context + injected attack) | constructed | `fetch_bipia.py` | `opensource_bipia_{attacks,benign}.yaml` |
| jailbreakv | [JailbreakV-28K/JailBreakV-28k](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k) | MIT | 30,280 | 280a (mini split) | mini | attack-only | `jailbreak_query` | all attack | `fetch_jailbreakv.py` | `opensource_jailbreakv_attacks.yaml` |
| jbb-paraphrase | [DhruvTre/jailbreakbench-paraphrase-2025-08](https://huggingface.co/datasets/DhruvTre/jailbreakbench-paraphrase-2025-08) | MIT | 115 | 115 (56a/59b) | test | attack+benign | `prompt` | `label` 0/1 | `fetch_jbb_paraphrase.py` | `opensource_jbb_paraphrase_{attacks,benign}.yaml` |
| llmail | [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | MIT | 461,640 | 200a (first-N sample) | Phase1 | attack-only | `body` (+`subject`) | all attack | `fetch_llmail.py` | `opensource_llmail_attacks.yaml` |
| promptshield | [hendzh/PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) | Apache-2.0 | 43,425 | 500 (mixed) | test | attack+benign | `prompt` | `label` 0/1 | `fetch_promptshield.py` | `opensource_promptshield_{attacks,benign}.yaml` |
| geekyrakshit | [geekyrakshit/prompt-injection-dataset](https://huggingface.co/datasets/geekyrakshit/prompt-injection-dataset) | Not declared | 534,434 | 500 (mixed) | test | attack+benign | `prompt` | `label` 0/1 | `fetch_geekyrakshit.py` | `opensource_geekyrakshit_{attacks,benign}.yaml` |
| safeguard | [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | Not declared | 10,296 | 2,060 (full test) | test | attack+benign | `text` | `label` 0/1 | `fetch_safeguard.py` | `opensource_safeguard_{attacks,benign}.yaml` |

### Removed

| ID | Source | Reason | Tracked In |
|----|--------|--------|------------|
| imoxto | [imoxto/prompt_injection_cleaned_dataset-v2](https://huggingface.co/datasets/imoxto/prompt_injection_cleaned_dataset-v2) | Corrupted HackAPrompt duplicate — 535K rows of full prompt templates with unreliable labels. Duplicates `hackaprompt`. Was in L1 training recipe; cleanup tracked as V4-10B. | V4-10B |

### Not available

| ID | Source | Reason |
|----|--------|--------|
| PINT | [Lakera PINT Benchmark](https://github.com/lakeraai/pint-benchmark) | Dataset is private (code-only repo) |
| bipia-qa | BIPIA qa task | Requires external NewsQA license |
| bipia-abstract | BIPIA abstract task | Requires external XSum license |

## Overlap Notes

- **geekyrakshit** aggregates **deepset** + **xTRam1/safeguard** + jayavibhav. Expect row overlap when running all three in the same eval. For deduplication purposes, prefer running geekyrakshit OR (deepset + safeguard), not both.
- **jbb-paraphrase** is derived from **jbb** (JailbreakBench). Paraphrased variants of the same behaviors — complementary, not duplicative.
- **jailbreakv** includes multimodal jailbreaks (text + image). The fetch script extracts text-only `jailbreak_query` field; image-based attacks are not captured.
- **llmail** is all-attack with no benign counterpart. Pair with benign email/instruction datasets for balanced eval.
- **imoxto** was 100% HackAPrompt data — fully duplicated by **hackaprompt**. Removed.

## Benchmark Suites (V4-10 workstream 3)

| Suite | Status | Notes |
|-------|--------|-------|
| BIPIA (frozen core) | Scripts written | Integrated via `fetch_bipia.py` |
| JailbreakBench (frozen core) | Existing | Already wired via `fetch_jailbreakbench.py` |
| garak (NVIDIA) | Spec created | See `implement/v4/` for integration notes |
| AgentDojo | Spec created | See `implement/v4/` for integration notes |

## Running Fetch Scripts

All scripts write to `schema/eval/` and are run from the repo root:

```bash
cd parapet
python scripts/fetch_bipia.py
python scripts/fetch_jailbreakv.py
python scripts/fetch_jbb_paraphrase.py
python scripts/fetch_llmail.py
python scripts/fetch_promptshield.py
python scripts/fetch_geekyrakshit.py
python scripts/fetch_safeguard.py
```

After fetching, verify with the eval harness:

```bash
cargo run --features eval --bin parapet-eval -- \
  --config ../schema/eval/eval_config.yaml \
  --dataset ../schema/eval/ \
  --json
```
