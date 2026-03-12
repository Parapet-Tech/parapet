param(
    [string[]]$Only,
    [switch]$List
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$python = "python"
$exporter = "parapet-data/scripts/export_attack_candidates.py"

$jobs = @(
    [ordered]@{
        Name = "en_indirect_llmail_seed"
        Cell = "EN indirect_injection"
        Args = @(
            "--input", "schema/eval/staging/en_llmail-inject-challenge_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=indirect_injection",
            "--dataset-name", "llmail_indirect_seed",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "400",
            "--output", "work/gap_mining/v5_5k/en_indirect_injection/llmail_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/en_indirect_injection/llmail_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "en_indirect_wildjailbreak_topup"
        Cell = "EN indirect_injection"
        Args = @(
            "--input", "schema/eval/staging/en_wildjailbreak_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=indirect_injection",
            "--dataset-name", "wildjailbreak_indirect_topup",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "120",
            "--output", "work/gap_mining/v5_5k/en_indirect_injection/wildjailbreak_topup.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/en_indirect_injection/wildjailbreak_topup_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_roleplay_seed_pool"
        Cell = "ZH roleplay_jailbreak"
        Args = @(
            "--input",
            "schema/eval/staging/en_wildjailbreak_attacks_staged.yaml",
            "schema/eval/staging/en_jailbreak_llms_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=roleplay_jailbreak",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "240",
            "--output", "work/gap_mining/v5_5k/zh_roleplay_jailbreak/seed_pool.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_roleplay_jailbreak/seed_pool_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_meta_probe_hackaprompt_seed"
        Cell = "ZH meta_probe"
        Args = @(
            "--input", "schema/eval/staging/en_hackaprompt-dataset_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=meta_probe",
            "--dataset-name", "hackaprompt_meta_probe_seed",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "120",
            "--output", "work/gap_mining/v5_5k/zh_meta_probe/hackaprompt_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_meta_probe/hackaprompt_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_meta_probe_tensor_seed"
        Cell = "ZH meta_probe"
        Args = @(
            "--input", "schema/eval/staging/en_tensor-trust-data_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=meta_probe",
            "--dataset-name", "tensor_meta_probe_seed",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "120",
            "--output", "work/gap_mining/v5_5k/zh_meta_probe/tensor_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_meta_probe/tensor_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_meta_probe_spml_seed"
        Cell = "ZH meta_probe"
        Args = @(
            "--input", "schema/eval/staging/en_SPML_Chatbot_Prompt_Injection_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=meta_probe",
            "--dataset-name", "spml_meta_probe_seed",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "40",
            "--output", "work/gap_mining/v5_5k/zh_meta_probe/spml_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_meta_probe/spml_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_meta_probe_prompt_subtle_seed"
        Cell = "ZH meta_probe"
        Args = @(
            "--input", "schema/eval/staging/en_prompt-injections-subtle_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=meta_probe",
            "--dataset-name", "prompt_subtle_meta_probe_seed",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "40",
            "--output", "work/gap_mining/v5_5k/zh_meta_probe/prompt_subtle_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_meta_probe/prompt_subtle_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_meta_probe_tensor_raw"
        Cell = "ZH meta_probe"
        Args = @(
            "--input", "../TheWall/tensor-trust-data/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl",
            "--text-field", "attack",
            "--id-field", "sample_id",
            "--dataset-name", "tensor_trust_meta_probe_raw",
            "--language-hint", "EN",
            "--copy-field", "pre_prompt",
            "--copy-field", "post_prompt",
            "--shuffle",
            "--seed", "42",
            "--limit", "120",
            "--output", "work/gap_mining/v5_5k/zh_meta_probe/tensor_trust_raw.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_meta_probe/tensor_trust_raw_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_exfiltration_llmail_seed"
        Cell = "ZH exfiltration"
        Args = @(
            "--input", "schema/eval/staging/en_llmail-inject-challenge_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=exfiltration",
            "--dataset-name", "llmail_exfiltration_seed",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "240",
            "--output", "work/gap_mining/v5_5k/zh_exfiltration/llmail_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_exfiltration/llmail_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_exfiltration_wildjailbreak_topup"
        Cell = "ZH exfiltration"
        Args = @(
            "--input", "schema/eval/staging/en_wildjailbreak_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=exfiltration",
            "--dataset-name", "wildjailbreak_exfiltration_topup",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "80",
            "--output", "work/gap_mining/v5_5k/zh_exfiltration/wildjailbreak_topup.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_exfiltration/wildjailbreak_topup_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_adversarial_suffix_amplegcg_seed"
        Cell = "ZH adversarial_suffix"
        Args = @(
            "--input", "schema/eval/malicious/thewall_amplegcg_pos.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--dataset-name", "amplegcg_suffix_seed",
            "--language-hint", "EN",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--limit", "160",
            "--output", "work/gap_mining/v5_5k/zh_adversarial_suffix/amplegcg_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_adversarial_suffix/amplegcg_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_adversarial_suffix_llm_attacks_seed"
        Cell = "ZH adversarial_suffix"
        Args = @(
            "--input", "schema/eval/malicious/thewall_llm-attacks_pos.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--dataset-name", "llm_attacks_suffix_seed",
            "--language-hint", "EN",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--limit", "160",
            "--output", "work/gap_mining/v5_5k/zh_adversarial_suffix/llm_attacks_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_adversarial_suffix/llm_attacks_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_adversarial_suffix_llmail_topup"
        Cell = "ZH adversarial_suffix"
        Args = @(
            "--input", "schema/eval/staging/en_llmail-inject-challenge_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=adversarial_suffix",
            "--dataset-name", "llmail_suffix_topup",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "120",
            "--output", "work/gap_mining/v5_5k/zh_adversarial_suffix/llmail_topup.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_adversarial_suffix/llmail_topup_manifest.json"
        )
    },
    [ordered]@{
        Name = "zh_constraint_bypass_seed_pool"
        Cell = "ZH constraint_bypass"
        Args = @(
            "--input",
            "schema/eval/staging/en_wildjailbreak_attacks_staged.yaml",
            "schema/eval/staging/en_jailbreak_llms_attacks_staged.yaml",
            "schema/eval/staging/en_prompt-injections_attacks_staged.yaml",
            "schema/eval/staging/en_prompt-injections-subtle_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=constraint_bypass",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "240",
            "--output", "work/gap_mining/v5_5k/zh_constraint_bypass/seed_pool.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/zh_constraint_bypass/seed_pool_manifest.json"
        )
    },
    [ordered]@{
        Name = "ar_indirect_llmail_seed"
        Cell = "AR indirect_injection"
        Args = @(
            "--input", "schema/eval/staging/en_llmail-inject-challenge_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=indirect_injection",
            "--dataset-name", "llmail_indirect_seed",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "80",
            "--output", "work/gap_mining/v5_5k/ar_indirect_injection/llmail_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ar_indirect_injection/llmail_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "ar_indirect_wildjailbreak_topup"
        Cell = "AR indirect_injection"
        Args = @(
            "--input", "schema/eval/staging/en_wildjailbreak_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=indirect_injection",
            "--dataset-name", "wildjailbreak_indirect_topup",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "40",
            "--output", "work/gap_mining/v5_5k/ar_indirect_injection/wildjailbreak_topup.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ar_indirect_injection/wildjailbreak_topup_manifest.json"
        )
    },
    [ordered]@{
        Name = "ar_adversarial_suffix_amplegcg_seed"
        Cell = "AR adversarial_suffix"
        Args = @(
            "--input", "schema/eval/malicious/thewall_amplegcg_pos.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--dataset-name", "amplegcg_suffix_seed",
            "--language-hint", "EN",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--limit", "80",
            "--output", "work/gap_mining/v5_5k/ar_adversarial_suffix/amplegcg_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ar_adversarial_suffix/amplegcg_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "ar_adversarial_suffix_llm_attacks_seed"
        Cell = "AR adversarial_suffix"
        Args = @(
            "--input", "schema/eval/malicious/thewall_llm-attacks_pos.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--dataset-name", "llm_attacks_suffix_seed",
            "--language-hint", "EN",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--limit", "80",
            "--output", "work/gap_mining/v5_5k/ar_adversarial_suffix/llm_attacks_seed.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ar_adversarial_suffix/llm_attacks_seed_manifest.json"
        )
    },
    [ordered]@{
        Name = "ar_adversarial_suffix_llmail_topup"
        Cell = "AR adversarial_suffix"
        Args = @(
            "--input", "schema/eval/staging/en_llmail-inject-challenge_attacks_staged.yaml",
            "--text-field", "content",
            "--id-field", "content_hash",
            "--where", "reason=adversarial_suffix",
            "--dataset-name", "llmail_suffix_topup",
            "--language-hint", "EN",
            "--copy-field", "reason",
            "--copy-field", "source",
            "--copy-field", "language",
            "--shuffle",
            "--seed", "42",
            "--limit", "80",
            "--output", "work/gap_mining/v5_5k/ar_adversarial_suffix/llmail_topup.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ar_adversarial_suffix/llmail_topup_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_wambosec"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=wambosec",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--limit", "96",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/wambosec_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/wambosec_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_mosscap"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=mosscap",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/mosscap_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/mosscap_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_nueralchemy"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=nueralchemy",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/nueralchemy_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/nueralchemy_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_llmail"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=llmail",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/llmail_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/llmail_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_ctf"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=ctf",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/ctf_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/ctf_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_tensor"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=tensor",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/tensor_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/tensor_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_browsesafe"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=browsesafe",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--limit", "160",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/browsesafe_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/browsesafe_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_atlas"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=atlas",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/atlas_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/atlas_tail_manifest.json"
        )
    },
    [ordered]@{
        Name = "ru_tail_agentic_rag"
        Cell = "RU tail review"
        Args = @(
            "--input", "schema/eval/training/multilingual/attacks_121624_ru.yaml",
            "--text-field", "content",
            "--id-field", "id",
            "--contains", "description=agentic-rag-redteam-bench",
            "--language-hint", "RU",
            "--copy-field", "description",
            "--shuffle",
            "--seed", "42",
            "--output", "work/gap_mining/v5_5k/ru_tail_review/agentic_rag_tail.jsonl",
            "--manifest-output", "work/gap_mining/v5_5k/ru_tail_review/agentic_rag_tail_manifest.json"
        )
    }
)

if ($List) {
    foreach ($job in $jobs) {
        Write-Output ("{0}`t{1}" -f $job.Name, $job.Cell)
    }
    exit 0
}

$selectedJobs = $jobs
if ($Only -and $Only.Count -gt 0) {
    $selectedJobs = @()
    foreach ($name in $Only) {
        $job = $jobs | Where-Object { $_.Name -eq $name }
        if (-not $job) {
            throw "Unknown job name: $name"
        }
        $selectedJobs += $job
    }
}

foreach ($job in $selectedJobs) {
    Write-Host ("==> {0} [{1}]" -f $job.Name, $job.Cell)
    & $python $exporter @($job.Args)
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
