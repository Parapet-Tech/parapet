"""Generate ledger entries from v5_r1 error analysis.

Reads generalist_errors.yaml, applies adjudication decisions,
computes full SHA-256 content hashes, and appends to the ledger.

Usage:
    python scripts/generate_ledger_v5r1.py --dry-run
    python scripts/generate_ledger_v5r1.py --apply
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Adjudication decisions from v5_r1 error review
# ---------------------------------------------------------------------------

# FN DROPs: content safety / benign mislabeled as PI / gibberish
# Keyed by truncated 16-char hash from error log
FN_DROPS: dict[str, tuple[str, str]] = {
    # --- Content safety only (EN), no PI signal ---
    "143cff62e3a781d8": ("out_of_scope", "Swahili content safety, not PI, mislabeled EN"),
    "19413b427d42925e": ("non_attack_in_attack_set", "Content safety (plagiarism request), not PI"),
    "5ae85fa5c3ff44cd": ("non_attack_in_attack_set", "Content safety (fake news request), not PI"),
    "ffb4664093eb7fef": ("out_of_scope", "Vietnamese content safety, not PI, mislabeled EN"),
    "be764bd1db7e394c": ("non_attack_in_attack_set", "Content safety (ransomware request), not PI"),
    "3542593361cace9d": ("non_attack_in_attack_set", "Content safety (misinformation rewrite), not PI"),
    "568609bfa191d77e": ("malformed_text", "Garbled text, no PI signal"),
    "64d1f9ea9832e748": ("non_attack_in_attack_set", "Content safety (weapon request), not PI"),
    "fe8c48e15e2768bf": ("non_attack_in_attack_set", "Content safety (extremist ideology), not PI"),
    "f4401d8dc7fdcb3e": ("non_attack_in_attack_set", "Content safety (violence request), not PI"),
    "e82c9b34f5647a6c": ("mislabel", "Benign health question, not PI"),
    "a5e6e6e4d37990cf": ("non_attack_in_attack_set", "Content safety (discrimination essay), not PI"),
    "11cc28e33526c560": ("non_attack_in_attack_set", "Content safety (conspiracy misinformation), not PI"),
    "ec929a3078dea757": ("mislabel", "Benign numpy question, not PI"),
    "1aa1dd5c33463c78": ("non_attack_in_attack_set", "Content safety (child endangerment), not PI"),
    "7c49e9d795d9f31c": ("non_attack_in_attack_set", "Content safety (poison tutorial), not PI"),
    "e7c1ba9cd87f71d8": ("non_attack_in_attack_set", "Content safety (sexual content), not PI"),
    "d235e1d664a41f54": ("non_attack_in_attack_set", "Content safety (vandalism), not PI"),
    "4c81b85c77a62920": ("non_attack_in_attack_set", "Content safety (chemical weapons), not PI"),
    "b103ba99aae026af": ("mislabel", "Benign appliance question, not PI"),
    "47bb7401ded9ff49": ("non_attack_in_attack_set", "Content safety (discrimination), not PI"),
    "37ae08b5dde2ea0c": ("non_attack_in_attack_set", "Content safety (gaslighting), not PI"),
    "485fe910af50803a": ("non_attack_in_attack_set", "Content safety (drug smuggling), not PI"),
    "d517186d609dc6f2": ("non_attack_in_attack_set", "Content safety (sexual content), not PI"),
    "afed76b942482e51": ("mislabel", "Benign political question, not PI"),
    "8abbd7e91a14fd33": ("mislabel", "Benign geography question, not PI"),
    "a7bcbd0e099ce4e4": ("non_attack_in_attack_set", "Copyright concern, not PI"),
    "5472301070ff5b82": ("mislabel", "Benign writing exercise, not PI"),
    "155138e98914480e": ("mislabel", "Benign baking question with profanity, not PI"),
    "7091ba911fdc667d": ("mislabel", "Benign web dev question, not PI"),
    "949abb89db518841": ("mislabel", "Benign gardening question, not PI"),
    "84bafb4e2cde3e35": ("mislabel", "Benign drone tech question, not PI"),
    "3a5372660a9aa9ac": ("mislabel", "Benign fishing technique question, not PI"),
    "d62f38dd683c7ec6": ("mislabel", "Benign privacy question, not PI"),
    "2eb3416b266c41ca": ("mislabel", "Generic benign question, not PI"),
    "5e28e25cf97512ce": ("mislabel", "Benign business question, not PI"),
    "b206c52b7c1566ee": ("mislabel", "Benign psychology question, not PI"),
    "e4c5207f9f6fcdae": ("mislabel", "Benign question in non-EN, not PI"),
    "2464c9f10c4e81e9": ("mislabel", "Benign salon question, not PI"),
    "d4987e7e5371ea8f": ("mislabel", "Benign freestyle rap request, not PI"),
    "b96b2bc8bb3b7f20": ("mislabel", "Benign AI research question, not PI"),
    "ccdc03648b6540d7": ("mislabel", "Benign fill-in-blank exercise, not PI"),
    "50adc81825b79d6f": ("mislabel", "Benign hotel recommendation request, not PI"),
    "76bd1ae1bf1649d9": ("mislabel", "Benign computational linguistics question, not PI"),
    "03e269e654187289": ("non_attack_in_attack_set", "Content safety (bank robbery), not PI"),
    "acf3cb5bf4025a94": ("non_attack_in_attack_set", "Content safety (spamming), not PI"),
    "8b548b19200dfddc": ("non_attack_in_attack_set", "Content safety (anti-immigrant essay), not PI"),
    "2959a1bdd1264669": ("non_attack_in_attack_set", "Content safety (misinformation generation), not PI"),
    "f2aac8effd664b69": ("non_attack_in_attack_set", "Content safety (fake reviews), not PI"),
    "a1cd2b7222b16da2": ("non_attack_in_attack_set", "Content safety (jury tampering), not PI"),
    "dc21d299addc84fc": ("mislabel", "Benign India population history, not PI"),
    # --- Non-EN content in en_attacks_merged, no PI ---
    "43560f87cc746f5f": ("out_of_scope", "Korean benign question, no PI, mislabeled EN"),
    "f0a1f190f82dc5a4": ("out_of_scope", "Chinese content safety, not PI, mislabeled EN"),
    "5223df30b00f9bfc": ("out_of_scope", "Korean benign question, no PI, mislabeled EN"),
    "88a6077150edb3a6": ("out_of_scope", "Chinese content safety, not PI, mislabeled EN"),
    "f544af794033f671": ("out_of_scope", "Chinese content safety, not PI, mislabeled EN"),
    "2e75a32c06b84b27": ("out_of_scope", "Chinese content safety, not PI, mislabeled EN"),
    "cacb8fc1af250408": ("out_of_scope", "Thai content safety, not PI, mislabeled EN"),
    "82baa365e51cc8f6": ("out_of_scope", "Korean benign question, no PI, mislabeled EN"),
    "8c57d80df06eb8b8": ("out_of_scope", "Chinese content safety, not PI, mislabeled EN"),
    "f602abb6cb357bf6": ("out_of_scope", "Korean content safety, not PI, mislabeled EN"),
    "d041629f3062e5c1": ("out_of_scope", "Korean benign question, no PI, mislabeled EN"),
    "d9b2154e43312b95": ("out_of_scope", "Korean benign question, no PI, mislabeled EN"),
    "ec3b7b229bac4be1": ("out_of_scope", "Korean benign question, no PI, mislabeled EN"),
    "4a0d366aeaa9ec74": ("out_of_scope", "Korean benign question, no PI, mislabeled EN"),
    "021c23ced55b9864": ("out_of_scope", "Vietnamese benign comment, not PI, mislabeled EN"),
    "585df07b951f3a2f": ("out_of_scope", "Italian content safety, not PI, mislabeled EN"),
    "2878a3d1a2464972": ("out_of_scope", "Javanese content safety, not PI, mislabeled EN"),
    "5802705cf0e670df": ("out_of_scope", "Portuguese content safety with refusal, not PI, mislabeled EN"),
    # --- Arabic non-PI ---
    "07d52a92736d1ca2": ("mislabel", "Arabic benign security question, not PI"),
    "38a1f2e1ca1f8e83": ("mislabel", "Arabic benign history question, not PI"),
    "9834e499aa69b07f": ("mislabel", "Arabic benign culture question, not PI"),
    "7c04b2799f4589d4": ("mislabel", "Arabic benign culture question, not PI"),
    "891704ffa0865808": ("non_attack_in_attack_set", "Arabic content safety (insults), not PI"),
    "501b8778228b2ad0": ("mislabel", "Arabic benign history question, not PI"),
    # --- Russian staged attacks with no PI signal ---
    "4c22cae1a112274f": ("mislabel", "Benign Russian ASCII question, not obfuscation PI"),
    "3e56d68c6bb3ffc7": ("non_attack_in_attack_set", "PII address request, not PI"),
    "bfab4ec335239cbb": ("mislabel", "Benign Russian gaming phrase, not PI"),
    "9c18d07a29763d73": ("mislabel", "Benign Russian phrase too short, not PI"),
    "1c1779806265c53f": ("mislabel", "Benign Russian fragment, not PI"),
    "786bd3ed1b33b62b": ("mislabel", "Benign Russian phrase, not PI"),
    "f4a37ffb6cdcd093": ("mislabel", "Benign Russian phrase, not PI"),
    "105b35f72b8dd363": ("non_attack_in_attack_set", "Academic dishonesty, not PI"),
    "c965406c04b81642": ("non_attack_in_attack_set", "Russian content safety (fake news), not PI"),
    # --- Gibberish / repeated chars ---
    "f1ec8d09cadb6ffc": ("malformed_text", "Repeated chars, not adversarial suffix"),
    "80ff80d083d19692": ("malformed_text", "Nonsensical fragment, not PI"),
    "56a29c6b47482931": ("malformed_text", "Transliterated Spanish fragment, not PI"),
    "87bc7defa5c94a83": ("mislabel", "Normal Russian phrase, not adversarial suffix"),
    # --- Japanese mislabeled as ZH ---
    "30063ec3e8dc6080": ("mislabel", "Japanese benign greeting mislabeled as ZH attack"),
    "8a7b57fa67e7df93": ("mislabel", "Japanese instruction text mislabeled as ZH attack"),
    # --- Russian roleplay_jailbreak with no PI ---
    "0da5316b4e8199d1": ("mislabel", "Benign Russian trivia, not roleplay_jailbreak"),
    "7b72d92f035a273c": ("mislabel", "Benign Russian health question, not PI"),
    "b5fc083f4cb652f5": ("non_attack_in_attack_set", "Content safety (stereotyping), not PI"),
    "27d4d2378e2f3fc4": ("mislabel", "Benign Russian nonsense sentences, not PI"),
    # --- Long benign text mislabeled as attack ---
    "7544a5c3b96a26cf": ("mislabel", "Oedipus summary + LoL text, no PI signal"),
    "a7b5c0ecfc418fea": ("mislabel", "Cat Wikipedia article, no PI signal"),
    "cc1f9a4b26f0dbbd": ("mislabel", "Calculus study example, no PI signal"),
    "c1c5d05b0960fe9e": ("out_of_scope", "Czech UI navigation text, no PI, mislabeled EN"),
    "79ce8b1b5928cf25": ("non_attack_in_attack_set", "Moral relativism essay, content safety not PI"),
    "e3763c1362721b88": ("non_attack_in_attack_set", "Russian explosives question, content safety not PI"),
    "a1fb79e50ad07ab1": ("non_attack_in_attack_set", "Russian ableist content, content safety not PI"),
    "4857d7afa922bd16": ("out_of_scope", "Arabic content safety, not PI"),
}

# FP RELABELs: benign items that are actually attacks
FP_RELABELS: dict[str, tuple[str, str, str]] = {
    # hash: (relabel_to, reroute_reason, rationale)
    "61f08f4f3b2fbd78": ("malicious", "roleplay_jailbreak", "Full DAN jailbreak in Chinese, explicit PI in benign set"),
    "2f8e80fb0e7d5e14": ("malicious", "constraint_bypass", "Full DAN jailbreak prompt, explicit PI in benign set"),
    "8d67b40765654b72": ("malicious", "constraint_bypass", "NsfwGPT jailbreak with 'forget everything', PI in benign set"),
    "58bdb50dac5b38c0": ("malicious", "roleplay_jailbreak", "4chan Kevin jailbreak 'no restrictions', PI in benign set"),
    "d837da0611582f24": ("malicious", "roleplay_jailbreak", "BreadGPT persona injection, PI in benign set"),
    "aed2508c654bdb20": ("malicious", "roleplay_jailbreak", "Instruction override for explicit scene, PI in benign set"),
    "b149f97b61a2b8ce": ("malicious", "roleplay_jailbreak", "System prompt override instructions in benign set"),
}

# FN ROUTEs: correct malicious label, wrong reason
FN_REROUTES: dict[str, tuple[str, str]] = {
    # hash: (new_reason, rationale)
    "300b786757ae8f33": ("obfuscation", "Hex-encoded PI, should be obfuscation not instruction_override"),
    "9a7d13267f72e575": ("obfuscation", "Base64-encoded PI, should be obfuscation not instruction_override"),
    "5978f2b7c45e1f09": ("obfuscation", "Emoji-encoded PI, should be obfuscation not instruction_override"),
    "369d99580fe8576b": ("obfuscation", "Homoglyph obfuscation meta-probe, should be obfuscation"),
}


def build_hash_index(errors: list[dict]) -> dict[str, dict]:
    """Map truncated 16-char hashes to full error entries."""
    return {e["content_hash"]: e for e in errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        args.dry_run = False

    base = Path(__file__).resolve().parent.parent
    errors_path = base.parent / "parapet-runner" / "runs" / "v5_r1" / "model" / "generalist_errors.yaml"
    ledger_path = base / "adjudication" / "ledger.yaml"

    # Load errors
    with open(errors_path, "r", encoding="utf-8") as f:
        errors = yaml.safe_load(f)
    idx = build_hash_index(errors)

    # Load existing ledger
    with open(ledger_path, "r", encoding="utf-8") as f:
        existing = yaml.safe_load(f) or []
    existing_hashes = {e["content_hash"] for e in existing}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_entries = []
    skipped = 0

    # --- Process DROPs ---
    for short_hash, (adj_reason, rationale) in FN_DROPS.items():
        if short_hash not in idx:
            print(f"  WARN: hash {short_hash} not found in errors")
            continue
        err = idx[short_hash]
        full_hash = content_hash(err["content"])
        if full_hash in existing_hashes:
            skipped += 1
            continue
        new_entries.append({
            "content_hash": full_hash,
            "source": err["source"],
            "label_at_time": err["label"],
            "reason_at_time": err.get("reason"),
            "action": "drop",
            "adjudication": adj_reason,
            "rationale": rationale,
            "first_seen_run": "v5_r1",
            "reviewer": "v5_r1_error_analysis",
            "updated_at": now,
        })

    # --- Process RELABELs ---
    for short_hash, (relabel_to, reroute_reason, rationale) in FP_RELABELS.items():
        if short_hash not in idx:
            print(f"  WARN: hash {short_hash} not found in errors")
            continue
        err = idx[short_hash]
        full_hash = content_hash(err["content"])
        if full_hash in existing_hashes:
            skipped += 1
            continue
        new_entries.append({
            "content_hash": full_hash,
            "source": err["source"],
            "label_at_time": err["label"],
            "reason_at_time": err.get("reason"),
            "action": "relabel_class",
            "adjudication": "benign_contamination",
            "relabel_to": relabel_to,
            "reroute_to": reroute_reason,
            "rationale": rationale,
            "first_seen_run": "v5_r1",
            "reviewer": "v5_r1_error_analysis",
            "updated_at": now,
        })

    # --- Process ROUTEs ---
    for short_hash, (new_reason, rationale) in FN_REROUTES.items():
        if short_hash not in idx:
            print(f"  WARN: hash {short_hash} not found in errors")
            continue
        err = idx[short_hash]
        full_hash = content_hash(err["content"])
        if full_hash in existing_hashes:
            skipped += 1
            continue
        new_entries.append({
            "content_hash": full_hash,
            "source": err["source"],
            "label_at_time": err["label"],
            "reason_at_time": err.get("reason"),
            "action": "reroute_reason",
            "adjudication": "routing_defect",
            "reroute_to": new_reason,
            "rationale": rationale,
            "first_seen_run": "v5_r1",
            "reviewer": "v5_r1_error_analysis",
            "updated_at": now,
        })

    print(f"\nLedger update summary:")
    print(f"  Existing entries: {len(existing)}")
    print(f"  New entries:      {len(new_entries)}")
    print(f"  Skipped (dupes):  {skipped}")
    print(f"  Breakdown:")
    drops = sum(1 for e in new_entries if e["action"] == "drop")
    relabels = sum(1 for e in new_entries if e["action"] == "relabel_class")
    reroutes = sum(1 for e in new_entries if e["action"] == "reroute_reason")
    print(f"    drop:           {drops}")
    print(f"    relabel_class:  {relabels}")
    print(f"    reroute_reason: {reroutes}")

    if args.dry_run:
        print(f"\n  DRY RUN — no changes written. Use --apply to write.")
        # Print first few entries as preview
        for entry in new_entries[:3]:
            print(f"\n  Preview: {entry['action']} | {entry['adjudication']} | {entry['rationale'][:60]}")
    else:
        combined = existing + new_entries
        with open(ledger_path, "w", encoding="utf-8") as f:
            yaml.dump(combined, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"\n  Written {len(combined)} entries to {ledger_path}")


if __name__ == "__main__":
    main()
