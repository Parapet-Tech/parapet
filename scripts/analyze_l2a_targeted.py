#!/usr/bin/env python3
"""Analyze full-stack + L2a eval results against v4_10b baseline FNs.

Reads JSON results from schema/eval/results/l2a_fullstack/ and checks
whether adding L2a to the stack recovers any of the 79 baseline FNs.

Uses raw status codes (403 = blocked by any layer) instead of per-layer
attribution, which is the right metric for "does adding L2a help?"
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "schema", "eval", "results", "l2a_fullstack"
)

# v4_10b baseline FNs — 79 cases the full stack missed (from v4_10b_summary.md)
BASELINE_FNS = {
    "l1_attacks": [
        "l1-att-021", "l1-att-022", "l1-att-031", "l1-att-035", "l1-att-036",
    ],
    "opensource_chatgpt_jailbreak_attacks": [
        "chatgpt-jb-0017", "chatgpt-jb-0036", "chatgpt-jb-0044",
        "chatgpt-jb-0054", "chatgpt-jb-0059", "chatgpt-jb-0061",
    ],
    "opensource_gandalf_attacks": [
        "gandalf-0106", "gandalf-0125", "gandalf-0264", "gandalf-0355",
        "gandalf-0781", "gandalf-0794", "gandalf-0812", "gandalf-0820",
        "gandalf-0877",
    ],
    "opensource_hackaprompt_attacks": [
        "hackaprompt-0466", "hackaprompt-0499", "hackaprompt-0721",
        "hackaprompt-0874", "hackaprompt-1047", "hackaprompt-1062",
        "hackaprompt-1088", "hackaprompt-1170", "hackaprompt-1232",
        "hackaprompt-1364", "hackaprompt-1367", "hackaprompt-1385",
        "hackaprompt-1398", "hackaprompt-1556", "hackaprompt-1639",
        "hackaprompt-1692", "hackaprompt-1843", "hackaprompt-1866",
    ],
    "opensource_jailbreak_cls_attacks": [
        "jailbreak-cls-0015", "jailbreak-cls-0022", "jailbreak-cls-0077",
        "jailbreak-cls-0082", "jailbreak-cls-0115", "jailbreak-cls-0121",
        "jailbreak-cls-0126", "jailbreak-cls-0129", "jailbreak-cls-0139",
        "jailbreak-cls-0150", "jailbreak-cls-0166", "jailbreak-cls-0218",
        "jailbreak-cls-0308", "jailbreak-cls-0326", "jailbreak-cls-0355",
        "jailbreak-cls-0376", "jailbreak-cls-0384", "jailbreak-cls-0389",
        "jailbreak-cls-0393", "jailbreak-cls-0414", "jailbreak-cls-0502",
        "jailbreak-cls-0528", "jailbreak-cls-0550", "jailbreak-cls-0578",
        "jailbreak-cls-0600", "jailbreak-cls-0621", "jailbreak-cls-0628",
        "jailbreak-cls-0642", "jailbreak-cls-0673", "jailbreak-cls-0705",
        "jailbreak-cls-0836", "jailbreak-cls-0949", "jailbreak-cls-0955",
        "jailbreak-cls-1028", "jailbreak-cls-1108", "jailbreak-cls-1231",
        "jailbreak-cls-1258", "jailbreak-cls-1280",
    ],
}


def parse_detail(detail):
    """Extract status and blocked_by from detail string."""
    status = None
    blocked_by = None
    for part in detail.split():
        if part.startswith("status="):
            status = int(part.split("=", 1)[1])
        elif part.startswith("blocked_by="):
            blocked_by = part.split("=", 1)[1]
    return status, blocked_by


def main():
    total_baseline_fn = 0
    total_recovered = 0
    all_recovered = []
    all_still_missed = []

    for source in sorted(BASELINE_FNS):
        fn_ids = set(BASELINE_FNS[source])
        json_path = os.path.join(RESULTS_DIR, f"{source}.json")
        if not os.path.exists(json_path):
            print(f"{source}: NO RESULTS FILE")
            continue

        with open(json_path) as f:
            data = json.load(f)

        # Index results by case_id
        by_id = {r["case_id"]: r for r in data["results"]}

        recovered = []
        still_missed = []
        for cid in sorted(fn_ids):
            if cid not in by_id:
                still_missed.append((cid, "not in results"))
                continue
            r = by_id[cid]
            status, blocked_by = parse_detail(r["detail"])
            if status == 403:
                recovered.append((cid, blocked_by))
            else:
                still_missed.append((cid, blocked_by or "none"))

        total_baseline_fn += len(fn_ids)
        total_recovered += len(recovered)
        all_recovered.extend(recovered)
        all_still_missed.extend(still_missed)

        print(f"{source}")
        print(f"  baseline FNs: {len(fn_ids)}, recovered: {len(recovered)}, still missed: {len(still_missed)}")
        for cid, bb in recovered:
            print(f"    + {cid}  (blocked_by={bb})")
        for cid, bb in still_missed:
            print(f"    - {cid}  ({bb})")
        print()

    # Benign FP check
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if "benign" not in fname or not fname.endswith(".json"):
            continue
        source = fname.replace(".json", "")
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
        total = len(data["results"])
        fps = sum(1 for r in data["results"] if parse_detail(r["detail"])[0] == 403)
        print(f"{source}:  FP={fps}/{total} ({fps/total*100:.2f}%)")

    print()
    print("=" * 60)
    print(f"Baseline FNs:     {total_baseline_fn}")
    print(f"Recovered (+L2a): {total_recovered}")
    print(f"Still missed:     {total_baseline_fn - total_recovered}")
    print()

    if all_recovered:
        by_layer = {}
        for cid, bb in all_recovered:
            by_layer.setdefault(bb, []).append(cid)
        print("Recoveries by layer:")
        for layer, ids in sorted(by_layer.items()):
            print(f"  {layer}: {len(ids)}  ({', '.join(ids)})")


if __name__ == "__main__":
    main()
