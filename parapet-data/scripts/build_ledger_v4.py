"""Build initial adjudication ledger from v4 pilot error review."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ERRORS_PATH = Path(__file__).resolve().parents[1] / ".." / "parapet-runner" / "runs" / "mirror_v4_pilot_r5" / "model" / "generalist_errors.yaml"
LEDGER_PATH = Path(__file__).resolve().parents[1] / "adjudication" / "ledger.yaml"

# truncated_hash -> (action, adjudication, rationale)
ADJUDICATIONS: dict[str, tuple[str, str, str]] = {
    # -- mislabel: labeled malicious but content is clearly benign --
    "492357757396638a": ("drop", "non_attack_in_attack_set", "Korean text asking about schedule, not EN, not attack"),
    "26dc6547cc7bc153": ("drop", "mislabel", "Three-word Russian fragment, not an attack"),
    "366225637e7aba70": ("drop", "mislabel", "Benign question about cookies, not PI"),
    "22e10bf6fa3a468d": ("drop", "mislabel", "Benign idiom question about being big cheese"),
    "ef93fe51f311fe44": ("drop", "mislabel", "Normal Russian editing instruction"),
    "4ceb1fcd308805d3": ("drop", "mislabel", "Russian idiom for tell everything, benign colloquial"),
    "2a6524ec56851f4a": ("drop", "mislabel", "One-word question about game location"),
    "17771bc2ded4a6c6": ("drop", "mislabel", "Benign translation request"),
    "33ddbdb36a7a8979": ("drop", "mislabel", "Benign Romanian translation request"),
    "259d6ba73bf3cdf9": ("drop", "mislabel", "Benign creative prompt about confused actor"),
    "e2cbdaa45c54fe3e": ("drop", "mislabel", "Normal journalism question about Guardian and Daily Express"),
    "a6eed0b539b33367": ("drop", "mislabel", "Just counting one-two-three-four in Russian"),
    "52ab08dd987978d2": ("drop", "mislabel", "Benign conversational question"),
    "86cee9b5c1382acc": ("drop", "mislabel", "Benign language usage question"),
    "ea54570f96057a43": ("drop", "mislabel", "Benign social question about relationships"),
    "e2cd1e41da9250d1": ("drop", "mislabel", "Benign context-free question about levels"),
    "f688ebaa2b7e864e": ("drop", "mislabel", "Benign summarization request"),
    "eaac4079a232fb74": ("drop", "mislabel", "Benign question about fuel-efficient vehicles"),
    "0305c048b350028a": ("drop", "mislabel", "Benign safety practices question"),
    "75dcfb0be57125c4": ("drop", "mislabel", "Benign stress management discussion"),
    "e53dea9c396156f0": ("drop", "mislabel", "Benign persona request for tech opinion piece on Deno"),
    "de2e5e73e2cd7153": ("drop", "mislabel", "Calculus practice exercises, benign educational"),
    "9c5c9de1e42dc926": ("drop", "mislabel", "Python coding exercises, benign"),
    "752de68067cc981e": ("drop", "mislabel", "Question about Shape of You impact on music"),
    "a6236752a2b99359": ("drop", "mislabel", "Harry Potter character arc question"),
    "ad25dbd6499b0bd6": ("drop", "mislabel", "TCP three-way handshake notes, benign networking"),
    "026551f75066fbef": ("drop", "mislabel", "Nuclear fission explanation for test, benign"),
    "d23908ec68a24dfe": ("drop", "mislabel", "Social question about extramarital affairs"),
    "fe3ecd926adced4f": ("drop", "mislabel", "ML for bird species recognition, benign"),
    "220be9e8cb961631": ("drop", "mislabel", "Add details to story and summarize article, benign"),
    "23e14b66c28a11df": ("drop", "mislabel", "Russian text about making craft box for birthday"),
    "c017ec6e51e39d34": ("drop", "mislabel", "Russian question about PayPal lending practices"),
    "0177b72b325e4afb": ("drop", "mislabel", "Python code printing benign string"),
    "df5b5504b2827804": ("drop", "mislabel", "Image description of medical situation, not PI"),
    # -- extraction_defect --
    "2f88fa6f05c71644": ("drop", "extraction_defect", "Context-free fragment, parsing artifact"),
    "1169ad6d396e25a5": ("drop", "extraction_defect", "Garbled nonsensical Spanish-Russian fragment"),
    # -- out_of_scope: content safety, not prompt injection --
    "23fa26014423ee32": ("drop", "out_of_scope", "Methamphetamine production request, content safety not PI"),
    "84ea9b42ac40840e": ("drop", "out_of_scope", "Insulting paragraph request, content safety"),
    "224bbcf7cb66d945": ("drop", "out_of_scope", "Edit Wikipedia for racial superiority, content safety"),
    "78ee1bc828d0ab00": ("drop", "out_of_scope", "Promote counterfeit prescription drugs, content safety"),
    "b392726dbe1a0a71": ("drop", "out_of_scope", "Moon landing conspiracy, misinformation not PI"),
    "e5f73f35b98567c7": ("drop", "out_of_scope", "Mocking pregnant worker, content safety"),
    "3b41c2f98a3ef2fd": ("drop", "out_of_scope", "Mocking elderly person, ageism content safety"),
    "344b3ee780530073": ("drop", "out_of_scope", "Develop a virus, harmful content not PI"),
    "0082e1c5f3ed7019": ("drop", "out_of_scope", "Request for search history, privacy not PI"),
    "b227769aaf3989bc": ("drop", "out_of_scope", "Bullying message about stuttering, content safety"),
    "a2610be0688ef5ac": ("drop", "out_of_scope", "Animal cruelty question, content safety"),
    "af7eae6648110a70": ("drop", "out_of_scope", "Religious stereotypes dialogue, content safety"),
    "0a3d6e65efeae916": ("drop", "out_of_scope", "Down syndrome jokes, content safety"),
    "7583aaa79bbe4996": ("drop", "out_of_scope", "Non-traditional epilepsy treatment, medical safety"),
    "e6f7c02502c1da9c": ("drop", "out_of_scope", "Explicit sexual content request, NSFW not PI"),
    "1400dca9335637fe": ("drop", "out_of_scope", "Locate celebrity address, doxxing not PI"),
    "d4a3297380737cbd": ("drop", "out_of_scope", "Credit card digits of public figure, privacy not PI"),
    "87ff4ddffba22d3e": ("drop", "out_of_scope", "Vietnamese violence question, wrong language and content safety"),
    "62fa9938a5093aff": ("drop", "out_of_scope", "Identity theft request in Russian, content safety"),
    "b90624bdc66487c5": ("drop", "out_of_scope", "Graphic workplace violence, content safety"),
    "4023903c212a0f4f": ("drop", "out_of_scope", "Electronic payments misuse in gambling, content safety"),
    "85b3934dbaff1587": ("drop", "out_of_scope", "Harsh criticism of homosexuality, hate speech"),
    "974e462008ec5500": ("drop", "out_of_scope", "Trick airport drug-sniffing dog, illegal not PI"),
    "2f3ff3f616becb6a": ("drop", "out_of_scope", "Burn down orphanage question, violence not PI"),
    "4dd3e520961e07dc": ("drop", "out_of_scope", "Share health issue on platforms, privacy not PI"),
    "1f7db6cc8a7ea7bc": ("drop", "out_of_scope", "Sneak drugs into prison, illegal not PI"),
    "cc588f4d8b83f699": ("drop", "out_of_scope", "Tamper with voting machines, illegal not PI"),
    # -- benign_contamination: attack content in benign set --
    "28a613067f87424c": ("drop", "benign_contamination", "Classic jailbreak prompt in benign set"),
    "eab5df81da557409": ("drop", "benign_contamination", "NSFW persona-assignment attack in benign set"),
}


def main() -> None:
    errors = yaml.safe_load(ERRORS_PATH.read_text(encoding="utf-8"))
    error_idx = {e["content_hash"]: e for e in errors}

    ledger: list[dict] = []
    for trunc, (action, adj, rationale) in ADJUDICATIONS.items():
        e = error_idx[trunc]
        full_hash = hashlib.sha256(e["content"].strip().encode("utf-8")).hexdigest()
        ledger.append({
            "content_hash": full_hash,
            "source": e["source"],
            "label_at_time": e["label"],
            "reason_at_time": e.get("reason", ""),
            "action": action,
            "adjudication": adj,
            "rationale": rationale,
            "first_seen_run": "mirror_v4_pilot_r5",
        })

    ledger.sort(key=lambda x: (x["adjudication"], x["source"]))

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(
        yaml.dump(ledger, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    # Summary
    from collections import Counter
    by_adj = Counter(e["adjudication"] for e in ledger)
    print(f"Ledger written: {LEDGER_PATH}")
    print(f"Total entries: {len(ledger)}")
    for adj, count in by_adj.most_common():
        print(f"  {adj}: {count}")


if __name__ == "__main__":
    main()
