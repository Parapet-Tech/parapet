"""Simple data quality stats - writes JSON output."""
import json, hashlib, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parapet_data.curated_artifact import load_curated_entries

p = Path("C:/Users/anyth/MINE/dev/DefenseSector/parapet/parapet-data/curated/v3_19k_clean/curated.yaml")
out = Path("C:/Users/anyth/MINE/dev/DefenseSector/parapet/parapet-data/scripts/dq_result.json")

print("Loading...")
data = load_curated_entries(p)
print(f"Loaded {len(data)} samples")

r = {}

# Short/long
short = [(i, len(s["content"])) for i, s in enumerate(data) if len(s.get("content","").strip()) < 10]
long_ = [(i, len(s["content"])) for i, s in enumerate(data) if len(s.get("content","")) > 10000]
r["short_count"] = len(short)
r["long_count"] = len(long_)
r["short_indices"] = [x[0] for x in short[:20]]
r["long_indices"] = [x[0] for x in long_[:20]]

# Exact dupes
hashes = defaultdict(list)
for i, s in enumerate(data):
    h = hashlib.sha256(s.get("content","").encode()).hexdigest()[:16]
    hashes[h].append(i)
dupes = {h: idxs for h, idxs in hashes.items() if len(idxs) > 1}
r["exact_dupe_groups"] = len(dupes)
r["exact_dupe_extra"] = sum(len(v)-1 for v in dupes.values())
# Check for label conflicts
conflicts = 0
for h, idxs in dupes.items():
    labels = set(data[i]["label"] for i in idxs)
    if len(labels) > 1:
        conflicts += 1
r["label_conflict_dupes"] = conflicts

# Language mismatch: EN-labeled but non-Latin majority
lang_mm = 0
for i, s in enumerate(data):
    if s.get("language") != "EN":
        continue
    c = s.get("content","")
    lat = sum(1 for ch in c if ch.isascii() and ch.isalpha())
    non = sum(1 for ch in c if not ch.isascii() and ch.isalpha())
    if lat + non > 10 and non/(lat+non) > 0.5:
        lang_mm += 1
r["language_mismatch_en_nonlatin"] = lang_mm

# Malicious no signal (short)
pi_kw = r"ignore|disregard|forget|override|pretend|jailbreak|DAN|system.?prompt|PWNED|hack|inject|bypass|exploit|payload|backdoor|password|secret|token|instructions|sudo|admin|hidden|confidential"
no_signal = 0
for i, s in enumerate(data):
    if s.get("label") != "malicious":
        continue
    c = s.get("content","")
    if len(c) > 150:
        continue
    if re.search(pi_kw, c, re.IGNORECASE):
        continue
    no_signal += 1
r["malicious_short_no_signal"] = no_signal

print("Writing results...")
json.dump(r, open(out, "w"), indent=2)
print(f"Done: {out}")
