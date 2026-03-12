"""Step 1: Basic stats only - no content echoing."""
import yaml, json
from collections import Counter, defaultdict
from pathlib import Path

p = Path("C:/Users/anyth/MINE/dev/DefenseSector/parapet/parapet-data/curated/v3_19k_clean/curated.yaml")
data = yaml.safe_load(open(p, encoding="utf-8"))

stats = {
    "total": len(data),
    "labels": dict(Counter(s.get("label","?") for s in data).most_common()),
    "languages": dict(Counter(s.get("language","?") for s in data).most_common()),
    "reasons": dict(Counter(s.get("reason","?") for s in data).most_common()),
    "sources_top20": dict(Counter(s.get("source","?") for s in data).most_common(20)),
    "format_bins": dict(Counter(s.get("format_bin","?") for s in data).most_common()),
    "length_bins": dict(Counter(s.get("length_bin","?") for s in data).most_common()),
}

# Cross-tab
xtab = defaultdict(Counter)
for s in data:
    xtab[s.get("label","?")][s.get("language","?")] += 1
stats["label_x_language"] = {k: dict(v) for k, v in xtab.items()}

# Length stats
lengths = [len(s.get("content","")) for s in data]
stats["content_length"] = {
    "min": min(lengths),
    "max": max(lengths),
    "mean": sum(lengths) / len(lengths),
    "short_lt10": sum(1 for l in lengths if l < 10),
    "long_gt10000": sum(1 for l in lengths if l > 10000),
}

out = Path("C:/Users/anyth/MINE/dev/DefenseSector/parapet/parapet-data/scripts/stats_output.json")
json.dump(stats, open(out, "w"), indent=2)
print(f"Stats written to {out}")
