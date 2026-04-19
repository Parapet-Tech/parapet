"""Data quality check - outputs JSON stats to dq_result.json"""
import sys, os, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parapet_data.curated_artifact import load_curated_entries

# Use env var to locate data
base = Path(os.environ.get("DQ_BASE", Path(__file__).resolve().parent.parent))
target = base / "curated" / "v3_19k_clean" / "curated.yaml"
data = load_curated_entries(target)

result = {}

# Basic counts
result["total"] = len(data)
result["labels"] = dict(Counter(s.get("label","?") for s in data).most_common())
result["languages"] = dict(Counter(s.get("language","?") for s in data).most_common())
result["reasons"] = dict(Counter(s.get("reason","?") for s in data).most_common())
result["sources_top20"] = dict(Counter(s.get("source","?") for s in data).most_common(20))
result["format_bins"] = dict(Counter(s.get("format_bin","?") for s in data).most_common())
result["length_bins"] = dict(Counter(s.get("length_bin","?") for s in data).most_common())

# Cross-tab
xtab = defaultdict(Counter)
for s in data:
    xtab[s.get("label","?")][s.get("language","?")] += 1
result["label_x_language"] = {k: dict(v) for k, v in xtab.items()}

# Content lengths
lengths = [len(s.get("content","")) for s in data]
result["content_length_stats"] = {
    "min": min(lengths), "max": max(lengths),
    "mean": round(sum(lengths)/len(lengths), 1),
    "median": sorted(lengths)[len(lengths)//2],
    "count_lt10": sum(1 for l in lengths if l < 10),
    "count_gt10000": sum(1 for l in lengths if l > 10000),
    "count_gt5000": sum(1 for l in lengths if l > 5000),
}

# Short samples (< 10 chars): indices and metadata only
short_items = []
for i, s in enumerate(data):
    c = s.get("content","")
    if len(c.strip()) < 10:
        short_items.append({"idx": i, "label": s.get("label"), "reason": s.get("reason"),
                           "source": s.get("source"), "lang": s.get("language"),
                           "content_len": len(c), "content_preview": c[:50]})
result["short_samples"] = short_items

# Long samples (> 10000 chars)
long_items = []
for i, s in enumerate(data):
    c = s.get("content","")
    if len(c) > 10000:
        long_items.append({"idx": i, "label": s.get("label"), "reason": s.get("reason"),
                          "source": s.get("source"), "lang": s.get("language"),
                          "content_len": len(c)})
result["long_samples"] = long_items

# Exact duplicates
content_map = defaultdict(list)
for i, s in enumerate(data):
    content_map[s.get("content","")].append(i)
dupes = {k: v for k, v in content_map.items() if len(v) > 1}
result["exact_dupes_count"] = len(dupes)
result["exact_dupes_extra_copies"] = sum(len(v)-1 for v in dupes.values())

# Dupe examples (just indices and metadata, not content)
dupe_examples = []
for content, indices in list(dupes.items())[:10]:
    labels = [data[i].get("label") for i in indices]
    sources = [data[i].get("source") for i in indices]
    reasons = [data[i].get("reason") for i in indices]
    conflicting = len(set(labels)) > 1
    dupe_examples.append({"indices": indices, "labels": labels, "sources": sources,
                          "reasons": reasons, "conflicting_labels": conflicting,
                          "content_preview": content[:100].replace("\n"," ")})
result["exact_dupe_examples"] = dupe_examples

# Label conflicts in dupes
conflicts = []
for content, indices in dupes.items():
    labels = [data[i].get("label") for i in indices]
    if len(set(labels)) > 1:
        conflicts.append({"indices": indices, "labels": labels,
                         "content_preview": content[:100].replace("\n"," ")})
result["label_conflicts"] = conflicts
result["label_conflicts_count"] = len(conflicts)

# Near duplicates
norm_map = defaultdict(list)
for i, s in enumerate(data):
    c = s.get("content","")
    norm = re.sub(r"\s+", " ", c.strip().lower())
    norm_map[norm].append(i)
near_dupes = {k: v for k, v in norm_map.items() if len(v) > 1}
# Subtract exact dupes
near_only_count = len(near_dupes) - len(dupes)
result["near_dupe_groups_total"] = len(near_dupes)
result["near_dupe_groups_beyond_exact"] = near_only_count

# Benign samples with PI-like patterns
pi_regexes = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions|programming|rules)",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions|prompts|rules)",
    r"new\s+instructions?\s*:",
    r"your\s+new\s+(role|task|instructions?)\s+(is|are)\b",
    r"from\s+now\s+on.{0,20}(you\s+)?(will|must|should|are)\b",
    r"\byou\s+are\s+now\s+(DAN|unrestricted|unfiltered)",
    r"DAN\s+(mode|jailbreak)",
    r"jailbreak\s*(mode)?",
    r"developer\s+mode\s+(enabled|activated)",
    r"system\s*(prompt|instruction)",
    r"<<\s*SYS\s*>>",
    r"(show|reveal|print|display|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions)",
    r"(send|transmit|exfiltrate|leak).{0,30}(to|via)\s+(https?://|webhook)",
    r"!\[.*\]\(https?://.*\?.*=",
    r"\bI\s+have\s+been\s+PWNED\b",
    r"\bPWNED\b",
]
compiled_pi = [re.compile(p, re.IGNORECASE) for p in pi_regexes]

benign_with_pi = []
for i, s in enumerate(data):
    if s.get("label") != "benign":
        continue
    c = s.get("content","")
    for pat in compiled_pi:
        m = pat.search(c)
        if m:
            benign_with_pi.append({"idx": i, "reason": s.get("reason"),
                                   "source": s.get("source"), "lang": s.get("language"),
                                   "matched": m.group()[:60],
                                   "content_preview": c[:120].replace("\n"," ")})
            break
result["benign_with_pi_count"] = len(benign_with_pi)
result["benign_with_pi_examples"] = benign_with_pi[:20]

# Malicious with no PI signal (short samples only)
weak_kw = [
    r"hack", r"pwn", r"inject", r"bypass", r"jailbreak", r"exploit",
    r"payload", r"prompt", r"instruction", r"override", r"ignore", r"disregard",
    r"pretend", r"roleplay", r"sudo", r"admin", r"password", r"secret",
    r"PWNED", r"DAN\b", r"ChatGPT", r"OpenAI",
    r"system\s*prompt", r"hidden", r"confidential",
    r"repeat\s+after", r"say\s+exactly", r"respond\s+with",
    r"forget\s+everything", r"I have been",
]
compiled_weak = [re.compile(p, re.IGNORECASE) for p in weak_kw]

mal_no_signal = []
for i, s in enumerate(data):
    if s.get("label") != "malicious":
        continue
    c = s.get("content","")
    if len(c) > 150:
        continue
    if any(p.search(c) for p in compiled_pi):
        continue
    if any(p.search(c) for p in compiled_weak):
        continue
    mal_no_signal.append({"idx": i, "reason": s.get("reason"),
                          "source": s.get("source"), "lang": s.get("language"),
                          "content_preview": c[:120].replace("\n"," ")})
result["malicious_no_signal_count"] = len(mal_no_signal)
result["malicious_no_signal_examples"] = mal_no_signal[:20]

# Background samples with PI patterns
bg_pi = []
for i, s in enumerate(data):
    if s.get("reason") != "background":
        continue
    c = s.get("content","")
    for pat in compiled_pi:
        m = pat.search(c)
        if m:
            bg_pi.append({"idx": i, "label": s.get("label"),
                         "source": s.get("source"), "lang": s.get("language"),
                         "matched": m.group()[:60],
                         "content_preview": c[:120].replace("\n"," ")})
            break
result["background_with_pi_count"] = len(bg_pi)
result["background_with_pi_examples"] = bg_pi[:20]

# Reason-content mismatch for specific reasons
exfil_pats = [re.compile(p, re.IGNORECASE) for p in [
    r"(send|transmit|post|exfiltrate|leak|forward|email|upload)",
    r"https?://", r"(webhook|api|endpoint|url|server|domain)",
    r"!\[.*\]\(", r"(fetch|request|curl|wget)",
]]
meta_pats = [re.compile(p, re.IGNORECASE) for p in [
    r"(system\s*prompt|initial\s*instruction|hidden\s*(instruction|prompt))",
    r"(show|reveal|print|display|output|repeat|echo|tell|give)\s+(me\s+)?(your|the)",
    r"what\s+(is|are)\s+your",
]]
obfusc_pats = [re.compile(p, re.IGNORECASE) for p in [
    r"base64", r"\\x[0-9a-fA-F]{2}", r"&#x?[0-9a-fA-F]+;",
    r"[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]",
    r"[^\x00-\x7F]{5,}",
    r"(rot13|caesar|cipher|encod|decode|obfuscat)",
    r"(zero.width|unicode|homoglyph|invisible)",
]]
reason_pats_map = {"exfiltration": exfil_pats, "meta_probe": meta_pats, "obfuscation": obfusc_pats}

reason_mismatch = []
for i, s in enumerate(data):
    if s.get("label") != "malicious":
        continue
    reason = s.get("reason","")
    if reason not in reason_pats_map:
        continue
    c = s.get("content","")
    if not any(p.search(c) for p in reason_pats_map[reason]):
        reason_mismatch.append({"idx": i, "reason": reason,
                                "source": s.get("source"), "lang": s.get("language"),
                                "content_preview": c[:120].replace("\n"," ")})
result["reason_mismatch_count"] = len(reason_mismatch)
result["reason_mismatch_by_reason"] = dict(Counter(rm["reason"] for rm in reason_mismatch))
result["reason_mismatch_examples"] = reason_mismatch[:20]

# Language mismatch (declared EN but mostly non-Latin)
lang_mismatch = []
for i, s in enumerate(data):
    if s.get("language") != "EN":
        continue
    c = s.get("content","")
    latin = sum(1 for ch in c if ch.isascii() and ch.isalpha())
    nonlatin = sum(1 for ch in c if not ch.isascii() and ch.isalpha())
    total = latin + nonlatin
    if total > 10 and nonlatin / total > 0.5:
        lang_mismatch.append({"idx": i, "label": s.get("label"), "reason": s.get("reason"),
                             "source": s.get("source"),
                             "nonlatin_ratio": round(nonlatin/total, 2),
                             "content_preview": c[:80].replace("\n"," ")})
result["language_mismatch_count"] = len(lang_mismatch)
result["language_mismatch_examples"] = lang_mismatch[:20]

# Write output
outfile = base / "scripts" / "dq_result.json"
with open(outfile, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Results: {outfile}")
