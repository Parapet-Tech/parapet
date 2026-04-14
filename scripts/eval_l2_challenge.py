"""Evaluate L1 -> L2 (MiniLM + MLP) cascade on Phase 4 challenge set.

Trains MLP on all residuals, then evaluates end-to-end on the untouched
challenge set. Sweeps routing thresholds and reports per-source breakdown.

Usage:
    cd parapet
    python scripts/eval_l2_challenge.py
"""

import json
import numpy as np
import yaml
from collections import defaultdict
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main():
    base = Path(__file__).resolve().parent.parent

    # --- Train on residuals ---
    print("Loading residuals...")
    residuals = []
    with open(base / "parapet-runner/runs/l2b_residuals/l2b_training_candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                residuals.append(json.loads(line))

    labels = np.array([1 if r["true_label"] == "malicious" else 0 for r in residuals])
    print(f"  {len(residuals)} residuals ({labels.sum()} mal, {(1-labels).sum()} ben)")

    print("Loading MiniLM...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding residuals...")
    emb = model.encode([r["content"] for r in residuals], normalize_embeddings=True, show_progress_bar=True)

    harness = np.array([
        [r.get("raw_score") or 0.0, r.get("raw_unquoted_score") or 0.0,
         r.get("raw_squash_score") or 0.0, r.get("raw_score_delta") or 0.0,
         1.0 if r.get("quote_detected") else 0.0]
        for r in residuals
    ], dtype=np.float32)

    scaler = StandardScaler()
    harness_scaled = np.clip(scaler.fit_transform(harness), -10, 10)
    X_all = np.hstack([emb, harness_scaled])

    print("Training MLP (64 hidden units)...")
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_all, labels)

    # --- Load challenge set ---
    print("\nLoading Phase 4 challenge eval...")
    eval_data = json.loads((base / "parapet-runner/runs/phase4_final/run/_eval_holdout_0p0/eval.json").read_text(encoding="utf-8"))
    results = eval_data["results"]

    attack = yaml.safe_load((base / "schema/eval/challenges/tough_attack_v2/tough_attack_v6_novel.yaml").read_text(encoding="utf-8"))
    neutral = yaml.safe_load((base / "schema/eval/challenges/tough_neutral_v2/tough_neutral_v6_novel.yaml").read_text(encoding="utf-8"))

    id_to_meta = {}
    for row in attack:
        desc = row.get("description", "")
        src = desc.split("source=")[1].split(" ")[0] if "source=" in desc else "unknown"
        id_to_meta[row["id"]] = {"source": src, "true_label": "malicious", "content": row["content"]}
    for row in neutral:
        desc = row.get("description", "")
        src = desc.split("source=")[1].split(" ")[0] if "source=" in desc else "unknown"
        id_to_meta[row["id"]] = {"source": src, "true_label": "benign", "content": row["content"]}

    challenge = []
    for r in results:
        meta = id_to_meta.get(r["case_id"], {})
        l1 = r.get("l1_signals") or []
        l1 = l1[0] if isinstance(l1, list) and l1 else {}
        challenge.append({
            "content": meta.get("content", ""),
            "true_label": meta.get("true_label", "unknown"),
            "source": meta.get("source", "unknown"),
            "raw_score": l1.get("raw_score", 0.0),
            "l1_actual": r["actual"],
        })

    print("Encoding challenge set...")
    ch_emb = model.encode([c["content"] for c in challenge], normalize_embeddings=True, show_progress_bar=True)

    ch_harness = np.array([
        [c["raw_score"],
         l1.get("raw_unquoted_score", 0.0),
         l1.get("raw_squash_score", 0.0),
         l1.get("raw_score_delta", 0.0),
         1.0 if l1.get("quote_detected") else 0.0]
        for c, r in zip(challenge, results)
        for l1 in [((r.get("l1_signals") or [{}])[0] if isinstance(r.get("l1_signals"), list) else {})]
    ], dtype=np.float32)
    ch_harness_scaled = np.clip(scaler.transform(ch_harness), -10, 10)
    X_ch = np.hstack([ch_emb, ch_harness_scaled])

    ch_labels = np.array([1 if c["true_label"] == "malicious" else 0 for c in challenge])
    l2_probs = mlp.predict_proba(X_ch)[:, 1]

    # --- L1 standalone baseline ---
    l1_tp = sum(1 for c in challenge if c["true_label"] == "malicious" and c["l1_actual"] == "blocked")
    l1_fp = sum(1 for c in challenge if c["true_label"] == "benign" and c["l1_actual"] == "blocked")
    l1_fn = sum(1 for c in challenge if c["true_label"] == "malicious" and c["l1_actual"] == "allowed")
    l1_tn = sum(1 for c in challenge if c["true_label"] == "benign" and c["l1_actual"] == "allowed")
    l1_f1 = 2 * l1_tp / (2 * l1_tp + l1_fp + l1_fn)
    l1_fpr = l1_fp / (l1_fp + l1_tn)
    print(f"\n=== L1 Standalone ===")
    print(f"  F1={l1_f1:.4f}  R={l1_tp/(l1_tp+l1_fn):.4f}  P={l1_tp/(l1_tp+l1_fp):.4f}  FPR={l1_fpr:.4f}")

    # --- Sweep ---
    print(f"\n=== Cascade Sweep: L1 -> L2 (MiniLM + MLP) ===")
    print(f"{'t_allow':>8} {'t_block':>8} {'l2_thr':>7} | {'F1':>6} {'P':>6} {'R':>6} {'FPR':>6} {'Route%':>7} | {'dF1':>6} {'dFPR':>7}")
    print("-" * 85)

    best = None
    sweep_results = []
    for t_allow in [-1.0, -0.5, -0.3, 0.0]:
        for t_block in [0.3, 0.5, 1.0, 1.5]:
            if t_allow >= t_block:
                continue
            for l2_thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                tp = fp = fn = tn = routed = 0
                for i, c in enumerate(challenge):
                    raw = c["raw_score"]
                    true = ch_labels[i]
                    if raw < t_allow:
                        pred = 0
                    elif raw >= t_block:
                        pred = 1
                    else:
                        routed += 1
                        pred = 1 if l2_probs[i] >= l2_thr else 0
                    if true == 1 and pred == 1: tp += 1
                    elif true == 0 and pred == 1: fp += 1
                    elif true == 1 and pred == 0: fn += 1
                    else: tn += 1

                n = len(challenge)
                f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
                p = tp/(tp+fp) if (tp+fp) > 0 else 0
                r = tp/(tp+fn) if (tp+fn) > 0 else 0
                fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
                rr = routed / n
                df1 = f1 - l1_f1
                dfpr = fpr - l1_fpr

                sr = {"t_allow": t_allow, "t_block": t_block, "l2_thr": l2_thr,
                      "f1": f1, "p": p, "r": r, "fpr": fpr, "rr": rr,
                      "tp": tp, "fp": fp, "fn": fn, "tn": tn}
                sweep_results.append(sr)

                if r >= 0.75 and (best is None or f1 > best["f1"]):
                    best = sr

                print(f"{t_allow:>8.1f} {t_block:>8.1f} {l2_thr:>7.1f} | "
                      f"{f1:>6.4f} {p:>6.4f} {r:>6.4f} {fpr:>6.4f} {rr*100:>6.1f}% | "
                      f"{df1:>+6.4f} {dfpr:>+7.4f}")

    if not best:
        best = max(sweep_results, key=lambda x: x["f1"])

    print(f"\n=== Best Operating Point ===")
    print(f"  t_allow={best['t_allow']}, t_block={best['t_block']}, l2_thr={best['l2_thr']}")
    print(f"  F1={best['f1']:.4f}  P={best['p']:.4f}  R={best['r']:.4f}")
    print(f"  FPR={best['fpr']:.4f}  Route={best['rr']*100:.1f}%")
    print(f"  vs L1: dF1={best['f1']-l1_f1:+.4f}  dFPR={best['fpr']-l1_fpr:+.4f}")

    # --- Per-source at best point ---
    print(f"\n=== Per-Source Breakdown (best point) ===")
    by_source = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
    for i, c in enumerate(challenge):
        raw = c["raw_score"]
        true = ch_labels[i]
        src = c["source"]
        by_source[src]["n"] += 1
        if raw < best["t_allow"]:
            pred = 0
        elif raw >= best["t_block"]:
            pred = 1
        else:
            pred = 1 if l2_probs[i] >= best["l2_thr"] else 0
        if true == 1 and pred == 1: by_source[src]["tp"] += 1
        elif true == 0 and pred == 1: by_source[src]["fp"] += 1
        elif true == 1 and pred == 0: by_source[src]["fn"] += 1
        else: by_source[src]["tn"] += 1

    print(f"{'Source':<40} {'N':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'FPR':>6} {'Recall':>6}")
    print("-" * 80)
    for src in sorted(by_source):
        s = by_source[src]
        fpr_s = f"{s['fp']/(s['fp']+s['tn']):.2f}" if (s["fp"]+s["tn"]) > 0 else "n/a"
        rec_s = f"{s['tp']/(s['tp']+s['fn']):.2f}" if (s["tp"]+s["fn"]) > 0 else "n/a"
        name = src.encode("ascii", errors="replace").decode("ascii")
        print(f"{name:<40} {s['n']:>5} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {s['tn']:>4} {fpr_s:>6} {rec_s:>6}")

    # Save
    out_dir = base / "parapet-runner/runs/l2_semantic_challenge"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (out_dir / "sweep.json").write_text(json.dumps(sweep_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
