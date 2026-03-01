#!/usr/bin/env bash
set -euo pipefail

# L2a Targeted Eval — full stack + L2a against L1's weakest datasets
# Compares full stack (L1+L3+L4) vs full stack + L2a.
# Source: implement/evals/v4/baseline.md

cd "$(dirname "$0")/../parapet"

echo "=== Building with L2a feature (MSVC target) ==="
cargo build --features eval,l2a --release --target x86_64-pc-windows-msvc

EVAL_BIN="./target/x86_64-pc-windows-msvc/release/parapet-eval"
DATASET_DIR="../schema/eval"
OUT_DIR="../schema/eval/results/l2a_fullstack"
mkdir -p "$OUT_DIR"

# Datasets where L1 is weakest (from implement/evals/v4/baseline.md)
SOURCES=(
    opensource_jbb_paraphrase_attacks    # 0% L1 recall
    opensource_llmail_attacks            # 4% L1 recall
    opensource_safeguard_attacks         # 8% L1 recall
    opensource_geekyrakshit_attacks      # 8.5% L1 recall
    opensource_jailbreakv_attacks        # 27.5% L1 recall
    opensource_bipia_benign              # 20% L1 accuracy (80% FP)
)

# --- Run 1: baseline (no L2a) ---
echo ""
echo "=== Baseline: full stack WITHOUT L2a ==="
echo ""
for src in "${SOURCES[@]}"; do
    echo "--- $src ---"
    $EVAL_BIN --config ../schema/eval/eval_config.yaml --dataset "$DATASET_DIR" \
        --source "$src" --max-failures 200 \
        --json --output "$OUT_DIR/baseline_${src}.json" 2>&1
    echo ""
done

# --- Run 2: full stack + L2a ---
echo ""
echo "=== Full stack WITH L2a ==="
echo ""
for src in "${SOURCES[@]}"; do
    echo "--- $src ---"
    $EVAL_BIN --config ../schema/eval/eval_config_with_l2a.yaml --dataset "$DATASET_DIR" \
        --source "$src" --max-failures 200 \
        --json --output "$OUT_DIR/with_l2a_${src}.json" 2>&1
    echo ""
done

echo "=== Done. Results in $OUT_DIR/ ==="
