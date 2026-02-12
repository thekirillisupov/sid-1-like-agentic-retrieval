#!/usr/bin/env bash
# Evaluate a trained model and run baselines.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

CONFIG="${1:-training/config.yaml}"
CHECKPOINT="${2:-checkpoints/epoch_3}"
NUM_ROLLOUTS="${3:-1}"

echo "=== Running baselines ==="
python -m evaluation.baselines \
    --config "$CONFIG" \
    --baselines vector reranker \
    --output results/baselines.json

echo ""
echo "=== Evaluating trained model (${NUM_ROLLOUTS}x rollouts) ==="
python -m evaluation.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --output results/eval.json

echo ""
echo "=== Results ==="
echo "Baselines: results/baselines.json"
echo "Eval:      results/eval.json"
