#!/usr/bin/env bash
# Run GRPO training for agentic retrieval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

CONFIG="${1:-training/config.yaml}"

echo "=== Training with config: $CONFIG ==="
python -m training.train_grpo --config "$CONFIG"
