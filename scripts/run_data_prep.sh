#!/usr/bin/env bash
# Download MuSiQue and build the unified corpus + FAISS index.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Step 1: Download MuSiQue ==="
python -m data.download_musique --out-dir artifacts/raw

echo ""
echo "=== Step 2: Build corpus + FAISS index ==="
python -m data.build_corpus \
    --raw-dir artifacts/raw \
    --out-dir artifacts/corpus \
    --splits train dev

echo ""
echo "=== Done ==="
echo "Corpus:    artifacts/corpus/corpus.jsonl"
echo "Questions: artifacts/corpus/questions.jsonl"
echo "Index:     artifacts/corpus/corpus.faiss"
echo "ID map:    artifacts/corpus/doc_id_map.json"
