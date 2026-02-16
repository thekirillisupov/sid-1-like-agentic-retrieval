# sid-1-like-agentic-retrieval

Reproduction of the core experiment from the [SID-1 technical report](https://www.sid.ai/research/sid-1-technical-report): train a small LLM with multi-turn reinforcement learning (GRPO) to perform agentic retrieval over the [MuSiQue](https://github.com/StonyBrookNLP/musique) multi-hop QA dataset.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (24 GB+ VRAM recommended for Qwen3-4B)
- ~10 GB disk for corpus embeddings and model checkpoints

## Installation

```bash
pip install -e .

# If you have a CUDA-capable GPU and want faiss-gpu:
pip install -e ".[gpu]"
```

## Quickstart

The pipeline has three stages. Run them in order:

### 1. Data preparation

Downloads MuSiQue from HuggingFace, builds a unified paragraph corpus, and creates a FAISS vector index.

```bash
bash scripts/run_data_prep.sh
```

Or step by step:

```bash
# Download raw dataset
python -m data.download_musique --out-dir artifacts/raw

# Build corpus + FAISS index
python -m data.build_corpus --raw-dir artifacts/raw --out-dir artifacts/corpus
```

This produces:

| File | Description |
|------|-------------|
| `artifacts/corpus/corpus.jsonl` | Deduplicated paragraphs with `doc_id`, `title`, `text` |
| `artifacts/corpus/questions.jsonl` | Questions with `target_doc_ids` for reward computation |
| `artifacts/corpus/corpus.faiss` | FAISS inner-product index over paragraph embeddings |
| `artifacts/corpus/doc_id_map.json` | Ordered mapping from FAISS row index to `doc_id` |

### 2. Training

Runs the GRPO training loop. Edit `training/config.yaml` to adjust hyperparameters.

```bash
bash scripts/run_train.sh

# Or with a custom config:
bash scripts/run_train.sh path/to/config.yaml

# Or directly:
python -m training.train_grpo --config training/config.yaml
```

Key config knobs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `Qwen/Qwen3-4B` | Base model (scale to `Qwen/Qwen3-14B` later) |
| `grpo.group_size` | 8 | Rollouts per question |
| `training.batch_size` | 4 | Questions per batch |
| `training.epochs` | 3 | Passes over the training set |
| `environment.max_turns` | 8 | Max tool-use turns per episode |
| `reward.ndcg_weight` | 0.9 | Weight for NDCG in combined reward |
| `reward.format_weight` | 0.1 | Weight for format adherence reward |

Checkpoints are saved to `checkpoints/epoch_N/`.

### 3. Evaluation

Run baselines and evaluate the trained model on the dev set.

```bash
bash scripts/run_eval.sh                              # defaults
bash scripts/run_eval.sh training/config.yaml checkpoints/epoch_3 4  # 4x RRF
```

Or individually:

```bash
# Baselines (vector-only, reranker)
python -m evaluation.baselines --config training/config.yaml --output results/baselines.json

# Trained model (1x)
python -m evaluation.evaluate --config training/config.yaml \
    --checkpoint checkpoints/epoch_3 --output results/eval_1x.json

# Trained model (4x with RRF)
python -m evaluation.evaluate --config training/config.yaml \
    --checkpoint checkpoints/epoch_3 --num-rollouts 4 --output results/eval_4x.json
```

## Project structure

```
├── data/
│   ├── download_musique.py        # Download and preprocess MuSiQue
│   └── build_corpus.py            # Build unified corpus + FAISS index
├── environment/
│   ├── embedding_index.py         # FAISS index wrapper
│   ├── tools.py                   # search() / read() with ID obfuscation
│   └── retrieval_env.py           # Multi-turn retrieval environment
├── reward/
│   ├── ndcg.py                    # NDCG reward (binary relevance)
│   └── format_reward.py           # Format adherence reward
├── training/
│   ├── config.yaml                # Hyperparameters
│   ├── tokenization.py            # TI/TO-safe tokenization utilities
│   └── train_grpo.py              # GRPO training loop
├── evaluation/
│   ├── metrics.py                 # NDCG, recall, precision, F1, RRF
│   ├── evaluate.py                # Eval on held-out set
│   └── baselines.py               # Vector-only, reranker, base model
└── scripts/
    ├── run_data_prep.sh
    ├── run_train.sh
    └── run_eval.sh
```

## Key design decisions from SID-1

- **TI/TO (Tokens-In/Tokens-Out)**: raw token IDs flow from vLLM generation through tool execution to the training step — no lossy retokenisation.
- **Per-sequence length normalisation**: advantages are divided by the generated token count per sequence, not per group (avoids the Dr. GRPO OOV-sampling issue).
- **ID obfuscation**: document IDs are randomised each episode to prevent memorisation.
- **Unified corpus**: paragraphs are pooled across all questions for realistic retrieval difficulty.
- **Hierarchical retrieval**: `search()` returns 200-char snippets; `read()` returns full text.
- **NDCG reward**: training signal comes from document retrieval quality, not answer correctness.
- **Format reward**: included from the start to prevent format regression during RL.
