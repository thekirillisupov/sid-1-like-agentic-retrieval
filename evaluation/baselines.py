"""Baseline methods for comparison with the RL-trained retrieval agent.

1. **Vector-only @K**: embed the question, return top-K nearest paragraphs.
2. **Reranker @K**: retrieve top-N candidates, rerank with a cross-encoder,
   return top-K.
3. **Base model (no RL)**: run the untrained base model in the same
   multi-turn environment.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment.embedding_index import EmbeddingIndex
from environment.retrieval_env import RetrievalEnvironment
from environment.tools import ToolContext
from evaluation.metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    compute_f1,
    compute_precision,
    compute_recall,
)
from reward.ndcg import compute_ndcg
from training.train_grpo import Question, QuestionDataset, run_episode

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Vector-only baseline
# ======================================================================

def vector_only_baseline(
    index: EmbeddingIndex,
    questions: list[Question],
    top_k: int = 10,
) -> RetrievalMetrics:
    """Embed each question and return the top-K nearest paragraphs."""
    results: list[dict] = []
    for question in questions:
        hits = index.search(question.question, top_k=top_k)
        reported = [h["doc_id"] for h in hits]
        results.append(
            {
                "reported_doc_ids": reported,
                "target_doc_ids": list(question.target_doc_ids),
                "num_turns": 1,
                "search_calls": 1,
                "read_calls": 0,
            }
        )
    return aggregate_metrics(results)


# ======================================================================
# 2. Reranker baseline
# ======================================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reranker_baseline(
    index: EmbeddingIndex,
    questions: list[Question],
    retrieve_k: int = 30,
    rerank_k: int = 10,
    reranker_name: str = RERANKER_MODEL,
) -> RetrievalMetrics:
    """Retrieve top-N, rerank with a cross-encoder, return top-K."""
    logger.info("Loading reranker: %s", reranker_name)
    reranker = CrossEncoder(reranker_name)

    results: list[dict] = []
    for q_idx, question in enumerate(questions):
        # Retrieve candidates.
        hits = index.search(question.question, top_k=retrieve_k)
        candidates = []
        for h in hits:
            doc = index.read(h["doc_id"])
            if doc:
                candidates.append((h["doc_id"], doc["text"]))

        if not candidates:
            results.append(
                {
                    "reported_doc_ids": [],
                    "target_doc_ids": list(question.target_doc_ids),
                    "num_turns": 1,
                    "search_calls": 1,
                    "read_calls": 0,
                }
            )
            continue

        # Rerank.
        pairs = [(question.question, text) for _, text in candidates]
        scores = reranker.predict(pairs)

        ranked_indices = np.argsort(scores)[::-1][:rerank_k]
        reported = [candidates[i][0] for i in ranked_indices]

        results.append(
            {
                "reported_doc_ids": reported,
                "target_doc_ids": list(question.target_doc_ids),
                "num_turns": 1,
                "search_calls": 1,
                "read_calls": len(candidates),
            }
        )

        if (q_idx + 1) % 100 == 0:
            logger.info("Reranked %d / %d", q_idx + 1, len(questions))

    return aggregate_metrics(results)


# ======================================================================
# 3. Base model (no RL) baseline
# ======================================================================

def base_model_baseline(
    model_name: str,
    index: EmbeddingIndex,
    questions: list[Question],
    max_turns: int = 8,
    max_tokens: int = 4096,
    device: str = "cuda",
) -> RetrievalMetrics:
    """Run the untrained base model in the retrieval environment."""
    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    env = RetrievalEnvironment(index=index, max_turns=max_turns)

    results: list[dict] = []
    for q_idx, question in enumerate(questions):
        rollout = run_episode(
            model=model,
            tokenizer=tokenizer,
            env=env,
            question=question,
            max_tokens=max_tokens,
            device=device,
        )
        results.append(
            {
                "reported_doc_ids": [],  # not directly available from Rollout
                "target_doc_ids": list(question.target_doc_ids),
                "num_turns": rollout.num_turns,
                "search_calls": rollout.search_calls,
                "read_calls": rollout.read_calls,
                "ndcg": rollout.ndcg,
            }
        )
        if (q_idx + 1) % 50 == 0:
            logger.info("Base model: %d / %d", q_idx + 1, len(questions))

    return aggregate_metrics(results)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval baselines")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["vector", "reranker"],
        choices=["vector", "reranker", "base_model"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    corpus_dir = Path(config["data"]["corpus_dir"])
    index = EmbeddingIndex(corpus_dir)

    questions_path = corpus_dir / "questions.jsonl"
    dataset = QuestionDataset(questions_path, split=config["data"]["eval_split"])
    questions = dataset.questions

    all_metrics: dict[str, dict] = {}

    if "vector" in args.baselines:
        logger.info("--- Vector-only @%d ---", args.top_k)
        m = vector_only_baseline(index, questions, top_k=args.top_k)
        logger.info("Results: %s", json.dumps(m.to_dict(), indent=2))
        all_metrics["vector_only"] = m.to_dict()

    if "reranker" in args.baselines:
        logger.info("--- Reranker @%d ---", args.top_k)
        m = reranker_baseline(index, questions, rerank_k=args.top_k)
        logger.info("Results: %s", json.dumps(m.to_dict(), indent=2))
        all_metrics["reranker"] = m.to_dict()

    if "base_model" in args.baselines:
        logger.info("--- Base model (no RL) ---")
        m = base_model_baseline(
            config["model"]["name"],
            index,
            questions,
            max_turns=config["environment"]["max_turns"],
        )
        logger.info("Results: %s", json.dumps(m.to_dict(), indent=2))
        all_metrics["base_model"] = m.to_dict()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info("Saved all metrics → %s", out_path)


if __name__ == "__main__":
    main()
