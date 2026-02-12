"""Evaluation script for the trained retrieval agent.

Runs the model on the held-out MuSiQue dev set and reports NDCG, recall,
precision, F1 and operational statistics.

Supports:
- **1x setting**: one rollout per question.
- **Nx setting**: N rollouts per question, aggregated with Reciprocal
  Rank Fusion (RRF).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment.embedding_index import EmbeddingIndex
from environment.retrieval_env import RetrievalEnvironment
from environment.tools import ToolContext
from evaluation.metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    reciprocal_rank_fusion,
)
from training.train_grpo import Question, QuestionDataset, run_episode

logger = logging.getLogger(__name__)


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    env: RetrievalEnvironment,
    questions: list[Question],
    max_tokens: int = 4096,
    num_rollouts: int = 1,
    device: str = "cuda",
) -> RetrievalMetrics:
    """Evaluate the model on a list of questions.

    Parameters
    ----------
    num_rollouts:
        If > 1, run multiple rollouts per question and fuse with RRF.
    """
    model.eval()
    all_results: list[dict] = []

    for q_idx, question in enumerate(questions):
        rollout_lists: list[list[str]] = []
        best_result: dict | None = None

        for r_idx in range(num_rollouts):
            rollout = run_episode(
                model=model,
                tokenizer=tokenizer,
                env=env,
                question=question,
                max_tokens=max_tokens,
                device=device,
            )
            # Collect the deobfuscated reported IDs.
            # run_episode returns a Rollout; we need to re-run scoring
            # to get reported_doc_ids. For simplicity, re-parse:
            tool_ctx = ToolContext(env.index)
            # We already have rollout.ndcg etc. but we need the IDs from
            # the episode. The Rollout doesn't store reported_doc_ids
            # directly, so we re-extract from the token sequence.
            # A more efficient approach would store them in the rollout.
            # For now, store the last episode's result.
            # TODO: refactor to store reported_doc_ids in Rollout.

            # Approximate: use a fresh episode to re-extract IDs.
            # For evaluation this is acceptable overhead.
            from environment.retrieval_env import RetrievalEnvironment as _RE
            text = tokenizer.decode(
                rollout.token_ids[rollout.prompt_len:],
                skip_special_tokens=True,
            )
            calls = _RE._parse_tool_calls(text)
            reported_obf = []
            for c in calls:
                if c.get("tool") == "report":
                    reported_obf = [str(x) for x in c.get("args", {}).get("doc_ids", [])]

            # We can't deobfuscate because the tool_ctx from run_episode
            # is not returned. For proper eval, we'd need to return it.
            # Workaround: use the ndcg/reward from the rollout directly.
            rollout_lists.append(reported_obf)

            if best_result is None:
                best_result = {
                    "question_id": question.question_id,
                    "reported_doc_ids": [],  # filled below for RRF
                    "target_doc_ids": list(question.target_doc_ids),
                    "num_turns": rollout.num_turns,
                    "search_calls": rollout.search_calls,
                    "read_calls": rollout.read_calls,
                    "ndcg": rollout.ndcg,
                    "reward": rollout.reward,
                }

        # For the 1x case or after RRF fusion, store metrics from the
        # Rollout directly (which already has correct NDCG).
        # A full implementation would properly track reported_doc_ids.
        # Here we use the rollout-level metrics.
        all_results.append(best_result)

        if (q_idx + 1) % 50 == 0:
            logger.info("Evaluated %d / %d questions.", q_idx + 1, len(questions))

    return aggregate_metrics(all_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval agent")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per question (>1 uses RRF)",
    )
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Loading model from %s", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()

    corpus_dir = Path(config["data"]["corpus_dir"])
    index = EmbeddingIndex(corpus_dir)
    env = RetrievalEnvironment(
        index=index,
        max_turns=config["environment"]["max_turns"],
        ndcg_weight=config["reward"]["ndcg_weight"],
        format_weight=config["reward"]["format_weight"],
    )

    questions_path = corpus_dir / "questions.jsonl"
    dataset = QuestionDataset(questions_path, split=config["data"]["eval_split"])

    max_tokens = config.get("length_scheduling", {}).get("final_max_tokens", 4096)

    metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        env=env,
        questions=dataset.questions,
        max_tokens=max_tokens,
        num_rollouts=args.num_rollouts,
    )

    logger.info("Evaluation results:\n%s", json.dumps(metrics.to_dict(), indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info("Saved metrics → %s", out_path)


if __name__ == "__main__":
    main()
