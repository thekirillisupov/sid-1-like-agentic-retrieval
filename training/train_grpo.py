"""Main GRPO training loop for agentic retrieval.

Implements Group Relative Policy Optimisation (GRPO) with multi-turn
environment interaction, following the SID-1 paper:

* Multiple rollouts per question (``group_size``).
* Per-group advantage normalisation (mean/std of rewards within the group).
* Per-sequence length normalisation (divide advantage by token count) –
  **not** the Dr. GRPO per-group variant which causes OOV sampling.
* TI/TO: raw token IDs flow from vLLM generation through environment
  interaction to the training step without retokenisation.
* Length scheduling: ``max_tokens`` linearly ramps up over training.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment.embedding_index import EmbeddingIndex
from environment.retrieval_env import SYSTEM_PROMPT, RetrievalEnvironment
from environment.tools import ToolContext
from training.tokenization import (
    encode_initial_prompt,
    encode_text,
    encode_tool_result,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Data
# ======================================================================

@dataclass
class Question:
    question_id: str
    question: str
    target_doc_ids: set[str] = field(default_factory=set)


class QuestionDataset(Dataset):
    def __init__(self, questions_path: Path, split: str = "train") -> None:
        self.questions: list[Question] = []
        with open(questions_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("split", split) != split:
                    continue
                self.questions.append(
                    Question(
                        question_id=row["question_id"],
                        question=row["question"],
                        target_doc_ids=set(row["target_doc_ids"]),
                    )
                )
        logger.info("Loaded %d questions for split=%s", len(self.questions), split)

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Question:
        return self.questions[idx]


# ======================================================================
# Rollout
# ======================================================================

@dataclass
class Rollout:
    """Stores the result of one episode rollout."""
    token_ids: list[int] = field(default_factory=list)
    prompt_len: int = 0  # tokens in the initial prompt (not trained on)
    reward: float = 0.0
    advantage: float = 0.0
    ndcg: float = 0.0
    format_reward: float = 0.0
    num_turns: int = 0
    search_calls: int = 0
    read_calls: int = 0
    logprobs: list[float] = field(default_factory=list)


def run_episode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    env: RetrievalEnvironment,
    question: Question,
    max_tokens: int,
    max_new_tokens_per_turn: int = 512,
    device: str = "cuda",
) -> Rollout:
    """Run a single multi-turn episode with greedy/sampling generation.

    This is a simplified version for prototyping that uses HuggingFace
    generate() instead of vLLM.  For production training, replace this
    with vLLM-backed generation for much higher throughput.
    """
    tool_ctx = ToolContext(env.index)

    # Encode initial prompt (only place we use the chat template).
    prompt_ids = encode_initial_prompt(tokenizer, SYSTEM_PROMPT, question.question)
    all_token_ids = list(prompt_ids)
    prompt_len = len(prompt_ids)
    all_logprobs: list[float] = []

    model_outputs: list[str] = []

    for turn in range(env.max_turns):
        remaining = max_tokens - len(all_token_ids)
        if remaining <= 0:
            break

        gen_len = min(max_new_tokens_per_turn, remaining)

        input_ids = torch.tensor([all_token_ids], device=device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=gen_len,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_ids = output.sequences[0, len(all_token_ids):].tolist()
        if not new_ids:
            break

        # Compute per-token log-probs from scores.
        if output.scores:
            for step_idx, score_tensor in enumerate(output.scores):
                if step_idx >= len(new_ids):
                    break
                log_probs = torch.log_softmax(score_tensor[0], dim=-1)
                token_logprob = log_probs[new_ids[step_idx]].item()
                all_logprobs.append(token_logprob)

        # TI/TO: append raw token IDs directly.
        all_token_ids.extend(new_ids)

        # Decode only the new tokens for tool parsing.
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        model_outputs.append(new_text)

        # Execute environment step.
        tool_result, done = env.step(new_text, tool_ctx)

        if done:
            break

        if tool_result is not None:
            # Append tool result tokens (TI/TO – no retokenisation).
            result_ids = encode_tool_result(tokenizer, tool_result)
            all_token_ids.extend(result_ids)

    # Score the episode.
    result = env.score_episode(model_outputs, question.target_doc_ids, tool_ctx)

    return Rollout(
        token_ids=all_token_ids,
        prompt_len=prompt_len,
        reward=result.reward,
        ndcg=result.ndcg,
        format_reward=result.format_reward,
        num_turns=result.num_turns,
        search_calls=result.search_calls,
        read_calls=result.read_calls,
        logprobs=all_logprobs,
    )


# ======================================================================
# GRPO Training
# ======================================================================

def compute_group_advantages(
    rollouts: list[Rollout],
    group_size: int,
) -> None:
    """Compute per-group normalised advantages in-place.

    For each group of ``group_size`` rollouts (all for the same question):
        advantage_i = (reward_i - mean) / (std + eps)

    This is standard GRPO advantage normalisation.
    """
    eps = 1e-8
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start: start + group_size]
        rewards = np.array([r.reward for r in group])
        mean = rewards.mean()
        std = rewards.std()
        for r in group:
            r.advantage = (r.reward - mean) / (std + eps)


def grpo_loss(
    model: AutoModelForCausalLM,
    rollouts: list[Rollout],
    tokenizer: AutoTokenizer,
    clip_epsilon: float = 0.2,
    kl_coeff: float = 0.01,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute the clipped GRPO surrogate loss over a batch of rollouts.

    Uses **per-sequence** length normalisation (divide advantage by
    generated-token count), NOT per-group (Dr. GRPO), which the paper
    shows causes OOV sampling.
    """
    total_loss = torch.tensor(0.0, device=device)
    n_tokens = 0

    for rollout in rollouts:
        gen_ids = rollout.token_ids[rollout.prompt_len:]
        if not gen_ids:
            continue

        # Forward pass to get current log-probs.
        input_ids = torch.tensor([rollout.token_ids], device=device)
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids)
        logits = outputs.logits[0, rollout.prompt_len - 1: -1]  # align with gen tokens
        log_probs = torch.log_softmax(logits, dim=-1)

        gen_tensor = torch.tensor(gen_ids, device=device)
        current_logprobs = log_probs.gather(
            1, gen_tensor.unsqueeze(1),
        ).squeeze(1)

        # Old log-probs from generation (stored in rollout).
        old_logprobs = rollout.logprobs[: len(gen_ids)]
        if len(old_logprobs) < len(gen_ids):
            # Pad if we're missing some (shouldn't happen in practice).
            old_logprobs = old_logprobs + [0.0] * (len(gen_ids) - len(old_logprobs))
        old_logprobs_t = torch.tensor(old_logprobs, device=device)

        # Per-sequence length normalisation of advantage.
        per_token_adv = rollout.advantage / len(gen_ids)

        # Importance ratio.
        ratio = torch.exp(current_logprobs - old_logprobs_t)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        surrogate = torch.min(ratio * per_token_adv, clipped_ratio * per_token_adv)

        # KL penalty (approx: current - old log-probs).
        kl = current_logprobs - old_logprobs_t  # approximate KL

        token_loss = -(surrogate - kl_coeff * kl)
        total_loss = total_loss + token_loss.sum()
        n_tokens += len(gen_ids)

    if n_tokens > 0:
        total_loss = total_loss / n_tokens

    return total_loss


# ======================================================================
# Length scheduling
# ======================================================================

def get_max_tokens(step: int, config: dict) -> int:
    """Linearly ramp max_tokens from initial to final over warmup_steps."""
    ls = config.get("length_scheduling", {})
    initial = ls.get("initial_max_tokens", 2048)
    final = ls.get("final_max_tokens", 8192)
    warmup = ls.get("warmup_steps", 500)
    if step >= warmup:
        return final
    frac = step / warmup
    return int(initial + (final - initial) * frac)


# ======================================================================
# Main training loop
# ======================================================================

def train(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup.
    model_name = config["model"]["name"]
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()

    # Load environment.
    corpus_dir = Path(config["data"]["corpus_dir"])
    index = EmbeddingIndex(corpus_dir)
    env = RetrievalEnvironment(
        index=index,
        max_turns=config["environment"]["max_turns"],
        ndcg_weight=config["reward"]["ndcg_weight"],
        format_weight=config["reward"]["format_weight"],
    )

    # Load questions.
    questions_path = corpus_dir / "questions.jsonl"
    dataset = QuestionDataset(questions_path, split=config["data"]["train_split"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: batch,  # keep as list of Question
    )

    # Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    group_size = config["grpo"]["group_size"]
    clip_eps = config["grpo"]["clip_epsilon"]
    kl_coeff = config["grpo"]["kl_coeff"]
    max_grad_norm = config["training"]["max_grad_norm"]
    log_interval = config["training"].get("log_interval", 10)

    global_step = 0

    for epoch in range(config["training"]["epochs"]):
        logger.info("=== Epoch %d ===", epoch + 1)

        for batch_idx, batch in enumerate(dataloader):
            max_tokens = get_max_tokens(global_step, config)

            # ----------------------------------------------------------
            # 1. Generate rollouts.
            # ----------------------------------------------------------
            all_rollouts: list[Rollout] = []
            for question in batch:
                for _ in range(group_size):
                    rollout = run_episode(
                        model=model,
                        tokenizer=tokenizer,
                        env=env,
                        question=question,
                        max_tokens=max_tokens,
                        device="cuda",
                    )
                    all_rollouts.append(rollout)

            # ----------------------------------------------------------
            # 2. Compute advantages.
            # ----------------------------------------------------------
            compute_group_advantages(all_rollouts, group_size)

            # ----------------------------------------------------------
            # 3. Training step.
            # ----------------------------------------------------------
            model.train()
            optimizer.zero_grad()

            loss = grpo_loss(
                model=model,
                rollouts=all_rollouts,
                tokenizer=tokenizer,
                clip_epsilon=clip_eps,
                kl_coeff=kl_coeff,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            global_step += 1

            # ----------------------------------------------------------
            # 4. Logging.
            # ----------------------------------------------------------
            if global_step % log_interval == 0:
                avg_reward = np.mean([r.reward for r in all_rollouts])
                avg_ndcg = np.mean([r.ndcg for r in all_rollouts])
                avg_fmt = np.mean([r.format_reward for r in all_rollouts])
                avg_turns = np.mean([r.num_turns for r in all_rollouts])
                avg_search = np.mean([r.search_calls for r in all_rollouts])
                avg_read = np.mean([r.read_calls for r in all_rollouts])
                logger.info(
                    "step=%d  loss=%.4f  reward=%.3f  ndcg=%.3f  "
                    "fmt=%.3f  turns=%.1f  search=%.1f  read=%.1f  "
                    "max_tokens=%d",
                    global_step,
                    loss.item(),
                    avg_reward,
                    avg_ndcg,
                    avg_fmt,
                    avg_turns,
                    avg_search,
                    avg_read,
                    max_tokens,
                )

        # Save checkpoint at end of epoch.
        ckpt_dir = Path("checkpoints") / f"epoch_{epoch + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info("Saved checkpoint → %s", ckpt_dir)

    logger.info("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO for agentic retrieval")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    train(args.config)


if __name__ == "__main__":
    main()
