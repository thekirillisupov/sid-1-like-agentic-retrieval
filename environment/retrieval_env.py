"""Multi-turn retrieval environment for agentic retrieval training.

The environment implements the interaction loop described in the SID-1 paper:
1. System prompt instructs the model to use search/read tools.
2. User turn provides the question.
3. Model interacts with tools over multiple turns.
4. Model calls report() to submit a ranked list of document IDs.
5. Environment scores the submission with NDCG + format reward.

**ID obfuscation**: each episode generates fresh random IDs so the model
cannot memorise corpus doc_ids across epochs.

**TI/TO**: all token sequences are kept as raw token IDs throughout the
episode – we never round-trip through message objects.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from environment.embedding_index import EmbeddingIndex
from environment.tools import ToolContext
from reward.format_reward import compute_format_reward
from reward.ndcg import compute_ndcg

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a retrieval agent. Given a question and search tools, find the documents \
needed to answer the question.

Available tools:
- search(query: str) -> returns relevant document snippets
- read(doc_id: str) -> returns full document text

When you are done, call report(doc_ids: list[str]) with the document IDs ordered \
by relevance (most relevant first).

Think step by step. You may need multiple searches to find all relevant documents.

To call a tool, output JSON on a single line:
{"tool": "<name>", "args": {<arguments>}}

Examples:
{"tool": "search", "args": {"query": "population of France"}}
{"tool": "read", "args": {"doc_id": "abc123"}}
{"tool": "report", "args": {"doc_ids": ["abc123", "def456"]}}
"""


@dataclass
class EpisodeResult:
    """Outcome of a single retrieval episode."""

    token_ids: list[int] = field(default_factory=list)
    reported_doc_ids: list[str] = field(default_factory=list)
    target_doc_ids: set[str] = field(default_factory=set)
    ndcg: float = 0.0
    format_reward: float = 0.0
    reward: float = 0.0
    num_turns: int = 0
    search_calls: int = 0
    read_calls: int = 0


@dataclass
class RetrievalEnvironment:
    """Multi-turn retrieval environment.

    Parameters
    ----------
    index:
        The shared :class:`EmbeddingIndex` backing search/read.
    max_turns:
        Maximum interaction turns before forcing termination.
    ndcg_weight:
        Weight for NDCG component of the combined reward.
    format_weight:
        Weight for format-adherence component of the combined reward.
    """

    index: EmbeddingIndex
    max_turns: int = 8
    ndcg_weight: float = 0.9
    format_weight: float = 0.1

    # ------------------------------------------------------------------
    # Episode runner (non-LLM – used for scoring given text output)
    # ------------------------------------------------------------------

    def score_episode(
        self,
        model_outputs: list[str],
        target_doc_ids: set[str],
        tool_context: ToolContext,
    ) -> EpisodeResult:
        """Run through a completed episode's model outputs and compute reward.

        This is used after vLLM generation: we already have the model's text
        and just need to parse tool calls, execute them, and score.

        Parameters
        ----------
        model_outputs:
            List of model-generated text strings, one per turn.
        target_doc_ids:
            Set of *real* (non-obfuscated) target document IDs.
        tool_context:
            The per-episode :class:`ToolContext` (provides obfuscation).
        """
        reported_obf_ids: list[str] = []
        all_tool_results: list[str] = []

        for turn_idx, output_text in enumerate(model_outputs):
            # Parse tool calls from the model output.
            tool_calls = self._parse_tool_calls(output_text)

            if not tool_calls:
                # No tool calls – possibly thinking or malformed output.
                continue

            for call in tool_calls:
                name = call.get("tool", "")
                args = call.get("args", {})

                if name == "search":
                    query = args.get("query", "")
                    results = tool_context.search(query)
                    all_tool_results.append(json.dumps(results))

                elif name == "read":
                    doc_id = args.get("doc_id", "")
                    result = tool_context.read(doc_id)
                    all_tool_results.append(json.dumps(result))

                elif name == "report":
                    reported_obf_ids = [
                        str(x) for x in args.get("doc_ids", [])
                    ]

        # Deobfuscate reported IDs for scoring.
        reported_real_ids = tool_context.deobfuscate_list(reported_obf_ids)

        # Compute rewards.
        ndcg = compute_ndcg(reported_real_ids, target_doc_ids)

        # Format reward: did the model produce a valid report?
        combined_output = "\n".join(model_outputs)
        fmt_reward, _ = compute_format_reward(combined_output)

        reward = self.ndcg_weight * ndcg + self.format_weight * fmt_reward

        return EpisodeResult(
            reported_doc_ids=reported_real_ids,
            target_doc_ids=target_doc_ids,
            ndcg=ndcg,
            format_reward=fmt_reward,
            reward=reward,
            num_turns=len(model_outputs),
            search_calls=tool_context.search_calls,
            read_calls=tool_context.read_calls,
        )

    # ------------------------------------------------------------------
    # Interactive episode (for inference / baselines)
    # ------------------------------------------------------------------

    def build_initial_messages(self, question: str) -> list[dict]:
        """Build the initial message list for an episode."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    def step(
        self,
        model_text: str,
        tool_context: ToolContext,
    ) -> tuple[str | None, bool]:
        """Execute one turn of interaction.

        Parse model_text for tool calls, execute them, and return:
        - tool_result_text: string to feed back as the next user message
          (None if no tool calls were found).
        - done: True if the model called report() (episode finished).
        """
        tool_calls = self._parse_tool_calls(model_text)

        if not tool_calls:
            return None, False

        results: list[str] = []
        done = False

        for call in tool_calls:
            name = call.get("tool", "")
            args = call.get("args", {})

            if name == "search":
                query = args.get("query", "")
                res = tool_context.search(query)
                results.append(f"[search results]\n{json.dumps(res, indent=2)}")

            elif name == "read":
                doc_id = args.get("doc_id", "")
                res = tool_context.read(doc_id)
                results.append(f"[read result]\n{json.dumps(res, indent=2)}")

            elif name == "report":
                done = True
                results.append("[report received]")

        return "\n\n".join(results), done

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict]:
        """Extract tool-call JSON objects from model output text."""
        calls: list[dict] = []
        # Match JSON objects containing a "tool" key.
        for m in re.finditer(r'\{[^{}]*"tool"\s*:[^{}]*\}', text, re.DOTALL):
            try:
                obj = json.loads(m.group(0))
                if "tool" in obj:
                    calls.append(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        return calls
