"""Evaluation metrics for agentic retrieval.

Reports NDCG, recall, precision, F1, and operational statistics
(documents reported, turns used, tool calls per turn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reward.ndcg import compute_ndcg


@dataclass
class RetrievalMetrics:
    """Aggregated metrics over a set of evaluation episodes."""
    ndcg: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    avg_docs_reported: float = 0.0
    avg_turns: float = 0.0
    avg_tool_calls_per_turn: float = 0.0
    n_questions: int = 0

    def to_dict(self) -> dict:
        return {
            "ndcg": round(self.ndcg, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "avg_docs_reported": round(self.avg_docs_reported, 2),
            "avg_turns": round(self.avg_turns, 2),
            "avg_tool_calls_per_turn": round(self.avg_tool_calls_per_turn, 2),
            "n_questions": self.n_questions,
        }


def compute_recall(
    reported: list[str],
    target: set[str],
) -> float:
    """Fraction of target docs that appear in reported."""
    if not target:
        return 1.0
    return len(set(reported) & target) / len(target)


def compute_precision(
    reported: list[str],
    target: set[str],
) -> float:
    """Fraction of reported docs that are relevant."""
    if not reported:
        return 0.0
    return len(set(reported) & target) / len(reported)


def compute_f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision."""
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    For each document appearing across the lists:
        RRF_score = sum(1 / (k + rank_in_list_i))
    where rank is 1-indexed.

    Returns documents sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank_0, doc_id in enumerate(ranked):
            rank_1 = rank_0 + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_1)
    return sorted(scores, key=scores.get, reverse=True)


def aggregate_metrics(
    results: list[dict],
) -> RetrievalMetrics:
    """Aggregate per-question result dicts into overall metrics.

    Each dict in *results* should contain:
        reported_doc_ids, target_doc_ids, num_turns, search_calls, read_calls
    """
    if not results:
        return RetrievalMetrics()

    ndcgs, recalls, precisions, f1s = [], [], [], []
    docs_reported, turns, tool_calls_per_turn = [], [], []

    for r in results:
        reported = r["reported_doc_ids"]
        target = set(r["target_doc_ids"])

        ndcgs.append(compute_ndcg(reported, target))
        rec = compute_recall(reported, target)
        prec = compute_precision(reported, target)
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(compute_f1(rec, prec))
        docs_reported.append(len(reported))
        turns.append(r.get("num_turns", 0))

        total_calls = r.get("search_calls", 0) + r.get("read_calls", 0)
        n_turns = r.get("num_turns", 1) or 1
        tool_calls_per_turn.append(total_calls / n_turns)

    n = len(results)
    return RetrievalMetrics(
        ndcg=sum(ndcgs) / n,
        recall=sum(recalls) / n,
        precision=sum(precisions) / n,
        f1=sum(f1s) / n,
        avg_docs_reported=sum(docs_reported) / n,
        avg_turns=sum(turns) / n,
        avg_tool_calls_per_turn=sum(tool_calls_per_turn) / n,
        n_questions=n,
    )
