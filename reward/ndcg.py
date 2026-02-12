"""NDCG reward computation for agentic retrieval.

Implements the binary-relevance NDCG used in the SID-1 paper:
    DCG  = sum_{i=1}^{n} 1[l_i in T] / log2(i + 1)
    IDCG = sum_{i=1}^{|T|} 1 / log2(i + 1)
    NDCG = DCG / IDCG
"""

from __future__ import annotations

import math


def compute_ndcg(
    reported_doc_ids: list[str],
    target_doc_ids: set[str],
) -> float:
    """Compute NDCG with binary relevance.

    Parameters
    ----------
    reported_doc_ids:
        Ordered list of document IDs reported by the model (most relevant
        first).
    target_doc_ids:
        Set of ground-truth relevant document IDs.

    Returns
    -------
    float in [0, 1].
    """
    if not target_doc_ids:
        return 1.0  # no targets → trivially correct

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(reported_doc_ids):
        if doc_id in target_doc_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed

    # IDCG – perfect ranking places all |T| relevant docs at top positions
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(target_doc_ids)))

    return dcg / idcg if idcg > 0 else 0.0
