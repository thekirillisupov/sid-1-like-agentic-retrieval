"""Format-adherence reward.

Returns 1.0 when the model produces a valid ``report()`` call with a
non-empty list of document IDs, 0.0 otherwise.

The SID-1 paper found that omitting a format reward leads to format
regression during RL training, so we include it from the start.
"""

from __future__ import annotations

import json
import re


def compute_format_reward(model_output: str) -> tuple[float, list[str]]:
    """Score format adherence and extract reported doc IDs.

    The expected format is a tool call of the form::

        report(doc_ids=["id1", "id2", ...])

    We also accept JSON-style variants for robustness::

        {"tool": "report", "doc_ids": ["id1", "id2"]}

    Parameters
    ----------
    model_output:
        The raw text produced by the model in its final turn (or the full
        conversation transcript – the function searches for the *last*
        report call).

    Returns
    -------
    (reward, doc_ids):
        reward is 1.0 if a valid report was found, else 0.0.
        doc_ids is the extracted list (empty if reward is 0.0).
    """
    doc_ids = _extract_report_ids(model_output)
    if doc_ids:
        return 1.0, doc_ids
    return 0.0, []


def _extract_report_ids(text: str) -> list[str]:
    """Try several patterns to pull out the reported doc ID list."""

    # Pattern 1: function-call style  report(doc_ids=[...])
    # Allow optional quotes, whitespace, etc.
    match = re.search(
        r'report\s*\(\s*doc_ids\s*=\s*(\[.*?\])\s*\)',
        text,
        re.DOTALL,
    )
    if match:
        return _parse_id_list(match.group(1))

    # Pattern 2: JSON tool-call style  {"tool": "report", "doc_ids": [...]}
    for m in re.finditer(r'\{[^{}]*"tool"\s*:\s*"report"[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if "doc_ids" in obj and isinstance(obj["doc_ids"], list):
                return [str(x) for x in obj["doc_ids"]]
        except (json.JSONDecodeError, TypeError):
            continue

    # Pattern 3: bare JSON list right after the word "report"
    match = re.search(r'report[:\s]+(\[[^\]]*\])', text, re.DOTALL)
    if match:
        return _parse_id_list(match.group(1))

    return []


def _parse_id_list(raw: str) -> list[str]:
    """Parse a JSON array string into a list of string IDs."""
    try:
        arr = json.loads(raw)
        if isinstance(arr, list) and all(isinstance(x, (str, int)) for x in arr):
            return [str(x) for x in arr]
    except (json.JSONDecodeError, TypeError):
        pass
    return []
