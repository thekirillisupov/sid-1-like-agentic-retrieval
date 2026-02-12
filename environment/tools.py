"""Search and read tool implementations for the retrieval environment.

These functions wrap the :class:`EmbeddingIndex` and add **ID obfuscation**
so that every episode uses random document identifiers, preventing the
model from memorising real corpus IDs across training epochs.
"""

from __future__ import annotations

import random
import string
from typing import Optional

from environment.embedding_index import EmbeddingIndex


def _random_id(length: int = 6) -> str:
    """Generate a random alphanumeric ID."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


class ToolContext:
    """Per-episode tool context with ID obfuscation.

    Create a new ``ToolContext`` for every rollout episode.  It manages:
    - A random mapping from real ``doc_id`` → obfuscated ``doc_id``.
    - Reverse mapping for scoring.
    - Tool call counting for metrics.
    """

    def __init__(self, index: EmbeddingIndex) -> None:
        self.index = index
        # Obfuscation maps (lazily populated on first encounter).
        self._real_to_obf: dict[str, str] = {}
        self._obf_to_real: dict[str, str] = {}
        # Counters
        self.search_calls: int = 0
        self.read_calls: int = 0

    # ------------------------------------------------------------------
    # Obfuscation helpers
    # ------------------------------------------------------------------

    def _obfuscate(self, real_id: str) -> str:
        """Return (or create) the obfuscated ID for *real_id*."""
        if real_id not in self._real_to_obf:
            while True:
                obf = _random_id()
                if obf not in self._obf_to_real:
                    break
            self._real_to_obf[real_id] = obf
            self._obf_to_real[obf] = real_id
        return self._real_to_obf[real_id]

    def deobfuscate(self, obf_id: str) -> Optional[str]:
        """Map an obfuscated ID back to the real corpus ID."""
        return self._obf_to_real.get(obf_id)

    def deobfuscate_list(self, obf_ids: list[str]) -> list[str]:
        """Deobfuscate a list of IDs, dropping unknowns."""
        result: list[str] = []
        for oid in obf_ids:
            real = self.deobfuscate(oid)
            if real is not None:
                result.append(real)
        return result

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Embed *query* and return the *top_k* nearest snippets.

        Returns obfuscated ``doc_id`` values.
        """
        self.search_calls += 1
        raw_results = self.index.search(query, top_k=top_k)
        obfuscated: list[dict] = []
        for r in raw_results:
            obfuscated.append(
                {
                    "doc_id": self._obfuscate(r["doc_id"]),
                    "title": r["title"],
                    "snippet": r["snippet"],
                }
            )
        return obfuscated

    def read(self, doc_id: str) -> dict | None:
        """Return full text for the given *obfuscated* ``doc_id``."""
        self.read_calls += 1
        real_id = self.deobfuscate(doc_id)
        if real_id is None:
            return {"error": f"Unknown doc_id: {doc_id}"}
        result = self.index.read(real_id)
        if result is None:
            return {"error": f"Document not found: {doc_id}"}
        return {
            "doc_id": doc_id,  # keep obfuscated
            "title": result["title"],
            "text": result["text"],
        }
