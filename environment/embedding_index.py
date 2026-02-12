"""FAISS-backed embedding index for vector search over the corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "corpus"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingIndex:
    """Wraps a FAISS index + corpus for nearest-neighbour lookup."""

    def __init__(
        self,
        corpus_dir: Path = CORPUS_DIR,
        embed_model_name: str = EMBED_MODEL_NAME,
    ) -> None:
        self.corpus_dir = corpus_dir
        self._load_corpus()
        self._load_index()
        logger.info("Loading embedding model: %s", embed_model_name)
        self.embed_model = SentenceTransformer(embed_model_name)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_corpus(self) -> None:
        """Load corpus JSONL and doc_id map."""
        # doc_id ordered list (row index → doc_id)
        id_map_path = self.corpus_dir / "doc_id_map.json"
        with open(id_map_path) as f:
            self.doc_ids: list[str] = json.load(f)

        # Full corpus keyed by doc_id
        self.corpus: dict[str, dict] = {}
        corpus_path = self.corpus_dir / "corpus.jsonl"
        with open(corpus_path) as f:
            for line in f:
                rec = json.loads(line)
                self.corpus[rec["doc_id"]] = rec

        logger.info("Loaded corpus with %d documents.", len(self.corpus))

    def _load_index(self) -> None:
        """Load the FAISS index."""
        index_path = self.corpus_dir / "corpus.faiss"
        self.index = faiss.read_index(str(index_path))
        logger.info(
            "Loaded FAISS index: %d vectors, dim=%d",
            self.index.ntotal,
            self.index.d,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Embed *query* and return the *top_k* nearest corpus paragraphs.

        Returns a list of dicts with keys: ``doc_id``, ``title``,
        ``snippet`` (first 200 chars of text), ``score``.
        """
        vec = self.embed_model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(vec, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS sentinel for missing results
            doc_id = self.doc_ids[idx]
            doc = self.corpus[doc_id]
            results.append(
                {
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "snippet": doc["text"][:200],
                    "score": float(score),
                }
            )
        return results

    def read(self, doc_id: str) -> dict | None:
        """Return the full document for *doc_id*, or ``None``."""
        doc = self.corpus.get(doc_id)
        if doc is None:
            return None
        return {
            "doc_id": doc_id,
            "title": doc["title"],
            "text": doc["text"],
        }

    def get_all_doc_ids(self) -> list[str]:
        """Return the ordered list of all document IDs in the corpus."""
        return list(self.doc_ids)
