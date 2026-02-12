"""Build a unified corpus and question index from raw MuSiQue data.

The corpus pools *all* paragraphs across every question in the training set,
deduplicates them, and assigns each a stable ``doc_id``.  A separate questions
file records which ``doc_id``s are the supporting documents for each question.
Finally a FAISS vector index is built over the corpus using a lightweight
sentence-transformer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "corpus"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 512


def _paragraph_key(title: str, text: str) -> str:
    """Deterministic dedup key for a paragraph."""
    h = hashlib.sha256(f"{title}|||{text}".encode()).hexdigest()[:12]
    return h


def build_corpus(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OUT_DIR,
    splits: tuple[str, ...] = ("train", "dev"),
    embed_model_name: str = EMBED_MODEL_NAME,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect unique paragraphs across all requested splits.
    # ------------------------------------------------------------------
    corpus: dict[str, dict] = {}  # doc_id -> {title, text}
    questions: list[dict] = []

    for split in splits:
        path = raw_dir / f"musique_{split}.jsonl"
        if not path.exists():
            logger.warning("File %s not found – skipping split.", path)
            continue

        logger.info("Processing %s …", path)
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                target_ids: list[str] = []
                for para in row["paragraphs"]:
                    title = para.get("title", "")
                    text = para.get("paragraph_text", "")
                    doc_id = _paragraph_key(title, text)

                    if doc_id not in corpus:
                        corpus[doc_id] = {"title": title, "text": text}

                    if para.get("is_supporting", False):
                        target_ids.append(doc_id)

                questions.append(
                    {
                        "question_id": row["question_id"],
                        "question": row["question"],
                        "answer": row.get("answer", ""),
                        "target_doc_ids": target_ids,
                        "split": split,
                    }
                )

    logger.info(
        "Corpus size: %d unique paragraphs from %d questions.",
        len(corpus),
        len(questions),
    )

    # ------------------------------------------------------------------
    # 2. Write corpus JSONL (deterministic order by doc_id).
    # ------------------------------------------------------------------
    sorted_ids = sorted(corpus.keys())
    corpus_path = out_dir / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc_id in sorted_ids:
            rec = {"doc_id": doc_id, **corpus[doc_id]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote corpus → %s", corpus_path)

    # ------------------------------------------------------------------
    # 3. Write questions JSONL.
    # ------------------------------------------------------------------
    questions_path = out_dir / "questions.jsonl"
    with open(questions_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info("Wrote questions → %s", questions_path)

    # ------------------------------------------------------------------
    # 4. Build FAISS index.
    # ------------------------------------------------------------------
    logger.info("Loading embedding model: %s", embed_model_name)
    model = SentenceTransformer(embed_model_name)

    texts = [f"{corpus[did]['title']}: {corpus[did]['text']}" for did in sorted_ids]
    logger.info("Encoding %d paragraphs …", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on normalised vecs = cosine
    index.add(embeddings)

    index_path = out_dir / "corpus.faiss"
    faiss.write_index(index, str(index_path))
    logger.info("Wrote FAISS index → %s (%d vectors, dim=%d)", index_path, index.ntotal, dim)

    # Also save the ordered doc_id list so we can map FAISS row → doc_id.
    id_map_path = out_dir / "doc_id_map.json"
    with open(id_map_path, "w") as f:
        json.dump(sorted_ids, f)
    logger.info("Wrote doc_id map → %s", id_map_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified corpus + FAISS index")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        help="Which raw splits to include",
    )
    parser.add_argument("--embed-model", default=EMBED_MODEL_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_corpus(args.raw_dir, args.out_dir, tuple(args.splits), args.embed_model)


if __name__ == "__main__":
    main()
