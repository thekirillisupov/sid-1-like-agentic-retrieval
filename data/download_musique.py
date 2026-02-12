"""Download and preprocess the MuSiQue multi-hop QA dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "raw"


def download_musique(out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Path]:
    """Download MuSiQue from HuggingFace and save train/dev splits as JSONL.

    Each output line contains:
        question_id, question, answer, paragraphs (list of
        {title, paragraph_text, is_supporting}), question_decomposition.

    Returns a mapping from split name to file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading MuSiQue dataset from HuggingFace …")
    ds = load_dataset("dgslibisey/MuSiQue")

    saved: dict[str, Path] = {}
    for split_name in ("train", "validation"):
        if split_name not in ds:
            logger.warning("Split %s not found – skipping.", split_name)
            continue

        out_name = "dev" if split_name == "validation" else split_name
        out_path = out_dir / f"musique_{out_name}.jsonl"

        with open(out_path, "w") as f:
            for row in ds[split_name]:
                # Normalise the record to a stable schema.
                paragraphs = row.get("paragraphs", [])
                # HF datasets may store sub-fields as parallel lists; handle
                # both dict-of-lists and list-of-dicts layouts.
                if isinstance(paragraphs, dict):
                    keys = list(paragraphs.keys())
                    n = len(paragraphs[keys[0]])
                    paragraphs = [
                        {k: paragraphs[k][i] for k in keys} for i in range(n)
                    ]

                record = {
                    "question_id": row.get("id", row.get("question_id", "")),
                    "question": row["question"],
                    "answer": row.get("answer", ""),
                    "paragraphs": paragraphs,
                    "question_decomposition": row.get(
                        "question_decomposition", []
                    ),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Wrote %s → %s", split_name, out_path)
        saved[out_name] = out_path

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MuSiQue dataset")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for raw JSONL files",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    download_musique(args.out_dir)


if __name__ == "__main__":
    main()
