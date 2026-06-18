#!/usr/bin/env python3
"""Export RRF retrieval chunks for all ground-truth questions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configure this evaluator before importing src.config/src.indexer/src.pipeline_config.
# The slash in the HuggingFace model name cannot be used safely as a Windows filename,
# so this script is named retrieval_vietnamese_bi_encoder.py.
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DB_NAME = "legal"
COLLECTION_NAME = "legal_v2"
VECTOR_SEARCH_INDEX = "vector_index"
TEXT_SEARCH_INDEX = "default"
TEXT_SEARCH_FIELD = "text"
DEFAULT_TOP_K = 100

os.environ["DB_NAME"] = DB_NAME
os.environ["COLLECTION_NAME"] = COLLECTION_NAME
os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL
os.environ["MONGODB_VECTOR_SEARCH_INDEX"] = VECTOR_SEARCH_INDEX
os.environ["MONGODB_TEXT_SEARCH_INDEX"] = TEXT_SEARCH_INDEX
os.environ["MONGODB_SEARCH_INDEX"] = TEXT_SEARCH_INDEX
os.environ["MONGODB_TEXT_SEARCH_FIELD"] = TEXT_SEARCH_FIELD
os.environ["MONGODB_SEARCH_FIELD"] = TEXT_SEARCH_FIELD

from src.pipeline_config import DEFAULT_RETRIEVAL_CONFIG
from src.retrieval_pipeline import _get_indexer_cached, retrieve_rrf_candidates


DEFAULT_GROUND_TRUTH = REPO_ROOT / "ground_truth" / "ground_truth_by_vehicle.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluate" / "logs" / "vietnamese_bi_encoder_by_vehicle"


def _canonical_column(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(text.casefold().split())


def _pick_column(
    fieldnames: Sequence[str],
    aliases: Sequence[str],
    fallback_index: int,
) -> str:
    normalized = {_canonical_column(name): name for name in fieldnames}
    for alias in aliases:
        hit = normalized.get(_canonical_column(alias))
        if hit:
            return hit

    if len(fieldnames) > fallback_index:
        return fieldnames[fallback_index]

    raise ValueError(f"Cannot find CSV column from aliases={aliases!r}")


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _split_groundtruth_ids(value: str) -> List[str]:
    return [
        item.strip()
        for item in str(value or "").split(",")
        if item and item.strip()
    ]


def _normalize_record_id(value: Any) -> str:
    return str(value or "").strip().casefold()


def load_questions(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        fieldnames = list(reader.fieldnames)
        id_column = _pick_column(fieldnames, ["ID", "id"], 0)
        question_column = _pick_column(fieldnames, ["Câu hỏi", "Cau hoi", "question"], 1)
        groundtruth_column = _pick_column(
            fieldnames,
            ["Groundtruth", "groundtruth", "ground_truth"],
            3,
        )

        items: List[Dict[str, str]] = []
        for row_num, row in enumerate(reader, start=2):
            question = str(row.get(question_column) or "").strip()
            if not question:
                raise ValueError(f"Missing question at CSV row {row_num}")

            row_id = str(row.get(id_column) or row_num - 1).strip()
            groundtruth = str(row.get(groundtruth_column) or "").strip()
            items.append(
                {
                    "id": row_id,
                    "question": question,
                    "groundtruth": groundtruth,
                }
            )

    if not items:
        raise ValueError(f"No questions found in {path}")

    return items


def next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(path.name)
        for path in output_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_number = max(existing_numbers, default=0) + 1

    while True:
        run_dir = output_root / str(next_number)
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            next_number += 1


def _record_id_from_metadata(metadata: Dict[str, Any]) -> str:
    for key in ("record_id", "_id", "id", "doc_id", "chunk_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _source_rank_slots(
    source_ranks: Any,
    *,
    query_count: int,
    source_suffix: str,
) -> str:
    ranks: List[Any] = [None] * query_count
    if not isinstance(source_ranks, list):
        return "-".join("null" for _ in ranks)

    for item in source_ranks:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source") or "")
        rank = item.get("rank")
        if rank is None or not source.endswith(source_suffix):
            continue

        source_prefix = source[: -len(source_suffix)]
        if not source_prefix.startswith("q"):
            continue

        try:
            query_index = int(source_prefix[1:])
        except ValueError:
            continue

        if 0 <= query_index < query_count:
            ranks[query_index] = rank

    return "-".join("null" if rank is None else str(rank) for rank in ranks)


def chunk_payload(
    doc: Any,
    query_count: int,
    groundtruth_ids: Sequence[str],
) -> Dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source_ranks = metadata.get("source_ranks", [])
    record_id = _record_id_from_metadata(metadata)
    groundtruth_id_set = {_normalize_record_id(item) for item in groundtruth_ids}
    return {
        "record_id": record_id,
        "source_doc": metadata.get("source_doc", ""),
        "article": metadata.get("article", ""),
        "clause": metadata.get("clause", ""),
        "point": metadata.get("point", ""),
        "title": metadata.get("title", ""),
        "subjects": metadata.get("subjects", []),
        "topics": metadata.get("topics", []),
        "is_groundtruth": _normalize_record_id(record_id) in groundtruth_id_set,
        "text": str(getattr(doc, "page_content", "") or ""),
        "rrf_rank": metadata.get("rrf_rank"),
        "rrf_score": metadata.get("rrf_score"),
        "rerank_score": metadata.get("rerank_score") or metadata.get("bge_score"),
        "rerank_rank": metadata.get("rerank_rank") or metadata.get("bge_rank"),
        "vector_rank": _source_rank_slots(
            source_ranks,
            query_count=query_count,
            source_suffix="_vector",
        ),
        "bm25_rank": _source_rank_slots(
            source_ranks,
            query_count=query_count,
            source_suffix="_text",
        ),
    }


def record_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_id = ""
    for key in ("record_id", "_id", "id", "doc_id", "chunk_id"):
        value = record.get(key) or metadata.get(key)
        if value:
            record_id = str(value)
            break

    text = record.get("text") or record.get("page_content") or ""
    source_doc = record.get("source_doc") or metadata.get("source_doc", "")
    article = record.get("article") or metadata.get("article", "")
    clause = record.get("clause") or metadata.get("clause", "") or ""
    point = record.get("point") or metadata.get("point", "") or ""
    location = ", ".join(str(part) for part in [article, clause, point] if part)
    if text and (source_doc or location):
        text = f"{source_doc} - {location}:\n{text}".strip()

    return {
        "record_id": record_id,
        "source_doc": str(source_doc),
        "article": str(article),
        "clause": str(clause),
        "point": str(point),
        "title": str(record.get("title") or metadata.get("title", "")),
        "text": str(text),
    }


def load_groundtruth_chunks(
    groundtruth_ids: Sequence[str],
    retrieved_chunks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not groundtruth_ids:
        return []

    retrieved_by_id = {
        chunk["record_id"]: chunk
        for chunk in retrieved_chunks
        if chunk.get("record_id")
    }
    missing_ids = [
        record_id
        for record_id in groundtruth_ids
        if record_id not in retrieved_by_id
    ]

    fetched_by_id: Dict[str, Dict[str, str]] = {}
    if missing_ids:
        _, collection = _get_indexer_cached()
        records = collection.find(
            {
                "$or": [
                    {"record_id": {"$in": missing_ids}},
                    {"id": {"$in": missing_ids}},
                    {"doc_id": {"$in": missing_ids}},
                    {"chunk_id": {"$in": missing_ids}},
                    {"metadata.record_id": {"$in": missing_ids}},
                ]
            },
            {"embedding": 0},
        )
        for record in records:
            payload = record_payload(record)
            if payload["record_id"]:
                fetched_by_id[payload["record_id"]] = payload

    output: List[Dict[str, Any]] = []
    for record_id in groundtruth_ids:
        chunk = retrieved_by_id.get(record_id) or fetched_by_id.get(record_id)
        payload = dict(chunk) if chunk else {"record_id": record_id, "text": ""}
        payload["is_groundtruth"] = True
        output.append(payload)
    return output


def export_retrieved_chunks(
    ground_truth_path: Path,
    output_root: Path,
    *,
    limit: int = 0,
    top_k: int = DEFAULT_TOP_K,
) -> Path:
    questions = load_questions(ground_truth_path)
    if limit > 0:
        questions = questions[:limit]

    run_dir = next_run_dir(output_root)
    output_path = run_dir / "retrieved_chunks.json"
    records = []

    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] retrieving row_id={item['id']}")
        effective_config = DEFAULT_RETRIEVAL_CONFIG.with_overrides(
            rrf_top_k=top_k,
            rerank_top_n=top_k,
        )
        docs, generated_queries = retrieve_rrf_candidates(
            item["question"],
            effective_config,
        )
        groundtruth_ids = _split_groundtruth_ids(item.get("groundtruth", ""))
        chunks = [
            chunk_payload(doc, len(generated_queries), groundtruth_ids)
            for doc in docs
        ]
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "candidate_top_k": top_k,
                "groundtruth_ids": groundtruth_ids,
                "groundtruth_chunks": load_groundtruth_chunks(groundtruth_ids, chunks),
                "generated_queries": generated_queries,
                "chunks": chunks,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Legal RAG retrieval chunks from ground-truth questions."
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="Debug only; 0 means all rows.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="RRF candidate count per question.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(
        "Retrieval config: "
        f"db={DB_NAME}, collection={COLLECTION_NAME}, "
        f"embedding_model={EMBEDDING_MODEL}, "
        f"vector_index={VECTOR_SEARCH_INDEX}, text_index={TEXT_SEARCH_INDEX}"
    )
    output_path = export_retrieved_chunks(
        _resolve_repo_path(args.ground_truth),
        _resolve_repo_path(args.output_root),
        limit=args.limit,
        top_k=args.top_k,
    )
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
