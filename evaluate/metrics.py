#!/usr/bin/env python3
"""Evaluate reranked retrieval chunks with offline metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH = REPO_ROOT / "ground_truth" / "grounth_truth_record_id.csv"
DEFAULT_INPUT = REPO_ROOT / "evaluate" / "logs" / "3" / "reranked_chunks.json"
DEFAULT_K_VALUES = [1, 3, 5, 10, 20]

METRIC_NAMES = [
    "hit@k",
    "recall@k",
    "precision@k",
    "f1@k",
    "map@k",
    "mrr@k",
    "ndcg@k",
    "context_precision@k",
    "context_recall@k",
    "context_entities_recall@k",
]


@dataclass(frozen=True)
class GroundTruthItem:
    row_id: str
    question: str
    answer: str
    relevant_ids: List[str]


@dataclass(frozen=True)
class Chunk:
    record_id: str
    text: str


@dataclass(frozen=True)
class RerankRecord:
    row_id: str
    question: str
    chunks: List[Chunk]


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _canonical_column(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    return "".join(text.casefold().split())


def _pick_column(fieldnames: Sequence[str], aliases: Sequence[str], fallback_index: int) -> str:
    normalized = {_canonical_column(name): name for name in fieldnames}
    for alias in aliases:
        hit = normalized.get(_canonical_column(alias))
        if hit:
            return hit

    if len(fieldnames) > fallback_index:
        return fieldnames[fallback_index]

    raise ValueError(f"Cannot find CSV column from aliases={aliases!r}")


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("\"'")
    return unicodedata.normalize("NFC", text).casefold()


def split_relevant_ids(value: Any) -> List[str]:
    parts = re.split(r"[,;|]+", str(value or ""))
    output: List[str] = []
    seen = set()
    for part in parts:
        item = normalize_id(part)
        if not item or item.upper() == "N/A" or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def load_ground_truth(path: Path) -> List[GroundTruthItem]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        fieldnames = list(reader.fieldnames)
        id_col = _pick_column(fieldnames, ["id"], 0)
        question_col = _pick_column(fieldnames, ["question", "cau hoi"], 1)
        answer_col = _pick_column(fieldnames, ["answer"], 2)
        gt_col = _pick_column(fieldnames, ["groundtruth", "ground_truth"], 3)

        items: List[GroundTruthItem] = []
        for row_num, row in enumerate(reader, start=2):
            question = str(row.get(question_col) or "").strip()
            relevant_ids = split_relevant_ids(row.get(gt_col))
            if not question or not relevant_ids:
                raise ValueError(f"Invalid ground-truth row {row_num}: missing question or Groundtruth")

            items.append(
                GroundTruthItem(
                    row_id=str(row.get(id_col) or row_num - 1).strip(),
                    question=question,
                    answer=str(row.get(answer_col) or "").strip(),
                    relevant_ids=relevant_ids,
                )
            )

    if not items:
        raise ValueError(f"No ground-truth rows found in {path}")
    return items


def _chunk_from_payload(payload: Dict[str, Any]) -> Chunk:
    return Chunk(
        record_id=normalize_id(payload.get("record_id") or payload.get("id")),
        text=str(payload.get("text") or payload.get("page_content") or ""),
    )


def load_reranked_chunks(path: Path) -> List[RerankRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        records = payload["results"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Rerank JSON must be a list or an object with a 'results' list")

    output: List[RerankRecord] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid rerank record at index {index}: expected object")

        raw_chunks = record.get("chunks") or record.get("candidates") or record.get("retrieved") or []
        chunks = [_chunk_from_payload(chunk) for chunk in raw_chunks if isinstance(chunk, dict)]
        output.append(
            RerankRecord(
                row_id=str(record.get("id") or record.get("row_id") or index).strip(),
                question=str(record.get("question") or "").strip(),
                chunks=chunks,
            )
        )

    return output


def unique_retrieved_ids(chunks: Sequence[Chunk]) -> List[str]:
    output: List[str] = []
    seen = set()
    for chunk in chunks:
        record_id = normalize_id(chunk.record_id)
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        output.append(record_id)
    return output


def hit_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return 1.0 if set(actual) & set(retrieved[:k]) else 0.0


def recall_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return len(set(actual) & set(retrieved[:k])) / len(set(actual)) if actual else 0.0


def precision_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return len(set(actual) & set(retrieved[:k])) / k if k > 0 else 0.0


def f1_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    precision = precision_at_k(actual, retrieved, k)
    recall = recall_at_k(actual, retrieved, k)
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def map_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    score = 0.0
    hits = 0
    for rank, record_id in enumerate(retrieved[:k], start=1):
        if record_id in actual_set:
            hits += 1
            score += hits / rank
    return score / min(len(actual_set), k)


def mrr_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    actual_set = set(actual)
    for rank, record_id in enumerate(retrieved[:k], start=1):
        if record_id in actual_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    dcg = 0.0
    for rank, record_id in enumerate(retrieved[:k], start=1):
        if record_id in actual_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def context_precision_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    actual_set = set(actual)
    score = 0.0
    hits = 0
    for rank, record_id in enumerate(retrieved[:k], start=1):
        if record_id in actual_set:
            hits += 1
            score += hits / rank
    return score / hits if hits else 0.0


def context_recall_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return recall_at_k(actual, retrieved, k)


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.replace("\u0111", "d")


def extract_entities(text: str) -> set[str]:
    folded = _ascii_fold(str(text or ""))
    entities: set[str] = set()

    patterns = [
        r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b",
        r"\b\d+(?:,\d+)?\s*(?:km/h|%|ngay|thang|nam|tuoi|diem|met|m)\b",
        r"\b(?:dieu|khoan|diem)\s+[a-z0-9]+\b",
        r"\b(?:nghi dinh|nd|nd-cp|qh|tt|qcvn)\s*[-_/]?\s*\d+(?:[-_/]\d+)?\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, folded, flags=re.IGNORECASE):
            entity = " ".join(str(match).split())
            if entity:
                entities.add(entity)

    legal_terms = [
        "bao hiem",
        "bien bao",
        "cao toc",
        "chat kich thich",
        "cho hang",
        "dang kiem",
        "di nguoc chieu",
        "den do",
        "giay phep lai xe",
        "mu bao hiem",
        "nong do con",
        "qua toc do",
        "tai nan giao thong",
        "tru diem",
        "tuoc quyen su dung",
        "vuot den do",
        "xe dap",
        "xe gan may",
        "xe may",
        "xe mo to",
        "xe o to",
    ]
    for term in legal_terms:
        if term in folded:
            entities.add(term)

    return entities


def context_entities_recall_at_k(answer: str, chunks: Sequence[Chunk], k: int) -> Optional[float]:
    answer_entities = extract_entities(answer)
    if not answer_entities:
        return None

    context_text = "\n".join(chunk.text for chunk in chunks[:k])
    context_entities = extract_entities(context_text)
    if not context_entities:
        return 0.0

    return len(answer_entities & context_entities) / len(answer_entities)


def first_hit_rank(actual: Sequence[str], retrieved: Sequence[str]) -> Optional[int]:
    actual_set = set(actual)
    for rank, record_id in enumerate(retrieved, start=1):
        if record_id in actual_set:
            return rank
    return None


def metrics_for_k(item: GroundTruthItem, chunks: Sequence[Chunk], k: int) -> Dict[str, Optional[float]]:
    retrieved = unique_retrieved_ids(chunks)
    actual = item.relevant_ids
    return {
        "hit@k": hit_at_k(actual, retrieved, k),
        "recall@k": recall_at_k(actual, retrieved, k),
        "precision@k": precision_at_k(actual, retrieved, k),
        "f1@k": f1_at_k(actual, retrieved, k),
        "map@k": map_at_k(actual, retrieved, k),
        "mrr@k": mrr_at_k(actual, retrieved, k),
        "ndcg@k": ndcg_at_k(actual, retrieved, k),
        "context_precision@k": context_precision_at_k(actual, retrieved, k),
        "context_recall@k": context_recall_at_k(actual, retrieved, k),
        "context_entities_recall@k": context_entities_recall_at_k(item.answer, chunks, k),
    }


def _prediction_indexes(records: Sequence[RerankRecord]) -> Tuple[Dict[str, RerankRecord], Dict[str, RerankRecord]]:
    by_id = {record.row_id: record for record in records if record.row_id}
    by_question = {record.question: record for record in records if record.question}
    return by_id, by_question


def evaluate_all_metrics(
    ground_truth: Sequence[GroundTruthItem],
    reranked_records: Sequence[RerankRecord],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    by_id, by_question = _prediction_indexes(reranked_records)
    results = []
    missing_predictions: List[str] = []

    for item in ground_truth:
        prediction = by_id.get(item.row_id) or by_question.get(item.question)
        chunks = prediction.chunks if prediction else []
        if prediction is None:
            missing_predictions.append(item.row_id)

        retrieved_ids = unique_retrieved_ids(chunks)
        metrics = {f"@{k}": metrics_for_k(item, chunks, k) for k in k_values}
        results.append(
            {
                "id": item.row_id,
                "question": item.question,
                "groundtruth": item.relevant_ids,
                "retrieved_ids": retrieved_ids,
                "candidate_count": len(retrieved_ids),
                "first_hit_rank": first_hit_rank(item.relevant_ids, retrieved_ids),
                "metrics": metrics,
            }
        )

    summary: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        key = f"@{k}"
        summary[key] = {
            "evaluated_rows": len(results),
            "avg_candidate_count": mean(item["candidate_count"] for item in results) if results else 0.0,
        }
        for metric in METRIC_NAMES:
            values = [
                item["metrics"][key][metric]
                for item in results
                if item["metrics"][key][metric] is not None
            ]
            summary[key][metric] = mean(values) if values else 0.0

    return {
        "k_values": list(k_values),
        "evaluated_rows": len(results),
        "reranked_records": len(reranked_records),
        "missing_predictions": missing_predictions,
        "summary": summary,
        "results": results,
    }


def write_outputs(report: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.csv"
    detail_csv_path = output_dir / "metrics_detail.csv"
    detail_json_path = output_dir / "metrics_detail.json"

    with detail_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["k", "evaluated_rows", "avg_candidate_count", *METRIC_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in report["k_values"]:
            row = {"k": k, **report["summary"][f"@{k}"]}
            writer.writerow(_format_csv_row(row, fieldnames))

    with detail_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "question",
            "k",
            "candidate_count",
            "first_hit_rank",
            "groundtruth",
            "retrieved_ids",
            *METRIC_NAMES,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in report["results"]:
            for k in report["k_values"]:
                row = {
                    "id": item["id"],
                    "question": item["question"],
                    "k": k,
                    "candidate_count": item["candidate_count"],
                    "first_hit_rank": item["first_hit_rank"],
                    "groundtruth": ", ".join(item["groundtruth"]),
                    "retrieved_ids": ", ".join(item["retrieved_ids"][:k]),
                    **item["metrics"][f"@{k}"],
                }
                writer.writerow(_format_csv_row(row, fieldnames))

    return summary_path, detail_csv_path, detail_json_path


def _format_csv_row(row: Dict[str, Any], fieldnames: Sequence[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for field in fieldnames:
        value = row.get(field)
        if value is None:
            output[field] = ""
        elif isinstance(value, float):
            output[field] = f"{value:.6f}"
        else:
            output[field] = value
    return output


def parse_k_values(values: Sequence[int]) -> List[int]:
    parsed = sorted({int(value) for value in values})
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("All k values must be positive integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate reranked Legal RAG chunks offline.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ground_truth_path = _resolve_repo_path(args.ground_truth)
    input_path = _resolve_repo_path(args.input)
    output_dir = _resolve_repo_path(args.output_dir) if args.output_dir else input_path.parent
    k_values = parse_k_values(args.k)

    ground_truth = load_ground_truth(ground_truth_path)
    reranked_records = load_reranked_chunks(input_path)
    report = evaluate_all_metrics(ground_truth, reranked_records, k_values)
    report["ground_truth_path"] = str(ground_truth_path)
    report["input_path"] = str(input_path)

    summary_path, detail_csv_path, detail_json_path = write_outputs(report, output_dir)
    print(f"evaluated_rows={report['evaluated_rows']}")
    print(f"reranked_records={report['reranked_records']}")
    if report["missing_predictions"]:
        print(f"missing_predictions={len(report['missing_predictions'])}", file=sys.stderr)
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {detail_csv_path}")
    print(f"Wrote: {detail_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
