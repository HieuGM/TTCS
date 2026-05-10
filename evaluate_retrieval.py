#!/usr/bin/env python3
"""
Evaluate Legal RAG retrieval against a ground-truth CSV.

Default behavior is safe: the script only computes metrics from a predictions
JSON file. Use --run-retriever when you intentionally want to call the live
retriever, which may use MongoDB and paid LLM/rerank APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_GROUND_TRUTH = Path("ground_truth") / "grounth_truth.csv"
DEFAULT_K_VALUES = [1, 3, 5, 10]


@dataclass
class GroundTruthItem:
    row_id: str
    question: str
    answer: str
    relevant_ids: List[str]


@dataclass
class RetrievedItem:
    doc_id: str
    text: str = ""
    metadata: Optional[Dict[str, Any]] = None


def normalize_id(value: Any) -> str:
    """Return a stable string id from ObjectId/string-like values."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    object_id_match = re.search(r"[a-fA-F0-9]{24}", text)
    if object_id_match:
        return object_id_match.group(0).lower()

    return text.strip().strip("\"'").lower()


def split_relevant_ids(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    parts = re.split(r"[,;|\s]+", text)
    seen = set()
    ids: List[str] = []
    for part in parts:
        doc_id = normalize_id(part)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
    return ids


def pick_column(fieldnames: Sequence[str], aliases: Sequence[str], fallback_index: int) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for alias in aliases:
        hit = normalized.get(alias.strip().lower())
        if hit:
            return hit
    if len(fieldnames) > fallback_index:
        return fieldnames[fallback_index]
    raise ValueError(f"Cannot find column from aliases={aliases!r}")


def load_ground_truth(path: Path) -> List[GroundTruthItem]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        fieldnames = list(reader.fieldnames)
        id_col = pick_column(fieldnames, ["id"], 0)
        question_col = pick_column(fieldnames, ["question", "cau hoi"], 1)
        answer_col = pick_column(fieldnames, ["answer"], 2)
        gt_col = pick_column(fieldnames, ["groundtruth", "ground_truth", "chunk", "chunks"], 3)

        items: List[GroundTruthItem] = []
        for row_num, row in enumerate(reader, start=2):
            question = (row.get(question_col) or "").strip()
            relevant_ids = split_relevant_ids(row.get(gt_col))
            if not question or not relevant_ids:
                print(f"WARNING: skip row {row_num}: missing question or Groundtruth", file=sys.stderr)
                continue
            items.append(
                GroundTruthItem(
                    row_id=(row.get(id_col) or str(row_num)).strip(),
                    question=question,
                    answer=(row.get(answer_col) or "").strip(),
                    relevant_ids=relevant_ids,
                )
            )

    if not items:
        raise ValueError(f"No valid ground-truth rows found in {path}")
    return items


def doc_to_retrieved_item(doc: Any, id_field: str = "auto") -> RetrievedItem:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    text = str(getattr(doc, "page_content", "") or "")

    if id_field != "auto":
        id_candidates = [metadata.get(id_field), getattr(doc, id_field, None)]
    else:
        id_candidates = [
            metadata.get("_id"),
            metadata.get("id"),
            metadata.get("doc_id"),
            metadata.get("chunk_id"),
            metadata.get("record_id"),
            getattr(doc, "id", None),
        ]

    doc_id = ""
    for candidate in id_candidates:
        doc_id = normalize_id(candidate)
        if doc_id:
            break

    return RetrievedItem(doc_id=doc_id, text=text, metadata=metadata)


def load_predictions(path: Path) -> Dict[str, List[RetrievedItem]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload:
        records = payload["results"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Predictions JSON must be a list or an object with a 'results' list")

    predictions: Dict[str, List[RetrievedItem]] = {}
    for record in records:
        row_id = str(record.get("id") or record.get("row_id") or "").strip()
        question = str(record.get("question") or "").strip()
        key = row_id or question
        retrieved_raw = record.get("retrieved") or record.get("contexts") or []

        retrieved: List[RetrievedItem] = []
        for item in retrieved_raw:
            if isinstance(item, str):
                retrieved.append(RetrievedItem(doc_id=normalize_id(item)))
                continue

            metadata = item.get("metadata") or {}
            doc_id = (
                normalize_id(item.get("id"))
                or normalize_id(item.get("doc_id"))
                or normalize_id(item.get("_id"))
                or normalize_id(metadata.get("_id"))
                or normalize_id(metadata.get("record_id"))
            )
            text = str(item.get("text") or item.get("page_content") or item.get("content") or "")
            retrieved.append(RetrievedItem(doc_id=doc_id, text=text, metadata=metadata))

        if key:
            predictions[key] = retrieved

    return predictions


def run_live_retriever(
    items: Sequence[GroundTruthItem],
    top_k: int,
    top_n: int,
    id_field: str = "auto",
) -> Dict[str, List[RetrievedItem]]:
    from src.retrieval_pipeline import retrieve_and_rerank

    predictions: Dict[str, List[RetrievedItem]] = {}
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] retrieving row_id={item.row_id}")
        docs = retrieve_and_rerank(item.question, top_k=top_k, top_n=top_n)
        predictions[item.row_id] = [doc_to_retrieved_item(doc, id_field=id_field) for doc in docs]
    return predictions


def run_single_query_retriever(
    items: Sequence[GroundTruthItem],
    top_k: int,
    top_n: int,
    use_rerank: bool,
    id_field: str = "auto",
) -> Dict[str, List[RetrievedItem]]:
    """Run retrieval with query generation disabled."""
    from src.pipeline_config import DEFAULT_RETRIEVAL_CONFIG
    from src.retrieval_pipeline import bge_rerank, retrieve_rrf_candidates

    config = DEFAULT_RETRIEVAL_CONFIG.with_overrides(
        enable_query_generation=False,
        rrf_top_k=top_k,
        rerank_top_n=top_n,
    )

    predictions: Dict[str, List[RetrievedItem]] = {}
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] retrieving row_id={item.row_id}")
        docs, _ = retrieve_rrf_candidates(item.question, config)

        if use_rerank:
            final_docs = bge_rerank(item.question, docs, config)
        else:
            final_docs = docs[:top_n]

        predictions[item.row_id] = [doc_to_retrieved_item(doc, id_field=id_field) for doc in final_docs]
    return predictions


def unique_top_ids(retrieved: Sequence[RetrievedItem], k: int) -> List[str]:
    ids: List[str] = []
    seen = set()
    for item in retrieved:
        doc_id = normalize_id(item.doc_id)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ids.append(doc_id)
        if len(ids) >= k:
            break
    return ids


def binary_relevance(pred_ids: Sequence[str], relevant_ids: Iterable[str]) -> List[int]:
    relevant = set(relevant_ids)
    return [1 if doc_id in relevant else 0 for doc_id in pred_ids]


def average_precision_at_k(rel: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    score = 0.0
    hits = 0
    for rank, is_rel in enumerate(rel[:k], start=1):
        if is_rel:
            hits += 1
            score += hits / rank
    return score / min(total_relevant, k)


def reciprocal_rank_at_k(rel: Sequence[int], k: int) -> float:
    for rank, is_rel in enumerate(rel[:k], start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(rel: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    dcg = sum(is_rel / math.log2(rank + 1) for rank, is_rel in enumerate(rel[:k], start=1))
    ideal_hits = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def context_precision_at_k(rel: Sequence[int], k: int) -> float:
    """RAGAS-style rank-aware precision over retrieved relevant contexts."""
    score = 0.0
    hits = 0
    for rank, is_rel in enumerate(rel[:k], start=1):
        if is_rel:
            hits += 1
            score += hits / rank
    return score / hits if hits else 0.0


def extract_entities(text: str) -> set[str]:
    """Heuristic non-LLM entity extraction for legal RAG context metrics."""
    if not text:
        return set()

    lowered = text.casefold()
    entities: set[str] = set()

    patterns = [
        r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b",
        r"\b\d+(?:,\d+)?\s*(?:km/h|%)\b",
        r"\b\d+\s*(?:ngay|thang|nam|tuoi|diem)\b",
        r"\b(?:dieu|khoan|diem)\s+[a-z0-9]+\b",
        r"\b(?:nd|nd-cp|qh|tt|qcvn)\s*[-_/]?\s*\d+(?:[-_/]\d+)?\b",
        r"\b(?:oto|o to|xe may|xe mo to|xe gan may|xe dap|nguoi lai xe|giay phep lai xe)\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, lowered, flags=re.IGNORECASE):
            entity = " ".join(str(match).split())
            if entity:
                entities.add(entity)

    # Preserve common Vietnamese terms if the source file is correctly encoded.
    vietnamese_terms = [
        "o to",
        "xe may",
        "xe mo to",
        "xe gan may",
        "xe dap",
        "nguoi lai xe",
        "giay phep lai xe",
        "mu bao hiem",
        "vuot den do",
        "nong do con",
    ]
    ascii_text = remove_accents(lowered)
    for term in vietnamese_terms:
        if term in ascii_text:
            entities.add(term)

    return entities


def remove_accents(text: str) -> str:
    replacements = {
        "à": "a",
        "á": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ằ": "a",
        "ắ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ầ": "a",
        "ấ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "đ": "d",
        "è": "e",
        "é": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ề": "e",
        "ế": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "ì": "i",
        "í": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ò": "o",
        "ó": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ồ": "o",
        "ố": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ờ": "o",
        "ớ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ù": "u",
        "ú": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ừ": "u",
        "ứ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ỳ": "y",
        "ý": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def context_entities_recall(answer: str, retrieved: Sequence[RetrievedItem], k: int) -> Optional[float]:
    answer_entities = extract_entities(answer)
    if not answer_entities:
        return None

    context_text = "\n".join(item.text for item in retrieved[:k])
    context_entities = extract_entities(context_text)
    if not context_entities:
        return 0.0

    return len(answer_entities & context_entities) / len(answer_entities)


def metrics_for_item(item: GroundTruthItem, retrieved: Sequence[RetrievedItem], k_values: Sequence[int]) -> Dict[int, Dict[str, Optional[float]]]:
    result: Dict[int, Dict[str, Optional[float]]] = {}
    relevant_set = set(item.relevant_ids)
    total_relevant = len(relevant_set)

    for k in k_values:
        pred_ids = unique_top_ids(retrieved, k)
        rel = binary_relevance(pred_ids, relevant_set)
        hits = sum(rel)
        precision = hits / k if k else 0.0
        recall = hits / total_relevant if total_relevant else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        result[k] = {
            "hit": 1.0 if hits > 0 else 0.0,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "map": average_precision_at_k(rel, total_relevant, k),
            "mrr": reciprocal_rank_at_k(rel, k),
            "ndcg": ndcg_at_k(rel, total_relevant, k),
            "context_precision": context_precision_at_k(rel, k),
            "context_recall": recall,
            "context_entities_recall": context_entities_recall(item.answer, retrieved, k),
        }

    return result


def aggregate(per_item: Sequence[Dict[int, Dict[str, Optional[float]]]], k_values: Sequence[int]) -> Dict[str, Dict[str, float]]:
    metric_names = [
        "hit",
        "recall",
        "precision",
        "f1",
        "map",
        "mrr",
        "ndcg",
        "context_precision",
        "context_recall",
        "context_entities_recall",
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        summary[f"@{k}"] = {}
        for metric in metric_names:
            values = [item[k][metric] for item in per_item if item[k][metric] is not None]
            summary[f"@{k}"][metric] = mean(values) if values else 0.0
    return summary


def write_outputs(
    output_dir: Path,
    items: Sequence[GroundTruthItem],
    predictions: Dict[str, List[RetrievedItem]],
    per_item_metrics: Sequence[Dict[int, Dict[str, Optional[float]]]],
    summary: Dict[str, Dict[str, float]],
    k_values: Sequence[int],
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_json = output_dir / "retrieval_eval_detail.json"
    summary_csv = output_dir / "retrieval_eval_summary.csv"
    detail_csv = output_dir / "retrieval_eval_detail.csv"

    detail_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "k_values": list(k_values),
        "summary": summary,
        "results": [],
    }

    for item, metric_map in zip(items, per_item_metrics):
        retrieved = predictions.get(item.row_id) or predictions.get(item.question) or []
        detail_payload["results"].append(
            {
                "id": item.row_id,
                "question": item.question,
                "groundtruth": item.relevant_ids,
                "retrieved": [
                    {
                        "id": r.doc_id,
                        "text": r.text,
                        "metadata": r.metadata or {},
                    }
                    for r in retrieved
                ],
                "metrics": {f"@{k}": metric_map[k] for k in k_values},
            }
        )

    with detail_json.open("w", encoding="utf-8") as f:
        json.dump(detail_payload, f, ensure_ascii=False, indent=2)

    metric_names = list(next(iter(summary.values())).keys()) if summary else []
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", *metric_names])
        for k in k_values:
            writer.writerow([k, *[f"{summary[f'@{k}'][m]:.6f}" for m in metric_names]])

    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "question", "k", *metric_names, "groundtruth", "retrieved_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item, metric_map in zip(items, per_item_metrics):
            retrieved = predictions.get(item.row_id) or predictions.get(item.question) or []
            for k in k_values:
                row = {
                    "id": item.row_id,
                    "question": item.question,
                    "k": k,
                    "groundtruth": ", ".join(item.relevant_ids),
                    "retrieved_ids": ", ".join(unique_top_ids(retrieved, k)),
                }
                for metric in metric_names:
                    value = metric_map[k][metric]
                    row[metric] = "" if value is None else f"{value:.6f}"
                writer.writerow(row)

    return summary_csv, detail_csv, detail_json


def parse_k_values(raw: str) -> List[int]:
    values = []
    for part in re.split(r"[,;\s]+", raw.strip()):
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("k values must be positive")
        values.append(value)
    if not values:
        raise ValueError("At least one k value is required")
    return sorted(set(values))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics for Legal RAG.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--predictions", type=Path, help="JSON predictions from a previous run.")
    parser.add_argument("--run-retriever", action="store_true", help="Call src.retrieval_pipeline.retrieve_and_rerank.")
    parser.add_argument(
        "--no-generate-queries",
        action="store_true",
        help="When used with --run-retriever, retrieve only with the original question.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="When used with --no-generate-queries, evaluate vector+Atlas-BM25+RRF retrieval without BGE rerank.",
    )
    parser.add_argument("--k", default=",".join(str(k) for k in DEFAULT_K_VALUES), help="Comma/space separated k values.")
    parser.add_argument("--retriever-top-k", type=int, default=20, help="Candidate k passed to the retriever.")
    parser.add_argument(
        "--id-field",
        default="auto",
        help="Retrieved document metadata field to evaluate against, e.g. record_id or _id.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows.")
    parser.add_argument("--output-dir", type=Path, help="Directory for summary/detail outputs.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    k_values = parse_k_values(args.k)
    max_k = max(k_values)

    items = load_ground_truth(args.ground_truth)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    if args.predictions:
        predictions = load_predictions(args.predictions)
    elif args.run_retriever:
        top_k = max(args.retriever_top_k, max_k)
        if args.no_generate_queries:
            predictions = run_single_query_retriever(
                items,
                top_k=top_k,
                top_n=max_k,
                use_rerank=not args.no_rerank,
                id_field=args.id_field,
            )
        else:
            predictions = run_live_retriever(items, top_k=top_k, top_n=max_k, id_field=args.id_field)
    else:
        print(
            "No predictions supplied. Pass --predictions FILE to compute metrics only, "
            "or --run-retriever to call the live retriever.",
            file=sys.stderr,
        )
        return 2

    missing = [item.row_id for item in items if item.row_id not in predictions and item.question not in predictions]
    if missing:
        print(f"WARNING: {len(missing)} rows have no predictions. They will score as 0.", file=sys.stderr)

    per_item_metrics = []
    for item in items:
        retrieved = predictions.get(item.row_id) or predictions.get(item.question) or []
        per_item_metrics.append(metrics_for_item(item, retrieved, k_values))

    summary = aggregate(per_item_metrics, k_values)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path("logs") / f"retrieval_eval_{timestamp}")
    summary_csv, detail_csv, detail_json = write_outputs(
        output_dir=output_dir,
        items=items,
        predictions=predictions,
        per_item_metrics=per_item_metrics,
        summary=summary,
        k_values=k_values,
    )

    print("\nSummary")
    metric_order = [
        "hit",
        "recall",
        "precision",
        "f1",
        "map",
        "mrr",
        "ndcg",
        "context_precision",
        "context_recall",
        "context_entities_recall",
    ]
    print("k," + ",".join(metric_order))
    for k in k_values:
        values = [f"{summary[f'@{k}'][metric]:.6f}" for metric in metric_order]
        print(f"{k}," + ",".join(values))

    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {detail_csv}")
    print(f"Wrote: {detail_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
