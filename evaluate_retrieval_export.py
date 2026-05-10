#!/usr/bin/env python3
"""Export RRF retrieval candidates for BGE rerank/evaluation on Colab."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

from src.pipeline_config import DEFAULT_RETRIEVAL_CONFIG
from src.retrieval_pipeline import documents_to_candidate_payload, retrieve_rrf_candidates


DEFAULT_GROUND_TRUTH = Path("ground_truth") / "grounth_truth_record_id.csv"


def _pick(row: Dict[str, Any], candidates: Sequence[str]) -> str:
    normalized = {key.strip().lower(): key for key in row.keys()}
    for candidate in candidates:
        key = normalized.get(candidate.strip().lower())
        if key:
            return str(row.get(key) or "").strip()
    return ""


def split_groundtruth(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    items: List[Dict[str, Any]] = []
    for row_num, row in enumerate(rows, start=2):
        row_id = _pick(row, ["ID", "id"]) or str(row_num - 1)
        question = _pick(row, ["Câu hỏi", "CÃ¢u há»i", "question"])
        answer = _pick(row, ["Answer", "answer"])
        groundtruth = split_groundtruth(_pick(row, ["Groundtruth", "groundtruth", "ground_truth"]))
        if not question or not groundtruth:
            raise ValueError(f"Invalid ground truth row {row_num}: missing question or Groundtruth")
        if any(item.upper() == "N/A" for item in groundtruth):
            raise ValueError(f"Invalid ground truth row {row_num}: Groundtruth contains N/A")
        items.append(
            {
                "id": row_id,
                "question": question,
                "answer": answer,
                "groundtruth": groundtruth,
            }
        )
    return items


def hit_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    retrieved_k = set(retrieved[:k])
    return 1.0 if any(item in retrieved_k for item in relevant) else 0.0


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(set(relevant) & retrieved_k) / len(set(relevant))


def export_candidates(
    ground_truth_path: Path,
    output_dir: Path,
    *,
    limit: int = 0,
) -> Dict[str, Any]:
    items = load_ground_truth(ground_truth_path)
    expected_rows = len(items)
    if limit:
        print(f"DEBUG: limiting evaluation export to first {limit} rows; this is not full evaluation.")
        items = items[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_jsonl = output_dir / "candidates.jsonl"
    detail_json = output_dir / "candidates_detail.json"
    summary_csv = output_dir / "candidate_recall_summary.csv"

    records = []
    per_item_metrics = []

    with candidates_jsonl.open("w", encoding="utf-8") as jsonl_file:
        for idx, item in enumerate(items, start=1):
            print(f"[{idx}/{len(items)}] retrieving candidates for row_id={item['id']}")
            docs, generated_queries = retrieve_rrf_candidates(item["question"], DEFAULT_RETRIEVAL_CONFIG)
            candidates = documents_to_candidate_payload(docs)
            retrieved_ids = [candidate.get("record_id") or candidate.get("id") for candidate in candidates]

            metrics = {
                "candidate_hit@100": hit_at_k(item["groundtruth"], retrieved_ids, DEFAULT_RETRIEVAL_CONFIG.rrf_top_k),
                "candidate_recall@100": recall_at_k(item["groundtruth"], retrieved_ids, DEFAULT_RETRIEVAL_CONFIG.rrf_top_k),
                "candidate_count": len(candidates),
            }
            per_item_metrics.append(metrics)

            record = {
                **item,
                "generated_queries": generated_queries,
                "candidates": candidates,
                "metrics": metrics,
            }
            records.append(record)
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ground_truth_path": str(ground_truth_path),
        "expected_rows_in_ground_truth": expected_rows,
        "evaluated_rows": len(items),
        "rrf_top_k": DEFAULT_RETRIEVAL_CONFIG.rrf_top_k,
        "candidate_hit@100": mean(item["candidate_hit@100"] for item in per_item_metrics) if per_item_metrics else 0.0,
        "candidate_recall@100": mean(item["candidate_recall@100"] for item in per_item_metrics) if per_item_metrics else 0.0,
        "avg_candidate_count": mean(item["candidate_count"] for item in per_item_metrics) if per_item_metrics else 0.0,
        "full_evaluation": limit == 0,
    }

    with detail_json.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": records}, f, ensure_ascii=False, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"\nWrote: {candidates_jsonl}")
    print(f"Wrote: {detail_json}")
    print(f"Wrote: {summary_csv}")
    print(f"Evaluated rows: {len(items)}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export retrieval candidates before BGE rerank.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to logs/retrieval_candidates_<timestamp>.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug only; do not use for full evaluation.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("logs") / f"retrieval_candidates_{timestamp}"
    export_candidates(args.ground_truth, output_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
