#!/usr/bin/env python3
"""Evaluate retrieval metrics from an already-reranked JSON file.

This script does not call MongoDB, rerankers, embeddings, or LLMs. It only
compares each record's `groundtruth` ids with ordered `candidates.record_id`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


DEFAULT_INPUT = Path("rerank_chunk/reranked_legal_results.json")
DEFAULT_K_VALUES = [1, 3, 5, 10]


def unique_ids(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    output = []
    for item in candidates or []:
        candidate_id = str(item.get("record_id") or item.get("id") or "").strip()
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            output.append(candidate_id)
    return output


def hit_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return 1.0 if set(actual) & set(retrieved[:k]) else 0.0


def precision_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return len(set(actual) & set(retrieved[:k])) / k if k else 0.0


def recall_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    return len(set(actual) & set(retrieved[:k])) / len(set(actual)) if actual else 0.0


def average_precision_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in actual_set:
            hits += 1
            score += hits / rank
    return score / min(len(actual_set), k)


def mrr_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    actual_set = set(actual)
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in actual_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    actual_set = set(actual)
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in actual_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def metrics_for(actual: Sequence[str], retrieved: Sequence[str], k: int) -> Dict[str, float]:
    precision = precision_at_k(actual, retrieved, k)
    recall = recall_at_k(actual, retrieved, k)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "effective_k": float(min(k, len(retrieved))),
        "hit": hit_at_k(actual, retrieved, k),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "map": average_precision_at_k(actual, retrieved, k),
        "mrr": mrr_at_k(actual, retrieved, k),
        "ndcg": ndcg_at_k(actual, retrieved, k),
        "context_precision": average_precision_at_k(actual, retrieved, k),
        "context_recall": recall,
    }


def evaluate(records: Sequence[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, Any]:
    results = []
    for record in records:
        actual = [str(item).strip() for item in record.get("groundtruth", []) if str(item).strip()]
        retrieved = unique_ids(record.get("candidates", []))
        first_hit_rank = None
        actual_set = set(actual)
        for idx, retrieved_id in enumerate(retrieved, start=1):
            if retrieved_id in actual_set:
                first_hit_rank = idx
                break

        results.append(
            {
                "id": str(record.get("id", "")),
                "question": record.get("question", ""),
                "groundtruth": actual,
                "retrieved_ids": retrieved,
                "candidate_count": len(retrieved),
                "first_hit_rank": first_hit_rank,
                "metrics": {f"@{k}": metrics_for(actual, retrieved, k) for k in k_values},
            }
        )

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
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        key = f"@{k}"
        summary[key] = {
            "effective_k_avg": mean(item["metrics"][key]["effective_k"] for item in results) if results else 0.0,
            **{
                metric: mean(item["metrics"][key][metric] for item in results) if results else 0.0
                for metric in metric_names
            },
        }

    return {"summary": summary, "results": results}


def write_outputs(detail: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "rerank_eval_detail.json"
    summary_path = output_dir / "rerank_eval_summary.csv"
    detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

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
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "effective_k_avg", *metric_names])
        for key, row in detail["summary"].items():
            writer.writerow(
                [
                    key.lstrip("@"),
                    f"{row['effective_k_avg']:.6f}",
                    *[f"{row[metric]:.6f}" for metric in metric_names],
                ]
            )

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {detail_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics from reranked JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path("logs") / f"reranked_retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    evaluated = evaluate(records, args.k)
    detail = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(args.input),
        "evaluated_rows": len(evaluated["results"]),
        "avg_candidates": mean(item["candidate_count"] for item in evaluated["results"])
        if evaluated["results"]
        else 0.0,
        "k_values": args.k,
        "summary": evaluated["summary"],
        "results": evaluated["results"],
    }
    write_outputs(detail, output_dir)

    print(f"evaluated_rows={detail['evaluated_rows']}")
    print(f"avg_candidates={detail['avg_candidates']:.4f}")
    print(json.dumps(detail["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
