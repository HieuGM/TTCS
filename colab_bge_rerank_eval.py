#!/usr/bin/env python3
"""BGE rerank and retrieval evaluation for Colab.

Expected inputs:
- candidates.jsonl exported by evaluate_retrieval_export.py
- ground_truth/grounth_truth_record_id.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_K_VALUES = [1, 3, 5, 10]


def _pick(row: Dict[str, Any], candidates: Sequence[str]) -> str:
    normalized = {key.strip().lower(): key for key in row.keys()}
    for candidate in candidates:
        key = normalized.get(candidate.strip().lower())
        if key:
            return str(row.get(key) or "").strip()
    return ""


def split_groundtruth(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def load_ground_truth(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    output: Dict[str, Dict[str, Any]] = {}
    for row_num, row in enumerate(rows, start=2):
        row_id = _pick(row, ["ID", "id"]) or str(row_num - 1)
        question = _pick(row, ["Câu hỏi", "CÃ¢u há»i", "question"])
        answer = _pick(row, ["Answer", "answer"])
        groundtruth = split_groundtruth(_pick(row, ["Groundtruth", "groundtruth", "ground_truth"]))
        if not question or not groundtruth:
            raise ValueError(f"Invalid ground truth row {row_num}: missing question or Groundtruth")
        output[row_id] = {
            "id": row_id,
            "question": question,
            "answer": answer,
            "groundtruth": groundtruth,
        }
    return output


def load_candidates(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record or "candidates" not in record:
                raise ValueError(f"Invalid candidates row at line {line_num}")
            records.append(record)
    return records


def unique_ids(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    output = []
    for item in candidates:
        candidate_id = str(item.get("record_id") or item.get("id") or "").strip()
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            output.append(candidate_id)
    return output


def hit_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    retrieved_k = set(retrieved[:k])
    return 1.0 if any(item in retrieved_k for item in actual) else 0.0


def precision_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(actual) & set(retrieved[:k])) / k


def recall_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not actual:
        return 0.0
    return len(set(actual) & set(retrieved[:k])) / len(set(actual))


def average_precision_at_k(actual: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not actual:
        return 0.0
    actual_set = set(actual)
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
    import math

    actual_set = set(actual)
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in actual_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def metrics_for(actual: Sequence[str], retrieved: Sequence[str], k_values: Sequence[int]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        effective_k = min(k, len(retrieved))
        precision = precision_at_k(actual, retrieved, k)
        recall = recall_at_k(actual, retrieved, k)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"@{k}"] = {
            "effective_k": float(effective_k),
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
    return metrics


def rerank_records(
    records: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    top_n: int,
    use_fp16: bool,
) -> List[Dict[str, Any]]:
    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(model_name, use_fp16=use_fp16)
    output = []
    for idx, record in enumerate(records, start=1):
        question = record["question"]
        candidates = record.get("candidates") or []
        print(f"[{idx}/{len(records)}] BGE rerank row_id={record['id']} candidates={len(candidates)}")
        pairs = [[question, item.get("text", "")] for item in candidates]
        scores = reranker.compute_score(pairs, normalize=True) if pairs else []
        if pairs and not isinstance(scores, list):
            scores = [scores]

        scored = []
        for item, score in zip(candidates, scores):
            updated = dict(item)
            updated["bge_score"] = float(score)
            scored.append(updated)
        scored.sort(key=lambda item: item.get("bge_score", 0.0), reverse=True)
        output.append({**record, "reranked": scored[:top_n]})
    return output


def evaluate(
    candidates_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    top_n: int = 5,
    use_fp16: bool = True,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth(ground_truth_path)
    records = load_candidates(candidates_path)
    if len(records) != len(ground_truth):
        print(f"WARNING: candidates rows={len(records)} ground_truth rows={len(ground_truth)}")

    reranked = rerank_records(records, model_name=model_name, top_n=top_n, use_fp16=use_fp16)
    results = []
    for record in reranked:
        gt = ground_truth.get(str(record["id"]), record)
        actual = gt["groundtruth"]
        retrieved = unique_ids(record["reranked"])
        metrics = metrics_for(actual, retrieved, k_values)
        results.append(
            {
                "id": record["id"],
                "question": gt["question"],
                "groundtruth": actual,
                "retrieved": record["reranked"],
                "metrics": metrics,
            }
        )

    summary: Dict[str, Dict[str, float]] = {}
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
    for k in k_values:
        key = f"@{k}"
        summary[key] = {
            metric: mean(item["metrics"][key][metric] for item in results) if results else 0.0
            for metric in metric_names
        }

    detail = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "top_n": top_n,
        "evaluated_rows": len(results),
        "k_values": list(k_values),
        "summary": summary,
        "results": results,
    }

    reranked_jsonl = output_dir / "reranked_results.jsonl"
    summary_csv = output_dir / "rerank_eval_summary.csv"
    detail_json = output_dir / "rerank_eval_detail.json"

    with reranked_jsonl.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with detail_json.open("w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", *metric_names])
        for k in k_values:
            writer.writerow([k, *[f"{summary[f'@{k}'][metric]:.6f}" for metric in metric_names]])

    print(f"Wrote: {reranked_jsonl}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {detail_json}")
    print(f"evaluated_rows={len(results)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BGE rerank and evaluate retrieval candidates.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("rerank_eval_output"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--no-fp16", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evaluate(
        args.candidates,
        args.ground_truth,
        args.output_dir,
        model_name=args.model,
        top_n=args.top_n,
        use_fp16=not args.no_fp16,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
