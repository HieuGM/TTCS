#!/usr/bin/env python3
"""Run retrieval export, remote file rerank, and offline metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests
from dotenv import dotenv_values


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVALUATE_DIR = Path(__file__).resolve().parent
if str(EVALUATE_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATE_DIR))

from metrics import (  # noqa: E402
    DEFAULT_K_VALUES,
    evaluate_all_metrics,
    load_ground_truth,
    load_reranked_chunks,
    parse_k_values,
    write_outputs,
)
from retrieval_vietnamese_bi_encoder import (  # noqa: E402
    DEFAULT_GROUND_TRUTH,
    DEFAULT_OUTPUT_ROOT,
    export_retrieved_chunks,
)


ENV_VALUES = {
    key: str(value)
    for key, value in dotenv_values(REPO_ROOT / ".env").items()
    if key and value is not None
}


def _env_value(name: str, default: str = "") -> str:
    return os.getenv(name, ENV_VALUES.get(name, default))


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _normalize_remote_base_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    for endpoint in ("/rerank-batch", "/rerank-file", "/rerank"):
        if cleaned.endswith(endpoint):
            return cleaned[: -len(endpoint)].rstrip("/")
    return cleaned


def _normalize_live_rerank_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if cleaned.endswith("/rerank-batch"):
        return cleaned[:-len("/rerank-batch")] + "/rerank"
    if cleaned.endswith("/rerank-file"):
        return cleaned[:-len("/rerank-file")] + "/rerank"
    if cleaned.endswith("/rerank"):
        return cleaned
    return cleaned + "/rerank"


def _normalize_batch_rerank_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if cleaned.endswith("/rerank-batch"):
        return cleaned
    if cleaned.endswith("/rerank-file"):
        return cleaned[:-len("/rerank-file")] + "/rerank-batch"
    if cleaned.endswith("/rerank"):
        return cleaned[:-len("/rerank")] + "/rerank-batch"
    return cleaned + "/rerank-batch"


def _derive_batch_rerank_url() -> str:
    base_url = _normalize_remote_base_url(_env_value("REMOTE_BGE_RERANK_BASE_URL"))
    if base_url:
        return _normalize_batch_rerank_url(base_url)

    explicit_url = _env_value("REMOTE_EVAL_RERANK_BATCH_URL").strip()
    if explicit_url:
        return _normalize_batch_rerank_url(explicit_url)

    live_url = _env_value("REMOTE_BGE_RERANK_URL").strip()
    if live_url:
        return _normalize_batch_rerank_url(live_url)

    eval_url = _env_value("REMOTE_EVAL_RERANK_URL").strip()
    return _normalize_batch_rerank_url(eval_url) if eval_url else ""


def _question_for_record(record: Dict[str, Any]) -> str:
    generated_queries = record.get("generated_queries")
    if isinstance(generated_queries, list) and generated_queries:
        first_query = str(generated_queries[0] or "").strip()
        if first_query:
            return first_query
    return str(record.get("question") or "").strip()


def _post_single_rerank(
    *,
    server_url: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
    query: str,
    chunks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "query": query,
        "documents": [
            {
                "index": index,
                "id": str(chunk.get("record_id") or chunk.get("id") or index),
                "text": str(chunk.get("text") or "").replace("\n", " "),
            }
            for index, chunk in enumerate(chunks)
        ],
        "top_n": len(chunks),
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                server_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            results = response.json().get("results")
            break
        except requests.RequestException as exc:
            last_error = exc
            status_code = getattr(exc.response, "status_code", None)
            retryable = status_code is None or status_code >= 500
            if attempt >= max_retries or not retryable:
                raise

            wait_seconds = retry_sleep_seconds * attempt
            print(
                f"  Remote rerank error"
                f"{f' HTTP {status_code}' if status_code else ''}; "
                f"retry {attempt}/{max_retries - 1} after {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)
    else:
        raise RuntimeError("Remote rerank failed without response.") from last_error

    if not isinstance(results, list):
        raise RuntimeError("Remote rerank response must contain a 'results' list.")
    return results


def _post_batch_rerank(
    *,
    server_url: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {"records": list(records)}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                server_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            response_records = response.json().get("records")
            break
        except requests.RequestException as exc:
            last_error = exc
            status_code = getattr(exc.response, "status_code", None)
            retryable = status_code is None or status_code >= 500
            if attempt >= max_retries or not retryable:
                raise

            wait_seconds = retry_sleep_seconds * attempt
            print(
                f"  Remote batch rerank error"
                f"{f' HTTP {status_code}' if status_code else ''}; "
                f"retry {attempt}/{max_retries - 1} after {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)
    else:
        raise RuntimeError("Remote batch rerank failed without response.") from last_error

    if not isinstance(response_records, list):
        raise RuntimeError("Remote batch rerank response must contain a 'records' list.")
    return response_records


def _apply_rerank_results(record: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> None:
    chunks = record.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return

    by_index: Dict[int, Dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        by_index[index] = item

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        item = by_index.get(index)
        if item is None:
            continue
        chunk["rerank_score"] = float(item.get("score", 0.0))

    sorted_chunks = sorted(
        [chunk for chunk in chunks if isinstance(chunk, dict)],
        key=lambda item: item.get("rerank_score", float("-inf")),
        reverse=True,
    )
    for rank, chunk in enumerate(sorted_chunks, start=1):
        chunk["rerank_rank"] = rank

    groundtruth_chunks = record.get("groundtruth_chunks")
    if isinstance(groundtruth_chunks, list):
        sorted_by_record_id = {
            str(chunk.get("record_id")): chunk
            for chunk in sorted_chunks
            if chunk.get("record_id") is not None
        }
        for chunk in groundtruth_chunks:
            if not isinstance(chunk, dict):
                continue
            chunk["rerank_score"] = "NONE"
            chunk["rerank_rank"] = "NONE"
            matched = sorted_by_record_id.get(str(chunk.get("record_id")))
            if matched is not None:
                chunk["rerank_score"] = matched.get("rerank_score", "NONE")
                chunk["rerank_rank"] = matched.get("rerank_rank", "NONE")

    record["chunks"] = sorted_chunks


def rerank_records_remotely(
    records: Any,
    *,
    server_url: str,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
    batch_size: int,
    checkpoint_path: Path,
) -> Any:
    if isinstance(records, dict) and isinstance(records.get("results"), list):
        iterable_records = records["results"]
    elif isinstance(records, list):
        iterable_records = records
    else:
        raise ValueError("Retrieved chunks JSON must be a list or an object with a 'results' list.")

    total = len(iterable_records)
    pending: List[tuple[int, Dict[str, Any]]] = []
    for index, record in enumerate(iterable_records):
        if not isinstance(record, dict):
            continue
        if record.get("remote_rerank_done") is True:
            continue
        chunks = record.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            continue
        pending.append((index, record))

    if not pending:
        return records

    effective_batch_size = max(1, int(batch_size))
    total_batches = (len(pending) + effective_batch_size - 1) // effective_batch_size
    for batch_index, start in enumerate(range(0, len(pending), effective_batch_size), start=1):
        batch = pending[start:start + effective_batch_size]
        start_row = batch[0][0] + 1
        end_row = batch[-1][0] + 1
        print(
            f"[batch {batch_index}/{total_batches}] remote reranking "
            f"rows {start_row}-{end_row}/{total} ({len(batch)} questions)"
        )

        response_records = _post_batch_rerank(
            server_url=server_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            records=[record for _, record in batch],
        )

        if len(response_records) != len(batch):
            raise RuntimeError(
                "Remote batch rerank returned "
                f"{len(response_records)} records for {len(batch)} input records."
            )

        for (record_index, _), response_record in zip(batch, response_records):
            if not isinstance(response_record, dict):
                raise RuntimeError("Remote batch rerank returned a non-object record.")
            response_record["remote_rerank_done"] = True
            iterable_records[record_index] = response_record

        write_json(checkpoint_path, records)

    return records


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_metrics(
    *,
    ground_truth_path: Path,
    reranked_path: Path,
    output_dir: Path,
    k_values: Sequence[int],
) -> None:
    ground_truth = load_ground_truth(ground_truth_path)
    reranked_records = load_reranked_chunks(reranked_path)
    report = evaluate_all_metrics(ground_truth, reranked_records, k_values)
    report["ground_truth_path"] = str(ground_truth_path)
    report["input_path"] = str(reranked_path)
    summary_path, detail_csv_path, detail_json_path = write_outputs(report, output_dir)

    print(f"evaluated_rows={report['evaluated_rows']}")
    print(f"reranked_records={report['reranked_records']}")
    if report["missing_predictions"]:
        print(f"missing_predictions={len(report['missing_predictions'])}", file=sys.stderr)
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {detail_csv_path}")
    print(f"Wrote: {detail_json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full retrieval -> remote Colab rerank -> metrics evaluation."
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--retrieved-input", type=Path, default=None, help="Existing retrieved_chunks.json to rerank.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only; 0 means all rows.")
    parser.add_argument("--top-k", type=int, default=100, help="RRF candidate count per question.")
    parser.add_argument("--server-url", type=str, default="", help="Override /rerank-batch URL.")
    parser.add_argument("--api-key", type=str, default="", help="Override X-API-Key.")
    parser.add_argument("--timeout", type=float, default=0, help="Remote request timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=0, help="Retries per remote rerank request.")
    parser.add_argument("--retry-sleep", type=float, default=0, help="Base sleep seconds between retries.")
    parser.add_argument("--batch-size", type=int, default=0, help="Questions per remote batch request.")
    parser.add_argument("--resume", action="store_true", help="Resume from reranked_chunks.partial.json if present.")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ground_truth_path = _resolve_repo_path(args.ground_truth)
    output_root = _resolve_repo_path(args.output_root)
    k_values = parse_k_values(args.k)

    server_url = _normalize_batch_rerank_url(args.server_url) if args.server_url.strip() else _derive_batch_rerank_url()
    if not server_url:
        raise RuntimeError(
            "No rerank-batch server URL. Set REMOTE_BGE_RERANK_BASE_URL or pass --server-url."
        )

    api_key = args.api_key or _env_value("REMOTE_BGE_RERANK_API_KEY", "")
    timeout_seconds = args.timeout or float(_env_value("REMOTE_BGE_RERANK_TIMEOUT_SECONDS", "60"))
    max_retries = args.max_retries or int(_env_value("REMOTE_EVAL_RERANK_MAX_RETRIES", "5"))
    retry_sleep_seconds = args.retry_sleep or float(_env_value("REMOTE_EVAL_RERANK_RETRY_SLEEP_SECONDS", "10"))
    batch_size = args.batch_size or int(_env_value("REMOTE_EVAL_RERANK_BATCH_SIZE", "10"))

    if args.retrieved_input:
        print("Step 1/3: using existing retrieved chunks...")
        retrieved_path = _resolve_repo_path(args.retrieved_input)
    else:
        print("Step 1/3: exporting retrieved chunks...")
        retrieved_path = export_retrieved_chunks(
            ground_truth_path,
            output_root,
            limit=args.limit,
            top_k=args.top_k,
        )
    run_dir = retrieved_path.parent
    reranked_path = run_dir / "reranked_chunks.json"
    checkpoint_path = run_dir / "reranked_chunks.partial.json"
    print(f"Wrote: {retrieved_path}")

    print("Step 2/3: reranking retrieved chunks via remote Colab /rerank-batch endpoint...")
    print(f"Remote rerank URL: {server_url}")
    print(f"Remote batch size: {batch_size} questions/request")
    if args.resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        retrieved_payload = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
    else:
        retrieved_payload = json.loads(retrieved_path.read_text(encoding="utf-8-sig"))
    reranked_payload = rerank_records_remotely(
        retrieved_payload,
        server_url=server_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
    )
    write_json(reranked_path, reranked_payload)
    print(f"Wrote: {reranked_path}")

    print("Step 3/3: computing offline metrics...")
    run_metrics(
        ground_truth_path=ground_truth_path,
        reranked_path=reranked_path,
        output_dir=run_dir,
        k_values=k_values,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
