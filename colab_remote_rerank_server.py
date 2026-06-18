"""
Colab T4 remote BGE rerank server.

Run this file in a Colab runtime with GPU enabled, then copy the printed
trycloudflare base URL into local .env as REMOTE_BGE_RERANK_BASE_URL.

Colab setup cell:
    !pip install -U "transformers==4.44.2"
    !pip install -U FlagEmbedding fastapi uvicorn nest_asyncio pydantic python-multipart

Run server cell:
    !python colab_remote_rerank_server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from FlagEmbedding import FlagReranker


API_KEY = os.getenv("REMOTE_BGE_RERANK_API_KEY", "change-me")
MODEL_NAME = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
BATCH_SIZE = int(os.getenv("REMOTE_BGE_BATCH_SIZE", "64"))
PORT = int(os.getenv("REMOTE_BGE_RERANK_PORT", "8000"))


class RerankDocument(BaseModel):
    index: int
    id: str = ""
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: List[RerankDocument] = Field(default_factory=list)
    top_n: int = 20


class RerankBatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list)


def _scores_to_list(scores: Any) -> List[Any]:
    if isinstance(scores, float):
        return [scores]
    if isinstance(scores, list):
        return scores
    try:
        return list(scores)
    except TypeError:
        return [scores]


app = FastAPI(title="TTCS Remote BGE Reranker")


print(f"Dang tai model rerank: {MODEL_NAME}")
reranker = FlagReranker(MODEL_NAME, use_fp16=True)
print("Da tai model rerank thanh cong.")


@app.get("/health")
def health():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
        device_count = torch.cuda.device_count()
    except Exception as exc:
        cuda_available = False
        device_name = f"unknown: {exc}"
        device_count = 0

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "cuda_available": cuda_available,
        "device": device_name,
        "cuda_device_count": device_count,
    }


@app.post("/rerank")
def rerank(payload: RerankRequest, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    if not payload.documents:
        return {"results": []}

    pairs = [
        [payload.query, doc.text.replace("\n", " ")]
        for doc in payload.documents
    ]
    scores = reranker.compute_score(
        pairs,
        normalize=True,
        batch_size=BATCH_SIZE,
    )
    scores = _scores_to_list(scores)

    results = []
    for doc, score in zip(payload.documents, scores):
        results.append(
            {
                "index": doc.index,
                "id": doc.id,
                "score": float(score),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    top_n = max(0, min(payload.top_n, len(results)))
    results = results[:top_n]
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return {"results": results}


@app.post("/rerank-batch")
def rerank_batch(payload: RerankBatchRequest, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    total_questions = len(payload.records)
    for index, record in enumerate(payload.records, start=1):
        print(f"Da xu ly batch item {index}/{total_questions} cau hoi")
        _rerank_eval_record(record)

    return {"records": payload.records}


def _question_for_eval_record(record: Dict[str, Any]) -> str:
    generated_queries = record.get("generated_queries")
    if isinstance(generated_queries, list) and generated_queries:
        first_query = str(generated_queries[0] or "").strip()
        if first_query:
            return first_query
    return str(record.get("question") or "").strip()


def _rerank_eval_record(record: Dict[str, Any]) -> None:
    question = _question_for_eval_record(record)
    retrieved_chunks = record.get("chunks")
    if not isinstance(retrieved_chunks, list) or not retrieved_chunks:
        return

    texts = [
        str(chunk.get("text") or "").replace("\n", " ")
        for chunk in retrieved_chunks
        if isinstance(chunk, dict)
    ]
    pairs = [[question, text] for text in texts]
    if not pairs:
        return

    scores = reranker.compute_score(
        pairs,
        normalize=True,
        batch_size=BATCH_SIZE,
    )
    scores = _scores_to_list(scores)

    for chunk, score in zip(retrieved_chunks, scores):
        if isinstance(chunk, dict):
            chunk["rerank_score"] = float(score)

    sorted_chunks = sorted(
        [chunk for chunk in retrieved_chunks if isinstance(chunk, dict)],
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


@app.post("/rerank-file")
async def rerank_file(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        records = payload["results"]
        response_payload = payload
    elif isinstance(payload, list):
        records = payload
        response_payload = payload
    else:
        raise HTTPException(
            status_code=400,
            detail="Rerank file must be a JSON list or an object with a 'results' list",
        )

    total_questions = len(records)
    for index, record in enumerate(records):
        if index % 10 == 0:
            print(f"Da xu ly {index}/{total_questions} cau hoi")
        if isinstance(record, dict):
            _rerank_eval_record(record)

    print("Hoan thanh qua trinh Rerank!")
    return response_payload


def _cloudflared_path() -> str:
    existing = shutil.which("cloudflared")
    if existing:
        return existing

    target = Path("/tmp/cloudflared")
    if target.exists():
        return str(target)

    print("Dang tai cloudflared de mo Cloudflare quick tunnel...")
    subprocess.run(
        [
            "wget",
            "-q",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-O",
            str(target),
        ],
        check=True,
    )
    target.chmod(0o755)
    return str(target)


def start_cloudflare_tunnel() -> subprocess.Popen:
    cloudflared = _cloudflared_path()
    process = subprocess.Popen(
        [
            cloudflared,
            "tunnel",
            "--url",
            f"http://127.0.0.1:{PORT}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _reader():
        assert process.stdout is not None
        pattern = re.compile(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com")
        for line in process.stdout:
            print(line.rstrip())
            match = pattern.search(line)
            if match:
                base_url = match.group(0)
                print("\nREMOTE_BGE_RERANK_BASE_URL=" + base_url)
                print("# Derived endpoints:")
                print("#   live rerank: " + base_url + "/rerank")
                print("#   eval file rerank: " + base_url + "/rerank-file")
                print("#   eval batch rerank: " + base_url + "/rerank-batch")
                print("REMOTE_BGE_RERANK_API_KEY=" + API_KEY + "\n")

    threading.Thread(target=_reader, daemon=True).start()
    return process


def run_uvicorn_server() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT)
    server = uvicorn.Server(config)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(server.serve())
        return

    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    nest_asyncio.apply()
    tunnel_process = start_cloudflare_tunnel()
    time.sleep(2)
    try:
        run_uvicorn_server()
    finally:
        tunnel_process.terminate()
