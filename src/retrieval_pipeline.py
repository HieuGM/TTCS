import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pymongo.errors import OperationFailure, PyMongoError

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .indexer import get_indexer
from .pipeline_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalPipelineConfig


logger = logging.getLogger(__name__)
_INDEXER_CACHE: Optional[Tuple[Any, Any]] = None


def _get_indexer_cached() -> Tuple[Any, Any]:
    global _INDEXER_CACHE
    if _INDEXER_CACHE is None:
        _INDEXER_CACHE = get_indexer()
    return _INDEXER_CACHE


def _build_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


def generate_query_variants(
    query: str,
    config: RetrievalPipelineConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> List[str]:
    if not query or not query.strip():
        return []

    original = query.strip()
    if not config.enable_query_generation or config.generated_query_count <= 0:
        return [original]

    prompt = f"""Bạn là trợ lý tìm kiếm pháp luật Việt Nam.
Tạo đúng {config.generated_query_count} câu hỏi đồng nghĩa hoặc diễn đạt lại từ câu hỏi gốc.
Mục tiêu là bổ sung truy vấn cho hệ thống retrieval, không trả lời câu hỏi.
Chỉ trả về mỗi câu trên một dòng, không đánh số, không giải thích.

Câu hỏi gốc: {original}"""

    try:
        response = _build_llm().invoke(prompt)
        variants = []
        for line in response.content.strip().splitlines():
            cleaned = line.strip().strip("-*0123456789. ")
            if cleaned and cleaned != original and cleaned not in variants:
                variants.append(cleaned)
            if len(variants) >= config.generated_query_count:
                break
        return [original, *variants]
    except Exception as exc:
        logger.warning("Query generation failed, using original query only: %s", exc)
        return [original]


def document_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    for key in ("record_id", "_id", "id", "doc_id", "chunk_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return str(hash(doc.page_content))


def _to_document(record: Dict[str, Any]) -> Document:
    text = record.get("text") or record.get("page_content") or ""
    metadata: Dict[str, Any] = {}

    nested_metadata = record.get("metadata")
    if isinstance(nested_metadata, dict):
        metadata.update(nested_metadata)

    for key, value in record.items():
        if key in {"text", "page_content", "embedding", "metadata"}:
            continue
        metadata[key] = str(value) if key == "_id" else value

    source_doc = metadata.get("source_doc", "")
    article = metadata.get("article", "")
    clause = metadata.get("clause", "") or ""
    point = metadata.get("point", "") or ""
    location = ", ".join(str(part) for part in [article, clause, point] if part)
    if source_doc or location:
        page_content = f"{source_doc} - {location}:\n{text}".strip()
    else:
        page_content = str(text)

    return Document(page_content=page_content, metadata=metadata)


def vector_search(query: str, k: int) -> List[Document]:
    vector_store, _ = _get_indexer_cached()
    return vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)


def atlas_text_search(
    query: str,
    k: int,
    config: RetrievalPipelineConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> List[Document]:
    _, collection = _get_indexer_cached()
    field_path: Any = config.mongodb_text_search_field
    if "," in field_path:
        field_path = [field.strip() for field in field_path.split(",") if field.strip()]

    pipeline = [
        {
            "$search": {
                "index": config.mongodb_text_search_index,
                "text": {
                    "query": query,
                    "path": field_path,
                },
            }
        },
        {"$limit": k},
        {"$addFields": {"search_score": {"$meta": "searchScore"}}},
        {"$project": {"embedding": 0}},
    ]

    try:
        records = list(collection.aggregate(pipeline))
    except OperationFailure as exc:
        raise RuntimeError(
            "MongoDB Atlas Search text index is not usable. "
            f"index={config.mongodb_text_search_index!r}, "
            f"field={config.mongodb_text_search_field!r}. "
            "Create a dedicated Atlas Search text index on the configured field."
        ) from exc
    except PyMongoError as exc:
        raise RuntimeError(f"MongoDB Atlas Search failed: {exc}") from exc

    return [_to_document(record) for record in records]


def _ranked_source(
    source_name: str,
    docs: Sequence[Document],
) -> List[Tuple[str, int, Document]]:
    return [(source_name, rank, doc) for rank, doc in enumerate(docs, start=1)]


def rrf_fuse(
    ranked_sources: Sequence[Tuple[str, Sequence[Document]]],
    *,
    top_k: int,
    c: int,
) -> List[Document]:
    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    source_ranks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for source_name, docs in ranked_sources:
        for _, rank, doc in _ranked_source(source_name, docs):
            doc_key = document_id(doc)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            scores[doc_key] += 1 / (rank + c)
            source_ranks[doc_key].append({"source": source_name, "rank": rank})

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    output: List[Document] = []
    for rrf_rank, (doc_key, score) in enumerate(ranked, start=1):
        doc = doc_map[doc_key]
        metadata = dict(doc.metadata or {})
        metadata["rrf_score"] = score
        metadata["rrf_rank"] = rrf_rank
        metadata["source_ranks"] = source_ranks[doc_key]
        output.append(Document(page_content=doc.page_content, metadata=metadata))
    return output


def retrieve_rrf_candidates(
    query: str,
    config: RetrievalPipelineConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> Tuple[List[Document], List[str]]:
    queries = generate_query_variants(query, config)
    ranked_sources: List[Tuple[str, Sequence[Document]]] = []
    total_text_docs = 0

    for index, expanded_query in enumerate(queries):
        vector_docs = vector_search(expanded_query, config.rrf_top_k)
        text_docs = atlas_text_search(expanded_query, config.rrf_top_k, config)
        total_text_docs += len(text_docs)
        ranked_sources.append((f"q{index}_vector", vector_docs))
        ranked_sources.append((f"q{index}_text", text_docs))

    if total_text_docs == 0:
        raise RuntimeError(
            "Atlas Search text retrieval returned 0 documents for all queries. "
            f"index={config.mongodb_text_search_index!r}, "
            f"field={config.mongodb_text_search_field!r}. "
            "Verify that a text Atlas Search index exists and is queryable."
        )

    return (
        rrf_fuse(ranked_sources, top_k=config.rrf_top_k, c=config.rrf_c),
        queries,
    )


def bge_rerank(
    query: str,
    docs: Sequence[Document],
    config: RetrievalPipelineConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> List[Document]:
    if not docs:
        return []

    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "FlagEmbedding is required for local BGE rerank. "
            "Install dependencies or set ENABLE_LOCAL_BGE_RERANK=False."
        ) from exc

    reranker = FlagReranker(config.bge_reranker_model, use_fp16=config.bge_use_fp16)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]

    scored_docs = []
    for doc, score in zip(docs, scores):
        metadata = dict(doc.metadata or {})
        metadata["bge_score"] = float(score)
        scored_docs.append(Document(page_content=doc.page_content, metadata=metadata))

    scored_docs.sort(key=lambda doc: doc.metadata.get("bge_score", 0.0), reverse=True)
    return scored_docs[: config.rerank_top_n]


def retrieve_and_rerank(
    query: str,
    top_k: Optional[int] = None,
    top_n: Optional[int] = None,
    config: RetrievalPipelineConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> List[Document]:
    if not query or not query.strip():
        return []

    effective_config = config.with_overrides(rrf_top_k=top_k, rerank_top_n=top_n)
    candidates, queries = retrieve_rrf_candidates(query, effective_config)

    logger.info("Expanded queries: %s", queries)
    logger.info("RRF candidates: %s", len(candidates))

    if effective_config.enable_local_bge_rerank:
        return bge_rerank(query, candidates, effective_config)

    logger.warning(
        "Local BGE rerank is disabled. Returning top %s chunks after RRF.",
        effective_config.rerank_top_n,
    )
    return candidates[: effective_config.rerank_top_n]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def documents_to_candidate_payload(docs: Iterable[Document]) -> List[Dict[str, Any]]:
    payload = []
    for doc in docs:
        metadata = _json_safe(dict(doc.metadata or {}))
        payload.append(
            {
                "id": document_id(doc),
                "record_id": metadata.get("record_id", ""),
                "text": doc.page_content,
                "metadata": metadata,
                "rrf_score": metadata.get("rrf_score", 0.0),
                "rrf_rank": metadata.get("rrf_rank"),
                "source_ranks": metadata.get("source_ranks", []),
            }
        )
    return payload
