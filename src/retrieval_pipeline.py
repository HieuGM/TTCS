import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pymongo.errors import OperationFailure, PyMongoError

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .indexer import get_indexer
from .pipeline_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalPipelineConfig


logger = logging.getLogger(__name__)
_INDEXER_CACHE: Optional[Tuple[Any, Any]] = None
_STATIC_QUERY_EXPANSION_CACHE: Dict[Path, Tuple[float, Dict[str, List[str]]]] = {}


QUERY_NORMALIZATION_RULES: Dict[Tuple[str, ...], str] = {
    ("vượt đèn đỏ", "vượi đèn", "vượt đèn tín hiệu"): (
        "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
    ),
    
    ("Xe máy") :("xe mô tô, xe gắn máy, xe hai bánh có động cơ"),
    
    ("chở quá số người", "chở quá tải người", "chở quá số chỗ ngồi"): (
        "chở quá số người quy định"
    ),
    ("quá tải", "chở quá tải trọng", "chở quá trọng tải"): (
        "chở quá tải trọng quy định"
    ),
    ("nồng độ cồn", "điều khiển phương tiện khi có nồng độ cồn", "uống rượu bia lái xe"): (
        "điều khiển phương tiện giao thông khi trong máu hoặc hơi thở có nồng độ cồn vượt quá mức quy định"
    ),
    ("không đội mũ bảo hiểm", "không đội mũ bảo hiểm cho người ngồi trên xe máy", "không đội mũ bảo hiểm khi đi xe máy"): (
        "không đội mũ bảo hiểm khi tham gia giao thông"
    ),
    
}


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


def normalize_query(query: str) -> str:
    original = " ".join(str(query or "").split())
    if not original:
        return ""

    replacements: List[Tuple[str, str]] = []
    for source_terms, replacement in QUERY_NORMALIZATION_RULES.items():
        if isinstance(source_terms, str):
            source_terms = (source_terms,)
        for source_term in source_terms:
            source_term = str(source_term).strip()
            if source_term:
                replacements.append((source_term, replacement))

    if not replacements:
        return original

    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    replacement_by_term = {
        source_term.casefold(): replacement for source_term, replacement in replacements
    }
    pattern = re.compile(
        "|".join(re.escape(source_term) for source_term, _ in replacements),
        flags=re.IGNORECASE,
    )

    normalized = pattern.sub(
        lambda match: replacement_by_term[match.group(0).casefold()],
        original,
    )

    return " ".join(normalized.split())


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).casefold()


def _load_static_query_expansions(path_value: str) -> Dict[str, List[str]]:
    if not path_value:
        return {}

    path = Path(path_value)
    if not path.exists():
        logger.warning("Static query expansion file does not exist: %s", path)
        return {}

    mtime = path.stat().st_mtime
    cached = _STATIC_QUERY_EXPANSION_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    expansions: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = str(row.get("original_question") or "").strip()
            if not original:
                continue

            variants = []
            for index in range(1, 20):
                value = str(row.get(f"supplement_query_{index}") or "").strip()
                if value and value not in variants:
                    variants.append(value)

            if variants:
                expansions[_query_key(original)] = variants

    _STATIC_QUERY_EXPANSION_CACHE[path] = (mtime, expansions)
    return expansions


def _static_query_variants(
    primary_query: str,
    config: RetrievalPipelineConfig,
    fallback_query: Optional[str] = None,
) -> List[str]:
    if not config.enable_static_query_expansion:
        return []

    expansions = _load_static_query_expansions(config.query_expansion_file)
    variants = expansions.get(_query_key(primary_query), [])
    if not variants and fallback_query:
        variants = expansions.get(_query_key(fallback_query), [])
    return variants[: config.generated_query_count]


def _append_unique_query(queries: List[str], query: str) -> None:
    cleaned = " ".join(str(query or "").split())
    if not cleaned:
        return

    existing = {_query_key(item) for item in queries}
    if _query_key(cleaned) not in existing:
        queries.append(cleaned)


def generate_query_variants(
    query: str,
    config: RetrievalPipelineConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> List[str]:
    if not query or not query.strip():
        return []

    original = query.strip()
    normalized = normalize_query(original) or original
    queries = [normalized]
    _append_unique_query(queries, original)

    if config.generated_query_count <= 0:
        return queries

    static_variants = _static_query_variants(normalized, config, fallback_query=original)
    if static_variants:
        for variant in static_variants:
            _append_unique_query(queries, variant)
        return queries

    if not config.enable_query_generation:
        return queries

    prompt = f"""Bạn là bộ sinh truy vấn bổ sung cho hệ thống Legal RAG pháp luật Việt Nam.

    Nhiệm vụ của bạn là phân tích câu hỏi gốc và tạo đúng {config.generated_query_count} truy vấn bổ sung giúp tìm được các điều khoản pháp luật liên quan trong cơ sở dữ liệu.

    Mỗi truy vấn nên bổ sung một hướng tìm kiếm khác nhau, ưu tiên:
    - tên hành vi vi phạm theo ngôn ngữ pháp lý chính thức;
    - đối tượng hoặc phương tiện liên quan;
    - căn cứ về mức phạt tiền;
    - căn cứ về hình phạt bổ sung hoặc tước giấy phép;
    - căn cứ về biện pháp khắc phục hậu quả;
    - trường hợp đặc biệt nếu có trong câu hỏi như gây tai nạn, không có giấy phép, chở quá số người, quá tải, nồng độ cồn, vượt đèn đỏ.

    Ràng buộc:
    - Không tạo các câu đồng nghĩa đơn thuần.
    - Không mở rộng sang lỗi vi phạm khác nếu câu hỏi gốc không gợi ý.
    - Không trả lời câu hỏi.
    - Không nêu nhận xét.
    - Chỉ trả về đúng {config.generated_query_count} dòng, mỗi dòng là một truy vấn tìm kiếm hoàn chỉnh.

    Câu hỏi gốc: {normalized}"""

    try:
        response = _build_llm().invoke(prompt)
        for line in response.content.strip().splitlines():
            cleaned = line.strip().strip("-*0123456789. ")
            _append_unique_query(queries, cleaned)
            if len(queries) >= config.generated_query_count + 1:
                break
        return queries
    except Exception as exc:
        logger.warning("Query generation failed, using original query only: %s", exc)
        return queries


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


# def vector_search(query: str, k: int) -> List[Document]:
#     vector_store, _ = _get_indexer_cached()
#     return vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)

def vector_search(
    query: str, 
    k: int, 
    filter_subjects: Optional[List[str]] = None, 
    filter_topics: Optional[List[str]] = None
) -> List[Document]:
    vector_store, _ = _get_indexer_cached()
    
    # ---- TẠO PRE-FILTER CHO MONGODB ----
    pre_filter = {}
    if filter_subjects:
        # Quan trọng: Luôn tự động gắn thêm 'tat_ca' vào câu query để lấy cả luật chung
        pre_filter["subjects"] = {"$in": filter_subjects + ["tat_ca"]}
    if filter_topics:
        pre_filter["topics"] = {"$in": filter_topics}

    search_kwargs = {"k": k}
    if pre_filter:
        search_kwargs["pre_filter"] = pre_filter
    # ------------------------------------

    return vector_store.as_retriever(search_kwargs=search_kwargs).invoke(query)


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
        # {
        #     "$search": {
        #         "index": config.mongodb_text_search_index,
        #         "text": {
        #             "query": query,
        #             "path": field_path,
        #         },
        #     }
        # },
        {
            "$search": {
                "index": config.mongodb_text_search_index,
                "compound": {
                    "should": [
                        {
                            # 1. Điểm nền BM25 (Bag-of-words thông thường)
                            "text": {
                                "query": query,
                                "path": field_path,
                                "score": {"boost": {"value": 1.0}}
                            }
                        },
                        {
                            # 2. Điểm Proximity / Bigram (Các từ đứng gần nhau)
                            "phrase": {
                                "query": query,
                                "path": field_path,
                                "slop": 2,  # Cho phép các từ cách nhau tối đa 2 khoảng trắng (nếu bị chen ngang)
                                "score": {"boost": {"value": 3.0}} # Thưởng điểm cực mạnh cho ngữ cảnh đúng
                            }
                        }
                    ]
                }
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




_RERANKER_CACHE: Optional[Any] = None
_RERANKER_MODEL_NAME: Optional[str] = None


def _get_reranker(model_name: str, use_fp16: bool) -> Any:
    """Singleton FlagReranker — chỉ load model 1 lần duy nhất."""
    global _RERANKER_CACHE, _RERANKER_MODEL_NAME
    if _RERANKER_CACHE is None or _RERANKER_MODEL_NAME != model_name:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is required for local BGE rerank. "
                "Install dependencies or set ENABLE_LOCAL_BGE_RERANK=False."
            ) from exc

        import time
        logger.info("Đang tải BGE Reranker model: %s (lần đầu tiên)...", model_name)
        start = time.perf_counter()
        _RERANKER_CACHE = FlagReranker(model_name, use_fp16=use_fp16)
        elapsed = time.perf_counter() - start
        logger.info("BGE Reranker đã sẵn sàng (%.1fs). Các lần sau sẽ dùng lại model này.", elapsed)
        _RERANKER_MODEL_NAME = model_name
    return _RERANKER_CACHE


def bge_rerank(
    query: str,
    docs: Sequence[Document],
    config: RetrievalPipelineConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> List[Document]:
    if not docs:
        return []

    reranker = _get_reranker(config.bge_reranker_model, config.bge_use_fp16)

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.compute_score(pairs, normalize=True, batch_size=256)
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
