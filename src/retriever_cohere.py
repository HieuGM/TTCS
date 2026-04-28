import json
import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank

from .indexer import get_indexer
from .config import GOOGLE_API_KEY

# Đảm bảo bạn đã thêm COHERE_API_KEY vào file .env
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BM25 cache singleton
_bm25_retriever_cache: Optional[BM25Retriever] = None

# -------------------------------------------------------------
# 1. Custom Ensemble Retriever (Reciprocal Rank Fusion - RRF)
# -------------------------------------------------------------
def get_base_retriever(k: int = 20):
    """Get vector retriever with standard similarity search."""
    vectorStore, _ = get_indexer()
    return vectorStore.as_retriever(search_kwargs={"k": k})

def get_bm25_retriever(k: int = 20) -> Optional[BM25Retriever]:
    """Get cached BM25 retriever, build on first call."""
    global _bm25_retriever_cache

    if _bm25_retriever_cache is not None:
        _bm25_retriever_cache.k = k
        return _bm25_retriever_cache

    file_path = r"E:\TTCS\Code_base\TTCS\person3_resolved.jsonl"
    documents = []

    try:
        if not os.path.exists(file_path):
            logger.warning(f"BM25 data file not found: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)

                source_doc = data.get("source_doc", "")
                article = data.get("article", "")
                clause = data.get("clause", "") or ""
                point = data.get("point", "") or ""
                raw_text = data.get("text", "")

                location = ", ".join(filter(None, [article, clause, point]))
                page_content = f"{source_doc} — {location}:\n{raw_text}"

                metadata = {key: val for key, val in data.items() if key != "text"}
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)

        if documents:
            _bm25_retriever_cache = BM25Retriever.from_documents(documents)
            _bm25_retriever_cache.k = k
            logger.info(f"BM25 retriever built with {len(documents)} documents")
            return _bm25_retriever_cache

    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading BM25 data: {e}")

    return None

def reciprocal_rank_fusion(doc_lists: List[List[Document]], c: int = 60) -> List[Document]:
    """Kết hợp danh sách document từ nhiều nguồn (BM25, Vector) bằng RRF."""
    fused_scores = {}
    doc_map = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            doc_id = hash(doc.page_content + str(doc.metadata))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (rank + c)

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, v in reranked]

def ensemble_retrieve(query: str, vector_retriever, bm25_retriever) -> List[Document]:
    """Gọi cả 2 retriever và trộn kết quả bằng RRF."""
    vector_docs = vector_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query) if bm25_retriever else []
    return reciprocal_rank_fusion([vector_docs, bm25_docs])

# -------------------------------------------------------------
# 2. Custom Multi-Query Generator (Vẫn dùng Gemini)
# -------------------------------------------------------------
def generate_queries(query: str, llm: ChatGoogleGenerativeAI) -> List[str]:
    prompt = f"""Bạn là một trợ lý pháp lý AI. Người dùng đang tìm kiếm thông tin về pháp luật Việt Nam.
Nhiệm vụ của bạn là tạo ra đúng 3 câu hỏi biến thể khác nhau từ câu hỏi gốc của người dùng.
Hãy sử dụng từ đồng nghĩa, cấu trúc lại câu hoặc mở rộng các khái niệm liên quan để giúp hệ thống tìm kiếm tài liệu tốt hơn.
Chỉ trả về 3 câu hỏi (mỗi câu 1 dòng), KHÔNG giải thích gì thêm, KHÔNG dùng dấu gạch đầu dòng hay số thứ tự.

Câu hỏi gốc: {query}"""

    try:
        res = llm.invoke(prompt)
        lines = res.content.strip().split('\n')

        variations = []
        for line in lines:
            clean_line = line.strip('-*1234567890. ')
            if clean_line:
                variations.append(clean_line)

        if query not in variations:
            variations.insert(0, query)

        return variations[:4]
    except Exception as e:
        logger.warning(f"LLM query generation failed: {e}. Using original query.")
        return [query]

# -------------------------------------------------------------
# 3. Pipeline Chính (Retrieve & Rerank)
# -------------------------------------------------------------
def retrieve_and_rerank(query: str, top_k: int = 20, top_n: int = 5) -> List[Document]:
    if not query or not query.strip():
        return []
    if top_k < 1 or top_n < 1:
        return []
    if top_n > top_k:
        top_n = top_k

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)

    print(f"\n========== [BƯỚC 1] TẠO MULTI-QUERY ==========")
    queries = generate_queries(query, llm)
    print("Các biến thể câu hỏi sẽ tìm kiếm:")
    for q in queries:
        print(f" - {q}")

    print(f"\n========== [BƯỚC 2] ENSEMBLE RETRIEVE (VECTOR + BM25) ==========")
    vector_retriever = get_base_retriever(k=top_k)
    bm25_retriever = get_bm25_retriever(k=top_k)

    all_docs = []
    seen_hashes = set()

    for q in queries:
        docs = ensemble_retrieve(q, vector_retriever, bm25_retriever)
        for doc in docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                all_docs.append(doc)

    print(f"\nĐã gộp tổng cộng {len(all_docs)} tài liệu không trùng lặp từ tất cả câu hỏi.")

    print(f"\n... Đang chạy COHERE Reranker để chọn ra Top {top_n} tài liệu. Vui lòng đợi ...\n")

    # Sử dụng Cohere Rerank thay vì Gemini
    # Model: rerank-multilingual-v3.0 hỗ trợ tiếng Việt cực mạnh
    try:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("Không tìm thấy COHERE_API_KEY trong môi trường!")
            
        reranker = CohereRerank(
            cohere_api_key=cohere_api_key,
            model="rerank-multilingual-v3.0",
            top_n=top_n
        )
        final_docs = reranker.compress_documents(all_docs, query)
    except Exception as e:
        print(f"Lỗi Cohere Rerank: {e}")
        print("-> Fallback: Trả về tài liệu gốc chưa rerank.")
        final_docs = all_docs[:top_n]

    print(f"\n========== [BƯỚC 3] TOP {top_n} TÀI LIỆU CUỐI CÙNG (COHERE) ==========")
    for i, doc in enumerate(final_docs):
        # Điểm của Cohere trả về là dạng float từ 0 đến 1, nhân 10 để tương đồng với code cũ
        raw_score = doc.metadata.get('relevance_score', 0)
        formatted_score = round(raw_score * 10, 2)
        print(f"\n[Reranked Doc {i+1}] | Điểm Cohere chấm: {formatted_score}/10")
        print(f"Nội dung: {doc.page_content[:250]}...")
        clean_meta = {k: v for k, v in doc.metadata.items() if k not in ['embedding', '_id']}
        print(f"Metadata: {clean_meta}")
    print("\n===========================================================\n")

    return final_docs
