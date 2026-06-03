"""
Semantic Cache cho Legal RAG Pipeline.

Cache câu trả lời dựa trên ngữ nghĩa (semantic similarity) của câu hỏi.
Sử dụng cosine similarity trên embedding vectors để nhận diện câu hỏi tương đồng.

Kiến trúc:
    - In-memory LRU cache (OrderedDict) cho tốc độ truy xuất nhanh
    - MongoDB persistent storage cho khả năng phục hồi sau restart
    - TTL-based auto-expiration qua MongoDB TTL index

Cùng embedding model với project: Quockhanh05/Vietnam_legal_embeddings
"""

import logging
import os
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

from .config import MONGODB_URI, DB_NAME

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Singleton embedding model — chia sẻ với indexer, chỉ load 1 lần
# ──────────────────────────────────────────────────────────────
_EMBEDDING_MODEL_CACHE: Optional[HuggingFaceEmbeddings] = None
DEFAULT_EMBEDDING_MODEL = "Quockhanh05/Vietnam_legal_embeddings"


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Lấy hoặc khởi tạo singleton embedding model."""
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is None:
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        logger.info("Đang tải embedding model cho Semantic Cache: %s", model_name)
        _EMBEDDING_MODEL_CACHE = HuggingFaceEmbeddings(model_name=model_name)
    return _EMBEDDING_MODEL_CACHE


# ──────────────────────────────────────────────────────────────
# Cosine Similarity
# ──────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính cosine similarity giữa 2 vector."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def batch_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Tính cosine similarity giữa 1 query vector và ma trận nhiều vectors.
    Nhanh hơn so với tính từng cặp khi cache lớn.
    
    Args:
        query_vec: vector shape (d,)
        matrix: ma trận shape (n, d) — n cached vectors
    
    Returns:
        array shape (n,) — similarity scores
    """
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)
    
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    
    matrix_norms = np.linalg.norm(matrix, axis=1)
    # Tránh chia cho 0
    matrix_norms = np.where(matrix_norms == 0, 1e-10, matrix_norms)
    
    dot_products = matrix @ query_vec
    similarities = dot_products / (matrix_norms * query_norm)
    return similarities.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# SemanticCache Class
# ──────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Semantic Cache cho Legal RAG Pipeline.
    
    Lưu trữ các cặp (query, response) cùng embedding vector.
    Khi có query mới, tính cosine similarity với tất cả cached embeddings
    để tìm câu hỏi tương đồng nhất.
    
    Nếu similarity >= threshold → trả về cached response (CACHE HIT).
    Nếu không → trả về None (CACHE MISS), pipeline RAG sẽ chạy bình thường.
    
    Args:
        similarity_threshold: Ngưỡng cosine similarity (mặc định 0.92)
        ttl_hours: Thời gian sống cache tính bằng giờ (mặc định 168 = 7 ngày)
        max_size: Số entry tối đa trong in-memory LRU cache (mặc định 500)
        collection_name: Tên MongoDB collection cho cache (mặc định "semantic_cache")
        enabled: Bật/tắt cache (mặc định True)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_hours: int = 168,
        max_size: int = 500,
        collection_name: str = "semantic_cache",
        enabled: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self.max_size = max_size
        self.collection_name = collection_name
        self.enabled = enabled

        # In-memory LRU cache: OrderedDict giữ thứ tự truy cập
        # key = query text, value = dict{embedding, response, created_at, hit_count}
        self._memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # MongoDB collection handle
        self._collection: Optional[Collection] = None

        # Embedding model handle
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

        # Vectorized cache cho batch similarity (rebuild khi cache thay đổi)
        self._cached_keys: List[str] = []
        self._cached_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True

        if self.enabled:
            self._init_mongodb()
            self._init_embeddings()
            self._load_from_db()

    # ── Khởi tạo ──────────────────────────────────────────────

    def _init_mongodb(self) -> None:
        """Kết nối MongoDB và tạo TTL index cho auto-expiration."""
        try:
            client = MongoClient(MONGODB_URI)
            db = client[DB_NAME]
            self._collection = db[self.collection_name]
            # TTL index: MongoDB tự xóa document khi created_at + ttl hết hạn
            self._collection.create_index(
                "created_at", expireAfterSeconds=self.ttl_hours * 3600
            )
            # Index cho query lookup
            self._collection.create_index("query", unique=True)
            logger.info(
                "Semantic Cache MongoDB initialized: db=%s, collection=%s",
                DB_NAME, self.collection_name,
            )
        except Exception as e:
            logger.warning("Không thể khởi tạo MongoDB cho cache: %s", e)
            self._collection = None

    def _init_embeddings(self) -> None:
        """Khởi tạo embedding model (singleton, chia sẻ với indexer)."""
        try:
            self._embeddings = _get_embedding_model()
        except Exception as e:
            logger.warning("Không thể tải embedding model cho cache: %s", e)
            self._embeddings = None

    def _load_from_db(self) -> None:
        """Load cache entries từ MongoDB vào memory khi khởi động."""
        if self._collection is None:
            return

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
            cursor = self._collection.find(
                {"created_at": {"$gte": cutoff}},
                sort=[("created_at", DESCENDING)],
                limit=self.max_size,
            )

            entries = list(cursor)
            # Insert theo thứ tự cũ nhất trước → mới nhất cuối (LRU order)
            for entry in reversed(entries):
                key = entry.get("query", "")
                if key and key not in self._memory_cache:
                    self._memory_cache[key] = {
                        "embedding": np.array(entry["embedding"], dtype=np.float32),
                        "response": entry["response"],
                        "created_at": entry["created_at"],
                        "hit_count": entry.get("hit_count", 0),
                    }

            self._matrix_dirty = True
            logger.info(
                "Đã tải %d cache entries từ MongoDB vào memory", len(self._memory_cache)
            )
        except Exception as e:
            logger.warning("Không thể tải cache từ MongoDB: %s", e)

    # ── Embedding & Similarity ────────────────────────────────

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Tính embedding vector cho một câu query."""
        if self._embeddings is None:
            return None
        try:
            vector = self._embeddings.embed_query(query)
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            logger.warning("Không thể embed query cho cache: %s", e)
            return None

    def _rebuild_matrix(self) -> None:
        """Rebuild ma trận embeddings từ memory cache để dùng batch similarity."""
        valid_keys = []
        vectors = []
        expired_keys = []

        for key, entry in self._memory_cache.items():
            if self._is_expired(entry["created_at"]):
                expired_keys.append(key)
                continue
            valid_keys.append(key)
            vectors.append(entry["embedding"])

        # Xóa entries hết hạn
        for key in expired_keys:
            del self._memory_cache[key]

        self._cached_keys = valid_keys
        if vectors:
            self._cached_matrix = np.stack(vectors)
        else:
            self._cached_matrix = np.empty((0, 0), dtype=np.float32)

        self._matrix_dirty = False

    def _is_expired(self, created_at: Any) -> bool:
        """Kiểm tra xem cache entry đã hết hạn chưa."""
        if not isinstance(created_at, datetime):
            return True
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return (now - created_at) > timedelta(hours=self.ttl_hours)

    # ── Eviction ──────────────────────────────────────────────

    def _evict_lru(self) -> None:
        """Xóa entries cũ nhất nếu cache vượt max_size."""
        while len(self._memory_cache) > self.max_size:
            self._memory_cache.popitem(last=False)
        self._matrix_dirty = True

    # ── Public API ────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[str]:
        """
        Tìm câu trả lời đã cache cho một query tương đồng.
        
        Quy trình:
            1. Embed query → vector
            2. Tính cosine similarity với tất cả cached embeddings (batch)
            3. Nếu max similarity >= threshold → return cached response
            4. Ngược lại → return None
        
        Args:
            query: Câu hỏi của người dùng
        
        Returns:
            Cached response nếu tìm thấy match, None nếu cache miss
        """
        if not self.enabled or not query or not query.strip():
            return None

        query = query.strip()
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return None

        # Rebuild matrix nếu cache đã thay đổi
        if self._matrix_dirty:
            self._rebuild_matrix()

        if not self._cached_keys or self._cached_matrix is None or self._cached_matrix.shape[0] == 0:
            logger.info("Cache MISS (cache rỗng): query='%s'", query[:60])
            return None

        # Batch cosine similarity — nhanh hơn loop
        similarities = batch_cosine_similarity(query_embedding, self._cached_matrix)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.similarity_threshold:
            best_key = self._cached_keys[best_idx]
            entry = self._memory_cache[best_key]

            # Cập nhật LRU order
            self._memory_cache.move_to_end(best_key)
            entry["hit_count"] = entry.get("hit_count", 0) + 1

            # Cập nhật hit_count trong MongoDB (non-blocking, non-critical)
            if self._collection is not None:
                try:
                    self._collection.update_one(
                        {"query": best_key},
                        {"$inc": {"hit_count": 1}},
                    )
                except Exception:
                    pass

            logger.info(
                "Cache HIT: score=%.4f | query='%s' → matched='%s'",
                best_score, query[:50], best_key[:50],
            )
            return entry["response"]

        logger.info(
            "Cache MISS: best_score=%.4f < threshold=%.2f | query='%s'",
            best_score, self.similarity_threshold, query[:60],
        )
        return None

    def store(self, query: str, response: str) -> None:
        """
        Lưu cặp (query, response) vào cache.
        
        Lưu cả vào in-memory LRU và MongoDB persistent storage.
        
        Args:
            query: Câu hỏi gốc
            response: Câu trả lời từ LLM
        """
        if not self.enabled or not query or not query.strip() or not response:
            return

        query = query.strip()
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return

        now = datetime.now(timezone.utc)

        # Lưu vào memory
        self._memory_cache[query] = {
            "embedding": query_embedding,
            "response": response,
            "created_at": now,
            "hit_count": 0,
        }
        self._memory_cache.move_to_end(query)
        self._evict_lru()
        self._matrix_dirty = True

        # Persist vào MongoDB
        if self._collection is not None:
            try:
                self._collection.update_one(
                    {"query": query},
                    {
                        "$set": {
                            "query": query,
                            "embedding": query_embedding.tolist(),
                            "response": response,
                            "created_at": now,
                            "hit_count": 0,
                        }
                    },
                    upsert=True,
                )
            except Exception as e:
                logger.warning("Không thể lưu cache entry vào MongoDB: %s", e)

        logger.info("Cache STORE: query='%s'", query[:60])

    def invalidate_all(self) -> int:
        """
        Xóa toàn bộ cache (memory + MongoDB).
        
        Returns:
            Số entries đã xóa
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()
        self._cached_keys.clear()
        self._cached_matrix = None
        self._matrix_dirty = True

        if self._collection is not None:
            try:
                result = self._collection.delete_many({})
                count = max(count, result.deleted_count)
            except Exception as e:
                logger.warning("Không thể xóa MongoDB cache: %s", e)

        logger.info("Cache INVALIDATED: %d entries đã xóa", count)
        return count

    def stats(self) -> Dict[str, Any]:
        """
        Thống kê cache.
        
        Returns:
            Dict chứa thông tin cache: số entries, hits, config, ...
        """
        total_hits = sum(
            entry.get("hit_count", 0) for entry in self._memory_cache.values()
        )
        db_count = 0
        if self._collection is not None:
            try:
                db_count = self._collection.count_documents({})
            except Exception:
                pass

        return {
            "enabled": self.enabled,
            "memory_entries": len(self._memory_cache),
            "db_entries": db_count,
            "total_hits": total_hits,
            "similarity_threshold": self.similarity_threshold,
            "ttl_hours": self.ttl_hours,
            "max_size": self.max_size,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SemanticCache(enabled={s['enabled']}, "
            f"memory={s['memory_entries']}, "
            f"db={s['db_entries']}, "
            f"hits={s['total_hits']}, "
            f"threshold={s['similarity_threshold']})"
        )
