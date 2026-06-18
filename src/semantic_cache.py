"""
Semantic Cache cho Legal RAG Pipeline — Phiên bản cải thiện toàn diện.

Cải thiện so với phiên bản gốc:
    1. Query Normalization  — normalize query trước khi embed/lookup
    2. lookup_with_metadata — trả về score, matched_key, hit flag
    3. Smart Deduplication  — store() tránh tạo duplicate entry gần giống
    4. Thread Safety         — RLock bảo vệ in-memory cache
    5. MongoDB Resilience    — graceful degradation khi DB down; retry
    6. Diagnostics mở rộng  — hit_rate, last_hit_at, invalidate_by_query, invalidate_expired
"""

from __future__ import annotations

import logging
import os
import threading
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection

from .config import DB_NAME, MONGODB_URI

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None  # type: ignore

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
except ImportError:
    LangChainDeprecationWarning = Warning  # type: ignore

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Singleton embedding model
# ──────────────────────────────────────────────────────────────────────────────
_EMBEDDING_MODEL_CACHE: Optional[Any] = None
_EMBEDDING_LOCK = threading.Lock()
DEFAULT_EMBEDDING_MODEL = "Quockhanh05/Vietnam_legal_embeddings"


def _get_embedding_model() -> Any:
    """Lấy hoặc khởi tạo singleton embedding model (thread-safe)."""
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is not None:
        return _EMBEDDING_MODEL_CACHE
    with _EMBEDDING_LOCK:
        if _EMBEDDING_MODEL_CACHE is None:
            if HuggingFaceEmbeddings is None:
                raise ImportError("langchain_community is required for HuggingFaceEmbeddings.")
            model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            logger.info("Đang tải embedding model cho Semantic Cache: %s", model_name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
                _EMBEDDING_MODEL_CACHE = HuggingFaceEmbeddings(model_name=model_name)
    return _EMBEDDING_MODEL_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# Vector math helpers
# ──────────────────────────────────────────────────────────────────────────────

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

    Args:
        query_vec: shape (d,)
        matrix:    shape (n, d)
    Returns:
        array shape (n,)
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)

    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.where(matrix_norms == 0, 1e-10, matrix_norms)

    dot_products = matrix @ query_vec
    similarities = dot_products / (matrix_norms * query_norm)

    # Clip to [-1, 1] để tránh numerical issues (NaN/Inf)
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Query normalization (lazy import để tránh circular)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_query_safe(query: str) -> str:
    """Normalize query dùng retrieval_pipeline.normalize_query nếu có thể."""
    try:
        from .retrieval_pipeline import normalize_query
        return normalize_query(query) or query
    except Exception:
        return query


# ──────────────────────────────────────────────────────────────────────────────
# CacheEntry TypedDict-like structure
# ──────────────────────────────────────────────────────────────────────────────

def _make_entry(
    embedding: np.ndarray,
    response: str,
    original_query: str,
    normalized_query: str,
    source: str = "auto",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "embedding": embedding,
        "response": response,
        "original_query": original_query,
        "normalized_query": normalized_query,
        "created_at": now,
        "last_hit_at": None,
        "hit_count": 0,
        "response_length": len(response),
        "embedding_dim": len(embedding),
        "source": source,         # "auto" | "manual"
    }


def _as_utc_datetime(value: Any) -> Optional[datetime]:
    """Chuẩn hóa datetime từ MongoDB/in-memory về timezone-aware UTC."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# SemanticCache
# ──────────────────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Semantic Cache cho Legal RAG Pipeline.

    Lưu trữ các cặp (query, response) cùng embedding vector.
    Khi có query mới, tính cosine similarity với tất cả cached embeddings
    để tìm câu hỏi tương đồng nhất.

    Nếu similarity >= threshold → trả về cached response (CACHE HIT).
    Nếu không → trả về None (CACHE MISS).

    Cải thiện:
        - normalize_query trước khi embed
        - lookup_with_metadata() trả về score + matched_key
        - Smart dedup: store() kiểm tra entry tương đồng trước khi ghi mới
        - Thread-safe: RLock bảo vệ mọi thao tác trên _memory_cache
        - Graceful degradation: MongoDB down → chỉ dùng in-memory
        - Diagnostics: hit_rate, invalidate_by_query, invalidate_expired, ...

    Args:
        similarity_threshold:    Ngưỡng cosine similarity để xác nhận CACHE HIT (mặc định 0.92)
        dedup_threshold:         Ngưỡng cao hơn để phát hiện duplicate khi store() (mặc định 0.97)
        ttl_hours:               Thời gian sống cache (giờ, mặc định 168 = 7 ngày)
        max_size:                Số entry tối đa trong in-memory LRU (mặc định 500)
        collection_name:         Tên MongoDB collection (mặc định "semantic_cache")
        enabled:                 Bật/tắt cache (mặc định True)
        normalize_queries:       Tự động normalize query trước khi embed (mặc định True)
        mongo_retry_attempts:    Số lần retry khi MongoDB lỗi (mặc định 2)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        dedup_threshold: float = 0.97,
        ttl_hours: int = 168,
        max_size: int = 500,
        collection_name: str = "semantic_cache",
        enabled: bool = True,
        normalize_queries: bool = True,
        mongo_retry_attempts: int = 2,
    ):
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.ttl_hours = ttl_hours
        self.max_size = max_size
        self.collection_name = collection_name
        self.enabled = enabled
        self.normalize_queries = normalize_queries
        self.mongo_retry_attempts = mongo_retry_attempts

        # In-memory LRU cache (OrderedDict)
        self._memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

        # MongoDB
        self._collection: Optional[Collection] = None
        self._mongo_available: bool = False

        # Embedding model
        self._embeddings: Optional[Any] = None
        self._embedding_init_attempted: bool = False

        # Vectorized matrix (lazy rebuild)
        self._cached_keys: List[str] = []
        self._cached_matrix: Optional[np.ndarray] = None
        self._matrix_dirty: bool = True

        # Stats counters
        self._total_lookups: int = 0
        self._total_hits: int = 0

        if self.enabled:
            self._init_mongodb()
            self._load_from_db()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_mongodb(self) -> None:
        """Kết nối MongoDB và tạo indexes. Không raise nếu thất bại."""
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            db = client[DB_NAME]
            self._collection = db[self.collection_name]

            self._collection.create_index(
                "created_at", expireAfterSeconds=self.ttl_hours * 3600
            )
            self._collection.create_index("query", unique=True)
            self._collection.create_index("normalized_query")

            self._mongo_available = True
            logger.info(
                "Semantic Cache MongoDB: db=%s, collection=%s",
                DB_NAME, self.collection_name,
            )
        except Exception as exc:
            logger.warning("MongoDB không khả dụng cho Semantic Cache: %s. Dùng in-memory only.", exc)
            self._collection = None
            self._mongo_available = False

    def _init_embeddings(self) -> None:
        """Khởi tạo embedding model (singleton)."""
        if self._embedding_init_attempted:
            return
        self._embedding_init_attempted = True
        try:
            self._embeddings = _get_embedding_model()
        except Exception as exc:
            logger.warning("Không thể tải embedding model cho cache: %s", exc)
            self._embeddings = None

    def _load_from_db(self) -> None:
        """Load cache entries từ MongoDB vào memory khi khởi động."""
        if not self._mongo_available or self._collection is None:
            return
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
            cursor = self._collection.find(
                {"created_at": {"$gte": cutoff}},
                sort=[("created_at", DESCENDING)],
                limit=self.max_size,
            )
            entries = list(cursor)
            with self._lock:
                for entry in reversed(entries):
                    key = entry.get("query", "")
                    if not key or key in self._memory_cache:
                        continue
                    created_at = _as_utc_datetime(entry.get("created_at")) or datetime.now(timezone.utc)
                    last_hit_at = _as_utc_datetime(entry.get("last_hit_at"))
                    self._memory_cache[key] = {
                        "embedding": np.array(entry["embedding"], dtype=np.float32),
                        "response": entry["response"],
                        "original_query": entry.get("original_query", key),
                        "normalized_query": entry.get("normalized_query", key),
                        "created_at": created_at,
                        "last_hit_at": last_hit_at,
                        "hit_count": entry.get("hit_count", 0),
                        "response_length": entry.get("response_length", len(entry["response"])),
                        "embedding_dim": entry.get("embedding_dim", len(entry["embedding"])),
                        "source": entry.get("source", "auto"),
                    }
                self._matrix_dirty = True
            logger.info("Đã tải %d cache entries từ MongoDB", len(self._memory_cache))
        except Exception as exc:
            logger.warning("Không thể tải cache từ MongoDB: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_query(self, raw_query: str) -> Tuple[str, str]:
        """
        Chuẩn hóa query.
        Returns: (normalized_query, original_query)
        """
        original = " ".join(raw_query.split())
        if self.normalize_queries:
            normalized = _normalize_query_safe(original)
        else:
            normalized = original
        return normalized, original

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Tính embedding vector. Trả về None nếu lỗi."""
        if self._embeddings is None:
            self._init_embeddings()
            if self._embeddings is None:
                return None
        try:
            vector = self._embeddings.embed_query(query)
            arr = np.array(vector, dtype=np.float32)
            # Kiểm tra NaN/Inf
            if not np.all(np.isfinite(arr)):
                logger.warning("Embedding chứa NaN/Inf, bỏ qua.")
                return None
            return arr
        except Exception as exc:
            logger.warning("Lỗi embed query: %s", exc)
            return None

    def _rebuild_matrix(self) -> None:
        """Rebuild ma trận embedding từ in-memory cache (phải hold _lock)."""
        valid_keys: List[str] = []
        vectors: List[np.ndarray] = []
        expired_keys: List[str] = []

        for key, entry in self._memory_cache.items():
            if self._is_expired(entry["created_at"]):
                expired_keys.append(key)
            else:
                valid_keys.append(key)
                vectors.append(entry["embedding"])

        for k in expired_keys:
            del self._memory_cache[k]

        self._cached_keys = valid_keys
        if vectors:
            self._cached_matrix = np.stack(vectors).astype(np.float32)
        else:
            self._cached_matrix = np.empty((0, 0), dtype=np.float32)
        self._matrix_dirty = False

    def _is_expired(self, created_at: Any) -> bool:
        """Kiểm tra entry đã hết hạn chưa."""
        normalized_created_at = _as_utc_datetime(created_at)
        if normalized_created_at is None:
            return True
        now = datetime.now(timezone.utc)
        return (now - normalized_created_at) > timedelta(hours=self.ttl_hours)

    def _evict_lru(self) -> None:
        """Xóa entries cũ nhất nếu vượt max_size (phải hold _lock)."""
        while len(self._memory_cache) > self.max_size:
            self._memory_cache.popitem(last=False)
        self._matrix_dirty = True

    def _find_best_match(
        self, query_embedding: np.ndarray, threshold: float
    ) -> Tuple[Optional[str], float]:
        """
        Tìm entry tốt nhất trong cache với similarity >= threshold.
        Phải hold _lock trước khi gọi.

        Returns: (best_key, best_score) — best_key=None nếu không match
        """
        if self._matrix_dirty:
            self._rebuild_matrix()

        if not self._cached_keys or self._cached_matrix is None or self._cached_matrix.shape[0] == 0:
            return None, 0.0

        similarities = batch_cosine_similarity(query_embedding, self._cached_matrix)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            return self._cached_keys[best_idx], best_score
        return None, best_score

    def _mongo_upsert(self, key: str, entry: Dict[str, Any]) -> None:
        """Persist entry vào MongoDB với retry. Non-blocking về logic."""
        if not self._mongo_available or self._collection is None:
            return
        doc = {
            "query": key,
            "embedding": entry["embedding"].tolist(),
            "response": entry["response"],
            "original_query": entry["original_query"],
            "normalized_query": entry["normalized_query"],
            "created_at": entry["created_at"],
            "last_hit_at": entry.get("last_hit_at"),
            "hit_count": entry.get("hit_count", 0),
            "response_length": entry.get("response_length", 0),
            "embedding_dim": entry.get("embedding_dim", 0),
            "source": entry.get("source", "auto"),
        }
        for attempt in range(max(1, self.mongo_retry_attempts)):
            try:
                self._collection.update_one(
                    {"query": key}, {"$set": doc}, upsert=True
                )
                return
            except Exception as exc:
                if attempt < self.mongo_retry_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logger.warning("Không thể ghi cache vào MongoDB sau %d lần thử: %s", self.mongo_retry_attempts, exc)
                    self._mongo_available = False

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> Optional[str]:
        """
        Tìm câu trả lời đã cache cho query tương đồng.

        Returns:
            Cached response nếu HIT, None nếu MISS.
        """
        result = self.lookup_with_metadata(query)
        return result["response"] if result["hit"] else None

    def lookup_with_metadata(self, query: str) -> Dict[str, Any]:
        """
        Tra cứu cache và trả về metadata đầy đủ.

        Returns dict:
            hit           (bool)
            response      (str | None)
            score         (float)  — best similarity score
            matched_key   (str | None) — query đã match
            query_used    (str) — query sau normalize
        """
        default = {"hit": False, "response": None, "score": 0.0, "matched_key": None, "query_used": query}

        if not self.enabled or not query or not query.strip():
            return default

        normalized, original = self._prepare_query(query)
        default["query_used"] = normalized

        query_embedding = self._embed_query(normalized)
        if query_embedding is None:
            return default

        with self._lock:
            self._total_lookups += 1
            best_key, best_score = self._find_best_match(query_embedding, self.similarity_threshold)

            if best_key is not None:
                entry = self._memory_cache[best_key]
                self._memory_cache.move_to_end(best_key)
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                entry["last_hit_at"] = datetime.now(timezone.utc)
                self._total_hits += 1

                logger.info(
                    "Cache HIT: score=%.4f | '%s' → matched='%s'",
                    best_score, normalized[:50], best_key[:50],
                )

                # Update MongoDB hit_count async-ish (non-critical)
                if self._mongo_available and self._collection is not None:
                    try:
                        self._collection.update_one(
                            {"query": best_key},
                            {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": entry["last_hit_at"]}},
                        )
                    except Exception:
                        pass

                return {
                    "hit": True,
                    "response": entry["response"],
                    "score": best_score,
                    "matched_key": best_key,
                    "query_used": normalized,
                }

        logger.info(
            "Cache MISS: best_score=%.4f < threshold=%.2f | '%s'",
            best_score, self.similarity_threshold, normalized[:60],
        )
        return {"hit": False, "response": None, "score": best_score, "matched_key": None, "query_used": normalized}

    def store(self, query: str, response: str, source: str = "auto", force: bool = False) -> bool:
        """
        Lưu cặp (query, response) vào cache.

        Nếu đã có entry tương đồng (>= dedup_threshold) → update response thay vì tạo mới,
        trừ khi force=True.

        Args:
            query:    Câu hỏi gốc
            response: Câu trả lời từ LLM
            source:   "auto" hoặc "manual"
            force:    Nếu True, bypass dedup check, luôn tạo entry mới

        Returns:
            True nếu đã store thành công, False nếu bỏ qua
        """
        if not self.enabled or not query or not query.strip() or not response:
            return False

        normalized, original = self._prepare_query(query)
        query_embedding = self._embed_query(normalized)
        if query_embedding is None:
            return False

        now = datetime.now(timezone.utc)

        with self._lock:
            # Dedup check — tìm entry tương đồng rất cao
            if not force:
                dedup_key, dedup_score = self._find_best_match(query_embedding, self.dedup_threshold)
                if dedup_key is not None:
                    # Update response cho entry cũ thay vì tạo mới
                    entry = self._memory_cache[dedup_key]
                    entry["response"] = response
                    entry["response_length"] = len(response)
                    entry["last_hit_at"] = now
                    self._memory_cache.move_to_end(dedup_key)
                    logger.info(
                        "Cache DEDUP UPDATE (score=%.4f): updated entry '%s'",
                        dedup_score, dedup_key[:50],
                    )
                    self._mongo_upsert(dedup_key, entry)
                    return True

            # Lưu entry mới
            entry = _make_entry(query_embedding, response, original, normalized, source)
            self._memory_cache[normalized] = entry
            self._memory_cache.move_to_end(normalized)
            self._evict_lru()
            self._matrix_dirty = True

        self._mongo_upsert(normalized, entry)
        logger.info("Cache STORE: '%s'", normalized[:60])
        return True

    def invalidate_all(self) -> int:
        """Xóa toàn bộ cache (memory + MongoDB). Trả về số entries đã xóa."""
        with self._lock:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            self._cached_keys.clear()
            self._cached_matrix = None
            self._matrix_dirty = True
            self._total_hits = 0
            self._total_lookups = 0

        if self._mongo_available and self._collection is not None:
            try:
                result = self._collection.delete_many({})
                count = max(count, result.deleted_count)
            except Exception as exc:
                logger.warning("Không thể xóa MongoDB cache: %s", exc)

        logger.info("Cache INVALIDATED: %d entries", count)
        return count

    def invalidate_by_query(self, query: str) -> bool:
        """
        Xóa 1 entry cụ thể theo query text (khớp chính xác sau normalize).

        Returns:
            True nếu tìm thấy và xóa, False nếu không có.
        """
        normalized, _ = self._prepare_query(query)
        removed = False

        with self._lock:
            if normalized in self._memory_cache:
                del self._memory_cache[normalized]
                self._matrix_dirty = True
                removed = True

        if removed and self._mongo_available and self._collection is not None:
            try:
                self._collection.delete_one({"query": normalized})
            except Exception as exc:
                logger.warning("Không thể xóa MongoDB entry: %s", exc)

        return removed

    def invalidate_expired(self) -> int:
        """Chủ động xóa các entries đã hết hạn khỏi memory. Trả về số đã xóa."""
        expired = []
        with self._lock:
            for key, entry in self._memory_cache.items():
                if self._is_expired(entry["created_at"]):
                    expired.append(key)
            for key in expired:
                del self._memory_cache[key]
            if expired:
                self._matrix_dirty = True
        logger.info("Đã dọn %d expired entries", len(expired))
        return len(expired)

    def is_healthy(self) -> Dict[str, bool]:
        """Kiểm tra trạng thái các thành phần."""
        mongo_ok = False
        if self._mongo_available and self._collection is not None:
            try:
                self._collection.database.client.admin.command("ping")
                mongo_ok = True
            except Exception:
                mongo_ok = False
                self._mongo_available = False

        return {
            "enabled": self.enabled,
            "embedding_model": self._embeddings is not None,
            "mongodb": mongo_ok,
        }

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Export tất cả in-memory entries (không kèm embedding vector)."""
        with self._lock:
            result = []
            for key, entry in self._memory_cache.items():
                result.append({
                    "key": key,
                    "original_query": entry.get("original_query", key),
                    "normalized_query": entry.get("normalized_query", key),
                    "response_length": entry.get("response_length", 0),
                    "hit_count": entry.get("hit_count", 0),
                    "created_at": entry.get("created_at"),
                    "last_hit_at": entry.get("last_hit_at"),
                    "source": entry.get("source", "auto"),
                })
            return result

    def stats(self) -> Dict[str, Any]:
        """Thống kê cache chi tiết."""
        with self._lock:
            mem_count = len(self._memory_cache)
            total_hits_in_mem = sum(e.get("hit_count", 0) for e in self._memory_cache.values())

            oldest: Optional[datetime] = None
            newest: Optional[datetime] = None
            for entry in self._memory_cache.values():
                ca = _as_utc_datetime(entry.get("created_at"))
                if ca is not None:
                    if oldest is None or ca < oldest:
                        oldest = ca
                    if newest is None or ca > newest:
                        newest = ca

        db_count = 0
        if self._mongo_available and self._collection is not None:
            try:
                db_count = self._collection.count_documents({})
            except Exception:
                pass

        hit_rate = (self._total_hits / self._total_lookups) if self._total_lookups > 0 else 0.0

        return {
            "enabled": self.enabled,
            "memory_entries": mem_count,
            "db_entries": db_count,
            "total_hits": total_hits_in_mem,
            "session_lookups": self._total_lookups,
            "session_hits": self._total_hits,
            "hit_rate": round(hit_rate, 4),
            "similarity_threshold": self.similarity_threshold,
            "dedup_threshold": self.dedup_threshold,
            "ttl_hours": self.ttl_hours,
            "max_size": self.max_size,
            "normalize_queries": self.normalize_queries,
            "mongodb_available": self._mongo_available,
            "oldest_entry": oldest.isoformat() if oldest else None,
            "newest_entry": newest.isoformat() if newest else None,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SemanticCache(enabled={s['enabled']}, "
            f"memory={s['memory_entries']}, "
            f"db={s['db_entries']}, "
            f"hits={s['total_hits']}, "
            f"hit_rate={s['hit_rate']:.2%}, "
            f"threshold={s['similarity_threshold']}, "
            f"mongo={'OK' if s['mongodb_available'] else 'DOWN'})"
        )
