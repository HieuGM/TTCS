import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_CONFIG_FILE = REPO_ROOT / ".env"


def _load_pipeline_values() -> Dict[str, str]:
    config_path = Path(os.getenv("PIPELINE_CONFIG_FILE", str(DEFAULT_PIPELINE_CONFIG_FILE)))
    if not config_path.exists():
        return {}
    return {
        key: str(value)
        for key, value in dotenv_values(config_path).items()
        if key and value is not None
    }


_PIPELINE_VALUES = _load_pipeline_values()


def _value(name: str, default: str) -> str:
    return os.getenv(name, _PIPELINE_VALUES.get(name, default))


def _bool(name: str, default: bool) -> bool:
    raw = _value(name, "true" if default else "false")
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    raw = _value(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = _value(name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RetrievalPipelineConfig:
    enable_query_generation: bool = _bool("ENABLE_QUERY_GENERATION", True)
    enable_static_query_expansion: bool = _bool("ENABLE_STATIC_QUERY_EXPANSION", True)
    query_expansion_file: str = _value(
        "QUERY_EXPANSION_FILE",
        str(REPO_ROOT / "evaluate" / "query_expansion_supplementary_queries.csv"),
    )
    generated_query_count: int = _int("GENERATED_QUERY_COUNT", 3)
    rrf_top_k: int = _int("RRF_TOP_K", 100)
    rerank_top_n: int = _int("RERANK_TOP_N", 5)
    mongodb_text_search_index: str = _value("MONGODB_TEXT_SEARCH_INDEX", "default")
    mongodb_text_search_field: str = _value("MONGODB_TEXT_SEARCH_FIELD", "text")
    bge_reranker_model: str = _value("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    enable_local_bge_rerank: bool = _bool("ENABLE_LOCAL_BGE_RERANK", False)
    bge_use_fp16: bool = _bool("BGE_USE_FP16", False)
    rrf_c: int = _int("RRF_C", 60)

    def with_overrides(
        self,
        *,
        enable_query_generation: Optional[bool] = None,
        rrf_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> "RetrievalPipelineConfig":
        return RetrievalPipelineConfig(
            enable_query_generation=(
                self.enable_query_generation if enable_query_generation is None else enable_query_generation
            ),
            enable_static_query_expansion=self.enable_static_query_expansion,
            query_expansion_file=self.query_expansion_file,
            generated_query_count=self.generated_query_count,
            rrf_top_k=rrf_top_k or self.rrf_top_k,
            rerank_top_n=rerank_top_n or self.rerank_top_n,
            mongodb_text_search_index=self.mongodb_text_search_index,
            mongodb_text_search_field=self.mongodb_text_search_field,
            bge_reranker_model=self.bge_reranker_model,
            enable_local_bge_rerank=self.enable_local_bge_rerank,
            bge_use_fp16=self.bge_use_fp16,
            rrf_c=self.rrf_c,
        )


DEFAULT_RETRIEVAL_CONFIG = RetrievalPipelineConfig()


@dataclass(frozen=True)
class CacheConfig:
    """Cấu hình cho Semantic Cache, đọc từ .env"""
    enable_semantic_cache: bool = _bool("ENABLE_SEMANTIC_CACHE", True)
    cache_similarity_threshold: float = _float("CACHE_SIMILARITY_THRESHOLD", 0.92)
    cache_ttl_hours: int = _int("CACHE_TTL_HOURS", 168)
    cache_max_size: int = _int("CACHE_MAX_SIZE", 500)
    cache_collection_name: str = _value("CACHE_COLLECTION_NAME", "semantic_cache")


DEFAULT_CACHE_CONFIG = CacheConfig()
