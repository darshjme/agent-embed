"""
agent-embed: Zero-dependency embedding utilities for LLM agents.

Provides cosine similarity, vector normalization, nearest-neighbor search,
and semantic deduplication — all in pure Python with no dependencies.
"""

from .vectors import (
    cosine_similarity,
    dot_product,
    normalize,
    magnitude,
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
)
from .search import NearestNeighborSearch, SearchResult
from .dedup import SemanticDeduplicator
from .routing import SemanticRouter, Route

__all__ = [
    "cosine_similarity",
    "dot_product",
    "normalize",
    "magnitude",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "NearestNeighborSearch",
    "SearchResult",
    "SemanticDeduplicator",
    "SemanticRouter",
    "Route",
]

__version__ = "0.1.0"
