"""
Nearest-neighbor search implementations.

Supports:
  - Brute-force exact search (always correct, fine for < ~50k vectors)
  - Metrics: cosine similarity (default), euclidean, dot product
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .vectors import cosine_similarity, euclidean_distance, dot_product


@dataclass
class SearchResult:
    """A single nearest-neighbor result."""
    index: int
    score: float
    metadata: Any = None

    def __repr__(self) -> str:
        return f"SearchResult(index={self.index}, score={self.score:.4f})"


class NearestNeighborSearch:
    """Brute-force nearest-neighbor search for embedding vectors.

    Usage::

        index = NearestNeighborSearch(metric="cosine")
        index.add([0.1, 0.9, 0.3], metadata={"text": "hello"})
        index.add([0.8, 0.2, 0.1], metadata={"text": "world"})
        results = index.query([0.1, 0.85, 0.35], k=1)
        # [SearchResult(index=0, score=0.9998)]
    """

    METRICS = ("cosine", "euclidean", "dot")

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {self.METRICS}")
        self.metric = metric
        self._vectors: List[Optional[List[float]]] = []
        self._metadata: List[Any] = []

    def add(self, vector: Sequence[float], metadata: Any = None) -> int:
        """Add a vector to the index. Returns its assigned index."""
        self._vectors.append(list(vector))
        self._metadata.append(metadata)
        return len(self._vectors) - 1

    def add_batch(
        self,
        vectors: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Any]] = None,
    ) -> List[int]:
        """Add multiple vectors at once."""
        if metadatas is None:
            metadatas = [None] * len(vectors)
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas must have the same length")
        return [self.add(v, m) for v, m in zip(vectors, metadatas)]

    def __len__(self) -> int:
        return sum(1 for v in self._vectors if v is not None)

    def _score(self, a: Sequence[float], b: Sequence[float]) -> float:
        if self.metric == "cosine":
            return cosine_similarity(a, b)
        elif self.metric == "dot":
            return dot_product(a, b)
        else:  # euclidean — negate so higher = closer
            return -euclidean_distance(a, b)

    def query(
        self,
        vector: Sequence[float],
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Return top-k nearest neighbors sorted by descending score."""
        active = [(i, v) for i, v in enumerate(self._vectors) if v is not None]
        if not active:
            return []
        k = min(k, len(active))
        scored = [(self._score(vector, v), i) for i, v in active]
        top = heapq.nlargest(k, scored, key=lambda x: x[0])
        results = []
        for score, idx in top:
            if threshold is not None and score < threshold:
                continue
            results.append(
                SearchResult(index=idx, score=score, metadata=self._metadata[idx])
            )
        return results

    def remove(self, index: int) -> None:
        """Remove a vector by index (marks as None, preserves other indices)."""
        if index < 0 or index >= len(self._vectors):
            raise IndexError(f"Index {index} out of range")
        self._vectors[index] = None
        self._metadata[index] = None

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._vectors.clear()
        self._metadata.clear()
