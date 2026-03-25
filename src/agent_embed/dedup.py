"""
Semantic deduplication using cosine similarity.

Given a stream of embeddings, filters out items that are semantically
too similar to already-seen items.
"""

from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple

from .search import NearestNeighborSearch, SearchResult


class SemanticDeduplicator:
    """Filter near-duplicate items based on embedding similarity.

    Usage::

        dedup = SemanticDeduplicator(threshold=0.92)
        vectors = [[0.1, 0.9], [0.11, 0.89], [0.8, 0.2]]
        unique_indices = dedup.filter(vectors)
        # [0, 2]  — index 1 is a near-duplicate of 0
    """

    def __init__(self, threshold: float = 0.90) -> None:
        """
        Args:
            threshold: Cosine similarity above which two items are considered
                       duplicates. Range [0, 1]. Default 0.90.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self._index = NearestNeighborSearch(metric="cosine")

    def is_duplicate(
        self, vector: Sequence[float]
    ) -> Tuple[bool, Optional[SearchResult]]:
        """Check if a vector is a near-duplicate of any stored vector.

        Returns:
            (is_dup, nearest_result) — nearest_result is None if index is empty.
        """
        if len(self._index) == 0:
            return False, None
        results = self._index.query(vector, k=1)
        if results and results[0].score >= self.threshold:
            return True, results[0]
        return False, None

    def add(self, vector: Sequence[float], metadata: Any = None) -> int:
        """Add a vector without dedup check. Returns index."""
        return self._index.add(vector, metadata)

    def add_if_unique(
        self, vector: Sequence[float], metadata: Any = None
    ) -> Tuple[bool, int]:
        """Add vector only if it's not a duplicate.

        Returns:
            (was_added, index) — if not added, index is the duplicate's index.
        """
        is_dup, nearest = self.is_duplicate(vector)
        if is_dup:
            assert nearest is not None
            return False, nearest.index
        idx = self._index.add(vector, metadata)
        return True, idx

    def filter(
        self,
        vectors: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Any]] = None,
    ) -> List[int]:
        """Filter a batch of vectors, returning indices of unique ones.

        Processes in order; earlier items take precedence over later duplicates.
        """
        if metadatas is None:
            metadatas = [None] * len(vectors)
        unique_indices: List[int] = []
        for i, (vec, meta) in enumerate(zip(vectors, metadatas)):
            added, _ = self.add_if_unique(vec, meta)
            if added:
                unique_indices.append(i)
        return unique_indices

    def reset(self) -> None:
        """Clear all stored vectors."""
        self._index.clear()

    @property
    def size(self) -> int:
        """Number of unique items stored."""
        return len(self._index)
