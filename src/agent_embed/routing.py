"""
Semantic routing: map an input embedding to the best matching Route.

Useful for directing LLM agent queries to specialized handlers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from .vectors import cosine_similarity


@dataclass
class Route:
    """A named route with one or more reference embeddings."""

    name: str
    embeddings: List[List[float]]
    metadata: Any = None
    threshold: float = 0.0  # per-route minimum similarity

    def best_score(self, query: Sequence[float]) -> float:
        """Return the highest cosine similarity between query and any reference."""
        if not self.embeddings:
            return -1.0
        return max(cosine_similarity(query, ref) for ref in self.embeddings)


class SemanticRouter:
    """Route queries to named handlers based on embedding similarity.

    Usage::

        router = SemanticRouter(default="fallback")
        router.add_route(Route("math", [[0.1, 0.9, 0.2]]))
        router.add_route(Route("code", [[0.8, 0.1, 0.5]]))

        name, score = router.route([0.12, 0.88, 0.25])
        # ("math", 0.998...)
    """

    def __init__(
        self, default: str = "default", global_threshold: float = 0.0
    ) -> None:
        self.default = default
        self.global_threshold = global_threshold
        self._routes: List[Route] = []

    def add_route(self, route: Route) -> None:
        """Register a route."""
        self._routes.append(route)

    def remove_route(self, name: str) -> None:
        """Remove a route by name."""
        self._routes = [r for r in self._routes if r.name != name]

    def route(self, query: Sequence[float]) -> Tuple[str, float]:
        """Return (route_name, score) for the best-matching route.

        Falls back to self.default if no route meets threshold.
        """
        best_name = self.default
        best_score = -1.0

        for route in self._routes:
            score = route.best_score(query)
            min_score = max(self.global_threshold, route.threshold)
            if score > best_score and score >= min_score:
                best_score = score
                best_name = route.name

        return best_name, best_score

    def route_all(self, query: Sequence[float]) -> List[Tuple[str, float]]:
        """Return all routes sorted by descending score."""
        scored = [(r.name, r.best_score(query)) for r in self._routes]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    @property
    def routes(self) -> List[Route]:
        return list(self._routes)
