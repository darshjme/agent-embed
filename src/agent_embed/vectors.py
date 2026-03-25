"""
Pure-Python vector math operations for embedding utilities.
No numpy, no scipy — just math.
"""

from __future__ import annotations
import math
from typing import List, Sequence


Vector = List[float]


def magnitude(v: Sequence[float]) -> float:
    """Return the L2 (Euclidean) magnitude of a vector."""
    return math.sqrt(sum(x * x for x in v))


def normalize(v: Sequence[float]) -> List[float]:
    """Return the unit vector (L2-normalized) of v.

    Raises ValueError if the zero vector is given.
    """
    mag = magnitude(v)
    if mag == 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    return [x / mag for x in v]


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity in [-1, 1] between vectors a and b.

    cos(theta) = (a . b) / (|a| * |b|)
    """
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0.0 or mag_b == 0.0:
        raise ValueError("Cannot compute cosine similarity with zero vector.")
    return dot_product(a, b) / (mag_a * mag_b)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean (L2) distance between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Manhattan (L1) distance between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    return sum(abs(x - y) for x, y in zip(a, b))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine distance (1 - cosine_similarity) in [0, 2]."""
    return 1.0 - cosine_similarity(a, b)
