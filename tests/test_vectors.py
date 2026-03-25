"""Tests for agent_embed.vectors"""
import math
import pytest
from agent_embed.vectors import (
    magnitude, normalize, dot_product,
    cosine_similarity, euclidean_distance, manhattan_distance, cosine_distance,
)


def test_magnitude_basic():
    assert math.isclose(magnitude([3.0, 4.0]), 5.0)


def test_magnitude_unit():
    assert math.isclose(magnitude([1.0, 0.0, 0.0]), 1.0)


def test_normalize_unit_vector():
    result = normalize([3.0, 4.0])
    assert math.isclose(result[0], 0.6)
    assert math.isclose(result[1], 0.8)
    assert math.isclose(magnitude(result), 1.0)


def test_normalize_zero_raises():
    with pytest.raises(ValueError, match="zero"):
        normalize([0.0, 0.0])


def test_dot_product():
    assert math.isclose(dot_product([1, 2, 3], [4, 5, 6]), 32.0)


def test_dot_product_dimension_mismatch():
    with pytest.raises(ValueError):
        dot_product([1, 2], [1, 2, 3])


def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert math.isclose(cosine_similarity(v, v), 1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-10)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert math.isclose(cosine_similarity(a, b), -1.0)


def test_cosine_similarity_zero_raises():
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])


def test_euclidean_distance():
    assert math.isclose(euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0)


def test_euclidean_distance_same():
    v = [1.0, 2.0, 3.0]
    assert math.isclose(euclidean_distance(v, v), 0.0)


def test_manhattan_distance():
    assert math.isclose(manhattan_distance([0.0, 0.0], [3.0, 4.0]), 7.0)


def test_cosine_distance():
    v = [1.0, 0.0]
    assert math.isclose(cosine_distance(v, v), 0.0)
    assert math.isclose(cosine_distance(v, [-1.0, 0.0]), 2.0)
