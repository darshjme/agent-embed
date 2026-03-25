"""Tests for agent_embed.search"""
import pytest
from agent_embed.search import NearestNeighborSearch, SearchResult


def test_add_and_query_cosine():
    idx = NearestNeighborSearch(metric="cosine")
    idx.add([1.0, 0.0], metadata="a")
    idx.add([0.0, 1.0], metadata="b")
    results = idx.query([1.0, 0.1], k=1)
    assert len(results) == 1
    assert results[0].index == 0
    assert results[0].metadata == "a"


def test_query_empty_returns_empty():
    idx = NearestNeighborSearch()
    assert idx.query([1.0, 0.0]) == []


def test_query_k_larger_than_index():
    idx = NearestNeighborSearch()
    idx.add([1.0, 0.0])
    results = idx.query([1.0, 0.0], k=100)
    assert len(results) == 1


def test_add_batch():
    idx = NearestNeighborSearch()
    indices = idx.add_batch([[1.0, 0.0], [0.0, 1.0]], metadatas=["x", "y"])
    assert indices == [0, 1]
    assert len(idx) == 2


def test_euclidean_metric():
    idx = NearestNeighborSearch(metric="euclidean")
    idx.add([0.0, 0.0])
    idx.add([10.0, 10.0])
    results = idx.query([0.1, 0.1], k=1)
    assert results[0].index == 0


def test_dot_metric():
    idx = NearestNeighborSearch(metric="dot")
    idx.add([1.0, 0.0])
    idx.add([0.0, 1.0])
    results = idx.query([0.9, 0.1], k=1)
    assert results[0].index == 0


def test_threshold_filters():
    idx = NearestNeighborSearch()
    idx.add([1.0, 0.0])
    results = idx.query([0.0, 1.0], k=1, threshold=0.99)
    assert len(results) == 0


def test_remove():
    idx = NearestNeighborSearch()
    idx.add([1.0, 0.0])
    idx.remove(0)
    assert len(idx) == 0


def test_invalid_metric():
    with pytest.raises(ValueError):
        NearestNeighborSearch(metric="l3")


def test_search_result_repr():
    r = SearchResult(index=3, score=0.987654)
    assert "0.9877" in repr(r)
