"""Tests for agent_embed.dedup"""
import pytest
from agent_embed.dedup import SemanticDeduplicator


def test_filter_removes_near_duplicates():
    dedup = SemanticDeduplicator(threshold=0.95)
    v1 = [1.0, 0.0]
    v2 = [0.999, 0.001]   # near-duplicate of v1
    v3 = [0.0, 1.0]       # different direction
    unique = dedup.filter([v1, v2, v3])
    assert 0 in unique
    assert 1 not in unique
    assert 2 in unique


def test_first_item_always_unique():
    dedup = SemanticDeduplicator(threshold=0.99)
    result = dedup.filter([[1.0, 0.0]])
    assert result == [0]


def test_add_if_unique():
    dedup = SemanticDeduplicator(threshold=0.95)
    added1, idx1 = dedup.add_if_unique([1.0, 0.0])
    assert added1 is True
    added2, idx2 = dedup.add_if_unique([0.999, 0.001])
    assert added2 is False
    assert idx2 == idx1


def test_is_duplicate_empty():
    dedup = SemanticDeduplicator()
    is_dup, nearest = dedup.is_duplicate([1.0, 0.0])
    assert is_dup is False
    assert nearest is None


def test_reset():
    dedup = SemanticDeduplicator(threshold=0.99)
    dedup.add([1.0, 0.0])
    assert dedup.size == 1
    dedup.reset()
    assert dedup.size == 0


def test_invalid_threshold():
    with pytest.raises(ValueError):
        SemanticDeduplicator(threshold=1.5)
