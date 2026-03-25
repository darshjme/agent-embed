"""Tests for agent_embed.routing"""
import pytest
from agent_embed.routing import SemanticRouter, Route


def test_basic_routing():
    router = SemanticRouter(default="fallback")
    router.add_route(Route("math", [[1.0, 0.0, 0.0]]))
    router.add_route(Route("code", [[0.0, 1.0, 0.0]]))
    name, score = router.route([0.99, 0.01, 0.0])
    assert name == "math"
    assert score > 0.99


def test_default_when_no_routes():
    router = SemanticRouter(default="fallback")
    name, score = router.route([1.0, 0.0])
    assert name == "fallback"


def test_route_all_sorted():
    router = SemanticRouter()
    router.add_route(Route("a", [[1.0, 0.0]]))
    router.add_route(Route("b", [[0.0, 1.0]]))
    all_routes = router.route_all([0.9, 0.1])
    assert all_routes[0][0] == "a"
    assert all_routes[1][0] == "b"


def test_remove_route():
    router = SemanticRouter(default="fallback")
    router.add_route(Route("math", [[1.0, 0.0]]))
    router.remove_route("math")
    name, _ = router.route([1.0, 0.0])
    assert name == "fallback"


def test_global_threshold():
    router = SemanticRouter(default="fallback", global_threshold=0.999)
    router.add_route(Route("math", [[1.0, 0.0]]))
    # Query is close but not above 0.999
    name, _ = router.route([0.9, 0.1])
    assert name == "fallback"


def test_multiple_embeddings_per_route():
    router = SemanticRouter(default="fallback")
    router.add_route(Route("science", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    name, score = router.route([0.0, 0.99, 0.01])
    assert name == "science"
