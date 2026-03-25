# agent-embed

Zero-dependency embedding utilities for LLM agents — pure Python.

**Cosine similarity · Vector normalization · Nearest-neighbor search · Semantic dedup · Semantic routing**

No numpy. No scipy. No external dependencies of any kind.

## Install

```bash
pip install agent-embed
```

## Quick Start

```python
from agent_embed import cosine_similarity, normalize, NearestNeighborSearch, SemanticRouter, Route

# Vector math
a, b = [1.0, 0.0, 0.5], [0.9, 0.1, 0.5]
print(cosine_similarity(a, b))   # 0.9994...

# Nearest-neighbor search
index = NearestNeighborSearch(metric="cosine")
index.add([1.0, 0.0], metadata={"text": "math question"})
index.add([0.0, 1.0], metadata={"text": "code question"})
results = index.query([0.95, 0.05], k=1)
# [SearchResult(index=0, score=0.9994)]

# Semantic routing
router = SemanticRouter(default="fallback")
router.add_route(Route("math", [[1.0, 0.0, 0.0]]))
router.add_route(Route("code", [[0.0, 1.0, 0.0]]))
name, score = router.route([0.98, 0.02, 0.0])
# ("math", 0.999...)
```

## Features

| Module | What it provides |
|--------|-----------------|
| `vectors` | `cosine_similarity`, `dot_product`, `normalize`, `magnitude`, `euclidean_distance`, `manhattan_distance` |
| `search` | `NearestNeighborSearch` — brute-force k-NN with cosine/euclidean/dot metrics |
| `dedup` | `SemanticDeduplicator` — filter near-duplicate embeddings by threshold |
| `routing` | `SemanticRouter`, `Route` — map queries to named handlers |

## Zero Dependencies

Only the Python standard library (`math`, `heapq`, `dataclasses`).
