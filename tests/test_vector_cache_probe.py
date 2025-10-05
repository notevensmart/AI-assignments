import math

from src.hashmind.vector_cache import VectorCache


def _make_cache():
    cache = VectorCache(dimension=3)
    cache.batch_upsert(
        [
            ("a", [1.0, 0.0, 0.0], {"tag": "red"}),
            ("b", [0.0, 1.0, 0.0], {"tag": "blue"}),
            ("c", [0.0, 0.0, 1.0], {"tag": "blue"}),
        ]
    )
    return cache


def test_similarity_search_returns_sorted_results():
    cache = _make_cache()
    results = cache.similarity_search([0.9, 0.1, 0.0], top_k=2)
    keys = [key for key, *_ in results]
    assert keys == ["a", "b"]
    assert results[0][1] > results[1][1]


def test_similarity_search_metadata_filter():
    cache = _make_cache()
    results = cache.similarity_search(
        [0.1, 0.0, 0.9],
        top_k=5,
        metadata_filter=lambda meta: meta and meta.get("tag") == "blue",
    )
    assert {key for key, *_ in results} == {"b", "c"}


def test_probe_uses_similarity_search():
    cache = _make_cache()
    probe_results = cache.probe([0.0, 1.0, 0.0], probe_k=3)
    assert len(probe_results) == 3
    best_key, score, _ = probe_results[0]
    assert best_key == "b"
    assert math.isclose(score, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_similarity_search_requires_positive_top_k():
    cache = _make_cache()
    try:
        cache.similarity_search([1.0, 0.0, 0.0], top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("top_k <= 0 should raise ValueError")
