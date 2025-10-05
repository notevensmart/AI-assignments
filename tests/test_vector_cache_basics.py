import math

from src.hashmind.vector_cache import VectorCache


def assert_vectors_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert math.isclose(a, e, rel_tol=tol, abs_tol=tol)


def test_upsert_and_get_round_trip():
    cache = VectorCache(dimension=3)
    cache.upsert("doc-1", [1.0, 0.0, 0.0], metadata={"topic": "ai"})

    vector, metadata = cache.get("doc-1")
    assert_vectors_close(vector, (1.0, 0.0, 0.0))
    assert metadata == {"topic": "ai"}


def test_get_missing_returns_none():
    cache = VectorCache(dimension=2)
    assert cache.get("unknown") is None


def test_dimension_mismatch_raises():
    cache = VectorCache(dimension=2)
    try:
        cache.upsert("doc", [1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("dimension mismatch should raise ValueError")


def test_batch_upsert_validates_duplicates():
    cache = VectorCache(dimension=2)
    try:
        cache.batch_upsert([
            ("a", [1.0, 0.0], None),
            ("a", [0.0, 1.0], None),
        ])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate keys should trigger ValueError")


def test_batch_upsert_inserts_all_items():
    cache = VectorCache(dimension=2)
    cache.batch_upsert([
        ("a", [1.0, 0.0], {"lang": "en"}),
        ("b", [0.0, 1.0], {"lang": "fr"}),
    ])
    assert set(cache.keys()) == {"a", "b"}
    assert cache.get("a")[1] == {"lang": "en"}
    assert cache.get("b")[1] == {"lang": "fr"}


def test_delete_removes_key():
    cache = VectorCache(dimension=2)
    cache.upsert("doc", [1.0, 0.0])
    assert cache.delete("doc") is True
    assert cache.get("doc") is None
    assert cache.delete("doc") is False
