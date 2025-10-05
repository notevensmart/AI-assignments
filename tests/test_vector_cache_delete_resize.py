import pytest

from src.hashmind.vector_cache import VectorCache


def test_resize_truncates_old_entries():
    cache = VectorCache(dimension=2)
    cache.batch_upsert(
        [
            ("a", [1.0, 0.0], None),
            ("b", [0.0, 1.0], None),
            ("c", [0.5, 0.5], None),
        ]
    )
    removed = cache.resize(2)
    assert removed == 1
    assert list(cache.keys()) == ["b", "c"]


def test_resize_accepts_growth():
    cache = VectorCache(dimension=2)
    cache.upsert("a", [1.0, 0.0])
    removed = cache.resize(5)
    assert removed == 0
    assert list(cache.keys()) == ["a"]


def test_resize_rejects_negative_sizes():
    cache = VectorCache(dimension=2)
    with pytest.raises(ValueError):
        cache.resize(-1)


def test_delete_then_resize_behaves_consistently():
    cache = VectorCache(dimension=2)
    cache.batch_upsert(
        [
            ("a", [1.0, 0.0], None),
            ("b", [0.0, 1.0], None),
            ("c", [0.5, 0.5], None),
        ]
    )
    cache.delete("b")
    removed = cache.resize(1)
    assert removed == 1
    assert list(cache.keys()) == ["c"]
