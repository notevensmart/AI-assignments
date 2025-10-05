"""Micro-benchmarks for the VectorCache implementation."""

from __future__ import annotations

import random
import time
from typing import Iterable, List, Sequence, Tuple

from src.hashmind.vector_cache import VectorCache


def _random_vector(dimension: int) -> Tuple[float, ...]:
    return tuple(random.random() for _ in range(dimension))


def benchmark_insertions(num_vectors: int = 10_000, dimension: int = 128) -> float:
    cache = VectorCache(dimension=dimension)
    vectors = [_random_vector(dimension) for _ in range(num_vectors)]
    start = time.perf_counter()
    for idx, vector in enumerate(vectors):
        cache.upsert(f"item-{idx}", vector)
    duration = time.perf_counter() - start
    return duration


def benchmark_similarity_search(
    num_vectors: int = 10_000, dimension: int = 128, queries: int = 100
) -> float:
    cache = VectorCache(dimension=dimension)
    cache.batch_upsert(
        (f"item-{idx}", _random_vector(dimension), None)
        for idx in range(num_vectors)
    )
    query_vectors = [_random_vector(dimension) for _ in range(queries)]
    start = time.perf_counter()
    for query in query_vectors:
        cache.similarity_search(query, top_k=5)
    duration = time.perf_counter() - start
    return duration


def main() -> None:
    insert_duration = benchmark_insertions()
    search_duration = benchmark_similarity_search()
    print(f"Insertion benchmark: {insert_duration:.3f}s for 10k vectors")
    print(f"Similarity search benchmark: {search_duration:.3f}s for 100 queries")


if __name__ == "__main__":
    main()
