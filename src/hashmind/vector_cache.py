"""Vector cache implementation for the Hashmind assignment."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class _VectorRecord:
    vector: Tuple[float, ...]
    metadata: Optional[dict]
    norm: float


def _to_vector(values: Sequence[float], dimension: int) -> Tuple[float, ...]:
    vector = tuple(float(v) for v in values)
    if len(vector) != dimension:
        raise ValueError(
            f"expected vector of length {dimension} but received length {len(vector)}"
        )
    if not vector:
        raise ValueError("vector dimension must be greater than zero")
    return vector


def _norm(vector: Tuple[float, ...]) -> float:
    total = sum(component * component for component in vector)
    magnitude = sqrt(total)
    if magnitude == 0:
        raise ValueError("vector norm must be > 0 for cosine similarity")
    return magnitude


def _dot(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    return sum(l * r for l, r in zip(left, right))


class VectorCache:
    """An in-memory cache that stores dense vectors with optional metadata.

    The cache preserves insertion order and supports cosine similarity queries.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._dimension = dimension
        self._store: "OrderedDict[str, _VectorRecord]" = OrderedDict()

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)

    def keys(self) -> Iterator[str]:
        return iter(self._store.keys())

    def upsert(
        self, key: str, vector: Sequence[float], metadata: Optional[dict] = None
    ) -> None:
        record = self._validate_and_create_record(vector, metadata)
        if key in self._store:
            del self._store[key]
        self._store[key] = record

    def batch_upsert(
        self, items: Iterable[Tuple[str, Sequence[float], Optional[dict]]]
    ) -> None:
        materialised = list(items)
        keys = [key for key, *_ in materialised]
        if len(set(keys)) != len(keys):
            raise ValueError("batch_upsert received duplicate keys")

        records = []
        for key, vector, metadata in materialised:
            records.append((key, self._validate_and_create_record(vector, metadata)))

        for key, record in records:
            if key in self._store:
                del self._store[key]
            self._store[key] = record

    def get(self, key: str) -> Optional[Tuple[Tuple[float, ...], Optional[dict]]]:
        record = self._store.get(key)
        if record is None:
            return None
        metadata_copy = None if record.metadata is None else dict(record.metadata)
        return tuple(record.vector), metadata_copy

    def delete(self, key: str) -> bool:
        try:
            del self._store[key]
        except KeyError:
            return False
        return True

    def similarity_search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Callable[[Optional[dict]], bool]] = None,
    ) -> List[Tuple[str, float, Optional[dict]]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector, query_norm = self._normalise(query)
        scores: List[Tuple[str, float, Optional[dict]]] = []
        for key, record in self._store.items():
            if metadata_filter is not None and not metadata_filter(record.metadata):
                continue
            similarity = _dot(query_vector, record.vector) / (query_norm * record.norm)
            scores.append((key, similarity, record.metadata))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[: min(top_k, len(scores))]

    def probe(
        self,
        query: Sequence[float],
        probe_k: int,
        metadata_filter: Optional[Callable[[Optional[dict]], bool]] = None,
    ) -> List[Tuple[str, float, Optional[dict]]]:
        return self.similarity_search(query, top_k=probe_k, metadata_filter=metadata_filter)

    def resize(self, max_size: int) -> int:
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        removed = 0
        while len(self._store) > max_size:
            self._store.popitem(last=False)
            removed += 1
        return removed

    def _validate_and_create_record(
        self, vector: Sequence[float], metadata: Optional[dict]
    ) -> _VectorRecord:
        vector_tuple = _to_vector(vector, self._dimension)
        magnitude = _norm(vector_tuple)
        metadata_copy = None if metadata is None else dict(metadata)
        return _VectorRecord(vector=vector_tuple, metadata=metadata_copy, norm=magnitude)

    def _normalise(self, vector: Sequence[float]) -> Tuple[Tuple[float, ...], float]:
        vector_tuple = _to_vector(vector, self._dimension)
        magnitude = _norm(vector_tuple)
        return vector_tuple, magnitude
