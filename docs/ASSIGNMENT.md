# Hashmind Vector Cache – Assignment Brief

Welcome to the Hashmind Vector Cache assignment. Your objective is to design and implement an efficient in-memory cache for dense vector embeddings. The cache will be used in retrieval-augmented generation (RAG) pipelines where low-latency similarity lookups are critical.

## Learning Goals

- Practice designing a clean API for vector storage.
- Implement cosine similarity search over an in-memory collection.
- Maintain cache hygiene with deletion, resizing, and metadata filtering.
- Write tests and benchmarks to evaluate performance and correctness.

## Deliverables

1. **Implementation** – Complete the TODO sections in `src/hashmind/vector_cache.py`.
2. **Unit Tests** – Ensure the provided tests in `tests/` all pass. You may add additional tests if needed.
3. **Benchmarks** – Run the micro-benchmark script to profile your solution (optional but recommended).
4. **Report** – Summarise your design decisions, trade-offs, and benchmark results in `docs/REPORT.md`.

## Functional Requirements

Your vector cache must support the following operations:

- `upsert(key, vector, metadata=None)` – Insert or replace a vector. Reject vectors whose dimension does not match the cache configuration. Keys are unique.
- `get(key)` – Retrieve the stored vector and metadata for a key.
- `delete(key)` – Remove a key and its data from the cache. Return a boolean indicating whether the key existed.
- `batch_upsert(items)` – Insert a sequence of `(key, vector, metadata)` triples atomically. The operation should validate the input before mutating the cache.
- `similarity_search(query, top_k=5, metadata_filter=None)` – Return the `top_k` most similar entries by cosine similarity. If a `metadata_filter` callable is provided, only consider entries whose metadata satisfies the predicate.
- `probe(query, probe_k)` – Perform a similarity search that also returns the raw similarity scores for the `probe_k` nearest neighbours. This is used to evaluate the health of the index.
- `resize(max_size)` – Limit the cache to `max_size` entries by removing the least-recently inserted items beyond the limit. The method should return the number of items removed.

The cache should preserve insertion order to support deterministic resizing.

## Constraints & Expectations

- Use NumPy for vector operations.
- Focus on code clarity and correctness over micro-optimisations.
- Raise informative `ValueError` exceptions for invalid inputs (dimension mismatch, duplicate keys in a batch, etc.).
- Document any assumptions directly in the code.

## Getting Started

1. Review the starter code in `src/hashmind/vector_cache.py` and familiarise yourself with the API skeleton.
2. Run `pytest` to see the currently failing tests.
3. Implement the functionality iteratively, starting with the basic CRUD operations before tackling similarity search and resizing.
4. Once all tests pass, optionally run `python benchmarks/benchmarks.py` to gather timing information.
5. Write your report in `docs/REPORT.md`.

Good luck, and have fun building!
