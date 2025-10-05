# Hashmind Vector Cache Assignment

This repository contains the starter code, tests, and documentation for the Hashmind Vector Cache assignment. The goal of the assignment is to implement an in-memory vector cache that supports inserting, retrieving, deleting, and querying high-dimensional vectors by similarity.

## Repository Layout

- `docs/ASSIGNMENT.md` – detailed assignment brief and step-by-step instructions.
- `docs/REPORT.md` – scaffold for the final report students must submit.
- `src/hashmind/vector_cache.py` – starter implementation that students will complete.
- `tests/` – automated unit tests used to evaluate submissions.
- `benchmarks/benchmarks.py` – optional micro-benchmarks to profile the cache.
- `requirements.txt` – Python dependencies required to run the starter code and tests.

## Getting Started

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run the unit tests to see the current status:

   ```bash
   pytest
   ```

3. Implement the missing functionality in `src/hashmind/vector_cache.py` until all tests pass.

For more details, read the assignment brief in `docs/ASSIGNMENT.md` and provide your findings in `docs/REPORT.md`.
