"""Offline-evaluable, free-tier-only AI enrichment evaluation harness.

This package is intentionally isolated from the production pipeline: it reads
the local seed catalog and taxonomy, writes only to ``scratch/ai_eval/``, and
never mutates DuckDB, Parquet exports, overrides, or the review queue.
"""

from __future__ import annotations

__all__ = [
    "dataset",
    "parsing",
    "prompts",
    "providers",
    "reporting",
    "runner",
    "taxonomy",
]
