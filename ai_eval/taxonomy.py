"""Canonical taxonomy loader and validator.

The taxonomy is loaded from ``data/catalogs/taxonomy.json`` at runtime. The
category list is never hardcoded in prompts, datasets, or validation code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "data" / "catalogs" / "taxonomy.json"


class TaxonomyError(ValueError):
    """Raised when the taxonomy file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Taxonomy:
    """Immutable, order-preserving view of the macro -> categories mapping."""

    macros: tuple[str, ...]
    categories: tuple[str, ...]
    category_to_macro: Mapping[str, str]

    def is_category(self, value: object) -> bool:
        return isinstance(value, str) and value in self.category_to_macro

    def macro_for(self, category: str) -> str:
        try:
            return self.category_to_macro[category]
        except KeyError as exc:
            raise TaxonomyError(
                f"Category {category!r} is not part of the canonical taxonomy."
            ) from exc

    def category_index(self, category: str) -> int:
        try:
            return self.categories.index(category)
        except ValueError as exc:
            raise TaxonomyError(
                f"Category {category!r} is not part of the canonical taxonomy."
            ) from exc

    def categories_by_macro(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {macro: [] for macro in self.macros}
        for category in self.categories:
            grouped[self.category_to_macro[category]].append(category)
        return {macro: tuple(items) for macro, items in grouped.items()}

    def prompt_lines(self) -> str:
        """Render the taxonomy as an indented list for a prompt."""
        lines: list[str] = []
        for macro in self.macros:
            lines.append(f"{macro}:")
            for category in self.categories_by_macro()[macro]:
                lines.append(f"  - {category}")
        return "\n".join(lines)


class _DuplicateMemberError(ValueError):
    """Internal marker for a repeated key in a JSON object."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate JSON object member {key!r}")


def _reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``object_pairs_hook`` that refuses to silently drop a duplicate key.

    Plain ``json.loads`` keeps the last value, so ``{"A": [...], "A": [...]}``
    would silently lose a whole macro.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateMemberError(str(key))
        result[key] = value
    return result


def _source_position(raw: str, key: str) -> str:
    """Best-effort ``line/column`` of the first textual occurrence of ``key``."""
    needle = json.dumps(key)
    index = raw.find(needle)
    if index < 0:
        return ""
    line = raw.count("\n", 0, index) + 1
    column = index - raw.rfind("\n", 0, index)
    return f" (first occurrence at line {line}, column {column})"


def _require_non_empty_label(value: object, kind: str, location: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaxonomyError(
            f"{kind} must be a non-empty string in {location}; got {value!r}."
        )
    if value != value.strip():
        raise TaxonomyError(
            f"{kind} {value!r} in {location} has leading/trailing whitespace."
        )
    return value


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load and validate the canonical taxonomy.

    Raises:
        TaxonomyError: on unreadable files, malformed JSON, empty macros or
            categories, duplicate categories, or an empty taxonomy.
    """
    location = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
    try:
        raw = location.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaxonomyError(f"Cannot read taxonomy file {location}: {exc}") from exc

    try:
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_members)
    except _DuplicateMemberError as exc:
        raise TaxonomyError(
            f"Duplicate key {exc.key!r} in {location}"
            f"{_source_position(raw, exc.key)}; a macro may only appear once, "
            "otherwise one of its category lists is silently dropped."
        ) from exc
    except json.JSONDecodeError as exc:
        raise TaxonomyError(
            f"Malformed taxonomy JSON in {location}: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(data, dict) or not data:
        raise TaxonomyError(
            f"Taxonomy in {location} must be a non-empty JSON object mapping "
            "macros to category lists."
        )

    macros: list[str] = []
    categories: list[str] = []
    mapping: dict[str, str] = {}
    seen_categories: set[str] = set()

    for raw_macro, raw_categories in data.items():
        macro = _require_non_empty_label(raw_macro, "Macro name", location)
        if not isinstance(raw_categories, list) or not raw_categories:
            raise TaxonomyError(
                f"Macro {macro!r} in {location} must map to a non-empty list "
                "of categories."
            )
        macros.append(macro)
        for raw_category in raw_categories:
            category = _require_non_empty_label(
                raw_category, "Category name", location
            )
            if category in seen_categories:
                raise TaxonomyError(
                    f"Duplicate category {category!r} in {location}; every "
                    "category must belong to exactly one macro."
                )
            seen_categories.add(category)
            categories.append(category)
            mapping[category] = macro

    if not categories:
        raise TaxonomyError(f"Taxonomy in {location} contains no categories.")

    return Taxonomy(
        macros=tuple(macros),
        categories=tuple(categories),
        category_to_macro=MappingProxyType(mapping),
    )
