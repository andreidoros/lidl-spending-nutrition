"""Deterministic, stratified evaluation dataset built from the local seed.

Reference answers stay in :class:`EvalProduct` and are never part of
:meth:`EvalProduct.to_model_payload` / :meth:`EvalDataset.model_payload`.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .constants import (
    ALCOHOL_CATEGORY,
    CONFIDENCE_TIERS,
    HIGH_CONFIDENCE_STATUSES,
    MEDIUM_CONFIDENCE_STATUSES,
    NON_FOOD_CATEGORY,
    NUTRITION_FIELDS,
    REQUIRED_NUTRITION_FIELDS,
)
from .taxonomy import DEFAULT_TAXONOMY_PATH, PROJECT_ROOT, Taxonomy, load_taxonomy

DEFAULT_SEED_PATH = PROJECT_ROOT / "data" / "catalogs" / "ro" / "catalog_seed.csv"
DEFAULT_SAMPLE_LIMIT = 100
DEFAULT_RANDOM_SEED = 20240517

REQUIRED_SEED_COLUMNS = (
    "product_key",
    "sample_name",
    "macro_category",
    "category",
    "package_weight_g",
    "match_status",
)


class DatasetError(ValueError):
    """Raised when the seed catalog is missing or structurally unusable."""


def confidence_for(match_status: str | None) -> str:
    """Map a seed ``match_status`` to HIGH / MEDIUM / LOW confidence."""
    status = (match_status or "").strip().upper()
    if status in HIGH_CONFIDENCE_STATUSES:
        return "HIGH"
    if status in MEDIUM_CONFIDENCE_STATUSES:
        return "MEDIUM"
    return "LOW"


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def is_nutrition_comparable(product: "EvalProduct") -> bool:
    """Whether reference nutrition is usable for MAE comparisons.

    Only non-LOW confidence rows with every required macro value finite and a
    food category are comparable. Non-Food is always excluded. ``NaN`` /
    ``Infinity`` references must never enter MAE.
    """
    if product.reference_confidence == "LOW":
        return False
    if product.non_food or product.reference_category == NON_FOOD_CATEGORY:
        return False
    return all(
        _is_finite_number(product.nutrition.get(key))
        for key in REQUIRED_NUTRITION_FIELDS
    )


def _to_float(value: object) -> float | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_int(value: object) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except (OverflowError, ValueError):
        return None


def _to_bool(value: object) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    return text in {"1", "true", "t", "yes", "y"}


def _hardness_score(
    name: str,
    category: str,
    non_food: bool,
    package_weight_g: float | None,
) -> int:
    """Heuristic priority for genuinely difficult real seed rows.

    Higher scores are sampled first within each category. This deliberately
    prefers abbreviated counter names, percentages, alcohol, non-food, and
    products without a recorded package weight, while never inventing rows.
    """
    lowered = name.lower()
    score = 0
    if "%" in name:
        score += 3
    if non_food or category == NON_FOOD_CATEGORY:
        score += 2
    if category == ALCOHOL_CATEGORY:
        score += 2
    if any(
        token in lowered
        for token in ("buc_", "dez.", "sort.", " pm_", "pm_", "sk", "xxl", "3g#", "baut.")
    ):
        score += 2
    digit_count = sum(1 for char in name if char.isdigit())
    if len(name) <= 8 or digit_count >= 4:
        score += 1
    if package_weight_g is None:
        score += 1
    return score


@dataclass(frozen=True)
class EvalProduct:
    product_key: str
    sample_name: str
    package_weight_g: float | None
    reference_category: str
    reference_macro: str
    reference_confidence: str
    match_status: str
    nutrition: Mapping[str, float | None] = field(default_factory=dict)
    non_food: bool = False
    times_bought: int | None = None
    hardness: int = 0

    def to_model_payload(self) -> dict[str, object]:
        """Only the minimum fields a model is allowed to see."""
        payload: dict[str, object] = {
            "product_key": self.product_key,
            "sample_name": self.sample_name,
        }
        if self.package_weight_g is not None:
            payload["package_weight_g"] = self.package_weight_g
        return payload

    def reference_snapshot(self) -> dict[str, object]:
        return {
            "category": self.reference_category,
            "macro_category": self.reference_macro,
            "confidence": self.reference_confidence,
            "match_status": self.match_status,
            "non_food": self.non_food,
            "nutrition": dict(self.nutrition),
            "nutrition_comparable": is_nutrition_comparable(self),
        }


@dataclass(frozen=True)
class EvalDataset:
    products: tuple[EvalProduct, ...]
    seed: int
    source_path: str
    requested_limit: int

    def model_payload(self) -> list[dict[str, object]]:
        return [product.to_model_payload() for product in self.products]

    @property
    def product_keys(self) -> tuple[str, ...]:
        return tuple(product.product_key for product in self.products)

    def by_key(self) -> dict[str, EvalProduct]:
        return {product.product_key: product for product in self.products}

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for product in self.products:
            counts[product.reference_category] = (
                counts.get(product.reference_category, 0) + 1
            )
        return counts

    def confidence_counts(self) -> dict[str, int]:
        counts = {tier: 0 for tier in CONFIDENCE_TIERS}
        for product in self.products:
            counts[product.reference_confidence] += 1
        return counts

    def comparable_nutrition_count(self) -> int:
        return sum(1 for product in self.products if is_nutrition_comparable(product))


def _read_seed_rows(csv_path: Path) -> list[dict[str, str]]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetError(f"Seed catalog {csv_path} has no header row.")
            missing = [col for col in REQUIRED_SEED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise DatasetError(
                    f"Seed catalog {csv_path} is missing required columns: "
                    + ", ".join(missing)
                )
            return list(reader)
    except OSError as exc:
        raise DatasetError(f"Cannot read seed catalog {csv_path}: {exc}") from exc


def _build_product(
    row: Mapping[str, str],
    taxonomy: Taxonomy,
) -> EvalProduct | None:
    product_key = (row.get("product_key") or "").strip()
    sample_name = (row.get("sample_name") or "").strip()
    category = (row.get("category") or "").strip()
    if not product_key or not sample_name or not taxonomy.is_category(category):
        return None

    match_status = (row.get("match_status") or "").strip()
    non_food = _to_bool(row.get("non_food")) or category == NON_FOOD_CATEGORY
    nutrition = {
        field_name: _to_float(row.get(field_name))
        for field_name in NUTRITION_FIELDS
    }
    package_weight = _to_float(row.get("package_weight_g"))
    return EvalProduct(
        product_key=product_key,
        sample_name=sample_name,
        package_weight_g=package_weight,
        reference_category=category,
        # The taxonomy is authoritative; the seed's macro is not trusted.
        reference_macro=taxonomy.macro_for(category),
        reference_confidence=confidence_for(match_status),
        match_status=match_status,
        nutrition=nutrition,
        non_food=non_food,
        times_bought=_to_int(row.get("times_bought")),
        hardness=_hardness_score(sample_name, category, non_food, package_weight),
    )


def build_eval_dataset(
    limit: int = DEFAULT_SAMPLE_LIMIT,
    seed: int = DEFAULT_RANDOM_SEED,
    csv_path: str | Path | None = None,
    taxonomy: Taxonomy | None = None,
) -> EvalDataset:
    """Build a deterministic stratified sample of the local seed catalog.

    Sampling is round-robin across taxonomy categories (randomly rotated once
    per fixed seed) so that every category has a chance to appear even for
    small ``limit`` values. Within a category, difficult real rows are
    prioritized, with seeded random tie-breaking.
    """
    if not isinstance(limit, int) or limit < 1:
        raise DatasetError("limit must be a positive integer.")
    path = Path(csv_path) if csv_path is not None else DEFAULT_SEED_PATH
    tax = taxonomy if taxonomy is not None else load_taxonomy(DEFAULT_TAXONOMY_PATH)

    rows = _read_seed_rows(path)
    unique: dict[str, EvalProduct] = {}
    for row in rows:
        product = _build_product(row, tax)
        if product is None:
            continue
        # Keep the first occurrence of a product key: stable and unambiguous.
        unique.setdefault(product.product_key, product)

    if not unique:
        raise DatasetError(f"No usable rows found in seed catalog {path}.")

    by_category: dict[str, list[EvalProduct]] = {}
    for product in unique.values():
        by_category.setdefault(product.reference_category, []).append(product)

    rng = random.Random(seed)
    prepared: dict[str, list[EvalProduct]] = {}
    for category in tax.categories:
        items = by_category.get(category, [])
        if not items:
            continue
        # Stable base order, seeded shuffle, then stable hardness sort.
        items = sorted(items, key=lambda product: product.product_key)
        rng.shuffle(items)
        items.sort(key=lambda product: product.hardness, reverse=True)
        prepared[category] = items

    category_order = list(tax.categories)
    offset = rng.randrange(len(category_order))
    rotation = category_order[offset:] + category_order[:offset]
    cursors = {category: 0 for category in category_order}
    selected: list[EvalProduct] = []

    while len(selected) < limit:
        progressed = False
        for category in rotation:
            if len(selected) >= limit:
                break
            items = prepared.get(category, [])
            cursor = cursors[category]
            if cursor < len(items):
                selected.append(items[cursor])
                cursors[category] = cursor + 1
                progressed = True
        if not progressed:
            break

    order_index = {category: idx for idx, category in enumerate(tax.categories)}
    selected.sort(
        key=lambda product: (
            order_index[product.reference_category],
            product.product_key,
        )
    )

    return EvalDataset(
        products=tuple(selected),
        seed=seed,
        source_path=str(path),
        requested_limit=limit,
    )


def chunked(items: Sequence[EvalProduct], size: int) -> Iterable[list[EvalProduct]]:
    if size < 1:
        raise DatasetError("batch size must be a positive integer.")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
