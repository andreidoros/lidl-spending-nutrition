"""Build and safely import offline chatbot batches for product review."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import duckdb

from ai_eval.constants import NUTRITION_RANGES
from ai_eval.parsing import ParseError, parse_model_output, validate_response
from ai_eval.taxonomy import Taxonomy
from product_store import (
    FIELD_NAMES,
    NUTRIENT_FIELDS,
    PACKAGE_FIELD,
    REQUIRED_FOOD_NUTRIENTS,
    missing_required_fields,
    save_ai_enrichment_result,
)
from product_store.schema import METADATA_TABLE

DEFAULT_CHAT_BATCH_PATH = Path("ai_chat_batch.txt")
_PROMPT_FIELD_ORDER = ("category", *NUTRIENT_FIELDS, PACKAGE_FIELD)
_NUTRITION_ISSUES = frozenset({
    "nutrition_not_numeric",
    "nutrition_out_of_range",
    "macro_sum_exceeds_105",
    "sugars_exceeds_carbs",
    "saturated_fat_exceeds_fat",
    "atwater_mismatch",
    "non_food_nutrition",
})


@dataclass(frozen=True)
class ChatBatchIssue:
    code: str
    message: str
    product_key: str | None = None
    field_name: str | None = None


@dataclass(frozen=True)
class ChatBatchSuggestion:
    product_key: str
    category: str | None = None
    nutrients: Mapping[str, float] | None = None
    package_weight_g: float | None = None


@dataclass(frozen=True)
class ChatBatchPreview:
    suggestions: tuple[ChatBatchSuggestion, ...]
    issues: tuple[ChatBatchIssue, ...]
    skipped: tuple[ChatBatchIssue, ...]

    @property
    def fields_suggested(self) -> int:
        return sum(
            int(suggestion.category is not None)
            + len(suggestion.nutrients or {})
            + int(suggestion.package_weight_g is not None)
            for suggestion in self.suggestions
        )


@dataclass(frozen=True)
class ChatBatchSaveReport:
    products_updated: int
    fields_saved: int
    preview: ChatBatchPreview


def _protected_fields(con: duckdb.DuckDBPyConnection) -> dict[str, set[str]]:
    return {
        key: fields
        for key, fields in _group_protection_rows(
            con.execute(
                f"SELECT product_key, field_name FROM {METADATA_TABLE} WHERE is_protected"
            ).fetchall()
        ).items()
    }


def _group_protection_rows(rows: list[tuple[str, str]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for key, field_name in rows:
        grouped.setdefault(str(key), set()).add(field_name)
    return grouped


def _all_products(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, object]]:
    fields = ("product_key", "sample_name", *FIELD_NAMES)
    rows = con.execute(
        f"SELECT {', '.join(fields)} FROM nutrition_cache ORDER BY product_key"
    ).fetchall()
    return {
        str(raw[0]): dict(zip(fields, raw))
        for raw in rows
        if raw[0] is not None
    }


def _requested_fields(
    product: Mapping[str, object],
    category_to_macro: Mapping[str, str],
    protected: set[str],
) -> tuple[str, ...]:
    missing = set(missing_required_fields(product, category_to_macro))
    if not missing:
        return ()

    category = product.get("category")
    category_is_known = isinstance(category, str) and category in category_to_macro
    category_is_blank = category is None or (
        isinstance(category, str) and not category.strip()
    )
    fields: list[str] = []
    # Automatic import fills unknown (NULL/blank) values only. A nonblank but
    # noncanonical category needs an explicit human correction, not overwrite.
    if "category" in missing and category_is_blank and "category" not in protected:
        fields.append("category")

    # Nutrition can be estimated only when the product is already Food or its
    # missing category is eligible to be supplied in the same response.
    can_request_nutrition = (
        category_is_known and category != "Non-Food"
    ) or "category" in fields
    if can_request_nutrition and category != "Non-Food":
        for field_name in REQUIRED_FOOD_NUTRIENTS:
            if product.get(field_name) is None and field_name not in protected:
                fields.append(field_name)
        if product.get("fiber_100g") is None and "fiber_100g" not in protected:
            fields.append("fiber_100g")

    if product.get(PACKAGE_FIELD) is None and PACKAGE_FIELD not in protected:
        fields.append(PACKAGE_FIELD)
    return tuple(field_name for field_name in _PROMPT_FIELD_ORDER if field_name in fields)


def _batch_products(
    con: duckdb.DuckDBPyConnection,
    category_to_macro: Mapping[str, str],
) -> list[dict[str, object]]:
    protected = _protected_fields(con)
    prepared: list[dict[str, object]] = []
    for product in _all_products(con).values():
        if not missing_required_fields(product, category_to_macro):
            continue
        key = str(product["product_key"])
        fields = _requested_fields(product, category_to_macro, protected.get(key, set()))
        if not fields:
            continue
        prepared.append({
            "product_key": key,
            "sample_name": str(product.get("sample_name") or ""),
            "fields_to_fill": fields,
        })
    prepared.sort(key=lambda item: (
        str(item["sample_name"]).casefold(),
        str(item["product_key"]),
    ))
    return prepared


def build_chat_batch_text(
    con: duckdb.DuckDBPyConnection,
    taxonomy: Taxonomy,
    category_to_macro: Mapping[str, str],
) -> str | None:
    """Return a readable prompt containing only unresolved, eligible fields."""
    products = _batch_products(con, category_to_macro)
    if not products:
        return None
    lines = [
        "Help complete the listed grocery products. Use only the supplied product names and taxonomy; do not invent sources.",
        "Return only one JSON object with a 'products' array. Keep product_key exactly as the input text.",
        "For each product, return only product_key and requested fields. Do not return fields not listed in fields_to_fill.",
        "For category, use an exact canonical taxonomy value or null if you cannot classify it. Do not return macro_category; it is derived.",
        "Nutrition values are per 100 g: kcal_100g is kcal; other nutrition fields are grams. Return JSON numbers or null.",
        "package_weight_g, when requested, is a number in grams or null. Null means unknown and never clears a database value.",
        "For Non-Food, set category to 'Non-Food' and do not estimate nutrition.",
        "",
        "Canonical taxonomy:",
        taxonomy.prompt_lines(),
        "",
        "Products and requested fields:",
        json.dumps(products, ensure_ascii=False, indent=2, separators=(",", ": ")),
        "",
        "Response shape: one JSON object with a 'products' array. Each product object contains product_key as a JSON string and only the requested field/value pairs.",
    ]
    return "\n".join(lines)


def refresh_chat_batch_file(
    con: duckdb.DuckDBPyConnection,
    taxonomy: Taxonomy,
    category_to_macro: Mapping[str, str],
    output_path: Path | str = DEFAULT_CHAT_BATCH_PATH,
) -> int:
    """Atomically refresh the derived chat batch; remove stale empty batches."""
    path = Path(output_path)
    prompt = build_chat_batch_text(con, taxonomy, category_to_macro)
    if prompt is None:
        if path.exists():
            path.unlink()
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(prompt)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return len(_batch_products(con, category_to_macro))


def preview_chat_batch_import(
    con: duckdb.DuckDBPyConnection,
    response_text: str,
    taxonomy: Taxonomy,
    category_to_macro: Mapping[str, str],
    blocked_fields: Mapping[str, Iterable[str]] | None = None,
) -> ChatBatchPreview:
    """Validate a pasted response without writing values or task state."""
    try:
        payload = parse_model_output(response_text)
    except (ParseError, TypeError, ValueError, RecursionError):
        return ChatBatchPreview((), (ChatBatchIssue("malformed_json", "The response is not valid products JSON."),), ())

    products = payload.get("products")
    if not isinstance(products, list):
        return ChatBatchPreview((), (ChatBatchIssue("invalid_products", "The response must contain a products array."),), ())

    current_products = _all_products(con)
    protected = _protected_fields(con)
    entries_by_key: dict[str, list[Mapping[str, object]]] = {}
    issues: list[ChatBatchIssue] = []
    skipped: list[ChatBatchIssue] = []
    for index, entry in enumerate(products):
        if not isinstance(entry, Mapping):
            issues.append(ChatBatchIssue("invalid_product_entry", f"Product entry {index + 1} is not an object."))
            continue
        key = entry.get("product_key")
        if not isinstance(key, str) or not key:
            issues.append(ChatBatchIssue("invalid_product_key", f"Product entry {index + 1} has no text product_key."))
            continue
        if key not in current_products:
            issues.append(ChatBatchIssue("unknown_product", "The product_key does not exist in the current database.", key))
            continue
        entries_by_key.setdefault(key, []).append(entry)

    duplicates = {key for key, entries in entries_by_key.items() if len(entries) > 1}
    for key in sorted(duplicates):
        issues.append(ChatBatchIssue("duplicate_product", "This product_key appears more than once; none of its suggestions will be used.", key))

    suggestions: list[ChatBatchSuggestion] = []
    allowed_response_fields = {"product_key", "category", "abstain_reason", *NUTRIENT_FIELDS, PACKAGE_FIELD}
    for key in sorted(entries_by_key):
        if key in duplicates:
            continue
        entry = entries_by_key[key][0]
        product = current_products[key]
        protected_fields = protected.get(key, set())
        requested = set(_requested_fields(product, category_to_macro, protected_fields))
        requested.difference_update(
            blocked_fields.get(key, ()) if blocked_fields is not None else ()
        )
        current_category = product.get("category")
        current_category_is_blank = current_category is None or (
            isinstance(current_category, str) and not current_category.strip()
        )
        invalid_nonblank_category = (
            not current_category_is_blank
            and not (
                isinstance(current_category, str)
                and current_category in category_to_macro
            )
        )
        if invalid_nonblank_category:
            issues.append(ChatBatchIssue(
                "category_not_replaceable",
                "The existing nonblank category is not canonical; correct it manually before importing category or nutrition suggestions.",
                key,
                "category",
            ))
        for field_name in entry:
            if field_name not in allowed_response_fields:
                issues.append(ChatBatchIssue("unexpected_field", "The response contains a field that cannot be imported.", key, str(field_name)))
        stale_fields = ({"category", *NUTRIENT_FIELDS, PACKAGE_FIELD} & set(entry)) - requested
        for field_name in sorted(stale_fields):
            if field_name == "category" and invalid_nonblank_category:
                continue
            skipped.append(ChatBatchIssue("field_no_longer_eligible", "The field is no longer missing, is protected, or was not requested; it will be skipped.", key, field_name))

        category_value: str | None = None
        effective_category = current_category if isinstance(current_category, str) and current_category in category_to_macro else None
        if "category" in requested:
            if "category" not in entry:
                issues.append(ChatBatchIssue("missing_category", "The response omitted the requested category field.", key, "category"))
            elif entry.get("category") is None:
                issues.append(ChatBatchIssue("category_abstained", "The chatbot left category unknown.", key, "category"))
            elif not isinstance(entry.get("category"), str) or entry.get("category") not in category_to_macro:
                issues.append(ChatBatchIssue("invalid_category", "The category is not an exact canonical taxonomy value.", key, "category"))
            else:
                category_value = str(entry["category"])
                effective_category = category_value

        if category_value == "Non-Food":
            protected_existing_nutrition = any(
                field_name in protected_fields and product.get(field_name) is not None
                for field_name in NUTRIENT_FIELDS
            )
            if protected_existing_nutrition:
                issues.append(ChatBatchIssue("protected_non_food_conflict", "Non-Food would clear a manually protected nutrition value; category suggestion rejected.", key, "category"))
                category_value = None
                effective_category = None

        candidate_nutrition: dict[str, float] = {}
        nutrition_field_names = [field for field in NUTRIENT_FIELDS if field in requested]
        if nutrition_field_names:
            if effective_category is None:
                issues.append(ChatBatchIssue("nutrition_needs_category", "Nutrition cannot be previewed until a canonical Food category is known.", key))
            elif effective_category == "Non-Food":
                if any(entry.get(field_name) is not None for field_name in NUTRIENT_FIELDS):
                    issues.append(ChatBatchIssue("non_food_nutrition", "Nutrition suggestions are not accepted for Non-Food products.", key))
            else:
                for field_name in nutrition_field_names:
                    if field_name not in entry or entry[field_name] is None:
                        continue
                    value = entry[field_name]
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError, OverflowError):
                        numeric_value = math.nan
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(numeric_value)
                    ):
                        issues.append(ChatBatchIssue("nutrition_not_numeric", "The value must be a finite JSON number or null.", key, field_name))
                        continue
                    lower, upper = NUTRITION_RANGES[field_name]
                    if not lower <= numeric_value <= upper:
                        issues.append(ChatBatchIssue("nutrition_out_of_range", f"The value is outside the allowed range [{lower:g}, {upper:g}].", key, field_name))
                        continue
                    candidate_nutrition[field_name] = numeric_value

                if candidate_nutrition:
                    merged = dict(product)
                    merged["category"] = effective_category
                    merged["abstain_reason"] = None
                    for field_name in NUTRIENT_FIELDS:
                        if field_name in candidate_nutrition:
                            merged[field_name] = candidate_nutrition[field_name]
                    validation = validate_response(
                        {"products": [merged]}, [key], taxonomy, "estimate", allow_abstain=True
                    ).products[key]
                    consistency_issues = [issue for issue in validation.issues if issue.code in _NUTRITION_ISSUES]
                    if consistency_issues:
                        candidate_nutrition.clear()
                        issues.extend(
                            ChatBatchIssue(issue.code, issue.message, key)
                            for issue in consistency_issues
                        )

        package_weight: float | None = None
        if PACKAGE_FIELD in requested and PACKAGE_FIELD in entry and entry[PACKAGE_FIELD] is not None:
            value = entry[PACKAGE_FIELD]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                numeric_value = math.nan
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(numeric_value)
                or numeric_value < 0
            ):
                issues.append(ChatBatchIssue("invalid_package_weight", "Package weight must be a finite, non-negative number in grams or null.", key, PACKAGE_FIELD))
            else:
                package_weight = numeric_value

        if category_value is not None or candidate_nutrition or package_weight is not None:
            suggestions.append(ChatBatchSuggestion(
                product_key=key,
                category=category_value,
                nutrients=dict(candidate_nutrition) or None,
                package_weight_g=package_weight,
            ))

    return ChatBatchPreview(tuple(suggestions), tuple(issues), tuple(skipped))


def save_chat_batch_import(
    con: duckdb.DuckDBPyConnection,
    response_text: str,
    taxonomy: Taxonomy,
    category_to_macro: Mapping[str, str],
    blocked_fields: Mapping[str, Iterable[str]] | None = None,
) -> ChatBatchSaveReport:
    """Revalidate against current DB state, then atomically save valid products."""
    preview = preview_chat_batch_import(
        con, response_text, taxonomy, category_to_macro, blocked_fields
    )
    changed_products = 0
    saved_fields = 0
    for suggestion in preview.suggestions:
        category_result = (
            {"status": "succeeded", "value": suggestion.category}
            if suggestion.category is not None
            else None
        )
        nutrition_result = (
            {"status": "succeeded", "values": suggestion.nutrients}
            if suggestion.nutrients
            else None
        )
        changed = save_ai_enrichment_result(
            con,
            suggestion.product_key,
            category_result=category_result,
            nutrition_result=nutrition_result,
            package_weight_g=suggestion.package_weight_g,
            category_to_macro=category_to_macro,
            provider="chat_import",
            model=None,
            attempt_count=0,
            success_requires_saved_fields=True,
        )
        if changed:
            changed_products += 1
            saved_fields += len(changed)
    return ChatBatchSaveReport(changed_products, saved_fields, preview)
