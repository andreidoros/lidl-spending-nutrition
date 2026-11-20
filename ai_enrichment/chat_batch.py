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
    "fibre_energy_exceeds_kcal",
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
        # A complete product can still lack its package weight (see the
        # weight-only request); that weight is the only field it accepts.
        if product.get(PACKAGE_FIELD) is None and PACKAGE_FIELD not in protected:
            return (PACKAGE_FIELD,)
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


def build_weight_batch_text(products: Iterable[Mapping[str, object]]) -> str | None:
    """A chatbot request for package weights only.

    ``products`` carry ``product_key``, ``sample_name`` and ``sold_as``
    ("per piece" or "package"). The answer is imported like any chat batch.
    """
    items = [
        {
            "product_key": str(product["product_key"]),
            "sample_name": str(product.get("sample_name") or ""),
            "sold_as": str(product.get("sold_as") or "package"),
        }
        for product in products
    ]
    if not items:
        return None
    lines = [
        "Estimate the weight of these grocery products bought at Lidl Romania. Use only the product names; do not invent sources.",
        "Return only one JSON object with a 'products' array. Keep product_key exactly as the input text.",
        "For each product return product_key and package_weight_g: a number in grams, or null if you cannot tell.",
        "'per piece' means one item as sold at the counter or bakery (for example one croissant or one avocado);",
        "'package' means the whole package as sold (for example a tray of minced meat or a pack of 10 eggs).",
        "Null means unknown and never clears a database value.",
        "",
        "Products:",
        json.dumps(items, ensure_ascii=False, indent=2, separators=(",", ": ")),
        "",
        'Response shape: {"products": [{"product_key": "...", "package_weight_g": 120}]}',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact file format: one "product_key;value;value" line per product. Chatbots
# on free accounts cut long answers; these lines cost a fraction of JSON.
# ---------------------------------------------------------------------------

COMPACT_COLUMNS = {
    "nutrition": _PROMPT_FIELD_ORDER,
    "weights": (PACKAGE_FIELD,),
}
_COMPACT_LABELS = {
    "category": "category",
    "kcal_100g": "kcal",
    "proteins_100g": "protein",
    "carbs_100g": "carbs",
    "sugars_100g": "sugar",
    "fat_100g": "fat",
    "saturated_fat_100g": "sat_fat",
    "fiber_100g": "fibre",
    "salt_100g": "salt",
    PACKAGE_FIELD: "weight_g",
}


def _compact_header(kind: str) -> str:
    return ";".join(("product_key", *(_COMPACT_LABELS[field] for field in COMPACT_COLUMNS[kind])))


def build_compact_request(
    kind: str,
    products: Iterable[Mapping[str, object]],
    taxonomy: Taxonomy | None = None,
) -> str | None:
    """A file-sized chatbot request: instructions plus one line per product.

    ``kind`` is "nutrition" (products carry ``fields_to_fill``) or "weights".
    The answer is parsed by ``parse_compact_answer``.
    """
    lines_in = []
    for product in products:
        name = str(product.get("sample_name") or "").replace(";", ",")
        line = f"{product['product_key']};{name}"
        if kind == "nutrition":
            fields = product.get("fields_to_fill") or ()
            line += ";" + " ".join(_COMPACT_LABELS[field] for field in fields)
        lines_in.append(line)
    if not lines_in:
        return None
    header = _compact_header(kind)
    if kind == "weights":
        task = [
            "Estimate the weight of each grocery product below, bought at Lidl Romania, from its name only.",
            "weight_g is the weight in grams of one item as it appears on the receipt: one croissant, one pack, one bottle.",
        ]
        product_header = "product_key;product name"
    else:
        task = [
            "Complete the missing data of each grocery product below, bought at Lidl Romania, from its name only.",
            "The last column of each product lists the fields to fill; leave every other field empty.",
            "Nutrition is per 100 g: kcal, then grams of protein, carbs, sugar, fat, sat_fat, fibre and salt.",
            "weight_g is the weight in grams of one item as it appears on the receipt.",
            "category must be one exact value from the list below; use Non-Food for anything that is not food and give it no nutrition.",
        ]
        product_header = "product_key;product name;fields to fill"
    lines = [
        *task,
        "",
        "Answer with one line per product, in a code block, in exactly this format:",
        header,
        "Use ; between values, keep product_key exactly as given, leave a value empty when you are not sure,",
        "and write numbers without units. No other text is needed.",
        "If the answer gets too long, stop after a complete line; you will be asked to continue.",
    ]
    if kind == "nutrition" and taxonomy is not None:
        lines += ["", "Categories:", taxonomy.prompt_lines()]
    lines += ["", "Products:", product_header, *lines_in]
    return "\n".join(lines)


def build_compact_chat_batch(
    con: duckdb.DuckDBPyConnection,
    taxonomy: Taxonomy,
    category_to_macro: Mapping[str, str],
) -> str | None:
    """The review batch (incomplete products) in the compact file format."""
    return build_compact_request("nutrition", _batch_products(con, category_to_macro), taxonomy)


def _compact_number(text: str) -> float | None:
    """A number from a compact answer; "0,75 kg" is 750 (grams)."""
    cleaned = text.strip().replace(" ", "").replace(",", ".")
    factor = 1.0
    for suffix, suffix_factor in (("kcal", 1.0), ("kg", 1000.0), ("g", 1.0)):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            factor = suffix_factor
            break
    try:
        value = float(cleaned) * factor
    except ValueError:
        return None
    return value if math.isfinite(value) else None


# Words a chatbot writes for "I don't know"; they count as an empty value.
_EMPTY_WORDS = frozenset({"n/a", "na", "null", "none", "-", "--", "?", "unknown", "nan"})


def _default_categories() -> frozenset[str]:
    try:
        from ai_eval.taxonomy import load_taxonomy

        return frozenset(load_taxonomy().categories)
    except Exception:  # noqa: BLE001 - parsing still works without the list
        return frozenset()


def _is_text(value: str) -> bool:
    return bool(value) and _compact_number(value) is None


def _copied_name(values: list[str], kind: str, expected: int, categories: frozenset[str]) -> bool:
    """Whether the first value is the product name the chatbot copied back.

    The answer has ``expected`` values after the key; one more means a copied
    name. A name is text that is not a category; the category column is the
    only other text column, which tells the two apart when lines are short.
    """
    if not values or not _is_text(values[0]) or values[0] in categories:
        return False
    if kind == "weights":
        return len(values) >= 2
    if len(values) == expected + 1:
        return True
    if len(values) >= 2 and categories:
        return values[1] in categories
    return len(values) >= 2 and _is_text(values[1])


def parse_compact_answer(
    text: str,
    kind: str,
    categories: Iterable[str] | None = None,
) -> tuple[dict[str, list[dict[str, object]]], int]:
    """Read "product_key;value;..." lines into the JSON shape of a chat answer.

    Explanations, code fences and header lines are ignored, decimal commas
    are accepted and empty values (or N/A, null, -, ?) mean unknown. Values
    are matched to columns by position; a copied product name is recognised
    by the number of values and by the category list, never by guessing from
    one cell. A line with more values than the format allows is not read.
    Returns the payload and the number of lines that looked like data but
    could not be read.
    """
    columns = COMPACT_COLUMNS[kind]
    known = frozenset(categories) if categories is not None else _default_categories()
    products: list[dict[str, object]] = []
    unreadable = 0
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        if ";" not in line and "\t" not in line:
            continue
        parts = [part.strip().strip('"').strip() for part in line.replace("\t", ";").split(";")]
        key = parts[0]
        if not key.replace("-", "").replace("_", "").isalnum() or key.lower() == "product_key":
            continue  # prose, headers and code fences
        values = ["" if part.lower() in _EMPTY_WORDS else part for part in parts[1:]]
        if _copied_name(values, kind, len(columns), known):
            values = values[1:]
        # A trailing ";" adds empty values; anything else beyond the format
        # means the columns cannot be matched safely.
        while len(values) > len(columns) and values[-1] == "":
            values.pop()
        if len(values) > len(columns):
            unreadable += 1
            continue
        entry: dict[str, object] = {"product_key": key}
        readable = False
        for field, value in zip(columns, values):
            if not value:
                continue
            if field == "category":
                entry[field] = value
                readable = True
                continue
            number = _compact_number(value)
            if number is None:
                continue
            entry[field] = number
            readable = True
        if readable:
            products.append(entry)
        else:
            unreadable += 1
    return {"products": products}, unreadable


def normalize_chat_answer(
    text: str,
    kind: str,
    categories: Iterable[str] | None = None,
) -> tuple[str, int]:
    """Return a JSON chat answer for either format, plus unreadable compact lines.

    A JSON answer (the copy-and-paste route) is passed through unchanged.
    """
    stripped = text.strip()
    if "{" in stripped and '"products"' in stripped:
        return text, 0
    payload, unreadable = parse_compact_answer(text, kind, categories)
    return json.dumps(payload, ensure_ascii=False), unreadable


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
