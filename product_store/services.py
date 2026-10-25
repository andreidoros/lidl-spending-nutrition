"""UI-independent product completeness, manual-edit and review services."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import duckdb

from .migration import (
    DEFAULT_OVERRIDES_PATH,
    DEFAULT_SEED_PATH,
    _load_overrides,
    _load_seed,
    _resolve_source,
)
from .schema import (
    AI_STATE_TABLE,
    CATEGORY_FIELDS,
    FIELD_NAMES,
    METADATA_TABLE,
    NUTRIENT_FIELDS,
    SOURCE_AI,
    SOURCE_MANUAL,
    create_schema,
    validate_field_name,
)

REQUIRED_FOOD_NUTRIENTS: tuple[str, ...] = (
    "kcal_100g",
    "proteins_100g",
    "carbs_100g",
    "sugars_100g",
    "fat_100g",
    "saturated_fat_100g",
    "salt_100g",
)

_NUMERIC_FIELDS = frozenset(FIELD_NAMES) - frozenset(CATEGORY_FIELDS)


def missing_required_fields(
    product: Mapping[str, object], category_to_macro: Mapping[str, str]
) -> tuple[str, ...]:
    """Return required fields still missing for a Food product.

    A canonical Non-Food category needs no nutrition. Fiber and package weight
    are optional. Blank strings are unknown; numeric zero is present.
    """
    missing: list[str] = []
    category = _text(product.get("category"))
    if category == "Non-Food":
        canonical_macro = category_to_macro.get(category)
        if canonical_macro is not None and _text(product.get("macro_category")) != canonical_macro:
            missing.append("macro_category")
        return tuple(missing)

    canonical_macro = category_to_macro.get(category)
    if canonical_macro is None:
        missing.append("category")
    elif _text(product.get("macro_category")) != canonical_macro:
        # The macro is derived from the canonical category. A valid category
        # with a stale macro is still inconsistent and needs review/repair.
        missing.append("macro_category")

    for field_name in REQUIRED_FOOD_NUTRIENTS:
        if product.get(field_name) is None or _text(product.get(field_name)) == "":
            missing.append(field_name)
    return tuple(missing)


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def save_manual_fields(
    con: duckdb.DuckDBPyConnection,
    product_key: str,
    edits: Mapping[str, object],
    category_to_macro: Mapping[str, str],
    *,
    clear_legacy_override_fields: Iterable[str] = (),
) -> tuple[str, ...]:
    """Persist explicit human edits and protect only the affected fields.

    Category selection derives its macro category from the supplied canonical
    taxonomy. Explicit ``None``/blank values clear that field and its metadata;
    omitted fields stay unchanged. Selecting Non-Food also clears all nutrition
    values and their metadata, while preserving package weight.
    """
    key = _text(product_key)
    if not key:
        raise ValueError("product_key must not be empty")
    if not edits:
        return ()

    legacy_fields_to_clear = set(clear_legacy_override_fields)
    invalid_legacy_fields = legacy_fields_to_clear - (set(FIELD_NAMES) | {"non_food"})
    if invalid_legacy_fields:
        raise ValueError(f"Unsupported legacy override fields: {sorted(invalid_legacy_fields)}")

    normalised: dict[str, object] = {}
    for field_name, value in edits.items():
        validate_field_name(field_name)
        if field_name == "macro_category":
            raise ValueError("macro_category is derived from category and cannot be edited directly")
        normalised_value = _normalise_field(field_name, value)
        normalised[field_name] = normalised_value

    current_category_row = con.execute(
        "SELECT category FROM nutrition_cache WHERE product_key = ?", [key]
    ).fetchone()
    if current_category_row is None:
        raise KeyError(f"Unknown product_key: {key}")
    current_category = current_category_row[0]
    effective_category = normalised.get("category", current_category)
    nutrient_fields = REQUIRED_FOOD_NUTRIENTS + ("fiber_100g",)
    if effective_category == "Non-Food" and any(
        normalised.get(field_name) is not None for field_name in nutrient_fields
    ):
        raise ValueError("Non-Food products cannot have nutrient edits")

    if "category" in normalised:
        category = normalised["category"]
        if category == "Non-Food":
            normalised["macro_category"] = "Non-Food"
        elif category is None:
            normalised["macro_category"] = None
        elif category not in category_to_macro:
            raise ValueError(f"Unknown taxonomy category: {category!r}")
        else:
            normalised["macro_category"] = category_to_macro[category]

    create_schema(con)
    con.execute("BEGIN TRANSACTION")
    try:
        exists = con.execute(
            "SELECT 1 FROM nutrition_cache WHERE product_key = ?", [key]
        ).fetchone()
        if exists is None:
            raise KeyError(f"Unknown product_key: {key}")

        if legacy_fields_to_clear:
            legacy_columns = {
                row[0]
                for row in con.execute(
                    """SELECT column_name FROM information_schema.columns
                       WHERE table_name = 'catalog_overrides'"""
                ).fetchall()
            }
            for field_name in sorted(legacy_fields_to_clear & legacy_columns):
                con.execute(
                    f"UPDATE catalog_overrides SET {field_name} = NULL WHERE product_key = ?",
                    [key],
                )

        new_category = normalised.get("category", current_category)
        if new_category == "Non-Food":
            # Explicitly changing category to Non-Food follows the approved
            # domain rule: nutrition values and their metadata cease to apply;
            # package weight intentionally remains untouched.
            for field_name in nutrient_fields:
                con.execute(
                    f"UPDATE nutrition_cache SET {field_name} = NULL WHERE product_key = ?",
                    [key],
                )
                con.execute(
                    f"DELETE FROM {METADATA_TABLE} WHERE product_key = ? AND field_name = ?",
                    [key, field_name],
                )
            con.execute(
                "UPDATE nutrition_cache SET is_verified = TRUE, match_status = 'NON_FOOD', "
                "data_source = 'RULE_NON_FOOD' WHERE product_key = ?",
                [key],
            )
        elif current_category == "Non-Food" and "category" in normalised:
            # A food reclassification starts a fresh incomplete-food state.
            # Nutrients stay NULL; the old Non-Food status is not evidence of
            # successful food enrichment or validation.
            con.execute(
                """UPDATE nutrition_cache
                   SET is_verified = FALSE,
                       match_status = CASE WHEN match_status = 'NON_FOOD' THEN NULL ELSE match_status END,
                       data_source = CASE WHEN data_source = 'RULE_NON_FOOD' THEN NULL ELSE data_source END
                   WHERE product_key = ?""",
                [key],
            )
            if new_category != "Non-Food":
                # Nutrition was cleared during the Non-Food transition. A past
                # AI result must not prevent a fresh estimate after reclassifying
                # the product back to Food.
                con.execute(
                    f"DELETE FROM {AI_STATE_TABLE} WHERE product_key = ? AND task = 'nutrition'",
                    [key],
                )
        elif "category" in normalised and new_category != current_category:
            # Category edits can change validation rules (e.g. the alcohol
            # exemption), so a previous product-level verification is stale.
            con.execute(
                "UPDATE nutrition_cache SET is_verified = FALSE WHERE product_key = ?",
                [key],
            )

        if new_category != "Non-Food" and any(
            field_name in normalised for field_name in nutrient_fields
        ):
            # Changing or clearing any nutrient invalidates a prior product-wide
            # validation result. Protection remains independent per field.
            con.execute(
                "UPDATE nutrition_cache SET is_verified = FALSE WHERE product_key = ?",
                [key],
            )

        edited_fields = tuple(normalised)
        for field_name, value in normalised.items():
            con.execute(
                f"UPDATE nutrition_cache SET {field_name} = ? WHERE product_key = ?",
                [value, key],
            )
            if value is None:
                # An unknown value has no active source/protection. Other
                # fields' metadata remains untouched.
                con.execute(
                    f"DELETE FROM {METADATA_TABLE} WHERE product_key = ? AND field_name = ?",
                    [key, field_name],
                )
            else:
                con.execute(
                    f"""INSERT INTO {METADATA_TABLE}
                        (product_key, field_name, source, is_protected, provider, model, updated_at)
                        VALUES (?, ?, ?, TRUE, NULL, NULL, CURRENT_TIMESTAMP)
                        ON CONFLICT (product_key, field_name) DO UPDATE SET
                            source = excluded.source,
                            is_protected = TRUE,
                            provider = NULL,
                            model = NULL,
                            updated_at = now()""",
                    [key, field_name, SOURCE_MANUAL],
                )
        con.execute("COMMIT")
        return edited_fields
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise


_AI_TERMINAL_STATUSES = frozenset({"succeeded", "abstained", "invalid"})


def list_ai_enrichment_candidates(
    con: duckdb.DuckDBPyConnection,
    category_to_macro: Mapping[str, str],
    *,
    nutrition_enabled: bool,
    selected_product_keys: Iterable[str] | None = None,
    retry_terminal: bool = False,
) -> list[dict[str, object]]:
    """Return products with AI tasks not completed or terminally rejected.

    Category and nutrition have independent state rows. Nutrition is eligible
    only for a canonical Food category, except that a product with no category
    may receive both suggestions in one response; the nutrition suggestion is
    accepted only if its category suggestion is valid Food.
    """
    selected_keys = None
    if selected_product_keys is not None:
        selected_keys = {_text(key) for key in selected_product_keys if _text(key)}
        if not selected_keys:
            return []
    if retry_terminal and selected_keys is None:
        raise ValueError("Terminal AI tasks may only be retried for an explicit product selection")

    create_schema(con)
    selected_fields = ("product_key", "sample_name", *FIELD_NAMES, "match_status")
    rows = con.execute(
        f"""SELECT {', '.join(f'nc.{field}' for field in selected_fields)},
                   category_state.status, nutrition_state.status,
                   COALESCE(protection.category_protected, FALSE),
                   {', '.join(f'COALESCE(protection.{field}_protected, FALSE)' for field in REQUIRED_FOOD_NUTRIENTS)}
            FROM nutrition_cache nc
            LEFT JOIN {AI_STATE_TABLE} category_state
              ON category_state.product_key = nc.product_key
             AND category_state.task = 'category'
            LEFT JOIN {AI_STATE_TABLE} nutrition_state
              ON nutrition_state.product_key = nc.product_key
             AND nutrition_state.task = 'nutrition'
            LEFT JOIN (
                SELECT product_key,
                       BOOL_OR(field_name = 'category' AND is_protected) AS category_protected,
                       {', '.join(f"BOOL_OR(field_name = '{field}' AND is_protected) AS {field}_protected" for field in REQUIRED_FOOD_NUTRIENTS)}
                FROM {METADATA_TABLE}
                GROUP BY product_key
            ) protection ON protection.product_key = nc.product_key
            ORDER BY nc.product_key"""
    ).fetchall()

    candidates: list[dict[str, object]] = []
    for raw in rows:
        protected_fields = ("_category_protected", *(
            f"_{field}_protected" for field in REQUIRED_FOOD_NUTRIENTS
        ))
        product = dict(zip(
            (*selected_fields, "_category_state", "_nutrition_state", *protected_fields),
            raw,
        ))
        key = _text(product["product_key"])
        if not key:
            continue
        if selected_keys is not None and key not in selected_keys:
            continue
        category = _text(product.get("category"))
        category_is_blank = not category
        category_due = (
            category_is_blank
            and not product["_category_protected"]
            and (
                retry_terminal
                or product["_category_state"] not in _AI_TERMINAL_STATUSES
            )
        )
        nutrition_due = False
        if nutrition_enabled:
            category_known = category in category_to_macro
            nutrition_missing = any(
                product.get(field_name) is None and not product[f"_{field_name}_protected"]
                for field_name in REQUIRED_FOOD_NUTRIENTS
            )
            status_allows_retry = (
                retry_terminal
                or product["_nutrition_state"] not in _AI_TERMINAL_STATUSES
            )
            nutrition_due = status_allows_retry and nutrition_missing and (
                (category_known and category != "Non-Food")
                or (category_is_blank and category_due)
            )

        if category_due or nutrition_due:
            product["category_task"] = category_due
            product["nutrition_task"] = nutrition_due
            candidates.append(product)
    return candidates


def save_ai_enrichment_result(
    con: duckdb.DuckDBPyConnection,
    product_key: str,
    *,
    category_result: Mapping[str, object] | None = None,
    nutrition_result: Mapping[str, object] | None = None,
    category_to_macro: Mapping[str, str],
    provider: str,
    model: str | None,
    package_weight_g: float | None = None,
    attempt_count: int = 1,
    success_requires_saved_fields: bool = False,
) -> tuple[str, ...]:
    """Atomically save accepted missing fields and independent task outcomes.

    ``category_result`` and ``nutrition_result`` contain ``status`` and may
    contain an accepted ``value``/``values`` or a stable ``error_code``. Only
    allow-listed fields are written; raw model replies and error messages are
    deliberately not accepted by this API.
    """
    key = _text(product_key)
    if not key:
        raise ValueError("product_key must not be empty")
    if attempt_count < 0:
        raise ValueError("attempt_count must not be negative")
    create_schema(con)
    changed_fields: list[str] = []
    con.execute("BEGIN TRANSACTION")
    try:
        row = con.execute(
            f"SELECT category, macro_category, package_weight_g, {', '.join(NUTRIENT_FIELDS)} "
            "FROM nutrition_cache WHERE product_key = ?",
            [key],
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown product_key: {key}")
        current_category = row[0]
        current_macro = row[1]
        current_package_weight = row[2]
        current_nutrition = dict(zip(NUTRIENT_FIELDS, row[3:]))

        accepted_category = None
        if category_result is not None and category_result.get("status") == "succeeded":
            proposed = category_result.get("value")
            if not isinstance(proposed, str) or proposed not in category_to_macro:
                raise ValueError("Accepted AI category must be canonical taxonomy text")
            accepted_category = proposed

        effective_category = current_category
        if accepted_category is not None and not _text(current_category):
            protected_fields = {
                field_name
                for (field_name,) in con.execute(
                    f"""SELECT field_name FROM {METADATA_TABLE}
                        WHERE product_key = ?
                          AND is_protected""",
                    [key],
                ).fetchall()
            }
            protected_nutrition_present = any(
                field_name in protected_fields and current_nutrition[field_name] is not None
                for field_name in NUTRIENT_FIELDS
            )
            if "category" in protected_fields:
                category_result = {
                    **category_result,
                    "status": "invalid",
                    "error_code": "category_protected",
                }
                accepted_category = None
            elif accepted_category == "Non-Food" and protected_nutrition_present:
                # Non-Food classification clears nutrition. Reject it rather
                # than erase a human-protected value or its provenance.
                category_result = {
                    **category_result,
                    "status": "invalid",
                    "error_code": "protected_nutrition_conflict",
                }
                accepted_category = None
            else:
                effective_category = accepted_category
                derived_macro = category_to_macro[accepted_category]
                con.execute(
                    "UPDATE nutrition_cache SET category = ? WHERE product_key = ? "
                    "AND (category IS NULL OR TRIM(category) = '')",
                    [accepted_category, key],
                )
                changed_fields.append("category")
                category_fields_written = ["category"]
                if "macro_category" not in protected_fields and not _text(current_macro):
                    con.execute(
                        "UPDATE nutrition_cache SET macro_category = ? WHERE product_key = ? "
                        "AND (macro_category IS NULL OR TRIM(macro_category) = '')",
                        [derived_macro, key],
                    )
                    changed_fields.append("macro_category")
                    category_fields_written.append("macro_category")
                for field_name in category_fields_written:
                    con.execute(
                        f"""INSERT INTO {METADATA_TABLE}
                            (product_key, field_name, source, is_protected, provider, model, updated_at)
                            VALUES (?, ?, ?, FALSE, ?, ?, CURRENT_TIMESTAMP)
                            ON CONFLICT (product_key, field_name) DO UPDATE SET
                                source = excluded.source, is_protected = FALSE,
                                provider = excluded.provider, model = excluded.model,
                                updated_at = now()""",
                        [key, field_name, SOURCE_AI, provider, model],
                    )

        if effective_category == "Non-Food" and accepted_category == "Non-Food":
            for field_name in NUTRIENT_FIELDS:
                con.execute(f"UPDATE nutrition_cache SET {field_name} = NULL WHERE product_key = ?", [key])
                con.execute(
                    f"DELETE FROM {METADATA_TABLE} "
                    "WHERE product_key = ? AND field_name = ? AND NOT is_protected",
                    [key, field_name],
                )
                current_nutrition[field_name] = None
            con.execute(
                "UPDATE nutrition_cache SET is_verified = TRUE, match_status = 'NON_FOOD', "
                "data_source = 'RULE_NON_FOOD' WHERE product_key = ?",
                [key],
            )
            nutrition_result = None

        if nutrition_result is not None and nutrition_result.get("status") == "succeeded":
            proposed_values = nutrition_result.get("values")
            if not isinstance(proposed_values, Mapping):
                raise ValueError("Accepted AI nutrition result must contain a values mapping")
            if effective_category in category_to_macro and effective_category != "Non-Food":
                for field_name in NUTRIENT_FIELDS:
                    value = proposed_values.get(field_name)
                    if value is None or current_nutrition[field_name] is not None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"AI value for {field_name} must be numeric or NULL")
                    if not math.isfinite(float(value)):
                        raise ValueError(f"AI value for {field_name} must be finite")
                    protected = con.execute(
                        f"""SELECT 1 FROM {METADATA_TABLE}
                            WHERE product_key = ? AND field_name = ? AND is_protected""",
                        [key, field_name],
                    ).fetchone()
                    if protected is not None:
                        continue
                    con.execute(
                        f"UPDATE nutrition_cache SET {field_name} = ? WHERE product_key = ? AND {field_name} IS NULL",
                        [float(value), key],
                    )
                    con.execute(
                        f"""INSERT INTO {METADATA_TABLE}
                            (product_key, field_name, source, is_protected, provider, model, updated_at)
                            VALUES (?, ?, ?, FALSE, ?, ?, CURRENT_TIMESTAMP)
                            ON CONFLICT (product_key, field_name) DO UPDATE SET
                                source = excluded.source,
                                is_protected = FALSE,
                                provider = excluded.provider,
                                model = excluded.model,
                                updated_at = now()
                            WHERE NOT {METADATA_TABLE}.is_protected""",
                        [key, field_name, SOURCE_AI, provider, model],
                    )
                    current_nutrition[field_name] = float(value)
                    changed_fields.append(field_name)

        if package_weight_g is not None:
            if (
                isinstance(package_weight_g, bool)
                or not isinstance(package_weight_g, (int, float))
                or not math.isfinite(float(package_weight_g))
                or float(package_weight_g) < 0
            ):
                raise ValueError("AI package_weight_g must be a finite, non-negative number")
            if current_package_weight is None:
                protected = con.execute(
                    f"""SELECT 1 FROM {METADATA_TABLE}
                        WHERE product_key = ? AND field_name = ? AND is_protected""",
                    [key, "package_weight_g"],
                ).fetchone()
                if protected is None:
                    con.execute(
                        "UPDATE nutrition_cache SET package_weight_g = ? "
                        "WHERE product_key = ? AND package_weight_g IS NULL",
                        [float(package_weight_g), key],
                    )
                    con.execute(
                        f"""INSERT INTO {METADATA_TABLE}
                            (product_key, field_name, source, is_protected, provider, model, updated_at)
                            VALUES (?, ?, ?, FALSE, ?, ?, CURRENT_TIMESTAMP)
                            ON CONFLICT (product_key, field_name) DO UPDATE SET
                                source = excluded.source,
                                is_protected = FALSE,
                                provider = excluded.provider,
                                model = excluded.model,
                                updated_at = now()
                            WHERE NOT {METADATA_TABLE}.is_protected""",
                        [key, "package_weight_g", SOURCE_AI, provider, model],
                    )
                    changed_fields.append("package_weight_g")

        for task, result in (("category", category_result), ("nutrition", nutrition_result)):
            if result is None:
                continue
            status = result.get("status")
            if status not in {"succeeded", "abstained", "invalid", "retryable_error"}:
                raise ValueError(f"Invalid AI task status for {task}: {status!r}")
            if success_requires_saved_fields and status == "succeeded":
                task_fields = (
                    {"category"}
                    if task == "category"
                    else set(NUTRIENT_FIELDS)
                )
                if not task_fields.intersection(changed_fields):
                    continue
            error_code = result.get("error_code")
            if error_code is not None and not isinstance(error_code, str):
                raise ValueError("AI error_code must be a string or NULL")
            con.execute(
                f"""INSERT INTO {AI_STATE_TABLE}
                    (product_key, task, status, attempt_count, provider, model, error_code, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (product_key, task) DO UPDATE SET
                        status = excluded.status,
                        attempt_count = {AI_STATE_TABLE}.attempt_count + excluded.attempt_count,
                        provider = excluded.provider,
                        model = excluded.model,
                        error_code = excluded.error_code,
                        updated_at = now()""",
                [key, task, status, attempt_count, provider, model, error_code],
            )

        con.execute("COMMIT")
        return tuple(changed_fields)
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise


def synchronize_field_metadata(
    con: duckdb.DuckDBPyConnection,
    overrides_path: Path | str | None = DEFAULT_OVERRIDES_PATH,
    seed_path: Path | str | None = DEFAULT_SEED_PATH,
) -> int:
    """Add provenance for current values not yet tracked by the one-off migration.

    Existing metadata is never downgraded or rewritten, except that a current
    value explicitly confirmed by the manual overrides file is promoted to
    protected MANUAL. This lets later syncs track newly discovered products
    while keeping the original migration's one-time semantics intact.
    """
    create_schema(con)
    overrides = _load_overrides(overrides_path)
    seed = _load_seed(seed_path)
    rows = con.execute(
        f"SELECT product_key, {', '.join(FIELD_NAMES)}, match_status, data_source "
        "FROM nutrition_cache ORDER BY product_key"
    ).fetchall()
    current = ["product_key", *FIELD_NAMES, "match_status", "data_source"]
    existing = {
        (key, field_name): (source, protected)
        for key, field_name, source, protected in con.execute(
            f"SELECT product_key, field_name, source, is_protected FROM {METADATA_TABLE}"
        ).fetchall()
    }

    inserts: list[tuple[str, str, str, bool]] = []
    promotes: list[tuple[str, str]] = []
    for raw in rows:
        row = dict(zip(current, raw))
        key = row["product_key"]
        override = overrides.get(key)
        for field_name in FIELD_NAMES:
            value = row[field_name]
            if value is None:
                continue
            existing_row = existing.get((key, field_name))
            source, protected = _resolve_source(
                field_name=field_name,
                current_value=value,
                override_row=override,
                seed_row=seed.get(key),
                match_status=row["match_status"],
                data_source=row["data_source"],
            )
            if source == SOURCE_MANUAL:
                if existing_row is None:
                    inserts.append((key, field_name, SOURCE_MANUAL, True))
                elif existing_row[0] != SOURCE_MANUAL or not existing_row[1]:
                    promotes.append((key, field_name))
                continue
            if existing_row is not None:
                continue
            inserts.append((key, field_name, source, protected))

    if not inserts and not promotes:
        return 0

    con.execute("BEGIN TRANSACTION")
    try:
        if inserts:
            con.execute("CREATE TEMP TABLE _pfm_sync_stage (product_key VARCHAR, field_name VARCHAR, source VARCHAR, is_protected BOOLEAN)")
            con.executemany("INSERT INTO _pfm_sync_stage VALUES (?, ?, ?, ?)", inserts)
            con.execute(
                f"""INSERT INTO {METADATA_TABLE} (product_key, field_name, source, is_protected)
                    SELECT product_key, field_name, source, is_protected FROM _pfm_sync_stage
                    ON CONFLICT (product_key, field_name) DO NOTHING"""
            )
            con.execute("DROP TABLE _pfm_sync_stage")
        if promotes:
            con.executemany(
                f"UPDATE {METADATA_TABLE} SET source = ?, is_protected = TRUE, provider = NULL, model = NULL, updated_at = now() WHERE product_key = ? AND field_name = ?",
                [(SOURCE_MANUAL, key, field_name) for key, field_name in promotes],
            )
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    return len(inserts) + len(promotes)


def _normalise_field(field_name: str, value: object) -> object:
    if value is None:
        return None
    if field_name in _NUMERIC_FIELDS:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be numeric, not boolean")
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {field_name}: {value!r}") from exc
    text = _text(value)
    return text or None


def list_incomplete_products(
    con: duckdb.DuckDBPyConnection,
    category_to_macro: Mapping[str, str],
) -> list[dict[str, object]]:
    """Build a deterministic full snapshot of products requiring review."""
    fields = ("product_key", "sample_name") + FIELD_NAMES + (
        "match_status", "off_product_name", "atwater_debug"
    )
    rows = con.execute(
        f"SELECT {', '.join(fields)} FROM nutrition_cache"
    ).fetchall()
    purchases = dict(con.execute(
        """SELECT TRIM(CAST(COALESCE(NULLIF(TRIM(ean), ''), raw_code) AS VARCHAR)) AS k,
                  COUNT(*) FROM purchases GROUP BY 1"""
    ).fetchall())

    products: list[dict[str, object]] = []
    for raw in rows:
        product = dict(zip(fields, raw))
        if not missing_required_fields(product, category_to_macro):
            continue
        product["times_bought"] = purchases.get(product["product_key"], 0)
        product["missing_fields"] = missing_required_fields(product, category_to_macro)
        products.append(product)

    products.sort(key=lambda row: (-int(row["times_bought"] or 0), _text(row.get("sample_name")).casefold(), _text(row.get("product_key"))))
    return products


def regenerate_review_csv(
    con: duckdb.DuckDBPyConnection,
    category_to_macro: Mapping[str, str],
    output_path: Path | str,
    columns: tuple[str, ...] | list[str],
    *,
    delimiter: str = ",",
) -> int:
    """Overwrite the review CSV from DuckDB's current incomplete-product set.

    The CSV is a derived worklist only. Existing queue contents are not used as
    input, so resolved rows disappear and stale rows cannot persist.
    """
    path = Path(output_path)
    products = list_incomplete_products(con, category_to_macro)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter=delimiter, extrasaction="ignore")
            writer.writeheader()
            for product in products:
                row = {
                "product_key": product["product_key"],
                "sample_name": product["sample_name"] or "",
                "macro_category": product["macro_category"] or "",
                "category": product["category"] or "",
                "package_weight_g": _csv_value(product["package_weight_g"]),
                **{name: _csv_value(product[name]) for name in REQUIRED_FOOD_NUTRIENTS + ("fiber_100g",)},
                "non_food": "TRUE" if product["category"] == "Non-Food" else "",
                "times_bought": product["times_bought"],
                "match_status": product["match_status"] or "",
                "off_product_name": product["off_product_name"] or "",
                "atwater_debug": product["atwater_debug"] or "",
                }
                writer.writerow({name: _sanitize_csv_cell(value) for name, value in row.items()})
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return len(products)


def _csv_value(value: object) -> str:
    return "" if value is None else str(value)


def _sanitize_csv_cell(value: object) -> object:
    """Neutralize spreadsheet formula injection in exported text cells."""
    if value is None:
        return ""
    text = str(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{text}"
    return text
