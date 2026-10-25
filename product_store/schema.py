"""Storage schema for the product store (provenance metadata only).

DuckDB is the source of truth. This module defines the allow-listed editable
fields, the provenance source vocabulary, small validation helpers and the
``CREATE TABLE IF NOT EXISTS`` statements for the two bookkeeping tables.

Deliberately UI-independent: nothing here imports a presentation layer or the
legacy sync script, and importing this module has no side effects.
"""

from __future__ import annotations

import duckdb

# Editable/enrichable fields of ``nutrition_cache`` that can carry provenance.
# ``sample_name`` is intentionally excluded: it is a read-only receipt label,
# not a value that an enrichment pipeline is allowed to take ownership of.
FIELD_NAMES: tuple[str, ...] = (
    "macro_category",
    "category",
    "package_weight_g",
    "kcal_100g",
    "proteins_100g",
    "carbs_100g",
    "sugars_100g",
    "fat_100g",
    "saturated_fat_100g",
    "fiber_100g",
    "salt_100g",
)

CATEGORY_FIELDS: tuple[str, ...] = ("macro_category", "category")

NUTRIENT_FIELDS: tuple[str, ...] = (
    "kcal_100g",
    "proteins_100g",
    "carbs_100g",
    "sugars_100g",
    "fat_100g",
    "saturated_fat_100g",
    "fiber_100g",
    "salt_100g",
)

PACKAGE_FIELD = "package_weight_g"

# Provenance vocabulary. Exactly one of these is stored per metadata row.
SOURCE_MANUAL = "MANUAL"
SOURCE_SEED = "SEED"
SOURCE_OFF = "OFF"
SOURCE_AI = "AI"
SOURCE_LEGACY_UNKNOWN = "LEGACY_UNKNOWN"

SOURCE_VALUES: tuple[str, ...] = (
    SOURCE_MANUAL,
    SOURCE_SEED,
    SOURCE_OFF,
    SOURCE_AI,
    SOURCE_LEGACY_UNKNOWN,
)

MIGRATIONS_TABLE = "schema_migrations"
METADATA_TABLE = "product_field_metadata"
AI_STATE_TABLE = "ai_enrichment_state"

_SOURCE_SQL_LIST = ", ".join(f"'{value}'" for value in SOURCE_VALUES)

# Trade-off: ``field_name`` has NO CHECK constraint, on purpose. Field names are
# validated in Python (``validate_field_name``) so that adding a new editable
# field later is a code change only, never a CHECK-constraint table rebuild.
# ``source`` DOES have a CHECK constraint because its vocabulary is closed and
# enforced on every write path.
_CREATE_METADATA_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
    product_key VARCHAR NOT NULL,
    field_name  VARCHAR NOT NULL,
    source      VARCHAR NOT NULL,
    is_protected BOOLEAN NOT NULL DEFAULT FALSE,
    provider    VARCHAR,
    model       VARCHAR,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_key, field_name),
    CHECK (source IN ({_SOURCE_SQL_LIST}))
);
"""

_CREATE_MIGRATIONS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
    migration_id VARCHAR PRIMARY KEY,
    applied_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details      VARCHAR
);
"""

_CREATE_AI_STATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {AI_STATE_TABLE} (
    product_key VARCHAR NOT NULL,
    task VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    provider VARCHAR,
    model VARCHAR,
    error_code VARCHAR,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_key, task),
    CHECK (task IN ('category', 'nutrition')),
    CHECK (status IN ('succeeded', 'abstained', 'invalid', 'retryable_error')),
    CHECK (attempt_count >= 0)
);
"""


def validate_field_name(name: str) -> str:
    """Return ``name`` if it is an allow-listed editable field, else raise.

    Raises:
        ValueError: when ``name`` is not part of :data:`FIELD_NAMES`.
    """
    if name not in FIELD_NAMES:
        raise ValueError(f"Unknown field name: {name!r}. Allowed: {FIELD_NAMES}")
    return name


def validate_source(source: str) -> str:
    """Return ``source`` if it is a known provenance source, else raise.

    Raises:
        ValueError: when ``source`` is not part of :data:`SOURCE_VALUES`.
    """
    if source not in SOURCE_VALUES:
        raise ValueError(f"Unknown provenance source: {source!r}. Allowed: {SOURCE_VALUES}")
    return source


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the provenance metadata tables if they do not exist yet.

    Idempotent: safe to call on every run. Nothing is dropped or altered.
    """
    con.execute(_CREATE_METADATA_TABLE_SQL)
    con.execute(_CREATE_MIGRATIONS_TABLE_SQL)
    con.execute(_CREATE_AI_STATE_TABLE_SQL)
