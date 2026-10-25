"""Product store: provenance metadata on top of DuckDB.

DuckDB is the source of truth for product data; this package only records
where each editable value came from (manual, AI, seed catalog, OFF, unknown)
and whether it is protected from automatic overwrites.

The package is UI-independent and has no import side effects: importing it
creates no files and opens no database connection.
"""

from .migration import (
    DEFAULT_OVERRIDES_PATH,
    DEFAULT_SEED_PATH,
    MIGRATION_ID,
    MigrationReport,
    OverrideMismatch,
    apply_migration,
)
from .schema import (
    AI_STATE_TABLE,
    CATEGORY_FIELDS,
    FIELD_NAMES,
    NUTRIENT_FIELDS,
    PACKAGE_FIELD,
    SOURCE_AI,
    SOURCE_LEGACY_UNKNOWN,
    SOURCE_MANUAL,
    SOURCE_OFF,
    SOURCE_SEED,
    SOURCE_VALUES,
    create_schema,
    validate_field_name,
    validate_source,
)
from .services import (
    REQUIRED_FOOD_NUTRIENTS,
    list_incomplete_products,
    list_ai_enrichment_candidates,
    missing_required_fields,
    regenerate_review_csv,
    save_manual_fields,
    save_ai_enrichment_result,
    synchronize_field_metadata,
)

__all__ = [
    "AI_STATE_TABLE",
    "DEFAULT_OVERRIDES_PATH",
    "DEFAULT_SEED_PATH",
    "MIGRATION_ID",
    "MigrationReport",
    "OverrideMismatch",
    "apply_migration",
    "create_schema",
    "FIELD_NAMES",
    "CATEGORY_FIELDS",
    "NUTRIENT_FIELDS",
    "PACKAGE_FIELD",
    "SOURCE_MANUAL",
    "SOURCE_SEED",
    "SOURCE_OFF",
    "SOURCE_AI",
    "SOURCE_LEGACY_UNKNOWN",
    "SOURCE_VALUES",
    "validate_field_name",
    "validate_source",
    "REQUIRED_FOOD_NUTRIENTS",
    "list_ai_enrichment_candidates",
    "save_ai_enrichment_result",
    "missing_required_fields",
    "list_incomplete_products",
    "regenerate_review_csv",
    "save_manual_fields",
    "synchronize_field_metadata",
]
