"""One-off provenance backfill for existing ``nutrition_cache`` values.

``apply_migration`` scans the current values of ``nutrition_cache`` and records,
for each non-NULL editable field, which source most plausibly produced it
(``MANUAL`` / ``AI`` / ``SEED`` / ``OFF`` / ``LEGACY_UNKNOWN``). ``SEED`` needs
per-field evidence: the same value must exist in the seed catalog, because a
row-level ``data_source`` flag mixes fields. It never modifies
``nutrition_cache`` and never corrects conflicting overrides; the legacy table
stays the source of truth for values.

Everything happens in a single transaction: schema, metadata rows and the
``schema_migrations`` evidence row are committed together or not at all.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .schema import (
    CATEGORY_FIELDS,
    FIELD_NAMES,
    METADATA_TABLE,
    MIGRATIONS_TABLE,
    NUTRIENT_FIELDS,
    PACKAGE_FIELD,
    SOURCE_AI,
    SOURCE_LEGACY_UNKNOWN,
    SOURCE_MANUAL,
    SOURCE_OFF,
    SOURCE_SEED,
    create_schema,
)

MIGRATION_ID = "0001_product_field_metadata"
DEFAULT_OVERRIDES_PATH = Path("catalog_overrides.csv")
DEFAULT_SEED_PATH = Path("data/catalogs/ro/catalog_seed.csv")

# Values considered "true" in the ``non_food`` column of the overrides CSV.
# Mirrors the legacy reader so both see the same file identically.
TRUTHY = ("1", "x", "yes", "y", "true", "da", "adevarat")

# Excel-corrupted product keys (e.g. ``4.05649E+12``) cannot match anything.
SCI_NOTATION_RE = re.compile(r"^\d+(\.\d+)?e[+-]?\d+$", re.IGNORECASE)

OFF_MATCH_STATUSES = ("MATCHED_OFF", "MATCHED_OFF_VERIFIED", "MATCHED_OFF_WATERFALL")
AI_MATCH_STATUS = "AI_ESTIMATED"

_NUMERIC_FIELDS = frozenset((PACKAGE_FIELD,) + NUTRIENT_FIELDS)

_SCAN_COLUMNS = ("product_key",) + FIELD_NAMES + ("match_status", "data_source")
_SCAN_SQL = f"SELECT {', '.join(_SCAN_COLUMNS)} FROM nutrition_cache ORDER BY product_key"

# Rows are staged in a temporary table and moved with one set-based INSERT.
# A row-by-row ``executemany`` with ``ON CONFLICT`` on a composite primary key
# is orders of magnitude slower in DuckDB (~40s for 20k rows vs <1s here).
_STAGE_TABLE = "_pfm_migration_stage"
_STAGE_SQL = (
    f"CREATE TEMP TABLE {_STAGE_TABLE} ("
    "product_key VARCHAR, field_name VARCHAR, source VARCHAR, is_protected BOOLEAN)"
)
_STAGE_INSERT_SQL = f"INSERT INTO {_STAGE_TABLE} VALUES (?, ?, ?, ?)"
_INSERT_METADATA_SQL = (
    f"INSERT INTO {METADATA_TABLE} (product_key, field_name, source, is_protected) "
    f"SELECT product_key, field_name, source, is_protected FROM {_STAGE_TABLE} "
    "ON CONFLICT (product_key, field_name) DO NOTHING"
)
_COUNT_METADATA_SQL = f"SELECT COUNT(*) FROM {METADATA_TABLE}"

_EXISTING_METADATA_SQL = f"SELECT product_key, field_name FROM {METADATA_TABLE}"

_MIGRATION_EXISTS_SQL = f"SELECT migration_id FROM {MIGRATIONS_TABLE} WHERE migration_id = ?"

_INSERT_EVIDENCE_SQL = (
    f"INSERT INTO {MIGRATIONS_TABLE} (migration_id, applied_at, details) "
    "VALUES (?, CURRENT_TIMESTAMP, ?)"
)


@dataclass(frozen=True)
class OverrideMismatch:
    """A non-empty override cell whose value differs from the stored value."""

    product_key: str
    field_name: str
    override_value: object
    current_value: object


@dataclass
class MigrationReport:
    """Outcome of one ``apply_migration`` call."""

    migration_id: str
    already_applied: bool
    products_scanned: int
    metadata_rows_inserted: int
    source_counts: dict[str, int] = field(default_factory=dict)
    protected_rows: int = 0
    confirmed_overrides: int = 0
    mismatched_overrides: list[OverrideMismatch] = field(default_factory=list)
    missing_products: list[str] = field(default_factory=list)
    seed_rows_loaded: int = 0


# ---------------------------------------------------------------------------
# CSV helpers (self-contained: no import of the legacy sync module)
# ---------------------------------------------------------------------------
def _parse_optional_float(value: object) -> float | None:
    """Parse a CSV numeric cell, tolerating blanks, padded spaces and commas.

    Returns ``None`` for empty or unparseable input. Comma decimals
    (``"2,5"``) are normalised to dots before parsing.
    """
    if value is None:
        return None
    text = _clean_text(value).replace(" ", "").replace(",", ".")
    if not text or text.lower() in ("none", "null", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _clean_text(value: object) -> str:
    """Return a trimmed string, or ``""`` when the value is missing.

    Mirrors the legacy reader: cells whose first character is ``= + - @ \\t \\r``
    are written with a protective leading apostrophe to stop spreadsheet
    formula injection (``sanitize_csv_cell``) and must be restored on read
    (``desanitize_csv_cell``), otherwise ``'-5`` would not parse as ``-5``.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > 1 and text[0] == "'" and text[1] in ("=", "+", "-", "@", "\t", "\r"):
        return text[1:]
    return text


def _clean_product_key(value: object) -> str:
    """Return a trimmed product key, or ``""`` when the cell is empty."""
    return _clean_text(value)


def _parse_field_values(row: dict[str, object]) -> dict[str, object]:
    """Normalise the editable fields of one CSV row.

    Empty cells become ``None`` so "unknown" is never confused with ``0``.
    """
    parsed: dict[str, object] = {}
    for name in FIELD_NAMES:
        if name in _NUMERIC_FIELDS:
            parsed[name] = _parse_optional_float(row.get(name))
        else:
            text = _clean_text(row.get(name))
            parsed[name] = text or None
    return parsed


def _parse_override_row(row: dict[str, object]) -> dict[str, object]:
    """Normalise one override CSV row, including its ``non_food`` flag."""
    parsed = _parse_field_values(row)
    parsed["non_food"] = _clean_text(row.get("non_food")).lower() in TRUTHY
    return parsed


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read a catalog CSV, returning normalised rows and the delimiter used.

    Returns ``([], ",")`` for a missing, ``None`` or empty file. Raises
    ``ValueError`` when the file exists and is non-empty but has no
    ``product_key`` column: a silently empty result would permanently record
    the wrong provenance, because this migration only runs once.
    """
    if path is None:
        return [], ","
    file_path = Path(path)
    if not file_path.is_file():
        return [], ","

    raw = file_path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    if not text.strip():
        return [], ","

    first_line = text.strip().splitlines()[0]
    delimiter = ";" if ";" in first_line and "," not in first_line else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    header = [(name or "").strip().lower() for name in (reader.fieldnames or [])]
    if "product_key" not in header:
        raise ValueError(
            f"{file_path}: missing required 'product_key' column "
            f"(header found: {header or 'none'})"
        )

    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append({
            (key or "").strip().lower(): (value if value is not None else "")
            for key, value in row.items()
            if key is not None
        })
    return rows, delimiter


def _load_overrides(path: Path | str | None) -> dict[str, dict[str, object]]:
    """Load the overrides CSV keyed by product key.

    Missing/``None`` path means "no overrides". Rows with an empty product key
    or an Excel scientific-notation key are skipped: those keys are corrupt by
    construction and can never match a stored product.

    Duplicate keys keep the last row, matching the legacy reader's behaviour.
    """
    rows, _delimiter = _read_csv_rows(path)
    overrides: dict[str, dict[str, object]] = {}
    for row in rows:
        product_key = _clean_product_key(row.get("product_key"))
        if not product_key or SCI_NOTATION_RE.match(product_key):
            continue
        overrides[product_key] = _parse_override_row(row)
    return overrides


def _load_seed(path: Path | str | None) -> dict[str, dict[str, object]]:
    """Load the seed catalog as per-field evidence, keyed by product key.

    Only the values themselves are kept. They are used to prove that a stored
    value really came from the seed file, instead of trusting a row-level
    ``data_source`` flag that the seed upsert itself mixes per field.
    """
    rows, _delimiter = _read_csv_rows(path)
    seed: dict[str, dict[str, object]] = {}
    for row in rows:
        product_key = _clean_product_key(row.get("product_key"))
        if not product_key or SCI_NOTATION_RE.match(product_key):
            continue
        # The legacy seed loader keeps the FIRST row for a duplicated key
        # (``sync_incremental.py`` seed upsert), so evidence must do the same.
        seed.setdefault(product_key, _parse_field_values(row))
    return seed


# ---------------------------------------------------------------------------
# Provenance decisions
# ---------------------------------------------------------------------------
def _is_manual_confirmation(
    field_name: str,
    current_value: object,
    override_row: dict[str, object] | None,
) -> bool:
    """Return True when the override CSV explicitly confirms the stored value.

    A confirmation requires a non-empty override cell that equals the current
    value exactly (strings trimmed, numbers as floats). ``None`` never equals.
    """
    if not override_row:
        return False

    if field_name in CATEGORY_FIELDS and override_row.get("non_food"):
        # A truthy ``non_food`` flag confirms the canonical non-food category
        # labels that the legacy override step writes for those products.
        return _clean_text(current_value) == "Non-Food"

    override_value = override_row.get(field_name)
    if override_value is None:
        return False

    if isinstance(override_value, float):
        if current_value is None or isinstance(current_value, bool):
            return False
        try:
            return float(current_value) == override_value
        except (TypeError, ValueError):
            return False

    return _clean_text(current_value) == str(override_value)


def _resolve_source(
    *,
    field_name: str,
    current_value: object,
    override_row: dict[str, object] | None,
    seed_row: dict[str, object] | None,
    match_status: object,
    data_source: object,
) -> tuple[str, bool]:
    """Return ``(source, is_protected)`` for one non-NULL stored value.

    Precedence: MANUAL > AI > SEED > OFF > LEGACY_UNKNOWN.

    ``SEED`` requires positive per-field evidence: the seed catalog must hold
    the same value for that product and field. A row-level ``data_source`` is
    not enough, because the seed upsert fills columns with ``COALESCE`` and an
    OFF row can therefore contain seed values (and vice versa).

    Protection is field-level and earned, not inherited: only a value that the
    overrides CSV explicitly confirms (``MANUAL``) is protected. The legacy
    ``is_verified`` flag is product-level and true for almost every enriched
    product, so inheriting it would mark every field protected and make the
    flag meaningless. Legacy values are never modified by this migration.
    """
    if _is_manual_confirmation(field_name, current_value, override_row):
        return SOURCE_MANUAL, True

    status = _clean_text(match_status)
    origin = _clean_text(data_source)

    seed_value = (seed_row or {}).get(field_name)
    seed_confirms = seed_value is not None and _values_equal(
        field_name, current_value, seed_value
    )

    if status == AI_MATCH_STATUS:
        source = SOURCE_AI
    elif seed_confirms:
        source = SOURCE_SEED
    elif origin == "OFF_API" or status in OFF_MATCH_STATUSES:
        source = SOURCE_OFF
    else:
        source = SOURCE_LEGACY_UNKNOWN

    return source, False


def _values_equal(field_name: str, current_value: object, override_value: object) -> bool:
    """Compare a stored value with a non-empty override value (exact match)."""
    if current_value is None:
        return False
    if isinstance(override_value, float):
        if isinstance(current_value, bool):
            return False
        try:
            return float(current_value) == override_value
        except (TypeError, ValueError):
            return False
    return _clean_text(current_value) == str(override_value)


# ---------------------------------------------------------------------------
# Migration entry point
# ---------------------------------------------------------------------------
def _insert_migration_evidence(con: duckdb.DuckDBPyConnection, details: str) -> None:
    """Record that this migration ran (single evidence row)."""
    con.execute(_INSERT_EVIDENCE_SQL, [MIGRATION_ID, details])


def apply_migration(
    con: duckdb.DuckDBPyConnection,
    overrides_path: Path | str | None = DEFAULT_OVERRIDES_PATH,
    seed_path: Path | str | None = DEFAULT_SEED_PATH,
) -> MigrationReport:
    """Backfill ``product_field_metadata`` from the current ``nutrition_cache``.

    Runs in one explicit transaction. If the migration id is already recorded,
    nothing is read or written and ``already_applied`` is True: existing
    metadata is never rewritten.

    Args:
        con: Open DuckDB connection (``nutrition_cache`` must already exist).
        overrides_path: Path to ``catalog_overrides.csv``; ``None`` or a missing
            file simply means "no overrides are available".
        seed_path: Path to the seed catalog used as per-field evidence for
            ``SEED`` provenance; ``None`` or a missing file means "no seed
            evidence is available" and such fields fall back to
            ``OFF``/``LEGACY_UNKNOWN`` instead of being labelled ``SEED``.

    Returns:
        A :class:`MigrationReport` describing this run.

    Raises:
        Exception: any failure is propagated after rolling back everything.
    """
    con.execute("BEGIN TRANSACTION")
    try:
        create_schema(con)

        if con.execute(_MIGRATION_EXISTS_SQL, [MIGRATION_ID]).fetchone() is not None:
            con.execute("COMMIT")
            return MigrationReport(
                migration_id=MIGRATION_ID,
                already_applied=True,
                products_scanned=0,
                metadata_rows_inserted=0,
            )

        rows = [
            dict(zip(_SCAN_COLUMNS, raw_row))
            for raw_row in con.execute(_SCAN_SQL).fetchall()
        ]
        current_by_product = {row["product_key"]: row for row in rows}

        overrides = _load_overrides(overrides_path)
        seed = _load_seed(seed_path)

        existing_metadata = {
            (key, name)
            for key, name in con.execute(_EXISTING_METADATA_SQL).fetchall()
        }

        candidates: list[tuple[str, str, str, bool]] = []
        confirmed_overrides = 0

        for row in rows:
            product_key = row["product_key"]
            override_row = overrides.get(product_key)
            for field_name in FIELD_NAMES:
                current_value = row[field_name]
                if current_value is None:
                    # NULL means "unknown": we never claim provenance for a
                    # value that does not exist. 0.0 is a real value.
                    continue
                source, is_protected = _resolve_source(
                    field_name=field_name,
                    current_value=current_value,
                    override_row=override_row,
                    seed_row=seed.get(product_key),
                    match_status=row["match_status"],
                    data_source=row["data_source"],
                )
                if source == SOURCE_MANUAL:
                    confirmed_overrides += 1
                if (product_key, field_name) in existing_metadata:
                    continue
                candidates.append((product_key, field_name, source, is_protected))

        if candidates:
            rows_before = con.execute(_COUNT_METADATA_SQL).fetchone()[0]
            con.execute(_STAGE_SQL)
            con.executemany(_STAGE_INSERT_SQL, candidates)
            con.execute(_INSERT_METADATA_SQL)
            inserted_rows = con.execute(_COUNT_METADATA_SQL).fetchone()[0] - rows_before
            con.execute(f"DROP TABLE IF EXISTS {_STAGE_TABLE}")
        else:
            inserted_rows = 0

        missing_products: set[str] = set()
        mismatched: list[OverrideMismatch] = []
        reported: set[tuple[str, str]] = set()
        for product_key in sorted(overrides):
            current_row = current_by_product.get(product_key)
            if current_row is None:
                missing_products.add(product_key)
                continue
            override_row = overrides[product_key]
            for field_name in FIELD_NAMES:
                override_value = override_row.get(field_name)
                if override_value is None:
                    continue
                current_value = current_row[field_name]
                if not _values_equal(field_name, current_value, override_value):
                    reported.add((product_key, field_name))
                    mismatched.append(
                        OverrideMismatch(
                            product_key=product_key,
                            field_name=field_name,
                            override_value=override_value,
                            current_value=current_value,
                        )
                    )

            # A truthy ``non_food`` flag is an implicit claim that the product
            # is canonical ``Non-Food``. When the stored categories say
            # otherwise, that contradiction is reported (never applied).
            if override_row.get("non_food"):
                for field_name in CATEGORY_FIELDS:
                    if (product_key, field_name) in reported:
                        continue
                    current_value = current_row[field_name]
                    if _clean_text(current_value) != "Non-Food":
                        mismatched.append(
                            OverrideMismatch(
                                product_key=product_key,
                                field_name=field_name,
                                override_value="Non-Food",
                                current_value=current_value,
                            )
                        )

        source_counts: dict[str, int] = {}
        for _key, _name, source, _protected in candidates:
            source_counts[source] = source_counts.get(source, 0) + 1
        protected_rows = sum(1 for candidate in candidates if candidate[3])

        details = (
            f"metadata_rows={inserted_rows};"
            f"products_scanned={len(rows)};"
            f"mismatched_overrides={len(mismatched)}"
        )
        _insert_migration_evidence(con, details)

        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            # The transaction may already be aborted by DuckDB; the original
            # error is what callers need to see.
            pass
        raise

    return MigrationReport(
        migration_id=MIGRATION_ID,
        already_applied=False,
        products_scanned=len(rows),
        metadata_rows_inserted=inserted_rows,
        source_counts=source_counts,
        protected_rows=protected_rows,
        confirmed_overrides=confirmed_overrides,
        mismatched_overrides=mismatched,
        missing_products=sorted(missing_products),
        seed_rows_loaded=len(seed),
    )
