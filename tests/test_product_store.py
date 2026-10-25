"""Tests for the product store provenance metadata (Step 1: storage + migration)."""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb

# Add project root to sys.path so the package can be imported like the rest of
# the test suite does (see tests/test_barcode_extract.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from product_store import (  # noqa: E402
    FIELD_NAMES,
    MIGRATION_ID,
    apply_migration,
    create_schema,
    validate_field_name,
    validate_source,
)
from product_store import migration as migration_module  # noqa: E402

# Copy of the production nutrition_cache definition (sync_incremental.py).
NUTRITION_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS nutrition_cache (
    product_key VARCHAR PRIMARY KEY,
    sample_name VARCHAR,
    ean VARCHAR,
    off_product_name VARCHAR,
    off_brands VARCHAR,
    macro_category VARCHAR,
    category VARCHAR,
    nutriscore_grade VARCHAR,
    nutriscore_score INTEGER,
    nova_group INTEGER,
    kcal_100g DOUBLE,
    proteins_100g DOUBLE,
    carbs_100g DOUBLE,
    sugars_100g DOUBLE,
    fat_100g DOUBLE,
    saturated_fat_100g DOUBLE,
    fiber_100g DOUBLE,
    salt_100g DOUBLE,
    package_weight_g DOUBLE,
    is_weighed_default BOOLEAN DEFAULT FALSE,
    match_status VARCHAR,
    data_source VARCHAR DEFAULT 'BON_ONLY',
    is_verified BOOLEAN DEFAULT FALSE,
    atwater_debug VARCHAR,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_PRODUCT_SQL = """
INSERT INTO nutrition_cache (
    product_key, macro_category, category, package_weight_g, kcal_100g,
    proteins_100g, carbs_100g, sugars_100g, fat_100g, saturated_fat_100g,
    fiber_100g, salt_100g, match_status, data_source, is_verified
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

OVERRIDE_HEADER = [
    "product_key",
    "sample_name",
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
    "non_food",
    "times_bought",
    "match_status",
    "off_product_name",
    "atwater_debug",
]


class ProductStoreTestCase(unittest.TestCase):
    """Shared fixtures: a temporary DuckDB database and a nutrition_cache table."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db_path = os.path.join(self._tmp.name, "test.duckdb")
        self.con = duckdb.connect(self.db_path)
        self.con.execute(NUTRITION_CACHE_DDL)
        self.overrides_path = self.tmp_path / "catalog_overrides.csv"
        self.seed_path = self.tmp_path / "catalog_seed.csv"

    def tearDown(self):
        self.con.close()
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def insert_product(self, product_key, *, match_status=None, data_source="BON_ONLY",
                       is_verified=False, **values):
        """Insert one nutrition_cache fixture row; unspecified fields stay NULL."""
        params = [product_key]
        params.extend(values.get(name) for name in FIELD_NAMES)
        params.extend([match_status, data_source, is_verified])
        self.con.execute(INSERT_PRODUCT_SQL, params)

    def write_overrides(self, rows, path=None):
        """Write an overrides CSV with the production header and given rows."""
        target = path or self.overrides_path
        with open(target, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OVERRIDE_HEADER, restval="")
            writer.writeheader()
            writer.writerows(rows)

    def write_seed(self, rows):
        """Write a seed catalog (same header as production) used as evidence."""
        with open(self.seed_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OVERRIDE_HEADER, restval="")
            writer.writeheader()
            writer.writerows(rows)

    def metadata_row(self, product_key, field_name):
        return self.con.execute(
            "SELECT source, is_protected, provider, model FROM product_field_metadata "
            "WHERE product_key = ? AND field_name = ?",
            [product_key, field_name],
        ).fetchone()

    def metadata_count(self, product_key=None):
        if product_key is None:
            query = "SELECT COUNT(*) FROM product_field_metadata"
            return self.con.execute(query).fetchone()[0]
        return self.con.execute(
            "SELECT COUNT(*) FROM product_field_metadata WHERE product_key = ?",
            [product_key],
        ).fetchone()[0]

    def table_row_count(self, table_name):
        """Row count for a table that may not exist (rolled-back DDL)."""
        exists = self.con.execute(
            "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?", [table_name]
        ).fetchone()[0]
        if not exists:
            return 0
        return self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    # -- tests -----------------------------------------------------------
    def test_create_schema_is_idempotent(self):
        create_schema(self.con)
        create_schema(self.con)
        tables = {
            row[0]
            for row in self.con.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        }
        self.assertIn("product_field_metadata", tables)
        self.assertIn("schema_migrations", tables)
        self.assertIn("ai_enrichment_state", tables)

    def test_metadata_primary_key_is_enforced(self):
        create_schema(self.con)
        insert_sql = (
            "INSERT INTO product_field_metadata (product_key, field_name, source) "
            "VALUES (?, ?, ?)"
        )
        self.con.execute(insert_sql, ["1111", "kcal_100g", "SEED"])
        with self.assertRaises(duckdb.Error):
            self.con.execute(insert_sql, ["1111", "kcal_100g", "OFF"])

    def test_migration_preserves_values_and_product_count(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.insert_product("2222", kcal_100g=None, data_source="OFF_API",
                            match_status="MATCHED_OFF")
        self.write_overrides([{"product_key": "1111", "kcal_100g": "100"}])

        before = self.con.execute(
            "SELECT * FROM nutrition_cache ORDER BY product_key"
        ).fetchall()

        apply_migration(self.con, self.overrides_path, self.seed_path)

        after = self.con.execute(
            "SELECT * FROM nutrition_cache ORDER BY product_key"
        ).fetchall()
        self.assertEqual(before, after)
        self.assertEqual(len(after), 2)

    def test_provenance_mapping_per_source(self):
        self.insert_product("1111", kcal_100g=100.0, salt_100g=1.5,
                            data_source="SEED_CATALOG")
        self.insert_product("2222", kcal_100g=200.0, data_source="SEED_CATALOG")
        self.insert_product("3333", kcal_100g=300.0, data_source="OFF_API",
                            match_status="MATCHED_OFF")
        self.insert_product("4444", kcal_100g=400.0, data_source="SEED_CATALOG",
                            match_status="AI_ESTIMATED", is_verified=True)
        self.insert_product("5555", kcal_100g=500.0, data_source="RULE_NON_FOOD")
        # Override confirms 1111's kcal_100g -> MANUAL / protected.
        self.write_overrides([{"product_key": "1111", "kcal_100g": "100"}])
        # Seed evidence is per field: only values that match the seed file are
        # labelled SEED.
        self.write_seed([
            {"product_key": "1111", "salt_100g": "1.5"},
            {"product_key": "2222", "kcal_100g": "200"},
        ])

        report = apply_migration(self.con, self.overrides_path, self.seed_path)

        manual = self.metadata_row("1111", "kcal_100g")
        self.assertEqual(manual, ("MANUAL", True, None, None))

        seed = self.metadata_row("2222", "kcal_100g")
        self.assertEqual(seed, ("SEED", False, None, None))

        off = self.metadata_row("3333", "kcal_100g")
        self.assertEqual(off, ("OFF", False, None, None))

        # AI wins over the seed data_source on the same row. The legacy
        # is_verified flag must NOT make the field protected.
        ai = self.metadata_row("4444", "kcal_100g")
        self.assertEqual(ai, ("AI", False, None, None))

        legacy = self.metadata_row("5555", "kcal_100g")
        self.assertEqual(legacy, ("LEGACY_UNKNOWN", False, None, None))

        # A non-confirmed field of the manual product keeps its real source.
        self.assertEqual(self.metadata_row("1111", "salt_100g"),
                         ("SEED", False, None, None))

        self.assertEqual(report.confirmed_overrides, 1)
        self.assertFalse(report.already_applied)
        self.assertEqual(report.source_counts["MANUAL"], 1)
        # Only the MANUAL confirmation is protected; 4444 is is_verified=TRUE
        # in the legacy table and still must not be protected.
        self.assertEqual(report.protected_rows, 1)

    def test_legacy_is_verified_does_not_protect_fields(self):
        self.insert_product("1212", kcal_100g=120.0, salt_100g=0.5,
                            data_source="SEED_CATALOG", is_verified=True)
        self.write_seed([{"product_key": "1212", "kcal_100g": "120", "salt_100g": "0.5"}])

        report = apply_migration(self.con, None, self.seed_path)

        self.assertEqual(self.metadata_row("1212", "kcal_100g"),
                         ("SEED", False, None, None))
        self.assertEqual(self.metadata_row("1212", "salt_100g"),
                         ("SEED", False, None, None))
        self.assertEqual(report.protected_rows, 0)

    def test_conflicting_override_does_not_become_manual(self):
        self.insert_product("8888", kcal_100g=100.0, data_source="OFF_API",
                            match_status="MATCHED_OFF")
        self.write_overrides([{"product_key": "8888", "kcal_100g": "50"}])

        report = apply_migration(self.con, self.overrides_path, self.seed_path)

        row = self.metadata_row("8888", "kcal_100g")
        self.assertEqual(row[0], "OFF")
        self.assertNotEqual(row[0], "MANUAL")
        self.assertEqual(len(report.mismatched_overrides), 1)
        mismatch = report.mismatched_overrides[0]
        self.assertEqual(mismatch.product_key, "8888")
        self.assertEqual(mismatch.field_name, "kcal_100g")
        self.assertEqual(mismatch.override_value, 50.0)
        self.assertEqual(mismatch.current_value, 100.0)
        self.assertEqual(report.confirmed_overrides, 0)

        stored = self.con.execute(
            "SELECT kcal_100g FROM nutrition_cache WHERE product_key = '8888'"
        ).fetchone()[0]
        self.assertEqual(stored, 100.0)

    def test_override_for_missing_product_is_reported(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.write_overrides([{"product_key": "9999", "kcal_100g": "10"}])

        report = apply_migration(self.con, self.overrides_path, self.seed_path)

        self.assertIn("9999", report.missing_products)
        self.assertEqual(self.metadata_count("9999"), 0)

    def test_null_vs_zero(self):
        self.insert_product("7777", kcal_100g=0.0, data_source="SEED_CATALOG")
        self.write_seed([{"product_key": "7777", "kcal_100g": "0"}])

        apply_migration(self.con, None, self.seed_path)

        self.assertEqual(self.metadata_row("7777", "kcal_100g"),
                         ("SEED", False, None, None))
        self.assertIsNone(self.metadata_row("7777", "salt_100g"))
        self.assertEqual(self.metadata_count("7777"), 1)

    def test_second_run_is_idempotent(self):
        self.insert_product("1111", kcal_100g=100.0, salt_100g=2.0,
                            data_source="SEED_CATALOG")
        self.insert_product("2222", kcal_100g=200.0, data_source="OFF_API")

        first = apply_migration(self.con, None, self.seed_path)
        self.assertFalse(first.already_applied)
        self.assertGreater(first.metadata_rows_inserted, 0)

        count_before = self.metadata_count()
        updated_at_before = self.con.execute(
            "SELECT product_key, field_name, updated_at FROM product_field_metadata "
            "ORDER BY product_key, field_name"
        ).fetchall()

        second = apply_migration(self.con, None, self.seed_path)

        self.assertTrue(second.already_applied)
        self.assertEqual(second.metadata_rows_inserted, 0)
        self.assertEqual(self.metadata_count(), count_before)
        updated_at_after = self.con.execute(
            "SELECT product_key, field_name, updated_at FROM product_field_metadata "
            "ORDER BY product_key, field_name"
        ).fetchall()
        self.assertEqual(updated_at_before, updated_at_after)

    def test_rollback_on_failure(self):
        self.insert_product("1111", kcal_100g=100.0, salt_100g=2.0,
                            data_source="SEED_CATALOG")

        with mock.patch.object(
            migration_module,
            "_insert_migration_evidence",
            side_effect=RuntimeError("forced failure"),
        ):
            with self.assertRaises(RuntimeError):
                apply_migration(self.con, None, self.seed_path)

        self.assertEqual(self.table_row_count("product_field_metadata"), 0)
        self.assertEqual(self.table_row_count("schema_migrations"), 0)

        report = apply_migration(self.con, None, self.seed_path)
        self.assertFalse(report.already_applied)
        self.assertGreater(report.metadata_rows_inserted, 0)
        self.assertEqual(self.table_row_count("schema_migrations"), 1)
        self.assertEqual(
            self.con.execute(
                "SELECT migration_id FROM schema_migrations"
            ).fetchone()[0],
            MIGRATION_ID,
        )

    def test_manual_non_food_override_is_manual(self):
        self.insert_product("6666", macro_category="Non-Food", category="Non-Food",
                            data_source="RULE_NON_FOOD", match_status="NON_FOOD")
        self.write_overrides([{"product_key": "6666", "non_food": "1"}])

        report = apply_migration(self.con, self.overrides_path, self.seed_path)

        self.assertEqual(self.metadata_row("6666", "macro_category"),
                         ("MANUAL", True, None, None))
        self.assertEqual(self.metadata_row("6666", "category"),
                         ("MANUAL", True, None, None))
        self.assertEqual(report.confirmed_overrides, 2)

    def test_validation_rejects_unknown_field_and_source(self):
        with self.assertRaises(ValueError):
            validate_field_name("sample_name")
        with self.assertRaises(ValueError):
            validate_source("HUMAN")

    def test_no_metadata_for_sample_name(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")

        apply_migration(self.con, None, self.seed_path)

        count = self.con.execute(
            "SELECT COUNT(*) FROM product_field_metadata WHERE field_name = 'sample_name'"
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_migration_does_not_require_overrides_file(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="OFF_API",
                            match_status="MATCHED_OFF")
        missing_path = self.tmp_path / "does_not_exist.csv"

        report = apply_migration(self.con, missing_path, self.seed_path)

        self.assertFalse(report.already_applied)
        self.assertEqual(report.products_scanned, 1)
        self.assertEqual(report.confirmed_overrides, 0)
        self.assertEqual(report.mismatched_overrides, [])
        self.assertEqual(report.missing_products, [])
        self.assertEqual(self.metadata_row("1111", "kcal_100g"),
                         ("OFF", False, None, None))

    # -- per-field seed evidence -----------------------------------------
    def test_seed_evidence_requires_matching_value(self):
        # Same row-level flag, two fields: only the one present in the seed
        # file with the same value is SEED.
        self.insert_product("1111", kcal_100g=100.0, salt_100g=2.0,
                            data_source="SEED_CATALOG")
        self.write_seed([{"product_key": "1111", "kcal_100g": "100"}])

        report = apply_migration(self.con, None, self.seed_path)

        self.assertEqual(report.seed_rows_loaded, 1)
        self.assertEqual(self.metadata_row("1111", "kcal_100g"),
                         ("SEED", False, None, None))
        self.assertEqual(self.metadata_row("1111", "salt_100g"),
                         ("LEGACY_UNKNOWN", False, None, None))

    def test_off_row_can_still_prove_seed_value(self):
        self.insert_product("2222", kcal_100g=210.0, data_source="OFF_API",
                            match_status="MATCHED_OFF")
        self.write_seed([{"product_key": "2222", "kcal_100g": "210"}])

        apply_migration(self.con, None, self.seed_path)

        self.assertEqual(self.metadata_row("2222", "kcal_100g"),
                         ("SEED", False, None, None))

    def test_seed_value_difference_is_not_seed(self):
        self.insert_product("3333", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.write_seed([{"product_key": "3333", "kcal_100g": "90"}])

        apply_migration(self.con, None, self.seed_path)

        self.assertEqual(self.metadata_row("3333", "kcal_100g"),
                         ("LEGACY_UNKNOWN", False, None, None))

    def test_duplicate_seed_keys_keep_first_row(self):
        # The legacy seed loader keeps the first row for a duplicated key.
        self.insert_product("4444", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.write_seed([
            {"product_key": "4444", "kcal_100g": "100"},
            {"product_key": "4444", "kcal_100g": "200"},
        ])

        apply_migration(self.con, None, self.seed_path)

        self.assertEqual(self.metadata_row("4444", "kcal_100g"),
                         ("SEED", False, None, None))

    # -- override CSV edge cases -----------------------------------------
    def test_comma_decimal_override_confirms_manual(self):
        self.insert_product("1111", package_weight_g=2.5, data_source="SEED_CATALOG")
        self.write_overrides([{"product_key": "1111", "package_weight_g": "2,5"}])

        report = apply_migration(self.con, self.overrides_path, None)

        self.assertEqual(self.metadata_row("1111", "package_weight_g"),
                         ("MANUAL", True, None, None))
        self.assertEqual(report.confirmed_overrides, 1)
        self.assertEqual(report.mismatched_overrides, [])

    def test_sanitized_negative_number_is_desanitized(self):
        self.insert_product("1111", fiber_100g=-5.0, data_source="SEED_CATALOG")
        self.write_overrides([{"product_key": "1111", "fiber_100g": "'-5"}])

        report = apply_migration(self.con, self.overrides_path, None)

        self.assertEqual(self.metadata_row("1111", "fiber_100g"),
                         ("MANUAL", True, None, None))
        self.assertEqual(report.confirmed_overrides, 1)
        self.assertEqual(report.mismatched_overrides, [])

    def test_semicolon_delimited_overrides_csv(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        path = self.tmp_path / "overrides_semicolon.csv"
        path.write_text(
            "product_key;kcal_100g\n1111;100\n",
            encoding="utf-8-sig",
            newline="",
        )

        report = apply_migration(self.con, path, None)

        self.assertEqual(self.metadata_row("1111", "kcal_100g"),
                         ("MANUAL", True, None, None))
        self.assertEqual(report.confirmed_overrides, 1)

    def test_malformed_overrides_header_raises(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        path = self.tmp_path / "broken.csv"
        path.write_text("kcal_100g\n100\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            apply_migration(self.con, path, None)

        self.assertEqual(self.table_row_count("product_field_metadata"), 0)
        self.assertEqual(self.table_row_count("schema_migrations"), 0)

    def test_duplicate_override_keys_last_row_wins(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.write_overrides([
            {"product_key": "1111", "kcal_100g": "100"},
            {"product_key": "1111", "kcal_100g": "200"},
        ])

        report = apply_migration(self.con, self.overrides_path, None)

        self.assertEqual(self.metadata_row("1111", "kcal_100g")[0], "LEGACY_UNKNOWN")
        self.assertEqual(len(report.mismatched_overrides), 1)
        mismatch = report.mismatched_overrides[0]
        self.assertEqual((mismatch.override_value, mismatch.current_value),
                         (200.0, 100.0))
        self.assertEqual(report.confirmed_overrides, 0)

    def test_non_food_flag_conflict_is_reported(self):
        # Mirrors the real row 8585002461889: non_food=1 but a food category.
        self.insert_product("4242", macro_category="Pantry & Groceries",
                            category="Sauces, Oils & Condiments",
                            kcal_100g=250.0, data_source="SEED_CATALOG")
        self.write_overrides([{"product_key": "4242", "non_food": "1"}])

        report = apply_migration(self.con, self.overrides_path, None)

        fields = {m.field_name for m in report.mismatched_overrides}
        self.assertEqual(fields, {"macro_category", "category"})
        self.assertEqual(report.confirmed_overrides, 0)
        self.assertEqual(report.protected_rows, 0)
        self.assertEqual(self.metadata_row("4242", "category")[0], "LEGACY_UNKNOWN")
        stored = self.con.execute(
            "SELECT category FROM nutrition_cache WHERE product_key = '4242'"
        ).fetchone()[0]
        self.assertEqual(stored, "Sauces, Oils & Condiments")

    # -- migration edge cases --------------------------------------------
    def test_migration_on_empty_nutrition_cache(self):
        report = apply_migration(self.con, None, None)

        self.assertFalse(report.already_applied)
        self.assertEqual((report.products_scanned, report.metadata_rows_inserted), (0, 0))
        self.assertEqual(report.source_counts, {})
        self.assertEqual(self.metadata_count(), 0)
        self.assertEqual(self.table_row_count("schema_migrations"), 1)

    def test_product_with_all_fields_null_gets_no_metadata(self):
        self.insert_product("9090", data_source="SEED_CATALOG")

        report = apply_migration(self.con, None, None)

        self.assertEqual(report.products_scanned, 1)
        self.assertEqual(report.metadata_rows_inserted, 0)
        self.assertEqual(self.metadata_count("9090"), 0)

    def test_provider_and_model_stay_null_everywhere(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.insert_product("2222", kcal_100g=200.0, data_source="OFF_API",
                            match_status="MATCHED_OFF")
        self.insert_product("3333", kcal_100g=300.0, data_source="SEED_CATALOG")
        self.write_overrides([{"product_key": "3333", "kcal_100g": "300"}])

        apply_migration(self.con, self.overrides_path, None)

        rows = self.con.execute(
            "SELECT provider, model FROM product_field_metadata"
        ).fetchall()
        self.assertTrue(rows)
        self.assertFalse(any(p is not None or m is not None for p, m in rows))

    def test_already_applied_ignores_a_different_overrides_file(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.write_overrides([{"product_key": "1111", "kcal_100g": "100"}])
        first = apply_migration(self.con, self.overrides_path, None)
        self.assertFalse(first.already_applied)

        conflicting = self.tmp_path / "conflicting.csv"
        self.write_overrides([{"product_key": "1111", "kcal_100g": "55"}], path=conflicting)
        second = apply_migration(self.con, conflicting, None)

        self.assertTrue(second.already_applied)
        self.assertEqual(second.metadata_rows_inserted, 0)
        self.assertEqual(second.mismatched_overrides, [])
        self.assertEqual(second.missing_products, [])
        self.assertEqual(self.table_row_count("schema_migrations"), 1)
        self.assertEqual(self.metadata_row("1111", "kcal_100g"),
                         ("MANUAL", True, None, None))

    def test_stage_insert_failure_rolls_back_and_cleans_temp_table(self):
        self.insert_product("1111", kcal_100g=100.0, data_source="SEED_CATALOG")
        self.insert_product("2222", kcal_100g=200.0, data_source="OFF_API")

        with mock.patch.object(
            migration_module,
            "_STAGE_INSERT_SQL",
            "INSERT INTO __pfm_missing_table__ VALUES (?, ?, ?, ?)",
        ):
            with self.assertRaises(duckdb.Error):
                apply_migration(self.con, None, None)

        self.assertEqual(self.table_row_count("product_field_metadata"), 0)
        self.assertEqual(self.table_row_count("schema_migrations"), 0)
        stage_exists = self.con.execute(
            "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?",
            [migration_module._STAGE_TABLE],
        ).fetchone()[0]
        self.assertEqual(stage_exists, 0)

        # The same connection must work for a clean retry.
        report = apply_migration(self.con, None, None)
        self.assertFalse(report.already_applied)
        self.assertEqual(self.metadata_count(), 2)

    def test_source_check_constraint_blocks_invalid_value(self):
        create_schema(self.con)
        insert_sql = (
            "INSERT INTO product_field_metadata (product_key, field_name, source) "
            "VALUES (?, ?, ?)"
        )
        with self.assertRaises(duckdb.Error):
            self.con.execute(insert_sql, ["1", "kcal_100g", "HUMAN"])

        self.con.execute(insert_sql, ["1", "kcal_100g", "SEED"])
        is_protected, updated_at_is_set = self.con.execute(
            "SELECT is_protected, updated_at IS NOT NULL FROM product_field_metadata"
        ).fetchone()
        self.assertFalse(is_protected)
        self.assertTrue(updated_at_is_set)


if __name__ == "__main__":
    unittest.main()
