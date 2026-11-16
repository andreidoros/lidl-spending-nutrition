"""Tests for UI-independent product completeness and manual-edit services."""

import csv
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from product_store import (  # noqa: E402
    AI_STATE_TABLE,
    FIELD_NAMES,
    REQUIRED_FOOD_NUTRIENTS,
    clear_implausible_nutrition,
    create_schema,
    list_incomplete_products,
    atwater_check,
    list_review_products,
    missing_required_fields,
    regenerate_review_csv,
    review_status_text,
    save_manual_fields,
    synchronize_field_metadata,
)
import sync_incremental as sync  # noqa: E402

TAXONOMY = {
    "Fresh Food": ["Dairy & Eggs"],
    "Pantry & Groceries": ["Sauces, Oils & Condiments", "Herbs & Spices"],
    "Non-Food": ["Non-Food"],
}
CATEGORY_TO_MACRO = {category: macro for macro, categories in TAXONOMY.items() for category in categories}

NUTRITION_DDL = """
CREATE TABLE nutrition_cache (
 product_key VARCHAR PRIMARY KEY, sample_name VARCHAR, ean VARCHAR,
 off_product_name VARCHAR, off_brands VARCHAR, macro_category VARCHAR,
 category VARCHAR, nutriscore_grade VARCHAR, nutriscore_score INTEGER,
 nova_group INTEGER, kcal_100g DOUBLE, proteins_100g DOUBLE,
 carbs_100g DOUBLE, sugars_100g DOUBLE, fat_100g DOUBLE,
 saturated_fat_100g DOUBLE, fiber_100g DOUBLE, salt_100g DOUBLE,
 package_weight_g DOUBLE, is_weighed_default BOOLEAN DEFAULT FALSE,
 match_status VARCHAR, data_source VARCHAR DEFAULT 'BON_ONLY',
 is_verified BOOLEAN DEFAULT FALSE, atwater_debug VARCHAR,
 cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
PURCHASES_DDL = """
CREATE TABLE purchases (
 ticket_id VARCHAR, date TIMESTAMP, line_idx INTEGER, name VARCHAR,
 quantity DOUBLE, unit_price DOUBLE, line_total DOUBLE, raw_code VARCHAR,
 barcode_type VARCHAR, ean VARCHAR, discount_amount DOUBLE DEFAULT 0,
 net_line_total DOUBLE DEFAULT 0
)
"""
INSERT_PRODUCT = """INSERT INTO nutrition_cache
    (product_key, sample_name, macro_category, category, kcal_100g, proteins_100g,
     carbs_100g, sugars_100g, fat_100g, saturated_fat_100g, fiber_100g,
     salt_100g, package_weight_g, match_status, data_source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

CSV_COLUMNS = (
    "product_key", "sample_name", "macro_category", "category", "package_weight_g",
    "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g", "fat_100g",
    "saturated_fat_100g", "fiber_100g", "salt_100g", "non_food", "times_bought",
    "match_status", "off_product_name", "atwater_debug",
)


class ProductServicesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.con = duckdb.connect(os.path.join(self.tmp.name, "test.duckdb"))
        self.con.execute(NUTRITION_DDL)
        self.con.execute(PURCHASES_DDL)
        self.con.execute("CREATE TABLE catalog_overrides (product_key VARCHAR PRIMARY KEY)")

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def product(self, key, *, name=None, macro="Fresh Food", category="Dairy & Eggs",
                kcal=10, protein=1, carbs=2, sugar=1, fat=1, saturated=0.2,
                fiber=None, salt=0.1, weight=None, status="TEST", source="BON_ONLY"):
        self.con.execute(INSERT_PRODUCT, [key, name or key, macro, category, kcal,
            protein, carbs, sugar, fat, saturated, fiber, salt, weight, status, source])

    def test_food_required_fields_and_optional_fields(self):
        complete = {
            "category": "Dairy & Eggs",
            "macro_category": "Fresh Food",
            "kcal_100g": 0.0,
            "proteins_100g": 0.0,
            "carbs_100g": 0.0,
            "sugars_100g": 0.0,
            "fat_100g": 0.0,
            "saturated_fat_100g": 0.0,
            "salt_100g": 0.0,
            "fiber_100g": None,
            "package_weight_g": None,
        }
        self.assertEqual(missing_required_fields(complete, CATEGORY_TO_MACRO), ())
        incomplete = dict(complete, salt_100g=None)
        self.assertEqual(missing_required_fields(incomplete, CATEGORY_TO_MACRO), ("salt_100g",))

    def test_each_required_nutrient_is_independently_required(self):
        complete = {
            "category": "Dairy & Eggs",
            "macro_category": "Fresh Food",
            **{field: 0.0 for field in REQUIRED_FOOD_NUTRIENTS},
            "fiber_100g": None,
            "package_weight_g": None,
        }
        for field_name in REQUIRED_FOOD_NUTRIENTS:
            with self.subTest(field=field_name):
                product = dict(complete, **{field_name: None})
                self.assertEqual(missing_required_fields(product, CATEGORY_TO_MACRO), (field_name,))

    def test_complete_food_with_macro_category_mismatch_is_incomplete(self):
        product = {
            "category": "Dairy & Eggs",
            "macro_category": "Non-Food",
            **{field: 1.0 for field in REQUIRED_FOOD_NUTRIENTS},
        }
        self.assertEqual(missing_required_fields(product, CATEGORY_TO_MACRO), ("macro_category",))

        self.product("p1", macro="Non-Food")
        incomplete = list_incomplete_products(self.con, CATEGORY_TO_MACRO)
        self.assertEqual(incomplete[0]["product_key"], "p1")
        self.assertIn("macro_category", incomplete[0]["missing_fields"])

    def test_food_requires_valid_taxonomy_category(self):
        fields = {name: 1.0 for name in FIELD_NAMES}
        fields.update(category="Unlisted", macro_category="Fresh Food")
        self.assertEqual(missing_required_fields(fields, CATEGORY_TO_MACRO)[0], "category")

    def test_non_food_does_not_require_nutrients(self):
        self.assertEqual(missing_required_fields(
            {"macro_category": "Non-Food", "category": "Non-Food"}, CATEGORY_TO_MACRO
        ), ())

    def test_non_food_macro_conflict_does_not_exempt_food_category(self):
        product = {"macro_category": "Non-Food", "category": "Dairy & Eggs"}
        self.assertIn("kcal_100g", missing_required_fields(product, CATEGORY_TO_MACRO))

    def test_invalid_category_with_non_food_macro_still_requires_review(self):
        product = {"macro_category": "Non-Food", "category": "Unknown"}
        self.assertEqual(missing_required_fields(product, CATEGORY_TO_MACRO)[0], "category")

    def test_manual_category_edit_derives_macro_and_protects_edited_fields(self):
        self.product("p1", kcal=100)
        edited = save_manual_fields(self.con, "p1", {"category": "Dairy & Eggs"}, CATEGORY_TO_MACRO)
        self.assertEqual(set(edited), {"category", "macro_category"})
        row = self.con.execute(
            "SELECT category, macro_category FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()
        self.assertEqual(row, ("Dairy & Eggs", "Fresh Food"))
        metadata = self.con.execute(
            "SELECT field_name, source, is_protected FROM product_field_metadata ORDER BY field_name"
        ).fetchall()
        self.assertEqual(metadata, [("category", "MANUAL", True), ("macro_category", "MANUAL", True)])
        self.assertIsNone(self.con.execute(
            "SELECT 1 FROM product_field_metadata WHERE field_name='kcal_100g'"
        ).fetchone())

    def test_manual_nutrient_edit_protects_only_that_field_and_keeps_zero(self):
        self.product("p1")
        save_manual_fields(self.con, "p1", {"salt_100g": 0}, CATEGORY_TO_MACRO)
        self.assertEqual(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0], 0.0)
        self.assertEqual(self.con.execute(
            "SELECT field_name, source, is_protected FROM product_field_metadata"
        ).fetchall(), [("salt_100g", "MANUAL", True)])

    def test_existing_protected_field_can_be_explicitly_edited(self):
        self.product("p1", kcal=100)
        save_manual_fields(self.con, "p1", {"kcal_100g": 120}, CATEGORY_TO_MACRO)
        self.assertEqual(self.con.execute(
            "SELECT kcal_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0], 120.0)

    def test_sync_tracks_new_fields_without_rewriting_existing_metadata(self):
        self.product("p1", kcal=100, protein=None, carbs=None, sugar=None, fat=None,
                     saturated=None, salt=None, source="OFF_API", status="MATCHED_OFF")
        create_schema(self.con)
        self.con.execute(
            "INSERT INTO product_field_metadata (product_key, field_name, source, is_protected) "
            "VALUES ('p1', 'kcal_100g', 'MANUAL', TRUE)"
        )
        self.con.execute("UPDATE nutrition_cache SET salt_100g=0 WHERE product_key='p1'")

        inserted = synchronize_field_metadata(self.con, None, None)

        self.assertEqual(inserted, 3)  # macro/category and newly populated salt
        rows = self.con.execute(
            "SELECT field_name, source, is_protected FROM product_field_metadata ORDER BY field_name"
        ).fetchall()
        self.assertEqual(rows, [
            ("category", "OFF", False),
            ("kcal_100g", "MANUAL", True),
            ("macro_category", "OFF", False),
            ("salt_100g", "OFF", False),
        ])

    def test_sync_promotes_a_current_matching_override_to_manual(self):
        self.product("p1", kcal=100, protein=None, carbs=None, sugar=None, fat=None,
                     saturated=None, salt=None, source="OFF_API", status="MATCHED_OFF")
        create_schema(self.con)
        self.con.execute(
            "INSERT INTO product_field_metadata (product_key, field_name, source, is_protected) "
            "VALUES ('p1', 'kcal_100g', 'OFF', FALSE)"
        )
        override_path = self.root / "overrides.csv"
        with override_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["product_key", "kcal_100g"])
            writer.writeheader()
            writer.writerow({"product_key": "p1", "kcal_100g": "100"})

        changed = synchronize_field_metadata(self.con, override_path, None)

        self.assertEqual(changed, 3)  # two missing category metadata rows plus the promotion
        self.assertEqual(self.con.execute(
            "SELECT source, is_protected FROM product_field_metadata WHERE product_key='p1' AND field_name='kcal_100g'"
        ).fetchone(), ("MANUAL", True))

    def test_non_food_edit_clears_nutrients_metadata_but_preserves_weight(self):
        self.product("p1", weight=500)
        save_manual_fields(self.con, "p1", {"category": "Non-Food"}, CATEGORY_TO_MACRO)
        sync.run_deterministic_non_food_filter(self.con)
        values = self.con.execute(
            "SELECT category, macro_category, kcal_100g, salt_100g, package_weight_g "
            "FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()
        self.assertEqual(values, ("Non-Food", "Non-Food", None, None, 500.0))
        self.assertEqual(self.con.execute(
            "SELECT field_name FROM product_field_metadata ORDER BY field_name"
        ).fetchall(), [("category",), ("macro_category",)])

    def test_food_non_food_food_transition_clears_stale_status_and_stays_in_review(self):
        self.product("p1", weight=500)
        self.con.execute("UPDATE nutrition_cache SET is_verified=TRUE WHERE product_key='p1'")

        save_manual_fields(self.con, "p1", {"category": "Non-Food"}, CATEGORY_TO_MACRO)
        save_manual_fields(self.con, "p1", {"category": "Dairy & Eggs"}, CATEGORY_TO_MACRO)

        row = self.con.execute(
            """SELECT category, macro_category, kcal_100g, salt_100g,
                      package_weight_g, match_status, data_source, is_verified
               FROM nutrition_cache WHERE product_key='p1'"""
        ).fetchone()
        self.assertEqual(row, (
            "Dairy & Eggs", "Fresh Food", None, None, 500.0,
            None, None, False,
        ))
        self.assertEqual(
            [product["product_key"] for product in list_incomplete_products(self.con, CATEGORY_TO_MACRO)],
            ["p1"],
        )

    def test_explicit_nutrient_clear_removes_only_that_field_and_metadata(self):
        self.product("p1")
        save_manual_fields(self.con, "p1", {"salt_100g": 0.5, "proteins_100g": 9}, CATEGORY_TO_MACRO)
        self.con.execute("UPDATE nutrition_cache SET is_verified=TRUE WHERE product_key='p1'")

        changed = save_manual_fields(self.con, "p1", {"salt_100g": " "}, CATEGORY_TO_MACRO)

        self.assertEqual(changed, ("salt_100g",))
        self.assertIsNone(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0])
        self.assertEqual(self.con.execute(
            "SELECT field_name, source, is_protected FROM product_field_metadata ORDER BY field_name"
        ).fetchall(), [("proteins_100g", "MANUAL", True)])
        self.assertFalse(self.con.execute(
            "SELECT is_verified FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0])

    def test_explicit_category_clear_clears_derived_macro_metadata(self):
        self.product("p1")
        save_manual_fields(self.con, "p1", {"category": "Dairy & Eggs"}, CATEGORY_TO_MACRO)

        save_manual_fields(self.con, "p1", {"category": None}, CATEGORY_TO_MACRO)

        self.assertEqual(self.con.execute(
            "SELECT category, macro_category FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), (None, None))
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM product_field_metadata WHERE product_key='p1'"
        ).fetchone()[0], 0)
        self.assertIn("category", missing_required_fields(
            {"category": None}, CATEGORY_TO_MACRO
        ))

    def test_init_database_does_not_leave_verified_partial_food(self):
        self.product("p1", salt=None, status="MATCHED_OFF_VERIFIED")
        self.con.execute("UPDATE nutrition_cache SET is_verified=TRUE WHERE product_key='p1'")

        sync.init_database(self.con)

        self.assertFalse(self.con.execute(
            "SELECT is_verified FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0])

    def test_init_database_clears_legacy_non_food_status_on_food_category(self):
        self.product("p1", status="NON_FOOD", source="RULE_NON_FOOD")
        self.con.execute("UPDATE nutrition_cache SET is_verified=TRUE WHERE product_key='p1'")

        sync.init_database(self.con)

        self.assertEqual(self.con.execute(
            "SELECT match_status, data_source, is_verified FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), (None, None, False))

    def test_atwater_skips_missing_core_macros_without_erasing_partial_values(self):
        self.product("p1", protein=None, salt=None, status="MATCHED_OFF")
        create_schema(self.con)
        before = self.con.execute(
            "SELECT kcal_100g, carbs_100g, fat_100g, match_status FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()

        sync.validate_atwater(self.con)

        self.assertEqual(self.con.execute(
            "SELECT kcal_100g, carbs_100g, fat_100g, match_status FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), before)

    def test_partial_verified_food_with_core_macros_is_still_validated_and_reviewed(self):
        self.product("p1", salt=None, status="MATCHED_OFF_VERIFIED")
        self.con.execute("UPDATE nutrition_cache SET is_verified=TRUE WHERE product_key='p1'")
        sync.init_database(self.con)

        sync.validate_atwater(self.con)

        row = self.con.execute(
            "SELECT is_verified, atwater_debug FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()
        self.assertTrue(row[0])
        self.assertIn("OK", row[1])
        self.assertEqual(
            [product["product_key"] for product in list_incomplete_products(self.con, CATEGORY_TO_MACRO)],
            ["p1"],
        )

    def test_atwater_validation_does_not_clear_protected_manual_nutrient(self):
        self.product("p1", kcal=None, protein=None, carbs=None, sugar=None, fat=None,
                     saturated=None, salt=None)
        save_manual_fields(self.con, "p1", {"salt_100g": 0.5}, CATEGORY_TO_MACRO)

        sync.validate_atwater(self.con)

        self.assertEqual(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0], 0.5)

    def test_egg_standard_patch_does_not_overwrite_protected_nutrient(self):
        self.product("egg1", name="Oua 10 buc", kcal=None, protein=None,
                     carbs=None, sugar=None, fat=None, saturated=None, salt=None)
        save_manual_fields(self.con, "egg1", {"salt_100g": 0.5}, CATEGORY_TO_MACRO)

        sync.run_deterministic_non_food_filter(self.con)

        self.assertEqual(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='egg1'"
        ).fetchone()[0], 0.5)

    def test_weight_decoding_does_not_overwrite_protected_package_weight(self):
        self.product("2501234004508", weight=None)
        save_manual_fields(self.con, "2501234004508", {"package_weight_g": 777}, CATEGORY_TO_MACRO)
        original_catalogs = sync.CATALOGS_BASE_DIR
        try:
            sync.CATALOGS_BASE_DIR = self.root / "catalogs"
            sync.CATALOGS_BASE_DIR.mkdir()
            sync.compute_weights_and_quantities(self.con)
        finally:
            sync.CATALOGS_BASE_DIR = original_catalogs
        self.assertEqual(self.con.execute(
            "SELECT package_weight_g FROM nutrition_cache WHERE product_key='2501234004508'"
        ).fetchone()[0], 777.0)

    def test_stale_nutrition_override_cannot_restore_nutrients_on_non_food(self):
        self.product("p1", weight=500)
        save_manual_fields(self.con, "p1", {"category": "Non-Food"}, CATEGORY_TO_MACRO)
        overrides = self.root / "catalog_overrides.csv"
        sync.write_csv_records(overrides, [{"product_key": "p1", "kcal_100g": "100"}],
                               sync.OVERRIDE_COLUMNS)
        original_overrides = sync.OVERRIDE_CSV_PATH
        try:
            sync.OVERRIDE_CSV_PATH = overrides
            sync.apply_nutrition_overrides(self.con, quiet=True)
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides
        self.assertEqual(self.con.execute(
            "SELECT category, kcal_100g, package_weight_g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Non-Food", None, 500.0))

    def test_seed_enrichment_does_not_restore_nutrients_on_protected_non_food(self):
        self.product("p1", weight=500)
        save_manual_fields(self.con, "p1", {"category": "Non-Food"}, CATEGORY_TO_MACRO)
        original_catalogs, original_seed = sync.CATALOGS_BASE_DIR, sync.SEED_CATALOG_PATH
        try:
            sync.CATALOGS_BASE_DIR = self.root / "catalogs"
            country_dir = sync.CATALOGS_BASE_DIR / "ro"
            country_dir.mkdir(parents=True)
            sync.SEED_CATALOG_PATH = country_dir / "catalog_seed.csv"
            sync.write_csv_records(sync.SEED_CATALOG_PATH, [{
                "product_key": "p1", "macro_category": "Fresh Food",
                "category": "Dairy & Eggs", "kcal_100g": "100",
            }], sync.OVERRIDE_COLUMNS)
            sync.apply_seed_catalog_to_db(self.con, quiet=True)
        finally:
            sync.CATALOGS_BASE_DIR, sync.SEED_CATALOG_PATH = original_catalogs, original_seed
        self.assertEqual(self.con.execute(
            "SELECT category, kcal_100g, package_weight_g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Non-Food", None, 500.0))

    def test_nutrient_edit_on_existing_non_food_is_rejected(self):
        self.product("p1", macro="Non-Food", category="Non-Food", kcal=None,
                     protein=None, carbs=None, sugar=None, fat=None, saturated=None,
                     salt=None)
        with self.assertRaises(ValueError):
            save_manual_fields(self.con, "p1", {"kcal_100g": 100}, CATEGORY_TO_MACRO)
        self.assertIsNone(self.con.execute(
            "SELECT kcal_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0])

    def test_invalid_taxonomy_edit_rolls_back_without_mutation(self):
        self.product("p1", category="Dairy & Eggs")
        create_schema(self.con)
        with self.assertRaises(ValueError):
            save_manual_fields(self.con, "p1", {"category": "Made Up"}, CATEGORY_TO_MACRO)
        self.assertEqual(self.con.execute(
            "SELECT category, macro_category FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Dairy & Eggs", "Fresh Food"))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM product_field_metadata").fetchone()[0], 0)

    def test_sample_name_and_macro_category_are_not_editable(self):
        for field in ("sample_name", "macro_category"):
            with self.assertRaises(ValueError):
                save_manual_fields(self.con, "p1", {field: "x"}, CATEGORY_TO_MACRO)

    def test_review_snapshot_includes_override_products_and_ignores_stale_queue(self):
        self.product("p1", name="Incomplete", salt=None)
        self.product("p2", name="Complete")
        self.con.execute("INSERT INTO catalog_overrides (product_key) VALUES ('p1')")
        self.con.execute("INSERT INTO purchases (ticket_id, line_idx, name, quantity, ean, raw_code) VALUES ('t',1,'x',1,'p1','p1')")
        queue = self.root / "review.csv"
        queue.write_text("stale contents that must be replaced\n", encoding="utf-8")

        count = regenerate_review_csv(self.con, CATEGORY_TO_MACRO, queue, CSV_COLUMNS)

        self.assertEqual(count, 1)
        with queue.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["product_key"] for row in rows], ["p1"])
        self.assertEqual(rows[0]["salt_100g"], "")
        self.assertEqual(rows[0]["times_bought"], "1")

        first_bytes = queue.read_bytes()
        regenerate_review_csv(self.con, CATEGORY_TO_MACRO, queue, CSV_COLUMNS)
        self.assertEqual(queue.read_bytes(), first_bytes)

    def test_review_is_rebuilt_after_product_becomes_complete(self):
        self.product("p1", salt=None)
        first = list_incomplete_products(self.con, CATEGORY_TO_MACRO)
        self.assertEqual([row["product_key"] for row in first], ["p1"])
        self.con.execute("UPDATE nutrition_cache SET salt_100g=0 WHERE product_key='p1'")
        second = list_incomplete_products(self.con, CATEGORY_TO_MACRO)
        self.assertEqual(second, [])

    def test_review_order_is_deterministic_by_frequency_then_name_then_key(self):
        self.product("b", name="Alpha", salt=None)
        self.product("a", name="Alpha", salt=None)
        self.product("c", name="Zed", salt=None)
        self.con.execute("INSERT INTO purchases (ticket_id,line_idx,name,quantity,ean,raw_code) VALUES ('t1',1,'x',1,'b','b'), ('t2',1,'x',1,'b','b'), ('t3',1,'x',1,'c','c')")
        self.assertEqual([p["product_key"] for p in list_incomplete_products(self.con, CATEGORY_TO_MACRO)], ["b", "c", "a"])

    def test_review_csv_escapes_formula_cells(self):
        self.product("=bad", name="=SUM(A1:A2)", salt=None)
        path = self.root / "review.csv"
        regenerate_review_csv(self.con, CATEGORY_TO_MACRO, path, CSV_COLUMNS)
        with path.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["sample_name"], "'=SUM(A1:A2)")

    def test_generated_prefilled_values_are_not_imported_as_manual(self):
        self.product("p1", name="Product", kcal=100, salt=None)
        queue = self.root / "review_queue.csv"
        overrides = self.root / "catalog_overrides.csv"
        sync.write_csv_records(queue, [{
            "product_key": "p1", "sample_name": "Product",
            "macro_category": "Fresh Food", "category": "Dairy & Eggs",
            "kcal_100g": "100", "salt_100g": "",
        }], sync.OVERRIDE_COLUMNS)
        original_queue, original_overrides = sync.REVIEW_QUEUE_PATH, sync.OVERRIDE_CSV_PATH
        try:
            sync.REVIEW_QUEUE_PATH = queue
            sync.OVERRIDE_CSV_PATH = overrides
            sync.consume_completed_reviews(self.con)
            self.assertFalse(overrides.exists())

            # An actual cell edit is archived alone; unchanged values from the
            # generated snapshot do not become manual override evidence.
            sync.write_csv_records(queue, [{
                "product_key": "p1", "sample_name": "Product",
                "macro_category": "Fresh Food", "category": "Dairy & Eggs",
                "kcal_100g": "100", "salt_100g": "0.5",
            }], sync.OVERRIDE_COLUMNS)
            sync.consume_completed_reviews(self.con)
            rows, _, _ = sync.read_csv_records(overrides)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["salt_100g"], "0.5")
            self.assertEqual(rows[0]["kcal_100g"], "")
            self.assertEqual(rows[0]["category"], "")
        finally:
            sync.REVIEW_QUEUE_PATH = original_queue
            sync.OVERRIDE_CSV_PATH = original_overrides

    def test_legacy_override_csv_cannot_overwrite_protected_database_edit(self):
        self.product("p1", kcal=100, protein=None, carbs=None, sugar=None, fat=None,
                     saturated=None, salt=None)
        save_manual_fields(self.con, "p1", {"kcal_100g": 120}, CATEGORY_TO_MACRO)
        overrides = self.root / "catalog_overrides.csv"
        sync.write_csv_records(overrides, [{
            "product_key": "p1", "kcal_100g": "100",
        }], sync.OVERRIDE_COLUMNS)
        original_overrides = sync.OVERRIDE_CSV_PATH
        try:
            sync.OVERRIDE_CSV_PATH = overrides
            sync.apply_nutrition_overrides(self.con, quiet=True)
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides
        self.assertEqual(self.con.execute(
            "SELECT kcal_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0], 120.0)

    def test_non_food_legacy_override_preserves_package_weight(self):
        self.product("p1", weight=500)
        create_schema(self.con)
        overrides = self.root / "catalog_overrides.csv"
        sync.write_csv_records(overrides, [{
            "product_key": "p1", "macro_category": "Non-Food",
            "category": "Non-Food", "non_food": "TRUE",
        }], sync.OVERRIDE_COLUMNS)
        original_overrides = sync.OVERRIDE_CSV_PATH
        try:
            sync.OVERRIDE_CSV_PATH = overrides
            sync.apply_nutrition_overrides(self.con, quiet=True)
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides
        self.assertEqual(self.con.execute(
            "SELECT category, kcal_100g, package_weight_g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Non-Food", None, 500.0))

    def test_non_food_category_override_clears_nutrition_without_legacy_flag(self):
        self.product("p1", weight=500)
        self.con.execute("UPDATE nutrition_cache SET is_verified=TRUE WHERE product_key='p1'")
        create_schema(self.con)
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('p1', 'salt_100g', 'OFF', FALSE)"""
        )
        overrides = self.root / "catalog_overrides.csv"
        sync.write_csv_records(overrides, [{
            "product_key": "p1",
            "macro_category": "Non-Food",
            "category": "Non-Food",
            "non_food": "",
        }], sync.OVERRIDE_COLUMNS)
        original_overrides = sync.OVERRIDE_CSV_PATH
        try:
            sync.OVERRIDE_CSV_PATH = overrides
            # Applying the unchanged legacy override again must remain safe.
            sync.apply_nutrition_overrides(self.con, quiet=True)
            sync.apply_nutrition_overrides(self.con, quiet=True)
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides

        self.assertEqual(self.con.execute(
            """SELECT category, macro_category, kcal_100g, proteins_100g,
                      carbs_100g, sugars_100g, fat_100g, saturated_fat_100g,
                      fiber_100g, salt_100g, package_weight_g, match_status,
                      data_source, is_verified
               FROM nutrition_cache WHERE product_key='p1'"""
        ).fetchone(), (
            "Non-Food", "Non-Food", None, None, None, None, None, None,
            None, None, 500.0, "NON_FOOD", "RULE_NON_FOOD", True,
        ))
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM product_field_metadata WHERE product_key='p1' "
            "AND field_name IN ('kcal_100g', 'proteins_100g', 'carbs_100g', "
            "'sugars_100g', 'fat_100g', 'saturated_fat_100g', 'fiber_100g', 'salt_100g')"
        ).fetchone()[0], 0)

    def test_legacy_non_food_flag_cannot_override_protected_food_category(self):
        self.product("p1")
        save_manual_fields(self.con, "p1", {"category": "Dairy & Eggs"}, CATEGORY_TO_MACRO)
        self.con.execute(
            "UPDATE nutrition_cache SET is_verified=TRUE, match_status='MANUAL_FOOD' WHERE product_key='p1'"
        )
        overrides = self.root / "catalog_overrides.csv"
        sync.write_csv_records(overrides, [{
            "product_key": "p1",
            "macro_category": "Non-Food",
            "category": "Non-Food",
            "non_food": "TRUE",
        }], sync.OVERRIDE_COLUMNS)
        original_overrides = sync.OVERRIDE_CSV_PATH
        try:
            sync.OVERRIDE_CSV_PATH = overrides
            sync.apply_nutrition_overrides(self.con, quiet=True)
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides

        self.assertEqual(self.con.execute(
            "SELECT category, macro_category, kcal_100g, salt_100g, match_status, is_verified "
            "FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Dairy & Eggs", "Fresh Food", 10.0, 0.1, "MANUAL_FOOD", True))

    # --- Atwater: implausible nutrition is emptied and the product is incomplete ---

    def bad_cheese(self, key="bad", *, salt=0.14, status="SEED_CATALOG", verified=True):
        # Protein alone is 49.2 kcal, so 34.5 kcal declared is implausible.
        self.product(key, name="Synthetic protein quark", kcal=34.5, protein=12.3, carbs=3.7,
                     sugar=3.2, fat=0.5, saturated=0.4, salt=salt, status=status, source="SEED_CATALOG")
        self.con.execute("UPDATE nutrition_cache SET is_verified=? WHERE product_key=?", [verified, key])

    def nutrients(self, key):
        return self.con.execute(
            f"SELECT {', '.join(NUTRIENTS)} FROM nutrition_cache WHERE product_key=?", [key]
        ).fetchone()

    def test_atwater_check_flags_implausible_values_and_respects_exemptions(self):
        cheese = {"category": "Dairy & Eggs", "kcal_100g": 34.5, "proteins_100g": 12.3,
                  "carbs_100g": 3.7, "fat_100g": 0.5}
        self.assertEqual(atwater_check(cheese), "Calories 34.5 declared vs 68.5 calculated")
        self.assertIsNone(atwater_check({**cheese, "kcal_100g": 69}))
        self.assertIsNone(atwater_check({**cheese, "fat_100g": None}))
        self.assertIsNone(atwater_check({**cheese, "category": "Non-Food"}))
        beer = {**cheese, "category": "Alcoholic Beverages", "kcal_100g": 43, "proteins_100g": 0.5,
                "carbs_100g": 3.6, "fat_100g": 0}
        self.assertIsNone(atwater_check(beer))
        self.assertEqual(atwater_check({**beer, "sugars_100g": 9}), "Sugar is higher than carbs")

    def test_herbs_and_spices_skip_energy_and_mass_checks_only(self):
        # Fails the energy balance in both fibre readings (335 or 315 vs 100 kcal).
        spice = {"category": "Herbs & Spices", "kcal_100g": 100, "proteins_100g": 10,
                 "carbs_100g": 60, "fat_100g": 5, "fiber_100g": 5, "salt_100g": 0.05}
        self.assertIsNotNone(atwater_check({**spice, "category": "Sauces, Oils & Condiments"}))
        self.assertIsNone(atwater_check(spice))
        heavy = {**spice, "proteins_100g": 20, "carbs_100g": 75, "fat_100g": 15, "kcal_100g": 515}
        self.assertEqual(atwater_check({**heavy, "category": "Sauces, Oils & Condiments"}),
                         "Nutrients add up to more than 100 g")
        self.assertIsNone(atwater_check(heavy))
        self.assertEqual(atwater_check({**spice, "sugars_100g": 70}), "Sugar is higher than carbs")

    def test_fibre_already_included_in_carbs_is_tolerated(self):
        # 300 kcal: EU reading gives 405, carbs-with-fibre reading gives 245 (within 20 %).
        product = {"category": "Sauces, Oils & Condiments", "kcal_100g": 300, "proteins_100g": 10,
                   "carbs_100g": 60, "fat_100g": 5, "fiber_100g": 40}
        self.assertIsNone(atwater_check(product))
        self.assertIsNone(atwater_check({**product, "carbs_100g": 70, "fat_100g": 20, "kcal_100g": 500}))
        self.assertIsNotNone(atwater_check({**product, "kcal_100g": 150}))

    def test_clearing_empties_all_unprotected_nutrients_and_lists_the_product(self):
        self.bad_cheese("bad")
        self.product("ok")
        create_schema(self.con)
        save_manual_fields(self.con, "bad", {"salt_100g": 0.2}, CATEGORY_TO_MACRO)

        cleared = clear_implausible_nutrition(self.con)

        self.assertEqual(cleared, [("bad", "Calories 34.5 declared vs 68.5 calculated")])
        self.assertEqual(self.nutrients("bad"), (None,) * 7 + (0.2,))
        self.assertEqual(self.con.execute(
            "SELECT match_status, is_verified, atwater_debug FROM nutrition_cache WHERE product_key='bad'"
        ).fetchone(), ("QUARANTINE_ATWATER_FAIL", False, "CLEARED: Calories 34.5 declared vs 68.5 calculated"))
        self.assertEqual(self.con.execute(
            "SELECT field_name FROM product_field_metadata WHERE product_key='bad' AND field_name LIKE '%_100g'"
        ).fetchall(), [("salt_100g",)])
        products = {row["product_key"]: row for row in list_review_products(self.con, CATEGORY_TO_MACRO)}
        self.assertEqual(set(products), {"bad"})
        self.assertEqual(products["bad"]["cleared_reason"], "Calories 34.5 declared vs 68.5 calculated")
        self.assertEqual(review_status_text(products["bad"]),
                         "Incomplete · values cleared: Calories 34.5 declared vs 68.5 calculated")

    def test_values_saved_by_hand_are_confirmed_and_never_cleared(self):
        self.bad_cheese("check")
        create_schema(self.con)
        save_manual_fields(self.con, "check", {"kcal_100g": 34.5, "proteins_100g": 12.3,
                                               "carbs_100g": 3.7, "fat_100g": 0.5}, CATEGORY_TO_MACRO)

        self.assertEqual(clear_implausible_nutrition(self.con), [])
        sync.validate_atwater(self.con)

        self.assertEqual(self.nutrients("check")[:3], (34.5, 12.3, 3.7))
        self.assertEqual(self.con.execute(
            "SELECT is_verified, atwater_debug FROM nutrition_cache WHERE product_key='check'"
        ).fetchone(), (True, "OK (confirmed manually: Calories 34.5 declared vs 68.5 calculated)"))
        self.assertEqual(list_review_products(self.con, CATEGORY_TO_MACRO), [])

    def test_clearing_resets_or_rejects_a_previous_ai_answer(self):
        self.bad_cheese("before_ai")
        self.bad_cheese("after_ai")
        create_schema(self.con)
        self.con.execute(f"INSERT INTO {AI_STATE_TABLE} (product_key, task, status) VALUES "
                         "('before_ai', 'nutrition', 'succeeded'), ('after_ai', 'nutrition', 'succeeded')")

        sync.triage_implausible_nutrition(self.con)
        self.assertEqual(self.con.execute(
            f"SELECT product_key FROM {AI_STATE_TABLE} ORDER BY 1").fetchall(), [])

        self.bad_cheese("again")
        self.con.execute(f"INSERT INTO {AI_STATE_TABLE} (product_key, task, status) VALUES ('again', 'nutrition', 'succeeded')")
        sync.validate_atwater(self.con)
        self.assertEqual(self.con.execute(
            f"SELECT status, error_code FROM {AI_STATE_TABLE} WHERE product_key='again'"
        ).fetchone(), ("invalid", "atwater_failed"))
        self.assertEqual(self.nutrients("again"), (None,) * 8)

    def test_review_csv_has_status_with_the_clearing_reason(self):
        self.product("incomplete", salt=None)
        self.bad_cheese("check")
        clear_implausible_nutrition(self.con)
        queue = self.root / "review_queue.csv"

        count = regenerate_review_csv(self.con, CATEGORY_TO_MACRO, queue, CSV_COLUMNS)

        self.assertEqual(count, 2)
        with queue.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames[:2], ["product_key", "review_status"])
            status = {row["product_key"]: row["review_status"] for row in reader}
        self.assertEqual(status, {
            "incomplete": "Incomplete",
            "check": "Incomplete · values cleared: Calories 34.5 declared vs 68.5 calculated",
        })

    def test_validation_verifies_good_values_and_notes_exemptions(self):
        self.product("good", kcal=69, protein=12.3, carbs=3.7, fat=0.5, status="MATCHED_OFF")
        self.product("pepper", macro="Pantry & Groceries", category="Herbs & Spices", kcal=251,
                     protein=10.4, carbs=64.8, fat=3.3, fiber=25.3, status="MATCHED_OFF")
        create_schema(self.con)

        sync.validate_atwater(self.con)

        self.assertEqual(self.con.execute(
            "SELECT product_key, is_verified, match_status, atwater_debug FROM nutrition_cache ORDER BY 1"
        ).fetchall(), [
            ("good", True, "MATCHED_OFF_VERIFIED", "OK (Atwater=68.5)"),
            ("pepper", True, "MATCHED_OFF_VERIFIED", "OK (Herbs & Spices - exempt from Atwater)"),
        ])

    # --- Herbs & spices rules ---

    def test_spice_rules_match_spices_and_keep_sauces(self):
        rules = sync.load_spice_rules(Path(__file__).resolve().parent.parent / "data" / "catalogs" / "ro" / "spice_rules.json")
        for name in ("Piper negru macinat", "Boia dulce afumată", "Rasnita condim sort.", "Foi de dafin",
                     "Mix de condimente", "Scortisoara macinata", "Usturoi granulat"):
            self.assertTrue(sync.is_herb_or_spice(rules, name), name)
        for name in ("Sos de roșii cu busuioc", "Pasta de ardei", "Cub concentrat pentru supa",
                     "Ulei masline spray", "Ketchup iute", "Dr. Oetker Esenta vanilie", "Ardeipiper"):
            self.assertFalse(sync.is_herb_or_spice(rules, name), name)
        self.assertTrue(sync.spice_rule_applies(rules, "Sauces, Oils & Condiments"))
        self.assertTrue(sync.spice_rule_applies(rules, None))
        self.assertFalse(sync.spice_rule_applies(rules, "Biscuits & Savory Snacks"))

    def test_classify_moves_spices_but_not_manual_or_other_categories(self):
        rules_path = self.root / "spice_rules.json"
        rules_path.write_text('{"from_categories": ["Sauces, Oils & Condiments"], "include": ["piper", "boia"], '
                              '"exclude": ["sos"]}', encoding="utf-8")
        taxonomy_path = self.root / "taxonomy.json"
        taxonomy_path.write_text('{"Pantry & Groceries": ["Sauces, Oils & Condiments", "Herbs & Spices"], '
                                 '"Sweets & Snacks": ["Biscuits & Savory Snacks"]}', encoding="utf-8")
        self.product("pepper", name="Piper negru", macro="Pantry & Groceries", category="Sauces, Oils & Condiments")
        self.product("sauce", name="Sos cu piper", macro="Pantry & Groceries", category="Sauces, Oils & Condiments")
        self.product("chips", name="Chipsuri boia", macro="Sweets & Snacks", category="Biscuits & Savory Snacks")
        self.product("manual", name="Boia dulce", macro="Pantry & Groceries", category="Sauces, Oils & Condiments")
        self.product("blank", name="Boia iute", macro=None, category=None)
        create_schema(self.con)
        save_manual_fields(self.con, "manual", {"category": "Sauces, Oils & Condiments"},
                           {"Sauces, Oils & Condiments": "Pantry & Groceries"})

        with unittest.mock.patch.object(sync, "SPICE_RULES_PATH", rules_path), \
                unittest.mock.patch.object(sync, "TAXONOMY_PATH", taxonomy_path):
            moved = sync.classify_herbs_and_spices(self.con)

        self.assertEqual(moved, 2)
        self.assertEqual(dict(self.con.execute(
            "SELECT product_key, category FROM nutrition_cache").fetchall()), {
            "pepper": "Herbs & Spices", "sauce": "Sauces, Oils & Condiments",
            "chips": "Biscuits & Savory Snacks", "manual": "Sauces, Oils & Condiments",
            "blank": "Herbs & Spices",
        })
        self.assertEqual(self.con.execute(
            "SELECT macro_category FROM nutrition_cache WHERE product_key='blank'").fetchone()[0], "Pantry & Groceries")

    def test_seed_corrections_pass_the_atwater_check(self):
        seed = Path(__file__).resolve().parent.parent / "data" / "catalogs" / "ro" / "catalog_seed.csv"
        with seed.open(encoding="utf-8-sig", newline="") as handle:
            rows = {row["product_key"]: row for row in csv.DictReader(handle)}
        for key in ("4056489129660", "5942321000213", "4056489539797", "5941872102438",
                    "8000500023976", "5949026104857"):
            self.assertIsNone(atwater_check(rows[key]), rows[key]["sample_name"])
        spices = [row for row in rows.values() if row["category"] == "Herbs & Spices"]
        self.assertGreaterEqual(len(spices), 20)
        self.assertTrue(all(row["macro_category"] == "Pantry & Groceries" for row in spices))


NUTRIENTS = ("kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g", "fat_100g",
             "saturated_fat_100g", "fiber_100g", "salt_100g")


if __name__ == "__main__":
    unittest.main()
