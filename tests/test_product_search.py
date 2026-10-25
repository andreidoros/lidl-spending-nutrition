"""All-product lookup and manual-save integration, using temporary DuckDB only."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb

import pipeline_services as services
import sync_incremental as sync
from product_store import create_schema


class ProductSearchTests(unittest.TestCase):
    def test_baseline_equality_preserves_null_zero_and_real_value_changes(self):
        same = services._same_baseline_value
        self.assertTrue(same(float("nan"), float("nan")))
        self.assertTrue(same(None, None))
        self.assertTrue(same(0.0, 0.0))
        for left, right in ((float("nan"), None), (None, float("nan")),
                            (float("nan"), 0.0), (float("nan"), 250.0),
                            (None, 0.0), (250.0, 300.0)):
            with self.subTest(left=left, right=right):
                self.assertFalse(same(left, right))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "synthetic.duckdb"
        lock = patch.object(services, "_workspace_lock_path", return_value=self.root / "test.lock")
        lock.start()
        self.addCleanup(lock.stop)
        with duckdb.connect(str(self.db)) as connection:
            sync.init_database(connection)
            create_schema(connection)
            for key, name in (("0001", "LĂPTE sintetic"), ("0002", "Lapte sintetic"),
                              ("0003", "Synthetic 50%_drink"), ("0004", "Lapte sintetic")):
                connection.execute(
                    "INSERT INTO nutrition_cache (product_key, sample_name, category, macro_category) VALUES (?, ?, 'Non-Food', 'Non-Food')",
                    [key, name],
                )
            connection.execute(
                "INSERT INTO product_field_metadata (product_key, field_name, source, is_protected) VALUES ('0001', 'category', 'MANUAL', TRUE)"
            )

    def test_partial_accent_case_matching_includes_complete_products_and_duplicates(self):
        self.assertEqual(services.read_review_products(self.db), [])
        matches = services.search_products("  LaPt  ", self.db)
        self.assertEqual([item["product_key"] for item in matches], ["0001", "0002", "0004"])
        self.assertEqual(services.search_products("lăpte", self.db), matches)

    def test_literal_wildcards_limit_empty_and_missing_database(self):
        self.assertEqual([row["product_key"] for row in services.search_products("%_", self.db)], ["0003"])
        self.assertEqual(len(services.search_products("lap", self.db, limit=2)), 2)
        self.assertEqual(services.search_products("", self.db), [])
        self.assertEqual(services.search_products("not-present", self.db), [])
        for limit in (0, 51, True, 1.5):
            with self.assertRaises(ValueError):
                services.search_products("lap", self.db, limit=limit)
        with self.assertRaises(FileNotFoundError):
            services.search_products("lap", self.root / "absent.duckdb")

    def test_read_returns_opaque_key_nulls_and_provenance_and_parameterizes_lookup(self):
        product = services.read_product("0001", self.db)
        self.assertEqual(product["product_key"], "0001")
        self.assertIsNone(product["salt_100g"])
        self.assertEqual(product["field_metadata"]["category"][:2], ("MANUAL", True))
        self.assertIsNone(services.read_product("' OR 1=1 --", self.db))
        with self.assertRaises(FileNotFoundError):
            services.read_product("0001", self.root / "absent.duckdb")

    def test_legacy_product_read_does_not_create_missing_metadata(self):
        with duckdb.connect(str(self.db)) as connection:
            connection.execute("DROP TABLE product_field_metadata")
        self.assertEqual(services.read_product("0001", self.db)["field_metadata"], {})
        with duckdb.connect(str(self.db), read_only=True) as connection:
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'product_field_metadata'"
            ).fetchone()[0], 0)

    def test_save_complete_product_uses_existing_protection_exports_and_stale_checks(self):
        product = services.read_product("0001", self.db)
        baseline = {**product, "__field_metadata__": product["field_metadata"]}
        with patch.object(sync, "REVIEW_QUEUE_PATH", self.root / "review.csv"), patch.object(
            sync, "OVERRIDE_CSV_PATH", self.root / "overrides.csv"
        ), patch.object(sync, "export_views_for_bi") as export, patch.object(
            services, "run_sync"
        ) as run_sync:
            report = services.save_review_session(
                manual_edits={"0001": {"category": "Dairy & Eggs", "salt_100g": 0.0}},
                baselines={"0001": baseline}, db_path=self.db, output_path=self.root / "chat.txt",
            )
            self.assertIsNone(report.database_error)
            self.assertFalse(report.conflicts)
            self.assertFalse(report.artifact_failures)
            updated = services.read_product("0001", self.db)
            self.assertEqual(updated["macro_category"], "Fresh Food")
            self.assertEqual(updated["salt_100g"], 0.0)
            self.assertEqual(updated["field_metadata"]["salt_100g"][:2], ("MANUAL", True))
            stale = services.save_review_session(
                manual_edits={"0001": {"salt_100g": 1.0}}, baselines={"0001": baseline},
                db_path=self.db, output_path=self.root / "chat.txt",
            )
            self.assertIn(("0001", "salt_100g"), stale.conflicts)
            self.assertEqual(services.read_product("0001", self.db)["salt_100g"], 0.0)
            export.assert_called_once()
            run_sync.assert_not_called()

    def test_first_save_of_three_nan_weights_succeeds_without_false_conflicts(self):
        with duckdb.connect(str(self.db)) as connection:
            connection.execute("UPDATE nutrition_cache SET package_weight_g = 'NaN'::DOUBLE")
        baselines = {}
        for key in ("0001", "0002", "0003"):
            product = services.read_product(key, self.db)
            baselines[key] = {**product, "__field_metadata__": product["field_metadata"]}
        edits = {key: {"package_weight_g": weight} for key, weight in zip(baselines, (250.0, 300.0, 2000.0))}
        with patch.object(sync, "REVIEW_QUEUE_PATH", self.root / "review.csv"), patch.object(
            sync, "OVERRIDE_CSV_PATH", self.root / "overrides.csv"
        ), patch.object(sync, "export_views_for_bi") as export:
            report = services.save_review_session(
                manual_edits=edits, baselines=baselines, db_path=self.db, output_path=self.root / "chat.txt",
            )
        self.assertEqual(report.conflicts, ())
        self.assertIsNone(report.database_error)
        self.assertEqual(report.manual_fields_saved, 3)
        export.assert_called_once()
        for key, fields in edits.items():
            product = services.read_product(key, self.db)
            self.assertEqual(product["package_weight_g"], fields["package_weight_g"])
            self.assertEqual(product["field_metadata"]["package_weight_g"][:2], ("MANUAL", True))

    def test_nan_baseline_does_not_allow_overwriting_a_new_real_weight(self):
        with duckdb.connect(str(self.db)) as connection:
            connection.execute("UPDATE nutrition_cache SET package_weight_g = 'NaN'::DOUBLE WHERE product_key = '0001'")
        product = services.read_product("0001", self.db)
        with duckdb.connect(str(self.db)) as connection:
            connection.execute("UPDATE nutrition_cache SET package_weight_g = 500 WHERE product_key = '0001'")
        report = services.save_review_session(
            manual_edits={"0001": {"package_weight_g": 250.0}}, baselines={"0001": product},
            db_path=self.db, output_path=self.root / "chat.txt",
        )
        self.assertEqual(report.conflicts, (("0001", "package_weight_g"),))
        self.assertEqual(services.read_product("0001", self.db)["package_weight_g"], 500.0)

    def test_non_food_transition_accepts_unchanged_nan_but_checks_new_provenance(self):
        with duckdb.connect(str(self.db)) as connection:
            connection.execute("UPDATE nutrition_cache SET category = 'Dairy & Eggs', macro_category = 'Fresh Food', salt_100g = 'NaN'::DOUBLE")
        baselines = {}
        for key in ("0001", "0002"):
            product = services.read_product(key, self.db)
            baselines[key] = {**product, "__field_metadata__": product["field_metadata"]}
        with duckdb.connect(str(self.db)) as connection:
            connection.execute("INSERT INTO product_field_metadata (product_key, field_name, source, is_protected) VALUES ('0002', 'salt_100g', 'MANUAL', TRUE)")
        with patch.object(sync, "REVIEW_QUEUE_PATH", self.root / "review.csv"), patch.object(
            sync, "OVERRIDE_CSV_PATH", self.root / "overrides.csv"
        ), patch.object(sync, "export_views_for_bi"):
            report = services.save_review_session(
                manual_edits={key: {"category": "Non-Food"} for key in baselines},
                baselines=baselines, db_path=self.db, output_path=self.root / "chat.txt",
            )
        self.assertNotIn(("0001", "salt_100g"), report.conflicts)
        self.assertIn(("0002", "salt_100g"), report.conflicts)
        self.assertEqual(services.read_product("0001", self.db)["category"], "Non-Food")
        self.assertIsNone(services.read_product("0001", self.db)["salt_100g"])
        self.assertEqual(services.read_product("0002", self.db)["category"], "Dairy & Eggs")


if __name__ == "__main__":
    unittest.main()
