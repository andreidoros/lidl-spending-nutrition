"""Offline tests for the chatbot batch export/import boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_enrichment.chat_batch import (
    build_chat_batch_text,
    preview_chat_batch_import,
    refresh_chat_batch_file,
    save_chat_batch_import,
)
from ai_eval.taxonomy import load_taxonomy
from product_store import create_schema


NUTRIENTS = (
    "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
    "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
)
VALID_NUTRIENTS = {
    "kcal_100g": 100.0,
    "proteins_100g": 3.0,
    "carbs_100g": 5.0,
    "sugars_100g": 4.0,
    "fat_100g": 7.0,
    "saturated_fat_100g": 2.0,
    "fiber_100g": 1.0,
    "salt_100g": 0.1,
}
NUTRITION_DDL = """
CREATE TABLE nutrition_cache (
    product_key VARCHAR PRIMARY KEY,
    sample_name VARCHAR,
    macro_category VARCHAR,
    category VARCHAR,
    package_weight_g DOUBLE,
    kcal_100g DOUBLE,
    proteins_100g DOUBLE,
    carbs_100g DOUBLE,
    sugars_100g DOUBLE,
    fat_100g DOUBLE,
    saturated_fat_100g DOUBLE,
    fiber_100g DOUBLE,
    salt_100g DOUBLE,
    match_status VARCHAR,
    data_source VARCHAR,
    is_verified BOOLEAN DEFAULT FALSE
)
"""


class ChatBatchTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect(":memory:")
        self.con.execute(NUTRITION_DDL)
        create_schema(self.con)
        self.taxonomy = load_taxonomy()
        self.category_to_macro = self.taxonomy.category_to_macro

    def tearDown(self):
        self.con.close()

    def add_product(
        self,
        key="000123",
        *,
        name="Synthetic product",
        category="Dairy & Eggs",
        macro="Fresh Food",
        package_weight=50.0,
        nutrients=None,
    ):
        values = dict(VALID_NUTRIENTS)
        values.update(nutrients or {})
        columns = (
            "product_key", "sample_name", "macro_category", "category",
            "package_weight_g", *NUTRIENTS, "match_status", "data_source",
        )
        row = (
            key, name, macro, category, package_weight,
            *(values[field] for field in NUTRIENTS), "TEST", "SYNTHETIC",
        )
        self.con.execute(
            f"INSERT INTO nutrition_cache ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            row,
        )

    @staticmethod
    def response(*products):
        return json.dumps({"products": list(products)})

    def state(self, key="000123"):
        return self.con.execute(
            "SELECT task, status, attempt_count, provider, model "
            "FROM ai_enrichment_state WHERE product_key=? ORDER BY task",
            [key],
        ).fetchall()

    def test_complete_catalog_builds_no_batch_and_removes_stale_file(self):
        self.add_product()
        self.assertIsNone(build_chat_batch_text(
            self.con, self.taxonomy, self.category_to_macro
        ))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ai_chat_batch.txt"
            path.write_text("old batch", encoding="utf-8")
            count = refresh_chat_batch_file(
                self.con, self.taxonomy, self.category_to_macro, path
            )
            self.assertEqual(count, 0)
            self.assertFalse(path.exists())

    def test_batch_is_deterministic_text_and_lists_only_missing_fields(self):
        self.add_product(
            name="Milk carton",
            package_weight=0,
            nutrients={"salt_100g": None},
        )
        text = build_chat_batch_text(self.con, self.taxonomy, self.category_to_macro)
        self.assertIsNotNone(text)
        self.assertEqual(text, build_chat_batch_text(
            self.con, self.taxonomy, self.category_to_macro
        ))
        self.assertIn("Canonical taxonomy:", text)
        self.assertIn('"product_key": "000123"', text)
        self.assertIn('"sample_name": "Milk carton"', text)
        self.assertIn('"fields_to_fill": [\n      "salt_100g"', text)
        self.assertNotIn("macro_category", text.split("Products and requested fields:", 1)[1])
        self.assertNotIn("ticket_id", text)
        self.assertNotIn("times_bought", text)

    def test_code_fence_import_preserves_zero_and_records_ai_provenance(self):
        self.add_product(
            package_weight=None,
            nutrients={"salt_100g": None},
        )
        response = "```json\n" + self.response({
            "product_key": "000123",
            "salt_100g": 0,
            "package_weight_g": 0,
        }) + "\n```"

        preview = preview_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        self.assertEqual(preview.fields_suggested, 2)
        self.assertEqual(preview.issues, ())

        report = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        self.assertEqual((report.products_updated, report.fields_saved), (1, 2))
        self.assertEqual(self.con.execute(
            "SELECT salt_100g, package_weight_g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone(), (0.0, 0.0))
        self.assertEqual(self.con.execute(
            "SELECT field_name, source, is_protected, provider, model "
            "FROM product_field_metadata WHERE product_key='000123' ORDER BY field_name"
        ).fetchall(), [
            ("package_weight_g", "AI", False, "chat_import", None),
            ("salt_100g", "AI", False, "chat_import", None),
        ])
        self.assertEqual(self.state(), [("nutrition", "succeeded", 0, "chat_import", None)])

    def test_blocked_explicit_manual_field_is_not_imported_as_chat_ai(self):
        self.add_product(
            package_weight=None,
            nutrients={"salt_100g": None},
        )
        response = self.response({
            "product_key": "000123",
            "salt_100g": 0,
            "package_weight_g": 0,
        })
        preview = preview_chat_batch_import(
            self.con,
            response,
            self.taxonomy,
            self.category_to_macro,
            blocked_fields={"000123": ("salt_100g",)},
        )
        self.assertEqual(preview.fields_suggested, 1)
        self.assertEqual(
            [(item.product_key, item.field_name) for item in preview.skipped],
            [("000123", "salt_100g")],
        )

    def test_category_import_derives_macro_and_marks_only_accepted_tasks(self):
        self.add_product(category=None, macro=None)
        response = self.response({"product_key": "000123", "category": "Dairy & Eggs"})

        report = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )

        self.assertEqual(report.fields_saved, 2)
        self.assertEqual(self.con.execute(
            "SELECT category, macro_category FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone(), ("Dairy & Eggs", "Fresh Food"))
        self.assertEqual(self.state(), [("category", "succeeded", 0, "chat_import", None)])

    def test_category_and_partial_nutrition_import_are_saved_as_separate_tasks(self):
        self.add_product(
            category=None,
            macro=None,
            nutrients={"salt_100g": None},
        )
        response = self.response({
            "product_key": "000123",
            "category": "Dairy & Eggs",
            "salt_100g": 0,
        })

        report = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )

        self.assertEqual(report.fields_saved, 3)
        self.assertEqual(self.state(), [
            ("category", "succeeded", 0, "chat_import", None),
            ("nutrition", "succeeded", 0, "chat_import", None),
        ])
        self.assertEqual(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()[0], 0.0)

    def test_mixed_valid_invalid_duplicate_unknown_entries_apply_partially(self):
        self.add_product(key="dup", nutrients={"salt_100g": None})
        self.add_product(key="valid", nutrients={"salt_100g": None})
        response = self.response(
            {"product_key": "dup", "salt_100g": 0},
            {"product_key": "dup", "salt_100g": 1},
            {"product_key": "valid", "salt_100g": 0},
            {"product_key": "missing", "salt_100g": 1},
            "not an object",
        )

        preview = preview_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        self.assertEqual([item.product_key for item in preview.suggestions], ["valid"])
        self.assertEqual(
            {issue.code for issue in preview.issues},
            {"duplicate_product", "unknown_product", "invalid_product_entry"},
        )
        report = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        self.assertEqual((report.products_updated, report.fields_saved), (1, 1))
        self.assertIsNone(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='dup'"
        ).fetchone()[0])
        self.assertEqual(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='valid'"
        ).fetchone()[0], 0.0)

    def test_malformed_json_and_duplicate_json_members_are_reported(self):
        for response in (
            "{not json",
            '{"products":[],"products":[]}',
            "```json\n{\"products\":[\n```",
        ):
            with self.subTest(response=response):
                preview = preview_chat_batch_import(
                    self.con, response, self.taxonomy, self.category_to_macro
                )
                self.assertEqual(preview.suggestions, ())
                self.assertEqual(preview.issues[0].code, "malformed_json")

    def test_invalid_range_and_atwater_values_are_not_saved(self):
        self.add_product(nutrients={"salt_100g": None})
        range_preview = preview_chat_batch_import(
            self.con,
            self.response({"product_key": "000123", "salt_100g": 101}),
            self.taxonomy,
            self.category_to_macro,
        )
        self.assertEqual(range_preview.suggestions, ())
        self.assertIn("nutrition_out_of_range", {i.code for i in range_preview.issues})

        self.con.execute(
            "UPDATE nutrition_cache SET kcal_100g=NULL, proteins_100g=NULL, "
            "carbs_100g=NULL, fat_100g=NULL WHERE product_key='000123'"
        )
        atwater_preview = preview_chat_batch_import(
            self.con,
            self.response({
                "product_key": "000123",
                "kcal_100g": 500,
                "proteins_100g": 1,
                "carbs_100g": 1,
                "fat_100g": 1,
                "salt_100g": 0,
            }),
            self.taxonomy,
            self.category_to_macro,
        )
        self.assertEqual(atwater_preview.suggestions, ())
        self.assertIn("atwater_mismatch", {i.code for i in atwater_preview.issues})

    def test_null_omission_and_stale_or_protected_fields_do_not_clear_or_overwrite(self):
        self.add_product(nutrients={"kcal_100g": None, "salt_100g": None})
        response = self.response({
            "product_key": "000123",
            "kcal_100g": 120,
            "salt_100g": 0.5,
        })
        self.con.execute(
            "UPDATE nutrition_cache SET kcal_100g=0 WHERE product_key='000123'"
        )
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('000123', 'salt_100g', 'MANUAL', TRUE)"""
        )

        preview = preview_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        self.assertEqual(preview.suggestions, ())
        self.assertEqual({item.code for item in preview.skipped}, {"field_no_longer_eligible"})
        report = save_chat_batch_import(
            self.con,
            self.response({"product_key": "000123", "salt_100g": None}),
            self.taxonomy,
            self.category_to_macro,
        )
        self.assertEqual(report.fields_saved, 0)
        self.assertEqual(self.con.execute(
            "SELECT kcal_100g, salt_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone(), (0.0, None))
        self.assertEqual(self.con.execute(
            "SELECT source, is_protected FROM product_field_metadata "
            "WHERE product_key='000123' AND field_name='salt_100g'"
        ).fetchone(), ("MANUAL", True))
        self.assertEqual(self.state(), [])

    def test_valid_field_is_saved_when_another_field_is_stale(self):
        self.add_product(nutrients={"kcal_100g": None, "salt_100g": None})
        response = self.response({
            "product_key": "000123",
            "kcal_100g": 100,
            "salt_100g": 0,
        })
        self.con.execute(
            "UPDATE nutrition_cache SET kcal_100g=100 WHERE product_key='000123'"
        )

        report = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )

        self.assertEqual(report.fields_saved, 1)
        self.assertEqual(self.con.execute(
            "SELECT kcal_100g, salt_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone(), (100.0, 0.0))
        self.assertEqual(self.state(), [("nutrition", "succeeded", 0, "chat_import", None)])

    def test_nonblank_invalid_category_is_not_offered_for_ai_replacement(self):
        self.add_product(
            category="Unlisted legacy category",
            macro="Legacy macro",
            package_weight=None,
            nutrients={"kcal_100g": None},
        )
        batch = build_chat_batch_text(self.con, self.taxonomy, self.category_to_macro)
        self.assertIn('"fields_to_fill": [\n      "package_weight_g"', batch)

        response = self.response({
            "product_key": "000123",
            "category": "Dairy & Eggs",
            "kcal_100g": 100,
            "package_weight_g": 0,
        })
        preview = preview_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        self.assertEqual(preview.suggestions[0].category, None)
        self.assertIsNone(preview.suggestions[0].nutrients)
        self.assertEqual(preview.suggestions[0].package_weight_g, 0.0)
        self.assertIn("category_not_replaceable", {item.code for item in preview.issues})

        save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        self.assertEqual(self.con.execute(
            "SELECT category, kcal_100g, package_weight_g FROM nutrition_cache "
            "WHERE product_key='000123'"
        ).fetchone(), ("Unlisted legacy category", None, 0.0))
        self.assertEqual(self.state(), [])

    def test_repeated_import_is_idempotent(self):
        self.add_product(nutrients={"salt_100g": None})
        response = self.response({"product_key": "000123", "salt_100g": 0})

        first = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )
        second = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )

        self.assertEqual(first.fields_saved, 1)
        self.assertEqual(second.fields_saved, 0)
        self.assertEqual(self.state(), [("nutrition", "succeeded", 0, "chat_import", None)])

    def test_non_food_clears_unprotected_nutrition_and_preserves_package_weight(self):
        self.add_product(category=None, macro=None, package_weight=50)
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('000123', 'salt_100g', 'OFF', FALSE)"""
        )
        response = self.response({"product_key": "000123", "category": "Non-Food"})

        report = save_chat_batch_import(
            self.con, response, self.taxonomy, self.category_to_macro
        )

        row = self.con.execute(
            "SELECT category, macro_category, package_weight_g, "
            "kcal_100g, proteins_100g, salt_100g "
            "FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()
        self.assertEqual(row[:3], ("Non-Food", "Non-Food", 50.0))
        self.assertEqual(row[3:], (None, None, None))
        self.assertIsNone(self.con.execute(
            "SELECT 1 FROM product_field_metadata "
            "WHERE product_key='000123' AND field_name='salt_100g'"
        ).fetchone())
        self.assertEqual(self.state(), [("category", "succeeded", 0, "chat_import", None)])
        self.assertEqual(report.fields_saved, 2)

    def test_non_food_rejects_protected_nonnull_nutrition_and_keeps_protected_null_metadata(self):
        self.add_product(
            category=None,
            macro=None,
            nutrients={"proteins_100g": None},
        )
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('000123', 'proteins_100g', 'MANUAL', TRUE)"""
        )
        preview = preview_chat_batch_import(
            self.con,
            self.response({"product_key": "000123", "category": "Non-Food"}),
            self.taxonomy,
            self.category_to_macro,
        )
        self.assertEqual(preview.suggestions[0].category, "Non-Food")

        save_chat_batch_import(
            self.con,
            self.response({"product_key": "000123", "category": "Non-Food"}),
            self.taxonomy,
            self.category_to_macro,
        )
        self.assertEqual(self.con.execute(
            "SELECT category, proteins_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone(), ("Non-Food", None))
        self.assertEqual(self.con.execute(
            "SELECT source, is_protected FROM product_field_metadata "
            "WHERE product_key='000123' AND field_name='proteins_100g'"
        ).fetchone(), ("MANUAL", True))

        other = "000456"
        self.add_product(key=other, category=None, macro=None)
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES (?, 'proteins_100g', 'MANUAL', TRUE)""",
            [other],
        )
        conflict = preview_chat_batch_import(
            self.con,
            self.response({"product_key": other, "category": "Non-Food"}),
            self.taxonomy,
            self.category_to_macro,
        )
        self.assertFalse(any(item.product_key == other for item in conflict.suggestions))
        self.assertIn("protected_non_food_conflict", {item.code for item in conflict.issues})
        save_chat_batch_import(
            self.con,
            self.response({"product_key": other, "category": "Non-Food"}),
            self.taxonomy,
            self.category_to_macro,
        )
        self.assertEqual(self.con.execute(
            "SELECT category, proteins_100g FROM nutrition_cache WHERE product_key=?", [other]
        ).fetchone(), (None, 3.0))
        self.assertEqual(self.con.execute(
            "SELECT source, is_protected FROM product_field_metadata "
            "WHERE product_key=? AND field_name='proteins_100g'", [other]
        ).fetchone(), ("MANUAL", True))

    def test_transaction_rolls_back_category_and_provenance_if_package_write_fails(self):
        con = duckdb.connect(":memory:")
        try:
            con.execute(NUTRITION_DDL.replace(
                "package_weight_g DOUBLE,",
                "package_weight_g DOUBLE CHECK (package_weight_g IS NULL OR package_weight_g < 100),",
            ))
            create_schema(con)
            values = dict(VALID_NUTRIENTS)
            columns = (
                "product_key", "sample_name", "macro_category", "category",
                "package_weight_g", *NUTRIENTS, "match_status", "data_source",
            )
            row = (
                "rollback", "Rollback product", None, None, None,
                *(values[field] for field in NUTRIENTS), "TEST", "SYNTHETIC",
            )
            con.execute(
                f"INSERT INTO nutrition_cache ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                row,
            )
            with self.assertRaises(duckdb.ConstraintException):
                save_chat_batch_import(
                    con,
                    self.response({
                        "product_key": "rollback",
                        "category": "Dairy & Eggs",
                        "package_weight_g": 100,
                    }),
                    self.taxonomy,
                    self.category_to_macro,
                )
            self.assertEqual(con.execute(
                "SELECT category, macro_category, package_weight_g "
                "FROM nutrition_cache WHERE product_key='rollback'"
            ).fetchone(), (None, None, None))
            self.assertEqual(con.execute("SELECT COUNT(*) FROM product_field_metadata").fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM ai_enrichment_state").fetchone()[0], 0)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
