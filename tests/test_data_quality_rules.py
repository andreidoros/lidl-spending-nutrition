"""Audit package 3: one set of nutrition rules, OFF category order, compact answers."""

import csv
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

import sync_incremental as sync
from ai_enrichment.chat_batch import parse_compact_answer
from ai_eval.parsing import validate_response
from ai_eval.taxonomy import load_taxonomy
from product_store import atwater_check

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY = load_taxonomy()
NUTRIENTS = (
    "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
    "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
)
CONSISTENCY_CODES = {
    "macro_sum_exceeds_105", "sugars_exceeds_carbs", "saturated_fat_exceeds_fat",
    "atwater_mismatch", "fibre_energy_exceeds_kcal",
}


def ai_issues(category, **values):
    item = {"product_key": "1", "category": category, "abstain_reason": None}
    item.update({name: values.get(name) for name in NUTRIENTS})
    result = validate_response({"products": [item]}, ["1"], TAXONOMY, "estimate", allow_abstain=True)
    return set(result.products["1"].issue_codes) & CONSISTENCY_CODES


MIX_LEGUME = dict(kcal_100g=28.0, proteins_100g=1.5, carbs_100g=3.4, sugars_100g=2.9,
                  fat_100g=0.3, saturated_fat_100g=0.3, salt_100g=0.05)
PAPRIKA = dict(kcal_100g=282.0, proteins_100g=14.1, carbs_100g=54.0, sugars_100g=10.3,
               fat_100g=12.9, saturated_fat_100g=2.1, fiber_100g=34.9, salt_100g=0.08)


class FibreEnergyRuleTests(unittest.TestCase):
    def test_fibre_cannot_give_more_calories_than_the_product(self):
        category = "Ready Meals & Chilled Convenience"
        bad = dict(MIX_LEGUME, fiber_100g=15.0)  # 30 kcal of fibre in a 28 kcal product
        good = dict(MIX_LEGUME, fiber_100g=2.3)
        self.assertEqual(atwater_check({"category": category, **bad}), "Fibre has more calories than the product")
        self.assertIn("fibre_energy_exceeds_kcal", ai_issues(category, **bad))
        self.assertIsNone(atwater_check({"category": category, **good}))
        self.assertEqual(ai_issues(category, **good), set())

    def test_konjac_noodles_with_little_energy_still_pass(self):
        # "Taitei Konjac" as stored in the seed catalog.
        konjac = dict(kcal_100g=10.0, proteins_100g=0.1, carbs_100g=3.0, sugars_100g=0.0,
                      fat_100g=0.0, saturated_fat_100g=0.0, fiber_100g=3.0, salt_100g=0.1)
        category = "Baking & Cooking Ingredients"
        self.assertTrue(TAXONOMY.is_category(category))
        self.assertIsNone(atwater_check({"category": category, **konjac}))
        self.assertEqual(ai_issues(category, **konjac), set())


class SpiceRuleTests(unittest.TestCase):
    def test_label_values_of_paprika_pass_the_ai_validator(self):
        self.assertIsNone(atwater_check({"category": "Herbs & Spices", **PAPRIKA}))
        self.assertEqual(ai_issues("Herbs & Spices", **PAPRIKA), set())

    def test_fibre_inside_carbs_is_not_counted_twice(self):
        # Wholemeal-style label: 60 g carbs of which 12 g fibre, 12 g protein, 3 g fat.
        values = dict(kcal_100g=320.0, proteins_100g=12.0, carbs_100g=60.0, sugars_100g=2.0,
                      fat_100g=3.0, saturated_fat_100g=0.5, fiber_100g=12.0, salt_100g=1.0)
        self.assertEqual(ai_issues("Bread & Bakery Specialties", **values), set())


class ValidatorParityTests(unittest.TestCase):
    def test_sync_check_and_ai_validator_agree_on_every_seed_food(self):
        path = ROOT / "data" / "catalogs" / "ro" / "catalog_seed.csv"
        disagreements = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                category = row["category"]
                if not category or category == "Non-Food" or not TAXONOMY.is_category(category):
                    continue
                values = {name: float(row[name]) if row[name] else None for name in NUTRIENTS}
                sync_verdict = atwater_check({"category": category, **values}) is None
                ai_verdict = not ai_issues(category, **values)
                if sync_verdict != ai_verdict:
                    disagreements.append((row["product_key"], row["sample_name"]))
        self.assertEqual(disagreements, [])


class OffCategoryOrderTests(unittest.TestCase):
    def classify(self, tags):
        con = duckdb.connect(":memory:")
        sync.init_database(con)
        con.execute("""
            INSERT INTO purchases (ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean)
            VALUES ('T1', '2026-11-01', 1, 'Product', 1, 4.0, 4.0, '4056489999999', 'GLOBAL_EAN', '4056489999999')
        """)
        answer = {
            "code": "4056489999999", "product_name": "Product", "brands": "Test",
            "categories_tags": tags, "nutriments": {"energy-kcal_100g": 60.0}, "not_found": False,
        }
        orig_mode = sync.OFF_MODE
        sync.OFF_MODE = "api"
        try:
            with patch("sync_incremental.fetch_off_product_api", return_value=answer), \
                 patch("sync_incremental.time.sleep"):
                sync.update_nutrition_cache_from_off(con)
        finally:
            sync.OFF_MODE = orig_mode
        return con.execute("SELECT category FROM nutrition_cache WHERE product_key = '4056489999999'").fetchone()[0]

    def test_drinkable_dairy_is_dairy_not_a_soft_drink(self):
        tags = ["en:beverages", "en:dairies", "en:fermented-dairy-products", "en:kefirs"]
        self.assertEqual(self.classify(tags), "Dairy & Eggs")

    def test_plain_soft_drinks_stay_soft_drinks(self):
        self.assertEqual(self.classify(["en:beverages", "en:sodas"]), "Water & Soft Drinks")

    def test_salamis_are_deli_meat(self):
        self.assertEqual(self.classify(["en:meats", "en:salamis"]), "Processed Meat & Deli")


class CompactColumnTests(unittest.TestCase):
    def parse(self, *lines):
        payload, unreadable = parse_compact_answer("\n".join(lines), "nutrition")
        return payload["products"], unreadable

    def test_copied_name_with_empty_category_does_not_shift_columns(self):
        products, unreadable = self.parse("4056489000001;Branza;;80;10;5;1;2;1;0;0,5;250")
        self.assertEqual(unreadable, 0)
        self.assertEqual(products[0], {
            "product_key": "4056489000001", "kcal_100g": 80.0, "proteins_100g": 10.0,
            "carbs_100g": 5.0, "sugars_100g": 1.0, "fat_100g": 2.0, "saturated_fat_100g": 1.0,
            "fiber_100g": 0.0, "salt_100g": 0.5, "package_weight_g": 250.0,
        })

    def test_na_means_empty_and_keeps_the_category(self):
        products, _ = self.parse("4056489000002;Dairy & Eggs;N/A;5;4;3;2;1;null;0,1;-")
        self.assertEqual(products[0]["category"], "Dairy & Eggs")
        self.assertNotIn("kcal_100g", products[0])
        self.assertEqual(products[0]["proteins_100g"], 5.0)
        self.assertNotIn("fiber_100g", products[0])
        self.assertNotIn("package_weight_g", products[0])

    def test_trailing_semicolon_is_not_a_copied_name(self):
        products, _ = self.parse("4056489000003;Dairy & Eggs;;5;4;3;2;1;0;0,1;;")
        self.assertEqual(products[0]["category"], "Dairy & Eggs")
        self.assertEqual(products[0]["proteins_100g"], 5.0)

    def test_copied_name_and_empty_weight(self):
        products, _ = self.parse("4056489000004;Lapte 1,5%;Dairy & Eggs;45;3,3;4,8;4,8;1,5;1;0;0,1;")
        self.assertEqual(products[0]["category"], "Dairy & Eggs")
        self.assertEqual(products[0]["kcal_100g"], 45.0)
        self.assertEqual(products[0]["salt_100g"], 0.1)

    def test_too_many_values_are_not_guessed(self):
        products, unreadable = self.parse("4056489000005;A;B;Dairy & Eggs;1;2;3;4;5;6;7;8;9")
        self.assertEqual(products, [])
        self.assertEqual(unreadable, 1)


if __name__ == "__main__":
    unittest.main()
