import unittest
import duckdb
from unittest.mock import patch, MagicMock

import sync_incremental as sync


class TestAuditRemediation(unittest.TestCase):
    def test_parse_weight_from_text_fat_percentages(self):
        """Test that % GR fat tokens are stripped and true package weights are extracted."""
        # 1. Smântână 12% GR 200G -> 200g
        self.assertEqual(sync.parse_weight_from_text("SMANTANA 12% GR 200G"), 200.0)
        # 2. Iaurt 2.8%GR 140G -> 140g
        self.assertEqual(sync.parse_weight_from_text("IAURT 2.8%GR 140G"), 140.0)
        # 3. Smântână 12 GR 200G (missing % symbol) -> 200g
        self.assertEqual(sync.parse_weight_from_text("SMANTANA 12 GR 200G"), 200.0)
        # 4. Unt 82% grasime 200G -> 200g
        self.assertEqual(sync.parse_weight_from_text("UNT 82% GRASIME 200G"), 200.0)
        # 5. Telemea vacă 50% gr/su 300g -> 300g
        self.assertEqual(sync.parse_weight_from_text("TELEMEA VACA 50% GR/SU 300G"), 300.0)
        # 6. Small standalone package weight without larger weight -> 25g
        self.assertEqual(sync.parse_weight_from_text("DROJDIE PROASPATA 25G"), 25.0)

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.fetch_off_product_api")
    def test_canned_goods_waterfall_priority_over_fresh(self, mock_fetch, mock_sleep):
        """Test that canned fish and canned tomatoes are classified as Pantry, not Fresh."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        con.execute("""
            INSERT INTO purchases (
                ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean
            ) VALUES 
            ('T1', '2026-10-18', 1, 'Ton Bucati Conserva', 1, 8.0, 8.0, '555941111111111', 'GLOBAL_EAN', '5941111111111'),
            ('T1', '2026-10-18', 2, 'Rosii decojite conserva', 1, 5.0, 5.0, '555942222222222', 'GLOBAL_EAN', '5942222222222'),
            ('T1', '2026-10-18', 3, 'Croissant Ambalat Ciocolata', 1, 3.0, 3.0, '555943333333333', 'GLOBAL_EAN', '5943333333333');
        """)

        def side_effect_fetch(ean, session=None):
            if ean == "5941111111111":
                return {
                    "code": ean,
                    "product_name": "Canned Tuna in Oil",
                    "categories_tags": ["en:fishes", "en:canned-fishes", "en:canned-foods", "en:tunas"],
                    "nutriments": {"energy-kcal_100g": 180.0, "proteins_100g": 25.0, "fat_100g": 9.0, "carbohydrates_100g": 0.0},
                    "not_found": False
                }
            elif ean == "5942222222222":
                return {
                    "code": ean,
                    "product_name": "Canned Peeled Tomatoes",
                    "categories_tags": ["en:vegetables", "en:tomatoes", "en:canned-vegetables"],
                    "nutriments": {"energy-kcal_100g": 22.0, "proteins_100g": 1.2, "fat_100g": 0.2, "carbohydrates_100g": 3.5},
                    "not_found": False
                }
            elif ean == "5943333333333":
                return {
                    "code": ean,
                    "product_name": "Packaged Croissant",
                    "categories_tags": ["en:viennoiseries", "en:croissants", "en:pastries"],
                    "nutriments": {"energy-kcal_100g": 410.0, "proteins_100g": 7.0, "fat_100g": 20.0, "carbohydrates_100g": 50.0},
                    "not_found": False
                }
            return None

        mock_fetch.side_effect = side_effect_fetch

        sync.OFF_MODE = "api"
        sync.update_nutrition_cache_from_off(con)

        # 1. Canned tuna -> Canned Goods & Pickles / Pantry & Groceries
        tuna_row = con.execute("SELECT category, macro_category FROM nutrition_cache WHERE product_key = '5941111111111'").fetchone()
        self.assertIsNotNone(tuna_row)
        self.assertEqual(tuna_row[0], "Canned Goods & Pickles")
        self.assertEqual(tuna_row[1], "Pantry & Groceries")

        # 2. Canned tomatoes -> Canned Goods & Pickles / Pantry & Groceries
        tomato_row = con.execute("SELECT category, macro_category FROM nutrition_cache WHERE product_key = '5942222222222'").fetchone()
        self.assertIsNotNone(tomato_row)
        self.assertEqual(tomato_row[0], "Canned Goods & Pickles")
        self.assertEqual(tomato_row[1], "Pantry & Groceries")

        # 3. Packaged croissant with EAN -> Desserts & Pastries / Sweets & Snacks
        croissant_row = con.execute("SELECT category, macro_category FROM nutrition_cache WHERE product_key = '5943333333333'").fetchone()
        self.assertIsNotNone(croissant_row)
        self.assertEqual(croissant_row[0], "Desserts & Pastries")
        self.assertEqual(croissant_row[1], "Sweets & Snacks")

    def test_validate_atwater_exempts_alcoholic_beverages(self):
        """Test that Alcoholic Beverages with ethanol calories pass Atwater validation."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Beer: 42 kcal, 0.5g prot, 3.5g carbs, 0g fat (macros give 16 kcal, but ethanol gives 26 kcal)
        # Under naive Atwater: abs(42 - 16) = 26 > 25 -> fails!
        # With alcoholic exemption: passes!
        con.execute("""
            INSERT INTO nutrition_cache (
                product_key, sample_name, macro_category, category,
                kcal_100g, proteins_100g, carbs_100g, sugars_100g, fat_100g, saturated_fat_100g, fiber_100g, salt_100g,
                is_verified, match_status
            ) VALUES (
                '5941234500001', 'Bere Blonda 500ml', 'Beverages', 'Alcoholic Beverages',
                42.0, 0.5, 3.5, 0.5, 0.0, 0.0, 0.0, 0.01,
                FALSE, 'MATCHED_OFF'
            );
        """)

        sync.validate_atwater(con)

        row = con.execute("""
            SELECT kcal_100g, match_status, is_verified, atwater_debug
            FROM nutrition_cache
            WHERE product_key = '5941234500001'
        """).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 42.0)  # Calories must NOT be wiped!
        self.assertEqual(row[1], "MATCHED_OFF_VERIFIED")
        self.assertTrue(row[2])
        self.assertIn("Alcoholic Beverages", row[3])

    def test_non_food_filter_sets_explicit_categories(self):
        """Test that deterministic non-food filter explicitly sets macro and category to 'Non-Food'."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        con.execute("""
            INSERT INTO purchases (ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean)
            VALUES ('T1', '2026-10-18', 1, 'HARTIE IGIENICA 3 STRATURI', 1, 15.0, 15.0, '5940000000001', 'GLOBAL_EAN', '5940000000001');
        """)
        con.execute("""
            INSERT INTO nutrition_cache (product_key, sample_name, ean, match_status, is_verified)
            VALUES ('5940000000001', 'HARTIE IGIENICA 3 STRATURI', '5940000000001', 'NO_GLOBAL_EAN', FALSE);
        """)

        sync.run_deterministic_non_food_filter(con)

        row = con.execute("""
            SELECT macro_category, category, match_status, kcal_100g, is_verified
            FROM nutrition_cache
            WHERE product_key = '5940000000001'
        """).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "Non-Food")
        self.assertEqual(row[1], "Non-Food")
        self.assertEqual(row[2], "NON_FOOD")
        self.assertIsNone(row[3])
        self.assertTrue(row[4])

    def test_variable_weight_25_prefix_inheritance(self):
        """Test that weighable caseroles (25...) inherit nutrition and category via 7-digit prefix."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # 1. Existing verified "parent" item in nutrition_cache with prefix 2501234
        con.execute("""
            INSERT INTO nutrition_cache (
                product_key, sample_name, macro_category, category,
                kcal_100g, proteins_100g, carbs_100g, sugars_100g, fat_100g, saturated_fat_100g, fiber_100g, salt_100g,
                is_verified, match_status, data_source
            ) VALUES (
                '2501234005001', 'Piept pui dezosat caserola 500g', 'Fresh Food', 'Fresh Meat & Fish',
                110.0, 23.0, 0.0, 0.0, 2.0, 0.6, 0.0, 0.15,
                TRUE, 'ENRICHED_MANUAL', 'MANUAL_CSV'
            );
        """)

        # 2. New purchase of the same product with a different weight/price: 2501234004508 (450g)
        con.execute("""
            INSERT INTO nutrition_cache (
                product_key, sample_name, macro_category, category,
                is_verified, match_status, data_source
            ) VALUES (
                '2501234004508', 'Piept pui dezosat caserola', NULL, NULL,
                FALSE, 'NO_GLOBAL_EAN', 'BON_ONLY'
            );
        """)

        sync.compute_weights_and_quantities(con)

        row = con.execute("""
            SELECT package_weight_g, macro_category, category, kcal_100g, proteins_100g, match_status, data_source, is_verified
            FROM nutrition_cache
            WHERE product_key = '2501234004508'
        """).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 450.0)  # Decoded piece weight from digits 8-12
        self.assertEqual(row[1], "Fresh Food")
        self.assertEqual(row[2], "Fresh Meat & Fish")
        self.assertEqual(row[3], 110.0)
        self.assertEqual(row[4], 23.0)
        self.assertEqual(row[5], "INHERITED_PREFIX")
        self.assertEqual(row[6], "PREFIX_INHERIT")
        self.assertTrue(row[7])

    @patch("sync_incremental.requests.Session")
    def test_fetch_off_batch_api_success(self, mock_session_cls):
        """Test batch OFF API parses multi-code response correctly."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"products": [{"code": "594111", "product_name": "Tuna", "nutriments": {"energy-kcal_100g": 150}}]}'
        mock_resp.json.return_value = {
            "products": [
                {"code": "594111", "product_name": "Tuna", "nutriments": {"energy-kcal_100g": 150}}
            ]
        }
        mock_session.get.return_value = mock_resp

        results = sync.fetch_off_batch_api(["594111", "594222"], session=mock_session)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)
        # First product found
        self.assertEqual(results[0]["code"], "594111")
        self.assertFalse(results[0]["not_found"])
        self.assertEqual(results[0]["product_name"], "Tuna")
        # Second product missing
        self.assertEqual(results[1]["code"], "594222")
        self.assertTrue(results[1]["not_found"])

    @patch("sync_incremental.requests.Session")
    def test_fetch_off_product_api_v3_success(self, mock_session_cls):
        """Test single product API v3 success parsing."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"status": "success", "product": {"product_name": "Iaurt Grecesc", "brands": "Pilos"}}'
        mock_resp.json.return_value = {
            "status": "success",
            "product": {"product_name": "Iaurt Grecesc", "brands": "Pilos"}
        }
        mock_session.get.return_value = mock_resp

        res = sync.fetch_off_product_api("4056489360155", session=mock_session)
        self.assertIsNotNone(res)
        self.assertFalse(res["not_found"])
        self.assertEqual(res["product_name"], "Iaurt Grecesc")
        self.assertEqual(res["brands"], "Pilos")


if __name__ == "__main__":
    unittest.main()

