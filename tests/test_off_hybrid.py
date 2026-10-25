import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import duckdb
import requests

import sync_incremental as sync
from product_store import save_manual_fields


class TestOffHybridEngine(unittest.TestCase):
    def setUp(self):
        # Save original configs to restore in tearDown
        self.orig_off_mode = sync.OFF_MODE
        self.orig_ttl = sync.OFF_NOT_FOUND_TTL_DAYS

    def tearDown(self):
        sync.OFF_MODE = self.orig_off_mode
        sync.OFF_NOT_FOUND_TTL_DAYS = self.orig_ttl

    @patch("sync_incremental.requests.get")
    def test_fetch_off_product_api_success(self, mock_get):
        """Test successful Open Food Facts API response parsing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": 1,
            "product": {
                "product_name": "Lapte 1.5%",
                "product_name_ro": "Lapte Semidegresat 1.5%",
                "brands": "Milbona",
                "nova_group": 1,
                "categories_tags": ["en:dairies", "en:milks"],
                "nutriments": {
                    "energy-kcal_100g": 45.0,
                    "proteins_100g": 3.2,
                    "carbohydrates_100g": 4.8,
                    "sugars_100g": 4.8,
                    "fat_100g": 1.5,
                    "saturated-fat_100g": 1.0,
                    "fiber_100g": 0.0,
                    "salt_100g": 0.13
                }
            }
        }
        mock_get.return_value = mock_resp

        res = sync.fetch_off_product_api("4056489360155")
        self.assertIsNotNone(res)
        self.assertFalse(res["not_found"])
        self.assertEqual(res["code"], "4056489360155")
        self.assertEqual(res["product_name_ro"], "Lapte Semidegresat 1.5%")
        self.assertEqual(res["brands"], "Milbona")
        self.assertEqual(res["nova_group"], 1)
        self.assertIn("en:milks", res["categories_tags"])
        self.assertEqual(res["nutriments"]["energy-kcal_100g"], 45.0)

    @patch("sync_incremental.requests.get")
    def test_fetch_off_product_api_not_found_404(self, mock_get):
        """Test that HTTP 404 is identified as definitively not found."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        res = sync.fetch_off_product_api("5941234567890")
        self.assertIsNotNone(res)
        self.assertTrue(res["not_found"])
        self.assertEqual(res["code"], "5941234567890")

    @patch("sync_incremental.requests.get")
    def test_fetch_off_product_api_status_zero(self, mock_get):
        """Test that HTTP 200 with status=0 is identified as definitively not found."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": 0,
            "status_verbose": "product not found"
        }
        mock_get.return_value = mock_resp

        res = sync.fetch_off_product_api("5941234567891")
        self.assertIsNotNone(res)
        self.assertTrue(res["not_found"])
        self.assertEqual(res["code"], "5941234567891")

    @patch("sync_incremental.requests.get")
    def test_fetch_off_product_api_network_error(self, mock_get):
        """Test that network connection errors return None for retry."""
        mock_get.side_effect = requests.RequestException("Connection timed out")

        res = sync.fetch_off_product_api("5941234567892")
        self.assertIsNone(res)

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.requests.get")
    def test_fetch_off_product_api_rate_limit_backoff(self, mock_get, mock_sleep):
        """Test HTTP 429 rate limit backoff and successful retry."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "status": 1,
            "product": {"product_name": "Test Drink"}
        }

        mock_get.side_effect = [resp_429, resp_200]

        res = sync.fetch_off_product_api("5941234567893")
        self.assertIsNotNone(res)
        self.assertEqual(res["product_name"], "Test Drink")
        mock_sleep.assert_called_with(10.0)

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.requests.get")
    def test_fetch_off_product_api_retry_after_header(self, mock_get, mock_sleep):
        """Test HTTP 429 parsing Retry-After header duration."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "5"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "status": 1,
            "product": {"product_name": "Test Juice"}
        }

        mock_get.side_effect = [resp_429, resp_200]

        res = sync.fetch_off_product_api("5941234567894")
        self.assertIsNotNone(res)
        self.assertEqual(res["product_name"], "Test Juice")
        mock_sleep.assert_called_with(5.0)

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.fetch_off_product_api")
    def test_update_nutrition_cache_live_api_end_to_end(self, mock_fetch, mock_sleep):
        """Test end-to-end Live API enrichment and DuckDB SQL Waterfall classification."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Seed purchases with 1 global EAN and 1 non-EAN code
        con.execute("""
            INSERT INTO purchases (
                ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean
            ) VALUES 
            ('T1', '2026-10-18', 1, 'Milbona Milk', 1, 5.0, 5.0, '554056489360155', 'GLOBAL_EAN', '4056489360155'),
            ('T1', '2026-10-18', 2, 'Banane Cantar', 1.5, 6.0, 9.0, '550000050001', 'PLU_SHORT', '50001');
        """)

        # Mock API returning dairy product for EAN
        mock_fetch.return_value = {
            "code": "4056489360155",
            "product_name": "Lapte 1.5%",
            "product_name_ro": "Lapte Semidegresat 1.5%",
            "product_name_en": "Milk 1.5%",
            "brands": "Milbona",
            "nova_group": 1,
            "categories_tags": ["en:dairies", "en:milks"],
            "nutriments": {
                "energy-kcal_100g": 45.0,
                "proteins_100g": 3.2,
                "carbohydrates_100g": 4.8,
                "sugars_100g": 4.8,
                "fat_100g": 1.5,
                "saturated-fat_100g": 1.0,
                "fiber_100g": 0.0,
                "salt_100g": 0.13
            },
            "not_found": False
        }

        sync.OFF_MODE = "api"
        sync.update_nutrition_cache_from_off(con)

        # Check non-EAN item
        banana_row = con.execute("SELECT product_key, match_status, data_source FROM nutrition_cache WHERE product_key = '50001'").fetchone()
        self.assertIsNotNone(banana_row)
        self.assertEqual(banana_row[1], "NO_GLOBAL_EAN")
        self.assertEqual(banana_row[2], "BON_ONLY")

        # Check enriched global EAN item
        milk_row = con.execute("""
            SELECT product_key, off_product_name, off_brands, category, macro_category,
                   kcal_100g, proteins_100g, match_status, data_source
            FROM nutrition_cache 
            WHERE product_key = '4056489360155'
        """).fetchone()

        self.assertIsNotNone(milk_row)
        self.assertEqual(milk_row[0], "4056489360155")
        self.assertEqual(milk_row[1], "Lapte Semidegresat 1.5%")
        self.assertEqual(milk_row[2], "Milbona")
        self.assertEqual(milk_row[3], "Dairy & Eggs")
        self.assertEqual(milk_row[4], "Fresh Food")
        self.assertEqual(milk_row[5], 45.0)
        self.assertEqual(milk_row[6], 3.2)
        self.assertEqual(milk_row[7], "MATCHED_OFF_WATERFALL")
        self.assertEqual(milk_row[8], "OFF_API")

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.fetch_off_product_api")
    def test_update_nutrition_cache_live_api_not_found(self, mock_fetch, mock_sleep):
        """Test that missing products are safely recorded as NOT_FOUND_OFF in nutrition_cache."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        con.execute("""
            INSERT INTO purchases (
                ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean
            ) VALUES 
            ('T1', '2026-10-18', 1, 'Necunoscut', 1, 10.0, 10.0, '555941234567890', 'GLOBAL_EAN', '5941234567890');
        """)

        mock_fetch.return_value = {
            "code": "5941234567890",
            "not_found": True
        }

        sync.OFF_MODE = "api"
        sync.update_nutrition_cache_from_off(con)

        row = con.execute("""
            SELECT product_key, match_status, data_source, is_verified 
            FROM nutrition_cache 
            WHERE product_key = '5941234567890'
        """).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], "NOT_FOUND_OFF")
        self.assertEqual(row[2], "BON_ONLY")
        self.assertFalse(row[3])

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.fetch_off_product_api")
    def test_update_nutrition_cache_ttl_auto_healing(self, mock_fetch, mock_sleep):
        """Test that expired NOT_FOUND_OFF items (>30 days) are re-queried and updated."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Insert historical purchase
        con.execute("""
            INSERT INTO purchases (
                ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean
            ) VALUES 
            ('T1', '2026-08-01', 1, 'Old Cola', 1, 4.0, 4.0, '554056489999999', 'GLOBAL_EAN', '4056489999999');
        """)

        # Insert historical NOT_FOUND_OFF cached 45 days ago
        con.execute("""
            INSERT INTO nutrition_cache (
                product_key, sample_name, ean, match_status, data_source, is_verified, cached_at
            ) VALUES (
                '4056489999999', 'Old Cola', '4056489999999', 'NOT_FOUND_OFF', 'BON_ONLY', FALSE,
                CURRENT_TIMESTAMP - INTERVAL 45 DAY
            );
        """)
        save_manual_fields(
            con,
            "4056489999999",
            {"category": "Dairy & Eggs", "salt_100g": 0.5},
            {"Dairy & Eggs": "Fresh Food"},
        )

        # Now mock API returning data for this previously missing item (auto-healing)
        mock_fetch.return_value = {
            "code": "4056489999999",
            "product_name": "Freeway Cola",
            "brands": "Freeway",
            "categories_tags": ["en:beverages", "en:soft-drinks", "en:sodas"],
            "nutriments": {
                "energy-kcal_100g": 42.0,
                "sugars_100g": 10.5
            },
            "not_found": False
        }

        sync.OFF_MODE = "api"
        sync.OFF_NOT_FOUND_TTL_DAYS = 30
        sync.update_nutrition_cache_from_off(con)

        row = con.execute("""
            SELECT product_key, off_product_name, category, macro_category, match_status, data_source
            FROM nutrition_cache 
            WHERE product_key = '4056489999999'
        """).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], "Freeway Cola")
        self.assertEqual(row[2], "Dairy & Eggs")
        self.assertEqual(row[3], "Fresh Food")
        self.assertEqual(row[4], "MATCHED_OFF_WATERFALL")
        self.assertEqual(row[5], "OFF_API")
        protected_values = con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key = '4056489999999'"
        ).fetchone()[0]
        self.assertEqual(protected_values, 0.5)
        self.assertEqual(con.execute(
            "SELECT kcal_100g FROM nutrition_cache WHERE product_key = '4056489999999'"
        ).fetchone()[0], 42.0)


if __name__ == "__main__":
    unittest.main()
