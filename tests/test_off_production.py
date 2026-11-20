"""Open Food Facts uses production servers only, plus the one-time re-query."""

import unittest
from unittest.mock import MagicMock, patch

import duckdb

import sync_incremental as sync


def _response(status_code, payload=None, headers=None):
    response = MagicMock()
    response.status_code = status_code
    response.content = b""
    response.json.return_value = payload or {}
    response.headers = headers or {}
    return response


class OffProductionServerTests(unittest.TestCase):
    def test_no_staging_server_is_ever_asked(self):
        for url in sync.OFF_PRODUCT_URLS + sync.OFF_SEARCH_URLS:
            self.assertIn("openfoodfacts.org", url)
            self.assertNotIn("openfoodfacts.net", url)

    @patch("sync_incremental.requests.get")
    def test_single_product_lookup_asks_production_first(self, mock_get):
        mock_get.return_value = _response(404)
        result = sync.fetch_off_product_api("4056489999999")
        self.assertEqual(result, {"code": "4056489999999", "not_found": True})
        first_url = mock_get.call_args_list[0].args[0]
        self.assertEqual(first_url, "https://world.openfoodfacts.org/api/v3/product/4056489999999.json")

    @patch("sync_incremental.requests.get")
    def test_batch_lookup_never_asks_staging(self, mock_get):
        mock_get.return_value = _response(500)
        sync.fetch_off_batch_api(["4056489999999"])
        urls = [call.args[0] for call in mock_get.call_args_list]
        self.assertTrue(urls)
        self.assertTrue(all("openfoodfacts.org" in url for url in urls), urls)

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.requests.get")
    def test_retry_after_is_capped(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _response(429, headers={"Retry-After": "86400"}),
            _response(404),
        ]
        sync.fetch_off_product_api("4056489999999")
        mock_sleep.assert_called_once_with(sync.MAX_RETRY_AFTER_SECONDS)


class StagingNotFoundRequeryTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect(":memory:")
        sync.init_database(self.con)
        self.con.execute("""
            INSERT INTO nutrition_cache (product_key, sample_name, ean, match_status, data_source, is_verified, cached_at)
            VALUES
              ('1111111111116', 'Recent not found', '1111111111116', 'NOT_FOUND_OFF', 'BON_ONLY', FALSE, CURRENT_TIMESTAMP),
              ('2222222222222', 'Manual row', '2222222222222', 'NOT_FOUND_OFF', 'MANUAL_CSV', FALSE, CURRENT_TIMESTAMP),
              ('3333333333333', 'Matched', '3333333333333', 'MATCHED_OFF', 'OFF_API', FALSE, CURRENT_TIMESTAMP)
        """)

    def _cached_year(self, key):
        return self.con.execute(
            "SELECT YEAR(cached_at) FROM nutrition_cache WHERE product_key = ?", [key]
        ).fetchone()[0]

    def test_marks_only_automatic_not_found_rows_once(self):
        self.assertEqual(sync.requery_staging_not_found(self.con), 1)
        self.assertEqual(self._cached_year("1111111111116"), 2000)
        self.assertNotEqual(self._cached_year("2222222222222"), 2000)
        self.assertNotEqual(self._cached_year("3333333333333"), 2000)

        # The second sync leaves new "not found" answers alone.
        self.con.execute(
            "UPDATE nutrition_cache SET cached_at = CURRENT_TIMESTAMP WHERE product_key = '1111111111116'"
        )
        self.assertEqual(sync.requery_staging_not_found(self.con), 0)
        self.assertNotEqual(self._cached_year("1111111111116"), 2000)
        recorded = self.con.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_id = ?",
            [sync.OFF_STAGING_REQUERY_MIGRATION],
        ).fetchone()[0]
        self.assertEqual(recorded, 1)

    @patch("sync_incremental.time.sleep")
    @patch("sync_incremental.fetch_off_product_api")
    def test_sync_asks_again_about_a_recent_not_found(self, mock_fetch, _sleep):
        self.con.execute("""
            INSERT INTO purchases (ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean)
            VALUES ('T1', '2026-11-01', 1, 'Recent not found', 1, 4.0, 4.0, '1111111111116', 'GLOBAL_EAN', '1111111111116')
        """)
        mock_fetch.return_value = {"code": "1111111111116", "not_found": True}
        orig_mode, orig_ttl = sync.OFF_MODE, sync.OFF_NOT_FOUND_TTL_DAYS
        sync.OFF_MODE, sync.OFF_NOT_FOUND_TTL_DAYS = "api", 30
        try:
            sync.update_nutrition_cache_from_off(self.con)
        finally:
            sync.OFF_MODE, sync.OFF_NOT_FOUND_TTL_DAYS = orig_mode, orig_ttl
        asked = [call.args[0] for call in mock_fetch.call_args_list]
        self.assertIn("1111111111116", asked)


class AppConfigTests(unittest.TestCase):
    def test_app_listens_on_this_computer_only_and_sends_no_stats(self):
        import tomllib
        from pathlib import Path

        config = tomllib.loads((Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["server"]["address"], "127.0.0.1")
        self.assertFalse(config["browser"]["gatherUsageStats"])

    def test_requirements_ask_for_the_streamlit_the_app_needs(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("streamlit>=1.66.0,<2.0.0", text)


if __name__ == "__main__":
    unittest.main()
