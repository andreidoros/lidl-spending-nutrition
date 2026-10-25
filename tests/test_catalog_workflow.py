import unittest
import sys
import tempfile
import json
import csv
from pathlib import Path
import duckdb

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sync_incremental as sync


class TestCatalogWorkflow(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test catalogs and CSV files
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Patch paths in sync_incremental
        self.orig_base_dir = sync.CATALOGS_BASE_DIR
        self.orig_tax_path = sync.TAXONOMY_PATH
        self.orig_seed_path = sync.SEED_CATALOG_PATH
        self.orig_ovr_path = sync.OVERRIDE_CSV_PATH
        self.orig_queue_path = sync.REVIEW_QUEUE_PATH

        sync.CATALOGS_BASE_DIR = self.test_path / "catalogs"
        sync.CATALOGS_BASE_DIR.mkdir(parents=True, exist_ok=True)
        sync.TAXONOMY_PATH = sync.CATALOGS_BASE_DIR / "taxonomy.json"
        
        country_dir = sync.CATALOGS_BASE_DIR / "ro"
        country_dir.mkdir(parents=True, exist_ok=True)
        sync.SEED_CATALOG_PATH = country_dir / "catalog_seed.csv"
        
        sync.OVERRIDE_CSV_PATH = self.test_path / "catalog_overrides.csv"
        sync.REVIEW_QUEUE_PATH = self.test_path / "review_queue.csv"

        # Create a test canonical English taxonomy
        self.tax_data = {
            "Fresh Food": ["Fresh Meat & Fish", "Dairy & Eggs"],
            "Sweets & Snacks": ["Chocolate & Confectionery"]
        }
        with open(sync.TAXONOMY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.tax_data, f)

    def tearDown(self):
        # Restore paths and cleanup temp directory
        sync.CATALOGS_BASE_DIR = self.orig_base_dir
        sync.TAXONOMY_PATH = self.orig_tax_path
        sync.SEED_CATALOG_PATH = self.orig_seed_path
        sync.OVERRIDE_CSV_PATH = self.orig_ovr_path
        sync.REVIEW_QUEUE_PATH = self.orig_queue_path
        self.test_dir.cleanup()

    def test_load_taxonomy_reverse_mapping(self):
        tax, cat_to_macro = sync.load_taxonomy()
        self.assertEqual(len(tax), 2)
        self.assertEqual(cat_to_macro.get("Fresh Meat & Fish"), "Fresh Food")
        self.assertEqual(cat_to_macro.get("Chocolate & Confectionery"), "Sweets & Snacks")

    def test_live_taxonomy_json_integrity(self):
        """Verify data/catalogs/taxonomy.json is strictly disjoint (1:N tree) and Fresh Bakery & Pastry is in Bakery & Bread."""
        live_tax_path = Path(__file__).resolve().parent.parent / "data" / "catalogs" / "taxonomy.json"
        with open(live_tax_path, "r", encoding="utf-8") as f:
            live_tax = json.load(f)

        seen_cats = set()
        for macro, cats in live_tax.items():
            for cat in cats:
                self.assertNotIn(cat, seen_cats, f"Duplicate category '{cat}' found in macro '{macro}'!")
                seen_cats.add(cat)

        self.assertIn("Fresh Bakery & Pastry", live_tax["Bakery & Bread"])
        self.assertNotIn("Fresh Bakery & Pastry", live_tax["Fresh Food"])

    def test_consume_completed_reviews_bilingual_tolerance(self):
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Create review_queue.csv with items testing bilingual normalization:
        # 1. Romanian input "Lactate & Ouă" -> auto-normalized to "Dairy & Eggs", macro "Fresh Food"
        # 2. English input "Fresh Meat & Fish" -> valid as-is, macro auto-filled to "Fresh Food"
        # 3. Invalid category "Categoria Inexistenta" -> kept in review queue
        # 4. User set non_food = TRUE -> macro "Non-Food", category "Non-Food"
        # 5. Unreviewed item -> kept in review queue
        queue_items = [
            {"product_key": "4056489000001", "sample_name": "Lapte", "category": "Lactate & Ouă"},
            {"product_key": "4056489000002", "sample_name": "Piept pui", "category": "Fresh Meat & Fish"},
            {"product_key": "4056489000003", "sample_name": "Produs Ciudat", "category": "Categoria Inexistenta"},
            {"product_key": "4056489000004", "sample_name": "Baterii AA", "non_food": "TRUE"},
            {"product_key": "4056489000005", "sample_name": "Neclasificat"}
        ]
        sync.write_csv_records(sync.REVIEW_QUEUE_PATH, queue_items, sync.OVERRIDE_COLUMNS)

        sync.consume_completed_reviews(con)

        # 1. Overrides should have items 1, 2, and 4 in canonical English
        ovr_rows, _, _ = sync.read_csv_records(sync.OVERRIDE_CSV_PATH)
        ovr_keys = {r["product_key"]: r for r in ovr_rows}
        
        self.assertIn("4056489000001", ovr_keys)
        self.assertEqual(ovr_keys["4056489000001"]["macro_category"], "Fresh Food")
        self.assertEqual(ovr_keys["4056489000001"]["category"], "Dairy & Eggs")

        self.assertIn("4056489000002", ovr_keys)
        self.assertEqual(ovr_keys["4056489000002"]["macro_category"], "Fresh Food")
        self.assertEqual(ovr_keys["4056489000002"]["category"], "Fresh Meat & Fish")

        self.assertIn("4056489000004", ovr_keys)
        self.assertEqual(ovr_keys["4056489000004"]["macro_category"], "Non-Food")
        self.assertEqual(ovr_keys["4056489000004"]["category"], "Non-Food")

        # 2. Review queue should still have items 3 (invalid) and 5 (unreviewed)
        remaining_rows, _, _ = sync.read_csv_records(sync.REVIEW_QUEUE_PATH)
        remaining_keys = {r["product_key"] for r in remaining_rows}
        self.assertIn("4056489000003", remaining_keys)
        self.assertIn("4056489000005", remaining_keys)
        self.assertNotIn("4056489000001", remaining_keys)
        self.assertNotIn("4056489000002", remaining_keys)
        self.assertNotIn("4056489000004", remaining_keys)

        con.close()

    def test_apply_seed_catalog_smart_upsert_and_cross_border(self):
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Pre-insert existing item into nutrition_cache with MANUAL_CSV data_source
        con.execute("""
            INSERT INTO nutrition_cache (
                product_key, sample_name, macro_category, category, kcal_100g, data_source, is_verified
            ) VALUES (
                '4056489111111', 'Piept pui', 'Fresh Food', 'Pui Special', 120.0, 'MANUAL_CSV', TRUE
            );
        """)

        # Item 3: in nutrition_cache from purchases but missing category/nutrition
        con.execute("""
            INSERT INTO nutrition_cache (
                product_key, sample_name, macro_category, category, kcal_100g, data_source, is_verified
            ) VALUES (
                '4056489333333', 'Iaurt Milbona', NULL, NULL, NULL, 'BON_ONLY', FALSE
            );
        """)

        # 1. Local seed catalog:
        # Item 1: Same product_key '4056489111111' but different category (should NOT overwrite MANUAL_CSV)
        # Item 2: New product '4056489222222' (should be inserted)
        local_seed_items = [
            {
                "product_key": "4056489111111",
                "sample_name": "Piept pui",
                "macro_category": "Fresh Food",
                "category": "Fresh Meat & Fish",
                "kcal_100g": "110.0"
            },
            {
                "product_key": "4056489222222",
                "sample_name": "Ciocolată Lapte",
                "macro_category": "Sweets & Snacks",
                "category": "Chocolate & Confectionery",
                "kcal_100g": "530.0"
            }
        ]
        sync.write_csv_records(sync.SEED_CATALOG_PATH, local_seed_items, sync.OVERRIDE_COLUMNS)

        # 2. Sibling cross-border seed catalog (e.g. data/catalogs/fr/catalog_seed.csv):
        # Contains Item 3 ('4056489333333') with verified category and nutrition
        fr_dir = sync.CATALOGS_BASE_DIR / "fr"
        fr_dir.mkdir(parents=True, exist_ok=True)
        fr_seed_path = fr_dir / "catalog_seed.csv"
        fr_seed_items = [
            {
                "product_key": "4056489333333",
                "sample_name": "Yaourt Milbona",
                "macro_category": "Fresh Food",
                "category": "Dairy & Eggs",
                "kcal_100g": "65.0"
            }
        ]
        sync.write_csv_records(fr_seed_path, fr_seed_items, sync.OVERRIDE_COLUMNS)

        sync.apply_seed_catalog_to_db(con)

        # Item 1 should preserve manual override
        row1 = con.execute("SELECT category, data_source FROM nutrition_cache WHERE product_key = '4056489111111'").fetchone()
        self.assertEqual(row1[0], "Pui Special")
        self.assertEqual(row1[1], "MANUAL_CSV")

        # Item 2 should be newly added with SEED_CATALOG source
        row2 = con.execute("SELECT category, data_source FROM nutrition_cache WHERE product_key = '4056489222222'").fetchone()
        self.assertEqual(row2[0], "Chocolate & Confectionery")
        self.assertEqual(row2[1], "SEED_CATALOG")

        # Item 3 should be enriched via Cross-Border pooling
        row3 = con.execute("SELECT category, kcal_100g, match_status FROM nutrition_cache WHERE product_key = '4056489333333'").fetchone()
        self.assertEqual(row3[0], "Dairy & Eggs")
        self.assertEqual(row3[1], 65.0)
        self.assertEqual(row3[2], "SEED_CROSS_BORDER")

        con.close()

    def test_romanian_taxonomy_idempotent_migration(self):
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE purchases (
                ticket_id VARCHAR, line_idx INTEGER, raw_code VARCHAR, barcode_type VARCHAR, ean VARCHAR
            );
            CREATE TABLE nutrition_cache (
                product_key VARCHAR PRIMARY KEY,
                sample_name VARCHAR,
                macro_category VARCHAR,
                category VARCHAR,
                kcal_100g DOUBLE,
                is_verified BOOLEAN DEFAULT FALSE,
                match_status VARCHAR
            );
        """)

        # Insert historical Romanian categories
        con.execute("""
            INSERT INTO nutrition_cache (product_key, macro_category, category) VALUES
                ('P1', 'Alimente Proaspete', 'Lactate & Ouă'),
                ('P2', 'Dulciuri & Snacks', 'Ciocolată, Batoane & Dulciuri'),
                ('P3', 'Fresh Food', 'Fresh Meat & Fish');
        """)

        # Run migration
        sync.migrate_romanian_taxonomy_to_english(con)

        rows = con.execute("SELECT product_key, macro_category, category FROM nutrition_cache ORDER BY product_key").fetchall()
        self.assertEqual(rows[0], ('P1', 'Fresh Food', 'Dairy & Eggs'))
        self.assertEqual(rows[1], ('P2', 'Sweets & Snacks', 'Chocolate & Confectionery'))
        self.assertEqual(rows[2], ('P3', 'Fresh Food', 'Fresh Meat & Fish'))

        # Run migration a second time (must be idempotent 0ms no-op)
        sync.migrate_romanian_taxonomy_to_english(con)
        rows_again = con.execute("SELECT product_key, macro_category, category FROM nutrition_cache ORDER BY product_key").fetchall()
        self.assertEqual(rows, rows_again)

        con.close()

    def test_cross_border_anti_collision_and_determinism(self):
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Pre-insert products into nutrition_cache needing enrichment
        con.execute("""
            INSERT INTO nutrition_cache (product_key, sample_name, macro_category, category, kcal_100g, match_status, data_source) VALUES
                ('5474', 'Felie alune local', NULL, NULL, NULL, 'BON_ONLY', 'BON_ONLY'),
                ('2504410007454', 'Caserola carne local', NULL, NULL, NULL, 'BON_ONLY', 'BON_ONLY'),
                ('20093341', 'Lidl EAN-8 Global', NULL, NULL, NULL, 'BON_ONLY', 'BON_ONLY'),
                ('4056489000001', 'Lidl EAN-13 Global', NULL, NULL, NULL, 'BON_ONLY', 'BON_ONLY');
        """)

        # Local seed (empty)
        sync.write_csv_records(sync.SEED_CATALOG_PATH, [], sync.OVERRIDE_COLUMNS)

        # Sibling seed 1: France (fr)
        fr_dir = sync.CATALOGS_BASE_DIR / "fr"
        fr_dir.mkdir(parents=True, exist_ok=True)
        fr_seed_path = fr_dir / "catalog_seed.csv"
        fr_items = [
            {"product_key": "5474", "sample_name": "Baguette tradition FR", "macro_category": "Fresh Food", "category": "Dairy & Eggs", "kcal_100g": "250.0"},
            {"product_key": "2504410007454", "sample_name": "Viande FR", "macro_category": "Fresh Food", "category": "Fresh Meat & Fish", "kcal_100g": "150.0"},
            {"product_key": "20093341", "sample_name": "Produit EAN8 FR", "macro_category": "Sweets & Snacks", "category": "Chocolate & Confectionery", "kcal_100g": "500.0"},
            {"product_key": "4056489000001", "sample_name": "Produit EAN13 FR", "macro_category": "Fresh Food", "category": "Dairy & Eggs", "kcal_100g": ""}
        ]
        sync.write_csv_records(fr_seed_path, fr_items, sync.OVERRIDE_COLUMNS)

        # Sibling seed 2: Germany (de)
        de_dir = sync.CATALOGS_BASE_DIR / "de"
        de_dir.mkdir(parents=True, exist_ok=True)
        de_seed_path = de_dir / "catalog_seed.csv"
        de_items = [
            # Has the same EAN-13 as FR, but DE has verified nutrition!
            {"product_key": "4056489000001", "sample_name": "Produkt EAN13 DE", "macro_category": "Fresh Food", "category": "Dairy & Eggs", "kcal_100g": "65.0"}
        ]
        sync.write_csv_records(de_seed_path, de_items, sync.OVERRIDE_COLUMNS)

        sync.apply_seed_catalog_to_db(con)

        # 1. Anti-collision verification: PLU 5474 and Caserole 2504410007454 must NOT be enriched from FR!
        plu_row = con.execute("SELECT category, match_status FROM nutrition_cache WHERE product_key = '5474'").fetchone()
        self.assertIsNone(plu_row[0])
        self.assertEqual(plu_row[1], "BON_ONLY")

        caserole_row = con.execute("SELECT category, match_status FROM nutrition_cache WHERE product_key = '2504410007454'").fetchone()
        self.assertIsNone(caserole_row[0])
        self.assertEqual(caserole_row[1], "BON_ONLY")

        # 2. Global EAN-8 (20093341) must cross border!
        ean8_row = con.execute("SELECT category, kcal_100g, match_status FROM nutrition_cache WHERE product_key = '20093341'").fetchone()
        self.assertEqual(ean8_row[0], "Chocolate & Confectionery")
        self.assertEqual(ean8_row[1], 500.0)
        self.assertEqual(ean8_row[2], "SEED_CROSS_BORDER")

        # 3. Determinism: DE's verified nutrition (65.0 kcal) must win over FR's missing nutrition
        ean13_row = con.execute("SELECT category, kcal_100g, match_status FROM nutrition_cache WHERE product_key = '4056489000001'").fetchone()
        self.assertEqual(ean13_row[0], "Dairy & Eggs")
        self.assertEqual(ean13_row[1], 65.0)
        self.assertEqual(ean13_row[2], "SEED_CROSS_BORDER")

        con.close()

    def test_review_queue_category_vs_nutrition_orthogonality(self):
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Item 1: Known category (from OFF Waterfall), missing nutrition -> must enter review_queue with category pre-filled
        # Item 2: Missing category and nutrition -> must enter review_queue with empty category
        # Item 3: Non-Food item -> must NOT enter review_queue
        con.execute("""
            INSERT INTO nutrition_cache (product_key, sample_name, macro_category, category, kcal_100g, match_status) VALUES
                ('4056489000010', 'Fromage Blanc', 'Fresh Food', 'Dairy & Eggs', NULL, 'MATCHED_OFF_WATERFALL'),
                ('4056489000020', 'Produs Necunoscut', NULL, NULL, NULL, 'BON_ONLY'),
                ('4056489000030', 'Punga Hartie', 'Non-Food', 'Non-Food', NULL, 'NON_FOOD');
        """)

        # 1. Update review queue
        sync.update_review_queue(con)

        queue_rows, _, _ = sync.read_csv_records(sync.REVIEW_QUEUE_PATH)
        queue_by_pk = {r["product_key"]: r for r in queue_rows}

        self.assertIn("4056489000010", queue_by_pk)
        self.assertEqual(queue_by_pk["4056489000010"]["macro_category"], "Fresh Food")
        self.assertEqual(queue_by_pk["4056489000010"]["category"], "Dairy & Eggs")
        self.assertEqual(queue_by_pk["4056489000010"]["kcal_100g"], "")

        self.assertIn("4056489000020", queue_by_pk)
        self.assertEqual(queue_by_pk["4056489000020"]["category"], "")

        self.assertNotIn("4056489000030", queue_by_pk)

        # 2. Run consume_completed_reviews without user edits -> queue must remain unchanged (not consumed prematurely)
        sync.consume_completed_reviews(con)
        queue_rows_after, _, _ = sync.read_csv_records(sync.REVIEW_QUEUE_PATH)
        self.assertEqual(len(queue_rows_after), 2)

        # 3. User edits Item 1 by filling in kcal_100g = 75.0
        queue_by_pk["4056489000010"]["kcal_100g"] = "75.0"
        sync.write_csv_records(sync.REVIEW_QUEUE_PATH, list(queue_by_pk.values()), sync.OVERRIDE_COLUMNS)

        sync.consume_completed_reviews(con)

        # Item 1 must now be in catalog_overrides.csv, and Item 2 remains in review_queue.csv
        ovr_rows, _, _ = sync.read_csv_records(sync.OVERRIDE_CSV_PATH)
        ovr_by_pk = {r["product_key"]: r for r in ovr_rows}
        self.assertIn("4056489000010", ovr_by_pk)
        # The queue's prefilled category was not edited; archive only the
        # explicitly changed nutrient field to avoid false MANUAL provenance.
        self.assertEqual(ovr_by_pk["4056489000010"]["category"], "")
        self.assertEqual(ovr_by_pk["4056489000010"]["kcal_100g"], "75.0")

        remaining_queue, _, _ = sync.read_csv_records(sync.REVIEW_QUEUE_PATH)
        self.assertEqual(len(remaining_queue), 1)
        self.assertEqual(remaining_queue[0]["product_key"], "4056489000020")

        con.close()

    def test_auto_detect_country_from_mock_jwt(self):
        import base64
        claims = {"address_country": "DE", "email": "test@example.de"}
        p_b64 = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        mock_id_token = f"header.{p_b64}.signature"

        class MockResponse:
            def json(self):
                return {
                    "access_token": "mock_access_token",
                    "refresh_token": "mock_refresh_token",
                    "expires_in": 3600,
                    "id_token": mock_id_token
                }

        orig_post = sync.requests.post
        try:
            sync.requests.post = lambda *args, **kwargs: MockResponse()
            api = sync.AutoDetectLidlPlusApi(refresh_token="initial_refresh")
            self.assertEqual(api.detected_country, "DE")
            self.assertEqual(api._country, "DE")

            # Configure country dynamically
            sync.configure_country("DE")
            self.assertEqual(sync.LIDL_COUNTRY, "de")
            self.assertEqual(sync.COUNTRY_CATALOG_DIR, sync.CATALOGS_BASE_DIR / "de")
            self.assertEqual(sync.SEED_CATALOG_PATH, sync.CATALOGS_BASE_DIR / "de" / "catalog_seed.csv")
        finally:
            sync.requests.post = orig_post
            sync.configure_country("ro")

    def test_update_env_country_atomic(self):
        import os
        env_file = self.test_path / ".env"
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("LIDL_REFRESH_TOKEN=some_token\n")

        orig_cwd = os.getcwd()
        try:
            os.chdir(self.test_path)
            # 1. Append country
            sync.update_env_country("FR")
            with open(env_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("LIDL_COUNTRY=fr", content)
            self.assertIn("LIDL_REFRESH_TOKEN=some_token", content)

            # 2. Update country to DE (must not duplicate)
            sync.update_env_country("DE")
            with open(env_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertIn("LIDL_COUNTRY=de", lines)
            self.assertNotIn("LIDL_COUNTRY=fr", lines)
            self.assertEqual(len([l for l in lines if l.startswith("LIDL_COUNTRY=")]), 1)
        finally:
            os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
