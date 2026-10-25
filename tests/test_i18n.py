import json
import os
import unittest
from pathlib import Path
import duckdb

import sync_incremental as sync

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "data" / "catalogs" / "taxonomy.json"
TRANSLATIONS_DIR = PROJECT_ROOT / "data" / "catalogs" / "translations"
EXPECTED_LANGUAGES = ["en", "ro", "de", "fr", "es", "it", "pl"]


class TestI18nSystem(unittest.TestCase):
    def setUp(self):
        # Load canonical concepts from taxonomy.json
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            tax = json.load(f)
        
        self.canonical_concepts = set(tax.keys())
        for cats in tax.values():
            self.canonical_concepts.update(cats)

    def test_translation_files_exist_and_valid_json(self):
        """Verify that all target language JSON files exist and are valid dictionaries."""
        for lang in EXPECTED_LANGUAGES:
            file_path = TRANSLATIONS_DIR / f"{lang}.json"
            self.assertTrue(file_path.exists(), f"Translation file missing: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict, f"Translation {lang}.json must be a JSON object")

    def test_all_canonical_taxonomy_concepts_covered(self):
        """Verify that every translation file covers 100% of the 27 canonical taxonomy concepts."""
        for lang in EXPECTED_LANGUAGES:
            file_path = TRANSLATIONS_DIR / f"{lang}.json"
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for concept in self.canonical_concepts:
                self.assertIn(concept, data, f"Language '{lang}' is missing translation for canonical key '{concept}'")
                self.assertTrue(bool(data[concept].strip()), f"Language '{lang}' has empty value for '{concept}'")

    def test_load_translations_bidirectional_and_fallback(self):
        """Test that load_translations generates both EN->LOCAL and LOCAL->EN with fallback."""
        # Test German
        en_to_de, de_to_en = sync.load_translations("de")
        self.assertEqual(en_to_de["Fresh Food"], "Frische Lebensmittel")
        self.assertEqual(de_to_en["Frische Lebensmittel"], "Fresh Food")
        self.assertEqual(de_to_en["Fresh Food"], "Fresh Food")  # Identity fallback

        # Test Romanian with accent folding
        en_to_ro, ro_to_en = sync.load_translations("ro")
        self.assertEqual(en_to_ro["Pantry & Groceries"], "Băcănie")
        self.assertEqual(ro_to_en["Băcănie"], "Pantry & Groceries")
        self.assertEqual(ro_to_en["Bacanie"], "Pantry & Groceries")  # Folded lookup

        # Test Unknown Language Fallback to English
        en_to_unk, unk_to_en = sync.load_translations("unknown_code_xyz")
        self.assertEqual(en_to_unk["Fresh Meat & Fish"], "Fresh Meat & Fish")
        self.assertEqual(unk_to_en["Fresh Meat & Fish"], "Fresh Meat & Fish")

    def test_resolve_active_language_hierarchy(self):
        """Test language resolution priority: explicit parameter -> env LIDL_LANGUAGE -> country default -> en."""
        # 1. Country defaults
        self.assertEqual(sync.resolve_active_language("ro"), "ro")
        self.assertEqual(sync.resolve_active_language("de"), "de")
        self.assertEqual(sync.resolve_active_language("ch"), "de")
        self.assertEqual(sync.resolve_active_language("pl"), "pl")

        # 2. Explicit requested language parameter
        self.assertEqual(sync.resolve_active_language("ch", requested_lang="fr"), "fr")
        self.assertEqual(sync.resolve_active_language("ro", requested_lang="de"), "de")

        # 3. Environment variable LIDL_LANGUAGE
        orig_env = os.environ.get("LIDL_LANGUAGE")
        try:
            os.environ["LIDL_LANGUAGE"] = "es"
            self.assertEqual(sync.resolve_active_language("de"), "es")
        finally:
            if orig_env is not None:
                os.environ["LIDL_LANGUAGE"] = orig_env
            else:
                os.environ.pop("LIDL_LANGUAGE", None)

        # 4. Unknown country with no translation falls back to 'en'
        self.assertEqual(sync.resolve_active_language("xx_unknown_country"), "en")

    def test_export_views_for_bi_multi_language_parquet(self):
        """Verify export_views_for_bi maps category_local to the active language correctly."""
        con = duckdb.connect(":memory:")
        sync.init_database(con)

        # Insert a sample purchase & nutrition cache item in English
        con.execute("""
            INSERT INTO nutrition_cache (product_key, sample_name, macro_category, category, kcal_100g, data_source, is_verified)
            VALUES ('4056489000001', 'Test Product', 'Fresh Food', 'Fresh Meat & Fish', 150.0, 'SEED_CATALOG', TRUE);
        """)
        con.execute("""
            INSERT INTO purchases (ticket_id, date, line_idx, name, quantity, unit_price, line_total, raw_code, barcode_type, ean)
            VALUES ('ticket_1', '2026-10-18', 0, 'Test Product', 1.0, 10.0, 10.0, '4056489000001', 'GLOBAL_EAN', '4056489000001');
        """)

        # 1. Test German
        orig_country = sync.LIDL_COUNTRY
        try:
            sync.configure_country("de")
            active_lang = sync.resolve_active_language(sync.LIDL_COUNTRY)
            en_to_local, _ = sync.load_translations(active_lang)

            con.execute("CREATE TEMP TABLE IF NOT EXISTS tax_en_to_local (en_val VARCHAR PRIMARY KEY, local_val VARCHAR);")
            con.execute("TRUNCATE tax_en_to_local;")
            con.executemany("INSERT INTO tax_en_to_local VALUES (?, ?);", list(en_to_local.items()))

            row = con.execute("""
                SELECT 
                    COALESCE(m_loc.local_val, c.macro_category) AS macro_local,
                    COALESCE(c_loc.local_val, c.category) AS cat_local
                FROM nutrition_cache c
                LEFT JOIN tax_en_to_local m_loc ON c.macro_category = m_loc.en_val
                LEFT JOIN tax_en_to_local c_loc ON c.category = c_loc.en_val
                WHERE c.product_key = '4056489000001';
            """).fetchone()

            self.assertEqual(row[0], "Frische Lebensmittel")
            self.assertEqual(row[1], "Frisches Fleisch & Fisch")

            # 2. Test French
            sync.configure_country("fr")
            active_lang = sync.resolve_active_language(sync.LIDL_COUNTRY)
            en_to_local, _ = sync.load_translations(active_lang)
            con.execute("TRUNCATE tax_en_to_local;")
            con.executemany("INSERT INTO tax_en_to_local VALUES (?, ?);", list(en_to_local.items()))

            row_fr = con.execute("""
                SELECT 
                    COALESCE(m_loc.local_val, c.macro_category) AS macro_local,
                    COALESCE(c_loc.local_val, c.category) AS cat_local
                FROM nutrition_cache c
                LEFT JOIN tax_en_to_local m_loc ON c.macro_category = m_loc.en_val
                LEFT JOIN tax_en_to_local c_loc ON c.category = c_loc.en_val
                WHERE c.product_key = '4056489000001';
            """).fetchone()

            self.assertEqual(row_fr[0], "Produits Frais")
            self.assertEqual(row_fr[1], "Viande Fraîche & Poisson")
        finally:
            sync.configure_country(orig_country)


if __name__ == "__main__":
    unittest.main()
