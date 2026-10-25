import unittest
import sys
from pathlib import Path
import duckdb

# Add project root to sys.path so we can import from sync_incremental
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sync_incremental import is_valid_gs1, clean_product_key, extract_clean_ean, migrate_legacy_55_barcodes


class TestBarcodeExtract(unittest.TestCase):
    def test_gs1_ean13_validation(self):
        # Valid EAN-13 codes
        self.assertTrue(is_valid_gs1("4056489577980"))  # DE Lidl Nori Sheets
        self.assertTrue(is_valid_gs1("4056489293057"))  # DE Lidl Chicken
        self.assertTrue(is_valid_gs1("5941872103060"))  # RO 13-digit EAN

        # Invalid EAN-13 check digit
        self.assertFalse(is_valid_gs1("4056489577981"))
        self.assertFalse(is_valid_gs1("4056489293058"))

        # Invalid lengths and formats
        self.assertFalse(is_valid_gs1("12345"))
        self.assertFalse(is_valid_gs1("405648957798a"))
        self.assertFalse(is_valid_gs1(""))

    def test_gs1_ean8_validation(self):
        # Valid EAN-8 codes from Lidl receipts
        self.assertTrue(is_valid_gs1("40881959"))  # Cotton Candy (Oaklands)
        self.assertTrue(is_valid_gs1("20005924"))  # Mini Choko Softies (Mister Choc)
        self.assertTrue(is_valid_gs1("20430153"))  # Salam secuiesc sort. 2
        self.assertTrue(is_valid_gs1("96385074"))  # Standard GS1 EAN-8 test

        # Invalid EAN-8 / store PLU failing GS1 checksum
        self.assertFalse(is_valid_gs1("00119131"))  # Internal store code
        self.assertFalse(is_valid_gs1("40881950"))  # Wrong check digit

    def test_clean_product_key(self):
        # Clean numeric string
        self.assertEqual(clean_product_key("4056489577980"), "4056489577980")
        self.assertEqual(clean_product_key("20430153"), "20430153")

        # Excel formula wrapping
        self.assertEqual(clean_product_key('="4056489577980"'), "4056489577980")
        self.assertEqual(clean_product_key("='20430153'"), "20430153")
        self.assertEqual(clean_product_key('=20430153'), "20430153")

        # Plain quotes and whitespace
        self.assertEqual(clean_product_key('  "4056489577980"  '), "4056489577980")
        self.assertEqual(clean_product_key(" '20430153' "), "20430153")

        # Empty / None
        self.assertEqual(clean_product_key(None), "")
        self.assertEqual(clean_product_key(""), "")

    def test_extract_clean_ean_15_digits_55_prefix(self):
        # Real 15-digit Lidl code masking a 13-digit EAN
        b_type, ean = extract_clean_ean("554056489577980")
        self.assertEqual(b_type, "GLOBAL_EAN")
        self.assertEqual(ean, "4056489577980")

        # 15-digit starting with 55 but invalid check digit
        b_type, ean = extract_clean_ean("554056489577981")
        self.assertEqual(b_type, "OTHER")
        self.assertIsNone(ean)

        # 15-digit starting with 5500000 masking an 8-digit GS1 EAN-8
        b_type, ean = extract_clean_ean("550000040881959")
        self.assertEqual(b_type, "GLOBAL_EAN")
        self.assertEqual(ean, "40881959")

        b_type, ean = extract_clean_ean("550000020430153")
        self.assertEqual(b_type, "GLOBAL_EAN")
        self.assertEqual(ean, "20430153")

        # 15-digit starting with 5500000 masking an 8-digit internal store PLU (not valid GS1)
        # Extracts clean 8-digit PLU candidate, marked as PLU_SHORT so downstream OFF lookup is filtered out
        b_type, ean = extract_clean_ean("550000000119131")
        self.assertEqual(b_type, "PLU_SHORT")
        self.assertEqual(ean, "00119131")

    def test_extract_clean_ean_standard_barcodes(self):
        # Direct 13-digit EAN
        b_type, ean = extract_clean_ean("4056489577980")
        self.assertEqual(b_type, "GLOBAL_EAN")
        self.assertEqual(ean, "4056489577980")

        # Direct 8-digit EAN-8
        b_type, ean = extract_clean_ean("40881959")
        self.assertEqual(b_type, "GLOBAL_EAN")
        self.assertEqual(ean, "40881959")

        # Direct 8-digit store PLU (failing GS1 -> returns clean code with PLU_SHORT)
        b_type, ean = extract_clean_ean("00119131")
        self.assertEqual(b_type, "PLU_SHORT")
        self.assertEqual(ean, "00119131")

        # Internal weighted barcode (starts with 2)
        b_type, ean = extract_clean_ean("2508941003817")
        self.assertEqual(b_type, "INTERNAL_WEIGHTED")
        self.assertIsNone(ean)

        b_type, ean = extract_clean_ean("2012345678901")
        self.assertEqual(b_type, "INTERNAL_WEIGHTED")
        self.assertIsNone(ean)

        # Short PLU
        b_type, ean = extract_clean_ean("5012")
        self.assertEqual(b_type, "PLU_SHORT")
        self.assertIsNone(ean)

        # Invalid non-numeric
        b_type, ean = extract_clean_ean("ABC123")
        self.assertEqual(b_type, "INVALID")
        self.assertIsNone(ean)

    def test_migration_idempotency_and_acid(self):
        # In-memory database to test migration and 0ms idempotency
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE purchases (
                ticket_id VARCHAR,
                line_idx INTEGER,
                raw_code VARCHAR,
                barcode_type VARCHAR,
                ean VARCHAR
            );
            CREATE TABLE nutrition_cache (
                product_key VARCHAR PRIMARY KEY,
                ean VARCHAR
            );
        """)

        # Insert 4 legacy purchases:
        # 1. 15-digit valid EAN-13
        # 2. 15-digit valid EAN-8
        # 3. 15-digit invalid/unparseable barcode
        # 4. 15-digit internal PLU (5500000 prefix)
        con.execute("""
            INSERT INTO purchases VALUES
                ('T1', 1, '554056489577980', NULL, NULL),
                ('T2', 1, '550000040881959', NULL, NULL),
                ('T3', 1, '559999999999999', NULL, NULL),
                ('T4', 1, '550000000119131', NULL, NULL);
        """)

        # Insert corresponding legacy/phantom cache entries
        con.execute("""
            INSERT INTO nutrition_cache VALUES
                ('554056489577980', '554056489577980'),
                ('0000040881959', '0000040881959'),
                ('0000000119131', '0000000119131');
        """)

        # First run: should migrate purchases and rename cache keys
        migrate_legacy_55_barcodes(con)

        rows = con.execute("SELECT raw_code, barcode_type, ean FROM purchases ORDER BY ticket_id").fetchall()
        self.assertEqual(rows[0], ('554056489577980', 'GLOBAL_EAN', '4056489577980'))
        self.assertEqual(rows[1], ('550000040881959', 'GLOBAL_EAN', '40881959'))
        # Unparseable code must be marked as OTHER with ean = '' so it's never re-queried
        self.assertEqual(rows[2], ('559999999999999', 'OTHER', ''))
        # Legacy PLU code must be migrated to clean 8-digit PLU_SHORT
        self.assertEqual(rows[3], ('550000000119131', 'PLU_SHORT', '00119131'))

        cache_keys = set(r[0] for r in con.execute("SELECT product_key FROM nutrition_cache").fetchall())
        self.assertIn("4056489577980", cache_keys)
        self.assertIn("40881959", cache_keys)
        self.assertIn("00119131", cache_keys)
        self.assertNotIn("554056489577980", cache_keys)
        self.assertNotIn("0000040881959", cache_keys)
        self.assertNotIn("0000000119131", cache_keys)

        # Second run: strictly idempotent (0 rows matched by startup query)
        unmigrated_count = con.execute("""
            SELECT COUNT(*) 
            FROM purchases 
            WHERE raw_code LIKE '55%' 
              AND LENGTH(raw_code) = 15 
              AND (ean IS NULL OR ean LIKE '55%' OR ean LIKE '00000%')
        """).fetchone()[0]
        self.assertEqual(unmigrated_count, 0)

        # Calling migration again should be a clean 0ms no-op without error
        migrate_legacy_55_barcodes(con)
        con.close()


if __name__ == "__main__":
    unittest.main()
