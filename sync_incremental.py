import base64
import csv
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import io
import json
import os
import re
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Callable
import duckdb
import requests
import unicodedata
from lidlplus_api import LidlPlusApi

# Ensure UTF-8 console output on Windows terminals
if sys.platform == "win32":
    try:
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RAW_DIR = Path("data/raw_tickets")

DB_PATH = "lidl_consumption.duckdb"
PARQUET_OFF_PATH = "food.parquet"
OVERRIDE_CSV_PATH = Path("catalog_overrides.csv")
REVIEW_QUEUE_PATH = Path("review_queue.csv")
EXPORT_TRANSACTIONS_PATH = "receipts.parquet"
EXPORT_CATALOG_PATH = "products.parquet"

OFF_MODE = os.getenv("OFF_MODE", "auto").lower().strip()
OFF_NOT_FOUND_TTL_DAYS = int(os.getenv("OFF_NOT_FOUND_TTL_DAYS", "30"))
OFF_USER_AGENT = os.getenv("OFF_USER_AGENT", "LidlConsumptionPipeline/1.0 (https://github.com/lidl-consumption-pipeline)")

LIDL_COUNTRY = os.getenv("LIDL_COUNTRY", "ro").lower().strip()
CATALOGS_BASE_DIR = Path("data/catalogs")
COUNTRY_CATALOG_DIR = CATALOGS_BASE_DIR / LIDL_COUNTRY
UNIVERSAL_TAXONOMY_PATH = CATALOGS_BASE_DIR / "taxonomy.json"
TAXONOMY_PATH = UNIVERSAL_TAXONOMY_PATH if UNIVERSAL_TAXONOMY_PATH.exists() else (COUNTRY_CATALOG_DIR / "taxonomy.json")
SEED_CATALOG_PATH = COUNTRY_CATALOG_DIR / "catalog_seed.csv"


@dataclass(frozen=True)
class ReceiptSyncResult:
    """Outcome counts for incremental receipt retrieval."""

    downloaded: int
    failed_pages: int
    failed_receipts: int


def configure_country(country_code: str) -> None:
    """Dynamically configure active country and catalog paths."""
    global LIDL_COUNTRY, COUNTRY_CATALOG_DIR, TAXONOMY_PATH, SEED_CATALOG_PATH
    LIDL_COUNTRY = country_code.lower().strip()
    COUNTRY_CATALOG_DIR = CATALOGS_BASE_DIR / LIDL_COUNTRY
    TAXONOMY_PATH = UNIVERSAL_TAXONOMY_PATH if UNIVERSAL_TAXONOMY_PATH.exists() else (COUNTRY_CATALOG_DIR / "taxonomy.json")
    SEED_CATALOG_PATH = COUNTRY_CATALOG_DIR / "catalog_seed.csv"

# =====================================================================
# MODULAR INTERNATIONALIZATION (i18n) ENGINE
# =====================================================================
TRANSLATIONS_BASE_DIR = Path("data/catalogs/translations")
TRANSLATIONS_DIR = CATALOGS_BASE_DIR / "translations"

COUNTRY_DEFAULT_LANGUAGE = {
    "ro": "ro", "de": "de", "at": "de", "ch": "de",
    "fr": "fr", "it": "it", "es": "es", "pt": "pt",
    "nl": "nl", "be": "nl", "pl": "pl", "cz": "cs",
    "sk": "sk", "si": "sl", "hr": "hr", "hu": "hu",
    "bg": "bg", "gr": "el", "cy": "el", "lt": "lt",
    "lv": "lv", "ee": "et", "fi": "fi", "se": "sv",
    "dk": "da", "ie": "en", "gb": "en", "rs": "sr",
    "mt": "mt", "tr": "tr"
}

_TRANSLATIONS_CACHE: dict[str, tuple[dict[str, str], dict[str, str]]] = {}


def fold_accents(text: str) -> str:
    """Normalize and strip diacritical marks for tolerant keyword lookup."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def resolve_active_language(country_code: str, requested_lang: str | None = None) -> str:
    """Resolve the active language code based on user override, country default, or English fallback."""
    if requested_lang and requested_lang.strip():
        lang = requested_lang.lower().strip()
        if (TRANSLATIONS_DIR / f"{lang}.json").exists() or (TRANSLATIONS_BASE_DIR / f"{lang}.json").exists():
            return lang

    env_lang = os.getenv("LIDL_LANGUAGE")
    if env_lang and env_lang.strip():
        lang = env_lang.lower().strip()
        if (TRANSLATIONS_DIR / f"{lang}.json").exists() or (TRANSLATIONS_BASE_DIR / f"{lang}.json").exists():
            return lang

    c = country_code.lower().strip()
    if (TRANSLATIONS_DIR / f"{c}.json").exists() or (TRANSLATIONS_BASE_DIR / f"{c}.json").exists():
        return c

    default_lang = COUNTRY_DEFAULT_LANGUAGE.get(c, "en")
    if (TRANSLATIONS_DIR / f"{default_lang}.json").exists() or (TRANSLATIONS_BASE_DIR / f"{default_lang}.json").exists():
        return default_lang

    return "en"


def load_translations(lang_code: str) -> tuple[dict[str, str], dict[str, str]]:
    """Load bidirectional translation dictionaries (EN -> LOCAL and LOCAL -> EN) for a given language code.

    Returns (en_to_local, local_to_en).
    Guarantees English fallback for missing keys and includes accent-tolerant lookups.
    """
    clean_code = (lang_code or "en").lower().strip()
    if clean_code in _TRANSLATIONS_CACHE:
        return _TRANSLATIONS_CACHE[clean_code]

    target_file = None
    for candidate in [
        TRANSLATIONS_DIR / f"{clean_code}.json",
        TRANSLATIONS_BASE_DIR / f"{clean_code}.json",
        TRANSLATIONS_DIR / "en.json",
        TRANSLATIONS_BASE_DIR / "en.json"
    ]:
        if candidate.exists():
            target_file = candidate
            break

    data = {}
    if target_file and target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Warning reading translation file '{target_file}': {e}")

    en_to_local = dict(data)
    local_to_en = {}

    for en_key, loc_val in en_to_local.items():
        if loc_val:
            local_to_en[loc_val] = en_key
            folded = fold_accents(loc_val)
            if folded not in local_to_en:
                local_to_en[folded] = en_key
        if en_key not in local_to_en:
            local_to_en[en_key] = en_key

    _TRANSLATIONS_CACHE[clean_code] = (en_to_local, local_to_en)
    return en_to_local, local_to_en


def __getattr__(name: str):
    if name == "RO_TO_EN_TAXONOMY":
        return load_translations("ro")[1]
    if name == "EN_TO_RO_TAXONOMY":
        return load_translations("ro")[0]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


OVERRIDE_TEXT_COLUMNS = ["macro_category", "category"]
OVERRIDE_NUM_COLUMNS = [
    "package_weight_g",
    "kcal_100g",
    "proteins_100g",
    "carbs_100g",
    "sugars_100g",
    "fat_100g",
    "saturated_fat_100g",
    "fiber_100g",
    "salt_100g",
]
# Informative columns (written by script, ignored on read): helps prioritization
OVERRIDE_INFO_COLUMNS = ["times_bought", "match_status", "off_product_name", "atwater_debug"]
OVERRIDE_COLUMNS = (
    ["product_key", "sample_name"]
    + OVERRIDE_TEXT_COLUMNS
    + OVERRIDE_NUM_COLUMNS
    + ["non_food"]
    + OVERRIDE_INFO_COLUMNS
)
SCI_NOTATION_RE = re.compile(r"^\d+(\.\d+)?e[+-]?\d+$", re.IGNORECASE)
TRUTHY = ("1", "x", "yes", "y", "true", "da", "adevarat")


# =====================================================================
# 1. CONFIGURATION & NUMERIC PARSING
# =====================================================================
def load_lidl_token() -> str:
    """Load the Lidl Plus refresh token from the local .env file."""
    if not os.path.exists(".env"):
        print("\n" + "=" * 70)
        print("[!] CONFIGURATION ERROR: The '.env' file is missing!")
        print("=" * 70)
        print("To set up your Lidl Plus authentication:")
        print("  Option 1 (Windows 1-Click): Double-click 'run.bat'")
        print("  Option 2 (Terminal CLI):    Run 'python auth_login.py'")
        print("=" * 70 + "\n")
        raise FileNotFoundError("The .env file is missing. Run 'auth_login.py' or 'run.bat' to generate it.")

    lidl_token = None
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line_clean = line.strip()
            if line_clean.startswith("LIDL_REFRESH_TOKEN="):
                lidl_token = line_clean.split("=", 1)[1].strip()

    if not lidl_token:
        print("\n" + "=" * 70)
        print("[!] CONFIGURATION ERROR: 'LIDL_REFRESH_TOKEN' is missing or empty in '.env'!")
        print("=" * 70)
        print("Please run 'python auth_login.py' (or 'run.bat') to authenticate your account.")
        print("=" * 70 + "\n")
        raise ValueError("LIDL_REFRESH_TOKEN is missing from .env. Run 'auth_login.py' to generate it.")

    return lidl_token


def update_env_token(token: str) -> None:
    """Atomically update LIDL_REFRESH_TOKEN in .env to prevent file corruption."""
    lines = []
    token_updated = False
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            lines = f.readlines()

    # Write to a temporary file first (atomic write pattern)
    tmp_env_path = ".env.tmp"
    with open(tmp_env_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("LIDL_REFRESH_TOKEN="):
                f.write(f"LIDL_REFRESH_TOKEN={token}\n")
                token_updated = True
            else:
                f.write(line)
        if not token_updated:
            f.write(f"LIDL_REFRESH_TOKEN={token}\n")

    # Replace the target file atomically once writing succeeds
    os.replace(tmp_env_path, ".env")
    print(f"[+] Token safely updated (atomic write) in .env ({len(token)} chars).")


def update_env_country(country: str) -> None:
    """Atomically append or update LIDL_COUNTRY in .env."""
    lines = []
    country_updated = False
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            lines = f.readlines()

    tmp_env_path = ".env.tmp"
    with open(tmp_env_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("LIDL_COUNTRY="):
                f.write(f"LIDL_COUNTRY={country.lower()}\n")
                country_updated = True
            else:
                f.write(line)
        if not country_updated:
            if lines and not lines[-1].endswith("\n"):
                f.write("\n")
            f.write(f"LIDL_COUNTRY={country.lower()}\n")

    os.replace(tmp_env_path, ".env")
    print(f"[+] Saved LIDL_COUNTRY={country.lower()} (atomic write) to .env.")


class AutoDetectLidlPlusApi(LidlPlusApi):
    """LidlPlusApi subclass with automatic account country detection from OAuth2 JWT id_token."""

    def __init__(self, refresh_token: str, country: str | None = None, language: str | None = None):
        self.detected_country = None
        self.detected_id_token = None
        super().__init__(
            language=(language or "ro"),
            country=(country or "RO"),
            refresh_token=refresh_token
        )
        if self._refresh_token and not self._token:
            try:
                self._renew_token()
            except Exception as e:
                print(f"[!] Warning during initial token renewal: {e}")

    def _default_headers(self, requiresCountryAndStoreID=False, store_id="", iscoupons=False, ishome=False, requiresStoreIDs=False):
        if (not self._token and self._refresh_token) or (self._expires and datetime.now(timezone.utc) >= self._expires):
            self._renew_token()
        if not self._token:
            raise PermissionError("Lidl Plus authentication error: Missing or invalid access token. Please re-run auth_login.py.")
        return {
            "Authorization": f"Bearer {self._token}",
            "App-Version": "16.43.4",
            "Operating-System": "Android",
            "App": "com.lidl.eci.lidlplus",
            "Accept-Language": self._language,
            "User-Agent": "okhttp/5.3.2",
            "OS-Version": "16",
        }

    def _auth(self, payload):
        default_secret = base64.b64encode(b"LidlPlusNativeClient:secret").decode()
        headers = {
            "Authorization": f"Basic {default_secret}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        kwargs = {"headers": headers, "data": payload, "timeout": 15}
        for attempt in range(1, 4):
            try:
                resp = requests.post("https://accounts.lidl.com/connect/token", **kwargs)
                response = resp.json()
                if "error" in response:
                    err_msg = response.get("error_description") or response.get("error")
                    if attempt < 3:
                        time.sleep(1.0)
                        continue
                    print(f"[!] Lidl token exchange warning: {err_msg}")
                    return

                if "expires_in" in response:
                    self._expires = datetime.now(timezone.utc) + timedelta(seconds=response["expires_in"])
                if "access_token" in response:
                    self._token = response["access_token"]
                if "refresh_token" in response:
                    self._refresh_token = response["refresh_token"]

                if "id_token" in response:
                    self.detected_id_token = response["id_token"]
                    try:
                        parts = response["id_token"].split(".")
                        if len(parts) >= 2:
                            p_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
                            claims = json.loads(base64.urlsafe_b64decode(p_b64))
                            country = claims.get("address_country")
                            if country:
                                self.detected_country = country.upper()
                                self._country = self.detected_country
                    except Exception as e:
                        print(f"[!] Warning decoding id_token: {e}")
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(1.0)
                    continue
                print(f"[!] Token renewal request warning: {e}")


def parse_decimal(val, default: str = "0.00") -> Decimal:
    """Safely parse decimal values from currency or quantity strings."""
    if val is None:
        return Decimal(default)
    s = str(val).strip().replace(" ", "").replace(",", ".")
    if not s or s.lower() in ("none", "null", "nan"):
        return Decimal(default)
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match:
        return Decimal(default)
    try:
        return Decimal(match.group(0))
    except Exception:
        return Decimal(default)


def parse_optional_float(val) -> float | None:
    """Parse numeric float values, returning None on empty or invalid inputs."""
    if val is None:
        return None
    s = str(val).strip().replace(" ", "").replace(",", ".")
    if not s or s.lower() in ("none", "null", "nan", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_line_discounts(line: dict) -> Decimal:
    """Extract total line discounts from receipt item dictionary."""
    raw_discounts = line.get("discounts") or []
    total_disc = Decimal("0.00")
    if isinstance(raw_discounts, list):
        for d in raw_discounts:
            if isinstance(d, dict):
                amt = parse_decimal(d.get("amount", "0.00"))
                total_disc += abs(amt)
    return total_disc.quantize(Decimal("0.01"))


def parse_weight_from_text(text: str) -> float | None:
    """Extract package weight in grams from raw product description text.
    
    Protects against fat percentages (e.g. '12% GR', '2.8%GR', '10% GRASIME')
    being erroneously captured as gram weights.
    """
    if not text:
        return None
    t = text.lower().replace(",", ".")

    # 1. Strip explicit fat percentages: e.g. "12% gr", "2.8%gr", "10% grasime", "50% gr/su"
    t = re.sub(r'\b\d+(?:\.\d+)?\s*%\s*(?:gr\b|grasime\b|gr/su\b)?', ' ', t)

    # 2. Context cleansing: in dairy products (smantana, iaurt, lapte, etc.),
    # small 'gr' numbers (< 50) without '%' often represent fat content if there is another larger weight.
    is_dairy_context = bool(re.search(r'\b(smantana|iaurt|lapte|chefir|sana|branza|telemea|crema)\b', t))
    if is_dairy_context:
        t = re.sub(r'\b([0-4]?\d(?:\.\d+)?)\s*(?:gr\b|%)\s*(?=.*\b\d+\s*(?:kg|g|gr|l|ml|cl)\b)', ' ', t)

    m_mult = re.search(r'(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*(kg|g|l|ml)', t)
    if m_mult:
        count = float(m_mult.group(1))
        unit_val = float(m_mult.group(2))
        unit = m_mult.group(3)
        factor = 1000.0 if unit in ('kg', 'l') else 1.0
        return count * unit_val * factor

    m = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|gr|l|ml|cl)\b', t)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit in ('kg', 'l'):
            return val * 1000.0
        elif unit == 'cl':
            return val * 10.0
        else:
            return val
    return None


# =====================================================================
# 2. INCREMENTAL INGESTION (API -> RAW JSON)
# =====================================================================
def sync_tickets_incremental(
    lidl: LidlPlusApi,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> ReceiptSyncResult:
    """Fetch recent receipts from Lidl Plus API incrementally with rate limiting."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = {p.stem for p in RAW_DIR.glob("*.json")}
    page = 1
    new_downloaded = 0
    failed_pages = 0
    failed_receipts = 0

    print("\n--- STEP 1: Incremental Lidl Receipts Ingestion ---")
    print(f"[i] Receipts already present on disk: {len(existing_ids)}")

    while True:
        try:
            res = lidl.receipts(pageNumber=page, only_favorite=False)
        except Exception as e:
            failed_pages += 1
            print(f"[!] Error fetching page {page} from Lidl API: {e}")
            if progress_callback:
                progress_callback({
                    "stage": "receipt_sync",
                    "status": "progress",
                    "message": "Receipt page download failed",
                    "completed": new_downloaded,
                })
            break

        tickets = res.get("tickets", [])
        if not tickets:
            break

        page_new_count = 0
        for meta in tickets:
            # Clean ID to avoid path traversal or invalid characters on disk
            raw_id = str(meta.get("id") or "").strip()
            t_id = re.sub(r"[^a-zA-Z0-9_\-]", "", raw_id)
            if not t_id:
                continue

            ticket_file = RAW_DIR / f"{t_id}.json"

            if t_id not in existing_ids:
                try:
                    ticket_data = lidl.receipt(t_id)
                    with open(ticket_file, "w", encoding="utf-8") as f:
                        json.dump(ticket_data, f, ensure_ascii=False, indent=2)

                    existing_ids.add(t_id)
                    new_downloaded += 1
                    page_new_count += 1
                    if progress_callback:
                        progress_callback({
                            "stage": "receipt_sync",
                            "status": "progress",
                            "message": "Receipt downloaded",
                            "completed": new_downloaded,
                        })

                    # Polite rate-limiting delay between requests
                    time.sleep(0.4)
                except Exception as e:
                    failed_receipts += 1
                    print(f"[!] Error downloading a receipt: {e}")
                    if progress_callback:
                        progress_callback({
                            "stage": "receipt_sync",
                            "status": "progress",
                            "message": "Receipt download failed",
                            "completed": new_downloaded,
                        })

        print(f"Page {page}: {page_new_count} new receipts downloaded out of {len(tickets)} checked.")

        if len(tickets) < res.get("size", 10):
            break

        page += 1
        time.sleep(0.2)  # Short pause between pagination requests

    print(f"[+] Total new receipts retrieved from server: {new_downloaded}")
    return ReceiptSyncResult(new_downloaded, failed_pages, failed_receipts)


# =====================================================================
# 3. DUCKDB SCHEMA & PRODUCT REGISTRATION
# =====================================================================
def is_valid_gs1(code: str) -> bool:
    """Validate barcode using official GS1 Modulo 10 checksum (supports EAN-8, EAN-12, EAN-13, EAN-14)."""
    code_str = str(code).strip()
    if not code_str.isdigit() or len(code_str) not in (8, 12, 13, 14):
        return False
    digits = [int(d) for d in code_str]
    data_digits = digits[:-1]
    weights = [3 if i % 2 == 0 else 1 for i in range(len(data_digits))][::-1]
    checksum = sum(d * w for d, w in zip(data_digits, weights))
    check_digit = (10 - (checksum % 10)) % 10
    return check_digit == digits[-1]


def is_cross_border_eligible(code: str) -> bool:
    """Filter global barcodes eligible for cross-border pooling (exclude local store PLUs and weighed caseroles)."""
    code_str = str(code).strip()
    if not code_str.isdigit():
        return False
    if len(code_str) not in (8, 12, 13):
        return False
    if len(code_str) == 13 and code_str.startswith("2"):
        return False
    return True


def clean_product_key(raw_key: str | None) -> str:
    """Strip Excel formula wrapping (='CODE'), quotes, and whitespace from product keys."""
    if not raw_key:
        return ""
    cleaned_key = str(raw_key).strip()
    if cleaned_key.startswith("=") and len(cleaned_key) > 1:
        cleaned_key = cleaned_key[1:].strip()
    if (cleaned_key.startswith('"') and cleaned_key.endswith('"')) or (cleaned_key.startswith("'") and cleaned_key.endswith("'")):
        cleaned_key = cleaned_key[1:-1].strip()
    return cleaned_key


def extract_clean_ean(raw_code: str) -> tuple[str, str | None]:
    """Classify barcode and unpack clean EAN, stripping internal Lidl '55' / '5500000' prefixes.
    
    Applies GS1 Modulo 10 checksum validation to both 13-digit and 8-digit extracted candidates.
    If an extracted candidate does not pass GS1 checksum, returns None for clean EAN.
    """
    code_clean = str(raw_code).strip()
    if not code_clean.isdigit():
        return "INVALID", None

    # Unpack Lidl internal 15-digit barcodes starting with '55'
    if len(code_clean) == 15 and code_clean.startswith("55"):
        if code_clean.startswith("5500000"):
            # Internal 8-digit PLU zero-padded by Lidl API (e.g. 550000020430153 -> 20430153)
            candidate_8 = code_clean[7:]
            if is_valid_gs1(candidate_8):
                return "GLOBAL_EAN", candidate_8
            return "PLU_SHORT", candidate_8

        candidate_13 = code_clean[2:]
        if not candidate_13.startswith("0") and is_valid_gs1(candidate_13):
            return "GLOBAL_EAN", candidate_13
        return "OTHER", None

    if len(code_clean) == 13 and re.match(r"^2[0-9]", code_clean):
        return "INTERNAL_WEIGHTED", None
    if len(code_clean) in (8, 12, 13):
        if is_valid_gs1(code_clean):
            return "GLOBAL_EAN", code_clean
        if len(code_clean) == 8:
            return "PLU_SHORT", code_clean
        return "OTHER", None
    if len(code_clean) <= 6:
        return "PLU_SHORT", None
    return "OTHER", None


def init_database(con: duckdb.DuckDBPyConnection) -> None:
    """Initialize DuckDB relational schema for receipts and nutrition cache."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            ticket_id VARCHAR,
            date TIMESTAMP,
            line_idx INTEGER,
            name VARCHAR,
            quantity DECIMAL(12, 3),
            unit_price DECIMAL(10, 2),
            line_total DECIMAL(10, 2),
            raw_code VARCHAR,
            barcode_type VARCHAR,
            ean VARCHAR,
            discount_amount DECIMAL(10, 2) DEFAULT 0.00,
            net_line_total DECIMAL(10, 2) DEFAULT 0.00,
            PRIMARY KEY (ticket_id, line_idx)
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS nutrition_cache (
            product_key VARCHAR PRIMARY KEY,
            sample_name VARCHAR,
            ean VARCHAR,
            off_product_name VARCHAR,
            off_brands VARCHAR,
            macro_category VARCHAR,
            category VARCHAR,
            nutriscore_grade VARCHAR,
            nutriscore_score INTEGER,
            nova_group INTEGER,
            kcal_100g DOUBLE,
            proteins_100g DOUBLE,
            carbs_100g DOUBLE,
            sugars_100g DOUBLE,
            fat_100g DOUBLE,
            saturated_fat_100g DOUBLE,
            fiber_100g DOUBLE,
            salt_100g DOUBLE,
            package_weight_g DOUBLE,
            is_weighed_default BOOLEAN DEFAULT FALSE,
            match_status VARCHAR,
            data_source VARCHAR DEFAULT 'BON_ONLY',
            is_verified BOOLEAN DEFAULT FALSE,
            atwater_debug VARCHAR,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    con.execute("ALTER TABLE purchases ADD COLUMN IF NOT EXISTS discount_amount DECIMAL(10, 2) DEFAULT 0.00;")
    con.execute("ALTER TABLE purchases ADD COLUMN IF NOT EXISTS net_line_total DECIMAL(10, 2) DEFAULT 0.00;")
    con.execute("ALTER TABLE nutrition_cache ADD COLUMN IF NOT EXISTS package_weight_g DOUBLE;")
    con.execute("ALTER TABLE nutrition_cache ADD COLUMN IF NOT EXISTS is_weighed_default BOOLEAN DEFAULT FALSE;")
    con.execute("ALTER TABLE nutrition_cache ADD COLUMN IF NOT EXISTS atwater_debug VARCHAR;")
    con.execute("ALTER TABLE nutrition_cache ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE;")

    # Automatically fill sample_name from purchases if empty in nutrition_cache
    con.execute("""
        UPDATE nutrition_cache
        SET sample_name = sub.p_name
        FROM (
            SELECT 
                TRIM(CAST(COALESCE(NULLIF(TRIM(ean), ''), raw_code) AS VARCHAR)) AS p_key,
                MIN(TRIM(name)) AS p_name
            FROM purchases
            GROUP BY 1
        ) sub
        WHERE nutrition_cache.product_key = sub.p_key
          AND (nutrition_cache.sample_name IS NULL OR TRIM(nutrition_cache.sample_name) = '');
    """)

    # Clear legacy Non-Food status after a product has been reclassified. The
    # current category is authoritative; old row-level status is not.
    con.execute("""
        UPDATE nutrition_cache
        SET is_verified = FALSE,
            match_status = CASE WHEN match_status = 'NON_FOOD' THEN NULL ELSE match_status END,
            data_source = CASE WHEN data_source = 'RULE_NON_FOOD' THEN NULL ELSE data_source END
        WHERE COALESCE(category, '') != 'Non-Food'
          AND (match_status = 'NON_FOOD' OR data_source = 'RULE_NON_FOOD');
    """)

    # A partial Food row is never product-verified just because it has calories
    # (or a legacy verified status). Keep it eligible for field-level review.
    con.execute("""
        UPDATE nutrition_cache
        SET is_verified = FALSE
        WHERE is_verified = TRUE
          AND COALESCE(category, '') != 'Non-Food'
          AND (
              category IS NULL OR TRIM(category) = ''
              OR kcal_100g IS NULL
              OR proteins_100g IS NULL
              OR carbs_100g IS NULL
              OR sugars_100g IS NULL
              OR fat_100g IS NULL
              OR saturated_fat_100g IS NULL
              OR salt_100g IS NULL
          );
    """)

    migrate_legacy_55_barcodes(con)
    migrate_romanian_taxonomy_to_english(con)
    # The provenance tables must exist before manual override application so
    # field-level protected values can be preserved during this sync.
    from product_store import create_schema
    create_schema(con)


def migrate_legacy_55_barcodes(con: duckdb.DuckDBPyConnection) -> None:
    """Migrate historical 15-digit Lidl '55' prefixed barcodes to clean EAN-13 or EAN-8."""
    legacy_rows = con.execute("""
        SELECT DISTINCT raw_code 
        FROM purchases 
        WHERE raw_code LIKE '55%' 
          AND LENGTH(raw_code) = 15 
          AND (ean IS NULL OR ean LIKE '55%' OR ean LIKE '00000%');
    """).fetchall()

    if not legacy_rows:
        return

    migrated_purchases = 0
    merged_cache_deletes = 0
    updated_cache_keys = 0

    con.begin()
    try:
        existing_cache_keys = set(r[0] for r in con.execute("SELECT product_key FROM nutrition_cache").fetchall())

        for (raw_code,) in legacy_rows:
            b_type, clean_code = extract_clean_ean(raw_code)
            if not clean_code:
                # Mark unparseable legacy codes as migrated/OTHER with non-null empty ean
                # to guarantee they are never re-queried (ensuring 0ms startup no-op)
                con.execute("""
                    UPDATE purchases 
                    SET barcode_type = 'OTHER', ean = ''
                    WHERE raw_code = ?;
                """, [raw_code])
                migrated_purchases += 1
                continue

            # 1. Update purchases to link to the clean EAN / PLU
            con.execute("""
                UPDATE purchases 
                SET ean = ?, barcode_type = ?
                WHERE raw_code = ?;
            """, [clean_code, b_type, raw_code])
            migrated_purchases += 1

            # Handle phantom 00000... or raw_code keys in nutrition_cache
            phantom_key = "00000" + clean_code if raw_code.startswith("5500000") else None
            
            # Deduplicated cache cleanup across phantom_key and raw_code
            for old_key in [k for k in (phantom_key, raw_code) if k and k in existing_cache_keys]:
                if clean_code in existing_cache_keys:
                    con.execute("DELETE FROM nutrition_cache WHERE product_key = ?;", [old_key])
                    existing_cache_keys.remove(old_key)
                    merged_cache_deletes += 1
                else:
                    con.execute("UPDATE nutrition_cache SET product_key = ?, ean = ? WHERE product_key = ?;", [clean_code, clean_code, old_key])
                    existing_cache_keys.remove(old_key)
                    existing_cache_keys.add(clean_code)
                    updated_cache_keys += 1

        con.commit()
        if migrated_purchases > 0:
            print(f"[+] Migrated {migrated_purchases} legacy '55' barcodes in purchases.")
            print(f"    - Updated {updated_cache_keys} clean keys in nutrition_cache.")
            print(f"    - Merged & deleted {merged_cache_deletes} duplicate keys.")
    except Exception as e:
        con.rollback()
        print(f"[!] Error migrating legacy '55' barcodes: {e}")
        raise e


def migrate_romanian_taxonomy_to_english(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotently migrate historical Romanian taxonomy values in nutrition_cache to canonical English."""
    _, ro_to_en = load_translations("ro")
    ro_cats = [ro for ro, en in ro_to_en.items() if ro != en]
    if not ro_cats:
        return
    placeholders = ", ".join(["?"] * len(ro_cats))
    matched_row = con.execute(f"""
        SELECT COUNT(*) 
        FROM nutrition_cache 
        WHERE category IN ({placeholders}) OR macro_category IN ({placeholders})
    """, ro_cats + ro_cats).fetchone()

    matched = matched_row[0] if matched_row else 0
    if matched == 0:
        return

    con.begin()
    try:
        con.execute("CREATE TEMP TABLE IF NOT EXISTS tax_migration (ro_val VARCHAR PRIMARY KEY, en_val VARCHAR);")
        con.execute("TRUNCATE tax_migration;")
        con.executemany("INSERT INTO tax_migration VALUES (?, ?);", [(ro, en) for ro, en in ro_to_en.items() if ro != en])

        con.execute("""
            UPDATE nutrition_cache
            SET category = m.en_val
            FROM tax_migration m
            WHERE nutrition_cache.category = m.ro_val;
        """)
        con.execute("""
            UPDATE nutrition_cache
            SET macro_category = m.en_val
            FROM tax_migration m
            WHERE nutrition_cache.macro_category = m.ro_val;
        """)
        con.execute("DROP TABLE IF EXISTS tax_migration;")
        con.commit()
        print(f"[+] Migrated {matched} Romanian taxonomy entries in nutrition_cache to canonical English.")
    except Exception as e:
        con.rollback()
        print(f"[!] Error migrating Romanian taxonomy in DuckDB: {e}")
        raise e


# =====================================================================
# 4. TICKET INSERTION & DISCOUNT BACKFILL
# =====================================================================
def backfill_discounts_if_needed(con: duckdb.DuckDBPyConnection) -> None:
    """Retroactively compute discount amounts for older purchases lines."""
    unbackfilled_row = con.execute(
        "SELECT COUNT(*) FROM purchases WHERE discount_amount IS NULL OR (discount_amount = 0.00 AND net_line_total = 0.00)"
    ).fetchone()
    if unbackfilled_row is None:
        raise RuntimeError("Query for unbackfilled discounts returned no result.")
    unbackfilled = unbackfilled_row[0]
    if unbackfilled == 0:
        return

    print(f"\n--- Backfilling Discounts Retroactively ({unbackfilled} rows detected) ---")
    disk_files = list(RAW_DIR.glob("*.json"))
    update_rows = []

    for ticket_file in disk_files:
        t_id = ticket_file.stem
        try:
            with open(ticket_file, "r", encoding="utf-8") as f:
                ticket = json.load(f)
        except Exception as e:
            print(f"[!] Warning: File '{ticket_file.name}' could not be read ({e}). Skipping.")
            continue

        lines = ticket.get("itemsLine", ticket.get("lines", []))
        for idx, line in enumerate(lines):
            disc = parse_line_discounts(line)
            raw_price = line.get("price")
            raw_unit = line.get("currentUnitPrice")
            raw_orig = line.get("originalAmount")
            qty = parse_decimal(line.get("quantity"), default="1.000")

            if raw_price is not None and str(raw_price).strip() not in ("", "None", "null"):
                line_total = parse_decimal(raw_price).quantize(Decimal("0.01"))
            elif raw_unit is not None and str(raw_unit).strip() not in ("", "None", "null"):
                line_total = (qty * parse_decimal(raw_unit)).quantize(Decimal("0.01"))
            elif raw_orig is not None and str(raw_orig).strip() not in ("", "None", "null"):
                line_total = parse_decimal(raw_orig).quantize(Decimal("0.01"))
            else:
                line_total = Decimal("0.00")

            net_total = max(Decimal("0.00"), (line_total - disc).quantize(Decimal("0.01")))
            update_rows.append((disc, net_total, t_id, idx))

    if update_rows:
        con.executemany("""
            UPDATE purchases 
            SET discount_amount = ?, net_line_total = ? 
            WHERE ticket_id = ? AND line_idx = ?;
        """, update_rows)
        print(f"[+] Reconciled discounts retroactively across {len(update_rows)} rows.")


def process_unprocessed_tickets_to_db(con: duckdb.DuckDBPyConnection) -> None:
    """Parse JSON ticket files from disk and insert missing purchases into DuckDB."""
    disk_files = list(RAW_DIR.glob("*.json"))
    existing_db_ids = set(r[0] for r in con.execute("SELECT DISTINCT ticket_id FROM purchases").fetchall())

    tickets_to_process = [p for p in disk_files if p.stem not in existing_db_ids]
    print(f"\n--- STEP 3: Loading Receipts into DuckDB ({len(tickets_to_process)} receipts to process out of {len(disk_files)} on disk) ---")

    if not tickets_to_process:
        print("[i] All receipts on disk are already stored in DuckDB.")
        backfill_discounts_if_needed(con)
        return

    rows_to_insert = []
    for ticket_file in tickets_to_process:
        t_id = ticket_file.stem
        try:
            with open(ticket_file, "r", encoding="utf-8") as f:
                ticket = json.load(f)
        except Exception as e:
            print(f"[!] Warning: Receipt '{ticket_file.name}' is corrupted or could not be read ({e}). Skipping.")
            continue

        date_str = ticket.get("date", "")
        lines = ticket.get("itemsLine", ticket.get("lines", []))

        for idx, line in enumerate(lines):
            raw_code = str(line.get("codeInput") or "").strip()
            b_type, clean_ean = extract_clean_ean(raw_code)

            qty = parse_decimal(line.get("quantity"), default="1.000")
            raw_unit = line.get("currentUnitPrice")
            raw_price = line.get("price")
            raw_orig = line.get("originalAmount")

            if raw_unit is not None and str(raw_unit).strip() not in ("", "None", "null"):
                unit_price = parse_decimal(raw_unit)
            elif raw_price is not None and str(raw_price).strip() not in ("", "None", "null"):
                unit_price = (parse_decimal(raw_price) / qty) if qty != Decimal("0.000") else parse_decimal(raw_price)
            elif raw_orig is not None and str(raw_orig).strip() not in ("", "None", "null"):
                unit_price = (parse_decimal(raw_orig) / qty) if qty != Decimal("0.000") else parse_decimal(raw_orig)
            else:
                unit_price = Decimal("0.00")

            if raw_price is not None and str(raw_price).strip() not in ("", "None", "null"):
                line_total = parse_decimal(raw_price).quantize(Decimal("0.01"))
            else:
                line_total = (qty * unit_price).quantize(Decimal("0.01"))

            disc = parse_line_discounts(line)
            net_total = max(Decimal("0.00"), (line_total - disc).quantize(Decimal("0.01")))
            name = str(line.get("name") or line.get("description") or "").strip()

            rows_to_insert.append((
                t_id, date_str, idx, name, qty, unit_price, line_total, raw_code, b_type, clean_ean, disc, net_total
            ))

    con.executemany("""
        INSERT OR IGNORE INTO purchases (
            ticket_id, 
            date, 
            line_idx, 
            name, 
            quantity, 
            unit_price, 
            line_total,  
            raw_code, 
            barcode_type, 
            ean,
            discount_amount,
            net_line_total
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    print(f"[+] Inserted {len(rows_to_insert)} item lines into 'purchases' table.")
    backfill_discounts_if_needed(con)


# =====================================================================
# 5. OPEN FOOD FACTS SYNC (HYBRID: LOCAL PARQUET OR LIVE REST API v3/v2)
# =====================================================================
def fetch_off_product_api(ean: str, session: requests.Session | None = None) -> dict | None:
    """Query Open Food Facts Live REST API (v3 / v2 compatible) for a single EAN barcode.

    Returns:
        - dict containing product details when found (status == 1 or 'success').
        - {'code': ean, 'not_found': True} when product is definitively missing (404 or status == 0 / 'failure').
        - None when a transient network error, timeout, or 5xx occurs (allows retry on next sync).
    """
    clean_code = str(ean).strip()
    if not clean_code:
        return None

    country_sub = LIDL_COUNTRY.lower() if LIDL_COUNTRY else "ro"
    urls_to_try = [
        f"https://world.openfoodfacts.net/api/v3/product/{clean_code}.json",
        f"https://world.openfoodfacts.org/api/v3/product/{clean_code}.json",
        f"https://{country_sub}.openfoodfacts.org/api/v3/product/{clean_code}.json"
    ]
    params = {
        "fields": "code,product_name,product_name_ro,product_name_en,brands,nutriments,nutriscore_grade,nutriscore_score,nova_group,categories_tags"
    }
    headers = {
        "User-Agent": OFF_USER_AGENT
    }

    http_client = session if session is not None else requests

    for url in urls_to_try:
        try:
            resp = http_client.get(url, params=params, headers=headers, timeout=6.0)

            # Handle rate limiting (HTTP 429) with Retry-After header and progressive backoff
            if resp.status_code == 429:
                for backoff in (10.0, 30.0, 60.0):
                    headers_obj = getattr(resp, "headers", None)
                    retry_header = headers_obj.get("Retry-After") if hasattr(headers_obj, "get") else None
                    sleep_sec = backoff
                    if isinstance(retry_header, (str, int, float)):
                        try:
                            sleep_sec = max(1.0, float(retry_header))
                        except (ValueError, TypeError):
                            sleep_sec = backoff
                    time.sleep(sleep_sec)
                    resp = http_client.get(url, params=params, headers=headers, timeout=6.0)
                    if resp.status_code != 429:
                        break

            if resp.status_code == 404:
                return {"code": clean_code, "not_found": True}

            if resp.status_code != 200:
                continue

            try:
                payload = json.loads(resp.content.decode("utf-8")) if hasattr(resp, "content") and resp.content else resp.json()
            except Exception:
                payload = resp.json()

            st = payload.get("status")
            res_id = payload.get("result", {}).get("id") if isinstance(payload.get("result"), dict) else None
            if st in (0, "failure", "product_not_found") or res_id == "product_not_found":
                return {"code": clean_code, "not_found": True}

            product = payload.get("product") or {}
            if not product and st not in (1, "success"):
                return {"code": clean_code, "not_found": True}

            return {
                "code": clean_code,
                "product_name": product.get("product_name"),
                "product_name_ro": product.get("product_name_ro"),
                "product_name_en": product.get("product_name_en"),
                "brands": product.get("brands"),
                "nutriments": product.get("nutriments") or {},
                "nutriscore_grade": product.get("nutriscore_grade"),
                "nutriscore_score": product.get("nutriscore_score"),
                "nova_group": product.get("nova_group"),
                "categories_tags": product.get("categories_tags") or [],
                "not_found": False
            }

        except (requests.RequestException, Exception):
            continue

    print(f"\n    [!] Network error querying Open Food Facts for EAN {clean_code} (will retry on next sync)")
    return None


def fetch_off_batch_api(eans: list[str], session: requests.Session | None = None) -> list[dict] | None:
    """Query Open Food Facts Live REST API in batch using national/world search endpoint.
    
    Returns a list of parsed product dicts, or None if transient network error.
    """
    if not eans:
        return []
    clean_codes = [str(c).strip() for c in eans if str(c).strip()]
    if not clean_codes:
        return []

    country_sub = LIDL_COUNTRY.lower() if LIDL_COUNTRY else "ro"
    urls_to_try = [
        "https://world.openfoodfacts.net/api/v2/search",
        f"https://{country_sub}.openfoodfacts.net/api/v2/search",
        f"https://{country_sub}.openfoodfacts.org/api/v2/search",
        "https://world.openfoodfacts.org/api/v2/search"
    ]

    http_client = session if session is not None else requests
    headers = {
        "User-Agent": OFF_USER_AGENT
    }
    params = {
        "code": ",".join(clean_codes),
        "fields": "code,product_name,product_name_ro,product_name_en,brands,nutriments,nutriscore_grade,nutriscore_score,nova_group,categories_tags",
        "page_size": max(20, min(100, len(clean_codes)))
    }

    for url in urls_to_try:
        try:
            resp = http_client.get(url, params=params, headers=headers, timeout=10.0)
            if resp.status_code == 429:
                for backoff in (10.0, 30.0, 60.0):
                    headers_obj = getattr(resp, "headers", None)
                    retry_header = headers_obj.get("Retry-After") if hasattr(headers_obj, "get") else None
                    sleep_sec = backoff
                    if isinstance(retry_header, (str, int, float)):
                        try:
                            sleep_sec = max(1.0, float(retry_header))
                        except (ValueError, TypeError):
                            sleep_sec = backoff
                    time.sleep(sleep_sec)
                    resp = http_client.get(url, params=params, headers=headers, timeout=10.0)
                    if resp.status_code != 429:
                        break

            if resp.status_code != 200:
                continue

            try:
                payload = json.loads(resp.content.decode("utf-8")) if hasattr(resp, "content") and resp.content else resp.json()
            except Exception:
                payload = resp.json()

            raw_prods = payload.get("products") or []
            found_map = {}
            for p in raw_prods:
                c = str(p.get("code") or "").strip()
                if c:
                    found_map[c] = {
                        "code": c,
                        "product_name": p.get("product_name"),
                        "product_name_ro": p.get("product_name_ro"),
                        "product_name_en": p.get("product_name_en"),
                        "brands": p.get("brands"),
                        "nutriments": p.get("nutriments") or {},
                        "nutriscore_grade": p.get("nutriscore_grade"),
                        "nutriscore_score": p.get("nutriscore_score"),
                        "nova_group": p.get("nova_group"),
                        "categories_tags": p.get("categories_tags") or [],
                        "not_found": False
                    }

            results = []
            for c in clean_codes:
                if c in found_map:
                    results.append(found_map[c])
                else:
                    results.append({"code": c, "not_found": True})
            return results

        except Exception:
            continue

    return None


def update_nutrition_cache_from_off(
    con: duckdb.DuckDBPyConnection,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Enrich new global EAN codes using local Open Food Facts Parquet dump or Live REST API v2."""
    print("\n--- STEP 4: Matching New Products with Open Food Facts ---")

    # 1. Register new non-EAN codes into cache as BON_ONLY
    con.execute("""
        INSERT INTO nutrition_cache (product_key, sample_name, ean, match_status, data_source, is_verified)
        SELECT 
            TRIM(CAST(COALESCE(NULLIF(TRIM(p.ean), ''), p.raw_code) AS VARCHAR)) AS product_key,
            MIN(TRIM(p.name)) AS sample_name,
            NULLIF(TRIM(p.ean), '') AS ean,
            'NO_GLOBAL_EAN' AS match_status,
            'BON_ONLY' AS data_source,
            FALSE AS is_verified
        FROM purchases p
        LEFT JOIN nutrition_cache c 
          ON TRIM(CAST(COALESCE(NULLIF(TRIM(p.ean), ''), p.raw_code) AS VARCHAR)) = c.product_key
        WHERE (p.barcode_type != 'GLOBAL_EAN' OR p.ean IS NULL OR TRIM(p.ean) = '') 
          AND c.product_key IS NULL
          AND p.raw_code IS NOT NULL
        GROUP BY 1, 3;
    """)

    # 2. Identify global EAN codes to look up (new items OR expired NOT_FOUND_OFF with TTL)
    ttl_clause = ""
    if OFF_NOT_FOUND_TTL_DAYS > 0:
        ttl_clause = f"""
            OR (
                c.match_status = 'NOT_FOUND_OFF'
                AND c.cached_at < CURRENT_TIMESTAMP - INTERVAL '{OFF_NOT_FOUND_TTL_DAYS}' DAY
                AND c.is_verified = FALSE
                AND c.data_source != 'MANUAL_CSV'
            )
        """

    missing = con.execute(f"""
        SELECT 
            TRIM(p.ean) AS ean,
            MIN(TRIM(p.name)) AS sample_name
        FROM purchases p
        LEFT JOIN nutrition_cache c ON TRIM(p.ean) = c.product_key
        WHERE p.barcode_type = 'GLOBAL_EAN'
          AND p.ean IS NOT NULL 
          AND TRIM(p.ean) != '' 
          AND (
              c.product_key IS NULL
              {ttl_clause}
          )
        GROUP BY 1
    """).fetchall()

    if not missing:
        print("[+] No new or expired global EAN codes to look up in Open Food Facts.")
        return

    # Determine execution mode (auto, parquet, api)
    is_parquet_mode = (OFF_MODE == "parquet") or (OFF_MODE == "auto" and os.path.exists(PARQUET_OFF_PATH))

    editable_fields = OVERRIDE_TEXT_COLUMNS + OVERRIDE_NUM_COLUMNS
    protected_flags = [
        f"MAX(CASE WHEN metadata.field_name = '{field_name}' AND metadata.is_protected "
        f"THEN 1 ELSE 0 END) AS protect_{field_name}"
        for field_name in editable_fields
    ]
    grouped_fields = ", ".join(f"cache.{field_name}" for field_name in editable_fields)
    backup_projection = ", ".join(f"cache.{field_name}" for field_name in editable_fields)

    def capture_protected_off_values() -> None:
        con.execute("DROP TABLE IF EXISTS temp_off_protected_values")
        con.execute(f"""
            CREATE TEMP TABLE temp_off_protected_values AS
            SELECT cache.product_key, {backup_projection}, {', '.join(protected_flags)}
            FROM nutrition_cache cache
            LEFT JOIN product_field_metadata metadata
              ON metadata.product_key = cache.product_key
            WHERE cache.product_key IN (SELECT UNNEST(?))
            GROUP BY cache.product_key, {grouped_fields}
        """, [[ean for ean, _name in missing]])

    def restore_protected_off_values(product_keys: list[str], *, drop: bool = False) -> None:
        update_parts = []
        nutrition_fields = {
            "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
            "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
        }
        for field_name in editable_fields:
            if field_name in nutrition_fields:
                expression = (
                    f"CASE WHEN backup.category = 'Non-Food' THEN NULL "
                    f"WHEN backup.protect_{field_name} = 1 THEN backup.{field_name} "
                    f"ELSE nutrition_cache.{field_name} END"
                )
            elif field_name == "package_weight_g":
                # OFF has no package-weight column; preserve any existing
                # value rather than losing it during the cache-row refresh.
                expression = f"backup.{field_name}"
            else:
                expression = f"CASE WHEN backup.protect_{field_name} = 1 THEN backup.{field_name} ELSE nutrition_cache.{field_name} END"
            update_parts.append(f"{field_name} = {expression}")
        con.execute(f"""
            UPDATE nutrition_cache
            SET {', '.join(update_parts)}
            FROM temp_off_protected_values backup
            WHERE nutrition_cache.product_key = backup.product_key
              AND nutrition_cache.product_key IN (SELECT UNNEST(?))
        """, [product_keys])
        if drop:
            con.execute("DROP TABLE IF EXISTS temp_off_protected_values")

    if is_parquet_mode:
        if not os.path.exists(PARQUET_OFF_PATH):
            print(f"[!] Warning: OFF_MODE='parquet' was requested but '{PARQUET_OFF_PATH}' does not exist. Skipping OFF enrichment.")
            return

        print(f"[i] Mode: Local Parquet Dump. Detected {len(missing)} global EAN barcodes. Querying '{PARQUET_OFF_PATH}'...")
        capture_protected_off_values()

        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f"""
                DELETE FROM nutrition_cache
                WHERE product_key IN (
                    SELECT TRIM(p.ean) FROM purchases p
                    LEFT JOIN nutrition_cache c ON TRIM(p.ean) = c.product_key
                    WHERE p.barcode_type = 'GLOBAL_EAN'
                      AND p.ean IS NOT NULL AND TRIM(p.ean) != ''
                      AND (c.product_key IS NULL {ttl_clause})
                )
                  AND is_verified = FALSE
                  AND data_source != 'MANUAL_CSV'
            """)

            query = f"""
            INSERT INTO nutrition_cache (
                product_key, sample_name, ean, off_product_name, off_brands,
                macro_category, category,
                nova_group, kcal_100g, proteins_100g, carbs_100g, sugars_100g,
                fat_100g, saturated_fat_100g, fiber_100g, salt_100g,
                match_status, data_source, is_verified, cached_at
            )
            WITH new_eans AS (
                SELECT 
                    TRIM(p.ean) AS ean,
                    MIN(TRIM(p.name)) AS sample_name
                FROM purchases p
                LEFT JOIN nutrition_cache c ON TRIM(p.ean) = c.product_key
                WHERE p.barcode_type = 'GLOBAL_EAN'
                  AND p.ean IS NOT NULL 
                  AND TRIM(p.ean) != '' 
                  AND (
                      c.product_key IS NULL
                      {ttl_clause}
                  )
                GROUP BY 1
            ),
            dedup_off AS (
                SELECT 
                    code, product_name, brands, nova_group, nutriments, categories_tags
                FROM '{PARQUET_OFF_PATH}'
                WHERE code IN (SELECT ean FROM new_eans)
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY code 
                    ORDER BY completeness DESC NULLS LAST, last_modified_t DESC NULLS LAST
                ) = 1
            ),
            classified_off AS (
                SELECT
                    off.*,
                    CASE
                        -- 1. Frozen
                        WHEN list_has_any(off.categories_tags, ['en:ice-creams', 'en:ice-cream-tubs', 'en:sorbets', 'en:frozen-desserts'])
                            THEN 'Ice Cream & Frozen Desserts'
                        WHEN list_has_any(off.categories_tags, ['en:frozen-fishes', 'en:frozen-seafood'])
                            THEN 'Frozen Fish & Seafood'
                        WHEN list_has_any(off.categories_tags, ['en:frozen-vegetables', 'en:frozen-fruits'])
                            THEN 'Frozen Fruits & Vegetables'
                        WHEN list_has_any(off.categories_tags, ['en:frozen-foods', 'en:frozen-ready-meals', 'en:frozen-pizzas', 'en:frozen-pastries'])
                            THEN 'Frozen Ready Meals & Pizza'

                        -- 2. Alcoholic Beverages
                        WHEN list_has_any(off.categories_tags, ['en:alcoholic-beverages', 'en:beers', 'en:wines', 'en:spirits', 'en:liqueurs', 'en:ciders'])
                            THEN 'Alcoholic Beverages'

                        -- 3. Hot Beverages (Coffee & Tea)
                        WHEN list_has_any(off.categories_tags, ['en:coffees', 'en:teas', 'en:herbal-teas', 'en:coffee-beans', 'en:instant-coffees'])
                            THEN 'Coffee & Tea'

                        -- 4. Water & Soft Drinks
                        WHEN list_has_any(off.categories_tags, ['en:beverages', 'en:waters', 'en:spring-waters', 'en:mineral-waters', 'en:soft-drinks', 'en:sodas', 'en:fruit-juices', 'en:fruit-nectars', 'en:energy-drinks', 'en:iced-teas', 'en:lemonades'])
                            THEN 'Water & Soft Drinks'

                        -- 5. Dairy & Eggs
                        WHEN list_has_any(off.categories_tags, ['en:dairies', 'en:cheeses', 'en:milks', 'en:yogurts', 'en:butters', 'en:eggs', 'en:creams', 'en:fermented-dairy-products', 'en:cottage-cheeses'])
                            THEN 'Dairy & Eggs'

                        -- 6. Pantry & Groceries (Evaluated before Fresh to prioritize canned & preserved items)
                        WHEN list_has_any(off.categories_tags, ['en:canned-foods', 'en:canned-vegetables', 'en:canned-fishes', 'en:pickles', 'en:canned-fruits', 'en:canned-meals', 'en:preserves', 'en:tinned-foods', 'en:tinned-fishes'])
                            THEN 'Canned Goods & Pickles'
                        WHEN list_has_any(off.categories_tags, ['en:sauces', 'en:condiments', 'en:vegetable-oils', 'en:olive-oils', 'en:vinegars', 'en:ketchups', 'en:mayonnaises', 'en:mustards', 'en:spices', 'en:dressings'])
                            THEN 'Sauces, Oils & Condiments'
                        WHEN list_has_any(off.categories_tags, ['en:cereals-and-potatoes', 'en:pastas', 'en:flours', 'en:rices', 'en:noodles', 'en:baking-ingredients', 'en:grains', 'en:semolinas'])
                            THEN 'Baking & Cooking Ingredients'

                        -- 7. Meat & Fish
                        WHEN list_has_any(off.categories_tags, ['en:prepared-meats', 'en:sausages', 'en:hams', 'en:salmis', 'en:bacon', 'en:pork-charcuterie', 'en:cold-cuts'])
                            THEN 'Processed Meat & Deli'
                        WHEN list_has_any(off.categories_tags, ['en:fresh-fishes', 'en:fishes', 'en:seafood', 'en:salmons', 'en:trouts', 'en:tunas', 'en:fresh-meats', 'en:poultry', 'en:chickens', 'en:turkey', 'en:beef', 'en:pork', 'en:meats'])
                            THEN 'Fresh Meat & Fish'

                        -- 8. Fresh Produce
                        WHEN list_has_any(off.categories_tags, ['en:fresh-fruits', 'en:fresh-vegetables', 'en:fruits', 'en:vegetables', 'en:tomatoes', 'en:apples', 'en:bananas', 'en:potatoes', 'en:citrus', 'en:salads'])
                            THEN 'Fresh Fruits & Vegetables'

                        -- 9. Bread & Bakery Specialties
                        WHEN list_has_any(off.categories_tags, ['en:breads', 'en:baguettes', 'en:sliced-breads', 'en:buns', 'en:rolls', 'en:toast-breads'])
                            THEN 'Bread & Bakery Specialties'

                        -- 10. Sweets & Snacks (Packaged viennoiserie/croissant with EAN route here)
                        WHEN list_has_any(off.categories_tags, ['en:viennoiseries', 'en:croissants', 'en:pain-au-chocolat', 'en:brioches', 'en:pastries', 'en:desserts', 'en:puddings', 'en:custards', 'en:pies', 'en:tarts', 'en:cakes'])
                            THEN 'Desserts & Pastries'
                        WHEN list_has_any(off.categories_tags, ['en:chocolates', 'en:candies', 'en:bonbons', 'en:bars', 'en:cacao-and-chocolates', 'en:sweet-spreads', 'en:confectioneries', 'en:nougats'])
                            THEN 'Chocolate & Confectionery'
                        WHEN list_has_any(off.categories_tags, ['en:biscuits', 'en:cookies', 'en:crackers', 'en:chips', 'en:crisps', 'en:appetizers', 'en:salty-snacks', 'en:popcorn', 'en:nuts', 'en:peanuts'])
                            THEN 'Biscuits & Savory Snacks'

                        -- 11. Ready Meals & Chilled Convenience
                        WHEN list_has_any(off.categories_tags, ['en:meals', 'en:prepared-dishes', 'en:sandwiches', 'en:pizzas', 'en:salads-prepared', 'en:sushi'])
                            THEN 'Ready Meals & Chilled Convenience'

                        ELSE NULL
                    END AS wf_category
                FROM dedup_off off
            )
            SELECT 
                ne.ean AS product_key,
                ne.sample_name,
                ne.ean,
                COALESCE(
                    list_filter(off.product_name, x -> x.lang = 'ro')[1]."text",
                    list_filter(off.product_name, x -> x.lang = 'en')[1]."text",
                    off.product_name[1]."text"
                ) AS off_product_name,
                off.brands AS off_brands,
                CASE
                    WHEN off.wf_category IN ('Fresh Meat & Fish', 'Fresh Fruits & Vegetables', 'Dairy & Eggs', 'Processed Meat & Deli', 'Ready Meals & Chilled Convenience') THEN 'Fresh Food'
                    WHEN off.wf_category IN ('Canned Goods & Pickles', 'Baking & Cooking Ingredients', 'Sauces, Oils & Condiments') THEN 'Pantry & Groceries'
                    WHEN off.wf_category IN ('Water & Soft Drinks', 'Alcoholic Beverages', 'Coffee & Tea') THEN 'Beverages'
                    WHEN off.wf_category IN ('Ice Cream & Frozen Desserts', 'Frozen Fruits & Vegetables', 'Frozen Fish & Seafood', 'Frozen Ready Meals & Pizza') THEN 'Frozen'
                    WHEN off.wf_category IN ('Biscuits & Savory Snacks', 'Chocolate & Confectionery', 'Desserts & Pastries') THEN 'Sweets & Snacks'
                    WHEN off.wf_category IN ('Fresh Bakery & Pastry', 'Bread & Bakery Specialties') THEN 'Bakery & Bread'
                    WHEN off.wf_category = 'Non-Food' THEN 'Non-Food'
                    ELSE NULL
                END AS macro_category,
                off.wf_category AS category,
                off.nova_group,
                ROUND(CAST(
                    COALESCE(
                        list_filter(off.nutriments, x -> x.name = 'energy-kcal')[1]."100g",
                        list_filter(off.nutriments, x -> x.name = 'energy-kj')[1]."100g" / 4.184
                    ) AS NUMERIC
                ), 1) AS kcal_100g,
                ROUND(CAST(list_filter(off.nutriments, x -> x.name = 'proteins')[1]."100g" AS NUMERIC), 2) AS proteins_100g,
                ROUND(CAST(list_filter(off.nutriments, x -> x.name = 'carbohydrates')[1]."100g" AS NUMERIC), 2) AS carbs_100g,
                ROUND(CAST(list_filter(off.nutriments, x -> x.name = 'sugars')[1]."100g" AS NUMERIC), 2) AS sugars_100g,
                ROUND(CAST(list_filter(off.nutriments, x -> x.name = 'fat')[1]."100g" AS NUMERIC), 2) AS fat_100g,
                ROUND(CAST(list_filter(off.nutriments, x -> x.name = 'saturated-fat')[1]."100g" AS NUMERIC), 2) AS saturated_fat_100g,
                ROUND(CAST(list_filter(off.nutriments, x -> x.name = 'fiber')[1]."100g" AS NUMERIC), 2) AS fiber_100g,
                ROUND(CAST(list_filter(off.nutriments, x -> x.name = 'salt')[1]."100g" AS NUMERIC), 2) AS salt_100g,
                CASE 
                    WHEN off.code IS NOT NULL AND off.wf_category IS NOT NULL THEN 'MATCHED_OFF_WATERFALL'
                    WHEN off.code IS NOT NULL THEN 'MATCHED_OFF' 
                    ELSE 'NOT_FOUND_OFF' 
                END AS match_status,
                CASE WHEN off.code IS NOT NULL THEN 'OFF_LABEL' ELSE 'BON_ONLY' END AS data_source,
                FALSE AS is_verified,
                CURRENT_TIMESTAMP AS cached_at
            FROM new_eans ne
            LEFT JOIN classified_off off ON ne.ean = off.code;
        """
            con.execute(query)
            restore_protected_off_values([ean for ean, _name in missing], drop=True)
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            con.execute("DROP TABLE IF EXISTS temp_off_protected_values")
            raise
        if progress_callback:
            progress_callback({
                "stage": "off_enrichment",
                "status": "progress",
                "message": "Open Food Facts lookups complete",
                "completed": len(missing),
                "total": len(missing),
            })
    else:
        # Live REST API v2 mode
        print(f"[i] Mode: Live REST API v2. Detected {len(missing)} global EAN barcodes to query.")
        capture_protected_off_values()
        staged_records = []
        matched_count = 0
        not_found_count = 0
        network_error_count = 0
        session = requests.Session()
        session.headers.update({"User-Agent": OFF_USER_AGENT})

        def _flush_staged_records_impl(con_db: duckdb.DuckDBPyConnection, records: list) -> None:
            if not records:
                return
            con_db.execute("DROP TABLE IF EXISTS temp_off_staged;")
            con_db.execute("""
                CREATE TEMP TABLE temp_off_staged (
                    code VARCHAR,
                    sample_name VARCHAR,
                    ean VARCHAR,
                    off_product_name VARCHAR,
                    off_brands VARCHAR,
                    nova_group INTEGER,
                    kcal_100g DOUBLE,
                    proteins_100g DOUBLE,
                    carbs_100g DOUBLE,
                    sugars_100g DOUBLE,
                    fat_100g DOUBLE,
                    saturated_fat_100g DOUBLE,
                    fiber_100g DOUBLE,
                    salt_100g DOUBLE,
                    categories_tags VARCHAR[]
                );
            """)
            con_db.executemany("""
                INSERT INTO temp_off_staged VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            con_db.execute("""
                DELETE FROM nutrition_cache
                WHERE product_key IN (SELECT code FROM temp_off_staged)
                  AND is_verified = FALSE
                  AND data_source != 'MANUAL_CSV'
            """)

            con_db.execute(f"""
                INSERT INTO nutrition_cache (
                    product_key, sample_name, ean, off_product_name, off_brands,
                    macro_category, category,
                    nova_group, kcal_100g, proteins_100g, carbs_100g, sugars_100g,
                    fat_100g, saturated_fat_100g, fiber_100g, salt_100g,
                    match_status, data_source, is_verified, cached_at
                )
                SELECT 
                    s.code AS product_key,
                    s.sample_name,
                    s.ean,
                    s.off_product_name,
                    s.off_brands,
                    CASE
                        WHEN off.wf_category IN ('Fresh Meat & Fish', 'Fresh Fruits & Vegetables', 'Dairy & Eggs', 'Processed Meat & Deli', 'Ready Meals & Chilled Convenience') THEN 'Fresh Food'
                        WHEN off.wf_category IN ('Canned Goods & Pickles', 'Baking & Cooking Ingredients', 'Sauces, Oils & Condiments') THEN 'Pantry & Groceries'
                        WHEN off.wf_category IN ('Water & Soft Drinks', 'Alcoholic Beverages', 'Coffee & Tea') THEN 'Beverages'
                        WHEN off.wf_category IN ('Ice Cream & Frozen Desserts', 'Frozen Fruits & Vegetables', 'Frozen Fish & Seafood', 'Frozen Ready Meals & Pizza') THEN 'Frozen'
                        WHEN off.wf_category IN ('Biscuits & Savory Snacks', 'Chocolate & Confectionery', 'Desserts & Pastries') THEN 'Sweets & Snacks'
                        WHEN off.wf_category IN ('Fresh Bakery & Pastry', 'Bread & Bakery Specialties') THEN 'Bakery & Bread'
                        WHEN off.wf_category = 'Non-Food' THEN 'Non-Food'
                        ELSE NULL
                    END AS macro_category,
                    off.wf_category AS category,
                    s.nova_group,
                    s.kcal_100g,
                    s.proteins_100g,
                    s.carbs_100g,
                    s.sugars_100g,
                    s.fat_100g,
                    s.saturated_fat_100g,
                    s.fiber_100g,
                    s.salt_100g,
                    CASE 
                        WHEN off.wf_category IS NOT NULL THEN 'MATCHED_OFF_WATERFALL'
                        ELSE 'MATCHED_OFF' 
                    END AS match_status,
                    'OFF_API' AS data_source,
                    FALSE AS is_verified,
                    CURRENT_TIMESTAMP AS cached_at
                FROM temp_off_staged s
                CROSS JOIN LATERAL (
                    SELECT
                        CASE
                            -- 1. Frozen
                            WHEN list_has_any(s.categories_tags, ['en:ice-creams', 'en:ice-cream-tubs', 'en:sorbets', 'en:frozen-desserts'])
                                THEN 'Ice Cream & Frozen Desserts'
                            WHEN list_has_any(s.categories_tags, ['en:frozen-fishes', 'en:frozen-seafood'])
                                THEN 'Frozen Fish & Seafood'
                            WHEN list_has_any(s.categories_tags, ['en:frozen-vegetables', 'en:frozen-fruits'])
                                THEN 'Frozen Fruits & Vegetables'
                            WHEN list_has_any(s.categories_tags, ['en:frozen-foods', 'en:frozen-ready-meals', 'en:frozen-pizzas', 'en:frozen-pastries'])
                                THEN 'Frozen Ready Meals & Pizza'

                            -- 2. Alcoholic Beverages
                            WHEN list_has_any(s.categories_tags, ['en:alcoholic-beverages', 'en:beers', 'en:wines', 'en:spirits', 'en:liqueurs', 'en:ciders'])
                                THEN 'Alcoholic Beverages'

                            -- 3. Hot Beverages (Coffee & Tea)
                            WHEN list_has_any(s.categories_tags, ['en:coffees', 'en:teas', 'en:herbal-teas', 'en:coffee-beans', 'en:instant-coffees'])
                                THEN 'Coffee & Tea'

                            -- 4. Water & Soft Drinks
                            WHEN list_has_any(s.categories_tags, ['en:beverages', 'en:waters', 'en:spring-waters', 'en:mineral-waters', 'en:soft-drinks', 'en:sodas', 'en:fruit-juices', 'en:fruit-nectars', 'en:energy-drinks', 'en:iced-teas', 'en:lemonades'])
                                THEN 'Water & Soft Drinks'

                            -- 5. Dairy & Eggs
                            WHEN list_has_any(s.categories_tags, ['en:dairies', 'en:cheeses', 'en:milks', 'en:yogurts', 'en:butters', 'en:eggs', 'en:creams', 'en:fermented-dairy-products', 'en:cottage-cheeses'])
                                THEN 'Dairy & Eggs'

                            -- 6. Pantry & Groceries (Evaluated before Fresh to prioritize canned & preserved items)
                            WHEN list_has_any(s.categories_tags, ['en:canned-foods', 'en:canned-vegetables', 'en:canned-fishes', 'en:pickles', 'en:canned-fruits', 'en:canned-meals', 'en:preserves', 'en:tinned-foods', 'en:tinned-fishes'])
                                THEN 'Canned Goods & Pickles'
                            WHEN list_has_any(s.categories_tags, ['en:sauces', 'en:condiments', 'en:vegetable-oils', 'en:olive-oils', 'en:vinegars', 'en:ketchups', 'en:mayonnaises', 'en:mustards', 'en:spices', 'en:dressings'])
                                THEN 'Sauces, Oils & Condiments'
                            WHEN list_has_any(s.categories_tags, ['en:cereals-and-potatoes', 'en:pastas', 'en:flours', 'en:rices', 'en:noodles', 'en:baking-ingredients', 'en:grains', 'en:semolinas'])
                                THEN 'Baking & Cooking Ingredients'

                            -- 7. Meat & Fish
                            WHEN list_has_any(s.categories_tags, ['en:prepared-meats', 'en:sausages', 'en:hams', 'en:salmis', 'en:bacon', 'en:pork-charcuterie', 'en:cold-cuts'])
                                THEN 'Processed Meat & Deli'
                            WHEN list_has_any(s.categories_tags, ['en:fresh-fishes', 'en:fishes', 'en:seafood', 'en:salmons', 'en:trouts', 'en:tunas', 'en:fresh-meats', 'en:poultry', 'en:chickens', 'en:turkey', 'en:beef', 'en:pork', 'en:meats'])
                                THEN 'Fresh Meat & Fish'

                            -- 8. Fresh Produce
                            WHEN list_has_any(s.categories_tags, ['en:fresh-fruits', 'en:fresh-vegetables', 'en:fruits', 'en:vegetables', 'en:tomatoes', 'en:apples', 'en:bananas', 'en:potatoes', 'en:citrus', 'en:salads'])
                                THEN 'Fresh Fruits & Vegetables'

                            -- 9. Bread & Bakery Specialties
                            WHEN list_has_any(s.categories_tags, ['en:breads', 'en:baguettes', 'en:sliced-breads', 'en:buns', 'en:rolls', 'en:toast-breads'])
                                THEN 'Bread & Bakery Specialties'

                            -- 10. Sweets & Snacks (Packaged viennoiserie/croissant with EAN route here)
                            WHEN list_has_any(s.categories_tags, ['en:viennoiseries', 'en:croissants', 'en:pain-au-chocolat', 'en:brioches', 'en:pastries', 'en:desserts', 'en:puddings', 'en:custards', 'en:pies', 'en:tarts', 'en:cakes'])
                                THEN 'Desserts & Pastries'
                            WHEN list_has_any(s.categories_tags, ['en:chocolates', 'en:candies', 'en:bonbons', 'en:bars', 'en:cacao-and-chocolates', 'en:sweet-spreads', 'en:confectioneries', 'en:nougats'])
                                THEN 'Chocolate & Confectionery'
                            WHEN list_has_any(s.categories_tags, ['en:biscuits', 'en:cookies', 'en:crackers', 'en:chips', 'en:crisps', 'en:appetizers', 'en:salty-snacks', 'en:popcorn', 'en:nuts', 'en:peanuts'])
                                THEN 'Biscuits & Savory Snacks'

                            -- 11. Ready Meals & Chilled Convenience
                            WHEN list_has_any(s.categories_tags, ['en:meals', 'en:prepared-dishes', 'en:sandwiches', 'en:pizzas', 'en:salads-prepared', 'en:sushi'])
                                THEN 'Ready Meals & Chilled Convenience'

                            ELSE NULL
                        END AS wf_category
                ) off;
            """)
            restore_protected_off_values([record[0] for record in records])
            con_db.execute("DROP TABLE IF EXISTS temp_off_staged;")
            records.clear()

        def _flush_staged_records(con_db: duckdb.DuckDBPyConnection, records: list) -> None:
            if not records:
                return
            con_db.execute("BEGIN TRANSACTION")
            try:
                _flush_staged_records_impl(con_db, records)
                con_db.execute("COMMIT")
            except Exception:
                try:
                    con_db.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        import shutil

        def report_progress(current_idx: int, total_count: int, ean_str: str, outcome_msg: str) -> None:
            pct = (current_idx / total_count) * 100 if total_count > 0 else 100.0
            term_cols = shutil.get_terminal_size((80, 20)).columns
            max_len = max(40, min(80, term_cols - 2))
            prefix = f"[i] OFF Sync: [{current_idx}/{total_count}] ({pct:.1f}%) - EAN {ean_str}: "
            avail = max_len - len(prefix)
            if avail > 10:
                truncated_outcome = outcome_msg[:avail - 3] + "..." if len(outcome_msg) > avail else outcome_msg
            else:
                truncated_outcome = outcome_msg[:max(0, avail)]
            line = prefix + truncated_outcome
            sys.stdout.write("\r" + line.ljust(max_len))
            sys.stdout.flush()
            if progress_callback:
                progress_callback({
                    "stage": "off_enrichment",
                    "status": "progress",
                    "message": "Open Food Facts lookup complete",
                    "completed": current_idx,
                    "total": total_count,
                })

        def process_product_info(ean_code: str, s_name: str, prod_info: dict | None, idx: int, total: int) -> None:
            nonlocal matched_count, not_found_count, network_error_count
            if prod_info is None:
                network_error_count += 1
                report_progress(idx, total, ean_code, "Network Error (retrying next sync)")
                return

            if prod_info.get("not_found"):
                not_found_count += 1
                existing_check = con.execute("SELECT 1 FROM nutrition_cache WHERE product_key = ?", [ean_code]).fetchone()
                if existing_check:
                    con.execute("UPDATE nutrition_cache SET cached_at = CURRENT_TIMESTAMP WHERE product_key = ?", [ean_code])
                else:
                    con.execute("""
                        INSERT INTO nutrition_cache (
                            product_key, sample_name, ean, match_status, data_source, is_verified, cached_at
                        ) VALUES (?, ?, ?, 'NOT_FOUND_OFF', 'BON_ONLY', FALSE, CURRENT_TIMESTAMP)
                    """, [ean_code, s_name, ean_code])
                report_progress(idx, total, ean_code, "Not found in Open Food Facts")
                return

            matched_count += 1
            off_name = prod_info.get("product_name_ro") or prod_info.get("product_name") or prod_info.get("product_name_en") or s_name
            off_brand = prod_info.get("brands") or ""
            nova_grp = prod_info.get("nova_group")
            tags = prod_info.get("categories_tags") or []
            nutri = prod_info.get("nutriments") or {}

            kcal = nutri.get("energy-kcal_100g")
            if kcal is None:
                kcal = nutri.get("energy-kcal")
            if kcal is None and nutri.get("energy-kj_100g") is not None:
                try:
                    kcal = round(float(nutri.get("energy-kj_100g")) / 4.184, 1)
                except (ValueError, TypeError):
                    kcal = None
            elif kcal is not None:
                try:
                    kcal = round(float(kcal), 1)
                except (ValueError, TypeError):
                    kcal = None

            def parse_macro(key: str) -> float | None:
                val = nutri.get(f"{key}_100g")
                if val is None:
                    val = nutri.get(key)
                if val is not None:
                    try:
                        return round(float(val), 2)
                    except (ValueError, TypeError):
                        return None
                return None

            staged_records.append((
                ean_code,
                s_name,
                ean_code,
                off_name,
                off_brand,
                int(nova_grp) if nova_grp is not None else None,
                kcal,
                parse_macro("proteins"),
                parse_macro("carbohydrates"),
                parse_macro("sugars"),
                parse_macro("fat"),
                parse_macro("saturated-fat"),
                parse_macro("fiber"),
                parse_macro("salt"),
                tags
            ))
            report_progress(idx, total, ean_code, f"Matched '{off_name}'")

            if len(staged_records) >= 25:
                _flush_staged_records(con, staged_records)

        total_missing = len(missing)
        # Adaptively use batch API when > 20 codes detected
        if total_missing > 20:
            chunk_size = 50
            for c_start in range(0, total_missing, chunk_size):
                chunk = missing[c_start:c_start + chunk_size]
                chunk_eans = [c[0] for c in chunk]
                batch_res = fetch_off_batch_api(chunk_eans, session=session)
                if batch_res is not None:
                    by_code = {p.get("code"): p for p in batch_res}
                    for offset, (ean_c, s_name) in enumerate(chunk):
                        cur_idx = c_start + offset + 1
                        p_info = by_code.get(ean_c, {"code": ean_c, "not_found": True})
                        process_product_info(ean_c, s_name, p_info, cur_idx, total_missing)
                else:
                    # Fallback to single requests on network issue
                    for offset, (ean_c, s_name) in enumerate(chunk):
                        cur_idx = c_start + offset + 1
                        p_info = fetch_off_product_api(ean_c, session=session)
                        process_product_info(ean_c, s_name, p_info, cur_idx, total_missing)
                        time.sleep(0.5)

                if c_start + chunk_size < total_missing:
                    time.sleep(1.5)
        else:
            for cur_idx, (ean_c, s_name) in enumerate(missing, start=1):
                p_info = fetch_off_product_api(ean_c, session=session)
                process_product_info(ean_c, s_name, p_info, cur_idx, total_missing)
                time.sleep(0.8)

        # Final flush for any remaining records
        if staged_records:
            _flush_staged_records(con, staged_records)

        con.execute("DROP TABLE IF EXISTS temp_off_protected_values")
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"[+] OFF Live API enrichment complete: {matched_count} matched, {not_found_count} not found, {network_error_count} network errors.")


# =====================================================================
# 6. NON-FOOD FILTERING, EGG PROFILE & NUTRITION OVERRIDES
# =====================================================================
def run_deterministic_non_food_filter(con: duckdb.DuckDBPyConnection) -> None:
    """Apply rule-based classification for eggs and non-food supermarket items."""
    print("\n--- STEP 5: Filtering Non-Food Items & Egg Profiles ---")

    # Egg nutrition standard patch
    egg_nutrients = {
        "kcal_100g": "143.0", "proteins_100g": "12.6", "carbs_100g": "0.7",
        "sugars_100g": "0.4", "fat_100g": "9.5", "saturated_fat_100g": "3.1",
        "fiber_100g": "0.0", "salt_100g": "0.33",
    }
    egg_field_updates = ",\n".join(
        f"{field_name} = CASE WHEN EXISTS (SELECT 1 FROM product_field_metadata metadata "
        "WHERE metadata.product_key = nutrition_cache.product_key "
        f"AND metadata.field_name = '{field_name}' AND metadata.is_protected) "
        f"THEN nutrition_cache.{field_name} ELSE {value} END"
        for field_name, value in egg_nutrients.items()
    )
    con.execute(f"""
        UPDATE nutrition_cache
        SET 
            {egg_field_updates},
            match_status       = 'PATCHED_EXACT',
            data_source        = 'NUTRITION_STANDARD',
            is_verified        = TRUE
        WHERE is_verified = FALSE
          AND (
            (REGEXP_MATCHES(LOWER(sample_name), '.*\\b(oua|ou)\\b.*') OR REGEXP_MATCHES(LOWER(sample_name), '.*\\d+\\s+oua\\b.*'))
            AND NOT REGEXP_MATCHES(LOWER(sample_name), '.*(vopsea|martipan|croissant|gofra|fidea|maioneza|pappardelle).*')
        );
    """)

    # Strict non-food matching (hygiene, cleaning, packaging, bags, clothing, tools, pet supplies, etc.)
    con.execute("""
        UPDATE nutrition_cache
        SET 
            macro_category     = 'Non-Food',
            category           = 'Non-Food',
            match_status       = 'NON_FOOD',
            data_source        = 'RULE_NON_FOOD',
            kcal_100g          = NULL,
            proteins_100g      = NULL,
            carbs_100g         = NULL,
            sugars_100g        = NULL,
            fat_100g           = NULL,
            saturated_fat_100g = NULL,
            fiber_100g         = NULL,
            salt_100g          = NULL,
            is_weighed_default = FALSE,
            is_verified        = TRUE
        WHERE (
           match_status = 'NON_FOOD'
           OR (
            is_verified = FALSE
            AND (
                REGEXP_MATCHES(LOWER(product_key), '.*(sgr|garantie|reducere|discount|cupon|voucher|taxa|ambalaj).*')
                OR EXISTS (
                    SELECT 1 FROM purchases p 
                    WHERE TRIM(CAST(COALESCE(NULLIF(TRIM(p.ean), ''), p.raw_code) AS VARCHAR)) = nutrition_cache.product_key
                      AND (
                        REGEXP_MATCHES(p.name, '([0-9]G#[0-9]+|[0-9]G[0-9]+|#[0-9]{4,})')
                        OR REGEXP_MATCHES(LOWER(p.name), '(^|[^a-z0-9])(sgr|garantie|reducere|discount|cupon|voucher|ambalaj|taxa|prosop|hartie|servetel|servetele|serv|foite|foita|periuta|pasta de dinti|pasta dinti|apa de gura|irigator|dentara|dentalux|colgate|parodontax|sensodyne|aquafresh|blend a med|oral b|sapun|gel dus|gel de dus|sampon|balsam|crema maini|crema corp|crema fata|lotiune|spray prot|spray protectie|pantene|clear sampon|head shoulders|palmolive|dove|nivea|elseve|burete|bureti|laveta|lavete|mop|galeata|manusi|sita chiuveta|cutit buc|strecuratoare|castron|tacmuri|vesela|farfurie|pahar|tigaie|oala|detergent|solutie|clor min|clor|anticalcar|dezinfectant|detartrant|degresant|capsule spalat|sano|nufar|persil|dero|ariel|somat|finish|domestos|cif|ajax|clin|pronto|bref|cilit bang|vanish|odorizant|wc|bazin wc|color aktiv|air wick|glade|little joe|tableta rezervor|absorbante|tampoane|dischete|discuri demachiante|prezervative|scutece|scutec|pampers|wipes|vata|dizolvant|lama ras|lame ras|aparat ras|gillette|old spice|deodorant|deo|antiperspirant|pungi|punga|sacosa|saci|sac|folie alu|folie alimentara|folie prospet|hartie pt copt|scobitori|chibrit|brichete|pisici|caini|caine|pisica|hrana animale|hrana pisici|hrana caini|batoane pentru pisici|nisip|flori|buchet|ghiveci|plante|pamant|ingrasamant|seminte gazon|peperomia|sosete|ciorapi|chilot|boxeri|pantaloni|pijama|tricou|hanorac|geaca|vesta|papuci|slapi|turban|lenjerie|patura|cearsaf|camasa|baterii|baterie|bec|cablu|banda led|lampa|instalatie|priza|prelungitor|casti|suport auto|bijuterii|bratara|inel|ceas|jucarie|puzzle|carte|jurnal|pixuri|caiet|scame|adeziv|banda adeziva|rola scame|aparat scame|perie|snork|hamac|gradinarit|bricolaj|unelte|surub|cleste|ciocan|lumanare|lumanari|alcool sanitar|spirt|amestec acid acetic|autan|raid rezerva|respiran|praf univ indep petelor|filtre maxtra|super glue|vopsea 10 oua)($|[^a-z0-9])')
                      )
                      AND NOT REGEXP_MATCHES(LOWER(p.name), '.*(mere.*punga|ceapa.*plasa|floricele|praf de copt|purcari|foitaj|oua|ou|carnat|croissant|macrou|gofra|fidea|bulgur|couscous|souffle|maioneza|popcorn).*')
                )
            )
           )
          )
          AND NOT EXISTS (
              SELECT 1 FROM product_field_metadata metadata
              WHERE metadata.product_key = nutrition_cache.product_key
                AND metadata.field_name IN (
                    'category', 'macro_category', 'kcal_100g', 'proteins_100g',
                    'carbs_100g', 'sugars_100g', 'fat_100g',
                    'saturated_fat_100g', 'fiber_100g', 'salt_100g'
                )
                AND metadata.is_protected
          );
    """)
    con.execute("""
        DELETE FROM product_field_metadata
        WHERE field_name IN (
            'kcal_100g', 'proteins_100g', 'carbs_100g', 'sugars_100g',
            'fat_100g', 'saturated_fat_100g', 'fiber_100g', 'salt_100g'
        )
          AND product_key IN (
              SELECT product_key FROM nutrition_cache WHERE category = 'Non-Food'
          )
    """)


def sanitize_csv_cell(val) -> str:
    """Neutralize potential CSV / Formula Injection vulnerabilities for spreadsheet viewers."""
    if val is None:
        return ""
    s = str(val)
    if s.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{s}"
    return s


def desanitize_csv_cell(val: str) -> str:
    """Strip protective single quote prefix when reading back CSV data into Python."""
    if val.startswith("'") and len(val) > 1 and val[1] in ("=", "+", "-", "@", "\t", "\r"):
        return val[1:]
    return val


def read_csv_records(path: Path) -> tuple[list[dict], str, list[str]]:
    """Generic CSV reader returning (rows, delimiter, header)."""
    if not path.exists():
        return [], ",", []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        content = f.read()
    if not content.strip():
        return [], ",", []

    first_line = content.strip().split("\n")[0]
    delimiter = ";" if ";" in first_line and "," not in first_line else ","
    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    header = [h.strip().lower() for h in (reader.fieldnames or []) if h]
    rows = []
    for row in reader:
        rows.append({
            k.strip().lower(): desanitize_csv_cell(v.strip() if v else "")
            for k, v in row.items() if k
        })
    return rows, delimiter, header


def write_csv_records(path: Path, rows: list[dict], fieldnames: list[str], delimiter: str = ",") -> None:
    """Atomically rewrite CSV file with sanitized cells."""
    extra = []
    for r in rows:
        for k in r:
            if k not in fieldnames and k not in extra:
                extra.append(k)
    tmp_path = path.with_name(path.name + ".tmp")
    sanitized_rows = [
        {k: sanitize_csv_cell(v) for k, v in r.items()}
        for r in rows
    ]
    with open(tmp_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra, delimiter=delimiter, restval="")
        writer.writeheader()
        writer.writerows(sanitized_rows)
    os.replace(tmp_path, path)


def read_overrides_csv() -> tuple[list[dict], str, list[str]]:
    """Read catalog_overrides.csv and return (rows, delimiter, header)."""
    return read_csv_records(OVERRIDE_CSV_PATH)


def write_overrides_csv(rows: list[dict], delimiter: str = ",") -> None:
    """Atomically rewrite catalog_overrides.csv with sanitized cells and custom columns preserved."""
    write_csv_records(OVERRIDE_CSV_PATH, rows, OVERRIDE_COLUMNS, delimiter)


def load_taxonomy() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load taxonomy JSON and build a reverse lookup map (category -> macro_category)."""
    if not TAXONOMY_PATH.exists():
        print(f"[i] Taxonomy file '{TAXONOMY_PATH}' not found. Macro-category resolution will be manual.")
        return {}, {}
    try:
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            tax = json.load(f)
        category_to_macro = {}
        for macro, subcats in tax.items():
            for sub in subcats:
                sub_clean = sub.strip()
                if sub_clean and sub_clean not in category_to_macro:
                    category_to_macro[sub_clean] = macro
        return tax, category_to_macro
    except Exception as e:
        print(f"[!] Warning reading '{TAXONOMY_PATH}': {e}")
        return {}, {}


def apply_seed_catalog_to_db(con: duckdb.DuckDBPyConnection, quiet: bool = False) -> None:
    """Smart Upsert seed catalog data from data/catalogs/{country}/catalog_seed.csv into DuckDB.
    
    Provides Day-1 coverage without overwriting manual user overrides in nutrition_cache.
    """
    seed_tuples = []
    if SEED_CATALOG_PATH.exists():
        if not quiet:
            print(f"\n--- STEP 2: Bootstrapping Seed Catalog from '{SEED_CATALOG_PATH}' ---")
        rows, _, _ = read_csv_records(SEED_CATALOG_PATH)
        seen = set()
        for row in rows:
            pk = clean_product_key(row.get("product_key"))
            if not pk or pk in seen:
                continue
            seen.add(pk)

            sample_name = row.get("sample_name") or ""
            macro = row.get("macro_category") or None
            cat = row.get("category") or None
            pkg_w = parse_optional_float(row.get("package_weight_g"))
            kcal = parse_optional_float(row.get("kcal_100g"))
            prot = parse_optional_float(row.get("proteins_100g"))
            carbs = parse_optional_float(row.get("carbs_100g"))
            sugars = parse_optional_float(row.get("sugars_100g"))
            fat = parse_optional_float(row.get("fat_100g"))
            sat_fat = parse_optional_float(row.get("saturated_fat_100g"))
            fiber = parse_optional_float(row.get("fiber_100g"))
            salt = parse_optional_float(row.get("salt_100g"))
            non_food = (row.get("non_food") or "").strip().lower() in TRUTHY
            match_status = "NON_FOOD" if non_food else (row.get("match_status") or "SEED_CATALOG")
            off_name = row.get("off_product_name") or None
            is_ver = True if (kcal is not None or non_food) else False

            seed_tuples.append((
                pk, sample_name, macro, cat, pkg_w,
                kcal, prot, carbs, sugars, fat, sat_fat, fiber, salt,
                match_status, off_name, is_ver
            ))
    else:
        print(f"[i] Seed catalog '{SEED_CATALOG_PATH}' not found. Checking sibling catalogs for cross-border pooling.")

    con.begin()
    try:
        if seed_tuples:
            con.execute("CREATE TEMP TABLE IF NOT EXISTS seed_staging AS SELECT * FROM nutrition_cache WHERE 1=0;")
            con.execute("TRUNCATE seed_staging;")
            con.executemany("""
                INSERT INTO seed_staging (
                    product_key, sample_name, macro_category, category, package_weight_g,
                    kcal_100g, proteins_100g, carbs_100g, sugars_100g, fat_100g, saturated_fat_100g,
                    fiber_100g, salt_100g, match_status, off_product_name, is_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, seed_tuples)

            # 1. Insert new products from seed that don't exist yet in cache
            con.execute("""
                INSERT INTO nutrition_cache (
                    product_key, sample_name, macro_category, category, package_weight_g,
                    kcal_100g, proteins_100g, carbs_100g, sugars_100g, fat_100g, saturated_fat_100g,
                    fiber_100g, salt_100g, match_status, off_product_name, data_source, is_verified
                )
                SELECT 
                    s.product_key, s.sample_name, s.macro_category, s.category, s.package_weight_g,
                    s.kcal_100g, s.proteins_100g, s.carbs_100g, s.sugars_100g, s.fat_100g, s.saturated_fat_100g,
                    s.fiber_100g, s.salt_100g, s.match_status, s.off_product_name, 'SEED_CATALOG', s.is_verified
                FROM seed_staging s
                LEFT JOIN nutrition_cache c ON s.product_key = c.product_key
                WHERE c.product_key IS NULL;
            """)

            # 2. Update existing products missing category/nutrition, without overwriting MANUAL_CSV overrides
            con.execute("""
                UPDATE nutrition_cache
                SET
                    sample_name        = COALESCE(NULLIF(TRIM(nutrition_cache.sample_name), ''), s.sample_name),
                    macro_category     = COALESCE(nutrition_cache.macro_category, s.macro_category),
                    category           = COALESCE(nutrition_cache.category, s.category),
                    package_weight_g   = COALESCE(nutrition_cache.package_weight_g, s.package_weight_g),
                    kcal_100g          = COALESCE(nutrition_cache.kcal_100g, s.kcal_100g),
                    proteins_100g      = COALESCE(nutrition_cache.proteins_100g, s.proteins_100g),
                    carbs_100g         = COALESCE(nutrition_cache.carbs_100g, s.carbs_100g),
                    sugars_100g        = COALESCE(nutrition_cache.sugars_100g, s.sugars_100g),
                    fat_100g           = COALESCE(nutrition_cache.fat_100g, s.fat_100g),
                    saturated_fat_100g = COALESCE(nutrition_cache.saturated_fat_100g, s.saturated_fat_100g),
                    fiber_100g         = COALESCE(nutrition_cache.fiber_100g, s.fiber_100g),
                    salt_100g          = COALESCE(nutrition_cache.salt_100g, s.salt_100g),
                    match_status       = CASE WHEN nutrition_cache.kcal_100g IS NULL AND s.kcal_100g IS NOT NULL THEN s.match_status ELSE nutrition_cache.match_status END,
                    is_verified        = CASE WHEN s.is_verified THEN TRUE ELSE nutrition_cache.is_verified END
                FROM seed_staging s
                WHERE nutrition_cache.product_key = s.product_key
                  AND COALESCE(nutrition_cache.data_source, '') != 'MANUAL_CSV'
                  AND COALESCE(nutrition_cache.category, '') != 'Non-Food';
            """)

            con.execute("DROP TABLE IF EXISTS seed_staging;")

        # 3. Cross-Border Seed Pooling: scan other countries' seed catalogs (data/catalogs/*/catalog_seed.csv)
        other_seeds = sorted(
            [p for p in CATALOGS_BASE_DIR.glob("*/catalog_seed.csv") if p.resolve() != SEED_CATALOG_PATH.resolve()],
            key=lambda p: p.parent.name.lower()
        )
        if other_seeds:
            all_cross_tuples = []
            for other_seed in other_seeds:
                source_country = other_seed.parent.name.lower()
                other_rows, _, _ = read_csv_records(other_seed)
                if not other_rows:
                    continue
                _, sibling_to_en = load_translations(source_country)
                for row in other_rows:
                    pk = clean_product_key(row.get("product_key"))
                    if not is_cross_border_eligible(pk):
                        continue
                    sample_name = row.get("sample_name") or ""
                    macro = row.get("macro_category") or None
                    cat = row.get("category") or None
                    if macro in sibling_to_en: macro = sibling_to_en[macro]
                    if cat in sibling_to_en: cat = sibling_to_en[cat]
                    pkg_w = parse_optional_float(row.get("package_weight_g"))
                    kcal = parse_optional_float(row.get("kcal_100g"))
                    prot = parse_optional_float(row.get("proteins_100g"))
                    carbs = parse_optional_float(row.get("carbs_100g"))
                    sugars = parse_optional_float(row.get("sugars_100g"))
                    fat = parse_optional_float(row.get("fat_100g"))
                    sat_fat = parse_optional_float(row.get("saturated_fat_100g"))
                    fiber = parse_optional_float(row.get("fiber_100g"))
                    salt = parse_optional_float(row.get("salt_100g"))
                    non_food = (row.get("non_food") or "").strip().lower() in TRUTHY
                    match_status = "NON_FOOD" if non_food else (row.get("match_status") or "SEED_CROSS_BORDER")
                    off_name = row.get("off_product_name") or None
                    is_ver = True if (kcal is not None or non_food) else False
                    all_cross_tuples.append((
                        pk, source_country, sample_name, macro, cat, pkg_w,
                        kcal, prot, carbs, sugars, fat, sat_fat, fiber, salt,
                        match_status, off_name, is_ver
                    ))

            if all_cross_tuples:
                con.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS cross_seed_staging (
                        product_key VARCHAR, source_country VARCHAR, sample_name VARCHAR,
                        macro_category VARCHAR, category VARCHAR, package_weight_g DOUBLE,
                        kcal_100g DOUBLE, proteins_100g DOUBLE, carbs_100g DOUBLE, sugars_100g DOUBLE,
                        fat_100g DOUBLE, saturated_fat_100g DOUBLE, fiber_100g DOUBLE, salt_100g DOUBLE,
                        match_status VARCHAR, off_product_name VARCHAR, is_verified BOOLEAN
                    );
                """)
                con.execute("TRUNCATE cross_seed_staging;")
                con.executemany("""
                    INSERT INTO cross_seed_staging VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, all_cross_tuples)

                con.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS dedup_cross AS
                    SELECT *
                    FROM cross_seed_staging
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY product_key 
                        ORDER BY (kcal_100g IS NOT NULL) DESC, source_country ASC
                    ) = 1;
                """)

                # 3a. Update existing products missing category or nutrition
                con.execute("""
                    UPDATE nutrition_cache
                    SET
                        macro_category     = COALESCE(nutrition_cache.macro_category, cs.macro_category),
                        category           = COALESCE(nutrition_cache.category, cs.category),
                        package_weight_g   = COALESCE(nutrition_cache.package_weight_g, cs.package_weight_g),
                        kcal_100g          = COALESCE(nutrition_cache.kcal_100g, cs.kcal_100g),
                        proteins_100g      = COALESCE(nutrition_cache.proteins_100g, cs.proteins_100g),
                        carbs_100g         = COALESCE(nutrition_cache.carbs_100g, cs.carbs_100g),
                        sugars_100g        = COALESCE(nutrition_cache.sugars_100g, cs.sugars_100g),
                        fat_100g           = COALESCE(nutrition_cache.fat_100g, cs.fat_100g),
                        saturated_fat_100g = COALESCE(nutrition_cache.saturated_fat_100g, cs.saturated_fat_100g),
                        fiber_100g         = COALESCE(nutrition_cache.fiber_100g, cs.fiber_100g),
                        salt_100g          = COALESCE(nutrition_cache.salt_100g, cs.salt_100g),
                        match_status       = CASE WHEN nutrition_cache.kcal_100g IS NULL AND cs.kcal_100g IS NOT NULL THEN 'SEED_CROSS_BORDER' ELSE nutrition_cache.match_status END,
                        is_verified        = CASE WHEN cs.is_verified THEN TRUE ELSE nutrition_cache.is_verified END
                    FROM dedup_cross cs
                    WHERE nutrition_cache.product_key = cs.product_key
                      AND (nutrition_cache.category IS NULL OR nutrition_cache.kcal_100g IS NULL)
                      AND COALESCE(nutrition_cache.data_source, '') != 'MANUAL_CSV'
                      AND COALESCE(nutrition_cache.category, '') != 'Non-Food';
                """)

                # 3b. Insert new products from purchases matching cross-border seed
                con.execute("""
                    INSERT INTO nutrition_cache (
                        product_key, sample_name, macro_category, category, package_weight_g,
                        kcal_100g, proteins_100g, carbs_100g, sugars_100g, fat_100g, saturated_fat_100g,
                        fiber_100g, salt_100g, match_status, off_product_name, data_source, is_verified
                    )
                    SELECT 
                        cs.product_key, cs.sample_name, cs.macro_category, cs.category, cs.package_weight_g,
                        cs.kcal_100g, cs.proteins_100g, cs.carbs_100g, cs.sugars_100g, cs.fat_100g, cs.saturated_fat_100g,
                        cs.fiber_100g, cs.salt_100g, 'SEED_CROSS_BORDER', cs.off_product_name, 'SEED_CROSS_BORDER', cs.is_verified
                    FROM dedup_cross cs
                    JOIN (
                        SELECT DISTINCT TRIM(CAST(COALESCE(NULLIF(TRIM(ean), ''), raw_code) AS VARCHAR)) AS p_key
                        FROM purchases
                    ) p ON cs.product_key = p.p_key
                    LEFT JOIN nutrition_cache c ON cs.product_key = c.product_key
                    WHERE c.product_key IS NULL;
                """)

                con.execute("DROP TABLE IF EXISTS dedup_cross;")
                con.execute("DROP TABLE IF EXISTS cross_seed_staging;")
                print(f"[+] Applied cross-border seed pooling across {len(other_seeds)} sibling catalogs.")

        con.commit()
        print(f"[+] Loaded/verified {len(seed_tuples)} seed products from '{SEED_CATALOG_PATH}'.")
    except Exception as e:
        con.rollback()
        print(f"[!] Warning applying seed catalog: {e}")


def consume_completed_reviews(con: duckdb.DuckDBPyConnection) -> None:
    """Consume reviewed rows from review_queue.csv, validate taxonomy, update DB, and archive to catalog_overrides.csv."""
    if not REVIEW_QUEUE_PATH.exists():
        return

    rows, delimiter, _ = read_csv_records(REVIEW_QUEUE_PATH)
    if not rows:
        return

    _, category_to_macro = load_taxonomy()

    # Snapshot editable values so the generated queue's prefilled cells are not
    # mistaken for user edits. Only cells that differ from DuckDB are archived.
    cache_map = {}
    cache_fields = ["product_key", "macro_category", "category", *OVERRIDE_NUM_COLUMNS]
    try:
        cache_rows = con.execute(
            f"SELECT {', '.join(cache_fields)} FROM nutrition_cache"
        ).fetchall()
        cache_map = {
            clean_product_key(values[0]): dict(zip(cache_fields[1:], values[1:]))
            for values in cache_rows if clean_product_key(values[0])
        }
    except Exception:
        pass

    active_lang = resolve_active_language(LIDL_COUNTRY)
    _, local_to_en = load_translations(active_lang)
    if active_lang != "ro":
        _, ro_to_en = load_translations("ro")
        for k, v in ro_to_en.items():
            if k not in local_to_en:
                local_to_en[k] = v

    completed_reviews = []
    remaining_queue = []

    for row in rows:
        pk = clean_product_key(row.get("product_key"))
        if not pk:
            continue

        current = cache_map.get(pk, {})
        raw_cat = (row.get("category") or "").strip()
        cat_norm = local_to_en.get(raw_cat, raw_cat)
        is_non_food = (row.get("non_food") or "").strip().lower() in TRUTHY
        changes = {"product_key": pk, "sample_name": row.get("sample_name") or ""}

        # An unchanged prefilled Non-Food marker is not a new human decision.
        if is_non_food and current.get("category") != "Non-Food":
            changes.update({
                "macro_category": "Non-Food",
                "category": "Non-Food",
                "non_food": "TRUE",
            })
        else:
            if raw_cat and cat_norm != current.get("category"):
                if cat_norm not in category_to_macro:
                    print(f"[!] Warning: Category '{raw_cat}' for '{pk}' ({row.get('sample_name')}) is not in taxonomy.json. Kept in review queue.")
                    remaining_queue.append(row)
                    continue
                changes["category"] = cat_norm
                changes["macro_category"] = category_to_macro[cat_norm]
                if cat_norm == "Non-Food":
                    changes["non_food"] = "TRUE"

            for col in OVERRIDE_NUM_COLUMNS:
                raw_value = (row.get(col) or "").strip()
                if not raw_value:
                    continue
                parsed_value = parse_optional_float(raw_value)
                if parsed_value is None:
                    continue
                cached_value = current.get(col)
                if cached_value is None or float(cached_value) != float(parsed_value):
                    changes[col] = raw_value

        changed_fields = set(changes) - {"product_key", "sample_name"}
        if not changed_fields:
            remaining_queue.append(row)
            continue

        # Keep only the explicitly changed cells. This prevents a queue snapshot
        # from turning untouched OFF/seed values into manual overrides.
        review = {column: "" for column in OVERRIDE_COLUMNS}
        review.update(changes)
        completed_reviews.append(review)

    if not completed_reviews:
        return

    print(f"\n--- Consuming {len(completed_reviews)} completed reviews from '{REVIEW_QUEUE_PATH}' ---")

    # 1. Archive to catalog_overrides.csv
    existing_overrides, ovr_delim, _ = read_csv_records(OVERRIDE_CSV_PATH)
    ovr_map = {clean_product_key(r.get("product_key")): r for r in existing_overrides if clean_product_key(r.get("product_key"))}

    for rev in completed_reviews:
        pk = clean_product_key(rev.get("product_key"))
        merged = dict(ovr_map.get(pk, {}))
        merged.update({key: value for key, value in rev.items() if value not in (None, "")})
        ovr_map[pk] = merged

    write_csv_records(OVERRIDE_CSV_PATH, list(ovr_map.values()), OVERRIDE_COLUMNS, ovr_delim)

    # 2. Update review_queue.csv with remaining unreviewed rows
    write_csv_records(REVIEW_QUEUE_PATH, remaining_queue, OVERRIDE_COLUMNS, delimiter)
    print(f"[+] Archived {len(completed_reviews)} items into '{OVERRIDE_CSV_PATH}'. ({len(remaining_queue)} remaining in queue)")


def apply_nutrition_overrides(con: duckdb.DuckDBPyConnection, quiet: bool = False) -> None:
    """Inject manual categories and nutritional values from catalog_overrides.csv into DuckDB."""
    if not quiet:
        print("\n--- STEP 7: Applying User Overrides from 'catalog_overrides.csv' ---")
    con.execute("""
        CREATE TABLE IF NOT EXISTS catalog_overrides (
            product_key VARCHAR PRIMARY KEY,
            macro_category VARCHAR,
            category VARCHAR,
            non_food BOOLEAN DEFAULT FALSE,
            package_weight_g DOUBLE,
            kcal_100g DOUBLE,
            proteins_100g DOUBLE,
            carbs_100g DOUBLE,
            sugars_100g DOUBLE,
            fat_100g DOUBLE,
            saturated_fat_100g DOUBLE,
            fiber_100g DOUBLE,
            salt_100g DOUBLE
        );
    """)
    for col in OVERRIDE_NUM_COLUMNS:
        con.execute(f"ALTER TABLE catalog_overrides ADD COLUMN IF NOT EXISTS {col} DOUBLE;")
    for col in OVERRIDE_TEXT_COLUMNS:
        con.execute(f"ALTER TABLE catalog_overrides ADD COLUMN IF NOT EXISTS {col} VARCHAR;")
    con.execute("ALTER TABLE catalog_overrides ADD COLUMN IF NOT EXISTS non_food BOOLEAN DEFAULT FALSE;")

    if not OVERRIDE_CSV_PATH.exists():
        write_overrides_csv([], ",")
        return

    try:
        rows, _, _ = read_overrides_csv()

        rows_to_insert = []
        seen = set()
        for row in rows:
            pk = clean_product_key(row.get("product_key"))
            if not pk:
                continue
            if SCI_NOTATION_RE.match(pk):
                print(f"[!] product_key '{pk}' appears corrupted by Excel (scientific notation) -> row ignored. "
                      f"Open CSV with product_key as Text.")
                continue
            if pk in seen:
                print(f"[!] Duplicate product_key in '{OVERRIDE_CSV_PATH}': {pk} (keeping last row).")
            seen.add(pk)

            macro = row.get("macro_category") or None
            cat = row.get("category") or None
            non_food = (row.get("non_food") or "").strip().lower() in TRUTHY
            nums = [parse_optional_float(row.get(c)) for c in OVERRIDE_NUM_COLUMNS]

            if non_food or macro or cat or any(v is not None for v in nums):
                rows_to_insert.append((pk, macro, cat, non_food, *nums))

        # Guard: if no valid override rows exist, do not wipe existing table!
        if not rows_to_insert:
            print(f"[i] No valid override data in '{OVERRIDE_CSV_PATH}'. Existing table preserved.")
            return

        # Execute updates inside a safe database transaction
        con.begin()
        try:
            con.execute("TRUNCATE catalog_overrides;")
            con.executemany("""
                INSERT OR REPLACE INTO catalog_overrides (
                    product_key, macro_category, category, non_food,
                    package_weight_g, kcal_100g, proteins_100g, carbs_100g, sugars_100g,
                    fat_100g, saturated_fat_100g, fiber_100g, salt_100g
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, rows_to_insert)

            field_updates_list = []
            nutrition_fields = {
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            }
            for field_name in (OVERRIDE_TEXT_COLUMNS + OVERRIDE_NUM_COLUMNS):
                protected = (
                    "EXISTS (SELECT 1 FROM product_field_metadata metadata "
                    "WHERE metadata.product_key = nutrition_cache.product_key "
                    f"AND metadata.field_name = '{field_name}' AND metadata.is_protected)"
                )
                if field_name in nutrition_fields:
                    protected = (
                        f"({protected} OR nutrition_cache.category = 'Non-Food')"
                    )
                field_updates_list.append(
                    f"{field_name} = CASE WHEN {protected} "
                    f"THEN nutrition_cache.{field_name} "
                    f"ELSE COALESCE(ovr.{field_name}, nutrition_cache.{field_name}) END"
                )
            field_updates = ",\n".join(field_updates_list)
            manual_kcal_allowed = (
                "ovr.kcal_100g IS NOT NULL "
                "AND nutrition_cache.category != 'Non-Food' "
                "AND NOT EXISTS (SELECT 1 FROM product_field_metadata metadata "
                "WHERE metadata.product_key = nutrition_cache.product_key "
                "AND metadata.field_name = 'kcal_100g' AND metadata.is_protected)"
            )
            con.execute(f"""
                UPDATE nutrition_cache
                SET
                    {field_updates},
                    match_status       = CASE WHEN {manual_kcal_allowed} THEN 'ENRICHED_MANUAL' ELSE nutrition_cache.match_status END,
                    data_source        = CASE WHEN {manual_kcal_allowed} THEN 'MANUAL_CSV' ELSE nutrition_cache.data_source END,
                    is_verified        = CASE WHEN {manual_kcal_allowed} THEN TRUE ELSE nutrition_cache.is_verified END
                FROM catalog_overrides ovr
                WHERE nutrition_cache.product_key = ovr.product_key;
            """)

            # Any product whose resulting canonical category is Non-Food must
            # have nutrition cleared. The legacy flag remains a supported way
            # to classify products, but it is not the only source of category.
            con.execute("""
                UPDATE nutrition_cache
                SET
                    macro_category     = 'Non-Food',
                    category           = 'Non-Food',
                    match_status       = 'NON_FOOD',
                    data_source        = 'RULE_NON_FOOD',
                    kcal_100g          = NULL,
                    proteins_100g      = NULL,
                    carbs_100g         = NULL,
                    sugars_100g        = NULL,
                    fat_100g           = NULL,
                    saturated_fat_100g = NULL,
                    fiber_100g         = NULL,
                    salt_100g          = NULL,
                    is_weighed_default = FALSE,
                    is_verified        = TRUE
                WHERE nutrition_cache.category = 'Non-Food'
                   OR (
                    product_key IN (SELECT product_key FROM catalog_overrides WHERE non_food)
                    AND NOT EXISTS (
                      SELECT 1 FROM product_field_metadata metadata
                      WHERE metadata.product_key = nutrition_cache.product_key
                        AND metadata.field_name = 'category'
                        AND metadata.is_protected
                    )
                  );
            """)
            con.execute("""
                DELETE FROM product_field_metadata
                WHERE field_name IN (
                    'kcal_100g', 'proteins_100g', 'carbs_100g', 'sugars_100g',
                    'fat_100g', 'saturated_fat_100g', 'fiber_100g', 'salt_100g'
                )
                  AND product_key IN (
                      SELECT product_key FROM nutrition_cache WHERE category = 'Non-Food'
                  )
            """)
            con.commit()
            print(f"[+] Successfully injected {len(rows_to_insert)} rows from '{OVERRIDE_CSV_PATH}'.")
        except Exception as e:
            con.rollback()
            raise e
    except Exception as e:
        print(f"[!] Warning processing '{OVERRIDE_CSV_PATH}': {e}.")


def warn_suspicious_overrides(con: duckdb.DuckDBPyConnection) -> None:
    """Check manually entered override values for potential consistency issues (warning only)."""
    rows = con.execute("""
        SELECT o.product_key, COALESCE(c.sample_name, ''), COALESCE(c.category, ''),
               o.kcal_100g, o.proteins_100g, o.carbs_100g, o.sugars_100g, o.fat_100g,
               o.saturated_fat_100g, o.fiber_100g, o.salt_100g
        FROM catalog_overrides o
        LEFT JOIN nutrition_cache c ON c.product_key = o.product_key
        WHERE o.kcal_100g IS NOT NULL
          AND COALESCE(c.match_status, '') != 'NON_FOOD'
        ORDER BY o.product_key
    """).fetchall()

    warnings = []
    for pk, name, cat, kcal, prot, carbs, sug, fat, sat, fib, salt in rows:
        issues = []
        if None in (prot, carbs, fat):
            issues.append("missing protein/carbs/fat")
        else:
            atw = prot * 4 + carbs * 4 + fat * 9 + (fib or 0) * 2
            if cat not in ("Alcoholic Beverages", "Băuturi Alcoolice") and abs(kcal - atw) > max(25.0, kcal * 0.20):
                if atw >= 15 and 3.8 <= kcal / atw <= 4.6:
                    issues.append(f"kcal={kcal:g} appears to be kJ (≈{kcal / 4.184:.0f} kcal; macros total {atw:.0f})")
                else:
                    issues.append(f"kcal={kcal:g} vs macros={atw:.0f}")
            if prot + carbs + fat + (fib or 0) + (salt or 0) > 105:
                issues.append("protein+carbs+fat+fiber+salt > 105 g/100g")
            if sug is not None and sug > carbs + 1:
                issues.append(f"sugars {sug:g} > carbs {carbs:g}")
            if sat is not None and sat > fat + 1:
                issues.append(f"saturated fat {sat:g} > fat {fat:g}")
        if kcal > 900 and not any("kJ" in i for i in issues):
            issues.append(f"kcal={kcal:g} > 900 (kJ instead of kcal?)")
        if issues:
            warnings.append(f"    {pk} | {name[:40]} | " + "; ".join(issues))

    if warnings:
        print(f"[!] {len(warnings)} manual rows with suspicious values (warning only, no data modified):")
        for w in warnings[:30]:
            print(w)
        if len(warnings) > 30:
            print(f"    ... {len(warnings) - 30} more")


# =====================================================================
# 7. ATWATER VALIDATION & QUARANTINE (NUTRITION QA)
# =====================================================================
def validate_atwater(con: duckdb.DuckDBPyConnection) -> None:
    """Validate caloric values from Open Food Facts against standard Atwater factor calculation."""
    print("\n--- STEP 8: Atwater Validation & Caloric QA for New Products ---")

    nutrient_fields = (
        "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
        "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
    )
    protected_nutrient_updates = ",\n".join(
        f"{field_name} = CASE WHEN ev.validation_verdict = 'VALID' "
        "OR EXISTS (SELECT 1 FROM product_field_metadata metadata "
        "WHERE metadata.product_key = nutrition_cache.product_key "
        f"AND metadata.field_name = '{field_name}' AND metadata.is_protected) "
        f"THEN nutrition_cache.{field_name} ELSE NULL END"
        for field_name in nutrient_fields
    )
    con.execute(f"""
        WITH evaluation AS (
            SELECT
                product_key,
                category,
                kcal_100g,
                proteins_100g,
                carbs_100g,
                sugars_100g,
                fat_100g,
                saturated_fat_100g,
                COALESCE(fiber_100g, 0.0) AS fib,
                COALESCE(salt_100g, 0.0) AS salt,
                ROUND(
                    (COALESCE(proteins_100g, 0.0) * 4.0) +
                    (COALESCE(carbs_100g, 0.0) * 4.0) +
                    (COALESCE(fat_100g, 0.0) * 9.0) +
                    (COALESCE(fiber_100g, 0.0) * 2.0),
                    1
                ) AS kcal_atwater,
                CASE
                    WHEN kcal_100g IS NULL OR proteins_100g IS NULL OR carbs_100g IS NULL OR fat_100g IS NULL
                        THEN 'MISSING_MACROS'
                    WHEN kcal_100g < 0 OR proteins_100g < 0 OR carbs_100g < 0 OR fat_100g < 0
                        THEN 'NEGATIVE_VALUES'
                    WHEN (proteins_100g + carbs_100g + fat_100g + COALESCE(fiber_100g, 0.0) + COALESCE(salt_100g, 0.0)) > 105.0
                        THEN 'MASS_OVER_105G'
                    WHEN COALESCE(sugars_100g, 0.0) > (carbs_100g + 1.0)
                        THEN 'SUGARS_GT_CARBS'
                    WHEN COALESCE(saturated_fat_100g, 0.0) > (fat_100g + 1.0)
                        THEN 'SATFAT_GT_FAT'
                    WHEN category = 'Alcoholic Beverages'
                        THEN 'VALID'
                    WHEN ABS(kcal_100g - ((proteins_100g * 4.0) + (carbs_100g * 4.0) + (fat_100g * 9.0) + (COALESCE(fiber_100g, 0.0) * 2.0)))
                         > GREATEST(25.0, kcal_100g * 0.20)
                        THEN 'ATWATER_MISMATCH'
                    ELSE 'VALID'
                END AS validation_verdict
            FROM nutrition_cache
            WHERE is_verified = FALSE
              AND COALESCE(match_status, '') != 'NON_FOOD'
              AND kcal_100g IS NOT NULL
              AND proteins_100g IS NOT NULL
              AND carbs_100g IS NOT NULL
              AND fat_100g IS NOT NULL
        )
        UPDATE nutrition_cache
        SET
            atwater_debug = CASE
                WHEN ev.validation_verdict = 'VALID' AND ev.category = 'Alcoholic Beverages'
                    THEN 'OK (Alcoholic Beverages - exempt from Atwater)'
                WHEN ev.validation_verdict = 'VALID' THEN 'OK (Atwater=' || CAST(ev.kcal_atwater AS VARCHAR) || ')'
                WHEN ev.validation_verdict = 'MISSING_MACROS' THEN 'Missing macros in OFF'
                ELSE ev.validation_verdict || ': OFF_kcal=' || COALESCE(CAST(ev.kcal_100g AS VARCHAR), 'NULL')
                     || ' vs Atwater=' || CAST(ev.kcal_atwater AS VARCHAR)
            END,
            {protected_nutrient_updates},
            match_status       = CASE
                WHEN nutrition_cache.match_status = 'NOT_FOUND_OFF' THEN 'NOT_FOUND_OFF'
                WHEN ev.validation_verdict = 'VALID' THEN 'MATCHED_OFF_VERIFIED'
                WHEN ev.validation_verdict = 'MISSING_MACROS' THEN 'QUARANTINE_MISSING'
                ELSE 'QUARANTINE_ATWATER_FAIL'
            END,
            is_verified        = CASE
                WHEN ev.validation_verdict = 'VALID' THEN TRUE
                ELSE FALSE
            END
        FROM evaluation ev
        WHERE nutrition_cache.product_key = ev.product_key;
    """)

    quarantined_row = con.execute("""
        SELECT COUNT(*)
        FROM nutrition_cache
        WHERE is_verified = FALSE
          AND COALESCE(match_status, '') != 'NON_FOOD'
          AND kcal_100g IS NULL
    """).fetchone()
    if quarantined_row is None:
        raise RuntimeError("Query for unverified food products returned no result.")
    quarantined = quarantined_row[0]
    print(f"[i] Food products without valid nutritional values: {quarantined}.")


def update_review_queue(con: duckdb.DuckDBPyConnection) -> None:
    """Regenerate the full review worklist from DuckDB's required fields."""
    from product_store import regenerate_review_csv

    print("\n--- STEP 9: Updating Pending Review Queue ('review_queue.csv') ---")
    _, category_to_macro = load_taxonomy()
    if not category_to_macro:
        raise RuntimeError(f"Cannot regenerate review queue without a valid taxonomy: {TAXONOMY_PATH}")

    try:
        count = regenerate_review_csv(
            con,
            category_to_macro,
            REVIEW_QUEUE_PATH,
            OVERRIDE_COLUMNS,
        )
        if count:
            print(f"[!] {count} products awaiting review in '{REVIEW_QUEUE_PATH}'.")
        else:
            print("[+] All products meet required food-data completeness. Review queue is clean.")
    except (PermissionError, OSError) as e:
        print(f"[!] Cannot write '{REVIEW_QUEUE_PATH}' (close it in Excel and re-run): {e}")


# =====================================================================
# 8. PACKAGE WEIGHT DECODING (BULK VS 25... BARCODES)
# =====================================================================
def compute_weights_and_quantities(con: duckdb.DuckDBPyConnection) -> None:
    """Decode piece weights from Lidl variable-weight barcodes, bulk sales, and regex."""
    print("\n--- STEP 6: Decoding Package Weights & Quantities ---")

    # Barcodes starting with '25' have piece weight encoded in digits 8-12
    con.execute("""
        UPDATE nutrition_cache
        SET 
            package_weight_g   = CAST(SUBSTRING(product_key, 8, 5) AS DOUBLE),
            is_weighed_default = FALSE
        WHERE product_key LIKE '25%' 
          AND LENGTH(product_key) = 13
          AND NOT EXISTS (
              SELECT 1 FROM product_field_metadata metadata
              WHERE metadata.product_key = nutrition_cache.product_key
                AND metadata.field_name = 'package_weight_g'
                AND metadata.is_protected
          );
    """)

    # Inherit category and nutrition for variable-weight items (prefix 25...) based on first 7 characters
    con.execute("CREATE TABLE IF NOT EXISTS catalog_overrides (product_key VARCHAR PRIMARY KEY);")
    for col in OVERRIDE_NUM_COLUMNS:
        con.execute(f"ALTER TABLE catalog_overrides ADD COLUMN IF NOT EXISTS {col} DOUBLE;")
    for col in OVERRIDE_TEXT_COLUMNS:
        con.execute(f"ALTER TABLE catalog_overrides ADD COLUMN IF NOT EXISTS {col} VARCHAR;")

    con.execute("""
        WITH prefix_sources AS (
            SELECT 
                SUBSTRING(product_key, 1, 7) AS pfx7,
                macro_category, category, kcal_100g, proteins_100g, carbs_100g, sugars_100g,
                fat_100g, saturated_fat_100g, fiber_100g, salt_100g
            FROM (
                SELECT product_key, macro_category, category, kcal_100g, proteins_100g, carbs_100g, sugars_100g,
                       fat_100g, saturated_fat_100g, fiber_100g, salt_100g, 1 AS prio
                FROM catalog_overrides
                WHERE macro_category IS NOT NULL
                UNION ALL
                SELECT product_key, macro_category, category, kcal_100g, proteins_100g, carbs_100g, sugars_100g,
                       fat_100g, saturated_fat_100g, fiber_100g, salt_100g, 2 AS prio
                FROM nutrition_cache
                WHERE is_verified = TRUE AND macro_category IS NOT NULL
            ) combined
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY SUBSTRING(product_key, 1, 7)
                ORDER BY prio ASC, (kcal_100g IS NOT NULL) DESC
            ) = 1
        )
        UPDATE nutrition_cache
        SET
            macro_category     = COALESCE(nutrition_cache.macro_category, s.macro_category),
            category           = COALESCE(nutrition_cache.category, s.category),
            kcal_100g          = COALESCE(nutrition_cache.kcal_100g, s.kcal_100g),
            proteins_100g      = COALESCE(nutrition_cache.proteins_100g, s.proteins_100g),
            carbs_100g         = COALESCE(nutrition_cache.carbs_100g, s.carbs_100g),
            sugars_100g        = COALESCE(nutrition_cache.sugars_100g, s.sugars_100g),
            fat_100g           = COALESCE(nutrition_cache.fat_100g, s.fat_100g),
            saturated_fat_100g = COALESCE(nutrition_cache.saturated_fat_100g, s.saturated_fat_100g),
            fiber_100g         = COALESCE(nutrition_cache.fiber_100g, s.fiber_100g),
            salt_100g          = COALESCE(nutrition_cache.salt_100g, s.salt_100g),
            match_status       = CASE WHEN s.macro_category IS NOT NULL THEN 'INHERITED_PREFIX' ELSE nutrition_cache.match_status END,
            data_source        = CASE WHEN s.macro_category IS NOT NULL THEN 'PREFIX_INHERIT' ELSE nutrition_cache.data_source END,
            is_verified        = CASE WHEN s.macro_category IS NOT NULL AND s.kcal_100g IS NOT NULL THEN TRUE ELSE nutrition_cache.is_verified END
        FROM prefix_sources s
        WHERE nutrition_cache.product_key LIKE '25%' 
          AND LENGTH(nutrition_cache.product_key) = 13
          AND SUBSTRING(nutrition_cache.product_key, 1, 7) = s.pfx7
          AND COALESCE(nutrition_cache.category, '') != 'Non-Food'
          AND (nutrition_cache.macro_category IS NULL OR nutrition_cache.kcal_100g IS NULL);
    """)

    # Mark bulk/weighed produce items
    con.execute("""
        UPDATE nutrition_cache
        SET 
            is_weighed_default = TRUE,
            package_weight_g   = NULL
        WHERE ((product_key LIKE '50%' AND LENGTH(product_key) <= 5)
           OR (
                product_key IN (
                    SELECT DISTINCT TRIM(CAST(COALESCE(NULLIF(TRIM(ean), ''), raw_code) AS VARCHAR))
                    FROM purchases
                    WHERE (quantity - FLOOR(quantity)) > 0
                )
                AND NOT (product_key LIKE '25%' AND LENGTH(product_key) = 13)
           ))
          AND NOT EXISTS (
              SELECT 1 FROM product_field_metadata metadata
              WHERE metadata.product_key = nutrition_cache.product_key
                AND metadata.field_name = 'package_weight_g'
                AND metadata.is_protected
          );
    """)

    # Parse weights from product description text
    rows_regex = con.execute("""
        SELECT product_key, sample_name 
        FROM nutrition_cache 
        WHERE package_weight_g IS NULL AND is_weighed_default = FALSE
          AND NOT EXISTS (
              SELECT 1 FROM product_field_metadata metadata
              WHERE metadata.product_key = nutrition_cache.product_key
                AND metadata.field_name = 'package_weight_g'
                AND metadata.is_protected
          )
    """).fetchall()

    regex_updates = []
    for pk, name in rows_regex:
        w = parse_weight_from_text(name)
        if w and w > 0:
            regex_updates.append((w, pk))

    if regex_updates:
        con.executemany("""
            UPDATE nutrition_cache 
            SET package_weight_g = ? 
            WHERE product_key = ?;
        """, regex_updates)


# =====================================================================
# 9. PARQUET EXPORT FOR POWER BI
# =====================================================================
def export_views_for_bi(con: duckdb.DuckDBPyConnection) -> None:
    """Export optimized ZSTD-compressed Parquet files for Power BI reporting."""
    print("\n--- STEP 10: Exporting Parquet Files for Power BI ---")

    # Populate localized category names via active language mapping (with canonical fallback)
    try:
        active_lang = resolve_active_language(LIDL_COUNTRY)
        en_to_local, _ = load_translations(active_lang)
        con.execute("CREATE TEMP TABLE IF NOT EXISTS tax_en_to_local (en_val VARCHAR PRIMARY KEY, local_val VARCHAR);")
        con.execute("TRUNCATE tax_en_to_local;")
        if en_to_local:
            con.executemany("INSERT INTO tax_en_to_local VALUES (?, ?);", list(en_to_local.items()))

        con.execute(f"""
            COPY (
                SELECT 
                    c.product_key,
                    COALESCE(c.sample_name, sub.p_name, '') AS store_product_name,
                    COALESCE(c.sample_name, sub.p_name, '') AS final_product_name,
                    COALESCE(NULLIF(TRIM(c.macro_category), ''), 'Uncategorized') AS macro_category,
                    COALESCE(NULLIF(TRIM(c.category), ''), 'Pending Review') AS category,
                    COALESCE(m_loc.local_val, NULLIF(TRIM(c.macro_category), ''), 'Uncategorized') AS macro_category_local,
                    COALESCE(c_loc.local_val, NULLIF(TRIM(c.category), ''), 'Pending Review') AS category_local,
                    c.kcal_100g,
                    c.proteins_100g,
                    c.carbs_100g,
                    c.sugars_100g,
                    c.fat_100g,
                    c.saturated_fat_100g,
                    c.fiber_100g,
                    c.salt_100g,
                    c.package_weight_g,
                    c.is_weighed_default,
                    c.nutriscore_score,
                    c.nutriscore_grade,
                    c.match_status,
                    c.data_source
                FROM nutrition_cache c
                LEFT JOIN (
                    SELECT 
                        TRIM(CAST(COALESCE(NULLIF(TRIM(ean), ''), raw_code) AS VARCHAR)) AS p_key,
                        MIN(TRIM(name)) AS p_name
                    FROM purchases 
                    GROUP BY 1
                ) sub ON c.product_key = sub.p_key
                LEFT JOIN tax_en_to_local m_loc ON TRIM(c.macro_category) = m_loc.en_val
                LEFT JOIN tax_en_to_local c_loc ON TRIM(c.category) = c_loc.en_val
            ) TO '{EXPORT_CATALOG_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
    finally:
        con.execute("DROP TABLE IF EXISTS tax_en_to_local;")

    con.execute(f"""
        COPY (
            SELECT 
                p.ticket_id,
                p.date,
                p.line_idx,
                TRIM(CAST(COALESCE(NULLIF(TRIM(p.ean), ''), raw_code) AS VARCHAR)) AS product_key,
                p.quantity,
                p.unit_price,
                p.line_total,
                p.discount_amount,
                p.net_line_total,
                p.ean,
                p.raw_code,
                p.barcode_type
            FROM purchases p
            ORDER BY p.date DESC, p.ticket_id, p.line_idx ASC
        ) TO '{EXPORT_TRANSACTIONS_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    total_fact_row = con.execute("SELECT COUNT(*) FROM purchases").fetchone()
    if total_fact_row is None:
        raise RuntimeError("Query for total transactions returned no result.")
    total_fact = total_fact_row[0]

    total_disc_row = con.execute("SELECT SUM(discount_amount) FROM purchases").fetchone()
    if total_disc_row is None:
        raise RuntimeError("Query for total discounts returned no result.")
    total_disc = total_disc_row[0] or 0.0

    print("=" * 65)
    print("FINAL SUMMARY:")
    print(f"Total transaction rows:          {total_fact}")
    print(f"Total discounts identified:      {total_disc:.2f} RON")
    print(f"[+] '{EXPORT_TRANSACTIONS_PATH}' (net total and discount columns)")
    print(f"[+] '{EXPORT_CATALOG_PATH}' (Atwater-verified nutrition)")
    print("=" * 65)


# =====================================================================
# 10. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    from pipeline_services import PipelineBusyError, run_sync

    try:
        run_sync()
    except PipelineBusyError as exc:
        print(f"[!] {exc}")
        raise SystemExit(2) from None
