import csv
import json
import os
from pathlib import Path
import duckdb

import sync_incremental as sync

SEED_PATH = Path("data/catalogs/ro/catalog_seed.csv")
TAXONOMY_PATH = Path("data/catalogs/taxonomy.json")
DB_PATH = Path("lidl_consumption.duckdb")

# 1. Load canonical taxonomy directly from taxonomy.json (Single Source of Truth)
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    tax_data = json.load(f)

CATEGORY_TO_MACRO = {}
for macro, cats in tax_data.items():
    for c in cats:
        CATEGORY_TO_MACRO[c] = macro


def get_corrected_category(pk: str, name: str, current_cat: str) -> str:
    n = name.lower()

    # Poarta 1: Non-Food
    if any(k in n for k in ["vopsea 10 oua", "2 saculeti reutilizabili", "figurina disney"]):
        return "Non-Food"

    # Poarta 2: Patiserie ambalată cu EAN vs. Brutărie caldă PLU
    if current_cat == "Fresh Bakery & Pastry" and len(pk) > 6 and not pk.startswith("50"):
        if "saleuri" in n:
            return "Biscuits & Savory Snacks"
        return "Desserts & Pastries"

    if pk == "50980" or n == "rulou cu mere":
        return "Fresh Bakery & Pastry"

    # Poarta 3: Aluat focaccia
    if "aluat pentru focaccia" in n:
        return "Baking & Cooking Ingredients"

    # Poarta 4: Conserve de pește/fructe rătăcite în băuturi
    if current_cat == "Water & Soft Drinks" and any(k in n for k in ["ton ", "ton bucati", "ton maruntit", "file de ton", "ananas bucati", "cocktail fructe"]):
        return "Canned Goods & Pickles"

    # Poarta 5: Lactate poluate
    if current_cat == "Dairy & Eggs":
        if "pringles" in n:
            return "Biscuits & Savory Snacks"
        if "hellmann" in n:
            return "Sauces, Oils & Condiments"
        if any(k in n for k in ["pappardelle", "paste proaspete"]):
            return "Baking & Cooking Ingredients"
        if "legume cu unt" in n:
            return "Frozen Fruits & Vegetables"
        if any(k in n for k in ["gofra", "prajituri cu crema"]):
            return "Desserts & Pastries"
        if any(k in n for k in ["dots baton", "batoane cioc"]):
            return "Chocolate & Confectionery"

    # Poarta 6: Fructe & Legume proaspete poluate
    if current_cat == "Fresh Fruits & Vegetables":
        if "bake rolls" in n:
            return "Biscuits & Savory Snacks"
        if any(k in n for k in ["radler", "cola", "schweppes", "ursus cooler", "bundaberg", "nectar "]):
            return "Water & Soft Drinks"
        if any(k in n for k in ["boia", "usturoi granulat", "dressing"]):
            return "Sauces, Oils & Condiments"
        if any(k in n for k in ["compot", "zacusca", "kimchi"]):
            return "Canned Goods & Pickles"
        if any(k in n for k in ["baklava", "galuste", "souffle"]):
            return "Desserts & Pastries"

    # Poarta 7: Carne crudă de pui în Mezeluri
    if current_cat == "Processed Meat & Deli" and any(k in n for k in ["muschiulet piept", "muschiulet pui", "buc_muschiulet"]):
        return "Fresh Meat & Fish"

    # Poarta 8: Uleiuri și făinoase în Conserve
    if current_cat == "Canned Goods & Pickles":
        if any(k in n for k in ["ulei masline", "pm_ulei"]):
            return "Sauces, Oils & Condiments"
        if "fussili din linte" in n:
            return "Baking & Cooking Ingredients"
        if "snacks din porumb" in n:
            return "Biscuits & Savory Snacks"

    # Poarta 9: Zahăr tos la pungă în Dulciuri
    if current_cat == "Chocolate & Confectionery" and any(k in n for k in ["zahar bod", "delmario zahar", "furnicuta zahar"]):
        return "Baking & Cooking Ingredients"

    # Poarta 10: Pește congelat în Pizza
    if current_cat == "Frozen Ready Meals & Pizza" and any(k in n for k in ["peste in aluat", "cod", "somon", "creveti", "sprot", "fructe de mare", "batoane de peste"]):
        return "Frozen Fish & Seafood"

    # Poarta 11: Produse rătăcite în Ready Meals & Chilled Convenience
    if current_cat == "Ready Meals & Chilled Convenience":
        if any(k in n for k in ["chardonnay", "amaretto", "hugo", "nero d'avola", "pinot grigio", "sauvignon", "secco rosato", "purcari", "primitivo", "wembley", "pm_bob cu bob", "villa paradiso", "terra romana"]):
            return "Alcoholic Beverages"
        if any(k in n for k in ["baut. rac", "bautura aloe", "bautura bio", "bautura carb", "bautura proteica", "ciuc radler", "fever tree", "high protein clear", "smoothie", "socata", "ursus cooler", "pm_bautura"]):
            return "Water & Soft Drinks"
        if any(k in n for k in ["chilli macinat", "cimbru", "cosmin condim", "dr. oetker esenta", "knorr cub", "oregano maruntit", "rasnita condim", "marmelada", "amestec acid acetic"]):
            return "Sauces, Oils & Condiments"
        if any(k in n for k in ["bicarbonat", "cacao pudra", "bors, 1l", "indulcitor pastile", "malita soia"]):
            return "Baking & Cooking Ingredients"
        if any(k in n for k in ["capricii & delicii zac"]):
            return "Canned Goods & Pickles"
        if any(k in n for k in ["chipsuri orez", "covrigei cu caramel", "lays wavy", "takis", "pringles", "rondele proteice"]):
            return "Biscuits & Savory Snacks"
        if any(k in n for k in ["eugenia", "ferrero collect", "kinder maxi", "m m's", "ritter sport", "mochi in forma", "turta dulce in cutie", "patratele cioco", "pm_guma", "pm_orbit"]):
            return "Chocolate & Confectionery"
        if any(k in n for k in ["creme de la creme", "oreo brownie", "rulouri vanilie", "waffle", "fructe crocante", "patiserie cu cioco"]):
            return "Desserts & Pastries"
        if any(k in n for k in ["dovleac", "kaki", "leurda", "leustean", "rodii"]):
            return "Fresh Fruits & Vegetables"
        if any(k in n for k in ["mini pave", "painea campionilor", "rulou mexican"]):
            return "Fresh Bakery & Pastry"
        if any(k in n for k in ["parm. reggiano", "skyr natur"]):
            return "Dairy & Eggs"

    return current_cat


def run_patch():
    print(f"[i] Reading seed catalog: {SEED_PATH}")
    rows, delimiter, header = sync.read_csv_records(SEED_PATH)

    patched_count = 0
    updated_rows = []

    for r in rows:
        pk = sync.clean_product_key(r.get("product_key"))
        name = r.get("sample_name") or ""
        old_cat = r.get("category") or ""
        old_macro = r.get("macro_category") or ""

        new_cat = get_corrected_category(pk, name, old_cat)
        new_macro = CATEGORY_TO_MACRO.get(new_cat, old_macro)

        if new_cat != old_cat or new_macro != old_macro:
            patched_count += 1
            r["category"] = new_cat
            r["macro_category"] = new_macro
            if new_cat == "Non-Food":
                r["non_food"] = "TRUE"
                r["match_status"] = "NON_FOOD"

        # Validate that category exists in taxonomy.json
        if new_cat not in CATEGORY_TO_MACRO:
            raise ValueError(f"Category '{new_cat}' for product {pk} ({name}) not found in taxonomy.json!")

        updated_rows.append(r)

    print(f"[+] Total patched products: {patched_count} / {len(rows)}")

    # Atomically write back to catalog_seed.csv using official sync helper
    sync.write_csv_records(SEED_PATH, updated_rows, sync.OVERRIDE_COLUMNS)
    print(f"[+] Successfully wrote clean seed catalog to '{SEED_PATH}'.")

    # Sync DuckDB and export Parquet via sync_incremental official functions
    if DB_PATH.exists():
        con = duckdb.connect(str(DB_PATH))
        try:
            print(f"[i] Syncing 'nutrition_cache' table in '{DB_PATH}'...")
            con.execute("CREATE TEMP TABLE IF NOT EXISTS temp_patch (product_key VARCHAR PRIMARY KEY, macro_category VARCHAR, category VARCHAR, non_food VARCHAR);")
            con.execute("TRUNCATE temp_patch;")

            patch_tuples = [
                (sync.clean_product_key(r.get("product_key")), r.get("macro_category"), r.get("category"), r.get("non_food") or "")
                for r in updated_rows
            ]
            con.executemany("INSERT INTO temp_patch VALUES (?, ?, ?, ?);", patch_tuples)

            con.execute("""
                UPDATE nutrition_cache
                SET 
                    macro_category = p.macro_category,
                    category = p.category,
                    match_status = CASE WHEN p.non_food = 'TRUE' THEN 'NON_FOOD' ELSE nutrition_cache.match_status END
                FROM temp_patch p
                WHERE nutrition_cache.product_key = p.product_key
                  AND COALESCE(nutrition_cache.data_source, '') != 'MANUAL_CSV';
            """)
            con.execute("DROP TABLE IF EXISTS temp_patch;")
            print("[+] Updated 'nutrition_cache' successfully (respected manual overrides).")

            # Official Parquet export with localized agnostic columns
            sync.export_views_for_bi(con)
            print("[+] Successfully exported Parquet views with canonical and localized columns.")
        finally:
            con.close()
    else:
        print(f"[!] '{DB_PATH}' does not exist locally; seed file was cleaned.")


if __name__ == "__main__":
    run_patch()

