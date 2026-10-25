"""Shared constants for the offline AI enrichment evaluation harness.

Nothing in this module talks to a network or to the production pipeline.
The canonical taxonomy itself is *not* duplicated here; it is loaded at
runtime from ``data/catalogs/taxonomy.json`` (see :mod:`ai_eval.taxonomy`).
"""

from __future__ import annotations

# Nutrition fields requested from a model, in stable output order.
NUTRITION_FIELDS = (
    "kcal_100g",
    "proteins_100g",
    "carbs_100g",
    "sugars_100g",
    "fat_100g",
    "saturated_fat_100g",
    "fiber_100g",
    "salt_100g",
)

# Fields required for an Atwater consistency check / reference comparison.
REQUIRED_NUTRITION_FIELDS = (
    "kcal_100g",
    "proteins_100g",
    "carbs_100g",
    "fat_100g",
)

# Inclusive allowed ranges for model output.
NUTRITION_RANGES = {
    "kcal_100g": (0.0, 900.0),
    "proteins_100g": (0.0, 100.0),
    "carbs_100g": (0.0, 100.0),
    "sugars_100g": (0.0, 100.0),
    "fat_100g": (0.0, 100.0),
    "saturated_fat_100g": (0.0, 100.0),
    "fiber_100g": (0.0, 100.0),
    "salt_100g": (0.0, 100.0),
}

# Confidence tiers derived from the seed's match_status column.
HIGH_CONFIDENCE_STATUSES = frozenset(
    {
        "MANUAL_CORRECTED",
        "PATCHED_EXACT",
        "MATCHED_OFF_VERIFIED",
        "NON_FOOD",
    }
)
MEDIUM_CONFIDENCE_STATUSES = frozenset(
    {
        "ENRICHED_MANUAL",
        "MATCHED_OFF",
        "SEED_CATALOG",
    }
)
CONFIDENCE_TIERS = ("HIGH", "MEDIUM", "LOW")

# Nutrition QA thresholds, matching the production pipeline's philosophy
# (sync_incremental.validate_atwater) without importing production code.
MAX_MACRO_SUM_G = 105.0
SUGARS_TOLERANCE_G = 1.0
SATURATED_FAT_TOLERANCE_G = 1.0
ATWATER_ABS_TOLERANCE_KCAL = 25.0
ATWATER_REL_TOLERANCE = 0.20
ALCOHOL_CATEGORY = "Alcoholic Beverages"
NON_FOOD_CATEGORY = "Non-Food"

NUTRITION_MODES = ("none", "estimate")
PROVIDER_NAMES = ("groq", "openrouter")

# A clean abstention must carry a short, human-readable reason. The limit only
# bounds the reason text; it never turns an abstention into an answered record.
MAX_ABSTAIN_REASON_LENGTH = 200
