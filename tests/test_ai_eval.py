"""Offline tests for the isolated AI evaluation harness.

No real network calls are made: transports are faked and dry-run tests patch
``requests`` to fail loudly if touched.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import socket
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import patch

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_eval.constants import (  # noqa: E402
    MAX_ABSTAIN_REASON_LENGTH,
    NUTRITION_FIELDS,
    NON_FOOD_CATEGORY,
)
from ai_eval.dataset import (  # noqa: E402
    DatasetError,
    EvalDataset,
    EvalProduct,
    _to_float,
    _to_int,
    build_eval_dataset,
    confidence_for,
    is_nutrition_comparable,
)
from ai_eval.parsing import (  # noqa: E402
    ParseError,
    parse_model_output,
    validate_response,
)
from ai_eval.prompts import build_messages, build_prompt  # noqa: E402
from ai_eval.providers import (  # noqa: E402
    GROQ_FREE_MODELS,
    MAX_ATTEMPTS_LIMIT,
    FreeModelError,
    GroqClient,
    OpenRouterClient,
    ProviderResponse,
    get_api_key,
    get_setting,
    list_openrouter_free_models,
    parse_env_file,
    redact_text,
    validate_groq_model,
    validate_openrouter_model,
)
from ai_eval.reporting import (  # noqa: E402
    append_results,
    build_summary,
    completed_keys,
    latest_results,
    load_results,
    render_report,
    sanitize_value,
    write_json,
    write_text,
)
from ai_eval.taxonomy import (  # noqa: E402
    TaxonomyError,
    load_taxonomy,
)

from ai_eval import runner as runner_mod  # noqa: E402


LIVE_TAXONOMY = load_taxonomy()


def make_item(key: str, category: str, **nutrition: object) -> dict[str, object]:
    item: dict[str, object] = {"product_key": key, "category": category}
    for field_name in NUTRITION_FIELDS:
        item[field_name] = nutrition.get(field_name)
    return item


def make_payload(*items: dict[str, object]) -> dict[str, object]:
    return {"products": list(items)}


class TestTaxonomy(unittest.TestCase):
    def test_live_taxonomy_shape(self):
        self.assertEqual(len(LIVE_TAXONOMY.macros), 7)
        self.assertEqual(len(LIVE_TAXONOMY.categories), 21)
        self.assertIn(NON_FOOD_CATEGORY, LIVE_TAXONOMY.categories)
        self.assertEqual(LIVE_TAXONOMY.macro_for(NON_FOOD_CATEGORY), "Non-Food")
        self.assertEqual(
            len(set(LIVE_TAXONOMY.categories)), len(LIVE_TAXONOMY.categories)
        )
        self.assertEqual(LIVE_TAXONOMY.category_to_macro["Dairy & Eggs"], "Fresh Food")
        self.assertIn("Alcoholic Beverages", LIVE_TAXONOMY.categories)

    def test_prompt_lines_include_every_category(self):
        rendered = LIVE_TAXONOMY.prompt_lines()
        for category in LIVE_TAXONOMY.categories:
            self.assertIn(category, rendered)

    def test_duplicate_category_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taxonomy.json"
            path.write_text(
                json.dumps({"A": ["X", "X"], "B": ["Y"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(TaxonomyError, "Duplicate category"):
                load_taxonomy(path)

    def test_malformed_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taxonomy.json"
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaisesRegex(TaxonomyError, "Malformed taxonomy JSON"):
                load_taxonomy(path)

    def test_empty_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taxonomy.json"
            path.write_text(json.dumps({"A": ["  "]}), encoding="utf-8")
            with self.assertRaisesRegex(TaxonomyError, "non-empty string"):
                load_taxonomy(path)
            path.write_text(json.dumps({"": ["X"]}), encoding="utf-8")
            with self.assertRaisesRegex(TaxonomyError, "non-empty string"):
                load_taxonomy(path)


class TestDataset(unittest.TestCase):
    def test_confidence_tiers(self):
        for status in (
            "MANUAL_CORRECTED",
            "PATCHED_EXACT",
            "MATCHED_OFF_VERIFIED",
            "NON_FOOD",
        ):
            self.assertEqual(confidence_for(status), "HIGH")
        for status in ("ENRICHED_MANUAL", "MATCHED_OFF", "SEED_CATALOG"):
            self.assertEqual(confidence_for(status), "MEDIUM")
        for status in ("AI_ESTIMATED", "QUARANTINE_ATWATER_FAIL", "", None):
            self.assertEqual(confidence_for(status), "LOW")

    def test_build_is_deterministic_and_stratified(self):
        first = build_eval_dataset(limit=100, seed=7, taxonomy=LIVE_TAXONOMY)
        second = build_eval_dataset(limit=100, seed=7, taxonomy=LIVE_TAXONOMY)
        self.assertEqual(len(first.products), 100)
        self.assertEqual(first.product_keys, second.product_keys)
        # The FULL payload (keys, names, package weights, ordering) must match.
        self.assertEqual(first.model_payload(), second.model_payload())
        self.assertEqual(
            json.dumps(first.model_payload(), sort_keys=True),
            json.dumps(second.model_payload(), sort_keys=True),
        )
        self.assertEqual(
            sorted(first.category_counts()), sorted(LIVE_TAXONOMY.categories)
        )
        self.assertEqual(len(set(first.product_keys)), 100)
        for tier in ("HIGH", "MEDIUM", "LOW"):
            self.assertGreater(first.confidence_counts()[tier], 0)

    def test_difficult_real_cases_present(self):
        dataset = build_eval_dataset(limit=100, seed=11, taxonomy=LIVE_TAXONOMY)
        names = [product.sample_name for product in dataset.products]
        categories = {product.reference_category for product in dataset.products}
        self.assertTrue(any("%" in name for name in names))
        self.assertTrue(
            any(
                token in name
                for name in names
                for token in ("BUC_", "dez.", "sort.", "3G#")
            )
        )
        self.assertIn(NON_FOOD_CATEGORY, categories)
        self.assertIn("Alcoholic Beverages", categories)
        self.assertTrue(any(product.package_weight_g is None for product in dataset.products))

    def test_model_payload_contains_no_reference_fields(self):
        dataset = build_eval_dataset(limit=30, seed=3, taxonomy=LIVE_TAXONOMY)
        for payload in dataset.model_payload():
            self.assertTrue(
                set(payload).issubset({"product_key", "sample_name", "package_weight_g"})
            )
            for forbidden in (
                "category",
                "macro_category",
                "reference_category",
                "nutrition",
                "match_status",
                "kcal_100g",
                "reference_confidence",
            ):
                self.assertNotIn(forbidden, payload)

    def test_comparable_nutrition_rules(self):
        complete = {
            field: 1.0 for field in ("kcal_100g", "proteins_100g", "carbs_100g", "fat_100g")
        }
        high = EvalProduct(
            product_key="1",
            sample_name="Test",
            package_weight_g=None,
            reference_category="Dairy & Eggs",
            reference_macro="Fresh Food",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition=complete,
        )
        low = EvalProduct(**{**high.__dict__, "reference_confidence": "LOW"})
        non_food = EvalProduct(
            **{
                **high.__dict__,
                "reference_category": NON_FOOD_CATEGORY,
                "reference_macro": "Non-Food",
                "non_food": True,
            }
        )
        incomplete = EvalProduct(
            **{
                **high.__dict__,
                "nutrition": {"kcal_100g": 1.0, "proteins_100g": None,
                              "carbs_100g": 1.0, "fat_100g": 1.0},
            }
        )
        self.assertTrue(is_nutrition_comparable(high))
        self.assertFalse(is_nutrition_comparable(low))
        self.assertFalse(is_nutrition_comparable(non_food))
        self.assertFalse(is_nutrition_comparable(incomplete))

    def test_invalid_limit_and_missing_csv(self):
        with self.assertRaises(DatasetError):
            build_eval_dataset(limit=0, taxonomy=LIVE_TAXONOMY)
        with self.assertRaises(DatasetError):
            build_eval_dataset(limit=1, csv_path=Path("does-not-exist.csv"),
                               taxonomy=LIVE_TAXONOMY)


class TestPrompts(unittest.TestCase):
    def setUp(self):
        self.products = [{"product_key": "abc", "sample_name": "Lapte 1.5%"}]

    def test_prompt_lists_taxonomy_and_schema(self):
        prompt = build_prompt(LIVE_TAXONOMY, self.products, mode="estimate")
        for category in LIVE_TAXONOMY.categories:
            self.assertIn(category, prompt)
        for field_name in NUTRITION_FIELDS:
            self.assertIn(field_name, prompt)
        self.assertIn('"products"', prompt)
        self.assertIn("Lapte 1.5%", prompt)
        self.assertNotIn("match_status", prompt)
        self.assertNotIn("reference_category", prompt)

    def test_none_mode_disables_nutrition(self):
        none_prompt = build_prompt(LIVE_TAXONOMY, self.products, mode="none")
        estimate_prompt = build_prompt(LIVE_TAXONOMY, self.products, mode="estimate")
        self.assertIn('Nutrition mode: "none"', none_prompt)
        self.assertIn("do not estimate", none_prompt.lower())
        self.assertIn('Nutrition mode: "estimate"', estimate_prompt)
        self.assertIn("best-effort estimate", estimate_prompt)
        self.assertNotIn("best-effort estimate", none_prompt)
        self.assertNotEqual(none_prompt, estimate_prompt)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt(LIVE_TAXONOMY, self.products, mode="verbose")

    def test_messages_have_system_and_user_roles(self):
        messages = build_messages(LIVE_TAXONOMY, self.products, mode="none")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])

    def test_forced_choice_prompt_is_unchanged(self):
        """Byte-level lock on the forced-choice baseline prompt.

        The two prompt variants are compared against each other in real runs, so
        the baseline must not drift when abstention support is added.
        """
        payload_block = json.dumps(
            [{"product_key": "abc", "sample_name": "Lapte 1.5%"}],
            ensure_ascii=True,
            indent=2,
        )
        nutrition_schema = ", ".join(
            f'"{field_name}": number|null' for field_name in NUTRITION_FIELDS
        )
        expected = "\n".join(
            [
                "Classify each grocery product into exactly one category from "
                "the canonical taxonomy below.",
                "",
                "Output format: return ONLY a valid JSON object with a single "
                '"products" array, no Markdown fences and no commentary:',
                '{"products": [{"product_key": string, "category": string, '
                + nutrition_schema
                + "}]}",
                "",
                "Rules:",
                "- Return exactly one object for every requested product_key, "
                "in the same order. Never add products that were not requested.",
                "- product_key must be copied exactly from the input.",
                "- category must exactly match one of the taxonomy categories. "
                "Do not invent, translate, abbreviate, or re-case category "
                "names.",
                "- Do not output macro_category: the macro is derived from the "
                "category.",
                '- For non-food products use category "Non-Food" and set every '
                "nutrition field to null.",
                "- Nutrition values are per 100 g: kcal_100g in kcal, all other "
                "fields in grams. Use numbers or null, never strings or ranges.",
                "- No explanations, no confidence scores, no chain of thought.",
                "",
                'Nutrition mode: "estimate". For every food product provide a '
                "best-effort estimate per 100 g in the fields below. Use null "
                "for any value you do not know; never invent a source or "
                "citation. Do not leave fields out: explicitly use null for "
                "unknown values.",
                "",
                "Canonical taxonomy:",
                LIVE_TAXONOMY.prompt_lines(),
                "",
                "Products to classify:",
                payload_block,
            ]
        )
        self.assertEqual(
            build_prompt(LIVE_TAXONOMY, self.products, mode="estimate"), expected
        )
        self.assertEqual(
            build_prompt(
                LIVE_TAXONOMY,
                self.products,
                mode="estimate",
                allow_abstain=False,
            ),
            expected,
        )
        self.assertEqual(
            build_messages(
                LIVE_TAXONOMY, self.products, mode="estimate", allow_abstain=False
            )[1]["content"],
            expected,
        )

    def test_abstain_prompt_allows_null_and_requests_reason(self):
        forced = build_prompt(LIVE_TAXONOMY, self.products, mode="estimate")
        abstain = build_prompt(
            LIVE_TAXONOMY, self.products, mode="estimate", allow_abstain=True
        )
        self.assertIn('"category": string|null', abstain)
        self.assertIn('"abstain_reason": string|null', abstain)
        self.assertIn(
            "set category to null and provide a short abstain_reason", abstain
        )
        self.assertIn(
            "Prefer null over a plausible but unverified category.", abstain
        )
        self.assertIn(
            "Never invent information that is not in the product name.", abstain
        )
        self.assertIn(
            "abstain_reason must be null when category is not null.", abstain
        )
        # Still JSON only, still no chain of thought.
        self.assertIn(
            "- No explanations, no confidence scores, no chain of thought.",
            abstain,
        )
        # The abstain variant only adds rules; it never drops the baseline ones.
        self.assertIn(
            '- For non-food products use category "Non-Food" and set every '
            "nutrition field to null.",
            abstain,
        )
        self.assertNotIn('"category": string|null', forced)
        self.assertNotIn("abstain_reason", forced)
        self.assertNotEqual(forced, abstain)
        none_abstain = build_messages(
            LIVE_TAXONOMY, self.products, mode="none", allow_abstain=True
        )[1]["content"]
        self.assertIn('"abstain_reason": string|null', none_abstain)
        self.assertIn('Nutrition mode: "none"', none_abstain)


class TestParserAndValidation(unittest.TestCase):
    def test_valid_payload(self):
        payload = make_payload(make_item("1", "Dairy & Eggs"))
        result = validate_response(payload, ["1"], LIVE_TAXONOMY, "estimate")
        self.assertTrue(result.valid)
        product = result.products["1"]
        self.assertTrue(product.valid)
        self.assertEqual(product.macro_category, "Fresh Food")
        self.assertEqual(product.nutrition["kcal_100g"], None)

    def test_code_fences_are_stripped(self):
        payload = make_payload(make_item("1", "Dairy & Eggs"))
        text = "```json\n" + json.dumps(payload) + "\n```"
        parsed = parse_model_output(text)
        self.assertEqual(parsed["products"][0]["product_key"], "1")

    def test_json_wrapped_in_prose_is_recovered(self):
        payload = make_payload(make_item("1", "Dairy & Eggs"))
        parsed = parse_model_output("Here you go: " + json.dumps(payload) + " done.")
        self.assertEqual(parsed["products"][0]["category"], "Dairy & Eggs")

    def test_malformed_payloads_raise(self):
        for text in ("", "not json", "[]", '{"items": []}', '{"products": {}}'):
            with self.assertRaises(ParseError):
                parse_model_output(text)

    def test_missing_key_flagged(self):
        payload = make_payload(make_item("2", "Dairy & Eggs"))
        result = validate_response(payload, ["1"], LIVE_TAXONOMY)
        product = result.products["1"]
        self.assertFalse(product.found)
        self.assertIn("missing_product", product.issue_codes)
        self.assertFalse(result.valid)

    def test_extra_key_flagged(self):
        payload = make_payload(
            make_item("1", "Dairy & Eggs"), make_item("2", "Dairy & Eggs")
        )
        result = validate_response(payload, ["1"], LIVE_TAXONOMY)
        self.assertIn(
            "extra_product", [issue.code for issue in result.batch_issues]
        )
        self.assertTrue(result.products["1"].valid)

    def test_duplicate_key_flagged(self):
        payload = make_payload(
            make_item("1", "Dairy & Eggs"),
            make_item("1", "Fresh Meat & Fish"),
        )
        result = validate_response(payload, ["1"], LIVE_TAXONOMY)
        product = result.products["1"]
        self.assertIn("duplicate_product_key", product.issue_codes)
        self.assertFalse(product.valid)
        self.assertIn(
            "duplicate_product_key",
            [issue.code for issue in result.batch_issues],
        )

    def test_invalid_category_flagged(self):
        payload = make_payload(make_item("1", "Dairy"))
        result = validate_response(payload, ["1"], LIVE_TAXONOMY)
        product = result.products["1"]
        self.assertIn("invalid_category", product.issue_codes)
        self.assertIsNone(product.category)
        self.assertIsNone(product.macro_category)
        self.assertFalse(product.valid)

    def test_macro_mismatch_flagged_but_category_kept(self):
        item = make_item("1", "Dairy & Eggs")
        item["macro_category"] = "Pantry & Groceries"
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        product = result.products["1"]
        self.assertIn("macro_mismatch", product.issue_codes)
        self.assertEqual(product.category, "Dairy & Eggs")
        self.assertEqual(product.macro_category, "Fresh Food")

    def test_atwater_consistency_and_mismatch(self):
        consistent = make_item(
            "1", "Dairy & Eggs", kcal_100g=100.0, proteins_100g=10.0,
            carbs_100g=10.0, fat_100g=2.0,
        )
        result = validate_response(make_payload(consistent), ["1"], LIVE_TAXONOMY)
        self.assertNotIn("atwater_mismatch", result.products["1"].issue_codes)

        inconsistent = make_item(
            "1", "Dairy & Eggs", kcal_100g=500.0, proteins_100g=1.0,
            carbs_100g=1.0, fat_100g=1.0,
        )
        result = validate_response(make_payload(inconsistent), ["1"], LIVE_TAXONOMY)
        self.assertIn("atwater_mismatch", result.products["1"].issue_codes)

    def test_alcoholic_beverages_exempt_from_atwater(self):
        item = make_item(
            "1", "Alcoholic Beverages", kcal_100g=200.0, proteins_100g=0.0,
            carbs_100g=0.0, fat_100g=0.0,
        )
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        self.assertNotIn("atwater_mismatch", result.products["1"].issue_codes)

    def test_range_checks(self):
        item = make_item(
            "1", "Dairy & Eggs", kcal_100g=1000.0, proteins_100g=150.0
        )
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        codes = result.products["1"].issue_codes
        self.assertIn("nutrition_out_of_range", codes)
        self.assertFalse(result.products["1"].valid)

    def test_macro_relations(self):
        item = make_item(
            "1", "Dairy & Eggs", kcal_100g=590.0, proteins_100g=40.0,
            carbs_100g=40.0, fat_100g=30.0,
        )
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        self.assertIn("macro_sum_exceeds_105", result.products["1"].issue_codes)

        item = make_item("1", "Dairy & Eggs", carbs_100g=10.0, sugars_100g=20.0)
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        self.assertIn("sugars_exceeds_carbs", result.products["1"].issue_codes)

        item = make_item(
            "1", "Dairy & Eggs", fat_100g=1.0, saturated_fat_100g=5.0
        )
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        self.assertIn(
            "saturated_fat_exceeds_fat", result.products["1"].issue_codes
        )

    def test_non_food_nutrition_flagged(self):
        item = make_item("1", "Non-Food", kcal_100g=100.0)
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        self.assertIn("non_food_nutrition", result.products["1"].issue_codes)
        self.assertFalse(result.products["1"].valid)

    def test_none_mode_rejects_nutrition(self):
        item = make_item("1", "Dairy & Eggs", kcal_100g=100.0)
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY, "none")
        product = result.products["1"]
        self.assertIn("nutrition_provided_in_none_mode", product.issue_codes)
        self.assertTrue(all(value is None for value in product.nutrition.values()))

    def test_non_numeric_nutrition_flagged(self):
        item = make_item("1", "Dairy & Eggs")
        item["kcal_100g"] = "100"
        result = validate_response(make_payload(item), ["1"], LIVE_TAXONOMY)
        self.assertIn(
            "nutrition_not_numeric", result.products["1"].issue_codes
        )


class TestAbstentionParsing(unittest.TestCase):
    def validate(self, item, mode="estimate", *, allow_abstain=True):
        return validate_response(
            make_payload(item),
            [item["product_key"]],
            LIVE_TAXONOMY,
            mode,
            allow_abstain=allow_abstain,
        ).products[item["product_key"]]

    def test_explicit_null_category_is_a_clean_abstention(self):
        reason = "Name only contains a size fragment"
        product = self.validate(
            {"product_key": "1", "category": None, "abstain_reason": reason}
        )
        self.assertTrue(product.found)
        self.assertTrue(product.abstained)
        self.assertTrue(product.valid)
        self.assertEqual(product.issue_codes, [])
        self.assertIsNone(product.category)
        self.assertIsNone(product.macro_category)
        self.assertEqual(product.abstain_reason, reason)
        # An abstention never carries nutrition.
        self.assertTrue(all(value is None for value in product.nutrition.values()))

    def test_explicit_null_is_a_protocol_violation_in_a_forced_run(self):
        product = self.validate(
            {
                "product_key": "1",
                "category": None,
                "abstain_reason": "Name only contains a size fragment",
            },
            allow_abstain=False,
        )
        self.assertIn("abstention_not_allowed", product.issue_codes)
        # Still recorded as an abstention so the metrics stay honest.
        self.assertTrue(product.abstained)
        self.assertIsNone(product.category)
        self.assertIsNone(product.macro_category)
        # ... but never a valid answer: the forced baseline must be flagged.
        self.assertFalse(product.valid)
        self.assertFalse(
            validate_response(
                make_payload(
                    {"product_key": "1", "category": None, "abstain_reason": "x"}
                ),
                ["1"],
                LIVE_TAXONOMY,
            ).valid,
            "validate_response must default to the forced-choice baseline",
        )

    def test_abstention_allowed_in_both_nutrition_modes(self):
        for mode in ("none", "estimate"):
            with self.subTest(mode=mode):
                product = self.validate(
                    {
                        "product_key": "1",
                        "category": None,
                        "abstain_reason": "Ambiguous brand-only name",
                    },
                    mode=mode,
                )
                self.assertTrue(product.abstained)
                self.assertTrue(product.valid)
                self.assertEqual(product.issue_codes, [])
                self.assertTrue(
                    all(value is None for value in product.nutrition.values())
                )

    def test_nutrition_on_abstention_is_flagged(self):
        item = {
            "product_key": "1",
            "category": None,
            "abstain_reason": "Too vague",
            "kcal_100g": 100.0,
        }
        for mode in ("estimate", "none"):
            with self.subTest(mode=mode):
                product = self.validate(item, mode=mode)
                self.assertIn("nutrition_on_abstention", product.issue_codes)
                self.assertTrue(product.abstained)
                self.assertIsNone(product.nutrition["kcal_100g"])
                self.assertFalse(product.valid)

    def test_missing_category_key_is_flagged_abstention(self):
        product = self.validate(
            {"product_key": "1", "abstain_reason": "No usable product name"}
        )
        self.assertIn("missing_category", product.issue_codes)
        self.assertTrue(product.abstained)
        self.assertIsNone(product.category)
        self.assertIsNone(product.macro_category)
        # Never a silent success: the record must not be reported as ok.
        self.assertFalse(product.valid)

    def test_missing_or_long_abstain_reason_is_flagged(self):
        cases = (
            {"product_key": "1", "category": None},
            {"product_key": "1", "category": None, "abstain_reason": ""},
            {"product_key": "1", "category": None, "abstain_reason": "   "},
            {"product_key": "1", "category": None, "abstain_reason": 7},
            {"product_key": "1", "category": None, "abstain_reason": "x" * 201},
        )
        for item in cases:
            with self.subTest(reason=item.get("abstain_reason")):
                product = self.validate(item)
                self.assertIn("missing_abstain_reason", product.issue_codes)
                # Still an abstention, not an answered product.
                self.assertTrue(product.abstained)
                self.assertIsNone(product.category)
                self.assertFalse(product.valid)
        at_limit = self.validate(
            {"product_key": "1", "category": None, "abstain_reason": "x" * 200}
        )
        self.assertEqual(at_limit.issue_codes, [])
        self.assertTrue(at_limit.abstained)

    def test_over_long_abstain_reason_is_stored_truncated(self):
        product = self.validate(
            {"product_key": "1", "category": None, "abstain_reason": "x" * 500}
        )
        # Over-length is still flagged ...
        self.assertIn("missing_abstain_reason", product.issue_codes)
        self.assertTrue(product.abstained)
        # ... but the stored value is bounded, never copied verbatim.
        self.assertIsNotNone(product.abstain_reason)
        self.assertLessEqual(
            len(product.abstain_reason or ""), MAX_ABSTAIN_REASON_LENGTH
        )
        self.assertEqual(len(product.abstain_reason or ""), MAX_ABSTAIN_REASON_LENGTH)

    def test_abstain_reason_on_non_null_category_is_flagged(self):
        item = make_item("1", "Dairy & Eggs")
        item["abstain_reason"] = "not sure"
        product = self.validate(item)
        self.assertIn("abstain_reason_on_category", product.issue_codes)
        self.assertFalse(product.abstained)
        self.assertEqual(product.category, "Dairy & Eggs")
        self.assertIsNone(product.abstain_reason)
        self.assertFalse(product.valid)

    def test_invalid_category_is_not_turned_into_an_abstention(self):
        product = self.validate(make_item("1", "Dairy"))
        self.assertIn("invalid_category", product.issue_codes)
        self.assertFalse(product.abstained)
        self.assertIsNone(product.category)
        self.assertFalse(product.valid)

    def test_abstained_record_keeps_macro_null_when_model_supplies_one(self):
        product = self.validate(
            {
                "product_key": "1",
                "category": None,
                "abstain_reason": "Ambiguous",
                "macro_category": "Fresh Food",
            }
        )
        self.assertTrue(product.abstained)
        self.assertIsNone(product.macro_category)
        # A supplied macro must not be turned into a macro_mismatch on an
        # abstention: there is no derived macro to disagree with.
        self.assertNotIn("macro_mismatch", product.issue_codes)


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        if isinstance(body, Exception):
            self.text = "error"
        else:
            self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def success_response(content: str) -> FakeResponse:
    return FakeResponse(
        200, {"choices": [{"message": {"content": content}}]}
    )


class TestProviderGuards(unittest.TestCase):
    def test_groq_allowlist_and_rejection(self):
        for model in GROQ_FREE_MODELS:
            self.assertEqual(validate_groq_model(model), model)
        with self.assertRaises(FreeModelError):
            validate_groq_model("llama-3.3-70b-versatile")
        with self.assertRaises(FreeModelError):
            GroqClient(api_key="test", model="paid/model")
        client = GroqClient(
            api_key="test", model="paid/model", allow_nonfree_model=True
        )
        self.assertEqual(client.model, "paid/model")

    def test_groq_default_model(self):
        client = GroqClient(api_key="test")
        self.assertEqual(client.model, "openai/gpt-oss-120b")

    def test_openrouter_requires_free_model(self):
        with self.assertRaises(FreeModelError):
            OpenRouterClient(api_key="test", model=None)
        with self.assertRaises(FreeModelError):
            OpenRouterClient(api_key="test", model="openai/gpt-4o")
        for model in ("vendor/model:free", "openrouter/free"):
            client = OpenRouterClient(api_key="test", model=model)
            self.assertEqual(client.model, model)

    def test_openrouter_payload_has_no_fallback_array(self):
        client = OpenRouterClient(api_key="test", model="vendor/model:free")
        payload = client.build_payload([{"role": "user", "content": "x"}])
        self.assertNotIn("models", payload)
        self.assertEqual(payload["model"], "vendor/model:free")


class TestProviderTransport(unittest.TestCase):
    def make_client(self, responses):
        session = FakeSession(responses)
        sleeps = []
        client = GroqClient(
            api_key="test",
            model="openai/gpt-oss-120b",
            session=session,
            sleep_fn=sleeps.append,
            base_delay=0.01,
            rng=random.Random(1),
        )
        return client, session, sleeps

    def test_success_request(self):
        client, session, sleeps = self.make_client(
            [success_response('{"products": []}')]
        )
        response = client.chat([{"role": "user", "content": "hello"}])
        self.assertTrue(response.ok)
        self.assertEqual(response.content, '{"products": []}')
        self.assertEqual(response.attempts, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(session.calls[0]["json"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"], "Bearer test"
        )
        self.assertNotIn("models", session.calls[0]["json"])

    def test_429_retries_and_respects_retry_after(self):
        client, _session, sleeps = self.make_client(
            [
                FakeResponse(429, {"error": {"message": "rate limited"}},
                             {"Retry-After": "2"}),
                success_response('{"products": []}'),
            ]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertTrue(response.ok)
        self.assertEqual(response.attempts, 2)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 2.0)

    def test_401_is_not_retried(self):
        client, _session, sleeps = self.make_client(
            [FakeResponse(401, {"error": {"message": "invalid key"}})]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.attempts, 1)
        self.assertEqual(response.error_kind, "http_error")
        self.assertEqual(sleeps, [])

    def test_5xx_retries_max_three_attempts(self):
        client, session, sleeps = self.make_client(
            [
                FakeResponse(500, {"error": {"message": "boom"}}),
                FakeResponse(503, {"error": {"message": "unavailable"}}),
                FakeResponse(502, {"error": {"message": "bad gateway"}}),
            ]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.attempts, 3)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(sleeps), 2)

    def test_timeouts_are_retried(self):
        client, session, _sleeps = self.make_client(
            [
                requests.Timeout("t1"),
                requests.Timeout("t2"),
                requests.Timeout("t3"),
            ]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.error_kind, "timeout")
        self.assertEqual(response.attempts, 3)
        self.assertEqual(len(session.calls), 3)

    def test_invalid_json_body_not_retried(self):
        client, session, _sleeps = self.make_client(
            [FakeResponse(200, ValueError("bad json"))]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(response.error_kind, "invalid_json_response")
        self.assertEqual(len(session.calls), 1)

    def test_missing_key_fails_without_request(self):
        client = GroqClient(api_key="", model="openai/gpt-oss-120b")
        session = FakeSession([])
        client.session = session
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(response.error_kind, "missing_api_key")
        self.assertEqual(session.calls, [])

    def test_listing_uses_mocked_public_endpoint(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "data": [
                            {"id": "vendor/free:free", "name": "Free",
                             "pricing": {"prompt": "0", "completion": "0"}},
                            {"id": "google/lyria-3-pro-preview",
                             "name": "Zero price, no suffix",
                             "pricing": {"prompt": "0", "completion": "0"}},
                            {"id": "vendor/paid", "name": "Paid",
                             "pricing": {"prompt": "0.000001",
                                         "completion": "0.000002"}},
                        ]
                    },
                )
            ]
        )
        models = list_openrouter_free_models(session=session)
        self.assertEqual(
            [model["id"] for model in models],
            ["google/lyria-3-pro-preview", "vendor/free:free"],
        )
        usable = {model["id"]: model["usable"] for model in models}
        # Zero-priced models are listed, but the :free guard still decides.
        self.assertTrue(usable["vendor/free:free"])
        self.assertFalse(usable["google/lyria-3-pro-preview"])


class TestEnvAndSecrets(unittest.TestCase):
    def test_env_parser_and_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\n"
                "GROQ_API_KEY='from-file'\n"
                "export OPENROUTER_MODEL=\"vendor/model:free\"\n"
                "MALFORMED_LINE\n",
                encoding="utf-8",
            )
            parsed = parse_env_file(path)
            self.assertEqual(parsed["GROQ_API_KEY"], "from-file")
            self.assertEqual(parsed["OPENROUTER_MODEL"], "vendor/model:free")
            self.assertEqual(
                get_api_key("GROQ_API_KEY", env_path=path, environ={}), "from-file"
            )
            self.assertEqual(
                get_api_key(
                    "GROQ_API_KEY",
                    env_path=path,
                    environ={"GROQ_API_KEY": "from-env"},
                ),
                "from-env",
            )

    def test_redaction(self):
        text = (
            "Authorization: Bearer sk-or-v1-SECRETVALUE "
            "key=gsk_SUPERSECRET123 and sk-abc123456789"
        )
        redacted = redact_text(text, secrets=("sk-or-v1-SECRETVALUE",))
        self.assertNotIn("SECRETVALUE", redacted)
        self.assertNotIn("SUPERSECRET", redacted)
        self.assertNotIn("sk-abc123456789", redacted)
        self.assertIn("***", redacted)

    def test_results_file_never_contains_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            secret = "gsk_TOPSECRETVALUE123456"
            record = {
                "provider": "groq",
                "product_key": "1",
                "errors": [f"Authorization: Bearer {secret}", f"key {secret}"],
            }
            append_results(path, [record], secrets=[secret])
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("TOPSECRETVALUE", text)
            self.assertNotIn(secret, text)
            self.assertEqual(json.loads(text.splitlines()[0])["provider"], "groq")


class TestRunnerDryRun(unittest.TestCase):
    def test_dry_run_needs_no_keys_and_no_network(self):
        output_dir = (
            PROJECT_ROOT / "scratch" / "ai_eval" / f"_dry_run_{uuid.uuid4().hex}"
        )
        stdout = io.StringIO()

        def guarded_get_api_key(name, *args, **kwargs):
            if str(name).endswith("_API_KEY"):
                raise AssertionError("dry run must not read API keys")
            return None

        network_guard = AssertionError("dry run must not touch the network")
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "", "OPENROUTER_API_KEY": ""},
            clear=False,
        ), patch(
            "ai_eval.providers.get_api_key", side_effect=guarded_get_api_key
        ), patch(
            "requests.post", side_effect=network_guard
        ), patch(
            "requests.get", side_effect=network_guard
        ), patch.object(
            requests.Session, "post", side_effect=network_guard
        ), patch.object(
            requests.Session, "get", side_effect=network_guard
        ), patch.object(
            socket, "socket", side_effect=network_guard
        ), redirect_stdout(stdout):
            code = runner_mod.main(
                [
                    "--provider",
                    "both",
                    "--mode",
                    "estimate",
                    "--limit",
                    "2",
                    "--batch-size",
                    "2",
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ]
            )
        try:
            text = stdout.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("DRY RUN", text)
            self.assertIn("No network calls were made", text)
            self.assertIn("Dairy & Eggs", text)
            self.assertIn("Nutrition mode", text)
            self.assertIn("Prompt variant: forced-choice", text)
            self.assertNotIn("abstain_reason", text)
            self.assertFalse(output_dir.exists())
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_dry_run_abstain_shows_variant_and_needs_no_keys(self):
        output_dir = (
            PROJECT_ROOT
            / "scratch"
            / "ai_eval"
            / f"_dry_run_abstain_{uuid.uuid4().hex}"
        )
        stdout = io.StringIO()

        def guarded_get_api_key(name, *args, **kwargs):
            if str(name).endswith("_API_KEY"):
                raise AssertionError("dry run must not read API keys")
            return None

        network_guard = AssertionError("dry run must not touch the network")
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "", "OPENROUTER_API_KEY": ""},
            clear=False,
        ), patch(
            "ai_eval.providers.get_api_key", side_effect=guarded_get_api_key
        ), patch(
            "requests.post", side_effect=network_guard
        ), patch(
            "requests.get", side_effect=network_guard
        ), patch.object(
            requests.Session, "post", side_effect=network_guard
        ), patch.object(
            requests.Session, "get", side_effect=network_guard
        ), patch.object(
            socket, "socket", side_effect=network_guard
        ), redirect_stdout(stdout):
            code = runner_mod.main(
                [
                    "--provider",
                    "both",
                    "--mode",
                    "estimate",
                    "--limit",
                    "2",
                    "--batch-size",
                    "2",
                    "--dry-run",
                    "--abstain",
                    "--output-dir",
                    str(output_dir),
                ]
            )
        try:
            text = stdout.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Prompt variant: abstain-allowed", text)
            self.assertIn("abstain=True", text)
            self.assertIn('"abstain_reason": string|null', text)
            self.assertIn("set category to null", text)
            self.assertIn("No network calls were made", text)
            self.assertFalse(output_dir.exists())
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_dry_run_rejects_listing_flags_without_network(self):
        network_guard = AssertionError("listing flags must not run here")

        def guarded_get_api_key(name, *args, **kwargs):
            if str(name).endswith("_API_KEY"):
                raise AssertionError("rejected flags must not read API keys")
            return None

        for flag in ("--list-free-models", "--list-groq-models"):
            stdout = io.StringIO()
            with patch(
                "ai_eval.providers.get_api_key", side_effect=guarded_get_api_key
            ), patch(
                "requests.get", side_effect=network_guard
            ), patch.object(
                requests.Session, "get", side_effect=network_guard
            ), patch.object(
                requests.Session, "post", side_effect=network_guard
            ), redirect_stdout(stdout):
                code = runner_mod.main(["--dry-run", flag])
            self.assertEqual(code, 2)
            self.assertIn("--dry-run", stdout.getvalue())
            self.assertIn("network", stdout.getvalue())

    def test_listing_annotates_usable_and_rejected_models(self):
        listing = [
            {"id": "vendor/model:free", "name": "Usable", "usable": True},
            {
                "id": "google/lyria-3-pro-preview",
                "name": "Zero-priced but rejected",
                "usable": False,
            },
        ]
        stdout = io.StringIO()
        with patch(
            "ai_eval.runner.list_openrouter_free_models", return_value=listing
        ), redirect_stdout(stdout):
            code = runner_mod.main(["--list-free-models"])
        text = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("[usable]", text)
        self.assertIn("[not :free - rejected by guard]", text)
        self.assertIn("lyria-3-pro-preview", text)
        self.assertIn("vendor/model:free", text)


class TestOutputDirConfinement(unittest.TestCase):
    def test_absolute_path_outside_repo_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "leaked.jsonl"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = runner_mod.main(
                    ["--dry-run", "--output-dir", str(target.parent)]
                )
            self.assertEqual(code, 2)
            self.assertIn("outside the allowed output root", stdout.getvalue())
            self.assertFalse(target.exists())

    def test_parent_traversal_escape_is_rejected(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = runner_mod.main(
                ["--dry-run", "--output-dir", "scratch/../data"]
            )
        self.assertEqual(code, 2)
        self.assertIn("outside the allowed output root", stdout.getvalue())

    def test_resolve_output_dir_accepts_nested_scratch_path(self):
        nested = Path("scratch") / "ai_eval" / "_confined" / uuid.uuid4().hex
        resolved = runner_mod.resolve_output_dir(nested)
        try:
            resolved.relative_to(runner_mod.SCRATCH_ROOT)
        except ValueError:
            self.fail(f"{resolved} is not confined to {runner_mod.SCRATCH_ROOT}")
        self.assertEqual(resolved, (PROJECT_ROOT / nested).resolve())

    def test_dry_run_accepts_nested_scratch_path(self):
        nested = Path("scratch") / "ai_eval" / f"_ok_{uuid.uuid4().hex}"
        stdout = io.StringIO()
        with patch(
            "ai_eval.providers.get_api_key", return_value=None
        ), redirect_stdout(stdout):
            code = runner_mod.main(["--dry-run", "--output-dir", str(nested)])
        self.assertEqual(code, 0)
        self.assertFalse((PROJECT_ROOT / nested).exists())

    def test_resolve_output_dir_raises_config_error_for_escape(self):
        with self.assertRaises(runner_mod.ConfigError):
            runner_mod.resolve_output_dir(PROJECT_ROOT / "data")


class TestReporting(unittest.TestCase):
    def make_dataset(self):
        complete = {
            "kcal_100g": 100.0,
            "proteins_100g": 5.0,
            "carbs_100g": 10.0,
            "sugars_100g": 3.0,
            "fat_100g": 2.0,
            "saturated_fat_100g": 1.0,
            "fiber_100g": 1.0,
            "salt_100g": 0.1,
        }
        high = EvalProduct(
            product_key="1",
            sample_name="High product",
            package_weight_g=100.0,
            reference_category="Dairy & Eggs",
            reference_macro="Fresh Food",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition=complete,
        )
        low = EvalProduct(
            product_key="2",
            sample_name="Low product",
            package_weight_g=100.0,
            reference_category="Non-Food",
            reference_macro="Non-Food",
            reference_confidence="LOW",
            match_status="AI_ESTIMATED",
            nutrition=complete,
            non_food=True,
        )
        return EvalDataset(
            products=(high, low),
            seed=1,
            source_path="fixture.csv",
            requested_limit=2,
        )

    def test_summary_accuracy_by_tier_and_nutrition_mae(self):
        dataset = self.make_dataset()
        timestamp = "2024-01-01T00:00:00+00:00"
        base = {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "mode": "estimate",
            "latency_ms": 100.0,
            "attempts": 1,
            "errors": [],
            "error_kind": None,
            "batch_id": "batch-1",
            "batch_index": 1,
            "batch_size": 2,
            "json_valid": True,
            "timestamp": timestamp,
        }
        records = [
            {
                **base,
                "product_key": "1",
                "reference": dataset.products[0].reference_snapshot(),
                "prediction": {
                    "category": "Dairy & Eggs",
                    "macro_category": "Fresh Food",
                    "nutrition": {"kcal_100g": 110.0},
                },
                "validation": {"valid": True, "issues": [], "batch_issues": []},
                "status": "ok",
            },
            {
                **base,
                "product_key": "2",
                "reference": dataset.products[1].reference_snapshot(),
                "prediction": {
                    "category": "Dairy & Eggs",
                    "macro_category": "Fresh Food",
                    "nutrition": {"kcal_100g": 0.0},
                },
                "validation": {"valid": False, "issues": [], "batch_issues": []},
                "status": "flagged",
            },
        ]
        summary = build_summary(
            records, dataset, {"groq": "openai/gpt-oss-120b"}, "estimate"
        )
        provider = summary["providers"]["groq"]
        self.assertEqual(provider["batches"]["json_valid"], 1)
        self.assertEqual(provider["products"]["valid"], 1)
        self.assertEqual(provider["products"]["flagged"], 1)
        overall = provider["category_accuracy"]["overall"]
        # Headline accuracy is measured only against HIGH/MEDIUM references.
        self.assertEqual((overall["correct"], overall["total"]), (1, 1))
        self.assertTrue(overall["ground_truth"])
        high = provider["category_accuracy"]["by_confidence"]["HIGH"]
        self.assertEqual((high["correct"], high["total"]), (1, 1))
        low = provider["category_accuracy"]["by_confidence"]["LOW"]
        self.assertFalse(low["ground_truth"])
        self.assertEqual((low["correct"], low["total"]), (0, 1))
        low_agreement = provider["category_accuracy"]["reference_agreement_low"]
        self.assertFalse(low_agreement["ground_truth"])
        self.assertEqual(
            (low_agreement["correct"], low_agreement["total"]), (0, 1)
        )
        self.assertIn("unverified", low_agreement["label"])
        rendered = render_report(summary)
        self.assertIn("Category accuracy (HIGH+MEDIUM references)", rendered)
        self.assertIn("not accuracy", rendered)
        self.assertNotIn("Category accuracy (all)", rendered)
        # Only the HIGH food row is nutrition-comparable.
        self.assertEqual(provider["nutrition"]["eligible"], 1)
        kcal = provider["nutrition"]["fields"]["kcal_100g"]
        self.assertEqual(kcal["compared"], 1)
        self.assertAlmostEqual(kcal["mae"], 10.0)

    def test_completed_keys_only_valid_successes(self):
        records = [
            {
                "provider": "groq",
                "model": "m",
                "product_key": "a",
                "status": "ok",
                "validation": {"valid": True},
            },
            {
                "provider": "groq",
                "model": "m",
                "product_key": "b",
                "status": "flagged",
                "validation": {"valid": False},
            },
            {
                "provider": "groq",
                "model": "m",
                "product_key": "c",
                "status": "error",
                "validation": {"valid": False},
            },
            {
                "provider": "groq",
                "model": "other",
                "product_key": "d",
                "status": "ok",
                "validation": {"valid": True},
            },
        ]
        self.assertEqual(completed_keys(records, "groq", "m"), {"a"})

    def test_completed_keys_filters_by_mode(self):
        records = [
            {
                "provider": "groq",
                "model": "m",
                "product_key": "a",
                "mode": "none",
                "status": "ok",
                "validation": {"valid": True},
            },
            {
                "provider": "groq",
                "model": "m",
                "product_key": "b",
                "mode": "estimate",
                "status": "ok",
                "validation": {"valid": True},
            },
            {
                "provider": "groq",
                "model": "m",
                "product_key": "c",
                "mode": None,
                "status": "ok",
                "validation": {"valid": True},
            },
        ]
        self.assertEqual(completed_keys(records, "groq", "m"), {"a", "b", "c"})
        # Strict per-mode resume: a stored mode is required to match.
        self.assertEqual(completed_keys(records, "groq", "m", mode="none"), {"a"})
        self.assertEqual(
            completed_keys(records, "groq", "m", mode="estimate"), {"b"}
        )

    def test_legacy_records_without_mode_never_match_a_mode(self):
        legacy = {
            "provider": "groq",
            "model": "m",
            "product_key": "legacy",
            "status": "ok",
            "validation": {"valid": True},
        }
        for mode in ("none", "estimate"):
            self.assertEqual(
                completed_keys([legacy], "groq", "m", mode=mode), set()
            )
            self.assertEqual(latest_results([legacy], mode=mode), [])
        # Without an explicit mode the record is still readable.
        self.assertEqual(completed_keys([legacy], "groq", "m"), {"legacy"})

    def test_latest_results_are_mode_aware(self):
        base = {"provider": "groq", "model": "m", "product_key": "1"}
        records = [
            {**base, "mode": "estimate", "status": "ok"},
            {**base, "mode": "none", "status": "error"},
        ]
        # Default last-wins behaviour still reports the most recent record.
        self.assertEqual(latest_results(records)[0]["status"], "error")
        estimate = latest_results(records, mode="estimate")
        self.assertEqual(len(estimate), 1)
        self.assertEqual(estimate[0]["status"], "ok")


class TestModelResolution(unittest.TestCase):
    def make_args(self, provider: str = "both", model: str | None = None):
        return argparse.Namespace(provider=provider, model=model)

    def write_env(self, tmp: str, text: str) -> Path:
        path = Path(tmp) / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_models_resolve_from_env_file_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_env(
                tmp,
                "GROQ_MODEL=groq-model-from-file\n"
                "OPENROUTER_MODEL=vendor/file-model:free\n",
            )
            models = runner_mod.resolve_models(
                self.make_args(), env_path=path, environ={}
            )
        self.assertEqual(models["groq"], "groq-model-from-file")
        self.assertEqual(models["openrouter"], "vendor/file-model:free")

    def test_environment_wins_over_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_env(
                tmp,
                "GROQ_MODEL=groq-model-from-file\n"
                "OPENROUTER_MODEL=vendor/file-model:free\n",
            )
            models = runner_mod.resolve_models(
                self.make_args(),
                env_path=path,
                environ={
                    "GROQ_MODEL": "groq-model-from-env",
                    "OPENROUTER_MODEL": "vendor/env-model:free",
                },
            )
        self.assertEqual(models["groq"], "groq-model-from-env")
        self.assertEqual(models["openrouter"], "vendor/env-model:free")

    def test_cli_model_wins_over_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_env(tmp, "OPENROUTER_MODEL=vendor/file-model:free\n")
            models = runner_mod.resolve_models(
                self.make_args(provider="openrouter", model="vendor/cli-model:free"),
                env_path=path,
                environ={"OPENROUTER_MODEL": "vendor/env-model:free"},
            )
        self.assertEqual(models, {"openrouter": "vendor/cli-model:free"})

    def test_defaults_without_settings(self):
        missing = (
            Path(tempfile.gettempdir())
            / f"ai_eval_missing_env_{uuid.uuid4().hex}"
        )
        models = runner_mod.resolve_models(
            self.make_args(), env_path=missing, environ={}
        )
        self.assertEqual(models["groq"], "openai/gpt-oss-120b")
        self.assertIsNone(models["openrouter"])

    def test_model_with_both_providers_still_rejected(self):
        with self.assertRaises(runner_mod.ConfigError):
            runner_mod.resolve_models(
                self.make_args(provider="both", model="vendor/model:free"),
                environ={},
            )

    def test_get_setting_precedence_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_env(tmp, "SAMPLE_SETTING=from-file\n")
            self.assertEqual(
                get_setting("SAMPLE_SETTING", env_path=path, environ={}),
                "from-file",
            )
            self.assertEqual(
                get_setting(
                    "SAMPLE_SETTING",
                    env_path=path,
                    environ={"SAMPLE_SETTING": "from-env"},
                ),
                "from-env",
            )
            self.assertEqual(
                get_setting("MISSING_SETTING", "fallback", env_path=path, environ={}),
                "fallback",
            )
            self.assertIsNone(
                get_setting("MISSING_SETTING", env_path=path, environ={})
            )


class TestDuplicateJsonMembers(unittest.TestCase):
    def test_taxonomy_duplicate_macro_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taxonomy.json"
            path.write_text('{"A": ["X"], "A": ["Y"]}', encoding="utf-8")
            with self.assertRaisesRegex(TaxonomyError, r"Duplicate key 'A'"):
                load_taxonomy(path)

    def test_duplicate_products_member_rejected(self):
        raw = '{"products": [], "products": [{"product_key": "1"}]}'
        with self.assertRaisesRegex(ParseError, "products"):
            parse_model_output(raw)

    def test_duplicate_product_key_member_rejected(self):
        raw = (
            '{"products": [{"product_key": "1", "product_key": "2", '
            '"category": "Dairy & Eggs"}]}'
        )
        with self.assertRaisesRegex(ParseError, "product_key"):
            parse_model_output(raw)

    def test_duplicate_member_detected_inside_prose_wrapper(self):
        raw = (
            'Here you go: {"products": [{"product_key": "1", '
            '"category": "Dairy & Eggs", "category": "Non-Food"}]} done.'
        )
        with self.assertRaisesRegex(ParseError, "category"):
            parse_model_output(raw)


class TestRetryAfterBudget(unittest.TestCase):
    def make_client(self, responses):
        session = FakeSession(responses)
        sleeps: list[float] = []
        client = GroqClient(
            api_key="test",
            model="openai/gpt-oss-120b",
            session=session,
            sleep_fn=sleeps.append,
            base_delay=0.01,
            rng=random.Random(1),
        )
        return client, session, sleeps

    def test_retry_after_over_budget_is_not_retried_early(self):
        client, session, sleeps = self.make_client(
            [
                FakeResponse(
                    429,
                    {"error": {"message": "quota exhausted"}},
                    {"Retry-After": "300"},
                )
            ]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.error_kind, "rate_limited")
        self.assertEqual(response.attempts, 1)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(sleeps, [])
        self.assertIn("300", response.error or "")

    def test_retry_after_within_budget_still_retries(self):
        client, _session, sleeps = self.make_client(
            [
                FakeResponse(
                    429,
                    {"error": {"message": "slow down"}},
                    {"Retry-After": "2"},
                ),
                success_response('{"products": []}'),
            ]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertTrue(response.ok)
        self.assertEqual(response.attempts, 2)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 2.0)
        self.assertLessEqual(sleeps[0], 120.0)

    def test_far_future_http_date_is_not_retried_early(self):
        future = format_datetime(
            datetime.now(timezone.utc) + timedelta(hours=5), usegmt=True
        )
        client, session, sleeps = self.make_client(
            [
                FakeResponse(
                    429,
                    {"error": {"message": "quota exhausted"}},
                    {"Retry-After": future},
                )
            ]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.error_kind, "rate_limited")
        self.assertEqual(response.attempts, 1)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(sleeps, [])


class TestTornJsonlTail(unittest.TestCase):
    def test_partial_tail_is_discarded_before_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                '{"provider": "groq", "product_key": "old"}\n'
                '{"provider": "groq", "product_key": "tor',
                encoding="utf-8",
            )
            append_results(
                path,
                [{"provider": "groq", "product_key": "new", "status": "ok"}],
            )
            records = load_results(path)
            self.assertEqual(
                [record["product_key"] for record in records], ["old", "new"]
            )
            text = path.read_text(encoding="utf-8")
            self.assertEqual(len(text.splitlines()), 2)
            self.assertTrue(text.endswith("\n"))
            self.assertNotIn('"tor', text)

    def test_file_without_trailing_newline_is_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text('{"provider": "groq", "product_key": "torn"', encoding="utf-8")
            append_results(
                path, [{"provider": "groq", "product_key": "good"}]
            )
            records = load_results(path)
            self.assertEqual([record["product_key"] for record in records], ["good"])
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_non_finite_floats_are_written_as_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            record = {
                "provider": "groq",
                "latency_ms": float("nan"),
                "nested": {"mean": float("inf"), "items": [float("-inf")]},
            }
            append_results(path, [record])
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            loaded = load_results(path)[0]
            self.assertIsNone(loaded["latency_ms"])
            self.assertIsNone(loaded["nested"]["mean"])
            self.assertEqual(loaded["nested"]["items"], [None])
            json.loads(text)

    def test_write_json_forbids_non_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            write_json(path, sanitize_value({"value": float("nan")}))
            self.assertIsNone(json.loads(path.read_text(encoding="utf-8"))["value"])


class TestNonFiniteAndOverflowNumbers(unittest.TestCase):
    SEED_HEADER = (
        "product_key,sample_name,macro_category,category,package_weight_g,"
        "match_status,kcal_100g,proteins_100g,carbs_100g,fat_100g,times_bought\n"
    )

    def build_from_rows(self, tmp: str, rows: str):
        csv_path = Path(tmp) / "seed.csv"
        csv_path.write_text(self.SEED_HEADER + rows, encoding="utf-8")
        return build_eval_dataset(
            limit=10, seed=1, csv_path=csv_path, taxonomy=LIVE_TAXONOMY
        )

    def test_nan_and_infinity_references_are_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.build_from_rows(
                tmp,
                "q1,Nan food,Fresh Food,Dairy & Eggs,100,MANUAL_CORRECTED,"
                "NaN,1,1,1,NaN\n"
                "q2,Inf food,Fresh Food,Dairy & Eggs,Infinity,MANUAL_CORRECTED,"
                "Infinity,1,1,1,NaN\n"
                "q3,Good food,Fresh Food,Dairy & Eggs,100,MANUAL_CORRECTED,"
                "100,1,1,1,3\n",
            )
        by_key = dataset.by_key()
        self.assertIsNone(by_key["q1"].nutrition["kcal_100g"])
        self.assertIsNone(by_key["q1"].times_bought)
        self.assertIsNone(by_key["q2"].nutrition["kcal_100g"])
        self.assertIsNone(by_key["q2"].package_weight_g)
        self.assertFalse(is_nutrition_comparable(by_key["q1"]))
        self.assertFalse(is_nutrition_comparable(by_key["q2"]))
        self.assertTrue(is_nutrition_comparable(by_key["q3"]))
        self.assertEqual(dataset.comparable_nutrition_count(), 1)

    def test_nan_and_infinity_json_literals_are_flagged(self):
        raw = (
            '{"products": [{"product_key": "1", "category": "Dairy & Eggs", '
            '"kcal_100g": NaN, "proteins_100g": Infinity}]}'
        )
        payload = parse_model_output(raw)
        result = validate_response(payload, ["1"], LIVE_TAXONOMY, "estimate")
        product = result.products["1"]
        self.assertIn("nutrition_not_numeric", product.issue_codes)
        self.assertIsNone(product.nutrition["kcal_100g"])
        self.assertIsNone(product.nutrition["proteins_100g"])

    def test_to_float_and_to_int_reject_non_finite_and_overflow(self):
        for value in ("NaN", "Infinity", "-Infinity", 10 ** 400, "1e999"):
            self.assertIsNone(_to_float(value), value)
        for value in ("NaN", "Infinity", 10 ** 400):
            self.assertIsNone(_to_int(value), value)
        self.assertEqual(_to_int("3"), 3)
        self.assertEqual(_to_float("1.5"), 1.5)

    def test_overflowing_json_integer_becomes_validation_issue(self):
        huge = "9" * 400
        raw = (
            '{"products": [{"product_key": "1", "category": "Dairy & Eggs", '
            '"kcal_100g": ' + huge + "}]}"
        )
        payload = parse_model_output(raw)
        result = validate_response(payload, ["1"], LIVE_TAXONOMY, "estimate")
        product = result.products["1"]
        self.assertIn("nutrition_not_numeric", product.issue_codes)
        self.assertIsNone(product.nutrition["kcal_100g"])

    def test_one_overflowing_product_does_not_abort_the_batch(self):
        huge = "9" * 400
        raw = (
            '{"products": [{"product_key": "1", "category": "Dairy & Eggs", '
            '"kcal_100g": ' + huge + ", \"proteins_100g\": 1, \"carbs_100g\": 1, "
            '"fat_100g": 1}]}'
        )
        client = GroqClient(api_key="test", model="openai/gpt-oss-120b")
        product = EvalProduct(
            product_key="1",
            sample_name="Test",
            package_weight_g=100.0,
            reference_category="Dairy & Eggs",
            reference_macro="Fresh Food",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition={
                "kcal_100g": 100.0,
                "proteins_100g": 1.0,
                "carbs_100g": 1.0,
                "fat_100g": 1.0,
            },
        )
        response = ProviderResponse(
            content=raw, status_code=200, attempts=1, latency_ms=1.0
        )
        records = runner_mod._batch_records(
            client,
            [product],
            response,
            batch_id="batch-1",
            batch_index=1,
            timestamp="2024-01-01T00:00:00+00:00",
            taxonomy=LIVE_TAXONOMY,
            mode="estimate",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "flagged")
        self.assertIn(
            "nutrition_not_numeric",
            [issue["code"] for issue in records[0]["validation"]["issues"]],
        )


class TestRetryClassification(unittest.TestCase):
    def test_non_retryable_request_exception_is_single_attempt(self):
        session = FakeSession([requests.exceptions.InvalidHeader("bad header")])
        sleeps: list[float] = []
        client = GroqClient(
            api_key="test",
            model="openai/gpt-oss-120b",
            session=session,
            sleep_fn=sleeps.append,
            base_delay=0.01,
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.error_kind, "request_error")
        self.assertEqual(response.attempts, 1)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(sleeps, [])

    def test_connection_errors_are_still_retried(self):
        session = FakeSession(
            [requests.ConnectionError("down"), success_response('{"products": []}')]
        )
        client = GroqClient(
            api_key="test",
            model="openai/gpt-oss-120b",
            session=session,
            sleep_fn=lambda _delay: None,
            base_delay=0.01,
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertTrue(response.ok)
        self.assertEqual(response.attempts, 2)
        self.assertEqual(len(session.calls), 2)

    def test_max_attempts_is_clamped(self):
        client = GroqClient(
            api_key="test", model="openai/gpt-oss-120b", max_attempts=99
        )
        self.assertEqual(client.max_attempts, MAX_ATTEMPTS_LIMIT)
        self.assertLessEqual(client.max_attempts, 3)
        client = GroqClient(
            api_key="test", model="openai/gpt-oss-120b", max_attempts=0
        )
        self.assertEqual(client.max_attempts, 1)


class TestSummarySecretHygiene(unittest.TestCase):
    SECRET = "gsk_SUMMARYSECRET987654"

    def make_fixture(self):
        product = EvalProduct(
            product_key="1",
            sample_name="Secret product",
            package_weight_g=100.0,
            reference_category="Dairy & Eggs",
            reference_macro="Fresh Food",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition={
                "kcal_100g": 100.0,
                "proteins_100g": 5.0,
                "carbs_100g": 10.0,
                "fat_100g": 2.0,
            },
        )
        dataset = EvalDataset(
            products=(product,),
            seed=1,
            source_path="fixture.csv",
            requested_limit=1,
        )
        record = {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "mode": "estimate",
            "product_key": "1",
            "reference": product.reference_snapshot(),
            "prediction": {"category": "Dairy & Eggs", "nutrition": {}},
            "validation": {
                "valid": False,
                "issues": [
                    {
                        "code": "provider_leak",
                        "message": f"Authorization: Bearer {self.SECRET}",
                    }
                ],
                "batch_issues": [],
            },
            "status": "flagged",
            "errors": [f"key {self.SECRET} rejected"],
            "error_kind": "http_error",
            "latency_ms": 1.0,
            "attempts": 1,
            "batch_id": "batch-1",
            "batch_index": 1,
            "batch_size": 1,
            "json_valid": True,
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        return dataset, record

    def test_summary_and_report_redact_key_shaped_strings(self):
        dataset, record = self.make_fixture()
        summary = build_summary(
            [record], dataset, {"groq": "openai/gpt-oss-120b"}, "estimate"
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            report_path = Path(tmp) / "report.md"
            write_json(summary_path, sanitize_value(summary, (self.SECRET,)))
            write_text(
                report_path,
                sanitize_value(render_report(summary), (self.SECRET,)),
            )
            for path in (summary_path, report_path):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("SUMMARYSECRET", text)
                self.assertNotIn(self.SECRET, text)
            json.loads(summary_path.read_text(encoding="utf-8"))


class TestTornUtf8Lines(unittest.TestCase):
    def test_truncated_multibyte_tail_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            first = json.dumps(
                {"product_key": "a", "input_name": "Lapte"}, ensure_ascii=False
            ).encode("utf-8")
            second = json.dumps(
                {"product_key": "b"}, ensure_ascii=False
            ).encode("utf-8")
            # A multibyte character (e.g. Romanian "ș") cut in half by a kill.
            torn = (
                '{"product_key": "c", "input_name": "Legume '.encode("utf-8")
                + b"\xc8"
            )
            path.write_bytes(first + b"\n" + second + b"\n" + torn)
            records = load_results(path)
            self.assertEqual(
                [record["product_key"] for record in records], ["a", "b"]
            )

    def test_undecodable_middle_line_does_not_lose_following_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_bytes(
                b'{"product_key": "a"}\n'
                b'{"product_key": "\xc8"}\n'
                b'{"product_key": "b"}\n'
            )
            records = load_results(path)
            self.assertEqual(
                [record["product_key"] for record in records], ["a", "b"]
            )


class TestCompleteTailWithoutNewline(unittest.TestCase):
    def test_complete_record_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text('{"product_key": "old"}', encoding="utf-8")
            append_results(path, [{"product_key": "new"}])
            records = load_results(path)
            self.assertEqual(
                [record["product_key"] for record in records], ["old", "new"]
            )
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(len(text.splitlines()), 2)

    def test_incomplete_tail_is_still_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text('{"product_key": "old"', encoding="utf-8")
            append_results(path, [{"product_key": "new"}])
            records = load_results(path)
            self.assertEqual(
                [record["product_key"] for record in records], ["new"]
            )


class TestArtifactConfinement(unittest.TestCase):
    class _StubClient:
        provider = "groq"
        model = "openai/gpt-oss-120b"
        api_key = "test-key"

        def chat(self, messages):
            raise AssertionError(
                "artifact confinement must be checked before any request"
            )

    def test_symlinked_artifact_escaping_scratch_is_rejected(self):
        with tempfile.TemporaryDirectory() as probe_tmp:
            probe_target = Path(probe_tmp) / "target"
            probe_target.write_text("x", encoding="utf-8")
            probe_link = Path(probe_tmp) / "link"
            try:
                os.symlink(probe_target, probe_link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(
                    f"cannot create a symlink on this platform: {exc}"
                )
            try:
                probe_link.unlink()
            except OSError:
                pass

        for artifact_name in ("results.jsonl", "summary.json", "report.md"):
            with self.subTest(artifact=artifact_name):
                output_dir = (
                    runner_mod.SCRATCH_ROOT
                    / "ai_eval"
                    / f"_confinement_{uuid.uuid4().hex}"
                )
                with tempfile.TemporaryDirectory() as tmp:
                    external = Path(tmp) / artifact_name
                    external.write_text("external-content\n", encoding="utf-8")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    os.symlink(external, output_dir / artifact_name)
                    stdout = io.StringIO()
                    try:
                        with patch.object(
                            runner_mod,
                            "resolve_models",
                            return_value={"groq": "openai/gpt-oss-120b"},
                        ), patch.object(
                            runner_mod,
                            "_make_clients",
                            return_value=[self._StubClient()],
                        ), redirect_stdout(stdout):
                            code = runner_mod.main(
                                [
                                    "--provider",
                                    "groq",
                                    "--mode",
                                    "none",
                                    "--limit",
                                    "1",
                                    "--batch-size",
                                    "1",
                                    "--delay",
                                    "0",
                                    "--output-dir",
                                    str(output_dir),
                                ]
                            )
                    finally:
                        shutil.rmtree(output_dir, ignore_errors=True)
                    self.assertEqual(code, 2)
                    self.assertIn(
                        "outside the allowed output root", stdout.getvalue()
                    )
                    self.assertEqual(
                        external.read_text(encoding="utf-8"),
                        "external-content\n",
                    )

    def test_plain_artifacts_inside_scratch_are_allowed(self):
        output_dir = (
            runner_mod.SCRATCH_ROOT / "ai_eval" / f"_allowed_{uuid.uuid4().hex}"
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ("results.jsonl", "summary.json", "report.md"):
                artifact = output_dir / name
                self.assertEqual(
                    runner_mod._ensure_confined_artifact(artifact), artifact
                )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_config_error_for_escaping_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = (
                runner_mod.SCRATCH_ROOT
                / "ai_eval"
                / f"_escape_{uuid.uuid4().hex}"
            )
            external = Path(tmp) / "target.jsonl"
            external.write_text("x", encoding="utf-8")
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                try:
                    os.symlink(external, output_dir / "results.jsonl")
                except (OSError, NotImplementedError) as exc:
                    self.skipTest(
                        f"cannot create a symlink on this platform: {exc}"
                    )
                with self.assertRaises(runner_mod.ConfigError):
                    runner_mod._ensure_confined_artifact(
                        output_dir / "results.jsonl"
                    )
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)


class TestProviderDiagnostics(unittest.TestCase):
    def test_error_metadata_fields_are_surfaced_and_redacted(self):
        secret = "gsk_DIAGNOSTICSECRET123456"
        body = {
            "error": {
                "message": f"rate limited ({secret})",
                "metadata": {
                    "error_type": "rate_limit_exceeded",
                    "provider_code": "rate_limited",
                    "provider_name": "ExampleProvider",
                    "raw": "x" * 5000,
                },
            }
        }
        session = FakeSession([FakeResponse(429, body)])
        client = GroqClient(
            api_key=secret,
            model="openai/gpt-oss-120b",
            session=session,
            sleep_fn=lambda _delay: None,
            base_delay=0.01,
            max_attempts=1,
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        error = response.error or ""
        self.assertIn("rate_limit_exceeded", error)
        self.assertIn("rate_limited", error)
        self.assertIn("ExampleProvider", error)
        self.assertNotIn(secret, error)
        self.assertNotIn("DIAGNOSTICSECRET", error)
        self.assertNotIn("Authorization", error)
        self.assertLessEqual(len(error), 500)


class TestEmptyContentClassification(unittest.TestCase):
    def make_client(self, responses):
        session = FakeSession(responses)
        client = GroqClient(
            api_key="test",
            model="openai/gpt-oss-120b",
            session=session,
            sleep_fn=lambda _delay: None,
            base_delay=0.01,
        )
        return client, session

    def test_empty_content_is_classified_and_not_retried(self):
        body = {
            "choices": [
                {"message": {"content": ""}, "finish_reason": "length"}
            ],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 4096}},
        }
        client, session = self.make_client(
            [FakeResponse(200, body), success_response('{"products": []}')]
        )
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertFalse(response.ok)
        self.assertEqual(response.error_kind, "empty_content")
        self.assertEqual(response.attempts, 1)
        self.assertEqual(len(session.calls), 1)
        self.assertIn("finish_reason=length", response.error or "")
        self.assertIn("reasoning_tokens=4096", response.error or "")

    def test_null_content_is_empty_content(self):
        body = {
            "choices": [
                {"message": {"content": None}, "finish_reason": "length"}
            ]
        }
        client, session = self.make_client([FakeResponse(200, body)])
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(response.error_kind, "empty_content")
        self.assertEqual(len(session.calls), 1)

    def test_missing_choices_is_still_unexpected_shape(self):
        client, session = self.make_client([FakeResponse(200, {"id": "x"})])
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(response.error_kind, "unexpected_response_shape")
        self.assertEqual(len(session.calls), 1)

    def test_non_mapping_message_is_still_unexpected_shape(self):
        body = {"choices": [{"message": "nope"}]}
        client, session = self.make_client([FakeResponse(200, body)])
        response = client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(response.error_kind, "unexpected_response_shape")
        self.assertEqual(len(session.calls), 1)


class TestInputNameAuditability(unittest.TestCase):
    def test_records_store_input_name(self):
        client = GroqClient(api_key="test", model="openai/gpt-oss-120b")
        product = EvalProduct(
            product_key="1",
            sample_name="Legume cu unt XXL",
            package_weight_g=100.0,
            reference_category="Frozen Fruits & Vegetables",
            reference_macro="Frozen",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition={},
        )
        record = runner_mod._base_record(
            client,
            product,
            batch_id="batch-1",
            batch_index=1,
            batch_size=1,
            timestamp="2024-01-01T00:00:00+00:00",
            latency_ms=1.0,
            attempts=1,
            json_valid=True,
        )
        self.assertEqual(record["input_name"], "Legume cu unt XXL")
        self.assertEqual(record["product_key"], "1")

    def test_report_lists_misclassification_with_input_name(self):
        product = EvalProduct(
            product_key="1",
            sample_name="Legume cu unt XXL",
            package_weight_g=100.0,
            reference_category="Frozen Fruits & Vegetables",
            reference_macro="Frozen",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition={},
        )
        dataset = EvalDataset(
            products=(product,),
            seed=1,
            source_path="fixture.csv",
            requested_limit=1,
        )
        record = {
            "provider": "groq",
            "model": "test-model",
            "mode": "estimate",
            "product_key": "1",
            "input_name": product.sample_name,
            "reference": product.reference_snapshot(),
            "prediction": {"category": "Fresh Fruits & Vegetables"},
            "validation": {"valid": True, "issues": [], "batch_issues": []},
            "status": "ok",
        }
        summary = build_summary(
            [record], dataset, {"groq": "test-model"}, "estimate"
        )
        rendered = render_report(summary)
        self.assertIn(
            "Legume cu unt XXL | Frozen Fruits & Vegetables -> "
            "Fresh Fruits & Vegetables",
            rendered,
        )


class TestAbstainRecords(unittest.TestCase):
    def make_product(self, key: str = "1") -> EvalProduct:
        return EvalProduct(
            product_key=key,
            sample_name="Legume cu unt",
            package_weight_g=100.0,
            reference_category="Dairy & Eggs",
            reference_macro="Fresh Food",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition={},
        )

    def make_client(self) -> GroqClient:
        return GroqClient(api_key="test", model="openai/gpt-oss-120b")

    def test_clean_abstention_record_is_ok(self):
        client = self.make_client()
        product = self.make_product()
        content = json.dumps(
            make_payload(
                {
                    "product_key": "1",
                    "category": None,
                    "abstain_reason": "Name too short to classify",
                }
            )
        )
        response = ProviderResponse(
            content=content, status_code=200, attempts=1, latency_ms=1.0
        )
        records = runner_mod._batch_records(
            client,
            [product],
            response,
            batch_id="batch-1",
            batch_index=1,
            timestamp="2024-01-01T00:00:00+00:00",
            taxonomy=LIVE_TAXONOMY,
            mode="estimate",
            abstain=True,
        )
        record = records[0]
        self.assertEqual(record["status"], "ok")
        self.assertTrue(record["validation"]["valid"])
        self.assertEqual(record["validation"]["issues"], [])
        self.assertIsNone(record["prediction"]["category"])
        self.assertIs(record["prediction"]["abstained"], True)
        self.assertEqual(
            record["prediction"]["abstain_reason"], "Name too short to classify"
        )
        self.assertIsNone(record["prediction"]["macro_category"])
        self.assertIs(record["abstain"], True)

    def test_same_null_reply_is_flagged_in_a_forced_run(self):
        client = self.make_client()
        product = self.make_product()
        content = json.dumps(
            make_payload(
                {
                    "product_key": "1",
                    "category": None,
                    "abstain_reason": "Name too short to classify",
                }
            )
        )
        response = ProviderResponse(
            content=content, status_code=200, attempts=1, latency_ms=1.0
        )
        batch_kwargs = dict(
            batch_id="batch-1",
            batch_index=1,
            timestamp="2024-01-01T00:00:00+00:00",
            taxonomy=LIVE_TAXONOMY,
            mode="estimate",
        )
        forced = runner_mod._batch_records(
            client, [product], response, abstain=False, **batch_kwargs
        )[0]
        abstained = runner_mod._batch_records(
            client, [product], response, abstain=True, **batch_kwargs
        )[0]

        # The exact same mocked reply is a protocol violation in the forced
        # baseline and a clean answer in the abstain variant.
        self.assertEqual(forced["status"], "flagged")
        self.assertFalse(forced["validation"]["valid"])
        self.assertIn(
            "abstention_not_allowed",
            [issue["code"] for issue in forced["validation"]["issues"]],
        )
        self.assertIs(forced["prediction"]["abstained"], True)
        self.assertIsNone(forced["prediction"]["category"])

        self.assertEqual(abstained["status"], "ok")
        self.assertTrue(abstained["validation"]["valid"])
        self.assertEqual(abstained["validation"]["issues"], [])
        self.assertIs(abstained["prediction"]["abstained"], True)
        self.assertIsNone(abstained["prediction"]["category"])

    def test_flagged_abstention_record_is_not_ok(self):
        client = self.make_client()
        product = self.make_product()
        content = json.dumps(
            make_payload({"product_key": "1", "category": None})
        )
        response = ProviderResponse(
            content=content, status_code=200, attempts=1, latency_ms=1.0
        )
        record = runner_mod._batch_records(
            client,
            [product],
            response,
            batch_id="batch-1",
            batch_index=1,
            timestamp="2024-01-01T00:00:00+00:00",
            taxonomy=LIVE_TAXONOMY,
            mode="estimate",
            abstain=True,
        )[0]
        self.assertEqual(record["status"], "flagged")
        self.assertIs(record["prediction"]["abstained"], True)
        self.assertIn(
            "missing_abstain_reason",
            [issue["code"] for issue in record["validation"]["issues"]],
        )

    def test_missing_category_key_record_is_flagged(self):
        client = self.make_client()
        product = self.make_product()
        content = json.dumps(make_payload({"product_key": "1"}))
        response = ProviderResponse(
            content=content, status_code=200, attempts=1, latency_ms=1.0
        )
        record = runner_mod._batch_records(
            client,
            [product],
            response,
            batch_id="batch-1",
            batch_index=1,
            timestamp="2024-01-01T00:00:00+00:00",
            taxonomy=LIVE_TAXONOMY,
            mode="estimate",
            abstain=True,
        )[0]
        # A missing category is an abstention, but never a silent success.
        self.assertEqual(record["status"], "flagged")
        self.assertIs(record["prediction"]["abstained"], True)
        self.assertIsNone(record["prediction"]["category"])
        codes = [issue["code"] for issue in record["validation"]["issues"]]
        self.assertIn("missing_category", codes)
        self.assertIn("missing_abstain_reason", codes)

    def test_every_record_carries_the_abstain_boolean(self):
        client = self.make_client()
        product = self.make_product()
        for abstain in (False, True):
            with self.subTest(abstain=abstain):
                base = runner_mod._base_record(
                    client,
                    product,
                    batch_id="batch-1",
                    batch_index=1,
                    batch_size=1,
                    timestamp="2024-01-01T00:00:00+00:00",
                    latency_ms=1.0,
                    attempts=1,
                    json_valid=True,
                    abstain=abstain,
                )
                self.assertIs(base["abstain"], abstain)
                response = ProviderResponse(
                    content=None,
                    status_code=500,
                    attempts=1,
                    latency_ms=1.0,
                    error="boom",
                    error_kind="http_error",
                )
                error_records = runner_mod._error_records(
                    client,
                    [product],
                    response,
                    batch_id="batch-1",
                    batch_index=1,
                    timestamp="2024-01-01T00:00:00+00:00",
                    error_kind="http_error",
                    error_message="boom",
                    json_valid=False,
                    abstain=abstain,
                )
                self.assertIs(error_records[0]["abstain"], abstain)
        # An unset flag defaults to the forced-choice baseline, never to None.
        default_record = runner_mod._base_record(
            client,
            product,
            batch_id="batch-1",
            batch_index=1,
            batch_size=1,
            timestamp="2024-01-01T00:00:00+00:00",
            latency_ms=1.0,
            attempts=1,
            json_valid=True,
        )
        self.assertIs(default_record["abstain"], False)


class TestPerCategoryAccuracy(unittest.TestCase):
    def _product(self, key: str, name: str, category: str) -> EvalProduct:
        return EvalProduct(
            product_key=key,
            sample_name=name,
            package_weight_g=100.0,
            reference_category=category,
            reference_macro="Fresh Food",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition={},
        )

    def _record(self, key: str, name: str, product: EvalProduct, predicted: str):
        return {
            "provider": "groq",
            "model": "test-model",
            "mode": "estimate",
            "product_key": key,
            "input_name": name,
            "reference": product.reference_snapshot(),
            "prediction": {"category": predicted},
            "validation": {"valid": True, "issues": [], "batch_issues": []},
            "status": "ok",
        }

    def test_by_category_counts_and_worst_first_rendering(self):
        dairy_ok = self._product("1", "Dairy ok", "Dairy & Eggs")
        dairy_wrong = self._product("2", "Dairy wrong", "Dairy & Eggs")
        meat_wrong = self._product("3", "Meat wrong", "Fresh Meat & Fish")
        dataset = EvalDataset(
            products=(dairy_ok, dairy_wrong, meat_wrong),
            seed=1,
            source_path="fixture.csv",
            requested_limit=3,
        )
        records = [
            self._record("1", "Dairy ok", dairy_ok, "Dairy & Eggs"),
            self._record("2", "Dairy wrong", dairy_wrong, "Frozen"),
            self._record("3", "Meat wrong", meat_wrong, "Dairy & Eggs"),
        ]
        summary = build_summary(
            records, dataset, {"groq": "test-model"}, "estimate"
        )
        provider = summary["providers"]["groq"]
        by_category = provider["category_accuracy"]["by_category"]
        # A forced run answers every product: coverage is total and accuracy on
        # answered equals plain accuracy.
        self.assertEqual(
            by_category["Dairy & Eggs"],
            {
                "correct": 1,
                "total": 2,
                "accuracy": 0.5,
                "answered": 2,
                "abstained": 0,
                "missing": 0,
                "coverage": 1.0,
                "accuracy_on_answered": 0.5,
            },
        )
        self.assertEqual(
            by_category["Fresh Meat & Fish"],
            {
                "correct": 0,
                "total": 1,
                "accuracy": 0.0,
                "answered": 1,
                "abstained": 0,
                "missing": 0,
                "coverage": 1.0,
                "accuracy_on_answered": 0.0,
            },
        )
        # The headline total must equal the sum of per-category totals.
        self.assertEqual(
            provider["category_accuracy"]["overall"]["total"],
            sum(metrics["total"] for metrics in by_category.values()),
        )
        rendered = render_report(summary)
        self.assertIn("Accuracy by category", rendered)
        table_start = rendered.index("#### Accuracy by category")
        table_end = rendered.index("#### Misclassified")
        table = rendered[table_start:table_end]
        self.assertLess(
            table.index("Fresh Meat & Fish"), table.index("Dairy & Eggs")
        )


class TestAbstentionMetrics(unittest.TestCase):
    TS = "2024-01-01T00:00:00+00:00"

    def make_product(self, key: str, category: str) -> EvalProduct:
        return EvalProduct(
            product_key=key,
            sample_name=f"Product {key}",
            package_weight_g=100.0,
            reference_category=category,
            reference_macro="Fresh Food",
            reference_confidence="HIGH",
            match_status="MANUAL_CORRECTED",
            nutrition={},
        )

    def make_dataset(self, *products: EvalProduct) -> EvalDataset:
        return EvalDataset(
            products=tuple(products),
            seed=1,
            source_path="fixture.csv",
            requested_limit=len(products),
        )

    def make_record(
        self,
        product: EvalProduct,
        predicted: str | None,
        *,
        abstain: bool,
    ) -> dict[str, object]:
        return {
            "provider": "groq",
            "model": "test-model",
            "mode": "estimate",
            "product_key": product.product_key,
            "input_name": product.sample_name,
            "reference": product.reference_snapshot(),
            "prediction": {
                "category": predicted,
                "abstained": predicted is None,
                "raw_category": predicted,
                "macro_category": None if predicted is None else "Fresh Food",
                "nutrition": {},
            },
            "validation": {"valid": True, "issues": [], "batch_issues": []},
            "status": "ok",
            "abstain": abstain,
        }

    def outcomes_for(self, records, dataset, *, abstain):
        summary = build_summary(
            records,
            dataset,
            {"groq": "test-model"},
            "estimate",
            abstain=abstain,
        )
        return summary, summary["providers"]["groq"]["category_accuracy"]

    def test_forced_run_outcome_identities(self):
        dairy = self.make_product("1", "Dairy & Eggs")
        meat = self.make_product("2", "Fresh Meat & Fish")
        wrong = self.make_product("3", "Fresh Meat & Fish")
        dataset = self.make_dataset(dairy, meat, wrong)
        records = [
            self.make_record(dairy, "Dairy & Eggs", abstain=False),
            self.make_record(meat, "Fresh Meat & Fish", abstain=False),
            self.make_record(wrong, "Dairy & Eggs", abstain=False),
        ]
        _summary, accuracy = self.outcomes_for(records, dataset, abstain=False)
        outcomes = accuracy["outcomes"]
        self.assertEqual(outcomes["total"], 3)
        self.assertEqual(outcomes["answered"], 3)
        self.assertEqual(outcomes["abstained"], 0)
        self.assertEqual(outcomes["missing"], 0)
        self.assertEqual(outcomes["correct"], 2)
        self.assertEqual(outcomes["wrong"], 1)
        self.assertEqual(outcomes["coverage"], 1.0)
        self.assertEqual(outcomes["abstention_rate"], 0.0)
        self.assertEqual(outcomes["missing_rate"], 0.0)
        # A forced run: answered accuracy and overall accuracy must coincide.
        self.assertEqual(
            outcomes["accuracy_on_answered"], accuracy["overall"]["accuracy"]
        )
        # The headline total and the outcome total describe the same rows.
        self.assertEqual(accuracy["overall"]["total"], outcomes["total"])
        self.assertNotIn("accuracy_on_all", outcomes)

    def test_abstain_run_metrics(self):
        dairy = self.make_product("1", "Dairy & Eggs")
        meat = self.make_product("2", "Fresh Meat & Fish")
        vague = self.make_product("3", "Dairy & Eggs")
        dataset = self.make_dataset(dairy, meat, vague)
        records = [
            self.make_record(dairy, "Dairy & Eggs", abstain=True),
            self.make_record(meat, "Dairy & Eggs", abstain=True),
            self.make_record(vague, None, abstain=True),
        ]
        _summary, accuracy = self.outcomes_for(records, dataset, abstain=True)
        outcomes = accuracy["outcomes"]
        self.assertEqual(outcomes["total"], 3)
        self.assertEqual(outcomes["answered"], 2)
        self.assertEqual(outcomes["abstained"], 1)
        self.assertEqual(outcomes["missing"], 0)
        self.assertEqual(outcomes["correct"], 1)
        self.assertEqual(outcomes["wrong"], 1)
        self.assertAlmostEqual(outcomes["coverage"], 2 / 3)
        self.assertAlmostEqual(outcomes["abstention_rate"], 1 / 3)
        self.assertAlmostEqual(outcomes["missing_rate"], 0.0)
        # Accuracy on answered is computed over the answered rows only.
        self.assertAlmostEqual(outcomes["accuracy_on_answered"], 0.5)
        # The abstention is never counted as correct or as wrong.
        self.assertEqual(
            outcomes["correct"] + outcomes["wrong"], outcomes["answered"]
        )
        self.assertNotIn("accuracy_on_all", outcomes)
        self.assertEqual(accuracy["overall"]["correct"], 1)
        self.assertEqual(accuracy["overall"]["total"], outcomes["total"])

    def test_omitted_product_is_missing_not_abstained(self):
        dairy_ok = self.make_product("1", "Dairy & Eggs")
        dairy_vague = self.make_product("2", "Dairy & Eggs")
        dairy_omitted = self.make_product("3", "Dairy & Eggs")
        dataset = self.make_dataset(dairy_ok, dairy_vague, dairy_omitted)
        omitted = self.make_record(dairy_omitted, None, abstain=True)
        # The runner stores no usable prediction for a product the model never
        # returned: abstained is false, status is flagged.
        omitted["prediction"] = {
            "category": None,
            "abstained": False,
            "raw_category": None,
            "macro_category": None,
            "nutrition": {},
        }
        omitted["validation"] = {
            "valid": False,
            "issues": [{"code": "missing_product", "message": "not returned"}],
            "batch_issues": [],
        }
        omitted["status"] = "flagged"
        records = [
            self.make_record(dairy_ok, "Dairy & Eggs", abstain=True),
            self.make_record(dairy_vague, None, abstain=True),
            omitted,
        ]

        _summary, accuracy = self.outcomes_for(records, dataset, abstain=True)
        outcomes = accuracy["outcomes"]
        self.assertEqual(outcomes["total"], 3)
        self.assertEqual(outcomes["answered"], 1)
        self.assertEqual(outcomes["abstained"], 1)
        self.assertEqual(outcomes["missing"], 1)
        self.assertEqual(outcomes["correct"], 1)
        self.assertEqual(outcomes["wrong"], 0)
        # The invariant holds, and missing is not folded into abstentions.
        self.assertEqual(
            outcomes["answered"] + outcomes["abstained"] + outcomes["missing"],
            outcomes["total"],
        )
        self.assertAlmostEqual(outcomes["coverage"], 1 / 3)
        self.assertAlmostEqual(outcomes["abstention_rate"], 1 / 3)
        self.assertAlmostEqual(outcomes["missing_rate"], 1 / 3)
        self.assertEqual(accuracy["overall"]["total"], outcomes["total"])
        self.assertEqual(accuracy["overall"]["correct"], 1)

        row = accuracy["by_category"]["Dairy & Eggs"]
        self.assertEqual(row["total"], 3)
        self.assertEqual(row["answered"], 1)
        self.assertEqual(row["abstained"], 1)
        self.assertEqual(row["missing"], 1)
        self.assertAlmostEqual(row["coverage"], 1 / 3)
        self.assertAlmostEqual(row["accuracy_on_answered"], 1.0)

    def test_variants_are_reported_separately(self):
        product = self.make_product("1", "Dairy & Eggs")
        dataset = self.make_dataset(product)
        forced = self.make_record(product, "Dairy & Eggs", abstain=False)
        abstained = self.make_record(product, None, abstain=True)
        records = [forced, abstained]

        forced_summary, forced_accuracy = self.outcomes_for(
            records, dataset, abstain=False
        )
        abstain_summary, abstain_accuracy = self.outcomes_for(
            records, dataset, abstain=True
        )

        self.assertEqual(forced_accuracy["outcomes"]["answered"], 1)
        self.assertEqual(forced_accuracy["outcomes"]["abstained"], 0)
        self.assertEqual(forced_accuracy["outcomes"]["correct"], 1)
        self.assertEqual(abstain_accuracy["outcomes"]["answered"], 0)
        self.assertEqual(abstain_accuracy["outcomes"]["abstained"], 1)
        self.assertEqual(abstain_accuracy["outcomes"]["missing"], 0)
        self.assertEqual(abstain_accuracy["outcomes"]["coverage"], 0.0)
        self.assertIsNone(abstain_accuracy["outcomes"]["accuracy_on_answered"])
        self.assertEqual(abstain_accuracy["overall"]["correct"], 0)
        self.assertEqual(abstain_accuracy["overall"]["total"], 1)
        self.assertIs(forced_summary["abstain"], False)
        self.assertIs(abstain_summary["abstain"], True)
        # An abstention is never listed as a misclassification.
        self.assertEqual(
            abstain_summary["providers"]["groq"]["misclassifications"], []
        )

    def test_per_category_counts_reconcile_with_totals(self):
        dairy_ok = self.make_product("1", "Dairy & Eggs")
        dairy_vague = self.make_product("2", "Dairy & Eggs")
        meat_wrong = self.make_product("3", "Fresh Meat & Fish")
        dataset = self.make_dataset(dairy_ok, dairy_vague, meat_wrong)
        records = [
            self.make_record(dairy_ok, "Dairy & Eggs", abstain=True),
            self.make_record(dairy_vague, None, abstain=True),
            self.make_record(meat_wrong, "Dairy & Eggs", abstain=True),
        ]
        _summary, accuracy = self.outcomes_for(records, dataset, abstain=True)
        outcomes = accuracy["outcomes"]
        by_category = accuracy["by_category"]

        self.assertEqual(
            by_category["Dairy & Eggs"]["correct"],
            by_category["Dairy & Eggs"]["answered"],
        )
        self.assertEqual(by_category["Dairy & Eggs"]["total"], 2)
        self.assertEqual(by_category["Dairy & Eggs"]["answered"], 1)
        self.assertEqual(by_category["Dairy & Eggs"]["abstained"], 1)
        self.assertEqual(by_category["Dairy & Eggs"]["missing"], 0)
        self.assertEqual(by_category["Dairy & Eggs"]["coverage"], 0.5)
        self.assertEqual(by_category["Dairy & Eggs"]["accuracy_on_answered"], 1.0)
        self.assertEqual(by_category["Fresh Meat & Fish"]["total"], 1)
        self.assertEqual(by_category["Fresh Meat & Fish"]["answered"], 1)
        self.assertEqual(by_category["Fresh Meat & Fish"]["abstained"], 0)
        self.assertEqual(by_category["Fresh Meat & Fish"]["missing"], 0)
        self.assertEqual(
            by_category["Fresh Meat & Fish"]["accuracy_on_answered"], 0.0
        )
        for field in ("total", "answered", "abstained", "missing", "correct"):
            self.assertEqual(
                sum(row[field] for row in by_category.values()),
                outcomes[field],
                field,
            )

    def test_report_shows_abstention_section_and_columns(self):
        dairy = self.make_product("1", "Dairy & Eggs")
        vague = self.make_product("2", "Dairy & Eggs")
        dataset = self.make_dataset(dairy, vague)
        records = [
            self.make_record(dairy, "Dairy & Eggs", abstain=True),
            self.make_record(vague, None, abstain=True),
        ]
        summary, _accuracy = self.outcomes_for(records, dataset, abstain=True)
        rendered = render_report(summary)
        self.assertIn("Prompt variant: abstain-allowed", rendered)
        self.assertIn("## Abstention", rendered)
        self.assertIn("Coverage (answered / total)", rendered)
        self.assertIn("Abstention rate", rendered)
        self.assertIn("Missing rate", rendered)
        self.assertIn("Accuracy on answered", rendered)
        self.assertNotIn("Accuracy on all evaluated", rendered)
        self.assertIn(
            "| Category | Correct | Total | Accuracy | Answered | Abstained |",
            rendered,
        )
        self.assertIn("| Missing | Coverage | Accuracy on answered |", rendered)
        self.assertIn("not directly comparable", rendered)
        # The existing LOW-reference section keeps its meaning.
        self.assertIn(
            "Agreement with unverified LOW references (not accuracy)", rendered
        )

    def test_resume_identity_includes_the_variant(self):
        base = {
            "provider": "groq",
            "model": "m",
            "mode": "estimate",
            "product_key": "1",
            "status": "ok",
            "validation": {"valid": True},
        }
        forced = {**base, "abstain": False}
        abstained = {**base, "abstain": True}
        legacy = dict(base)
        self.assertEqual(
            completed_keys([forced], "groq", "m", mode="estimate", abstain=True),
            set(),
        )
        self.assertEqual(
            completed_keys([abstained], "groq", "m", mode="estimate", abstain=False),
            set(),
        )
        self.assertEqual(
            completed_keys([forced], "groq", "m", mode="estimate", abstain=False),
            {"1"},
        )
        self.assertEqual(
            completed_keys([abstained], "groq", "m", mode="estimate", abstain=True),
            {"1"},
        )
        # Legacy rows are forced-choice baseline rows, never abstain rows.
        self.assertEqual(
            completed_keys([legacy], "groq", "m", mode="estimate", abstain=False),
            {"1"},
        )
        self.assertEqual(
            completed_keys([legacy], "groq", "m", mode="estimate", abstain=True),
            set(),
        )
        both = [forced, abstained]
        self.assertIs(
            latest_results(both, mode="estimate", abstain=True)[0]["abstain"], True
        )
        self.assertIs(
            latest_results(both, mode="estimate", abstain=False)[0]["abstain"], False
        )


if __name__ == "__main__":
    unittest.main()
