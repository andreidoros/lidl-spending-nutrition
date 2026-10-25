"""Offline tests for the production Groq enrichment stage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import duckdb
import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_enrichment.config import AIEnrichmentConfig, AIEnrichmentConfigError, load_config
from ai_enrichment.provider import GroqProductionClient
from ai_enrichment.service import run_ai_enrichment
from ai_eval.providers import GroqClient, ProviderResponse
from ai_eval.taxonomy import load_taxonomy
from product_store import create_schema


class FakeClient:
    def __init__(self, responses, **kwargs):
        self.responses = list(responses)
        self.kwargs = kwargs
        self.calls = []

    def chat(self, messages, max_tokens=4096, *, deadline=None):
        self.calls.append((messages, max_tokens, deadline))
        if not self.responses:
            raise AssertionError("Unexpected extra model request")
        return self.responses.pop(0)


class FakeHTTPResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self.body


class FakeHTTPSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class AIEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect(":memory:")
        self.con.execute("""
            CREATE TABLE nutrition_cache (
                product_key VARCHAR PRIMARY KEY,
                sample_name VARCHAR,
                macro_category VARCHAR,
                category VARCHAR,
                package_weight_g DOUBLE,
                kcal_100g DOUBLE,
                proteins_100g DOUBLE,
                carbs_100g DOUBLE,
                sugars_100g DOUBLE,
                fat_100g DOUBLE,
                saturated_fat_100g DOUBLE,
                fiber_100g DOUBLE,
                salt_100g DOUBLE,
                match_status VARCHAR,
                data_source VARCHAR,
                is_verified BOOLEAN DEFAULT FALSE,
                is_weighed_default BOOLEAN DEFAULT FALSE,
                atwater_debug VARCHAR
            )
        """)
        create_schema(self.con)
        self.taxonomy = load_taxonomy()
        self.category_to_macro = self.taxonomy.category_to_macro

    def tearDown(self):
        self.con.close()

    def add_product(
        self,
        key="p1",
        *,
        category=None,
        macro=None,
        package_weight=250.0,
        kcal=None,
        protein=None,
        carbs=None,
        sugars=None,
        fat=None,
        saturated=None,
        fiber=None,
        salt=None,
    ):
        self.con.execute(
            """INSERT INTO nutrition_cache
               (product_key, sample_name, macro_category, category, package_weight_g,
                kcal_100g, proteins_100g, carbs_100g, sugars_100g, fat_100g,
                saturated_fat_100g, fiber_100g, salt_100g, match_status, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'TEST', 'OFF_API')""",
            [key, f"Synthetic {key}", macro, category, package_weight, kcal, protein,
             carbs, sugars, fat, saturated, fiber, salt],
        )

    def response(self, content, *, attempts=1):
        return ProviderResponse(content, 200, attempts, 1.0)

    def config(self, mode="categories_and_nutrition", **kwargs):
        return AIEnrichmentConfig(
            mode=mode,
            api_key="test-groq-key",
            batch_size=kwargs.pop("batch_size", 20),
            budget_seconds=kwargs.pop("budget_seconds", 300.0),
            **kwargs,
        )

    def product_json(self, key="p1", **overrides):
        values = {
            "product_key": key,
            "category": "Dairy & Eggs",
            "kcal_100g": 100,
            "proteins_100g": 3,
            "carbs_100g": 5,
            "sugars_100g": 4,
            "fat_100g": 7,
            "saturated_fat_100g": 2,
            "fiber_100g": 1,
            "salt_100g": 0.1,
        }
        values.update(overrides)
        return values

    def payload_response(self, *products):
        return self.response(json.dumps({"products": list(products)}))

    def state(self, key="p1"):
        return self.con.execute(
            "SELECT task, status, attempt_count, provider, model, error_code "
            "FROM ai_enrichment_state WHERE product_key = ? ORDER BY task",
            [key],
        ).fetchall()

    def test_ai_is_disabled_by_default_and_does_not_call_provider(self):
        self.add_product()
        client = FakeClient([])

        report = run_ai_enrichment(
            self.con,
            config=AIEnrichmentConfig(),
            taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(report.status, "disabled")
        self.assertEqual(client.calls, [])
        self.assertEqual(self.state(), [])

    def test_enabled_without_key_skips_without_creating_attempts(self):
        self.add_product()
        config = AIEnrichmentConfig(mode="categories", api_key=None)
        client = FakeClient([])

        report = run_ai_enrichment(
            self.con, config=config, taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual((report.status, report.error_code), ("skipped", "missing_api_key"))
        self.assertEqual(client.calls, [])
        self.assertEqual(self.state(), [])

    def test_invalid_api_key_is_non_blocking_and_secret_is_not_persisted(self):
        self.add_product()
        client = FakeClient([
            ProviderResponse(None, 401, 1, 1.0, "invalid key test-groq-key", "http_error")
        ])

        report = run_ai_enrichment(
            self.con, config=self.config(mode="categories"), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(report.status, "partial")
        self.assertEqual({row[1] for row in self.state()}, {"retryable_error"})
        self.assertEqual({row[5] for row in self.state()}, {"http_401"})
        self.assertNotIn("test-groq-key", str(self.state()))

    def test_empty_provider_reply_is_terminal_and_not_retried_next_sync(self):
        self.add_product()
        client = FakeClient([
            ProviderResponse(None, 200, 1, 1.0, "provider returned empty content", "empty_content")
        ])
        client_factory = lambda **kwargs: client

        first = run_ai_enrichment(
            self.con, config=self.config(mode="categories"), taxonomy=self.taxonomy,
            client_factory=client_factory,
        )
        second = run_ai_enrichment(
            self.con, config=self.config(mode="categories"), taxonomy=self.taxonomy,
            client_factory=client_factory,
        )

        self.assertEqual(first.terminal_tasks, 1)
        self.assertEqual(second.products_considered, 0)
        self.assertEqual(self.state()[0][1], "invalid")
        self.assertEqual(self.state()[0][5], "empty_content")
        self.assertEqual(len(client.calls), 1)

    def test_configuration_is_opt_in_and_free_model_only(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("", encoding="utf-8")
            self.assertEqual(load_config(env_path=env_path, environ={}).mode, "off")
            with self.assertRaises(AIEnrichmentConfigError):
                load_config(
                    env_path=env_path,
                    environ={"AI_ENRICHMENT_MODE": "categories", "GROQ_MODEL": "paid/model"},
                )

    def test_success_saves_category_nutrients_and_ai_provenance(self):
        self.add_product()
        client = FakeClient([self.payload_response(self.product_json())])

        report = run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(report.status, "completed")
        self.assertEqual(report.fields_saved, 10)  # category, derived macro, eight nutrient fields
        self.assertEqual(self.con.execute(
            "SELECT category, macro_category, kcal_100g, salt_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Dairy & Eggs", "Fresh Food", 100.0, 0.1))
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM product_field_metadata WHERE product_key='p1' AND source='AI' AND NOT is_protected"
        ).fetchone()[0], 10)
        self.assertEqual([row[1] for row in self.state()], ["succeeded", "succeeded"])
        self.assertNotIn("test-groq-key", str(self.state()))
        sent = client.calls[0][0][1]["content"]
        self.assertIn("Synthetic p1", sent)
        self.assertNotIn("unit_price", sent)
        self.assertNotIn("ticket_id", sent)

    def test_ai_fill_replaces_stale_unprotected_provenance(self):
        self.add_product(
            category="Dairy & Eggs", macro="Fresh Food", kcal=None, protein=3,
            carbs=5, sugars=4, fat=7, saturated=2, fiber=1, salt=0.1,
        )
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected, provider, model)
               VALUES ('p1', 'kcal_100g', 'OFF', FALSE, NULL, NULL)"""
        )
        client = FakeClient([self.payload_response(self.product_json())])

        run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(self.con.execute(
            "SELECT source, is_protected, provider, model FROM product_field_metadata "
            "WHERE product_key='p1' AND field_name='kcal_100g'"
        ).fetchone(), ("AI", False, "groq", self.config().model))

    def test_categories_only_mode_does_not_persist_nutrition(self):
        self.add_product()
        category_only = {
            "product_key": "p1",
            "category": "Dairy & Eggs",
            "abstain_reason": None,
            **{field: None for field in (
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            )},
        }
        client = FakeClient([self.payload_response(category_only)])

        run_ai_enrichment(
            self.con, config=self.config(mode="categories"), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(self.con.execute(
            "SELECT category, kcal_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Dairy & Eggs", None))
        self.assertEqual(self.state(), [
            ("category", "succeeded", 1, "groq", self.config().model, None),
        ])

    def test_failed_atomic_ai_save_rolls_back_fields_metadata_and_state(self):
        from product_store import save_ai_enrichment_result

        self.add_product()
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            save_ai_enrichment_result(
                self.con,
                "p1",
                category_result={"status": "succeeded", "value": "Dairy & Eggs"},
                nutrition_result={
                    "status": "succeeded",
                    "values": {"kcal_100g": "not numeric"},
                },
                category_to_macro=self.category_to_macro,
                provider="groq",
                model="model",
            )

        self.assertEqual(self.con.execute(
            "SELECT category, macro_category FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), (None, None))
        self.assertEqual(self.state(), [])
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM product_field_metadata WHERE product_key='p1'"
        ).fetchone()[0], 0)

    def test_category_is_kept_when_nutrition_is_invalid_and_is_not_retried(self):
        self.add_product()
        client = FakeClient([self.payload_response(self.product_json(salt_100g=200))])

        first = run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )
        second = run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(first.status, "completed")
        self.assertEqual(second.products_considered, 0)
        self.assertEqual(self.con.execute(
            "SELECT category, kcal_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Dairy & Eggs", None))
        self.assertEqual(self.state(), [
            ("category", "succeeded", 1, "groq", self.config().model, None),
            ("nutrition", "invalid", 1, "groq", self.config().model, "nutrition_out_of_range"),
        ])
        self.assertEqual(len(client.calls), 1)

    def test_transient_request_failure_is_retried_on_next_run(self):
        self.add_product()
        failed = ProviderResponse(None, 503, 3, 10.0, "temporary outage", "http_error")
        client = FakeClient([failed, self.payload_response(self.product_json())])
        client_factory = lambda **kwargs: client

        run_ai_enrichment(self.con, config=self.config(), taxonomy=self.taxonomy, client_factory=client_factory)
        self.assertEqual({row[1] for row in self.state()}, {"retryable_error"})
        run_ai_enrichment(self.con, config=self.config(), taxonomy=self.taxonomy, client_factory=client_factory)

        self.assertEqual([row[1] for row in self.state()], ["succeeded", "succeeded"])
        self.assertEqual([row[2] for row in self.state()], [4, 4])
        self.assertEqual(len(client.calls), 2)

    def test_abstention_is_terminal_until_explicit_future_retry(self):
        self.add_product()
        abstention = {
            "product_key": "p1",
            "category": None,
            "abstain_reason": "The name is not specific enough.",
            **{field: None for field in (
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            )},
        }
        client = FakeClient([self.payload_response(abstention)])
        config = self.config(mode="categories")

        run_ai_enrichment(self.con, config=config, taxonomy=self.taxonomy, client_factory=lambda **kwargs: client)
        later = run_ai_enrichment(self.con, config=config, taxonomy=self.taxonomy, client_factory=lambda **kwargs: client)

        self.assertEqual(self.state()[0][1], "abstained")
        self.assertEqual(later.products_considered, 0)
        self.assertEqual(len(client.calls), 1)

    def test_explicit_retry_only_reenables_selected_terminal_product(self):
        self.add_product("p1")
        self.add_product("p2")
        abstentions = [
            {
                "product_key": key,
                "category": None,
                "abstain_reason": "Unclear product name.",
                **{field: None for field in (
                    "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                    "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
                )},
            }
            for key in ("p1", "p2")
        ]
        config = self.config(mode="categories")
        first_client = FakeClient([self.payload_response(*abstentions)])
        first = run_ai_enrichment(
            self.con,
            config=config,
            taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: first_client,
        )
        self.assertEqual(first.terminal_tasks, 2)
        self.assertEqual(self.state("p1")[0][1], "abstained")
        self.assertEqual(self.state("p2")[0][1], "abstained")

        retry_client = FakeClient([self.payload_response({
            "product_key": "p1",
            "category": "Dairy & Eggs",
            "abstain_reason": None,
            **{field: None for field in (
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            )},
        })])
        retried = run_ai_enrichment(
            self.con,
            config=config,
            taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: retry_client,
            retry_product_keys=("p1",),
        )

        self.assertEqual(retried.products_considered, 1)
        self.assertEqual(len(retry_client.calls), 1)
        self.assertEqual(self.state("p1")[0][1], "succeeded")
        self.assertEqual(self.state("p2")[0][1], "abstained")

    def test_partial_batches_keep_success_and_retry_only_failed_product(self):
        self.add_product("p1")
        self.add_product("p2")
        client = FakeClient([
            self.payload_response(self.product_json("p1")),
            ProviderResponse(None, 429, 3, 10.0, "rate limit", "http_error"),
        ])
        config = self.config(batch_size=1)

        report = run_ai_enrichment(
            self.con, config=config, taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(report.status, "partial")
        self.assertEqual(self.con.execute(
            "SELECT category FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0], "Dairy & Eggs")
        self.assertEqual(self.state("p1")[0][1], "succeeded")
        self.assertEqual({row[1] for row in self.state("p2")}, {"retryable_error"})
        self.assertEqual(len(client.calls), 2)

    def test_duplicate_and_unknown_ids_are_never_applied(self):
        self.add_product()
        content = json.dumps({"products": [
            self.product_json("p1"),
            self.product_json("p1", category="Non-Food"),
            self.product_json("not-requested"),
        ]})
        client = FakeClient([self.response(content)])

        run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertIsNone(self.con.execute(
            "SELECT category FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0])
        self.assertEqual({row[1] for row in self.state()}, {"invalid"})
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM nutrition_cache WHERE product_key='not-requested'"
        ).fetchone()[0], 0)

    def test_unrequested_product_id_invalidates_the_returned_batch(self):
        self.add_product()
        client = FakeClient([self.payload_response(
            self.product_json("p1"),
            self.product_json("unexpected-key"),
        )])

        run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertIsNone(self.con.execute(
            "SELECT category FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0])
        self.assertEqual({row[1] for row in self.state()}, {"invalid"})
        self.assertEqual({row[5] for row in self.state()}, {"extra_product"})
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM nutrition_cache WHERE product_key='unexpected-key'"
        ).fetchone()[0], 0)

    def test_existing_protected_values_are_not_overwritten(self):
        self.add_product(
            category="Dairy & Eggs", macro="Fresh Food", kcal=100, protein=3,
            carbs=5, sugars=None, fat=7, saturated=2, fiber=1, salt=0.5,
        )
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('p1', 'salt_100g', 'MANUAL', TRUE)"""
        )
        client = FakeClient([self.payload_response(self.product_json(salt_100g=8))])

        run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(self.con.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone()[0], 0.5)
        self.assertEqual(self.con.execute(
            "SELECT source, is_protected FROM product_field_metadata "
            "WHERE product_key='p1' AND field_name='salt_100g'"
        ).fetchone(), ("MANUAL", True))

    def test_protected_blank_category_is_not_sent_for_ai_completion(self):
        self.add_product()
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('p1', 'category', 'MANUAL', TRUE)"""
        )
        client = FakeClient([])

        report = run_ai_enrichment(
            self.con, config=self.config(mode="categories"), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(report.products_considered, 0)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.state(), [])

    def test_ai_non_food_cannot_clear_manual_protected_nutrition(self):
        self.add_product(kcal=120, protein=4, carbs=10, sugars=3, fat=6, saturated=2, salt=0.3)
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('p1', 'kcal_100g', 'MANUAL', TRUE)"""
        )
        non_food = {
            "product_key": "p1", "category": "Non-Food", "abstain_reason": None,
            **{field: None for field in (
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            )},
        }
        client = FakeClient([self.payload_response(non_food)])

        run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(self.con.execute(
            "SELECT category, kcal_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), (None, 120.0))
        self.assertEqual(self.con.execute(
            "SELECT source, is_protected FROM product_field_metadata "
            "WHERE product_key='p1' AND field_name='kcal_100g'"
        ).fetchone(), ("MANUAL", True))
        self.assertEqual(self.state(), [
            ("category", "invalid", 1, "groq", self.config().model, "protected_nutrition_conflict"),
        ])

    def test_non_food_transition_preserves_protected_metadata_for_null_nutrient(self):
        self.add_product()
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('p1', 'proteins_100g', 'MANUAL', TRUE)"""
        )
        non_food = {
            "product_key": "p1", "category": "Non-Food", "abstain_reason": None,
            **{field: None for field in (
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            )},
        }
        client = FakeClient([self.payload_response(non_food)])

        run_ai_enrichment(
            self.con, config=self.config(mode="categories"), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(self.con.execute(
            "SELECT category, proteins_100g FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Non-Food", None))
        self.assertEqual(self.con.execute(
            "SELECT source, is_protected FROM product_field_metadata "
            "WHERE product_key='p1' AND field_name='proteins_100g'"
        ).fetchone(), ("MANUAL", True))

    def test_non_food_classification_clears_nutrition_but_keeps_package_weight(self):
        self.add_product(package_weight=450, kcal=100, protein=3, carbs=5, sugars=4,
                         fat=7, saturated=2, fiber=1, salt=0.1)
        self.con.execute(
            """INSERT INTO product_field_metadata
               (product_key, field_name, source, is_protected)
               VALUES ('p1', 'kcal_100g', 'OFF', FALSE)"""
        )
        non_food = {
            "product_key": "p1", "category": "Non-Food", "abstain_reason": None,
            **{field: None for field in (
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            )},
        }
        client = FakeClient([self.payload_response(non_food)])

        run_ai_enrichment(
            self.con, config=self.config(), taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
        )

        self.assertEqual(self.con.execute(
            "SELECT category, package_weight_g, kcal_100g, salt_100g, match_status, is_verified "
            "FROM nutrition_cache WHERE product_key='p1'"
        ).fetchone(), ("Non-Food", 450.0, None, None, "NON_FOOD", True))
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM product_field_metadata WHERE product_key='p1' "
            "AND field_name IN ('kcal_100g', 'proteins_100g', 'carbs_100g', 'sugars_100g', "
            "'fat_100g', 'saturated_fat_100g', 'fiber_100g', 'salt_100g')"
        ).fetchone()[0], 0)

    def test_food_reclassification_resets_old_ai_nutrition_attempt(self):
        self.add_product(category="Non-Food", macro="Non-Food")
        self.con.execute(
            """INSERT INTO ai_enrichment_state
               (product_key, task, status, attempt_count, provider, model)
               VALUES ('p1', 'nutrition', 'succeeded', 1, 'groq', 'model')"""
        )
        from product_store import save_manual_fields

        save_manual_fields(self.con, "p1", {"category": "Dairy & Eggs"}, self.category_to_macro)

        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM ai_enrichment_state WHERE product_key='p1' AND task='nutrition'"
        ).fetchone()[0], 0)

    def test_provider_retries_429_and_5xx_but_obeys_deadline(self):
        success_body = {"choices": [{"message": {"content": '{"products": []}'}}]}
        session = FakeHTTPSession([
            FakeHTTPResponse(429, {"error": {"message": "rate limited"}}),
            FakeHTTPResponse(503, {"error": {"message": "temporary"}}),
            FakeHTTPResponse(200, success_body),
        ])
        client = GroqClient(
            api_key="test-key", model="openai/gpt-oss-120b", session=session,
            sleep_fn=lambda _seconds: None,
        )
        result = client.chat([{"role": "user", "content": "test"}], deadline=time_monotonic() + 10)
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(session.calls), 3)

        delayed = FakeHTTPSession([
            FakeHTTPResponse(429, {"error": {"message": "rate limited"}}, {"Retry-After": "15"})
        ])
        sleeps = []
        client = GroqClient(
            api_key="test-key", model="openai/gpt-oss-120b", session=delayed,
            sleep_fn=sleeps.append,
        )
        with patch("ai_eval.providers.time.monotonic", return_value=100.0):
            result = client.chat([{"role": "user", "content": "test"}], deadline=110.0)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "deadline_exceeded")
        self.assertEqual(sleeps, [])

    def test_production_groq_transport_uses_mocked_http(self):
        requests_seen = []
        body = {"choices": [{"message": {"content": '{"products": []}'}}]}

        def handler(request):
            requests_seen.append(request)
            return httpx.Response(200, json=body)

        client = GroqProductionClient(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            transport=httpx.MockTransport(handler),
            sleep_fn=lambda _seconds: None,
        )
        result = client.chat(
            [{"role": "user", "content": "test"}],
            deadline=time.monotonic() + 2,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.content, '{"products": []}')
        self.assertEqual(len(requests_seen), 1)
        self.assertEqual(requests_seen[0].headers["authorization"], "Bearer test-key")

    def test_production_groq_transport_retries_rate_limit_and_server_errors(self):
        statuses = [429, 503, 200]
        requests_seen = []
        body = {"choices": [{"message": {"content": '{"products": []}'}}]}

        def handler(request):
            requests_seen.append(request)
            status = statuses.pop(0)
            return httpx.Response(status, json=body if status == 200 else {"error": "temporary"})

        sleeps = []
        client = GroqProductionClient(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            transport=httpx.MockTransport(handler),
            sleep_fn=sleeps.append,
        )
        result = client.chat(
            [{"role": "user", "content": "test"}],
            deadline=time.monotonic() + 10,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(requests_seen), 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_remote_protocol_disconnect_is_retryable_on_later_sync(self):
        self.add_product()
        category_only = {
            "product_key": "p1", "category": "Dairy & Eggs", "abstain_reason": None,
            **{field: None for field in (
                "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
                "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
            )},
        }
        failing_transport = httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                httpx.RemoteProtocolError("peer disconnected")
            )
        )
        successful_transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"choices": [{
                "message": {"content": json.dumps({"products": [category_only]})}
            }]})
        )
        created_clients = 0

        def client_factory(**kwargs):
            nonlocal created_clients
            created_clients += 1
            return GroqProductionClient(
                api_key=kwargs["api_key"],
                model=kwargs["model"],
                timeout=kwargs["timeout"],
                max_attempts=kwargs["max_attempts"],
                sleep_fn=kwargs["sleep_fn"],
                transport=failing_transport if created_clients == 1 else successful_transport,
            )

        config = self.config(mode="categories")
        run_ai_enrichment(
            self.con, config=config, taxonomy=self.taxonomy,
            client_factory=client_factory, sleep_fn=lambda _seconds: None,
        )
        self.assertEqual(self.state()[0][1], "retryable_error")

        run_ai_enrichment(
            self.con, config=config, taxonomy=self.taxonomy,
            client_factory=client_factory, sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(self.state()[0][1:3], ("succeeded", 4))

    def test_trickling_groq_response_is_cancelled_at_absolute_deadline(self):
        class TricklingBody(httpx.AsyncByteStream):
            def __init__(self):
                self.closed = False

            async def __aiter__(self):
                yield b'{"choices":'
                while True:
                    await asyncio.sleep(0.01)
                    yield b" "

            async def aclose(self):
                self.closed = True

        stream = TricklingBody()
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=stream)
        )
        client = GroqProductionClient(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            timeout=5,
            transport=transport,
            sleep_fn=lambda _seconds: None,
        )
        started = time.monotonic()
        result = client.chat(
            [{"role": "user", "content": "test"}],
            deadline=started + 0.12,
        )
        elapsed = time.monotonic() - started

        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "deadline_exceeded")
        self.assertLess(elapsed, 0.5)
        self.assertTrue(stream.closed)

    def test_deeply_nested_http_json_is_terminal_and_non_blocking(self):
        self.add_product()
        nested_member = b'{"nested":' * 20000 + b"0" + b"}" * 20000
        deeply_nested = (
            b'{"choices":[{"message":{"content":"{\\"products\\": []}"},"extra":'
            + nested_member
            + b"}]}"
        )
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, content=deeply_nested)
        )

        report = run_ai_enrichment(
            self.con,
            config=self.config(mode="categories"),
            taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: GroqProductionClient(
                api_key=kwargs["api_key"],
                model=kwargs["model"],
                timeout=kwargs["timeout"],
                max_attempts=kwargs["max_attempts"],
                sleep_fn=kwargs["sleep_fn"],
                transport=transport,
            ),
        )

        self.assertEqual(report.status, "partial")
        self.assertEqual(self.state()[0][1], "invalid")
        self.assertEqual(self.state()[0][5], "invalid_json_response")

    def test_deadline_marks_unstarted_tasks_retryable(self):
        self.add_product()
        client = FakeClient([])
        times = iter([0.0, 301.0])

        report = run_ai_enrichment(
            self.con,
            config=self.config(budget_seconds=300.0),
            taxonomy=self.taxonomy,
            client_factory=lambda **kwargs: client,
            clock=lambda: next(times),
        )

        self.assertEqual(report.status, "partial")
        self.assertTrue(all(row[1] == "retryable_error" for row in self.state()))
        self.assertTrue(all(row[2] == 0 for row in self.state()))
        self.assertEqual(client.calls, [])


def time_monotonic():
    import time

    return time.monotonic()


if __name__ == "__main__":
    unittest.main()
