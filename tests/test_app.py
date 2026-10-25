"""Synthetic Streamlit smoke tests; pipeline operations are mocked."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from ai_enrichment.chat_batch import (
    ChatBatchPreview,
    ChatBatchSaveReport,
    ChatBatchSuggestion,
)
import pipeline_services
from pipeline_services import ReviewSaveReport
from ai_enrichment.service import AIEnrichmentReport
from app import NUMERIC_COLUMNS, TABLE_COLUMNS, _blank


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


@contextmanager
def synthetic_review_app():
    """An enabled Groq UI whose data and external operations are all mocked."""
    products = [
        {"product_key": key, "sample_name": f"Synthetic {key}",
         **{field: None for field in TABLE_COLUMNS[1:]}}
        for key in ("demo-1", "demo-2")
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        database = root / "synthetic.duckdb"
        database.touch()
        with patch.dict(os.environ, {
            "LIDL_APP_DB_PATH": str(database),
            "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
            "LIDL_APP_ENV_PATH": str(root / ".env"),
            "AI_ENRICHMENT_MODE": "categories_and_nutrition",
            "GROQ_API_KEY": "synthetic-key-not-a-real-credential",
        }), patch.object(pipeline_services, "read_review_products", return_value=products) as read, patch.object(
            pipeline_services, "retry_ai_products",
            return_value=pipeline_services.AIRetryReport(AIEnrichmentReport("completed"), ()),
        ) as groq, patch.object(pipeline_services, "run_sync") as sync:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.button(key="open_review").click().run(timeout=30)
            yield app, products, read, groq, sync


class StreamlitAppTests(unittest.TestCase):
    def test_review_is_disabled_before_sync_service_starts_and_enabled_after(self):
        rendered = {}
        real_button = st.button

        def capture_button(label, *args, **kwargs):
            result = real_button(label, *args, **kwargs)
            rendered[kwargs.get("key")] = kwargs.get("disabled", False)
            return result

        with synthetic_review_app() as (app, products, read, groq, sync):
            def during_sync(*args, **kwargs):
                self.assertTrue(rendered["open_review"], "Review is clickable during Sync")

            sync.side_effect = during_sync
            with patch.object(st, "button", side_effect=capture_button):
                app.button(key="sync_pipeline").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error, [error.value for error in app.error])
            sync.assert_called_once()
            app.run()
            self.assertFalse(app.button(key="open_review").disabled)

    def test_single_review_button_after_sync_and_new_title(self):
        with synthetic_review_app() as (app, products, read, groq, sync):
            app.button(key="sync_pipeline").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, "Lidl Spending & Nutrition")
            caption_values = [caption.value for caption in app.caption]
            self.assertFalse(any("ReceiptEngine" in value for value in caption_values))
            self.assertNotIn("Local ETL & Catalog Manager", caption_values)
            self.assertNotIn(
                "Ingestion and enrichment pipeline for local JSON receipts.",
                caption_values,
            )
            self.assertEqual(
                app.caption[-1].value,
                "Independent project. Not affiliated with Lidl.",
            )
            self.assertEqual(sum(button.label == "Review products" for button in app.button), 1)
            self.assertNotIn("Review now", [button.label for button in app.button])
            self.assertNotIn("Open product review", [button.label for button in app.button])
            self.assertTrue(app.session_state["sync_completion_prompt"])
            app.button(key="open_review").click().run()
            self.assertFalse(app.session_state["sync_completion_prompt"])
            sync.assert_called_once()
            groq.assert_not_called()

    def test_review_conflict_lists_product_and_field_without_claiming_export_success(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "save_review_session",
            return_value=ReviewSaveReport(0, None, (("demo-1", "package_weight_g"),), ()),
        ):
            app.session_state["review_manual"] = {"demo-1": {"package_weight_g": 250.0}}
            app.run()
            app.button(key="save_review").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("No changes were saved" in item.value for item in app.info))
            self.assertTrue(any("Synthetic demo-1 [demo-1]: Package weight (g)" in item.value for item in app.warning))
            self.assertFalse(any("Report files were updated" in item.value for item in app.info))
            self.assertEqual(app.session_state["review_manual"], {"demo-1": {"package_weight_g": 250.0}})
            groq.assert_not_called()
            sync.assert_not_called()

    def test_ai_settings_appear_only_inside_review_including_empty_and_initial_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "synthetic.duckdb"
            with patch.dict(os.environ, {
                "LIDL_APP_DB_PATH": str(database),
                "LIDL_APP_ENV_PATH": str(root / ".env"),
                "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
            }), patch.object(pipeline_services, "read_review_products", return_value=[]), patch.object(
                pipeline_services, "run_sync"
            ) as sync, patch.object(pipeline_services, "retry_ai_products") as groq:
                app = AppTest.from_file(str(APP_PATH)).run()
                self.assertNotIn("save_ai_settings", [button.key for button in app.button])
                app.button(key="open_review").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(sum(button.key == "save_ai_settings" for button in app.button), 1)
                database.touch()
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(sum(button.key == "save_ai_settings" for button in app.button), 1)
                app.button(key="close_review").click().run()
                self.assertNotIn("save_ai_settings", [button.key for button in app.button])
                sync.assert_not_called()
                groq.assert_not_called()

    def test_product_editor_callback_tracks_only_changes_and_preserves_null_and_zero(self):
        def callback_probe():
            import streamlit as st
            from app import _capture_product_edits

            st.session_state.product_base = {"salt_100g": 1.0, "fiber_100g": 2.0, "category": "Dairy & Eggs"}
            st.session_state.product_editor = {"edited_rows": {0: {
                "salt_100g": 0.0, "fiber_100g": None, "category": "Dairy & Eggs",
                "sample_name": "Do not edit", "macro_category": "Do not edit",
            }}}
            _capture_product_edits()

        app = AppTest.from_function(callback_probe).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["product_edits"], {"salt_100g": 0.0, "fiber_100g": None})

    def test_product_search_selects_complete_product_and_blocks_switch_while_dirty(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "search_products",
            return_value=[{"product_key": "0001", "sample_name": "Synthetic complete milk"}],
        ) as search, patch.object(
            pipeline_services, "read_product",
            return_value={**products[0], "product_key": "0001", "category": "Non-Food", "macro_category": "Non-Food", "field_metadata": {}},
        ) as load:
            app.text_input(key="product_search_query").set_value("MIL").run()
            self.assertEqual(search.call_args.args[0], "MIL")
            app.selectbox(key="product_search_selection").select("0001").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["product_base"]["product_key"], "0001")
            self.assertEqual(load.call_args.args[0], "0001")
            app.session_state["product_edits"] = {"category": "Dairy & Eggs"}
            app.run()
            self.assertTrue(app.text_input(key="product_search_query").disabled)
            self.assertTrue(app.selectbox(key="product_search_selection").disabled)
            self.assertTrue(app.button(key="sync_pipeline").disabled)
            self.assertTrue(app.button(key="retry_ai").disabled)
            self.assertTrue(app.button(key="save_review").disabled)
            self.assertTrue(app.text_area(key="chat_response_input").disabled)
            self.assertFalse(app.button(key="save_product").disabled)
            app.button(key="discard_product").click().run()
            self.assertEqual(app.session_state["product_edits"], {})
            self.assertFalse(app.selectbox(key="product_search_selection").disabled)
            groq.assert_not_called()
            sync.assert_not_called()

    def test_search_product_save_calls_existing_service_with_only_manual_edits(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "search_products",
            return_value=[{"product_key": "0001", "sample_name": "Synthetic complete milk"}],
        ), patch.object(
            pipeline_services, "read_product", return_value={**products[0], "product_key": "0001", "field_metadata": {}},
        ), patch.object(
            pipeline_services, "save_review_session", return_value=ReviewSaveReport(1, None, (), ()),
        ) as save:
            app.text_input(key="product_search_query").set_value("milk").run()
            app.selectbox(key="product_search_selection").select("0001").run()
            app.session_state["product_edits"] = {"salt_100g": 0.0}
            app.run()
            app.button(key="save_product").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(save.call_args.kwargs["manual_edits"], {"0001": {"salt_100g": 0.0}})
            self.assertEqual(save.call_args.kwargs["baselines"]["0001"]["__field_metadata__"], {})
            self.assertEqual(app.session_state["product_edits"], {})
            self.assertTrue(any("Refresh Power BI" in item.value for item in app.success))
            groq.assert_not_called()
            sync.assert_not_called()

    def test_search_save_conflicts_and_exceptions_preserve_product_draft(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "search_products", return_value=[{"product_key": "0001", "sample_name": "Synthetic complete milk"}],
        ), patch.object(
            pipeline_services, "read_product", return_value={**products[0], "product_key": "0001", "field_metadata": {}},
        ), patch.object(
            pipeline_services, "save_review_session", return_value=ReviewSaveReport(0, None, (("0001", "salt_100g"),), ()),
        ) as save:
            app.text_input(key="product_search_query").set_value("milk").run()
            app.selectbox(key="product_search_selection").select("0001").run()
            draft = {"salt_100g": 0.0}
            app.session_state["product_edits"] = draft
            app.run()
            app.button(key="save_product").click().run()
            self.assertEqual(app.session_state["product_edits"], draft)
            self.assertTrue(any("not saved" in item.value and "0001" in item.value for item in app.warning))
            save.side_effect = OSError("synthetic save failure")
            app.button(key="save_product").click().run()
            app.run()
            self.assertEqual(app.session_state["product_edits"], draft)
            self.assertTrue(any("Product save failed" in item.value for item in app.error))
            groq.assert_not_called()
            sync.assert_not_called()

    def test_search_save_export_failure_retains_export_only_retry(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "search_products", return_value=[{"product_key": "0001", "sample_name": "Synthetic milk"}],
        ), patch.object(
            pipeline_services, "read_product", return_value={**products[0], "product_key": "0001", "field_metadata": {}},
        ), patch.object(
            pipeline_services, "save_review_session", return_value=ReviewSaveReport(1, None, (), ("exports",)),
        ), patch.object(pipeline_services, "refresh_exports") as export:
            app.text_input(key="product_search_query").set_value("milk").run()
            app.selectbox(key="product_search_selection").select("0001").run()
            app.session_state["product_edits"] = {"salt_100g": 0.0}
            app.run()
            app.button(key="save_product").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["export_retry_needed"])
            self.assertEqual(app.session_state["product_edits"], {})
            self.assertTrue(any("could not be updated" in item.value for item in app.warning))
            app.button(key="retry_export").click().run()
            export.assert_called_once()
            sync.assert_not_called()
            groq.assert_not_called()

    def test_groq_defaults_to_all_and_reports_before_after_without_sync(self):
        with synthetic_review_app() as (app, products, read, groq, sync):
            self.assertFalse(app.exception)
            self.assertEqual(app.button(key="retry_ai").label, "Complete with Groq")
            self.assertFalse(app.button(key="retry_ai").disabled)
            app.run()
            groq.assert_not_called()
            self.assertIn("Save AI settings", [button.label for button in app.button])
            read.return_value = [products[1]]
            app.button(key="retry_ai").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(groq.call_args.args[0], ["demo-1", "demo-2"])
            self.assertEqual(app.session_state["groq_summary"]["completed"], 1)
            self.assertEqual(app.session_state["groq_summary"]["remaining"], 1)
            self.assertEqual(len(app.session_state["review_initial_keys"]), 2)
            self.assertEqual(app.button(key="retry_ai").label, "Try Groq again")
            self.assertTrue(any("Refresh Power BI" in item.value for item in app.info))
            app.run()
            groq.assert_called_once()
            sync.assert_not_called()

    def test_groq_optional_subset_requires_selection(self):
        with synthetic_review_app() as (app, products, read, groq, sync):
            app.checkbox(key="groq_subset").check().run()
            self.assertTrue(app.button(key="retry_ai").disabled)
            app.multiselect(key="retry_ai_products").select("demo-2").run()
            app.button(key="retry_ai").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(groq.call_args.args[0], ["demo-2"])
            sync.assert_not_called()

    def test_manual_draft_blocks_groq_and_is_preserved(self):
        with synthetic_review_app() as (app, products, read, groq, sync):
            app.session_state["review_manual"] = {"demo-1": {"salt_100g": 0.1}}
            app.run()
            self.assertTrue(app.button(key="retry_ai").disabled)
            self.assertTrue(app.button(key="sync_pipeline").disabled)
            self.assertEqual(app.session_state["review_manual"], {"demo-1": {"salt_100g": 0.1}})
            groq.assert_not_called()
            sync.assert_not_called()

    def test_save_success_persists_after_all_products_resolved(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "save_review_session",
            return_value=ReviewSaveReport(1, None, (), ()),
        ) as save:
            app.session_state["review_manual"] = {"demo-1": {"salt_100g": 0.1}}
            app.run()
            read.return_value = []
            app.button(key="save_review").click().run(timeout=30)
            self.assertFalse(app.exception)
            save.assert_called_once()
            app.run()
            self.assertTrue(any("Changes saved and report files updated. Refresh Power BI" in item.value for item in app.success))
            groq.assert_not_called()
            sync.assert_not_called()

    def test_groq_partial_and_export_failure_are_not_claimed_as_success(self):
        with synthetic_review_app() as (app, products, read, groq, sync):
            groq.return_value = pipeline_services.AIRetryReport(
                AIEnrichmentReport("partial", retryable_tasks=1, terminal_tasks=2, error_code="http_429"),
                ("exports",),
            )
            app.button(key="retry_ai").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["export_retry_needed"])
            self.assertIn("retry_export", [button.key for button in app.button])
            self.assertEqual(app.session_state["groq_summary"]["remaining"], 2)
            self.assertTrue(any("http_429" in item.value for item in app.warning))
            self.assertTrue(any("could not be updated" in item.value for item in app.warning))
            sync.assert_not_called()

    def test_save_exception_replaces_old_success_and_preserves_draft_on_rerun(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "save_review_session", side_effect=OSError("synthetic save failure")
        ):
            draft = {"demo-1": {"salt_100g": 0.1}}
            app.session_state["review_manual"] = draft
            app.session_state["review_notices"] = [("success", "Old success")]
            app.run()
            app.button(key="save_review").click().run(timeout=30)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["review_manual"], draft)
            self.assertTrue(any("Save failed" in item.value for item in app.error))
            self.assertFalse(any("Old success" in item.value for item in app.success))
            sync.assert_not_called()
            groq.assert_not_called()

    def test_export_exception_persists_and_keeps_export_only_retry_available(self):
        with synthetic_review_app() as (app, products, read, groq, sync), patch.object(
            pipeline_services, "refresh_exports", side_effect=OSError("synthetic export failure")
        ) as export:
            app.session_state["export_retry_needed"] = True
            app.session_state["review_notices"] = [("success", "Old success")]
            app.run()
            app.button(key="retry_export").click().run(timeout=30)
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["export_retry_needed"])
            self.assertTrue(any("Export retry failed" in item.value for item in app.error))
            self.assertFalse(any("Old success" in item.value for item in app.success))
            export.assert_called_once()
            sync.assert_not_called()
            groq.assert_not_called()

    def test_rerun_does_not_start_sync_but_explicit_click_does(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {
                "LIDL_APP_DB_PATH": str(root / "synthetic.duckdb"),
                "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
                "LIDL_APP_ENV_PATH": str(root / ".env"),
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                pipeline_services, "run_sync"
            ) as run_sync:
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                self.assertFalse(app.exception)
                run_sync.assert_not_called()

                app.run(timeout=30)
                run_sync.assert_not_called()

                app.button(key="sync_pipeline").click().run(timeout=30)
                run_sync.assert_called_once()
                self.assertFalse(app.exception)

    def test_review_table_uses_exact_columns_and_keeps_null_distinct_from_zero(self):
        product = {
            "product_key": "000123",
            "sample_name": "Synthetic milk",
            "macro_category": "Fresh Food",
            "category": "Dairy & Eggs",
            "package_weight_g": 0.0,
            "kcal_100g": 0.0,
            "proteins_100g": 3.0,
            "carbs_100g": 5.0,
            "sugars_100g": 4.0,
            "fat_100g": 7.0,
            "saturated_fat_100g": 2.0,
            "fiber_100g": None,
            "salt_100g": None,
        }
        self.assertEqual(_blank(0), 0)
        self.assertIsNone(_blank(None))
        self.assertEqual(TABLE_COLUMNS, (
            "sample_name", "macro_category", "category", "package_weight_g",
            "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
            "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
        ))
        self.assertEqual(len(NUMERIC_COLUMNS), 9)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "synthetic.duckdb"
            db_path.touch()
            env = {
                "LIDL_APP_DB_PATH": str(db_path),
                "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
                "LIDL_APP_ENV_PATH": str(root / ".env"),
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                pipeline_services, "read_review_products", return_value=[product]
            ):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                app.button(key="open_review").click().run(timeout=30)
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("dataframe")), 1)
                frame = app.get("dataframe")[0].value
                self.assertEqual(tuple(frame.columns), TABLE_COLUMNS)
                self.assertEqual(frame.iloc[0]["package_weight_g"], 0.0)
                self.assertTrue(pd.isna(frame.iloc[0]["fiber_100g"]))

    def test_preview_is_for_exact_response_and_required_before_chat_save(self):
        product = {
            "product_key": "000123",
            "sample_name": "Synthetic milk",
            "macro_category": "Fresh Food",
            "category": "Dairy & Eggs",
            "package_weight_g": 0.0,
            "kcal_100g": 0.0,
            "proteins_100g": 3.0,
            "carbs_100g": 5.0,
            "sugars_100g": 4.0,
            "fat_100g": 7.0,
            "saturated_fat_100g": 2.0,
            "fiber_100g": None,
            "salt_100g": None,
        }
        preview = ChatBatchPreview(
            suggestions=(ChatBatchSuggestion("000123", nutrients={"salt_100g": 0.0}),),
            issues=(),
            skipped=(),
        )
        response = '{"products":[{"product_key":"000123","salt_100g":0}]}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "synthetic.duckdb"
            db_path.touch()
            env = {
                "LIDL_APP_DB_PATH": str(db_path),
                "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
                "LIDL_APP_ENV_PATH": str(root / ".env"),
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                pipeline_services, "read_review_products", return_value=[product]
            ), patch.object(
                pipeline_services, "preview_chat_import", return_value=preview
            ) as validate:
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                app.button(key="open_review").click().run(timeout=30)
                app.text_area(key="chat_response_input").set_value(response).run(timeout=30)
                self.assertEqual(app.session_state["chat_response_input"], response)
                app.button(key="validate_chat_response").click().run(timeout=30)
                validate.assert_called_once()
                self.assertEqual(validate.call_args.args[0], response)
                self.assertFalse(app.exception)
                self.assertFalse(app.button(key="save_review").disabled)

                app.text_area(key="chat_response_input").set_value(response + " ").run(timeout=30)
                self.assertTrue(app.button(key="save_review").disabled)

    def test_sync_invalidates_an_already_loaded_review_snapshot(self):
        def product(name: str) -> dict[str, object]:
            return {
                "product_key": "000123",
                "sample_name": name,
                "macro_category": "Fresh Food",
                "category": "Dairy & Eggs",
                "package_weight_g": None,
                "kcal_100g": None,
                "proteins_100g": None,
                "carbs_100g": None,
                "sugars_100g": None,
                "fat_100g": None,
                "saturated_fat_100g": None,
                "fiber_100g": None,
                "salt_100g": None,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "synthetic.duckdb"
            db_path.touch()
            env = {
                "LIDL_APP_DB_PATH": str(db_path),
                "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
                "LIDL_APP_ENV_PATH": str(root / ".env"),
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                pipeline_services,
                "read_review_products",
                side_effect=[[product("Before sync")], [], [product("After sync")]],
            ), patch.object(pipeline_services, "run_sync") as run_sync:
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                app.button(key="open_review").click().run(timeout=30)
                self.assertEqual(
                    app.session_state["review_base"]["000123"]["sample_name"],
                    "Before sync",
                )
                app.button(key="sync_pipeline").click().run(timeout=30)
                run_sync.assert_called_once()
                self.assertEqual(
                    app.session_state["review_base"]["000123"]["sample_name"],
                    "After sync",
                )

    def test_export_retry_is_available_after_saved_export_failure_and_is_export_only(self):
        product = {
            "product_key": "000123",
            "sample_name": "Synthetic milk",
            "macro_category": "Fresh Food",
            "category": "Dairy & Eggs",
            "package_weight_g": 0.0,
            "kcal_100g": 0.0,
            "proteins_100g": 3.0,
            "carbs_100g": 5.0,
            "sugars_100g": 4.0,
            "fat_100g": 7.0,
            "saturated_fat_100g": 2.0,
            "fiber_100g": None,
            "salt_100g": None,
        }
        preview = ChatBatchPreview(
            suggestions=(ChatBatchSuggestion("000123", nutrients={"salt_100g": 0.0}),),
            issues=(),
            skipped=(),
        )
        saved_preview = ChatBatchPreview((), (), ())
        save_report = ReviewSaveReport(
            manual_fields_saved=0,
            chat_import=ChatBatchSaveReport(1, 1, saved_preview),
            conflicts=(),
            artifact_failures=("exports",),
        )
        response = '{"products":[{"product_key":"000123","salt_100g":0}]}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "synthetic.duckdb"
            db_path.touch()
            env = {
                "LIDL_APP_DB_PATH": str(db_path),
                "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
                "LIDL_APP_ENV_PATH": str(root / ".env"),
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                pipeline_services, "read_review_products", return_value=[product]
            ), patch.object(
                pipeline_services, "preview_chat_import", return_value=preview
            ), patch.object(
                pipeline_services, "save_review_session", return_value=save_report
            ), patch.object(
                pipeline_services, "refresh_exports"
            ) as refresh_exports, patch.object(pipeline_services, "run_sync") as run_sync:
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                app.button(key="open_review").click().run(timeout=30)
                app.text_area(key="chat_response_input").set_value(response).run(timeout=30)
                app.button(key="validate_chat_response").click().run(timeout=30)
                app.button(key="save_review").click().run(timeout=30)
                self.assertFalse(app.exception)
                app.button(key="retry_export").click().run(timeout=30)
                refresh_exports.assert_called_once()
                run_sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
