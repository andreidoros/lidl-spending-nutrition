"""Synthetic Streamlit tests; every pipeline operation is mocked."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import app
import pipeline_services
from ai_enrichment.chat_batch import ChatBatchPreview, ChatBatchSaveReport, ChatBatchSuggestion
from ai_enrichment.service import AIEnrichmentReport
from pipeline_services import ReviewSaveReport


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
SAVED = ReviewSaveReport(1, None, (), ())


def synthetic_products():
    return [
        {"product_key": key, "sample_name": f"Synthetic {key}", "cleared_reason": None,
         "field_metadata": {}, **{field: None for field in app.TABLE_COLUMNS[1:]}}
        for key in ("demo-1", "demo-2")
    ]


def weight_products():
    return [
        {"product_key": "50980", "sample_name": "Rulou cu mere", "category": "Desserts & Pastries",
         "package_weight_g": None, "field_metadata": {}, "cleared_reason": None,
         "sold_as": "per piece", "times_bought": 53, "units": 60.0, "spend": 190.0},
    ]


@contextmanager
def synthetic_app(products=None, *, database=True, ai_configured=True, weights=None):
    """Run app.py against mocked services and a throwaway workspace."""
    products = synthetic_products() if products is None else products
    weights = [] if weights is None else weights
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        db_path = root / "synthetic.duckdb"
        if database:
            db_path.touch()
        environment = {
            "LIDL_APP_DB_PATH": str(db_path),
            "LIDL_APP_CHAT_BATCH_PATH": str(root / "chat.txt"),
            "LIDL_APP_ENV_PATH": str(root / ".env"),
        }
        if ai_configured:
            environment.update({
                "AI_ENRICHMENT_MODE": "categories_and_nutrition",
                "GROQ_API_KEY": "synthetic-key-not-a-real-credential",
            })
        stack.enter_context(patch.dict(os.environ, environment))

        def read_products(keys=None, db_path=None):
            return [dict(p) for p in products if keys is None or p["product_key"] in keys]

        mocks = {
            "read_review_products": stack.enter_context(patch.object(
                pipeline_services, "read_review_products",
                side_effect=lambda *_: [dict(p) for p in products])),
            "read_products": stack.enter_context(patch.object(
                pipeline_services, "read_products", side_effect=read_products)),
            "save_review_session": stack.enter_context(patch.object(
                pipeline_services, "save_review_session", return_value=SAVED)),
            "refresh_review_artifacts": stack.enter_context(patch.object(
                pipeline_services, "refresh_review_artifacts", return_value=())),
            "retry_ai_products": stack.enter_context(patch.object(
                pipeline_services, "retry_ai_products",
                return_value=pipeline_services.AIRetryReport(AIEnrichmentReport("completed"), ()))),
            "run_sync": stack.enter_context(patch.object(pipeline_services, "run_sync")),
            "read_weight_worklist": stack.enter_context(patch.object(
                pipeline_services, "read_weight_worklist",
                side_effect=lambda keys=None, db_path=None: [
                    dict(p) for p in weights if keys is None or p["product_key"] in keys])),
            "weight_batch_text": stack.enter_context(patch.object(
                pipeline_services, "weight_batch_text", return_value="Estimate the weight...")),
        }
        test = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        yield test, mocks


class FakeSessionState(dict):
    """Attribute and item access, like st.session_state, for callback tests."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@contextmanager
def callback_state(products=None):
    """A session state holding one rendered "To complete" table."""
    products = synthetic_products() if products is None else products
    state = FakeSessionState()
    with patch.object(app.st, "session_state", state), patch.object(app.st, "toast"):
        app._init_state()
        shown = {}
        for product in products:
            values = {field: product.get(field) for field in app.TABLE_COLUMNS}
            values["__field_metadata__"] = {}
            shown[product["product_key"]] = values
        state.editor_context["todo_table_0_x"] = {
            "view": "todo", "keys": [p["product_key"] for p in products], "shown": shown,
        }
        yield state


class CellChangeTests(unittest.TestCase):
    shown = {"a": {"package_weight_g": 250.0, "category": None}, "b": {"package_weight_g": None}}

    def test_new_edit_uses_the_stored_value_as_before(self):
        changes = app.cell_changes({0: {"package_weight_g": 300}}, {}, ["a", "b"], self.shown)
        self.assertEqual(changes, [("a", "package_weight_g", 250.0, 300.0)])

    def test_an_edit_already_handled_is_not_saved_again(self):
        edited = {0: {"package_weight_g": 300}}
        self.assertEqual(app.cell_changes(edited, edited, ["a", "b"], self.shown), [])

    def test_a_second_edit_of_the_same_cell_starts_from_the_first_edit(self):
        changes = app.cell_changes(
            {0: {"package_weight_g": 320}}, {0: {"package_weight_g": 300}}, ["a", "b"], self.shown
        )
        self.assertEqual(changes, [("a", "package_weight_g", 300.0, 320.0)])

    def test_blank_means_unknown_and_zero_stays_zero(self):
        changes = app.cell_changes(
            {0: {"package_weight_g": None, "category": ""}, 1: {"package_weight_g": 0}},
            {}, ["a", "b"], self.shown,
        )
        self.assertEqual(changes, [
            ("a", "package_weight_g", 250.0, None),
            ("b", "package_weight_g", None, 0.0),
        ])

    def test_read_only_columns_are_ignored(self):
        changes = app.cell_changes(
            {0: {"sample_name": "Renamed", "macro_category": "Fresh Food"}}, {}, ["a"], self.shown
        )
        self.assertEqual(changes, [])


class TableCallbackTests(unittest.TestCase):
    def _edit(self, state, edited):
        state["todo_table_0_x"] = {"edited_rows": edited}
        app._on_table_edit("todo_table_0_x")

    def test_a_cell_is_saved_once_as_soon_as_it_is_left(self):
        with callback_state() as state, patch.object(
            pipeline_services, "save_review_session", return_value=SAVED
        ) as save:
            self._edit(state, {0: {"package_weight_g": 400}})
            # The editor reports the same edit again on a rerun: nothing new.
            self._edit(state, {0: {"package_weight_g": 400}})
        save.assert_called_once()
        kwargs = save.call_args.kwargs
        self.assertEqual(kwargs["manual_edits"], {"demo-1": {"package_weight_g": 400.0}})
        self.assertIsNone(kwargs["baselines"]["demo-1"]["package_weight_g"])
        self.assertFalse(kwargs["refresh_artifacts"])
        self.assertEqual(state.notice, ("success", "Saved: Synthetic demo-1 · Package weight 400 g"))
        self.assertTrue(state.artifacts_pending)
        self.assertEqual(len(state.undo_stack), 1)
        self.assertEqual(state.editor_versions, {})

    def test_a_value_changed_elsewhere_resets_the_table_and_says_so(self):
        conflict = ReviewSaveReport(0, None, (("demo-1", "salt_100g"),), ())
        with callback_state() as state, patch.object(
            pipeline_services, "save_review_session", return_value=conflict
        ):
            self._edit(state, {0: {"salt_100g": 1.2}})
        level, text = state.notice
        self.assertEqual(level, "warning")
        self.assertIn("Salt / 100 g changed somewhere else", text)
        self.assertEqual(state.editor_versions, {"todo": 1})
        self.assertEqual(state.undo_stack, [])

    def test_a_busy_pipeline_does_not_lose_the_edit_silently(self):
        with callback_state() as state, patch.object(
            pipeline_services, "save_review_session",
            side_effect=pipeline_services.PipelineBusyError("busy"),
        ):
            self._edit(state, {1: {"kcal_100g": 90}})
        self.assertEqual(state.notice[0], "warning")
        self.assertIn("Not saved", state.notice[1])
        self.assertEqual(state.editor_versions, {"todo": 1})

    def test_non_food_waits_for_confirmation(self):
        with callback_state() as state, patch.object(
            pipeline_services, "save_review_session", return_value=SAVED
        ) as save:
            self._edit(state, {0: {"category": "Non-Food"}})
            save.assert_not_called()
            self.assertEqual(state.pending_non_food["key"], "demo-1")
            app._cancel_non_food()
        self.assertIsNone(state.pending_non_food)
        self.assertEqual(state.editor_versions, {"todo": 1})

    def test_undo_of_non_food_restores_category_and_nutrition_together(self):
        before = {"category": "Dairy & Eggs", "kcal_100g": 60.0, "salt_100g": 0.1}
        with callback_state() as state, patch.object(
            pipeline_services, "save_review_session", return_value=SAVED
        ) as save:
            app._apply_edit("todo", "demo-1", "Milk", {"category": "Non-Food"}, before,
                            {"category": "Dairy & Eggs", "__field_metadata__": {}})
            app._undo_last()
        undo = save.call_args.kwargs
        self.assertEqual(undo["manual_edits"], {"demo-1": before})
        # The emptied nutrition must be part of the expected state, or the
        # save would report it as changed and restore only the category.
        self.assertEqual(
            {field: undo["baselines"]["demo-1"][field] for field in before},
            {"category": "Non-Food", "kcal_100g": None, "salt_100g": None},
        )

    def test_undo_restores_the_previous_value(self):
        with callback_state() as state, patch.object(
            pipeline_services, "save_review_session", return_value=SAVED
        ) as save:
            self._edit(state, {0: {"package_weight_g": 400}})
            app._undo_last()
        undo = save.call_args.kwargs
        self.assertEqual(undo["manual_edits"], {"demo-1": {"package_weight_g": None}})
        self.assertEqual(undo["baselines"]["demo-1"]["package_weight_g"], 400.0)
        self.assertEqual(state.undo_stack, [])
        self.assertEqual(state.notice[0], "info")
        self.assertEqual(state.editor_versions, {"todo": 1})


class StreamlitAppTests(unittest.TestCase):
    def test_layout_has_three_views_and_no_save_buttons(self):
        with synthetic_app() as (test, mocks):
            self.assertFalse(test.exception)
            self.assertEqual(test.title[0].value, "Lidl Spending & Nutrition")
            labels = [button.label for button in test.button]
            for label in ("Sync", "Complete with Groq", "Use your chatbot"):
                self.assertIn(label, labels)
            for label in ("Save changes", "Discard changes", "Review products"):
                self.assertNotIn(label, labels)
            captions = [caption.value for caption in test.caption]
            self.assertEqual(captions[-1], app.APP_FOOTER)
            self.assertIn(app.PRIVACY_NOTE, captions)
            self.assertTrue(any("Edits save automatically." in item.value for item in test.markdown))
            mocks["run_sync"].assert_not_called()
            mocks["retry_ai_products"].assert_not_called()

    def test_settings_live_only_in_the_settings_view(self):
        with synthetic_app() as (test, mocks):
            self.assertNotIn("setting_groq_key", [widget.key for widget in test.text_input])
            test.session_state["view"] = "settings"
            test.run()
            self.assertFalse(test.exception)
            self.assertIn("setting_groq_key", [widget.key for widget in test.text_input])
            self.assertIn("Save settings", [button.label for button in test.button])

    def test_sync_runs_once_per_click_and_reports_the_worklist(self):
        with synthetic_app() as (test, mocks):
            test.run()
            mocks["run_sync"].assert_not_called()
            test.button(key="sync_pipeline").click().run()
            self.assertFalse(test.exception)
            mocks["run_sync"].assert_called_once()
            test.run()
            mocks["run_sync"].assert_called_once()
            self.assertEqual(test.session_state["notice"], ("success", "Sync complete. 2 products to complete."))

    def test_failed_sync_is_reported_without_retrying(self):
        with synthetic_app() as (test, mocks):
            mocks["run_sync"].side_effect = RuntimeError("receipt outage")
            test.button(key="sync_pipeline").click().run()
            level, text = test.session_state["notice"]
            self.assertEqual(level, "error")
            self.assertIn("Sync failed (RuntimeError)", text)
            test.run()
            mocks["run_sync"].assert_called_once()

    def test_groq_completes_all_incomplete_products_from_its_dialog(self):
        with synthetic_app() as (test, mocks):
            test.button(key="open_groq").click().run()
            self.assertFalse(test.exception)
            test.button(key="start_groq").click().run()
            self.assertFalse(test.exception)
            mocks["retry_ai_products"].assert_called_once()
            self.assertEqual(mocks["retry_ai_products"].call_args.args[0], ["demo-1", "demo-2"])
            level, text = test.session_state["notice"]
            self.assertIn("Groq completed 0 of 2 products.", text)
            self.assertEqual(level, "warning")

    def test_groq_without_a_key_points_to_settings(self):
        with synthetic_app(ai_configured=False) as (test, mocks):
            test.button(key="open_groq").click().run()
            self.assertIn("Open Settings", [button.label for button in test.button])
            self.assertNotIn("start_groq", [button.key for button in test.button])
            mocks["retry_ai_products"].assert_not_called()

    def test_chatbot_answer_is_checked_before_it_can_be_saved(self):
        preview = ChatBatchPreview(
            (ChatBatchSuggestion("demo-1", nutrients={"salt_100g": 0.5}),), (), ()
        )
        saved = ReviewSaveReport(0, ChatBatchSaveReport(1, 1, preview), (), ())
        with synthetic_app() as (test, mocks), patch.object(
            pipeline_services, "refresh_chat_batch",
            side_effect=lambda db, out: Path(out).write_text("Help complete...", encoding="utf-8"),
        ), patch.object(pipeline_services, "preview_chat_import", return_value=preview) as check:
            mocks["save_review_session"].return_value = saved
            test.button(key="open_chatbot").click().run()
            self.assertFalse(test.exception)
            self.assertNotIn("save_chat_answer", [button.key for button in test.button])
            test.text_area(key="chat_answer").input('{"products": []}').run()
            test.button(key="check_chat_answer").click().run()
            check.assert_called_once()
            self.assertIn("Save 1 suggested value", [button.label for button in test.button])
            test.button(key="save_chat_answer").click().run()
            self.assertFalse(test.exception)
            kwargs = mocks["save_review_session"].call_args.kwargs
            self.assertEqual(kwargs["response_text"], '{"products": []}')
            self.assertEqual(kwargs["manual_edits"], {})
            self.assertEqual(test.session_state["notice"], ("success", "Saved 1 value from your chatbot."))

    def test_report_files_follow_after_a_pause_and_offer_retry_on_failure(self):
        with synthetic_app() as (test, mocks):
            test.session_state["artifacts_pending"] = True
            test.session_state["last_edit_at"] = 0.0
            mocks["refresh_review_artifacts"].return_value = ("exports",)
            test.run()
            mocks["refresh_review_artifacts"].assert_called_once()
            self.assertFalse(test.session_state["artifacts_pending"])
            self.assertTrue(test.session_state["export_retry_needed"])
            self.assertIn("retry_export", [button.key for button in test.button])

    def test_missing_weight_view_lists_products_and_counts_them(self):
        with synthetic_app(weights=weight_products()) as (test, mocks):
            labels = test.get("button_group")[0].proto.options if test.get("button_group") else []
            self.assertTrue(any("Missing weight (1)" in option.content for option in labels))
            test.session_state["view"] = "weights"
            test.run()
            self.assertFalse(test.exception)
            self.assertTrue(any("Type the weight in grams" in item.value for item in test.markdown))
            self.assertFalse(test.button(key="open_weight_chatbot").disabled)

    def test_weight_chatbot_asks_only_for_weights(self):
        with synthetic_app(weights=weight_products()) as (test, mocks):
            test.session_state["view"] = "weights"
            test.run()
            test.button(key="open_weight_chatbot").click().run()
            self.assertFalse(test.exception)
            mocks["weight_batch_text"].assert_called_once()
            self.assertEqual(mocks["weight_batch_text"].call_args.args[0], ["50980"])

    def test_first_run_without_database_explains_sync(self):
        with synthetic_app(database=False) as (test, mocks):
            self.assertFalse(test.exception)
            self.assertTrue(any("First run" in item.value for item in test.info))
            self.assertFalse(test.button(key="sync_pipeline").disabled)
            mocks["read_review_products"].assert_not_called()

    def test_nothing_to_complete(self):
        with synthetic_app(products=[]) as (test, mocks):
            self.assertFalse(test.exception)
            self.assertTrue(any("Nothing to complete" in item.value for item in test.success))
            self.assertTrue(test.button(key="open_groq").disabled)


if __name__ == "__main__":
    unittest.main()
