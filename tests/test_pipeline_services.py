"""Temporary-resource tests for reusable pipeline service entry points."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline_services
import sync_incremental as sync
from product_store import create_schema


class PipelineServicesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "test.duckdb"
        lock_patcher = patch.object(
            pipeline_services,
            "_workspace_lock_path",
            return_value=self.root / ".pipeline-test.lock",
        )
        lock_patcher.start()
        self.addCleanup(lock_patcher.stop)

    def tearDown(self):
        self.temp.cleanup()

    def test_importing_pipeline_modules_creates_no_files_or_runs_sync(self):
        with tempfile.TemporaryDirectory() as cwd:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import pipeline_services; import sync_incremental",
                ],
                cwd=cwd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse((Path(cwd) / "data").exists())
            self.assertFalse((Path(cwd) / "lidl_consumption.duckdb").exists())

    def test_run_sync_preserves_stage_order_and_forwards_real_progress(self):
        calls: list[str] = []
        events: list[pipeline_services.ProgressEvent] = []

        class FakeLidl:
            refresh_token = "test-token"
            detected_country = None

            def __init__(self, **_kwargs):
                pass

        def fake_receipt_sync(_lidl, progress_callback=None):
            calls.append("receipt_sync")
            progress_callback({
                "stage": "receipt_sync",
                "status": "progress",
                "message": "Receipt downloaded",
                "completed": 1,
            })
            return sync.ReceiptSyncResult(downloaded=1, failed_pages=1, failed_receipts=0)

        def mark(name):
            def operation(*_args, **_kwargs):
                calls.append(name)
            return operation

        def fake_off(_connection, progress_callback=None):
            calls.append("off_enrichment")
            progress_callback({
                "stage": "off_enrichment",
                "status": "progress",
                "message": "Open Food Facts lookup complete",
                "completed": 1,
                "total": 1,
            })

        def fake_ai(_connection, progress_callback=None):
            calls.append("ai_enrichment")
            if progress_callback:
                progress_callback({
                    "stage": "ai_enrichment",
                    "status": "progress",
                    "message": "AI request batch saved",
                    "completed": 1,
                    "total": 1,
                })

        def fake_seed(_connection, *, quiet):
            calls.append("seed_refresh" if quiet else "seed_catalog")

        def fake_overrides(_connection, *, quiet):
            calls.append("overrides_initial" if quiet else "overrides_final")

        report = SimpleNamespace(
            already_applied=True,
            products_scanned=0,
            metadata_rows_inserted=0,
            mismatched_overrides=[],
        )
        import product_store
        import ai_enrichment

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"LIDL_COUNTRY": "ro"}))
            stack.enter_context(patch.object(sync, "load_lidl_token", return_value="test-token"))
            stack.enter_context(patch.object(sync, "AutoDetectLidlPlusApi", FakeLidl))
            stack.enter_context(patch.object(sync, "configure_country"))
            stack.enter_context(patch.object(sync, "sync_tickets_incremental", side_effect=fake_receipt_sync))

            # Keep database setup real, but replace every business stage with a
            # recorder so this test cannot call external services or touch data.
            stack.enter_context(patch.object(sync, "apply_seed_catalog_to_db", side_effect=fake_seed))
            stack.enter_context(patch.object(sync, "apply_nutrition_overrides", side_effect=fake_overrides))
            stack.enter_context(patch.object(sync, "process_unprocessed_tickets_to_db", side_effect=mark("receipt_import")))
            stack.enter_context(patch.object(sync, "update_nutrition_cache_from_off", side_effect=fake_off))
            stack.enter_context(patch.object(ai_enrichment, "run_ai_enrichment", side_effect=fake_ai))
            stack.enter_context(patch.object(sync, "run_deterministic_non_food_filter", side_effect=mark("non_food")))
            stack.enter_context(patch.object(sync, "compute_weights_and_quantities", side_effect=mark("weights")))
            stack.enter_context(patch.object(sync, "validate_atwater", side_effect=mark("atwater")))
            stack.enter_context(patch.object(sync, "warn_suspicious_overrides", side_effect=mark("warnings")))
            stack.enter_context(patch.object(sync, "update_review_queue", side_effect=mark("review")))
            stack.enter_context(patch.object(sync, "export_views_for_bi", side_effect=mark("exports")))
            stack.enter_context(patch.object(
                pipeline_services,
                "refresh_chat_batch_file",
                side_effect=lambda *_args, **_kwargs: calls.append("chat_batch") or 0,
            ))
            stack.enter_context(patch.object(product_store, "apply_migration", return_value=report))
            stack.enter_context(patch.object(product_store, "synchronize_field_metadata", return_value=0))

            pipeline_services.run_sync(self.db_path, progress_callback=events.append)

        self.assertEqual(
            calls,
            [
                "receipt_sync", "seed_catalog", "overrides_initial", "receipt_import",
                "seed_refresh", "off_enrichment", "non_food", "weights", "ai_enrichment", "atwater",
                "overrides_final", "warnings", "review", "chat_batch", "exports",
            ],
        )
        receipt_event = next(event for event in events if event.stage == "receipt_sync" and event.status == "progress")
        self.assertEqual(receipt_event.completed, 1)
        off_event = next(event for event in events if event.stage == "off_enrichment" and event.status == "progress")
        self.assertEqual((off_event.completed, off_event.total), (1, 1))
        ai_event = next(event for event in events if event.stage == "ai_enrichment" and event.status == "progress")
        self.assertEqual((ai_event.completed, ai_event.total), (1, 1))
        receipt_stage = next(event for event in events if event.stage == "receipt_sync" and event.status == "partial")
        self.assertIn("continuing with saved receipts", receipt_stage.message)
        self.assertEqual(events[-1].status, "partial")

    def test_receipt_sync_reports_actual_download_count(self):
        class FakeLidl:
            def receipts(self, pageNumber, only_favorite):
                if pageNumber == 1:
                    return {"tickets": [{"id": "ticket-1"}], "size": 10}
                return {"tickets": [], "size": 10}

            def receipt(self, ticket_id):
                return {"id": ticket_id, "itemsLine": []}

        events = []
        with patch.object(sync, "RAW_DIR", self.root / "raw"), patch.object(sync.time, "sleep"):
            result = sync.sync_tickets_incremental(FakeLidl(), progress_callback=events.append)

        self.assertTrue((self.root / "raw" / "ticket-1.json").is_file())
        self.assertEqual(events[0]["completed"], 1)
        self.assertEqual(events[0]["stage"], "receipt_sync")
        self.assertEqual((result.downloaded, result.failed_pages, result.failed_receipts), (1, 0, 0))

    def test_receipt_page_failure_is_returned_as_partial_result(self):
        class FakeLidl:
            def receipts(self, **_kwargs):
                raise ConnectionError("test outage")

        events = []
        with patch.object(sync, "RAW_DIR", self.root / "raw"):
            result = sync.sync_tickets_incremental(FakeLidl(), progress_callback=events.append)

        self.assertEqual((result.downloaded, result.failed_pages, result.failed_receipts), (0, 1, 0))
        self.assertEqual(events[0]["message"], "Receipt page download failed")

    def test_individual_receipt_failure_is_counted_without_stopping_stage(self):
        class FakeLidl:
            def receipts(self, **_kwargs):
                return {"tickets": [{"id": "bad"}, {"id": "good"}], "size": 10}

            def receipt(self, ticket_id):
                if ticket_id == "bad":
                    raise ConnectionError("test receipt outage")
                return {"id": ticket_id, "itemsLine": []}

        events = []
        with patch.object(sync, "RAW_DIR", self.root / "raw"), patch.object(sync.time, "sleep"):
            result = sync.sync_tickets_incremental(FakeLidl(), progress_callback=events.append)

        self.assertEqual((result.downloaded, result.failed_pages, result.failed_receipts), (1, 0, 1))
        self.assertTrue((self.root / "raw" / "good.json").is_file())
        self.assertFalse((self.root / "raw" / "bad.json").exists())
        self.assertTrue(any(event["message"] == "Receipt download failed" for event in events))

    def test_off_lookup_reports_completed_product_count_without_network(self):
        connection = duckdb.connect(":memory:")
        try:
            sync.init_database(connection)
            connection.execute(
                """INSERT INTO purchases
                   (ticket_id, date, line_idx, name, quantity, unit_price,
                    line_total, raw_code, barcode_type, ean)
                   VALUES ('ticket-1', CURRENT_TIMESTAMP, 1, 'Test item', 1,
                           1, 1, '5941234567890', 'GLOBAL_EAN', '5941234567890')"""
            )
            events = []
            with patch.object(sync, "OFF_MODE", "api"), \
                 patch.object(sync, "OFF_NOT_FOUND_TTL_DAYS", 0), \
                 patch.object(sync, "fetch_off_product_api", return_value={"not_found": True}), \
                 patch.object(sync.time, "sleep"):
                sync.update_nutrition_cache_from_off(connection, progress_callback=events.append)
        finally:
            connection.close()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["stage"], "off_enrichment")
        self.assertEqual((events[0]["completed"], events[0]["total"]), (1, 1))

    def test_export_only_does_not_authenticate_or_fetch_receipts(self):
        connection = duckdb.connect(str(self.db_path))
        connection.close()
        events = []
        with patch.object(sync, "load_lidl_token") as load_token, \
             patch.object(sync, "AutoDetectLidlPlusApi") as api_factory, \
             patch.object(sync, "sync_tickets_incremental") as receipt_sync, \
             patch.object(sync, "export_views_for_bi") as export:
            pipeline_services.refresh_exports(self.db_path, events.append)

        export.assert_called_once()
        load_token.assert_not_called()
        api_factory.assert_not_called()
        receipt_sync.assert_not_called()
        self.assertEqual([event.status for event in events], ["started", "completed"])

    def test_review_read_and_write_services_use_database_and_regenerate_worklist(self):
        connection = duckdb.connect(str(self.db_path))
        try:
            sync.init_database(connection)
            connection.execute(
                """INSERT INTO nutrition_cache
                   (product_key, sample_name, macro_category, category, kcal_100g,
                    proteins_100g, carbs_100g, sugars_100g, fat_100g,
                    saturated_fat_100g, salt_100g, match_status)
                   VALUES ('item-1', 'Yogurt', 'Fresh Food', 'Dairy & Eggs',
                           60, 3, 5, 5, 2, 1, NULL, 'TEST')"""
            )
        finally:
            connection.close()

        queue_path = self.root / "review.csv"
        original_queue = sync.REVIEW_QUEUE_PATH
        try:
            sync.REVIEW_QUEUE_PATH = queue_path
            with patch.object(sync, "load_taxonomy", return_value=(
                {"Fresh Food": ["Dairy & Eggs"]}, {"Dairy & Eggs": "Fresh Food"}
            )):
                before = pipeline_services.read_review_products(self.db_path)
                self.assertEqual([row["product_key"] for row in before], ["item-1"])

                changed = pipeline_services.save_review_fields(
                    "item-1", {"salt_100g": 0.0}, self.db_path
                )
                after = pipeline_services.read_review_products(self.db_path)
        finally:
            sync.REVIEW_QUEUE_PATH = original_queue

        self.assertEqual(changed, ("salt_100g",))
        self.assertEqual(after, [])
        self.assertTrue(queue_path.is_file())

    def test_clearing_review_field_removes_stale_override_before_next_sync(self):
        import product_store

        connection = duckdb.connect(str(self.db_path))
        try:
            sync.init_database(connection)
            connection.execute(
                """INSERT INTO nutrition_cache
                   (product_key, sample_name, macro_category, category, kcal_100g,
                    proteins_100g, carbs_100g, sugars_100g, fat_100g,
                    saturated_fat_100g, salt_100g, match_status)
                   VALUES ('item-1', 'Yogurt', 'Fresh Food', 'Dairy & Eggs',
                           60, 3, 5, 5, 2, 1, 0.5, 'TEST')"""
            )
            product_store.create_schema(connection)
            connection.execute(
                """INSERT INTO product_field_metadata
                   (product_key, field_name, source, is_protected)
                   VALUES ('item-1', 'salt_100g', 'MANUAL', TRUE)"""
            )
        finally:
            connection.close()

        override_path = self.root / "catalog_overrides.csv"
        queue_path = self.root / "review.csv"
        sync.write_csv_records(
            override_path,
            [
                {"product_key": "item-1", "proteins_100g": "3", "salt_100g": "0.5"},
                {"product_key": "item-2", "salt_100g": "0.2"},
            ],
            sync.OVERRIDE_COLUMNS,
        )
        original_overrides, original_queue = sync.OVERRIDE_CSV_PATH, sync.REVIEW_QUEUE_PATH
        try:
            sync.OVERRIDE_CSV_PATH = override_path
            sync.REVIEW_QUEUE_PATH = queue_path
            with patch.object(sync, "load_taxonomy", return_value=(
                {"Fresh Food": ["Dairy & Eggs"]}, {"Dairy & Eggs": "Fresh Food"}
            )):
                connection = duckdb.connect(str(self.db_path))
                try:
                    sync.apply_nutrition_overrides(connection, quiet=True)
                finally:
                    connection.close()

                pipeline_services.save_review_fields(
                    "item-1", {"salt_100g": None}, self.db_path
                )

            rows, _, _ = sync.read_csv_records(override_path)
            rows_by_key = {row["product_key"]: row for row in rows}
            self.assertEqual(rows_by_key["item-1"]["salt_100g"], "")
            self.assertEqual(rows_by_key["item-1"]["proteins_100g"], "3")
            self.assertEqual(rows_by_key["item-2"]["salt_100g"], "0.2")

            connection = duckdb.connect(str(self.db_path))
            try:
                sync.apply_nutrition_overrides(connection, quiet=True)
                self.assertIsNone(connection.execute(
                    "SELECT salt_100g FROM nutrition_cache WHERE product_key='item-1'"
                ).fetchone()[0])
                self.assertIsNone(connection.execute(
                    "SELECT salt_100g FROM catalog_overrides WHERE product_key='item-1'"
                ).fetchone()[0])
            finally:
                connection.close()
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides
            sync.REVIEW_QUEUE_PATH = original_queue

    def test_clearing_last_override_row_keeps_mirror_out_of_prefix_sources(self):
        connection = duckdb.connect(str(self.db_path))
        try:
            sync.init_database(connection)
            connection.execute(
                """INSERT INTO nutrition_cache
                   (product_key, sample_name, macro_category, category, kcal_100g,
                    proteins_100g, carbs_100g, sugars_100g, fat_100g,
                    saturated_fat_100g, salt_100g, match_status)
                   VALUES ('2501234004508', 'Product', 'Fresh Food', 'Dairy & Eggs',
                           100, 3, 5, 1, 2, 1, 0.5, 'TEST')"""
            )
        finally:
            connection.close()

        override_path = self.root / "catalog_overrides.csv"
        queue_path = self.root / "review.csv"
        sync.write_csv_records(
            override_path,
            [{
                "product_key": "2501234004508",
                "macro_category": "Fresh Food",
                "category": "Dairy & Eggs",
                "kcal_100g": "100",
                "salt_100g": "0.5",
            }],
            sync.OVERRIDE_COLUMNS,
        )
        original_overrides, original_queue = sync.OVERRIDE_CSV_PATH, sync.REVIEW_QUEUE_PATH
        try:
            sync.OVERRIDE_CSV_PATH = override_path
            sync.REVIEW_QUEUE_PATH = queue_path
            with patch.object(sync, "load_taxonomy", return_value=(
                {"Fresh Food": ["Dairy & Eggs"]}, {"Dairy & Eggs": "Fresh Food"}
            )):
                connection = duckdb.connect(str(self.db_path))
                try:
                    sync.apply_nutrition_overrides(connection, quiet=True)
                finally:
                    connection.close()

                pipeline_services.save_review_fields(
                    "2501234004508",
                    {"category": None, "kcal_100g": None, "salt_100g": None},
                    self.db_path,
                )

                rows, _, _ = sync.read_csv_records(override_path)
                self.assertEqual(rows, [])

                connection = duckdb.connect(str(self.db_path))
                try:
                    # Empty CSVs are intentionally not allowed to truncate the
                    # legacy mirror; the explicit edit already cleared its cells.
                    sync.apply_nutrition_overrides(connection, quiet=True)
                    mirror = connection.execute(
                        """SELECT macro_category, category, kcal_100g, salt_100g
                           FROM catalog_overrides WHERE product_key='2501234004508'"""
                    ).fetchone()
                    self.assertEqual(mirror, (None, None, None, None))
                    prefix_sources = connection.execute(
                        "SELECT COUNT(*) FROM catalog_overrides WHERE macro_category IS NOT NULL"
                    ).fetchone()[0]
                    self.assertEqual(prefix_sources, 0)
                finally:
                    connection.close()
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides
            sync.REVIEW_QUEUE_PATH = original_queue

    def test_csv_snapshot_is_restored_if_database_save_fails(self):
        import product_store

        connection = duckdb.connect(str(self.db_path))
        try:
            sync.init_database(connection)
            connection.execute(
                """INSERT INTO nutrition_cache
                   (product_key, sample_name, macro_category, category, salt_100g, match_status)
                   VALUES ('item-1', 'Yogurt', 'Fresh Food', 'Dairy & Eggs', 0.5, 'TEST')"""
            )
            product_store.create_schema(connection)
            connection.execute(
                """INSERT INTO product_field_metadata
                   (product_key, field_name, source, is_protected)
                   VALUES ('item-1', 'salt_100g', 'MANUAL', TRUE)"""
            )
        finally:
            connection.close()

        override_path = self.root / "catalog_overrides.csv"
        queue_path = self.root / "review.csv"
        sync.write_csv_records(
            override_path,
            [{"product_key": "item-1", "salt_100g": "0.5"}],
            sync.OVERRIDE_COLUMNS,
        )
        original_bytes = override_path.read_bytes()
        original_overrides, original_queue = sync.OVERRIDE_CSV_PATH, sync.REVIEW_QUEUE_PATH
        try:
            sync.OVERRIDE_CSV_PATH = override_path
            sync.REVIEW_QUEUE_PATH = queue_path
            with patch.object(sync, "load_taxonomy", return_value=(
                {"Fresh Food": ["Dairy & Eggs"]}, {"Dairy & Eggs": "Fresh Food"}
            )), patch.object(
                pipeline_services,
                "save_manual_fields",
                side_effect=RuntimeError("simulated database failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated database failure"):
                    pipeline_services.save_review_fields(
                        "item-1", {"salt_100g": None}, self.db_path
                    )

            self.assertEqual(override_path.read_bytes(), original_bytes)
            connection = duckdb.connect(str(self.db_path))
            try:
                self.assertEqual(connection.execute(
                    "SELECT salt_100g FROM nutrition_cache WHERE product_key='item-1'"
                ).fetchone()[0], 0.5)
                self.assertEqual(connection.execute(
                    "SELECT is_protected FROM product_field_metadata "
                    "WHERE product_key='item-1' AND field_name='salt_100g'"
                ).fetchone()[0], True)
            finally:
                connection.close()
        finally:
            sync.OVERRIDE_CSV_PATH = original_overrides
            sync.REVIEW_QUEUE_PATH = original_queue

    def test_busy_lock_is_rejected_and_released_after_operation(self):
        with pipeline_services._exclusive_operation(self.db_path):
            with self.assertRaises(pipeline_services.PipelineBusyError):
                with pipeline_services._exclusive_operation(self.root / "other.duckdb"):
                    self.fail("A duplicate operation unexpectedly acquired the lock")

        with pipeline_services._exclusive_operation(self.db_path):
            pass

    def test_writer_lock_is_shared_across_processes(self):
        child_code = (
            "from pathlib import Path; import sys; "
            "from pipeline_services import _exclusive_operation; "
            "p = Path(sys.argv[1]); lock = Path(sys.argv[2]); "
            "ctx = _exclusive_operation(p, lock); ctx.__enter__(); "
            "print('LOCKED', flush=True); sys.stdin.readline(); ctx.__exit__(None, None, None)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", child_code, str(self.db_path), str(self.root / ".pipeline-test.lock")],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        child_stderr = ""
        try:
            self.assertEqual(process.stdout.readline().strip(), "LOCKED")
            with self.assertRaises(pipeline_services.PipelineBusyError):
                with pipeline_services._exclusive_operation(self.db_path):
                    self.fail("A second process unexpectedly acquired the writer lock")
        finally:
            if process.stdin:
                process.stdin.write("release\n")
                process.stdin.flush()
            process.wait(timeout=10)
            if process.stderr:
                child_stderr = process.stderr.read()
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        self.assertEqual(process.returncode, 0, child_stderr)

    def test_export_failure_closes_connection_and_releases_lock(self):
        connection = duckdb.connect(str(self.db_path))
        connection.close()
        with patch.object(sync, "export_views_for_bi", side_effect=RuntimeError("expected test failure")):
            with self.assertRaisesRegex(RuntimeError, "expected test failure"):
                pipeline_services.refresh_exports(self.db_path)

        # Reopening verifies the operation released DuckDB's file handle; the
        # adjacent lock is acquirable again after the raised exception.
        reopened = duckdb.connect(str(self.db_path))
        reopened.close()
        with pipeline_services._exclusive_operation(self.db_path):
            pass

    def test_sync_failure_closes_database_and_releases_lock(self):
        class FakeLidl:
            refresh_token = "test-token"
            detected_country = None

            def __init__(self, **_kwargs):
                pass

        class FakeConnection:
            closed = False

            def close(self):
                self.closed = True

        connection = FakeConnection()
        events = []
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"LIDL_COUNTRY": "ro"}))
            stack.enter_context(patch.object(sync, "load_lidl_token", return_value="test-token"))
            stack.enter_context(patch.object(sync, "AutoDetectLidlPlusApi", FakeLidl))
            stack.enter_context(patch.object(sync, "configure_country"))
            stack.enter_context(patch.object(sync, "sync_tickets_incremental"))
            stack.enter_context(patch.object(pipeline_services.duckdb, "connect", return_value=connection))
            stack.enter_context(patch.object(sync, "init_database"))
            stack.enter_context(
                patch.object(sync, "apply_seed_catalog_to_db", side_effect=RuntimeError("expected failure"))
            )

            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                pipeline_services.run_sync(self.db_path, events.append)

        self.assertTrue(connection.closed)
        self.assertIn(("seed_catalog", "failed"), [(event.stage, event.status) for event in events])
        with pipeline_services._exclusive_operation(self.db_path):
            pass

    def _create_chat_import_db(self):
        con = duckdb.connect(str(self.db_path))
        con.execute("""
            CREATE TABLE nutrition_cache (
                product_key VARCHAR PRIMARY KEY, sample_name VARCHAR,
                macro_category VARCHAR, category VARCHAR, package_weight_g DOUBLE,
                kcal_100g DOUBLE, proteins_100g DOUBLE, carbs_100g DOUBLE,
                sugars_100g DOUBLE, fat_100g DOUBLE, saturated_fat_100g DOUBLE,
                fiber_100g DOUBLE, salt_100g DOUBLE, match_status VARCHAR,
                off_product_name VARCHAR, atwater_debug VARCHAR,
                data_source VARCHAR, is_verified BOOLEAN DEFAULT FALSE
            )
        """)
        con.execute("CREATE TABLE purchases (ean VARCHAR, raw_code VARCHAR)")
        create_schema(con)
        con.execute("""
            INSERT INTO nutrition_cache (
                product_key, sample_name, macro_category, category,
                package_weight_g, kcal_100g, proteins_100g, carbs_100g,
                sugars_100g, fat_100g, saturated_fat_100g, fiber_100g,
                salt_100g, match_status, data_source
            ) VALUES (
                '000123', 'Synthetic milk', 'Fresh Food', 'Dairy & Eggs',
                50, 100, 3, 5, 4, 7, 2, 1, NULL, 'TEST', 'SYNTHETIC'
            )
        """)
        con.close()

    def test_chat_batch_pipeline_entrypoints_preview_and_save_without_sync(self):
        self._create_chat_import_db()
        output = self.root / "ai_chat_batch.txt"
        review = self.root / "review.csv"
        response = '{"products":[{"product_key":"000123","salt_100g":0}]}'
        exports = []

        with ExitStack() as stack:
            stack.enter_context(patch.object(sync, "REVIEW_QUEUE_PATH", review))
            stack.enter_context(patch.object(
                sync, "export_views_for_bi", side_effect=lambda _con: exports.append("export")
            ))
            stack.enter_context(patch.object(
                sync, "sync_tickets_incremental", side_effect=AssertionError("sync must not run")
            ))

            self.assertEqual(pipeline_services.refresh_chat_batch(self.db_path, output), 1)
            self.assertTrue(output.is_file())
            preview = pipeline_services.preview_chat_import(response, self.db_path)
            self.assertEqual(preview.fields_suggested, 1)
            read_only = duckdb.connect(str(self.db_path), read_only=True)
            self.assertIsNone(read_only.execute(
                "SELECT salt_100g FROM nutrition_cache WHERE product_key='000123'"
            ).fetchone()[0])
            read_only.close()

            report = pipeline_services.save_chat_import(
                response, self.db_path, output_path=output
            )

        self.assertEqual((report.products_updated, report.fields_saved), (1, 1))
        self.assertEqual(exports, ["export"])
        self.assertTrue(review.is_file())
        self.assertFalse(output.exists())
        check = duckdb.connect(str(self.db_path), read_only=True)
        self.assertEqual(check.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()[0], 0.0)
        check.close()

    def test_post_save_export_failure_is_visible_and_keeps_database_save(self):
        self._create_chat_import_db()
        output = self.root / "ai_chat_batch.txt"
        review = self.root / "review.csv"
        response = '{"products":[{"product_key":"000123","salt_100g":0}]}'

        with ExitStack() as stack:
            stack.enter_context(patch.object(sync, "REVIEW_QUEUE_PATH", review))
            stack.enter_context(patch.object(
                sync, "export_views_for_bi", side_effect=OSError("simulated export lock")
            ))
            with self.assertRaisesRegex(RuntimeError, "import was saved.*exports"):
                pipeline_services.save_chat_import(
                    response, self.db_path, output_path=output
                )

        self.assertTrue(review.is_file())
        check = duckdb.connect(str(self.db_path), read_only=True)
        self.assertEqual(check.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()[0], 0.0)
        check.close()

    def test_review_session_keeps_manual_and_chat_provenance_separate(self):
        self._create_chat_import_db()
        con = duckdb.connect(str(self.db_path))
        con.execute(
            "UPDATE nutrition_cache SET package_weight_g=NULL, salt_100g=NULL, sugars_100g=NULL "
            "WHERE product_key='000123'"
        )
        from product_store import FIELD_NAMES
        baseline_row = con.execute(
            f"SELECT {', '.join(FIELD_NAMES)} FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()
        baseline = {"000123": dict(zip(FIELD_NAMES, baseline_row))}
        con.close()
        output = self.root / "ai_chat_batch.txt"
        review = self.root / "review.csv"
        response = (
            '{"products":[{"product_key":"000123",'
            '"salt_100g":5,"package_weight_g":0}]}'
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(sync, "REVIEW_QUEUE_PATH", review))
            stack.enter_context(patch.object(
                sync, "export_views_for_bi", side_effect=lambda _con: None
            ))
            report = pipeline_services.save_review_session(
                manual_edits={"000123": {"salt_100g": 0}},
                baselines=baseline,
                response_text=response,
                db_path=self.db_path,
                output_path=output,
            )

        self.assertEqual(report.manual_fields_saved, 1)
        self.assertEqual(report.chat_import.fields_saved, 1)
        self.assertFalse(report.artifact_failures)
        check = duckdb.connect(str(self.db_path), read_only=True)
        self.assertEqual(check.execute(
            "SELECT salt_100g, package_weight_g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone(), (0.0, 0.0))
        self.assertEqual(check.execute(
            "SELECT field_name, source, is_protected, provider, model "
            "FROM product_field_metadata WHERE product_key='000123' "
            "AND field_name IN ('salt_100g','package_weight_g') ORDER BY field_name"
        ).fetchall(), [
            ("package_weight_g", "AI", False, "chat_import", None),
            ("salt_100g", "MANUAL", True, None, None),
        ])
        check.close()

    def test_review_session_reports_stale_manual_field_without_overwriting(self):
        self._create_chat_import_db()
        con = duckdb.connect(str(self.db_path))
        from product_store import FIELD_NAMES
        baseline_row = con.execute(
            f"SELECT {', '.join(FIELD_NAMES)} FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()
        baseline = {"000123": dict(zip(FIELD_NAMES, baseline_row))}
        con.execute("UPDATE nutrition_cache SET salt_100g=0.2 WHERE product_key='000123'")
        con.close()
        with patch.object(sync, "REVIEW_QUEUE_PATH", self.root / "review.csv"):
            report = pipeline_services.save_review_session(
                manual_edits={"000123": {"salt_100g": 0}},
                baselines=baseline,
                db_path=self.db_path,
                output_path=self.root / "ai_chat_batch.txt",
            )
        self.assertEqual(report.conflicts, (("000123", "salt_100g"),))
        self.assertEqual(report.manual_fields_saved, 0)
        check = duckdb.connect(str(self.db_path), read_only=True)
        self.assertEqual(check.execute(
            "SELECT salt_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()[0], 0.2)
        check.close()

    def test_non_food_transition_rejects_newer_nutrient_value_and_protection(self):
        self._create_chat_import_db()
        con = duckdb.connect(str(self.db_path))
        from product_store import FIELD_NAMES
        baseline_row = con.execute(
            f"SELECT {', '.join(FIELD_NAMES)} FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone()
        baseline = {
            "000123": {
                **dict(zip(FIELD_NAMES, baseline_row)),
                "__field_metadata__": {},
            }
        }
        con.execute("UPDATE nutrition_cache SET salt_100g=0.2 WHERE product_key='000123'")
        con.execute(
            "INSERT INTO product_field_metadata "
            "(product_key,field_name,source,is_protected) VALUES ('000123','salt_100g','MANUAL',TRUE)"
        )
        con.close()

        report = pipeline_services.save_review_session(
            manual_edits={"000123": {"category": "Non-Food"}},
            baselines=baseline,
            db_path=self.db_path,
            output_path=self.root / "ai_chat_batch.txt",
        )
        self.assertIn(("000123", "salt_100g"), report.conflicts)
        self.assertEqual(report.manual_fields_saved, 0)
        check = duckdb.connect(str(self.db_path), read_only=True)
        self.assertEqual(check.execute(
            "SELECT category, salt_100g FROM nutrition_cache WHERE product_key='000123'"
        ).fetchone(), ("Dairy & Eggs", 0.2))
        self.assertEqual(check.execute(
            "SELECT source, is_protected FROM product_field_metadata "
            "WHERE product_key='000123' AND field_name='salt_100g'"
        ).fetchone(), ("MANUAL", True))
        check.close()


if __name__ == "__main__":
    unittest.main()
