"""Offline regressions for a cancelled receipt download and its retry."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sync_incremental as sync


class InterruptedDownload(BaseException):
    """Match a script cancellation, which is not a normal API failure."""


class SyncRecoveryTests(unittest.TestCase):
    def test_retry_reaches_older_history_after_interruption(self):
        class FakeLidl:
            interrupted = False

            def receipts(self, pageNumber, only_favorite):
                tickets = {1: ["new-1", "new-2"], 2: ["old-1", "old-2"]}
                return {"tickets": [{"id": key} for key in tickets.get(pageNumber, [])], "size": 2}

            def receipt(self, ticket_id):
                if ticket_id == "old-1" and not self.interrupted:
                    self.interrupted = True
                    raise InterruptedDownload()
                return {"id": ticket_id, "itemsLine": []}

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            client = FakeLidl()
            with patch.object(sync, "RAW_DIR", raw), patch.object(sync.time, "sleep"):
                with self.assertRaises(InterruptedDownload):
                    sync.sync_tickets_incremental(client)
                first_page = {path.stem: path.read_bytes() for path in raw.glob("*.json")}
                self.assertEqual(set(first_page), {"new-1", "new-2"})
                result = sync.sync_tickets_incremental(client)
                self.assertEqual(
                    {path.stem for path in raw.glob("*.json")},
                    {"new-1", "new-2", "old-1", "old-2"},
                )
                self.assertEqual(result.downloaded, 2)
                for key, original in first_page.items():
                    self.assertEqual((raw / f"{key}.json").read_bytes(), original)
                self.assertEqual(sync.sync_tickets_incremental(client).downloaded, 0)

    def test_old_hole_is_retried_even_when_newest_page_is_present(self):
        class FakeLidl:
            def receipts(self, pageNumber, only_favorite):
                return {"tickets": [{"id": "latest" if pageNumber == 1 else "missing"}]
                        if pageNumber <= 2 else [], "size": 1}

            def receipt(self, ticket_id):
                return {"id": ticket_id, "itemsLine": []}

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "latest.json").write_text(json.dumps({"id": "latest"}), encoding="utf-8")
            with patch.object(sync, "RAW_DIR", raw), patch.object(sync.time, "sleep"):
                result = sync.sync_tickets_incremental(FakeLidl())
            self.assertTrue((raw / "missing.json").is_file())
            self.assertEqual(result.downloaded, 1)
