"""Reusable entry points for the local Lidl data pipeline.

The command-line script and future local UI share these operations. Importing
this module is intentionally inert: it does not open the database, read
credentials, start a sync, or create any files.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import unicodedata
from typing import Callable, Iterator, Literal, Mapping

import duckdb

from ai_enrichment.chat_batch import (
    DEFAULT_CHAT_BATCH_PATH,
    ChatBatchPreview,
    ChatBatchSaveReport,
    preview_chat_batch_import,
    refresh_chat_batch_file,
    save_chat_batch_import,
)
from product_store import (
    FIELD_NAMES,
    NUTRIENT_FIELDS,
    create_schema,
    list_incomplete_products,
    regenerate_review_csv,
    save_manual_fields,
)
from product_store.schema import METADATA_TABLE

logger = logging.getLogger(__name__)

ProgressStatus = Literal["started", "progress", "completed", "partial", "failed"]
ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class ProgressEvent:
    """A truthful, presentation-neutral pipeline progress event."""

    stage: str
    status: ProgressStatus
    message: str
    completed: int | None = None
    total: int | None = None


@dataclass(frozen=True)
class ReviewSaveReport:
    """Outcome of a mixed manual/chat review save and artifact refresh."""

    manual_fields_saved: int
    chat_import: ChatBatchSaveReport | None
    conflicts: tuple[tuple[str, str], ...]
    artifact_failures: tuple[str, ...]
    database_error: str | None = None

    @property
    def database_saved(self) -> bool:
        return self.manual_fields_saved > 0 or bool(
            self.chat_import and self.chat_import.fields_saved
        ) or self.database_error is not None


@dataclass(frozen=True)
class AIRetryReport:
    """Selected AI retry result, including derived artifact failures."""

    enrichment: object
    artifact_failures: tuple[str, ...]


class PipelineBusyError(RuntimeError):
    """Raised when another process is already using this pipeline workspace."""


_thread_locks_guard = threading.Lock()
_thread_locks: dict[str, threading.Lock] = {}


def _load_sync_module():
    """Import the legacy pipeline only when an operation actually needs it."""
    return importlib.import_module("sync_incremental")


def _database_path(db_path: str | os.PathLike[str] | None, sync_module=None) -> Path:
    if db_path is None:
        sync_module = sync_module or _load_sync_module()
        db_path = sync_module.DB_PATH
    return Path(db_path).expanduser().resolve()


def _workspace_lock_path() -> Path:
    """Use one lock for the workspace's shared review/export artifacts."""
    return Path.cwd() / ".lidl-pipeline-operation.lock"


@contextmanager
def _exclusive_operation(_db_path: Path, lock_path: Path | None = None) -> Iterator[None]:
    """Serialize local operations across both threads and OS processes.

    The lock is workspace-scoped because review/export files are shared even
    when a caller points at a different database. The lock file is persistent;
    the OS releases its lock if a process exits unexpectedly.
    """
    lock_path = (lock_path or _workspace_lock_path()).resolve()
    lock_key = str(lock_path).casefold() if os.name == "nt" else str(lock_path)
    with _thread_locks_guard:
        thread_lock = _thread_locks.setdefault(lock_key, threading.Lock())

    if not thread_lock.acquire(blocking=False):
        raise PipelineBusyError("Another pipeline operation is already running in this workspace.")

    lock_file = None
    os_lock_acquired = False
    try:
        if not lock_path.parent.exists():
            raise FileNotFoundError(f"Pipeline workspace does not exist: {lock_path.parent}")
        lock_file = lock_path.open("a+b")
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)

        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            os_lock_acquired = True
        except (BlockingIOError, OSError) as exc:
            raise PipelineBusyError(
                "Another pipeline operation is already running in this workspace."
            ) from exc

        yield
    finally:
        try:
            if lock_file is not None:
                try:
                    if os_lock_acquired:
                        if os.name == "nt":
                            import msvcrt

                            lock_file.seek(0)
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_file.close()
        finally:
            thread_lock.release()


def _emit(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:  # UI/reporting failures must not abort data work.
        logger.warning("Progress callback failed (%s).", type(exc).__name__)


def _sync_event_adapter(callback: ProgressCallback | None) -> Callable[[dict[str, object]], None] | None:
    if callback is None:
        return None

    def forward(payload: dict[str, object]) -> None:
        _emit(
            callback,
            ProgressEvent(
                stage=str(payload.get("stage", "pipeline")),
                status=payload.get("status", "progress"),  # type: ignore[arg-type]
                message=str(payload.get("message", "Working")),
                completed=payload.get("completed") if isinstance(payload.get("completed"), int) else None,
                total=payload.get("total") if isinstance(payload.get("total"), int) else None,
            ),
        )

    return forward


def _run_stage(
    callback: ProgressCallback | None,
    stage: str,
    operation: Callable[[], object],
) -> object:
    _emit(callback, ProgressEvent(stage, "started", f"Starting {stage.replace('_', ' ')}."))
    try:
        result = operation()
    except Exception:
        _emit(callback, ProgressEvent(stage, "failed", f"{stage.replace('_', ' ').capitalize()} failed."))
        raise
    _emit(callback, ProgressEvent(stage, "completed", f"{stage.replace('_', ' ').capitalize()} complete."))
    return result


def run_sync(
    db_path: str | os.PathLike[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Run the existing receipt/catalog pipeline under one exclusive lock."""
    sync = _load_sync_module()
    path = _database_path(db_path, sync)

    with _exclusive_operation(path):
        _emit(progress_callback, ProgressEvent("sync", "started", "Starting synchronization."))
        connection = None
        try:
            env_country = os.getenv("LIDL_COUNTRY")
            lidl_token = _run_stage(progress_callback, "load_credentials", sync.load_lidl_token)
            lidl = _run_stage(
                progress_callback,
                "lidl_authentication",
                lambda: sync.AutoDetectLidlPlusApi(refresh_token=lidl_token, country=env_country),
            )

            active_country = env_country or lidl.detected_country or "ro"
            sync.configure_country(active_country)
            if not env_country and lidl.detected_country:
                print(f"[+] Auto-detected account country from JWT id_token: {lidl.detected_country}")
                sync.update_env_country(lidl.detected_country)
            else:
                print(f"[+] Active country: {active_country.upper()}")

            receipt_progress = _sync_event_adapter(progress_callback)
            _emit(progress_callback, ProgressEvent("receipt_sync", "started", "Downloading new receipts."))
            try:
                receipt_result = sync.sync_tickets_incremental(lidl, progress_callback=receipt_progress)
            except Exception:
                _emit(progress_callback, ProgressEvent("receipt_sync", "failed", "Receipt synchronization failed."))
                raise
            finally:
                if lidl.refresh_token and lidl.refresh_token != lidl_token:
                    sync.update_env_token(lidl.refresh_token)

            failed_pages = getattr(receipt_result, "failed_pages", 0)
            failed_receipts = getattr(receipt_result, "failed_receipts", 0)
            receipt_sync_partial = bool(failed_pages or failed_receipts)
            if receipt_sync_partial:
                _emit(
                    progress_callback,
                    ProgressEvent(
                        "receipt_sync",
                        "partial",
                        f"Receipt sync had {failed_pages} page and {failed_receipts} receipt download errors; continuing with saved receipts.",
                    ),
                )
            else:
                _emit(progress_callback, ProgressEvent("receipt_sync", "completed", "Receipt synchronization complete."))

            connection = duckdb.connect(str(path))
            _run_stage(progress_callback, "database_setup", lambda: sync.init_database(connection))
            _run_stage(progress_callback, "seed_catalog", lambda: sync.apply_seed_catalog_to_db(connection, quiet=False))
            # review_queue.csv is a derived worklist; it is never consumed here.
            _run_stage(progress_callback, "legacy_overrides", lambda: sync.apply_nutrition_overrides(connection, quiet=True))
            _run_stage(progress_callback, "receipt_import", lambda: sync.process_unprocessed_tickets_to_db(connection))
            _run_stage(progress_callback, "seed_refresh", lambda: sync.apply_seed_catalog_to_db(connection, quiet=True))
            _run_stage(
                progress_callback,
                "off_enrichment",
                lambda: sync.update_nutrition_cache_from_off(
                    connection,
                    progress_callback=receipt_progress,
                ),
            )
            _run_stage(progress_callback, "deterministic_filter", lambda: sync.run_deterministic_non_food_filter(connection))
            _run_stage(progress_callback, "package_weights", lambda: sync.compute_weights_and_quantities(connection))

            def initialize_product_metadata() -> None:
                from product_store import apply_migration, synchronize_field_metadata

                migration_report = apply_migration(connection, sync.OVERRIDE_CSV_PATH, sync.SEED_CATALOG_PATH)
                tracked_fields = synchronize_field_metadata(
                    connection,
                    sync.OVERRIDE_CSV_PATH,
                    sync.SEED_CATALOG_PATH,
                )
                if migration_report.already_applied:
                    print(
                        "[i] Product-field metadata migration already applied; "
                        f"tracked {tracked_fields} newly discovered or manually confirmed fields."
                    )
                else:
                    print(
                        f"[+] Product-field metadata initialized for {migration_report.products_scanned} "
                        f"products ({migration_report.metadata_rows_inserted + tracked_fields} fields; "
                        f"{len(migration_report.mismatched_overrides)} override mismatches reported)."
                    )

            # Initialize field-level protection before AI can consider any
            # product. This is essential on the first sync: legacy manual
            # overrides must be protected before model results are persisted.
            _run_stage(progress_callback, "field_metadata_prepare", initialize_product_metadata)

            from ai_enrichment import run_ai_enrichment

            try:
                run_ai_enrichment(
                    connection,
                    progress_callback=_sync_event_adapter(progress_callback),
                )
            except Exception:
                _emit(progress_callback, ProgressEvent("ai_enrichment", "failed", "AI enrichment storage failed."))
                raise

            _run_stage(progress_callback, "nutrition_validation", lambda: sync.validate_atwater(connection))
            _run_stage(progress_callback, "legacy_overrides_final", lambda: sync.apply_nutrition_overrides(connection, quiet=False))
            _run_stage(progress_callback, "override_warnings", lambda: sync.warn_suspicious_overrides(connection))

            def synchronize_final_metadata() -> None:
                from product_store import synchronize_field_metadata

                tracked_fields = synchronize_field_metadata(
                    connection,
                    sync.OVERRIDE_CSV_PATH,
                    sync.SEED_CATALOG_PATH,
                )
                print(f"[i] Final product metadata synchronization tracked {tracked_fields} fields.")

            _run_stage(progress_callback, "field_metadata", synchronize_final_metadata)
            _run_stage(progress_callback, "review_refresh", lambda: sync.update_review_queue(connection))
            def refresh_sync_chat_batch() -> int:
                from ai_eval.taxonomy import load_taxonomy

                taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
                count = refresh_chat_batch_file(
                    connection,
                    taxonomy,
                    taxonomy.category_to_macro,
                    DEFAULT_CHAT_BATCH_PATH,
                )
                print(f"[i] Chat batch refreshed for {count} unresolved products.")
                return count

            _run_stage(progress_callback, "chat_batch_refresh", refresh_sync_chat_batch)
            _run_stage(progress_callback, "exports", lambda: sync.export_views_for_bi(connection))
        except Exception:
            _emit(progress_callback, ProgressEvent("sync", "failed", "Synchronization failed."))
            raise
        finally:
            if connection is not None:
                connection.close()

        if receipt_sync_partial:
            _emit(
                progress_callback,
                ProgressEvent(
                    "sync",
                    "partial",
                    "Synchronization completed, but some receipts could not be downloaded.",
                ),
            )
        else:
            _emit(progress_callback, ProgressEvent("sync", "completed", "Synchronization complete."))


def _search_text(value: str) -> str:
    """Literal, accent-insensitive matching; not fuzzy or SQL-pattern matching."""
    return "".join(
        char for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )


def search_products(
    query: str,
    db_path: str | os.PathLike[str] | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, str | None]]:
    """Return bounded name suggestions from all products, including complete ones.

    Scan only identities/names, never nutrition or purchase history. Matching is
    a literal substring after case/accent folding; duplicate names retain keys.
    """
    if not isinstance(query, str):
        raise ValueError("Search text must be a string.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Search limit must be an integer between 1 and 50.")
    needle = _search_text(query.strip())
    if not needle:
        return []
    path = _database_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")
    with _exclusive_operation(path):
        connection = duckdb.connect(str(path), read_only=True)
        try:
            rows = connection.execute(
                "SELECT product_key, sample_name FROM nutrition_cache"
            ).fetchall()
        finally:
            connection.close()
    matches = [
        {"product_key": str(key), "sample_name": name}
        for key, name in rows
        if needle in _search_text(name or "")
    ]
    matches.sort(key=lambda row: (_search_text(row["sample_name"] or ""), row["product_key"]))
    return matches[:limit]


def read_product(
    product_key: str,
    db_path: str | os.PathLike[str] | None = None,
) -> dict[str, object] | None:
    """Read one product with a fresh value/provenance baseline, without writes."""
    path = _database_path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")
    with _exclusive_operation(path):
        connection = duckdb.connect(str(path), read_only=True)
        try:
            fields = ("product_key", "sample_name", *FIELD_NAMES)
            row = connection.execute(
                f"SELECT {', '.join(fields)} FROM nutrition_cache WHERE product_key = ?",
                [product_key],
            ).fetchone()
            if row is None:
                return None
            product = dict(zip(fields, row))
            try:
                metadata = connection.execute(
                    f"SELECT field_name, source, is_protected, provider, model, updated_at "
                    f"FROM {METADATA_TABLE} WHERE product_key = ?",
                    [product_key],
                ).fetchall()
            except duckdb.CatalogException:
                metadata = []
            product["field_metadata"] = {
                field: (source, bool(protected), provider, model, updated_at)
                for field, source, protected, provider, model, updated_at in metadata
            }
            return product
        finally:
            connection.close()


def read_review_products(
    db_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, object]]:
    """Read the current incomplete-product worklist from DuckDB."""
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    with _exclusive_operation(path):
        connection = duckdb.connect(str(path), read_only=True)
        try:
            _, category_to_macro = sync.load_taxonomy()
            products = list_incomplete_products(connection, category_to_macro)
            if products:
                keys = [str(product["product_key"]) for product in products]
                placeholders = ", ".join("?" for _ in keys)
                try:
                    metadata_rows = connection.execute(
                        f"SELECT product_key, field_name, source, is_protected, provider, model, updated_at "
                        f"FROM {METADATA_TABLE} WHERE product_key IN ({placeholders})",
                        keys,
                    ).fetchall()
                except duckdb.CatalogException:
                    metadata_rows = []
                metadata_by_key: dict[str, dict[str, tuple[object, ...]]] = {}
                for key, field_name, source, protected, provider, model, updated_at in metadata_rows:
                    metadata_by_key.setdefault(str(key), {})[str(field_name)] = (
                        source, bool(protected), provider, model, updated_at
                    )
                for product in products:
                    product["field_metadata"] = metadata_by_key.get(
                        str(product["product_key"]), {}
                    )
            return products
        finally:
            connection.close()


def save_review_fields(
    product_key: str,
    edits: Mapping[str, object],
    db_path: str | os.PathLike[str] | None = None,
) -> tuple[str, ...]:
    """Save explicit human edits and regenerate the derived review CSV."""
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    with _exclusive_operation(path):
        connection = duckdb.connect(str(path))
        try:
            _, category_to_macro = sync.load_taxonomy()
            legacy_fields_to_clear = _legacy_override_fields_to_clear(sync, edits)
            override_path = Path(sync.OVERRIDE_CSV_PATH)
            original_override_bytes = (
                override_path.read_bytes()
                if legacy_fields_to_clear and override_path.is_file()
                else None
            )
            try:
                _clear_legacy_override_values(sync, product_key, legacy_fields_to_clear)
                changed_fields = save_manual_fields(
                    connection,
                    product_key,
                    edits,
                    category_to_macro,
                    clear_legacy_override_fields=legacy_fields_to_clear,
                )
            except Exception as save_error:
                if original_override_bytes is not None:
                    try:
                        _restore_file_bytes(override_path, original_override_bytes)
                    except Exception as restore_error:
                        raise RuntimeError(
                            "The database save failed and the prior override CSV could not be restored. "
                            f"Save error: {save_error}; restore error: {restore_error}"
                        ) from save_error
                raise
            regenerate_review_csv(
                connection,
                category_to_macro,
                sync.REVIEW_QUEUE_PATH,
                sync.OVERRIDE_COLUMNS,
            )
            return changed_fields
        finally:
            connection.close()


def _restore_file_bytes(path: Path, content: bytes) -> None:
    """Atomically restore a CSV snapshot if the paired database edit fails."""
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".restore.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _legacy_override_fields_to_clear(sync_module, edits: Mapping[str, object]) -> set[str]:
    """Find stale legacy override cells superseded by an explicit clear."""
    cleared = {
        field_name
        for field_name, value in edits.items()
        if value is None or (isinstance(value, str) and not value.strip())
    }
    category = edits.get("category")
    category_cleared = "category" in edits and (
        category is None or (isinstance(category, str) and not category.strip())
    )
    if category_cleared:
        cleared.update(("category", "macro_category", "non_food"))
    elif isinstance(category, str) and category.strip() == "Non-Food":
        cleared.update(
            field_name
            for field_name in sync_module.OVERRIDE_NUM_COLUMNS
            if field_name != "package_weight_g"
        )
    return cleared


def _clear_legacy_override_values(
    sync_module,
    product_key: str,
    field_names: set[str],
) -> None:
    """Remove only explicitly cleared values from the legacy override CSV.

    The DuckDB transaction remains authoritative. This atomic CSV rewrite keeps
    the older sync importer from restoring a value that was explicitly cleared.
    """
    path = Path(sync_module.OVERRIDE_CSV_PATH)
    if not field_names or not path.is_file():
        return

    rows, delimiter, header = sync_module.read_csv_records(path)
    if not rows:
        return

    active_override_fields = (
        set(sync_module.OVERRIDE_TEXT_COLUMNS)
        | set(sync_module.OVERRIDE_NUM_COLUMNS)
        | {"non_food"}
    )
    known_columns = set(sync_module.OVERRIDE_COLUMNS)
    extra_columns = set(header) - known_columns
    updated_rows: list[dict[str, object]] = []
    changed = False

    for row in rows:
        if sync_module.clean_product_key(row.get("product_key")) == sync_module.clean_product_key(product_key):
            for field_name in field_names:
                if row.get(field_name) not in (None, ""):
                    changed = True
                row[field_name] = ""

            has_active_override = any(
                str(row.get(field_name) or "").strip()
                for field_name in active_override_fields | extra_columns
            )
            if not has_active_override:
                changed = True
                continue
        updated_rows.append(row)

    if changed:
        sync_module.write_csv_records(
            path,
            updated_rows,
            header or sync_module.OVERRIDE_COLUMNS,
            delimiter,
        )


def refresh_exports(
    db_path: str | os.PathLike[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Refresh report Parquet files only; does not fetch receipts or run AI."""
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    with _exclusive_operation(path):
        _emit(progress_callback, ProgressEvent("exports", "started", "Refreshing report files."))
        connection = None
        try:
            connection = duckdb.connect(str(path))
            sync.export_views_for_bi(connection)
        except Exception:
            _emit(progress_callback, ProgressEvent("exports", "failed", "Report export failed."))
            raise
        finally:
            if connection is not None:
                connection.close()
        _emit(progress_callback, ProgressEvent("exports", "completed", "Report files are up to date."))


def refresh_chat_batch(
    db_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] = DEFAULT_CHAT_BATCH_PATH,
) -> int:
    """Regenerate the offline chatbot prompt without sync or network calls."""
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    from ai_eval.taxonomy import load_taxonomy

    with _exclusive_operation(path):
        connection = duckdb.connect(str(path), read_only=True)
        try:
            taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
            return refresh_chat_batch_file(
                connection,
                taxonomy,
                taxonomy.category_to_macro,
                output_path,
            )
        finally:
            connection.close()


def preview_chat_import(
    response_text: str,
    db_path: str | os.PathLike[str] | None = None,
    *,
    blocked_fields: Mapping[str, tuple[str, ...] | list[str]] | None = None,
) -> ChatBatchPreview:
    """Validate a pasted chatbot response against current DB state only."""
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    from ai_eval.taxonomy import load_taxonomy

    with _exclusive_operation(path):
        connection = duckdb.connect(str(path), read_only=True)
        try:
            taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
            return preview_chat_batch_import(
                connection,
                response_text,
                taxonomy,
                taxonomy.category_to_macro,
                blocked_fields=blocked_fields,
            )
        finally:
            connection.close()


def _refresh_chat_import_artifacts(
    connection: duckdb.DuckDBPyConnection,
    sync,
    taxonomy,
    output_path: str | os.PathLike[str],
    progress_callback: ProgressCallback | None,
) -> None:
    """Attempt all derived refreshes and report any failures after DB commit."""
    operations = (
        (
            "review_refresh",
            lambda: regenerate_review_csv(
                connection,
                taxonomy.category_to_macro,
                sync.REVIEW_QUEUE_PATH,
                sync.OVERRIDE_COLUMNS,
            ),
        ),
        (
            "chat_batch_refresh",
            lambda: refresh_chat_batch_file(
                connection,
                taxonomy,
                taxonomy.category_to_macro,
                output_path,
            ),
        ),
        ("exports", lambda: sync.export_views_for_bi(connection)),
    )
    failures: list[tuple[str, Exception]] = []
    for stage, operation in operations:
        try:
            _run_stage(progress_callback, stage, operation)
        except Exception as exc:
            failures.append((stage, exc))
    if failures:
        failed_stages = ", ".join(stage for stage, _ in failures)
        raise RuntimeError(
            "The chatbot import was saved, but these derived artifacts could not "
            f"be refreshed: {failed_stages}. The database values were not rolled back."
        ) from failures[0][1]


def save_chat_import(
    response_text: str,
    db_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] = DEFAULT_CHAT_BATCH_PATH,
    progress_callback: ProgressCallback | None = None,
) -> ChatBatchSaveReport:
    """Revalidate/save an import, then refresh review, chat, and BI artifacts."""
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    from ai_eval.taxonomy import load_taxonomy

    with _exclusive_operation(path):
        connection = duckdb.connect(str(path))
        try:
            create_schema(connection)
            taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
            try:
                report = save_chat_batch_import(
                    connection,
                    response_text,
                    taxonomy,
                    taxonomy.category_to_macro,
                )
            except Exception as import_error:
                # A prior product may already have committed before a later
                # per-product transaction failed; keep derived files aligned.
                try:
                    _refresh_chat_import_artifacts(
                        connection, sync, taxonomy, output_path, progress_callback
                    )
                except Exception as refresh_error:
                    raise RuntimeError(
                        "The chatbot import may have partially saved; the import "
                        f"failed ({type(import_error).__name__}) and artifact refresh "
                        f"also failed ({refresh_error})."
                    ) from import_error
                raise

            _refresh_chat_import_artifacts(
                connection,
                sync,
                taxonomy,
                output_path,
                progress_callback,
            )
            return report
        finally:
            connection.close()


def _same_baseline_value(current: object, original: object) -> bool:
    """An unchanged stored NaN is not a concurrent edit.

    Keep NaN distinct from NULL/zero/real values: only two floating-point NaNs
    compare equal here. This changes comparison, not stored values or validation.
    """
    if isinstance(current, float) and isinstance(original, float):
        if math.isnan(current) and math.isnan(original):
            return True
    return current == original


def save_review_session(
    *,
    manual_edits: Mapping[str, Mapping[str, object]],
    baselines: Mapping[str, Mapping[str, object]],
    response_text: str = "",
    db_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] = DEFAULT_CHAT_BATCH_PATH,
    progress_callback: ProgressCallback | None = None,
) -> ReviewSaveReport:
    """Save only explicit manual cell edits and still-eligible chatbot fields.

    Manual edits are compared with their original DB values while holding the
    workspace lock. If another operation changed a field since the draft was
    loaded, that field is skipped and reported rather than overwritten.
    Chatbot suggestions for explicitly edited fields are blocked, preserving
    AI provenance for all untouched suggestions.
    """
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    from ai_eval.taxonomy import load_taxonomy

    conflicts: list[tuple[str, str]] = []
    manual_fields_saved = 0
    chat_report: ChatBatchSaveReport | None = None
    database_error: str | None = None
    chat_save_attempted = False
    with _exclusive_operation(path):
        connection = duckdb.connect(str(path))
        try:
            taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
            blocked_fields = {
                str(key): tuple(fields)
                for key, fields in manual_edits.items()
                if fields
            }

            # Preview is read-only and happens before any manual write so users
            # receive current eligibility/protection feedback for this save.
            if response_text.strip():
                preview_chat_batch_import(
                    connection,
                    response_text,
                    taxonomy,
                    taxonomy.category_to_macro,
                    blocked_fields=blocked_fields,
                )

            for raw_key, raw_edits in manual_edits.items():
                key = str(raw_key)
                if not key or not isinstance(raw_edits, Mapping):
                    continue
                row = connection.execute(
                    f"SELECT {', '.join(FIELD_NAMES)} FROM nutrition_cache WHERE product_key = ?",
                    [key],
                ).fetchone()
                if row is None:
                    conflicts.extend((key, str(field)) for field in raw_edits)
                    continue
                current = dict(zip(FIELD_NAMES, row))
                baseline = baselines.get(key, {})
                baseline_metadata = baseline.get("__field_metadata__")
                accepted: dict[str, object] = {}
                for field_name, value in raw_edits.items():
                    if field_name not in FIELD_NAMES or field_name == "macro_category":
                        continue
                    if field_name not in baseline or not _same_baseline_value(current.get(field_name), baseline.get(field_name)):
                        conflicts.append((key, field_name))
                        continue
                    accepted[field_name] = value
                if not accepted:
                    continue
                # The service owns Non-Food semantics; stale visible nutrition
                # cells are not sent as edits when changing category to Non-Food.
                if accepted.get("category") == "Non-Food":
                    current_metadata_rows = connection.execute(
                        f"SELECT field_name, source, is_protected, provider, model, updated_at "
                        f"FROM {METADATA_TABLE} WHERE product_key = ?",
                        [key],
                    ).fetchall()
                    current_metadata = {
                        str(field): (source, bool(protected), provider, model, updated_at)
                        for field, source, protected, provider, model, updated_at
                        in current_metadata_rows
                    }
                    dependent_conflicts = [
                        field
                        for field in NUTRIENT_FIELDS
                        if baseline_metadata is None
                        or not _same_baseline_value(current.get(field), baseline.get(field))
                        or current_metadata.get(field) != baseline_metadata.get(field)
                    ]
                    if dependent_conflicts:
                        conflicts.extend((key, field) for field in dependent_conflicts)
                        accepted.pop("category", None)
                    else:
                        accepted = {
                            field: value for field, value in accepted.items()
                            if field not in NUTRIENT_FIELDS
                        }
                if accepted:
                    legacy_fields = _legacy_override_fields_to_clear(sync, accepted)
                    override_path = Path(sync.OVERRIDE_CSV_PATH)
                    original_override_bytes = (
                        override_path.read_bytes()
                        if legacy_fields and override_path.is_file()
                        else None
                    )
                    try:
                        _clear_legacy_override_values(sync, key, legacy_fields)
                        changed = save_manual_fields(
                            connection,
                            key,
                            accepted,
                            taxonomy.category_to_macro,
                            clear_legacy_override_fields=legacy_fields,
                        )
                    except Exception as exc:
                        if original_override_bytes is not None:
                            _restore_file_bytes(override_path, original_override_bytes)
                        database_error = f"Manual save stopped ({type(exc).__name__})."
                        break
                    manual_fields_saved += len(changed)

            if response_text.strip() and database_error is None:
                try:
                    chat_save_attempted = True
                    chat_report = save_chat_batch_import(
                        connection,
                        response_text,
                        taxonomy,
                        taxonomy.category_to_macro,
                        blocked_fields=blocked_fields,
                    )
                except Exception as exc:
                    database_error = f"Chat import stopped ({type(exc).__name__})."

            database_changed = manual_fields_saved > 0 or chat_save_attempted or bool(
                chat_report and chat_report.fields_saved
            )
            if database_changed:
                failures: list[str] = []
                operations = (
                    (
                        "review_refresh",
                        lambda: regenerate_review_csv(
                            connection,
                            taxonomy.category_to_macro,
                            sync.REVIEW_QUEUE_PATH,
                            sync.OVERRIDE_COLUMNS,
                        ),
                    ),
                    (
                        "chat_batch_refresh",
                        lambda: refresh_chat_batch_file(
                            connection,
                            taxonomy,
                            taxonomy.category_to_macro,
                            output_path,
                        ),
                    ),
                    ("exports", lambda: sync.export_views_for_bi(connection)),
                )
                for stage, operation in operations:
                    try:
                        _run_stage(progress_callback, stage, operation)
                    except Exception:
                        failures.append(stage)
                artifact_failures = tuple(failures)
            else:
                artifact_failures = ()
        except Exception:
            # Keep derived artifacts aligned if an earlier product committed
            # before a later manual operation failed.
            if manual_fields_saved:
                try:
                    taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
                    _refresh_chat_import_artifacts(
                        connection, sync, taxonomy, output_path, progress_callback
                    )
                except Exception:
                    pass
            raise
        finally:
            connection.close()

    return ReviewSaveReport(
        manual_fields_saved=manual_fields_saved,
        chat_import=chat_report,
        conflicts=tuple(conflicts),
        artifact_failures=artifact_failures,
        database_error=database_error,
    )


def retry_ai_products(
    product_keys: list[str] | tuple[str, ...],
    db_path: str | os.PathLike[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AIRetryReport:
    """Explicitly retry AI only for selected products, then refresh artifacts."""
    selected = tuple(dict.fromkeys(key for key in product_keys if isinstance(key, str) and key))
    if not selected:
        raise ValueError("Select at least one product for AI retry.")
    sync = _load_sync_module()
    path = _database_path(db_path, sync)
    if not path.is_file():
        raise FileNotFoundError(f"Product database does not exist: {path}")

    from ai_enrichment import run_ai_enrichment
    from ai_eval.taxonomy import load_taxonomy

    with _exclusive_operation(path):
        connection = duckdb.connect(str(path))
        try:
            enrichment = run_ai_enrichment(
                connection,
                progress_callback=_sync_event_adapter(progress_callback),
                retry_product_keys=selected,
            )
            taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
            failures: list[str] = []
            for stage, operation in (
                (
                    "review_refresh",
                    lambda: regenerate_review_csv(
                        connection,
                        taxonomy.category_to_macro,
                        sync.REVIEW_QUEUE_PATH,
                        sync.OVERRIDE_COLUMNS,
                    ),
                ),
                (
                    "chat_batch_refresh",
                    lambda: refresh_chat_batch_file(
                        connection,
                        taxonomy,
                        taxonomy.category_to_macro,
                        DEFAULT_CHAT_BATCH_PATH,
                    ),
                ),
                ("exports", lambda: sync.export_views_for_bi(connection)),
            ):
                try:
                    _run_stage(progress_callback, stage, operation)
                except Exception:
                    failures.append(stage)
            return AIRetryReport(enrichment, tuple(failures))
        finally:
            connection.close()
