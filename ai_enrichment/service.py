"""Production Groq enrichment workflow over DuckDB-backed product state."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence

import duckdb

from ai_eval.parsing import ParseError, parse_model_output, validate_response
from ai_eval.prompts import build_messages
from ai_eval.providers import ProviderResponse
from ai_eval.taxonomy import Taxonomy, load_taxonomy
from product_store import list_ai_enrichment_candidates, save_ai_enrichment_result

from .config import AIEnrichmentConfig, AIEnrichmentConfigError, load_config
from .provider import GroqProductionClient

_NUTRITION_ISSUE_CODES = frozenset({
    "nutrition_not_numeric",
    "nutrition_out_of_range",
    "macro_sum_exceeds_105",
    "sugars_exceeds_carbs",
    "saturated_fat_exceeds_fat",
    "atwater_mismatch",
    "non_food_nutrition",
    "nutrition_provided_in_none_mode",
})
_MAX_COMPLETION_TOKENS = 4096


@dataclass(frozen=True)
class AIEnrichmentReport:
    status: str
    products_considered: int = 0
    batches_completed: int = 0
    batches_total: int = 0
    fields_saved: int = 0
    retryable_tasks: int = 0
    terminal_tasks: int = 0
    error_code: str | None = None


def _emit(
    callback: Callable[[dict[str, object]], None] | None,
    status: str,
    message: str,
    completed: int = 0,
    total: int = 0,
) -> None:
    if callback is None:
        return
    try:
        callback({
            "stage": "ai_enrichment",
            "status": status,
            "message": message,
            "completed": completed,
            "total": total,
        })
    except Exception:
        # Progress reporting must never make data writes fail.
        return


def _chunks(items: Sequence[dict[str, object]], size: int) -> list[list[dict[str, object]]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _model_payload(product: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "product_key": str(product.get("product_key") or ""),
        "sample_name": str(product.get("sample_name") or ""),
    }
    weight = product.get("package_weight_g")
    if weight is not None:
        payload["package_weight_g"] = weight
    return payload


def _category_result(validation: Any) -> dict[str, object]:
    issue_codes = set(validation.issue_codes)
    if not validation.found:
        return {"status": "invalid", "error_code": "missing_product"}
    if validation.abstained:
        return {"status": "abstained", "error_code": "category_abstained"}
    if validation.category is None or "duplicate_product_key" in issue_codes:
        return {"status": "invalid", "error_code": "category_invalid"}
    # A model-supplied macro is not authoritative; the canonical taxonomy
    # derives it from the category. The parser flags disagreement, but the
    # independently valid category remains usable.
    return {"status": "succeeded", "value": validation.category}


def _safe_http_error_code(response: ProviderResponse) -> str:
    if response.status_code is not None:
        return f"http_{response.status_code}"
    return response.error_kind or "provider_error"


def _is_retryable_provider_failure(response: ProviderResponse) -> bool:
    """Distinguish transient transport/quota failures from invalid replies.

    Authentication failures stay retryable because the user may correct the
    key before the next sync. Invalid JSON/content/response shapes and other
    non-transient client errors are terminal until an explicit retry exists.
    """
    if response.error_kind in {
        "timeout",
        "connection_error",
        "rate_limited",
        "deadline_exceeded",
    }:
        return True
    if response.error_kind == "http_error":
        status = response.status_code
        return status == 429 or status in {401, 403} or (status is not None and status >= 500)
    return False


def _mark_batch_retryable(
    con: duckdb.DuckDBPyConnection,
    batch: Sequence[dict[str, object]],
    category_to_macro: Mapping[str, str],
    model: str,
    attempts: int,
    error_code: str,
) -> int:
    count = 0
    for product in batch:
        category_result = (
            {"status": "retryable_error", "error_code": error_code}
            if product.get("category_task")
            else None
        )
        nutrition_result = (
            {"status": "retryable_error", "error_code": error_code}
            if product.get("nutrition_task")
            else None
        )
        save_ai_enrichment_result(
            con,
            str(product["product_key"]),
            category_result=category_result,
            nutrition_result=nutrition_result,
            category_to_macro=category_to_macro,
            provider="groq",
            model=model,
            attempt_count=attempts,
        )
        count += int(category_result is not None) + int(nutrition_result is not None)
    return count


def _parse_batch(
    content: str,
    requested_keys: Sequence[str],
    taxonomy: Taxonomy,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, object]], set[str], set[str]]:
    payload = parse_model_output(content)
    raw_products = payload.get("products")
    if not isinstance(raw_products, list):
        raise ParseError("Model response must contain a products array.")
    raw_by_key: dict[str, Mapping[str, object]] = {}
    duplicates: set[str] = set()
    for raw_product in raw_products:
        if not isinstance(raw_product, Mapping):
            continue
        key = raw_product.get("product_key")
        if not isinstance(key, str):
            continue
        if key in raw_by_key:
            duplicates.add(key)
        else:
            raw_by_key[key] = raw_product
    validation = validate_response(
        payload,
        requested_keys,
        taxonomy,
        mode,
        allow_abstain=True,
    )
    batch_issue_codes = {issue.code for issue in validation.batch_issues}
    return validation.products, raw_by_key, duplicates, batch_issue_codes


def _validate_product_result(
    product: Mapping[str, object],
    raw_product: Mapping[str, object] | None,
    category_validation: Any,
    duplicate_keys: set[str],
    taxonomy: Taxonomy,
    category_to_macro: Mapping[str, str],
    nutrition_requested: bool,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    category_result = None
    if product.get("category_task"):
        if raw_product is not None and str(product["product_key"]) in duplicate_keys:
            category_result = {"status": "invalid", "error_code": "duplicate_product_key"}
        else:
            category_result = _category_result(category_validation)

    nutrition_result = None
    if not nutrition_requested or not product.get("nutrition_task"):
        return category_result, nutrition_result
    if raw_product is None or str(product["product_key"]) in duplicate_keys:
        return category_result, {"status": "invalid", "error_code": "missing_or_duplicate_product"}

    current_category = product.get("category")
    if isinstance(current_category, str) and current_category in category_to_macro:
        effective_category = current_category
    elif category_result is not None and category_result.get("status") == "succeeded":
        effective_category = str(category_result["value"])
    else:
        # Nutrition has not become an independent task yet: without a usable
        # category, keep it eligible for a later sync after category resolution.
        return category_result, None

    if effective_category == "Non-Food":
        return category_result, None

    merged = dict(raw_product)
    merged["category"] = effective_category
    # When category is already known, the model's category/abstention is not a
    # nutrition answer; discard its abstention metadata before checking nutrition.
    merged["abstain_reason"] = None
    for field_name in (
        "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
        "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
    ):
        current = product.get(field_name)
        if current is not None:
            merged[field_name] = current
    for field_name in (
        "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
        "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
    ):
        if field_name not in raw_product:
            return category_result, {"status": "invalid", "error_code": "missing_nutrition_field"}

    checked = validate_response(
        {"products": [merged]},
        [str(product["product_key"])],
        taxonomy,
        "estimate",
        allow_abstain=True,
    ).products[str(product["product_key"])]
    nutrition_issue = next(
        (issue.code for issue in checked.issues if issue.code in _NUTRITION_ISSUE_CODES),
        None,
    )
    if nutrition_issue is not None:
        return category_result, {"status": "invalid", "error_code": nutrition_issue}

    has_new_estimate = any(
        raw_product.get(field_name) is not None and product.get(field_name) is None
        for field_name in (
            "kcal_100g", "proteins_100g", "carbs_100g", "sugars_100g",
            "fat_100g", "saturated_fat_100g", "fiber_100g", "salt_100g",
        )
    )
    if not has_new_estimate:
        return category_result, {"status": "abstained", "error_code": "nutrition_abstained"}
    return category_result, {"status": "succeeded", "values": checked.nutrition}


def run_ai_enrichment(
    con: duckdb.DuckDBPyConnection,
    *,
    config: AIEnrichmentConfig | None = None,
    taxonomy: Taxonomy | None = None,
    client_factory: Callable[..., Any] = GroqProductionClient,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    env_path: str | None = None,
    environ: Mapping[str, str] | None = None,
    retry_product_keys: Sequence[str] | None = None,
) -> AIEnrichmentReport:
    """Run opt-in Groq enrichment and persist each product result immediately."""
    _emit(progress_callback, "started", "Checking AI enrichment settings.")
    try:
        active_config = config or load_config(env_path=env_path, environ=environ)
    except AIEnrichmentConfigError:
        report = AIEnrichmentReport(status="skipped", error_code="invalid_configuration")
        _emit(progress_callback, "partial", "AI enrichment configuration is invalid; continuing without AI.")
        return report

    if not active_config.enabled:
        report = AIEnrichmentReport(status="disabled")
        _emit(progress_callback, "completed", "AI enrichment is disabled.")
        return report
    if not active_config.api_key:
        report = AIEnrichmentReport(status="skipped", error_code="missing_api_key")
        _emit(progress_callback, "partial", "AI enrichment is enabled but GROQ_API_KEY is not configured.")
        return report

    try:
        selected_taxonomy = taxonomy or load_taxonomy()
    except Exception:
        report = AIEnrichmentReport(status="skipped", error_code="taxonomy_unavailable")
        _emit(progress_callback, "partial", "Canonical taxonomy is unavailable; continuing without AI.")
        return report

    if retry_product_keys is not None:
        if any(not isinstance(key, str) or not key.strip() for key in retry_product_keys):
            raise ValueError("Explicit AI retry requires non-empty text product keys")
        retry_product_keys = tuple(dict.fromkeys(retry_product_keys))

    category_to_macro = selected_taxonomy.category_to_macro
    nutrition_enabled = active_config.nutrition_enabled
    candidates = list_ai_enrichment_candidates(
        con,
        category_to_macro,
        nutrition_enabled=nutrition_enabled,
        selected_product_keys=retry_product_keys,
        retry_terminal=retry_product_keys is not None,
    )
    if not candidates:
        report = AIEnrichmentReport(status="completed")
        _emit(progress_callback, "completed", "No eligible products need AI enrichment.")
        return report

    batches = _chunks(candidates, active_config.batch_size)
    deadline = clock() + active_config.budget_seconds
    client = client_factory(
        api_key=active_config.api_key,
        model=active_config.model,
        timeout=active_config.request_timeout_seconds,
        max_attempts=active_config.max_attempts,
        sleep_fn=sleep_fn,
    )
    completed_batches = 0
    fields_saved = 0
    retryable_tasks = 0
    terminal_tasks = 0
    status = "completed"
    stopped_for_budget = False

    for batch_index, batch in enumerate(batches):
        if clock() >= deadline:
            stopped_for_budget = True
            status = "partial"
            remaining_products = [product for pending in batches[batch_index:] for product in pending]
            retryable_tasks += _mark_batch_retryable(
                con, remaining_products, category_to_macro, active_config.model, 0, "budget_exceeded"
            )
            break

        payload = [_model_payload(product) for product in batch]
        mode = "estimate" if nutrition_enabled else "none"
        messages = build_messages(
            selected_taxonomy,
            payload,
            mode=mode,
            allow_abstain=True,
        )
        response = client.chat(messages, max_tokens=_MAX_COMPLETION_TOKENS, deadline=deadline)
        if not response.ok:
            code = response.error_kind or "provider_error"
            if code == "missing_api_key":
                # Configuration errors are reported but do not create product
                # attempts; a later configuration fix can run normally.
                _emit(progress_callback, "partial", "Groq credentials are unavailable; continuing without AI.", batch_index, len(batches))
                return AIEnrichmentReport(
                    status="partial",
                    products_considered=len(candidates),
                    batches_completed=completed_batches,
                    batches_total=len(batches),
                    fields_saved=fields_saved,
                    retryable_tasks=retryable_tasks,
                    terminal_tasks=terminal_tasks,
                    error_code=code,
                )
            if code == "deadline_exceeded":
                stopped_for_budget = True
                remaining_products = [product for pending in batches[batch_index:] for product in pending]
                retryable_tasks += _mark_batch_retryable(
                    con,
                    remaining_products,
                    category_to_macro,
                    active_config.model,
                    response.attempts,
                    "budget_exceeded",
                )
                status = "partial"
                break
            if not _is_retryable_provider_failure(response):
                terminal_tasks += _mark_invalid_batch(
                    con,
                    batch,
                    category_to_macro,
                    active_config.model,
                    max(response.attempts, 1),
                    code,
                )
                status = "partial"
                completed_batches += 1
                _emit(
                    progress_callback,
                    "progress",
                    "Groq returned an invalid response; affected products will not be retried automatically.",
                    completed_batches,
                    len(batches),
                )
                continue
            retryable_tasks += _mark_batch_retryable(
                con,
                batch,
                category_to_macro,
                active_config.model,
                max(response.attempts, 1),
                _safe_http_error_code(response),
            )
            status = "partial"
            _emit(progress_callback, "progress", "A Groq request failed; its products can be retried next sync.", batch_index + 1, len(batches))
            completed_batches += 1
            continue

        requested_keys = [str(product["product_key"]) for product in batch]
        try:
            validations, raw_by_key, duplicate_keys, batch_issue_codes = _parse_batch(
                response.content or "", requested_keys, selected_taxonomy, mode
            )
        except (ParseError, ValueError, RecursionError):
            terminal_tasks += _mark_invalid_batch(
                con, batch, category_to_macro, active_config.model, max(response.attempts, 1), "invalid_response"
            )
            status = "partial"
            completed_batches += 1
            _emit(progress_callback, "progress", "Groq returned an invalid response; affected products remain in review.", completed_batches, len(batches))
            continue

        protocol_issues = batch_issue_codes & {
            "extra_product",
            "invalid_product_entry",
            "invalid_product_key",
        }
        if protocol_issues:
            error_code = sorted(protocol_issues)[0]
            terminal_tasks += _mark_invalid_batch(
                con,
                batch,
                category_to_macro,
                active_config.model,
                max(response.attempts, 1),
                error_code,
            )
            status = "partial"
            completed_batches += 1
            _emit(
                progress_callback,
                "progress",
                "Groq returned unrequested or malformed product IDs; affected products will not be applied.",
                completed_batches,
                len(batches),
            )
            continue

        for product in batch:
            key = str(product["product_key"])
            validation = validations[key]
            raw_product = raw_by_key.get(key)
            category_result, nutrition_result = _validate_product_result(
                product,
                raw_product,
                validation,
                duplicate_keys,
                selected_taxonomy,
                category_to_macro,
                nutrition_enabled,
            )
            if category_result is not None and category_result["status"] in {"abstained", "invalid"}:
                terminal_tasks += 1
            if nutrition_result is not None and nutrition_result["status"] in {"abstained", "invalid"}:
                terminal_tasks += 1
            if category_result is not None and category_result["status"] == "retryable_error":
                retryable_tasks += 1
            if nutrition_result is not None and nutrition_result["status"] == "retryable_error":
                retryable_tasks += 1
            changed = save_ai_enrichment_result(
                con,
                key,
                category_result=category_result,
                nutrition_result=nutrition_result,
                category_to_macro=category_to_macro,
                provider="groq",
                model=active_config.model,
                attempt_count=max(response.attempts, 1),
            )
            fields_saved += len(changed)
        completed_batches += 1
        _emit(
            progress_callback,
            "progress",
            "AI response batch saved.",
            completed_batches,
            len(batches),
        )

    if status == "completed" and terminal_tasks:
        # Abstentions and invalid model answers are expected unresolved outcomes,
        # not stage failures. Keep the stage completed; products stay reviewable.
        status = "completed"
    if stopped_for_budget:
        _emit(progress_callback, "partial", "AI time budget reached; saved results are kept and remaining products can retry next sync.", completed_batches, len(batches))
    else:
        _emit(
            progress_callback,
            "partial" if status == "partial" else "completed",
            "AI enrichment finished; unresolved products remain in review." if terminal_tasks or status == "partial" else "AI enrichment completed.",
            completed_batches,
            len(batches),
        )
    return AIEnrichmentReport(
        status=status,
        products_considered=len(candidates),
        batches_completed=completed_batches,
        batches_total=len(batches),
        fields_saved=fields_saved,
        retryable_tasks=retryable_tasks,
        terminal_tasks=terminal_tasks,
        error_code="budget_exceeded" if stopped_for_budget else None,
    )


def _mark_invalid_batch(
    con: duckdb.DuckDBPyConnection,
    batch: Sequence[dict[str, object]],
    category_to_macro: Mapping[str, str],
    model: str,
    attempts: int,
    error_code: str,
) -> int:
    count = 0
    for product in batch:
        category_result = (
            {"status": "invalid", "error_code": error_code}
            if product.get("category_task")
            else None
        )
        nutrition_result = (
            {"status": "invalid", "error_code": error_code}
            if product.get("nutrition_task")
            else None
        )
        save_ai_enrichment_result(
            con,
            str(product["product_key"]),
            category_result=category_result,
            nutrition_result=nutrition_result,
            category_to_macro=category_to_macro,
            provider="groq",
            model=model,
            attempt_count=attempts,
        )
        count += int(category_result is not None) + int(nutrition_result is not None)
    return count
