"""Result persistence, aggregation, and human-readable reporting."""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import CONFIDENCE_TIERS, NUTRITION_FIELDS
from .dataset import EvalDataset, is_nutrition_comparable
from .providers import redact_text


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_value(value: Any, secrets: Sequence[str] = ()) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, float):
        # JSON has no NaN/Infinity: never emit invalid, non-round-trippable JSON.
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: sanitize_value(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item, secrets) for item in value]
    return value


_TAIL_CHUNK_BYTES = 8192
# One JSONL record is normally tiny. A larger trailing fragment is treated as
# torn instead of being loaded into memory to test for completeness.
_MAX_TAIL_REPAIR_BYTES = 1024 * 1024


def _last_line_start(handle: Any, size: int) -> int:
    """Offset just after the last newline, or 0 when the file has none."""
    position = size
    while position > 0:
        chunk_size = min(_TAIL_CHUNK_BYTES, position)
        position -= chunk_size
        handle.seek(position)
        chunk = handle.read(chunk_size)
        newline = chunk.rfind(b"\n")
        if newline != -1:
            return position + newline + 1
    return 0


def _tail_is_complete_record(handle: Any, tail_start: int, size: int) -> bool:
    """True when the bytes after the last newline form one valid JSON object."""
    length = size - tail_start
    if length <= 0 or length > _MAX_TAIL_REPAIR_BYTES:
        return False
    handle.seek(tail_start)
    raw = handle.read(length)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return False
    return isinstance(parsed, dict)


def _repair_torn_tail(destination: Path) -> None:
    """Ensure the file ends on a clean line boundary before an append.

    Appending to a file whose last byte is not ``\\n`` would glue the next
    JSON record onto the fragment, losing both records. A killed process can
    leave a partial final line, but it can also leave a *complete* record that
    merely lacks its final newline: truncating back to the last newline would
    delete that record, so a trailing object that parses cleanly is kept and
    only the missing newline is added.
    """
    if not destination.exists():
        return
    try:
        with destination.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return
            tail_start = _last_line_start(handle, size)
            if _tail_is_complete_record(handle, tail_start, size):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                return
            # The tail is a torn record (possibly the whole file): drop it.
            handle.seek(tail_start)
            handle.truncate()
    except OSError:
        # Best effort: never let tail repair break a live run.
        return


def append_results(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    secrets: Sequence[str] = (),
) -> None:
    """Append JSONL records after redacting all string fields."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _repair_torn_tail(destination)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            safe = sanitize_value(dict(record), secrets)
            handle.write(
                json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n"
            )


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL results, tolerating a partially written final record.

    The file is read as bytes and each complete line is decoded on its own. A
    process killed mid-write can leave a truncated multibyte character
    (Romanian diacritics appear in product names); decoding the whole file at
    once would raise ``UnicodeDecodeError`` and abort resume. An undecodable
    line is a torn record: it is skipped without dropping the lines after it.
    """
    source = Path(path)
    if not source.exists():
        return []
    try:
        raw = source.read_bytes()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    # Split on b"\n" only: str.splitlines() also splits on characters such as
    # U+2028, which are legal inside a JSON string (ensure_ascii=False).
    for raw_line in raw.split(b"\n"):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            # Torn record: skip it, keep the following complete lines.
            continue
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            # A truncated final line must not make resume unusable.
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _record_abstain(record: Mapping[str, Any]) -> bool:
    """True only for records written by an ``--abstain`` run.

    Forced-choice records, including legacy lines written before the field
    existed, are the baseline; they never satisfy an abstain selection.
    """
    return record.get("abstain") is True


def latest_results(
    records: Iterable[Mapping[str, Any]],
    providers: Sequence[str] | None = None,
    mode: str | None = None,
    abstain: bool | None = None,
) -> list[dict[str, Any]]:
    """Keep the last record per provider + model + product key.

    When ``mode`` is given, only records with exactly that stored mode are
    considered. Legacy records without a ``mode`` field never match a specific
    mode, so a category-only success can never hide an untested estimate run.
    When ``abstain`` is given, only records from the same prompt variant are
    considered, so a forced-choice run can never contaminate an abstain summary.
    """
    allowed = set(providers) if providers else None
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        provider = str(record.get("provider") or "")
        if allowed is not None and provider not in allowed:
            continue
        if mode is not None and record.get("mode") != mode:
            continue
        if abstain is not None and _record_abstain(record) != abstain:
            continue
        key = (
            provider,
            str(record.get("model") or ""),
            str(record.get("product_key") or ""),
        )
        latest[key] = dict(record)
    return list(latest.values())


def completed_keys(
    records: Iterable[Mapping[str, Any]],
    provider: str,
    model: str,
    mode: str | None = None,
    abstain: bool | None = None,
) -> set[str]:
    """Product keys with a valid successful result for this provider/model.

    A specific ``mode`` requires exact mode equality; records without a stored
    mode are treated as untested for that mode. An explicit ``abstain`` value
    is part of the result identity as well: a forced-choice record never
    satisfies an abstain run and vice versa.
    """
    completed: set[str] = set()
    for record in records:
        if record.get("provider") != provider or record.get("model") != model:
            continue
        if mode is not None and record.get("mode") != mode:
            continue
        if abstain is not None and _record_abstain(record) != abstain:
            continue
        validation = record.get("validation") or {}
        if record.get("status") == "ok" and validation.get("valid") is True:
            key = record.get("product_key")
            if isinstance(key, str) and key:
                completed.add(key)
    return completed


def _predicted_category(record: Mapping[str, Any]) -> Any:
    """Predicted category, preferring the raw model value over the mapped one."""
    prediction = record.get("prediction") or {}
    if "raw_category" in prediction:
        return prediction.get("raw_category")
    return prediction.get("category")


def _accuracy(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    correct = 0
    total = 0
    for record in records:
        if record.get("status") == "error":
            continue
        reference = record.get("reference") or {}
        predicted_category = _predicted_category(record)
        reference_category = reference.get("category")
        if reference_category is None:
            continue
        total += 1
        if predicted_category == reference_category:
            correct += 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else None,
    }


def _outcomes(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage / abstention / missing indicators over the ``_accuracy`` set.

    Each non-error record with a reference category lands in exactly one bucket:

    * ``answered`` - the model returned a non-null category;
    * ``abstained`` - the model explicitly returned ``category: null`` (the
      stored prediction carries ``abstained: true``);
    * ``missing`` - the product was not returned at all or there is no usable
      prediction for it. That is a model failure, never a principled
      abstention, so it must not inflate ``abstention_rate``.

    Only answered rows can be correct or wrong. Error records and rows without a
    reference category are excluded exactly as in ``_accuracy``, so ``total``
    matches ``overall.total`` and ``answered + abstained + missing == total``.
    """
    answered = 0
    abstained = 0
    missing = 0
    correct = 0
    for record in records:
        if record.get("status") == "error":
            continue
        reference = record.get("reference") or {}
        reference_category = reference.get("category")
        if reference_category is None:
            continue
        predicted_category = _predicted_category(record)
        if predicted_category is not None:
            answered += 1
            if predicted_category == reference_category:
                correct += 1
            continue
        prediction = record.get("prediction") or {}
        if prediction.get("abstained") is True:
            abstained += 1
        else:
            missing += 1
    total = answered + abstained + missing
    return {
        "total": total,
        "answered": answered,
        "abstained": abstained,
        "missing": missing,
        "correct": correct,
        "wrong": answered - correct,
        "coverage": (answered / total) if total else None,
        "accuracy_on_answered": (correct / answered) if answered else None,
        "abstention_rate": (abstained / total) if total else None,
        "missing_rate": (missing / total) if total else None,
    }


def _by_category_accuracy(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Accuracy per reference category, with the same rules as ``overall``.

    The caller passes the same HIGH+MEDIUM ground-truth records used for the
    headline metric, so error records are excluded and a missing prediction
    counts as incorrect in ``accuracy`` (and as ``missing`` or ``abstained`` in
    the coverage fields). Only categories actually present in those references
    are reported; empty rows would add noise without information. Each row also
    carries the answered/abstained/missing split so coverage can be read per
    category.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        reference_category = (record.get("reference") or {}).get("category")
        if reference_category is None:
            continue
        grouped.setdefault(str(reference_category), []).append(record)
    rows: dict[str, dict[str, Any]] = {}
    for category, items in sorted(grouped.items()):
        metrics = _accuracy(items)
        outcomes = _outcomes(items)
        # ``_outcomes`` recomputes ``correct``/``total`` with the same rules, so
        # these reconcile with ``overall`` and with the per-category totals.
        metrics["answered"] = outcomes["answered"]
        metrics["abstained"] = outcomes["abstained"]
        metrics["missing"] = outcomes["missing"]
        metrics["coverage"] = outcomes["coverage"]
        metrics["accuracy_on_answered"] = outcomes["accuracy_on_answered"]
        rows[category] = metrics
    return rows


def _misclassifications(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Human-readable wrong predictions for the report (input name included).

    Abstentions and missing products are excluded: a null prediction is either a
    coverage decision or a model failure, never a wrong answer, and must not be
    presented as a misclassification.
    """
    items: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") == "error":
            continue
        reference_category = (record.get("reference") or {}).get("category")
        if reference_category is None:
            continue
        predicted_category = _predicted_category(record)
        if predicted_category is None:
            continue
        if predicted_category == reference_category:
            continue
        input_name = record.get("input_name")
        items.append(
            {
                "input_name": input_name if isinstance(input_name, str) else None,
                "reference_category": reference_category,
                "predicted_category": predicted_category,
            }
        )
    items.sort(
        key=lambda item: (
            str(item.get("input_name") or ""),
            str(item.get("reference_category") or ""),
            str(item.get("predicted_category") or ""),
        )
    )
    return items


def _nutrition_metrics(
    records: Sequence[Mapping[str, Any]],
    dataset: EvalDataset,
) -> dict[str, Any]:
    comparable = [product for product in dataset.products if is_nutrition_comparable(product)]
    by_key = {
        str(record.get("product_key")): record
        for record in records
        if record.get("product_key")
    }
    fields: dict[str, dict[str, Any]] = {}
    for field_name in NUTRITION_FIELDS:
        differences: list[float] = []
        for product in comparable:
            record = by_key.get(product.product_key)
            if record is None:
                continue
            prediction = record.get("prediction") or {}
            nutrition = prediction.get("nutrition") or {}
            predicted = nutrition.get(field_name)
            reference = product.nutrition.get(field_name)
            if (
                isinstance(predicted, (int, float))
                and not isinstance(predicted, bool)
                and math.isfinite(float(predicted))
                and reference is not None
            ):
                differences.append(abs(float(predicted) - float(reference)))
        fields[field_name] = {
            "mae": (sum(differences) / len(differences)) if differences else None,
            "compared": len(differences),
            "coverage": (len(differences) / len(comparable)) if comparable else None,
        }
    return {"eligible": len(comparable), "fields": fields}


def _provider_summary(
    provider: str,
    model: str,
    records: Sequence[Mapping[str, Any]],
    dataset: EvalDataset,
) -> dict[str, Any]:
    # Group records by batch, keeping the first record per batch for batch-level
    # facts (latency, parse success, error kind, batch issues).
    batches: dict[str, Mapping[str, Any]] = {}
    for record in records:
        batch_id = str(record.get("batch_id") or "")
        if not batch_id:
            batch_id = (
                f"legacy-{record.get('timestamp')}-"
                f"{record.get('product_key')}"
            )
        batches.setdefault(batch_id, record)

    batch_list = list(batches.values())
    json_valid_batches = sum(1 for batch in batch_list if batch.get("json_valid") is True)
    error_counts = Counter(
        str(batch.get("error_kind"))
        for batch in batch_list
        if batch.get("error_kind")
    )
    latencies = [
        float(batch["latency_ms"])
        for batch in batch_list
        if isinstance(batch.get("latency_ms"), (int, float))
    ]
    latency_summary = {
        "count": len(latencies),
        "mean": (statistics.fmean(latencies) if latencies else None),
        "p50": (statistics.median(latencies) if latencies else None),
        "max": (max(latencies) if latencies else None),
    }

    statuses = Counter(str(record.get("status") or "unknown") for record in records)
    products_valid = statuses.get("ok", 0)
    products_flagged = statuses.get("flagged", 0)
    products_error = statuses.get("error", 0)

    if not batch_list:
        availability = "not_run"
    elif json_valid_batches == 0:
        availability = "unavailable"
    elif json_valid_batches < len(batch_list) or products_error:
        availability = "partial"
    else:
        availability = "ok"
    available = availability in {"ok", "partial"}

    category_by_tier: dict[str, dict[str, Any]] = {}
    for tier in CONFIDENCE_TIERS:
        tier_records = [
            record
            for record in records
            if (record.get("reference") or {}).get("confidence", "LOW") == tier
        ]
        metrics = _accuracy(tier_records)
        metrics["ground_truth"] = tier != "LOW"
        category_by_tier[tier] = metrics

    # The headline accuracy is measured only against verified references.
    # LOW-tier rows (AI_ESTIMATED and anything unmapped) are unverified and are
    # reported separately as agreement, never as accuracy.
    ground_truth_records = [
        record
        for record in records
        if (record.get("reference") or {}).get("confidence", "LOW") in ("HIGH", "MEDIUM")
    ]
    low_records = [
        record
        for record in records
        if (record.get("reference") or {}).get("confidence", "LOW") == "LOW"
    ]
    overall_accuracy = _accuracy(ground_truth_records)
    overall_accuracy["ground_truth"] = True
    # "reference_tiers": which reference rows this headline covers.
    overall_accuracy["reference_tiers"] = ["HIGH", "MEDIUM"]
    # The abstention-aware view of the exact same HIGH+MEDIUM record set:
    # ``overall`` stays the comparable headline, ``outcomes`` splits it into
    # answered / abstained / missing so coverage never hides a trade-off.
    overall_outcomes = _outcomes(ground_truth_records)
    low_agreement = _accuracy(low_records)
    low_agreement["ground_truth"] = False
    low_agreement["label"] = (
        "agreement with unverified LOW-confidence references (not accuracy)"
    )

    validation_counts: Counter[str] = Counter()
    batch_issue_counts: Counter[str] = Counter()
    for record in records:
        validation = record.get("validation") or {}
        for issue in validation.get("issues") or []:
            if isinstance(issue, Mapping) and issue.get("code"):
                validation_counts[str(issue["code"])] += 1
    for batch in batch_list:
        validation = batch.get("validation") or {}
        for issue in validation.get("batch_issues") or []:
            if isinstance(issue, Mapping) and issue.get("code"):
                batch_issue_counts[str(issue["code"])] += 1

    return {
        "provider": provider,
        "model": model,
        "available": available,
        "availability": availability,
        "batches": {
            "total": len(batch_list),
            "json_valid": json_valid_batches,
            "json_valid_rate": (
                json_valid_batches / len(batch_list) if batch_list else None
            ),
        },
        "products": {
            "expected": len(dataset.products),
            "reported": len(records),
            "valid": products_valid,
            "flagged": products_flagged,
            "error": products_error,
        },
        "error_counts": dict(sorted(error_counts.items())),
        "latency_ms": latency_summary,
        "category_accuracy": {
            "overall": overall_accuracy,
            "outcomes": overall_outcomes,
            "by_category": _by_category_accuracy(ground_truth_records),
            "reference_agreement_low": low_agreement,
            "by_confidence": category_by_tier,
        },
        "misclassifications": _misclassifications(ground_truth_records),
        "nutrition": _nutrition_metrics(records, dataset),
        "validation_issue_counts": dict(sorted(validation_counts.items())),
        "batch_issue_counts": dict(sorted(batch_issue_counts.items())),
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    dataset: EvalDataset,
    provider_models: Mapping[str, str],
    mode: str,
    abstain: bool = False,
) -> dict[str, Any]:
    dataset_keys = set(dataset.product_keys)
    latest = [
        record
        for record in latest_results(
            records,
            providers=tuple(provider_models),
            mode=mode,
            abstain=abstain,
        )
        if str(record.get("product_key") or "") in dataset_keys
    ]
    providers: dict[str, Any] = {}
    for provider in provider_models:
        model = provider_models[provider]
        selected = [
            record
            for record in latest
            if record.get("provider") == provider
            and str(record.get("model") or "") == model
        ]
        providers[provider] = _provider_summary(provider, model, selected, dataset)

    return {
        "generated_at": utc_timestamp(),
        "mode": mode,
        # Which prompt variant produced this summary. A forced-choice baseline
        # and an abstain run are separate result identities.
        "abstain": abstain,
        "dataset": {
            "source": dataset.source_path,
            "seed": dataset.seed,
            "requested_limit": dataset.requested_limit,
            "products": len(dataset.products),
            "by_category": dataset.category_counts(),
            "by_confidence": dataset.confidence_counts(),
            "nutrition_comparable": dataset.comparable_nutrition_count(),
        },
        "providers": providers,
        "notes": [
            "LOW confidence rows are reported for coverage but are not ground truth.",
            "Category accuracy (headline) counts only HIGH and MEDIUM reference "
            "rows: every non-error record is compared against the reference "
            "category; an invalid predicted category counts as incorrect and "
            "failed batches are excluded.",
            "A null predicted category is either an explicit abstention (it is "
            "never correct or incorrect and is excluded from accuracy on "
            "answered) or a missing product that the model never answered. "
            "Missing products are counted separately and do not inflate the "
            "abstention rate.",
            "Agreement with LOW confidence references is reported separately and "
            "is never presented as accuracy.",
            "Nutrition MAE uses only reference rows with HIGH/MEDIUM confidence, "
            "all of kcal/protein/carbs/fat present, and a food category.",
            "Free-tier models and quotas can change or be exhausted; all nutrition "
            "values are model estimates, not verified label data.",
        ],
    }


def _format_percent(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _format_accuracy(metrics: Mapping[str, Any]) -> str:
    total = metrics.get("total")
    correct = metrics.get("correct")
    if total in (None, 0):
        return "n/a (0 evaluated)"
    return f"{correct}/{total} ({_format_percent(metrics.get('accuracy'))})"


def _format_outcome_ratio(
    metrics: Mapping[str, Any],
    denominator_key: str,
    ratio_key: str,
) -> str:
    """``correct/answered`` with its ratio."""
    denominator = metrics.get(denominator_key)
    if (
        not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or not denominator
    ):
        return "n/a (0 evaluated)"
    return (
        f"{metrics.get('correct', 0)}/{denominator} "
        f"({_format_percent(metrics.get(ratio_key))})"
    )


def _format_coverage(metrics: Mapping[str, Any]) -> str:
    """``answered/total`` with the coverage ratio."""
    total = metrics.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or not total:
        return "n/a (0 evaluated)"
    return (
        f"{metrics.get('answered', 0)}/{total} "
        f"({_format_percent(metrics.get('coverage'))})"
    )


def _variant_label(abstain: object) -> str:
    if abstain is True:
        return "abstain-allowed (`--abstain`)"
    return "forced-choice (baseline)"


def _format_number(value: object, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _category_sort_key(
    item: tuple[str, Mapping[str, Any]],
) -> tuple[bool, float, str]:
    """Sort categories worst-first by accuracy on answered; unknown last."""
    category, metrics = item
    accuracy = metrics.get("accuracy_on_answered")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        return (True, 0.0, category)
    return (False, float(accuracy), category)


def render_report(summary: Mapping[str, Any]) -> str:
    providers = summary.get("providers") or {}
    ordered = [name for name in ("groq", "openrouter") if name in providers]
    dataset = summary.get("dataset") or {}

    lines: list[str] = [
        "# AI Enrichment Evaluation Report",
        "",
        f"Generated: {summary.get('generated_at', 'unknown')}",
        f"Mode: `{summary.get('mode', 'unknown')}`",
        f"Prompt variant: {_variant_label(summary.get('abstain'))}",
        (
            f"Dataset: {dataset.get('products', 0)} products from "
            f"`{dataset.get('source', 'unknown')}` (seed {dataset.get('seed')}, "
            f"limit {dataset.get('requested_limit')})"
        ),
        (
            "Confidence tiers: "
            + ", ".join(
                f"{tier}={count}"
                for tier, count in (dataset.get("by_confidence") or {}).items()
            )
        ),
        "",
        "## Head-to-head",
        "",
        "| Metric | " + " | ".join(ordered) + " |",
        "| --- | " + " | ".join("---" for _ in ordered) + " |",
    ]

    def row(label: str, getter: Any) -> None:
        lines.append(
            "| " + label + " | " + " | ".join(str(getter(providers[name])) for name in ordered) + " |"
        )

    row("Model", lambda p: f"`{p.get('model')}`")
    row("Availability", lambda p: str(p.get("availability")))
    row(
        "JSON-valid batches",
        lambda p: f"{p.get('batches', {}).get('json_valid', 0)}/"
        f"{p.get('batches', {}).get('total', 0)} "
        f"({_format_percent(p.get('batches', {}).get('json_valid_rate'))})",
    )
    row(
        "Products valid / flagged / error",
        lambda p: f"{p.get('products', {}).get('valid', 0)} / "
        f"{p.get('products', {}).get('flagged', 0)} / "
        f"{p.get('products', {}).get('error', 0)}",
    )
    row(
        "Category accuracy (HIGH+MEDIUM references)",
        lambda p: _format_accuracy(p.get("category_accuracy", {}).get("overall", {})),
    )
    row(
        "Accuracy HIGH tier",
        lambda p: _format_accuracy(
            p.get("category_accuracy", {}).get("by_confidence", {}).get("HIGH", {})
        ),
    )
    row(
        "Accuracy MEDIUM tier",
        lambda p: _format_accuracy(
            p.get("category_accuracy", {}).get("by_confidence", {}).get("MEDIUM", {})
        ),
    )
    row(
        "Agreement with unverified LOW references (not accuracy)",
        lambda p: _format_accuracy(
            p.get("category_accuracy", {}).get("reference_agreement_low", {})
        ),
    )
    row(
        "kcal MAE",
        lambda p: _format_number(
            (p.get("nutrition", {}).get("fields", {}).get("kcal_100g", {})).get("mae")
        ),
    )
    row(
        "Mean latency (ms)",
        lambda p: _format_number(p.get("latency_ms", {}).get("mean"), digits=1),
    )
    row(
        "Errors",
        lambda p: ", ".join(
            f"{kind}: {count}" for kind, count in (p.get("error_counts") or {}).items()
        )
        or "none",
    )

    lines.extend(["", "## Abstention", ""])
    lines.append(
        "Coverage is the share of evaluated products answered with a concrete "
        "category. An explicit abstention (null category) is never counted as "
        "correct or incorrect and is reported through the abstention rate. A "
        "product the model did not return at all is counted as missing, not as "
        "an abstention."
    )
    lines.append("")
    lines.append("| Metric | " + " | ".join(ordered) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in ordered) + " |")
    row(
        "Coverage (answered / total)",
        lambda p: _format_coverage(
            p.get("category_accuracy", {}).get("outcomes", {})
        ),
    )
    row(
        "Abstention rate",
        lambda p: _format_percent(
            p.get("category_accuracy", {}).get("outcomes", {}).get("abstention_rate")
        ),
    )
    row(
        "Missing rate",
        lambda p: _format_percent(
            p.get("category_accuracy", {}).get("outcomes", {}).get("missing_rate")
        ),
    )
    row(
        "Accuracy on answered",
        lambda p: _format_outcome_ratio(
            p.get("category_accuracy", {}).get("outcomes", {}),
            "answered",
            "accuracy_on_answered",
        ),
    )
    row(
        "Answered / abstained / missing / total",
        lambda p: (
            f"{p.get('category_accuracy', {}).get('outcomes', {}).get('answered', 0)}"
            " / "
            f"{p.get('category_accuracy', {}).get('outcomes', {}).get('abstained', 0)}"
            " / "
            f"{p.get('category_accuracy', {}).get('outcomes', {}).get('missing', 0)}"
            " / "
            f"{p.get('category_accuracy', {}).get('outcomes', {}).get('total', 0)}"
        ),
    )
    lines.extend(
        [
            "",
            "Forced-choice and abstain runs are not directly comparable: read "
            "coverage and the missing rate next to the accuracy figures. "
            "Allowing abstention trades coverage for accuracy on answered, so a "
            "higher accuracy on answered with lower coverage is not "
            "automatically a better model. `overall` accuracy (correct / total "
            "over the same HIGH+MEDIUM rows) is the number to compare between "
            "the forced and abstain runs; accuracy on answered is the quality "
            "of the answers the model actually committed to.",
        ]
    )

    lines.extend(["", "## Details", ""])
    for name in ordered:
        data = providers[name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Model: `{data.get('model')}`")
        lines.append(f"- Availability: {data.get('availability')}")
        nutrition = data.get("nutrition") or {}
        lines.append(f"- Nutrition-eligible reference rows: {nutrition.get('eligible', 0)}")
        for field_name in ("kcal_100g", "proteins_100g", "carbs_100g", "fat_100g"):
            metrics = (nutrition.get("fields") or {}).get(field_name, {})
            lines.append(
                f"- {field_name}: MAE {_format_number(metrics.get('mae'))}, "
                f"coverage {_format_percent(metrics.get('coverage'))} "
                f"({metrics.get('compared', 0)} compared)"
            )
        if data.get("validation_issue_counts"):
            lines.append(
                "- Validation issues: "
                + ", ".join(
                    f"{code}: {count}"
                    for code, count in data["validation_issue_counts"].items()
                )
            )
        if data.get("batch_issue_counts"):
            lines.append(
                "- Batch issues: "
                + ", ".join(
                    f"{code}: {count}"
                    for code, count in data["batch_issue_counts"].items()
                )
            )
        categories = (data.get("category_accuracy") or {}).get("by_category") or {}
        if categories:
            lines.append("")
            lines.append("#### Accuracy by category (HIGH+MEDIUM references)")
            lines.append("")
            lines.append(
                "| Category | Correct | Total | Accuracy | Answered | Abstained "
                "| Missing | Coverage | Accuracy on answered |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for category, metrics in sorted(categories.items(), key=_category_sort_key):
                lines.append(
                    f"| {category} | {metrics.get('correct', 0)} | "
                    f"{metrics.get('total', 0)} | "
                    f"{_format_percent(metrics.get('accuracy'))} | "
                    f"{metrics.get('answered', metrics.get('total', 0))} | "
                    f"{metrics.get('abstained', 0)} | "
                    f"{metrics.get('missing', 0)} | "
                    f"{_format_percent(metrics.get('coverage'))} | "
                    f"{_format_percent(metrics.get('accuracy_on_answered'))} |"
                )
        misclassified = data.get("misclassifications") or []
        lines.append("")
        lines.append(
            "#### Misclassified products (HIGH+MEDIUM references, answered only): "
            f"{len(misclassified)}"
        )
        if misclassified:
            lines.append("")
            for item in misclassified:
                name = item.get("input_name") or "(unknown product)"
                predicted = item.get("predicted_category")
                predicted_text = predicted if predicted not in (None, "") else "(none)"
                lines.append(
                    f"- {name} | {item.get('reference_category')} -> "
                    f"{predicted_text}"
                )
        lines.append("")

    lines.extend(["## Notes", ""])
    for note in summary.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
