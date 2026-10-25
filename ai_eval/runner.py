"""CLI runner for the offline-evaluable AI enrichment harness.

Examples:
    python -m ai_eval.runner --provider both --mode estimate --limit 5 \\
        --batch-size 5 --dry-run
    python -m ai_eval.runner --provider groq --mode estimate --limit 100
    python -m ai_eval.runner --list-free-models
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import NUTRITION_MODES, PROVIDER_NAMES
from .dataset import (
    DEFAULT_RANDOM_SEED,
    DatasetError,
    EvalDataset,
    EvalProduct,
    build_eval_dataset,
    chunked,
)
from .parsing import ParseError, ProductValidation, validate_response, parse_model_output
from .prompts import build_messages, render_prompt_preview
from .providers import (
    GROQ_DEFAULT_MODEL,
    GROQ_FREE_MODELS,
    FreeModelError,
    GroqClient,
    OpenRouterClient,
    ProviderClient,
    ProviderError,
    ProviderResponse,
    get_setting,
    list_groq_models,
    list_openrouter_free_models,
    validate_groq_model,
    validate_openrouter_model,
)
from .reporting import (
    append_results,
    build_summary,
    completed_keys,
    load_results,
    render_report,
    sanitize_value,
    utc_timestamp,
    write_json,
    write_text,
)
from .taxonomy import PROJECT_ROOT, Taxonomy, TaxonomyError, load_taxonomy


class ConfigError(RuntimeError):
    """Raised for invalid CLI model/key configuration."""


# Every generated artefact must stay below this root (see --output-dir).
SCRATCH_ROOT = (PROJECT_ROOT / "scratch").resolve()


def resolve_output_dir(raw: str | Path) -> Path:
    """Resolve ``--output-dir`` and confine it to ``scratch/``.

    Relative paths are resolved against the project root; absolute paths and
    ``..`` escapes are rejected unless the final (symlink-resolved) path is
    inside ``<project>/scratch``.
    """
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(SCRATCH_ROOT)
    except ValueError as exc:
        raise ConfigError(
            f"--output-dir {str(raw)!r} resolves to {resolved}, which is outside "
            f"the allowed output root {SCRATCH_ROOT}. All harness outputs must "
            "stay under scratch/."
        ) from exc
    return resolved


def _ensure_confined_artifact(path: Path) -> Path:
    """Validate that writing through ``path`` cannot escape ``scratch/``.

    Confining the output *directory* is not enough: a pre-existing
    ``results.jsonl``, ``summary.json``, or ``report.md`` could be a symlink or
    Windows junction pointing outside. Truncating/overwriting through it would
    clobber an external file, so the real target is resolved and checked before
    every write.
    """
    try:
        real = Path(os.path.realpath(path))
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"Cannot resolve output artifact {path!s}: {exc}"
        ) from exc
    try:
        real.relative_to(SCRATCH_ROOT)
    except ValueError as exc:
        raise ConfigError(
            f"Output artifact {path!s} resolves to {real!s}, which is outside "
            f"the allowed output root {SCRATCH_ROOT!s}. Refusing to write "
            "through a symlink or junction that escapes scratch/."
        ) from exc
    return path


def _confine_artifact(output_dir: Path, name: str) -> Path:
    return _ensure_confined_artifact(output_dir / name)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai_eval.runner",
        description=(
            "Evaluate free-tier Groq/OpenRouter models on unresolved grocery "
            "products. Read-only with respect to the production pipeline."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("groq", "openrouter", "both"),
        default="both",
        help="Provider(s) to run sequentially (default: both).",
    )
    parser.add_argument(
        "--mode",
        choices=NUTRITION_MODES,
        default="estimate",
        help="Nutrition mode: none (category only) or estimate (default).",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=100,
        help="Maximum number of stratified products (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=10,
        help="Products per model request (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model ID for a single provider. With --provider both, use "
            "GROQ_MODEL / OPENROUTER_MODEL instead."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dataset and print a prompt preview without any network call.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip products already having a valid result for the same provider/model.",
    )
    parser.add_argument(
        "--abstain",
        action="store_true",
        help=(
            "Opt-in variant: allow the model to answer category=null with a short "
            "abstain_reason instead of forcing a guess. Off by default; the "
            "forced-choice baseline is unchanged."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between live requests (default: 1.5).",
    )
    parser.add_argument(
        "--output-dir",
        default="scratch/ai_eval",
        help="Directory for results.jsonl/summary.json/report.md; must resolve "
        "inside scratch/ (default: scratch/ai_eval).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Deterministic dataset seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--allow-nonfree-model",
        action="store_true",
        help=(
            "Groq only: allow a model outside the documented free-tier list. "
            "Never enables OpenRouter paid models."
        ),
    )
    parser.add_argument(
        "--list-free-models",
        action="store_true",
        help="Live: list OpenRouter models with zero prompt/completion pricing.",
    )
    parser.add_argument(
        "--list-groq-models",
        action="store_true",
        help="Live: list Groq models available to the configured key.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _selected_providers(args: argparse.Namespace) -> tuple[str, ...]:
    return PROVIDER_NAMES if args.provider == "both" else (args.provider,)


def resolve_models(
    args: argparse.Namespace,
    env_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Resolve provider models: ``--model`` > environment > ``.env`` > default."""
    if args.model and args.provider == "both":
        raise ConfigError(
            "--model cannot be used with --provider both because Groq and "
            "OpenRouter require different IDs. Set GROQ_MODEL and "
            "OPENROUTER_MODEL instead."
        )
    models: dict[str, str | None] = {}
    for provider in _selected_providers(args):
        if args.model:
            models[provider] = args.model
        elif provider == "groq":
            models[provider] = get_setting(
                "GROQ_MODEL",
                GROQ_DEFAULT_MODEL,
                env_path=env_path,
                environ=environ,
            )
        else:
            models[provider] = get_setting(
                "OPENROUTER_MODEL", env_path=env_path, environ=environ
            )
    return models


def _model_plan_line(provider: str, model: str | None, allow_nonfree: bool) -> str:
    if provider == "groq":
        try:
            validate_groq_model(model, allow_nonfree=allow_nonfree)
        except FreeModelError as exc:
            return f"{provider}: model={model or '(default)'} - NOT ALLOWED: {exc}"
        if model not in GROQ_FREE_MODELS:
            return f"{provider}: model={model} (NON-FREE opt-in via --allow-nonfree-model)"
        return f"{provider}: model={model} (documented free-tier allowlist)"
    try:
        validate_openrouter_model(model)
    except FreeModelError:
        return (
            f"{provider}: model={model or '(not configured)'} - a free model is "
            "required for live runs; use --list-free-models."
        )
    return f"{provider}: model={model} (free model)"


def print_dry_run(
    args: argparse.Namespace,
    taxonomy: Taxonomy,
    dataset: EvalDataset,
    models: Mapping[str, str | None],
) -> None:
    providers = _selected_providers(args)
    batch_count = (len(dataset.products) + args.batch_size - 1) // args.batch_size
    print("AI enrichment evaluation - DRY RUN")
    print("==================================")
    print(f"Dataset: {len(dataset.products)} products from {dataset.source_path}")
    print(f"Seed: {dataset.seed}; requested limit: {args.limit}")
    print(f"Mode: {args.mode}; batch size: {args.batch_size}; batches: {batch_count}")
    print(
        "Prompt variant: "
        + (
            "abstain-allowed (--abstain: category may be null with abstain_reason)"
            if args.abstain
            else "forced-choice (baseline: every product needs a category)"
        )
        + f"; abstain={args.abstain}"
    )
    print("Category coverage:")
    for category in taxonomy.categories:
        count = dataset.category_counts().get(category, 0)
        print(f"  - {category}: {count}")
    print("Confidence tiers: " + ", ".join(
        f"{tier}={count}" for tier, count in dataset.confidence_counts().items()
    ))
    print("Providers:")
    for provider in providers:
        print("  " + _model_plan_line(provider, models.get(provider), args.allow_nonfree_model))
    print("No network calls were made and no API keys were required.")
    print(f"No files were written (live output would go to {args.output_dir}).")
    print()
    print("--- Prompt preview (first batch) ---")
    first_batch = list(chunked(list(dataset.products), args.batch_size))[0]
    messages = build_messages(
        taxonomy,
        [product.to_model_payload() for product in first_batch],
        args.mode,
        allow_abstain=args.abstain,
    )
    print(render_prompt_preview(messages))


def _make_clients(
    args: argparse.Namespace,
    models: Mapping[str, str | None],
) -> list[ProviderClient]:
    clients: list[ProviderClient] = []
    for provider in _selected_providers(args):
        model = models.get(provider)
        if provider == "groq":
            client: ProviderClient = GroqClient(
                model=model,
                allow_nonfree_model=args.allow_nonfree_model,
            )
        else:
            client = OpenRouterClient(model=model)
        if not client.api_key:
            raise ConfigError(
                f"{provider}: {client.key_env} is not configured. Add it to .env "
                "(see .env.example) or export it in the environment."
            )
        clients.append(client)
    return clients


def _base_record(
    client: ProviderClient,
    product: EvalProduct,
    *,
    batch_id: str,
    batch_index: int,
    batch_size: int,
    timestamp: str,
    latency_ms: float,
    attempts: int,
    json_valid: bool,
    abstain: bool = False,
) -> dict[str, Any]:
    return {
        "provider": client.provider,
        "model": client.model,
        "product_key": product.product_key,
        "input_name": product.sample_name,
        "reference": product.reference_snapshot(),
        "prediction": None,
        "validation": {"valid": False, "issues": [], "batch_issues": []},
        "status": "error",
        "latency_ms": round(float(latency_ms), 2),
        "attempts": attempts,
        "errors": [],
        "error_kind": None,
        "batch_id": batch_id,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "json_valid": json_valid,
        # Which prompt variant produced this record. It is part of the result
        # identity: resume and summaries never mix the two variants.
        "abstain": bool(abstain),
        "timestamp": timestamp,
    }


def _error_records(
    client: ProviderClient,
    batch: Sequence[EvalProduct],
    response: ProviderResponse,
    *,
    batch_id: str,
    batch_index: int,
    timestamp: str,
    error_kind: str,
    error_message: str,
    json_valid: bool,
    abstain: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for product in batch:
        record = _base_record(
            client,
            product,
            batch_id=batch_id,
            batch_index=batch_index,
            batch_size=len(batch),
            timestamp=timestamp,
            latency_ms=response.latency_ms,
            attempts=response.attempts,
            json_valid=json_valid,
            abstain=abstain,
        )
        record["errors"] = [error_message]
        record["error_kind"] = error_kind
        records.append(record)
    return records


def _batch_records(
    client: ProviderClient,
    batch: Sequence[EvalProduct],
    response: ProviderResponse,
    *,
    batch_id: str,
    batch_index: int,
    timestamp: str,
    taxonomy: Taxonomy,
    mode: str,
    abstain: bool = False,
) -> list[dict[str, Any]]:
    if response.error:
        return _error_records(
            client,
            batch,
            response,
            batch_id=batch_id,
            batch_index=batch_index,
            timestamp=timestamp,
            error_kind=response.error_kind or "provider_error",
            error_message=response.error,
            json_valid=False,
            abstain=abstain,
        )

    try:
        payload = parse_model_output(response.content or "")
        validation = validate_response(
            payload,
            [product.product_key for product in batch],
            taxonomy,
            mode,
            # The validator must know which prompt variant produced the reply:
            # a null category is only a clean answer when --abstain allowed it.
            allow_abstain=abstain,
        )
    except ParseError as exc:
        return _error_records(
            client,
            batch,
            response,
            batch_id=batch_id,
            batch_index=batch_index,
            timestamp=timestamp,
            error_kind="parse_error",
            error_message=str(exc),
            json_valid=False,
            abstain=abstain,
        )

    batch_issues = [
        {"code": issue.code, "message": issue.message}
        for issue in validation.batch_issues
    ]
    records: list[dict[str, Any]] = []
    for product in batch:
        product_validation = validation.products.get(product.product_key)
        if product_validation is None:
            product_validation = ProductValidation(
                product_key=product.product_key, found=False
            )
        issues = [
            {"code": issue.code, "message": issue.message}
            for issue in product_validation.issues
        ]
        record = _base_record(
            client,
            product,
            batch_id=batch_id,
            batch_index=batch_index,
            batch_size=len(batch),
            timestamp=timestamp,
            latency_ms=response.latency_ms,
            attempts=response.attempts,
            json_valid=True,
            abstain=abstain,
        )
        record["prediction"] = {
            "category": product_validation.category,
            "abstained": product_validation.abstained,
            "abstain_reason": product_validation.abstain_reason,
            "raw_category": product_validation.raw_category,
            "macro_category": product_validation.macro_category,
            "model_macro_category": product_validation.model_macro,
            "nutrition": product_validation.nutrition,
        }
        record["validation"] = {
            "valid": product_validation.valid and not batch_issues,
            "issues": issues,
            "batch_issues": batch_issues,
        }
        record["status"] = "ok" if record["validation"]["valid"] else "flagged"
        records.append(record)
    return records


def run_live(
    args: argparse.Namespace,
    taxonomy: Taxonomy,
    dataset: EvalDataset,
    clients: Sequence[ProviderClient],
    models: Mapping[str, str | None],
) -> int:
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Check every artifact BEFORE anything is written. Without --resume the
    # results file is truncated immediately, so an escaping symlink/junction
    # must be rejected first (ConfigError -> exit 2 in main()).
    results_path = _confine_artifact(output_dir, "results.jsonl")
    summary_path = _confine_artifact(output_dir, "summary.json")
    report_path = _confine_artifact(output_dir, "report.md")
    if not args.resume:
        _ensure_confined_artifact(results_path)
        results_path.write_text("", encoding="utf-8")

    existing = load_results(results_path) if args.resume else []
    all_records: list[dict[str, Any]] = list(existing)
    secrets = tuple(client.api_key for client in clients if client.api_key)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    provider_models = {
        client.provider: str(client.model) for client in clients
    }

    for client in clients:
        done = (
            completed_keys(
                existing,
                client.provider,
                str(client.model),
                mode=args.mode,
                abstain=args.abstain,
            )
            if args.resume
            else set()
        )
        pending = [
            product for product in dataset.products if product.product_key not in done
        ]
        print(
            f"{client.provider}: model={client.model}; "
            f"{len(pending)} pending of {len(dataset.products)} products."
        )
        if not pending:
            continue
        batches = list(chunked(pending, args.batch_size))
        for batch_index, batch in enumerate(batches, start=1):
            batch_id = f"{run_id}-{client.provider}-{batch_index:03d}"
            timestamp = utc_timestamp()
            messages = build_messages(
                taxonomy,
                [product.to_model_payload() for product in batch],
                args.mode,
                allow_abstain=args.abstain,
            )
            response = client.chat(messages)
            records = _batch_records(
                client,
                batch,
                response,
                batch_id=batch_id,
                batch_index=batch_index,
                timestamp=timestamp,
                taxonomy=taxonomy,
                mode=args.mode,
                abstain=args.abstain,
            )
            for record in records:
                record["mode"] = args.mode
            # Re-check right before the append: the append is the second
            # write path through results.jsonl.
            _ensure_confined_artifact(results_path)
            append_results(results_path, records, secrets)
            all_records.extend(records)
            if response.error:
                print(
                    f"  batch {batch_index}/{len(batches)} failed "
                    f"({response.error_kind}); continuing."
                )
            else:
                valid_count = sum(1 for record in records if record["status"] == "ok")
                print(
                    f"  batch {batch_index}/{len(batches)}: "
                    f"{valid_count}/{len(records)} valid."
                )
            if args.delay > 0 and batch_index < len(batches):
                time.sleep(args.delay)

    summary = build_summary(
        all_records, dataset, provider_models, args.mode, abstain=args.abstain
    )
    _ensure_confined_artifact(summary_path)
    write_json(summary_path, sanitize_value(summary, secrets))
    if len(clients) == 2:
        _ensure_confined_artifact(report_path)
        write_text(
            report_path,
            sanitize_value(render_report(summary), secrets),
        )

    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    if len(clients) == 2:
        print(f"Report: {report_path}")
    for provider, data in (summary.get("providers") or {}).items():
        print(
            f"  {provider}: availability={data.get('availability')} "
            f"batches={data.get('batches', {}).get('json_valid', 0)}/"
            f"{data.get('batches', {}).get('total', 0)} json-valid, "
            f"products valid/flagged/error="
            f"{data.get('products', {}).get('valid', 0)}/"
            f"{data.get('products', {}).get('flagged', 0)}/"
            f"{data.get('products', {}).get('error', 0)}"
        )
    return 0


def _list_models(args: argparse.Namespace) -> int:
    exit_code = 0
    if args.list_free_models:
        try:
            models = list_openrouter_free_models()
        except ProviderError as exc:
            print(f"OpenRouter model listing failed: {exc}")
            exit_code = 2
        else:
            print(f"OpenRouter free models (zero prompt/completion price): {len(models)}")
            for model in models:
                marker = (
                    "[usable]"
                    if model.get("usable")
                    else "[not :free - rejected by guard]"
                )
                print(
                    f"  {model.get('id')} {marker}  "
                    f"({model.get('name') or 'unnamed'})"
                )
            print("Use one of these IDs with --model or set OPENROUTER_MODEL.")
    if args.list_groq_models:
        try:
            model_ids = list_groq_models()
        except ProviderError as exc:
            print(f"Groq model listing failed: {exc}")
            exit_code = 2
        else:
            print(f"Groq models visible to the configured key: {len(model_ids)}")
            for model_id in model_ids:
                marker = " [free-tier allowlist]" if model_id in GROQ_FREE_MODELS else ""
                print(f"  {model_id}{marker}")
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # --dry-run must never touch the network: the listing flags perform a
    # public GET (OpenRouter) or an authenticated GET (Groq).
    if args.dry_run and (args.list_free_models or args.list_groq_models):
        print(
            "Configuration error: --dry-run cannot be combined with "
            "--list-free-models or --list-groq-models because the listing "
            "flags perform live network calls. Run the listing flag on its own."
        )
        return 2

    if args.list_free_models or args.list_groq_models:
        return _list_models(args)

    try:
        output_dir = resolve_output_dir(args.output_dir)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2
    args.output_dir = str(output_dir)

    try:
        taxonomy = load_taxonomy()
    except TaxonomyError as exc:
        print(f"Taxonomy error: {exc}")
        return 2

    try:
        dataset = build_eval_dataset(
            limit=args.limit,
            seed=args.seed,
            taxonomy=taxonomy,
        )
    except DatasetError as exc:
        print(f"Dataset error: {exc}")
        return 2
    if not dataset.products:
        print("Dataset error: no products were selected.")
        return 2

    try:
        models = resolve_models(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2

    if args.dry_run:
        print_dry_run(args, taxonomy, dataset, models)
        return 0

    try:
        clients = _make_clients(args, models)
    except (ProviderError, ConfigError) as exc:
        print(f"Configuration error: {exc}")
        return 2

    print(
        f"Live evaluation: {len(dataset.products)} products, mode={args.mode}, "
        f"variant={'abstain' if args.abstain else 'forced-choice'}."
    )
    for client in clients:
        print(f"  {client.provider}: {client.model}")
    try:
        return run_live(args, taxonomy, dataset, clients, models)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
