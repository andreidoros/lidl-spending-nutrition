"""Robust parsing and per-product validation of model responses."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .constants import (
    ALCOHOL_CATEGORY,
    ATWATER_ABS_TOLERANCE_KCAL,
    ATWATER_REL_TOLERANCE,
    MAX_ABSTAIN_REASON_LENGTH,
    MAX_MACRO_SUM_G,
    NON_FOOD_CATEGORY,
    NUTRITION_FIELDS,
    NUTRITION_MODES,
    NUTRITION_RANGES,
    REQUIRED_NUTRITION_FIELDS,
    SATURATED_FAT_TOLERANCE_G,
    SUGARS_TOLERANCE_G,
)
from .taxonomy import Taxonomy


class ParseError(ValueError):
    """Raised when a model response is not the required JSON shape."""


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass
class ProductValidation:
    product_key: str
    found: bool = False
    category: str | None = None
    raw_category: object = None
    macro_category: str | None = None
    model_macro: str | None = None
    nutrition: dict[str, float | None] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.found and not self.issues

    @property
    def issue_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


@dataclass
class BatchValidation:
    products: dict[str, ProductValidation] = field(default_factory=dict)
    batch_issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.batch_issues and all(
            product.valid for product in self.products.values()
        )

    @property
    def missing_keys(self) -> list[str]:
        return [key for key, product in self.products.items() if not product.found]


_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


class _DuplicateMemberError(ValueError):
    """Internal marker for a repeated key inside one JSON object."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate JSON object member {key!r}")


def _reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Refuse ambiguous replies instead of letting the last value win.

    ``json.loads`` keeps the last of duplicate ``products`` / ``product_key`` /
    ``category`` members, which could score an ambiguous reply as a success and
    then be skipped by resume.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateMemberError(str(key))
        result[key] = value
    return result


def _duplicate_member_error(exc: _DuplicateMemberError) -> ParseError:
    return ParseError(
        f"Model response repeats the JSON member {exc.key!r} inside one object; "
        "the reply is ambiguous and cannot be scored."
    )



def strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence if one is present."""
    if not isinstance(text, str):
        raise ParseError("Model response content must be a string.")
    cleaned = text.strip().lstrip("\ufeff")
    match = _FENCE_RE.match(cleaned)
    if match:
        return match.group(1).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return cleaned


def parse_model_output(text: str) -> dict[str, Any]:
    """Parse JSON from raw model text, tolerating code fences/prose wrappers.

    Structural requirements: a JSON object with a ``products`` array.
    """
    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ParseError("Model response was empty.")

    try:
        data = json.loads(cleaned, object_pairs_hook=_reject_duplicate_members)
    except _DuplicateMemberError as exc:
        raise _duplicate_member_error(exc) from exc
    except json.JSONDecodeError as first_error:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ParseError(
                f"Model response is not valid JSON: {first_error.msg} "
                f"(line {first_error.lineno}, column {first_error.colno})."
            ) from first_error
        try:
            data = json.loads(
                cleaned[start : end + 1],
                object_pairs_hook=_reject_duplicate_members,
            )
        except _DuplicateMemberError as exc:
            raise _duplicate_member_error(exc) from exc
        except json.JSONDecodeError as exc:
            raise ParseError(
                f"Model response is not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})."
            ) from exc

    if not isinstance(data, dict):
        raise ParseError(
            "Model response must be a JSON object with a 'products' array."
        )
    if "products" not in data:
        raise ParseError("Model response JSON is missing the 'products' key.")
    if not isinstance(data["products"], list):
        raise ParseError("Model response 'products' value must be an array.")
    return data


def _coerce_requested_keys(requested: Iterable[object]) -> list[str]:
    keys: list[str] = []
    for item in requested:
        if isinstance(item, str):
            keys.append(item)
        elif hasattr(item, "product_key"):
            keys.append(str(item.product_key))  # type: ignore[attr-defined]
        elif isinstance(item, Mapping):
            keys.append(str(item.get("product_key", "")))
        else:
            raise TypeError(f"Unsupported requested product reference: {item!r}")
    return keys


def _finite_number(value: object) -> float | None:
    """Convert a JSON number to a finite float, else ``None``.

    A JSON integer with hundreds of digits passes the ``int`` check but raises
    ``OverflowError`` from ``float()``; that must become a per-product issue,
    never an exception that aborts the remaining batches.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _number(
    field_name: str,
    value: object,
    issues: list[ValidationIssue],
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(
            ValidationIssue(
                "nutrition_not_numeric",
                f"{field_name} must be a number or null; got {value!r}.",
            )
        )
        return None
    number = _finite_number(value)
    if number is None:
        issues.append(
            ValidationIssue(
                "nutrition_not_numeric",
                f"{field_name} must be a finite number or null; got {value!r}.",
            )
        )
        return None
    lower, upper = NUTRITION_RANGES[field_name]
    if not (lower <= number <= upper):
        issues.append(
            ValidationIssue(
                "nutrition_out_of_range",
                f"{field_name}={number:g} is outside the allowed range "
                f"[{lower:g}, {upper:g}].",
            )
        )
    return number


def _validate_nutrition(
    raw_product: Mapping[str, Any],
    category: str | None,
    mode: str,
    issues: list[ValidationIssue],
) -> dict[str, float | None]:
    if mode == "none":
        provided = [
            field_name
            for field_name in NUTRITION_FIELDS
            if raw_product.get(field_name) is not None
        ]
        if provided:
            issues.append(
                ValidationIssue(
                    "nutrition_provided_in_none_mode",
                    "Nutrition mode is 'none', but values were provided for: "
                    + ", ".join(provided)
                    + ".",
                )
            )
        return {field_name: None for field_name in NUTRITION_FIELDS}

    nutrition: dict[str, float | None] = {
        field_name: _number(field_name, raw_product.get(field_name), issues)
        for field_name in NUTRITION_FIELDS
    }

    if category == NON_FOOD_CATEGORY:
        provided = [
            field_name
            for field_name in NUTRITION_FIELDS
            if nutrition[field_name] is not None
        ]
        if provided:
            issues.append(
                ValidationIssue(
                    "non_food_nutrition",
                    "Non-Food products must have null/omitted nutrition; values "
                    "were provided for: " + ", ".join(provided) + ".",
                )
            )
        return nutrition

    present = {
        field_name: value
        for field_name, value in nutrition.items()
        if value is not None
    }
    macro_sum = sum(
        present.get(field_name, 0.0)
        for field_name in (
            "proteins_100g",
            "carbs_100g",
            "fat_100g",
            "fiber_100g",
            "salt_100g",
        )
    )
    if macro_sum > MAX_MACRO_SUM_G:
        issues.append(
            ValidationIssue(
                "macro_sum_exceeds_105",
                f"Sum of protein/carbs/fat/fiber/salt is {macro_sum:g} g per "
                f"100 g, above {MAX_MACRO_SUM_G:g} g.",
            )
        )

    sugars = present.get("sugars_100g")
    carbs = present.get("carbs_100g")
    if sugars is not None and carbs is not None and sugars > carbs + SUGARS_TOLERANCE_G:
        issues.append(
            ValidationIssue(
                "sugars_exceeds_carbs",
                f"sugars_100g={sugars:g} exceeds carbs_100g={carbs:g} by more "
                f"than {SUGARS_TOLERANCE_G:g} g.",
            )
        )

    saturated = present.get("saturated_fat_100g")
    fat = present.get("fat_100g")
    if (
        saturated is not None
        and fat is not None
        and saturated > fat + SATURATED_FAT_TOLERANCE_G
    ):
        issues.append(
            ValidationIssue(
                "saturated_fat_exceeds_fat",
                f"saturated_fat_100g={saturated:g} exceeds fat_100g={fat:g} by "
                f"more than {SATURATED_FAT_TOLERANCE_G:g} g.",
            )
        )

    if all(nutrition.get(field_name) is not None for field_name in REQUIRED_NUTRITION_FIELDS):
        kcal = nutrition["kcal_100g"]
        proteins = nutrition["proteins_100g"]
        carbs_value = nutrition["carbs_100g"]
        fat_value = nutrition["fat_100g"]
        fiber = nutrition.get("fiber_100g") or 0.0
        atwater = (
            proteins * 4.0
            + carbs_value * 4.0
            + fat_value * 9.0
            + fiber * 2.0
        )
        tolerance = max(ATWATER_ABS_TOLERANCE_KCAL, kcal * ATWATER_REL_TOLERANCE)
        if (
            category != ALCOHOL_CATEGORY
            and category != NON_FOOD_CATEGORY
            and abs(kcal - atwater) > tolerance
        ):
            issues.append(
                ValidationIssue(
                    "atwater_mismatch",
                    f"kcal_100g={kcal:g} is not consistent with Atwater "
                    f"estimate {atwater:.1f} kcal (tolerance {tolerance:.1f}).",
                )
            )

    return nutrition


def _validate_product(
    product_key: str,
    raw_product: Mapping[str, Any],
    taxonomy: Taxonomy,
    mode: str,
    allow_abstain: bool = False,
) -> ProductValidation:
    issues: list[ValidationIssue] = []
    abstained = False
    if "category" not in raw_product:
        # A missing key is an abstention, never a silent success: the model did
        # not answer, so the record must stay flagged.
        abstained = True
        raw_category: object = None
        issues.append(
            ValidationIssue(
                "missing_category",
                "Product object has no 'category' key; a missing category is "
                "an abstention and cannot be scored as a valid answer.",
            )
        )
    else:
        raw_category = raw_product.get("category")
        if raw_category is None:
            # An explicit JSON null is a deliberate abstention only when the
            # prompt allowed one. In a forced-choice run it is a protocol
            # violation: accepting it as a clean answer would silently turn the
            # baseline into an abstain run and corrupt the A/B comparison.
            abstained = True
            if not allow_abstain:
                issues.append(
                    ValidationIssue(
                        "abstention_not_allowed",
                        "category is null but the forced-choice prompt does not "
                        "allow abstention; every product must get a taxonomy "
                        "category. This record is flagged, not scored as a "
                        "valid answer.",
                    )
                )

    category: str | None = None
    macro: str | None = None
    if not abstained:
        if taxonomy.is_category(raw_category):
            category = str(raw_category)
            macro = taxonomy.macro_for(category)
        else:
            issues.append(
                ValidationIssue(
                    "invalid_category",
                    f"category {raw_category!r} is not an exact canonical "
                    "taxonomy category.",
                )
            )

    model_macro = raw_product.get("macro_category")
    if model_macro is not None and category is not None:
        if model_macro != macro:
            issues.append(
                ValidationIssue(
                    "macro_mismatch",
                    f"Model macro_category {model_macro!r} disagrees with the "
                    f"taxonomy-derived macro {macro!r} for category "
                    f"{category!r}; the derived macro is used.",
                )
            )

    raw_reason = raw_product.get("abstain_reason")
    abstain_reason: str | None = None
    if abstained:
        # An abstention carries no nutrition and no macro: any value would be
        # an invented answer for a product the model declined to classify.
        provided = [
            field_name
            for field_name in NUTRITION_FIELDS
            if raw_product.get(field_name) is not None
        ]
        if provided:
            issues.append(
                ValidationIssue(
                    "nutrition_on_abstention",
                    "An abstained product must not carry nutrition values; "
                    "values were provided for: " + ", ".join(provided) + ".",
                )
            )
        nutrition: dict[str, float | None] = {
            field_name: None for field_name in NUTRITION_FIELDS
        }
        if isinstance(raw_reason, str) and raw_reason.strip():
            abstain_reason = raw_reason
        if abstain_reason is None:
            issues.append(
                ValidationIssue(
                    "missing_abstain_reason",
                    "category is null but abstain_reason is missing or empty; "
                    "a short reason is required for an abstention.",
                )
            )
        else:
            if len(abstain_reason) > MAX_ABSTAIN_REASON_LENGTH:
                issues.append(
                    ValidationIssue(
                        "missing_abstain_reason",
                        f"abstain_reason is {len(abstain_reason)} characters long; "
                        f"a short reason of at most {MAX_ABSTAIN_REASON_LENGTH} "
                        "characters is required.",
                    )
                )
            # The stored value is bounded: an over-long reason is flagged AND
            # persisted truncated, so results.jsonl never carries unbounded
            # model text.
            abstain_reason = abstain_reason[:MAX_ABSTAIN_REASON_LENGTH]
    else:
        nutrition = _validate_nutrition(raw_product, category, mode, issues)
        if raw_reason is not None:
            issues.append(
                ValidationIssue(
                    "abstain_reason_on_category",
                    f"abstain_reason {raw_reason!r} was provided for category "
                    f"{raw_category!r}; abstain_reason must be null when "
                    "category is not null.",
                )
            )

    return ProductValidation(
        product_key=product_key,
        found=True,
        category=category,
        raw_category=raw_category,
        macro_category=macro,
        model_macro=model_macro if isinstance(model_macro, str) else None,
        nutrition=nutrition,
        issues=issues,
        abstained=abstained,
        abstain_reason=abstain_reason,
    )


def validate_response(
    payload: Mapping[str, Any],
    requested: Sequence[object],
    taxonomy: Taxonomy,
    mode: str = "estimate",
    *,
    allow_abstain: bool = False,
) -> BatchValidation:
    """Validate a parsed payload against the exact requested product keys.

    ``allow_abstain`` mirrors the prompt variant: only an abstain run accepts an
    explicit ``category: null`` as a clean answer. In a forced-choice run the
    same reply is a protocol violation (``abstention_not_allowed``) and stays
    flagged, so the baseline can never silently become an abstain run.
    """
    if mode not in NUTRITION_MODES:
        raise ValueError(f"Unknown nutrition mode {mode!r}; expected one of {NUTRITION_MODES}.")
    requested_keys = _coerce_requested_keys(requested)
    requested_set = set(requested_keys)
    batch_issues: list[ValidationIssue] = []

    if not isinstance(payload, Mapping) or not isinstance(payload.get("products"), list):
        raise ParseError("Parsed payload must be an object with a 'products' array.")

    parsed: dict[str, Mapping[str, Any]] = {}
    duplicate_keys: set[str] = set()
    for index, item in enumerate(payload["products"]):
        if not isinstance(item, Mapping):
            batch_issues.append(
                ValidationIssue(
                    "invalid_product_entry",
                    f"products[{index}] is not a JSON object.",
                )
            )
            continue
        raw_key = item.get("product_key")
        if not isinstance(raw_key, str) or not raw_key:
            batch_issues.append(
                ValidationIssue(
                    "invalid_product_key",
                    f"products[{index}] has a missing or non-string product_key.",
                )
            )
            continue
        if raw_key in parsed:
            duplicate_keys.add(raw_key)
            continue
        parsed[raw_key] = item

    for key in sorted(duplicate_keys):
        batch_issues.append(
            ValidationIssue(
                "duplicate_product_key",
                f"product_key {key!r} appears more than once; the first entry "
                "was used.",
            )
        )

    extra_keys = sorted(set(parsed) - requested_set)
    for key in extra_keys:
        batch_issues.append(
            ValidationIssue(
                "extra_product",
                f"Model returned unrequested product_key {key!r}.",
            )
        )

    products: dict[str, ProductValidation] = {}
    for key in requested_keys:
        if key in products:
            continue
        raw_product = parsed.get(key)
        if raw_product is None:
            products[key] = ProductValidation(
                product_key=key,
                found=False,
                issues=[
                    ValidationIssue(
                        "missing_product",
                        f"Model response did not include requested product_key {key!r}.",
                    )
                ],
            )
            continue
        validation = _validate_product(
            key, raw_product, taxonomy, mode, allow_abstain=allow_abstain
        )
        if key in duplicate_keys:
            validation.issues.append(
                ValidationIssue(
                    "duplicate_product_key",
                    f"product_key {key!r} was returned more than once.",
                )
            )
        products[key] = validation

    return BatchValidation(products=products, batch_issues=batch_issues)


def parse_and_validate(
    text: str,
    requested: Sequence[object],
    taxonomy: Taxonomy,
    mode: str = "estimate",
    *,
    allow_abstain: bool = False,
) -> BatchValidation:
    return validate_response(
        parse_model_output(text),
        requested,
        taxonomy,
        mode,
        allow_abstain=allow_abstain,
    )
