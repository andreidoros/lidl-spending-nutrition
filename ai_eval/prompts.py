"""Prompt construction for the enrichment evaluation.

Prompts are generated from the runtime taxonomy and the minimal model payload
(``product_key``, ``sample_name``, optional package weight). Reference
categories, nutrition, and match statuses are never included.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

from .constants import NUTRITION_FIELDS, NUTRITION_MODES
from .taxonomy import Taxonomy

SYSTEM_PROMPT = (
    "You are a strict grocery product classifier. Reply with one JSON object "
    "and nothing else."
)


def _coerce_payload(
    products: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    coerced: list[dict[str, object]] = []
    for product in products:
        if hasattr(product, "to_model_payload"):
            coerced.append(dict(product.to_model_payload()))  # type: ignore[attr-defined]
            continue
        payload: dict[str, object] = {
            "product_key": str(product.get("product_key", "")),
            "sample_name": str(product.get("sample_name", "")),
        }
        weight = product.get("package_weight_g")
        if weight is not None:
            payload["package_weight_g"] = weight
        coerced.append(payload)
    return coerced


def _schema_example(allow_abstain: bool = False) -> str:
    fields = [f'"{field_name}": number|null' for field_name in NUTRITION_FIELDS]
    if allow_abstain:
        return (
            '{"products": [{"product_key": string, "category": string|null, '
            '"abstain_reason": string|null, '
            + ", ".join(fields)
            + "}]}"
        )
    return (
        '{"products": [{"product_key": string, "category": string, '
        + ", ".join(fields)
        + "}]}"
    )


# Forced-choice rules, kept verbatim: the baseline prompt must not drift so that
# forced and abstain runs stay comparable. The abstain variant only inserts
# rules; it never rewrites or reorders these.
_FORCED_CHOICE_RULES = [
    "- Return exactly one object for every requested product_key, in the same "
    "order. Never add products that were not requested.",
    "- product_key must be copied exactly from the input.",
    "- category must exactly match one of the taxonomy categories. Do not "
    "invent, translate, abbreviate, or re-case category names.",
    "- Do not output macro_category: the macro is derived from the category.",
    "- For non-food products use category \"Non-Food\" and set every "
    "nutrition field to null.",
    "- Nutrition values are per 100 g: kcal_100g in kcal, all other fields "
    "in grams. Use numbers or null, never strings or ranges.",
    "- No explanations, no confidence scores, no chain of thought.",
]

# Inserted right after the exact-category rule when abstention is enabled.
_ABSTAIN_RULES = [
    "- If the product name does not contain enough information to choose "
    "exactly one category with confidence, set category to null and provide a "
    "short abstain_reason.",
    "- Prefer null over a plausible but unverified category. Never invent "
    "information that is not in the product name.",
    "- abstain_reason must be null when category is not null.",
]


def _rules(allow_abstain: bool) -> list[str]:
    if not allow_abstain:
        return list(_FORCED_CHOICE_RULES)
    return _FORCED_CHOICE_RULES[:3] + _ABSTAIN_RULES + _FORCED_CHOICE_RULES[3:]


def build_prompt(
    taxonomy: Taxonomy,
    products: Sequence[Mapping[str, object]],
    mode: str = "estimate",
    allow_abstain: bool = False,
) -> str:
    """Build the single user prompt for a batch of products."""
    if mode not in NUTRITION_MODES:
        raise ValueError(f"Unknown nutrition mode {mode!r}; expected one of {NUTRITION_MODES}.")
    payload = _coerce_payload(products)
    product_block = json.dumps(payload, ensure_ascii=True, indent=2)

    if mode == "estimate":
        nutrition_instruction = (
            'Nutrition mode: "estimate". For every food product provide a '
            "best-effort estimate per 100 g in the fields below. Use null for "
            "any value you do not know; never invent a source or citation. "
            "Do not leave fields out: explicitly use null for unknown values."
        )
    else:
        nutrition_instruction = (
            'Nutrition mode: "none". Do not estimate nutrition. Set every '
            "nutrition field to null (or omit it). Only classify the category."
        )

    lines = [
        "Classify each grocery product into exactly one category from the "
        "canonical taxonomy below.",
        "",
        "Output format: return ONLY a valid JSON object with a single "
        '"products" array, no Markdown fences and no commentary:',
        _schema_example(allow_abstain),
        "",
        "Rules:",
        *_rules(allow_abstain),
        "",
        nutrition_instruction,
        "",
        "Canonical taxonomy:",
        taxonomy.prompt_lines(),
        "",
        "Products to classify:",
        product_block if payload else "[]",
    ]
    return "\n".join(lines)


def build_messages(
    taxonomy: Taxonomy,
    products: Sequence[Mapping[str, object]],
    mode: str = "estimate",
    allow_abstain: bool = False,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_prompt(
                taxonomy, products, mode, allow_abstain=allow_abstain
            ),
        },
    ]


def render_prompt_preview(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n\n".join(
        f"[{message.get('role', 'unknown')}]\n{message.get('content', '')}"
        for message in messages
    )
