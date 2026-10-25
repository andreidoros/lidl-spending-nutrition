# AI Enrichment Evaluation Harness (`ai_eval/`)

This is a **standalone evaluation tool**, not a production integration. It asks
free-tier Groq and OpenRouter models to classify unresolved grocery products
and (optionally) estimate nutrition per 100 g, then compares those predictions
against the local catalog seed.

It does **not** modify `sync_incremental.py`, `auth_login.py`, `run.bat`, the
DuckDB files, Parquet exports, `catalog_overrides.csv`, `review_queue.csv`, or
`data/catalogs/ro/catalog_seed.csv`. It has no Streamlit UI, no Ollama, and no
paid-model fallback.

## What it does

- Loads the canonical taxonomy at runtime from
  `data/catalogs/taxonomy.json` (7 macros / 21 categories, including `Non-Food`).
  Category names are never hardcoded.
- Builds a deterministic, stratified sample of up to ~100 real rows from
  `data/catalogs/ro/catalog_seed.csv` (fixed seed, stable ordering). Difficult
  rows such as abbreviated counter names, percentages, alcohol, non-food, and
  products without a package weight are prioritized.
- Keeps reference fields (category, macro, nutrition, `match_status`,
  confidence) local. Only `product_key`, `sample_name`, and optional
  `package_weight_g` are sent to a model.
- Requests strict JSON only: `{"products": [...]}` with an exact category and
  optional nutrition values per 100 g. `null` is required for unknown values.
- Validates every product individually: category membership, macro mismatch,
  numeric ranges, macro sum, sugar/carb and saturated-fat/fat relationships,
  Atwater consistency (alcohol exempt), Non-Food nutrition, and exact keys.
- Supports an optional **abstention** variant (`--abstain`, off by default): the
  model may answer `category: null` with a short `abstain_reason` instead of
  guessing. The forced-choice baseline flags an explicit null as
  `abstention_not_allowed`. See [Abstention](#abstention-opt-in) below.
- Runs providers sequentially, never falls back across providers or to a paid
  model, and continues gracefully when a provider request fails at runtime. If a
  required key is missing, the live run stops before any request with a clear
  message instead of silently skipping the provider.

Reference confidence is derived from the seed's `match_status`:

| Tier | `match_status` values |
| --- | --- |
| HIGH | `MANUAL_CORRECTED`, `PATCHED_EXACT`, `MATCHED_OFF_VERIFIED`, `NON_FOOD` |
| MEDIUM | `ENRICHED_MANUAL`, `MATCHED_OFF`, `SEED_CATALOG` |
| LOW | `AI_ESTIMATED` and anything else |

LOW-tier accuracy is reported for coverage but **must not be treated as ground
truth**, and it is never merged into the headline number: the
`category_accuracy.overall` entry counts HIGH and MEDIUM references only, while
LOW references are reported separately as "agreement with unverified
references". Nutrition MAE
only compares HIGH/MEDIUM food rows that have all of `kcal_100g`,
`proteins_100g`, `carbs_100g`, and `fat_100g`, and rejects non-finite (`NaN`,
`Infinity`) reference values.

## Abstention (opt-in)

By default the harness runs the **forced-choice baseline**: every product must get
exactly one taxonomy category. That baseline is unchanged, including its prompt
text, so old results stay comparable.

`--abstain` switches to the abstention variant for the whole run:

```powershell
& .\.venv\Scripts\python.exe -B -m ai_eval.runner --provider both --mode estimate --limit 100 --batch-size 10 --abstain
```

In this variant the prompt allows:

```json
{"products": [{"product_key": "string", "category": "string|null", "abstain_reason": "string|null", "kcal_100g": "number|null", "...": "..."}]}
```

and adds three rules: use `null` when the product name does not identify exactly
one category with confidence; prefer `null` over a plausible but unverified
guess; and keep `abstain_reason` null whenever `category` is not null.

**What counts as an abstention.** With `--abstain`, an explicit `category: null`
is a valid answer: the record gets `status: "ok"`, `prediction.abstained: true`,
`prediction.category: null`, and `prediction.macro_category: null`. Without
`--abstain` the same reply is a protocol violation: it keeps
`prediction.abstained: true` so the metrics stay honest, but it also gets the
`abstention_not_allowed` validation issue and `status: "flagged"`. The
forced-choice baseline therefore can never silently turn into an abstain run.
A *missing* `category` key is always a validation issue (`missing_category`) and
never a silent success. An abstained product must not carry nutrition values
(`nutrition_on_abstention`) and should carry a short reason of at most 200
characters (`missing_abstain_reason` otherwise; it stays an abstention). The
stored `abstain_reason` is truncated to 200 characters even when an over-long
reason is flagged, so `results.jsonl` never carries unbounded model text. A
non-null `category` must still be exact, must not carry an `abstain_reason`
(`abstain_reason_on_category`), and an out-of-taxonomy value is still
`invalid_category`. Abstention works in both `none` and `estimate` modes.

**Indicators.** For each provider over the same HIGH+MEDIUM reference rows, the
three abstention indicators (coverage, accuracy on answered, abstention rate)
are computed over three mutually exclusive buckets, plus the pre-existing
`overall` accuracy for baseline comparability. `category_accuracy.outcomes`
reports:

| Field | Meaning |
| --- | --- |
| `answered` / `abstained` / `missing` / `total` | Records with a non-null category, with an explicit `category: null` (or a missing `category` key), with no usable prediction (the product was not returned at all), and all three together. `answered + abstained + missing == total`. |
| `coverage` | `answered / total` — how much of the sample the model actually classified (`null` when nothing was evaluated). |
| `accuracy_on_answered` | `correct / answered` — accuracy among the products it did classify. |
| `abstention_rate` | `abstained / total` — explicit abstentions only. Missing products are never counted as abstentions. |
| `missing_rate` | `missing / total` — the share of products the model failed to return. |

An abstention is **never** correct or incorrect, and neither is a missing
product. Both are excluded from `wrong` and never listed as a misclassification;
`correct` and `wrong` only ever involve `answered` rows. A product the model
never answered is `missing`, so a model failure can never masquerade as a
principled abstention.

**`overall` and accuracy on answered.** `category_accuracy.overall` keeps its
previous definition — `correct / total` over the same HIGH+MEDIUM rows — and is
the number to compare between the forced and abstain runs; `overall.total`
always equals `outcomes.total`. `accuracy_on_answered` is the quality of the
answers the model actually committed to. In a forced run where every product was
answered the two coincide; they diverge only when some products are missing or
abstained, so the difference is never a property of the forced baseline itself.

**Comparing runs.** A forced run with a complete reply has `abstained = 0`,
`missing = 0`, `coverage = 1.0`, and `accuracy_on_answered` equal to
`overall.accuracy`. An abstain run is only interpretable together with coverage:
read coverage and abstention rate next to accuracy on answered, otherwise a high
accuracy on answered looks like an improvement when it is simply a smaller,
easier answered set.

The prompt variant is part of the result identity. Each `results.jsonl` line
stores `"abstain": true|false`; `--resume`, `latest_results`, and the summary all
match on provider + model + mode + `abstain`, so a forced record never satisfies
an abstain run (or vice versa) and the two variants are never summed together.
Legacy lines without the field are treated as forced-choice baseline rows.
`--dry-run` prints the active variant and the exact prompt.

Both variants share the same output directory, and a run without `--resume`
truncates `results.jsonl`. To keep the forced baseline and the abstain run side
by side, use a separate `--output-dir` per variant (for example
`scratch/ai_eval/forced` and `scratch/ai_eval/abstain`); with `--resume` the two
variants can also coexist in one file because they never match each other.

## Setup

Use the existing virtual environment. No new dependencies are required (only
`requests` and the Python standard library).

```powershell
& .\.venv\Scripts\python.exe -B -m ai_eval.runner --provider both --mode estimate --limit 5 --batch-size 5 --dry-run
```

A dry run builds the dataset, prints the plan (including the active prompt
variant, forced-choice or abstain) and the exact prompt, requires no API keys,
and makes **zero** network calls. It refuses to run together with
`--list-free-models` or `--list-groq-models` (exit code 2), because those flags
perform live HTTP requests; run them separately.

## Add API keys

1. Copy `.env.example` to `.env` if you do not already have one.
2. Fill in only what you need:

```dotenv
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
# Optional model overrides (also read from .env, not only the environment).
GROQ_MODEL=openai/gpt-oss-120b
# Required for OpenRouter unless you pass --model.
OPENROUTER_MODEL=some-vendor/some-model:free
```

Keys **and** model settings (`GROQ_MODEL`, `OPENROUTER_MODEL`) are read from the
environment first, then from `.env` with a minimal parser. `--model` overrides
both for a single-provider run. Key values are never printed, logged, or written
to results. `.env` is gitignored; never commit it.

To replace keys without touching `.env` (current PowerShell session only):

```powershell
$env:GROQ_API_KEY = "..."
$env:OPENROUTER_API_KEY = "..."
$env:OPENROUTER_MODEL = "..."
```

## Choose an OpenRouter free model

There is no guessed default free model ID, because they change over time. List
models whose prompt **and** completion prices are zero:

```powershell
& .\.venv\Scripts\python.exe -B -m ai_eval.runner --list-free-models
```

Pick an ID ending in `:free` (or exactly `openrouter/free`) and pass it via
`--model` for a single-provider run, or set `OPENROUTER_MODEL` for
`--provider both`.

Every zero-priced ID is listed, annotated with `[usable]` or
`[not :free - rejected by guard]`. Some models are zero-priced but still lack
the `:free` suffix; the client lists them for transparency yet keeps rejecting
them, so the `:free` guard is never weakened.

```powershell
& .\.venv\Scripts\python.exe -B -m ai_eval.runner --provider openrouter --model "vendor/model:free" --mode estimate --limit 10
```

The OpenRouter client rejects paid model IDs and never sends a `models`
fallback array. `openrouter/free` is the only accepted non-`:free` route.

## Groq free-tier guard

Groq requests are limited to the documented free-tier allowlist:

```text
openai/gpt-oss-120b          (default)
openai/gpt-oss-20b
openai/gpt-oss-safeguard-20b
qwen/qwen3.8-27b
```

Any other Groq model is rejected before a request is made. If you explicitly
accept possible billing, pass `--allow-nonfree-model`; there is still no silent
fallback. To see what the configured key can access:

```powershell
& .\.venv\Scripts\python.exe -B -m ai_eval.runner --list-groq-models
```

## Run the evaluation

Category-only mode:

```powershell
& .\.venv\Scripts\python.exe -B -m ai_eval.runner --provider both --mode none --limit 100 --batch-size 10
```

Category + estimated nutrition, resumable:

```powershell
& .\.venv\Scripts\python.exe -B -m ai_eval.runner --provider both --mode estimate --limit 100 --batch-size 10 --resume
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--provider groq\|openrouter\|both` | `both` | Sequential providers. |
| `--mode none\|estimate` | `estimate` | Category only or category + nutrition. |
| `--limit` | `100` | Maximum stratified products. |
| `--batch-size` | `10` | Products per request. |
| `--model` | provider-specific | Single-provider model ID. Not allowed with `both`. |
| `--dry-run` | off | Prompt preview only; no keys, no network, no files. Cannot be combined with `--list-free-models` / `--list-groq-models`. |
| `--resume` | off | Skip products with a valid successful result for the same provider + model + mode + prompt variant (`--abstain`). |
| `--abstain` | off | Opt-in abstention variant: `category: null` + `abstain_reason` is allowed instead of forcing a guess. Off keeps the forced-choice baseline unchanged. |
| `--delay` | `1.5` | Seconds between live requests. |
| `--output-dir` | `scratch/ai_eval` | Output directory; must resolve inside `scratch/` (see below). |
| `--seed` | `20240517` | Deterministic dataset seed. |
| `--allow-nonfree-model` | off | Groq only; explicit opt-in to a non-allowlisted model. |

`--output-dir` is resolved against the project root and confined: absolute paths
outside the repository, `..` escapes, and symlinks that lead out of the tree are
rejected with exit code 2 before anything is written. Without `--resume` a
confinement-checked run still starts fresh (the existing `results.jsonl` is
truncated), but only after the path check passes. The check is applied to each
artifact file as well: a pre-existing `results.jsonl`, `summary.json`, or
`report.md` that is a symlink or Windows junction pointing outside `scratch/` is
rejected with exit code 2 before the first write, so an external file is never
truncated or overwritten.

Retries happen only for timeouts, `429`, `5xx`, and connection errors: three
attempts maximum, exponential backoff with jitter, and `Retry-After` respected
as a minimum delay. `400`, `401`, and `403` are never retried, and neither are
other non-transient `requests` errors.

If `Retry-After` asks for more than the 120 s budget (numeric or HTTP-date), the
harness does **not** retry early: it records a single attempt with
`error_kind="rate_limited"` and the advised wait, and moves on. Retrying a
quota-exhausted endpoint too soon only burns the remaining attempts.

Provider error text is bounded to 500 characters and redacted before truncation.
When OpenRouter supplies `error.metadata`, the stable diagnostic fields
`error_type`, `provider_code`, and `provider_name` are appended to the message
(for example `HTTP 429: rate limited [error_type=rate_limit_exceeded,
provider_code=rate_limited]`). This distinguishes a rate limit from provider
overload or a guardrail block. Headers and the raw metadata payload are never
included.

A `200` response whose `choices[0].message.content` is empty or `null` (common
when a reasoning model spends its whole budget on reasoning, with
`finish_reason: "length"`) is classified as `error_kind="empty_content"` and is
**not retried**: the message includes `finish_reason` and, when present,
`usage.completion_tokens_details.reasoning_tokens`, because a retry does not
help. Genuinely malformed payloads (no `choices` array, a non-object `message`,
...) still report `error_kind="unexpected_response_shape"`.

Duplicate JSON object members (for example `{"products": [...], "products": [...]}`
or a repeated `product_key` / `category`) are rejected as malformed output
instead of silently keeping the last value. The taxonomy loader rejects a
duplicated macro the same way.

## Outputs

All generated files go under `scratch/ai_eval/` by default and are gitignored.
`--output-dir` may point at any nested directory below `scratch/`, but nothing
outside `scratch/` is ever written (the CLI exits 2 first).

- `results.jsonl` — one line per requested product per provider:
  provider, model, mode, `abstain` (true/false prompt variant), product key,
  `input_name` (the product's sample name, so a run is auditable without
  rebuilding the dataset), local reference snapshot, validated prediction
  (including `abstained` and `abstain_reason`), per-product validation issues,
  status (`ok`/`flagged`/`error`), batch-level issues, latency, attempts, errors,
  timestamp. Only the product name is added; no quantities, prices, or ticket
  ids are stored.
- `summary.json` — availability and error counts, JSON-valid batch rate,
  latency, category accuracy overall (HIGH + MEDIUM references only), the
  abstention block `category_accuracy.outcomes` (coverage, accuracy on answered,
  abstention rate, missing rate), per taxonomy category
  (`category_accuracy.by_category`, same HIGH+MEDIUM restriction, with
  `answered`/`abstained`/`missing`/`coverage`/`accuracy_on_answered`) and by
  confidence tier, agreement with LOW references tracked separately,
  misclassified products with their input names, nutrition MAE / coverage, and
  validation issue counts. The top-level `abstain` flag records which prompt
  variant produced the file.
- `report.md` — a short head-to-head comparison, written when both providers
  are run in the same invocation. The headline row reads "Category accuracy
  (HIGH+MEDIUM references)"; LOW references appear only as "Agreement with
  unverified LOW references (not accuracy)". An "Abstention" section reports
  coverage, missing rate, abstention rate, and accuracy on answered per provider,
  and states that forced and abstain runs are not directly comparable without
  reading coverage. Per-provider details include an "Accuracy by category" table
  with answered/abstained/missing columns sorted worst-first by accuracy on
  answered, and a misclassification listing formatted as
  `Input name | Reference category -> Predicted category` (abstentions and
  missing products are never listed there).

Resume only skips `status == "ok"` rows with `validation.valid == true` for the
same provider + model + **exact** mode + prompt variant. Mode matching is strict:
a row saved by `--mode none` never satisfies an `--mode estimate` resume, legacy
rows with no stored `mode` are re-run rather than assumed to match, and a
forced-choice row never satisfies an `--abstain` resume (or vice versa). Flagged,
missing, failed, and parse-error products are re-run.

A killed process can leave a torn final JSONL line. Before the next append the
incomplete tail is truncated back to the last newline, so the following record
starts on a clean line instead of being glued to the fragment. A trailing
*complete* JSON object that merely lacks its final newline is kept and only the
missing newline is added, so a complete record is never deleted. Non-finite
floats (`NaN`, `Infinity`) are written as `null`; `json.dumps` runs with
`allow_nan=False`, so an invalid number can never produce unparsable JSON.

Resume reads the file as bytes and decodes each line separately. If a process
was killed mid-write of a multibyte character (Romanian diacritics appear in
product names), only the torn line is skipped; the valid records before and
after it are still loaded instead of aborting resume with a
`UnicodeDecodeError`. Lines are split on `\n` only, so line-separator characters
inside a JSON string cannot split a valid record.

## Tests (offline)

```powershell
& .\.venv\Scripts\python.exe -B -m unittest tests.test_ai_eval -v
& .\.venv\Scripts\python.exe -B -m unittest discover tests -v
```

The tests use mocked transports and temporary directories. They never call
Groq or OpenRouter and never mutate the seed, DuckDB files, or Parquet exports.

## Important limitations

- **Free quotas can be exhausted or changed.** A `429` is retried only while
  the advised `Retry-After` fits the 120 s budget; a longer wait is reported as
  `rate_limited` instead of retrying early. The harness never switches to a paid
  model.
- **Nutrition values are model estimates, not verified labels.** They are useful
  for comparison, not for production enrichment.
- LOW-tier reference rows are not ground truth.
- The evaluation sample is small (~100 products); treat accuracy as directional
  rather than statistically definitive.
- Network access is required only for live runs and the two explicit listing
  commands. Everything else is local and deterministic.
