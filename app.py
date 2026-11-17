"""Local Streamlit control panel for the Lidl product pipeline.

Three views share one header: *To complete* (incomplete products), *All
products* (search and edit anything) and *Settings*. Every table cell saves
itself when the user leaves it; report files follow a moment later, once the
user pauses, so typing is never slowed by a Parquet export.
"""

from __future__ import annotations

import copy
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

import pipeline_services
from ai_eval.providers import DEFAULT_ENV_PATH
from ai_eval.taxonomy import load_taxonomy
from app_settings import (
    AI_MODE_LABELS,
    GROQ_KEY_URL,
    load_public_ai_settings,
    save_ai_settings,
)
from product_store import missing_required_fields, review_status_text

APP_TITLE = "Lidl Spending & Nutrition"
APP_FOOTER = "Independent project. Not affiliated with Lidl."
PRIVACY_NOTE = (
    "AI completion sends only product names, codes, and package weight. "
    "Receipts, prices, and purchase history stay on this computer."
)
TABLE_COLUMNS = (
    "sample_name",
    "macro_category",
    "category",
    "package_weight_g",
    "kcal_100g",
    "proteins_100g",
    "carbs_100g",
    "sugars_100g",
    "fat_100g",
    "saturated_fat_100g",
    "fiber_100g",
    "salt_100g",
)
NUMERIC_COLUMNS = TABLE_COLUMNS[3:]
NUTRIENT_COLUMNS = TABLE_COLUMNS[4:]
EDITABLE_COLUMNS = TABLE_COLUMNS[2:]
REVIEW_TABLE_COLUMNS = ("review_status",) + TABLE_COLUMNS
FIELD_LABELS = {
    "sample_name": "Product",
    "macro_category": "Group",
    "category": "Category",
    "package_weight_g": "Package weight (g)",
    "kcal_100g": "Calories / 100 g",
    "proteins_100g": "Protein / 100 g",
    "carbs_100g": "Carbs / 100 g",
    "sugars_100g": "Sugar / 100 g",
    "fat_100g": "Fat / 100 g",
    "saturated_fat_100g": "Saturated fat / 100 g",
    "fiber_100g": "Fibre / 100 g",
    "salt_100g": "Salt / 100 g",
}
# Short table headers; the full label is the column tooltip.
COLUMN_TITLES = {
    "package_weight_g": "Weight (g)",
    "kcal_100g": "kcal",
    "proteins_100g": "Protein",
    "carbs_100g": "Carbs",
    "sugars_100g": "Sugar",
    "fat_100g": "Fat",
    "saturated_fat_100g": "Sat. fat",
    "fiber_100g": "Fibre",
    "salt_100g": "Salt",
}
COMPLETE_STATUS = "Complete"
VIEWS = ("todo", "weights", "all", "settings")
EXPORT_DELAY_SECONDS = 1.5
UNDO_LIMIT = 20
CHATBOT_FILE_STEPS = """1. Download the file and attach it in your chatbot (free accounts work).
2. Write: *Follow the instructions in this file.*
3. If the answer stops halfway, write *continue*. Paste every part below, or upload the file it gives you.
"""
# One chatbot answer can hold about this many weights; the request takes the
# products with the most money spent first, the next request the following ones.
WEIGHT_BATCH_SIZE = 100

DB_PATH = Path(os.environ.get("LIDL_APP_DB_PATH", "lidl_consumption.duckdb"))
CHAT_BATCH_PATH = Path(os.environ.get("LIDL_APP_CHAT_BATCH_PATH", "ai_chat_batch.txt"))
ENV_PATH = Path(os.environ.get("LIDL_APP_ENV_PATH", DEFAULT_ENV_PATH))
# The report's background photo. It is optional: without it, light mode uses
# the report's warm gradient instead.
BACKGROUND_PATH = Path(__file__).resolve().parent / "static" / "app-background.jpg"
BACKGROUND_URL = "app/static/app-background.jpg"
GRADIENT = "linear-gradient(135deg, #F2C29B 0%, #FAEBDA 38%, #FBF5EC 55%, #D3DFC3 100%)"


# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------

def _style(photo: bool) -> str:
    """Rules the theme cannot express. Dark mode keys off the data-lidl-theme
    attribute that the theme toggle keeps in sync with Streamlit's theme."""
    light_background = (
        f'url("{BACKGROUND_URL}") center / cover fixed no-repeat, #E9DCC6'
        if photo else f"{GRADIENT} fixed"
    )
    title_on_photo = (
        "html:not([data-lidl-theme='dark']) .st-key-masthead h1 {"
        " color: #FBF5EC !important; text-shadow: 0 1px 3px rgba(50,32,16,.55); }"
        if photo else ""
    )
    return f"""<style>
.block-container {{ max-width: 1440px; padding-top: 1.4rem; padding-bottom: 2rem; }}
[data-testid="stHeader"] {{ background: transparent; }}
/* The empty header bar lies over the masthead buttons; let clicks through
   everywhere except its own buttons (the main menu). */
[data-testid="stHeader"], [data-testid="stToolbar"] {{ pointer-events: none; }}
[data-testid="stHeader"] button, [data-testid="stMainMenu"] {{ pointer-events: auto; }}
/* Streamlit's "Running… Stop" badge would sit on the Sync button; progress
   is shown in the panel instead. */
[data-testid="stStatusWidget"] {{ visibility: hidden; }}
.stApp {{ background: {light_background}; }}
html[data-lidl-theme="dark"] .stApp {{ background: #1F2118; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
{title_on_photo}
.st-key-panel {{ background: rgba(255,255,255,.93); border-radius: 14px; padding: 20px 24px;
    box-shadow: 0 6px 22px rgba(62,42,24,.22); }}
html[data-lidl-theme="dark"] .st-key-panel {{ background: rgba(43,46,33,.96);
    box-shadow: 0 6px 22px rgba(0,0,0,.35); }}
.st-key-statusline, .st-key-footer {{ background: rgba(255,255,255,.78); border-radius: 10px;
    padding: 6px 12px; backdrop-filter: blur(6px); flex: none !important; }}
html[data-lidl-theme="dark"] .st-key-statusline, html[data-lidl-theme="dark"] .st-key-footer {{
    background: rgba(43,46,33,.85); }}
.st-key-views [data-testid="stButtonGroup"] button {{ background: rgba(255,255,255,.80);
    backdrop-filter: blur(6px); }}
html[data-lidl-theme="dark"] .st-key-views [data-testid="stButtonGroup"] button {{
    background: rgba(43,46,33,.88); }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ background: #5F6B3A !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] * {{ color: #FFFFFF !important; }}
html[data-lidl-theme="dark"] [data-testid="stButtonGroup"] button[aria-checked="true"] {{
    background: #9DAE6A !important; }}
html[data-lidl-theme="dark"] [data-testid="stButtonGroup"] button[aria-checked="true"] * {{
    color: #1F2118 !important; }}
textarea {{ font-family: Consolas, "Courier New", monospace !important; font-size: .85rem !important; }}
/* The table's hover toolbar sits above its top edge; keep it off the
   action buttons in the row above. */
.st-key-panel [data-testid="stElementContainer"]:has([data-testid="stDataFrame"]) {{ margin-top: 1.6rem; }}
</style>"""


_SUN = (
    "<svg viewBox='0 0 24 24' width='20' height='20' fill='none' stroke='currentColor' "
    "stroke-width='2' stroke-linecap='round'><circle cx='12' cy='12' r='4.2'/>"
    "<path d='M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6"
    "M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6'/></svg>"
)
_MOON = (
    "<svg viewBox='0 0 24 24' width='20' height='20' fill='none' stroke='currentColor' "
    "stroke-width='2' stroke-linejoin='round'><path d='M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 "
    "8.5 8.5 0 1 0 20.5 14.2z'/></svg>"
)
_THEME_TOGGLE_JS = """
const SUN = %SUN%;
const MOON = %MOON%;
// Streamlit remembers the chosen theme under this key (Settings > Theme).
const KEY = `stActiveTheme-${window.location.pathname}-v2`;

function currentMode() {
  const app = document.querySelector(".stApp");
  if (!app) return "light";
  const rgb = (getComputedStyle(app).color.match(/[0-9.]+/g) || [0, 0, 0]).map(Number);
  const luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255;
  return luminance > 0.5 ? "dark" : "light";
}

function mark() {
  const mode = currentMode();
  if (document.documentElement.dataset.lidlTheme !== mode) {
    document.documentElement.dataset.lidlTheme = mode;
  }
  return mode;
}

// The app has its own light and dark theme; this tells the Dark Reader
// extension not to darken it a second time (that turned the photo brown).
if (!document.querySelector('meta[name="darkreader-lock"]')) {
  const lock = document.createElement("meta");
  lock.name = "darkreader-lock";
  document.head.appendChild(lock);
}

export default function (component) {
  const { parentElement } = component;
  const button = parentElement.querySelector("button");
  if (!button) return;
  const paint = () => {
    const mode = mark();
    if (button.dataset.mode === mode) return;
    button.dataset.mode = mode;
    const dark = mode === "dark";
    button.innerHTML = dark ? SUN : MOON;
    const label = dark ? "Switch to light mode" : "Switch to dark mode";
    button.title = label;
    button.setAttribute("aria-label", label);
  };
  paint();
  // The menu can change the theme too; keep the page background in step.
  const timer = setInterval(paint, 300);
  button.onclick = () => {
    const next = currentMode() === "dark" ? "Light" : "Dark";
    try { window.localStorage.setItem(KEY, JSON.stringify(next)); } catch (error) {}
    window.location.reload();
  };
  return () => clearInterval(timer);
}
""".replace("%SUN%", repr(_SUN)).replace("%MOON%", repr(_MOON))
_THEME_TOGGLE_CSS = """
button {
  width: 40px; height: 40px; display: inline-flex; align-items: center; justify-content: center;
  border-radius: var(--st-button-radius); border: 1px solid var(--st-border-color);
  background: color-mix(in srgb, var(--st-background-color) 80%, transparent);
  color: var(--st-text-color); cursor: pointer; backdrop-filter: blur(6px);
}
button:hover { border-color: var(--st-primary-color); color: var(--st-primary-color); }
"""
_theme_toggle = st.components.v2.component(
    "lidl_theme_toggle",
    html="<button type='button' aria-label='Switch theme'></button>",
    css=_THEME_TOGGLE_CSS,
    js=_THEME_TOGGLE_JS,
)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "notice": None,               # (level, text) shown above the table
        "undo_stack": [],             # last saved cell edits, newest last
        "todo_keys": None,            # rows of "To complete", fixed per visit
        "weight_keys": None,          # rows of "Missing weight", fixed per visit
        "products_cache": None,       # all products, dropped after each save
        "editor_versions": {},        # bumped to reset a table to DB values
        "editor_context": {},         # what each table last showed
        "editor_seen": {},            # edits each table already handled
        "pending_non_food": None,     # category change waiting for confirmation
        "dialog": None,               # the open Groq or chatbot dialog
        "artifacts_pending": False,   # report files lag behind the DB
        "last_edit_at": 0.0,
        "export_retry_needed": False,
        "sync_running": False,
        "toast": None,
        "chat_prompt": None,
        "chat_preview": None,
        "chat_preview_response": None,
    }
    for name, value in defaults.items():
        st.session_state.setdefault(name, copy.deepcopy(value))


def _blank(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _cell_value(field: str, value: Any) -> Any:
    """Normalise an editor value: blank is unknown, numbers are floats."""
    value = _blank(value)
    if value is None or field not in NUMERIC_COLUMNS:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _format_value(field: str, value: Any) -> str:
    if value is None:
        return "empty"
    if field in NUMERIC_COLUMNS:
        number = float(value)
        text = f"{number:.4g}"
        return f"{text} g" if field == "package_weight_g" else text
    return str(value)


def _short_label(field: str) -> str:
    return FIELD_LABELS[field].replace(" / 100 g", "").replace(" (g)", "")


def _set_notice(level: str, text: str) -> None:
    st.session_state.notice = (level, text)


def _reset_table(view: str) -> None:
    """Show DB values again: a new editor key drops the editor's own edits."""
    versions = st.session_state.editor_versions
    versions[view] = versions.get(view, 0) + 1


def _data_changed() -> None:
    st.session_state.products_cache = None


def _mark_artifacts_pending() -> None:
    st.session_state.artifacts_pending = True
    st.session_state.last_edit_at = time.time()


def _go_to(view: str) -> None:
    st.session_state.dialog = None
    st.session_state.view = view
    _on_view_change()


def _on_view_change() -> None:
    # A new visit to "To complete" lists what is incomplete now; products
    # completed during the last visit drop out only at this point.
    st.session_state.todo_keys = None
    st.session_state.weight_keys = None
    st.session_state.notice = None


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _taxonomy():
    return load_taxonomy(pipeline_services._load_sync_module().TAXONOMY_PATH)


def _all_products() -> list[dict[str, Any]]:
    if st.session_state.products_cache is None:
        st.session_state.products_cache = pipeline_services.read_products(None, DB_PATH)
    return st.session_state.products_cache


def _todo_products() -> list[dict[str, Any]]:
    if st.session_state.todo_keys is None:
        st.session_state.todo_keys = [
            str(product["product_key"])
            for product in pipeline_services.read_review_products(DB_PATH)
        ]
    return pipeline_services.read_products(st.session_state.todo_keys, DB_PATH)


def _status(product: Mapping[str, Any], category_to_macro: Mapping[str, str]) -> str:
    if missing_required_fields(product, category_to_macro):
        return review_status_text(product)
    return COMPLETE_STATUS


def cell_changes(
    edited_rows: Mapping[Any, Mapping[str, Any]],
    seen_rows: Mapping[Any, Mapping[str, Any]],
    keys: list[str],
    shown: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str, Any, Any]]:
    """New cell edits as (product_key, field, value before, value after).

    ``edited_rows`` is the editor's cumulative edit state and ``seen_rows``
    the part already handled. "Before" is what the cell showed before this
    edit: the user's earlier edit if there was one, otherwise the DB value.
    """
    changes: list[tuple[str, str, Any, Any]] = []
    for row, fields in edited_rows.items():
        index = int(row)
        if index >= len(keys):
            continue
        key = keys[index]
        old = seen_rows.get(row, seen_rows.get(index, {})) or {}
        for field, value in fields.items():
            if field not in EDITABLE_COLUMNS:
                continue
            after = _cell_value(field, value)
            if field in old:
                before = _cell_value(field, old[field])
            else:
                before = _cell_value(field, shown.get(key, {}).get(field))
            if before == after:
                continue
            changes.append((key, field, before, after))
    return changes


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def _save_fields(
    key: str,
    fields: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> pipeline_services.ReviewSaveReport:
    """Save one product's fields now; report files follow after a pause."""
    report = pipeline_services.save_review_session(
        manual_edits={key: dict(fields)},
        baselines={key: dict(baseline)},
        db_path=DB_PATH,
        output_path=CHAT_BATCH_PATH,
        refresh_artifacts=False,
    )
    if report.manual_fields_saved:
        _mark_artifacts_pending()
        _data_changed()
    return report


def _apply_edit(view: str, key: str, name: str, fields: dict[str, Any],
                before: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Save an edit and report it. Returns False when the table must reset."""
    try:
        report = _save_fields(key, fields, baseline)
    except pipeline_services.PipelineBusyError:
        _set_notice("warning", "Not saved: Sync or Groq is still running. Try again when it finishes.")
        return False
    except Exception as exc:  # noqa: BLE001 - shown to the user, value kept in DB
        _set_notice("error", f"Not saved ({type(exc).__name__}). The table shows the stored values again.")
        return False
    if report.database_error:
        _set_notice("error", "Not saved. The table shows the stored values again.")
        return False
    if report.conflicts:
        labels = ", ".join(FIELD_LABELS.get(field, field) for _, field in report.conflicts)
        _set_notice(
            "warning",
            f"{name}: {labels} changed somewhere else in the meantime. "
            "The table now shows the latest value; type yours again if it is still right.",
        )
        return False
    changed = ", ".join(
        f"{_short_label(field)} {_format_value(field, value)}"
        for field, value in fields.items()
    )
    _set_notice("success", f"Saved: {name} · {changed}")
    st.session_state.toast = "Saved"
    stack = st.session_state.undo_stack
    stack.append({"view": view, "key": key, "name": name, "before": before, "after": fields})
    del stack[:-UNDO_LIMIT]
    return True


def _on_table_edit(editor_key: str) -> None:
    """Runs as soon as a cell is left. Must not call Streamlit elements before
    ``editor_seen`` is updated, so an interrupted run cannot save twice."""
    context = st.session_state.editor_context.get(editor_key)
    widget = st.session_state.get(editor_key) or {}
    edited = copy.deepcopy(dict(widget.get("edited_rows", {})))
    if context is None:
        return
    seen = st.session_state.editor_seen.get(editor_key, {})
    changes = cell_changes(edited, seen, context["keys"], context["shown"])
    st.session_state.editor_seen[editor_key] = edited
    if not changes:
        return
    view = context["view"]
    by_product: dict[str, dict[str, tuple[Any, Any]]] = {}
    for key, field, before, after in changes:
        by_product.setdefault(key, {})[field] = (before, after)
    keep_table = True
    for key, fields in by_product.items():
        shown = context["shown"][key]
        name = shown.get("sample_name") or key
        baseline = {field: shown.get(field) for field in TABLE_COLUMNS}
        baseline.update({field: before for field, (before, _) in fields.items()})
        baseline["__field_metadata__"] = shown.get("__field_metadata__", {})
        category = fields.get("category")
        if category and category[1] == "Non-Food" and category[0] != "Non-Food":
            # Non-Food empties the nutrition: ask first (see _non_food_dialog).
            fields = {field: value for field, value in fields.items() if field != "category"}
            st.session_state.pending_non_food = {
                "view": view, "key": key, "name": name, "baseline": baseline,
                "before": {field: shown.get(field) for field in ("category",) + NUTRIENT_COLUMNS},
            }
        if not fields:
            continue
        after = {field: value for field, (_, value) in fields.items()}
        before = {field: value for field, (value, _) in fields.items()}
        keep_table &= _apply_edit(view, key, name, after, before, baseline)
    if not keep_table:
        _reset_table(view)


def _undo_last() -> None:
    stack = st.session_state.undo_stack
    if not stack:
        return
    entry = stack.pop()
    # Fields missing from "after" were emptied by the edit (Non-Food clears
    # the nutrition), so the database holds blanks there now.
    baseline = {field: entry["after"].get(field) for field in entry["before"]}
    baseline["__field_metadata__"] = None
    try:
        report = _save_fields(entry["key"], entry["before"], baseline)
    except pipeline_services.PipelineBusyError:
        stack.append(entry)
        _set_notice("warning", "Undo waits: Sync or Groq is still running. Try again when it finishes.")
        return
    except Exception as exc:  # noqa: BLE001
        stack.append(entry)
        _set_notice("error", f"Undo failed ({type(exc).__name__}).")
        return
    _reset_table(entry["view"])
    if report.conflicts or report.database_error:
        _set_notice("warning", f"{entry['name']} changed again since that edit, so it was not undone.")
        return
    if len(entry["before"]) == 1:
        field, value = next(iter(entry["before"].items()))
        _set_notice("info", f"Undone: {entry['name']} · {_short_label(field)} back to {_format_value(field, value)}")
    else:
        _set_notice("info", f"Undone: {entry['name']} is back to how it was.")


@st.dialog("Mark as Non-Food?", width="small", on_dismiss=lambda: _cancel_non_food())
def _non_food_dialog() -> None:
    pending = st.session_state.pending_non_food
    st.write(
        f"**{pending['name']}** will lose its calories and nutrition values. "
        "The package weight is kept, and Undo brings the values back."
    )
    with st.container(horizontal=True):
        if st.button("Mark as Non-Food", type="primary", key="confirm_non_food"):
            st.session_state.pending_non_food = None
            before = dict(pending["before"])
            if _apply_edit(pending["view"], pending["key"], pending["name"],
                           {"category": "Non-Food"}, before, pending["baseline"]):
                _set_notice("success", f"Saved: {pending['name']} is now Non-Food; its nutrition was cleared.")
            _reset_table(pending["view"])
            st.rerun()
        if st.button("Keep the old category", key="cancel_non_food"):
            _cancel_non_food()
            st.rerun()


def _cancel_non_food() -> None:
    pending = st.session_state.pending_non_food
    st.session_state.pending_non_food = None
    if pending:
        _reset_table(pending["view"])


# --------------------------------------------------------------------------
# Report files (refreshed after a pause in editing)
# --------------------------------------------------------------------------

@st.fragment(run_every=1.0)
def _report_files_status() -> None:
    state = st.session_state
    if state.artifacts_pending and time.time() - state.last_edit_at >= EXPORT_DELAY_SECONDS:
        try:
            failures = pipeline_services.refresh_review_artifacts(DB_PATH, CHAT_BATCH_PATH)
            state.artifacts_pending = False
            state.export_retry_needed = "exports" in failures
            state.chat_prompt = None
        except pipeline_services.PipelineBusyError:
            pass  # try again on the next tick
        except Exception:  # noqa: BLE001
            state.artifacts_pending = False
            state.export_retry_needed = True
    if state.artifacts_pending:
        st.caption(":material/sync: Updating report files…")
    elif state.export_retry_needed:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.caption(":material/warning: Report files are out of date.")
            if st.button("Retry export", key="retry_export", type="tertiary", icon=":material/refresh:"):
                try:
                    pipeline_services.refresh_exports(DB_PATH)
                    state.export_retry_needed = False
                    st.toast("Report files updated", icon=":material/check_circle:")
                except Exception as exc:  # noqa: BLE001
                    st.toast(f"Export failed ({type(exc).__name__})", icon=":material/error:")
                st.rerun(scope="fragment")
    else:
        st.caption(":material/check_circle: Report files up to date. Refresh Power BI to see changes.")


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------

WEIGHT_TABLE_COLUMNS = (
    "sample_name", "package_weight_g", "category", "times_bought", "spend",
)


def _render_table(
    view: str,
    products: list[dict[str, Any]],
    *,
    columns: tuple[str, ...] = TABLE_COLUMNS,
    status: Any = None,
    editable: tuple[str, ...] = EDITABLE_COLUMNS,
) -> None:
    """An autosaving product table. ``status(product)`` fills the Status
    column; only ``editable`` columns accept edits."""
    taxonomy = _taxonomy()
    keys = [str(product["product_key"]) for product in products]
    records = []
    shown: dict[str, dict[str, Any]] = {}
    for product in products:
        values = {field: product.get(field) for field in TABLE_COLUMNS}
        values["__field_metadata__"] = product.get("field_metadata", {})
        shown[str(product["product_key"])] = values
        record = {column: product.get(column) for column in columns}
        if "review_status" in columns:
            record["review_status"] = status(product) if status else ""
        records.append(record)
    frame = pd.DataFrame(records, columns=columns, index=pd.Index(keys, name="product_key"))
    for column in NUMERIC_COLUMNS + ("times_bought", "spend"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    digest = hashlib.sha1("|".join(keys).encode()).hexdigest()[:10]
    version = st.session_state.editor_versions.get(view, 0)
    editor_key = f"{view}_table_{version}_{digest}"
    st.session_state.editor_context[editor_key] = {"view": view, "keys": keys, "shown": shown}
    st.data_editor(
        frame,
        key=editor_key,
        hide_index=True,
        width="stretch",
        height=min(600, 35 * (len(frame) + 1) + 3),
        num_rows="fixed",
        placeholder="",
        disabled=[column for column in columns if column not in editable],
        on_change=_on_table_edit,
        args=(editor_key,),
        column_config={
            "review_status": st.column_config.TextColumn("Status", width=130),
            "sold_as": st.column_config.TextColumn(
                "Sold as", width=95, help="Per piece: counter or bakery items. Package: the whole pack."
            ),
            "times_bought": st.column_config.NumberColumn("Bought", format="%d", width=75,
                                                          help="Receipt lines with this product."),
            "spend": st.column_config.NumberColumn("Spent (lei)", format="%.2f", width=100),
            "sample_name": st.column_config.TextColumn("Product", pinned=True, width=240),
            "macro_category": st.column_config.TextColumn(
                "Group", help="Follows from the category.", width=125
            ),
            "category": st.column_config.SelectboxColumn(
                "Category", options=[""] + list(taxonomy.category_to_macro), width=165
            ),
            **{
                field: st.column_config.NumberColumn(
                    COLUMN_TITLES[field], help=FIELD_LABELS[field], min_value=0.0, format="%.4g",
                    width=92 if field == "package_weight_g" else 70,
                )
                for field in NUMERIC_COLUMNS
            },
        },
    )


_NOTICE_ICONS = {
    "success": ":green[:material/check_circle:]",
    "info": ":blue[:material/undo:]",
    "warning": ":orange[:material/warning:]",
    "error": ":red[:material/error:]",
}


def _render_notice(default: str) -> None:
    """The latest save message (or ``default``) and Undo, in one row.

    Call this inside a container that is rendered on every run. The table
    below must keep its position in the page: if an element appeared above it
    only after a save, Streamlit would rebuild the table and the cell cursor
    would be lost while the user is typing.
    """
    notice = st.session_state.notice
    if notice:
        level, text = notice
        st.markdown(f"{_NOTICE_ICONS.get(level, '')} {text}")
    else:
        st.markdown(default)
    if st.session_state.undo_stack:
        st.button("Undo", key="undo_last", type="tertiary", icon=":material/undo:", on_click=_undo_last)


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------

def _render_todo() -> None:
    products = _todo_products()
    taxonomy = _taxonomy()
    incomplete = [
        product for product in products
        if _status(product, taxonomy.category_to_macro) != COMPLETE_STATUS
    ]
    if not products:
        default = ""
    elif incomplete:
        default = "Edits save automatically."
    else:
        default = ":green[:material/task_alt:] All done. These products are complete now."
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        _render_notice(default)
        st.space("stretch")
        names = {str(product["product_key"]): product.get("sample_name") for product in products}
        st.button("Complete with Groq", key="open_groq", icon=":material/done_all:",
                  disabled=not incomplete, on_click=_open_dialog,
                  args=("groq", [str(product["product_key"]) for product in incomplete], names))
        st.button("Use your chatbot", key="open_chatbot", icon=":material/chat:",
                  disabled=not incomplete, on_click=_open_dialog, args=("chatbot", [], names))
    if not products:
        st.success("Nothing to complete. Every product has a category and nutrition values.",
                   icon=":material/task_alt:")
        return
    _render_table("todo", products, columns=REVIEW_TABLE_COLUMNS,
                  status=lambda product: _status(product, taxonomy.category_to_macro))
    if any(product.get("cleared_reason") for product in incomplete):
        st.caption(
            "\"Values cleared\": the calories did not match protein, carbs and fat, so the nutrition "
            "was emptied to be filled again. If the label really says so, type its values; "
            "values you type are kept."
        )


def _weight_products() -> list[dict[str, Any]]:
    if st.session_state.weight_keys is None:
        st.session_state.weight_keys = [
            product["product_key"] for product in pipeline_services.read_weight_worklist(None, DB_PATH)
        ]
    return pipeline_services.read_weight_worklist(st.session_state.weight_keys, DB_PATH)


def _render_weights() -> None:
    products = _weight_products()
    missing = [product for product in products if not product.get("package_weight_g")]
    if not products:
        default = ""
    elif missing:
        default = "Type how many grams one item weighs, as it appears on the receipt."
    else:
        default = ":green[:material/task_alt:] All done. These products have a weight now."
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        _render_notice(default)
        st.space("stretch")
        names = {product["product_key"]: product.get("sample_name") for product in products}
        st.button("Use your chatbot", key="open_weight_chatbot", icon=":material/chat:",
                  disabled=not missing, on_click=_open_dialog,
                  args=("chatbot_weights", [product["product_key"] for product in missing[:WEIGHT_BATCH_SIZE]],
                        names),
                  help=f"Asks for the {WEIGHT_BATCH_SIZE} products you spent most on; run it again for the next ones.")
    if not products:
        st.success("Every food product you bought has a weight.", icon=":material/task_alt:")
        return
    _render_table("weights", products, columns=WEIGHT_TABLE_COLUMNS,
                  editable=("package_weight_g",))
    st.caption(
        "Without a weight a product is left out of the grams and nutrition totals in the report. "
        "Loose produce sold by the kilo is weighed on the receipt and never shows up here."
    )


def _render_all() -> None:
    query = st.text_input(
        "Search", key="query", placeholder="Search by name, for example: lapte",
        label_visibility="collapsed", icon=":material/search:",
    )
    products = _all_products()
    needle = pipeline_services._search_text(query.strip())
    if needle:
        products = [
            product for product in products
            if needle in pipeline_services._search_text(product.get("sample_name") or "")
        ]
    count = f"{len(products):,} product{'s' if len(products) != 1 else ''}"
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        _render_notice(f"{count}. Edits save automatically.")
    if not products:
        st.info("No product matches. Try a shorter part of the name.")
        return
    _render_table("all", products)


def _render_settings() -> None:
    public = load_public_ai_settings(ENV_PATH)
    with st.container(width=620):
        st.subheader("AI completion")
        labels = list(AI_MODE_LABELS.values())
        mode_by_label = {label: mode for mode, label in AI_MODE_LABELS.items()}
        selected_label = st.selectbox(
            "What should AI fill in?",
            labels,
            index=labels.index(AI_MODE_LABELS[
                public.mode if public.settings_configured else "categories_and_nutrition"
            ]),
            key="setting_ai_mode",
            help="AI fills empty fields only. It never replaces values you typed.",
        )
        new_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="Leave blank to keep the current key",
            key="setting_groq_key",
            autocomplete="new-password",
        )
        status = "set" if public.api_key_configured else "not set yet"
        st.caption(f"Key: {status}. [Get a free Groq API key]({GROQ_KEY_URL}).")
        if public.api_key_managed_by_environment:
            st.caption("The key comes from the process environment and is not shown here.")
        if st.button("Save settings", key="save_ai_settings", type="primary", icon=":material/save:"):
            try:
                save_ai_settings(
                    mode=mode_by_label[selected_label],
                    model=public.model,
                    new_api_key=new_key or None,
                    env_path=ENV_PATH,
                )
                st.session_state["clear_groq_key_next_run"] = True
                st.session_state.toast = "Settings saved"
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        st.caption(
            "AI nutrition values are estimates and may differ from the label. Groq usage limits apply. "
            "Sync uses these settings too."
        )
        st.subheader("Lidl account")
        st.caption(
            "Sign-in uses the local authentication command (auth_login.py). "
            "This app never asks for your Lidl password."
        )


# --------------------------------------------------------------------------
# Dialogs
# --------------------------------------------------------------------------

def _progress(label: str):
    status = st.status(label, expanded=True)
    bar = st.progress(0.0)

    def update(event: pipeline_services.ProgressEvent) -> None:
        status.write(event.message)
        if event.total and event.total > 0 and event.completed is not None:
            bar.progress(min(1.0, max(0.0, event.completed / event.total)))

    return status, update


def _open_dialog(name: str, keys: list[str], names: Mapping[str, Any]) -> None:
    """Dialogs stay open across reruns until closed or finished."""
    st.session_state.dialog = {"name": name, "keys": list(keys), "names": dict(names)}
    if name in {"chatbot", "chatbot_weights"}:
        st.session_state.chat_prompt = None
        st.session_state.chat_file_request = None
        st.session_state.chat_preview_source = None
        st.session_state.chat_upload_version = st.session_state.get("chat_upload_version", 0) + 1
        st.session_state.chat_preview = None
        st.session_state.chat_preview_response = None


def _close_dialog() -> None:
    st.session_state.dialog = None


def _render_open_dialog() -> None:
    dialog = st.session_state.dialog
    if not dialog:
        return
    if dialog["name"] == "groq":
        _groq_dialog(dialog["keys"], dialog["names"])
    elif dialog["name"] == "chatbot":
        _chatbot_dialog(dialog["names"])
    elif dialog["name"] == "chatbot_weights":
        _chatbot_dialog(dialog["names"], weight_keys=dialog["keys"])


@st.dialog("Complete with Groq", width="medium", on_dismiss=_close_dialog)
def _groq_dialog(keys: list[str], names: Mapping[str, Any]) -> None:
    settings = load_public_ai_settings(ENV_PATH)
    if settings.mode == "off" or not settings.api_key_configured:
        st.write("Groq needs an API key and an AI mode first.")
        st.button("Open Settings", type="primary", icon=":material/settings:",
                  on_click=_go_to, args=("settings",))
        return
    st.write(
        f"Groq fills the empty fields of **{len(keys)} incomplete products** with a free AI model. "
        "Values you typed yourself are never replaced."
    )
    with st.expander("Only some products", icon=":material/tune:"):
        selected = st.multiselect(
            "Products", options=keys, format_func=lambda key: names.get(key) or key,
            key="groq_selection", placeholder="All incomplete products",
        )
    targets = selected or keys
    st.caption("Sends only product names, codes and package weight. It can take a few minutes.")
    if st.button(f"Complete {len(targets)} product{'s' if len(targets) != 1 else ''}",
                 type="primary", icon=":material/done_all:", key="start_groq"):
        status, update = _progress("Completing products with Groq")
        try:
            report = pipeline_services.retry_ai_products(targets, DB_PATH, update)
        except pipeline_services.PipelineBusyError:
            status.update(label="Sync or another Groq run is still going. Try again shortly.", state="error")
            return
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"Groq stopped ({type(exc).__name__}). Values already accepted are saved.",
                          state="error")
            return
        remaining = {
            str(product["product_key"]) for product in pipeline_services.read_review_products(DB_PATH)
        }
        done = len(set(targets) - remaining)
        left = len(set(targets) & remaining)
        error = getattr(report.enrichment, "error_code", None)
        text = f"Groq completed {done} of {len(targets)} products."
        if left:
            text += f" {left} still need values: try again, use your chatbot, or type them."
        if error:
            text += f" Groq reported: {error}."
        st.session_state.export_retry_needed = "exports" in report.artifact_failures
        st.session_state.artifacts_pending = False
        _set_notice("warning" if left or error else "success", text)
        _data_changed()
        _reset_table("todo")
        _close_dialog()
        st.rerun()


def _suggestion_rows(preview, names: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in preview.suggestions:
        row: dict[str, Any] = {"Product": names.get(item.product_key) or item.product_key}
        if item.category is not None:
            row["Category"] = item.category
        if item.package_weight_g is not None:
            row[COLUMN_TITLES["package_weight_g"]] = item.package_weight_g
        for field, value in (item.nutrients or {}).items():
            row[COLUMN_TITLES.get(field, field)] = value
        rows.append(row)
    return pd.DataFrame(rows)


@st.dialog("Use your chatbot", width="large", on_dismiss=_close_dialog)
def _chatbot_dialog(names: Mapping[str, Any], weight_keys: list[str] | None = None) -> None:
    """Weight-only request when ``weight_keys`` is given, else the review batch."""
    state = st.session_state
    if state.chat_prompt is None:
        try:
            if weight_keys is not None:
                state.chat_prompt = pipeline_services.weight_batch_text(weight_keys, DB_PATH)
            else:
                pipeline_services.refresh_chat_batch(DB_PATH, CHAT_BATCH_PATH)
                state.chat_prompt = (
                    CHAT_BATCH_PATH.read_text(encoding="utf-8") if CHAT_BATCH_PATH.is_file() else ""
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not prepare the request ({type(exc).__name__}).")
            return
    if not state.chat_prompt:
        st.info("Nothing to ask right now: no product has empty fields a chatbot can fill.")
        return
    kind = "weights" if weight_keys is not None else "nutrition"
    st.markdown("**1. Give your chatbot the request** (ChatGPT, Claude, Gemini…). No API key needed.")
    copy_tab, file_tab = st.tabs(["Copy and paste", "Many products: use a file"])
    with copy_tab:
        hint = " It covers the 100 products you spent most on." if kind == "weights" else ""
        st.caption("Use the copy icon in the top-right corner of the box." + hint)
        st.code(state.chat_prompt, language=None, height=180, wrap_lines=True)
    with file_tab:
        st.markdown(CHATBOT_FILE_STEPS)
        if state.get("chat_file_request") is None:
            try:
                state.chat_file_request = pipeline_services.compact_chat_request(kind, DB_PATH)
            except Exception as exc:  # noqa: BLE001
                state.chat_file_request = ""
                st.error(f"Could not prepare the file ({type(exc).__name__}).")
        st.download_button(
            "Download for chatbot (.txt)", data=state.chat_file_request or "",
            file_name=f"lidl-{kind}-for-chatbot.txt", mime="text/plain",
            icon=":material/download:", key="download_chat_file", disabled=not state.chat_file_request,
            on_click="ignore",
        )
    st.markdown("**2. Paste its answer, or upload the file it gives you**")
    answer = st.text_area(
        "Chatbot answer", key="chat_answer", height=140, label_visibility="collapsed",
        placeholder='Paste the whole answer: JSON, or lines like 4056489123456;250',
    )
    upload = st.file_uploader(
        "Answer file", type=["txt", "csv"], label_visibility="collapsed",
        key=f"chat_answer_file_{state.get('chat_upload_version', 0)}",
    )
    source = upload.getvalue().decode("utf-8", errors="replace") if upload is not None else answer
    # Never disabled: a pasted answer reaches the app only when the box loses
    # focus, which is the same click that presses this button.
    if st.button("Check answer", key="check_chat_answer", icon=":material/fact_check:"):
        if not source.strip():
            st.warning("Paste the chatbot's answer or upload its file first.")
        else:
            try:
                normalized, unreadable = pipeline_services.normalize_chat_answer(source, kind)
                state.chat_preview = pipeline_services.preview_chat_import(normalized, DB_PATH)
                state.chat_preview_response = normalized
                state.chat_preview_source = source
                state.chat_unreadable = unreadable
            except Exception as exc:  # noqa: BLE001
                state.chat_preview = None
                st.error(f"Could not read this answer ({type(exc).__name__}). Paste the whole answer.")
    current = state.get("chat_preview_source") == source and source.strip()
    preview = state.chat_preview if current else None
    if preview is None:
        return
    answer = state.chat_preview_response
    if state.get("chat_unreadable"):
        st.caption(f"{state.chat_unreadable} line(s) of the answer could not be read and were left out.")
    st.markdown("**3. Check the values and save**")
    if preview.suggestions:
        st.dataframe(_suggestion_rows(preview, names), hide_index=True, width="stretch")
    problems = list(preview.issues) + list(preview.skipped)
    if problems:
        with st.expander(f"{len(problems)} value(s) left out", icon=":material/info:"):
            for issue in problems:
                product = names.get(issue.product_key) or issue.product_key or "Answer"
                st.caption(f"{product} · {FIELD_LABELS.get(issue.field_name or '', issue.field_name or issue.code)}: "
                           f"{issue.message}")
    count = preview.fields_suggested
    if not count:
        st.info("Nothing in this answer can be saved.")
        return
    if st.button(f"Save {count} suggested value{'s' if count != 1 else ''}", type="primary",
                 icon=":material/save:", key="save_chat_answer"):
        try:
            report = pipeline_services.save_review_session(
                manual_edits={}, baselines={}, response_text=answer,
                db_path=DB_PATH, output_path=CHAT_BATCH_PATH,
            )
        except pipeline_services.PipelineBusyError:
            st.warning("Sync or Groq is still running. Try again when it finishes.")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Saving stopped ({type(exc).__name__}). Some values may already be saved.")
            return
        saved = report.chat_import.fields_saved if report.chat_import else 0
        state.export_retry_needed = "exports" in report.artifact_failures
        state.chat_prompt = None
        state.chat_preview = None
        state.chat_preview_response = None
        state["clear_chat_answer_next_run"] = True
        _set_notice("success", f"Saved {saved} value{'s' if saved != 1 else ''} from your chatbot.")
        _data_changed()
        _reset_table("todo")
        _reset_table("weights")
        _close_dialog()
        st.rerun()


# --------------------------------------------------------------------------
# Sync
# --------------------------------------------------------------------------

def _run_sync() -> None:
    status, update = _progress("Syncing receipts and products")
    exports_failed = {"value": False}

    def callback(event: pipeline_services.ProgressEvent) -> None:
        update(event)
        if event.stage == "exports" and event.status == "failed":
            exports_failed["value"] = True

    try:
        pipeline_services.run_sync(DB_PATH, callback)
        count = len(pipeline_services.read_review_products(DB_PATH))
        st.session_state.export_retry_needed = False
        st.session_state.artifacts_pending = False
        text = "Sync complete."
        text += f" {count} product{'s' if count != 1 else ''} to complete." if count else " Nothing to complete."
        _set_notice("success", text)
    except pipeline_services.PipelineBusyError:
        _set_notice("warning", "Another update is already running. Wait for it to finish.")
    except Exception as exc:  # noqa: BLE001
        if exports_failed["value"]:
            st.session_state.export_retry_needed = True
            _set_notice("error", "Sync saved the data, but the report files failed. Use Retry export.")
        else:
            _set_notice("error", f"Sync failed ({type(exc).__name__}). Nothing restarts on its own.")
    finally:
        st.session_state.sync_running = False
        st.session_state.todo_keys = None
        st.session_state.undo_stack = []
        _data_changed()
        for view in ("todo", "all"):
            _reset_table(view)
    st.rerun()


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def _start_sync() -> None:
    st.session_state.sync_running = True
    st.session_state.notice = None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":material/receipt_long:", layout="wide")
    _init_state()
    st.html(_style(BACKGROUND_PATH.is_file()))
    if st.session_state.pop("clear_groq_key_next_run", False):
        st.session_state.setting_groq_key = ""
    if st.session_state.pop("clear_chat_answer_next_run", False):
        st.session_state.chat_answer = ""
    if st.session_state.toast:
        st.toast(st.session_state.toast, icon=":material/check_circle:")
        st.session_state.toast = None

    has_db = DB_PATH.is_file()
    running = st.session_state.sync_running
    with st.container(key="masthead"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.title(APP_TITLE, icon=":material/receipt_long:", width="content")
            st.space("stretch")
            _theme_toggle(key="theme_toggle", width="content")
            st.button("Sync", key="sync_pipeline", type="primary", icon=":material/sync:",
                      disabled=running, on_click=_start_sync,
                      help="Bring in new receipts, then complete and export products.")

    review_count = weight_count = None
    if has_db and not running:
        try:
            review_count = len(pipeline_services.read_review_products(DB_PATH))
            weight_count = len(pipeline_services.read_weight_worklist(None, DB_PATH))
        except pipeline_services.PipelineBusyError:
            review_count = weight_count = None
    labels = {
        "todo": ":material/fact_check: To complete"
        + (f" ({review_count})" if review_count is not None else ""),
        "weights": ":material/scale: Missing weight"
        + (f" ({weight_count})" if weight_count is not None else ""),
        "all": ":material/inventory_2: All products",
        "settings": ":material/settings: Settings",
    }
    # The view lives in ?view=<id> so the theme toggle's page reload keeps it.
    # (Streamlit's bind="query-params" would store the label, count included.)
    if st.session_state.get("view") not in VIEWS:
        requested = st.query_params.get("view")
        st.session_state.view = requested if requested in VIEWS else "todo"
    with st.container(horizontal=True, vertical_alignment="center", key="views"):
        view = st.segmented_control(
            "View", VIEWS, format_func=labels.get, required=True,
            key="view", on_change=_on_view_change, label_visibility="collapsed",
        ) or "todo"
        st.space("stretch")
        if has_db:
            with st.container(key="statusline", width="content"):
                _report_files_status()
    if st.query_params.get("view") != view:
        st.query_params["view"] = view

    with st.container(key="panel"):
        if running:
            _run_sync()
        elif not has_db and view != "settings":
            st.info(
                "First run: click Sync to create the local product database. If the Lidl sign-in is "
                "missing, run the local authentication command first; this app never asks for your "
                "Lidl password.",
                icon=":material/database:",
            )
        else:
            try:
                if view == "todo":
                    _render_todo()
                elif view == "weights":
                    _render_weights()
                elif view == "all":
                    _render_all()
                else:
                    _render_settings()
            except pipeline_services.PipelineBusyError:
                st.warning("Another update is running. This view loads when it finishes.",
                           icon=":material/hourglass_top:")

    if st.session_state.pending_non_food:
        _non_food_dialog()
    elif not running:
        _render_open_dialog()

    with st.container(key="footer"):
        st.caption(PRIVACY_NOTE)
        st.caption(APP_FOOTER)


if __name__ == "__main__":
    main()
