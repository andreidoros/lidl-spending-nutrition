"""Local Streamlit control panel for the Lidl product pipeline."""

from __future__ import annotations

import os
import html
from pathlib import Path
from typing import Any

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
from product_store import missing_required_fields

APP_TITLE = "Lidl Spending & Nutrition"
APP_FOOTER = "Independent project. Not affiliated with Lidl."

# Only the rules the native theme cannot express. Colours, radius, fonts and
# widget chrome stay in .streamlit/config.toml. No remote font import: the app
# promises that data stays on this computer, so it must not call Google Fonts.
_STYLE = """
<style>
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    letter-spacing: -0.02em;
}

textarea {
    font-family: "JetBrains Mono", Consolas, "Courier New", monospace !important;
    font-size: 0.85rem !important;
    background-color: #090D16 !important;
}

.incomplete-badge {
    display: inline-block;
    background-color: rgba(217, 119, 6, 0.18);
    color: #FACC15;
    border: 1px solid #D97706;
    border-radius: 4px;
    padding: 3px 10px;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap;
}
</style>
"""
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
DB_PATH = Path(os.environ.get("LIDL_APP_DB_PATH", "lidl_consumption.duckdb"))
CHAT_BATCH_PATH = Path(os.environ.get("LIDL_APP_CHAT_BATCH_PATH", "ai_chat_batch.txt"))
ENV_PATH = Path(os.environ.get("LIDL_APP_ENV_PATH", DEFAULT_ENV_PATH))


def _empty_review_state() -> None:
    st.session_state.setdefault("review_open", False)
    st.session_state.setdefault("review_base", None)
    st.session_state.setdefault("review_manual", {})
    st.session_state.setdefault("review_suggestions", {})
    st.session_state.setdefault("chat_response_input", "")
    st.session_state.setdefault("chat_preview", None)
    st.session_state.setdefault("chat_preview_response", None)
    st.session_state.setdefault("last_sync_complete", False)
    st.session_state.setdefault("last_sync_review_count", None)
    st.session_state.setdefault("last_operation_message", "")
    st.session_state.setdefault("review_initial_keys", None)
    st.session_state.setdefault("groq_summary", None)
    st.session_state.setdefault("review_notices", [])
    st.session_state.setdefault("product_base", None)
    st.session_state.setdefault("product_edits", {})
    st.session_state.setdefault("product_search_results", [])


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


def _has_review_draft() -> bool:
    return bool(
        st.session_state.review_manual
        or st.session_state.review_suggestions
        or st.session_state.chat_response_input.strip()
    )


def _has_unsaved_review() -> bool:
    return _has_review_draft() or bool(st.session_state.product_edits)


def _capture_product_edits() -> None:
    """Widget callback runs before Sync/Groq guards, including simultaneous clicks."""
    baseline = st.session_state.product_base
    if baseline is None:
        return
    delta = st.session_state.get("product_editor", {}).get("edited_rows", {}).get(0, {})
    edits = {
        field: _blank(value) for field, value in delta.items()
        if field in TABLE_COLUMNS[2:] and field != "macro_category"
        and _blank(value) != _blank(baseline.get(field))
    }
    st.session_state.product_edits = edits


def _invalidate_product_snapshot() -> None:
    st.session_state.product_base = None
    st.session_state["clear_product_editor_next_run"] = True


def _chat_preview_is_current() -> bool:
    response = st.session_state.chat_response_input
    return bool(
        response.strip()
        and st.session_state.chat_preview is not None
        and st.session_state.chat_preview_response == response
    )


def _read_review(*, force: bool = False) -> None:
    if st.session_state.review_base is not None and not force:
        return
    products = pipeline_services.read_review_products(DB_PATH)
    # Keep DB values separate from UI-only suggestions and manual cell edits.
    st.session_state.review_base = {
        str(product["product_key"]): dict(product) for product in products
    }
    if st.session_state.review_initial_keys is None:
        st.session_state.review_initial_keys = tuple(st.session_state.review_base)
    st.session_state.review_manual = {}
    st.session_state.review_suggestions = {}
    st.session_state.chat_preview = None
    st.session_state.chat_preview_response = None


def _display_rows() -> pd.DataFrame:
    base = st.session_state.review_base or {}
    suggestions = st.session_state.review_suggestions
    manual = st.session_state.review_manual
    sync = pipeline_services._load_sync_module()
    taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
    rows: list[dict[str, Any]] = []
    keys: list[str] = []
    for key, product in base.items():
        row = {column: product.get(column) for column in TABLE_COLUMNS}
        active_suggestions = suggestions if _chat_preview_is_current() else {}
        for field_name, value in active_suggestions.get(key, {}).items():
            if field_name in row and field_name not in manual.get(key, {}):
                row[field_name] = value
        for field_name, value in manual.get(key, {}).items():
            if field_name in row:
                row[field_name] = value
        category = _blank(row.get("category"))
        if category == "Non-Food":
            for field in NUMERIC_COLUMNS[1:]:
                if field not in manual.get(key, {}):
                    row[field] = None
        row["macro_category"] = taxonomy.category_to_macro.get(category) or ""
        rows.append(row)
        keys.append(key)
    frame = pd.DataFrame(rows, columns=TABLE_COLUMNS, index=pd.Index(keys, name="product_key"))
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _capture_editor_changes(edited: pd.DataFrame, rendered: pd.DataFrame) -> None:
    manual = st.session_state.review_manual
    base = st.session_state.review_base or {}
    for key in edited.index:
        product_key = str(key)
        if product_key not in base:
            continue
        for column in TABLE_COLUMNS:
            if column in {"sample_name", "macro_category"}:
                continue
            before = _blank(rendered.at[key, column])
            after = _blank(edited.at[key, column])
            if before != after:
                manual.setdefault(product_key, {})[column] = after


def _manual_edits_for_save() -> dict[str, dict[str, Any]]:
    edits: dict[str, dict[str, Any]] = {}
    for key, fields in st.session_state.review_manual.items():
        if fields:
            edits[key] = {field: _blank(value) for field, value in fields.items()}
    return edits


def _baseline_values() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, product in (st.session_state.review_base or {}).items():
        result[key] = {column: product.get(column) for column in TABLE_COLUMNS}
        result[key]["__field_metadata__"] = product.get("field_metadata", {})
    return result


def _conflict_message(conflicts, products) -> str:
    details = []
    for key, field in conflicts:
        name = products.get(key, {}).get("sample_name") or key
        details.append(f"{name} [{key}]: {FIELD_LABELS.get(field, field)}")
    return (
        "These fields changed after you opened them, so they were not saved. "
        "Your edits are still here. Copy them before discarding changes and reopening the product to load the latest values.\n\n"
        + "\n\n".join(details)
    )


def _progress_display(label: str):
    status = st.status(label, expanded=True)
    progress = st.progress(0.0)

    def update(event: pipeline_services.ProgressEvent) -> None:
        status.write(f"{event.stage}: {event.message}")
        if event.total and event.total > 0 and event.completed is not None:
            progress.progress(min(1.0, max(0.0, event.completed / event.total)))
        if event.status == "failed":
            status.update(label=f"{label} — failed", state="error")
        elif event.status in {"completed", "partial"} and event.stage in {"sync", "exports"}:
            status.update(label=event.message, state="complete" if event.status == "completed" else "error")

    return status, update


def _save_settings() -> None:
    public = load_public_ai_settings(ENV_PATH)
    with st.expander("AI settings", expanded=not public.settings_configured, icon=":material/settings:"):
        st.caption("Save settings, then click Complete with Groq. Saving settings does not start AI. Sync will also use these settings.")
        labels = list(AI_MODE_LABELS.values())
        mode_by_label = {label: mode for mode, label in AI_MODE_LABELS.items()}
        selected_label = st.selectbox(
            "What should AI fill in?",
            labels,
            index=labels.index(AI_MODE_LABELS[
                public.mode if public.settings_configured else "categories_and_nutrition"
            ]),
            key="setting_ai_mode",
            help="AI fills empty fields only. It does not replace your manual changes.",
        )
        st.markdown(
            f"[Get a Groq API key]({GROQ_KEY_URL}) · Key status: "
            f"{'configured' if public.api_key_configured else 'not configured'}"
        )
        if public.api_key_managed_by_environment:
            st.caption("The API key is supplied by the process environment and will not be shown here.")
        new_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="Leave blank to keep the configured key",
            key="setting_groq_key",
            autocomplete="new-password",
        )
        st.caption(
            "AI nutrition values are estimates and may differ from the label. "
            "Groq usage limits apply."
        )
        st.caption(
            "Lidl account setup uses the existing local authentication CLI. "
            "This app never asks for your Lidl password."
        )
        if st.button("Save AI settings", key="save_ai_settings", type="primary", icon=":material/settings:"):
            try:
                save_ai_settings(
                    mode=mode_by_label[selected_label],
                    model=public.model,
                    new_api_key=new_key or None,
                    env_path=ENV_PATH,
                )
                st.session_state["clear_groq_key_next_run"] = True
                st.session_state.review_notices = [("success", "AI settings saved. Click Complete with Groq to start, or use Sync for a full receipt update.")]
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))


def _render_groq_controls(base: dict[str, dict[str, Any]]) -> None:
    st.subheader("Automatic completion")
    st.write("Groq fills empty fields using a free AI model. Accepted values are saved right away, with no extra sync.")
    _save_settings()
    settings = load_public_ai_settings(ENV_PATH)
    with st.expander("Advanced: choose specific products", icon=":material/tune:"):
        subset = st.checkbox("Only process selected products", key="groq_subset")
        selected = st.multiselect(
            "Products to complete",
            options=list(base),
            format_func=lambda key: f"{base[key].get('sample_name') or key} · {key}",
            key="retry_ai_products",
            disabled=not subset,
        )
    targets = selected if subset else list(base)
    st.caption(f"{len(targets)} products selected. Groq fills empty fields allowed by your settings and leaves manual changes alone.")
    if settings.mode == "off" or not settings.api_key_configured:
        st.info("Choose an AI mode, enter a Groq key, and save AI settings above to enable completion.")
    if _has_unsaved_review():
        st.warning("Save changes or discard the current draft before running Groq.")
    disabled = not targets or settings.mode == "off" or not settings.api_key_configured or _has_unsaved_review()
    label = "Try Groq again" if st.session_state.groq_summary else "Complete with Groq"
    if st.button(label, key="retry_ai", disabled=disabled, type="primary", icon=":material/done_all:"):
        _, callback = _progress_display("Completing products with Groq")
        try:
            report = pipeline_services.retry_ai_products(targets, DB_PATH, callback)
            _invalidate_product_snapshot()
            _read_review(force=True)
            remaining = set(st.session_state.review_base)
            enrichment = report.enrichment
            st.session_state.groq_summary = {
                "requested": len(targets),
                "completed": len(set(targets) - remaining),
                "remaining": len(set(targets) & remaining),
                "status": getattr(enrichment, "status", "unknown"),
                "retryable_tasks": getattr(enrichment, "retryable_tasks", 0),
                "terminal_tasks": getattr(enrichment, "terminal_tasks", 0),
                "error_code": getattr(enrichment, "error_code", None),
            }
            st.session_state.export_retry_needed = "exports" in report.artifact_failures
            if report.artifact_failures:
                retry_hint = " Use Retry export for report files; no new Sync is needed." if "exports" in report.artifact_failures else " Report the failed review-file refresh before relying on those files."
                st.session_state.review_notices = [("warning", "Groq processing finished. Any accepted results were saved, but these derived files could not be updated: " + ", ".join(report.artifact_failures) + "." + retry_hint)]
            else:
                st.session_state.review_notices = [("info", "Groq processing finished. Any accepted results are already saved and report files updated. Refresh Power BI to see the changes. Check the outcome below for incomplete products.")]
            st.session_state.pop("chat_prompt", None)
            st.session_state["clear_editor_next_run"] = True
            st.rerun()
        except pipeline_services.PipelineBusyError:
            st.warning("Another app session is using the pipeline. Try again shortly.")
        except Exception as exc:
            st.error(f"Groq completion failed ({type(exc).__name__}). Previously accepted results may already be saved. No new Sync is required to retry.")


def _render_review() -> None:
    if not st.session_state.review_open:
        return
    title_col, badge_col, close_col = st.columns(
        [4.4, 2.2, 1.8], vertical_alignment="center"
    )
    with title_col:
        st.header("Product review", icon=":material/fact_check:")
    badge_slot = badge_col.empty()
    with close_col:
        if st.button("Close review", key="close_review", icon=":material/close:"):
            st.session_state.review_open = False
            st.rerun()
    if not DB_PATH.is_file():
        st.info("No local product database was found. Complete the initial sync from this app first.", icon=":material/database:")
        _save_settings()
        return
    try:
        _read_review()
    except pipeline_services.PipelineBusyError:
        st.warning("Another app session is updating the product database. Try again shortly.")
        return
    except Exception as exc:
        st.error(f"Could not load the review list ({type(exc).__name__}).")
        return

    base = st.session_state.review_base or {}
    sync = pipeline_services._load_sync_module()
    taxonomy = load_taxonomy(sync.TAXONOMY_PATH)
    initial_count = len(st.session_state.review_initial_keys or ())
    badge_slot.markdown(
        '<div style="text-align:right">'
        f'<span class="incomplete-badge">{len(base)} incomplete products</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.subheader("What needs attention")
    st.write(
        f"{len(base)} product(s) still incomplete — {initial_count} when you opened this review."
    )
    summary = st.session_state.groq_summary
    if summary:
        st.subheader("Last completion run")
        st.write(f"Last Groq run: {summary['requested']} requested · {summary['completed']} now complete · {summary['remaining']} still incomplete from that run.")
        st.caption(f"Service status: {summary['status']}. Retryable tasks: {summary['retryable_tasks']}. Invalid or abstained tasks: {summary['terminal_tasks']}. Task counts are not product counts.")
        if summary["error_code"]:
            st.warning(f"Groq reported: {summary['error_code']}", icon=":material/warning:")
    if base:
        with st.expander("Still needs attention" if summary else "See products and missing fields", icon=":material/list_alt:"):
            for product in base.values():
                missing = missing_required_fields(product, taxonomy.category_to_macro)
                st.write(f"{product.get('sample_name') or product['product_key']} — Missing: {', '.join(FIELD_LABELS.get(field, field) for field in missing)}")
            if summary:
                st.caption("These products still have missing fields. This does not always mean Groq gave a bad answer. Use your chatbot or edit them below.")
    if not base:
        st.success("No products currently need review.")
        _save_settings()
        return
    st.subheader("Two ways to finish these products")
    groq_column, chatbot_column = st.columns(2)
    groq_controls = groq_column.container(border=True)
    # Capture current editor changes before evaluating Groq's draft guard. The
    # container places its controls above the editor while executing them later.
    st.subheader("Check and edit")
    st.caption("Edit any value, then click Save changes. This updates report files without a new Sync or Groq call.")
    rendered = _display_rows()
    categories = [""] + list(taxonomy.category_to_macro)
    editor_result = st.data_editor(
        rendered,
        key="review_editor",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=True if st.session_state.product_edits else ["sample_name", "macro_category"],
        column_config={
            "sample_name": st.column_config.TextColumn(FIELD_LABELS["sample_name"]),
            "macro_category": st.column_config.TextColumn(FIELD_LABELS["macro_category"]),
            "category": st.column_config.SelectboxColumn(
                "Category", options=categories, required=False
            ),
            **{
                field: st.column_config.NumberColumn(FIELD_LABELS[field], min_value=0.0, format="%.4g")
                for field in NUMERIC_COLUMNS
            },
        },
    )
    _capture_editor_changes(editor_result, rendered)
    if st.session_state.review_manual:
        st.caption("You have unsaved changes. Click Save changes below.")
    with groq_controls:
        _render_groq_controls(base)

    with chatbot_column, st.container(border=True):
        st.subheader("Use your chatbot")
        st.write("No API key needed. Copy the incomplete-product batch into your chatbot, paste its JSON response here, validate and preview, then Save changes below.")
        if st.button("Prepare batch to copy", key="copy_chat_batch", disabled=bool(st.session_state.product_edits), icon=":material/content_copy:"):
            try:
                pipeline_services.refresh_chat_batch(DB_PATH, CHAT_BATCH_PATH)
                prompt = CHAT_BATCH_PATH.read_text(encoding="utf-8") if CHAT_BATCH_PATH.is_file() else ""
                st.session_state.chat_prompt = prompt
                if prompt:
                    st.success("Batch prepared. Use the copy button or the visible manual-copy/download options below.")
                else:
                    st.info("There are no chatbot-eligible fields right now.")
            except Exception as exc:
                st.error(f"Could not prepare the batch ({type(exc).__name__}).")
        prompt = st.session_state.get("chat_prompt", "")
        if prompt:
            safe_prompt = html.escape(prompt, quote=False)
            st.iframe(
                "<textarea id='chat-batch-copy-source' style='display:none'>"
                f"{safe_prompt}</textarea>"
                "<button id='chat-batch-copy-button' type='button'>Copy batch to clipboard</button>"
                "<span id='chat-batch-copy-status' role='status' style='margin-left:0.5rem'></span>"
                "<script>const b=document.getElementById('chat-batch-copy-button');"
                "b.addEventListener('click',async()=>{const s=document.getElementById('chat-batch-copy-status');"
                "try{await navigator.clipboard.writeText(document.getElementById('chat-batch-copy-source').value);"
                "s.textContent='Copied.';}catch(e){s.textContent='Clipboard access was blocked; copy the text below.';}});</script>",
                height=48,
            )
            st.text_area("Copy manually if needed", prompt, height=180, key="chat_prompt_fallback")
            st.download_button(
                "Download batch",
                prompt,
                file_name="ai_chat_batch.txt",
                mime="text/plain",
                icon=":material/download:",
            )
        st.text_area(
            "Paste AI response (JSON)",
            key="chat_response_input",
            height=180,
            disabled=bool(st.session_state.product_edits),
            help="Paste the JSON response separately; it will not be inserted into the table automatically.",
        )
        if st.button("Validate and preview", key="validate_chat_response", disabled=bool(st.session_state.product_edits), icon=":material/preview:"):
            blocked = {
                key: tuple(fields)
                for key, fields in st.session_state.review_manual.items()
            }
            try:
                preview = pipeline_services.preview_chat_import(
                    st.session_state.chat_response_input,
                    DB_PATH,
                    blocked_fields=blocked,
                )
                suggestions: dict[str, dict[str, Any]] = {}
                for item in preview.suggestions:
                    fields: dict[str, Any] = {}
                    if item.category is not None:
                        fields["category"] = item.category
                    fields.update(item.nutrients or {})
                    if item.package_weight_g is not None:
                        fields["package_weight_g"] = item.package_weight_g
                    suggestions[item.product_key] = fields
                st.session_state.review_suggestions = suggestions
                st.session_state.chat_preview = preview
                st.session_state.chat_preview_response = st.session_state.chat_response_input
                st.session_state["clear_editor_next_run"] = True
                st.success(f"Preview ready: {preview.fields_suggested} suggested field(s); nothing has been saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not validate the response ({type(exc).__name__}).")
        preview = st.session_state.chat_preview if _chat_preview_is_current() else None
        if st.session_state.chat_response_input.strip() and preview is None:
            st.info("This response has not been previewed. Validate it again before saving.")
        if preview:
            if preview.issues:
                st.warning("Validation issues")
                for issue in preview.issues:
                    st.write(f"{issue.product_key or 'Response'} · {issue.field_name or issue.code}: {issue.message}")
            if preview.skipped:
                st.info("Some fields were ignored because they are stale, protected, or not currently eligible.")
                for skipped in preview.skipped:
                    st.write(f"{skipped.product_key or 'Product'} · {skipped.field_name or skipped.code}: {skipped.message}")

    chat_text = st.session_state.chat_response_input
    preview_current = _chat_preview_is_current()
    if chat_text.strip() and not preview_current:
        st.info("Preview the exact pasted response before saving. Clear it to save manual edits only.")
    save_disabled = bool(st.session_state.product_edits) or not _has_review_draft() or (bool(chat_text.strip()) and not preview_current)
    col_save, col_discard = st.columns(2)
    with col_save:
        if st.button("Save changes", key="save_review", disabled=save_disabled, type="primary", icon=":material/save:"):
            try:
                report = pipeline_services.save_review_session(
                    manual_edits=_manual_edits_for_save(),
                    baselines=_baseline_values(),
                    response_text=chat_text if preview_current else "",
                    db_path=DB_PATH,
                    output_path=CHAT_BATCH_PATH,
                )
                _invalidate_product_snapshot()
                if "exports" in report.artifact_failures:
                    st.session_state.export_retry_needed = True
                notices = []
                if report.database_error:
                    notices.append(("error", "Saving stopped. Some changes may already be saved. Your edits are still here; check the values before trying again."))
                elif "exports" in report.artifact_failures:
                    notices.append(("error", "Your changes were saved, but report files could not be updated. Retry export — your edits have not been lost. No new Sync is needed."))
                elif report.artifact_failures:
                    notices.append(("warning",
                        "Your changes were saved, but these review artifacts failed to refresh: "
                        + ", ".join(report.artifact_failures)
                    ))
                else:
                    if report.conflicts:
                        saved = report.manual_fields_saved or (report.chat_import and report.chat_import.fields_saved)
                        notices.append(("info", "Some changes were saved. See the fields not saved below." if saved else "No changes were saved. See the fields below."))
                    else:
                        notices.append(("success", "Changes saved and report files updated. Refresh Power BI to see the changes."))
                if report.conflicts:
                    notices.append(("warning", _conflict_message(report.conflicts, st.session_state.review_base or {})))
                st.session_state.review_notices = notices
                for level, message in notices:
                    getattr(st, level)(message)
                if not report.database_error and not report.conflicts:
                    st.session_state.review_manual = {}
                    st.session_state.review_suggestions = {}
                    st.session_state.chat_preview = None
                    st.session_state.chat_preview_response = None
                    st.session_state["clear_chat_response_next_run"] = True
                    _read_review(force=True)
                    st.session_state.pop("chat_prompt", None)
                    st.session_state["clear_editor_next_run"] = True
                    st.session_state.export_retry_needed = "exports" in report.artifact_failures
                    st.rerun()
            except pipeline_services.PipelineBusyError:
                message = "Another app session is using the pipeline. Your draft is preserved; try again shortly."
                st.session_state.review_notices = [("warning", message)]
                st.warning(message)
            except Exception as exc:
                message = f"Save failed ({type(exc).__name__}). Your draft is preserved."
                st.session_state.review_notices = [("error", message)]
                st.error(message)
    with col_discard:
        if st.button("Discard changes", key="discard_review", disabled=not _has_review_draft(), icon=":material/undo:"):
            st.session_state.review_manual = {}
            st.session_state.review_suggestions = {}
            st.session_state["clear_chat_response_next_run"] = True
            st.session_state.chat_preview = None
            st.session_state.chat_preview_response = None
            st.session_state.review_base = None
            st.session_state["clear_editor_next_run"] = True
            st.rerun()


def _render_product_search() -> None:
    with st.expander("Find and edit a product", expanded=False, icon=":material/search:"):
        st.write("Search all local products, including those already complete. Type part of a name; accents and letter case do not matter.")
        if not DB_PATH.is_file():
            st.info("No local product database was found. Complete the initial sync first.")
            return
        review_dirty = _has_review_draft()
        dirty = bool(st.session_state.product_edits)
        if review_dirty:
            st.info("Save or discard your product-review draft before editing another product here.")
        if dirty:
            st.info("You have unsaved product changes. Save changes or discard them before changing product, syncing, or running Groq.")
        query = st.text_input(
            "Search by part of product name", key="product_search_query",
            placeholder="For example: lapte",
            help="Type a name fragment and press Enter to see matching suggestions. No exact name or diacritics needed.",
            disabled=dirty or review_dirty,
        )
        try:
            if not dirty and not review_dirty:
                st.session_state.product_search_results = pipeline_services.search_products(query, DB_PATH)
            matches = st.session_state.product_search_results
            names = {row["product_key"]: row["sample_name"] for row in matches}
            if query.strip() and not matches:
                st.info("No matching products. Try a shorter name fragment.")
            if len(matches) == 20:
                st.caption("Showing the first 20 matches. Type a longer fragment to narrow the suggestions.")
            selected = st.selectbox(
                "Matching products", options=list(names), index=None,
                format_func=lambda key: f"{names[key] or '(unnamed)'} · {key}",
                key="product_search_selection", placeholder="Select a product to edit",
                disabled=dirty or review_dirty,
            )
            if selected is None:
                return
            baseline = st.session_state.product_base
            if baseline is None or baseline["product_key"] != selected:
                if dirty:
                    st.warning("Save or discard the current product draft first.")
                    return
                baseline = pipeline_services.read_product(selected, DB_PATH)
                st.session_state.product_base = baseline
                st.session_state.product_edits = {}
                st.session_state.pop("product_editor", None)
            if baseline is None:
                st.warning("This product is no longer available. Search again.")
                return
        except pipeline_services.PipelineBusyError:
            st.warning("The workspace is busy. Your product draft is preserved; try again shortly.")
            return
        except Exception as exc:
            st.error(f"Could not load products ({type(exc).__name__}). Your draft is preserved.")
            return

        taxonomy = load_taxonomy(pipeline_services._load_sync_module().TAXONOMY_PATH)
        frame = pd.DataFrame([{field: baseline.get(field) for field in TABLE_COLUMNS}], columns=TABLE_COLUMNS)
        for field in NUMERIC_COLUMNS:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        edited = st.data_editor(
            frame, key="product_editor", hide_index=True, width="stretch", num_rows="fixed",
            disabled=True if review_dirty else ["sample_name", "macro_category"],
            column_config={
                # Display the derived macro below instead of a stale baseline
                # macro cell while the category is being edited.
                "macro_category": None,
                "sample_name": st.column_config.TextColumn("Product"),
                "category": st.column_config.SelectboxColumn("Category", options=[""] + list(taxonomy.category_to_macro)),
                **{field: st.column_config.NumberColumn(FIELD_LABELS[field], min_value=0.0, format="%.4g") for field in NUMERIC_COLUMNS},
            },
            on_change=_capture_product_edits,
        )
        category = _blank(edited.iloc[0]["category"])
        st.caption(f"Derived macro-category: {taxonomy.category_to_macro.get(category) or 'Unknown'}. Product name is read-only.")
        if category == "Non-Food" and baseline.get("category") != "Non-Food":
            st.warning("Saving Non-Food clears all nutrition values and their provenance; package weight is preserved.")
        edits = st.session_state.product_edits
        save_col, discard_col = st.columns(2)
        with save_col:
            if st.button("Save changes", key="save_product", disabled=not edits or review_dirty, type="primary", icon=":material/save:"):
                original = {field: baseline.get(field) for field in TABLE_COLUMNS}
                original["__field_metadata__"] = baseline.get("field_metadata", {})
                try:
                    report = pipeline_services.save_review_session(
                        manual_edits={selected: dict(edits)}, baselines={selected: original},
                        db_path=DB_PATH, output_path=CHAT_BATCH_PATH,
                    )
                    if "exports" in report.artifact_failures:
                        st.session_state.export_retry_needed = True
                    elif report.manual_fields_saved and not report.artifact_failures:
                        st.session_state.export_retry_needed = False
                    if report.database_error:
                        notices = [("error", "Product save stopped. Some changes may already be saved. Your draft is preserved; review current values before retrying.")]
                    elif report.conflicts:
                        notices = [("warning", _conflict_message(report.conflicts, {selected: baseline}))]
                    elif report.artifact_failures:
                        notices = [("warning", "Changes saved, but these derived files could not be updated: " + ", ".join(report.artifact_failures) + ". Use Retry export for report files; no new Sync is needed.")]
                    else:
                        notices = [("success", "Changes saved and report files updated. Refresh Power BI to see the changes.")]
                    st.session_state.review_notices = notices
                    st.session_state.review_base = None
                    st.session_state.pop("chat_prompt", None)
                    if not report.database_error and not report.conflicts:
                        st.session_state.product_edits = {}
                        _invalidate_product_snapshot()
                    st.rerun()
                except Exception as exc:
                    message = f"Product save failed ({type(exc).__name__}). Your draft is preserved."
                    st.session_state.review_notices = [("error", message)]
                    st.error(message)
        with discard_col:
            if st.button("Discard product changes", key="discard_product", disabled=not edits, icon=":material/undo:"):
                st.session_state.product_edits = {}
                _invalidate_product_snapshot()
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/receipt_long:",
        layout="wide",
    )
    st.markdown(_STYLE, unsafe_allow_html=True)
    _empty_review_state()
    if st.session_state.pop("clear_chat_response_next_run", False):
        st.session_state.chat_response_input = ""
    if st.session_state.pop("clear_groq_key_next_run", False):
        st.session_state.setting_groq_key = ""
    if st.session_state.pop("clear_editor_next_run", False):
        st.session_state.pop("review_editor", None)
    if st.session_state.pop("clear_product_editor_next_run", False):
        st.session_state.pop("product_editor", None)
    if st.session_state.get("chat_response_input", "").strip() and not _chat_preview_is_current():
        st.caption("Validate the pasted chatbot response before saving it.")
    st.title(APP_TITLE, icon=":material/receipt_long:")
    for level, message in st.session_state.review_notices:
        getattr(st, level)(message)
    if not st.session_state.review_open:
        st.caption("Sync to bring in new receipts. Product review finishes existing products without a new sync.")

    if not DB_PATH.is_file():
        st.info(
            "Initial setup is required. Use Sync to create the local product database. "
            "If Lidl authentication is missing, run the existing local authentication CLI; "
            "this app will not ask for your Lidl password.",
            icon=":material/database:",
        )
    if _has_unsaved_review():
        st.warning("You have an unsaved review draft. Save or discard it before Sync or AI retry.", icon=":material/edit_note:")
    sync_disabled = _has_unsaved_review()
    action_left, action_mid, action_right = st.columns(
        [1.2, 2.2, 6.6], vertical_alignment="center"
    )
    with action_left:
        sync_clicked = st.button(
            "Sync",
            type="primary",
            disabled=sync_disabled,
            key="sync_pipeline",
            icon=":material/sync:",
        )
    with action_mid:
        review_clicked = st.button(
            "Review products", key="open_review", icon=":material/fact_check:"
        )
        export_retry = st.session_state.get("export_retry_needed", False)
        export_clicked = export_retry and st.button(
            "Retry export", key="retry_export", icon=":material/refresh:"
        )
    # Search lives in the action row but is filled after _render_review(), so
    # review drafts are evaluated first and two editors cannot save over each
    # other's pending work.
    product_search_container = action_right.container()

    if sync_clicked:
        _, callback = _progress_display("Synchronizing")
        st.session_state.last_sync_complete = False
        sync_state = {"exports_failed": False}

        def sync_callback(event: pipeline_services.ProgressEvent) -> None:
            callback(event)
            if event.stage == "exports" and event.status == "failed":
                sync_state["exports_failed"] = True

        try:
            pipeline_services.run_sync(DB_PATH, sync_callback)
            _invalidate_product_snapshot()
            count = len(pipeline_services.read_review_products(DB_PATH))
            st.session_state.review_base = None
            st.session_state.review_initial_keys = None
            st.session_state.groq_summary = None
            st.session_state.review_notices = []
            st.session_state.review_suggestions = {}
            st.session_state.chat_preview = None
            st.session_state.chat_preview_response = None
            st.session_state["clear_editor_next_run"] = True
            st.session_state.last_sync_complete = True
            st.session_state.last_sync_review_count = count
            st.session_state.sync_completion_prompt = count > 0
            st.success("Synchronization complete.")
        except pipeline_services.PipelineBusyError:
            st.warning("Another app session is running an update. Wait for it to finish before trying again.")
        except Exception as exc:
            if sync_state["exports_failed"]:
                st.session_state.export_retry_needed = True
                st.error(
                    "Synchronization reached the report export stage and failed. "
                    "Earlier database updates may already be saved; use Retry export to refresh report files only."
                )
            else:
                st.error(f"Synchronization failed ({type(exc).__name__}). No operation was started automatically on page refresh.")

    if st.session_state.get("sync_completion_prompt"):
        count = st.session_state.last_sync_review_count or 0
        st.info(f"{count} products need review.", icon=":material/fact_check:")
        if st.button("Dismiss", key="close_sync_prompt", icon=":material/close:"):
            st.session_state.sync_completion_prompt = False
            st.rerun()

    if review_clicked:
        if not st.session_state.review_open and not _has_unsaved_review():
            st.session_state.review_initial_keys = None
            st.session_state.review_base = None
            st.session_state.groq_summary = None
        st.session_state.review_open = True
        st.session_state.sync_completion_prompt = False
        st.rerun()

    if export_clicked:
        _, callback = _progress_display("Refreshing report files")
        try:
            pipeline_services.refresh_exports(DB_PATH, callback)
            st.session_state.export_retry_needed = False
            st.session_state.review_notices = [("success", "Report files updated. Refresh Power BI to see the changes. No new Sync is needed.")]
            st.rerun()
        except Exception as exc:
            message = f"Export retry failed ({type(exc).__name__}). Saved database values are unchanged."
            st.session_state.review_notices = [("error", message)]
            st.error(message)

    product_search_container = action_right.container()
    _render_review()
    # Fill the action-row slot last: the search widget must not be evaluated
    # before the product-review editor.
    with product_search_container:
        _render_product_search()
    st.caption(
        "AI completion sends only product names, codes, and package weight. "
        "Receipts, prices, and purchase history stay on this computer."
    )
    st.caption(APP_FOOTER)


if __name__ == "__main__":
    main()
