"""Safe local settings persistence for the Streamlit control panel."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping, MutableMapping

from ai_enrichment.config import AI_MODES
from ai_eval.providers import (
    DEFAULT_ENV_PATH,
    GROQ_DEFAULT_MODEL,
    FreeModelError,
    get_api_key,
    get_setting,
    parse_env_file,
    validate_groq_model,
)

AI_MODE_LABELS = {
    "categories_and_nutrition": "AI categories + nutritional estimates (Recommended)",
    "categories": "AI categories only",
    "off": "No automatic AI — edit manually or use your AI chatbot",
}
GROQ_KEY_URL = "https://console.groq.com/keys"


@dataclass(frozen=True)
class PublicAISettings:
    """Settings safe to display; never contains an API key."""

    mode: str
    model: str
    api_key_configured: bool
    api_key_managed_by_environment: bool
    settings_configured: bool


def load_public_ai_settings(
    env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
    environ: Mapping[str, str] | None = None,
) -> PublicAISettings:
    """Load display-safe settings without returning the secret value."""
    environment = os.environ if environ is None else environ
    file_settings = parse_env_file(env_path)
    raw_mode = environment.get("AI_ENRICHMENT_MODE") or file_settings.get("AI_ENRICHMENT_MODE")
    mode = (raw_mode or "off").strip().lower()
    if mode not in AI_MODES:
        mode = "off"
    model = get_setting("GROQ_MODEL", GROQ_DEFAULT_MODEL, env_path, environ) or GROQ_DEFAULT_MODEL
    try:
        model = validate_groq_model(model, allow_nonfree=False)
    except FreeModelError:
        model = GROQ_DEFAULT_MODEL
    configured_in_environment = bool(environment.get("GROQ_API_KEY"))
    configured = bool(get_api_key("GROQ_API_KEY", env_path, environ))
    return PublicAISettings(
        mode=mode,
        model=model,
        api_key_configured=configured,
        api_key_managed_by_environment=configured_in_environment,
        settings_configured=raw_mode is not None,
    )


def save_ai_settings(
    *,
    mode: str,
    model: str,
    new_api_key: str | None = None,
    clear_api_key: bool = False,
    env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
    environ: MutableMapping[str, str] | None = None,
) -> PublicAISettings:
    """Validate and atomically update only AI settings in the local env file.

    ``new_api_key=None`` means keep the current key. Clearing the key is a
    separate, explicit operation. The key is never returned or logged.
    """
    if mode not in AI_MODES:
        raise ValueError("Select one of the supported AI modes.")
    try:
        selected_model = validate_groq_model(model, allow_nonfree=False)
    except FreeModelError as exc:
        raise ValueError("Select a model from the approved free-model list.") from exc
    if clear_api_key and new_api_key and new_api_key.strip():
        raise ValueError("Enter a new key or choose to clear the existing key, not both.")
    secret = new_api_key.strip() if new_api_key and new_api_key.strip() else None
    if secret is not None and ("\n" in secret or "\r" in secret):
        raise ValueError("The API key must be a single line.")

    path = Path(env_path)
    existing_lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.is_file() else []
    updates: dict[str, str | None] = {
        "AI_ENRICHMENT_MODE": mode,
        "GROQ_MODEL": selected_model,
    }
    if clear_api_key:
        updates["GROQ_API_KEY"] = None
    elif secret is not None:
        updates["GROQ_API_KEY"] = secret

    output: list[str] = []
    seen: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        candidate = stripped[7:].strip() if stripped.startswith("export ") else stripped
        key = candidate.split("=", 1)[0].strip() if "=" in candidate else ""
        if key not in updates:
            output.append(line)
            continue
        if key in seen:
            continue
        seen.add(key)
        value = updates[key]
        if value is not None:
            output.append(f"{key}={value}\n")

    for key, value in updates.items():
        if key not in seen and value is not None:
            if output and not output[-1].endswith(("\n", "\r")):
                output[-1] += "\n"
            output.append(f"{key}={value}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".settings.tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(output)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

    runtime_env = os.environ if environ is None else environ
    if secret is not None:
        runtime_env["GROQ_API_KEY"] = secret
    elif clear_api_key:
        runtime_env.pop("GROQ_API_KEY", None)
    runtime_env["AI_ENRICHMENT_MODE"] = mode
    runtime_env["GROQ_MODEL"] = selected_model

    # Verify the on-disk values without ever returning the credential.
    saved = parse_env_file(path)
    return PublicAISettings(
        mode=mode,
        model=selected_model,
        api_key_configured=bool(runtime_env.get("GROQ_API_KEY") or saved.get("GROQ_API_KEY")),
        api_key_managed_by_environment=bool(runtime_env.get("GROQ_API_KEY")),
        settings_configured=True,
    )
