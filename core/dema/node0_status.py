"""Read-only Node0/DEMA status aggregation for operator CLI wrappers.

This module wraps existing DEMA service/status functions. It does not start
daemons, load models, run missions, or mutate memory beyond the current
script-level behavior of the reused status helpers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from scripts.dema import dema_service
from scripts.dema.dema_status import status as current_gap_status

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMA_ROOT = REPO_ROOT / "sovereign_state" / "dema"
LOCAL_ENV = REPO_ROOT / ".env"


def _local_env_value(name: str) -> str | None:
    """Read selected local .env keys without shell-sourcing the whole file."""
    if name in os.environ:
        return os.environ[name]
    if not LOCAL_ENV.exists():
        return None
    try:
        lines = LOCAL_ENV.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{name}="
    for line in lines:
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return None


def resolve_lm_studio_url() -> str:
    """Return the LM Studio base URL without an OpenAI `/v1` suffix."""
    url = _local_env_value("LM_STUDIO_URL")
    if not url:
        try:
            from core.integration.constants import LMSTUDIO_HOST, LMSTUDIO_PORT

            url = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}"
        except ImportError:
            url = "http://127.0.0.1:1234"
    return url.rstrip("/").removesuffix("/v1")


def _lm_url_trust_error(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "LM Studio URL must use http or https"
    hostname = parsed.hostname
    if not hostname:
        return "LM Studio URL is missing a host"
    if hostname.lower() == "localhost":
        return None
    try:
        host_ip = ip_address(hostname)
    except ValueError:
        return "LM Studio URL host must be localhost or a private IP address"
    if host_ip.is_loopback or host_ip.is_private or host_ip.is_link_local:
        return None
    return "LM Studio URL host must be localhost or a private IP address"


def _lm_headers() -> dict[str, str]:
    token = (
        _local_env_value("LM_API_TOKEN")
        or _local_env_value("LMSTUDIO_API_KEY")
        or _local_env_value("LM_STUDIO_API_KEY")
    )
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _read_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_lm_headers())
    with urllib.request.urlopen(  # nosec B310 - operator-configured local URL.
        request, timeout=timeout
    ) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if isinstance(parsed, dict):
        return parsed
    return {"data": parsed}


def _model_id(model: dict[str, Any]) -> str:
    return str(model.get("key") or model.get("id") or model.get("name") or "unknown")


def _loaded_models(models: list[Any]) -> list[dict[str, Any]]:
    loaded = []
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("loaded_instances") or model.get("loaded"):
            loaded.append(model)
            continue
        state = str(model.get("state") or "").lower()
        if state == "loaded":
            loaded.append(model)
    return loaded


def probe_lm_studio(*, timeout: float = 3.0) -> dict[str, Any]:
    """Probe LM Studio without loading models or changing server state."""
    base_url = resolve_lm_studio_url()
    attempts: list[dict[str, str]] = []
    trust_error = _lm_url_trust_error(base_url)
    if trust_error:
        return {
            "connected": False,
            "base_url": base_url,
            "endpoint": None,
            "source": None,
            "auth_required": False,
            "token_present": "Authorization" in _lm_headers(),
            "model_count": 0,
            "loaded_count": 0,
            "model_ids": [],
            "loaded_model_ids": [],
            "load_state_known": False,
            "attempts": [{"url": base_url, "error": trust_error}],
        }

    for suffix, source in (
        ("/api/v1/models", "native"),
        ("/v1/models", "openai_compat"),
    ):
        endpoint = f"{base_url}{suffix}"
        try:
            payload = _read_json(endpoint, timeout=timeout)
        except urllib.error.HTTPError as exc:
            attempts.append({"url": endpoint, "error": f"HTTP {exc.code}"})
            if exc.code == 401:
                return {
                    "connected": False,
                    "base_url": base_url,
                    "endpoint": endpoint,
                    "source": source,
                    "auth_required": True,
                    "token_present": "Authorization" in _lm_headers(),
                    "model_count": 0,
                    "loaded_count": 0,
                    "model_ids": [],
                    "loaded_model_ids": [],
                    "load_state_known": False,
                    "attempts": attempts,
                }
            continue
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            attempts.append({"url": endpoint, "error": str(exc)})
            continue

        models = payload.get("models", payload.get("data", []))
        if not isinstance(models, list):
            models = []
        loaded = _loaded_models(models)
        return {
            "connected": True,
            "base_url": base_url,
            "endpoint": endpoint,
            "source": source,
            "auth_required": False,
            "token_present": "Authorization" in _lm_headers(),
            "model_count": len(models),
            "loaded_count": len(loaded),
            "model_ids": [
                _model_id(model) for model in models if isinstance(model, dict)
            ][:10],
            "loaded_model_ids": [_model_id(model) for model in loaded][:10],
            "load_state_known": source == "native" or bool(loaded),
            "attempts": attempts,
        }

    return {
        "connected": False,
        "base_url": base_url,
        "endpoint": None,
        "source": None,
        "auth_required": any(
            attempt.get("error") == "HTTP 401" for attempt in attempts
        ),
        "token_present": "Authorization" in _lm_headers(),
        "model_count": 0,
        "loaded_count": 0,
        "model_ids": [],
        "loaded_model_ids": [],
        "load_state_known": False,
        "attempts": attempts,
    }


def read_node0_dema_status(root: Path = DEFAULT_DEMA_ROOT) -> dict[str, Any]:
    """Return a measured read-only Node0/DEMA operator status payload."""
    root = Path(root)
    service_status = dema_service.cmd_status(root)
    service_doctor = dema_service.cmd_doctor(root)
    current_gap = current_gap_status(root)
    lm_studio = probe_lm_studio()

    findings = list(service_doctor.get("findings", []))
    if not lm_studio["connected"]:
        findings.append("LM Studio local API is not reachable")
    if lm_studio.get("auth_required") and not lm_studio["token_present"]:
        findings.append("LM Studio requires auth, but no API token is configured")

    return {
        "kind": "node0_dema_status",
        "schema_version": "0.1.0",
        "ready": not findings,
        "truth_label": "MEASURED",
        "findings": findings,
        "root": str(root),
        "dema_service": service_status,
        "dema_doctor": service_doctor,
        "dema_current_gap": current_gap,
        "lm_studio": lm_studio,
    }
