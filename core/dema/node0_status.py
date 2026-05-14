"""Read-only Node0/DEMA status aggregation for operator CLI wrappers.

This module wraps existing DEMA service/status functions. It does not start
daemons, load models, run missions, or mutate memory beyond the current
script-level behavior of the reused status helpers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from ipaddress import ip_address
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMA_ROOT = REPO_ROOT / "sovereign_state" / "dema"
LOCAL_ENV = REPO_ROOT / ".env"
NODE_CONSOLE_FORBIDDEN_ACTIONS = (
    "daemon_start",
    "mission_dispatch",
    "node1_activation",
    "public_demo",
    "external_provider_routing",
    "economic_token_claim",
)


class _LazyDemaService:
    """Lazy proxy that avoids importing service modules during core.dema import."""

    def __getattr__(self, name: str) -> Any:
        from scripts.dema import dema_service as service

        return getattr(service, name)


dema_service = _LazyDemaService()


def current_gap_status(root: Path) -> dict[str, Any]:
    """Lazy wrapper for the current-gap status helper."""
    from scripts.dema.dema_status import status

    return status(root)


class NodeConsoleDependencyId(str, Enum):
    """Stable identifiers for Dema Node Console dependency panels."""

    PYTHON_VENV = "python_venv"
    PYO3_BRIDGE = "pyo3_bridge"
    RUST_BUS = "rust_bus"
    MODEL_BACKEND = "model_backend"
    TOKEN_CURRENT_PROCESS = "token_current_process"
    DAEMON_STATE = "daemon_state"
    EVIDENCE_LEDGER = "evidence_ledger"


class NodeConsoleDependencyStatus(str, Enum):
    """Fail-closed dependency states for the Node Console."""

    READY = "READY"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class NodeConsoleDependency:
    """One operator-facing dependency row for the Dema Node Console."""

    dependency_id: NodeConsoleDependencyId
    label: str
    status: NodeConsoleDependencyStatus
    observed: str
    required: str
    detail: str
    next_action: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-safe dependency payload."""
        return {
            "id": self.dependency_id.value,
            "label": self.label,
            "status": self.status.value,
            "observed": self.observed,
            "required": self.required,
            "detail": self.detail,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class DemaNodeConsoleStatus:
    """Pure Dema Node Console v0.1 dependency/status contract."""

    ready: bool
    activation_gate: str
    dependencies: tuple[NodeConsoleDependency, ...]
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe console status payload."""
        return {
            "kind": "dema_node_console_status",
            "schema_version": "0.1.0",
            "truth_label": "MEASURED",
            "ready": self.ready,
            "activation_gate": self.activation_gate,
            "forbidden_actions": NODE_CONSOLE_FORBIDDEN_ACTIONS,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "findings": list(self.findings),
        }


def _dependency(
    dependency_id: NodeConsoleDependencyId,
    *,
    label: str,
    status: NodeConsoleDependencyStatus,
    observed: str,
    required: str,
    detail: str,
    next_action: str | None = None,
) -> NodeConsoleDependency:
    return NodeConsoleDependency(
        dependency_id=dependency_id,
        label=label,
        status=status,
        observed=observed,
        required=required,
        detail=detail,
        next_action=next_action,
    )


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
    if host_ip.is_link_local:
        return "LM Studio URL host must not be a link-local metadata address"
    if host_ip.is_loopback or host_ip.is_private:
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


def python_venv_active() -> bool:
    """Return whether this process is running inside a Python virtualenv."""
    return bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != sys.base_prefix


def pyo3_bridge_importable() -> bool:
    """Return whether the PyO3 Rust event bridge is importable."""
    try:
        from core.sovereign.event_bus import is_rust_event_bus_available
    except ImportError:
        return False
    return is_rust_event_bus_available()


def build_node_console_status(
    *,
    python_venv: bool,
    pyo3_bridge: bool,
    rust_bus_available: bool,
    model_backend_connected: bool,
    loaded_model_count: int,
    token_visible: bool,
    auth_required: bool,
    daemon_running: bool,
    evidence_ledger_observable: bool,
) -> DemaNodeConsoleStatus:
    """Build the safe Dema Node Console dependency contract.

    The console is an operator status surface only. It never grants runtime
    activation; bounded missions remain gated on an explicit operator GO.
    """
    dependencies = [
        _dependency(
            NodeConsoleDependencyId.PYTHON_VENV,
            label="Python venv",
            status=(
                NodeConsoleDependencyStatus.READY
                if python_venv
                else NodeConsoleDependencyStatus.BLOCKED
            ),
            observed="active" if python_venv else "not active",
            required="active repo virtualenv",
            detail=(
                "Python commands are scoped to a virtualenv."
                if python_venv
                else "Activate .venv before running Node0 Python gates."
            ),
            next_action=None if python_venv else "source .venv/bin/activate",
        ),
        _dependency(
            NodeConsoleDependencyId.PYO3_BRIDGE,
            label="PyO3 bridge",
            status=(
                NodeConsoleDependencyStatus.READY
                if pyo3_bridge
                else NodeConsoleDependencyStatus.BLOCKED
            ),
            observed="importable" if pyo3_bridge else "missing",
            required="bizra.PyEventBridge importable",
            detail=(
                "Rust bridge binding is visible to Python."
                if pyo3_bridge
                else "Build/install bizra-omega/bizra-python into this venv."
            ),
            next_action=(
                None
                if pyo3_bridge
                else "cd bizra-omega/bizra-python && maturin develop --release"
            ),
        ),
        _dependency(
            NodeConsoleDependencyId.RUST_BUS,
            label="Rust Bus",
            status=(
                NodeConsoleDependencyStatus.READY
                if rust_bus_available
                else NodeConsoleDependencyStatus.BLOCKED
            ),
            observed="binding available" if rust_bus_available else "unavailable",
            required="read-only visibility of the Rust event bus binding",
            detail=(
                "Console does not wire subscribers or emit events."
                if rust_bus_available
                else "Rust Bus cannot be observed from this Python process."
            ),
            next_action=(
                None
                if rust_bus_available
                else "run the Rust Bus bootstrap before runtime activation"
            ),
        ),
        _dependency(
            NodeConsoleDependencyId.MODEL_BACKEND,
            label="Model backend",
            status=(
                NodeConsoleDependencyStatus.READY
                if model_backend_connected and loaded_model_count > 0
                else (
                    NodeConsoleDependencyStatus.WARNING
                    if model_backend_connected
                    else NodeConsoleDependencyStatus.BLOCKED
                )
            ),
            observed=(
                f"connected, {loaded_model_count} loaded"
                if model_backend_connected
                else "not reachable"
            ),
            required="local LM Studio reachable with at least one loaded model",
            detail=(
                "Local model backend is ready for bounded diagnostics."
                if model_backend_connected and loaded_model_count > 0
                else (
                    "Backend reached, but no loaded model was confirmed."
                    if model_backend_connected
                    else "Local model backend was not reachable by the probe."
                )
            ),
            next_action=(
                None
                if model_backend_connected and loaded_model_count > 0
                else "load a local LM Studio model and rerun status"
            ),
        ),
        _dependency(
            NodeConsoleDependencyId.TOKEN_CURRENT_PROCESS,
            label="Token visibility",
            status=(
                NodeConsoleDependencyStatus.READY
                if token_visible
                else (
                    NodeConsoleDependencyStatus.BLOCKED
                    if auth_required
                    else NodeConsoleDependencyStatus.WARNING
                )
            ),
            observed="visible" if token_visible else "not visible",
            required="visible in the current process before activation",
            detail=(
                "The current process can see an LM Studio token."
                if token_visible
                else (
                    "Token is not required by this probe, but activation remains gated."
                    if not auth_required
                    else "LM Studio requires auth and this process cannot see a token."
                )
            ),
            next_action=(
                None
                if token_visible
                else "rerun status from the token-bearing terminal"
            ),
        ),
        _dependency(
            NodeConsoleDependencyId.DAEMON_STATE,
            label="Daemon state",
            status=(
                NodeConsoleDependencyStatus.WARNING
                if daemon_running
                else NodeConsoleDependencyStatus.READY
            ),
            observed="running" if daemon_running else "stopped",
            required="stopped before first bounded diagnostic activation",
            detail=(
                "A daemon is already running; inspect before issuing GO."
                if daemon_running
                else "Daemon is stopped; runtime pulse has not been fired."
            ),
            next_action=None if not daemon_running else "inspect daemon PID and logs",
        ),
        _dependency(
            NodeConsoleDependencyId.EVIDENCE_LEDGER,
            label="Evidence ledger",
            status=(
                NodeConsoleDependencyStatus.READY
                if evidence_ledger_observable
                else NodeConsoleDependencyStatus.WARNING
            ),
            observed="observable" if evidence_ledger_observable else "not observed",
            required="ledger path or receipt surface observable",
            detail=(
                "Evidence surface is visible for receipt inspection."
                if evidence_ledger_observable
                else "Empty/missing evidence before first pulse is acceptable."
            ),
            next_action=(
                None
                if evidence_ledger_observable
                else "create first receipt only after explicit bounded GO"
            ),
        ),
    ]
    blockers = [
        dependency
        for dependency in dependencies
        if dependency.status is NodeConsoleDependencyStatus.BLOCKED
    ]
    findings = tuple(
        f"{dependency.label}: {dependency.detail}" for dependency in blockers
    )
    return DemaNodeConsoleStatus(
        ready=not blockers,
        activation_gate="EXPLICIT_GO_REQUIRED",
        dependencies=tuple(dependencies),
        findings=findings,
    )


def _evidence_ledger_observable(root: Path) -> bool:
    evidence_candidates = (
        REPO_ROOT / "evidence",
        REPO_ROOT / ".proof-forge" / "EVIDENCE_INDEX.json",
        root / "receipts",
    )
    return any(path.exists() for path in evidence_candidates)


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
    node_console = build_node_console_status(
        python_venv=python_venv_active(),
        pyo3_bridge=pyo3_bridge_importable(),
        rust_bus_available=pyo3_bridge_importable(),
        model_backend_connected=bool(lm_studio["connected"]),
        loaded_model_count=int(lm_studio.get("loaded_count", 0)),
        token_visible=bool(lm_studio["token_present"]),
        auth_required=bool(lm_studio.get("auth_required")),
        daemon_running=bool(service_status.get("running", False)),
        evidence_ledger_observable=_evidence_ledger_observable(root),
    )

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
        "dema_node_console": node_console.to_dict(),
    }
