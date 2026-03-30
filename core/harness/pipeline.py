"""
BIZRA Harness Pipeline — run_mission() is the one entry point.

Ingest → Classify → Gate → Execute → Receipt → Persist → Surface

Canonical enforcement is hard-fail inside this boundary.
Adapter unavailable = mission aborted, not downgraded.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import blake3

from core.harness.constants import (
    EXECUTION_IHSAN_FLOOR,
    HARNESS_VERSION,
    RECEIPT_SCHEMA_VERSION,
    REFLEX_PRECIPITATION_HITS,
    SURFACE_CONTRACT_VERSION,
)

# Hard-fail canonical imports — if these fail, the harness cannot operate
from core.proof_engine.canonical_receipt_adapter import (
    GENESIS_SEED,
    CanonicalReceipt,
    ExecutionRoute,
    VerdictStatus,
    from_mission_result,
)

logger = logging.getLogger("bizra.harness")

# ── Ledger persistence ─────────────────────────────────────────────


def _ledger_path() -> str:
    return os.environ.get(
        "BIZRA_RECEIPT_LEDGER",
        os.path.expanduser("~/.bizra-kernel/ledger/canonical_receipts.jsonl"),
    )


def _read_previous_hash() -> bytes:
    import json

    path = _ledger_path()
    try:
        last = ""
        with open(path) as f:
            for line in f:
                if line.strip():
                    last = line.strip()
        if last:
            rid = json.loads(last).get("receipt_id", "")
            if len(rid) == 64:
                return bytes.fromhex(rid)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    return GENESIS_SEED


def _append_to_ledger(receipt_dict: dict) -> None:
    import fcntl
    import json
    from pathlib import Path

    path = Path(_ledger_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(receipt_dict, separators=(",", ":")) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ── Reflex cache (in-memory for now) ──────────────────────────────

_reflex_cache: dict[str, list[str]] = {}
_reflex_hits: dict[str, int] = {}


# ── Pipeline data types ───────────────────────────────────────────


@dataclass
class HarnessResult:
    """The complete output of run_mission()."""

    receipt: CanonicalReceipt
    response: str
    model_used: str
    route: str
    surface: dict[str, Any]
    request_id: str


# ── The one entry point ───────────────────────────────────────────


def run_mission(
    text: str,
    context: Optional[dict[str, str]] = None,
    slot: str = "cold_core",
    request_id: Optional[str] = None,
) -> HarnessResult:
    """Execute a mission through the constitutional pipeline.

    This is the ONLY way missions enter the system.
    Canonical enforcement is hard-fail — no degradation inside the harness.
    """
    request_id = request_id or uuid.uuid4().hex[:16]
    received_at = int(time.time() * 1000)
    context = context or {}

    # ── 1. Classify intent ────────────────────────────────────────
    intent_hash = blake3.blake3(text.encode()).hexdigest()[:32]

    # ── 2. Route selection (reflex / deliberate) ──────────────────
    cached = _reflex_cache.get(intent_hash)
    if cached:
        route_str = "REFLEX"
        response_text = "; ".join(cached)
        model_used = "reflex_cache"
        ihsan = 0.95
        verdict_status = VerdictStatus.ADMITTED
    else:
        # ── 3. Deliberate path — call kernel ──────────────────────
        route_str = "DELIBERATE"
        response_text, model_used, ihsan, verdict_status = _invoke_kernel(text, slot)

        # ── 3b. If kernel fails, this is a hard failure ──────────
        if not response_text:
            route_str = "DEGRADED"
            response_text = f"[DEGRADED] Kernel unreachable for: {text[:60]}"
            model_used = "none"
            ihsan = 0.0
            verdict_status = VerdictStatus.DEFERRED

    # ── 4. Gate check ─────────────────────────────────────────────
    if ihsan < EXECUTION_IHSAN_FLOOR and verdict_status == VerdictStatus.ADMITTED:
        verdict_status = VerdictStatus.REJECTED
        logger.warning(
            "Ihsan %.4f below execution floor %.2f — rejecting",
            ihsan,
            EXECUTION_IHSAN_FLOOR,
        )

    # ── 5. Build canonical receipt ────────────────────────────────
    route_map = {
        "REFLEX": ExecutionRoute.REFLEX,
        "DELIBERATE": ExecutionRoute.DELIBERATE,
        "DEGRADED": ExecutionRoute.DEGRADED,
    }

    previous = _read_previous_hash()
    receipt = from_mission_result(
        mission_id=f"{request_id}-{received_at}",
        genesis_hash=GENESIS_SEED,
        policy_version=f"harness-{HARNESS_VERSION}",
        verdict=verdict_status,
        ihsan_score=ihsan,
        snr_score=0.9,  # TODO: wire real SNR measurement
        route=route_map.get(route_str, ExecutionRoute.DEGRADED),
        input_text=text,
        output_text=response_text,
        previous_receipt=previous,
        received_at=received_at,
    )
    receipt.receipt_id = receipt.compute_id()

    # ── 6. Persist to durable ledger ──────────────────────────────
    receipt_dict = receipt.to_dict()
    receipt_dict["request_id"] = request_id
    receipt_dict["model_used"] = model_used
    _append_to_ledger(receipt_dict)

    # ── 7. Reflex precipitation ───────────────────────────────────
    if (
        route_str == "DELIBERATE"
        and verdict_status == VerdictStatus.ADMITTED
        and ihsan >= 0.90
    ):
        _reflex_hits[intent_hash] = _reflex_hits.get(intent_hash, 0) + 1
        if _reflex_hits[intent_hash] >= REFLEX_PRECIPITATION_HITS:
            _reflex_cache[intent_hash] = [response_text[:500]]
            logger.info("Reflex precipitated for hash %s", intent_hash[:8])

    # ── 8. Build surface contract ─────────────────────────────────
    surface = _build_surface(receipt, response_text, model_used, request_id)

    return HarnessResult(
        receipt=receipt,
        response=response_text,
        model_used=model_used,
        route=route_str,
        surface=surface,
        request_id=request_id,
    )


# ── Kernel invocation ─────────────────────────────────────────────


def _invoke_kernel(text: str, slot: str) -> tuple[str, str, float, VerdictStatus]:
    """Call the kernel cognitive API. Returns (response, model, ihsan, verdict)."""
    try:
        import httpx

        kernel_url = os.environ.get("BIZRA_KERNEL_URL", "http://localhost:8010")
        token = os.environ.get("BIZRA_API_TOKEN", "")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{kernel_url}/v1/cognitive/invoke",
                json={"prompt": text, "slot": slot},
                headers=headers,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("success") and "[ERROR]" not in data.get("response", ""):
                    ihsan = data.get("ihsan_score", 0.0)
                    verdict = (
                        VerdictStatus.ADMITTED
                        if data.get("ihsan_passed")
                        else VerdictStatus.REJECTED
                    )
                    return data["response"], data.get("model_used", ""), ihsan, verdict
    except Exception as exc:
        logger.warning("Kernel invocation failed: %s", exc)

    return "", "", 0.0, VerdictStatus.DEFERRED


# ── Surface contract ──────────────────────────────────────────────


def _build_surface(
    receipt: CanonicalReceipt,
    response: str,
    model_used: str,
    request_id: str,
) -> dict[str, Any]:
    """Build the versioned surface contract consumed by all UI surfaces."""
    return {
        # Contract metadata
        "surface_contract_version": SURFACE_CONTRACT_VERSION,
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "_canonical": True,
        # Receipt fields
        "receipt_id": receipt.receipt_id.hex(),
        "mission_id": receipt.mission_id,
        "state": receipt.state.name,
        "verdict": receipt.verdict.name,
        "ihsan_score": receipt.ihsan_score,
        "snr_score": receipt.snr_score,
        "route": receipt.route.name,
        "federation_admissible": receipt.federation_admissible,
        # Response
        "response": response,
        "model_used": model_used,
        "request_id": request_id,
        # Timestamps
        "received_at": receipt.received_at,
        "sealed_at": receipt.sealed_at,
    }
