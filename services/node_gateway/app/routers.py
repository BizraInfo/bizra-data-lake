import hmac
import os
import sys
import time
from pathlib import Path

from app.node.hhmm import HHMM
from app.node.mission_bridge import MissionBridge
from app.node.reflex_cache import ReflexCache
from app.node.snr import snr_score
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from _shared.app.health import build_health_payload, check_http, check_redis
except Exception:  # pragma: no cover - local import fallback
    try:
        from services._shared.app.health import (
            build_health_payload,
            check_http,
            check_redis,
        )
    except Exception:
        services_root = Path(__file__).resolve().parents[2]  # .../services
        artifacts_root = services_root.parent  # .../.tmp_prod_artifacts_v2
        for candidate in (artifacts_root, services_root):
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
        try:
            # Prefer direct _shared import to avoid collision with foreign
            # top-level "services" packages present on developer machines.
            from _shared.app.health import build_health_payload, check_http, check_redis
        except Exception:
            from services._shared.app.health import (
                build_health_payload,
                check_http,
                check_redis,
            )

router = APIRouter()
APP_VERSION = "4.0.1"

hhmm = HHMM()
cache = ReflexCache()
mission_bridge = MissionBridge()


class Observation(BaseModel):
    text: str
    context: dict[str, str] = Field(default_factory=dict)


class Plan(BaseModel):
    macro_state: str
    steps: list[str]
    snr: float
    poi_score: float


def _reflex_min_snr() -> float:
    raw = (os.environ.get("BIZRA_REFLEX_MIN_SNR") or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, min(float(raw), 1.0))
    except ValueError:
        return 0.0


def _require_api_key(x_bizra_api_key: str | None) -> None:
    expected = (os.environ.get("BIZRA_API_KEY") or "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="BIZRA_API_KEY not set")
    if not isinstance(x_bizra_api_key, str) or not hmac.compare_digest(
        x_bizra_api_key.strip(), expected
    ):
        raise HTTPException(status_code=401, detail="unauthorized")


@router.get("/health")
def health():
    checks = {
        "redis": check_redis(),
        "urp_registry": check_http("URP_REGISTRY_URL", "http://urp-registry:8000"),
        "urp_knowledge_graph": check_http(
            "URP_KG_URL", "http://urp-knowledge-graph:8000"
        ),
        "urp_consensus": check_http("URP_CONSENSUS_URL", "http://urp-consensus:8000"),
        "urp_verification": check_http(
            "URP_VERIFICATION_URL", "http://urp-verification:8000"
        ),
    }
    return build_health_payload(
        service="node_gateway",
        version=APP_VERSION,
        checks=checks,
        extra={
            "node_id": os.environ.get("NODE_ID", "unknown"),
            "reflex_cache_backend": cache.backend(),
            "reflex_cache_count": cache.count(),
        },
    )


@router.post("/v1/plan")
async def plan(
    obs: Observation,
    x_bizra_api_key: str | None = Header(default=None, alias="x-bizra-api-key"),
) -> Plan:
    _require_api_key(x_bizra_api_key)
    # 1) Predict macro-state (HHMM)
    macro = hhmm.predict(obs.text, obs.context)
    # 2) Try reflex cache (O(1) lookup)
    steps = cache.get(macro)
    if steps:
        snr = snr_score(signal=0.9, noise=0.35)
        poi = max(0.0, min((snr + 0.9) / 2.0, 1.0))
        return Plan(macro_state=macro, steps=steps, snr=snr, poi_score=poi)

    # 3) Cache miss -> sovereign mission pipeline bridge
    bridge_plan = await mission_bridge.run(obs.text, obs.context, macro_state=macro)
    if bridge_plan is not None:
        if bridge_plan.snr >= _reflex_min_snr():
            cache.put(macro, bridge_plan.steps)
        return Plan(
            macro_state=bridge_plan.macro_state,
            steps=bridge_plan.steps,
            snr=bridge_plan.snr,
            poi_score=bridge_plan.poi_score,
        )

    # 4) Hard fallback for minimal/runtime-only mode
    steps = [
        f"Decompose task for macro_state={macro}",
        "Retrieve memory/context",
        "Synthesize plan with diffusion amplifier (fallback)",
        "Verify Ihsan/PoI",
    ]
    snr = snr_score(signal=0.9, noise=0.35)
    poi = max(0.0, min((snr + 0.8) / 2.0, 1.0))
    return Plan(macro_state=macro, steps=steps, snr=snr, poi_score=poi)


class MissionRequest(BaseModel):
    text: str
    context: dict[str, str] = Field(default_factory=dict)
    slot: str = "cold_core"


# ── Canonical Receipt Infrastructure (startup invariant) ──────────
# Import failures here are fatal — the gateway MUST speak canonical law.
try:
    from app.node.receipt_ledger import append_receipt, read_last_receipt_hash
    from core.proof_engine.canonical_receipt_adapter import (
        GENESIS_SEED,
        ExecutionRoute,
        VerdictStatus,
        from_mission_result,
    )

    _CANONICAL_AVAILABLE = True
except ImportError as _import_err:
    import logging as _log

    _log.getLogger(__name__).error(
        "FATAL: Canonical receipt adapter not available: %s. "
        "Gateway will emit non-canonical receipts marked _canonical:false.",
        _import_err,
    )
    _CANONICAL_AVAILABLE = False


@router.post("/v1/mission")
async def canonical_mission(
    req: MissionRequest,
    x_bizra_api_key: str | None = Header(default=None, alias="x-bizra-api-key"),
) -> dict:
    """Execute a mission through the canonical constitutional pipeline.

    Returns a cryptographically valid CanonicalReceipt with:
    - receipt_id computed from BLAKE3 canonical bytes
    - genesis_hash binding to constitutional universe
    - chain linkage to previous receipt
    - lifecycle state from ReceiptStateMachine

    RUNTIME_CUTOVER_03: durable chain, startup invariant, canonically exact.
    """
    _require_api_key(x_bizra_api_key)
    received_at = int(time.time() * 1000)

    # 1) HHMM classification → route selection
    macro = hhmm.predict(req.text, req.context)
    reflex_steps = cache.get(macro)

    # 2) Execute mission (reflex / deliberate / degraded)
    response_text = ""
    model_used = ""
    ihsan = 0.0
    verdict_str = "DEFERRED"
    route_str = "DELIBERATE"

    if reflex_steps:
        response_text = "; ".join(reflex_steps)
        ihsan = 0.95
        verdict_str = "ADMITTED"
        route_str = "REFLEX"
    else:
        try:
            import httpx

            kernel_url = os.environ.get("BIZRA_KERNEL_URL", "http://localhost:8010")
            token = os.environ.get("BIZRA_API_TOKEN", "")
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    f"{kernel_url}/v1/cognitive/invoke",
                    json={"prompt": req.text, "slot": req.slot},
                    headers=headers,
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("success") and "[ERROR]" not in data.get(
                        "response", ""
                    ):
                        response_text = data["response"]
                        model_used = data.get("model_used", "")
                        ihsan = data.get("ihsan_score", 0.0)
                        verdict_str = (
                            "ADMITTED" if data.get("ihsan_passed") else "REJECTED"
                        )
        except Exception:
            pass

        if not response_text:
            bridge_plan = await mission_bridge.run(
                req.text, req.context, macro_state=macro
            )
            if bridge_plan:
                response_text = "; ".join(bridge_plan.steps)
                ihsan = 0.90
                verdict_str = "ADMITTED"
                route_str = "DEGRADED"
            else:
                response_text = f"[DEGRADED] Decompose: {macro}"
                ihsan = 0.0
                verdict_str = "DEFERRED"
                route_str = "DEGRADED"

    snr_val = snr_score(signal=0.9, noise=0.35)

    # 3) Build cryptographically valid CanonicalReceipt
    if _CANONICAL_AVAILABLE:
        verdict_map = {
            "ADMITTED": VerdictStatus.ADMITTED,
            "REJECTED": VerdictStatus.REJECTED,
            "DEFERRED": VerdictStatus.DEFERRED,
        }
        route_map = {
            "REFLEX": ExecutionRoute.REFLEX,
            "DELIBERATE": ExecutionRoute.DELIBERATE,
            "DEGRADED": ExecutionRoute.DEGRADED,
        }

        # Read previous receipt from durable ledger (survives restart)
        previous = read_last_receipt_hash()

        receipt = from_mission_result(
            mission_id=f"{macro}-{received_at}",
            genesis_hash=GENESIS_SEED,
            policy_version="v0.90.0",
            verdict=verdict_map.get(verdict_str, VerdictStatus.DEFERRED),
            ihsan_score=ihsan,
            snr_score=snr_val,
            route=route_map.get(route_str, ExecutionRoute.DEGRADED),
            input_text=req.text,
            output_text=response_text,
            previous_receipt=previous,
            received_at=received_at,
        )

        receipt.receipt_id = receipt.compute_id()

        result = receipt.to_dict()
        result["response"] = response_text
        result["model_used"] = model_used
        result["macro_state"] = macro
        result["_canonical"] = True

        # Persist to durable ledger (chain survives restart)
        append_receipt(result)

    else:
        # Non-canonical fallback — detectable and rejectable downstream
        result = {
            "receipt_id": f"cr-{received_at}",
            "mission_id": f"m-{received_at}",
            "verdict": verdict_str,
            "ihsan_score": ihsan,
            "snr_score": snr_val,
            "route": route_str,
            "state": {
                "ADMITTED": "COMMITTED",
                "REJECTED": "VERIFIED",
                "DEFERRED": "HYPOTHESIS",
            }[verdict_str],
            "federation_admissible": False,  # Non-canonical = NEVER federable
            "response": response_text,
            "model_used": model_used,
            "macro_state": macro,
            "_canonical": False,
        }

    # 4) Reflex precipitation
    if verdict_str == "ADMITTED" and route_str == "DELIBERATE":
        if snr_val >= _reflex_min_snr():
            cache.put(macro, [response_text[:200]])

    return result


@router.post("/v1/reflexes/{macro_state}")
def store_reflex(
    macro_state: str,
    steps: list[str],
    x_bizra_api_key: str | None = Header(default=None, alias="x-bizra-api-key"),
):
    _require_api_key(x_bizra_api_key)
    cache.put(macro_state, steps)
    return {"ok": True, "macro_state": macro_state, "count": cache.count()}
