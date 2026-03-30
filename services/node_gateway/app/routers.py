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


class CanonicalReceiptResponse(BaseModel):
    receipt_id: str
    mission_id: str
    verdict: str
    ihsan_score: float
    snr_score: float
    route: str
    state: str
    federation_admissible: bool
    response: str
    model_used: str = ""


@router.post("/v1/mission")
async def canonical_mission(
    req: MissionRequest,
    x_bizra_api_key: str | None = Header(default=None, alias="x-bizra-api-key"),
) -> CanonicalReceiptResponse:
    """Execute a mission through the canonical constitutional pipeline.

    Returns a CanonicalReceipt-shaped response. Every externally visible
    effect produces exactly one receipt.
    """
    _require_api_key(x_bizra_api_key)
    received_at = int(time.time() * 1000)

    # 1) HHMM classification → route selection
    macro = hhmm.predict(req.text, req.context)
    reflex_steps = cache.get(macro)
    route = "REFLEX" if reflex_steps else "DELIBERATE"

    # 2) Try reflex path first (System 1)
    if reflex_steps:
        snr = snr_score(signal=0.9, noise=0.35)
        return CanonicalReceiptResponse(
            receipt_id=f"rx-{received_at}",
            mission_id=f"m-{received_at}",
            verdict="ADMITTED",
            ihsan_score=0.95,
            snr_score=snr,
            route="REFLEX",
            state="COMMITTED",
            federation_admissible=True,
            response="; ".join(reflex_steps),
        )

    # 3) Deliberate path — call kernel cognitive engine
    response_text = ""
    model_used = ""
    ihsan = 0.0
    verdict = "DEFERRED"

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
                if data.get("success") and "[ERROR]" not in data.get("response", ""):
                    response_text = data["response"]
                    model_used = data.get("model_used", "")
                    ihsan = data.get("ihsan_score", 0.0)
                    verdict = "ADMITTED" if data.get("ihsan_passed") else "REJECTED"
    except Exception:
        pass

    # 4) Fallback if kernel unreachable
    if not response_text:
        bridge_plan = await mission_bridge.run(req.text, req.context, macro_state=macro)
        if bridge_plan:
            response_text = "; ".join(bridge_plan.steps)
            ihsan = 0.90
            verdict = "ADMITTED"
            route = "DEGRADED"
        else:
            response_text = f"[DEGRADED] Decompose: {macro}"
            ihsan = 0.0
            verdict = "DEFERRED"
            route = "DEGRADED"

    snr = snr_score(signal=0.9, noise=0.35)
    federation = verdict == "ADMITTED" and ihsan >= 0.95

    # 5) Emit canonical receipt
    state = {"ADMITTED": "COMMITTED", "REJECTED": "VERIFIED", "DEFERRED": "HYPOTHESIS"}[
        verdict
    ]

    # Cache compilation: if deliberate succeeded, consider reflex precipitation
    if verdict == "ADMITTED" and route == "DELIBERATE" and snr >= _reflex_min_snr():
        cache.put(macro, [response_text[:200]])

    return CanonicalReceiptResponse(
        receipt_id=f"cr-{received_at}",
        mission_id=f"m-{received_at}",
        verdict=verdict,
        ihsan_score=ihsan,
        snr_score=snr,
        route=route,
        state=state,
        federation_admissible=federation,
        response=response_text,
        model_used=model_used,
    )


@router.post("/v1/reflexes/{macro_state}")
def store_reflex(
    macro_state: str,
    steps: list[str],
    x_bizra_api_key: str | None = Header(default=None, alias="x-bizra-api-key"),
):
    _require_api_key(x_bizra_api_key)
    cache.put(macro_state, steps)
    return {"ok": True, "macro_state": macro_state, "count": cache.count()}
