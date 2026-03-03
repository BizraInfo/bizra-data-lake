import hmac
import os
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from app.node.hhmm import HHMM
from app.node.snr import snr_score
from app.node.reflex_cache import ReflexCache

try:
    from _shared.app.health import build_health_payload, check_http, check_redis
except Exception:  # pragma: no cover - local import fallback
    from services._shared.app.health import build_health_payload, check_http, check_redis

router = APIRouter()
APP_VERSION = "4.0.1"

hhmm = HHMM()
cache = ReflexCache()

class Observation(BaseModel):
    text: str
    context: dict[str, str] = Field(default_factory=dict)

class Plan(BaseModel):
    macro_state: str
    steps: list[str]
    snr: float
    poi_score: float


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
def plan(
    obs: Observation,
    x_bizra_api_key: str | None = Header(default=None, alias="x-bizra-api-key"),
) -> Plan:
    _require_api_key(x_bizra_api_key)
    # 1) Predict macro-state (HHMM)
    macro = hhmm.predict(obs.text, obs.context)
    # 2) Try reflex cache (O(1) lookup)
    steps = cache.get(macro) or [
        f"Decompose task for macro_state={macro}",
        "Retrieve memory/context",
        "Synthesize plan with diffusion amplifier (stub)",
        "Verify Ihsan/PoI",
    ]
    # 3) Compute SNR (stub)
    snr = snr_score(signal=0.9, noise=0.35)
    # 4) Compute PoI placeholder
    poi = 0.86 if snr >= 2.0 else 0.78
    return Plan(macro_state=macro, steps=steps, snr=snr, poi_score=poi)

@router.post("/v1/reflexes/{macro_state}")
def store_reflex(
    macro_state: str,
    steps: list[str],
    x_bizra_api_key: str | None = Header(default=None, alias="x-bizra-api-key"),
):
    _require_api_key(x_bizra_api_key)
    cache.put(macro_state, steps)
    return {"ok": True, "macro_state": macro_state, "count": cache.count()}
