from fastapi import APIRouter, Header
from pydantic import BaseModel, Field
from app.auth import require_admin

try:
    from _shared.app.health import build_health_payload, check_redis
    from _shared.app.persistence import JsonHashStore
except Exception:  # pragma: no cover - local import fallback
    from services._shared.app.health import build_health_payload, check_redis
    from services._shared.app.persistence import JsonHashStore

router = APIRouter()
APP_VERSION = "4.0.1"

class ModelRecord(BaseModel):
    model_id: str
    capability_embedding: list[float] = Field(default_factory=list)
    skills_bloom_filter: str | None = None
    latency_budget_ms: int | None = None
    cost_per_1m_tokens_usd: float | None = None
    policy_compliance: list[str] = Field(default_factory=list)
    version: str | None = None

_STORE = JsonHashStore("bizra:urp_registry:models")

@router.get("/health")
def health():
    return build_health_payload(
        service="urp_registry",
        version=APP_VERSION,
        checks={"redis": check_redis()},
        extra={"store_backend": _STORE.backend(), "records": _STORE.count()},
    )

@router.get("/v1/models")
def list_models():
    items = [ModelRecord.model_validate(v) for v in _STORE.values()]
    return {"models": items}

@router.post("/v1/models", status_code=201)
def register_model(rec: ModelRecord, x_urp_admin: str | None = Header(default=None)):
    require_admin(x_urp_admin)
    _STORE.upsert(rec.model_id, rec.model_dump())
    return rec
