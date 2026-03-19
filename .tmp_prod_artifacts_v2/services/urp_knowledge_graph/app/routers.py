from app.auth import require_admin
from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

try:
    from _shared.app.health import build_health_payload, check_redis
    from _shared.app.persistence import JsonHashStore
except Exception:  # pragma: no cover - local import fallback
    from services._shared.app.health import build_health_payload, check_redis
    from services._shared.app.persistence import JsonHashStore

router = APIRouter()
APP_VERSION = "4.0.1"


class ReflexPattern(BaseModel):
    reflex_id: str
    hhmm_macro_state: str
    abstract_steps: list[str] = Field(default_factory=list)
    privacy_level: str = "local-only"
    poi_consensus: float | None = None
    avg_latency_ms: float | None = None
    ihsan_score: float | None = None


_STORE = JsonHashStore("bizra:urp_knowledge_graph:reflexes")


@router.get("/health")
def health():
    return build_health_payload(
        service="urp_knowledge_graph",
        version=APP_VERSION,
        checks={"redis": check_redis()},
        extra={"store_backend": _STORE.backend(), "records": _STORE.count()},
    )


@router.get("/v1/reflexes")
def list_reflexes():
    items = [ReflexPattern.model_validate(v) for v in _STORE.values()]
    return {"reflexes": items}


@router.post("/v1/reflexes", status_code=201)
def publish_reflex(p: ReflexPattern, x_urp_admin: str | None = Header(default=None)):
    require_admin(x_urp_admin)
    _STORE.upsert(p.reflex_id, p.model_dump())
    return p
