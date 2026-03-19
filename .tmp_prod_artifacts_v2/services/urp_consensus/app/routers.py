from app.auth import require_admin
from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

try:
    from _shared.app.health import build_health_payload, check_redis
    from _shared.app.persistence import JsonListStore
except Exception:  # pragma: no cover - local import fallback
    from services._shared.app.health import build_health_payload, check_redis
    from services._shared.app.persistence import JsonListStore

router = APIRouter()
APP_VERSION = "4.0.1"


class PoISubmission(BaseModel):
    reflex_id: str
    node_id: str
    poi_score: float
    ihsan_tensor: dict[str, float] = Field(default_factory=dict)
    signature: str | None = None  # placeholder


_STORE = JsonListStore("bizra:urp_consensus:poi")


@router.get("/health")
def health():
    return build_health_payload(
        service="urp_consensus",
        version=APP_VERSION,
        checks={"redis": check_redis()},
        extra={"store_backend": _STORE.backend(), "records": _STORE.count()},
    )


@router.post("/v1/poi", status_code=201)
def submit_poi(s: PoISubmission, x_urp_admin: str | None = Header(default=None)):
    # In production: verify signature, rate-limit, and store in DB.
    require_admin(x_urp_admin)
    _STORE.append(s.model_dump())
    return {"accepted": True, "count": _STORE.count()}
