from app.auth import require_admin
from fastapi import APIRouter, Header
from pydantic import BaseModel

try:
    from _shared.app.health import build_health_payload, check_redis
    from _shared.app.persistence import JsonHashStore
except Exception:  # pragma: no cover - local import fallback
    from services._shared.app.health import build_health_payload, check_redis
    from services._shared.app.persistence import JsonHashStore

router = APIRouter()
APP_VERSION = "4.0.1"


class CrownSignature(BaseModel):
    ui_surface: str
    entropy_baseline: float
    signature: str  # placeholder


_STORE = JsonHashStore("bizra:urp_verification:crown")


@router.get("/health")
def health():
    return build_health_payload(
        service="urp_verification",
        version=APP_VERSION,
        checks={"redis": check_redis()},
        extra={"store_backend": _STORE.backend(), "records": _STORE.count()},
    )


@router.get("/v1/crown")
def list_crown():
    items = [CrownSignature.model_validate(v) for v in _STORE.values()]
    return {"baselines": items}


@router.post("/v1/crown", status_code=201)
def upsert_crown(c: CrownSignature, x_urp_admin: str | None = Header(default=None)):
    require_admin(x_urp_admin)
    _STORE.upsert(c.ui_surface, c.model_dump())
    return c
