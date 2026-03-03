import hmac
import os
from fastapi import HTTPException


def _resolve_admin_token() -> str:
    # Canonical name: URP_ADMIN_TOKEN. Keep URP_ADMIN_KEY for backward compatibility.
    return (os.environ.get("URP_ADMIN_TOKEN") or os.environ.get("URP_ADMIN_KEY") or "").strip()


def require_admin(x_urp_admin: str | None) -> None:
    token = _resolve_admin_token()
    if not token:
        raise HTTPException(
            status_code=500,
            detail="URP_ADMIN_TOKEN not set",
        )
    if not isinstance(x_urp_admin, str) or not hmac.compare_digest(x_urp_admin, token):
        raise HTTPException(status_code=401, detail="unauthorized")
