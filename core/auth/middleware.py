"""
Auth Middleware — FastAPI Integration for Multi-User Auth
=========================================================
Phase 21: Multi-User Foundation

Provides:
- FastAPI Depends() for route-level authentication
- Bearer token extraction (JWT) + API key fallback
- Per-user rate limiting
- Request-scoped user context

Usage with FastAPI:
    from core.auth.middleware import AuthMiddleware, get_current_user

    auth = AuthMiddleware(user_store=store, jwt_auth=jwt)

    @app.post("/v1/query")
    async def query(user = Depends(auth.require_auth)):
        # user is a UserRecord
        ...

Standing on Giants:
- OWASP API Security Top 10 (2023)
- RFC 6750 (Bearer Token Usage)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("bizra.auth.middleware")


class AuthMiddleware:
    """
    Authentication middleware for FastAPI and raw asyncio API server.

    Supports two authentication methods:
    1. JWT Bearer token (preferred) — `Authorization: Bearer <jwt>`
    2. API key header (legacy compat) — `X-API-Key: bzr_<hex>`

    Also provides per-user rate limiting.
    """

    def __init__(
        self,
        user_store: Any,  # UserStore
        jwt_auth: Any,  # JWTAuth
        rate_limit_per_minute: int = 100,
        burst_size: int = 10,
    ):
        self.user_store = user_store
        self.jwt_auth = jwt_auth
        self._rate = rate_limit_per_minute / 60.0
        self._burst = burst_size
        self._buckets: dict[str, dict[str, float]] = {}

    def authenticate(
        self,
        authorization: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Authenticate a request from explicit header values.

        Tries JWT Bearer first, then falls back to API key.
        Returns UserRecord if valid, None if not.
        """
        # Method 1: JWT Bearer token
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            claims = self.jwt_auth.verify_token(token, expected_type="access")
            if claims:
                user = self.user_store.get_by_id(claims.sub)
                if user and user.is_active:
                    return user

        # Method 2: API key header
        if api_key:
            user = self.user_store.verify_api_key(api_key)
            if user and user.is_active:
                return user

        return None

    def authenticate_request(self, request: Any) -> Optional[Any]:
        """
        Authenticate from a FastAPI/ASGI Request object.

        Extracts Authorization and X-API-Key headers, then delegates
        to authenticate(). This is the correct entry point for route
        handlers that receive a Request object.

        Returns UserRecord if valid, None if not.
        """
        authorization = None
        api_key = None
        # Support FastAPI Request (has .headers dict-like)
        headers = getattr(request, "headers", None)
        if headers is not None:
            authorization = headers.get("authorization")
            api_key = headers.get("x-api-key")
        return self.authenticate(authorization=authorization, api_key=api_key)

    def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user is within rate limits.

        Returns True if allowed, False if throttled.
        Uses token bucket algorithm (same as existing RateLimiter).
        """
        now = time.time()

        if user_id not in self._buckets:
            self._buckets[user_id] = {"tokens": float(self._burst), "last": now}
            return True

        bucket = self._buckets[user_id]
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(self._burst, bucket["tokens"] + elapsed * self._rate)
        bucket["last"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

    def rate_limit_remaining(self, user_id: str) -> int:
        """Get remaining rate limit tokens for a user."""
        if user_id not in self._buckets:
            return self._burst
        return int(self._buckets[user_id]["tokens"])


# ==============================================================================
# FASTAPI DEPENDENCY INJECTION
# ==============================================================================

# Global middleware instance (set during app initialization)
_global_middleware: Optional[AuthMiddleware] = None


def init_auth_middleware(middleware: AuthMiddleware) -> None:
    """Initialize the global auth middleware instance."""
    global _global_middleware
    _global_middleware = middleware


async def get_current_user(
    authorization: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> Any:
    """
    FastAPI dependency for authenticated routes.

    Usage:
        @app.post("/v1/query")
        async def query(user = Depends(get_current_user)):
            ...

    For FastAPI, the Header parameters are automatically extracted.
    For raw asyncio server, call authenticate() directly.
    """
    if _global_middleware is None:
        # Auth not configured — allow anonymous access (dev mode)
        logger.warning("Auth middleware not initialized — anonymous access allowed")
        return None

    user = _global_middleware.authenticate(
        authorization=authorization,
        api_key=x_api_key,
    )

    if user is None:
        # In FastAPI context, raise HTTPException
        try:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except ImportError:
            return None

    # Check rate limit
    if not _global_middleware.check_rate_limit(user.user_id):
        try:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Remaining": "0",
                },
            )
        except ImportError:
            return None

    return user
