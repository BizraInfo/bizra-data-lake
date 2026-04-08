"""
JWT Authentication — Token Issuance and Verification
=====================================================
Phase 21: Multi-User Foundation

Implements JWT (RFC 7519) token management:
- Access tokens (short-lived, 15 min default)
- Refresh tokens (long-lived, 7 days default)
- HMAC-SHA256 signing (symmetric, for single-node)
- Token blacklisting for logout/revocation

Design note:
    For single-node deployment, HMAC-SHA256 is appropriate.
    For multi-node federation, switch to Ed25519 (asymmetric) so
    any node can verify tokens without sharing the secret.

Standing on Giants:
- RFC 7519 (JSON Web Tokens)
- OWASP JWT Cheat Sheet
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("bizra.auth.jwt")


def _production_mode_enabled() -> bool:
    """Return True when the runtime is explicitly in production mode."""
    return os.environ.get("BIZRA_ENV", "").strip().lower() == "production"


# ==============================================================================
# CONSTANTS
# ==============================================================================

ACCESS_TOKEN_EXPIRY = 15 * 60  # 15 minutes
REFRESH_TOKEN_EXPIRY = 7 * 24 * 3600  # 7 days
ALGORITHM = "HS256"

# ==============================================================================
# DATA TYPES
# ==============================================================================


@dataclass
class TokenPair:
    """Access + refresh token pair issued on login."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRY

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }


@dataclass
class TokenClaims:
    """Decoded JWT claims."""

    sub: str  # user_id
    username: str
    iat: int  # issued at (unix timestamp)
    exp: int  # expiration (unix timestamp)
    typ: str = "access"  # "access" or "refresh"
    jti: str = ""  # JWT ID (for blacklisting)


# ==============================================================================
# JWT ENGINE (Minimal, no external deps)
# ==============================================================================


class JWTAuth:
    """
    JWT token manager using HMAC-SHA256.

    No external JWT library required — pure Python implementation.
    This keeps the dependency footprint minimal while remaining
    fully standards-compliant for single-node deployment.

    Usage:
        auth = JWTAuth(secret="your-256-bit-secret")
        tokens = auth.issue_tokens(user_id="abc", username="mumu")
        claims = auth.verify_token(tokens.access_token)
    """

    def __init__(
        self,
        secret: Optional[str] = None,
        access_expiry: int = ACCESS_TOKEN_EXPIRY,
        refresh_expiry: int = REFRESH_TOKEN_EXPIRY,
        db_path: Optional[Path] = None,
    ):
        # Generate secret if not provided
        self._secret = (secret or os.environ.get("BIZRA_JWT_SECRET", "")).encode(
            "utf-8"
        )
        if not self._secret:
            if _production_mode_enabled():
                raise ValueError("BIZRA_JWT_SECRET is required in production")
            self._secret = os.urandom(32)
            logger.warning(
                "JWT secret auto-generated. set BIZRA_JWT_SECRET env var for persistence."
            )

        self._access_expiry = access_expiry
        self._refresh_expiry = refresh_expiry

        # Token blacklist — SQLite-backed with in-memory cache.
        # Bounded: stores (jti, expiry) tuples; evicts expired entries
        # when size exceeds _BLACKLIST_MAX to prevent unbounded growth (SAPE-011).
        self._blacklist: dict[str, int] = {}  # jti -> expiry timestamp (cache)
        self._BLACKLIST_MAX = 10_000
        self._db_path = db_path or Path("sovereign_state/jwt_blacklist.db")
        self._ensure_blacklist_schema()
        self._load_blacklist_from_db()

    # --------------------------------------------------------------------------
    # TOKEN ISSUANCE
    # --------------------------------------------------------------------------

    def issue_tokens(self, user_id: str, username: str) -> TokenPair:
        """
        Issue an access + refresh token pair.

        Access tokens are short-lived (15 min) for API requests.
        Refresh tokens are long-lived (7 days) for re-authentication.
        """
        now = int(time.time())

        access_token = self._encode(
            {
                "sub": user_id,
                "username": username,
                "typ": "access",
                "iat": now,
                "exp": now + self._access_expiry,
                "jti": os.urandom(8).hex(),
            }
        )

        refresh_token = self._encode(
            {
                "sub": user_id,
                "username": username,
                "typ": "refresh",
                "iat": now,
                "exp": now + self._refresh_expiry,
                "jti": os.urandom(8).hex(),
            }
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self._access_expiry,
        )

    def refresh_access_token(self, refresh_token: str) -> Optional[TokenPair]:
        """
        Use a refresh token to get new access + refresh tokens.

        The old refresh token is blacklisted (single-use rotation).
        """
        claims = self.verify_token(refresh_token, expected_type="refresh")
        if claims is None:
            return None

        # Blacklist old refresh token
        self._blacklist_add(claims.jti, claims.exp)

        # Issue fresh pair
        return self.issue_tokens(claims.sub, claims.username)

    # --------------------------------------------------------------------------
    # TOKEN VERIFICATION
    # --------------------------------------------------------------------------

    def verify_token(
        self, token: str, expected_type: str = "access"
    ) -> Optional[TokenClaims]:
        """
        Verify and decode a JWT token.

        Checks:
        1. Signature validity (HMAC-SHA256)
        2. Expiration (exp claim)
        3. Token type (access vs refresh)
        4. Blacklist (revoked tokens)

        Returns TokenClaims if valid, None if invalid.
        """
        payload = self._decode(token)
        if payload is None:
            return None

        claims = TokenClaims(
            sub=payload.get("sub", ""),
            username=payload.get("username", ""),
            iat=payload.get("iat", 0),
            exp=payload.get("exp", 0),
            typ=payload.get("typ", "access"),
            jti=payload.get("jti", ""),
        )

        # Check expiration
        now = int(time.time())
        if claims.exp <= now:
            logger.debug(f"Token expired: {claims.jti}")
            return None

        # Check type
        if claims.typ != expected_type:
            logger.debug(
                f"Token type mismatch: expected {expected_type}, got {claims.typ}"
            )
            return None

        # Check blacklist
        if claims.jti in self._blacklist:
            logger.debug(f"Token blacklisted: {claims.jti}")
            return None

        return claims

    def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding its JTI to the blacklist."""
        payload = self._decode(token)
        if payload and "jti" in payload:
            self._blacklist_add(payload["jti"], payload.get("exp", 0))
            return True
        return False

    def _blacklist_add(self, jti: str, exp: int) -> None:
        """Add JTI to blacklist with bounded eviction (SAPE-011)."""
        self._blacklist[jti] = exp
        self._persist_blacklist_entry(jti, exp)
        if len(self._blacklist) > self._BLACKLIST_MAX:
            now = int(time.time())
            expired = [k for k, v in self._blacklist.items() if v <= now]
            for k in expired:
                del self._blacklist[k]
            self._purge_expired_from_db(now)
        overflow = len(self._blacklist) - self._BLACKLIST_MAX
        if overflow > 0:
            oldest_expiry = sorted(self._blacklist.items(), key=lambda item: item[1])
            for key, _ in oldest_expiry[:overflow]:
                del self._blacklist[key]

    # --------------------------------------------------------------------------
    # DURABLE BLACKLIST (SQLite-backed)
    # --------------------------------------------------------------------------

    def _blacklist_connect(self) -> sqlite3.Connection:
        """Create connection to blacklist DB with WAL mode."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_blacklist_schema(self) -> None:
        """Create blacklist table if it doesn't exist."""
        try:
            with self._blacklist_connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS token_blacklist (
                        jti TEXT PRIMARY KEY,
                        expires_at INTEGER NOT NULL,
                        revoked_at INTEGER NOT NULL
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_blacklist_expires "
                    "ON token_blacklist(expires_at)"
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(
                "Blacklist DB init failed (falling back to memory-only): %s", e
            )

    def _load_blacklist_from_db(self) -> None:
        """Load non-expired blacklist entries from SQLite into memory cache."""
        now = int(time.time())
        try:
            with self._blacklist_connect() as conn:
                rows = conn.execute(
                    "SELECT jti, expires_at FROM token_blacklist WHERE expires_at > ?",
                    (now,),
                ).fetchall()
                for jti, exp in rows:
                    self._blacklist[jti] = exp
                if rows:
                    logger.info(
                        "Loaded %d revoked tokens from durable blacklist", len(rows)
                    )
                # Purge expired while we're here
                conn.execute(
                    "DELETE FROM token_blacklist WHERE expires_at <= ?", (now,)
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning("Blacklist DB load failed (starting empty): %s", e)

    def _persist_blacklist_entry(self, jti: str, exp: int) -> None:
        """Persist a single blacklist entry to SQLite."""
        try:
            with self._blacklist_connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO token_blacklist (jti, expires_at, revoked_at) "
                    "VALUES (?, ?, ?)",
                    (jti, exp, int(time.time())),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning("Blacklist persist failed for jti=%s: %s", jti, e)

    def _purge_expired_from_db(self, now: int) -> None:
        """Remove expired entries from SQLite."""
        try:
            with self._blacklist_connect() as conn:
                conn.execute(
                    "DELETE FROM token_blacklist WHERE expires_at <= ?", (now,)
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning("Blacklist purge failed: %s", e)

    # --------------------------------------------------------------------------
    # JWT ENCODING / DECODING (RFC 7519 compliant)
    # --------------------------------------------------------------------------

    def _encode(self, payload: dict[str, Any]) -> str:
        """Encode payload to JWT string."""
        header = {"alg": ALGORITHM, "typ": "JWT"}

        header_b64 = self._b64url_encode(json.dumps(header).encode())
        payload_b64 = self._b64url_encode(json.dumps(payload).encode())

        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self._secret, signing_input.encode(), hashlib.sha256
        ).digest()
        signature_b64 = self._b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def _decode(self, token: str) -> Optional[dict[str, Any]]:
        """Decode and verify JWT token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            signing_input = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(
                self._secret, signing_input.encode(), hashlib.sha256
            ).digest()
            actual_sig = self._b64url_decode(signature_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                logger.debug("JWT signature verification failed")
                return None

            # Decode payload
            payload_bytes = self._b64url_decode(payload_b64)
            return json.loads(payload_bytes)

        except (ValueError, json.JSONDecodeError) as e:  # SEC-003
            logger.debug(f"JWT decode failed: {e}")
            return None

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        """Base64url encode (RFC 4648 §5)."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64url_decode(s: str) -> bytes:
        """Base64url decode with padding restoration."""
        padding = 4 - len(s) % 4
        if padding != 4:
            s += "=" * padding
        return base64.urlsafe_b64decode(s)
