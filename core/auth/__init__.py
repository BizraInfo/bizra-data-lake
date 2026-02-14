"""
BIZRA Auth Module — Multi-User Identity Foundation
===================================================
Phase 21: Multi-User Foundation

Provides:
- User registration with Ed25519 keypair generation
- Password hashing (argon2id via PBKDF2-SHA256 fallback)
- JWT token issuance and verification
- Per-user data isolation (namespace-based)
- API key management (CRUD + rotation)

Standing on Giants:
- NIST SP 800-63B (Digital Identity Guidelines)
- RFC 7519 (JSON Web Tokens)
- Bernstein (2006): Ed25519 signatures
"""

from .jwt_auth import JWTAuth, TokenPair
from .middleware import AuthMiddleware, get_current_user
from .user_store import UserRecord, UserStore

__all__ = [
    "UserStore",
    "UserRecord",
    "JWTAuth",
    "TokenPair",
    "AuthMiddleware",
    "get_current_user",
]
