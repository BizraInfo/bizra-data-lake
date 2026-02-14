"""
User Store — SQLite-backed User Identity Management
====================================================
Phase 21: Multi-User Foundation

Handles:
- User registration with PBKDF2-SHA256 password hashing
- Ed25519 keypair generation per user
- API key generation and rotation
- Per-user namespace/data isolation paths
- User lookup, verification, and lifecycle management

Schema Design:
    users(
        user_id TEXT PRIMARY KEY,       -- UUID4
        username TEXT UNIQUE NOT NULL,   -- display name
        email TEXT UNIQUE NOT NULL,      -- verification target
        password_hash TEXT NOT NULL,     -- PBKDF2-SHA256(600K iterations)
        ed25519_public_key TEXT NOT NULL,-- hex-encoded public key
        ed25519_secret_key_enc TEXT NOT NULL, -- vault-encrypted secret key
        api_key TEXT UNIQUE NOT NULL,    -- bearer token for API access
        status TEXT DEFAULT 'active',   -- active | suspended | revoked
        covenant_accepted BOOLEAN DEFAULT FALSE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_login_at TEXT,
        query_count INTEGER DEFAULT 0,
        metadata TEXT DEFAULT '{}'      -- JSON blob for extensions
    )

Standing on Giants:
- NIST SP 800-132 (PBKDF2 recommendation)
- OWASP Password Storage Cheat Sheet (600K iterations for SHA-256)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("bizra.auth.user_store")

# ==============================================================================
# CONSTANTS
# ==============================================================================

# PBKDF2 iterations (OWASP 2024 recommendation for SHA-256)
PBKDF2_ITERATIONS = 600_000
PBKDF2_HASH_NAME = "sha256"
SALT_BYTES = 32
API_KEY_BYTES = 32  # 256-bit API keys

# User status values
STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "suspended"
STATUS_REVOKED = "revoked"
STATUS_PENDING = "pending"  # awaiting email verification

# ==============================================================================
# DATA TYPES
# ==============================================================================


@dataclass
class UserRecord:
    """Represents a registered user in the BIZRA network."""

    user_id: str
    username: str
    email: str
    password_hash: str  # format: "pbkdf2:sha256:600000$<salt>$<hash>"
    ed25519_public_key: str  # hex-encoded
    ed25519_secret_key_enc: str  # vault-encrypted hex
    api_key: str
    status: str = STATUS_ACTIVE
    covenant_accepted: bool = False
    created_at: str = ""
    updated_at: str = ""
    last_login_at: Optional[str] = None
    query_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == STATUS_ACTIVE

    @property
    def namespace(self) -> str:
        """Per-user data isolation namespace."""
        return f"user_{self.user_id[:8]}"

    def to_safe_dict(self) -> dict[str, Any]:
        """Serialize without secrets (for API responses)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "ed25519_public_key": self.ed25519_public_key,
            "status": self.status,
            "covenant_accepted": self.covenant_accepted,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "query_count": self.query_count,
        }


# ==============================================================================
# PASSWORD HASHING (PBKDF2-SHA256, 600K iterations)
# ==============================================================================


def hash_password(password: str) -> str:
    """
    Hash password using PBKDF2-SHA256 with 600K iterations.

    Returns format: "pbkdf2:sha256:600000$<salt_hex>$<hash_hex>"
    Standing on: OWASP Password Storage Cheat Sheet (2024)
    """
    salt = os.urandom(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_NAME,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2:{PBKDF2_HASH_NAME}:{PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify password against stored hash.

    Timing-safe comparison to prevent timing attacks.
    """
    try:
        parts = password_hash.split("$")
        if len(parts) != 3:
            return False

        header = parts[0]  # "pbkdf2:sha256:600000"
        salt_hex = parts[1]
        stored_hash_hex = parts[2]

        header_parts = header.split(":")
        if len(header_parts) != 3:
            return False

        hash_name = header_parts[1]
        iterations = int(header_parts[2])
        salt = bytes.fromhex(salt_hex)

        dk = hashlib.pbkdf2_hmac(
            hash_name,
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(dk.hex(), stored_hash_hex)
    except (ValueError, IndexError):
        return False


def generate_api_key() -> str:
    """Generate a cryptographically random API key."""
    return f"bzr_{secrets.token_hex(API_KEY_BYTES)}"


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return str(uuid.uuid4())


# ==============================================================================
# ED25519 KEY GENERATION
# ==============================================================================


def generate_ed25519_keypair() -> tuple[str, str]:
    """
    Generate Ed25519 keypair for user identity.

    Returns (public_key_hex, secret_key_hex).
    Standing on: Bernstein (2006) — Ed25519 high-speed signatures.
    """
    try:
        from nacl.signing import SigningKey

        sk = SigningKey.generate()
        pk = sk.verify_key
        return pk.encode().hex(), sk.encode().hex()
    except ImportError:
        # Fallback: use cryptography library
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
            from cryptography.hazmat.primitives.serialization import (
                Encoding,
                NoEncryption,
                PrivateFormat,
                PublicFormat,
            )

            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key()

            pk_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
            sk_bytes = private_key.private_bytes(
                Encoding.Raw, PrivateFormat.Raw, NoEncryption()
            )
            return pk_bytes.hex(), sk_bytes.hex()
        except ImportError:
            # Last resort: use os.urandom as placeholder (NOT production-safe)
            logger.warning(
                "Neither PyNaCl nor cryptography available — "
                "generating placeholder keypair (NOT PRODUCTION SAFE)"
            )
            fake_sk = os.urandom(32)
            fake_pk = os.urandom(32)
            return fake_pk.hex(), fake_sk.hex()


# ==============================================================================
# USER STORE
# ==============================================================================


class UserStore:
    """
    SQLite-backed user identity store.

    Thread-safe via SQLite WAL mode + connection-per-operation pattern.
    Each user gets a namespace for data isolation.

    Usage:
        store = UserStore(db_path=Path("sovereign_state/users.db"))
        user = store.register("mumu", "mumu@bizra.ai", "strong_password_123")
        verified = store.verify_login("mumu", "strong_password_123")
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("sovereign_state/users.db")
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode for concurrent reads."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    ed25519_public_key TEXT NOT NULL,
                    ed25519_secret_key_enc TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    status TEXT DEFAULT 'active',
                    covenant_accepted BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT,
                    query_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    label TEXT DEFAULT 'default',
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            # Indexes for fast lookup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(api_key)"
            )
            conn.commit()

    # --------------------------------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------------------------------

    def register(
        self,
        username: str,
        email: str,
        password: str,
        accept_covenant: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UserRecord:
        """
        Register a new user.

        1. Validate inputs
        2. Hash password (PBKDF2-SHA256, 600K iterations)
        3. Generate Ed25519 keypair
        4. Generate API key
        5. Store in database
        6. Return UserRecord (without secret key)

        Raises:
            ValueError: If username/email already taken or validation fails.
        """
        # Input validation
        username = username.strip()
        email = email.strip().lower()

        if not username or len(username) < 2 or len(username) > 64:
            raise ValueError("Username must be 2-64 characters")
        if not email or "@" not in email or len(email) > 254:
            raise ValueError("Invalid email address")
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        # Generate identity
        now = datetime.now(timezone.utc).isoformat()
        user_id = generate_user_id()
        pwd_hash = hash_password(password)
        public_key, secret_key = generate_ed25519_keypair()
        api_key = generate_api_key()

        # Encrypt secret key at rest (simplified — production would use SovereignVault)
        # For now, store hex-encoded with a marker for future vault integration
        sk_enc = f"vault:v1:{secret_key}"

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, username, email, password_hash,
                        ed25519_public_key, ed25519_secret_key_enc,
                        api_key, status, covenant_accepted,
                        created_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        username,
                        email,
                        pwd_hash,
                        public_key,
                        sk_enc,
                        api_key,
                        STATUS_ACTIVE,
                        int(accept_covenant),
                        now,
                        now,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
        except sqlite3.IntegrityError as e:
            error_msg = str(e).lower()
            if "username" in error_msg:
                raise ValueError(f"Username '{username}' already taken")
            elif "email" in error_msg:
                raise ValueError(f"Email '{email}' already registered")
            else:
                raise ValueError(f"Registration failed: {e}")

        logger.info(f"User registered: {username} ({user_id[:8]})")

        return UserRecord(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=pwd_hash,
            ed25519_public_key=public_key,
            ed25519_secret_key_enc="[encrypted]",  # Don't return raw key
            api_key=api_key,
            status=STATUS_ACTIVE,
            covenant_accepted=accept_covenant,
            created_at=now,
            updated_at=now,
        )

    # --------------------------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------------------------

    def verify_login(self, username: str, password: str) -> Optional[UserRecord]:
        """
        Verify username + password. Returns UserRecord if valid, None if not.

        Updates last_login_at on success.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? AND status = ?",
                (username.strip(), STATUS_ACTIVE),
            ).fetchone()

        if row is None:
            return None

        if not verify_password(password, row["password_hash"]):
            return None

        # Update last login
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE user_id = ?",
                (now, now, row["user_id"]),
            )
            conn.commit()

        return self._row_to_record(row)

    def verify_api_key(self, api_key: str) -> Optional[UserRecord]:
        """
        Verify an API key. Returns the associated UserRecord if valid.

        Checks both the primary user api_key and the api_keys table.
        """
        with self._connect() as conn:
            # Check primary API key
            row = conn.execute(
                "SELECT * FROM users WHERE api_key = ? AND status = ?",
                (api_key, STATUS_ACTIVE),
            ).fetchone()

            if row:
                return self._row_to_record(row)

            # Check api_keys table (secondary keys)
            key_row = conn.execute(
                """
                SELECT u.* FROM users u
                JOIN api_keys ak ON u.user_id = ak.user_id
                WHERE ak.api_key = ? AND ak.status = 'active' AND u.status = ?
                """,
                (api_key, STATUS_ACTIVE),
            ).fetchone()

            if key_row:
                # Update last_used_at
                conn.execute(
                    "UPDATE api_keys SET last_used_at = ? WHERE api_key = ?",
                    (datetime.now(timezone.utc).isoformat(), api_key),
                )
                conn.commit()
                return self._row_to_record(key_row)

        return None

    # --------------------------------------------------------------------------
    # LOOKUP
    # --------------------------------------------------------------------------

    def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Look up user by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_username(self, username: str) -> Optional[UserRecord]:
        """Look up user by username."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username.strip(),)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_email(self, email: str) -> Optional[UserRecord]:
        """Look up user by email."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_users(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UserRecord]:
        """list users with optional status filter."""
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM users WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_users(self, status: Optional[str] = None) -> int:
        """Count users with optional status filter."""
        with self._connect() as conn:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM users WHERE status = ?", (status,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
        return row["cnt"] if row else 0

    # --------------------------------------------------------------------------
    # MANAGEMENT
    # --------------------------------------------------------------------------

    def update_status(self, user_id: str, status: str) -> bool:
        """Update user status (active, suspended, revoked)."""
        if status not in (
            STATUS_ACTIVE,
            STATUS_SUSPENDED,
            STATUS_REVOKED,
            STATUS_PENDING,
        ):
            raise ValueError(f"Invalid status: {status}")

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET status = ?, updated_at = ? WHERE user_id = ?",
                (status, now, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def rotate_api_key(self, user_id: str) -> Optional[str]:
        """
        Rotate a user's primary API key. Returns the new key.

        The old key becomes invalid immediately.
        """
        new_key = generate_api_key()
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET api_key = ?, updated_at = ? WHERE user_id = ?",
                (new_key, now, user_id),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"API key rotated for user {user_id[:8]}")
                return new_key
        return None

    def increment_query_count(self, user_id: str) -> None:
        """Increment the query count for a user (called per API request)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET query_count = query_count + 1 WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()

    def create_secondary_api_key(
        self,
        user_id: str,
        label: str = "secondary",
        expires_in_days: Optional[int] = None,
    ) -> Optional[str]:
        """Create an additional API key for a user."""
        new_key = generate_api_key()
        key_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        expires_at = None
        if expires_in_days:
            from datetime import timedelta

            expires_at = (
                datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            ).isoformat()

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO api_keys (key_id, user_id, api_key, label, status, created_at, expires_at)
                    VALUES (?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (key_id, user_id, new_key, label, now, expires_at),
                )
                conn.commit()
                return new_key
        except sqlite3.IntegrityError:
            return None

    # --------------------------------------------------------------------------
    # DATA ISOLATION
    # --------------------------------------------------------------------------

    def get_user_data_dir(self, user_id: str, base_dir: Optional[Path] = None) -> Path:
        """
        Get the per-user data directory for isolation.

        Creates the directory if it doesn't exist.
        Pattern: <base_dir>/user_<user_id[:8]>/
        """
        base = base_dir or Path("sovereign_state/users")
        user_dir = base / f"user_{user_id[:8]}"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    # --------------------------------------------------------------------------
    # INTERNALS
    # --------------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> UserRecord:
        """Convert a database row to UserRecord."""
        return UserRecord(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            ed25519_public_key=row["ed25519_public_key"],
            ed25519_secret_key_enc="[encrypted]",  # Never expose raw encrypted key
            api_key=row["api_key"],
            status=row["status"],
            covenant_accepted=bool(row["covenant_accepted"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login_at=row["last_login_at"],
            query_count=row["query_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
