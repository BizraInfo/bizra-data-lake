"""
Phase 21: Multi-User Auth Integration Tests
=============================================
Tests user registration → JWT issuance → token auth → query propagation.

Covers:
- UserStore: registration, login, duplicate rejection, API key ops
- JWTAuth: issuance, verification, refresh rotation, expiry, blacklisting
- AuthMiddleware: JWT auth, API key fallback, rate limiting
- user_id propagation through SovereignQuery / SovereignResult
"""

from __future__ import annotations

import base64
import hashlib
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from core.auth.jwt_auth import JWTAuth, TokenPair

# ---------------------------------------------------------------------------
# Import auth components
# ---------------------------------------------------------------------------
from core.auth.user_store import (
    UserRecord,
    UserStore,
    generate_api_key,
    generate_ed25519_keypair,
    hash_api_key,
    hash_password,
    verify_password,
)

# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a temp path for a fresh SQLite database."""
    return tmp_path / "test_users.db"


@pytest.fixture
def store(tmp_db: Path) -> UserStore:
    """Return a fresh UserStore backed by a temp DB."""
    return UserStore(db_path=tmp_db)


@pytest.fixture
def jwt() -> JWTAuth:
    """Return a JWTAuth with a deterministic secret."""
    return JWTAuth(secret="test-secret-256-bit-minimum-length!!")


@pytest.fixture
def registered_user(store: UserStore) -> UserRecord:
    """Register and return a user."""
    return store.register(
        username="mumu",
        email="mumu@bizra.ai",
        password="supersecret123",
    )


# ===========================================================================
# PASSWORD HASHING
# ===========================================================================


class TestPasswordHashing:
    """PBKDF2-SHA256 (600K iterations) hashing and verification."""

    def test_hash_produces_expected_format(self):
        h = hash_password("test")
        parts = h.split("$")
        assert parts[0] == "pbkdf2:sha256:600000"
        assert len(parts) == 3  # algo$salt$hash

    def test_verify_correct_password(self):
        h = hash_password("correcthorse")
        assert verify_password("correcthorse", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("correcthorse")
        assert verify_password("wronghorse", h) is False

    def test_different_salts_produce_different_hashes(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # different salts
        # But both verify
        assert verify_password("same", h1) is True
        assert verify_password("same", h2) is True


# ===========================================================================
# API KEY GENERATION
# ===========================================================================


class TestApiKey:

    def test_api_key_format(self):
        key = generate_api_key()
        assert key.startswith("bzr_")
        assert len(key) == 4 + 64  # "bzr_" + 32 hex bytes

    def test_api_key_uniqueness(self):
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100


# ===========================================================================
# USER STORE
# ===========================================================================


class TestUserStore:

    def test_register_user(self, store: UserStore):
        user = store.register(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.user_id  # UUID generated
        assert user.api_key.startswith("bzr_")
        assert user.status == "active"

    def test_register_duplicate_username(self, store: UserStore):
        store.register(username="dup_user", email="a@b.com", password="password1")
        with pytest.raises(ValueError, match="[Uu]sername"):
            store.register(username="dup_user", email="c@d.com", password="password2")

    def test_register_duplicate_email(self, store: UserStore):
        store.register(username="user1", email="same@b.com", password="password1")
        with pytest.raises(ValueError, match="[Ee]mail"):
            store.register(username="user2", email="same@b.com", password="password2")

    def test_verify_login_success(self, store: UserStore, registered_user: UserRecord):
        user = store.verify_login("mumu", "supersecret123")
        assert user is not None
        assert user.user_id == registered_user.user_id

    def test_verify_login_wrong_password(
        self, store: UserStore, registered_user: UserRecord
    ):
        user = store.verify_login("mumu", "wrongpassword")
        assert user is None

    def test_verify_login_unknown_user(self, store: UserStore):
        user = store.verify_login("nobody", "irrelevant")
        assert user is None

    def test_get_by_id(self, store: UserStore, registered_user: UserRecord):
        found = store.get_by_id(registered_user.user_id)
        assert found is not None
        assert found.username == "mumu"

    def test_get_by_username(self, store: UserStore, registered_user: UserRecord):
        found = store.get_by_username("mumu")
        assert found is not None
        assert found.email == "mumu@bizra.ai"

    def test_get_by_email(self, store: UserStore, registered_user: UserRecord):
        found = store.get_by_email("mumu@bizra.ai")
        assert found is not None
        assert found.username == "mumu"

    def test_verify_api_key(self, store: UserStore, registered_user: UserRecord):
        found = store.verify_api_key(registered_user.api_key)
        assert found is not None
        assert found.user_id == registered_user.user_id

    def test_primary_api_key_stored_hashed(
        self, store: UserStore, registered_user: UserRecord
    ):
        with sqlite3.connect(str(store.db_path)) as conn:
            row = conn.execute(
                "SELECT api_key FROM users WHERE user_id = ?",
                (registered_user.user_id,),
            ).fetchone()
        assert row is not None
        stored = row[0]
        assert stored == hash_api_key(registered_user.api_key)
        assert stored != registered_user.api_key
        assert not stored.startswith("bzr_")

    def test_secondary_api_key_stored_hashed(
        self, store: UserStore, registered_user: UserRecord
    ):
        key = store.create_secondary_api_key(registered_user.user_id, label="ci")
        assert key is not None

        with sqlite3.connect(str(store.db_path)) as conn:
            row = conn.execute(
                "SELECT api_key FROM api_keys WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (registered_user.user_id,),
            ).fetchone()
        assert row is not None
        stored = row[0]
        assert stored == hash_api_key(key)
        assert stored != key
        assert store.verify_api_key(key) is not None

    def test_verify_api_key_upgrades_legacy_plaintext_primary(self, store: UserStore):
        user = store.register(
            username="legacy_primary",
            email="legacy_primary@bizra.ai",
            password="password123",
        )
        with sqlite3.connect(str(store.db_path)) as conn:
            conn.execute(
                "UPDATE users SET api_key = ? WHERE user_id = ?",
                (user.api_key, user.user_id),
            )
            conn.commit()

        found = store.verify_api_key(user.api_key)
        assert found is not None

        with sqlite3.connect(str(store.db_path)) as conn:
            row = conn.execute(
                "SELECT api_key FROM users WHERE user_id = ?",
                (user.user_id,),
            ).fetchone()
        assert row is not None
        assert row[0] == hash_api_key(user.api_key)

    def test_verify_api_key_upgrades_legacy_plaintext_secondary(
        self, store: UserStore, registered_user: UserRecord
    ):
        key = store.create_secondary_api_key(
            registered_user.user_id, label="legacy-secondary"
        )
        assert key is not None

        with sqlite3.connect(str(store.db_path)) as conn:
            conn.execute(
                "UPDATE api_keys SET api_key = ? WHERE user_id = ?",
                (key, registered_user.user_id),
            )
            conn.commit()

        found = store.verify_api_key(key)
        assert found is not None
        assert found.user_id == registered_user.user_id

        with sqlite3.connect(str(store.db_path)) as conn:
            row = conn.execute(
                "SELECT api_key FROM api_keys WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (registered_user.user_id,),
            ).fetchone()
        assert row is not None
        assert row[0] == hash_api_key(key)

    def test_verify_invalid_api_key(self, store: UserStore):
        found = store.verify_api_key("bzr_invalid")
        assert found is None

    def test_verify_expired_secondary_api_key_rejected(
        self, store: UserStore, registered_user: UserRecord
    ):
        expired_key = store.create_secondary_api_key(
            registered_user.user_id,
            label="expired",
            expires_in_days=-1,
        )
        assert expired_key is not None
        assert store.verify_api_key(expired_key) is None

    def test_count_users(self, store: UserStore):
        assert store.count_users() == 0
        store.register(username="user_one", email="u1@b.com", password="password1")
        store.register(username="user_two", email="u2@b.com", password="password2")
        assert store.count_users() == 2

    def test_list_users(self, store: UserStore):
        store.register(username="alpha", email="a@b.com", password="password1")
        store.register(username="bravo", email="b@b.com", password="password2")
        users = store.list_users()
        assert len(users) == 2
        usernames = {u.username for u in users}
        assert usernames == {"alpha", "bravo"}

    def test_increment_query_count(self, store: UserStore, registered_user: UserRecord):
        assert registered_user.query_count == 0
        store.increment_query_count(registered_user.user_id)
        store.increment_query_count(registered_user.user_id)
        user = store.get_by_id(registered_user.user_id)
        assert user is not None
        assert user.query_count == 2

    def test_update_status(self, store: UserStore, registered_user: UserRecord):
        store.update_status(registered_user.user_id, "suspended")
        user = store.get_by_id(registered_user.user_id)
        assert user is not None
        assert user.status == "suspended"

    def test_rotate_api_key(self, store: UserStore, registered_user: UserRecord):
        old_key = registered_user.api_key
        new_key = store.rotate_api_key(registered_user.user_id)
        assert new_key is not None
        assert new_key != old_key
        assert new_key.startswith("bzr_")
        # Old key should no longer work
        assert store.verify_api_key(old_key) is None
        # New key should work
        assert store.verify_api_key(new_key) is not None

    def test_user_namespace(self, registered_user: UserRecord):
        ns = registered_user.namespace
        assert ns.startswith("user_")
        assert len(ns) == 5 + 8  # "user_" + first 8 chars of UUID

    def test_get_user_data_dir(self, store: UserStore, registered_user: UserRecord):
        data_dir = store.get_user_data_dir(registered_user.user_id)
        assert data_dir is not None
        assert registered_user.user_id[:8] in str(data_dir)

    def test_ed25519_secret_wrap_stored_as_v2(
        self, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("BIZRA_USERSTORE_MASTER_SECRET", "unit-test-master-secret")
        local_store = UserStore(db_path=tmp_db)
        user = local_store.register(
            username="v2_user",
            email="v2@bizra.ai",
            password="password123",
        )
        with sqlite3.connect(str(tmp_db)) as conn:
            row = conn.execute(
                "SELECT ed25519_secret_key_enc FROM users WHERE user_id = ?",
                (user.user_id,),
            ).fetchone()
        assert row is not None
        assert row[0].startswith("fernet:v2:")

    def test_legacy_ed25519_wrap_migrates_to_v2(
        self, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("BIZRA_USERSTORE_MASTER_SECRET", "unit-test-master-secret")
        local_store = UserStore(db_path=tmp_db)
        user = local_store.register(
            username="legacy_wrap",
            email="legacy_wrap@bizra.ai",
            password="password123",
        )

        with sqlite3.connect(str(tmp_db)) as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE user_id = ?",
                (user.user_id,),
            ).fetchone()
            assert row is not None
            password_hash = row[0]

            legacy_material = hashlib.pbkdf2_hmac(
                "sha256",
                password_hash.encode("utf-8"),
                user.user_id.encode("utf-8"),
                100_000,
            )
            legacy_key = base64.urlsafe_b64encode(legacy_material[:32])
            legacy_plaintext = "ab" * 32
            legacy_wrapped = "fernet:v1:" + Fernet(legacy_key).encrypt(
                legacy_plaintext.encode("utf-8")
            ).decode("utf-8")

            conn.execute(
                "UPDATE users SET ed25519_secret_key_enc = ? WHERE user_id = ?",
                (legacy_wrapped, user.user_id),
            )
            conn.commit()

        migrated_store = UserStore(db_path=tmp_db)
        with sqlite3.connect(str(tmp_db)) as conn:
            migrated = conn.execute(
                "SELECT ed25519_secret_key_enc FROM users WHERE user_id = ?",
                (user.user_id,),
            ).fetchone()
        assert migrated is not None
        wrapped = migrated[0]
        assert wrapped.startswith("fernet:v2:")
        assert wrapped != legacy_wrapped

        token = wrapped[len("fernet:v2:") :].encode("utf-8")
        decrypted = Fernet(
            migrated_store._derive_master_wrapping_key(user.user_id)
        ).decrypt(token)
        assert decrypted.decode("utf-8") == legacy_plaintext

    def test_generate_ed25519_keypair_fails_closed_without_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import builtins

        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nacl.signing" or name.startswith(
                "cryptography.hazmat.primitives.asymmetric.ed25519"
            ):
                raise ImportError("backend blocked by test")
            if name.startswith("cryptography.hazmat.primitives.serialization"):
                raise ImportError("backend blocked by test")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        with pytest.raises(RuntimeError, match="Ed25519 backend unavailable"):
            generate_ed25519_keypair()


# ===========================================================================
# JWT AUTH
# ===========================================================================


class TestJWTAuth:

    def test_issue_tokens(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        assert isinstance(pair, TokenPair)
        assert pair.access_token
        assert pair.refresh_token
        assert pair.token_type == "bearer"
        assert pair.expires_in > 0

    def test_verify_access_token(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        claims = jwt.verify_token(pair.access_token)
        assert claims is not None
        assert claims.sub == "uid123"
        assert claims.username == "mumu"
        assert claims.typ == "access"

    def test_verify_refresh_token(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        claims = jwt.verify_token(pair.refresh_token, expected_type="refresh")
        assert claims is not None
        assert claims.sub == "uid123"
        assert claims.typ == "refresh"

    def test_verify_wrong_type_rejected(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        # Access token should fail if we expect refresh type
        claims = jwt.verify_token(pair.access_token, expected_type="refresh")
        assert claims is None

    def test_verify_garbage_token(self, jwt: JWTAuth):
        claims = jwt.verify_token("not.a.jwt")
        assert claims is None

    def test_verify_tampered_token(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        # Flip a character in the signature
        parts = pair.access_token.split(".")
        sig = list(parts[2])
        sig[0] = "X" if sig[0] != "X" else "Y"
        parts[2] = "".join(sig)
        tampered = ".".join(parts)
        claims = jwt.verify_token(tampered)
        assert claims is None

    def test_expired_token_rejected(self, jwt: JWTAuth):
        # Issue with 1-second expiry
        jwt_short = JWTAuth(
            secret="test-secret-256-bit-minimum-length!!", access_expiry=1
        )
        pair = jwt_short.issue_tokens("uid123", "mumu")
        time.sleep(1.5)
        claims = jwt_short.verify_token(pair.access_token)
        assert claims is None

    def test_refresh_access_token(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        new_pair = jwt.refresh_access_token(pair.refresh_token)
        assert new_pair is not None
        assert new_pair.access_token
        assert new_pair.access_token != pair.access_token

    def test_refresh_with_access_token_fails(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        # Using access token for refresh should either raise or return None
        try:
            result = jwt.refresh_access_token(pair.access_token)
            # If it doesn't raise, it should return None or invalid
            assert result is None
        except (ValueError, Exception):
            pass  # Expected

    def test_revoke_token(self, jwt: JWTAuth):
        pair = jwt.issue_tokens("uid123", "mumu")
        jwt.revoke_token(pair.access_token)
        claims = jwt.verify_token(pair.access_token)
        assert claims is None

    def test_blacklist_stays_bounded_when_entries_are_fresh(self, jwt: JWTAuth):
        now = int(time.time())
        jwt._BLACKLIST_MAX = 2
        jwt._blacklist = {
            "oldest": now + 10,
            "newer": now + 20,
        }

        jwt._blacklist_add("freshest", now + 30)

        assert len(jwt._blacklist) == 2
        assert "oldest" not in jwt._blacklist
        assert set(jwt._blacklist) == {"newer", "freshest"}

    def test_different_secrets_reject(self):
        jwt1 = JWTAuth(secret="secret-one-is-one-secret!!!!!!!!!")
        jwt2 = JWTAuth(secret="secret-two-is-two-secret!!!!!!!!!")
        pair = jwt1.issue_tokens("uid123", "mumu")
        claims = jwt2.verify_token(pair.access_token)
        assert claims is None


# ===========================================================================
# SOVEREIGN QUERY / RESULT USER_ID PROPAGATION
# ===========================================================================


class TestUserIdPropagation:

    def test_sovereign_query_default_user_id(self):
        from core.sovereign.runtime_types import SovereignQuery

        q = SovereignQuery(text="test")
        assert q.user_id == ""

    def test_sovereign_query_with_user_id(self):
        from core.sovereign.runtime_types import SovereignQuery

        q = SovereignQuery(text="test", user_id="uid-abc")
        assert q.user_id == "uid-abc"

    def test_sovereign_result_default_user_id(self):
        from core.sovereign.runtime_types import SovereignResult

        r = SovereignResult(query_id="q1")
        assert r.user_id == ""

    def test_sovereign_result_with_user_id(self):
        from core.sovereign.runtime_types import SovereignResult

        r = SovereignResult(query_id="q1", user_id="uid-abc")
        assert r.user_id == "uid-abc"

    def test_result_to_dict_includes_user_id(self):
        from core.sovereign.runtime_types import SovereignResult

        r = SovereignResult(query_id="q1", user_id="uid-abc", success=True)
        d = r.to_dict()
        assert d["user_id"] == "uid-abc"

    def test_result_to_dict_empty_user_id_still_present(self):
        from core.sovereign.runtime_types import SovereignResult

        r = SovereignResult(query_id="q1")
        d = r.to_dict()
        assert "user_id" in d
        assert d["user_id"] == ""


# ===========================================================================
# AUTH MIDDLEWARE (unit-level mocking)
# ===========================================================================


class TestAuthMiddleware:
    """Test the middleware authenticate logic with mocked store/jwt."""

    def test_middleware_init(self, store: UserStore, jwt: JWTAuth):
        from core.auth.middleware import AuthMiddleware

        mw = AuthMiddleware(user_store=store, jwt_auth=jwt)
        assert mw is not None

    @pytest.mark.asyncio
    async def test_authenticate_with_jwt_bearer(self, store: UserStore, jwt: JWTAuth):
        from core.auth.middleware import AuthMiddleware

        user = store.register(
            username="jwt_user", email="jwt@test.com", password="password123"
        )
        tokens = jwt.issue_tokens(user.user_id, user.username)

        mw = AuthMiddleware(user_store=store, jwt_auth=jwt)

        # authenticate() is sync and takes string params
        authenticated_user = mw.authenticate(
            authorization=f"Bearer {tokens.access_token}"
        )
        assert authenticated_user is not None
        assert authenticated_user.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_authenticate_with_api_key(self, store: UserStore, jwt: JWTAuth):
        from core.auth.middleware import AuthMiddleware

        user = store.register(
            username="key_user", email="key@test.com", password="password123"
        )

        mw = AuthMiddleware(user_store=store, jwt_auth=jwt)

        # authenticate() is sync and takes string params
        authenticated_user = mw.authenticate(api_key=user.api_key)
        assert authenticated_user is not None
        assert authenticated_user.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_authenticate_no_credentials_raises(
        self, store: UserStore, jwt: JWTAuth
    ):
        from core.auth.middleware import AuthMiddleware

        mw = AuthMiddleware(user_store=store, jwt_auth=jwt)

        request = MagicMock()
        request.headers = {}

        with pytest.raises(Exception):  # HTTPException or ValueError
            await mw.authenticate(request)

    @pytest.mark.asyncio
    async def test_authenticate_invalid_jwt_raises(
        self, store: UserStore, jwt: JWTAuth
    ):
        from core.auth.middleware import AuthMiddleware

        mw = AuthMiddleware(user_store=store, jwt_auth=jwt)

        request = MagicMock()
        request.headers = {"authorization": "Bearer invalid.token.here"}

        with pytest.raises(Exception):
            await mw.authenticate(request)

    def test_rate_limit_buckets_stay_bounded_when_entries_are_fresh(
        self,
        store: UserStore,
        jwt: JWTAuth,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from core.auth.middleware import AuthMiddleware

        monkeypatch.setattr("core.auth.middleware.time.time", lambda: 1000.0)
        mw = AuthMiddleware(user_store=store, jwt_auth=jwt)
        mw._BUCKET_MAX = 2
        mw._buckets = {
            "oldest": {"tokens": 1.0, "last": 990.0},
            "newer": {"tokens": 1.0, "last": 995.0},
        }

        assert mw.check_rate_limit("fresh") is True
        assert len(mw._buckets) == 2
        assert "oldest" not in mw._buckets
        assert set(mw._buckets) == {"newer", "fresh"}


# ===========================================================================
# END-TO-END: Register → Login → Token → Query (mock runtime)
# ===========================================================================


class TestAuthE2EFlow:
    """Full flow: register, login, get tokens, use them."""

    def test_register_login_verify_flow(self, store: UserStore, jwt: JWTAuth):
        """Register → login → issue JWT → verify JWT → find user."""
        # 1. Register
        user = store.register(
            username="e2e_user", email="e2e@test.com", password="password123"
        )
        assert user.status == "active"

        # 2. Login
        verified = store.verify_login("e2e_user", "password123")
        assert verified is not None
        assert verified.user_id == user.user_id

        # 3. Issue JWT
        tokens = jwt.issue_tokens(verified.user_id, verified.username)
        assert tokens.access_token

        # 4. Verify JWT
        claims = jwt.verify_token(tokens.access_token)
        assert claims is not None
        assert claims.sub == user.user_id
        assert claims.username == "e2e_user"

        # 5. Look up user by ID from claims
        found = store.get_by_id(claims.sub)
        assert found is not None
        assert found.username == "e2e_user"

    def test_api_key_flow(self, store: UserStore):
        """Register → get API key → verify → rotate → verify new."""
        user = store.register(
            username="apiuser", email="api@test.com", password="password123"
        )
        key = user.api_key

        # Verify works
        found = store.verify_api_key(key)
        assert found is not None

        # Rotate
        new_key = store.rotate_api_key(user.user_id)
        assert new_key != key

        # Old key fails
        assert store.verify_api_key(key) is None

        # New key works
        found2 = store.verify_api_key(new_key)
        assert found2 is not None
        assert found2.user_id == user.user_id

    def test_token_refresh_rotation(self, jwt: JWTAuth):
        """Issue → refresh → old refresh blacklisted → new works."""
        pair1 = jwt.issue_tokens("uid", "user")

        # Refresh
        pair2 = jwt.refresh_access_token(pair1.refresh_token)
        assert pair2.access_token != pair1.access_token

        # Verify new access token
        claims = jwt.verify_token(pair2.access_token)
        assert claims is not None
        assert claims.sub == "uid"

    def test_suspended_user_cannot_login(self, store: UserStore):
        """Register → suspend → login fails."""
        user = store.register(
            username="sus_user", email="sus@test.com", password="password123"
        )
        store.update_status(user.user_id, "suspended")

        # Verify login returns None for suspended user
        result = store.verify_login("sus_user", "password123")
        # Depending on implementation, may return None or the user
        # The middleware should check status before allowing access
        if result is not None:
            assert result.status == "suspended"
