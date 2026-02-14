"""
Deterministic Canonicalization Tests — SP-001 / INT-005.

Proves that canonical encoding is deterministic, unicode-safe,
and produces stable hashes across re-serialization.

Standing on Giants:
- Shannon (1948): Information representation
- Lamport (1978): Deterministic state
- BIZRA Spearpoint PRD SP-001: "same input -> same bytes -> same hash"
"""

import json
import unicodedata
import pytest

from core.proof_engine.canonical import (
    canonical_json,
    canonical_bytes,
    blake3_digest,
    hex_digest,
    CanonQuery,
    CanonPolicy,
    CanonEnvironment,
    verify_determinism,
)


# =============================================================================
# CANONICAL_JSON
# =============================================================================

class TestCanonicalJson:
    """Tests for canonical_json() recursive normalization."""

    def test_sorts_dict_keys(self):
        """Dict keys are sorted alphabetically."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_nested_dict_keys_sorted(self):
        """Nested dicts have keys sorted recursively."""
        obj = {"b": {"z": 1, "a": 2}, "a": {"y": 3, "x": 4}}
        result = canonical_json(obj)
        assert list(result.keys()) == ["a", "b"]
        assert list(result["a"].keys()) == ["x", "y"]
        assert list(result["b"].keys()) == ["a", "z"]

    def test_preserves_array_order(self):
        """Arrays maintain their order."""
        obj = [3, 1, 2]
        assert canonical_json(obj) == [3, 1, 2]

    def test_normalizes_strings_nfc(self):
        """Strings are NFC-normalized and stripped."""
        raw = "  caf\u0065\u0301  "  # e + combining acute (NFD)
        result = canonical_json(raw)
        assert result == unicodedata.normalize("NFC", raw.strip())

    def test_preserves_numbers(self):
        """Integers and floats preserved as-is."""
        assert canonical_json(42) == 42
        assert canonical_json(3.14) == 3.14

    def test_preserves_booleans(self):
        """Booleans preserved as-is."""
        assert canonical_json(True) is True
        assert canonical_json(False) is False

    def test_preserves_none(self):
        """None preserved."""
        assert canonical_json(None) is None

    def test_converts_unknown_types_to_string(self):
        """Unknown types are stringified."""
        result = canonical_json(set([1, 2]))
        assert isinstance(result, str)

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert canonical_json({}) == {}

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert canonical_json([]) == []

    def test_deeply_nested_structure(self):
        """Deep nesting is handled correctly."""
        obj = {"a": [{"z": 1, "a": 2}, {"y": [{"c": 3, "b": 4}]}]}
        result = canonical_json(obj)
        inner = result["a"][0]
        assert list(inner.keys()) == ["a", "z"]
        deep = result["a"][1]["y"][0]
        assert list(deep.keys()) == ["b", "c"]


# =============================================================================
# CANONICAL_BYTES
# =============================================================================

class TestCanonicalBytes:
    """Tests for canonical_bytes() serialization."""

    def test_returns_bytes(self):
        """Returns bytes."""
        result = canonical_bytes({"a": 1})
        assert isinstance(result, bytes)

    def test_deterministic(self):
        """Same input produces same bytes."""
        obj = {"b": 2, "a": 1}
        b1 = canonical_bytes(obj)
        b2 = canonical_bytes(obj)
        assert b1 == b2

    def test_no_whitespace(self):
        """Output has no whitespace (compact separators)."""
        result = canonical_bytes({"key": "value"})
        decoded = result.decode("utf-8")
        assert " " not in decoded
        assert "\n" not in decoded

    def test_sorted_keys_in_output(self):
        """Keys are sorted in JSON output."""
        result = canonical_bytes({"z": 1, "a": 2})
        decoded = result.decode("utf-8")
        assert decoded.index('"a"') < decoded.index('"z"')

    def test_utf8_encoding(self):
        """Output is UTF-8 encoded."""
        result = canonical_bytes({"text": "hello"})
        # Should be valid UTF-8
        result.decode("utf-8")

    def test_different_inputs_different_bytes(self):
        """Different inputs produce different bytes."""
        b1 = canonical_bytes({"a": 1})
        b2 = canonical_bytes({"a": 2})
        assert b1 != b2


# =============================================================================
# BLAKE3_DIGEST / HEX_DIGEST
# =============================================================================

class TestBlake3Digest:
    """Tests for blake3_digest() and hex_digest()."""

    def test_returns_bytes(self):
        """blake3_digest returns bytes."""
        result = blake3_digest(b"hello")
        assert isinstance(result, bytes)

    def test_returns_32_bytes(self):
        """Digest is 32 bytes (256-bit)."""
        result = blake3_digest(b"hello")
        assert len(result) == 32

    def test_deterministic(self):
        """Same input produces same digest."""
        d1 = blake3_digest(b"test")
        d2 = blake3_digest(b"test")
        assert d1 == d2

    def test_different_inputs(self):
        """Different inputs produce different digests."""
        d1 = blake3_digest(b"alpha")
        d2 = blake3_digest(b"beta")
        assert d1 != d2

    def test_empty_input(self):
        """Empty input has a valid digest."""
        result = blake3_digest(b"")
        assert len(result) == 32

    def test_hex_digest_returns_string(self):
        """hex_digest returns hex string."""
        result = hex_digest(b"hello")
        assert isinstance(result, str)
        assert len(result) == 64  # 32 bytes * 2 hex chars

    def test_hex_digest_is_hex(self):
        """hex_digest output is valid hex."""
        result = hex_digest(b"test")
        int(result, 16)  # Should not raise


# =============================================================================
# CANON QUERY
# =============================================================================

class TestCanonQuery:
    """Tests for CanonQuery dataclass."""

    def test_basic_construction(self):
        """CanonQuery constructs with required fields."""
        q = CanonQuery(user_id="alice", user_state="active", intent="query")
        assert q.user_id == "alice"
        assert q.user_state == "active"
        assert q.intent == "query"

    def test_strips_whitespace(self):
        """Fields are stripped of whitespace."""
        q = CanonQuery(user_id="  alice  ", user_state=" active ", intent=" query ")
        assert q.user_id == "alice"
        assert q.user_state == "active"
        assert q.intent == "query"

    def test_unicode_normalization(self):
        """Fields are NFC-normalized."""
        q = CanonQuery(
            user_id="caf\u0065\u0301",
            user_state="active",
            intent="query",
        )
        expected = unicodedata.normalize("NFC", "caf\u0065\u0301")
        assert q.user_id == expected

    def test_canonical_bytes_deterministic(self):
        """canonical_bytes is deterministic."""
        q = CanonQuery(user_id="alice", user_state="active", intent="query")
        b1 = q.canonical_bytes()
        b2 = q.canonical_bytes()
        assert b1 == b2

    def test_digest_is_32_bytes(self):
        """digest() returns 32 bytes."""
        q = CanonQuery(user_id="alice", user_state="active", intent="query")
        assert len(q.digest()) == 32

    def test_hex_digest_is_64_chars(self):
        """hex_digest() returns 64 hex chars."""
        q = CanonQuery(user_id="alice", user_state="active", intent="query")
        hd = q.hex_digest()
        assert len(hd) == 64
        int(hd, 16)

    def test_different_queries_different_digests(self):
        """Different queries produce different digests."""
        q1 = CanonQuery(user_id="alice", user_state="active", intent="query1")
        q2 = CanonQuery(user_id="alice", user_state="active", intent="query2")
        assert q1.hex_digest() != q2.hex_digest()

    def test_to_dict_contains_all_fields(self):
        """to_dict() includes all required fields."""
        q = CanonQuery(user_id="alice", user_state="active", intent="query")
        d = q.to_dict()
        assert "user_id" in d
        assert "user_state" in d
        assert "intent" in d
        assert "payload" in d
        assert "timestamp" in d
        assert "nonce" in d
        assert "digest" in d

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) preserves semantics."""
        q = CanonQuery(
            user_id="alice",
            user_state="active",
            intent="test",
            payload={"key": "value"},
            nonce="abc123",
        )
        d = q.to_dict()
        q2 = CanonQuery.from_dict(d)
        assert q2.user_id == q.user_id
        assert q2.intent == q.intent
        assert q2.payload == q.payload
        assert q2.nonce == q.nonce

    def test_nonce_included_in_canonical_bytes(self):
        """Nonce affects canonical bytes."""
        q1 = CanonQuery(user_id="alice", user_state="a", intent="q", nonce="abc")
        q2 = CanonQuery(user_id="alice", user_state="a", intent="q", nonce="xyz")
        assert q1.canonical_bytes() != q2.canonical_bytes()

    def test_nonce_omitted_when_none(self):
        """Without nonce, canonical bytes exclude it."""
        q = CanonQuery(user_id="alice", user_state="a", intent="q")
        decoded = q.canonical_bytes().decode("utf-8")
        assert "nonce" not in decoded

    def test_payload_canonicalized(self):
        """Payload is canonicalized (keys sorted)."""
        q = CanonQuery(
            user_id="alice", user_state="a", intent="q",
            payload={"z": 1, "a": 2},
        )
        assert list(q.payload.keys()) == ["a", "z"]

    def test_auto_timestamp(self):
        """Timestamp is auto-set if not provided."""
        q = CanonQuery(user_id="alice", user_state="a", intent="q")
        assert q.timestamp is not None


# =============================================================================
# CANON POLICY
# =============================================================================

class TestCanonPolicy:
    """Tests for CanonPolicy dataclass."""

    def test_basic_construction(self):
        """CanonPolicy constructs with required fields."""
        p = CanonPolicy(
            policy_id="pol_001",
            version="1.0.0",
            rules={"snr_min": 0.95},
            thresholds={"ihsan": 0.95},
        )
        assert p.policy_id == "pol_001"
        assert p.version == "1.0.0"

    def test_canonical_bytes_deterministic(self):
        """canonical_bytes is deterministic."""
        p = CanonPolicy(
            policy_id="pol_001",
            version="1.0.0",
            rules={"rule": True},
            thresholds={"t": 0.5},
        )
        b1 = p.canonical_bytes()
        b2 = p.canonical_bytes()
        assert b1 == b2

    def test_digest_changes_with_rules(self):
        """Different rules produce different digests."""
        p1 = CanonPolicy(
            policy_id="pol", version="1", rules={"a": 1}, thresholds={"t": 0.5},
        )
        p2 = CanonPolicy(
            policy_id="pol", version="1", rules={"a": 2}, thresholds={"t": 0.5},
        )
        assert p1.hex_digest() != p2.hex_digest()

    def test_constraints_sorted(self):
        """Constraints are sorted in canonical representation."""
        p = CanonPolicy(
            policy_id="pol", version="1",
            rules={}, thresholds={},
            constraints=["z_constraint", "a_constraint"],
        )
        decoded = p.canonical_bytes().decode("utf-8")
        assert decoded.index("a_constraint") < decoded.index("z_constraint")


# =============================================================================
# CANON ENVIRONMENT
# =============================================================================

class TestCanonEnvironment:
    """Tests for CanonEnvironment dataclass."""

    def test_basic_construction(self):
        """CanonEnvironment constructs with required fields."""
        env = CanonEnvironment(
            platform="Linux",
            python_version="3.11.0",
            hostname="node0",
            cpu_count=8,
            memory_gb=16.0,
        )
        assert env.platform == "Linux"

    def test_canonical_bytes_deterministic(self):
        """canonical_bytes is deterministic."""
        env = CanonEnvironment(
            platform="Linux", python_version="3.11.0",
            hostname="node0", cpu_count=8, memory_gb=16.0,
        )
        b1 = env.canonical_bytes()
        b2 = env.canonical_bytes()
        assert b1 == b2

    def test_memory_gb_rounded(self):
        """memory_gb is rounded to 2 decimal places."""
        env = CanonEnvironment(
            platform="Linux", python_version="3.11.0",
            hostname="node0", cpu_count=8, memory_gb=15.99999,
        )
        decoded = env.canonical_bytes().decode("utf-8")
        data = json.loads(decoded)
        assert data["memory_gb"] == 16.0

    def test_capture_returns_instance(self):
        """capture() returns a CanonEnvironment."""
        env = CanonEnvironment.capture()
        assert isinstance(env, CanonEnvironment)
        assert env.cpu_count >= 1
        assert env.python_version != ""

    def test_extra_included(self):
        """Extra metadata is included in canonical bytes."""
        env1 = CanonEnvironment(
            platform="L", python_version="3", hostname="n",
            cpu_count=1, memory_gb=1.0, extra={"key": "value"},
        )
        env2 = CanonEnvironment(
            platform="L", python_version="3", hostname="n",
            cpu_count=1, memory_gb=1.0,
        )
        assert env1.canonical_bytes() != env2.canonical_bytes()


# =============================================================================
# VERIFY_DETERMINISM
# =============================================================================

class TestVerifyDeterminism:
    """Tests for verify_determinism() function."""

    def test_deterministic_with_default_iterations(self):
        """Standard query is deterministic over 100 iterations."""
        q = CanonQuery(
            user_id="alice", user_state="active", intent="test",
            nonce="fixed_nonce",
        )
        result = verify_determinism(q)
        assert result["deterministic"] is True
        assert result["unique_hashes"] == 1
        assert result["iterations"] == 100

    def test_deterministic_with_custom_iterations(self):
        """Determinism holds for custom iteration count."""
        q = CanonQuery(
            user_id="bob", user_state="idle", intent="verify",
            nonce="nonce_42",
        )
        result = verify_determinism(q, iterations=50)
        assert result["deterministic"] is True
        assert result["iterations"] == 50

    def test_with_complex_payload(self):
        """Complex payload is deterministic."""
        q = CanonQuery(
            user_id="alice", user_state="active", intent="complex",
            payload={"nested": {"z": [3, 2, 1], "a": {"deep": True}}},
            nonce="complex_nonce",
        )
        result = verify_determinism(q, iterations=20)
        assert result["deterministic"] is True

    def test_returns_canonical_digest(self):
        """Result includes the canonical digest."""
        q = CanonQuery(
            user_id="test", user_state="s", intent="i",
            nonce="n",
        )
        result = verify_determinism(q, iterations=5)
        assert result["canonical_digest"] is not None
        assert len(result["canonical_digest"]) == 64
