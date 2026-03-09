"""
Tests for Declaration of Digital Sovereignty — Genesis Block Handler
════════════════════════════════════════════════════════════════════

TDD anchors from Phase 67.03 specification.

Standing on Giants:
- Beck (2002): Test-Driven Development by Example
- Nakamoto (2008): Genesis block verification
"""

from __future__ import annotations

import hashlib

import pytest

from core.constitutional.declaration import (
    DECLARATION_BLAKE2B_256,
    INVARIANTS,
    ConstitutionalInvariant,
    ConstitutionalViolation,
    compute_declaration_hash,
    create_genesis_event,
    verify_covenant_chain,
    verify_declaration_hash,
)
from core.constitutional.fixed_point import fp
from core.constitutional.types import Event

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

# Canonical Declaration text that matches the BLAKE2b hash constant.
# We reverse-engineer a test fixture: compute text → hash, then
# use that text consistently in tests.
FIXTURE_DECLARATION_TEXT = "BIZRA Declaration of Digital Sovereignty — Test Fixture"
FIXTURE_DECLARATION_HASH = hashlib.blake2b(
    FIXTURE_DECLARATION_TEXT.encode("utf-8"), digest_size=32
).hexdigest()


@pytest.fixture
def declaration_text():
    """Declaration text that matches the known hash."""
    return FIXTURE_DECLARATION_TEXT


# ═══════════════════════════════════════════════════════════════════
# Test: Constitutional Invariants
# ═══════════════════════════════════════════════════════════════════


class TestConstitutionalInvariants:
    """Verify the 7 constitutional invariants are correctly defined."""

    def test_seven_invariants_exist(self):
        """I-1 through I-7 must all be present."""
        assert len(INVARIANTS) == 7

    def test_invariant_codes_sequential(self):
        """Codes must be I-1, I-2, ..., I-7."""
        expected = [f"I-{i}" for i in range(1, 8)]
        actual = [inv.code for inv in INVARIANTS]
        assert actual == expected

    def test_invariants_are_frozen(self):
        """Constitutional invariants cannot be mutated after creation."""
        inv = INVARIANTS[0]
        with pytest.raises(AttributeError):
            inv.code = "I-99"  # type: ignore[misc]

    def test_i3_gini_threshold(self):
        """I-3 must enforce Gini <= 0.35."""
        i3 = INVARIANTS[2]
        assert i3.code == "I-3"
        assert i3.algorithm == "A4_GINI_ENFORCER"
        assert i3.threshold == fp(0.35)

    def test_i4_ihsan_threshold(self):
        """I-4 must enforce Ihsan >= 0.95."""
        i4 = INVARIANTS[3]
        assert i4.code == "I-4"
        assert i4.algorithm == "A1_IHSAN_SCORER"
        assert i4.threshold == fp(0.95)

    def test_i7_zakat_threshold(self):
        """I-7 must enforce 2.5% Zakat rate."""
        i7 = INVARIANTS[6]
        assert i7.code == "I-7"
        assert i7.algorithm == "A5_ZAKAT_ENGINE"
        assert i7.threshold == fp(0.025)

    def test_each_invariant_has_algorithm(self):
        """Every invariant must name the algorithm that enforces it."""
        for inv in INVARIANTS:
            assert inv.algorithm, f"{inv.code} has no algorithm"
            assert inv.guarantee, f"{inv.code} has no guarantee"


# ═══════════════════════════════════════════════════════════════════
# Test: Hash Functions
# ═══════════════════════════════════════════════════════════════════


class TestDeclarationHash:
    """Verify BLAKE2b hash computation and verification."""

    def test_compute_hash_deterministic(self):
        """Same input must always produce the same hash."""
        text = "Hello, Sovereignty"
        h1 = compute_declaration_hash(text)
        h2 = compute_declaration_hash(text)
        assert h1 == h2

    def test_compute_hash_is_64_hex_chars(self):
        """BLAKE2b-256 produces 32 bytes = 64 hex chars."""
        h = compute_declaration_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_single_byte_change_flips_hash(self):
        """Even a one-character difference must produce a completely different hash."""
        h1 = compute_declaration_hash("Declaration v1")
        h2 = compute_declaration_hash("Declaration v2")
        assert h1 != h2

    def test_verify_returns_false_for_wrong_text(self):
        """Arbitrary text should not match the canonical hash."""
        assert verify_declaration_hash("this is not the declaration") is False

    def test_crlf_normalization_matters(self):
        """Windows CRLF vs Unix LF produces different hashes."""
        h_lf = compute_declaration_hash("line1\nline2")
        h_crlf = compute_declaration_hash("line1\r\nline2")
        assert h_lf != h_crlf

    def test_canonical_hash_is_valid_hex(self):
        """The stored canonical hash constant must be valid hex."""
        assert len(DECLARATION_BLAKE2B_256) == 64
        # Must parse without error
        int(DECLARATION_BLAKE2B_256, 16)


# ═══════════════════════════════════════════════════════════════════
# Test: Genesis Event Creation
# ═══════════════════════════════════════════════════════════════════


class TestGenesisEvent:
    """Verify genesis event creation and covenant locking."""

    def test_genesis_rejects_wrong_declaration(self):
        """Creating genesis with wrong text must raise ConstitutionalViolation."""
        with pytest.raises(ConstitutionalViolation, match="covenant broken"):
            create_genesis_event("this is a fake declaration")

    def test_genesis_event_structure(self):
        """Genesis event must have correct structure when given valid text.

        Since we can't easily produce text matching the hardcoded hash,
        we monkeypatch the verification to test structure.
        """
        import core.constitutional.declaration as decl_mod

        original_hash = decl_mod.DECLARATION_BLAKE2B_256
        try:
            # Temporarily set the hash to match our fixture
            decl_mod.DECLARATION_BLAKE2B_256 = FIXTURE_DECLARATION_HASH
            event = create_genesis_event(FIXTURE_DECLARATION_TEXT)

            assert event.event_id == 0
            assert event.event_type == "genesis"
            assert event.actor == b"\x00" * 32
            assert event.timestamp == 0
            assert event.prev_hash == b"\x00" * 32
            assert len(event.hash) == 32  # BLAKE2b-256 = 32 bytes
            assert event.data["declaration_hash_blake2b"] == FIXTURE_DECLARATION_HASH
            assert event.data["invariants"] == [f"I-{i}" for i in range(1, 8)]
            assert event.data["version"] == "1.0.0"
        finally:
            decl_mod.DECLARATION_BLAKE2B_256 = original_hash

    def test_genesis_hash_is_deterministic(self):
        """Same declaration text must produce identical genesis events."""
        import core.constitutional.declaration as decl_mod

        original_hash = decl_mod.DECLARATION_BLAKE2B_256
        try:
            decl_mod.DECLARATION_BLAKE2B_256 = FIXTURE_DECLARATION_HASH
            e1 = create_genesis_event(FIXTURE_DECLARATION_TEXT)
            e2 = create_genesis_event(FIXTURE_DECLARATION_TEXT)
            assert e1.hash == e2.hash
            assert e1.data == e2.data
        finally:
            decl_mod.DECLARATION_BLAKE2B_256 = original_hash


# ═══════════════════════════════════════════════════════════════════
# Test: Covenant Chain Verification
# ═══════════════════════════════════════════════════════════════════


class TestCovenantChain:
    """Verify event log chain integrity checking."""

    def _make_genesis_event(self) -> Event:
        """Helper: create a valid genesis event for chain tests."""
        content = FIXTURE_DECLARATION_TEXT.encode("utf-8")
        genesis_hash = hashlib.blake2b(content, digest_size=32).digest()
        return Event(
            event_id=0,
            event_type="genesis",
            actor=b"\x00" * 32,
            data={
                "declaration_hash_blake2b": DECLARATION_BLAKE2B_256,
                "version": "1.0.0",
            },
            timestamp=0,
            prev_hash=b"\x00" * 32,
            hash=genesis_hash,
        )

    def _make_chained_event(self, prev: Event, event_id: int) -> Event:
        """Helper: create an event that chains to the previous one."""
        data = {"event_id": event_id}
        import json

        canonical = json.dumps(data, sort_keys=True).encode("utf-8")
        h = hashlib.blake2b(canonical, digest_size=32).digest()
        return Event(
            event_id=event_id,
            event_type="mint",
            actor=b"\x01" * 32,
            data=data,
            timestamp=event_id * 1000,
            prev_hash=prev.hash,
            hash=h,
        )

    def test_empty_log_is_invalid(self):
        """Empty event log must fail verification."""
        valid, errors = verify_covenant_chain([])
        assert valid is False
        assert "Empty event log" in errors[0]

    def test_valid_genesis_only(self):
        """A log with just a valid genesis should pass."""
        genesis = self._make_genesis_event()
        valid, errors = verify_covenant_chain([genesis])
        assert valid is True
        assert errors == []

    def test_valid_chain_of_three(self):
        """A properly chained log of 3 events should pass."""
        genesis = self._make_genesis_event()
        e1 = self._make_chained_event(genesis, 1)
        e2 = self._make_chained_event(e1, 2)
        valid, errors = verify_covenant_chain([genesis, e1, e2])
        assert valid is True
        assert errors == []

    def test_broken_chain_detected(self):
        """A chain break must be reported with the event index."""
        genesis = self._make_genesis_event()
        e1 = self._make_chained_event(genesis, 1)
        # e2 points to genesis instead of e1 — chain break
        e2_broken = Event(
            event_id=2,
            event_type="mint",
            actor=b"\x01" * 32,
            data={},
            timestamp=2000,
            prev_hash=genesis.hash,  # Should be e1.hash
            hash=b"\xff" * 32,
        )
        valid, errors = verify_covenant_chain([genesis, e1, e2_broken])
        assert valid is False
        assert any("Chain break at event 2" in e for e in errors)

    def test_non_genesis_first_event(self):
        """First event must be type 'genesis'."""
        bad_first = Event(
            event_id=0,
            event_type="mint",  # Not genesis
            actor=b"\x00" * 32,
            data={"declaration_hash_blake2b": DECLARATION_BLAKE2B_256},
            timestamp=0,
            prev_hash=b"\x00" * 32,
            hash=b"\xaa" * 32,
        )
        valid, errors = verify_covenant_chain([bad_first])
        assert valid is False
        assert any("not genesis" in e for e in errors)

    def test_wrong_declaration_hash_in_genesis(self):
        """Genesis with wrong declaration hash must fail."""
        bad_genesis = Event(
            event_id=0,
            event_type="genesis",
            actor=b"\x00" * 32,
            data={"declaration_hash_blake2b": "0000000000000000" * 4},
            timestamp=0,
            prev_hash=b"\x00" * 32,
            hash=b"\xbb" * 32,
        )
        valid, errors = verify_covenant_chain([bad_genesis])
        assert valid is False
        assert any("does not match canonical" in e for e in errors)
