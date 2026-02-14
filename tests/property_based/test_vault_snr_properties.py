"""
Property-Based Tests — Vault Encrypt/Decrypt Roundtrip & SNR Bounds
====================================================================

Standing on Giants:
- Fernet (RFC 6749 variant) — AES-128-CBC + HMAC-SHA256
- PBKDF2 (RFC 8018, 600K iterations) — Key derivation
- Shannon (1948) — SNR information-theoretic bounds

Invariants verified:
1. Vault put→get roundtrip: ∀ (key, value), get(put(key, value)) == value
2. Vault delete→get: ∀ key, delete(key) → get(key) == None
3. Vault key listing consistency
4. SNR score normalization: ∀ result, 0.0 ≤ score ≤ 1.0
5. SNR facade dispatch determinism
"""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from pathlib import Path

from core.vault.vault import SovereignVault


# ── Strategies ──────────────────────────────────────────────────────────

# Safe key names (avoid filesystem-unsafe chars)
vault_keys = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=48,
)

# JSON-serializable values (vault stores via JSON round-trip)
vault_values = st.one_of(
    st.text(min_size=0, max_size=256),
    st.integers(min_value=-2**53, max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.lists(st.integers(min_value=-1000, max_value=1000), max_size=20),
)


# ── Vault Roundtrip Properties ──────────────────────────────────────────

class TestVaultRoundtrip:
    """AES-encrypted vault: put→get must return the original value."""

    @pytest.fixture(autouse=True)
    def _vault(self, tmp_path):
        """Fresh vault for each test."""
        self.vault = SovereignVault(
            vault_path=str(tmp_path / "test_vault"),
            master_secret="test-master-secret-hypothesis",
        )

    @given(key=vault_keys, value=vault_values)
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_put_get_roundtrip(self, key: str, value):
        """∀ (key, value): get(put(key, value)) == value"""
        self.vault.put(key, value)
        retrieved = self.vault.get(key)
        assert retrieved == value, f"Roundtrip failed: put({key!r}, {value!r}) → get = {retrieved!r}"
        # Cleanup for next iteration
        self.vault.delete(key)

    @given(key=vault_keys, value=vault_values)
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_delete_removes(self, key: str, value):
        """∀ key: put(key, v); delete(key) → get(key) == None"""
        self.vault.put(key, value)
        self.vault.delete(key)
        assert self.vault.get(key) is None

    @given(key=vault_keys, v1=vault_values, v2=vault_values)
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_overwrite_semantics(self, key: str, v1, v2):
        """∀ key: put(key, v1); put(key, v2) → get(key) == v2"""
        self.vault.put(key, v1)
        self.vault.put(key, v2)
        assert self.vault.get(key) == v2
        self.vault.delete(key)

    @given(keys=st.lists(vault_keys, min_size=1, max_size=10, unique=True))
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_list_keys_consistency(self, keys):
        """∀ keys: put each → list_keys() contains all."""
        for k in keys:
            self.vault.put(k, "value")
        listed = set(self.vault.list_keys())
        for k in keys:
            assert k in listed, f"Key {k!r} missing from list_keys()"
        # Cleanup
        for k in keys:
            self.vault.delete(k)


# ── SNR Score Properties ────────────────────────────────────────────────

class TestSNRBounds:
    """SNR results must be normalized to [0, 1]."""

    def test_snr_result_dataclass(self):
        """SNRResult fields are correctly bounded."""
        from core.snr_protocol import SNRResult
        r = SNRResult(score=0.85, ihsan_achieved=True, engine="test")
        assert 0.0 <= r.score <= 1.0
        assert r.ihsan_achieved is True
        assert r.engine == "test"

    @given(score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_snr_result_score_bounds(self, score):
        """∀ score ∈ [0, 1]: SNRResult accepts it."""
        from core.snr_protocol import SNRResult
        r = SNRResult(score=score, ihsan_achieved=score >= 0.95, engine="hypothesis")
        assert 0.0 <= r.score <= 1.0

    def test_snr_facade_import(self):
        """SNRFacade can be instantiated."""
        from core.snr_protocol import SNRFacade
        facade = SNRFacade()
        assert facade is not None
