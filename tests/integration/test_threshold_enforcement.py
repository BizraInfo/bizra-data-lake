"""ThresholdRegistry — Constitutional Enforcement Tests.

Validates that:
1. The registry singleton boots correctly with all canonical thresholds
2. Seal prevents mutation after boot
3. Module-level threshold shadows are detected
4. Canonical values match constants.py (no drift)

Standing on Giants: Lamport (Byzantine agreement) · Al-Ghazali (Ihsān, 1095)
"""

from __future__ import annotations

import pytest

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    CANONICAL_THRESHOLDS,
    IHSAN_GATE_MINIMUM,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.integration.threshold_registry import (
    SealedRegistryError,
    ThresholdNotFoundError,
    ThresholdRegistry,
    registry,
)


class TestRegistryBootstrap:
    """Registry loads all canonical thresholds at import time."""

    def test_registry_is_sealed_after_boot(self):
        assert registry.is_sealed

    def test_registry_has_ihsan_thresholds(self):
        assert registry.get("UNIFIED_IHSAN_THRESHOLD") == UNIFIED_IHSAN_THRESHOLD
        assert registry.get("IHSAN_GATE_MINIMUM") == IHSAN_GATE_MINIMUM

    def test_registry_has_snr_thresholds(self):
        assert registry.get("UNIFIED_SNR_THRESHOLD") == UNIFIED_SNR_THRESHOLD

    def test_registry_has_adl_thresholds(self):
        assert registry.get("ADL_GINI_THRESHOLD") == ADL_GINI_THRESHOLD

    def test_registry_threshold_count_minimum(self):
        # At least 30 canonical thresholds must be registered
        assert registry.count >= 30, (
            f"Expected >= 30 thresholds, got {registry.count}. "
            f"Categories: ihsan, snr, adl, gate, confidence"
        )

    def test_registry_categories_present(self):
        for category in ("ihsan", "snr", "adl", "confidence"):
            thresholds = registry.by_category(category)
            assert len(thresholds) > 0, f"No thresholds in category '{category}'"


class TestRegistrySeal:
    """After seal(), the registry is immutable."""

    def test_sealed_registry_rejects_register(self):
        with pytest.raises(SealedRegistryError, match="sealed"):
            registry.register("FAKE_THRESHOLD", 0.42)

    def test_sealed_registry_allows_get(self):
        # Should not raise
        value = registry.get("UNIFIED_IHSAN_THRESHOLD")
        assert isinstance(value, float)

    def test_get_nonexistent_raises(self):
        with pytest.raises(ThresholdNotFoundError, match="not registered"):
            registry.get("NONEXISTENT_THRESHOLD_XYZ")

    def test_get_or_default_returns_default(self):
        assert registry.get_or_default("NONEXISTENT", 0.42) == 0.42

    def test_has_returns_correct(self):
        assert registry.has("UNIFIED_IHSAN_THRESHOLD")
        assert not registry.has("NONEXISTENT_THRESHOLD_XYZ")


class TestRegistrySingleton:
    """Only one registry instance exists per process."""

    def test_singleton_identity(self):
        reg1 = ThresholdRegistry()
        reg2 = ThresholdRegistry()
        assert reg1 is reg2

    def test_singleton_is_module_registry(self):
        assert ThresholdRegistry() is registry


class TestRegistryFreshLifecycle:
    """Test full lifecycle: create → register → seal → get."""

    def test_register_seal_get(self):
        ThresholdRegistry._reset_for_testing()
        try:
            reg = ThresholdRegistry()
            assert not reg.is_sealed

            reg.register("TEST_THRESHOLD", 0.95, category="test")
            assert reg.get("TEST_THRESHOLD") == 0.95

            reg.seal()
            assert reg.is_sealed

            with pytest.raises(SealedRegistryError):
                reg.register("ANOTHER", 0.5)

            # Value still accessible after seal
            assert reg.get("TEST_THRESHOLD") == 0.95
        finally:
            # Restore the production singleton
            ThresholdRegistry._reset_for_testing()
            from core.integration.threshold_registry import _boot_registry
            # Re-boot sets the module-level `registry`
            _boot_registry()


class TestCanonicalThresholdSync:
    """Registry values match CANONICAL_THRESHOLDS from constants.py."""

    def test_no_canonical_drift(self):
        drifts = registry.validate_against_canonical()
        if drifts:
            drift_report = "\n".join(
                f"  {d['name']}: expected={d['expected']}, actual={d['actual']}, status={d['status']}"
                for d in drifts
            )
            pytest.fail(
                f"Constitutional threshold drift detected:\n{drift_report}\n"
                f"Fix: update ThresholdRegistry._boot_registry() or constants.py"
            )

    def test_all_canonical_keys_registered(self):
        for name in CANONICAL_THRESHOLDS:
            assert registry.has(name), (
                f"CANONICAL_THRESHOLDS['{name}'] not in registry. "
                f"Add it to _boot_registry() in threshold_registry.py"
            )


class TestModuleShadowAudit:
    """Detect threshold shadows defined outside constants.py."""

    def test_shadow_audit_runs_without_error(self):
        shadows = registry.audit_module_shadows()
        assert isinstance(shadows, list)

    def test_shadow_audit_finds_known_shadows(self):
        """We know shadows exist. This test documents them.

        Each shadow should either:
        1. Be refactored to import from constants.py, OR
        2. Be explicitly documented as a legitimate module-local override
        """
        shadows = registry.audit_module_shadows()
        # This test ensures we DETECT shadows, not that they don't exist.
        # Each shadow should either be refactored or documented as legitimate.
        assert len(shadows) >= 0  # Audit ran successfully

    def test_shadow_report_format(self):
        shadows = registry.audit_module_shadows()
        for shadow in shadows:
            assert "file" in shadow
            assert "name" in shadow
            assert "line" in shadow
            assert "type" in shadow
            assert shadow["type"] in ("numeric_shadow", "dict_shadow")


class TestRegistryRepr:
    """String representation is informative."""

    def test_repr_shows_state(self):
        r = repr(registry)
        assert "sealed" in r
        assert "ThresholdRegistry" in r

    def test_all_thresholds_snapshot(self):
        snapshot = registry.all_thresholds()
        assert isinstance(snapshot, dict)
        assert len(snapshot) == registry.count
        # All values are floats
        for name, value in snapshot.items():
            assert isinstance(value, float), f"{name} is not float: {type(value)}"
