"""Tests for core.integration.constants -- Authoritative threshold constants.

Covers:
- All Ihsan thresholds are within valid [0.0, 1.0] range
- All SNR thresholds are within valid [0.0, 1.0] range
- Ihsan weight dictionary sums to 1.0
- Tier ordering (T0 > T1 > T2 > T3 > T4)
- Four Pillars thresholds
- ADL (justice) constants
- Timing constants are positive
- Network constants
- CANONICAL_THRESHOLDS consistency
- Module __all__ exports
- validate_cross_repo_consistency function
"""

import math

import pytest

from core.integration.constants import (
    A2A_PORT_OFFSET,
    ADL_GINI_THRESHOLD,
    ADL_HARBERGER_TAX_RATE,
    ADL_MINIMUM_HOLDING,
    CANONICAL_THRESHOLDS,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_MINIMUM,
    CROSS_REPO_CONSTANTS,
    DEFAULT_FEDERATION_BIND,
    GENESIS_CUTOFF_HOURS,
    IHSAN_THRESHOLD,
    IHSAN_THRESHOLD_CI,
    IHSAN_THRESHOLD_DEV,
    IHSAN_THRESHOLD_PRODUCTION,
    IHSAN_THRESHOLD_STAGING,
    IHSAN_WEIGHTS,
    LMSTUDIO_URL,
    MAX_RETRY_ATTEMPTS,
    MODEL_DIR,
    MUSEUM_SNR_FLOOR,
    OLLAMA_URL,
    PILLAR_1_RUNTIME_IHSAN,
    PILLAR_2_MUSEUM_SNR_FLOOR,
    PILLAR_3_SANDBOX_SNR_FLOOR,
    RUNTIME_IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    SNR_THRESHOLD_T2_STANDARD,
    SNR_THRESHOLD_T3_ACCEPTABLE,
    SNR_THRESHOLD_T4_MINIMUM,
    STRICT_IHSAN_THRESHOLD,
    UNIFIED_AGENT_TIMEOUT_MS,
    UNIFIED_CLOCK_SKEW_SECONDS,
    UNIFIED_CONSENSUS_INTERVAL_SECONDS,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_NONCE_TTL_SECONDS,
    UNIFIED_SNR_THRESHOLD,
    UNIFIED_SYNC_INTERVAL_SECONDS,
    validate_cross_repo_consistency,
)


# ---------------------------------------------------------------------------
# IHSAN THRESHOLD RANGE TESTS
# ---------------------------------------------------------------------------


class TestIhsanThresholds:

    @pytest.mark.parametrize("name,value", [
        ("UNIFIED_IHSAN_THRESHOLD", UNIFIED_IHSAN_THRESHOLD),
        ("IHSAN_THRESHOLD", IHSAN_THRESHOLD),
        ("STRICT_IHSAN_THRESHOLD", STRICT_IHSAN_THRESHOLD),
        ("RUNTIME_IHSAN_THRESHOLD", RUNTIME_IHSAN_THRESHOLD),
        ("IHSAN_THRESHOLD_PRODUCTION", IHSAN_THRESHOLD_PRODUCTION),
        ("IHSAN_THRESHOLD_STAGING", IHSAN_THRESHOLD_STAGING),
        ("IHSAN_THRESHOLD_CI", IHSAN_THRESHOLD_CI),
        ("IHSAN_THRESHOLD_DEV", IHSAN_THRESHOLD_DEV),
    ])
    def test_ihsan_in_valid_range(self, name, value):
        assert 0.0 <= value <= 1.0, f"{name}={value} out of [0.0, 1.0]"

    def test_canonical_values(self):
        """Verify canonical Ihsan values from the spec."""
        assert UNIFIED_IHSAN_THRESHOLD == 0.95
        assert IHSAN_THRESHOLD == 0.95
        assert STRICT_IHSAN_THRESHOLD == 0.99
        assert RUNTIME_IHSAN_THRESHOLD == 1.0

    def test_alias_matches_unified(self):
        assert IHSAN_THRESHOLD == UNIFIED_IHSAN_THRESHOLD

    def test_environment_ordering(self):
        """More relaxed environments should have lower thresholds."""
        assert IHSAN_THRESHOLD_PRODUCTION >= IHSAN_THRESHOLD_CI
        assert IHSAN_THRESHOLD_CI >= IHSAN_THRESHOLD_DEV
        assert IHSAN_THRESHOLD_PRODUCTION == IHSAN_THRESHOLD_STAGING


# ---------------------------------------------------------------------------
# SNR THRESHOLD RANGE TESTS
# ---------------------------------------------------------------------------


class TestSNRThresholds:

    @pytest.mark.parametrize("name,value", [
        ("UNIFIED_SNR_THRESHOLD", UNIFIED_SNR_THRESHOLD),
        ("SNR_THRESHOLD", SNR_THRESHOLD),
        ("MUSEUM_SNR_FLOOR", MUSEUM_SNR_FLOOR),
        ("SNR_THRESHOLD_T0_ELITE", SNR_THRESHOLD_T0_ELITE),
        ("SNR_THRESHOLD_T1_HIGH", SNR_THRESHOLD_T1_HIGH),
        ("SNR_THRESHOLD_T2_STANDARD", SNR_THRESHOLD_T2_STANDARD),
        ("SNR_THRESHOLD_T3_ACCEPTABLE", SNR_THRESHOLD_T3_ACCEPTABLE),
        ("SNR_THRESHOLD_T4_MINIMUM", SNR_THRESHOLD_T4_MINIMUM),
    ])
    def test_snr_in_valid_range(self, name, value):
        assert 0.0 <= value <= 1.0, f"{name}={value} out of [0.0, 1.0]"

    def test_canonical_snr_values(self):
        assert UNIFIED_SNR_THRESHOLD == 0.85
        assert SNR_THRESHOLD == 0.85
        assert MUSEUM_SNR_FLOOR == 0.85

    def test_alias_matches_unified(self):
        assert SNR_THRESHOLD == UNIFIED_SNR_THRESHOLD

    def test_tier_ordering(self):
        """Tiers must be strictly ordered: T0 > T1 > T2 > T3 > T4."""
        assert SNR_THRESHOLD_T0_ELITE > SNR_THRESHOLD_T1_HIGH
        assert SNR_THRESHOLD_T1_HIGH > SNR_THRESHOLD_T2_STANDARD
        assert SNR_THRESHOLD_T2_STANDARD > SNR_THRESHOLD_T3_ACCEPTABLE
        assert SNR_THRESHOLD_T3_ACCEPTABLE > SNR_THRESHOLD_T4_MINIMUM


# ---------------------------------------------------------------------------
# IHSAN WEIGHTS TESTS
# ---------------------------------------------------------------------------


class TestIhsanWeights:

    def test_weights_sum_to_one(self):
        total = sum(IHSAN_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_expected_dimensions(self):
        expected_keys = {
            "correctness", "safety", "user_benefit", "efficiency",
            "auditability", "anti_centralization", "robustness", "adl_fairness",
        }
        assert set(IHSAN_WEIGHTS.keys()) == expected_keys

    def test_all_weights_positive(self):
        for key, weight in IHSAN_WEIGHTS.items():
            assert weight > 0.0, f"Weight for '{key}' must be positive"

    def test_safety_and_correctness_highest(self):
        """Safety and correctness should be the highest weighted dimensions."""
        max_weight = max(IHSAN_WEIGHTS.values())
        assert IHSAN_WEIGHTS["safety"] == max_weight
        assert IHSAN_WEIGHTS["correctness"] == max_weight


# ---------------------------------------------------------------------------
# FOUR PILLARS TESTS
# ---------------------------------------------------------------------------


class TestFourPillars:

    def test_pillar_1_runtime(self):
        assert PILLAR_1_RUNTIME_IHSAN == 1.0

    def test_pillar_2_museum(self):
        assert PILLAR_2_MUSEUM_SNR_FLOOR == 0.85

    def test_pillar_3_sandbox(self):
        assert PILLAR_3_SANDBOX_SNR_FLOOR == 0.70

    def test_pillar_4_genesis_cutoff(self):
        assert GENESIS_CUTOFF_HOURS == 72
        assert GENESIS_CUTOFF_HOURS > 0

    def test_pillar_ordering(self):
        """Runtime > Museum > Sandbox in strictness."""
        assert PILLAR_1_RUNTIME_IHSAN > PILLAR_2_MUSEUM_SNR_FLOOR
        assert PILLAR_2_MUSEUM_SNR_FLOOR > PILLAR_3_SANDBOX_SNR_FLOOR


# ---------------------------------------------------------------------------
# CONFIDENCE THRESHOLDS
# ---------------------------------------------------------------------------


class TestConfidenceThresholds:

    @pytest.mark.parametrize("name,value", [
        ("CONFIDENCE_HIGH", CONFIDENCE_HIGH),
        ("CONFIDENCE_MEDIUM", CONFIDENCE_MEDIUM),
        ("CONFIDENCE_LOW", CONFIDENCE_LOW),
        ("CONFIDENCE_MINIMUM", CONFIDENCE_MINIMUM),
    ])
    def test_confidence_in_valid_range(self, name, value):
        assert 0.0 <= value <= 1.0, f"{name}={value} out of [0.0, 1.0]"

    def test_confidence_ordering(self):
        assert CONFIDENCE_HIGH > CONFIDENCE_MEDIUM
        assert CONFIDENCE_MEDIUM > CONFIDENCE_LOW
        assert CONFIDENCE_LOW > CONFIDENCE_MINIMUM


# ---------------------------------------------------------------------------
# ADL (JUSTICE) INVARIANT TESTS
# ---------------------------------------------------------------------------


class TestADLConstants:

    def test_gini_threshold(self):
        assert 0.0 < ADL_GINI_THRESHOLD < 1.0
        assert ADL_GINI_THRESHOLD == 0.40

    def test_harberger_tax_rate(self):
        assert 0.0 < ADL_HARBERGER_TAX_RATE < 1.0
        assert ADL_HARBERGER_TAX_RATE == 0.05

    def test_minimum_holding(self):
        assert ADL_MINIMUM_HOLDING > 0.0
        assert ADL_MINIMUM_HOLDING == 1e-9


# ---------------------------------------------------------------------------
# TIMING CONSTANTS
# ---------------------------------------------------------------------------


class TestTimingConstants:

    def test_all_positive(self):
        assert UNIFIED_CLOCK_SKEW_SECONDS > 0
        assert UNIFIED_NONCE_TTL_SECONDS > 0
        assert UNIFIED_SYNC_INTERVAL_SECONDS > 0
        assert UNIFIED_CONSENSUS_INTERVAL_SECONDS > 0
        assert UNIFIED_AGENT_TIMEOUT_MS > 0

    def test_nonce_ttl_exceeds_clock_skew(self):
        """Nonce TTL should be larger than clock skew to prevent replay."""
        assert UNIFIED_NONCE_TTL_SECONDS > UNIFIED_CLOCK_SKEW_SECONDS


# ---------------------------------------------------------------------------
# NETWORK CONSTANTS
# ---------------------------------------------------------------------------


class TestNetworkConstants:

    def test_federation_bind_format(self):
        assert ":" in DEFAULT_FEDERATION_BIND
        host, port = DEFAULT_FEDERATION_BIND.rsplit(":", 1)
        assert port.isdigit()

    def test_a2a_port_offset_positive(self):
        assert A2A_PORT_OFFSET > 0

    def test_max_retry_attempts_positive(self):
        assert MAX_RETRY_ATTEMPTS > 0
        assert MAX_RETRY_ATTEMPTS == 3


# ---------------------------------------------------------------------------
# INFERENCE CONSTANTS
# ---------------------------------------------------------------------------


class TestInferenceConstants:

    def test_lmstudio_url_is_http(self):
        assert LMSTUDIO_URL.startswith("http")

    def test_ollama_url_is_http(self):
        assert OLLAMA_URL.startswith("http")

    def test_model_dir_is_string(self):
        assert isinstance(MODEL_DIR, str)
        assert len(MODEL_DIR) > 0


# ---------------------------------------------------------------------------
# CANONICAL THRESHOLDS DICT
# ---------------------------------------------------------------------------


class TestCanonicalThresholds:

    def test_contains_expected_keys(self):
        expected = {
            "IHSAN_THRESHOLD",
            "SNR_THRESHOLD_MINIMUM",
            "SNR_THRESHOLD_T0_ELITE",
            "MUSEUM_SNR_FLOOR",
            "RUNTIME_IHSAN",
            "ADL_GINI_THRESHOLD",
        }
        assert set(CANONICAL_THRESHOLDS.keys()) == expected

    def test_values_match_module_constants(self):
        assert CANONICAL_THRESHOLDS["IHSAN_THRESHOLD"] == UNIFIED_IHSAN_THRESHOLD
        assert CANONICAL_THRESHOLDS["SNR_THRESHOLD_MINIMUM"] == UNIFIED_SNR_THRESHOLD
        assert CANONICAL_THRESHOLDS["SNR_THRESHOLD_T0_ELITE"] == SNR_THRESHOLD_T0_ELITE
        assert CANONICAL_THRESHOLDS["MUSEUM_SNR_FLOOR"] == MUSEUM_SNR_FLOOR
        assert CANONICAL_THRESHOLDS["RUNTIME_IHSAN"] == RUNTIME_IHSAN_THRESHOLD
        assert CANONICAL_THRESHOLDS["ADL_GINI_THRESHOLD"] == ADL_GINI_THRESHOLD

    def test_all_values_are_numeric(self):
        for key, value in CANONICAL_THRESHOLDS.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"


# ---------------------------------------------------------------------------
# CROSS-REPO SYNC
# ---------------------------------------------------------------------------


class TestCrossRepoConstants:

    def test_has_expected_repos(self):
        assert "bizra-data-lake" in CROSS_REPO_CONSTANTS
        assert "dual-agentic-system" in CROSS_REPO_CONSTANTS
        assert "bizra-omega-rust" in CROSS_REPO_CONSTANTS

    def test_validate_cross_repo_returns_dict(self):
        result = validate_cross_repo_consistency()
        assert isinstance(result, dict)
        # Each repo should have a status entry
        for repo in CROSS_REPO_CONSTANTS:
            assert repo in result
            assert "status" in result[repo]


# ---------------------------------------------------------------------------
# MODULE __all__ EXPORTS
# ---------------------------------------------------------------------------


class TestModuleExports:

    def test_all_exports_importable(self):
        """Every name in core.integration.__all__ should be importable."""
        import core.integration as integration_module

        for name in integration_module.__all__:
            # Use getattr which triggers lazy imports
            obj = getattr(integration_module, name)
            assert obj is not None, f"__all__ export '{name}' resolved to None"

    def test_all_contains_expected_names(self):
        import core.integration as integration_module

        expected = {
            "IntegrationBridge",
            "BridgeConfig",
            "create_integrated_system",
            "UNIFIED_IHSAN_THRESHOLD",
            "UNIFIED_SNR_THRESHOLD",
            "UNIFIED_CLOCK_SKEW_SECONDS",
        }
        assert set(integration_module.__all__) == expected
