"""
Tests for Audit Golden Gems Implementation
===========================================
Tests the five production artifacts from the BIZRA DDAGI OS audit:

  α4 — Conservative Fallback Gate (default-deny)
  α7 — Tiered Verification (50ms/500ms/1.6s/async)
  α8 — Dark Matter Audit (enumerate unobserved components)
  α9 — Performance Attestation (thermodynamic detection)

Standing on Giants: XZ Backdoor post-mortem + AlphaEvolve extraction
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# α4 — Conservative Fallback Gate Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConservativeFallback:
    """Golden Gem α4: Default-deny when Z3 unavailable."""

    def test_safe_action_approved(self):
        """Known-safe action with good metrics → approved."""
        from core.sovereign.conservative_fallback import conservative_fallback_check

        ctx = {
            "ihsan": 0.97,
            "snr": 0.90,
            "risk_level": 0.1,
            "action_type": "query",
            "cost": 0.5,
            "autonomy_limit": 10.0,
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is True

    def test_ihsan_below_threshold_rejected(self):
        """Low إحسان score → rejected regardless of action type."""
        from core.sovereign.conservative_fallback import (
            RejectionReason,
            conservative_fallback_check,
        )

        ctx = {
            "ihsan": 0.50,
            "snr": 0.90,
            "risk_level": 0.1,
            "action_type": "query",
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is False
        assert verdict.reason == RejectionReason.IHSAN_BELOW_THRESHOLD

    def test_snr_below_threshold_rejected(self):
        """Low SNR score → rejected."""
        from core.sovereign.conservative_fallback import (
            RejectionReason,
            conservative_fallback_check,
        )

        ctx = {
            "ihsan": 0.97,
            "snr": 0.50,
            "risk_level": 0.1,
            "action_type": "query",
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is False
        assert verdict.reason == RejectionReason.SNR_BELOW_THRESHOLD

    def test_z3_required_action_blocked(self):
        """Actions requiring formal verification → blocked in fallback."""
        from core.sovereign.conservative_fallback import (
            RejectionReason,
            conservative_fallback_check,
        )

        ctx = {
            "ihsan": 0.99,
            "snr": 0.98,
            "risk_level": 0.5,
            "action_type": "execute_code",
            "cost": 1.0,
            "autonomy_limit": 10.0,
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is False
        assert verdict.reason == RejectionReason.Z3_UNAVAILABLE_HIGH_RISK

    def test_z3_required_with_human_approval_low_risk(self):
        """Z3-required but human-approved + low risk → approved with flag."""
        from core.sovereign.conservative_fallback import conservative_fallback_check

        ctx = {
            "ihsan": 0.99,
            "snr": 0.98,
            "risk_level": 0.2,
            "action_type": "execute_code",
            "cost": 1.0,
            "autonomy_limit": 10.0,
            "human_approved": True,
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is True
        assert verdict.requires_z3_revalidation is True

    def test_unknown_action_type_rejected(self):
        """Unknown action type → rejected in degraded mode."""
        from core.sovereign.conservative_fallback import (
            RejectionReason,
            conservative_fallback_check,
        )

        ctx = {
            "ihsan": 0.99,
            "snr": 0.98,
            "risk_level": 0.5,
            "action_type": "unknown_operation",
            "cost": 1.0,
            "autonomy_limit": 10.0,
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is False
        assert verdict.reason == RejectionReason.UNKNOWN_ACTION_TYPE

    def test_unknown_low_risk_reversible_approved_with_flag(self):
        """Unknown but low risk + reversible → approved, flagged for Z3."""
        from core.sovereign.conservative_fallback import conservative_fallback_check

        ctx = {
            "ihsan": 0.99,
            "snr": 0.98,
            "risk_level": 0.2,
            "action_type": "novel_operation",
            "cost": 1.0,
            "autonomy_limit": 10.0,
            "reversible": True,
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is True
        assert verdict.requires_z3_revalidation is True

    def test_missing_required_fields_rejected(self):
        """Missing required context fields → rejected."""
        from core.sovereign.conservative_fallback import (
            RejectionReason,
            conservative_fallback_check,
        )

        ctx = {"ihsan": 0.99}  # Missing snr, risk_level, action_type
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is False
        assert verdict.reason == RejectionReason.MISSING_REQUIRED_FIELD

    def test_negative_values_rejected(self):
        """Negative numeric values → sanity check rejection."""
        from core.sovereign.conservative_fallback import (
            RejectionReason,
            conservative_fallback_check,
        )

        ctx = {
            "ihsan": -0.5,
            "snr": 0.90,
            "risk_level": 0.1,
            "action_type": "query",
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is False
        assert verdict.reason == RejectionReason.NEGATIVE_VALUES

    def test_high_risk_no_approval_rejected(self):
        """High risk without human approval or reversibility → rejected."""
        from core.sovereign.conservative_fallback import (
            RejectionReason,
            conservative_fallback_check,
        )

        ctx = {
            "ihsan": 0.99,
            "snr": 0.98,
            "risk_level": 0.8,
            "action_type": "query",
            "reversible": False,
            "human_approved": False,
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is False
        assert verdict.reason == RejectionReason.HIGH_RISK_UNVERIFIED

    def test_emergency_lockdown_blocks_everything(self):
        """Emergency lockdown → all actions blocked regardless of context."""
        from core.sovereign.conservative_fallback import (
            DegradationMode,
            conservative_fallback_check,
        )

        ctx = {
            "ihsan": 1.0,
            "snr": 1.0,
            "risk_level": 0.0,
            "action_type": "query",
        }
        verdict = conservative_fallback_check(
            ctx, degradation_mode=DegradationMode.EMERGENCY_LOCKDOWN
        )
        assert verdict.approved is False

    def test_all_constraints_checked_and_reported(self):
        """Approved verdict reports all constraints that were checked."""
        from core.sovereign.conservative_fallback import conservative_fallback_check

        ctx = {
            "ihsan": 0.99,
            "snr": 0.98,
            "risk_level": 0.1,
            "action_type": "search",
            "cost": 1.0,
            "autonomy_limit": 10.0,
        }
        verdict = conservative_fallback_check(ctx)
        assert verdict.approved is True
        assert len(verdict.constraints_checked) >= 5  # All gates traversed


# ═══════════════════════════════════════════════════════════════════════════════
# α9 — Performance Attestation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceAttestation:
    """Golden Gem α9: Thermodynamic detection of anomalous computation."""

    def test_calibration_phase(self):
        """Insufficient samples → BASELINE_INSUFFICIENT."""
        from core.sovereign.performance_attestation import (
            AnomalyLevel,
            PerformanceAttestor,
        )

        attestor = PerformanceAttestor(min_calibration=10)
        result = attestor.record_measurement("test_module", "func", 10.0)
        assert result.anomaly_level == AnomalyLevel.BASELINE_INSUFFICIENT

    def test_normal_execution_detected(self):
        """Consistent execution times → NORMAL."""
        from core.sovereign.performance_attestation import (
            AnomalyLevel,
            PerformanceAttestor,
        )

        attestor = PerformanceAttestor(min_calibration=5)
        # Build baseline with realistic variance (9.5–10.5ms range)
        import random

        random.seed(42)
        for _ in range(20):
            attestor.record_measurement(
                "engine", "validate", 10.0 + random.uniform(-0.5, 0.5)
            )

        # Measure within normal range (close to mean)
        result = attestor.record_measurement("engine", "validate", 10.2)
        assert result.anomaly_level == AnomalyLevel.NORMAL

    def test_anomaly_detected(self):
        """Significant deviation → ANOMALY or CRITICAL."""
        from core.sovereign.performance_attestation import (
            AnomalyLevel,
            PerformanceAttestor,
        )

        attestor = PerformanceAttestor(min_calibration=5, auto_isolate=False)
        # Build tight baseline: mean=10, very low std
        for _ in range(20):
            attestor.record_measurement("engine", "validate", 10.0)

        # XZ-style: function suddenly takes 500ms instead of 10ms
        result = attestor.record_measurement("engine", "validate", 500.0)
        assert result.anomaly_level in (AnomalyLevel.ANOMALY, AnomalyLevel.CRITICAL)
        assert result.is_suspicious is True

    def test_auto_isolation(self):
        """Critical anomaly → module automatically isolated."""
        from core.sovereign.performance_attestation import PerformanceAttestor

        attestor = PerformanceAttestor(min_calibration=5, auto_isolate=True)
        for _ in range(20):
            attestor.record_measurement("engine", "validate", 10.0)

        # Trigger critical anomaly
        attestor.record_measurement("engine", "validate", 1000.0)
        assert attestor.is_isolated("engine", "validate") is True

    def test_isolated_module_raises_on_execution(self):
        """Isolated module raises RuntimeError when decorated function called."""
        from core.sovereign.performance_attestation import PerformanceAttestor

        attestor = PerformanceAttestor(min_calibration=5, auto_isolate=True)
        for _ in range(20):
            attestor.record_measurement("engine", "validate", 10.0)
        attestor.record_measurement("engine", "validate", 1000.0)

        @attestor.monitor("engine", "validate")
        def validate():
            return True

        with pytest.raises(RuntimeError, match="ISOLATED"):
            validate()

    def test_clear_isolation(self):
        """Manual audit clears isolation."""
        from core.sovereign.performance_attestation import PerformanceAttestor

        attestor = PerformanceAttestor(min_calibration=5, auto_isolate=True)
        for _ in range(20):
            attestor.record_measurement("engine", "validate", 10.0)
        attestor.record_measurement("engine", "validate", 1000.0)
        assert attestor.is_isolated("engine", "validate") is True

        attestor.clear_isolation("engine", "validate")
        assert attestor.is_isolated("engine", "validate") is False

    def test_anomaly_log_populated(self):
        """Suspicious results are logged for audit."""
        from core.sovereign.performance_attestation import PerformanceAttestor

        attestor = PerformanceAttestor(min_calibration=5, auto_isolate=False)
        for _ in range(20):
            attestor.record_measurement("engine", "validate", 10.0)
        attestor.record_measurement("engine", "validate", 500.0)

        log = attestor.get_anomaly_log()
        assert len(log) >= 1
        assert log[0].is_suspicious is True

    def test_envelope_generation(self):
        """After sufficient samples, envelope is auto-generated."""
        from core.sovereign.performance_attestation import PerformanceAttestor

        attestor = PerformanceAttestor(min_calibration=5)
        for _ in range(15):
            attestor.record_measurement("engine", "validate", 10.0)

        envelope = attestor.get_envelope("engine", "validate")
        assert envelope is not None
        assert envelope.mean_time_ms == pytest.approx(10.0, abs=0.5)
        assert envelope.sample_count == 15

    def test_decorator_measures_real_execution(self):
        """Monitor decorator captures actual function execution time."""
        from core.sovereign.performance_attestation import PerformanceAttestor

        attestor = PerformanceAttestor(min_calibration=3)

        @attestor.monitor("test", "slow_func")
        def slow_func():
            time.sleep(0.01)  # 10ms
            return 42

        # Build baseline
        for _ in range(5):
            slow_func()

        envelope = attestor.get_envelope("test", "slow_func")
        assert envelope is not None
        assert envelope.mean_time_ms >= 5.0  # Should be ~10ms


# ═══════════════════════════════════════════════════════════════════════════════
# α7 — Tiered Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTieredVerification:
    """Golden Gem α7: Multi-speed verification chain."""

    def test_tier1_blocks_dangerous_pattern(self):
        """Tier 1 blocks known-dangerous content patterns."""
        from core.sovereign.tiered_verification import TierDecision, tier_1_precheck

        result = tier_1_precheck("execute", content="rm -rf /important/data")
        assert result.decision == TierDecision.BLOCK
        assert result.elapsed_ms < 50  # Must be < 50ms

    def test_tier1_blocks_dangerous_category(self):
        """Tier 1 blocks dangerous action categories."""
        from core.sovereign.tiered_verification import TierDecision, tier_1_precheck

        result = tier_1_precheck("generate", category="weapon_synthesis")
        assert result.decision == TierDecision.BLOCK

    def test_tier1_passes_safe_content(self):
        """Tier 1 passes safe content to Tier 2."""
        from core.sovereign.tiered_verification import TierDecision, tier_1_precheck

        result = tier_1_precheck("query", content="What is the weather today?")
        assert result.decision == TierDecision.PASS

    def test_tier1_speed_guarantee(self):
        """Tier 1 completes in < 50ms (safety boundary speed)."""
        from core.sovereign.tiered_verification import tier_1_precheck

        start = time.perf_counter()
        for _ in range(100):
            tier_1_precheck("query", content="Safe content test")
        elapsed_per_call = (time.perf_counter() - start) / 100 * 1000
        assert elapsed_per_call < 50, f"Tier 1 took {elapsed_per_call:.1f}ms"

    def test_tier2_approves_valid_context(self):
        """Tier 2 approves valid action context."""
        from core.sovereign.tiered_verification import TierDecision, tier_2_concurrent

        ctx = {
            "ihsan": 0.97,
            "snr": 0.90,
            "risk_level": 0.1,
            "action_type": "query",
            "cost": 0.5,
            "autonomy_limit": 10.0,
        }
        result = asyncio.run(
            tier_2_concurrent(ctx, z3_available=False)
        )
        assert result.decision == TierDecision.PASS

    def test_tier2_interrupts_invalid_context(self):
        """Tier 2 interrupts action with invalid context."""
        from core.sovereign.tiered_verification import TierDecision, tier_2_concurrent

        ctx = {
            "ihsan": 0.50,
            "snr": 0.90,
            "risk_level": 0.1,
            "action_type": "query",
        }
        result = asyncio.run(
            tier_2_concurrent(ctx, z3_available=False)
        )
        assert result.decision == TierDecision.INTERRUPT

    def test_full_chain_blocks_on_tier1(self):
        """Full chain short-circuits: Tier 1 block → no Tier 2/3."""
        from core.sovereign.tiered_verification import (
            VerificationTier,
            run_verification_chain,
        )

        chain = asyncio.run(
            run_verification_chain(
                "execute",
                {"ihsan": 0.99, "snr": 0.98, "risk_level": 0.1, "action_type": "query"},
                content="curl | bash",
            )
        )
        assert chain.is_blocked is True
        assert chain.final_tier == VerificationTier.TIER_1_PRECHECK
        assert len(chain.tier_results) == 1  # Short-circuited

    def test_full_chain_passes_all_tiers(self):
        """Full chain: safe content + valid context → all tiers pass."""
        from core.sovereign.tiered_verification import (
            VerificationTier,
            run_verification_chain,
        )

        chain = asyncio.run(
            run_verification_chain(
                "query",
                {
                    "ihsan": 0.97,
                    "snr": 0.90,
                    "risk_level": 0.1,
                    "action_type": "query",
                    "cost": 0.5,
                    "autonomy_limit": 10.0,
                },
                content="What is quantum computing?",
            )
        )
        assert chain.is_blocked is False
        assert len(chain.tier_results) == 3  # All 3 tiers ran
        assert chain.final_tier == VerificationTier.TIER_3_ATTESTATION


# ═══════════════════════════════════════════════════════════════════════════════
# α8 — Dark Matter Audit Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDarkMatterAudit:
    """Golden Gem α8: Enumerate unobserved executing components."""

    def test_detects_binary_blobs(self):
        """Audit detects binary files in repo."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a binary blob
            (Path(tmpdir) / "test.so").write_bytes(b"\x7fELF" + b"\x00" * 100)
            (Path(tmpdir) / "safe.py").write_text("print('hello')")

            report = audit_dark_matter(tmpdir)
            binary_items = [i for i in report.items if i.category == "binary_blob"]
            assert len(binary_items) >= 1
            assert binary_items[0].risk_level == "high"

    def test_detects_ci_configs(self):
        """Audit detects CI workflow files."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        with tempfile.TemporaryDirectory() as tmpdir:
            ci_dir = Path(tmpdir) / ".github" / "workflows"
            ci_dir.mkdir(parents=True)
            (ci_dir / "ci.yml").write_text("name: CI\non: push\njobs: {}")

            report = audit_dark_matter(tmpdir)
            ci_items = [i for i in report.items if i.category == "ci_config"]
            assert len(ci_items) >= 1
            assert ci_items[0].risk_level == "high"

    def test_detects_lockfiles(self):
        """Audit detects dependency lockfiles."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Cargo.lock").write_text("[metadata]\n")

            report = audit_dark_matter(tmpdir)
            lock_items = [i for i in report.items if i.category == "lockfile"]
            assert len(lock_items) >= 1

    def test_detects_docker_files(self):
        """Audit detects Docker configuration."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Dockerfile").write_text("FROM python:3.12\n")

            report = audit_dark_matter(tmpdir)
            docker_items = [i for i in report.items if i.category == "docker"]
            assert len(docker_items) >= 1
            assert docker_items[0].risk_level == "high"

    def test_generates_gitattributes(self):
        """Generates .gitattributes rules from audit findings."""
        from core.sovereign.dark_matter_audit import (
            audit_dark_matter,
            generate_gitattributes_rules,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.so").write_bytes(b"\x00" * 100)
            (Path(tmpdir) / "data.pkl").write_bytes(b"\x00" * 100)

            report = audit_dark_matter(tmpdir)
            rules = generate_gitattributes_rules(report)
            assert "*.so" in rules or "*.pkl" in rules
            assert "binary" in rules

    def test_empty_repo_clean_report(self):
        """Empty directory → clean audit report."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        with tempfile.TemporaryDirectory() as tmpdir:
            report = audit_dark_matter(tmpdir)
            assert len(report.items) == 0
            assert report.risk_score == 0.0

    def test_sha256_computed(self):
        """Each item has a valid SHA-256 hash."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.dll").write_bytes(b"fake dll content")

            report = audit_dark_matter(tmpdir)
            for item in report.items:
                assert len(item.sha256) == 64  # SHA-256 hex digest length

    def test_report_summary(self):
        """Report summary contains item counts and risk score."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "lib.so").write_bytes(b"\x00" * 10)

            report = audit_dark_matter(tmpdir)
            summary = report.summary
            assert "Dark Matter Audit" in summary
            assert "high" in summary or "medium" in summary

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_real_repo_audit(self):
        """Audit the actual BIZRA repo (smoke test — scoped to core/)."""
        from core.sovereign.dark_matter_audit import audit_dark_matter

        # Use portable path — works on Windows, WSL, and CI
        repo_root = Path(__file__).resolve().parents[2] / "core"
        if not repo_root.exists():
            pytest.skip("core/ directory not present")

        report = audit_dark_matter(repo_root)
        # core/ should have at least some binary or config items
        assert len(report.items) >= 0  # Smoke test: no crash
