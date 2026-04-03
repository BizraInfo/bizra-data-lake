"""
Test suite for SNR Enforcer
============================
Tests threshold enforcement, receipt emission, and integration.
"""

import json
import tempfile
from pathlib import Path

import pytest

from bizra_kernel.snr_enforcer import (
    SNREnforcer,
    SNRThresholds,
    EnforcementContext,
    OperationType,
    enforce_snr,
    get_snr_enforcer,
)
from bizra_kernel.snr_tracker import SNRMetrics, SNRTracker


class TestSNRThresholds:
    """Test threshold loading and configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = SNRThresholds()
        assert thresholds.target_snr == 0.98
        assert thresholds.minimum_snr == 0.95
        assert thresholds.escalate_below == 0.90

    def test_get_threshold_default(self):
        """Test getting default threshold."""
        thresholds = SNRThresholds()
        threshold = thresholds.get_threshold(OperationType.REASONING)
        assert threshold == 0.95  # minimum_snr

    def test_get_threshold_override(self):
        """Test getting overridden threshold."""
        thresholds = SNRThresholds(
            operation_thresholds={"reasoning": 0.97}
        )
        threshold = thresholds.get_threshold(OperationType.REASONING)
        assert threshold == 0.97

    def test_from_constitution_missing_file(self):
        """Test loading from non-existent constitution."""
        thresholds = SNRThresholds.from_constitution("nonexistent.yaml")
        # Should fall back to defaults
        assert thresholds.target_snr == 0.98


class TestEnforcementContext:
    """Test enforcement context."""

    def test_create_context(self):
        """Test creating enforcement context."""
        context = EnforcementContext(
            operation_type=OperationType.REASONING,
            agent_id="test-agent",
            snr_score=0.96,
            task_id="task-123",
            details={"foo": "bar"},
        )

        assert context.operation_type == OperationType.REASONING
        assert context.agent_id == "test-agent"
        assert context.snr_score == 0.96
        assert context.task_id == "task-123"
        assert context.details["foo"] == "bar"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        context = EnforcementContext(
            operation_type=OperationType.SYNTHESIS,
            agent_id="synth-agent",
            snr_score=0.98,
        )

        data = context.to_dict()
        assert data["operation_type"] == "synthesis"
        assert data["agent_id"] == "synth-agent"
        assert data["snr_score"] == 0.98


class TestSNREnforcer:
    """Test SNR enforcer logic."""

    def test_initialize_enforcer(self):
        """Test enforcer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(
                constitution_path="constitution/pat_enforcement_v1.yaml",
                receipt_dir=tmpdir,
            )

            assert enforcer.thresholds.target_snr >= 0.95
            assert enforcer.snr_tracker is not None
            assert enforcer.emit_receipts is True

    def test_enforce_pass(self):
        """Test enforcement passing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(receipt_dir=tmpdir)

            context = EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="test-agent",
                snr_score=0.97,  # Above minimum (0.95)
            )

            result = enforcer.enforce(context)

            assert result.passed is True
            assert result.snr_score == 0.97
            assert result.rejection_code is None
            assert result.receipt_id is None

    def test_enforce_reject(self):
        """Test enforcement rejection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(receipt_dir=tmpdir, emit_receipts=True)

            context = EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="test-agent",
                snr_score=0.93,  # Below minimum (0.95)
            )

            result = enforcer.enforce(context)

            assert result.passed is False
            assert result.snr_score == 0.93
            assert result.rejection_code == 7  # REJECT_SNR_BELOW_MIN
            assert result.receipt_id is not None
            assert "REJECTED" in result.message

    def test_receipt_emission(self):
        """Test that rejection receipts are emitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            receipt_dir = Path(tmpdir)
            enforcer = SNREnforcer(receipt_dir=receipt_dir, emit_receipts=True)

            context = EnforcementContext(
                operation_type=OperationType.VALIDATION,
                agent_id="validator",
                snr_score=0.90,  # Below threshold
                task_id="task-456",
            )

            result = enforcer.enforce(context)

            # Check receipt file was created
            receipt_files = list(receipt_dir.glob("*.jsonl"))
            assert len(receipt_files) > 0

            # Read receipt
            with open(receipt_files[0], 'r') as f:
                receipt_line = f.readline()
                receipt = json.loads(receipt_line)

            assert receipt["receipt_id"] == result.receipt_id
            assert receipt["rejection_code"] == 7
            assert receipt["snr_score"] == 0.90
            assert receipt["agent_id"] == "validator"
            assert receipt["task_id"] == "task-456"
            assert "integrity_hash" in receipt

    def test_no_receipt_when_disabled(self):
        """Test that receipts are not emitted when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            receipt_dir = Path(tmpdir)
            enforcer = SNREnforcer(receipt_dir=receipt_dir, emit_receipts=False)

            context = EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="test-agent",
                snr_score=0.90,  # Below threshold
            )

            result = enforcer.enforce(context)

            assert result.passed is False
            # No receipt files should be created
            receipt_files = list(receipt_dir.glob("*.jsonl"))
            assert len(receipt_files) == 0

    def test_statistics(self):
        """Test enforcement statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(receipt_dir=tmpdir, emit_receipts=True)

            # Pass 1
            enforcer.enforce(EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="agent-1",
                snr_score=0.97,
            ))

            # Reject 1
            enforcer.enforce(EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="agent-2",
                snr_score=0.92,
            ))

            # Reject 2
            enforcer.enforce(EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="agent-3",
                snr_score=0.88,
            ))

            stats = enforcer.get_statistics()

            assert stats["enforcements"] == 3
            assert stats["rejections"] == 2
            assert stats["rejection_rate"] == pytest.approx(2/3)
            assert stats["receipts_emitted"] == 2
            assert "thresholds" in stats

    def test_custom_thresholds(self):
        """Test enforcer with custom thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a custom constitution
            constitution_path = Path(tmpdir) / "custom_constitution.yaml"
            constitution_content = """
snr_integration:
  target_snr: 0.99
  minimum_snr: 0.97
  escalate_below: 0.92
  operation_thresholds:
    reasoning: 0.98
    synthesis: 0.96
"""
            constitution_path.write_text(constitution_content)

            enforcer = SNREnforcer(
                constitution_path=constitution_path,
                receipt_dir=tmpdir,
            )

            assert enforcer.thresholds.target_snr == 0.99
            assert enforcer.thresholds.minimum_snr == 0.97
            assert enforcer.thresholds.operation_thresholds["reasoning"] == 0.98

            # Test with reasoning operation (threshold 0.98)
            context = EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="test",
                snr_score=0.975,  # Between 0.97 and 0.98
            )

            result = enforcer.enforce(context)
            # Should be rejected because reasoning requires 0.98
            assert result.passed is False
            assert result.threshold == 0.98


class TestIntegration:
    """Test integration with SNR tracker and convenience functions."""

    def test_record_metrics(self):
        """Test recording metrics to tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = SNRTracker()
            enforcer = SNREnforcer(snr_tracker=tracker, receipt_dir=tmpdir)

            metrics = SNRMetrics(
                total_tokens=1000,
                useful_tokens=900,
                confidence_score=0.95,
                ethical_compliance=0.98,
                tool_directness=1.0,
                latency_ms=250,
                agent_role="test-agent",
            )

            enforcer.record_metrics(metrics)

            # Check tracker recorded it
            assert tracker.get_current_snr() == metrics.snr_score

    def test_convenience_function(self):
        """Test enforce_snr convenience function."""
        result = enforce_snr(
            operation_type="reasoning",
            agent_id="test-agent",
            snr_score=0.96,
            task_id="task-789",
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_async_enforcement(self):
        """Test async enforcement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(receipt_dir=tmpdir)

            context = EnforcementContext(
                operation_type=OperationType.SYNTHESIS,
                agent_id="async-agent",
                snr_score=0.97,
            )

            result = await enforcer.enforce_async(context)

            assert result.passed is True

    def test_global_enforcer(self):
        """Test global enforcer singleton."""
        enforcer1 = get_snr_enforcer()
        enforcer2 = get_snr_enforcer()

        # Should be same instance
        assert enforcer1 is enforcer2

        # Force reload
        enforcer3 = get_snr_enforcer(force_reload=True)
        assert enforcer3 is not enforcer1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_exact_threshold(self):
        """Test SNR exactly at threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(receipt_dir=tmpdir)

            context = EnforcementContext(
                operation_type=OperationType.DEFAULT,
                agent_id="test",
                snr_score=0.95,  # Exactly at minimum_snr
            )

            result = enforcer.enforce(context)
            # Should pass (>= threshold)
            assert result.passed is True

    def test_zero_snr(self):
        """Test with zero SNR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(receipt_dir=tmpdir)

            context = EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="test",
                snr_score=0.0,
            )

            result = enforcer.enforce(context)
            assert result.passed is False
            assert result.snr_score == 0.0

    def test_perfect_snr(self):
        """Test with perfect SNR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = SNREnforcer(receipt_dir=tmpdir)

            context = EnforcementContext(
                operation_type=OperationType.REASONING,
                agent_id="test",
                snr_score=1.0,
            )

            result = enforcer.enforce(context)
            assert result.passed is True
            assert result.snr_score == 1.0

    def test_invalid_operation_type(self):
        """Test with invalid operation type string."""
        result = enforce_snr(
            operation_type="invalid_type",
            agent_id="test",
            snr_score=0.96,
        )

        # Should default to DEFAULT operation type
        assert result.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
