"""Tests for core.pek.kernel -- Proactive Execution Kernel.

Covers:
- ProactiveExecutionKernelConfig defaults and boundary values
- PEKProposal data class instantiation
- PEKProofBlock data class and serialization
- ProactiveExecutionKernel: confidence, tau, attention budget, intervention modes
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# PEK depends on core.sovereign imports -- mock them before importing the kernel
with patch.dict("sys.modules", {
    "core.sovereign.autonomy_matrix": MagicMock(),
    "core.sovereign.opportunity_pipeline": MagicMock(),
    "core.sovereign.proactive_scheduler": MagicMock(),
}):
    # Provide the enums/classes that PEK references at import time
    import sys

    _autonomy_mod = sys.modules["core.sovereign.autonomy_matrix"]
    _autonomy_mod.AutonomyLevel = type("AutonomyLevel", (), {
        "AUTOLOW": "AUTOLOW",
        "OBSERVER": "OBSERVER",
        "SUGGESTER": "SUGGESTER",
    })

    _opp_mod = sys.modules["core.sovereign.opportunity_pipeline"]
    _opp_mod.OpportunityPipeline = MagicMock
    _opp_mod.PipelineOpportunity = MagicMock

    _sched_mod = sys.modules["core.sovereign.proactive_scheduler"]
    _sched_mod.JobPriority = type("JobPriority", (), {"HIGH": "HIGH", "NORMAL": "NORMAL"})
    _sched_mod.ProactiveScheduler = MagicMock
    _sched_mod.ScheduleType = type("ScheduleType", (), {"ONE_TIME": "ONE_TIME"})

    from core.pek.kernel import (
        PEKProofBlock,
        PEKProposal,
        ProactiveExecutionKernel,
        ProactiveExecutionKernelConfig,
    )


# ---------------------------------------------------------------------------
# CONFIG TESTS
# ---------------------------------------------------------------------------


class TestProactiveExecutionKernelConfig:

    def test_defaults(self):
        cfg = ProactiveExecutionKernelConfig()
        assert cfg.cycle_interval_seconds == 5.0
        assert cfg.max_proposals_per_cycle == 3
        assert cfg.min_confidence == 0.58
        assert cfg.min_snr == 0.85
        assert cfg.base_tau == 0.55
        assert cfg.attention_budget_capacity == 8.0

    def test_custom_values(self):
        cfg = ProactiveExecutionKernelConfig(
            cycle_interval_seconds=1.0,
            max_proposals_per_cycle=10,
            base_tau=0.80,
        )
        assert cfg.cycle_interval_seconds == 1.0
        assert cfg.max_proposals_per_cycle == 10
        assert cfg.base_tau == 0.80

    def test_attention_budget_params(self):
        cfg = ProactiveExecutionKernelConfig()
        assert cfg.attention_budget_recovery_per_cycle == 0.75
        assert cfg.attention_cost_auto_execute == 0.45
        assert cfg.attention_cost_propose == 1.0
        assert cfg.attention_cost_queue_silent == 0.20

    def test_proof_log_defaults(self):
        cfg = ProactiveExecutionKernelConfig()
        assert cfg.proof_log_relpath == "proofs/pek_proof_blocks.jsonl"
        assert cfg.emit_proof_events is False
        assert cfg.proof_event_topic == "pek.proof.block"


# ---------------------------------------------------------------------------
# PEKProposal TESTS
# ---------------------------------------------------------------------------


class TestPEKProposal:

    def test_instantiation(self):
        p = PEKProposal(
            id="pek-test001",
            domain="orchestration",
            action_type="queue_pressure_relief",
            description="Test proposal",
            snr_score=0.91,
            ihsan_score=0.97,
            urgency=0.8,
            estimated_value=0.78,
            risk=0.20,
        )
        assert p.id == "pek-test001"
        assert p.reversible is True  # default
        assert p.context == {}  # default

    def test_with_context(self):
        p = PEKProposal(
            id="pek-test002",
            domain="memory",
            action_type="refresh",
            description="Refresh working set",
            snr_score=0.87,
            ihsan_score=0.96,
            urgency=0.62,
            estimated_value=0.70,
            risk=0.12,
            reversible=False,
            context={"queue_utilization": 0.85},
        )
        assert p.reversible is False
        assert p.context["queue_utilization"] == 0.85


# ---------------------------------------------------------------------------
# PEKProofBlock TESTS
# ---------------------------------------------------------------------------


class TestPEKProofBlock:

    def test_to_dict(self):
        proof = PEKProofBlock(
            proposal_id="pek-001",
            action_type="queue_pressure_relief",
            decision="auto_execute",
            tau=0.75,
            confidence=0.88,
            fate_passed=True,
            fate_proof_id="fate-123",
            reason_trace=["confidence=0.880", "tau=0.750"],
            reversible=True,
            created_at=1700000000.0,
            metadata={"domain": "orchestration"},
        )
        d = proof.to_dict()
        assert d["proposal_id"] == "pek-001"
        assert d["decision"] == "auto_execute"
        assert d["tau"] == 0.75
        assert d["fate_passed"] is True
        assert d["fate_proof_id"] == "fate-123"
        assert isinstance(d["reason_trace"], list)
        assert isinstance(d["metadata"], dict)

    def test_to_dict_empty_metadata(self):
        proof = PEKProofBlock(
            proposal_id="pek-002",
            action_type="test",
            decision="ignore",
            tau=0.55,
            confidence=0.30,
            fate_passed=False,
            fate_proof_id=None,
            reason_trace=[],
            reversible=True,
            created_at=time.time(),
        )
        d = proof.to_dict()
        assert d["fate_proof_id"] is None
        assert d["metadata"] == {}


# ---------------------------------------------------------------------------
# ProactiveExecutionKernel TESTS (unit-level, no event loop)
# ---------------------------------------------------------------------------


class TestKernelConfidenceComputation:

    @pytest.fixture
    def kernel(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        return ProactiveExecutionKernel(
            opportunity_pipeline=mock_pipeline,
            config=ProactiveExecutionKernelConfig(),
        )

    def test_compute_confidence_weights(self, kernel):
        """Verify confidence = 0.35*snr + 0.35*ihsan + 0.20*value + 0.10*tau."""
        proposal = PEKProposal(
            id="test", domain="test", action_type="test",
            description="test", snr_score=1.0, ihsan_score=1.0,
            urgency=0.5, estimated_value=1.0, risk=0.1,
        )
        confidence = kernel._compute_confidence(proposal, tau=1.0)
        expected = 1.0 * 0.35 + 1.0 * 0.35 + 1.0 * 0.20 + 1.0 * 0.10
        assert abs(confidence - expected) < 1e-9

    def test_compute_confidence_zero_inputs(self, kernel):
        proposal = PEKProposal(
            id="test", domain="test", action_type="test",
            description="test", snr_score=0.0, ihsan_score=0.0,
            urgency=0.0, estimated_value=0.0, risk=0.0,
        )
        confidence = kernel._compute_confidence(proposal, tau=0.0)
        assert confidence == 0.0

    def test_compute_confidence_clamped_to_one(self, kernel):
        """Even with scores > 1.0, confidence is clamped to [0, 1]."""
        proposal = PEKProposal(
            id="test", domain="test", action_type="test",
            description="test", snr_score=2.0, ihsan_score=2.0,
            urgency=0.5, estimated_value=2.0, risk=0.1,
        )
        confidence = kernel._compute_confidence(proposal, tau=2.0)
        assert confidence <= 1.0


class TestEffectiveMinConfidence:

    @pytest.fixture
    def kernel(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        return ProactiveExecutionKernel(
            opportunity_pipeline=mock_pipeline,
            config=ProactiveExecutionKernelConfig(min_confidence=0.58),
        )

    def test_low_tau_raises_threshold(self, kernel):
        """When tau < 0.35, threshold increases by 0.06."""
        threshold = kernel._effective_min_confidence(0.20)
        assert threshold == 0.58 + 0.06

    def test_moderate_low_tau_raises_threshold(self, kernel):
        """When 0.35 <= tau < 0.50, threshold increases by 0.03."""
        threshold = kernel._effective_min_confidence(0.40)
        assert threshold == 0.58 + 0.03

    def test_normal_tau_no_change(self, kernel):
        """When 0.50 <= tau <= 0.85, threshold is unchanged."""
        threshold = kernel._effective_min_confidence(0.60)
        assert threshold == 0.58

    def test_high_tau_lowers_threshold(self, kernel):
        """When tau > 0.85, threshold decreases by 0.03."""
        threshold = kernel._effective_min_confidence(0.90)
        assert threshold == 0.58 - 0.03


class TestComputeTau:

    @pytest.fixture
    def kernel(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        return ProactiveExecutionKernel(
            opportunity_pipeline=mock_pipeline,
            config=ProactiveExecutionKernelConfig(base_tau=0.55),
        )

    def test_baseline_tau_with_empty_signals(self, kernel):
        # Empty signals: avg_snr defaults to 0.0 which is < 0.75, so -0.04
        tau = kernel._compute_tau({})
        assert tau == 0.55 - 0.04  # 0.51

    def test_high_queue_utilization_reduces_tau(self, kernel):
        # queue_util > 0.80 -> -0.15, plus memory avg_snr=0.0 < 0.75 -> -0.04
        signals = {"pipeline": {"queue_utilization": 0.85}}
        tau = kernel._compute_tau(signals)
        assert abs(tau - (0.55 - 0.15 - 0.04)) < 1e-9

    def test_gateway_offline_reduces_tau(self, kernel):
        # gateway offline -> -0.25, plus memory avg_snr=0.0 < 0.75 -> -0.04
        signals = {"gateway": {"status": "offline"}}
        tau = kernel._compute_tau(signals)
        assert abs(tau - (0.55 - 0.25 - 0.04)) < 1e-9

    def test_high_snr_memory_boosts_tau(self, kernel):
        signals = {"memory": {"avg_snr": 0.95}}
        tau = kernel._compute_tau(signals)
        assert tau == 0.55 + 0.04

    def test_tau_clamped_to_range(self, kernel):
        """Tau should always be within [0.10, 1.00]."""
        # Worst case: everything is bad
        signals = {
            "pipeline": {"queue_utilization": 0.90, "pending_approval": 20},
            "gateway": {"status": "offline", "avg_latency_ms": 5000},
            "memory": {"avg_snr": 0.5},
        }
        tau = kernel._compute_tau(signals)
        assert tau >= 0.10
        assert tau <= 1.00

    def test_tau_updates_last_tau(self, kernel):
        signals = {"pipeline": {"queue_utilization": 0.65}}
        tau = kernel._compute_tau(signals)
        assert kernel._last_tau == tau


class TestAttentionBudget:

    @pytest.fixture
    def kernel(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        return ProactiveExecutionKernel(
            opportunity_pipeline=mock_pipeline,
            config=ProactiveExecutionKernelConfig(
                attention_budget_capacity=8.0,
                attention_budget_recovery_per_cycle=0.75,
            ),
        )

    def test_initial_budget_at_capacity(self, kernel):
        assert kernel._attention_budget == 8.0

    def test_replenish_does_not_exceed_capacity(self, kernel):
        kernel._attention_budget = 7.5
        kernel._replenish_attention_budget()
        assert kernel._attention_budget == 8.0

    def test_replenish_from_zero(self, kernel):
        kernel._attention_budget = 0.0
        kernel._replenish_attention_budget()
        assert kernel._attention_budget == 0.75

    def test_replenish_increments_metric(self, kernel):
        kernel._attention_budget = 5.0
        before = kernel._metrics["budget_replenishments"]
        kernel._replenish_attention_budget()
        assert kernel._metrics["budget_replenishments"] == before + 1


class TestKernelStats:

    def test_stats_structure(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        mock_scheduler = MagicMock()
        mock_scheduler.stats.return_value = {"jobs": 0}
        kernel = ProactiveExecutionKernel(
            opportunity_pipeline=mock_pipeline,
            scheduler=mock_scheduler,
        )
        stats = kernel.stats()
        assert stats["running"] is False
        assert stats["cycle_count"] == 0
        assert stats["attention_budget"] == 8.0
        assert "metrics" in stats
        assert "proof_blocks_in_memory" in stats

    def test_initial_metrics_all_zero(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        kernel = ProactiveExecutionKernel(opportunity_pipeline=mock_pipeline)
        for key, value in kernel._metrics.items():
            assert value == 0, f"metric {key} should be 0, got {value}"


class TestKernelStateManagement:

    @pytest.fixture
    def kernel(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        return ProactiveExecutionKernel(
            opportunity_pipeline=mock_pipeline,
            config=ProactiveExecutionKernelConfig(),
        )

    def test_get_persistable_state(self, kernel):
        state = kernel.get_persistable_state()
        assert "running" in state
        assert "cycle_count" in state
        assert "last_tau" in state
        assert "attention_budget" in state
        assert "metrics" in state
        assert "recent_proofs" in state

    def test_restore_persistable_state(self, kernel):
        kernel.restore_persistable_state({
            "cycle_count": 42,
            "last_tau": 0.70,
            "attention_budget": 5.5,
            "metrics": {"cycles": 42, "proposals_generated": 10},
        })
        assert kernel._cycle_count == 42
        assert kernel._last_tau == 0.70
        assert kernel._attention_budget == 5.5
        assert kernel._metrics["cycles"] == 42
        assert kernel._metrics["proposals_generated"] == 10

    def test_restore_clamps_budget(self, kernel):
        kernel.restore_persistable_state({
            "attention_budget": 999.0,
        })
        assert kernel._attention_budget == kernel.config.attention_budget_capacity

    def test_register_sensor(self, kernel):
        kernel.register_sensor("custom", lambda: {"test": True})
        assert "custom" in kernel._sensors

    def test_set_optional_components(self, kernel):
        mock_memory = MagicMock()
        mock_fate = MagicMock()
        mock_snr = MagicMock()
        mock_council = MagicMock()
        mock_ledger = MagicMock()
        kernel.set_living_memory(mock_memory)
        kernel.set_fate_gate(mock_fate)
        kernel.set_snr_optimizer(mock_snr)
        kernel.set_guardian_council(mock_council)
        kernel.set_evidence_ledger(mock_ledger)
        assert kernel._living_memory is mock_memory
        assert kernel._fate_gate is mock_fate
        assert kernel._snr_optimizer is mock_snr
        assert kernel._guardian_council is mock_council
        assert kernel._evidence_ledger is mock_ledger


class TestVerifyWithFate:

    @pytest.fixture
    def kernel(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stats.return_value = {}
        return ProactiveExecutionKernel(opportunity_pipeline=mock_pipeline)

    def test_no_fate_gate_soft_pass(self, kernel):
        proposal = PEKProposal(
            id="test", domain="test", action_type="test",
            description="test", snr_score=0.9, ihsan_score=0.96,
            urgency=0.5, estimated_value=0.7, risk=0.2,
        )
        passed, proof_id, note = kernel._verify_with_fate(proposal, tau=0.55)
        assert passed is True
        assert proof_id is None
        assert "soft-pass" in note

    def test_fate_gate_passes(self, kernel):
        mock_gate = MagicMock()
        mock_proof = MagicMock()
        mock_proof.proof_id = "fate-abc"
        mock_proof.satisfiable = True
        mock_gate.generate_proof.return_value = mock_proof
        kernel.set_fate_gate(mock_gate)

        proposal = PEKProposal(
            id="test", domain="test", action_type="test",
            description="test", snr_score=0.9, ihsan_score=0.96,
            urgency=0.5, estimated_value=0.7, risk=0.2,
        )
        passed, proof_id, note = kernel._verify_with_fate(proposal, tau=0.55)
        assert passed is True
        assert proof_id == "fate-abc"

    def test_fate_gate_rejects(self, kernel):
        mock_gate = MagicMock()
        mock_proof = MagicMock()
        mock_proof.proof_id = "fate-reject"
        mock_proof.satisfiable = False
        mock_gate.generate_proof.return_value = mock_proof
        kernel.set_fate_gate(mock_gate)

        proposal = PEKProposal(
            id="test", domain="test", action_type="test",
            description="test", snr_score=0.9, ihsan_score=0.96,
            urgency=0.5, estimated_value=0.7, risk=0.9,
        )
        passed, proof_id, note = kernel._verify_with_fate(proposal, tau=0.55)
        assert passed is False
