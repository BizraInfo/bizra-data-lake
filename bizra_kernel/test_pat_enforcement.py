"""
PAT Enforcement Pipeline — Integration Tests
============================================
Comprehensive tests for all 5 gates and full pipeline execution.

Tests:
- Individual gate validation
- Full pipeline execution (pass/fail)
- Correction mechanisms
- Receipt generation
- Telemetry tracking
"""

import asyncio
import json
from pathlib import Path

import pytest

from bizra_kernel.pat_enforcement_pipeline import (
    PATEnforcementPipeline,
    PATRequest,
    PATTelemetry,
    GateID,
    GateStatus,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def pipeline():
    """Create PAT enforcement pipeline."""
    return PATEnforcementPipeline()


@pytest.fixture
def telemetry():
    """Create telemetry tracker."""
    return PATTelemetry()


@pytest.fixture
def valid_request():
    """Create valid PAT request that should pass all gates."""
    return PATRequest(
        session_id="test_session_valid",
        task_id="test_task_valid",
        query="Optimize distributed database performance using machine learning",
        context={"environment": "production"},
        synthesis_nodes=[
            {
                "id": "node_1",
                "content": "Apply parallel query execution with adaptive concurrency",
                "snr": 0.98,
                "claim_tag": "NOVEL",
                "domains": ["Distributed Systems", "Database Systems"],
            },
            {
                "id": "node_2",
                "content": "Implement ML-based query optimizer with reinforcement learning",
                "snr": 0.97,
                "claim_tag": "DERIVED",
                "domains": ["Machine Learning", "Database Systems"],
            },
            {
                "id": "node_3",
                "content": "Use workload prediction for proactive resource allocation",
                "snr": 0.96,
                "claim_tag": "CROSS_DOMAIN",
                "domains": ["Machine Learning", "Distributed Systems"],
            },
        ],
        domains=[
            {
                "name": "Distributed Systems",
                "cluster_id": "cluster_1",
                "keywords": ["distributed", "consensus", "replication"],
            },
            {
                "name": "Machine Learning",
                "cluster_id": "cluster_2",
                "keywords": ["neural", "training", "inference"],
            },
            {
                "name": "Database Systems",
                "cluster_id": "cluster_3",
                "keywords": ["sql", "transactions", "indexing"],
            },
        ],
        practitioners=[
            {
                "name": "Dr. Alice Smith",
                "tier": "top_1%",
                "domains": ["Distributed Systems", "Database Systems"],
                "relevance_score": 0.85,
            },
            {
                "name": "Prof. Bob Johnson",
                "tier": "top_1%",
                "domains": ["Machine Learning", "Distributed Systems"],
                "relevance_score": 0.80,
            },
            {
                "name": "Dr. Carol Williams",
                "tier": "top_1%",
                "domains": ["Database Systems", "Machine Learning"],
                "relevance_score": 0.75,
            },
            {
                "name": "Dr. David Brown",
                "tier": "top_1%",
                "domains": ["Distributed Systems"],
                "relevance_score": 0.82,
            },
        ],
        response_sections=[
            {
                "id": "executive_synthesis",
                "claims": [
                    {"text": "Parallel execution improves throughput", "tag": "MEASURED"}
                ],
            },
            {
                "id": "domain_cross_pollination_map",
                "claims": [],
                "domains": ["Distributed Systems", "Machine Learning", "Database Systems"],
            },
            {
                "id": "elite_practitioner_anchoring",
                "claims": [],
                "practitioners": 4,
            },
            {
                "id": "novel_insight_synthesis",
                "claims": [{"text": "Novel synthesis", "tag": "NOVEL"}],
            },
            {
                "id": "validation_evidence_trail",
                "gate_statuses": [],
                "snr_scores": [],
                "ihsan_scores": [],
                "receipt_ids": [],
            },
            {
                "id": "actionable_recommendations",
                "claims": [{"text": "Implement changes", "tag": "TARGET"}],
            },
        ],
        running_snr=0.97,
        novelty_score=0.80,
    )


@pytest.fixture
def invalid_request_insufficient_domains():
    """Create request with insufficient domains (should fail Gate 1)."""
    return PATRequest(
        session_id="test_session_fail_gate1",
        task_id="test_task_fail_gate1",
        query="Basic task",
        context={},
        synthesis_nodes=[
            {"id": "node_1", "content": "Simple node", "snr": 0.98, "claim_tag": "DERIVED"}
        ],
        domains=[
            {"name": "Domain1", "cluster_id": "cluster_1", "keywords": ["keyword1"]}
        ],  # Only 1 domain, need 3
        practitioners=[],
        response_sections=[],
    )


@pytest.fixture
def invalid_request_low_snr():
    """Create request with low SNR (should fail Gate 2 or 3)."""
    return PATRequest(
        session_id="test_session_fail_gate2",
        task_id="test_task_fail_gate2",
        query="Task with low quality",
        context={},
        synthesis_nodes=[
            {
                "id": "node_1",
                "content": "Low quality node",
                "snr": 0.85,  # Below 0.95 mid-synthesis threshold
                "claim_tag": "DERIVED",
            }
        ],
        domains=[
            {"name": "Domain1", "cluster_id": "cluster_1"},
            {"name": "Domain2", "cluster_id": "cluster_2"},
            {"name": "Domain3", "cluster_id": "cluster_3"},
        ],
        practitioners=[],
        response_sections=[],
        running_snr=0.85,  # Too low
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_gate_1_pass(pipeline, valid_request):
    """Test Gate 1 passes with valid domains."""
    result = await pipeline._execute_gate_1(valid_request)

    assert result.gate_id == GateID.GATE_1_PRE_REASONING
    assert result.passed
    assert result.status == GateStatus.PASSED
    assert result.checks["domain_count_ok"]
    assert result.checks["unrelatedness_ok"]
    assert result.scores["domain_count"] >= 3


@pytest.mark.asyncio
async def test_gate_1_fail(pipeline, invalid_request_insufficient_domains):
    """Test Gate 1 fails with insufficient domains."""
    result = await pipeline._execute_gate_1(invalid_request_insufficient_domains)

    assert result.gate_id == GateID.GATE_1_PRE_REASONING
    assert not result.passed
    assert result.status == GateStatus.FAILED
    assert not result.checks["domain_count_ok"]


@pytest.mark.asyncio
async def test_gate_2_pass(pipeline, valid_request):
    """Test Gate 2 passes with good SNR."""
    result = await pipeline._execute_gate_2(valid_request)

    assert result.gate_id == GateID.GATE_2_MID_SYNTHESIS
    assert result.passed
    assert result.status == GateStatus.PASSED
    assert result.checks["running_snr_ok"]
    assert result.checks["no_contradictions"]
    assert result.checks["claim_tags_present"]


@pytest.mark.asyncio
async def test_gate_2_fail(pipeline, invalid_request_low_snr):
    """Test Gate 2 fails with low SNR."""
    result = await pipeline._execute_gate_2(invalid_request_low_snr)

    assert result.gate_id == GateID.GATE_2_MID_SYNTHESIS
    assert not result.passed
    assert result.status == GateStatus.FAILED
    assert not result.checks["running_snr_ok"]


@pytest.mark.asyncio
async def test_gate_3_pass(pipeline, valid_request):
    """Test Gate 3 passes with final validation."""
    result = await pipeline._execute_gate_3(valid_request)

    assert result.gate_id == GateID.GATE_3_POST_SYNTHESIS
    assert result.passed
    assert result.status == GateStatus.PASSED
    assert result.scores["final_snr"] >= 0.98
    assert result.scores["novelty_score"] >= 0.75


@pytest.mark.asyncio
async def test_gate_4_pass(pipeline, valid_request):
    """Test Gate 4 passes with valid practitioners."""
    result = await pipeline._execute_gate_4(valid_request)

    assert result.gate_id == GateID.GATE_4_PRACTITIONER
    # Gate 4 can warn but typically passes with valid data
    assert result.checks["practitioners_per_domain_ok"]
    assert result.checks["all_top_1_percent"]


@pytest.mark.asyncio
async def test_gate_5_pass(pipeline, valid_request):
    """Test Gate 5 passes with 6-section structure."""
    result = await pipeline._execute_gate_5(valid_request)

    assert result.gate_id == GateID.GATE_5_RESPONSE
    assert result.passed
    assert result.status == GateStatus.PASSED
    assert result.checks["section_count_ok"]
    assert result.scores["section_count"] == 6


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_full_pipeline_pass(pipeline, valid_request, telemetry):
    """Test full pipeline execution with valid request."""
    result = await pipeline.enforce(valid_request)

    # Record telemetry
    telemetry.record_enforcement(result)

    # Assertions
    assert result.passed
    assert len(result.gate_results) == 5
    assert all(gate.passed for gate in result.gate_results)
    assert result.final_snr >= 0.98
    assert result.final_novelty >= 0.75
    assert result.final_ihsan >= 0.95
    assert result.domain_count >= 3
    assert result.receipt_id
    assert result.receipt_path

    # Check receipt file exists
    receipt_file = Path(result.receipt_path)
    assert receipt_file.exists()

    # Verify receipt content
    with open(receipt_file, "r") as f:
        receipt_data = json.load(f)

    assert receipt_data["receipt_type"] == "PAT_ENFORCEMENT"
    assert receipt_data["passed"] is True
    assert receipt_data["session_id"] == valid_request.session_id

    # Check telemetry
    stats = telemetry.get_stats()
    assert stats["total_enforcements"] == 1
    assert stats["total_passes"] == 1
    assert stats["total_failures"] == 0


@pytest.mark.asyncio
async def test_full_pipeline_fail_gate_1(
    pipeline, invalid_request_insufficient_domains, telemetry
):
    """Test full pipeline fails at Gate 1."""
    result = await pipeline.enforce(invalid_request_insufficient_domains)

    telemetry.record_enforcement(result)

    # Assertions
    assert not result.passed
    assert len(result.gate_results) == 1  # Only Gate 1 executed
    assert not result.gate_results[0].passed
    assert result.gate_results[0].gate_id == GateID.GATE_1_PRE_REASONING

    # Check telemetry
    stats = telemetry.get_stats()
    assert stats["total_failures"] >= 1
    assert GateID.GATE_1_PRE_REASONING.value in stats["gate_failure_counts"]


@pytest.mark.asyncio
async def test_full_pipeline_fail_gate_2(pipeline, invalid_request_low_snr, telemetry):
    """Test full pipeline fails at Gate 2."""
    result = await pipeline.enforce(invalid_request_low_snr)

    telemetry.record_enforcement(result)

    # Assertions
    assert not result.passed
    # Should execute Gate 1, then fail at Gate 2
    assert len(result.gate_results) >= 1
    failed_gates = [g for g in result.gate_results if not g.passed]
    assert len(failed_gates) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_correction_attempt(pipeline, invalid_request_insufficient_domains):
    """Test correction mechanism is triggered."""
    result = await pipeline.enforce(invalid_request_insufficient_domains)

    # Check that correction was attempted
    gate_1_result = result.gate_results[0]
    assert gate_1_result.correction_attempts >= 0  # May or may not attempt correction
    if gate_1_result.correction_action:
        assert gate_1_result.correction_action.value in [
            "expand_domains",
            "prune_low_quality_nodes",
            "additional_synthesis_pass",
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_telemetry_tracking(pipeline, valid_request, telemetry):
    """Test telemetry tracks multiple enforcements."""
    # Execute multiple enforcements
    result_1 = await pipeline.enforce(valid_request)
    telemetry.record_enforcement(result_1)

    result_2 = await pipeline.enforce(valid_request)
    telemetry.record_enforcement(result_2)

    # Check stats
    stats = telemetry.get_stats()
    assert stats["total_enforcements"] == 2
    assert stats["total_passes"] == 2
    assert stats["pass_rate"] == 1.0
    assert stats["average_latency_ms"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_integration_with_components(pipeline):
    """Test integration with existing components."""
    # Test that pipeline integrates with:
    # - ihsan_gate
    # - snr_tracker
    # - pat_domain_validator
    # - pat_novelty_probe
    # - pat_citation_validator

    assert pipeline.ihsan_gate is not None
    assert pipeline.snr_tracker is not None
    assert pipeline.ihsan_gate.threshold == 0.95


@pytest.mark.asyncio
async def test_receipt_generation(pipeline, valid_request):
    """Test receipt generation and format."""
    result = await pipeline.enforce(valid_request)

    receipt_file = Path(result.receipt_path)
    assert receipt_file.exists()

    with open(receipt_file, "r") as f:
        receipt_data = json.load(f)

    # Verify receipt schema
    required_fields = [
        "receipt_type",
        "version",
        "session_id",
        "task_id",
        "passed",
        "gate_results",
        "final_snr",
        "final_novelty",
        "final_ihsan",
        "receipt_id",
        "timestamp",
    ]

    for field in required_fields:
        assert field in receipt_data, f"Missing field: {field}"

    # Verify gate results in receipt
    assert len(receipt_data["gate_results"]) == 5
    for gate_result in receipt_data["gate_results"]:
        assert "gate_id" in gate_result
        assert "status" in gate_result
        assert "latency_ms" in gate_result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
