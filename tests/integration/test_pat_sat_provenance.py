"""
Integration test: PAT→IRP→SAT Provenance Boundary

Demonstrates the full flow:
1. PAT receives market data with provenance chains
2. IRP grades the data using Isnad classification
3. PAT produces IrpAssessment (raw data never crosses)
4. SAT Provenance Gate validates the assessments
5. Gate produces GateResult with constitutional verdict

This is the proof that the PAT→SAT security boundary
described in the Declaration actually functions.
"""

import pytest

from core.irp import (
    DataPoint,
    IsnadChain,
    IsnadGrade,
    Source,
    pat_assess,
)
from core.sat.gate_result import CheckStatus
from core.sat.provenance_gate import provenance_verify

# ============================================================================
# FIXTURES: Simulate real-world data sources
# ============================================================================


@pytest.fixture
def trusted_sources():
    """Three independent verified sources — produces SAHIH grade."""
    return [
        Source(id="reuters", name="Reuters", reliability=0.95, verified=True),
        Source(id="bloomberg", name="Bloomberg", reliability=0.93, verified=True),
        Source(id="coinbase", name="Coinbase", reliability=0.90, verified=True),
    ]


@pytest.fixture
def weak_source():
    """Single unverified source — produces DAIF grade."""
    return Source(id="telegram_tip", name="Anon Telegram", reliability=0.3)


@pytest.fixture
def fabricated_source():
    """Unreliable source — produces MAWDU grade (excluded)."""
    return Source(id="anon_paste", name="Anonymous Pastebin", reliability=0.05)


# ============================================================================
# THE FULL PAT→IRP→SAT FLOW
# ============================================================================


class TestPatSatProvenanceBoundary:
    """End-to-end test of the provenance security boundary."""

    def test_sahih_data_passes_gate(self, trusted_sources):
        """SAHIH data (3 independent chains) passes all SAT checks."""
        # Step 1: PAT receives data with provenance
        chains = [IsnadChain(sources=[s]) for s in trusted_sources]
        dp = DataPoint(asset_id="BTC", value=50_000.0, chains=chains)

        # Step 2: PAT grades via IRP (raw data stays local)
        assessment = pat_assess(dp)
        assert assessment.grade == IsnadGrade.SAHIH

        # Step 3: Convert to dict (simulates serialization across boundary)
        assessment_dict = {
            "asset_id": assessment.asset_id,
            "grade": assessment.grade.name,
            "chain_strength": assessment.chain_strength,
            "independent_chain_count": assessment.independent_chain_count,
            "assessment_hash": assessment.assessment_hash,
            "recommended_variance_multiplier": assessment.recommended_variance_multiplier,
        }

        # Step 4: SAT Provenance Gate validates
        result = provenance_verify(assessments=[assessment_dict])
        assert result.passed, f"Gate failed: {[c.evidence for c in result.failed]}"
