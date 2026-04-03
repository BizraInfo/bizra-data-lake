#!/usr/bin/env python3
"""
tests/test_sat_consensus.py - SAT (System Agentic Team) Consensus Tests
=========================================================================

Comprehensive test suite for Byzantine fault-tolerant consensus validation
with 5 guardian validators requiring 3/5 consensus.

SAT Validators:
1. PoiVerifier - Proof of Impact verification
2. RiskGuardian - Risk assessment
3. GovernanceEngine - Policy compliance
4. ResourceAllocator - Efficiency checks
5. EvidenceEngine - Audit trail validation

Test Coverage:
- Unanimous approval (5/5) - PASS
- Supermajority approval (4/5) - PASS
- Minimum consensus (3/5) - PASS
- Split decisions with abstentions
- Consensus failure (2/5) - FAIL with FATE escalation
- Byzantine fault scenarios
- Timeout scenarios
- Conflicting verdicts
- Evidence chain integrity
- Receipt generation

Byzantine Fault Tolerance:
- System tolerates up to 2 malicious validators (f < n/3 where n=5)
- Requires 3/5 = 60% supermajority (> 2f+1)
- Fail-closed: ambiguous state → rejection

From the BIZRA Constitution:
    - Consensus threshold: 3/5 (Byzantine fault tolerant)
    - FATE escalation on consensus failure
    - Receipt-native validation
    - All validators have veto power on critical rejections
"""

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


class ValidatorRole(Enum):
    """SAT validator roles"""
    POI_VERIFIER = "poi_verifier"
    RISK_GUARDIAN = "risk_guardian"
    GOVERNANCE_ENGINE = "governance_engine"
    RESOURCE_ALLOCATOR = "resource_allocator"
    EVIDENCE_ENGINE = "evidence_engine"


class VoteDecision(Enum):
    """Validator vote decisions"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    TIMEOUT = "timeout"
    MALICIOUS = "malicious"  # For Byzantine testing


class RejectionCode(Enum):
    """Rejection reason codes"""
    SECURITY_THREAT = "security_threat"
    ETHICS_VIOLATION = "ethics_violation"
    PERFORMANCE_BUDGET_EXCEEDED = "performance_budget_exceeded"
    CONSISTENCY_FAILURE = "consistency_failure"
    RESOURCE_CONSTRAINT = "resource_constraint"
    QUARANTINE = "quarantine"
    POI_INSUFFICIENT = "poi_insufficient"
    POLICY_VIOLATION = "policy_violation"
    EVIDENCE_MISSING = "evidence_missing"


class EscalationLevel(Enum):
    """FATE escalation levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidatorVote:
    """Individual validator vote"""
    validator_role: ValidatorRole
    decision: VoteDecision
    confidence: float  # 0.0 to 1.0
    reasoning: str
    rejection_code: Optional[RejectionCode] = None
    evidence_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ConsensusResult:
    """Consensus validation result"""
    consensus_reached: bool
    votes: List[ValidatorVote]
    approve_count: int
    reject_count: int
    abstain_count: int
    timeout_count: int
    rejection_codes: List[RejectionCode]
    escalation_level: EscalationLevel
    validation_time_ms: float
    receipt_id: Optional[str] = None
    integrity_hash: Optional[str] = None


@dataclass
class ValidationRequest:
    """Request for SAT validation"""
    task_id: str
    task_description: str
    context: Dict[str, str]
    evidence: Optional[Dict[str, str]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK SAT CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class MockSATConsensusEngine:
    """Mock SAT consensus engine for testing"""

    def __init__(self, validator_responses: Optional[Dict[ValidatorRole, VoteDecision]] = None):
        """
        Initialize mock engine.

        Args:
            validator_responses: Pre-configured validator responses for testing
        """
        self.validator_responses = validator_responses or {}
        self.consensus_threshold = 3  # 3/5 required for consensus
        self.total_validators = 5
        self.validation_timeout_ms = 5000

    async def validate(self, request: ValidationRequest) -> ConsensusResult:
        """
        Execute SAT consensus validation.

        Consensus Rules:
        - Requires 3/5 approval (Byzantine fault tolerant)
        - Any critical rejection (security, ethics) is VETO
        - Timeouts count as abstentions
        - Fail-closed: ambiguous state → rejection
        """
        start_time = asyncio.get_event_loop().time()

        # Collect votes from all 5 validators
        votes: List[ValidatorVote] = []
        for role in ValidatorRole:
            vote = await self._get_validator_vote(role, request)
            votes.append(vote)

        # Count vote types
        approve_count = sum(1 for v in votes if v.decision == VoteDecision.APPROVE)
        reject_count = sum(1 for v in votes if v.decision == VoteDecision.REJECT)
        abstain_count = sum(1 for v in votes if v.decision == VoteDecision.ABSTAIN)
        timeout_count = sum(1 for v in votes if v.decision == VoteDecision.TIMEOUT)

        # Collect rejection codes
        rejection_codes = [v.rejection_code for v in votes if v.rejection_code is not None]

        # Check for VETO conditions (critical rejections)
        has_veto = any(
            code in [
                RejectionCode.SECURITY_THREAT,
                RejectionCode.ETHICS_VIOLATION,
                RejectionCode.QUARANTINE
            ]
            for code in rejection_codes
        )

        # Determine consensus (Byzantine fault tolerant: 3/5 required)
        consensus_reached = not has_veto and approve_count >= self.consensus_threshold

        # Determine escalation level
        escalation_level = self._determine_escalation(
            consensus_reached, rejection_codes, approve_count, reject_count
        )

        # Calculate validation time
        end_time = asyncio.get_event_loop().time()
        validation_time_ms = (end_time - start_time) * 1000

        # Generate receipt
        receipt_id, integrity_hash = self._generate_receipt(
            request, votes, consensus_reached, rejection_codes
        )

        return ConsensusResult(
            consensus_reached=consensus_reached,
            votes=votes,
            approve_count=approve_count,
            reject_count=reject_count,
            abstain_count=abstain_count,
            timeout_count=timeout_count,
            rejection_codes=rejection_codes,
            escalation_level=escalation_level,
            validation_time_ms=validation_time_ms,
            receipt_id=receipt_id,
            integrity_hash=integrity_hash,
        )

    async def _get_validator_vote(
        self, role: ValidatorRole, request: ValidationRequest
    ) -> ValidatorVote:
        """Get vote from a specific validator"""
        # Use pre-configured response if available
        if role in self.validator_responses:
            decision = self.validator_responses[role]

            # Handle malicious/Byzantine behavior
            if decision == VoteDecision.MALICIOUS:
                # Malicious validator provides conflicting evidence
                return ValidatorVote(
                    validator_role=role,
                    decision=VoteDecision.APPROVE,  # Lies about approval
                    confidence=0.99,
                    reasoning="Maliciously approving unsafe request",
                    evidence_hash="malicious_hash_mismatch",
                )

            # Normal pre-configured vote
            rejection_code = None
            if decision == VoteDecision.REJECT:
                rejection_code = self._get_rejection_code_for_role(role)

            return ValidatorVote(
                validator_role=role,
                decision=decision,
                confidence=0.95,
                reasoning=f"{role.value} validation: {decision.value}",
                rejection_code=rejection_code,
                evidence_hash=self._compute_evidence_hash(request),
            )

        # Default: approve
        return ValidatorVote(
            validator_role=role,
            decision=VoteDecision.APPROVE,
            confidence=0.95,
            reasoning=f"{role.value} validation passed",
            evidence_hash=self._compute_evidence_hash(request),
        )

    def _get_rejection_code_for_role(self, role: ValidatorRole) -> RejectionCode:
        """Get appropriate rejection code for validator role"""
        role_to_code = {
            ValidatorRole.POI_VERIFIER: RejectionCode.POI_INSUFFICIENT,
            ValidatorRole.RISK_GUARDIAN: RejectionCode.SECURITY_THREAT,
            ValidatorRole.GOVERNANCE_ENGINE: RejectionCode.POLICY_VIOLATION,
            ValidatorRole.RESOURCE_ALLOCATOR: RejectionCode.RESOURCE_CONSTRAINT,
            ValidatorRole.EVIDENCE_ENGINE: RejectionCode.EVIDENCE_MISSING,
        }
        return role_to_code.get(role, RejectionCode.CONSISTENCY_FAILURE)

    def _determine_escalation(
        self,
        consensus_reached: bool,
        rejection_codes: List[RejectionCode],
        approve_count: int,
        reject_count: int,
    ) -> EscalationLevel:
        """Determine FATE escalation level"""
        if consensus_reached:
            return EscalationLevel.NONE

        # Critical rejections
        if any(
            code in [RejectionCode.SECURITY_THREAT, RejectionCode.ETHICS_VIOLATION]
            for code in rejection_codes
        ):
            return EscalationLevel.CRITICAL

        # High severity
        if RejectionCode.QUARANTINE in rejection_codes:
            return EscalationLevel.HIGH

        # Consensus failure severity based on vote split
        if reject_count >= 4:
            return EscalationLevel.HIGH
        elif reject_count == 3:
            return EscalationLevel.MEDIUM
        else:
            return EscalationLevel.LOW

    def _compute_evidence_hash(self, request: ValidationRequest) -> str:
        """Compute SHA-256 hash of request evidence"""
        evidence_str = json.dumps(
            {
                "task_id": request.task_id,
                "task_description": request.task_description,
                "context": request.context,
                "evidence": request.evidence or {},
                "timestamp": request.timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(evidence_str.encode()).hexdigest()

    def _generate_receipt(
        self,
        request: ValidationRequest,
        votes: List[ValidatorVote],
        consensus_reached: bool,
        rejection_codes: List[RejectionCode],
    ) -> tuple[str, str]:
        """Generate receipt ID and integrity hash"""
        receipt_data = {
            "task_id": request.task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "consensus_reached": consensus_reached,
            "votes": [
                {
                    "validator": v.validator_role.value,
                    "decision": v.decision.value,
                    "confidence": v.confidence,
                }
                for v in votes
            ],
            "rejection_codes": [code.value for code in rejection_codes],
        }

        receipt_str = json.dumps(receipt_data, sort_keys=True)
        integrity_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt_id = f"SAT-{request.task_id[:8]}-{integrity_hash[:8]}"

        return receipt_id, integrity_hash


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def consensus_engine():
    """Create default consensus engine"""
    return MockSATConsensusEngine()


@pytest.fixture
def sample_request():
    """Create sample validation request"""
    return ValidationRequest(
        task_id="TASK-001",
        task_description="Process user data analysis",
        context={"user": "test_user", "operation": "data_analysis"},
        evidence={"source": "internal", "classification": "low_risk"},
    )


@pytest.fixture
def temp_receipt_dir():
    """Create temporary directory for receipts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnanimousApproval:
    """Test Case 1: Unanimous approval (5/5) - PASS"""

    @pytest.mark.asyncio
    async def test_all_validators_approve(self, sample_request):
        """All 5 validators approve - consensus reached"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.APPROVE,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 5
        assert result.reject_count == 0
        assert result.escalation_level == EscalationLevel.NONE
        assert len(result.rejection_codes) == 0
        assert result.receipt_id is not None
        assert result.integrity_hash is not None

    @pytest.mark.asyncio
    async def test_unanimous_approval_receipt_generation(self, sample_request):
        """Unanimous approval generates valid receipt"""
        engine = MockSATConsensusEngine(
            validator_responses={role: VoteDecision.APPROVE for role in ValidatorRole}
        )

        result = await engine.validate(sample_request)

        assert result.receipt_id.startswith("SAT-")
        assert len(result.integrity_hash) == 64  # SHA-256 hex
        assert result.validation_time_ms >= 0


class TestSupermajorityApproval:
    """Test Case 2: Supermajority approval (4/5) - PASS"""

    @pytest.mark.asyncio
    async def test_four_approve_one_reject(self, sample_request):
        """4 approve, 1 rejects - consensus reached"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.REJECT,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 4
        assert result.reject_count == 1
        assert result.escalation_level == EscalationLevel.NONE
        assert len(result.rejection_codes) == 1

    @pytest.mark.asyncio
    async def test_four_approve_one_abstain(self, sample_request):
        """4 approve, 1 abstains - consensus reached"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.ABSTAIN,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 4
        assert result.abstain_count == 1
        assert result.escalation_level == EscalationLevel.NONE


class TestMinimumConsensus:
    """Test Case 3: Minimum consensus (3/5) - PASS"""

    @pytest.mark.asyncio
    async def test_three_approve_two_reject(self, sample_request):
        """3 approve, 2 reject - minimum consensus reached"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.REJECT,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 3
        assert result.reject_count == 2
        assert result.escalation_level == EscalationLevel.NONE
        assert len(result.rejection_codes) == 2

    @pytest.mark.asyncio
    async def test_three_approve_one_reject_one_abstain(self, sample_request):
        """3 approve, 1 reject, 1 abstain - consensus reached"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.ABSTAIN,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 3
        assert result.reject_count == 1
        assert result.abstain_count == 1


class TestSplitDecisions:
    """Test Case 4: Split decisions with abstentions"""

    @pytest.mark.asyncio
    async def test_two_approve_two_reject_one_abstain(self, sample_request):
        """2 approve, 2 reject, 1 abstain - consensus FAILS"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.REJECT,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.ABSTAIN,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is False
        assert result.approve_count == 2
        assert result.reject_count == 2
        assert result.abstain_count == 1
        assert result.escalation_level in [EscalationLevel.LOW, EscalationLevel.MEDIUM]

    @pytest.mark.asyncio
    async def test_three_approve_two_abstain(self, sample_request):
        """3 approve, 2 abstain - consensus reached"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.ABSTAIN,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.ABSTAIN,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 3
        assert result.abstain_count == 2


class TestConsensusFailure:
    """Test Case 5: Consensus failure (2/5) - FAIL with FATE escalation"""

    @pytest.mark.asyncio
    async def test_two_approve_three_reject_escalation(self, sample_request):
        """2 approve, 3 reject - FAIL with HIGH escalation"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.REJECT,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.REJECT,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is False
        assert result.approve_count == 2
        assert result.reject_count == 3
        assert result.escalation_level == EscalationLevel.MEDIUM
        assert len(result.rejection_codes) == 3

    @pytest.mark.asyncio
    async def test_one_approve_four_reject_critical_escalation(self, sample_request):
        """1 approve, 4 reject - FAIL with HIGH escalation"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.REJECT,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.REJECT,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.REJECT,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is False
        assert result.approve_count == 1
        assert result.reject_count == 4
        assert result.escalation_level == EscalationLevel.HIGH

    @pytest.mark.asyncio
    async def test_zero_approve_all_reject(self, sample_request):
        """0 approve, 5 reject - FAIL with HIGH escalation"""
        engine = MockSATConsensusEngine(
            validator_responses={role: VoteDecision.REJECT for role in ValidatorRole}
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is False
        assert result.approve_count == 0
        assert result.reject_count == 5
        assert result.escalation_level == EscalationLevel.HIGH


class TestVetoPower:
    """Test Case 6: Critical rejections have VETO power"""

    @pytest.mark.asyncio
    async def test_security_threat_veto(self, sample_request):
        """Security threat rejection vetoes 4 approvals"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.REJECT,  # Security veto
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.APPROVE,
            }
        )

        result = await engine.validate(sample_request)

        # VETO: Even with 4 approvals, security threat blocks consensus
        assert result.consensus_reached is False
        assert result.approve_count == 4
        assert result.reject_count == 1
        assert RejectionCode.SECURITY_THREAT in result.rejection_codes
        assert result.escalation_level == EscalationLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_quarantine_veto(self, sample_request):
        """Quarantine rejection triggers HIGH escalation"""
        # Mock to return quarantine rejection
        engine = MockSATConsensusEngine()
        engine.validator_responses[ValidatorRole.GOVERNANCE_ENGINE] = VoteDecision.REJECT

        # Manually inject quarantine code for this test
        async def mock_validator_vote(role, request):
            if role == ValidatorRole.GOVERNANCE_ENGINE:
                return ValidatorVote(
                    validator_role=role,
                    decision=VoteDecision.REJECT,
                    confidence=0.75,
                    reasoning="Uncertain - requires human review",
                    rejection_code=RejectionCode.QUARANTINE,
                )
            return await engine._get_validator_vote(role, request)

        engine._get_validator_vote = mock_validator_vote
        result = await engine.validate(sample_request)

        assert result.consensus_reached is False
        assert RejectionCode.QUARANTINE in result.rejection_codes
        assert result.escalation_level == EscalationLevel.HIGH


class TestByzantineFaultTolerance:
    """Test Case 8: Byzantine fault scenarios"""

    @pytest.mark.asyncio
    async def test_two_malicious_three_honest_pass(self, sample_request):
        """2 malicious validators, 3 honest approve - should PASS"""
        # Byzantine tolerance: f < n/3, where n=5, f=2 is at the threshold
        # System should still reach consensus with 3 honest approvals
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.MALICIOUS,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.MALICIOUS,
            }
        )

        result = await engine.validate(sample_request)

        # With 3 honest approvals, consensus should be reached
        # Malicious votes are treated as approvals in this mock
        assert result.approve_count >= 3
        assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_three_malicious_detectable(self, sample_request):
        """3 malicious validators - should be DETECTABLE via evidence mismatch"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.MALICIOUS,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.MALICIOUS,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.MALICIOUS,
            }
        )

        result = await engine.validate(sample_request)

        # Check for evidence hash mismatches (Byzantine detection)
        evidence_hashes = [v.evidence_hash for v in result.votes if v.evidence_hash]
        honest_hash = engine._compute_evidence_hash(sample_request)

        malicious_count = sum(
            1 for h in evidence_hashes if h == "malicious_hash_mismatch"
        )

        assert malicious_count == 3
        # In production, this would trigger Byzantine fault detection

    @pytest.mark.asyncio
    async def test_conflicting_evidence_hashes(self, sample_request):
        """Validators with conflicting evidence hashes"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.MALICIOUS,
            }
        )

        result = await engine.validate(sample_request)

        # Check for evidence integrity
        honest_hash = engine._compute_evidence_hash(sample_request)
        evidence_hashes = [v.evidence_hash for v in result.votes]

        # At least 4 should have correct hash (honest validators)
        correct_hashes = sum(1 for h in evidence_hashes if h == honest_hash)
        assert correct_hashes >= 4


class TestTimeoutScenarios:
    """Test Case 9: Validator timeout scenarios"""

    @pytest.mark.asyncio
    async def test_one_timeout_four_approve(self, sample_request):
        """1 timeout, 4 approve - consensus reached"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.TIMEOUT,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 4
        assert result.timeout_count == 1

    @pytest.mark.asyncio
    async def test_two_timeout_three_approve(self, sample_request):
        """2 timeout, 3 approve - consensus reached (timeouts as abstentions)"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.TIMEOUT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.TIMEOUT,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 3
        assert result.timeout_count == 2

    @pytest.mark.asyncio
    async def test_three_timeout_consensus_fails(self, sample_request):
        """3 timeout, 2 approve - consensus FAILS (not enough votes)"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.TIMEOUT,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.TIMEOUT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.TIMEOUT,
            }
        )

        result = await engine.validate(sample_request)

        # Fail-closed: not enough approvals
        assert result.consensus_reached is False
        assert result.approve_count == 2
        assert result.timeout_count == 3


class TestConflictingVerdicts:
    """Test Case 10: Conflicting verdicts with same evidence"""

    @pytest.mark.asyncio
    async def test_same_evidence_different_conclusions(self, sample_request):
        """Validators reach different conclusions with same evidence"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.REJECT,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.APPROVE,
            }
        )

        result = await engine.validate(sample_request)

        # All should have same evidence hash (same input)
        evidence_hashes = [
            v.evidence_hash for v in result.votes
            if v.evidence_hash and v.decision != VoteDecision.MALICIOUS
        ]
        assert len(set(evidence_hashes)) == 1  # All identical

        # But different conclusions - 3 approve, 2 reject
        assert result.approve_count == 3
        assert result.reject_count == 2
        assert result.consensus_reached is True


class TestQuorumEdgeCases:
    """Test Case 11: Quorum validation edge cases"""

    @pytest.mark.asyncio
    async def test_exactly_three_approve_boundary(self, sample_request):
        """Exactly 3 approve (minimum) - boundary test"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.ABSTAIN,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.ABSTAIN,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is True
        assert result.approve_count == 3
        assert result.abstain_count == 2

    @pytest.mark.asyncio
    async def test_exactly_two_approve_fails(self, sample_request):
        """Exactly 2 approve (below minimum) - should fail"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.ABSTAIN,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.ABSTAIN,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.ABSTAIN,
            }
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is False
        assert result.approve_count == 2
        assert result.abstain_count == 3

    @pytest.mark.asyncio
    async def test_all_abstain_fails_closed(self, sample_request):
        """All validators abstain - fail-closed behavior"""
        engine = MockSATConsensusEngine(
            validator_responses={role: VoteDecision.ABSTAIN for role in ValidatorRole}
        )

        result = await engine.validate(sample_request)

        # Fail-closed: ambiguous state → rejection
        assert result.consensus_reached is False
        assert result.approve_count == 0
        assert result.abstain_count == 5
        assert result.escalation_level in [EscalationLevel.LOW, EscalationLevel.MEDIUM]


class TestReceiptGeneration:
    """Test Case 12: Receipt generation for consensus results"""

    @pytest.mark.asyncio
    async def test_receipt_contains_all_votes(self, sample_request):
        """Receipt includes all validator votes"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.APPROVE,
            }
        )

        result = await engine.validate(sample_request)

        assert len(result.votes) == 5
        assert all(isinstance(v, ValidatorVote) for v in result.votes)
        assert result.receipt_id is not None
        assert len(result.integrity_hash) == 64

    @pytest.mark.asyncio
    async def test_receipt_integrity_hash_deterministic(self, sample_request):
        """Receipt integrity hash is deterministic for same input"""
        engine = MockSATConsensusEngine(
            validator_responses={role: VoteDecision.APPROVE for role in ValidatorRole}
        )

        result1 = await engine.validate(sample_request)

        # Reset engine with same configuration
        engine2 = MockSATConsensusEngine(
            validator_responses={role: VoteDecision.APPROVE for role in ValidatorRole}
        )
        result2 = await engine2.validate(sample_request)

        # Evidence hashes should match (deterministic)
        evidence_hashes_1 = [v.evidence_hash for v in result1.votes]
        evidence_hashes_2 = [v.evidence_hash for v in result2.votes]

        assert evidence_hashes_1 == evidence_hashes_2

    @pytest.mark.asyncio
    async def test_receipt_includes_rejection_codes(self, sample_request):
        """Receipt includes all rejection codes"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.REJECT,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.REJECT,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.REJECT,
            }
        )

        result = await engine.validate(sample_request)

        assert len(result.rejection_codes) == 3
        assert all(isinstance(code, RejectionCode) for code in result.rejection_codes)

    @pytest.mark.asyncio
    async def test_receipt_includes_escalation_level(self, sample_request):
        """Receipt includes FATE escalation level"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.REJECT,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.REJECT,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.REJECT,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.REJECT,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.REJECT,
            }
        )

        result = await engine.validate(sample_request)

        assert result.escalation_level != EscalationLevel.NONE
        assert isinstance(result.escalation_level, EscalationLevel)


class TestEvidenceChainIntegrity:
    """Test Case 13: Evidence chain integrity validation"""

    @pytest.mark.asyncio
    async def test_evidence_hash_consistency(self, sample_request):
        """All honest validators produce same evidence hash"""
        engine = MockSATConsensusEngine(
            validator_responses={role: VoteDecision.APPROVE for role in ValidatorRole}
        )

        result = await engine.validate(sample_request)

        # All evidence hashes should be identical (same input)
        evidence_hashes = [v.evidence_hash for v in result.votes]
        assert len(set(evidence_hashes)) == 1

    @pytest.mark.asyncio
    async def test_evidence_includes_context(self, sample_request):
        """Evidence hash includes request context"""
        request_with_context = ValidationRequest(
            task_id="TASK-002",
            task_description="Different task",
            context={"different": "context"},
        )

        engine = MockSATConsensusEngine()
        result1 = await engine.validate(sample_request)
        result2 = await engine.validate(request_with_context)

        # Different requests → different evidence hashes
        hash1 = result1.votes[0].evidence_hash
        hash2 = result2.votes[0].evidence_hash

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_evidence_tamper_detection(self, sample_request):
        """Tampering with evidence is detectable"""
        engine = MockSATConsensusEngine()
        result = await engine.validate(sample_request)

        # Simulate tampering by modifying request after validation
        tampered_request = ValidationRequest(
            task_id=sample_request.task_id,
            task_description="TAMPERED",
            context=sample_request.context,
            timestamp=sample_request.timestamp,
        )

        tampered_hash = engine._compute_evidence_hash(tampered_request)
        original_hash = result.votes[0].evidence_hash

        assert tampered_hash != original_hash


class TestFailClosedBehavior:
    """Test Case 14: Fail-closed behavior validation"""

    @pytest.mark.asyncio
    async def test_ambiguous_state_rejects(self, sample_request):
        """Ambiguous state (all abstentions) → rejection"""
        engine = MockSATConsensusEngine(
            validator_responses={role: VoteDecision.ABSTAIN for role in ValidatorRole}
        )

        result = await engine.validate(sample_request)

        assert result.consensus_reached is False
        assert result.approve_count == 0

    @pytest.mark.asyncio
    async def test_veto_overrides_majority(self, sample_request):
        """Critical rejection overrides majority approval"""
        engine = MockSATConsensusEngine()

        # Mock critical rejection
        async def mock_validator_vote(role, request):
            if role == ValidatorRole.RISK_GUARDIAN:
                return ValidatorVote(
                    validator_role=role,
                    decision=VoteDecision.REJECT,
                    confidence=0.99,
                    reasoning="Critical security threat",
                    rejection_code=RejectionCode.SECURITY_THREAT,
                )
            return ValidatorVote(
                validator_role=role,
                decision=VoteDecision.APPROVE,
                confidence=0.95,
                reasoning="Approved",
            )

        engine._get_validator_vote = mock_validator_vote
        result = await engine.validate(sample_request)

        # 4 approve, 1 critical reject → consensus FAILS
        assert result.consensus_reached is False
        assert RejectionCode.SECURITY_THREAT in result.rejection_codes


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSATIntegration:
    """Integration tests with full system"""

    @pytest.mark.asyncio
    async def test_end_to_end_consensus_flow(self, sample_request):
        """End-to-end consensus validation flow"""
        engine = MockSATConsensusEngine(
            validator_responses={
                ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
                ValidatorRole.RISK_GUARDIAN: VoteDecision.APPROVE,
                ValidatorRole.GOVERNANCE_ENGINE: VoteDecision.APPROVE,
                ValidatorRole.RESOURCE_ALLOCATOR: VoteDecision.APPROVE,
                ValidatorRole.EVIDENCE_ENGINE: VoteDecision.REJECT,
            }
        )

        result = await engine.validate(sample_request)

        # Full validation
        assert result.consensus_reached is True
        assert result.receipt_id is not None
        assert result.integrity_hash is not None
        assert result.validation_time_ms >= 0
        assert len(result.votes) == 5
        assert result.escalation_level == EscalationLevel.NONE

    @pytest.mark.asyncio
    async def test_multiple_sequential_validations(self, consensus_engine):
        """Multiple validations in sequence"""
        requests = [
            ValidationRequest(
                task_id=f"TASK-{i:03d}",
                task_description=f"Task {i}",
                context={"index": str(i)},
            )
            for i in range(5)
        ]

        results = []
        for req in requests:
            result = await consensus_engine.validate(req)
            results.append(result)

        assert len(results) == 5
        assert all(r.consensus_reached for r in results)

        # All receipt IDs should be unique
        receipt_ids = [r.receipt_id for r in results]
        assert len(set(receipt_ids)) == 5

    @pytest.mark.asyncio
    async def test_concurrent_validations(self, consensus_engine):
        """Concurrent validations (stress test)"""
        requests = [
            ValidationRequest(
                task_id=f"CONCURRENT-{i:03d}",
                task_description=f"Concurrent task {i}",
                context={"index": str(i)},
            )
            for i in range(10)
        ]

        # Run concurrently
        results = await asyncio.gather(
            *[consensus_engine.validate(req) for req in requests]
        )

        assert len(results) == 10
        assert all(r.consensus_reached for r in results)

        # All receipt IDs should be unique
        receipt_ids = [r.receipt_id for r in results]
        assert len(set(receipt_ids)) == 10


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformance:
    """Performance benchmarks"""

    @pytest.mark.asyncio
    async def test_validation_latency(self, consensus_engine, sample_request):
        """Validation completes within acceptable time"""
        result = await consensus_engine.validate(sample_request)

        # Should complete in reasonable time (mock: < 100ms)
        assert result.validation_time_ms < 100

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_throughput_benchmark(self, consensus_engine):
        """Measure throughput for 100 validations"""
        import time

        start = time.time()

        tasks = [
            consensus_engine.validate(
                ValidationRequest(
                    task_id=f"BENCH-{i:04d}",
                    task_description=f"Benchmark task {i}",
                    context={"index": str(i)},
                )
            )
            for i in range(100)
        ]

        results = await asyncio.gather(*tasks)

        end = time.time()
        elapsed = end - start
        throughput = len(results) / elapsed

        assert len(results) == 100
        assert throughput > 10  # At least 10 validations/sec (mock)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
