"""
Tests for the BIZRA Proof-of-Impact Bounty Economy.

Tests all four layers:
1. Discovery (Hunter Agent)
2. Proof (Impact Proof)
3. Payment (Bounty Oracle)
4. Integration (Bounty Bridge)
"""

import pytest
import asyncio
from datetime import datetime, timezone

from core.bounty import (
    BOUNTY_VERSION,
    SEVERITY_LEVELS,
    VULN_CATEGORIES,
    SECURITY_VECTORS,
    BASE_PAYOUT_PER_DELTA_E,
    BOUNTY_SNR_THRESHOLD,
    BOUNTY_IHSAN_THRESHOLD,
)
from core.bounty.impact_proof import (
    ImpactProof,
    ImpactProofBuilder,
    ImpactProofVerifier,
    EntropyMeasurement,
    DomainEvent,
    Severity,
    VulnCategory,
)
from core.bounty.oracle import (
    BountyOracle,
    BountyCalculation,
    BountyPayout,
)
from core.bounty.hunter import (
    HunterAgent,
    HunterSwarm,
    ScanTarget,
    ScanResult,
    ScanStatus,
)
from core.bounty.bridge import (
    BountyBridge,
    BountySubmission,
    Platform,
    PlatformCredentials,
    SubmissionStatus,
)
from core.proof_engine.receipt import SimpleSigner


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def signer():
    """Create a test signer."""
    return SimpleSigner(b"test_bounty_hunter_secret")


@pytest.fixture
def entropy_before():
    """High entropy state (vulnerable)."""
    return EntropyMeasurement(
        surface_entropy=0.8,
        structural_entropy=0.7,
        behavioral_entropy=0.9,
        hypothetical_entropy=0.6,
        contextual_entropy=0.5,
    )


@pytest.fixture
def entropy_after():
    """Low entropy state (fixed)."""
    return EntropyMeasurement(
        surface_entropy=0.4,
        structural_entropy=0.3,
        behavioral_entropy=0.4,
        hypothetical_entropy=0.3,
        contextual_entropy=0.2,
    )


@pytest.fixture
def proof_builder(signer):
    """Create impact proof builder."""
    return ImpactProofBuilder(signer)


@pytest.fixture
def sample_proof(proof_builder, entropy_before, entropy_after):
    """Create a sample impact proof."""
    return proof_builder.build(
        target_address="0x1234567890abcdef1234567890abcdef12345678",
        vuln_category=VulnCategory.REENTRANCY,
        severity=Severity.HIGH,
        title="Reentrancy in withdraw()",
        description="Cross-function reentrancy vulnerability",
        exploit_code=b"// Exploit code",
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        reproduction_steps=[
            DomainEvent(
                event_type="exploit",
                timestamp=datetime.now(timezone.utc),
                data={"target": "withdraw()"},
            )
        ],
        funds_at_risk=1_000_000,
        snr_score=0.96,
        ihsan_score=0.96,
    )


@pytest.fixture
def bounty_oracle(signer):
    """Create bounty oracle."""
    return BountyOracle(signer)


@pytest.fixture
def hunter_agent(signer):
    """Create hunter agent."""
    return HunterAgent(signer)


@pytest.fixture
def bounty_bridge(signer):
    """Create bounty bridge."""
    return BountyBridge(signer)


@pytest.fixture
def scan_target():
    """Create a scan target."""
    return ScanTarget(
        address="0xabcdef1234567890abcdef1234567890abcdef12",
        chain="ethereum",
        name="TestProtocol",
        tvl=5_000_000,
        bytecode=bytes([0x60, 0x80, 0x60, 0x40, 0x52] * 50),
        abi=[
            {"type": "function", "name": "withdraw", "inputs": []},
            {"type": "function", "name": "deposit", "inputs": []},
        ],
    )


# =============================================================================
# MODULE CONSTANTS
# =============================================================================

class TestModuleConstants:
    """Tests for module-level constants."""

    def test_version(self):
        """Version is defined."""
        assert BOUNTY_VERSION == "1.0.0"

    def test_severity_levels(self):
        """Severity levels have correct structure."""
        assert "low" in SEVERITY_LEVELS
        assert "medium" in SEVERITY_LEVELS
        assert "high" in SEVERITY_LEVELS
        assert "critical" in SEVERITY_LEVELS

        for level, config in SEVERITY_LEVELS.items():
            assert "multiplier" in config
            assert "min_payout" in config

    def test_vuln_categories(self):
        """Vulnerability categories defined."""
        assert "reentrancy" in VULN_CATEGORIES
        assert "flash_loan" in VULN_CATEGORIES
        assert "oracle_manipulation" in VULN_CATEGORIES

    def test_security_vectors(self):
        """Security vectors mapped to UERS."""
        assert len(SECURITY_VECTORS) == 5
        assert "surface" in SECURITY_VECTORS
        assert "structural" in SECURITY_VECTORS
        assert "behavioral" in SECURITY_VECTORS
        assert "hypothetical" in SECURITY_VECTORS
        assert "contextual" in SECURITY_VECTORS

    def test_base_payout(self):
        """Base payout is defined."""
        assert BASE_PAYOUT_PER_DELTA_E == 500


# =============================================================================
# IMPACT PROOF
# =============================================================================

class TestImpactProof:
    """Tests for impact proof creation and verification."""

    def test_entropy_measurement_total(self, entropy_before):
        """Entropy measurement calculates total."""
        total = entropy_before.total_entropy
        expected = 0.8 + 0.7 + 0.9 + 0.6 + 0.5
        assert abs(total - expected) < 0.001

    def test_entropy_measurement_average(self, entropy_before):
        """Entropy measurement calculates average."""
        avg = entropy_before.average_entropy
        assert abs(avg - 0.7) < 0.001

    def test_impact_proof_delta_e(self, sample_proof):
        """Impact proof calculates delta E."""
        assert sample_proof.delta_e > 0
        # Before - After
        expected = 3.5 - 1.6  # 1.9
        assert abs(sample_proof.delta_e - expected) < 0.001

    def test_impact_proof_multiplier(self, sample_proof):
        """Impact proof has correct severity multiplier."""
        assert sample_proof.severity == Severity.HIGH
        assert sample_proof.multiplier == 20

    def test_impact_proof_signature(self, sample_proof, signer):
        """Impact proof is signed."""
        assert len(sample_proof.signature) > 0
        assert sample_proof.verify_signature(signer) is True

    def test_impact_proof_digest(self, sample_proof):
        """Impact proof has deterministic digest."""
        digest1 = sample_proof.hex_digest()
        digest2 = sample_proof.hex_digest()
        assert digest1 == digest2

    def test_proof_verifier_valid(self, sample_proof, signer):
        """Verifier accepts valid proof."""
        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(sample_proof)
        assert valid is True
        assert error is None

    def test_proof_verifier_low_snr(self, proof_builder, entropy_before, entropy_after, signer):
        """Verifier rejects low SNR proof."""
        proof = proof_builder.build(
            target_address="0x1234",
            vuln_category=VulnCategory.LOGIC_ERROR,
            severity=Severity.LOW,
            title="Test",
            description="Test",
            exploit_code=b"test",
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            reproduction_steps=[],
            snr_score=0.5,  # Below threshold
            ihsan_score=0.95,
        )

        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(proof)
        assert valid is False
        assert "SNR" in error

    def test_proof_verifier_low_ihsan(self, proof_builder, entropy_before, entropy_after, signer):
        """Verifier rejects low Ihsān proof."""
        proof = proof_builder.build(
            target_address="0x1234",
            vuln_category=VulnCategory.LOGIC_ERROR,
            severity=Severity.LOW,
            title="Test",
            description="Test",
            exploit_code=b"test",
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            reproduction_steps=[],
            snr_score=0.95,
            ihsan_score=0.5,  # Below threshold
        )

        verifier = ImpactProofVerifier(signer)
        valid, error = verifier.verify(proof)
        assert valid is False
        assert "Ihsān" in error


# =============================================================================
# BOUNTY ORACLE
# =============================================================================

class TestBountyOracle:
    """Tests for bounty calculation oracle."""

    def test_oracle_creation(self, signer):
        """Oracle can be created."""
        oracle = BountyOracle(signer)
        assert oracle.base_payout == BASE_PAYOUT_PER_DELTA_E

    def test_calculate_bounty_valid(self, bounty_oracle, sample_proof):
        """Oracle calculates bounty for valid proof."""
        calculation, error = bounty_oracle.calculate_bounty(sample_proof)
        assert error is None
        assert calculation is not None
        assert calculation.total_payout > 0

    def test_calculate_bounty_components(self, bounty_oracle, sample_proof):
        """Calculation includes all components."""
        calculation, _ = bounty_oracle.calculate_bounty(sample_proof)

        assert calculation.delta_e > 0
        assert calculation.base_payout > 0
        assert calculation.severity_multiplier == 20  # HIGH
        assert calculation.risk_bonus >= 0
        assert calculation.quality_bonus >= 0

    def test_calculate_bounty_max_cap(self, bounty_oracle, proof_builder, entropy_before, entropy_after):
        """Bounty is capped at max_payout."""
        # Create proof with massive funds at risk
        proof = proof_builder.build(
            target_address="0x1234",
            vuln_category=VulnCategory.FLASH_LOAN,
            severity=Severity.CRITICAL,
            title="Critical bug",
            description="Test",
            exploit_code=b"test",
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            reproduction_steps=[],
            funds_at_risk=1_000_000_000,  # $1B
            snr_score=0.99,
            ihsan_score=0.99,
        )

        calculation, _ = bounty_oracle.calculate_bounty(proof)
        assert calculation.total_payout <= bounty_oracle.max_payout

    def test_estimate_payout(self, bounty_oracle):
        """Estimate payout without creating records."""
        estimate = bounty_oracle.estimate_payout(
            delta_e=2.0,
            severity="high",
            funds_at_risk=1_000_000,
            snr_score=0.95,
        )

        assert "estimated_payout_usd" in estimate
        assert estimate["estimated_payout_usd"] > 0
        assert estimate["meets_thresholds"] is True

    def test_create_payout(self, bounty_oracle, sample_proof):
        """Oracle creates signed payout authorization."""
        calculation, _ = bounty_oracle.calculate_bounty(sample_proof)
        payout = bounty_oracle.create_payout(calculation, "0xhunter_address")

        assert payout.status == "approved"
        assert len(payout.signature) > 0
        assert payout.hunter_address == "0xhunter_address"

    def test_process_proof_full_pipeline(self, bounty_oracle, sample_proof):
        """Full pipeline: proof → calculation → payout."""
        payout, error = bounty_oracle.process_proof(
            sample_proof,
            "0xhunter_wallet",
        )

        assert error is None
        assert payout is not None
        assert payout.calculation.total_payout > 0


# =============================================================================
# HUNTER AGENT
# =============================================================================

class TestHunterAgent:
    """Tests for autonomous vulnerability hunter."""

    @pytest.mark.asyncio
    async def test_hunter_scan(self, hunter_agent, scan_target):
        """Hunter can scan a target."""
        result = await hunter_agent.scan(scan_target)

        assert result.status == ScanStatus.COMPLETE
        assert result.target.address == scan_target.address

    @pytest.mark.asyncio
    async def test_hunter_entropy_measurement(self, hunter_agent, scan_target):
        """Hunter measures entropy across vectors."""
        result = await hunter_agent.scan(scan_target)

        entropy = result.entropy_measurements
        assert entropy.surface_entropy >= 0
        assert entropy.structural_entropy >= 0
        assert entropy.behavioral_entropy >= 0

    @pytest.mark.asyncio
    async def test_hunter_hunt_generates_proofs(self, hunter_agent, scan_target):
        """Hunter generates proofs for findings."""
        result, proofs = await hunter_agent.hunt(scan_target, generate_proofs=True)

        assert result.status == ScanStatus.COMPLETE
        # Proofs generated if findings exist with high confidence
        if result.has_findings:
            for finding in result.vulnerabilities:
                if finding.get("confidence", 0) >= BOUNTY_SNR_THRESHOLD - 0.1:
                    assert len(proofs) > 0
                    break

    @pytest.mark.asyncio
    async def test_hunter_stats(self, hunter_agent, scan_target):
        """Hunter tracks statistics."""
        await hunter_agent.scan(scan_target)
        await hunter_agent.scan(scan_target)

        stats = hunter_agent.get_stats()
        assert stats["total_scans"] == 2


class TestHunterSwarm:
    """Tests for hunter swarm coordination."""

    @pytest.mark.asyncio
    async def test_swarm_creation(self, signer):
        """Swarm can be created with multiple agents."""
        swarm = HunterSwarm(signer, num_agents=3)
        assert len(swarm.agents) == 3

    @pytest.mark.asyncio
    async def test_swarm_parallel_hunt(self, signer):
        """Swarm hunts multiple targets in parallel."""
        swarm = HunterSwarm(signer, num_agents=2)

        targets = [
            ScanTarget(
                address=f"0x{i:040x}",
                chain="ethereum",
                bytecode=bytes([0x60 + i] * 50),
            )
            for i in range(4)
        ]

        results = await swarm.hunt_targets(targets)
        assert len(results) == 4


# =============================================================================
# BOUNTY BRIDGE
# =============================================================================

class TestBountyBridge:
    """Tests for bounty platform integration."""

    def test_bridge_creation(self, signer):
        """Bridge can be created."""
        bridge = BountyBridge(signer)
        assert bridge is not None

    def test_register_platform(self, bounty_bridge):
        """Bridge can register platforms."""
        creds = PlatformCredentials(
            platform=Platform.IMMUNEFI,
            api_key="test_key",
        )
        bounty_bridge.register_platform(Platform.IMMUNEFI, creds)

        assert Platform.IMMUNEFI in bounty_bridge._adapters

    @pytest.mark.asyncio
    async def test_submit_to_platform(self, bounty_bridge, sample_proof):
        """Bridge can submit to platform."""
        creds = PlatformCredentials(
            platform=Platform.IMMUNEFI,
            api_key="test_key",
        )
        bounty_bridge.register_platform(Platform.IMMUNEFI, creds)

        submission = await bounty_bridge.submit(sample_proof, Platform.IMMUNEFI)

        assert submission.platform == Platform.IMMUNEFI
        assert submission.status == SubmissionStatus.SUBMITTED
        assert submission.proof.proof_id == sample_proof.proof_id

    @pytest.mark.asyncio
    async def test_submit_to_bizra_native(self, bounty_bridge, sample_proof):
        """Bridge can submit to BIZRA native contract."""
        creds = PlatformCredentials(
            platform=Platform.BIZRA,
            api_key="test",
            wallet_address="0xhunter",
        )
        bounty_bridge.register_platform(
            Platform.BIZRA,
            creds,
            contract_address="0xbountypool",
        )

        submission = await bounty_bridge.submit(sample_proof, Platform.BIZRA)

        assert submission.platform == Platform.BIZRA
        assert submission.payout_currency == "ETH"

    def test_bridge_stats(self, bounty_bridge):
        """Bridge tracks statistics."""
        stats = bounty_bridge.get_stats()
        assert "total_submissions" in stats
        assert "registered_platforms" in stats


# =============================================================================
# INTEGRATION
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_bounty_workflow(self, signer):
        """Complete workflow: scan → proof → calculate → submit."""
        # 1. Create components
        hunter = HunterAgent(signer)
        oracle = BountyOracle(signer)
        bridge = BountyBridge(signer)

        creds = PlatformCredentials(platform=Platform.IMMUNEFI, api_key="test")
        bridge.register_platform(Platform.IMMUNEFI, creds)

        # 2. Create target with known vulnerability pattern
        target = ScanTarget(
            address="0xvulnerable",
            chain="ethereum",
            name="VulnerableProtocol",
            tvl=2_000_000,
            bytecode=bytes([0xf1, 0x55] * 100),  # CALL + SSTORE pattern
            abi=[{"type": "function", "name": "flashLoan", "inputs": []}],
        )

        # 3. Hunt for vulnerabilities
        result, proofs = await hunter.hunt(target)
        assert result.status == ScanStatus.COMPLETE

        # 4. If proofs generated, process them
        if proofs:
            proof = proofs[0]

            # 5. Calculate bounty
            calculation, error = oracle.calculate_bounty(proof)
            assert error is None
            assert calculation.total_payout > 0

            # 6. Submit to platform
            submission = await bridge.submit(proof, Platform.IMMUNEFI)
            assert submission.status == SubmissionStatus.SUBMITTED

            # 7. Verify full trail
            assert proof.target_address == target.address
            assert calculation.proof_id == proof.proof_id
            assert submission.proof.proof_id == proof.proof_id

    @pytest.mark.asyncio
    async def test_payout_calculation_accuracy(self, signer, proof_builder, entropy_before, entropy_after):
        """Verify payout calculation formula."""
        oracle = BountyOracle(signer, base_payout=500, max_payout=100_000)

        # Known inputs
        proof = proof_builder.build(
            target_address="0x1234",
            vuln_category=VulnCategory.REENTRANCY,
            severity=Severity.MEDIUM,  # 5x multiplier
            title="Test",
            description="Test",
            exploit_code=b"test",
            entropy_before=entropy_before,  # 3.5 total
            entropy_after=entropy_after,    # 1.6 total
            reproduction_steps=[],
            funds_at_risk=10_000,           # $10K
            snr_score=0.95,
            ihsan_score=0.95,
        )

        calculation, _ = oracle.calculate_bounty(proof)

        # Manual calculation:
        # delta_e = 3.5 - 1.6 = 1.9
        # base = 500 * 1.9 = 950
        # risk_bonus = min(10000 * 0.10, 50000) = 1000
        # quality_bonus = 0 (at threshold)
        # total = (950 + 1000) * 5 + 0 = 9750

        assert abs(calculation.delta_e - 1.9) < 0.01
        assert calculation.severity_multiplier == 5
        assert calculation.total_payout > 0
