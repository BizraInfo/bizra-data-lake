"""
BIZRA Genesis Module - Genesis Block State Machine
===================================================
Core genesis functionality for BIZRA node initialization and activation.

This module implements:
- PhaseStateMachine: 4-phase genesis lifecycle (Primordial -> Activation)
- GenesisSealer: BLAKE3 + Ed25519 seal generation with domain separation
- GenesisVerifier: Proof attestation chain verification
- GuardianConstellation: 7+1 sacred guardian agent constellation

Domain: bizra-genesis-v1:
Threshold: 0.95 Ihsan minimum (fail-closed on ambiguous state)

Architecture:
    Phase 0: PRIMORDIAL - Integrity checks, cryptographic validation
    Phase 1: AWAKENING  - Agent initialization, warm pool bootstrap
    Phase 2: CRYSTALLIZATION - Consensus formation, quorum establishment
    Phase 3: ACTIVATION - Full operational mode

Guardian Constellation (7+1):
    The 7 Guardians (Al-Hafidhun):
    1. Al-Muhasib (المحاسب) - The Accountant
    2. Al-Mujtahid (المجتهد) - The Jurist
    3. Al-Murabbi (المربي) - The Educator
    4. Ar-Ruh (الروح) - The Spirit (ABSOLUTE VETO)
    5. Al-Amin (الأمين) - The Trustee (ABSOLUTE VETO)
    6. Al-Mustashar (المستشار) - The Advisor
    7. Al-Raqib (الرقيب) - The Watcher

    The +1 Meta-Council:
    8. Majlis Al-Kawni (مجلس الكوني) - Cosmic Council

Usage:
    from core.genesis import (
        # Phase Machine
        PhaseState,
        PhaseTransition,
        GenesisStateMachine,
        GenesisSeal,
        GenesisSealer,
        ProofAttestation,
        GenesisVerifier,
        # Guardian Constellation
        GuardianRole,
        VetoPower,
        Guardian,
        GuardianConstellation,
        create_guardian_constellation,
    )

    # Initialize genesis state machine
    machine = GenesisStateMachine()

    # Check current phase
    current = machine.current_phase  # PhaseState.PRIMORDIAL

    # Transition with validation
    if machine.can_transition(PhaseState.AWAKENING):
        await machine.transition(PhaseState.AWAKENING)

    # Seal the phase
    sealer = GenesisSealer(private_key=your_key)
    seal = sealer.seal_phase(PhaseState.AWAKENING, attestations)

    # Verify attestation chain
    verifier = GenesisVerifier()
    is_valid = await verifier.verify_attestation_chain(attestations)

    # Initialize guardian constellation
    constellation = create_guardian_constellation()
    ar_ruh = constellation.get_guardian(GuardianRole.AR_RUH)

    # Request veto check
    response = constellation.request_veto_check(
        guardian=ar_ruh,
        action_type="execute_task",
        action_payload={"task": "Generate response"},
        ihsan_score=0.96,
    )

    # Convene Majlis for collective decision
    majlis_response = constellation.convene_majlis(
        query_type="approval",
        query_content="Should we proceed?",
    )
"""

from .phase_machine import (
    # Enums
    PhaseState,
    TransitionResult,
    # Data classes
    PhaseTransition,
    PhaseRequirement,
    PhaseReceipt,
    # Main class
    GenesisStateMachine,
)

from .sealer import (
    # Data classes
    GenesisSeal,
    SealAttestation,
    # Main class
    GenesisSealer,
    # Constants
    GENESIS_DOMAIN_PREFIX,
    GENESIS_VERSION,
)

from .verifier import (
    # Enums
    VerificationStatus,
    ConstraintType,
    # Data classes
    ProofAttestation,
    VerificationResult,
    SovereigntyConstraint,
    # Main class
    GenesisVerifier,
)

from .constellation_7plus1 import (
    # Enums
    GuardianRole,
    VetoPower,
    VetoResult,
    MajlisDecision,
    # Data classes
    VetoCheckRequest,
    VetoCheckResponse,
    MajlisQuery,
    MajlisResponse,
    ConstellationReceipt,
    Guardian,
    # Main class
    GuardianConstellation,
    # Factory functions
    create_guardian_constellation,
    get_guardian_by_arabic_name,
    get_guardian_for_veto_domain,
    # Constants
    CONSTELLATION_VERSION,
    CONSTELLATION_DOMAIN,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    QUORUM_THRESHOLD,
    GUARDIAN_SYSTEM_PROMPTS,
)

__all__ = [
    # Phase Machine
    "PhaseState",
    "TransitionResult",
    "PhaseTransition",
    "PhaseRequirement",
    "PhaseReceipt",
    "GenesisStateMachine",
    # Sealer
    "GenesisSeal",
    "SealAttestation",
    "GenesisSealer",
    "GENESIS_DOMAIN_PREFIX",
    "GENESIS_VERSION",
    # Verifier
    "VerificationStatus",
    "ConstraintType",
    "ProofAttestation",
    "VerificationResult",
    "SovereigntyConstraint",
    "GenesisVerifier",
    # Guardian Constellation - Enums
    "GuardianRole",
    "VetoPower",
    "VetoResult",
    "MajlisDecision",
    # Guardian Constellation - Data classes
    "VetoCheckRequest",
    "VetoCheckResponse",
    "MajlisQuery",
    "MajlisResponse",
    "ConstellationReceipt",
    "Guardian",
    # Guardian Constellation - Main class
    "GuardianConstellation",
    # Guardian Constellation - Factory functions
    "create_guardian_constellation",
    "get_guardian_by_arabic_name",
    "get_guardian_for_veto_domain",
    # Guardian Constellation - Constants
    "CONSTELLATION_VERSION",
    "CONSTELLATION_DOMAIN",
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "QUORUM_THRESHOLD",
    "GUARDIAN_SYSTEM_PROMPTS",
]

__version__ = "1.0.0"
__domain__ = "bizra-genesis-v1:"
