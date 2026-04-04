"""
BIZRA PCI Protocol — Proof-Carrying Inference
==============================================
Cryptographic message protocol for PAT↔SAT communication.

Version: 1.0.0
Status: PRODUCTION
Alignment: BIZRA_SOT.md Section 3.1 (Ihsān IM ≥ 0.95)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                       PCIEnvelope                           │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────────┐   │
    │  │ Sender  │ │ Payload │ │Metadata │ │ Ed25519 Signature│   │
    │  └─────────┘ └─────────┘ └─────────┘ └──────────────────┘   │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                       Gate Chain                            │
    │  CHEAP(<10ms)     MEDIUM(<150ms)    EXPENSIVE(<2000ms)      │
    │  ┌───────────┐    ┌───────────┐     ┌───────────┐           │
    │  │ SCHEMA    │    │ SNR       │     │ FATE      │           │
    │  │ SIGNATURE │    │ IHSAN     │     │ FORMAL    │           │
    │  │ TIMESTAMP │    │ POLICY    │     └───────────┘           │
    │  │ REPLAY    │    └───────────┘                             │
    │  │ ROLE      │                                              │
    │  └───────────┘                                              │
    └────────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     CommitReceipt                           │
    │  ┌──────────┐ ┌────────────┐ ┌───────────────────────────┐  │
    │  │CommitRef │ │Verification│ │Verifier Signatures (n/5)  │  │
    │  └──────────┘ └────────────┘ └───────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from core.pci import (
        PCIEnvelope,
        EnvelopeBuilder,
        GateChain,
        CommitReceipt,
        RejectCode,
        AgentType,
        verify_envelope,
        generate_keypair,
    )

    # Generate key pair
    keypair = generate_keypair()

    # Create envelope
    envelope = (EnvelopeBuilder()
        .with_sender(AgentType.PAT, "pat-001", keypair.public_key_hex)
        .with_action("propose", {"task": "analyze_code"})
        .with_policy(policy_hash)
        .with_state(state_hash)
        .with_scores(ihsan=0.97, snr=0.85)
        .build()
        .sign(keypair.private_key))

    # Verify through gate chain
    passed, rejection, results = verify_envelope(
        envelope,
        policy_hash=policy_hash,
        state_hash=state_hash,
    )

    if passed:
        # Generate commit receipt
        receipt = receipt_generator.create_receipt(...)
"""

from .types import (
    # Version
    PCI_VERSION,
    # Constants
    DOMAIN_PREFIX,
    NONCE_BYTES,
    TIMESTAMP_SKEW_SECONDS,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD_DEFAULT,
    LATENCY_BUDGET_CHEAP,
    LATENCY_BUDGET_MEDIUM,
    LATENCY_BUDGET_EXPENSIVE,
    # Enums
    AgentType,
    Urgency,
    VerificationTier,
    GateTier,
    Gate,
    CommitRefType,
    SignatureAlgorithm,
    # Data classes
    Sender,
    Payload,
    Metadata,
    Signature,
    CommitRef,
    Verification,
    VerifierSignature,
    Quorum,
    AuditTrail,
    # Utilities
    utc_now_iso,
    generate_envelope_id,
    generate_receipt_id,
)

from .reject_codes import (
    RejectCode,
    RejectionResponse,
    # Rejection helpers
    reject_schema,
    reject_signature,
    reject_ihsan,
    reject_snr,
    reject_replay,
    reject_timestamp_stale,
    reject_timestamp_future,
    reject_role_violation,
    reject_fate_violation,
    reject_internal_error,
)

from .crypto import (
    # Canonical JSON
    canonical_json,
    canonical_json_str,
    # BLAKE3 hashing
    blake3_digest,
    domain_separated_digest,
    envelope_digest,
    policy_hash,
    state_hash,
    # Nonce
    generate_nonce,
    validate_nonce,
    check_nonce_replay,
    # Ed25519
    KeyPair,
    generate_keypair,
    sign_message,
    verify_signature,
    sign_envelope,
    verify_envelope_signature,
    # Nonce cache
    NonceCache,
    get_nonce_cache,
)

from .envelope import (
    PCIEnvelope,
    EnvelopeBuilder,
    validate_envelope_schema,
)

from .receipt import (
    CommitReceipt,
    ReceiptGenerator,
    ReceiptStore,
    get_receipt_generator,
    get_receipt_store,
)

from .receipt_store_persistent import (
    StoredReceipt,
    ReceiptChain,
    JSONLReceiptStore,
    PostgreSQLReceiptStore,
    HybridReceiptStore,
    create_receipt_store,
    get_persistent_receipt_store,
    reset_persistent_receipt_store,
)

from .gates import (
    GateResult,
    GateChain,
    verify_envelope,
)

from .integration import (
    FlowResult,
    SATValidator,
    SATValidatorRegistry,
    PATSATBridge,
    create_pat_envelope,
    verify_with_sat,
    complete_pat_sat_flow,
    get_validator_registry,
)

__all__ = [
    # Version
    "PCI_VERSION",
    # Constants
    "DOMAIN_PREFIX",
    "NONCE_BYTES",
    "TIMESTAMP_SKEW_SECONDS",
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD_DEFAULT",
    "LATENCY_BUDGET_CHEAP",
    "LATENCY_BUDGET_MEDIUM",
    "LATENCY_BUDGET_EXPENSIVE",
    # Enums
    "AgentType",
    "Urgency",
    "VerificationTier",
    "GateTier",
    "Gate",
    "CommitRefType",
    "SignatureAlgorithm",
    "RejectCode",
    # Data classes
    "Sender",
    "Payload",
    "Metadata",
    "Signature",
    "CommitRef",
    "Verification",
    "VerifierSignature",
    "Quorum",
    "AuditTrail",
    "KeyPair",
    # Responses
    "RejectionResponse",
    "GateResult",
    # Main classes
    "PCIEnvelope",
    "EnvelopeBuilder",
    "CommitReceipt",
    "ReceiptGenerator",
    "ReceiptStore",
    "StoredReceipt",
    "ReceiptChain",
    "JSONLReceiptStore",
    "PostgreSQLReceiptStore",
    "HybridReceiptStore",
    "GateChain",
    "NonceCache",
    # Functions
    "utc_now_iso",
    "generate_envelope_id",
    "generate_receipt_id",
    "canonical_json",
    "canonical_json_str",
    "blake3_digest",
    "domain_separated_digest",
    "envelope_digest",
    "policy_hash",
    "state_hash",
    "generate_nonce",
    "validate_nonce",
    "check_nonce_replay",
    "generate_keypair",
    "sign_message",
    "verify_signature",
    "sign_envelope",
    "verify_envelope_signature",
    "validate_envelope_schema",
    "verify_envelope",
    "get_nonce_cache",
    "get_receipt_generator",
    "get_receipt_store",
    "create_receipt_store",
    "get_persistent_receipt_store",
    "reset_persistent_receipt_store",
    # Rejection helpers
    "reject_schema",
    "reject_signature",
    "reject_ihsan",
    "reject_snr",
    "reject_replay",
    "reject_timestamp_stale",
    "reject_timestamp_future",
    "reject_role_violation",
    "reject_fate_violation",
    "reject_internal_error",
    # Integration bridge
    "FlowResult",
    "SATValidator",
    "SATValidatorRegistry",
    "PATSATBridge",
    "create_pat_envelope",
    "verify_with_sat",
    "complete_pat_sat_flow",
    "get_validator_registry",
]
