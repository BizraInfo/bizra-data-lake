"""
Reason Codes — Machine-readable rejection/quarantine reasons.

Every non-APPROVED decision MUST include at least one reason code.
These codes are the "language" that makes rejection explainable,
auditable, and actionable.

Standing on Giants:
- HTTP status codes (Fielding, 2000): Standardized error communication
- gRPC error model (Google): Rich error details
- BIZRA Spearpoint PRD: "make rejection explainable"
"""

from enum import Enum


class ReasonCode(str, Enum):
    """
    Unified reason codes for the Spearpoint pipeline.

    Categories:
    - EVIDENCE_*: Evidence integrity failures
    - CLAIM_*: Claim provenance violations
    - SNR_*/IHSAN_*: Quality gate failures
    - SIGNATURE_*/REPLAY_*/TIMESTAMP_*/GENESIS_*: Cryptographic failures
    - AUTH_*/RATE_*/BUDGET_*: Authorization and budgets
    - HOOK_*/TOOL_*: Tool runtime safety
    - POLICY_*/SCHEMA_*/INVARIANT_*/ROLE_*: Policy enforcement
    """

    # Evidence integrity
    EVIDENCE_MISSING = "EVIDENCE_MISSING"
    EVIDENCE_TAMPERED = "EVIDENCE_TAMPERED"
    EVIDENCE_EXPIRED = "EVIDENCE_EXPIRED"
    NONDETERMINISTIC_BUILD = "NONDETERMINISTIC_BUILD"

    # Claim provenance
    CLAIM_UNTAGGED = "CLAIM_UNTAGGED"
    CLAIM_TAG_INVALID = "CLAIM_TAG_INVALID"
    CLAIM_INFLATION = "CLAIM_INFLATION"

    # Quality gates
    SNR_BELOW_THRESHOLD = "SNR_BELOW_THRESHOLD"
    IHSAN_BELOW_THRESHOLD = "IHSAN_BELOW_THRESHOLD"
    CONFIDENCE_BELOW_THRESHOLD = "CONFIDENCE_BELOW_THRESHOLD"

    # Cryptographic
    UNSIGNED_ATTESTATION = "UNSIGNED_ATTESTATION"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    REPLAY_DETECTED = "REPLAY_DETECTED"
    TIMESTAMP_STALE = "TIMESTAMP_STALE"
    TIMESTAMP_FUTURE = "TIMESTAMP_FUTURE"
    GENESIS_MISMATCH = "GENESIS_MISMATCH"

    # Authorization
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTH_INVALID = "AUTH_INVALID"
    RATE_LIMITED = "RATE_LIMITED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

    # Tool runtime
    HOOK_NOT_ALLOWLISTED = "HOOK_NOT_ALLOWLISTED"
    HOOK_EXEC_DENIED = "HOOK_EXEC_DENIED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_OUTPUT_UNVERIFIABLE = "TOOL_OUTPUT_UNVERIFIABLE"

    # Policy
    POLICY_MISMATCH = "POLICY_MISMATCH"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    INVARIANT_FAILED = "INVARIANT_FAILED"
    ROLE_VIOLATION = "ROLE_VIOLATION"


# Human-readable descriptions (for logs and error messages)
REASON_DESCRIPTIONS: dict[str, str] = {
    "EVIDENCE_MISSING": "Required evidence artifact not found",
    "EVIDENCE_TAMPERED": "Evidence hash mismatch — content modified",
    "EVIDENCE_EXPIRED": "Evidence beyond acceptable freshness window",
    "NONDETERMINISTIC_BUILD": "Build artifact hash differs across runs",
    "CLAIM_UNTAGGED": "Output contains claims without provenance tags",
    "CLAIM_TAG_INVALID": "Claim tag does not match actual verification status",
    "CLAIM_INFLATION": "Claim tagged MEASURED but no measurement evidence exists",
    "SNR_BELOW_THRESHOLD": "Signal-to-noise ratio below minimum policy threshold",
    "IHSAN_BELOW_THRESHOLD": "Ihsan excellence score below minimum policy threshold",
    "CONFIDENCE_BELOW_THRESHOLD": "Confidence score below minimum for this operation",
    "UNSIGNED_ATTESTATION": "Attestation envelope has no signature",
    "SIGNATURE_INVALID": "Ed25519 signature verification failed",
    "REPLAY_DETECTED": "Nonce has been seen before within TTL window",
    "TIMESTAMP_STALE": "Timestamp exceeds maximum message age",
    "TIMESTAMP_FUTURE": "Timestamp is in the future beyond tolerance",
    "GENESIS_MISMATCH": "Envelope genesis_hash does not match node genesis",
    "AUTH_REQUIRED": "Authentication token required but not provided",
    "AUTH_INVALID": "Authentication token is invalid or expired",
    "RATE_LIMITED": "Request rate exceeds configured limit",
    "BUDGET_EXCEEDED": "Resource budget (bytes/time/calls) exhausted",
    "HOOK_NOT_ALLOWLISTED": "Requested hook is not in the allowlist",
    "HOOK_EXEC_DENIED": "Hook execution denied by policy",
    "TOOL_TIMEOUT": "Tool execution exceeded wall-time budget",
    "TOOL_OUTPUT_UNVERIFIABLE": "Tool produced output that cannot be verified",
    "POLICY_MISMATCH": "Request policy version does not match node policy",
    "SCHEMA_VIOLATION": "Input does not conform to expected schema",
    "INVARIANT_FAILED": "System invariant check failed",
    "ROLE_VIOLATION": "Agent role not authorized for this operation",
}


def describe(code: ReasonCode) -> str:
    """Get human-readable description for a reason code."""
    return REASON_DESCRIPTIONS.get(code.value, code.value)


__all__ = ["ReasonCode", "REASON_DESCRIPTIONS", "describe"]
