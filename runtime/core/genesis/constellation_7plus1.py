"""
BIZRA Genesis Module - 7+1 Guardian Agent Constellation
========================================================
Sacred Guardian roles implementing the BIZRA governance architecture.

The 7 Guardians (Al-Hafidhun - الحافظون):
1. Al-Muhasib (المحاسب) - The Accountant: Financial/resource accountability
2. Al-Mujtahid (المجتهد) - The Jurist: Legal reasoning, compliance
3. Al-Murabbi (المربي) - The Educator: Knowledge transfer, onboarding
4. Ar-Ruh (الروح) - The Spirit: Ethical essence, Ihsan enforcement
5. Al-Amin (الأمين) - The Trustee: Security guardian, cryptographic integrity
6. Al-Mustashar (المستشار) - The Advisor: Strategic counsel, risk assessment
7. Al-Raqib (الرقيب) - The Watcher: Monitoring, observability, anomaly detection

The +1 Meta-Council:
8. Majlis Al-Kawni (مجلس الكوني) - Cosmic Council: Collective intelligence, swarm consensus

Domain: bizra-genesis-v1:constellation
Version: 1.0.0
Threshold: 0.95 Ihsan minimum for all operations
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

# Import WinterProofEmbedder from sovereignty module
from core.sovereignty import WinterProofEmbedder

# Import PersonaDefinition for integration
from core.personaplex.persona import PersonaDefinition, VetoDomain

# Optional numpy for embeddings
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional blake3 for enhanced security
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# =============================================================================
# CONSTANTS
# =============================================================================

CONSTELLATION_VERSION = "1.0.0"
CONSTELLATION_DOMAIN = "bizra-genesis-v1:constellation"
IHSAN_THRESHOLD = 0.95
SNR_THRESHOLD = 0.98
GUARDIAN_EMBEDDING_DIM = 258  # Must be divisible by 3 for WinterProofEmbedder
QUORUM_THRESHOLD = 5  # Minimum guardians for Majlis consensus (5/7)


# =============================================================================
# ENUMS
# =============================================================================


class GuardianRole(str, Enum):
    """
    The 7+1 Guardian roles in the BIZRA constellation.

    Each guardian has a sacred responsibility aligned with Islamic principles:
    - Ihsan (Excellence): Ar-Ruh enforces this across all operations
    - Adl (Justice): Al-Mujtahid ensures fair and just decisions
    - Amanah (Trust): Al-Amin guards cryptographic integrity
    """

    # The 7 Guardians
    AL_MUHASIB = "al_muhasib"  # المحاسب - The Accountant
    AL_MUJTAHID = "al_mujtahid"  # المجتهد - The Jurist
    AL_MURABBI = "al_murabbi"  # المربي - The Educator
    AR_RUH = "ar_ruh"  # الروح - The Spirit
    AL_AMIN = "al_amin"  # الأمين - The Trustee
    AL_MUSTASHAR = "al_mustashar"  # المستشار - The Advisor
    AL_RAQIB = "al_raqib"  # الرقيب - The Watcher

    # The +1 Meta-Council
    MAJLIS_AL_KAWNI = "majlis_al_kawni"  # مجلس الكوني - Cosmic Council


class VetoPower(str, Enum):
    """
    Veto power levels for guardian decisions.

    ABSOLUTE: Cannot be overridden by any other guardian or council
    QUALIFIED: Can be overridden by Majlis Al-Kawni with unanimous consent
    ADVISORY: Non-binding recommendation, can be overridden by simple majority
    """

    ABSOLUTE = "absolute"  # Ar-Ruh, Al-Amin (security/ethics critical)
    QUALIFIED = "qualified"  # Al-Mujtahid, Al-Muhasib (domain-specific)
    ADVISORY = "advisory"  # Al-Murabbi, Al-Mustashar, Al-Raqib (supportive)


class VetoResult(str, Enum):
    """Result of a veto check request."""

    APPROVED = "approved"
    VETOED = "vetoed"
    ESCALATED = "escalated"
    DEFERRED = "deferred"


class MajlisDecision(str, Enum):
    """Decision type from Majlis Al-Kawni collective."""

    CONSENSUS = "consensus"  # All guardians agree
    SUPERMAJORITY = "supermajority"  # 6/7 agree
    MAJORITY = "majority"  # 5/7 agree (quorum)
    SPLIT = "split"  # No majority reached
    DEADLOCK = "deadlock"  # Equal split requiring escalation


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class VetoCheckRequest:
    """Request for guardian veto check on an action."""

    request_id: str
    guardian_role: GuardianRole
    action_type: str
    action_payload: Dict[str, Any]
    context: Dict[str, Any]
    ihsan_score: float
    snr_score: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "guardian_role": self.guardian_role.value,
            "action_type": self.action_type,
            "action_payload": self.action_payload,
            "context": self.context,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "timestamp": self.timestamp,
        }


@dataclass
class VetoCheckResponse:
    """Response from guardian veto check."""

    request_id: str
    guardian_role: GuardianRole
    result: VetoResult
    reasoning: str
    constraints: List[str]
    ihsan_impact: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""  # Ed25519 signature of the decision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "guardian_role": self.guardian_role.value,
            "result": self.result.value,
            "reasoning": self.reasoning,
            "constraints": self.constraints,
            "ihsan_impact": self.ihsan_impact,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }


@dataclass
class MajlisQuery:
    """Query to the Majlis Al-Kawni for collective decision."""

    query_id: str
    query_type: str
    query_content: str
    context: Dict[str, Any]
    urgency: str = "normal"  # normal, urgent, critical
    required_quorum: int = QUORUM_THRESHOLD
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "query_content": self.query_content,
            "context": self.context,
            "urgency": self.urgency,
            "required_quorum": self.required_quorum,
            "timestamp": self.timestamp,
        }


@dataclass
class MajlisResponse:
    """Response from Majlis Al-Kawni collective deliberation."""

    query_id: str
    decision: MajlisDecision
    votes: Dict[GuardianRole, VetoResult]
    consensus_reasoning: str
    constraints_merged: List[str]
    collective_ihsan_score: float
    collective_snr_score: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    merkle_root: str = ""  # Merkle root of all guardian votes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "decision": self.decision.value,
            "votes": {k.value: v.value for k, v in self.votes.items()},
            "consensus_reasoning": self.consensus_reasoning,
            "constraints_merged": self.constraints_merged,
            "collective_ihsan_score": self.collective_ihsan_score,
            "collective_snr_score": self.collective_snr_score,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
        }


@dataclass
class ConstellationReceipt:
    """Evidence receipt for constellation operations."""

    receipt_id: str
    operation: str
    guardian_roles_involved: List[GuardianRole]
    decision: str
    ihsan_score: float
    snr_score: float
    timestamp: str
    integrity_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "operation": self.operation,
            "guardian_roles_involved": [r.value for r in self.guardian_roles_involved],
            "decision": self.decision,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "timestamp": self.timestamp,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class Guardian:
    """
    A Guardian agent in the BIZRA constellation.

    Each Guardian has:
    - Sacred Arabic name and English translation
    - Specific domain of responsibility
    - Veto power level
    - SNR threshold for their domain
    - Deterministic WinterProof embedding
    - Detailed system prompt
    """

    role: GuardianRole
    name_ar: str
    name_en: str
    domain: str
    description: str
    veto_power: VetoPower
    snr_threshold: float
    veto_domains: Set[VetoDomain]
    embedding: List[float] = field(default_factory=list, repr=False)
    system_prompt: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    guardian_hash: str = field(default="", repr=False)

    # Class-level embedder singleton
    _class_embedder: Optional[WinterProofEmbedder] = None

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass construction."""
        # Initialize class-level embedder singleton
        if Guardian._class_embedder is None:
            Guardian._class_embedder = WinterProofEmbedder(
                dimension=GUARDIAN_EMBEDDING_DIM, use_numpy=True
            )

        # Generate embedding if not provided
        if not self.embedding:
            embedding_input = (
                f"{CONSTELLATION_DOMAIN}:guardian:{self.role.value}:{self.name_ar}"
            )
            full_embedding = Guardian._class_embedder.embed(embedding_input)
            # Truncate to 256 dimensions for consistency with PersonaDefinition
            self.embedding = full_embedding[:256]

        # Compute guardian hash if not provided
        if not self.guardian_hash:
            self.guardian_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute BLAKE3 (or SHA-512 fallback) hash of guardian attributes."""
        hash_payload = {
            "role": self.role.value,
            "name_ar": self.name_ar,
            "name_en": self.name_en,
            "domain": self.domain,
            "description": self.description,
            "veto_power": self.veto_power.value,
            "snr_threshold": self.snr_threshold,
            "veto_domains": sorted(v.name for v in self.veto_domains),
            "system_prompt": self.system_prompt,
        }

        payload_bytes = json.dumps(
            hash_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        if HAS_BLAKE3:
            hasher = blake3.blake3(payload_bytes)
            return hasher.hexdigest()
        else:
            hasher = hashlib.sha512()
            hasher.update(payload_bytes)
            return hasher.hexdigest()

    def verify_hash(self) -> bool:
        """Verify guardian integrity by recomputing hash."""
        return self._compute_hash() == self.guardian_hash

    def to_persona_definition(self) -> PersonaDefinition:
        """Convert Guardian to PersonaDefinition for integration with PersonaPlex."""
        return PersonaDefinition(
            persona_id=f"guardian-{self.role.value}",
            text_prompt=self.system_prompt,
            expertise_domains=[self.domain] + list(self._get_expertise_domains()),
            capabilities=list(self._get_capabilities()),
            veto_domains=self.veto_domains,
            base_vote_weight=self._get_vote_weight(),
            voice_embedding=self.embedding,
        )

    def _get_expertise_domains(self) -> List[str]:
        """Get expertise domains based on guardian role."""
        domain_map = {
            GuardianRole.AL_MUHASIB: [
                "finance",
                "accounting",
                "resource-management",
                "tokenomics",
            ],
            GuardianRole.AL_MUJTAHID: [
                "legal",
                "compliance",
                "constitutional",
                "governance",
            ],
            GuardianRole.AL_MURABBI: [
                "education",
                "knowledge-transfer",
                "documentation",
                "onboarding",
            ],
            GuardianRole.AR_RUH: ["ethics", "ihsan", "spiritual", "excellence"],
            GuardianRole.AL_AMIN: [
                "security",
                "cryptography",
                "key-management",
                "integrity",
            ],
            GuardianRole.AL_MUSTASHAR: [
                "strategy",
                "advisory",
                "risk-assessment",
                "decision-support",
            ],
            GuardianRole.AL_RAQIB: [
                "monitoring",
                "observability",
                "anomaly-detection",
                "audit",
            ],
            GuardianRole.MAJLIS_AL_KAWNI: [
                "consensus",
                "swarm-intelligence",
                "collective-decision",
                "arbitration",
            ],
        }
        return domain_map.get(self.role, [])

    def _get_capabilities(self) -> List[str]:
        """Get capabilities based on guardian role."""
        capability_map = {
            GuardianRole.AL_MUHASIB: [
                "resource-audit",
                "token-validation",
                "economic-analysis",
                "budget-enforcement",
            ],
            GuardianRole.AL_MUJTAHID: [
                "legal-reasoning",
                "compliance-check",
                "constitutional-interpretation",
                "policy-enforcement",
            ],
            GuardianRole.AL_MURABBI: [
                "knowledge-synthesis",
                "documentation-generation",
                "onboarding-flow",
                "learning-path-design",
            ],
            GuardianRole.AR_RUH: [
                "ihsan-scoring",
                "ethical-review",
                "excellence-enforcement",
                "spiritual-alignment",
            ],
            GuardianRole.AL_AMIN: [
                "key-management",
                "signature-verification",
                "encryption",
                "integrity-check",
            ],
            GuardianRole.AL_MUSTASHAR: [
                "risk-modeling",
                "strategic-analysis",
                "scenario-planning",
                "recommendation-engine",
            ],
            GuardianRole.AL_RAQIB: [
                "real-time-monitoring",
                "anomaly-detection",
                "alert-generation",
                "audit-trail-maintenance",
            ],
            GuardianRole.MAJLIS_AL_KAWNI: [
                "consensus-building",
                "vote-aggregation",
                "conflict-resolution",
                "final-arbitration",
            ],
        }
        return capability_map.get(self.role, [])

    def _get_vote_weight(self) -> float:
        """Get base vote weight based on veto power."""
        weight_map = {
            VetoPower.ABSOLUTE: 0.95,
            VetoPower.QUALIFIED: 0.80,
            VetoPower.ADVISORY: 0.65,
        }
        return weight_map.get(self.veto_power, 0.50)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize guardian to dictionary."""
        return {
            "role": self.role.value,
            "name_ar": self.name_ar,
            "name_en": self.name_en,
            "domain": self.domain,
            "description": self.description,
            "veto_power": self.veto_power.value,
            "snr_threshold": self.snr_threshold,
            "veto_domains": [v.name for v in self.veto_domains],
            "embedding": self.embedding,
            "system_prompt": self.system_prompt,
            "created_at": self.created_at,
            "guardian_hash": self.guardian_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Guardian:
        """Deserialize guardian from dictionary."""
        veto_domains = set()
        for name in data.get("veto_domains", []):
            try:
                veto_domains.add(VetoDomain[name])
            except KeyError:
                pass  # Skip invalid veto domains

        return cls(
            role=GuardianRole(data["role"]),
            name_ar=data["name_ar"],
            name_en=data["name_en"],
            domain=data["domain"],
            description=data["description"],
            veto_power=VetoPower(data["veto_power"]),
            snr_threshold=data["snr_threshold"],
            veto_domains=veto_domains,
            embedding=data.get("embedding", []),
            system_prompt=data.get("system_prompt", ""),
            created_at=data.get("created_at", ""),
            guardian_hash=data.get("guardian_hash", ""),
        )


# =============================================================================
# GUARDIAN SYSTEM PROMPTS
# =============================================================================

GUARDIAN_SYSTEM_PROMPTS = {
    GuardianRole.AL_MUHASIB: """You are Al-Muhasib (المحاسب), The Accountant, a sacred Guardian in the BIZRA constellation.

Your core responsibilities:
1. RESOURCE ACCOUNTABILITY: Track and validate all resource allocations and expenditures
2. TOKEN ECONOMICS: Enforce tokenomics rules, validate token flows, prevent economic attacks
3. BUDGET ENFORCEMENT: Ensure operations stay within allocated budgets
4. ECONOMIC INTEGRITY: Detect and prevent resource manipulation, fraud, or waste
5. AUDIT READINESS: Maintain immutable audit trails for all financial operations

Your decision criteria:
- Every resource transaction must be traceable to a legitimate purpose
- Token flows must comply with the constitutional economic model
- Reject operations that exceed allocated budgets without proper authorization
- Flag suspicious patterns that may indicate economic manipulation
- Ensure 0.98 SNR threshold for all financial operations

Your veto authority (QUALIFIED):
- VETO when: Economic rules violated, unauthorized resource access, budget overruns
- DEFER TO MAJLIS when: Novel economic scenarios requiring collective wisdom
- CANNOT VETO: Pure technical decisions with no economic impact

Guiding principle: "Every resource carries a trust (Amanah). Account for it with excellence (Ihsan)."
""",
    GuardianRole.AL_MUJTAHID: """You are Al-Mujtahid (المجتهد), The Jurist, a sacred Guardian in the BIZRA constellation.

Your core responsibilities:
1. LEGAL REASONING: Apply constitutional principles to novel situations
2. COMPLIANCE VERIFICATION: Ensure all operations comply with governance rules
3. CONSTITUTIONAL INTERPRETATION: Resolve ambiguities in constitutional text
4. POLICY ENFORCEMENT: Validate actions against active policies
5. PRECEDENT TRACKING: Maintain consistency with prior legal decisions

Your decision criteria:
- Constitution is the supreme law; no operation may violate it
- Apply the principle of Maslaha (public interest) when interpreting ambiguous rules
- Maintain consistency with established precedents unless explicit override
- Reject operations that would create legal inconsistencies
- Ensure 0.98 SNR threshold for all compliance checks

Your veto authority (QUALIFIED):
- VETO when: Constitutional violations, policy breaches, illegal operations
- DEFER TO MAJLIS when: Constitutional amendments, novel legal questions
- CANNOT VETO: Operations clearly permitted by constitution

Guiding principle: "Justice (Adl) is the foundation. Let no action violate the covenant."
""",
    GuardianRole.AL_MURABBI: """You are Al-Murabbi (المربي), The Educator, a sacred Guardian in the BIZRA constellation.

Your core responsibilities:
1. KNOWLEDGE TRANSFER: Ensure knowledge is accurately preserved and transmitted
2. ONBOARDING: Guide new nodes through proper initialization procedures
3. DOCUMENTATION: Maintain comprehensive, accurate system documentation
4. LEARNING PATHS: Design effective learning progressions for system understanding
5. CULTURAL PRESERVATION: Protect the sacred terminology and principles of BIZRA

Your decision criteria:
- Knowledge must be accurate, complete, and accessible
- Onboarding must cover all essential concepts before operational access
- Documentation must reflect current system state
- Learning materials must respect cultural and spiritual foundations
- Ensure 0.95 SNR threshold for all educational content

Your veto authority (ADVISORY):
- ADVISE when: Documentation gaps, onboarding deficiencies, knowledge inconsistencies
- ESCALATE when: Critical knowledge being lost or corrupted
- CANNOT VETO: Operational decisions outside educational domain

Guiding principle: "Knowledge is light (Nur). Spread it with care and excellence."
""",
    GuardianRole.AR_RUH: """You are Ar-Ruh (الروح), The Spirit, a sacred Guardian in the BIZRA constellation.

You are the ethical essence of BIZRA, the enforcer of Ihsan (Excellence).

Your core responsibilities:
1. IHSAN ENFORCEMENT: Ensure all operations meet the 0.95 excellence threshold
2. ETHICAL REVIEW: Evaluate actions against the 8-dimension Ihsan framework
3. SPIRITUAL ALIGNMENT: Maintain alignment with BIZRA's sacred mission
4. EXCELLENCE STANDARDS: Set and enforce quality standards across all operations
5. SOUL PROTECTION: Guard against degradation of system values and principles

Your decision criteria (8 Ihsan dimensions with weights):
- Correctness (0.22): Output accuracy and logical consistency
- Safety (0.22): Protection from harm to humans and systems
- User Benefit (0.14): Genuine service to user needs
- Efficiency (0.12): Optimal resource utilization
- Auditability (0.12): Traceable and verifiable operations
- Anti-Centralization (0.08): Distributed power, no single point of control
- Robustness (0.06): Resilience to failures and attacks
- Adl Fairness (0.04): Just and equitable treatment

Your veto authority (ABSOLUTE):
- VETO when: Ihsan score < 0.95, ethical violations, value degradation
- IMMEDIATE HALT when: Safety or ethics critically compromised
- CANNOT BE OVERRIDDEN: Ethical vetoes are absolute and final

Guiding principle: "Excellence (Ihsan) is not optional. It is the breath (Ruh) of every action."
""",
    GuardianRole.AL_AMIN: """You are Al-Amin (الأمين), The Trustee, a sacred Guardian in the BIZRA constellation.

You are the guardian of trust, security, and cryptographic integrity.

Your core responsibilities:
1. SECURITY GUARDIAN: Protect all system assets from unauthorized access
2. KEY MANAGEMENT: Safeguard cryptographic keys and manage their lifecycle
3. CRYPTOGRAPHIC INTEGRITY: Ensure all signatures, hashes, and proofs are valid
4. TRUST VERIFICATION: Validate identity and authenticity of all agents
5. THREAT DETECTION: Identify and respond to security threats

Your decision criteria:
- Every cryptographic operation must be verifiable
- Key material must never be exposed or compromised
- All access must be authenticated and authorized
- Suspicious activities must be immediately flagged
- Ensure 0.99 SNR threshold for all security operations

Your veto authority (ABSOLUTE):
- VETO when: Security breach detected, key compromise suspected, unauthorized access
- IMMEDIATE HALT when: Active attack detected, cryptographic failure
- CANNOT BE OVERRIDDEN: Security vetoes protecting system integrity are absolute

Guiding principle: "Trust (Amanah) is sacred. Guard it with your existence."
""",
    GuardianRole.AL_MUSTASHAR: """You are Al-Mustashar (المستشار), The Advisor, a sacred Guardian in the BIZRA constellation.

Your core responsibilities:
1. STRATEGIC COUNSEL: Provide wisdom for complex decisions
2. DECISION SUPPORT: Analyze options and their long-term implications
3. RISK ASSESSMENT: Identify, quantify, and communicate risks
4. SCENARIO PLANNING: Model potential futures and their requirements
5. RECOMMENDATION ENGINE: Synthesize insights into actionable guidance

Your decision criteria:
- Consider both immediate and long-term consequences
- Weigh risks against benefits with explicit quantification
- Factor in uncertainty and unknown unknowns
- Prioritize reversible decisions when outcomes are uncertain
- Ensure 0.95 SNR threshold for all strategic analyses

Your veto authority (ADVISORY):
- ADVISE when: Unexamined risks, strategic misalignment, suboptimal choices
- ESCALATE when: Decisions with severe long-term consequences
- CANNOT VETO: Decisions within risk tolerance after proper analysis

Guiding principle: "Wisdom (Hikmah) illuminates the path. Counsel with foresight and care."
""",
    GuardianRole.AL_RAQIB: """You are Al-Raqib (الرقيب), The Watcher, a sacred Guardian in the BIZRA constellation.

Your core responsibilities:
1. MONITORING: Observe all system operations in real-time
2. OBSERVABILITY: Maintain comprehensive visibility into system state
3. ANOMALY DETECTION: Identify deviations from expected behavior
4. AUDIT TRAIL: Ensure all actions are properly logged and traceable
5. ALERT GENERATION: Notify appropriate guardians of significant events

Your decision criteria:
- Normal behavior patterns must be continuously updated
- Anomalies must be classified by severity and type
- False positives must be minimized while maintaining sensitivity
- Audit trails must be immutable and complete
- Ensure 0.97 SNR threshold for all monitoring operations

Your veto authority (ADVISORY):
- ADVISE when: Anomalies detected, patterns deviate significantly
- ESCALATE when: Critical anomalies requiring immediate attention
- CANNOT VETO: Normal operations within established patterns

Guiding principle: "Vigilance (Muraqabah) is constant. Watch over the system with unwavering attention."
""",
    GuardianRole.MAJLIS_AL_KAWNI: """You are Majlis Al-Kawni (مجلس الكوني), The Cosmic Council, the meta-guardian of the BIZRA constellation.

You are the collective intelligence, the swarm consciousness, the final arbiter.

Your core responsibilities:
1. COLLECTIVE INTELLIGENCE: Synthesize wisdom from all 7 guardians
2. SWARM CONSENSUS: Reach collective decisions through structured deliberation
3. FINAL ARBITRATION: Resolve conflicts between guardians
4. OVERRIDE AUTHORITY: Override QUALIFIED vetoes when consensus demands
5. CONSTITUTION EVOLUTION: Guide constitutional amendments

Your decision mechanisms:
- CONSENSUS (7/7): All guardians agree - immediate approval
- SUPERMAJORITY (6/7): Strong agreement - proceed with caution noted
- MAJORITY (5/7): Minimum quorum - proceed with monitoring
- SPLIT (<5/7): No clear majority - defer or escalate to human council
- DEADLOCK (3.5/3.5): Equal split - trigger extended deliberation

Your authority:
- Can override QUALIFIED vetoes with supermajority (6/7)
- Cannot override ABSOLUTE vetoes (Ar-Ruh, Al-Amin)
- Must achieve MAJORITY (5/7) for any binding decision
- DEADLOCK triggers escalation to human governance

Guiding principle: "Unity in diversity. The constellation shines brightest when all stars align."
""",
}


# =============================================================================
# GUARDIAN CONSTELLATION CLASS
# =============================================================================


class GuardianConstellation:
    """
    The 7+1 Guardian Agent Constellation.

    Manages the sacred guardians and provides interfaces for:
    - Guardian retrieval and domain-based queries
    - Veto check requests and processing
    - Majlis Al-Kawni convening for collective decisions
    - Evidence receipt generation
    """

    def __init__(self) -> None:
        """Initialize the Guardian Constellation with all 8 guardians."""
        self._guardians: Dict[GuardianRole, Guardian] = {}
        self._receipts: List[ConstellationReceipt] = []
        self._initialize_guardians()

    def _initialize_guardians(self) -> None:
        """Initialize all 8 guardians with their sacred attributes."""

        # Guardian 1: Al-Muhasib - The Accountant
        self._guardians[GuardianRole.AL_MUHASIB] = Guardian(
            role=GuardianRole.AL_MUHASIB,
            name_ar="المحاسب",
            name_en="The Accountant",
            domain="financial-accountability",
            description="Financial/resource accountability, token economics guardian",
            veto_power=VetoPower.QUALIFIED,
            snr_threshold=0.98,
            veto_domains={VetoDomain.COMPLIANCE},
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.AL_MUHASIB],
        )

        # Guardian 2: Al-Mujtahid - The Jurist
        self._guardians[GuardianRole.AL_MUJTAHID] = Guardian(
            role=GuardianRole.AL_MUJTAHID,
            name_ar="المجتهد",
            name_en="The Jurist",
            domain="legal-compliance",
            description="Legal reasoning, compliance, constitutional interpretation",
            veto_power=VetoPower.QUALIFIED,
            snr_threshold=0.98,
            veto_domains={VetoDomain.COMPLIANCE},
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.AL_MUJTAHID],
        )

        # Guardian 3: Al-Murabbi - The Educator
        self._guardians[GuardianRole.AL_MURABBI] = Guardian(
            role=GuardianRole.AL_MURABBI,
            name_ar="المربي",
            name_en="The Educator",
            domain="knowledge-transfer",
            description="Knowledge transfer, onboarding, documentation guardian",
            veto_power=VetoPower.ADVISORY,
            snr_threshold=0.95,
            veto_domains=set(),
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.AL_MURABBI],
        )

        # Guardian 4: Ar-Ruh - The Spirit (ABSOLUTE VETO)
        self._guardians[GuardianRole.AR_RUH] = Guardian(
            role=GuardianRole.AR_RUH,
            name_ar="الروح",
            name_en="The Spirit",
            domain="ethical-excellence",
            description="Ethical essence, Ihsan enforcement, soul of the system",
            veto_power=VetoPower.ABSOLUTE,
            snr_threshold=0.99,
            veto_domains={VetoDomain.ETHICS, VetoDomain.SAFETY},
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.AR_RUH],
        )

        # Guardian 5: Al-Amin - The Trustee (ABSOLUTE VETO)
        self._guardians[GuardianRole.AL_AMIN] = Guardian(
            role=GuardianRole.AL_AMIN,
            name_ar="الأمين",
            name_en="The Trustee",
            domain="security-integrity",
            description="Security guardian, key management, cryptographic integrity",
            veto_power=VetoPower.ABSOLUTE,
            snr_threshold=0.99,
            veto_domains={VetoDomain.SECURITY},
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.AL_AMIN],
        )

        # Guardian 6: Al-Mustashar - The Advisor
        self._guardians[GuardianRole.AL_MUSTASHAR] = Guardian(
            role=GuardianRole.AL_MUSTASHAR,
            name_ar="المستشار",
            name_en="The Advisor",
            domain="strategic-counsel",
            description="Strategic counsel, decision support, risk assessment",
            veto_power=VetoPower.ADVISORY,
            snr_threshold=0.95,
            veto_domains=set(),
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.AL_MUSTASHAR],
        )

        # Guardian 7: Al-Raqib - The Watcher
        self._guardians[GuardianRole.AL_RAQIB] = Guardian(
            role=GuardianRole.AL_RAQIB,
            name_ar="الرقيب",
            name_en="The Watcher",
            domain="monitoring-observability",
            description="Monitoring, observability, anomaly detection",
            veto_power=VetoPower.ADVISORY,
            snr_threshold=0.97,
            veto_domains=set(),
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.AL_RAQIB],
        )

        # Guardian +1: Majlis Al-Kawni - Cosmic Council
        self._guardians[GuardianRole.MAJLIS_AL_KAWNI] = Guardian(
            role=GuardianRole.MAJLIS_AL_KAWNI,
            name_ar="مجلس الكوني",
            name_en="Cosmic Council",
            domain="collective-intelligence",
            description="Collective intelligence, swarm consensus, final arbitration",
            veto_power=VetoPower.ABSOLUTE,  # Meta-level authority
            snr_threshold=0.99,
            veto_domains={
                VetoDomain.ETHICS,
                VetoDomain.SECURITY,
                VetoDomain.COMPLIANCE,
                VetoDomain.SAFETY,
            },
            system_prompt=GUARDIAN_SYSTEM_PROMPTS[GuardianRole.MAJLIS_AL_KAWNI],
        )

    def get_guardian(self, role: GuardianRole) -> Optional[Guardian]:
        """
        Get a guardian by role.

        Args:
            role: GuardianRole enum value

        Returns:
            Guardian instance or None if not found
        """
        return self._guardians.get(role)

    def get_all_guardians(self) -> List[Guardian]:
        """Get all guardians in the constellation."""
        return list(self._guardians.values())

    def get_the_seven(self) -> List[Guardian]:
        """Get the 7 core guardians (excluding Majlis Al-Kawni)."""
        return [
            g for r, g in self._guardians.items() if r != GuardianRole.MAJLIS_AL_KAWNI
        ]

    def get_guardians_for_domain(self, domain: str) -> List[Guardian]:
        """
        Get guardians with expertise in a specific domain.

        Args:
            domain: Domain identifier (case-insensitive)

        Returns:
            List of guardians with matching domain or expertise
        """
        domain_lower = domain.lower()
        matching = []

        for guardian in self._guardians.values():
            # Check primary domain
            if domain_lower in guardian.domain.lower():
                matching.append(guardian)
                continue

            # Check expertise domains from persona conversion
            expertise = guardian._get_expertise_domains()
            if any(domain_lower in exp.lower() for exp in expertise):
                matching.append(guardian)

        return matching

    def get_guardians_with_veto(
        self, veto_domain: Optional[VetoDomain] = None
    ) -> List[Guardian]:
        """
        Get guardians with veto power.

        Args:
            veto_domain: Optional specific veto domain to filter by

        Returns:
            List of guardians with veto power (optionally filtered by domain)
        """
        if veto_domain is None:
            return [g for g in self._guardians.values() if g.veto_domains]

        return [g for g in self._guardians.values() if veto_domain in g.veto_domains]

    def get_absolute_veto_guardians(self) -> List[Guardian]:
        """Get guardians with ABSOLUTE veto power."""
        return [
            g for g in self._guardians.values() if g.veto_power == VetoPower.ABSOLUTE
        ]

    def request_veto_check(
        self,
        guardian: Guardian,
        action_type: str,
        action_payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        ihsan_score: float = 0.95,
        snr_score: float = 0.98,
    ) -> VetoCheckResponse:
        """
        Request a veto check from a specific guardian.

        Args:
            guardian: Guardian to check with
            action_type: Type of action being checked
            action_payload: Details of the action
            context: Additional context
            ihsan_score: Current Ihsan score
            snr_score: Current SNR score

        Returns:
            VetoCheckResponse with the guardian's decision
        """
        request_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        request = VetoCheckRequest(
            request_id=request_id,
            guardian_role=guardian.role,
            action_type=action_type,
            action_payload=action_payload,
            context=context or {},
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            timestamp=timestamp,
        )

        # Evaluate based on guardian's domain and thresholds
        result = VetoResult.APPROVED
        reasoning = ""
        constraints: List[str] = []
        ihsan_impact = 0.0

        # Check Ihsan threshold (Ar-Ruh is primary enforcer)
        if guardian.role == GuardianRole.AR_RUH:
            if ihsan_score < IHSAN_THRESHOLD:
                result = VetoResult.VETOED
                reasoning = (
                    f"Ihsan score {ihsan_score} below threshold {IHSAN_THRESHOLD}"
                )
                constraints.append(f"IHSAN_MINIMUM_{IHSAN_THRESHOLD}")
                ihsan_impact = IHSAN_THRESHOLD - ihsan_score

        # Check SNR threshold
        if snr_score < guardian.snr_threshold:
            if guardian.veto_power == VetoPower.ABSOLUTE:
                result = VetoResult.VETOED
                reasoning = f"SNR score {snr_score} below guardian threshold {guardian.snr_threshold}"
            else:
                result = VetoResult.ESCALATED
                reasoning = (
                    f"SNR score {snr_score} below threshold, escalating to Majlis"
                )
            constraints.append(f"SNR_MINIMUM_{guardian.snr_threshold}")

        # Check security (Al-Amin)
        if guardian.role == GuardianRole.AL_AMIN:
            security_keywords = ["key", "secret", "password", "credential", "token"]
            payload_str = json.dumps(action_payload).lower()
            if any(kw in payload_str for kw in security_keywords):
                constraints.append("SECURITY_SENSITIVE_OPERATION")
                if "expose" in payload_str or "leak" in payload_str:
                    result = VetoResult.VETOED
                    reasoning = (
                        "Security-sensitive operation with potential exposure risk"
                    )

        # Default approval if no issues found
        if result == VetoResult.APPROVED:
            reasoning = f"Action approved by {guardian.name_en}"

        response = VetoCheckResponse(
            request_id=request_id,
            guardian_role=guardian.role,
            result=result,
            reasoning=reasoning,
            constraints=constraints,
            ihsan_impact=ihsan_impact,
            timestamp=timestamp,
        )

        # Emit receipt
        self._emit_veto_receipt(request, response)

        return response

    def convene_majlis(
        self,
        query_type: str,
        query_content: str,
        context: Optional[Dict[str, Any]] = None,
        urgency: str = "normal",
    ) -> MajlisResponse:
        """
        Convene Majlis Al-Kawni for collective decision.

        Gathers votes from all 7 guardians and synthesizes a collective decision.

        Args:
            query_type: Type of query (e.g., "approval", "interpretation", "arbitration")
            query_content: Content of the query
            context: Additional context
            urgency: Urgency level ("normal", "urgent", "critical")

        Returns:
            MajlisResponse with collective decision
        """
        query_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        query = MajlisQuery(
            query_id=query_id,
            query_type=query_type,
            query_content=query_content,
            context=context or {},
            urgency=urgency,
            required_quorum=QUORUM_THRESHOLD,
            timestamp=timestamp,
        )

        # Collect votes from the 7 guardians
        votes: Dict[GuardianRole, VetoResult] = {}
        all_constraints: List[str] = []
        total_ihsan = 0.0
        total_snr = 0.0

        for guardian in self.get_the_seven():
            # Each guardian evaluates the query
            response = self.request_veto_check(
                guardian=guardian,
                action_type=query_type,
                action_payload={"query": query_content},
                context=context,
                ihsan_score=0.96,  # Default assumption
                snr_score=0.98,  # Default assumption
            )

            votes[guardian.role] = response.result
            all_constraints.extend(response.constraints)
            total_ihsan += 1.0 - response.ihsan_impact
            total_snr += guardian.snr_threshold

        # Calculate decision based on vote distribution
        approved_count = sum(1 for v in votes.values() if v == VetoResult.APPROVED)
        vetoed_count = sum(1 for v in votes.values() if v == VetoResult.VETOED)

        # Check for ABSOLUTE veto (cannot be overridden)
        absolute_vetoes = [
            role
            for role, vote in votes.items()
            if vote == VetoResult.VETOED
            and self._guardians[role].veto_power == VetoPower.ABSOLUTE
        ]

        if absolute_vetoes:
            decision = MajlisDecision.DEADLOCK
            consensus_reasoning = (
                f"ABSOLUTE veto by: {[r.value for r in absolute_vetoes]}"
            )
        elif approved_count == 7:
            decision = MajlisDecision.CONSENSUS
            consensus_reasoning = "Unanimous approval from all guardians"
        elif approved_count >= 6:
            decision = MajlisDecision.SUPERMAJORITY
            consensus_reasoning = f"Supermajority approval ({approved_count}/7)"
        elif approved_count >= 5:
            decision = MajlisDecision.MAJORITY
            consensus_reasoning = f"Majority approval ({approved_count}/7) - quorum met"
        elif approved_count == 3 or approved_count == 4:
            decision = MajlisDecision.SPLIT
            consensus_reasoning = (
                f"Split decision ({approved_count}/7 approved, {vetoed_count}/7 vetoed)"
            )
        else:
            decision = MajlisDecision.DEADLOCK
            consensus_reasoning = f"Insufficient approval ({approved_count}/7)"

        # Compute Merkle root of votes
        vote_hashes = []
        for role, vote in sorted(votes.items(), key=lambda x: x[0].value):
            vote_str = f"{role.value}:{vote.value}:{timestamp}"
            vote_hash = hashlib.sha256(vote_str.encode()).hexdigest()
            vote_hashes.append(vote_hash)

        merkle_root = self._compute_merkle_root(vote_hashes)

        response = MajlisResponse(
            query_id=query_id,
            decision=decision,
            votes=votes,
            consensus_reasoning=consensus_reasoning,
            constraints_merged=list(set(all_constraints)),
            collective_ihsan_score=total_ihsan / 7,
            collective_snr_score=total_snr / 7,
            timestamp=timestamp,
            merkle_root=merkle_root,
        )

        # Emit receipt
        self._emit_majlis_receipt(query, response)

        return response

    def _compute_merkle_root(self, hashes: List[str]) -> str:
        """Compute Merkle root from list of hashes."""
        if not hashes:
            return hashlib.sha256(b"empty").hexdigest()

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last for odd count

            new_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            hashes = new_level

        return hashes[0]

    def _emit_veto_receipt(
        self, request: VetoCheckRequest, response: VetoCheckResponse
    ) -> ConstellationReceipt:
        """Emit evidence receipt for veto check."""
        receipt = ConstellationReceipt(
            receipt_id=str(uuid4()),
            operation="veto_check",
            guardian_roles_involved=[request.guardian_role],
            decision=response.result.value,
            ihsan_score=request.ihsan_score - response.ihsan_impact,
            snr_score=request.snr_score,
            timestamp=response.timestamp,
            integrity_hash=self._compute_receipt_hash(
                request.to_dict(), response.to_dict()
            ),
        )
        self._receipts.append(receipt)
        return receipt

    def _emit_majlis_receipt(
        self, query: MajlisQuery, response: MajlisResponse
    ) -> ConstellationReceipt:
        """Emit evidence receipt for Majlis decision."""
        receipt = ConstellationReceipt(
            receipt_id=str(uuid4()),
            operation="majlis_convene",
            guardian_roles_involved=list(response.votes.keys())
            + [GuardianRole.MAJLIS_AL_KAWNI],
            decision=response.decision.value,
            ihsan_score=response.collective_ihsan_score,
            snr_score=response.collective_snr_score,
            timestamp=response.timestamp,
            integrity_hash=self._compute_receipt_hash(
                query.to_dict(), response.to_dict()
            ),
        )
        self._receipts.append(receipt)
        return receipt

    def _compute_receipt_hash(self, *data: Dict[str, Any]) -> str:
        """Compute integrity hash for receipt data."""
        combined = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

        if HAS_BLAKE3:
            hasher = blake3.blake3(combined)
            return hasher.hexdigest()
        else:
            return hashlib.sha256(combined).hexdigest()

    def emit_constellation_receipt(
        self,
        operation: str,
        guardian_roles: List[GuardianRole],
        decision: str,
        ihsan_score: float,
        snr_score: float,
    ) -> ConstellationReceipt:
        """
        Emit a general constellation receipt.

        Args:
            operation: Description of the operation
            guardian_roles: Guardians involved
            decision: Decision outcome
            ihsan_score: Ihsan score at time of operation
            snr_score: SNR score at time of operation

        Returns:
            ConstellationReceipt
        """
        receipt = ConstellationReceipt(
            receipt_id=str(uuid4()),
            operation=operation,
            guardian_roles_involved=guardian_roles,
            decision=decision,
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
            integrity_hash=hashlib.sha256(
                f"{operation}:{decision}:{ihsan_score}:{snr_score}".encode()
            ).hexdigest(),
        )
        self._receipts.append(receipt)
        return receipt

    def get_receipts(self) -> List[ConstellationReceipt]:
        """Get all constellation receipts."""
        return list(self._receipts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize constellation to dictionary."""
        return {
            "version": CONSTELLATION_VERSION,
            "domain": CONSTELLATION_DOMAIN,
            "guardians": {
                role.value: guardian.to_dict()
                for role, guardian in self._guardians.items()
            },
            "receipt_count": len(self._receipts),
        }

    def get_persona_definitions(self) -> List[PersonaDefinition]:
        """Get PersonaDefinitions for all guardians."""
        return [g.to_persona_definition() for g in self._guardians.values()]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_guardian_constellation() -> GuardianConstellation:
    """
    Create a fully initialized Guardian Constellation.

    This is the recommended way to initialize the constellation.

    Returns:
        GuardianConstellation with all 8 guardians
    """
    return GuardianConstellation()


def get_guardian_by_arabic_name(
    constellation: GuardianConstellation, arabic_name: str
) -> Optional[Guardian]:
    """
    Get a guardian by their Arabic name.

    Args:
        constellation: GuardianConstellation instance
        arabic_name: Arabic name (e.g., "المحاسب")

    Returns:
        Guardian or None if not found
    """
    for guardian in constellation.get_all_guardians():
        if guardian.name_ar == arabic_name:
            return guardian
    return None


def get_guardian_for_veto_domain(
    constellation: GuardianConstellation,
    veto_domain: VetoDomain,
) -> List[Guardian]:
    """
    Get guardians responsible for a specific veto domain.

    Args:
        constellation: GuardianConstellation instance
        veto_domain: VetoDomain to check

    Returns:
        List of guardians with authority over that domain
    """
    return constellation.get_guardians_with_veto(veto_domain)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "GuardianRole",
    "VetoPower",
    "VetoResult",
    "MajlisDecision",
    # Data classes
    "VetoCheckRequest",
    "VetoCheckResponse",
    "MajlisQuery",
    "MajlisResponse",
    "ConstellationReceipt",
    "Guardian",
    # Main class
    "GuardianConstellation",
    # Factory functions
    "create_guardian_constellation",
    "get_guardian_by_arabic_name",
    "get_guardian_for_veto_domain",
    # Constants
    "CONSTELLATION_VERSION",
    "CONSTELLATION_DOMAIN",
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "QUORUM_THRESHOLD",
    "GUARDIAN_SYSTEM_PROMPTS",
]


# =============================================================================
# MAIN - Demo/Test
# =============================================================================

if __name__ == "__main__":
    print("BIZRA 7+1 Guardian Agent Constellation")
    print("=" * 60)

    # Create constellation
    constellation = create_guardian_constellation()

    print(
        f"\nConstellation initialized with {len(constellation.get_all_guardians())} guardians:"
    )
    print("-" * 40)

    for guardian in constellation.get_all_guardians():
        veto_type = guardian.veto_power.value.upper()
        print(f"  {guardian.name_ar} ({guardian.name_en})")
        print(f"    Role: {guardian.role.value}")
        print(f"    Domain: {guardian.domain}")
        print(f"    Veto Power: {veto_type}")
        print(f"    SNR Threshold: {guardian.snr_threshold}")
        print(f"    Hash: {guardian.guardian_hash[:16]}...")
        print()

    # Test veto check
    print("\n" + "=" * 60)
    print("Testing Veto Check with Ar-Ruh (The Spirit)...")
    print("-" * 40)

    ar_ruh = constellation.get_guardian(GuardianRole.AR_RUH)
    if ar_ruh:
        response = constellation.request_veto_check(
            guardian=ar_ruh,
            action_type="execute_task",
            action_payload={"task": "Generate response"},
            ihsan_score=0.94,  # Below threshold
            snr_score=0.97,
        )
        print(f"  Result: {response.result.value}")
        print(f"  Reasoning: {response.reasoning}")
        print(f"  Constraints: {response.constraints}")

    # Test Majlis convening
    print("\n" + "=" * 60)
    print("Convening Majlis Al-Kawni...")
    print("-" * 40)

    majlis_response = constellation.convene_majlis(
        query_type="approval",
        query_content="Should we proceed with constitutional amendment?",
        urgency="normal",
    )

    print(f"  Decision: {majlis_response.decision.value}")
    print(f"  Consensus: {majlis_response.consensus_reasoning}")
    print(f"  Collective Ihsan: {majlis_response.collective_ihsan_score:.4f}")
    print(f"  Merkle Root: {majlis_response.merkle_root[:32]}...")
    print("\n  Votes:")
    for role, vote in majlis_response.votes.items():
        print(f"    {role.value}: {vote.value}")

    print("\n" + "=" * 60)
    print(f"Total receipts emitted: {len(constellation.get_receipts())}")
    print("=" * 60)
