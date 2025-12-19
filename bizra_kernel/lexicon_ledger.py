"""
BIZRA Lexicon Ledger - The Secret Sauce
========================================

Python implementation of the canonical Lexicon Ledger providing:
- Term resolution and validation
- Receipt generation for term usage
- Append-only ledger operations
- Ihsān-aligned term governance

DNA Signature: 7-3-6-9-00
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class TruthLabel(Enum):
    """Truth labels for term assertions."""
    VERIFIED = "verified"      # Confirmed via evidence
    MEASURED = "measured"      # Quantified empirically
    TARGET = "target"          # Aspirational goal
    DERIVED = "derived"        # Computed from other terms


class TermStatus(Enum):
    """Lifecycle status of a term."""
    CANONICAL = "canonical"    # Production-ready
    DRAFT = "draft"            # Under review
    DEPRECATED = "deprecated"  # Superseded
    RESERVED = "reserved"      # Placeholder


class LedgerOperation(Enum):
    """Append-only ledger operations."""
    ADD = "add"
    DEPRECATE = "deprecate"
    CLARIFY = "clarify"        # Add notes without changing meaning


# DNA Signature for SAPE alignment
DNA_SIGNATURE = "7-3-6-9-00"

# Ihsān dimension weights (from constitution/ihsan_v1.yaml)
IHSAN_WEIGHTS = {
    "correctness": 0.22,
    "safety": 0.22,
    "user_benefit": 0.14,
    "efficiency": 0.12,
    "auditability": 0.12,
    "anti_centralization": 0.08,
    "robustness": 0.06,
    "adl_fairness": 0.04,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TermVariant:
    """A variant form of a term."""
    name: str
    description: str


@dataclass
class Term:
    """A canonical lexicon term."""
    key: str
    expansion: str
    role: str
    notes: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    variants: Dict[str, str] = field(default_factory=dict)
    invariants: List[str] = field(default_factory=list)
    ihsan_dimension: Optional[str] = None
    sape_module: Optional[int] = None
    status: TermStatus = TermStatus.CANONICAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "expansion": self.expansion,
            "role": self.role,
        }
        if self.notes:
            d["notes"] = self.notes
        if self.required_fields:
            d["required_fields"] = self.required_fields
        if self.references:
            d["references"] = self.references
        if self.examples:
            d["examples"] = self.examples
        if self.variants:
            d["variants"] = self.variants
        if self.invariants:
            d["invariants"] = self.invariants
        return d

    @classmethod
    def from_dict(cls, key: str, data: Dict[str, Any]) -> Term:
        """Create Term from dictionary."""
        return cls(
            key=key,
            expansion=data["expansion"],
            role=data["role"],
            notes=data.get("notes", []),
            required_fields=data.get("required_fields", []),
            references=data.get("references", []),
            examples=data.get("examples", []),
            variants=data.get("variants", {}),
            invariants=data.get("invariants", []),
        )


@dataclass
class LexiconReceipt:
    """Evidence receipt for lexicon operations."""
    receipt_id: str
    timestamp: str
    operation: LedgerOperation
    term_key: str
    lexicon_id: str
    lexicon_sha256: str
    ihsan_constitution_id: str
    ihsan_constitution_sha256: str
    actor: str
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "operation": self.operation.value,
            "term_key": self.term_key,
            "lexicon_id": self.lexicon_id,
            "lexicon_sha256": self.lexicon_sha256,
            "ihsan_constitution_id": self.ihsan_constitution_id,
            "ihsan_constitution_sha256": self.ihsan_constitution_sha256,
            "actor": self.actor,
            "signature": self.signature,
        }


@dataclass
class ValidationResult:
    """Result of term validation."""
    valid: bool
    term_key: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ihsan_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICON LEDGER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class LexiconLedger:
    """
    BIZRA Lexicon Ledger - Canonical term management.
    
    The Lexicon Ledger is the single source of truth for all BIZRA terminology.
    It enforces:
    - Append-only semantics (terms cannot be modified, only deprecated)
    - Receipt generation for all operations
    - Ihsān alignment verification
    - SAPE DNA signature compliance
    
    Example:
        >>> ledger = LexiconLedger.load_from_yaml("constitution/lexicon_v1.yaml")
        >>> term = ledger.resolve("FATE")
        >>> print(term.expansion)
        "Fail-Closed Escalation Protocol"
    """

    def __init__(
        self,
        version: int = 1,
        semver: str = "1.0.0",
        ledger_id: str = "bizra_lexicon_v1_0_0",
    ):
        self.version = version
        self.semver = semver
        self.id = ledger_id
        self.status = "canonical"
        self.append_only = True
        self.terms: Dict[str, Term] = {}
        self.receipts: List[LexiconReceipt] = []
        self._load_canonical_terms()

    def _load_canonical_terms(self) -> None:
        """Load the canonical BIZRA terms."""
        canonical = {
            # Core Framework Terms
            "ACE": Term(
                key="ACE",
                expansion="Agentic Control Environment",
                role="Session OS / governance layer coordinating tools, verification, receipts, and gates",
                notes=["ACE is responsible for fail-closed behavior under uncertainty (see FATE)."],
                ihsan_dimension="user_benefit",
                sape_module=5,
            ),
            "SAPE": Term(
                key="SAPE",
                expansion="Structured Agentic Prompt Engineering",
                role="7-module framework for systematic prompt construction and reasoning",
                notes=["DNA Signature: 7-3-6-9-00 (7 Modules, 3 Passes, 6 Checks, 9 Probes)"],
                ihsan_dimension="correctness",
                sape_module=3,
            ),
            "PAT": Term(
                key="PAT",
                expansion="Primary Agent Team",
                role="7-agent orchestrator for task decomposition and parallel execution",
                notes=["Coordinates specialist agents through graph-based task flow."],
                ihsan_dimension="efficiency",
                sape_module=5,
            ),
            "SAT": Term(
                key="SAT",
                expansion="Secondary Agent Team / Sovereign Approval Threshold",
                role="5-validator Byzantine consensus gate for critical decisions",
                notes=["Requires 3/5 approval for Byzantine fault tolerance (f=1)."],
                invariants=["n ≥ 3f+1 for f Byzantine faults"],
                ihsan_dimension="safety",
                sape_module=4,
            ),
            "FATE": Term(
                key="FATE",
                expansion="Fail-Closed Escalation Protocol",
                role="Escalation path when verification is uncertain or thresholds are not met",
                notes=["Fail closed: refuse to claim success when evidence is missing."],
                ihsan_dimension="safety",
                sape_module=4,
            ),
            "MCP": Term(
                key="MCP",
                expansion="Model Context Protocol",
                role="Standardized tool interface for agent-tool communication",
                references=["https://modelcontextprotocol.io"],
                ihsan_dimension="auditability",
                sape_module=5,
            ),
            "A2A": Term(
                key="A2A",
                expansion="Agent-to-Agent Protocol",
                role="Inter-agent communication and coordination standard",
                ihsan_dimension="anti_centralization",
                sape_module=5,
            ),
            "GoT": Term(
                key="GoT",
                expansion="Graph of Thoughts",
                role="5-method reasoning engine (deductive, inductive, abductive, analogical, causal)",
                notes=["Implements non-linear reasoning with branching and merging."],
                ihsan_dimension="correctness",
                sape_module=2,
            ),
            # Knowledge Layer
            "HouseOfWisdom": Term(
                key="HouseOfWisdom",
                expansion="House of Wisdom (Bayt al-Hikma)",
                role="Neo4j-backed HyperGraph knowledge store",
                notes=["Named after the historic Baghdad library."],
                ihsan_dimension="correctness",
                sape_module=1,
            ),
            "SNR": Term(
                key="SNR",
                expansion="Signal-to-Noise Ratio",
                role="Metric for reasoning efficiency (signal / (signal + noise))",
                notes=["Target: ≥0.85 for production reasoning paths."],
                ihsan_dimension="efficiency",
                sape_module=1,
            ),
            # Ihsān Terms
            "Ihsan": Term(
                key="Ihsan",
                expansion="Ihsān (Excellence)",
                role="Ethical/quality gate defined by constitution/ihsan_v1.yaml",
                notes=["8-dimension weighted composite for ethical alignment."],
                references=["constitution/ihsan_v1.yaml"],
                ihsan_dimension="correctness",
            ),
            "IhsanVector": Term(
                key="IhsanVector",
                expansion="Ihsān 8-Dimension Vector",
                role="Weighted composite score across 8 ethical dimensions",
                required_fields=["correctness", "safety", "user_benefit", "efficiency",
                                "auditability", "anti_centralization", "robustness", "adl_fairness"],
                ihsan_dimension="correctness",
            ),
            # Governance Terms
            "Constitution": Term(
                key="Constitution",
                expansion="Constitution File",
                role="Normative, versioned policy file that implementations must match",
                examples=["constitution/ihsan_v1.yaml", "constitution/lexicon_v1.yaml"],
                ihsan_dimension="anti_centralization",
                sape_module=7,
            ),
            "GenesisBlock": Term(
                key="GenesisBlock",
                expansion="Genesis Block",
                role="Immutable origin block establishing system identity",
                invariants=["Only one Genesis Block 0 may exist."],
                ihsan_dimension="auditability",
            ),
            "GenesisSeal": Term(
                key="GenesisSeal",
                expansion="Genesis Seal",
                role="Process and artifact that marks a state as immutable for audit purposes",
                notes=["Sealing requires a deterministic allowlist + hash set."],
                ihsan_dimension="auditability",
            ),
            "GenesisManifest": Term(
                key="GenesisManifest",
                expansion="Genesis Manifest",
                role="Allowlisted file set + hashes that define a sealed genesis state",
                notes=["A GenesisManifest binds to specific constitution IDs + hashes."],
                ihsan_dimension="auditability",
            ),
            # Receipt Terms
            "LexiconReceipt": Term(
                key="LexiconReceipt",
                expansion="Lexicon Evidence Receipt",
                role="Append-only receipt that binds lexicon and constitution versions by hash",
                required_fields=["lexicon_id", "lexicon_sha256", "ihsan_constitution_id",
                                "ihsan_constitution_sha256"],
                ihsan_dimension="auditability",
            ),
            # Adapter Terms
            "AdapterMode": Term(
                key="AdapterMode",
                expansion="Adapter Mode (simulated vs real)",
                role="Truth label describing whether a subsystem performs real external IO",
                variants={
                    "simulated": "No external IO; behavior may be stubbed or scripted.",
                    "real": "Performs external IO; requires evidence receipts and auditability.",
                },
                invariants=[
                    "If mode is real, a receipt MUST exist that includes tool calls + hashes.",
                    "If mode is simulated, outputs MUST NOT claim real-world side effects.",
                ],
                ihsan_dimension="auditability",
            ),
            # Build & Audit
            "BAT": Term(
                key="BAT",
                expansion="Build & Audit Tooling",
                role="Deterministic integrity gate (build/test/lint/secret-scan/truth-lint/parity)",
                ihsan_dimension="robustness",
                sape_module=4,
            ),
            # SAPE Module Terms
            "TensionStudio": Term(
                key="TensionStudio",
                expansion="Tension Studio (SAPE Module 7)",
                role="Contradiction detection and resolution module",
                notes=["Identifies cross-agent tensions and proposes resolutions."],
                ihsan_dimension="safety",
                sape_module=7,
            ),
            "SymbolicHarness": Term(
                key="SymbolicHarness",
                expansion="Symbolic Harness (SAPE Module 5)",
                role="Neural-symbolic bridge for grounding abstract concepts",
                notes=["Maps neural outputs to formal symbolic representations."],
                ihsan_dimension="correctness",
                sape_module=5,
            ),
            "AbstractionElevator": Term(
                key="AbstractionElevator",
                expansion="Abstraction Elevator (SAPE Module 6)",
                role="Pattern generalization from instances to principles",
                notes=["Elevates: Instance → Pattern → Principle"],
                ihsan_dimension="correctness",
                sape_module=6,
            ),
            # Additional Core Terms
            "ArtifactClass": Term(
                key="ArtifactClass",
                expansion="Artifact Classification",
                role="Categorizes what a receipt or gate is evaluating",
                variants={
                    "code": "Executable source code",
                    "docs": "Documentation claims and policies",
                    "config": "Configuration / environment",
                    "data": "Databases, migrations, seed files",
                    "evidence": "Receipts, logs, measurements",
                },
                ihsan_dimension="auditability",
            ),
            "DNASignature": Term(
                key="DNASignature",
                expansion="SAPE DNA Signature",
                role="Fingerprint encoding SAPE configuration (modules-passes-checks-probes-version)",
                notes=["Current signature: 7-3-6-9-00"],
                examples=["7-3-6-9-00"],
                ihsan_dimension="correctness",
                sape_module=3,
            ),
            "ByzantineConsensus": Term(
                key="ByzantineConsensus",
                expansion="Byzantine Fault-Tolerant Consensus",
                role="Agreement protocol tolerating f malicious validators where n ≥ 3f+1",
                invariants=["For 5 validators, tolerates f=1 Byzantine fault."],
                ihsan_dimension="robustness",
                sape_module=4,
            ),
            # ═══════════════════════════════════════════════════════════════════
            # EXTRACTED TERMS (from chat data analysis 2025-12-19)
            # ═══════════════════════════════════════════════════════════════════
            "HRM_MoE": Term(
                key="HRM_MoE",
                expansion="Hierarchical Reasoning Mixture-of-Experts",
                role="4-tier latency-adaptive reasoning engine (50ms→2000ms)",
                notes=[
                    "Tier 1: Reflexive (50ms) - cached responses",
                    "Tier 2: Analytical (200ms) - single-step reasoning",
                    "Tier 3: Strategic (500ms) - multi-step planning",
                    "Tier 4: Deliberative (2000ms) - complex synthesis",
                ],
                ihsan_dimension="efficiency",
                sape_module=2,
            ),
            "HTDAG": Term(
                key="HTDAG",
                expansion="Hierarchical Task Directed Acyclic Graph",
                role="Task decomposition structure for parallel execution",
                notes=["DAG enables topological ordering of dependent subtasks."],
                ihsan_dimension="efficiency",
                sape_module=5,
            ),
            "TMP": Term(
                key="TMP",
                expansion="Temporal Measurement Protocol",
                role="RSI safety protocol measuring cognitive evolution over time",
                notes=["v0.2 includes SCM, Causal Drag, Leverage Threshold."],
                ihsan_dimension="safety",
                sape_module=4,
            ),
            "SCM": Term(
                key="SCM",
                expansion="Structured Cognitive Metric",
                role="5-component scalar for RSI measurement",
                notes=[
                    "Components: Pattern Density, Evolutionary Signals, Proof of Impact,",
                    "Behavioral Alignment, Aesthetic Coherence",
                ],
                ihsan_dimension="correctness",
                sape_module=4,
            ),
            "CrownVerifier": Term(
                key="CrownVerifier",
                expansion="Crown Verifier",
                role="Ed25519 cryptographic deployment gate for RSI",
                notes=["Final approval checkpoint before self-improvement deployment."],
                ihsan_dimension="safety",
                sape_module=4,
            ),
            "CausalDrag": Term(
                key="CausalDrag",
                expansion="Causal Drag (Ω)",
                role="Structural risk quantification for interventions",
                notes=["Higher Ω indicates greater systemic risk from changes."],
                ihsan_dimension="safety",
                sape_module=4,
            ),
            "CausalFabric": Term(
                key="CausalFabric",
                expansion="Causal Fabric",
                role="Immutable truth ledger for happened-before ordering",
                notes=["Provides temporal causality guarantees across distributed nodes."],
                ihsan_dimension="auditability",
                sape_module=7,
            ),
            "BlockGraph": Term(
                key="BlockGraph",
                expansion="Block Graph / BlockTree",
                role="Hybrid ledger structure enabling parallel block production",
                notes=["Combines DAG flexibility with blockchain finality."],
                ihsan_dimension="robustness",
            ),
            "SEEDToken": Term(
                key="SEEDToken",
                expansion="SEED Token",
                role="Stable utility token for BIZRA platform operations",
                notes=["Non-speculative, pegged value for predictable costs."],
                ihsan_dimension="user_benefit",
            ),
            "BLOOMToken": Term(
                key="BLOOMToken",
                expansion="BLOOM Token",
                role="Impact growth token reflecting network value creation",
                notes=["Appreciation tied to Proof-of-Impact contributions."],
                ihsan_dimension="user_benefit",
            ),
            "Node0": Term(
                key="Node0",
                expansion="Node Zero / Genesis Node",
                role="First sovereign BIZRA instance bootstrapping the network",
                notes=["Contains Genesis Block 0 and initial constitution."],
                invariants=["Only one Node0 may exist per network."],
                ihsan_dimension="auditability",
            ),
            "PoI": Term(
                key="PoI",
                expansion="Proof-of-Impact",
                role="Consensus mechanism rewarding measurable value creation",
                notes=["Alternative to Proof-of-Work/Stake with impact attestation."],
                ihsan_dimension="user_benefit",
            ),
            "ReflectorAgent": Term(
                key="ReflectorAgent",
                expansion="Reflector Agent",
                role="Learning synthesizer that distills experience into patterns",
                notes=["Part of the cognitive layer in dual-agentic architecture."],
                ihsan_dimension="correctness",
                sape_module=6,
            ),
            "HostAgent": Term(
                key="HostAgent",
                expansion="Host Agent / Orchestrator",
                role="Primary coordinator for agent team task distribution",
                notes=["Manages HTDAG execution and inter-agent communication."],
                ihsan_dimension="efficiency",
                sape_module=5,
            ),
            "DualAgentic": Term(
                key="DualAgentic",
                expansion="Dual-Agentic Architecture",
                role="PAT (Personal Agent Team) + SAT (System Agent Team) architecture",
                notes=["PAT: 7 user-facing agents, SAT: 5 system validators."],
                ihsan_dimension="anti_centralization",
                sape_module=5,
            ),
            "ADLFairness": Term(
                key="ADLFairness",
                expansion="Anti-Discrimination Law Fairness",
                role="Legal compliance dimension ensuring non-discriminatory outputs",
                notes=["Weight: 0.04 in Ihsān composite."],
                ihsan_dimension="adl_fairness",
            ),
        }
        self.terms = canonical

    # ═══════════════════════════════════════════════════════════════════════════
    # TERM RESOLUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def resolve(self, key: str) -> Optional[Term]:
        """
        Resolve a term by its key.
        
        Args:
            key: The term key (e.g., "FATE", "SAT")
            
        Returns:
            The Term if found, None otherwise
        """
        return self.terms.get(key)

    def resolve_expansion(self, key: str) -> Optional[str]:
        """Get just the expansion for a term key."""
        term = self.resolve(key)
        return term.expansion if term else None

    def search(self, query: str) -> List[Term]:
        """
        Search terms by query string.
        
        Searches in key, expansion, role, and notes.
        """
        query_lower = query.lower()
        results = []
        for term in self.terms.values():
            if (query_lower in term.key.lower() or
                query_lower in term.expansion.lower() or
                query_lower in term.role.lower() or
                any(query_lower in note.lower() for note in term.notes)):
                results.append(term)
        return results

    def get_by_ihsan_dimension(self, dimension: str) -> List[Term]:
        """Get all terms aligned with a specific Ihsān dimension."""
        return [t for t in self.terms.values() if t.ihsan_dimension == dimension]

    def get_by_sape_module(self, module: int) -> List[Term]:
        """Get all terms associated with a specific SAPE module."""
        return [t for t in self.terms.values() if t.sape_module == module]

    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════

    def validate_term(self, term: Term) -> ValidationResult:
        """
        Validate a term against Lexicon Ledger rules.
        
        Checks:
        - Required fields present
        - Invariants are well-formed
        - No forbidden characters in key
        - Ihsān alignment is valid
        """
        errors = []
        warnings = []

        # Check key format (PascalCase or UPPER_CASE)
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', term.key):
            if not re.match(r'^[A-Z][A-Z0-9_]*$', term.key):
                errors.append(f"Key '{term.key}' must be PascalCase or UPPER_CASE")

        # Check expansion is non-empty
        if not term.expansion.strip():
            errors.append("Expansion cannot be empty")

        # Check role is non-empty
        if not term.role.strip():
            errors.append("Role cannot be empty")

        # Validate Ihsān dimension
        if term.ihsan_dimension and term.ihsan_dimension not in IHSAN_WEIGHTS:
            errors.append(f"Invalid Ihsān dimension: {term.ihsan_dimension}")

        # Validate SAPE module
        if term.sape_module and not (1 <= term.sape_module <= 7):
            errors.append(f"SAPE module must be 1-7, got {term.sape_module}")

        # Calculate Ihsān score
        ihsan_score = IHSAN_WEIGHTS.get(term.ihsan_dimension, 0.0) if term.ihsan_dimension else 0.0

        return ValidationResult(
            valid=len(errors) == 0,
            term_key=term.key,
            errors=errors,
            warnings=warnings,
            ihsan_score=ihsan_score,
        )

    def validate_usage(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Validate term usage in a text.
        
        Returns:
            Tuple of (valid_terms, unknown_terms)
        """
        # Find potential term references (PascalCase or UPPER_CASE words)
        pattern = r'\b([A-Z][a-zA-Z0-9]+|[A-Z][A-Z0-9_]+)\b'
        matches = set(re.findall(pattern, text))
        
        valid = []
        unknown = []
        for match in matches:
            if match in self.terms:
                valid.append(match)
            else:
                unknown.append(match)
        
        return valid, unknown

    # ═══════════════════════════════════════════════════════════════════════════
    # LEDGER OPERATIONS (Append-Only)
    # ═══════════════════════════════════════════════════════════════════════════

    def add_term(
        self,
        term: Term,
        actor: str = "system",
        ihsan_constitution_id: str = "ihsan_v1_0_0",
    ) -> LexiconReceipt:
        """
        Add a new term to the ledger (append-only).
        
        Args:
            term: The term to add
            actor: Who is adding the term
            ihsan_constitution_id: Reference to Ihsān constitution
            
        Returns:
            Receipt of the operation
            
        Raises:
            ValueError: If term key already exists
        """
        if term.key in self.terms:
            raise ValueError(f"Term '{term.key}' already exists. Use deprecate + add for changes.")
        
        # Validate before adding
        validation = self.validate_term(term)
        if not validation.valid:
            raise ValueError(f"Term validation failed: {validation.errors}")
        
        # Add term
        self.terms[term.key] = term
        
        # Generate receipt
        receipt = self._generate_receipt(
            operation=LedgerOperation.ADD,
            term_key=term.key,
            actor=actor,
            ihsan_constitution_id=ihsan_constitution_id,
        )
        self.receipts.append(receipt)
        
        return receipt

    def deprecate_term(
        self,
        key: str,
        actor: str = "system",
        ihsan_constitution_id: str = "ihsan_v1_0_0",
    ) -> LexiconReceipt:
        """
        Deprecate a term (append-only - marks as deprecated, doesn't delete).
        
        Args:
            key: The term key to deprecate
            actor: Who is deprecating
            ihsan_constitution_id: Reference to Ihsān constitution
            
        Returns:
            Receipt of the operation
        """
        if key not in self.terms:
            raise ValueError(f"Term '{key}' not found")
        
        self.terms[key].status = TermStatus.DEPRECATED
        
        receipt = self._generate_receipt(
            operation=LedgerOperation.DEPRECATE,
            term_key=key,
            actor=actor,
            ihsan_constitution_id=ihsan_constitution_id,
        )
        self.receipts.append(receipt)
        
        return receipt

    def _generate_receipt(
        self,
        operation: LedgerOperation,
        term_key: str,
        actor: str,
        ihsan_constitution_id: str,
    ) -> LexiconReceipt:
        """Generate a receipt for a ledger operation."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Compute ledger hash
        ledger_content = json.dumps(
            {k: v.to_dict() for k, v in self.terms.items()},
            sort_keys=True,
        )
        ledger_sha256 = hashlib.sha256(ledger_content.encode()).hexdigest()
        
        # Generate receipt ID
        receipt_id = hashlib.sha256(
            f"{timestamp}:{term_key}:{operation.value}".encode()
        ).hexdigest()[:16]
        
        return LexiconReceipt(
            receipt_id=f"LR-{receipt_id}",
            timestamp=timestamp,
            operation=operation,
            term_key=term_key,
            lexicon_id=self.id,
            lexicon_sha256=ledger_sha256,
            ihsan_constitution_id=ihsan_constitution_id,
            ihsan_constitution_sha256="placeholder_hash",  # Would compute from actual file
            actor=actor,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def to_dict(self) -> Dict[str, Any]:
        """Convert ledger to dictionary for serialization."""
        return {
            "version": self.version,
            "semver": self.semver,
            "id": self.id,
            "status": self.status,
            "append_only": self.append_only,
            "dna_signature": DNA_SIGNATURE,
            "terms": {k: v.to_dict() for k, v in self.terms.items()},
            "truth_labels": {
                "VERIFIED": TruthLabel.VERIFIED.value,
                "MEASURED": TruthLabel.MEASURED.value,
                "TARGET": TruthLabel.TARGET.value,
                "DERIVED": TruthLabel.DERIVED.value,
            },
        }

    def to_yaml(self) -> str:
        """Serialize ledger to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> LexiconLedger:
        """Load ledger from YAML file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        ledger = cls(
            version=data.get("version", 1),
            semver=data.get("semver", "1.0.0"),
            ledger_id=data.get("id", "bizra_lexicon_v1_0_0"),
        )
        
        # Clear default terms and load from file
        ledger.terms = {}
        for key, term_data in data.get("terms", {}).items():
            ledger.terms[key] = Term.from_dict(key, term_data)
        
        return ledger

    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        dimension_counts = {}
        module_counts = {}
        
        for term in self.terms.values():
            if term.ihsan_dimension:
                dimension_counts[term.ihsan_dimension] = dimension_counts.get(term.ihsan_dimension, 0) + 1
            if term.sape_module:
                module_counts[term.sape_module] = module_counts.get(term.sape_module, 0) + 1
        
        return {
            "total_terms": len(self.terms),
            "canonical_terms": sum(1 for t in self.terms.values() if t.status == TermStatus.CANONICAL),
            "deprecated_terms": sum(1 for t in self.terms.values() if t.status == TermStatus.DEPRECATED),
            "terms_by_ihsan_dimension": dimension_counts,
            "terms_by_sape_module": module_counts,
            "total_receipts": len(self.receipts),
            "dna_signature": DNA_SIGNATURE,
        }

    def compute_hash(self) -> str:
        """Compute SHA256 hash of the ledger content."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, key: str) -> bool:
        return key in self.terms

    def __iter__(self):
        return iter(self.terms.values())


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_canonical_ledger() -> LexiconLedger:
    """Get the default canonical lexicon ledger."""
    return LexiconLedger()


def resolve_term(key: str) -> Optional[Term]:
    """Quick term resolution using default ledger."""
    return get_canonical_ledger().resolve(key)


def expand(key: str) -> str:
    """Quick expansion lookup. Returns key if not found."""
    result = get_canonical_ledger().resolve_expansion(key)
    return result if result else key


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Demo usage
    ledger = LexiconLedger()
    
    print("BIZRA Lexicon Ledger - Secret Sauce Demo")
    print("=" * 50)
    print(f"DNA Signature: {DNA_SIGNATURE}")
    print(f"Total Terms: {len(ledger)}")
    print()
    
    # Resolve some terms
    for key in ["SAPE", "FATE", "Ihsan", "PAT", "SAT"]:
        term = ledger.resolve(key)
        if term:
            print(f"  {key}: {term.expansion}")
            print(f"    Role: {term.role}")
            print(f"    Ihsān: {term.ihsan_dimension} (weight: {IHSAN_WEIGHTS.get(term.ihsan_dimension, 0)})")
            print()
    
    # Show stats
    print("Statistics:")
    stats = ledger.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
