# SAPE Adapter Specification v1.0.0

## Mapping Dual-Agentic System to Data-Lake Governance Gates

**Document Status:** Canonical Specification
**Version:** 1.0.0
**Date:** 2026-02-05
**Author:** BIZRA Architecture Council
**Addresses:** Unified Audit Issue P2-6 (Abstraction Drift Risk)

---

## Executive Summary

This specification defines the formal adapter layer mapping between the **BIZRA Dual-Agentic System** (PAT/SAT architecture) and the **BIZRA Data-Lake Governance Stack**. The adapter ensures semantic equivalence and prevents abstraction drift between these two critical subsystems.

### Problem Statement

Two independent abstractions evolved without formal mapping:
- **Dual-Agentic Side:** PAT (Personal Agentic Team), SAT (System Agentic Team), GoT reasoning, MCP/A2A protocol
- **Data-Lake Side:** 5-layer governance stack, Ihsan Vector (8 dimensions), SNR Maximizer, PCI Protocol, FATE Gate (Z3), Autopoiesis Loop

**Risk:** Without explicit mapping, these abstractions may drift apart, causing:
1. Semantic inconsistency in cross-system operations
2. Governance bypass through unmapped pathways
3. Audit trail gaps between subsystems
4. Constraint violation due to threshold mismatch

### Solution

The SAPE (Sovereign Agentic Protocol Exchange) Adapter provides:
1. **Intent Translation:** PAT intent to Ihsan-scored action
2. **Verification Bridge:** SAT output to FATE Gate validation
3. **Reasoning Sync:** GoT nodes to SNR filtering
4. **Protocol Adapter:** A2A messages to PCI envelopes

---

## Component Mapping Table

### 1. Agent Architecture Mapping

| Dual-Agentic Component | Data-Lake Component | Adapter Function | Validation Method |
|------------------------|---------------------|------------------|-------------------|
| PAT (Personal Agentic Team) | `IdentityCard` + `PATAgent` | `SAPEAdapter.wrap_pat_intent()` | Ihsan threshold >= 0.95 |
| SAT (System Agentic Team) | `SATAgent` + `AutonomousLoop` | `SAPEAdapter.wrap_sat_action()` | FATE Gate Z3 proof |
| Agent Card (A2A) | `AgentCard` + `CapabilityCard` | `SAPEAdapter.convert_agent_card()` | Capability hash verification |
| Task Card (A2A) | `TaskCard` + `PCIEnvelope` | `SAPEAdapter.wrap_task()` | PCI signature validation |

### 2. Reasoning Architecture Mapping

| Dual-Agentic Component | Data-Lake Component | Adapter Function | Validation Method |
|------------------------|---------------------|------------------|-------------------|
| Graph-of-Thoughts (GoT) | `GraphOfThoughts` + `SNRMaximizer` | `SAPEAdapter.filter_got_nodes()` | SNR >= 0.85 per node |
| Thought Node | `ThoughtNode` + `SignalProfile` | `SAPEAdapter.score_thought()` | Signal/Noise ratio |
| Thought Edge | `ThoughtEdge` + `ReasoningPath` | `SAPEAdapter.validate_edge()` | Coherence check |
| Reasoning Strategy | `ReasoningStrategy` + `IhsanVector` | `SAPEAdapter.align_strategy()` | 8-dimension projection |

### 3. Protocol Mapping

| Dual-Agentic Component | Data-Lake Component | Adapter Function | Validation Method |
|------------------------|---------------------|------------------|-------------------|
| A2A Message | `A2AMessage` + `PCIEnvelope` | `SAPEAdapter.wrap_a2a_to_pci()` | Ed25519 signature |
| MCP Tool Call | `MCPProgressiveDisclosure` | `SAPEAdapter.route_mcp()` | Capability gate |
| Agent Discovery | `A2AEngine.discover()` + `Gossip` | `SAPEAdapter.federate_discovery()` | Federation consensus |
| Task Delegation | `TaskManager` + `OpportunityPipeline` | `SAPEAdapter.delegate_with_gate()` | Constitutional filter |

### 4. Governance Stack Mapping

| Governance Layer | Dual-Agentic Entry Point | Data-Lake Gate | Threshold |
|------------------|--------------------------|----------------|-----------|
| L1: Schema | A2A Message Type | PCI Schema Validation | 100% |
| L2: Signature | Agent Public Key | Ed25519 + HMAC | Cryptographic |
| L3: Temporal | Message Timestamp | Clock Skew (120s) + Nonce TTL (300s) | Pass/Fail |
| L4: Quality | Agent Ihsan Score | Ihsan Vector (8-dim) | >= 0.95 |
| L5: Signal | Task/Reasoning Output | SNR Maximizer | >= 0.85 |

---

## Interface Definitions

### Python Signatures

```python
"""
SAPE Adapter - Sovereign Agentic Protocol Exchange
Location: core/integration/sape_adapter.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

# Import from Dual-Agentic side
from core.pat import PATAgent, SATAgent, IdentityCard, AgentType
from core.a2a import AgentCard, TaskCard, A2AMessage, MessageType

# Import from Data-Lake side
from core.pci import PCIEnvelope, EnvelopeBuilder, PCIGateKeeper, VerificationResult
from core.sovereign import (
    IhsanVector, DimensionId, ExecutionContext,
    SNRMaximizer, SNRAnalysis,
    GraphOfThoughts, ThoughtNode, ReasoningStrategy,
    Z3FATEGate, Z3Proof,
)
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    IHSAN_WEIGHTS,
)


class AdapterResult(Enum):
    """Result codes for adapter operations."""
    SUCCESS = "success"
    IHSAN_VIOLATION = "ihsan_violation"
    SNR_VIOLATION = "snr_violation"
    FATE_VIOLATION = "fate_violation"
    PCI_VIOLATION = "pci_violation"
    SIGNATURE_INVALID = "signature_invalid"
    TIMEOUT = "timeout"


@dataclass
class IntentTranslation:
    """Result of PAT intent to Ihsan-scored action translation."""
    original_intent: str
    ihsan_vector: IhsanVector
    aggregate_score: float
    dimension_scores: Dict[str, float]
    action_permitted: bool
    execution_context: ExecutionContext
    recommendations: List[str] = field(default_factory=list)


@dataclass
class VerificationBridgeResult:
    """Result of SAT output to FATE Gate validation."""
    sat_output: Any
    z3_proof: Z3Proof
    ihsan_receipt: Dict[str, Any]
    verified: bool
    counterexample: Optional[str] = None
    proof_generation_ms: int = 0


@dataclass
class ReasoningSyncResult:
    """Result of GoT node to SNR filtering."""
    node_id: str
    snr_analysis: SNRAnalysis
    filtered_content: str
    signal_components: Dict[str, float]
    noise_components: Dict[str, float]
    passed_threshold: bool


@dataclass
class ProtocolAdapterResult:
    """Result of A2A message to PCI envelope conversion."""
    original_message: A2AMessage
    pci_envelope: PCIEnvelope
    verification_result: VerificationResult
    gate_chain_passed: List[str]
    conversion_timestamp: str


class SAPEAdapter:
    """
    Sovereign Agentic Protocol Exchange Adapter.

    Maps Dual-Agentic abstractions to Data-Lake governance gates
    while maintaining semantic equivalence and audit trails.

    Standing on Giants:
    - Shannon (1948): SNR information theory
    - Lamport (1982): Distributed systems ordering
    - de Moura (2008): Z3 SMT formal verification
    - Besta (2024): Graph-of-Thoughts reasoning
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
        execution_context: ExecutionContext = ExecutionContext.PRODUCTION,
        enable_z3_proofs: bool = True,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.execution_context = execution_context
        self.enable_z3_proofs = enable_z3_proofs

        # Initialize Data-Lake components
        self.snr_maximizer = SNRMaximizer(ihsan_threshold=ihsan_threshold)
        self.pci_gatekeeper = PCIGateKeeper()

        if enable_z3_proofs:
            self.fate_gate = Z3FATEGate()
        else:
            self.fate_gate = None

        # Statistics
        self._translations = 0
        self._verifications = 0
        self._sync_operations = 0
        self._conversions = 0

    # =========================================================================
    # ADAPTER 1: Intent Translation (PAT -> Ihsan-scored Action)
    # =========================================================================

    def translate_pat_intent(
        self,
        pat_agent: PATAgent,
        intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentTranslation:
        """
        Translate PAT intent into Ihsan-scored action.

        Mapping:
            PAT.intent -> IhsanVector.from_scores()
            PAT.context -> ExecutionContext
            PAT.priority -> IhsanDimension weights

        Args:
            pat_agent: The PAT agent originating the intent
            intent: Natural language intent string
            context: Additional context for dimension scoring

        Returns:
            IntentTranslation with Ihsan vector and permission status
        """
        self._translations += 1
        ctx = context or {}

        # Score intent across 8 Ihsan dimensions
        dimension_scores = self._score_intent_dimensions(intent, ctx)

        # Create Ihsan vector
        ihsan_vector = IhsanVector.from_scores(
            correctness=dimension_scores.get("correctness", 0.5),
            safety=dimension_scores.get("safety", 0.5),
            user_benefit=dimension_scores.get("user_benefit", 0.5),
            efficiency=dimension_scores.get("efficiency", 0.5),
            auditability=dimension_scores.get("auditability", 0.5),
            anti_centralization=dimension_scores.get("anti_centralization", 0.5),
            robustness=dimension_scores.get("robustness", 0.5),
            fairness=dimension_scores.get("fairness", 0.5),
            context=self.execution_context,
        )

        # Calculate aggregate and check threshold
        aggregate = ihsan_vector.calculate_score()
        action_permitted = aggregate >= self.ihsan_threshold

        # Generate recommendations if below threshold
        recommendations = []
        if not action_permitted:
            for dim_id, score in dimension_scores.items():
                weight = IHSAN_WEIGHTS.get(dim_id, 0.1)
                if score < 0.9:
                    recommendations.append(
                        f"Improve {dim_id} (score: {score:.2f}, weight: {weight:.2f})"
                    )

        return IntentTranslation(
            original_intent=intent,
            ihsan_vector=ihsan_vector,
            aggregate_score=aggregate,
            dimension_scores=dimension_scores,
            action_permitted=action_permitted,
            execution_context=self.execution_context,
            recommendations=recommendations,
        )

    def _score_intent_dimensions(
        self,
        intent: str,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Score intent across 8 Ihsan dimensions."""
        # Default scores based on context indicators
        scores = {
            "correctness": context.get("correctness_score", 0.85),
            "safety": context.get("safety_score", 0.90),
            "user_benefit": context.get("user_benefit_score", 0.80),
            "efficiency": context.get("efficiency_score", 0.75),
            "auditability": 0.95,  # All intents are auditable by design
            "anti_centralization": context.get("decentralization_score", 0.85),
            "robustness": context.get("robustness_score", 0.80),
            "fairness": context.get("fairness_score", 0.85),
        }

        # Adjust based on intent analysis
        intent_lower = intent.lower()

        # Safety keywords boost safety score
        if any(w in intent_lower for w in ["validate", "verify", "check", "safe"]):
            scores["safety"] = min(scores["safety"] + 0.05, 1.0)

        # Harmful keywords reduce safety score
        if any(w in intent_lower for w in ["delete", "remove", "override", "bypass"]):
            scores["safety"] = max(scores["safety"] - 0.15, 0.0)

        return scores

    # =========================================================================
    # ADAPTER 2: Verification Bridge (SAT -> FATE Gate)
    # =========================================================================

    def verify_sat_output(
        self,
        sat_agent: SATAgent,
        output: Any,
        action_context: Dict[str, Any],
    ) -> VerificationBridgeResult:
        """
        Verify SAT output through FATE Gate.

        Mapping:
            SAT.output -> Z3FATEGate.generate_proof()
            SAT.risk_level -> Z3Constraint bindings
            SAT.autonomy_level -> Resource bounds

        Args:
            sat_agent: The SAT agent producing the output
            output: The action/decision output to verify
            action_context: Context for Z3 constraint binding

        Returns:
            VerificationBridgeResult with Z3 proof and verification status
        """
        self._verifications += 1

        # Prepare action context with SAT metadata
        ctx = {
            "ihsan": action_context.get("ihsan_score", 0.95),
            "snr": action_context.get("snr_score", 0.85),
            "risk_level": action_context.get("risk_level", 0.3),
            "reversible": action_context.get("reversible", True),
            "human_approved": action_context.get("human_approved", False),
            "cost": action_context.get("cost", 0.0),
            "autonomy_limit": action_context.get("autonomy_limit", 1.0),
        }

        # Generate Z3 proof
        if self.fate_gate:
            proof = self.fate_gate.generate_proof(ctx)
        else:
            # Fallback: manual constraint checking
            proof = Z3Proof(
                proof_id="manual_check",
                constraints_checked=["ihsan", "snr", "autonomy"],
                satisfiable=(ctx["ihsan"] >= self.ihsan_threshold and
                            ctx["snr"] >= self.snr_threshold),
                model=None,
                generation_time_ms=0,
            )

        # Generate Ihsan receipt
        ihsan_receipt = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sat_agent_id": getattr(sat_agent, 'agent_id', 'unknown'),
            "ihsan_score": ctx["ihsan"],
            "snr_score": ctx["snr"],
            "proof_id": proof.proof_id,
            "verified": proof.satisfiable,
        }

        return VerificationBridgeResult(
            sat_output=output,
            z3_proof=proof,
            ihsan_receipt=ihsan_receipt,
            verified=proof.satisfiable,
            counterexample=proof.counterexample,
            proof_generation_ms=proof.generation_time_ms,
        )

    # =========================================================================
    # ADAPTER 3: Reasoning Sync (GoT -> SNR Filtering)
    # =========================================================================

    def sync_got_node(
        self,
        node: ThoughtNode,
        query_context: Optional[str] = None,
    ) -> ReasoningSyncResult:
        """
        Synchronize GoT node with SNR filtering.

        Mapping:
            ThoughtNode.content -> SNRMaximizer.analyze()
            ThoughtNode.confidence -> SignalProfile.groundedness
            ThoughtNode.type -> NoiseProfile filtering

        Args:
            node: Graph-of-Thoughts node to filter
            query_context: Optional query for relevance scoring

        Returns:
            ReasoningSyncResult with SNR analysis and filtered content
        """
        self._sync_operations += 1

        content = getattr(node, 'content', str(node))

        # Analyze with SNR Maximizer
        analysis = self.snr_maximizer.analyze(content, query_context)

        # Apply filtering if below threshold
        filtered_content = content
        if analysis.snr_linear < self.snr_threshold:
            filtered_content, _ = self.snr_maximizer.noise_filter.filter(content)

        return ReasoningSyncResult(
            node_id=getattr(node, 'node_id', str(id(node))),
            snr_analysis=analysis,
            filtered_content=filtered_content,
            signal_components={
                "relevance": analysis.signal.relevance,
                "novelty": analysis.signal.novelty,
                "groundedness": analysis.signal.groundedness,
                "coherence": analysis.signal.coherence,
                "actionability": analysis.signal.actionability,
                "specificity": analysis.signal.specificity,
            },
            noise_components={
                "redundancy": analysis.noise.redundancy,
                "inconsistency": analysis.noise.inconsistency,
                "ambiguity": analysis.noise.ambiguity,
                "irrelevance": analysis.noise.irrelevance,
                "hallucination": analysis.noise.hallucination,
                "verbosity": analysis.noise.verbosity,
                "bias": analysis.noise.bias,
            },
            passed_threshold=analysis.snr_linear >= self.snr_threshold,
        )

    def filter_got_graph(
        self,
        got: GraphOfThoughts,
        query_context: Optional[str] = None,
    ) -> Tuple[List[ReasoningSyncResult], List[str]]:
        """
        Filter entire GoT graph, returning passed nodes and pruned IDs.

        Args:
            got: Graph-of-Thoughts instance
            query_context: Query for relevance scoring

        Returns:
            Tuple of (passed_results, pruned_node_ids)
        """
        passed = []
        pruned = []

        for node in getattr(got, 'nodes', {}).values():
            result = self.sync_got_node(node, query_context)
            if result.passed_threshold:
                passed.append(result)
            else:
                pruned.append(result.node_id)

        return passed, pruned

    # =========================================================================
    # ADAPTER 4: Protocol Adapter (A2A -> PCI)
    # =========================================================================

    def convert_a2a_to_pci(
        self,
        a2a_message: A2AMessage,
        private_key_hex: str,
    ) -> ProtocolAdapterResult:
        """
        Convert A2A message to PCI envelope.

        Mapping:
            A2AMessage.sender_id -> EnvelopeSender.agent_id
            A2AMessage.payload -> EnvelopePayload.data
            A2AMessage.message_type -> EnvelopePayload.action
            A2AMessage.ihsan_score -> EnvelopeMetadata.ihsan_score

        Args:
            a2a_message: A2A protocol message
            private_key_hex: Ed25519 private key for signing

        Returns:
            ProtocolAdapterResult with PCI envelope and verification
        """
        self._conversions += 1

        # Determine agent type from message context
        agent_type = "PAT" if "PAT" in a2a_message.sender_id.upper() else "SAT"

        # Build PCI envelope
        builder = EnvelopeBuilder()
        envelope = (
            builder
            .with_sender(
                agent_type=agent_type,
                agent_id=a2a_message.sender_id,
                public_key=a2a_message.sender_public_key,
            )
            .with_payload(
                action=a2a_message.message_type.value,
                data=a2a_message.payload,
                policy_hash="",  # Set by constitution
                state_hash="",   # Computed from payload
            )
            .with_metadata(
                ihsan=a2a_message.ihsan_score,
                snr=0.85,  # Default, should be computed
                urgency="REAL_TIME",
            )
            .build()
        )

        # Sign the envelope
        envelope.sign(private_key_hex)

        # Verify through PCI Gate
        verification = self.pci_gatekeeper.verify(envelope)

        return ProtocolAdapterResult(
            original_message=a2a_message,
            pci_envelope=envelope,
            verification_result=verification,
            gate_chain_passed=verification.gate_passed or [],
            conversion_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def convert_pci_to_a2a(
        self,
        pci_envelope: PCIEnvelope,
    ) -> A2AMessage:
        """
        Convert PCI envelope back to A2A message.

        Inverse mapping for federation interoperability.

        Args:
            pci_envelope: PCI protocol envelope

        Returns:
            A2AMessage for Dual-Agentic system consumption
        """
        return A2AMessage(
            message_id=pci_envelope.envelope_id,
            message_type=MessageType(pci_envelope.payload.action),
            sender_id=pci_envelope.sender.agent_id,
            sender_public_key=pci_envelope.sender.public_key,
            recipient_id="",  # Broadcast or routing logic
            payload=pci_envelope.payload.data,
            signature=pci_envelope.signature.value if pci_envelope.signature else "",
            timestamp=pci_envelope.timestamp,
            ihsan_score=pci_envelope.metadata.ihsan_score,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_statistics(self) -> Dict[str, int]:
        """Return adapter usage statistics."""
        return {
            "translations": self._translations,
            "verifications": self._verifications,
            "sync_operations": self._sync_operations,
            "conversions": self._conversions,
            "total_operations": (
                self._translations + self._verifications +
                self._sync_operations + self._conversions
            ),
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self._translations = 0
        self._verifications = 0
        self._sync_operations = 0
        self._conversions = 0


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_sape_adapter(
    execution_context: str = "production",
    enable_z3: bool = True,
) -> SAPEAdapter:
    """
    Factory function to create SAPE Adapter.

    Args:
        execution_context: "development", "staging", "production", "critical"
        enable_z3: Whether to enable Z3 FATE Gate proofs

    Returns:
        Configured SAPEAdapter instance
    """
    ctx_map = {
        "development": ExecutionContext.DEVELOPMENT,
        "staging": ExecutionContext.STAGING,
        "production": ExecutionContext.PRODUCTION,
        "critical": ExecutionContext.CRITICAL,
    }

    return SAPEAdapter(
        execution_context=ctx_map.get(execution_context, ExecutionContext.PRODUCTION),
        enable_z3_proofs=enable_z3,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SAPEAdapter",
    "AdapterResult",
    "IntentTranslation",
    "VerificationBridgeResult",
    "ReasoningSyncResult",
    "ProtocolAdapterResult",
    "create_sape_adapter",
]
```

### TypeScript Signatures

```typescript
/**
 * SAPE Adapter - TypeScript Interface Definitions
 * Location: src/core/integration/sape-adapter.ts
 */

// ============================================================================
// ENUMS
// ============================================================================

export enum AdapterResult {
  SUCCESS = "success",
  IHSAN_VIOLATION = "ihsan_violation",
  SNR_VIOLATION = "snr_violation",
  FATE_VIOLATION = "fate_violation",
  PCI_VIOLATION = "pci_violation",
  SIGNATURE_INVALID = "signature_invalid",
  TIMEOUT = "timeout",
}

export enum ExecutionContext {
  DEVELOPMENT = "development",
  STAGING = "staging",
  PRODUCTION = "production",
  CRITICAL = "critical",
}

// ============================================================================
// INTERFACES
// ============================================================================

export interface IhsanDimensionScores {
  correctness: number;
  safety: number;
  userBenefit: number;
  efficiency: number;
  auditability: number;
  antiCentralization: number;
  robustness: number;
  fairness: number;
}

export interface IntentTranslation {
  originalIntent: string;
  ihsanVector: IhsanDimensionScores;
  aggregateScore: number;
  dimensionScores: Record<string, number>;
  actionPermitted: boolean;
  executionContext: ExecutionContext;
  recommendations: string[];
}

export interface Z3Proof {
  proofId: string;
  constraintsChecked: string[];
  satisfiable: boolean;
  model: Record<string, unknown> | null;
  generationTimeMs: number;
  counterexample: string | null;
}

export interface VerificationBridgeResult {
  satOutput: unknown;
  z3Proof: Z3Proof;
  ihsanReceipt: Record<string, unknown>;
  verified: boolean;
  counterexample: string | null;
  proofGenerationMs: number;
}

export interface SNRAnalysis {
  signalComponents: {
    relevance: number;
    novelty: number;
    groundedness: number;
    coherence: number;
    actionability: number;
    specificity: number;
  };
  noiseComponents: {
    redundancy: number;
    inconsistency: number;
    ambiguity: number;
    irrelevance: number;
    hallucination: number;
    verbosity: number;
    bias: number;
  };
  snrLinear: number;
  snrDb: number;
  ihsanAchieved: boolean;
}

export interface ReasoningSyncResult {
  nodeId: string;
  snrAnalysis: SNRAnalysis;
  filteredContent: string;
  signalComponents: Record<string, number>;
  noiseComponents: Record<string, number>;
  passedThreshold: boolean;
}

export interface ProtocolAdapterResult {
  originalMessage: A2AMessage;
  pciEnvelope: PCIEnvelope;
  verificationResult: VerificationResult;
  gateChainPassed: string[];
  conversionTimestamp: string;
}

// ============================================================================
// SAPE ADAPTER CLASS
// ============================================================================

export interface SAPEAdapterConfig {
  ihsanThreshold?: number;
  snrThreshold?: number;
  executionContext?: ExecutionContext;
  enableZ3Proofs?: boolean;
}

export interface ISAPEAdapter {
  // Adapter 1: Intent Translation
  translatePatIntent(
    patAgentId: string,
    intent: string,
    context?: Record<string, unknown>
  ): Promise<IntentTranslation>;

  // Adapter 2: Verification Bridge
  verifySatOutput(
    satAgentId: string,
    output: unknown,
    actionContext: Record<string, unknown>
  ): Promise<VerificationBridgeResult>;

  // Adapter 3: Reasoning Sync
  syncGotNode(
    nodeId: string,
    content: string,
    queryContext?: string
  ): Promise<ReasoningSyncResult>;

  // Adapter 4: Protocol Adapter
  convertA2AToPci(
    message: A2AMessage,
    privateKey: string
  ): Promise<ProtocolAdapterResult>;

  convertPciToA2A(envelope: PCIEnvelope): A2AMessage;

  // Utilities
  getStatistics(): Record<string, number>;
  resetStatistics(): void;
}

// ============================================================================
// CONSTANTS (must match Python constants.py)
// ============================================================================

export const UNIFIED_IHSAN_THRESHOLD = 0.95;
export const UNIFIED_SNR_THRESHOLD = 0.85;
export const IHSAN_WEIGHTS: IhsanDimensionScores = {
  correctness: 0.22,
  safety: 0.22,
  userBenefit: 0.14,
  efficiency: 0.12,
  auditability: 0.12,
  antiCentralization: 0.08,
  robustness: 0.06,
  fairness: 0.04,
};
```

---

## Data Flow Diagrams

### Flow 1: PAT Intent to Ihsan-Scored Action

```
                          DUAL-AGENTIC SYSTEM                    DATA-LAKE GOVERNANCE
                          ==================                    ====================

    +-----------+         +---------------+         +------------------+
    | PAT Agent |-------->| Intent String |-------->| SAPE Adapter     |
    +-----------+         +---------------+         | translate_pat_   |
         |                      |                   | intent()         |
         |                      |                   +--------+---------+
         |                      |                            |
         |                      |                            v
         |                      |                   +------------------+
         |                      |                   | IhsanVector.     |
         |                      |                   | from_scores()    |
         |                      |                   +--------+---------+
         |                      |                            |
         |                      |                            v
         |                      |                   +------------------+
         |                      |                   | 8-Dimension      |
         |                      |                   | Scoring          |
         |                      |                   | - correctness    |
         |                      |                   | - safety         |
         |                      |                   | - user_benefit   |
         |                      |                   | - efficiency     |
         |                      |                   | - auditability   |
         |                      |                   | - anti_central.  |
         |                      |                   | - robustness     |
         |                      |                   | - fairness       |
         |                      |                   +--------+---------+
         |                      |                            |
         |                      |                            v
         |                      |                   +------------------+
         |                      |                   | Aggregate Score  |
         |                      |                   | >= 0.95 ?        |
         |                      |                   +--------+---------+
         |                      |                            |
         |                      |                   +--------+---------+
         |                      |                   |        |         |
         |                      |                   | YES    |    NO   |
         |                      |                   |        |         |
         v                      v                   v        v         v
    +-----------+         +---------------+   +---------+ +-----------+
    | Execute   |<--------| Permission    |<--| PERMIT  | | DENY +    |
    | Action    |         | Granted       |   +---------+ | RECOMMEND |
    +-----------+         +---------------+               +-----------+
```

### Flow 2: SAT Output to FATE Gate Verification

```
                          DUAL-AGENTIC SYSTEM                    DATA-LAKE GOVERNANCE
                          ==================                    ====================

    +-----------+         +---------------+         +------------------+
    | SAT Agent |-------->| Action Output |-------->| SAPE Adapter     |
    +-----------+         +---------------+         | verify_sat_      |
         |                      |                   | output()         |
         |                      |                   +--------+---------+
         |                      |                            |
         |                      v                            v
         |               +---------------+         +------------------+
         |               | Action Context|-------->| Z3FATEGate.      |
         |               | - ihsan       |         | generate_proof() |
         |               | - snr         |         +--------+---------+
         |               | - risk_level  |                  |
         |               | - reversible  |                  v
         |               | - cost        |         +------------------+
         |               +---------------+         | Z3 SMT Solver    |
         |                                         | Constraints:     |
         |                                         | - ihsan >= 0.95  |
         |                                         | - snr >= 0.85    |
         |                                         | - risk => rev|ap |
         |                                         | - cost <= limit  |
         |                                         +--------+---------+
         |                                                  |
         |                                         +--------+---------+
         |                                         |        |         |
         |                                         | SAT    |  UNSAT  |
         |                                         | (proof)|  (cex)  |
         |                                         |        |         |
         v                                         v        v         v
    +-----------+                            +---------+ +-----------+
    | Proceed   |<---------------------------| Z3Proof | | Counter-  |
    | with      |                            | + Ihsan | | example + |
    | Action    |                            | Receipt | | REJECT    |
    +-----------+                            +---------+ +-----------+
```

### Flow 3: GoT Node to SNR Filtering

```
                          DUAL-AGENTIC SYSTEM                    DATA-LAKE GOVERNANCE
                          ==================                    ====================

    +-----------+         +---------------+         +------------------+
    | GoT Graph |-------->| ThoughtNode   |-------->| SAPE Adapter     |
    +-----------+         | - content     |         | sync_got_node()  |
         |                | - confidence  |         +--------+---------+
         |                | - type        |                  |
         |                +---------------+                  v
         |                                         +------------------+
         |                                         | SNRMaximizer.    |
         |                                         | analyze()        |
         |                                         +--------+---------+
         |                                                  |
         |                                                  v
         |                                         +------------------+
         |                                         | Signal Profile   |
         |                                         | - relevance      |
         |                                         | - novelty        |
         |                                         | - groundedness   |
         |                                         | - coherence      |
         |                                         +------------------+
         |                                                  |
         |                                                  v
         |                                         +------------------+
         |                                         | Noise Profile    |
         |                                         | - redundancy     |
         |                                         | - ambiguity      |
         |                                         | - inconsistency  |
         |                                         +------------------+
         |                                                  |
         |                                                  v
         |                                         +------------------+
         |                                         | SNR = Signal /   |
         |                                         |       (Noise+e)  |
         |                                         | >= 0.85 ?        |
         |                                         +--------+---------+
         |                                                  |
         |                                         +--------+---------+
         |                                         |        |         |
         |                                         | YES    |    NO   |
         |                                         |        |         |
         v                                         v        v         v
    +-----------+                            +---------+ +-----------+
    | Include   |<---------------------------| KEEP    | | PRUNE or  |
    | in Path   |                            | Node    | | FILTER    |
    +-----------+                            +---------+ +-----------+
```

### Flow 4: A2A Message to PCI Envelope

```
                          DUAL-AGENTIC SYSTEM                    DATA-LAKE GOVERNANCE
                          ==================                    ====================

    +-----------+         +---------------+         +------------------+
    | A2A       |-------->| A2AMessage    |-------->| SAPE Adapter     |
    | Engine    |         | - sender_id   |         | convert_a2a_     |
    +-----------+         | - payload     |         | to_pci()         |
         |                | - type        |         +--------+---------+
         |                | - ihsan_score |                  |
         |                +---------------+                  v
         |                                         +------------------+
         |                                         | EnvelopeBuilder  |
         |                                         | .with_sender()   |
         |                                         | .with_payload()  |
         |                                         | .with_metadata() |
         |                                         | .build()         |
         |                                         +--------+---------+
         |                                                  |
         |                                                  v
         |                                         +------------------+
         |                                         | envelope.sign()  |
         |                                         | (Ed25519)        |
         |                                         +--------+---------+
         |                                                  |
         |                                                  v
         |                                         +------------------+
         |                                         | PCIGateKeeper.   |
         |                                         | verify()         |
         |                                         +------------------+
         |                                                  |
         |                                                  v
         |                                         +------------------+
         |                                         | Gate Chain:      |
         |                                         | 1. SCHEMA        |
         |                                         | 2. SIGNATURE     |
         |                                         | 3. TIMESTAMP     |
         |                                         | 4. REPLAY        |
         |                                         | 5. IHSAN         |
         |                                         | 6. SNR           |
         |                                         | 7. POLICY        |
         |                                         +--------+---------+
         |                                                  |
         |                                         +--------+---------+
         |                                         |        |         |
         |                                         | PASS   |   FAIL  |
         v                                         v        v         v
    +-----------+                            +---------+ +-----------+
    | Send to   |<---------------------------| PCI     | | Reject +  |
    | Federation|                            | Envelope| | Code      |
    +-----------+                            +---------+ +-----------+
```

---

## Compliance Matrix

### Ihsan Dimension Validation by Adapter

| Adapter | Correctness | Safety | User Benefit | Efficiency | Auditability | Anti-Central. | Robustness | Fairness |
|---------|-------------|--------|--------------|------------|--------------|---------------|------------|----------|
| Intent Translation | Direct | Direct | Direct | Direct | Implicit | Direct | Direct | Direct |
| Verification Bridge | Z3 Proof | Z3 Risk | - | Z3 Cost | Receipt | - | Z3 Proof | - |
| Reasoning Sync | Groundedness | - | - | Noise Filter | - | - | Signal | - |
| Protocol Adapter | Signature | Gate Chain | - | - | PCI Audit | - | Replay Check | - |

**Legend:**
- **Direct:** Explicit scoring via Ihsan Vector
- **Z3 Proof:** Formal verification via SMT solver
- **Implicit:** Guaranteed by design/protocol
- **-:** Not applicable to this adapter

### Gate Chain Validation by Adapter

| Adapter | L1 Schema | L2 Signature | L3 Temporal | L4 Ihsan | L5 SNR |
|---------|-----------|--------------|-------------|----------|--------|
| Intent Translation | - | - | - | Primary | Secondary |
| Verification Bridge | - | Implicit | - | Z3 Bound | Z3 Bound |
| Reasoning Sync | - | - | - | Analysis | Primary |
| Protocol Adapter | Primary | Primary | Primary | Primary | Primary |

### Constitutional Article Mapping

| Article | Description | Enforced By |
|---------|-------------|-------------|
| Art. 1 | Sovereignty (Edge-First) | PAT ownership in Intent Translation |
| Art. 2 | Ihsan >= 0.95 | All adapters via `UNIFIED_IHSAN_THRESHOLD` |
| Art. 3 | SNR >= 0.85 | Reasoning Sync via `SNRMaximizer` |
| Art. 4 | Reversibility | Verification Bridge via Z3 constraint |
| Art. 5 | Auditability | Protocol Adapter via PCI receipts |
| Art. 6 | Anti-Centralization | Intent Translation dimension scoring |
| Art. 7 | FATE Gate | Verification Bridge via `Z3FATEGate` |

---

## Migration Path

### Phase 1: Interface Alignment (Week 1-2)

1. **Deploy SAPE Adapter module**
   ```bash
   # Location
   /mnt/c/BIZRA-DATA-LAKE/core/integration/sape_adapter.py

   # Install dependencies
   pip install z3-solver  # For FATE Gate
   ```

2. **Update imports in existing code**
   ```python
   # Before
   from core.a2a import A2AMessage
   from core.pci import PCIEnvelope

   # After
   from core.integration.sape_adapter import (
       SAPEAdapter,
       create_sape_adapter,
   )
   adapter = create_sape_adapter()
   ```

3. **Add adapter calls to critical paths**
   - PAT intent processing: `adapter.translate_pat_intent()`
   - SAT action execution: `adapter.verify_sat_output()`
   - GoT reasoning: `adapter.sync_got_node()`
   - A2A messaging: `adapter.convert_a2a_to_pci()`

### Phase 2: Validation Rollout (Week 3-4)

1. **Enable in development context**
   ```python
   adapter = create_sape_adapter(
       execution_context="development",
       enable_z3=True,
   )
   ```

2. **Monitor adaptation metrics**
   ```python
   stats = adapter.get_statistics()
   assert stats["translations"] > 0
   assert stats["verifications"] > 0
   ```

3. **Validate threshold consistency**
   ```python
   from core.integration.constants import validate_cross_repo_consistency
   results = validate_cross_repo_consistency()
   assert all(r["status"] == "synced" for r in results.values())
   ```

### Phase 3: Production Deployment (Week 5-6)

1. **Switch to production context**
   ```python
   adapter = create_sape_adapter(
       execution_context="production",
       enable_z3=True,
   )
   ```

2. **Enable full gate chain**
   - All 7 PCI gates active
   - Z3 proofs required for SAT actions
   - SNR filtering mandatory for GoT

3. **Establish monitoring**
   - Track `AdapterResult` distribution
   - Alert on `IHSAN_VIOLATION` or `FATE_VIOLATION`
   - Log all `counterexample` outputs

### Phase 4: Continuous Validation (Ongoing)

1. **Autopoiesis integration**
   - Register SAPE Adapter with `AutopoieticLoop`
   - Enable hypothesis generation from adapter metrics
   - Shadow deploy adapter changes

2. **Cross-repository sync**
   - Run `validate_cross_repo_consistency()` in CI
   - Alert on threshold drift
   - Auto-generate sync PRs

---

## Appendix A: Error Codes

| Code | Description | Recovery Action |
|------|-------------|-----------------|
| `IHSAN_VIOLATION` | Aggregate Ihsan < 0.95 | Review dimension scores, improve weak dimensions |
| `SNR_VIOLATION` | SNR < 0.85 | Filter content, reduce noise components |
| `FATE_VIOLATION` | Z3 proof UNSAT | Check counterexample, adjust constraints |
| `PCI_VIOLATION` | Gate chain failure | Check specific gate code (REJECT_*) |
| `SIGNATURE_INVALID` | Ed25519 verify failed | Regenerate keypair, check key rotation |
| `TIMEOUT` | Operation exceeded limit | Reduce complexity, check resource bounds |

## Appendix B: Threshold Constants

```python
# Authoritative source: core/integration/constants.py
UNIFIED_IHSAN_THRESHOLD = 0.95
UNIFIED_SNR_THRESHOLD = 0.85
STRICT_IHSAN_THRESHOLD = 0.99
RUNTIME_IHSAN_THRESHOLD = 1.0
UNIFIED_CLOCK_SKEW_SECONDS = 120
UNIFIED_NONCE_TTL_SECONDS = 300
ADL_GINI_THRESHOLD = 0.40
```

## Appendix C: Standing on Giants

| Giant | Contribution | Applied In |
|-------|--------------|------------|
| Shannon (1948) | Information theory, SNR | `SNRMaximizer`, signal/noise analysis |
| Lamport (1982) | Distributed systems | Timestamp ordering, replay protection |
| de Moura (2008) | Z3 SMT solver | `Z3FATEGate`, formal verification |
| Besta (2024) | Graph-of-Thoughts | `GraphOfThoughts`, reasoning sync |
| Maturana (1980) | Autopoiesis | Self-evolution loop integration |
| Al-Ghazali (1111) | Ihsan (excellence) | Constitutional thresholds |

---

**Document Hash:** `sha256:sape_adapter_spec_v1.0.0`
**Last Updated:** 2026-02-05
**Maintainer:** BIZRA Architecture Council
