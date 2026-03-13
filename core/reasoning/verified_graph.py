"""
Verified Reasoning Graph -- proof-bearing Graph-of-Thoughts wrapper.

Builds a deterministic proof artifact over an existing Graph-of-Thoughts run.
The graph artifact remains the source of truth; this module adds:

1. Branch certificates for each thought node
2. A deterministic VRG root over the surviving proof set
3. A signed receipt whose payload digest binds to the VRG payload
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.proof_engine.canonical import CanonPolicy, CanonQuery, canonical_bytes
from core.proof_engine.receipt import Metrics, Receipt, ReceiptBuilder, SovereignSigner
from core.sovereign.graph_types import ThoughtNode, ThoughtType

if TYPE_CHECKING:
    from core.memory.types import SearchResult

    from .got_bridge import GoTBridgeResult


@dataclass(frozen=True)
class PCICertificate:
    depth: int
    gate_passed: str
    ihsan: float
    included_in_root: bool
    node_hash: str
    node_id: str
    parent_hashes: list[str]
    reject_reason: str | None
    snr: float
    terminal: bool
    thought_type: str
    certificate_hash: str
    pruned_children: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "gate_passed": self.gate_passed,
            "ihsan": self.ihsan,
            "included_in_root": self.included_in_root,
            "node_hash": self.node_hash,
            "node_id": self.node_id,
            "parent_hashes": self.parent_hashes,
            "reject_reason": self.reject_reason,
            "snr": self.snr,
            "terminal": self.terminal,
            "thought_type": self.thought_type,
            "certificate_hash": self.certificate_hash,
            "pruned_children": self.pruned_children,
        }


@dataclass(frozen=True)
class GoTNode:
    thought: ThoughtNode
    certificate: PCICertificate

    def to_dict(self) -> dict[str, Any]:
        return {
            "thought": self.thought.to_dict(),
            "certificate": self.certificate.to_dict(),
        }


_SURVIVING_THOUGHT_TYPES: frozenset[ThoughtType] = frozenset(
    {
        ThoughtType.HYPOTHESIS,
        ThoughtType.REASONING,
        ThoughtType.SYNTHESIS,
        ThoughtType.REFINEMENT,
        ThoughtType.CONCLUSION,
        ThoughtType.COUNTERPOINT,
    }
)


@dataclass(frozen=True)
class VerifiedGoTBridgeResult:
    """Proof-bearing additive result for GoTBridge.reason_verified()."""

    base_result: GoTBridgeResult
    graph_artifact: dict[str, Any]
    branch_certificates: list[dict[str, Any]]
    vrg_root: str
    receipt: Receipt
    verified: bool
    got_nodes: list[GoTNode] = field(default_factory=list)


class VerifiedReasoningGraphBuilder:
    """Build a deterministic VRG artifact and signed receipt from a GoT run."""

    def __init__(
        self,
        signer: SovereignSigner,
        convergence_snr: float,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        policy_version: str = "1.0.0",
    ) -> None:
        self._receipt_builder = ReceiptBuilder(signer)
        self._convergence_snr = convergence_snr
        self._ihsan_threshold = ihsan_threshold
        self._policy_version = policy_version

    def build(
        self,
        query: str,
        context: dict[str, Any],
        evidence: list[SearchResult],
        base_result: GoTBridgeResult,
        got_engine: Any,
    ) -> VerifiedGoTBridgeResult:
        """Build the VRG artifact, root, and signed receipt."""
        canon_query = self._build_query(query, context, evidence)
        policy = self._build_policy()
        graph_artifact = self._build_graph_artifact(
            got_engine, canon_query.hex_digest()
        )
        got_nodes = self._build_branch_certificates(got_engine)
        branch_certificates = [n.certificate.to_dict() for n in got_nodes]
        vrg_root = self._compute_vrg_root(
            graph_artifact,
            got_nodes,
            canon_query.hex_digest(),
            policy.version,
        )

        payload = {
            "graph_artifact": graph_artifact,
            "branch_certificates": branch_certificates,
            "query_digest": canon_query.hex_digest(),
            "verified": False,
            "vrg_root": vrg_root,
        }

        verified, gate_name, reason, receipt_snr, receipt_ihsan = self._decision(
            base_result, got_nodes
        )
        payload["verified"] = verified
        payload_bytes = canonical_bytes(payload)

        if verified:
            receipt = self._receipt_builder.accepted(
                query=canon_query,
                policy=policy,
                payload=payload_bytes,
                snr=receipt_snr,
                ihsan_score=receipt_ihsan,
                gate_passed="vrg",
                metrics=Metrics(),
            )
        else:
            receipt = self._receipt_builder.rejected(
                query=canon_query,
                policy=policy,
                snr=receipt_snr,
                ihsan_score=receipt_ihsan,
                gate_failed=gate_name,
                reason=reason,
                payload=payload_bytes,
                metrics=Metrics(),
            )

        return VerifiedGoTBridgeResult(
            base_result=base_result,
            graph_artifact=graph_artifact,
            branch_certificates=branch_certificates,
            vrg_root=vrg_root,
            receipt=receipt,
            verified=verified,
            got_nodes=got_nodes,
        )

    def _build_query(
        self,
        query: str,
        context: dict[str, Any],
        evidence: list[SearchResult],
    ) -> CanonQuery:
        return CanonQuery(
            user_id="got_bridge",
            user_state=str(context.get("domain", "reasoning")),
            intent="reason_verified",
            payload={
                "context": context,
                "evidence": self._compact_evidence(evidence),
                "query": query,
            },
        )

    def _build_policy(self) -> CanonPolicy:
        return CanonPolicy(
            policy_id="verified_reasoning_graph",
            version=self._policy_version,
            rules={
                "surviving_thought_types": sorted(
                    thought_type.value for thought_type in _SURVIVING_THOUGHT_TYPES
                ),
                "vrg_root_inputs": [
                    "graph_hash",
                    "certificate_hashes",
                    "query_digest",
                    "policy_version",
                ],
            },
            thresholds={
                "ihsan": self._ihsan_threshold,
                "snr": self._convergence_snr,
            },
            constraints=[
                "rejected_nodes_remain_auditable",
                "surviving_nodes_only_contribute_to_root",
            ],
        )

    @staticmethod
    def _compact_evidence(evidence: list[SearchResult]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in evidence:
            compact.append(
                {
                    "content_preview": item.record.content[:120],
                    "score": round(item.score, 6),
                    "source": item.record.source,
                    "source_id": item.record.source_id,
                }
            )
        return compact

    def _build_graph_artifact(
        self,
        got_engine: Any,
        query_digest: str,
    ) -> dict[str, Any]:
        build_id = query_digest[:16]
        if callable(getattr(got_engine, "to_artifact", None)):
            try:
                artifact = got_engine.to_artifact(
                    build_id=build_id,
                    policy_version=self._policy_version,
                )
                if isinstance(artifact, dict):
                    return artifact
            except Exception:
                pass

        empty_graph_bytes = canonical_bytes(
            {
                "build_id": build_id,
                "nodes": [],
                "policy_version": self._policy_version,
                "roots": [],
            }
        )
        empty_graph_hash = hashlib.blake2b(
            empty_graph_bytes, digest_size=32
        ).hexdigest()
        return {
            "build_id": build_id,
            "config": {
                "ihsan_threshold": self._ihsan_threshold,
                "snr_threshold": self._convergence_snr,
            },
            "edges": [],
            "graph_hash": empty_graph_hash,
            "nodes": [],
            "policy_version": self._policy_version,
            "roots": [],
            "stats": {},
        }

    def _build_branch_certificates(self, got_engine: Any) -> list[GoTNode]:
        nodes = getattr(got_engine, "nodes", None)
        if not isinstance(nodes, dict) or not nodes:
            return []

        reverse_adj = getattr(got_engine, "reverse_adj", {})
        adjacency = getattr(got_engine, "adjacency", {})
        ordered_node_ids = sorted(
            nodes, key=lambda node_id: (nodes[node_id].depth, node_id)
        )
        certificates_by_id: dict[str, PCICertificate] = {}
        got_nodes: list[GoTNode] = []

        for node_id in ordered_node_ids:
            node = nodes[node_id]
            parent_ids = sorted(reverse_adj.get(node_id, []))
            candidate_parent_ids = [
                parent_id
                for parent_id in parent_ids
                if parent_id in nodes
                and nodes[parent_id].thought_type in _SURVIVING_THOUGHT_TYPES
            ]
            parent_hashes = [
                nodes[parent_id].content_hash
                for parent_id in parent_ids
                if parent_id in nodes
            ]
            terminal = not any(
                child_id in nodes
                and nodes[child_id].thought_type in _SURVIVING_THOUGHT_TYPES
                for child_id in adjacency.get(node_id, [])
            )

            gate_passed = "PASS"
            included_in_root = True
            reject_reason: str | None = None
            pruned_children = 0

            if "gate_failed" in node.metadata:
                gate_passed = node.metadata["gate_failed"]
                included_in_root = False
                reject_reason = node.metadata.get(
                    "reject_reason", "Gate failed dynamically"
                )
                pruned_children = node.metadata.get("pruned_children", 0)
            elif node.thought_type not in _SURVIVING_THOUGHT_TYPES:
                gate_passed = "INFO_ONLY"
                included_in_root = False
                reject_reason = "non_surviving_thought_type"
            elif any(
                not certificates_by_id[parent_id].included_in_root
                for parent_id in candidate_parent_ids
            ):
                gate_passed = "REJECT_PARENT_CHAIN"
                included_in_root = False
                reject_reason = "ancestor_not_included"
            elif node.snr_score < self._convergence_snr:
                gate_passed = "REJECT_SNR"
                included_in_root = False
                reject_reason = (
                    f"snr {node.snr_score:.3f} < {self._convergence_snr:.3f}"
                )
            elif node.ihsan_score < self._ihsan_threshold:
                gate_passed = "REJECT_IHSAN"
                included_in_root = False
                reject_reason = (
                    f"ihsan {node.ihsan_score:.3f} < {self._ihsan_threshold:.3f}"
                )

            cert_data = {
                "depth": node.depth,
                "gate_passed": gate_passed,
                "ihsan": round(node.ihsan_score, 6),
                "included_in_root": included_in_root,
                "node_hash": node.content_hash,
                "node_id": node.id,
                "parent_hashes": parent_hashes,
                "reject_reason": reject_reason,
                "snr": round(node.snr_score, 6),
                "terminal": terminal,
                "thought_type": node.thought_type.value,
                "pruned_children": pruned_children,
            }
            canonical_cert = canonical_bytes(cert_data)
            cert_hash = hashlib.blake2b(canonical_cert, digest_size=32).hexdigest()

            certificate = PCICertificate(**cert_data, certificate_hash=cert_hash)
            certificates_by_id[node_id] = certificate
            got_nodes.append(GoTNode(thought=node, certificate=certificate))

        return got_nodes

    @staticmethod
    def _compute_vrg_root(
        graph_artifact: dict[str, Any],
        got_nodes: list[GoTNode],
        query_digest: str,
        policy_version: str,
    ) -> str:
        surviving_hashes = sorted(
            node.certificate.certificate_hash
            for node in got_nodes
            if node.certificate.included_in_root
        )
        canonical = canonical_bytes(
            {
                "certificate_hashes": surviving_hashes,
                "graph_hash": graph_artifact.get("graph_hash", ""),
                "policy_version": policy_version,
                "query_digest": query_digest,
            }
        )
        return hashlib.blake2b(canonical, digest_size=32).hexdigest()

    def _decision(
        self,
        base_result: GoTBridgeResult,
        got_nodes: list[GoTNode],
    ) -> tuple[bool, str, str, float, float]:
        terminal_candidates = [
            node.certificate
            for node in got_nodes
            if node.certificate.thought_type
            in {thought_type.value for thought_type in _SURVIVING_THOUGHT_TYPES}
            and node.certificate.terminal
        ]
        surviving_terminals = [
            cert for cert in terminal_candidates if cert.included_in_root
        ]

        best_snr = base_result.snr_score
        best_ihsan = 0.0
        if terminal_candidates:
            best_candidate = max(
                terminal_candidates,
                key=lambda cert: (cert.snr, cert.ihsan),
            )
            best_snr = float(best_candidate.snr)
            best_ihsan = float(best_candidate.ihsan)

        if not got_nodes:
            return False, "graph", "Graph artifact unavailable", best_snr, best_ihsan

        if not base_result.converged:
            return (
                False,
                "convergence",
                f"SNR {base_result.snr_score:.3f} < {self._convergence_snr:.3f}",
                best_snr,
                best_ihsan,
            )

        if surviving_terminals:
            accepted_terminal = max(
                surviving_terminals,
                key=lambda cert: (cert.snr, cert.ihsan),
            )
            return (
                True,
                "vrg",
                "",
                float(accepted_terminal.snr),
                float(accepted_terminal.ihsan),
            )

        if terminal_candidates:
            rejected_terminal = max(
                terminal_candidates,
                key=lambda cert: (cert.snr, cert.ihsan),
            )
            return (
                False,
                str(rejected_terminal.gate_passed).lower(),
                str(
                    rejected_terminal.reject_reason
                    or "No terminal branch survived VRG gating"
                ),
                float(rejected_terminal.snr),
                float(rejected_terminal.ihsan),
            )

        return (
            False,
            "vrg",
            "No terminal branch survived VRG gating",
            best_snr,
            best_ihsan,
        )


__all__ = ["VerifiedGoTBridgeResult", "VerifiedReasoningGraphBuilder"]
