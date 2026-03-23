"""Tests for the proof-bearing verified reasoning graph path."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

from core.proof_engine.canonical import blake3_digest, canonical_bytes, hex_digest
from core.proof_engine.receipt import ReceiptStatus, ReceiptVerifier, SimpleSigner
from core.reasoning.got_bridge import GoTBridge
from core.sovereign.graph_types import EdgeType, ThoughtEdge, ThoughtNode, ThoughtType


class FakeGraphEngine:
    """Deterministic graph engine for VRG tests."""

    def __init__(self, terminal_ihsan: float = 0.97, terminal_snr: float = 0.96):
        self.nodes: dict[str, ThoughtNode] = {}
        self.edges: list[ThoughtEdge] = []
        self.adjacency: dict[str, list[str]] = {}
        self.reverse_adj: dict[str, list[str]] = {}
        self.roots: list[str] = []
        self.stats = {
            "aggregations": 1,
            "edges_created": 3,
            "nodes_created": 4,
            "nodes_pruned": 0,
            "refinements": 0,
        }
        self._build_graph(terminal_ihsan=terminal_ihsan, terminal_snr=terminal_snr)

    def _build_graph(self, terminal_ihsan: float, terminal_snr: float) -> None:
        question = ThoughtNode(
            id="question_1",
            content="What is BIZRA?",
            thought_type=ThoughtType.QUESTION,
            confidence=0.90,
            snr_score=0.90,
            depth=0,
            truthfulness=0.90,
            dignity=0.90,
            fairness=0.90,
            excellence=0.90,
            sustainability=0.90,
            correctness=0.90,
            groundedness=0.95,
            coherence=0.95,
        )
        hypothesis = ThoughtNode(
            id="hypothesis_1",
            content="BIZRA is a sovereign reasoning organism.",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.98,
            snr_score=0.97,
            depth=1,
            truthfulness=0.98,
            dignity=0.98,
            fairness=0.98,
            excellence=0.98,
            sustainability=0.98,
            correctness=0.98,
            groundedness=0.98,
            coherence=0.97,
        )
        synthesis = ThoughtNode(
            id="synthesis_1",
            content="The system combines constitutional governance with reasoning.",
            thought_type=ThoughtType.SYNTHESIS,
            confidence=0.98,
            snr_score=0.97,
            depth=2,
            truthfulness=0.98,
            dignity=0.98,
            fairness=0.98,
            excellence=0.98,
            sustainability=0.98,
            correctness=0.98,
            groundedness=0.97,
            coherence=0.98,
        )
        conclusion = ThoughtNode(
            id="conclusion_1",
            content="BIZRA is a sovereign reasoning organism with constitutional guardrails.",
            thought_type=ThoughtType.CONCLUSION,
            confidence=terminal_ihsan,
            snr_score=terminal_snr,
            depth=3,
            truthfulness=terminal_ihsan,
            dignity=terminal_ihsan,
            fairness=terminal_ihsan,
            excellence=terminal_ihsan,
            sustainability=terminal_ihsan,
            correctness=terminal_ihsan,
            groundedness=terminal_ihsan,
            coherence=terminal_ihsan,
        )

        for node in (question, hypothesis, synthesis, conclusion):
            self.nodes[node.id] = node
            self.adjacency[node.id] = []
            self.reverse_adj[node.id] = []

        self.roots.append(question.id)
        self._link(question.id, hypothesis.id)
        self._link(hypothesis.id, synthesis.id)
        self._link(synthesis.id, conclusion.id)

    def _link(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DERIVES,
    ) -> None:
        self.edges.append(
            ThoughtEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
            )
        )
        self.adjacency[source_id].append(target_id)
        self.reverse_adj[target_id].append(source_id)

    async def reason(
        self,
        query: str,
        context: dict[str, Any],
        max_depth: int = 3,
    ) -> dict[str, Any]:
        return {
            "conclusion": self.nodes["conclusion_1"].content,
            "depth_reached": min(max_depth, 3),
            "graph_stats": self.stats,
            "ihsan_score": self.nodes["conclusion_1"].ihsan_score,
            "snr_score": self.nodes["conclusion_1"].snr_score,
            "thoughts": [
                "Hypothesis A is plausible",
                "Synthesizing strongest branch",
                "Conclusion reached",
            ],
        }

    def to_artifact(
        self, build_id: str = "", policy_version: str = "1.0.0"
    ) -> dict[str, Any]:
        nodes = [
            {
                "confidence": round(node.confidence, 6),
                "content": node.content,
                "content_hash": node.content_hash,
                "depth": node.depth,
                "id": node.id,
                "ihsan": round(node.ihsan_score, 6),
                "snr": round(node.snr_score, 6),
                "type": node.thought_type.value,
            }
            for node in self.nodes.values()
        ]
        edges = [edge.to_dict() for edge in self.edges]
        graph_hash = hex_digest(
            canonical_bytes(
                {
                    "edges": sorted(
                        edges, key=lambda edge: (edge["source"], edge["target"])
                    ),
                    "nodes": sorted(nodes, key=lambda node: node["id"]),
                    "roots": sorted(self.roots),
                }
            )
        )
        return {
            "build_id": build_id,
            "config": {
                "ihsan_threshold": 0.95,
                "max_depth": 3,
                "snr_threshold": 0.95,
            },
            "edges": edges,
            "graph_hash": graph_hash,
            "nodes": nodes,
            "policy_version": policy_version,
            "roots": list(self.roots),
            "stats": dict(self.stats),
        }


async def test_reason_verified_returns_additive_proof_metadata():
    signer = SimpleSigner(b"verified-graph-test-signer")
    bridge = GoTBridge(got_engine=FakeGraphEngine(), receipt_signer=signer)

    result = await bridge.reason_verified(
        "What is BIZRA?",
        context={"domain": "architecture"},
    )

    assert result.verified is True
    assert result.base_result.answer.startswith("BIZRA is a sovereign reasoning")
    assert result.graph_artifact["graph_hash"]
    assert result.branch_certificates
    assert result.vrg_root
    assert result.receipt.status == ReceiptStatus.ACCEPTED

    verifier = ReceiptVerifier(signer)
    valid, error = verifier.verify(result.receipt)
    assert valid is True
    assert error is None

    question_certificate = next(
        cert for cert in result.branch_certificates if cert["node_id"] == "question_1"
    )
    assert question_certificate["gate_passed"] == "INFO_ONLY"
    assert question_certificate["included_in_root"] is False


async def test_reason_verified_root_is_deterministic_for_same_graph():
    signer = SimpleSigner(b"verified-graph-test-signer")
    bridge = GoTBridge(got_engine=FakeGraphEngine(), receipt_signer=signer)

    result_a = await bridge.reason_verified(
        "What is BIZRA?",
        context={"domain": "architecture"},
    )
    result_b = await bridge.reason_verified(
        "What is BIZRA?",
        context={"domain": "architecture"},
    )

    assert result_a.vrg_root == result_b.vrg_root
    assert result_a.receipt.payload_digest == result_b.receipt.payload_digest


async def test_reason_verified_rejects_low_ihsan_terminal_branch():
    signer = SimpleSigner(b"verified-graph-test-signer")
    bridge = GoTBridge(
        got_engine=FakeGraphEngine(terminal_ihsan=0.80),
        receipt_signer=signer,
    )

    result = await bridge.reason_verified(
        "What is BIZRA?",
        context={"domain": "architecture"},
    )

    conclusion_certificate = next(
        cert for cert in result.branch_certificates if cert["node_id"] == "conclusion_1"
    )
    assert conclusion_certificate["gate_passed"] == "REJECT_IHSAN"
    assert conclusion_certificate["included_in_root"] is False
    assert "ihsan" in conclusion_certificate["reject_reason"]
    assert result.verified is False
    assert result.receipt.status == ReceiptStatus.REJECTED
    assert result.receipt.payload_digest != blake3_digest(b"")

    verifier = ReceiptVerifier(signer)
    valid, error = verifier.verify(result.receipt)
    assert valid is True
    assert error is None


async def test_reason_verified_precipitates_vrg_reflex_without_payload_warning(
    monkeypatch,
    caplog,
):
    signer = SimpleSigner(b"verified-graph-test-signer")
    bridge = GoTBridge(got_engine=FakeGraphEngine(), receipt_signer=signer)
    compiled: list[dict[str, Any]] = []

    class FakeReflexLedger:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity

        def compile_vrg_reflex(
            self,
            *,
            task_description: str,
            ihsan_score: float,
            timestamp_ns: int,
            vrg_root: str,
            branch_certificates: list[str],
        ) -> None:
            compiled.append(
                {
                    "task_description": task_description,
                    "ihsan_score": ihsan_score,
                    "timestamp_ns": timestamp_ns,
                    "vrg_root": vrg_root,
                    "branch_certificates": branch_certificates,
                }
            )

    monkeypatch.setitem(
        sys.modules,
        "bizra",
        types.SimpleNamespace(ReflexLedger=FakeReflexLedger),
    )

    with caplog.at_level(logging.WARNING):
        result = await bridge.reason_verified(
            "What is BIZRA?",
            context={"domain": "architecture"},
        )

    assert result.verified is True
    assert len(compiled) == 1
    assert compiled[0]["task_description"] == "What is BIZRA?"
    assert compiled[0]["ihsan_score"] == result.receipt.ihsan_score
    assert compiled[0]["vrg_root"] == result.vrg_root
    assert compiled[0]["branch_certificates"] == [
        certificate["certificate_hash"] for certificate in result.branch_certificates
    ]
    assert "Failed to precipitate VRG reflex" not in caplog.text
