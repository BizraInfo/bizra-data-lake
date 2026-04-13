"""
Canonical Loop Proof — The unified replayable proof-of-life artifact for Node0.

One mission → one gate chain → one verdict → one signed receipt lineage →
one manifest → one cockpit-visible proof-of-life story.

This is the closure point: it converts architecture into proof, proof into
product, and product into trust surface.

The loop proof bundles a complete governed execution cycle into a single
JSON artifact that can be:
  - replayed for audit
  - displayed in the Glass Cockpit
  - published as proof-of-life
  - verified by any party with the public key

Standing on Giants:
- Lamport (1978): Event ordering as proof structure
- Merkle (1979): Hash chains for tamper detection
- BIZRA NorthStar: "one mission, one front door, one gate chain, one receipt lineage"
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.proof_engine.evidence_audit import EvidenceAuditResult, audit_evidence
from core.proof_engine.fate_gate import FateResult, validate_with_evidence
from core.proof_engine.fate_telemetry import FateTelemetry
from core.proof_engine.model_routing import (
    PatRole,
    SatRole,
    get_pat_model,
    get_sat_model,
    routing_table_summary,
)
from core.proof_engine.sat_validator import SatVerdict, SimplePatOutput


@dataclass
class LoopStep:
    """A single step in the canonical loop proof."""

    seq: int
    timestamp: str
    actor: str
    action: str
    status: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""

    def compute_hash(self, prev_hash: str) -> str:
        payload = json.dumps(
            {
                "seq": self.seq,
                "actor": self.actor,
                "action": self.action,
                "status": self.status,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
        ).encode()
        self.hash = hashlib.blake2b(payload, digest_size=32).hexdigest()
        return self.hash


@dataclass
class LoopProof:
    """The canonical loop proof artifact — complete governed execution cycle."""

    version: str = "1.0"
    proof_class: str = "node0_loop_proof"
    canonical: bool = False  # True only after Ed25519 seal
    timestamp: str = ""
    node_id: str = "node0"
    mission: str = ""
    steps: List[LoopStep] = field(default_factory=list)
    fate_result: Dict[str, Any] = field(default_factory=dict)
    routing: Dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    genesis_hash: str = "0" * 64
    signature: str = ""  # Empty until Ed25519 seal

    def add_step(
        self, actor: str, action: str, status: str, evidence: Dict[str, Any] = None
    ) -> LoopStep:
        prev = self.steps[-1].hash if self.steps else self.genesis_hash
        step = LoopStep(
            seq=len(self.steps),
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            action=action,
            status=status,
            evidence=evidence or {},
        )
        step.compute_hash(prev)
        self.steps.append(step)
        return step

    def compute_manifest_hash(self) -> str:
        chain = json.dumps(
            {
                "version": self.version,
                "node_id": self.node_id,
                "mission": self.mission,
                "step_hashes": [s.hash for s in self.steps],
                "genesis_hash": self.genesis_hash,
            },
            sort_keys=True,
        ).encode()
        self.manifest_hash = hashlib.blake2b(chain, digest_size=32).hexdigest()
        return self.manifest_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "proof_class": self.proof_class,
            "canonical": self.canonical,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "mission": self.mission,
            "steps": [asdict(s) for s in self.steps],
            "fate_result": self.fate_result,
            "routing": self.routing,
            "manifest_hash": self.manifest_hash,
            "genesis_hash": self.genesis_hash,
            "signature": self.signature,
            "step_count": len(self.steps),
            "chain_valid": self.verify_chain(),
        }

    def verify_chain(self) -> bool:
        prev = self.genesis_hash
        for step in self.steps:
            expected = hashlib.blake2b(
                json.dumps(
                    {
                        "seq": step.seq,
                        "actor": step.actor,
                        "action": step.action,
                        "status": step.status,
                        "prev_hash": prev,
                    },
                    sort_keys=True,
                ).encode(),
                digest_size=32,
            ).hexdigest()
            if step.hash != expected:
                return False
            prev = step.hash
        return True

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def execute_loop_proof(
    mission: str,
    pat_answer: str,
    evidence_refs: List[str],
    confidence: str = "high",
    *,
    output_path: Optional[Path] = None,
) -> LoopProof:
    """Execute a complete canonical loop proof cycle.

    This is the single public entry point for producing a proof-of-life artifact.

    Args:
        mission: The mission/question being proved.
        pat_answer: The PAT agent's answer text.
        evidence_refs: List of evidence references (git-show:X, file:Y, etc.)
        confidence: PAT confidence level.
        output_path: Optional path to write the proof artifact.

    Returns:
        LoopProof — the complete, hash-chained, replayable proof artifact.
    """
    proof = LoopProof(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mission=mission,
        routing=routing_table_summary(),
    )

    # Step 0: Mission envelope
    proof.add_step(
        actor="mission_gate",
        action="mission_accepted",
        status="ok",
        evidence={"mission": mission, "timestamp": proof.timestamp},
    )

    # Step 1: PAT execution
    proof.add_step(
        actor=f"pat_researcher ({get_pat_model(PatRole.RESEARCHER)})",
        action="pat_execution",
        status="completed",
        evidence={
            "answer_length": len(pat_answer),
            "evidence_count": len(evidence_refs),
            "confidence": confidence,
        },
    )

    # Step 2: FATE crossing (evidence audit + SAT verdict)
    pat_output = SimplePatOutput(
        answer=pat_answer,
        evidence_refs=evidence_refs,
        confidence=confidence,
    )
    fate_result = validate_with_evidence(pat_output, emit_telemetry=True)

    proof.add_step(
        actor="evidence_auditor",
        action="evidence_audit",
        status="pass" if fate_result.evidence_audit.all_refs_valid else "fail",
        evidence={
            "valid_count": fate_result.evidence_audit.valid_count,
            "invalid_count": fate_result.evidence_audit.invalid_count,
            "invalid_refs": fate_result.evidence_audit.invalid_refs,
        },
    )

    proof.add_step(
        actor=f"sat_validator ({get_sat_model(SatRole.ORACLE_S)})",
        action="sat_verdict",
        status=fate_result.verdict.verdict.lower(),
        evidence={
            "verdict": fate_result.verdict.verdict,
            "ihsan_score": fate_result.verdict.ihsan_score,
            "reason": fate_result.verdict.reason,
            "evidence_sufficient": fate_result.verdict.evidence_sufficient,
            "short_circuited": fate_result.short_circuited,
        },
    )

    # Step 3: FATE result
    proof.add_step(
        actor="fate_gate",
        action="fate_crossing_complete",
        status="pass" if fate_result.passed else "blocked",
        evidence={
            "passed": fate_result.passed,
            "telemetry": fate_result.telemetry_summary,
        },
    )

    # Step 4: Receipt emission
    proof.fate_result = fate_result.to_dict()
    proof.compute_manifest_hash()

    proof.add_step(
        actor="loop_proof",
        action="manifest_sealed",
        status="unsigned",  # becomes "signed" after Ed25519 seal
        evidence={"manifest_hash": proof.manifest_hash, "step_count": len(proof.steps)},
    )

    # Write artifact
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(proof.to_json())

    return proof
