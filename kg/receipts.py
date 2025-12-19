"""
kg/receipts.py — Append-only audit receipts for Knowledge Substrate

Every operation (ingest, query, elevation, gate) MUST emit a receipt.
Receipts are:
- Append-only (enforced by database trigger)
- Signed (optional in dev, required in prod)
- Include policy hash, ihsan score, SAPE vector, SNR metrics

No evidence → no claim (Glass Box axiom).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

import psycopg


class ReceiptKind(str, Enum):
    """Types of receipts."""
    INGEST = "INGEST"
    QUERY = "QUERY"
    SAPE = "SAPE"
    ELEVATION = "ELEVATION"
    GATE = "GATE"
    SEED = "SEED"
    FEEDBACK = "FEEDBACK"


class Decision(str, Enum):
    """Possible decisions."""
    ALLOWED = "ALLOWED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


@dataclass
class RejectionReason:
    """Structured rejection reason."""
    code: str
    severity: str  # HIGH, MEDIUM, LOW
    message: str
    repair_hint: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    policy_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "repair_hint": self.repair_hint,
            "evidence_refs": self.evidence_refs,
            "policy_hash": self.policy_hash
        }


@dataclass
class IhsanScore:
    """Ihsan (excellence) gate result."""
    score: float  # 0.0 - 1.0
    tier: str  # GOLD, SILVER, BRONZE, UNRATED
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "tier": self.tier,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "notes": self.notes
        }


@dataclass
class SapeVector:
    """SAPE cycle tracking."""
    cycle_id: Optional[str] = None
    phase: Optional[str] = None  # symbolic, abstraction, probe, elevation
    stakes: str = "M"  # H, M, L
    lenses: List[str] = field(default_factory=list)
    probe_result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "phase": self.phase,
            "stakes": self.stakes,
            "lenses": self.lenses,
            "probe_result": self.probe_result
        }


@dataclass
class SnrMetrics:
    """Signal-to-Noise Ratio tracking."""
    budget: str = "default"  # budget tier
    input_tokens: int = 0
    output_tokens: int = 0
    ratio: Optional[float] = None  # actual SNR
    within_budget: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ratio": self.ratio,
            "within_budget": self.within_budget
        }


def sha256_json(obj: Any) -> str:
    """Generate SHA256 hash of JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_policy_hash() -> str:
    """Get current policy hash from environment or compute from files."""
    # In production, this would hash the actual policy files
    return os.environ.get("BIZRA_POLICY_HASH", "dev-policy-" + datetime.now(timezone.utc).strftime("%Y%m%d"))


def emit_receipt(
    conn: psycopg.Connection,
    kind: ReceiptKind,
    decision: Decision,
    evidence_refs: List[Dict[str, Any]],
    payload: Dict[str, Any],
    ihsan: Optional[IhsanScore] = None,
    sape: Optional[SapeVector] = None,
    snr: Optional[SnrMetrics] = None,
    rejection_reasons: Optional[List[RejectionReason]] = None,
    policy_hash: Optional[str] = None,
    signature: Optional[str] = None,
) -> str:
    """
    Emit an append-only receipt.
    
    Args:
        conn: Database connection
        kind: Type of operation
        decision: ALLOWED, REJECTED, or ESCALATED
        evidence_refs: List of evidence references
        payload: Operation-specific data
        ihsan: Ihsan gate result
        sape: SAPE cycle tracking
        snr: SNR metrics
        rejection_reasons: List of rejection reasons (if rejected)
        policy_hash: Hash of active policy
        signature: Cryptographic signature (required in prod)
    
    Returns:
        receipt_id (UUID string)
    """
    policy_hash = policy_hash or get_policy_hash()
    ihsan = ihsan or IhsanScore(score=1.0, tier="dev")
    sape = sape or SapeVector()
    snr = snr or SnrMetrics()
    rejection_reasons = rejection_reasons or []
    
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO kg_receipts
              (kind, policy_hash, decision, evidence_refs, payload, 
               ihsan, sape, snr, rejection_reasons, signature)
            VALUES
              (%s, %s, %s, %s::jsonb, %s::jsonb, 
               %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
            RETURNING receipt_id
            """,
            (
                kind.value,
                policy_hash,
                decision.value,
                json.dumps(evidence_refs),
                json.dumps(payload),
                json.dumps(ihsan.to_dict()),
                json.dumps(sape.to_dict()),
                json.dumps(snr.to_dict()),
                json.dumps([r.to_dict() for r in rejection_reasons]),
                signature,
            ),
        )
        rid = cur.fetchone()[0]
    
    conn.commit()
    return str(rid)


def emit_query_receipt(
    conn: psycopg.Connection,
    query: str,
    results: List[Dict[str, Any]],
    embedding_model: str,
    k: int,
    has_evidence: bool
) -> str:
    """
    Convenience function for query receipts.
    """
    decision = Decision.ALLOWED if has_evidence else Decision.REJECTED
    
    evidence_refs = [
        {"type": "chunk", "id": r.get("chunk_id")}
        for r in results
    ]
    
    rejection_reasons = []
    if not has_evidence:
        rejection_reasons.append(RejectionReason(
            code="INSUFFICIENT_EVIDENCE",
            severity="HIGH",
            message="No matching evidence found for query",
            repair_hint="Ingest more sources or adjust query terms"
        ))
    
    return emit_receipt(
        conn=conn,
        kind=ReceiptKind.QUERY,
        decision=decision,
        evidence_refs=evidence_refs,
        payload={
            "query": query,
            "k": k,
            "embedding_model": embedding_model,
            "result_count": len(results)
        },
        rejection_reasons=rejection_reasons
    )


def emit_ingest_receipt(
    conn: psycopg.Connection,
    source: str,
    doc_ids: List[str],
    chunk_count: int,
    entity_count: int
) -> str:
    """
    Convenience function for ingest receipts.
    """
    return emit_receipt(
        conn=conn,
        kind=ReceiptKind.INGEST,
        decision=Decision.ALLOWED,
        evidence_refs=[{"type": "document", "id": did} for did in doc_ids],
        payload={
            "source": source,
            "document_count": len(doc_ids),
            "chunk_count": chunk_count,
            "entity_count": entity_count
        }
    )


def get_receipt(conn: psycopg.Connection, receipt_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a receipt by ID.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                receipt_id, kind, created_at, policy_hash,
                ihsan, sape, snr, decision, rejection_reasons,
                evidence_refs, payload, signature
            FROM kg_receipts
            WHERE receipt_id = %s
            """,
            (receipt_id,),
        )
        row = cur.fetchone()
        
        if not row:
            return None
        
        return {
            "receipt_id": str(row[0]),
            "kind": row[1],
            "created_at": row[2].isoformat(),
            "policy_hash": row[3],
            "ihsan": row[4],
            "sape": row[5],
            "snr": row[6],
            "decision": row[7],
            "rejection_reasons": row[8],
            "evidence_refs": row[9],
            "payload": row[10],
            "signature": row[11]
        }


def list_recent_receipts(
    conn: psycopg.Connection,
    kind: Optional[ReceiptKind] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    List recent receipts, optionally filtered by kind.
    """
    with conn.cursor() as cur:
        if kind:
            cur.execute(
                """
                SELECT receipt_id, kind, created_at, decision, 
                       ihsan->>'tier' as ihsan_tier,
                       jsonb_array_length(evidence_refs) as evidence_count
                FROM kg_receipts
                WHERE kind = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (kind.value, limit),
            )
        else:
            cur.execute(
                """
                SELECT receipt_id, kind, created_at, decision,
                       ihsan->>'tier' as ihsan_tier,
                       jsonb_array_length(evidence_refs) as evidence_count
                FROM kg_receipts
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
        
        return [
            {
                "receipt_id": str(row[0]),
                "kind": row[1],
                "created_at": row[2].isoformat(),
                "decision": row[3],
                "ihsan_tier": row[4],
                "evidence_count": row[5]
            }
            for row in cur.fetchall()
        ]
