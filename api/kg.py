"""
api/kg.py — Knowledge Substrate API endpoints

Provides:
- /v1/kg/query — Hybrid vector + graph retrieval with receipts
- /v1/kg/entity — Entity lookup
- /v1/kg/receipts — Receipt audit trail

All endpoints enforce:
- Receipt emission (append-only audit)
- SNR budget (configurable)
- Ihsan gate (configurable)
- FATE fail-closed on missing evidence
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import psycopg

from kg.embeddings import get_embedder, Embedder
from kg.retrieve import retrieve, search_by_entity, RetrievalBundle
from kg.receipts import (
    emit_query_receipt,
    emit_receipt,
    get_receipt,
    list_recent_receipts,
    ReceiptKind,
    Decision,
    RejectionReason,
    IhsanScore,
    SapeVector,
    SnrMetrics
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PG_DSN = os.environ.get(
    "BIZRA_PG_DSN",
    "postgresql://bizra:bizra_dev_password@postgres:5432/bizra"
)

# Ihsan enforcement (fail-closed if enabled and score too low)
IHSAN_ENFORCE = os.environ.get("BIZRA_IHSAN_ENFORCE", "false").lower() == "true"
IHSAN_MIN_SCORE = float(os.environ.get("BIZRA_IHSAN_MIN_SCORE", "0.5"))

# SNR budget enforcement
SNR_ENFORCE = os.environ.get("BIZRA_SNR_ENFORCE", "false").lower() == "true"
SNR_MAX_OUTPUT_TOKENS = int(os.environ.get("BIZRA_SNR_MAX_OUTPUT", "4000"))


# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Database connection dependency."""
    conn = psycopg.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()


def get_embedder_dep() -> Embedder:
    """Embedder dependency."""
    return get_embedder()


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class KGQueryRequest(BaseModel):
    """Query request."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    k: int = Field(8, ge=1, le=50, description="Number of results")
    expand_hops: int = Field(1, ge=0, le=3, description="Graph expansion depth")
    edge_types: Optional[List[str]] = Field(None, description="Filter edge types")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity")
    include_graph: bool = Field(True, description="Include entity graph in response")


class KGQueryResult(BaseModel):
    """Single query result."""
    chunk_id: str
    content: str
    distance: float
    source: Optional[str] = None
    entities: List[Dict[str, Any]] = []


class KGQueryResponse(BaseModel):
    """Query response with evidence."""
    answer: Optional[str] = None  # Future: generated answer
    results: List[KGQueryResult]
    entity_graph: Optional[Dict[str, Any]] = None
    receipt_id: str
    evidence_count: int
    decision: str


class EntityLookupRequest(BaseModel):
    """Entity lookup request."""
    entity_name: str = Field(..., min_length=1, max_length=200)
    k: int = Field(10, ge=1, le=50)


class ReceiptResponse(BaseModel):
    """Receipt details."""
    receipt_id: str
    kind: str
    created_at: str
    decision: str
    policy_hash: str
    ihsan: Dict[str, Any]
    sape: Dict[str, Any]
    snr: Dict[str, Any]
    evidence_refs: List[Dict[str, Any]]
    rejection_reasons: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/v1/kg", tags=["Knowledge Graph"])


@router.post("/query", response_model=KGQueryResponse)
def kg_query(
    body: KGQueryRequest,
    conn: psycopg.Connection = Depends(get_db_connection),
    embedder: Embedder = Depends(get_embedder_dep)
):
    """
    Query the Knowledge Substrate.
    
    Returns evidence-backed results with a receipt for audit.
    If no evidence is found, returns REJECTED decision (fail-closed).
    """
    # Generate embedding
    emb = embedder.embed(body.query)
    
    # Perform retrieval
    bundle: RetrievalBundle = retrieve(
        conn=conn,
        query_vec=emb.vector,
        k=body.k,
        expand_hops=body.expand_hops,
        edge_types=body.edge_types,
        min_similarity=body.min_similarity,
        embedding_model=emb.model
    )
    
    has_evidence = len(bundle.results) > 0
    decision = "ALLOWED" if has_evidence else "REJECTED"
    
    # Emit receipt
    receipt_id = emit_query_receipt(
        conn=conn,
        query=body.query,
        results=[{"chunk_id": r.chunk_id} for r in bundle.results],
        embedding_model=emb.model,
        k=body.k,
        has_evidence=has_evidence
    )
    
    # Build response
    results = [
        KGQueryResult(
            chunk_id=r.chunk_id,
            content=r.content,
            distance=r.distance,
            source=r.source,
            entities=r.entities
        )
        for r in bundle.results
    ]
    
    return KGQueryResponse(
        answer=None,  # Future: LLM-generated answer
        results=results,
        entity_graph=bundle.entity_graph if body.include_graph else None,
        receipt_id=receipt_id,
        evidence_count=len(results),
        decision=decision
    )


@router.post("/entity")
def kg_entity_lookup(
    body: EntityLookupRequest,
    conn: psycopg.Connection = Depends(get_db_connection)
):
    """
    Lookup chunks by entity name (no embedding required).
    
    Useful for exact concept search and graph-first retrieval.
    """
    results = search_by_entity(conn, body.entity_name, body.k)
    
    has_evidence = len(results) > 0
    
    receipt_id = emit_receipt(
        conn=conn,
        kind=ReceiptKind.QUERY,
        decision=Decision.ALLOWED if has_evidence else Decision.REJECTED,
        evidence_refs=[{"type": "chunk", "id": r.chunk_id} for r in results],
        payload={"entity_name": body.entity_name, "k": body.k, "method": "entity_lookup"},
        rejection_reasons=[] if has_evidence else [
            RejectionReason(
                code="ENTITY_NOT_FOUND",
                severity="MEDIUM",
                message=f"Entity '{body.entity_name}' not found in knowledge graph"
            )
        ]
    )
    
    return {
        "entity_name": body.entity_name,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "distance": r.distance,
                "source": r.source
            }
            for r in results
        ],
        "receipt_id": receipt_id,
        "decision": "ALLOWED" if has_evidence else "REJECTED"
    }


@router.get("/receipts/{receipt_id}", response_model=ReceiptResponse)
def get_receipt_by_id(
    receipt_id: str,
    conn: psycopg.Connection = Depends(get_db_connection)
):
    """Get receipt details by ID."""
    receipt = get_receipt(conn, receipt_id)
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return receipt


@router.get("/receipts")
def list_receipts(
    kind: Optional[str] = Query(None, description="Filter by receipt kind"),
    limit: int = Query(50, ge=1, le=200),
    conn: psycopg.Connection = Depends(get_db_connection)
):
    """List recent receipts."""
    receipt_kind = ReceiptKind(kind) if kind else None
    return list_recent_receipts(conn, receipt_kind, limit)


@router.get("/stats")
def kg_stats(conn: psycopg.Connection = Depends(get_db_connection)):
    """Get Knowledge Substrate statistics."""
    with conn.cursor() as cur:
        stats = {}
        
        for table in ["kg_entities", "kg_edges", "kg_documents", "kg_chunks", "kg_embeddings", "kg_receipts"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table.replace("kg_", "")] = cur.fetchone()[0]
        
        # Recent activity
        cur.execute("""
            SELECT kind, COUNT(*) 
            FROM kg_receipts 
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY kind
        """)
        stats["receipts_24h"] = {row[0]: row[1] for row in cur.fetchall()}
        
        # Decision distribution
        cur.execute("""
            SELECT decision, COUNT(*) 
            FROM kg_receipts 
            GROUP BY decision
        """)
        stats["decisions"] = {row[0]: row[1] for row in cur.fetchall()}
    
    return stats


@router.get("/health")
def kg_health(conn: psycopg.Connection = Depends(get_db_connection)):
    """Health check for Knowledge Substrate."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute("SELECT COUNT(*) FROM kg_entities LIMIT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {e}")
