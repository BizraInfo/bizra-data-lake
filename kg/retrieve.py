"""
kg/retrieve.py — Hybrid retrieval: vector similarity + graph traversal

Implements:
1. Vector top-k from pgvector
2. Entity extraction from retrieved chunks
3. 1-3 hop graph expansion for evidence enrichment
4. Evidence bundling with provenance

All queries emit receipts (see kg/receipts.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psycopg


@dataclass
class RetrievalResult:
    """A single retrieval result with evidence."""
    chunk_id: str
    content: str
    distance: float
    doc_id: Optional[str] = None
    source: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalBundle:
    """Complete retrieval response with evidence and graph context."""
    results: List[RetrievalResult]
    entity_graph: Dict[str, Any]
    evidence_refs: List[Dict[str, Any]]
    query_embedding_model: str
    total_chunks_searched: int


def vector_search(
    conn: psycopg.Connection,
    query_vec: List[float],
    k: int = 8,
    min_similarity: float = 0.0
) -> List[Tuple[str, str, float, str, str, Dict]]:
    """
    Vector similarity search using pgvector.
    
    Returns: List of (chunk_id, content, distance, doc_id, source, provenance)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                c.chunk_id,
                c.content,
                (e.embedding <=> %s::vector) AS distance,
                c.doc_id,
                d.source,
                c.provenance
            FROM kg_embeddings e
            JOIN kg_chunks c ON c.chunk_id = e.chunk_id
            JOIN kg_documents d ON d.doc_id = c.doc_id
            WHERE (e.embedding <=> %s::vector) <= %s OR %s = 0.0
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, 1.0 - min_similarity, min_similarity, query_vec, k),
        )
        return cur.fetchall()


def get_chunk_entities(
    conn: psycopg.Connection,
    chunk_ids: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get entities mentioned in chunks.
    
    Returns: {chunk_id: [{entity_id, canonical, entity_type, confidence, role}]}
    """
    if not chunk_ids:
        return {}
    
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                m.chunk_id::text,
                e.entity_id::text,
                e.canonical,
                e.entity_type,
                m.confidence,
                m.role
            FROM kg_mentions m
            JOIN kg_entities e ON e.entity_id = m.entity_id
            WHERE m.chunk_id = ANY(%s::uuid[])
            ORDER BY m.confidence DESC
            """,
            (chunk_ids,),
        )
        
        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in cur.fetchall():
            chunk_id = row[0]
            if chunk_id not in result:
                result[chunk_id] = []
            result[chunk_id].append({
                "entity_id": row[1],
                "canonical": row[2],
                "entity_type": row[3],
                "confidence": row[4],
                "role": row[5]
            })
        
        return result


def expand_graph(
    conn: psycopg.Connection,
    entity_ids: List[str],
    max_hops: int = 1,
    edge_types: Optional[List[str]] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Expand graph from seed entities.
    
    Returns: {
        "nodes": [{entity_id, canonical, entity_type, weight}],
        "edges": [{src, dst, type, weight}]
    }
    """
    if not entity_ids:
        return {"nodes": [], "edges": []}
    
    with conn.cursor() as cur:
        # Get 1-hop neighbors
        if edge_types:
            cur.execute(
                """
                SELECT DISTINCT
                    e.src_entity_id::text,
                    e.dst_entity_id::text,
                    e.edge_type,
                    e.weight,
                    src_ent.canonical AS src_canonical,
                    dst_ent.canonical AS dst_canonical,
                    dst_ent.entity_type AS dst_type
                FROM kg_edges e
                JOIN kg_entities src_ent ON src_ent.entity_id = e.src_entity_id
                JOIN kg_entities dst_ent ON dst_ent.entity_id = e.dst_entity_id
                WHERE e.src_entity_id = ANY(%s::uuid[])
                  AND e.edge_type = ANY(%s)
                ORDER BY e.weight DESC
                LIMIT %s
                """,
                (entity_ids, edge_types, limit),
            )
        else:
            cur.execute(
                """
                SELECT DISTINCT
                    e.src_entity_id::text,
                    e.dst_entity_id::text,
                    e.edge_type,
                    e.weight,
                    src_ent.canonical AS src_canonical,
                    dst_ent.canonical AS dst_canonical,
                    dst_ent.entity_type AS dst_type
                FROM kg_edges e
                JOIN kg_entities src_ent ON src_ent.entity_id = e.src_entity_id
                JOIN kg_entities dst_ent ON dst_ent.entity_id = e.dst_entity_id
                WHERE e.src_entity_id = ANY(%s::uuid[])
                ORDER BY e.weight DESC
                LIMIT %s
                """,
                (entity_ids, limit),
            )
        
        rows = cur.fetchall()
        
        nodes_seen = set()
        nodes = []
        edges = []
        
        for row in rows:
            src_id, dst_id, edge_type, weight, src_canonical, dst_canonical, dst_type = row
            
            # Add destination node
            if dst_id not in nodes_seen:
                nodes_seen.add(dst_id)
                nodes.append({
                    "entity_id": dst_id,
                    "canonical": dst_canonical,
                    "entity_type": dst_type
                })
            
            edges.append({
                "src": src_id,
                "dst": dst_id,
                "type": edge_type,
                "weight": weight
            })
        
        return {"nodes": nodes, "edges": edges}


def retrieve(
    conn: psycopg.Connection,
    query_vec: List[float],
    k: int = 8,
    expand_hops: int = 1,
    edge_types: Optional[List[str]] = None,
    min_similarity: float = 0.0,
    embedding_model: str = "unknown"
) -> RetrievalBundle:
    """
    Hybrid retrieval: vector search + graph expansion.
    
    Args:
        conn: Database connection
        query_vec: Query embedding vector
        k: Number of top results
        expand_hops: How many graph hops to expand (0-3)
        edge_types: Filter to specific edge types
        min_similarity: Minimum cosine similarity (0-1)
        embedding_model: Name of embedding model used
    
    Returns:
        RetrievalBundle with results, graph context, and evidence refs
    """
    # 1. Vector search
    vector_results = vector_search(conn, query_vec, k, min_similarity)
    
    if not vector_results:
        return RetrievalBundle(
            results=[],
            entity_graph={"nodes": [], "edges": []},
            evidence_refs=[],
            query_embedding_model=embedding_model,
            total_chunks_searched=0
        )
    
    chunk_ids = [str(r[0]) for r in vector_results]
    
    # 2. Get entities from chunks
    chunk_entities = get_chunk_entities(conn, chunk_ids)
    
    # 3. Build results
    results = []
    all_entity_ids = set()
    
    for row in vector_results:
        chunk_id, content, distance, doc_id, source, provenance = row
        chunk_id_str = str(chunk_id)
        
        entities = chunk_entities.get(chunk_id_str, [])
        for e in entities:
            all_entity_ids.add(e["entity_id"])
        
        results.append(RetrievalResult(
            chunk_id=chunk_id_str,
            content=content,
            distance=float(distance),
            doc_id=str(doc_id) if doc_id else None,
            source=source,
            entities=entities,
            provenance=provenance if isinstance(provenance, dict) else {}
        ))
    
    # 4. Graph expansion
    entity_graph = {"nodes": [], "edges": []}
    if expand_hops > 0 and all_entity_ids:
        entity_graph = expand_graph(
            conn,
            list(all_entity_ids),
            max_hops=min(expand_hops, 3),
            edge_types=edge_types,
            limit=100
        )
    
    # 5. Build evidence refs
    evidence_refs = [
        {
            "type": "chunk",
            "id": r.chunk_id,
            "source": r.source,
            "distance": r.distance
        }
        for r in results
    ]
    
    # Get total chunks for stats
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM kg_chunks")
        total_chunks = cur.fetchone()[0]
    
    return RetrievalBundle(
        results=results,
        entity_graph=entity_graph,
        evidence_refs=evidence_refs,
        query_embedding_model=embedding_model,
        total_chunks_searched=total_chunks
    )


def search_by_entity(
    conn: psycopg.Connection,
    entity_name: str,
    k: int = 10
) -> List[RetrievalResult]:
    """
    Search chunks by entity mention (text-based, no embedding required).
    
    Useful for:
    - Exact concept lookup
    - Graph-first retrieval
    - When embeddings aren't available
    """
    with conn.cursor() as cur:
        # Find entity
        cur.execute(
            """
            SELECT entity_id FROM kg_entities 
            WHERE LOWER(canonical) = LOWER(%s)
               OR LOWER(%s) = ANY(SELECT LOWER(unnest(aliases)))
            LIMIT 1
            """,
            (entity_name, entity_name),
        )
        row = cur.fetchone()
        
        if not row:
            return []
        
        entity_id = row[0]
        
        # Get chunks mentioning this entity
        cur.execute(
            """
            SELECT 
                c.chunk_id::text,
                c.content,
                m.confidence,
                c.doc_id::text,
                d.source,
                c.provenance
            FROM kg_mentions m
            JOIN kg_chunks c ON c.chunk_id = m.chunk_id
            JOIN kg_documents d ON d.doc_id = c.doc_id
            WHERE m.entity_id = %s
            ORDER BY m.confidence DESC
            LIMIT %s
            """,
            (entity_id, k),
        )
        
        results = []
        for row in cur.fetchall():
            results.append(RetrievalResult(
                chunk_id=row[0],
                content=row[1],
                distance=1.0 - row[2],  # Convert confidence to distance-like
                doc_id=row[3],
                source=row[4],
                provenance=row[5] if isinstance(row[5], dict) else {}
            ))
        
        return results
