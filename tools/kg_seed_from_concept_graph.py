#!/usr/bin/env python3
"""
kg_seed_from_concept_graph.py — Import extracted entities + edges into Knowledge Substrate

Imports from:
  - combined_concept_graph.json (nodes)
  - combined_concept_graph_edges.csv (edges)
  - evidence/bizra_hypergraph_unified/entities.json (unified entities)
  - evidence/bizra_hypergraph_unified/hyperedges.json (hyperedges)

Usage:
    export BIZRA_PG_DSN="postgresql://bizra:bizra_dev_password@localhost:5432/bizra"
    python tools/kg_seed_from_concept_graph.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg
except ImportError:
    print("❌ psycopg not installed. Run: pip install psycopg[binary]")
    sys.exit(1)


PG_DSN = os.environ.get(
    "BIZRA_PG_DSN",
    "postgresql://bizra:bizra_dev_password@localhost:5432/bizra"
)


def sha256_text(s: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file, return empty list if not found."""
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]


def seed_from_concept_graph(conn: psycopg.Connection, nodes_path: Path, edges_path: Path) -> Dict[str, int]:
    """Import nodes and edges from concept graph files."""
    stats = {"entities": 0, "edges": 0, "skipped": 0}
    
    with conn.cursor() as cur:
        # Load and insert nodes
        graph_data = load_json_safe(nodes_path)
        if graph_data:
            nodes = graph_data.get("nodes", [])
            print(f"📊 Found {len(nodes)} nodes in concept graph")
            
            for n in nodes:
                canonical = (n.get("label") or n.get("id") or "").strip()
                if not canonical:
                    stats["skipped"] += 1
                    continue
                
                weight = float(n.get("weight") or n.get("value") or 0.0)
                entity_type = n.get("type", "CONCEPT").upper()
                
                try:
                    cur.execute(
                        """
                        INSERT INTO kg_entities (canonical, entity_type, weight, metadata)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (canonical) DO UPDATE SET 
                            weight = GREATEST(kg_entities.weight, EXCLUDED.weight),
                            metadata = kg_entities.metadata || EXCLUDED.metadata
                        """,
                        (
                            canonical,
                            entity_type,
                            weight,
                            json.dumps({"source": "concept_graph", "original": n})
                        ),
                    )
                    stats["entities"] += 1
                except Exception as e:
                    print(f"⚠️ Failed to insert entity '{canonical}': {e}")
                    stats["skipped"] += 1
        
        # Build canonical -> entity_id mapping
        cur.execute("SELECT entity_id, canonical FROM kg_entities")
        id_by_canonical = {c.lower(): eid for (eid, c) in cur.fetchall()}
        
        # Load and insert edges
        if edges_path.exists():
            with open(edges_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = (row.get("source") or "").strip().lower()
                    dst = (row.get("target") or "").strip().lower()
                    w = float(row.get("weight") or 1.0)
                    edge_type = (row.get("type") or "CO_OCCURS").upper()
                    
                    if src not in id_by_canonical or dst not in id_by_canonical:
                        stats["skipped"] += 1
                        continue
                    
                    if src == dst:  # Skip self-loops
                        stats["skipped"] += 1
                        continue
                    
                    try:
                        cur.execute(
                            """
                            INSERT INTO kg_edges (src_entity_id, dst_entity_id, edge_type, weight, metadata)
                            VALUES (%s, %s, %s, %s, %s::jsonb)
                            """,
                            (
                                id_by_canonical[src],
                                id_by_canonical[dst],
                                edge_type,
                                w,
                                json.dumps({"source": "concept_graph"})
                            ),
                        )
                        stats["edges"] += 1
                    except Exception as e:
                        print(f"⚠️ Failed to insert edge {src}->{dst}: {e}")
                        stats["skipped"] += 1
        
        conn.commit()
    
    return stats


def seed_from_unified_hypergraph(conn: psycopg.Connection, graph_dir: Path) -> Dict[str, int]:
    """Import from unified HyperGraphRAG extraction."""
    stats = {"entities": 0, "documents": 0, "chunks": 0, "skipped": 0}
    
    entities_path = graph_dir / "entities.json"
    hyperedges_path = graph_dir / "hyperedges.json"
    
    with conn.cursor() as cur:
        # Load entities
        entities = load_json_safe(entities_path)
        if entities and isinstance(entities, list):
            print(f"📊 Found {len(entities)} entities in unified hypergraph")
            
            for e in entities:
                canonical = (e.get("name") or e.get("label") or "").strip()
                if not canonical:
                    stats["skipped"] += 1
                    continue
                
                entity_type = (e.get("type") or "CONCEPT").upper()
                weight = float(e.get("weight") or e.get("frequency") or 0.0)
                aliases = e.get("aliases", [])
                
                try:
                    cur.execute(
                        """
                        INSERT INTO kg_entities (canonical, entity_type, weight, aliases, metadata)
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (canonical) DO UPDATE SET 
                            weight = GREATEST(kg_entities.weight, EXCLUDED.weight),
                            aliases = array_cat(kg_entities.aliases, EXCLUDED.aliases),
                            metadata = kg_entities.metadata || EXCLUDED.metadata
                        """,
                        (
                            canonical,
                            entity_type,
                            weight,
                            aliases,
                            json.dumps({"source": "unified_hypergraph", "original_type": e.get("type")})
                        ),
                    )
                    stats["entities"] += 1
                except Exception as e:
                    print(f"⚠️ Failed to insert entity '{canonical}': {e}")
                    stats["skipped"] += 1
        
        # Load hyperedges as chunks (they represent knowledge segments)
        hyperedges = load_json_safe(hyperedges_path)
        if hyperedges and isinstance(hyperedges, list):
            print(f"📊 Found {len(hyperedges)} hyperedges in unified hypergraph")
            
            # Create a synthetic document to hold all hyperedges
            doc_content = json.dumps(hyperedges, indent=2)
            doc_hash = sha256_text(doc_content)
            
            cur.execute(
                """
                INSERT INTO kg_documents (source, source_ref, sha256, text, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (sha256) DO NOTHING
                RETURNING doc_id
                """,
                (
                    "unified_hypergraph",
                    str(hyperedges_path),
                    doc_hash,
                    doc_content[:100000],  # Truncate if very large
                    json.dumps({"hyperedge_count": len(hyperedges)})
                ),
            )
            result = cur.fetchone()
            if result:
                doc_id = result[0]
                stats["documents"] += 1
                
                # Insert each hyperedge as a chunk
                for i, he in enumerate(hyperedges[:5000]):  # Limit for initial seed
                    segment = he.get("knowledge_segment", "")[:5000]
                    if not segment:
                        continue
                    
                    tags = he.get("entities", [])[:20]  # Limit tags
                    if isinstance(tags, list) and tags and isinstance(tags[0], dict):
                        tags = [t.get("name", "") for t in tags if t.get("name")]
                    
                    try:
                        cur.execute(
                            """
                            INSERT INTO kg_chunks (doc_id, span_start, span_end, content, tags, provenance)
                            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                            """,
                            (
                                doc_id,
                                i * 1000,  # Synthetic span
                                i * 1000 + len(segment),
                                segment,
                                tags[:10],
                                json.dumps({
                                    "source": "unified_hypergraph",
                                    "hyperedge_id": he.get("id", i),
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                })
                            ),
                        )
                        stats["chunks"] += 1
                    except Exception as e:
                        stats["skipped"] += 1
        
        conn.commit()
    
    return stats


def print_summary(conn: psycopg.Connection) -> None:
    """Print current database statistics."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM kg_entities")
        entities = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM kg_edges")
        edges = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM kg_documents")
        docs = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM kg_chunks")
        chunks = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM kg_receipts")
        receipts = cur.fetchone()[0]
    
    print("\n" + "=" * 60)
    print("📊 KNOWLEDGE SUBSTRATE STATUS")
    print("=" * 60)
    print(f"   Entities:  {entities:,}")
    print(f"   Edges:     {edges:,}")
    print(f"   Documents: {docs:,}")
    print(f"   Chunks:    {chunks:,}")
    print(f"   Receipts:  {receipts:,}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed Knowledge Substrate from extracted graphs"
    )
    parser.add_argument(
        "--concept-nodes",
        type=Path,
        default=Path("combined_concept_graph.json"),
        help="Path to concept graph nodes JSON"
    )
    parser.add_argument(
        "--concept-edges",
        type=Path,
        default=Path("combined_concept_graph_edges.csv"),
        help="Path to concept graph edges CSV"
    )
    parser.add_argument(
        "--unified-graph",
        type=Path,
        default=Path("evidence/bizra_hypergraph_unified"),
        help="Path to unified HyperGraphRAG directory"
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=PG_DSN,
        help="PostgreSQL connection string"
    )
    
    args = parser.parse_args()
    
    print("🚀 BIZRA Knowledge Substrate Seeder")
    print(f"   DSN: {args.dsn.split('@')[1] if '@' in args.dsn else args.dsn}")
    
    try:
        with psycopg.connect(args.dsn) as conn:
            conn.execute("SET statement_timeout = '120s'")
            
            # Seed from concept graph if available
            if args.concept_nodes.exists():
                print(f"\n📥 Importing from concept graph...")
                stats = seed_from_concept_graph(conn, args.concept_nodes, args.concept_edges)
                print(f"   ✅ Entities: {stats['entities']}, Edges: {stats['edges']}, Skipped: {stats['skipped']}")
            
            # Seed from unified hypergraph if available
            if args.unified_graph.exists():
                print(f"\n📥 Importing from unified HyperGraphRAG...")
                stats = seed_from_unified_hypergraph(conn, args.unified_graph)
                print(f"   ✅ Entities: {stats['entities']}, Documents: {stats['documents']}, Chunks: {stats['chunks']}")
            
            # Print summary
            print_summary(conn)
            
    except psycopg.OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        print("   Hint: Is Postgres running? Try: docker compose up -d postgres")
        return 1
    except Exception as e:
        print(f"❌ Seeding failed: {e}")
        return 1
    
    print("\n✅ Seeding complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
