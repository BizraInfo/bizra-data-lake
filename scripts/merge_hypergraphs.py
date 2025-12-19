#!/usr/bin/env python3
"""
BIZRA HyperGraph Merger
Merges multiple hypergraph extraction results into a unified knowledge base.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def merge_entities(entity_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple entity dictionaries."""
    merged = {}
    
    for entities in entity_dicts:
        for name, data in entities.items():
            if name not in merged:
                merged[name] = {
                    "type": data.get("type", "UNKNOWN"),
                    "key_score": data.get("key_score", 50),
                    "sources": [],
                    "count": 0
                }
            
            # Update with higher score
            if data.get("key_score", 0) > merged[name]["key_score"]:
                merged[name]["key_score"] = data["key_score"]
            
            # Merge sources (deduplicated)
            for source in data.get("sources", []):
                if source not in merged[name]["sources"]:
                    merged[name]["sources"].append(source)
            
            # Sum counts
            merged[name]["count"] += data.get("count", 1)
    
    return merged


def merge_hyperedges(edge_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge multiple hyperedge lists, deduplicating by content hash."""
    seen_hashes = set()
    merged = []
    
    for edges in edge_lists:
        for edge in edges:
            # Create a content hash for deduplication
            content = edge.get("knowledge_segment", edge.get("segment", edge.get("content", "")))
            source = edge.get("source", "")
            
            # Use combination of source and content start for uniqueness
            if len(content) > 100:
                content_hash = f"{source}:{content[:100]}"
            else:
                content_hash = f"{source}:{content}"
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                merged.append(edge)
    
    return merged


def load_hypergraph_outputs(dir_path: Path) -> Dict[str, Any]:
    """Load hypergraph outputs from a directory."""
    result = {
        "entities": {},
        "hyperedges": [],
        "stats": {}
    }
    
    entities_path = dir_path / "entities.json"
    if entities_path.exists():
        with open(entities_path, 'r', encoding='utf-8') as f:
            result["entities"] = json.load(f)
    
    hyperedges_path = dir_path / "hyperedges.json"
    if hyperedges_path.exists():
        with open(hyperedges_path, 'r', encoding='utf-8') as f:
            result["hyperedges"] = json.load(f)
    
    stats_path = dir_path / "stats.json"
    if stats_path.exists():
        with open(stats_path, 'r', encoding='utf-8') as f:
            result["stats"] = json.load(f)
    
    return result


def generate_cypher(entities: Dict[str, Any], hyperedges: List[Dict[str, Any]], output_path: Path):
    """Generate Neo4j Cypher import script."""
    cypher_lines = [
        "// BIZRA Unified HyperGraph Import",
        f"// Generated: {datetime.now().isoformat()}",
        f"// Entities: {len(entities)}, HyperEdges: {len(hyperedges)}",
        "",
        "// === Create Entity Nodes ===",
    ]
    
    for name, data in entities.items():
        etype = data.get("type", "Entity")
        score = data.get("key_score", 50)
        count = data.get("count", 1)
        
        # Escape single quotes
        safe_name = name.replace("'", "\\'")
        
        cypher_lines.append(
            f"MERGE (e:{etype} {{name: '{safe_name}'}}) "
            f"SET e.key_score = {score}, e.occurrence_count = {count};"
        )
    
    cypher_lines.append("")
    cypher_lines.append("// === Create HyperEdge Nodes and Relationships ===")
    
    # Create hyperedge nodes (limit to first 1000 for Neo4j import)
    for i, edge in enumerate(hyperedges[:1000]):
        segment = edge.get("knowledge_segment", edge.get("segment", edge.get("content", "")))[:500]
        safe_segment = segment.replace("'", "\\'").replace("\n", " ").replace("\\", "\\\\")
        score = edge.get("completeness_score", 5.0)
        entities_in_edge = edge.get("entities", [])
        
        cypher_lines.append(
            f"CREATE (he{i}:HyperEdge {{id: 'he_{i}', segment: '{safe_segment}', completeness: {score}}});"
        )
        
        # Link to entities
        for ent_name in entities_in_edge[:10]:  # Limit connections
            safe_ent = ent_name.replace("'", "\\'")
            cypher_lines.append(
                f"MATCH (e {{name: '{safe_ent}'}}), (he:HyperEdge {{id: 'he_{i}'}}) "
                f"MERGE (e)-[:CONNECTED_TO]->(he);"
            )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cypher_lines))


def main():
    # Find workspace root
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    evidence_dir = workspace / "evidence"
    
    # Find all hypergraph output directories
    hypergraph_dirs = [
        evidence_dir / "bizra_hypergraph_offline",  # From .md files
        evidence_dir / "bizra_hypergraph_json",     # From JSON exports
    ]
    
    print("="*60)
    print("BIZRA HyperGraph Merger")
    print("="*60)
    
    all_entities = []
    all_hyperedges = []
    total_docs = 0
    sources_summary = []
    
    for hg_dir in hypergraph_dirs:
        if hg_dir.exists():
            print(f"\n📂 Loading: {hg_dir.name}")
            data = load_hypergraph_outputs(hg_dir)
            
            entity_count = len(data["entities"])
            edge_count = len(data["hyperedges"])
            doc_count = data["stats"].get("documents", 0)
            
            print(f"   🧬 Entities: {entity_count}")
            print(f"   🔗 HyperEdges: {edge_count}")
            print(f"   📄 Documents: {doc_count}")
            
            all_entities.append(data["entities"])
            all_hyperedges.append(data["hyperedges"])
            total_docs += doc_count
            sources_summary.append({
                "source": hg_dir.name,
                "entities": entity_count,
                "hyperedges": edge_count,
                "documents": doc_count
            })
        else:
            print(f"\n⚠️ Not found: {hg_dir.name}")
    
    # Merge
    print("\n🔄 Merging knowledge bases...")
    merged_entities = merge_entities(all_entities)
    merged_hyperedges = merge_hyperedges(all_hyperedges)
    
    print(f"\n{'='*60}")
    print("📊 UNIFIED KNOWLEDGE BASE")
    print(f"{'='*60}")
    print(f"🧬 Total Entities: {len(merged_entities)}")
    print(f"🔗 Total HyperEdges: {len(merged_hyperedges)}")
    print(f"📄 Total Documents: {total_docs}")
    
    # Entity breakdown
    type_counts = {}
    for name, data in merged_entities.items():
        etype = data.get("type", "UNKNOWN")
        type_counts[etype] = type_counts.get(etype, 0) + 1
    
    print("\n📋 Entity Types:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {etype}: {count}")
    
    # Save merged output
    output_dir = evidence_dir / "bizra_hypergraph_unified"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "entities.json", 'w', encoding='utf-8') as f:
        json.dump(merged_entities, f, indent=2)
    
    with open(output_dir / "hyperedges.json", 'w', encoding='utf-8') as f:
        json.dump(merged_hyperedges, f, indent=2)
    
    # Generate unified stats
    unified_stats = {
        "merged_at": datetime.now().isoformat(),
        "total_entities": len(merged_entities),
        "total_hyperedges": len(merged_hyperedges),
        "total_documents": total_docs,
        "entity_types": type_counts,
        "sources": sources_summary
    }
    
    with open(output_dir / "unified_stats.json", 'w', encoding='utf-8') as f:
        json.dump(unified_stats, f, indent=2)
    
    # Generate Cypher
    cypher_path = output_dir / "unified_hypergraph.cypher"
    generate_cypher(merged_entities, merged_hyperedges, cypher_path)
    
    print(f"\n💾 Saved to: {output_dir}")
    print(f"   - entities.json")
    print(f"   - hyperedges.json")
    print(f"   - unified_stats.json")
    print(f"   - unified_hypergraph.cypher")
    
    # Top entities by score
    print("\n🏆 Top Entities by Importance:")
    sorted_entities = sorted(
        merged_entities.items(),
        key=lambda x: (x[1].get("key_score", 0), x[1].get("count", 0)),
        reverse=True
    )
    for name, data in sorted_entities[:15]:
        print(f"   [{data['type']}] {name}: score={data['key_score']}, count={data['count']}")
    
    return unified_stats


if __name__ == '__main__':
    main()
