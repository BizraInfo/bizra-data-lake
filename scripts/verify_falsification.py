#!/usr/bin/env python3
"""
🔬 FALSIFICATION VERIFICATION SCRIPT
Empirically validates the claims in FALSIFICATION_REPORT.md

Output: Console summary + updated falsification evidence
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity ratio."""
    return SequenceMatcher(None, a[:5000], b[:5000]).ratio()

def main():
    gold_path = Path('C:/BIZRA-DATA-LAKE/04_GOLD')
    indexed_path = Path('C:/BIZRA-DATA-LAKE/03_INDEXED')
    
    print("=" * 60)
    print("🔬 FALSIFICATION VERIFICATION")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    chunks_df = pd.read_parquet(gold_path / 'chunks.parquet')
    
    nodes = []
    with open(indexed_path / 'graph' / 'nodes.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            nodes.append(json.loads(line))
    nodes_df = pd.DataFrame(nodes)
    
    edges = []
    with open(indexed_path / 'graph' / 'edges.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            edges.append(json.loads(line))
    edges_df = pd.DataFrame(edges)
    
    print(f"  Loaded: {len(chunks_df)} chunks, {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    # ============================================================
    # Test 1: Duplicate Detection for Top Hubs
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Duplicate Detection")
    print("=" * 60)
    
    top_docs = [
        ('f3be7323ff3ccde0', 'Validate_BIZRA_Manifest'),
        ('515c366c28ae08d3', 'Validate_BIZRA_Manifest.md'),
        ('57b08166b6f3dd7f', 'Zenith_performance_vision'),
        ('ebfb0fe907e79f0e', 'Zenith_performance_vision.md'),
    ]
    
    # Get sample text from each
    doc_samples = {}
    for doc_id, name in top_docs:
        doc_chunks = chunks_df[chunks_df['doc_id'] == doc_id]['chunk_text'].head(10).tolist()
        if doc_chunks:
            doc_samples[doc_id] = ' '.join(doc_chunks)
    
    print("\nPairwise similarity (duplicates if > 0.8):")
    pairs_checked = []
    for i, (id1, name1) in enumerate(top_docs):
        for id2, name2 in top_docs[i+1:]:
            if id1 in doc_samples and id2 in doc_samples:
                sim = text_similarity(doc_samples[id1], doc_samples[id2])
                status = "⚠️ DUPLICATE" if sim > 0.8 else "✅ Distinct"
                print(f"  {name1[:30]:30} vs {name2[:30]:30}: {sim:.2f} {status}")
                pairs_checked.append((id1, id2, sim))
    
    # ============================================================
    # Test 2: Degree = Chunk Count Verification
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Degree = Chunk Count (Structural Inflation)")
    print("=" * 60)
    
    test_docs = ['f3be7323ff3ccde0', '57b08166b6f3dd7f', '6a1115d6042804f4']
    
    for doc_id in test_docs:
        chunk_count = len(chunks_df[chunks_df['doc_id'] == doc_id])
        in_degree = len(edges_df[edges_df['target'] == doc_id])
        out_degree = len(edges_df[edges_df['source'] == doc_id])
        
        inflation = "⚠️ INFLATED" if abs(chunk_count - in_degree) < 50 else "✅ Semantic"
        print(f"  {doc_id[:16]}: chunks={chunk_count}, in_degree={in_degree}, out={out_degree} {inflation}")
    
    # ============================================================
    # Test 3: PART_OF Dominance
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: PART_OF Edge Dominance")
    print("=" * 60)
    
    if 'relation' in edges_df.columns:
        rel_counts = edges_df['relation'].value_counts()
        total = len(edges_df)
        
        for rel, count in rel_counts.items():
            pct = count / total * 100
            status = "⚠️ NOISE" if pct > 90 else "✅ Signal"
            print(f"  {rel:25}: {count:6} ({pct:5.1f}%) {status}")
        
        # Semantic-only edge count
        semantic_edges = len(edges_df[edges_df['relation'] != 'PART_OF'])
        print(f"\n  Semantic edges (excl PART_OF): {semantic_edges}")
        print(f"  Graph density without PART_OF: {semantic_edges / len(nodes_df):.3f} edges/node")
    
    # ============================================================
    # Test 4: Entity Self-Reference Check
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Entity Self-Reference (Corpus Name = Top Entity)")
    print("=" * 60)
    
    entity_nodes = nodes_df[nodes_df['id'].str.startswith('ent::')]
    print(f"  Total entity nodes: {len(entity_nodes)}")
    
    # Get entity in-degrees
    entity_degrees = edges_df[edges_df['target'].str.startswith('ent::')]['target'].value_counts()
    print("\n  Top 5 entities by in-degree:")
    for ent, deg in entity_degrees.head(5).items():
        name = ent.replace('ent::', '')
        self_ref = "⚠️ SELF-REF" if 'bizra' in name.lower() else "✅ Discovery"
        print(f"    {name:20}: {deg:5} {self_ref}")
    
    # ============================================================
    # Test 5: Cross-Domain Edge Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 5: Cross-Domain Bridge Validity")
    print("=" * 60)
    
    if 'kind' in nodes_df.columns:
        unique_kinds = nodes_df['kind'].unique()
        print(f"  Unique 'kind' values: {len(unique_kinds)}")
        for k in unique_kinds[:10]:
            count = len(nodes_df[nodes_df['kind'] == k])
            print(f"    {k}: {count}")
        
        if len(unique_kinds) <= 2:
            print("  ⚠️ TRIVIAL TAXONOMY: Only 2 kinds means cross-domain is by definition")
    else:
        print("  ⚠️ No 'kind' column - cross-domain pattern unverifiable")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("📋 FALSIFICATION SUMMARY")
    print("=" * 60)
    
    issues = []
    
    # Check for duplicates
    for id1, id2, sim in pairs_checked:
        if sim > 0.8:
            issues.append(f"Duplicate detected: {id1[:8]} ↔ {id2[:8]} (sim={sim:.2f})")
    
    # Check for self-reference
    if 'ent::bizra' in entity_degrees.index:
        issues.append("Self-reference: ent::bizra is top entity in BIZRA corpus")
    
    # Check for PART_OF dominance
    if 'relation' in edges_df.columns:
        part_of_pct = edges_df['relation'].value_counts().get('PART_OF', 0) / len(edges_df) * 100
        if part_of_pct > 80:
            issues.append(f"Structural noise: PART_OF is {part_of_pct:.1f}% of edges")
    
    if issues:
        print("\n⚠️ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ No major falsification issues detected")
    
    print("\n" + "=" * 60)
    print("✅ Falsification verification complete")
    print("=" * 60)

if __name__ == '__main__':
    main()
