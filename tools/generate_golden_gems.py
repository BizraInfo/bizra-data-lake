#!/usr/bin/env python3
"""
🔍 BIZRA GOLDEN GEMS GENERATOR
Produces verifiable, reproducible pattern mining output.

Output: golden_gems_index.jsonl with full evidence trails
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib

def generate_gems():
    gold_path = Path('C:/BIZRA-DATA-LAKE/04_GOLD')
    indexed_path = Path('C:/BIZRA-DATA-LAKE/03_INDEXED')

    # Load actual data
    print("Loading data...")
    chunks_df = pd.read_parquet(gold_path / 'chunks.parquet')
    
    # Load nodes and edges from JSONL
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
    
    print(f"  Chunks: {len(chunks_df)}")
    print(f"  Nodes: {len(nodes_df)}")
    print(f"  Edges: {len(edges_df)}")

    gems = []

    # ============================================================
    # Pattern 1: High-connectivity nodes (bridging score)
    # Schema: edges have 'source', 'target', 'relation'
    # Schema: nodes have 'id', 'label', 'kind', 'source'
    # ============================================================
    print("\nAnalyzing hub nodes...")
    node_degree = edges_df.groupby('source').size().reset_index(name='out_degree')
    node_in_degree = edges_df.groupby('target').size().reset_index(name='in_degree')
    node_degree = node_degree.rename(columns={'source': 'node_id'})
    node_in_degree = node_in_degree.rename(columns={'target': 'node_id'})
    degree_merged = pd.merge(node_degree, node_in_degree, on='node_id', how='outer').fillna(0)
    degree_merged['total_degree'] = degree_merged['out_degree'] + degree_merged['in_degree']
    top_hubs = degree_merged.nlargest(10, 'total_degree')

    for _, row in top_hubs.iterrows():
        node_info = nodes_df[nodes_df['id'] == row['node_id']]
        label = "Unknown"
        if not node_info.empty and 'label' in node_info.columns:
            label = str(node_info.iloc[0]['label'])[:100]
        
        gems.append({
            'artifact_id': row['node_id'],
            'title': label,
            'type': 'hub_node',
            'score_connectivity': float(row['total_degree']),
            'score_signal': min(float(row['total_degree']) / 100, 1.0),
            'score_novelty': 0.5,  # Placeholder
            'evidence': {
                'out_degree': int(row['out_degree']),
                'in_degree': int(row['in_degree']),
                'total_degree': int(row['total_degree'])
            },
            'pattern': 'High-Connectivity Hub',
            'methodology': 'Computed from edges_df groupby source_id/target_id'
        })

    # ============================================================
    # Pattern 2: Most referenced documents (chunk density)
    # ============================================================
    print("Analyzing document density...")
    doc_counts = chunks_df['doc_id'].value_counts().head(10)
    
    for doc_id, count in doc_counts.items():
        sample_chunks = chunks_df[chunks_df['doc_id'] == doc_id].head(3)
        sample_texts = [c[:100] for c in sample_chunks['chunk_text'].tolist()]
        
        gems.append({
            'artifact_id': doc_id,
            'title': sample_texts[0][:80] + '...' if sample_texts else 'No text',
            'type': 'document',
            'score_connectivity': float(count),
            'score_signal': min(float(count) / 50, 1.0),
            'score_novelty': 0.7,
            'evidence': {
                'chunk_count': int(count),
                'sample_excerpts': sample_texts
            },
            'pattern': 'High-Chunk-Density Document',
            'methodology': 'Computed from chunks_df.doc_id.value_counts()'
        })

    # ============================================================
    # Pattern 3: Edge type distribution (using 'relation' field)
    # ============================================================
    print("Analyzing edge types...")
    if 'relation' in edges_df.columns:
        edge_types = edges_df['relation'].value_counts().head(10)
        for edge_type, count in edge_types.items():
            # Get sample edges
            sample_edges = edges_df[edges_df['relation'] == edge_type].head(3)
            sample_pairs = [(r['source'][:12], r['target'][:12]) for _, r in sample_edges.iterrows()]
            
            gems.append({
                'artifact_id': f'relation_{edge_type}',
                'title': str(edge_type),
                'type': 'relationship_type',
                'score_connectivity': float(count),
                'score_signal': min(float(count) / 1000, 1.0),
                'score_novelty': 0.6,
                'evidence': {
                    'edge_count': int(count),
                    'sample_pairs': sample_pairs
                },
                'pattern': 'Dominant Relationship Type',
                'methodology': 'Computed from edges_df.relation.value_counts()'
            })

    # ============================================================
    # Pattern 4: Cross-domain bridges (nodes connecting different kinds)
    # ============================================================
    print("Analyzing cross-domain bridges...")
    if 'kind' in nodes_df.columns:
        # Find nodes that connect to multiple node kinds
        edge_with_types = edges_df.merge(
            nodes_df[['id', 'kind']], 
            left_on='source', 
            right_on='id',
            how='left'
        ).rename(columns={'kind': 'source_kind'})
        
        edge_with_types = edge_with_types.merge(
            nodes_df[['id', 'kind']], 
            left_on='target', 
            right_on='id',
            how='left'
        ).rename(columns={'kind': 'target_kind'})
        
        # Cross-domain = source_kind != target_kind
        cross_domain = edge_with_types[edge_with_types['source_kind'] != edge_with_types['target_kind']]
        
        if len(cross_domain) > 0:
            # Count cross-domain bridges per source node
            bridge_counts = cross_domain.groupby('source').size().reset_index(name='bridge_count')
            top_bridges = bridge_counts.nlargest(5, 'bridge_count')
            
            for _, row in top_bridges.iterrows():
                node_info = nodes_df[nodes_df['id'] == row['source']]
                label = str(node_info.iloc[0]['label'])[:80] if not node_info.empty and 'label' in node_info.columns else row['source']
                
                gems.append({
                    'artifact_id': row['source'],
                    'title': label,
                    'type': 'cross_domain_bridge',
                    'score_connectivity': float(row['bridge_count']),
                    'score_signal': min(float(row['bridge_count']) / 20, 1.0),
                    'score_novelty': 0.9,  # High novelty for bridges
                    'evidence': {
                        'cross_domain_edge_count': int(row['bridge_count'])
                    },
                    'pattern': 'Cross-Domain Bridge Node',
                    'methodology': 'Edges where source_kind != target_kind'
                })

    # ============================================================
    # Save output with full provenance
    # ============================================================
    output_path = gold_path / 'golden_gems_index.jsonl'
    
    run_metadata = {
        'run_id': hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16],
        'generated_at': datetime.now().isoformat(),
        'corpus_id': 'BIZRA-DATA-LAKE-v1.0',
        'input_files': {
            'chunks': str(gold_path / 'chunks.parquet'),
            'nodes': str(indexed_path / 'graph' / 'nodes.parquet'),
            'edges': str(indexed_path / 'graph' / 'edges.parquet')
        },
        'counts': {
            'chunks': len(chunks_df),
            'nodes': len(nodes_df),
            'edges': len(edges_df)
        }
    }
    
    print(f"\nWriting {len(gems)} gems to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # First line: metadata
        f.write(json.dumps({'_metadata': run_metadata}) + '\n')
        
        # Gem entries
        for gem in gems:
            gem['run_id'] = run_metadata['run_id']
            f.write(json.dumps(gem, default=str) + '\n')
    
    # Print summary
    print("\n" + "="*60)
    print("GOLDEN GEMS SUMMARY")
    print("="*60)
    print(f"Total gems: {len(gems)}")
    print(f"Run ID: {run_metadata['run_id']}")
    print(f"\nBy pattern type:")
    
    from collections import Counter
    pattern_counts = Counter(g['pattern'] for g in gems)
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count}")
    
    print("\nTop 5 by connectivity score:")
    sorted_gems = sorted(gems, key=lambda x: x['score_connectivity'], reverse=True)[:5]
    for g in sorted_gems:
        print(f"  - [{g['type']}] {g['title'][:40]}... (score={g['score_connectivity']:.1f})")
    
    return gems, run_metadata


if __name__ == "__main__":
    generate_gems()
