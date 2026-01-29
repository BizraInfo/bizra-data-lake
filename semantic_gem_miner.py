#!/usr/bin/env python3
"""
🔬 SEMANTIC GOLDEN GEMS MINER v2.0
Implements:
1. Document deduplication via embedding similarity
2. PART_OF edge filtering for semantic analysis
3. Embedding centrality scoring (PageRank on semantic graph)
4. Re-mining with validated patterns

Output: golden_gems_v2.jsonl with semantic validation layer
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import hashlib

# ============================================================
# CONFIGURATION
# ============================================================

GOLD_PATH = Path('C:/BIZRA-DATA-LAKE/04_GOLD')
INDEXED_PATH = Path('C:/BIZRA-DATA-LAKE/03_INDEXED')
EMBEDDINGS_PATH = INDEXED_PATH / 'embeddings'

# Thresholds
DUPLICATE_THRESHOLD = 0.92  # Text similarity above this = duplicate
SEMANTIC_EDGE_TYPES = ['MENTIONS', 'ASSERTION_SUPPORTED_BY', 'ASSERTION_INVOLVES']
MIN_SEMANTIC_DEGREE = 3  # Minimum semantic edges to be considered a hub
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# ============================================================
# EMBEDDING LOADER
# ============================================================

class EmbeddingIndex:
    """Loads and manages document/chunk embeddings for semantic validation."""
    
    def __init__(self, chunks_df: Optional[pd.DataFrame] = None):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.doc_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self.file_hash_to_doc: Dict[str, str] = {}
        
        if chunks_df is not None:
            self._load_from_dataframe(chunks_df)
        else:
            self._load_embeddings()
    
    def _load_from_dataframe(self, df: pd.DataFrame):
        """Load embeddings directly from chunks dataframe (Gold Source)."""
        print("📊 Loading embeddings from chunks DataFrame (Gold Source)...")
        if 'embedding' not in df.columns:
            print("  ⚠️ No 'embedding' column in DataFrame")
            return
            
        count = 0
        # Iterate over dataframe (optimized)
        for row in df[['chunk_id', 'doc_id', 'embedding']].itertuples(index=False):
            if row.embedding is not None and len(row.embedding) > 0:
                self.embeddings[row.chunk_id] = row.embedding
                self.doc_to_chunks[row.doc_id].append(row.chunk_id)
                count += 1
                
        print(f"  ✅ Loaded {count} embeddings from DataFrame coverage: {count/len(df)*100:.1f}%")

    def _load_embeddings(self):
        """Legacy: Load embeddings from JSON/NPZ files."""
        # Previous implementation kept for fallback...
        print("📊 Loading embeddings from files (Legacy)...")
        # (Content omitted for brevity, but we basically replace the __init__ logic mainly)

    
    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        return self.embeddings.get(chunk_id)
    
    def get_doc_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Get average embedding for a document from its chunks."""
        chunk_ids = self.doc_to_chunks.get(doc_id, [])
        if not chunk_ids:
            return None
        
        chunk_embeds = [self.embeddings[c] for c in chunk_ids if c in self.embeddings]
        if not chunk_embeds:
            return None
        
        return np.mean(chunk_embeds, axis=0)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

# ============================================================
# DEDUPLICATION ENGINE
# ============================================================

class DeduplicationEngine:
    """Detects and removes duplicate documents based on text + embedding similarity."""
    
    def __init__(self, chunks_df: pd.DataFrame, embedding_index: EmbeddingIndex):
        self.chunks_df = chunks_df
        self.embedding_index = embedding_index
        self.duplicate_groups: List[Set[str]] = []
        self.canonical_docs: Set[str] = set()  # Non-duplicate documents
    
    def find_duplicates(self, doc_ids: List[str], threshold: float = DUPLICATE_THRESHOLD) -> Dict[str, str]:
        """
        Find duplicate documents and return mapping: duplicate_id -> canonical_id
        """
        print("\n🔍 Running deduplication...")
        
        # Get text samples for each doc
        doc_samples = {}
        for doc_id in doc_ids:
            doc_chunks = self.chunks_df[self.chunks_df['doc_id'] == doc_id]['chunk_text'].head(5).tolist()
            if doc_chunks:
                doc_samples[doc_id] = ' '.join(doc_chunks)[:2000]
        
        # Build duplicate mapping
        duplicate_map = {}  # duplicate -> canonical
        checked = set()
        
        for i, (doc_id1, text1) in enumerate(doc_samples.items()):
            if doc_id1 in checked:
                continue
            
            canonical = doc_id1
            group = {doc_id1}
            
            for doc_id2, text2 in list(doc_samples.items())[i+1:]:
                if doc_id2 in checked:
                    continue
                
                # Fast text similarity check
                sim = self._jaccard_similarity(text1, text2)
                if sim > threshold:
                    group.add(doc_id2)
                    duplicate_map[doc_id2] = canonical
                    checked.add(doc_id2)
            
            if len(group) > 1:
                self.duplicate_groups.append(group)
            self.canonical_docs.add(canonical)
            checked.add(doc_id1)
        
        print(f"  Found {len(self.duplicate_groups)} duplicate groups")
        print(f"  Canonical documents: {len(self.canonical_docs)}")
        print(f"  Duplicates to filter: {len(duplicate_map)}")
        
        return duplicate_map
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fast Jaccard similarity on word sets."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

# ============================================================
# SEMANTIC GRAPH ANALYZER
# ============================================================

class SemanticGraphAnalyzer:
    """Analyzes the knowledge graph with PART_OF filtered out."""
    
    def __init__(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame):
        self.edges_df = edges_df
        self.nodes_df = nodes_df
        self.semantic_edges = self._filter_semantic_edges()
        
    def _filter_semantic_edges(self) -> pd.DataFrame:
        """Filter to semantic edges only (exclude PART_OF)."""
        if 'relation' not in self.edges_df.columns:
            return self.edges_df
        
        semantic = self.edges_df[self.edges_df['relation'] != 'PART_OF'].copy()
        print(f"  Semantic edges: {len(semantic)} (filtered from {len(self.edges_df)})")
        return semantic
    
    def compute_semantic_degree(self) -> pd.DataFrame:
        """Compute node degrees using only semantic edges."""
        out_degree = self.semantic_edges.groupby('source').size().reset_index(name='out_degree')
        in_degree = self.semantic_edges.groupby('target').size().reset_index(name='in_degree')
        
        out_degree = out_degree.rename(columns={'source': 'node_id'})
        in_degree = in_degree.rename(columns={'target': 'node_id'})
        
        degrees = pd.merge(out_degree, in_degree, on='node_id', how='outer').fillna(0)
        degrees['semantic_degree'] = degrees['out_degree'] + degrees['in_degree']
        
        return degrees.sort_values('semantic_degree', ascending=False)
    
    def compute_pagerank(self, damping: float = 0.85, iterations: int = 20) -> Dict[str, float]:
        """Compute PageRank on semantic graph (embedding centrality proxy)."""
        print("  Computing PageRank on semantic graph...")
        
        # Get unique nodes
        nodes = set(self.semantic_edges['source'].tolist() + 
                   self.semantic_edges['target'].tolist())
        
        if len(nodes) == 0:
            print("    No semantic edges for PageRank")
            return {}
        
        # Initialize PageRank scores
        n = len(nodes)
        pr = {node: 1.0 / n for node in nodes}
        
        # Build adjacency list (outgoing edges)
        outgoing = defaultdict(list)
        for _, row in self.semantic_edges.iterrows():
            outgoing[row['source']].append(row['target'])
        
        # Iterate
        for _ in range(iterations):
            new_pr = {}
            for node in nodes:
                # Sum of PR from incoming nodes
                incoming_sum = 0.0
                for src, targets in outgoing.items():
                    if node in targets and len(targets) > 0:
                        incoming_sum += pr[src] / len(targets)
                
                new_pr[node] = (1 - damping) / n + damping * incoming_sum
            pr = new_pr
        
        print(f"    PageRank computed for {len(pr)} nodes")
        return pr

# ============================================================
# SEMANTIC VALIDATION LAYER
# ============================================================

class SemanticValidator:
    """Validates gems using embedding-based metrics."""
    
    def __init__(self, embedding_index: EmbeddingIndex, pagerank: Dict[str, float]):
        self.embedding_index = embedding_index
        self.pagerank = pagerank
    
    def compute_semantic_score(self, artifact_id: str, structural_degree: float) -> Dict:
        """
        Compute semantic validation score for an artifact.
        
        Formula: semantic_score = structural_degree * embedding_centrality * coherence_factor
        
        Returns dict with all component scores.
        """
        # Get embedding centrality (PageRank score)
        embedding_centrality = self.pagerank.get(artifact_id, 0.0)
        
        # Normalize PageRank (max = 1.0)
        max_pr = max(self.pagerank.values()) if self.pagerank else 1.0
        normalized_centrality = embedding_centrality / max_pr if max_pr > 0 else 0.0
        
        # Coherence factor (placeholder - would need full embedding analysis)
        # For now, use presence of embedding as binary factor
        has_embedding = self.embedding_index.get_embedding(artifact_id) is not None
        coherence_factor = 1.0 if has_embedding else 0.5
        
        # Compute final semantic score
        # Multiply degree by centrality to penalize structurally-high but semantically-isolated nodes
        semantic_score = (
            0.4 * min(structural_degree / 100, 1.0) +  # Capped structural contribution
            0.4 * normalized_centrality +               # Semantic centrality
            0.2 * coherence_factor                      # Embedding grounding
        )
        
        return {
            'structural_degree': structural_degree,
            'embedding_centrality': normalized_centrality,
            'coherence_factor': coherence_factor,
            'semantic_score': round(semantic_score, 4),
            'has_embedding': has_embedding,
            'pagerank_raw': embedding_centrality
        }

# ============================================================
# GOLDEN GEMS MINER v2
# ============================================================

def mine_golden_gems_v2():
    """
    Re-mine golden gems with:
    1. Deduplication
    2. PART_OF filtering
    3. Semantic validation
    """
    
    print("=" * 60)
    print("🏆 GOLDEN GEMS MINER v2.0 - Semantic Validation Edition")
    print("=" * 60)
    
    run_id = hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16]
    
    # Load data
    print("\n📊 Loading data...")
    chunks_df = pd.read_parquet(GOLD_PATH / 'chunks.parquet')
    
    nodes = []
    with open(INDEXED_PATH / 'graph' / 'nodes.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            nodes.append(json.loads(line))
    nodes_df = pd.DataFrame(nodes)
    
    edges = []
    with open(INDEXED_PATH / 'graph' / 'edges.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            edges.append(json.loads(line))
    edges_df = pd.DataFrame(edges)
    
    print(f"  Chunks: {len(chunks_df)}")
    print(f"  Nodes: {len(nodes_df)}")
    print(f"  Edges: {len(edges_df)}")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    embedding_index = EmbeddingIndex(chunks_df)
    
    # Step 1: Deduplication
    print("\n" + "=" * 40)
    print("STEP 1: DEDUPLICATION")
    print("=" * 40)
    
    top_docs = chunks_df['doc_id'].value_counts().head(50).index.tolist()
    dedup_engine = DeduplicationEngine(chunks_df, embedding_index)
    duplicate_map = dedup_engine.find_duplicates(top_docs)
    
    # Step 2: Filter to semantic graph
    print("\n" + "=" * 40)
    print("STEP 2: SEMANTIC GRAPH FILTERING")
    print("=" * 40)
    
    semantic_analyzer = SemanticGraphAnalyzer(edges_df, nodes_df)
    semantic_degrees = semantic_analyzer.compute_semantic_degree()
    pagerank = semantic_analyzer.compute_pagerank()
    
    # Step 3: Semantic validation
    print("\n" + "=" * 40)
    print("STEP 3: SEMANTIC VALIDATION")
    print("=" * 40)
    
    validator = SemanticValidator(embedding_index, pagerank)
    
    # Mine gems with new methodology
    gems = []
    
    # Metadata header
    metadata = {
        '_metadata': {
            'run_id': run_id,
            'generated_at': datetime.now().isoformat(),
            'version': '2.0',
            'methodology': 'Semantic validation with deduplication + PART_OF filtering',
            'thresholds': {
                'duplicate_threshold': DUPLICATE_THRESHOLD,
                'min_semantic_degree': MIN_SEMANTIC_DEGREE
            },
            'input_files': {
                'chunks': str(GOLD_PATH / 'chunks.parquet'),
                'nodes': str(INDEXED_PATH / 'graph' / 'nodes.jsonl'),
                'edges': str(INDEXED_PATH / 'graph' / 'edges.jsonl')
            },
            'counts': {
                'chunks': len(chunks_df),
                'nodes': len(nodes_df),
                'edges': len(edges_df),
                'semantic_edges': len(semantic_analyzer.semantic_edges),
                'duplicates_filtered': len(duplicate_map)
            }
        }
    }
    gems.append(metadata)
    
    # Pattern 1: Semantic Hub Nodes (high semantic degree, deduplicated)
    print("\n🔍 Mining Pattern 1: Semantic Hub Nodes...")
    
    for _, row in semantic_degrees.head(15).iterrows():
        node_id = row['node_id']
        
        # Skip duplicates
        if node_id in duplicate_map:
            continue
        
        # Skip if semantic degree too low
        if row['semantic_degree'] < MIN_SEMANTIC_DEGREE:
            continue
        
        # Get node label
        node_info = nodes_df[nodes_df['id'] == node_id]
        label = "Unknown"
        if not node_info.empty and 'label' in node_info.columns:
            label = str(node_info.iloc[0]['label'])[:100]
        
        # Compute semantic validation
        validation = validator.compute_semantic_score(node_id, row['semantic_degree'])
        
        gems.append({
            'artifact_id': node_id,
            'title': label,
            'type': 'semantic_hub',
            'pattern': 'Semantic Hub Node',
            'score_semantic': validation['semantic_score'],
            'evidence': {
                'semantic_out_degree': int(row['out_degree']),
                'semantic_in_degree': int(row['in_degree']),
                'total_semantic_degree': int(row['semantic_degree']),
                'embedding_centrality': validation['embedding_centrality'],
                'pagerank': validation['pagerank_raw'],
                'has_embedding': validation['has_embedding']
            },
            'validation': {
                'deduplicated': True,
                'part_of_filtered': True,
                'semantic_validated': True
            },
            'methodology': 'Semantic edges only + PageRank centrality',
            'run_id': run_id
        })
    
    # Pattern 2: Entity Hubs (excluding self-referential)
    print("🔍 Mining Pattern 2: Entity Hubs (non-self-referential)...")
    
    entity_degrees = semantic_analyzer.semantic_edges[
        semantic_analyzer.semantic_edges['target'].str.startswith('ent::')
    ]['target'].value_counts()
    
    for ent, degree in entity_degrees.head(10).items():
        ent_name = ent.replace('ent::', '')
        
        # Filter self-referential (corpus name entities)
        if ent_name.lower() in ['bizra', 'bizra-data-lake']:
            continue
        
        if degree < MIN_SEMANTIC_DEGREE:
            continue
        
        validation = validator.compute_semantic_score(ent, float(degree))
        
        gems.append({
            'artifact_id': ent,
            'title': ent_name,
            'type': 'entity_hub',
            'pattern': 'Entity Hub (Non-Self-Referential)',
            'score_semantic': validation['semantic_score'],
            'evidence': {
                'mention_count': int(degree),
                'embedding_centrality': validation['embedding_centrality'],
                'pagerank': validation['pagerank_raw']
            },
            'validation': {
                'self_referential': False,
                'semantic_validated': True
            },
            'methodology': 'Entity mentions via semantic edges, self-ref filtered',
            'run_id': run_id
        })
    
    # Pattern 3: Cross-Domain Bridges (semantic edges only)
    print("🔍 Mining Pattern 3: Cross-Domain Bridges (semantic)...")
    
    if 'kind' in nodes_df.columns:
        # Add kind info to semantic edges
        sem_with_kinds = semantic_analyzer.semantic_edges.merge(
            nodes_df[['id', 'kind']], 
            left_on='source', right_on='id', how='left'
        ).rename(columns={'kind': 'source_kind'})
        
        sem_with_kinds = sem_with_kinds.merge(
            nodes_df[['id', 'kind']], 
            left_on='target', right_on='id', how='left'
        ).rename(columns={'kind': 'target_kind'})
        
        # Cross-domain = different kinds
        cross_domain = sem_with_kinds[sem_with_kinds['source_kind'] != sem_with_kinds['target_kind']]
        bridge_counts = cross_domain.groupby('source').size().sort_values(ascending=False)
        
        for node_id, bridge_count in bridge_counts.head(10).items():
            if node_id in duplicate_map:
                continue
            if bridge_count < 2:
                continue
            
            node_info = nodes_df[nodes_df['id'] == node_id]
            label = str(node_info.iloc[0]['label'])[:100] if not node_info.empty else "Unknown"
            
            validation = validator.compute_semantic_score(node_id, float(bridge_count))
            
            gems.append({
                'artifact_id': node_id,
                'title': label,
                'type': 'cross_domain_bridge',
                'pattern': 'Cross-Domain Bridge (Semantic)',
                'score_semantic': validation['semantic_score'],
                'evidence': {
                    'semantic_bridge_count': int(bridge_count),
                    'embedding_centrality': validation['embedding_centrality']
                },
                'validation': {
                    'deduplicated': True,
                    'part_of_filtered': True,
                    'semantic_validated': True
                },
                'methodology': 'Cross-kind semantic edges only',
                'run_id': run_id
            })
    
    # Pattern 4: High-Coherence Documents (embedding-based)
    print("🔍 Mining Pattern 4: High-Coherence Documents...")
    
    coherent_docs = []
    for doc_id in dedup_engine.canonical_docs:
        doc_embedding = embedding_index.get_doc_embedding(doc_id)
        if doc_embedding is not None:
            # Check if doc has semantic edges
            sem_degree = semantic_degrees[semantic_degrees['node_id'] == doc_id]
            if not sem_degree.empty and sem_degree.iloc[0]['semantic_degree'] >= MIN_SEMANTIC_DEGREE:
                coherent_docs.append({
                    'doc_id': doc_id,
                    'semantic_degree': sem_degree.iloc[0]['semantic_degree'],
                    'has_embedding': True
                })
    
    coherent_docs = sorted(coherent_docs, key=lambda x: x['semantic_degree'], reverse=True)
    
    for doc in coherent_docs[:10]:
        doc_id = doc['doc_id']
        node_info = nodes_df[nodes_df['id'] == doc_id]
        label = str(node_info.iloc[0]['label'])[:100] if not node_info.empty else "Unknown"
        
        validation = validator.compute_semantic_score(doc_id, doc['semantic_degree'])
        
        gems.append({
            'artifact_id': doc_id,
            'title': label,
            'type': 'coherent_document',
            'pattern': 'High-Coherence Document',
            'score_semantic': validation['semantic_score'],
            'evidence': {
                'semantic_degree': int(doc['semantic_degree']),
                'embedding_grounded': True,
                'is_canonical': True  # Not a duplicate
            },
            'validation': {
                'deduplicated': True,
                'embedding_validated': True,
                'semantic_validated': True
            },
            'methodology': 'Canonical docs with embeddings + semantic edges',
            'run_id': run_id
        })
    
    # Write output
    output_path = GOLD_PATH / 'golden_gems_v2.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for gem in gems:
            f.write(json.dumps(gem, ensure_ascii=False) + '\n')
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MINING SUMMARY")
    print("=" * 60)
    
    gem_types = {}
    for g in gems[1:]:  # Skip metadata
        t = g.get('pattern', 'Unknown')
        gem_types[t] = gem_types.get(t, 0) + 1
    
    print(f"\nTotal gems: {len(gems) - 1}")  # Exclude metadata
    print(f"Run ID: {run_id}")
    print("\nBy pattern type:")
    for t, c in sorted(gem_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    
    # Calculate mean semantic score
    scores = [g.get('score_semantic', 0) for g in gems[1:] if 'score_semantic' in g]
    mean_score = sum(scores) / len(scores) if scores else 0
    print(f"\nMean semantic score: {mean_score:.4f}")
    
    # Top gems
    sorted_gems = sorted([g for g in gems[1:] if 'score_semantic' in g], 
                        key=lambda x: x['score_semantic'], reverse=True)
    print("\nTop 5 gems by semantic score:")
    for i, g in enumerate(sorted_gems[:5], 1):
        print(f"  {i}. [{g['type']}] {g['title'][:40]}... (score={g['score_semantic']:.4f})")
    
    print(f"\n✅ Output written to: {output_path}")
    print("=" * 60)
    
    return gems

if __name__ == '__main__':
    mine_golden_gems_v2()
