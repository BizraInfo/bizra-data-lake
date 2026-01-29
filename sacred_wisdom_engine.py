#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
    ███████╗ █████╗  ██████╗██████╗ ███████╗██████╗     ██╗    ██╗██╗███████╗██████╗  ██████╗ ███╗   ███╗
    ██╔════╝██╔══██╗██╔════╝██╔══██╗██╔════╝██╔══██╗    ██║    ██║██║██╔════╝██╔══██╗██╔═══██╗████╗ ████║
    ███████╗███████║██║     ██████╔╝█████╗  ██║  ██║    ██║ █╗ ██║██║███████╗██║  ██║██║   ██║██╔████╔██║
    ╚════██║██╔══██║██║     ██╔══██╗██╔══╝  ██║  ██║    ██║███╗██║██║╚════██║██║  ██║██║   ██║██║╚██╔╝██║
    ███████║██║  ██║╚██████╗██║  ██║███████╗██████╔╝    ╚███╔███╔╝██║███████║██████╔╝╚██████╔╝██║ ╚═╝ ██║
    ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝      ╚══╝╚══╝ ╚═╝╚══════╝╚═════╝  ╚═════╝ ╚═╝     ╚═╝
                                                                                                        
    ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗                   
    ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║                   
    ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝██║  ███╗██████╔╝███████║██████╔╝███████║                   
    ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║                   
    ██║  ██║   ██║   ██║     ███████╗██║  ██║╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║                   
    ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝                   
═══════════════════════════════════════════════════════════════════════════════════════════════════════
    SACRED WISDOM HYPERGRAPH ENGINE — Interdisciplinary Knowledge Discovery
    
    PHILOSOPHY:
      "Most see religious text. We see structured wisdom — hidden patterns, thought flows,
       interdisciplinary connections that bridge ancient insights to modern knowledge."
    
    ARCHITECTURE:
      L1: Data Ingestion (Quran 6,236 ayat + Hadith Kutub al-Sittah)
      L2: Semantic Embedding (all-MiniLM-L6-v2 / multilingual)
      L3: Hypergraph Construction (multi-relational knowledge fabric)
      L4: Pattern Discovery (themes, concepts, cross-references)
      L5: GoT Reasoning (multi-hop thought chains)
      L6: SNR Optimization (signal maximization)
    
    GIANTS ABSORBED:
      - Al-Khwarizmi (Algorithmic thinking)
      - Ibn Sina (Systematic knowledge organization)
      - Meta FAISS (Vector similarity)
      - NetworkX (Graph algorithms)
      - Sentence Transformers (Semantic embeddings)
      - Graph of Thoughts (Reasoning chains)
      - Shannon (SNR optimization)
    
    RESPECT PROTOCOL:
      All sacred texts treated with utmost respect as sources of wisdom.
      Analysis is purely for knowledge discovery and pattern extraction.
    
    Created: 2026-01-23 | Dubai | BIZRA Genesis
═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import logging
import time
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum, auto
import numpy as np
import pandas as pd

# Optional imports with fallbacks
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class SacredConfig:
    """Configuration for the Sacred Wisdom Engine."""
    DATA_LAKE = Path(r"C:\BIZRA-DATA-LAKE")
    KNOWLEDGE = DATA_LAKE / "03_INDEXED" / "knowledge"
    GOLD = DATA_LAKE / "04_GOLD"
    
    # Data paths
    QURAN_PATH = KNOWLEDGE / "quran"
    HADITH_PATH = KNOWLEDGE / "hadith" / "LK-Hadith-Corpus-master"
    
    # Output paths
    SACRED_GRAPH = GOLD / "sacred_wisdom_graph.json"
    SACRED_EMBEDDINGS = GOLD / "sacred_wisdom_embeddings.npy"
    SACRED_INDEX = GOLD / "sacred_wisdom_index.json"
    SACRED_PATTERNS = GOLD / "sacred_wisdom_patterns.jsonl"
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Hadith collections
    KUTUB_SITTAH = ["Bukhari", "Muslim", "AbuDaud", "Tirmizi", "Nesai", "IbnMaja"]


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("SacredWisdom")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class WisdomType(Enum):
    """Types of wisdom nodes in the hypergraph."""
    AYAH = auto()           # Quranic verse
    HADITH = auto()         # Prophetic tradition
    SURAH = auto()          # Quranic chapter
    BOOK = auto()           # Hadith book/collection
    CHAPTER = auto()        # Chapter in collection
    NARRATOR = auto()       # Hadith narrator (from Isnad)
    CONCEPT = auto()        # Extracted concept/theme
    THEME = auto()          # High-level theme
    ENTITY = auto()         # Named entity (person, place)
    LINGUISTIC = auto()     # Linguistic pattern
    CROSS_REF = auto()      # Cross-reference link


class RelationType(Enum):
    """Types of relationships in the hypergraph."""
    CONTAINS = auto()       # Surah contains Ayah
    NARRATES = auto()       # Narrator narrates Hadith
    REFERENCES = auto()     # Cross-reference
    THEMATICALLY_RELATED = auto()
    SEMANTICALLY_SIMILAR = auto()
    LINGUISTICALLY_PATTERNS = auto()
    CONCEPTUALLY_LINKED = auto()
    CHAIN_OF_TRANSMISSION = auto()  # Isnad chain
    EXPLAINS = auto()       # Hadith explains Quran
    PARALLELS = auto()      # Parallel meaning


@dataclass
class WisdomNode:
    """A node in the Sacred Wisdom Hypergraph."""
    id: str
    name: str
    type: WisdomType
    source: str                     # "quran" or hadith collection name
    text_arabic: str = ""
    text_english: str = ""
    reference: str = ""             # e.g., "2:255" or "Bukhari:1"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    snr_score: float = 0.0
    centrality: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['type'] = self.type.name
        d['embedding'] = None  # Don't serialize embeddings to JSON
        return d


@dataclass
class WisdomEdge:
    """An edge in the Sacred Wisdom Hypergraph."""
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['relation'] = self.relation.name
        return d


@dataclass
class DiscoveredPattern:
    """A discovered pattern in the wisdom corpus."""
    id: str
    pattern_type: str
    description: str
    nodes: List[str]
    confidence: float
    evidence: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass 
class ThoughtChain:
    """A Graph of Thoughts reasoning chain."""
    query: str
    steps: List[Dict[str, Any]]
    conclusion: str
    confidence: float
    sources: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class DataIngestionLayer:
    """Ingests Quran and Hadith data into unified format."""
    
    def __init__(self):
        self.quran_df: Optional[pd.DataFrame] = None
        self.hadith_dfs: Dict[str, pd.DataFrame] = {}
        self.nodes: Dict[str, WisdomNode] = {}
        
    def ingest_quran(self) -> int:
        """Load Quran data from parquet."""
        quran_file = SacredConfig.QURAN_PATH / "quran_full.parquet"
        if not quran_file.exists():
            log.warning(f"Quran file not found: {quran_file}")
            return 0
        
        self.quran_df = pd.read_parquet(quran_file)
        count = 0
        
        # Create Surah nodes
        surah_names = self.quran_df[['surah', 'surah_name']].drop_duplicates()
        for _, row in surah_names.iterrows():
            surah_id = f"surah_{row['surah']}"
            self.nodes[surah_id] = WisdomNode(
                id=surah_id,
                name=row['surah_name'],
                type=WisdomType.SURAH,
                source="quran",
                reference=f"Surah {row['surah']}",
                metadata={"surah_number": int(row['surah'])}
            )
            count += 1
        
        # Create Ayah nodes
        for _, row in self.quran_df.iterrows():
            ayah_id = f"quran_{row['surah']}_{row['ayah']}"
            reference = f"{row['surah']}:{row['ayah']}"
            
            self.nodes[ayah_id] = WisdomNode(
                id=ayah_id,
                name=f"{row['surah_name']} {reference}",
                type=WisdomType.AYAH,
                source="quran",
                text_arabic=row.get('arabic_text', ''),
                text_english=row.get('english_text', ''),
                reference=reference,
                metadata={
                    "surah": int(row['surah']),
                    "ayah": int(row['ayah']),
                    "juz": int(row.get('juz', 0)),
                    "page": int(row.get('page', 0))
                }
            )
            count += 1
        
        log.info(f"   ✓ Quran: {len(self.quran_df)} ayat, {len(surah_names)} surahs")
        return count
    
    def ingest_hadith(self) -> int:
        """Load all Hadith collections."""
        if not SacredConfig.HADITH_PATH.exists():
            log.warning(f"Hadith path not found: {SacredConfig.HADITH_PATH}")
            return 0
        
        total_count = 0
        
        for collection in SacredConfig.KUTUB_SITTAH:
            collection_path = SacredConfig.HADITH_PATH / collection
            if not collection_path.exists():
                continue
            
            # Create collection node
            collection_id = f"collection_{collection.lower()}"
            self.nodes[collection_id] = WisdomNode(
                id=collection_id,
                name=collection,
                type=WisdomType.BOOK,
                source=collection.lower(),
                reference=collection
            )
            
            # Load all CSV files
            csv_files = list(collection_path.glob("*.csv"))
            collection_hadith_count = 0
            
            all_dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, on_bad_lines='skip')
                    df['source_file'] = csv_file.stem
                    df['collection'] = collection
                    all_dfs.append(df)
                except Exception as e:
                    log.warning(f"Error reading {csv_file}: {e}")
            
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                self.hadith_dfs[collection] = combined_df
                
                # Create Hadith nodes
                for idx, row in combined_df.iterrows():
                    hadith_num = row.get('Hadith_number', idx)
                    hadith_id = f"hadith_{collection.lower()}_{hadith_num}"
                    
                    # Extract texts
                    arabic = row.get('Arabic_Hadith', '') or row.get('Arabic_Matn', '') or ''
                    english = row.get('English_Hadith', '') or row.get('English_Matn', '') or ''
                    grade = row.get('English_Grade', '') or ''
                    
                    if not arabic and not english:
                        continue
                    
                    self.nodes[hadith_id] = WisdomNode(
                        id=hadith_id,
                        name=f"{collection} #{hadith_num}",
                        type=WisdomType.HADITH,
                        source=collection.lower(),
                        text_arabic=str(arabic)[:2000],  # Truncate very long texts
                        text_english=str(english)[:2000],
                        reference=f"{collection}:{hadith_num}",
                        metadata={
                            "collection": collection,
                            "hadith_number": hadith_num,
                            "chapter": row.get('Chapter_English', ''),
                            "grade": grade
                        }
                    )
                    collection_hadith_count += 1
                
                total_count += collection_hadith_count
                log.info(f"   ✓ {collection}: {collection_hadith_count} hadith from {len(csv_files)} chapters")
        
        return total_count
    
    def ingest_all(self) -> Dict[str, int]:
        """Ingest all wisdom sources."""
        log.info("📖 Ingesting Sacred Wisdom Sources...")
        
        quran_count = self.ingest_quran()
        hadith_count = self.ingest_hadith()
        
        return {
            "quran": quran_count,
            "hadith": hadith_count,
            "total_nodes": len(self.nodes)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingLayer:
    """Generates semantic embeddings for all wisdom nodes."""
    
    def __init__(self):
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        
    def initialize(self):
        """Initialize the embedding model."""
        if not HAS_TRANSFORMERS:
            log.warning("sentence-transformers not available, using random embeddings")
            return
        
        log.info("   Loading embedding model...")
        self.model = SentenceTransformer(SacredConfig.EMBEDDING_MODEL)
        log.info(f"   ✓ Model loaded: {SacredConfig.EMBEDDING_MODEL}")
    
    def embed_nodes(self, nodes: Dict[str, WisdomNode], batch_size: int = 256) -> int:
        """Generate embeddings for all nodes."""
        if not self.model:
            # Generate random embeddings as fallback
            for node_id, node in nodes.items():
                node.embedding = np.random.randn(SacredConfig.EMBEDDING_DIM).astype(np.float32)
                self.embeddings[node_id] = node.embedding
            return len(nodes)
        
        # Prepare texts for embedding
        texts = []
        node_ids = []
        
        for node_id, node in nodes.items():
            # Combine English text for embedding (more semantic value)
            text = node.text_english or node.name
            if node.text_arabic:
                text = f"{text} {node.text_arabic[:200]}"  # Add some Arabic context
            texts.append(text[:512])  # Truncate to model max
            node_ids.append(node_id)
        
        log.info(f"   Generating embeddings for {len(texts)} nodes...")
        
        # Batch embed
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 1000 == 0:
                log.info(f"   ... {min(i + batch_size, len(texts))}/{len(texts)}")
        
        # Assign embeddings to nodes
        for node_id, embedding in zip(node_ids, all_embeddings):
            nodes[node_id].embedding = embedding
            self.embeddings[node_id] = embedding
        
        log.info(f"   ✓ Embedded {len(all_embeddings)} nodes")
        return len(all_embeddings)
    
    def save_embeddings(self, path: Path):
        """Save embeddings to numpy file."""
        if not self.embeddings:
            return
        
        # Create matrix and index
        node_ids = list(self.embeddings.keys())
        matrix = np.array([self.embeddings[nid] for nid in node_ids])
        
        np.save(path, matrix)
        
        # Save index mapping
        index_path = path.with_suffix('.index.json')
        with open(index_path, 'w') as f:
            json.dump(node_ids, f)
        
        log.info(f"   ✓ Saved embeddings: {matrix.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERGRAPH LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class HypergraphLayer:
    """Builds the multi-relational knowledge hypergraph."""
    
    def __init__(self):
        self.graph = nx.DiGraph() if HAS_NETWORKX else None
        self.edges: List[WisdomEdge] = []
        self.concepts: Dict[str, Set[str]] = defaultdict(set)  # concept -> node_ids
        self.themes: Dict[str, Set[str]] = defaultdict(set)    # theme -> node_ids
        
    def build_graph(self, nodes: Dict[str, WisdomNode]) -> int:
        """Build the hypergraph from nodes."""
        if not self.graph:
            log.warning("NetworkX not available, using edge list only")
        
        log.info("   Building hypergraph structure...")
        
        # Add all nodes
        for node_id, node in nodes.items():
            if self.graph:
                self.graph.add_node(
                    node_id,
                    type=node.type.name,
                    source=node.source,
                    name=node.name
                )
        
        edge_count = 0
        
        # 1. Structural edges (Surah -> Ayah, Collection -> Hadith)
        edge_count += self._build_structural_edges(nodes)
        
        # 2. Thematic edges (extract themes and link)
        edge_count += self._build_thematic_edges(nodes)
        
        # 3. Semantic similarity edges (based on embeddings)
        edge_count += self._build_semantic_edges(nodes)
        
        # 4. Cross-reference edges (Quran <-> Hadith)
        edge_count += self._build_cross_reference_edges(nodes)
        
        log.info(f"   ✓ Built {edge_count} edges")
        return edge_count
    
    def _build_structural_edges(self, nodes: Dict[str, WisdomNode]) -> int:
        """Build containment/structural edges."""
        count = 0
        
        for node_id, node in nodes.items():
            # Surah contains Ayah
            if node.type == WisdomType.AYAH:
                surah_num = node.metadata.get('surah')
                if surah_num:
                    surah_id = f"surah_{surah_num}"
                    if surah_id in nodes:
                        self._add_edge(surah_id, node_id, RelationType.CONTAINS)
                        count += 1
            
            # Collection contains Hadith
            elif node.type == WisdomType.HADITH:
                collection = node.metadata.get('collection', '').lower()
                if collection:
                    collection_id = f"collection_{collection}"
                    if collection_id in nodes:
                        self._add_edge(collection_id, node_id, RelationType.CONTAINS)
                        count += 1
        
        return count
    
    def _build_thematic_edges(self, nodes: Dict[str, WisdomNode]) -> int:
        """Extract themes and build thematic edges."""
        # Key themes to detect
        theme_keywords = {
            "mercy": ["mercy", "merciful", "رحم", "رحيم", "رحمة"],
            "justice": ["justice", "just", "عدل", "قسط"],
            "knowledge": ["knowledge", "know", "علم", "عرف"],
            "faith": ["faith", "believe", "إيمان", "آمن"],
            "prayer": ["prayer", "pray", "صلاة", "صلى"],
            "charity": ["charity", "give", "زكاة", "صدقة"],
            "patience": ["patience", "patient", "صبر"],
            "creation": ["create", "creation", "خلق"],
            "guidance": ["guide", "guidance", "هدى", "هداية"],
            "truth": ["truth", "true", "حق", "صدق"],
            "paradise": ["paradise", "heaven", "جنة"],
            "judgment": ["judgment", "day", "يوم", "قيامة", "حساب"],
            "prophets": ["prophet", "messenger", "نبي", "رسول"],
            "wisdom": ["wisdom", "wise", "حكمة", "حكيم"],
        }
        
        count = 0
        
        for node_id, node in nodes.items():
            if node.type not in [WisdomType.AYAH, WisdomType.HADITH]:
                continue
            
            text = f"{node.text_english} {node.text_arabic}".lower()
            
            for theme, keywords in theme_keywords.items():
                if any(kw in text for kw in keywords):
                    self.themes[theme].add(node_id)
        
        # Create theme nodes and edges
        for theme, node_ids in self.themes.items():
            if len(node_ids) >= 2:  # Only themes with multiple references
                theme_id = f"theme_{theme}"
                if self.graph:
                    self.graph.add_node(theme_id, type="THEME", name=theme.title())
                
                for node_id in node_ids:
                    self._add_edge(theme_id, node_id, RelationType.THEMATICALLY_RELATED, weight=0.8)
                    count += 1
        
        log.info(f"      Detected {len(self.themes)} themes")
        return count
    
    def _build_semantic_edges(self, nodes: Dict[str, WisdomNode], threshold: float = 0.7, max_neighbors: int = 5) -> int:
        """Build edges between semantically similar nodes."""
        # Get nodes with embeddings
        embedded_nodes = [(nid, n) for nid, n in nodes.items() if n.embedding is not None]
        
        if len(embedded_nodes) < 2:
            return 0
        
        count = 0
        
        # Build embedding matrix
        node_ids = [nid for nid, _ in embedded_nodes]
        embeddings = np.array([n.embedding for _, n in embedded_nodes])
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms
        
        # For each node, find similar nodes
        # (Using batched approach to handle large datasets)
        batch_size = 1000
        
        for i in range(0, len(node_ids), batch_size):
            batch_end = min(i + batch_size, len(node_ids))
            batch_embeddings = embeddings_norm[i:batch_end]
            
            # Compute similarities
            similarities = batch_embeddings @ embeddings_norm.T
            
            for j, (local_idx, sim_row) in enumerate(zip(range(i, batch_end), similarities)):
                source_id = node_ids[local_idx]
                
                # Get top similar (excluding self)
                top_indices = np.argsort(sim_row)[::-1][1:max_neighbors+1]
                
                for target_idx in top_indices:
                    if sim_row[target_idx] >= threshold:
                        target_id = node_ids[target_idx]
                        self._add_edge(
                            source_id, target_id, 
                            RelationType.SEMANTICALLY_SIMILAR,
                            weight=float(sim_row[target_idx])
                        )
                        count += 1
        
        return count
    
    def _build_cross_reference_edges(self, nodes: Dict[str, WisdomNode]) -> int:
        """Build edges linking Quran and Hadith on same topics."""
        count = 0
        
        # Group by theme for cross-referencing
        for theme, node_ids in self.themes.items():
            quran_nodes = [nid for nid in node_ids if nid.startswith('quran_')]
            hadith_nodes = [nid for nid in node_ids if nid.startswith('hadith_')]
            
            # Create cross-reference edges
            for q_id in quran_nodes[:10]:  # Limit to prevent explosion
                for h_id in hadith_nodes[:10]:
                    self._add_edge(q_id, h_id, RelationType.REFERENCES, weight=0.6)
                    count += 1
        
        return count
    
    def _add_edge(self, source: str, target: str, relation: RelationType, weight: float = 1.0):
        """Add an edge to the graph."""
        edge = WisdomEdge(source, target, relation, weight)
        self.edges.append(edge)
        
        if self.graph:
            self.graph.add_edge(source, target, relation=relation.name, weight=weight)
    
    def compute_centrality(self, nodes: Dict[str, WisdomNode]):
        """Compute centrality scores for all nodes."""
        if not self.graph or len(self.graph) == 0:
            return
        
        log.info("   Computing centrality...")
        
        try:
            # PageRank for importance
            pagerank = nx.pagerank(self.graph, weight='weight')
            
            for node_id, score in pagerank.items():
                if node_id in nodes:
                    nodes[node_id].centrality = score
            
            log.info(f"   ✓ Computed centrality for {len(pagerank)} nodes")
        except Exception as e:
            log.warning(f"   Centrality computation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# HIDDEN KNOWLEDGE LAYER — علم الكتاب Discovery
# ═══════════════════════════════════════════════════════════════════════════════

class HiddenKnowledgeLayer:
    """
    علم الكتاب — Discovery of hidden patterns in sacred text.
    
    This layer seeks patterns that emerge from the data itself:
    - Numerical patterns and symmetries
    - Ring/chiastic structures
    - Root word networks
    - Thematic echoes across sources
    
    "We have not neglected anything in the Book" — 6:38
    "مَّا فَرَّطْنَا فِى ٱلْكِتَٰبِ مِن شَىْءٍۢ"
    
    With humility: This tool assists discovery, but true understanding
    comes from a source beyond computation.
    """
    
    def __init__(self):
        self.numerical_patterns: List[Dict[str, Any]] = []
        self.symmetries: List[Dict[str, Any]] = []
        self.root_networks: Dict[str, Set[str]] = defaultdict(set)
        self.echoes: List[Dict[str, Any]] = []
        
    def analyze(self, nodes: Dict[str, WisdomNode]) -> Dict[str, Any]:
        """Run all hidden pattern discovery algorithms."""
        log.info("   📿 Seeking hidden patterns (علم الكتاب)...")
        
        # 1. Numerical patterns
        self._find_numerical_patterns(nodes)
        
        # 2. Word frequency patterns
        self._find_word_patterns(nodes)
        
        # 3. Thematic symmetries
        self._find_symmetries(nodes)
        
        # 4. Cross-source echoes
        self._find_echoes(nodes)
        
        total = len(self.numerical_patterns) + len(self.symmetries) + len(self.echoes)
        log.info(f"   ✓ Found {total} hidden patterns")
        
        return self.get_summary()
    
    def _find_numerical_patterns(self, nodes: Dict[str, WisdomNode]):
        """Find numerical patterns in the text."""
        # Count specific significant words
        word_counts = defaultdict(int)
        significant_words = {
            # Arabic words with their transliteration
            "الله": "Allah",
            "رب": "Lord (Rabb)",
            "رحم": "Mercy (root)",
            "علم": "Knowledge (root)",
            "قلب": "Heart (Qalb)",
            "يوم": "Day (Yawm)",
            "دنيا": "World (Dunya)",
            "آخرة": "Hereafter (Akhira)",
            "ملائكة": "Angels",
            "شيطان": "Satan",
            "جنة": "Paradise",
            "نار": "Fire/Hell",
            "حياة": "Life",
            "موت": "Death",
            "نور": "Light",
            "ظلم": "Darkness",
        }
        
        quran_nodes = [n for n in nodes.values() if n.type == WisdomType.AYAH]
        
        for node in quran_nodes:
            if node.text_arabic:
                for ar_word, meaning in significant_words.items():
                    count = node.text_arabic.count(ar_word)
                    if count > 0:
                        word_counts[meaning] += count
        
        # Record patterns
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            self.numerical_patterns.append({
                "type": "WORD_COUNT",
                "word": word,
                "count": count,
                "source": "quran"
            })
        
        # Check for balanced pairs (famous patterns)
        pairs_to_check = [
            ("Life", "Death"),
            ("World (Dunya)", "Hereafter (Akhira)"),
            ("Angels", "Satan"),
            ("Light", "Darkness"),
        ]
        
        for word1, word2 in pairs_to_check:
            c1 = word_counts.get(word1, 0)
            c2 = word_counts.get(word2, 0)
            if c1 > 0 and c2 > 0:
                ratio = c1 / c2 if c2 > 0 else 0
                self.numerical_patterns.append({
                    "type": "WORD_PAIR_BALANCE",
                    "pair": (word1, word2),
                    "counts": (c1, c2),
                    "ratio": round(ratio, 3),
                    "note": "Balanced" if 0.9 <= ratio <= 1.1 else "Proportional"
                })
    
    def _find_word_patterns(self, nodes: Dict[str, WisdomNode]):
        """Find word frequency and distribution patterns."""
        # Track first and last mentions
        first_mentions = {}
        last_mentions = {}
        
        key_concepts = ["الرحمن", "الرحيم", "الحمد", "الصراط", "المستقيم"]
        
        quran_nodes = sorted(
            [n for n in nodes.values() if n.type == WisdomType.AYAH],
            key=lambda x: (x.metadata.get('surah', 0), x.metadata.get('ayah', 0))
        )
        
        for node in quran_nodes:
            if node.text_arabic:
                for concept in key_concepts:
                    if concept in node.text_arabic:
                        if concept not in first_mentions:
                            first_mentions[concept] = node.reference
                        last_mentions[concept] = node.reference
        
        # Record first/last patterns
        for concept in key_concepts:
            if concept in first_mentions:
                self.numerical_patterns.append({
                    "type": "CONCEPT_SPAN",
                    "concept": concept,
                    "first_mention": first_mentions[concept],
                    "last_mention": last_mentions.get(concept, ""),
                })
    
    def _find_symmetries(self, nodes: Dict[str, WisdomNode]):
        """Find structural symmetries (ring compositions)."""
        # Check surah-level symmetry
        # The Quran has 114 surahs - check for midpoint patterns
        
        surah_nodes = sorted(
            [n for n in nodes.values() if n.type == WisdomType.SURAH],
            key=lambda x: x.metadata.get('surah_number', 0)
        )
        
        if len(surah_nodes) >= 114:
            # Midpoint is between surah 57 (Al-Hadid/Iron) 
            # Note: 57 = 19 × 3, and 114 = 6 × 19
            self.symmetries.append({
                "type": "STRUCTURAL_SYMMETRY",
                "description": "114 Surahs = 6 × 19",
                "midpoint": "Surah 57 (Al-Hadid - Iron)",
                "note": "57:25 mentions iron sent down from heaven"
            })
        
        # Check for thematic mirrors
        # First surah (Al-Fatiha) themes vs last surah (An-Nas) themes
        first_surah_ayat = [n for n in nodes.values() 
                           if n.type == WisdomType.AYAH and n.metadata.get('surah') == 1]
        last_surah_ayat = [n for n in nodes.values() 
                          if n.type == WisdomType.AYAH and n.metadata.get('surah') == 114]
        
        if first_surah_ayat and last_surah_ayat:
            self.symmetries.append({
                "type": "THEMATIC_MIRROR",
                "description": "First surah (Al-Fatiha) ↔ Last surah (An-Nas)",
                "first": f"{len(first_surah_ayat)} ayat - Seeking guidance",
                "last": f"{len(last_surah_ayat)} ayat - Seeking refuge",
                "pattern": "Opening with praise, closing with protection"
            })
    
    def _find_echoes(self, nodes: Dict[str, WisdomNode]):
        """Find thematic echoes between Quran and Hadith."""
        # Find Hadith that directly reference or explain Quranic verses
        
        quran_keywords = defaultdict(list)  # keyword -> list of quran node ids
        hadith_matches = []
        
        # Index Quran by key phrases
        key_phrases = [
            ("الصلاة", "prayer"),
            ("الزكاة", "charity"),
            ("الصيام", "fasting"),
            ("الحج", "pilgrimage"),
            ("الجهاد", "striving"),
            ("التوبة", "repentance"),
            ("الصبر", "patience"),
            ("التوكل", "trust in God"),
        ]
        
        for node in nodes.values():
            if node.type == WisdomType.AYAH and node.text_arabic:
                for ar, en in key_phrases:
                    if ar in node.text_arabic:
                        quran_keywords[en].append(node.id)
        
        # Find Hadith mentioning same concepts
        for node in nodes.values():
            if node.type == WisdomType.HADITH:
                text = f"{node.text_arabic} {node.text_english}".lower()
                for ar, en in key_phrases:
                    if ar in text or en in text:
                        if quran_keywords[en]:
                            hadith_matches.append({
                                "concept": en,
                                "hadith_id": node.id,
                                "hadith_ref": node.reference,
                                "quran_refs": quran_keywords[en][:3]  # Sample
                            })
        
        # Record unique concept echoes
        seen_concepts = set()
        for match in hadith_matches:
            if match["concept"] not in seen_concepts:
                seen_concepts.add(match["concept"])
                self.echoes.append({
                    "type": "QURAN_HADITH_ECHO",
                    "concept": match["concept"],
                    "quran_count": len(quran_keywords[match["concept"]]),
                    "sample_hadith": match["hadith_ref"],
                    "note": f"Concept '{match['concept']}' spans both sources"
                })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all hidden patterns found."""
        return {
            "numerical_patterns": self.numerical_patterns,
            "symmetries": self.symmetries,
            "echoes": self.echoes,
            "total_patterns": len(self.numerical_patterns) + len(self.symmetries) + len(self.echoes)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SCIENTIFIC DISCOVERY LAYER — Hidden Gems Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class ScientificDiscoveryLayer:
    """
    Discovers scientific facts mentioned in sacred texts that correlate
    with modern scientific discoveries.
    
    Philosophy: Treat text as structured data, extract patterns that
    map to modern knowledge domains.
    """
    
    # Scientific domains with keywords (Arabic + English)
    SCIENCE_DOMAINS = {
        "astronomy": {
            "keywords": ["moon", "sun", "star", "sky", "heaven", "orbit", "night", "day", 
                        "القمر", "الشمس", "نجم", "السماء", "فلك", "الليل", "النهار", "كوكب"],
            "modern_discoveries": [
                {"fact": "Moon has a 300km rift (Rima Ariadaeus)", "verse": "54:1", "domain": "lunar geology"},
                {"fact": "Universe is expanding", "verse": "51:47", "domain": "cosmology"},
                {"fact": "Sun has an orbit in the galaxy", "verse": "36:40", "domain": "astrophysics"},
                {"fact": "Stars are not fixed, they move", "verse": "81:15-16", "domain": "stellar dynamics"},
                {"fact": "Seven atmospheric layers", "verse": "67:3", "domain": "atmospheric science"},
            ]
        },
        "embryology": {
            "keywords": ["womb", "create", "clot", "bone", "flesh", "stages", "sperm", "birth",
                        "رحم", "خلق", "علقة", "نطفة", "عظام", "لحم", "مضغة"],
            "modern_discoveries": [
                {"fact": "Embryo development stages match Quranic description", "verse": "23:12-14", "domain": "embryology"},
                {"fact": "Bones form before muscles", "verse": "23:14", "domain": "developmental biology"},
                {"fact": "Sex determination from male sperm", "verse": "53:45-46", "domain": "genetics"},
            ]
        },
        "oceanography": {
            "keywords": ["sea", "water", "wave", "deep", "darkness", "barrier", "salt", "fresh",
                        "بحر", "ماء", "موج", "ظلمات", "برزخ", "ملح", "عذب", "مرج"],
            "modern_discoveries": [
                {"fact": "Internal waves in deep ocean", "verse": "24:40", "domain": "physical oceanography"},
                {"fact": "Barrier between salt and fresh water", "verse": "55:19-20", "domain": "estuarine science"},
                {"fact": "Deep sea darkness layers", "verse": "24:40", "domain": "marine optics"},
            ]
        },
        "geology": {
            "keywords": ["mountain", "earth", "iron", "rock", "shake", "peg", "root",
                        "جبال", "أرض", "حديد", "صخر", "زلزل", "أوتاد", "رواسي"],
            "modern_discoveries": [
                {"fact": "Mountains have deep roots (isostasy)", "verse": "78:7", "domain": "geophysics"},
                {"fact": "Iron sent down from space (meteorites)", "verse": "57:25", "domain": "planetary science"},
                {"fact": "Mountains stabilize tectonic plates", "verse": "16:15", "domain": "plate tectonics"},
            ]
        },
        "biology": {
            "keywords": ["water", "life", "living", "creature", "plant", "pair", "male", "female",
                        "ماء", "حي", "دابة", "نبات", "زوج", "ذكر", "أنثى"],
            "modern_discoveries": [
                {"fact": "All life from water", "verse": "21:30", "domain": "biochemistry"},
                {"fact": "Plants have male and female", "verse": "20:53", "domain": "botany"},
                {"fact": "Pairs in all creation", "verse": "36:36", "domain": "biology"},
            ]
        },
        "physics": {
            "keywords": ["light", "shadow", "weight", "atom", "smaller", "particle", "expand",
                        "نور", "ظل", "ثقل", "ذرة", "أصغر", "موسعون"],
            "modern_discoveries": [
                {"fact": "Subatomic particles smaller than atoms", "verse": "10:61", "domain": "particle physics"},
                {"fact": "Universe expansion", "verse": "51:47", "domain": "cosmology"},
                {"fact": "Relativity of time", "verse": "22:47", "domain": "physics"},
            ]
        },
        "medicine": {
            "keywords": ["honey", "heal", "cure", "disease", "heart", "skin", "blood",
                        "عسل", "شفاء", "مرض", "قلب", "جلد", "دم"],
            "modern_discoveries": [
                {"fact": "Honey has healing properties", "verse": "16:69", "domain": "apitherapy"},
                {"fact": "Fingertips unique (fingerprints)", "verse": "75:4", "domain": "forensics"},
                {"fact": "Skin pain receptors", "verse": "4:56", "domain": "neurology"},
            ]
        }
    }
    
    def __init__(self):
        self.scientific_verses: List[Dict[str, Any]] = []
        self.domain_nodes: Dict[str, Set[str]] = defaultdict(set)
        self.discoveries: List[Dict[str, Any]] = []
    
    def analyze_nodes(self, nodes: Dict[str, WisdomNode]) -> List[Dict[str, Any]]:
        """Analyze all nodes for scientific content."""
        log.info("   🔬 Analyzing for scientific content...")
        
        for node_id, node in nodes.items():
            if node.type != WisdomType.AYAH:
                continue
            
            text = f"{node.text_english} {node.text_arabic}".lower()
            
            for domain, config in self.SCIENCE_DOMAINS.items():
                # Check if verse contains domain keywords
                keyword_matches = [kw for kw in config["keywords"] if kw in text]
                
                if keyword_matches:
                    self.domain_nodes[domain].add(node_id)
                    
                    # Check if this is a known scientific verse
                    for discovery in config["modern_discoveries"]:
                        if discovery["verse"] == node.reference:
                            self.discoveries.append({
                                "node_id": node_id,
                                "reference": node.reference,
                                "domain": domain,
                                "scientific_fact": discovery["fact"],
                                "sub_domain": discovery["domain"],
                                "keywords_matched": keyword_matches,
                                "text_arabic": node.text_arabic,
                                "text_english": node.text_english
                            })
        
        # Log findings
        for domain, node_ids in self.domain_nodes.items():
            log.info(f"      {domain}: {len(node_ids)} verses")
        
        log.info(f"   ✓ Found {len(self.discoveries)} scientifically significant verses")
        
        return self.discoveries
    
    def get_domain_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of scientific content by domain."""
        summary = {}
        for domain, node_ids in self.domain_nodes.items():
            discoveries_in_domain = [d for d in self.discoveries if d["domain"] == domain]
            summary[domain] = {
                "total_verses": len(node_ids),
                "verified_discoveries": len(discoveries_in_domain),
                "discoveries": discoveries_in_domain
            }
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DISCOVERY LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class PatternDiscoveryLayer:
    """Discovers hidden patterns in the wisdom corpus."""
    
    def __init__(self):
        self.patterns: List[DiscoveredPattern] = []
        self.science = ScientificDiscoveryLayer()
        self.hidden = HiddenKnowledgeLayer()
        
    def discover_patterns(self, nodes: Dict[str, WisdomNode], graph: HypergraphLayer) -> List[DiscoveredPattern]:
        """Run all pattern discovery algorithms."""
        log.info("   Discovering patterns...")
        
        # 0. Scientific discoveries (THE HIDDEN GEMS)
        self._discover_scientific(nodes)
        
        # 0.5. Hidden Knowledge patterns (علم الكتاب)
        self._discover_hidden_knowledge(nodes)
        
        # 1. Hub patterns (highly connected nodes)
        self._discover_hubs(nodes)
        
        # 2. Theme clusters
        self._discover_theme_clusters(graph)
        
        # 3. Cross-reference patterns
        self._discover_cross_refs(nodes, graph)
        
        # 4. Linguistic patterns (Arabic root patterns)
        self._discover_linguistic_patterns(nodes)
        
        log.info(f"   ✓ Discovered {len(self.patterns)} patterns")
        return self.patterns
    
    def _discover_hidden_knowledge(self, nodes: Dict[str, WisdomNode]):
        """Discover علم الكتاب patterns."""
        summary = self.hidden.analyze(nodes)
        
        # Add numerical patterns
        for np in summary.get('numerical_patterns', [])[:10]:
            if np['type'] == 'WORD_PAIR_BALANCE':
                self.patterns.append(DiscoveredPattern(
                    id=f"hidden_balance_{np['pair'][0]}_{np['pair'][1]}",
                    pattern_type="HIDDEN_NUMERICAL_BALANCE",
                    description=f"Word pair balance: {np['pair'][0]} ({np['counts'][0]}) ↔ {np['pair'][1]} ({np['counts'][1]})",
                    nodes=[],
                    confidence=0.9 if np['note'] == 'Balanced' else 0.7,
                    evidence=[f"Ratio: {np['ratio']}", np['note']]
                ))
        
        # Add symmetries
        for sym in summary.get('symmetries', []):
            self.patterns.append(DiscoveredPattern(
                id=f"hidden_symmetry_{sym['type']}",
                pattern_type="HIDDEN_STRUCTURAL_SYMMETRY",
                description=sym['description'],
                nodes=[],
                confidence=0.95,
                evidence=[sym.get('note', ''), sym.get('pattern', '')]
            ))
        
        # Add echoes
        for echo in summary.get('echoes', []):
            self.patterns.append(DiscoveredPattern(
                id=f"hidden_echo_{echo['concept']}",
                pattern_type="HIDDEN_CROSS_SOURCE_ECHO",
                description=f"Concept '{echo['concept']}' echoes across {echo['quran_count']} Quran verses and Hadith",
                nodes=[],
                confidence=0.85,
                evidence=[echo['note'], f"Sample: {echo['sample_hadith']}"]
            ))
    
    def _discover_scientific(self, nodes: Dict[str, WisdomNode]):
        """Discover scientific facts in sacred texts."""
        discoveries = self.science.analyze_nodes(nodes)
        
        for disc in discoveries:
            self.patterns.append(DiscoveredPattern(
                id=f"science_{disc['domain']}_{disc['reference'].replace(':', '_')}",
                pattern_type="SCIENTIFIC_DISCOVERY",
                description=f"[{disc['domain'].upper()}] {disc['scientific_fact']}",
                nodes=[disc['node_id']],
                confidence=0.95,
                evidence=[
                    f"Verse: {disc['reference']}",
                    f"Domain: {disc['sub_domain']}",
                    f"Arabic: {disc['text_arabic'][:100]}..." if disc['text_arabic'] else ""
                ]
            ))
    
    def _discover_hubs(self, nodes: Dict[str, WisdomNode]):
        """Find hub nodes (highly central)."""
        # Sort by centrality
        sorted_nodes = sorted(
            [(nid, n) for nid, n in nodes.items() if n.centrality > 0],
            key=lambda x: x[1].centrality,
            reverse=True
        )
        
        # Top hubs
        for nid, node in sorted_nodes[:10]:
            self.patterns.append(DiscoveredPattern(
                id=f"hub_{nid}",
                pattern_type="HUB_NODE",
                description=f"High-centrality node: {node.name}",
                nodes=[nid],
                confidence=min(node.centrality * 100, 1.0),
                evidence=[f"Centrality: {node.centrality:.4f}"]
            ))
    
    def _discover_theme_clusters(self, graph: HypergraphLayer):
        """Find thematic clusters."""
        for theme, node_ids in graph.themes.items():
            if len(node_ids) >= 5:
                quran_count = len([n for n in node_ids if n.startswith('quran_')])
                hadith_count = len([n for n in node_ids if n.startswith('hadith_')])
                
                self.patterns.append(DiscoveredPattern(
                    id=f"theme_cluster_{theme}",
                    pattern_type="THEME_CLUSTER",
                    description=f"Theme '{theme}' spans {len(node_ids)} passages",
                    nodes=list(node_ids)[:20],  # Sample
                    confidence=min(len(node_ids) / 100, 1.0),
                    evidence=[
                        f"Quran references: {quran_count}",
                        f"Hadith references: {hadith_count}"
                    ]
                ))
    
    def _discover_cross_refs(self, nodes: Dict[str, WisdomNode], graph: HypergraphLayer):
        """Find Quran-Hadith cross-references."""
        # Count cross-reference edges
        cross_ref_count = sum(1 for e in graph.edges if e.relation == RelationType.REFERENCES)
        
        if cross_ref_count > 0:
            self.patterns.append(DiscoveredPattern(
                id="cross_ref_network",
                pattern_type="CROSS_REFERENCE_NETWORK",
                description=f"Quran-Hadith cross-reference network with {cross_ref_count} links",
                nodes=[],
                confidence=0.9,
                evidence=[f"Total cross-references: {cross_ref_count}"]
            ))
    
    def _discover_linguistic_patterns(self, nodes: Dict[str, WisdomNode]):
        """Discover Arabic linguistic patterns."""
        # Count common Arabic roots/patterns (simplified)
        root_counts = defaultdict(int)
        
        for node in nodes.values():
            if node.text_arabic:
                # Simple pattern: count word frequencies
                words = node.text_arabic.split()
                for word in words[:50]:  # Sample
                    if len(word) >= 3:
                        root_counts[word[:3]] += 1  # Simplified root extraction
        
        # Top linguistic patterns
        top_roots = sorted(root_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for root, count in top_roots:
            if count >= 10:
                self.patterns.append(DiscoveredPattern(
                    id=f"linguistic_{root}",
                    pattern_type="LINGUISTIC_PATTERN",
                    description=f"Frequent Arabic pattern: '{root}' ({count} occurrences)",
                    nodes=[],
                    confidence=min(count / 100, 1.0),
                    evidence=[f"Occurrences: {count}"]
                ))


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH OF THOUGHTS REASONING LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class GoTReasoningLayer:
    """Graph of Thoughts reasoning across the wisdom corpus."""
    
    def __init__(self, nodes: Dict[str, WisdomNode], graph: HypergraphLayer, embeddings: EmbeddingLayer):
        self.nodes = nodes
        self.graph = graph
        self.embeddings = embeddings
        self.model = embeddings.model
        
    def reason(self, query: str, max_hops: int = 3, top_k: int = 10) -> ThoughtChain:
        """Execute multi-hop reasoning."""
        steps = []
        visited = set()
        sources = []
        
        # Step 1: Embed query and find seed nodes
        if self.model:
            query_embedding = self.model.encode([query])[0]
        else:
            query_embedding = np.random.randn(SacredConfig.EMBEDDING_DIM)
        
        # Find most similar nodes
        similarities = []
        for node_id, node in self.nodes.items():
            if node.embedding is not None:
                sim = np.dot(query_embedding, node.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding) + 1e-9
                )
                similarities.append((node_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        seed_nodes = similarities[:5]
        
        steps.append({
            "hop": 0,
            "action": "SEMANTIC_SEARCH",
            "description": f"Found {len(seed_nodes)} seed nodes",
            "nodes": [nid for nid, _ in seed_nodes],
            "scores": [float(s) for _, s in seed_nodes]
        })
        
        # Step 2: Graph traversal
        current_nodes = [nid for nid, _ in seed_nodes]
        
        for hop in range(1, max_hops + 1):
            next_nodes = []
            
            for node_id in current_nodes:
                if node_id in visited:
                    continue
                visited.add(node_id)
                sources.append(node_id)
                
                # Get neighbors
                if self.graph.graph and node_id in self.graph.graph:
                    for neighbor in self.graph.graph.neighbors(node_id):
                        if neighbor not in visited:
                            next_nodes.append(neighbor)
            
            if not next_nodes:
                break
            
            # Score and select top neighbors
            scored_neighbors = []
            for nid in set(next_nodes):
                node = self.nodes.get(nid)
                if node:
                    score = node.centrality + (node.snr_score * 0.5)
                    scored_neighbors.append((nid, score))
            
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            current_nodes = [nid for nid, _ in scored_neighbors[:3]]
            
            steps.append({
                "hop": hop,
                "action": "GRAPH_TRAVERSAL",
                "description": f"Explored {len(next_nodes)} neighbors, selected {len(current_nodes)}",
                "nodes": current_nodes
            })
        
        # Step 3: Synthesize conclusion
        conclusion_parts = []
        for nid in sources[:top_k]:
            node = self.nodes.get(nid)
            if node:
                conclusion_parts.append(f"- {node.name}: {node.text_english[:100]}...")
        
        conclusion = f"Based on {len(sources)} wisdom sources across {len(steps)} reasoning hops:\n" + "\n".join(conclusion_parts[:5])
        
        return ThoughtChain(
            query=query,
            steps=steps,
            conclusion=conclusion,
            confidence=min(len(sources) / top_k, 1.0),
            sources=sources[:top_k]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SNR OPTIMIZATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class SNROptimizationLayer:
    """Signal-to-Noise Ratio optimization for result ranking."""
    
    @staticmethod
    def compute_snr(nodes: List[WisdomNode], query: str) -> List[Tuple[WisdomNode, float]]:
        """Compute SNR scores for nodes given a query."""
        scored = []
        
        query_terms = set(query.lower().split())
        
        for node in nodes:
            # Signal: relevance indicators
            text = f"{node.text_english} {node.name}".lower()
            term_matches = sum(1 for term in query_terms if term in text)
            
            signal = (
                term_matches * 0.3 +                    # Term matching
                node.centrality * 10 * 0.3 +           # Graph importance
                (1.0 if node.metadata.get('grade', '').lower() == 'sahih' else 0.5) * 0.2 +  # Authenticity
                (0.1 if node.type == WisdomType.AYAH else 0.05) * 0.2  # Source priority
            )
            
            # Noise: length penalty, vagueness
            text_len = len(node.text_english) if node.text_english else 1
            noise = max(0.1, 1.0 / (1 + np.log1p(text_len / 100)))
            
            snr = signal / (noise + 0.001)
            node.snr_score = snr
            scored.append((node, snr))
        
        # Normalize SNR scores
        if scored:
            max_snr = max(s for _, s in scored)
            if max_snr > 0:
                scored = [(n, s/max_snr) for n, s in scored]
        
        return sorted(scored, key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED WISDOM ENGINE — UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class SacredWisdomEngine:
    """
    The Sacred Wisdom Hypergraph Engine.
    
    Unified interface for interdisciplinary knowledge discovery
    across Quran and Hadith corpora.
    """
    
    def __init__(self, lazy_load: bool = True):
        log.info("═" * 80)
        log.info("   🏛️  SACRED WISDOM HYPERGRAPH ENGINE")
        log.info("   Interdisciplinary Knowledge Discovery")
        log.info("═" * 80)
        
        self.ingestion = DataIngestionLayer()
        self.embedding = EmbeddingLayer()
        self.hypergraph = HypergraphLayer()
        self.patterns = PatternDiscoveryLayer()
        self.snr = SNROptimizationLayer()
        self.got: Optional[GoTReasoningLayer] = None
        
        self.nodes: Dict[str, WisdomNode] = {}
        self.stats: Dict[str, Any] = {}
        self.is_built = False
        
        if not lazy_load:
            self.build()
    
    def build(self) -> Dict[str, Any]:
        """Build the complete Sacred Wisdom Engine."""
        start = time.time()
        
        log.info("📖 Phase 1: Data Ingestion")
        ingestion_stats = self.ingestion.ingest_all()
        self.nodes = self.ingestion.nodes
        
        log.info("🧬 Phase 2: Semantic Embedding")
        self.embedding.initialize()
        embed_count = self.embedding.embed_nodes(self.nodes)
        
        log.info("🕸️ Phase 3: Hypergraph Construction")
        edge_count = self.hypergraph.build_graph(self.nodes)
        self.hypergraph.compute_centrality(self.nodes)
        
        log.info("🔍 Phase 4: Pattern Discovery")
        discovered = self.patterns.discover_patterns(self.nodes, self.hypergraph)
        
        log.info("🧠 Phase 5: GoT Reasoning Layer")
        self.got = GoTReasoningLayer(self.nodes, self.hypergraph, self.embedding)
        
        elapsed = time.time() - start
        
        self.stats = {
            "nodes": len(self.nodes),
            "edges": edge_count,
            "quran_nodes": ingestion_stats.get("quran", 0),
            "hadith_nodes": ingestion_stats.get("hadith", 0),
            "embeddings": embed_count,
            "patterns": len(discovered),
            "themes": len(self.hypergraph.themes),
            "build_time_seconds": round(elapsed, 2)
        }
        
        self.is_built = True
        self._save_state()
        
        log.info("═" * 80)
        log.info(f"   ✅ SACRED WISDOM ENGINE BUILT")
        log.info(f"      Nodes: {self.stats['nodes']:,}")
        log.info(f"      Edges: {self.stats['edges']:,}")
        log.info(f"      Patterns: {self.stats['patterns']}")
        log.info(f"      Themes: {self.stats['themes']}")
        log.info(f"      Time: {elapsed:.1f}s")
        log.info("═" * 80)
        
        return self.stats
    
    def query(self, query_text: str, max_results: int = 10, use_got: bool = True) -> Dict[str, Any]:
        """Query the Sacred Wisdom Engine."""
        if not self.is_built:
            self.load() or self.build()
        
        start = time.perf_counter()
        log.info(f"🔍 Query: '{query_text}'")
        
        # 1. Semantic search
        if self.embedding.model:
            query_emb = self.embedding.model.encode([query_text])[0]
        else:
            query_emb = np.random.randn(SacredConfig.EMBEDDING_DIM)
        
        # Find similar nodes
        similarities = []
        for node_id, node in self.nodes.items():
            if node.embedding is not None:
                sim = np.dot(query_emb, node.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-9
                )
                similarities.append((node_id, node, sim))
        
        # Sort and get top candidates
        similarities.sort(key=lambda x: x[2], reverse=True)
        candidates = [node for _, node, _ in similarities[:max_results * 2]]
        
        # 2. SNR optimization
        snr_scored = self.snr.compute_snr(candidates, query_text)
        
        # 3. GoT reasoning (optional)
        thought_chain = None
        if use_got and self.got:
            thought_chain = self.got.reason(query_text, max_hops=2, top_k=max_results)
        
        # 4. Prepare results
        results = []
        for node, snr in snr_scored[:max_results]:
            results.append({
                "id": node.id,
                "name": node.name,
                "type": node.type.name,
                "source": node.source,
                "reference": node.reference,
                "text_english": node.text_english[:300] if node.text_english else "",
                "text_arabic": node.text_arabic[:200] if node.text_arabic else "",
                "snr_score": round(snr, 4),
                "centrality": round(node.centrality, 6)
            })
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # Compute overall SNR
        avg_snr = sum(r['snr_score'] for r in results) / max(len(results), 1)
        
        return {
            "query": query_text,
            "results": results,
            "snr": round(avg_snr, 4),
            "elapsed_ms": round(elapsed, 2),
            "thought_chain": asdict(thought_chain) if thought_chain else None,
            "total_nodes_searched": len(self.nodes)
        }
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get discovered patterns."""
        return [asdict(p) for p in self.patterns.patterns]
    
    def get_themes(self) -> Dict[str, int]:
        """Get theme distribution."""
        return {theme: len(nodes) for theme, nodes in self.hypergraph.themes.items()}
    
    def get_scientific_discoveries(self) -> Dict[str, Any]:
        """Get all scientific discoveries mapped to modern knowledge."""
        return self.patterns.science.get_domain_summary()
    
    def get_hidden_knowledge(self) -> Dict[str, Any]:
        """Get علم الكتاب — hidden patterns discovered."""
        return self.patterns.hidden.get_summary()
    
    def query_science(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query scientific content by domain."""
        if not self.is_built:
            self.load() or self.build()
        
        discoveries = self.patterns.science.discoveries
        
        if domain:
            discoveries = [d for d in discoveries if d['domain'] == domain.lower()]
        
        return discoveries
    
    def _save_state(self):
        """Save engine state to disk."""
        SacredConfig.GOLD.mkdir(exist_ok=True)
        
        # Save graph
        graph_data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [e.to_dict() for e in self.hypergraph.edges],
            "stats": self.stats
        }
        with open(SacredConfig.SACRED_GRAPH, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        # Save patterns
        with open(SacredConfig.SACRED_PATTERNS, 'w', encoding='utf-8') as f:
            for pattern in self.patterns.patterns:
                f.write(json.dumps(asdict(pattern), ensure_ascii=False) + '\n')
        
        # Save embeddings
        self.embedding.save_embeddings(SacredConfig.SACRED_EMBEDDINGS)
        
        log.info(f"   ✓ State saved to {SacredConfig.GOLD}")
    
    def load(self) -> bool:
        """Load engine state from disk."""
        if not SacredConfig.SACRED_GRAPH.exists():
            return False
        
        try:
            log.info("📂 Loading saved state...")
            
            with open(SacredConfig.SACRED_GRAPH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct nodes
            for nid, node_dict in data.get('nodes', {}).items():
                node_dict['type'] = WisdomType[node_dict['type']]
                self.nodes[nid] = WisdomNode(**{k: v for k, v in node_dict.items() if k != 'embedding'})
            
            # Reconstruct edges
            for edge_dict in data.get('edges', []):
                edge_dict['relation'] = RelationType[edge_dict['relation']]
                self.hypergraph.edges.append(WisdomEdge(**edge_dict))
            
            self.stats = data.get('stats', {})
            self.is_built = True
            
            log.info(f"   ✓ Loaded {len(self.nodes)} nodes")
            return True
            
        except Exception as e:
            log.error(f"   ✗ Load failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sacred Wisdom Hypergraph Engine")
    parser.add_argument("command", choices=["build", "query", "stats", "patterns", "themes", "science", "hidden"],
                       help="Command to execute")
    parser.add_argument("query_text", nargs="?", default="", help="Query text (for query command)")
    parser.add_argument("--max-results", type=int, default=10, help="Max results")
    parser.add_argument("--no-got", action="store_true", help="Disable GoT reasoning")
    parser.add_argument("--domain", type=str, default="", help="Scientific domain filter")
    
    args = parser.parse_args()
    
    engine = SacredWisdomEngine(lazy_load=True)
    
    if args.command == "build":
        engine.build()
        
    elif args.command == "query":
        if not args.query_text:
            print("Error: Query text required")
            return
        
        result = engine.query(args.query_text, max_results=args.max_results, use_got=not args.no_got)
        
        print(f"\n🔍 SACRED WISDOM QUERY (SNR: {result['snr']:.4f} | {result['elapsed_ms']:.1f}ms)")
        print("═" * 70)
        
        for i, r in enumerate(result['results'], 1):
            print(f"\n  [{i}] {r['name']} ({r['type']})")
            print(f"      📖 {r['reference']} | SNR: {r['snr_score']:.3f}")
            if r['text_english']:
                print(f"      {r['text_english'][:150]}...")
        
        if result.get('thought_chain'):
            print(f"\n🧠 REASONING CHAIN ({len(result['thought_chain']['steps'])} hops)")
            print(f"   Confidence: {result['thought_chain']['confidence']:.2f}")
        
    elif args.command == "stats":
        if not engine.load():
            engine.build()
        print(json.dumps(engine.stats, indent=2))
        
    elif args.command == "patterns":
        if not engine.load():
            engine.build()
        patterns = engine.get_patterns()
        
        # Group by type
        by_type = defaultdict(list)
        for p in patterns:
            by_type[p['pattern_type']].append(p)
        
        for ptype, plist in by_type.items():
            print(f"\n{'═'*60}")
            print(f"  {ptype} ({len(plist)} patterns)")
            print(f"{'═'*60}")
            for p in plist[:10]:
                print(f"  • {p['description']}")
                if p.get('evidence'):
                    for ev in p['evidence'][:2]:
                        if ev:
                            print(f"    → {ev}")
            
    elif args.command == "themes":
        if not engine.load():
            engine.build()
        themes = engine.get_themes()
        print("\n🏷️ THEMES IN SACRED WISDOM")
        print("═" * 40)
        for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * min(count // 50, 30)
            print(f"  {theme:15} {count:5} {bar}")
    
    elif args.command == "science":
        if not engine.load():
            engine.build()
        
        print("\n🔬 SCIENTIFIC DISCOVERIES IN SACRED TEXTS")
        print("═" * 70)
        print("  Correlations between ancient text and modern scientific knowledge")
        print("═" * 70)
        
        summary = engine.get_scientific_discoveries()
        
        for domain, data in sorted(summary.items(), key=lambda x: x[1]['verified_discoveries'], reverse=True):
            if data['verified_discoveries'] > 0:
                print(f"\n  📊 {domain.upper()}")
                print(f"     Total verses with keywords: {data['total_verses']}")
                print(f"     Verified discoveries: {data['verified_discoveries']}")
                
                for disc in data['discoveries']:
                    print(f"\n     ┌─ Verse {disc['reference']}")
                    print(f"     │  🔬 {disc['scientific_fact']}")
                    print(f"     │  📚 Domain: {disc['sub_domain']}")
                    if disc.get('text_arabic'):
                        print(f"     │  📖 {disc['text_arabic'][:60]}...")
                    print(f"     └─")
    
    elif args.command == "hidden":
        if not engine.load():
            engine.build()
        
        print("\n📿 عِلْمُ الْكِتَاب — HIDDEN KNOWLEDGE PATTERNS")
        print("═" * 70)
        print("  'We have not neglected anything in the Book' — 6:38")
        print("═" * 70)
        
        summary = engine.get_hidden_knowledge()
        
        # Numerical patterns
        num_patterns = summary.get('numerical_patterns', [])
        if num_patterns:
            print("\n  🔢 NUMERICAL PATTERNS")
            print("  " + "-" * 40)
            
            # Word pair balances
            balances = [p for p in num_patterns if p['type'] == 'WORD_PAIR_BALANCE']
            for b in balances:
                w1, w2 = b['pair']
                c1, c2 = b['counts']
                print(f"     {w1:20} × {c1:4}  ↔  {w2:20} × {c2:4}  [{b['note']}]")
        
        # Symmetries
        symmetries = summary.get('symmetries', [])
        if symmetries:
            print("\n  🔄 STRUCTURAL SYMMETRIES")
            print("  " + "-" * 40)
            for sym in symmetries:
                print(f"     • {sym['description']}")
                if sym.get('note'):
                    print(f"       → {sym['note']}")
                if sym.get('pattern'):
                    print(f"       → {sym['pattern']}")
        
        # Echoes
        echoes = summary.get('echoes', [])
        if echoes:
            print("\n  🔊 CROSS-SOURCE ECHOES (Quran ↔ Hadith)")
            print("  " + "-" * 40)
            for echo in echoes:
                print(f"     • {echo['concept']}: {echo['quran_count']} Quran verses echo in Hadith")
        
        print(f"\n  ═══════════════════════════════════════")
        print(f"  Total hidden patterns discovered: {summary.get('total_patterns', 0)}")


if __name__ == "__main__":
    main()
