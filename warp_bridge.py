# BIZRA WARP Bridge v1.0
# Multi-Vector Contextualized Retrieval Engine Adapter
# Standing on Giants: Stanford ColBERTv2/PLAID + Google DeepMind XTR
# Integrates XTR-WARP into BIZRA Data Lake as high-accuracy retrieval option

"""
WARP Bridge Architecture:

┌─────────────────────────────────────────────────────────────────────────┐
│                           BIZRA NEXUS                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ FAISS HNSW   │  │  WARP Engine │  │  Other Engines               │  │
│  │ (384-dim)    │  │  (ColBERT)   │  │  (Sacred, Graph, etc.)       │  │
│  │ [FAST]       │  │  [ACCURATE]  │  │                              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────────┘  │
│         │                 │                                             │
│         └─────────────────┼─────────────────────────────────────────────┘
│                           │                                              │
│                           ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      WARP BRIDGE ADAPTER                          │   │
│  │  • Lazy loading (only init when needed)                          │   │
│  │  • Index building from BIZRA chunks                              │   │
│  │  • Query → ColBERT encoding → WARP search                        │   │
│  │  • Result normalization to BIZRA format                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import sys
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Add xtr-warp to path
WARP_PATH = Path(__file__).parent / "xtr-warp"
if WARP_PATH.exists() and str(WARP_PATH) not in sys.path:
    sys.path.insert(0, str(WARP_PATH))

# Import BIZRA config
from bizra_config import (
    WARP_INDEX_ROOT, WARP_EXPERIMENT_ROOT, WARP_CHECKPOINT,
    WARP_NBITS, WARP_NPROBE, WARP_T_PRIME, WARP_BOUND,
    WARP_ENABLED, WARP_USE_GPU, WARP_RUNTIME, WARP_FUSED_EXT,
    CHUNKS_TABLE_PATH, INDEXED_PATH, GOLD_PATH,
    SNR_THRESHOLD, IHSAN_CONSTRAINT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | WARP-BRIDGE | %(message)s',
    handlers=[
        logging.FileHandler(INDEXED_PATH / "warp_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WARP-BRIDGE")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class WARPStatus(Enum):
    """Status of the WARP engine."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    INDEXING = "indexing"
    SEARCHING = "searching"
    ERROR = "error"


@dataclass
class WARPResult:
    """Single result from WARP retrieval."""
    doc_id: str
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WARPQueryResponse:
    """Complete response from WARP query."""
    query: str
    results: List[WARPResult]
    total_results: int
    execution_time_ms: float
    snr_estimate: float
    engine: str = "warp"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WARPIndexStats:
    """Statistics about a WARP index."""
    index_name: str
    num_documents: int
    num_passages: int
    index_size_bytes: int
    nbits: int
    checkpoint: str
    created_at: str
    last_query: Optional[str] = None


# ============================================================================
# WARP BRIDGE ADAPTER
# ============================================================================

class WARPBridge:
    """
    WARP Bridge - Connects XTR-WARP to BIZRA Data Lake
    
    Features:
    - Lazy initialization (only loads when first query)
    - Automatic index building from BIZRA chunks
    - Query routing with SNR estimation
    - Result normalization to BIZRA format
    - Health monitoring and self-healing
    """
    
    def __init__(
        self,
        index_name: str = "bizra_chunks",
        checkpoint: str = None,
        nbits: int = None,
        lazy_init: bool = True
    ):
        self.index_name = index_name
        self.checkpoint = checkpoint or WARP_CHECKPOINT
        self.nbits = nbits or WARP_NBITS
        
        self.status = WARPStatus.OFFLINE
        self.searcher = None
        self.indexer = None
        self.index_stats: Optional[WARPIndexStats] = None
        
        self._warp_available = False
        self._check_warp_availability()
        
        if not lazy_init and self._warp_available:
            self._initialize()
    
    def _check_warp_availability(self) -> bool:
        """Check if WARP dependencies are available."""
        try:
            import torch
            from warp.searcher import Searcher
            from warp.indexer import Indexer
            from warp.engine.config import WARPRunConfig
            self._warp_available = True
            logger.info("[OK] WARP dependencies available")
            return True
        except ImportError as e:
            logger.warning(f"[WARN] WARP not available: {e}")
            self._warp_available = False
            return False
    
    def _initialize(self) -> bool:
        """Initialize WARP searcher if index exists."""
        if not self._warp_available:
            return False
        
        self.status = WARPStatus.INITIALIZING
        
        try:
            index_path = WARP_INDEX_ROOT / self.index_name
            
            if not index_path.exists():
                logger.info(f"Index {self.index_name} not found. Build with build_index().")
                self.status = WARPStatus.OFFLINE
                return False
            
            from warp.searcher import Searcher
            from warp.infra import Run, RunConfig
            
            with Run().context(RunConfig(nranks=1)):
                self.searcher = Searcher(
                    index=self.index_name,
                    checkpoint=self.checkpoint,
                    index_root=str(WARP_INDEX_ROOT),
                    warp_engine=True,
                    verbose=1
                )
            
            self.status = WARPStatus.READY
            self._load_index_stats()
            logger.info(f"✓ WARP searcher initialized for index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ WARP initialization failed: {e}")
            self.status = WARPStatus.ERROR
            return False
    
    def _load_index_stats(self):
        """Load index statistics from metadata."""
        stats_path = WARP_INDEX_ROOT / self.index_name / "bizra_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                data = json.load(f)
                self.index_stats = WARPIndexStats(**data)
    
    def _save_index_stats(self, stats: WARPIndexStats):
        """Save index statistics to metadata."""
        stats_path = WARP_INDEX_ROOT / self.index_name / "bizra_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({
                "index_name": stats.index_name,
                "num_documents": stats.num_documents,
                "num_passages": stats.num_passages,
                "index_size_bytes": stats.index_size_bytes,
                "nbits": stats.nbits,
                "checkpoint": stats.checkpoint,
                "created_at": stats.created_at,
                "last_query": stats.last_query
            }, f, indent=2)
        self.index_stats = stats
    
    def build_index(
        self,
        collection_path: Path = None,
        overwrite: bool = False
    ) -> bool:
        """
        Build WARP index from BIZRA chunks.
        
        Args:
            collection_path: Path to collection TSV (defaults to auto-generated from chunks.parquet)
            overwrite: Whether to overwrite existing index
            
        Returns:
            True if successful
        """
        if not self._warp_available:
            logger.error("WARP not available. Cannot build index.")
            return False
        
        self.status = WARPStatus.INDEXING
        start_time = time.time()
        
        try:
            # Generate collection TSV from chunks.parquet if not provided
            if collection_path is None:
                collection_path = self._export_chunks_to_tsv()
            
            if not collection_path.exists():
                logger.error(f"Collection not found: {collection_path}")
                return False
            
            from warp.indexer import Indexer
            from warp.infra import Run, RunConfig, ColBERTConfig
            
            with Run().context(RunConfig(nranks=1, experiment="bizra")):
                config = ColBERTConfig(
                    nbits=self.nbits,
                    doc_maxlen=512,
                    query_maxlen=128,
                    index_bsize=64,
                )
                
                indexer = Indexer(checkpoint=self.checkpoint, config=config, verbose=2)
                index_path = indexer.index(
                    name=self.index_name,
                    collection=str(collection_path),
                    overwrite=overwrite
                )
            
            elapsed = time.time() - start_time
            
            # Calculate index size
            index_size = sum(
                f.stat().st_size for f in Path(index_path).rglob('*') if f.is_file()
            )
            
            # Count documents
            num_docs = sum(1 for _ in open(collection_path, 'r'))
            
            # Save stats
            from datetime import datetime
            self._save_index_stats(WARPIndexStats(
                index_name=self.index_name,
                num_documents=num_docs,
                num_passages=num_docs,  # 1:1 for now
                index_size_bytes=index_size,
                nbits=self.nbits,
                checkpoint=self.checkpoint,
                created_at=datetime.now().isoformat()
            ))
            
            logger.info(f"✓ Index built in {elapsed:.2f}s: {num_docs} docs, {index_size / 1e6:.2f} MB")
            self.status = WARPStatus.READY
            
            # Initialize searcher
            self._initialize()
            return True
            
        except Exception as e:
            logger.error(f"✗ Index build failed: {e}")
            self.status = WARPStatus.ERROR
            return False
    
    def _export_chunks_to_tsv(self) -> Path:
        """Export BIZRA chunks.parquet to WARP-compatible TSV."""
        import pandas as pd
        
        output_path = WARP_INDEX_ROOT / f"{self.index_name}_collection.tsv"
        
        if not CHUNKS_TABLE_PATH.exists():
            raise FileNotFoundError(f"Chunks table not found: {CHUNKS_TABLE_PATH}")
        
        logger.info(f"Exporting chunks from {CHUNKS_TABLE_PATH}")
        
        df = pd.read_parquet(CHUNKS_TABLE_PATH)
        
        # WARP expects: pid\ttext format
        tsv_data = []
        for idx, row in df.iterrows():
            text = row.get('chunk_text', row.get('text', ''))
            # Clean text for TSV
            text = text.replace('\t', ' ').replace('\n', ' ').strip()
            if text:
                tsv_data.append(f"{idx}\t{text}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tsv_data))
        
        logger.info(f"✓ Exported {len(tsv_data)} chunks to {output_path}")
        return output_path
    
    def search(
        self,
        query: str,
        k: int = 10,
        nprobe: int = None
    ) -> WARPQueryResponse:
        """
        Search using WARP multi-vector retrieval.
        
        Args:
            query: Search query text
            k: Number of results to return
            nprobe: Number of probes for approximate search
            
        Returns:
            WARPQueryResponse with ranked results
        """
        if not self._warp_available:
            return self._fallback_response(query, "WARP not available")
        
        if self.status != WARPStatus.READY:
            if not self._initialize():
                return self._fallback_response(query, "WARP not initialized")
        
        self.status = WARPStatus.SEARCHING
        start_time = time.time()
        
        try:
            # Execute WARP search
            ranking = self.searcher.search(query, k=k)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Convert to BIZRA format
            results = []
            for rank, (pid, score) in enumerate(zip(ranking.pids, ranking.scores)):
                results.append(WARPResult(
                    doc_id=str(pid),
                    chunk_id=f"warp_{pid}",
                    text=self._get_passage_text(pid),
                    score=float(score),
                    rank=rank + 1
                ))
            
            # Estimate SNR based on score distribution
            snr_estimate = self._estimate_snr(ranking.scores)
            
            # Update last query time
            if self.index_stats:
                self.index_stats.last_query = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._save_index_stats(self.index_stats)
            
            self.status = WARPStatus.READY
            
            return WARPQueryResponse(
                query=query,
                results=results,
                total_results=len(results),
                execution_time_ms=elapsed_ms,
                snr_estimate=snr_estimate,
                metadata={
                    "nprobe": nprobe or WARP_NPROBE,
                    "checkpoint": self.checkpoint,
                    "nbits": self.nbits
                }
            )
            
        except Exception as e:
            logger.error(f"✗ Search failed: {e}")
            self.status = WARPStatus.ERROR
            return self._fallback_response(query, str(e))
    
    def _get_passage_text(self, pid: int) -> str:
        """Retrieve passage text by PID from collection."""
        try:
            if hasattr(self.searcher, 'collection') and self.searcher.collection:
                return self.searcher.collection[pid]
        except:
            pass
        return f"[Passage {pid}]"
    
    def _estimate_snr(self, scores: List[float]) -> float:
        """Estimate SNR from score distribution."""
        if not scores:
            return 0.0
        
        scores = np.array(scores)
        
        # Signal: mean of top scores
        signal = np.mean(scores[:min(3, len(scores))])
        
        # Noise: variance in tail
        if len(scores) > 3:
            noise = np.std(scores[3:]) + 0.01
        else:
            noise = 0.1
        
        # Normalize to [0, 1]
        snr = signal / (signal + noise)
        return round(float(snr), 4)
    
    def _fallback_response(self, query: str, error: str) -> WARPQueryResponse:
        """Return empty response with error info."""
        return WARPQueryResponse(
            query=query,
            results=[],
            total_results=0,
            execution_time_ms=0,
            snr_estimate=0.0,
            metadata={"error": error, "fallback": True}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "engine": "warp",
            "status": self.status.value,
            "available": self._warp_available,
            "index_name": self.index_name,
            "checkpoint": self.checkpoint,
            "nbits": self.nbits,
            "stats": {
                "num_documents": self.index_stats.num_documents if self.index_stats else 0,
                "index_size_mb": (self.index_stats.index_size_bytes / 1e6) if self.index_stats else 0,
                "last_query": self.index_stats.last_query if self.index_stats else None
            } if self.index_stats else None
        }
    
    def health_check(self) -> Tuple[bool, str]:
        """Check engine health."""
        if not self._warp_available:
            return False, "WARP dependencies not installed"
        
        if self.status == WARPStatus.ERROR:
            return False, "Engine in error state"
        
        if self.status == WARPStatus.READY:
            return True, "Ready"
        
        if self.status == WARPStatus.OFFLINE:
            return False, "Index not loaded"
        
        return True, self.status.value


# ============================================================================
# NEXUS ADAPTER (For integration with bizra_nexus.py)
# ============================================================================

class WARPEngineAdapter:
    """
    Adapter for BIZRA Nexus integration.
    
    Follows the EngineAdapter protocol expected by bizra_nexus.py
    """
    
    def __init__(self, name: str = "WARP"):
        self.name = name
        self.bridge = WARPBridge(lazy_init=True)
        self.status = "offline"
        self.health = None
    
    def initialize(self) -> bool:
        """Initialize the engine."""
        success = self.bridge._initialize()
        self.status = "healthy" if success else "offline"
        return success
    
    def query(self, query_text: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        response = self.bridge.search(query_text, k=max_results)
        
        # Convert to Nexus format
        return [
            {
                "id": r.chunk_id,
                "doc_id": r.doc_id,
                "text": r.text,
                "score": r.score,
                "rank": r.rank,
                "source": "warp"
            }
            for r in response.results
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.bridge.get_stats()
    
    def check_health(self) -> Dict[str, Any]:
        """Check engine health."""
        healthy, message = self.bridge.health_check()
        return {
            "name": self.name,
            "status": "healthy" if healthy else "degraded",
            "message": message,
            "stats": self.get_stats()
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for WARP Bridge."""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog='BIZRA WARP Bridge',
        description='Multi-Vector Contextualized Retrieval for BIZRA Data Lake'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build WARP index from chunks')
    build_parser.add_argument('--name', default='bizra_chunks', help='Index name')
    build_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search the index')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--name', default='bizra_chunks', help='Index name')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show index statistics')
    stats_parser.add_argument('--name', default='bizra_chunks', help='Index name')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        bridge = WARPBridge(index_name=args.name)
        success = bridge.build_index(overwrite=args.overwrite)
        sys.exit(0 if success else 1)
    
    elif args.command == 'search':
        bridge = WARPBridge(index_name=args.name, lazy_init=False)
        response = bridge.search(args.query, k=args.k)
        
        print(f"\n🔍 Query: {response.query}")
        print(f"⏱️  Time: {response.execution_time_ms:.2f}ms")
        print(f"📊 SNR: {response.snr_estimate:.4f}")
        print(f"\n{'='*60}\n")
        
        for r in response.results:
            print(f"[{r.rank}] Score: {r.score:.4f} | Doc: {r.doc_id}")
            print(f"    {r.text[:200]}...")
            print()
    
    elif args.command == 'stats':
        bridge = WARPBridge(index_name=args.name)
        stats = bridge.get_stats()
        print(json.dumps(stats, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
