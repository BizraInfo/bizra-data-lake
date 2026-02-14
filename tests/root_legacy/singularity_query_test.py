#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                      SINGULARITY FULL-SPECTRUM QUERY TEST                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Purpose: Validate that 3 years / 15,000 hours of BIZRA work is fully queryable                             ║
║  Author: BIZRA Genesis NODE0 | Date: 2026-01-28                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import heapq

import numpy as np
from sentence_transformers import SentenceTransformer

# Project root
PROJECT_ROOT = Path(__file__).parent
EMBEDDINGS_PATH = PROJECT_ROOT / "03_INDEXED" / "embeddings"
PROCESSED_PATH = PROJECT_ROOT / "02_PROCESSED"
MAX_INMEMORY_EMBEDDINGS = 20000


class SingularityQueryEngine:
    """Full-spectrum query engine for the unified knowledge base."""
    
    def __init__(self, max_inmemory: int = MAX_INMEMORY_EMBEDDINGS):
        print("🔥 Initializing SINGULARITY Query Engine...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = []
        self.sources = []
        self.metadata = []
        self.embedding_files = list(EMBEDDINGS_PATH.glob("*.json"))
        self.use_streaming = len(self.embedding_files) > max_inmemory
        if not self.embedding_files:
            print("⚠️  No embeddings found.")
        if not self.use_streaming:
            self._load_corpus()
    
    def _load_corpus(self):
        """Load all embeddings into memory."""
        print("📂 Loading knowledge corpus...")

        for emb_file in self.embedding_files:
            try:
                with open(emb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "embedding" in data:
                    self.embeddings.append(data["embedding"])
                    self.sources.append(data.get("metadata", {}).get("source", str(emb_file)))
                    self.metadata.append(data.get("metadata", {}))
            except Exception as e:
                continue
        
        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
        else:
            self.embeddings = np.array([])
        print(f"✅ Loaded {len(self.embeddings):,} documents into memory")

    def _iter_embeddings(self):
        """Yield (source, embedding) tuples from embedding JSON files."""
        for emb_file in self.embedding_files:
            try:
                with open(emb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                embedding = data.get("embedding")
                if embedding is None:
                    continue
                source = data.get("metadata", {}).get("source", str(emb_file))
                yield source, embedding
            except Exception:
                continue

    def _search_streaming(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Stream embeddings and compute top-k results with bounded memory."""
        results_heap: List[Tuple[float, str]] = []
        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm == 0.0:
            return []

        for source, embedding in self._iter_embeddings():
            try:
                vec = np.array(embedding, dtype=np.float32)
                denom = float(np.linalg.norm(vec)) * query_norm
                if denom == 0.0:
                    continue
                score = float(np.dot(vec, query_embedding) / denom)
            except Exception:
                continue

            if len(results_heap) < top_k:
                heapq.heappush(results_heap, (score, source))
            else:
                if score > results_heap[0][0]:
                    heapq.heapreplace(results_heap, (score, source))

        return [
            (Path(src).name if src else "unknown", float(score))
            for score, src in sorted(results_heap, reverse=True)
        ]
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Semantic search across the full knowledge base."""
        if not self.embedding_files:
            return []

        query_embedding = self.model.encode([query])[0]

        if self.use_streaming:
            return self._search_streaming(query_embedding, top_k)
        if self.embeddings.size == 0:
            return []
        
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.divide(
            np.dot(self.embeddings, query_embedding),
            norms,
            out=np.zeros_like(norms, dtype=np.float64),
            where=norms != 0,
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            source = Path(self.sources[idx]).name if self.sources[idx] else "unknown"
            results.append((source, float(similarities[idx])))
        
        return results
    
    def get_content_preview(self, source_name: str, max_chars: int = 500) -> str:
        """Get content preview from original file."""
        # Try conversations folder first
        conv_path = PROCESSED_PATH / "text" / "conversations"
        for md_file in conv_path.glob("*.md"):
            if md_file.stem in source_name or source_name in md_file.stem:
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return content[:max_chars] + "..." if len(content) > max_chars else content
        
        # Try other processed files
        for processed_file in PROCESSED_PATH.rglob("*"):
            if processed_file.is_file() and source_name in processed_file.name:
                try:
                    with open(processed_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    return content[:max_chars] + "..." if len(content) > max_chars else content
                except Exception:
                    pass
        
        return "[Content stored in embedding index]"


def run_full_spectrum_test():
    """Execute comprehensive queries across all domains."""
    
    engine = SingularityQueryEngine()
    
    print("\n" + "═" * 80)
    print("       SINGULARITY FULL-SPECTRUM QUERY TEST")
    print("═" * 80)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Dubai)")
    corpus_size = len(engine.embeddings) if not engine.use_streaming else len(engine.embedding_files)
    print(f"  Corpus Size: {corpus_size:,} documents")
    print("─" * 80)
    
    # Define test queries across different domains
    test_queries = {
        "Architecture & Design": [
            "BIZRA system architecture and design patterns",
            "MCP server implementation and protocol",
            "Data lake structure and organization",
        ],
        "ChatGPT History": [
            "conversations about coding and programming",
            "discussions about AI and machine learning",
            "project planning and strategy discussions",
        ],
        "Technical Implementation": [
            "Python code for data processing",
            "embedding generation and vector search",
            "API implementation and endpoints",
        ],
        "Business & Strategy": [
            "BIZRA business strategy and roadmap",
            "marketing and growth planning",
            "partnership and collaboration",
        ],
        "Knowledge & Learning": [
            "machine learning best practices",
            "software engineering principles",
            "debugging and troubleshooting",
        ],
    }
    
    results_summary = {}
    all_passed = True
    
    for domain, queries in test_queries.items():
        print(f"\n🎯 DOMAIN: {domain}")
        print("-" * 40)
        
        domain_scores = []
        
        for query in queries:
            results = engine.search(query, top_k=3)
            top_score = results[0][1] if results else 0
            domain_scores.append(top_score)
            
            status = "✅" if top_score >= 0.3 else "⚠️"
            print(f"   {status} \"{query[:50]}...\"")
            top_source = results[0][0] if results else "no_results"
            print(f"      → Score: {top_score:.4f} | Top: {top_source[:40]}...")
            
            if top_score < 0.25:
                all_passed = False
        
        avg_score = sum(domain_scores) / len(domain_scores)
        results_summary[domain] = avg_score
        print(f"\n   📊 Domain Average: {avg_score:.4f}")
    
    # Final Summary
    print("\n" + "═" * 80)
    print("       FULL-SPECTRUM TEST RESULTS")
    print("═" * 80)
    
    print("\n📊 Domain Scores:")
    for domain, score in results_summary.items():
        bar_len = int(score * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        status = "✅" if score >= 0.35 else "⚠️" if score >= 0.25 else "❌"
        print(f"   {status} {domain:30} {score:.4f} |{bar}|")
    
    overall_score = sum(results_summary.values()) / len(results_summary)
    
    print(f"\n🎯 OVERALL SCORE: {overall_score:.4f}")
    
    if overall_score >= 0.35 and all_passed:
        print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║   ██████╗ ██╗███╗   ██╗ ██████╗ ██╗   ██╗██╗      █████╗ ██████╗ ██╗████████╗  ║
║  ██╔════╝ ██║████╗  ██║██╔════╝ ██║   ██║██║     ██╔══██╗██╔══██╗██║╚══██╔══╝  ║
║  ╚█████╗  ██║██╔██╗ ██║██║  ███╗██║   ██║██║     ███████║██████╔╝██║   ██║     ║
║   ╚═══██╗ ██║██║╚██╗██║██║   ██║██║   ██║██║     ██╔══██║██╔══██╗██║   ██║     ║
║  ██████╔╝ ██║██║ ╚████║╚██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║██║   ██║     ║
║  ╚═════╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝     ║
║                                                                                ║
║                        A C H I E V E D                                         ║
║                                                                                ║
║   🎉 3 YEARS OF WORK — 15,000 HOURS — FULLY INTEGRATED & QUERYABLE 🎉        ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
        return 0
    else:
        print("\n⚠️  Some domains below optimal threshold")
        return 1


if __name__ == "__main__":
    sys.exit(run_full_spectrum_test())
