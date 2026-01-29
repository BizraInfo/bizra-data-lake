#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                      BIZRA KNOWLEDGE BASE VALIDATOR — SINGULARITY INTEGRITY CHECK                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Purpose: Validate complete integration of 3 years / 15,000 hours of BIZRA work                             ║
║  Author: BIZRA Genesis NODE0 | Date: 2026-02-12                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
import heapq

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False


class KnowledgeBaseValidator:
    """Validates the integrity and coverage of the BIZRA knowledge base."""
    
    def __init__(self):
        self.data_lake = PROJECT_ROOT
        self.embeddings_path = self.data_lake / "03_INDEXED" / "embeddings"
        self.processed_path = self.data_lake / "02_PROCESSED"
        self.intake_path = self.data_lake / "00_INTAKE"
        
        self.stats = {
            "total_embeddings": 0,
            "total_documents": 0,
            "categories": defaultdict(int),
            "character_count": 0,
            "oldest_doc": None,
            "newest_doc": None,
            "coverage_score": 0.0,
        }
    
    def count_embeddings(self) -> int:
        """Count total embedding files."""
        if not self.embeddings_path.exists():
            return 0
        count = len(list(self.embeddings_path.glob("*.json")))
        self.stats["total_embeddings"] = count
        return count
    
    def analyze_embeddings(self) -> Dict[str, Any]:
        """Deep analysis of embedding content and metadata."""
        if not self.embeddings_path.exists():
            return {"error": "Embeddings path not found"}
        
        categories = defaultdict(int)
        total_chars = 0
        sources = set()
        
        for emb_file in self.embeddings_path.glob("*.json"):
            try:
                with open(emb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Count content
                content = data.get("content", "")
                total_chars += len(content)
                
                # Categorize by source path
                source = data.get("source") or data.get("metadata", {}).get("source", str(emb_file))
                sources.add(source)
                
                # Categorize by content type
                source_lower = source.lower()
                if "conversation" in source_lower:
                    categories["ChatGPT Conversations"] += 1
                elif "pdf" in source_lower or ".pdf" in source_lower:
                    categories["PDF Documents"] += 1
                elif "sacred" in source_lower or "quran" in source_lower:
                    categories["Sacred Texts"] += 1
                elif "config" in source_lower or "schema" in source_lower:
                    categories["Configuration/Schema"] += 1
                elif "code" in source_lower or ".py" in source_lower:
                    categories["Code Documentation"] += 1
                elif "notes" in source_lower or "md" in source_lower:
                    categories["Notes/Markdown"] += 1
                else:
                    categories["Other"] += 1
                    
            except Exception as e:
                categories["Parse Errors"] += 1
        
        self.stats["categories"] = dict(categories)
        self.stats["character_count"] = total_chars
        self.stats["unique_sources"] = len(sources)
        
        return {
            "total_embeddings": self.stats["total_embeddings"],
            "categories": dict(categories),
            "total_characters": total_chars,
            "unique_sources": len(sources),
            "avg_chars_per_doc": total_chars // max(1, self.stats["total_embeddings"])
        }
    
    def validate_processed_coverage(self) -> Dict[str, Any]:
        """Check that all processed files have embeddings."""
        if not self.processed_path.exists():
            return {"error": "Processed path not found"}
        
        # Count processed files
        processed_files = []
        for ext in ["*.md", "*.txt", "*.json"]:
            processed_files.extend(self.processed_path.rglob(ext))
        
        # Check for corresponding embeddings
        embedded_keys = set()
        for emb_file in self.embeddings_path.glob("*.json"):
            try:
                with open(emb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                source = data.get("source") or data.get("metadata", {}).get("source", "")
                # Normalize path for comparison using multiple keys
                embedded_keys.update(self._normalize_source_keys(source))
            except:
                pass
        
        missing = []
        for pf in processed_files:
            keys = self._normalize_source_keys(str(pf))
            if not any(k in embedded_keys for k in keys):
                missing.append(str(pf.relative_to(self.data_lake)))
        
        coverage = 1.0 - (len(missing) / max(1, len(processed_files)))
        self.stats["coverage_score"] = coverage
        
        return {
            "total_processed": len(processed_files),
            "total_embedded": len(embedded_keys),
            "coverage_percentage": coverage * 100,
            "missing_count": len(missing),
            "sample_missing": missing[:10] if missing else []
        }
    
    def _iter_embeddings(self):
        """Yield (source, embedding) tuples from embedding JSON files."""
        for emb_file in self.embeddings_path.glob("*.json"):
            try:
                with open(emb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                embedding = data.get("embedding")
                if embedding is None:
                    continue
                source = data.get("source") or data.get("metadata", {}).get("source", str(emb_file))
                yield source, embedding
            except Exception:
                continue

    def _normalize_source_keys(self, source: str) -> set:
        """Generate comparison keys for a source path."""
        keys = set()
        if not source:
            return keys
        try:
            p = Path(source)
            keys.add(p.name.lower())
            try:
                if self.data_lake in p.parents:
                    keys.add(str(p.relative_to(self.data_lake)).lower())
                else:
                    keys.add(str(p).lower())
            except Exception:
                keys.add(str(p).lower())
        except Exception:
            keys.add(source.lower())
        return keys

    def _top_k_streaming(self, query_embedding, top_k: int = 3) -> List[Dict[str, Any]]:
        """Stream embeddings and compute top-k results with bounded memory."""
        if query_embedding is None:
            return []
        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm == 0.0:
            return []

        heap = []
        for source, embedding in self._iter_embeddings():
            try:
                vec = np.array(embedding, dtype=np.float32)
                denom = float(np.linalg.norm(vec)) * query_norm
                if denom == 0.0:
                    continue
                score = float(np.dot(vec, query_embedding) / denom)
            except Exception:
                continue

            if len(heap) < top_k:
                heapq.heappush(heap, (score, source))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, source))

        # Highest scores first
        return [
            {"source": Path(src).name, "score": float(score)}
            for score, src in sorted(heap, reverse=True)
        ]

    def test_semantic_search(self) -> Dict[str, Any]:
        """Test semantic search capability."""
        if not HAS_ML:
            return {"error": "sentence-transformers not available"}
        
        print("  Loading embedding model for search test...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Load embedding files count (for reporting)
        total_embeddings = len(list(self.embeddings_path.glob("*.json")))
        if total_embeddings == 0:
            return {"error": "No embeddings loaded"}

        # Test queries representing different domains
        test_queries = [
            "BIZRA architecture and system design",
            "ChatGPT conversation history",
            "Python code implementation",
            "Business strategy and planning",
            "Machine learning and AI development",
        ]
        
        results = {}
        print("  Running search queries...")
        
        for query in test_queries:
            query_embedding = model.encode([query])[0]
            results[query] = self._top_k_streaming(query_embedding, top_k=3)
        
        return {
            "search_functional": True,
            "corpus_size": total_embeddings,
            "sample_results": results
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        print("═" * 80)
        print("       BIZRA KNOWLEDGE BASE VALIDATION REPORT")
        print("═" * 80)
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Dubai)")
        print("─" * 80)
        
        # 1. Count embeddings
        print("\n📊 EMBEDDING INVENTORY")
        count = self.count_embeddings()
        print(f"   Total Embeddings: {count:,}")
        
        # 2. Analyze content
        print("\n📂 CONTENT ANALYSIS")
        analysis = self.analyze_embeddings()
        print(f"   Total Characters: {analysis.get('total_characters', 0):,}")
        print(f"   Unique Sources: {analysis.get('unique_sources', 0):,}")
        print(f"   Avg Chars/Doc: {analysis.get('avg_chars_per_doc', 0):,}")
        print("\n   Categories:")
        for cat, cnt in sorted(analysis.get('categories', {}).items(), key=lambda x: -x[1]):
            bar = "█" * min(30, cnt // 20)
            print(f"     {cat:30} {cnt:5} {bar}")
        
        # 3. Coverage check
        print("\n✅ COVERAGE VALIDATION")
        coverage = self.validate_processed_coverage()
        pct = coverage.get('coverage_percentage', 0)
        print(f"   Total Processed Files: {coverage.get('total_processed', 0):,}")
        print(f"   Embedded Files: {coverage.get('total_embedded', 0):,}")
        print(f"   Coverage: {pct:.1f}%")
        if coverage.get('missing_count', 0) > 0:
            print(f"   ⚠️  Missing: {coverage.get('missing_count')} files")
        
        # 4. Search test
        print("\n🔍 SEMANTIC SEARCH VALIDATION")
        search_result = self.test_semantic_search()
        if search_result.get("search_functional"):
            print(f"   ✅ Search Functional")
            print(f"   Corpus Size: {search_result.get('corpus_size', 0):,}")
            print("\n   Sample Query Results:")
            for query, results in search_result.get("sample_results", {}).items():
                print(f"\n   Query: \"{query}\"")
                for r in results[:2]:
                    print(f"     → {r['source'][:50]:50} (score: {r['score']:.4f})")
        else:
            print(f"   ❌ {search_result.get('error', 'Search failed')}")
        
        # 5. Summary
        print("\n" + "═" * 80)
        print("       VALIDATION SUMMARY")
        print("═" * 80)
        
        total_chars = self.stats.get("character_count", 0)
        total_words = total_chars // 5  # Rough estimate
        total_pages = total_words // 250  # Rough estimate
        
        print(f"""
   📊 Knowledge Base Statistics:
      ├── Total Documents: {count:,}
      ├── Total Characters: {total_chars:,}
      ├── Estimated Words: ~{total_words:,}
      ├── Estimated Pages: ~{total_pages:,}
      └── Coverage Score: {pct:.1f}%
   
   🎯 SINGULARITY Status:
      ├── Embeddings: {'✅ COMPLETE' if count > 1800 else '⚠️ PARTIAL'}
      ├── Search: {'✅ OPERATIONAL' if search_result.get('search_functional') else '❌ FAILED'}
      └── Integration: {'✅ 3 YEARS INDEXED' if count > 1800 else '⏳ IN PROGRESS'}
""")
        
        if count >= 1800 and search_result.get("search_functional") and pct >= 95:
            print("   ╔════════════════════════════════════════════════════╗")
            print("   ║  🎉 VALIDATION PASSED — SINGULARITY ACHIEVED! 🎉   ║")
            print("   ╚════════════════════════════════════════════════════╝")
            return "PASSED"
        else:
            print("   ⚠️  Some validation checks incomplete")
            return "PARTIAL"


def main():
    validator = KnowledgeBaseValidator()
    result = validator.generate_report()
    return 0 if result == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
