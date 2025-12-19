#!/usr/bin/env python3
"""
BIZRA HyperGraphRAG Wrapper v1.0
================================
Wraps the official HyperGraphRAG library (NeurIPS 2025) for BIZRA knowledge mining.

HyperGraphRAG uses hyperedges (knowledge segments) that connect multiple entities,
providing richer knowledge representation than traditional binary-edge graphs.

Key Features:
- Hyperedge-based knowledge segments with completeness scoring
- Entity extraction with key_score importance ranking  
- BIZRA-specific entity types and Ihsān dimensions
- Integration with chat data sample corpus

Usage:
    python scripts/bizra_hypergraphrag_wrapper.py --limit 50 --query "What is Ihsān?"
    
Requirements:
    - HyperGraphRAG cloned to ./HyperGraphRAG
    - OpenAI API key (or Ollama for local inference)
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

# Add HyperGraphRAG to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
HYPERGRAPHRAG_DIR = PROJECT_ROOT / "HyperGraphRAG"

if HYPERGRAPHRAG_DIR.exists():
    sys.path.insert(0, str(HYPERGRAPHRAG_DIR))
    HYPERGRAPHRAG_AVAILABLE = True
else:
    HYPERGRAPHRAG_AVAILABLE = False
    print(f"⚠ HyperGraphRAG not found at {HYPERGRAPHRAG_DIR}")
    print("  Clone with: git clone https://github.com/LHRLAB/HyperGraphRAG.git")

# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA-SPECIFIC CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BIZRA_ENTITY_TYPES = [
    "CONCEPT",      # Core BIZRA concepts (Ihsān, PoI, Sovereignty)
    "AXIOM",        # Foundational principles  
    "PATTERN",      # Design patterns (HRM-MoE, HTDAG, Dual-Agentic)
    "PROTOCOL",     # Communication protocols (A2A, MCP, PAT/SAT)
    "AGENT",        # Agent types (HostAgent, ReflectorAgent, CrownVerifier)
    "METRIC",       # Measurement systems (TMP, SCM, PoI)
    "TOKEN",        # Token types (SEED, BLOOM)
    "STRUCTURE",    # Data structures (BlockGraph, HTDAG, Causal Fabric)
    "DIMENSION",    # Ihsān dimensions (correctness, safety, etc.)
    "MODULE",       # SAPE modules (HouseOfWisdom, PatternForge)
    "PERSON",       # Named individuals
    "ORGANIZATION", # Organizations
    "EVENT",        # Events or milestones
]

BIZRA_EXTRACTION_ADDON = """
-BIZRA Domain Context-
BIZRA is a dual-agentic AI framework emphasizing:
- Ihsān (إحسان): Islamic excellence principle with 8 evaluation dimensions
- SAPE: Symbolic AI Pattern Engine with 7 cognitive modules
- PoI: Proof of Impact for measuring AI contribution value
- HRM-MoE: Hierarchical Reflective Mixture of Experts pattern
- Node0: Genesis node for decentralized AI sovereignty
- Dual-Agentic: HostAgent + ReflectorAgent architecture

When extracting entities, prefer these BIZRA-specific types:
- CONCEPT for core ideas (Ihsān, Sovereignty, Causal Fabric)
- PATTERN for design patterns (HRM-MoE, HTDAG, Dual-Agentic)
- PROTOCOL for communication (A2A, MCP, PAT/SAT)
- AGENT for agent types (HostAgent, ReflectorAgent, CrownVerifier)
- METRIC for measurements (TMP, SCM, PoI, key_score)
- STRUCTURE for data structures (BlockGraph, HTDAG)
- DIMENSION for Ihsān dimensions (correctness, safety, auditability)
- MODULE for SAPE modules (HouseOfWisdom, PatternForge, AxiomAnvil)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA LLM FUNCTION (for local inference without OpenAI)
# ═══════════════════════════════════════════════════════════════════════════════

async def ollama_complete(
    prompt: str,
    model: str = "llama3.2:3b",
    system_prompt: str = "",
    **kwargs
) -> str:
    """
    Complete prompt using Ollama local inference.
    Fallback when OpenAI API key not available.
    """
    try:
        import ollama
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
        
    except ImportError:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def ollama_complete_sync(prompt: str, **kwargs) -> str:
    """Synchronous wrapper for Ollama."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(ollama_complete(prompt, **kwargs))
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA HYPERGRAPHRAG WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class BIZRAHyperGraphRAGWrapper:
    """
    Wrapper around HyperGraphRAG for BIZRA knowledge extraction.
    
    Configures HyperGraphRAG with:
    - BIZRA-specific entity types
    - Custom prompts for domain understanding
    - Ihsān dimension scoring
    - SAPE module attribution
    """
    
    def __init__(
        self,
        working_dir: str = "evidence/bizra_hypergraph",
        use_ollama: bool = False,
        ollama_model: str = "llama3.2:3b",
        openai_api_key: Optional[str] = None,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.rag = None
        
        # Set API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self._init_rag()
        
        # Stats
        self.stats = {
            "documents_inserted": 0,
            "files_processed": 0,
            "queries_executed": 0
        }
        
    def _init_rag(self):
        """Initialize HyperGraphRAG with BIZRA configuration."""
        if not HYPERGRAPHRAG_AVAILABLE:
            print("❌ HyperGraphRAG not available")
            return
            
        try:
            from hypergraphrag import HyperGraphRAG
            
            # Configure with BIZRA entity types
            self.rag = HyperGraphRAG(
                working_dir=str(self.working_dir),
                addon_params={
                    "entity_types": BIZRA_ENTITY_TYPES,
                    "language": "English",
                }
            )
            
            print(f"✅ HyperGraphRAG initialized")
            print(f"   Working dir: {self.working_dir}")
            print(f"   Entity types: {len(BIZRA_ENTITY_TYPES)}")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            self.rag = None
        except Exception as e:
            print(f"❌ Init error: {e}")
            self.rag = None
    
    def insert(self, texts: Union[str, list[str]], source_id: str = "") -> dict:
        """
        Insert documents into the HyperGraph.
        
        Args:
            texts: Single document or list of documents
            source_id: Optional source identifier
            
        Returns:
            Stats dictionary
        """
        if not self.rag:
            print("❌ RAG not initialized")
            return self.stats
            
        if isinstance(texts, str):
            texts = [texts]
            
        # Add BIZRA context to each document
        enhanced_texts = []
        for text in texts:
            # Prepend domain context (will be processed by entity extraction)
            enhanced = f"{BIZRA_EXTRACTION_ADDON}\n\n---\n\n{text}"
            enhanced_texts.append(enhanced)
        
        try:
            self.rag.insert(enhanced_texts)
            self.stats["documents_inserted"] += len(texts)
            print(f"  ✅ Inserted {len(texts)} documents")
        except Exception as e:
            print(f"  ❌ Insert error: {e}")
            
        return self.stats
    
    def insert_from_files(
        self,
        directory: Path,
        pattern: str = "*.md",
        limit: Optional[int] = None,
        batch_size: int = 10
    ) -> dict:
        """
        Insert documents from a directory.
        
        Args:
            directory: Source directory
            pattern: Glob pattern for files
            limit: Maximum files to process
            batch_size: Number of files per batch
        """
        directory = Path(directory)
        files = list(directory.rglob(pattern))
        
        if limit:
            files = files[:limit]
            
        print(f"\n📂 Processing {len(files)} files from {directory}")
        
        batch = []
        for i, filepath in enumerate(files):
            try:
                # Try multiple encodings
                content = None
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        content = filepath.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                        
                if content is None:
                    print(f"  ⚠ Could not decode: {filepath.name}")
                    continue
                
                # Add to batch
                batch.append(content)
                self.stats["files_processed"] += 1
                
                # Process batch
                if len(batch) >= batch_size:
                    self.insert(batch)
                    batch = []
                    print(f"  Progress: {i+1}/{len(files)} files")
                    
            except Exception as e:
                print(f"  ❌ Error with {filepath.name}: {e}")
        
        # Process remaining
        if batch:
            self.insert(batch)
            
        return self.stats
    
    def query(self, query_text: str) -> str:
        """
        Query the HyperGraph for knowledge.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Generated response
        """
        if not self.rag:
            return "❌ RAG not initialized"
            
        try:
            result = self.rag.query(query_text)
            self.stats["queries_executed"] += 1
            return result
        except Exception as e:
            return f"❌ Query error: {e}"
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self.stats


# ═══════════════════════════════════════════════════════════════════════════════
# OFFLINE MODE (No LLM required)
# ═══════════════════════════════════════════════════════════════════════════════

class BIZRAHyperGraphOffline:
    """
    Offline HyperGraph extraction using regex patterns.
    Works without LLM for basic entity/hyperedge extraction.
    """
    
    # Entity patterns: (regex, entity_name, entity_type, key_score)
    ENTITY_PATTERNS = [
        (r'\bBIZRA\b', "BIZRA", "CONCEPT", 95),
        (r'\bIhsān\b|إحسان|Ihsan\b', "IHSĀN", "CONCEPT", 90),
        (r'\bNode\s*0\b|Node\s*Zero|Genesis\s+Node', "NODE0", "STRUCTURE", 85),
        (r'\bPoI\b|Proof[- ]of[- ]Impact', "POI", "METRIC", 80),
        (r'\bHRM[- ]?MoE\b', "HRM_MOE", "PATTERN", 85),
        (r'\bHTDAG\b', "HTDAG", "STRUCTURE", 80),
        (r'\bSAPE\b', "SAPE", "PATTERN", 85),
        (r'\bdual[- ]?agentic\b', "DUAL_AGENTIC", "PATTERN", 80),
        (r'\bA2A\b|Agent[- ]to[- ]Agent', "A2A", "PROTOCOL", 75),
        (r'\bMCP\b|Model\s+Context\s+Protocol', "MCP", "PROTOCOL", 75),
        (r'\bPAT\s*/?\s*SAT\b', "PAT_SAT", "PROTOCOL", 70),
        (r'\bHost\s*Agent\b', "HOST_AGENT", "AGENT", 70),
        (r'\bReflector\s*Agent\b', "REFLECTOR_AGENT", "AGENT", 70),
        (r'\bCrown\s*Verifier\b', "CROWN_VERIFIER", "AGENT", 75),
        (r'\bTMP\b|Temporal\s+Measurement', "TMP", "METRIC", 65),
        (r'\bSCM\b|Structured\s+Cognitive\s+Metric', "SCM", "METRIC", 65),
        (r'\bSEED\s+token\b', "SEED_TOKEN", "TOKEN", 70),
        (r'\bBLOOM\s+token\b', "BLOOM_TOKEN", "TOKEN", 70),
        (r'\bBlock[- ]?Graph\b', "BLOCKGRAPH", "STRUCTURE", 65),
        (r'\bCausal\s+Fabric\b', "CAUSAL_FABRIC", "STRUCTURE", 70),
        (r'\bcorrectness\b', "CORRECTNESS", "DIMENSION", 50),
        (r'\bsafety\b', "SAFETY", "DIMENSION", 55),
        (r'\bauditability\b', "AUDITABILITY", "DIMENSION", 50),
        (r'\brobustness\b', "ROBUSTNESS", "DIMENSION", 50),
        (r'\banti[- ]?centralization\b', "ANTI_CENTRALIZATION", "DIMENSION", 55),
        (r'\bHouse\s+of\s+Wisdom\b', "HOUSE_OF_WISDOM", "MODULE", 60),
    ]
    
    def __init__(self, working_dir: str = "evidence/bizra_hypergraph_offline"):
        import re
        self.re = re
        
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile patterns
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), name, etype, score)
            for pattern, name, etype, score in self.ENTITY_PATTERNS
        ]
        
        # Storage
        self.entities = {}  # name -> {type, score, sources, count}
        self.hyperedges = []  # list of knowledge segments
        
        self.stats = {
            "documents": 0,
            "entities": 0,
            "hyperedges": 0,
            "sources": []
        }
        
    def _extract_entities(self, text: str, source: str) -> list:
        """Extract entities using regex patterns."""
        found = []
        for pattern, name, etype, score in self.compiled_patterns:
            if pattern.search(text):
                if name not in self.entities:
                    self.entities[name] = {
                        "type": etype,
                        "key_score": score,
                        "sources": [],
                        "count": 0
                    }
                self.entities[name]["sources"].append(source)
                self.entities[name]["count"] += 1
                found.append(name)
        return found
    
    def _extract_hyperedges(self, text: str, source: str) -> list:
        """Extract knowledge segments (hyperedges) from text."""
        hyperedges = []
        
        # Split into sentences
        sentences = self.re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(sentence) < 30:
                continue
                
            # Find entities in sentence
            entities_in_sentence = []
            for pattern, name, _, _ in self.compiled_patterns:
                if pattern.search(sentence):
                    entities_in_sentence.append(name)
            
            # Only create hyperedge if 2+ entities
            if len(entities_in_sentence) >= 2:
                # Score completeness
                completeness = self._score_completeness(sentence)
                
                hyperedge = {
                    "knowledge_segment": sentence.strip()[:200],
                    "completeness_score": completeness,
                    "entities": entities_in_sentence,
                    "source": source
                }
                hyperedges.append(hyperedge)
                self.hyperedges.append(hyperedge)
                
        return hyperedges
    
    def _score_completeness(self, sentence: str) -> float:
        """Score knowledge segment completeness (0-10)."""
        score = 5.0
        
        # Definitional
        if self.re.search(r'\b(is|are|means|represents|defines)\b', sentence, self.re.I):
            score += 1.5
        # Causal
        if self.re.search(r'\b(because|therefore|enables|causes)\b', sentence, self.re.I):
            score += 1.0
        # Specific
        if self.re.search(r'\d+', sentence):
            score += 0.5
        # Uncertain
        if self.re.search(r'\b(maybe|possibly|might)\b', sentence, self.re.I):
            score -= 1.0
            
        return min(10.0, max(0.0, score))
    
    def insert(self, text: str, source: str = ""):
        """Insert a document."""
        self._extract_entities(text, source)
        self._extract_hyperedges(text, source)
        self.stats["documents"] += 1
        self.stats["sources"].append(source)
        
    def insert_from_files(self, directory: Path, pattern: str = "*.md", limit: int = None):
        """Insert from directory."""
        directory = Path(directory)
        files = list(directory.rglob(pattern))
        if limit:
            files = files[:limit]
            
        print(f"📂 Processing {len(files)} files (offline mode)")
        
        for i, filepath in enumerate(files):
            try:
                for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
                    try:
                        content = filepath.read_text(encoding=enc)
                        break
                    except:
                        continue
                else:
                    continue
                    
                source = str(filepath.relative_to(directory))
                self.insert(content, source)
                
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{len(files)}")
                    
            except Exception as e:
                print(f"  ⚠ Error: {filepath.name}: {e}")
        
        self.stats["entities"] = len(self.entities)
        self.stats["hyperedges"] = len(self.hyperedges)
        
        return self.stats
    
    def save(self):
        """Save results to files."""
        # Entities
        entities_path = self.working_dir / "entities.json"
        with open(entities_path, 'w') as f:
            json.dump(self.entities, f, indent=2)
            
        # Hyperedges
        hyperedges_path = self.working_dir / "hyperedges.json"
        with open(hyperedges_path, 'w') as f:
            json.dump(self.hyperedges, f, indent=2)
            
        # Stats
        stats_path = self.working_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                **self.stats,
                "generated_at": datetime.now().isoformat()
            }, f, indent=2)
            
        # Cypher export
        cypher_path = self.working_dir / "hypergraph.cypher"
        with open(cypher_path, 'w', encoding='utf-8') as f:
            f.write("// BIZRA HyperGraph - Neo4j Cypher Export\n")
            f.write(f"// Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("// === ENTITIES ===\n")
            for name, data in self.entities.items():
                f.write(f'CREATE (:{data["type"]} {{name: "{name}", key_score: {data["key_score"]}, count: {data["count"]}}})\n')
            
            f.write("\n// === HYPEREDGES ===\n")
            for i, he in enumerate(self.hyperedges[:100]):  # Limit for readability
                segment = he["knowledge_segment"][:80].replace('"', '\\"')
                f.write(f'CREATE (:HYPEREDGE {{id: "he-{i}", segment: "{segment}", completeness: {he["completeness_score"]:.1f}}})\n')
        
        return {
            "entities": entities_path,
            "hyperedges": hyperedges_path,
            "stats": stats_path,
            "cypher": cypher_path
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA HyperGraphRAG Wrapper")
    parser.add_argument("--source", type=str, default=None, help="Source directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit files")
    parser.add_argument("--query", type=str, default=None, help="Query the graph")
    parser.add_argument("--offline", action="store_true", help="Use offline mode (no LLM)")
    parser.add_argument("--working-dir", type=str, default="evidence/bizra_hypergraph")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("BIZRA HyperGraphRAG Wrapper v1.0")
    print("=" * 60)
    
    # Determine source
    source_dir = Path(args.source) if args.source else PROJECT_ROOT / "chat data sample"
    
    if args.offline or not HYPERGRAPHRAG_AVAILABLE or not os.environ.get("OPENAI_API_KEY"):
        print("🔧 Mode: Offline (regex-based extraction)")
        
        rag = BIZRAHyperGraphOffline(working_dir=args.working_dir + "_offline")
        stats = rag.insert_from_files(source_dir, limit=args.limit)
        outputs = rag.save()
        
        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS")
        print("=" * 60)
        print(f"📄 Documents: {stats['documents']}")
        print(f"🧬 Entities: {stats['entities']}")
        print(f"🔗 HyperEdges: {stats['hyperedges']}")
        
        print(f"\n📊 Entity breakdown:")
        type_counts = {}
        for name, data in rag.entities.items():
            etype = data["type"]
            type_counts[etype] = type_counts.get(etype, 0) + 1
        for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"   {etype}: {count}")
        
        print(f"\n📄 Saved to:")
        for key, path in outputs.items():
            print(f"   {key}: {path}")
            
    else:
        print("🔧 Mode: LLM-powered (HyperGraphRAG)")
        
        rag = BIZRAHyperGraphRAGWrapper(working_dir=args.working_dir)
        
        if source_dir.exists():
            stats = rag.insert_from_files(source_dir, limit=args.limit)
            print(f"\n📄 Inserted: {stats['documents_inserted']} documents")
        
        if args.query:
            print(f"\n🔍 Query: {args.query}")
            result = rag.query(args.query)
            print(f"\n📝 Response:\n{result}")
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
