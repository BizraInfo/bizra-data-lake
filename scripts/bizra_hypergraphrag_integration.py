#!/usr/bin/env python3
"""
BIZRA HyperGraphRAG Integration v1.0
====================================
Full integration with HyperGraphRAG library for BIZRA knowledge mining.

Uses official HyperGraphRAG with BIZRA-specific:
- Entity types (CONCEPT, AXIOM, PATTERN, PROTOCOL, AGENT, etc.)
- Ihsān dimension extraction
- SAPE module attribution
- Custom prompts for domain understanding

Requirements:
    - HyperGraphRAG cloned to ./HyperGraphRAG
    - pip install -r HyperGraphRAG/requirements.txt
    - OPENAI_API_KEY or Ollama running locally

Usage:
    # With OpenAI
    export OPENAI_API_KEY="your-key"
    python scripts/bizra_hypergraphrag_integration.py --limit 50
    
    # With Ollama
    python scripts/bizra_hypergraphrag_integration.py --ollama --limit 50
    
    # Query mode
    python scripts/bizra_hypergraphrag_integration.py --query "What is Ihsān?"
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from functools import partial

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP PATHS
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
HYPERGRAPHRAG_DIR = PROJECT_ROOT / "HyperGraphRAG"

# Add HyperGraphRAG to path
if HYPERGRAPHRAG_DIR.exists():
    sys.path.insert(0, str(HYPERGRAPHRAG_DIR))

# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BIZRA_ENTITY_TYPES = [
    "CONCEPT",      # Core BIZRA concepts (Ihsān, PoI, Sovereignty)
    "AXIOM",        # Foundational principles
    "PATTERN",      # Design patterns (HRM-MoE, HTDAG, Dual-Agentic)
    "PROTOCOL",     # Communication protocols (A2A, MCP, PAT/SAT)
    "AGENT",        # Agent types (HostAgent, ReflectorAgent)
    "METRIC",       # Measurement systems (TMP, SCM, PoI)
    "TOKEN",        # Token types (SEED, BLOOM)
    "STRUCTURE",    # Data structures (BlockGraph, HTDAG)
    "DIMENSION",    # Ihsān dimensions
    "MODULE",       # SAPE modules
    "PERSON",       # Named individuals
    "ORGANIZATION", # Organizations
    "EVENT",        # Events
]


def setup_bizra_prompts():
    """Inject BIZRA-specific prompts into HyperGraphRAG."""
    try:
        from hypergraphrag import prompt as hgr_prompt
        
        # Store original for reference
        original_entity_types = hgr_prompt.PROMPTS["DEFAULT_ENTITY_TYPES"]
        
        # Override with BIZRA types
        hgr_prompt.PROMPTS["DEFAULT_ENTITY_TYPES"] = BIZRA_ENTITY_TYPES
        
        # Add BIZRA-specific example
        bizra_example = """Example 3:

Text:
BIZRA implements Ihsān through eight dimensions including correctness and safety. The HRM-MoE pattern enables hierarchical reasoning with multiple expert agents. Node0 serves as the genesis node for decentralized AI sovereignty.

################
Output:
("hyper-relation"<|>"BIZRA implements Ihsān through eight dimensions for AI excellence"<|>9)##
("entity"<|>"BIZRA"<|>"CONCEPT"<|>"A dual-agentic AI framework emphasizing Islamic excellence principles and decentralized sovereignty"<|>95)##
("entity"<|>"IHSĀN"<|>"CONCEPT"<|>"Arabic term for excellence, core ethical principle with 8 evaluation dimensions (correctness, safety, user_benefit, efficiency, auditability, anti_centralization, robustness, adl_fairness)"<|>90)##
("hyper-relation"<|>"HRM-MoE pattern enables hierarchical reasoning with multiple expert agents"<|>8)##
("entity"<|>"HRM-MOE"<|>"PATTERN"<|>"Hierarchical Reflective Mixture of Experts - multi-level AI reasoning pattern with specialized expert agents"<|>85)##
("hyper-relation"<|>"Node0 serves as the genesis node for decentralized AI sovereignty"<|>8)##
("entity"<|>"NODE0"<|>"STRUCTURE"<|>"Genesis node establishing the foundation for BIZRA's decentralized AI network"<|>80)##
("entity"<|>"CORRECTNESS"<|>"DIMENSION"<|>"Ihsān dimension measuring accuracy and truth of AI outputs"<|>70)##
("entity"<|>"SAFETY"<|>"DIMENSION"<|>"Ihsān dimension ensuring AI alignment with human values and harm prevention"<|>75)
#############################"""
        
        # Append BIZRA example to existing examples
        hgr_prompt.PROMPTS["entity_extraction_examples"].append(bizra_example)
        
        print(f"✅ BIZRA prompts injected")
        print(f"   Entity types: {len(BIZRA_ENTITY_TYPES)}")
        print(f"   Examples: {len(hgr_prompt.PROMPTS['entity_extraction_examples'])}")
        
        return True
        
    except ImportError:
        print("⚠ Could not import HyperGraphRAG prompts")
        return False


def create_ollama_llm_func(model: str = "llama3.2:3b"):
    """Create an Ollama-based LLM function compatible with HyperGraphRAG."""
    try:
        import ollama
        
        async def ollama_complete(
            prompt: str,
            system_prompt: str = "",
            hashing_kv=None,
            **kwargs
        ) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await asyncio.to_thread(
                ollama.chat,
                model=model,
                messages=messages
            )
            return response['message']['content']
        
        print(f"✅ Ollama LLM function created (model: {model})")
        return ollama_complete
        
    except ImportError:
        print("⚠ Ollama not installed. Run: pip install ollama")
        return None


def create_ollama_embedding_func(model: str = "nomic-embed-text"):
    """Create an Ollama-based embedding function."""
    try:
        import ollama
        import numpy as np
        
        async def ollama_embedding(texts: list[str]) -> np.ndarray:
            embeddings = []
            for text in texts:
                response = await asyncio.to_thread(
                    ollama.embeddings,
                    model=model,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            return np.array(embeddings)
        
        print(f"✅ Ollama embedding function created (model: {model})")
        return ollama_embedding
        
    except ImportError:
        print("⚠ Could not create Ollama embedding function")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTEGRATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BIZRAHyperGraphRAGIntegration:
    """
    Full integration with HyperGraphRAG for BIZRA knowledge extraction.
    """
    
    def __init__(
        self,
        working_dir: str = "evidence/bizra_hypergraph",
        use_ollama: bool = False,
        ollama_model: str = "llama3.2:3b",
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.rag = None
        self.stats = {
            "documents_processed": 0,
            "queries_executed": 0,
            "start_time": datetime.now().isoformat()
        }
        
        self._initialize()
    
    def _initialize(self):
        """Initialize HyperGraphRAG with BIZRA configuration."""
        try:
            from hypergraphrag import HyperGraphRAG
            
            # Setup BIZRA prompts first
            setup_bizra_prompts()
            
            # Build configuration
            config = {
                "working_dir": str(self.working_dir),
                "chunk_token_size": self.chunk_size,
                "chunk_overlap_token_size": self.chunk_overlap,
                "addon_params": {
                    "entity_types": BIZRA_ENTITY_TYPES,
                    "language": "English",
                }
            }
            
            # Configure LLM based on settings
            if self.use_ollama:
                llm_func = create_ollama_llm_func(self.ollama_model)
                if llm_func:
                    config["llm_model_func"] = llm_func
                    
                embed_func = create_ollama_embedding_func()
                if embed_func:
                    config["embedding_func"] = embed_func
            else:
                # Check for OpenAI key
                if not os.environ.get("OPENAI_API_KEY"):
                    print("⚠ OPENAI_API_KEY not set")
                    print("  Set it or use --ollama flag")
            
            # Create HyperGraphRAG instance
            self.rag = HyperGraphRAG(**config)
            
            print(f"\n✅ HyperGraphRAG initialized")
            print(f"   Working dir: {self.working_dir}")
            print(f"   Chunk size: {self.chunk_size}")
            print(f"   LLM: {'Ollama' if self.use_ollama else 'OpenAI'}")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Make sure HyperGraphRAG is cloned and dependencies installed")
            self.rag = None
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.rag = None
    
    def insert(self, texts: list[str]) -> dict:
        """Insert documents into the hypergraph."""
        if not self.rag:
            print("❌ RAG not initialized")
            return self.stats
            
        try:
            self.rag.insert(texts)
            self.stats["documents_processed"] += len(texts)
            print(f"  ✅ Inserted {len(texts)} documents")
        except Exception as e:
            print(f"  ❌ Insert error: {e}")
            import traceback
            traceback.print_exc()
            
        return self.stats
    
    def insert_from_files(
        self,
        directory: Path,
        pattern: str = "*.md",
        limit: Optional[int] = None,
        batch_size: int = 5
    ) -> dict:
        """Insert documents from a directory."""
        directory = Path(directory)
        
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return self.stats
            
        files = list(directory.rglob(pattern))
        
        if limit:
            files = files[:limit]
            
        print(f"\n📂 Processing {len(files)} files from {directory}")
        
        batch = []
        for i, filepath in enumerate(files):
            try:
                # Read with fallback encodings
                content = None
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        content = filepath.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                        
                if content is None:
                    continue
                
                # Skip very short files
                if len(content.strip()) < 100:
                    continue
                    
                batch.append(content)
                
                # Process batch
                if len(batch) >= batch_size:
                    print(f"\n  Batch {(i+1)//batch_size}: Processing {len(batch)} documents...")
                    self.insert(batch)
                    batch = []
                    
            except Exception as e:
                print(f"  ⚠ Error with {filepath.name}: {e}")
        
        # Process remaining
        if batch:
            print(f"\n  Final batch: Processing {len(batch)} documents...")
            self.insert(batch)
            
        print(f"\n📊 Total documents processed: {self.stats['documents_processed']}")
        return self.stats
    
    def query(self, query_text: str) -> str:
        """Query the hypergraph for knowledge."""
        if not self.rag:
            return "❌ RAG not initialized"
            
        try:
            result = self.rag.query(query_text)
            self.stats["queries_executed"] += 1
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Query error: {e}"
    
    def export_stats(self):
        """Export processing statistics."""
        stats_path = self.working_dir / "integration_stats.json"
        self.stats["end_time"] = datetime.now().isoformat()
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        print(f"📄 Stats saved: {stats_path}")
        return stats_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA HyperGraphRAG Integration")
    parser.add_argument("--source", type=str, default=None, help="Source directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit files")
    parser.add_argument("--query", type=str, default=None, help="Query the graph")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama instead of OpenAI")
    parser.add_argument("--ollama-model", type=str, default="llama3.2:3b", help="Ollama model")
    parser.add_argument("--working-dir", type=str, default="evidence/bizra_hypergraph")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for insertion")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("BIZRA HyperGraphRAG Integration v1.0")
    print("Based on: HyperGraphRAG (NeurIPS 2025)")
    print("=" * 60)
    
    # Check HyperGraphRAG availability
    if not HYPERGRAPHRAG_DIR.exists():
        print(f"\n❌ HyperGraphRAG not found at: {HYPERGRAPHRAG_DIR}")
        print("   Clone with: git clone https://github.com/LHRLAB/HyperGraphRAG.git")
        return 1
    
    # Initialize integration
    integration = BIZRAHyperGraphRAGIntegration(
        working_dir=args.working_dir,
        use_ollama=args.ollama,
        ollama_model=args.ollama_model,
    )
    
    if not integration.rag:
        print("\n❌ Failed to initialize. Check requirements.")
        return 1
    
    # Determine source directory
    source_dir = Path(args.source) if args.source else PROJECT_ROOT / "chat data sample"
    
    # Query mode
    if args.query:
        print(f"\n🔍 Query Mode")
        print(f"   Query: {args.query}")
        print("-" * 60)
        
        result = integration.query(args.query)
        print(f"\n📝 Response:\n{result}")
        
    # Insert mode
    elif source_dir.exists():
        print(f"\n📥 Insert Mode")
        print(f"   Source: {source_dir}")
        if args.limit:
            print(f"   Limit: {args.limit} files")
        
        integration.insert_from_files(
            source_dir,
            pattern="*.md",
            limit=args.limit,
            batch_size=args.batch_size
        )
        
        integration.export_stats()
    else:
        print(f"\n❌ Source directory not found: {source_dir}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
