#!/usr/bin/env python3
"""
BIZRA LangExtract Knowledge Miner v1.0
======================================
Uses Google's LangExtract library for LLM-powered structured extraction
from AI conversation exports with precise source grounding.

Features:
- Few-shot learning for BIZRA-specific entity extraction
- Source grounding with exact text mapping
- Interactive HTML visualization
- Parallel processing for large document sets
- Ollama support for local LLM inference (no API key needed)

DNA Signature: LANGEXTRACT-7-3-6-9-00

Usage:
    python scripts/langextract_knowledge_miner.py [--model MODEL] [--limit N]
    
Models:
    gemini-2.5-flash  - Cloud (requires LANGEXTRACT_API_KEY)
    gemma2:2b         - Local via Ollama (no API key)
"""

import json
import os
import sys
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

try:
    import langextract as lx
except ImportError:
    print("❌ langextract not installed. Run: pip install langextract")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA EXTRACTION SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

# Prompt describing what to extract from BIZRA conversations
BIZRA_EXTRACTION_PROMPT = textwrap.dedent("""\
    Extract BIZRA-specific knowledge entities from AI conversation transcripts.
    Focus on:
    1. CONCEPTS: Technical terms, frameworks, and architectural patterns
    2. DECISIONS: Design choices with rationale
    3. PATTERNS: Reusable solutions and best practices
    4. TENSIONS: Trade-offs and conflicts identified
    5. AXIOMS: Foundational principles and invariants
    
    Use exact text for extractions. Do not paraphrase.
    Provide meaningful attributes for context including:
    - ihsan_dimension: Which of the 8 Ihsān dimensions this relates to
    - sape_module: Which SAPE module (1-7) if applicable
    - confidence: How confident is this extraction (high/medium/low)
    
    Ihsān dimensions: correctness, safety, user_benefit, efficiency, 
                      auditability, anti_centralization, robustness, adl_fairness
    
    SAPE modules: 1-HouseOfWisdom, 2-GoT, 3-StrategicPlanner, 4-AdaptiveExecutor,
                  5-SymbolicHarness, 6-AbstractionElevator, 7-TensionStudio
""")

# Few-shot examples for BIZRA extraction
BIZRA_EXAMPLES = [
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            The Proof-of-Impact (PoI) consensus mechanism replaces traditional 
            Proof-of-Work with measurable value creation attestation. This enables 
            the SEED token to maintain stability while BLOOM appreciates with impact.
        """),
        extractions=[
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="Proof-of-Impact (PoI)",
                attributes={
                    "definition": "Consensus mechanism based on value creation",
                    "ihsan_dimension": "user_benefit",
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="SEED token",
                attributes={
                    "definition": "Stable utility token",
                    "token_type": "utility",
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="concept", 
                extraction_text="BLOOM",
                attributes={
                    "definition": "Impact growth token",
                    "token_type": "impact",
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="decision",
                extraction_text="replaces traditional Proof-of-Work with measurable value creation",
                attributes={
                    "rationale": "Align consensus with impact rather than computation",
                    "ihsan_dimension": "efficiency",
                    "confidence": "high"
                }
            ),
        ]
    ),
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            The PAT/SAT architecture uses 7 personal agents (PAT) for user tasks
            and 5 system validators (SAT) for Byzantine consensus with f=1 fault
            tolerance. This dual-agentic design ensures both efficiency and safety.
        """),
        extractions=[
            lx.data.Extraction(
                extraction_class="pattern",
                extraction_text="PAT/SAT architecture",
                attributes={
                    "definition": "Dual-agentic architecture",
                    "pat_agents": 7,
                    "sat_validators": 5,
                    "ihsan_dimension": "anti_centralization",
                    "sape_module": 5,
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="axiom",
                extraction_text="Byzantine consensus with f=1 fault tolerance",
                attributes={
                    "invariant": "n >= 3f+1 for f Byzantine faults",
                    "ihsan_dimension": "safety",
                    "sape_module": 4,
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="tension",
                extraction_text="efficiency and safety",
                attributes={
                    "tension_type": "trade_off",
                    "resolution": "Dual-agentic separation of concerns",
                    "confidence": "medium"
                }
            ),
        ]
    ),
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            The HRM-MoE (Hierarchical Reasoning Mixture-of-Experts) engine uses
            4 latency tiers: Tier 1 (50ms) for reflexive responses, Tier 2 (200ms)
            for analytical reasoning, Tier 3 (500ms) for strategic planning, and
            Tier 4 (2000ms) for deep deliberation.
        """),
        extractions=[
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="HRM-MoE (Hierarchical Reasoning Mixture-of-Experts)",
                attributes={
                    "definition": "4-tier latency-adaptive reasoning engine",
                    "ihsan_dimension": "efficiency",
                    "sape_module": 2,
                    "confidence": "high"
                }
            ),
            lx.data.Extraction(
                extraction_class="pattern",
                extraction_text="4 latency tiers",
                attributes={
                    "tier_1_ms": 50,
                    "tier_2_ms": 200,
                    "tier_3_ms": 500,
                    "tier_4_ms": 2000,
                    "ihsan_dimension": "efficiency",
                    "confidence": "high"
                }
            ),
        ]
    ),
]


@dataclass
class ExtractionResult:
    """Result from langextract processing."""
    source_file: str
    extraction_count: int
    extractions: list
    processing_time_ms: float
    model_used: str


class LangExtractKnowledgeMiner:
    """
    BIZRA Knowledge Miner using Google's LangExtract.
    
    Extracts structured knowledge from chat conversation exports
    with precise source grounding and few-shot learning.
    """
    
    def __init__(
        self,
        model_id: str = "gemma2:2b",
        model_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_workers: int = 5,
        extraction_passes: int = 2,
    ):
        """
        Initialize the knowledge miner.
        
        Args:
            model_id: LLM model to use (e.g., "gemini-2.5-flash", "gemma2:2b")
            model_url: URL for Ollama (default: http://localhost:11434)
            api_key: API key for cloud models (or use LANGEXTRACT_API_KEY env var)
            max_workers: Parallel processing workers
            extraction_passes: Number of extraction passes for higher recall
        """
        self.model_id = model_id
        self.model_url = model_url or "http://localhost:11434"
        self.api_key = api_key or os.environ.get("LANGEXTRACT_API_KEY")
        self.max_workers = max_workers
        self.extraction_passes = extraction_passes
        self.results: list = []
        
        # Determine if using local model
        self.is_local = self._is_ollama_model(model_id)
        
    def _is_ollama_model(self, model_id: str) -> bool:
        """Check if model is an Ollama local model."""
        ollama_models = [
            "gemma", "llama", "mistral", "qwen", "phi", "codellama",
            "deepseek", "mixtral", "neural-chat", "starling"
        ]
        return any(m in model_id.lower() for m in ollama_models)
    
    def extract_from_text(self, text: str, source_file: str = "unknown") -> ExtractionResult:
        """
        Extract BIZRA knowledge from text using langextract.
        
        Args:
            text: The text content to process
            source_file: Source file name for tracking
            
        Returns:
            ExtractionResult with extracted entities
        """
        import time
        start = time.time()
        
        try:
            # Build extraction kwargs
            kwargs = {
                "text_or_documents": text,
                "prompt_description": BIZRA_EXTRACTION_PROMPT,
                "examples": BIZRA_EXAMPLES,
                "model_id": self.model_id,
                "max_workers": self.max_workers,
                "extraction_passes": self.extraction_passes,
                "max_char_buffer": 2000,  # Chunk size for processing
            }
            
            # Add model-specific options
            if self.is_local:
                kwargs["model_url"] = self.model_url
                kwargs["fence_output"] = False
                kwargs["use_schema_constraints"] = False
            elif self.api_key:
                kwargs["api_key"] = self.api_key
            
            # Run extraction
            result = lx.extract(**kwargs)
            
            elapsed_ms = (time.time() - start) * 1000
            
            # Extract entities from result
            extractions = []
            if hasattr(result, 'extractions'):
                for ext in result.extractions:
                    extractions.append({
                        "class": ext.extraction_class,
                        "text": ext.extraction_text,
                        "attributes": ext.attributes,
                        "start_char": getattr(ext, 'start_char', None),
                        "end_char": getattr(ext, 'end_char', None),
                    })
                    
            return ExtractionResult(
                source_file=source_file,
                extraction_count=len(extractions),
                extractions=extractions,
                processing_time_ms=elapsed_ms,
                model_used=self.model_id
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            print(f"  ⚠ Extraction error: {e}")
            return ExtractionResult(
                source_file=source_file,
                extraction_count=0,
                extractions=[],
                processing_time_ms=elapsed_ms,
                model_used=self.model_id
            )
    
    def process_conversation_file(self, md_path: Path) -> Optional[ExtractionResult]:
        """Process a single .md conversation file."""
        try:
            # Try multiple encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(md_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None
                
            if len(content) < 500:
                return None  # Skip very short files
                
            # Truncate very long content to avoid token limits
            max_chars = 50000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Content truncated...]"
                
            return self.extract_from_text(content, source_file=md_path.name)
            
        except Exception as e:
            print(f"  ⚠ Error reading {md_path.name}: {e}")
            return None
    
    def mine_chat_data(
        self,
        chat_data_root: Path,
        limit: Optional[int] = None
    ) -> list:
        """
        Mine knowledge from all chat data files.
        
        Args:
            chat_data_root: Root directory of chat exports
            limit: Maximum files to process (None for all)
            
        Returns:
            List of ExtractionResult objects
        """
        print(f"\n{'='*60}")
        print("BIZRA LangExtract Knowledge Miner v1.0")
        print(f"{'='*60}")
        print(f"📂 Source: {chat_data_root}")
        print(f"🤖 Model: {self.model_id}")
        print(f"🔧 Local: {self.is_local}")
        
        # Find all .md files
        md_files = list(chat_data_root.glob('**/*.md'))
        print(f"📄 Found {len(md_files)} .md files")
        
        if limit:
            md_files = md_files[:limit]
            print(f"🔢 Limited to {limit} files")
            
        print(f"\n{'─'*60}")
        print("Processing with LangExtract...")
        print(f"{'─'*60}")
        
        results = []
        total_extractions = 0
        
        for i, md_path in enumerate(md_files):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(md_files)} ({total_extractions} extractions)")
                
            result = self.process_conversation_file(md_path)
            if result and result.extraction_count > 0:
                results.append(result)
                total_extractions += result.extraction_count
                
        self.results = results
        return results
    
    def generate_visualization(self, output_path: Path) -> None:
        """Generate interactive HTML visualization."""
        if not self.results:
            print("⚠ No results to visualize")
            return
            
        # Save to JSONL first
        jsonl_path = output_path.with_suffix('.jsonl')
        
        documents = []
        for result in self.results:
            doc = {
                "text": result.source_file,
                "extractions": result.extractions
            }
            documents.append(doc)
            
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                
        print(f"📄 JSONL saved to: {jsonl_path}")
        
        # Generate HTML visualization
        try:
            html_content = lx.visualize(str(jsonl_path))
            html_path = output_path.with_suffix('.html')
            
            with open(html_path, 'w', encoding='utf-8') as f:
                if hasattr(html_content, 'data'):
                    f.write(html_content.data)
                else:
                    f.write(str(html_content))
                    
            print(f"🎨 Visualization saved to: {html_path}")
            
        except Exception as e:
            print(f"⚠ Could not generate HTML visualization: {e}")
    
    def save_manifest(self, output_path: Path) -> None:
        """Save extraction manifest JSON."""
        # Aggregate statistics
        by_class = {}
        all_extractions = []
        
        for result in self.results:
            for ext in result.extractions:
                ext_class = ext.get('class', 'unknown')
                by_class[ext_class] = by_class.get(ext_class, 0) + 1
                all_extractions.append({
                    **ext,
                    'source_file': result.source_file
                })
                
        manifest = {
            "extraction_version": "langextract-1.0.0",
            "timestamp": datetime.now().isoformat(),
            "dna_signature": "LANGEXTRACT-7-3-6-9-00",
            "model": self.model_id,
            "statistics": {
                "files_processed": len(self.results),
                "total_extractions": sum(r.extraction_count for r in self.results),
                "by_class": by_class,
                "avg_processing_time_ms": sum(r.processing_time_ms for r in self.results) / len(self.results) if self.results else 0,
            },
            "extractions": all_extractions[:500]  # Limit for file size
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            
        print(f"📋 Manifest saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BIZRA LangExtract Knowledge Miner')
    parser.add_argument('--model', type=str, default='gemma2:2b',
                       help='LLM model (gemini-2.5-flash, gemma2:2b, etc.)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit files to process')
    parser.add_argument('--chat-root', type=str, default=None,
                       help='Path to chat data folder')
    parser.add_argument('--workers', type=int, default=5,
                       help='Parallel workers')
    parser.add_argument('--passes', type=int, default=2,
                       help='Extraction passes')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for cloud models')
    args = parser.parse_args()
    
    # Check for API key or Ollama
    api_key = args.api_key or os.environ.get("LANGEXTRACT_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    is_local_model = any(m in args.model.lower() for m in [
        "gemma", "llama", "mistral", "qwen", "phi", "codellama",
        "deepseek", "mixtral", "neural-chat", "starling"
    ])
    
    if not api_key and not is_local_model:
        print("❌ No API key found for cloud model.")
        print("   Set LANGEXTRACT_API_KEY environment variable or use --api-key")
        print("   Or use a local Ollama model: --model gemma2:2b")
        return 1
        
    if is_local_model:
        # Check if Ollama is running
        import requests
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code != 200:
                raise Exception("Ollama not responding")
            models = [m['name'] for m in resp.json().get('models', [])]
            if not any(args.model.split(':')[0] in m for m in models):
                print(f"⚠ Model '{args.model}' not found in Ollama.")
                print(f"   Available models: {', '.join(models[:5])}")
                print(f"   Run: ollama pull {args.model}")
                return 1
        except Exception as e:
            print(f"❌ Ollama not running or not accessible: {e}")
            print("   Start Ollama: ollama serve")
            print("   Pull model: ollama pull gemma2:2b")
            print("   Or use cloud model with API key: --model gemini-2.5-flash --api-key YOUR_KEY")
            return 1
    
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    chat_root = Path(args.chat_root) if args.chat_root else \
                project_root / 'chat data sample'
                
    if not chat_root.exists():
        print(f"❌ Chat data folder not found: {chat_root}")
        return 1
        
    # Initialize miner
    miner = LangExtractKnowledgeMiner(
        model_id=args.model,
        max_workers=args.workers,
        extraction_passes=args.passes,
        api_key=api_key
    )
    
    # Run extraction
    results = miner.mine_chat_data(chat_root, limit=args.limit)
    
    if not results:
        print("\n⚠ No extractions found")
        return 1
        
    # Report summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Processed: {len(results)} files")
    
    total_extractions = sum(r.extraction_count for r in results)
    print(f"🧬 Total extractions: {total_extractions}")
    
    # Count by class
    by_class = {}
    for result in results:
        for ext in result.extractions:
            ext_class = ext.get('class', 'unknown')
            by_class[ext_class] = by_class.get(ext_class, 0) + 1
            
    print(f"📊 By class:")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"   {cls}: {count}")
        
    # Save outputs
    evidence_dir = project_root / 'evidence'
    
    miner.save_manifest(evidence_dir / 'langextract_manifest.json')
    miner.generate_visualization(evidence_dir / 'langextract_extractions')
    
    print(f"\n{'='*60}")
    print("✅ LangExtract mining complete!")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
