#!/usr/bin/env python3
"""
BIZRA Memories JSON Streamer v1.0
=================================
Streams large memories.json files (>50MB) without loading entirely into memory.
Extracts consolidated institutional knowledge from ChatGPT batch exports.

Usage:
    python scripts/memories_streamer.py [memories_path]
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional
from datetime import datetime


@dataclass
class MemoryChunk:
    """Single memory extraction from the consolidated memories."""
    memory_type: str  # 'conversation' or 'project'
    project_id: Optional[str]
    content: str
    concepts: list  # Extracted concepts
    word_count: int


class MemoriesStreamer:
    """
    Stream large memories.json files in chunks.
    
    ChatGPT batch exports contain:
    - conversations_memory: Consolidated chat memory (~15,000+ hours)
    - project_memories: Per-project context memories
    """
    
    # BIZRA concept patterns for extraction
    CONCEPT_PATTERNS = [
        (r'\bBIZRA\b', 'BIZRA'),
        (r'\bNode0?\s*(?:Genesis)?\b', 'Node0'),
        (r'\bPoI\b|Proof[- ]of[- ]Impact', 'PoI'),
        (r'\bPAT\s*/?\s*SAT\b', 'PAT_SAT'),
        (r'\bHRM[- ]?MoE\b', 'HRM_MoE'),
        (r'\bHTDAG\b', 'HTDAG'),
        (r'\bIhsan\b|إحسان|excellence\s+principle', 'Ihsan'),
        (r'\bSAPE\b', 'SAPE'),
        (r'\bMCP\b|Model\s+Context\s+Protocol', 'MCP'),
        (r'\bTMP\b|Temporal\s+Measurement', 'TMP'),
        (r'\bSCM\b|Structured\s+Cognitive\s+Metric', 'SCM'),
        (r'\bCrown\s+Verifier\b', 'Crown_Verifier'),
        (r'\bCausal\s+Fabric\b', 'Causal_Fabric'),
        (r'\bBlock[- ]?Graph\b', 'BlockGraph'),
        (r'\bSEED\s+token\b', 'SEED_Token'),
        (r'\bBLOOM\s+token\b', 'BLOOM_Token'),
        (r'\bA2A\b', 'A2A'),
        (r'\bNeo4j\b', 'Neo4j'),
        (r'\bHouse\s+of\s+Wisdom\b', 'House_of_Wisdom'),
        (r'\bdual[- ]?agentic\b', 'Dual_Agentic'),
    ]
    
    def __init__(self, memories_path: Path):
        """Initialize streamer with path to memories.json."""
        self.memories_path = Path(memories_path)
        self.stats = {
            'total_chunks': 0,
            'total_words': 0,
            'concepts_found': set(),
            'project_count': 0
        }
        
    def _extract_concepts(self, text: str) -> list:
        """Extract BIZRA concepts from text."""
        concepts = []
        for pattern, concept in self.CONCEPT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                concepts.append(concept)
                self.stats['concepts_found'].add(concept)
        return concepts
    
    def _chunk_text(self, text: str, chunk_size: int = 10000) -> Iterator[str]:
        """Split large text into manageable chunks preserving sentence boundaries."""
        if len(text) <= chunk_size:
            yield text
            return
            
        # Split on paragraph boundaries first
        paragraphs = re.split(r'\n\n+', text)
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    yield current_chunk.strip()
                current_chunk = para + "\n\n"
                
        if current_chunk:
            yield current_chunk.strip()
    
    def stream_with_ijson(self) -> Iterator[MemoryChunk]:
        """Stream using ijson for memory-efficient parsing."""
        try:
            import ijson
        except ImportError:
            print("⚠ ijson not installed. Install with: pip install ijson")
            print("  Falling back to full load...")
            yield from self.stream_full_load()
            return
            
        print(f"📂 Streaming with ijson: {self.memories_path}")
        
        with open(self.memories_path, 'rb') as f:
            # First pass: get conversation memory
            parser = ijson.parse(f)
            conversation_memory = ""
            
            for prefix, event, value in parser:
                if prefix == 'conversations_memory' and event == 'string':
                    conversation_memory = value
                    break
                    
        if conversation_memory:
            # Chunk the conversation memory
            for chunk in self._chunk_text(conversation_memory):
                concepts = self._extract_concepts(chunk)
                word_count = len(chunk.split())
                self.stats['total_words'] += word_count
                self.stats['total_chunks'] += 1
                
                yield MemoryChunk(
                    memory_type='conversation',
                    project_id=None,
                    content=chunk,
                    concepts=concepts,
                    word_count=word_count
                )
                
        # Second pass: stream project memories
        with open(self.memories_path, 'rb') as f:
            for project_id, memory in ijson.kvitems(f, 'project_memories'):
                self.stats['project_count'] += 1
                
                for chunk in self._chunk_text(memory):
                    concepts = self._extract_concepts(chunk)
                    word_count = len(chunk.split())
                    self.stats['total_words'] += word_count
                    self.stats['total_chunks'] += 1
                    
                    yield MemoryChunk(
                        memory_type='project',
                        project_id=project_id,
                        content=chunk,
                        concepts=concepts,
                        word_count=word_count
                    )
    
    def stream_full_load(self) -> Iterator[MemoryChunk]:
        """Fallback: load entire file (memory intensive for large files)."""
        print(f"📂 Full load: {self.memories_path}")
        print(f"  ⚠ File may consume significant memory")
        
        with open(self.memories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process conversation memory
        if 'conversations_memory' in data:
            conv_mem = data['conversations_memory']
            for chunk in self._chunk_text(conv_mem):
                concepts = self._extract_concepts(chunk)
                word_count = len(chunk.split())
                self.stats['total_words'] += word_count
                self.stats['total_chunks'] += 1
                
                yield MemoryChunk(
                    memory_type='conversation',
                    project_id=None,
                    content=chunk,
                    concepts=concepts,
                    word_count=word_count
                )
                
        # Process project memories
        if 'project_memories' in data:
            for project_id, memory in data['project_memories'].items():
                self.stats['project_count'] += 1
                
                for chunk in self._chunk_text(memory):
                    concepts = self._extract_concepts(chunk)
                    word_count = len(chunk.split())
                    self.stats['total_words'] += word_count
                    self.stats['total_chunks'] += 1
                    
                    yield MemoryChunk(
                        memory_type='project',
                        project_id=project_id,
                        content=chunk,
                        concepts=concepts,
                        word_count=word_count
                    )
    
    def stream(self) -> Iterator[MemoryChunk]:
        """Stream memories, auto-selecting best method."""
        file_size = self.memories_path.stat().st_size
        
        print(f"\n{'='*60}")
        print("BIZRA Memories Streamer v1.0")
        print(f"{'='*60}")
        print(f"📄 File: {self.memories_path.name}")
        print(f"📊 Size: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size > 10 * 1024 * 1024:  # > 10MB
            print(f"🔄 Using streaming parser (large file)")
            yield from self.stream_with_ijson()
        else:
            print(f"⚡ Using full load (small file)")
            yield from self.stream_full_load()
            
    def print_stats(self):
        """Print extraction statistics."""
        print(f"\n{'─'*60}")
        print("Streaming Statistics:")
        print(f"{'─'*60}")
        print(f"  Total chunks: {self.stats['total_chunks']}")
        print(f"  Total words: {self.stats['total_words']:,}")
        print(f"  Projects: {self.stats['project_count']}")
        print(f"  Unique concepts: {len(self.stats['concepts_found'])}")
        print(f"  Concepts: {', '.join(sorted(self.stats['concepts_found']))}")


def extract_key_sections(content: str) -> dict:
    """
    Extract key sections from memory content.
    
    Memories often have structured headers like:
    **Work context**
    **Technical stack**
    **Current focus**
    """
    sections = {}
    
    # Match **Header** patterns
    header_pattern = re.compile(r'\*\*([^*]+)\*\*\s*\n([^*]+?)(?=\*\*|\Z)', re.DOTALL)
    
    for match in header_pattern.finditer(content):
        header = match.group(1).strip()
        body = match.group(2).strip()
        sections[header] = body
        
    return sections


def main():
    """Main entry point."""
    # Determine memories path
    if len(sys.argv) > 1:
        memories_path = Path(sys.argv[1])
    else:
        # Default to batch export folder
        project_root = Path(__file__).parent.parent
        memories_path = project_root / 'chat data sample' / \
                       'data-2025-12-15-17-59-21-batch-0000' / 'memories.json'
                       
    if not memories_path.exists():
        print(f"❌ File not found: {memories_path}")
        return 1
        
    # Initialize streamer
    streamer = MemoriesStreamer(memories_path)
    
    # Stream and process
    key_findings = []
    
    for i, chunk in enumerate(streamer.stream()):
        if chunk.concepts:
            # Extract key sections from chunks with concepts
            sections = extract_key_sections(chunk.content)
            
            finding = {
                'chunk_id': i,
                'type': chunk.memory_type,
                'project_id': chunk.project_id,
                'concepts': chunk.concepts,
                'word_count': chunk.word_count,
                'sections': list(sections.keys())[:5]  # First 5 section headers
            }
            key_findings.append(finding)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1} chunks...")
                
    # Print summary
    streamer.print_stats()
    
    print(f"\n{'─'*60}")
    print(f"Key Findings: {len(key_findings)} chunks with BIZRA concepts")
    print(f"{'─'*60}")
    
    # Show top findings
    for finding in key_findings[:10]:
        print(f"  [{finding['type']}] Chunk {finding['chunk_id']}: "
              f"{', '.join(finding['concepts'][:3])}")
        if finding['sections']:
            print(f"      Sections: {', '.join(finding['sections'][:3])}")
            
    # Save findings
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'evidence' / 'memories_extraction.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'source': str(memories_path),
            'stats': {
                'total_chunks': streamer.stats['total_chunks'],
                'total_words': streamer.stats['total_words'],
                'project_count': streamer.stats['project_count'],
                'unique_concepts': list(streamer.stats['concepts_found'])
            },
            'findings': key_findings
        }, f, indent=2)
        
    print(f"\n📋 Findings saved to: {output_path}")
    print(f"\n{'='*60}")
    print("✅ Streaming complete!")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
