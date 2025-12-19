#!/usr/bin/env python3
"""
BIZRA Chat Knowledge Extractor v1.0
===================================
Extracts valuable knowledge from AI conversation exports (ChatGPT, DeepSeek)
and integrates with BIZRA's ontology schema, lexicon ledger, and knowledge base.

DNA Signature: EXTRACT-7-3-6-9-00
- 7 knowledge categories
- 3 export formats (.md, .json, .txt)
- 6 quality dimensions
- 9 extraction probes

Usage:
    python scripts/chat_knowledge_extractor.py [--dry-run] [--limit N]
"""

import json
import re
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Iterator, Optional, Any
from collections import defaultdict

# Add parent to path for bizra_kernel imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bizra_kernel.lexicon_ledger import LexiconLedger, Term, TermStatus, TruthLabel


class KnowledgeCategory(Enum):
    """7 knowledge categories from chat data analysis."""
    BIZRA_CORE = "bizra_core"           # Architecture, PoI, PAT/SAT
    AI_AGENT_PATTERNS = "ai_agent"       # Multi-agent, LLM agentic
    SECURITY_SAFETY = "security_safety"  # Safe RSI, TMP, constraints
    PERFORMANCE = "performance"          # Peak skills, optimization
    ONTOLOGY = "ontology"               # Knowledge representation
    TOOLING = "tooling"                 # MCP, integrations
    GENERAL = "general"                 # Other valuable content


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    role: str  # user, assistant, system
    content: str
    timestamp: Optional[float] = None
    model_slug: Optional[str] = None
    turn_summary: Optional[str] = None


@dataclass
class ConversationMetadata:
    """Metadata extracted from companion JSON."""
    conversation_id: str
    title: str
    create_time: float
    update_time: float
    model_slug: str = "unknown"
    turn_count: int = 0
    has_attachments: bool = False
    safe_urls: list = field(default_factory=list)


@dataclass
class KnowledgeAtom:
    """Atomic unit of extracted knowledge."""
    concept: str
    definition: Optional[str] = None
    source_file: str = ""
    source_turn: int = 0
    category: KnowledgeCategory = KnowledgeCategory.GENERAL
    confidence: float = 0.5
    related_terms: list = field(default_factory=list)
    ihsan_dimension: Optional[str] = None  # Which Ihsān dimension this supports


@dataclass
class KnowledgePacket:
    """Complete extraction result for one conversation."""
    source_file: str
    conversation_id: str
    title: str
    category: KnowledgeCategory
    turns: list  # List[ConversationTurn]
    atoms: list  # List[KnowledgeAtom]
    quality_score: float
    metadata: Optional[ConversationMetadata] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "source_file": self.source_file,
            "conversation_id": self.conversation_id,
            "title": self.title,
            "category": self.category.value,
            "turn_count": len(self.turns),
            "atom_count": len(self.atoms),
            "quality_score": self.quality_score,
            "atoms": [
                {
                    "concept": a.concept,
                    "definition": a.definition,
                    "category": a.category.value,
                    "confidence": a.confidence,
                    "ihsan_dimension": a.ihsan_dimension
                }
                for a in self.atoms
            ]
        }


class ChatKnowledgeExtractor:
    """
    Main extractor for BIZRA chat knowledge harvesting.
    
    Parses .md conversation files, extracts .json metadata,
    identifies knowledge atoms, and yields structured packets.
    """
    
    # Pattern matchers for .md format
    USER_PATTERN = re.compile(r'^## USER\s*$', re.MULTILINE)
    ASSISTANT_PATTERN = re.compile(r'^## ASSISTANT\s*$', re.MULTILINE)
    THOUGHTS_PATTERN = re.compile(r'^## Thoughts\s*$', re.MULTILINE)
    TITLE_PATTERN = re.compile(r'^# (.+)$', re.MULTILINE)
    URL_PATTERN = re.compile(r'https://chat\.openai\.com/c/([a-f0-9-]+)')
    
    # BIZRA-specific term patterns
    BIZRA_TERMS = {
        # Core architecture
        r'\bHRM[-_]?MoE\b': ('HRM_MoE', 'Hierarchical Reasoning Mixture-of-Experts'),
        r'\bHTDAG\b': ('HTDAG', 'Hierarchical Task Directed Acyclic Graph'),
        r'\bPAT\s*/\s*SAT\b|\bPAT/SAT\b': ('PAT_SAT', 'Personal Agent Team / System Agent Team'),
        r'\bPoI\b': ('PoI', 'Proof-of-Impact consensus mechanism'),
        r'\bNode0\b': ('Node0', 'BIZRA Genesis Node - first sovereign instance'),
        
        # Safety & alignment
        r'\bTMP\s*v?\d*\.?\d*\b': ('TMP', 'Temporal Measurement Protocol for RSI'),
        r'\bSCM\s*v?\d*\.?\d*\b': ('SCM', 'Structured Cognitive Metric'),
        r'\bCrown\s*Verifier\b': ('Crown_Verifier', 'Ed25519 cryptographic deployment gate'),
        r'\bCausal\s*Drag\b': ('Causal_Drag', 'Structural risk quantification (Ω)'),
        r'\bIhsan\b|إحسان': ('Ihsan', 'Excellence principle - triadic ethical constraint'),
        
        # Ledger & economics
        r'\bBlock[-_]?Tree\b': ('BlockTree', 'Hybrid ledger structure for parallelism'),
        r'\bBlock[-_]?Graph\b': ('BlockGraph', 'Graph-based block structure'),
        r'\bCausal\s*Fabric\b': ('Causal_Fabric', 'Immutable truth ledger'),
        r'\bSEED\s*token\b': ('SEED_Token', 'Stable utility token'),
        r'\bBLOOM\s*token\b': ('BLOOM_Token', 'Impact growth token'),
        
        # Agent patterns
        r'\bSAPE\b': ('SAPE', 'Symbolic-Attentive Pattern Engine'),
        r'\bMCP\b': ('MCP', 'Model Context Protocol'),
        r'\bA2A\b': ('A2A', 'Agent-to-Agent communication protocol'),
        r'\bReflector\s*Agent\b': ('ReflectorAgent', 'Learning synthesizer agent'),
        r'\bHost\s*Agent\b': ('HostAgent', 'Orchestrator agent'),
    }
    
    # Category classification keywords
    CATEGORY_KEYWORDS = {
        KnowledgeCategory.BIZRA_CORE: [
            'bizra', 'node0', 'poi', 'proof-of-impact', 'pat/sat', 
            'dual-agentic', 'genesis', 'sovereignty'
        ],
        KnowledgeCategory.AI_AGENT_PATTERNS: [
            'agent', 'multi-agent', 'agentic', 'agentscope', 'swarm',
            'orchestrator', 'llm', 'framework'
        ],
        KnowledgeCategory.SECURITY_SAFETY: [
            'safety', 'security', 'rsi', 'recursive', 'self-improvement',
            'tmp', 'scm', 'constraint', 'alignment', 'ihsan'
        ],
        KnowledgeCategory.PERFORMANCE: [
            'performance', 'peak', 'optimization', 'latency', 'throughput',
            'efficiency', 'benchmark', 'scaling'
        ],
        KnowledgeCategory.ONTOLOGY: [
            'ontology', 'knowledge', 'semantic', 'taxonomy', 'schema',
            'lexicon', 'extraction'
        ],
        KnowledgeCategory.TOOLING: [
            'mcp', 'tool', 'integration', 'api', 'docker', 'kubernetes',
            'deployment', 'infrastructure'
        ],
    }
    
    def __init__(self, chat_data_root: Path):
        """
        Initialize extractor with chat data root directory.
        
        Args:
            chat_data_root: Path to 'chat data sample' folder
        """
        self.chat_data_root = Path(chat_data_root)
        self.seen_ids: set = set()  # For deduplication
        self.stats = defaultdict(int)
        self.ledger = LexiconLedger()
        
    def parse_markdown_turns(self, md_content: str) -> list:
        """
        Parse .md file into conversation turns.
        
        Format expected:
            # Title
            https://chat.openai.com/c/{id}
            
            ## USER
            {content}
            
            ## ASSISTANT
            {content}
        """
        turns = []
        
        # Split by role markers
        sections = re.split(r'(## USER|## ASSISTANT|## Thoughts)', md_content)
        
        current_role = None
        for section in sections:
            section = section.strip()
            if section == '## USER':
                current_role = 'user'
            elif section == '## ASSISTANT':
                current_role = 'assistant'
            elif section == '## Thoughts':
                current_role = 'thoughts'  # Extended thinking
            elif current_role and section:
                # Skip title and URL at the start
                if section.startswith('#') or section.startswith('http'):
                    continue
                turns.append(ConversationTurn(
                    role=current_role,
                    content=section
                ))
                
        return turns
    
    def extract_json_metadata(self, json_path: Path) -> Optional[ConversationMetadata]:
        """Extract metadata from companion .json file."""
        if not json_path.exists():
            return None
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data:
                return None
                
            # Extract conversation ID from mapping or generate from title
            conv_id = data.get('conversation_id', '')
            if not conv_id and 'mapping' in data:
                # Use first message ID as fallback
                conv_id = list(data['mapping'].keys())[0] if data['mapping'] else ''
            
            # Find model slug from any message
            model_slug = data.get('default_model_slug', 'unknown')
            
            # Count turns
            turn_count = 0
            has_attachments = False
            if 'mapping' in data:
                for msg_id, msg_data in data['mapping'].items():
                    if msg_data.get('message'):
                        msg = msg_data['message']
                        if msg.get('author', {}).get('role') in ['user', 'assistant']:
                            turn_count += 1
                        # Check for attachments
                        metadata = msg.get('metadata', {})
                        if metadata.get('attachments'):
                            has_attachments = True
                            
            return ConversationMetadata(
                conversation_id=conv_id or hashlib.md5(data.get('title', '').encode()).hexdigest(),
                title=data.get('title', 'Untitled'),
                create_time=data.get('create_time', 0),
                update_time=data.get('update_time', 0),
                model_slug=model_slug,
                turn_count=turn_count,
                has_attachments=has_attachments,
                safe_urls=data.get('safe_urls', [])
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.stats['json_parse_errors'] += 1
            return None
    
    def classify_knowledge(self, turns: list, title: str) -> KnowledgeCategory:
        """Classify conversation into knowledge category."""
        # Combine title and first few turns for classification
        text_sample = title.lower()
        for turn in turns[:5]:
            text_sample += ' ' + turn.content.lower()[:500]
            
        # Score each category
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_sample)
            scores[category] = score
            
        # Return highest scoring category
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return KnowledgeCategory.GENERAL
    
    def extract_concepts(self, turns: list, title: str) -> list:
        """Extract knowledge atoms from conversation turns."""
        atoms = []
        full_text = title + '\n' + '\n'.join(t.content for t in turns)
        
        # Match BIZRA-specific terms
        for pattern, (term_name, definition) in self.BIZRA_TERMS.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                # Find the turn where this term appears
                source_turn = 0
                for i, turn in enumerate(turns):
                    if re.search(pattern, turn.content, re.IGNORECASE):
                        source_turn = i
                        break
                        
                atoms.append(KnowledgeAtom(
                    concept=term_name,
                    definition=definition,
                    source_turn=source_turn,
                    category=self._term_to_category(term_name),
                    confidence=0.8,  # High confidence for pattern match
                ))
                
        return atoms
    
    def _term_to_category(self, term: str) -> KnowledgeCategory:
        """Map term to appropriate category."""
        safety_terms = {'TMP', 'SCM', 'Crown_Verifier', 'Causal_Drag', 'Ihsan'}
        core_terms = {'HRM_MoE', 'HTDAG', 'PAT_SAT', 'PoI', 'Node0', 'BlockTree', 'BlockGraph', 'Causal_Fabric'}
        agent_terms = {'SAPE', 'MCP', 'A2A', 'ReflectorAgent', 'HostAgent'}
        
        if term in safety_terms:
            return KnowledgeCategory.SECURITY_SAFETY
        elif term in core_terms:
            return KnowledgeCategory.BIZRA_CORE
        elif term in agent_terms:
            return KnowledgeCategory.AI_AGENT_PATTERNS
        return KnowledgeCategory.GENERAL
    
    def compute_quality(self, turns: list, metadata: Optional[ConversationMetadata]) -> float:
        """
        Compute quality score for conversation.
        
        6 quality dimensions:
        - Content length (more = richer context)
        - Turn count (more exchanges = deeper exploration)
        - Model quality (gpt-5 > gpt-4 > gpt-3.5)
        - Has attachments (indicates serious work)
        - BIZRA term density
        - Recency (newer = more relevant)
        """
        score = 0.0
        
        # Content length (up to 0.2)
        total_chars = sum(len(t.content) for t in turns)
        score += min(0.2, total_chars / 50000)
        
        # Turn count (up to 0.15)
        turn_count = metadata.turn_count if metadata else len(turns)
        score += min(0.15, turn_count / 50 * 0.15)
        
        # Model quality (up to 0.2)
        if metadata:
            model_scores = {
                'gpt-5': 0.2, 'gpt-5-thinking': 0.2,
                'gpt-4': 0.15, 'gpt-4o': 0.15, 'gpt-4-turbo': 0.15,
                'gpt-3.5': 0.1, 'unknown': 0.05
            }
            for model, mscore in model_scores.items():
                if model in (metadata.model_slug or ''):
                    score += mscore
                    break
                    
        # Has attachments (0.1)
        if metadata and metadata.has_attachments:
            score += 0.1
            
        # BIZRA term density (up to 0.25)
        full_text = '\n'.join(t.content for t in turns)
        term_matches = 0
        for pattern in self.BIZRA_TERMS:
            if re.search(pattern, full_text, re.IGNORECASE):
                term_matches += 1
        score += min(0.25, term_matches / 10 * 0.25)
        
        # Recency (up to 0.1) - newer is better
        if metadata and metadata.create_time > 0:
            age_days = (datetime.now().timestamp() - metadata.create_time) / 86400
            recency = max(0, 1 - age_days / 365)  # 1.0 for today, 0 for year old
            score += recency * 0.1
            
        return round(min(1.0, score), 3)
    
    def extract_from_file(self, md_path: Path) -> Optional[KnowledgePacket]:
        """Extract knowledge from a single .md file."""
        try:
            # Read markdown content
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                
            if len(md_content) < 100:
                self.stats['skipped_empty'] += 1
                return None
                
            # Parse turns
            turns = self.parse_markdown_turns(md_content)
            if not turns:
                self.stats['skipped_no_turns'] += 1
                return None
                
            # Extract title
            title_match = self.TITLE_PATTERN.search(md_content)
            title = title_match.group(1) if title_match else md_path.stem
            
            # Get companion JSON metadata
            json_path = md_path.with_suffix('.json')
            metadata = self.extract_json_metadata(json_path)
            
            # Generate conversation ID for deduplication
            conv_id = metadata.conversation_id if metadata else hashlib.md5(
                (title + str(len(turns))).encode()
            ).hexdigest()
            
            # Check for duplicates
            if conv_id in self.seen_ids:
                self.stats['duplicates_skipped'] += 1
                return None
            self.seen_ids.add(conv_id)
            
            # Classify and extract
            category = self.classify_knowledge(turns, title)
            atoms = self.extract_concepts(turns, title)
            quality = self.compute_quality(turns, metadata)
            
            self.stats['processed'] += 1
            self.stats[f'category_{category.value}'] += 1
            
            return KnowledgePacket(
                source_file=str(md_path.relative_to(self.chat_data_root)),
                conversation_id=conv_id,
                title=title,
                category=category,
                turns=turns,
                atoms=atoms,
                quality_score=quality,
                metadata=metadata
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"  ⚠ Error processing {md_path.name}: {e}")
            return None
    
    def extract_all(self, limit: Optional[int] = None) -> Iterator[KnowledgePacket]:
        """
        Extract knowledge from all conversation files.
        
        Scans all subdirectories for .md files, deduplicates,
        and yields KnowledgePacket objects.
        """
        print(f"\n{'='*60}")
        print("BIZRA Chat Knowledge Extractor v1.0")
        print(f"{'='*60}")
        print(f"📂 Source: {self.chat_data_root}")
        
        # Find all .md files
        md_files = list(self.chat_data_root.glob('**/*.md'))
        print(f"📄 Found {len(md_files)} .md files")
        
        if limit:
            md_files = md_files[:limit]
            print(f"🔢 Limited to {limit} files")
            
        print(f"\n{'─'*60}")
        print("Processing conversations...")
        print(f"{'─'*60}")
        
        for i, md_path in enumerate(md_files):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(md_files)}")
                
            packet = self.extract_from_file(md_path)
            if packet:
                yield packet
                
        print(f"\n{'─'*60}")
        print("Extraction Statistics:")
        print(f"{'─'*60}")
        for key, value in sorted(self.stats.items()):
            print(f"  {key}: {value}")
    
    def enrich_lexicon(self, packets: list) -> dict:
        """
        Enrich LexiconLedger with discovered terms.
        
        Returns dict of new terms added.
        """
        new_terms = {}
        # LexiconLedger uses 'terms' attribute, not 'canonical_terms'
        existing_terms = {key for key in self.ledger.terms.keys()}
        
        for packet in packets:
            for atom in packet.atoms:
                if atom.concept not in existing_terms and atom.concept not in new_terms:
                    # Create new term entry
                    new_terms[atom.concept] = {
                        'name': atom.concept,
                        'definition': atom.definition or f"Extracted from: {packet.title}",
                        'source': f"chat_extraction:{packet.source_file}",
                        'category': atom.category.value,
                        'ihsan_alignment': {atom.category.value: 0.6}
                    }
                    
        print(f"\n📖 Found {len(new_terms)} new terms for Lexicon")
        return new_terms


def stream_memories_json(memories_path: Path) -> Iterator[dict]:
    """
    Stream large memories.json file in chunks.
    
    Handles >50MB files without loading entirely into memory.
    """
    try:
        # Try ijson for streaming if available
        import ijson
        
        with open(memories_path, 'rb') as f:
            for item in ijson.items(f, 'project_memories.item'):
                yield item
                
    except ImportError:
        # Fallback: load entire file (may use significant memory)
        print("  ⚠ ijson not installed, loading entire file...")
        with open(memories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Yield conversation memory
        if 'conversations_memory' in data:
            yield {'type': 'conversation_memory', 'content': data['conversations_memory']}
            
        # Yield project memories
        if 'project_memories' in data:
            for proj_id, memory in data['project_memories'].items():
                yield {'type': 'project_memory', 'project_id': proj_id, 'content': memory}


def generate_extraction_manifest(packets: list, output_path: Path) -> None:
    """Generate extraction manifest JSON."""
    manifest = {
        "extraction_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "dna_signature": "EXTRACT-7-3-6-9-00",
        "statistics": {
            "total_conversations": len(packets),
            "by_category": {},
            "avg_quality_score": 0,
            "total_atoms": 0,
            "unique_terms": set()
        },
        "packets": []
    }
    
    # Aggregate statistics
    for packet in packets:
        cat = packet.category.value
        manifest["statistics"]["by_category"][cat] = \
            manifest["statistics"]["by_category"].get(cat, 0) + 1
        manifest["statistics"]["total_atoms"] += len(packet.atoms)
        for atom in packet.atoms:
            manifest["statistics"]["unique_terms"].add(atom.concept)
        manifest["packets"].append(packet.to_dict())
        
    # Calculate averages
    if packets:
        manifest["statistics"]["avg_quality_score"] = round(
            sum(p.quality_score for p in packets) / len(packets), 3
        )
    manifest["statistics"]["unique_terms"] = list(manifest["statistics"]["unique_terms"])
    
    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"\n📋 Manifest written to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BIZRA Chat Knowledge Extractor')
    parser.add_argument('--dry-run', action='store_true', help='Parse without saving')
    parser.add_argument('--limit', type=int, default=None, help='Limit files to process')
    parser.add_argument('--chat-root', type=str, default=None, help='Path to chat data folder')
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    chat_root = Path(args.chat_root) if args.chat_root else \
                project_root / 'chat data sample'
                
    if not chat_root.exists():
        print(f"❌ Chat data folder not found: {chat_root}")
        return 1
        
    # Initialize extractor
    extractor = ChatKnowledgeExtractor(chat_root)
    
    # Extract all knowledge
    packets = list(extractor.extract_all(limit=args.limit))
    
    if not packets:
        print("\n⚠ No knowledge packets extracted")
        return 1
        
    # Report summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Processed: {len(packets)} conversations")
    print(f"📊 By category:")
    for cat in KnowledgeCategory:
        count = sum(1 for p in packets if p.category == cat)
        if count > 0:
            print(f"   {cat.value}: {count}")
            
    avg_quality = sum(p.quality_score for p in packets) / len(packets)
    print(f"⭐ Average quality: {avg_quality:.3f}")
    
    total_atoms = sum(len(p.atoms) for p in packets)
    print(f"🧬 Total atoms: {total_atoms}")
    
    # Enrich lexicon
    new_terms = extractor.enrich_lexicon(packets)
    for term_name, term_data in new_terms.items():
        print(f"   + {term_name}: {term_data['definition'][:50]}...")
        
    if not args.dry_run:
        # Generate manifest
        manifest_path = project_root / 'evidence' / 'chat_extraction_manifest.json'
        generate_extraction_manifest(packets, manifest_path)
        
    print(f"\n{'='*60}")
    print("✅ Extraction complete!")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
