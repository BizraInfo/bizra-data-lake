#!/usr/bin/env python3
"""
BIZRA HyperGraph RAG v1.0
=========================
HyperGraph-based Retrieval Augmented Generation for BIZRA Knowledge Mining.

Uses HyperGraphRAG (https://github.com/LHRLAB/HyperGraphRAG) concepts:
- Hyperedges: Knowledge segments connecting multiple entities
- Entity extraction with importance scoring (key_score)
- Completeness scoring for knowledge segments
- Vector storage for entities and hyperedges
- NetworkX-based graph storage

BIZRA-specific enhancements:
- Ihsān dimension scoring for entities
- SAPE module attribution
- Custom entity types (CONCEPT, AXIOM, PATTERN, etc.)
- Dual-agentic relationship extraction

Usage:
    python scripts/bizra_hypergraph_rag.py [--limit N] [--query "question"]
"""

import asyncio
import json
import os
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Union, Iterator
from collections import defaultdict
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA ENTITY TYPES (extends HyperGraphRAG defaults)
# ═══════════════════════════════════════════════════════════════════════════════

class BIZRAEntityType(str, Enum):
    """BIZRA-specific entity types for knowledge graph."""
    CONCEPT = "CONCEPT"           # Core BIZRA concepts (e.g., Ihsān, PoI)
    AXIOM = "AXIOM"               # Foundational principles
    PATTERN = "PATTERN"           # Design patterns (e.g., HRM-MoE)
    PROTOCOL = "PROTOCOL"         # Communication protocols (e.g., A2A, MCP)
    AGENT = "AGENT"               # Agent types (e.g., HostAgent, ReflectorAgent)
    METRIC = "METRIC"             # Measurement systems (e.g., TMP, SCM)
    TOKEN = "TOKEN"               # Token types (e.g., SEED, BLOOM)
    STRUCTURE = "STRUCTURE"       # Data structures (e.g., HTDAG, BlockGraph)
    DIMENSION = "DIMENSION"       # Ihsān dimensions
    MODULE = "MODULE"             # SAPE modules
    PERSON = "PERSON"             # Named individuals
    ORGANIZATION = "ORGANIZATION" # Organizations
    EVENT = "EVENT"               # Events or milestones


# ═══════════════════════════════════════════════════════════════════════════════
# IHSĀN DIMENSION SCORING
# ═══════════════════════════════════════════════════════════════════════════════

IHSAN_DIMENSIONS = {
    "correctness": "Accuracy and truth of the knowledge",
    "safety": "Alignment with human values and harm prevention",
    "user_benefit": "Direct value to users and stakeholders",
    "efficiency": "Resource optimization and performance",
    "auditability": "Transparency and verifiability",
    "anti_centralization": "Decentralization and sovereignty",
    "robustness": "Resilience and fault tolerance",
    "adl_fairness": "Justice and equitable distribution"
}

SAPE_MODULES = {
    "1-HouseOfWisdom": "Foundation knowledge synthesis",
    "2-PatternForge": "Design pattern recognition",
    "3-AxiomAnvil": "Principle extraction",
    "4-MetricMill": "Measurement and scoring",
    "5-AgentAssembly": "Agent behavior modeling",
    "6-ProtocolPulse": "Communication protocol analysis",
    "7-TensionStudio": "Conflict and tradeoff mapping"
}


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERGRAPH DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HyperEdge:
    """
    A hyperedge represents a knowledge segment connecting multiple entities.
    Unlike binary edges, hyperedges can connect N entities in one semantic unit.
    """
    id: str
    knowledge_segment: str          # The knowledge statement
    completeness_score: float       # 0-10 score for segment completeness
    entity_names: list              # List of entity names in this hyperedge
    source_id: str                  # Source chunk/file
    ihsan_scores: dict = field(default_factory=dict)
    sape_module: str = ""
    weight: float = 1.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Entity:
    """An entity extracted from BIZRA knowledge corpus."""
    id: str
    name: str
    entity_type: str
    description: str
    key_score: float                # 0-100 importance score
    source_id: str
    hyperedge_refs: list = field(default_factory=list)  # Hyperedges containing this entity
    ihsan_scores: dict = field(default_factory=dict)
    sape_module: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA PROMPTS (HyperGraphRAG-style)
# ═══════════════════════════════════════════════════════════════════════════════

BIZRA_PROMPTS = {
    "tuple_delimiter": "<|>",
    "record_delimiter": "##",
    "completion_delimiter": "<|COMPLETE|>",
    
    "entity_types": [
        "CONCEPT", "AXIOM", "PATTERN", "PROTOCOL", "AGENT", 
        "METRIC", "TOKEN", "STRUCTURE", "DIMENSION", "MODULE",
        "PERSON", "ORGANIZATION", "EVENT"
    ],
    
    "entity_extraction": """-Goal-
Given a BIZRA-related document, identify knowledge segments (hyperedges) and entities.
This is for building a Knowledge HyperGraph for the BIZRA dual-agentic framework.

-Steps-
1. Divide the text into complete knowledge segments. For each segment:
   - knowledge_segment: A sentence describing a complete piece of knowledge
   - completeness_score: 0-10 indicating how complete/self-contained the segment is
   Format: ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. For each entity in the knowledge segments, extract:
   - entity_name: Name (capitalized if English)
   - entity_type: One of {entity_types}
   - entity_description: Comprehensive description
   - key_score: 0-100 importance score for BIZRA ecosystem
   Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return all items using {record_delimiter} as delimiter.

-BIZRA Context-
BIZRA is a dual-agentic AI framework emphasizing:
- Ihsān (إحسان): Excellence principle with 8 dimensions
- SAPE: Symbolic AI Pattern Engine with 7 modules
- PoI: Proof of Impact measurement
- HRM-MoE: Hierarchical Reflective Mixture of Experts
- Node0: Genesis node for decentralized AI sovereignty

-Entity Type Guidelines-
- CONCEPT: Core ideas (Ihsān, Sovereignty, Causal Fabric)
- AXIOM: Foundational truths ("AI must be auditable")
- PATTERN: Design patterns (HRM-MoE, HTDAG, Dual-Agentic)
- PROTOCOL: Communication (A2A, MCP, PAT/SAT)
- AGENT: Agent types (HostAgent, ReflectorAgent, CrownVerifier)
- METRIC: Measurements (TMP, SCM, PoI, key_score)
- TOKEN: Tokens (SEED, BLOOM)
- STRUCTURE: Data structures (BlockGraph, HTDAG)
- DIMENSION: Ihsān dimensions (correctness, safety, etc.)
- MODULE: SAPE modules (HouseOfWisdom, PatternForge, etc.)

-Input Text-
{input_text}

-Output-
""",

    "extraction_examples": [
        """Example 1:
Input: "BIZRA implements Ihsān through eight dimensions including correctness and safety. The HRM-MoE pattern enables hierarchical reasoning."

Output:
("hyper-relation"<|>"BIZRA implements Ihsān through eight dimensions for AI excellence"<|>8)##("hyper-relation"<|>"HRM-MoE pattern enables hierarchical reasoning in BIZRA"<|>7)##("entity"<|>"BIZRA"<|>"CONCEPT"<|>"A dual-agentic AI framework emphasizing Islamic excellence principles"<|>95)##("entity"<|>"IHSĀN"<|>"CONCEPT"<|>"Arabic term for excellence, core ethical principle with 8 evaluation dimensions"<|>90)##("entity"<|>"HRM-MOE"<|>"PATTERN"<|>"Hierarchical Reflective Mixture of Experts for multi-level AI reasoning"<|>85)##("entity"<|>"CORRECTNESS"<|>"DIMENSION"<|>"Ihsān dimension measuring accuracy and truth of AI outputs"<|>70)##("entity"<|>"SAFETY"<|>"DIMENSION"<|>"Ihsān dimension ensuring AI alignment with human values"<|>75)<|COMPLETE|>""",
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERGRAPH STORAGE (NetworkX-based, HyperGraphRAG compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class BIZRAHyperGraph:
    """
    HyperGraph storage for BIZRA knowledge.
    
    Unlike standard graphs with binary edges, hypergraphs have hyperedges
    that can connect multiple nodes simultaneously.
    
    Implementation uses NetworkX with special hyperedge nodes.
    """
    
    def __init__(self, working_dir: str = "bizra_hypergraph_cache"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.entities: dict[str, Entity] = {}
        self.hyperedges: dict[str, HyperEdge] = {}
        
        # Entity name to ID mapping
        self.entity_name_to_id: dict[str, str] = {}
        
        # Statistics
        self.stats = {
            "total_entities": 0,
            "total_hyperedges": 0,
            "entity_types": defaultdict(int),
            "avg_completeness": 0.0
        }
        
    def compute_id(self, content: str, prefix: str = "") -> str:
        """Compute deterministic ID from content."""
        hash_val = hashlib.md5(content.strip().encode()).hexdigest()[:12]
        return f"{prefix}{hash_val}" if prefix else hash_val
    
    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity in the hypergraph."""
        entity_key = entity.name.upper()
        
        if entity_key in self.entity_name_to_id:
            # Merge with existing entity
            existing_id = self.entity_name_to_id[entity_key]
            existing = self.entities[existing_id]
            
            # Update description if longer
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
                
            # Update key_score if higher
            existing.key_score = max(existing.key_score, entity.key_score)
            
            # Merge hyperedge refs
            for ref in entity.hyperedge_refs:
                if ref not in existing.hyperedge_refs:
                    existing.hyperedge_refs.append(ref)
                    
            return existing_id
        else:
            # New entity
            entity.id = self.compute_id(entity.name, prefix="ent-")
            self.entities[entity.id] = entity
            self.entity_name_to_id[entity_key] = entity.id
            self.stats["total_entities"] += 1
            self.stats["entity_types"][entity.entity_type] += 1
            return entity.id
    
    def add_hyperedge(self, hyperedge: HyperEdge) -> str:
        """Add a hyperedge connecting multiple entities."""
        if not hyperedge.id:
            hyperedge.id = self.compute_id(hyperedge.knowledge_segment, prefix="he-")
            
        self.hyperedges[hyperedge.id] = hyperedge
        self.stats["total_hyperedges"] += 1
        
        # Update completeness average
        total = sum(he.completeness_score for he in self.hyperedges.values())
        self.stats["avg_completeness"] = total / len(self.hyperedges)
        
        # Link entities to this hyperedge
        for entity_name in hyperedge.entity_names:
            entity_key = entity_name.upper()
            if entity_key in self.entity_name_to_id:
                entity_id = self.entity_name_to_id[entity_key]
                if hyperedge.id not in self.entities[entity_id].hyperedge_refs:
                    self.entities[entity_id].hyperedge_refs.append(hyperedge.id)
                    
        return hyperedge.id
    
    def get_entity_hyperedges(self, entity_name: str) -> list[HyperEdge]:
        """Get all hyperedges containing an entity."""
        entity_key = entity_name.upper()
        if entity_key not in self.entity_name_to_id:
            return []
            
        entity_id = self.entity_name_to_id[entity_key]
        entity = self.entities[entity_id]
        
        return [self.hyperedges[he_id] for he_id in entity.hyperedge_refs 
                if he_id in self.hyperedges]
    
    def get_connected_entities(self, entity_name: str) -> list[Entity]:
        """Get all entities connected via hyperedges to a given entity."""
        hyperedges = self.get_entity_hyperedges(entity_name)
        connected_names = set()
        
        for he in hyperedges:
            connected_names.update(he.entity_names)
            
        # Remove self
        entity_key = entity_name.upper()
        connected_names.discard(entity_key)
        
        return [self.entities[self.entity_name_to_id[name]] 
                for name in connected_names 
                if name in self.entity_name_to_id]
    
    def to_dict(self) -> dict:
        """Export hypergraph to dictionary."""
        return {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "stats": dict(self.stats)
            },
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "hyperedges": {hid: h.to_dict() for hid, h in self.hyperedges.items()}
        }
    
    def save(self, filename: str = "hypergraph.json"):
        """Save hypergraph to file."""
        filepath = self.working_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return filepath
    
    def to_cypher(self) -> str:
        """Export to Neo4j Cypher format."""
        lines = [
            "// BIZRA HyperGraph - Neo4j Cypher Export",
            f"// Generated: {datetime.now().isoformat()}",
            f"// Entities: {len(self.entities)}, HyperEdges: {len(self.hyperedges)}",
            "",
            "// === CREATE ENTITIES ===",
        ]
        
        for entity in self.entities.values():
            props = {
                "id": entity.id,
                "name": entity.name,
                "description": entity.description.replace('"', '\\"')[:200],
                "key_score": entity.key_score,
                "source_id": entity.source_id
            }
            props_str = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' 
                                   for k, v in props.items())
            lines.append(f'CREATE (:{entity.entity_type} {{{props_str}}})')
        
        lines.append("")
        lines.append("// === CREATE HYPEREDGES ===")
        lines.append("// Note: Hyperedges are represented as intermediate nodes")
        
        for he in self.hyperedges.values():
            # Create hyperedge as a node
            he_props = {
                "id": he.id,
                "knowledge_segment": he.knowledge_segment[:100].replace('"', '\\"'),
                "completeness_score": he.completeness_score
            }
            props_str = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}' 
                                   for k, v in he_props.items())
            lines.append(f'CREATE (:HYPEREDGE {{{props_str}}})')
            
            # Create relationships from hyperedge to entities
            for entity_name in he.entity_names:
                lines.append(
                    f'MATCH (h:HYPEREDGE {{id: "{he.id}"}}), '
                    f'(e {{name: "{entity_name.upper()}"}}) '
                    f'CREATE (h)-[:CONNECTS]->(e)'
                )
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA EXTRACTION PATTERNS (Regex-based for offline use)
# ═══════════════════════════════════════════════════════════════════════════════

class BIZRAPatternExtractor:
    """
    Extract BIZRA entities and relationships using regex patterns.
    Works offline without LLM for basic extraction.
    """
    
    # Entity patterns with type hints
    ENTITY_PATTERNS = [
        # Core concepts
        (r'\bBIZRA\b', "BIZRA", "CONCEPT", 95),
        (r'\bIhsān\b|إحسان', "IHSĀN", "CONCEPT", 90),
        (r'\bNode\s*0\b|Genesis\s+Node', "NODE0", "STRUCTURE", 85),
        (r'\bPoI\b|Proof[- ]of[- ]Impact', "POI", "METRIC", 80),
        
        # Patterns
        (r'\bHRM[- ]?MoE\b', "HRM_MOE", "PATTERN", 85),
        (r'\bHTDAG\b', "HTDAG", "STRUCTURE", 80),
        (r'\bSAPE\b', "SAPE", "PATTERN", 85),
        (r'\bdual[- ]?agentic\b', "DUAL_AGENTIC", "PATTERN", 80),
        
        # Protocols
        (r'\bA2A\b|Agent[- ]to[- ]Agent', "A2A", "PROTOCOL", 75),
        (r'\bMCP\b|Model\s+Context\s+Protocol', "MCP", "PROTOCOL", 75),
        (r'\bPAT\s*/?\s*SAT\b', "PAT_SAT", "PROTOCOL", 70),
        
        # Agents
        (r'\bHost\s*Agent\b', "HOST_AGENT", "AGENT", 70),
        (r'\bReflector\s*Agent\b', "REFLECTOR_AGENT", "AGENT", 70),
        (r'\bCrown\s*Verifier\b', "CROWN_VERIFIER", "AGENT", 75),
        
        # Metrics
        (r'\bTMP\b|Temporal\s+Measurement', "TMP", "METRIC", 65),
        (r'\bSCM\b|Structured\s+Cognitive\s+Metric', "SCM", "METRIC", 65),
        
        # Tokens
        (r'\bSEED\s+token\b', "SEED_TOKEN", "TOKEN", 70),
        (r'\bBLOOM\s+token\b', "BLOOM_TOKEN", "TOKEN", 70),
        
        # Structures
        (r'\bBlock[- ]?Graph\b', "BLOCKGRAPH", "STRUCTURE", 65),
        (r'\bCausal\s+Fabric\b', "CAUSAL_FABRIC", "STRUCTURE", 70),
        
        # Dimensions
        (r'\bcorrectness\b', "CORRECTNESS", "DIMENSION", 50),
        (r'\bsafety\b', "SAFETY", "DIMENSION", 55),
        (r'\bauditability\b', "AUDITABILITY", "DIMENSION", 50),
        (r'\brobustness\b', "ROBUSTNESS", "DIMENSION", 50),
        (r'\banti[- ]?centralization\b', "ANTI_CENTRALIZATION", "DIMENSION", 55),
        
        # SAPE Modules
        (r'\bHouse\s+of\s+Wisdom\b', "HOUSE_OF_WISDOM", "MODULE", 60),
        (r'\bPattern\s*Forge\b', "PATTERN_FORGE", "MODULE", 55),
        (r'\bAxiom\s*Anvil\b', "AXIOM_ANVIL", "MODULE", 55),
    ]
    
    # Knowledge segment patterns (for hyperedges)
    KNOWLEDGE_PATTERNS = [
        # Definitional statements
        r'(?P<subject>[A-Z][a-zA-Z_-]+)\s+(?:is|are|means|represents?|defines?)\s+(?P<definition>[^.!?]+[.!?])',
        
        # Implementation statements
        r'(?P<system>[A-Z][a-zA-Z_-]+)\s+(?:implements?|uses?|employs?|leverages?)\s+(?P<mechanism>[^.!?]+[.!?])',
        
        # Relationship statements
        r'(?P<entity1>[A-Z][a-zA-Z_-]+)\s+(?:enables?|supports?|connects?\s+to|depends?\s+on)\s+(?P<entity2>[^.!?]+[.!?])',
        
        # Principle statements
        r'(?:The\s+)?(?P<principle>[A-Z][a-zA-Z_-]+)\s+principle\s+(?:states?|requires?|ensures?)\s+(?P<content>[^.!?]+[.!?])',
    ]
    
    def __init__(self):
        self.compiled_entity_patterns = [
            (re.compile(pattern, re.IGNORECASE), name, etype, score)
            for pattern, name, etype, score in self.ENTITY_PATTERNS
        ]
        self.compiled_knowledge_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.KNOWLEDGE_PATTERNS
        ]
    
    def extract_entities(self, text: str, source_id: str = "") -> list[Entity]:
        """Extract entities from text using patterns."""
        entities = []
        seen = set()
        
        for pattern, name, etype, score in self.compiled_entity_patterns:
            if pattern.search(text) and name not in seen:
                seen.add(name)
                
                # Try to extract context/description
                description = self._extract_context(text, name)
                
                entities.append(Entity(
                    id="",
                    name=name,
                    entity_type=etype,
                    description=description,
                    key_score=score,
                    source_id=source_id
                ))
        
        return entities
    
    def extract_hyperedges(self, text: str, source_id: str = "") -> list[HyperEdge]:
        """Extract knowledge segments as hyperedges."""
        hyperedges = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Find entities in this sentence
            entity_names = []
            for pattern, name, _, _ in self.compiled_entity_patterns:
                if pattern.search(sentence):
                    entity_names.append(name)
            
            if len(entity_names) >= 2:  # Only create hyperedge if 2+ entities
                # Score completeness based on sentence structure
                completeness = self._score_completeness(sentence)
                
                hyperedges.append(HyperEdge(
                    id="",
                    knowledge_segment=sentence.strip(),
                    completeness_score=completeness,
                    entity_names=entity_names,
                    source_id=source_id
                ))
        
        return hyperedges
    
    def _extract_context(self, text: str, entity_name: str) -> str:
        """Extract description context for an entity."""
        # Look for definitional patterns
        patterns = [
            rf'{entity_name}\s+(?:is|are|means)\s+([^.!?]+[.!?])',
            rf'{entity_name}[:\s]+([^.!?\n]+[.!?]?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]
        
        return f"BIZRA entity: {entity_name}"
    
    def _score_completeness(self, sentence: str) -> float:
        """Score knowledge segment completeness (0-10)."""
        score = 5.0  # Base score
        
        # Boost for definitional words
        if re.search(r'\b(is|are|means|represents|defines)\b', sentence, re.I):
            score += 1.5
            
        # Boost for causal words
        if re.search(r'\b(because|therefore|enables|causes|results)\b', sentence, re.I):
            score += 1.0
            
        # Boost for completeness indicators
        if re.search(r'\b(complete|comprehensive|full|all)\b', sentence, re.I):
            score += 0.5
            
        # Boost for numeric specifics
        if re.search(r'\d+', sentence):
            score += 0.5
            
        # Penalize for uncertain language
        if re.search(r'\b(maybe|possibly|might|could)\b', sentence, re.I):
            score -= 1.0
            
        return min(10.0, max(0.0, score))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HYPERGRAPH RAG CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BIZRAHyperGraphRAG:
    """
    Main class for BIZRA HyperGraph RAG.
    
    Follows HyperGraphRAG patterns:
    - insert() for knowledge ingestion
    - query() for retrieval
    - Entity extraction with importance scoring
    - Hyperedge-based knowledge segments
    """
    
    def __init__(
        self,
        working_dir: str = "bizra_hypergraph_cache",
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Storage
        self.hypergraph = BIZRAHyperGraph(working_dir)
        self.extractor = BIZRAPatternExtractor()
        
        # Processing stats
        self.stats = {
            "documents_processed": 0,
            "chunks_processed": 0,
            "entities_extracted": 0,
            "hyperedges_extracted": 0,
            "sources": []
        }
        
        print(f"🔷 BIZRA HyperGraph RAG initialized")
        print(f"   Working dir: {self.working_dir}")
    
    def _chunk_text(self, text: str) -> Iterator[str]:
        """Chunk text with overlap."""
        words = text.split()
        
        if len(words) <= self.chunk_size:
            yield text
            return
            
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            yield chunk
            start = end - self.chunk_overlap
    
    def insert(self, string_or_strings: Union[str, list[str]], source_id: str = ""):
        """
        Insert documents into the hypergraph.
        
        Args:
            string_or_strings: Single document or list of documents
            source_id: Optional source identifier
        """
        if isinstance(string_or_strings, str):
            documents = [string_or_strings]
        else:
            documents = string_or_strings
            
        for doc_idx, doc in enumerate(documents):
            doc_source = source_id or f"doc-{doc_idx}"
            
            # Chunk the document
            for chunk_idx, chunk in enumerate(self._chunk_text(doc)):
                chunk_source = f"{doc_source}#chunk{chunk_idx}"
                
                # Extract entities
                entities = self.extractor.extract_entities(chunk, chunk_source)
                for entity in entities:
                    self.hypergraph.add_entity(entity)
                    self.stats["entities_extracted"] += 1
                
                # Extract hyperedges
                hyperedges = self.extractor.extract_hyperedges(chunk, chunk_source)
                for he in hyperedges:
                    self.hypergraph.add_hyperedge(he)
                    self.stats["hyperedges_extracted"] += 1
                
                self.stats["chunks_processed"] += 1
            
            self.stats["documents_processed"] += 1
            self.stats["sources"].append(doc_source)
        
        return self.stats
    
    def insert_from_files(self, directory: Path, pattern: str = "*.md", limit: int = None):
        """Insert documents from a directory."""
        directory = Path(directory)
        files = list(directory.rglob(pattern))
        
        if limit:
            files = files[:limit]
            
        print(f"📂 Processing {len(files)} files from {directory}")
        
        for i, filepath in enumerate(files):
            try:
                # Try multiple encodings
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        content = filepath.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"  ⚠ Could not decode: {filepath.name}")
                    continue
                
                self.insert(content, source_id=str(filepath.relative_to(directory)))
                
                if (i + 1) % 25 == 0:
                    print(f"  Progress: {i+1}/{len(files)} "
                          f"({self.stats['entities_extracted']} entities, "
                          f"{self.stats['hyperedges_extracted']} hyperedges)")
                    
            except Exception as e:
                print(f"  ❌ Error processing {filepath.name}: {e}")
        
        return self.stats
    
    def query(self, query_text: str, top_k: int = 5) -> dict:
        """
        Query the hypergraph for relevant knowledge.
        
        Args:
            query_text: Natural language query
            top_k: Number of top results to return
        """
        # Extract entities from query
        query_entities = self.extractor.extract_entities(query_text, "query")
        
        results = {
            "query": query_text,
            "matched_entities": [],
            "relevant_hyperedges": [],
            "connected_entities": []
        }
        
        for qe in query_entities:
            entity_key = qe.name.upper()
            
            if entity_key in self.hypergraph.entity_name_to_id:
                entity_id = self.hypergraph.entity_name_to_id[entity_key]
                entity = self.hypergraph.entities[entity_id]
                results["matched_entities"].append(entity.to_dict())
                
                # Get hyperedges
                hyperedges = self.hypergraph.get_entity_hyperedges(qe.name)
                for he in hyperedges[:top_k]:
                    results["relevant_hyperedges"].append(he.to_dict())
                
                # Get connected entities
                connected = self.hypergraph.get_connected_entities(qe.name)
                for ce in connected[:top_k]:
                    results["connected_entities"].append(ce.to_dict())
        
        return results
    
    def save(self):
        """Save the hypergraph to disk."""
        json_path = self.hypergraph.save("hypergraph.json")
        
        # Also save Cypher
        cypher_path = self.working_dir / "hypergraph.cypher"
        cypher_path.write_text(self.hypergraph.to_cypher(), encoding='utf-8')
        
        # Save stats
        stats_path = self.working_dir / "extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                **self.stats,
                "hypergraph_stats": dict(self.hypergraph.stats)
            }, f, indent=2)
        
        return {
            "json": json_path,
            "cypher": cypher_path,
            "stats": stats_path
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA HyperGraph RAG")
    parser.add_argument("--source", type=str, default=None,
                        help="Source directory for documents")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of files to process")
    parser.add_argument("--query", type=str, default=None,
                        help="Query the hypergraph")
    parser.add_argument("--working-dir", type=str, default="evidence/bizra_hypergraph",
                        help="Working directory for cache")
    args = parser.parse_args()
    
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("\n" + "=" * 60)
    print("BIZRA HyperGraph RAG v1.0")
    print("=" * 60)
    
    # Initialize RAG
    working_dir = project_root / args.working_dir
    rag = BIZRAHyperGraphRAG(working_dir=str(working_dir))
    
    # Determine source directory
    if args.source:
        source_dir = Path(args.source)
    else:
        source_dir = project_root / "chat data sample"
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return 1
    
    # Insert documents
    print(f"\n📂 Source: {source_dir}")
    stats = rag.insert_from_files(source_dir, pattern="*.md", limit=args.limit)
    
    # Print stats
    print("\n" + "=" * 60)
    print("HYPERGRAPH STATISTICS")
    print("=" * 60)
    print(f"✅ Documents processed: {stats['documents_processed']}")
    print(f"📦 Chunks processed: {stats['chunks_processed']}")
    print(f"🧬 Entities extracted: {stats['entities_extracted']}")
    print(f"🔗 HyperEdges extracted: {stats['hyperedges_extracted']}")
    
    print(f"\n📊 Entity types:")
    for etype, count in sorted(rag.hypergraph.stats["entity_types"].items()):
        print(f"   {etype}: {count}")
    
    print(f"\n📈 Avg completeness score: {rag.hypergraph.stats['avg_completeness']:.2f}")
    
    # Save outputs
    outputs = rag.save()
    print(f"\n📄 JSON saved: {outputs['json']}")
    print(f"📄 Cypher saved: {outputs['cypher']}")
    print(f"📄 Stats saved: {outputs['stats']}")
    
    # Run query if provided
    if args.query:
        print("\n" + "-" * 60)
        print(f"🔍 Query: {args.query}")
        print("-" * 60)
        
        results = rag.query(args.query)
        
        print(f"\n📍 Matched entities: {len(results['matched_entities'])}")
        for e in results['matched_entities'][:3]:
            print(f"   - {e['name']} ({e['entity_type']}): {e['description'][:50]}...")
        
        print(f"\n🔗 Relevant hyperedges: {len(results['relevant_hyperedges'])}")
        for he in results['relevant_hyperedges'][:3]:
            print(f"   - [{he['completeness_score']:.1f}] {he['knowledge_segment'][:60]}...")
        
        print(f"\n🌐 Connected entities: {len(results['connected_entities'])}")
        for ce in results['connected_entities'][:5]:
            print(f"   - {ce['name']} ({ce['entity_type']})")
    
    print("\n" + "=" * 60)
    print("✅ BIZRA HyperGraph RAG complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
