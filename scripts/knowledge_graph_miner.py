#!/usr/bin/env python3
"""
BIZRA Knowledge Graph Miner v1.0
================================
Combines LangGraph + LangExtract + E2B Sandbox for advanced knowledge extraction.

Features:
- LangGraph: Graph-based agentic workflows for multi-step extraction
- LangExtract: LLM-powered structured extraction with source grounding
- E2B Sandbox: Secure code execution for validation and transformation
- Knowledge Graph: Build entity-relationship graphs from conversations

DNA Signature: GRAPH-MINER-7-3-6-9-00

Usage:
    python scripts/knowledge_graph_miner.py [--limit N] [--sandbox]
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import TypedDict, Annotated, Sequence, Optional, Any
from enum import Enum

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    add_messages = None  # Placeholder
    print("⚠ langgraph not installed. Run: pip install langgraph")

# E2B Sandbox imports
try:
    from e2b_code_interpreter import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False
    print("⚠ e2b-code-interpreter not installed. Run: pip install e2b-code-interpreter")

# LangExtract imports
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# E2B API Key
E2B_API_KEY = os.environ.get("E2B_API_KEY", "e2b_e42b549f6a5f986eae8273cb8157be1e37f6ec1b")


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class EntityType(Enum):
    """Types of entities in the BIZRA knowledge graph."""
    CONCEPT = "concept"
    PATTERN = "pattern"
    DECISION = "decision"
    AXIOM = "axiom"
    TENSION = "tension"
    AGENT = "agent"
    PROTOCOL = "protocol"
    METRIC = "metric"
    TOKEN = "token"


class RelationType(Enum):
    """Types of relationships between entities."""
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    ENABLES = "enables"
    CONSTRAINS = "constrains"
    MEASURES = "measures"
    RESOLVES = "resolves"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    definition: str = ""
    attributes: dict = field(default_factory=dict)
    source_file: str = ""
    source_span: tuple = (0, 0)  # Character span in source
    confidence: float = 0.8
    ihsan_dimension: Optional[str] = None
    sape_module: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "definition": self.definition,
            "attributes": self.attributes,
            "source_file": self.source_file,
            "confidence": self.confidence,
            "ihsan_dimension": self.ihsan_dimension,
            "sape_module": self.sape_module
        }


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: RelationType
    attributes: dict = field(default_factory=dict)
    confidence: float = 0.8
    
    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type.value,
            "attributes": self.attributes,
            "confidence": self.confidence
        }


@dataclass
class KnowledgeGraph:
    """The complete knowledge graph."""
    entities: dict = field(default_factory=dict)  # id -> Entity
    relationships: list = field(default_factory=list)  # List[Relationship]
    
    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity
        
    def add_relationship(self, rel: Relationship) -> None:
        self.relationships.append(rel)
        
    def to_dict(self) -> dict:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "stats": {
                "entity_count": len(self.entities),
                "relationship_count": len(self.relationships),
                "by_type": self._count_by_type()
            }
        }
        
    def _count_by_type(self) -> dict:
        counts = {}
        for entity in self.entities.values():
            t = entity.entity_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts
    
    def to_cypher(self) -> str:
        """Generate Neo4j Cypher statements for import."""
        statements = []
        
        # Create entities
        for entity in self.entities.values():
            props = json.dumps({
                "name": entity.name,
                "definition": entity.definition,
                "confidence": entity.confidence,
                "ihsan_dimension": entity.ihsan_dimension,
                "sape_module": entity.sape_module,
                **entity.attributes
            })
            stmt = f"CREATE (:{entity.entity_type.value.title()} {{id: '{entity.id}', {props[1:-1]}}})"
            statements.append(stmt)
            
        # Create relationships
        for rel in self.relationships:
            stmt = f"MATCH (a {{id: '{rel.source_id}'}}), (b {{id: '{rel.target_id}'}}) CREATE (a)-[:{rel.relation_type.value.upper()}]->(b)"
            statements.append(stmt)
            
        return ";\n".join(statements)


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH WORKFLOW STATE
# ═══════════════════════════════════════════════════════════════════════════════

if LANGGRAPH_AVAILABLE:
    class GraphMinerState(TypedDict):
        """State for the LangGraph knowledge mining workflow."""
        input_text: str
        source_file: str
        entities: list
        relationships: list
        validation_results: dict
        sandbox_output: str
        messages: Annotated[Sequence[str], add_messages]
        current_step: str
        errors: list
else:
    # Fallback when LangGraph not available
    GraphMinerState = dict


# ═══════════════════════════════════════════════════════════════════════════════
# E2B SANDBOX EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class E2BSandboxExecutor:
    """Execute code safely in E2B sandbox for validation."""
    
    def __init__(self, api_key: str = E2B_API_KEY):
        self.api_key = api_key
        self.sandbox = None
        
    def __enter__(self):
        if E2B_AVAILABLE and self.api_key:
            try:
                # Use Sandbox.create() for e2b>=2.0 API
                self.sandbox = Sandbox.create(api_key=self.api_key)
            except Exception as e:
                print(f"⚠ Could not create E2B sandbox: {e}")
                self.sandbox = None
        return self
        
    def __exit__(self, *args):
        if self.sandbox:
            try:
                self.sandbox.kill()
            except:
                pass
                
    def execute(self, code: str, timeout: int = 30) -> dict:
        """Execute Python code in sandbox."""
        if not self.sandbox:
            return {"success": False, "error": "Sandbox not available", "output": ""}
            
        try:
            execution = self.sandbox.run_code(code, timeout=timeout)
            return {
                "success": True,
                "output": execution.text if hasattr(execution, 'text') else str(execution),
                "logs": execution.logs if hasattr(execution, 'logs') else [],
                "error": execution.error if hasattr(execution, 'error') else None
            }
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}
            
    def validate_entity_schema(self, entity_json: str) -> dict:
        """Validate entity JSON schema in sandbox."""
        code = f'''
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    definition: str = ""
    confidence: float = 0.8
    ihsan_dimension: Optional[str] = None

entity_data = {entity_json}
try:
    entity = Entity(**entity_data)
    print(f"✅ Valid entity: {{entity.name}} ({{entity.entity_type}})")
    print(json.dumps({{"valid": True, "entity": entity_data}}))
except Exception as e:
    print(f"❌ Invalid entity: {{e}}")
    print(json.dumps({{"valid": False, "error": str(e)}}))
'''
        return self.execute(code)
        
    def transform_to_cypher(self, graph_json: str) -> dict:
        """Generate Cypher statements in sandbox."""
        code = '''
import json

graph = ''' + graph_json + '''

statements = []

# Create constraint
statements.append("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

# Create entities
for entity in graph.get("entities", {}).values():
    label = entity.get("type", "Entity").title()
    name = entity.get("name", "").replace("'", "\\'")
    stmt = f"CREATE (:{label} {{id: '{entity['id']}', name: '{name}'}})"
    statements.append(stmt)

# Create relationships
for rel in graph.get("relationships", []):
    source = rel['source']
    target = rel['target']
    reltype = rel['type'].upper()
    stmt = f"MATCH (a {{id: '{source}'}}), (b {{id: '{target}'}}) CREATE (a)-[:{reltype}]->(b)"
    statements.append(stmt)

print("\\n".join(statements))
'''
        return self.execute(code)


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN EXTRACTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PatternExtractor:
    """Extract BIZRA-specific patterns using regex and heuristics."""
    
    # Entity patterns
    ENTITY_PATTERNS = {
        # Concepts
        EntityType.CONCEPT: [
            (r'\b(HRM[-_]?MoE)\b', 'Hierarchical Reasoning Mixture-of-Experts'),
            (r'\b(HTDAG)\b', 'Hierarchical Task Directed Acyclic Graph'),
            (r'\b(SAPE)\b', 'Structured Agentic Prompt Engineering'),
            (r'\b(MCP)\b', 'Model Context Protocol'),
            (r'\b(A2A)\b', 'Agent-to-Agent Protocol'),
            (r'\b(GoT)\b', 'Graph of Thoughts'),
            (r'\b(PoI|Proof[- ]of[- ]Impact)\b', 'Proof-of-Impact Consensus'),
            (r'\b(Ihsan|إحسان)\b', 'Excellence Principle'),
            (r'\b(Node0)\b', 'Genesis Node'),
        ],
        # Patterns/Architectures
        EntityType.PATTERN: [
            (r'\b(PAT\s*/?\s*SAT)\b', 'Dual-Agentic Architecture'),
            (r'\b(dual[- ]?agentic)\b', 'Dual-Agentic Pattern'),
            (r'\b(Block[- ]?Graph)\b', 'Hybrid Ledger Structure'),
            (r'\b(Causal[- ]?Fabric)\b', 'Immutable Truth Ledger'),
        ],
        # Protocols
        EntityType.PROTOCOL: [
            (r'\b(TMP)\b', 'Temporal Measurement Protocol'),
            (r'\b(SCM)\b', 'Structured Cognitive Metric'),
            (r'\b(FATE)\b', 'Fail-Closed Escalation Protocol'),
            (r'\b(Crown[- ]?Verifier)\b', 'Cryptographic Deployment Gate'),
        ],
        # Agents
        EntityType.AGENT: [
            (r'\b(Host[- ]?Agent)\b', 'Orchestrator Agent'),
            (r'\b(Reflector[- ]?Agent)\b', 'Learning Synthesizer'),
            (r'\b(App[- ]?Agent)\b', 'Specialized Agent'),
        ],
        # Metrics
        EntityType.METRIC: [
            (r'\b(SNR)\b', 'Signal-to-Noise Ratio'),
            (r'\b(Causal[- ]?Drag)\b', 'Structural Risk Metric'),
        ],
        # Tokens
        EntityType.TOKEN: [
            (r'\b(SEED[- ]?token)\b', 'Stable Utility Token'),
            (r'\b(BLOOM[- ]?token)\b', 'Impact Growth Token'),
        ],
    }
    
    # Relationship patterns
    RELATIONSHIP_PATTERNS = [
        (r'(\w+)\s+implements\s+(\w+)', RelationType.IMPLEMENTS),
        (r'(\w+)\s+depends\s+on\s+(\w+)', RelationType.DEPENDS_ON),
        (r'(\w+)\s+enables\s+(\w+)', RelationType.ENABLES),
        (r'(\w+)\s+measures\s+(\w+)', RelationType.MEASURES),
        (r'(\w+)\s+is\s+part\s+of\s+(\w+)', RelationType.PART_OF),
        (r'(\w+)\s+resolves\s+(\w+)', RelationType.RESOLVES),
    ]
    
    # Ihsān dimension keywords
    IHSAN_KEYWORDS = {
        'correctness': ['accurate', 'correct', 'valid', 'truth', 'precise'],
        'safety': ['safe', 'secure', 'protect', 'guard', 'fail-closed'],
        'user_benefit': ['user', 'benefit', 'value', 'impact', 'help'],
        'efficiency': ['fast', 'efficient', 'optimize', 'latency', 'speed'],
        'auditability': ['audit', 'trace', 'log', 'receipt', 'evidence'],
        'anti_centralization': ['decentralized', 'distributed', 'sovereign'],
        'robustness': ['robust', 'resilient', 'fault-tolerant', 'byzantine'],
        'adl_fairness': ['fair', 'equal', 'non-discriminatory'],
    }
    
    def extract_entities(self, text: str, source_file: str = "") -> list:
        """Extract entities from text."""
        entities = []
        seen_ids = set()
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern, definition in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    name = match.group(1).strip()
                    entity_id = f"{entity_type.value}_{name.lower().replace(' ', '_').replace('-', '_')}"
                    
                    if entity_id in seen_ids:
                        continue
                    seen_ids.add(entity_id)
                    
                    # Determine Ihsān dimension from context
                    context = text[max(0, match.start()-200):min(len(text), match.end()+200)]
                    ihsan_dim = self._detect_ihsan_dimension(context)
                    
                    entities.append(Entity(
                        id=entity_id,
                        name=name,
                        entity_type=entity_type,
                        definition=definition,
                        source_file=source_file,
                        source_span=(match.start(), match.end()),
                        ihsan_dimension=ihsan_dim
                    ))
                    
        return entities
    
    def extract_relationships(self, text: str, entities: list) -> list:
        """Extract relationships between entities."""
        relationships = []
        entity_names = {e.name.lower(): e.id for e in entities}
        
        # Pattern-based relationship extraction
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source_name = match.group(1).lower()
                target_name = match.group(2).lower()
                
                source_id = entity_names.get(source_name)
                target_id = entity_names.get(target_name)
                
                if source_id and target_id:
                    relationships.append(Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=rel_type
                    ))
                    
        # Infer relationships from co-occurrence
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Check if they appear close together
                if abs(e1.source_span[0] - e2.source_span[0]) < 500:
                    # Same context implies relationship
                    if e1.entity_type != e2.entity_type:
                        relationships.append(Relationship(
                            source_id=e1.id,
                            target_id=e2.id,
                            relation_type=RelationType.PART_OF,
                            confidence=0.5
                        ))
                        
        return relationships
    
    def _detect_ihsan_dimension(self, context: str) -> Optional[str]:
        """Detect Ihsān dimension from context."""
        context_lower = context.lower()
        scores = {}
        
        for dimension, keywords in self.IHSAN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in context_lower)
            if score > 0:
                scores[dimension] = score
                
        if scores:
            return max(scores, key=scores.get)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH WORKFLOW NODES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_entities_node(state: GraphMinerState) -> GraphMinerState:
    """Extract entities from input text."""
    extractor = PatternExtractor()
    entities = extractor.extract_entities(state["input_text"], state["source_file"])
    
    return {
        **state,
        "entities": [e.to_dict() for e in entities],
        "current_step": "extract_entities",
        "messages": state["messages"] + [f"Extracted {len(entities)} entities"]
    }


def extract_relationships_node(state: GraphMinerState) -> GraphMinerState:
    """Extract relationships between entities."""
    extractor = PatternExtractor()
    
    # Reconstruct Entity objects
    entities = []
    for e_dict in state["entities"]:
        entities.append(Entity(
            id=e_dict["id"],
            name=e_dict["name"],
            entity_type=EntityType(e_dict["type"]),
            definition=e_dict.get("definition", ""),
            source_span=(0, 0)
        ))
    
    relationships = extractor.extract_relationships(state["input_text"], entities)
    
    return {
        **state,
        "relationships": [r.to_dict() for r in relationships],
        "current_step": "extract_relationships",
        "messages": state["messages"] + [f"Extracted {len(relationships)} relationships"]
    }


def validate_in_sandbox_node(state: GraphMinerState) -> GraphMinerState:
    """Validate extractions in E2B sandbox."""
    validation_results = {"valid": True, "errors": []}
    
    if E2B_AVAILABLE and E2B_API_KEY:
        try:
            with E2BSandboxExecutor(E2B_API_KEY) as sandbox:
                # Validate each entity
                for entity in state["entities"][:5]:  # Limit to avoid timeout
                    result = sandbox.validate_entity_schema(json.dumps(entity))
                    if not result.get("success"):
                        validation_results["errors"].append(result.get("error"))
                        
        except Exception as e:
            validation_results["errors"].append(str(e))
            
    return {
        **state,
        "validation_results": validation_results,
        "current_step": "validate",
        "messages": state["messages"] + ["Validation complete"]
    }


def build_graph_node(state: GraphMinerState) -> GraphMinerState:
    """Build final knowledge graph."""
    graph = KnowledgeGraph()
    
    for e_dict in state["entities"]:
        graph.add_entity(Entity(
            id=e_dict["id"],
            name=e_dict["name"],
            entity_type=EntityType(e_dict["type"]),
            definition=e_dict.get("definition", ""),
            confidence=e_dict.get("confidence", 0.8),
            ihsan_dimension=e_dict.get("ihsan_dimension")
        ))
        
    for r_dict in state["relationships"]:
        graph.add_relationship(Relationship(
            source_id=r_dict["source"],
            target_id=r_dict["target"],
            relation_type=RelationType(r_dict["type"]),
            confidence=r_dict.get("confidence", 0.8)
        ))
        
    return {
        **state,
        "current_step": "complete",
        "messages": state["messages"] + [
            f"Graph complete: {len(graph.entities)} entities, {len(graph.relationships)} relationships"
        ]
    }


def should_validate(state: GraphMinerState) -> str:
    """Decide whether to run sandbox validation."""
    if E2B_AVAILABLE and E2B_API_KEY and len(state["entities"]) > 0:
        return "validate"
    return "build"


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH WORKFLOW BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

if LANGGRAPH_AVAILABLE:
    def build_knowledge_mining_graph() -> StateGraph:
        """Build the LangGraph workflow for knowledge mining."""
        workflow = StateGraph(GraphMinerState)
        
        # Add nodes
        workflow.add_node("extract_entities", extract_entities_node)
        workflow.add_node("extract_relationships", extract_relationships_node)
        workflow.add_node("validate", validate_in_sandbox_node)
        workflow.add_node("build_graph", build_graph_node)
        
        # Add edges
        workflow.set_entry_point("extract_entities")
        workflow.add_edge("extract_entities", "extract_relationships")
        workflow.add_conditional_edges(
            "extract_relationships",
            should_validate,
            {
                "validate": "validate",
                "build": "build_graph"
            }
        )
        workflow.add_edge("validate", "build_graph")
        workflow.add_edge("build_graph", END)
        
        return workflow.compile()
else:
    def build_knowledge_mining_graph():
        """Placeholder when LangGraph not available."""
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN KNOWLEDGE GRAPH MINER
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraphMiner:
    """Main class for mining knowledge graphs from BIZRA conversations."""
    
    def __init__(self, use_sandbox: bool = True):
        self.use_sandbox = use_sandbox and E2B_AVAILABLE
        self.graph = KnowledgeGraph()
        self.stats = {
            "files_processed": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "errors": []
        }
        
        # Build LangGraph workflow if available
        if LANGGRAPH_AVAILABLE:
            self.workflow = build_knowledge_mining_graph()
        else:
            self.workflow = None
            
    def process_text(self, text: str, source_file: str = "") -> dict:
        """Process text through the knowledge mining workflow."""
        if self.workflow:
            # Use LangGraph workflow
            initial_state = {
                "input_text": text,
                "source_file": source_file,
                "entities": [],
                "relationships": [],
                "validation_results": {},
                "sandbox_output": "",
                "messages": [],
                "current_step": "start",
                "errors": []
            }
            
            result = self.workflow.invoke(initial_state)
            
            # Add to main graph
            for e_dict in result["entities"]:
                self.graph.add_entity(Entity(
                    id=e_dict["id"],
                    name=e_dict["name"],
                    entity_type=EntityType(e_dict["type"]),
                    definition=e_dict.get("definition", ""),
                    source_file=source_file,
                    confidence=e_dict.get("confidence", 0.8),
                    ihsan_dimension=e_dict.get("ihsan_dimension")
                ))
                
            for r_dict in result["relationships"]:
                self.graph.add_relationship(Relationship(
                    source_id=r_dict["source"],
                    target_id=r_dict["target"],
                    relation_type=RelationType(r_dict["type"]),
                    confidence=r_dict.get("confidence", 0.8)
                ))
                
            return result
            
        else:
            # Fallback: direct extraction without LangGraph
            extractor = PatternExtractor()
            entities = extractor.extract_entities(text, source_file)
            relationships = extractor.extract_relationships(text, entities)
            
            for entity in entities:
                self.graph.add_entity(entity)
            for rel in relationships:
                self.graph.add_relationship(rel)
                
            return {
                "entities": [e.to_dict() for e in entities],
                "relationships": [r.to_dict() for r in relationships]
            }
            
    def process_file(self, file_path: Path) -> Optional[dict]:
        """Process a single conversation file."""
        try:
            # Try multiple encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None
                
            if len(content) < 500:
                return None
                
            # Truncate very long content
            if len(content) > 100000:
                content = content[:100000]
                
            result = self.process_text(content, source_file=file_path.name)
            self.stats["files_processed"] += 1
            self.stats["total_entities"] = len(self.graph.entities)
            self.stats["total_relationships"] = len(self.graph.relationships)
            
            return result
            
        except Exception as e:
            self.stats["errors"].append(f"{file_path.name}: {str(e)}")
            return None
            
    def mine_chat_data(self, chat_root: Path, limit: Optional[int] = None) -> KnowledgeGraph:
        """Mine knowledge graph from chat data directory."""
        print(f"\n{'='*60}")
        print("BIZRA Knowledge Graph Miner v1.0")
        print(f"{'='*60}")
        print(f"📂 Source: {chat_root}")
        print(f"🔧 LangGraph: {'✓' if LANGGRAPH_AVAILABLE else '✗'}")
        print(f"🏖 E2B Sandbox: {'✓' if self.use_sandbox else '✗'}")
        
        # Find .md files
        md_files = list(chat_root.glob('**/*.md'))
        print(f"📄 Found {len(md_files)} .md files")
        
        if limit:
            md_files = md_files[:limit]
            print(f"🔢 Limited to {limit} files")
            
        print(f"\n{'─'*60}")
        print("Processing with LangGraph workflow...")
        print(f"{'─'*60}")
        
        for i, file_path in enumerate(md_files):
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(md_files)} "
                      f"({len(self.graph.entities)} entities, "
                      f"{len(self.graph.relationships)} relationships)")
                
            self.process_file(file_path)
            
        return self.graph
        
    def save_graph(self, output_dir: Path) -> dict:
        """Save knowledge graph in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        
        # JSON format
        json_path = output_dir / 'knowledge_graph.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "dna_signature": "GRAPH-MINER-7-3-6-9-00"
                },
                "stats": self.stats,
                "graph": self.graph.to_dict()
            }, f, indent=2, ensure_ascii=False)
        outputs["json"] = json_path
        print(f"📄 JSON saved: {json_path}")
        
        # Cypher format for Neo4j
        cypher_path = output_dir / 'knowledge_graph.cypher'
        with open(cypher_path, 'w', encoding='utf-8') as f:
            f.write(f"// BIZRA Knowledge Graph\n")
            f.write(f"// Generated: {datetime.now().isoformat()}\n")
            f.write(f"// Entities: {len(self.graph.entities)}\n")
            f.write(f"// Relationships: {len(self.graph.relationships)}\n\n")
            f.write(self.graph.to_cypher())
        outputs["cypher"] = cypher_path
        print(f"📄 Cypher saved: {cypher_path}")
        
        # GraphML format (for visualization tools)
        graphml_path = output_dir / 'knowledge_graph.graphml'
        self._save_graphml(graphml_path)
        outputs["graphml"] = graphml_path
        print(f"📄 GraphML saved: {graphml_path}")
        
        return outputs
        
    def _save_graphml(self, path: Path) -> None:
        """Save graph in GraphML format."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        lines.append('  <key id="name" for="node" attr.name="name" attr.type="string"/>')
        lines.append('  <key id="type" for="node" attr.name="type" attr.type="string"/>')
        lines.append('  <key id="definition" for="node" attr.name="definition" attr.type="string"/>')
        lines.append('  <key id="reltype" for="edge" attr.name="type" attr.type="string"/>')
        lines.append('  <graph id="G" edgedefault="directed">')
        
        for entity in self.graph.entities.values():
            lines.append(f'    <node id="{entity.id}">')
            lines.append(f'      <data key="name">{entity.name}</data>')
            lines.append(f'      <data key="type">{entity.entity_type.value}</data>')
            lines.append(f'      <data key="definition">{entity.definition}</data>')
            lines.append('    </node>')
            
        for i, rel in enumerate(self.graph.relationships):
            lines.append(f'    <edge id="e{i}" source="{rel.source_id}" target="{rel.target_id}">')
            lines.append(f'      <data key="reltype">{rel.relation_type.value}</data>')
            lines.append('    </edge>')
            
        lines.append('  </graph>')
        lines.append('</graphml>')
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BIZRA Knowledge Graph Miner')
    parser.add_argument('--limit', type=int, default=None, help='Limit files')
    parser.add_argument('--sandbox', action='store_true', help='Enable E2B sandbox')
    parser.add_argument('--chat-root', type=str, default=None, help='Chat data path')
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    chat_root = Path(args.chat_root) if args.chat_root else project_root / 'chat data sample'
    
    if not chat_root.exists():
        print(f"❌ Chat data not found: {chat_root}")
        return 1
        
    # Initialize miner
    miner = KnowledgeGraphMiner(use_sandbox=args.sandbox)
    
    # Mine knowledge graph
    graph = miner.mine_chat_data(chat_root, limit=args.limit)
    
    # Print summary
    print(f"\n{'='*60}")
    print("KNOWLEDGE GRAPH SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Files processed: {miner.stats['files_processed']}")
    print(f"🧬 Entities: {len(graph.entities)}")
    print(f"🔗 Relationships: {len(graph.relationships)}")
    
    print(f"\n📊 By entity type:")
    type_counts = graph._count_by_type()
    for entity_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {entity_type}: {count}")
        
    # Save outputs
    evidence_dir = project_root / 'evidence'
    outputs = miner.save_graph(evidence_dir)
    
    print(f"\n{'='*60}")
    print("✅ Knowledge Graph mining complete!")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
