# BIZRA 47-Discipline Synthesis Engine v1.0
# Implements the 4-Generator Theory for Cognitive Topology Mapping
# 
# Core Insight: 4 Generator Disciplines × 7 Layers = 47 Emergent Disciplines
# Generators: Graph Theory, Information Theory, Ethics (Ihsān), Pedagogy
#
# Standing on Giants: NetworkX, NumPy, Information Theory (Shannon)

import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import time

from bizra_config import (
    INDEXED_PATH, GOLD_PATH, GRAPH_PATH,
    SNR_THRESHOLD, IHSAN_CONSTRAINT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | DISCIPLINE | %(message)s'
)
logger = logging.getLogger("DisciplineSynthesis")


# ============================================================================
# LAYER DEFINITIONS
# ============================================================================

class Layer(Enum):
    """The 7 cognitive layers of the Sovereign Organism."""
    L1_FOUNDATION = "L1_FOUNDATION"      # Axiomatic Roots
    L2_PHYSICALITY = "L2_PHYSICALITY"    # Material Constraints
    L3_SOCIETAL = "L3_SOCIETAL"          # Human Interface
    L4_CREATIVE = "L4_CREATIVE"          # Generative Spark
    L5_TRANSCENDENT = "L5_TRANSCENDENT"  # Covenant Layer
    L6_APPLIED = "L6_APPLIED"            # Engineering Core
    L7_SYNTHESIS = "L7_SYNTHESIS"        # BIZRA Meta-Layer


class Generator(Enum):
    """The 4 core disciplines that generate all others."""
    GRAPH_THEORY = "graph_theory"
    INFORMATION_THEORY = "information_theory"
    ETHICS = "ethics"
    PEDAGOGY = "pedagogy"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Discipline:
    """A single discipline in the 47-discipline taxonomy."""
    id: int
    name: str
    role: str
    layer: Layer
    generators: List[Generator]     # Which generators produce this
    coverage: float = 0.0           # 0.0 to 1.0
    evidence_files: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    synergies: List[int] = field(default_factory=list)  # IDs of synergistic disciplines


@dataclass
class SynergyLink:
    """A detected synergy between disciplines."""
    source_id: int
    target_id: int
    strength: float                 # 0.0 to 1.0
    generator: Generator            # Which generator enables this synergy
    evidence: List[str]
    description: str


@dataclass
class DisciplineReport:
    """Complete analysis report for discipline coverage."""
    timestamp: str
    total_disciplines: int
    covered_count: int              # Disciplines > 60% coverage
    gap_count: int                  # Disciplines < 40% coverage
    overall_coverage: float
    layer_coverage: Dict[str, float]
    generator_strength: Dict[str, float]
    critical_gaps: List[Discipline]
    strong_disciplines: List[Discipline]
    synergy_links: List[SynergyLink]
    recommendations: List[str]


# ============================================================================
# 47-DISCIPLINE TAXONOMY
# ============================================================================

def build_discipline_taxonomy() -> Dict[int, Discipline]:
    """
    Build the complete 47-discipline taxonomy.
    
    This is the cognitive topology of the Sovereign Organism.
    Each discipline is mapped to its generators and layer.
    """
    taxonomy = {}
    
    # L1: THE AXIOMATIC ROOTS (Foundation Layer)
    l1_disciplines = [
        (1, "Formal Logic", "Truth Verification (FATE)", 
         [Generator.GRAPH_THEORY, Generator.INFORMATION_THEORY],
         ["logic", "proof", "verification", "truth", "fate", "boolean"]),
        (2, "Number Theory", "Cryptographic Primitives",
         [Generator.INFORMATION_THEORY],
         ["prime", "cryptographic", "sha256", "ed25519", "hash", "number"]),
        (3, "Set Theory", "Ontological Categorization",
         [Generator.GRAPH_THEORY],
         ["set", "category", "ontology", "class", "type", "taxonomy"]),
        (4, "Graph Theory", "Network Topology Analysis",
         [Generator.GRAPH_THEORY],
         ["graph", "node", "edge", "network", "topology", "hypergraph"]),
        (5, "Information Theory", "SNR Optimization",
         [Generator.INFORMATION_THEORY],
         ["snr", "entropy", "shannon", "information", "signal", "noise"]),
        (6, "Cybernetics", "Feedback Loops & Autopoiesis",
         [Generator.GRAPH_THEORY, Generator.PEDAGOGY],
         ["cybernetics", "feedback", "autopoiesis", "control", "loop"]),
        (7, "Systems Theory", "Holarchic Integration",
         [Generator.GRAPH_THEORY, Generator.PEDAGOGY],
         ["system", "holarchy", "integration", "holon", "hierarchy"]),
    ]
    
    for id_, name, role, gens, keywords in l1_disciplines:
        taxonomy[id_] = Discipline(
            id=id_, name=name, role=role, layer=Layer.L1_FOUNDATION,
            generators=gens, keywords=keywords
        )
    
    # L2: THE MATERIAL CONSTRAINTS (Physicality Layer)
    l2_disciplines = [
        (8, "Thermodynamics", "Entropy Management",
         [Generator.INFORMATION_THEORY],
         ["thermodynamics", "entropy", "energy", "heat", "equilibrium"]),
        (9, "Quantum Mechanics", "Probabilistic Compute",
         [Generator.INFORMATION_THEORY],
         ["quantum", "probabilistic", "superposition", "entanglement"]),
        (10, "Neuroscience", "Cognitive Modeling",
         [Generator.GRAPH_THEORY, Generator.PEDAGOGY],
         ["neuroscience", "cognitive", "brain", "neural", "consciousness"]),
        (11, "Evolutionary Biology", "Genetic Algorithms",
         [Generator.INFORMATION_THEORY, Generator.PEDAGOGY],
         ["evolution", "genetic", "mutation", "selection", "fitness"]),
        (12, "Ecological Science", "Resource Metabolism",
         [Generator.GRAPH_THEORY, Generator.ETHICS],
         ["ecology", "ecosystem", "metabolism", "sustainability", "resource"]),
        (13, "Materials Science", "Hardware Substrate Optimization",
         [Generator.INFORMATION_THEORY],
         ["materials", "hardware", "substrate", "gpu", "rtx", "optimization"]),
    ]
    
    for id_, name, role, gens, keywords in l2_disciplines:
        taxonomy[id_] = Discipline(
            id=id_, name=name, role=role, layer=Layer.L2_PHYSICALITY,
            generators=gens, keywords=keywords
        )
    
    # L3: THE HUMAN INTERFACE (Societal Layer)
    l3_disciplines = [
        (14, "Economics", "Harberger Tax & Tokenomics",
         [Generator.GRAPH_THEORY, Generator.ETHICS],
         ["economics", "token", "harberger", "seed", "bloom", "poi"]),
        (15, "Game Theory", "Incentive Alignment",
         [Generator.GRAPH_THEORY, Generator.INFORMATION_THEORY],
         ["game theory", "incentive", "nash", "equilibrium", "reward"]),
        (16, "Sociology", "Group Dynamics",
         [Generator.GRAPH_THEORY, Generator.ETHICS],
         ["sociology", "group", "social", "community", "dynamics"]),
        (17, "Anthropology", "Cultural Context",
         [Generator.PEDAGOGY, Generator.ETHICS],
         ["anthropology", "culture", "arabic", "bilingual", "tradition"]),
        (18, "Political Science", "Decentralized Governance",
         [Generator.GRAPH_THEORY, Generator.ETHICS],
         ["governance", "decentralized", "voting", "bzg", "dao", "consensus"]),
        (19, "Law / Jurisprudence", "Smart Contract Logic",
         [Generator.GRAPH_THEORY, Generator.ETHICS],
         ["law", "contract", "smart contract", "jurisprudence", "legal"]),
        (20, "Linguistics", "Semantic Processing",
         [Generator.INFORMATION_THEORY, Generator.PEDAGOGY],
         ["linguistics", "semantic", "embedding", "language", "nlp"]),
        (21, "Psychology", "User Intent Analysis",
         [Generator.PEDAGOGY, Generator.ETHICS],
         ["psychology", "intent", "user", "behavior", "cognitive"]),
    ]
    
    for id_, name, role, gens, keywords in l3_disciplines:
        taxonomy[id_] = Discipline(
            id=id_, name=name, role=role, layer=Layer.L3_SOCIETAL,
            generators=gens, keywords=keywords
        )
    
    # L4: THE GENERATIVE SPARK (Creative Layer)
    l4_disciplines = [
        (22, "Architecture", "Spatial Reasoning",
         [Generator.GRAPH_THEORY],
         ["architecture", "spatial", "structure", "design", "layout"]),
        (23, "Design Thinking", "Problem Solving Heuristics",
         [Generator.PEDAGOGY],
         ["design thinking", "heuristic", "prototype", "iterate", "empathy"]),
        (24, "Music Theory", "Harmonic Resonance Patterns",
         [Generator.INFORMATION_THEORY],
         ["music", "harmonic", "rhythm", "melody", "frequency"]),
        (25, "Narratology", "Story/Myth Generation",
         [Generator.PEDAGOGY, Generator.ETHICS],
         ["narrative", "story", "myth", "plot", "character", "arc"]),
        (26, "Visual Arts", "Pattern Recognition",
         [Generator.INFORMATION_THEORY, Generator.PEDAGOGY],
         ["visual", "art", "image", "pattern", "recognition", "clip"]),
        (27, "Poetics", "High-Density Meaning Compression",
         [Generator.INFORMATION_THEORY, Generator.ETHICS],
         ["poetics", "poetry", "quran", "verse", "metaphor", "meaning"]),
    ]
    
    for id_, name, role, gens, keywords in l4_disciplines:
        taxonomy[id_] = Discipline(
            id=id_, name=name, role=role, layer=Layer.L4_CREATIVE,
            generators=gens, keywords=keywords
        )
    
    # L5: THE COVENANT LAYER (Transcendent)
    l5_disciplines = [
        (28, "Ethics", "Moral Constraint (Adl)",
         [Generator.ETHICS],
         ["ethics", "moral", "ihsan", "adl", "justice", "constraint"]),
        (29, "Theology", "Ultimate Purpose / Teleology",
         [Generator.ETHICS, Generator.PEDAGOGY],
         ["theology", "purpose", "teleology", "wisdom", "divine", "quran"]),
        (30, "Epistemology", "Knowledge Validation",
         [Generator.INFORMATION_THEORY, Generator.ETHICS],
         ["epistemology", "knowledge", "validation", "truth", "belief"]),
        (31, "Metaphysics", "Ontological Definition",
         [Generator.GRAPH_THEORY, Generator.ETHICS],
         ["metaphysics", "ontology", "being", "existence", "reality"]),
        (32, "Aesthetics", "Beauty/Elegance Optimization",
         [Generator.INFORMATION_THEORY, Generator.ETHICS],
         ["aesthetics", "beauty", "elegance", "harmony", "proportion"]),
        (33, "Phenomenology", "Conscious Experience Mapping",
         [Generator.PEDAGOGY, Generator.ETHICS],
         ["phenomenology", "consciousness", "experience", "qualia", "ddagi"]),
    ]
    
    for id_, name, role, gens, keywords in l5_disciplines:
        taxonomy[id_] = Discipline(
            id=id_, name=name, role=role, layer=Layer.L5_TRANSCENDENT,
            generators=gens, keywords=keywords
        )
    
    # L6: THE ENGINEERING CORE (Applied Layer)
    l6_disciplines = [
        (34, "Computer Science", "Algorithmic Execution",
         [Generator.GRAPH_THEORY, Generator.INFORMATION_THEORY],
         ["computer science", "algorithm", "code", "programming", "software"]),
        (35, "Cryptography", "Zero-Knowledge Proofs",
         [Generator.INFORMATION_THEORY, Generator.ETHICS],
         ["cryptography", "zkp", "zero knowledge", "encryption", "merkle"]),
        (36, "Data Science", "Pattern Mining",
         [Generator.INFORMATION_THEORY, Generator.PEDAGOGY],
         ["data science", "pattern", "mining", "analysis", "statistics"]),
        (37, "Network Engineering", "Distributed Systems",
         [Generator.GRAPH_THEORY, Generator.INFORMATION_THEORY],
         ["network", "distributed", "p2p", "consensus", "node"]),
        (38, "Robotics", "Physical Actuation",
         [Generator.GRAPH_THEORY],
         ["robotics", "actuator", "sensor", "motor", "physical"]),
        (39, "Energy Engineering", "Sustainability",
         [Generator.ETHICS],
         ["energy", "sustainability", "power", "renewable", "efficiency"]),
    ]
    
    for id_, name, role, gens, keywords in l6_disciplines:
        taxonomy[id_] = Discipline(
            id=id_, name=name, role=role, layer=Layer.L6_APPLIED,
            generators=gens, keywords=keywords
        )
    
    # L7: THE BIZRA META-LAYER (Synthesis)
    l7_disciplines = [
        (40, "Chaos Theory", "Non-Linear Dynamics",
         [Generator.GRAPH_THEORY, Generator.INFORMATION_THEORY],
         ["chaos", "nonlinear", "attractor", "bifurcation", "dynamics"]),
        (41, "Complexity Science", "Emergence Management",
         [Generator.GRAPH_THEORY, Generator.PEDAGOGY],
         ["complexity", "emergence", "self-organization", "adaptive"]),
        (42, "Semiotics", "Sign/Symbol Interpretation",
         [Generator.INFORMATION_THEORY, Generator.PEDAGOGY],
         ["semiotics", "sign", "symbol", "interpretation", "meaning"]),
        (43, "Decision Theory", "Action Selection",
         [Generator.INFORMATION_THEORY, Generator.ETHICS],
         ["decision", "action", "choice", "utility", "optimization"]),
        (44, "Pedagogy", "Learning Optimization",
         [Generator.PEDAGOGY],
         ["pedagogy", "learning", "teaching", "education", "sape"]),
        (45, "History", "Long-Term Temporal Context",
         [Generator.PEDAGOGY, Generator.ETHICS],
         ["history", "temporal", "archive", "chronicle", "legacy"]),
        (46, "Futures Studies", "Scenario Planning",
         [Generator.INFORMATION_THEORY, Generator.PEDAGOGY],
         ["futures", "scenario", "forecast", "prediction", "planning"]),
        (47, "Ihsān Studies", "Excellence Maximization",
         [Generator.ETHICS, Generator.PEDAGOGY],
         ["ihsan", "excellence", "perfection", "mastery", "quality"]),
    ]
    
    for id_, name, role, gens, keywords in l7_disciplines:
        taxonomy[id_] = Discipline(
            id=id_, name=name, role=role, layer=Layer.L7_SYNTHESIS,
            generators=gens, keywords=keywords
        )
    
    return taxonomy


# ============================================================================
# GENERATOR CLASSES
# ============================================================================

class GeneratorBase:
    """Base class for discipline generators."""
    
    def __init__(self, taxonomy: Dict[int, Discipline]):
        self.taxonomy = taxonomy
        self.epsilon = 1e-10
    
    def get_disciplines(self) -> List[Discipline]:
        """Get all disciplines this generator produces."""
        raise NotImplementedError
    
    def calculate_strength(self, corpus_stats: Dict) -> float:
        """Calculate the strength of this generator in the corpus."""
        raise NotImplementedError
    
    def amplify(self, discipline_id: int, factor: float) -> float:
        """Amplify a discipline's coverage via this generator."""
        raise NotImplementedError


class GraphTheoryGenerator(GeneratorBase):
    """
    Graph Theory Generator
    
    Generates: Network topologies, relationship structures, traversal paths
    Key metric: Graph density, connectivity, centrality
    """
    
    def __init__(self, taxonomy: Dict[int, Discipline]):
        super().__init__(taxonomy)
        self.generator = Generator.GRAPH_THEORY
    
    def get_disciplines(self) -> List[Discipline]:
        return [d for d in self.taxonomy.values() 
                if self.generator in d.generators]
    
    def calculate_strength(self, corpus_stats: Dict) -> float:
        """
        Calculate graph theory strength from corpus statistics.
        
        Metrics:
        - Node count (more nodes = more topology)
        - Edge density (edges / possible edges)
        - Connectivity (connected components)
        """
        nodes = corpus_stats.get("total_nodes", 0)
        edges = corpus_stats.get("total_edges", 0)
        
        if nodes < 2:
            return 0.0
        
        # Edge density: actual edges / max possible edges
        max_edges = nodes * (nodes - 1) / 2
        edge_density = min(edges / max_edges, 1.0) if max_edges > 0 else 0.0
        
        # Node score: logarithmic scaling (10K nodes = 1.0)
        node_score = min(np.log10(nodes + 1) / 4, 1.0)
        
        # Combined strength
        strength = (node_score * 0.6) + (edge_density * 0.4)
        return min(strength, 1.0)
    
    def amplify(self, discipline_id: int, factor: float) -> float:
        """Graph-based amplification via connectivity."""
        discipline = self.taxonomy.get(discipline_id)
        if not discipline or self.generator not in discipline.generators:
            return 0.0
        
        # Amplification is proportional to graph connectivity
        base_coverage = discipline.coverage
        amplified = base_coverage + (1 - base_coverage) * factor * 0.3
        return min(amplified, 1.0)


class InformationTheoryGenerator(GeneratorBase):
    """
    Information Theory Generator
    
    Generates: SNR optimization, entropy reduction, semantic density
    Key metric: Shannon entropy, mutual information, compression ratio
    """
    
    def __init__(self, taxonomy: Dict[int, Discipline]):
        super().__init__(taxonomy)
        self.generator = Generator.INFORMATION_THEORY
    
    def get_disciplines(self) -> List[Discipline]:
        return [d for d in self.taxonomy.values() 
                if self.generator in d.generators]
    
    def calculate_strength(self, corpus_stats: Dict) -> float:
        """
        Calculate information theory strength.
        
        Metrics:
        - SNR scores in the system
        - Embedding quality (dimension, coverage)
        - Compression effectiveness
        """
        snr_avg = corpus_stats.get("average_snr", 0.0)
        embedding_dim = corpus_stats.get("embedding_dim", 0)
        embedding_coverage = corpus_stats.get("embedding_coverage", 0.0)
        
        # SNR score (target: 0.99)
        snr_score = min(snr_avg / 0.99, 1.0) if snr_avg > 0 else 0.0
        
        # Embedding score (384-dim = optimal)
        if embedding_dim >= 384:
            embedding_score = 1.0
        elif embedding_dim >= 128:
            embedding_score = 0.7
        else:
            embedding_score = embedding_dim / 384
        
        # Combined strength
        strength = (snr_score * 0.5) + (embedding_score * 0.3) + (embedding_coverage * 0.2)
        return min(strength, 1.0)
    
    def amplify(self, discipline_id: int, factor: float) -> float:
        """Information-theoretic amplification via SNR improvement."""
        discipline = self.taxonomy.get(discipline_id)
        if not discipline or self.generator not in discipline.generators:
            return 0.0
        
        base_coverage = discipline.coverage
        # SNR-based amplification: signal enhancement
        amplified = base_coverage + (1 - base_coverage) * factor * 0.35
        return min(amplified, 1.0)


class EthicsGenerator(GeneratorBase):
    """
    Ethics Generator (Ihsān)
    
    Generates: Moral constraints, excellence thresholds, value alignment
    Key metric: Ihsān score, constitutional compliance, covenant adherence
    """
    
    def __init__(self, taxonomy: Dict[int, Discipline]):
        super().__init__(taxonomy)
        self.generator = Generator.ETHICS
    
    def get_disciplines(self) -> List[Discipline]:
        return [d for d in self.taxonomy.values() 
                if self.generator in d.generators]
    
    def calculate_strength(self, corpus_stats: Dict) -> float:
        """
        Calculate ethics generator strength.
        
        Metrics:
        - Ihsān threshold compliance
        - Constitutional AI checks
        - Covenant verification coverage
        """
        ihsan_compliance = corpus_stats.get("ihsan_compliance", 0.0)
        constitutional_checks = corpus_stats.get("constitutional_checks", 0)
        covenant_coverage = corpus_stats.get("covenant_coverage", 0.0)
        
        # Ihsān score (target: 0.99)
        ihsan_score = min(ihsan_compliance / IHSAN_CONSTRAINT, 1.0)
        
        # Constitutional AI presence
        const_score = 1.0 if constitutional_checks > 0 else 0.0
        
        # Combined strength
        strength = (ihsan_score * 0.5) + (const_score * 0.2) + (covenant_coverage * 0.3)
        return min(strength, 1.0)
    
    def amplify(self, discipline_id: int, factor: float) -> float:
        """Ethics-based amplification via value alignment."""
        discipline = self.taxonomy.get(discipline_id)
        if not discipline or self.generator not in discipline.generators:
            return 0.0
        
        base_coverage = discipline.coverage
        # Ethics amplification: constraint propagation
        amplified = base_coverage + (1 - base_coverage) * factor * 0.4
        return min(amplified, 1.0)


class PedagogyGenerator(GeneratorBase):
    """
    Pedagogy Generator
    
    Generates: Learning optimization, knowledge transfer, adaptive teaching
    Key metric: Learning rate, knowledge retention, SAPE effectiveness
    """
    
    def __init__(self, taxonomy: Dict[int, Discipline]):
        super().__init__(taxonomy)
        self.generator = Generator.PEDAGOGY
    
    def get_disciplines(self) -> List[Discipline]:
        return [d for d in self.taxonomy.values() 
                if self.generator in d.generators]
    
    def calculate_strength(self, corpus_stats: Dict) -> float:
        """
        Calculate pedagogy generator strength.
        
        Metrics:
        - SAPE layer presence and effectiveness
        - Learning loop completeness
        - Knowledge graph growth rate
        """
        sape_active = corpus_stats.get("sape_active", False)
        learning_loops = corpus_stats.get("learning_loops", 0)
        knowledge_growth = corpus_stats.get("knowledge_growth", 0.0)
        
        # SAPE presence
        sape_score = 1.0 if sape_active else 0.0
        
        # Learning loops (target: 3+)
        loop_score = min(learning_loops / 3, 1.0)
        
        # Knowledge growth (positive = learning)
        growth_score = min(max(knowledge_growth, 0.0), 1.0)
        
        # Combined strength
        strength = (sape_score * 0.4) + (loop_score * 0.3) + (growth_score * 0.3)
        return min(strength, 1.0)
    
    def amplify(self, discipline_id: int, factor: float) -> float:
        """Pedagogy-based amplification via learning transfer."""
        discipline = self.taxonomy.get(discipline_id)
        if not discipline or self.generator not in discipline.generators:
            return 0.0
        
        base_coverage = discipline.coverage
        # Pedagogy amplification: knowledge transfer
        amplified = base_coverage + (1 - base_coverage) * factor * 0.35
        return min(amplified, 1.0)


# ============================================================================
# DISCIPLINE SYNTHESIS ENGINE
# ============================================================================

class DisciplineSynthesisEngine:
    """
    Main engine for 47-discipline cognitive topology mapping.
    
    Implements the 4-Generator Theory:
    - Graph Theory: Network structures, topologies
    - Information Theory: SNR, entropy, semantics
    - Ethics (Ihsān): Moral constraints, excellence
    - Pedagogy: Learning, knowledge transfer
    
    Formula: Strengthening ANY generator cascades to ALL 47 disciplines.
    """
    
    def __init__(self):
        self.taxonomy = build_discipline_taxonomy()
        self.generators = {
            Generator.GRAPH_THEORY: GraphTheoryGenerator(self.taxonomy),
            Generator.INFORMATION_THEORY: InformationTheoryGenerator(self.taxonomy),
            Generator.ETHICS: EthicsGenerator(self.taxonomy),
            Generator.PEDAGOGY: PedagogyGenerator(self.taxonomy),
        }
        self.synergy_links: List[SynergyLink] = []
        logger.info(f"DisciplineSynthesisEngine initialized with {len(self.taxonomy)} disciplines")
    
    def load_corpus_statistics(self) -> Dict:
        """Load statistics from the data lake."""
        stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "average_snr": 0.95,  # Default from system
            "embedding_dim": 384,
            "embedding_coverage": 0.0,
            "ihsan_compliance": 0.99,
            "constitutional_checks": 1,
            "covenant_coverage": 0.9,
            "sape_active": True,
            "learning_loops": 3,
            "knowledge_growth": 0.5,
        }
        
        # Try to load graph statistics
        graph_stats_path = GRAPH_PATH / "statistics.json"
        if graph_stats_path.exists():
            try:
                with open(graph_stats_path, 'r') as f:
                    graph_stats = json.load(f)
                    stats["total_nodes"] = graph_stats.get("total_nodes", 0)
                    stats["total_edges"] = graph_stats.get("total_edges", 0)
            except Exception as e:
                logger.warning(f"Could not load graph stats: {e}")
        
        # Try to load embedding checkpoint
        embedding_checkpoint = INDEXED_PATH / "embeddings" / "checkpoint.json"
        if embedding_checkpoint.exists():
            try:
                with open(embedding_checkpoint, 'r') as f:
                    emb_stats = json.load(f)
                    total_docs = emb_stats.get("total_documents", 1)
                    processed = emb_stats.get("processed_count", 0)
                    stats["embedding_coverage"] = processed / total_docs if total_docs > 0 else 0.0
            except Exception as e:
                logger.warning(f"Could not load embedding stats: {e}")
        
        return stats
    
    def scan_corpus_for_disciplines(self) -> Dict[int, float]:
        """
        Scan the entire corpus to calculate actual discipline coverage.
        
        Scans:
        - Python source files (*.py)
        - Markdown files (*.md)
        - GOLD assertions and ledgers
        - Indexed knowledge files
        """
        logger.info("Scanning corpus for discipline coverage...")
        
        # Aggregate content from key sources
        corpus_text = []
        
        # Scan Python source files
        source_root = Path(__file__).parent
        for py_file in source_root.glob("*.py"):
            try:
                corpus_text.append(py_file.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                pass
        
        # Scan Markdown documentation
        for md_file in source_root.glob("*.md"):
            try:
                corpus_text.append(md_file.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                pass
        
        # Scan GOLD layer assertions
        if GOLD_PATH.exists():
            for gold_file in GOLD_PATH.glob("*.jsonl"):
                try:
                    corpus_text.append(gold_file.read_text(encoding='utf-8', errors='ignore'))
                except Exception:
                    pass
            for gold_file in GOLD_PATH.glob("*.json"):
                try:
                    corpus_text.append(gold_file.read_text(encoding='utf-8', errors='ignore'))
                except Exception:
                    pass
            for gold_file in GOLD_PATH.glob("*.md"):
                try:
                    corpus_text.append(gold_file.read_text(encoding='utf-8', errors='ignore'))
                except Exception:
                    pass
        
        # Scan indexed knowledge
        if INDEXED_PATH.exists():
            for idx_file in list(INDEXED_PATH.glob("*.jsonl"))[:50]:  # Limit
                try:
                    corpus_text.append(idx_file.read_text(encoding='utf-8', errors='ignore'))
                except Exception:
                    pass
        
        # Combine and analyze
        combined_text = "\n".join(corpus_text)
        logger.info(f"Scanned {len(corpus_text)} files, {len(combined_text):,} characters")
        
        return self.analyze_coverage(combined_text)
    
    def analyze_coverage(self, file_content: str) -> Dict[int, float]:
        """
        Analyze discipline coverage from file content.
        
        Returns: Dict of discipline_id -> coverage_score
        """
        content_lower = file_content.lower()
        coverage = {}
        
        for disc_id, disc in self.taxonomy.items():
            # Count keyword matches
            matches = sum(1 for kw in disc.keywords if kw.lower() in content_lower)
            # Normalize by keyword count
            score = min(matches / len(disc.keywords), 1.0) if disc.keywords else 0.0
            coverage[disc_id] = score
        
        return coverage
    
    def calculate_generator_strengths(self, corpus_stats: Dict) -> Dict[Generator, float]:
        """Calculate strength of each generator."""
        strengths = {}
        for gen_type, generator in self.generators.items():
            strengths[gen_type] = generator.calculate_strength(corpus_stats)
        return strengths
    
    def cascade_amplification(
        self,
        generator_strengths: Dict[Generator, float]
    ) -> Dict[int, float]:
        """
        Apply cascade amplification from generators to all disciplines.
        
        The core insight: Strong generators amplify all disciplines they produce.
        """
        amplified_coverage = {}
        
        for disc_id, disc in self.taxonomy.items():
            # Start with base coverage
            total_amplification = disc.coverage
            
            # Apply amplification from each relevant generator
            for gen_type in disc.generators:
                if gen_type in generator_strengths:
                    gen_strength = generator_strengths[gen_type]
                    generator = self.generators[gen_type]
                    amplified = generator.amplify(disc_id, gen_strength)
                    total_amplification = max(total_amplification, amplified)
            
            amplified_coverage[disc_id] = total_amplification
        
        return amplified_coverage
    
    def detect_synergies(self) -> List[SynergyLink]:
        """
        Detect synergies between disciplines based on shared generators.
        
        Synergy strength = overlap in generators × coverage product
        """
        synergies = []
        disc_list = list(self.taxonomy.values())
        
        for i, disc1 in enumerate(disc_list):
            for disc2 in disc_list[i+1:]:
                # Find shared generators
                shared_gens = set(disc1.generators) & set(disc2.generators)
                
                if shared_gens:
                    # Calculate synergy strength
                    coverage_product = disc1.coverage * disc2.coverage
                    gen_overlap = len(shared_gens) / 4  # Normalize by total generators
                    strength = (coverage_product + gen_overlap) / 2
                    
                    if strength > 0.3:  # Threshold for significant synergy
                        # Use the first shared generator as the bridge
                        primary_gen = list(shared_gens)[0]
                        
                        synergy = SynergyLink(
                            source_id=disc1.id,
                            target_id=disc2.id,
                            strength=strength,
                            generator=primary_gen,
                            evidence=[disc1.name, disc2.name],
                            description=f"{disc1.name} ↔ {disc2.name} via {primary_gen.value}"
                        )
                        synergies.append(synergy)
        
        # Sort by strength
        synergies.sort(key=lambda s: s.strength, reverse=True)
        self.synergy_links = synergies
        return synergies
    
    def identify_gaps(self, threshold: float = 0.4) -> List[Discipline]:
        """Identify disciplines below coverage threshold."""
        return [d for d in self.taxonomy.values() if d.coverage < threshold]
    
    def identify_strengths(self, threshold: float = 0.9) -> List[Discipline]:
        """Identify disciplines above strength threshold."""
        return [d for d in self.taxonomy.values() if d.coverage >= threshold]
    
    def generate_recommendations(
        self,
        gaps: List[Discipline],
        generator_strengths: Dict[Generator, float]
    ) -> List[str]:
        """Generate prioritized recommendations for closing gaps."""
        recommendations = []
        
        # Sort gaps by their generator overlap with strong generators
        for gap in gaps:
            gap_gens = set(gap.generators)
            
            # Find which generators could help this gap
            for gen_type, strength in sorted(
                generator_strengths.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                if gen_type in gap_gens:
                    rec = (
                        f"[P1] Strengthen '{gap.name}' via {gen_type.value} "
                        f"(current: {gap.coverage:.0%}, generator strength: {strength:.0%})"
                    )
                    recommendations.append(rec)
                    break
        
        # General recommendations
        weak_gens = [g for g, s in generator_strengths.items() if s < 0.6]
        for gen in weak_gens:
            recommendations.append(
                f"[P0] Boost {gen.value} generator - it amplifies "
                f"{len(self.generators[gen].get_disciplines())} disciplines"
            )
        
        return recommendations[:10]  # Top 10
    
    def generate_report(self, scan_corpus: bool = True) -> DisciplineReport:
        """Generate a complete discipline coverage report."""
        # Load corpus stats
        corpus_stats = self.load_corpus_statistics()
        
        # Scan actual corpus for discipline coverage
        if scan_corpus:
            corpus_coverage = self.scan_corpus_for_disciplines()
            # Update base coverage from corpus scan
            for disc_id, coverage in corpus_coverage.items():
                if disc_id in self.taxonomy:
                    self.taxonomy[disc_id].coverage = coverage
        
        # Calculate generator strengths
        gen_strengths = self.calculate_generator_strengths(corpus_stats)
        
        # Apply cascade amplification
        amplified = self.cascade_amplification(gen_strengths)
        
        # Update taxonomy with amplified coverage
        for disc_id, coverage in amplified.items():
            self.taxonomy[disc_id].coverage = coverage
        
        # Calculate layer coverage
        layer_coverage = {}
        for layer in Layer:
            layer_discs = [d for d in self.taxonomy.values() if d.layer == layer]
            if layer_discs:
                layer_coverage[layer.value] = np.mean([d.coverage for d in layer_discs])
            else:
                layer_coverage[layer.value] = 0.0
        
        # Detect synergies
        synergies = self.detect_synergies()
        
        # Identify gaps and strengths
        gaps = self.identify_gaps(0.4)
        strengths = self.identify_strengths(0.9)
        
        # Overall metrics
        all_coverages = [d.coverage for d in self.taxonomy.values()]
        overall_coverage = np.mean(all_coverages)
        covered_count = len([c for c in all_coverages if c >= 0.6])
        
        # Generate recommendations
        recommendations = self.generate_recommendations(gaps, gen_strengths)
        
        report = DisciplineReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            total_disciplines=len(self.taxonomy),
            covered_count=covered_count,
            gap_count=len(gaps),
            overall_coverage=overall_coverage,
            layer_coverage=layer_coverage,
            generator_strength={g.value: s for g, s in gen_strengths.items()},
            critical_gaps=gaps,
            strong_disciplines=strengths,
            synergy_links=synergies[:20],  # Top 20
            recommendations=recommendations
        )
        
        return report
    
    def print_report(self, report: DisciplineReport):
        """Print a formatted report to console."""
        print("\n" + "=" * 80)
        print("🧬 BIZRA 47-DISCIPLINE SYNTHESIS REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Disciplines: {report.total_disciplines}")
        print(f"Covered (>60%): {report.covered_count}")
        print(f"Gaps (<40%): {report.gap_count}")
        print(f"Overall Coverage: {report.overall_coverage:.1%}")
        
        print("\n📊 GENERATOR STRENGTHS:")
        print("-" * 40)
        for gen, strength in sorted(report.generator_strength.items(), 
                                     key=lambda x: x[1], reverse=True):
            bar = "█" * int(strength * 20)
            print(f"  {gen:<20}: {bar} {strength:.1%}")
        
        print("\n📈 LAYER COVERAGE:")
        print("-" * 40)
        for layer, coverage in sorted(report.layer_coverage.items()):
            bar = "█" * int(coverage * 20)
            print(f"  {layer:<20}: {bar} {coverage:.1%}")
        
        print("\n🔴 CRITICAL GAPS:")
        print("-" * 40)
        for gap in report.critical_gaps[:5]:
            print(f"  [{gap.id:2}] {gap.name:<25} {gap.coverage:.1%} ({gap.layer.value})")
        
        print("\n🟢 STRONG DISCIPLINES:")
        print("-" * 40)
        for strong in report.strong_disciplines[:5]:
            print(f"  [{strong.id:2}] {strong.name:<25} {strong.coverage:.1%} ({strong.layer.value})")
        
        print("\n🔗 TOP SYNERGIES:")
        print("-" * 40)
        for syn in report.synergy_links[:5]:
            print(f"  {syn.description} (strength: {syn.strength:.2f})")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report.recommendations[:5], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
    
    def to_json(self, report: DisciplineReport) -> str:
        """Convert report to JSON for storage."""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (Discipline, SynergyLink)):
                return obj.__dict__
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        return json.dumps(report.__dict__, default=serialize, indent=2)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the discipline synthesis engine."""
    print("\n🧬 BIZRA 47-DISCIPLINE SYNTHESIS ENGINE v1.0")
    print("━" * 50)
    
    # Initialize engine
    engine = DisciplineSynthesisEngine()
    
    # Generate report
    print("\n📊 Analyzing corpus and calculating coverage...")
    report = engine.generate_report()
    
    # Print report
    engine.print_report(report)
    
    # Save report
    report_path = GOLD_PATH / "discipline_synthesis_report.json"
    with open(report_path, 'w') as f:
        f.write(engine.to_json(report))
    print(f"\n📁 Report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    main()
