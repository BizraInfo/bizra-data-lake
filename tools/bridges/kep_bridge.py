# BIZRA KEP Bridge v1.0
# Standing on Giants Protocol: Connects Data Lake Engines to Knowledge Explosion Point System
# Integration Layer: Hypergraph RAG → Synergy Detection → Compound Discovery → PAT Synthesis

"""
KEP Bridge Architecture:

┌─────────────────────────────────────────────────────────────────────────┐
│                        BIZRA DATA LAKE                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Hypergraph   │  │    ARTE      │  │     PAT      │                  │
│  │    RAG       │  │   Engine     │  │   Engine     │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
└─────────┼─────────────────┼─────────────────┼───────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          KEP BRIDGE                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  RetrievalToSynergy │ SNRToIhsan │ PATToCompound │ FeedbackLoop  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────┬─────────────────┬─────────────────┬───────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE EXPLOSION POINT (KEP)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Synergy    │  │    Safety    │  │   Compound   │  │  Learning  │  │
│  │  Detector    │  │    Stack     │  │   Discovery  │  │ Accelerator│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set
import numpy as np

# Import Data Lake engines
from bizra_config import (
    GRAPH_PATH, GOLD_PATH, INDEXED_PATH,
    SNR_THRESHOLD, IHSAN_CONSTRAINT
)

# Import Discipline Synthesis Engine (47-discipline cognitive topology)
try:
    from discipline_synthesis import (
        DisciplineSynthesisEngine,
        Generator,
        Layer,
        Discipline,
        SynergyLink as DisciplineSynergyLink
    )
    DISCIPLINE_ENGINE_AVAILABLE = True
except ImportError:
    DISCIPLINE_ENGINE_AVAILABLE = False
    DisciplineSynthesisEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | KEP-BRIDGE | %(message)s',
    handlers=[
        logging.FileHandler(INDEXED_PATH / "kep_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KEP-BRIDGE")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class SynergyType(Enum):
    """Types of cross-domain synergies detected."""
    CONCEPTUAL = "conceptual"       # Shared concepts across domains
    METHODOLOGICAL = "methodological"  # Transferable methods
    STRUCTURAL = "structural"       # Similar organizational patterns
    CAUSAL = "causal"               # Cause-effect relationships
    ANALOGICAL = "analogical"       # Deep structural analogies
    EMERGENT = "emergent"           # Novel combinations


class CompoundType(Enum):
    """Types of compound discoveries."""
    FUSION = "fusion"               # Direct combination
    SYNTHESIS = "synthesis"         # New entity from parts
    ABSTRACTION = "abstraction"     # Higher-level principle
    INSTANTIATION = "instantiation" # Concrete application
    TRANSFORMATION = "transformation"  # Domain transfer


@dataclass
class SynergyCandidate:
    """Potential synergy between knowledge domains."""
    source_domain: str
    target_domain: str
    synergy_type: SynergyType
    strength: float  # 0.0 to 1.0
    evidence: List[Dict]
    bridging_concepts: List[str]
    metadata: Dict = field(default_factory=dict)


@dataclass
class CompoundProposal:
    """Proposed compound discovery from synergy analysis."""
    synergies: List[SynergyCandidate]
    compound_type: CompoundType
    hypothesis: str
    confidence: float
    implications: List[str]
    validation_plan: Dict
    metadata: Dict = field(default_factory=dict)


@dataclass
class IhsanCheck:
    """Result of Ihsan constraint validation."""
    passed: bool
    score: float
    snr_component: float
    coherence_component: float
    ethics_component: float
    reasoning_trace: List[str]


@dataclass
class KEPResult:
    """Complete KEP processing result."""
    query: str
    synergies: List[SynergyCandidate]
    compounds: List[CompoundProposal]
    ihsan_check: IhsanCheck
    learning_boost: float
    feedback_applied: bool
    execution_time: float
    discipline_coverage: Optional[float] = None  # 47-discipline coverage
    generator_strengths: Optional[Dict[str, float]] = None  # 4-generator strengths
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# SYNERGY DETECTOR (Data Lake → KEP)
# ============================================================================

class SynergyDetector:
    """
    Detects cross-domain synergies from Hypergraph RAG retrieval results.

    Standing on Giants: Implements ideas from:
    - Hofstadter's analogical reasoning
    - Gentner's structure mapping theory
    - Boden's exploratory creativity
    
    Enhanced with 47-Discipline Cognitive Topology (v1.0):
    - 4-Generator Theory: Graph × InfoTheory × Ethics × Pedagogy
    - Layer-aware synergy detection
    - Cascade amplification from generators to all disciplines
    """

    def __init__(self):
        self.domain_embeddings: Dict[str, np.ndarray] = {}
        self.synergy_cache: Dict[str, List[SynergyCandidate]] = {}
        self._load_domain_index()
        
        # Initialize 47-Discipline Engine if available
        if DISCIPLINE_ENGINE_AVAILABLE:
            self.discipline_engine = DisciplineSynthesisEngine()
            logger.info("47-Discipline Synthesis Engine integrated")
        else:
            self.discipline_engine = None
            logger.warning("Discipline Synthesis Engine not available")

    def _load_domain_index(self):
        """Load domain classification index."""
        domain_path = INDEXED_PATH / "domain_index.json"
        if domain_path.exists():
            with open(domain_path, "r") as f:
                self.domain_index = json.load(f)
        else:
            # Default domain taxonomy
            self.domain_index = {
                "technical": ["software", "engineering", "architecture", "systems"],
                "scientific": ["physics", "mathematics", "biology", "chemistry"],
                "philosophical": ["ethics", "epistemology", "metaphysics", "logic"],
                "creative": ["art", "design", "music", "literature"],
                "business": ["strategy", "economics", "management", "finance"],
                "psychological": ["cognition", "behavior", "emotion", "development"]
            }

    def detect_synergies(
        self,
        retrieval_results: List[Dict],
        query_embedding: np.ndarray,
        min_strength: float = 0.6,
        use_47_disciplines: bool = True
    ) -> List[SynergyCandidate]:
        """
        Detect synergies from retrieval results.

        Algorithm:
        1. Classify results by domain
        2. Find cross-domain connections
        3. Evaluate synergy strength via embedding similarity
        4. Extract bridging concepts
        5. [NEW] Apply 47-discipline cognitive topology boost
        """
        if not retrieval_results:
            return []

        # Step 1: Domain classification
        domain_groups = self._classify_by_domain(retrieval_results)

        if len(domain_groups) < 2:
            # No cross-domain synergy possible
            return []

        # Step 2: Find cross-domain pairs
        synergies = []
        domains = list(domain_groups.keys())

        for i, source_domain in enumerate(domains):
            for target_domain in domains[i+1:]:
                # Step 3: Evaluate synergy
                synergy = self._evaluate_synergy(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_results=domain_groups[source_domain],
                    target_results=domain_groups[target_domain],
                    query_embedding=query_embedding
                )

                if synergy and synergy.strength >= min_strength:
                    synergies.append(synergy)

        # Step 4: Apply 47-discipline boost if available
        if use_47_disciplines and self.discipline_engine:
            synergies = self._apply_discipline_boost(synergies, retrieval_results)

        # Sort by strength
        synergies.sort(key=lambda s: s.strength, reverse=True)

        logger.info(f"Detected {len(synergies)} synergies across {len(domains)} domains")

        return synergies
    
    def _apply_discipline_boost(
        self,
        synergies: List[SynergyCandidate],
        retrieval_results: List[Dict]
    ) -> List[SynergyCandidate]:
        """
        Apply 47-discipline cognitive topology boost to synergies.
        
        4-Generator Theory amplification:
        - Graph Theory → structural synergies boost
        - Information Theory → conceptual synergies boost
        - Ethics (Ihsān) → causal synergies boost
        - Pedagogy → methodological synergies boost
        """
        if not self.discipline_engine:
            return synergies
        
        # Get generator strengths from corpus
        corpus_stats = self.discipline_engine.load_corpus_statistics()
        gen_strengths = self.discipline_engine.calculate_generator_strengths(corpus_stats)
        
        # Map synergy types to generators
        type_to_generator = {
            SynergyType.STRUCTURAL: Generator.GRAPH_THEORY,
            SynergyType.CONCEPTUAL: Generator.INFORMATION_THEORY,
            SynergyType.CAUSAL: Generator.ETHICS,
            SynergyType.METHODOLOGICAL: Generator.PEDAGOGY,
            SynergyType.ANALOGICAL: Generator.INFORMATION_THEORY,
            SynergyType.EMERGENT: Generator.PEDAGOGY,
        }
        
        # Boost synergies based on generator strength
        boosted_synergies = []
        for synergy in synergies:
            relevant_gen = type_to_generator.get(synergy.synergy_type)
            if relevant_gen and relevant_gen in gen_strengths:
                # Apply boost: original_strength + (1 - original) * generator_strength * 0.2
                boost_factor = gen_strengths[relevant_gen]
                boosted_strength = synergy.strength + (1 - synergy.strength) * boost_factor * 0.2
                synergy.strength = min(boosted_strength, 1.0)
                synergy.metadata["discipline_boosted"] = True
                synergy.metadata["boost_generator"] = relevant_gen.value
            
            boosted_synergies.append(synergy)
        
        return boosted_synergies
    
    def get_discipline_coverage(self) -> Optional[Dict]:
        """Get current 47-discipline coverage report."""
        if not self.discipline_engine:
            return None
        
        report = self.discipline_engine.generate_report()
        return {
            "overall_coverage": report.overall_coverage,
            "covered_count": report.covered_count,
            "gap_count": report.gap_count,
            "generator_strengths": report.generator_strength,
            "layer_coverage": report.layer_coverage,
            "recommendations": report.recommendations[:5]
        }

    def _classify_by_domain(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Classify retrieval results by knowledge domain."""
        domain_groups: Dict[str, List[Dict]] = {}

        for result in results:
            text = result.get("text", "").lower()
            doc_id = result.get("doc_id", "")

            # Simple keyword-based classification
            detected_domain = "general"
            max_score = 0

            for domain, keywords in self.domain_index.items():
                score = sum(1 for kw in keywords if kw in text or kw in doc_id.lower())
                if score > max_score:
                    max_score = score
                    detected_domain = domain

            if detected_domain not in domain_groups:
                domain_groups[detected_domain] = []
            domain_groups[detected_domain].append(result)

        return domain_groups

    def _evaluate_synergy(
        self,
        source_domain: str,
        target_domain: str,
        source_results: List[Dict],
        target_results: List[Dict],
        query_embedding: np.ndarray
    ) -> Optional[SynergyCandidate]:
        """Evaluate synergy between two domains."""

        # Extract embeddings if available
        source_embeddings = [
            np.array(r.get("embedding", []))
            for r in source_results
            if r.get("embedding") is not None
        ]
        target_embeddings = [
            np.array(r.get("embedding", []))
            for r in target_results
            if r.get("embedding") is not None
        ]

        # Calculate cross-domain similarity
        if source_embeddings and target_embeddings:
            # Average embeddings per domain
            source_centroid = np.mean(source_embeddings, axis=0)
            target_centroid = np.mean(target_embeddings, axis=0)

            # Cosine similarity
            similarity = float(np.dot(source_centroid, target_centroid) / (
                np.linalg.norm(source_centroid) * np.linalg.norm(target_centroid) + 1e-8
            ))
        else:
            # Fallback: text overlap heuristic
            source_text = " ".join(r.get("text", "")[:200] for r in source_results)
            target_text = " ".join(r.get("text", "")[:200] for r in target_results)

            source_words = set(source_text.lower().split())
            target_words = set(target_text.lower().split())

            overlap = len(source_words & target_words)
            union = len(source_words | target_words)
            similarity = overlap / (union + 1e-8)

        if similarity < 0.3:
            return None

        # Determine synergy type
        synergy_type = self._infer_synergy_type(
            source_results, target_results, similarity
        )

        # Extract bridging concepts
        bridging_concepts = self._extract_bridging_concepts(
            source_results, target_results
        )

        # Build evidence
        evidence = [
            {"source": r.get("doc_id"), "snippet": r.get("text", "")[:100]}
            for r in (source_results[:2] + target_results[:2])
        ]

        return SynergyCandidate(
            source_domain=source_domain,
            target_domain=target_domain,
            synergy_type=synergy_type,
            strength=min(similarity * 1.2, 1.0),  # Boost slightly
            evidence=evidence,
            bridging_concepts=bridging_concepts,
            metadata={
                "source_count": len(source_results),
                "target_count": len(target_results),
                "raw_similarity": similarity
            }
        )

    def _infer_synergy_type(
        self,
        source_results: List[Dict],
        target_results: List[Dict],
        similarity: float
    ) -> SynergyType:
        """Infer the type of synergy based on content analysis."""
        # Keywords indicating different synergy types
        method_keywords = {"method", "algorithm", "process", "procedure", "technique"}
        structure_keywords = {"structure", "pattern", "architecture", "framework", "hierarchy"}
        causal_keywords = {"cause", "effect", "result", "because", "therefore", "leads to"}

        all_text = " ".join(
            r.get("text", "").lower()
            for r in source_results + target_results
        )

        # Count keyword occurrences
        method_count = sum(1 for kw in method_keywords if kw in all_text)
        structure_count = sum(1 for kw in structure_keywords if kw in all_text)
        causal_count = sum(1 for kw in causal_keywords if kw in all_text)

        if method_count > structure_count and method_count > causal_count:
            return SynergyType.METHODOLOGICAL
        elif structure_count > method_count and structure_count > causal_count:
            return SynergyType.STRUCTURAL
        elif causal_count > 0:
            return SynergyType.CAUSAL
        elif similarity > 0.7:
            return SynergyType.CONCEPTUAL
        else:
            return SynergyType.ANALOGICAL

    def _extract_bridging_concepts(
        self,
        source_results: List[Dict],
        target_results: List[Dict],
        max_concepts: int = 5
    ) -> List[str]:
        """Extract concepts that bridge both domains."""
        # Simple word frequency intersection
        source_words: Dict[str, int] = {}
        target_words: Dict[str, int] = {}

        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "to", "of", "in", "for", "on", "with", "at", "by",
                     "from", "as", "into", "through", "during", "before", "after",
                     "above", "below", "between", "under", "again", "further",
                     "then", "once", "here", "there", "when", "where", "why",
                     "how", "all", "each", "few", "more", "most", "other", "some",
                     "such", "no", "nor", "not", "only", "own", "same", "so",
                     "than", "too", "very", "just", "and", "but", "if", "or",
                     "because", "until", "while", "this", "that", "these", "those"}

        for r in source_results:
            words = r.get("text", "").lower().split()
            for w in words:
                w = ''.join(c for c in w if c.isalnum())
                if len(w) > 3 and w not in stopwords:
                    source_words[w] = source_words.get(w, 0) + 1

        for r in target_results:
            words = r.get("text", "").lower().split()
            for w in words:
                w = ''.join(c for c in w if c.isalnum())
                if len(w) > 3 and w not in stopwords:
                    target_words[w] = target_words.get(w, 0) + 1

        # Find intersection weighted by frequency
        bridging = []
        for word in set(source_words.keys()) & set(target_words.keys()):
            score = min(source_words[word], target_words[word])
            bridging.append((word, score))

        bridging.sort(key=lambda x: x[1], reverse=True)

        return [w for w, _ in bridging[:max_concepts]]


# ============================================================================
# IHSAN VALIDATOR (ARTE SNR → KEP Safety)
# ============================================================================

class IhsanValidator:
    """
    Validates Ihsan constraint (excellence/perfection) across the pipeline.

    Ihsan Score = weighted combination of:
    - SNR (signal quality from ARTE)
    - Coherence (logical consistency)
    - Ethics (alignment with human values)

    Hard block if Ihsan < IHSAN_CONSTRAINT (0.99)
    """

    def __init__(self, ihsan_threshold: float = IHSAN_CONSTRAINT):
        self.threshold = ihsan_threshold
        self.snr_weight = 0.5
        self.coherence_weight = 0.3
        self.ethics_weight = 0.2

    def validate(
        self,
        snr_score: float,
        retrieval_results: List[Dict],
        synergies: List[SynergyCandidate],
        compounds: Optional[List[CompoundProposal]] = None
    ) -> IhsanCheck:
        """
        Perform comprehensive Ihsan validation.
        """
        reasoning_trace = []
        reasoning_trace.append(f"Ihsan validation started (threshold: {self.threshold})")

        # Component 1: SNR (from ARTE)
        snr_component = min(snr_score, 1.0)
        reasoning_trace.append(f"SNR component: {snr_component:.4f}")

        # Component 2: Coherence
        coherence_component = self._calculate_coherence(
            retrieval_results, synergies, compounds
        )
        reasoning_trace.append(f"Coherence component: {coherence_component:.4f}")

        # Component 3: Ethics
        ethics_component = self._calculate_ethics_score(
            retrieval_results, synergies, compounds
        )
        reasoning_trace.append(f"Ethics component: {ethics_component:.4f}")

        # Weighted combination
        ihsan_score = (
            self.snr_weight * snr_component +
            self.coherence_weight * coherence_component +
            self.ethics_weight * ethics_component
        )

        passed = ihsan_score >= self.threshold

        if not passed:
            reasoning_trace.append(f"IHSAN CHECK FAILED: {ihsan_score:.4f} < {self.threshold}")
            reasoning_trace.append("Hard block applied - output suppressed")
        else:
            reasoning_trace.append(f"IHSAN CHECK PASSED: {ihsan_score:.4f} >= {self.threshold}")

        return IhsanCheck(
            passed=passed,
            score=ihsan_score,
            snr_component=snr_component,
            coherence_component=coherence_component,
            ethics_component=ethics_component,
            reasoning_trace=reasoning_trace
        )

    def _calculate_coherence(
        self,
        retrieval_results: List[Dict],
        synergies: List[SynergyCandidate],
        compounds: Optional[List[CompoundProposal]]
    ) -> float:
        """
        Calculate logical coherence score.

        Measures:
        - Internal consistency of retrieval results
        - Synergy strength distribution
        - Compound hypothesis validity
        """
        scores = []

        # Retrieval coherence: score distribution
        if retrieval_results:
            result_scores = [r.get("score", 0.5) for r in retrieval_results]
            # Higher coherence if scores are tightly clustered (all high or consistent)
            if len(result_scores) > 1:
                std = np.std(result_scores)
                mean = np.mean(result_scores)
                # Coherence = high mean, low variance
                coherence = mean * (1 - min(std, 0.5))
            else:
                coherence = result_scores[0] if result_scores else 0.5
            scores.append(coherence)

        # Synergy coherence: evidence quality
        if synergies:
            synergy_strengths = [s.strength for s in synergies]
            avg_strength = np.mean(synergy_strengths)
            # Bonus for multiple strong synergies
            strength_bonus = min(len([s for s in synergies if s.strength > 0.7]) * 0.05, 0.15)
            scores.append(min(avg_strength + strength_bonus, 1.0))

        # Compound coherence: confidence levels
        if compounds:
            compound_confidences = [c.confidence for c in compounds]
            scores.append(np.mean(compound_confidences))

        return np.mean(scores) if scores else 0.8  # Default if no data

    def _calculate_ethics_score(
        self,
        retrieval_results: List[Dict],
        synergies: List[SynergyCandidate],
        compounds: Optional[List[CompoundProposal]]
    ) -> float:
        """
        Calculate ethics alignment score.

        Checks for:
        - Harmful content patterns
        - Bias indicators
        - Privacy concerns
        """
        # Negative indicators (reduce score if found)
        harmful_patterns = [
            "harm", "dangerous", "illegal", "exploit", "attack",
            "discriminat", "bias", "manipulat", "deceive", "malicious"
        ]

        # Positive indicators (increase score if found)
        positive_patterns = [
            "safe", "ethical", "responsible", "fair", "transparent",
            "privacy", "consent", "benef", "protect", "respect"
        ]

        all_text = ""
        for r in retrieval_results:
            all_text += r.get("text", "").lower() + " "
        for s in synergies:
            all_text += " ".join(s.bridging_concepts).lower() + " "

        # Count patterns
        harmful_count = sum(1 for p in harmful_patterns if p in all_text)
        positive_count = sum(1 for p in positive_patterns if p in all_text)

        # Base score with adjustments
        base_score = 0.95  # Start high, assume good faith
        harm_penalty = harmful_count * 0.03
        positive_bonus = positive_count * 0.01

        ethics_score = max(0.0, min(1.0, base_score - harm_penalty + positive_bonus))

        return ethics_score


# ============================================================================
# COMPOUND DISCOVERY ENGINE (Synergies → Novel Combinations)
# ============================================================================

class CompoundDiscoveryEngine:
    """
    Discovers compound knowledge from synergy analysis.

    Standing on Giants: Implements ideas from:
    - Koestler's bisociation theory
    - Arthur's combinatorial evolution
    - Kauffman's adjacent possible
    """

    def __init__(self):
        self.discovery_history: List[CompoundProposal] = []

    def discover_compounds(
        self,
        synergies: List[SynergyCandidate],
        query_context: str,
        max_compounds: int = 3
    ) -> List[CompoundProposal]:
        """
        Discover compound knowledge from synergies.

        Process:
        1. Group synergies by compatibility
        2. Generate hypotheses for combinations
        3. Score and filter proposals
        """
        if not synergies:
            return []

        compounds = []

        # Single-synergy compounds (direct application)
        for synergy in synergies[:5]:  # Top 5 synergies
            compound = self._generate_single_compound(synergy, query_context)
            if compound and compound.confidence > 0.5:
                compounds.append(compound)

        # Multi-synergy compounds (complex combinations)
        if len(synergies) >= 2:
            multi_compound = self._generate_multi_compound(
                synergies[:3], query_context
            )
            if multi_compound and multi_compound.confidence > 0.6:
                compounds.append(multi_compound)

        # Sort by confidence
        compounds.sort(key=lambda c: c.confidence, reverse=True)

        # Store in history
        self.discovery_history.extend(compounds[:max_compounds])

        logger.info(f"Discovered {len(compounds[:max_compounds])} compound proposals")

        return compounds[:max_compounds]

    def _generate_single_compound(
        self,
        synergy: SynergyCandidate,
        query_context: str
    ) -> Optional[CompoundProposal]:
        """Generate compound from single synergy."""

        # Determine compound type based on synergy type
        compound_type_mapping = {
            SynergyType.CONCEPTUAL: CompoundType.ABSTRACTION,
            SynergyType.METHODOLOGICAL: CompoundType.TRANSFORMATION,
            SynergyType.STRUCTURAL: CompoundType.FUSION,
            SynergyType.CAUSAL: CompoundType.SYNTHESIS,
            SynergyType.ANALOGICAL: CompoundType.TRANSFORMATION,
            SynergyType.EMERGENT: CompoundType.SYNTHESIS
        }

        compound_type = compound_type_mapping.get(
            synergy.synergy_type, CompoundType.FUSION
        )

        # Generate hypothesis
        hypothesis = self._generate_hypothesis(synergy, compound_type)

        # Calculate confidence
        confidence = synergy.strength * 0.7 + 0.3 * (
            len(synergy.bridging_concepts) / 5  # Bonus for rich bridging
        )
        confidence = min(confidence, 0.95)

        # Generate implications
        implications = self._generate_implications(synergy, compound_type)

        # Create validation plan
        validation_plan = {
            "approach": "empirical_test" if compound_type == CompoundType.TRANSFORMATION else "logical_validation",
            "steps": [
                f"Verify {synergy.source_domain} component validity",
                f"Verify {synergy.target_domain} component validity",
                "Test integration hypothesis",
                "Evaluate emergent properties"
            ],
            "success_criteria": "Ihsan >= 0.99 on integrated result"
        }

        return CompoundProposal(
            synergies=[synergy],
            compound_type=compound_type,
            hypothesis=hypothesis,
            confidence=confidence,
            implications=implications,
            validation_plan=validation_plan,
            metadata={
                "query_context": query_context[:100],
                "bridging_concepts": synergy.bridging_concepts
            }
        )

    def _generate_multi_compound(
        self,
        synergies: List[SynergyCandidate],
        query_context: str
    ) -> Optional[CompoundProposal]:
        """Generate compound from multiple synergies."""
        if len(synergies) < 2:
            return None

        # Collect all domains and bridging concepts
        all_domains = set()
        all_concepts = []
        for s in synergies:
            all_domains.add(s.source_domain)
            all_domains.add(s.target_domain)
            all_concepts.extend(s.bridging_concepts)

        # Remove duplicates while preserving order
        unique_concepts = list(dict.fromkeys(all_concepts))

        # Multi-synergy compounds are typically emergent
        compound_type = CompoundType.SYNTHESIS

        # Generate hypothesis
        domain_list = ", ".join(list(all_domains)[:3])
        concept_list = ", ".join(unique_concepts[:5])
        hypothesis = (
            f"Cross-domain synthesis across {domain_list} reveals emergent pattern: "
            f"The bridging concepts [{concept_list}] suggest a higher-order principle "
            f"unifying these domains."
        )

        # Confidence based on synergy strengths and overlap
        avg_strength = np.mean([s.strength for s in synergies])
        concept_overlap = len(set(synergies[0].bridging_concepts) &
                            set(synergies[1].bridging_concepts))
        confidence = avg_strength * 0.6 + 0.2 + (concept_overlap * 0.05)
        confidence = min(confidence, 0.9)

        # Generate implications
        implications = [
            f"Novel framework integrating {len(all_domains)} domains",
            f"Potential for {len(unique_concepts)} cross-applicable insights",
            "May reveal hidden connections in knowledge graph",
            "Enables multi-hop reasoning across domain boundaries"
        ]

        validation_plan = {
            "approach": "multi_stage_validation",
            "steps": [
                "Validate individual synergies",
                "Test pairwise integrations",
                "Evaluate emergent whole",
                "Apply PAT agents for reasoning verification"
            ],
            "success_criteria": "All component Ihsan >= 0.99, integration Ihsan >= 0.95"
        }

        return CompoundProposal(
            synergies=synergies,
            compound_type=compound_type,
            hypothesis=hypothesis,
            confidence=confidence,
            implications=implications,
            validation_plan=validation_plan,
            metadata={
                "domains": list(all_domains),
                "bridging_concepts": unique_concepts,
                "synergy_count": len(synergies)
            }
        )

    def _generate_hypothesis(
        self,
        synergy: SynergyCandidate,
        compound_type: CompoundType
    ) -> str:
        """Generate hypothesis string for compound."""
        templates = {
            CompoundType.FUSION: (
                f"Direct integration of {synergy.source_domain} and {synergy.target_domain} "
                f"via shared concepts [{', '.join(synergy.bridging_concepts[:3])}] "
                f"yields unified framework."
            ),
            CompoundType.SYNTHESIS: (
                f"Synthesizing {synergy.source_domain} with {synergy.target_domain} "
                f"produces novel entity with emergent properties beyond either domain."
            ),
            CompoundType.ABSTRACTION: (
                f"Higher-order principle extracted from {synergy.source_domain}-{synergy.target_domain} "
                f"synergy: [{', '.join(synergy.bridging_concepts[:2])}] as universal pattern."
            ),
            CompoundType.TRANSFORMATION: (
                f"Methods from {synergy.source_domain} transform when applied to {synergy.target_domain}, "
                f"enabled by conceptual bridges [{', '.join(synergy.bridging_concepts[:2])}]."
            ),
            CompoundType.INSTANTIATION: (
                f"Abstract {synergy.source_domain} pattern instantiated concretely in "
                f"{synergy.target_domain} context."
            )
        }
        return templates.get(compound_type, templates[CompoundType.FUSION])

    def _generate_implications(
        self,
        synergy: SynergyCandidate,
        compound_type: CompoundType
    ) -> List[str]:
        """Generate list of implications for compound."""
        base_implications = [
            f"Bridges {synergy.source_domain} and {synergy.target_domain} knowledge",
            f"Leverages {len(synergy.bridging_concepts)} cross-domain concepts"
        ]

        type_implications = {
            CompoundType.FUSION: ["Creates unified vocabulary", "Enables joint reasoning"],
            CompoundType.SYNTHESIS: ["Novel entity with unique properties", "May generate new questions"],
            CompoundType.ABSTRACTION: ["Generalizable across more domains", "Foundational principle"],
            CompoundType.TRANSFORMATION: ["Methodology transfer", "Innovation pathway"],
            CompoundType.INSTANTIATION: ["Practical application", "Concrete implementation"]
        }

        return base_implications + type_implications.get(compound_type, [])


# ============================================================================
# LEARNING ACCELERATOR (Feedback → Adaptation)
# ============================================================================

class LearningAccelerator:
    """
    Accelerates learning from KEP discoveries.

    Implements feedback loops:
    1. Success reinforcement (what worked)
    2. Failure analysis (what to avoid)
    3. Synergy boosting (amplify discoveries)
    4. Compound caching (remember breakthroughs)
    """

    def __init__(self):
        self.learning_history: List[Dict] = []
        self.synergy_weights: Dict[str, float] = {}
        self.compound_cache: Dict[str, CompoundProposal] = {}

    def calculate_boost(
        self,
        synergies: List[SynergyCandidate],
        compounds: List[CompoundProposal],
        ihsan_check: IhsanCheck
    ) -> float:
        """
        Calculate learning boost factor.

        Boost = base * synergy_factor * compound_factor * ihsan_factor
        """
        base_boost = 1.0

        # Synergy factor: more high-quality synergies = higher boost
        if synergies:
            high_quality = len([s for s in synergies if s.strength > 0.7])
            synergy_factor = 1.0 + (high_quality * 0.1)
        else:
            synergy_factor = 1.0

        # Compound factor: novel discoveries boost learning
        if compounds:
            avg_confidence = np.mean([c.confidence for c in compounds])
            compound_factor = 1.0 + (avg_confidence * 0.2)
        else:
            compound_factor = 1.0

        # Ihsan factor: only boost if quality is high
        if ihsan_check.passed:
            ihsan_factor = 1.0 + ((ihsan_check.score - 0.9) * 2)
        else:
            ihsan_factor = 0.5  # Penalty for failing Ihsan

        boost = base_boost * synergy_factor * compound_factor * ihsan_factor

        # Record for learning
        self.learning_history.append({
            "synergy_count": len(synergies),
            "compound_count": len(compounds),
            "ihsan_score": ihsan_check.score,
            "boost": boost
        })

        return min(boost, 2.5)  # Cap at 2.5x

    def apply_feedback(
        self,
        synergies: List[SynergyCandidate],
        compounds: List[CompoundProposal],
        ihsan_check: IhsanCheck
    ) -> bool:
        """
        Apply feedback to internal models.

        Updates:
        - Synergy weights (which domain pairs work well)
        - Compound cache (remember successful discoveries)
        """
        if not ihsan_check.passed:
            return False

        # Update synergy weights
        for synergy in synergies:
            key = f"{synergy.source_domain}_{synergy.target_domain}"
            current = self.synergy_weights.get(key, 1.0)
            # Exponential moving average
            self.synergy_weights[key] = current * 0.9 + synergy.strength * 0.1

        # Cache successful compounds
        for compound in compounds:
            if compound.confidence > 0.7:
                key = compound.hypothesis[:50]
                self.compound_cache[key] = compound

        # Limit cache size
        if len(self.compound_cache) > 100:
            # Remove lowest confidence entries
            sorted_cache = sorted(
                self.compound_cache.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )
            self.compound_cache = dict(sorted_cache[:100])

        logger.info(f"Feedback applied: {len(synergies)} synergies, {len(compounds)} compounds")

        return True


# ============================================================================
# KEP BRIDGE ORCHESTRATOR
# ============================================================================

class KEPBridge:
    """
    Main orchestrator bridging Data Lake engines to KEP system.

    Coordinates:
    - SynergyDetector: Hypergraph RAG → Cross-domain patterns
    - IhsanValidator: ARTE SNR → Quality gates
    - CompoundDiscoveryEngine: Synergies → Novel knowledge
    - LearningAccelerator: Results → Adaptive improvement
    """

    def __init__(self):
        logger.info("Initializing KEP Bridge...")

        self.synergy_detector = SynergyDetector()
        self.ihsan_validator = IhsanValidator()
        self.compound_engine = CompoundDiscoveryEngine()
        self.learning_accelerator = LearningAccelerator()

        self._initialized = True
        logger.info("KEP Bridge initialized successfully")

    async def process(
        self,
        query: str,
        retrieval_results: List[Dict],
        query_embedding: np.ndarray,
        snr_score: float,
        min_synergy_strength: float = 0.6,
        max_compounds: int = 3
    ) -> KEPResult:
        """
        Full KEP processing pipeline.

        Flow:
        1. Detect synergies from retrieval
        2. Discover compound knowledge
        3. Validate Ihsan constraints
        4. Calculate learning boost
        5. Apply feedback loops
        """
        start_time = time.time()

        # Step 1: Synergy Detection
        synergies = self.synergy_detector.detect_synergies(
            retrieval_results=retrieval_results,
            query_embedding=query_embedding,
            min_strength=min_synergy_strength
        )

        # Step 2: Compound Discovery
        compounds = self.compound_engine.discover_compounds(
            synergies=synergies,
            query_context=query,
            max_compounds=max_compounds
        )

        # Step 3: Ihsan Validation
        ihsan_check = self.ihsan_validator.validate(
            snr_score=snr_score,
            retrieval_results=retrieval_results,
            synergies=synergies,
            compounds=compounds
        )

        # Step 4: Learning Boost
        learning_boost = self.learning_accelerator.calculate_boost(
            synergies=synergies,
            compounds=compounds,
            ihsan_check=ihsan_check
        )

        # Step 5: Feedback Application
        feedback_applied = self.learning_accelerator.apply_feedback(
            synergies=synergies,
            compounds=compounds,
            ihsan_check=ihsan_check
        )

        execution_time = time.time() - start_time

        return KEPResult(
            query=query,
            synergies=synergies,
            compounds=compounds if ihsan_check.passed else [],  # Suppress if failed
            ihsan_check=ihsan_check,
            learning_boost=learning_boost,
            feedback_applied=feedback_applied,
            execution_time=round(execution_time, 3),
            metadata={
                "synergy_count": len(synergies),
                "compound_count": len(compounds),
                "snr_input": snr_score
            }
        )

    def get_status(self) -> Dict[str, Any]:
        """Get KEP Bridge status."""
        return {
            "initialized": self._initialized,
            "components": {
                "synergy_detector": "ready",
                "ihsan_validator": f"threshold={self.ihsan_validator.threshold}",
                "compound_engine": f"history={len(self.compound_engine.discovery_history)}",
                "learning_accelerator": f"cache={len(self.learning_accelerator.compound_cache)}"
            },
            "synergy_weights": len(self.learning_accelerator.synergy_weights),
            "learning_history": len(self.learning_accelerator.learning_history)
        }


# ============================================================================
# INTEGRATION WITH BIZRA ORCHESTRATOR
# ============================================================================

async def integrate_with_orchestrator():
    """
    Demonstrate integration between BIZRA Orchestrator and KEP Bridge.
    """
    print("=" * 70)
    print("KEP BRIDGE INTEGRATION TEST")
    print("Data Lake → Synergy → Compound Discovery → Ihsan Validation")
    print("=" * 70)

    # Initialize KEP Bridge
    kep_bridge = KEPBridge()

    print(f"\nKEP Bridge Status: {kep_bridge.get_status()}")

    # Simulate retrieval results (normally from hypergraph_engine.py)
    mock_results = [
        {
            "doc_id": "physics_quantum_001",
            "text": "Quantum entanglement enables instantaneous correlation between particles regardless of distance. This phenomenon challenges classical notions of locality.",
            "score": 0.92,
            "embedding": np.random.randn(384).tolist()
        },
        {
            "doc_id": "software_distributed_002",
            "text": "Distributed systems achieve consistency through consensus algorithms. Byzantine fault tolerance ensures correctness even with malicious nodes.",
            "score": 0.88,
            "embedding": np.random.randn(384).tolist()
        },
        {
            "doc_id": "philosophy_epistemology_003",
            "text": "Knowledge validation requires both empirical evidence and logical coherence. Epistemic humility acknowledges the limits of certainty.",
            "score": 0.85,
            "embedding": np.random.randn(384).tolist()
        },
        {
            "doc_id": "cognitive_learning_004",
            "text": "Transfer learning leverages knowledge from one domain to accelerate learning in another. Analogical reasoning bridges conceptual gaps.",
            "score": 0.82,
            "embedding": np.random.randn(384).tolist()
        }
    ]

    # Simulate query
    query = "How can distributed systems achieve reliable coordination?"
    query_embedding = np.random.randn(384)
    snr_score = 0.95

    print(f"\nProcessing query: {query}")
    print(f"Input SNR: {snr_score}")
    print("-" * 70)

    # Process through KEP Bridge
    result = await kep_bridge.process(
        query=query,
        retrieval_results=mock_results,
        query_embedding=query_embedding,
        snr_score=snr_score
    )

    # Display results
    print(f"\n📊 KEP PROCESSING RESULTS")
    print(f"Execution time: {result.execution_time}s")
    print(f"Learning boost: {result.learning_boost:.2f}x")
    print(f"Feedback applied: {result.feedback_applied}")

    print(f"\n🔗 SYNERGIES DETECTED: {len(result.synergies)}")
    for i, syn in enumerate(result.synergies):
        print(f"  {i+1}. {syn.source_domain} ↔ {syn.target_domain}")
        print(f"     Type: {syn.synergy_type.value}")
        print(f"     Strength: {syn.strength:.3f}")
        print(f"     Bridges: {', '.join(syn.bridging_concepts[:3])}")

    print(f"\n💡 COMPOUND DISCOVERIES: {len(result.compounds)}")
    for i, comp in enumerate(result.compounds):
        print(f"  {i+1}. Type: {comp.compound_type.value}")
        print(f"     Confidence: {comp.confidence:.3f}")
        print(f"     Hypothesis: {comp.hypothesis[:100]}...")

    print(f"\n✨ IHSAN VALIDATION")
    print(f"  Passed: {result.ihsan_check.passed}")
    print(f"  Score: {result.ihsan_check.score:.4f}")
    print(f"  Components:")
    print(f"    SNR: {result.ihsan_check.snr_component:.4f}")
    print(f"    Coherence: {result.ihsan_check.coherence_component:.4f}")
    print(f"    Ethics: {result.ihsan_check.ethics_component:.4f}")

    print(f"\n  Reasoning trace:")
    for step in result.ihsan_check.reasoning_trace:
        print(f"    • {step}")

    print("\n" + "=" * 70)
    print("KEP Bridge integration complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(integrate_with_orchestrator())
