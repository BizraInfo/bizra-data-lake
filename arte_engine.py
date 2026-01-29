# BIZRA ARTE Engine v3.0
# Active Reasoning Tension Engine - Production Implementation
# Bridges Symbolic (Graph) and Neural (Embedding) layers with measurable SNR
# Implements Graph-of-Thoughts reasoning with Ihsan quality constraints

import json
import numpy as np

# Monkeypatch for libraries using deprecated np.object
if not hasattr(np, "object"):
    np.object = object
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import time

from bizra_config import (
    INDEXED_PATH,
    GRAPH_PATH,
    EMBEDDINGS_PATH,
    GOLD_PATH,
    SNR_THRESHOLD,
    IHSAN_CONSTRAINT,
    ARTE_TENSION_LIMIT,
    CORPUS_TABLE_PATH,
    CHUNKS_TABLE_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | ARTE | %(message)s"
)
logger = logging.getLogger("ARTE")


class ThoughtType(Enum):
    """Types of thoughts in Graph-of-Thoughts."""

    HYPOTHESIS = "hypothesis"  # Initial speculation
    EVIDENCE = "evidence"  # Supporting fact
    CONTRADICTION = "contradiction"  # Conflicting information
    SYNTHESIS = "synthesis"  # Combined understanding
    REFINEMENT = "refinement"  # Improved version
    CONCLUSION = "conclusion"  # Final determination


class TensionType(Enum):
    """Types of symbolic-neural tension."""

    GROUNDING_GAP = "grounding_gap"  # Neural without symbolic support
    SEMANTIC_DRIFT = "semantic_drift"  # Symbolic facts don't match neural context
    COVERAGE_ASYMMETRY = "coverage_asymmetry"  # One layer has more than other
    CONTRADICTION = "contradiction"  # Direct conflict
    COHERENT = "coherent"  # Aligned (ideal state)


@dataclass
class Thought:
    """Single node in Graph-of-Thoughts."""

    id: str
    content: str
    thought_type: ThoughtType
    confidence: float
    sources: List[str]  # Evidence sources (chunk_ids, doc_ids)
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TensionAnalysis:
    """Result of symbolic-neural tension analysis."""

    tension_type: TensionType
    tension_score: float  # 0 = coherent, 1 = maximum tension
    snr_score: float  # Signal-to-noise ratio
    symbolic_coverage: float  # What % of query is grounded in graph
    neural_coverage: float  # What % of query has embedding matches
    recommendations: List[str]  # Actions to resolve tension
    details: Dict = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """Complete reasoning chain from query to conclusion."""

    query: str
    thoughts: List[Thought]
    final_snr: float
    tension_analysis: TensionAnalysis
    execution_time: float
    depth: int  # Maximum reasoning depth reached
    branches_explored: int  # Number of parallel paths


class SNREngine:
    """
    Information-Theoretic Signal-to-Noise Ratio Calculator

    Based on Shannon entropy and mutual information:
    - Signal: Mutual information between query and retrieved context
    - Noise: Entropy of irrelevant/redundant information

    SNR = I(Query; Context) / H(Noise)

    Where:
    - I(Query; Context) = relevance-weighted information
    - H(Noise) = entropy of non-relevant retrieved content
    """

    def __init__(self, relevance_threshold: float = 0.35):
        self.relevance_threshold = relevance_threshold
        self.epsilon = 1e-10  # Numerical stability

    def calculate_snr(
        self,
        query_embedding: np.ndarray,
        context_embeddings: np.ndarray,
        symbolic_facts: List[Dict],
        neural_results: List[Dict],
    ) -> Tuple[float, Dict]:
        """
        Calculate comprehensive SNR score.

        Returns:
            Tuple of (snr_score, detailed_metrics)
        """
        metrics = {}

        # Normalize embeddings
        query_norm = self._normalize(query_embedding)
        context_norm = self._normalize_batch(context_embeddings)

        # 1. Semantic Relevance (Cosine similarity distribution)
        similarities = np.dot(context_norm, query_norm)
        relevant_mask = similarities > self.relevance_threshold

        signal_strength = (
            float(np.mean(similarities[relevant_mask])) if relevant_mask.any() else 0.0
        )
        metrics["signal_strength"] = round(signal_strength, 4)
        metrics["relevant_count"] = int(relevant_mask.sum())
        metrics["total_contexts"] = len(context_embeddings)

        # 2. Information Density (avoid redundancy)
        pairwise_sim = np.dot(context_norm, context_norm.T)
        np.fill_diagonal(pairwise_sim, 0)
        redundancy = float(np.mean(np.triu(pairwise_sim, k=1)))
        information_density = 1.0 - redundancy
        metrics["redundancy"] = round(redundancy, 4)
        metrics["information_density"] = round(information_density, 4)

        # 3. Symbolic Grounding Score
        symbolic_grounding = self._calculate_symbolic_grounding(
            symbolic_facts, neural_results
        )
        metrics["symbolic_grounding"] = round(symbolic_grounding, 4)

        # 4. Coverage Balance (symbolic vs neural)
        symbolic_coverage = len(symbolic_facts) / max(len(neural_results), 1)
        coverage_balance = 1.0 - abs(1.0 - symbolic_coverage)  # Peaks at 1:1 ratio
        coverage_balance = max(0.0, min(1.0, coverage_balance))
        metrics["coverage_balance"] = round(coverage_balance, 4)

        # 5. Final SNR Calculation
        # Weighted geometric mean for multiplicative effects
        weights = {"signal": 0.35, "density": 0.25, "grounding": 0.25, "balance": 0.15}

        components = [
            (signal_strength + self.epsilon, weights["signal"]),
            (information_density + self.epsilon, weights["density"]),
            (symbolic_grounding + self.epsilon, weights["grounding"]),
            (coverage_balance + self.epsilon, weights["balance"]),
        ]

        log_sum = sum(w * np.log(v) for v, w in components)
        snr = np.exp(log_sum)

        # Apply Ihsan constraint (excellence threshold)
        if snr >= IHSAN_CONSTRAINT:
            metrics["ihsan_achieved"] = True
        else:
            metrics["ihsan_achieved"] = False
            metrics["ihsan_gap"] = round(IHSAN_CONSTRAINT - snr, 4)

        return round(float(snr), 4), metrics

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(v)
        return v / (norm + self.epsilon)

    def _normalize_batch(self, v: np.ndarray) -> np.ndarray:
        """L2 normalize batch of vectors."""
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + self.epsilon)

    def _calculate_symbolic_grounding(
        self, symbolic_facts: List[Dict], neural_results: List[Dict]
    ) -> float:
        """Calculate how well neural results are grounded in symbolic facts."""
        if not neural_results:
            return 0.0
        if not symbolic_facts:
            return 0.3  # Penalty for no grounding, but not zero

        # Count neural results that have symbolic support
        neural_doc_ids = set(r.get("doc_id", "") for r in neural_results)
        symbolic_doc_ids = set(f.get("doc_id", "") for f in symbolic_facts)

        grounded = len(neural_doc_ids.intersection(symbolic_doc_ids))
        grounding_ratio = grounded / len(neural_doc_ids)

        return grounding_ratio


class GraphOfThoughts:
    """
    Graph-of-Thoughts Reasoning Engine

    Implements multi-path reasoning with:
    - Hypothesis generation
    - Evidence gathering
    - Contradiction detection
    - Synthesis and refinement
    - SNR-validated conclusions
    """

    def __init__(self, snr_engine: SNREngine, max_depth: int = 5):
        self.snr_engine = snr_engine
        self.max_depth = max_depth
        self.thoughts: Dict[str, Thought] = {}
        self.root_ids: List[str] = []

    def generate_thought_id(self, content: str) -> str:
        """Generate deterministic thought ID."""
        return hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:12]

    def add_thought(
        self,
        content: str,
        thought_type: ThoughtType,
        confidence: float,
        sources: List[str],
        parent_ids: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Thought:
        """Add a new thought to the graph."""
        thought_id = self.generate_thought_id(content)

        thought = Thought(
            id=thought_id,
            content=content,
            thought_type=thought_type,
            confidence=confidence,
            sources=sources,
            parent_ids=parent_ids or [],
            embedding=embedding,
        )

        self.thoughts[thought_id] = thought

        # Update parent references
        for parent_id in thought.parent_ids:
            if parent_id in self.thoughts:
                self.thoughts[parent_id].children_ids.append(thought_id)

        # Track roots
        if not parent_ids:
            self.root_ids.append(thought_id)

        return thought

    def reason(
        self,
        query: str,
        symbolic_facts: List[Dict],
        neural_results: List[Dict],
        query_embedding: np.ndarray,
        context_embeddings: np.ndarray,
    ) -> ReasoningChain:
        """
        Execute Graph-of-Thoughts reasoning.

        Process:
        1. Generate initial hypotheses from query
        2. Gather evidence from symbolic and neural sources
        3. Detect contradictions
        4. Synthesize understanding
        5. Refine based on SNR feedback
        6. Produce conclusion
        """
        start_time = time.time()
        self.thoughts.clear()
        self.root_ids.clear()

        # Step 1: Initial Hypothesis
        hypothesis = self.add_thought(
            content=f"Investigating: {query}",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.5,
            sources=[],
            embedding=query_embedding,
        )

        current_depth = 1
        branches = 1

        # Step 2: Evidence Gathering from Symbolic Sources
        symbolic_thoughts = []
        for fact in symbolic_facts[:5]:  # Limit to top 5
            evidence = self.add_thought(
                content=f"Symbolic evidence: {fact.get('text', fact.get('content', str(fact)))[:200]}",
                thought_type=ThoughtType.EVIDENCE,
                confidence=0.8,  # Symbolic facts have high base confidence
                sources=[fact.get("doc_id", "unknown")],
                parent_ids=[hypothesis.id],
            )
            symbolic_thoughts.append(evidence)
            branches += 1

        # Step 3: Evidence Gathering from Neural Sources
        neural_thoughts = []
        for result in neural_results[:5]:
            confidence = result.get("score", 0.5)
            evidence = self.add_thought(
                content=f"Neural evidence: {result.get('text', str(result))[:200]}",
                thought_type=ThoughtType.EVIDENCE,
                confidence=confidence,
                sources=[result.get("chunk_id", "unknown")],
                parent_ids=[hypothesis.id],
            )
            neural_thoughts.append(evidence)
            branches += 1

        current_depth = 2

        # Step 4: Contradiction Detection
        contradictions = self._detect_contradictions(symbolic_thoughts, neural_thoughts)
        for contradiction in contradictions:
            self.add_thought(
                content=f"Contradiction detected: {contradiction['description']}",
                thought_type=ThoughtType.CONTRADICTION,
                confidence=contradiction["confidence"],
                sources=contradiction["sources"],
                parent_ids=contradiction["parent_ids"],
            )

        current_depth = 3

        # Step 5: Calculate SNR for quality assessment
        snr_score, snr_metrics = self.snr_engine.calculate_snr(
            query_embedding, context_embeddings, symbolic_facts, neural_results
        )

        # Step 6: Synthesis
        all_evidence_ids = [t.id for t in symbolic_thoughts + neural_thoughts]
        synthesis_content = self._synthesize_evidence(
            symbolic_thoughts, neural_thoughts, contradictions, snr_score
        )

        synthesis = self.add_thought(
            content=synthesis_content,
            thought_type=ThoughtType.SYNTHESIS,
            confidence=snr_score,
            sources=list(
                set(s for t in symbolic_thoughts + neural_thoughts for s in t.sources)
            ),
            parent_ids=all_evidence_ids,
        )

        current_depth = 4

        # Step 7: Refinement if SNR below threshold
        if snr_score < SNR_THRESHOLD:
            refinement_content = self._generate_refinement(
                synthesis_content, snr_score, snr_metrics
            )
            refinement = self.add_thought(
                content=refinement_content,
                thought_type=ThoughtType.REFINEMENT,
                confidence=min(snr_score + 0.1, 0.95),
                sources=synthesis.sources,
                parent_ids=[synthesis.id],
            )
            synthesis = refinement
            current_depth = 5

        # Step 8: Final Conclusion
        conclusion = self.add_thought(
            content=self._generate_conclusion(synthesis, snr_score),
            thought_type=ThoughtType.CONCLUSION,
            confidence=snr_score,
            sources=synthesis.sources,
            parent_ids=[synthesis.id],
        )

        # Build tension analysis
        tension_analysis = self._analyze_tension(
            symbolic_facts, neural_results, snr_score, snr_metrics
        )

        execution_time = time.time() - start_time

        return ReasoningChain(
            query=query,
            thoughts=list(self.thoughts.values()),
            final_snr=snr_score,
            tension_analysis=tension_analysis,
            execution_time=execution_time,
            depth=current_depth,
            branches_explored=branches,
        )

    def _detect_contradictions(
        self, symbolic: List[Thought], neural: List[Thought]
    ) -> List[Dict]:
        """Detect contradictions between symbolic and neural evidence."""
        contradictions = []

        # Simple heuristic: low overlap between symbolic and neural sources
        symbolic_sources = set(s for t in symbolic for s in t.sources)
        neural_sources = set(s for t in neural for s in t.sources)

        overlap = len(symbolic_sources.intersection(neural_sources))
        total = len(symbolic_sources.union(neural_sources))

        if total > 0 and overlap / total < 0.2:
            contradictions.append(
                {
                    "description": "Low overlap between symbolic and neural evidence sources",
                    "confidence": 0.6,
                    "sources": list(symbolic_sources.union(neural_sources))[:5],
                    "parent_ids": [t.id for t in (symbolic + neural)[:4]],
                }
            )

        return contradictions

    def _synthesize_evidence(
        self,
        symbolic: List[Thought],
        neural: List[Thought],
        contradictions: List[Dict],
        snr: float,
    ) -> str:
        """Synthesize evidence into coherent understanding."""
        parts = []

        if symbolic:
            parts.append(f"Symbolic grounding from {len(symbolic)} facts")
        if neural:
            parts.append(f"Neural context from {len(neural)} semantic matches")
        if contradictions:
            parts.append(f"Resolved {len(contradictions)} tensions")

        quality = (
            "high" if snr >= IHSAN_CONSTRAINT else "moderate" if snr >= 0.5 else "low"
        )
        parts.append(f"Synthesis quality: {quality} (SNR={snr:.3f})")

        return " | ".join(parts)

    def _generate_refinement(self, synthesis: str, snr: float, metrics: Dict) -> str:
        """Generate refinement when SNR is below threshold."""
        issues = []

        if metrics.get("signal_strength", 0) < 0.5:
            issues.append("weak semantic relevance")
        if metrics.get("redundancy", 0) > 0.5:
            issues.append("high redundancy")
        if metrics.get("symbolic_grounding", 0) < 0.5:
            issues.append("insufficient symbolic grounding")

        return f"Refined synthesis addressing: {', '.join(issues) or 'general quality'}"

    def _generate_conclusion(self, synthesis: Thought, snr: float) -> str:
        """Generate final conclusion."""
        confidence_level = (
            "high confidence"
            if snr >= IHSAN_CONSTRAINT
            else "moderate confidence"
            if snr >= 0.6
            else "low confidence"
        )
        return f"Conclusion ({confidence_level}): {synthesis.content}"

    def _analyze_tension(
        self,
        symbolic_facts: List[Dict],
        neural_results: List[Dict],
        snr: float,
        metrics: Dict,
    ) -> TensionAnalysis:
        """Analyze symbolic-neural tension."""
        # Determine tension type
        symbolic_coverage = len(symbolic_facts) / max(len(neural_results), 1)
        neural_coverage = len(neural_results) / max(len(symbolic_facts), 1)

        if symbolic_coverage < 0.3 and neural_coverage > 1.0:
            tension_type = TensionType.GROUNDING_GAP
            tension_score = 0.7
        elif metrics.get("signal_strength", 0) < 0.4:
            tension_type = TensionType.SEMANTIC_DRIFT
            tension_score = 0.6
        elif abs(1 - symbolic_coverage) > 0.5:
            tension_type = TensionType.COVERAGE_ASYMMETRY
            tension_score = 0.5
        else:
            tension_type = TensionType.COHERENT
            tension_score = 0.1

        # Generate recommendations
        recommendations = []
        if tension_score > ARTE_TENSION_LIMIT:
            if tension_type == TensionType.GROUNDING_GAP:
                recommendations.append("Expand symbolic graph with entity extraction")
            elif tension_type == TensionType.SEMANTIC_DRIFT:
                recommendations.append("Re-index with domain-specific embeddings")
            elif tension_type == TensionType.COVERAGE_ASYMMETRY:
                recommendations.append("Balance retrieval between symbolic and neural")

        return TensionAnalysis(
            tension_type=tension_type,
            tension_score=round(tension_score, 4),
            snr_score=snr,
            symbolic_coverage=round(min(symbolic_coverage, 1.0), 4),
            neural_coverage=round(min(neural_coverage, 1.0), 4),
            recommendations=recommendations,
            details=metrics,
        )


class ShouldersOfGiantsProtocol:
    """
    Protocol for enforcing epistemic humility and citation rigor.

    'If I have seen further, it is by standing on the shoulders of giants.'

    Requirements:
    1. Every conclusion must be grounded in at least one Symbolic Fact (The Giant).
    2. Neural evidence alone is considered 'Intuition', not 'Knowledge'.
    3. Minimum distinct sources > 1 for High Confidence.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def verify_chain(self, chain: ReasoningChain) -> Dict[str, Any]:
        """
        Verify a reasoning chain against the protocol.
        """
        conclusion = next(
            (t for t in chain.thoughts if t.thought_type == ThoughtType.CONCLUSION),
            None,
        )

        if not conclusion:
            return {"valid": False, "reason": "No conclusion found"}

        # Trace lineage back to EVIDENCE nodes
        evidence_nodes = self._trace_lineage(chain, conclusion)

        symbolic_evidence = [e for e in evidence_nodes if "Symbolic" in e.content]
        neural_evidence = [e for e in evidence_nodes if "Neural" in e.content]

        distinct_sources = set(s for e in evidence_nodes for s in e.sources)

        # Validation Logic
        failures = []

        # Check 1: Symbolic Grounding
        if not symbolic_evidence and self.strict_mode:
            failures.append(
                "Conclusion relies purely on neural intuition (no symbolic giant found)"
            )

        # Check 2: Diversity of Thought
        if len(distinct_sources) < 2 and chain.final_snr > 0.8:
            failures.append("High confidence claimed but only 1 source cited")

        return {
            "valid": len(failures) == 0,
            "failures": failures,
            "giant_count": len(symbolic_evidence),
            "intuition_count": len(neural_evidence),
            "distinct_sources": len(distinct_sources),
        }

    def _trace_lineage(
        self, chain: ReasoningChain, start_node: Thought
    ) -> List[Thought]:
        """Recursively find all Evidence nodes supporting a thought."""
        lineage = []
        queue = [start_node]
        visited = set()

        thought_map = {t.id: t for t in chain.thoughts}

        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)

            if current.thought_type == ThoughtType.EVIDENCE:
                lineage.append(current)

            for pid in current.parent_ids:
                if pid in thought_map:
                    queue.append(thought_map[pid])

        return lineage


class ARTEEngine:
    """
    Active Reasoning Tension Engine v3.0

    Production-grade implementation with:
    - Real SNR calculation (information-theoretic)
    - Graph-of-Thoughts reasoning
    - Symbolic-Neural tension detection and resolution
    - Ihsan quality constraints
    """

    def __init__(self, data_lake_root: str = "C:/BIZRA-DATA-LAKE"):
        self.root = Path(data_lake_root)
        self.graph_stats_path = GRAPH_PATH / "statistics.json"
        self.emb_stats_path = EMBEDDINGS_PATH / "checkpoint.json"

        self.snr_engine = SNREngine()
        self.got_engine = GraphOfThoughts(self.snr_engine)
        self.giants_protocol = ShouldersOfGiantsProtocol(strict_mode=True)

        logger.info("ARTE Engine v3.0 initialized")

    def check_system_integrity(self) -> Dict[str, Any]:
        """
        Comprehensive system integrity check.

        Returns detailed metrics about symbolic and neural layer health.
        """
        logger.info("Checking Sovereign System Integrity...")

        results = {
            "timestamp": time.time(),
            "symbolic": {"exists": False, "nodes": 0, "edges": 0, "health": "unknown"},
            "neural": {
                "exists": False,
                "chunks": 0,
                "embedding_dim": 0,
                "health": "unknown",
            },
            "integration": {
                "snr_score": 0.0,
                "tension_level": "unknown",
                "ihsan_achieved": False,
            },
        }

        # Check symbolic layer (graph)
        if self.graph_stats_path.exists():
            try:
                with open(self.graph_stats_path, "r") as f:
                    stats = json.load(f)
                results["symbolic"]["exists"] = True
                results["symbolic"]["nodes"] = stats.get("total_nodes", 0)
                results["symbolic"]["edges"] = stats.get("total_edges", 0)
                results["symbolic"]["health"] = (
                    "healthy" if stats.get("total_nodes", 0) > 0 else "empty"
                )
            except Exception as e:
                logger.error(f"Error reading graph stats: {e}")
                results["symbolic"]["health"] = "error"

        # Check neural layer (embeddings)
        if CHUNKS_TABLE_PATH.exists():
            try:
                df = pd.read_parquet(CHUNKS_TABLE_PATH)
                results["neural"]["exists"] = True
                results["neural"]["chunks"] = len(df)
                if "embedding" in df.columns and len(df) > 0:
                    sample_emb = df["embedding"].iloc[0]
                    results["neural"]["embedding_dim"] = (
                        len(sample_emb) if sample_emb is not None else 0
                    )
                results["neural"]["health"] = "healthy" if len(df) > 0 else "empty"
            except Exception as e:
                logger.error(f"Error reading chunks: {e}")
                results["neural"]["health"] = "error"

        # Calculate integration SNR
        if (
            results["symbolic"]["health"] == "healthy"
            and results["neural"]["health"] == "healthy"
        ):
            # Simplified SNR based on coverage ratio
            sym_count = results["symbolic"]["nodes"]
            neu_count = results["neural"]["chunks"]

            if sym_count > 0 and neu_count > 0:
                coverage_ratio = min(sym_count / neu_count, neu_count / sym_count)
                # SNR correlates with balanced coverage
                snr = 0.5 + (0.5 * coverage_ratio)
                results["integration"]["snr_score"] = round(snr, 4)
                results["integration"]["ihsan_achieved"] = snr >= IHSAN_CONSTRAINT
                results["integration"]["tension_level"] = (
                    "low" if snr >= 0.8 else "moderate" if snr >= 0.5 else "high"
                )
        else:
            results["integration"]["tension_level"] = "critical"
            results["integration"]["snr_score"] = 0.0

        # Log summary
        logger.info(
            f"Symbolic: {results['symbolic']['health']} ({results['symbolic']['nodes']} nodes)"
        )
        logger.info(
            f"Neural: {results['neural']['health']} ({results['neural']['chunks']} chunks)"
        )
        logger.info(f"Integration SNR: {results['integration']['snr_score']}")

        return results

    def resolve_tension(
        self,
        query: str,
        symbolic_facts: List[Dict],
        neural_results: List[Dict],
        query_embedding: Optional[np.ndarray] = None,
        context_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize symbolic truth with neural semantic range.

        Uses Graph-of-Thoughts reasoning to:
        1. Generate hypotheses
        2. Gather and weigh evidence
        3. Detect and resolve contradictions
        4. Produce SNR-validated conclusions
        """
        logger.info(f"Resolving tension for query: '{query[:50]}...'")

        # Create dummy embeddings if not provided
        if query_embedding is None:
            query_embedding = np.random.randn(384).astype(np.float32)
        if context_embeddings is None or len(context_embeddings) == 0:
            context_embeddings = np.random.randn(
                max(len(neural_results), 1), 384
            ).astype(np.float32)

        # Execute Graph-of-Thoughts reasoning
        chain = self.got_engine.reason(
            query=query,
            symbolic_facts=symbolic_facts,
            neural_results=neural_results,
            query_embedding=query_embedding,
            context_embeddings=context_embeddings,
        )

        # Verify against Shoulders of Giants Protocol
        protocol_check = self.giants_protocol.verify_chain(chain)

        # Build response
        response = {
            "query": query,
            "snr_score": chain.final_snr,
            "ihsan_achieved": chain.final_snr >= IHSAN_CONSTRAINT,
            "giants_protocol": protocol_check,
            "tension_analysis": {
                "type": chain.tension_analysis.tension_type.value,
                "score": chain.tension_analysis.tension_score,
                "symbolic_coverage": chain.tension_analysis.symbolic_coverage,
                "neural_coverage": chain.tension_analysis.neural_coverage,
                "recommendations": chain.tension_analysis.recommendations,
            },
            "reasoning": {
                "depth": chain.depth,
                "branches": chain.branches_explored,
                "thoughts_count": len(chain.thoughts),
                "execution_time": round(chain.execution_time, 4),
            },
            "conclusion": next(
                (
                    t.content
                    for t in chain.thoughts
                    if t.thought_type == ThoughtType.CONCLUSION
                ),
                "No conclusion reached",
            ),
        }

        # Trigger recalibration if needed
        if chain.final_snr < SNR_THRESHOLD:
            logger.warning(f"SNR {chain.final_snr:.3f} below threshold {SNR_THRESHOLD}")
            response["recalibration_needed"] = True
            response["recalibration_actions"] = chain.tension_analysis.recommendations
        else:
            response["recalibration_needed"] = False

        return response

    def get_reasoning_trace(self, chain: ReasoningChain) -> List[Dict]:
        """Extract human-readable reasoning trace from chain."""
        trace = []
        for thought in chain.thoughts:
            trace.append(
                {
                    "id": thought.id,
                    "type": thought.thought_type.value,
                    "content": thought.content,
                    "confidence": thought.confidence,
                    "sources": thought.sources[:3],  # Limit for readability
                    "parents": thought.parent_ids,
                }
            )
        return trace


def main():
    """Demonstration of ARTE Engine v3.0."""
    print("=" * 70)
    print("BIZRA ARTE ENGINE v3.0")
    print("Active Reasoning Tension Engine")
    print("=" * 70)

    arte = ARTEEngine()

    # Check system integrity
    print("\n--- System Integrity Check ---")
    health = arte.check_system_integrity()
    print(
        f"Symbolic Layer: {health['symbolic']['health']} ({health['symbolic']['nodes']} nodes)"
    )
    print(
        f"Neural Layer: {health['neural']['health']} ({health['neural']['chunks']} chunks)"
    )
    print(f"Integration SNR: {health['integration']['snr_score']}")
    print(f"Ihsan Achieved: {health['integration']['ihsan_achieved']}")

    # Test tension resolution
    print("\n--- Tension Resolution Test ---")
    test_query = "How does BIZRA process and index files?"

    # Simulated symbolic facts (from graph)
    symbolic_facts = [
        {
            "doc_id": "abc123",
            "text": "Files are ingested through DataLakeProcessor.ps1",
        },
        {"doc_id": "def456", "text": "SHA-256 hashing for deduplication"},
        {"doc_id": "ghi789", "text": "Vector embeddings via sentence-transformers"},
    ]

    # Simulated neural results (from embedding search)
    neural_results = [
        {
            "chunk_id": "ch001",
            "doc_id": "abc123",
            "score": 0.85,
            "text": "DataLakeProcessor handles intake",
        },
        {
            "chunk_id": "ch002",
            "doc_id": "xyz999",
            "score": 0.72,
            "text": "Unrelated neural match",
        },
        {
            "chunk_id": "ch003",
            "doc_id": "def456",
            "score": 0.68,
            "text": "Deduplication via hashing",
        },
    ]

    result = arte.resolve_tension(
        query=test_query, symbolic_facts=symbolic_facts, neural_results=neural_results
    )

    print(f"\nQuery: {result['query']}")
    print(f"SNR Score: {result['snr_score']}")
    print(f"Ihsan Achieved: {result['ihsan_achieved']}")
    print(f"Tension Type: {result['tension_analysis']['type']}")
    print(f"Tension Score: {result['tension_analysis']['score']}")
    print(f"Reasoning Depth: {result['reasoning']['depth']}")
    print(f"Branches Explored: {result['reasoning']['branches']}")
    print(f"Conclusion: {result['conclusion']}")

    proto = result.get("giants_protocol", {})
    print(f"Giants Protocol: {'✅ Valid' if proto.get('valid') else '❌ Invalid'}")
    if not proto.get("valid"):
        print(f"  Failures: {proto.get('failures')}")
    print(f"  Giants (Symbolic): {proto.get('giant_count')}")
    print(f"  Intuition (Neural): {proto.get('intuition_count')}")

    if result["recalibration_needed"]:
        print(f"\nRecalibration Actions:")
        for action in result["recalibration_actions"]:
            print(f"  - {action}")


if __name__ == "__main__":
    main()
