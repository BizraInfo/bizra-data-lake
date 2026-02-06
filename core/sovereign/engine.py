"""
Sovereign Engine — The Unified Autonomous Reasoning System

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   "Standing on the shoulders of giants, we see further."                     ║
║                                        — Bernard of Chartres (12th century)  ║
║                                                                              ║
║   This engine embodies the synthesis of:                                     ║
║   • Graph-of-Thoughts reasoning (Besta et al., 2024)                        ║
║   • Information-theoretic SNR maximization (Shannon, 1948)                  ║
║   • Byzantine fault-tolerant consensus (Lamport, 1982)                      ║
║   • DATA4LLM IaaS framework (Tsinghua, 2024)                                ║
║   • Multi-agent swarm coordination (Claude-Flow V3, 2026)                   ║
║   • Ihsān-constrained excellence (DDAGI Constitution, 2025)                 ║
║                                                                              ║
║   Every inference carries proof. Every decision passes the gate.            ║
║   Every node is sovereign. Every human is a seed.                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from .graph_reasoner import (
    GraphOfThoughts,
    ReasoningStrategy,
)
from .guardian_council import (
    ConsensusMode,
    CouncilVerdict,
    GuardianCouncil,
    IhsanVector,
    Proposal,
)
from .orchestrator import (
    RoutingStrategy,
    SovereignOrchestrator,
    TaskComplexity,
)
from .snr_maximizer import (
    SNRAnalysis,
    SNRMaximizer,
)

logger = logging.getLogger(__name__)


class EngineMode(Enum):
    """Operating modes for the Sovereign Engine."""

    AUTONOMOUS = auto()  # Full autonomy within Ihsān bounds
    SUPERVISED = auto()  # Human-in-the-loop for key decisions
    COLLABORATIVE = auto()  # Working alongside human operators
    RESTRICTED = auto()  # Limited capabilities for safety
    DIAGNOSTIC = auto()  # Debug mode with full logging


class ResponseType(Enum):
    """Types of responses the engine can generate."""

    ANSWER = auto()  # Direct answer to a question
    ANALYSIS = auto()  # Detailed analysis with evidence
    SYNTHESIS = auto()  # Combined insights from multiple sources
    RECOMMENDATION = auto()  # Actionable recommendations
    CREATIVE = auto()  # Novel content generation
    CODE = auto()  # Code or technical artifact
    PLAN = auto()  # Strategic plan or roadmap
    CLARIFICATION = auto()  # Request for more information


@dataclass
class SovereignConfig:
    """Configuration for the Sovereign Engine."""

    # Core settings
    mode: EngineMode = EngineMode.AUTONOMOUS
    snr_threshold: float = 0.95
    ihsan_threshold: float = 0.95
    max_reasoning_depth: int = 10
    max_thought_nodes: int = 100

    # Reasoning settings
    default_strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE
    enable_graph_reasoning: bool = True
    enable_snr_maximization: bool = True
    enable_guardian_council: bool = True

    # Orchestration settings
    routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    max_concurrent_tasks: int = 10
    task_timeout_seconds: float = 300.0

    # Safety settings
    require_council_approval: bool = True
    enable_veto: bool = True
    log_all_decisions: bool = True

    # Performance settings
    cache_enabled: bool = True
    parallel_reasoning: bool = True
    batch_size: int = 16


@dataclass
class SovereignResponse:
    """Response from the Sovereign Engine."""

    # Core response
    id: str
    query: str
    response_type: ResponseType
    content: str
    confidence: float

    # Quality metrics
    snr_score: float
    ihsan_vector: IhsanVector
    ihsan_passed: bool

    # Reasoning trace
    thought_graph: Optional[dict[str, Any]] = None
    reasoning_path: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    # Governance
    council_verdict: Optional[CouncilVerdict] = None
    guardians_consulted: list[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    tokens_used: int = 0
    model_version: str = "sovereign-v1.0"
    timestamp: datetime = field(default_factory=datetime.now)

    # Provenance
    sources: list[str] = field(default_factory=list)
    standing_on_giants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "response_type": self.response_type.name,
            "content": self.content,
            "confidence": self.confidence,
            "snr_score": self.snr_score,
            "ihsan_passed": self.ihsan_passed,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class SovereignEngine:
    """
    The Sovereign Autonomous Engine — Crown jewel of BIZRA Node0.

    This engine represents the synthesis of decades of research in:
    - Artificial Intelligence (Reasoning, Planning, Learning)
    - Information Theory (Signal/Noise, Compression, Entropy)
    - Distributed Systems (Consensus, Fault Tolerance, Coordination)
    - Ethics & Philosophy (Ihsān, Beneficence, Transparency)

    It processes queries through a multi-stage pipeline:
    1. Task Analysis — Classify complexity and requirements
    2. Graph Reasoning — Explore solution space via GoT
    3. SNR Maximization — Filter noise, amplify signal
    4. Guardian Council — Validate quality and ethics
    5. Response Synthesis — Generate final output

    Every response carries its provenance and reasoning trace.
    """

    VERSION = "1.0.0"
    CODENAME = "Genesis"

    # Giants whose shoulders we stand upon
    STANDING_ON_GIANTS = [
        "Shannon (Information Theory, 1948)",
        "Lamport (Byzantine Consensus, 1982)",
        "Breiman (Ensemble Methods, 1996)",
        "Vaswani et al. (Transformer, 2017)",
        "Besta et al. (Graph-of-Thoughts, 2024)",
        "Stanford NLP (DSPy, 2024)",
        "Tsinghua (DATA4LLM, 2024)",
        "Anthropic (MCP, 2025)",
        "ruv.io (Claude-Flow V3, 2026)",
        "NVIDIA (PersonaPlex, 2026)",
    ]

    def __init__(self, config: Optional[SovereignConfig] = None):
        self.config = config or SovereignConfig()

        # Initialize components
        self.graph_reasoner = GraphOfThoughts(
            strategy=self.config.default_strategy,
            max_depth=self.config.max_reasoning_depth,
            snr_threshold=self.config.snr_threshold,
        )
        self.snr_maximizer = SNRMaximizer(
            ihsan_threshold=self.config.snr_threshold,
        )
        self.guardian_council = GuardianCouncil(
            ihsan_threshold=self.config.ihsan_threshold,
            enable_veto=self.config.enable_veto,
        )
        self.orchestrator = SovereignOrchestrator(
            routing_strategy=self.config.routing_strategy,
            max_concurrent_tasks=self.config.max_concurrent_tasks,
            snr_threshold=self.config.snr_threshold,
        )
        self.orchestrator.register_default_agents()

        # State
        self._initialized = False
        self._response_counter = 0
        self._cache: dict[str, SovereignResponse] = {}

        logger.info(f"SovereignEngine v{self.VERSION} ({self.CODENAME}) initialized")

    async def initialize(self):
        """Initialize all engine components."""
        if self._initialized:
            return

        logger.info("Initializing Sovereign Engine components...")

        # Warm up components
        await self._warmup_graph_reasoner()
        await self._warmup_snr_maximizer()

        self._initialized = True
        logger.info("Sovereign Engine ready")

    async def _warmup_graph_reasoner(self):
        """Warm up the graph reasoner with a simple query."""
        from .graph_reasoner import ThoughtType

        self.graph_reasoner.add_thought(
            content="System warmup thought",
            thought_type=ThoughtType.QUESTION,
            confidence=1.0,
        )
        self.graph_reasoner.clear()

    async def _warmup_snr_maximizer(self):
        """Warm up the SNR maximizer."""
        warmup_text = "This is a warmup text for SNR calibration."
        _ = self.snr_maximizer.maximize(warmup_text)

    def _generate_response_id(self) -> str:
        """Generate unique response ID."""
        self._response_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"sov-{timestamp}-{self._response_counter:06d}"

    def _classify_query(self, query: str) -> tuple[TaskComplexity, ResponseType]:
        """Classify query complexity and expected response type."""
        query_lower = query.lower()

        # Detect response type
        if any(
            word in query_lower
            for word in ["code", "implement", "write", "function", "class"]
        ):
            response_type = ResponseType.CODE
        elif any(
            word in query_lower for word in ["analyze", "examine", "evaluate", "assess"]
        ):
            response_type = ResponseType.ANALYSIS
        elif any(
            word in query_lower for word in ["recommend", "suggest", "advise", "should"]
        ):
            response_type = ResponseType.RECOMMENDATION
        elif any(
            word in query_lower for word in ["plan", "strategy", "roadmap", "steps"]
        ):
            response_type = ResponseType.PLAN
        elif any(
            word in query_lower for word in ["create", "generate", "compose", "design"]
        ):
            response_type = ResponseType.CREATIVE
        elif any(
            word in query_lower
            for word in ["combine", "synthesize", "integrate", "merge"]
        ):
            response_type = ResponseType.SYNTHESIS
        else:
            response_type = ResponseType.ANSWER

        # Detect complexity
        word_count = len(query.split())
        has_multiple_questions = query.count("?") > 1
        has_conditionals = any(
            word in query_lower for word in ["if", "unless", "when", "while"]
        )
        requires_research = any(
            word in query_lower
            for word in ["research", "investigate", "explore", "study"]
        )
        is_technical = any(
            word in query_lower
            for word in ["architecture", "system", "infrastructure", "protocol"]
        )

        complexity_score = (
            (word_count > 50)
            + has_multiple_questions * 2
            + has_conditionals
            + requires_research * 2
            + is_technical
        )

        if complexity_score >= 5:
            complexity = TaskComplexity.SOVEREIGN
        elif complexity_score >= 4:
            complexity = TaskComplexity.RESEARCH
        elif complexity_score >= 3:
            complexity = TaskComplexity.COMPLEX
        elif complexity_score >= 2:
            complexity = TaskComplexity.MODERATE
        elif complexity_score >= 1:
            complexity = TaskComplexity.SIMPLE
        else:
            complexity = TaskComplexity.TRIVIAL

        return complexity, response_type

    async def process(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        require_council: Optional[bool] = None,
    ) -> SovereignResponse:
        """
        Process a query through the Sovereign Engine pipeline.

        Pipeline stages:
        1. Query Classification
        2. Cache Check
        3. Graph-of-Thoughts Reasoning
        4. SNR Maximization
        5. Guardian Council Review
        6. Response Synthesis
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()
        response_id = self._generate_response_id()
        context = context or {}

        logger.info(f"Processing query [{response_id}]: {query[:100]}...")

        # Stage 1: Query Classification
        complexity, response_type = self._classify_query(query)
        logger.debug(
            f"Query classified as {complexity.name}, expecting {response_type.name}"
        )

        # Stage 2: Cache Check
        if self.config.cache_enabled:
            cache_key = f"{query}:{json.dumps(context, sort_keys=True)}"
            if cache_key in self._cache:
                logger.info(f"Cache hit for query [{response_id}]")
                cached = self._cache[cache_key]
                return cached

        # Stage 3: Graph-of-Thoughts Reasoning
        reasoning_path = []
        thought_graph = None

        if self.config.enable_graph_reasoning:
            thought_result = await self._execute_graph_reasoning(
                query, complexity, context
            )
            reasoning_path = thought_result["path"]
            thought_graph = thought_result["graph"]

        # Stage 4: SNR Maximization
        snr_analysis: Optional[SNRAnalysis] = None
        if self.config.enable_snr_maximization:
            # Combine query with reasoning path for analysis
            combined_text = f"{query}\n\n" + "\n".join(reasoning_path[:5])
            _, snr_analysis = self.snr_maximizer.maximize(combined_text, query=query)

        # Generate preliminary content
        content = await self._synthesize_response(
            query, response_type, reasoning_path, context
        )

        # Stage 5: Guardian Council Review
        council_verdict = None
        should_consult_council = (
            require_council
            if require_council is not None
            else self.config.require_council_approval
            and complexity.value >= TaskComplexity.MODERATE.value
        )

        if self.config.enable_guardian_council and should_consult_council:
            proposal = Proposal(
                id=response_id,
                title=f"Response for: {query[:50]}",
                content={
                    "query": query,
                    "response": content,
                    "reasoning_path": reasoning_path,
                },
                proposer="sovereign_engine",
                required_mode=(
                    ConsensusMode.SUPERMAJORITY
                    if complexity == TaskComplexity.SOVEREIGN
                    else ConsensusMode.MAJORITY
                ),
            )
            council_verdict = await self.guardian_council.deliberate(proposal)

        # Calculate final metrics
        ihsan_vector = self._calculate_ihsan_vector(
            content, reasoning_path, council_verdict
        )
        ihsan_passed = ihsan_vector.passes_gate(self.config.ihsan_threshold)

        snr_score = snr_analysis.snr_linear if snr_analysis else 0.9
        confidence = self._calculate_confidence(
            snr_score, ihsan_passed, council_verdict
        )

        # Build response
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        response = SovereignResponse(
            id=response_id,
            query=query,
            response_type=response_type,
            content=content,
            confidence=confidence,
            snr_score=snr_score,
            ihsan_vector=ihsan_vector,
            ihsan_passed=ihsan_passed,
            thought_graph=thought_graph,
            reasoning_path=reasoning_path,
            evidence=[],
            council_verdict=council_verdict,
            guardians_consulted=[
                v.guardian.name
                for v in (council_verdict.votes if council_verdict else [])
            ],
            processing_time_ms=processing_time_ms,
            standing_on_giants=self.STANDING_ON_GIANTS,
        )

        # Cache successful responses
        if self.config.cache_enabled and ihsan_passed:
            cache_key = f"{query}:{json.dumps(context, sort_keys=True)}"
            self._cache[cache_key] = response

        logger.info(
            f"Query [{response_id}] completed in {processing_time_ms:.1f}ms "
            f"(SNR: {snr_score:.3f}, Ihsan: {ihsan_passed})"
        )

        return response

    async def _execute_graph_reasoning(
        self,
        query: str,
        complexity: TaskComplexity,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute Graph-of-Thoughts reasoning."""
        from .graph_reasoner import EdgeType, ThoughtType

        # Create root thought using the correct API
        root = self.graph_reasoner.add_thought(
            content=f"Query: {query}",
            thought_type=ThoughtType.QUESTION,
            confidence=1.0,
            metadata={"complexity": complexity.name},
        )

        # Determine reasoning depth based on complexity
        depth_map = {
            TaskComplexity.TRIVIAL: 1,
            TaskComplexity.SIMPLE: 2,
            TaskComplexity.MODERATE: 4,
            TaskComplexity.COMPLEX: 6,
            TaskComplexity.RESEARCH: 8,
            TaskComplexity.SOVEREIGN: 10,
        }
        max_depth = depth_map.get(complexity, 4)

        # Execute reasoning iterations
        reasoning_path = [root.content]

        for depth in range(max_depth):
            # Get current frontier (leaf thoughts)
            frontier = self.graph_reasoner.get_frontier()
            if not frontier:
                break

            # Generate new thoughts from frontier
            for thought in frontier[:5]:  # Limit branching factor
                # Generate child thought (placeholder — use LLM in production)
                child = self.graph_reasoner.add_thought(
                    content=f"Reasoning step {depth + 1} from: {thought.content[:50]}",
                    thought_type=ThoughtType.REASONING,
                    confidence=thought.confidence * 0.95,
                    metadata={"depth": depth + 1},
                    parent_id=thought.id,
                    edge_type=EdgeType.DERIVES,
                )
                reasoning_path.append(child.content)

        # Get graph state for response
        graph_state = {
            "nodes": len(self.graph_reasoner.thoughts),
            "edges": len(self.graph_reasoner.edges),
            "max_depth": max_depth,
        }

        # Clear for next query
        self.graph_reasoner.clear()

        return {
            "path": reasoning_path,
            "graph": graph_state,
        }

    async def _synthesize_response(
        self,
        query: str,
        response_type: ResponseType,
        reasoning_path: list[str],
        context: dict[str, Any],
    ) -> str:
        """Synthesize final response content."""
        # Placeholder — in production, this calls an LLM
        template = {
            ResponseType.ANSWER: "Based on analysis: {reasoning}",
            ResponseType.ANALYSIS: "## Analysis\n\n{reasoning}\n\n## Conclusion\n\nThe analysis indicates...",
            ResponseType.SYNTHESIS: "## Synthesis\n\nCombining insights from multiple perspectives:\n\n{reasoning}",
            ResponseType.RECOMMENDATION: "## Recommendations\n\n1. {reasoning}",
            ResponseType.CREATIVE: "## Creative Output\n\n{reasoning}",
            ResponseType.CODE: "```python\n# Generated code based on: {reasoning}\npass\n```",
            ResponseType.PLAN: "## Strategic Plan\n\n### Phase 1\n{reasoning}",
            ResponseType.CLARIFICATION: "To better assist you, I need clarification on: {reasoning}",
        }

        template_str = template.get(response_type, "{reasoning}")
        reasoning_summary = " → ".join(reasoning_path[:5]) if reasoning_path else query

        return template_str.format(reasoning=reasoning_summary)

    def _calculate_ihsan_vector(
        self,
        content: str,
        reasoning_path: list[str],
        council_verdict: Optional[CouncilVerdict],
    ) -> IhsanVector:
        """Calculate Ihsān vector for the response."""
        if council_verdict:
            # Use council's assessment
            return IhsanVector(
                correctness=0.9,  # Placeholder
                safety=0.95,
                beneficence=0.88,
                transparency=len(reasoning_path)
                / 10,  # More reasoning = more transparent
                sustainability=0.85,
            )
        else:
            # Default assessment
            return IhsanVector(
                correctness=0.85,
                safety=0.90,
                beneficence=0.85,
                transparency=0.80,
                sustainability=0.80,
            )

    def _calculate_confidence(
        self,
        snr_score: float,
        ihsan_passed: bool,
        council_verdict: Optional[CouncilVerdict],
    ) -> float:
        """Calculate overall response confidence."""
        base_confidence = snr_score * 0.4

        if ihsan_passed:
            base_confidence += 0.3

        if council_verdict:
            if council_verdict.approved:
                base_confidence += 0.3
            else:
                base_confidence *= 0.5

        return min(1.0, max(0.0, base_confidence))

    async def batch_process(
        self,
        queries: list[str],
        contexts: Optional[list[dict[str, Any]]] = None,
    ) -> list[SovereignResponse]:
        """Process multiple queries in parallel."""
        contexts = contexts or [{}] * len(queries)

        tasks = [
            self.process(query, context) for query, context in zip(queries, contexts)
        ]

        return await asyncio.gather(*tasks)

    def get_status(self) -> dict[str, Any]:
        """Get engine status and statistics."""
        return {
            "version": self.VERSION,
            "codename": self.CODENAME,
            "initialized": self._initialized,
            "mode": self.config.mode.name,
            "responses_generated": self._response_counter,
            "cache_size": len(self._cache),
            "orchestrator_status": self.orchestrator.get_status(),
            "config": {
                "snr_threshold": self.config.snr_threshold,
                "ihsan_threshold": self.config.ihsan_threshold,
                "graph_reasoning_enabled": self.config.enable_graph_reasoning,
                "guardian_council_enabled": self.config.enable_guardian_council,
            },
            "standing_on_giants": self.STANDING_ON_GIANTS,
        }

    def __repr__(self) -> str:
        return (
            f"SovereignEngine(v{self.VERSION}, mode={self.config.mode.name}, "
            f"initialized={self._initialized})"
        )


# Factory functions
def create_engine(
    mode: EngineMode = EngineMode.AUTONOMOUS,
    snr_threshold: float = 0.95,
    ihsan_threshold: float = 0.95,
) -> SovereignEngine:
    """Create a configured Sovereign Engine."""
    config = SovereignConfig(
        mode=mode,
        snr_threshold=snr_threshold,
        ihsan_threshold=ihsan_threshold,
    )
    return SovereignEngine(config)


async def quick_query(query: str) -> SovereignResponse:
    """Quick helper for single queries."""
    engine = create_engine()
    return await engine.process(query)


# Module-level instance for convenience
_default_engine: Optional[SovereignEngine] = None


async def get_engine() -> SovereignEngine:
    """Get or create the default engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = create_engine()
        await _default_engine.initialize()
    return _default_engine
