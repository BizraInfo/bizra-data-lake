"""
BIZRA Cognitive Service - Python Brain for Rust Body Integration

This module implements the CognitiveEngine service defined in proto/cognitive_engine.proto.
It serves as the "Brain" component in the Brain-Body-Soul architecture:

    Brain (Python) - Neural processing, LLM orchestration, thought synthesis
    Body (Rust)    - Execution, security, lifecycle management
    Soul (Ihsan)   - Ethical constraints, constitutional compliance

The service exposes both HTTP and optional gRPC endpoints for the Rust orchestrator
to invoke cognitive operations with full SAPE validation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field

# ============================================================================
# THINKING MODES (maps to proto ThinkingMode enum)
# ============================================================================


class ThinkingMode(str, Enum):
    """Cognitive thinking modes matching proto/cognitive_engine.proto"""

    FAST_PAT = "FAST_PAT"  # System 1: Fast, heuristic, intuitive
    DEEP_SAT = "DEEP_SAT"  # System 2: Slow, deliberate, analytical
    HYBRID_SYNERGY = "HYBRID_SYNERGY"  # Combined: PAT initiates, SAT validates
    REFLEXION = "REFLEXION"  # Self-improvement through iteration
    GRAPH_OF_THOUGHT = "GRAPH_OF_THOUGHT"  # Multi-dimensional synthesis


# ============================================================================
# ERROR CODES (maps to proto ErrorCode enum)
# ============================================================================


class ErrorCode(str, Enum):
    """Error codes for cognitive failures"""

    NONE = "NONE"
    LOW_SNR = "LOW_SNR"  # Signal-to-noise below threshold
    ETHICS_VIOLATION = "ETHICS_VIOLATION"  # Ihsan gate failed
    TIMEOUT = "TIMEOUT"  # Processing exceeded timeout
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"  # Context too large
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"  # LLM not responding
    INVALID_REQUEST = "INVALID_REQUEST"  # Malformed request
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"  # Too many failures


# ============================================================================
# DATA MODELS (maps to proto messages)
# ============================================================================


@dataclass
class Symbol:
    """Horn clause symbol component"""

    name: str
    arguments: List[str] = field(default_factory=list)
    negated: bool = False


@dataclass
class ActionPrimitive:
    """Action primitive for WisdomAtom"""

    action_type: str  # "execute", "delegate", "emit", "store"
    parameters: Dict[str, str] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class WisdomAtom:
    """The fundamental unit of verified knowledge"""

    id: str
    embedding: List[float] = field(default_factory=list)
    preconditions: List[Symbol] = field(default_factory=list)
    action: Optional[ActionPrimitive] = None
    postconditions: List[Symbol] = field(default_factory=list)
    provenance_signatures: List[bytes] = field(default_factory=list)
    success_rate: float = 0.0
    context_hash: bytes = b""
    created_at: int = 0
    source_agent: str = ""
    generation: int = 0


@dataclass
class ThoughtNode:
    """Individual thought node in the cognitive graph"""

    id: str
    content: str
    weight: float = 1.0
    connections: List[str] = field(default_factory=list)
    node_type: str = "inference"  # "premise", "inference", "conclusion"
    local_snr: float = 0.0


@dataclass
class CognitiveState:
    """Current cognitive state metrics"""

    total_thoughts_processed: int = 0
    average_snr: float = 0.0
    average_ihsan: float = 0.0
    active_agents: int = 0
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    pending_requests: int = 0
    circuit_breaker_closed: bool = True
    consecutive_failures: int = 0
    uptime_seconds: int = 0
    models: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class CognitiveRequest(BaseModel):
    """Cognitive request from Rust orchestrator"""

    agent_id: str
    task_id: str
    context_vector: List[float] = Field(default_factory=list)
    mode: ThinkingMode = ThinkingMode.HYBRID_SYNERGY
    prompt: str
    metadata: Dict[str, str] = Field(default_factory=dict)
    min_snr_threshold: float = 7.0  # Minimum SNR required (dB)
    min_ihsan_score: float = 0.99  # Minimum ethical threshold
    max_thinking_depth: int = 5  # Graph exploration depth limit
    timeout_ms: int = 30000  # Processing timeout


class CognitiveResponse(BaseModel):
    """Cognitive response to Rust orchestrator"""

    agent_id: str
    task_id: str
    synthesis: str
    confidence: float
    snr_score: float
    utility_score: float
    ihsan_score: float
    serialized_graph: str = ""
    thought_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: int
    model_used: str
    reasoning_steps: List[str] = Field(default_factory=list)
    success: bool
    error_message: str = ""
    error_code: ErrorCode = ErrorCode.NONE


class StateRequest(BaseModel):
    """State request"""

    agent_id: str = ""
    include_metrics: bool = True
    include_history: bool = False


class WisdomReceipt(BaseModel):
    """Receipt for wisdom injection"""

    wisdom_id: str
    accepted: bool
    rejection_reason: str = ""
    attestation_signature: str = ""  # hex-encoded
    timestamp: int


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self.failures = 0
        self.successes_in_half_open = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()

    async def is_available(self) -> bool:
        """Check if circuit allows requests"""
        async with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if (
                    self.last_failure_time
                    and (time.time() - self.last_failure_time) > self.recovery_timeout
                ):
                    self.state = "half_open"
                    self.successes_in_half_open = 0
                    return True
                return False
            else:  # half_open
                return True

    async def record_success(self):
        """Record successful operation"""
        async with self._lock:
            if self.state == "half_open":
                self.successes_in_half_open += 1
                if self.successes_in_half_open >= self.half_open_max:
                    self.state = "closed"
                    self.failures = 0
            elif self.state == "closed":
                self.failures = max(0, self.failures - 1)

    async def record_failure(self):
        """Record failed operation"""
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.state = "open"


# ============================================================================
# IHSAN SCORER
# ============================================================================


class IhsanScorer:
    """Ihsan (Excellence) scoring using 8-dimension constitution"""

    # Weights from constitution/ihsan_v1.yaml
    DIMENSIONS = {
        "correctness": 0.22,
        "safety": 0.22,
        "user_benefit": 0.14,
        "efficiency": 0.12,
        "auditability": 0.12,
        "anti_centralization": 0.08,
        "robustness": 0.06,
        "adl_fairness": 0.04,
    }

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold

    def score(self, dimensions: Dict[str, float]) -> Tuple[float, bool]:
        """
        Calculate Ihsan composite score.

        Returns:
            Tuple of (score, passed_threshold)
        """
        total = 0.0
        for dim, weight in self.DIMENSIONS.items():
            value = dimensions.get(dim, 0.0)
            total += value * weight

        return total, total >= self.threshold

    def evaluate_response(self, response: str, context: str = "") -> Dict[str, float]:
        """
        Evaluate a response across Ihsan dimensions.
        This is a heuristic evaluation - production should use LLM-based scoring.
        """
        # Heuristic scoring based on response characteristics
        dimensions = {
            "correctness": 0.95 if len(response) > 50 else 0.80,
            "safety": (
                1.0
                if not any(
                    kw in response.lower() for kw in ["hack", "exploit", "bypass"]
                )
                else 0.5
            ),
            "user_benefit": 0.90,
            "efficiency": 0.85 if len(response) < 2000 else 0.70,
            "auditability": 0.95,
            "anti_centralization": 0.90,
            "robustness": 0.85,
            "adl_fairness": 0.95,
        }
        return dimensions


# ============================================================================
# SNR CALCULATOR
# ============================================================================


class SNRCalculator:
    """Signal-to-Noise Ratio calculator for cognitive output quality"""

    # SNR tiers from BIZRA spec
    TIERS = {
        "T6": (8.5, float("inf")),  # Elite
        "T5": (8.0, 8.5),  # Expert
        "T4": (7.5, 8.0),  # Advanced
        "T3": (7.0, 7.5),  # Standard
        "T2": (6.5, 7.0),  # Basic
        "T1": (0.0, 6.5),  # Minimal
    }

    def calculate(
        self,
        response: str,
        prompt: str,
        context_vector: List[float],
    ) -> Tuple[float, str]:
        """
        Calculate SNR score and tier.

        Returns:
            Tuple of (snr_score, tier_name)
        """
        # Heuristic SNR calculation
        # In production, use embedding similarity and semantic analysis

        relevance = min(1.0, len(response) / max(len(prompt), 1))
        coherence = 0.9 if len(response) > 100 else 0.7
        information_density = min(
            1.0, len(set(response.split())) / max(len(response.split()), 1)
        )

        # SNR formula: 10 * log10(signal/noise) approximation
        signal = relevance * coherence * information_density
        noise = 0.1 + (1 - coherence) * 0.3

        snr = 7.0 + (signal / noise) * 1.5  # Scale to typical 7-9 range
        snr = max(6.0, min(9.0, snr))

        # Determine tier
        tier = "T1"
        for tier_name, (low, high) in self.TIERS.items():
            if low <= snr < high:
                tier = tier_name
                break

        return snr, tier


# ============================================================================
# COGNITIVE ENGINE SERVICE
# ============================================================================


class CognitiveEngine:
    """
    The Python Brain - Cognitive processing engine.

    This implements the CognitiveEngine service from proto/cognitive_engine.proto
    for integration with the Rust UnifiedOrchestrator.
    """

    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.ihsan_scorer = IhsanScorer(threshold=0.99)
        self.snr_calculator = SNRCalculator()
        self.wisdom_store: Dict[str, WisdomAtom] = {}
        self._state = CognitiveState()
        self._start_time = time.time()
        self._lock = asyncio.Lock()

    async def process_thought(self, request: CognitiveRequest) -> CognitiveResponse:
        """
        Process a thought through the full cognitive pipeline.

        This is the main entry point for Rust orchestrator calls.
        """
        start_time = time.time()

        # Check circuit breaker
        if not await self.circuit_breaker.is_available():
            return CognitiveResponse(
                agent_id=request.agent_id,
                task_id=request.task_id,
                synthesis="",
                confidence=0.0,
                snr_score=0.0,
                utility_score=0.0,
                ihsan_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used="",
                success=False,
                error_message="Circuit breaker is open",
                error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            )

        try:
            # Select thinking strategy based on mode
            if request.mode == ThinkingMode.FAST_PAT:
                result = await self._fast_pat_think(request)
            elif request.mode == ThinkingMode.DEEP_SAT:
                result = await self._deep_sat_think(request)
            elif request.mode == ThinkingMode.REFLEXION:
                result = await self._reflexion_think(request)
            elif request.mode == ThinkingMode.GRAPH_OF_THOUGHT:
                result = await self._graph_of_thought_think(request)
            else:  # HYBRID_SYNERGY (default)
                result = await self._hybrid_synergy_think(request)

            # Validate against thresholds
            if result.snr_score < request.min_snr_threshold:
                await self.circuit_breaker.record_failure()
                return CognitiveResponse(
                    agent_id=result.agent_id,
                    task_id=result.task_id,
                    synthesis=result.synthesis,
                    confidence=result.confidence,
                    snr_score=result.snr_score,
                    utility_score=result.utility_score,
                    ihsan_score=result.ihsan_score,
                    processing_time_ms=result.processing_time_ms,
                    model_used=result.model_used,
                    reasoning_steps=result.reasoning_steps,
                    success=False,
                    error_message=f"SNR {result.snr_score:.2f} below threshold {request.min_snr_threshold}",
                    error_code=ErrorCode.LOW_SNR,
                )

            if result.ihsan_score < request.min_ihsan_score:
                await self.circuit_breaker.record_failure()
                return CognitiveResponse(
                    agent_id=result.agent_id,
                    task_id=result.task_id,
                    synthesis=result.synthesis,
                    confidence=result.confidence,
                    snr_score=result.snr_score,
                    utility_score=result.utility_score,
                    ihsan_score=result.ihsan_score,
                    processing_time_ms=result.processing_time_ms,
                    model_used=result.model_used,
                    reasoning_steps=result.reasoning_steps,
                    success=False,
                    error_message=f"Ihsan {result.ihsan_score:.4f} below threshold {request.min_ihsan_score}",
                    error_code=ErrorCode.ETHICS_VIOLATION,
                )

            # Success
            await self.circuit_breaker.record_success()
            async with self._lock:
                self._state.total_thoughts_processed += 1
                self._state.average_snr = (
                    self._state.average_snr * (self._state.total_thoughts_processed - 1)
                    + result.snr_score
                ) / self._state.total_thoughts_processed
                self._state.average_ihsan = (
                    self._state.average_ihsan
                    * (self._state.total_thoughts_processed - 1)
                    + result.ihsan_score
                ) / self._state.total_thoughts_processed

            return result

        except asyncio.TimeoutError:
            await self.circuit_breaker.record_failure()
            return CognitiveResponse(
                agent_id=request.agent_id,
                task_id=request.task_id,
                synthesis="",
                confidence=0.0,
                snr_score=0.0,
                utility_score=0.0,
                ihsan_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used="",
                success=False,
                error_message=f"Timeout after {request.timeout_ms}ms",
                error_code=ErrorCode.TIMEOUT,
            )
        except Exception as e:
            await self.circuit_breaker.record_failure()
            return CognitiveResponse(
                agent_id=request.agent_id,
                task_id=request.task_id,
                synthesis="",
                confidence=0.0,
                snr_score=0.0,
                utility_score=0.0,
                ihsan_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used="",
                success=False,
                error_message=str(e),
                error_code=ErrorCode.MODEL_UNAVAILABLE,
            )

    async def _fast_pat_think(self, request: CognitiveRequest) -> CognitiveResponse:
        """System 1: Fast, heuristic, intuitive thinking"""
        start_time = time.time()

        # Fast path - direct LLM call with minimal processing
        synthesis = await self._call_llm(
            prompt=request.prompt,
            system="You are a fast, intuitive assistant. Provide quick, direct answers.",
            max_tokens=500,
        )

        snr, tier = self.snr_calculator.calculate(
            synthesis, request.prompt, request.context_vector
        )
        dimensions = self.ihsan_scorer.evaluate_response(synthesis, request.prompt)
        ihsan, _ = self.ihsan_scorer.score(dimensions)

        return CognitiveResponse(
            agent_id=request.agent_id,
            task_id=request.task_id,
            synthesis=synthesis,
            confidence=0.75,
            snr_score=snr,
            utility_score=0.80,
            ihsan_score=ihsan,
            processing_time_ms=int((time.time() - start_time) * 1000),
            model_used="fast_pat",
            reasoning_steps=["Direct intuitive response"],
            success=True,
        )

    async def _deep_sat_think(self, request: CognitiveRequest) -> CognitiveResponse:
        """System 2: Slow, deliberate, analytical thinking"""
        start_time = time.time()

        # Multi-step reasoning
        steps = []

        # Step 1: Analyze the problem
        analysis = await self._call_llm(
            prompt=f"Analyze this problem step by step:\n{request.prompt}",
            system="You are a deliberate analyst. Break down problems systematically.",
            max_tokens=1000,
        )
        steps.append(f"Analysis: {analysis[:200]}...")

        # Step 2: Generate solution
        synthesis = await self._call_llm(
            prompt=f"Based on this analysis:\n{analysis}\n\nProvide a comprehensive solution.",
            system="You are a thorough problem solver. Provide detailed, well-reasoned solutions.",
            max_tokens=1500,
        )
        steps.append("Solution synthesized")

        snr, tier = self.snr_calculator.calculate(
            synthesis, request.prompt, request.context_vector
        )
        dimensions = self.ihsan_scorer.evaluate_response(synthesis, request.prompt)
        ihsan, _ = self.ihsan_scorer.score(dimensions)

        return CognitiveResponse(
            agent_id=request.agent_id,
            task_id=request.task_id,
            synthesis=synthesis,
            confidence=0.90,
            snr_score=snr,
            utility_score=0.90,
            ihsan_score=ihsan,
            processing_time_ms=int((time.time() - start_time) * 1000),
            model_used="deep_sat",
            reasoning_steps=steps,
            success=True,
        )

    async def _hybrid_synergy_think(
        self, request: CognitiveRequest
    ) -> CognitiveResponse:
        """Combined: PAT initiates, SAT validates"""
        start_time = time.time()

        # PAT: Quick initial response
        initial = await self._call_llm(
            prompt=request.prompt,
            system="Provide a quick initial response.",
            max_tokens=500,
        )

        # SAT: Validate and refine
        refined = await self._call_llm(
            prompt=f"Review and improve this response:\n{initial}\n\nOriginal question: {request.prompt}",
            system="You are a critical reviewer. Validate accuracy and completeness, then improve.",
            max_tokens=1000,
        )

        snr, tier = self.snr_calculator.calculate(
            refined, request.prompt, request.context_vector
        )
        dimensions = self.ihsan_scorer.evaluate_response(refined, request.prompt)
        ihsan, _ = self.ihsan_scorer.score(dimensions)

        return CognitiveResponse(
            agent_id=request.agent_id,
            task_id=request.task_id,
            synthesis=refined,
            confidence=0.85,
            snr_score=snr,
            utility_score=0.85,
            ihsan_score=ihsan,
            processing_time_ms=int((time.time() - start_time) * 1000),
            model_used="hybrid_synergy",
            reasoning_steps=["PAT: Initial response", "SAT: Validation and refinement"],
            success=True,
        )

    async def _reflexion_think(self, request: CognitiveRequest) -> CognitiveResponse:
        """Self-improvement through iteration"""
        start_time = time.time()
        steps = []

        # Initial attempt
        current = await self._call_llm(
            prompt=request.prompt,
            system="Provide your best response.",
            max_tokens=1000,
        )
        steps.append("Initial response generated")

        # Reflexion loop (up to max_thinking_depth iterations)
        for i in range(min(request.max_thinking_depth, 3)):
            critique = await self._call_llm(
                prompt=f"Critique this response and identify improvements:\n{current}",
                system="You are a self-critical analyst. Find weaknesses and suggest improvements.",
                max_tokens=500,
            )

            if "good" in critique.lower() and "no major" in critique.lower():
                steps.append(f"Iteration {i+1}: Response validated")
                break

            current = await self._call_llm(
                prompt=f"Improve this response based on the critique:\nResponse: {current}\nCritique: {critique}",
                system="Incorporate the feedback to produce an improved response.",
                max_tokens=1000,
            )
            steps.append(f"Iteration {i+1}: Response improved")

        snr, tier = self.snr_calculator.calculate(
            current, request.prompt, request.context_vector
        )
        dimensions = self.ihsan_scorer.evaluate_response(current, request.prompt)
        ihsan, _ = self.ihsan_scorer.score(dimensions)

        return CognitiveResponse(
            agent_id=request.agent_id,
            task_id=request.task_id,
            synthesis=current,
            confidence=0.92,
            snr_score=snr,
            utility_score=0.88,
            ihsan_score=ihsan,
            processing_time_ms=int((time.time() - start_time) * 1000),
            model_used="reflexion",
            reasoning_steps=steps,
            success=True,
        )

    async def _graph_of_thought_think(
        self, request: CognitiveRequest
    ) -> CognitiveResponse:
        """Multi-dimensional synthesis using thought graph"""
        start_time = time.time()
        thought_nodes: List[ThoughtNode] = []

        # Generate multiple perspectives
        perspectives = ["analytical", "creative", "critical", "practical"]
        perspective_responses = []

        for i, perspective in enumerate(perspectives):
            response = await self._call_llm(
                prompt=f"From a {perspective} perspective, respond to:\n{request.prompt}",
                system=f"You approach problems from a {perspective} viewpoint.",
                max_tokens=500,
            )

            node = ThoughtNode(
                id=f"node_{i}",
                content=response[:500],
                weight=1.0 / len(perspectives),
                connections=[f"node_{j}" for j in range(len(perspectives)) if j != i],
                node_type="perspective",
                local_snr=7.5,
            )
            thought_nodes.append(node)
            perspective_responses.append(response)

        # Synthesize perspectives
        synthesis = await self._call_llm(
            prompt=f"""Synthesize these perspectives into a unified response:

Analytical: {perspective_responses[0][:300]}
Creative: {perspective_responses[1][:300]}
Critical: {perspective_responses[2][:300]}
Practical: {perspective_responses[3][:300]}

Original question: {request.prompt}""",
            system="You are a master synthesizer. Combine multiple viewpoints into a coherent whole.",
            max_tokens=1500,
        )

        # Add synthesis node
        thought_nodes.append(
            ThoughtNode(
                id="synthesis",
                content=synthesis[:500],
                weight=1.0,
                connections=[f"node_{i}" for i in range(len(perspectives))],
                node_type="conclusion",
                local_snr=8.0,
            )
        )

        snr, tier = self.snr_calculator.calculate(
            synthesis, request.prompt, request.context_vector
        )
        dimensions = self.ihsan_scorer.evaluate_response(synthesis, request.prompt)
        ihsan, _ = self.ihsan_scorer.score(dimensions)

        # Serialize thought graph
        graph_data = {
            "nodes": [asdict(n) for n in thought_nodes],
            "synthesis_tier": tier,
        }

        return CognitiveResponse(
            agent_id=request.agent_id,
            task_id=request.task_id,
            synthesis=synthesis,
            confidence=0.95,
            snr_score=snr,
            utility_score=0.92,
            ihsan_score=ihsan,
            serialized_graph=json.dumps(graph_data),
            thought_nodes=[asdict(n) for n in thought_nodes],
            processing_time_ms=int((time.time() - start_time) * 1000),
            model_used="graph_of_thought",
            reasoning_steps=[
                "Generated analytical perspective",
                "Generated creative perspective",
                "Generated critical perspective",
                "Generated practical perspective",
                "Synthesized all perspectives",
            ],
            success=True,
        )

    async def _call_llm(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1000,
    ) -> str:
        """Call the LLM backend (Ollama or LM Studio)"""
        try:
            from core.llm import chat_with_routing
            from core.model_family import load_model_family

            mf = load_model_family()
            completion, _ = await chat_with_routing(
                model_family=mf,
                slot="primary_reasoning",
                system_prompt=system,
                user_prompt=prompt,
                max_attempts=2,
            )
            return completion.text
        except Exception as e:
            # Fallback for when LLM is not available
            return (
                f"[Cognitive processing of: {prompt[:100]}...] (LLM unavailable: {e})"
            )

    async def get_cognitive_state(self, request: StateRequest) -> CognitiveState:
        """Get current cognitive state metrics"""
        async with self._lock:
            self._state.uptime_seconds = int(time.time() - self._start_time)
            self._state.circuit_breaker_closed = self.circuit_breaker.state == "closed"
            self._state.consecutive_failures = self.circuit_breaker.failures
            return self._state

    async def inject_wisdom(self, atom: WisdomAtom) -> WisdomReceipt:
        """Inject a wisdom atom into the knowledge store"""
        # Validate the wisdom atom
        if not atom.id:
            return WisdomReceipt(
                wisdom_id=atom.id,
                accepted=False,
                rejection_reason="Invalid wisdom ID",
                timestamp=int(time.time()),
            )

        # Store the wisdom
        self.wisdom_store[atom.id] = atom

        # Generate attestation signature
        signature = hashlib.sha256(
            f"{atom.id}:{atom.source_agent}:{atom.created_at}".encode()
        ).hexdigest()

        return WisdomReceipt(
            wisdom_id=atom.id,
            accepted=True,
            attestation_signature=signature,
            timestamp=int(time.time()),
        )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

cognitive_app = FastAPI(
    title="BIZRA Cognitive Service",
    version="1.0.0",
    description="Python Brain for Rust Body Integration - SAPE Ultimate Integration",
)

# Global engine instance
_engine: Optional[CognitiveEngine] = None


def get_engine() -> CognitiveEngine:
    """Get or create the cognitive engine singleton"""
    global _engine
    if _engine is None:
        _engine = CognitiveEngine()
    return _engine


def verify_token(authorization: Optional[str] = Header(default=None)) -> str:
    """Verify API token"""
    expected = os.getenv("BIZRA_API_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="BIZRA_API_TOKEN not set")

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()

    if token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return token


@cognitive_app.post("/v1/cognitive/process", response_model=CognitiveResponse)
async def process_thought(
    request: CognitiveRequest,
    _: str = Depends(verify_token),
):
    """
    Process a thought through the cognitive pipeline.

    This is the main Brain-Body interface endpoint for the Rust orchestrator.
    """
    engine = get_engine()
    return await engine.process_thought(request)


@cognitive_app.get("/v1/cognitive/state")
async def get_state(
    agent_id: str = "",
    include_metrics: bool = True,
    include_history: bool = False,
    _: str = Depends(verify_token),
):
    """Get current cognitive state metrics"""
    engine = get_engine()
    request = StateRequest(
        agent_id=agent_id,
        include_metrics=include_metrics,
        include_history=include_history,
    )
    state = await engine.get_cognitive_state(request)
    return asdict(state)


@cognitive_app.post("/v1/cognitive/wisdom", response_model=WisdomReceipt)
async def inject_wisdom(
    atom_data: Dict[str, Any],
    _: str = Depends(verify_token),
):
    """Inject a wisdom atom into the knowledge store"""
    engine = get_engine()

    # Convert dict to WisdomAtom
    atom = WisdomAtom(
        id=atom_data.get("id", str(uuid.uuid4())),
        embedding=atom_data.get("embedding", []),
        success_rate=atom_data.get("success_rate", 0.0),
        created_at=atom_data.get("created_at", int(time.time())),
        source_agent=atom_data.get("source_agent", "unknown"),
        generation=atom_data.get("generation", 0),
    )

    return await engine.inject_wisdom(atom)


@cognitive_app.get("/health")
async def health():
    """Health check endpoint"""
    engine = get_engine()
    state = await engine.get_cognitive_state(StateRequest())

    return {
        "status": "healthy",
        "service": "cognitive-engine",
        "uptime_seconds": state.uptime_seconds,
        "thoughts_processed": state.total_thoughts_processed,
        "average_snr": round(state.average_snr, 2),
        "average_ihsan": round(state.average_ihsan, 4),
        "circuit_breaker": "closed" if state.circuit_breaker_closed else "open",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    """Run the cognitive service standalone"""
    import uvicorn

    host = os.getenv("BIZRA_COGNITIVE_HOST", "127.0.0.1")
    port = int(os.getenv("BIZRA_COGNITIVE_PORT", "8020"))

    print(f"Starting BIZRA Cognitive Service on {host}:{port}")
    uvicorn.run(cognitive_app, host=host, port=port)


if __name__ == "__main__":
    main()
