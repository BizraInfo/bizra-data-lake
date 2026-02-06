"""
BIZRA LOCAL-FIRST MULTI-MODEL CONFIGURATION
===============================================================================

Comprehensive configuration for local-first inference with multiple specialized
models. Implements the Bicameral Mind architecture (Cold Core + Warm Surface)
for robust, high-quality reasoning with local-only guarantees.

Architecture:
- Model Registry: Configure multiple local models for different tasks
- Model Router: Intelligent routing based on task type and complexity
- Bicameral Orchestrator: Cold Core (reasoning) + Warm Surface (creativity/verification)
- Health Monitor: Periodic health checks with automatic failover

Bicameral Mind (Jaynes, 1976):
- Cold Core: Logical, deterministic reasoning (DeepSeek-R1, Qwen-Coder)
- Warm Surface: Creative, nuanced generation (Mistral-Nemo)
- Generate-Verify Loop (Karpathy, 2024): Cold generates, Warm verifies

Local-First Guarantees:
- No external API calls by default
- Fallback chain: Local -> Pool -> Cloud (with explicit consent)
- Privacy preservation (no data leaves node without consent)

Standing on Giants:
- Jaynes (1976): Bicameral Mind - dual processing architecture
- Karpathy (2024): Generate-verify loops for quality
- DeepSeek (2025): R1 reasoning model architecture
- Anthropic: Constitutional AI constraints
- Shannon: SNR maximization, noise minimization

Created: 2026-02-05 | BIZRA Sovereignty
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, TypeVar
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class ModelRole(str, Enum):
    """Model specialization roles."""
    REASONING = "reasoning"      # Chain-of-thought, logic, math
    CODING = "coding"            # Code generation, debugging
    CREATIVE = "creative"        # Writing, conversation, nuance
    FAST = "fast"                # Quick tasks, classification
    EMBEDDING = "embedding"      # Vector embeddings
    VISION = "vision"            # Image understanding
    AGENTIC = "agentic"          # Tool use, planning


class TaskType(str, Enum):
    """Task classification for routing."""
    REASONING = "reasoning"          # Complex analysis, proof
    CODING = "coding"                # Code generation/review
    CONVERSATION = "conversation"    # Natural dialogue
    SUMMARIZATION = "summarization"  # Text condensation
    CLASSIFICATION = "classification"  # Categorization
    EXTRACTION = "extraction"        # Information extraction
    CREATIVE_WRITING = "creative_writing"
    MATH = "math"                    # Mathematical computation
    PLANNING = "planning"            # Multi-step planning


class FallbackLevel(str, Enum):
    """Fallback chain levels (sovereignty-preserving)."""
    LOCAL_ONLY = "local_only"        # Never leave node
    LOCAL_POOL = "local_pool"        # Allow federated pool
    LOCAL_POOL_CLOUD = "local_pool_cloud"  # Allow cloud (explicit consent)


class HealthStatus(str, Enum):
    """Model health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    endpoint: str
    model_id: str
    role: ModelRole

    # Model characteristics
    context_window: int = 4096
    dimensions: int = 0  # For embedding models only

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048

    # Capabilities
    strengths: List[str] = field(default_factory=list)
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False

    # Performance
    typical_latency_ms: float = 0.0
    vram_gb: float = 0.0

    # Priority (higher = preferred when multiple options)
    priority: int = 50


@dataclass
class HealthReport:
    """Health status report for a model."""
    model_id: str
    status: HealthStatus
    latency_ms: float
    last_check: float  # timestamp
    consecutive_failures: int = 0
    error_message: str = ""

    @property
    def is_available(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


@dataclass
class RoutingDecision:
    """Result of model routing decision."""
    model_name: str
    model_config: ModelConfig
    confidence: float
    reason: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class BicameralResult:
    """Result from bicameral reasoning."""
    cold_response: str           # Primary reasoning output
    warm_critique: str           # Verification/critique
    synthesis: str               # Final synthesized answer
    cold_model: str
    warm_model: str
    consensus_reached: bool
    iterations: int
    total_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MODEL REGISTRY
# =============================================================================


LOCAL_MODELS: Dict[str, ModelConfig] = {
    # =========================================================================
    # REASONING MODELS (Cold Core - Logic)
    # =========================================================================
    "deepseek-r1": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="deepseek-r1-distill-qwen-32b",
        role=ModelRole.REASONING,
        context_window=32768,
        temperature=0.1,  # Low for deterministic reasoning
        top_p=0.9,
        max_tokens=4096,
        strengths=["chain-of-thought", "math", "code", "logic", "proof"],
        supports_streaming=True,
        supports_function_calling=True,
        typical_latency_ms=2000,
        vram_gb=20.0,
        priority=100,
    ),

    "qwq-reasoning": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="qwq-32b-preview",
        role=ModelRole.REASONING,
        context_window=32768,
        temperature=0.1,
        top_p=0.9,
        max_tokens=4096,
        strengths=["chain-of-thought", "analysis", "research", "synthesis"],
        supports_streaming=True,
        typical_latency_ms=2500,
        vram_gb=20.0,
        priority=90,
    ),

    # =========================================================================
    # CODING MODELS
    # =========================================================================
    "qwen-coder": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="qwen2.5-coder-32b-instruct",
        role=ModelRole.CODING,
        context_window=32768,
        temperature=0.2,  # Slightly higher for creativity in code
        top_p=0.9,
        max_tokens=4096,
        strengths=["code-generation", "debugging", "refactoring", "review"],
        supports_streaming=True,
        supports_function_calling=True,
        typical_latency_ms=1800,
        vram_gb=20.0,
        priority=100,
    ),

    "deepseek-coder": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="deepseek-coder-v2-lite-instruct",
        role=ModelRole.CODING,
        context_window=16384,
        temperature=0.2,
        top_p=0.9,
        max_tokens=4096,
        strengths=["code-generation", "code-completion", "debugging"],
        supports_streaming=True,
        typical_latency_ms=800,
        vram_gb=8.0,
        priority=80,
    ),

    # =========================================================================
    # NUANCE MODELS (Warm Surface - Creativity)
    # =========================================================================
    "mistral-nemo": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="mistral-nemo-instruct-2407",
        role=ModelRole.CREATIVE,
        context_window=128000,
        temperature=0.7,  # Higher for creativity
        top_p=0.95,
        max_tokens=4096,
        strengths=["conversation", "writing", "nuance", "tone", "critique"],
        supports_streaming=True,
        typical_latency_ms=600,
        vram_gb=8.0,
        priority=100,
    ),

    "llama-creative": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="llama-3.1-8b-instruct",
        role=ModelRole.CREATIVE,
        context_window=131072,
        temperature=0.8,
        top_p=0.95,
        max_tokens=4096,
        strengths=["creative-writing", "storytelling", "brainstorming"],
        supports_streaming=True,
        typical_latency_ms=500,
        vram_gb=6.0,
        priority=80,
    ),

    # =========================================================================
    # FAST MODELS (Edge - Quick responses)
    # =========================================================================
    "phi-4": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="phi-4",
        role=ModelRole.FAST,
        context_window=16384,
        temperature=0.3,
        top_p=0.9,
        max_tokens=2048,
        strengths=["quick-tasks", "classification", "summarization", "qa"],
        supports_streaming=True,
        typical_latency_ms=200,
        vram_gb=4.0,
        priority=100,
    ),

    "qwen-nano": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="qwen2.5-1.5b-instruct",
        role=ModelRole.FAST,
        context_window=32768,
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
        strengths=["ultra-fast", "simple-qa", "formatting"],
        supports_streaming=True,
        typical_latency_ms=100,
        vram_gb=2.0,
        priority=90,
    ),

    "gemma-fast": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="gemma-2-2b-instruct",
        role=ModelRole.FAST,
        context_window=8192,
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
        strengths=["fast", "concise", "factual"],
        supports_streaming=True,
        typical_latency_ms=150,
        vram_gb=2.5,
        priority=70,
    ),

    # =========================================================================
    # EMBEDDING MODELS
    # =========================================================================
    "nomic-embed": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="nomic-embed-text-v1.5",
        role=ModelRole.EMBEDDING,
        dimensions=768,
        context_window=8192,
        supports_streaming=False,
        typical_latency_ms=50,
        vram_gb=0.5,
        priority=100,
    ),

    "bge-embed": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="bge-m3",
        role=ModelRole.EMBEDDING,
        dimensions=1024,
        context_window=8192,
        supports_streaming=False,
        typical_latency_ms=60,
        vram_gb=0.8,
        priority=90,
    ),

    # =========================================================================
    # VISION MODELS
    # =========================================================================
    "qwen-vl": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="qwen2-vl-7b-instruct",
        role=ModelRole.VISION,
        context_window=32768,
        temperature=0.3,
        top_p=0.9,
        max_tokens=2048,
        strengths=["image-understanding", "ocr", "visual-qa"],
        supports_streaming=True,
        supports_vision=True,
        typical_latency_ms=1500,
        vram_gb=8.0,
        priority=100,
    ),

    "llava": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="llava-v1.6-mistral-7b",
        role=ModelRole.VISION,
        context_window=4096,
        temperature=0.3,
        top_p=0.9,
        max_tokens=2048,
        strengths=["image-description", "visual-qa"],
        supports_streaming=True,
        supports_vision=True,
        typical_latency_ms=1200,
        vram_gb=7.0,
        priority=80,
    ),

    # =========================================================================
    # AGENTIC MODELS
    # =========================================================================
    "agentflow": ModelConfig(
        endpoint="http://192.168.56.1:1234/v1",
        model_id="agentflow-planner-7b",
        role=ModelRole.AGENTIC,
        context_window=32768,
        temperature=0.2,
        top_p=0.9,
        max_tokens=4096,
        strengths=["planning", "tool-use", "orchestration", "decomposition"],
        supports_streaming=True,
        supports_function_calling=True,
        typical_latency_ms=800,
        vram_gb=6.0,
        priority=100,
    ),
}


# Task type to model role mapping
TASK_TO_ROLE: Dict[TaskType, List[ModelRole]] = {
    TaskType.REASONING: [ModelRole.REASONING, ModelRole.CODING],
    TaskType.CODING: [ModelRole.CODING, ModelRole.REASONING],
    TaskType.CONVERSATION: [ModelRole.CREATIVE, ModelRole.FAST],
    TaskType.SUMMARIZATION: [ModelRole.FAST, ModelRole.CREATIVE],
    TaskType.CLASSIFICATION: [ModelRole.FAST],
    TaskType.EXTRACTION: [ModelRole.FAST, ModelRole.REASONING],
    TaskType.CREATIVE_WRITING: [ModelRole.CREATIVE],
    TaskType.MATH: [ModelRole.REASONING],
    TaskType.PLANNING: [ModelRole.AGENTIC, ModelRole.REASONING],
}


# =============================================================================
# MODEL ROUTER
# =============================================================================


class ModelRouter:
    """
    Intelligent model routing based on task type and complexity.

    Routes requests to the optimal model based on:
    - Task type (reasoning, coding, creative, etc.)
    - Token complexity (simple tasks -> fast models)
    - Resource availability (healthy models only)

    Standing on Giants:
    - Shazeer (MoE): Sparse routing to specialists
    - Adaptive Computation: Right model for right task
    """

    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        health_monitor: Optional["HealthMonitor"] = None,
    ):
        self.models = models or LOCAL_MODELS
        self.health_monitor = health_monitor

    def route_by_task(self, task_type: TaskType) -> RoutingDecision:
        """
        Select best model for a given task type.

        Args:
            task_type: Type of task to route

        Returns:
            RoutingDecision with selected model and reasoning
        """
        preferred_roles = TASK_TO_ROLE.get(task_type, [ModelRole.FAST])
        candidates = self._get_candidates_by_roles(preferred_roles)

        if not candidates:
            # Fallback to any available model
            candidates = list(self.models.items())

        # Filter by health if monitor available
        if self.health_monitor:
            candidates = [
                (name, cfg) for name, cfg in candidates
                if self.health_monitor.is_available(name)
            ]

        if not candidates:
            raise RuntimeError(f"No models available for task type: {task_type}")

        # Sort by priority (descending)
        candidates.sort(key=lambda x: x[1].priority, reverse=True)

        selected_name, selected_config = candidates[0]
        alternatives = [name for name, _ in candidates[1:3]]

        return RoutingDecision(
            model_name=selected_name,
            model_config=selected_config,
            confidence=0.9 if selected_config.role in preferred_roles else 0.7,
            reason=f"Selected {selected_name} for {task_type.value} (role: {selected_config.role.value})",
            alternatives=alternatives,
        )

    def route_by_complexity(
        self,
        tokens: int,
        requires_reasoning: bool = False,
        requires_code: bool = False,
    ) -> RoutingDecision:
        """
        Select model based on task complexity (token count + requirements).

        Tiered selection:
        - Simple (< 500 tokens): Fast models
        - Medium (500-2000 tokens): Standard models
        - Complex (> 2000 tokens): Reasoning/coding models

        Args:
            tokens: Estimated total tokens
            requires_reasoning: Task needs chain-of-thought
            requires_code: Task involves code generation

        Returns:
            RoutingDecision with selected model
        """
        # Determine preferred roles based on complexity
        if requires_reasoning:
            preferred_roles = [ModelRole.REASONING, ModelRole.CODING]
        elif requires_code:
            preferred_roles = [ModelRole.CODING, ModelRole.REASONING]
        elif tokens < 500:
            preferred_roles = [ModelRole.FAST]
        elif tokens < 2000:
            preferred_roles = [ModelRole.CREATIVE, ModelRole.FAST]
        else:
            preferred_roles = [ModelRole.REASONING, ModelRole.CREATIVE]

        candidates = self._get_candidates_by_roles(preferred_roles)

        # Filter by context window
        candidates = [
            (name, cfg) for name, cfg in candidates
            if cfg.context_window >= tokens
        ]

        if not candidates:
            # Fallback to any model with sufficient context
            candidates = [
                (name, cfg) for name, cfg in self.models.items()
                if cfg.context_window >= tokens
            ]

        if not candidates:
            raise RuntimeError(f"No models with sufficient context ({tokens} tokens)")

        # Sort by priority
        candidates.sort(key=lambda x: x[1].priority, reverse=True)

        selected_name, selected_config = candidates[0]

        complexity_tier = "simple" if tokens < 500 else "medium" if tokens < 2000 else "complex"

        return RoutingDecision(
            model_name=selected_name,
            model_config=selected_config,
            confidence=0.85,
            reason=f"Selected {selected_name} for {complexity_tier} task ({tokens} tokens)",
            alternatives=[name for name, _ in candidates[1:3]],
        )

    def route_bicameral(self, query: str) -> Tuple[RoutingDecision, RoutingDecision]:
        """
        Route to bicameral pair: Cold Core (reasoning) + Warm Surface (verification).

        Implements Jaynes' bicameral mind:
        - Cold Core: Logical, deterministic (low temperature)
        - Warm Surface: Creative, critical (higher temperature)

        Args:
            query: Input query to analyze

        Returns:
            Tuple of (cold_decision, warm_decision)
        """
        # Cold Core: Select best reasoning model
        cold_candidates = self._get_candidates_by_roles([ModelRole.REASONING, ModelRole.CODING])
        if not cold_candidates:
            cold_candidates = list(self.models.items())
        cold_candidates.sort(key=lambda x: x[1].priority, reverse=True)
        cold_name, cold_config = cold_candidates[0]

        cold_decision = RoutingDecision(
            model_name=cold_name,
            model_config=cold_config,
            confidence=0.95,
            reason=f"Cold Core: {cold_name} (deterministic reasoning)",
            alternatives=[name for name, _ in cold_candidates[1:2]],
        )

        # Warm Surface: Select best creative/verification model
        warm_candidates = self._get_candidates_by_roles([ModelRole.CREATIVE])
        if not warm_candidates:
            # Fallback: use a different model than cold
            warm_candidates = [
                (name, cfg) for name, cfg in self.models.items()
                if name != cold_name
            ]
        warm_candidates.sort(key=lambda x: x[1].priority, reverse=True)
        warm_name, warm_config = warm_candidates[0] if warm_candidates else (cold_name, cold_config)

        warm_decision = RoutingDecision(
            model_name=warm_name,
            model_config=warm_config,
            confidence=0.90,
            reason=f"Warm Surface: {warm_name} (creative verification)",
            alternatives=[name for name, _ in warm_candidates[1:2]],
        )

        return cold_decision, warm_decision

    def _get_candidates_by_roles(
        self,
        roles: List[ModelRole],
    ) -> List[Tuple[str, ModelConfig]]:
        """Get models matching any of the given roles."""
        return [
            (name, cfg) for name, cfg in self.models.items()
            if cfg.role in roles
        ]

    def get_embedding_model(self) -> RoutingDecision:
        """Get best available embedding model."""
        candidates = self._get_candidates_by_roles([ModelRole.EMBEDDING])
        if not candidates:
            raise RuntimeError("No embedding models configured")

        candidates.sort(key=lambda x: x[1].priority, reverse=True)
        name, config = candidates[0]

        return RoutingDecision(
            model_name=name,
            model_config=config,
            confidence=1.0,
            reason=f"Embedding model: {name} ({config.dimensions} dimensions)",
        )

    def get_vision_model(self) -> RoutingDecision:
        """Get best available vision model."""
        candidates = self._get_candidates_by_roles([ModelRole.VISION])
        if not candidates:
            raise RuntimeError("No vision models configured")

        candidates.sort(key=lambda x: x[1].priority, reverse=True)
        name, config = candidates[0]

        return RoutingDecision(
            model_name=name,
            model_config=config,
            confidence=1.0,
            reason=f"Vision model: {name}",
        )


# =============================================================================
# HEALTH MONITOR
# =============================================================================


class HealthMonitor:
    """
    Monitors model health with periodic checks and automatic failover.

    Features:
    - Async health probes to all configured endpoints
    - Latency tracking for performance monitoring
    - Automatic failover to alternatives on failure
    - Configurable check intervals and failure thresholds
    """

    DEFAULT_CHECK_INTERVAL_S = 60.0
    DEFAULT_TIMEOUT_S = 5.0
    DEFAULT_FAILURE_THRESHOLD = 3

    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    ):
        self.models = models or LOCAL_MODELS
        self.check_interval_s = check_interval_s
        self.timeout_s = timeout_s
        self.failure_threshold = failure_threshold

        self._health_reports: Dict[str, HealthReport] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the health monitor background task."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def check_all(self) -> Dict[str, HealthReport]:
        """
        Check health of all configured models.

        Returns:
            Dict mapping model names to their HealthReport
        """
        tasks = [
            self._check_model(name, config)
            for name, config in self.models.items()
        ]

        reports = await asyncio.gather(*tasks, return_exceptions=True)

        for report in reports:
            if isinstance(report, HealthReport):
                self._health_reports[report.model_id] = report

        return dict(self._health_reports)

    async def check_model(self, model_name: str) -> HealthReport:
        """Check health of a specific model."""
        if model_name not in self.models:
            return HealthReport(
                model_id=model_name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                last_check=time.time(),
                error_message="Model not found in registry",
            )

        return await self._check_model(model_name, self.models[model_name])

    def get_report(self, model_name: str) -> Optional[HealthReport]:
        """Get the latest health report for a model."""
        return self._health_reports.get(model_name)

    def is_available(self, model_name: str) -> bool:
        """Check if a model is available for use."""
        report = self._health_reports.get(model_name)
        return report is not None and report.is_available

    def get_healthy_models(self) -> List[str]:
        """Get list of all healthy models."""
        return [
            name for name, report in self._health_reports.items()
            if report.is_available
        ]

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.check_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)  # Brief pause on error

    async def _check_model(
        self,
        model_name: str,
        config: ModelConfig,
    ) -> HealthReport:
        """Perform health check on a single model."""
        start = time.perf_counter()

        try:
            import httpx

            # Extract base URL from endpoint
            base_url = config.endpoint.rstrip("/")

            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                # Try /models endpoint (OpenAI compatible)
                resp = await client.get(f"{base_url}/models")
                latency_ms = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    # Update consecutive failures
                    prev_report = self._health_reports.get(model_name)

                    return HealthReport(
                        model_id=model_name,
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency_ms,
                        last_check=time.time(),
                        consecutive_failures=0,
                    )
                else:
                    return self._handle_failure(
                        model_name,
                        f"HTTP {resp.status_code}",
                        (time.perf_counter() - start) * 1000,
                    )

        except httpx.TimeoutException:
            return self._handle_failure(
                model_name,
                "Connection timeout",
                (time.perf_counter() - start) * 1000,
            )
        except httpx.ConnectError as e:
            return self._handle_failure(
                model_name,
                f"Connection error: {str(e)[:50]}",
                (time.perf_counter() - start) * 1000,
            )
        except ImportError:
            return HealthReport(
                model_id=model_name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                last_check=time.time(),
                error_message="httpx not installed",
            )
        except Exception as e:
            return self._handle_failure(
                model_name,
                str(e)[:100],
                (time.perf_counter() - start) * 1000,
            )

    def _handle_failure(
        self,
        model_name: str,
        error_message: str,
        latency_ms: float,
    ) -> HealthReport:
        """Handle a failed health check."""
        prev_report = self._health_reports.get(model_name)
        consecutive_failures = (prev_report.consecutive_failures + 1) if prev_report else 1

        status = (
            HealthStatus.UNHEALTHY
            if consecutive_failures >= self.failure_threshold
            else HealthStatus.DEGRADED
        )

        return HealthReport(
            model_id=model_name,
            status=status,
            latency_ms=latency_ms,
            last_check=time.time(),
            consecutive_failures=consecutive_failures,
            error_message=error_message,
        )


# =============================================================================
# BICAMERAL ORCHESTRATOR
# =============================================================================


class BicameralOrchestrator:
    """
    Coordinates Cold Core (reasoning) + Warm Surface (verification) for
    high-quality outputs through generate-verify loops.

    Architecture (Jaynes, 1976):
    - Cold Core: Logical, deterministic reasoning (low temperature)
    - Warm Surface: Creative verification and critique (higher temperature)

    Process (Karpathy, 2024 - Generate-Verify):
    1. Cold Core generates candidate responses
    2. Warm Surface verifies/critiques the response
    3. If consensus: return synthesis
    4. If disagreement: iterate with feedback

    Constitutional Constraints (Anthropic):
    - Outputs must be helpful, harmless, honest
    - SNR threshold >= 0.85 (Ihsan constraint)
    """

    DEFAULT_MAX_ITERATIONS = 3
    DEFAULT_CONSENSUS_THRESHOLD = 0.8

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        health_monitor: Optional[HealthMonitor] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    ):
        self.router = router or ModelRouter()
        self.health_monitor = health_monitor
        self.max_iterations = max_iterations
        self.consensus_threshold = consensus_threshold

        # HTTP client for API calls
        self._client: Optional[Any] = None

    async def _ensure_client(self) -> Any:
        """Ensure HTTP client is available."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def reason(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> BicameralResult:
        """
        Perform bicameral reasoning on a query.

        Args:
            query: User query to reason about
            system_prompt: Optional system prompt override
            context: Optional additional context

        Returns:
            BicameralResult with reasoning, critique, and synthesis
        """
        start_time = time.perf_counter()

        # Route to bicameral pair
        cold_decision, warm_decision = self.router.route_bicameral(query)

        # Build full prompt with context
        full_query = query
        if context:
            full_query = f"Context:\n{context}\n\nQuery:\n{query}"

        # Iteration loop
        cold_response = ""
        warm_critique = ""
        consensus_reached = False
        iterations = 0

        for i in range(self.max_iterations):
            iterations = i + 1

            # Step 1: Cold Core generates response
            cold_prompt = self._build_cold_prompt(full_query, warm_critique, i)
            cold_response = await self._call_model(
                cold_decision.model_config,
                cold_prompt,
                system_prompt or self._cold_system_prompt(),
            )

            # Step 2: Warm Surface verifies/critiques
            warm_prompt = self._build_warm_prompt(full_query, cold_response)
            warm_critique = await self._call_model(
                warm_decision.model_config,
                warm_prompt,
                self._warm_system_prompt(),
            )

            # Step 3: Check for consensus
            consensus_reached = self._check_consensus(warm_critique)
            if consensus_reached:
                break

        # Step 4: Synthesize final answer
        synthesis = await self._synthesize(
            cold_response,
            warm_critique,
            cold_decision.model_config,
        )

        total_time = (time.perf_counter() - start_time) * 1000

        return BicameralResult(
            cold_response=cold_response,
            warm_critique=warm_critique,
            synthesis=synthesis,
            cold_model=cold_decision.model_name,
            warm_model=warm_decision.model_name,
            consensus_reached=consensus_reached,
            iterations=iterations,
            total_time_ms=total_time,
            metadata={
                "cold_confidence": cold_decision.confidence,
                "warm_confidence": warm_decision.confidence,
            },
        )

    async def generate_verify(
        self,
        query: str,
        candidates: int = 3,
    ) -> BicameralResult:
        """
        Generate multiple candidates and verify the best one.

        Implements Karpathy's generate-verify loop:
        1. Generate N candidate responses
        2. Score each candidate
        3. Return the best verified response

        Args:
            query: User query
            candidates: Number of candidates to generate

        Returns:
            BicameralResult with best verified response
        """
        start_time = time.perf_counter()

        cold_decision, warm_decision = self.router.route_bicameral(query)

        # Generate candidates
        candidate_responses: List[str] = []
        for i in range(candidates):
            # Vary temperature slightly for diversity
            temp_config = ModelConfig(
                endpoint=cold_decision.model_config.endpoint,
                model_id=cold_decision.model_config.model_id,
                role=cold_decision.model_config.role,
                context_window=cold_decision.model_config.context_window,
                temperature=cold_decision.model_config.temperature + (i * 0.1),
                max_tokens=cold_decision.model_config.max_tokens,
            )

            response = await self._call_model(
                temp_config,
                query,
                self._cold_system_prompt(),
            )
            candidate_responses.append(response)

        # Verify and score candidates
        best_response = ""
        best_score = 0.0
        best_critique = ""

        for response in candidate_responses:
            warm_prompt = self._build_verification_prompt(query, response)
            critique = await self._call_model(
                warm_decision.model_config,
                warm_prompt,
                self._warm_system_prompt(),
            )

            score = self._extract_score(critique)
            if score > best_score:
                best_score = score
                best_response = response
                best_critique = critique

        total_time = (time.perf_counter() - start_time) * 1000

        return BicameralResult(
            cold_response=best_response,
            warm_critique=best_critique,
            synthesis=best_response,  # Best candidate is the synthesis
            cold_model=cold_decision.model_name,
            warm_model=warm_decision.model_name,
            consensus_reached=best_score >= self.consensus_threshold,
            iterations=candidates,
            total_time_ms=total_time,
            metadata={
                "candidates_generated": candidates,
                "best_score": best_score,
            },
        )

    async def _call_model(
        self,
        config: ModelConfig,
        prompt: str,
        system_prompt: str,
    ) -> str:
        """Call a model and return the response text."""
        client = await self._ensure_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await client.post(
                f"{config.endpoint}/chat/completions",
                json={
                    "model": config.model_id,
                    "messages": messages,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_tokens": config.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return f"[Error: {str(e)[:100]}]"

    def _build_cold_prompt(
        self,
        query: str,
        previous_critique: str,
        iteration: int,
    ) -> str:
        """Build prompt for Cold Core."""
        if iteration == 0 or not previous_critique:
            return query

        return f"""Previous attempt received this feedback:
{previous_critique}

Please revise your response to address the feedback.

Original query:
{query}"""

    def _build_warm_prompt(self, query: str, cold_response: str) -> str:
        """Build verification prompt for Warm Surface."""
        return f"""Please critically evaluate this response:

QUERY:
{query}

RESPONSE:
{cold_response}

Evaluate for:
1. Correctness: Are the facts and logic correct?
2. Completeness: Does it fully address the query?
3. Clarity: Is it clear and well-organized?
4. Helpfulness: Does it actually help the user?

If the response is acceptable, start with "APPROVED:" followed by brief praise.
If it needs improvement, start with "NEEDS REVISION:" followed by specific feedback."""

    def _build_verification_prompt(self, query: str, response: str) -> str:
        """Build scoring prompt for candidate verification."""
        return f"""Score this response from 0.0 to 1.0:

QUERY: {query}

RESPONSE: {response}

Provide a score from 0.0 (poor) to 1.0 (excellent) based on accuracy, completeness, and helpfulness.
Format: SCORE: X.X followed by brief justification."""

    def _cold_system_prompt(self) -> str:
        """System prompt for Cold Core (reasoning)."""
        return """You are a precise, logical reasoning engine. Your responses should be:
- Accurate and well-reasoned
- Step-by-step when appropriate
- Factual and verifiable
- Structured and clear

Think carefully before responding. Show your reasoning."""

    def _warm_system_prompt(self) -> str:
        """System prompt for Warm Surface (verification)."""
        return """You are a thoughtful critic and verifier. Your role is to:
- Identify errors, gaps, or unclear parts
- Provide constructive feedback
- Evaluate completeness and helpfulness
- Be fair but thorough

If something is good, acknowledge it. If it needs work, be specific about why."""

    def _check_consensus(self, critique: str) -> bool:
        """Check if Warm Surface approved the response."""
        critique_lower = critique.lower()
        return (
            critique_lower.startswith("approved:") or
            "approved" in critique_lower[:50] or
            "looks good" in critique_lower[:100] or
            "well done" in critique_lower[:100]
        )

    def _extract_score(self, critique: str) -> float:
        """Extract numerical score from critique."""
        import re

        # Look for SCORE: X.X pattern
        match = re.search(r"score:\s*(\d+\.?\d*)", critique.lower())
        if match:
            try:
                return min(1.0, max(0.0, float(match.group(1))))
            except ValueError:
                pass

        # Heuristic scoring based on sentiment
        critique_lower = critique.lower()
        if "excellent" in critique_lower or "perfect" in critique_lower:
            return 0.95
        elif "good" in critique_lower or "approved" in critique_lower:
            return 0.85
        elif "acceptable" in critique_lower:
            return 0.7
        elif "poor" in critique_lower or "incorrect" in critique_lower:
            return 0.3

        return 0.5  # Neutral default

    async def _synthesize(
        self,
        cold_response: str,
        warm_critique: str,
        config: ModelConfig,
    ) -> str:
        """Synthesize final response from Cold response and Warm feedback."""
        # If approved, return cold response as-is
        if self._check_consensus(warm_critique):
            return cold_response

        # Otherwise, synthesize incorporating feedback
        synthesis_prompt = f"""Given this response and critique, provide an improved final answer:

ORIGINAL RESPONSE:
{cold_response}

CRITIQUE:
{warm_critique}

Provide an improved response that addresses the critique while maintaining accuracy."""

        return await self._call_model(
            config,
            synthesis_prompt,
            self._cold_system_prompt(),
        )


# =============================================================================
# LOCAL-FIRST INFERENCE MANAGER
# =============================================================================


class LocalFirstManager:
    """
    High-level manager for local-first inference with sovereignty guarantees.

    Provides:
    - Unified interface for all local models
    - Automatic routing based on task
    - Health monitoring and failover
    - Bicameral reasoning when needed

    Sovereignty Guarantees:
    - No external API calls by default
    - Explicit consent required for cloud fallback
    - All data stays on node unless authorized
    """

    def __init__(
        self,
        models: Optional[Dict[str, ModelConfig]] = None,
        fallback_level: FallbackLevel = FallbackLevel.LOCAL_ONLY,
        enable_health_monitor: bool = True,
    ):
        self.models = models or LOCAL_MODELS
        self.fallback_level = fallback_level

        self.health_monitor = HealthMonitor(self.models) if enable_health_monitor else None
        self.router = ModelRouter(self.models, self.health_monitor)
        self.bicameral = BicameralOrchestrator(self.router, self.health_monitor)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize manager and start health monitoring."""
        if self._initialized:
            return

        if self.health_monitor:
            await self.health_monitor.start()
            # Initial health check
            await self.health_monitor.check_all()

        self._initialized = True
        logger.info(f"LocalFirstManager initialized with {len(self.models)} models")

    async def shutdown(self) -> None:
        """Shutdown manager and cleanup resources."""
        if self.health_monitor:
            await self.health_monitor.stop()
        await self.bicameral.close()
        self._initialized = False

    async def infer(
        self,
        prompt: str,
        task_type: Optional[TaskType] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Perform inference with automatic routing.

        Args:
            prompt: User prompt
            task_type: Optional task type for routing
            model_name: Optional specific model to use
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Model response text
        """
        # Get model config
        if model_name and model_name in self.models:
            config = self.models[model_name]
        elif task_type:
            decision = self.router.route_by_task(task_type)
            config = decision.model_config
        else:
            # Default to fast model
            decision = self.router.route_by_task(TaskType.CONVERSATION)
            config = decision.model_config

        # Apply overrides
        if temperature is not None:
            config = ModelConfig(
                endpoint=config.endpoint,
                model_id=config.model_id,
                role=config.role,
                context_window=config.context_window,
                temperature=temperature,
                max_tokens=max_tokens or config.max_tokens,
            )

        # Call model
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await client.post(
                f"{config.endpoint}/chat/completions",
                json={
                    "model": config.model_id,
                    "messages": messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def reason(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> BicameralResult:
        """
        Perform bicameral reasoning on a query.

        Uses Cold Core + Warm Surface for high-quality reasoning.

        Args:
            query: Query to reason about
            context: Optional additional context

        Returns:
            BicameralResult with reasoning and synthesis
        """
        return await self.bicameral.reason(query, context=context)

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        decision = self.router.get_embedding_model()
        config = decision.model_config

        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{config.endpoint}/embeddings",
                json={
                    "model": config.model_id,
                    "input": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    def get_available_models(self) -> List[str]:
        """Get list of available (healthy) models."""
        if self.health_monitor:
            return self.health_monitor.get_healthy_models()
        return list(self.models.keys())

    def get_model_info(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)

    def get_health_report(self) -> Dict[str, HealthReport]:
        """Get health reports for all models."""
        if self.health_monitor:
            return dict(self.health_monitor._health_reports)
        return {}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_manager_instance: Optional[LocalFirstManager] = None


async def get_local_first_manager() -> LocalFirstManager:
    """
    Get or create the singleton LocalFirstManager.

    Usage:
        manager = await get_local_first_manager()
        response = await manager.infer("Hello!")
    """
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = LocalFirstManager()
        await _manager_instance.initialize()

    return _manager_instance


def get_model_router() -> ModelRouter:
    """Get a ModelRouter instance for routing decisions."""
    return ModelRouter(LOCAL_MODELS)


def get_bicameral_orchestrator() -> BicameralOrchestrator:
    """Get a BicameralOrchestrator for cold/warm reasoning."""
    return BicameralOrchestrator()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ModelRole",
    "TaskType",
    "FallbackLevel",
    "HealthStatus",

    # Data models
    "ModelConfig",
    "HealthReport",
    "RoutingDecision",
    "BicameralResult",

    # Registry
    "LOCAL_MODELS",
    "TASK_TO_ROLE",

    # Classes
    "ModelRouter",
    "HealthMonitor",
    "BicameralOrchestrator",
    "LocalFirstManager",

    # Convenience functions
    "get_local_first_manager",
    "get_model_router",
    "get_bicameral_orchestrator",
]
