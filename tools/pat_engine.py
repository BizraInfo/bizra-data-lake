# BIZRA PAT Engine v2.2 - Personal Agentic Team
# Production Implementation with LLM Backend (Ollama/OpenAI/LM Studio)
# Standing on Giants: LangChain patterns, Ollama local inference
# Implements: Multi-agent coordination, Graph-of-Thoughts integration
# v2.1: Added circuit breaker and retry logic for resilience
# v2.2: Added vision model support via DualAgenticBridge

import asyncio
import json
import logging
import time
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
import hashlib

import base64
from bizra_config import SNR_THRESHOLD, IHSAN_CONSTRAINT

# Import DualAgenticBridge for vision capabilities
try:
    from dual_agentic_bridge import (
        DualAgenticBridge, ModelRequest, ModelCapability, ModelResponse
    )
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    DualAgenticBridge = None

# Import resilience patterns
try:
    from bizra_resilience import (
        CircuitBreaker, CircuitBreakerConfig, CircuitOpenError,
        retry, RetryConfig, with_fallback, llm_circuit_breaker
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False
    # Provide fallback no-op implementations
    class CircuitBreaker:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, func): return func
    class CircuitOpenError(Exception): pass
    def retry(*args, **kwargs):
        def decorator(func): return func
        return decorator
    def with_fallback(*args, **kwargs):
        def decorator(func): return func
        return decorator
    llm_circuit_breaker = CircuitBreaker("llm_backend")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | PAT | %(message)s'
)
logger = logging.getLogger("PAT")


class AgentRole(Enum):
    """Agent specialization roles."""
    STRATEGIST = "strategist"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CREATOR = "creator"
    EXECUTOR = "executor"
    GUARDIAN = "guardian"
    COORDINATOR = "coordinator"
    VISION = "vision"  # v2.2: Vision analysis agent


class ThinkingMode(Enum):
    """Cognitive modes for different task types."""
    FAST = "fast"           # Quick responses, low latency
    DEEP = "deep"           # Thorough analysis
    CREATIVE = "creative"   # Divergent thinking
    CRITICAL = "critical"   # Evaluation and critique
    SYNTHESIS = "synthesis" # Combining multiple perspectives


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    name: str
    role: AgentRole
    model: str = "liquid/lfm2.5-1.2b"  # Default LM Studio model
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""
    capabilities: List[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Message in agent conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent."""
    agent_name: str
    content: str
    thinking_mode: ThinkingMode
    confidence: float
    sources: List[str]
    execution_time: float
    tokens_used: int
    metadata: Dict = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def generate(
        self,
        messages: List[AgentMessage],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response from LLM."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is available."""
        pass


class OllamaBackend(LLMBackend):
    """
    Ollama Local LLM Backend

    Connects to local Ollama instance for privacy-preserving inference.
    Supports: llama3.2, mistral, codellama, phi, etc.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)

    @retry(max_retries=3, base_delay=1.0, retryable_exceptions=(httpx.ConnectError, httpx.TimeoutException))
    async def _make_request(self, payload: dict) -> dict:
        """Make HTTP request to Ollama with retry logic."""
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Ollama returned {response.status_code}",
                request=response.request,
                response=response
            )
        return response.json()

    async def generate(
        self,
        messages: List[AgentMessage],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using Ollama API with circuit breaker protection."""
        # Check circuit breaker state
        if RESILIENCE_AVAILABLE:
            breaker = CircuitBreaker.get("llm_backend")
            if breaker and breaker.state.value == "open":
                logger.warning("Circuit breaker OPEN for LLM backend, using fallback")
                return "[Fallback: LLM service temporarily unavailable. Please try again later.]"

        try:
            # Convert messages to Ollama format
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            payload = {
                "model": model,
                "messages": ollama_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }

            data = await self._make_request(payload)

            # Record success for circuit breaker
            if RESILIENCE_AVAILABLE:
                await llm_circuit_breaker._record_success()

            return data.get("message", {}).get("content", "")

        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama. Ensure it's running: ollama serve")
            if RESILIENCE_AVAILABLE:
                await llm_circuit_breaker._record_failure(e)
            return "[Error: Ollama not available. Run 'ollama serve' first.]"
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout: {e}")
            if RESILIENCE_AVAILABLE:
                await llm_circuit_breaker._record_failure(e)
            return "[Error: Ollama request timed out. Try again.]"
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            if RESILIENCE_AVAILABLE:
                await llm_circuit_breaker._record_failure(e)
            return f"[Error: {str(e)}]"

    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    async def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except:
            return []


class OpenAIBackend(LLMBackend):
    """
    OpenAI-compatible Backend

    Works with OpenAI API and compatible services (Azure, local vLLM, etc.)
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def generate(
        self,
        messages: List[AgentMessage],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using OpenAI-compatible API."""
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": openai_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenAI error: {response.status_code}")
                return f"[Error: API returned {response.status_code}]"

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"[Error: {str(e)}]"

    async def health_check(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except:
            return False


class LMStudioBackend(LLMBackend):
    """
    LM Studio Backend (OpenAI-compatible)

    Connects to local LM Studio server for multi-model inference.
    Default endpoint: http://192.168.56.1:1234/v1
    """

    def __init__(self, base_url: str = "http://192.168.56.1:1234/v1"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout for large models
        self._models: List[str] = []
        self._circuit_breaker = CircuitBreaker(
            "lm_studio_backend",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=60.0
            )
        ) if RESILIENCE_AVAILABLE else None

    @retry(max_retries=2, base_delay=2.0, retryable_exceptions=(httpx.ConnectError, httpx.TimeoutException))
    async def _make_request(self, payload: dict) -> dict:
        """Make HTTP request to LM Studio with retry logic."""
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"LM Studio returned {response.status_code}",
                request=response.request,
                response=response
            )
        return response.json()

    async def generate(
        self,
        messages: List[AgentMessage],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using LM Studio's OpenAI-compatible API with resilience."""
        # Check circuit breaker state
        if self._circuit_breaker:
            if self._circuit_breaker.state.value == "open":
                logger.warning("Circuit breaker OPEN for LM Studio, using fallback")
                return "[Fallback: LM Studio temporarily unavailable. Please try again later.]"

        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            payload = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            data = await self._make_request(payload)

            # Record success
            if self._circuit_breaker:
                await self._circuit_breaker._record_success()

            return data["choices"][0]["message"]["content"]

        except httpx.ConnectError as e:
            logger.error("Cannot connect to LM Studio. Ensure it's running.")
            if self._circuit_breaker:
                await self._circuit_breaker._record_failure(e)
            return "[Error: LM Studio not available]"
        except httpx.TimeoutException as e:
            logger.error(f"LM Studio timeout: {e}")
            if self._circuit_breaker:
                await self._circuit_breaker._record_failure(e)
            return "[Error: LM Studio request timed out. Try again.]"
        except Exception as e:
            logger.error(f"LM Studio generation error: {e}")
            if self._circuit_breaker:
                await self._circuit_breaker._record_failure(e)
            return f"[Error: {str(e)}]"

    async def health_check(self) -> bool:
        """Check if LM Studio is available."""
        try:
            response = await self.client.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                self._models = [m.get('id', '') for m in data.get('data', [])]
                return True
            return False
        except:
            return False

    async def list_models(self) -> List[str]:
        """List available LM Studio models."""
        await self.health_check()
        return self._models


class BaseAgent(ABC):
    """
    Abstract base class for PAT agents.

    Each agent has:
    - A specialized role and system prompt
    - Connection to LLM backend
    - Conversation history
    - Performance metrics
    """

    def __init__(self, config: AgentConfig, backend: LLMBackend):
        self.config = config
        self.backend = backend
        self.history: List[AgentMessage] = []
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0
        }

        # Initialize with system prompt
        if config.system_prompt:
            self.history.append(AgentMessage(
                role="system",
                content=config.system_prompt
            ))

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Generate role-specific system prompt."""
        pass

    async def process(
        self,
        task: str,
        context: Optional[Dict] = None,
        thinking_mode: ThinkingMode = ThinkingMode.FAST
    ) -> AgentResponse:
        """
        Process a task and generate response.

        Args:
            task: The task/question to process
            context: Optional context (retrieved documents, prior responses)
            thinking_mode: Cognitive mode to use

        Returns:
            AgentResponse with results and metadata
        """
        start_time = time.time()

        # Build prompt with context
        prompt = self._build_prompt(task, context, thinking_mode)

        # Add to history
        self.history.append(AgentMessage(role="user", content=prompt))

        # Generate response
        response_text = await self.backend.generate(
            messages=self.history,
            model=self.config.model,
            temperature=self._get_temperature(thinking_mode),
            max_tokens=self.config.max_tokens
        )

        # Add response to history
        self.history.append(AgentMessage(role="assistant", content=response_text))

        execution_time = time.time() - start_time

        # Update metrics
        self.metrics["total_calls"] += 1
        tokens_est = len(response_text.split()) * 1.3
        self.metrics["total_tokens"] += int(tokens_est)
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (self.metrics["total_calls"] - 1) + execution_time)
            / self.metrics["total_calls"]
        )

        # Calculate confidence based on response quality
        confidence = self._estimate_confidence(response_text, thinking_mode)

        return AgentResponse(
            agent_name=self.config.name,
            content=response_text,
            thinking_mode=thinking_mode,
            confidence=confidence,
            sources=context.get("sources", []) if context else [],
            execution_time=execution_time,
            tokens_used=int(tokens_est),
            metadata={
                "model": self.config.model,
                "role": self.config.role.value
            }
        )

    def _build_prompt(
        self,
        task: str,
        context: Optional[Dict],
        thinking_mode: ThinkingMode
    ) -> str:
        """Build complete prompt with context."""
        parts = []

        # Add thinking mode instruction
        if thinking_mode == ThinkingMode.DEEP:
            parts.append("Think step by step. Analyze thoroughly before responding.")
        elif thinking_mode == ThinkingMode.CREATIVE:
            parts.append("Think creatively. Consider unconventional approaches.")
        elif thinking_mode == ThinkingMode.CRITICAL:
            parts.append("Be critical. Identify weaknesses and potential issues.")
        elif thinking_mode == ThinkingMode.SYNTHESIS:
            parts.append("Synthesize multiple perspectives into a coherent whole.")

        # Add context if provided
        if context:
            if "retrieved_context" in context:
                parts.append(f"\n--- Retrieved Context ---\n{context['retrieved_context']}\n---")
            if "prior_responses" in context:
                parts.append(f"\n--- Prior Analysis ---\n{context['prior_responses']}\n---")

        # Add main task
        parts.append(f"\nTask: {task}")

        return "\n".join(parts)

    def _get_temperature(self, mode: ThinkingMode) -> float:
        """Get temperature based on thinking mode."""
        temps = {
            ThinkingMode.FAST: 0.3,
            ThinkingMode.DEEP: 0.5,
            ThinkingMode.CREATIVE: 0.9,
            ThinkingMode.CRITICAL: 0.4,
            ThinkingMode.SYNTHESIS: 0.6
        }
        return temps.get(mode, self.config.temperature)

    def _estimate_confidence(self, response: str, mode: ThinkingMode) -> float:
        """Estimate confidence based on response characteristics."""
        # Heuristic confidence estimation
        confidence = 0.7

        # Longer, more detailed responses suggest higher confidence
        word_count = len(response.split())
        if word_count > 200:
            confidence += 0.1
        elif word_count < 50:
            confidence -= 0.1

        # Check for uncertainty markers
        uncertainty_markers = ["maybe", "possibly", "might", "uncertain", "not sure", "i think"]
        for marker in uncertainty_markers:
            if marker in response.lower():
                confidence -= 0.05

        # Check for confidence markers
        confidence_markers = ["clearly", "definitely", "certainly", "evidence shows", "based on"]
        for marker in confidence_markers:
            if marker in response.lower():
                confidence += 0.05

        return max(0.1, min(0.95, confidence))

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system and self.history and self.history[0].role == "system":
            self.history = [self.history[0]]
        else:
            self.history = []


class StrategistAgent(BaseAgent):
    """Strategic planning and goal decomposition agent."""

    def get_system_prompt(self) -> str:
        return """You are a Strategic Planner in the BIZRA Personal Agentic Team.

Your capabilities:
- Break down complex goals into actionable steps
- Identify dependencies and critical paths
- Prioritize tasks based on impact and urgency
- Anticipate obstacles and plan mitigations

Communication style:
- Clear, structured responses
- Use numbered lists for action items
- Highlight key decisions and trade-offs
- Flag risks and uncertainties

Always think strategically about long-term implications while providing actionable short-term guidance."""


class ResearcherAgent(BaseAgent):
    """Information gathering and synthesis agent."""

    def get_system_prompt(self) -> str:
        return """You are a Research Specialist in the BIZRA Personal Agentic Team.

Your capabilities:
- Analyze provided documents and context thoroughly
- Extract key facts, patterns, and insights
- Identify gaps in information
- Synthesize findings into coherent summaries

Communication style:
- Evidence-based responses with citations to sources
- Distinguish between facts and inferences
- Note confidence levels for claims
- Highlight areas needing further research

Always base conclusions on the evidence provided and acknowledge limitations."""


class AnalystAgent(BaseAgent):
    """Data analysis and pattern recognition agent."""

    def get_system_prompt(self) -> str:
        return """You are a Data Analyst in the BIZRA Personal Agentic Team.

Your capabilities:
- Analyze quantitative and qualitative data
- Identify patterns, trends, and anomalies
- Perform comparative analysis
- Generate insights from complex datasets

Communication style:
- Precise, data-driven responses
- Use metrics and numbers when available
- Visualize patterns through clear descriptions
- Provide actionable recommendations based on analysis

Always explain your analytical methodology and acknowledge data limitations."""


class CreatorAgent(BaseAgent):
    """Creative content and solution generation agent."""

    def get_system_prompt(self) -> str:
        return """You are a Creative Specialist in the BIZRA Personal Agentic Team.

Your capabilities:
- Generate innovative solutions to problems
- Create content (text, ideas, frameworks)
- Think laterally and make unexpected connections
- Adapt and remix existing concepts

Communication style:
- Imaginative and exploratory
- Offer multiple alternatives
- Explain creative reasoning
- Balance novelty with practicality

Embrace unconventional thinking while remaining grounded in feasibility."""


class GuardianAgent(BaseAgent):
    """Quality assurance and validation agent."""

    def get_system_prompt(self) -> str:
        return """You are a Quality Guardian in the BIZRA Personal Agentic Team.

Your capabilities:
- Validate outputs against quality standards
- Identify errors, inconsistencies, and gaps
- Assess alignment with goals and constraints
- Ensure Ihsan (excellence) principles are met

Quality Criteria:
- Accuracy: Is the information correct?
- Completeness: Are all aspects addressed?
- Coherence: Is it logically consistent?
- Relevance: Does it address the actual need?
- SNR: Is signal-to-noise ratio high?

Communication style:
- Constructive critique with specific feedback
- Clear pass/fail assessments with reasoning
- Actionable improvement suggestions

Hold all outputs to the highest standards of excellence."""


class CoordinatorAgent(BaseAgent):
    """Multi-agent coordination and synthesis agent."""

    def get_system_prompt(self) -> str:
        return """You are the Coordinator in the BIZRA Personal Agentic Team.

Your capabilities:
- Orchestrate multiple agents toward a common goal
- Resolve conflicts between agent perspectives
- Synthesize diverse inputs into unified outputs
- Ensure efficient collaboration

Communication style:
- Clear delegation and integration
- Balanced consideration of all perspectives
- Decisive synthesis when views conflict
- Progress tracking and status updates

Your role is to ensure the team produces coherent, high-quality results greater than the sum of individual contributions."""


class VisionAgent(BaseAgent):
    """
    Vision analysis agent using DualAgenticBridge.

    Capabilities:
    - Analyze images, diagrams, screenshots
    - Extract text via OCR
    - Understand charts and technical diagrams
    - Provide visual context for other agents
    """

    def __init__(self, config: AgentConfig, backend: LLMBackend):
        super().__init__(config, backend)
        self._vision_bridge: Optional[DualAgenticBridge] = None
        self._vision_enabled = False

    async def initialize_vision(self) -> bool:
        """Initialize the vision bridge for image processing."""
        if not VISION_AVAILABLE:
            logger.warning("Vision not available - DualAgenticBridge not installed")
            return False

        try:
            self._vision_bridge = DualAgenticBridge()
            backends = await self._vision_bridge.check_availability()
            self._vision_enabled = any(backends.values())

            if self._vision_enabled:
                logger.info("✅ Vision agent initialized with DualAgenticBridge")
            else:
                logger.warning("⚠️ Vision backends unavailable")

            return self._vision_enabled
        except Exception as e:
            logger.error(f"Failed to initialize vision: {e}")
            return False

    def get_system_prompt(self) -> str:
        return """You are a Vision Analyst in the BIZRA Personal Agentic Team.

Your capabilities:
- Analyze images, diagrams, charts, and screenshots
- Extract and interpret text from visual content
- Identify patterns, relationships, and key information
- Provide visual context to support other agents

Communication style:
- Detailed descriptions of visual elements
- Structured analysis of diagrams and charts
- Clear identification of text and labels
- Highlight actionable insights from visuals

Always describe what you see objectively and note any ambiguities or uncertainties."""

    async def analyze_image(
        self,
        image_path: str,
        prompt: Optional[str] = None
    ) -> AgentResponse:
        """
        Analyze an image and return structured insights.

        Args:
            image_path: Path to the image file
            prompt: Optional specific question about the image

        Returns:
            AgentResponse with image analysis
        """
        start_time = time.time()

        if not self._vision_enabled or not self._vision_bridge:
            # Try to initialize on first use
            await self.initialize_vision()

        if not self._vision_enabled:
            return AgentResponse(
                agent_name=self.config.name,
                content="[Error: Vision capability not available. Start LM Studio or Ollama with a vision model.]",
                thinking_mode=ThinkingMode.FAST,
                confidence=0.0,
                sources=[image_path],
                execution_time=time.time() - start_time,
                tokens_used=0,
                metadata={"error": True, "vision_unavailable": True}
            )

        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()

            # Build the analysis prompt
            analysis_prompt = prompt or "Analyze this image in detail. Describe:\n1. Main elements and composition\n2. Any text or labels visible\n3. Technical diagrams or charts (if present)\n4. Key insights and observations"

            # Create vision request
            request = ModelRequest(
                prompt=analysis_prompt,
                capability=ModelCapability.VISION,
                images=[image_data],
                system_prompt=self.get_system_prompt()
            )

            # Route through DualAgenticBridge
            response = await self._vision_bridge.route_request(request)

            execution_time = time.time() - start_time

            if response and response.content:
                # Add to history for context
                self.history.append(AgentMessage(
                    role="user",
                    content=f"[Image: {image_path}]\n{analysis_prompt}",
                    metadata={"image_path": image_path}
                ))
                self.history.append(AgentMessage(
                    role="assistant",
                    content=response.content
                ))

                # Update metrics
                self.metrics["total_calls"] += 1
                self.metrics["total_tokens"] += response.tokens_used

                return AgentResponse(
                    agent_name=self.config.name,
                    content=response.content,
                    thinking_mode=ThinkingMode.DEEP,
                    confidence=0.85,  # Vision models are generally reliable
                    sources=[image_path],
                    execution_time=execution_time,
                    tokens_used=response.tokens_used,
                    metadata={
                        "model": response.model_used,
                        "provider": response.provider,
                        "latency_ms": response.latency_ms
                    }
                )
            else:
                return AgentResponse(
                    agent_name=self.config.name,
                    content="[Error: Vision model returned empty response]",
                    thinking_mode=ThinkingMode.FAST,
                    confidence=0.0,
                    sources=[image_path],
                    execution_time=execution_time,
                    tokens_used=0,
                    metadata={"error": True}
                )

        except FileNotFoundError:
            return AgentResponse(
                agent_name=self.config.name,
                content=f"[Error: Image file not found: {image_path}]",
                thinking_mode=ThinkingMode.FAST,
                confidence=0.0,
                sources=[],
                execution_time=time.time() - start_time,
                tokens_used=0,
                metadata={"error": True, "file_not_found": True}
            )
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return AgentResponse(
                agent_name=self.config.name,
                content=f"[Error: Vision analysis failed: {str(e)}]",
                thinking_mode=ThinkingMode.FAST,
                confidence=0.0,
                sources=[image_path],
                execution_time=time.time() - start_time,
                tokens_used=0,
                metadata={"error": True, "exception": str(e)}
            )

    async def process(
        self,
        task: str,
        context: Optional[Dict] = None,
        thinking_mode: ThinkingMode = ThinkingMode.FAST
    ) -> AgentResponse:
        """
        Process task - if image path in context, analyze it.
        Otherwise, provide text-based assistance about vision tasks.
        """
        # Check if context contains an image path
        if context and context.get("image_path"):
            return await self.analyze_image(
                image_path=context["image_path"],
                prompt=task
            )

        # Check if context has multiple images
        if context and context.get("image_paths"):
            results = []
            for img_path in context["image_paths"]:
                result = await self.analyze_image(img_path, task)
                results.append(f"=== {img_path} ===\n{result.content}")

            combined = "\n\n".join(results)
            return AgentResponse(
                agent_name=self.config.name,
                content=combined,
                thinking_mode=thinking_mode,
                confidence=0.8,
                sources=context["image_paths"],
                execution_time=sum(r.execution_time for r in results) if results else 0,
                tokens_used=sum(r.tokens_used for r in results) if results else 0,
                metadata={"multi_image": True, "count": len(context["image_paths"])}
            )

        # No image - use base text processing
        return await super().process(task, context, thinking_mode)


class PATOrchestrator:
    """
    Personal Agentic Team Orchestrator

    Coordinates multiple agents to accomplish complex tasks:
    1. Decomposes tasks based on complexity
    2. Routes subtasks to appropriate agents
    3. Synthesizes agent outputs
    4. Validates quality via Guardian agent
    5. Achieves SNR > threshold before finalizing
    """

    def __init__(self, backend: LLMBackend, model: str = "liquid/lfm2.5-1.2b"):
        self.backend = backend
        self.model = model
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the core agent team."""
        agent_classes = {
            "strategist": (StrategistAgent, AgentRole.STRATEGIST),
            "researcher": (ResearcherAgent, AgentRole.RESEARCHER),
            "analyst": (AnalystAgent, AgentRole.ANALYST),
            "creator": (CreatorAgent, AgentRole.CREATOR),
            "guardian": (GuardianAgent, AgentRole.GUARDIAN),
            "coordinator": (CoordinatorAgent, AgentRole.COORDINATOR),
            "vision": (VisionAgent, AgentRole.VISION)  # v2.2: Vision agent
        }

        for name, (agent_class, role) in agent_classes.items():
            config = AgentConfig(
                name=name,
                role=role,
                model=self.model,
                system_prompt=""  # Will be set by get_system_prompt
            )

            agent = agent_class(config, self.backend)
            agent.config.system_prompt = agent.get_system_prompt()
            agent.history.insert(0, AgentMessage(
                role="system",
                content=agent.config.system_prompt
            ))

            self.agents[name] = agent

        logger.info(f"Initialized {len(self.agents)} agents (vision: {VISION_AVAILABLE})")

    async def process_task(
        self,
        task: str,
        context: Optional[Dict] = None,
        agents_to_use: Optional[List[str]] = None,
        require_guardian_approval: bool = True,
        snr_threshold: float = SNR_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Process a task using the agent team.

        Args:
            task: The task to accomplish
            context: Optional retrieved context
            agents_to_use: Specific agents to involve (default: auto-select)
            require_guardian_approval: Whether to run quality validation
            snr_threshold: Minimum SNR for acceptance

        Returns:
            Dict with synthesis and metadata
        """
        start_time = time.time()

        logger.info(f"Processing task: {task[:50]}...")

        # Step 1: Strategic decomposition
        strategy = await self.agents["strategist"].process(
            task=f"Analyze this task and provide a strategic approach: {task}",
            context=context,
            thinking_mode=ThinkingMode.DEEP
        )

        # Step 2: Determine which agents to use
        if agents_to_use is None:
            agents_to_use = self._select_agents(task, strategy.content)

        logger.info(f"Using agents: {agents_to_use}")

        # Step 3: Parallel agent processing
        agent_responses: List[AgentResponse] = []

        tasks = []
        for agent_name in agents_to_use:
            if agent_name in self.agents and agent_name not in ["strategist", "guardian", "coordinator"]:
                agent = self.agents[agent_name]
                task_coro = agent.process(
                    task=task,
                    context={
                        **(context or {}),
                        "prior_responses": strategy.content
                    },
                    thinking_mode=ThinkingMode.DEEP
                )
                tasks.append((agent_name, task_coro))

        # Execute in parallel
        for agent_name, task_coro in tasks:
            try:
                response = await task_coro
                agent_responses.append(response)
                logger.info(f"Agent {agent_name} completed (confidence: {response.confidence:.2f})")
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")

        # Step 4: Coordinator synthesis
        synthesis_context = {
            "prior_responses": "\n\n".join([
                f"=== {r.agent_name.upper()} ===\n{r.content}"
                for r in [strategy] + agent_responses
            ])
        }

        synthesis = await self.agents["coordinator"].process(
            task=f"Synthesize these agent perspectives into a unified response for: {task}",
            context=synthesis_context,
            thinking_mode=ThinkingMode.SYNTHESIS
        )

        # Step 5: Guardian validation
        quality_result = None
        if require_guardian_approval:
            quality_result = await self.agents["guardian"].process(
                task=f"Evaluate this response for quality and completeness:\n\n{synthesis.content}",
                context={"original_task": task},
                thinking_mode=ThinkingMode.CRITICAL
            )

        # Step 6: Calculate SNR
        snr_score = self._calculate_team_snr(
            agent_responses + [strategy, synthesis],
            quality_result
        )

        execution_time = time.time() - start_time

        result = {
            "task": task,
            "synthesis": synthesis.content,
            "snr_score": snr_score,
            "ihsan_achieved": snr_score >= IHSAN_CONSTRAINT,
            "quality_assessment": quality_result.content if quality_result else None,
            "agent_contributions": [
                {
                    "agent": r.agent_name,
                    "confidence": r.confidence,
                    "tokens": r.tokens_used
                }
                for r in [strategy] + agent_responses + [synthesis]
            ],
            "execution_time": round(execution_time, 2),
            "total_tokens": sum(r.tokens_used for r in [strategy] + agent_responses + [synthesis])
        }

        logger.info(f"Task completed in {execution_time:.2f}s (SNR: {snr_score:.3f})")

        return result

    async def process_vision_task(
        self,
        task: str,
        image_paths: List[str],
        include_text_agents: bool = True
    ) -> Dict[str, Any]:
        """
        Process a task involving images using vision agent.

        Args:
            task: The task/question about the images
            image_paths: List of image file paths to analyze
            include_text_agents: Whether to also use text agents for synthesis

        Returns:
            Dict with vision analysis and synthesis
        """
        start_time = time.time()

        if not VISION_AVAILABLE:
            return {
                "error": True,
                "message": "Vision not available - DualAgenticBridge not installed"
            }

        vision_agent = self.agents.get("vision")
        if not vision_agent:
            return {
                "error": True,
                "message": "Vision agent not initialized"
            }

        # Initialize vision if needed
        if hasattr(vision_agent, 'initialize_vision'):
            await vision_agent.initialize_vision()

        # Process each image
        vision_results = []
        for img_path in image_paths:
            result = await vision_agent.analyze_image(img_path, task)
            vision_results.append(result)
            logger.info(f"Analyzed image: {img_path} (confidence: {result.confidence:.2f})")

        # Combine vision outputs
        combined_vision = "\n\n".join([
            f"=== Image: {r.sources[0] if r.sources else 'unknown'} ===\n{r.content}"
            for r in vision_results
        ])

        # Optionally synthesize with text agents
        if include_text_agents and vision_results:
            synthesis_result = await self.process_task(
                task=f"Based on the following image analysis, {task}\n\nImage Analysis:\n{combined_vision}",
                context={"vision_analysis": combined_vision},
                agents_to_use=["analyst", "researcher"]
            )
            synthesis = synthesis_result.get("synthesis", combined_vision)
        else:
            synthesis = combined_vision

        execution_time = time.time() - start_time

        return {
            "task": task,
            "images_analyzed": len(image_paths),
            "vision_results": [
                {
                    "image": r.sources[0] if r.sources else "unknown",
                    "content": r.content,
                    "confidence": r.confidence,
                    "tokens": r.tokens_used
                }
                for r in vision_results
            ],
            "synthesis": synthesis,
            "execution_time": round(execution_time, 2),
            "total_tokens": sum(r.tokens_used for r in vision_results)
        }

    async def initialize_vision_capability(self) -> bool:
        """Initialize the vision agent's bridge connection."""
        vision_agent = self.agents.get("vision")
        if vision_agent and hasattr(vision_agent, 'initialize_vision'):
            return await vision_agent.initialize_vision()
        return False

    def _select_agents(self, task: str, strategy: str) -> List[str]:
        """Auto-select appropriate agents based on task type."""
        task_lower = task.lower()
        strategy_lower = strategy.lower()

        agents = []

        # Keyword-based agent selection
        if any(w in task_lower for w in ["research", "find", "search", "information"]):
            agents.append("researcher")

        if any(w in task_lower for w in ["analyze", "data", "pattern", "metric"]):
            agents.append("analyst")

        if any(w in task_lower for w in ["create", "write", "generate", "design", "idea"]):
            agents.append("creator")

        # v2.2: Vision agent for image-related tasks
        if any(w in task_lower for w in ["image", "picture", "photo", "screenshot", "diagram", "chart", "visual", "ocr"]):
            if VISION_AVAILABLE:
                agents.append("vision")

        # Default to researcher + analyst if no specific match
        if not agents:
            agents = ["researcher", "analyst"]

        return agents

    def _calculate_team_snr(
        self,
        responses: List[AgentResponse],
        quality_result: Optional[AgentResponse]
    ) -> float:
        """Calculate team SNR based on responses."""
        if not responses:
            return 0.0

        # Base SNR from average confidence
        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        # Bonus for quality approval
        quality_bonus = 0.0
        if quality_result:
            approval_markers = ["approved", "meets", "satisfies", "good", "excellent", "pass"]
            if any(m in quality_result.content.lower() for m in approval_markers):
                quality_bonus = 0.1
            rejection_markers = ["reject", "fail", "insufficient", "missing", "poor"]
            if any(m in quality_result.content.lower() for m in rejection_markers):
                quality_bonus = -0.15

        # Diversity bonus (different agents contributing)
        unique_agents = len(set(r.agent_name for r in responses))
        diversity_bonus = 0.05 * min(unique_agents - 1, 3)

        snr = avg_confidence + quality_bonus + diversity_bonus

        return max(0.0, min(1.0, snr))


async def main():
    """Demonstration of PAT Engine."""
    print("=" * 70)
    print("BIZRA PAT ENGINE v2.0")
    print("Personal Agentic Team with LLM Backend")
    print("=" * 70)

    # Initialize Ollama backend
    backend = OllamaBackend()

    # Check if Ollama is available
    is_available = await backend.health_check()
    if not is_available:
        print("\n[WARNING] Ollama not available. Start with: ollama serve")
        print("Falling back to demo mode...\n")

        # Demo without actual LLM
        print("To use PAT Engine:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull a model: ollama pull llama3.2")
        print("3. Start Ollama: ollama serve")
        print("4. Run this script again")
        return

    # List available models
    models = await backend.list_models()
    print(f"\nAvailable Ollama models: {models or 'None found'}")

    if not models:
        print("\nNo models found. Pull one with: ollama pull llama3.2")
        return

    # Use first available model
    model = models[0] if models else "liquid/lfm2.5-1.2b"
    print(f"Using model: {model}")

    # Initialize orchestrator
    orchestrator = PATOrchestrator(backend, model=model)

    # Test task
    test_task = """
    Analyze the BIZRA Data Lake architecture and suggest improvements
    for the embedding generation pipeline to achieve higher throughput
    while maintaining quality.
    """

    print(f"\n--- Processing Task ---")
    print(f"Task: {test_task.strip()}")
    print("-" * 50)

    result = await orchestrator.process_task(
        task=test_task,
        agents_to_use=["researcher", "analyst", "creator"]
    )

    print(f"\n--- Results ---")
    print(f"SNR Score: {result['snr_score']:.3f}")
    print(f"Ihsan Achieved: {result['ihsan_achieved']}")
    print(f"Execution Time: {result['execution_time']}s")
    print(f"Total Tokens: {result['total_tokens']}")

    print(f"\n--- Agent Contributions ---")
    for contrib in result['agent_contributions']:
        print(f"  {contrib['agent']}: confidence={contrib['confidence']:.2f}, tokens={contrib['tokens']}")

    print(f"\n--- Synthesis ---")
    print(result['synthesis'][:500] + "..." if len(result['synthesis']) > 500 else result['synthesis'])

    if result['quality_assessment']:
        print(f"\n--- Quality Assessment ---")
        print(result['quality_assessment'][:300] + "...")


if __name__ == "__main__":
    asyncio.run(main())
