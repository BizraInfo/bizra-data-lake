# BIZRA Dual Agentic Bridge v1.2
# Connects the Data Lake to the Dual Agentic System's Multi-Model Router
# Enables vision, voice, and advanced reasoning capabilities
# v1.1: Added resilience patterns (circuit breaker, retry, auto-failover)
# v1.2: Integrated with bizra_config for centralized configuration

import os
import json
import asyncio
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import httpx

from bizra_config import (
    DUAL_AGENTIC_URL, DUAL_AGENTIC_ENABLED,
    OLLAMA_BASE_URL, OLLAMA_ENABLED, OLLAMA_TIMEOUT,
    DEFAULT_TEXT_MODEL, DEFAULT_REASONING_MODEL, DEFAULT_CODE_MODEL, DEFAULT_VISION_MODEL,
    OLLAMA_TEXT_MODEL, OLLAMA_CODE_MODEL, OLLAMA_VISION_MODEL,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD, CIRCUIT_BREAKER_TIMEOUT, CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
    HEALTH_CHECK_TIMEOUT
)

# Import resilience patterns
try:
    from bizra_resilience import (
        CircuitBreaker, CircuitBreakerConfig, retry, with_fallback
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

# LM Studio OpenAI-compatible endpoint
LM_STUDIO_CHAT_ENDPOINT = f"{DUAL_AGENTIC_URL}/v1/chat/completions"
LM_STUDIO_MODELS_ENDPOINT = f"{DUAL_AGENTIC_URL}/v1/models"


# ============================================================================
# MODEL TYPES & CONFIGURATIONS
# ============================================================================

class ModelCapability(Enum):
    """Capabilities available through the Dual Agentic System."""
    TEXT = "text"
    VISION = "vision"
    VOICE = "voice"
    REASONING = "reasoning"
    CODE = "code"
    MULTIMODAL = "multimodal"


class ModelProvider(Enum):
    """Model providers available through the router."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class ModelRequest:
    """Request to the model router."""
    prompt: str
    capability: ModelCapability
    provider: Optional[ModelProvider] = None
    model_name: Optional[str] = None
    images: List[str] = field(default_factory=list)  # Base64 encoded images
    audio: Optional[str] = None  # Base64 encoded audio
    max_tokens: int = 2048
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from the model router."""
    content: str
    model_used: str
    provider: str
    capability: str
    tokens_used: int = 0
    latency_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DUAL AGENTIC BRIDGE
# ============================================================================

class DualAgenticBridge:
    """
    Bridge to the Dual Agentic System for multi-model capabilities.

    The Dual Agentic System provides:
    - Model routing (auto-select best model for task)
    - Vision processing (image understanding)
    - Voice processing (speech-to-text, text-to-speech)
    - Advanced reasoning (chain-of-thought, multi-step)
    - Safety validation (Ihsan >= 0.99 constraint)

    v1.1 Features:
    - Circuit breaker for LM Studio and Ollama
    - Automatic retry with exponential backoff
    - Seamless failover between backends
    """

    def __init__(self, base_url: str = DUAL_AGENTIC_URL, timeout: float = OLLAMA_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self.enabled = DUAL_AGENTIC_ENABLED
        self.ollama_enabled = OLLAMA_ENABLED
        self.ollama_url = OLLAMA_BASE_URL
        self._available = False
        self._ollama_available = False
        self._capabilities: List[str] = []
        self._models: List[str] = []
        self._ollama_models: List[str] = []

        # Circuit breakers for resilience (using config values)
        if RESILIENCE_AVAILABLE:
            self._lm_studio_breaker = CircuitBreaker(
                "lm_studio",
                CircuitBreakerConfig(
                    failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                    success_threshold=CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
                    timeout_seconds=CIRCUIT_BREAKER_TIMEOUT
                )
            )
            self._ollama_breaker = CircuitBreaker(
                "ollama_fallback",
                CircuitBreakerConfig(
                    failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD + 2,  # More lenient
                    success_threshold=CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
                    timeout_seconds=CIRCUIT_BREAKER_TIMEOUT * 2
                )
            )
        else:
            self._lm_studio_breaker = None
            self._ollama_breaker = None

        # Default model mappings for LM Studio (from config)
        self._model_mapping = {
            ModelCapability.TEXT: DEFAULT_TEXT_MODEL,
            ModelCapability.REASONING: DEFAULT_REASONING_MODEL,
            ModelCapability.CODE: DEFAULT_CODE_MODEL,
            ModelCapability.VISION: DEFAULT_VISION_MODEL,
            ModelCapability.MULTIMODAL: DEFAULT_VISION_MODEL,
        }

        # Ollama model mappings (fallback, from config)
        self._ollama_mapping = {
            ModelCapability.TEXT: OLLAMA_TEXT_MODEL,
            ModelCapability.REASONING: OLLAMA_TEXT_MODEL,
            ModelCapability.CODE: OLLAMA_CODE_MODEL,
            ModelCapability.VISION: OLLAMA_VISION_MODEL,
            ModelCapability.MULTIMODAL: OLLAMA_VISION_MODEL,
        }

    async def check_availability(self) -> Dict[str, bool]:
        """
        Check availability of all backends (LM Studio and Ollama).

        Returns:
            Dict with availability status for each backend
        """
        results = {"lm_studio": False, "ollama": False}

        # Check LM Studio
        if self.enabled:
            try:
                async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                    response = await client.get(LM_STUDIO_MODELS_ENDPOINT)
                    if response.status_code == 200:
                        data = response.json()
                        self._models = [m.get('id', '') for m in data.get('data', [])]
                        self._available = True
                        self._capabilities = ['text', 'reasoning', 'code']

                        # Check for vision models
                        vision_keywords = ['vision', 'llava', 'vl', 'bakllava', 'moondream']
                        vision_models = [m for m in self._models
                                        if any(kw in m.lower() for kw in vision_keywords)]
                        if vision_models:
                            self._capabilities.append('vision')
                            self._model_mapping[ModelCapability.VISION] = vision_models[0]
                            self._model_mapping[ModelCapability.MULTIMODAL] = vision_models[0]

                        print(f"[DUAL-AGENTIC] ✅ LM Studio: {len(self._models)} models")
                        results["lm_studio"] = True
            except Exception as e:
                print(f"[DUAL-AGENTIC] ❌ LM Studio unavailable: {e}")
                self._available = False

        # Check Ollama (always check as fallback)
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self._ollama_models = [m.get('name', '') for m in data.get('models', [])]
                    self._ollama_available = True

                    # Update Ollama model mappings based on available models
                    for model in self._ollama_models:
                        model_lower = model.lower()
                        if 'llava' in model_lower:
                            self._ollama_mapping[ModelCapability.VISION] = model
                            self._ollama_mapping[ModelCapability.MULTIMODAL] = model
                        elif 'code' in model_lower:
                            self._ollama_mapping[ModelCapability.CODE] = model
                        elif 'llama' in model_lower or 'mistral' in model_lower:
                            self._ollama_mapping[ModelCapability.TEXT] = model
                            self._ollama_mapping[ModelCapability.REASONING] = model

                    print(f"[DUAL-AGENTIC] ✅ Ollama: {len(self._ollama_models)} models")
                    results["ollama"] = True
        except Exception as e:
            print(f"[DUAL-AGENTIC] ❌ Ollama unavailable: {e}")
            self._ollama_available = False

        # Summary
        if results["lm_studio"]:
            print(f"[DUAL-AGENTIC] 📍 Primary: LM Studio @ {self.base_url}")
        if results["ollama"]:
            fallback_str = " (fallback)" if results["lm_studio"] else " (primary)"
            print(f"[DUAL-AGENTIC] 📍 Ollama{fallback_str} @ localhost:11434")

        if not any(results.values()):
            print("[DUAL-AGENTIC] ⚠️ No backends available! System will operate in degraded mode.")

        return results

    async def route_request(self, request: ModelRequest) -> Optional[ModelResponse]:
        """
        Route a request to the appropriate model with automatic failover.

        Priority:
        1. LM Studio (if available and circuit breaker closed)
        2. Ollama (fallback)

        The system automatically selects the best model based on:
        - Requested capability
        - Input modality
        - Backend availability
        - Circuit breaker state
        """
        # Check availability if needed
        if not self._available and not self._ollama_available:
            await self.check_availability()

        # Determine which backend to use
        use_lm_studio = self._available
        use_ollama = self._ollama_available

        # Check circuit breaker states
        if RESILIENCE_AVAILABLE and self._lm_studio_breaker:
            if self._lm_studio_breaker.state.value == "open":
                print("[DUAL-AGENTIC] ⚠️ LM Studio circuit breaker OPEN, trying Ollama")
                use_lm_studio = False

        if RESILIENCE_AVAILABLE and self._ollama_breaker:
            if self._ollama_breaker.state.value == "open":
                use_ollama = False

        # Try LM Studio first
        if use_lm_studio:
            response = await self._call_lm_studio(request)
            if response and not response.metadata.get('error'):
                return response
            # LM Studio failed, try Ollama
            print("[DUAL-AGENTIC] 🔄 LM Studio failed, falling back to Ollama")

        # Try Ollama
        if use_ollama:
            return await self._fallback_local(request)

        # Both failed
        return ModelResponse(
            content="[Error: All backends unavailable. Please check LM Studio or Ollama.]",
            model_used="none",
            provider="none",
            capability=request.capability.value,
            metadata={"error": True, "all_backends_failed": True}
        )

    async def _call_lm_studio(self, request: ModelRequest) -> Optional[ModelResponse]:
        """Call LM Studio backend with circuit breaker protection."""
        try:
            # Select model based on capability
            model = request.model_name or self._model_mapping.get(
                request.capability,
                self._model_mapping[ModelCapability.TEXT]
            )

            # Build messages in OpenAI format
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})

            # Handle multimodal content (vision)
            if request.images:
                content = [{"type": "text", "text": request.prompt}]
                for img in request.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"}
                    })
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": request.prompt})

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                start = datetime.now()
                response = await client.post(
                    LM_STUDIO_CHAT_ENDPOINT,
                    json=payload
                )
                latency = (datetime.now() - start).total_seconds() * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    usage = data.get('usage', {})

                    # Record success for circuit breaker
                    if RESILIENCE_AVAILABLE and self._lm_studio_breaker:
                        await self._lm_studio_breaker._record_success()

                    return ModelResponse(
                        content=content,
                        model_used=model,
                        provider='lm_studio',
                        capability=request.capability.value,
                        tokens_used=usage.get('total_tokens', 0),
                        latency_ms=latency,
                        metadata={
                            "prompt_tokens": usage.get('prompt_tokens', 0),
                            "completion_tokens": usage.get('completion_tokens', 0)
                        }
                    )
                else:
                    # Record failure for circuit breaker
                    if RESILIENCE_AVAILABLE and self._lm_studio_breaker:
                        await self._lm_studio_breaker._record_failure(
                            Exception(f"HTTP {response.status_code}")
                        )
                    return ModelResponse(
                        content="",
                        model_used=model,
                        provider='lm_studio',
                        capability=request.capability.value,
                        latency_ms=latency,
                        metadata={"error": True, "status_code": response.status_code}
                    )

        except Exception as e:
            print(f"[DUAL-AGENTIC] LM Studio request failed: {e}")
            # Record failure for circuit breaker
            if RESILIENCE_AVAILABLE and self._lm_studio_breaker:
                await self._lm_studio_breaker._record_failure(e)
            return ModelResponse(
                content="",
                model_used="unknown",
                provider='lm_studio',
                capability=request.capability.value,
                metadata={"error": True, "exception": str(e)}
            )

    async def _fallback_local(self, request: ModelRequest) -> Optional[ModelResponse]:
        """Fallback to local Ollama with circuit breaker protection."""
        try:
            # Select model based on capability using ollama mapping
            model = self._ollama_mapping.get(
                request.capability,
                self._ollama_mapping[ModelCapability.TEXT]
            )

            # Use chat endpoint for better compatibility
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})

            # Build user message
            user_message = {"role": "user", "content": request.prompt}
            if request.images:
                user_message["images"] = request.images
            messages.append(user_message)

            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                start = datetime.now()
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload
                )
                latency = (datetime.now() - start).total_seconds() * 1000

                if response.status_code == 200:
                    data = response.json()
                    content = data.get('message', {}).get('content', data.get('response', ''))

                    # Record success for circuit breaker
                    if RESILIENCE_AVAILABLE and self._ollama_breaker:
                        await self._ollama_breaker._record_success()

                    return ModelResponse(
                        content=content,
                        model_used=model,
                        provider='ollama',
                        capability=request.capability.value,
                        latency_ms=latency,
                        metadata={'fallback': True}
                    )

        except Exception as e:
            print(f"[DUAL-AGENTIC] Fallback also failed: {e}")

        return None

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    async def analyze_image(self, image_path: str, prompt: Optional[str] = None) -> Optional[str]:
        """
        Analyze an image using vision models.

        Args:
            image_path: Path to image file
            prompt: Optional specific question about the image

        Returns:
            Text analysis of the image
        """
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()

        default_prompt = "Analyze this image in detail. Describe what you see, any text content, diagrams, charts, or technical information."

        request = ModelRequest(
            prompt=prompt or default_prompt,
            capability=ModelCapability.VISION,
            images=[image_data]
        )

        response = await self.route_request(request)
        return response.content if response else None

    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using voice models.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode()

        request = ModelRequest(
            prompt="Transcribe this audio accurately.",
            capability=ModelCapability.VOICE,
            audio=audio_data
        )

        response = await self.route_request(request)
        return response.content if response else None

    async def reason(self, prompt: str, context: Optional[Dict] = None) -> Optional[str]:
        """
        Perform advanced reasoning with chain-of-thought.

        Args:
            prompt: The reasoning task
            context: Additional context for reasoning

        Returns:
            Reasoned response
        """
        system_prompt = """You are a sophisticated reasoning agent. Use step-by-step thinking.
Consider multiple perspectives. Validate your conclusions.
Maintain Ihsan (excellence) in your reasoning quality."""

        request = ModelRequest(
            prompt=prompt,
            capability=ModelCapability.REASONING,
            system_prompt=system_prompt,
            context=context or {},
            temperature=0.3  # Lower for more precise reasoning
        )

        response = await self.route_request(request)
        return response.content if response else None

    async def generate_code(self, prompt: str, language: str = "python") -> Optional[str]:
        """
        Generate code with the code-specialized model.

        Args:
            prompt: Description of code to generate
            language: Target programming language

        Returns:
            Generated code
        """
        request = ModelRequest(
            prompt=f"Generate {language} code: {prompt}",
            capability=ModelCapability.CODE,
            context={"language": language}
        )

        response = await self.route_request(request)
        return response.content if response else None

    async def multimodal_query(
        self,
        text: str,
        images: Optional[List[str]] = None,
        audio: Optional[str] = None
    ) -> Optional[str]:
        """
        Process a query with multiple modalities.

        Args:
            text: Text query
            images: List of image file paths
            audio: Audio file path

        Returns:
            Combined analysis
        """
        image_data = []
        if images:
            for img_path in images:
                with open(img_path, 'rb') as f:
                    image_data.append(base64.b64encode(f.read()).decode())

        audio_data = None
        if audio:
            with open(audio, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode()

        request = ModelRequest(
            prompt=text,
            capability=ModelCapability.MULTIMODAL,
            images=image_data,
            audio=audio_data
        )

        response = await self.route_request(request)
        return response.content if response else None

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            'enabled': self.enabled,
            'available': self._available,
            'base_url': self.base_url,
            'capabilities': self._capabilities,
            'timeout': self.timeout
        }


# ============================================================================
# KNOWLEDGE-ENHANCED MODEL ROUTER
# ============================================================================

class KnowledgeEnhancedRouter:
    """
    Routes model requests with knowledge from the Data Lake.
    Combines retrieval results with model capabilities.
    """

    def __init__(self, bridge: Optional[DualAgenticBridge] = None):
        self.bridge = bridge or DualAgenticBridge()

    async def query_with_knowledge(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        capability: ModelCapability = ModelCapability.REASONING
    ) -> Optional[str]:
        """
        Process a query with retrieved knowledge context.

        Args:
            query: User query
            retrieved_chunks: Chunks retrieved from Data Lake
            capability: Required model capability

        Returns:
            Knowledge-enhanced response
        """
        # Build context from retrieved chunks
        context_parts = []
        for chunk in retrieved_chunks[:5]:  # Top 5 chunks
            context_parts.append(f"[Source: {chunk.get('source', 'unknown')}]\n{chunk.get('text', '')}")

        knowledge_context = "\n\n---\n\n".join(context_parts)

        enhanced_prompt = f"""Based on the following knowledge context, answer the query.

KNOWLEDGE CONTEXT:
{knowledge_context}

QUERY: {query}

Provide a comprehensive answer based on the knowledge context. Cite sources when relevant."""

        request = ModelRequest(
            prompt=enhanced_prompt,
            capability=capability,
            system_prompt="You are a knowledge synthesis agent with access to the BIZRA Data Lake. Provide accurate, well-sourced responses."
        )

        response = await self.bridge.route_request(request)
        return response.content if response else None

    async def analyze_with_vision(
        self,
        image_path: str,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> Optional[str]:
        """
        Analyze an image with knowledge context.

        Args:
            image_path: Path to image
            query: Question about the image
            retrieved_chunks: Related knowledge from Data Lake

        Returns:
            Vision + knowledge enhanced response
        """
        # Build context
        context_parts = []
        for chunk in retrieved_chunks[:3]:
            context_parts.append(chunk.get('text', ''))

        context = "\n".join(context_parts)

        enhanced_prompt = f"""Analyze this image in the context of the following knowledge:

KNOWLEDGE CONTEXT:
{context}

QUESTION: {query}

Provide detailed analysis connecting the image to the knowledge context."""

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()

        request = ModelRequest(
            prompt=enhanced_prompt,
            capability=ModelCapability.VISION,
            images=[image_data]
        )

        response = await self.bridge.route_request(request)
        return response.content if response else None


# ============================================================================
# MAIN / DEMO
# ============================================================================

async def main():
    """Demonstrate Dual Agentic Bridge capabilities."""
    print("=" * 70)
    print("BIZRA Dual Agentic Bridge v1.0")
    print("=" * 70)

    bridge = DualAgenticBridge()

    # Check availability
    available = await bridge.check_availability()
    print(f"\nDual Agentic System Available: {available}")

    status = bridge.get_status()
    print("\nBridge Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    if available:
        print("\n" + "=" * 70)
        print("Testing Reasoning Capability")
        print("=" * 70)

        result = await bridge.reason(
            "What are the key principles for building a knowledge graph from unstructured data?"
        )
        if result:
            print(f"\nReasoning Result:\n{result[:500]}...")

    # Example usage
    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)
    print("""
    from dual_agentic_bridge import DualAgenticBridge, KnowledgeEnhancedRouter

    # Initialize bridge
    bridge = DualAgenticBridge()
    await bridge.check_availability()

    # Analyze an image
    analysis = await bridge.analyze_image("diagram.png",
        "What architecture pattern does this diagram show?")

    # Transcribe audio
    transcript = await bridge.transcribe_audio("meeting.mp3")

    # Advanced reasoning
    reasoning = await bridge.reason(
        "How should we integrate multi-modal search into our RAG pipeline?"
    )

    # Knowledge-enhanced query
    router = KnowledgeEnhancedRouter(bridge)
    result = await router.query_with_knowledge(
        query="What is the BIZRA architecture?",
        retrieved_chunks=[{"text": "BIZRA uses hypergraph RAG...", "source": "docs"}]
    )
    """)


if __name__ == "__main__":
    asyncio.run(main())
