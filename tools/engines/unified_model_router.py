# BIZRA Unified Model Router v1.0
# Seamless routing between LM Studio, Ollama, and Cloud backends
# Supports: Text reasoning, Vision models, Auto-failover, Offline operation

import asyncio
import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
import httpx

# Configuration
LM_STUDIO_URL = "http://192.168.56.1:1234/v1"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0
VISION_TIMEOUT = 180.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BIZRA.ModelRouter")


class ModelCapability(Enum):
    """Model capabilities"""
    TEXT = "text"
    VISION = "vision"
    CODE = "code"
    REASONING = "reasoning"
    EMBEDDING = "embedding"


class BackendType(Enum):
    """Available backend types"""
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class BackendStatus(Enum):
    """Backend health status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Information about an available model"""
    id: str
    name: str
    backend: BackendType
    capabilities: List[ModelCapability]
    context_length: int = 4096
    is_local: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True


@dataclass
class BackendHealth:
    """Backend health information"""
    backend: BackendType
    status: BackendStatus
    latency_ms: float = 0.0
    available_models: List[str] = field(default_factory=list)
    last_check: str = ""
    error_message: str = ""


@dataclass
class ChatMessage:
    """Chat message with optional image support"""
    role: str  # system, user, assistant
    content: str
    images: List[str] = field(default_factory=list)  # Base64 encoded images
    image_paths: List[str] = field(default_factory=list)  # Local file paths


@dataclass
class ModelResponse:
    """Response from model"""
    content: str
    model: str
    backend: BackendType
    latency_ms: float
    tokens_used: int = 0
    finish_reason: str = "stop"
    metadata: Dict = field(default_factory=dict)


class ModelBackend(ABC):
    """Abstract base class for model backends"""

    def __init__(self, name: str, base_url: str, timeout: float = DEFAULT_TIMEOUT):
        self.name = name
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._available_models: List[ModelInfo] = []
        self._is_healthy = False

    @abstractmethod
    async def health_check(self) -> BackendHealth:
        """Check backend health and list available models"""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate response from model"""
        pass

    @abstractmethod
    async def generate_with_vision(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate response with vision capability"""
        pass

    def get_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        return self._available_models

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class LMStudioBackend(ModelBackend):
    """
    LM Studio Backend

    Connects to LM Studio server for multi-model inference.
    Supports both text and vision models via OpenAI-compatible API.
    """

    # Known vision-capable models in LM Studio
    VISION_MODELS = [
        "llava", "bakllava", "moondream", "cogvlm", "qwen-vl",
        "minicpm-v", "internvl", "phi-3-vision"
    ]

    def __init__(self, base_url: str = LM_STUDIO_URL):
        super().__init__("LM Studio", base_url, timeout=VISION_TIMEOUT)

    async def health_check(self) -> BackendHealth:
        """Check LM Studio health"""
        start = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/models")
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])

                self._available_models = []
                model_ids = []

                for model in models:
                    model_id = model.get("id", "")
                    model_ids.append(model_id)

                    # Detect vision capability
                    is_vision = any(vm in model_id.lower() for vm in self.VISION_MODELS)

                    capabilities = [ModelCapability.TEXT, ModelCapability.REASONING]
                    if is_vision:
                        capabilities.append(ModelCapability.VISION)
                    if "code" in model_id.lower():
                        capabilities.append(ModelCapability.CODE)

                    self._available_models.append(ModelInfo(
                        id=model_id,
                        name=model_id,
                        backend=BackendType.LM_STUDIO,
                        capabilities=capabilities,
                        is_local=True,
                        supports_vision=is_vision
                    ))

                self._is_healthy = True
                return BackendHealth(
                    backend=BackendType.LM_STUDIO,
                    status=BackendStatus.ONLINE,
                    latency_ms=latency,
                    available_models=model_ids,
                    last_check=time.strftime("%Y-%m-%d %H:%M:%S")
                )

            self._is_healthy = False
            return BackendHealth(
                backend=BackendType.LM_STUDIO,
                status=BackendStatus.OFFLINE,
                error_message=f"HTTP {response.status_code}"
            )

        except Exception as e:
            self._is_healthy = False
            return BackendHealth(
                backend=BackendType.LM_STUDIO,
                status=BackendStatus.OFFLINE,
                error_message=str(e)[:100]
            )

    async def generate(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate text response"""
        start = time.time()

        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": openai_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})

                return ModelResponse(
                    content=content,
                    model=model,
                    backend=BackendType.LM_STUDIO,
                    latency_ms=latency,
                    tokens_used=usage.get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop")
                )

            return ModelResponse(
                content=f"[Error: LM Studio returned {response.status_code}]",
                model=model,
                backend=BackendType.LM_STUDIO,
                latency_ms=latency,
                metadata={"error": True}
            )

        except Exception as e:
            return ModelResponse(
                content=f"[Error: {str(e)}]",
                model=model,
                backend=BackendType.LM_STUDIO,
                latency_ms=(time.time() - start) * 1000,
                metadata={"error": True, "exception": str(e)}
            )

    async def generate_with_vision(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate response with vision capability"""
        start = time.time()

        # Build messages with image content
        openai_messages = []

        for msg in messages:
            if msg.images or msg.image_paths:
                # Multi-modal message
                content_parts = [{"type": "text", "text": msg.content}]

                # Add base64 images
                for img_b64 in msg.images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    })

                # Load and encode image paths
                for img_path in msg.image_paths:
                    try:
                        with open(img_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {e}")

                openai_messages.append({
                    "role": msg.role,
                    "content": content_parts
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": openai_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")

                return ModelResponse(
                    content=content,
                    model=model,
                    backend=BackendType.LM_STUDIO,
                    latency_ms=latency,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
                    metadata={"vision": True}
                )

            return ModelResponse(
                content=f"[Error: Vision request failed with {response.status_code}]",
                model=model,
                backend=BackendType.LM_STUDIO,
                latency_ms=latency,
                metadata={"error": True, "vision": True}
            )

        except Exception as e:
            return ModelResponse(
                content=f"[Error: {str(e)}]",
                model=model,
                backend=BackendType.LM_STUDIO,
                latency_ms=(time.time() - start) * 1000,
                metadata={"error": True, "vision": True}
            )


class OllamaBackend(ModelBackend):
    """
    Ollama Backend

    Connects to local Ollama instance for offline inference.
    Supports text and vision models (llava, bakllava, etc.)
    """

    VISION_MODELS = ["llava", "bakllava", "moondream", "llava-phi3"]

    def __init__(self, base_url: str = OLLAMA_URL):
        super().__init__("Ollama", base_url, timeout=DEFAULT_TIMEOUT)

    async def health_check(self) -> BackendHealth:
        """Check Ollama health"""
        start = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                self._available_models = []
                model_ids = []

                for model in models:
                    model_name = model.get("name", "")
                    model_ids.append(model_name)

                    is_vision = any(vm in model_name.lower() for vm in self.VISION_MODELS)

                    capabilities = [ModelCapability.TEXT, ModelCapability.REASONING]
                    if is_vision:
                        capabilities.append(ModelCapability.VISION)
                    if "code" in model_name.lower():
                        capabilities.append(ModelCapability.CODE)

                    self._available_models.append(ModelInfo(
                        id=model_name,
                        name=model_name,
                        backend=BackendType.OLLAMA,
                        capabilities=capabilities,
                        is_local=True,
                        supports_vision=is_vision
                    ))

                self._is_healthy = True
                return BackendHealth(
                    backend=BackendType.OLLAMA,
                    status=BackendStatus.ONLINE,
                    latency_ms=latency,
                    available_models=model_ids,
                    last_check=time.strftime("%Y-%m-%d %H:%M:%S")
                )

            self._is_healthy = False
            return BackendHealth(
                backend=BackendType.OLLAMA,
                status=BackendStatus.OFFLINE,
                error_message=f"HTTP {response.status_code}"
            )

        except Exception as e:
            self._is_healthy = False
            return BackendHealth(
                backend=BackendType.OLLAMA,
                status=BackendStatus.OFFLINE,
                error_message=str(e)[:100]
            )

    async def generate(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate text response"""
        start = time.time()

        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                }
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")

                return ModelResponse(
                    content=content,
                    model=model,
                    backend=BackendType.OLLAMA,
                    latency_ms=latency,
                    tokens_used=data.get("eval_count", 0),
                    finish_reason="stop"
                )

            return ModelResponse(
                content=f"[Error: Ollama returned {response.status_code}]",
                model=model,
                backend=BackendType.OLLAMA,
                latency_ms=latency,
                metadata={"error": True}
            )

        except Exception as e:
            return ModelResponse(
                content=f"[Error: {str(e)}]",
                model=model,
                backend=BackendType.OLLAMA,
                latency_ms=(time.time() - start) * 1000,
                metadata={"error": True}
            )

    async def generate_with_vision(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """Generate response with vision (Ollama format)"""
        start = time.time()

        # Build Ollama vision request
        ollama_messages = []

        for msg in messages:
            ollama_msg = {"role": msg.role, "content": msg.content}

            # Add images for Ollama vision
            images = []

            for img_b64 in msg.images:
                images.append(img_b64)

            for img_path in msg.image_paths:
                try:
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    images.append(img_data)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")

            if images:
                ollama_msg["images"] = images

            ollama_messages.append(ollama_msg)

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                },
                timeout=VISION_TIMEOUT
            )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")

                return ModelResponse(
                    content=content,
                    model=model,
                    backend=BackendType.OLLAMA,
                    latency_ms=latency,
                    tokens_used=data.get("eval_count", 0),
                    metadata={"vision": True}
                )

            return ModelResponse(
                content=f"[Error: Vision request failed]",
                model=model,
                backend=BackendType.OLLAMA,
                latency_ms=latency,
                metadata={"error": True, "vision": True}
            )

        except Exception as e:
            return ModelResponse(
                content=f"[Error: {str(e)}]",
                model=model,
                backend=BackendType.OLLAMA,
                latency_ms=(time.time() - start) * 1000,
                metadata={"error": True, "vision": True}
            )


class UnifiedModelRouter:
    """
    Unified Model Router

    Intelligently routes requests between LM Studio and Ollama:
    - Automatic backend discovery
    - Health monitoring with failover
    - Vision model detection
    - Offline-first operation
    """

    def __init__(
        self,
        lm_studio_url: str = LM_STUDIO_URL,
        ollama_url: str = OLLAMA_URL,
        prefer_local: bool = True
    ):
        self.backends: Dict[BackendType, ModelBackend] = {
            BackendType.LM_STUDIO: LMStudioBackend(lm_studio_url),
            BackendType.OLLAMA: OllamaBackend(ollama_url)
        }
        self.prefer_local = prefer_local
        self.backend_health: Dict[BackendType, BackendHealth] = {}
        self._initialized = False
        self._primary_backend: Optional[BackendType] = None
        self._fallback_backend: Optional[BackendType] = None

    async def initialize(self) -> Dict[BackendType, BackendHealth]:
        """Initialize router and check all backends"""
        logger.info("🔄 Initializing Unified Model Router...")

        # Check all backends
        for backend_type, backend in self.backends.items():
            health = await backend.health_check()
            self.backend_health[backend_type] = health

            status_symbol = "✅" if health.status == BackendStatus.ONLINE else "❌"
            logger.info(
                f"  {status_symbol} {backend_type.value}: {health.status.value} "
                f"({len(health.available_models)} models)"
            )

        # Determine primary and fallback
        online_backends = [
            bt for bt, h in self.backend_health.items()
            if h.status == BackendStatus.ONLINE
        ]

        if BackendType.LM_STUDIO in online_backends:
            self._primary_backend = BackendType.LM_STUDIO
            if BackendType.OLLAMA in online_backends:
                self._fallback_backend = BackendType.OLLAMA
        elif BackendType.OLLAMA in online_backends:
            self._primary_backend = BackendType.OLLAMA
            self._fallback_backend = None

        if self._primary_backend:
            logger.info(f"  📍 Primary backend: {self._primary_backend.value}")
            if self._fallback_backend:
                logger.info(f"  📍 Fallback backend: {self._fallback_backend.value}")
        else:
            logger.warning("  ⚠️ No backends available!")

        self._initialized = True
        return self.backend_health

    def get_available_models(
        self,
        capability: Optional[ModelCapability] = None
    ) -> List[ModelInfo]:
        """Get all available models, optionally filtered by capability"""
        models = []

        for backend in self.backends.values():
            for model in backend.get_models():
                if capability is None or capability in model.capabilities:
                    models.append(model)

        return models

    def get_vision_models(self) -> List[ModelInfo]:
        """Get all vision-capable models"""
        return self.get_available_models(ModelCapability.VISION)

    def get_reasoning_models(self) -> List[ModelInfo]:
        """Get all reasoning models"""
        return self.get_available_models(ModelCapability.REASONING)

    async def _select_backend(
        self,
        preferred: Optional[BackendType] = None,
        require_vision: bool = False
    ) -> Tuple[ModelBackend, str]:
        """Select best available backend"""

        # If specific backend requested and available
        if preferred and preferred in self.backends:
            backend = self.backends[preferred]
            health = self.backend_health.get(preferred)
            if health and health.status == BackendStatus.ONLINE:
                # Find suitable model
                models = backend.get_models()
                if require_vision:
                    models = [m for m in models if m.supports_vision]
                if models:
                    return backend, models[0].id

        # Try primary backend
        if self._primary_backend:
            backend = self.backends[self._primary_backend]
            models = backend.get_models()
            if require_vision:
                models = [m for m in models if m.supports_vision]
            if models:
                return backend, models[0].id

        # Try fallback
        if self._fallback_backend:
            backend = self.backends[self._fallback_backend]
            models = backend.get_models()
            if require_vision:
                models = [m for m in models if m.supports_vision]
            if models:
                return backend, models[0].id

        raise RuntimeError("No suitable backend available")

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        backend: Optional[BackendType] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ModelResponse:
        """
        Generate response using best available backend.

        Args:
            messages: Chat messages
            model: Specific model to use (auto-selected if None)
            backend: Preferred backend (auto-selected if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        if not self._initialized:
            await self.initialize()

        # Check if any message has images
        has_images = any(msg.images or msg.image_paths for msg in messages)

        try:
            # Select backend and model
            selected_backend, selected_model = await self._select_backend(
                preferred=backend,
                require_vision=has_images
            )

            if model:
                selected_model = model

            logger.info(
                f"🤖 Routing to {selected_backend.name} "
                f"(model: {selected_model}, vision: {has_images})"
            )

            # Generate response
            if has_images:
                response = await selected_backend.generate_with_vision(
                    messages, selected_model, temperature, max_tokens, **kwargs
                )
            else:
                response = await selected_backend.generate(
                    messages, selected_model, temperature, max_tokens, **kwargs
                )

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")

            # Try failover
            if self._fallback_backend:
                logger.info(f"🔄 Attempting failover to {self._fallback_backend.value}")
                try:
                    fallback = self.backends[self._fallback_backend]
                    models = fallback.get_models()
                    if has_images:
                        models = [m for m in models if m.supports_vision]

                    if models:
                        if has_images:
                            return await fallback.generate_with_vision(
                                messages, models[0].id, temperature, max_tokens
                            )
                        else:
                            return await fallback.generate(
                                messages, models[0].id, temperature, max_tokens
                            )
                except Exception as fallback_error:
                    logger.error(f"Failover also failed: {fallback_error}")

            return ModelResponse(
                content=f"[Error: All backends failed. {str(e)}]",
                model="none",
                backend=BackendType.OLLAMA,
                latency_ms=0,
                metadata={"error": True}
            )

    async def generate_with_vision(
        self,
        prompt: str,
        image_paths: List[str] = None,
        image_base64: List[str] = None,
        model: Optional[str] = None,
        backend: Optional[BackendType] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Convenience method for vision queries.

        Args:
            prompt: Text prompt
            image_paths: List of local image file paths
            image_base64: List of base64-encoded images
            model: Specific model (auto-selected if None)
            backend: Preferred backend
        """
        messages = [
            ChatMessage(
                role="user",
                content=prompt,
                images=image_base64 or [],
                image_paths=image_paths or []
            )
        ]

        return await self.generate(messages, model, backend, **kwargs)

    async def close(self):
        """Close all backend connections"""
        for backend in self.backends.values():
            await backend.close()

    def print_status(self):
        """Print router status"""
        print()
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 14 + "UNIFIED MODEL ROUTER STATUS" + " " * 17 + "║")
        print("╠" + "═" * 58 + "╣")

        for bt, health in self.backend_health.items():
            symbol = "✅" if health.status == BackendStatus.ONLINE else "❌"
            primary = " (PRIMARY)" if bt == self._primary_backend else ""
            fallback = " (FALLBACK)" if bt == self._fallback_backend else ""

            print(f"║  {symbol} {bt.value:<12} {health.status.value:<10}{primary}{fallback}")
            print(f"║     Models: {len(health.available_models)}")
            if health.latency_ms > 0:
                print(f"║     Latency: {health.latency_ms:.1f}ms")

        # List vision models
        vision_models = self.get_vision_models()
        print("╠" + "═" * 58 + "╣")
        print("║  🖼️  VISION MODELS" + " " * 39 + "║")
        if vision_models:
            for m in vision_models[:5]:
                print(f"║     • {m.id[:45]:<45}  ║")
        else:
            print("║     (none available)" + " " * 36 + "║")

        print("╚" + "═" * 58 + "╝")
        print()


# Singleton instance
_router: Optional[UnifiedModelRouter] = None


async def get_router() -> UnifiedModelRouter:
    """Get or create the global model router"""
    global _router
    if _router is None:
        _router = UnifiedModelRouter()
        await _router.initialize()
    return _router


# Convenience functions
async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> str:
    """Quick text generation"""
    router = await get_router()

    messages = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=prompt))

    response = await router.generate(messages, model, **kwargs)
    return response.content


async def generate_with_image(
    prompt: str,
    image_path: str,
    model: Optional[str] = None,
    **kwargs
) -> str:
    """Quick vision generation"""
    router = await get_router()
    response = await router.generate_with_vision(
        prompt, image_paths=[image_path], model=model, **kwargs
    )
    return response.content


# Main execution
async def main():
    print("🚀 BIZRA Unified Model Router v1.0")
    print("=" * 50)

    router = UnifiedModelRouter()
    await router.initialize()

    router.print_status()

    # Test text generation
    print("\n--- Testing Text Generation ---")
    response = await router.generate([
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is BIZRA Data Lake?")
    ])

    print(f"Backend: {response.backend.value}")
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Response: {response.content[:200]}...")

    # List vision models
    print("\n--- Vision Models Available ---")
    for model in router.get_vision_models():
        print(f"  • {model.id} ({model.backend.value})")

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
