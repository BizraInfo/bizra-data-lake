"""
BIZRA Sovereign LLM - LM Studio Backend Integration

Integrates with LM Studio's v1 REST API for local inference.
Supports model management, stateful chats, and MCP integration.

"Every model is welcome if they accept the rules of BIZRA."

Dependencies:
    pip install httpx
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.inference.response_utils import strip_think_tokens
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

logger = logging.getLogger(__name__)

# Optional dependency - httpx
_HTTPX_AVAILABLE = False
try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    logger.warning(
        "httpx not installed. LM Studio backend unavailable. Install with: pip install httpx"
    )
    httpx = None  # type: ignore

# Constitutional thresholds (from single source of truth)
IHSAN_THRESHOLD = UNIFIED_IHSAN_THRESHOLD
SNR_THRESHOLD = UNIFIED_SNR_THRESHOLD


class LMStudioEndpoint(Enum):
    """LM Studio v1 API endpoints."""

    CHAT = "/api/v1/chat"
    MODELS = "/api/v1/models"
    LOAD = "/api/v1/models/load"
    UNLOAD = "/api/v1/models/unload"
    DOWNLOAD = "/api/v1/models/download"
    DOWNLOAD_STATUS = "/api/v1/models/download/status"
    # OpenAI-compatible
    CHAT_COMPLETIONS = "/v1/chat/completions"
    RESPONSES = "/v1/responses"
    # Anthropic-compatible
    MESSAGES = "/v1/messages"


@dataclass
class LMStudioConfig:
    """Configuration for LM Studio backend."""

    host: str = field(
        default_factory=lambda: os.getenv("LMSTUDIO_HOST", "192.168.56.1")
    )
    port: int = field(default_factory=lambda: int(os.getenv("LMSTUDIO_PORT", "1234")))
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("LM_API_TOKEN")
        or os.getenv("LMSTUDIO_API_KEY")
        or os.getenv("LM_STUDIO_API_KEY")
    )
    timeout: float = 120.0
    use_native_api: bool = True  # Use /api/v1/* instead of OpenAI-compat
    context_length: Optional[int] = None
    enable_mcp: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    id: str
    name: str
    path: Optional[str] = None
    loaded: bool = False
    context_length: int = 4096
    parameters: Optional[int] = None
    quantization: Optional[str] = None


@dataclass
class ChatMessage:
    """Chat message structure."""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    """Response from chat endpoint."""

    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Optional[Dict] = None


class LMStudioBackend:
    """
    LM Studio v1 API Backend for BIZRA Sovereign LLM.

    Provides:
    - Model listing, loading, unloading, downloading
    - Native /api/v1/chat with stateful chats and MCP support
    - OpenAI-compatible /v1/chat/completions fallback
    - Streaming support

    Usage:
        backend = LMStudioBackend(LMStudioConfig())
        await backend.connect()

        # List models
        models = await backend.list_models()

        # Load a model
        await backend.load_model("lmstudio-community/Meta-Llama-3-8B-GGUF")

        # Chat
        response = await backend.chat([
            ChatMessage(role="user", content="Hello!")
        ])
    """

    def __init__(self, config: Optional[LMStudioConfig] = None):
        self.config = config or LMStudioConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False
        self._current_model: Optional[str] = None
        self._chat_id: Optional[str] = None  # For native /api/v1/chat stateful sessions
        self._response_id: Optional[str] = None  # For /v1/responses stateful sessions

    def _require_client(self) -> "httpx.AsyncClient":
        """Return the HTTP client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Not connected to LM Studio")
        return self._client

    async def connect(self) -> bool:
        """Connect to LM Studio server."""
        try:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )

            # Test connection by listing models
            response = await self._client.get(LMStudioEndpoint.MODELS.value)
            response.raise_for_status()

            self._connected = True
            logger.info(f"Connected to LM Studio at {self.config.base_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to LM Studio: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from LM Studio."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False

    async def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        response = await self._require_client().get(LMStudioEndpoint.MODELS.value)
        response.raise_for_status()
        data = response.json()

        models = []
        # v1 API uses "models" key; OpenAI-compat uses "data"
        model_list = data.get("models", data.get("data", []))
        for model_data in model_list:
            # v1 API: loaded_instances list; OpenAI-compat: "loaded" bool
            loaded_instances = model_data.get("loaded_instances", [])
            is_loaded = bool(loaded_instances) or model_data.get("loaded", False)
            # v1 API uses "key" for model ID
            model_id = model_data.get("key", model_data.get("id", ""))
            ctx_len = model_data.get(
                "max_context_length", model_data.get("context_length", 4096)
            )
            # If loaded, use the instance's context_length if available
            if loaded_instances:
                inst_ctx = loaded_instances[0].get("config", {}).get("context_length")
                if inst_ctx:
                    ctx_len = inst_ctx
            models.append(
                ModelInfo(
                    id=model_id,
                    name=model_data.get("display_name", model_id.split("/")[-1]),
                    path=model_data.get("path"),
                    loaded=is_loaded,
                    context_length=ctx_len,
                    quantization=(
                        model_data.get("quantization", {}).get("name")
                        if isinstance(model_data.get("quantization"), dict)
                        else None
                    ),
                )
            )

        return models

    async def load_model(
        self,
        model_id: str,
        context_length: Optional[int] = None,
        gpu_layers: Optional[int] = None,
    ) -> bool:
        """
        Load a model into LM Studio.

        Args:
            model_id: Model identifier (e.g., "lmstudio-community/Meta-Llama-3-8B-GGUF")
            context_length: Optional context window size
            gpu_layers: Optional number of GPU layers (-1 for all)
        """
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        payload: Dict[str, Any] = {"model": model_id}

        if context_length:
            payload["context_length"] = context_length
        if gpu_layers is not None:
            payload["gpu_layers"] = gpu_layers

        response = await self._require_client().post(
            LMStudioEndpoint.LOAD.value, json=payload
        )
        response.raise_for_status()

        self._current_model = model_id
        logger.info(f"Loaded model: {model_id}")
        return True

    async def unload_model(self, model_id: Optional[str] = None) -> bool:
        """Unload a model from LM Studio."""
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        model_to_unload = model_id or self._current_model
        if not model_to_unload:
            raise ValueError("No model specified to unload")

        response = await self._require_client().post(
            LMStudioEndpoint.UNLOAD.value, json={"model": model_to_unload}
        )
        response.raise_for_status()

        if model_to_unload == self._current_model:
            self._current_model = None

        logger.info(f"Unloaded model: {model_to_unload}")
        return True

    async def download_model(
        self, model_id: str, quantization: Optional[str] = None
    ) -> str:
        """
        Download a model from Hugging Face.

        Returns download task ID for status tracking.
        """
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        payload = {"model": model_id}
        if quantization:
            payload["quantization"] = quantization

        response = await self._require_client().post(
            LMStudioEndpoint.DOWNLOAD.value, json=payload
        )
        response.raise_for_status()
        data = response.json()

        task_id = data.get("task_id", "")
        logger.info(f"Started download for {model_id}, task_id: {task_id}")
        return task_id

    async def get_download_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a download task."""
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        response = await self._require_client().get(
            LMStudioEndpoint.DOWNLOAD_STATUS.value, params={"task_id": task_id}
        )
        response.raise_for_status()
        return response.json()

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        context_length: Optional[int] = None,
        mcp_servers: Optional[List[str]] = None,
    ) -> ChatResponse:
        """
        Send a chat request using native /api/v1/chat endpoint.

        Features:
        - Stateful chats (maintains conversation context)
        - MCP integration
        - Model load streaming events
        - Prompt processing streaming events
        - Custom context length
        """
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        model_id = model or self._current_model
        if not model_id:
            raise ValueError("No model specified or loaded")

        if self.config.use_native_api:
            # v1 Native API: uses "input" string, returns "output" array
            # Combine messages into single input string
            input_text = "\n".join(
                f"{m.role}: {m.content}" if m.role != "user" else m.content
                for m in messages
            )
            if len(messages) == 1:
                input_text = messages[0].content

            payload: Dict[str, Any] = {
                "model": model_id,
                "input": input_text,
                "temperature": temperature,
            }

            # v1 API uses context_length (not max_tokens)
            payload["context_length"] = (
                context_length or self.config.context_length or 8000
            )

            # MCP integrations
            if mcp_servers and self.config.enable_mcp:
                payload["integrations"] = mcp_servers

            # Stateful chat continuation
            if self._chat_id:
                payload["chat_id"] = self._chat_id

            if stream:
                payload["stream"] = True
                return self._stream_chat(payload)  # type: ignore[return-value]

            response = await self._require_client().post(
                LMStudioEndpoint.CHAT.value, json=payload
            )
            response.raise_for_status()
            data = response.json()

            # Store response_id for stateful conversations
            if "response_id" in data:
                self._response_id = data["response_id"]

            # Extract content from v1 output array
            content_parts = []
            for output_item in data.get("output", []):
                if output_item.get("type") == "message":
                    content_parts.append(output_item.get("content", ""))

            content = "\n".join(content_parts) if content_parts else ""
            content = strip_think_tokens(content)

            # Extract stats
            stats = data.get("stats", {})
            usage = {
                "input_tokens": stats.get("input_tokens", 0),
                "output_tokens": stats.get("total_output_tokens", 0),
                "reasoning_tokens": stats.get("reasoning_output_tokens", 0),
            }

            return ChatResponse(
                content=content,
                model=data.get("model_instance_id", model_id),
                finish_reason="stop",
                usage=usage,
                raw_response=data,
            )
        else:
            # OpenAI-compat fallback: uses "messages" array, returns "choices"
            payload = {
                "model": model_id,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

            if context_length:
                payload["context_length"] = context_length

            if self._chat_id:
                payload["chat_id"] = self._chat_id

            if stream:
                return self._stream_chat(payload)  # type: ignore[return-value]

            response = await self._require_client().post(
                "/v1/chat/completions", json=payload
            )
            response.raise_for_status()
            data = response.json()

            if "chat_id" in data:
                self._chat_id = data["chat_id"]

            raw_content = (
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            return ChatResponse(
                content=strip_think_tokens(raw_content),
                model=data.get("model", model_id),
                finish_reason=data.get("choices", [{}])[0].get("finish_reason", "stop"),
                usage=data.get("usage", {}),
                raw_response=data,
            )

    async def _stream_chat(self, payload: Dict) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        async with self._require_client().stream(
            "POST", LMStudioEndpoint.CHAT.value, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    async def chat_openai_compat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
    ) -> ChatResponse:
        """
        Send a chat request using OpenAI-compatible endpoint.

        Use this for custom tools support.
        """
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        model_id = model or self._current_model
        if not model_id:
            raise ValueError("No model specified or loaded")

        payload = {
            "model": model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        response = await self._require_client().post(
            LMStudioEndpoint.CHAT_COMPLETIONS.value, json=payload
        )
        response.raise_for_status()
        data = response.json()

        return ChatResponse(
            content=data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            model=data.get("model", model_id),
            finish_reason=data.get("choices", [{}])[0].get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def chat_responses(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        previous_response_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Send a chat request using OpenAI-compatible /v1/responses endpoint.

        LM Studio 0.4.0+ feature - combines OpenAI compatibility with:
        - Stateful chats (via previous_response_id)
        - Remote MCP server support
        - Local MCP servers configured in LM Studio
        - Custom tools

        Args:
            messages: Chat messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Custom tool definitions (OpenAI format)
            mcp_servers: Remote MCP server configurations
            previous_response_id: ID from previous response for stateful chat

        Returns:
            ChatResponse with response_id for continuation
        """
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        model_id = model or self._current_model
        if not model_id:
            raise ValueError("No model specified or loaded")

        payload = {
            "model": model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Stateful chat continuation
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        # Custom tools (OpenAI format)
        if tools:
            payload["tools"] = tools

        # Remote MCP servers (LM Studio 0.4.0+)
        if mcp_servers and self.config.enable_mcp:
            payload["mcp_servers"] = mcp_servers

        response = await self._require_client().post(
            LMStudioEndpoint.RESPONSES.value, json=payload
        )
        response.raise_for_status()
        data = response.json()

        # Store response_id for stateful continuation
        response_id = data.get("id")
        if response_id:
            self._response_id = response_id

        return ChatResponse(
            content=data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            model=data.get("model", model_id),
            finish_reason=data.get("choices", [{}])[0].get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            raw_response=data,
        )

    async def chat_anthropic_compat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
    ) -> ChatResponse:
        """
        Send a chat request using Anthropic-compatible /v1/messages endpoint.

        LM Studio 0.4.0+ feature for Anthropic SDK compatibility.
        Supports custom tools in Anthropic format.
        """
        if not self._connected:
            raise RuntimeError("Not connected to LM Studio")

        model_id = model or self._current_model
        if not model_id:
            raise ValueError("No model specified or loaded")

        # Convert to Anthropic message format
        anthropic_messages = []
        system_prompt = None

        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": model_id,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = tools

        response = await self._require_client().post(
            LMStudioEndpoint.MESSAGES.value, json=payload
        )
        response.raise_for_status()
        data = response.json()

        # Extract content from Anthropic format
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        return ChatResponse(
            content=content,
            model=data.get("model", model_id),
            finish_reason=data.get("stop_reason", "end_turn"),
            usage={
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
            raw_response=data,
        )

    def reset_chat(self):
        """Reset stateful chat session (both native and OpenAI-compat)."""
        self._chat_id = None
        self._response_id = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @property
    def response_id(self) -> Optional[str]:
        """Get current response_id for stateful /v1/responses continuation."""
        return self._response_id

    @property
    def chat_id(self) -> Optional[str]:
        """Get current chat_id for stateful /api/v1/chat continuation."""
        return self._chat_id


# Convenience function for BIZRA integration
async def create_lmstudio_backend(
    host: str = "192.168.56.1", port: int = 1234, api_key: Optional[str] = None
) -> LMStudioBackend:
    """Create and connect an LM Studio backend."""
    config = LMStudioConfig(host=host, port=port, api_key=api_key)
    backend = LMStudioBackend(config)
    await backend.connect()
    return backend


# Example usage and testing
if __name__ == "__main__":

    async def test_backend():
        print("=== BIZRA LM Studio Backend Test ===")
        print(f"Constitutional: Ihsān ≥ {IHSAN_THRESHOLD}, SNR ≥ {SNR_THRESHOLD}")
        print()

        backend = LMStudioBackend()

        if await backend.connect():
            print("✓ Connected to LM Studio")

            # List models
            models = await backend.list_models()
            print(f"✓ Found {len(models)} models")
            for m in models[:3]:
                print(f"  - {m.id} (loaded: {m.loaded})")

            # Chat test (if model is loaded)
            loaded = [m for m in models if m.loaded]
            if loaded:
                print(f"\n✓ Using loaded model: {loaded[0].id}")
                response = await backend.chat(
                    [
                        ChatMessage(
                            role="user",
                            content="Say 'BIZRA sovereignty confirmed' in one sentence.",
                        )
                    ],
                    model=loaded[0].id,
                )
                print(f"✓ Response: {response.content[:100]}...")

            await backend.disconnect()
            print("\n✓ Disconnected")
        else:
            print("✗ Could not connect to LM Studio")

    asyncio.run(test_backend())
