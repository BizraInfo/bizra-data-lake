"""
BIZRA OLLAMA BACKEND
═══════════════════════════════════════════════════════════════════════════════

Ollama backend for fallback inference.
Requires Ollama server running externally.
"""

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

from .base import InferenceBackend, InferenceBackendBase

if TYPE_CHECKING:
    from ..gateway import InferenceConfig


class OllamaBackend(InferenceBackendBase):
    """
    Ollama backend for fallback inference.

    Requires Ollama server running externally.
    """

    def __init__(self, config: "InferenceConfig"):
        self.config = config
        self._available_models: List[str] = []
        self._current_model: Optional[str] = None

    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.OLLAMA

    async def initialize(self) -> bool:
        """Check Ollama availability and list models."""
        try:
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                data = json.loads(resp.read().decode())
                self._available_models = [m["name"] for m in data.get("models", [])]

                if self._available_models:
                    self._current_model = self._available_models[0]
                    print(f"[Ollama] Available models: {self._available_models}")
                    return True
                else:
                    print("[Ollama] No models available")
                    return False

        except Exception as e:
            print(f"[Ollama] Not available: {e}")
            return False

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate via Ollama API."""
        # Separate system prompt from user query if delimiter present
        system_prompt = None
        user_prompt = prompt
        if "--- QUERY ---" in prompt:
            parts = prompt.split("--- QUERY ---", 1)
            system_prompt = parts[0].strip()
            user_prompt = parts[1].strip()

        payload_dict = {
            "model": self._current_model,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload_dict["system"] = system_prompt

        payload = json.dumps(payload_dict).encode()

        req = urllib.request.Request(
            f"{self.config.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=180) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
            data = json.loads(resp.read().decode())
            return data.get("response", "")

    async def generate_stream(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate with streaming via Ollama API."""
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield response

    async def health_check(self) -> bool:
        """Check Ollama health."""
        try:
            req = urllib.request.Request(f"{self.config.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                return resp.status == 200
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        return self._current_model
