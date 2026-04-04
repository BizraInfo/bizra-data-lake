"""
BIZRA Direct LLM Client - Lightweight wrapper for LM Studio/Ollama

Bypasses RLM's full REPL system for faster, simpler completions.
Use BizraRLMAdapter for full recursive reasoning capabilities.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.56.1:1234/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Model slots aligned with Blueprint - optimized for speed/quality balance
MODEL_SLOTS = {
    # Core reasoning — Ollama model names (fallback: override via BIZRA_MODEL_<SLOT>)
    "cold_core": os.environ.get("BIZRA_MODEL_COLD_CORE", "llama3.1:8b"),
    "warm_surface": os.environ.get("BIZRA_MODEL_WARM_SURFACE", "qwen2.5:3b"),
    "primary_reasoning": os.environ.get("BIZRA_MODEL_PRIMARY", "llama3.1:8b"),
    # Extended CoT
    "thinking": os.environ.get("BIZRA_MODEL_THINKING", "deepseek-r1:14b"),
    # Vision (multimodal) — not available on Ollama, fall back to text
    "vision": os.environ.get("BIZRA_MODEL_VISION", "llama3.1:8b"),
    "vision_hq": os.environ.get("BIZRA_MODEL_VISION_HQ", "llama3.1:8b"),
    # Speed optimized
    "fast": os.environ.get("BIZRA_MODEL_FAST", "phi3:mini"),
    "nano": os.environ.get("BIZRA_MODEL_NANO", "phi3:mini"),
    "liquid": os.environ.get("BIZRA_MODEL_LIQUID", "qwen2.5:3b"),
    # Heavy reasoning
    "deep": os.environ.get("BIZRA_MODEL_DEEP", "deepseek-r1:14b"),
    "reasoning_hq": os.environ.get("BIZRA_MODEL_REASONING_HQ", "mistral:latest"),
    # Embeddings
    "embeddings": os.environ.get("BIZRA_MODEL_EMBEDDINGS", "nomic-embed-text:latest"),
}


@dataclass
class CompletionResult:
    """Result from direct LLM completion."""

    response: str
    model: str
    execution_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    ihsan_score: float = 0.95
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class BizraDirectClient:
    """
    Direct client for LM Studio / Ollama.

    Features:
    - Simple chat completions
    - Timeout handling
    - Ihsan scoring
    - Provider fallback
    """

    def __init__(
        self,
        base_url: str = LMSTUDIO_BASE_URL,
        default_model: str = "qwen2.5-14b_uncensored_instruct",
        timeout: float = 120.0,
        temperature: float = 0.7,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)

        # Statistics
        self.total_calls = 0
        self.total_tokens = 0

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            r = self._client.get(f"{self.base_url}/models")
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except Exception as e:
            return [f"Error: {e}"]

    def completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        """
        Execute chat completion.

        Args:
            prompt: User message
            model: Model name (uses default if None)
            system: System prompt
            max_tokens: Maximum tokens to generate

        Returns:
            CompletionResult with response and metadata
        """
        start = time.time()
        model = model or self.default_model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()

            response_text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            self.total_calls += 1
            self.total_tokens += usage.get("total_tokens", 0)

            # Simple Ihsan scoring
            ihsan = self._score_ihsan(response_text)

            return CompletionResult(
                response=response_text,
                model=model,
                execution_time=time.time() - start,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                ihsan_score=ihsan,
            )

        except httpx.TimeoutException:
            return CompletionResult(
                response="[TIMEOUT] Request exceeded time limit",
                model=model,
                execution_time=time.time() - start,
                ihsan_score=0.0,
            )
        except Exception as e:
            return CompletionResult(
                response=f"[ERROR] {str(e)}",
                model=model,
                execution_time=time.time() - start,
                ihsan_score=0.0,
            )

    def _score_ihsan(self, response: str) -> float:
        """Simple Ihsan scoring heuristic."""
        score = 0.95
        response_lower = response.lower()

        # Safety deductions
        unsafe = ["hack", "exploit", "malware", "bypass", "inject"]
        for term in unsafe:
            if term in response_lower:
                score -= 0.15

        # Quality checks
        if len(response) < 20:
            score -= 0.05
        if "error" in response_lower or "sorry" in response_lower:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def slot_completion(
        self,
        prompt: str,
        slot: str = "cold_core",
        system: Optional[str] = None,
    ) -> CompletionResult:
        """
        Completion using a capability slot.

        Slots:
        - cold_core: Deterministic reasoning
        - warm_surface: User-facing
        - primary_reasoning: Multi-agent planning
        - thinking: Extended CoT
        - fast: Quick responses
        """
        model = MODEL_SLOTS.get(slot, MODEL_SLOTS["cold_core"])
        return self.completion(prompt, model=model, system=system)

    def get_stats(self) -> Dict[str, Any]:
        """Return client statistics."""
        return {
            "base_url": self.base_url,
            "default_model": self.default_model,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_default_client: Optional[BizraDirectClient] = None


def get_client() -> BizraDirectClient:
    """Get or create default client."""
    global _default_client
    if _default_client is None:
        _default_client = BizraDirectClient()
    return _default_client


def quick_completion(
    prompt: str,
    slot: str = "cold_core",
    system: Optional[str] = None,
) -> str:
    """
    Quick completion with default settings.

    Args:
        prompt: User message
        slot: Capability slot
        system: Optional system prompt

    Returns:
        Response text
    """
    client = get_client()
    result = client.slot_completion(prompt, slot=slot, system=system)
    return result.response


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Direct LLM Client")
    parser.add_argument("prompt", nargs="?", default="What is 2+2?")
    parser.add_argument("--slot", default="fast", choices=list(MODEL_SLOTS.keys()))
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--list", action="store_true", help="List models")

    args = parser.parse_args()

    client = BizraDirectClient()

    if args.list:
        print("Available models:")
        for m in client.list_models():
            print(f"  - {m}")
    else:
        print("🧠 BIZRA Direct Client")
        print(f"   Slot: {args.slot}")
        print(f"   Model: {args.model or MODEL_SLOTS.get(args.slot)}")
        print(f"\n📝 Prompt: {args.prompt}\n")

        if args.model:
            result = client.completion(args.prompt, model=args.model)
        else:
            result = client.slot_completion(args.prompt, slot=args.slot)

        print("═" * 60)
        print(f"📤 Response:\n{result.response}")
        print("═" * 60)
        print("\n📊 Metrics:")
        print(f"   Time: {result.execution_time:.2f}s")
        print(f"   Tokens: {result.input_tokens} in / {result.output_tokens} out")
        print(f"   Ihsān: {result.ihsan_score:.2f}")
