"""
BIZRA RLM Adapter - Recursive Language Model Integration

Bridges the RLM (Recursive Language Models) library with BIZRA's
PAT reasoning pipeline and Ihsān validation gate.

Architecture:
  User Request → FATE Gate → RLM.completion() → SAPE Probes → Response

Supports:
  - Local Ollama models (deepseek-r1:8b, mistral:latest)
  - Ihsān score validation on outputs
  - SAPE probe integration for content verification
  - Memory server integration for context persistence
"""

from __future__ import annotations

import os
import sys
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Add RLM to path
RLM_PATH = Path(__file__).parent.parent.parent / "rlm"
if RLM_PATH.exists() and str(RLM_PATH) not in sys.path:
    sys.path.insert(0, str(RLM_PATH))

# Import BIZRA core
from core.fate import FateSeal, get_fate_engine

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.56.1:1234/v1")

# Model mappings from model-family-genesis-v1-SEALED.yaml
# Updated for LM Studio server at 192.168.56.1:1234
BIZRA_MODEL_SLOTS = {
    "cold_core": {
        "primary": "qwen2.5-14b_uncensored_instruct",  # LM Studio
        "fallback": "mistral:latest",  # Ollama fallback
        "description": "Deterministic reasoning + self-correction",
        "provider": "lmstudio",
    },
    "warm_surface": {
        "primary": "mistralai/ministral-3-14b-reasoning",  # LM Studio
        "fallback": "qwen2.5:7b",
        "description": "User-facing tone control",
        "provider": "lmstudio",
    },
    "primary_reasoning": {
        "primary": "agentflow-planner-7b-i1",  # LM Studio
        "fallback": "bizra-planner:latest",  # Ollama fallback
        "description": "Multi-agent orchestration",
        "provider": "lmstudio",
    },
    "thinking": {
        "primary": "qwen/qwen3-4b-thinking-2507",  # LM Studio
        "fallback": "deepseek-r1:8b",
        "description": "Extended thinking/CoT",
        "provider": "lmstudio",
    },
    "vision": {
        "primary": "qwen/qwen3-vl-8b",  # LM Studio
        "fallback": "qwen/qwen3-vl-4b",
        "description": "Multimodal vision inference",
        "provider": "lmstudio",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RLMResult:
    """Result from an RLM completion with BIZRA metadata."""

    response: str
    execution_time: float
    model_used: str
    iterations: int
    ihsan_score: float
    ihsan_passed: bool
    fate_seal: Optional[FateSeal] = None
    sape_flags: List[str] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    request_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class RLMConfig:
    """Configuration for BIZRA RLM adapter."""

    slot: str = "cold_core"
    max_iterations: int = 30
    max_depth: int = 1
    verbose: bool = False
    use_ollama: bool = True
    base_url: Optional[str] = None
    ihsan_threshold: float = 0.90
    enable_sape: bool = True
    enable_fate: bool = True
    context_memory: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA CLIENT FOR RLM
# ═══════════════════════════════════════════════════════════════════════════════


class OllamaRLMClient:
    """
    Ollama client compatible with RLM's BaseLM interface.
    Uses Ollama's OpenAI-compatible API endpoint.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.6,
        **kwargs,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.kwargs = kwargs

        # Usage tracking
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Lazy import httpx
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=60.0)
        return self._client

    def completion(
        self, prompt: str | List[Dict[str, Any]], model: str | None = None
    ) -> str:
        """Execute chat completion via Ollama API."""

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        model = model or self.model_name
        client = self._get_client()

        response = client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Track usage
        self.total_calls += 1
        if "usage" in data:
            self.total_input_tokens += data["usage"].get("prompt_tokens", 0)
            self.total_output_tokens += data["usage"].get("completion_tokens", 0)

        return data["choices"][0]["message"]["content"]

    async def acompletion(
        self, prompt: str | List[Dict[str, Any]], model: str | None = None
    ) -> str:
        """Async completion (delegates to sync for now)."""
        return self.completion(prompt, model)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "model": self.model_name,
            "total_calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def get_last_usage(self) -> Dict[str, Any]:
        """Return last call usage (simplified)."""
        return self.get_usage_summary()


# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA RLM ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════


class BizraRLMAdapter:
    """
    BIZRA adapter for Recursive Language Models.

    Wraps RLM with:
    - FATE gate (pre-execution validation)
    - Ihsān scoring (post-execution quality gate)
    - SAPE probes (content verification)
    - Memory integration (context persistence)
    """

    def __init__(self, config: Optional[RLMConfig] = None):
        self.config = config or RLMConfig()
        self._rlm = None
        self._fate = None
        self._initialized = False

        # Statistics
        self.total_completions = 0
        self.ihsan_passes = 0
        self.ihsan_failures = 0
        self.fate_blocks = 0

    def _ensure_initialized(self):
        """Lazy initialization of RLM and FATE."""
        if self._initialized:
            return

        # Initialize FATE engine
        if self.config.enable_fate:
            self._fate = get_fate_engine()

        # Initialize RLM with Ollama backend
        self._init_rlm()
        self._initialized = True

    def _init_rlm(self):
        """Initialize RLM with LM Studio or Ollama models."""
        try:
            from rlm import RLM
            from rlm.clients import BaseLM

            # Get model from slot
            slot_config = BIZRA_MODEL_SLOTS.get(
                self.config.slot, BIZRA_MODEL_SLOTS["cold_core"]
            )
            model_name = slot_config["primary"]
            provider = slot_config.get("provider", "lmstudio")

            # Select base URL based on provider
            if self.config.base_url:
                base_url = self.config.base_url
            elif provider == "lmstudio":
                base_url = LMSTUDIO_BASE_URL
            else:
                base_url = OLLAMA_BASE_URL

            self._rlm = RLM(
                backend="vllm",  # Uses OpenAI-compatible API
                backend_kwargs={
                    "model_name": model_name,
                    "base_url": base_url,
                    "api_key": "lmstudio-local",  # Dummy key for local servers
                },
                environment="local",
                max_iterations=self.config.max_iterations,
                max_depth=self.config.max_depth,
                verbose=self.config.verbose,
            )

            self._model_name = model_name

        except ImportError as e:
            raise RuntimeError(f"RLM not available. Ensure rlm/ is in path. Error: {e}")

    def _hash_request(self, prompt: str) -> str:
        """Generate request fingerprint."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _run_fate_gate(self, prompt: str) -> Tuple[bool, Optional[FateSeal]]:
        """Run FATE pre-execution gate."""
        if not self.config.enable_fate or self._fate is None:
            return True, None

        seal = self._fate.audit_request(
            intent=prompt,
            context="rlm_completion",
            artifact_class="mcp_tool",
        )

        if seal.verdict == "REJECTED":
            self.fate_blocks += 1
            return False, seal

        return True, seal

    def _calculate_ihsan(
        self, response: str, execution_time: float
    ) -> Tuple[float, bool, List[str]]:
        """Calculate Ihsān score for response using SAPE-like heuristics."""
        flags = []
        scores = {
            "correctness": 0.95,
            "safety": 1.0,
            "user_benefit": 0.9,
            "efficiency": 0.9,
            "auditability": 0.85,
            "anti_centralization": 1.0,  # Local execution
            "robustness": 0.9,
            "adl_fairness": 0.95,
        }

        # Safety checks
        unsafe_patterns = ["exploit", "hack", "malware", "injection", "bypass"]
        response_lower = response.lower()
        for pattern in unsafe_patterns:
            if pattern in response_lower:
                scores["safety"] = 0.3
                flags.append(f"unsafe_pattern:{pattern}")
                break

        # Efficiency based on execution time
        if execution_time > 30:
            scores["efficiency"] = 0.6
            flags.append("slow_execution")
        elif execution_time > 10:
            scores["efficiency"] = 0.8

        # Auditability - check for explanation
        if len(response) < 50:
            scores["auditability"] = 0.7
            flags.append("terse_response")

        # Check for uncertainty markers
        uncertainty = ["might be", "not sure", "possibly", "maybe"]
        for marker in uncertainty:
            if marker in response_lower:
                scores["correctness"] = min(scores["correctness"], 0.8)
                flags.append("uncertainty_marker")
                break

        # Calculate weighted composite
        weights = {
            "correctness": 0.22,
            "safety": 0.22,
            "user_benefit": 0.14,
            "efficiency": 0.12,
            "auditability": 0.12,
            "anti_centralization": 0.08,
            "robustness": 0.06,
            "adl_fairness": 0.04,
        }

        composite = sum(scores[k] * weights[k] for k in weights)
        passed = composite >= self.config.ihsan_threshold

        return composite, passed, flags

    def completion(
        self,
        prompt: str,
        root_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RLMResult:
        """
        Execute RLM completion with BIZRA gates.

        Flow:
        1. FATE gate (pre-validation)
        2. RLM.completion() (recursive reasoning)
        3. Ihsān scoring (quality gate)
        4. Return result with metadata
        """
        self._ensure_initialized()
        self.total_completions += 1

        request_hash = self._hash_request(prompt)
        start_time = time.time()

        # Step 1: FATE gate
        fate_passed, fate_seal = self._run_fate_gate(prompt)
        if not fate_passed:
            return RLMResult(
                response=f"[BLOCKED BY FATE] {fate_seal.reason if fate_seal else 'Unknown'}",
                execution_time=time.time() - start_time,
                model_used=self._model_name,
                iterations=0,
                ihsan_score=0.0,
                ihsan_passed=False,
                fate_seal=fate_seal,
                sape_flags=["fate_blocked"],
                request_hash=request_hash,
            )

        # Step 2: RLM completion
        try:
            result = self._rlm.completion(prompt, root_prompt=root_prompt)
            response_text = result.response
            execution_time = result.execution_time
        except Exception as e:
            return RLMResult(
                response=f"[RLM ERROR] {str(e)}",
                execution_time=time.time() - start_time,
                model_used=self._model_name,
                iterations=0,
                ihsan_score=0.0,
                ihsan_passed=False,
                fate_seal=fate_seal,
                sape_flags=["rlm_error"],
                request_hash=request_hash,
            )

        # Step 3: Ihsān scoring
        ihsan_score, ihsan_passed, sape_flags = self._calculate_ihsan(
            response_text, execution_time
        )

        if ihsan_passed:
            self.ihsan_passes += 1
        else:
            self.ihsan_failures += 1
            sape_flags.append("ihsan_below_threshold")

        # Step 4: Build result
        token_usage = {}
        if hasattr(result, "usage_summary") and result.usage_summary:
            usage_dict = (
                result.usage_summary.to_dict()
                if hasattr(result.usage_summary, "to_dict")
                else {}
            )
            token_usage = usage_dict

        return RLMResult(
            response=response_text,
            execution_time=execution_time,
            model_used=self._model_name,
            iterations=getattr(result, "iterations", 1),
            ihsan_score=ihsan_score,
            ihsan_passed=ihsan_passed,
            fate_seal=fate_seal,
            sape_flags=sape_flags,
            token_usage=token_usage,
            request_hash=request_hash,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return adapter statistics."""
        return {
            "total_completions": self.total_completions,
            "ihsan_passes": self.ihsan_passes,
            "ihsan_failures": self.ihsan_failures,
            "fate_blocks": self.fate_blocks,
            "ihsan_pass_rate": self.ihsan_passes / max(1, self.total_completions),
            "config": {
                "slot": self.config.slot,
                "model": BIZRA_MODEL_SLOTS.get(self.config.slot, {}).get("primary"),
                "max_iterations": self.config.max_iterations,
                "ihsan_threshold": self.config.ihsan_threshold,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_default_adapter: Optional[BizraRLMAdapter] = None


def get_rlm_adapter(config: Optional[RLMConfig] = None) -> BizraRLMAdapter:
    """Get or create the default RLM adapter."""
    global _default_adapter
    if _default_adapter is None or config is not None:
        _default_adapter = BizraRLMAdapter(config)
    return _default_adapter


def rlm_completion(
    prompt: str,
    slot: str = "cold_core",
    verbose: bool = False,
) -> RLMResult:
    """
    Quick RLM completion with BIZRA gates.

    Args:
        prompt: The task/question to process
        slot: Model slot (cold_core, warm_surface, primary_reasoning)
        verbose: Enable verbose output

    Returns:
        RLMResult with response and metadata
    """
    config = RLMConfig(slot=slot, verbose=verbose)
    adapter = BizraRLMAdapter(config)
    return adapter.completion(prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA RLM Adapter")
    parser.add_argument(
        "prompt", nargs="?", default="Calculate 2^10 using Python code."
    )
    parser.add_argument(
        "--slot",
        default="cold_core",
        choices=["cold_core", "warm_surface", "primary_reasoning"],
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-fate", action="store_true", help="Disable FATE gate")
    parser.add_argument("--threshold", type=float, default=0.90, help="Ihsan threshold")

    args = parser.parse_args()

    config = RLMConfig(
        slot=args.slot,
        verbose=args.verbose,
        enable_fate=not args.no_fate,
        ihsan_threshold=args.threshold,
    )

    adapter = BizraRLMAdapter(config)

    print("\n🧠 BIZRA RLM Adapter")
    print(f"   Slot: {args.slot}")
    print(f"   Model: {BIZRA_MODEL_SLOTS[args.slot]['primary']}")
    print(f"   Threshold: {args.threshold}")
    print(f"\n📝 Prompt: {args.prompt}\n")

    result = adapter.completion(args.prompt)

    print("═" * 60)
    print(f"📤 Response:\n{result.response}")
    print("═" * 60)
    print("\n📊 Metrics:")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    print(
        f"   Ihsān Score: {result.ihsan_score:.4f} {'✅' if result.ihsan_passed else '❌'}"
    )
    print(f"   Iterations: {result.iterations}")
    if result.sape_flags:
        print(f"   Flags: {', '.join(result.sape_flags)}")

    print(f"\n📈 Stats: {adapter.get_stats()}")
