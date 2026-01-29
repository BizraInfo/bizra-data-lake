"""
BIZRA INFERENCE GATEWAY (PR1 IMPLEMENTATION)
═══════════════════════════════════════════════════════════════════════════════

Embedded LLM inference with fail-closed semantics.

Priority order:
1. llama.cpp (embedded, offline-capable) 
2. Ollama (if available and allowed)
3. LM Studio (if available and allowed)
4. DENY (fail-closed)

This is the core of thermodynamic entropy reduction.
Local inference = local world model = sovereignty.

Created: 2026-01-29 | BIZRA Sovereignty
Principle: لا نفترض — We do not assume.
"""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default model paths
DEFAULT_MODEL_DIR = Path("/var/lib/bizra/models")
CACHE_DIR = Path("/var/lib/bizra/cache")

# Tier definitions
TIER_CONFIGS = {
    "EDGE": {
        "max_params": "1.7B",
        "default_model": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "context_length": 4096,
        "n_gpu_layers": 0,  # CPU only for edge
        "target_speed": 12,  # tok/s
    },
    "LOCAL": {
        "max_params": "7B",
        "default_model": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "context_length": 8192,
        "n_gpu_layers": -1,  # All layers on GPU
        "target_speed": 35,  # tok/s
    },
    "POOL": {
        "max_params": "70B+",
        "default_model": None,  # Federated
        "context_length": 32768,
        "n_gpu_layers": -1,
        "target_speed": None,  # Varies
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ComputeTier(str, Enum):
    """Inference compute tiers."""
    EDGE = "edge"      # Always-on, low-power (TPU/CPU)
    LOCAL = "local"    # On-demand, high-power (GPU)
    POOL = "pool"      # URP federated compute


class InferenceBackend(str, Enum):
    """Available inference backends."""
    LLAMACPP = "llamacpp"      # Embedded (primary)
    OLLAMA = "ollama"          # External (fallback 1)
    LMSTUDIO = "lmstudio"      # External (fallback 2)
    POOL = "pool"              # URP federated
    OFFLINE = "offline"        # No inference available


class InferenceStatus(str, Enum):
    """Gateway status."""
    COLD = "cold"              # Not initialized
    WARMING = "warming"        # Loading models
    READY = "ready"            # Fully operational
    DEGRADED = "degraded"      # Fallback mode
    OFFLINE = "offline"        # No inference available


@dataclass
class InferenceConfig:
    """Configuration for the inference gateway."""
    # Model settings
    default_model: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    model_path: Optional[str] = None
    model_dir: Path = DEFAULT_MODEL_DIR
    
    # Context settings
    context_length: int = 8192
    max_tokens: int = 2048
    
    # Hardware settings
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    n_threads: int = 8
    n_batch: int = 512
    
    # Tier settings
    default_tier: ComputeTier = ComputeTier.LOCAL
    
    # Fallback chain
    fallbacks: List[str] = field(default_factory=lambda: ["ollama", "lmstudio"])
    
    # Fail-closed: deny if no local model available
    require_local: bool = True
    
    # External endpoints (for fallbacks)
    ollama_url: str = "http://localhost:11434"
    lmstudio_url: str = "http://192.168.56.1:1234"


@dataclass
class InferenceResult:
    """Result of an inference call."""
    content: str
    model: str
    backend: InferenceBackend
    tier: ComputeTier
    
    # Metrics
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    
    # Metadata
    timestamp: str = ""
    receipt_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass  
class TaskComplexity:
    """Estimated complexity of an inference task."""
    input_tokens: int
    estimated_output_tokens: int
    reasoning_depth: float  # 0.0 = simple, 1.0 = complex
    domain_specificity: float  # 0.0 = general, 1.0 = specialized
    
    @property
    def score(self) -> float:
        """Overall complexity score (0.0 - 1.0)."""
        token_factor = min(1.0, (self.input_tokens + self.estimated_output_tokens) / 4000)
        return (
            0.3 * token_factor +
            0.4 * self.reasoning_depth +
            0.3 * self.domain_specificity
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceBackendBase(ABC):
    """Abstract base class for inference backends."""
    
    @property
    @abstractmethod
    def backend_type(self) -> InferenceBackend:
        """Return the backend type."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend. Returns True if successful."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a completion."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a completion with streaming."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass
    
    @abstractmethod
    def get_loaded_model(self) -> Optional[str]:
        """Return the currently loaded model name."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# LLAMA.CPP BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class LlamaCppBackend(InferenceBackendBase):
    """
    Embedded inference via llama-cpp-python.
    
    This is the primary backend for sovereign inference.
    No external dependencies, works offline.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._model = None
        self._model_path: Optional[str] = None
        self._lock = threading.Lock()
    
    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.LLAMACPP
    
    async def initialize(self) -> bool:
        """Initialize llama.cpp with configured model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[LlamaCpp] llama-cpp-python not installed")
            return False
        
        model_path = self._resolve_model_path()
        if not model_path:
            print("[LlamaCpp] No model found")
            return False
        
        try:
            print(f"[LlamaCpp] Loading model: {model_path}")
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                n_batch=self.config.n_batch,
                verbose=False,
            )
            self._model_path = str(model_path)
            print(f"[LlamaCpp] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[LlamaCpp] Failed to load model: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion."""
        if not self._model:
            raise RuntimeError("Model not initialized")
        
        with self._lock:
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                **kwargs
            )
        
        return result["choices"][0]["text"]
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate completion with streaming."""
        if not self._model:
            raise RuntimeError("Model not initialized")
        
        with self._lock:
            for chunk in self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stream=True,
                **kwargs
            ):
                if "choices" in chunk and chunk["choices"]:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text
    
    async def health_check(self) -> bool:
        """Check if model is loaded and responsive."""
        if not self._model:
            return False
        try:
            # Quick inference test
            with self._lock:
                self._model("test", max_tokens=1)
            return True
        except Exception:
            return False
    
    def get_loaded_model(self) -> Optional[str]:
        """Return loaded model path."""
        return self._model_path
    
    def _resolve_model_path(self) -> Optional[Path]:
        """Resolve the model path."""
        # 1. Explicit path
        if self.config.model_path:
            path = Path(self.config.model_path)
            if path.exists():
                return path
        
        # 2. Look in model directory
        if self.config.model_dir.exists():
            # Find any .gguf file
            gguf_files = list(self.config.model_dir.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]
        
        # 3. Try to download from HuggingFace
        # This would be implemented in a separate model manager
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA BACKEND (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaBackend(InferenceBackendBase):
    """
    Ollama backend for fallback inference.
    
    Requires Ollama server running externally.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._available_models: List[str] = []
        self._current_model: Optional[str] = None
    
    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.OLLAMA
    
    async def initialize(self) -> bool:
        """Check Ollama availability and list models."""
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/tags",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
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
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate via Ollama API."""
        import urllib.request
        
        payload = json.dumps({
            "model": self._current_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }).encode()
        
        req = urllib.request.Request(
            f"{self.config.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate with streaming via Ollama API."""
        # For simplicity, just return full response
        # Full streaming implementation would use httpx or aiohttp
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield response
    
    async def health_check(self) -> bool:
        """Check Ollama health."""
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(f"{self.config.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    def get_loaded_model(self) -> Optional[str]:
        return self._current_model


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE GATEWAY
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceGateway:
    """
    Tiered inference gateway with fail-closed semantics.
    
    Routes requests to appropriate compute tier based on complexity.
    Provides fallback chain when primary backend unavailable.
    
    Fail-closed: If no backend available and require_local=True, deny request.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.status = InferenceStatus.COLD
        
        # Backends by tier
        self._backends: Dict[ComputeTier, InferenceBackendBase] = {}
        self._active_backend: Optional[InferenceBackendBase] = None
        
        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
    
    async def initialize(self) -> bool:
        """
        Initialize the gateway and backends.
        
        Returns True if at least one backend is available.
        """
        self.status = InferenceStatus.WARMING
        
        # Try llama.cpp first (embedded, sovereign)
        llamacpp = LlamaCppBackend(self.config)
        if await llamacpp.initialize():
            self._backends[ComputeTier.LOCAL] = llamacpp
            self._backends[ComputeTier.EDGE] = llamacpp  # Same for now
            self._active_backend = llamacpp
            self.status = InferenceStatus.READY
            print("[Gateway] llama.cpp backend ready (SOVEREIGN MODE)")
            return True
        
        # Try fallbacks
        if not self.config.require_local:
            for fallback in self.config.fallbacks:
                if fallback == "ollama":
                    ollama = OllamaBackend(self.config)
                    if await ollama.initialize():
                        self._backends[ComputeTier.LOCAL] = ollama
                        self._active_backend = ollama
                        self.status = InferenceStatus.DEGRADED
                        print("[Gateway] Ollama fallback ready (DEGRADED MODE)")
                        return True
        
        # Fail-closed
        self.status = InferenceStatus.OFFLINE
        print("[Gateway] No backend available (OFFLINE MODE)")
        return False
    
    def estimate_complexity(self, prompt: str) -> TaskComplexity:
        """
        Estimate task complexity for routing decisions.
        
        Simple heuristics for now. Could be replaced with classifier.
        """
        words = prompt.split()
        input_tokens = len(words) * 1.3  # Rough estimate
        
        # Heuristics for reasoning depth
        reasoning_keywords = ["why", "how", "explain", "analyze", "compare", "prove"]
        reasoning_depth = sum(1 for w in words if w.lower() in reasoning_keywords) / max(len(words), 1)
        
        # Heuristics for domain specificity  
        technical_keywords = ["algorithm", "equation", "theorem", "protocol", "architecture"]
        domain_specificity = sum(1 for w in words if w.lower() in technical_keywords) / max(len(words), 1)
        
        return TaskComplexity(
            input_tokens=int(input_tokens),
            estimated_output_tokens=min(int(input_tokens * 2), 2048),
            reasoning_depth=min(reasoning_depth * 5, 1.0),
            domain_specificity=min(domain_specificity * 5, 1.0),
        )
    
    def route(self, complexity: TaskComplexity) -> ComputeTier:
        """
        Route task to appropriate compute tier.
        
        EDGE: complexity < 0.3
        LOCAL: 0.3 <= complexity < 0.8
        POOL: complexity >= 0.8
        """
        score = complexity.score
        
        if score < 0.3:
            return ComputeTier.EDGE
        elif score < 0.8:
            return ComputeTier.LOCAL
        else:
            return ComputeTier.POOL
    
    async def infer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        tier: Optional[ComputeTier] = None,
        stream: bool = False,
    ) -> Union[InferenceResult, AsyncIterator[str]]:
        """
        Run inference on prompt.
        
        Fail-closed: Raises RuntimeError if no backend available.
        """
        # Check availability
        if self.status == InferenceStatus.OFFLINE:
            raise RuntimeError("Inference denied: no backend available (fail-closed)")
        
        if not self._active_backend:
            raise RuntimeError("Inference denied: no active backend")
        
        # Estimate complexity and route
        complexity = self.estimate_complexity(prompt)
        target_tier = tier or self.route(complexity)
        
        # Get backend for tier (fallback to active)
        backend = self._backends.get(target_tier, self._active_backend)
        
        # Run inference
        start_time = time.time()
        max_tokens = max_tokens or self.config.max_tokens
        
        if stream:
            return backend.generate_stream(prompt, max_tokens, temperature)
        
        response = await backend.generate(prompt, max_tokens, temperature)
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens_generated = len(response.split())  # Rough estimate
        tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        
        # Update stats
        self._total_requests += 1
        self._total_tokens += tokens_generated
        self._total_latency_ms += latency_ms
        
        return InferenceResult(
            content=response,
            model=backend.get_loaded_model() or "unknown",
            backend=backend.backend_type,
            tier=target_tier,
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
            latency_ms=round(latency_ms, 2),
        )
    
    async def health(self) -> Dict[str, Any]:
        """Get gateway health status."""
        backends_health = {}
        for tier, backend in self._backends.items():
            backends_health[tier.value] = await backend.health_check()
        
        return {
            "status": self.status.value,
            "active_backend": self._active_backend.backend_type.value if self._active_backend else None,
            "active_model": self._active_backend.get_loaded_model() if self._active_backend else None,
            "backends": backends_health,
            "stats": {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "avg_latency_ms": (
                    self._total_latency_ms / self._total_requests
                    if self._total_requests > 0 else 0
                ),
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_gateway_instance: Optional[InferenceGateway] = None

def get_inference_gateway() -> InferenceGateway:
    """Get the singleton inference gateway."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = InferenceGateway()
    return _gateway_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Inference Gateway")
    parser.add_argument("command", choices=["init", "infer", "health"])
    parser.add_argument("--prompt", help="Prompt for inference")
    parser.add_argument("--model", help="Model path")
    parser.add_argument("--tier", choices=["edge", "local", "pool"])
    args = parser.parse_args()
    
    gateway = get_inference_gateway()
    
    if args.model:
        gateway.config.model_path = args.model
    
    if args.command == "init":
        success = await gateway.initialize()
        print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")
        print(f"Status: {gateway.status.value}")
    
    elif args.command == "infer":
        if not args.prompt:
            print("Error: --prompt required")
            return
        
        await gateway.initialize()
        tier = ComputeTier(args.tier) if args.tier else None
        
        result = await gateway.infer(args.prompt, tier=tier)
        print(f"\n{'='*60}")
        print(f"Model: {result.model}")
        print(f"Backend: {result.backend.value}")
        print(f"Tier: {result.tier.value}")
        print(f"Tokens: {result.tokens_generated} @ {result.tokens_per_second} tok/s")
        print(f"Latency: {result.latency_ms}ms")
        print(f"{'='*60}")
        print(result.content)
    
    elif args.command == "health":
        await gateway.initialize()
        health = await gateway.health()
        print(json.dumps(health, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
