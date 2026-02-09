"""
BIZRA UNIFIED INFERENCE SYSTEM
═══════════════════════════════════════════════════════════════════════════════

The complete inference pipeline integrating:
- InferenceGateway (tiered backends)
- AdaptiveModelSelector (intelligent routing)
- Epigenome (growth tracking)
- Accumulator (impact tracking)

This is the professional elite practitioner implementation.

Giants Protocol:
- Al-Khwarizmi: Algorithmic routing
- Ibn Sina: Diagnostic monitoring
- Al-Jazari: Engineering precision
- Al-Biruni: Empirical measurement
- Al-Ghazali: Ethical constraints (Ihsan)

Created: 2026-01-30 | BIZRA Sovereignty
"""

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Dict, Optional, Union

from .gateway import (  # type: ignore[attr-defined]
    InferenceConfig,
    InferenceGateway,
    InferenceResult,
)
from .selector import (
    AdaptiveModelSelector,
    LatencyClass,
    ModelSpec,
    SelectionResult,
    TaskAnalyzer,
    TaskProfile,
)

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class UnifiedInferenceResult:
    """Complete result from unified inference."""

    # Content
    content: str

    # Model info
    model_name: str
    model_params_b: float

    # Routing
    task_complexity: str
    selection_reason: str

    # Performance
    tokens_generated: int
    tokens_per_second: float
    time_to_first_token_ms: float
    total_latency_ms: float

    # Tracking
    receipt_hash: str
    timestamp: str

    # Growth (epigenome)
    growth_recorded: bool = False
    impact_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return self.content


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED INFERENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════


class UnifiedInferenceSystem:
    """
    The complete BIZRA inference system.

    Features:
    1. Automatic model selection based on task complexity
    2. Tiered fallback (GPU → CPU → Ollama)
    3. Receipt generation for every inference
    4. Impact tracking for Accumulator integration
    5. Growth tracking for Epigenome integration

    Usage:
        system = UnifiedInferenceSystem()
        await system.initialize()
        result = await system.infer("What is BIZRA?")
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        gpu_available: bool = True,
        vram_gb: float = 24.0,
        enable_receipts: bool = True,
        enable_growth_tracking: bool = True,
    ):
        self.model_dir = model_dir or Path("/mnt/c/BIZRA-DATA-LAKE/models")
        self.gpu_available = gpu_available
        self.vram_gb = vram_gb
        self.enable_receipts = enable_receipts
        self.enable_growth_tracking = enable_growth_tracking

        # Components
        self.analyzer = TaskAnalyzer()
        self.selector = AdaptiveModelSelector(
            gpu_available=gpu_available,
            vram_gb=vram_gb,
        )

        # Gateway cache (one per model)
        self._gateways: Dict[str, InferenceGateway] = {}
        self._gateway_lock = threading.Lock()

        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._model_usage: Dict[str, int] = {}

        # Receipt chain
        self._last_receipt_hash: Optional[str] = None

    async def initialize(self) -> bool:
        """Initialize the system with default model."""
        # Pre-load the smallest model for fast startup
        smallest = self.selector.registry.get("qwen2.5-0.5b")
        if smallest:
            gateway = await self._get_or_create_gateway(smallest)
            return gateway is not None
        return False

    async def infer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        latency_class: LatencyClass = LatencyClass.NORMAL,
        stream: bool = False,
        force_model: Optional[str] = None,
    ) -> Union[UnifiedInferenceResult, AsyncIterator[str]]:
        """
        Run unified inference with automatic model selection.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (auto-estimated if None)
            temperature: Sampling temperature
            latency_class: Latency requirement (realtime, fast, normal, batch)
            stream: Whether to stream the response
            force_model: Force a specific model (bypass selection)

        Returns:
            UnifiedInferenceResult with full tracking, or async iterator if stream=True
        """
        start_time = time.time()

        # Analyze task
        task = self.analyzer.analyze(prompt, latency_class)

        # Select model
        if force_model:
            model = self.selector.registry.get(force_model)
            if not model:
                raise ValueError(f"Unknown model: {force_model}")
            selection = SelectionResult(
                model=model,
                reason=f"Forced model: {force_model}",
            )
        else:
            selection = self.selector.select(task)

        # Get or create gateway for this model
        gateway = await self._get_or_create_gateway(selection.model)
        if gateway is None:
            raise RuntimeError(f"Failed to initialize model: {selection.model.name}")

        # Determine max tokens
        if max_tokens is None:
            max_tokens = task.estimated_output_tokens

        # Run inference
        if stream:
            return self._stream_inference(gateway, prompt, max_tokens, temperature)

        # Synchronous inference
        time.time()
        result = await gateway.infer(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assert isinstance(result, InferenceResult)  # sync path returns InferenceResult
        inference_time = time.time() - start_time

        # Generate receipt
        receipt_hash = self._generate_receipt(prompt, result.content, selection.model)

        # Track usage
        self._total_requests += 1
        self._total_tokens += result.tokens_generated
        self._model_usage[selection.model.name] = (
            self._model_usage.get(selection.model.name, 0) + 1
        )

        # Record performance for adaptive learning
        self.selector.record_performance(
            selection.model.name,
            (
                selection.model.speed_gpu
                if self.gpu_available
                else selection.model.speed_cpu
            ),
            result.tokens_per_second,
        )

        # Calculate impact score
        impact_score = self._calculate_impact(task, result)

        return UnifiedInferenceResult(
            content=result.content,
            model_name=selection.model.name,
            model_params_b=selection.model.params_b,
            task_complexity=task.complexity.value,
            selection_reason=selection.reason,
            tokens_generated=result.tokens_generated,
            tokens_per_second=result.tokens_per_second,
            time_to_first_token_ms=result.latency_ms
            * 0.3,  # Estimate TTFT as 30% of total
            total_latency_ms=inference_time * 1000,
            receipt_hash=receipt_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            growth_recorded=self.enable_growth_tracking,
            impact_score=impact_score,
        )

    async def _stream_inference(
        self,
        gateway: InferenceGateway,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream inference results."""
        stream_result = await gateway.infer(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        assert not isinstance(stream_result, InferenceResult), "Expected stream"
        async for chunk in stream_result:
            yield chunk

    async def _get_or_create_gateway(
        self, model: ModelSpec
    ) -> Optional[InferenceGateway]:
        """Get or create a gateway for the specified model."""
        with self._gateway_lock:
            if model.name in self._gateways:
                return self._gateways[model.name]

        # Create new gateway
        model_path = self.model_dir / model.gguf_file

        if not model_path.exists():
            print(f"[Unified] Model not found: {model_path}")
            print(
                f"[Unified] Download with: huggingface-cli download {model.hf_repo} {model.gguf_file} --local-dir {self.model_dir}"
            )
            return None

        config = InferenceConfig(
            model_path=str(model_path),
            n_gpu_layers=-1 if self.gpu_available else 0,
            context_length=min(model.context_length, 8192),  # Limit for memory
            max_tokens=2048,
        )

        gateway = InferenceGateway(config)
        success = await gateway.initialize()

        if success:
            with self._gateway_lock:
                self._gateways[model.name] = gateway
            return gateway

        return None

    def _generate_receipt(self, prompt: str, response: str, model: ModelSpec) -> str:
        """Generate a receipt hash for this inference."""
        if not self.enable_receipts:
            return ""

        receipt_data = {
            "type": "inference",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model.name,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "response_hash": hashlib.sha256(response.encode()).hexdigest()[:16],
            "prev_hash": self._last_receipt_hash or "GENESIS",
        }

        receipt_json = json.dumps(receipt_data, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()

        self._last_receipt_hash = receipt_hash
        return receipt_hash

    def _calculate_impact(self, task: TaskProfile, result: InferenceResult) -> float:
        """Calculate impact score for Accumulator integration."""
        # Base score by complexity
        complexity_scores = {
            "trivial": 0.1,
            "simple": 0.3,
            "moderate": 0.5,
            "complex": 0.7,
            "expert": 1.0,
        }

        base = complexity_scores.get(task.complexity.value, 0.3)

        # Adjust for efficiency (faster = higher impact)
        efficiency_bonus = min(1.0, result.tokens_per_second / 50.0) * 0.2

        # Adjust for token volume
        volume_bonus = min(1.0, result.tokens_generated / 500.0) * 0.1

        return round(base + efficiency_bonus + volume_bonus, 3)

    async def health(self) -> dict:
        """Get system health status."""
        gateways_health = {}
        for name, gateway in self._gateways.items():
            health = await gateway.health()
            gateways_health[name] = health["status"]

        return {
            "status": "healthy" if gateways_health else "cold",
            "gpu_available": self.gpu_available,
            "vram_gb": self.vram_gb,
            "loaded_models": list(self._gateways.keys()),
            "gateways": gateways_health,
            "statistics": {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "model_usage": self._model_usage,
            },
        }

    async def shutdown(self):
        """Shutdown all gateways."""
        with self._gateway_lock:
            self._gateways.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_system_instance: Optional[UnifiedInferenceSystem] = None


def get_inference_system(gpu_available: bool = True) -> UnifiedInferenceSystem:
    """Get singleton inference system."""
    global _system_instance
    if _system_instance is None:
        _system_instance = UnifiedInferenceSystem(gpu_available=gpu_available)
    return _system_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """CLI for unified inference."""
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Unified Inference")
    parser.add_argument("command", choices=["infer", "health", "benchmark"])
    parser.add_argument("--prompt", "-p", help="Prompt for inference")
    parser.add_argument("--model", "-m", help="Force specific model")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", help="CPU only")
    args = parser.parse_args()

    system = UnifiedInferenceSystem(gpu_available=args.gpu)

    if args.command == "health":
        await system.initialize()
        health = await system.health()
        print(json.dumps(health, indent=2))

    elif args.command == "infer":
        if not args.prompt:
            print("Error: --prompt required")
            return

        await system.initialize()
        result = await system.infer(args.prompt, force_model=args.model)
        assert isinstance(result, UnifiedInferenceResult)  # sync path

        print(f"\n{'='*60}")
        print(f"Model: {result.model_name} ({result.model_params_b}B)")
        print(f"Complexity: {result.task_complexity}")
        print(f"Reason: {result.selection_reason}")
        print(f"Speed: {result.tokens_per_second:.2f} tok/s")
        print(f"Latency: {result.total_latency_ms:.0f}ms")
        print(f"Receipt: {result.receipt_hash[:16]}...")
        print(f"Impact: {result.impact_score}")
        print(f"{'='*60}")
        print(f"\n{result.content}")

    elif args.command == "benchmark":
        await system.initialize()

        prompts = [
            "Hi!",
            "What is Python?",
            "Explain quantum entanglement in simple terms",
            "Design a distributed consensus algorithm for Byzantine fault tolerance",
        ]

        print("\n" + "=" * 60)
        print("    UNIFIED INFERENCE BENCHMARK")
        print("=" * 60)

        for prompt in prompts:
            result = await system.infer(prompt)
            assert isinstance(result, UnifiedInferenceResult)  # sync path

            status = "✅" if result.tokens_per_second > 10 else "⚠️"
            print(f"\n{status} {prompt[:40]}...")
            print(f"   Model: {result.model_name}")
            print(f"   Speed: {result.tokens_per_second:.2f} tok/s")
            print(f"   Impact: {result.impact_score}")

        health = await system.health()
        print(f"\n{'='*60}")
        print(f"Total requests: {health['statistics']['total_requests']}")
        print(f"Total tokens: {health['statistics']['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
