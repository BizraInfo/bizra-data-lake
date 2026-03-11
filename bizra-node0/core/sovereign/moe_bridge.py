"""MOE Bridge — Expert-to-Model Dispatch for the Sovereign Nervous System.

Standing on Giants:
  Shazeer (2017)  — Sparsely-gated MOE top-K routing
  Kahneman (2011) — System-2 multi-expert deliberation
  Shannon (1948)  — Information-theoretic scoring (SNR)
  Boyd (1976)     — OODA loop (Observe→Orient→Decide→Act)
  Hewitt (1973)   — Actor model (each expert = independent actor)

Innovation:
  This bridge turns the theoretical 5-expert MOE panel into a LIVE
  multi-model reasoning system. Each expert routes to a different
  Ollama model, creating domain-specialized inference:

    pat_r → deepseek-r1:14b   (reasoning, planning, decomposition)
    pat_k → qwen2.5:3b        (knowledge retrieval, factual)
    pat_s → qwen2.5-coder:7b  (code generation, tool use)
    sat_g → phi3:mini          (governance, constitutional checks)
    sat_v → phi3:mini          (verification, proof validation)

  The bridge implements the InferenceProvider protocol, so it
  drops into SovereignNervousSystem as a plug-in replacement:

    ns = SovereignNervousSystem.create(
        inference=MOEBridge.create(),      # ← Drop-in replacement
        persistence_dir=Path("./state"),
    )

  After S2 deliberation, the observation is recorded with the
  ihsan_tensor carrying expert contributions — enabling the
  ReflexCompiler to learn WHICH expert combinations produce
  high-Ihsan results for future O(1) precipitation.

Usage:
    bridge = MOEBridge.create()
    response = await bridge.infer("How do I optimize database queries?")
    # Internally: MOE routes → pat_r + pat_s → two Ollama calls → synthesis
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger("bizra.sovereign.moe_bridge")

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS (from single source of truth)
# ═══════════════════════════════════════════════════════════════════

from core.integration.constants import (
    MOE_TOP_K,
    OLLAMA_URL,
)

# Expert → Ollama model mapping (overridable via env vars)
_EXPERT_MODEL_MAP: Dict[str, str] = {
    "pat_r": os.getenv("BIZRA_MODEL_PAT_R", "deepseek-r1:14b"),
    "pat_k": os.getenv("BIZRA_MODEL_PAT_K", "qwen2.5:3b"),
    "pat_s": os.getenv("BIZRA_MODEL_PAT_S", "qwen2.5-coder:7b"),
    "sat_g": os.getenv("BIZRA_MODEL_SAT_G", "phi3:mini"),
    "sat_v": os.getenv("BIZRA_MODEL_SAT_V", "phi3:mini"),
}

# Expert → system prompt specialization
_EXPERT_SYSTEM_PROMPTS: Dict[str, str] = {
    "pat_r": (
        "You are a reasoning expert. Analyze problems step-by-step. "
        "Focus on logic, planning, and decomposition. Be thorough and precise."
    ),
    "pat_k": (
        "You are a knowledge expert. Provide accurate, factual information. "
        "Focus on retrieval, definitions, and structured knowledge."
    ),
    "pat_s": (
        "You are a skills expert. Write clean, efficient code. "
        "Focus on implementation, tool use, and executable solutions."
    ),
    "sat_g": (
        "You are a governance expert. Evaluate constitutional compliance. "
        "Focus on policy alignment, ethical implications, and threshold checks."
    ),
    "sat_v": (
        "You are a verification expert. Validate claims with evidence. "
        "Focus on proof, testing, and correctness verification."
    ),
}


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ExpertCallResult:
    """Result from a single expert's model call."""

    expert_id: str
    model: str
    text: str
    latency_ms: float
    success: bool
    error: str = ""


@dataclass
class MOEBridgeStats:
    """Telemetry for the bridge."""

    total_inferences: int = 0
    expert_calls: int = 0
    expert_failures: int = 0
    avg_latency_ms: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# MOE BRIDGE — InferenceProvider Implementation
# ═══════════════════════════════════════════════════════════════════


class MOEBridge:
    """Expert-to-Model dispatch bridge implementing InferenceProvider protocol.

    This is the integration layer that makes the 12-agent organism LIVE.
    It connects MOE Engine routing decisions to actual Ollama model calls,
    synthesizes multi-expert outputs, and provides the ihsan_tensor for
    ReflexCompiler precipitation learning.

    Thread-safe for concurrent mission execution.
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        expert_models: Dict[str, str] | None = None,
        top_k: int = MOE_TOP_K,
        timeout_s: float = 30.0,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        self._ollama_url = ollama_url.rstrip("/")
        self._expert_models = expert_models or dict(_EXPERT_MODEL_MAP)
        self._top_k = top_k
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._stats = MOEBridgeStats()

        # Lazy-init MOE engine
        self._engine: Any = None

    def _get_engine(self) -> Any:
        """Lazy-init MOE engine to avoid circular imports."""
        if self._engine is None:
            from core.living_model.moe_engine import MOEEngine

            self._engine = MOEEngine(top_k=self._top_k)
        return self._engine

    @classmethod
    def create(
        cls,
        ollama_url: str | None = None,
        top_k: int = MOE_TOP_K,
        timeout_s: float = 30.0,
    ) -> MOEBridge:
        """Factory method for production use."""
        url = ollama_url or OLLAMA_URL
        return cls(ollama_url=url, top_k=top_k, timeout_s=timeout_s)

    @property
    def stats(self) -> MOEBridgeStats:
        return self._stats

    @property
    def last_ihsan_tensor(self) -> Dict[str, float]:
        """Expert contributions from the last inference call.

        This tensor feeds into ReflexCompiler.record_observation(ihsan_tensor=...)
        enabling the system to learn which expert combinations produce
        high-quality results for specific input patterns.
        """
        return (
            self._last_ihsan_tensor.copy()
            if hasattr(self, "_last_ihsan_tensor")
            else {}
        )

    # ─── InferenceProvider Protocol ──────────────────────────────

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        """Execute MOE-routed multi-expert inference.

        This is the main entry point, implementing the InferenceProvider
        protocol used by SovereignNervousSystem.

        Flow:
            1. MOE Engine routes input to top-K experts
            2. Each expert dispatches to its Ollama model
            3. Results are synthesized with weighted combination
            4. ihsan_tensor is recorded for ReflexCompiler learning

        Args:
            prompt: The mission text to process.
            **kwargs: Additional context (macro_state, expert_override, etc.)

        Returns:
            Synthesized response text from the activated experts.
        """
        engine = self._get_engine()
        context = kwargs.get("context", {})
        expert_override = kwargs.get("expert_override")
        t0 = time.monotonic()

        # Step 1: Route to top-K experts
        assignments = engine.route(
            prompt, context=context, expert_override=expert_override
        )
        logger.info(
            "MOE route: %s",
            " + ".join(f"{a.expert_id}({a.weight:.2f})" for a in assignments),
        )

        # Step 2: Execute each expert against its Ollama model
        results: list[ExpertCallResult] = []
        for assignment in assignments:
            result = await self._call_expert(assignment.expert_id, prompt)
            results.append(result)

        # Step 3: Synthesize with weighted combination
        successful = [r for r in results if r.success]
        if not successful:
            logger.warning("All experts failed — returning error")
            self._last_ihsan_tensor = {}
            self._stats.total_inferences += 1
            return "[All experts failed to produce a response]"

        # Build weight map from assignments
        weight_map = {a.expert_id: a.weight for a in assignments}

        # Weighted synthesis
        parts: list[str] = []
        ihsan_tensor: Dict[str, float] = {}

        for r in successful:
            w = weight_map.get(r.expert_id, 1.0 / len(successful))
            if len(successful) == 1:
                # Single expert: use its output directly (no prefixes)
                parts.append(r.text)
            else:
                parts.append(f"[{r.expert_id}] {r.text}")
            ihsan_tensor[r.expert_id] = w

        synthesized = "\n\n".join(parts)
        self._last_ihsan_tensor = ihsan_tensor

        # Stats
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._stats.total_inferences += 1
        n = self._stats.total_inferences
        self._stats.avg_latency_ms = (
            self._stats.avg_latency_ms * (n - 1) + elapsed_ms
        ) / n

        logger.info(
            "MOE inference: %d/%d experts succeeded, %.0fms",
            len(successful),
            len(results),
            elapsed_ms,
        )

        return synthesized

    # ─── Expert Model Dispatch ───────────────────────────────────

    async def _call_expert(self, expert_id: str, prompt: str) -> ExpertCallResult:
        """Dispatch a single expert to its Ollama model."""
        model = self._expert_models.get(expert_id, "phi3:mini")
        system_prompt = _EXPERT_SYSTEM_PROMPTS.get(expert_id, "")
        t0 = time.monotonic()

        self._stats.expert_calls += 1
        self._stats.model_usage[model] = self._stats.model_usage.get(model, 0) + 1

        try:
            text = await self._ollama_generate(model, prompt, system_prompt)
            elapsed = (time.monotonic() - t0) * 1000
            logger.debug(
                "Expert %s (%s): %.0fms, %d chars",
                expert_id,
                model,
                elapsed,
                len(text),
            )
            return ExpertCallResult(
                expert_id=expert_id,
                model=model,
                text=text,
                latency_ms=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            self._stats.expert_failures += 1
            logger.warning(
                "Expert %s (%s) failed after %.0fms: %s",
                expert_id,
                model,
                elapsed,
                e,
            )
            return ExpertCallResult(
                expert_id=expert_id,
                model=model,
                text="",
                latency_ms=elapsed,
                success=False,
                error=str(e),
            )

    async def _ollama_generate(self, model: str, prompt: str, system: str = "") -> str:
        """Call Ollama /api/generate endpoint."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for Ollama calls: pip install httpx")

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self._max_tokens,
                "temperature": self._temperature,
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(f"{self._ollama_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    # ─── Utility ─────────────────────────────────────────────────

    def get_expert_model(self, expert_id: str) -> str:
        """Return the Ollama model assigned to an expert."""
        return self._expert_models.get(expert_id, "phi3:mini")

    def set_expert_model(self, expert_id: str, model: str) -> None:
        """Override the model for an expert (runtime hot-swap)."""
        self._expert_models[expert_id] = model
        logger.info("Expert %s → model %s (hot-swapped)", expert_id, model)
