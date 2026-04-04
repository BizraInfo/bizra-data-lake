"""
BIZRA Cognitive Hooks - T1.2 Hook Integration

Provides the hook interface between BIZRA's kernel and LLM backends.
Implements FATE gating with Ihsān threshold enforcement (0.85 per Blueprint).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pathlib import Path

# Handle import path for both module and script execution
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.cognitive.direct_client import (
    BizraDirectClient,
    MODEL_SLOTS,
    LMSTUDIO_BASE_URL,
)
from core.fate import FateEngine

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Blueprint specifies 0.85 threshold
IHSAN_THRESHOLD = float(os.getenv("BIZRA_IHSAN_THRESHOLD", "0.85"))

# Maximum retries on FATE rejection
MAX_FATE_RETRIES = int(os.getenv("BIZRA_MAX_FATE_RETRIES", "2"))


class HookType(Enum):
    """Types of cognitive hooks."""

    COMPLETION = "completion"
    REASONING = "reasoning"
    PLANNING = "planning"
    VISION = "vision"
    EMBEDDING = "embedding"


@dataclass
class HookResult:
    """Result from a cognitive hook call."""

    success: bool
    response: str
    slot_used: str
    model_used: str
    ihsan_score: float
    ihsan_passed: bool
    fate_seal_id: Optional[str]
    execution_time: float
    tokens_used: int
    hook_type: HookType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class HookConfig:
    """Configuration for a cognitive hook."""

    slot: str = "cold_core"
    temperature: float = 0.7
    max_tokens: int = 2048
    require_fate_gate: bool = True
    ihsan_threshold: float = IHSAN_THRESHOLD
    system_prompt: Optional[str] = None


class CognitiveHook:
    """
    T1.2 Cognitive Hook - Bridge between BIZRA kernel and LLM backends.

    Features:
    - FATE gate validation on input prompts
    - Ihsān scoring on outputs
    - Slot-based model routing
    - Automatic fallback on failures

    Usage:
        hook = CognitiveHook()
        result = hook.invoke("What is the capital of France?")
        if result.success and result.ihsan_passed:
            print(result.response)
    """

    def __init__(
        self,
        config: Optional[HookConfig] = None,
        client: Optional[BizraDirectClient] = None,
    ):
        self.config = config or HookConfig()
        self.client = client or BizraDirectClient(
            base_url=LMSTUDIO_BASE_URL,
            default_model=MODEL_SLOTS.get(self.config.slot, MODEL_SLOTS["cold_core"]),
            timeout=120.0,
            temperature=self.config.temperature,
        )
        self.fate_engine = FateEngine()

        # Statistics
        self.total_invocations = 0
        self.fate_rejections = 0
        self.ihsan_failures = 0
        self.successful_completions = 0

    def invoke(
        self,
        prompt: str,
        slot: Optional[str] = None,
        hook_type: HookType = HookType.COMPLETION,
        context: str = "",
        **kwargs,
    ) -> HookResult:
        """
        Invoke the cognitive hook with FATE gating.

        Args:
            prompt: The user prompt
            slot: Override slot (uses config.slot if None)
            hook_type: Type of cognitive operation
            context: Additional context for FATE evaluation
            **kwargs: Additional parameters

        Returns:
            HookResult with response and metadata
        """
        start = time.time()
        self.total_invocations += 1
        slot = slot or self.config.slot

        # ─────────────────────────────────────────────────────────────────────
        # FATE Gate (Input Validation)
        # ─────────────────────────────────────────────────────────────────────

        fate_seal = None
        if self.config.require_fate_gate:
            fate_seal = self.fate_engine.audit_request(
                intent=prompt,
                context=context,
                artifact_class="mcp_tool",
            )

            if fate_seal.verdict == "REJECTED":
                self.fate_rejections += 1
                return HookResult(
                    success=False,
                    response=f"[FATE REJECTED] {fate_seal.reason}",
                    slot_used=slot,
                    model_used="N/A",
                    ihsan_score=fate_seal.composite_score,
                    ihsan_passed=False,
                    fate_seal_id=fate_seal.id,
                    execution_time=time.time() - start,
                    tokens_used=0,
                    hook_type=hook_type,
                    metadata={
                        "rejection_reason": fate_seal.reason,
                        "fate_threshold": fate_seal.threshold,
                    },
                )

        # ─────────────────────────────────────────────────────────────────────
        # LLM Completion
        # ─────────────────────────────────────────────────────────────────────

        try:
            result = self.client.slot_completion(
                prompt=prompt,
                slot=slot,
                system=self.config.system_prompt,
            )
        except Exception as e:
            return HookResult(
                success=False,
                response=f"[ERROR] {str(e)}",
                slot_used=slot,
                model_used="N/A",
                ihsan_score=0.0,
                ihsan_passed=False,
                fate_seal_id=fate_seal.id if fate_seal else None,
                execution_time=time.time() - start,
                tokens_used=0,
                hook_type=hook_type,
            )

        # ─────────────────────────────────────────────────────────────────────
        # Ihsān Threshold Check (Output Validation)
        # ─────────────────────────────────────────────────────────────────────

        ihsan_passed = result.ihsan_score >= self.config.ihsan_threshold

        if not ihsan_passed:
            self.ihsan_failures += 1
        else:
            self.successful_completions += 1

        return HookResult(
            success=True,
            response=result.response,
            slot_used=slot,
            model_used=result.model,
            ihsan_score=result.ihsan_score,
            ihsan_passed=ihsan_passed,
            fate_seal_id=fate_seal.id if fate_seal else None,
            execution_time=time.time() - start,
            tokens_used=result.input_tokens + result.output_tokens,
            hook_type=hook_type,
            metadata={
                "model_execution_time": result.execution_time,
                "ihsan_threshold": self.config.ihsan_threshold,
            },
        )

    def reason(self, query: str, **kwargs) -> HookResult:
        """Invoke reasoning hook using cold_core slot."""
        return self.invoke(
            query, slot="cold_core", hook_type=HookType.REASONING, **kwargs
        )

    def plan(self, objective: str, **kwargs) -> HookResult:
        """Invoke planning hook using primary_reasoning slot."""
        return self.invoke(
            objective, slot="primary_reasoning", hook_type=HookType.PLANNING, **kwargs
        )

    def quick(self, query: str, **kwargs) -> HookResult:
        """Invoke fast completion using nano slot."""
        return self.invoke(query, slot="fast", hook_type=HookType.COMPLETION, **kwargs)

    def think(self, problem: str, **kwargs) -> HookResult:
        """Invoke extended thinking using thinking slot (CoT)."""
        return self.invoke(
            problem, slot="thinking", hook_type=HookType.REASONING, **kwargs
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return hook statistics."""
        return {
            "total_invocations": self.total_invocations,
            "fate_rejections": self.fate_rejections,
            "ihsan_failures": self.ihsan_failures,
            "successful_completions": self.successful_completions,
            "success_rate": (
                self.successful_completions / max(1, self.total_invocations)
            ),
            "ihsan_threshold": self.config.ihsan_threshold,
            "default_slot": self.config.slot,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL HOOK REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_hook_registry: Dict[str, CognitiveHook] = {}


def register_hook(name: str, hook: CognitiveHook):
    """Register a named cognitive hook."""
    _hook_registry[name] = hook


def get_hook(name: str = "default") -> CognitiveHook:
    """Get a registered hook by name."""
    if name not in _hook_registry:
        _hook_registry[name] = CognitiveHook()
    return _hook_registry[name]


def invoke(prompt: str, hook: str = "default", **kwargs) -> HookResult:
    """Convenience function to invoke a hook by name."""
    return get_hook(hook).invoke(prompt, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-REGISTERED HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

# Default general-purpose hook
register_hook("default", CognitiveHook(HookConfig(slot="cold_core")))

# Fast hook for simple queries
register_hook("fast", CognitiveHook(HookConfig(slot="fast", require_fate_gate=False)))

# Planning hook for multi-step tasks
register_hook(
    "planner",
    CognitiveHook(
        HookConfig(
            slot="primary_reasoning",
            system_prompt="You are a strategic planner. Break down objectives into actionable steps.",
        )
    ),
)

# Thinking hook for extended reasoning
register_hook(
    "thinker",
    CognitiveHook(
        HookConfig(
            slot="thinking",
            system_prompt="Think step by step. Use /think tokens for extended reasoning.",
        )
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Cognitive Hooks T1.2")
    parser.add_argument("prompt", nargs="?", default="What is 2+2?")
    parser.add_argument(
        "--hook", default="default", choices=["default", "fast", "planner", "thinker"]
    )
    parser.add_argument("--slot", help="Override slot")
    parser.add_argument("--no-fate", action="store_true", help="Disable FATE gate")
    parser.add_argument("--stats", action="store_true", help="Show hook stats")

    args = parser.parse_args()

    hook = get_hook(args.hook)

    if args.no_fate:
        hook.config.require_fate_gate = False

    print(f"🪝 BIZRA Cognitive Hook: {args.hook}")
    print(f"   Slot: {hook.config.slot}")
    print(f"   FATE Gate: {'ON' if hook.config.require_fate_gate else 'OFF'}")
    print(f"   Ihsān Threshold: {hook.config.ihsan_threshold}")
    print(f"\n📝 Prompt: {args.prompt}\n")

    result = hook.invoke(args.prompt, slot=args.slot)

    print("═" * 60)

    if result.success:
        print("✅ SUCCESS")
        print(f"\n📤 Response:\n{result.response}")
    else:
        print(f"❌ FAILED: {result.response}")

    print("\n═" * 60)
    print("📊 Metrics:")
    print(f"   Slot: {result.slot_used} → {result.model_used}")
    print(f"   Time: {result.execution_time:.2f}s")
    print(f"   Tokens: {result.tokens_used}")
    print(
        f"   Ihsān: {result.ihsan_score:.2f} ({'✓ PASS' if result.ihsan_passed else '✗ FAIL'})"
    )
    if result.fate_seal_id:
        print(f"   FATE Seal: {result.fate_seal_id[:16]}...")

    if args.stats:
        print("\n📈 Hook Stats:")
        for k, v in hook.get_stats().items():
            print(f"   {k}: {v}")
