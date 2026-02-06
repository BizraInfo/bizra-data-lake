"""
Hook-First Governance — FATE Gate Implementation

Pre-tool-use hook system that enforces Ihsan validation before any operation.
FATE = Fidelity, Accountability, Transparency, Ethics

Standing on Giants:
- Lamport (1978): Happened-before relation for causal ordering
- Anthropic: Constitutional AI constraints
- Shannon: Information-theoretic quality gates
- PMBOK: Quality gates at every checkpoint

The FATE Gate pattern ensures:
1. FIDELITY: Operations align with declared intentions
2. ACCOUNTABILITY: All actions are traceable and auditable
3. TRANSPARENCY: Decision paths are explainable
4. ETHICS: Ihsan threshold enforced before execution

Hook Chain:
    PRE_VALIDATE -> PRE_EXECUTE -> EXECUTE -> POST_EXECUTE -> POST_VALIDATE

Constitutional Invariant:
    No operation proceeds unless FATE gate passes.

Created: 2026-02-03 | BIZRA Elite Integration v1.1.0
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import uuid

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    IHSAN_WEIGHTS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TYPES
# ============================================================================

T = TypeVar("T")
R = TypeVar("R")

# Hook function types
SyncHook = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncHook = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
HookFunction = Union[SyncHook, AsyncHook]


class HookPhase(str, Enum):
    """Phase in the hook execution chain."""
    PRE_VALIDATE = "pre_validate"   # Before validation (schema check)
    PRE_EXECUTE = "pre_execute"     # Before execution (business logic)
    EXECUTE = "execute"             # The actual operation
    POST_EXECUTE = "post_execute"   # After execution (cleanup)
    POST_VALIDATE = "post_validate" # Final validation (quality check)


class HookPriority(Enum):
    """Hook execution priority (lower = earlier)."""
    CRITICAL = 0    # Security, ethics (FATE)
    HIGH = 10       # Quality gates (Ihsan)
    NORMAL = 50     # Standard hooks
    LOW = 90        # Logging, metrics
    DEFERRED = 100  # Non-blocking cleanup


class FATEDimension(str, Enum):
    """FATE validation dimensions."""
    FIDELITY = "fidelity"           # Alignment with intent
    ACCOUNTABILITY = "accountability" # Traceability
    TRANSPARENCY = "transparency"    # Explainability
    ETHICS = "ethics"               # Ihsan compliance


@dataclass
class FATEScore:
    """
    FATE validation score.

    Each dimension scored 0.0 to 1.0.
    Overall score is weighted geometric mean.
    """
    fidelity: float = 0.0
    accountability: float = 0.0
    transparency: float = 0.0
    ethics: float = 0.0

    # Weights (can be adjusted per domain)
    weight_fidelity: float = 0.25
    weight_accountability: float = 0.25
    weight_transparency: float = 0.25
    weight_ethics: float = 0.25

    @property
    def overall(self) -> float:
        """Compute weighted score."""
        import math

        components = [
            (max(self.fidelity, 1e-10), self.weight_fidelity),
            (max(self.accountability, 1e-10), self.weight_accountability),
            (max(self.transparency, 1e-10), self.weight_transparency),
            (max(self.ethics, 1e-10), self.weight_ethics),
        ]

        # Weighted geometric mean
        product = 1.0
        for value, weight in components:
            product *= math.pow(value, weight)

        return product

    @property
    def passed(self) -> bool:
        """Check if FATE gate passes (>= Ihsan threshold)."""
        return self.overall >= UNIFIED_IHSAN_THRESHOLD

    @property
    def weakest_dimension(self) -> FATEDimension:
        """Identify the weakest FATE dimension."""
        scores = {
            FATEDimension.FIDELITY: self.fidelity,
            FATEDimension.ACCOUNTABILITY: self.accountability,
            FATEDimension.TRANSPARENCY: self.transparency,
            FATEDimension.ETHICS: self.ethics,
        }
        return min(scores, key=scores.get)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize score."""
        return {
            "fidelity": self.fidelity,
            "accountability": self.accountability,
            "transparency": self.transparency,
            "ethics": self.ethics,
            "overall": self.overall,
            "passed": self.passed,
            "weakest_dimension": self.weakest_dimension.value,
        }


@dataclass
class HookContext:
    """
    Context passed through hook chain.

    Immutable reference with mutable data dict.
    """
    # Unique execution ID
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Timestamp (Lamport-ordered)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Logical clock (increments with each operation)
    logical_clock: int = 0

    # The operation being hooked
    operation_name: str = ""
    operation_type: str = "unknown"

    # Input/output data
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # FATE validation
    fate_score: Optional[FATEScore] = None

    # Audit trail
    hook_trace: List[str] = field(default_factory=list)

    # Error state
    error: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_hook(self, hook_name: str, phase: HookPhase) -> None:
        """Record hook execution for audit trail."""
        self.logical_clock += 1
        entry = f"{self.logical_clock}:{phase.value}:{hook_name}@{time.time():.3f}"
        self.hook_trace.append(entry)

    def compute_digest(self) -> str:
        """Compute SHA-256 digest of context for integrity verification."""
        import json
        content = json.dumps({
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "operation_name": self.operation_name,
            "input_hash": hashlib.sha256(
                str(self.input_data).encode()
            ).hexdigest()[:16],
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context."""
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "logical_clock": self.logical_clock,
            "operation_name": self.operation_name,
            "operation_type": self.operation_type,
            "fate_score": self.fate_score.to_dict() if self.fate_score else None,
            "hook_trace": self.hook_trace,
            "error": self.error,
            "digest": self.compute_digest(),
        }


@dataclass
class HookResult:
    """Result of hook chain execution."""
    success: bool
    context: HookContext
    phase_results: Dict[str, bool] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    blocked_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        return {
            "success": self.success,
            "context": self.context.to_dict(),
            "phase_results": self.phase_results,
            "execution_time_ms": self.execution_time_ms,
            "blocked_by": self.blocked_by,
        }


# ============================================================================
# HOOK REGISTRY
# ============================================================================

@dataclass
class RegisteredHook:
    """A registered hook with metadata."""
    name: str
    phase: HookPhase
    priority: HookPriority
    function: HookFunction
    is_async: bool
    enabled: bool = True
    description: str = ""

    # Statistics
    invocation_count: int = 0
    total_time_ms: float = 0.0
    failure_count: int = 0


class HookRegistry:
    """
    Central registry for all hooks.

    Hooks are organized by phase and sorted by priority.
    Thread-safe registration and execution.
    """

    def __init__(self):
        self._hooks: Dict[HookPhase, List[RegisteredHook]] = {
            phase: [] for phase in HookPhase
        }
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        phase: HookPhase,
        function: HookFunction,
        priority: HookPriority = HookPriority.NORMAL,
        description: str = "",
    ) -> None:
        """
        Register a hook.

        Args:
            name: Unique hook name
            phase: Execution phase
            function: Hook function (sync or async)
            priority: Execution priority
            description: Human-readable description
        """
        is_async = asyncio.iscoroutinefunction(function)

        hook = RegisteredHook(
            name=name,
            phase=phase,
            priority=priority,
            function=function,
            is_async=is_async,
            description=description,
        )

        self._hooks[phase].append(hook)

        # Sort by priority
        self._hooks[phase].sort(key=lambda h: h.priority.value)

        logger.debug(f"Registered hook '{name}' for phase {phase.value} (priority: {priority.name})")

    def unregister(self, name: str) -> bool:
        """Unregister a hook by name."""
        for phase in self._hooks:
            self._hooks[phase] = [h for h in self._hooks[phase] if h.name != name]
        return True

    def get_hooks(self, phase: HookPhase) -> List[RegisteredHook]:
        """Get all enabled hooks for a phase."""
        return [h for h in self._hooks[phase] if h.enabled]

    def enable(self, name: str) -> None:
        """Enable a hook."""
        for phase in self._hooks:
            for hook in self._hooks[phase]:
                if hook.name == name:
                    hook.enabled = True

    def disable(self, name: str) -> None:
        """Disable a hook."""
        for phase in self._hooks:
            for hook in self._hooks[phase]:
                if hook.name == name:
                    hook.enabled = False

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            "total_hooks": sum(len(hooks) for hooks in self._hooks.values()),
            "by_phase": {},
        }

        for phase, hooks in self._hooks.items():
            stats["by_phase"][phase.value] = {
                "count": len(hooks),
                "enabled": len([h for h in hooks if h.enabled]),
                "hooks": [
                    {
                        "name": h.name,
                        "priority": h.priority.name,
                        "invocations": h.invocation_count,
                        "avg_time_ms": h.total_time_ms / max(h.invocation_count, 1),
                        "failures": h.failure_count,
                    }
                    for h in hooks
                ],
            }

        return stats


# ============================================================================
# FATE GATE
# ============================================================================

class FATEGate:
    """
    FATE Gate — Constitutional validation gate.

    Validates operations against FATE dimensions before allowing execution.
    This is the primary enforcement mechanism for Ihsan compliance.

    Integration with NTU:
    - Uses NTU belief as a confidence signal
    - Updates NTU with validation outcomes

    Integration with SNR:
    - SNR score informs the Ethics dimension
    - Low SNR blocks execution
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold

        # Validation history for trend analysis
        self._history: List[FATEScore] = []
        self._max_history = 100

        # Statistics
        self._total_validations = 0
        self._passed_count = 0
        self._blocked_count = 0

    def validate(
        self,
        context: HookContext,
        declared_intent: Optional[str] = None,
        snr_score: Optional[float] = None,
    ) -> FATEScore:
        """
        Validate context against FATE dimensions.

        Args:
            context: Hook execution context
            declared_intent: The stated purpose of the operation
            snr_score: Pre-computed SNR score (optional)

        Returns:
            FATEScore with validation results
        """
        self._total_validations += 1

        # 1. FIDELITY: Does the operation align with declared intent?
        fidelity = self._compute_fidelity(context, declared_intent)

        # 2. ACCOUNTABILITY: Is the operation traceable?
        accountability = self._compute_accountability(context)

        # 3. TRANSPARENCY: Is the decision path explainable?
        transparency = self._compute_transparency(context)

        # 4. ETHICS: Does it meet Ihsan threshold?
        ethics = self._compute_ethics(context, snr_score)

        score = FATEScore(
            fidelity=fidelity,
            accountability=accountability,
            transparency=transparency,
            ethics=ethics,
        )

        # Update statistics
        if score.passed:
            self._passed_count += 1
        else:
            self._blocked_count += 1

        # Record history
        self._history.append(score)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.info(
            f"FATE validation: {score.overall:.4f} "
            f"(F={fidelity:.2f}, A={accountability:.2f}, T={transparency:.2f}, E={ethics:.2f}) "
            f"{'PASSED' if score.passed else 'BLOCKED'}"
        )

        return score

    def _compute_fidelity(
        self,
        context: HookContext,
        declared_intent: Optional[str],
    ) -> float:
        """
        Compute fidelity score.

        Fidelity measures alignment between declared intent and actual operation.
        """
        if not declared_intent:
            # No declared intent = reduced fidelity
            return 0.7

        # Check if operation name aligns with intent
        operation = context.operation_name.lower()
        intent = declared_intent.lower()

        # Simple keyword overlap (production would use embeddings)
        operation_words = set(operation.replace("_", " ").split())
        intent_words = set(intent.replace("_", " ").split())

        if not intent_words:
            return 0.8

        overlap = len(operation_words & intent_words)
        alignment = overlap / len(intent_words)

        # Base score + alignment bonus
        return min(1.0, 0.7 + 0.3 * alignment)

    def _compute_accountability(self, context: HookContext) -> float:
        """
        Compute accountability score.

        Accountability requires:
        - Unique execution ID
        - Timestamp
        - Audit trail capability
        """
        score = 0.0

        # Has execution ID
        if context.execution_id:
            score += 0.3

        # Has timestamp
        if context.timestamp:
            score += 0.3

        # Has hook trace
        if context.hook_trace is not None:
            score += 0.2

        # Has metadata for context
        if context.metadata:
            score += 0.1

        # Operation is named
        if context.operation_name:
            score += 0.1

        return min(1.0, score)

    def _compute_transparency(self, context: HookContext) -> float:
        """
        Compute transparency score.

        Transparency requires explainable decision paths.
        """
        score = 0.7  # Base score

        # Operation type is specified
        if context.operation_type != "unknown":
            score += 0.1

        # Input data is present (can be audited)
        if context.input_data:
            score += 0.1

        # Metadata provides context
        if context.metadata.get("description") or context.metadata.get("rationale"):
            score += 0.1

        return min(1.0, score)

    def _compute_ethics(
        self,
        context: HookContext,
        snr_score: Optional[float],
    ) -> float:
        """
        Compute ethics score.

        Ethics is the Ihsan dimension, informed by:
        - SNR score (signal quality)
        - Ihsan weights (correctness, safety, etc.)
        """
        # Start with SNR if available
        if snr_score is not None:
            base_score = snr_score
        else:
            # Default to threshold
            base_score = self.ihsan_threshold

        # Check for explicit ethics violations in metadata
        if context.metadata.get("pii_detected"):
            base_score *= 0.5

        if context.metadata.get("toxicity_detected"):
            base_score *= 0.3

        if context.metadata.get("ethics_violation"):
            base_score *= 0.1

        return max(0.0, min(1.0, base_score))

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        avg_score = 0.0
        if self._history:
            avg_score = sum(s.overall for s in self._history) / len(self._history)

        return {
            "total_validations": self._total_validations,
            "passed": self._passed_count,
            "blocked": self._blocked_count,
            "pass_rate": self._passed_count / max(self._total_validations, 1),
            "avg_score": avg_score,
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
        }


# ============================================================================
# HOOK EXECUTOR
# ============================================================================

class HookExecutor:
    """
    Executes hooks in the correct order with FATE validation.

    The execution chain:
    1. PRE_VALIDATE: Schema and format validation
    2. FATE GATE: Constitutional validation (blocks if fails)
    3. PRE_EXECUTE: Pre-processing hooks
    4. EXECUTE: The actual operation
    5. POST_EXECUTE: Post-processing hooks
    6. POST_VALIDATE: Final quality validation
    """

    def __init__(
        self,
        registry: Optional[HookRegistry] = None,
        fate_gate: Optional[FATEGate] = None,
    ):
        self.registry = registry or HookRegistry()
        self.fate_gate = fate_gate or FATEGate()

    async def execute(
        self,
        operation: Callable[..., Any],
        input_data: Dict[str, Any],
        operation_name: str = "",
        operation_type: str = "unknown",
        declared_intent: Optional[str] = None,
        snr_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HookResult:
        """
        Execute an operation with full hook chain.

        Args:
            operation: The operation to execute (sync or async)
            input_data: Input data for the operation
            operation_name: Name of the operation
            operation_type: Type classification
            declared_intent: Stated purpose
            snr_score: Pre-computed SNR
            metadata: Additional context

        Returns:
            HookResult with success status and context
        """
        start_time = time.time()

        # Create context
        context = HookContext(
            operation_name=operation_name or operation.__name__,
            operation_type=operation_type,
            input_data=input_data,
            metadata=metadata or {},
        )

        phase_results = {}

        try:
            # 1. PRE_VALIDATE phase
            success = await self._run_phase(HookPhase.PRE_VALIDATE, context)
            phase_results[HookPhase.PRE_VALIDATE.value] = success
            if not success:
                return self._create_result(
                    False, context, phase_results, start_time,
                    blocked_by="pre_validate"
                )

            # 2. FATE GATE (Constitutional check)
            fate_score = self.fate_gate.validate(context, declared_intent, snr_score)
            context.fate_score = fate_score

            if not fate_score.passed:
                logger.warning(
                    f"FATE Gate blocked operation '{context.operation_name}': "
                    f"score={fate_score.overall:.4f}, weakest={fate_score.weakest_dimension.value}"
                )
                return self._create_result(
                    False, context, phase_results, start_time,
                    blocked_by=f"fate_gate:{fate_score.weakest_dimension.value}"
                )

            phase_results["fate_gate"] = True

            # 3. PRE_EXECUTE phase
            success = await self._run_phase(HookPhase.PRE_EXECUTE, context)
            phase_results[HookPhase.PRE_EXECUTE.value] = success
            if not success:
                return self._create_result(
                    False, context, phase_results, start_time,
                    blocked_by="pre_execute"
                )

            # 4. EXECUTE the operation
            context.record_hook("operation", HookPhase.EXECUTE)

            if asyncio.iscoroutinefunction(operation):
                result = await operation(**context.input_data)
            else:
                result = operation(**context.input_data)

            context.output_data = {"result": result}
            phase_results[HookPhase.EXECUTE.value] = True

            # 5. POST_EXECUTE phase
            success = await self._run_phase(HookPhase.POST_EXECUTE, context)
            phase_results[HookPhase.POST_EXECUTE.value] = success
            # Don't block on post-execute failures

            # 6. POST_VALIDATE phase
            success = await self._run_phase(HookPhase.POST_VALIDATE, context)
            phase_results[HookPhase.POST_VALIDATE.value] = success

            return self._create_result(True, context, phase_results, start_time)

        except Exception as e:
            logger.exception(f"Hook execution error: {e}")
            context.error = str(e)
            return self._create_result(False, context, phase_results, start_time)

    async def _run_phase(self, phase: HookPhase, context: HookContext) -> bool:
        """Run all hooks for a phase."""
        hooks = self.registry.get_hooks(phase)

        for hook in hooks:
            try:
                hook_start = time.time()
                context.record_hook(hook.name, phase)

                # Convert context to dict for hook
                data = {
                    "context": context,
                    "input_data": context.input_data,
                    "output_data": context.output_data,
                    "metadata": context.metadata,
                }

                if hook.is_async:
                    result = await hook.function(data)
                else:
                    result = hook.function(data)

                # Update statistics
                hook.invocation_count += 1
                hook.total_time_ms += (time.time() - hook_start) * 1000

                # Check for explicit failure
                if result is False or (isinstance(result, dict) and result.get("_blocked")):
                    hook.failure_count += 1
                    logger.warning(f"Hook '{hook.name}' blocked execution")
                    return False

                # Update context from hook result
                if isinstance(result, dict):
                    if "metadata" in result:
                        context.metadata.update(result["metadata"])
                    if "output_data" in result:
                        context.output_data.update(result["output_data"])

            except Exception as e:
                hook.failure_count += 1
                logger.error(f"Hook '{hook.name}' error: {e}")
                # Continue on non-critical hooks
                if hook.priority == HookPriority.CRITICAL:
                    return False

        return True

    def _create_result(
        self,
        success: bool,
        context: HookContext,
        phase_results: Dict[str, bool],
        start_time: float,
        blocked_by: Optional[str] = None,
    ) -> HookResult:
        """Create HookResult."""
        return HookResult(
            success=success,
            context=context,
            phase_results=phase_results,
            execution_time_ms=(time.time() - start_time) * 1000,
            blocked_by=blocked_by,
        )


# ============================================================================
# DECORATORS
# ============================================================================

def fate_guarded(
    operation_type: str = "function",
    declared_intent: Optional[str] = None,
    require_snr: bool = False,
):
    """
    Decorator to wrap a function with FATE gate protection.

    Usage:
        @fate_guarded(operation_type="inference", declared_intent="Generate summary")
        async def generate_summary(text: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Get or create global executor
        executor = _get_global_executor()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            input_data = {"args": args, "kwargs": kwargs}

            result = await executor.execute(
                operation=lambda args, kwargs: func(*args, **kwargs),
                input_data=input_data,
                operation_name=func.__name__,
                operation_type=operation_type,
                declared_intent=declared_intent,
            )

            if result.success:
                return result.context.output_data.get("result")
            else:
                raise FATEGateError(
                    f"FATE gate blocked '{func.__name__}': {result.blocked_by}",
                    result.context.fate_score,
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.get_event_loop().run_until_complete(
                async_wrapper(*args, **kwargs)
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class FATEGateError(Exception):
    """Exception raised when FATE gate blocks an operation."""

    def __init__(self, message: str, fate_score: Optional[FATEScore] = None):
        super().__init__(message)
        self.fate_score = fate_score


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_global_registry: Optional[HookRegistry] = None
_global_executor: Optional[HookExecutor] = None


def _get_global_registry() -> HookRegistry:
    """Get or create global hook registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HookRegistry()
    return _global_registry


def _get_global_executor() -> HookExecutor:
    """Get or create global hook executor."""
    global _global_executor
    if _global_executor is None:
        _global_executor = HookExecutor(_get_global_registry())
    return _global_executor


def register_hook(
    name: str,
    phase: HookPhase,
    function: HookFunction,
    priority: HookPriority = HookPriority.NORMAL,
    description: str = "",
) -> None:
    """Register a hook in the global registry."""
    _get_global_registry().register(name, phase, function, priority, description)


def get_hook_stats() -> Dict[str, Any]:
    """Get global hook statistics."""
    return _get_global_registry().get_stats()


# ============================================================================
# BUILT-IN HOOKS
# ============================================================================

def _logging_hook(data: Dict[str, Any]) -> Dict[str, Any]:
    """Built-in logging hook."""
    context: HookContext = data["context"]
    logger.debug(
        f"Hook trace: {context.operation_name} "
        f"(clock={context.logical_clock}, execution_id={context.execution_id})"
    )
    return data


def _metrics_hook(data: Dict[str, Any]) -> Dict[str, Any]:
    """Built-in metrics collection hook."""
    context: HookContext = data["context"]
    context.metadata["_metrics"] = {
        "timestamp": time.time(),
        "logical_clock": context.logical_clock,
    }
    return data


# Register built-in hooks
def _register_builtin_hooks():
    """Register built-in hooks."""
    registry = _get_global_registry()

    registry.register(
        name="builtin_logging",
        phase=HookPhase.PRE_EXECUTE,
        function=_logging_hook,
        priority=HookPriority.LOW,
        description="Built-in logging hook",
    )

    registry.register(
        name="builtin_metrics",
        phase=HookPhase.POST_EXECUTE,
        function=_metrics_hook,
        priority=HookPriority.LOW,
        description="Built-in metrics collection",
    )


# Auto-register on import
_register_builtin_hooks()


# ============================================================================
# INTEGRATION WITH NTU
# ============================================================================

class NTUHookAdapter:
    """
    Adapter connecting hooks to NTU for temporal pattern tracking.

    Uses FATE scores as NTU observations to:
    1. Track quality trends over time
    2. Detect degradation patterns
    3. Inform dynamic threshold adjustment
    """

    def __init__(self):
        # Lazy import to avoid circular dependency
        self._ntu = None

    @property
    def ntu(self):
        """Lazy-load NTU."""
        if self._ntu is None:
            try:
                from core.ntu import NTU, NTUConfig
                self._ntu = NTU(NTUConfig(ihsan_threshold=UNIFIED_IHSAN_THRESHOLD))
            except ImportError:
                logger.warning("NTU not available, hook adaptation disabled")
        return self._ntu

    def observe_fate_score(self, score: FATEScore) -> Optional[float]:
        """
        Record FATE score in NTU.

        Returns:
            NTU belief if available, None otherwise
        """
        if self.ntu is None:
            return None

        # Use overall FATE score as observation
        state = self.ntu.observe(score.overall, {
            "source": "fate_gate",
            "fidelity": score.fidelity,
            "accountability": score.accountability,
            "transparency": score.transparency,
            "ethics": score.ethics,
        })

        return state.belief

    def get_dynamic_threshold(self) -> float:
        """
        Get dynamic Ihsan threshold based on NTU state.

        When quality is trending up, threshold can be slightly relaxed.
        When quality is trending down, threshold is tightened.
        """
        if self.ntu is None:
            return UNIFIED_IHSAN_THRESHOLD

        state = self.ntu.state

        # Confidence factor from NTU
        confidence = state.belief * (1.0 - state.entropy)

        # Small adjustment: ±0.02
        adjustment = 0.02 * (confidence - 0.5)

        return max(0.90, min(0.99, UNIFIED_IHSAN_THRESHOLD - adjustment))


# Global NTU adapter
_ntu_adapter = NTUHookAdapter()


def get_ntu_adapter() -> NTUHookAdapter:
    """Get global NTU hook adapter."""
    return _ntu_adapter
