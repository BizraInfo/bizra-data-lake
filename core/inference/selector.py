"""
BIZRA ADAPTIVE MODEL SELECTOR
═══════════════════════════════════════════════════════════════════════════════

Intelligent model selection based on:
- Task complexity (simple → complex)
- Available compute (edge → local → pool)
- Latency requirements (real-time → batch)
- Token budget (cheap → quality)

Giants Protocol:
- Al-Khwarizmi: Algorithmic routing decisions
- Ibn Rushd: Rational cost-benefit analysis
- Al-Biruni: Empirical performance measurement

Created: 2026-01-30 | BIZRA Sovereignty
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ModelTier(str, Enum):
    """Model capability tiers."""

    NANO = "nano"  # 0.5B - Fast, simple tasks
    MICRO = "micro"  # 1.5B - Routine tasks
    SMALL = "small"  # 3B - Standard tasks
    MEDIUM = "medium"  # 7B - Complex tasks
    LARGE = "large"  # 14B+ - Expert tasks
    POOL = "pool"  # 70B+ - Federated compute


class TaskComplexity(str, Enum):
    """Task complexity levels."""

    TRIVIAL = "trivial"  # Greeting, acknowledgment
    SIMPLE = "simple"  # Factual Q&A, classification
    MODERATE = "moderate"  # Explanation, summarization
    COMPLEX = "complex"  # Analysis, reasoning
    EXPERT = "expert"  # Research, synthesis, creation


class LatencyClass(str, Enum):
    """Latency requirements."""

    REALTIME = "realtime"  # < 200ms TTFT (voice)
    FAST = "fast"  # < 500ms TTFT (chat)
    NORMAL = "normal"  # < 2s TTFT (standard)
    BATCH = "batch"  # > 2s acceptable


@dataclass
class ModelSpec:
    """Specification for a model."""

    name: str
    tier: ModelTier
    params_b: float
    context_length: int

    # Performance characteristics (measured)
    speed_gpu: float = 0.0  # tok/s with GPU
    speed_cpu: float = 0.0  # tok/s CPU only
    ttft_gpu_ms: float = 0.0  # Time to first token (GPU)
    ttft_cpu_ms: float = 0.0  # Time to first token (CPU)

    # Resource requirements
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    disk_gb: float = 0.0

    # Capabilities
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_code: bool = True

    # Paths
    gguf_file: str = ""
    hf_repo: str = ""


@dataclass
class TaskProfile:
    """Profile of a task for routing."""

    complexity: TaskComplexity
    latency_class: LatencyClass
    estimated_input_tokens: int
    estimated_output_tokens: int
    requires_function_calling: bool = False
    requires_vision: bool = False
    requires_code: bool = False

    @property
    def total_tokens(self) -> int:
        return self.estimated_input_tokens + self.estimated_output_tokens


@dataclass
class SelectionResult:
    """Result of model selection."""

    model: ModelSpec
    reason: str
    fallback: Optional[ModelSpec] = None
    estimated_latency_ms: float = 0.0
    estimated_cost: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class ModelRegistry:
    """Registry of available models with specs."""

    def __init__(self):
        self._models: Dict[str, ModelSpec] = {}
        self._load_default_models()

    def _load_default_models(self):
        """Load default Qwen model family."""

        # Qwen 2.5 family (Q4_K_M quantization)
        self.register(
            ModelSpec(
                name="qwen2.5-0.5b",
                tier=ModelTier.NANO,
                params_b=0.5,
                context_length=32768,
                speed_gpu=80.0,  # Estimated
                speed_cpu=8.63,  # Measured Day 1
                ttft_gpu_ms=50,
                ttft_cpu_ms=500,
                vram_gb=0.5,
                ram_gb=1.0,
                disk_gb=0.5,
                gguf_file="qwen2.5-0.5b-instruct-q4_k_m.gguf",
                hf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            )
        )

        self.register(
            ModelSpec(
                name="qwen2.5-1.5b",
                tier=ModelTier.MICRO,
                params_b=1.5,
                context_length=32768,
                speed_gpu=50.0,
                speed_cpu=4.0,
                ttft_gpu_ms=80,
                ttft_cpu_ms=1000,
                vram_gb=1.2,
                ram_gb=2.5,
                disk_gb=1.1,
                gguf_file="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                hf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            )
        )

        self.register(
            ModelSpec(
                name="qwen2.5-3b",
                tier=ModelTier.SMALL,
                params_b=3.0,
                context_length=32768,
                speed_gpu=35.0,
                speed_cpu=2.5,
                ttft_gpu_ms=120,
                ttft_cpu_ms=2000,
                vram_gb=2.5,
                ram_gb=4.5,
                disk_gb=2.0,
                gguf_file="qwen2.5-3b-instruct-q4_k_m.gguf",
                hf_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
            )
        )

        self.register(
            ModelSpec(
                name="qwen2.5-7b",
                tier=ModelTier.MEDIUM,
                params_b=7.0,
                context_length=32768,
                speed_gpu=25.0,
                speed_cpu=1.5,
                ttft_gpu_ms=200,
                ttft_cpu_ms=4000,
                vram_gb=5.0,
                ram_gb=9.0,
                disk_gb=4.5,
                supports_function_calling=True,
                gguf_file="qwen2.5-7b-instruct-q4_k_m.gguf",
                hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
            )
        )

        self.register(
            ModelSpec(
                name="qwen2.5-14b",
                tier=ModelTier.LARGE,
                params_b=14.0,
                context_length=32768,
                speed_gpu=15.0,
                speed_cpu=0.8,
                ttft_gpu_ms=400,
                ttft_cpu_ms=8000,
                vram_gb=10.0,
                ram_gb=18.0,
                disk_gb=9.0,
                supports_function_calling=True,
                gguf_file="qwen2.5-14b-instruct-q4_k_m.gguf",
                hf_repo="Qwen/Qwen2.5-14B-Instruct-GGUF",
            )
        )

    def register(self, model: ModelSpec):
        """Register a model."""
        self._models[model.name] = model

    def get(self, name: str) -> Optional[ModelSpec]:
        """Get model by name."""
        return self._models.get(name)

    def get_by_tier(self, tier: ModelTier) -> List[ModelSpec]:
        """Get all models of a tier."""
        return [m for m in self._models.values() if m.tier == tier]

    def list_all(self) -> List[ModelSpec]:
        """List all registered models."""
        return list(self._models.values())


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════


class AdaptiveModelSelector:
    """
    Selects optimal model based on task profile and system state.

    Selection Algorithm (Al-Khwarizmi's method):
    1. Filter by capability (vision, function calling, code)
    2. Filter by latency requirement
    3. Filter by available resources
    4. Select minimum viable tier for complexity
    5. Apply cost optimization if multiple candidates
    """

    # Complexity → Minimum tier mapping
    COMPLEXITY_TIERS: Dict[TaskComplexity, ModelTier] = {
        TaskComplexity.TRIVIAL: ModelTier.NANO,
        TaskComplexity.SIMPLE: ModelTier.NANO,
        TaskComplexity.MODERATE: ModelTier.MICRO,
        TaskComplexity.COMPLEX: ModelTier.SMALL,
        TaskComplexity.EXPERT: ModelTier.MEDIUM,
    }

    # Latency → Maximum TTFT mapping (ms)
    LATENCY_LIMITS: Dict[LatencyClass, float] = {
        LatencyClass.REALTIME: 200,
        LatencyClass.FAST: 500,
        LatencyClass.NORMAL: 2000,
        LatencyClass.BATCH: float("inf"),
    }

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        gpu_available: bool = True,
        vram_gb: float = 24.0,  # RTX 4090
        ram_gb: float = 64.0,
    ):
        self.registry = registry or ModelRegistry()
        self.gpu_available = gpu_available
        self.vram_gb = vram_gb
        self.ram_gb = ram_gb

        # Performance history for adaptive optimization
        self._history: List[Tuple[str, float, float]] = []  # (model, expected, actual)

    def select(self, task: TaskProfile) -> SelectionResult:
        """
        Select optimal model for task.

        Returns the best model and reasoning.
        """
        candidates = self._get_candidates(task)

        if not candidates:
            # Fallback to smallest model
            smallest = min(self.registry.list_all(), key=lambda m: m.params_b)
            return SelectionResult(
                model=smallest,
                reason="No suitable model found, using smallest available",
            )

        # Select best candidate (smallest that meets requirements)
        best = min(candidates, key=lambda m: m.params_b)

        # Find fallback (next tier up)
        fallback = None
        for model in sorted(candidates, key=lambda m: m.params_b):
            if model.params_b > best.params_b:
                fallback = model
                break

        # Estimate latency
        speed = best.speed_gpu if self.gpu_available else best.speed_cpu
        estimated_time = (
            task.estimated_output_tokens / speed if speed > 0 else float("inf")
        )
        ttft = best.ttft_gpu_ms if self.gpu_available else best.ttft_cpu_ms

        return SelectionResult(
            model=best,
            reason=self._explain_selection(task, best),
            fallback=fallback,
            estimated_latency_ms=ttft + (estimated_time * 1000),
        )

    def _get_candidates(self, task: TaskProfile) -> List[ModelSpec]:
        """Get models that can handle the task."""
        min_tier = self.COMPLEXITY_TIERS.get(task.complexity, ModelTier.MICRO)
        max_ttft = self.LATENCY_LIMITS.get(task.latency_class, float("inf"))

        candidates = []

        for model in self.registry.list_all():
            # Check tier (must meet minimum)
            if self._tier_rank(model.tier) < self._tier_rank(min_tier):
                continue

            # Check capabilities
            if task.requires_function_calling and not model.supports_function_calling:
                continue
            if task.requires_vision and not model.supports_vision:
                continue

            # Check latency
            ttft = model.ttft_gpu_ms if self.gpu_available else model.ttft_cpu_ms
            if ttft > max_ttft:
                continue

            # Check resources
            if self.gpu_available and model.vram_gb > self.vram_gb:
                continue
            if not self.gpu_available and model.ram_gb > self.ram_gb:
                continue

            # Check context fits
            if task.total_tokens > model.context_length:
                continue

            candidates.append(model)

        return candidates

    def _tier_rank(self, tier: ModelTier) -> int:
        """Get numeric rank of tier for comparison."""
        ranks = {
            ModelTier.NANO: 0,
            ModelTier.MICRO: 1,
            ModelTier.SMALL: 2,
            ModelTier.MEDIUM: 3,
            ModelTier.LARGE: 4,
            ModelTier.POOL: 5,
        }
        return ranks.get(tier, 0)

    def _explain_selection(self, task: TaskProfile, model: ModelSpec) -> str:
        """Generate human-readable explanation."""
        parts = [
            f"Selected {model.name} ({model.params_b}B)",
            f"for {task.complexity.value} task",
            f"with {task.latency_class.value} latency requirement",
        ]

        if self.gpu_available:
            parts.append(f"(GPU: ~{model.speed_gpu:.0f} tok/s)")
        else:
            parts.append(f"(CPU: ~{model.speed_cpu:.0f} tok/s)")

        return " ".join(parts)

    def record_performance(
        self, model_name: str, expected_speed: float, actual_speed: float
    ):
        """Record actual performance for adaptive learning."""
        self._history.append((model_name, expected_speed, actual_speed))

        # Update model specs based on actual performance
        if len(self._history) > 10:
            self._update_estimates()

    def _update_estimates(self):
        """Update speed estimates based on history."""
        # Group by model
        by_model: Dict[str, List[float]] = {}
        for name, _, actual in self._history[-100:]:  # Last 100 samples
            if name not in by_model:
                by_model[name] = []
            by_model[name].append(actual)

        # Update registry
        for name, speeds in by_model.items():
            model = self.registry.get(name)
            if model and speeds:
                avg_speed = sum(speeds) / len(speeds)
                if self.gpu_available:
                    model.speed_gpu = avg_speed
                else:
                    model.speed_cpu = avg_speed


# ═══════════════════════════════════════════════════════════════════════════════
# TASK ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════


class TaskAnalyzer:
    """
    Analyzes prompts to determine task profile.

    Uses heuristics to estimate:
    - Complexity (based on keywords, length, structure)
    - Output length (based on task type)
    - Required capabilities (function calling, vision, code)
    """

    # Keywords indicating complexity
    TRIVIAL_PATTERNS = ["hi", "hello", "thanks", "ok", "yes", "no"]
    SIMPLE_PATTERNS = ["what is", "who is", "when", "where", "define"]
    MODERATE_PATTERNS = ["explain", "describe", "summarize", "how does"]
    COMPLEX_PATTERNS = ["analyze", "compare", "evaluate", "why", "what if"]
    EXPERT_PATTERNS = ["prove", "synthesize", "design", "create", "research"]

    # Keywords indicating capabilities
    CODE_PATTERNS = [
        "code",
        "function",
        "class",
        "implement",
        "debug",
        "python",
        "javascript",
    ]
    VISION_PATTERNS = ["image", "picture", "photo", "screenshot", "diagram"]
    FUNCTION_PATTERNS = ["search", "calculate", "lookup", "fetch", "call"]

    def analyze(
        self, prompt: str, latency_class: LatencyClass = LatencyClass.NORMAL
    ) -> TaskProfile:
        """Analyze prompt and return task profile."""
        prompt_lower = prompt.lower()
        words = prompt_lower.split()

        # Determine complexity
        complexity = self._estimate_complexity(prompt_lower, words)

        # Estimate tokens
        input_tokens = len(words) * 1.3  # Rough estimate
        output_tokens = self._estimate_output_tokens(complexity, prompt_lower)

        # Detect capabilities
        requires_code = any(p in prompt_lower for p in self.CODE_PATTERNS)
        requires_vision = any(p in prompt_lower for p in self.VISION_PATTERNS)
        requires_function = any(p in prompt_lower for p in self.FUNCTION_PATTERNS)

        return TaskProfile(
            complexity=complexity,
            latency_class=latency_class,
            estimated_input_tokens=int(input_tokens),
            estimated_output_tokens=int(output_tokens),
            requires_function_calling=requires_function,
            requires_vision=requires_vision,
            requires_code=requires_code,
        )

    def _estimate_complexity(
        self, prompt_lower: str, words: List[str]
    ) -> TaskComplexity:
        """Estimate task complexity from prompt."""
        # Check patterns in order of complexity
        if any(p in prompt_lower for p in self.EXPERT_PATTERNS):
            return TaskComplexity.EXPERT
        if any(p in prompt_lower for p in self.COMPLEX_PATTERNS):
            return TaskComplexity.COMPLEX
        if any(p in prompt_lower for p in self.MODERATE_PATTERNS):
            return TaskComplexity.MODERATE
        if any(p in prompt_lower for p in self.SIMPLE_PATTERNS):
            return TaskComplexity.SIMPLE
        if len(words) <= 5 or any(p in prompt_lower for p in self.TRIVIAL_PATTERNS):
            return TaskComplexity.TRIVIAL

        # Default based on length
        if len(words) > 100:
            return TaskComplexity.COMPLEX
        elif len(words) > 50:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    def _estimate_output_tokens(
        self, complexity: TaskComplexity, prompt_lower: str
    ) -> int:
        """Estimate expected output tokens."""
        base_tokens = {
            TaskComplexity.TRIVIAL: 20,
            TaskComplexity.SIMPLE: 100,
            TaskComplexity.MODERATE: 300,
            TaskComplexity.COMPLEX: 500,
            TaskComplexity.EXPERT: 1000,
        }

        tokens = base_tokens.get(complexity, 200)

        # Adjust for code requests (usually longer)
        if any(p in prompt_lower for p in self.CODE_PATTERNS):
            tokens = int(tokens * 1.5)

        return int(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_selector_instance: Optional[AdaptiveModelSelector] = None
_analyzer_instance: Optional[TaskAnalyzer] = None


def get_model_selector(gpu_available: bool = True) -> AdaptiveModelSelector:
    """Get singleton model selector."""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = AdaptiveModelSelector(gpu_available=gpu_available)
    return _selector_instance


def get_task_analyzer() -> TaskAnalyzer:
    """Get singleton task analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TaskAnalyzer()
    return _analyzer_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("═" * 70)
    print("    BIZRA ADAPTIVE MODEL SELECTOR")
    print("═" * 70)

    # Test prompts
    test_prompts = [
        "Hi there!",
        "What is Python?",
        "Explain how neural networks learn",
        "Analyze the trade-offs between microservices and monolithic architectures",
        "Design a distributed consensus algorithm with Byzantine fault tolerance",
        "Write a Python function to calculate fibonacci numbers",
    ]

    analyzer = get_task_analyzer()
    selector = get_model_selector(gpu_available=True)

    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt[:50]}...")

        task = analyzer.analyze(prompt)
        result = selector.select(task)

        print(f"   Complexity: {task.complexity.value}")
        print(
            f"   Tokens: ~{task.estimated_input_tokens} in, ~{task.estimated_output_tokens} out"
        )
        print(f"   → {result.reason}")
        print(f"   Estimated latency: {result.estimated_latency_ms:.0f}ms")

    print("\n" + "═" * 70)
