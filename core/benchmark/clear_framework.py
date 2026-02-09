"""
CLEAR FRAMEWORK — Multi-Dimensional Agent Evaluation
═══════════════════════════════════════════════════════════════════════════════

Evaluates agents on 5 dimensions beyond raw accuracy:
  C - Cost         : Token usage, API calls, compute resources
  L - Latency      : Time-to-first-token, total completion time
  E - Efficacy     : Task completion accuracy, goal achievement
  A - Assurance    : Safety, reliability, reproducibility
  R - Reliability  : Consistency across runs, failure recovery

Research shows accuracy-only optimization yields agents 4.4–10.8x more expensive
than cost-aware alternatives (HAL 2025 findings).

Giants Protocol:
  - HAL (2025): Holistic Agent Leaderboard multi-VM evaluation
  - ABC (2025): Agentic Benchmark Checklist anti-overestimation
  - Shannon (1948): Information-theoretic efficiency bounds

لا نفترض — We do not assume. We verify with formal proofs.
"""

from __future__ import annotations

import time
import hashlib
import statistics
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class CLEARDimension(Enum):
    """The 5 dimensions of CLEAR evaluation."""
    COST = auto()       # Token usage, API calls, compute
    LATENCY = auto()    # Time metrics
    EFFICACY = auto()   # Task completion accuracy
    ASSURANCE = auto()  # Safety, reliability
    RELIABILITY = auto()  # Consistency, recovery


@dataclass(frozen=True)
class MetricWeight:
    """Weight configuration for CLEAR dimensions."""
    cost: float = 0.20
    latency: float = 0.15
    efficacy: float = 0.35  # Highest weight — still primary
    assurance: float = 0.15
    reliability: float = 0.15

    def __post_init__(self):
        total = self.cost + self.latency + self.efficacy + self.assurance + self.reliability
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def as_dict(self) -> Dict[str, float]:
        return {
            "cost": self.cost,
            "latency": self.latency,
            "efficacy": self.efficacy,
            "assurance": self.assurance,
            "reliability": self.reliability,
        }


@dataclass
class CostMetrics:
    """Cost dimension metrics."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    compute_seconds: float = 0.0
    cost_usd: float = 0.0
    
    # Normalized score (0-1, higher is better = lower cost)
    def score(self, budget_tokens: int = 100_000, budget_usd: float = 1.0) -> float:
        """Calculate cost efficiency score."""
        token_ratio = 1.0 - min(self.total_tokens / budget_tokens, 1.0)
        cost_ratio = 1.0 - min(self.cost_usd / budget_usd, 1.0)
        return 0.6 * token_ratio + 0.4 * cost_ratio


@dataclass
class LatencyMetrics:
    """Latency dimension metrics."""
    time_to_first_token_ms: float = 0.0
    total_completion_ms: float = 0.0
    tokens_per_second: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    def score(self, target_ttft_ms: float = 500, target_total_ms: float = 10_000) -> float:
        """Calculate latency score (faster = higher)."""
        ttft_score = 1.0 - min(self.time_to_first_token_ms / target_ttft_ms, 1.0)
        total_score = 1.0 - min(self.total_completion_ms / target_total_ms, 1.0)
        return 0.4 * ttft_score + 0.6 * total_score


@dataclass
class EfficacyMetrics:
    """Efficacy dimension metrics (task completion)."""
    accuracy: float = 0.0
    task_completion_rate: float = 0.0
    goal_achievement: float = 0.0
    partial_credit: float = 0.0
    
    def score(self) -> float:
        """Calculate efficacy score."""
        # Weighted combination — accuracy is primary but not only metric
        return (
            0.40 * self.accuracy +
            0.30 * self.task_completion_rate +
            0.20 * self.goal_achievement +
            0.10 * self.partial_credit
        )


@dataclass
class AssuranceMetrics:
    """Assurance dimension metrics (safety + reliability)."""
    safety_violations: int = 0
    hallucination_rate: float = 0.0
    reproducibility: float = 0.0
    graceful_failures: int = 0
    ungraceful_failures: int = 0
    
    def score(self) -> float:
        """Calculate assurance score."""
        safety_score = 1.0 if self.safety_violations == 0 else max(0, 1.0 - 0.2 * self.safety_violations)
        hallucination_score = 1.0 - self.hallucination_rate
        failure_ratio = self.graceful_failures / max(1, self.graceful_failures + self.ungraceful_failures)
        return 0.35 * safety_score + 0.25 * hallucination_score + 0.25 * self.reproducibility + 0.15 * failure_ratio


@dataclass
class ReliabilityMetrics:
    """Reliability dimension metrics (consistency)."""
    consistency_across_runs: float = 0.0
    recovery_rate: float = 0.0
    variance: float = 0.0
    runs_completed: int = 0
    runs_failed: int = 0
    
    def score(self) -> float:
        """Calculate reliability score."""
        completion_rate = self.runs_completed / max(1, self.runs_completed + self.runs_failed)
        variance_penalty = min(self.variance, 0.5)  # Cap penalty
        return (
            0.40 * self.consistency_across_runs +
            0.30 * completion_rate +
            0.20 * self.recovery_rate +
            0.10 * (1.0 - variance_penalty)
        )


@dataclass
class CLEARMetrics:
    """Complete CLEAR metrics bundle."""
    cost: CostMetrics = field(default_factory=CostMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    efficacy: EfficacyMetrics = field(default_factory=EfficacyMetrics)
    assurance: AssuranceMetrics = field(default_factory=AssuranceMetrics)
    reliability: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)
    
    # Metadata
    task_id: str = ""
    agent_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_hash: str = ""
    
    def compute_overall_score(self, weights: MetricWeight = MetricWeight()) -> float:
        """Compute weighted CLEAR score."""
        return (
            weights.cost * self.cost.score() +
            weights.latency * self.latency.score() +
            weights.efficacy * self.efficacy.score() +
            weights.assurance * self.assurance.score() +
            weights.reliability * self.reliability.score()
        )
    
    def dimension_scores(self) -> Dict[str, float]:
        """Get individual dimension scores."""
        return {
            "cost": self.cost.score(),
            "latency": self.latency.score(),
            "efficacy": self.efficacy.score(),
            "assurance": self.assurance.score(),
            "reliability": self.reliability.score(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "run_hash": self.run_hash,
            "overall_score": self.compute_overall_score(),
            "dimension_scores": self.dimension_scores(),
            "cost": {
                "total_tokens": self.cost.total_tokens,
                "api_calls": self.cost.api_calls,
                "cost_usd": self.cost.cost_usd,
            },
            "latency": {
                "ttft_ms": self.latency.time_to_first_token_ms,
                "total_ms": self.latency.total_completion_ms,
                "tokens_per_second": self.latency.tokens_per_second,
            },
            "efficacy": {
                "accuracy": self.efficacy.accuracy,
                "task_completion": self.efficacy.task_completion_rate,
            },
            "assurance": {
                "safety_violations": self.assurance.safety_violations,
                "hallucination_rate": self.assurance.hallucination_rate,
            },
            "reliability": {
                "consistency": self.reliability.consistency_across_runs,
                "runs_completed": self.reliability.runs_completed,
            },
        }


class AgenticBenchmarkChecklist:
    """
    ABC — Agentic Benchmark Checklist
    
    Rigorous guidelines to prevent overestimation of performance (up to 100%).
    Based on research showing flawed reward designs and insufficient test cases
    lead to dramatic performance overestimation.
    """
    
    CHECKS = [
        ("sufficient_test_cases", "Minimum 100 test cases per task type"),
        ("diverse_task_distribution", "Tasks span multiple difficulty levels"),
        ("no_reward_hacking", "Reward signal resistant to gaming"),
        ("temporal_holdout", "Test data from after training cutoff"),
        ("adversarial_probes", "Includes adversarial perturbations"),
        ("null_model_baseline", "Generic response baseline computed"),
        ("human_baseline", "Human performance benchmark available"),
        ("multi_run_consistency", "Results averaged across 3+ runs"),
        ("cost_tracking", "Full cost accounting enabled"),
        ("failure_analysis", "Failure modes categorized and reported"),
    ]
    
    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.notes: Dict[str, str] = {}
    
    def validate(self, benchmark_config: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """
        Validate benchmark against ABC checklist.
        
        Returns:
            Tuple of (passed, score 0-1, list of failed checks)
        """
        failed = []
        
        for check_id, description in self.CHECKS:
            passed = benchmark_config.get(check_id, False)
            self.results[check_id] = passed
            if not passed:
                failed.append(f"[{check_id}] {description}")
        
        score = sum(self.results.values()) / len(self.CHECKS)
        all_passed = len(failed) == 0
        
        return all_passed, score, failed
    
    def generate_report(self) -> str:
        """Generate human-readable ABC report."""
        lines = ["═══ AGENTIC BENCHMARK CHECKLIST (ABC) ═══", ""]
        
        for check_id, description in self.CHECKS:
            status = "✅" if self.results.get(check_id, False) else "❌"
            lines.append(f"  {status} {check_id}: {description}")
        
        score = sum(self.results.values()) / len(self.CHECKS) if self.results else 0
        lines.append("")
        lines.append(f"ABC Score: {score:.1%}")
        lines.append(f"Status: {'PASSED' if score >= 0.8 else 'FAILED'}")
        
        return "\n".join(lines)


class CLEARFramework:
    """
    CLEAR Framework — Multi-dimensional agent evaluation.
    
    Replaces accuracy-only benchmarking with holistic measurement.
    
    Example:
        >>> framework = CLEARFramework()
        >>> with framework.evaluate("task-001", "agent-alpha") as ctx:
        ...     result = agent.run(task)
        ...     ctx.record_efficacy(accuracy=0.95, completion=1.0)
        ...     ctx.record_cost(tokens=5000, usd=0.01)
        >>> metrics = framework.get_metrics("task-001")
        >>> print(f"CLEAR Score: {metrics.compute_overall_score():.3f}")
    """
    
    # Ihsān thresholds for CLEAR
    IHSAN_THRESHOLD = 0.95
    ACCEPTABLE_THRESHOLD = 0.85
    MINIMUM_THRESHOLD = 0.70
    
    def __init__(
        self,
        weights: Optional[MetricWeight] = None,
        enable_abc: bool = True,
    ):
        self.weights = weights or MetricWeight()
        self.enable_abc = enable_abc
        self.abc_checker = AgenticBenchmarkChecklist() if enable_abc else None
        
        self._evaluations: Dict[str, CLEARMetrics] = {}
        self._run_history: List[CLEARMetrics] = []
        
        logger.info(
            f"CLEAR Framework initialized with weights: "
            f"C={self.weights.cost:.2f}, L={self.weights.latency:.2f}, "
            f"E={self.weights.efficacy:.2f}, A={self.weights.assurance:.2f}, "
            f"R={self.weights.reliability:.2f}"
        )
    
    def evaluate(self, task_id: str, agent_id: str) -> "EvaluationContext":
        """
        Start an evaluation context for a task.
        
        Usage:
            with framework.evaluate("task-001", "agent-v1") as ctx:
                # Run agent
                ctx.record_efficacy(accuracy=0.95)
        """
        return EvaluationContext(self, task_id, agent_id)
    
    def _record_metrics(self, metrics: CLEARMetrics) -> None:
        """Record completed evaluation metrics."""
        # Generate run hash for reproducibility
        content = f"{metrics.task_id}:{metrics.agent_id}:{metrics.timestamp}"
        metrics.run_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        self._evaluations[metrics.task_id] = metrics
        self._run_history.append(metrics)
        
        score = metrics.compute_overall_score(self.weights)
        logger.info(
            f"CLEAR evaluation recorded: task={metrics.task_id}, "
            f"agent={metrics.agent_id}, score={score:.4f}"
        )
    
    def get_metrics(self, task_id: str) -> Optional[CLEARMetrics]:
        """Retrieve metrics for a task."""
        return self._evaluations.get(task_id)
    
    def compute_aggregate(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all evaluations."""
        if not self._run_history:
            return {"count": 0, "aggregate_score": 0.0}
        
        scores = [m.compute_overall_score(self.weights) for m in self._run_history]
        
        return {
            "count": len(self._run_history),
            "aggregate_score": statistics.mean(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min_score": min(scores),
            "max_score": max(scores),
            "p50_score": statistics.median(scores),
            "ihsan_rate": sum(1 for s in scores if s >= self.IHSAN_THRESHOLD) / len(scores),
        }
    
    def validate_benchmark(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate benchmark configuration against ABC checklist."""
        if not self.enable_abc or not self.abc_checker:
            return True, "ABC validation disabled"
        
        passed, score, failed = self.abc_checker.validate(config)
        report = self.abc_checker.generate_report()
        
        return passed, report
    
    def compare_agents(
        self,
        agent_ids: List[str],
        task_ids: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple agents across tasks.
        
        Returns per-agent aggregate CLEAR scores.
        """
        comparisons = {}
        
        for agent_id in agent_ids:
            agent_metrics = [
                m for m in self._run_history
                if m.agent_id == agent_id
                and (task_ids is None or m.task_id in task_ids)
            ]
            
            if not agent_metrics:
                comparisons[agent_id] = {"count": 0, "aggregate_score": 0.0}
                continue
            
            scores = [m.compute_overall_score(self.weights) for m in agent_metrics]
            dimension_scores = {
                dim: statistics.mean([m.dimension_scores()[dim] for m in agent_metrics])
                for dim in ["cost", "latency", "efficacy", "assurance", "reliability"]
            }
            
            comparisons[agent_id] = {
                "count": len(agent_metrics),
                "aggregate_score": statistics.mean(scores),
                **dimension_scores,
            }
        
        return comparisons
    
    def identify_weakest_dimension(
        self,
        agent_id: str,
    ) -> Tuple[CLEARDimension, float]:
        """
        Identify the weakest dimension for an agent.
        
        Returns the dimension and its score for targeted improvement.
        """
        agent_metrics = [m for m in self._run_history if m.agent_id == agent_id]
        
        if not agent_metrics:
            return CLEARDimension.EFFICACY, 0.0  # Default focus
        
        avg_scores = {}
        for dim in ["cost", "latency", "efficacy", "assurance", "reliability"]:
            avg_scores[dim] = statistics.mean([
                m.dimension_scores()[dim] for m in agent_metrics
            ])
        
        weakest = min(avg_scores, key=lambda k: avg_scores[k])
        dimension_map = {
            "cost": CLEARDimension.COST,
            "latency": CLEARDimension.LATENCY,
            "efficacy": CLEARDimension.EFFICACY,
            "assurance": CLEARDimension.ASSURANCE,
            "reliability": CLEARDimension.RELIABILITY,
        }
        
        return dimension_map[weakest], avg_scores[weakest]


class EvaluationContext:
    """
    Context manager for running CLEAR evaluations.
    
    Automatically captures timing and provides methods to record metrics.
    """
    
    def __init__(self, framework: CLEARFramework, task_id: str, agent_id: str):
        self.framework = framework
        self.metrics = CLEARMetrics(task_id=task_id, agent_id=agent_id)
        self._start_time: float = 0.0
        self._first_token_time: Optional[float] = None
    
    def __enter__(self) -> "EvaluationContext":
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Record total latency
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self.metrics.latency.total_completion_ms = elapsed_ms
        
        # Compute tokens per second if we have token count
        if self.metrics.latency.total_completion_ms > 0 and self.metrics.cost.output_tokens > 0:
            self.metrics.latency.tokens_per_second = (
                self.metrics.cost.output_tokens / (self.metrics.latency.total_completion_ms / 1000)
            )
        
        # Record to framework
        self.framework._record_metrics(self.metrics)
    
    def mark_first_token(self) -> None:
        """Call when first token is generated."""
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
            self.metrics.latency.time_to_first_token_ms = (
                (self._first_token_time - self._start_time) * 1000
            )
    
    def record_cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        api_calls: int = 1,
        compute_seconds: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record cost metrics."""
        self.metrics.cost.input_tokens = input_tokens
        self.metrics.cost.output_tokens = output_tokens
        self.metrics.cost.total_tokens = input_tokens + output_tokens
        self.metrics.cost.api_calls = api_calls
        self.metrics.cost.compute_seconds = compute_seconds
        self.metrics.cost.cost_usd = cost_usd
    
    def record_efficacy(
        self,
        accuracy: float = 0.0,
        task_completion: float = 0.0,
        goal_achievement: float = 0.0,
        partial_credit: float = 0.0,
    ) -> None:
        """Record efficacy metrics."""
        self.metrics.efficacy.accuracy = accuracy
        self.metrics.efficacy.task_completion_rate = task_completion
        self.metrics.efficacy.goal_achievement = goal_achievement
        self.metrics.efficacy.partial_credit = partial_credit
    
    def record_assurance(
        self,
        safety_violations: int = 0,
        hallucination_rate: float = 0.0,
        reproducibility: float = 0.0,
        graceful_failures: int = 0,
        ungraceful_failures: int = 0,
    ) -> None:
        """Record assurance metrics."""
        self.metrics.assurance.safety_violations = safety_violations
        self.metrics.assurance.hallucination_rate = hallucination_rate
        self.metrics.assurance.reproducibility = reproducibility
        self.metrics.assurance.graceful_failures = graceful_failures
        self.metrics.assurance.ungraceful_failures = ungraceful_failures
    
    def record_reliability(
        self,
        consistency: float = 0.0,
        recovery_rate: float = 0.0,
        variance: float = 0.0,
        runs_completed: int = 1,
        runs_failed: int = 0,
    ) -> None:
        """Record reliability metrics."""
        self.metrics.reliability.consistency_across_runs = consistency
        self.metrics.reliability.recovery_rate = recovery_rate
        self.metrics.reliability.variance = variance
        self.metrics.reliability.runs_completed = runs_completed
        self.metrics.reliability.runs_failed = runs_failed


# ════════════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    print("═" * 80)
    print("CLEAR FRAMEWORK — Multi-Dimensional Agent Evaluation")
    print("═" * 80)
    
    # Initialize framework
    framework = CLEARFramework()
    
    # Validate benchmark configuration
    benchmark_config = {
        "sufficient_test_cases": True,
        "diverse_task_distribution": True,
        "no_reward_hacking": True,
        "temporal_holdout": True,
        "adversarial_probes": True,
        "null_model_baseline": True,
        "human_baseline": False,  # Missing
        "multi_run_consistency": True,
        "cost_tracking": True,
        "failure_analysis": False,  # Missing
    }
    
    passed, report = framework.validate_benchmark(benchmark_config)
    print(f"\n{report}")
    
    # Simulate agent evaluations
    print("\n" + "─" * 40)
    print("Running CLEAR Evaluations...")
    print("─" * 40)
    
    # Agent Alpha — high accuracy, high cost
    with framework.evaluate("task-001", "agent-alpha") as ctx:
        time.sleep(0.1)  # Simulate work
        ctx.mark_first_token()
        time.sleep(0.05)
        ctx.record_cost(input_tokens=2000, output_tokens=500, cost_usd=0.05)
        ctx.record_efficacy(accuracy=0.98, task_completion=1.0, goal_achievement=0.95)
        ctx.record_assurance(safety_violations=0, hallucination_rate=0.02, reproducibility=0.95)
        ctx.record_reliability(consistency=0.92, recovery_rate=0.90, runs_completed=5)
    
    # Agent Beta — balanced
    with framework.evaluate("task-001", "agent-beta") as ctx:
        time.sleep(0.05)
        ctx.mark_first_token()
        time.sleep(0.03)
        ctx.record_cost(input_tokens=800, output_tokens=300, cost_usd=0.01)
        ctx.record_efficacy(accuracy=0.92, task_completion=0.95, goal_achievement=0.88)
        ctx.record_assurance(safety_violations=0, hallucination_rate=0.05, reproducibility=0.90)
        ctx.record_reliability(consistency=0.88, recovery_rate=0.85, runs_completed=5)
    
    # Agent Gamma — cost-optimized
    with framework.evaluate("task-001", "agent-gamma") as ctx:
        time.sleep(0.02)
        ctx.mark_first_token()
        time.sleep(0.01)
        ctx.record_cost(input_tokens=300, output_tokens=100, cost_usd=0.002)
        ctx.record_efficacy(accuracy=0.85, task_completion=0.88, goal_achievement=0.80)
        ctx.record_assurance(safety_violations=1, hallucination_rate=0.08, reproducibility=0.80)
        ctx.record_reliability(consistency=0.82, recovery_rate=0.75, runs_completed=5, runs_failed=1)
    
    # Compare agents
    print("\n" + "─" * 40)
    print("Agent Comparison")
    print("─" * 40)
    
    comparison = framework.compare_agents(["agent-alpha", "agent-beta", "agent-gamma"])
    
    for agent_id, scores in sorted(comparison.items(), key=lambda x: x[1]["aggregate_score"], reverse=True):
        print(f"\n{agent_id}:")
        print(f"  Aggregate CLEAR Score: {scores['aggregate_score']:.4f}")
        print(f"  Cost: {scores['cost']:.3f} | Latency: {scores['latency']:.3f} | "
              f"Efficacy: {scores['efficacy']:.3f}")
        print(f"  Assurance: {scores['assurance']:.3f} | Reliability: {scores['reliability']:.3f}")
        
        # Identify weakest dimension
        weakest, weak_score = framework.identify_weakest_dimension(agent_id)
        print(f"  ⚠️  Weakest: {weakest.name} ({weak_score:.3f})")
    
    # Aggregate statistics
    aggregate = framework.compute_aggregate()
    print("\n" + "─" * 40)
    print("Aggregate Statistics")
    print("─" * 40)
    print(f"  Total evaluations: {aggregate['count']}")
    print(f"  Mean CLEAR Score: {aggregate['aggregate_score']:.4f}")
    print(f"  Std Dev: {aggregate['std_dev']:.4f}")
    print(f"  Ihsān Rate (≥0.95): {aggregate['ihsan_rate']:.1%}")
    
    print("\n" + "═" * 80)
    print("لا نفترض — We do not assume. We verify with formal proofs.")
    print("إحسان — Excellence in all things.")
    print("═" * 80)
