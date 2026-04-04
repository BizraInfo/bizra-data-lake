"""
Cost Analyzer - Cost-Aware Model Selection
===========================================
Tracks token usage and optimizes model selection for
60-70% cost savings while maintaining quality.

From the Blueprint:
    - Track token usage per agent/model
    - Compute cost-performance ratio for routing decisions
    - Optimize model selection for cost-aware routing
    - Generate cost reports for observability

Key Features:
    - Token and cost tracking per agent/model
    - Cost-performance ratio computation
    - Model selection optimization (60-70% savings target)
    - Integration with corpus_manager patterns
    - Report generation for observability
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelCostConfig:
    """Cost configuration for a model."""

    model_name: str
    provider: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    vram_gb: float = 0.0
    avg_latency_ms: float = 0.0
    quality_baseline: float = 0.8

    @property
    def avg_cost_per_1k_tokens(self) -> float:
        """Average cost per 1K tokens (input/output combined)."""
        return (self.cost_per_1k_input_tokens + self.cost_per_1k_output_tokens) / 2


# Default model costs (based on typical local/hosted pricing)
DEFAULT_MODEL_COSTS: Dict[str, ModelCostConfig] = {
    "deepseek-r1:7b": ModelCostConfig(
        model_name="deepseek-r1:7b",
        provider="ollama",
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0002,
        vram_gb=4.5,
        avg_latency_ms=1500,
        quality_baseline=0.85,
    ),
    "qwen2.5:7b": ModelCostConfig(
        model_name="qwen2.5:7b",
        provider="ollama",
        cost_per_1k_input_tokens=0.00008,
        cost_per_1k_output_tokens=0.00016,
        vram_gb=4.0,
        avg_latency_ms=1000,
        quality_baseline=0.80,
    ),
    "mistral:7b": ModelCostConfig(
        model_name="mistral:7b",
        provider="ollama",
        cost_per_1k_input_tokens=0.00008,
        cost_per_1k_output_tokens=0.00016,
        vram_gb=4.0,
        avg_latency_ms=900,
        quality_baseline=0.80,
    ),
    "agentflow-7b": ModelCostConfig(
        model_name="agentflow-7b",
        provider="lmstudio",
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0002,
        vram_gb=4.0,
        avg_latency_ms=1100,
        quality_baseline=0.82,
    ),
    # Higher-capability models (more expensive)
    "deepseek-r1:14b": ModelCostConfig(
        model_name="deepseek-r1:14b",
        provider="ollama",
        cost_per_1k_input_tokens=0.0003,
        cost_per_1k_output_tokens=0.0006,
        vram_gb=8.0,
        avg_latency_ms=2500,
        quality_baseline=0.90,
    ),
    "qwen2.5:14b": ModelCostConfig(
        model_name="qwen2.5:14b",
        provider="ollama",
        cost_per_1k_input_tokens=0.00025,
        cost_per_1k_output_tokens=0.0005,
        vram_gb=8.0,
        avg_latency_ms=1800,
        quality_baseline=0.88,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# COST METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class UsageRecord:
    """Record of token usage for a single execution."""

    execution_id: str
    model_name: str
    agent_name: str
    task_category: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    quality_score: float
    success: bool
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class CostMetrics:
    """Aggregated cost metrics."""

    total_executions: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    # By model
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    tokens_by_model: Dict[str, int] = field(default_factory=dict)

    # By agent
    cost_by_agent: Dict[str, float] = field(default_factory=dict)
    tokens_by_agent: Dict[str, int] = field(default_factory=dict)

    # By category
    cost_by_category: Dict[str, float] = field(default_factory=dict)
    tokens_by_category: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_weighted_cost: float = 0.0

    # Savings tracking
    baseline_cost: float = 0.0
    optimized_cost: float = 0.0
    savings_rate: float = 0.0

    # Time window
    window_start: str = ""
    window_end: str = ""

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_cost_per_execution(self) -> float:
        """Average cost per execution."""
        if self.total_executions == 0:
            return 0.0
        return self.total_cost / self.total_executions

    @property
    def cost_per_1k_tokens(self) -> float:
        """Effective cost per 1K tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_cost / (self.total_tokens / 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_executions": self.total_executions,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "avg_cost_per_execution": round(self.avg_cost_per_execution, 6),
            "cost_per_1k_tokens": round(self.cost_per_1k_tokens, 6),
            "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model.items()},
            "cost_by_agent": {k: round(v, 6) for k, v in self.cost_by_agent.items()},
            "cost_by_category": {
                k: round(v, 6) for k, v in self.cost_by_category.items()
            },
            "avg_quality_score": round(self.avg_quality_score, 4),
            "quality_weighted_cost": round(self.quality_weighted_cost, 6),
            "baseline_cost": round(self.baseline_cost, 6),
            "optimized_cost": round(self.optimized_cost, 6),
            "savings_rate": round(self.savings_rate, 4),
            "window_start": self.window_start,
            "window_end": self.window_end,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COST ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════


class CostAnalyzer:
    """
    Cost-aware model selection and tracking.

    Features:
        - Token usage tracking per agent/model
        - Cost-performance ratio computation
        - Model selection optimization
        - Cost report generation
        - 60-70% savings target tracking
    """

    # Target savings rate
    TARGET_SAVINGS_MIN = 0.60
    TARGET_SAVINGS_MAX = 0.70

    def __init__(
        self,
        model_configs: Optional[Dict[str, ModelCostConfig]] = None,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize Cost Analyzer.

        Args:
            model_configs: Custom model cost configurations
            persistence_path: Path for saving/loading state
        """
        self.model_configs = model_configs or dict(DEFAULT_MODEL_COSTS)
        self.persistence_path = persistence_path or Path(
            os.getenv("BIZRA_COST_STATE", "docs/evidence/apex/cost_state.json")
        )

        # Usage history
        self._usage_history: List[UsageRecord] = []
        self._max_history = 10000

        # Running metrics
        self._metrics = CostMetrics()

        # Cost-performance cache for routing
        self._cost_perf_cache: Dict[Tuple[str, str], float] = {}

        # Load state
        self._load_state()

        logger.info("CostAnalyzer initialized")

    def track_usage(self, record: UsageRecord) -> None:
        """
        Track token usage for an execution.

        Args:
            record: Usage record to track
        """
        self._usage_history.append(record)

        # Trim history if needed
        if len(self._usage_history) > self._max_history:
            self._usage_history = self._usage_history[-self._max_history :]

        # Update running metrics
        self._update_metrics(record)

        # Invalidate cost-perf cache
        cache_key = (record.agent_name, record.task_category)
        if cache_key in self._cost_perf_cache:
            del self._cost_perf_cache[cache_key]

    def _update_metrics(self, record: UsageRecord) -> None:
        """Update running metrics with new record."""
        self._metrics.total_executions += 1
        self._metrics.total_input_tokens += record.input_tokens
        self._metrics.total_output_tokens += record.output_tokens
        self._metrics.total_cost += record.cost

        # By model
        if record.model_name not in self._metrics.cost_by_model:
            self._metrics.cost_by_model[record.model_name] = 0.0
            self._metrics.tokens_by_model[record.model_name] = 0
        self._metrics.cost_by_model[record.model_name] += record.cost
        self._metrics.tokens_by_model[record.model_name] += (
            record.input_tokens + record.output_tokens
        )

        # By agent
        if record.agent_name not in self._metrics.cost_by_agent:
            self._metrics.cost_by_agent[record.agent_name] = 0.0
            self._metrics.tokens_by_agent[record.agent_name] = 0
        self._metrics.cost_by_agent[record.agent_name] += record.cost
        self._metrics.tokens_by_agent[record.agent_name] += (
            record.input_tokens + record.output_tokens
        )

        # By category
        if record.task_category not in self._metrics.cost_by_category:
            self._metrics.cost_by_category[record.task_category] = 0.0
            self._metrics.tokens_by_category[record.task_category] = 0
        self._metrics.cost_by_category[record.task_category] += record.cost
        self._metrics.tokens_by_category[record.task_category] += (
            record.input_tokens + record.output_tokens
        )

        # Update averages
        n = self._metrics.total_executions
        self._metrics.avg_quality_score = (
            self._metrics.avg_quality_score * (n - 1) + record.quality_score
        ) / n

    def compute_cost_performance_ratio(
        self,
        model_name: str,
        task_category: str,
        quality_weight: float = 0.6,
        latency_weight: float = 0.2,
        cost_weight: float = 0.2,
    ) -> float:
        """
        Compute cost-performance ratio for a model-category pair.

        Higher is better: balances quality, latency, and cost.

        Args:
            model_name: The model to evaluate
            task_category: The task category
            quality_weight: Weight for quality (0-1)
            latency_weight: Weight for latency (0-1, lower latency = higher score)
            cost_weight: Weight for cost (0-1, lower cost = higher score)

        Returns:
            Cost-performance ratio (0-1, higher is better)
        """
        cache_key = (model_name, task_category)
        if cache_key in self._cost_perf_cache:
            return self._cost_perf_cache[cache_key]

        # Get model config
        config = self.model_configs.get(model_name)
        if config is None:
            logger.warning(f"No cost config for model: {model_name}")
            return 0.5

        # Get historical performance for this model-category
        relevant_records = [
            r
            for r in self._usage_history
            if r.model_name == model_name and r.task_category == task_category
        ]

        if relevant_records:
            # Use historical data
            avg_quality = np.mean([r.quality_score for r in relevant_records])
            avg_latency = np.mean([r.latency_ms for r in relevant_records])
            avg_cost = np.mean([r.cost for r in relevant_records])
        else:
            # Use baseline from config
            avg_quality = config.quality_baseline
            avg_latency = config.avg_latency_ms
            avg_cost = config.avg_cost_per_1k_tokens

        # Normalize scores (0-1)
        # Quality: already 0-1
        quality_score = avg_quality

        # Latency: normalize against max expected (5000ms)
        latency_score = 1.0 - min(avg_latency / 5000, 1.0)

        # Cost: normalize against max expected (0.01 per execution)
        cost_score = 1.0 - min(avg_cost / 0.01, 1.0)

        # Weighted combination
        ratio = (
            quality_weight * quality_score
            + latency_weight * latency_score
            + cost_weight * cost_score
        )

        # Cache result
        self._cost_perf_cache[cache_key] = ratio

        return ratio

    def optimize_model_selection(
        self,
        task_category: str,
        available_models: List[str],
        min_quality: float = 0.7,
        max_latency_ms: float = 3000,
    ) -> List[Tuple[str, float]]:
        """
        Optimize model selection for cost-aware routing.

        Returns models ranked by cost-performance ratio that meet
        quality and latency constraints.

        Args:
            task_category: The task category
            available_models: List of available model names
            min_quality: Minimum acceptable quality
            max_latency_ms: Maximum acceptable latency

        Returns:
            List of (model_name, score) tuples sorted by score descending
        """
        candidates: List[Tuple[str, float]] = []

        for model_name in available_models:
            config = self.model_configs.get(model_name)
            if config is None:
                continue

            # Check constraints
            if config.quality_baseline < min_quality:
                continue
            if config.avg_latency_ms > max_latency_ms:
                continue

            # Compute cost-performance ratio
            ratio = self.compute_cost_performance_ratio(model_name, task_category)
            candidates.append((model_name, ratio))

        # Sort by ratio descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Model selection for {task_category}: "
            f"{[(m, f'{s:.3f}') for m, s in candidates[:3]]}"
        )

        return candidates

    def compute_savings(self) -> Dict[str, Any]:
        """
        Compute cost savings vs baseline.

        Baseline is computed as if always using the most expensive model.

        Returns:
            Savings metrics
        """
        if not self._usage_history:
            return {"savings_rate": 0.0, "status": "no_data"}

        # Find most expensive model
        max_cost_model = max(
            self.model_configs.values(), key=lambda c: c.avg_cost_per_1k_tokens
        )

        # Compute baseline cost (if always used expensive model)
        baseline_cost = 0.0
        for record in self._usage_history:
            total_tokens = record.input_tokens + record.output_tokens
            baseline_cost += (
                total_tokens / 1000
            ) * max_cost_model.avg_cost_per_1k_tokens

        # Actual cost
        actual_cost = self._metrics.total_cost

        # Savings rate
        if baseline_cost > 0:
            savings_rate = (baseline_cost - actual_cost) / baseline_cost
        else:
            savings_rate = 0.0

        # Update metrics
        self._metrics.baseline_cost = baseline_cost
        self._metrics.optimized_cost = actual_cost
        self._metrics.savings_rate = savings_rate

        # Check if meeting target
        target_met = self.TARGET_SAVINGS_MIN <= savings_rate <= self.TARGET_SAVINGS_MAX
        target_status = (
            "below_target"
            if savings_rate < self.TARGET_SAVINGS_MIN
            else (
                "above_target"
                if savings_rate > self.TARGET_SAVINGS_MAX
                else "on_target"
            )
        )

        return {
            "baseline_cost": baseline_cost,
            "actual_cost": actual_cost,
            "savings": baseline_cost - actual_cost,
            "savings_rate": savings_rate,
            "savings_percent": savings_rate * 100,
            "target_min": self.TARGET_SAVINGS_MIN,
            "target_max": self.TARGET_SAVINGS_MAX,
            "target_met": target_met,
            "target_status": target_status,
        }

    def generate_cost_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cost report for observability.

        Returns:
            Cost report dictionary
        """
        savings = self.compute_savings()

        # Time window
        if self._usage_history:
            self._metrics.window_start = self._usage_history[0].timestamp
            self._metrics.window_end = self._usage_history[-1].timestamp

        # Model efficiency ranking
        model_efficiency = []
        for model_name, config in self.model_configs.items():
            if model_name in self._metrics.tokens_by_model:
                tokens = self._metrics.tokens_by_model[model_name]
                cost = self._metrics.cost_by_model[model_name]
                efficiency = tokens / cost if cost > 0 else 0
                model_efficiency.append(
                    {
                        "model": model_name,
                        "tokens": tokens,
                        "cost": cost,
                        "efficiency": efficiency,  # tokens per dollar
                    }
                )

        model_efficiency.sort(key=lambda x: x["efficiency"], reverse=True)

        # Agent cost breakdown
        agent_costs = []
        for agent_name, cost in self._metrics.cost_by_agent.items():
            tokens = self._metrics.tokens_by_agent.get(agent_name, 0)
            agent_costs.append(
                {
                    "agent": agent_name,
                    "cost": cost,
                    "tokens": tokens,
                    "avg_cost_per_1k": (cost / tokens * 1000) if tokens > 0 else 0,
                }
            )

        agent_costs.sort(key=lambda x: x["cost"], reverse=True)

        # Category breakdown
        category_costs = []
        for category, cost in self._metrics.cost_by_category.items():
            tokens = self._metrics.tokens_by_category.get(category, 0)
            category_costs.append(
                {
                    "category": category,
                    "cost": cost,
                    "tokens": tokens,
                }
            )

        category_costs.sort(key=lambda x: x["cost"], reverse=True)

        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": self._metrics.to_dict(),
            "savings": savings,
            "model_efficiency": model_efficiency,
            "agent_costs": agent_costs,
            "category_costs": category_costs,
            "recommendations": self._generate_recommendations(savings),
        }

        return report

    def _generate_recommendations(self, savings: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        savings_rate = savings.get("savings_rate", 0)

        if savings_rate < self.TARGET_SAVINGS_MIN:
            recommendations.append(
                f"Savings rate ({savings_rate:.1%}) is below target "
                f"({self.TARGET_SAVINGS_MIN:.0%}). Consider using more cost-efficient models."
            )

            # Find most expensive model usage
            if self._metrics.cost_by_model:
                top_cost_model = max(
                    self._metrics.cost_by_model.items(), key=lambda x: x[1]
                )[0]
                recommendations.append(
                    f"Top cost contributor: {top_cost_model}. "
                    "Consider routing simpler tasks to lighter models."
                )

        elif savings_rate > self.TARGET_SAVINGS_MAX:
            recommendations.append(
                f"Savings rate ({savings_rate:.1%}) exceeds target "
                f"({self.TARGET_SAVINGS_MAX:.0%}). Quality may be impacted."
            )
            recommendations.append(
                "Consider using higher-capability models for complex tasks."
            )

        else:
            recommendations.append(
                f"Savings rate ({savings_rate:.1%}) is within target range. "
                "Cost optimization is performing well."
            )

        # Quality-based recommendations
        if self._metrics.avg_quality_score < 0.8:
            recommendations.append(
                f"Average quality score ({self._metrics.avg_quality_score:.2f}) is low. "
                "Consider prioritizing quality over cost for important tasks."
            )

        return recommendations

    def get_model_config(self, model_name: str) -> Optional[ModelCostConfig]:
        """Get cost configuration for a model."""
        return self.model_configs.get(model_name)

    def add_model_config(self, config: ModelCostConfig) -> None:
        """Add or update a model's cost configuration."""
        self.model_configs[config.model_name] = config
        logger.info(f"Added cost config for model: {config.model_name}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Only save recent history for efficiency
            recent_history = self._usage_history[-1000:]

            state = {
                "version": "1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": self._metrics.to_dict(),
                "history": [
                    {
                        "execution_id": r.execution_id,
                        "model_name": r.model_name,
                        "agent_name": r.agent_name,
                        "task_category": r.task_category,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "cost": r.cost,
                        "latency_ms": r.latency_ms,
                        "quality_score": r.quality_score,
                        "success": r.success,
                        "timestamp": r.timestamp,
                    }
                    for r in recent_history
                ],
            }

            self.persistence_path.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
            logger.debug(f"Saved cost state to {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to save cost state: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.persistence_path.exists():
            return

        try:
            data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            # Load history
            for h in data.get("history", []):
                record = UsageRecord(
                    execution_id=h["execution_id"],
                    model_name=h["model_name"],
                    agent_name=h["agent_name"],
                    task_category=h["task_category"],
                    input_tokens=h["input_tokens"],
                    output_tokens=h["output_tokens"],
                    cost=h["cost"],
                    latency_ms=h["latency_ms"],
                    quality_score=h["quality_score"],
                    success=h["success"],
                    timestamp=h["timestamp"],
                )
                self._usage_history.append(record)
                self._update_metrics(record)

            logger.info(f"Loaded cost state from {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to load cost state: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Test Cost Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Cost Analyzer")
    parser.add_argument("--simulate", type=int, default=0, help="Simulate N executions")
    parser.add_argument("--report", action="store_true", help="Generate cost report")
    parser.add_argument(
        "--optimize", type=str, help="Optimize model selection for category"
    )
    args = parser.parse_args()

    analyzer = CostAnalyzer()

    if args.simulate > 0:
        print(f"\nSimulating {args.simulate} executions...\n")

        models = ["deepseek-r1:7b", "qwen2.5:7b", "mistral:7b"]
        agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer"]
        categories = ["reasoning", "creative", "analysis"]

        for i in range(args.simulate):
            # Simulate cost-aware selection (prefer cheaper models)
            model_weights = [0.3, 0.4, 0.3]  # Bias toward middle-cost
            model = np.random.choice(models, p=model_weights)
            agent = np.random.choice(agents)
            category = np.random.choice(categories)

            config = analyzer.get_model_config(model)
            input_tokens = np.random.randint(100, 500)
            output_tokens = np.random.randint(100, 1000)

            if config:
                cost = (input_tokens / 1000) * config.cost_per_1k_input_tokens + (
                    output_tokens / 1000
                ) * config.cost_per_1k_output_tokens
                latency = config.avg_latency_ms * (1 + np.random.uniform(-0.2, 0.2))
            else:
                cost = 0.001
                latency = 1000

            record = UsageRecord(
                execution_id=f"exec_{i:04d}",
                model_name=model,
                agent_name=agent,
                task_category=category,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency,
                quality_score=np.random.uniform(0.7, 1.0),
                success=np.random.random() > 0.1,
            )
            analyzer.track_usage(record)

        print("Simulation complete.")

    if args.report or args.simulate > 0:
        report = analyzer.generate_cost_report()
        print("\n" + "=" * 60)
        print("  COST REPORT")
        print("=" * 60)
        print("\nSummary:")
        print(f"  Total Executions: {report['summary']['total_executions']}")
        print(f"  Total Cost: ${report['summary']['total_cost']:.6f}")
        print(f"  Total Tokens: {report['summary']['total_tokens']:,}")
        print(f"  Avg Quality: {report['summary']['avg_quality_score']:.2%}")

        print("\nSavings:")
        savings = report["savings"]
        print(f"  Baseline Cost: ${savings['baseline_cost']:.6f}")
        print(f"  Actual Cost: ${savings['actual_cost']:.6f}")
        print(f"  Savings Rate: {savings['savings_percent']:.1f}%")
        print(f"  Target Status: {savings['target_status']}")

        print("\nModel Efficiency:")
        for m in report["model_efficiency"][:3]:
            print(f"  {m['model']}: {m['efficiency']:.0f} tokens/$")

        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")

    if args.optimize:
        print(f"\nOptimizing model selection for: {args.optimize}")
        models = list(analyzer.model_configs.keys())
        selections = analyzer.optimize_model_selection(args.optimize, models)
        print("Ranked models:")
        for model, score in selections:
            print(f"  {model}: {score:.3f}")


if __name__ == "__main__":
    main()
