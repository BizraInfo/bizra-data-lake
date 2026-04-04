"""
SONA Learner - Self-Optimizing Novelty Architecture
====================================================
Continuous learning system for routing optimization with
pattern elevation to SAPE cache when repetitions > 3.

From the Blueprint:
    - Extract patterns from success/failure executions
    - Optimize routing weights based on performance
    - Integrate with Ihsan gate for quality validation
    - Target: +55% improvement in routing efficiency

Key Features:
    - Continuous learning loop with async support
    - Pattern mining from execution history
    - Weight adjustment with gradient-free optimization
    - SAPE elevation for high-repetition patterns
    - Ihsan integration for performance evaluation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

logger = logging.getLogger(__name__)

# Import constitutional thresholds
from core.constants import IHSAN_THRESHOLD

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LearningConfig:
    """Configuration for SONA learning."""

    # Learning parameters
    learning_rate: float = 0.1
    momentum: float = 0.9
    min_samples_for_update: int = 10
    update_interval_seconds: float = 60.0

    # Pattern elevation thresholds
    elevation_threshold: int = 3  # Repetitions needed for SAPE elevation
    min_success_rate_for_elevation: float = 0.7

    # Performance targets
    target_improvement: float = 0.55  # +55% improvement target
    ihsan_threshold: float = IHSAN_THRESHOLD  # From core.constants

    # Persistence
    state_path: str = "docs/evidence/apex/sona_state.json"

    @classmethod
    def from_env(cls) -> "LearningConfig":
        """Create config from environment variables."""
        return cls(
            learning_rate=float(os.getenv("SONA_LEARNING_RATE", "0.1")),
            momentum=float(os.getenv("SONA_MOMENTUM", "0.9")),
            min_samples_for_update=int(os.getenv("SONA_MIN_SAMPLES", "10")),
            update_interval_seconds=float(os.getenv("SONA_UPDATE_INTERVAL", "60.0")),
            elevation_threshold=int(os.getenv("SONA_ELEVATION_THRESHOLD", "3")),
            target_improvement=float(os.getenv("SONA_TARGET_IMPROVEMENT", "0.55")),
            ihsan_threshold=float(os.getenv("IHSAN_THRESHOLD", str(IHSAN_THRESHOLD))),
            state_path=os.getenv(
                "SONA_STATE_PATH", "docs/evidence/apex/sona_state.json"
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExecutionRecord:
    """Record of a single execution."""

    task_id: str
    task_category: str
    agent_name: str
    success: bool
    quality_score: float  # 0-1, Ihsan-aligned
    latency_ms: float
    token_count: int
    cost: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pattern_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""

    total_executions: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Quality metrics
    avg_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0

    # Efficiency metrics
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0

    # Improvement tracking
    baseline_success_rate: float = 0.5
    current_success_rate: float = 0.5
    improvement_rate: float = 0.0

    # Time window
    window_start: str = ""
    window_end: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.success_count / self.total_executions

    @property
    def ihsan_compliant(self) -> bool:
        """Check if average quality meets Ihsan threshold."""
        return self.avg_quality_score >= 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_executions": self.total_executions,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "avg_quality_score": self.avg_quality_score,
            "min_quality_score": self.min_quality_score,
            "max_quality_score": self.max_quality_score,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "baseline_success_rate": self.baseline_success_rate,
            "current_success_rate": self.current_success_rate,
            "improvement_rate": self.improvement_rate,
            "ihsan_compliant": self.ihsan_compliant,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN TRACKING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrackedPattern:
    """A pattern being tracked for potential elevation."""

    pattern_hash: str
    pattern_signature: List[str]  # Sequence of (category, agent) pairs
    occurrence_count: int = 0
    success_count: int = 0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    elevated: bool = False
    elevation_timestamp: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate for this pattern."""
        if self.occurrence_count == 0:
            return 0.0
        return self.success_count / self.occurrence_count

    def should_elevate(self, threshold: int = 3, min_success_rate: float = 0.7) -> bool:
        """Check if pattern should be elevated to SAPE."""
        return (
            not self.elevated
            and self.occurrence_count >= threshold
            and self.success_rate >= min_success_rate
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_hash": self.pattern_hash,
            "pattern_signature": self.pattern_signature,
            "occurrence_count": self.occurrence_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "avg_quality": self.avg_quality,
            "avg_latency_ms": self.avg_latency_ms,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "elevated": self.elevated,
            "elevation_timestamp": self.elevation_timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RoutingWeights:
    """Learnable routing weights."""

    # Agent weights per category
    agent_category_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Global agent preferences
    agent_global_weights: Dict[str, float] = field(default_factory=dict)

    # Feature weights for routing decisions
    feature_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "capability_affinity": 0.4,
            "historical_success": 0.3,
            "cost_efficiency": 0.2,
            "latency": 0.1,
        }
    )

    # Momentum for updates
    _velocity: Dict[str, float] = field(default_factory=dict)

    def get_weight(self, agent: str, category: str) -> float:
        """Get weight for agent-category pair."""
        if agent not in self.agent_category_weights:
            return self.agent_global_weights.get(agent, 0.5)
        return self.agent_category_weights[agent].get(category, 0.5)

    def update_weight(
        self,
        agent: str,
        category: str,
        gradient: float,
        learning_rate: float,
        momentum: float,
    ) -> None:
        """Update weight with momentum."""
        key = f"{agent}|{category}"

        # Initialize if needed
        if agent not in self.agent_category_weights:
            self.agent_category_weights[agent] = {}
        if category not in self.agent_category_weights[agent]:
            self.agent_category_weights[agent][category] = 0.5

        # Compute velocity with momentum
        prev_velocity = self._velocity.get(key, 0.0)
        velocity = momentum * prev_velocity + learning_rate * gradient
        self._velocity[key] = velocity

        # Update weight
        current = self.agent_category_weights[agent][category]
        new_weight = current + velocity

        # Clamp to [0.01, 0.99]
        new_weight = max(0.01, min(0.99, new_weight))
        self.agent_category_weights[agent][category] = new_weight

    def normalize(self) -> None:
        """Normalize weights so they sum to 1 per category."""
        for agent, categories in self.agent_category_weights.items():
            total = sum(categories.values())
            if total > 0:
                for category in categories:
                    categories[category] /= total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_category_weights": self.agent_category_weights,
            "agent_global_weights": self.agent_global_weights,
            "feature_weights": self.feature_weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingWeights":
        """Deserialize from dictionary."""
        weights = cls()
        weights.agent_category_weights = data.get("agent_category_weights", {})
        weights.agent_global_weights = data.get("agent_global_weights", {})
        weights.feature_weights = data.get("feature_weights", weights.feature_weights)
        return weights


# ═══════════════════════════════════════════════════════════════════════════════
# SONA LEARNER
# ═══════════════════════════════════════════════════════════════════════════════


class SONALearner:
    """
    Self-Optimizing Novelty Architecture for continuous routing improvement.

    Features:
        - Pattern extraction from execution history
        - Weight optimization using gradient-free methods
        - Performance evaluation with Ihsan integration
        - SAPE elevation for high-repetition patterns
        - +55% improvement target tracking
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        sape_elevation_callback: Optional[Callable[[TrackedPattern], None]] = None,
    ):
        """
        Initialize SONA learner.

        Args:
            config: Learning configuration
            sape_elevation_callback: Callback when pattern is elevated to SAPE
        """
        self.config = config or LearningConfig.from_env()
        self.sape_elevation_callback = sape_elevation_callback

        # State
        self._execution_history: List[ExecutionRecord] = []
        self._tracked_patterns: Dict[str, TrackedPattern] = {}
        self._routing_weights = RoutingWeights()
        self._metrics = PerformanceMetrics()

        # Learning loop state
        self._running = False
        self._learning_task: Optional[asyncio.Task] = None

        # Load persisted state
        self._load_state()

        logger.info("SONALearner initialized")

    async def start_learning_loop(self) -> None:
        """Start the continuous learning loop."""
        if self._running:
            logger.warning("Learning loop already running")
            return

        self._running = True
        self._learning_task = asyncio.create_task(self._learning_loop())
        logger.info("SONA learning loop started")

    async def stop_learning_loop(self) -> None:
        """Stop the continuous learning loop."""
        self._running = False
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("SONA learning loop stopped")

    async def _learning_loop(self) -> None:
        """Main learning loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.update_interval_seconds)

                if len(self._execution_history) >= self.config.min_samples_for_update:
                    # Extract patterns
                    patterns = self.extract_patterns()

                    # Optimize routing
                    self.optimize_routing()

                    # Evaluate performance
                    metrics = self.evaluate_performance()

                    # Check for pattern elevation
                    await self._check_elevation(patterns)

                    # Log progress
                    logger.info(
                        f"SONA update: success_rate={metrics.success_rate:.3f}, "
                        f"improvement={metrics.improvement_rate:.1%}, "
                        f"patterns={len(patterns)}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")

    def record_execution(self, record: ExecutionRecord) -> None:
        """
        Record an execution for learning.

        Args:
            record: The execution record to add
        """
        self._execution_history.append(record)

        # Update pattern tracking
        pattern_hash = self._compute_pattern_hash(record)
        record.pattern_hash = pattern_hash

        if pattern_hash not in self._tracked_patterns:
            self._tracked_patterns[pattern_hash] = TrackedPattern(
                pattern_hash=pattern_hash,
                pattern_signature=[f"{record.task_category}:{record.agent_name}"],
            )

        pattern = self._tracked_patterns[pattern_hash]
        pattern.occurrence_count += 1
        pattern.last_seen = record.timestamp
        if record.success:
            pattern.success_count += 1

        # Update running averages
        n = pattern.occurrence_count
        pattern.avg_quality = (pattern.avg_quality * (n - 1) + record.quality_score) / n
        pattern.avg_latency_ms = (
            pattern.avg_latency_ms * (n - 1) + record.latency_ms
        ) / n

        # Trim history if needed
        max_history = 10000
        if len(self._execution_history) > max_history:
            self._execution_history = self._execution_history[-max_history:]

    def extract_patterns(self) -> List[TrackedPattern]:
        """
        Extract patterns from execution history.

        Returns:
            List of tracked patterns sorted by occurrence count
        """
        # Return patterns sorted by occurrence
        patterns = list(self._tracked_patterns.values())
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
        return patterns

    def optimize_routing(self) -> None:
        """
        Optimize routing weights based on execution history.

        Uses a gradient-free approach: compute performance deltas
        and adjust weights proportionally.
        """
        if len(self._execution_history) < self.config.min_samples_for_update:
            return

        # Group executions by (agent, category)
        performance_by_key: Dict[Tuple[str, str], List[ExecutionRecord]] = {}
        for record in self._execution_history[-1000:]:
            key = (record.agent_name, record.task_category)
            if key not in performance_by_key:
                performance_by_key[key] = []
            performance_by_key[key].append(record)

        # Compute gradients based on performance
        for (agent, category), records in performance_by_key.items():
            if len(records) < 5:
                continue

            # Compute average success and quality
            success_rate = sum(1 for r in records if r.success) / len(records)
            avg_quality = sum(r.quality_score for r in records) / len(records)

            # Gradient: positive if performing well, negative otherwise
            target_success = 0.8
            gradient = (success_rate - target_success) * avg_quality

            # Update weight
            self._routing_weights.update_weight(
                agent,
                category,
                gradient,
                self.config.learning_rate,
                self.config.momentum,
            )

        # Normalize weights
        self._routing_weights.normalize()

        logger.debug("Routing weights optimized")

    def evaluate_performance(self) -> PerformanceMetrics:
        """
        Evaluate current performance with Ihsan integration.

        Returns:
            Aggregated performance metrics
        """
        if not self._execution_history:
            return self._metrics

        # Use recent history for evaluation
        recent = self._execution_history[-500:]

        # Compute metrics
        self._metrics.total_executions = len(recent)
        self._metrics.success_count = sum(1 for r in recent if r.success)
        self._metrics.failure_count = len(recent) - self._metrics.success_count

        quality_scores = [r.quality_score for r in recent]
        self._metrics.avg_quality_score = (
            np.mean(quality_scores) if quality_scores else 0.0
        )
        self._metrics.min_quality_score = min(quality_scores) if quality_scores else 0.0
        self._metrics.max_quality_score = max(quality_scores) if quality_scores else 0.0

        latencies = [r.latency_ms for r in recent]
        self._metrics.avg_latency_ms = np.mean(latencies) if latencies else 0.0
        self._metrics.total_tokens = sum(r.token_count for r in recent)
        self._metrics.total_cost = sum(r.cost for r in recent)

        # Compute improvement
        self._metrics.current_success_rate = self._metrics.success_rate
        if self._metrics.baseline_success_rate > 0:
            improvement = (
                self._metrics.current_success_rate - self._metrics.baseline_success_rate
            ) / self._metrics.baseline_success_rate
            self._metrics.improvement_rate = improvement
        else:
            self._metrics.improvement_rate = 0.0

        # Set time window
        if recent:
            self._metrics.window_start = recent[0].timestamp
            self._metrics.window_end = recent[-1].timestamp

        return self._metrics

    async def _check_elevation(self, patterns: List[TrackedPattern]) -> None:
        """Check patterns for SAPE elevation."""
        for pattern in patterns:
            if pattern.should_elevate(
                self.config.elevation_threshold,
                self.config.min_success_rate_for_elevation,
            ):
                # Mark as elevated
                pattern.elevated = True
                pattern.elevation_timestamp = datetime.now(timezone.utc).isoformat()

                logger.info(
                    f"Elevating pattern {pattern.pattern_hash[:8]} to SAPE "
                    f"(occurrences={pattern.occurrence_count}, "
                    f"success_rate={pattern.success_rate:.2%})"
                )

                # Call elevation callback if provided
                if self.sape_elevation_callback:
                    try:
                        self.sape_elevation_callback(pattern)
                    except Exception as e:
                        logger.error(f"SAPE elevation callback failed: {e}")

    def _compute_pattern_hash(self, record: ExecutionRecord) -> str:
        """Compute hash for pattern identification."""
        signature = f"{record.task_category}:{record.agent_name}"
        return hashlib.sha256(signature.encode()).hexdigest()[:16]

    def get_routing_recommendation(
        self,
        category: str,
        agents: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Get routing recommendations based on learned weights.

        Args:
            category: Task category
            agents: Available agents

        Returns:
            List of (agent, score) tuples sorted by score descending
        """
        recommendations = []
        for agent in agents:
            weight = self._routing_weights.get_weight(agent, category)
            recommendations.append((agent, weight))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def get_improvement_progress(self) -> Dict[str, Any]:
        """Get progress toward +55% improvement target."""
        target = self.config.target_improvement
        current = self._metrics.improvement_rate

        return {
            "target_improvement": target,
            "current_improvement": current,
            "progress_percent": (current / target * 100) if target > 0 else 0,
            "target_met": current >= target,
            "samples_collected": len(self._execution_history),
            "patterns_tracked": len(self._tracked_patterns),
            "patterns_elevated": sum(
                1 for p in self._tracked_patterns.values() if p.elevated
            ),
        }

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            state_path = Path(self.config.state_path)
            state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "version": "1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "routing_weights": self._routing_weights.to_dict(),
                "metrics": self._metrics.to_dict(),
                "patterns": {h: p.to_dict() for h, p in self._tracked_patterns.items()},
                "history_count": len(self._execution_history),
            }

            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            logger.debug(f"Saved SONA state to {state_path}")
        except Exception as e:
            logger.warning(f"Failed to save SONA state: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        try:
            state_path = Path(self.config.state_path)
            if not state_path.exists():
                return

            data = json.loads(state_path.read_text(encoding="utf-8"))

            # Load routing weights
            if "routing_weights" in data:
                self._routing_weights = RoutingWeights.from_dict(
                    data["routing_weights"]
                )

            # Load patterns
            if "patterns" in data:
                for h, p_data in data["patterns"].items():
                    self._tracked_patterns[h] = TrackedPattern(
                        pattern_hash=p_data["pattern_hash"],
                        pattern_signature=p_data["pattern_signature"],
                        occurrence_count=p_data["occurrence_count"],
                        success_count=p_data["success_count"],
                        avg_quality=p_data["avg_quality"],
                        avg_latency_ms=p_data["avg_latency_ms"],
                        first_seen=p_data["first_seen"],
                        last_seen=p_data["last_seen"],
                        elevated=p_data["elevated"],
                        elevation_timestamp=p_data.get("elevation_timestamp"),
                    )

            # Load metrics baseline
            if "metrics" in data:
                self._metrics.baseline_success_rate = data["metrics"].get(
                    "baseline_success_rate", 0.5
                )

            logger.info(f"Loaded SONA state from {state_path}")
        except Exception as e:
            logger.warning(f"Failed to load SONA state: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Test SONA Learner."""
    import argparse

    parser = argparse.ArgumentParser(description="SONA Learner")
    parser.add_argument("--simulate", type=int, default=0, help="Simulate N executions")
    parser.add_argument(
        "--progress", action="store_true", help="Show improvement progress"
    )
    args = parser.parse_args()

    learner = SONALearner()

    if args.simulate > 0:
        print(f"\nSimulating {args.simulate} executions...\n")

        agents = [
            "MasterReasoner",
            "CreativeSynthesizer",
            "DataAnalyzer",
            "ExecutionPlanner",
        ]
        categories = ["reasoning", "creative", "analysis", "planning"]

        for i in range(args.simulate):
            # Generate random execution
            agent = np.random.choice(agents)
            category = np.random.choice(categories)
            success = np.random.random() > 0.3
            quality = (
                np.random.uniform(0.7, 1.0) if success else np.random.uniform(0.3, 0.7)
            )

            record = ExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category=category,
                agent_name=agent,
                success=success,
                quality_score=quality,
                latency_ms=np.random.uniform(500, 2000),
                token_count=np.random.randint(100, 1000),
                cost=np.random.uniform(0.001, 0.01),
            )
            learner.record_execution(record)

        # Run optimization
        patterns = learner.extract_patterns()
        learner.optimize_routing()
        metrics = learner.evaluate_performance()

        print(f"Patterns tracked: {len(patterns)}")
        print(f"Metrics: {json.dumps(metrics.to_dict(), indent=2)}")

    if args.progress:
        progress = learner.get_improvement_progress()
        print("\nImprovement Progress:")
        print(f"  Target: +{progress['target_improvement']:.0%}")
        print(f"  Current: +{progress['current_improvement']:.1%}")
        print(f"  Progress: {progress['progress_percent']:.1f}%")
        print(f"  Target Met: {progress['target_met']}")
        print(f"  Samples: {progress['samples_collected']}")
        print(f"  Patterns: {progress['patterns_tracked']}")
        print(f"  Elevated: {progress['patterns_elevated']}")


if __name__ == "__main__":
    asyncio.run(main())
