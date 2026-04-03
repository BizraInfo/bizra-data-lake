"""
Thompson Sampling Router - Bayesian Agent Selection
====================================================
Implements Thompson Sampling with Beta distribution priors for
optimal explore-exploit balance in multi-agent routing.

From the Blueprint:
    - Use Beta(alpha, beta) priors for each agent-task pair
    - Sample from posterior to select agent (exploration)
    - Update posteriors with Bayesian learning (exploitation)
    - Track exploration rate for monitoring

Key Features:
    - CapabilityMatrix: Task categorization and agent capabilities
    - Posterior sampling with numpy for randomness
    - Serialization to/from JSON for persistence
    - Integration with Ihsan gate for quality validation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

import math
import random

logger = logging.getLogger(__name__)


def _stdlib_beta_sample(alpha: float, beta: float, rng: Optional[random.Random] = None) -> float:
    """Fallback Beta distribution sampling using stdlib when numpy unavailable."""
    if rng is None:
        rng = random.Random()
    # Use gamma distribution relationship: Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)
    return x / (x + y) if (x + y) > 0 else 0.5


def _stdlib_log(x: float) -> float:
    """Fallback log function using stdlib."""
    return math.log(x) if x > 0 else float('-inf')


# ═══════════════════════════════════════════════════════════════════════════════
# TASK CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

class TaskCategory(str, Enum):
    """Categories for task classification."""
    REASONING = "reasoning"           # Complex analysis, decision synthesis
    MEMORY = "memory"                 # Knowledge organization, recall
    CREATIVE = "creative"             # Writing, ideation, creative problem-solving
    ANALYSIS = "analysis"             # Data analysis, pattern recognition
    COMMUNICATION = "communication"   # External communications, messaging
    PLANNING = "planning"             # Task planning, scheduling, workflows
    ETHICS = "ethics"                 # Safety, bias detection, Ihsan compliance
    VALIDATION = "validation"         # Verification, proof, attestation
    GENERAL = "general"               # Unclassified tasks


class AgentCapability(str, Enum):
    """Agent capability dimensions."""
    STRATEGIC_THINKING = "strategic_thinking"
    KNOWLEDGE_MANAGEMENT = "knowledge_management"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    DATA_ANALYSIS = "data_analysis"
    COMMUNICATION = "communication"
    WORKFLOW_PLANNING = "workflow_planning"
    ETHICAL_JUDGMENT = "ethical_judgment"
    VERIFICATION = "verification"


# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentProfile:
    """Profile of an agent's capabilities."""
    agent_name: str
    capabilities: Dict[AgentCapability, float]  # Capability -> strength (0-1)
    task_affinities: Dict[TaskCategory, float]  # Task category -> affinity (0-1)
    cost_per_token: float = 0.0  # Cost in units per token
    avg_latency_ms: float = 0.0  # Average response latency


class CapabilityMatrix:
    """
    Matrix mapping agents to capabilities and task categories.

    Used by ThompsonSamplingRouter to determine which agents are
    suitable for a given task category.
    """

    # Default agent profiles (PAT agents)
    DEFAULT_PROFILES: Dict[str, AgentProfile] = {
        "MasterReasoner": AgentProfile(
            agent_name="MasterReasoner",
            capabilities={
                AgentCapability.STRATEGIC_THINKING: 0.95,
                AgentCapability.DATA_ANALYSIS: 0.80,
                AgentCapability.ETHICAL_JUDGMENT: 0.75,
            },
            task_affinities={
                TaskCategory.REASONING: 0.95,
                TaskCategory.ANALYSIS: 0.85,
                TaskCategory.PLANNING: 0.80,
                TaskCategory.ETHICS: 0.70,
            },
            cost_per_token=0.002,
            avg_latency_ms=1500,
        ),
        "MemoryArchitect": AgentProfile(
            agent_name="MemoryArchitect",
            capabilities={
                AgentCapability.KNOWLEDGE_MANAGEMENT: 0.95,
                AgentCapability.DATA_ANALYSIS: 0.70,
            },
            task_affinities={
                TaskCategory.MEMORY: 0.95,
                TaskCategory.ANALYSIS: 0.75,
                TaskCategory.GENERAL: 0.60,
            },
            cost_per_token=0.001,
            avg_latency_ms=800,
        ),
        "CreativeSynthesizer": AgentProfile(
            agent_name="CreativeSynthesizer",
            capabilities={
                AgentCapability.CREATIVE_SYNTHESIS: 0.95,
                AgentCapability.COMMUNICATION: 0.80,
            },
            task_affinities={
                TaskCategory.CREATIVE: 0.95,
                TaskCategory.COMMUNICATION: 0.85,
                TaskCategory.GENERAL: 0.65,
            },
            cost_per_token=0.001,
            avg_latency_ms=1000,
        ),
        "DataAnalyzer": AgentProfile(
            agent_name="DataAnalyzer",
            capabilities={
                AgentCapability.DATA_ANALYSIS: 0.95,
                AgentCapability.STRATEGIC_THINKING: 0.70,
            },
            task_affinities={
                TaskCategory.ANALYSIS: 0.95,
                TaskCategory.REASONING: 0.75,
                TaskCategory.GENERAL: 0.60,
            },
            cost_per_token=0.001,
            avg_latency_ms=900,
        ),
        "Communicator": AgentProfile(
            agent_name="Communicator",
            capabilities={
                AgentCapability.COMMUNICATION: 0.95,
                AgentCapability.CREATIVE_SYNTHESIS: 0.70,
            },
            task_affinities={
                TaskCategory.COMMUNICATION: 0.95,
                TaskCategory.CREATIVE: 0.75,
                TaskCategory.GENERAL: 0.65,
            },
            cost_per_token=0.001,
            avg_latency_ms=700,
        ),
        "ExecutionPlanner": AgentProfile(
            agent_name="ExecutionPlanner",
            capabilities={
                AgentCapability.WORKFLOW_PLANNING: 0.95,
                AgentCapability.STRATEGIC_THINKING: 0.75,
            },
            task_affinities={
                TaskCategory.PLANNING: 0.95,
                TaskCategory.REASONING: 0.80,
                TaskCategory.GENERAL: 0.65,
            },
            cost_per_token=0.001,
            avg_latency_ms=1100,
        ),
        "EthicsGuardian": AgentProfile(
            agent_name="EthicsGuardian",
            capabilities={
                AgentCapability.ETHICAL_JUDGMENT: 0.95,
                AgentCapability.VERIFICATION: 0.85,
            },
            task_affinities={
                TaskCategory.ETHICS: 0.95,
                TaskCategory.VALIDATION: 0.90,
                TaskCategory.GENERAL: 0.50,
            },
            cost_per_token=0.001,
            avg_latency_ms=600,
        ),
    }

    def __init__(self, profiles: Optional[Dict[str, AgentProfile]] = None):
        """Initialize with agent profiles."""
        self.profiles = profiles or dict(self.DEFAULT_PROFILES)

    def get_candidates(self, category: TaskCategory, min_affinity: float = 0.5) -> List[str]:
        """
        Get agents suitable for a task category.

        Args:
            category: The task category
            min_affinity: Minimum affinity threshold (0-1)

        Returns:
            List of agent names sorted by affinity (descending)
        """
        candidates = []
        for name, profile in self.profiles.items():
            affinity = profile.task_affinities.get(category, 0.0)
            if affinity >= min_affinity:
                candidates.append((name, affinity))

        # Sort by affinity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates]

    def get_profile(self, agent_name: str) -> Optional[AgentProfile]:
        """Get profile for an agent."""
        return self.profiles.get(agent_name)

    def add_profile(self, profile: AgentProfile) -> None:
        """Add or update an agent profile."""
        self.profiles[profile.agent_name] = profile

    def classify_task(self, task_text: str) -> TaskCategory:
        """
        Classify a task into a category based on text analysis.

        Uses keyword matching for efficiency. More sophisticated
        classification can use embeddings or LLM.
        """
        text_lower = task_text.lower()

        # Keyword mapping to categories
        keywords = {
            TaskCategory.REASONING: [
                "analyze", "reason", "think", "decide", "evaluate",
                "compare", "synthesize", "strategic", "complex"
            ],
            TaskCategory.MEMORY: [
                "remember", "recall", "store", "retrieve", "context",
                "history", "previous", "knowledge", "memory"
            ],
            TaskCategory.CREATIVE: [
                "create", "write", "generate", "imagine", "design",
                "brainstorm", "creative", "story", "idea"
            ],
            TaskCategory.ANALYSIS: [
                "data", "pattern", "statistics", "metrics", "chart",
                "graph", "trend", "insight", "numbers"
            ],
            TaskCategory.COMMUNICATION: [
                "email", "message", "present", "communicate", "explain",
                "summarize", "report", "notify", "announce"
            ],
            TaskCategory.PLANNING: [
                "plan", "schedule", "organize", "workflow", "task",
                "deadline", "milestone", "roadmap", "priority"
            ],
            TaskCategory.ETHICS: [
                "ethical", "moral", "safe", "bias", "fair", "harm",
                "risk", "compliance", "ihsan", "guardian"
            ],
            TaskCategory.VALIDATION: [
                "verify", "validate", "check", "test", "prove",
                "attest", "confirm", "audit", "certify"
            ],
        }

        # Score each category
        scores: Dict[TaskCategory, int] = {cat: 0 for cat in TaskCategory}
        for category, kw_list in keywords.items():
            for kw in kw_list:
                if kw in text_lower:
                    scores[category] += 1

        # Return highest scoring category, or GENERAL if no matches
        max_score = max(scores.values())
        if max_score == 0:
            return TaskCategory.GENERAL

        for category, score in scores.items():
            if score == max_score:
                return category

        return TaskCategory.GENERAL


# ═══════════════════════════════════════════════════════════════════════════════
# BETA DISTRIBUTION PRIOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BetaPrior:
    """
    Beta distribution prior for Thompson Sampling.

    Beta(alpha, beta) represents our belief about success probability.
    - alpha: pseudo-count of successes
    - beta: pseudo-count of failures
    - Mean: alpha / (alpha + beta)
    - Variance decreases as (alpha + beta) increases
    """
    alpha: float = 1.0  # Successes + 1 (uniform prior)
    beta: float = 1.0   # Failures + 1 (uniform prior)

    @property
    def mean(self) -> float:
        """Expected success probability."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def total_samples(self) -> int:
        """Total number of observations."""
        return int(self.alpha + self.beta - 2)  # Subtract initial priors

    def sample(self, rng: Optional[Any] = None) -> float:
        """Sample from the posterior distribution."""
        if NUMPY_AVAILABLE:
            if rng is None:
                rng = np.random.default_rng()
            return rng.beta(self.alpha, self.beta)
        else:
            # Fallback to stdlib implementation
            return _stdlib_beta_sample(self.alpha, self.beta)

    def update(self, success: bool) -> None:
        """Update posterior with new observation."""
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BetaPrior":
        """Deserialize from dictionary."""
        return cls(alpha=data.get("alpha", 1.0), beta=data.get("beta", 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# THOMPSON SAMPLING ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SelectionResult:
    """Result of agent selection."""
    agent_name: str
    task_category: TaskCategory
    sampled_value: float
    exploration_rate: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ThompsonSamplingRouter:
    """
    Thompson Sampling-based agent router.

    Maintains Beta distribution priors for each (agent, task_category) pair.
    Uses posterior sampling for exploration-exploitation balance.

    Features:
        - Bayesian learning from execution outcomes
        - Automatic exploration with uncertainty
        - Serialization for persistence
        - Integration with CapabilityMatrix
    """

    def __init__(
        self,
        capability_matrix: Optional[CapabilityMatrix] = None,
        persistence_path: Optional[Path] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Thompson Sampling router.

        Args:
            capability_matrix: Matrix of agent capabilities
            persistence_path: Path for saving/loading state
            seed: Random seed for reproducibility
        """
        self.capability_matrix = capability_matrix or CapabilityMatrix()
        self.persistence_path = persistence_path or Path(
            os.getenv("BIZRA_APEX_STATE", "docs/evidence/apex/thompson_state.json")
        )

        # Random number generator
        if NUMPY_AVAILABLE:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = random.Random(seed)
            logger.warning("numpy not available - using stdlib random (reduced precision)")

        # Priors: (agent_name, task_category) -> BetaPrior
        self._priors: Dict[Tuple[str, TaskCategory], BetaPrior] = {}

        # Selection history for exploration rate calculation
        self._selection_history: List[SelectionResult] = []
        self._max_history = 1000

        # Try to load existing state
        self._load_state()

        logger.info("ThompsonSamplingRouter initialized")

    def _get_prior(self, agent_name: str, category: TaskCategory) -> BetaPrior:
        """Get or create prior for agent-category pair."""
        key = (agent_name, category)
        if key not in self._priors:
            # Initialize with capability-informed prior
            profile = self.capability_matrix.get_profile(agent_name)
            if profile:
                affinity = profile.task_affinities.get(category, 0.5)
                # Scale alpha/beta based on affinity (weak prior)
                alpha = 1.0 + affinity * 2.0
                beta = 1.0 + (1.0 - affinity) * 2.0
                self._priors[key] = BetaPrior(alpha=alpha, beta=beta)
            else:
                # Uniform prior for unknown agents
                self._priors[key] = BetaPrior()
        return self._priors[key]

    def select_agent(
        self,
        task_text: str,
        category: Optional[TaskCategory] = None,
        candidates: Optional[List[str]] = None,
    ) -> SelectionResult:
        """
        Select an agent for the task using Thompson Sampling.

        Args:
            task_text: Description of the task
            category: Task category (auto-classified if None)
            candidates: Optional list of candidate agents (auto-selected if None)

        Returns:
            SelectionResult with chosen agent and metadata
        """
        # Classify task if not provided
        if category is None:
            category = self.capability_matrix.classify_task(task_text)

        # Get candidates if not provided
        if candidates is None:
            candidates = self.capability_matrix.get_candidates(category)

        if not candidates:
            # Fallback to all agents
            candidates = list(self.capability_matrix.profiles.keys())

        # Sample from each candidate's posterior
        samples: List[Tuple[str, float]] = []
        for agent_name in candidates:
            prior = self._get_prior(agent_name, category)
            sampled = prior.sample(self.rng)
            samples.append((agent_name, sampled))

        # Select agent with highest sampled value
        samples.sort(key=lambda x: x[1], reverse=True)
        selected_agent, sampled_value = samples[0]

        # Calculate exploration rate
        exploration_rate = self.get_exploration_rate(category)

        # Record selection
        result = SelectionResult(
            agent_name=selected_agent,
            task_category=category,
            sampled_value=sampled_value,
            exploration_rate=exploration_rate,
        )
        self._selection_history.append(result)

        # Trim history if needed
        if len(self._selection_history) > self._max_history:
            self._selection_history = self._selection_history[-self._max_history:]

        logger.debug(
            f"Selected {selected_agent} for {category.value} "
            f"(sampled={sampled_value:.3f}, exploration={exploration_rate:.3f})"
        )

        return result

    def update_posterior(
        self,
        agent_name: str,
        category: TaskCategory,
        success: bool,
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Update posterior with execution outcome.

        Args:
            agent_name: The agent that executed the task
            category: The task category
            success: Whether execution was successful
            quality_score: Optional quality score (0-1) for weighted updates
        """
        prior = self._get_prior(agent_name, category)

        if quality_score is not None:
            # Weighted update based on quality score
            # High quality -> more alpha, low quality -> more beta
            prior.alpha += quality_score
            prior.beta += (1.0 - quality_score)
        else:
            # Binary update
            prior.update(success)

        logger.debug(
            f"Updated posterior for {agent_name}/{category.value}: "
            f"alpha={prior.alpha:.2f}, beta={prior.beta:.2f}, mean={prior.mean:.3f}"
        )

        # Persist state periodically
        if self._selection_history and len(self._selection_history) % 10 == 0:
            self._save_state()

    def get_exploration_rate(self, category: Optional[TaskCategory] = None) -> float:
        """
        Calculate current exploration rate.

        Exploration rate is estimated as the variance in selection patterns.
        High variance = high exploration, low variance = exploitation.

        Args:
            category: Optional category to filter by

        Returns:
            Exploration rate (0-1)
        """
        if not self._selection_history:
            return 1.0  # Maximum exploration when no history

        # Get recent selections
        recent = self._selection_history[-100:]
        if category:
            recent = [r for r in recent if r.task_category == category]

        if len(recent) < 5:
            return 0.8  # High exploration with few samples

        # Calculate entropy of agent selection
        agent_counts: Dict[str, int] = {}
        for result in recent:
            agent_counts[result.agent_name] = agent_counts.get(result.agent_name, 0) + 1

        total = sum(agent_counts.values())
        probabilities = [count / total for count in agent_counts.values()]

        # Shannon entropy normalized by max entropy
        log_fn = np.log if NUMPY_AVAILABLE else _stdlib_log
        entropy = -sum(p * log_fn(p + 1e-10) for p in probabilities)
        max_entropy = log_fn(len(self.capability_matrix.profiles))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics for an agent across all categories."""
        stats: Dict[str, Any] = {"agent_name": agent_name, "categories": {}}

        for category in TaskCategory:
            prior = self._get_prior(agent_name, category)
            stats["categories"][category.value] = {
                "mean": prior.mean,
                "variance": prior.variance,
                "total_samples": prior.total_samples,
                "alpha": prior.alpha,
                "beta": prior.beta,
            }

        return stats

    def get_category_rankings(self, category: TaskCategory) -> List[Dict[str, Any]]:
        """Get ranked agents for a category."""
        rankings = []

        for agent_name in self.capability_matrix.profiles.keys():
            prior = self._get_prior(agent_name, category)
            rankings.append({
                "agent_name": agent_name,
                "mean": prior.mean,
                "variance": prior.variance,
                "total_samples": prior.total_samples,
            })

        # Sort by mean descending
        rankings.sort(key=lambda x: x["mean"], reverse=True)
        return rankings

    def to_json(self) -> str:
        """Serialize state to JSON."""
        state = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priors": {
                f"{agent}|{cat.value}": prior.to_dict()
                for (agent, cat), prior in self._priors.items()
            },
            "selection_count": len(self._selection_history),
        }
        return json.dumps(state, indent=2)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        capability_matrix: Optional[CapabilityMatrix] = None,
        seed: Optional[int] = None,
    ) -> "ThompsonSamplingRouter":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        router = cls(capability_matrix=capability_matrix, seed=seed)

        # Load priors
        for key_str, prior_data in data.get("priors", {}).items():
            parts = key_str.split("|")
            if len(parts) == 2:
                agent_name, cat_str = parts
                try:
                    category = TaskCategory(cat_str)
                    router._priors[(agent_name, category)] = BetaPrior.from_dict(prior_data)
                except ValueError:
                    logger.warning(f"Unknown category in saved state: {cat_str}")

        return router

    def _save_state(self) -> None:
        """Save state to persistence path."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self.persistence_path.write_text(self.to_json(), encoding="utf-8")
            logger.debug(f"Saved Thompson router state to {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to save Thompson router state: {e}")

    def _load_state(self) -> None:
        """Load state from persistence path."""
        if not self.persistence_path.exists():
            return

        try:
            json_str = self.persistence_path.read_text(encoding="utf-8")
            data = json.loads(json_str)

            # Load priors
            for key_str, prior_data in data.get("priors", {}).items():
                parts = key_str.split("|")
                if len(parts) == 2:
                    agent_name, cat_str = parts
                    try:
                        category = TaskCategory(cat_str)
                        self._priors[(agent_name, category)] = BetaPrior.from_dict(prior_data)
                    except ValueError:
                        pass

            logger.info(f"Loaded Thompson router state from {self.persistence_path}")
        except Exception as e:
            logger.warning(f"Failed to load Thompson router state: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test Thompson Sampling Router."""
    import argparse

    parser = argparse.ArgumentParser(description="Thompson Sampling Router")
    parser.add_argument("--task", type=str, help="Task to route")
    parser.add_argument("--simulate", type=int, default=0, help="Simulate N selections")
    parser.add_argument("--stats", action="store_true", help="Show agent stats")
    args = parser.parse_args()

    router = ThompsonSamplingRouter()

    if args.task:
        result = router.select_agent(args.task)
        print(f"\nTask: {args.task}")
        print(f"Category: {result.task_category.value}")
        print(f"Selected: {result.agent_name}")
        print(f"Sampled Value: {result.sampled_value:.4f}")
        print(f"Exploration Rate: {result.exploration_rate:.4f}")

    elif args.simulate > 0:
        tasks = [
            "Analyze the quarterly financial data",
            "Write a creative marketing copy",
            "Plan the project timeline",
            "Check for ethical compliance",
            "Remember our previous conversation",
            "Explain the technical architecture",
        ]

        print(f"\nSimulating {args.simulate} selections...\n")

        for i in range(args.simulate):
            task = tasks[i % len(tasks)]
            result = router.select_agent(task)

            # Simulate outcome (random with bias toward capability)
            profile = router.capability_matrix.get_profile(result.agent_name)
            rand_val = np.random.random() if NUMPY_AVAILABLE else random.random()
            if profile:
                affinity = profile.task_affinities.get(result.task_category, 0.5)
                success = rand_val < (0.5 + affinity * 0.4)
            else:
                success = rand_val < 0.5

            router.update_posterior(
                result.agent_name,
                result.task_category,
                success,
            )

        print("Category Rankings after simulation:")
        for category in [TaskCategory.REASONING, TaskCategory.CREATIVE, TaskCategory.PLANNING]:
            print(f"\n  {category.value}:")
            rankings = router.get_category_rankings(category)
            for r in rankings[:3]:
                print(f"    {r['agent_name']}: mean={r['mean']:.3f}, samples={r['total_samples']}")

    elif args.stats:
        print("\nAgent Statistics:")
        for agent_name in router.capability_matrix.profiles.keys():
            stats = router.get_agent_stats(agent_name)
            print(f"\n  {agent_name}:")
            for cat, cat_stats in stats["categories"].items():
                if cat_stats["total_samples"] > 0:
                    print(f"    {cat}: mean={cat_stats['mean']:.3f}, samples={cat_stats['total_samples']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
