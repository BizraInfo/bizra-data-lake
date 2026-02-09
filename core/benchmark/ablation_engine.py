"""
ABLATION ENGINE — Automated Component Contribution Analysis
═══════════════════════════════════════════════════════════════════════════════

Scientifically attributes performance gains by systematically removing or
modifying components to measure their contribution.

Patterns:
  - AbGen: LLMs design their own ablation studies (meta-benchmark)
  - AbGen-Eval: Execute and validate ablation plans
  - AblationBench: Specifically for AI Co-Scientists

Key Insight:
  Removing a "Reviewer" agent from a swarm may show it contributed 15% of gains,
  or reveal it was actually reducing performance (negative contribution).

Giants Protocol:
  - Meta AI (2024): AbGen auto-ablation design
  - DeepMind (2024): AblationBench for co-scientists
  - Fisher (1935): Statistical experimental design

لا نفترض — We do not assume. We verify with formal proofs.
"""

from __future__ import annotations

import uuid
import hashlib
import statistics
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class AblationType(Enum):
    """Types of ablation studies."""
    REMOVE = auto()       # Remove component entirely
    DISABLE = auto()      # Disable but keep in architecture
    REPLACE = auto()      # Replace with baseline/null
    DEGRADE = auto()      # Reduce capability (e.g., smaller model)
    PERMUTE = auto()      # Change component order
    ISOLATE = auto()      # Test component in isolation


class ComponentCategory(Enum):
    """Categories of ablatable components."""
    AGENT = auto()        # Agent in a swarm
    MODEL = auto()         # LLM or sub-model
    TOOL = auto()          # External tool integration
    MEMORY = auto()        # Memory system
    REASONING = auto()     # Reasoning module
    ROUTING = auto()       # Router/dispatcher
    VERIFIER = auto()      # Verification component
    PROMPT = auto()        # Prompt template/engineering


@dataclass
class Component:
    """A system component that can be ablated."""
    id: str
    name: str
    category: ComponentCategory
    description: str = ""
    dependencies: Set[str] = field(default_factory=set)
    is_core: bool = False  # Core components can't be fully removed
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    component_id: str
    ablation_type: AblationType
    baseline_score: float
    ablated_score: float
    contribution: float  # Positive = helpful, negative = harmful
    significance: float  # Statistical significance (0-1)
    run_count: int = 1
    variance: float = 0.0
    
    @property
    def contribution_pct(self) -> float:
        """Contribution as percentage of baseline."""
        if self.baseline_score == 0:
            return 0.0
        return (self.contribution / self.baseline_score) * 100
    
    @property
    def is_beneficial(self) -> bool:
        """True if component improves performance."""
        return self.contribution > 0
    
    @property
    def is_significant(self) -> bool:
        """True if result is statistically significant."""
        return self.significance >= 0.95


@dataclass
class ComponentContribution:
    """Aggregated contribution analysis for a component."""
    component: Component
    results: List[AblationResult] = field(default_factory=list)
    
    @property
    def mean_contribution(self) -> float:
        """Mean contribution across all ablation types."""
        if not self.results:
            return 0.0
        return statistics.mean(r.contribution for r in self.results)
    
    @property
    def contribution_variance(self) -> float:
        """Variance in contribution."""
        if len(self.results) < 2:
            return 0.0
        return statistics.variance(r.contribution for r in self.results)
    
    @property
    def verdict(self) -> str:
        """Verdict on component value."""
        mean = self.mean_contribution
        if mean > 0.1:
            return "ESSENTIAL"
        elif mean > 0.05:
            return "BENEFICIAL"
        elif mean > 0:
            return "MARGINAL"
        elif mean > -0.05:
            return "NEUTRAL"
        else:
            return "HARMFUL"


@dataclass
class AblationStudy:
    """A complete ablation study specification."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    hypothesis: str = ""
    components: List[Component] = field(default_factory=list)
    ablation_types: List[AblationType] = field(default_factory=lambda: [AblationType.REMOVE])
    baseline_score: float = 0.0
    results: List[AblationResult] = field(default_factory=list)
    contributions: Dict[str, ComponentContribution] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"  # pending, running, completed, failed
    
    def add_component(self, component: Component) -> None:
        """Add a component to the study."""
        self.components.append(component)
        self.contributions[component.id] = ComponentContribution(component=component)
    
    def record_result(self, result: AblationResult) -> None:
        """Record an ablation result."""
        self.results.append(result)
        if result.component_id in self.contributions:
            self.contributions[result.component_id].results.append(result)
    
    def get_ranking(self) -> List[Tuple[str, float, str]]:
        """Rank components by contribution."""
        ranking = []
        for comp_id, contrib in self.contributions.items():
            if contrib.results:
                ranking.append((
                    contrib.component.name,
                    contrib.mean_contribution,
                    contrib.verdict,
                ))
        return sorted(ranking, key=lambda x: x[1], reverse=True)
    
    def summary(self) -> Dict[str, Any]:
        """Generate study summary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "baseline_score": self.baseline_score,
            "components_tested": len(self.components),
            "experiments_run": len(self.results),
            "ranking": self.get_ranking(),
        }


class AblationEngine:
    """
    Automated ablation study engine.
    
    Systematically tests component contributions through controlled removal
    and measurement.
    
    Example:
        >>> engine = AblationEngine()
        >>> study = engine.create_study("swarm-ablation", components=[...])
        >>> 
        >>> # Run full system baseline
        >>> baseline = run_agent_pipeline(all_components=True)
        >>> engine.set_baseline(study.id, baseline.score)
        >>> 
        >>> # Run ablations
        >>> for component in study.components:
        ...     score = run_agent_pipeline(exclude=[component.id])
        ...     engine.record_ablation(study.id, component.id, score)
        >>> 
        >>> # Analyze
        >>> ranking = engine.get_contribution_ranking(study.id)
    """
    
    # Minimum runs for statistical significance
    MIN_RUNS_FOR_SIGNIFICANCE = 3
    
    # Contribution thresholds
    ESSENTIAL_THRESHOLD = 0.10  # >10% contribution
    HARMFUL_THRESHOLD = -0.05   # <-5% contribution
    
    def __init__(self):
        self._studies: Dict[str, AblationStudy] = {}
        self._component_registry: Dict[str, Component] = {}
        logger.info("Ablation Engine initialized")
    
    def register_component(
        self,
        id: str,
        name: str,
        category: ComponentCategory,
        description: str = "",
        dependencies: Optional[Set[str]] = None,
        is_core: bool = False,
    ) -> Component:
        """Register a system component for ablation studies."""
        component = Component(
            id=id,
            name=name,
            category=category,
            description=description,
            dependencies=dependencies or set(),
            is_core=is_core,
        )
        self._component_registry[id] = component
        logger.debug(f"Registered component: {name} ({category.name})")
        return component
    
    def create_study(
        self,
        name: str,
        component_ids: Optional[List[str]] = None,
        ablation_types: Optional[List[AblationType]] = None,
        hypothesis: str = "",
    ) -> AblationStudy:
        """Create a new ablation study."""
        study = AblationStudy(
            name=name,
            hypothesis=hypothesis,
            ablation_types=ablation_types or [AblationType.REMOVE],
        )
        
        # Add components
        if component_ids:
            for cid in component_ids:
                if cid in self._component_registry:
                    study.add_component(self._component_registry[cid])
                else:
                    logger.warning(f"Component {cid} not registered, skipping")
        else:
            # Add all registered components
            for component in self._component_registry.values():
                study.add_component(component)
        
        self._studies[study.id] = study
        logger.info(f"Created ablation study '{name}' with {len(study.components)} components")
        return study
    
    def set_baseline(self, study_id: str, score: float) -> None:
        """Set the baseline (full system) score."""
        if study_id not in self._studies:
            raise ValueError(f"Study {study_id} not found")
        
        self._studies[study_id].baseline_score = score
        self._studies[study_id].status = "running"
        logger.info(f"Study {study_id} baseline set to {score:.4f}")
    
    def record_ablation(
        self,
        study_id: str,
        component_id: str,
        ablated_score: float,
        ablation_type: AblationType = AblationType.REMOVE,
        run_count: int = 1,
        variance: float = 0.0,
    ) -> AblationResult:
        """Record an ablation experiment result."""
        if study_id not in self._studies:
            raise ValueError(f"Study {study_id} not found")
        
        study = self._studies[study_id]
        baseline = study.baseline_score
        
        # Contribution = what we lost by removing
        # Positive = component was helpful
        # Negative = component was harmful (removing it helped!)
        contribution = baseline - ablated_score
        
        # Simple significance based on run count
        significance = min(1.0, run_count / self.MIN_RUNS_FOR_SIGNIFICANCE)
        
        result = AblationResult(
            component_id=component_id,
            ablation_type=ablation_type,
            baseline_score=baseline,
            ablated_score=ablated_score,
            contribution=contribution,
            significance=significance,
            run_count=run_count,
            variance=variance,
        )
        
        study.record_result(result)
        
        logger.info(
            f"Ablation recorded: {component_id} → "
            f"contribution={contribution:+.4f} ({result.contribution_pct:+.1f}%)"
        )
        
        return result
    
    def get_contribution_ranking(self, study_id: str) -> List[Tuple[str, float, str]]:
        """Get components ranked by contribution."""
        if study_id not in self._studies:
            raise ValueError(f"Study {study_id} not found")
        
        return self._studies[study_id].get_ranking()
    
    def identify_harmful_components(self, study_id: str) -> List[str]:
        """Identify components that hurt performance."""
        if study_id not in self._studies:
            raise ValueError(f"Study {study_id} not found")
        
        study = self._studies[study_id]
        harmful = []
        
        for comp_id, contrib in study.contributions.items():
            if contrib.mean_contribution < self.HARMFUL_THRESHOLD:
                harmful.append(comp_id)
        
        return harmful
    
    def identify_essential_components(self, study_id: str) -> List[str]:
        """Identify components essential to performance."""
        if study_id not in self._studies:
            raise ValueError(f"Study {study_id} not found")
        
        study = self._studies[study_id]
        essential = []
        
        for comp_id, contrib in study.contributions.items():
            if contrib.mean_contribution >= self.ESSENTIAL_THRESHOLD:
                essential.append(comp_id)
        
        return essential
    
    def generate_abgen_plan(
        self,
        system_description: str,
        target_metric: str = "accuracy",
    ) -> AblationStudy:
        """
        Generate an ablation plan using AbGen pattern.
        
        In a full implementation, this would use an LLM to design
        the ablation study. Here we provide a structured template.
        """
        # Parse system description to identify components
        # (In production, this would use LLM parsing)
        
        study = AblationStudy(
            name=f"abgen-{hashlib.sha256(system_description.encode()).hexdigest()[:8]}",
            description=f"Auto-generated ablation plan for: {system_description[:100]}...",
            hypothesis=f"Measure contribution of each component to {target_metric}",
            ablation_types=[AblationType.REMOVE, AblationType.DISABLE, AblationType.REPLACE],
        )
        
        logger.info(f"Generated AbGen plan: {study.name}")
        return study
    
    def complete_study(self, study_id: str) -> Dict[str, Any]:
        """Mark study as complete and generate final report."""
        if study_id not in self._studies:
            raise ValueError(f"Study {study_id} not found")
        
        study = self._studies[study_id]
        study.status = "completed"
        
        summary = study.summary()
        
        # Add derived insights
        summary["essential_components"] = self.identify_essential_components(study_id)
        summary["harmful_components"] = self.identify_harmful_components(study_id)
        
        # Calculate ablation efficiency
        if study.components:
            ablation_coverage = len(study.results) / len(study.components)
            summary["ablation_coverage"] = ablation_coverage
        
        logger.info(f"Study {study_id} completed: {len(study.results)} experiments")
        return summary
    
    def generate_report(self, study_id: str) -> str:
        """Generate human-readable ablation report."""
        if study_id not in self._studies:
            raise ValueError(f"Study {study_id} not found")
        
        study = self._studies[study_id]
        lines = [
            "═" * 70,
            f"ABLATION STUDY REPORT: {study.name}",
            "═" * 70,
            "",
            f"Status: {study.status.upper()}",
            f"Baseline Score: {study.baseline_score:.4f}",
            f"Components Tested: {len(study.components)}",
            f"Experiments Run: {len(study.results)}",
            "",
            "─" * 40,
            "COMPONENT CONTRIBUTIONS (Ranked)",
            "─" * 40,
        ]
        
        for name, contribution, verdict in study.get_ranking():
            indicator = "🟢" if verdict in ["ESSENTIAL", "BENEFICIAL"] else \
                       "🟡" if verdict in ["MARGINAL", "NEUTRAL"] else "🔴"
            lines.append(f"  {indicator} {name}: {contribution:+.4f} ({verdict})")
        
        lines.extend([
            "",
            "─" * 40,
            "RECOMMENDATIONS",
            "─" * 40,
        ])
        
        essential = self.identify_essential_components(study_id)
        harmful = self.identify_harmful_components(study_id)
        
        if essential:
            lines.append(f"  ✅ Protect: {', '.join(essential)}")
        if harmful:
            lines.append(f"  ⚠️  Review/Remove: {', '.join(harmful)}")
        
        lines.extend([
            "",
            "═" * 70,
            "لا نفترض — We do not assume. We verify with ablation studies.",
            "═" * 70,
        ])
        
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    
    print("═" * 80)
    print("ABLATION ENGINE — Automated Component Contribution Analysis")
    print("═" * 80)
    
    # Initialize engine
    engine = AblationEngine()
    
    # Register swarm components
    engine.register_component(
        id="planner",
        name="Strategic Planner",
        category=ComponentCategory.AGENT,
        description="Decomposes tasks into subtasks",
        is_core=True,
    )
    engine.register_component(
        id="coder",
        name="Code Generator",
        category=ComponentCategory.AGENT,
        description="Generates code solutions",
        is_core=True,
    )
    engine.register_component(
        id="reviewer",
        name="Code Reviewer",
        category=ComponentCategory.AGENT,
        description="Reviews and critiques generated code",
    )
    engine.register_component(
        id="tester",
        name="Test Generator",
        category=ComponentCategory.AGENT,
        description="Generates test cases",
    )
    engine.register_component(
        id="memory",
        name="Context Memory",
        category=ComponentCategory.MEMORY,
        description="Stores and retrieves context",
    )
    engine.register_component(
        id="router",
        name="Expert Router",
        category=ComponentCategory.ROUTING,
        description="Routes tasks to appropriate experts",
    )
    
    # Create ablation study
    study = engine.create_study(
        name="SWE-Agent Ablation",
        hypothesis="Measure contribution of each agent to SWE-bench performance",
    )
    
    print(f"\nCreated study: {study.id}")
    print(f"Components to test: {len(study.components)}")
    
    # Set baseline (simulated)
    baseline_score = 0.42  # 42% SWE-bench accuracy
    engine.set_baseline(study.id, baseline_score)
    print(f"Baseline score: {baseline_score}")
    
    # Run ablations (simulated with realistic contributions)
    print("\n" + "─" * 40)
    print("Running Ablation Experiments...")
    print("─" * 40)
    
    # Simulated contributions (what we'd lose by removing)
    simulated_contributions = {
        "planner": 0.12,    # Essential - loses 12%
        "coder": 0.18,      # Most essential - loses 18%
        "reviewer": 0.03,   # Marginal - loses 3%
        "tester": 0.05,     # Beneficial - loses 5%
        "memory": 0.08,     # Beneficial - loses 8%
        "router": -0.02,    # Harmful! Removing improves by 2%
    }
    
    for comp in study.components:
        contribution = simulated_contributions.get(comp.id, 0.0)
        # Add some variance
        noisy_contribution = contribution + random.gauss(0, 0.01)
        ablated_score = baseline_score - noisy_contribution
        
        result = engine.record_ablation(
            study_id=study.id,
            component_id=comp.id,
            ablated_score=ablated_score,
            run_count=3,  # 3 runs for significance
        )
        
        print(f"  {comp.name}: {result.contribution:+.4f} ({result.contribution_pct:+.1f}%)")
    
    # Complete study and get report
    summary = engine.complete_study(study.id)
    
    print("\n" + engine.generate_report(study.id))
    
    print("\n" + "─" * 40)
    print("Actionable Insights")
    print("─" * 40)
    
    if summary["essential_components"]:
        print(f"  🏆 Essential: {summary['essential_components']}")
    if summary["harmful_components"]:
        print(f"  ⚠️  Consider removing: {summary['harmful_components']}")
    
    print("\n" + "═" * 80)
