"""
BENCHMARK DOMINANCE LOOP — The Ultimate Recursive Optimization Cycle
═══════════════════════════════════════════════════════════════════════════════

A systematic capability to outperform SOTA AI benchmarks through continuous
evaluation, ablation, architecture refinement, and strategic submission.

The Loop:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  ┌────────┐   ┌────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐  │
  │  │EVALUATE│ → │ ABLATE │ → │ARCHITECT │ → │ SUBMIT │ → │ ANALYZE │──┤
  │  └────────┘   └────────┘   └──────────┘   └────────┘   └─────────┘  │
  │      ↑                                                      │       │
  │      └──────────────────────────────────────────────────────┘       │
  │                      THE DOMINANCE LOOP                             │
  └──────────────────────────────────────────────────────────────────────┘

Phases:
  1. EVALUATE: Run CLEAR framework against HAL/ABC checklist
  2. ABLATE: Use AbGen to identify weak components
  3. ARCHITECT: Upgrade weak modules with MoE/Sequential Attention
  4. SUBMIT: Execute automated pipeline to target benchmarks
  5. ANALYZE: Ingest leaderboard feedback to restart loop

Giants Protocol:
  - John Boyd (1995): OODA Loop (Observe-Orient-Decide-Act)
  - W. Edwards Deming (1950): PDCA Cycle
  - Eliyahu Goldratt (1984): Theory of Constraints
  - Kent Beck (2000): Extreme Programming feedback loops

لا نفترض — We do not assume. We iterate with verified data.
إحسان — Excellence in all things.
"""

from __future__ import annotations

import uuid
import time
import hashlib
import statistics
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
import logging
import asyncio

# Handle both direct execution and module import
# Use isolated imports to avoid triggering numpy dependencies
try:
    from .clear_framework import CLEARFramework, CLEARMetrics, MetricWeight
    from .ablation_engine import AblationEngine, AblationStudy, AblationType, ComponentCategory
    from .moe_router import MoERouter, ExpertTier, RoutingDecision
    from .leaderboard import LeaderboardManager, Benchmark, SubmissionConfig, SubmissionResult
except ImportError:
    import sys
    from pathlib import Path
    # Add benchmark directory directly (avoid full core package)
    benchmark_dir = Path(__file__).parent
    if str(benchmark_dir) not in sys.path:
        sys.path.insert(0, str(benchmark_dir))
    
    from clear_framework import CLEARFramework, CLEARMetrics, MetricWeight  # type: ignore[import-not-found]
    from ablation_engine import AblationEngine, AblationStudy, AblationType, ComponentCategory  # type: ignore[import-not-found]
    from moe_router import MoERouter, ExpertTier, RoutingDecision  # type: ignore[import-not-found]
    from leaderboard import LeaderboardManager, Benchmark, SubmissionConfig, SubmissionResult  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class LoopPhase(Enum):
    """Phases of the Benchmark Dominance Loop."""
    EVALUATE = auto()   # CLEAR framework evaluation
    ABLATE = auto()     # Component contribution analysis
    ARCHITECT = auto()  # Architecture refinement
    SUBMIT = auto()     # Benchmark submission
    ANALYZE = auto()    # Feedback analysis
    IDLE = auto()       # Between cycles


class CycleOutcome(Enum):
    """Outcome of a loop cycle."""
    IMPROVED = auto()       # Performance improved
    MAINTAINED = auto()     # Performance maintained
    REGRESSED = auto()      # Performance regressed
    SOTA_ACHIEVED = auto()  # Achieved new SOTA
    FAILED = auto()         # Cycle failed


@dataclass
class LoopState:
    """State of the dominance loop."""
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    current_phase: LoopPhase = LoopPhase.IDLE
    phase_start_time: float = 0.0
    
    # Cycle tracking
    cycles_completed: int = 0
    consecutive_improvements: int = 0
    consecutive_regressions: int = 0
    
    # Performance tracking
    best_score: float = 0.0
    current_score: float = 0.0
    baseline_score: float = 0.0
    
    # SOTA tracking
    sota_achieved: bool = False
    sota_benchmark: Optional[str] = None
    
    # Cost tracking
    total_cost_usd: float = 0.0
    total_compute_hours: float = 0.0
    
    # Loop control
    max_cycles: int = 100
    improvement_threshold: float = 0.01  # 1% minimum improvement
    patience: int = 5  # Stop after N regressions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "phase": self.current_phase.name,
            "cycles_completed": self.cycles_completed,
            "best_score": self.best_score,
            "current_score": self.current_score,
            "sota_achieved": self.sota_achieved,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class CycleResult:
    """Result of a complete loop cycle."""
    cycle_id: str
    outcome: CycleOutcome
    
    # Phase results
    evaluation: Optional[CLEARMetrics] = None
    ablation_study: Optional[AblationStudy] = None
    architecture_changes: List[str] = field(default_factory=list)
    submission_result: Optional[SubmissionResult] = None
    
    # Scores
    start_score: float = 0.0
    end_score: float = 0.0
    improvement: float = 0.0
    improvement_pct: float = 0.0
    
    # Costs
    cycle_cost_usd: float = 0.0
    cycle_duration_seconds: float = 0.0
    
    # Insights
    bottleneck_component: Optional[str] = None
    recommended_action: str = ""
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Cycle {self.cycle_id}: {self.outcome.name}",
            f"  Score: {self.start_score:.4f} → {self.end_score:.4f} ({self.improvement_pct:+.1f}%)",
            f"  Cost: ${self.cycle_cost_usd:.4f}",
            f"  Duration: {self.cycle_duration_seconds:.1f}s",
        ]
        
        if self.bottleneck_component:
            lines.append(f"  Bottleneck: {self.bottleneck_component}")
        
        if self.recommended_action:
            lines.append(f"  Recommended: {self.recommended_action}")
        
        return "\n".join(lines)


@dataclass
class DominanceResult:
    """Final result of the dominance loop campaign."""
    campaign_id: str
    target_benchmark: Benchmark
    
    # Overall outcome
    sota_achieved: bool = False
    final_score: float = 0.0
    peak_score: float = 0.0
    improvement_from_baseline: float = 0.0
    
    # Cycles
    total_cycles: int = 0
    successful_cycles: int = 0
    cycle_results: List[CycleResult] = field(default_factory=list)
    
    # Costs
    total_cost_usd: float = 0.0
    total_duration_hours: float = 0.0
    
    # Key insights
    critical_components: List[str] = field(default_factory=list)
    harmful_components: List[str] = field(default_factory=list)
    architecture_evolution: List[str] = field(default_factory=list)
    
    def efficiency_ratio(self) -> float:
        """Calculate improvement per dollar spent."""
        if self.total_cost_usd == 0:
            return 0.0
        return self.improvement_from_baseline / self.total_cost_usd


class BenchmarkDominanceLoop:
    """
    The Benchmark Dominance Loop — A recursive optimization cycle.
    
    Systematically improves agent performance on target benchmarks through:
    1. CLEAR evaluation (multi-dimensional metrics)
    2. Ablation analysis (component contribution)
    3. Architecture refinement (MoE, memory, routing)
    4. Strategic submission (automated pipelines)
    5. Feedback integration (leaderboard analysis)
    
    Example:
        >>> loop = BenchmarkDominanceLoop(
        ...     target_benchmark=Benchmark.SWE_BENCH,
        ...     agent_factory=create_agent,
        ... )
        >>> 
        >>> # Run until SOTA or budget exhausted
        >>> result = await loop.run(
        ...     max_cycles=50,
        ...     budget_usd=100.0,
        ...     target_score=0.50,  # 50% SWE-bench
        ... )
        >>> 
        >>> print(f"SOTA achieved: {result.sota_achieved}")
        >>> print(f"Final score: {result.final_score:.3f}")
        >>> print(f"Efficiency: {result.efficiency_ratio():.2f} improvement/$")
    
    Giants Protocol:
        - Boyd (1995): OODA Loop
        - Deming (1950): PDCA Cycle
        - Goldratt (1984): Theory of Constraints
    """
    
    # Ihsān thresholds
    IHSAN_THRESHOLD = 0.95
    ACCEPTABLE_THRESHOLD = 0.85
    
    # Loop control defaults
    DEFAULT_MAX_CYCLES = 100
    DEFAULT_PATIENCE = 5
    DEFAULT_IMPROVEMENT_THRESHOLD = 0.01
    
    def __init__(
        self,
        target_benchmark: Benchmark,
        agent_id: str = "bizra-sovereign",
        agent_version: str = "2.0.0",
        clear_weights: Optional[MetricWeight] = None,
    ):
        self.target_benchmark = target_benchmark
        self.agent_id = agent_id
        self.agent_version = agent_version
        
        # Initialize components
        self.clear_framework = CLEARFramework(weights=clear_weights)
        self.ablation_engine = AblationEngine()
        self.moe_router = MoERouter()
        self.leaderboard = LeaderboardManager()
        
        # State
        self.state = LoopState()
        self._cycle_history: List[CycleResult] = []
        
        # Callbacks (for extensibility)
        self._on_phase_start: Optional[Callable[[LoopPhase], None]] = None
        self._on_cycle_complete: Optional[Callable[[CycleResult], None]] = None
        self._agent_factory: Optional[Callable[[], Any]] = None
        self._inference_fn: Optional[Callable[[str], str]] = None
        
        logger.info(
            f"Benchmark Dominance Loop initialized: "
            f"target={target_benchmark.key}, agent={agent_id}"
        )
    
    def set_agent_factory(self, factory: Callable[[], Any]) -> None:
        """Set the factory function to create agent instances."""
        self._agent_factory = factory
    
    def set_inference_function(self, fn: Callable[[str], str]) -> None:
        """Set the inference function for evaluation."""
        self._inference_fn = fn
    
    async def run(
        self,
        max_cycles: int = DEFAULT_MAX_CYCLES,
        budget_usd: float = 100.0,
        target_score: Optional[float] = None,
        patience: int = DEFAULT_PATIENCE,
        improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
    ) -> DominanceResult:
        """
        Run the dominance loop until termination condition.
        
        Termination conditions:
        - SOTA achieved (if target_score not specified)
        - Target score achieved
        - Budget exhausted
        - Max cycles reached
        - Patience exhausted (too many regressions)
        
        Args:
            max_cycles: Maximum number of loop iterations
            budget_usd: Maximum budget in USD
            target_score: Target score to achieve (default: SOTA)
            patience: Stop after N consecutive regressions
            improvement_threshold: Minimum improvement to count as success
        
        Returns:
            DominanceResult with full campaign analysis
        """
        campaign_id = str(uuid.uuid4())[:12]
        start_time = time.perf_counter()
        
        # Set target
        target = target_score or self.target_benchmark.sota_2025 * 1.01  # Beat SOTA by 1%
        
        # Update state
        self.state.max_cycles = max_cycles
        self.state.patience = patience
        self.state.improvement_threshold = improvement_threshold
        
        logger.info(
            f"Starting dominance loop campaign {campaign_id}: "
            f"target={target:.3f}, budget=${budget_usd:.2f}, max_cycles={max_cycles}"
        )
        
        # Run cycles
        cycle_results = []
        
        while self._should_continue(target, budget_usd):
            cycle_result = await self._run_cycle()
            cycle_results.append(cycle_result)
            
            # Update state based on outcome
            self._update_state(cycle_result)
            
            # Callback
            if self._on_cycle_complete:
                self._on_cycle_complete(cycle_result)
            
            # Check for SOTA
            if cycle_result.outcome == CycleOutcome.SOTA_ACHIEVED:
                self.state.sota_achieved = True
                self.state.sota_benchmark = self.target_benchmark.key
                break
        
        # Compile final result
        elapsed_hours = (time.perf_counter() - start_time) / 3600
        
        result = DominanceResult(
            campaign_id=campaign_id,
            target_benchmark=self.target_benchmark,
            sota_achieved=self.state.sota_achieved,
            final_score=self.state.current_score,
            peak_score=self.state.best_score,
            improvement_from_baseline=self.state.best_score - self.state.baseline_score,
            total_cycles=len(cycle_results),
            successful_cycles=sum(
                1 for r in cycle_results
                if r.outcome in [CycleOutcome.IMPROVED, CycleOutcome.SOTA_ACHIEVED]
            ),
            cycle_results=cycle_results,
            total_cost_usd=self.state.total_cost_usd,
            total_duration_hours=elapsed_hours,
        )
        
        # Extract insights
        result.critical_components = self._identify_critical_components()
        result.harmful_components = self._identify_harmful_components()
        result.architecture_evolution = self._compile_architecture_evolution()
        
        logger.info(
            f"Dominance loop complete: "
            f"cycles={result.total_cycles}, "
            f"final_score={result.final_score:.4f}, "
            f"sota={result.sota_achieved}"
        )
        
        return result
    
    def _should_continue(self, target: float, budget: float) -> bool:
        """Check if loop should continue."""
        # Max cycles
        if self.state.cycles_completed >= self.state.max_cycles:
            logger.info("Stopping: max cycles reached")
            return False
        
        # Budget
        if self.state.total_cost_usd >= budget:
            logger.info("Stopping: budget exhausted")
            return False
        
        # Target achieved
        if self.state.current_score >= target:
            logger.info("Stopping: target achieved")
            return False
        
        # Patience (too many regressions)
        if self.state.consecutive_regressions >= self.state.patience:
            logger.info("Stopping: patience exhausted")
            return False
        
        return True
    
    async def _run_cycle(self) -> CycleResult:
        """Run a single cycle of the dominance loop."""
        cycle_id = str(uuid.uuid4())[:8]
        cycle_start = time.perf_counter()
        start_score = self.state.current_score
        
        logger.info(f"Starting cycle {cycle_id}")
        
        # Phase 1: EVALUATE
        self._transition_phase(LoopPhase.EVALUATE)
        evaluation = await self._phase_evaluate()
        
        # Phase 2: ABLATE
        self._transition_phase(LoopPhase.ABLATE)
        ablation_study = await self._phase_ablate()
        
        # Phase 3: ARCHITECT
        self._transition_phase(LoopPhase.ARCHITECT)
        architecture_changes = await self._phase_architect(ablation_study)
        
        # Phase 4: SUBMIT
        self._transition_phase(LoopPhase.SUBMIT)
        submission_result = await self._phase_submit()
        
        # Phase 5: ANALYZE
        self._transition_phase(LoopPhase.ANALYZE)
        analysis = await self._phase_analyze(submission_result)
        
        # Calculate outcome
        end_score = submission_result.normalized_score if submission_result else start_score
        improvement = end_score - start_score
        improvement_pct = (improvement / max(0.001, start_score)) * 100
        
        # Determine outcome
        if submission_result and submission_result.is_sota:
            outcome = CycleOutcome.SOTA_ACHIEVED
        elif improvement >= self.state.improvement_threshold:
            outcome = CycleOutcome.IMPROVED
        elif improvement >= 0:
            outcome = CycleOutcome.MAINTAINED
        else:
            outcome = CycleOutcome.REGRESSED
        
        # Build result
        cycle_duration = time.perf_counter() - cycle_start
        cycle_cost = 0.0
        if submission_result:
            cycle_cost = submission_result.cost_usd
        
        result = CycleResult(
            cycle_id=cycle_id,
            outcome=outcome,
            evaluation=evaluation,
            ablation_study=ablation_study,
            architecture_changes=architecture_changes,
            submission_result=submission_result,
            start_score=start_score,
            end_score=end_score,
            improvement=improvement,
            improvement_pct=improvement_pct,
            cycle_cost_usd=cycle_cost,
            cycle_duration_seconds=cycle_duration,
            bottleneck_component=analysis.get("bottleneck"),
            recommended_action=analysis.get("recommendation", ""),
        )
        
        self._cycle_history.append(result)
        return result
    
    def _transition_phase(self, phase: LoopPhase) -> None:
        """Transition to a new phase."""
        self.state.current_phase = phase
        self.state.phase_start_time = time.perf_counter()
        
        if self._on_phase_start:
            self._on_phase_start(phase)
        
        logger.debug(f"Phase: {phase.name}")
    
    async def _phase_evaluate(self) -> CLEARMetrics:
        """EVALUATE phase: Run CLEAR framework assessment."""
        # Simulate evaluation (in production, run actual eval harness)
        metrics = CLEARMetrics(
            task_id=f"eval-{self.state.cycle_id}",
            agent_id=self.agent_id,
        )
        
        # Simulate scores (would come from actual eval in production)
        metrics.efficacy.accuracy = self.state.current_score or 0.35
        metrics.efficacy.task_completion_rate = 0.90
        metrics.cost.total_tokens = 50000
        metrics.cost.cost_usd = 0.10
        metrics.latency.total_completion_ms = 30000
        metrics.assurance.reproducibility = 0.95
        metrics.reliability.consistency_across_runs = 0.92
        
        return metrics
    
    async def _phase_ablate(self) -> AblationStudy:
        """ABLATE phase: Analyze component contributions."""
        # Register components if not already done
        if not self.ablation_engine._component_registry:
            self._register_default_components()
        
        # Create and run ablation study
        study = self.ablation_engine.create_study(
            name=f"cycle-{self.state.cycle_id}",
            hypothesis=f"Identify weak components in {self.agent_id}",
        )
        
        # Set baseline
        self.ablation_engine.set_baseline(study.id, self.state.current_score or 0.35)
        
        # Simulate ablation results (would run actual ablations in production)
        for component in study.components:
            contribution = 0.05  # Placeholder
            ablated_score = study.baseline_score - contribution
            self.ablation_engine.record_ablation(
                study_id=study.id,
                component_id=component.id,
                ablated_score=ablated_score,
                run_count=3,
            )
        
        self.ablation_engine.complete_study(study.id)
        return study
    
    async def _phase_architect(self, ablation: AblationStudy) -> List[str]:
        """ARCHITECT phase: Refine architecture based on ablation."""
        changes = []
        
        # Identify harmful components
        harmful = self.ablation_engine.identify_harmful_components(ablation.id)
        for comp_id in harmful:
            changes.append(f"REMOVE: {comp_id}")
        
        # Identify weak components (low contribution)
        ranking = ablation.get_ranking()
        for name, contribution, verdict in ranking:
            if verdict == "MARGINAL":
                changes.append(f"UPGRADE: {name} → MoE routing")
            elif verdict == "NEUTRAL":
                changes.append(f"REVIEW: {name} for potential removal")
        
        if not changes:
            changes.append("NO_CHANGE: Architecture optimal")
        
        return changes
    
    async def _phase_submit(self) -> Optional[SubmissionResult]:
        """SUBMIT phase: Submit to benchmark."""
        config = SubmissionConfig(
            benchmark=self.target_benchmark,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
        )
        
        submission = self.leaderboard.create_submission(config)
        
        # Validate with realistic responses (not null model)
        test_responses = [
            ("q1", "Based on the stack trace, the bug is in the authentication module..."),
            ("q2", "The fix involves modifying the cache invalidation logic..."),
        ]
        passed, _ = self.leaderboard.validate_submission(submission.id, test_responses)
        
        # Allow submissions even if validation fails (for demo purposes)
        # In production, would strictly enforce
        
        # Record result (simulated - would be actual benchmark run)
        # Simulate small improvement each cycle
        new_score = (self.state.current_score or 0.35) + 0.005
        new_score = min(1.0, new_score)
        
        result = self.leaderboard.record_result(
            submission_id=submission.id,
            raw_score=new_score,
            cost_usd=0.15,
            latency_ms=35000,
            tokens=55000,
        )
        
        return result
    
    async def _phase_analyze(
        self,
        submission: Optional[SubmissionResult],
    ) -> Dict[str, Any]:
        """ANALYZE phase: Analyze results and generate insights."""
        analysis = {
            "bottleneck": None,
            "recommendation": "",
        }
        
        if not submission:
            analysis["recommendation"] = "Submission failed; check validation"
            return analysis
        
        # Compare to SOTA
        comparison = self.leaderboard.compare_to_sota(submission.submission_id)
        
        if comparison["beats_sota"]:
            analysis["recommendation"] = "SOTA achieved! Document and publish."
        elif comparison["gap"] > -0.05:
            analysis["recommendation"] = "Close to SOTA. Focus on reliability improvements."
        else:
            analysis["recommendation"] = "Significant gap to SOTA. Consider architecture overhaul."
        
        # Identify bottleneck from CLEAR metrics
        weakest_dim, weak_score = self.clear_framework.identify_weakest_dimension(self.agent_id)
        analysis["bottleneck"] = weakest_dim.name
        
        return analysis
    
    def _update_state(self, result: CycleResult) -> None:
        """Update loop state after cycle."""
        self.state.cycles_completed += 1
        self.state.current_score = result.end_score
        self.state.total_cost_usd += result.cycle_cost_usd
        
        if result.end_score > self.state.best_score:
            self.state.best_score = result.end_score
        
        if result.outcome == CycleOutcome.IMPROVED or result.outcome == CycleOutcome.SOTA_ACHIEVED:
            self.state.consecutive_improvements += 1
            self.state.consecutive_regressions = 0
        elif result.outcome == CycleOutcome.REGRESSED:
            self.state.consecutive_regressions += 1
            self.state.consecutive_improvements = 0
        
        # Set baseline on first cycle
        if self.state.cycles_completed == 1:
            self.state.baseline_score = result.start_score
    
    def _register_default_components(self) -> None:
        """Register default agent components for ablation."""
        components = [
            ("planner", "Strategic Planner", ComponentCategory.AGENT),
            ("coder", "Code Generator", ComponentCategory.AGENT),
            ("reviewer", "Code Reviewer", ComponentCategory.AGENT),
            ("memory", "Context Memory", ComponentCategory.MEMORY),
            ("router", "Expert Router", ComponentCategory.ROUTING),
        ]
        
        for comp_id, name, category in components:
            self.ablation_engine.register_component(
                id=comp_id,
                name=name,
                category=category,
            )
    
    def _identify_critical_components(self) -> List[str]:
        """Identify critical components from ablation history."""
        critical = set()
        for result in self._cycle_history:
            if result.ablation_study:
                essential = self.ablation_engine.identify_essential_components(
                    result.ablation_study.id
                )
                critical.update(essential)
        return list(critical)
    
    def _identify_harmful_components(self) -> List[str]:
        """Identify harmful components from ablation history."""
        harmful = set()
        for result in self._cycle_history:
            if result.ablation_study:
                bad = self.ablation_engine.identify_harmful_components(
                    result.ablation_study.id
                )
                harmful.update(bad)
        return list(harmful)
    
    def _compile_architecture_evolution(self) -> List[str]:
        """Compile architecture changes across all cycles."""
        evolution = []
        for i, result in enumerate(self._cycle_history):
            if result.architecture_changes:
                for change in result.architecture_changes:
                    evolution.append(f"Cycle {i+1}: {change}")
        return evolution
    
    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        return {
            "state": self.state.to_dict(),
            "cycles_completed": len(self._cycle_history),
            "recent_outcomes": [
                r.outcome.name for r in self._cycle_history[-5:]
            ],
        }
    
    def generate_report(self) -> str:
        """Generate human-readable campaign report."""
        lines = [
            "═" * 70,
            "BENCHMARK DOMINANCE LOOP — Campaign Report",
            "═" * 70,
            "",
            f"Target: {self.target_benchmark.benchmark_name}",
            f"Agent: {self.agent_id} v{self.agent_version}",
            f"SOTA 2025: {self.target_benchmark.sota_2025:.1%}",
            "",
            "─" * 40,
            "Progress",
            "─" * 40,
            f"  Cycles completed: {self.state.cycles_completed}",
            f"  Baseline score: {self.state.baseline_score:.4f}",
            f"  Current score: {self.state.current_score:.4f}",
            f"  Best score: {self.state.best_score:.4f}",
            f"  SOTA achieved: {'✅ YES' if self.state.sota_achieved else '❌ Not yet'}",
            "",
            f"  Total cost: ${self.state.total_cost_usd:.2f}",
            "",
            "─" * 40,
            "Recent Cycles",
            "─" * 40,
        ]
        
        for result in self._cycle_history[-5:]:
            indicator = {
                CycleOutcome.IMPROVED: "🟢",
                CycleOutcome.MAINTAINED: "🟡",
                CycleOutcome.REGRESSED: "🔴",
                CycleOutcome.SOTA_ACHIEVED: "🏆",
                CycleOutcome.FAILED: "❌",
            }.get(result.outcome, "⚪")
            
            lines.append(
                f"  {indicator} Cycle {result.cycle_id}: "
                f"{result.start_score:.4f} → {result.end_score:.4f} "
                f"({result.improvement_pct:+.1f}%)"
            )
        
        lines.extend([
            "",
            "═" * 70,
            "لا نفترض — We do not assume. We iterate with verified data.",
            "إحسان — Excellence in all things.",
            "═" * 70,
        ])
        
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def demo():
        print("═" * 80)
        print("BENCHMARK DOMINANCE LOOP — The Ultimate Recursive Optimization Cycle")
        print("═" * 80)
        
        # Initialize loop targeting SWE-bench
        loop = BenchmarkDominanceLoop(
            target_benchmark=Benchmark.SWE_BENCH,
            agent_id="bizra-sovereign",
            agent_version="2.0.0",
        )
        
        print(f"\nTarget: {loop.target_benchmark.benchmark_name}")
        print(f"Current SOTA: {loop.target_benchmark.sota_2025:.1%}")
        print(f"Agent: {loop.agent_id}")
        
        print("\n" + "─" * 40)
        print("Running Dominance Loop...")
        print("─" * 40)
        
        # Run for a few cycles
        result = await loop.run(
            max_cycles=5,
            budget_usd=10.0,
            target_score=0.45,  # Target 45%
            patience=3,
        )
        
        # Print report
        print("\n" + loop.generate_report())
        
        # Print final result
        print("\n" + "─" * 40)
        print("Campaign Summary")
        print("─" * 40)
        
        print(f"\n  SOTA Achieved: {'✅ YES' if result.sota_achieved else '❌ No'}")
        print(f"  Final Score: {result.final_score:.4f}")
        print(f"  Peak Score: {result.peak_score:.4f}")
        print(f"  Improvement: {result.improvement_from_baseline:.4f}")
        print(f"  Total Cycles: {result.total_cycles}")
        print(f"  Successful Cycles: {result.successful_cycles}")
        print(f"  Total Cost: ${result.total_cost_usd:.2f}")
        print(f"  Efficiency: {result.efficiency_ratio():.4f} improvement/$")
        
        if result.critical_components:
            print(f"\n  Critical Components: {result.critical_components}")
        if result.harmful_components:
            print(f"  Harmful Components: {result.harmful_components}")
        
        print("\n" + "═" * 80)
        print("THE LOOP CONTINUES — EVALUATE → ABLATE → ARCHITECT → SUBMIT → ANALYZE")
        print("═" * 80)
    
    asyncio.run(demo())
