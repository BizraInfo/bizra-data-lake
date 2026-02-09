"""
BENCHMARK DOMINANCE LOOP — The True Spearpoint
═══════════════════════════════════════════════════════════════════════════════

A recursive optimization cycle to systematically outperform SOTA AI benchmarks.

Three Pillars:
  1. EVALUATION HARNESS — Rigorous, agentic-first testing ground
  2. MODEL ARCHITECTURE — MoE, Federated, Sequential Attention
  3. LEADERBOARD SUBMISSION — Automated, anti-gaming, cost-aware

The Loop:
  EVALUATE → ABLATE → ARCHITECT → SUBMIT → ANALYZE → (repeat)

7-3-6-9 DNA:
  - 7 agents: Orchestrator, Evaluator, Ablator, Architect, Submitter, Analyst, Guardian
  - 3 gates: Reproducibility → Integrity → Budget (FAIL-CLOSED)
  - 6 stages: Setup → Run → Validate → Ablate → Patch → Submit
  - 9 guardrails: leakage, injection, null-model, regression, seed-sweep, sandbox, provenance, cost-cap, rollback

Giants Protocol:
  - Eleuther AI (2023): lm-evaluation-harness
  - Berkeley RDI (2025): AgentBeats protocol
  - Noam Shazeer (2017): Mixture-of-Experts
  - John Boyd (1995): OODA Loop
  - Claude Shannon (1948): Information theory

لا نفترض — We do not assume. We verify with formal proofs.
إحسان — Excellence in all things.
"""

from .clear_framework import (
    CLEARFramework,
    CLEARMetrics,
    CLEARDimension,
    MetricWeight,
    AgenticBenchmarkChecklist,
    EvaluationContext,
)
from .ablation_engine import (
    AblationEngine,
    AblationStudy,
    AblationResult,
    ComponentContribution,
    AblationType,
    Component,
    ComponentCategory,
)
from .moe_router import (
    MoERouter,
    ExpertTier,
    RoutingDecision,
    FederatedDispatch,
    SequentialAttention,
    QueryComplexity,
)
from .leaderboard import (
    LeaderboardManager,
    Submission,
    SubmissionResult,
    SubmissionConfig,
    SubmissionStatus,
    Benchmark,
    AntiGamingValidator,
)
from .dominance_loop import (
    BenchmarkDominanceLoop,
    LoopPhase,
    LoopState,
    CycleResult,
    CycleOutcome,
    DominanceResult,
)
from .guardrails import (
    GuardrailSuite,
    GuardrailType,
    GuardrailStatus,
    GuardrailResult,
    LeakageScanner,
    PromptInjectionGuard,
    NullModelProbe,
    RegressionGate,
    SeedSweepValidator,
    ToolSandbox,
    ProvenanceLogger,
    CostCapEnforcer,
    RollbackManager,
)

__all__ = [
    # CLEAR Framework
    "CLEARFramework",
    "CLEARMetrics",
    "CLEARDimension",
    "MetricWeight",
    "AgenticBenchmarkChecklist",
    "EvaluationContext",
    # Ablation Engine
    "AblationEngine",
    "AblationStudy",
    "AblationResult",
    "ComponentContribution",
    "AblationType",
    "Component",
    "ComponentCategory",
    # MoE Router
    "MoERouter",
    "ExpertTier",
    "RoutingDecision",
    "FederatedDispatch",
    "SequentialAttention",
    "QueryComplexity",
    # Leaderboard
    "LeaderboardManager",
    "Submission",
    "SubmissionResult",
    "SubmissionConfig",
    "SubmissionStatus",
    "Benchmark",
    "AntiGamingValidator",
    # Dominance Loop
    "BenchmarkDominanceLoop",
    "LoopPhase",
    "LoopState",
    "CycleResult",
    "CycleOutcome",
    "DominanceResult",
    # Guardrails (9)
    "GuardrailSuite",
    "GuardrailType",
    "GuardrailStatus",
    "GuardrailResult",
    "LeakageScanner",
    "PromptInjectionGuard",
    "NullModelProbe",
    "RegressionGate",
    "SeedSweepValidator",
    "ToolSandbox",
    "ProvenanceLogger",
    "CostCapEnforcer",
    "RollbackManager",
]

__version__ = "1.0.0"
__giants__ = [
    "John Boyd (1995) — OODA Loop",
    "W. Edwards Deming (1950) — PDCA Cycle",
    "Eliyahu Goldratt (1984) — Theory of Constraints",
    "Noam Shazeer (2017) — Mixture-of-Experts",
    "Claude Shannon (1948) — Information theory",
    "Eleuther AI (2023) — lm-evaluation-harness",
    "Berkeley RDI (2025) — AgentBeats protocol",
    "Saltzer & Schroeder (1975) — Fail-closed design",
]

# 7-3-6-9 DNA summary
DNA_739 = {
    "agents": ["Orchestrator", "Evaluator", "Ablator", "Architect", "Submitter", "Analyst", "Guardian"],
    "gates": ["Reproducibility", "Integrity", "Budget"],
    "stages": ["Setup", "Run", "Validate", "Ablate", "Patch", "Submit"],
    "guardrails": [
        "Leakage Scan",
        "Prompt Injection",
        "Null Model",
        "Regression",
        "Seed Sweep",
        "Tool Sandbox",
        "Provenance",
        "Cost Cap",
        "Rollback",
    ],
}
