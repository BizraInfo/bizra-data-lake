"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA ELITE FRAMEWORK — Professional Excellence Blueprint                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   • PMBOK (Project Management Body of Knowledge)                             ║
║   • DevOps + GitOps (Infrastructure as Code)                                 ║
║   • SRE (Site Reliability Engineering)                                       ║
║   • Anthropic (Constitutional AI / Ihsān)                                    ║
║   • Shannon (Signal-to-Noise Optimization)                                   ║
║   • IEEE/ISO Standards (Quality Management)                                  ║
║   • Lamport (Logical Clocks / Happened-Before)                               ║
║   • Merkle (Hash Trees / DAG Structures)                                     ║
║   • Kahneman (Cognitive Load / System 1&2)                                   ║
║   • Harberger (Self-Assessed Taxation)                                       ║
║   • Gini (Inequality Measurement)                                            ║
║                                                                              ║
║   Elite Principles (Ihsān + Adl + Amānah):                                   ║
║   • IHSĀN (إحسان): Excellence in every artifact                              ║
║   • ADL (عدل): Justice and fairness in resource allocation                   ║
║   • AMĀNAH (أمانة): Trustworthiness and accountability                       ║
║                                                                              ║
║   Framework Components:                                                      ║
║   • Quality Gates: SNR-based validation at every stage                       ║
║   • CI/CD Pipeline: Constitutional validation in automation                  ║
║   • Metrics Dashboard: Real-time Ihsān compliance monitoring                 ║
║   • Risk Management: Cascading risk mitigation                               ║
║   • FATE Hooks: Pre-tool-use governance (v1.1.0)                             ║
║   • Session DAG: Merkle-DAG session state machine (v1.1.0)                   ║
║   • Cognitive Budget: 7-3-6-9 thinking allocation (v1.1.0)                   ║
║   • Compute Market: Harberger Tax + Gini enforcement (v1.1.0)                ║
║                                                                              ║
║   Created: 2026-02-02 | BIZRA Elite Integration v1.1.0                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Version
ELITE_VERSION = "1.1.0"

# PMBOK Knowledge Areas mapped to BIZRA
PMBOK_KNOWLEDGE_AREAS = {
    "integration": "core/elite/integration.py",
    "scope": "core/elite/scope.py",
    "schedule": "core/elite/schedule.py",
    "cost": "core/elite/cost.py",
    "quality": "core/elite/quality.py",
    "resource": "core/elite/resource.py",
    "communications": "core/elite/communications.py",
    "risk": "core/elite/risk.py",
    "procurement": "core/elite/procurement.py",
    "stakeholder": "core/elite/stakeholder.py",
}

# Ihsān Quality Dimensions
IHSAN_DIMENSIONS = {
    "correctness": 0.22,
    "safety": 0.22,
    "user_benefit": 0.14,
    "efficiency": 0.12,
    "auditability": 0.12,
    "anti_centralization": 0.08,
    "robustness": 0.06,
    "adl_justice": 0.04,
}

# SAPE Layers
SAPE_LAYERS = ["data", "information", "knowledge", "wisdom"]

# SNR Targets by Layer
SNR_TARGETS = {
    "data": 0.90,
    "information": 0.95,
    "knowledge": 0.99,
    "wisdom": 0.999,
}


# Lazy imports
def __getattr__(name: str):
    if name == "QualityGate":
        from .quality_gates import QualityGate

        return QualityGate
    elif name == "ElitePipeline":
        from .pipeline import ElitePipeline

        return ElitePipeline
    elif name == "MetricsDashboard":
        from .metrics import MetricsDashboard

        return MetricsDashboard
    elif name == "RiskManager":
        from .risk import RiskManager

        return RiskManager
    elif name == "SAPEOptimizer":
        from .sape import SAPEOptimizer

        return SAPEOptimizer
    elif name == "IhsanValidator":
        from .ihsan import IhsanValidator

        return IhsanValidator
    # v1.1.0 additions
    elif name == "FATEGate":
        from .hooks import FATEGate

        return FATEGate
    elif name == "HookRegistry":
        from .hooks import HookRegistry

        return HookRegistry
    elif name == "HookExecutor":
        from .hooks import HookExecutor

        return HookExecutor
    elif name == "fate_guarded":
        from .hooks import fate_guarded

        return fate_guarded
    elif name == "SessionStateMachine":
        from .session_dag import SessionStateMachine

        return SessionStateMachine
    elif name == "MerkleDAG":
        from .session_dag import MerkleDAG

        return MerkleDAG
    elif name == "create_session":
        from .session_dag import create_session

        return create_session
    elif name == "CognitiveBudgetAllocator":
        from .cognitive_budget import CognitiveBudgetAllocator

        return CognitiveBudgetAllocator
    elif name == "BudgetTracker":
        from .cognitive_budget import BudgetTracker

        return BudgetTracker
    elif name == "allocate_budget":
        from .cognitive_budget import allocate_budget

        return allocate_budget
    elif name == "TaskType":
        from .cognitive_budget import TaskType

        return TaskType
    elif name == "ComputeMarket":
        from .compute_market import ComputeMarket

        return ComputeMarket
    elif name == "create_market":
        from .compute_market import create_market

        return create_market
    elif name == "create_inference_license":
        from .compute_market import create_inference_license

        return create_inference_license
    raise AttributeError(f"module 'core.elite' has no attribute '{name}'")


__all__ = [
    # Constants
    "ELITE_VERSION",
    "PMBOK_KNOWLEDGE_AREAS",
    "IHSAN_DIMENSIONS",
    "SAPE_LAYERS",
    "SNR_TARGETS",
    # Quality Gates
    "QualityGate",
    "ElitePipeline",
    "MetricsDashboard",
    "RiskManager",
    "SAPEOptimizer",
    "IhsanValidator",
    # v1.1.0: Hook-First Governance (FATE)
    "FATEGate",
    "HookRegistry",
    "HookExecutor",
    "fate_guarded",
    # v1.1.0: Session as State Machine (Merkle-DAG)
    "SessionStateMachine",
    "MerkleDAG",
    "create_session",
    # v1.1.0: Cognitive Budget (7-3-6-9)
    "CognitiveBudgetAllocator",
    "BudgetTracker",
    "allocate_budget",
    "TaskType",
    # v1.1.0: Compute Market (Harberger + Gini)
    "ComputeMarket",
    "create_market",
    "create_inference_license",
]
