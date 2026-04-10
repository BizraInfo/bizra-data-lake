"""
SAPE v2.0 Type Definitions — 7–3–6–9 DNA Architecture

Canonical type system for the Synaptic Activation Prompt Engine.

7 Modules × 3 Passes × 6 Checks × 9 Probes

Standing on Giants:
- Besta (Graph-of-Thoughts)
- Shannon (Signal-to-Noise Ratio)
- Al-Ghazali (Intent Gate / Ihsān)
- Kahneman (System 1/2 cognitive load)

Created: 2026-04-10 | BIZRA SAPE v2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal


# ═══════════════════════════════════════════════════════════════
# Execution Modes — Choose the lightest that preserves correctness
# ═══════════════════════════════════════════════════════════════


class ExecutionMode(str, Enum):
    """SAPE execution modes — lightest first."""

    LITE = "lite"  # Low stakes: Intent Gate → Lenses → Converge → 6 Checks
    STANDARD = "standard"  # Medium stakes: Lite + Rare-Path + Symbolic + Selection Gate
    DEEP = "deep"  # High stakes: Full SAPE 7-3-6-9


# ═══════════════════════════════════════════════════════════════
# 7 Modules
# ═══════════════════════════════════════════════════════════════


class Module(str, Enum):
    """The 7 SAPE modules."""

    INTENT_GATE = "intent_gate"
    COGNITIVE_LENSES = "cognitive_lenses"
    KNOWLEDGE_KERNELS = "knowledge_kernels"
    RARE_PATH_PROBER = "rare_path_prober"
    SYMBOLIC_HARNESS = "symbolic_harness"
    ABSTRACTION_ELEVATOR = "abstraction_elevator"
    TENSION_STUDIO = "tension_studio"


# ═══════════════════════════════════════════════════════════════
# 3 Passes
# ═══════════════════════════════════════════════════════════════


class Pass(str, Enum):
    """The 3 SAPE passes."""

    DIVERGE = "diverge"
    CONVERGE = "converge"
    PROVE_OR_BOUND = "prove_or_bound"


# ═══════════════════════════════════════════════════════════════
# 6 Checks
# ═══════════════════════════════════════════════════════════════


class Check(str, Enum):
    """The 6 SAPE checks."""

    CORRECTNESS = "correctness"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    CAUSALITY = "causality"
    ETHICS = "ethics"  # Ihsān
    EVIDENCE = "evidence"


# ═══════════════════════════════════════════════════════════════
# 9 Probes
# ═══════════════════════════════════════════════════════════════


class Probe(str, Enum):
    """The 9 SAPE probes."""

    COUNTERFACTUAL = "counterfactual"
    BOUNDARY = "boundary"
    ANALOGICAL = "analogical"
    FORMALIZATION = "formalization"
    PROGRAM_SKETCH = "program_sketch"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    ADVERSARIAL = "adversarial"
    ETHICAL_OVERLAY = "ethical_overlay"


# ═══════════════════════════════════════════════════════════════
# Stakes Mapping
# ═══════════════════════════════════════════════════════════════

Stakes = Literal["L", "M", "H"]

STAKES_TO_MODE: Dict[str, ExecutionMode] = {
    "L": ExecutionMode.LITE,
    "M": ExecutionMode.STANDARD,
    "H": ExecutionMode.DEEP,
}


# ═══════════════════════════════════════════════════════════════
# Mode → Pipeline Configuration
# ═══════════════════════════════════════════════════════════════

# Modules activated per mode
MODE_MODULES: Dict[ExecutionMode, List[Module]] = {
    ExecutionMode.LITE: [
        Module.INTENT_GATE,
        Module.COGNITIVE_LENSES,
    ],
    ExecutionMode.STANDARD: [
        Module.INTENT_GATE,
        Module.COGNITIVE_LENSES,
        Module.KNOWLEDGE_KERNELS,
        Module.RARE_PATH_PROBER,
        Module.SYMBOLIC_HARNESS,
    ],
    ExecutionMode.DEEP: [
        Module.INTENT_GATE,
        Module.COGNITIVE_LENSES,
        Module.KNOWLEDGE_KERNELS,
        Module.RARE_PATH_PROBER,
        Module.SYMBOLIC_HARNESS,
        Module.ABSTRACTION_ELEVATOR,
        Module.TENSION_STUDIO,
    ],
}

# Passes activated per mode
MODE_PASSES: Dict[ExecutionMode, List[Pass]] = {
    ExecutionMode.LITE: [
        Pass.CONVERGE,
    ],
    ExecutionMode.STANDARD: [
        Pass.DIVERGE,
        Pass.CONVERGE,
    ],
    ExecutionMode.DEEP: [
        Pass.DIVERGE,
        Pass.CONVERGE,
        Pass.PROVE_OR_BOUND,
    ],
}

# Probes activated per mode (Lite runs none, Standard runs targeted, Deep runs all)
MODE_PROBES: Dict[ExecutionMode, List[Probe]] = {
    ExecutionMode.LITE: [],
    ExecutionMode.STANDARD: [
        Probe.COUNTERFACTUAL,
        Probe.BOUNDARY,
        Probe.ADVERSARIAL,
    ],
    ExecutionMode.DEEP: list(Probe),  # All 9
}

# All modes run all 6 checks (non-negotiable)
ALL_CHECKS: List[Check] = list(Check)


# ═══════════════════════════════════════════════════════════════
# Evidence Classification
# ═══════════════════════════════════════════════════════════════


class EvidenceLevel(str, Enum):
    """Classification of evidence confidence."""

    VERIFIED = "VERIFIED"  # Proven, with source
    GROUNDED_INFERENCE = "GROUNDED_INFERENCE"  # Logically derived from verified facts
    CONJECTURE = "CONJECTURE"  # Plausible but unproven
    UNKNOWN = "UNKNOWN"  # No basis


# ═══════════════════════════════════════════════════════════════
# Intent Gate Slots
# ═══════════════════════════════════════════════════════════════


@dataclass
class IntentSlots:
    """Structured slots for the Intent Gate module."""

    domain: str = ""
    objective: str = ""
    stakes: Stakes = "M"
    constraints: str = ""
    success_criteria: str = ""
    forbidden_moves: List[str] = field(default_factory=lambda: [
        "hallucination",
        "hidden assumptions",
        "skipped proof obligation",
        "decorative complexity",
    ])
    sources_allowed: List[str] = field(default_factory=list)

    @property
    def execution_mode(self) -> ExecutionMode:
        """Derive execution mode from stakes level."""
        return STAKES_TO_MODE.get(self.stakes, ExecutionMode.STANDARD)


# ═══════════════════════════════════════════════════════════════
# Module Result Containers
# ═══════════════════════════════════════════════════════════════


@dataclass
class CheckResult:
    """Result of a single check."""

    check: Check
    passed: bool
    score: float = 0.0
    detail: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN


@dataclass
class ProbeResult:
    """Result of a single probe."""

    probe: Probe
    findings: str = ""
    score: float = 0.0
    flagged: bool = False


@dataclass
class ModuleResult:
    """Result of a single module execution."""

    module: Module
    output: str = ""
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PassResult:
    """Result of a single pass execution."""

    pass_type: Pass
    output: str = ""
    modules_run: List[Module] = field(default_factory=list)
    probes_run: List[Probe] = field(default_factory=list)
    snr_score: float = 0.0


@dataclass
class SAPEv2Result:
    """Complete result of a SAPE v2.0 execution."""

    mode: ExecutionMode
    intent: IntentSlots
    module_results: List[ModuleResult] = field(default_factory=list)
    pass_results: List[PassResult] = field(default_factory=list)
    check_results: List[CheckResult] = field(default_factory=list)
    probe_results: List[ProbeResult] = field(default_factory=list)
    final_output: str = ""
    overall_snr: float = 0.0
    overall_ihsan: float = 0.0
    duration_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def all_checks_passed(self) -> bool:
        return all(c.passed for c in self.check_results)

    @property
    def flagged_probes(self) -> List[ProbeResult]:
        return [p for p in self.probe_results if p.flagged]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "intent": {
                "domain": self.intent.domain,
                "objective": self.intent.objective,
                "stakes": self.intent.stakes,
            },
            "modules_run": [m.module.value for m in self.module_results],
            "passes_run": [p.pass_type.value for p in self.pass_results],
            "checks": {
                c.check.value: {"passed": c.passed, "score": c.score}
                for c in self.check_results
            },
            "probes_flagged": [p.probe.value for p in self.flagged_probes],
            "overall_snr": self.overall_snr,
            "overall_ihsan": self.overall_ihsan,
            "all_checks_passed": self.all_checks_passed,
            "duration_ms": self.duration_ms,
            "warnings": self.warnings,
        }
