"""
SAPE v2.0 — Synaptic Activation Prompt Engine (BIZRA Peak Edition)

Purpose: Activate high-value reasoning capacity through a disciplined,
multi-layer prompt architecture that expands search, bridges neural
intuition with symbolic rigor, and converges toward verifiable, bounded,
ethically aligned output.

Core Ethos:
  Ihsān is the non-negotiable constraint.
  Do not present speculation as fact.
  Distinguish verified evidence, grounded inference, conjecture, and unknowns explicitly.
  No hidden assumption. No skipped proof obligation. No decorative complexity.

Operating Frame — 7–3–6–9 DNA:
  7 Modules: Intent Gate, Cognitive Lenses, Knowledge Kernels, Rare-Path Prober,
             Symbolic Harness, Abstraction Elevator, Tension Studio
  3 Passes:  Diverge, Converge, Prove or Bound
  6 Checks:  Correctness, Consistency, Completeness, Causality, Ethics, Evidence
  9 Probes:  Counterfactual, Boundary, Analogical, Formalization, Program Sketch,
             Compression, Expansion, Adversarial, Ethical Overlay

Execution Modes:
  Lite     — Low-stakes: Intent Gate + Lenses + Converge + 6 Checks
  Standard — Medium-stakes: Lite + Rare-Path + Symbolic + Selection Gate
  Deep     — High-stakes: Full 7-3-6-9 SAPE

Created: 2026-04-10 | BIZRA SAPE v2.0.0
"""

from .engine import SAPE_VERSION, SAPEv2Engine
from .prompt import SAPE_SYSTEM_PROMPT_V2, compile_sape_v2_prompt
from .types import (
    ALL_CHECKS,
    Check,
    CheckResult,
    EvidenceLevel,
    ExecutionMode,
    IntentSlots,
    MODE_MODULES,
    MODE_PASSES,
    MODE_PROBES,
    Module,
    ModuleResult,
    Pass,
    PassResult,
    Probe,
    ProbeResult,
    SAPEv2Result,
    STAKES_TO_MODE,
    Stakes,
)

__all__ = [
    # Engine
    "SAPE_VERSION",
    "SAPEv2Engine",
    # Prompt
    "SAPE_SYSTEM_PROMPT_V2",
    "compile_sape_v2_prompt",
    # Types
    "ALL_CHECKS",
    "Check",
    "CheckResult",
    "EvidenceLevel",
    "ExecutionMode",
    "IntentSlots",
    "MODE_MODULES",
    "MODE_PASSES",
    "MODE_PROBES",
    "Module",
    "ModuleResult",
    "Pass",
    "PassResult",
    "Probe",
    "ProbeResult",
    "SAPEv2Result",
    "STAKES_TO_MODE",
    "Stakes",
]
