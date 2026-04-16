"""
SAPE v2.0 Modules — The 7-Module Pipeline

Module 1: Intent Gate — Al-Ghazali pre-gate (see intent_gate.py)
Module 2: Cognitive Lenses — Multi-perspective analysis
Module 3: Knowledge Kernels — Evidence retrieval and grounding
Module 4: Rare-Path Prober — Unconventional pattern discovery
Module 5: Symbolic Harness — Types, invariants, proof sketches
Module 6: Abstraction Elevator — Micro/Meso/Macro + Meta-Reflection
Module 7: Tension Studio — Generator ↔ Critic ↔ Synthesizer

Created: 2026-04-10 | BIZRA SAPE v2.0
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .types import (
    EvidenceLevel,
    Module,
    ModuleResult,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Canonical Lenses (7 perspectives)
# ═══════════════════════════════════════════════════════════════

CANONICAL_LENSES: List[str] = [
    "Systems Architect",
    "Formal Theorist",
    "Pragmatic Engineer",
    "Ethicist",
    "Poet/Designer",
    "Historian",
    "Futurist",
]


# ═══════════════════════════════════════════════════════════════
# Module 2: Cognitive Lenses
# ═══════════════════════════════════════════════════════════════


def run_cognitive_lenses(
    content: str,
    lenses: Optional[List[str]] = None,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> ModuleResult:
    """
    Apply 2–3 cognitive lenses to the content.

    Each lens examines the problem from a different professional perspective.
    This module generates multi-perspective analysis that feeds divergence.
    """
    chosen = lenses[:3] if lenses else CANONICAL_LENSES[:3]
    non_canonical = [l for l in chosen if l not in CANONICAL_LENSES]

    lines = [f"[Cognitive Lenses: {', '.join(chosen)}]"]
    for lens in chosen:
        lines.append(f"\n--- {lens} ---")
        lines.append(f"Perspective on: {content[:200]}...")
        lines.append("(Analysis placeholder — populated by LLM or reasoning engine)")

    snr = snr_fn(content) if snr_fn else _heuristic_snr(content)

    warnings: Dict[str, Any] = {}
    if non_canonical:
        warnings["non_canonical_lenses"] = non_canonical

    return ModuleResult(
        module=Module.COGNITIVE_LENSES,
        output="\n".join(lines),
        snr_score=snr,
        ihsan_score=min(snr + 0.05, 1.0),
        metadata={
            "lenses_applied": chosen,
            "non_canonical": non_canonical,
        },
    )


# ═══════════════════════════════════════════════════════════════
# Module 3: Knowledge Kernels
# ═══════════════════════════════════════════════════════════════


def run_knowledge_kernels(
    content: str,
    sources: Optional[List[str]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> ModuleResult:
    """
    Retrieve and bind evidence kernels to the reasoning chain.

    Every claim must be tagged with its evidence level:
    VERIFIED, GROUNDED_INFERENCE, CONJECTURE, or UNKNOWN.
    """
    evidence = evidence or []
    sources = sources or []

    lines = ["[Knowledge Kernels]"]
    if sources:
        lines.append(f"Sources allowed: {', '.join(sources)}")
    if evidence:
        for item in evidence:
            label = item.get("label", item.get("filename", "artifact"))
            level = item.get("evidence_level", "UNKNOWN")
            lines.append(f"- [{level}] {label}")
    else:
        lines.append("(No evidence kernels provided — output bounded to CONJECTURE)")

    snr = snr_fn(content) if snr_fn else _heuristic_snr(content)

    max_evidence_level = EvidenceLevel.UNKNOWN
    if evidence:
        max_evidence_level = EvidenceLevel.GROUNDED_INFERENCE
        if any(e.get("evidence_level") == "VERIFIED" for e in evidence):
            max_evidence_level = EvidenceLevel.VERIFIED

    return ModuleResult(
        module=Module.KNOWLEDGE_KERNELS,
        output="\n".join(lines),
        snr_score=snr,
        ihsan_score=snr,
        metadata={
            "source_count": len(sources),
            "evidence_count": len(evidence),
            "max_evidence_level": max_evidence_level.value,
        },
    )


# ═══════════════════════════════════════════════════════════════
# Module 4: Rare-Path Prober
# ═══════════════════════════════════════════════════════════════


def run_rare_path_prober(
    content: str,
    n_moves: int = 5,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> ModuleResult:
    """
    Probe rarely-fired reasoning paths.

    Generates three paths:
    - I-Path (Incremental): Safe, step-by-step progression
    - C-Path (Creative): Lateral thinking with ≥3 rarity moves
    - O-Path (Oppositional): Adversarial counterexample path

    Each path produces N moves. C/O paths must justify divergence.
    """
    n_moves = max(3, min(n_moves, 9))

    lines = [
        "[Rare-Path Prober]",
        f"Moves per path: {n_moves}",
        "",
        "I-Path (Incremental):",
    ]
    for i in range(1, n_moves + 1):
        lines.append(f"  I{i}. (incremental move placeholder)")

    lines.append("\nC-Path (Creative):")
    for i in range(1, n_moves + 1):
        prefix = f"R{i}" if i <= 3 else f"C{i}"
        lines.append(f"  {prefix}. (creative/rarity move placeholder)")
    lines.append("  Divergence justification: (to be filled)")

    lines.append("\nO-Path (Oppositional):")
    for i in range(1, n_moves + 1):
        prefix = f"R{i}" if i <= 3 else f"O{i}"
        lines.append(f"  {prefix}. (oppositional move placeholder)")
    lines.append("  Divergence justification: (to be filled)")

    snr = snr_fn(content) if snr_fn else _heuristic_snr(content)

    return ModuleResult(
        module=Module.RARE_PATH_PROBER,
        output="\n".join(lines),
        snr_score=snr,
        ihsan_score=snr,
        metadata={
            "n_moves": n_moves,
            "paths": ["I-Path", "C-Path", "O-Path"],
        },
    )


# ═══════════════════════════════════════════════════════════════
# Module 5: Symbolic Harness
# ═══════════════════════════════════════════════════════════════


def run_symbolic_harness(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> ModuleResult:
    """
    Bind the reasoning to symbolic rigor.

    Produces:
    - Types / State / Events definitions
    - Invariants (must-hold properties)
    - Rules (transition constraints)
    - Proof sketch (obligation → approach)
    - Program sketch (pre/postconditions)
    - Test oracles (falsifiable predictions)
    """
    lines = [
        "[Symbolic Harness]",
        "",
        "Types / State / Events:",
        "  (to be defined from domain analysis)",
        "",
        "Invariants:",
        "  INV-1: (property that must always hold)",
        "",
        "Rules:",
        "  RULE-1: (state transition constraint)",
        "",
        "Proof Sketch:",
        "  Obligation: (what must be proven)",
        "  Approach: (how to prove it)",
        "",
        "Program Sketch:",
        "  Pre: (preconditions)",
        "  Post: (postconditions)",
        "  Body: (algorithm sketch)",
        "",
        "Test Oracles:",
        "  ORACLE-1: (falsifiable prediction from invariants)",
    ]

    snr = snr_fn(content) if snr_fn else _heuristic_snr(content)

    return ModuleResult(
        module=Module.SYMBOLIC_HARNESS,
        output="\n".join(lines),
        snr_score=snr,
        ihsan_score=snr,
        metadata={
            "sections": [
                "types_state_events",
                "invariants",
                "rules",
                "proof_sketch",
                "program_sketch",
                "test_oracles",
            ],
        },
    )


# ═══════════════════════════════════════════════════════════════
# Module 6: Abstraction Elevator
# ═══════════════════════════════════════════════════════════════


def run_abstraction_elevator(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> ModuleResult:
    """
    Elevate reasoning across abstraction layers.

    Produces:
    - Micro: Implementation-level detail
    - Meso: Module/component-level design
    - Macro: System/architecture-level view
    - Meta-Reflection: What the analysis itself reveals
    """
    lines = [
        "[Abstraction Elevator]",
        "",
        "Micro (Implementation):",
        "  (concrete implementation details)",
        "",
        "Meso (Component):",
        "  (module-level design and interfaces)",
        "",
        "Macro (System):",
        "  (architectural implications and constraints)",
        "",
        "Meta-Reflection:",
        "  (what this multi-level analysis reveals about the problem)",
    ]

    snr = snr_fn(content) if snr_fn else _heuristic_snr(content)

    return ModuleResult(
        module=Module.ABSTRACTION_ELEVATOR,
        output="\n".join(lines),
        snr_score=snr,
        ihsan_score=snr,
        metadata={
            "levels": ["micro", "meso", "macro", "meta"],
        },
    )


# ═══════════════════════════════════════════════════════════════
# Module 7: Tension Studio
# ═══════════════════════════════════════════════════════════════


def run_tension_studio(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> ModuleResult:
    """
    Generator ↔ Critic ↔ Synthesizer dialectic.

    Produces:
    - Generator: Initial proposal
    - Critic: Adversarial challenge
    - Synthesizer: Pareto-optimal resolution
    - Constraint Clash: 2–3 Pareto points
    - Adversarial Flip: Strongest counter-argument
    - Narrative Reframe: exec vs engineer perspective
    """
    lines = [
        "[Tension Studio]",
        "",
        "Generator:",
        "  (initial proposal based on prior modules)",
        "",
        "Critic:",
        "  (strongest objection to the proposal)",
        "",
        "Synthesizer:",
        "  (resolution that addresses the objection)",
        "",
        "Constraint Clash:",
        "  P1: (Pareto point 1)",
        "  P2: (Pareto point 2)",
        "  P3: (Pareto point 3, if applicable)",
        "",
        "Adversarial Flip:",
        "  (what would make this solution fail?)",
        "",
        "Narrative Reframe:",
        "  Executive view: (business/strategy framing)",
        "  Engineer view: (technical/implementation framing)",
    ]

    snr = snr_fn(content) if snr_fn else _heuristic_snr(content)

    return ModuleResult(
        module=Module.TENSION_STUDIO,
        output="\n".join(lines),
        snr_score=snr,
        ihsan_score=snr,
        metadata={
            "dialectic": ["generator", "critic", "synthesizer"],
            "pareto_points": 3,
        },
    )


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _heuristic_snr(text: str) -> float:
    """Heuristic SNR score based on text characteristics."""
    if not text.strip():
        return 0.0
    words = text.split()
    if len(words) < 3:
        return 0.5
    unique_ratio = len(set(words)) / len(words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    structure = min(text.count(".") / max(len(words) / 10, 1), 1.0)
    return min(
        0.4 * unique_ratio + 0.3 * min(avg_word_len / 8, 1.0) + 0.3 * structure, 1.0
    )
