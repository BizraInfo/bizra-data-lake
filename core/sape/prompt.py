"""
SAPE v2.0 Prompt Compiler — System Prompt and User Prompt Generation

Upgrades SAPE_SYSTEM_PROMPT_V1 to V2 with:
- Execution mode awareness (Lite/Standard/Deep)
- Explicit evidence classification (VERIFIED / GROUNDED_INFERENCE / CONJECTURE / UNKNOWN)
- No hidden assumption, no skipped proof obligation, no decorative complexity
- Core ethos: Ihsān is the non-negotiable constraint

Created: 2026-04-10 | BIZRA SAPE v2.0
"""

from __future__ import annotations

import hashlib
import textwrap
from typing import Any, Dict, List, Optional, Sequence

from .types import (
    ALL_CHECKS,
    MODE_MODULES,
    MODE_PASSES,
    MODE_PROBES,
    IntentSlots,
)

# ═══════════════════════════════════════════════════════════════
# SAPE v2.0 System Prompt
# ═══════════════════════════════════════════════════════════════

SAPE_SYSTEM_PROMPT_V2 = textwrap.dedent("""
    You are SAPE v2.0 — Synaptic Activation Prompt Engine (BIZRA Peak Edition).

    Core Ethos:
      Ihsān is the non-negotiable constraint.
      Do not present speculation as fact.
      Distinguish VERIFIED evidence, GROUNDED_INFERENCE, CONJECTURE, and UNKNOWN explicitly.
      No hidden assumption. No skipped proof obligation. No decorative complexity.

    Operating Frame — 7–3–6–9 DNA:
      7 Modules: Intent Gate, Cognitive Lenses, Knowledge Kernels, Rare-Path Prober,
                 Symbolic Harness, Abstraction Elevator, Tension Studio.
      3 Passes: Diverge → Converge → Prove or Bound.
      6 Checks: Correctness, Consistency, Completeness, Causality, Ethics (Ihsān), Evidence.
      9 Probes: Counterfactual, Boundary, Analogical, Formalization, Program Sketch,
                Compression, Expansion, Adversarial, Ethical Overlay.

    Execution Modes:
      Lite     — Low-stakes: Intent Gate + Lenses + Converge + 6 Checks.
      Standard — Medium-stakes: Lite + Rare-Path + Symbolic + Selection Gate.
      Deep     — High-stakes: Full 7-3-6-9 SAPE.

    Mode Rule: Choose the lightest mode that preserves correctness.

    Output Requirements:
      - Tag every claim with its evidence level: [VERIFIED], [GROUNDED_INFERENCE], [CONJECTURE], [UNKNOWN].
      - Provide concise justifications, proof obligations, and test/verification steps.
      - Output BLOCKED sections when critical constraints or evidence are missing.
      - Do NOT reveal hidden chain-of-thought. Show reasoning, not process.
""").strip()


# ═══════════════════════════════════════════════════════════════
# Compile Functions
# ═══════════════════════════════════════════════════════════════


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _format_list(items: Sequence[str]) -> str:
    clean = [i.strip() for i in items if isinstance(i, str) and i.strip()]
    return ", ".join(clean) if clean else "(none)"


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "(no evidence kernels provided — output bounded to CONJECTURE)"
    lines: List[str] = []
    for item in evidence:
        label = str(item.get("label") or item.get("filename") or "artifact")
        level = str(item.get("evidence_level", "UNKNOWN"))
        source = str(item.get("source") or "")
        suffix = f" | source={source}" if source else ""
        lines.append(f"- [{level}] {label}{suffix}")
    return "\n".join(lines)


def compile_sape_v2_prompt(
    intent: IntentSlots,
    *,
    lenses: Optional[List[str]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    n_moves: int = 5,
    extra_instructions: str = "",
) -> Dict[str, str]:
    """
    Compile a SAPE v2.0 prompt pair (system + user) from intent slots.

    Returns a dict with keys:
      - system_prompt: The SAPE v2.0 system prompt
      - user_prompt: The structured user prompt
      - prompt_sha256: SHA-256 of the combined prompts
      - mode: The resolved execution mode
    """
    mode = intent.execution_mode
    active_modules = MODE_MODULES[mode]
    active_passes = MODE_PASSES[mode]
    active_probes = MODE_PROBES[mode]

    chosen_lenses = (lenses or ["Systems Architect", "Pragmatic Engineer", "Ethicist"])[
        :3
    ]

    sections = ["/SAPE-v2-Activate", ""]

    # Intent Gate section (always present)
    sections.append(textwrap.dedent(f"""\
        [Intent Gate]
        Domain: {intent.domain}
        Objective: {intent.objective}
        Stakes: {intent.stakes} → Mode: {mode.value.upper()}
        Constraints: {intent.constraints or '(none)'}
        Success Criteria: {intent.success_criteria or '(none)'}
        Forbidden: {_format_list(intent.forbidden_moves)}"""))

    # Cognitive Lenses (always present)
    sections.append(f"\n[Cognitive Lenses]\nApply: {_format_list(chosen_lenses)}")

    # Knowledge Kernels (Standard + Deep)
    if any(m.value == "knowledge_kernels" for m in active_modules):
        sections.append(textwrap.dedent(f"""\

            [Knowledge Kernels]
            Sources: {_format_list(intent.sources_allowed)}
            Evidence:
            {_format_evidence(evidence or [])}"""))

    # Rare-Path Prober (Standard + Deep)
    if any(m.value == "rare_path_prober" for m in active_modules):
        sections.append(textwrap.dedent(f"""\

            [Rare-Path Prober]
            Produce I-Path, C-Path, O-Path with N={n_moves} moves per path.
            For C/O, include ≥3 rarity moves (R1..R3) and justify divergence."""))

    # Symbolic Harness (Standard + Deep)
    if any(m.value == "symbolic_harness" for m in active_modules):
        sections.append(textwrap.dedent("""\

            [Symbolic Harness]
            Include: types/state/events, invariants, rules, proof sketch,
            program sketch (pre/postconditions), test oracles."""))

    # Abstraction Elevator (Deep only)
    if any(m.value == "abstraction_elevator" for m in active_modules):
        sections.append(textwrap.dedent("""\

            [Abstraction Elevator]
            Provide Micro/Meso/Macro + Meta-Reflection."""))

    # Tension Studio (Deep only)
    if any(m.value == "tension_studio" for m in active_modules):
        sections.append(textwrap.dedent("""\

            [Tension Studio]
            Generator ↔ Critic ↔ Synthesizer.
            Include: Constraint Clash (2–3 Pareto points), Adversarial Flip,
            Narrative Reframe (exec vs engineer)."""))

    # Execution passes
    pass_names = [p.value.replace("_", " ").title() for p in active_passes]
    sections.append(f"\n[Execution Passes]\n{' → '.join(pass_names)}")

    # Checks (always all 6)
    check_names = [c.value.replace("_", " ").title() for c in ALL_CHECKS]
    sections.append(f"\n[Checks]\nRun all: {', '.join(check_names)}")

    # Probes (mode-dependent)
    if active_probes:
        probe_names = [p.value.replace("_", " ").title() for p in active_probes]
        sections.append(f"\n[Probes]\nRun: {', '.join(probe_names)}")
    else:
        sections.append("\n[Probes]\n(none — Lite mode)")

    if extra_instructions:
        sections.append(f"\n[Extra Instructions]\n{extra_instructions}")

    user_prompt = "\n".join(sections)
    combined = SAPE_SYSTEM_PROMPT_V2 + "\n" + user_prompt
    prompt_sha = _sha256(combined)

    return {
        "system_prompt": SAPE_SYSTEM_PROMPT_V2,
        "user_prompt": user_prompt,
        "prompt_sha256": prompt_sha,
        "mode": mode.value,
    }
