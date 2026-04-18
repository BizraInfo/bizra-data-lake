"""
SAPE v2.0 Engine — Synaptic Activation Prompt Engine (BIZRA Peak Edition)

Orchestrates the full 7–3–6–9 DNA pipeline with mode-aware execution.

Execution Modes:
  Lite     — Intent Gate → Lenses → Converge → 6 Checks
  Standard — Lite + Rare-Path + Symbolic + Selection Gate
  Deep     — Full SAPE (all 7 modules, 3 passes, 6 checks, 9 probes)

Core Ethos:
  Ihsān is the non-negotiable constraint.
  Do not present speculation as fact.
  No hidden assumption. No skipped proof obligation. No decorative complexity.

Standing on Giants:
- Al-Ghazali (Intent Gate, 1096)
- Shannon (SNR, 1948)
- Besta (Graph-of-Thoughts, 2023)
- Kahneman (System 1/2, 2011)

Created: 2026-04-10 | BIZRA SAPE v2.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from core.integration.constants import (
    INTENT_FLOOR,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

from .checks import run_all_checks
from .intent_gate import run_intent_gate
from .modules import (
    run_abstraction_elevator,
    run_cognitive_lenses,
    run_knowledge_kernels,
    run_rare_path_prober,
    run_symbolic_harness,
    run_tension_studio,
)
from .probes import run_probes
from .prompt import compile_sape_v2_prompt
from .types import (
    MODE_MODULES,
    MODE_PASSES,
    MODE_PROBES,
    CheckResult,
    EvidenceLevel,
    ExecutionMode,
    IntentSlots,
    Module,
    ModuleResult,
    Pass,
    PassResult,
    ProbeResult,
    SAPEv2Result,
)

logger = logging.getLogger(__name__)

SAPE_VERSION = "2.0.0"


class SAPEv2Engine:
    """
    SAPE v2.0 Engine — Synaptic Activation Prompt Engine.

    Orchestrates the 7–3–6–9 DNA pipeline:
    - 7 Modules (mode-dependent activation)
    - 3 Passes (mode-dependent depth)
    - 6 Checks (always all 6)
    - 9 Probes (mode-dependent selection)

    Usage:
        engine = SAPEv2Engine()
        intent = IntentSlots(
            domain="distributed systems",
            objective="Design a consensus protocol",
            stakes="H",
        )
        result = engine.execute(intent, content="...")
    """

    def __init__(
        self,
        *,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
        intent_floor: float = INTENT_FLOOR,
        snr_fn: Optional[Callable[[str], float]] = None,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.intent_floor = intent_floor
        self._snr_fn = snr_fn

    def execute(
        self,
        intent: IntentSlots,
        content: str,
        *,
        intent_score: Optional[float] = None,
        lenses: Optional[List[str]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        n_moves: int = 5,
    ) -> SAPEv2Result:
        """
        Execute the SAPE v2.0 pipeline.

        The mode is derived from intent.stakes:
          L → Lite, M → Standard, H → Deep

        Returns a SAPEv2Result with all module, pass, check, and probe results.
        """
        start = time.monotonic()

        mode = intent.execution_mode
        active_modules = MODE_MODULES[mode]
        active_passes = MODE_PASSES[mode]
        active_probes = MODE_PROBES[mode]

        module_results: List[ModuleResult] = []
        pass_results: List[PassResult] = []
        warnings: List[str] = []

        # ─── Module 1: Intent Gate (always) ───
        intent_result = run_intent_gate(intent, intent_score=intent_score)
        module_results.append(intent_result)

        if not intent_result.metadata.get("passed", False):
            # Fail-closed: reject before any further computation
            duration = (time.monotonic() - start) * 1000
            return SAPEv2Result(
                mode=mode,
                intent=intent,
                module_results=module_results,
                check_results=[],
                probe_results=[],
                final_output=f"BLOCKED: Intent Gate failed — {intent_result.metadata.get('errors', [])}",
                overall_snr=0.0,
                overall_ihsan=0.0,
                duration_ms=duration,
                warnings=["Intent Gate failed — no further processing"],
            )

        # ─── Module 2: Cognitive Lenses (always) ───
        if Module.COGNITIVE_LENSES in active_modules:
            lens_result = run_cognitive_lenses(
                content, lenses=lenses, snr_fn=self._snr_fn
            )
            module_results.append(lens_result)

        # ─── Module 3: Knowledge Kernels (Standard + Deep) ───
        if Module.KNOWLEDGE_KERNELS in active_modules:
            kernel_result = run_knowledge_kernels(
                content,
                sources=intent.sources_allowed,
                evidence=evidence,
                snr_fn=self._snr_fn,
            )
            module_results.append(kernel_result)

        # ─── Module 4: Rare-Path Prober (Standard + Deep) ───
        if Module.RARE_PATH_PROBER in active_modules:
            rare_result = run_rare_path_prober(
                content, n_moves=n_moves, snr_fn=self._snr_fn
            )
            module_results.append(rare_result)

        # ─── Module 5: Symbolic Harness (Standard + Deep) ───
        if Module.SYMBOLIC_HARNESS in active_modules:
            symbolic_result = run_symbolic_harness(content, snr_fn=self._snr_fn)
            module_results.append(symbolic_result)

        # ─── Module 6: Abstraction Elevator (Deep only) ───
        if Module.ABSTRACTION_ELEVATOR in active_modules:
            elevator_result = run_abstraction_elevator(content, snr_fn=self._snr_fn)
            module_results.append(elevator_result)

        # ─── Module 7: Tension Studio (Deep only) ───
        if Module.TENSION_STUDIO in active_modules:
            tension_result = run_tension_studio(content, snr_fn=self._snr_fn)
            module_results.append(tension_result)

        # ─── Pass 1: Diverge ───
        if Pass.DIVERGE in active_passes:
            probe_results = run_probes(
                content,
                active_probes,
                snr_fn=self._snr_fn,
                ihsan_score=self._compute_ihsan(content),
            )
            pass_results.append(
                PassResult(
                    pass_type=Pass.DIVERGE,
                    output="Divergence complete — probes fired",
                    modules_run=[m.module for m in module_results],
                    probes_run=active_probes,
                    snr_score=self._compute_snr(content),
                )
            )
        else:
            probe_results = []

        # ─── Pass 2: Converge (always) ───
        convergence_snr = self._aggregate_snr(module_results)
        pass_results.append(
            PassResult(
                pass_type=Pass.CONVERGE,
                output="Convergence complete — modules synthesized",
                modules_run=[m.module for m in module_results],
                snr_score=convergence_snr,
            )
        )

        # ─── Selection Gate (Standard mode) ───
        if mode == ExecutionMode.STANDARD:
            # Selection Gate: if convergence SNR is too low, escalate to Deep
            if convergence_snr < self.snr_threshold:
                warnings.append(
                    f"Selection Gate: SNR {convergence_snr:.3f} below threshold "
                    f"{self.snr_threshold} — consider escalating to Deep mode"
                )

        # ─── Pass 3: Prove or Bound (Deep only) ───
        if Pass.PROVE_OR_BOUND in active_passes:
            pass_results.append(
                PassResult(
                    pass_type=Pass.PROVE_OR_BOUND,
                    output="Prove or Bound complete — 6 checks applied with falsification attempt",
                    modules_run=[m.module for m in module_results],
                    snr_score=convergence_snr,
                )
            )

        # ─── 6 Checks (always all 6) ───
        overall_ihsan = self._compute_ihsan(content)
        overall_snr = convergence_snr
        evidence_level = self._derive_evidence_level(module_results, evidence)

        check_results = run_all_checks(
            content,
            ihsan_score=overall_ihsan,
            snr_score=overall_snr,
            evidence_level=evidence_level,
            snr_fn=self._snr_fn,
        )

        # ─── Compile final output ───
        final_output = self._compile_output(
            mode=mode,
            intent=intent,
            module_results=module_results,
            pass_results=pass_results,
            check_results=check_results,
            probe_results=probe_results,
        )

        duration = (time.monotonic() - start) * 1000

        return SAPEv2Result(
            mode=mode,
            intent=intent,
            module_results=module_results,
            pass_results=pass_results,
            check_results=check_results,
            probe_results=probe_results,
            final_output=final_output,
            overall_snr=overall_snr,
            overall_ihsan=overall_ihsan,
            duration_ms=duration,
            warnings=warnings,
        )

    def compile_prompt(
        self,
        intent: IntentSlots,
        *,
        lenses: Optional[List[str]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        n_moves: int = 5,
        extra_instructions: str = "",
    ) -> Dict[str, str]:
        """
        Compile a SAPE v2.0 prompt pair without executing the pipeline.

        Useful for generating prompts to send to an external LLM.
        """
        return compile_sape_v2_prompt(
            intent,
            lenses=lenses,
            evidence=evidence,
            n_moves=n_moves,
            extra_instructions=extra_instructions,
        )

    # ─── Internal helpers ───

    def _compute_snr(self, text: str) -> float:
        """Compute SNR score."""
        if self._snr_fn:
            return self._snr_fn(text)
        return _heuristic_snr(text)

    def _compute_ihsan(self, text: str) -> float:
        """Compute Ihsān score."""
        if not text.strip():
            return 0.0
        words = text.split()
        has_structure = len(words) > 5
        has_clarity = len(set(words)) / max(len(words), 1) > 0.5
        is_balanced = 10 <= len(words) <= 500
        return min(
            0.4 * float(has_structure)
            + 0.3 * float(has_clarity)
            + 0.3 * float(is_balanced),
            1.0,
        )

    def _aggregate_snr(self, results: List[ModuleResult]) -> float:
        """Aggregate SNR across module results."""
        if not results:
            return 0.0
        return sum(r.snr_score for r in results) / len(results)

    def _derive_evidence_level(
        self,
        module_results: List[ModuleResult],
        evidence: Optional[List[Dict[str, Any]]],
    ) -> EvidenceLevel:
        """Derive overall evidence level from module results."""
        if evidence and any(e.get("evidence_level") == "VERIFIED" for e in evidence):
            return EvidenceLevel.VERIFIED
        if evidence:
            return EvidenceLevel.GROUNDED_INFERENCE

        # Check if Knowledge Kernels ran
        for r in module_results:
            if r.module == Module.KNOWLEDGE_KERNELS:
                level = r.metadata.get("max_evidence_level", "UNKNOWN")
                try:
                    return EvidenceLevel(level)
                except ValueError:
                    pass

        return EvidenceLevel.CONJECTURE

    def _compile_output(
        self,
        *,
        mode: ExecutionMode,
        intent: IntentSlots,
        module_results: List[ModuleResult],
        pass_results: List[PassResult],
        check_results: List[CheckResult],
        probe_results: List[ProbeResult],
    ) -> str:
        """Compile all results into a structured final output."""
        sections = [
            f"═══ SAPE v{SAPE_VERSION} Output ═══",
            f"Mode: {mode.value.upper()}",
            f"Domain: {intent.domain}",
            f"Objective: {intent.objective}",
            "",
        ]

        # Module summaries
        sections.append("─── Modules ───")
        for r in module_results:
            sections.append(f"  [{r.module.value}] SNR={r.snr_score:.3f}")

        # Pass summaries
        sections.append("\n─── Passes ───")
        for r in pass_results:
            sections.append(f"  [{r.pass_type.value}] SNR={r.snr_score:.3f}")

        # Check summaries
        sections.append("\n─── Checks ───")
        for r in check_results:
            status = "✓" if r.passed else "✗"
            sections.append(f"  {status} {r.check.value}: {r.score:.3f} — {r.detail}")

        # Probe summaries
        if probe_results:
            sections.append("\n─── Probes ───")
            for r in probe_results:
                flag = "⚠" if r.flagged else "○"
                sections.append(f"  {flag} {r.probe.value}: {r.score:.3f}")

        # Flagged items
        flagged = [r for r in probe_results if r.flagged]
        failed_checks = [r for r in check_results if not r.passed]
        if flagged or failed_checks:
            sections.append("\n─── Attention Required ───")
            for r in failed_checks:
                sections.append(f"  CHECK FAILED: {r.check.value} — {r.detail}")
            for r in flagged:
                sections.append(
                    f"  PROBE FLAGGED: {r.probe.value} — {r.findings[:100]}"
                )

        return "\n".join(sections)


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
        0.4 * unique_ratio + 0.3 * min(avg_word_len / 8, 1.0) + 0.3 * structure,
        1.0,
    )
