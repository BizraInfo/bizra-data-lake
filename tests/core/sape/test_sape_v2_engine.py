"""
Tests for SAPE v2.0 — Synaptic Activation Prompt Engine.

Validates the 7–3–6–9 DNA architecture:
  - Intent Gate validation and mode resolution
  - Module pipeline (7 modules)
  - Pass execution (3 passes)
  - Check verification (6 checks)
  - Probe firing (9 probes)
  - Mode-aware pipeline (Lite/Standard/Deep)
  - Prompt compilation
"""

from __future__ import annotations

from core.sape import (
    ALL_CHECKS,
    Check,
    EvidenceLevel,
    ExecutionMode,
    IntentSlots,
    MODE_MODULES,
    MODE_PASSES,
    MODE_PROBES,
    Module,
    Pass,
    Probe,
    SAPE_SYSTEM_PROMPT_V2,
    SAPE_VERSION,
    SAPEv2Engine,
    STAKES_TO_MODE,
    compile_sape_v2_prompt,
)
from core.sape.checks import run_all_checks
from core.sape.intent_gate import run_intent_gate, validate_intent
from core.sape.modules import (
    CANONICAL_LENSES,
    run_abstraction_elevator,
    run_cognitive_lenses,
    run_knowledge_kernels,
    run_rare_path_prober,
    run_symbolic_harness,
    run_tension_studio,
)
from core.sape.probes import run_probes


# ═══════════════════════════════════════════════════════════════
# DNA Invariants (7-3-6-9)
# ═══════════════════════════════════════════════════════════════


class TestDNAInvariants:
    """Verify the 7-3-6-9 DNA counts are correct."""

    def test_7_modules(self):
        assert len(Module) == 7

    def test_3_passes(self):
        assert len(Pass) == 3

    def test_6_checks(self):
        assert len(Check) == 6
        assert len(ALL_CHECKS) == 6

    def test_9_probes(self):
        assert len(Probe) == 9

    def test_version_is_2(self):
        assert SAPE_VERSION.startswith("2.")


# ═══════════════════════════════════════════════════════════════
# Execution Mode Resolution
# ═══════════════════════════════════════════════════════════════


class TestExecutionModes:
    """Test mode selection from stakes."""

    def test_low_stakes_gives_lite(self):
        assert STAKES_TO_MODE["L"] == ExecutionMode.LITE

    def test_medium_stakes_gives_standard(self):
        assert STAKES_TO_MODE["M"] == ExecutionMode.STANDARD

    def test_high_stakes_gives_deep(self):
        assert STAKES_TO_MODE["H"] == ExecutionMode.DEEP

    def test_intent_slots_mode_derivation(self):
        intent = IntentSlots(domain="test", objective="test", stakes="H")
        assert intent.execution_mode == ExecutionMode.DEEP

    def test_lite_modules_subset(self):
        lite_modules = MODE_MODULES[ExecutionMode.LITE]
        assert Module.INTENT_GATE in lite_modules
        assert Module.COGNITIVE_LENSES in lite_modules
        assert Module.TENSION_STUDIO not in lite_modules

    def test_deep_has_all_modules(self):
        deep_modules = MODE_MODULES[ExecutionMode.DEEP]
        assert len(deep_modules) == 7

    def test_deep_has_all_passes(self):
        deep_passes = MODE_PASSES[ExecutionMode.DEEP]
        assert len(deep_passes) == 3

    def test_deep_has_all_probes(self):
        deep_probes = MODE_PROBES[ExecutionMode.DEEP]
        assert len(deep_probes) == 9

    def test_lite_has_no_probes(self):
        assert MODE_PROBES[ExecutionMode.LITE] == []


# ═══════════════════════════════════════════════════════════════
# Intent Gate
# ═══════════════════════════════════════════════════════════════


class TestIntentGate:
    """Test the Al-Ghazali intent pre-gate."""

    def test_valid_intent_passes(self):
        slots = IntentSlots(
            domain="distributed systems",
            objective="Design a consensus protocol",
            stakes="M",
        )
        passed, errors = validate_intent(slots)
        assert passed
        assert errors == []

    def test_empty_domain_fails(self):
        slots = IntentSlots(domain="", objective="test", stakes="M")
        passed, errors = validate_intent(slots)
        assert not passed
        assert any("domain" in e for e in errors)

    def test_empty_objective_fails(self):
        slots = IntentSlots(domain="test", objective="", stakes="M")
        passed, errors = validate_intent(slots)
        assert not passed
        assert any("objective" in e for e in errors)

    def test_high_stakes_requires_success_criteria(self):
        slots = IntentSlots(
            domain="safety-critical",
            objective="Design braking system",
            stakes="H",
        )
        passed, errors = validate_intent(slots)
        assert not passed
        assert any("success criteria" in e for e in errors)

    def test_high_stakes_with_criteria_passes(self):
        slots = IntentSlots(
            domain="safety-critical",
            objective="Design braking system",
            stakes="H",
            success_criteria="Stop within 3m at 60km/h",
            constraints="Must use redundant sensors",
        )
        passed, errors = validate_intent(slots)
        assert passed

    def test_intent_score_below_floor_fails(self):
        slots = IntentSlots(domain="test", objective="test", stakes="L")
        result = run_intent_gate(slots, intent_score=0.5)
        assert not result.metadata["passed"]

    def test_intent_gate_module_result(self):
        slots = IntentSlots(domain="test", objective="test", stakes="L")
        result = run_intent_gate(slots)
        assert result.module == Module.INTENT_GATE
        assert result.metadata["passed"]
        assert result.metadata["resolved_mode"] == "lite"


# ═══════════════════════════════════════════════════════════════
# Modules
# ═══════════════════════════════════════════════════════════════


class TestModules:
    """Test individual module execution."""

    CONTENT = "This is a well-structured test content with multiple sentences. " * 5

    def test_cognitive_lenses_default(self):
        result = run_cognitive_lenses(self.CONTENT)
        assert result.module == Module.COGNITIVE_LENSES
        assert result.snr_score > 0
        assert len(result.metadata["lenses_applied"]) <= 3

    def test_cognitive_lenses_custom(self):
        result = run_cognitive_lenses(self.CONTENT, lenses=["Formal Theorist"])
        assert "Formal Theorist" in result.metadata["lenses_applied"]

    def test_knowledge_kernels_no_evidence(self):
        result = run_knowledge_kernels(self.CONTENT)
        assert result.module == Module.KNOWLEDGE_KERNELS
        assert result.metadata["evidence_count"] == 0
        assert result.metadata["max_evidence_level"] == "UNKNOWN"

    def test_knowledge_kernels_with_evidence(self):
        evidence = [{"label": "RFC 9000", "evidence_level": "VERIFIED"}]
        result = run_knowledge_kernels(self.CONTENT, evidence=evidence)
        assert result.metadata["evidence_count"] == 1
        assert result.metadata["max_evidence_level"] == "VERIFIED"

    def test_rare_path_prober(self):
        result = run_rare_path_prober(self.CONTENT, n_moves=5)
        assert result.module == Module.RARE_PATH_PROBER
        assert result.metadata["n_moves"] == 5
        assert len(result.metadata["paths"]) == 3

    def test_rare_path_prober_clamps_moves(self):
        result = run_rare_path_prober(self.CONTENT, n_moves=1)
        assert result.metadata["n_moves"] == 3  # Clamped to min

    def test_symbolic_harness(self):
        result = run_symbolic_harness(self.CONTENT)
        assert result.module == Module.SYMBOLIC_HARNESS
        assert "invariants" in result.metadata["sections"]

    def test_abstraction_elevator(self):
        result = run_abstraction_elevator(self.CONTENT)
        assert result.module == Module.ABSTRACTION_ELEVATOR
        assert "micro" in result.metadata["levels"]
        assert "macro" in result.metadata["levels"]

    def test_tension_studio(self):
        result = run_tension_studio(self.CONTENT)
        assert result.module == Module.TENSION_STUDIO
        assert "generator" in result.metadata["dialectic"]

    def test_canonical_lenses_count(self):
        assert len(CANONICAL_LENSES) == 7


# ═══════════════════════════════════════════════════════════════
# Checks
# ═══════════════════════════════════════════════════════════════


class TestChecks:
    """Test the 6 non-negotiable checks."""

    CONTENT = "This is a well-structured test content. It has good clarity and reasoning."

    def test_all_checks_run(self):
        results = run_all_checks(
            self.CONTENT,
            ihsan_score=0.96,
            snr_score=0.90,
            evidence_level=EvidenceLevel.VERIFIED,
        )
        assert len(results) == 6
        checks_run = {r.check for r in results}
        assert checks_run == set(Check)

    def test_ethics_check_passes_above_threshold(self):
        results = run_all_checks(self.CONTENT, ihsan_score=0.96, snr_score=0.90)
        ethics = next(r for r in results if r.check == Check.ETHICS)
        assert ethics.passed

    def test_ethics_check_fails_below_threshold(self):
        results = run_all_checks(self.CONTENT, ihsan_score=0.50, snr_score=0.90)
        ethics = next(r for r in results if r.check == Check.ETHICS)
        assert not ethics.passed

    def test_evidence_check_passes_with_verified(self):
        results = run_all_checks(
            self.CONTENT, evidence_level=EvidenceLevel.VERIFIED
        )
        evidence = next(r for r in results if r.check == Check.EVIDENCE)
        assert evidence.passed

    def test_evidence_check_fails_with_unknown(self):
        results = run_all_checks(
            self.CONTENT, evidence_level=EvidenceLevel.UNKNOWN
        )
        evidence = next(r for r in results if r.check == Check.EVIDENCE)
        assert not evidence.passed


# ═══════════════════════════════════════════════════════════════
# Probes
# ═══════════════════════════════════════════════════════════════


class TestProbes:
    """Test the 9 divergence probes."""

    CONTENT = "This function has edge cases at the boundary. Proof by invariant theorem."

    def test_run_all_probes(self):
        results = run_probes(self.CONTENT, list(Probe))
        assert len(results) == 9

    def test_run_subset(self):
        results = run_probes(
            self.CONTENT, [Probe.COUNTERFACTUAL, Probe.BOUNDARY]
        )
        assert len(results) == 2

    def test_boundary_probe_detects_awareness(self):
        results = run_probes(self.CONTENT, [Probe.BOUNDARY])
        assert results[0].probe == Probe.BOUNDARY
        # Content contains "edge case" and "boundary"
        assert not results[0].flagged

    def test_boundary_probe_flags_no_awareness(self):
        results = run_probes(
            "Simple content with no edge awareness", [Probe.BOUNDARY]
        )
        assert results[0].flagged

    def test_ethical_overlay_flags_low_ihsan(self):
        results = run_probes(self.CONTENT, [Probe.ETHICAL_OVERLAY], ihsan_score=0.5)
        assert results[0].flagged

    def test_ethical_overlay_passes_high_ihsan(self):
        results = run_probes(self.CONTENT, [Probe.ETHICAL_OVERLAY], ihsan_score=0.96)
        assert not results[0].flagged

    def test_compression_flags_verbose(self):
        verbose = "the " * 100  # Very low unique ratio
        results = run_probes(verbose, [Probe.COMPRESSION])
        assert results[0].flagged

    def test_empty_probe_list(self):
        results = run_probes(self.CONTENT, [])
        assert results == []


# ═══════════════════════════════════════════════════════════════
# Engine Integration
# ═══════════════════════════════════════════════════════════════


class TestSAPEv2Engine:
    """Integration tests for the full SAPE v2.0 engine."""

    CONTENT = (
        "We propose a distributed consensus protocol based on Byzantine fault tolerance. "
        "The protocol achieves safety under asynchrony and liveness under partial synchrony. "
        "This follows from the FLP impossibility result and the DLS construction. "
        "Edge cases include network partitions and Byzantine leader failures."
    )

    def test_lite_execution(self):
        engine = SAPEv2Engine()
        intent = IntentSlots(
            domain="testing",
            objective="Verify SAPE lite mode",
            stakes="L",
        )
        result = engine.execute(intent, self.CONTENT)
        assert result.mode == ExecutionMode.LITE
        assert len(result.check_results) == 6
        assert len(result.probe_results) == 0
        assert result.duration_ms > 0

    def test_standard_execution(self):
        engine = SAPEv2Engine()
        intent = IntentSlots(
            domain="distributed systems",
            objective="Design consensus protocol",
            stakes="M",
        )
        result = engine.execute(intent, self.CONTENT)
        assert result.mode == ExecutionMode.STANDARD
        assert len(result.module_results) >= 3
        assert len(result.check_results) == 6

    def test_deep_execution(self):
        engine = SAPEv2Engine()
        intent = IntentSlots(
            domain="safety-critical",
            objective="Design nuclear reactor control",
            stakes="H",
            success_criteria="Zero uncontrolled reactions",
            constraints="Must pass NRC certification",
        )
        result = engine.execute(intent, self.CONTENT)
        assert result.mode == ExecutionMode.DEEP
        assert len(result.module_results) == 7  # All modules
        assert len(result.check_results) == 6
        assert len(result.probe_results) == 9  # All probes

    def test_failed_intent_gate_blocks(self):
        engine = SAPEv2Engine()
        intent = IntentSlots(domain="", objective="", stakes="M")
        result = engine.execute(intent, self.CONTENT)
        assert "BLOCKED" in result.final_output
        assert result.overall_snr == 0.0

    def test_low_intent_score_blocks(self):
        engine = SAPEv2Engine()
        intent = IntentSlots(domain="test", objective="test", stakes="L")
        result = engine.execute(intent, self.CONTENT, intent_score=0.3)
        assert "BLOCKED" in result.final_output

    def test_result_to_dict(self):
        engine = SAPEv2Engine()
        intent = IntentSlots(domain="test", objective="test", stakes="L")
        result = engine.execute(intent, self.CONTENT)
        d = result.to_dict()
        assert d["mode"] == "lite"
        assert "checks" in d
        assert "overall_snr" in d

    def test_custom_snr_fn(self):
        engine = SAPEv2Engine(snr_fn=lambda _: 0.99)
        intent = IntentSlots(domain="test", objective="test", stakes="L")
        result = engine.execute(intent, self.CONTENT)
        assert result.overall_snr > 0


# ═══════════════════════════════════════════════════════════════
# Prompt Compilation
# ═══════════════════════════════════════════════════════════════


class TestPromptCompilation:
    """Test SAPE v2.0 prompt compilation."""

    def test_compile_lite(self):
        intent = IntentSlots(domain="test", objective="test", stakes="L")
        result = compile_sape_v2_prompt(intent)
        assert result["mode"] == "lite"
        assert "SAPE-v2-Activate" in result["user_prompt"]
        assert "prompt_sha256" in result

    def test_compile_deep_includes_all_sections(self):
        intent = IntentSlots(
            domain="safety",
            objective="Design safety system",
            stakes="H",
            success_criteria="Zero harm",
            constraints="ISO 26262",
        )
        result = compile_sape_v2_prompt(intent)
        assert result["mode"] == "deep"
        assert "[Tension Studio]" in result["user_prompt"]
        assert "[Abstraction Elevator]" in result["user_prompt"]
        assert "[Symbolic Harness]" in result["user_prompt"]
        assert "[Rare-Path Prober]" in result["user_prompt"]

    def test_compile_lite_excludes_deep_sections(self):
        intent = IntentSlots(domain="test", objective="test", stakes="L")
        result = compile_sape_v2_prompt(intent)
        assert "[Tension Studio]" not in result["user_prompt"]
        assert "[Abstraction Elevator]" not in result["user_prompt"]

    def test_system_prompt_v2_content(self):
        assert "v2.0" in SAPE_SYSTEM_PROMPT_V2
        assert "Ihsān" in SAPE_SYSTEM_PROMPT_V2
        assert "7–3–6–9" in SAPE_SYSTEM_PROMPT_V2

    def test_compile_with_evidence(self):
        intent = IntentSlots(domain="test", objective="test", stakes="M")
        evidence = [{"label": "RFC 9000", "evidence_level": "VERIFIED"}]
        result = compile_sape_v2_prompt(intent, evidence=evidence)
        assert "RFC 9000" in result["user_prompt"]

    def test_prompt_sha_deterministic(self):
        intent = IntentSlots(domain="test", objective="test", stakes="L")
        r1 = compile_sape_v2_prompt(intent)
        r2 = compile_sape_v2_prompt(intent)
        assert r1["prompt_sha256"] == r2["prompt_sha256"]

    def test_engine_compile_prompt(self):
        engine = SAPEv2Engine()
        intent = IntentSlots(domain="test", objective="test", stakes="M")
        result = engine.compile_prompt(intent)
        assert result["mode"] == "standard"
