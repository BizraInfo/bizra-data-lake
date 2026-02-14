"""
Z3 FATE Gate Test Suite -- Formal Verification of Constitutional Constraints
=============================================================================

Comprehensive test coverage for the Z3FATEGate class (core/sovereign/z3_fate_gate.py).
Validates SMT-based constitutional constraint enforcement: Ihsan thresholds,
SNR floors, reversibility requirements, resource bounds, and autonomy levels.

Standing on Giants:
- Z3 SMT Solver (de Moura & Bjorner, 2008)
- Shannon (1948) -- Signal-to-Noise Ratio as quality metric
- Saltzer & Schroeder (1975) -- Fail-safe defaults

Test Categories:
1.  Z3_AVAILABLE sentinel
2.  Z3Constraint dataclass
3.  Z3Proof dataclass
4.  Z3FATEGate initialization (default constraints, symbolic vars)
5.  add_constraint (custom, overwrite)
6.  generate_proof (satisfiable/unsatisfiable across all constraint axes)
7.  verify_ihsan (threshold boundary)
8.  verify_snr (threshold boundary)
9.  verify_autonomy (levels 0-4, boundaries, invalid)
10. get_constraints (enumeration)
11. _find_counterexample (individual + combined failures)

Created: 2026-02-11
"""

from __future__ import annotations

import pytest

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD
from core.sovereign.z3_fate_gate import (
    Z3_AVAILABLE,
    Z3Constraint,
    Z3FATEGate,
    Z3Proof,
)

# Skip the entire module if Z3 is not installed, except for TestZ3Available
# which tests the sentinel itself.
z3_required = pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver not installed")


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def gate():
    """Create a fresh Z3FATEGate instance (skips if Z3 unavailable)."""
    if not Z3_AVAILABLE:
        pytest.skip("z3-solver not installed")
    return Z3FATEGate()


@pytest.fixture
def valid_context():
    """A fully-valid action context that satisfies all default constraints."""
    return {
        "ihsan": 0.96,
        "snr": 0.90,
        "risk_level": 0.3,
        "reversible": True,
        "human_approved": False,
        "cost": 10.0,
        "autonomy_limit": 100.0,
    }


# =============================================================================
# 1. Z3_AVAILABLE SENTINEL
# =============================================================================


class TestZ3Available:
    """Z3_AVAILABLE must be a boolean reflecting z3-solver availability."""

    def test_z3_available_is_bool(self):
        assert isinstance(Z3_AVAILABLE, bool)

    def test_z3_available_matches_import(self):
        """If z3 can be imported, Z3_AVAILABLE is True, else False."""
        try:
            import z3  # noqa: F401

            assert Z3_AVAILABLE is True
        except ImportError:
            assert Z3_AVAILABLE is False


# =============================================================================
# 2. Z3Constraint DATACLASS
# =============================================================================


@z3_required
class TestZ3Constraint:
    """Z3Constraint dataclass creation and field access."""

    def test_creation_with_all_fields(self):
        from z3 import Bool

        expr = Bool("test_flag")
        c = Z3Constraint(constraint_id="test_id", expression=expr, description="A test constraint")
        assert c.constraint_id == "test_id"
        assert c.expression is expr
        assert c.description == "A test constraint"

    def test_fields_are_accessible(self):
        from z3 import Real

        v = Real("x")
        c = Z3Constraint("c1", v >= 0, "non-negative x")
        assert c.constraint_id == "c1"
        assert c.description == "non-negative x"

    def test_expression_can_be_any_z3_type(self):
        from z3 import Bool, Implies, Or, Real

        r = Real("r")
        b = Bool("b")
        complex_expr = Implies(r > 0.5, Or(b, r < 1.0))
        c = Z3Constraint("complex", complex_expr, "complex constraint")
        assert c.constraint_id == "complex"


# =============================================================================
# 3. Z3Proof DATACLASS
# =============================================================================


class TestZ3Proof:
    """Z3Proof dataclass creation and default values."""

    def test_creation_with_required_fields(self):
        proof = Z3Proof(
            proof_id="proof_000001",
            constraints_checked=["ihsan_threshold", "snr_floor"],
            satisfiable=True,
        )
        assert proof.proof_id == "proof_000001"
        assert proof.constraints_checked == ["ihsan_threshold", "snr_floor"]
        assert proof.satisfiable is True

    def test_model_defaults_to_none(self):
        proof = Z3Proof(proof_id="p", constraints_checked=[], satisfiable=False)
        assert proof.model is None

    def test_counterexample_defaults_to_none(self):
        proof = Z3Proof(proof_id="p", constraints_checked=[], satisfiable=False)
        assert proof.counterexample is None

    def test_generation_time_defaults_to_zero(self):
        proof = Z3Proof(proof_id="p", constraints_checked=[], satisfiable=True)
        assert proof.generation_time_ms == 0

    def test_creation_with_all_fields(self):
        proof = Z3Proof(
            proof_id="proof_000042",
            constraints_checked=["a", "b", "c"],
            satisfiable=False,
            model={"x": "1"},
            generation_time_ms=17,
            counterexample="ihsan 0.80 < 0.95",
        )
        assert proof.proof_id == "proof_000042"
        assert proof.model == {"x": "1"}
        assert proof.generation_time_ms == 17
        assert proof.counterexample == "ihsan 0.80 < 0.95"

    def test_satisfiable_true_with_model(self):
        proof = Z3Proof(
            proof_id="p",
            constraints_checked=["c1"],
            satisfiable=True,
            model={"ihsan": "0.96"},
        )
        assert proof.satisfiable is True
        assert proof.model is not None

    def test_satisfiable_false_with_counterexample(self):
        proof = Z3Proof(
            proof_id="p",
            constraints_checked=["c1"],
            satisfiable=False,
            counterexample="snr 0.80 < 0.85",
        )
        assert proof.satisfiable is False
        assert "snr" in proof.counterexample


# =============================================================================
# 4. Z3FATEGate INITIALIZATION
# =============================================================================


@z3_required
class TestZ3FATEGateInit:
    """Z3FATEGate constructor: default constraints, symbolic vars, ImportError."""

    def test_creates_four_default_constraints(self, gate):
        constraints = gate.get_constraints()
        assert len(constraints) == 4

    def test_default_constraint_names(self, gate):
        constraints = gate.get_constraints()
        assert "ihsan_threshold" in constraints
        assert "snr_floor" in constraints
        assert "reversibility" in constraints
        assert "resource_bounds" in constraints

    def test_ihsan_constraint_description_contains_threshold(self, gate):
        desc = gate.get_constraints()["ihsan_threshold"]
        assert str(UNIFIED_IHSAN_THRESHOLD) in desc

    def test_snr_constraint_description_contains_threshold(self, gate):
        desc = gate.get_constraints()["snr_floor"]
        assert str(UNIFIED_SNR_THRESHOLD) in desc

    def test_reversibility_description_mentions_keywords(self, gate):
        desc = gate.get_constraints()["reversibility"]
        assert "reversible" in desc.lower() or "human_approved" in desc.lower()

    def test_resource_bounds_description_mentions_cost(self, gate):
        desc = gate.get_constraints()["resource_bounds"]
        assert "cost" in desc.lower() or "autonomy" in desc.lower()

    def test_symbolic_vars_created(self, gate):
        """Internal symbolic variables should exist after construction."""
        assert gate._ihsan is not None
        assert gate._snr is not None
        assert gate._risk_level is not None
        assert gate._reversible is not None
        assert gate._human_approved is not None
        assert gate._cost is not None
        assert gate._autonomy_limit is not None

    def test_proof_counter_starts_at_zero(self, gate):
        assert gate._proof_counter == 0

    def test_import_error_when_z3_unavailable(self):
        """If Z3_AVAILABLE is False, constructor must raise ImportError."""
        from unittest.mock import patch

        with patch("core.sovereign.z3_fate_gate.Z3_AVAILABLE", False):
            with pytest.raises(ImportError, match="Z3 not available"):
                Z3FATEGate()


# =============================================================================
# 5. add_constraint
# =============================================================================


@z3_required
class TestAddConstraint:
    """Adding and overwriting constraints."""

    def test_add_custom_constraint(self, gate):
        from z3 import Real

        x = Real("custom_x")
        gate.add_constraint("custom_check", x > 0, "custom_x must be positive")
        constraints = gate.get_constraints()
        assert "custom_check" in constraints
        assert constraints["custom_check"] == "custom_x must be positive"
        assert len(constraints) == 5  # 4 defaults + 1 custom

    def test_overwrite_existing_constraint(self, gate):
        """Overwriting a constraint replaces description and expression."""
        from z3 import Real

        v = Real("v")
        gate.add_constraint("ihsan_threshold", v >= 0.99, "ihsan >= 0.99 (strict)")
        constraints = gate.get_constraints()
        assert constraints["ihsan_threshold"] == "ihsan >= 0.99 (strict)"
        assert len(constraints) == 4  # no new entry, just replaced

    def test_add_constraint_uses_name_as_default_description(self, gate):
        from z3 import Bool

        b = Bool("flag")
        gate.add_constraint("my_flag", b, "")
        # When description is empty string, the code falls back to the name
        constraints = gate.get_constraints()
        assert constraints["my_flag"] == "my_flag"

    def test_add_multiple_custom_constraints(self, gate):
        from z3 import Real

        for i in range(5):
            v = Real(f"var_{i}")
            gate.add_constraint(f"custom_{i}", v >= i, f"var_{i} >= {i}")
        constraints = gate.get_constraints()
        assert len(constraints) == 9  # 4 defaults + 5 custom


# =============================================================================
# 6. generate_proof
# =============================================================================


@z3_required
class TestGenerateProof:
    """SMT proof generation across all constraint axes."""

    def test_all_constraints_satisfied(self, gate, valid_context):
        proof = gate.generate_proof(valid_context)
        assert proof.satisfiable is True
        assert proof.model is not None
        assert isinstance(proof.model, dict)

    def test_proof_id_increments(self, gate, valid_context):
        p1 = gate.generate_proof(valid_context)
        p2 = gate.generate_proof(valid_context)
        assert p1.proof_id == "proof_000001"
        assert p2.proof_id == "proof_000002"

    def test_proof_id_format(self, gate, valid_context):
        proof = gate.generate_proof(valid_context)
        assert proof.proof_id.startswith("proof_")
        numeric_part = proof.proof_id.split("_")[1]
        assert numeric_part.isdigit()
        assert len(numeric_part) == 6

    def test_constraints_checked_lists_all_defaults(self, gate, valid_context):
        proof = gate.generate_proof(valid_context)
        expected = {"ihsan_threshold", "snr_floor", "reversibility", "resource_bounds"}
        assert set(proof.constraints_checked) == expected

    def test_generation_time_non_negative(self, gate, valid_context):
        proof = gate.generate_proof(valid_context)
        assert proof.generation_time_ms >= 0

    def test_satisfiable_proof_has_no_counterexample(self, gate, valid_context):
        proof = gate.generate_proof(valid_context)
        assert proof.satisfiable is True
        assert proof.counterexample is None

    # --- Individual constraint failures ---

    def test_low_ihsan_fails(self, gate, valid_context):
        ctx = {**valid_context, "ihsan": 0.90}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        assert proof.counterexample is not None
        assert "ihsan" in proof.counterexample.lower()

    def test_ihsan_just_below_threshold_fails(self, gate, valid_context):
        ctx = {**valid_context, "ihsan": 0.9499}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        assert "ihsan" in proof.counterexample.lower()

    def test_ihsan_at_threshold_succeeds(self, gate, valid_context):
        ctx = {**valid_context, "ihsan": UNIFIED_IHSAN_THRESHOLD}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is True

    def test_low_snr_fails(self, gate, valid_context):
        ctx = {**valid_context, "snr": 0.80}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        assert proof.counterexample is not None
        assert "snr" in proof.counterexample.lower()

    def test_snr_just_below_threshold_fails(self, gate, valid_context):
        ctx = {**valid_context, "snr": 0.8499}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        assert "snr" in proof.counterexample.lower()

    def test_snr_at_threshold_succeeds(self, gate, valid_context):
        ctx = {**valid_context, "snr": UNIFIED_SNR_THRESHOLD}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is True

    def test_high_risk_without_reversible_or_approved_fails(self, gate, valid_context):
        ctx = {
            **valid_context,
            "risk_level": 0.8,
            "reversible": False,
            "human_approved": False,
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        assert proof.counterexample is not None
        assert "high_risk" in proof.counterexample.lower() or "revers" in proof.counterexample.lower()

    def test_high_risk_with_reversible_succeeds(self, gate, valid_context):
        ctx = {
            **valid_context,
            "risk_level": 0.8,
            "reversible": True,
            "human_approved": False,
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is True

    def test_high_risk_with_human_approved_succeeds(self, gate, valid_context):
        ctx = {
            **valid_context,
            "risk_level": 0.8,
            "reversible": False,
            "human_approved": True,
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is True

    def test_high_risk_with_both_reversible_and_approved_succeeds(self, gate, valid_context):
        ctx = {
            **valid_context,
            "risk_level": 0.8,
            "reversible": True,
            "human_approved": True,
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is True

    def test_risk_at_boundary_0_7_does_not_trigger_reversibility(self, gate, valid_context):
        """risk_level == 0.7 does NOT exceed 0.7 threshold, so reversibility not required."""
        ctx = {
            **valid_context,
            "risk_level": 0.7,
            "reversible": False,
            "human_approved": False,
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is True

    def test_risk_just_above_0_7_triggers_reversibility(self, gate, valid_context):
        """risk_level == 0.71 exceeds 0.7, so reversibility IS required."""
        ctx = {
            **valid_context,
            "risk_level": 0.71,
            "reversible": False,
            "human_approved": False,
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False

    def test_cost_exceeds_limit_fails(self, gate, valid_context):
        ctx = {**valid_context, "cost": 200.0, "autonomy_limit": 100.0}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        assert proof.counterexample is not None
        assert "cost" in proof.counterexample.lower()

    def test_cost_equals_limit_succeeds(self, gate, valid_context):
        ctx = {**valid_context, "cost": 100.0, "autonomy_limit": 100.0}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is True

    def test_cost_just_above_limit_fails(self, gate, valid_context):
        ctx = {**valid_context, "cost": 100.01, "autonomy_limit": 100.0}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False

    # --- Multiple violations ---

    def test_multiple_violations_reported(self, gate):
        ctx = {
            "ihsan": 0.50,
            "snr": 0.50,
            "risk_level": 0.9,
            "reversible": False,
            "human_approved": False,
            "cost": 500.0,
            "autonomy_limit": 10.0,
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        ce = proof.counterexample.lower()
        assert "ihsan" in ce
        assert "snr" in ce
        assert "high_risk" in ce
        assert "cost" in ce

    def test_unsatisfiable_proof_has_no_model(self, gate, valid_context):
        ctx = {**valid_context, "ihsan": 0.50}
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False
        assert proof.model is None

    # --- Edge cases ---

    def test_missing_optional_bool_defaults_to_false(self, gate):
        """When reversible/human_approved are not in context, they default to False."""
        ctx = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.3,
            "cost": 10.0,
            "autonomy_limit": 100.0,
            # reversible and human_approved NOT provided
        }
        proof = gate.generate_proof(ctx)
        # Low risk, so reversibility not needed; should pass
        assert proof.satisfiable is True

    def test_missing_optional_bool_with_high_risk_fails(self, gate):
        """High risk without reversible/human_approved in context should fail."""
        ctx = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.8,
            "cost": 10.0,
            "autonomy_limit": 100.0,
            # reversible and human_approved NOT provided -> default False
        }
        proof = gate.generate_proof(ctx)
        assert proof.satisfiable is False

    def test_partial_context_solver_finds_assignment(self, gate):
        """When numeric keys are missing, solver is free to find satisfying values."""
        ctx = {
            "ihsan": 0.96,
            "snr": 0.90,
            # risk_level, cost, autonomy_limit not bound
            # reversible/human_approved default to False
        }
        proof = gate.generate_proof(ctx)
        # Solver can pick risk_level <= 0.7, cost <= autonomy_limit to satisfy
        assert proof.satisfiable is True


# =============================================================================
# 7. verify_ihsan
# =============================================================================


@z3_required
class TestVerifyIhsan:
    """Ihsan threshold boundary checks."""

    def test_at_threshold(self, gate):
        assert gate.verify_ihsan(UNIFIED_IHSAN_THRESHOLD) is True

    def test_above_threshold(self, gate):
        assert gate.verify_ihsan(0.99) is True

    def test_below_threshold(self, gate):
        assert gate.verify_ihsan(0.94) is False

    def test_exactly_one(self, gate):
        assert gate.verify_ihsan(1.0) is True

    def test_zero(self, gate):
        assert gate.verify_ihsan(0.0) is False

    def test_just_below_threshold(self, gate):
        assert gate.verify_ihsan(UNIFIED_IHSAN_THRESHOLD - 0.001) is False


# =============================================================================
# 8. verify_snr
# =============================================================================


@z3_required
class TestVerifySNR:
    """SNR threshold boundary checks."""

    def test_at_threshold(self, gate):
        assert gate.verify_snr(UNIFIED_SNR_THRESHOLD) is True

    def test_above_threshold(self, gate):
        assert gate.verify_snr(0.95) is True

    def test_below_threshold(self, gate):
        assert gate.verify_snr(0.80) is False

    def test_exactly_one(self, gate):
        assert gate.verify_snr(1.0) is True

    def test_zero(self, gate):
        assert gate.verify_snr(0.0) is False

    def test_just_below_threshold(self, gate):
        assert gate.verify_snr(UNIFIED_SNR_THRESHOLD - 0.001) is False

    def test_negative_value(self, gate):
        assert gate.verify_snr(-0.5) is False


# =============================================================================
# 9. verify_autonomy
# =============================================================================


@z3_required
class TestVerifyAutonomy:
    """Autonomy level vs risk validation across all 5 levels."""

    # Level 0: max_risk = 1.0
    def test_level_0_accepts_max_risk(self, gate):
        assert gate.verify_autonomy(0, 1.0) is True

    def test_level_0_accepts_low_risk(self, gate):
        assert gate.verify_autonomy(0, 0.0) is True

    # Level 1: max_risk = 0.9
    def test_level_1_accepts_at_boundary(self, gate):
        assert gate.verify_autonomy(1, 0.9) is True

    def test_level_1_rejects_above_boundary(self, gate):
        assert gate.verify_autonomy(1, 0.91) is False

    # Level 2: max_risk = 0.7
    def test_level_2_accepts_at_boundary(self, gate):
        assert gate.verify_autonomy(2, 0.7) is True

    def test_level_2_rejects_above_boundary(self, gate):
        assert gate.verify_autonomy(2, 0.71) is False

    # Level 3: max_risk = 0.5
    def test_level_3_accepts_at_boundary(self, gate):
        assert gate.verify_autonomy(3, 0.5) is True

    def test_level_3_rejects_above_boundary(self, gate):
        assert gate.verify_autonomy(3, 0.51) is False

    # Level 4: max_risk = 0.3
    def test_level_4_accepts_at_boundary(self, gate):
        assert gate.verify_autonomy(4, 0.3) is True

    def test_level_4_rejects_above_boundary(self, gate):
        assert gate.verify_autonomy(4, 0.31) is False

    def test_level_4_accepts_zero_risk(self, gate):
        assert gate.verify_autonomy(4, 0.0) is True

    # Invalid level: max_risk defaults to 0.0
    def test_invalid_level_returns_false_for_any_positive_risk(self, gate):
        assert gate.verify_autonomy(5, 0.1) is False

    def test_invalid_level_accepts_zero_risk(self, gate):
        """Level 5 (invalid) maps to max_risk=0.0, and 0.0 <= 0.0 is True."""
        assert gate.verify_autonomy(5, 0.0) is True

    def test_negative_level_returns_false(self, gate):
        assert gate.verify_autonomy(-1, 0.5) is False


# =============================================================================
# 10. get_constraints
# =============================================================================


@z3_required
class TestGetConstraints:
    """Constraint enumeration."""

    def test_returns_dict(self, gate):
        result = gate.get_constraints()
        assert isinstance(result, dict)

    def test_returns_all_four_defaults(self, gate):
        result = gate.get_constraints()
        assert len(result) == 4

    def test_keys_are_constraint_ids(self, gate):
        result = gate.get_constraints()
        expected_keys = {"ihsan_threshold", "snr_floor", "reversibility", "resource_bounds"}
        assert set(result.keys()) == expected_keys

    def test_values_are_descriptions(self, gate):
        result = gate.get_constraints()
        for desc in result.values():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_reflects_added_constraints(self, gate):
        from z3 import Real

        v = Real("extra")
        gate.add_constraint("extra_check", v > 0, "extra must be positive")
        result = gate.get_constraints()
        assert "extra_check" in result
        assert len(result) == 5


# =============================================================================
# 11. _find_counterexample
# =============================================================================


@z3_required
class TestFindCounterexample:
    """Counterexample identification for individual and combined failures."""

    def test_ihsan_failure_only(self, gate):
        ctx = {
            "ihsan": 0.80,
            "snr": 0.90,
            "risk_level": 0.3,
            "reversible": True,
            "cost": 10.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert "ihsan" in result.lower()
        assert "snr" not in result.lower()

    def test_snr_failure_only(self, gate):
        ctx = {
            "ihsan": 0.96,
            "snr": 0.50,
            "risk_level": 0.3,
            "reversible": True,
            "cost": 10.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert "snr" in result.lower()
        assert "ihsan" not in result.lower()

    def test_high_risk_failure_only(self, gate):
        ctx = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.9,
            "reversible": False,
            "human_approved": False,
            "cost": 10.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert "high_risk" in result.lower()

    def test_cost_failure_only(self, gate):
        ctx = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.3,
            "reversible": True,
            "cost": 500.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert "cost" in result.lower()

    def test_all_failures_combined(self, gate):
        ctx = {
            "ihsan": 0.50,
            "snr": 0.50,
            "risk_level": 0.9,
            "reversible": False,
            "human_approved": False,
            "cost": 999.0,
            "autonomy_limit": 1.0,
        }
        result = gate._find_counterexample(ctx)
        assert "ihsan" in result.lower()
        assert "snr" in result.lower()
        assert "high_risk" in result.lower()
        assert "cost" in result.lower()

    def test_all_pass_returns_unknown_violation(self, gate):
        """When no individual check fails, result is 'Unknown violation'."""
        ctx = {
            "ihsan": 0.99,
            "snr": 0.99,
            "risk_level": 0.3,
            "reversible": True,
            "cost": 10.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert result == "Unknown violation"

    def test_missing_keys_raises_type_error(self, gate):
        """Empty dict triggers latent bug: ctx.get('ihsan') returns None, :.3f on None raises TypeError.

        This documents a known defect in _find_counterexample: the comparison
        `ctx.get("ihsan", 0) < threshold` correctly defaults to 0, but the
        format string `f"ihsan {ctx.get('ihsan'):.3f}"` does not supply a
        default, so None.__format__(":.3f") raises TypeError.
        """
        with pytest.raises(TypeError):
            gate._find_counterexample({})

    def test_high_risk_not_triggered_when_reversible(self, gate):
        ctx = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.9,
            "reversible": True,
            "human_approved": False,
            "cost": 10.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert "high_risk" not in result.lower()

    def test_high_risk_not_triggered_when_human_approved(self, gate):
        ctx = {
            "ihsan": 0.96,
            "snr": 0.90,
            "risk_level": 0.9,
            "reversible": False,
            "human_approved": True,
            "cost": 10.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert "high_risk" not in result.lower()

    def test_semicolon_separator_for_multiple_failures(self, gate):
        ctx = {
            "ihsan": 0.50,
            "snr": 0.50,
            "risk_level": 0.3,
            "reversible": True,
            "cost": 10.0,
            "autonomy_limit": 100.0,
        }
        result = gate._find_counterexample(ctx)
        assert "; " in result  # Two failures separated by "; "


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
