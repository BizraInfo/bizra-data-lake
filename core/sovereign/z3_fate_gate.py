"""
Z3 FATE Gate — Formal Verification of Constitutional Constraints via SMT
=========================================================================
Standing on Giants: Z3 SMT Solver (de Moura & Bjorner, 2008)

Uses Z3 theorem prover to formally verify agent actions satisfy
constitutional constraints BEFORE execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from z3 import Bool, Implies, Or, Real, Solver, sat

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    Solver = Real = Bool = Or = Implies = sat = None

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class Z3Constraint:
    """A Z3 constraint: constraint_id, expression (Z3 formula), description."""

    constraint_id: str
    expression: Any  # z3.BoolRef
    description: str


@dataclass
class Z3Proof:
    """Z3 proof result: proof_id, constraints_checked, satisfiable, model, generation_time_ms."""

    proof_id: str
    constraints_checked: List[str]
    satisfiable: bool
    model: Optional[Dict[str, Any]] = None
    generation_time_ms: int = 0
    counterexample: Optional[str] = None


class Z3FATEGate:
    """
    Formal Assertion Through Enumeration Gate.
    Uses Z3 SMT solver to verify actions satisfy constitutional constraints.
    """

    def __init__(self):
        """Initialize Z3 solver with default constitutional constraints."""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 not available. Install: pip install z3-solver")

        self._constraints: Dict[str, Z3Constraint] = {}
        self._proof_counter = 0

        # Symbolic variables
        self._ihsan = Real("ihsan")
        self._snr = Real("snr")
        self._risk_level = Real("risk_level")
        self._reversible = Bool("reversible")
        self._human_approved = Bool("human_approved")
        self._cost = Real("cost")
        self._autonomy_limit = Real("autonomy_limit")

        # Default constraints
        self.add_constraint(
            "ihsan_threshold",
            self._ihsan >= UNIFIED_IHSAN_THRESHOLD,
            f"ihsan >= {UNIFIED_IHSAN_THRESHOLD}",
        )
        self.add_constraint(
            "snr_floor",
            self._snr >= UNIFIED_SNR_THRESHOLD,
            f"snr >= {UNIFIED_SNR_THRESHOLD}",
        )
        self.add_constraint(
            "reversibility",
            Implies(self._risk_level > 0.7, Or(self._reversible, self._human_approved)),
            "high_risk => reversible OR human_approved",
        )
        self.add_constraint(
            "resource_bounds",
            self._cost <= self._autonomy_limit,
            "cost <= autonomy_limit",
        )

    def add_constraint(self, name: str, z3_expr: Any, description: str = "") -> None:
        """Register a constitutional constraint."""
        self._constraints[name] = Z3Constraint(name, z3_expr, description or name)

    def generate_proof(self, action_context: Dict[str, Any]) -> Z3Proof:
        """Check if action satisfies all constraints. Returns Z3Proof."""
        start_ns = time.perf_counter_ns()
        self._proof_counter += 1
        proof_id = f"proof_{self._proof_counter:06d}"

        solver = Solver()
        for c in self._constraints.values():
            solver.add(c.expression)

        # Bind concrete values
        bindings = [
            ("ihsan", self._ihsan),
            ("snr", self._snr),
            ("risk_level", self._risk_level),
            ("cost", self._cost),
            ("autonomy_limit", self._autonomy_limit),
        ]
        for key, var in bindings:
            if key in action_context:
                solver.add(var == action_context[key])

        for key, var in [
            ("reversible", self._reversible),
            ("human_approved", self._human_approved),
        ]:
            solver.add(var == action_context.get(key, False))

        result = solver.check()
        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

        if result == sat:
            model = solver.model()
            return Z3Proof(
                proof_id,
                list(self._constraints.keys()),
                True,
                {d.name(): str(model[d]) for d in model.decls()},
                elapsed_ms,
            )

        return Z3Proof(
            proof_id,
            list(self._constraints.keys()),
            False,
            None,
            elapsed_ms,
            self._find_counterexample(action_context),
        )

    def _find_counterexample(self, ctx: Dict[str, Any]) -> str:
        """Identify which constraint failed."""
        failed = []
        if ctx.get("ihsan", 0) < UNIFIED_IHSAN_THRESHOLD:
            failed.append(f"ihsan {ctx.get('ihsan'):.3f} < {UNIFIED_IHSAN_THRESHOLD}")
        if ctx.get("snr", 0) < UNIFIED_SNR_THRESHOLD:
            failed.append(f"snr {ctx.get('snr'):.3f} < {UNIFIED_SNR_THRESHOLD}")
        if (
            ctx.get("risk_level", 0) > 0.7
            and not ctx.get("reversible")
            and not ctx.get("human_approved")
        ):
            failed.append("high_risk without reversible/approved")
        if ctx.get("cost", 0) > ctx.get("autonomy_limit", 0):
            failed.append(f"cost {ctx.get('cost')} > limit {ctx.get('autonomy_limit')}")
        return "; ".join(failed) or "Unknown violation"

    def verify_ihsan(self, ihsan_score: float) -> bool:
        """Verify ihsan >= 0.95."""
        return ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    def verify_snr(self, snr_score: float) -> bool:
        """Verify SNR >= 0.85."""
        return snr_score >= UNIFIED_SNR_THRESHOLD

    def verify_autonomy(self, level: int, risk: float) -> bool:
        """Check autonomy level (0-4) is appropriate for risk level."""
        max_risk = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3}.get(level, 0.0)
        return risk <= max_risk

    def get_constraints(self) -> Dict[str, str]:
        """Return constraint IDs mapped to descriptions."""
        return {c.constraint_id: c.description for c in self._constraints.values()}


__all__ = ["Z3Constraint", "Z3Proof", "Z3FATEGate", "Z3_AVAILABLE"]
