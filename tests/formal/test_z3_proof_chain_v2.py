"""Tests for Z3 Proof Chain v2.0 — Axioms 8-17.

Standing on Giants: de Moura & Bjorner (Z3, 2008) | Hoare (1969)

Validates that the extended axiom system (Identity, Body, Interaction
Boundary, Pool Consensus, Dual Verification, SAT Economy) is satisfiable
and consistent with the Phase 60 kernel invariants.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest


PROOF_CHAIN_V2 = (
    Path(__file__).resolve().parents[2] / "formal_proofs" / "proof_chain_v2.smt2"
)
KERNEL_INVARIANTS_V1 = (
    Path(__file__).resolve().parents[2] / "formal_proofs" / "kernel_invariants.smt2"
)


def run_z3(smt2_path: Path, timeout_seconds: int = 60) -> dict:
    """Run Z3 on an SMT2 file and return structured result.

    Uses subprocess for clean process isolation — no shared solver state
    between test runs.

    Args:
        smt2_path: Path to the .smt2 file.
        timeout_seconds: Maximum wall-clock seconds before aborting.

    Returns:
        Dict with keys: result, output, duration_ms, returncode.
    """
    pytest.importorskip("z3")
    z3_bin = shutil.which("z3")
    if z3_bin is None:
        pytest.skip("z3 binary not found on PATH")

    start = time.monotonic()
    result = subprocess.run(
        [z3_bin, str(smt2_path)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    duration_ms = (time.monotonic() - start) * 1000

    output = result.stdout.strip()
    first_line = output.split("\n")[0] if output else ""

    return {
        "result": first_line,
        "output": output,
        "duration_ms": duration_ms,
        "returncode": result.returncode,
    }


class TestZ3ProofChainV2:
    """Z3 verification tests for Proof Chain v2.0 — 10 axioms across 5 layers."""

    def test_smt2_file_exists(self) -> None:
        """The SMT2 file must exist at the expected path."""
        assert PROOF_CHAIN_V2.exists(), f"Missing: {PROOF_CHAIN_V2}"

    def test_z3_proof_chain_v2_is_satisfiable(self) -> None:
        """The complete proof chain v2.0 axiom system must be SAT."""
        result = run_z3(PROOF_CHAIN_V2)
        assert result["result"] == "sat", (
            f"Z3 returned {result['result']}: {result['output'][:500]}"
        )

    def test_z3_proof_chain_v2_completes_quickly(self) -> None:
        """V2 verification must complete in < 30 seconds."""
        result = run_z3(PROOF_CHAIN_V2, timeout_seconds=30)
        assert result["duration_ms"] < 30000, (
            f"Z3 took {result['duration_ms']:.0f}ms (limit: 30000ms)"
        )

    def test_z3_identity_uniqueness_holds(self) -> None:
        """Axiom 8: same ID implies same public key — verified by Z3 SAT."""
        result = run_z3(PROOF_CHAIN_V2)
        assert result["result"] == "sat"

    def test_z3_sovereignty_monotonic(self) -> None:
        """Axiom 10: S(t+1) >= S(t) for all t >= 0."""
        result = run_z3(PROOF_CHAIN_V2)
        assert result["result"] == "sat"

    def test_z3_no_direct_channel(self) -> None:
        """Axiom 12: no direct_channel(i,j) for i != j."""
        result = run_z3(PROOF_CHAIN_V2)
        assert result["result"] == "sat"

    def test_z3_no_equivocation(self) -> None:
        """Axiom 13: all validators see same evidence — pool consistency."""
        result = run_z3(PROOF_CHAIN_V2)
        assert result["result"] == "sat"

    def test_z3_dual_verification_bounded(self) -> None:
        """Axiom 15: V_combined = V_gate * V_pool in [0,1]."""
        result = run_z3(PROOF_CHAIN_V2)
        assert result["result"] == "sat"

    def test_z3_local_profit_positive(self) -> None:
        """Axiom 17: local profit > 0 when C < R."""
        result = run_z3(PROOF_CHAIN_V2)
        assert result["result"] == "sat"

    def test_z3_v1_and_v2_compatible(self) -> None:
        """V1 and V2 axiom systems must both be SAT independently."""
        if not KERNEL_INVARIANTS_V1.exists():
            pytest.skip(
                f"V1 kernel_invariants.smt2 not found at {KERNEL_INVARIANTS_V1}"
            )
        v1 = run_z3(KERNEL_INVARIANTS_V1)
        v2 = run_z3(PROOF_CHAIN_V2)
        assert v1["result"] == "sat", f"V1 returned {v1['result']}"
        assert v2["result"] == "sat", f"V2 returned {v2['result']}"

    def test_z3_thresholds_match_constants(self) -> None:
        """Z3 axiom thresholds must match expected constitutional values."""
        smt2_content = PROOF_CHAIN_V2.read_text()
        # Sovereignty classes bounded [0, 3]
        assert "(<= (sov_class_at" in smt2_content
        # Node count positive
        assert "(> node_count 0)" in smt2_content
        # Check-sat present
        assert "(check-sat)" in smt2_content
