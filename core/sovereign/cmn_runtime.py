"""
CMN Runtime Harness — Wires the Constitutional Membrane Network into Node0.

Bridges the formal CMN modules into the live sovereign runtime:
  - Boots the GlobalInvariantChecker on kernel start
  - Freezes P5-Ethicist and S2-Oracle at genesis time
  - Initializes IRP trust model with SAT validator registry
  - Exposes constitutional_health() for the /v1/health/constitutional endpoint
  - Feeds autopoiesis receipt observations into the invariant checker

This is the bridge between mathematical proof and production reality.

Standing on Giants:
- Lamport (1977): TLA+ — system invariants as runtime constraints
- Dijkstra (1976): Weakest preconditions — verify before execute
- Hoare (1969): Pre/post conditions as program contracts
- Al-Ghazali (1095): The revival of knowledge demands living proof
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("bizra.sovereign.cmn_runtime")


class CMNRuntime:
    """Constitutional Membrane Network runtime harness.

    Wires the five CMN modules into the Node0 kernel:
    1. WorkspaceBoundary — sovereignty enforcement
    2. MembraneVerifier — DFA property verification
    3. ProofOfTruth — Zann Zero chain validation
    4. RibaZeroAuditor — exact arithmetic enforcement
    5. GlobalInvariantChecker — continuous S∧M∧Z∧R composition
    Plus:
    6. FrozenAgentRegistry — Godel Escape enforcement
    7. IsnadTrustModel — IRP trust for mission chains
    """

    def __init__(
        self,
        data_dir: Path,
        node_id: str = "node0",
        seed_ledger_path: Optional[Path] = None,
        health_ledger_path: Optional[Path] = None,
    ) -> None:
        self._data_dir = data_dir
        self._node_id = node_id
        self._seed_ledger_path = seed_ledger_path
        self._health_ledger_path = health_ledger_path or (data_dir / "cmn_health.jsonl")
        self._booted = False

        # Lazy-initialized components
        self._boundary: Any = None
        self._verifier: Any = None
        self._zann: Any = None
        self._riba: Any = None
        self._checker: Any = None
        self._frozen_registry: Any = None
        self._irp_model: Any = None
        self._boot_time: float = 0.0
        self._check_count: int = 0
        self._last_receipt: Optional[Dict[str, Any]] = None

    def boot(self) -> Dict[str, Any]:
        """Boot the CMN runtime. Call once at kernel start.

        Returns a boot report with component status.
        """
        report: Dict[str, str] = {}
        self._boot_time = time.time()

        # 1. Sovereignty — workspace boundary
        try:
            from core.sovereign.workspace_boundary import WorkspaceBoundary

            self._boundary = WorkspaceBoundary(self._node_id, self._data_dir)
            disjoint = self._boundary.check_disjoint()
            report["sovereignty"] = "ok" if disjoint.disjoint else "VIOLATION"
        except Exception as exc:
            report["sovereignty"] = f"error: {exc}"
            logger.warning("CMN boot: sovereignty check failed: %s", exc)

        # 2. Membrane verifier
        try:
            from core.pci.membrane_verifier import MembraneVerifier

            self._verifier = MembraneVerifier()
            report["membrane"] = "ok"
        except Exception as exc:
            report["membrane"] = f"error: {exc}"

        # 3. Zann Zero — proof of truth
        try:
            from core.proof_engine.proof_of_truth import ProofOfTruth

            self._zann = ProofOfTruth()
            report["zann_zero"] = "ok"
        except Exception as exc:
            report["zann_zero"] = f"error: {exc}"

        # 4. Riba Zero — exact arithmetic auditor
        try:
            from core.treasury.riba_zero_auditor import RibaZeroAuditor

            if self._seed_ledger_path and self._seed_ledger_path.exists():
                self._riba = RibaZeroAuditor(self._seed_ledger_path)
                report["riba_zero"] = "ok"
            else:
                report["riba_zero"] = "no_ledger"
        except Exception as exc:
            report["riba_zero"] = f"error: {exc}"

        # 5. Global invariant checker
        try:
            from core.governance.invariant_checker import GlobalInvariantChecker

            self._checker = GlobalInvariantChecker(
                sovereignty=self._boundary,
                membrane=self._verifier,
                zann=self._zann,
                riba=self._riba,
                health_ledger_path=self._health_ledger_path,
            )
            report["invariant_checker"] = "ok"
        except Exception as exc:
            report["invariant_checker"] = f"error: {exc}"

        # 6. Frozen Agent Registry — freeze P5 + S2 at genesis
        try:
            from core.governance.frozen_agent import FrozenAgentRegistry

            self._frozen_registry = FrozenAgentRegistry()
            self._freeze_constitutional_agents()
            report["frozen_agents"] = "ok"
        except Exception as exc:
            report["frozen_agents"] = f"error: {exc}"

        # 7. IRP Trust Model
        try:
            from core.reasoning.isnad_trust import IsnadTrustModel

            self._irp_model = IsnadTrustModel()
            self._register_sat_validators()
            report["irp_trust"] = "ok"
        except Exception as exc:
            report["irp_trust"] = f"error: {exc}"

        self._booted = True
        logger.info(
            "CMN runtime booted in %.1fms — %d/%d components ok",
            (time.time() - self._boot_time) * 1000,
            sum(1 for v in report.values() if v == "ok"),
            len(report),
        )
        return report

    def constitutional_health(self) -> Dict[str, Any]:
        """Run all invariant checks and return health status.

        This is the backing function for /v1/health/constitutional.
        """
        if not self._booted:
            return {
                "status": "not_booted",
                "invariants": {},
                "ihsan_score": 0.0,
            }

        if self._checker is None:
            return {
                "status": "checker_unavailable",
                "invariants": {},
                "ihsan_score": 0.0,
            }

        receipt = self._checker.check_all()
        self._check_count += 1
        self._last_receipt = receipt.to_dict()

        all_pass = all(receipt.invariants.values())
        return {
            "status": "constitutional" if all_pass else "violation",
            "invariants": receipt.invariants,
            "ihsan_score": receipt.ihsan_score,
            "receipt_hash": receipt.receipt_hash[:16],
            "chain_length": self._check_count,
            "violations": receipt.violations,
            "uptime_s": round(time.time() - self._boot_time, 1),
        }

    def verify_mission_result(self, result: Any) -> Dict[str, Any]:
        """Verify a mission result against membrane properties P1/P2/P3."""
        if self._verifier is None:
            return {"verified": False, "reason": "verifier_not_booted"}

        verification = self._verifier.verify_transformation(result)
        return {
            "verified": verification.passed,
            "checks": {
                k: {"passed": v.passed, "reason": v.reason}
                for k, v in verification.checks.items()
            },
        }

    def evaluate_trust_chain(
        self, narrator_ids: list[str], claim: str = ""
    ) -> Dict[str, Any]:
        """Evaluate IRP trust for a narrator chain."""
        if self._irp_model is None:
            return {"trust": 0.0, "reason": "irp_not_booted"}

        result = self._irp_model.evaluate_chain(narrator_ids, claim)
        return {
            "trust": result.trust,
            "chain_length": result.chain_length,
            "weakest_link": result.weakest_link,
            "poisoned": result.poisoned,
        }

    def guard_frozen_agent(self, agent_id: str) -> bool:
        """Check if an agent modification is allowed. Returns True if allowed."""
        if self._frozen_registry is None:
            return True
        try:
            self._frozen_registry.guard_modification(agent_id)
            return True
        except Exception:
            return False

    def status(self) -> Dict[str, Any]:
        """Full CMN runtime status."""
        return {
            "booted": self._booted,
            "node_id": self._node_id,
            "check_count": self._check_count,
            "uptime_s": round(time.time() - self._boot_time, 1) if self._booted else 0,
            "frozen_agents": list(
                self._frozen_registry._snapshots.keys() if self._frozen_registry else []
            ),
            "irp_narrator_count": (
                self._irp_model.narrator_count() if self._irp_model else 0
            ),
            "last_receipt": self._last_receipt,
        }

    # --- internal ---

    def _freeze_constitutional_agents(self) -> None:
        """Freeze P5-Ethicist and S2-Oracle at genesis."""
        from core.governance.frozen_agent import FROZEN_AGENT_IDS

        for agent_id in FROZEN_AGENT_IDS:
            if not self._frozen_registry.is_frozen(agent_id):
                # Genesis config — these define the constitutional evaluation
                config = {"agent_id": agent_id, "version": "genesis", "mutable": False}
                policy = {"constitutional_gate": True, "drift_allowed": False}
                self._frozen_registry.freeze(
                    agent_id, config, policy, timestamp=self._boot_time
                )
                logger.info("Frozen agent %s at genesis", agent_id)

    def _register_sat_validators(self) -> None:
        """Register SAT-5 agents as IRP trust validators."""
        sat_agents = [
            ("S1-Validator", 0.95),
            ("S2-Oracle", 1.0),  # frozen => maximum trust
            ("S3-Mediator", 0.90),
            ("S4-Archivist", 0.92),
            ("S5-Sentinel", 0.95),
        ]
        for agent_id, trust in sat_agents:
            self._irp_model.register_narrator(agent_id, trust)
