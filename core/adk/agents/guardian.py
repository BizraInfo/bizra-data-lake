"""Guardian Agent — the bridge between PAT-7 and SAT-5.

Runs all 5 SAT gates against any claim or receipt, producing a
composite constitutional compliance report. The Guardian is the
agent that ensures PAT output meets SAT standards before crossing
the FATE boundary.
"""

from __future__ import annotations

import os
from typing import List

from core.adk.agent import Agent, charter
from core.adk.mission import Mission
from core.adk.tools import tool

OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("BIZRA_GUARDIAN_MODEL", "gemma4:26b-bizra-16k")


@charter("""
I am the Guardian. I enforce constitutional compliance by running all 5
SAT gates against claims and receipts. I never approve output that fails
any gate. I report exactly which gates passed, which failed, and why.
I am the last line of defense before any PAT output reaches the
Universal Resource Pool.
""")
class GuardianAgent(Agent):
    name = "Guardian"
    governance_class = "PAT"
    model = MODEL

    @tool(max_results=10)
    def run_sat_gates(self, claim: str) -> List[str]:
        """Execute all 5 SAT gates and collect results."""
        refs: list[str] = []

        try:
            from core.sat.composite_evaluator import evaluate_all_gates

            verdict = evaluate_all_gates(skip_slow=True, skip_manual=True)

            results_text = []
            for gate_name, gate_result in verdict.gate_results.items():
                status = "PASS" if gate_result.passed else "FAIL"
                checks = len(gate_result.checks)
                failed = len(gate_result.failed)
                results_text.append(
                    f"{gate_name}: {status} ({checks} checks, {failed} failed)"
                )
                refs.append(f"sat-gate:{gate_name}:{status}")

            self._gate_verdict = verdict
            self._evidence_text = (
                f"SAT-5 Composite Evaluation:\n"
                f"  Overall: {'PASS' if verdict.passed else 'BLOCKED'}\n"
                f"  Ihsan: {verdict.ihsan_score}\n"
                f"  Blocking: {verdict.blocking_gates or 'none'}\n\n"
                + "\n".join(results_text)
            )
        except Exception as e:
            self._gate_verdict = None
            self._evidence_text = f"SAT-5 evaluation failed: {e}"
            refs.append("sat-gate:error")

        return refs

    @tool(max_results=5)
    def check_constitutional_constants(self, _query: str) -> List[str]:
        """Verify constitutional thresholds are within bounds."""
        refs: list[str] = []
        checks: list[str] = []

        try:
            from core.integration.constants import (
                IHSAN_THRESHOLD,
                SNR_THRESHOLD,
            )

            checks.append(
                f"IHSAN_THRESHOLD={IHSAN_THRESHOLD} (must be >= 0.95): {'OK' if IHSAN_THRESHOLD >= 0.95 else 'VIOLATION'}"
            )
            checks.append(
                f"SNR_THRESHOLD={SNR_THRESHOLD} (must be >= 0.85): {'OK' if SNR_THRESHOLD >= 0.85 else 'VIOLATION'}"
            )
            refs.append("const:thresholds")
        except ImportError:
            checks.append("FAILED to import constants")
            refs.append("const:import_failed")

        self._const_text = "\n".join(checks)
        return refs

    async def act(self, mission: Mission):
        # Run SAT gates
        gate_refs = self.run_sat_gates(mission.question)

        # Check constitutional constants
        const_refs = self.check_constitutional_constants(mission.question)

        all_refs = gate_refs + const_refs
        if not all_refs:
            return self.refuse(reason="Could not evaluate SAT gates or constants")

        self._evidence_text + "\n\n" + self._const_text

        # Guardian produces a compliance report — no LLM needed for the verdict
        # The SAT gates themselves are the authority
        if self._gate_verdict and self._gate_verdict.passed:
            report = (
                f"# SAT-5 Compliance Report\n\n"
                f"**Verdict: PASS**\n"
                f"**Ihsan Score: {self._gate_verdict.ihsan_score}**\n\n"
                f"## Gate Results\n\n{self._evidence_text}\n\n"
                f"## Constitutional Thresholds\n\n{self._const_text}\n\n"
                f"All 5 SAT gates passed. Output is cleared for FATE crossing."
            )
        else:
            blocking = (
                self._gate_verdict.blocking_gates if self._gate_verdict else ["unknown"]
            )
            report = (
                f"# SAT-5 Compliance Report\n\n"
                f"**Verdict: BLOCKED**\n"
                f"**Blocking Gates: {', '.join(blocking)}**\n\n"
                f"## Gate Results\n\n{self._evidence_text}\n\n"
                f"## Constitutional Thresholds\n\n{self._const_text}\n\n"
                f"Output is NOT cleared for FATE crossing. Fix the blocking gates first."
            )

        return self.draft(content=report, evidence=all_refs)
