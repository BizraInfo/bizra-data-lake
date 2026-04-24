"""Runner — the 7-step lifecycle engine for BIZRA agents.

Calls existing core/proof_engine/ functions at each step.
This is the ADK's sole new orchestration logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from core.adk.agent import AgentResult, _DraftOutput, _RefuseOutput
from core.adk.mission import BudgetExhausted, Mission
from core.integration.constants import IHSAN_THRESHOLD

if TYPE_CHECKING:
    from core.adk.agent import Agent


async def execute_agent_lifecycle(agent: "Agent", mission: Mission) -> AgentResult:
    """Execute the full 7-step lifecycle for a BIZRA agent.

    Steps:
    1. NIYYAH   — verify charter integrity, governance class match
    2. BAYYINAH — agent gathers evidence (via act()), auditor verifies
    3. HADD     — budget enforcement (woven into act() via @tool decorator)
    4. AMANAH   — agent produces draft (the act() call itself)
    5. THAMARA  — FATE gate evaluates ihsan + evidence
    6. IISAL    — receipt sealed if PASS, reasons returned if BLOCK
    7. RETROSPECTIVE — loop proof chain finalized
    """
    agent._active_mission = mission

    try:
        # ── Step 1: NIYYAH (intent / charter check) ──
        niyyah_result = _step_niyyah(agent, mission)
        if niyyah_result is not None:
            return niyyah_result

        # ── Steps 2-4: BAYYINAH + HADD + AMANAH (agent acts) ──
        try:
            output = await agent.act(mission)
        except BudgetExhausted as e:
            return AgentResult(
                success=False,
                content="",
                evidence_refs=[],
                ihsan_score=0.0,
                verdict="BLOCKED_BY_BUDGET",
                reason=str(e),
                mission_id=mission.id,
            )

        # Handle refusal — no FATE gate, no loop proof
        if isinstance(output, _RefuseOutput):
            return AgentResult(
                success=False,
                content="",
                evidence_refs=[],
                ihsan_score=0.0,
                verdict="REFUSED",
                reason=output.reason,
                mission_id=mission.id,
                receipt=None,
                loop_proof=None,
            )

        if not isinstance(output, _DraftOutput):
            raise TypeError(
                f"Agent.act() must return self.draft() or self.refuse(), "
                f"got {type(output).__name__}"
            )

        # ── Step 2b: BAYYINAH (evidence audit) ──
        evidence_audit = _step_bayyinah(output.evidence_refs)
        if not evidence_audit.all_refs_valid and output.evidence_refs:
            invalid = evidence_audit.invalid_refs or []
            return AgentResult(
                success=False,
                content=output.content,
                evidence_refs=output.evidence_refs,
                ihsan_score=0.0,
                verdict="BLOCKED_BY_EVIDENCE",
                reason=f"Fabricated evidence detected: {invalid}",
                mission_id=mission.id,
            )

        # ── Step 5: THAMARA (FATE gate) ──
        fate_result = _step_thamara(output, agent)

        verdict_str = fate_result.verdict.verdict
        ihsan = fate_result.verdict.ihsan_score

        # External unverified evidence ceiling
        if mission.allow_external_unverified and ihsan >= IHSAN_THRESHOLD:
            ihsan = IHSAN_THRESHOLD - 0.01  # cap at 0.94
            verdict_str = "BLOCKED_BY_IHSAN"

        is_pass = verdict_str == "PASS" and ihsan >= IHSAN_THRESHOLD

        # ── Step 6: IISAL (receipt) ──
        receipt = _step_iisal(agent, mission, output, ihsan, verdict_str, is_pass)

        # ── Step 7: RETROSPECTIVE (loop proof) ──
        loop_proof = _step_retrospective(mission, output, agent, fate_result, is_pass)

        return AgentResult(
            success=is_pass,
            content=output.content if is_pass else "",
            evidence_refs=output.evidence_refs,
            ihsan_score=ihsan,
            verdict=verdict_str,
            reason=(
                fate_result.verdict.reason
                if hasattr(fate_result.verdict, "reason")
                else ""
            ),
            receipt=receipt,
            loop_proof=loop_proof,
            mission_id=mission.id,
        )

    finally:
        agent._active_mission = None


def _step_niyyah(agent: "Agent", mission: Mission) -> AgentResult | None:
    """Step 1: Verify charter integrity and governance class match."""
    if not agent._charter_hash:
        return AgentResult(
            success=False,
            content="",
            evidence_refs=[],
            ihsan_score=0.0,
            verdict="BLOCKED_BY_CHARTER",
            reason="Agent has no charter",
            mission_id=mission.id,
        )

    # Recompute charter hash to detect drift
    import hashlib

    current_hash = hashlib.blake2b(
        agent._charter_text.encode(), digest_size=32
    ).hexdigest()
    if current_hash != agent._charter_hash:
        return AgentResult(
            success=False,
            content="",
            evidence_refs=[],
            ihsan_score=0.0,
            verdict="BLOCKED_BY_CHARTER",
            reason=f"Charter drift: expected {agent._charter_hash[:16]}..., got {current_hash[:16]}...",
            mission_id=mission.id,
        )

    return None  # NIYYAH passed


def _step_bayyinah(evidence_refs: list[str]):
    """Step 2: Audit evidence via existing proof_engine."""
    from core.proof_engine.evidence_audit import audit_evidence

    return audit_evidence(evidence_refs)


def _step_thamara(output: _DraftOutput, agent: "Agent"):
    """Step 5: FATE gate evaluation via existing proof_engine."""
    from core.proof_engine.fate_gate import validate_with_evidence

    class _PatBridge:
        """Adapts ADK draft output to the PatOutput protocol."""

        def __init__(self, draft: _DraftOutput, confidence: float = 0.95):
            self.answer = draft.content
            self.evidence_refs = draft.evidence_refs
            self.confidence = confidence

    return validate_with_evidence(_PatBridge(output), emit_telemetry=False)


def _step_iisal(agent, mission, output, ihsan, verdict_str, is_pass):
    """Step 6: Build and optionally seal receipt."""
    from core.proof_engine.canonical import CanonPolicy, CanonQuery
    from core.proof_engine.receipt import ReceiptBuilder, SimpleSigner

    signer = SimpleSigner(secret=os.urandom(32))
    builder = ReceiptBuilder(signer=signer)
    query = CanonQuery(
        user_id=mission.requester,
        user_state="active",
        intent=mission.question[:200],
        payload={"agent": agent.name, "mission_id": mission.id},
    )
    policy = CanonPolicy(
        policy_id="bizra-adk-v0.2",
        version="0.2",
        rules={"hard_mode": True},
        thresholds={"ihsan": IHSAN_THRESHOLD},
        constraints=["ZANN_ZERO", "RIBA_ZERO", "CLAIM_MUST_BIND"],
    )
    payload = output.content.encode("utf-8")

    if is_pass:
        receipt = builder.accepted(
            query=query,
            policy=policy,
            payload=payload,
            snr=0.95,
            ihsan_score=ihsan,
            gate_passed="fate",
        )
    else:
        receipt = builder.rejected(
            query=query,
            policy=policy,
            snr=0.0,
            ihsan_score=ihsan,
            gate_failed="fate",
            reason=verdict_str,
            payload=payload,
        )

    return receipt


def _step_retrospective(mission, output, agent, fate_result, is_pass):
    """Step 7: Produce loop proof artifact."""
    from bizra_config import DATA_LAKE_ROOT
    from core.proof_engine.loop_proof import execute_loop_proof

    if not is_pass:
        return None

    confidence = "high"
    proof_dir = Path(
        os.getenv("BIZRA_PROOFS_DIR", str(DATA_LAKE_ROOT / "artifacts" / "proofs"))
    )
    output_path = proof_dir / f"loop-proof-{mission.id}.json"

    try:
        proof_dir.mkdir(parents=True, exist_ok=True)
        loop_proof = execute_loop_proof(
            mission=f"[{agent.name}] {mission.question[:200]}",
            pat_answer=output.content[:2000],
            evidence_refs=output.evidence_refs,
            confidence=confidence,
            output_path=output_path,
        )
        return loop_proof
    except Exception:
        return None
