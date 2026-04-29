"""A2.0 Proof Convergence Verifier.

This module is intentionally pure and runtime-light: it does not boot
SovereignRuntime or read receipt stores. Callers pass already-computed proof
signals, and the verifier collapses them into one canonical CSL
DecisionVerdict.

Policy:
- no proof signals -> require explicit approval
- missing/unknown proof state -> require explicit approval
- any failed proof signal -> forbid
- passing but disagreeing decision signals -> safest verdict + no convergence
- passing and agreeing proof signals -> converged canonical verdict
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from core.dema.csl import DECISION_VERDICTS, DISPLAY_TRUTH_LABELS, DecisionVerdict

_DECISION_PRECEDENCE: dict[str, int] = {
    DecisionVerdict.AUTO_LOW_RISK.value: 0,
    DecisionVerdict.NOTIFY.value: 1,
    DecisionVerdict.REQUIRE_APPROVAL.value: 2,
    DecisionVerdict.FORBID.value: 3,
}


def _validate_decision(value: str | None) -> None:
    if value is not None and value not in DECISION_VERDICTS:
        raise ValueError(f"decision must be one of {DECISION_VERDICTS}, got {value!r}")


def _safest_decision(values: Iterable[str]) -> str:
    decisions = tuple(values)
    if not decisions:
        return DecisionVerdict.REQUIRE_APPROVAL.value
    return max(decisions, key=lambda v: _DECISION_PRECEDENCE[v])


@dataclass(frozen=True)
class ProofSignal:
    """One proof subsystem's evidence toward a Dema decision.

    ``passed`` is tri-state:
    - True: the source accepted the claim/action.
    - False: the source rejected it; convergence fails closed to ``forbid``.
    - None: the source could not decide; explicit approval is required.
    """

    source: str
    passed: bool | None
    truth_label: str = "DERIVED"
    decision: str | None = None
    reason: str = ""
    receipt_id: str | None = None
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if self.truth_label not in DISPLAY_TRUTH_LABELS:
            raise ValueError(
                f"truth_label must be one of {DISPLAY_TRUTH_LABELS}, "
                f"got {self.truth_label!r}"
            )
        _validate_decision(self.decision)
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProofConvergenceResult:
    """Canonical verifier output for proof-gated Dema decisions."""

    converged: bool
    verdict: str
    truth_label: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)
    blocking_sources: tuple[str, ...] = field(default_factory=tuple)
    missing_sources: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_decision(self.verdict)
        if self.truth_label not in DISPLAY_TRUTH_LABELS:
            raise ValueError(
                f"truth_label must be one of {DISPLAY_TRUTH_LABELS}, "
                f"got {self.truth_label!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ProofConvergenceVerifier:
    """Collapse independent proof signals into one CSL decision verdict."""

    def verify(
        self,
        signals: Iterable[ProofSignal],
        *,
        requested_decision: str | None = None,
        required_sources: Iterable[str] = (),
    ) -> ProofConvergenceResult:
        _validate_decision(requested_decision)

        ordered = tuple(sorted(signals, key=lambda s: s.source))
        sources = tuple(signal.source for signal in ordered)
        if len(set(sources)) != len(sources):
            duplicates = sorted(
                {source for source in sources if sources.count(source) > 1}
            )
            raise ValueError(f"duplicate proof signal sources: {duplicates}")

        required = tuple(sorted(set(required_sources)))
        missing = tuple(source for source in required if source not in sources)
        evidence_refs = tuple(
            ref for signal in ordered for ref in sorted(signal.evidence_refs)
        )

        if not ordered:
            return ProofConvergenceResult(
                converged=False,
                verdict=DecisionVerdict.REQUIRE_APPROVAL.value,
                truth_label="UNKNOWN",
                reasons=("no_proof_signals",),
                sources=(),
                missing_sources=required,
                evidence_refs=evidence_refs,
            )

        if missing:
            return ProofConvergenceResult(
                converged=False,
                verdict=DecisionVerdict.REQUIRE_APPROVAL.value,
                truth_label="UNKNOWN",
                reasons=("missing_required_sources",),
                sources=sources,
                missing_sources=missing,
                evidence_refs=evidence_refs,
            )

        indeterminate = tuple(
            signal.source
            for signal in ordered
            if signal.passed is None or signal.truth_label == "UNKNOWN"
        )
        if indeterminate:
            return ProofConvergenceResult(
                converged=False,
                verdict=DecisionVerdict.REQUIRE_APPROVAL.value,
                truth_label="UNKNOWN",
                reasons=("indeterminate_proof_state",),
                sources=sources,
                missing_sources=(),
                evidence_refs=evidence_refs,
            )

        blocking = tuple(signal.source for signal in ordered if signal.passed is False)
        if blocking:
            reasons = tuple(
                signal.reason or f"{signal.source}: proof signal rejected"
                for signal in ordered
                if signal.passed is False
            )
            return ProofConvergenceResult(
                converged=True,
                verdict=DecisionVerdict.FORBID.value,
                truth_label="DERIVED",
                reasons=reasons,
                sources=sources,
                blocking_sources=blocking,
                evidence_refs=evidence_refs,
            )

        decisions = tuple(
            signal.decision for signal in ordered if signal.decision is not None
        )
        if requested_decision is not None:
            decisions = (*decisions, requested_decision)

        unique_decisions = tuple(sorted(set(decisions)))
        if len(unique_decisions) > 1:
            return ProofConvergenceResult(
                converged=False,
                verdict=_safest_decision(unique_decisions),
                truth_label="DERIVED",
                reasons=("decision_disagreement",),
                sources=sources,
                evidence_refs=evidence_refs,
            )

        verdict = (
            unique_decisions[0]
            if unique_decisions
            else DecisionVerdict.REQUIRE_APPROVAL.value
        )
        return ProofConvergenceResult(
            converged=True,
            verdict=verdict,
            truth_label="DERIVED",
            reasons=("proof_signals_converged",),
            sources=sources,
            evidence_refs=evidence_refs,
        )


def converge_proofs(
    signals: Iterable[ProofSignal],
    *,
    requested_decision: str | None = None,
    required_sources: Iterable[str] = (),
) -> ProofConvergenceResult:
    """Convenience wrapper for one-shot proof convergence."""

    return ProofConvergenceVerifier().verify(
        signals,
        requested_decision=requested_decision,
        required_sources=required_sources,
    )
