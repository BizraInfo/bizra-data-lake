"""Dema proactive coworker policy — Phase A0.6.

The proactive layer turns the ambient kernel from a heartbeat into a
disciplined coworker: it senses local signals, predicts intent, applies an
interruption policy gated by Ihsān + reversibility, and emits proposal
receipts. It NEVER auto-executes destructive actions.

Pipeline:

    AmbientSignal → IntentPrediction → Decision → ProactiveProposal (receipt)

The four decision outcomes:

    auto_low_risk      low risk + reversible + high confidence — silent queue
    notify             medium-low risk — notify but require ack
    require_approval   medium-or-higher risk — explicit approval before action
    forbid             high-risk irreversible — proactive action forbidden

Storage: every proposal lives under sovereign_state/dema/proposals/<date>/
which is gitignored. No raw private content, no network call, no desktop
control, no MEMORY.md edit.
"""

from __future__ import annotations

from core.dema.proactive.intent_model import IntentPrediction, predict_intent
from core.dema.proactive.interruption_policy import (
    Decision,
    DecisionVerdict,
    decide,
)
from core.dema.proactive.proposal import (
    ProactiveProposal,
    ProposalWriter,
    compose_proposal,
)
from core.dema.proactive.signals import (
    VALID_SIGNAL_KINDS,
    AmbientSignal,
    is_known_signal,
)

__all__ = [
    "AmbientSignal",
    "VALID_SIGNAL_KINDS",
    "is_known_signal",
    "IntentPrediction",
    "predict_intent",
    "Decision",
    "DecisionVerdict",
    "decide",
    "ProactiveProposal",
    "ProposalWriter",
    "compose_proposal",
]
