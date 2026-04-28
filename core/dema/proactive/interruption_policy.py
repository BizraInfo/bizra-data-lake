"""Interruption policy — turns an IntentPrediction into a Decision.

The four verdicts:

    auto_low_risk     low risk + reversible + confidence ≥ AUTO_LOW_THRESHOLD
                      (silent queue; surfaced as a non-blocking proposal)
    notify            low risk + reversible at lower confidence; or medium
                      risk reversible — notify, do not act yet
    require_approval  medium risk OR low-confidence on a low-risk action;
                      explicit approval gate
    forbid            high risk OR irreversible OR a destructive intent —
                      proactive action forbidden; only a manually-issued
                      command can run it

Policy is intentionally conservative. The threshold constants live here so
all decisions are centralised and auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from core.dema.proactive.intent_model import IntentPrediction

DecisionVerdict = Literal["auto_low_risk", "notify", "require_approval", "forbid"]

AUTO_LOW_THRESHOLD: float = 0.85
NOTIFY_THRESHOLD: float = 0.55

# Intent labels that are forbidden to ever auto-fire regardless of
# confidence; they always escalate at least to require_approval, and the
# destructive ones drop to forbid.
DESTRUCTIVE_INTENTS: frozenset[str] = frozenset(
    {
        "propose_format_drive",
        "propose_credential_audit",
    }
)
APPROVAL_ONLY_INTENTS: frozenset[str] = frozenset(
    {
        "propose_duplicate_review",
        "propose_social_draft",
    }
)


@dataclass
class Decision:
    verdict: DecisionVerdict
    reason: str
    intent: IntentPrediction
    user_preference: str = "default"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["intent"] = self.intent.to_dict()
        return d


def decide(
    intent: IntentPrediction,
    *,
    user_preference: str = "default",
) -> Decision:
    """Apply the policy.

    Order of evaluation:
      1. destructive intent  → forbid
      2. high risk            → forbid
      3. irreversible         → require_approval (or forbid if also high risk)
      4. approval-only intent → require_approval
      5. confidence vs thresholds + risk class →
            auto_low_risk | notify | require_approval | forbid
    """
    # 1 + 2: hard blockers.
    if intent.intent in DESTRUCTIVE_INTENTS:
        return Decision(
            verdict="forbid",
            reason="destructive intent — proactive action is forbidden by policy",
            intent=intent,
            user_preference=user_preference,
        )
    if intent.risk == "high":
        return Decision(
            verdict="forbid",
            reason="risk class is high — proactive action is forbidden by policy",
            intent=intent,
            user_preference=user_preference,
        )

    # 3: irreversible work always escalates.
    if not intent.reversible:
        return Decision(
            verdict="require_approval",
            reason="action is not reversible — explicit approval required",
            intent=intent,
            user_preference=user_preference,
        )

    # 4: approval-only intents.
    if intent.intent in APPROVAL_ONLY_INTENTS:
        return Decision(
            verdict="require_approval",
            reason="intent is on the approval-only list",
            intent=intent,
            user_preference=user_preference,
        )

    # 5: confidence + risk gates (only reaches here for reversible non-destructive intents).
    if intent.risk == "medium":
        return Decision(
            verdict="require_approval",
            reason="medium-risk intents always require explicit approval in v0.1",
            intent=intent,
            user_preference=user_preference,
        )

    # risk == "low" and reversible.
    if intent.confidence >= AUTO_LOW_THRESHOLD:
        return Decision(
            verdict="auto_low_risk",
            reason=(f"low risk + reversible + confidence ≥ {AUTO_LOW_THRESHOLD}"),
            intent=intent,
            user_preference=user_preference,
        )
    if intent.confidence >= NOTIFY_THRESHOLD:
        return Decision(
            verdict="notify",
            reason=(f"low risk + reversible but confidence below {AUTO_LOW_THRESHOLD}"),
            intent=intent,
            user_preference=user_preference,
        )
    return Decision(
        verdict="require_approval",
        reason=f"confidence below notify threshold {NOTIFY_THRESHOLD}",
        intent=intent,
        user_preference=user_preference,
    )
