"""Reward shaping and token mint bridges for agent reinforcement learning."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from core.integration.constants import IHSAN_THRESHOLD, SNR_THRESHOLD
from core.sovereign.adl_kernel import ADL_GINI_THRESHOLD, calculate_gini_detailed
from core.token.types import TokenReceipt, TokenType

_SNR_WEIGHT = 0.30
_IHSAN_WEIGHT = 0.25
_EFFICIENCY_WEIGHT = 0.15
_FEEDBACK_WEIGHT = 0.20


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _verified_impact_eligible(metrics: dict[str, Any]) -> bool:
    """Return True only when canonical quality gates permit impact settlement.

    Missing, malformed, or below-threshold evidence fails closed.  The thresholds
    are imported from the integration constants single source of truth so token
    economics cannot silently drift from the constitutional quality gates.
    """
    try:
        snr = float(metrics.get("snr", metrics.get("snr_score", 0.0)))
        ihsan = float(metrics.get("ihsan", metrics.get("ihsan_score", 0.0)))
    except (TypeError, ValueError):
        return False
    return snr >= SNR_THRESHOLD and ihsan >= IHSAN_THRESHOLD


def token_efficiency_reward(tokens_used: int, quality: float) -> float:
    """Quality-per-token signal with logistic squashing into `[0, 1]`."""
    if tokens_used <= 0:
        return 0.0

    quality_score = _clamp01(quality)
    per_1k = quality_score / (tokens_used / 1000.0)
    midpoint = 0.5
    slope = 8.0
    exponent = -slope * (per_1k - midpoint)
    exponent = max(min(exponent, 80.0), -80.0)
    return 1.0 / (1.0 + math.exp(exponent))


def composite_reward(
    mission_result: dict[str, Any] | None = None,
    **legacy_kwargs: Any,
) -> float:
    """Compute bounded composite reward from verified mission metrics.

    Formula after quality admission:
        0.30*SNR + 0.25*Ihsan + 0.15*Efficiency + 0.20*UserFeedback - penalties

    Economic settlement fails closed to ``0.0`` unless both canonical SNR and
    Ihsan floors are met.  This prevents efficiency or feedback from creating a
    positive reward for a rejected/quarantined mission.

    The function keeps backward compatibility with prior call sites that passed
    keyword metrics directly.
    """
    metrics = dict(mission_result or {})
    metrics.update(legacy_kwargs)

    if not _verified_impact_eligible(metrics):
        return 0.0

    snr = _clamp01(metrics.get("snr", metrics.get("snr_score", 0.0)))
    ihsan = _clamp01(metrics.get("ihsan", metrics.get("ihsan_score", 0.0)))

    efficiency_raw = metrics.get("efficiency")
    if efficiency_raw is None:
        tokens_used = int(metrics.get("tokens_used", metrics.get("total_tokens", 0)))
        quality = float(metrics.get("quality", snr))
        efficiency = token_efficiency_reward(tokens_used=tokens_used, quality=quality)
    else:
        efficiency = _clamp01(float(efficiency_raw))

    feedback = _clamp01(metrics.get("user_feedback", 0.5))
    penalties = _clamp01(metrics.get("penalties", 0.0))

    reward = (
        _SNR_WEIGHT * snr
        + _IHSAN_WEIGHT * ihsan
        + _EFFICIENCY_WEIGHT * efficiency
        + _FEEDBACK_WEIGHT * feedback
        - penalties
    )
    return _clamp01(reward)


def _seed_holdings_from_minter(minter: Any, agent_ids: list[str]) -> list[float]:
    holdings: list[float] = []
    if minter is None or getattr(minter, "ledger", None) is None:
        return holdings

    for agent_id in agent_ids:
        try:
            bal = minter.ledger.get_balance(agent_id, TokenType.SEED)
            holdings.append(float(getattr(bal, "balance", 0.0)))
        except Exception:  # noqa: BLE001 — boundary boundary
            holdings.append(0.0)
    return holdings


def compute_agent_reward(
    agent_id: str,
    mission_result: dict[str, Any],
    minter: Any,
    emission_gate: Any,
    epoch_id: str,
) -> TokenReceipt:
    """Mint SEED only for canonically verified mission impact."""
    if minter is None:
        return TokenReceipt(success=False, error="minter_unavailable")
    if not _verified_impact_eligible(mission_result):
        return TokenReceipt(success=False, error="unverified_impact")

    reward_score = composite_reward(mission_result)
    requested_seed = float(mission_result.get("seed_base", 100.0)) * reward_score
    gated_seed = requested_seed

    if emission_gate is not None:
        account_ids = []
        try:
            account_ids = list(minter.ledger.list_accounts())
        except Exception:  # noqa: BLE001 — boundary boundary
            account_ids = []
        if agent_id not in account_ids:
            account_ids.append(agent_id)
        holdings = _seed_holdings_from_minter(minter, account_ids)
        try:
            gate = emission_gate.compute_gated_emission(
                requested_amount=requested_seed,
                current_holdings=holdings,
            )
            gated_seed = float(gate.get("gated_amount", requested_seed))
        except Exception:  # noqa: BLE001 — boundary boundary
            gated_seed = requested_seed

    if gated_seed <= 0:
        return TokenReceipt(success=False, error="zero_gated_emission")

    return minter.mint_seed(
        to_account=agent_id,
        amount=gated_seed,
        epoch_id=epoch_id,
        poi_score=reward_score,
        memo="RL composite reward mint",
    )


def update_agent_reputation(
    agent_id: str,
    reward_score: float,
    minter: Any,
) -> TokenReceipt:
    """Mint IMPT with diminishing returns only for positive verified reward."""
    if minter is None:
        return TokenReceipt(success=False, error="minter_unavailable")

    bounded = _clamp01(reward_score)
    if bounded <= 0.0:
        return TokenReceipt(success=False, error="unverified_impact")

    amount = math.sqrt(bounded) * 10.0
    epoch_id = datetime.now(timezone.utc).strftime("epoch-%Y%m%d")

    return minter.mint_impt(
        to_account=agent_id,
        amount=amount,
        epoch_id=epoch_id,
        poi_score=bounded,
        memo="RL reputation update",
    )


def enforce_agent_gini(
    minter: Any,
    agent_ids: list[str],
    threshold: float = ADL_GINI_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate Gini compliance across current agent SEED holdings."""
    holdings_map: dict[str, float] = {}
    if minter is None or getattr(minter, "ledger", None) is None:
        return {
            "gini": 0.0,
            "threshold": threshold,
            "compliant": True,
            "holdings": holdings_map,
            "reason": "minter_unavailable",
        }

    for agent_id in agent_ids:
        try:
            bal = minter.ledger.get_balance(agent_id, TokenType.SEED)
            holdings_map[agent_id] = float(getattr(bal, "balance", 0.0))
        except Exception:  # noqa: BLE001 — boundary boundary
            holdings_map[agent_id] = 0.0

    detail = calculate_gini_detailed(list(holdings_map.values()), threshold=threshold)
    return {
        "gini": detail.gini,
        "threshold": threshold,
        "compliant": detail.passes_threshold,
        "alert_triggered": detail.alert_triggered,
        "holdings": holdings_map,
    }


__all__ = [
    "composite_reward",
    "compute_agent_reward",
    "enforce_agent_gini",
    "token_efficiency_reward",
    "update_agent_reputation",
]
