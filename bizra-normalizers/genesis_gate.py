"""GENESIS-stage fail-closed gate for stereoscopic report validation."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class NodeMaturityStage(enum.Enum):
    """Progressive maturity stages for BIZRA node onboarding.

    Every node starts as a SEED (zero data) and graduates through
    stages as interaction history accumulates.  Gate thresholds relax
    for early stages so that new users are never locked out.

    Stage progression:
        SEED    -> zero-data cold start, immediate pass
        SPROUT  -> 10+ TEACH atoms ingested
        GROWING -> 100+ messages OR 1+ provider import
        ROOTED  -> 3+ provider imports, full gate enforcement
    """

    SEED = "seed"
    SPROUT = "sprout"
    GROWING = "growing"
    ROOTED = "rooted"


# Ordered list for stage comparison helpers.
_STAGE_ORDER: tuple[NodeMaturityStage, ...] = (
    NodeMaturityStage.SEED,
    NodeMaturityStage.SPROUT,
    NodeMaturityStage.GROWING,
    NodeMaturityStage.ROOTED,
)


@dataclass(frozen=True)
class GenesisGateConfig:
    """Gate thresholds for allowing downstream GENESIS compilation.

    Provider sets:
    - `available_providers`: platforms with confirmed export capability today.
      CV is computed against this set.  Gate passes when all are present.
    - `target_providers`: aspirational CORE8 target.  Missing members are
      reported as INFO-level collection gaps, not gate-blocking failures.
    """

    min_cv: float = 1.0
    min_elite_nodes: int = 1
    min_nodes: int = 1
    fail_closed: bool = True
    required_providers: tuple[str, ...] = ()  # back-compat alias → available
    available_providers: tuple[str, ...] = ()
    target_providers: tuple[str, ...] = ()

    # --- Factory class methods for tiered genesis progression ---

    @classmethod
    def for_cold_start(cls) -> GenesisGateConfig:
        """Seed stage -- zero-data users pass immediately.

        This is the most permissive configuration: no CV, no node
        minimums.  ``fail_closed`` remains True so the gate still
        evaluates deterministically; it simply has nothing to block on.
        """
        return cls(min_cv=0.0, min_nodes=0, min_elite_nodes=0, fail_closed=True)

    @classmethod
    def for_stage(
        cls,
        stage: NodeMaturityStage,
        available_providers: tuple[str, ...] = (),
        target_providers: tuple[str, ...] = (),
    ) -> GenesisGateConfig:
        """Get gate config appropriate for the given maturity stage.

        Parameters
        ----------
        stage:
            The node's current maturity stage.
        available_providers:
            Platforms with confirmed export capability (gate-blocking).
        target_providers:
            Aspirational CORE8 target (advisory only).

        Returns
        -------
        GenesisGateConfig
            A frozen config with thresholds calibrated to *stage*.
        """
        stage_params: dict[NodeMaturityStage, dict[str, Any]] = {
            NodeMaturityStage.SEED: {
                "min_cv": 0.0,
                "min_nodes": 0,
                "min_elite_nodes": 0,
            },
            NodeMaturityStage.SPROUT: {
                "min_cv": 0.0,
                "min_nodes": 1,
                "min_elite_nodes": 0,
            },
            NodeMaturityStage.GROWING: {
                "min_cv": 0.5,
                "min_nodes": 3,
                "min_elite_nodes": 1,
            },
            NodeMaturityStage.ROOTED: {
                "min_cv": 1.0,
                "min_nodes": 5,
                "min_elite_nodes": 1,
            },
        }
        params = stage_params[stage]
        return cls(
            min_cv=params["min_cv"],
            min_nodes=params["min_nodes"],
            min_elite_nodes=params["min_elite_nodes"],
            fail_closed=True,
            available_providers=available_providers,
            target_providers=target_providers,
        )


@dataclass(frozen=True)
class GenesisGateVerdict:
    """Gate result with explicit reason codes for auditability."""

    passed: bool
    reasons: tuple[str, ...]
    cv: float
    node_count: int
    elite_count: int
    min_cv: float
    min_nodes: int
    min_elite_nodes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "cv": self.cv,
            "node_count": self.node_count,
            "elite_count": self.elite_count,
            "min_cv": self.min_cv,
            "min_nodes": self.min_nodes,
            "min_elite_nodes": self.min_elite_nodes,
        }


def evaluate_genesis_gate(
    report: dict[str, Any], config: GenesisGateConfig
) -> GenesisGateVerdict:
    """Evaluate report against fail-closed GENESIS thresholds.

    Gate-blocking checks use `available_providers` (or `required_providers`
    for back-compat).  `target_providers` generate advisory INFO reasons
    that do not block the gate.
    """
    cv = float(report.get("cv", 0.0))
    node_count = int(report.get("node_count", len(report.get("nodes") or [])))
    elite_count = int(report.get("elite_count", len(report.get("elite_nodes") or [])))
    coverage = {
        str(provider).strip().lower()
        for provider in (report.get("provider_coverage") or [])
        if str(provider).strip()
    }

    # Resolve which set is gate-blocking.
    gate_providers = set(
        p.lower()
        for p in (config.available_providers or config.required_providers)
        if p.strip()
    )
    # Recompute CV against the gate-blocking set if it's non-empty.
    if gate_providers:
        gate_present = coverage & gate_providers
        effective_cv = round(len(gate_present) / len(gate_providers), 4)
    else:
        effective_cv = cv

    reasons: list[str] = []

    # Gate-blocking: available/required providers.
    for provider in sorted(gate_providers):
        if provider not in coverage:
            reasons.append(f"MISSING_PROVIDER_EXPORT:{provider}")

    # Advisory: target (CORE8) collection gaps — never block gate.
    target_providers = {p.lower() for p in config.target_providers if p.strip()}
    for provider in sorted(target_providers - gate_providers):
        if provider not in coverage:
            reasons.append(f"INFO:COLLECTION_GAP:{provider}")
    if effective_cv < config.min_cv:
        reasons.append(f"CV_BELOW_THRESHOLD:{effective_cv:.4f}<{config.min_cv:.4f}")
    if node_count < config.min_nodes:
        reasons.append(f"NODE_COUNT_BELOW_THRESHOLD:{node_count}<{config.min_nodes}")
    if elite_count < config.min_elite_nodes:
        reasons.append(
            f"ELITE_COUNT_BELOW_THRESHOLD:{elite_count}<{config.min_elite_nodes}"
        )

    # Only non-INFO reasons block the gate.
    blocking = [r for r in reasons if not r.startswith("INFO:")]
    passed = len(blocking) == 0
    if not passed and not config.fail_closed:
        passed = True
        reasons.append("FAIL_OPEN_OVERRIDE")

    return GenesisGateVerdict(
        passed=passed,
        reasons=tuple(reasons),
        cv=effective_cv,
        node_count=node_count,
        elite_count=elite_count,
        min_cv=config.min_cv,
        min_nodes=config.min_nodes,
        min_elite_nodes=config.min_elite_nodes,
    )


def determine_maturity_stage(
    atom_count: int = 0,
    message_count: int = 0,
    provider_count: int = 0,
) -> NodeMaturityStage:
    """Determine the appropriate maturity stage from runtime metrics.

    The function evaluates metrics from most mature to least mature,
    returning the highest stage the node qualifies for.

    Parameters
    ----------
    atom_count:
        Number of TEACH atoms ingested by this node.
    message_count:
        Total messages across all conversations.
    provider_count:
        Number of distinct provider imports completed.

    Returns
    -------
    NodeMaturityStage
        The highest maturity stage the node qualifies for.

    Examples
    --------
    >>> determine_maturity_stage(atom_count=0, message_count=0, provider_count=0)
    <NodeMaturityStage.SEED: 'seed'>

    >>> determine_maturity_stage(atom_count=15, message_count=0, provider_count=0)
    <NodeMaturityStage.SPROUT: 'sprout'>

    >>> determine_maturity_stage(atom_count=50, message_count=200, provider_count=1)
    <NodeMaturityStage.GROWING: 'growing'>

    >>> determine_maturity_stage(atom_count=500, message_count=1000, provider_count=5)
    <NodeMaturityStage.ROOTED: 'rooted'>
    """
    if provider_count >= 3:
        return NodeMaturityStage.ROOTED
    if message_count >= 100 or provider_count >= 1:
        return NodeMaturityStage.GROWING
    if atom_count >= 10:
        return NodeMaturityStage.SPROUT
    return NodeMaturityStage.SEED
