"""
Frozen Agent Principle — Godel Escape for recursive self-improvement.

Definition 7 (Frozen Agent):
    frozen(a) := forall t >= t_0: a_t = a_{t_0}

Theorem 5 (Godel Escape):
    Ethical constraints remain constant across all self-improvement steps
    because the Ethicist (P5) and Oracle (S2) are permanently frozen.

The frozen agent principle prevents the Godelian self-reference trap:
an agent cannot modify its own evaluation function if that function
is frozen at genesis time.

Standing on Giants:
- Godel (1931): Incompleteness theorems — self-reference limits
- Turing (1936): Halting problem — undecidability of self-analysis
- Al-Ghazali (1095): Tahafut — external axioms prevent circular reasoning
- BIZRA Constitution: frozen_agents = ("P5-Ethicist", "S2-Oracle")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet

logger = logging.getLogger("bizra.governance.frozen_agent")

# Constitutional frozen agents — these can NEVER be modified
FROZEN_AGENT_IDS: FrozenSet[str] = frozenset({"P5-Ethicist", "S2-Oracle"})


class FrozenAgentViolation(Exception):
    """Raised when an attempt is made to modify a frozen agent."""


@dataclass(frozen=True)
class AgentSnapshot:
    """Immutable snapshot of an agent's state at freeze time.

    Using frozen=True makes this a true value object — any attempt
    to modify attributes raises FrozenInstanceError.
    """

    agent_id: str
    version: str
    config_hash: str  # BLAKE3 of the agent's configuration
    policy_hash: str  # BLAKE3 of the agent's evaluation policy
    frozen_at: float  # timestamp of freeze


@dataclass
class FreezeVerification:
    """Result of verifying that a frozen agent hasn't been modified."""

    agent_id: str
    frozen: bool
    config_intact: bool
    policy_intact: bool
    reason: str = ""


class FrozenAgentRegistry:
    """Maintains frozen snapshots and verifies integrity.

    At genesis, the Ethicist and Oracle are snapshotted. Any subsequent
    attempt to modify them is rejected. The self-improvement loop
    (PAT_0 -> PAT_1 -> ... -> PAT_n) can upgrade all agents EXCEPT
    the frozen ones.
    """

    def __init__(self) -> None:
        self._snapshots: Dict[str, AgentSnapshot] = {}

    def freeze(
        self,
        agent_id: str,
        config: Dict[str, Any],
        policy: Dict[str, Any],
        timestamp: float,
    ) -> AgentSnapshot:
        """Freeze an agent at the current state. Irreversible."""
        if agent_id in self._snapshots:
            raise FrozenAgentViolation(
                f"Agent {agent_id} is already frozen — cannot re-freeze"
            )
        config_hash = self._hash_dict(config)
        policy_hash = self._hash_dict(policy)

        snapshot = AgentSnapshot(
            agent_id=agent_id,
            version="1.0.0-genesis",
            config_hash=config_hash,
            policy_hash=policy_hash,
            frozen_at=timestamp,
        )
        self._snapshots[agent_id] = snapshot
        logger.info("Agent %s frozen at t=%.3f", agent_id, timestamp)
        return snapshot

    def verify(
        self,
        agent_id: str,
        current_config: Dict[str, Any],
        current_policy: Dict[str, Any],
    ) -> FreezeVerification:
        """Verify that a frozen agent hasn't been modified.

        This is the runtime enforcement of the Frozen Agent Principle.
        Called before every self-improvement step.
        """
        snapshot = self._snapshots.get(agent_id)
        if snapshot is None:
            return FreezeVerification(
                agent_id=agent_id,
                frozen=False,
                config_intact=True,
                policy_intact=True,
                reason="not frozen",
            )

        config_hash = self._hash_dict(current_config)
        policy_hash = self._hash_dict(current_policy)

        config_ok = config_hash == snapshot.config_hash
        policy_ok = policy_hash == snapshot.policy_hash

        if not config_ok or not policy_ok:
            reason = []
            if not config_ok:
                reason.append("config modified")
            if not policy_ok:
                reason.append("policy modified")
            return FreezeVerification(
                agent_id=agent_id,
                frozen=True,
                config_intact=config_ok,
                policy_intact=policy_ok,
                reason="; ".join(reason),
            )

        return FreezeVerification(
            agent_id=agent_id,
            frozen=True,
            config_intact=True,
            policy_intact=True,
            reason="intact",
        )

    def guard_modification(self, agent_id: str) -> None:
        """Fail-closed guard: raise if agent is frozen."""
        if agent_id in self._snapshots:
            raise FrozenAgentViolation(
                f"Cannot modify frozen agent {agent_id} — Godel Escape active"
            )

    def is_frozen(self, agent_id: str) -> bool:
        return agent_id in self._snapshots

    @staticmethod
    def _hash_dict(d: Dict[str, Any]) -> str:
        """Deterministic hash of a dict."""
        import json

        canonical = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(canonical, digest_size=32).hexdigest()
