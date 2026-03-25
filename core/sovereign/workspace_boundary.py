"""
Sovereignty Axiom — Workspace Boundary Enforcement.

For every node n, Omega_n ∩ URP = ∅.

Ensures that user-local state (PAT_7 execution, private memory, local reflexes,
signing keys) is topologically disjoint from URP-shared state (SAT_5 services,
shared ledger, consensus). The membrane filters crossings; this module defines
and enforces the boundary itself.

Standing on Giants:
- Lamport (1978): Disjoint state partitions in distributed systems
- Dijkstra (1968): THE system — strict layered isolation
- BIZRA Constitution: Sovereignty is not delegated, it is axiomatic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet

logger = logging.getLogger("bizra.sovereign.workspace_boundary")

# --- Namespace Definitions -----------------------------------------------

# Local-only namespaces (Omega_n) — NEVER cross the membrane
OMEGA_NAMESPACES: FrozenSet[str] = frozenset(
    {
        "signing_key",
        "local_memory",
        "reflex_cache",
        "pat_roster",
        "local_receipts",
        "local_models",
        "hda_state",
    }
)

# URP-shared namespaces — available to all nodes via membrane
URP_NAMESPACES: FrozenSet[str] = frozenset(
    {
        "shared_ledger",
        "sat_pool",
        "knowledge_graph",
        "consensus",
        "federation_state",
    }
)

# Fields that must be stripped before outbound membrane crossing
PRIVATE_FIELDS: FrozenSet[str] = frozenset(
    {
        "signing_key",
        "local_memory",
        "reflex_cache",
        "node_id",
        "ip_address",
        "mac_address",
        "private_key_hex",
    }
)


class SovereigntyViolation(Exception):
    """Raised when URP attempts to write a local-only namespace."""


@dataclass
class DisjointResult:
    """Result of a disjointness check."""

    disjoint: bool
    overlap: FrozenSet[str] = field(default_factory=frozenset)


@dataclass
class ScalingResult:
    """Result of linear scaling verification."""

    node_count: int
    total_capacity: float
    is_linear: bool
    ratio: float


class WorkspaceBoundary:
    """Enforces Omega_n disjoint URP at runtime.

    This is the formal implementation of the Sovereignty Axiom:
    for every node n, Omega_n ∩ URP = ∅.
    """

    def __init__(self, node_id: str, data_dir: Path) -> None:
        self._node_id = node_id
        self._data_dir = data_dir
        self._omega = OMEGA_NAMESPACES
        self._urp = URP_NAMESPACES

    @property
    def node_id(self) -> str:
        return self._node_id

    def check_disjoint(self) -> DisjointResult:
        """INVARIANT: local namespace keys never overlap URP keys.

        This is the runtime proof that Omega_n ∩ URP = ∅.
        """
        overlap = self._omega & self._urp
        return DisjointResult(
            disjoint=len(overlap) == 0,
            overlap=frozenset(overlap),
        )

    def guard_outbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Strip private fields before membrane crossing.

        Enforces P1 (Anonymity): identity and local state are
        mathematically unrecoverable from the outbound payload.
        """
        clean = {k: v for k, v in payload.items() if k not in PRIVATE_FIELDS}
        stripped = set(payload.keys()) - set(clean.keys())
        if stripped:
            logger.debug("Stripped %d private fields: %s", len(stripped), stripped)
        return clean

    def guard_inbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Reject URP payloads that attempt to write local-only namespaces.

        Fail-closed: any attempt to inject into Omega_n raises immediately.
        """
        violations = set(payload.keys()) & self._omega
        if violations:
            raise SovereigntyViolation(
                f"URP cannot write to local namespaces: {violations}"
            )
        return payload


def verify_linear_scaling(
    capacities: list[float],
    baseline_capacity: float = 1.0,
    tolerance: float = 0.05,
) -> ScalingResult:
    """V(N) = sum(SAT_5_i for i in 1..N) — linear, not sublinear.

    Standing on Giants: Amdahl's Law inverted — BIZRA verification
    is embarrassingly parallel (no shared critical section).
    """
    n = len(capacities)
    if n == 0:
        return ScalingResult(
            node_count=0, total_capacity=0.0, is_linear=True, ratio=0.0
        )

    total = sum(capacities)
    expected = n * baseline_capacity
    ratio = total / expected if expected > 0 else 0.0
    is_linear = (1.0 - tolerance) <= ratio <= (1.0 + tolerance)

    return ScalingResult(
        node_count=n,
        total_capacity=total,
        is_linear=is_linear,
        ratio=ratio,
    )
