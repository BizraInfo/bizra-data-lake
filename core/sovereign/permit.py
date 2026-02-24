"""
Telescript Permit System — Python-native, mirrors Rust bizra-telescript
=====================================================================
Capability-scoped, time-limited, budget-constrained authority delegation.
This is the trust boundary for autonomous action. Every HDA action
requires a valid Permit with matching Capability + unexpired TTL + budget.

When the PyO3 bridge to bizra-telescript is ready, this module becomes
a thin wrapper around the Rust implementation. Until then, it provides
the exact same API contract using HMAC-SHA256 for signature integrity.

Standing on Giants:
- General Magic (Telescript, 1994): Permits = capability + intent + proof
- Lamport (1978): Hash-chained authority delegation
- Shannon (1948): Capability enum reduces channel noise to 6 discrete signals
- Al-Ghazali (1095): Budget constraint = ethical limit on autonomous action

Created: 2026-02-22 | BIZRA Permit v1.0
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DELEGATION_DEPTH = 7  # Matches Rust bizra-telescript
PERMIT_SIGNING_KEY_ENV = "BIZRA_PERMIT_SIGNING_KEY"
DEFAULT_TTL_SECONDS = 300  # 5 minutes — matches permit_guard.rs TTL
MAX_ACTIONS_PER_PERMIT = 30  # Matches permit_guard.rs rate limit


# ---------------------------------------------------------------------------
# Capability Enum — mirrors Rust Capability exactly
# ---------------------------------------------------------------------------


class Capability(IntEnum):
    """Capabilities that can be granted to agents.

    Mirrors bizra-telescript/src/lib.rs Capability enum 1:1.
    """

    GO = 0  # Travel to other places / switch window
    CREATE = 1  # Create new agents / open app
    MEET = 2  # Participate in meetings
    COMPUTE = 3  # Execute computations / type text
    STORE = 4  # File operations / clipboard
    NETWORK = 5  # Browser navigation / network calls


# Map HDA methods to required capabilities
HDA_CAPABILITY_MAP: dict[str, Capability] = {
    "open_app": Capability.CREATE,
    "switch_window": Capability.GO,
    "type_text": Capability.COMPUTE,
    "click_element": Capability.COMPUTE,
    "screenshot": Capability.STORE,
    "read_clipboard": Capability.STORE,
    "file_open": Capability.STORE,
    "browser_navigate": Capability.NETWORK,
}


# ---------------------------------------------------------------------------
# Authority — chain of delegation
# ---------------------------------------------------------------------------


@dataclass
class Authority:
    """Authority represents the chain of delegation for permits.

    Every permit traces back to the Genesis Authority (Node0).
    Mirrors bizra-telescript Authority struct.

    Standing on Giants: Lamport (hash-chained trust delegation)
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = ""
    delegated_from: Optional[str] = None  # Parent authority ID
    delegation_depth: int = 0
    chain_hash: str = ""  # BLAKE3 in Rust, SHA-256 here
    created_at: float = field(default_factory=time.time)

    @classmethod
    def genesis(cls) -> Authority:
        """Create the Genesis Authority (Node0 — root of all trust)."""
        auth = cls(
            name="Node0-Genesis",
            delegation_depth=0,
        )
        hasher = hashlib.sha256()
        hasher.update(b"BIZRA_GENESIS_AUTHORITY_NODE0")
        hasher.update(auth.id.encode())
        auth.chain_hash = hasher.hexdigest()
        return auth

    def delegate(self, name: str) -> Authority:
        """Delegate authority to a new entity.

        Raises ValueError if delegation depth exceeds MAX_DELEGATION_DEPTH.
        """
        new_depth = self.delegation_depth + 1
        if new_depth > MAX_DELEGATION_DEPTH:
            raise ValueError(
                f"Delegation depth {new_depth} exceeds max {MAX_DELEGATION_DEPTH}"
            )

        child = Authority(
            name=name,
            delegated_from=self.id,
            delegation_depth=new_depth,
        )
        hasher = hashlib.sha256()
        hasher.update(self.chain_hash.encode())
        hasher.update(child.id.encode())
        hasher.update(name.encode())
        child.chain_hash = hasher.hexdigest()
        return child

    def verify_chain(self) -> bool:
        """Verify the authority chain hash is valid."""
        if self.delegation_depth == 0:
            hasher = hashlib.sha256()
            hasher.update(b"BIZRA_GENESIS_AUTHORITY_NODE0")
            hasher.update(self.id.encode())
            return self.chain_hash == hasher.hexdigest()
        return len(self.chain_hash) == 64  # Basic format check

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "delegation_depth": self.delegation_depth,
            "chain_hash": self.chain_hash[:16] + "...",
        }


# ---------------------------------------------------------------------------
# ResourceBudget — action and token limits
# ---------------------------------------------------------------------------


@dataclass
class ResourceBudget:
    """Budget constraints for a permit.

    Standing on Giants: General Magic (resource-limited agents, 1994)
    """

    max_actions: int = MAX_ACTIONS_PER_PERMIT
    max_tokens: int = 4096
    actions_used: int = 0
    tokens_used: int = 0

    @property
    def actions_remaining(self) -> int:
        return max(0, self.max_actions - self.actions_used)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def exhausted(self) -> bool:
        return self.actions_remaining <= 0 or self.tokens_remaining <= 0

    def consume_action(self, tokens: int = 0) -> bool:
        """Consume one action from the budget. Returns False if exhausted."""
        if self.exhausted:
            return False
        self.actions_used += 1
        self.tokens_used += tokens
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_actions": self.max_actions,
            "actions_used": self.actions_used,
            "actions_remaining": self.actions_remaining,
            "max_tokens": self.max_tokens,
            "tokens_used": self.tokens_used,
        }


# ---------------------------------------------------------------------------
# Permit — the core trust boundary
# ---------------------------------------------------------------------------


@dataclass
class Permit:
    """Permit defines what an agent is allowed to do.

    Capability-scoped, time-limited, budget-constrained. Every HDA action
    requires a valid Permit. The signature is HMAC-SHA256 of the permit
    fields, keyed by the bridge signing secret.

    Mirrors bizra-telescript/src/lib.rs Permit struct 1:1.

    Standing on Giants:
    - General Magic (Telescript, 1994): The original mobile agent permit
    - Shannon: 6 capabilities = minimal signal set for desktop automation
    - Al-Ghazali: Budget = ethical constraint on autonomous reach
    """

    permit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    issuer: Authority = field(default_factory=Authority.genesis)
    capabilities: list[Capability] = field(default_factory=list)
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    signature: str = ""

    def __post_init__(self) -> None:
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + self.ttl_seconds

    @classmethod
    def create(
        cls,
        issuer: Authority,
        capabilities: list[Capability],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_actions: int = MAX_ACTIONS_PER_PERMIT,
        max_tokens: int = 4096,
        signing_key: Optional[str] = None,
    ) -> Permit:
        """Create a new signed permit.

        Args:
            issuer: The authority issuing this permit.
            capabilities: List of capabilities granted.
            ttl_seconds: Time-to-live in seconds.
            max_actions: Maximum actions allowed.
            max_tokens: Maximum inference tokens allowed.
            signing_key: HMAC key (defaults to env var).

        Returns:
            A signed Permit ready for verification.
        """
        permit = cls(
            issuer=issuer,
            capabilities=capabilities,
            budget=ResourceBudget(max_actions=max_actions, max_tokens=max_tokens),
            ttl_seconds=ttl_seconds,
        )
        permit.signature = permit._compute_signature(signing_key)
        return permit

    def verify(self, signing_key: Optional[str] = None) -> PermitVerification:
        """Verify the permit is valid (signature, expiry, budget).

        Returns a PermitVerification with detailed status.
        """
        # 1. Check expiry
        if time.time() > self.expires_at:
            return PermitVerification(
                valid=False,
                reason="Permit expired",
                permit_id=self.permit_id,
            )

        # 2. Check budget
        if self.budget.exhausted:
            return PermitVerification(
                valid=False,
                reason="Budget exhausted",
                permit_id=self.permit_id,
            )

        # 3. Check signature
        expected = self._compute_signature(signing_key)
        if not hmac.compare_digest(self.signature, expected):
            return PermitVerification(
                valid=False,
                reason="Invalid signature",
                permit_id=self.permit_id,
            )

        return PermitVerification(
            valid=True,
            reason="OK",
            permit_id=self.permit_id,
            remaining_actions=self.budget.actions_remaining,
            ttl_remaining=max(0.0, self.expires_at - time.time()),
        )

    def has_capability(self, capability: Capability) -> bool:
        """Check if this permit grants a specific capability."""
        return capability in self.capabilities

    def check_action(
        self, method: str, signing_key: Optional[str] = None
    ) -> PermitVerification:
        """Check if an HDA action is permitted.

        Verifies: signature + expiry + budget + capability match.
        """
        verification = self.verify(signing_key)
        if not verification.valid:
            return verification

        required_cap = HDA_CAPABILITY_MAP.get(method)
        if required_cap is None:
            return PermitVerification(
                valid=False,
                reason=f"Unknown HDA method: {method}",
                permit_id=self.permit_id,
            )

        if not self.has_capability(required_cap):
            return PermitVerification(
                valid=False,
                reason=f"Missing capability {required_cap.name} for {method}",
                permit_id=self.permit_id,
            )

        return verification

    def consume(self, tokens: int = 0) -> bool:
        """Consume one action from the budget. Returns False if exhausted."""
        return self.budget.consume_action(tokens)

    def _compute_signature(self, signing_key: Optional[str] = None) -> str:
        """Compute HMAC-SHA256 signature over permit fields."""
        key = (
            signing_key
            or os.getenv(PERMIT_SIGNING_KEY_ENV, "")
            or os.getenv("BIZRA_BRIDGE_TOKEN", "")
        )
        if not key:
            key = "bizra-dev-permit-key"  # Development fallback

        msg = (
            f"{self.permit_id}|"
            f"{self.issuer.chain_hash}|"
            f"{','.join(str(int(c)) for c in self.capabilities)}|"
            f"{self.ttl_seconds}|"
            f"{self.budget.max_actions}|"
            f"{self.created_at}"
        )
        return hmac.new(
            key.encode(), msg.encode(), hashlib.sha256
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "permit_id": self.permit_id,
            "issuer": self.issuer.to_dict(),
            "capabilities": [c.name for c in self.capabilities],
            "budget": self.budget.to_dict(),
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "signature": self.signature[:16] + "...",
        }


# ---------------------------------------------------------------------------
# PermitVerification — verification result
# ---------------------------------------------------------------------------


@dataclass
class PermitVerification:
    """Result of permit verification."""

    valid: bool = False
    reason: str = ""
    permit_id: str = ""
    remaining_actions: int = 0
    ttl_remaining: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "permit_id": self.permit_id,
            "remaining_actions": self.remaining_actions,
            "ttl_remaining": round(self.ttl_remaining, 1),
        }


# ---------------------------------------------------------------------------
# Convenience: Default Node0 permit for HDA actions
# ---------------------------------------------------------------------------


def create_hda_permit(
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    max_actions: int = MAX_ACTIONS_PER_PERMIT,
    signing_key: Optional[str] = None,
) -> Permit:
    """Create a default HDA permit with all desktop capabilities.

    This is the standard permit for Founder Ops Agent desktop actions.
    """
    genesis = Authority.genesis()
    ops_authority = genesis.delegate("founder-ops-agent")
    return Permit.create(
        issuer=ops_authority,
        capabilities=[
            Capability.GO,
            Capability.CREATE,
            Capability.COMPUTE,
            Capability.STORE,
            Capability.NETWORK,
        ],
        ttl_seconds=ttl_seconds,
        max_actions=max_actions,
        signing_key=signing_key,
    )
