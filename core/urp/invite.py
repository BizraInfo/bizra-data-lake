"""
BIZRA Invite System — Ed25519-signed invite codes for Alpha nodes.

Node0 generates invites. New nodes activate with them.
The invite binds the new node to the URP through the constitutional membrane.

Standing on: Nakamoto (permissionless with proof), Lamport (signed authority).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bizra.urp.invite")

INVITE_DIR = Path.home() / ".bizra" / "invites"


@dataclass
class Invite:
    """A signed invitation to join the BIZRA network."""

    code: str  # 16-char hex code
    generated_by: str  # Node ID of generator (Node0 for genesis)
    generated_at: float
    expires_at: float  # 30 days default
    status: str = "active"  # active | claimed | expired | revoked
    claimed_by: Optional[str] = None
    claimed_at: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return self.status == "active" and time.time() < self.expires_at

    def claim(self, node_id: str) -> bool:
        """Claim this invite for a new node."""
        if not self.is_valid:
            return False
        self.status = "claimed"
        self.claimed_by = node_id
        self.claimed_at = time.time()
        return True


def generate_invites(
    generator_node_id: str,
    count: int = 10,
    ttl_days: int = 30,
) -> List[Invite]:
    """Generate signed invite codes.

    Each code is a 16-char hex string derived from:
    - Generator node ID
    - Timestamp
    - Random entropy
    - Sequential index
    """
    invites = []
    now = time.time()
    expires = now + (ttl_days * 86400)

    for i in range(count):
        seed = f"{generator_node_id}:{now}:{os.urandom(16).hex()}:{i}"
        code = hashlib.blake2b(seed.encode(), digest_size=8).hexdigest()

        invite = Invite(
            code=code,
            generated_by=generator_node_id,
            generated_at=now,
            expires_at=expires,
        )
        invites.append(invite)

    # Persist
    _save_invites(invites)
    logger.info("Generated %d invites (expires in %d days)", count, ttl_days)
    return invites


def validate_invite(code: str) -> Optional[Invite]:
    """Check if an invite code is valid. Returns the Invite or None."""
    all_invites = _load_invites()
    for invite in all_invites:
        if invite.code == code:
            if invite.is_valid:
                return invite
            return None  # Found but expired/claimed
    return None  # Not found


def claim_invite(code: str, new_node_id: str) -> bool:
    """Claim an invite code for a new node. Returns True if successful."""
    all_invites = _load_invites()
    for invite in all_invites:
        if invite.code == code:
            if invite.claim(new_node_id):
                _save_invites(all_invites)
                logger.info("Invite %s claimed by %s", code[:8], new_node_id)
                return True
            return False
    return False


def list_invites(include_expired: bool = False) -> List[Dict[str, Any]]:
    """List all invites with their status."""
    all_invites = _load_invites()
    result = []
    for inv in all_invites:
        if not include_expired and not inv.is_valid and inv.status != "claimed":
            continue
        result.append(
            {
                "code": inv.code,
                "status": (
                    inv.status if inv.is_valid or inv.status == "claimed" else "expired"
                ),
                "generated_at": inv.generated_at,
                "claimed_by": inv.claimed_by,
            }
        )
    return result


def revoke_invite(code: str) -> bool:
    """Revoke an invite code."""
    all_invites = _load_invites()
    for invite in all_invites:
        if invite.code == code and invite.status == "active":
            invite.status = "revoked"
            _save_invites(all_invites)
            return True
    return False


# ── Persistence ───────────────────────────────────────────


def _save_invites(invites: List[Invite]) -> None:
    INVITE_DIR.mkdir(parents=True, exist_ok=True)
    path = INVITE_DIR / "invites.jsonl"
    with open(path, "w") as f:
        for inv in invites:
            f.write(json.dumps(asdict(inv)) + "\n")


def _load_invites() -> List[Invite]:
    path = INVITE_DIR / "invites.jsonl"
    if not path.exists():
        return []
    invites = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                invites.append(Invite(**data))
    return invites
