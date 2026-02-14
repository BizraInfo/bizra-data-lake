"""
Token Types — Data Classes, Enums, and Constants
=================================================

type definitions for the BIZRA token system.

Token Design (from genesis.json):
    SEED (BZR_S): Utility token — earned via PoI, used for compute/storage/tools
    BLOOM (BZR_B): Governance token — earned via SEED staking, used for voting
    IMPT: Reputation score — non-transferable, lifetime impact accumulator

Standing on Giants:
- Nakamoto (2008): UTXO model for transaction integrity
- Lamport (1978): Logical clocks for deterministic ordering
- Merkle (1979): Hash chains for tamper detection
- Shannon (1948): SNR for quality measurement
- Al-Ghazali (1058-1111): Proportional justice in distribution
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

# =============================================================================
# CONSTANTS (from genesis.json token_system)
# =============================================================================

SEED_SYMBOL = "BZR_S"
BLOOM_SYMBOL = "BZR_B"
IMPT_SYMBOL = "IMPT"

SEED_SUPPLY_CAP_PER_YEAR = 1_000_000
BLOOM_GOVERNANCE_QUORUM = 0.5

# Genesis epoch — the first token distribution period
GENESIS_EPOCH_ID = "epoch-0-genesis"

# Founder allocation — Node0's share of genesis SEED
# 10% of first year supply = 100,000 SEED for 3 years of foundational work
FOUNDER_GENESIS_ALLOCATION = 100_000

# System treasury initial allocation
SYSTEM_TREASURY_ALLOCATION = 50_000

# Computational Zakat — 2.5% of all mints go to community fund
ZAKAT_RATE = 0.025

# Minimum Ihsan score for token operations
IHSAN_THRESHOLD = UNIFIED_IHSAN_THRESHOLD

# Domain prefix for token digests (distinct from PCI domain)
TOKEN_DOMAIN_PREFIX = "bizra-token-v1:"

# =============================================================================
# ENUMS
# =============================================================================


class TokenType(Enum):
    """Token types in the BIZRA ecosystem."""

    SEED = "SEED"  # Utility — earned via PoI
    BLOOM = "BLOOM"  # Governance — earned via SEED staking
    IMPT = "IMPT"  # Reputation — non-transferable


class TokenOp(Enum):
    """Token operation types."""

    MINT = "mint"  # New tokens created (from PoI or staking)
    TRANSFER = "transfer"  # Tokens moved between accounts
    BURN = "burn"  # Tokens destroyed
    STAKE = "stake"  # Tokens locked for staking
    UNSTAKE = "unstake"  # Tokens released from staking
    ZAKAT = "zakat"  # Computational zakat (2.5%)
    GENESIS_MINT = "genesis_mint"  # Genesis allocation (one-time)


# =============================================================================
# DATA CLASSES
# =============================================================================


def _utc_now_iso() -> str:
    """UTC now in ISO 8601."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class TokenBalance:
    """Balance record for a single account + token type."""

    account_id: str
    token_type: TokenType
    balance: float
    staked: float = 0.0
    last_updated: str = field(default_factory=_utc_now_iso)

    @property
    def available(self) -> float:
        """Available (non-staked) balance."""
        return self.balance - self.staked

    def to_dict(self) -> dict[str, Any]:
        """Serialize token balance to dictionary including available amount."""
        return {
            "account_id": self.account_id,
            "token_type": self.token_type.value,
            "balance": self.balance,
            "staked": self.staked,
            "available": self.available,
            "last_updated": self.last_updated,
        }


@dataclass
class TransactionEntry:
    """A single token transaction in the ledger.

    Every transaction is hash-chained to the previous one (Merkle 1979).
    Every transaction includes a nonce for replay protection (Nakamoto 2008).
    """

    tx_id: str = field(default_factory=lambda: secrets.token_hex(16))
    sequence: int = 0
    op: TokenOp = TokenOp.MINT
    token_type: TokenType = TokenType.SEED
    from_account: str = ""  # Empty for mints
    to_account: str = ""  # Empty for burns
    amount: float = 0.0
    memo: str = ""
    epoch_id: str = ""
    poi_score: float = 0.0  # PoI score that justified this mint
    prev_hash: str = "0" * 64  # Hash chain link
    tx_hash: str = ""  # BLAKE3 hash of canonical transaction (SEC-001)
    signature: str = ""  # Ed25519 signature of tx_hash
    signer_pubkey: str = ""  # Public key of signer
    nonce: str = field(default_factory=lambda: secrets.token_hex(8))
    timestamp: str = field(default_factory=_utc_now_iso)

    def canonical_bytes(self) -> bytes:
        """Canonical JSON for hashing (deterministic, sorted keys)."""
        signable = {
            "amount": self.amount,
            "epoch_id": self.epoch_id,
            "from_account": self.from_account,
            "memo": self.memo,
            "nonce": self.nonce,
            "op": self.op.value,
            "poi_score": self.poi_score,
            "prev_hash": self.prev_hash,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "to_account": self.to_account,
            "token_type": self.token_type.value,
            "tx_id": self.tx_id,
        }
        return json.dumps(signable, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )

    def compute_hash(self) -> str:
        """Compute BLAKE3 hash of canonical transaction (SEC-001)."""
        from core.proof_engine.canonical import hex_digest

        prefixed = TOKEN_DOMAIN_PREFIX.encode("utf-8") + self.canonical_bytes()
        return hex_digest(prefixed)

    def to_dict(self) -> dict[str, Any]:
        """Serialize transaction entry to dictionary for ledger storage."""
        return {
            "tx_id": self.tx_id,
            "seq": self.sequence,
            "op": self.op.value,
            "token_type": self.token_type.value,
            "from": self.from_account,
            "to": self.to_account,
            "amount": self.amount,
            "memo": self.memo,
            "epoch_id": self.epoch_id,
            "poi_score": self.poi_score,
            "prev_hash": self.prev_hash,
            "tx_hash": self.tx_hash,
            "signature": self.signature,
            "signer_pubkey": self.signer_pubkey,
            "nonce": self.nonce,
            "ts": self.timestamp,
        }

    def to_jsonl(self) -> str:
        """Serialize to compact JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransactionEntry":
        """Deserialize from dictionary."""
        return cls(
            tx_id=data["tx_id"],
            sequence=data["seq"],
            op=TokenOp(data["op"]),
            token_type=TokenType(data["token_type"]),
            from_account=data.get("from", ""),
            to_account=data.get("to", ""),
            amount=data["amount"],
            memo=data.get("memo", ""),
            epoch_id=data.get("epoch_id", ""),
            poi_score=data.get("poi_score", 0.0),
            prev_hash=data["prev_hash"],
            tx_hash=data["tx_hash"],
            signature=data.get("signature", ""),
            signer_pubkey=data.get("signer_pubkey", ""),
            nonce=data["nonce"],
            timestamp=data["ts"],
        )


@dataclass
class TokenReceipt:
    """Receipt for a completed token operation.

    Every operation produces a receipt — fail-closed, audit-ready.
    """

    success: bool
    tx_entry: Optional[TransactionEntry] = None
    balance_after: float = 0.0
    error: Optional[str] = None
    receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize token receipt to dictionary, including transaction and error if present."""
        result: dict[str, Any] = {
            "success": self.success,
            "balance_after": self.balance_after,
            "receipt_hash": self.receipt_hash,
        }
        if self.tx_entry:
            result["transaction"] = self.tx_entry.to_dict()
        if self.error:
            result["error"] = self.error
        return result


__all__ = [
    # Constants
    "SEED_SYMBOL",
    "BLOOM_SYMBOL",
    "IMPT_SYMBOL",
    "SEED_SUPPLY_CAP_PER_YEAR",
    "BLOOM_GOVERNANCE_QUORUM",
    "GENESIS_EPOCH_ID",
    "FOUNDER_GENESIS_ALLOCATION",
    "SYSTEM_TREASURY_ALLOCATION",
    "ZAKAT_RATE",
    "IHSAN_THRESHOLD",
    "TOKEN_DOMAIN_PREFIX",
    # Enums
    "TokenType",
    "TokenOp",
    # Data classes
    "TokenBalance",
    "TransactionEntry",
    "TokenReceipt",
]
