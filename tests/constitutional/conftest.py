"""
Shared Fixtures for Constitutional Tests
═════════════════════════════════════════

TDD anchors from Phase 67.07 specification.

Standing on Giants:
- Beck (2002): Test-Driven Development by Example
"""

from __future__ import annotations

import random

import pytest

from core.constitutional.fixed_point import fp
from core.constitutional.types import ActionReceipt, WalletState


@pytest.fixture
def deterministic_seed():
    """All chaos tests use seed 42 for reproducibility."""
    random.seed(42)
    yield


@pytest.fixture
def quality_receipt() -> ActionReceipt:
    """A receipt that passes both intent gate and ihsan check."""
    return ActionReceipt(
        receipt_id=b"\x01" * 32,
        actor_id=b"\x02" * 32,
        action_type="contribution",
        timestamp=1741392000000,
        intent_score=fp(0.98),
        efficiency_score=fp(0.96),
        impact_score=fp(0.97),
        reproducibility_score=fp(0.95),
        oracle_signature=b"\x03" * 64,
        metadata_hash=b"\x04" * 32,
        co_actors=(),
    )


@pytest.fixture
def low_intent_receipt() -> ActionReceipt:
    """A receipt that fails the Al-Ghazali intent gate."""
    return ActionReceipt(
        receipt_id=b"\x05" * 32,
        actor_id=b"\x06" * 32,
        action_type="spam",
        timestamp=1741392000000,
        intent_score=fp(0.50),  # Below 0.90 floor
        efficiency_score=fp(0.90),
        impact_score=fp(0.90),
        reproducibility_score=fp(0.90),
        oracle_signature=b"\x07" * 64,
        metadata_hash=b"\x08" * 32,
        co_actors=(),
    )


@pytest.fixture
def newcomer_wallet() -> WalletState:
    """A freshly initialized wallet with zero balance."""
    return WalletState(
        node_id=b"\x10" * 32,
        seed_balance=0,
        bloom_balance=0,
        created_at=1741392000000,
        last_active=1741392000000,
    )


@pytest.fixture
def wealthy_wallet() -> WalletState:
    """A wallet with substantial balance."""
    return WalletState(
        node_id=b"\x20" * 32,
        seed_balance=fp(5000),
        bloom_balance=fp(10),
        created_at=1741000000000,
        last_active=1741392000000,
        total_actions=500,
        ihsan_history=[fp(0.96)] * 50,
    )


@pytest.fixture
def network_wallets(deterministic_seed) -> list[WalletState]:
    """50-node network for Asabiyyah and Gini tests."""
    wallets = []
    for i in range(50):
        balance = fp(random.uniform(10, 1000))
        wallets.append(
            WalletState(
                node_id=bytes([i]) * 32,
                seed_balance=balance,
                created_at=1741392000000,
                last_active=1741392000000,
            )
        )
    return wallets
