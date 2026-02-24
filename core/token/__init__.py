"""
BIZRA Token Module — SEED, BLOOM, and IMPT Token Operations
============================================================

The missing 40% that turns architecture into reality.

Token Types:
- SEED (BZR_S): Utility token, earned via Proof of Impact
- BLOOM (BZR_B): Governance token, earned via SEED staking
- IMPT: Non-transferable reputation score (off-chain)

Standing on Giants:
- Nakamoto (2008): UTXO model and genesis block concept
- Lamport (1978): Logical clocks for event ordering
- Merkle (1979): Hash chains for tamper detection
- Shannon (1948): SNR as quality gate on all operations
- Al-Ghazali (1058-1111): Ihsan as ethical constraint floor
- Szabo (1997): Smart contract theory
"""

from core.token.ledger import TokenLedger
from core.token.mint import TokenMinter
from core.token.rl_rewards import (
    composite_reward,
    compute_agent_reward,
    enforce_agent_gini,
    token_efficiency_reward,
    update_agent_reputation,
)
from core.token.strategy import (
    AgentStrategy,
    load_strategy,
    persist_strategy,
    update_strategy,
)
from core.token.types import (
    BLOOM_SYMBOL,
    GENESIS_EPOCH_ID,
    IMPT_SYMBOL,
    SEED_SUPPLY_CAP_PER_YEAR,
    SEED_SYMBOL,
    TokenBalance,
    TokenOp,
    TokenReceipt,
    TokenType,
    TransactionEntry,
)

__all__ = [
    # Types
    "TokenType",
    "TokenOp",
    "TokenBalance",
    "TransactionEntry",
    "TokenReceipt",
    # Constants
    "SEED_SYMBOL",
    "BLOOM_SYMBOL",
    "IMPT_SYMBOL",
    "SEED_SUPPLY_CAP_PER_YEAR",
    "GENESIS_EPOCH_ID",
    # Core classes
    "TokenLedger",
    "TokenMinter",
    # RL Rewards
    "composite_reward",
    "token_efficiency_reward",
    "compute_agent_reward",
    "update_agent_reputation",
    "enforce_agent_gini",
    # Strategy
    "AgentStrategy",
    "update_strategy",
    "persist_strategy",
    "load_strategy",
]
