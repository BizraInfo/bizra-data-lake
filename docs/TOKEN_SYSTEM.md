# BIZRA Token System

Last updated: 2026-02-14

The token module (`core/token/`) implements a three-token economy with hash-chained ledger, Ed25519-signed transactions, and Proof of Impact (PoI) driven distribution.

---

## Token Types

| Token | Symbol | Purpose | Transferable | Earned Via |
|-------|--------|---------|--------------|------------|
| SEED | `BZR_S` | Utility | Yes | Proof of Impact |
| BLOOM | `BZR_B` | Governance | Yes | SEED staking rewards |
| IMPT | `IMPT` | Reputation | No (soulbound) | Lifetime impact accumulation |

**SEED** is the primary utility token. Nodes earn SEED by contributing compute, data, or verified reasoning to the network. It can be spent on compute time, storage, and tool access.

**BLOOM** is the governance token. Earned exclusively through SEED staking, BLOOM grants voting rights on protocol changes. Governance quorum requires 50% of staked BLOOM (`BLOOM_GOVERNANCE_QUORUM = 0.5`).

**IMPT** is a non-transferable reputation score. It compounds reward multipliers over time, incentivizing sustained contribution rather than short-term extraction. The ledger enforces soulbound semantics: any transfer of IMPT is rejected with `"IMPT tokens are non-transferable (soulbound)"`.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   PoI Engine         │
                        │ (core/proof_engine/) │
                        └─────────┬───────────┘
                                  │ poi_scores
                                  ▼
┌───────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ Genesis Mint  │────▶│    TokenMinter       │────▶│   TokenLedger    │
│ (one-time)    │     │ (core/token/mint.py) │     │ (core/token/     │
└───────────────┘     └─────────┬───────────┘     │  ledger.py)      │
                                │                  ├──────────────────┤
                                │ 2.5% zakat       │ SQLite (balances)│
                                ▼                  │ JSONL (hash      │
                        ┌───────────────┐          │  chain log)      │
                        │ Community Fund│          └──────────────────┘
                        │ (BIZRA-       │
                        │  COMMUNITY-   │
                        │  FUND)        │
                        └───────────────┘
```

### Module Layout

| File | Purpose | Lines |
|------|---------|-------|
| `core/token/__init__.py` | Re-exports all public symbols | 54 |
| `core/token/types.py` | Enums, dataclasses, constants | 269 |
| `core/token/ledger.py` | Hash-chained ledger + SQLite balances | 594 |
| `core/token/mint.py` | Minting engine + zakat + genesis | 577 |

---

## Constants

All constants are defined in `core/token/types.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `SEED_SUPPLY_CAP_PER_YEAR` | 1,000,000 | Maximum SEED minted per calendar year |
| `ZAKAT_RATE` | 0.025 (2.5%) | Computational zakat on all mints |
| `FOUNDER_GENESIS_ALLOCATION` | 100,000 | Node0 genesis SEED allocation |
| `SYSTEM_TREASURY_ALLOCATION` | 50,000 | System treasury initial SEED |
| `BLOOM_GOVERNANCE_QUORUM` | 0.5 | 50% staked BLOOM required for votes |
| `GENESIS_EPOCH_ID` | `"epoch-0-genesis"` | First epoch identifier |
| `TOKEN_DOMAIN_PREFIX` | `"bizra-token-v1:"` | BLAKE3 hash domain separator |
| `IHSAN_THRESHOLD` | Imported from `core/integration/constants.py` | Quality gate for operations |

---

## Operations

Seven transaction types are supported (`TokenOp` enum):

| Operation | Description | Validation |
|-----------|-------------|------------|
| `mint` | Create new tokens from PoI | Yearly supply cap check |
| `genesis_mint` | One-time founder allocation | Once-only guard |
| `transfer` | Move tokens between accounts | Sufficient balance, not self-transfer |
| `burn` | Remove tokens from circulation | Sufficient balance |
| `stake` | Lock tokens for governance/rewards | Sufficient available balance |
| `unstake` | Release staked tokens | Sufficient staked balance |
| `zakat` | Computational zakat (2.5%) | Auto-triggered by mint |

---

## Ledger Design

The `TokenLedger` class implements dual storage:

1. **SQLite database** (`token_balances`, `token_transactions`, `token_supply` tables) — queryable materialized view of balances and history
2. **JSONL append log** (`04_GOLD/token_ledger.jsonl`) — immutable hash-chained source of truth

### Hash Chain

Every transaction links to its predecessor via `prev_hash`:

```
TX #0 (genesis)     TX #1               TX #2
prev_hash: 000...   prev_hash: H(TX#0)  prev_hash: H(TX#1)
tx_hash: H(TX#0)    tx_hash: H(TX#1)    tx_hash: H(TX#2)
```

The genesis sentinel hash is 64 zero characters (`GENESIS_TX_HASH = "0" * 64`).

### Transaction Hash Computation

Each transaction is hashed using BLAKE3 with domain separation:

```python
# From TransactionEntry.compute_hash()
prefixed = "bizra-token-v1:".encode() + canonical_bytes()
tx_hash = hex_digest(prefixed)  # BLAKE3 via core.proof_engine.canonical
```

The canonical form uses RFC 8785-style deterministic JSON (sorted keys, minimal separators).

### Chain Verification

```python
ledger = TokenLedger()
is_valid, entries_checked, error = ledger.verify_chain()
# Walks entire JSONL log, verifying prev_hash linkage and hash correctness
```

---

## Minting Engine

The `TokenMinter` holds an Ed25519 keypair and enforces all minting rules.

### SEED Minting with Zakat

```python
minter = TokenMinter.create()
receipt = minter.mint_seed(
    to_account="node-42",
    amount=1000.0,
    epoch_id="epoch-7",
    poi_score=0.87,
)
# Net to node-42: 975.0 SEED
# Zakat to BIZRA-COMMUNITY-FUND: 25.0 SEED
```

Every SEED mint automatically routes 2.5% to the community fund. This is the computational zakat — distributive justice enforced at the protocol level.

### Genesis Mint

```python
receipts = minter.genesis_mint()
# Allocates:
#   BIZRA-00000000 (Node0):     100,000 SEED
#   SYSTEM-TREASURY:             50,000 SEED
#   BIZRA-COMMUNITY-FUND:         3,750 SEED (2.5% zakat)
#   BIZRA-00000000 (Node0):      1,000 IMPT (reputation)
```

The genesis mint executes exactly once. Re-execution is blocked both by an in-memory flag and by checking the ledger for existing `genesis_mint` transactions.

### PoI-Driven Distribution

```python
# Bridge from PoI engine output to actual token minting
receipts = minter.distribute_from_poi(
    distributions={"node-1": 500, "node-2": 300, "node-3": 200},
    epoch_id="epoch-12",
    epoch_reward=1000.0,
    poi_scores={"node-1": 0.95, "node-2": 0.88, "node-3": 0.72},
)
```

---

## Cryptographic Integrity

| Layer | Algorithm | Source |
|-------|-----------|--------|
| Transaction hashing | BLAKE3 with domain prefix | `core/proof_engine/canonical.hex_digest()` |
| Transaction signing | Ed25519 | `core/pci/crypto.sign_message()` |
| Signature verification | Ed25519 | `core/pci/crypto.verify_signature()` |
| Replay protection | Per-transaction nonce (16 hex chars) | `secrets.token_hex(8)` |
| Chain ordering | Monotonic sequence numbers | Lamport logical clock |

Signing happens **after** the ledger assigns `sequence` and `prev_hash`, ensuring the signature covers the final hash (CRITICAL-5 fix in `mint.py:525`).

---

## Thread Safety

`TokenLedger.record_transaction()` acquires a `threading.Lock` before any mutation. The lock covers:
1. Sequence increment
2. Hash computation
3. Validation
4. SQLite write
5. JSONL append
6. Chain state update

---

## Balance Queries

```python
ledger = TokenLedger()

# Single balance
bal = ledger.get_balance("node-42", TokenType.SEED)
print(f"Total: {bal.balance}, Staked: {bal.staked}, Available: {bal.available}")

# All balances for an account
all_bals = ledger.get_all_balances("node-42")

# Total supply
supply = ledger.get_total_supply(TokenType.SEED)

# Transaction history
history = ledger.get_transaction_history(
    account_id="node-42",
    token_type=TokenType.SEED,
    limit=50,
)

# All accounts with non-zero balances
accounts = ledger.list_accounts()

# Yearly minted
yearly = ledger.get_yearly_minted(TokenType.SEED, 2026)
```

---

## Testing

```bash
# Run token system tests
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/token/ -v

# Run integration tests that exercise token + PoI pipeline
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/integration/test_token_integration.py -v
```

---

## Failure Modes

| Failure | Behavior | Recovery |
|---------|----------|----------|
| Yearly cap exceeded | `TokenReceipt(success=False, error="Yearly supply cap exceeded...")` | Wait for next year or reduce amount |
| Insufficient balance | Rejected before write | No state change |
| IMPT transfer attempt | Rejected with soulbound error | Use correct token type |
| Duplicate genesis mint | Blocked by ledger check + in-memory flag | No action needed |
| SQLite write failure | Sequence rolled back, error logged | Retry or investigate disk |
| Chain break detected | `verify_chain()` returns `(False, count, error_msg)` | Investigate JSONL tampering |

---

## Standing on Giants

- **Nakamoto (2008)**: Hash-chained transaction ledger, genesis block concept
- **Lamport (1978)**: Logical clocks, monotonic sequence numbers
- **Merkle (1979)**: Hash chains for tamper detection
- **Shannon (1948)**: SNR as quality gate for PoI scoring
- **Al-Ghazali (1058-1111)**: Zakat (2.5%) as computational distributive justice
- **Gini (1912)**: Inequality measurement for ADL constraint
- **Szabo (1997)**: Smart contracts as automated enforcement

---

*Source of truth: `core/token/types.py`, `core/token/ledger.py`, `core/token/mint.py`*
