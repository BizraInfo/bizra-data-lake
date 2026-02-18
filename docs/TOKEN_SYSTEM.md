# BIZRA Token System

Last updated: 2026-02-18

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
                        │ Community Fund│          │ ADL Gini gate    │
                        │ (BIZRA-       │          │ Harberger tax    │
                        │  COMMUNITY-   │          └────────┬─────────┘
                        │  FUND)        │                   │ 7%/epoch
                        └───────────────┘                   ▼
                                                   ┌──────────────┐
                                                   │  UBC Pool    │
                                                   │(__UBC_POOL__)│
                                                   └──────────────┘
```

### Module Layout

| File | Purpose | Lines |
|------|---------|-------|
| `core/token/__init__.py` | Re-exports all public symbols | 54 |
| `core/token/types.py` | Enums, dataclasses, constants | 269 |
| `core/token/ledger.py` | Hash-chained ledger + SQLite balances + ADL Gini gate + Harberger tax | 837 |
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

ADL-related constants from `core/integration/constants.py` (authoritative):

| Constant | Value | Description |
|----------|-------|-------------|
| `ADL_GINI_THRESHOLD` | 0.35 | Maximum allowed Gini coefficient for SEED distribution |
| `ADL_HARBERGER_TAX_RATE` | 0.07 (7%) | Per-epoch Harberger tax rate on SEED holdings |

System pool accounts from `core/token/ledger.py`:

| Account | Constant | Purpose |
|---------|----------|---------|
| `__UBC_POOL__` | `UBC_POOL_ID` | Universal Basic Compute redistribution pool |
| `BIZRA-COMMUNITY-FUND` | `COMMUNITY_FUND_ACCOUNT` | Computational zakat recipient |
| `SYSTEM-TREASURY` | `SYSTEM_TREASURY_ACCOUNT` | System treasury |

These are collected in `SYSTEM_POOL_IDS` (frozenset) and excluded from individual Gini computation.

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

## ADL Gini Gate

The ledger enforces the Adl (Justice) invariant at the transaction level. Every SEED `TRANSFER` or `MINT` operation is pre-validated against the Gini coefficient of the resulting balance distribution. If a transaction would push inequality above `ADL_GINI_THRESHOLD` (0.35), it is rejected before any state change.

### How It Works

1. **Scope**: Only SEED token. BLOOM and IMPT are exempt.
2. **Exempt operations**: `GENESIS_MINT`, `ZAKAT`, `BURN`, `STAKE`, `UNSTAKE` bypass the gate.
3. **System pool exclusion**: Communal accounts (`SYSTEM_POOL_IDS`) are excluded from Gini calculation — they are redistribution pools, not individual wealth.
4. **Directional awareness**: Transfers that *reduce* Gini are always allowed, even if the absolute level remains above threshold. This prevents system lockup when bootstrapping from unequal initial allocations.
5. **Transfers to system pools**: Always allowed (redistributive by nature).

### Validation Flow

```
_validate_transaction(tx)
  └─ if tx.token_type == SEED and tx.op in (TRANSFER, MINT):
       └─ _check_gini_impact(tx)
            ├─ target in SYSTEM_POOL_IDS? → ALLOW
            ├─ simulate post-tx balances (exclude system pools)
            ├─ compute pre_gini and post_gini
            ├─ post_gini <= pre_gini? → ALLOW (directionally improving)
            └─ post_gini > 0.35? → REJECT ("Plutocratic concentration rejected")
```

### Example: Blocked Concentration

```python
ledger = TokenLedger()
# After genesis: whale has 90,000 SEED, 5 nodes have 2,000 each
tx = TransactionEntry(
    op=TokenOp.MINT, token_type=TokenType.SEED,
    to_account="whale", amount=50000.0,
)
receipt = ledger.record_transaction(tx)
# receipt.success == False
# receipt.error == "ADL Gini gate: transaction would push Gini to 0.6821
#                   (threshold=0.35). Plutocratic concentration rejected."
```

### Example: Allowed Equalization

```python
# whale (90,000 SEED) transfers to small-node (2,000 SEED)
tx = TransactionEntry(
    op=TokenOp.TRANSFER, token_type=TokenType.SEED,
    from_account="whale", to_account="small-node", amount=10000.0,
)
receipt = ledger.record_transaction(tx)
# receipt.success == True (Gini decreased, directionally improving)
```

---

## Harberger Tax

Per-epoch taxation on all SEED holdings, flowing proceeds to the Universal Basic Compute (UBC) pool. The Harberger mechanism ensures that resources held but not productively used are gradually redistributed.

### Configuration

| Parameter | Default | Source |
|-----------|---------|--------|
| Tax rate | 7% per epoch | `ADL_HARBERGER_TAX_RATE` in `core/integration/constants.py` |
| Recipient | `__UBC_POOL__` | `UBC_POOL_ID` in `core/sovereign/adl_invariant.py` |
| UBC pool exempt | Yes | Pool is not taxed on its own holdings |

### Usage

```python
ledger = TokenLedger()

# Apply tax with default rate (7%)
result = ledger.apply_harberger_tax(epoch_id="epoch-12")
print(result)
# {
#     "total_taxed": 7000.0,
#     "accounts_affected": 5,
#     "ubc_pool_credit": 7000.0,
#     "tax_rate": 0.07,
#     "epoch_id": "epoch-12"
# }

# Override rate for special epochs
result = ledger.apply_harberger_tax(tax_rate=0.03, epoch_id="epoch-13")
```

### Tax Transfer Bypass

Tax transfers use `_record_tax_transfer()`, which bypasses the Gini gate. This is necessary because:

- Harberger taxes are inherently redistributive (they reduce Gini)
- The Gini gate would block transfers *to* the UBC pool if the pool's exclusion from the Gini calculation were not in place
- All other validations (balance sufficiency, hash chaining, signature) still apply

### Interaction with ADL Gini Gate

The Gini gate and Harberger tax form a complementary pair:

```
Gini Gate (preventive)              Harberger Tax (corrective)
├─ Blocks concentrating transfers   ├─ Taxies all SEED holdings per epoch
├─ Allows equalizing transfers      ├─ Proceeds flow to UBC pool
├─ Pre-transaction validation       ├─ Post-epoch redistribution
└─ Threshold: 0.35                  └─ Rate: 7% per epoch
```

Together they enforce the ADL invariant: Gini < 0.35 under normal operation, with continuous pressure toward equality.

---

## Testing

```bash
# Run all token system tests (68 tests, including ADL Gini + Harberger)
pytest tests/core/token/ -v

# Run ADL Gini gate tests only
pytest tests/core/token/test_token_ledger.py -v -k "TestAdlGiniGate"

# Run Harberger tax tests only
pytest tests/core/token/test_token_ledger.py -v -k "TestHarbergerTax"

# Run token + ADL + FATE regression suite
pytest tests/core/token/ tests/core/sovereign/test_adl_invariant.py tests/core/sovereign/test_fate_validation.py -v
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
| ADL Gini gate rejection | `"ADL Gini gate: transaction would push Gini to X.XXXX..."` | Redistribute SEED first, or use equalizing transfers |
| Harberger tax insufficient balance | Tax transfer fails for individual account, logged as warning | Account effectively exempt for that epoch |

---

## Standing on Giants

- **Nakamoto (2008)**: Hash-chained transaction ledger, genesis block concept
- **Lamport (1978)**: Logical clocks, monotonic sequence numbers
- **Merkle (1979)**: Hash chains for tamper detection
- **Shannon (1948)**: SNR as quality gate for PoI scoring
- **Al-Ghazali (1058-1111)**: Zakat (2.5%) as computational distributive justice
- **Gini (1912)**: Inequality measurement for ADL Gini gate
- **Harberger (1962)**: Self-assessed value taxation for redistribution
- **Rawls (1971)**: Justice as a hard gate, not a soft metric
- **Szabo (1997)**: Smart contracts as automated enforcement

---

*Source of truth: `core/token/types.py`, `core/token/ledger.py`, `core/token/mint.py`*
