# Phase 04: Memory Ledger + Token System

Last updated: 2026-02-14
Standing on: Merkle (1979, Hash Chains) · Nakamoto (2008, Genesis Block) · Lamport (1978, Logical Clocks) · Tulving (1972, Episodic Memory) · Al-Ghazali (Zakat)

---

## Purpose

Layer 4 is the **persistent memory** of the node. Every mutation — every inference, every token transfer, every judgment verdict — is recorded in a hash-chained, append-only ledger. The node can reconstruct its entire history from the ledger, detect any tampering, and prove the integrity of its memory to external verifiers.

Three independent ledgers serve different domains:
1. **Tamper-Evident Log** — Operational audit trail (HMAC-SHA256 chain)
2. **Experience Ledger** — Episodic memory for reasoning (BLAKE3 chain)
3. **Token Ledger** — Economic transactions (BLAKE3 chain, SQLite materialized view)

---

## Data Structures

### Tamper-Evident Log

```pseudocode
STRUCT TamperEvidentEntry:
    sequence:       u64              # Monotonic, never reused (Lamport clock)
    timestamp:      Timestamp        # Nanosecond precision
    event_type:     String           # e.g., "inference", "gate_pass", "config_change"
    payload:        bytes            # Canonical JSON of event data
    prev_hash:      Hash             # SHA-256 of previous entry
    entry_hash:     Hash             # HMAC-SHA256(key, sequence || timestamp || payload || prev_hash)
    hmac_key_id:    String           # Key identifier for rotation support

    INVARIANT: entry_hash == HMAC_SHA256(key, canonical(sequence, timestamp, payload, prev_hash))
    INVARIANT: prev_hash == previous_entry.entry_hash
    INVARIANT: sequence == previous_entry.sequence + 1

STRUCT TamperEvidentLog:
    entries:        List<TamperEvidentEntry>   # In-memory buffer
    head_hash:      Hash                       # Current chain tip
    head_sequence:  u64                        # Current sequence number
    hmac_key:       bytes                      # Active HMAC key
    storage_path:   Path                       # Append-only file

    METHOD append(event_type: String, payload: Any) -> TamperEvidentEntry:
        entry = TamperEvidentEntry(
            sequence   = self.head_sequence + 1,
            timestamp  = now_nanoseconds(),
            event_type = event_type,
            payload    = canonical_json(payload),
            prev_hash  = self.head_hash,
        )
        entry.entry_hash = hmac_sha256(self.hmac_key, canonical(entry))

        # Append to file (atomic write)
        append_to_file(self.storage_path, entry)

        # Update chain state
        self.head_hash = entry.entry_hash
        self.head_sequence = entry.sequence

        RETURN entry

    METHOD verify_chain() -> VerificationResult:
        # Walk entire log, verify linkage and HMAC integrity
        prev_hash = GENESIS_HASH   # 64 zero chars
        FOR EACH entry IN read_all_entries():
            # Verify hash chain linkage
            IF entry.prev_hash != prev_hash:
                RETURN VerificationResult(valid=false, error="Chain break at seq {}")

            # Verify HMAC integrity
            expected = hmac_sha256(key, canonical(entry))
            IF NOT timing_safe_compare(entry.entry_hash, expected):
                RETURN VerificationResult(valid=false, error="HMAC mismatch at seq {}")

            prev_hash = entry.entry_hash

        RETURN VerificationResult(valid=true, entries_checked=count)

ENUM TamperType:
    HASH_MISMATCH       # Entry hash doesn't match recomputation
    CHAIN_BREAK         # prev_hash doesn't link to predecessor
    SEQUENCE_GAP        # Missing sequence numbers
    TIMESTAMP_REVERSAL  # Later entry has earlier timestamp
    DUPLICATE_SEQUENCE  # Same sequence number appears twice
```

**Source:** `core/sovereign/tamper_evident_log.py` (986 lines)

### Experience Ledger

```pseudocode
STRUCT Episode:
    episode_hash:    Hash            # BLAKE3(canonical_json(self)), content-addressed
    context:         String          # What was the situation
    reasoning_graph: GraphSnapshot   # GoT artifact
    actions_taken:   List<Action>    # What was done
    impact:          EpisodeImpact   # Quality scores
    timestamp:       Timestamp
    prev_hash:       Hash            # Chain linkage

STRUCT EpisodeImpact:
    snr_score:        f64            # Signal quality of the episode
    ihsan_score:      f64            # Ethical quality
    efficiency:       f64            # Resource efficiency (deterministic integer arithmetic)
    verdict:          Verdict        # SNR_OK | SNR_LOW | SNR_FAIL

    # Standing on: deterministic integer arithmetic (no float drift)
    # efficiency = integer_log2(quality_numerator) - integer_log2(cost_denominator)
    # Uses bit operations, not floating point, for cross-platform determinism

STRUCT EpisodeLedger:
    episodes:        Map<Hash, Episode>     # Content-addressed store
    chain_head:      Hash                   # Latest episode hash
    chain_length:    u64

    METHOD commit(episode: Episode) -> Hash:
        # Only commits if verdict == SNR_OK
        IF episode.impact.verdict != SNR_OK:
            RETURN Error("Cannot commit episode with verdict: {}")

        episode.prev_hash = self.chain_head
        episode.episode_hash = blake3_hash(canonical_json(episode))

        self.episodes[episode.episode_hash] = episode
        self.chain_head = episode.episode_hash
        self.chain_length += 1

        RETURN episode.episode_hash

    METHOD retrieve(query: String, k: u32) -> List<Episode>:
        # RIR retrieval: Recency-Importance-Relevance scoring
        # Standing on: Tulving (1972) — episodic memory retrieval cues
        candidates = self.episodes.values()
        scored = []
        FOR EACH episode IN candidates:
            recency     = exp(-decay_rate * age_days(episode))
            importance  = episode.impact.snr_score * episode.impact.ihsan_score
            relevance   = semantic_similarity(query, episode.context)
            rir_score   = w_r * recency + w_i * importance + w_v * relevance
            scored.append((episode, rir_score))

        scored.sort_by(rir_score, descending=true)
        RETURN scored[:k]
```

**Source:** `core/sovereign/experience_ledger.py` (598 lines), `bizra-omega/bizra-core/src/sovereign/experience_ledger.rs` (1,491 lines)

### Token Ledger

```pseudocode
STRUCT TransactionEntry:
    tx_hash:        Hash             # BLAKE3("bizra-token-v1:" || canonical_json(self))
    sequence:       u64              # Monotonic sequence number
    timestamp:      Timestamp
    operation:      TokenOp          # mint | transfer | burn | stake | unstake | genesis_mint | zakat
    token_type:     TokenType        # SEED | BLOOM | IMPT
    from_account:   String           # Source account (empty for mint)
    to_account:     String           # Destination account
    amount:         f64              # Token amount
    epoch_id:       String           # Epoch context
    nonce:          String           # 16 hex chars (replay protection)
    prev_hash:      Hash             # Chain linkage
    signature:      Ed25519Signature # Signed AFTER sequence + prev_hash assigned

    INVARIANT: tx_hash == blake3("bizra-token-v1:" || canonical(self_without_hash))
    INVARIANT: signature covers tx_hash (signed AFTER finalization — CRITICAL-5 fix)

ENUM TokenType:
    SEED    # BZR_S — Utility token, transferable, earned via PoI
    BLOOM   # BZR_B — Governance token, transferable, earned via SEED staking
    IMPT    # IMPT  — Reputation (soulbound, non-transferable)

ENUM TokenOp:
    MINT           # Create from PoI (yearly cap: 1,000,000 SEED)
    GENESIS_MINT   # One-time founder allocation (100K + 50K + 3.75K zakat)
    TRANSFER       # Move between accounts
    BURN           # Remove from circulation
    STAKE          # Lock for governance/rewards
    UNSTAKE        # Release staked tokens
    ZAKAT          # 2.5% computational zakat (auto-triggered by MINT)

STRUCT TokenLedger:
    # Dual storage strategy:
    sqlite_db:      SQLiteConnection    # Materialized view (queryable balances)
    jsonl_path:     Path                # Immutable chain (source of truth)
    chain_head:     Hash                # Current chain tip
    chain_length:   u64
    lock:           Mutex               # Thread-safe mutations

    METHOD record_transaction(entry: TransactionEntry) -> Result<Hash, Error>:
        WITH self.lock:
            # Step 1: Assign sequence and prev_hash
            entry.sequence = self.chain_length + 1
            entry.prev_hash = self.chain_head

            # Step 2: Compute tx_hash
            entry.tx_hash = blake3("bizra-token-v1:" || canonical_json(entry))

            # Step 3: Validate
            self.validate(entry)?

            # Step 4: Write to SQLite (materialized balances)
            self.update_balances(entry)

            # Step 5: Append to JSONL (immutable chain)
            append_jsonl(self.jsonl_path, entry)

            # Step 6: Update chain state
            self.chain_head = entry.tx_hash
            self.chain_length += 1

            RETURN Ok(entry.tx_hash)

    METHOD validate(entry: TransactionEntry) -> Result<(), Error>:
        MATCH entry.operation:
            MINT:
                # Check yearly supply cap
                yearly = self.get_yearly_minted(entry.token_type, current_year())
                IF yearly + entry.amount > SEED_SUPPLY_CAP_PER_YEAR:
                    RETURN Error("Yearly supply cap exceeded")

            TRANSFER:
                # Check sufficient balance
                balance = self.get_balance(entry.from_account, entry.token_type)
                IF balance.available < entry.amount:
                    RETURN Error("Insufficient balance")
                # IMPT is soulbound
                IF entry.token_type == IMPT:
                    RETURN Error("IMPT tokens are non-transferable (soulbound)")
                # No self-transfer
                IF entry.from_account == entry.to_account:
                    RETURN Error("Cannot transfer to self")

            GENESIS_MINT:
                # Once-only guard
                IF self.has_genesis_mint():
                    RETURN Error("Genesis mint already executed")

            STAKE:
                balance = self.get_balance(entry.from_account, entry.token_type)
                IF balance.available < entry.amount:
                    RETURN Error("Insufficient available balance for staking")

            UNSTAKE:
                balance = self.get_balance(entry.from_account, entry.token_type)
                IF balance.staked < entry.amount:
                    RETURN Error("Insufficient staked balance")

        RETURN Ok(())
```

**Source:** `core/token/ledger.py` (594 lines), `core/token/types.py` (269 lines)

### Minting Engine

```pseudocode
STRUCT TokenMinter:
    keypair:          Ed25519Keypair    # Minting authority
    ledger:           TokenLedger
    genesis_done:     bool              # In-memory guard

    METHOD mint_seed(to: AccountId, amount: f64, epoch: String, poi: f64) -> TokenReceipt:
        # Standing on: Al-Ghazali — computational zakat (2.5%)

        # Step 1: Compute zakat
        zakat_amount = amount * ZAKAT_RATE   # 0.025
        net_amount = amount - zakat_amount

        # Step 2: Record main mint
        main_tx = TransactionEntry(
            operation  = MINT,
            token_type = SEED,
            to_account = to,
            amount     = net_amount,
            epoch_id   = epoch,
        )
        main_tx.signature = self.keypair.sign(main_tx.tx_hash)
        main_hash = self.ledger.record_transaction(main_tx)?

        # Step 3: Record zakat (auto-routed to community fund)
        zakat_tx = TransactionEntry(
            operation  = ZAKAT,
            token_type = SEED,
            to_account = "BIZRA-COMMUNITY-FUND",
            amount     = zakat_amount,
            epoch_id   = epoch,
        )
        zakat_tx.signature = self.keypair.sign(zakat_tx.tx_hash)
        self.ledger.record_transaction(zakat_tx)?

        RETURN TokenReceipt(success=true, tx_hash=main_hash, net=net_amount, zakat=zakat_amount)

    METHOD genesis_mint() -> List<TokenReceipt>:
        # Once-only allocation
        IF self.genesis_done OR self.ledger.has_genesis_mint():
            RETURN Error("Genesis already executed")

        receipts = []
        # Founder: 100,000 SEED (with 2.5% zakat)
        receipts.append(self.mint_seed("BIZRA-00000000", 100_000, "epoch-0-genesis", 1.0))
        # Treasury: 50,000 SEED (with 2.5% zakat)
        receipts.append(self.mint_seed("SYSTEM-TREASURY", 50_000, "epoch-0-genesis", 1.0))
        # Reputation: 1,000 IMPT to founder (soulbound)
        receipts.append(self.mint_impt("BIZRA-00000000", 1_000, "epoch-0-genesis"))

        self.genesis_done = true
        RETURN receipts

    METHOD distribute_from_poi(
        distributions: Map<NodeId, f64>,
        epoch: String,
        epoch_reward: f64,
        poi_scores: Map<NodeId, f64>,
    ) -> List<TokenReceipt>:
        # Bridge from compute_token_distribution() to actual minting
        receipts = []
        FOR EACH (node_id, amount) IN distributions:
            receipt = self.mint_seed(node_id, amount, epoch, poi_scores[node_id])
            receipts.append(receipt)
        RETURN receipts
```

**Source:** `core/token/mint.py` (577 lines)

---

## Constants

| Constant | Value | Defined In |
|----------|-------|-----------|
| `SEED_SUPPLY_CAP_PER_YEAR` | 1,000,000 | `core/token/types.py` |
| `ZAKAT_RATE` | 0.025 (2.5%) | `core/token/types.py` |
| `FOUNDER_GENESIS_ALLOCATION` | 100,000 | `core/token/types.py` |
| `SYSTEM_TREASURY_ALLOCATION` | 50,000 | `core/token/types.py` |
| `BLOOM_GOVERNANCE_QUORUM` | 0.5 | `core/token/types.py` |
| `GENESIS_TX_HASH` | `"0" * 64` | `core/token/ledger.py` |
| `TOKEN_DOMAIN_PREFIX` | `"bizra-token-v1:"` | `core/token/types.py` |
| `GENESIS_EPOCH_ID` | `"epoch-0-genesis"` | `core/token/types.py` |

---

## TDD Anchors

| Test | File | Validates |
|------|------|-----------|
| `test_tamper_evident_append` | `tests/core/sovereign/test_tamper_evident_log.py` | Entries chain correctly |
| `test_tamper_detection` | `tests/core/sovereign/test_tamper_evident_log.py` | Modified entries detected |
| `test_chain_verification` | `tests/core/sovereign/test_tamper_evident_log.py` | Full chain walks successfully |
| `test_experience_commit` | `tests/core/sovereign/test_experience_ledger.py` | Episodes commit with BLAKE3 chain |
| `test_rir_retrieval` | `tests/core/sovereign/test_experience_ledger.py` | RIR scoring retrieves relevant episodes |
| `test_token_mint_with_zakat` | `tests/core/token/` | Net = amount - 2.5%, zakat to fund |
| `test_genesis_once_only` | `tests/core/token/` | Second genesis_mint fails |
| `test_impt_soulbound` | `tests/core/token/` | IMPT transfer rejected |
| `test_yearly_cap` | `tests/core/token/` | Mint > 1M SEED/year rejected |
| `test_hash_chain_integrity` | `tests/core/token/` | verify_chain() passes on clean ledger |
| `test_poi_determinism` | `tests/core/proof_engine/test_poi_determinism.py` | Same inputs produce identical distribution |

---

## Storage Layout

```
sovereign_state/
├── genesis.json              # Birth certificate (Layer 1)
├── tamper_evident.log        # Operational audit trail
└── experience_ledger.db      # Episodic memory (SQLite)

.swarm/
└── memory.db                 # Token ledger (SQLite materialized view)

04_GOLD/
└── token_ledger.jsonl        # Immutable token transaction chain
```

---

*Source of truth: `core/sovereign/tamper_evident_log.py`, `core/sovereign/experience_ledger.py`, `core/token/ledger.py`, `core/token/mint.py`, `core/token/types.py`*
