# Phase 04 — View 3: RECEIPTS (Proof Chain)

> **Purpose:** Cryptographic evidence trail. Every action has a receipt.
> **Status:** NOT BUILT — event_schema_v1.json defines the format. No UI exists.

## 4.1 Content Spec

```
┌── RECEIPTS ──────────────────────────────────────────────────┐
│                                                                │
│  Chain Height: 147 | Integrity: VERIFIED | Last: 2m ago       │
│                                                                │
│  ┌─ #147 ────────────────────────────────────────────────┐    │
│  │  organize invoices     Ihsan: 0.9587   +2.38 SEED    │    │
│  │  d3e4f5a6...  <- a1b2c3d4...   2026-03-07 10:14 GST │    │
│  └───────────────────────────────────────────────────────┘    │
│  ┌─ #146 ────────────────────────────────────────────────┐    │
│  │  review auth PR        Ihsan: 0.9723   +1.85 SEED    │    │
│  │  a1b2c3d4...  <- 9f8e7d6c...   2026-03-07 09:42 GST │    │
│  └───────────────────────────────────────────────────────┘    │
│  ... (scrollable)                                             │
│                                                                │
│  Filter: [all] [qualified] [rejected] | Export: [json] [csv]  │
└──────────────────────────────────────────────────────────────┘
```

## 4.2 Data Model

```pseudocode
struct ReceiptView:
    chain_height: int
    chain_valid: bool
    last_receipt_time: str
    receipts: ReceiptEntry[]

struct ReceiptEntry:
    index: int
    mission_summary: str      # First 40 chars of intent
    ihsan_composite: float
    ihsan_components: {       # 8 dimensions
        accuracy: float,
        safety: float,
        fairness: float,
        transparency: float,
        privacy: float,
        accountability: float,
        sustainability: float,
        beneficence: float,
    }
    seed_earned: float
    pool_share: float
    zakat: float
    receipt_hash: str         # BLAKE2b-256
    prev_hash: str
    signature: str            # Ed25519
    verified: bool
    qualified: bool           # All gates passed
    reason_code: str          # POI_OK or rejection code
    timestamp: str

function fetch_receipts(page: int, filter: str) -> ReceiptView:
    TRY:
        episodes = api.get("/v1/seed/episodes", {page, limit: 20, filter})
        chain = api.get("/v1/token/verify")
        RETURN ReceiptView(
            chain_height=chain.chain_length,
            chain_valid=chain.valid,
            receipts=episodes.map(to_receipt_entry),
        )
    CATCH:
        // Read from local EventBus chain
        local_chain = read_local_event_chain()
        RETURN ReceiptView(
            chain_height=local_chain.length,
            chain_valid=verify_local_chain(),
            receipts=local_chain.filter(is_receipt_event),
        )
```

## 4.3 Existing Implementation

**event_schema_v1.json:** Defines `receipt.emitted`, `receipt.signed`, `receipt.audited` events with full payload spec.

**subscribers.py (SUB-3):** `TeleScriptStepReceiptAppend` — appends step-level receipts to chain.

**sovereign_terminal.py:345-358:** `evidence()` — minimal display (chain height + last hash). No list, no filtering.

## 4.4 What to Build

| Component | Surface | LOC Est | Priority |
|-----------|---------|---------|----------|
| Receipt list widget (scrollable) | Rust | 150 | P0 |
| Receipt detail panel (8-dim Ihsan) | Rust | 100 | P0 |
| Chain verification display | Both | 40 | P0 |
| Filter (all/qualified/rejected) | Both | 30 | P1 |
| Export (JSON/CSV) | Python | 50 | P2 |
| Hash chain visualization | Rust | 80 | P2 |

## 4.5 TDD Anchors

```
TEST: receipts_chain_integrity
  GIVEN 10 chained receipts
  WHEN verify_chain() called
  THEN returns True and every prev_hash matches previous event_hash

TEST: receipts_filter_qualified
  GIVEN 10 receipts, 7 qualified, 3 rejected
  WHEN filter="qualified"
  THEN returns exactly 7 entries

TEST: receipts_offline_fallback
  GIVEN backend unreachable
  WHEN receipts view requested
  THEN shows local chain with "offline" indicator

TEST: receipts_export_json_valid
  GIVEN receipt list
  WHEN export("json") called
  THEN output is valid JSON array of ReceiptEntry objects

TEST: receipts_ihsan_components_sum_to_composite
  GIVEN receipt with 8 ihsan dimensions
  THEN weighted sum of dimensions == ihsan_composite (within 0.001)
```
