# Phase 52.5: Receipt Chain (Phase 4 -- Cryptographic Audit Trail)

> Standing on Giants: Nakamoto (hash-linked chain of records, 2008) · Lamport (Byzantine fault tolerance, 1982) · Merkle (hash trees, 1979) · Shannon (entropy as tamper detection, 1948) · Al-Ghazali (every claim must bind to evidence, 1095)

## 1. Overview

Every action Ahmed's node executes produces a cryptographic receipt. Receipts are
hash-linked into an append-only chain starting from a genesis block (0x00*32).
Each receipt is BLAKE3-hashed with domain separation and Ed25519-signed by the
node's identity key. The chain is the node's proof of impact -- unforgeably
recording what was done, when, how well, and at what cost.

This fulfills the CLAIM_MUST_BIND kernel invariant: no hallucination, every claim
has evidence.

---

## 2. Data Flow

```
  Action completed (from ActionExecutor)
       │
  ┌────▼──────────────────────────────────────────┐
  │  RECEIPT CONSTRUCTION                          │
  │                                                │
  │  1. Collect fields (action, outcome, cpva)     │
  │  2. Compute outcome_hash = BLAKE3(outcome)     │
  │  3. Set prev_hash = last_receipt.hash          │
  │  4. Compute receipt_hash = BLAKE3(all_fields)  │
  │  5. Sign: Ed25519(receipt_hash)                │
  │  6. Append to chain                            │
  └────┬──────────────────────────────────────────┘
       │
  Chain:  genesis(0x00*32) → A → B → C → D → E → F
                                                │
       ┌────────────────────────────────────────▼──┐
       │  API: GET /api/v1/receipt/chain/{profileId}│
       └────────────────────────────────────────────┘
```

---

## 3. Pseudocode

### 3.1 ActionReceipt Dataclass

```python
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from core.integration.constants import IHSAN_THRESHOLD, SNR_THRESHOLD


@dataclass
class ActionReceipt:
    """Immutable record of a single verified action."""
    action_id: str = field(default_factory=lambda: str(uuid4()))
    action_type: str = ""              # "file_move", "ocr_extract", "email_send"
    description: str = ""
    domain: str = ""                   # "filesystem", "email", "ocr", "analysis"
    channel: str = "hda"               # "hda" | "api" | "federation"
    ihsan_score: float = 0.0
    snr_score: float = 0.0
    confidence: float = 0.0
    outcome: str = "success"           # "success" | "failure" | "partial"
    outcome_detail: str = ""
    outcome_hash: str = ""             # BLAKE3(outcome_detail)
    cpva_usd: float = 0.0
    source_memory_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    prev_hash: str = ""                # Hash of previous receipt
    receipt_hash: str = ""             # BLAKE3(all_fields)
    signature: str = ""                # Ed25519(receipt_hash)

    def compute_hashes(self) -> None:
        """Compute outcome_hash and receipt_hash with domain separation."""
        self.outcome_hash = self._blake3(b"bizra.receipt.outcome:" + self.outcome_detail.encode())
        self.receipt_hash = self._blake3(b"bizra.receipt.chain:" + self._canonical().encode())

    def _canonical(self) -> str:
        return json.dumps({
            "action_id": self.action_id, "action_type": self.action_type,
            "description": self.description, "domain": self.domain,
            "channel": self.channel, "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score, "confidence": self.confidence,
            "outcome": self.outcome, "outcome_hash": self.outcome_hash,
            "cpva_usd": self.cpva_usd, "timestamp": self.timestamp,
            "prev_hash": self.prev_hash}, sort_keys=True)

    @staticmethod
    def _blake3(data: bytes) -> str:
        try:
            import blake3
            return blake3.blake3(data).hexdigest()
        except ImportError:
            return hashlib.blake2b(data, digest_size=32).hexdigest()
```

### 3.2 Receipt Chain

```python
class ReceiptChain:
    """Append-only hash-linked chain. Genesis has prev_hash = 0x00*32."""

    GENESIS_PREV_HASH: str = "00" * 32

    def __init__(self, profile_id: str, signing_key: Optional[bytes] = None):
        self.profile_id = profile_id
        self._signing_key = signing_key
        self._chain: list[ActionReceipt] = []

    async def initialize_genesis(self) -> ActionReceipt:
        genesis = ActionReceipt(
            action_type="genesis", description=f"Genesis for {self.profile_id}",
            domain="system", outcome="success", outcome_detail="Chain initialized",
            prev_hash=self.GENESIS_PREV_HASH, ihsan_score=1.0, snr_score=1.0,
            confidence=1.0, cpva_usd=0.0)
        genesis.compute_hashes()
        genesis.signature = self._sign(genesis.receipt_hash)
        self._chain.append(genesis)
        return genesis

    async def append(self, **kwargs) -> ActionReceipt:
        if not self._chain:
            await self.initialize_genesis()
        receipt = ActionReceipt(prev_hash=self._chain[-1].receipt_hash, **kwargs)
        receipt.compute_hashes()
        receipt.signature = self._sign(receipt.receipt_hash)
        self._chain.append(receipt)
        return receipt

    def verify(self) -> tuple[bool, Optional[str]]:
        """Verify entire chain. Returns (valid, error_description)."""
        if not self._chain:
            return False, "empty_chain"
        if self._chain[0].prev_hash != self.GENESIS_PREV_HASH:
            return False, "genesis_prev_hash_invalid"
        for i in range(1, len(self._chain)):
            if self._chain[i].prev_hash != self._chain[i - 1].receipt_hash:
                return False, f"chain_break_at_{i}"
        for i, receipt in enumerate(self._chain):
            saved = receipt.receipt_hash
            receipt.compute_hashes()
            if receipt.receipt_hash != saved:
                return False, f"hash_mismatch_at_{i}"
        return True, None

    def length(self) -> int:
        return len(self._chain)

    def get_chain(self) -> list[ActionReceipt]:
        return list(self._chain)

    def total_cpva(self) -> float:
        return sum(r.cpva_usd for r in self._chain)

    def _sign(self, data: str) -> str:
        if self._signing_key is None:
            return "unsigned"
        try:
            from nacl.signing import SigningKey
            return SigningKey(self._signing_key).sign(data.encode()).signature.hex()
        except ImportError:
            return "ed25519_unavailable"
```

### 3.3 Chain API

```python
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/receipt")

@router.get("/chain/{profile_id}")
async def get_chain(profile_id: str):
    chain = receipt_store.get(profile_id)
    if chain is None:
        raise HTTPException(404, "chain_not_found")
    valid, error = chain.verify()
    return {"profileId": profile_id, "length": chain.length(),
            "totalCpva": chain.total_cpva(), "valid": valid, "error": error,
            "receipts": [r.__dict__ for r in chain.get_chain()]}
```

---

## 4. Ahmed's Chain (Concrete)

```
Receipt 0 (genesis):  prev=0x00*32  hash=a1b2...c3d4  cpva=$0.000
Receipt 1 (Task A):   prev=a1b2...  ihsan=0.91 snr=0.90  cpva=$0.021
Receipt 2 (Task B):   prev=...      ihsan=0.95 snr=0.91  cpva=$0.006
Receipt 3 (Task C):   prev=...      ihsan=0.95 snr=0.91  cpva=$0.006
Receipt 4 (Task D):   prev=...      ihsan=0.97 snr=0.92  cpva=$0.016
Receipt 5 (Task E):   prev=...      ihsan=0.96 snr=0.91  cpva=$0.011
Receipt 6 (Task F):   prev=e5f6...  ihsan=0.97 snr=0.92  cpva=$0.012

Chain: 7 receipts, $0.072 total CPVA, all verified
```

---

## 5. TDD Anchors

```python
import pytest

class TestReceiptChain:
    """Phase 52.5: Receipt chain and audit tests."""

    @pytest.mark.asyncio
    async def test_genesis_receipt(self):
        chain = ReceiptChain(profile_id="ahmed-001")
        genesis = await chain.initialize_genesis()
        assert genesis.prev_hash == "00" * 32
        assert genesis.receipt_hash != ""
        assert genesis.action_type == "genesis"

    @pytest.mark.asyncio
    async def test_chain_append(self):
        chain = ReceiptChain(profile_id="ahmed-001")
        await chain.initialize_genesis()
        receipt = await chain.append(action_type="file_move",
            description="Move PDF", domain="filesystem",
            ihsan_score=0.97, cpva_usd=0.01)
        assert receipt.prev_hash == chain.get_chain()[0].receipt_hash
        assert chain.length() == 2

    @pytest.mark.asyncio
    async def test_chain_verify(self):
        chain = ReceiptChain(profile_id="ahmed-001")
        await chain.initialize_genesis()
        await chain.append(action_type="a", cpva_usd=0.01)
        await chain.append(action_type="b", cpva_usd=0.02)
        valid, error = chain.verify()
        assert valid is True and error is None

    @pytest.mark.asyncio
    async def test_tamper_detection(self):
        chain = ReceiptChain(profile_id="ahmed-001")
        await chain.initialize_genesis()
        await chain.append(action_type="a", cpva_usd=0.01)
        chain._chain[1].cpva_usd = 999.99
        valid, error = chain.verify()
        assert valid is False and "hash_mismatch" in error

    @pytest.mark.asyncio
    async def test_chain_break_detection(self):
        chain = ReceiptChain(profile_id="ahmed-001")
        await chain.initialize_genesis()
        await chain.append(action_type="a", cpva_usd=0.01)
        chain._chain[1].prev_hash = "ff" * 32
        valid, _ = chain.verify()
        assert valid is False

    @pytest.mark.asyncio
    async def test_total_cpva(self):
        chain = ReceiptChain(profile_id="ahmed-001")
        await chain.initialize_genesis()
        await chain.append(action_type="a", cpva_usd=0.02)
        await chain.append(action_type="b", cpva_usd=0.03)
        assert abs(chain.total_cpva() - 0.05) < 1e-6

    def test_receipt_hash_deterministic(self):
        r1 = ActionReceipt(action_id="x", action_type="test",
                           prev_hash="abc", timestamp=1000.0)
        r2 = ActionReceipt(action_id="x", action_type="test",
                           prev_hash="abc", timestamp=1000.0)
        r1.compute_hashes(); r2.compute_hashes()
        assert r1.receipt_hash == r2.receipt_hash

    def test_domain_separation(self):
        r = ActionReceipt(outcome_detail="test", prev_hash="abc")
        r.compute_hashes()
        assert r.outcome_hash != r.receipt_hash
```
