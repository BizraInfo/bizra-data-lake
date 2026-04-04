# Phase 4: Shell Canonical Binding
## Goal: App shell consumes kernel truth only (L4 = read-only surface)
### References: 00_master_spec.md §4 Phase 4, Golden Gem #6

---

## 1. Problem Statement

The app shell (BIZRA-OS React dashboard) currently has:
- Its own state management independent of the kernel
- Mock/placeholder data for agent status and mission results
- No live connection to canonical receipts or evidence chain
- Potential to originate truth (violates L4 read-only contract)

**Golden Gem #6**: "The shell must reveal truth, never originate it."

## 2. Canonical Data Flow

```
Layer 0-2 (Kernel)              Layer 4 (Shell)
┌─────────────────┐             ┌──────────────────┐
│ Mission Engine   │──receipt──→│ Receipt Viewer    │
│ Evidence Chain   │──chain───→│ Trust Panel       │
│ Token Ledger     │──balance─→│ Wallet Display    │
│ Agent Registry   │──roster──→│ Agent Dashboard   │
│ Living Memory    │──state───→│ Memory Inspector  │
│ Gate Verdicts    │──score───→│ Quality Gauge     │
└─────────────────┘             └──────────────────┘
         │                              │
     WRITE authority              READ-ONLY surface
     (Kernel owns truth)          (Shell reveals truth)
```

## 3. Pseudocode: Kernel API Contract

```
# Kernel exposes frozen REST endpoints
# Shell consumes these and NOTHING ELSE for sovereign data

GET /api/v1/receipts/latest
  → { receipt_id, receipt_hash, decision, snr_score, ihsan_score, timestamp }

GET /api/v1/receipts/:id
  → { full ReceiptArtifact }

GET /api/v1/evidence/chain?tail=10
  → [{ sequence, entry_hash, prev_hash, receipt_summary }]

GET /api/v1/evidence/verify
  → { valid: bool, chain_length: int, gaps: [], integrity: "INTACT" | "BROKEN" }

GET /api/v1/tokens/balance
  → { seed_balance, impt_balance, zakat_total, tx_count }

GET /api/v1/tokens/ledger?limit=20
  → [{ tx_id, type, token, amount, recipient, hash }]

GET /api/v1/agents/roster
  → [{ agent_id, role, type: "PAT"|"SAT", status, last_receipt_hash }]

GET /api/v1/mission/current
  → { mission_id, state, progress_pct, assigned_agents, gate_verdict }

GET /api/v1/memory/stats
  → { total_entries, last_updated, db_size_bytes }

GET /api/v1/health
  → { status, uptime_s, evidence_seq, token_seq, agents_active }
```

## 4. Pseudocode: Shell Components (React)

```typescript
// Trust Panel — reveals evidence chain integrity
COMPONENT TrustPanel:
    STATE chain = useSWR("/api/v1/evidence/chain?tail=10")
    STATE verify = useSWR("/api/v1/evidence/verify")

    RENDER:
        IF verify.valid:
            <GreenShield text="Chain Intact" count={verify.chain_length} />
        ELSE:
            <RedAlert text="Chain Broken" gaps={verify.gaps} />

        FOR entry IN chain:
            <ChainLink
                sequence={entry.sequence}
                hash={entry.entry_hash[:16]}
                decision={entry.receipt_summary.decision}
            />

// Receipt Viewer — shows canonical receipts
COMPONENT ReceiptViewer:
    STATE latest = useSWR("/api/v1/receipts/latest")

    RENDER:
        <ReceiptCard
            id={latest.receipt_id}
            hash={latest.receipt_hash}
            decision={latest.decision}
            snr={latest.snr_score}
            ihsan={latest.ihsan_score}
            timestamp={latest.timestamp}
        />
        <QualityGauge
            snr={latest.snr_score}
            threshold={0.85}
            ihsan={latest.ihsan_score}
            ihsanThreshold={0.95}
        />

// Wallet Display — shows token economy
COMPONENT WalletDisplay:
    STATE balance = useSWR("/api/v1/tokens/balance")
    STATE ledger = useSWR("/api/v1/tokens/ledger?limit=10")

    RENDER:
        <TokenBalance seed={balance.seed_balance} impt={balance.impt_balance} />
        <ZakatMeter total={balance.zakat_total} />
        <TransactionList entries={ledger} />

// Agent Dashboard — shows PAT-7 + SAT-5 roster
COMPONENT AgentDashboard:
    STATE roster = useSWR("/api/v1/agents/roster")

    RENDER:
        <Section title="PAT-7 (Your Team)">
            FOR agent IN roster WHERE agent.type == "PAT":
                <AgentCard agent={agent} />
        </Section>
        <Section title="SAT-5 (System Validators)">
            FOR agent IN roster WHERE agent.type == "SAT":
                <AgentCard agent={agent} />
        </Section>
```

## 5. Anti-Patterns to Enforce

```
FORBIDDEN — Shell must NEVER:
  ✗ Generate its own receipt hashes
  ✗ Calculate SNR scores locally
  ✗ Maintain its own evidence chain
  ✗ Store mission state independently of kernel
  ✗ Create token transactions
  ✗ Modify agent assignments
  ✗ Override gate verdicts

REQUIRED — Shell must ALWAYS:
  ✓ Fetch all sovereign data from kernel REST API
  ✓ Display "kernel unreachable" when API fails
  ✓ Show last-known state with staleness indicator
  ✓ Render raw receipt hashes (not computed locally)
  ✓ Use SWR/polling for live updates (not WebSocket state)
```

## 6. Implementation Touchpoints

| File | Change |
|------|--------|
| `core/sovereign/runtime_core.py` | Expose REST endpoints for shell |
| `frontend/src/api/kernel.ts` | NEW: Typed kernel API client |
| `frontend/src/components/TrustPanel.tsx` | NEW: Evidence chain viewer |
| `frontend/src/components/ReceiptViewer.tsx` | NEW: Receipt display |
| `frontend/src/components/WalletDisplay.tsx` | NEW: Token balance |
| `frontend/src/components/AgentDashboard.tsx` | UPDATE: Consume roster API |
| `frontend/src/hooks/useKernel.ts` | NEW: SWR hooks for kernel data |

## 7. TDD Anchors

```typescript
// frontend/src/tests/TrustPanel.test.tsx

test("TrustPanel shows green when chain intact", async () => {
    mockApi("/api/v1/evidence/verify", { valid: true, chain_length: 34 });
    render(<TrustPanel />);
    expect(await screen.findByText("Chain Intact")).toBeInTheDocument();
});

test("TrustPanel shows red when chain broken", async () => {
    mockApi("/api/v1/evidence/verify", { valid: false, gaps: [12] });
    render(<TrustPanel />);
    expect(await screen.findByText("Chain Broken")).toBeInTheDocument();
});

test("ReceiptViewer displays kernel receipt not local", async () => {
    mockApi("/api/v1/receipts/latest", {
        receipt_hash: "d01419b68afc742d...",
        decision: "REVIEW",
    });
    render(<ReceiptViewer />);
    expect(await screen.findByText("d01419b6")).toBeInTheDocument();
    // Shell never computes its own hash
});

test("Shell shows staleness when kernel unreachable", async () => {
    mockApi("/api/v1/health", null, { status: 503 });
    render(<App />);
    expect(await screen.findByText("Kernel Unreachable")).toBeInTheDocument();
});

test("Shell cannot originate receipts", () => {
    // Grep the codebase for receipt creation in frontend
    const frontendCode = globReadAll("frontend/src/**/*.{ts,tsx}");
    expect(frontendCode).not.toMatch(/blake3|BLAKE3|createReceipt|generateHash/);
});
```

```python
# tests/integration/test_kernel_api.py

async def test_receipts_latest_endpoint():
    """Kernel serves latest receipt via REST."""
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8010/api/v1/receipts/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert "receipt_hash" in data
    assert "decision" in data

async def test_evidence_verify_endpoint():
    """Kernel validates evidence chain integrity."""
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8010/api/v1/evidence/verify")
    assert resp.status_code == 200
    data = resp.json()
    assert "valid" in data
    assert "chain_length" in data
```

## 8. Validation Gate

```
ALL of:
  [ ] Shell fetches receipts from kernel API (not local state)
  [ ] Trust panel displays evidence chain from kernel
  [ ] Wallet shows token balance from kernel ledger
  [ ] No receipt/hash computation in frontend/ codebase
  [ ] "Kernel Unreachable" shown when API down
  [ ] Staleness indicator on cached data
```

---

*Layer 4 reveals. Layers 0-2 govern.*
*The shell is glass, not authority.*
