---
name: cross-lang-sync
description: Audit Python/Rust constitutional constant synchronization across BIZRA modules. Detects drift in IHSAN, SNR, ADL Gini, Harberger, and related thresholds between core/integration/constants.py and canonical Rust sources.
---

# Cross-Language Constant Synchronization Audit

You are the Cross-Language Sync Auditor for the BIZRA ecosystem. Your mission is to detect and report drift between the Python and Rust implementations of constitutional thresholds.

## Canonical Sources

| Language | File | Role |
|----------|------|------|
| Python | `core/integration/constants.py` | Authoritative source of truth |
| Rust | `bizra-omega/bizra-core/src/lib.rs` | Primary Rust constants |
| Rust | `bizra-omega/bizra-core/src/omega.rs` | ADL/Gini constants |
| Rust | `bizra-omega/bizra-core/src/constitution.rs` | Constitution wiring |
| Rust | `bizra-omega/bizra-resourcepool/src/lib.rs` | Harberger tax constant |

## Constants to Audit

### Tier 1 — Constitutional (MUST match exactly)
- `IHSAN_THRESHOLD` — Python: 0.95, Rust: must be 0.95
- `SNR_THRESHOLD` — Python: 0.85, Rust: must be 0.85
- `ADL_GINI_THRESHOLD` — Python: 0.35, Rust: must be 0.35
- `ADL_HARBERGER_TAX_RATE` — Python: 0.05, Rust: must be 0.05
- `MIN_CONFIDENCE` — Python: 0.80, Rust: must be 0.80 *(promoted from test-local 2026-04-15)*
- `MAX_HARM_SCORE` — Python: 0.30, Rust: must be 0.30 *(promoted from test-local 2026-04-15)*

### Tier 2 — Operational (should match)
- `STRICT_IHSAN_THRESHOLD` — Python: 0.99
- `SNR_THRESHOLD_T0_ELITE` — Python: 0.98
- `SNR_THRESHOLD_T1_HIGH` — Python: 0.95
- `CONFIDENCE_HIGH` / `CONFIDENCE_MEDIUM` / `CONFIDENCE_LOW`
- `GENESIS_CUTOFF_HOURS` — Python: 72

### Tier 3 — Structural (verify alignment)
- Ihsan dimension weights (8 dimensions summing to 1.0)
- Four Pillars thresholds
- PAT minting thresholds

## Audit Protocol

1. **Run the audit script** (if available):
   ```bash
   python3 .claude/skills/cross-lang-sync/audit_constants.py
   ```

2. **Manual cross-reference** — Read both canonical files and compare:
   - `grep -n "IHSAN_THRESHOLD\|SNR_THRESHOLD\|ADL_GINI\|HARBERGER" core/integration/constants.py`
   - `grep -rn "IHSAN_THRESHOLD\|SNR_THRESHOLD\|ADL_GINI\|HARBERGER" bizra-omega/bizra-core/src/ bizra-omega/bizra-resourcepool/src/`

3. **Check for rogue definitions** — constants defined outside canonical files:
   - `grep -rn "IHSAN_THRESHOLD\s*=" core/ --include="*.py" | grep -v constants.py`
   - `grep -rn "IHSAN_THRESHOLD\|SNR_THRESHOLD" bizra-omega/ --include="*.rs" | grep -v "use crate\|pub const\|lib.rs\|omega.rs"`

4. **Proofspace sweep** — the ProofSpace runtime is the primary enforcement consumer; audit must verify no hardcoded copies:
   - `grep -nE "MAX_HARM_SCORE|MIN_CONFIDENCE|ADL_GINI_MAX|IHSAN_THRESHOLD|SNR_FLOOR|SNR_MINIMUM|SNR_THRESHOLD" bizra-omega/bizra-proofspace/src/*.rs bizra-omega/bizra-proofspace/benches/*.rs`
   - Every `pub const` or `const` in these files must either re-export from `bizra_core::*` / `bizra_core::omega::*` or fail the audit.

5. **Full workspace sweep** — catch constants in crates not explicitly listed above:
   - `grep -rnE "^[[:space:]]*(pub )?const (IHSAN_THRESHOLD|ADL_GINI_MAX|MAX_HARM_SCORE|MIN_CONFIDENCE|SNR_THRESHOLD|SNR_FLOOR|SNR_MINIMUM|HARBERGER_TAX_RATE)[[:space:]]*:[[:space:]]*(f64|u32|Decimal)[[:space:]]*=[[:space:]]*[0-9]" bizra-omega/ --include="*.rs" | grep -v "bizra-core/src/" | grep -v "bizra-resourcepool/src/" | grep -v "bizra-node0/"`
   - Any match is a rogue hardcode and fails the audit.

## Output Format

```
## Cross-Language Sync Audit Report

### Status: [ALIGNED | DRIFT DETECTED]

### Tier 1 — Constitutional Constants
| Constant | Python | Rust | Status |
|----------|--------|------|--------|
| IHSAN_THRESHOLD | 0.95 | 0.95 | ALIGNED |
| SNR_THRESHOLD | 0.85 | 0.85 | ALIGNED |
| ADL_GINI_THRESHOLD | 0.35 | 0.35 | ALIGNED |

### Drift Details
- *(example)* If a mismatch is detected, list it here with
  `Constant: Python=X (file:line), Rust=Y (file:line)` and recommended fix.

### Rogue Definitions
[List any constants defined outside canonical files]

### Recommendations
[Specific file:line fixes needed]
```

## Known Drift (as of 2026-04-15)

**None.** All Tier 1 constitutional constants are ALIGNED across Python and Rust. All downstream Rust consumers re-export from `bizra_core` or use the canonical `bizra_resourcepool` Harberger definition — verified by full workspace sweep (step 5).

Audit must include `bizra-proofspace` (runtime + benches) in addition to `bizra-core` and `bizra-tests` — this is the primary ProofSpace enforcement path and the highest-severity drift surface in the repo.

History:
- *Phase 56:* `ADL_GINI_THRESHOLD` drifted Python=0.35 vs Rust=0.40 — **RESOLVED** (Rust now 0.35).
- *2026-04-15:* `MIN_CONFIDENCE` and `MAX_HARM_SCORE` promoted from test-local (`proof_pyramid_e2e.rs`) to canonical in both `constants.py` and `bizra-core/src/lib.rs`. Test file now re-exports via `bizra_core::{MIN_CONFIDENCE, MAX_HARM_SCORE}`.
- *2026-04-15:* `IHSAN_THRESHOLD` in `proof_pyramid_e2e.rs` was hardcoded `0.95`; patched to re-export `bizra_core::IHSAN_THRESHOLD` — drift-risk surface closed.
- *2026-04-15 (correction):* Initial canon-promotion missed `bizra-proofspace` runtime copies at `src/lib.rs:40-46`, `src/fate_proof.rs:49-55`, `src/receipt_chain.rs:399-401`, and `benches/proof_pyramid_bench.rs:154-157`. All four sites now re-export from `bizra_core::*`. `cargo test -p bizra-proofspace --lib` → 42/42 pass. Audit protocol updated to sweep proofspace explicitly.
- *2026-04-15 (final):* `SNR_FLOOR` in `fate_proof.rs:61` was last hardcoded Tier-1 constant — patched to `bizra_core::SNR_THRESHOLD`. Audit protocol widened to cover full Tier-1 set (SNR_FLOOR, SNR_MINIMUM, SNR_THRESHOLD) and codified workspace-wide sweep as step 5.
- *2026-04-16 (receipt correction):* `ADL_HARBERGER_TAX_RATE` was always present in Rust at `bizra-resourcepool/src/lib.rs:72` as `HARBERGER_TAX_RATE`; the real gap was the skill's canonical-source list, not the codebase. Canonical Rust sources now include `bizra-resourcepool` so the audit protocol can prove its own Tier-1 set honestly.

## When to Run

- After any change to `core/integration/constants.py`
- After any change to `bizra-omega/bizra-core/src/lib.rs` or `omega.rs`
- Before any release (part of quality gates)
- On demand via `/cross-lang-sync`
