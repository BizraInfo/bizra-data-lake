# SUBSTRATE TRANSITION SPEARPOINT v1.0

**Document ID:** BIZRA-STS-001
**Date Sealed:** April 7, 2026
**Author:** Cross-model convergence (GPT-5.4, Perplexity, Claude Desktop)
**Status:** FROZEN — governs the next 30 days
**Chain-to:** `59a7f1e6` (last commit before seal)
**Governing Law:** Every deliverable produces a receipt. Every receipt chains to this document's hash.

---

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

## Mission

In 30 days, BIZRA transitions from a system that claims constitutional sovereignty while running on someone else's kernel to a system that demonstrates it on its own substrate. Every day of this transition is receipted. Every deliverable is verifiable. The public front door is already live — the audience exists. This spearpoint converts the migration from an ops task into a named constitutional operation with evidence.

---

## The Three Projects (Named)

| Project | Current State | 30-Day Target |
|:--------|:-------------|:--------------|
| **BIZRA-Application** | 556K LOC, 95% operational on WSL2 | 95% operational on native Linux (no translation tax) |
| **BIZRA-CLI** | 30,874 LOC Rust, 92% ready | 95% ready, `no_std` compatibility tracked as constitutional metric |
| **BIZRA-Substrate** | 0 LOC, 5 months canon intent | 1 QEMU boot, 1 witness record, 1 frozen governance artifact |

---

## Seven Sealed Deliverables

### D1. Linux Dual-Boot on MSI Titan 18 HX

**What:** Ubuntu 24.04 LTS installed on dedicated NVMe, dual-boot with Windows 11.
**Acceptance:**
- [ ] Native Linux boot (no WSL2 translation layer)
- [ ] Full GPU passthrough: `nvidia-smi` shows RTX 4090 with 16GB VRAM
- [ ] 128GB DDR5 fully addressable
- [ ] BIZRA codebase cloned to native ext4 filesystem
- [ ] `cargo build --workspace --release` completes successfully
- [ ] All 1,122+ Rust tests pass on native Linux
- [ ] Docker Compose stack: 31 containers healthy
**Rollback:** Windows remains untouched on separate drive. Receipt chain on RAID stays read-only mountable from both OSes.
**Ihsan:** Substrate sovereignty begins. The asterisk on every constitutional claim starts to close.

### D2. Cross-OS Receipt Verification

**What:** Mount the Windows RAID read-only from Linux. BLAKE3-verify every receipt in `.proof-forge/receipts/` and `evidence/`. Produce a verification artifact signed from the Linux substrate.
**Acceptance:**
- [ ] Every receipt hash verified matches stored hash
- [ ] Every Ed25519 signature verified valid
- [ ] Verification artifact produced: `CROSS_OS_VERIFICATION_v1.json`
- [ ] Artifact includes: OS identifier, kernel version, timestamp, per-receipt verdict
- [ ] Artifact signed with Ed25519 from the Linux-resident key
**Rollback:** Verification is read-only — no rollback needed. If verification fails, that failure is itself evidence of a chain integrity issue.
**Ihsan:** The genesis chain proven continuous across two operating systems. First step toward Node1 reproducibility.

### D3. First `bizra.efi` QEMU Boot

**What:** A minimal UEFI application that boots in QEMU, prints "BIZRA Node0" to the framebuffer, emits one BLAKE3-hashed witness record to serial output, and halts.
**Acceptance:**
- [ ] `bizra.efi` compiles from Rust `no_std` + UEFI target
- [ ] Boots in QEMU/OVMF
- [ ] Prints "BIZRA Node0 — Genesis Substrate" to screen
- [ ] Emits one witness record: `{ "event": "substrate_genesis", "hash": "<blake3>", "timestamp": "<utc>" }`
- [ ] Screenshot captured as evidence
- [ ] Witness record hash stored in receipt chain
**Rollback:** QEMU is isolated. No risk to production system.
**Ihsan:** The bare-metal road has its first artifact. Zero LOC becomes non-zero. Five months of intent meets execution.

### D4. CLI Hardening to 95% with `no_std` Tracking

**What:** Complete the remaining 8% of CLI functionality. Add `no_std` compatibility as a tracked metric.
**Acceptance:**
- [ ] All 14 CLI commands operational with live data
- [ ] `bizra receipt verify <path>` works as standalone cross-process binary
- [ ] `no_std` audit completed: count of `std`-dependent vs `no_std`-clean modules
- [ ] New metric in `METRICS_CANONICAL.md`: "CLI no_std compatibility: X%"
- [ ] 205+ OmniKernel tests pass (current baseline)
**Rollback:** Feature flag any new CLI command. Revert individually.
**Ihsan:** The bridge artifact between Application and Substrate hardens. Every `no_std`-clean line is a free transit ticket to bare-metal.

### D5. P0 Gaps as CI Gates

**What:** Convert all remaining P0 items to automated CI gates that fail the build.
**Current P0 inventory (complete — reconciled across all prior analyses):**

| P0 ID | Description | Source | Current Status |
|:------|:-----------|:-------|:---------------|
| P0-IHSAN | Ihsan gate 0.85→0.95 across all code paths | March 28 audit | CLOSED (`0115016b`) |
| P0-REDIS | Redis auth contract (requirepass + bind) | March 28 audit | CLOSED (`e9d700f3`) |
| P0-RECEIPT | Receipt cross-process verification | Canon Closure Program | OPEN |
| P0-HEARTBEAT | 24-hour heartbeat (288/288 ticks) | Canon Closure Program | OPEN |
| P0-TESTCOUNT | Test count reconciliation from canonical workspace | Canon Closure Program | OPEN |
| P0-DILITHIUM | Dilithium fallback returning true for any signature | March 28 audit | STATUS UNKNOWN — no `dilithium` files found in current tree. May have been removed in earlier cleanup. Requires explicit verification. |
| P0-CROSSLANG | Cross-language sealing golden vector CI | March 28 audit | LIKELY CLOSED — CI `Cross-Language Sync` gate exists and passes. Requires explicit verification. |
| P0-REFLEX-FLAG | `BIZRA_CLOSED_LOOP_ENABLED` feature flag for reflexes | March 28 audit | STATUS UNKNOWN — no matching files found. Requires explicit verification. |
| P0-SNR | Hybrid search SNR overclaim | GPT-5.4 session | OPEN — needs recall@k benchmark |
| RFC-04 | SNR optimization dead code | SAPE audit | HAS TEST (`test_runtime_p0_fixes.py`) — status needs verification |
| RFC-03 | Query metrics double-counted | SAPE audit | HAS TEST — status needs verification |
| RFC-06 | Muraqabah bridge wrong Event attributes | SAPE audit | HAS TEST — status needs verification |

**Acceptance:**
- [ ] Each P0 has a CI gate (test or workflow check) that fails the build if regressed
- [ ] Each closed P0 has a closure receipt: commit hash + test name + verification date
- [ ] `P0_REGISTRY.md` in repo root tracks all P0s with chain-to-prior hashes
- [ ] No P0 can be removed from the registry without a closure receipt
**Rollback:** CI gates are additive. Removing a gate requires a documented constitutional exception.
**Ihsan:** "P0 closed" becomes a verifiable claim, not a list that can silently shrink.

### D6. Recall@k Benchmark for Hybrid Search

**What:** Run a proper information retrieval benchmark on the hybrid RRF search (RuVector HNSW + BM25).
**Acceptance:**
- [ ] Test set: 100+ query-document pairs with relevance labels
- [ ] Metrics: Recall@1, Recall@5, Recall@10, MRR, nDCG@10
- [ ] Results documented in `METRICS_CANONICAL.md`
- [ ] If results < claimed performance → CORRECTED label applied honestly
- [ ] If results ≥ claimed → VERIFIED label with methodology link
**Rollback:** Benchmark is measurement, not mutation. No rollback needed.
**Ihsan:** The overclaim either becomes verified or gets corrected. Both outcomes are honest.

### D7. Daily Manifest Chain (30 entries)

**What:** One manifest entry per day for the entire 30-day spearpoint, chaining every deliverable into the receipt canon.
**Acceptance:**
- [ ] Entry template: `{ "day": N, "date": "<ISO>", "deliverables_advanced": [...], "receipts_emitted": [...], "ihsan_score": <float>, "chain_hash": "<blake3_of_prior>" }`
- [ ] First entry chains to `59a7f1e6` (last commit before seal)
- [ ] Last entry chains to all 30 prior entries
- [ ] Zero gaps in the chain (if a day has no work, the entry says "no work" — it doesn't skip)
- [ ] Published to `evidence/manifests/substrate_transition/`
**Rollback:** The chain is append-only. Gaps are evidence of failure, not something to fix retroactively.
**Ihsan:** 30 days of sustained sovereignty proven by an unbroken receipt sequence. This is the evidence that converts "elite prototype" to "operationally sovereign system."

---

## Rollback Contracts

| Phase | If This Fails | Rollback To |
|:------|:-------------|:------------|
| D1 (Linux install) | GPU not detected, RAID not mountable | Stay on WSL2, continue all other deliverables |
| D2 (Cross-OS verify) | Receipt hash mismatch | Investigate chain integrity. This is a finding, not a failure. |
| D3 (bizra.efi) | UEFI target doesn't compile | Document the blockers. Ship as D3-PARTIAL with gap analysis. |
| D4 (CLI 95%) | Some commands can't be wired to live data | Ship at 93% with honest label. Document remaining gaps. |
| D5 (P0 CI gates) | Some P0s resist automation | Manual verification receipt for those items. CI for the rest. |
| D6 (Recall@k) | Performance below claims | Apply CORRECTED label. This is the right outcome. |
| D7 (Manifest chain) | Missed days | Record the gap. The chain is honest or it's not a chain. |

---

## Ihsan / Adl / Amanah Grounding

**Ihsan (Excellence):** Every deliverable has measurable acceptance criteria. No deliverable is accepted on narrative alone.

**Adl (Justice):** All three BIZRA projects (Application, CLI, Substrate) receive attention proportional to their constitutional role, not their current LOC count. The Substrate gets its first real investment after five months of intent.

**Amanah (Trustworthiness):** The genesis receipt chain stays sacred throughout. The cross-OS verification (D2) produces the first independent attestation of chain continuity. The daily manifest (D7) extends the chain for 30 consecutive days.

---

## Success Criteria (Day 30)

The spearpoint succeeds if:
1. BIZRA runs on native Linux with full GPU access
2. The genesis chain has been verified across two operating systems
3. The bare-metal road has its first artifact (even if minimal)
4. The CLI reports its own `no_std` compatibility percentage
5. Every P0 is either CI-gated or has an explicit closure receipt
6. The hybrid search overclaim is either VERIFIED or CORRECTED
7. 30 consecutive manifest entries exist with zero chain gaps

The spearpoint fails if:
- The daily manifest chain has gaps (evidence of abandonment)
- P0 items disappear from the registry without closure receipts
- Claims in the public README cannot be verified from the substrate

---

## Schedule

| Week | Focus | Key Deliverable |
|:-----|:------|:---------------|
| Week 1 (Apr 7-13) | P0 closure + heartbeat | D5 (CI gates), D7 starts, 24-hour heartbeat |
| Week 2 (Apr 14-20) | v1.0.0 + Linux migration | D1 (dual-boot), D2 (cross-OS verification) |
| Week 3 (Apr 21-27) | CLI hardening + search benchmark | D4 (CLI 95%), D6 (recall@k) |
| Week 4 (Apr 28-May 4) | Bare-metal genesis + chain closure | D3 (bizra.efi), D7 completes (30 entries) |

---

*This document is frozen at seal time. Modifications require a new version (v1.1+) with chain-to-prior hash. The hash of this document is the anchor for the next 30 days of receipts.*

بذرة واحدة تصنع غابة — والثلاثون يوماً تبدأ الآن.
