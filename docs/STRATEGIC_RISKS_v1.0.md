# BIZRA Node0 Strategic Risk Register

**Version:** 1.0 | **Date:** February 10, 2026
**Classification:** Internal / Investor-Facing
**Author:** PAT-GRD (Guardian) + PAT-VAL (Validator) Audit Synthesis
**Baseline:** `.proof-forge/pat-baseline-scorecard.json`

---

## Risk Summary Matrix

| # | Risk | Severity | Likelihood | Impact | Status |
|---|------|----------|------------|--------|--------|
| SR-001 | Cross-Language Crypto Incompatibility | **CRITICAL** | High | Federation launch blocked | Open |
| SR-002 | Documentation Gap | **HIGH** | Certain | Investor confidence, onboarding friction | Open |
| SR-003 | Solo Founder Concentration | **HIGH** | Medium | Single point of failure in human capital | Open |

---

## SR-001: Cross-Language Crypto Incompatibility

### Summary

Python and Rust implementations use **different hash algorithms** in non-PCI code paths. Python defaults to SHA-256 in several modules (`epigenome.py`, `evidence_ledger.py`), while Rust uses BLAKE3 exclusively. The PCI protocol itself is aligned (both use BLAKE3 with `bizra-pci-v1:` domain separation), but **non-PCI paths diverge**.

### Why This Blocks Federation

Federation requires nodes running different implementations (Python-heavy data nodes, Rust-heavy validator nodes) to **verify each other's signatures and proofs**. If Node A (Python) signs an adaptation proof with SHA-256 and Node B (Rust) attempts verification with BLAKE3, verification fails silently.

### Affected Components

| Module | Hash Used | Expected |
|--------|-----------|----------|
| `core/pci/crypto.py` (PCI path) | BLAKE3 | BLAKE3 |
| `core/pci/epigenome.py` (adaptation) | SHA-256 | BLAKE3 |
| `core/proof_engine/evidence_ledger.py` | SHA-256 | BLAKE3 |
| `core/auth/user_store.py` (passwords) | PBKDF2-SHA256 | SHA-256 (correct for passwords) |
| `bizra-omega/bizra-core/` (all paths) | BLAKE3 | BLAKE3 |

### Remediation Plan

| Step | Action | Effort | Owner |
|------|--------|--------|-------|
| 1 | Audit all `hashlib.sha256` calls in `core/` — classify as PCI-path or local-only | 2h | PAT-RSC |
| 2 | Migrate PCI-adjacent paths (`epigenome.py`, `evidence_ledger.py`) to BLAKE3 | 4h | Engineering |
| 3 | Create cross-language interop test suite: Python sign -> Rust verify, Rust sign -> Python verify | 4h | PAT-VAL |
| 4 | Document which contexts use SHA-256 (password hashing) vs BLAKE3 (everything else) | 1h | PAT-SYN |
| 5 | Add CI gate: any new `hashlib.sha256` in `core/pci/` or `core/proof_engine/` fails lint | 1h | PAT-WRK |

**Estimated Resolution:** 12 engineering hours
**Deadline:** Before any federation beta (blocks multi-node deployment)
**Verification:** Interop test suite passes with 100% cross-language signature verification

### Investor Impact

Without resolution, BIZRA cannot demonstrate multi-node operation. Federation is a core differentiator in the investment thesis. This is the single highest-priority engineering task.

---

## SR-002: Documentation Gap

### Summary

Current documentation scores approximately **0.62 on a readability/completeness scale** (PAT assessment). While extensive internal docs exist (38+ markdown files in `docs/`), they are:

- **Internally focused** — written for the engineering team, not investors or integrators
- **Fragmented** — architecture spread across 6+ files with overlapping content
- **Missing key specs** — no standalone PCI protocol spec, no federation wire format spec, no API reference

### Current State

| Document Category | Files | Investor-Ready? |
|-------------------|-------|-----------------|
| Strategy/Vision | `BIZRA_STRATEGY_DECK_2026.md`, `BIZRA_TECHNICAL_BRIEF_INVESTORS.md` | Partial |
| Architecture | `ARCHITECTURE_BLUEPRINT_v2.3.0.md`, `ARCHITECTURE_DIAGRAMS.md` | Internal only |
| Security | `SECURITY-ARCHITECTURE.md`, `THREAT-MODEL-V3.md`, `CVE-REMEDIATION-PLAN.md` | Internal only |
| Protocol Specs | None standalone | Missing |
| API Reference | None | Missing |
| Onboarding | `QUICK-START.md` | Partial |
| Constitution | `DDAGI_CONSTITUTION_v1.1.0-FINAL.md` | Good |

### What Investors Need

1. **One-Pager Technical Summary** (exists as `BIZRA_TECHNICAL_BRIEF_INVESTORS.md` but metrics are stale)
2. **PCI Protocol Specification** — standalone, referenceable, versioned
3. **Federation Wire Format** — how nodes communicate (message schemas, transport)
4. **API Reference** — OpenAPI/Swagger for REST endpoints
5. **Verification Guide** — "how to independently verify our claims" (the most powerful investor doc)

### Remediation Plan

| Step | Action | Effort | Priority |
|------|--------|--------|----------|
| 1 | Update `BIZRA_TECHNICAL_BRIEF_INVESTORS.md` with current metrics (390K LOC, 6103 tests, 13 crates) | 2h | P0 |
| 2 | Create `docs/specs/PCI_PROTOCOL_v1.0.md` — standalone protocol specification | 8h | P0 |
| 3 | Create `docs/specs/FEDERATION_WIRE_FORMAT_v1.0.md` — message schemas and transport | 6h | P1 |
| 4 | Generate OpenAPI spec from FastAPI (`core/sovereign/api.py`) | 2h | P1 |
| 5 | Create `docs/VERIFICATION_GUIDE.md` — step-by-step independent verification | 4h | P0 |
| 6 | Consolidate architecture docs into single `ARCHITECTURE.md` (retire v2.3.0 fragments) | 4h | P2 |

**Estimated Resolution:** 26 engineering hours
**Target:** P0 items within 2 weeks, P1 within 4 weeks

### Investor Impact

Technical investors perform due diligence by reading specs. Missing protocol specifications signal "early prototype" rather than "production-grade system." The verification guide is the highest-leverage document — it converts skeptics into believers by letting them prove claims independently.

---

## SR-003: Solo Founder Concentration

### Summary

BIZRA Node0 is currently developed by a **single founder**. This creates:

- **Bus factor of 1** — if the founder is unavailable, development halts
- **Review gap** — no peer review on critical security/crypto code
- **Investor concern** — VCs weight team composition heavily in early-stage diligence

### Current Mitigations (Already in Place)

| Mitigation | Status | Effectiveness |
|------------|--------|---------------|
| Comprehensive test suite (6,103 tests) | Active | High — code is self-documenting via tests |
| CI/CD with quality gates (SNR, Ihsan, coverage) | Active | High — prevents regression without human review |
| PAT (Personal Agentic Team) as force multiplier | Active | Medium — augments but doesn't replace human judgment |
| Extensive documentation (38+ docs, CLAUDE.md) | Active | Medium — enables future contributors |
| MIT License (open source) | Active | Medium — lowers barrier to contribution |
| Pre-commit hooks (black, ruff, clippy) | Active | Medium — enforces consistency |

### Additional Mitigations Needed

| Step | Action | Timeline |
|------|--------|----------|
| 1 | **Publish contributor onboarding guide** with "good first issues" | 2 weeks |
| 2 | **External security audit** of crypto/PCI layer by third party | 4-8 weeks |
| 3 | **Identify 1-2 technical advisors** who can review architecture decisions | Ongoing |
| 4 | **Document key-person dependencies** — which decisions require founder input vs. can be delegated | 1 week |
| 5 | **Automate more** — expand CI to cover integration paths that currently need manual verification | 2-4 weeks |

### Investor Framing

The solo-founder risk is real but **mitigated by unusually high automation**:

- 6,103 automated tests (99.98% pass rate)
- Constitutional quality gates enforced by CI (not by human discipline)
- PAT agents provide 5x research/review throughput
- Dual-language implementation (Python + Rust) means no single-language dependency

**Narrative:** "The founder built the system to not need the founder." The test suite, quality gates, and constitutional constraints encode engineering judgment into the codebase itself. This is a feature, not a bug — it's how you build a system designed for 8 billion nodes.

---

## Cross-Risk Dependencies

```
SR-001 (Crypto) ──blocks──> Federation Demo ──blocks──> Investor Confidence
                                                              │
SR-002 (Docs) ──────────────────────────────blocks──> Investor Due Diligence
                                                              │
SR-003 (Solo) ──────────────────────────────affects──> Execution Timeline
```

**Critical Path:** SR-001 must be resolved first. It is the only risk that blocks a technical milestone (federation). SR-002 and SR-003 are addressable in parallel.

---

## Recommended Priority Order

1. **SR-001** (12h) — Unify crypto hashing. Unblocks federation demo.
2. **SR-002 P0 items** (14h) — Update investor brief, write verification guide, PCI spec.
3. **SR-003 Step 1-2** (2-8 weeks) — Contributor guide + external security audit.
4. **SR-002 P1-P2 items** (12h) — API reference, federation spec, architecture consolidation.

**Total engineering investment:** ~38 hours + external audit engagement.

---

## Verification

This risk register is derived from:
- PAT-GRD Security Audit (2026-02-10) — SEC-001 through SEC-010
- PAT-VAL Quality Matrix (2026-02-10) — Coverage and gate compliance
- PAT-SYN Integration Map (2026-02-10) — Cross-language alignment scorecard
- PAT-WRK Operations Report (2026-02-10) — CI/CD and deployment gaps
- Baseline Scorecard: `.proof-forge/pat-baseline-scorecard.json`

Standing on Giants: Shannon (SNR) | Lamport (BFT) | Besta (GoT) | Vaswani (Transformers) | Anthropic (Constitutional AI)
