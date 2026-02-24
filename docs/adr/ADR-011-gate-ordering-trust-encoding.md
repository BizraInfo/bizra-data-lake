# ADR-011: Gate Ordering as Trust-Level Encoding

> Status: ACCEPTED
> Date: 2026-02-24
> Standing on Giants: Al-Ghazali (trust levels, 1095) · Shannon (signal theory, 1948) · Anderson (security policy, 1972)

## Context

The BIZRA organism uses quality gate chains in three languages:

| Language | Module | Gate Order | Trust Context |
|----------|--------|-----------|---------------|
| Python | `core/pci/gates.py` | SCHEMA → SIGNATURE → TIMESTAMP → REPLAY → **IHSAN → SNR** → POLICY | Untrusted peers (network messages) |
| TypeScript | `src/core/sovereign/fate-binding.ts` | SCHEMA → **SNR → IHSAN** → LICENSE | Trusted local (own inference output) |
| Rust | `bizra-core/src/sovereign/omega.rs` | Circuit check → **SNR → IHSAN** (strict enforcement) | Trusted local (compiled binaries) |

The gate ordering **differs intentionally** between Python and TypeScript/Rust.

## Decision

Gate ordering encodes the trust level of the data source:

### Untrusted sources (Python PCI): Ihsan before SNR

When processing messages from untrusted peers, ethical violations (Ihsan) represent
a more severe class of failure than signal quality (SNR). A high-SNR message that
violates Ihsan constraints (e.g., attempts to extract private data) must be rejected
immediately. Checking Ihsan first ensures malicious high-quality content is caught
before SNR scoring runs.

### Trusted sources (TypeScript FATE, Rust Omega): SNR before Ihsan

When processing output from locally-run models that have already passed constitutional
challenges (CapabilityCard verification), the dominant risk is signal quality
degradation, not ethical violation. The model was already vetted for ethical
compliance during card issuance. Checking SNR first catches the more likely failure
mode (low-quality output) before the more expensive Ihsan check.

## Consequences

1. **This ordering difference is NOT a bug** — it is trust-level encoding.
2. New gate chains must document their trust context and justify their ordering.
3. The Python PCI chain adds SIGNATURE, TIMESTAMP, and REPLAY gates (absent in TS/Rust)
   because those are adversarial-context defenses not needed for local inference.
4. If BIZRA adds a new data channel (e.g., federated inference from semi-trusted nodes),
   the gate ordering should be: Ihsan-first for untrusted, SNR-first for trusted.

## Cross-Reference

- `core/pci/gates.py:7-42` — Python gate ordering rationale
- `src/core/sovereign/fate-binding.ts:176-194` — TypeScript gate ordering with trust comment
- `bizra-omega/bizra-core/src/sovereign/omega.rs:445-466` — Rust enforcement order
- `core/integration/constants.py` — Threshold values (single source of truth)
