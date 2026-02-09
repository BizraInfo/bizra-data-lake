# bizra-core

**BIZRA Core** — The Sovereign Kernel.

## Overview

Foundation crate for the entire BIZRA ecosystem. Provides:

- **Identity** — Ed25519 keypair management (`NodeIdentity`, `NodeId`)
- **PCI Protocol** — Provably Correct Inference envelopes (create, sign, verify)
- **Constitutional Governance** — FATE gates, Ihsān threshold (≥ 0.95), SNR enforcement
- **Islamic Finance Layer** — No Riba, Zakat distribution, Halal services, risk sharing
- **PAT/SAT Agent System** — Personal (7) and Shared (5) Agentic Team minting

## Performance

- SIMD-accelerated gate validation (2× throughput)
- Batch signature verification (4× throughput)
- Parallel BLAKE3 hashing (2× throughput)

## Features

| Feature | Description |
|---------|-------------|
| `default` | Standard build |
| `parallel` | Enable Rayon-based parallelism |
| `dev-mode` | Relaxed thresholds for local testing (**never enable in production**) |

## Standing on Giants

Bernstein (Ed25519), O'Connor (BLAKE3), Al-Ghazali (Maqasid al-Shariah), Shannon (SNR)

## License

MIT
