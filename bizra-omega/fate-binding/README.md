# fate-binding

**FATE Binding** — Formal Assertion Through Execution for BIZRA Sovereign LLM.

## Overview

Native Rust layer providing:

- **Z3 SMT Verification** — Formal proof of Ihsān constraints (≥ 0.95)
- **Post-Quantum Cryptography** — ML-DSA-87 (formerly Dilithium-5) for CapabilityCards
- **Ed25519 Signatures** — PCI envelope signing and verification
- **Gate Chain** — FATE gate enforcement pipeline
- **Node-API Bindings** — TypeScript integration via NAPI-RS

## Security

- **Post-quantum ready** — ML-DSA-87 (NIST standardized, FIPS 204)
- **Dual signature** — Ed25519 for speed + ML-DSA-87 for quantum resistance
- **Formal verification** — Z3-backed constraint proofs

## Build Requirements

- Z3 SMT solver (v4.13+)
- Rust stable toolchain
- Node.js (for NAPI bindings)

### Z3 Setup

```bash
# Ubuntu/Debian
sudo apt-get install libz3-dev

# Windows (manual)
# Download from https://github.com/Z3Prover/z3/releases
# Set Z3_SYS_Z3_HEADER=<path>/include/z3.h
```

## License

MIT
