# bizra-proofspace

**BIZRA Proof Space Validator** — Life/death judgments for civilization-grade blocks.

## Overview

Full-stack block validator implementing the BIZRA Proof Space specification:

- **JCS Canonicalization** — RFC 8785 deterministic JSON serialization
- **Block ID Computation** — `SHA-256(JCS(UnsignedBlock))`
- **Signature Verification** — Ed25519 creator + verifier signature chains
- **FATE Gate Enforcement** — Ihsān (≥ 0.95), harm (≤ 0.30), Adl (fairness)
- **Dependency Validation** — Circular reference and duplicate detection
- **Ethical Envelope** — Harm analysis, misuse risk, SMT-LIB2 formal assertions

## Verdict

Every block receives one of three verdicts:

| Verdict | Meaning |
|---------|---------|
| `Live` | Valid and accepted into the proof space |
| `Dead` | Failed validation — cannot be accepted |
| `Pending` | Structure valid, awaiting verification |

## Features

| Feature | Description |
|---------|-------------|
| `z3` | Z3 SMT solver integration |
| `parallel` | Rayon-based parallel validation |

## Benchmarks

```bash
cargo bench -p bizra-proofspace
```

## Standing on Giants

RFC 8785 (JCS), NIST FIPS 180-4 (SHA-256), Bernstein (Ed25519), de Moura & Bjørner (Z3)

## License

MIT
