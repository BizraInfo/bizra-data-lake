# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 9.x     | Yes       |
| < 9.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email: security@bizra.info (or use GitHub Security Advisories)
3. Include: description, reproduction steps, and potential impact
4. We will acknowledge within 48 hours and provide a timeline for resolution

## Security Architecture

BIZRA implements defense-in-depth across every layer:

### Cryptographic Foundations

| Primitive | Usage | Implementation |
|-----------|-------|----------------|
| Ed25519 | PCI receipt signatures, identity verification | `core/proof_engine/receipt.py` |
| BLAKE3 | Content hashing, domain-separated digests | `core/proof_engine/canonical.py`, Rust PyO3 |
| SHA-256 | File deduplication, Merkle-DAG verification | `core/iaas/deduplication.py` |

### Constitutional Gates

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| Ihsan (Excellence) | >= 0.95 | Hard gate on all outputs |
| SNR (Signal Quality) | >= 0.85 | Information quality filter |
| FATE (Ethics) | >= 0.95 | Fidelity, Accountability, Transparency, Ethics |
| ADL (Justice) | Gini <= 0.35 | Anti-plutocracy resource distribution |

All thresholds are defined in `core/integration/constants.py` as the single source of truth.

### Network and API Security

- **Desktop Bridge**: Localhost-only binding (`127.0.0.1:9742`), token-authenticated, nonce replay protection, rate limited (20 req/s)
- **API Server**: Explicit route exposure contract in `core/sovereign/api_exposure_policy.py`; authenticated data/control endpoints, intentionally public observability/bootstrap/verifier endpoints, Prometheus metrics, HTTPS for external communication
- **Federation**: Ed25519-signed gossip messages, Byzantine fault tolerance

### CI/CD Security

- All GitHub Actions SHA-pinned to immutable commit hashes
- Runner OS pinned to `ubuntu-24.04` (no `latest`)
- Docker images SHA256-pinned in production (`docker-compose.yml`, K8s overlays)
- Security scanning: Bandit (Python), cargo-audit (Rust), Trivy (containers), pip-audit
- Concurrency groups prevent parallel deployments

## Best Practices

- All secrets via environment variables (never hardcoded)
- HTTPS for all external communication
- Rate limiting on authenticated API data/control endpoints and bridge control surfaces; intentionally public health/metrics/verifier routes are explicitly enumerated and reviewed in CI
- Input validation at system boundaries
- No raw `eval()` — sandboxed `exec()` is permitted only in `core/zpk/kernel.py` and
  `tools/engines/agentic_cleaner.py`, gated by AST validation + restricted `__builtins__`
- Path traversal protection on file operations

## Incident Response

### Tracked Configuration Secret Exposure

If a tracked config file contains a literal credential:

1. Rotate the exposed key immediately.
2. Replace the committed value with an environment placeholder or remove it entirely.
3. Run `python scripts/ci_secret_scan.py` and verify the old pattern now fails while the remediated tree passes.
4. Assess whether git history rewrite is required.
5. If history rewrite is deferred, the minimum compensating control is:
   - confirmed key rotation
   - widened scanner coverage for tracked config files
   - CI enforcement on every push and pull request

Tracked config directories such as `.claude/`, `.claude-flow/`, `.github/`, and root-level config files are in scope for secret hygiene. A green scan that ignores those surfaces is not a valid security signal.

## Further Reading

- [Security Architecture](docs/SECURITY-ARCHITECTURE.md)
- [Threat Model](docs/THREAT-MODEL-V3.md)
- [Secure Patterns](docs/SECURE-PATTERNS.md)
- [CVE Remediation Plan](docs/CVE-REMEDIATION-PLAN.md)
- [Ihsan Compliance Matrix](docs/IHSAN_COMPLIANCE_MATRIX.md)
