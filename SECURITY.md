# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email: security@bizra.info (or use GitHub Security Advisories)
3. Include: description, reproduction steps, and potential impact
4. We will acknowledge within 48 hours and provide a timeline for resolution

## Security Architecture

BIZRA implements multiple security layers:

- **Proof-Carrying Inference (PCI)**: Ed25519 signatures on every inference
- **FATE Gates**: Ethical firewall enforcing Ihsan >= 0.95
- **ADL Invariant**: Anti-plutocracy enforcement (Gini <= 0.40)
- **Byzantine Fault Tolerance**: Consensus protocol resilient to malicious nodes

## Best Practices

- All secrets via environment variables (never hardcoded)
- HTTPS for all external communication
- Rate limiting on all API endpoints
- Input validation at system boundaries
