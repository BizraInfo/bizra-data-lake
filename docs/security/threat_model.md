# Threat Model (Template - STRIDE)

Scope: Identify and mitigate key threats (OWASP + AI-specific) with explicit owners and evidence.

## Assets
- Secrets (tokens, DSNs, keys)
- Evidence receipts and audit trail
- User data (PII) and prompts
- Tool execution surfaces (hooks, MCP/A2A)

## Threats (starter)
- Spoofing: forged requests, token theft
- Tampering: evidence manipulation, config drift
- Repudiation: actions without receipts
- Information disclosure: secrets/logs/prompts leaked
- Denial of service: cost-exhaustion, rate-limit bypass
- Elevation of privilege: hook execution, tool injection

## Required controls (baseline)
- Secret scanning + no tracked secrets
- Authn/z + least privilege tool scopes
- Receipts for all actions + sealing
- DAST/SAST schedules (e.g., OWASP ZAP baseline in CI)

