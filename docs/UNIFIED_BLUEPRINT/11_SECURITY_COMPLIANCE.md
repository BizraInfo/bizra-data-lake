# Module 11 — Security & Compliance

> **Domain:** SAP v0, hardening, certification, audits, auth, RBAC
> **Source Specs:** SAP v0, Alpha-100 Sprint 3, Phase 60 (boundary), Track 1/2 hardening
> **Key Paths:** `core/auth/`, `core/vault/`, `core/pci/`, `scripts/`

## 11.1 API Authentication Guard

**Status:** [x] BUILT
**Path:** `core/sovereign/api.py` — `_authenticate_http_request(request)` 3-tuple check

8 POST routes auth-guarded: validate, spearpoint/{reproduce,improve,pattern},
sel/retrieve, memory/search, cognitive/fuse, judgment/simulate.
Intentionally open: /v1/verify/* (auditors), /v1/auth/* (bootstrap).

---

## 11.2 Node Identity (Ed25519)

**Status:** [x] BUILT
**Path:** `core/sovereign/` (identity management)

Ed25519 key pairs for all signing. Persistent signer at
`sovereign_state/mission_signer.json`. PyNaCl (Python), ed25519-dalek (Rust).

---

## 11.3 FATE Constitutional Firewall

**Status:** [x] BUILT
**Path:** `core/pci/gates.py`, `core/pci/fate.py`

Default-deny. Every action passes FATE before execution.
(Shared with Module 01, listed here for security coverage.)

---

## 11.4 Bandit SAST (Python)

**Status:** [x] BUILT
**Path:** `.github/workflows/ci.yml` (security scan job)

Static analysis for Python security issues. Runs in CI.

---

## 11.5 Cargo-Audit (Rust)

**Status:** [x] BUILT
**Path:** `.github/workflows/ci.yml` (security scan job)

Dependency vulnerability scanning for Rust crates.

---

## 11.6 Pip-Audit (Python Dependencies)

**Status:** [x] BUILT
**Path:** `.github/workflows/ci.yml` (security scan job)

Python dependency vulnerability scanning.

---

## 11.7 Trivy Container Scan

**Status:** [x] BUILT
**Path:** `.github/workflows/ci.yml` (security scan job)

Container image vulnerability scanning for Docker images.

---

## 11.8 Dependency Governance

**Status:** [x] BUILT
**Path:** `scripts/ci_dependency_governance.py`

Automated dependency review and governance policy enforcement.

**Tests:** `tests/scripts/test_dependency_governance.py`

---

## 11.9 Security Headers Middleware

**Status:** [x] BUILT
**Path:** `deploy/argocd/rollouts.yaml` (Traefik Middleware `bizra-security-headers`)

6 headers configured:
1. `X-Frame-Options: DENY` — Clickjacking protection
2. `X-Content-Type-Options: nosniff` — MIME sniffing prevention
3. `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload` — HSTS
4. `Referrer-Policy: strict-origin-when-cross-origin` — Referrer leakage control
5. `Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()` — Feature restrictions
6. `Content-Security-Policy` — Script/style/connect-src allowlisting, frame-ancestors none

---

## 11.10 Sovereign Vault (Secrets at Rest)

**Status:** [x] BUILT
**Path:** `core/vault/vault.py` (~400 LOC)

**SovereignVault class:**
- Encryption: Fernet (AES-128-CBC + HMAC-SHA256)
- Key derivation: PBKDF2-SHA256, 600,000 iterations (OWASP 2023)
- Per-entry salt: 32 bytes unique random
- Two-phase commit key rotation (decrypt all -> re-encrypt all, atomic)
- Fail-closed: corrupted index raises, never silently continues

---

## 11.11 JWT Auth + Rate Limiting

**Status:** [x] BUILT
**Path:** `core/auth/` (~150 LOC across 3 files)

- `jwt_auth.py` — HMAC-SHA256, 15min access / 7d refresh tokens
- `middleware.py` — Bearer token + X-API-Key fallback, per-user rate limiting (token bucket, 100 req/min)
- `user_store.py` — user database + API key management
- Token blacklisting: bounded eviction (max 10K entries)

---

## 11.12 Audit Trail

**Status:** [~] PARTIAL
**Path:** Evidence ledger provides action trail
**Gap:** No dedicated security audit log, no tamper-evident audit export

---

## 11.13 Declaration Hash Manifest

**Status:** [~] PARTIAL
**Path:** `scripts/ops/declaration_hash_manifest.py`
**Tests:** `tests/scripts/test_declaration_hash_manifest.py`
**Gap:** Exists but needs integration into release pipeline

---

## 11.14 RBAC (Role-Based Access Control)

**Status:** [ ] NOT BUILT
**Spec:** Required for multi-user/multi-agent access control
**Gap:** No role definitions, no permission matrix, no role assignment

### Pseudocode
```
class RBACEngine:
    ROLES = {
        "owner": ["*"],  # Full access
        "operator": ["read", "execute", "configure"],
        "auditor": ["read", "verify"],
        "agent": ["execute"],  # Scoped to assigned tasks
        "observer": ["read"],
    }

    def check_permission(self, identity: str, action: str, resource: str) -> bool:
        role = self.get_role(identity)
        permissions = self.ROLES.get(role, [])
        return "*" in permissions or action in permissions
```

---

## 11.15 TLS/mTLS Configuration

**Status:** [ ] NOT BUILT
**Spec:** Required for inter-service encryption
**Gap:** All internal communication is plaintext HTTP. No cert management.

### Pseudocode
```
# deploy/k8s/base/tls/
# cert-manager ClusterIssuer for automatic TLS
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: bizra-ca
spec:
  selfSigned: {}

# Service mesh or manual mTLS between pods
# Each service gets a TLS certificate from bizra-ca
```

---

## 11.16 DAST (Dynamic Testing)

**Status:** [x] BUILT
(See Module 10.19 — ZAP Baseline Scan in CI pipeline)

---

## 11.17 Container Image Signing

**Status:** [ ] NOT BUILT
(See Module 10.20 — shared concern)

---

## 11.18 Compliance Certification Framework

**Status:** [ ] NOT BUILT
**Spec:** SOC2/ISO 27001 readiness
**Gap:** No compliance mapping, no control matrix, no evidence collection automation

### Pseudocode
```
class ComplianceCertificationEngine:
    """Map BIZRA controls to compliance frameworks"""

    FRAMEWORKS = {
        "soc2": {
            "CC1.1": "constitutional_gate",      # Control environment
            "CC6.1": "fate_gates",                # Logical access
            "CC7.2": "evidence_ledger",           # System monitoring
            "CC8.1": "ci_pipeline",               # Change management
        },
        "iso27001": {
            "A.9.1": "rbac_engine",               # Access control
            "A.12.6": "dependency_governance",     # Vulnerability management
            "A.14.2": "ci_security_scan",          # Secure development
        }
    }

    def assess(self, framework: str) -> ComplianceReport:
        controls = self.FRAMEWORKS[framework]
        results = {}
        for control_id, component in controls.items():
            results[control_id] = self.evaluate_control(component)
        return ComplianceReport(framework, results)
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 11.1 API Auth Guard | BUILT | 8 routes |
| 11.2 Ed25519 Identity | BUILT | Full |
| 11.3 FATE Firewall | BUILT | Default-deny |
| 11.4 Bandit SAST | BUILT | CI |
| 11.5 Cargo-Audit | BUILT | CI |
| 11.6 Pip-Audit | BUILT | CI |
| 11.7 Trivy Scan | BUILT | CI |
| 11.8 Dep Governance | BUILT | Tests |
| 11.9 Security Headers | BUILT | 6/6 headers |
| 11.10 Sovereign Vault | BUILT | 400 LOC |
| 11.11 JWT + Rate Limit | BUILT | 150 LOC |
| 11.12 Audit Trail | PARTIAL | Evidence only |
| 11.13 Hash Manifest | PARTIAL | Not in pipeline |
| 11.14 RBAC | NOT BUILT | Zero |
| 11.15 TLS/mTLS | NOT BUILT | Zero |
| 11.16 DAST | BUILT | ZAP in CI |
| 11.17 Container Signing | NOT BUILT | Zero |
| 11.18 Compliance Framework | NOT BUILT | Zero |
| **TOTAL** | **12/18 + 2P + 4N** | **72%** |
