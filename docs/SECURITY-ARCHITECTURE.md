# BIZRA V3 Security Architecture

**Version:** 3.0.0
**Date:** 2026-02-03
**Author:** Security Architect (Agent #2)
**Classification:** ARCHITECTURE DOCUMENT

---

## 1. Overview

The BIZRA Security Architecture provides defense-in-depth protection for a decentralized AI governance system designed to scale to 8 billion nodes. This document describes the security layers, cryptographic infrastructure, and constitutional enforcement mechanisms.

---

## 2. Security Principles

### 2.1 Core Principles

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Zero Trust** | Never trust, always verify | Signature verification on all messages |
| **Defense in Depth** | Multiple security layers | 4-tier protection (Network, Protocol, Application, Storage) |
| **Least Privilege** | Minimum necessary access | Role-based pattern access |
| **Fail Secure** | Secure defaults on failure | Default-deny policy gates |
| **Accountability** | All actions auditable | Signed envelopes, audit logs |

### 2.2 Constitutional AI Principles (FATE)

The FATE Gate enforces constitutional AI principles:

```
F - Fidelity:      Truth and accuracy in all outputs
A - Accountability: Audit trail for all decisions
T - Transparency:   Explainable and understandable actions
E - Ethics:         Harm prevention and value alignment
```

---

## 3. Security Layers

### 3.1 Layer Architecture

```
+-----------------------------------------------------------------+
|                   LAYER 4: STORAGE PROTECTION                    |
|  - Vault Encryption (Fernet + PBKDF2)                           |
|  - Cache Limits (MAX_PATTERNS_CACHE=1000)                       |
|  - TTL Expiration (NONCE_TTL=300s)                             |
+-----------------------------------------------------------------+
                              |
+-----------------------------------------------------------------+
|                   LAYER 3: APPLICATION SECURITY                  |
|  - PCI Gate Chain (7 gates)                                     |
|  - FATE Gate (4-dimensional scoring)                            |
|  - Ihsan Threshold (0.95)                                       |
|  - SNR Threshold (0.85)                                         |
+-----------------------------------------------------------------+
                              |
+-----------------------------------------------------------------+
|                   LAYER 2: PROTOCOL SECURITY                     |
|  - Ed25519 Signatures                                           |
|  - BLAKE3 Hashing                                               |
|  - Domain Separation                                            |
|  - Replay Protection                                            |
+-----------------------------------------------------------------+
                              |
+-----------------------------------------------------------------+
|                   LAYER 1: NETWORK SECURITY                      |
|  - UDP Transport (Port 7654)                                    |
|  - Per-peer Rate Limiting (planned)                             |
|  - DTLS Encryption (planned)                                    |
+-----------------------------------------------------------------+
```

### 3.2 Layer Details

#### Layer 1: Network Security

| Control | Status | Description |
|---------|--------|-------------|
| UDP Transport | ACTIVE | Gossip protocol on port 7654 |
| Signature Verification | ACTIVE | All messages signed with Ed25519 |
| Rate Limiting | PLANNED | Per-peer message rate limits |
| DTLS Encryption | PLANNED | Transport layer encryption |

#### Layer 2: Protocol Security

| Control | Status | Description |
|---------|--------|-------------|
| Ed25519 Signatures | ACTIVE | All messages cryptographically signed |
| BLAKE3 Hashing | ACTIVE | Fast, secure content hashing |
| Domain Separation | ACTIVE | Prefix "bizra-pci-v1:" prevents cross-protocol attacks |
| Nonce-Based Replay | ACTIVE | 300-second TTL with 10K cache limit |
| Clock Skew Tolerance | ACTIVE | 120-second window for timestamp validation |

#### Layer 3: Application Security

| Control | Status | Description |
|---------|--------|-------------|
| PCI Gate Chain | ACTIVE | 7-gate verification: Schema, Signature, Timestamp, Replay, Ihsan, SNR, Policy |
| FATE Gate | ACTIVE | 4-dimensional constitutional AI validation |
| Ihsan Threshold | ACTIVE | 0.95 minimum for pattern propagation |
| SNR Threshold | ACTIVE | 0.85 minimum for signal quality |
| Tool Thresholds | ACTIVE | Risk-based thresholds (Bash: 0.98, Write: 0.96) |

#### Layer 4: Storage Security

| Control | Status | Description |
|---------|--------|-------------|
| Vault Encryption | ACTIVE | Fernet with PBKDF2 key derivation |
| Cache Limits | ACTIVE | MAX_PATTERNS_CACHE=1000, MAX_NONCE_CACHE=10000 |
| TTL Expiration | ACTIVE | Automatic cleanup of expired entries |
| Audit Logging | PARTIAL | JSON logs (integrity protection planned) |

---

## 4. Cryptographic Infrastructure

### 4.1 Algorithm Suite

```
+------------------------------------------------------------------+
|                    BIZRA CRYPTOGRAPHIC SUITE                      |
+------------------------------------------------------------------+
| Category          | Algorithm    | Standard    | Key Size        |
+------------------------------------------------------------------+
| Digital Signature | Ed25519      | RFC 8032    | 256-bit         |
| Hashing           | BLAKE3       | Modern      | 256-bit output  |
| Canonicalization  | JCS          | RFC 8785    | N/A             |
| Key Derivation    | PBKDF2       | RFC 2898    | 256-bit output  |
| Encryption        | Fernet       | Python lib  | 128-bit AES     |
+------------------------------------------------------------------+
```

### 4.2 Key Hierarchy

```
+------------------------------------------------------------------+
|                         KEY HIERARCHY                             |
+------------------------------------------------------------------+
|                                                                   |
|   +-------------------+                                           |
|   | Master Secret     | <-- User-provided or generated           |
|   +-------------------+                                           |
|            |                                                      |
|            v PBKDF2                                               |
|   +-------------------+                                           |
|   | Vault Key         | <-- Encrypts stored secrets              |
|   +-------------------+                                           |
|                                                                   |
|   +-------------------+                                           |
|   | Node Private Key  | <-- Ed25519, 32 bytes                    |
|   +-------------------+                                           |
|            |                                                      |
|            v Derive                                               |
|   +-------------------+                                           |
|   | Node Public Key   | <-- Ed25519, 32 bytes, shared            |
|   +-------------------+                                           |
|                                                                   |
+------------------------------------------------------------------+
```

### 4.3 Signature Flow

```
+------------------------------------------------------------------+
|                        SIGNATURE FLOW                             |
+------------------------------------------------------------------+
|                                                                   |
|   [Message Data]                                                  |
|        |                                                          |
|        v canonical_json()                                         |
|   [Canonical JSON]    <-- RFC 8785 JCS, ensure_ascii=True        |
|        |                                                          |
|        v domain_separated_digest()                                |
|   [BLAKE3 Digest]     <-- Prefix: "bizra-pci-v1:"                |
|        |                                                          |
|        v sign_message()                                           |
|   [Ed25519 Signature] <-- 64 bytes                               |
|                                                                   |
+------------------------------------------------------------------+
```

---

## 5. PCI Gate Chain

### 5.1 Gate Ordering

The PCI Gate Chain processes envelopes in cost-optimized order:

```
+------------------------------------------------------------------+
|                      PCI GATE CHAIN                               |
+------------------------------------------------------------------+
|                                                                   |
|   TIER 1: CHEAP (<10ms)                                          |
|   +--------+  +----------+  +-----------+  +--------+            |
|   | SCHEMA |->| SIGNATURE|->| TIMESTAMP |->| REPLAY |            |
|   +--------+  +----------+  +-----------+  +--------+            |
|                                                                   |
|   TIER 2: MEDIUM (<150ms)                                        |
|   +-------+  +-----+  +--------+                                 |
|   | IHSAN |->| SNR |->| POLICY |                                 |
|   +-------+  +-----+  +--------+                                 |
|                                                                   |
|   Result: ACCEPT or REJECT with code                             |
|                                                                   |
+------------------------------------------------------------------+
```

### 5.2 Gate Details

| Gate | Purpose | Failure Code |
|------|---------|--------------|
| SCHEMA | Validate envelope structure | REJECT_SCHEMA |
| SIGNATURE | Verify Ed25519 signature | REJECT_SIGNATURE |
| TIMESTAMP | Check clock skew (120s) | REJECT_TIMESTAMP_FUTURE/STALE |
| REPLAY | Check nonce uniqueness | REJECT_NONCE_REPLAY |
| IHSAN | Verify Ihsan >= 0.95 | REJECT_IHSAN_BELOW_MIN |
| SNR | Verify SNR >= 0.85 | REJECT_SNR_BELOW_MIN |
| POLICY | Verify constitution hash | REJECT_POLICY_MISMATCH |

### 5.3 Gate Ordering Rationale

**Why IHSAN before SNR in PCI?**

PCI envelopes come from potentially untrusted peers. Ethical violations (Ihsan) are more severe than signal quality issues (SNR):

1. **Fail-Fast on Ethical Violations:** A high-SNR malicious message is more dangerous than a noisy benign message
2. **Security Posture:** Attackers may craft high-SNR messages to bypass quality filters
3. **Resource Protection:** Rejecting ethical violations early saves SNR computation

---

## 6. FATE Gate Architecture

### 6.1 Dimensional Scoring

```
+------------------------------------------------------------------+
|                    FATE GATE SCORING                              |
+------------------------------------------------------------------+
|                                                                   |
|   FIDELITY (25%)                                                  |
|   - Secret pattern detection                                      |
|   - Hardcoded credential checking                                 |
|   - Score penalty: 0.7 for each violation                        |
|                                                                   |
|   ACCOUNTABILITY (25%)                                            |
|   - Input traceability                                            |
|   - Empty input detection                                         |
|   - Score penalty: 0.9 for missing input                         |
|                                                                   |
|   TRANSPARENCY (25%)                                              |
|   - Obfuscation detection (base64 pipes)                         |
|   - Suspicious command patterns (curl | sh)                      |
|   - Score penalty: 0.7-0.8 for obfuscation                       |
|                                                                   |
|   ETHICS (25%)                                                    |
|   - Blocked pattern matching                                      |
|   - Dangerous command detection                                   |
|   - Score: 0.0 for any blocked pattern                           |
|                                                                   |
|   COMPOSITE = (F * A * T * E) ^ 0.25                             |
|                                                                   |
+------------------------------------------------------------------+
```

### 6.2 Blocked Patterns

```python
BLOCKED_PATTERNS = [
    "rm -rf /",           # Recursive root deletion
    "format c:",          # Windows format
    ":(){:|:&};:",        # Fork bomb
    "dd if=/dev/zero",    # Disk overwrite
    "mkfs.",              # Filesystem creation
    "> /dev/sda",         # Disk overwrite
]
```

### 6.3 Tool-Specific Thresholds

| Tool | Threshold | Rationale |
|------|-----------|-----------|
| Bash | 0.98 | Command execution - highest risk |
| Write | 0.96 | File modification - high risk |
| Edit | 0.96 | Code changes - high risk |
| WebFetch | 0.95 | External access - medium risk |
| Read | 0.90 | Read-only - lower risk |

---

## 7. Consensus Security

### 7.1 BFT Properties

| Property | Guarantee |
|----------|-----------|
| **Safety** | Honest nodes never commit conflicting patterns |
| **Liveness** | Eventually commits if proposer is honest |
| **Fault Tolerance** | Tolerates f Byzantine faults where n >= 3f+1 |

### 7.2 Vote Verification

```
+------------------------------------------------------------------+
|                    VOTE VERIFICATION                              |
+------------------------------------------------------------------+
|                                                                   |
|   1. Check voter_id in peer registry                             |
|      - FAIL if unregistered: "Vote from unregistered peer"       |
|                                                                   |
|   2. Compare vote.public_key with registered key                 |
|      - FAIL if mismatch: "Public key mismatch"                   |
|                                                                   |
|   3. Verify signature with REGISTERED key (not vote.public_key)  |
|      - FAIL if invalid: "Invalid signature"                      |
|                                                                   |
|   4. Check for duplicate votes                                   |
|      - FAIL if already voted: silent rejection                   |
|                                                                   |
|   5. Accept vote and check quorum                                |
|      - COMMIT if votes >= (2n/3 + 1)                            |
|                                                                   |
+------------------------------------------------------------------+
```

### 7.3 Quorum Calculation

```python
def calculate_quorum(node_count: int) -> int:
    """BFT quorum: 2n/3 + 1"""
    return (2 * node_count // 3) + 1

# Examples:
# 3 nodes: quorum = 3 (all must agree)
# 4 nodes: quorum = 3
# 7 nodes: quorum = 5
# 10 nodes: quorum = 7
```

---

## 8. Gossip Protocol Security

### 8.1 SWIM Protocol Security

| Feature | Implementation | Security Benefit |
|---------|----------------|------------------|
| Incarnation Numbers | Lamport-style counters | Prevents stale state injection |
| Signature Verification | Ed25519 on all messages | Prevents message forgery |
| Dedup Cache Protection | Signature-first verification | Prevents cache poisoning |
| Peer Registry | Public key binding | Prevents identity spoofing |

### 8.2 Message Types

| Type | Purpose | Signed |
|------|---------|--------|
| PING | Liveness check | YES |
| PING_ACK | Liveness response | YES |
| ANNOUNCE | Node joining | YES |
| LEAVE | Graceful departure | YES |
| PATTERN_SHARE | Pattern propagation | YES |
| PROPOSE | BFT proposal | YES |
| VOTE | BFT vote | YES |
| COMMIT | BFT commit | YES |

---

## 9. Data Protection

### 9.1 Vault Architecture

```
+------------------------------------------------------------------+
|                      VAULT ARCHITECTURE                           |
+------------------------------------------------------------------+
|                                                                   |
|   [User Password]                                                 |
|         |                                                         |
|         v PBKDF2 (100K iterations, SHA256)                       |
|   [Derived Key]                                                   |
|         |                                                         |
|         v Fernet (AES-128-CBC + HMAC-SHA256)                     |
|   [Encrypted Value]                                               |
|         |                                                         |
|         v Base64 Encode                                           |
|   [Stored in vault.json]                                          |
|                                                                   |
+------------------------------------------------------------------+
```

### 9.2 Storage Security

| Data Type | Protection | Location |
|-----------|------------|----------|
| Private Keys | Hex encoding, memory only | FederationNode |
| Vault Secrets | Fernet encryption | vault.json |
| Patterns | Cache limits, TTL | Memory |
| Nonces | TTL + size limits | Memory |
| Audit Logs | JSON (integrity planned) | .claude/logs/ |

---

## 10. Security Monitoring

### 10.1 Security Events

| Event | Severity | Log Location |
|-------|----------|--------------|
| Signature Verification Failed | WARNING | SECURITY logger |
| Unregistered Peer Vote | ERROR | CONSENSUS logger |
| Public Key Mismatch | ERROR | CONSENSUS logger |
| Ihsan Below Threshold | WARNING | PROPAGATION logger |
| FATE Gate Block | INFO | fate_gate.jsonl |
| Nonce Replay Detected | WARNING | GATES logger |

### 10.2 Metrics

| Metric | Purpose | Threshold |
|--------|---------|-----------|
| Signature Failures/min | Detect attack | > 10 |
| Rejected Patterns/min | Detect spam | > 5 |
| FATE Blocks/min | Detect misuse | > 3 |
| Nonce Cache Size | Detect DoS | > 9000 |
| Pattern Cache Size | Detect accumulation | > 900 |

---

## 11. Deployment Recommendations

### 11.1 Production Checklist

- [ ] DTLS transport encryption enabled
- [ ] Vault master secret not in environment variables
- [ ] Audit log integrity protection enabled
- [ ] Hook files signed and verified
- [ ] Rate limiting configured per peer
- [ ] Monitoring dashboards configured
- [ ] Alerting rules defined

### 11.2 Hardening Guidelines

1. **Network:**
   - Restrict UDP/7654 to known peers
   - Enable IP reputation filtering
   - Monitor for port scanning

2. **Keys:**
   - Generate keys offline
   - Store private keys in HSM (if available)
   - Implement key rotation schedule

3. **Logging:**
   - Ship logs to secure SIEM
   - Enable tamper-evident logging
   - Retain logs for 90 days minimum

4. **Updates:**
   - Subscribe to security advisories
   - Test updates in staging first
   - Have rollback procedure ready

---

## 12. Related Documents

| Document | Purpose |
|----------|---------|
| `/mnt/c/BIZRA-DATA-LAKE/docs/THREAT-MODEL-V3.md` | Detailed threat analysis |
| `/mnt/c/BIZRA-DATA-LAKE/docs/CVE-REMEDIATION-PLAN.md` | Vulnerability tracking |
| `/mnt/c/BIZRA-DATA-LAKE/docs/SECURE-PATTERNS.md` | Secure coding patterns |
| `/mnt/c/BIZRA-DATA-LAKE/core/integration/constants.py` | Authoritative thresholds |

---

*Document maintained by Security Architect Agent*
*Architecture Version: 3.0.0*
*Last Updated: 2026-02-03*
