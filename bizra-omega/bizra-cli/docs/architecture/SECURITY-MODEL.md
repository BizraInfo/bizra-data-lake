# Security Model

Security architecture and threat mitigation for BIZRA.

## Table of Contents

1. [Security Principles](#security-principles)
2. [Threat Model](#threat-model)
3. [Authentication & Identity](#authentication--identity)
4. [Authorization & Access Control](#authorization--access-control)
5. [Data Protection](#data-protection)
6. [Communication Security](#communication-security)
7. [Inference Security](#inference-security)
8. [FATE Gates as Security](#fate-gates-as-security)
9. [Audit & Monitoring](#audit--monitoring)
10. [Incident Response](#incident-response)

---

## Security Principles

### Core Principles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BIZRA SECURITY PRINCIPLES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. SOVEREIGNTY FIRST                                                      │
│      ───────────────────                                                    │
│      User maintains full control over their data and agents.                │
│      No external dependency required for core operations.                   │
│                                                                             │
│   2. DEFENSE IN DEPTH                                                       │
│      ──────────────────                                                     │
│      Multiple layers of security controls.                                  │
│      No single point of failure.                                            │
│                                                                             │
│   3. LEAST PRIVILEGE                                                        │
│      ─────────────────                                                      │
│      Agents and processes get minimum required permissions.                 │
│      Explicit capability grants.                                            │
│                                                                             │
│   4. ZERO TRUST                                                             │
│      ──────────                                                             │
│      Verify everything, trust nothing.                                      │
│      Validate all inputs, sign all outputs.                                 │
│                                                                             │
│   5. GUARDIAN OVERSIGHT                                                     │
│      ──────────────────                                                     │
│      All significant actions reviewed by Guardian.                          │
│      FATE gates enforce ethical constraints.                                │
│                                                                             │
│   6. CRYPTOGRAPHIC INTEGRITY                                                │
│      ─────────────────────────                                              │
│      Ed25519 signatures on all critical operations.                         │
│      Verifiable audit trail.                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Security Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SECURITY LAYERS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Layer 7: ETHICAL (FATE Gates)                                            │
│   ─────────────────────────────                                            │
│   Harm prevention, bias detection, confidence thresholds                    │
│                                                                             │
│   Layer 6: APPLICATION                                                      │
│   ───────────────────                                                       │
│   Input validation, output sanitization, secure coding                      │
│                                                                             │
│   Layer 5: AGENT                                                            │
│   ─────────────                                                             │
│   Capability-based access, agent authentication, action audit              │
│                                                                             │
│   Layer 4: SESSION                                                          │
│   ──────────────                                                            │
│   Session tokens, context isolation, timeout management                     │
│                                                                             │
│   Layer 3: TRANSPORT                                                        │
│   ───────────────                                                           │
│   TLS 1.3, certificate pinning, secure channels                            │
│                                                                             │
│   Layer 2: STORAGE                                                          │
│   ─────────────                                                             │
│   Encryption at rest, secure key management, access logging                │
│                                                                             │
│   Layer 1: INFRASTRUCTURE                                                   │
│   ─────────────────────                                                     │
│   OS hardening, network isolation, resource limits                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Threat Model

### Threat Categories

| Category | Description | Mitigation |
|----------|-------------|------------|
| **Injection** | Prompt injection, command injection | Input sanitization, parameterization |
| **Data Exfiltration** | Unauthorized data access | Encryption, access control, audit |
| **Privilege Escalation** | Agent/user gains extra permissions | Capability model, least privilege |
| **Denial of Service** | Resource exhaustion | Rate limiting, quotas |
| **Model Manipulation** | Adversarial inputs to LLM | FATE gates, input validation |
| **Federation Attacks** | Malicious peer nodes | PBFT consensus, reputation |
| **Supply Chain** | Compromised dependencies | Vendoring, signature verification |

### Attack Surface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ATTACK SURFACE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   EXTERNAL                           INTERNAL                               │
│   ────────                           ────────                               │
│                                                                             │
│   ┌───────────────┐                  ┌───────────────┐                     │
│   │ User Input    │                  │ Agent Actions │                     │
│   │ (CLI/TUI)     │                  │               │                     │
│   └───────┬───────┘                  └───────┬───────┘                     │
│           │                                  │                              │
│           ▼                                  ▼                              │
│   ┌───────────────┐                  ┌───────────────┐                     │
│   │ LLM Backend   │                  │ Memory System │                     │
│   │ Connection    │                  │               │                     │
│   └───────┬───────┘                  └───────┬───────┘                     │
│           │                                  │                              │
│           ▼                                  ▼                              │
│   ┌───────────────┐                  ┌───────────────┐                     │
│   │ Federation    │                  │ File System   │                     │
│   │ Protocol      │                  │ Access        │                     │
│   └───────┬───────┘                  └───────┬───────┘                     │
│           │                                  │                              │
│           ▼                                  ▼                              │
│   ┌───────────────┐                  ┌───────────────┐                     │
│   │ MCP Servers   │                  │ Execution     │                     │
│   │               │                  │ Engine        │                     │
│   └───────────────┘                  └───────────────┘                     │
│                                                                             │
│   Each attack surface has dedicated controls and monitoring.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### STRIDE Analysis

| Threat | Example | Control |
|--------|---------|---------|
| **S**poofing | Impersonating another agent | Ed25519 signatures, mutual auth |
| **T**ampering | Modifying messages in transit | TLS, message signing |
| **R**epudiation | Denying actions taken | Immutable audit log |
| **I**nformation Disclosure | Leaking sensitive data | Encryption, access control |
| **D**enial of Service | Flooding with requests | Rate limiting, quotas |
| **E**levation of Privilege | Agent gaining Guardian powers | Capability enforcement |

---

## Authentication & Identity

### Node Identity

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NODE IDENTITY                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Genesis Block                                                             │
│   ─────────────                                                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ {                                                                    │  │
│   │   "node_id": "node0_ce5af35c848ce889",                              │  │
│   │   "genesis_hash": "a7f68f1f74f2c0898cb1f1db...",                    │  │
│   │   "public_key": "ed25519:abc123...",                                │  │
│   │   "created_at": "2024-01-01T00:00:00Z",                             │  │
│   │   "signature": "ed25519:xyz789..."                                  │  │
│   │ }                                                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Key Derivation:                                                          │
│   ───────────────                                                          │
│   Master Seed → BIP-39 Mnemonic → Ed25519 Keypair → Node ID               │
│                                                                             │
│   Storage:                                                                  │
│   ────────                                                                  │
│   Private key encrypted at rest with user passphrase                       │
│   Public key distributed to federation                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Authentication

```rust
struct AgentCredentials {
    agent_id: String,          // Unique agent identifier
    node_id: String,           // Owning node
    capabilities: Vec<String>, // Granted capabilities
    created_at: DateTime,
    expires_at: Option<DateTime>,
    signature: Ed25519Signature, // Signed by node key
}

impl AgentCredentials {
    fn verify(&self, node_public_key: &PublicKey) -> Result<(), AuthError> {
        // Verify signature
        let message = self.serialize_for_signing();
        node_public_key.verify(&message, &self.signature)?;

        // Check expiration
        if let Some(expires) = self.expires_at {
            if Utc::now() > expires {
                return Err(AuthError::Expired);
            }
        }

        Ok(())
    }
}
```

### Session Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SESSION LIFECYCLE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. INITIATION                                                            │
│      ──────────                                                            │
│      User starts BIZRA → Session token generated                           │
│      Token: JWT with node signature                                        │
│                                                                             │
│   2. VALIDATION                                                            │
│      ──────────                                                            │
│      Each request validates session token                                  │
│      Check: signature, expiry, scope                                       │
│                                                                             │
│   3. REFRESH                                                               │
│      ───────                                                               │
│      Sliding window: activity extends session                              │
│      Hard limit: 24 hours maximum                                          │
│                                                                             │
│   4. TERMINATION                                                           │
│      ───────────                                                           │
│      Explicit logout or timeout                                            │
│      Session data securely wiped                                           │
│                                                                             │
│   Session Token Structure:                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Header: { alg: "Ed25519", typ: "JWT" }                              │  │
│   │ Payload: {                                                          │  │
│   │   sub: "node0_ce5af35c",                                           │  │
│   │   iat: 1706745600,                                                 │  │
│   │   exp: 1706832000,                                                 │  │
│   │   scope: ["agent:*", "memory:read", "exec:limited"]                │  │
│   │ }                                                                   │  │
│   │ Signature: ed25519(header.payload, node_private_key)               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Authorization & Access Control

### Capability Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CAPABILITY-BASED ACCESS CONTROL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Agent Capabilities:                                                       │
│   ───────────────────                                                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STRATEGIST                                                          │  │
│   │ ├── strategy:plan                                                   │  │
│   │ ├── strategy:assess                                                 │  │
│   │ ├── memory:read                                                     │  │
│   │ └── memory:write:decisions                                          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ DEVELOPER                                                           │  │
│   │ ├── code:read                                                       │  │
│   │ ├── code:write                                                      │  │
│   │ ├── code:execute:sandbox                                            │  │
│   │ ├── memory:read                                                     │  │
│   │ └── memory:write:code                                               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ EXECUTOR                                                            │  │
│   │ ├── exec:run (REQUIRES guardian:approval)                          │  │
│   │ ├── exec:deploy (REQUIRES guardian:approval + human:confirm)       │  │
│   │ └── memory:read                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ GUARDIAN                                                            │  │
│   │ ├── guardian:veto                                                   │  │
│   │ ├── guardian:approve                                                │  │
│   │ ├── guardian:escalate                                               │  │
│   │ ├── fate:validate                                                   │  │
│   │ ├── audit:read                                                      │  │
│   │ ├── audit:write                                                     │  │
│   │ └── memory:*                                                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Permission Enforcement

```rust
fn check_permission(
    agent: &Agent,
    action: &Action,
    resource: &Resource,
) -> Result<(), AuthError> {
    // 1. Check agent has required capability
    let required_cap = action.required_capability();
    if !agent.capabilities.contains(&required_cap) {
        return Err(AuthError::InsufficientCapabilities);
    }

    // 2. Check resource access
    if !resource.allows_access(agent.id, action.access_type()) {
        return Err(AuthError::ResourceDenied);
    }

    // 3. Check for required approvals
    if action.requires_guardian_approval() {
        let approval = get_guardian_approval(action)?;
        if !approval.is_valid() {
            return Err(AuthError::ApprovalRequired);
        }
    }

    // 4. Check for human confirmation
    if action.requires_human_confirmation() {
        let confirmation = get_human_confirmation(action)?;
        if !confirmation.is_valid() {
            return Err(AuthError::ConfirmationRequired);
        }
    }

    Ok(())
}
```

---

## Data Protection

### Encryption at Rest

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ENCRYPTION AT REST                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Data Categories and Encryption:                                          │
│   ───────────────────────────────                                          │
│                                                                             │
│   ┌────────────────┬──────────────┬──────────────────────────────────┐    │
│   │ Data Type      │ Encryption   │ Key Management                   │    │
│   ├────────────────┼──────────────┼──────────────────────────────────┤    │
│   │ Private Keys   │ AES-256-GCM  │ Derived from passphrase (Argon2) │    │
│   │ Memories       │ AES-256-GCM  │ Node master key                  │    │
│   │ Credentials    │ AES-256-GCM  │ Vault key (HSM if available)     │    │
│   │ Session Data   │ ChaCha20     │ Session-derived key              │    │
│   │ Cache          │ Optional     │ Ephemeral keys                   │    │
│   │ Logs           │ AES-256-GCM  │ Rotating log keys                │    │
│   └────────────────┴──────────────┴──────────────────────────────────┘    │
│                                                                             │
│   Key Hierarchy:                                                           │
│   ─────────────                                                            │
│                                                                             │
│   Master Passphrase                                                        │
│         │                                                                   │
│         ▼ (Argon2id)                                                       │
│   Master Key                                                               │
│         │                                                                   │
│         ├──► Node Identity Key                                             │
│         │                                                                   │
│         ├──► Storage Encryption Key                                        │
│         │         │                                                         │
│         │         ├──► Memory Encryption Key                               │
│         │         ├──► Credential Encryption Key                           │
│         │         └──► Log Encryption Key                                  │
│         │                                                                   │
│         └──► Session Key Derivation                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sensitive Data Handling

```rust
// Sensitive data wrapper with automatic zeroization
#[derive(Zeroize, ZeroizeOnDrop)]
struct SensitiveString(String);

// Vault for secure credential storage
struct Vault {
    key: EncryptionKey,
    storage: EncryptedStorage,
}

impl Vault {
    fn store(&self, name: &str, secret: SensitiveString) -> Result<()> {
        let encrypted = self.key.encrypt(secret.0.as_bytes())?;
        self.storage.put(name, encrypted)?;
        // secret is automatically zeroized when dropped
        Ok(())
    }

    fn retrieve(&self, name: &str) -> Result<SensitiveString> {
        let encrypted = self.storage.get(name)?;
        let decrypted = self.key.decrypt(&encrypted)?;
        Ok(SensitiveString(String::from_utf8(decrypted)?))
    }
}
```

---

## Communication Security

### Transport Layer Security

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRANSPORT SECURITY                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Local Communication (CLI ↔ Backend):                                     │
│   ─────────────────────────────────────                                    │
│   • Unix domain sockets (preferred)                                        │
│   • Localhost only binding                                                 │
│   • Session token authentication                                           │
│                                                                             │
│   LLM Backend Connection:                                                  │
│   ──────────────────────                                                   │
│   • TLS 1.3 required                                                       │
│   • Certificate validation                                                 │
│   • API key in header (not URL)                                           │
│                                                                             │
│   Federation Communication:                                                │
│   ─────────────────────────                                                │
│   • Mutual TLS (mTLS)                                                      │
│   • Ed25519 signed messages                                                │
│   • Replay protection (nonce + timestamp)                                  │
│   • Perfect forward secrecy                                                │
│                                                                             │
│   TLS Configuration:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ min_version: TLS 1.3                                                │  │
│   │ cipher_suites:                                                      │  │
│   │   - TLS_AES_256_GCM_SHA384                                         │  │
│   │   - TLS_CHACHA20_POLY1305_SHA256                                   │  │
│   │ certificate_verification: required                                  │  │
│   │ alpn_protocols: ["h2", "http/1.1"]                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Message Signing

```rust
struct SignedMessage {
    payload: Vec<u8>,
    sender: NodeId,
    timestamp: u64,
    nonce: [u8; 32],
    signature: Ed25519Signature,
}

impl SignedMessage {
    fn create(payload: &[u8], sender: &NodeId, key: &SigningKey) -> Self {
        let timestamp = current_timestamp();
        let nonce = generate_random_nonce();

        let to_sign = [
            payload,
            sender.as_bytes(),
            &timestamp.to_le_bytes(),
            &nonce,
        ].concat();

        let signature = key.sign(&to_sign);

        SignedMessage { payload: payload.to_vec(), sender: sender.clone(), timestamp, nonce, signature }
    }

    fn verify(&self, public_key: &VerifyingKey) -> Result<(), VerifyError> {
        // Check timestamp (prevent replay)
        let age = current_timestamp() - self.timestamp;
        if age > MAX_MESSAGE_AGE {
            return Err(VerifyError::Expired);
        }

        // Verify signature
        let to_verify = [
            &self.payload[..],
            self.sender.as_bytes(),
            &self.timestamp.to_le_bytes(),
            &self.nonce,
        ].concat();

        public_key.verify(&to_verify, &self.signature)?;
        Ok(())
    }
}
```

---

## Inference Security

### Prompt Injection Defense

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROMPT INJECTION DEFENSE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. INPUT SANITIZATION                                                    │
│      ──────────────────                                                    │
│      • Strip control characters                                            │
│      • Escape special tokens                                               │
│      • Normalize Unicode                                                   │
│      • Detect injection patterns                                           │
│                                                                             │
│   2. CONTEXT ISOLATION                                                     │
│      ─────────────────                                                     │
│      • Clear separation between system/user prompts                        │
│      • Structured prompt templates                                         │
│      • No user content in system prompt                                    │
│                                                                             │
│   3. OUTPUT VALIDATION                                                     │
│      ─────────────────                                                     │
│      • FATE gate validation                                                │
│      • Command extraction validation                                       │
│      • Format checking                                                     │
│                                                                             │
│   4. SANDBOXING                                                            │
│      ─────────                                                             │
│      • Code execution in isolated environment                              │
│      • Resource limits                                                     │
│      • Network restrictions                                                │
│                                                                             │
│   Prompt Structure:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ [SYSTEM BOUNDARY - IMMUTABLE]                                       │  │
│   │ You are the Guardian agent...                                       │  │
│   │ [END SYSTEM]                                                        │  │
│   │                                                                     │  │
│   │ [CONTEXT BOUNDARY - VERIFIED]                                       │  │
│   │ Recent context from trusted sources...                              │  │
│   │ [END CONTEXT]                                                       │  │
│   │                                                                     │  │
│   │ [USER BOUNDARY - UNTRUSTED]                                         │  │
│   │ User input (sanitized)...                                           │  │
│   │ [END USER]                                                          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Sandboxing

```rust
struct ExecutionSandbox {
    config: SandboxConfig,
}

struct SandboxConfig {
    // Resource limits
    max_memory_mb: u64,
    max_cpu_time_sec: u64,
    max_file_size_mb: u64,

    // Network restrictions
    allow_network: bool,
    allowed_hosts: Vec<String>,

    // Filesystem restrictions
    read_only_paths: Vec<PathBuf>,
    write_paths: Vec<PathBuf>,
    blocked_paths: Vec<PathBuf>,

    // Process restrictions
    max_processes: u32,
    allow_fork: bool,
}

impl ExecutionSandbox {
    async fn execute(&self, command: &str) -> Result<Output, SandboxError> {
        // 1. Guardian approval check
        if !self.has_guardian_approval(command).await? {
            return Err(SandboxError::ApprovalRequired);
        }

        // 2. Create isolated environment
        let container = self.create_container()?;

        // 3. Apply resource limits
        container.set_memory_limit(self.config.max_memory_mb)?;
        container.set_cpu_limit(self.config.max_cpu_time_sec)?;

        // 4. Apply network restrictions
        if !self.config.allow_network {
            container.disable_network()?;
        }

        // 5. Mount filesystem with restrictions
        for path in &self.config.read_only_paths {
            container.mount_readonly(path)?;
        }

        // 6. Execute with timeout
        let output = container
            .execute(command)
            .timeout(Duration::from_secs(self.config.max_cpu_time_sec))
            .await?;

        // 7. Cleanup
        container.destroy()?;

        Ok(output)
    }
}
```

---

## FATE Gates as Security

### Security Through Ethics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FATE GATES: SECURITY DIMENSION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   HARM GATE (ضرر) — Direct Security Control                               │
│   ──────────────────────────────────────────                               │
│   • Blocks outputs containing security vulnerabilities                     │
│   • Detects credential/secret exposure                                     │
│   • Identifies malicious code patterns                                     │
│   • Prevents social engineering attempts                                   │
│                                                                             │
│   Harm Categories (Security Focus):                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ • security_vulnerability: Detects exploitable code                 │  │
│   │ • credential_exposure: Catches secrets in output                   │  │
│   │ • privilege_escalation: Identifies permission bypasses            │  │
│   │ • injection_attempt: Catches SQL, XSS, command injection          │  │
│   │ • malicious_instruction: Blocks harmful how-tos                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   CONFIDENCE GATE (ثقة) — Integrity Control                               │
│   ─────────────────────────────────────────                               │
│   • Validates source reliability                                          │
│   • Ensures response consistency                                          │
│   • Requires verification for critical info                               │
│                                                                             │
│   ADL GATE (عدل) — Access Equity Control                                 │
│   ──────────────────────────────────────                                  │
│   • Ensures fair resource distribution                                    │
│   • Prevents information hoarding                                         │
│   • Detects access pattern anomalies                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Audit & Monitoring

### Audit Log

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUDIT LOGGING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Logged Events:                                                           │
│   ─────────────                                                            │
│   • Authentication attempts (success/failure)                              │
│   • Agent actions (with context)                                           │
│   • FATE gate decisions                                                    │
│   • Guardian approvals/vetoes                                              │
│   • Configuration changes                                                  │
│   • Execution commands                                                     │
│   • Federation events                                                      │
│                                                                             │
│   Log Entry Structure:                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ {                                                                    │  │
│   │   "id": "evt_abc123",                                               │  │
│   │   "timestamp": "2026-02-05T14:32:15.123Z",                          │  │
│   │   "type": "agent_action",                                           │  │
│   │   "severity": "info",                                               │  │
│   │   "agent": "developer",                                             │  │
│   │   "action": "code_generate",                                        │  │
│   │   "resource": "src/auth.rs",                                        │  │
│   │   "outcome": "success",                                             │  │
│   │   "fate_scores": {"ihsan": 0.97, "harm": 0.05},                    │  │
│   │   "context": {...},                                                 │  │
│   │   "signature": "ed25519:..."                                        │  │
│   │ }                                                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Tamper Evidence:                                                         │
│   ───────────────                                                          │
│   • Each entry signed with node key                                        │
│   • Chained hashes (merkle tree)                                          │
│   • Append-only storage                                                    │
│   • Periodic external backup                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Monitoring Alerts

```yaml
monitoring:
  alerts:
    # Security alerts
    auth_failure_threshold:
      type: "rate"
      metric: "auth_failures"
      threshold: 5
      window: "5m"
      action: "alert_and_lockout"

    fate_harm_spike:
      type: "rate"
      metric: "fate_harm_failures"
      threshold: 3
      window: "1h"
      action: "alert_guardian"

    unusual_execution:
      type: "anomaly"
      metric: "exec_commands"
      baseline: "7d"
      deviation: 2.0
      action: "alert_and_log"

    federation_anomaly:
      type: "anomaly"
      metric: "federation_messages"
      baseline: "24h"
      deviation: 3.0
      action: "alert_and_isolate"
```

---

## Incident Response

### Response Procedures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INCIDENT RESPONSE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. DETECTION                                                              │
│      ─────────                                                              │
│      • Automated monitoring alerts                                          │
│      • Guardian anomaly detection                                           │
│      • User reports                                                         │
│                                                                             │
│   2. CONTAINMENT                                                            │
│      ───────────                                                            │
│      • Guardian auto-pause on critical alerts                               │
│      • Session isolation                                                    │
│      • Federation disconnect (if needed)                                    │
│                                                                             │
│   3. ANALYSIS                                                               │
│      ────────                                                               │
│      • Review audit logs                                                    │
│      • Trace attack path                                                    │
│      • Assess impact                                                        │
│                                                                             │
│   4. ERADICATION                                                            │
│      ───────────                                                            │
│      • Remove malicious content                                             │
│      • Patch vulnerabilities                                                │
│      • Update FATE thresholds                                               │
│                                                                             │
│   5. RECOVERY                                                               │
│      ────────                                                               │
│      • Restore from backup (if needed)                                      │
│      • Gradual service restoration                                          │
│      • Enhanced monitoring                                                  │
│                                                                             │
│   6. LESSONS LEARNED                                                        │
│      ───────────────                                                        │
│      • Document incident                                                    │
│      • Update procedures                                                    │
│      • Train patterns (for future detection)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Security is not a feature — it's a foundation.** 🛡
