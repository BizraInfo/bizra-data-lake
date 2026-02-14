# Phase 01: Identity Layer

Last updated: 2026-02-14
Standing on: Bernstein (Ed25519) · Kocher (Timing Safety) · Lampson (Capability Model)

---

## Purpose

Layer 1 establishes **who the node is**. Every subsequent layer depends on a verified, hardware-attested identity. Without Layer 1, no signature is valid, no ledger entry is authentic, and no gate can trust its inputs.

The identity layer answers three questions:
1. **Who is this node?** — Ed25519 public key, derived NodeId
2. **Is the hardware authentic?** — 3-tier hardware attestation (ROOT/MUTABLE/CONTEXTUAL)
3. **What can this node do?** — Capability cards with expiration and revocation

---

## Data Structures

### NodeIdentity

```pseudocode
STRUCT NodeIdentity:
    node_id:        Hash           # BLAKE3(public_key_bytes), 32 bytes
    public_key:     Ed25519PubKey  # 32 bytes
    private_key:    Ed25519SecKey  # 64 bytes, NEVER serializable
    created_at:     Timestamp      # Nanosecond precision
    genesis_hash:   Hash           # Links to genesis ceremony output

    INVARIANT: private_key is wrapped in PrivateKeyWrapper
               PrivateKeyWrapper.__repr__() returns "[REDACTED]"
               PrivateKeyWrapper.__str__() returns "[REDACTED]"
               # Prevents accidental key leakage in logs/debug output

    METHOD sign(message: bytes) -> Signature:
        RETURN ed25519_sign(self.private_key, message)

    METHOD verify(message: bytes, signature: Signature) -> bool:
        RETURN ed25519_verify(self.public_key, message, signature)

    METHOD domain_separated_digest(data: bytes) -> Hash:
        # Standing on: Bernstein (domain separation prevents cross-protocol attacks)
        prefixed = b"bizra-pci-v1:" + data
        RETURN blake3_hash(prefixed)
```

**Source:** `core/pci/crypto.py:PrivateKeyWrapper` (500 lines), `bizra-omega/bizra-core/src/identity.rs` (170 lines)

### GenesisState

```pseudocode
STRUCT GenesisState:
    node_identity:      NodeIdentity
    agent_identities:   Map<AgentRole, AgentIdentity>  # PAT + SAT agents
    hardware_quote:     HardwareAttestation
    birth_certificate:  SignedEnvelope                  # Self-signed genesis record
    merkle_root:        Hash                           # Root of genesis ceremony DAG
    created_at:         Timestamp

    INVARIANT: birth_certificate.verify(node_identity.public_key) == true
    INVARIANT: merkle_root == compute_merkle_root(all_ceremony_entries)
```

**Source:** `core/sovereign/genesis_identity.py:GenesisState` (277 lines), `bizra-omega/bizra-core/src/genesis.rs` (1,253 lines)

### AgentIdentity

```pseudocode
STRUCT AgentIdentity:
    agent_id:       Hash
    agent_role:     AgentRole        # PAT_STRATEGIST | PAT_RESEARCHER | SAT_VALIDATOR | ...
    parent_node:    NodeId           # Derived from parent node's public key
    capabilities:   List<Capability>
    public_key:     Ed25519PubKey    # Agent-specific keypair (subordinate to node)

ENUM AgentRole:
    # PAT (Primary Agent Types) — User-facing
    PAT_STRATEGIST    # Goal decomposition and planning
    PAT_RESEARCHER    # Information gathering and synthesis
    PAT_DEVELOPER     # Code generation and system modification
    PAT_ANALYST       # Data analysis and pattern detection
    PAT_REVIEWER      # Quality assurance and validation
    PAT_EXECUTOR      # Task execution and deployment
    PAT_GUARDIAN      # Constitutional enforcement and security
    # SAT (Secondary Agent Types) — System-facing
    SAT_VALIDATOR     # Consensus participation, proof verification
    SAT_ORACLE        # External data sourcing with provenance
    SAT_MEDIATOR      # Conflict resolution between agents
    SAT_ARCHIVIST     # Knowledge persistence and retrieval
    SAT_SENTINEL      # Threat detection and response
```

**Source:** `core/sovereign/genesis_identity.py:AgentIdentity`

### HardwareAttestation

```pseudocode
STRUCT HardwareAttestation:
    root_tier:        RootAttestation       # CPU + GPU + Platform — HARD FAIL on mismatch
    mutable_tier:     MutableAttestation    # RAM, Storage, MAC — WARN + require re-attestation
    contextual_tier:  ContextualAttestation # BIOS, OS, WSL — LOG ONLY

STRUCT RootAttestation:
    cpu_id:          String    # e.g., "Intel Core i9-14900HX"
    gpu_id:          String    # e.g., "NVIDIA RTX 4090"
    platform_id:     String    # e.g., "MSI Titan GT77 HX"
    fingerprint:     Hash      # BLAKE3(cpu_id || gpu_id || platform_id)

    INVARIANT: On boot, recompute fingerprint and compare.
               If mismatch: KERNEL PANIC — node identity invalid.
```

**Source:** `bizra-omega/bizra-core/src/genesis.rs` (hardware attestation section)

### CapabilityCard

```pseudocode
STRUCT CapabilityCard:
    card_id:         Hash
    subject:         NodeId | AgentId | ModelId
    capabilities:    List<Capability>
    issuer:          NodeId
    issued_at:       Timestamp
    expires_at:      Timestamp         # Default: 90 days
    revoked:         bool
    signature:       Ed25519Signature  # Signed by issuer

ENUM Capability:
    INFERENCE_LOCAL      # Execute local model inference
    INFERENCE_FEDERATED  # Participate in federated inference pools
    STORAGE_READ         # Read from data lake
    STORAGE_WRITE        # Write to data lake (requires PoI)
    FEDERATION_GOSSIP    # Participate in gossip protocol
    FEDERATION_VOTE      # Vote in consensus rounds
    TOKEN_MINT           # Mint tokens (requires SAT authority)
    TOKEN_TRANSFER       # Transfer tokens between accounts
    GATE_VERIFY          # Execute FATE gate verification
    QUEST_CREATE         # Create sovereign quests

    INVARIANT: expires_at - issued_at <= 90 days
    INVARIANT: revoked cards fail verification immediately
```

**Source:** `native/fate-binding/src/capability_card.rs` (342 lines)

---

## Procedures

### Genesis Ceremony

The genesis ceremony executes exactly once per node lifetime. It creates the immutable birth certificate.

```pseudocode
PROCEDURE genesis_ceremony(hardware: HardwareInfo) -> GenesisState:
    # Step 1: Generate master keypair
    keypair = ed25519_generate_keypair()
    node_id = blake3_hash(keypair.public_key)

    # Step 2: Attest hardware (3-tier)
    root = RootAttestation(
        cpu_id   = hardware.cpu,
        gpu_id   = hardware.gpu,
        platform = hardware.platform,
        fingerprint = blake3_hash(cpu_id || gpu_id || platform)
    )
    attestation = HardwareAttestation(root, mutable, contextual)

    # Step 3: Mint agent identities (PAT + SAT)
    agents = {}
    FOR EACH role IN AgentRole:
        agent_kp = ed25519_generate_keypair()
        agents[role] = AgentIdentity(
            agent_id   = blake3_hash(agent_kp.public_key),
            agent_role = role,
            parent_node = node_id,
            public_key = agent_kp.public_key,
        )

    # Step 4: Build Merkle DAG of ceremony entries
    entries = [keypair.public_key, attestation, agents, timestamp]
    merkle_root = compute_merkle_root(entries)

    # Step 5: Self-sign birth certificate
    certificate = SignedEnvelope(
        payload   = canonical_json(entries),
        signature = keypair.sign(merkle_root),
        signer    = keypair.public_key,
    )

    # Step 6: Persist to sovereign_state/
    state = GenesisState(
        node_identity     = NodeIdentity(node_id, keypair, ...),
        agent_identities  = agents,
        hardware_quote    = attestation,
        birth_certificate = certificate,
        merkle_root       = merkle_root,
    )
    write_to_disk("sovereign_state/genesis.json", state)

    RETURN state

    # POST-CONDITION: genesis.json exists and is self-consistent
    # POST-CONDITION: Re-running genesis_ceremony FAILS (once-only guard)
```

**Source:** `bizra-omega/bizra-core/src/genesis.rs` (1,253 lines)

### Identity Verification on Boot

```pseudocode
PROCEDURE verify_identity_on_boot() -> Result<NodeIdentity, KernelPanic>:
    # Step 1: Load genesis state from disk
    state = load_genesis_state("sovereign_state/genesis.json")
    IF state IS None:
        RETURN Error(KernelPanic("No genesis state found. Run genesis ceremony."))

    # Step 2: Verify birth certificate signature
    valid = ed25519_verify(
        state.node_identity.public_key,
        state.merkle_root,
        state.birth_certificate.signature,
    )
    IF NOT valid:
        RETURN Error(KernelPanic("Birth certificate signature invalid. Tampering detected."))

    # Step 3: Re-attest hardware (ROOT tier)
    current_hw = detect_hardware()
    current_fingerprint = blake3_hash(
        current_hw.cpu || current_hw.gpu || current_hw.platform
    )
    IF current_fingerprint != state.hardware_quote.root_tier.fingerprint:
        RETURN Error(KernelPanic("ROOT hardware mismatch. Node identity invalid."))

    # Step 4: Check MUTABLE tier (warn only)
    IF mutable_tier_changed(state.hardware_quote, current_hw):
        LOG_WARNING("MUTABLE hardware changed. Re-attestation recommended.")

    # Step 5: Identity verified
    LOG_INFO("Identity verified: node_id={}", state.node_identity.node_id)
    RETURN Ok(state.node_identity)
```

**Source:** `core/sovereign/genesis_identity.py:load_and_validate_genesis()`

### Cryptographic Primitives

```pseudocode
# Standing on: Kocher (1996) — timing-safe comparison
PROCEDURE timing_safe_compare(a: bytes, b: bytes) -> bool:
    # MUST use hmac.compare_digest or equivalent
    # MUST iterate ALL bytes regardless of mismatch position
    # NEVER short-circuit on first difference
    RETURN hmac_compare_digest(a, b)

# Standing on: Bernstein (2012) — Ed25519 signatures
PROCEDURE sign_message(private_key: Ed25519SecKey, message: bytes) -> Signature:
    RETURN ed25519_sign(private_key, message)

PROCEDURE verify_signature(public_key: Ed25519PubKey, message: bytes, sig: Signature) -> bool:
    TRY:
        ed25519_verify(public_key, message, sig)
        RETURN true
    CATCH InvalidSignature:
        RETURN false

# Standing on: RFC 8785 — deterministic JSON serialization
PROCEDURE canonicalize_json(data: Any) -> bytes:
    # Sorted keys, no trailing whitespace, minimal separators
    # Ensures identical inputs produce identical hashes regardless of key order
    RETURN json_serialize(data, sort_keys=true, separators=(',', ':'))
```

**Source:** `core/pci/crypto.py` (500 lines)

### Post-Quantum Readiness

```pseudocode
# Standing on: NIST PQC standardization (2024)
# Dilithium-5 is available but NOT the default signing algorithm.
# Transition plan: Ed25519 → Hybrid(Ed25519 + Dilithium-5) → Dilithium-5

STRUCT HybridSignature:
    ed25519_sig:    Ed25519Signature     # Current production
    dilithium_sig:  DilithiumSignature   # Future production (optional)
    algorithm:      SignatureAlgorithm   # ED25519 | HYBRID | DILITHIUM5

    METHOD verify(public_keys, message) -> bool:
        MATCH self.algorithm:
            ED25519:   RETURN verify_ed25519(...)
            HYBRID:    RETURN verify_ed25519(...) AND verify_dilithium(...)
            DILITHIUM5: RETURN verify_dilithium(...)
```

**Source:** `native/fate-binding/src/dilithium.rs` (206 lines)

---

## TDD Anchors

| Test | File | Validates |
|------|------|-----------|
| `test_keypair_generation` | `tests/core/sovereign/test_keypair_security.py` | Ed25519 keypair creation + PrivateKeyWrapper redaction |
| `test_sign_verify_roundtrip` | `tests/core/pci/test_crypto.py` | Sign → Verify cycle produces consistent results |
| `test_timing_safe_compare` | `tests/core/pci/test_timing_safe.py` | Constant-time comparison (27 tests, <2x variance) |
| `test_genesis_identity_creation` | `tests/core/sovereign/test_genesis_identity.py` | Genesis ceremony creates valid state |
| `test_genesis_once_only` | `tests/core/sovereign/test_genesis_identity.py` | Re-running genesis fails |
| `test_capability_card_expiry` | `tests/core/sovereign/test_capability_card.py` | Expired cards fail verification |
| `test_hardware_root_mismatch` | `tests/core/sovereign/test_fate_validation.py` | ROOT tier mismatch triggers kernel panic |
| `test_rfc8785_canonicalization` | `tests/core/pci/test_rfc8785_canonicalization.py` | JSON canonicalization is deterministic |
| `test_cross_language_crypto` | `tests/integration/test_cross_language_crypto.py` | Python and Rust produce identical signatures |

---

## Security Considerations

1. **Key Storage:** Private keys are stored encrypted at rest in `sovereign_state/`. The `PrivateKeyWrapper` class prevents accidental serialization. `__repr__` and `__str__` return `[REDACTED]`.

2. **No Key Export:** There is no API endpoint or CLI command to export private keys. Key material is memory-only during runtime.

3. **Timing Attacks:** All comparison operations use `hmac.compare_digest()` (Python) or constant-time comparison (Rust). Statistical timing tests enforce <2x variance tolerance.

4. **Genesis Replay:** The genesis ceremony checks for existing `sovereign_state/genesis.json` before proceeding. If the file exists, the ceremony aborts. This prevents identity replacement attacks.

5. **Hardware Binding:** ROOT tier (CPU + GPU + Platform) is a hard gate. If the hardware changes at the ROOT level, the node must re-genesis with a new identity. This prevents identity theft via disk cloning.

---

*Source of truth: `core/pci/crypto.py`, `core/sovereign/genesis_identity.py`, `bizra-omega/bizra-core/src/identity.rs`, `bizra-omega/bizra-core/src/genesis.rs`, `native/fate-binding/src/capability_card.rs`*
