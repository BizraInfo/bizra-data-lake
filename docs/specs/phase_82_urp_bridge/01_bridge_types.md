# 01 — URP Bridge Types: PyO3 Wrappers

> Module: `bizra-omega/bizra-python/src/urp_bridge.rs` (NEW)
> Language: Rust (PyO3) + Python wrapper
> Constitutional Anchor: Law 6 (Sovereign Economics)

## 1. Core Principle

Every Rust struct that crosses the Python boundary gets a `Py*` wrapper.
Wrappers are thin — they hold an `Arc` or owned copy of the Rust type,
expose getters as `#[getter]`, and provide `to_dict()` for Python interop.

## 2. PyURPPledge — Pledge Wrapper

```rust
STRUCT PyURPPledge:
    // Wraps the Python-side URPPledge for Rust validation
    node_id:          String
    ram_gb:           u32
    vram_gb:          u32
    storage_gb:       u32
    pledge_hash:      String
    pledged_at:       String      // ISO 8601
    signed:           bool
    signature:        String      // Ed25519 hex
    signer_public_key: String     // Ed25519 verifying key hex
    payload_digest:   String      // BLAKE2b/SHA-256 hex
    enforcement_mode: String
    status:           String      // "deferred" | "enforced" | "rejected"

METHODS:
    #[new]
    fn new(node_id, ram_gb, vram_gb, storage_gb, ...) -> Self

    #[getter] fn node_id(&self) -> String
    #[getter] fn ram_gb(&self) -> u32
    #[getter] fn signed(&self) -> bool
    #[getter] fn status(&self) -> String

    fn to_dict(&self) -> PyDict:
        // Return all fields as Python dict
        RETURN {
            "node_id": self.node_id,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "storage_gb": self.storage_gb,
            "pledge_hash": self.pledge_hash,
            "pledged_at": self.pledged_at,
            "signed": self.signed,
            "signature": self.signature,
            "signer_public_key": self.signer_public_key,
            "payload_digest": self.payload_digest,
            "enforcement_mode": self.enforcement_mode,
            "status": self.status,
        }

    fn verify_signature(&self) -> PyResult<bool>:
        // Verify Ed25519 signature in Rust (authoritative)
        IF NOT self.signed:
            RETURN False

        TRY:
            verifying_key = parse_ed25519_verifying_key(self.signer_public_key)
            signature = parse_ed25519_signature(self.signature)
            payload = canonical_pledge_payload(
                self.node_id, self.ram_gb, self.vram_gb,
                self.storage_gb, self.pledged_at
            )
            digest = domain_separated_digest(payload)
            RETURN verifying_key.verify(digest.as_bytes(), &signature).is_ok()
        CATCH:
            RETURN False

    @classmethod
    fn from_dict(cls, data: PyDict) -> PyResult<Self>:
        // Construct from Python dict (e.g., URPPledge.to_dict())
        RETURN Self::new(
            data["node_id"], data["ram_gb"], data["vram_gb"],
            data["storage_gb"], ...
        )
```

## 3. PyPoolNode — Node Registration Result

```rust
STRUCT PyPoolNode:
    // Read-only view of a registered node in the pool
    id:           String
    class:        String      // "micro" | "light" | "standard" | "contributor" | "anchor"
    status:       String      // "active" | "suspended" | "offline"
    resources:    PyNodeResources
    token_balance: u64
    ihsan_score:  f64
    registered_at: String     // ISO 8601

METHODS:
    #[getter] fn id(&self) -> String
    #[getter] fn class(&self) -> String
    #[getter] fn token_balance(&self) -> u64
    #[getter] fn ihsan_score(&self) -> f64

    fn to_dict(&self) -> PyDict:
        RETURN {
            "id": self.id,
            "class": self.class,
            "status": self.status,
            "resources": self.resources.to_dict(),
            "token_balance": self.token_balance,
            "ihsan_score": self.ihsan_score,
            "registered_at": self.registered_at,
        }

    fn passes_ihsan(&self) -> bool:
        RETURN self.ihsan_score >= IHSAN_THRESHOLD  // 0.95
```

## 4. PyNodeResources — Hardware Resources

```rust
STRUCT PyNodeResources:
    cpu_cores:    u32
    ram_gb:       u32
    vram_gb:      u32
    storage_gb:   u64
    network_mbps: f64

METHODS:
    #[new]
    fn new(cpu_cores, ram_gb, vram_gb, storage_gb, network_mbps) -> Self

    fn to_dict(&self) -> PyDict:
        RETURN {
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "storage_gb": self.storage_gb,
            "network_mbps": self.network_mbps,
        }

    fn compute_hash(&self) -> String:
        // BLAKE3 hash of canonical resource representation
        hasher = blake3::Hasher::new()
        hasher.update(canonical_bytes(self))
        RETURN hasher.finalize().to_hex()
```

## 5. PyContributionReceipt — Proof-of-Impact Receipt

```rust
STRUCT PyContributionReceipt:
    contribution_id: String    // UUID
    node_id:         String
    resource_type:   String    // "cpu" | "ram" | "gpu" | "storage" | "witness"
    amount:          f64       // Units contributed
    duration_ms:     u64       // Duration of contribution
    tokens_earned:   u64       // SEED tokens minted
    ihsan_score:     f64       // Quality score of contribution
    receipt_hash:    String    // BLAKE3 chain hash
    timestamp:       String    // ISO 8601

METHODS:
    fn to_dict(&self) -> PyDict
    fn verify_hash(&self) -> bool
```

## 6. PyPoolStats — Pool-Level Statistics

```rust
STRUCT PyPoolStats:
    total_nodes:          u64
    active_nodes:         u64
    total_compute_units:  u64
    total_tokens_minted:  u64
    total_services:       u64
    gini_coefficient:     f64
    adl_compliant:        bool
    zakat_distributed:    u64
    pool_health:          f64    // 0.0-1.0

METHODS:
    fn to_dict(&self) -> PyDict
```

## 7. PyO3 Module Registration

```rust
FUNCTION register_urp_types(m: &PyModule) -> PyResult<()>:
    m.add_class::<PyURPPledge>()?
    m.add_class::<PyPoolNode>()?
    m.add_class::<PyNodeResources>()?
    m.add_class::<PyContributionReceipt>()?
    m.add_class::<PyPoolStats>()?
    Ok(())

// In lib.rs #[pymodule] fn bizra(m):
//   ...existing registrations...
//   urp_bridge::register_urp_types(m)?;
```

## 8. Python Re-export Wrapper

```python
# In bizra-python/python/bizra/__init__.py
FILE: bizra/__init__.py (EXTEND)

TRY:
    from .bizra import (
        PyURPPledge,
        PyPoolNode,
        PyNodeResources,
        PyContributionReceipt,
        PyPoolStats,
    )
EXCEPT ImportError:
    # PyO3 not built — define stubs
    PyURPPledge = None
    PyPoolNode = None
    PyNodeResources = None
    PyContributionReceipt = None
    PyPoolStats = None
```

## 9. Conversion Layer: Python URPPledge ↔ PyURPPledge

```python
FILE: core/genesis/urp.py (EXTEND)

FUNCTION pledge_to_rust(pledge: URPPledge) -> Optional["PyURPPledge"]:
    """Convert Python URPPledge to Rust PyURPPledge for validation."""
    TRY:
        from bizra import PyURPPledge
        IF PyURPPledge IS None:
            RETURN None
        RETURN PyURPPledge.from_dict(pledge.to_dict())
    EXCEPT (ImportError, RuntimeError):
        RETURN None   # Graceful degradation — Level 0

FUNCTION rust_verify_pledge(pledge: URPPledge) -> Optional[bool]:
    """Verify pledge signature using Rust (authoritative).

    Returns None if Rust bridge unavailable (Level 0 degradation).
    Returns True/False if Rust verification succeeded.
    """
    rust_pledge = pledge_to_rust(pledge)
    IF rust_pledge IS None:
        RETURN None
    RETURN rust_pledge.verify_signature()
```

## 10. Type Mapping Table

| Python Type | Rust Type | PyO3 Wrapper | Direction |
|-------------|-----------|-------------|-----------|
| `URPPledge` | — | `PyURPPledge` | Python→Rust |
| — | `PoolNode` | `PyPoolNode` | Rust→Python |
| `dict` | `NodeResources` | `PyNodeResources` | Bidirectional |
| — | `ResourceContribution` | `PyContributionReceipt` | Rust→Python |
| — | `PoolStats` | `PyPoolStats` | Rust→Python |
| `str` (hex) | `VerifyingKey` | Internal | Python→Rust |
| `str` (hex) | `Signature` | Internal | Python→Rust |

## TDD Anchors

```
TEST pyurp_pledge_roundtrip:
    pledge = PyURPPledge(node_id="abc123", ram_gb=16, vram_gb=6, ...)
    d = pledge.to_dict()
    restored = PyURPPledge.from_dict(d)
    ASSERT restored.node_id == pledge.node_id
    ASSERT restored.ram_gb == pledge.ram_gb

TEST pyurp_pledge_verify_valid_signature:
    # Create pledge with real Ed25519 signing
    pledge = create_signed_test_pledge()
    rust_pledge = PyURPPledge.from_dict(pledge.to_dict())
    ASSERT rust_pledge.verify_signature() == True

TEST pyurp_pledge_verify_tampered_fails:
    pledge = create_signed_test_pledge()
    d = pledge.to_dict()
    d["ram_gb"] = 999  # Tamper
    rust_pledge = PyURPPledge.from_dict(d)
    ASSERT rust_pledge.verify_signature() == False

TEST pool_node_passes_ihsan:
    node = PyPoolNode(ihsan_score=0.96, ...)
    ASSERT node.passes_ihsan() == True
    node2 = PyPoolNode(ihsan_score=0.80, ...)
    ASSERT node2.passes_ihsan() == False

TEST node_resources_hash_deterministic:
    r1 = PyNodeResources(cpu_cores=8, ram_gb=16, vram_gb=6, ...)
    r2 = PyNodeResources(cpu_cores=8, ram_gb=16, vram_gb=6, ...)
    ASSERT r1.compute_hash() == r2.compute_hash()

TEST graceful_degradation_no_rust:
    # Mock PyO3 import failure
    mock_import_error("bizra")
    result = rust_verify_pledge(some_pledge)
    ASSERT result IS None  # Degraded, not crashed
```
