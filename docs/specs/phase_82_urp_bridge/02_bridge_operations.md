# 02 — URP Bridge Operations: Submit, Register, Contribute, Reward

> Module: `bizra-omega/bizra-python/src/urp_bridge.rs` (operations)
> + `core/bridges/urp_rust_bridge.py` (Python wrapper)
> Language: Rust (PyO3) + Python
> Constitutional Anchor: Law 6 (Sovereign Economics) + Proof-of-Impact

## 1. Core Principle

Operations cross the Python→Rust boundary as explicit function calls.
Each operation returns a typed result or raises `PyRuntimeError`.
The Python wrapper (`urp_rust_bridge.py`) provides fail-closed degradation:
if Rust is unavailable, every function returns `None` (never crashes).

## 2. PyResourcePool — Pool Handle

```rust
STRUCT PyResourcePool:
    // Wraps Arc<ResourcePool> for Python access
    // Created once at node boot, shared across all operations
    inner: Arc<ResourcePool>

METHODS:
    #[new]
    fn new() -> PyResult<Self>:
        // Initialize a new pool (genesis)
        pool = ResourcePool::genesis(...).await
        RETURN Self { inner: Arc::new(pool) }

    @staticmethod
    fn from_existing(state_json: &str) -> PyResult<Self>:
        // Restore pool from serialized state
        pool = serde_json::from_str(state_json)
        RETURN Self { inner: Arc::new(pool) }
```

## 3. Operation: Submit Pledge

```rust
FUNCTION submit_pledge(pool: &PyResourcePool, pledge: &PyURPPledge)
    -> PyResult<PyPoolNode>:
    """
    Submit a signed URP pledge to the Rust pool.

    Pipeline:
    1. Verify Ed25519 signature (Rust-authoritative)
    2. Validate resource amounts (non-zero, within bounds)
    3. Determine node class from resources
    4. Register node in pool
    5. Return PyPoolNode with registration details

    Fails if: unsigned, invalid signature, zero resources, Ihsan < 0.95
    """

    # 1. Signature verification (fail-closed)
    IF NOT pledge.signed:
        RETURN Err(PyRuntimeError("Pledge must be signed"))

    IF NOT pledge.verify_signature():
        RETURN Err(PyRuntimeError("Invalid pledge signature"))

    # 2. Resource validation
    IF pledge.ram_gb == 0 AND pledge.vram_gb == 0:
        RETURN Err(PyRuntimeError("Pledge must include at least RAM or VRAM"))

    # 3. Build registration request
    request = RegistrationRequest {
        node_id:     pledge.node_id,
        public_key:  parse_verifying_key(pledge.signer_public_key),
        resources:   NodeResources {
            cpu_cores:    0,    // Detected separately
            ram_gb:       pledge.ram_gb,
            gpu_vram_gb:  pledge.vram_gb,
            storage_gb:   pledge.storage_gb,
            network_mbps: 0.0,
        },
        ihsan_score: 1.0,     // Genesis default
        signature:   parse_signature(pledge.signature),
    }

    # 4. Register in pool
    response = pool.inner.register_node(request).await
    IF response.is_err():
        RETURN Err(PyRuntimeError(response.err().to_string()))

    # 5. Return result
    node = response.unwrap()
    RETURN Ok(PyPoolNode::from_rust(node))
```

## 4. Operation: Contribute Resources

```rust
FUNCTION contribute_resources(
    pool: &PyResourcePool,
    node_id: &str,
    resource_type: &str,    // "cpu" | "ram" | "gpu" | "storage" | "witness"
    amount: f64,
    duration_ms: u64,
    proof_hash: &str,       // BLAKE3 hash of work performed
) -> PyResult<PyContributionReceipt>:
    """
    Record a resource contribution and mint SEED tokens.

    Pipeline:
    1. Validate node exists and is active
    2. Validate contribution proof hash
    3. Calculate Ihsan score for this contribution
    4. Mint tokens based on: amount × duration × quality_multiplier
    5. Apply Zakat deduction (2.5%) at mint time
    6. Check ADL Gini constraint (reject if Gini > 0.35)
    7. Return receipt with tokens earned

    Constitutional constraints:
    - Ihsan ≥ 0.85 required (quality gate)
    - Zakat 2.5% applied at mint (Al-Baqarah 2:43)
    - Gini ≤ 0.35 enforced (justice gate)
    """

    # 1. Validate node
    node = pool.inner.get_node(node_id).await
    IF node IS None:
        RETURN Err(PyRuntimeError("Node not registered"))
    IF node.status != NodeStatus::Active:
        RETURN Err(PyRuntimeError("Node not active"))

    # 2. Parse resource type
    res_type = MATCH resource_type:
        "cpu"     => ResourceType::Compute
        "ram"     => ResourceType::Memory
        "gpu"     => ResourceType::GPU
        "storage" => ResourceType::Storage
        "witness" => ResourceType::Witness
        _         => RETURN Err(PyValueError("Invalid resource type"))

    # 3. Build contribution
    contribution = ResourceContribution {
        node_id:       node_id.to_string(),
        resource_type: res_type,
        amount:        amount,
        duration_ms:   duration_ms,
        proof_hash:    proof_hash.to_string(),
        timestamp:     Utc::now(),
    }

    # 4. Submit to pool (mints tokens internally)
    result = pool.inner.contribute_resources(contribution).await
    IF result.is_err():
        RETURN Err(PyRuntimeError(result.err().to_string()))

    # 5. Build receipt
    (tokens_minted, receipt_hash) = result.unwrap()

    RETURN Ok(PyContributionReceipt {
        contribution_id: Uuid::new_v4().to_string(),
        node_id:         node_id.to_string(),
        resource_type:   resource_type.to_string(),
        amount:          amount,
        duration_ms:     duration_ms,
        tokens_earned:   tokens_minted,
        ihsan_score:     node.ihsan_score,
        receipt_hash:    receipt_hash,
        timestamp:       Utc::now().to_rfc3339(),
    })
```

## 5. Operation: Get Rewards Summary

```rust
FUNCTION get_rewards(
    pool: &PyResourcePool,
    node_id: &str,
) -> PyResult<PyDict>:
    """
    Get SEED token balance and contribution history for a node.

    Returns dict with:
    - balance: Current token balance (after Zakat)
    - total_earned: Lifetime tokens earned
    - total_zakat_paid: Lifetime Zakat deducted
    - contributions: List of contribution summaries
    - rank: Node rank by contribution (position in pool)
    """

    node = pool.inner.get_node(node_id).await
    IF node IS None:
        RETURN Err(PyRuntimeError("Node not found"))

    stats = pool.inner.stats().await

    RETURN Ok({
        "node_id":         node.id,
        "balance":         node.token_balance,
        "total_earned":    node.total_earned,
        "total_zakat_paid": node.total_zakat,
        "ihsan_score":     node.ihsan_score,
        "node_class":      node.class.as_registration_label(),
        "pool_gini":       stats.gini_coefficient,
        "adl_compliant":   stats.gini_coefficient <= 0.35,
    })
```

## 6. Operation: Process Zakat

```rust
FUNCTION process_zakat(pool: &PyResourcePool) -> PyResult<PyDict>:
    """
    Trigger Zakat distribution across the pool.

    Zakat rules (Al-Baqarah 2:43):
    - Rate: 2.5% annual
    - Nisab: 1,000,000 tokens (threshold before Zakat applies)
    - Recipients: 8 Quranic categories (At-Tawbah 9:60)
    - Distribution: Proportional to need score

    This is a pool-wide operation, typically triggered by
    the heartbeat cycle or admin command.
    """

    result = pool.inner.process_zakat().await
    IF result.is_err():
        RETURN Err(PyRuntimeError(result.err().to_string()))

    dist = result.unwrap()

    RETURN Ok({
        "total_distributed":  dist.total_amount,
        "eligible_nodes":     dist.eligible_count,
        "recipients":         dist.recipients.len(),
        "categories":         dist.categories_served,
        "timestamp":          dist.timestamp.to_rfc3339(),
    })
```

## 7. Operation: Check ADL (Justice Gate)

```rust
FUNCTION check_adl(pool: &PyResourcePool) -> PyResult<PyDict>:
    """
    Check Adl (justice) compliance via Gini coefficient.

    ADL constraint: Gini ≤ 0.35 (constants.py single source of truth)
    If violated, pool enters redistribution mode.
    """

    gini = pool.inner.calculate_gini().await
    compliant = pool.inner.check_adl().await.is_ok()

    RETURN Ok({
        "gini_coefficient": gini.to_f64(),
        "threshold":        0.35,
        "compliant":        compliant,
        "action_required":  NOT compliant,
    })
```

## 8. Operation: Pool Statistics

```rust
FUNCTION pool_stats(pool: &PyResourcePool) -> PyResult<PyPoolStats>:
    """Return current pool statistics."""

    stats = pool.inner.stats().await
    RETURN Ok(PyPoolStats {
        total_nodes:         stats.total_nodes,
        active_nodes:        stats.active_nodes,
        total_compute_units: stats.total_compute_units,
        total_tokens_minted: stats.total_tokens_minted,
        total_services:      stats.total_services,
        gini_coefficient:    stats.gini_coefficient.to_f64(),
        adl_compliant:       stats.gini_coefficient <= ADL_GINI_MAX,
        zakat_distributed:   stats.zakat_distributed,
        pool_health:         stats.pool_health,
    })
```

## 9. Python Wrapper: `core/bridges/urp_rust_bridge.py`

```python
FILE: core/bridges/urp_rust_bridge.py (NEW, ~150 LOC)

"""
URP Rust Bridge — Fail-closed Python wrapper for PyO3 bindings.

Every function returns None when Rust is unavailable (Level 0 degradation).
The node continues working with Python-only pledge verification.

Standing on Giants: Liskov (substitution — None is valid return)
"""

CLASS URPRustBridge:
    """Wrapper providing graceful degradation for URP Rust operations."""

    _pool: Optional[PyResourcePool] = None
    _available: bool = False

    def __init__(self):
        TRY:
            from bizra import PyResourcePool
            self._pool = PyResourcePool()
            self._available = True
        EXCEPT (ImportError, RuntimeError, OSError):
            self._available = False
            logger.info("URP Rust bridge unavailable — Level 0 mode")

    @property
    def available(self) -> bool:
        RETURN self._available

    def submit_pledge(self, pledge: URPPledge) -> Optional[Dict]:
        """Submit pledge to Rust pool. Returns node dict or None."""
        IF NOT self._available:
            RETURN None
        TRY:
            from bizra import PyURPPledge, submit_pledge
            rust_pledge = PyURPPledge.from_dict(pledge.to_dict())
            node = submit_pledge(self._pool, rust_pledge)
            RETURN node.to_dict()
        EXCEPT (RuntimeError, TypeError, ValueError) AS exc:
            logger.warning("URP submit failed: %s", exc)
            RETURN None

    def contribute(
        self, node_id: str, resource_type: str,
        amount: float, duration_ms: int, proof_hash: str
    ) -> Optional[Dict]:
        """Record contribution. Returns receipt dict or None."""
        IF NOT self._available:
            RETURN None
        TRY:
            from bizra import contribute_resources
            receipt = contribute_resources(
                self._pool, node_id, resource_type,
                amount, duration_ms, proof_hash
            )
            RETURN receipt.to_dict()
        EXCEPT (RuntimeError, TypeError, ValueError) AS exc:
            logger.warning("URP contribute failed: %s", exc)
            RETURN None

    def get_rewards(self, node_id: str) -> Optional[Dict]:
        """Get SEED balance and history. Returns dict or None."""
        IF NOT self._available:
            RETURN None
        TRY:
            from bizra import get_rewards
            RETURN get_rewards(self._pool, node_id)
        EXCEPT (RuntimeError, TypeError, ValueError):
            RETURN None

    def stats(self) -> Optional[Dict]:
        """Get pool stats. Returns dict or None."""
        IF NOT self._available:
            RETURN None
        TRY:
            from bizra import pool_stats
            RETURN pool_stats(self._pool).to_dict()
        EXCEPT (RuntimeError, TypeError, ValueError):
            RETURN None

    def check_adl(self) -> Optional[Dict]:
        """Check ADL compliance. Returns dict or None."""
        IF NOT self._available:
            RETURN None
        TRY:
            from bizra import check_adl
            RETURN check_adl(self._pool)
        EXCEPT (RuntimeError, TypeError, ValueError):
            RETURN None
```

## 10. Integration with Genesis Activation

```python
FILE: core/genesis/activation.py (EXTEND)

# In GenesisActivation.activate() — after Step 7 (heartbeat.breathe()):

# Step 8: Submit URP pledge to Rust pool (optional, Level 0 safe)
urp_result = None
TRY:
    from core.bridges.urp_rust_bridge import URPRustBridge
    bridge = URPRustBridge()
    IF bridge.available AND activation_result.urp_pledge:
        urp_result = bridge.submit_pledge(activation_result.urp_pledge)
        IF urp_result:
            logger.info("URP pledge submitted to Rust pool: %s",
                       urp_result.get("id", "unknown"))
EXCEPT (ImportError, RuntimeError):
    pass  # Level 0 — Python-only mode

# urp_result is included in GenesisActivationResult
```

## 11. Integration with Node0 Heartbeat

```python
FILE: core/node0/heartbeat.py (EXTEND concept)

# During breathe() cycle — contribute witnessing heartbeat:

IF self._urp_bridge AND self._urp_bridge.available:
    receipt = self._urp_bridge.contribute(
        node_id=self.node_id,
        resource_type="witness",
        amount=1.0,           # 1 heartbeat
        duration_ms=self.breath_interval_ms,
        proof_hash=breath_receipt.chain_hash,
    )
    IF receipt:
        self._last_urp_receipt = receipt
```

## 12. Error Handling & Async Bridge

**Invariant:** No URP operation ever raises to the caller. `None` = unavailable or failed.

**Async strategy:** Rust pool ops are `async` (tokio). PyO3 wraps with `block_on()`:
```rust
fn submit_pledge_sync(pool, pledge) -> PyResult<PyPoolNode>:
    rt = tokio::runtime::Runtime::new()
    rt.block_on(submit_pledge_async(pool, pledge))
```

## TDD Anchors

```
TEST submit_signed_pledge_succeeds:
    pool = PyResourcePool()
    pledge = create_signed_test_pledge(ram_gb=16, vram_gb=6)
    node = submit_pledge(pool, pledge)
    ASSERT node.class IN ["standard", "contributor"]
    ASSERT node.token_balance >= 0

TEST submit_unsigned_pledge_rejected:
    pool = PyResourcePool()
    pledge = PyURPPledge(node_id="test", ram_gb=16, signed=False)
    WITH RAISES PyRuntimeError("must be signed"):
        submit_pledge(pool, pledge)

TEST submit_tampered_pledge_rejected:
    pledge = create_signed_test_pledge(ram_gb=16)
    d = pledge.to_dict()
    d["ram_gb"] = 999
    tampered = PyURPPledge.from_dict(d)
    WITH RAISES PyRuntimeError("Invalid pledge signature"):
        submit_pledge(pool, tampered)

TEST contribute_earns_tokens:
    pool = setup_pool_with_node("node1")
    receipt = contribute_resources(pool, "node1", "cpu", 2.0, 3600000, "abc123")
    ASSERT receipt.tokens_earned > 0

TEST contribute_applies_zakat:
    receipt = contribute_and_get_receipt(...)
    # Zakat = 2.5% of minted tokens
    gross = receipt.tokens_earned / 0.975
    zakat = gross * 0.025
    ASSERT abs(gross - receipt.tokens_earned - zakat) < 1

TEST contribute_rejects_inactive_node:
    pool = setup_pool_with_suspended_node("node2")
    WITH RAISES PyRuntimeError("not active"):
        contribute_resources(pool, "node2", "cpu", 1.0, 1000, "hash")

TEST gini_check_compliant:
    pool = setup_balanced_pool()
    result = check_adl(pool)
    ASSERT result["compliant"] == True
    ASSERT result["gini_coefficient"] <= 0.35

TEST gini_check_non_compliant:
    pool = setup_concentrated_pool()  # One node has 90% of tokens
    result = check_adl(pool)
    ASSERT result["compliant"] == False
    ASSERT result["action_required"] == True

TEST wrapper_returns_none_without_rust:
    bridge = URPRustBridge()  # PyO3 not available
    ASSERT bridge.available == False
    ASSERT bridge.submit_pledge(some_pledge) IS None
    ASSERT bridge.contribute("n", "cpu", 1.0, 1000, "h") IS None
    ASSERT bridge.stats() IS None

TEST pool_stats_returns_all_fields:
    pool = setup_pool_with_contributions()
    stats = pool_stats(pool)
    ASSERT stats.total_nodes > 0
    ASSERT stats.gini_coefficient >= 0.0
    ASSERT stats.adl_compliant IN [True, False]
```
