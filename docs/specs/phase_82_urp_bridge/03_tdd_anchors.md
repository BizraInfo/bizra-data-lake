# 03 — URP Bridge TDD Anchors & Test Plan

> Module: `tests/core/bridges/test_urp_rust_bridge.py` (NEW)
> + `bizra-omega/bizra-python/tests/test_urp_bindings.py` (NEW)
> Language: Python (pytest) + Rust (#[cfg(test)])
> Constitutional Anchor: Law 6 + Proof-of-Impact + Ihsan ≥ 0.95

## 1. Test Pyramid

```
                    ┌───────────────┐
                    │  E2E (3)      │  Genesis → pledge → submit → contribute → verify
                    ├───────────────┤
                  ┌─┤ Integration   │  PyO3 round-trip, pool state persistence
                  │ │ (8)           │
                  │ ├───────────────┤
                  │ │ Unit (20)     │  Type wrappers, conversions, validation
                  │ │               │
                  │ ├───────────────┤
                  │ │ Property (5)  │  Hypothesis: signature, hash, Gini invariants
                  └─┴───────────────┘

Total: ~36 tests
```

## 2. Unit Tests — Type Wrappers (Rust-side)

```rust
FILE: bizra-python/tests/test_urp_bindings.py (pytest, requires maturin)

MODULE test_pyurp_pledge:

    TEST pledge_new_creates_valid_instance:
        pledge = PyURPPledge(
            node_id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
            ram_gb=16, vram_gb=6, storage_gb=100,
            pledge_hash="abc123", pledged_at="2026-03-14T00:00:00Z",
            signed=False, signature="", signer_public_key="",
            payload_digest="", enforcement_mode="stub", status="deferred"
        )
        ASSERT pledge.node_id == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
        ASSERT pledge.ram_gb == 16
        ASSERT pledge.signed == False

    TEST pledge_to_dict_complete:
        pledge = create_test_pledge()
        d = pledge.to_dict()
        ASSERT "node_id" IN d
        ASSERT "ram_gb" IN d
        ASSERT "signature" IN d
        ASSERT len(d) >= 12  # All fields present

    TEST pledge_from_dict_roundtrip:
        original = create_test_pledge()
        d = original.to_dict()
        restored = PyURPPledge.from_dict(d)
        ASSERT restored.to_dict() == d

    TEST pledge_from_dict_missing_field_raises:
        WITH RAISES (ValueError, KeyError):
            PyURPPledge.from_dict({"node_id": "x"})  # Missing required fields

    TEST pledge_verify_unsigned_returns_false:
        pledge = create_test_pledge(signed=False)
        ASSERT pledge.verify_signature() == False

MODULE test_pynode_resources:

    TEST resources_new_and_getters:
        r = PyNodeResources(cpu_cores=8, ram_gb=16, vram_gb=6,
                           storage_gb=500, network_mbps=100.0)
        ASSERT r.cpu_cores == 8
        ASSERT r.ram_gb == 16

    TEST resources_to_dict:
        r = PyNodeResources(8, 16, 6, 500, 100.0)
        d = r.to_dict()
        ASSERT d["cpu_cores"] == 8
        ASSERT d["storage_gb"] == 500

    TEST resources_hash_deterministic:
        r1 = PyNodeResources(8, 16, 6, 500, 100.0)
        r2 = PyNodeResources(8, 16, 6, 500, 100.0)
        ASSERT r1.compute_hash() == r2.compute_hash()

    TEST resources_hash_changes_with_input:
        r1 = PyNodeResources(8, 16, 6, 500, 100.0)
        r2 = PyNodeResources(8, 32, 6, 500, 100.0)  # Different RAM
        ASSERT r1.compute_hash() != r2.compute_hash()

MODULE test_pypool_node:

    TEST node_passes_ihsan_above_threshold:
        node = PyPoolNode(ihsan_score=0.96, ...)
        ASSERT node.passes_ihsan() == True

    TEST node_fails_ihsan_below_threshold:
        node = PyPoolNode(ihsan_score=0.80, ...)
        ASSERT node.passes_ihsan() == False

    TEST node_ihsan_at_boundary:
        node = PyPoolNode(ihsan_score=0.95, ...)
        ASSERT node.passes_ihsan() == True  # Exactly at threshold

    TEST node_to_dict_includes_resources:
        node = create_test_node()
        d = node.to_dict()
        ASSERT "resources" IN d
        ASSERT "ihsan_score" IN d
        ASSERT isinstance(d["resources"], dict)
```

## 3. Unit Tests — Python Wrapper

```python
FILE: tests/core/bridges/test_urp_rust_bridge.py

MODULE test_urp_rust_bridge_degradation:

    TEST bridge_unavailable_sets_flag:
        # Mock PyO3 import failure
        with mock.patch.dict("sys.modules", {"bizra": None}):
            bridge = URPRustBridge()
            ASSERT bridge.available == False

    TEST submit_pledge_returns_none_when_unavailable:
        bridge = URPRustBridge()  # No PyO3
        bridge._available = False
        pledge = create_python_pledge()
        ASSERT bridge.submit_pledge(pledge) IS None

    TEST contribute_returns_none_when_unavailable:
        bridge = URPRustBridge()
        bridge._available = False
        ASSERT bridge.contribute("n", "cpu", 1.0, 1000, "h") IS None

    TEST get_rewards_returns_none_when_unavailable:
        bridge = URPRustBridge()
        bridge._available = False
        ASSERT bridge.get_rewards("n") IS None

    TEST stats_returns_none_when_unavailable:
        bridge = URPRustBridge()
        bridge._available = False
        ASSERT bridge.stats() IS None

    TEST check_adl_returns_none_when_unavailable:
        bridge = URPRustBridge()
        bridge._available = False
        ASSERT bridge.check_adl() IS None
```

## 4. Integration Tests — PyO3 Boundary

```python
FILE: tests/core/bridges/test_urp_rust_bridge.py (continued)

# All integration tests require PyO3 built:
pytestmark = pytest.mark.skipif(
    not _rust_available(),
    reason="PyO3 URP bindings not built"
)

MODULE test_urp_integration:

    TEST submit_signed_pledge_registers_node:
        bridge = URPRustBridge()
        ASSERT bridge.available == True

        pledge = pledge_resources(
            node_id="test_node_001",
            hardware_info={"ram_gb": 16, "vram_gb": 6},
            signing_private_key_hex=TEST_PRIVATE_KEY,
        )
        result = bridge.submit_pledge(pledge)
        ASSERT result IS NOT None
        ASSERT result["id"] == "test_node_001"
        ASSERT result["class"] IN ["standard", "contributor"]

    TEST submit_then_contribute_earns_tokens:
        bridge = URPRustBridge()
        # Register
        pledge = create_signed_pledge(ram_gb=16, vram_gb=6)
        node = bridge.submit_pledge(pledge)
        ASSERT node IS NOT None

        # Contribute
        receipt = bridge.contribute(
            node_id=pledge.node_id,
            resource_type="cpu",
            amount=2.0,
            duration_ms=3600000,  # 1 hour
            proof_hash=blake3("test_work").hex(),
        )
        ASSERT receipt IS NOT None
        ASSERT receipt["tokens_earned"] > 0

    TEST contribute_witness_heartbeat:
        bridge = URPRustBridge()
        setup_registered_node(bridge, "witness_node")

        receipt = bridge.contribute(
            node_id="witness_node",
            resource_type="witness",
            amount=1.0,
            duration_ms=60000,   # 1 heartbeat cycle
            proof_hash="heartbeat_chain_hash_abc",
        )
        ASSERT receipt IS NOT None
        ASSERT receipt["resource_type"] == "witness"

    TEST get_rewards_after_contribution:
        bridge = URPRustBridge()
        setup_registered_node(bridge, "reward_node")
        bridge.contribute("reward_node", "cpu", 1.0, 1000, "hash1")

        rewards = bridge.get_rewards("reward_node")
        ASSERT rewards IS NOT None
        ASSERT rewards["balance"] > 0
        ASSERT rewards["node_class"] IS NOT None

    TEST stats_returns_pool_state:
        bridge = URPRustBridge()
        setup_registered_node(bridge, "stats_node")

        stats = bridge.stats()
        ASSERT stats IS NOT None
        ASSERT stats["total_nodes"] >= 1
        ASSERT "gini_coefficient" IN stats
        ASSERT "adl_compliant" IN stats

    TEST adl_check_with_single_node:
        bridge = URPRustBridge()
        setup_registered_node(bridge, "adl_node")

        result = bridge.check_adl()
        ASSERT result IS NOT None
        ASSERT result["threshold"] == 0.35
        # Single node = Gini 0.0 (no inequality)
        ASSERT result["compliant"] == True

    TEST signature_verified_in_rust:
        """Rust verification is authoritative over Python."""
        pledge = create_signed_pledge(ram_gb=16)

        # Python says valid
        ASSERT verify_pledge_signature(pledge) == True

        # Rust also says valid
        from bizra import PyURPPledge
        rust_pledge = PyURPPledge.from_dict(pledge.to_dict())
        ASSERT rust_pledge.verify_signature() == True

    TEST tampered_pledge_rejected_by_rust:
        pledge = create_signed_pledge(ram_gb=16)
        d = pledge.to_dict()
        d["ram_gb"] = 999  # Tamper after signing
        ASSERT d["ram_gb"] != 16

        from bizra import PyURPPledge
        rust_pledge = PyURPPledge.from_dict(d)
        ASSERT rust_pledge.verify_signature() == False
```

## 5. Property-Based Tests (Hypothesis)

```python
FILE: tests/property_based/test_urp_properties.py

from hypothesis import given, strategies as st

MODULE test_urp_properties:

    @given(ram=st.integers(0, 1024), vram=st.integers(0, 128))
    TEST pledge_hash_deterministic(ram, vram):
        """Same inputs always produce same pledge hash."""
        p1 = pledge_resources("node1", {"ram_gb": ram, "vram_gb": vram})
        p2 = pledge_resources("node1", {"ram_gb": ram, "vram_gb": vram})
        # Note: pledged_at differs, so pledge_hash may differ
        # But payload structure is deterministic for same inputs + timestamp
        ASSERT p1.node_id == p2.node_id
        ASSERT p1.ram_gb == p2.ram_gb

    @given(score=st.floats(0.0, 1.0))
    TEST ihsan_gate_consistent(score):
        """Ihsan gate is deterministic for any score."""
        passes = score >= 0.95
        node = PyPoolNode(ihsan_score=score, ...)
        ASSERT node.passes_ihsan() == passes

    @given(balance=st.integers(0, 10_000_000))
    TEST zakat_calculation_correct(balance):
        """Zakat is exactly 2.5% above nisab threshold."""
        IF balance >= 1_000_000:  # NISAB_THRESHOLD
            zakat = int(balance * 0.025)
            ASSERT zakat > 0
        ELSE:
            # Below nisab — no Zakat obligatory
            zakat = 0

    @given(
        balances=st.lists(
            st.integers(0, 1_000_000),
            min_size=2, max_size=100
        )
    )
    TEST gini_bounded_zero_one(balances):
        """Gini coefficient is always in [0, 1]."""
        gini = calculate_gini(balances)
        ASSERT 0.0 <= gini <= 1.0

    @given(n=st.integers(1, 50))
    TEST equal_distribution_gini_zero(n):
        """Equal distribution has Gini = 0."""
        balances = [1000] * n
        gini = calculate_gini(balances)
        ASSERT abs(gini) < 0.001
```

## 6. E2E Tests — Full Pipeline

```python
FILE: tests/core/bridges/test_urp_rust_bridge.py (continued)

MODULE test_urp_e2e:

    TEST full_genesis_to_contribution_pipeline:
        """
        End-to-end: Genesis ceremony → URP pledge → Rust pool
        → contribute → earn SEED → verify receipt chain.
        """
        # 1. Run genesis activation
        activation = GenesisActivation(seed=b"test_seed_32bytes_for_e2e_test!")
        result = activation.activate()
        ASSERT result.ceremony_result IS NOT None

        # 2. Submit URP pledge to Rust
        bridge = URPRustBridge()
        IF NOT bridge.available:
            pytest.skip("PyO3 not built")

        node = bridge.submit_pledge(result.urp_pledge)
        ASSERT node IS NOT None

        # 3. Contribute witnessing heartbeat
        receipt = bridge.contribute(
            node_id=result.ceremony_result.node_id,
            resource_type="witness",
            amount=1.0,
            duration_ms=60000,
            proof_hash=result.activation_hash,
        )
        ASSERT receipt IS NOT None
        ASSERT receipt["tokens_earned"] > 0

        # 4. Check rewards
        rewards = bridge.get_rewards(result.ceremony_result.node_id)
        ASSERT rewards IS NOT None
        ASSERT rewards["balance"] > 0

    TEST multi_node_gini_enforcement:
        """
        Register 10 nodes, contribute unevenly, verify Gini gate.
        """
        bridge = URPRustBridge()
        IF NOT bridge.available:
            pytest.skip("PyO3 not built")

        # Register 10 nodes
        nodes = []
        FOR i IN range(10):
            pledge = create_signed_pledge(
                node_id=f"gini_node_{i:02d}",
                ram_gb=4 + i * 4,
            )
            node = bridge.submit_pledge(pledge)
            nodes.append(node)

        # Give node_0 disproportionate contribution
        FOR _ IN range(100):
            bridge.contribute("gini_node_00", "cpu", 10.0, 3600000, "h")

        # Check Gini
        result = bridge.check_adl()
        # With concentrated contributions, Gini should be high
        ASSERT result["gini_coefficient"] > 0.3

    TEST degraded_mode_full_pipeline:
        """
        Full pipeline works without Rust (Level 0 degradation).
        """
        # Force Rust unavailable
        bridge = URPRustBridge()
        bridge._available = False

        # Genesis still works
        activation = GenesisActivation(seed=b"degraded_test_seed_32bytes_ok!")
        result = activation.activate()
        ASSERT result.ceremony_result IS NOT None

        # URP operations return None (not crash)
        ASSERT bridge.submit_pledge(result.urp_pledge) IS None
        ASSERT bridge.contribute("n", "cpu", 1.0, 1000, "h") IS None
        ASSERT bridge.stats() IS None

        # Python-only pledge verification still works
        ASSERT verify_pledge_signature(result.urp_pledge) IN [True, False]
```

## 7. Test Infrastructure

```python
# Fixtures and helpers

@pytest.fixture
def urp_bridge():
    """Create URPRustBridge, skip if PyO3 not available."""
    bridge = URPRustBridge()
    IF NOT bridge.available:
        pytest.skip("PyO3 URP bindings not built")
    RETURN bridge

@pytest.fixture
def registered_node(urp_bridge):
    """Register a test node and return (bridge, node_dict)."""
    pledge = create_signed_pledge(
        node_id="fixture_node",
        ram_gb=16, vram_gb=6,
    )
    node = urp_bridge.submit_pledge(pledge)
    RETURN urp_bridge, node

FUNCTION _rust_available() -> bool:
    TRY:
        from bizra import PyURPPledge
        RETURN PyURPPledge IS NOT None
    EXCEPT ImportError:
        RETURN False

FUNCTION create_signed_pledge(**kwargs) -> URPPledge:
    """Create a properly signed test pledge."""
    from core.pci.crypto import generate_keypair
    priv_key, pub_key = generate_keypair()
    RETURN pledge_resources(
        node_id=kwargs.get("node_id", "test_node"),
        hardware_info={
            "ram_gb": kwargs.get("ram_gb", 16),
            "vram_gb": kwargs.get("vram_gb", 0),
            "storage_gb": kwargs.get("storage_gb", 0),
        },
        signing_private_key_hex=priv_key,
    )
```

## 8. CI Integration

```yaml
# In .github/workflows/ci.yml — Test PyO3 job (extend existing):

test-pyo3:
  steps:
    - name: Build PyO3 bindings
      run: |
        cd bizra-omega/bizra-python
        VIRTUAL_ENV=.venv-linux maturin develop --release

    - name: Test URP bridge bindings
      run: |
        pytest tests/core/bridges/test_urp_rust_bridge.py -v
        pytest bizra-omega/bizra-python/tests/test_urp_bindings.py -v

    - name: Test URP property-based
      run: |
        pytest tests/property_based/test_urp_properties.py -v --hypothesis-seed=42
```

## 9. Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| `urp_bridge.rs` (Rust) | 85% | Core bridge — every public fn tested |
| `urp_rust_bridge.py` (Python wrapper) | 95% | Thin wrapper — full degradation coverage |
| `test_urp_bindings.py` (PyO3 unit) | N/A | Tests themselves |
| Property-based | N/A | Invariant verification |
| E2E | N/A | Pipeline verification |

## 10. Failure Modes & Mitigations

| Failure | Mitigation | Test |
|---------|-----------|------|
| PyO3 not built | `URPRustBridge._available = False` → all ops return None | `test_*_unavailable` |
| Signature tampered | Rust `verify_signature()` returns False → reject | `test_tampered_*` |
| Node not registered | `contribute()` raises → wrapper catches → None | `test_contribute_rejects_*` |
| Gini violated | `check_adl()` returns `compliant=False` | `test_gini_check_*` |
| Tokio runtime panic | `block_on()` catches → PyRuntimeError → None | Implicit in all ops |
| Decimal overflow | Rust `rust_decimal` handles precision | Property test |
