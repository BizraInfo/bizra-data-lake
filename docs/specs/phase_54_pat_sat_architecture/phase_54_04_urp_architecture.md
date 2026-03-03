# Phase 54.4: Universal Resource Pool (URP) Architecture

> Standing on Giants: Nakamoto (decentralized consensus, 2008) · Lamport (distributed agreement, 1982) · Shannon (channel capacity, 1948) · Ostrom (commons governance, 2009) · Al-Ghazali (collective Ihsan, 1095)

## 1. Overview

The Universal Resource Pool (URP) is the collective infrastructure layer of BIZRA.
It is composed entirely of SAT agents contributed by every node. No single entity
controls the URP — it is governed by SAT consensus under constitutional constraints.

**Critical architecture**: Nodes NEVER connect directly to the BIZRA network.
Every node connects to the URP first. The URP connects to other URPs (the network).

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Node A  │    │  Node B  │    │  Node C  │
│ PAT↔SAT  │    │ PAT↔SAT  │    │ PAT↔SAT  │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              ┌──────▼──────┐
              │     URP     │
              │ (All SATs)  │
              │             │
              │ Guardian ×N │
              │ Librarian×N │
              │ Auditor  ×N │
              │ Healer   ×N │
              │ Herald   ×N │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   BIZRA     │
              │  Network    │
              │ (Other URPs)│
              └─────────────┘
```

## 2. URP Composition

The URP is NOT a separate server. It is the emergent collective of all SAT agents:

```pseudocode
CLASS UniversalResourcePool:
    """
    The URP is the sum of all SAT agents across all nodes.
    It has no central coordinator — departments self-organize.

    Standing on Giants: Ostrom (commons governance) — shared resources
    managed by the community through constitutional rules, not central authority.
    """

    # Department registries (one per SAT role)
    guardians:  Department[GuardianSAT]    # All Guardian SATs system-wide
    librarians: Department[LibrarianSAT]   # All Librarian SATs system-wide
    auditors:   Department[AuditorSAT]     # All Auditor SATs system-wide
    healers:    Department[HealerSAT]      # All Healer SATs system-wide
    heralds:    Department[HeraldSAT]      # All Herald SATs system-wide

    # Resource inventory (pledged by nodes)
    inventory:  ResourceInventory          # Compute, storage, bandwidth

    # Consensus engine
    consensus:  BFTConsensus               # Byzantine fault-tolerant agreement
```

## 3. Resource Inventory

Every node pledges resources when it joins. SAT manages the inventory:

```pseudocode
CLASS ResourceInventory:

    FUNCTION pledge(node_id: NodeID, hardware: HardwareInfo):
        """
        Node pledges a portion of its resources to the URP.
        This is NOT donation — it's reciprocal. You pledge to use.
        """
        pledge = ResourcePledge(
            node_id   = node_id,
            ram_gb    = hardware.ram * PLEDGE_RATIO,      # e.g., 50% of RAM
            vram_gb   = hardware.vram * PLEDGE_RATIO,     # e.g., 50% of VRAM
            storage_gb = hardware.storage * PLEDGE_RATIO, # e.g., 10% of disk
            bandwidth  = hardware.bandwidth * PLEDGE_RATIO,
            cpu_cores  = hardware.cores * PLEDGE_RATIO,
        )
        self.pledges[node_id] = pledge
        self.total_capacity = self._recalculate_total()

    FUNCTION allocate(request: ResourceRequest) -> Allocation:
        """
        SAT Healer department allocates resources from the pool.
        Uses ADL Gini constraint to ensure fair distribution.
        """
        # Check fairness — no single node can consume > fair share
        current_usage = self.usage_by_node[request.node_id]
        fair_share = self.total_capacity / len(self.pledges)

        IF current_usage + request.amount > fair_share * ADL_GINI_CEILING:
            RETURN Allocation(status=DENIED, reason="ADL Gini violation")

        # Allocate from nearest available node
        source = self.find_nearest_available(request)
        RETURN Allocation(
            status=APPROVED,
            source_node=source.node_id,
            amount=request.amount,
            expires=now() + request.duration,
        )
```

## 4. Node Connection Flow

```pseudocode
FUNCTION connect_node_to_network(node: Node) -> Connection:
    """
    A node NEVER connects directly to the network.
    It connects to URP, and URP connects to the network.

    This is the fundamental security innovation over blockchain:
    - Blockchain: Node → Network (direct, exposed)
    - BIZRA: Node → URP → Network (buffered, validated, constitutional)
    """

    # Step 1: Node presents credentials to URP
    credentials = NodeCredentials(
        node_id     = node.id,
        pat_count   = len(node.pat.agents),     # Should be 7
        sat_count   = len(node.sat.agents),     # Should be 5
        pledge      = node.resource_pledge,
        constitution_hash = node.constitution.hash(),
    )

    # Step 2: URP Guardian validates
    validation = urp.guardians.validate_node(credentials)
    IF NOT validation.approved:
        RETURN Connection(status=REJECTED, reason=validation.reason)

    # Step 3: URP assigns gateway endpoint
    gateway = urp.heralds.assign_gateway(node.id)

    # Step 4: Node gets a URP-mediated connection (not direct network)
    connection = Connection(
        status   = CONNECTED,
        gateway  = gateway,
        node_id  = node.id,
        scope    = "urp_mediated",    # NOT "direct_network"
    )

    # Step 5: All node traffic now flows through URP
    node.network_gateway = gateway
    RETURN connection
```

## 5. Why This Fixes Blockchain Security

### Problem 1: Direct Network Exposure
```
Traditional: Every node is a network endpoint.
             Attacker scans ports → finds nodes → attacks directly.

BIZRA:       Nodes are behind URP. Only URP endpoints are exposed.
             Attacker sees URP, not individual nodes.
             URP has 5N SAT agents validating every packet.
```

### Problem 2: No Separation of Concerns
```
Traditional: Validator logic and user logic run in same process.
             Compromised user wallet → compromised validator.

BIZRA:       PAT (user) and SAT (system) are isolated processes.
             Compromised PAT cannot affect SAT.
             SAT continues validating even if PAT is attacked.
```

### Problem 3: Sybil Attacks
```
Traditional: Attacker creates 1000 fake nodes → 1000 validators.
             Overwhelms consensus.

BIZRA:       Each node must pledge REAL resources (verified by hardware scan).
             Each node's SAT-5 joins URP with resource pledge.
             Fake nodes can't pledge real compute/storage.
             URP Guardians detect resource pledge fraud.
```

### Problem 4: Single Point of Failure
```
Traditional: Foundation/company controls network upgrades.
             If foundation compromised → network compromised.

BIZRA:       No foundation controls URP.
             SAT consensus governs changes.
             Constitution is immutable (Ihsan gate).
             Daughter Test prevents harmful upgrades.
```

## 6. Scaling: SAT Departments at Scale

```pseudocode
# Department self-organization based on network size

FUNCTION organize_departments(urp: URP):
    total_sats = urp.total_agent_count

    IF total_sats < 50:          # < 10 users
        # Simple: each department handles everything
        mode = "FLAT"

    ELIF total_sats < 5_000:     # < 1000 users
        # Regional: departments split by geography/latency
        mode = "REGIONAL"
        FOR dept IN urp.departments:
            dept.create_regional_shards(shard_count=total_sats // 100)

    ELIF total_sats < 5_000_000: # < 1M users
        # Hierarchical: departments have local leaders
        mode = "HIERARCHICAL"
        FOR dept IN urp.departments:
            dept.elect_coordinators(ratio=1_per_1000)
            dept.create_sub_departments(by="specialization")

    ELSE:                        # 1M+ users
        # Federated: multiple URPs form a network of URPs
        mode = "FEDERATED"
        urp.split_into_regional_urps(max_size=1_000_000)
        urp.create_inter_urp_heralds()
```

## 7. TDD Anchors

```python
class TestURPArchitecture:
    """Phase 54.4: Universal Resource Pool."""

    def test_node_cannot_connect_directly_to_network(self):
        node = create_test_node()
        with pytest.raises(DirectNetworkAccessDenied):
            node.connect_to_network_directly()

    def test_node_connects_through_urp(self):
        node = create_test_node()
        conn = node.connect_via_urp(mock_urp())
        assert conn.scope == "urp_mediated"

    def test_resource_pledge_required_for_connection(self):
        node = create_test_node(pledge=None)
        with pytest.raises(NoPledgeError):
            node.connect_via_urp(mock_urp())

    def test_adl_gini_enforced_on_allocation(self):
        urp = mock_urp(nodes=10)
        # One node tries to take 90% of resources
        greedy_request = ResourceRequest(amount=urp.total * 0.9)
        result = urp.allocate(greedy_request, node_id="greedy-node")
        assert result.status == AllocationStatus.DENIED
        assert "ADL Gini" in result.reason

    def test_sybil_detection_on_fake_pledge(self):
        urp = mock_urp()
        fake_node = create_test_node(fake_hardware=True)
        result = urp.guardians.validate_node(fake_node.credentials)
        assert result.approved == False

    def test_sat_count_scales_with_users(self):
        urp = mock_urp()
        for i in range(100):
            sat = mint_sat_team(f"node-{i}")
            urp.register_sat_team(sat)
        assert urp.total_agent_count == 500  # 100 * 5

    def test_department_strength_is_even(self):
        urp = mock_urp()
        for i in range(100):
            sat = mint_sat_team(f"node-{i}")
            urp.register_sat_team(sat)
        for role in SATRole:
            assert urp.get_department_strength(role) == 100
```
