# Phase 54.5: Three-Boundary Security Model

> Standing on Giants: Lamport (Byzantine generals, 1982) · Schneier (defense in depth, 2000) · Dijkstra (layered abstraction, 1968) · Nakamoto (proof-of-work trust, 2008) · Al-Ghazali (Ihsan as ethical boundary, 1095)

## 1. The Three Trust Boundaries

Traditional blockchain has ZERO trust boundaries between user and network.
BIZRA has THREE:

```
                    BOUNDARY 1          BOUNDARY 2          BOUNDARY 3
                    ──────────          ──────────          ──────────
User ──► PAT-7  │  PAT → SAT-5  │  SAT → URP  │  URP → Network
         (personal) │  (constitutional) │  (consensus)  │  (federation)
                    │                   │                │
     User's space   │  Validation      │  Resource     │  Inter-node
     User's data    │  layer           │  pool         │  communication
     User's agents  │                   │                │
```

### Boundary 1: User ↔ PAT (Personalization Boundary)
- PAT knows the user deeply (goals, preferences, style)
- PAT data is sovereign — encrypted, local, user-controlled
- Attack surface: social engineering the user
- Defense: PAT Ethicist scores Ihsan, applies Daughter Test

### Boundary 2: PAT ↔ SAT (Constitutional Boundary)
- PAT requests go through PCI envelope validation
- SAT Guardian checks: signature, Ihsan, ADL Gini, malicious payload
- SAT Auditor records evidence chain
- Attack surface: forged PCI envelopes, replay attacks
- Defense: ed25519 signatures, timestamp freshness, rate limiting

### Boundary 3: SAT ↔ URP (Consensus Boundary)
- SAT can only reach network through URP
- URP validates resource pledges, consensus participation
- BFT consensus ensures no single malicious SAT can corrupt the pool
- Attack surface: compromised SAT agents, resource pledge fraud
- Defense: Byzantine fault tolerance, hardware verification, ADL Gini

## 2. Attack Scenarios

### Scenario 1: Compromised User Account

```pseudocode
ATTACK: Attacker gains access to user's device/credentials.

BLOCKCHAIN RESULT:
    Attacker controls validator + user wallet.
    Can vote maliciously, steal funds, corrupt state.
    Network-wide damage.

BIZRA RESULT:
    Attacker controls PAT-7 (user's agents).
    PAT sends malicious request to SAT.
    SAT Guardian: "Ihsan 0.31 — REJECTED."
    SAT Auditor: "Anomalous behavior flagged."
    SAT Healer: "Account quarantined for review."
    DAMAGE: Zero to network. User's PAT isolated.
    SAT continues operating normally.
```

### Scenario 2: Sybil Attack (Fake Nodes)

```pseudocode
ATTACK: Create 10,000 fake nodes to overwhelm consensus.

BLOCKCHAIN RESULT:
    10,000 fake validators join consensus.
    51% attack possible.
    Network compromised.

BIZRA RESULT:
    Each fake node must:
    1. Mint PAT-7 (requires identity verification)
    2. Mint SAT-5 (requires hardware scan — REAL compute)
    3. Pledge resources (verified against actual hardware)
    4. Pass Guardian validation (constitutional check)

    Fake nodes fail at step 3: no real hardware to pledge.
    Guardian detects pledge fraud → node rejected.
    Even if some slip through: BFT consensus requires 2/3+1
    honest SAT agents. Attacker needs 3.3M+ real machines
    to overwhelm 5M honest SAT agents.
```

### Scenario 3: Malicious SAT Agent

```pseudocode
ATTACK: A SAT agent is corrupted (code injection).

BLOCKCHAIN RESULT:
    Corrupted validator can:
    - Sign invalid blocks
    - Censor transactions
    - Participate in nothing-at-stake attacks

BIZRA RESULT:
    Corrupted SAT tries to approve invalid request.
    Other Guardians in the department: "Constitutional hash mismatch."
    Auditor: "Evidence chain broken for SAT-node42-guardian."
    Healer: "Quarantining SAT-node42-guardian. Replacing with backup."

    BFT consensus: requires 2/3+1 agreement.
    One corrupted SAT out of 500+ in department = negligible.
    Automatically detected, quarantined, replaced.
```

### Scenario 4: Network Partition

```pseudocode
ATTACK: Internet outage splits network into two halves.

BLOCKCHAIN RESULT:
    Chain fork. Both halves continue producing blocks.
    When reconnected: one half's work is discarded.
    Users lose transactions. Chaos.

BIZRA RESULT:
    Each partition has its own URP subset.
    SAT Heralds detect partition → switch to partition-tolerant mode.
    Both halves continue operating (eventually consistent).
    When reconnected: SAT Librarians merge state via CRDT.
    No data loss. No forked chains. Constitutional constraints
    ensure both halves maintained Ihsan compliance.
```

## 3. Security Properties

| Property | How Achieved |
|----------|-------------|
| **Confidentiality** | PAT data encrypted with user's sovereign key. SAT cannot read user data. |
| **Integrity** | Every action hash-chained in evidence ledger (Auditor). Constitution hash verified. |
| **Availability** | SAT Healer auto-recovers. BFT tolerates f < n/3 failures. URP is distributed. |
| **Non-repudiation** | ed25519 signatures on every PCI envelope. Evidence chain is append-only. |
| **Fairness** | ADL Gini <= 0.35 on all resource allocation. No monopolization. |
| **Isolation** | PAT/SAT process isolation. User compromise cannot affect system. |
| **Self-healing** | Healer department detects and repairs without human intervention. |

## 4. Constitutional Security Gates

```pseudocode
CLASS ConstitutionalSecurityGate:
    """
    Every cross-boundary action passes through this gate.
    Gate is immutable — cannot be modified by any agent or user.
    """

    FUNCTION validate(action: Action) -> GateResult:
        checks = []

        # Ihsan check (excellence threshold)
        ihsan = compute_ihsan(action)
        checks.append(GateCheck(
            name="ihsan",
            passed=ihsan >= self.ihsan_threshold,
            value=ihsan,
            threshold=self.ihsan_threshold,
        ))

        # Daughter Test (ethical check)
        daughter_ok = self.daughter_test(action)
        checks.append(GateCheck(
            name="daughter_test",
            passed=daughter_ok,
        ))

        # ADL Gini (fairness check)
        gini = compute_adl_gini(action)
        checks.append(GateCheck(
            name="adl_gini",
            passed=gini <= ADL_GINI_THRESHOLD,
            value=gini,
            threshold=ADL_GINI_THRESHOLD,
        ))

        # SNR (signal quality)
        snr = compute_snr(action)
        checks.append(GateCheck(
            name="snr",
            passed=snr >= self.snr_threshold,
            value=snr,
            threshold=self.snr_threshold,
        ))

        # Final verdict
        all_passed = all(c.passed for c in checks)
        RETURN GateResult(
            status=APPROVED if all_passed else REJECTED,
            checks=checks,
            timestamp=now(),
        )
```

## 5. TDD Anchors

```python
class TestSecurityModel:
    """Phase 54.5: Three-boundary security model."""

    def test_compromised_pat_cannot_affect_sat(self):
        sat = mint_sat_team("node1")
        malicious_envelope = PCIEnvelope(
            sender="pat-node1-coder",
            action="delete_all_data",
            ihsan=0.10,  # Obviously malicious
        )
        result = sat.gateway.submit_request(malicious_envelope)
        assert result.status == GatewayStatus.REJECTED

    def test_three_boundaries_exist(self):
        node = create_test_node()
        # Boundary 1: user can reach PAT
        assert node.pat.is_accessible_to_user()
        # Boundary 2: PAT can reach SAT (via gateway)
        assert node.pat_to_sat_gateway.is_active()
        # Boundary 3: SAT can reach URP
        assert node.sat_to_urp_connection.is_active()
        # User CANNOT reach SAT directly
        assert not node.sat.is_accessible_to_user()
        # User CANNOT reach URP directly
        assert not node.urp_connection.is_accessible_to_user()

    def test_replay_attack_blocked(self):
        gateway = PATtoSATGateway(mock_sat_team())
        envelope = make_valid_envelope(timestamp=now())
        gateway.submit_request(envelope)  # First time: OK

        # Replay same envelope
        with pytest.raises(ReplayDetected):
            gateway.submit_request(envelope)  # Second time: blocked

    def test_constitutional_gate_is_immutable(self):
        gate = ConstitutionalSecurityGate()
        with pytest.raises(ImmutableError):
            gate.ihsan_threshold = 0.50  # Cannot lower threshold
```
