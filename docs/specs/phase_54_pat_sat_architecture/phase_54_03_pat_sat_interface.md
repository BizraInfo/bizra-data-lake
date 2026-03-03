# Phase 54.3: PAT ↔ SAT Interface Protocol

> Standing on Giants: Lamport (distributed message ordering, 1978) · Dijkstra (guarded commands, 1975) · Al-Ghazali (Ihsan as interface contract, 1095) · Anthropic (constitutional validation, 2023)

## 1. Overview

PAT and SAT communicate through a formal, constitutional interface. The user NEVER
talks to SAT directly — PAT acts as the user's proxy, translating user intent into
validated system requests.

This is the security boundary that blockchain ecosystems lack.

## 2. Communication Rules

```
RULE 1: User → PAT              (direct)
RULE 2: PAT → SAT               (via PCI envelope, validated)
RULE 3: SAT → PAT               (response only, never initiates contact with user)
RULE 4: SAT → URP               (system operations, autonomous)
RULE 5: SAT → SAT               (inter-department coordination)
RULE 6: User → SAT              (FORBIDDEN — hard block)
RULE 7: SAT → User              (FORBIDDEN — hard block)
```

## 3. Request Flow

```
User: "Send 10 BZ tokens to Ahmed"
    │
    ▼
┌─────────────────────────────────────────────┐
│ PAT Planner: Decomposes into sub-tasks      │
│   1. Verify sender balance                  │
│   2. Validate recipient exists              │
│   3. Create transfer transaction            │
│   4. Sign with user's sovereign key         │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ PAT Ethicist: Ihsan check                   │
│   - Amount reasonable? ✓                    │
│   - Recipient legitimate? ✓                 │
│   - Daughter test? ✓                        │
│   - Ihsan score: 0.97 (>= 0.95) ✓          │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ PAT → SAT GATEWAY (PCI Envelope)            │
│                                              │
│  PCIEnvelope {                               │
│    sender:    "pat-node42-integrator"        │
│    action:    "transfer_tokens"              │
│    payload:   { to: "ahmed", amount: 10 }   │
│    ihsan:     0.97                           │
│    signature: <ed25519_sig>                  │
│    timestamp: 2026-02-27T03:00:00Z          │
│  }                                           │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ SAT Guardian: Security validation           │
│   - Signature valid? ✓                      │
│   - Ihsan >= 0.95? ✓                        │
│   - ADL Gini impact? 0.12 (<= 0.35) ✓      │
│   - Malicious payload? No ✓                 │
│   - Rate limit? Within bounds ✓             │
│   STATUS: APPROVED                          │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ SAT Auditor: Evidence recording             │
│   - Chain hash: sha256(prev + action)       │
│   - Receipt written to evidence ledger      │
│   - Ihsan attestation logged                │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ SAT Herald: Forward to URP → Network        │
│   - Route to Ahmed's node                   │
│   - Consensus participation                 │
│   - Confirmation receipt                    │
└─────────────┬───────────────────────────────┘
              │
              ▼
Response flows back: SAT Herald → SAT → PAT → User
```

## 4. Interface Contract (Pseudocode)

```pseudocode
CLASS PATtoSATGateway:
    """
    The formal interface between PAT and SAT.
    All cross-boundary communication goes through this gateway.

    Standing on Giants: Dijkstra (guarded commands) — every request
    must satisfy preconditions before execution.
    """

    FUNCTION submit_request(envelope: PCIEnvelope) -> GatewayResponse:
        """PAT submits a validated request to SAT."""

        # ── Precondition checks ──
        ASSERT envelope.sender.startswith("pat-"), "Only PAT can submit"
        ASSERT envelope.ihsan >= UNIFIED_IHSAN_THRESHOLD, "Ihsan too low"
        ASSERT envelope.verify_signature(), "Invalid signature"
        ASSERT envelope.timestamp_fresh(max_age=60), "Stale request"

        # ── Route to appropriate SAT department ──
        IF envelope.action.requires_security_check:
            guardian_result = self.guardian.validate(envelope)
            IF guardian_result.rejected:
                RETURN GatewayResponse(
                    status=REJECTED,
                    reason=guardian_result.rejection_reason,
                )

        # ── Record in evidence chain ──
        self.auditor.record_action(envelope, guardian_result)

        # ── Execute via appropriate department ──
        IF envelope.action.is_network_operation:
            result = self.herald.forward_to_urp(envelope)
        ELIF envelope.action.is_data_operation:
            result = self.librarian.process(envelope)
        ELIF envelope.action.is_resource_request:
            result = self.healer.allocate(envelope)
        ELSE:
            result = self.guardian.handle_unknown(envelope)

        # ── Return result to PAT ──
        RETURN GatewayResponse(
            status=APPROVED,
            result=result,
            evidence_hash=self.auditor.last_hash(),
        )
```

## 5. What PAT Can Request from SAT

| Request Type | SAT Department | Example |
|-------------|----------------|---------|
| Network send | Herald | Send tokens, broadcast message |
| Resource allocation | Healer | Request more compute, storage |
| Data query | Librarian | Search system knowledge graph |
| Security check | Guardian | Verify another node's identity |
| Compliance report | Auditor | Get user's Ihsan history |

## 6. What SAT CANNOT Do to PAT

| Forbidden Action | Reason |
|-----------------|--------|
| Read user data | Sovereignty violation |
| Modify PAT behavior | PAT serves user, not system |
| Override user preferences | User autonomy |
| Initiate contact with user | User talks to PAT, not SAT |
| Block PAT without constitutional reason | Only constitution can block |

## 7. TDD Anchors

```python
class TestPATSATInterface:
    """Phase 54.3: PAT ↔ SAT communication protocol."""

    def test_pat_can_submit_valid_request(self):
        gateway = PATtoSATGateway(mock_sat_team())
        envelope = make_valid_envelope(sender="pat-node1-integrator")
        result = gateway.submit_request(envelope)
        assert result.status == GatewayStatus.APPROVED

    def test_non_pat_sender_rejected(self):
        gateway = PATtoSATGateway(mock_sat_team())
        envelope = make_valid_envelope(sender="rogue-agent")
        with pytest.raises(AssertionError):
            gateway.submit_request(envelope)

    def test_low_ihsan_rejected(self):
        gateway = PATtoSATGateway(mock_sat_team())
        envelope = make_valid_envelope(ihsan=0.50)
        with pytest.raises(AssertionError):
            gateway.submit_request(envelope)

    def test_stale_timestamp_rejected(self):
        gateway = PATtoSATGateway(mock_sat_team())
        envelope = make_valid_envelope(timestamp=old_timestamp())
        with pytest.raises(AssertionError):
            gateway.submit_request(envelope)

    def test_every_request_recorded_in_evidence(self):
        gateway = PATtoSATGateway(mock_sat_team())
        envelope = make_valid_envelope()
        gateway.submit_request(envelope)
        assert gateway.auditor.evidence_count == 1

    def test_network_ops_route_through_herald(self):
        gateway = PATtoSATGateway(mock_sat_team())
        envelope = make_valid_envelope(action=NetworkSend())
        gateway.submit_request(envelope)
        assert gateway.herald.forward_count == 1

    def test_sat_cannot_read_user_data_through_gateway(self):
        gateway = PATtoSATGateway(mock_sat_team())
        with pytest.raises(AccessDenied):
            gateway.guardian.read_user_data("user123")
```
