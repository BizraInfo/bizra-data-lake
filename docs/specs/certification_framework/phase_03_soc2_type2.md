# Phase 03 — SOC 2 Type II: Trust Service Criteria

> Source: BIZRA Quality Standards & Certification Framework
> Standard: AICPA SOC 2 Type II — Trust Service Criteria (2017 + 2022 supplement)
> Status: SPECIFICATION SEALED | SNR: 0.94

---

SOC 2 Type II evaluates control effectiveness over an audit period, not a point in time.
Traditional systems bolt on monitoring after the fact. BIZRA inverts this: every state
transition produces a cryptographically signed ActionReceipt, every action passes a FATE
Gate pre-execution veto, and every event chains into an append-only Merkle log. The audit
trail is not a feature -- it is the execution substrate. Continuous compliance is a
consequence of the architecture, not an addition to it.

The five Trust Service Categories map directly to existing BIZRA sovereignty primitives:
Security (Ed25519 + FATE + TeleScript Default Deny), Availability (Federation + Self-Healing),
Processing Integrity (Fixed-Point + PoI_EMIT + Merkle Chain), Confidentiality (Sovereignty I-6
+ Shamir 3-of-5), and Privacy (Local-First + Pseudonymous DID + Consent).

---

## 1. Functional Requirements

### FR-C30: Security Controls (Common Criteria)

**SOC 2 Criteria:** CC1 (Control Environment), CC5 (Monitoring), CC6 (Logical/Physical Access), CC7 (System Operations), CC8 (Change Management).

BIZRA enforces security at three nested perimeters:

| Perimeter | Mechanism | Failure Mode | Evidence |
|:---|:---|:---|:---|
| **Identity** | Ed25519 keypair per node, generated locally, never transmitted | Impersonation | Signature verification on every receipt |
| **Pre-Execution** | FATE Gate — Formal (Z3), Alignment, Testing, Ethical veto | Unauthorized state transition | `fate_pass` boolean on ActionReceipt |
| **Runtime** | TeleScript Default Deny — file/network/system gated by sovereign grant | Privilege escalation | TeleScript permission log with grant/deny entries |

**Quality Gates:**

| Metric | Threshold | Measurement |
|:---|:---|:---|
| FATE Gate uptime | 100% over audit period | Zero receipts with `fate_pass=None` |
| Authentication failure rate | < 0.1% of requests | Challenge-response rejection log |
| Unauthorized state transitions | 0 | Receipts without valid `oracle_signature` |
| TeleScript deny events (unauthorized) | 0 unresolved | All denied requests either escalated or confirmed-blocked |

**PCI Envelope:** Every action is wrapped in a Proof-Carrying Inference envelope. The proof
accompanies the action, not a separate audit artifact. An action without PCI cannot execute.

**Change Management (CC8):** All code changes flow through FATE Gate validation. The gate
enforces formal verification (Z3 constraint satisfaction), alignment checks, test coverage
requirements, and ethical review. No change reaches production without a signed receipt chain.

**Evidence:** FATE Gate logs, TeleScript permission audit trail, Ed25519 signature verification
records, PCI envelope chain, challenge-response failure log.

---

### FR-C31: Availability Controls

**SOC 2 Criteria:** A1 (Availability).

BIZRA achieves availability through decentralized redundancy and autonomous recovery:

| Layer | Control | RTO Target | Evidence |
|:---|:---|:---|:---|
| **Federation** | No single point of failure; Asabiyyah-weighted peer set | 0 (no downtime) | Federation topology snapshots |
| **Self-Healing** | Circuit breaker with tier degradation (T0 -> T1 -> T2 -> local-only) | < 30 seconds | Tier transition events in event log |
| **State Recovery** | Full reconstruction from any Merkle checkpoint | < 5 minutes | Recovery drill receipts |
| **Inference** | Tiered fallback: LM Studio -> Ollama -> Cloud API | < 2 seconds per hop | Inference tier selection log |

**Uptime Calculation:**

```
availability = (audit_period_seconds - unplanned_downtime_seconds) / audit_period_seconds
```

Target: >= 0.999 (99.9%) over audit period. Measured per-node; federation availability is
higher due to redundancy across sovereign nodes.

**Degraded Mode:** When a node loses federation connectivity, it continues operating in
sovereign-local mode. All actions produce valid receipts. Federation sync resumes automatically
on reconnection with conflict resolution via Asabiyyah-weighted consensus.

**Evidence:** Uptime metrics per node, tier transition events, recovery drill logs, federation
heartbeat records, Asabiyyah scores over audit period.

---

### FR-C32: Processing Integrity Controls

**SOC 2 Criteria:** PI1 (Processing Integrity).

Processing integrity in BIZRA is guaranteed by three reinforcing mechanisms:

**1. Fixed-Point Determinism.** All scoring arithmetic uses integer fixed-point
(`FP_PRECISION = 1,000,000`). No IEEE 754 floating-point. Byte-identical results across
ARM, x86, RISC-V. This eliminates an entire class of integrity failures: platform-dependent
rounding, NaN propagation, and denormal inconsistency.

**2. PoI_EMIT Receipt Chain.** Every transaction produces an ActionReceipt:
- `receipt_id` = BLAKE3 hash of receipt content (tamper detection)
- `oracle_signature` = Ed25519 signature by executing node (authenticity)
- `metadata_hash` = BLAKE3 hash of sensitive metadata (integrity without exposure)

**3. Append-Only Merkle Chain.** Every Event in the evidence ledger has:
- `event_id` = monotonically increasing sequence number (atomic, cross-process locked)
- `prev_hash` = BLAKE3 hash of previous event (chain integrity)
- `hash` = BLAKE3 hash of current event content (tamper detection)

**Integrity Verification:**

| Check | Frequency | Method |
|:---|:---|:---|
| Merkle chain continuity | Every event append | Verify `prev_hash` matches prior event `hash` |
| Receipt parity | Per audit tick | Receipt count matches event log entry count |
| Sequence gap detection | Per audit tick | No gaps in `event_id` sequence |
| FP determinism | Per release | 10,000 cross-platform parity tests |
| Signature verification | Every receipt read | Ed25519 verify on `oracle_signature` |

**Evidence:** Merkle root per audit period, receipt count vs. event count parity report,
sequence continuity proof, FP determinism test results, signature verification log.

---

### FR-C33: Confidentiality Controls

**SOC 2 Criteria:** C1 (Confidentiality).

**Constitutional Invariant I-6:** Every node is sovereign over its data. Data never leaves
the local node without explicit Ed25519-signed transfer authorization.

| Control | Mechanism | Evidence |
|:---|:---|:---|
| **Data Sovereignty** | I-6 enforcement: all egress requires Ed25519-signed consent | Data egress log with consent signatures |
| **Metadata Protection** | `metadata_hash` (BLAKE3) on every receipt | Hash present, raw metadata never in transit |
| **Key Recovery** | Shamir 3-of-5 secret sharing via trusted guardians | Recovery ceremony log (no key reconstruction without quorum) |
| **Key Revocation** | CRL published to BlockGraph, immutable audit trail | Revocation events in Merkle chain |
| **Data Classification** | Three tiers: PUBLIC (federation-shareable), INTERNAL (node-local), SOVEREIGN (encrypted-at-rest) | Classification tags on all data objects |

**Shamir Key Recovery Protocol:**

Five key shares distributed to trusted guardians. Any 3-of-5 can reconstruct the master key.
No single guardian holds sufficient information. Share distribution is logged as an immutable
event. Reconstruction requires multi-party ceremony with each guardian authenticating via their
own Ed25519 keypair.

**Federation Data Sharing:** When nodes share data for federation purposes (e.g., anonymized
PPB aggregates per FR-C25), only PUBLIC-classified aggregates leave the node. Raw receipts,
individual scores, and SOVEREIGN data never transit the network.

**Evidence:** Data egress authorization log, Shamir ceremony records, CRL events, data
classification audit, metadata_hash presence verification.

---

### FR-C34: Privacy Controls

**SOC 2 Criteria:** P1 (Privacy).

| Control | SOC 2 Privacy Principle | BIZRA Implementation |
|:---|:---|:---|
| **Notice** | Organization provides notice about privacy practices | DID scheme (`did:bizra:`) published; privacy policy embedded in node constitution |
| **Choice & Consent** | Individuals can opt out | Explicit consent records for every data processing operation; consent is an Ed25519-signed event |
| **Collection** | Only collect data for identified purposes | TeleScript Default Deny; data collection requires sovereign grant with stated purpose |
| **Use, Retention, Disposal** | Data used only as consented | Retention policies enforced by constitutional algorithm; disposal produces signed deletion receipt |
| **Access** | Individuals can access their data | Node sovereign has full read access; federation peers see only PUBLIC-classified aggregates |
| **Disclosure** | Data shared only as authorized | I-6 enforcement; no egress without Ed25519-signed consent (see FR-C33) |
| **Security** | Data protected from unauthorized access | Ed25519 + FATE Gate + TeleScript (see FR-C30) |
| **Quality** | Data is accurate and complete | PoI_EMIT receipt chain + FP determinism (see FR-C32) |
| **Monitoring & Enforcement** | Privacy compliance is monitored | Continuous audit tick checks I-6 every block |

**Local-First Processing:** All inference runs on the local node before any network call.
The inference tier cascade (LM Studio -> Ollama -> Cloud API) prefers local execution. Cloud
fallback is emergency-only and requires explicit opt-in per request.

**Pseudonymous Identity:** `did:bizra:<BLAKE3(Ed25519_pubkey)>` is the default identifier.
No PII is embedded in the DID. Mapping DID to real identity requires sovereign consent.

**Consent Lifecycle:** Consent is an immutable event in the Merkle chain. Consent grants and
withdrawals are both chained. A withdrawal mid-processing triggers immediate halt of the
affected operation and produces a signed halt receipt.

**Evidence:** Consent event chain, DID registry (pseudonymous), privacy impact assessments,
data processing purpose logs, deletion receipts.

---

### FR-C35: Continuous Monitoring

**SOC 2 Criteria:** CC4 (Monitoring Activities).

Traditional SOC 2 relies on annual audits with sample-based testing. BIZRA replaces sampling
with exhaustive real-time verification:

| Traditional SOC 2 | BIZRA Continuous Equivalent |
|:---|:---|
| Annual penetration test | FATE Gate runs formal Z3 verification on every action |
| Quarterly access review | TeleScript permission log verified every audit tick |
| Monthly availability report | Uptime computed from heartbeat stream, per-second granularity |
| Annual control assessment | Every receipt IS a control assessment -- 100% coverage |
| Sampling-based evidence | Exhaustive Merkle chain -- no sampling, no gaps |

**Audit Tick Frequency:** Every block (configurable; default every 60 seconds).

**Monitoring Dimensions:**

```
audit_tick:
    security:
        - fate_gate_active: boolean        # FATE Gate operational
        - auth_failures_since_last: int    # Challenge-response rejections
        - telescript_denials: int           # Unauthorized access attempts
    availability:
        - node_uptime_ms: int              # Since last restart
        - federation_peers: int            # Active peer count
        - current_tier: T0|T1|T2|LOCAL     # Inference tier
    integrity:
        - merkle_chain_valid: boolean      # Chain unbroken
        - receipt_event_parity: boolean    # Counts match
        - sequence_continuous: boolean     # No gaps
    confidentiality:
        - unauthorized_egress: int         # Must be 0
        - sovereign_data_encrypted: boolean
    privacy:
        - pending_consent_withdrawals: int # Must be 0 (all processed)
        - active_dids: int                 # Pseudonymous identity count
```

**Alerting:** Any audit tick failure triggers immediate escalation:
- IHSAN_STRICT (0.99) breach -> governance alert
- IHSAN_RUNTIME (1.0) breach -> node halt (safety-critical invariant)
- Unauthorized egress -> immediate block + Shura notification

**Evidence:** Complete audit tick history (append-only), alert log, escalation records.

---

### FR-C36: Audit Evidence Generation

**SOC 2 Criteria:** CC2 (Communication & Information), CC3 (Risk Assessment).

Quarterly SOC 2 report packs are generated automatically from the continuous monitoring
substrate:

**Report Pack Contents:**

| Section | Content | Source |
|:---|:---|:---|
| 1. Management Assertion | Node operator's assertion of control effectiveness | Template + node identity |
| 2. System Description | Architecture, data flows, trust boundaries | Generated from node configuration |
| 3. Security Controls | FATE Gate logs, auth stats, TeleScript audit | `assess_security_controls()` |
| 4. Availability Metrics | Uptime, tier transitions, recovery drills | Heartbeat + event log |
| 5. Processing Integrity | Merkle root, receipt parity, FP test results | `verify_processing_integrity()` |
| 6. Confidentiality | Egress log, Shamir ceremonies, CRL events | `audit_confidentiality()` |
| 7. Privacy | Consent chain, DID registry, PIA results | Consent events + DID log |
| 8. Exceptions | Any control failures during period | Audit tick failure log |
| 9. Remediation | Actions taken for each exception | Linked receipts for remediation actions |
| 10. Merkle Proof | Root hash covering all evidence in pack | MerkleTree over all receipt IDs in period |

**Report Signing:** The complete report pack is serialized, hashed with BLAKE3, and signed
with the node's Ed25519 keypair. The signature, merkle root, and receipt count form the
tamper-evident seal.

**Auditor Access:** External auditors receive the signed report pack. They can verify:
1. Signature authenticity (Ed25519 public key is the node's DID)
2. Merkle completeness (root matches independent computation over receipt IDs)
3. Receipt parity (count matches event log)
4. No gaps in sequence numbers
5. All exceptions have linked remediation

**Evidence:** Quarterly report packs (signed), auditor verification receipts, exception
remediation chains.

---

## 2. Edge Cases

**EC-C30: Key Compromise and Revocation.**
A node's Ed25519 private key is compromised. All receipts signed after compromise are suspect.
Resolution: (1) Shamir 3-of-5 key recovery ceremony generates new keypair. (2) CRL entry
published to BlockGraph with old public key and compromise timestamp. (3) Federation peers
reject receipts signed by compromised key after CRL publication. (4) All receipts between
estimated compromise time and CRL publication are flagged for manual review. (5) SOC 2 report
pack includes the incident as an exception with full remediation chain. The append-only event
log preserves the complete timeline -- nothing is deleted, only annotated.

**EC-C31: Merkle Chain Fork Detection.**
Two events claim the same `event_id` with different `prev_hash` values, indicating a fork.
Resolution: (1) Fork is detected at next `verify_chain()` call. (2) Both branches are
preserved in quarantine. (3) The branch with valid cross-process file lock ownership (POSIX
`fcntl.LOCK_EX`) is canonical. (4) The spurious branch is flagged as a potential integrity
incident. (5) Root-cause analysis determines whether the fork was caused by a bug (cross-process
lock failure), hardware fault (disk corruption), or adversarial action. (6) The SOC 2 report
includes the fork event, resolution, and corrective action.

**EC-C32: Federation Partition.**
Network partition isolates a subset of nodes. Each partition continues operating independently.
Resolution: (1) Nodes detect partition via missing heartbeats and transition to LOCAL tier.
(2) All actions continue producing valid local receipts. (3) On reconnection, Asabiyyah-weighted
consensus resolves conflicting state. (4) Availability metric accounts for partition: individual
node uptime is unaffected; federation-wide availability degrades proportionally. (5) Partition
duration and resolution are logged as availability events in the SOC 2 report.

**EC-C33: Unauthorized Data Egress Attempt.**
A TeleScript agent or external process attempts to exfiltrate SOVEREIGN-classified data.
Resolution: (1) TeleScript Default Deny blocks the request immediately (no sovereign grant
exists). (2) The denial is logged as a security event with full context: requesting identity,
target data, classification, timestamp. (3) If the requesting identity is a local agent, the
agent's permissions are reviewed and potentially revoked. (4) If the request originated from
a federation peer, the peer's Asabiyyah score is decreased. (5) The event is classified as a
security exception in the SOC 2 report with remediation documentation.

**EC-C34: Privacy Consent Withdrawal Mid-Processing.**
A sovereign withdraws consent for a data processing operation that is currently in progress.
Resolution: (1) Consent withdrawal is an Ed25519-signed immutable event, appended to the
Merkle chain. (2) The consent withdrawal event triggers an immediate interrupt on the
processing pipeline. (3) Partial results are discarded (not persisted). (4) A signed halt
receipt is produced documenting: what was processing, when withdrawal occurred, what data
was discarded. (5) The halt receipt chains into the Merkle log as evidence of compliance.
(6) If partial results were already shared with federation peers, a recall notification is
issued. Peers must acknowledge recall or face Asabiyyah penalty.

**EC-C35: Audit Period Boundary Transition.**
The audit period ends while events are still being processed. Resolution: (1) The period
boundary is defined by a specific `event_id`, not wall-clock time. (2) All events up to and
including the boundary `event_id` belong to the closing period. (3) The closing period's Merkle
root is computed and sealed. (4) The opening period begins with the next `event_id` and includes
a cross-reference to the prior period's Merkle root in its first event. (5) No events are
lost or double-counted.

---

## 3. Pseudocode

### 3.1 assess_security_controls

```
FUNCTION assess_security_controls(
    node: Node, audit_period: TimeRange
) -> SecurityAssessment:
    """Evaluate security control effectiveness over the audit period.
    Ref: core/auth/ (challenge-response), core/pci/gates.py (FATE),
         core/sovereign/ (TeleScript permissions)"""

    # Collect evidence streams
    receipts = EventLog.query(start=audit_period.start, end=audit_period.end,
                              event_type="action_receipt")
    auth_events = EventLog.query(start=audit_period.start, end=audit_period.end,
                                 event_type="auth_challenge")
    telescript_events = EventLog.query(start=audit_period.start, end=audit_period.end,
                                       event_type="telescript_permission")

    # CC6: FATE Gate uptime — every receipt must have fate_pass
    fate_total = len(receipts)
    fate_missing = len([r FOR r IN receipts IF r.fate_pass IS None])
    fate_passed = len([r FOR r IN receipts IF r.fate_pass == True])
    fate_rejected = len([r FOR r IN receipts IF r.fate_pass == False])
    fate_uptime = (fate_total - fate_missing) / fate_total IF fate_total > 0 ELSE 1.0

    IF fate_missing > 0:
        RAISE SecurityViolation("FATE Gate bypassed on {fate_missing} receipts — CRITICAL")

    # CC6: Authentication failure rate
    auth_total = len(auth_events)
    auth_failures = len([a FOR a IN auth_events IF a.result == "REJECTED"])
    auth_failure_rate = auth_failures / auth_total IF auth_total > 0 ELSE 0.0

    # CC7: TeleScript permission enforcement
    ts_denials = [t FOR t IN telescript_events IF t.decision == "DENY"]
    ts_grants = [t FOR t IN telescript_events IF t.decision == "GRANT"]
    ts_unresolved = [t FOR t IN ts_denials IF NOT t.has_resolution]

    # CC6: Signature integrity — verify every receipt signature
    sig_failures = []
    FOR receipt IN receipts:
        IF NOT Ed25519.verify(receipt.oracle_signature, receipt.receipt_id,
                              node.public_key):
            sig_failures.append(receipt.receipt_id)

    unauthorized_transitions = len(sig_failures)

    # CC8: Change management — check for receipts outside FATE-approved paths
    unapproved_changes = [r FOR r IN receipts
                          IF r.action_type == "CODE_CHANGE" AND r.fate_pass != True]

    # Scoring against thresholds
    security_pass = (
        fate_uptime == 1.0
        AND auth_failure_rate < 0.001
        AND unauthorized_transitions == 0
        AND len(ts_unresolved) == 0
        AND len(unapproved_changes) == 0
    )

    RETURN SecurityAssessment(
        audit_period=audit_period,
        fate_uptime=fate_uptime,
        fate_total=fate_total, fate_passed=fate_passed, fate_rejected=fate_rejected,
        auth_total=auth_total, auth_failures=auth_failures,
        auth_failure_rate=auth_failure_rate,
        telescript_grants=len(ts_grants), telescript_denials=len(ts_denials),
        telescript_unresolved=len(ts_unresolved),
        signature_failures=sig_failures,
        unauthorized_transitions=unauthorized_transitions,
        unapproved_changes=len(unapproved_changes),
        security_pass=security_pass,
        timestamp=now_ms()
    )
```

### 3.2 verify_processing_integrity

```
FUNCTION verify_processing_integrity(
    event_log: EventLog, receipts: list[ActionReceipt]
) -> IntegrityAssessment:
    """Verify Merkle chain continuity, receipt parity, and FP determinism.
    Ref: core/proof_engine/evidence_ledger.py (Merkle chain),
         core/constitutional/types.py (ActionReceipt, Event),
         core/constitutional/fixed_point.py (FP arithmetic)"""

    # Step 1: Merkle chain continuity
    events = event_log.get_all_events()
    chain_breaks = []
    prev_hash = None

    FOR i, event IN enumerate(events):
        # Verify event self-hash
        computed_hash = BLAKE3.hash(event.content)
        IF computed_hash != event.hash:
            chain_breaks.append(ChainBreak(
                event_id=event.event_id, type="SELF_HASH_MISMATCH",
                expected=event.hash, computed=computed_hash))

        # Verify chain linkage
        IF prev_hash IS NOT None AND event.prev_hash != prev_hash:
            chain_breaks.append(ChainBreak(
                event_id=event.event_id, type="PREV_HASH_MISMATCH",
                expected=prev_hash, actual=event.prev_hash))

        prev_hash = event.hash

    chain_valid = len(chain_breaks) == 0

    # Step 2: Sequence continuity — no gaps in event_id
    sequence_gaps = []
    FOR i IN range(1, len(events)):
        IF events[i].event_id != events[i-1].event_id + 1:
            sequence_gaps.append(SequenceGap(
                expected=events[i-1].event_id + 1,
                actual=events[i].event_id))

    sequence_continuous = len(sequence_gaps) == 0

    # Step 3: Receipt-to-event parity
    receipt_ids_from_log = set(
        e.receipt_id FOR e IN events IF e.event_type == "action_receipt"
    )
    receipt_ids_from_receipts = set(r.receipt_id FOR r IN receipts)

    missing_in_log = receipt_ids_from_receipts - receipt_ids_from_log
    missing_in_receipts = receipt_ids_from_log - receipt_ids_from_receipts
    parity_valid = len(missing_in_log) == 0 AND len(missing_in_receipts) == 0

    # Step 4: Receipt self-integrity — verify receipt_id matches content hash
    tampered_receipts = []
    FOR receipt IN receipts:
        computed_id = BLAKE3.hash(receipt.content)
        IF computed_id != receipt.receipt_id:
            tampered_receipts.append(receipt.receipt_id)

    # Step 5: Fixed-point determinism spot-check
    fp_test_results = run_fp_parity_tests(count=10000)
    fp_determinism_pass = fp_test_results.failures == 0

    # Step 6: Build Merkle root over all receipt IDs for the period
    merkle_root = MerkleTree.build(
        sorted(receipt_ids_from_receipts)
    ).root

    integrity_pass = (
        chain_valid
        AND sequence_continuous
        AND parity_valid
        AND len(tampered_receipts) == 0
        AND fp_determinism_pass
    )

    RETURN IntegrityAssessment(
        chain_valid=chain_valid, chain_breaks=chain_breaks,
        sequence_continuous=sequence_continuous, sequence_gaps=sequence_gaps,
        parity_valid=parity_valid,
        missing_in_log=missing_in_log, missing_in_receipts=missing_in_receipts,
        tampered_receipts=tampered_receipts,
        fp_determinism_pass=fp_determinism_pass,
        fp_test_count=fp_test_results.total, fp_test_failures=fp_test_results.failures,
        merkle_root=merkle_root,
        total_events=len(events), total_receipts=len(receipts),
        integrity_pass=integrity_pass,
        timestamp=now_ms()
    )
```

### 3.3 audit_confidentiality

```
FUNCTION audit_confidentiality(
    data_egress_log: list[EgressEvent],
    consent_records: list[ConsentEvent],
    shamir_ceremonies: list[ShamirCeremony],
    crl_events: list[CRLEvent]
) -> ConfidentialityAssessment:
    """Verify all data egress has Ed25519-signed consent and sovereignty is maintained.
    Ref: core/auth/ (Ed25519 signatures),
         core/constitutional/types.py (I-6 sovereignty invariant)"""

    # Step 1: Build consent index — map data_id to active consent
    consent_index = {}
    FOR consent IN sorted(consent_records, key=lambda c: c.timestamp):
        IF consent.action == "GRANT":
            consent_index[consent.data_id] = consent
        ELIF consent.action == "WITHDRAW":
            consent_index.pop(consent.data_id, None)

    # Step 2: Verify every egress event has valid consent
    unauthorized_egress = []
    FOR egress IN data_egress_log:
        consent = consent_index.get(egress.data_id)

        IF consent IS None:
            unauthorized_egress.append(EgressViolation(
                egress_id=egress.id, data_id=egress.data_id,
                reason="NO_CONSENT_RECORD"))
            CONTINUE

        # Consent must predate egress
        IF consent.timestamp > egress.timestamp:
            unauthorized_egress.append(EgressViolation(
                egress_id=egress.id, data_id=egress.data_id,
                reason="CONSENT_POSTDATES_EGRESS"))
            CONTINUE

        # Consent signature must be valid
        IF NOT Ed25519.verify(consent.signature, consent.content, consent.sovereign_pubkey):
            unauthorized_egress.append(EgressViolation(
                egress_id=egress.id, data_id=egress.data_id,
                reason="INVALID_CONSENT_SIGNATURE"))
            CONTINUE

        # Data classification must allow egress
        IF egress.data_classification == "SOVEREIGN":
            unauthorized_egress.append(EgressViolation(
                egress_id=egress.id, data_id=egress.data_id,
                reason="SOVEREIGN_DATA_EGRESS_BLOCKED"))

    # Step 3: Verify Shamir ceremonies
    shamir_violations = []
    FOR ceremony IN shamir_ceremonies:
        IF ceremony.guardian_count < 3:
            shamir_violations.append(ShamirViolation(
                ceremony_id=ceremony.id,
                reason="INSUFFICIENT_GUARDIANS",
                guardian_count=ceremony.guardian_count))
        FOR guardian IN ceremony.guardians:
            IF NOT Ed25519.verify(guardian.signature, ceremony.id, guardian.pubkey):
                shamir_violations.append(ShamirViolation(
                    ceremony_id=ceremony.id,
                    reason="INVALID_GUARDIAN_SIGNATURE",
                    guardian_id=guardian.id))

    # Step 4: Verify CRL integrity — revocations are in Merkle chain
    crl_unchained = [c FOR c IN crl_events IF NOT c.is_in_merkle_chain]

    # Step 5: Data classification coverage — all data objects classified
    unclassified_data = [e FOR e IN data_egress_log IF e.data_classification IS None]

    confidentiality_pass = (
        len(unauthorized_egress) == 0
        AND len(shamir_violations) == 0
        AND len(crl_unchained) == 0
        AND len(unclassified_data) == 0
    )

    RETURN ConfidentialityAssessment(
        total_egress_events=len(data_egress_log),
        authorized_egress=len(data_egress_log) - len(unauthorized_egress),
        unauthorized_egress=unauthorized_egress,
        shamir_ceremonies=len(shamir_ceremonies),
        shamir_violations=shamir_violations,
        crl_events=len(crl_events), crl_unchained=crl_unchained,
        unclassified_data=len(unclassified_data),
        confidentiality_pass=confidentiality_pass,
        timestamp=now_ms()
    )
```

### 3.4 generate_soc2_report

```
FUNCTION generate_soc2_report(
    security: SecurityAssessment,
    availability: AvailabilityAssessment,
    integrity: IntegrityAssessment,
    confidentiality: ConfidentialityAssessment,
    privacy: PrivacyAssessment,
    period: TimeRange
) -> SOC2Report:
    """Generate quarterly SOC 2 Type II report pack with signed Merkle seal.
    Ref: core/proof_engine/evidence_ledger.py (Merkle tree),
         core/integration/constants.py (IHSAN_STRICT, IHSAN_RUNTIME)"""

    # Section 1: Management assertion
    assertion = ManagementAssertion(
        node_did=node.did,
        period=period,
        statement="Controls were suitably designed and operating effectively "
                  "throughout the examination period.",
        assertion_pass=(security.security_pass AND availability.availability_pass
                        AND integrity.integrity_pass AND confidentiality.confidentiality_pass
                        AND privacy.privacy_pass)
    )

    # Section 2: System description (auto-generated from node config)
    system_desc = SystemDescription(
        architecture="Sovereign Operating System — BIZRA DDAGI-OS",
        trust_boundaries=node.config.trust_boundaries,
        data_flows=node.config.data_flow_diagram,
        components=["FATE Gate", "Evidence Ledger", "TeleScript Runtime",
                     "Federation Transport", "Inference Tier Cascade",
                     "Constitutional Kernel", "PCI Envelope"],
        infrastructure="Local-first; Ed25519 identity; BLAKE3 hashing; "
                       "Fixed-point arithmetic (FP_PRECISION=1000000)"
    )

    # Section 3-7: Trust service category results (already computed)
    # Collect all exceptions across categories
    exceptions = []

    IF NOT security.security_pass:
        exceptions.append(Exception(
            category="SECURITY", severity="CRITICAL",
            detail=f"FATE uptime: {security.fate_uptime}, "
                   f"Auth failures: {security.auth_failures}, "
                   f"Unauthorized transitions: {security.unauthorized_transitions}"))

    IF NOT availability.availability_pass:
        exceptions.append(Exception(
            category="AVAILABILITY", severity="HIGH",
            detail=f"Uptime: {availability.uptime_ratio}, "
                   f"Target: 0.999"))

    IF NOT integrity.integrity_pass:
        exceptions.append(Exception(
            category="PROCESSING_INTEGRITY", severity="CRITICAL",
            detail=f"Chain breaks: {len(integrity.chain_breaks)}, "
                   f"Sequence gaps: {len(integrity.sequence_gaps)}, "
                   f"Tampered: {len(integrity.tampered_receipts)}"))

    IF NOT confidentiality.confidentiality_pass:
        exceptions.append(Exception(
            category="CONFIDENTIALITY", severity="CRITICAL",
            detail=f"Unauthorized egress: {len(confidentiality.unauthorized_egress)}, "
                   f"Shamir violations: {len(confidentiality.shamir_violations)}"))

    IF NOT privacy.privacy_pass:
        exceptions.append(Exception(
            category="PRIVACY", severity="HIGH",
            detail=f"Pending withdrawals: {privacy.pending_consent_withdrawals}, "
                   f"Unlinked processing: {privacy.unlinked_processing_events}"))

    # Section 8-9: Link exceptions to remediation receipts
    FOR exception IN exceptions:
        remediation = find_remediation_receipts(exception, period)
        exception.remediation_chain = remediation
        exception.remediated = len(remediation) > 0

    # Section 10: Merkle proof over all evidence
    all_receipt_ids = collect_receipt_ids(period)
    merkle_tree = MerkleTree.build(sorted(all_receipt_ids))

    # Compute overall Ihsan for the audit period
    period_ihsan = compute_period_ihsan(all_receipt_ids)
    ihsan_gate_pass = fp_float(period_ihsan) >= IHSAN_STRICT  # 0.99 for SOC 2

    report = SOC2Report(
        standard="SOC2_Type_II",
        period=period,
        assertion=assertion,
        system_description=system_desc,
        security=security,
        availability=availability,
        integrity=integrity,
        confidentiality=confidentiality,
        privacy=privacy,
        exceptions=exceptions,
        exception_count=len(exceptions),
        all_remediated=all(e.remediated FOR e IN exceptions),
        merkle_root=merkle_tree.root,
        total_receipts=len(all_receipt_ids),
        period_ihsan=period_ihsan,
        ihsan_gate_pass=ihsan_gate_pass,
        generated_at=now_ms()
    )

    # Sign the complete report
    report_hash = BLAKE3.hash(report.serialize())
    report.signature = node_keypair.sign(report_hash)
    report.report_hash = report_hash

    RETURN report
```

### 3.5 assess_availability

```
FUNCTION assess_availability(
    node: Node, audit_period: TimeRange, heartbeats: list[Heartbeat]
) -> AvailabilityAssessment:
    """Compute availability metrics and verify recovery objectives.
    Ref: core/federation/ (heartbeat), core/sovereign/ (tier degradation)"""

    period_seconds = (audit_period.end - audit_period.start) / 1000

    # Step 1: Compute uptime from heartbeat gaps
    sorted_beats = sorted(heartbeats, key=lambda h: h.timestamp)
    downtime_seconds = 0
    max_gap_seconds = 0
    downtime_events = []

    FOR i IN range(1, len(sorted_beats)):
        gap = (sorted_beats[i].timestamp - sorted_beats[i-1].timestamp) / 1000
        expected_interval = node.config.heartbeat_interval_seconds

        IF gap > expected_interval * 3:  # 3x expected = unplanned downtime
            downtime = gap - expected_interval
            downtime_seconds += downtime
            max_gap_seconds = max(max_gap_seconds, downtime)
            downtime_events.append(DowntimeEvent(
                start=sorted_beats[i-1].timestamp,
                end=sorted_beats[i].timestamp,
                duration_seconds=downtime))

    uptime_ratio = (period_seconds - downtime_seconds) / period_seconds
        IF period_seconds > 0 ELSE 1.0

    # Step 2: Verify tier transitions were graceful
    tier_events = EventLog.query(start=audit_period.start, end=audit_period.end,
                                  event_type="tier_transition")
    ungraceful_transitions = [t FOR t IN tier_events IF t.transition_type == "CRASH"]

    # Step 3: Verify recovery drills met RTO
    recovery_drills = EventLog.query(start=audit_period.start, end=audit_period.end,
                                      event_type="recovery_drill")
    rto_failures = [d FOR d IN recovery_drills
                    IF d.recovery_time_seconds > node.config.rto_target_seconds]

    # Step 4: Federation peer count over time
    peer_counts = [h.federation_peers FOR h IN sorted_beats IF h.federation_peers IS NOT None]
    min_peers = min(peer_counts) IF peer_counts ELSE 0
    avg_peers = sum(peer_counts) / len(peer_counts) IF peer_counts ELSE 0

    availability_pass = (
        uptime_ratio >= 0.999
        AND len(ungraceful_transitions) == 0
        AND len(rto_failures) == 0
    )

    RETURN AvailabilityAssessment(
        audit_period=audit_period,
        uptime_ratio=uptime_ratio,
        downtime_seconds=downtime_seconds,
        downtime_events=downtime_events,
        max_gap_seconds=max_gap_seconds,
        tier_transitions=len(tier_events),
        ungraceful_transitions=len(ungraceful_transitions),
        recovery_drills=len(recovery_drills),
        rto_failures=len(rto_failures),
        min_federation_peers=min_peers,
        avg_federation_peers=avg_peers,
        availability_pass=availability_pass,
        timestamp=now_ms()
    )
```

### 3.6 assess_privacy

```
FUNCTION assess_privacy(
    consent_events: list[ConsentEvent],
    processing_log: list[ProcessingEvent],
    did_registry: DIDRegistry,
    audit_period: TimeRange
) -> PrivacyAssessment:
    """Verify privacy controls: consent lifecycle, local-first, pseudonymous identity.
    Ref: core/auth/ (DID scheme), core/sovereign/ (local-first inference)"""

    # Step 1: Build active consent set at each point in time
    consent_state = {}  # data_id -> ConsentEvent (latest)
    FOR event IN sorted(consent_events, key=lambda e: e.timestamp):
        IF event.action == "GRANT":
            consent_state[event.data_id] = event
        ELIF event.action == "WITHDRAW":
            consent_state.pop(event.data_id, None)

    # Step 2: Verify every processing event has valid consent
    unlinked_processing = []
    FOR proc IN processing_log:
        IF proc.data_id NOT IN consent_state:
            unlinked_processing.append(proc)

    # Step 3: Check consent withdrawal handling
    withdrawals = [e FOR e IN consent_events IF e.action == "WITHDRAW"]
    pending_withdrawals = []
    FOR withdrawal IN withdrawals:
        # Find any processing that continued after withdrawal
        post_withdrawal = [p FOR p IN processing_log
                           IF p.data_id == withdrawal.data_id
                           AND p.timestamp > withdrawal.timestamp]
        # Halt receipts prove processing stopped
        halt_receipts = [p FOR p IN post_withdrawal IF p.event_type == "HALT_RECEIPT"]
        continued_processing = [p FOR p IN post_withdrawal IF p.event_type != "HALT_RECEIPT"]

        IF len(continued_processing) > 0:
            pending_withdrawals.append(WithdrawalViolation(
                withdrawal=withdrawal,
                continued_events=continued_processing))

    # Step 4: Verify DID pseudonymity — no PII in DID
    pii_leaks = []
    FOR did_entry IN did_registry.all():
        IF did_entry.contains_pii():
            pii_leaks.append(did_entry.did)

    # Step 5: Verify local-first inference preference
    inference_events = EventLog.query(start=audit_period.start, end=audit_period.end,
                                       event_type="inference_tier_selection")
    cloud_without_consent = [e FOR e IN inference_events
                             IF e.tier == "CLOUD" AND NOT e.explicit_opt_in]

    privacy_pass = (
        len(unlinked_processing) == 0
        AND len(pending_withdrawals) == 0
        AND len(pii_leaks) == 0
        AND len(cloud_without_consent) == 0
    )

    RETURN PrivacyAssessment(
        total_consent_events=len(consent_events),
        active_consents=len(consent_state),
        withdrawals=len(withdrawals),
        pending_consent_withdrawals=len(pending_withdrawals),
        withdrawal_violations=pending_withdrawals,
        unlinked_processing_events=len(unlinked_processing),
        did_count=did_registry.count(),
        pii_leaks=pii_leaks,
        cloud_without_consent=len(cloud_without_consent),
        privacy_pass=privacy_pass,
        timestamp=now_ms()
    )
```

---

## 4. TDD Anchors

```
TEST security_detects_fate_gate_bypass:
    """Any receipt without fate_pass must trigger CRITICAL violation."""
    receipts = generate_test_receipts(count=100, fate_pass=True)
    receipts[50].fate_pass = None  # Simulate bypass
    EXPECT_RAISE assess_security_controls(test_node, last_90_days)
    # SecurityViolation: "FATE Gate bypassed on 1 receipts"

TEST security_passes_with_zero_violations:
    """Clean audit period produces security_pass=True."""
    receipts = generate_test_receipts(count=1000, fate_pass=True)
    auth_events = generate_auth_events(total=500, failures=0)
    ts_events = generate_telescript_events(grants=200, denials=5, all_resolved=True)
    result = assess_security_controls(test_node, last_90_days)
    ASSERT result.security_pass == True
    ASSERT result.fate_uptime == 1.0
    ASSERT result.unauthorized_transitions == 0

TEST security_fails_on_high_auth_failure_rate:
    """Auth failure rate >= 0.1% fails the security assessment."""
    auth_events = generate_auth_events(total=1000, failures=2)  # 0.2%
    result = assess_security_controls(test_node, last_90_days)
    ASSERT result.security_pass == False
    ASSERT result.auth_failure_rate >= 0.001

TEST integrity_detects_merkle_chain_break:
    """A single prev_hash mismatch fails the integrity assessment."""
    events = generate_valid_event_chain(count=500)
    events[250].prev_hash = BLAKE3.hash(b"tampered")  # Break chain
    receipts = extract_receipts(events)
    result = verify_processing_integrity(EventLog(events), receipts)
    ASSERT result.chain_valid == False
    ASSERT len(result.chain_breaks) == 1
    ASSERT result.chain_breaks[0].event_id == 250
    ASSERT result.integrity_pass == False

TEST integrity_detects_sequence_gap:
    """Missing event_id in sequence is detected."""
    events = generate_valid_event_chain(count=100)
    del events[50]  # Create gap: ..., 49, 51, ...
    result = verify_processing_integrity(EventLog(events), extract_receipts(events))
    ASSERT result.sequence_continuous == False
    ASSERT len(result.sequence_gaps) >= 1

TEST integrity_passes_clean_chain:
    """Valid chain with matching receipts produces integrity_pass=True."""
    events = generate_valid_event_chain(count=1000)
    receipts = extract_receipts(events)
    result = verify_processing_integrity(EventLog(events), receipts)
    ASSERT result.chain_valid == True
    ASSERT result.sequence_continuous == True
    ASSERT result.parity_valid == True
    ASSERT result.integrity_pass == True
    ASSERT len(result.merkle_root) == 32  # BLAKE3 digest

TEST confidentiality_detects_unauthorized_egress:
    """Egress without matching consent record is flagged."""
    consent = [create_consent("GRANT", data_id="d-001", ts=1000)]
    egress = [create_egress(data_id="d-001", ts=2000),  # authorized
              create_egress(data_id="d-002", ts=3000)]   # unauthorized — no consent
    result = audit_confidentiality(egress, consent, [], [])
    ASSERT result.confidentiality_pass == False
    ASSERT len(result.unauthorized_egress) == 1
    ASSERT result.unauthorized_egress[0].reason == "NO_CONSENT_RECORD"

TEST confidentiality_blocks_sovereign_data_egress:
    """SOVEREIGN-classified data egress is blocked even with consent."""
    consent = [create_consent("GRANT", data_id="d-001", ts=1000)]
    egress = [create_egress(data_id="d-001", ts=2000, classification="SOVEREIGN")]
    result = audit_confidentiality(egress, consent, [], [])
    ASSERT result.confidentiality_pass == False
    ASSERT result.unauthorized_egress[0].reason == "SOVEREIGN_DATA_EGRESS_BLOCKED"

TEST confidentiality_validates_shamir_quorum:
    """Shamir ceremony with fewer than 3 guardians is flagged."""
    ceremony = create_shamir_ceremony(guardian_count=2)
    result = audit_confidentiality([], [], [ceremony], [])
    ASSERT result.confidentiality_pass == False
    ASSERT len(result.shamir_violations) >= 1
    ASSERT result.shamir_violations[0].reason == "INSUFFICIENT_GUARDIANS"

TEST privacy_detects_consent_withdrawal_violation:
    """Processing that continues after consent withdrawal is flagged."""
    consent = [create_consent("GRANT", data_id="d-001", ts=1000),
               create_consent("WITHDRAW", data_id="d-001", ts=5000)]
    processing = [create_processing(data_id="d-001", ts=3000),   # before withdrawal — OK
                  create_processing(data_id="d-001", ts=6000)]    # after withdrawal — VIOLATION
    result = assess_privacy(consent, processing, empty_did_registry(), last_90_days)
    ASSERT result.privacy_pass == False
    ASSERT len(result.withdrawal_violations) == 1

TEST privacy_passes_with_proper_consent_lifecycle:
    """Full consent lifecycle with halt receipt produces privacy_pass=True."""
    consent = [create_consent("GRANT", data_id="d-001", ts=1000),
               create_consent("WITHDRAW", data_id="d-001", ts=5000)]
    processing = [create_processing(data_id="d-001", ts=3000),
                  create_halt_receipt(data_id="d-001", ts=5001)]
    result = assess_privacy(consent, processing, clean_did_registry(), last_90_days)
    ASSERT result.privacy_pass == True
    ASSERT result.pending_consent_withdrawals == 0

TEST soc2_report_is_signed_and_sealed:
    """Complete report has BLAKE3 hash, Ed25519 signature, and Merkle root."""
    report = generate_soc2_report(
        clean_security(), clean_availability(), clean_integrity(),
        clean_confidentiality(), clean_privacy(), last_90_days)
    ASSERT report.report_hash IS NOT None AND len(report.report_hash) == 32
    ASSERT report.signature IS NOT None
    ASSERT Ed25519.verify(report.signature, report.report_hash, node_pubkey)
    ASSERT report.merkle_root IS NOT None AND len(report.merkle_root) == 32
    ASSERT report.ihsan_gate_pass == True  # Clean data exceeds IHSAN_STRICT

TEST soc2_report_includes_exceptions_with_remediation:
    """Exceptions are linked to remediation receipt chains."""
    security = failing_security_assessment()  # Has auth failures
    report = generate_soc2_report(
        security, clean_availability(), clean_integrity(),
        clean_confidentiality(), clean_privacy(), last_90_days)
    ASSERT report.exception_count >= 1
    ASSERT report.assertion.assertion_pass == False
    FOR exception IN report.exceptions:
        ASSERT exception.remediation_chain IS NOT None

TEST availability_detects_downtime_from_heartbeat_gap:
    """Heartbeat gap > 3x interval registers as downtime."""
    heartbeats = generate_heartbeats(period_hours=24, interval_seconds=60)
    # Insert 10-minute gap at hour 12
    heartbeats = remove_heartbeats_in_range(heartbeats, hour=12, duration_minutes=10)
    result = assess_availability(test_node, last_24_hours, heartbeats)
    ASSERT result.downtime_seconds > 0
    ASSERT len(result.downtime_events) >= 1

TEST availability_passes_with_consistent_heartbeats:
    """No gaps produces availability_pass=True with 100% uptime."""
    heartbeats = generate_heartbeats(period_hours=24*90, interval_seconds=60)
    result = assess_availability(test_node, last_90_days, heartbeats)
    ASSERT result.uptime_ratio >= 0.999
    ASSERT result.availability_pass == True
```

---

## 5. Cross-References

### Codebase Modules

| Module | SOC 2 Trust Category | Relevance |
|:---|:---|:---|
| `core/auth/` | Security (CC6) | Ed25519 challenge-response, middleware authentication |
| `core/auth/user_store.py` | Security (CC6) | User identity management, credential storage |
| `core/pci/gates.py` | Security (CC7) | FATE Gate — pre-execution veto (Formal, Alignment, Testing, Ethical) |
| `core/proof_engine/evidence_ledger.py` | Processing Integrity (PI1) | Append-only Merkle chain, atomic sequence numbers, cross-process `fcntl.LOCK_EX` |
| `core/constitutional/types.py` | Processing Integrity (PI1) | ActionReceipt (`receipt_id` BLAKE3, `oracle_signature` Ed25519, `metadata_hash` BLAKE3), Event chain |
| `core/constitutional/fixed_point.py` | Processing Integrity (PI1) | FP_PRECISION=1,000,000 — byte-identical cross-platform arithmetic |
| `core/constitutional/algorithms.py` | All Categories | A1 (Ihsan), A5 (FATE), A6 (Merkle), A7 (SNR), A8 (Crown) |
| `core/integration/constants.py` | All Categories | IHSAN_STRICT=0.99, IHSAN_RUNTIME=1.0, ADL_GINI_THRESHOLD=0.35 |
| `core/sovereign/` | Confidentiality (C1), Privacy (P1) | TeleScript runtime, Default Deny, sovereignty enforcement |
| `core/federation/` | Availability (A1) | Peer discovery, Asabiyyah-weighted consensus, heartbeat protocol |
| `core/governance/` | Security (CC8), Privacy (P1) | Shura voting for escalated security/privacy decisions |
| `core/living_memory/` | Privacy (P1) | Local-first retrieval, consent-aware data access |
| `bizra-agent/src/omni_kernel.rs` | Availability (A1) | Self-healing loop, tier degradation, circuit breaker |
| `bizra-core/` | Security (CC6) | Ed25519 identity, BLAKE3 hashing, constitutional invariants |
| `fate-binding/` | Security (CC7) | Z3 formal verification, Dilithium post-quantum signatures |
| `bizra-federation/` | Availability (A1) | Gossip protocol, signed messages, partition detection |

### SOC 2 Trust Service Criteria Mapping

| SOC 2 Criteria | BIZRA Functional Requirement | Primary Evidence |
|:---|:---|:---|
| CC1 (Control Environment) | FR-C30 | Constitutional invariants I-1 through I-7 |
| CC2 (Communication) | FR-C36 | Quarterly report packs, signed and Merkle-sealed |
| CC3 (Risk Assessment) | FR-C36 | Exception log with severity classification |
| CC4 (Monitoring) | FR-C35 | Continuous audit tick (every 60s), exhaustive coverage |
| CC5 (Control Activities — Monitoring) | FR-C35 | Real-time alerting on threshold breach |
| CC6 (Logical/Physical Access) | FR-C30 | Ed25519 identity, FATE Gate, TeleScript Default Deny |
| CC7 (System Operations) | FR-C30 | PCI envelopes, FATE Gate pre-execution |
| CC8 (Change Management) | FR-C30 | FATE Gate on code changes, signed receipt chain |
| A1 (Availability) | FR-C31 | Uptime metrics, tier transitions, recovery drills |
| PI1 (Processing Integrity) | FR-C32 | Merkle chain, receipt parity, FP determinism |
| C1 (Confidentiality) | FR-C33 | Egress authorization log, Shamir ceremonies, CRL |
| P1 (Privacy) | FR-C34 | Consent chain, DID registry, halt receipts |

### Sibling Specs

- Phase 00 (Framework Overview) -- Unified Evidence Model, cross-standard invariants I-1 through I-7
- Phase 01 (ISO 25010) -- FR-C10 Functional Suitability shares FATE Gate with FR-C30 Security
- Phase 02 (CMMI Level 5) -- FR-C26 Improvement Event Trail uses same Merkle chain as FR-C32
- Phase 04 (ISO 9001) -- QMS Process Approach shares governance and Shura infrastructure

### Constitutional Thresholds (from `core/integration/constants.py`)

| Constant | Value | SOC 2 Usage |
|:---|:---|:---|
| IHSAN_STRICT | 0.99 | Minimum Ihsan for SOC 2 audit period pass |
| IHSAN_RUNTIME | 1.0 | Safety-critical invariant — breach triggers node halt |
| IHSAN_PRODUCTION | 0.95 | Per-receipt quality floor |
| FP_PRECISION | 1,000,000 | Processing Integrity determinism guarantee |
| ADL_GINI_THRESHOLD | 0.35 | Invariant I-4 checked in continuous audit tick |
| GINI_HEALTHY | 0.30 | Federation health indicator |
