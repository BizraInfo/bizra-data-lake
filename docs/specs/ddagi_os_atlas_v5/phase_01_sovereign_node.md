# Phase 01 — Sovereign Node: Identity, Self-Harness, Living Memory

> Source: Atlas v5.0 — Diagrams D0 (Node subgraph), D12 (Genesis), D20 (Identity)
> Status: SPECIFICATION SEALED | SNR: 0.93

---

## 1. Functional Requirements

### FR-010: Node Genesis Ceremony
1. Human accepts Constitution (Al-Risalah Covenant).
2. Mint Ed25519 keypair from hardware RNG + user entropy.
3. Derive DID: `did:bizra:base58(pubkey)`.
4. Store in encrypted local keystore.
5. Assign unique NodeID bound to DID.
6. Spawn PAT-7 (personal agents) + register with SAT-49.
7. Establish PAT-SAT negotiation channel (handshake).

### FR-011: Resource Pledge (6 Categories)
Every node pledges resources upon genesis:
- **Compute**: CPU/GPU cycles
- **Storage**: Disk space
- **Bandwidth**: Network capacity
- **Knowledge**: Expertise and data
- **Attention**: Human feedback and verification
- **Creativity**: Original content and ideas

### FR-012: Sovereign Identity Lifecycle
- **Generation**: Local-only, Ed25519, `did:bizra:` scheme.
- **Credentials**: Self-attestation → Node binding → Capability tokens → Expiry rotation.
- **Authentication**: Challenge-response with Ed25519 signatures.
- **Recovery**: Shamir 3-of-5 secret sharing via trusted guardians.
- **Cross-Device**: Delegate keys with limited capability tokens.
- **Revocation**: CRL on BlockGraph + immutable audit trail.

### FR-013: Constitutional Self-Harness (Always On)
Five-stage pipeline, every decision passes through in order:
1. **FATE Gate** — Pre-execution veto (Z3 + Alignment + Testing + Ethical)
2. **Ihsan Wall** — Adaptive quality floor (0.95 prod, 0.90 CI, 0.99 strict)
3. **Gini Attractor** — Economic homeostasis (≤ 0.35 hard gate)
4. **Reflex Pruner** — Quality-weighted cache eviction
5. **Continuous Auditor** — Auto-remediate violations

### FR-014: Living Memory (Brain That Remembers You)
Three memory systems, interconnected:
- **Episodic**: Last N receipts — what happened, when, outcomes.
- **Semantic**: User model — preferences, expertise, style patterns.
- **Procedural**: Compiled reflexes — myelinated System-1 skills.

---

## 2. Edge Cases

- **EC-001**: Keypair generation on low-entropy hardware → require minimum 256 bits of entropy or block.
- **EC-002**: Guardian unreachable during recovery → fallback to time-locked self-recovery after 30 days.
- **EC-003**: Delegate key compromised → immediate revocation propagated via gossip, guardian alert.
- **EC-004**: Node pledge exceeds available resources → cap at 90% of measured capacity, audit monthly.
- **EC-005**: Constitutional harness disabled by code mutation → Continuous Auditor detects + halts node.

---

## 3. Pseudocode: Node Genesis

```
FUNCTION genesis_ceremony(human, constitution):
    # Step 1: Covenant
    IF NOT human.accepts(constitution.AL_RISALAH):
        RETURN Error("Constitution must be accepted")

    # Step 2: Identity Minting
    entropy = collect_entropy(hardware_rng, human_input)
    ASSERT entropy.bits >= 256

    keypair = Ed25519.generate(entropy)
    did     = "did:bizra:" + base58_encode(keypair.public_key)
    keystore = EncryptedKeystore.create(keypair, human.passphrase)

    # Step 3: Node ID
    node_id = BLAKE3.hash(keypair.public_key + timestamp_ns())

    # Step 4: Shamir Recovery Setup
    shares = shamir_split(keypair.private_key, threshold=3, total=5)
    FOR guardian IN human.select_guardians(5):
        secure_deliver(guardian, shares.next())

    # Step 5: Dual-Agentic Bifurcation
    pat7 = spawn_pat7(node_id, keypair)
    sat_registration = SAT49.register(node_id, keypair.public_key)
    channel = PAT_SAT_Handshake(pat7, sat_registration)

    # Step 6: Resource Pledge
    pledge = ResourcePledge(
        compute    = measure_available_compute() * 0.9,
        storage    = measure_available_storage() * 0.9,
        bandwidth  = measure_available_bandwidth() * 0.9,
        knowledge  = human.declared_expertise,
        attention  = DEFAULT_ATTENTION_BUDGET,
        creativity = DEFAULT_CREATIVITY_BUDGET
    )

    # Step 7: Boot constitutional harness
    harness = ConstitutionalHarness.boot(constitution)

    RETURN SovereignNode(did, node_id, keypair, keystore,
                         pat7, channel, pledge, harness)
```

## 4. Pseudocode: Identity Authentication

```
FUNCTION authenticate(node, service):
    # Challenge-Response Protocol
    nonce = service.send_challenge(random_bytes(32))

    signature = node.keypair.sign(nonce)
    proof = IdentityProof(
        did       = node.did,
        nonce     = nonce,
        signature = signature,
        capabilities = node.active_capabilities()
    )

    result = service.verify(proof)
    IF result.valid:
        RETURN AccessGrant(scope=result.matched_capabilities)
    ELSE:
        log_auth_failure(node.did, service.id)
        RETURN AccessDenied(reason=result.reason)
```

## 5. Pseudocode: Key Recovery

```
FUNCTION recover_identity(guardians_available):
    ASSERT len(guardians_available) >= 3  # Shamir threshold

    shares = []
    FOR guardian IN guardians_available[:3]:
        share = guardian.retrieve_share(challenge_response)
        shares.append(share)

    recovered_key = shamir_reconstruct(shares, threshold=3)

    # Rotate immediately after recovery
    new_keypair = Ed25519.generate(fresh_entropy())
    new_did     = "did:bizra:" + base58_encode(new_keypair.public_key)

    # Migrate credentials
    migration_receipt = migrate_credentials(
        old_did     = recovered_key.did,
        new_did     = new_did,
        new_keypair = new_keypair
    )

    # Publish revocation
    BlockGraph.publish_revocation(old_did=recovered_key.did,
                                  new_did=new_did,
                                  proof=migration_receipt)

    RETURN new_keypair, new_did
```

---

## 6. TDD Anchors

```
TEST genesis_requires_constitution_acceptance:
    human = MockHuman(accepts_constitution=False)
    EXPECT_RAISE genesis_ceremony(human, CONSTITUTION)

TEST genesis_creates_valid_ed25519:
    node = genesis_ceremony(test_human, CONSTITUTION)
    ASSERT Ed25519.verify_keypair(node.keypair) == True
    ASSERT node.did.startswith("did:bizra:")

TEST shamir_recovery_with_3_of_5:
    node = genesis_ceremony(test_human, CONSTITUTION)
    guardians = node.recovery_guardians[:3]
    recovered = recover_identity(guardians)
    ASSERT recovered.new_did != node.did  # rotated
    ASSERT recovered.new_keypair.is_valid()

TEST resource_pledge_capped_at_90_percent:
    node = genesis_ceremony(test_human, CONSTITUTION)
    ASSERT node.pledge.compute <= available_compute() * 0.9

TEST constitutional_harness_boots_active:
    node = genesis_ceremony(test_human, CONSTITUTION)
    ASSERT node.harness.fate_gate.active == True
    ASSERT node.harness.gini_guard.ceiling == 0.35

TEST delegate_key_has_limited_scope:
    node = genesis_ceremony(test_human, CONSTITUTION)
    delegate = node.create_delegate(capabilities=["read_only"])
    ASSERT "write" NOT IN delegate.capabilities
    ASSERT delegate.expiry < now() + days(30)
```

---

## 7. Cross-References
- `core/auth/` — Middleware + auth implementation
- `core/integration/constants.py` — Ihsan/Gini thresholds
- `bizra-omega/bizra-core/` — Identity + FATE + Constitution (Rust)
- `bizra-omega/fate-binding/` — Z3 + Dilithium post-quantum
- Phase 06 — FATE Gate detailed specification
- Phase 03 — PAT-7 / SAT-49 detailed specification
