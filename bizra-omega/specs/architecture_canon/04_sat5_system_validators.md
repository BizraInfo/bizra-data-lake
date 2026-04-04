# 04 — SAT-5 System Validators

> SAT-5 = Shared Agentic Team. System-owned. Serves the constitution.
> SAT lives in the URP, not on the user's node.
> SAT validates proof-carrying requests that cross FATE.

## Agent Roles

```
ENUM SatAgent:
    S1_Sentinel     // Security & threat detection
    S2_Oracle       // Constitutional threshold enforcement (Ihsan, SNR, Gini)
    S3_Ledger       // Economics & fairness accounting (SEED, zakat, Gini)
    S4_Conductor    // Resource routing & URP allocation
    S5_Ambassador   // Federation gossip & external propagation
```

## Pseudocode: SAT Validation Pipeline

```
FUNCTION sat_validate(request: ProofCarryingRequest) -> SatVerdict:
    // SAT receives a FATE-admitted request and validates systemically.
    // Each SAT agent validates independently. All must pass.

    // S1 Sentinel: Security scan
    s1_result = S1_Sentinel.scan(request)
    IF s1_result.threat_detected:
        RETURN SatVerdict::Reject("Sentinel: threat detected")

    // S2 Oracle: Constitutional re-verification (system-side)
    s2_result = S2_Oracle.verify_thresholds(request)
    IF NOT s2_result.all_pass:
        RETURN SatVerdict::Reject("Oracle: threshold violation")

    // S3 Ledger: Economic admissibility
    s3_result = S3_Ledger.check_economics(request)
    IF s3_result.gini_violation OR s3_result.balance_insufficient:
        RETURN SatVerdict::Reject("Ledger: economic violation")

    // S4 Conductor: Resource availability
    s4_result = S4_Conductor.check_resources(request)
    IF NOT s4_result.resources_available:
        RETURN SatVerdict::Defer("Conductor: resources unavailable, queue")

    // S5 Ambassador: Federation policy
    s5_result = S5_Ambassador.check_federation_policy(request)
    IF NOT s5_result.policy_ok:
        RETURN SatVerdict::Reject("Ambassador: federation policy violation")

    RETURN SatVerdict::Attest(request)

ENUM SatVerdict:
    Attest(ProofCarryingRequest)    // all 5 pass -> settle into URP
    Reject(String)                  // any fail -> halts
    Defer(String)                   // temporarily queued (resources)
```

## Pseudocode: Individual SAT Agent Details

```
FUNCTION S1_Sentinel.scan(request: ProofCarryingRequest) -> SecurityResult:
    // Verify cryptographic integrity
    sig_ok = verify_ed25519(request.signature, request.origin_node)
    hash_ok = verify_blake3(request.receipt_hash)

    // Check for replay attacks
    replay = is_duplicate_receipt(request.receipt_hash)

    // Check node reputation
    reputation = lookup_node_reputation(request.origin_node)

    RETURN SecurityResult {
        threat_detected: NOT sig_ok OR NOT hash_ok OR replay,
        sig_valid: sig_ok,
        hash_valid: hash_ok,
        is_replay: replay,
        reputation: reputation,
    }

FUNCTION S2_Oracle.verify_thresholds(request: ProofCarryingRequest) -> ThresholdResult:
    // System-side re-verification (independent of user's P5_Crown)
    RETURN ThresholdResult {
        all_pass: request.ihsan_score >= IHSAN_THRESHOLD       // 0.95
             AND  request.snr_score   >= SNR_THRESHOLD         // 0.85
             AND  request.crown_verdict == PASS,
    }

FUNCTION S3_Ledger.check_economics(request: ProofCarryingRequest) -> EconomicResult:
    // Simulate SEED settlement
    current_balance = get_balance(request.origin_node)
    proposed_mint = compute_seed_reward(request)
    post_gini = simulate_gini_after(request.origin_node, proposed_mint)

    RETURN EconomicResult {
        gini_violation: post_gini > ADL_GINI_THRESHOLD,   // 0.35
        balance_insufficient: false,  // minting, not spending
        proposed_reward: proposed_mint,
        zakat_deduction: proposed_mint * ZAKAT_RATE,       // 2.5%
    }

FUNCTION S4_Conductor.check_resources(request: ProofCarryingRequest) -> ResourceResult:
    // Check if URP has capacity for this request
    available = urp_capacity()
    required = estimate_resource_cost(request)

    RETURN ResourceResult {
        resources_available: available >= required,
        queue_position: IF available < required THEN compute_queue_pos() ELSE 0,
    }

FUNCTION S5_Ambassador.check_federation_policy(request: ProofCarryingRequest) -> PolicyResult:
    // Check inter-node policies and federation rules
    origin_known = is_registered_node(request.origin_node)
    policy_compliant = check_federation_rules(request)

    RETURN PolicyResult {
        policy_ok: origin_known AND policy_compliant,
    }
```

## Ownership Contract

```
INVARIANT sat_ownership:
    // SAT agents serve the SYSTEM, not any individual user.
    // No user can direct a SAT agent to bypass validation.
    // SAT agents have no loyalty to the requesting node.

    FOR agent IN sat_agents:
        ASSERT agent.owner == URP_SYSTEM_KEY    // NOT user key
        ASSERT agent.can_be_directed_by_user == false
        ASSERT agent.serves == Constitution
        ASSERT agent.validates_independently == true
```

## TDD Anchors

```
TEST sat5_has_exactly_5_agents:
    sat = SAT5::register(urp)
    ASSERT sat.agents.len() == 5

TEST sentinel_detects_forged_signature:
    request = make_request(signature=FORGED)
    result = S1_Sentinel.scan(request)
    ASSERT result.threat_detected == true

TEST sentinel_detects_replay:
    request = make_valid_request()
    sat_validate(request)  // first time: succeeds
    result = S1_Sentinel.scan(request)  // second time: replay
    ASSERT result.is_replay == true

TEST oracle_rejects_low_scores:
    request = make_request(ihsan=0.80)
    result = S2_Oracle.verify_thresholds(request)
    ASSERT result.all_pass == false

TEST ledger_enforces_gini:
    request = make_request_that_concentrates_wealth()
    result = S3_Ledger.check_economics(request)
    ASSERT result.gini_violation == true

TEST conductor_defers_when_no_capacity:
    exhaust_urp_capacity()
    request = make_valid_request()
    verdict = sat_validate(request)
    ASSERT verdict IS Defer

TEST sat_agents_are_system_owned:
    sat = SAT5::register(urp)
    FOR agent IN sat.agents:
        ASSERT agent.owner == URP_SYSTEM_KEY
        ASSERT agent.can_be_directed_by_user == false
```
