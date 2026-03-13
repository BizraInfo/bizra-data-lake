# 05 — Universal Resource Pool (URP) Economy

> Module: `bizra-installer/src/urp/` + `bizra-resourcepool/`
> Language: Rust (resource management) + Python (PoI scoring)
> Constitutional Anchor: Law 6 (Sovereign Economics) + Al-Ma'idah 5:2

## 1. Core Principle

Resource sharing is cooperation (التعاون على البر), not extraction.
Users keep 100% of earned SEED. Zakat (2.5%) is the only deduction.
Sharing is always opt-in. A node without URP loses zero functionality.

## 2. Resource Types & Reward Rates

```
STRUCT ResourceContribution:
    cpu_cores:      u32          # Dedicated cores
    ram_gb:         f32          # Dedicated RAM
    disk_gb:        f32          # Dedicated storage
    vram_gb:        f32          # Dedicated GPU memory
    bandwidth_mbps: f32          # Network bandwidth
    witnessing:     bool         # Heartbeat validation

CONST REWARD_RATES = {
    "cpu":        0.05,   # SEED per compute-hour
    "ram":        0.02,   # SEED per GB-hour
    "disk":       0.01,   # SEED per GB-month
    "gpu":        0.20,   # SEED per GPU-hour
    "bandwidth":  0.01,   # SEED per GB-transferred
    "witnessing": 0.01,   # SEED per witness event
}
```

## 3. Dedication Tiers (Auto-Suggested)

```
FUNCTION suggest_urp_tier(profile: &DeviceProfile) -> URPSuggestion:
    ram = profile.ram_total_gb
    cores = profile.cpu_cores
    gpu_vram = profile.gpu.map(|g| g.vram_gb).unwrap_or(0.0)

    tier = MATCH ram:
        r IF r < 2.0  => URPTier::Witness
        r IF r < 6.0  => URPTier::Light
        r IF r < 24.0 => URPTier::Standard
        r IF r < 48.0 => URPTier::Contributor
        _             => URPTier::Anchor

    RETURN URPSuggestion {
        tier,
        cpu_cores:  min(cores / 4, 2),           # ~25% of cores
        ram_gb:     min(ram * 0.25, 4.0),         # ~25% of RAM
        disk_gb:    min(profile.disk_available_gb * 0.04, 10.0),
        vram_gb:    gpu_vram * 0.33,              # ~33% of VRAM
        schedule:   Schedule::WhenIdle,
    }
```

```
ENUM URPTier:
    Witness:
        description: "Heartbeat only"
        shared:      "Constitutional witnessing"
        kept:        "100% of everything else"
        min_device:  "1 GB RAM"

    Light:
        description: "Minimal sharing"
        shared:      "1 core, 512MB RAM"
        kept:        "Remaining cores + RAM"
        min_device:  "2 GB RAM"

    Standard:
        description: "Balanced sharing"
        shared:      "2 cores, 4GB RAM, 10GB disk"
        kept:        "Majority of resources"
        min_device:  "8 GB RAM"

    Contributor:
        description: "Generous sharing"
        shared:      "4 cores, 16GB RAM, 2GB VRAM"
        kept:        "Remaining for local work"
        min_device:  "32 GB RAM"

    Anchor:
        description: "Infrastructure node"
        shared:      "8+ cores, 32GB+ RAM, 8GB+ VRAM"
        kept:        "Still 50%+ for local"
        min_device:  "64 GB RAM"
```

## 4. URPConfig & Scheduling

```
STRUCT URPConfig:
    enabled:    bool
    cpu_cores:  u32
    ram_gb:     f32
    disk_gb:    f32
    vram_gb:    f32
    schedule:   Schedule
    child_mode: bool     # If true, max 25% sharing

ENUM Schedule:
    Always                          # 24/7
    WhenIdle { timeout_min: u32 }   # After N minutes idle (default 5)
    Scheduled { hours: TimeRange }  # Specific hours (e.g. 22:00-06:00)
    Manual                          # User activates explicitly
    Never                           # No sharing — full sovereignty

FUNCTION apply_schedule(config: &URPConfig, system_state: &SystemState) -> bool:
    # Returns true if resources should be shared right now

    MATCH config.schedule:
        Always    => RETURN true
        WhenIdle  => RETURN system_state.idle_minutes >= config.schedule.timeout_min
        Scheduled => RETURN current_time() IN config.schedule.hours
        Manual    => RETURN config.manual_active
        Never     => RETURN false
```

## 5. Resource Yielding (Priority)

```
FUNCTION yield_resources(config: &URPConfig, user_activity: &Activity):
    # When user becomes active, URP yields gracefully

    IF user_activity.is_active():
        # Progressive release over 10 seconds
        FOR step IN 1..=10:
            release_pct = step * 10  # 10%, 20%, ... 100%
            reduce_urp_allocation(release_pct)
            sleep(1s)

        # User's work is ALWAYS priority
        pause_urp_tasks()

    IF user_activity.becomes_idle():
        # Resume gradually over 30 seconds
        FOR step IN 1..=10:
            allocate_pct = step * 10
            increase_urp_allocation(allocate_pct)
            sleep(3s)
```

## 6. Proof of Impact (PoI) Verification

```
# Contributions are VERIFIED, not trusted

FUNCTION verify_cpu_contribution(request, result, timing) -> PoIResult:
    # 1. Request hash proves the work was assigned
    request_hash = blake2b(canonical_bytes(request))

    # 2. Result hash proves the work was done
    result_hash = blake2b(canonical_bytes(result))

    # 3. Timing proves the work was real (can't fake speed)
    IF timing.duration < MIN_COMPUTE_TIME(request.complexity):
        RETURN PoIResult::Suspicious("Too fast — likely cached or faked")

    IF timing.duration > MAX_COMPUTE_TIME(request.complexity):
        RETURN PoIResult::Timeout("Too slow — possible resource contention")

    # 4. Cross-validation (2+ nodes verify same request)
    IF cross_validate_count >= 2:
        IF all_results_match():
            RETURN PoIResult::Verified(request_hash, result_hash)
        ELSE:
            RETURN PoIResult::Disputed("Results diverge — arbitration needed")

    RETURN PoIResult::Verified(request_hash, result_hash)

FUNCTION verify_storage_contribution(node_id, claimed_data) -> PoIResult:
    # Merkle proof: node must actually store the data
    challenge = random_merkle_challenge(claimed_data)
    response = request_merkle_proof(node_id, challenge)

    IF verify_merkle_proof(response, claimed_data.root_hash):
        RETURN PoIResult::Verified
    ELSE:
        RETURN PoIResult::Failed("Cannot prove data is stored")

FUNCTION verify_gpu_inference(request, output, node_id) -> PoIResult:
    # Output hash verified by 2+ independent nodes
    our_hash = blake2b(canonical_bytes(output))
    peer_hashes = collect_peer_outputs(request, exclude=node_id, count=2)

    matching = count(h FOR h IN peer_hashes IF h == our_hash)
    IF matching >= 1:
        RETURN PoIResult::Verified
    ELSE:
        RETURN PoIResult::Disputed("Output diverges from peers")

FUNCTION verify_witnessing(heartbeat, node_id) -> PoIResult:
    # Ed25519 signature proves identity
    IF NOT ed25519_verify(heartbeat.signature, heartbeat.payload, node_id.public_key):
        RETURN PoIResult::Failed("Invalid signature")
    RETURN PoIResult::Verified
```

## 7. SEED Reward Calculation

```
FUNCTION calculate_reward(
    contribution: &ResourceContribution,
    duration_hours: f64,
    poi_result: &PoIResult,
    ihsan_score: f64
) -> f64:
    # Only quality contributions earn SEED
    IF poi_result != PoIResult::Verified:
        RETURN 0.0

    IF ihsan_score < 0.85:
        RETURN 0.0  # Constitutional gate

    # Base reward from contribution
    base = 0.0
    base += contribution.cpu_hours() * REWARD_RATES["cpu"]
    base += contribution.ram_gb * duration_hours * REWARD_RATES["ram"]
    base += contribution.disk_gb * (duration_hours / 720.0) * REWARD_RATES["disk"]
    base += contribution.gpu_hours() * REWARD_RATES["gpu"]
    base += contribution.bandwidth_gb * REWARD_RATES["bandwidth"]
    base += contribution.witness_count * REWARD_RATES["witnessing"]

    # Ihsan multiplier (excellence rewards more)
    ihsan_mult = 1.0 + (ihsan_score - 0.85) * 2.0  # 0.85→1.0, 1.0→1.3

    reward = base * ihsan_mult

    RETURN reward

FUNCTION mint_urp_reward(node_id, reward_amount):
    # Zakat is deducted at mint time
    zakat = reward_amount * 0.025
    net = reward_amount - zakat

    token_minter.mint(
        recipient = node_id,
        amount = net,
        reason = "urp_contribution",
        zakat_deducted = zakat
    )

    # Emit receipt
    receipt = ActionReceipt {
        action: "urp.reward",
        node_id: node_id,
        gross: reward_amount,
        zakat: zakat,
        net: net,
        timestamp: now_iso8601(),
    }
    evidence_ledger.append(receipt)
```

## 8. Reverse Scaling Proof

```
# More nodes → more shared reflexes → higher cache hit → faster for all

FUNCTION estimate_network_performance(node_count: u64) -> NetworkMetrics:
    # Cache hit rate follows logarithmic saturation
    cache_hit_rate = MATCH node_count:
        n IF n <= 1       => 0.0
        n IF n <= 100     => 0.10
        n IF n <= 10_000  => 0.30
        n IF n <= 1_000_000  => 0.60
        n IF n <= 1_000_000_000 => 0.90
        _                 => 0.95

    # Response time improves with cache hits
    s2_latency_ms = 1800.0  # Full deliberation
    s1_latency_ms = 50.0    # Cached reflex
    avg_latency = s2_latency_ms * (1.0 - cache_hit_rate)
                + s1_latency_ms * cache_hit_rate

    # Emission multiplier decreases (system gets smarter)
    emission_mult = 1.0 - (0.8 * cache_hit_rate)

    RETURN NetworkMetrics {
        node_count,
        cache_hit_rate,
        avg_latency_ms: avg_latency,
        emission_multiplier: emission_mult,
    }
```

## 9. Genesis-1: Founder's Contribution

```
# The first URP contribution — same law as everyone else

STRUCT Genesis1Contribution:
    hardware: [
        { device: "MSI Titan 18 HX (i9-14900HX, 128GB, RTX 4090 16GB)", dedication: 1.0 },
        { device: "Samsung Z Fold 6 (12GB, SD8G3)", dedication: 1.0 },
    ]
    data: {
        research_docs: "~150 original",
        total_size: "1.3TB+",
        hours: 15000,
    }
    code: {
        python_files: 880,
        rust_crates: 22,
        test_files: 560,
        repos: "All BizraInfo GitHub repos",
    }
    founding_papers: ["الرسالة (Ramadan 2023)", "البذرة (Ramadan 2023)"]

FUNCTION evaluate_genesis1(contribution: Genesis1Contribution) -> EvaluationResult:
    # Same process as ANY user contribution
    # PAT-7 indexes every artifact
    # SAT-5 verifies authenticity + quality
    # Ihsan gate applies (< 0.85 earns zero)

    FOR artifact IN contribution.all_artifacts():
        pat_index = pat7.index(artifact)
        sat_verify = sat5.verify(artifact)

        IF sat_verify.ihsan_score < 0.85:
            artifact.reward = 0.0  # Even founder's work must pass
            CONTINUE

        artifact.reward = calculate_fair_market_value(artifact, sat_verify)

    total_seed = sum(a.reward FOR a IN contribution.all_artifacts())

    # Founder's oath: 50% to community pool (sadaqah, NOT protocol tax)
    founder_keeps = total_seed * 0.50
    community_pool = total_seed * 0.50

    RETURN EvaluationResult {
        total_minted: total_seed,
        founder_share: founder_keeps,
        community_share: community_pool,
        artifacts_evaluated: contribution.all_artifacts().len(),
        artifacts_below_ihsan: count(a FOR a IF a.reward == 0.0),
    }
```

## 10. Constitutional Constraints

```
CONST URP_CONSTRAINTS = {
    # Default recommendation
    "default_max_share_pct":     0.50,   # Recommend max 50%

    # User CAN override to 100% (sovereignty)
    "absolute_max_share_pct":    1.00,

    # Override requires explicit confirmation
    "override_confirmation":     true,

    # User's active work always has priority
    "active_work_priority":      true,

    # Sharing is ALWAYS opt-in
    "opt_in_only":               true,

    # Default schedule
    "default_schedule":          "when_idle",

    # Earnings are 100% user's (no platform cut)
    "platform_cut":              0.0,

    # Only Ihsan >= 0.85 contributions earn SEED
    "ihsan_gate":                0.85,

    # Child safety: max 25% if child_mode
    "child_max_share_pct":       0.25,

    # Zakat: 2.5% at mint (protocol-level, cannot change)
    "zakat_rate":                0.025,
}

FUNCTION validate_urp_config(config: URPConfig) -> Result<(), Vec<String>>:
    errors = []

    IF config.child_mode AND config.total_share_pct() > 0.25:
        errors.push("Child mode limits sharing to 25%")

    IF config.total_share_pct() > 0.50 AND NOT config.user_confirmed_override:
        errors.push("Sharing > 50% requires explicit confirmation")

    # But if user confirms, allow up to 100%
    IF config.total_share_pct() > 1.0:
        errors.push("Cannot share more than 100% of resources")

    IF errors.len() > 0:
        RETURN Err(errors)
    RETURN Ok(())
```

## 11. URP Receipt Chain

```
FUNCTION emit_urp_receipt(
    node_id: &str,
    contribution: &ResourceContribution,
    poi: &PoIResult,
    reward: f64
) -> ActionReceipt:
    receipt = ActionReceipt {
        action_type: "urp.contribution",
        node_id: node_id,
        timestamp: now_iso8601(),
        payload: {
            cpu_hours: contribution.cpu_hours(),
            ram_gb_hours: contribution.ram_gb_hours(),
            disk_gb_months: contribution.disk_gb_months(),
            gpu_hours: contribution.gpu_hours(),
            poi_status: poi.status(),
            ihsan_score: poi.ihsan_score(),
            reward_gross: reward,
            reward_net: reward * (1.0 - 0.025),  # After Zakat
        },
        prev_hash: evidence_ledger.last_hash(),
    }

    receipt.hash = blake2b(canonical_bytes(receipt))
    receipt.signature = ed25519_sign(receipt.hash, node_keypair)

    evidence_ledger.append(receipt)
    RETURN receipt
```

## TDD Anchors

```
TEST witness_tier_for_1gb_device:
    profile = DeviceProfile { ram_total_gb: 1.0, ... }
    suggestion = suggest_urp_tier(&profile)
    ASSERT suggestion.tier == URPTier::Witness

TEST standard_tier_for_16gb_device:
    profile = DeviceProfile { ram_total_gb: 16.0, cpu_cores: 8, ... }
    suggestion = suggest_urp_tier(&profile)
    ASSERT suggestion.tier == URPTier::Standard
    ASSERT suggestion.cpu_cores <= 4  # Max 50% of cores

TEST reward_zero_below_ihsan_gate:
    contribution = mock_contribution(cpu_hours=10.0)
    poi = PoIResult::Verified
    reward = calculate_reward(&contribution, 10.0, &poi, 0.80)  # Below 0.85
    ASSERT reward == 0.0

TEST reward_positive_above_ihsan_gate:
    contribution = mock_contribution(cpu_hours=10.0)
    poi = PoIResult::Verified
    reward = calculate_reward(&contribution, 10.0, &poi, 0.95)
    ASSERT reward > 0.0

TEST zakat_deducted_at_mint:
    mint_urp_reward("node1", 100.0)
    balance = get_balance("node1")
    ASSERT abs(balance - 97.5) < 0.01  # 100 - 2.5% Zakat

TEST child_mode_caps_at_25pct:
    config = URPConfig { child_mode: true, cpu_cores: 8, ... }
    result = validate_urp_config(config)
    # Must reject if total share > 25%

TEST sharing_disabled_loses_zero_functionality:
    config = URPConfig::disabled()
    node = create_node_with_urp(config)
    ASSERT node.can_run_missions()
    ASSERT node.can_earn_seed_locally()
    ASSERT node.heartbeat_alive()

TEST schedule_when_idle_respects_activity:
    config = URPConfig { schedule: Schedule::WhenIdle { timeout_min: 5 }, ... }
    state = SystemState { idle_minutes: 3 }
    ASSERT apply_schedule(&config, &state) == false
    state.idle_minutes = 6
    ASSERT apply_schedule(&config, &state) == true

TEST poi_rejects_too_fast_compute:
    result = verify_cpu_contribution(
        request = complex_inference_request(),
        result = valid_result(),
        timing = Timing { duration: 0.001s }  # Impossibly fast
    )
    ASSERT result == PoIResult::Suspicious

TEST genesis1_same_ihsan_gate_as_everyone:
    # Founder's artifacts below Ihsan earn zero
    artifact = mock_artifact(ihsan_score=0.70)
    reward = evaluate_single_artifact(artifact)
    ASSERT reward == 0.0  # Even founder gets nothing below gate
```
