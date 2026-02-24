# Phase 05 — P0/P1 Build Sequence & Integration Map

> **Version:** 0.1.0 | **Status:** Specification
> **Standing on Giants:** Brooks (surgical team, 1975) · Conway (org ≈ architecture, 1967) · Lamport (ordering) · Deming (PDCA build cadence)

## 5.1 P0 Build Sequence (Fastest Path to "1 Human Empowered")

```
P0 CRITICAL PATH (estimated 8 weeks)
══════════════════════════════════════

Week 1-2: FOUNDATION
  ├── M1: Installer/Launcher skeleton (Tauri + Rust)
  │     ├── Tauri project init, tray icon, basic window
  │     ├── System service registration (Windows/macOS/Linux)
  │     └── Gate: App launches, shows tray icon, opens window

  ├── M2: Capsule runtime (Docker first)
  │     ├── Docker adapter implementing RuntimeCapsule interface
  │     ├── Resource limits enforcement (cgroups v2)
  │     └── Gate: Capsule starts/stops, health check passes

Week 3-4: IDENTITY + ONBOARDING
  ├── M3: Crypto & Wallet
  │     ├── Ed25519 keypair generation (reuse core.pci.crypto)
  │     ├── Encrypted local store (XChaCha20-Poly1305, Argon2id KDF)
  │     └── Gate: Wallet created, private key encrypted, signature verified

  ├── M4: Onboarding Wizard
  │     ├── Profile capture UI
  │     ├── PAT configuration (toggle agents)
  │     ├── First goal capture (SMART validation)
  │     ├── Baseline capture (Day 0 metrics)
  │     └── Gate: Onboarding completes < 10 min, baseline stored

Week 5-6: EMPOWERMENT CORE
  ├── M5: Daily Empowerment Loop
  │     ├── Goal intake UI
  │     ├── PAT planning pipeline (reuse core.sovereign.collective_intelligence)
  │     ├── Task decomposition + "Next Action" button
  │     ├── Task Force execution with allowlisted tools
  │     └── Gate: 3 outcomes generated, tasks executable, ≤ 30s plan time

  ├── M6: Receipt Pipeline
  │     ├── Receipt structure + generation (reuse core.pci.envelope)
  │     ├── ZANN_ZERO speculation detection
  │     ├── IHSAN_FLOOR enforcement
  │     ├── Hash chain linking
  │     └── Gate: Every task output has receipt, chain verifiable

Week 7-8: PROOF-OF-IMPACT
  ├── M7: Impact Measurement
  │     ├── Daily metrics collection (auto from receipts)
  │     ├── Weekly Impact Report generation
  │     ├── Impact Ledger (append-only, hash-chained)
  │     └── Gate: Week 1 report generated with baseline comparison

  ├── M8: Integration & Polish
  │     ├── End-to-end flow: install → onboard → plan → execute → report
  │     ├── Error handling, edge cases, UX polish
  │     ├── Smoke test suite (8 pillars)
  │     └── Gate: Full loop completes for 1 real user
```

## 5.2 Milestone Dependency Graph

```
M1 (Installer) ──┐
                  ├──► M4 (Onboarding) ──┐
M2 (Capsule) ────┘                       ├──► M5 (Loop) ──► M6 (Receipts) ──► M7 (Impact)
                                          │                                        │
M3 (Crypto) ──────────────────────────────┘                                        │
                                                                                    ▼
                                                                              M8 (Integration)
```

## 5.3 Integration Contracts

```pseudocode
INTEGRATION_CONTRACTS:
  """
  Defines the interfaces between milestones.
  Each contract is testable independently.
  """

  # Contract C1: Installer → Capsule
  CONTRACT installer_to_capsule:
    installer PRODUCES:
      - config.capsule_type:   CapsuleBackend
      - config.capsule_image:  Path
      - config.data_dir:       Path
    capsule CONSUMES:
      - RuntimeCapsule.start(config) -> Result<HealthStatus>

  # Contract C2: Installer → Onboarding
  CONTRACT installer_to_onboarding:
    installer PRODUCES:
      - config (with capsule running)
    onboarding CONSUMES:
      - OnboardingWizard.run() -> OnboardingResult
    onboarding PRODUCES:
      - config.pat_config:     PATConfiguration
      - config.initial_goals:  List<Goal>
      - baseline:              UserBaseline

  # Contract C3: Crypto → Receipts
  CONTRACT crypto_to_receipts:
    crypto PRODUCES:
      - keypair:               Ed25519Keypair
      - encrypted_store:       EncryptedVolume
    receipts CONSUMES:
      - sign_message(private_key, payload) -> bytes[64]
      - verify_signature(public_key, payload, sig) -> bool
      - sha256(data) -> bytes[32]
      - canonical_json(obj) -> String

  # Contract C4: Empowerment Loop → Receipts
  CONTRACT loop_to_receipts:
    loop PRODUCES:
      - task:         TaskNode
      - input_data:   Any
      - output_data:  Any
      - snr_score:    float
      - ihsan_score:  float
    receipts CONSUMES:
      - generate_receipt(...) -> Receipt

  # Contract C5: Receipts → Impact
  CONTRACT receipts_to_impact:
    receipts PRODUCES:
      - receipt_chain:     List<Receipt>  (append-only)
    impact CONSUMES:
      - load_receipts_for_date(date) -> List<Receipt>
      - verify_chain(chain, pubkey) -> ChainVerdict
    impact PRODUCES:
      - DailyMetric (auto-derived)
      - ImpactReport (weekly)
      - ImpactLedger (cumulative)
```

## 5.4 Tech Stack Decision Matrix

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Host App | **Tauri 2.x + Rust** | Native performance, small binary (~5MB), cross-platform, tray support |
| Frontend | **SolidJS** (in Tauri webview) | Reactive, tiny bundle, no virtual DOM overhead |
| Capsule Primary | **Docker** (P0) | Widest compatibility, 20+ containers already running on Node0 |
| Capsule Future | **Wasmtime** (P1) | Strongest sandbox, portable, no Docker dependency |
| Crypto | **Ed25519 (dalek)** Rust / **core.pci.crypto** Python | Already implemented in both stacks |
| KDF | **Argon2id** | Memory-hard, OWASP recommended, resists GPU attacks |
| Cipher | **XChaCha20-Poly1305** | 192-bit nonce (safe random), AEAD, NaCl compatible |
| LLM Backend | **Local-first** (LM Studio / Ollama) | Zero cloud dependency, existing infra |
| Storage | **SQLite + encrypted volume** | Embedded, no server, well-understood |
| Hash Chain | **SHA-256 Merkle** | Reuses core.pci, universally verifiable |

## 5.5 P1 Build List (After Node0 Passes "1 Human" Test)

```
P1 FEATURES (post-validation, estimated 6 weeks)
═════════════════════════════════════════════════

  P1-A: Token receive/send UI
    ├── Wallet UI in Tauri (view balance, send, receive)
    ├── Transaction receipts (extends receipt chain)
    └── Gate: Send/receive tokens between two Node0 instances

  P1-B: Resource sharing slider
    ├── CPU/GPU/Storage allocation controls
    ├── Background donation mode ("give idle resources")
    └── Gate: Resources adjustable in real-time, cgroup limits enforced

  P1-C: P2P mesh discovery
    ├── mDNS/gossip for local network discovery
    ├── NAT traversal (STUN/TURN or libp2p)
    ├── Peer identity verification (mutual Ed25519 auth)
    └── Gate: Two nodes discover each other, exchange signed messages

  P1-D: Content posting gateway
    ├── Signed posts (author = node public key)
    ├── No algorithmic feed (chronological only)
    ├── Receipt attached to every post
    └── Gate: Post created, signed, viewable by peers
```

## 5.6 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM model too large for user hardware | HIGH | HIGH | Tiered model selection: 0.5B → 3B → 8B based on available RAM/VRAM |
| Docker not installed on user machine | MEDIUM | HIGH | Detect at install, guide install; P1 adds WASM fallback |
| Install exceeds 10-minute target | MEDIUM | MEDIUM | Pre-built capsule images, delta updates, progress feedback |
| Receipt chain grows too large | LOW | MEDIUM | Pruning with Merkle checkpoints (keep root hashes, archive details) |
| LM Studio not available on user OS | MEDIUM | HIGH | Ollama as universal fallback; embedded GGUF loader as P1 option |
| User abandons after Day 1 | HIGH | HIGH | 10-min daily check-in, "Next Action" button reduces friction |
| Antivirus blocks Docker/capsule | MEDIUM | MEDIUM | Signed binaries, whitelist guide, WASM fallback has no AV triggers |

## 5.7 Success Criteria — "1 Human Empowered"

```pseudocode
SUCCESS_CRITERIA:
  """
  Node0 passes the Empower-One-Human test when ALL of these hold
  for at least 1 real user over 7 consecutive days.
  """

  # Hard gates
  ASSERT install_time <= 600_seconds
  ASSERT terminal_interactions == 0
  ASSERT cloud_api_calls_for_core_loop == 0
  ASSERT receipt_chain_valid == true
  ASSERT avg_ihsan >= 0.95
  ASSERT avg_snr >= 0.85

  # Impact gates (any 3 of 5)
  gates_passed = count_true([
    impact_report.hours_saved > 0,
    impact_report.tasks_shipped > baseline.tasks_per_week,
    impact_report.errors_avoided > 0,
    impact_report.clarity_improvement > 0,
    impact_report.goal_completion_change > 0,
  ])
  ASSERT gates_passed >= 3, "At least 3 of 5 impact dimensions must improve"

  # Qualitative gate
  ASSERT user_would_continue_using == true  # Asked at Day 7
```

## 5.8 TDD Anchors — Integration

```pseudocode
TEST "end_to_end_install_to_first_plan":
  # Simulates full flow on fresh system
  config = fresh_config()
  run_install_wizard(config)
  ASSERT config.setup_complete == true
  ASSERT config.wallet_fingerprint IS NOT NONE

  goals = capture_goals(mock_ui_with_goal)
  plan = generate_daily_plan(goals, mock_context)
  ASSERT len(plan.top_outcomes) == 3
  ASSERT plan.ihsan_score >= 0.95

TEST "end_to_end_task_to_receipt":
  task = plan.task_tree[0]
  result = execute_task(task, AutonomyLevel.AUTOLOW)
  ASSERT result.receipt IS NOT NONE
  ASSERT result.receipt.policy_result == APPROVED

TEST "end_to_end_week_to_impact_report":
  # Run 7 daily cycles
  FOR day IN range(7):
    run_daily_cycle(entity, goals)
  report = generate_weekly_report(week=1, baseline)
  ASSERT report.receipts_total > 0
  ASSERT report.chain_integrity == true
  ASSERT report.hours_saved >= 0  # Non-negative

TEST "all_contracts_satisfied":
  FOR contract IN INTEGRATION_CONTRACTS:
    producer_output = contract.producer.produce()
    consumer_result = contract.consumer.consume(producer_output)
    ASSERT consumer_result.is_ok()

TEST "no_cloud_calls_in_core_loop":
  with network_monitor() as monitor:
    run_daily_cycle(entity, goals)
  external_calls = monitor.calls_to_external_hosts()
  ASSERT len(external_calls) == 0, f"Cloud dependency detected: {external_calls}"
```
