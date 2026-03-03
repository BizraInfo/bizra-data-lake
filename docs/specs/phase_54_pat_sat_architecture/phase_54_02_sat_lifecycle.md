# Phase 54.2: SAT-5 Lifecycle — Minting, URP Registration, Daemon Operations

> Standing on Giants: Lamport (Byzantine consensus, 1982) · Dijkstra (self-stabilization, 1974) · Deming (continuous improvement, 1950) · Nakamoto (decentralized validation, 2008) · Al-Ghazali (Ihsan as systemic excellence, 1095) · Shannon (resource-information balance, 1948)

## 1. Overview

SAT (System Agentic Team) is a squad of 5 agents minted alongside every PAT team.
Unlike PAT, SAT does NOT serve the user — it serves the SYSTEM. SAT agents are
immediately contributed to the Universal Resource Pool (URP), where they join the
collective workforce that keeps BIZRA self-sustainable, self-correcting, and secure.

**Critical design rule**: Users NEVER interact with SAT directly. PAT talks to SAT
on behalf of the user when system resources are needed.

## 2. The 5 SAT Agents

| # | Agent | Role | Subsystem |
|---|-------|------|-----------|
| 1 | **Guardian** | Security monitoring, threat detection, constitutional enforcement | Security |
| 2 | **Librarian** | Knowledge indexing, data integrity, dedup, pipeline health | Data |
| 3 | **Auditor** | Evidence chain, compliance logging, Ihsan scoring, attestation | Governance |
| 4 | **Healer** | Self-repair, health monitoring, crash recovery, optimization | Reliability |
| 5 | **Herald** | Cross-node messaging, federation sync, network coordination | Network |

## 3. SAT Operational Properties

```pseudocode
CLASS SATAgent:
    """
    System Agentic Team agent.
    Always daemon. Always proactive_partner. Always constitutional.
    """

    # Identity
    agent_id:       str           # "sat-{node_id}-{role}"
    role:           SATRole       # Guardian | Librarian | Auditor | Healer | Herald
    origin_node:    NodeID        # Which node minted this SAT (provenance)

    # Operational mode — FIXED, never changes
    mode:           AgentMode = AgentMode.PROACTIVE_PARTNER   # Always autonomous
    daemon:         bool      = True                          # Always background
    uptime_target:  str       = "24/7"                        # Always on

    # Constitutional constraints — STRICTER than PAT
    ihsan_gate:     float     = STRICT_IHSAN_THRESHOLD        # 0.99 (vs PAT's 0.95)
    snr_gate:       float     = SNR_THRESHOLD_T0_ELITE        # 0.98 (vs PAT's 0.85)
    adl_gini_max:   float     = ADL_GINI_THRESHOLD            # 0.35
    daughter_test:  bool      = True

    # Resource boundaries
    data_access:    str       = "system_metrics_only"         # Cannot read user data
    user_access:    str       = "none"                        # Cannot contact users
    network_access: str       = "via_urp_only"                # Through URP gateway
```

## 4. Minting Flow

```pseudocode
FUNCTION mint_sat_team(node_id: NodeID) -> SATTeam:
    """
    Mint 5 SAT agents for a new node. These immediately move to URP.
    The user who owns this node has ZERO control over these agents.

    Standing on Giants: Dijkstra (separation of concerns) — user logic and
    system logic must be completely isolated.
    """
    sat = SATTeam(origin_node=node_id)

    FOR role IN [Guardian, Librarian, Auditor, Healer, Herald]:
        agent = SATAgent(
            agent_id    = f"sat-{node_id}-{role.name}",
            role        = role,
            origin_node = node_id,

            # Fixed operational state
            mode        = AgentMode.PROACTIVE_PARTNER,
            daemon      = True,

            # Strict constitutional gates
            ihsan_gate  = STRICT_IHSAN_THRESHOLD,    # 0.99
            snr_gate    = SNR_THRESHOLD_T0_ELITE,    # 0.98
            constitution = Constitution.load(),

            # System capabilities (role-specific)
            capabilities = SAT_ROLE_CAPABILITIES[role],

            # PID file for daemon management
            pid_file    = f"sovereign_state/sat-{role.name}.pid",
        )
        sat.add_agent(agent)

    # SAT team is IMMEDIATELY registered with URP
    # It does NOT stay on the user's node — it joins the collective
    urp.register_sat_team(sat)

    RETURN sat
```

## 5. URP Registration

When SAT is minted, it joins the Universal Resource Pool:

```pseudocode
CLASS UniversalResourcePool:

    FUNCTION register_sat_team(sat: SATTeam):
        """
        SAT agents are pooled by role. Each role forms a department.

        At 1 user  : 5 SAT agents (1 per department)
        At 100     : 500 SAT agents (100 per department)
        At 1M      : 5M SAT agents (1M per department)
        """
        FOR agent IN sat.agents:
            department = self.departments[agent.role]
            department.add_agent(agent)

            # Start agent as daemon
            agent.start_daemon()
            self.active_inventory.register(agent)

            # Log provenance (which node contributed this SAT)
            self.provenance_ledger.record(
                agent_id    = agent.agent_id,
                origin_node = agent.origin_node,
                registered  = now(),
                department  = agent.role.name,
            )

    FUNCTION get_department_strength(role: SATRole) -> int:
        """How many SAT agents serve this role system-wide."""
        RETURN len(self.departments[role].agents)
```

## 6. The 5 SAT Departments

### 6.1 Guardian Department (Security)

```pseudocode
CLASS GuardianSAT(SATAgent):
    """
    Monitors all inbound/outbound traffic. Detects attacks.
    Enforces constitutional gates on every action.
    """
    role = SATRole.GUARDIAN

    FUNCTION patrol():
        """Continuous security monitoring loop."""
        WHILE self.running:
            # Scan PAT requests coming through
            FOR request IN self.pending_requests():
                threat = self.assess_threat(request)
                IF threat.level >= CRITICAL:
                    self.block_and_alert(request, threat)
                ELIF threat.level >= SUSPICIOUS:
                    self.flag_for_review(request, threat)
                ELSE:
                    self.approve(request)

            # Check node integrity
            self.verify_constitution_integrity()
            self.check_for_tampering()

            SLEEP(GUARDIAN_PATROL_INTERVAL)   # 5 seconds

    FUNCTION validate_pat_request(envelope: PCIEnvelope) -> ValidationResult:
        """Every PAT request to the network goes through Guardian first."""
        checks = [
            self.check_ihsan(envelope),          # Ihsan >= 0.95
            self.check_daughter_test(envelope),   # Would I approve for my daughter?
            self.check_adl_gini(envelope),        # Fairness <= 0.35
            self.check_resource_budget(envelope), # Within user's allocation
            self.check_malicious_payload(envelope),# No injection/exploit patterns
        ]
        IF all(c.passed for c in checks):
            RETURN ValidationResult(status=APPROVED)
        ELSE:
            RETURN ValidationResult(status=REJECTED, reasons=[c for c in checks if not c.passed])
```

### 6.2 Librarian Department (Data)

```pseudocode
CLASS LibrarianSAT(SATAgent):
    """
    Maintains system-wide data integrity. Indexes, deduplicates,
    manages the knowledge graph across the network.
    """
    role = SATRole.LIBRARIAN

    FUNCTION maintain():
        WHILE self.running:
            self.scan_for_duplicates()           # SHA-256 dedup
            self.update_system_indexes()         # Knowledge graph
            self.verify_data_integrity()         # Checksums
            self.garbage_collect_orphans()        # Clean stale data
            self.optimize_storage_allocation()   # Balance hot/cold
            SLEEP(LIBRARIAN_CYCLE_INTERVAL)      # 60 seconds
```

### 6.3 Auditor Department (Governance)

```pseudocode
CLASS AuditorSAT(SATAgent):
    """
    Records every system action in the evidence chain.
    Scores Ihsan compliance. Detects constitutional drift.
    """
    role = SATRole.AUDITOR

    FUNCTION audit():
        WHILE self.running:
            FOR action IN self.recent_actions():
                receipt = EvidenceReceipt(
                    action     = action,
                    ihsan      = self.score_ihsan(action),
                    snr        = self.score_snr(action),
                    timestamp  = now(),
                    hash_chain = self.chain_hash(action),
                )
                self.evidence_ledger.append(receipt)

                IF receipt.ihsan < UNIFIED_IHSAN_THRESHOLD:
                    self.flag_quality_drift(action, receipt)

            SLEEP(AUDITOR_CYCLE_INTERVAL)  # 10 seconds
```

### 6.4 Healer Department (Reliability)

```pseudocode
CLASS HealerSAT(SATAgent):
    """
    Self-repair engine. Detects failures, restarts services,
    optimizes resource allocation, prevents cascading failures.

    Standing on Giants: Dijkstra (self-stabilization) — the system must
    converge to a correct state from ANY arbitrary state.
    """
    role = SATRole.HEALER

    FUNCTION heal():
        WHILE self.running:
            health = self.check_system_health()

            FOR component IN health.degraded:
                IF component.recoverable:
                    self.attempt_recovery(component)
                    self.log_recovery_attempt(component)
                ELSE:
                    self.escalate_to_department(SATRole.GUARDIAN, component)

            # Proactive optimization
            self.optimize_memory_usage()
            self.rebalance_load()
            self.defragment_storage()

            SLEEP(HEALER_CYCLE_INTERVAL)  # 30 seconds
```

### 6.5 Herald Department (Network)

```pseudocode
CLASS HeraldSAT(SATAgent):
    """
    Network coordination. Cross-node messaging. Federation sync.
    Gossip protocol for state propagation.

    Standing on Giants: Lamport (distributed time) · Nakamoto (gossip protocol)
    """
    role = SATRole.HERALD

    FUNCTION coordinate():
        WHILE self.running:
            # Propagate local state to peers
            self.gossip_state_update()

            # Receive and validate peer messages
            FOR msg IN self.inbox():
                IF self.validate_peer_message(msg):
                    self.apply_state_update(msg)
                ELSE:
                    self.report_invalid_peer(msg.sender)

            # Consensus participation
            IF self.is_consensus_round():
                self.participate_in_consensus()

            SLEEP(HERALD_CYCLE_INTERVAL)  # 5 seconds
```

## 7. Daemon Management

Every SAT agent runs as a daemon with PID file tracking:

```pseudocode
CLASS SATDaemonManager:

    FUNCTION start_all(sat_team: SATTeam):
        FOR agent IN sat_team.agents:
            # Check for stale PID
            IF agent.pid_file.exists():
                old_pid = agent.pid_file.read()
                IF process_alive(old_pid):
                    LOG.warning(f"{agent.role} already running (PID {old_pid})")
                    CONTINUE
                ELSE:
                    agent.pid_file.delete()  # Stale PID, clean up

            # Start daemon
            pid = fork_daemon(agent.run_loop)
            agent.pid_file.write(str(pid))
            LOG.info(f"SAT {agent.role} started: PID {pid}")

    FUNCTION stop_all(sat_team: SATTeam):
        FOR agent IN sat_team.agents:
            IF agent.pid_file.exists():
                pid = agent.pid_file.read()
                send_signal(pid, SIGTERM)
                wait_for_exit(pid, timeout=30)
                agent.pid_file.delete()

    FUNCTION health_check(sat_team: SATTeam) -> dict:
        status = {}
        FOR agent IN sat_team.agents:
            IF agent.pid_file.exists() AND process_alive(agent.pid_file.read()):
                status[agent.role] = "RUNNING"
            ELSE:
                status[agent.role] = "DOWN"
        RETURN status
```

## 8. SAT Cannot Be Controlled by Users

This is a hard architectural constraint:

```pseudocode
CLASS SATAccessControl:
    """
    Users cannot:
    - Start/stop SAT agents
    - Change SAT configuration
    - Read SAT internal state
    - Modify SAT constitutional thresholds
    - Redirect SAT to serve their interests

    Only the constitution and the URP consensus can modify SAT behavior.
    """

    FUNCTION validate_caller(caller: AgentID, action: str) -> bool:
        IF caller.startswith("pat-"):
            # PAT can ONLY send validated requests through the gateway
            RETURN action == "submit_request"

        IF caller.startswith("sat-"):
            # SAT agents can coordinate with each other
            RETURN action IN SAT_INTER_AGENT_ACTIONS

        IF caller == "urp-consensus":
            # URP consensus can reconfigure SAT (e.g., scale departments)
            RETURN True

        # Nobody else can touch SAT
        RETURN False
```

## 9. TDD Anchors

```python
class TestSATLifecycle:
    """Phase 54.2: SAT minting and daemon operations."""

    def test_mint_creates_5_agents(self):
        sat = mint_sat_team(mock_node_id())
        assert len(sat.agents) == 5

    def test_all_sat_agents_are_proactive_partner(self):
        sat = mint_sat_team(mock_node_id())
        assert all(a.mode == AgentMode.PROACTIVE_PARTNER for a in sat.agents)

    def test_sat_mode_cannot_be_changed(self):
        sat = mint_sat_team(mock_node_id())
        with pytest.raises(ImmutableError):
            sat.agents[0].mode = AgentMode.REACTIVE

    def test_sat_registers_with_urp_on_mint(self):
        urp = MockURP()
        sat = mint_sat_team(mock_node_id())
        urp.register_sat_team(sat)
        assert urp.total_agents == 5

    def test_sat_ihsan_threshold_is_strict(self):
        sat = mint_sat_team(mock_node_id())
        for agent in sat.agents:
            assert agent.ihsan_gate == 0.99  # STRICT_IHSAN_THRESHOLD

    def test_sat_snr_threshold_is_elite(self):
        sat = mint_sat_team(mock_node_id())
        for agent in sat.agents:
            assert agent.snr_gate == 0.98  # SNR_THRESHOLD_T0_ELITE

    def test_user_cannot_stop_sat(self):
        sat = mint_sat_team(mock_node_id())
        with pytest.raises(AccessDenied):
            sat.stop(caller="pat-user123-planner")

    def test_user_cannot_read_sat_state(self):
        sat = mint_sat_team(mock_node_id())
        with pytest.raises(AccessDenied):
            sat.agents[0].get_internal_state(caller="pat-user123-planner")

    def test_sat_cannot_read_user_data(self):
        sat = mint_sat_team(mock_node_id())
        with pytest.raises(AccessDenied):
            sat.agents[0].read_user_data(user_id="user123")

    def test_pid_file_created_on_daemon_start(self):
        sat = mint_sat_team(mock_node_id())
        manager = SATDaemonManager()
        manager.start_all(sat)
        for agent in sat.agents:
            assert agent.pid_file.exists()

    def test_stale_pid_cleaned_on_restart(self):
        sat = mint_sat_team(mock_node_id())
        # Simulate stale PID
        sat.agents[0].pid_file.write("99999")  # Dead PID
        manager = SATDaemonManager()
        manager.start_all(sat)
        assert int(sat.agents[0].pid_file.read()) != 99999
```
