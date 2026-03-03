# Phase 54.1: PAT-7 Lifecycle — Minting, Personalization, Growth

> Standing on Giants: Minsky (agent specialization, 1986) · Kahneman (System 1/2 cognitive modes, 2011) · Csikszentmihalyi (flow states, 1990) · Al-Ghazali (Ihsan as personal excellence, 1095) · General Magic (mobile agent persistence, 1994)

## 1. Overview

PAT (Personal Agentic Team) is a squad of 7 specialized agents minted for each user
at onboarding. They are the user's personal task force — loyal only to the user,
adapting to the user's goals, dreams, workflow, and personality over time.

The user interacts ONLY with PAT. PAT is the face of BIZRA for that user.

## 2. The 7 PAT Agents

| # | Agent | Role | Cognitive Mode |
|---|-------|------|----------------|
| 1 | **Planner** | Decomposes goals into actionable sub-tasks | System 2: strategic thinking |
| 2 | **Researcher** | Retrieves knowledge, learns user's domain | System 2: information gathering |
| 3 | **Coder** | Generates executable actions (Telescript) | System 2: implementation |
| 4 | **Evaluator** | Tests, simulates, scores outcomes | System 2: quality control |
| 5 | **Ethicist** | Applies Ihsan scoring, ensures alignment | System 2: moral reasoning |
| 6 | **Publisher** | Formats output for user consumption | System 1: communication |
| 7 | **Integrator** | Assembles final deliverable, coordinates team | System 2: synthesis |

Ref: Phase 52.3 (`phase_52_03_pat7_pipeline.md`) for detailed pipeline.

## 3. Minting Flow

```pseudocode
FUNCTION mint_pat_team(user: UserIdentity) -> PATTeam:
    """
    Mint 7 fresh PAT agents for a new user.
    Each agent starts with base capabilities and zero personalization.
    Personalization grows through interaction (TTRL on-device learning).
    """
    pat = PATTeam(owner=user.node_id)

    FOR role IN [Planner, Researcher, Coder, Evaluator, Ethicist, Publisher, Integrator]:
        agent = PATAgent(
            agent_id   = f"pat-{user.node_id}-{role.name}",
            role       = role,
            owner      = user.node_id,

            # Initial state
            mode       = AgentMode.REACTIVE,           # Start conservative
            ihsan_gate = UNIFIED_IHSAN_THRESHOLD,      # 0.95
            snr_gate   = UNIFIED_SNR_THRESHOLD,        # 0.85

            # Personalization (empty at mint, grows over time)
            user_prefs    = UserPreferences.default(),
            learned_patterns = [],
            trust_score   = 0.0,                       # Earned, never given

            # Constitutional constraints (immutable)
            constitution  = Constitution.load(),
            daughter_test = True,
            adl_gini_max  = ADL_GINI_THRESHOLD,        # 0.35
        )
        pat.add_agent(agent)

    # Register PAT with the node (NOT with URP — PAT stays local)
    pat.bind_to_node(user.node_id)

    RETURN pat
```

## 4. Personalization Engine

PAT agents learn the user over time through TTRL (on-device reinforcement learning):

```pseudocode
CLASS PATPersonalizer:
    """
    Adapts PAT behavior to user's goals, preferences, and communication style.
    All learning happens ON-DEVICE — user data never leaves the node.

    Standing on Giants: General Magic (agent memory), Csikszentmihalyi (flow matching)
    """

    FUNCTION observe_interaction(agent: PATAgent, interaction: Interaction):
        """Record what the user asked, how PAT responded, user's feedback."""
        trajectory = Trajectory(
            task      = interaction.task,
            response  = interaction.response,
            feedback  = interaction.user_feedback,    # thumbs up/down, edits, acceptance
            timestamp = now(),
        )
        agent.trajectory_buffer.append(trajectory)

    FUNCTION update_model(agent: PATAgent):
        """
        Periodic TTRL update — adjust agent weights based on accumulated feedback.
        Uses SSO spectral norm constraint to prevent catastrophic forgetting.
        """
        IF len(agent.trajectory_buffer) < MIN_TRAJECTORIES:
            RETURN  # Not enough data to learn

        # On-device RL update (no cloud, no data sharing)
        reward_signal = compute_reward(agent.trajectory_buffer)
        agent.policy = ttrl_update(
            policy    = agent.policy,
            rewards   = reward_signal,
            sso_bound = SSO_SPECTRAL_NORM,   # Prevent personality drift
        )

        # Update learned patterns
        agent.learned_patterns = extract_patterns(agent.trajectory_buffer)
        agent.trajectory_buffer.clear()

    FUNCTION adapt_to_goals(agent: PATAgent, goals: list[UserGoal]):
        """
        User declares goals → PAT restructures priorities.
        Goals persist across sessions (living memory).
        """
        agent.user_prefs.active_goals = goals
        agent.priority_weights = compute_priority_weights(goals)

        # Example: user says "I want to learn Arabic"
        # → Researcher prioritizes Arabic resources
        # → Planner structures daily learning plans
        # → Publisher formats output with Arabic translations
```

## 5. Trust Escalation

PAT mode scales with earned trust — never given:

```pseudocode
ENUM AgentMode:
    REACTIVE           = 0   # Only responds when asked
    PROACTIVE_SUGGEST  = 1   # Suggests actions, waits for approval
    PROACTIVE_AUTO     = 2   # Auto-executes low-risk, suggests high-risk
    PROACTIVE_PARTNER  = 3   # Full autonomous partner

FUNCTION compute_trust_score(agent: PATAgent) -> float:
    """
    Trust = f(interaction_count, success_rate, user_satisfaction, time)
    Range: 0.0 (new) to 1.0 (full trust)
    """
    factors = {
        "interactions":   min(agent.interaction_count / 1000, 1.0) * 0.2,
        "success_rate":   agent.success_rate * 0.3,
        "satisfaction":   agent.avg_user_satisfaction * 0.3,
        "tenure_days":    min(agent.days_active / 365, 1.0) * 0.2,
    }
    RETURN sum(factors.values())

FUNCTION determine_mode(trust_score: float) -> AgentMode:
    IF trust_score < 0.25:   RETURN AgentMode.REACTIVE
    IF trust_score < 0.50:   RETURN AgentMode.PROACTIVE_SUGGEST
    IF trust_score < 0.75:   RETURN AgentMode.PROACTIVE_AUTO
    RETURN AgentMode.PROACTIVE_PARTNER
```

Sovereignty tiers map directly:
- SEED (0.00-0.25) → `reactive`
- SPROUT (0.25-0.50) → `proactive_suggest`
- TREE (0.50-0.75) → `proactive_auto`
- FOREST (0.75-1.00) → `proactive_partner`

## 6. PAT ↔ SAT Communication

PAT never accesses the network directly. When PAT needs system resources:

```pseudocode
FUNCTION pat_request_resource(pat_agent: PATAgent, request: ResourceRequest) -> Response:
    """PAT asks SAT for something. SAT decides."""

    # PAT creates a formal request envelope
    envelope = PCIEnvelope(
        sender    = pat_agent.agent_id,
        recipient = "sat-gateway",
        payload   = request,
        ihsan     = pat_agent.compute_ihsan(),
        signature = pat_agent.sign(request),
    )

    # SAT validates before forwarding to URP
    sat_response = sat_gateway.validate_and_forward(envelope)

    IF sat_response.status == APPROVED:
        RETURN sat_response.result
    ELIF sat_response.status == NEEDS_REVIEW:
        # SAT asks user (through PAT) to confirm
        RETURN pat_agent.ask_user_confirmation(sat_response.reason)
    ELSE:
        # Constitutional violation — blocked
        RETURN Error(sat_response.rejection_reason)
```

## 7. Data Sovereignty

PAT data belongs to the USER. Period.

```pseudocode
CLASS PATDataPolicy:
    # User's data never leaves the node without explicit consent
    data_residency     = LOCAL_NODE_ONLY

    # PAT personalization is on-device (TTRL)
    learning_location  = ON_DEVICE

    # Backup is encrypted with user's sovereign key
    backup_encryption  = USER_SOVEREIGN_KEY

    # User can export/delete ALL PAT data at any time
    right_to_export    = True
    right_to_delete    = True
    right_to_port      = True  # Move to another node

    # SAT cannot read PAT's user data
    sat_access         = NONE

    # PAT can read system metrics (via SAT) but not other users' data
    cross_user_access  = NONE
```

## 8. TDD Anchors

```python
class TestPATLifecycle:
    """Phase 54.1: PAT minting and personalization."""

    def test_mint_creates_7_agents(self):
        pat = mint_pat_team(mock_identity())
        assert len(pat.agents) == 7
        assert all(a.mode == AgentMode.REACTIVE for a in pat.agents)

    def test_each_agent_has_unique_role(self):
        pat = mint_pat_team(mock_identity())
        roles = [a.role for a in pat.agents]
        assert len(set(roles)) == 7  # All unique

    def test_initial_trust_score_is_zero(self):
        pat = mint_pat_team(mock_identity())
        assert all(a.trust_score == 0.0 for a in pat.agents)

    def test_trust_escalation_follows_sovereignty_tiers(self):
        assert determine_mode(0.10) == AgentMode.REACTIVE
        assert determine_mode(0.30) == AgentMode.PROACTIVE_SUGGEST
        assert determine_mode(0.60) == AgentMode.PROACTIVE_AUTO
        assert determine_mode(0.80) == AgentMode.PROACTIVE_PARTNER

    def test_pat_data_never_shared_with_sat(self):
        pat = mint_pat_team(mock_identity())
        policy = pat.data_policy
        assert policy.sat_access == DataAccess.NONE

    def test_pat_cannot_access_network_directly(self):
        pat = mint_pat_team(mock_identity())
        with pytest.raises(AccessDenied):
            pat.agents[0].send_to_network(payload)  # Must go through SAT

    def test_personalization_stays_on_device(self):
        pat = mint_pat_team(mock_identity())
        assert pat.data_policy.learning_location == LearningLocation.ON_DEVICE
```
