# 02 — PAT-7 Agent Council

> PAT-7 = Personal Agentic Team. User-owned. Serves mission intent.
> Local loop: Human -> DEMA -> PAT plan/work -> local result + proof trace.

## Agent Roles

```
ENUM PatAgent:
    P1_Atlas      // Planning & decomposition — Sun Tzu, Clausewitz, Porter
    P2_Oracle     // Research & evidence gathering — Shannon, Turing, Dijkstra
    P3_Forge      // Building & execution — Knuth, Ritchie, Torvalds
    P4_Judge      // Evaluation & scoring — Tukey, Tufte, Cleveland
    P5_Crown      // Constitutional verification (user-side) — Al-Ghazali, Rawls
    P6_Herald     // Publishing & delivery — formatting, output
    P7_Nexus      // Coordinator / DEMA / front door persona
```

## Pseudocode: PAT Mission Loop

```
FUNCTION execute_local_mission(pat: PAT7, objective: String) -> MissionResult:
    // P7 Nexus receives and decomposes
    plan = pat.P7_Nexus.decompose(objective)

    // P1 Atlas creates execution strategy
    strategy = pat.P1_Atlas.plan(plan)

    // P2 Oracle gathers evidence
    evidence = pat.P2_Oracle.research(strategy.queries)

    // P3 Forge executes the work
    result = pat.P3_Forge.build(strategy, evidence)

    // P4 Judge evaluates quality
    score = pat.P4_Judge.evaluate(result)
    score.ihsan = compute_ihsan(result)
    score.snr   = compute_snr(result)

    // P5 Crown verifies constitutional compliance (user-side)
    crown_verdict = pat.P5_Crown.verify(result, score)
    IF crown_verdict == VETO:
        RETURN MissionResult::failed("Crown veto: constitutional violation")

    // P6 Herald formats the output
    output = pat.P6_Herald.format(result)

    // Build proof trace
    proof_trace = ProofTrace {
        objective:   objective,
        plan:        plan,
        evidence:    evidence.hashes(),   // not raw data — privacy
        score:       score,
        crown_ok:    crown_verdict,
        model_used:  result.model,
        timestamp:   now(),
    }

    RETURN MissionResult {
        output:      output,
        proof_trace: proof_trace,
        ihsan:       score.ihsan,
        snr:         score.snr,
    }
```

## Pseudocode: DEMA (P7 Nexus) Coordination

```
FUNCTION P7_Nexus.receive(human_input: String) -> Action:
    // DEMA is the front door — every human interaction enters here

    IF is_mission(human_input):
        RETURN Action::ExecuteMission(human_input)
    ELIF is_query(human_input):
        RETURN Action::DelegateToOracle(human_input)
    ELIF is_command(human_input):
        RETURN Action::SystemCommand(human_input)
    ELSE:
        RETURN Action::Chat(human_input)

FUNCTION P7_Nexus.decompose(objective: String) -> Plan:
    // Break complex objective into sub-tasks for PAT agents
    sub_tasks = analyze_objective(objective)

    plan = Plan::new(objective)
    FOR task IN sub_tasks:
        agent = select_best_agent(task)  // Atlas, Oracle, Forge, etc.
        plan.add_step(agent, task)

    // Crown always gets a verification step at the end
    plan.add_step(P5_Crown, "Verify constitutional compliance")

    RETURN plan
```

## Pseudocode: Agent Selection

```
FUNCTION select_best_agent(task: SubTask) -> PatAgent:
    MATCH task.category:
        Planning     => P1_Atlas
        Research     => P2_Oracle
        Building     => P3_Forge
        Evaluation   => P4_Judge
        Verification => P5_Crown
        Publishing   => P6_Herald
        Coordination => P7_Nexus
        _            => P3_Forge  // default to builder
```

## Ownership Contract

```
INVARIANT pat_ownership:
    // PAT agents NEVER communicate with URP directly.
    // PAT agents NEVER bypass FATE boundary.
    // PAT output must be wrapped in proof trace before crossing.

    FOR agent IN pat_agents:
        ASSERT agent.owner == node.identity.public_key
        ASSERT agent.can_access_urp == false
        ASSERT agent.can_access_network == false  // local only
        ASSERT agent.data_scope == LOCAL_ONLY
```

## TDD Anchors

```
TEST pat7_has_exactly_7_agents:
    pat = PAT7::mint(identity)
    ASSERT pat.agents.len() == 7
    ASSERT pat.agents[0].role == P1_Atlas
    ASSERT pat.agents[6].role == P7_Nexus

TEST nexus_receives_all_input:
    pat = PAT7::mint(identity)
    action = pat.P7_Nexus.receive("Build a dashboard")
    ASSERT action IS Action::ExecuteMission

TEST crown_can_veto_mission:
    pat = PAT7::mint(identity)
    // Force a result that violates Ihsan threshold
    result = MockResult { ihsan: 0.50 }
    verdict = pat.P5_Crown.verify(result, result.score)
    ASSERT verdict == VETO

TEST mission_produces_proof_trace:
    pat = PAT7::mint(identity)
    result = pat.execute_local_mission("Test objective")
    ASSERT result.proof_trace IS NOT null
    ASSERT result.proof_trace.objective == "Test objective"
    ASSERT result.proof_trace.ihsan >= 0.0

TEST pat_agents_are_user_owned:
    pat = PAT7::mint(identity)
    FOR agent IN pat.agents:
        ASSERT agent.owner == identity.public_key
        ASSERT agent.can_access_urp == false

TEST decomposition_always_includes_crown:
    plan = P7_Nexus.decompose("Any objective")
    last_step = plan.steps.last()
    ASSERT last_step.agent == P5_Crown
```
