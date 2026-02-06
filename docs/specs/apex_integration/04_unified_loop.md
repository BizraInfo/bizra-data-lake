# Unified OODA Loop Specification

## Phase 4: Extended OODA with Apex Integration

**Module**: `core/sovereign/apex_sovereign.py`
**Dependencies**: All Phase 1-3 modules, `core/sovereign/autonomy.py`
**Lines**: ~300

---

## Functional Requirements

### FR-1: Apex-Enhanced OODA

Extend the 9-state OODA loop to incorporate Apex inputs:
- **OBSERVE**: Include social, market, and swarm health readings
- **PREDICT**: Use market signals for trend prediction
- **COORDINATE**: Use social graph for team coordination
- **DECIDE**: Factor in autonomy levels from market signals
- **ACT**: Execute via hybrid swarm
- **LEARN**: Update social trust based on outcomes

### FR-2: Constitutional Compliance

All Apex-enhanced decisions must pass Ihsan validation:
- Social routing: Ihsan ≥ 0.95
- Market actions: Ihsan ≥ 0.95 + SNR ≥ 0.85
- Swarm scaling: Ihsan ≥ 0.95

### FR-3: Graceful Degradation

If Apex subsystems fail:
- Social: Fall back to capability-only routing
- Market: Disable autonomous trading, suggest-only mode
- Swarm: Fixed-size Python-only operation

---

## Pseudocode

```
MODULE apex_sovereign

IMPORT ApexSystem FROM core.apex
IMPORT SociallyAwareBridge FROM core.sovereign.social_integration
IMPORT MarketAwareMuraqabah FROM core.sovereign.market_integration
IMPORT HybridSwarmOrchestrator FROM core.sovereign.swarm_integration
IMPORT OODAState, AutonomyLevel FROM core.sovereign.autonomy
IMPORT IhsanValidator FROM core.sovereign.ihsan_projector

# Extended OODA States (Boyd + Apex)
ENUM ApexOODAState:
    OBSERVE = "observe"         # Collect all sensor data
    PREDICT = "predict"         # Forecast trends (market)
    COORDINATE = "coordinate"   # Team planning (social)
    ANALYZE = "analyze"         # PAT analysis
    DECIDE = "decide"           # Autonomy-based decision
    ACT = "act"                 # Execute via swarm
    LEARN = "learn"             # Update models
    REFLECT = "reflect"         # Metrics and improvement
    SLEEP = "sleep"             # Low-power monitoring

CLASS ApexSovereignEntity:
    """
    Proactive Sovereign Entity with full Apex integration.

    Combines:
    - Extended OODA loop (Boyd)
    - Social intelligence (Granovetter/PageRank)
    - Market intelligence (Shannon/Markowitz)
    - Swarm orchestration (Borg/K8s)
    - Constitutional AI (Ihsan)

    Standing on Giants:
    - Boyd (1995): OODA decision cycle
    - Shannon (1948): SNR information theory
    - Granovetter (1973): Social network dynamics
    - Lamport (1982): Distributed consensus
    - Anthropic: Constitutional AI principles
    """

    # Core components
    PROPERTY apex: ApexSystem
    PROPERTY social_bridge: SociallyAwareBridge
    PROPERTY market_muraqabah: MarketAwareMuraqabah
    PROPERTY swarm: HybridSwarmOrchestrator
    PROPERTY ihsan: IhsanValidator

    # State
    PROPERTY current_state: ApexOODAState = ApexOODAState.SLEEP
    PROPERTY cycle_count: INT = 0
    PROPERTY metrics: Dict[str, float]

    # Configuration
    PROPERTY ihsan_threshold: FLOAT = 0.95
    PROPERTY snr_floor: FLOAT = 0.85
    PROPERTY cycle_interval_ms: INT = 1000

    CONSTRUCTOR(self, node_id: STRING):
        self.node_id = node_id

        # Initialize Apex system
        self.apex = ApexSystem(
            node_id=node_id,
            ihsan_threshold=self.ihsan_threshold,
            snr_floor=self.snr_floor
        )

        # Initialize integrated components
        self.social_bridge = SociallyAwareBridge(node_id)
        self.market_muraqabah = MarketAwareMuraqabah(node_id)
        self.swarm = HybridSwarmOrchestrator()
        self.ihsan = IhsanValidator(threshold=self.ihsan_threshold)

        # Metrics
        self.metrics = {
            "cycles": 0,
            "actions_taken": 0,
            "autonomous_actions": 0,
            "ihsan_average": 0.0,
            "snr_average": 0.0,
        }

    ASYNC METHOD start(self):
        """Start the Apex Sovereign Entity."""
        LOG.info(f"Starting ApexSovereignEntity: {self.node_id}")

        # Start Apex subsystems
        AWAIT self.apex.start()

        # Start integrated components
        AWAIT self.social_bridge.start()
        AWAIT self.market_muraqabah.start_monitoring()
        AWAIT self.swarm.start()

        # Begin OODA loop
        self.current_state = ApexOODAState.OBSERVE
        AWAIT self._run_ooda_loop()

    ASYNC METHOD _run_ooda_loop(self):
        """Main OODA loop with Apex integration."""
        WHILE self._running:
            TRY:
                self.cycle_count += 1

                # State machine
                MATCH self.current_state:
                    CASE ApexOODAState.OBSERVE:
                        observations = AWAIT self._observe()
                        self.current_state = ApexOODAState.PREDICT

                    CASE ApexOODAState.PREDICT:
                        predictions = AWAIT self._predict(observations)
                        self.current_state = ApexOODAState.COORDINATE

                    CASE ApexOODAState.COORDINATE:
                        team_plan = AWAIT self._coordinate(observations, predictions)
                        self.current_state = ApexOODAState.ANALYZE

                    CASE ApexOODAState.ANALYZE:
                        analysis = AWAIT self._analyze(observations, predictions, team_plan)
                        self.current_state = ApexOODAState.DECIDE

                    CASE ApexOODAState.DECIDE:
                        decisions = AWAIT self._decide(analysis)
                        self.current_state = ApexOODAState.ACT

                    CASE ApexOODAState.ACT:
                        outcomes = AWAIT self._act(decisions)
                        self.current_state = ApexOODAState.LEARN

                    CASE ApexOODAState.LEARN:
                        AWAIT self._learn(outcomes)
                        self.current_state = ApexOODAState.REFLECT

                    CASE ApexOODAState.REFLECT:
                        AWAIT self._reflect()
                        self.current_state = ApexOODAState.OBSERVE

                # Throttle loop
                AWAIT asyncio.sleep(self.cycle_interval_ms / 1000)

            EXCEPT Exception AS e:
                LOG.error(f"OODA cycle error: {e}")
                AWAIT self._handle_cycle_error(e)

    ASYNC METHOD _observe(self) -> Dict:
        """
        OBSERVE phase: Collect all sensor readings.

        Sources:
        - Muraqabah sensors (health, financial, social, cognitive, environmental)
        - Market sensors (signals, arbitrage)
        - Swarm health (Python agents, Rust services)
        - Social graph (relationship changes)
        """
        observations = {
            "muraqabah": AWAIT self.market_muraqabah.scan_all_domains(),
            "swarm_health": AWAIT self.swarm.health_monitor.check_all(),
            "social_changes": self.apex.social.get_recent_changes(),
            "market_signals": self.apex.opportunity.signal_generator.get_active_signals(),
            "timestamp": datetime.now(timezone.utc),
        }

        # Log observation summary
        LOG.debug(f"Observed: {len(observations['muraqabah'])} readings, "
                  f"{len(observations['market_signals'])} signals")

        RETURN observations

    ASYNC METHOD _predict(self, observations: Dict) -> Dict:
        """
        PREDICT phase: Forecast trends from observations.

        Uses market analysis for trend prediction.
        """
        predictions = {
            "market_trends": {},
            "workload_forecast": None,
            "relationship_trends": {},
        }

        # Market trend predictions
        FOR signal IN observations.get("market_signals", []):
            IF signal.snr_score >= self.snr_floor:
                predictions["market_trends"][signal.symbol] = {
                    "direction": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "expected_return": signal.expected_return,
                }

        # Workload forecast from swarm metrics
        IF observations.get("swarm_health"):
            cpu_trend = self._calculate_cpu_trend(observations["swarm_health"])
            predictions["workload_forecast"] = {
                "direction": "up" IF cpu_trend > 0 ELSE "down",
                "magnitude": ABS(cpu_trend),
            }

        RETURN predictions

    ASYNC METHOD _coordinate(self, observations: Dict, predictions: Dict) -> Dict:
        """
        COORDINATE phase: Plan team activities using social intelligence.

        Uses SocialGraph for:
        - Finding collaboration partners
        - Trust-based task routing
        - Avoiding overloaded agents
        """
        team_plan = {
            "task_assignments": [],
            "collaborations": [],
            "scaling_recommendation": None,
        }

        # Find collaboration opportunities
        collab_opportunities = self.apex.social.find_collaborations()
        FOR opp IN collab_opportunities[:5]:  # Top 5 opportunities
            team_plan["collaborations"].APPEND({
                "agents": [opp.agent_id, opp.partner_id],
                "synergy": opp.synergy_score,
                "task_type": opp.recommended_task_type,
            })

        # Scaling recommendation based on predictions
        IF predictions.get("workload_forecast", {}).get("direction") == "up":
            team_plan["scaling_recommendation"] = ScalingAction.SCALE_UP

        RETURN team_plan

    ASYNC METHOD _decide(self, analysis: Dict) -> List[Decision]:
        """
        DECIDE phase: Make decisions based on analysis and autonomy levels.

        Constitutional compliance:
        - All decisions validated against Ihsan
        - Autonomy level determines approval requirement
        """
        decisions = []

        FOR goal IN analysis.get("goals", []):
            # Ihsan validation
            ihsan_score = self.ihsan.validate(goal)

            IF ihsan_score < self.ihsan_threshold:
                LOG.warning(f"Goal failed Ihsan: {goal.description}, score={ihsan_score}")
                CONTINUE

            # Determine if autonomous or requires approval
            autonomy_level = goal.autonomy_level
            requires_approval = autonomy_level <= AutonomyLevel.SUGGESTER

            decision = Decision(
                goal=goal,
                ihsan_score=ihsan_score,
                autonomy_level=autonomy_level,
                requires_approval=requires_approval,
                approved=NOT requires_approval,  # Auto-approve high autonomy
            )
            decisions.APPEND(decision)

        # Update metrics
        self.metrics["ihsan_average"] = (
            self.metrics["ihsan_average"] * 0.9 +
            MEAN([d.ihsan_score FOR d IN decisions]) * 0.1
        ) IF decisions ELSE self.metrics["ihsan_average"]

        RETURN decisions

    ASYNC METHOD _act(self, decisions: List[Decision]) -> List[Outcome]:
        """
        ACT phase: Execute approved decisions via hybrid swarm.

        Uses:
        - SociallyAwareBridge for agent selection
        - HybridSwarmOrchestrator for execution
        """
        outcomes = []

        FOR decision IN decisions:
            IF NOT decision.approved:
                CONTINUE

            # Select agents using social trust
            agents = self.social_bridge.select_agent_for_task(decision.goal.as_task())

            # Execute via swarm
            result = AWAIT self.swarm.execute_task(
                task=decision.goal.as_task(),
                agents=agents
            )

            outcome = Outcome(
                decision=decision,
                success=result.success,
                value=result.value,
                agents_used=agents,
            )
            outcomes.APPEND(outcome)

            # Update metrics
            self.metrics["actions_taken"] += 1
            IF decision.autonomy_level >= AutonomyLevel.AUTOLOW:
                self.metrics["autonomous_actions"] += 1

        RETURN outcomes

    ASYNC METHOD _learn(self, outcomes: List[Outcome]):
        """
        LEARN phase: Update models based on outcomes.

        Updates:
        - Social trust scores (success/failure)
        - Market signal accuracy tracking
        - Swarm performance metrics
        """
        FOR outcome IN outcomes:
            # Update social trust
            FOR agent IN outcome.agents_used:
                AWAIT self.social_bridge.report_task_outcome(
                    task=outcome.decision.goal.as_task(),
                    agent=agent,
                    success=outcome.success,
                    value=outcome.value
                )

            # Track signal accuracy if market-related
            IF outcome.decision.goal.domain == "financial":
                self.apex.opportunity.track_signal_outcome(
                    signal_id=outcome.decision.goal.source_signal_id,
                    actual_return=outcome.value
                )

    ASYNC METHOD _reflect(self):
        """
        REFLECT phase: Evaluate performance and adjust parameters.

        Checks:
        - Ihsan average trending
        - Autonomous action success rate
        - Swarm efficiency
        """
        self.metrics["cycles"] = self.cycle_count

        # Log cycle summary every 10 cycles
        IF self.cycle_count % 10 == 0:
            LOG.info(f"Cycle {self.cycle_count} Summary: "
                     f"actions={self.metrics['actions_taken']}, "
                     f"autonomous={self.metrics['autonomous_actions']}, "
                     f"ihsan_avg={self.metrics['ihsan_average']:.3f}")

        # Adjust parameters if needed
        IF self.metrics["ihsan_average"] < self.ihsan_threshold - 0.05:
            LOG.warning("Ihsan average dropping, reducing autonomy")
            # Reduce autonomous actions until Ihsan recovers
            self.snr_floor = MIN(self.snr_floor + 0.02, 0.95)

---

## TDD Anchors

### Test: Full OODA Cycle

```python
async def test_full_ooda_cycle():
    """Entity should complete all OODA states."""
    entity = ApexSovereignEntity(node_id="test")
    states_visited = []

    def track_state(state):
        states_visited.append(state)

    entity.on_state_change = track_state

    # Run one complete cycle
    await entity._run_one_cycle()

    expected_states = [
        ApexOODAState.OBSERVE,
        ApexOODAState.PREDICT,
        ApexOODAState.COORDINATE,
        ApexOODAState.ANALYZE,
        ApexOODAState.DECIDE,
        ApexOODAState.ACT,
        ApexOODAState.LEARN,
        ApexOODAState.REFLECT,
    ]
    assert states_visited == expected_states
```

### Test: Ihsan Filtering

```python
async def test_low_ihsan_goals_filtered():
    """Goals below Ihsan threshold should not become decisions."""
    entity = ApexSovereignEntity(node_id="test")

    analysis = {
        "goals": [
            ProactiveGoal(description="Good goal", ihsan_estimate=0.98),
            ProactiveGoal(description="Bad goal", ihsan_estimate=0.80),
        ]
    }

    decisions = await entity._decide(analysis)

    # Only good goal should pass
    assert len(decisions) == 1
    assert decisions[0].goal.description == "Good goal"
```

### Test: Graceful Degradation

```python
async def test_apex_failure_degrades_gracefully():
    """If Apex subsystem fails, entity should continue in degraded mode."""
    entity = ApexSovereignEntity(node_id="test")

    # Simulate market subsystem failure
    entity.apex._opportunity = None

    # Should still observe (without market data)
    observations = await entity._observe()

    assert "muraqabah" in observations
    assert observations["market_signals"] == []  # Empty but not error
```

---

## State Diagram

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
              ┌──────────┐                                    │
    START────▶│ OBSERVE  │◀───────────────────────────────────┤
              └────┬─────┘                                    │
                   │ Collect sensor data                      │
                   ▼                                          │
              ┌──────────┐                                    │
              │ PREDICT  │                                    │
              └────┬─────┘                                    │
                   │ Forecast trends                          │
                   ▼                                          │
              ┌────────────┐                                  │
              │ COORDINATE │                                  │
              └────┬───────┘                                  │
                   │ Team planning                            │
                   ▼                                          │
              ┌──────────┐                                    │
              │ ANALYZE  │                                    │
              └────┬─────┘                                    │
                   │ PAT analysis                             │
                   ▼                                          │
              ┌──────────┐                                    │
              │ DECIDE   │──────┐                             │
              └────┬─────┘      │ Ihsan < threshold           │
                   │            ▼                             │
                   │       ┌──────────┐                       │
                   │       │ REJECT   │                       │
                   │       └──────────┘                       │
                   │ Approved                                 │
                   ▼                                          │
              ┌──────────┐                                    │
              │   ACT    │                                    │
              └────┬─────┘                                    │
                   │ Execute via swarm                        │
                   ▼                                          │
              ┌──────────┐                                    │
              │  LEARN   │                                    │
              └────┬─────┘                                    │
                   │ Update models                            │
                   ▼                                          │
              ┌──────────┐                                    │
              │ REFLECT  │────────────────────────────────────┘
              └──────────┘
                   │ Adjust parameters
                   ▼
               (loop back to OBSERVE)
```

---

## File Output

**Target**: `core/sovereign/apex_sovereign.py`
**Lines**: ~300
**Imports**:
```python
from core.apex import ApexSystem
from core.sovereign.social_integration import SociallyAwareBridge
from core.sovereign.market_integration import MarketAwareMuraqabah
from core.sovereign.swarm_integration import HybridSwarmOrchestrator
from core.sovereign.autonomy import OODAState, AutonomyLevel
from core.sovereign.ihsan_projector import IhsanValidator
```
