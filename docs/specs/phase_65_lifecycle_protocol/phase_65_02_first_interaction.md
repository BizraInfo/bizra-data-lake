# Phase 65.2: First Interaction — System-2 Deliberation

> Standing on Giants: Boyd (OODA loop, 1976) · Kahneman (System 2 deliberation, 2011) · Besta (Graph of Thoughts, 2024) · Al-Ghazali (Ihsan 8D tensor, 1095)

## 1. Purpose

Execute the first user action through full System-2 deliberation. Every subsystem engages
at maximum depth: TEACH atom extraction, PAT 7-agent consensus, FATE gate verification,
HDA kinetic execution, UIA closed-loop verification, PoI receipt emission, RLVR reward.

**Entry State**: `[ROOTED]` — Temperature T = 2.0 (hot, full exploration)
**Exit State**: `[LEARNING]` — First PoI receipt in BlockGraph, RLVR pattern seed planted
**Target Latency**: < 5 seconds (System-2 full deliberation budget)

---

## 2. Pseudocode

### 2.1 TEACH Atom Extraction

```
FUNCTION extract_teach_atom(user_input: str) -> TeachAtom:
    """Parse user intent into structured TEACH atom."""

    # Source: core/sovereign/mission.py (intent parsing)
    atom = TeachAtom(
        id=generate_uuid("teach_"),
        timestamp=utc_now_iso(),
        raw=user_input,
        structured=parse_intent(user_input),
        energy_potential=estimate_energy(user_input)
    )

    # Store in episodic memory (L1)
    # Source: core/living_memory/core.py
    memory.store_episodic(atom)

    RETURN atom

@dataclass
class TeachAtom:
    id: str                    # "teach_{uuid}"
    timestamp: str             # ISO 8601
    raw: str                   # Original user text
    structured: StructuredIntent
    energy_potential: float    # 0.0 - 1.0

@dataclass
class StructuredIntent:
    action: str                # "ORGANIZE", "SUMMARIZE", etc.
    target: str                # File path or resource
    method: str                # "BY_TOPIC", "BY_DATE", etc.
    modality: str              # "FILE_SYSTEM", "EMAIL", "BROWSER"
```

### 2.2 PAT 7-Agent Deliberation

```
FUNCTION pat_deliberate(
    atom: TeachAtom,
    system_state: SystemState,
    budget_ms: int = 1800
) -> PATConsensus:
    """Launch 7-agent ensemble for System-2 deliberation."""

    # Source: core/pat/ (Protocol Agent Thought architecture)
    # Source: docs/specs/phase_54_pat_sat_architecture/

    agents = [
        PATAgent("Planner",   role="decompose task into steps"),
        PATAgent("Critic",    role="identify risks and edge cases"),
        PATAgent("Ethicist",  role="reversibility and harm analysis"),
        PATAgent("Executor",  role="propose concrete execution script"),
        PATAgent("Verifier",  role="define expected UIA events"),
        PATAgent("Security",  role="FATE gate pre-check"),
        PATAgent("Optimizer", role="estimate cost and latency"),
    ]

    results = []
    FOR agent IN agents:
        response = agent.deliberate(atom, system_state, timeout=budget_ms / 7)
        results.append(response)

    # Consensus: all agents must agree on PROCEED or ABORT
    vetoes = [r FOR r IN results IF r.decision == "VETO"]
    IF vetoes:
        RETURN PATConsensus(decision="ABORT", vetoes=vetoes)

    RETURN PATConsensus(
        decision="PROCEED",
        plan=merge_plans(results),
        estimated_cost=results[-1].estimated_cost,  # Optimizer
        ihsan_pre_score=results[-1].ihsan_pre_score,
        deliberation_time_ms=elapsed()
    )
```

### 2.3 FATE Gate Pre-Execution

```
FUNCTION fate_gate_verify(
    plan: ExecutionPlan,
    system_state: SystemState
) -> FateGateResult:
    """Constitutional verification before any action executes."""

    # Source: core/pci/gates.py (PCIGateKeeper)
    # Source: core/governance/constitutional_gate.py

    # Check 1: Sovereignty validity
    sovereignty_valid = verify_ed25519_signature(
        system_state.identity.public_key,
        plan.signature
    )

    # Check 2: TeleScript permissions
    telescript_allowed = system_state.constitution.telescript.check(
        plan.target_paths
    )

    # Check 3: Lyapunov stability
    lyapunov_delta = compute_lyapunov_delta(system_state, plan)
    lyapunov_stable = (lyapunov_delta <= 0)

    # Check 4: Ihsan 8D tensor pre-score
    ihsan_scores = {}
    FOR dim, threshold IN system_state.constitution.ihsan_dimensions.items():
        score = evaluate_ihsan_dimension(dim, plan)
        ihsan_scores[dim] = score

    all_ihsan_pass = ALL(
        score >= threshold
        FOR dim, (score, threshold)
        IN zip(ihsan_scores.values(),
               system_state.constitution.ihsan_dimensions.values())
    )

    composite_ihsan = mean(ihsan_scores.values())

    # Check 5: IMPT budget
    budget_ok = system_state.impt_balance >= plan.estimated_cost

    # Fail-closed: ALL checks must pass
    allowed = (
        sovereignty_valid
        AND telescript_allowed
        AND lyapunov_stable
        AND all_ihsan_pass
        AND budget_ok
    )

    RETURN FateGateResult(
        allowed=allowed,
        sovereignty_valid=sovereignty_valid,
        telescript_allowed=telescript_allowed,
        lyapunov_delta=lyapunov_delta,
        ihsan_scores=ihsan_scores,
        composite_ihsan=composite_ihsan,
        budget_ok=budget_ok,
        gate_latency_ms=elapsed()
    )
```

### 2.4 HDA Execution

```
FUNCTION hda_execute(
    plan: ExecutionPlan,
    action_bus: ActionBus,
    gate_result: FateGateResult
) -> HDAResult:
    """Kinetic actuation via AHK/PowerShell/UIA."""

    ASSERT gate_result.allowed, "FATE gate must pass before execution"

    # Source: core/sovereign/mission.py (HDAClient)
    # Generate and execute platform-specific script
    script = plan.executor_agent.generate_script(plan)
    script_hash = blake3_hash(script)

    result = action_bus.execute(script)

    RETURN HDAResult(
        script_hash=script_hash,
        files_affected=result.files_affected,
        execution_time_ms=result.elapsed_ms,
        state_changes=result.state_changes
    )
```

### 2.5 UIA Closed-Loop Verification

```
FUNCTION uia_verify(
    pre_state: UIASnapshot,
    post_state: UIASnapshot,
    expected_changes: list[ExpectedChange]
) -> UIAReceipt:
    """Verify physical reality matches expected outcomes."""

    # Source: core/bridges/desktop_bridge.py (accessibility tree)
    diffs = compute_state_diff(pre_state, post_state)

    verified = ALL(
        change.satisfied_by(diffs)
        FOR change IN expected_changes
    )

    RETURN UIAReceipt(
        verified=verified,
        confidence=1.0 IF verified ELSE compute_partial_confidence(diffs),
        diffs=diffs,
        verification_latency_ms=elapsed()
    )
```

### 2.6 PoI Receipt Emission

```
FUNCTION poi_emit(
    atom: TeachAtom,
    hda_result: HDAResult,
    uia_receipt: UIAReceipt,
    gate_result: FateGateResult,
    system_state: SystemState
) -> PoIReceipt:
    """Emit signed, hash-chained Proof-of-Impact receipt."""

    # Source: core/proof_engine/evidence_ledger.py
    receipt = {
        "action_id": generate_uuid("act_"),
        "user_intent": atom.id,
        "timestamp": utc_now_iso(),
        "execution": {
            "method": hda_result.method,
            "latency_ms": hda_result.execution_time_ms,
            "script_hash": hda_result.script_hash
        },
        "verification": {
            "uia_confirmed": uia_receipt.verified,
            "uia_confidence": uia_receipt.confidence,
            "files_affected": hda_result.files_affected,
        },
        "governance": {
            "fate_gate_decision": "ALLOW",
            "ihsan_score": gate_result.composite_ihsan,
            "lyapunov_delta": gate_result.lyapunov_delta,
        },
        "reason_codes": ["FIRST_INTERACTION", "SYSTEM_2_FULL"]
    }

    # Sign with persistent node signer
    receipt_hash = blake3_hash(json.dumps(receipt, sort_keys=True))
    signature = ed25519_sign(system_state.signer_private_key, receipt_hash)
    receipt["signature"] = signature
    receipt["hash"] = receipt_hash

    # Append to BlockGraph
    system_state.ledger.append(receipt=receipt)

    RETURN PoIReceipt(receipt)
```

### 2.7 RLVR Reward Signal

```
FUNCTION rlvr_reward(
    uia_receipt: UIAReceipt,
    gate_result: FateGateResult,
    hda_result: HDAResult,
    budget_ms: int
) -> RLVRReward:
    """Compute verifiable reward from action outcome."""

    # Components
    uia_success = 1.0 IF uia_receipt.verified ELSE 0.0
    ihsan_bonus = gate_result.composite_ihsan
    efficiency = 0.1 IF hda_result.execution_time_ms < budget_ms ELSE 0.0

    total_reward = uia_success + ihsan_bonus + efficiency

    # Record in episodic memory for future reflex compilation
    pattern = ActionPattern(
        intent=atom.structured,
        execution=hda_result.method,
        reward=total_reward,
        verified=uia_receipt.verified
    )

    RETURN RLVRReward(
        total=total_reward,
        pattern=pattern,
        pattern_count=1  # First occurrence
    )
```

### 2.8 Complete First Interaction Flow

```
FUNCTION first_interaction(
    user_input: str,
    system_state: SystemState
) -> InteractionResult:
    """Full System-2 execution path for first user action."""

    # Phase 1: Extract intent
    atom = extract_teach_atom(user_input)

    # Phase 2: PAT deliberation (1800ms budget)
    consensus = pat_deliberate(atom, system_state)
    IF consensus.decision != "PROCEED":
        RETURN InteractionResult(success=False, reason=consensus.vetoes)

    # Phase 3: FATE gate verification
    gate = fate_gate_verify(consensus.plan, system_state)
    IF NOT gate.allowed:
        RETURN InteractionResult(success=False, reason="FATE_VETO")

    # Phase 4: Capture pre-state, execute, capture post-state
    pre_state = UIA.snapshot(consensus.plan.target_paths)
    hda_result = hda_execute(consensus.plan, system_state.hda.action_bus, gate)
    post_state = UIA.snapshot(consensus.plan.target_paths)

    # Phase 5: Verify
    uia_receipt = uia_verify(pre_state, post_state, consensus.plan.expected_changes)

    # Phase 6: Emit receipt
    poi = poi_emit(atom, hda_result, uia_receipt, gate, system_state)

    # Phase 7: RLVR reward
    reward = rlvr_reward(uia_receipt, gate, hda_result, budget_ms=1500)

    # Update system state
    system_state.impt_balance += reward.total
    system_state.epistemic_entropy -= ENTROPY_REDUCTION_PER_ACTION
    system_state.state = "LEARNING"

    RETURN InteractionResult(
        success=True,
        poi=poi,
        reward=reward,
        total_latency_ms=elapsed()
    )
```

---

## 3. Performance Budget (System-2)

```
| Subsystem          | Budget (ms) | Typical (ms) |
|--------------------|-------------|--------------|
| TEACH extraction   | 50          | 20           |
| PAT deliberation   | 1800        | 1650         |
| FATE gate          | 100         | 45           |
| HDA execution      | 1500        | 1180         |
| UIA verification   | 200         | 85           |
| PoI emission       | 200         | 120          |
| RLVR computation   | 50          | 15           |
| TOTAL              | 3900        | 3115         |
```

---

## 4. TDD Anchors

### Existing Tests
- `tests/core/sovereign/test_mission.py` — 37 tests covering mission pipeline
- `tests/core/sovereign/test_hardening_track1.py` — auth guards, threshold, signer
- `tests/core/pci/test_gates.py` — FATE gate verification
- `tests/core/proof_engine/test_evidence_ledger.py` — receipt chain integrity

### New Tests Required

```python
# tests/core/sovereign/test_lifecycle_interaction.py

class TestFirstInteraction:

    def test_teach_atom_has_structured_intent(self):
        """TEACH atom parses action, target, method from input."""
        atom = extract_teach_atom("Organize my research papers by topic")
        assert atom.structured.action == "ORGANIZE"
        assert atom.structured.method == "BY_TOPIC"

    def test_pat_consensus_requires_all_agents(self):
        """PAT deliberation uses exactly 7 agents."""
        consensus = pat_deliberate(mock_atom, mock_state)
        assert len(consensus.agent_responses) == 7

    def test_fate_veto_blocks_execution(self):
        """If FATE gate returns allowed=False, no HDA execution occurs."""
        gate = fate_gate_verify(bad_plan, mock_state)
        assert not gate.allowed
        # HDA should never be called

    def test_uia_verifies_state_change(self):
        """UIA receipt confirms physical state matches expectations."""
        receipt = uia_verify(pre, post, expected)
        assert receipt.verified is True
        assert receipt.confidence == 1.0

    def test_poi_receipt_is_signed(self):
        """PoI receipt contains valid Ed25519 signature."""
        poi = poi_emit(atom, hda, uia, gate, state)
        assert verify_signature(state.identity.public_key, poi.hash, poi.signature)

    def test_rlvr_reward_positive_on_success(self):
        """Successful action produces reward > 0."""
        reward = rlvr_reward(verified_uia, passing_gate, fast_hda, 1500)
        assert reward.total > 0

    def test_first_interaction_transitions_to_learning(self):
        """After first successful interaction, state becomes LEARNING."""
        result = first_interaction("organize files", rooted_state)
        assert result.success
        assert rooted_state.state == "LEARNING"
```

---

## 5. Error Handling

```
ON PAT deliberation timeout:
    Extend budget by 50%, retry once
    IF still timeout: return ABORT with "deliberation_timeout"

ON FATE gate veto:
    Log veto reason, return to user with explanation
    Never attempt to bypass gate — this is constitutional

ON HDA execution failure:
    UIA will detect no state change
    Receipt records failure with uia_confirmed=False
    Reward = 0 (no positive reinforcement for failed actions)

ON UIA verification partial match:
    Record confidence < 1.0
    Still emit receipt (honesty over omission)
    RLVR reward proportional to confidence
```
