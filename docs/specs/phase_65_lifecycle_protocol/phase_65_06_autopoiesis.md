# Phase 65.6: Autopoietic Self-Modification

> Standing on Giants: Maturana & Varela (autopoiesis, 1972) · Deming (continuous improvement, 1950) · Al-Ghazali (consent-gated ethics, 1095)

## 1. Purpose

After 40+ days of operation, the system identifies optimization opportunities and
proposes self-modifications. Unlike reactive myelination (Phase 65.4), autopoiesis
is **proactive** — the system detects patterns that COULD be compiled before the user
explicitly repeats them enough times. All self-modifications require explicit user
consent and FATE gate verification.

**Entry State**: `[MYELINATED]` or `[FLOURISHING]` — 7+ reflexes, T < 0.8
**Exit State**: `[FLOURISHING]` — New capabilities added with user consent
**Trigger**: System detects high-frequency System-2 patterns (> 15 occurrences)

---

## 2. Pseudocode

### 2.1 Opportunity Detection

```
FUNCTION detect_optimization_opportunities(
    pattern_registry: PatternRegistry,
    reflex_registry: ReflexRegistry,
    system_state: SystemState
) -> list[OptimizationProposal]:
    """Scan for System-2 patterns worth compiling."""

    AUTOPOIESIS_THRESHOLD = 15  # More occurrences than standard myelination
    MIN_AVG_REWARD = 1.5        # Must be a valuable pattern

    proposals = []

    FOR pattern IN pattern_registry.patterns.values():
        # Skip already-compiled patterns
        IF any(r.source_pattern == pattern.pattern_id
               FOR r IN reflex_registry.reflexes.values()):
            CONTINUE

        # Check if pattern is worth autopoietic compilation
        IF (
            pattern.occurrences >= AUTOPOIESIS_THRESHOLD
            AND pattern.successes / pattern.occurrences >= MIN_SUCCESS_RATE
            AND pattern.total_reward / pattern.successes >= MIN_AVG_REWARD
        ):
            # Estimate improvement
            current_avg_latency = pattern.avg_latency_ms
            estimated_reflex_latency = current_avg_latency / 8.0  # ~8x speedup

            proposal = OptimizationProposal(
                pattern=pattern,
                current_latency_ms=current_avg_latency,
                estimated_latency_ms=estimated_reflex_latency,
                speedup_factor=current_avg_latency / estimated_reflex_latency,
                impt_cost=estimate_compilation_cost(pattern),
                total_time_savings_ms=compute_projected_savings(pattern),
            )
            proposals.append(proposal)

    # Sort by projected value (time savings / IMPT cost)
    proposals.sort(key=LAMBDA p: p.total_time_savings_ms / p.impt_cost, reverse=True)

    RETURN proposals
```

### 2.2 User Consent Gate

```
FUNCTION request_user_consent(
    proposal: OptimizationProposal,
    system_state: SystemState
) -> ConsentResult:
    """Present proposal to user and await explicit consent."""

    # CRITICAL: Self-modification requires user approval
    # Source: core/governance/constitutional_gate.py

    presentation = {
        "type": "SELF_MODIFICATION_PROPOSAL",
        "pattern": proposal.pattern.action_type,
        "occurrences": proposal.pattern.occurrences,
        "current_speed_ms": proposal.current_latency_ms,
        "proposed_speed_ms": proposal.estimated_latency_ms,
        "speedup": f"{proposal.speedup_factor:.0f}x faster",
        "cost_impt": proposal.impt_cost,
        "balance_after": system_state.impt_balance - proposal.impt_cost,
        "safety_note": "Full FATE gate checks still apply.",
        "options": ["Approve", "Reject", "Customize"]
    }

    # Display to user (blocks until response)
    user_response = present_consent_dialog(presentation)

    IF user_response.choice == "Approve":
        # Sign consent with user's Ed25519 key
        consent_hash = blake3_hash(json.dumps(presentation, sort_keys=True))
        signature = ed25519_sign(
            system_state.identity.private_key,
            consent_hash
        )
        RETURN ConsentResult(
            approved=True,
            signature=signature,
            consent_hash=consent_hash
        )
    ELSE:
        RETURN ConsentResult(approved=False)
```

### 2.3 Self-Directed RLVR Training

```
FUNCTION self_directed_training(
    pattern: ActionPattern,
    system_state: SystemState
) -> TrainingResult:
    """Train new reflex on historical verified examples."""

    # Source: core/sovereign/mission.py (historical receipts)
    # Retrieve all PoI receipts for this pattern
    historical_receipts = system_state.ledger.query(
        pattern_id=pattern.pattern_id,
        status="verified"
    )

    # Validate: all receipts have UIA confirmation
    valid_receipts = [
        r FOR r IN historical_receipts
        IF r["verification"]["uia_confirmed"]
    ]

    IF len(valid_receipts) < COMPILATION_THRESHOLD:
        RAISE InsufficientTrainingDataError()

    # Compute convergence metrics
    rewards = [r.reward FOR r IN valid_receipts]
    avg_reward = mean(rewards)
    success_rate = len(valid_receipts) / len(historical_receipts)

    RETURN TrainingResult(
        examples=len(valid_receipts),
        avg_reward=avg_reward,
        success_rate=success_rate,
        converged=True
    )
```

### 2.4 Autopoiesis Orchestrator

```
FUNCTION autopoiesis_cycle(
    system_state: SystemState,
    pattern_registry: PatternRegistry,
    reflex_registry: ReflexRegistry
) -> AutopoiesisResult:
    """Full autopoietic cycle: detect → propose → consent → compile."""

    # Step 1: Detect opportunities
    proposals = detect_optimization_opportunities(
        pattern_registry, reflex_registry, system_state
    )
    IF NOT proposals:
        RETURN AutopoiesisResult(action="NONE", reason="No optimization opportunities")

    # Step 2: Pick top proposal
    proposal = proposals[0]

    # Step 3: FATE gate pre-check on self-modification
    fate_check = fate_gate_verify_self_modification(proposal, system_state)
    IF NOT fate_check.allowed:
        RETURN AutopoiesisResult(action="BLOCKED", reason="FATE veto on self-modification")

    # Step 4: Request user consent
    consent = request_user_consent(proposal, system_state)
    IF NOT consent.approved:
        RETURN AutopoiesisResult(action="REJECTED", reason="User declined")

    # Step 5: Self-directed training
    training = self_directed_training(proposal.pattern, system_state)

    # Step 6: Compile reflex (reuses Phase 65.4 compiler)
    reflex = compile_reflex(
        proposal.pattern,
        system_state,
        impt_cost=proposal.impt_cost
    )
    register_reflex(reflex_registry, reflex)

    # Step 7: Emit autopoiesis receipt
    receipt = {
        "type": "AUTOPOIESIS",
        "reflex_id": reflex.reflex_id,
        "pattern_occurrences": proposal.pattern.occurrences,
        "user_consent_hash": consent.consent_hash,
        "user_consent_signature": consent.signature,
        "training_examples": training.examples,
        "impt_cost": proposal.impt_cost,
        "reason_codes": ["SELF_MODIFICATION", "USER_CONSENTED"]
    }
    system_state.ledger.append(receipt=receipt)

    # Step 8: Update state
    system_state.reflexes_compiled += 1
    IF system_state.state != "FLOURISHING":
        system_state.state = "FLOURISHING"

    RETURN AutopoiesisResult(
        action="COMPILED",
        reflex=reflex,
        consent=consent,
        training=training
    )
```

---

## 3. Data Structures

```
@dataclass
class OptimizationProposal:
    pattern: ActionPattern
    current_latency_ms: float
    estimated_latency_ms: float
    speedup_factor: float
    impt_cost: float
    total_time_savings_ms: float       # Projected over next 30 days

@dataclass
class ConsentResult:
    approved: bool
    signature: str | None = None       # Ed25519 signature of consent
    consent_hash: str | None = None    # BLAKE3 of proposal content

@dataclass
class TrainingResult:
    examples: int
    avg_reward: float
    success_rate: float
    converged: bool

@dataclass
class AutopoiesisResult:
    action: str                        # "COMPILED", "REJECTED", "BLOCKED", "NONE"
    reason: str | None = None
    reflex: CompiledReflex | None = None
    consent: ConsentResult | None = None
    training: TrainingResult | None = None
```

---

## 4. Constitutional Constraint

```
INVARIANT: Self-modification is consent-gated.

The system MAY propose optimizations.
The system MUST NOT modify itself without user approval.
The user's consent MUST be cryptographically signed.
The consent receipt MUST be stored in the BlockGraph.

This is the autopoietic version of the Daughter Test:
  "Would I be comfortable if my daughter's agent modified itself without her knowing?"
  Answer: No. Therefore, consent is mandatory.
```

---

## 5. TDD Anchors

### New Tests Required

```python
# tests/core/sovereign/test_lifecycle_autopoiesis.py

class TestOpportunityDetection:

    def test_detects_high_frequency_uncompiled_patterns(self):
        """Patterns with 15+ occurrences and no reflex are detected."""
        registry = make_registry_with_pattern(occurrences=20, successes=18)
        proposals = detect_optimization_opportunities(
            registry, make_empty_reflex_registry(), make_state()
        )
        assert len(proposals) == 1

    def test_ignores_already_compiled_patterns(self):
        """Patterns with existing reflexes are skipped."""
        p_reg = make_registry_with_pattern(occurrences=20, successes=18)
        r_reg = make_reflex_registry_for_pattern(p_reg)
        proposals = detect_optimization_opportunities(p_reg, r_reg, make_state())
        assert len(proposals) == 0

    def test_proposals_sorted_by_value(self):
        """Proposals are sorted by time_savings / cost ratio."""
        proposals = detect_optimization_opportunities(
            make_multi_pattern_registry(), make_empty_reflex_registry(), make_state()
        )
        values = [p.total_time_savings_ms / p.impt_cost FOR p IN proposals]
        assert values == sorted(values, reverse=True)


class TestUserConsent:

    def test_approved_consent_has_signature(self):
        """Approved consent includes Ed25519 signature."""
        consent = mock_user_approves(proposal, state)
        assert consent.approved
        assert consent.signature is not None
        assert consent.consent_hash is not None

    def test_rejected_consent_has_no_signature(self):
        """Rejected consent has no signature."""
        consent = mock_user_rejects(proposal, state)
        assert not consent.approved
        assert consent.signature is None


class TestAutopoiesisOrchestrator:

    def test_full_cycle_compiles_reflex(self):
        """Complete autopoiesis cycle produces new reflex."""
        result = autopoiesis_cycle(state, pattern_reg, reflex_reg)
        assert result.action == "COMPILED"
        assert result.reflex is not None

    def test_cycle_emits_consent_receipt(self):
        """Autopoiesis receipt includes user consent hash."""
        result = autopoiesis_cycle(state, pattern_reg, reflex_reg)
        last = state.ledger.last()
        assert last["type"] == "AUTOPOIESIS"
        assert "user_consent_hash" in last

    def test_rejected_proposal_does_not_compile(self):
        """User rejection stops compilation."""
        result = autopoiesis_cycle_with_rejection(state, pattern_reg, reflex_reg)
        assert result.action == "REJECTED"
        assert state.reflexes_compiled == 0  # No change

    def test_state_transitions_to_flourishing(self):
        """First autopoiesis transitions state to FLOURISHING."""
        state = make_state(state="MYELINATED")
        autopoiesis_cycle(state, pattern_reg, reflex_reg)
        assert state.state == "FLOURISHING"
```
