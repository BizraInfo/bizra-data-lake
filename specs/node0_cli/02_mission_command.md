# Phase 2: The Mission Command
## `bizra mission "<objective>"` — The Primary Command
### References: 00_cli_master_spec.md §4, §6

---

## 1. Design Principle

**Objective over prompts.** The user declares an objective.
The system decomposes, executes, gates, proves, and remembers.

The mission command IS the product loop:
**work → proof → memory → reflex → reward → sovereignty**

## 2. Command Grammar

```
bizra mission "<objective>"              # Execute mission
bizra mission --status                    # Show active mission
bizra mission --history [n]               # Last n missions
bizra mission --warm "<objective>"        # Force warm path
bizra mission --dry-run "<objective>"     # Plan without execute
```

## 3. Pseudocode: Full Governed Loop

```
COMMAND bizra_mission(objective: str, warm: bool = false, dry_run: bool = false):
    node_state = load_node_state()
    ASSERT node_state.genesis_complete, "Run 'bizra genesis' first"

    # ═══════════════════════════════════════════════
    # PHASE 1: ENVELOPE — Package the mission
    # ═══════════════════════════════════════════════

    envelope = MissionEnvelope(
        mission_id=generate_mission_id(),
        objective=objective,
        operator=node_state.operator_id,
        node_id=node_state.node_id,
        timestamp=utc_now(),
        path="warm" IF warm ELSE "cold",
    )

    tui_update(mission_panel, status="ENVELOPED", objective=objective)

    # ═══════════════════════════════════════════════
    # PHASE 2: PREFLIGHT — Check readiness
    # ═══════════════════════════════════════════════

    preflight = run_preflight(envelope)
    /*
        run_preflight(envelope):
            checks = [
                backend_available(),     # At least one LLM
                corpus_loaded(),         # FAISS index warm
                evidence_chain_intact(), # No gaps
                agents_healthy(7),       # PAT-7 responsive
                sat_gates_ready(5),      # SAT-5 loaded
            ]
            RETURN PreflightResult(
                ready=all(checks),
                issues=[c.issue FOR c IN checks IF NOT c.passed],
            )
    */

    IF NOT preflight.ready:
        print_preflight_issues(preflight.issues)
        RETURN

    tui_update(mission_panel, status="PREFLIGHT ✓")

    IF dry_run:
        print_dry_run_plan(envelope, preflight)
        RETURN

    # ═══════════════════════════════════════════════
    # PHASE 3: RETRIEVE — Knowledge grounding
    # ═══════════════════════════════════════════════

    tui_update(mission_panel, status="RETRIEVING...")

    retrieved = faiss_search(
        query=objective,
        index=node_state.faiss_index,
        k=10,
        threshold=0.6,
    )
    top_chunks = retrieved[:5]  # Top 5 for citation injection

    tui_update(mission_panel,
        status=f"RETRIEVED {len(retrieved)} chunks",
        detail=f"Top similarity: {retrieved[0].score:.3f}"
    )

    # ═══════════════════════════════════════════════
    # PHASE 4: DECOMPOSE — PAT-7 agent assignment
    # ═══════════════════════════════════════════════

    tui_update(mission_panel, status="DECOMPOSING...")

    assignments = decompose_objective(envelope, node_state.pat_agents)
    /*
        decompose_objective(envelope, agents):
            # Strategist decomposes objective into sub-tasks
            plan = strategist.decompose(envelope.objective)

            # Assign sub-tasks to agents based on role fit
            assignments = []
            FOR task IN plan.tasks:
                best_agent = match_agent(task, agents)
                assignments.append(Assignment(task, best_agent))

            RETURN assignments
    */

    FOR assignment IN assignments:
        tui_update(pat_panel, agent=assignment.agent.name, status="ASSIGNED")

    # ═══════════════════════════════════════════════
    # PHASE 5: EXECUTE — Agents work in parallel
    # ═══════════════════════════════════════════════

    tui_update(mission_panel, status="EXECUTING...")

    agent_results = []
    FOR assignment IN assignments PARALLEL:
        tui_update(pat_panel, agent=assignment.agent.name, status="WORKING")

        result = AWAIT assignment.agent.execute(
            task=assignment.task,
            context=retrieved,       # Knowledge grounding
            backend=select_backend(),
        )

        agent_results.append(result)
        tui_update(pat_panel,
            agent=assignment.agent.name,
            status="DONE" IF result.success ELSE "FAILED",
            tokens=result.tokens_generated,
        )

    # ═══════════════════════════════════════════════
    # PHASE 6: SYNTHESIZE — GoT with citation injection
    # ═══════════════════════════════════════════════

    tui_update(mission_panel, status="SYNTHESIZING...")

    synthesis = AWAIT got_synthesize_with_citations(
        objective=objective,
        agent_results=agent_results,
        retrieved_chunks=top_chunks,
        gateway=inference_gateway,
    )
    # See node0_closure/01_rag_citation_injection.md

    tui_update(mission_panel,
        status="SYNTHESIZED",
        thoughts=synthesis.thought_count,
        paths=synthesis.path_count,
    )

    # ═══════════════════════════════════════════════
    # PHASE 7: GATE — SAT-5 constitutional scoring
    # ═══════════════════════════════════════════════

    tui_update(mission_panel, status="GATING...")

    verdict = run_gate_chain(synthesis.conclusion, objective)
    /*
        run_gate_chain(text, query):
            snr = snr_facade.calculate(text=text, query=query)
            ihsan_score = snr.score
            ihsan_achieved = ihsan_score >= IHSAN_THRESHOLD

            # SAT-5 individual gates
            FOR sat IN sat_agents:
                sat_result = sat.evaluate(text, query, snr)
                tui_update(sat_panel, agent=sat.name,
                    status="PASS" IF sat_result.passed ELSE "FLAG")

            decision = "PERMIT" IF ihsan_achieved ELSE "REVIEW"
            reason_codes = [] IF ihsan_achieved ELSE ["SNR_BELOW_THRESHOLD"]

            RETURN GateVerdict(
                snr_score=snr.score,
                ihsan_score=ihsan_score,
                ihsan_achieved=ihsan_achieved,
                decision=decision,
                reason_codes=reason_codes,
            )
    */

    tui_update(mission_panel,
        status=verdict.decision,
        snr=verdict.snr_score,
        ihsan=verdict.ihsan_score,
    )

    # ═══════════════════════════════════════════════
    # PHASE 8: RECEIPT — Hash-chain the evidence
    # ═══════════════════════════════════════════════

    receipt = emit_receipt(envelope, verdict, synthesis)
    /*
        emit_receipt(envelope, verdict, synthesis):
            receipt = CanonicalReceipt(
                receipt_id=uuid4(),
                mission_id=envelope.mission_id,
                decision=verdict.decision,
                snr_score=verdict.snr_score,
                ihsan_score=verdict.ihsan_score,
                reason_codes=verdict.reason_codes,
                node_id=envelope.node_id,
            )
            # BLAKE3 chain
            prev = evidence_chain.latest_hash()
            entry_hash = blake3(serialize(receipt) + prev)
            evidence_chain.append(receipt, entry_hash, prev)

            # Token economy
            FOR agent IN contributing_agents:
                mint_seed(agent, amount=compute_seed_reward(agent))
                mint_impt(agent, amount=compute_impact(agent))

            RETURN receipt
    */

    # ═══════════════════════════════════════════════
    # PHASE 9: MEMORY — Persist for next run
    # ═══════════════════════════════════════════════

    persist_memory(envelope, synthesis, verdict, receipt)
    /*
        persist_memory(envelope, synthesis, verdict, receipt):
            living_memory.store(
                mission_id=envelope.mission_id,
                objective=envelope.objective,
                conclusion=synthesis.conclusion,
                snr=verdict.snr_score,
                receipt_hash=receipt.hash,
                timestamp=utc_now(),
            )
            # Check for reflex promotion
            IF verdict.decision == "PERMIT":
                reflex_cache.promote(envelope, synthesis, receipt)
    */

    # ═══════════════════════════════════════════════
    # PHASE 10: PROOF CARD — Show the result
    # ═══════════════════════════════════════════════

    print_proof_card(envelope, synthesis, verdict, receipt)
    # See 04_proof_card.md for format

    tui_update(ghost_feed, add_entry(
        agent="Guardian",
        message=generate_ghost_insight(verdict, synthesis),
    ))
```

## 4. Mission State Machine (TUI-visible)

```
ENVELOPED → PREFLIGHT → RETRIEVING → DECOMPOSING → EXECUTING
    → SYNTHESIZING → GATING → RECEIPTING → MEMORIZING → COMPLETE

Each transition updates the TUI mission panel in real time.
User sees progress, not a blank screen.
```

## 5. Permission Preview (for risky missions)

```
FUNCTION check_permission_envelope(mission):
    risk = assess_risk(mission)

    IF risk.level >= HIGH:
        print_permission_preview(
            action=mission.objective,
            affected=risk.affected_paths,
            reason=risk.justification,
            inside_telescript=risk.telescript_envelope,
        )
        confirmed = prompt_user("Proceed? [y/N/escalate]")
        IF confirmed == "escalate":
            escalate_to_guardian(mission)
        ELIF confirmed != "y":
            print_info("Mission aborted by operator")
            RETURN DENIED

    RETURN PERMITTED
```

## 6. TDD Anchors

```rust
#[test]
fn test_mission_requires_genesis() {
    clear_node_state();
    let result = bizra_mission("test objective");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("genesis"));
}

#[test]
fn test_mission_emits_receipt() {
    setup_genesis();
    let result = bizra_mission("What is Ihsan?");
    assert!(result.receipt.is_some());
    assert!(!result.receipt.unwrap().hash.is_empty());
}

#[test]
fn test_mission_advances_evidence_chain() {
    setup_genesis();
    let before = evidence_chain_length();
    bizra_mission("test");
    let after = evidence_chain_length();
    assert_eq!(after, before + 1);
}

#[test]
fn test_mission_persists_memory() {
    setup_genesis();
    bizra_mission("What is Ihsan?");
    let memory = load_living_memory();
    assert!(memory.search("Ihsan").len() > 0);
}

#[test]
fn test_mission_mints_tokens() {
    setup_genesis();
    let before = token_balance();
    bizra_mission("test");
    let after = token_balance();
    assert!(after.seed > before.seed);
    assert!(after.impt > before.impt);
}

#[test]
fn test_dry_run_no_side_effects() {
    setup_genesis();
    let before_evidence = evidence_chain_length();
    let before_tokens = token_balance();
    bizra_mission_dry_run("test");
    assert_eq!(evidence_chain_length(), before_evidence);
    assert_eq!(token_balance(), before_tokens);
}
```

## 7. Validation Gate

```
[ ] bizra mission "<objective>" completes full governed loop
[ ] TUI shows real-time state transitions (10 phases)
[ ] PAT-7 agents receive sub-tasks and execute
[ ] SAT-5 gates score the output
[ ] Receipt emitted with BLAKE3 chain link
[ ] Token economy: SEED + IMPT + zakat
[ ] Living memory persisted
[ ] Proof card displayed at completion
[ ] Permission preview for high-risk missions
[ ] dry-run mode produces no side effects
```

---

*The mission command is the product. Everything else is setup or inspection.*
