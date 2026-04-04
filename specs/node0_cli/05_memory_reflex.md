# Phase 5: Memory + Reflex System
## Carryover between missions — the compounding effect
### References: 00_cli_master_spec.md §8, MVP #5

---

## 1. Design Principle

**"...remembered the result, and made the next run faster."**

Memory is not logging. It is the mechanism by which the node **compounds**.
Each mission makes the next one better through:
- Stored conclusions (avoid redundant reasoning)
- Reflex promotion (fast path for proven queries)
- Context enrichment (prior results inform new missions)

## 2. Memory Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Living Memory                       │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │  Mission     │  │  Reflex     │  │  Knowledge   │ │
│  │  Memory      │  │  Cache      │  │  Memory      │ │
│  │              │  │             │  │              │ │
│  │  Past runs,  │  │  Promoted   │  │  Facts and   │ │
│  │  conclusions │  │  hot-path   │  │  learnings   │ │
│  │  + receipts  │  │  responses  │  │  from agents │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘ │
│         │                │                │          │
│         └────────────────┼────────────────┘          │
│                          │                            │
│                   ┌──────▼──────┐                     │
│                   │  SQLite DB  │                     │
│                   │  memory.db  │                     │
│                   └─────────────┘                     │
└──────────────────────────────────────────────────────┘
```

## 3. Pseudocode: `bizra memory` Command

```
COMMAND bizra_memory(subcommand: str = "list"):
    MATCH subcommand:
        "list"           → memory_list()
        "search" query   → memory_search(query)
        "stats"          → memory_stats()
        "reflexes"       → memory_reflexes()

FUNCTION memory_list(limit: int = 10):
    db = open_memory_db()
    entries = db.query("SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", limit)

    print_box(f"Living Memory (last {limit})")
    FOR entry IN entries:
        compact_row(
            type_badge(entry.type),
                # 🎯 mission | 💡 knowledge | ⚡ reflex
            truncate(entry.content, 50),
            f"SNR={entry.snr:.2f}" IF entry.snr ELSE "",
            relative_time(entry.timestamp),
        )

FUNCTION memory_search(query: str):
    db = open_memory_db()

    # Semantic search via embedding similarity
    query_embedding = embed(query)
    results = db.vector_search(query_embedding, k=5, threshold=0.5)

    print_box(f"Memory Search: \"{query}\"")
    FOR result IN results:
        row(
            similarity=f"{result.score:.2f}",
            type=type_badge(result.type),
            content=wrap_text(result.content, 60),
            mission=result.mission_id[:8] IF result.mission_id ELSE "—",
            receipt=result.receipt_hash[:8] IF result.receipt_hash ELSE "—",
        )

FUNCTION memory_stats():
    db = open_memory_db()
    stats = db.stats()

    print_box("Memory Statistics")
    row("Total entries",     stats.total)
    row("Mission memories",  stats.mission_count)
    row("Knowledge entries", stats.knowledge_count)
    row("Active reflexes",   stats.reflex_count)
    row("Database size",     human_bytes(stats.db_size))
    row("Oldest entry",      relative_time(stats.oldest))
    row("Newest entry",      relative_time(stats.newest))
    row("Avg SNR",           f"{stats.avg_snr:.3f}")

FUNCTION memory_reflexes():
    db = open_memory_db()
    reflexes = db.query_reflexes()

    print_box("Active Reflexes (hot-path cache)")
    FOR reflex IN reflexes:
        row(
            query_hash=reflex.query_hash[:12],
            cold_receipt=reflex.cold_receipt[:8],
            warm_receipt=reflex.warm_receipt[:8] IF reflex.warm_receipt ELSE "—",
            ttl=remaining_time(reflex.expires_at),
            hits=reflex.hit_count,
        )
```

## 4. Pseudocode: Memory Persistence During Mission

```
FUNCTION persist_mission_memory(
    envelope: MissionEnvelope,
    synthesis: SynthesisResult,
    verdict: GateVerdict,
    receipt: CanonicalReceipt,
):
    db = open_memory_db()

    # Store mission conclusion
    mission_entry = MemoryEntry(
        type="mission",
        content=synthesis.conclusion,
        mission_id=envelope.mission_id,
        query=envelope.objective,
        query_hash=blake3(envelope.objective),
        snr=verdict.snr_score,
        ihsan=verdict.ihsan_score,
        decision=verdict.decision,
        receipt_hash=receipt.receipt_id,
        embedding=embed(synthesis.conclusion),
        timestamp=utc_now(),
    )
    db.insert(mission_entry)

    # Store agent-specific knowledge
    FOR result IN synthesis.agent_results:
        IF result.success AND result.novelty_score > 0.3:
            knowledge_entry = MemoryEntry(
                type="knowledge",
                content=result.key_insight,
                source_agent=result.agent_name,
                mission_id=envelope.mission_id,
                embedding=embed(result.key_insight),
                timestamp=utc_now(),
            )
            db.insert(knowledge_entry)

    # Check for reflex promotion
    IF verdict.decision == "PERMIT":
        promote_to_reflex(envelope, synthesis, receipt, db)

    db.commit()
    RETURN MemoryDelta(
        total=db.count(),
        added=1 + count(novel_knowledge),
        reflex_promoted=(verdict.decision == "PERMIT"),
    )
```

## 5. Pseudocode: Reflex Promotion

```
FUNCTION promote_to_reflex(
    envelope: MissionEnvelope,
    synthesis: SynthesisResult,
    receipt: CanonicalReceipt,
    db: MemoryDB,
):
    """Promote a PERMIT'd mission to hot-path reflex cache."""

    query_hash = blake3(envelope.objective)

    # Check if cold-path lineage exists
    cold_ancestor = db.find_cold_ancestor(query_hash)
    IF cold_ancestor IS NONE:
        # This IS the cold-path origin
        cold_ancestor = receipt

    reflex = ReflexEntry(
        query_hash=query_hash,
        query=envelope.objective,
        response=synthesis.conclusion,
        snr=synthesis.snr_score,
        cold_receipt=cold_ancestor.receipt_id,
        warm_receipt=receipt.receipt_id IF cold_ancestor != receipt ELSE None,
        promoted_at=utc_now(),
        expires_at=utc_now() + timedelta(hours=24),
        hit_count=0,
    )
    db.upsert_reflex(reflex)


FUNCTION check_reflex(objective: str) -> Optional[ReflexHit]:
    """Check if a hot-path reflex exists for this query."""
    db = open_memory_db()
    query_hash = blake3(objective)

    reflex = db.get_reflex(query_hash)
    IF reflex IS NONE:
        RETURN None

    # Validate: not expired
    IF reflex.expires_at < utc_now():
        db.delete_reflex(query_hash)
        RETURN None

    # Validate: lineage intact
    IF NOT verify_receipt_exists(reflex.cold_receipt):
        db.delete_reflex(query_hash)
        RETURN None

    # Hit!
    reflex.hit_count += 1
    db.update_reflex(reflex)

    RETURN ReflexHit(
        response=reflex.response,
        cold_receipt=reflex.cold_receipt,
        path="hot",
        latency_estimate="<1s",
    )
```

## 6. Pseudocode: Context Enrichment (Memory → Mission)

```
FUNCTION enrich_mission_context(objective: str) -> MissionContext:
    """Use prior memories to enrich current mission."""
    db = open_memory_db()

    # Find similar past missions
    query_embedding = embed(objective)
    similar_missions = db.vector_search(
        query_embedding,
        k=3,
        type_filter="mission",
        threshold=0.6,
    )

    # Find relevant knowledge
    relevant_knowledge = db.vector_search(
        query_embedding,
        k=5,
        type_filter="knowledge",
        threshold=0.5,
    )

    context = MissionContext(
        prior_missions=[
            PriorMission(
                objective=m.query,
                conclusion=m.content[:200],
                snr=m.snr,
                decision=m.decision,
                receipt=m.receipt_hash[:8],
            )
            FOR m IN similar_missions
        ],
        prior_knowledge=[
            k.content FOR k IN relevant_knowledge
        ],
        warm_path_available=any(
            m.decision == "PERMIT" AND m.snr >= 0.85
            FOR m IN similar_missions
        ),
    )

    # Ghost feed hint
    IF context.warm_path_available:
        ghost_add("Strategist",
            f"Similar mission PERMIT'd — warm path available (faster)")

    IF context.prior_knowledge:
        ghost_add("Researcher",
            f"{len(context.prior_knowledge)} relevant knowledge entries found")

    RETURN context
```

## 7. TDD Anchors

```rust
#[test]
fn test_mission_persists_memory() {
    let db = test_memory_db();
    let before = db.count();
    persist_mission_memory(test_envelope(), test_synthesis(), test_verdict(), test_receipt());
    assert!(db.count() > before);
}

#[test]
fn test_memory_search_finds_related() {
    let db = test_memory_db();
    persist_mission_memory(ihsan_envelope(), ihsan_synthesis(), permit_verdict(), test_receipt());
    let results = db.vector_search(embed("What is Ihsan?"), 3, 0.5);
    assert!(!results.is_empty());
}

#[test]
fn test_reflex_promotion_on_permit() {
    let db = test_memory_db();
    let verdict = GateVerdict::permit(0.87, 0.95);
    persist_mission_memory(test_envelope(), test_synthesis(), verdict, test_receipt());
    let reflex = db.get_reflex(blake3(test_envelope().objective));
    assert!(reflex.is_some());
}

#[test]
fn test_reflex_not_promoted_on_review() {
    let db = test_memory_db();
    let verdict = GateVerdict::review(0.61, 0.61);
    persist_mission_memory(test_envelope(), test_synthesis(), verdict, test_receipt());
    let reflex = db.get_reflex(blake3(test_envelope().objective));
    assert!(reflex.is_none());
}

#[test]
fn test_reflex_expires_after_ttl() {
    let db = test_memory_db();
    create_reflex(db, ttl_hours=0); // Already expired
    let hit = check_reflex("test query");
    assert!(hit.is_none());
}

#[test]
fn test_reflex_requires_cold_lineage() {
    let db = test_memory_db();
    let mut reflex = create_reflex(db, ttl_hours=24);
    // Corrupt lineage
    db.delete_receipt(reflex.cold_receipt);
    let hit = check_reflex("test query");
    assert!(hit.is_none()); // Orphaned reflex rejected
}

#[test]
fn test_context_enrichment_finds_prior() {
    let db = test_memory_db();
    // Store a prior mission about Ihsan
    persist_mission_memory(ihsan_envelope(), ihsan_synthesis(), permit_verdict(), test_receipt());
    // New mission about similar topic
    let context = enrich_mission_context("How does Ihsan work in BIZRA?");
    assert!(!context.prior_missions.is_empty());
    assert!(context.warm_path_available);
}

#[test]
fn test_second_run_faster_with_memory() {
    // First run: cold path
    let t1 = measure(|| bizra_mission("What is Ihsan?"));
    // Second run: should use memory/reflex
    let t2 = measure(|| bizra_mission("What is Ihsan?"));
    // Not testing exact speedup, but memory should be used
    let memory = load_memory();
    assert!(memory.search("Ihsan").len() >= 2);
}
```

## 8. Validation Gate

```
[ ] bizra memory list shows mission + knowledge entries
[ ] bizra memory search returns semantically relevant results
[ ] bizra memory stats shows correct counts
[ ] bizra memory reflexes shows promoted entries with TTL
[ ] Mission persists conclusion + receipt hash to memory
[ ] PERMIT missions promote to reflex cache
[ ] REVIEW missions do NOT promote to reflex
[ ] Reflexes expire after TTL (no stale responses)
[ ] Reflexes validate cold-path lineage before serving
[ ] Context enrichment injects prior knowledge into new missions
[ ] Ghost feed hints when warm path available
[ ] Second run of same query uses memory context
```

---

*Memory is not logging. It is how the node compounds.*
*Each mission makes the next one faster, richer, and more grounded.*
*That is the killer moment: "...and made the next run faster."*
