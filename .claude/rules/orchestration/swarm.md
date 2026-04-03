---
paths:
  - "src/**/*.rs"
  - "core/**/*.py"
  - "constellation/**/*.py"
---

# Swarm Intelligence Rules

Rules for multi-agent swarm coordination, integrating patterns from claude-flow.

## Swarm Modes

### Independent Mode
Agents work autonomously without coordination.

```rust
pub enum SwarmMode {
    Independent,   // Agents work alone
    Collaborative, // Agents share results
    HiveMind,      // Collective decision-making
}
```

**Use when**:
- Tasks are easily parallelizable
- No shared state required
- Speed is priority over consensus

### Collaborative Mode
Agents share intermediate results and coordinate.

```python
async def collaborative_swarm(agents: list[Agent], task: Task) -> SwarmResult:
    # Phase 1: Independent work
    partial_results = await asyncio.gather(
        *[agent.partial_execute(task) for agent in agents]
    )

    # Phase 2: Share and refine
    shared_context = merge_partial_results(partial_results)
    final_results = await asyncio.gather(
        *[agent.refine(shared_context) for agent in agents]
    )

    # Phase 3: Synthesize
    return synthesize_results(final_results)
```

**Use when**:
- Complex tasks requiring diverse perspectives
- Results benefit from cross-pollination
- Quality over speed

### HiveMind Mode
Collective decision-making with consensus.

```rust
async fn hive_mind_execution(
    swarm: &Swarm,
    task: &Task,
) -> Result<HiveMindResult> {
    loop {
        // Propose solutions
        let proposals = swarm.collect_proposals(task).await?;

        // Vote on proposals
        let votes = swarm.vote_on_proposals(&proposals).await?;

        // Check consensus (Byzantine fault tolerant)
        if let Some(consensus) = check_bft_consensus(&votes, swarm.size()) {
            return Ok(consensus);
        }

        // Refine proposals based on feedback
        swarm.refine_proposals(&votes).await?;
    }
}
```

**Use when**:
- High-stakes decisions
- Need Byzantine fault tolerance
- Consensus is required

## Claude-Flow Integration Patterns

### Neural Routing
Route tasks to optimal agents based on learned patterns.

```python
class NeuralRouter:
    """Route tasks to specialists based on task characteristics."""

    def route(self, task: Task) -> Agent:
        # Analyze task embedding
        embedding = self.embed_task(task)

        # Find best-matching specialist
        scores = [
            (agent, self.match_score(embedding, agent.expertise))
            for agent in self.available_agents
        ]

        # Select highest-scoring agent
        best_agent = max(scores, key=lambda x: x[1])[0]

        # Log routing decision for learning
        self.log_routing(task, best_agent)

        return best_agent
```

### Pattern Elevation
Optimize frequently-used patterns.

```python
class PatternElevator:
    """Elevate patterns with >3 occurrences to optimized shortcuts."""

    async def check_and_elevate(self, pattern_hash: str, execution_result):
        key = f"bizra:swarm:pattern:{pattern_hash}"
        count = await self.redis.incr(key)

        if count > 3:
            # Create optimized shortcut
            shortcut = self.create_shortcut(execution_result)
            await self.redis.set(f"{key}:shortcut", shortcut)
            logger.info(f"Elevated pattern: {pattern_hash}")

            # Future executions skip full swarm
            return shortcut
        return None
```

### Token Optimization
Reduce token consumption through intelligent routing.

```python
class TokenOptimizer:
    """Route to cheapest capable handler."""

    def select_handler(self, task: Task) -> Handler:
        handlers = [
            # Ordered by cost (cheapest first)
            ("wasm", self.wasm_handler, self.can_wasm_handle),
            ("haiku", self.haiku_handler, self.can_haiku_handle),
            ("sonnet", self.sonnet_handler, self.can_sonnet_handle),
            ("opus", self.opus_handler, lambda t: True),
        ]

        for name, handler, can_handle in handlers:
            if can_handle(task):
                logger.debug(f"Routing to {name} for token efficiency")
                return handler

        return self.opus_handler  # Fallback
```

## Swarm Coordination

### Spawn Coordination
```rust
// Batch spawn for efficiency
async fn spawn_swarm(config: &SwarmConfig) -> Result<Swarm> {
    let spawn_futures: Vec<_> = (0..config.count)
        .map(|i| spawn_agent(config, i))
        .collect();

    let agents = futures::future::try_join_all(spawn_futures).await?;

    Ok(Swarm::new(agents, config.mode))
}
```

### Shared Memory
```python
class SwarmMemory:
    """Shared memory for swarm coordination."""

    async def read(self, key: str) -> Any:
        return await self.redis.get(f"swarm:memory:{key}")

    async def write(self, key: str, value: Any) -> None:
        await self.redis.set(f"swarm:memory:{key}", json.dumps(value))

    async def atomic_update(self, key: str, update_fn: Callable) -> Any:
        async with self.redis.lock(f"swarm:lock:{key}"):
            current = await self.read(key)
            updated = update_fn(current)
            await self.write(key, updated)
            return updated
```

### Consensus Protocols
```rust
/// Byzantine Fault Tolerant consensus (3f+1 nodes tolerate f failures)
fn check_bft_consensus(votes: &[Vote], total: usize) -> Option<Consensus> {
    let required = (2 * total / 3) + 1;

    // Group by proposal
    let mut vote_counts: HashMap<ProposalId, usize> = HashMap::new();
    for vote in votes {
        *vote_counts.entry(vote.proposal_id).or_insert(0) += 1;
    }

    // Check if any proposal has consensus
    for (proposal_id, count) in vote_counts {
        if count >= required {
            return Some(Consensus::Reached(proposal_id));
        }
    }

    None
}
```

## Performance Optimization

### Batch Processing
```python
async def batch_execute(swarm: Swarm, tasks: list[Task]) -> list[Result]:
    # Group tasks by type for efficient routing
    task_groups = group_by_type(tasks)

    # Process groups in parallel
    results = await asyncio.gather(
        *[swarm.execute_batch(group) for group in task_groups.values()]
    )

    # Flatten and reorder
    return flatten_and_reorder(results, tasks)
```

### Caching
```python
class SwarmCache:
    """Cache swarm decisions for repeated patterns."""

    async def get_or_execute(self, task: Task, swarm: Swarm) -> Result:
        cache_key = self.compute_cache_key(task)

        # Check cache
        cached = await self.redis.get(f"swarm:cache:{cache_key}")
        if cached:
            return json.loads(cached)

        # Execute and cache
        result = await swarm.execute(task)
        await self.redis.setex(
            f"swarm:cache:{cache_key}",
            self.cache_ttl,
            json.dumps(result)
        )
        return result
```

## BIZRA Integration

### Swarm with SAT Validation
```python
async def validated_swarm_execute(swarm: Swarm, task: Task) -> Result:
    # Pre-validation
    consensus = await sat_consensus(task)
    if not consensus.approved:
        raise ConsensusFailure(consensus)

    # Swarm execution
    result = await swarm.execute(task)

    # Post-validation
    validated = await sat_validate_result(result)
    if not validated.approved:
        raise ValidationFailure(validated)

    # Emit receipt
    await emit_swarm_receipt(task, result)

    return result
```

### Ihsān-Aware Swarm
```python
class IhsanAwareSwarm(Swarm):
    """Swarm that enforces Ihsān threshold."""

    async def execute(self, task: Task) -> Result:
        result = await super().execute(task)

        # Calculate Ihsān score for swarm output
        ihsan_score = await calculate_ihsan(result)

        if ihsan_score < IHSAN_THRESHOLD:
            # Escalate and fail
            await fate.escalate(EscalationLevel.HIGH, "Swarm output failed Ihsān")
            raise IhsanGateError(ihsan_score, IHSAN_THRESHOLD)

        return result
```

## Testing

- Test swarm mode switching
- Test consensus with various vote distributions
- Test pattern elevation trigger
- Test token optimization routing
- Test batch processing efficiency
- Test cache hit/miss scenarios
- Test BFT consensus edge cases
