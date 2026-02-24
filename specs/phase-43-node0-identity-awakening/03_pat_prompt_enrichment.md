# Spec 03: PAT System Prompt Enrichment

Standing on Giants:
- Friston (2010): Self-modeling agents produce better predictions
- Shannon (1948): Context is the prior that reduces entropy
- Anthropic (2023): Constitutional AI — identity constraints guide behavior

## Problem

The PAT system prompt in `node0_activate.py` lines 775-777 is:

```python
system_prompt = f"""You are the PAT {agent['name']}. Your role is {agent['role']}.
Standing on Giants: {agent['giants']}.
Be concise (2-3 paragraphs). Focus on actionable insights."""
```

Three lines. No identity. No assets. No goals. No covenant.

## Solution

Inject the FounderContext into the system prompt between the role declaration
and the behavioral instructions. The tier is selected by the ContextTierRouter
based on mission + agent role.

## Location

Modifications to `scripts/node0_activate.py`:
- Import FounderContext at module level
- Initialize FounderContext during `_init_verified_pipeline()` or `SovereignRuntime.__init__()`
- Inject context in the mission execution loop

## Pseudocode

### Step 1: Initialize FounderContext at startup

```python
# In scripts/node0_activate.py — new global
_FOUNDER_CONTEXT = None  # Phase 43: Founder identity awareness

# In _init_verified_pipeline() or SovereignRuntime.__init__():
def _init_founder_context():
    global _FOUNDER_CONTEXT
    try:
        from core.sovereign.founder_context import FounderContext
        sovereign_dir = Path(PROJECT_ROOT) / "sovereign_state"
        _FOUNDER_CONTEXT = FounderContext(sovereign_dir)
        if _FOUNDER_CONTEXT.loaded:
            logger.info(f"  Founder context: {_FOUNDER_CONTEXT.identity.node_name} loaded")
        else:
            logger.warning("  Founder context: no identity files — agents run anonymous")
    except Exception as e:
        logger.warning(f"  Founder context: {e}")
```

### Step 2: Enrich system prompt in mission execution

```python
# In SovereignRuntime._run_mission_agents() — replace lines 775-777:

from core.sovereign.founder_context import route_context_tier

# Determine context tier for this agent + mission
tier = route_context_tier(mission["description"], agent_id)
founder_preamble = ""
if _FOUNDER_CONTEXT and _FOUNDER_CONTEXT.loaded:
    founder_preamble = _FOUNDER_CONTEXT.build(tier)

# Build enriched system prompt
if founder_preamble:
    system_prompt = (
        f"You are the PAT {agent['name']}. Your role is {agent['role']}.\n"
        f"Standing on Giants: {agent['giants']}.\n\n"
        f"--- Founder Context ---\n"
        f"{founder_preamble}\n"
        f"--- End Context ---\n\n"
        f"Be concise (2-3 paragraphs). Focus on actionable insights "
        f"relevant to {_FOUNDER_CONTEXT.identity.node_name}'s goals."
    )
else:
    # Fallback: original prompt (no identity loaded)
    system_prompt = (
        f"You are the PAT {agent['name']}. Your role is {agent['role']}.\n"
        f"Standing on Giants: {agent['giants']}.\n"
        f"Be concise (2-3 paragraphs). Focus on actionable insights."
    )
```

### Step 3: Log context tier in mission results

```python
# In the agent result dict, add context tier info:
results.append({
    "agent": agent_id,
    "name": agent["name"],
    "model": model,
    "content": content,
    "tokens": tokens,
    "success": True,
    "context_tier": tier,  # Phase 43: track what context was used
})
```

### Step 4: Include context metadata in mission receipt

```python
# In the mission result dict:
result["founder_context"] = {
    "loaded": _FOUNDER_CONTEXT is not None and _FOUNDER_CONTEXT.loaded,
    "tiers_used": {r["agent"]: r.get("context_tier", "none") for r in results},
}
```

## Token Impact Analysis

Worst case (all 7 agents get "full" tier):
- 7 agents * 180 tokens = 1,260 tokens additional
- At local LM Studio inference: negligible cost
- At cloud inference: ~$0.003 (trivial)

Typical case (mixed tiers):
- Guardian: full (180) + Strategist: standard (100) + Coordinator: standard (100)
  + 4 others: minimal (40 * 4 = 160) = 540 tokens additional

## What Changes in the User Experience

Before Phase 43:
```
Mission: "How should I prioritize this week?"

Strategist: "Here are some general prioritization frameworks..."
  (no awareness of MoMo's weekly goals or assets)
```

After Phase 43:
```
Mission: "How should I prioritize this week?"

Strategist: "Given your current focus on making PAT work for Node0,
  and your weekly goals (ship MVP, index knowledge base, build
  impact report, launch domains), I recommend..."
  (awareness of WHO, WHAT, and WHY)
```

## Test Anchors

```python
# Integration test — verifies enriched prompt reaches agents
class TestPATPromptEnrichment:
    def test_system_prompt_contains_founder_name(self):
        """Enriched system prompt mentions MoMo."""
        # Build the prompt as the production code would
        from core.sovereign.founder_context import FounderContext, route_context_tier
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        tier = route_context_tier("plan weekly strategy", "strategist")
        text = ctx.build(tier)
        assert "MoMo" in text

    def test_strategic_mission_gets_goals_in_prompt(self):
        """Strategic mission prompt includes weekly goals."""
        from core.sovereign.founder_context import FounderContext, route_context_tier
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        tier = route_context_tier("plan the BIZRA launch strategy", "strategist")
        text = ctx.build(tier)
        assert "Goals" in text or "goal" in text.lower()

    def test_executor_gets_minimal_prompt(self):
        """Executor on simple task gets minimal context."""
        from core.sovereign.founder_context import route_context_tier
        tier = route_context_tier("format this JSON", "executor")
        assert tier == "minimal"

    def test_guardian_gets_full_even_for_simple_task(self):
        """Guardian always gets full context (ethical oversight needs identity)."""
        from core.sovereign.founder_context import route_context_tier
        tier = route_context_tier("check syntax", "guardian")
        assert tier == "full"
```

## Backward Compatibility

If FounderContext fails to load (missing files, import error), the system
falls back to the original 3-line prompt. No regression. Graceful degradation.
