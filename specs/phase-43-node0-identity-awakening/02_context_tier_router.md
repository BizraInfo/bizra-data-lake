# Spec 02: Context Tier Router

Standing on Giants:
- Shannon (1948): Don't waste channel capacity — send exactly the signal needed
- Boyd (1976): OODA — orient before deciding what context depth to use
- Kahneman (2011): System 1/System 2 — not every task needs deep reasoning

## Problem

Not every PAT mission needs full founder context. A quick "summarize this file"
doesn't need to know about Momo's 15K R&D hours. But "plan the BIZRA whitepaper
launch strategy" absolutely does. Sending full context to every agent wastes
tokens and dilutes signal.

## Solution

A `ContextTierRouter` that selects the appropriate tier (full / standard / minimal)
based on:
1. **Mission keywords** — strategic/personal missions get full context
2. **Agent role** — Guardian always gets full, Executor gets minimal
3. **Token budget** — if model has small context window, compress

## Location

Integrated into `core/sovereign/founder_context.py` — extends the existing module
(~40 additional lines)

## Pseudocode

```python
# Added to core/sovereign/founder_context.py

# Keyword sets that trigger higher context tiers
_FULL_CONTEXT_KEYWORDS = {
    "strategy", "whitepaper", "launch", "investor", "roadmap",
    "vision", "mission", "bizra", "genesis", "covenant", "ihsan",
    "architecture", "identity", "values", "goals", "weekly",
    "personal", "dream", "pain", "founder", "node0", "ramadan",
    "budget", "resources", "assets", "research", "papers",
}

_STANDARD_CONTEXT_KEYWORDS = {
    "plan", "analyze", "research", "investigate", "design",
    "evaluate", "compare", "review", "audit", "optimize",
    "build", "implement", "deploy", "create", "write",
}

# Agent role → default tier (overridden by mission keywords)
_AGENT_DEFAULT_TIER = {
    "guardian":    "full",       # Guardian always needs full identity awareness
    "strategist":  "standard",   # Strategist needs goals and focus
    "coordinator": "standard",   # Coordinator needs the big picture
    "researcher":  "standard",   # Researcher needs asset awareness
    "analyst":     "standard",   # Analyst needs data context
    "creator":     "minimal",    # Creator focuses on the task
    "executor":    "minimal",    # Executor just does the work
}


def route_context_tier(
    mission_description: str,
    agent_role: str,
) -> str:
    """
    Select the appropriate founder context tier for a PAT agent mission.

    Returns: "full" | "standard" | "minimal"

    Routing logic (highest tier wins):
    1. If mission contains FULL keywords → "full"
    2. If mission contains STANDARD keywords → max(agent_default, "standard")
    3. Otherwise → agent's default tier
    """
    mission_lower = mission_description.lower()
    mission_words = set(mission_lower.split())

    # Check for full-context triggers
    if mission_words & _FULL_CONTEXT_KEYWORDS:
        return "full"

    # Check for standard-context triggers
    if mission_words & _STANDARD_CONTEXT_KEYWORDS:
        agent_default = _AGENT_DEFAULT_TIER.get(agent_role, "minimal")
        # Upgrade to at least standard
        return "full" if agent_default == "full" else "standard"

    # Default to agent's role-based tier
    return _AGENT_DEFAULT_TIER.get(agent_role, "minimal")
```

## Test Anchors

```python
# tests/core/sovereign/test_founder_context.py (additional tests)

class TestContextTierRouter:
    def test_strategic_mission_gets_full(self):
        """Mission about strategy triggers full context."""
        tier = route_context_tier("Plan the BIZRA whitepaper launch strategy", "executor")
        assert tier == "full"

    def test_guardian_always_full(self):
        """Guardian agent gets full context regardless of mission."""
        tier = route_context_tier("check this file for errors", "guardian")
        assert tier == "full"

    def test_executor_gets_minimal_for_simple_tasks(self):
        """Executor gets minimal for non-strategic tasks."""
        tier = route_context_tier("format this JSON file", "executor")
        assert tier == "minimal"

    def test_researcher_gets_standard_for_analysis(self):
        """Research mission upgrades researcher to standard."""
        tier = route_context_tier("analyze the test coverage gaps", "researcher")
        assert tier == "standard"

    def test_identity_keywords_trigger_full(self):
        """Mission mentioning identity/values/goals → full."""
        tier = route_context_tier("review our core values alignment", "analyst")
        assert tier == "full"

    def test_unknown_agent_defaults_minimal(self):
        """Unknown agent role defaults to minimal."""
        tier = route_context_tier("do something", "unknown_agent")
        assert tier == "minimal"

    def test_bizra_keyword_triggers_full(self):
        """Any mission mentioning BIZRA gets full context."""
        tier = route_context_tier("How should bizra handle federation?", "creator")
        assert tier == "full"
```

## Token Budget Analysis

| Tier     | ~Tokens | When Used |
|----------|---------|-----------|
| minimal  | 30-40   | Routine execution tasks, file operations |
| standard | 80-100  | Analysis, research, implementation tasks |
| full     | 150-180 | Strategic, identity, vision, investor-facing tasks |

At LM Studio's local inference, these costs are negligible. The real value is
signal density — agents get exactly the context they need, no more, no less.
