# BIZRA Islamic Masterminds Agentic Constellation v1.0

> A production-ready multi-agent system featuring 27 historical Islamic masterminds + 2 meta-agents with SNR-tier routing and GoT/ToT/CoT reasoning controls.

## Overview

This constellation transforms historical intellectual giants into a practical, testable agent fleet designed to integrate with BIZRA's Dual-Sovereignty model (PAT/SAT).

### Key Features

- **29 Specialized Agents**: Each with domain expertise, SNR targets, and output contracts
- **8 Cross-Pollination Teams**: Pre-built squads for common task types
- **SNR-Tier Routing**: Precision-based agent selection (85%–98%)
- **Reasoning Architecture**: CoT → ToT → GoT escalation
- **Evidence Gates**: Claim tagging, contradiction detection, verification

## Directory Structure

```
constellation/
├── __init__.py                 # Package initialization
├── orchestrator.py             # Main orchestrator (Python)
├── README.md                   # This file
├── agents/
│   └── roster.yaml             # 27 domain + 2 meta agents
├── teams/
│   └── configurations.yaml     # 8 cross-pollination teams
├── router/
│   └── policy.yaml             # SNR routing + reasoning escalation
├── evaluation/
│   └── gates.yaml              # Evidence gates + claim tagging
└── prompts/
    └── system_prompts.yaml     # LangGraph-ready prompts
```

## Quick Start

### Python Usage

```python
from constellation import get_orchestrator

# Initialize
orchestrator = get_orchestrator()

# Execute a task
result = orchestrator.execute(
    "Design an ethical AI governance framework",
    context={"stakes": "high"}
)

# Access structured output
print(result.executive_summary)
print(f"Final SNR: {result.final_snr}")
print(f"Team used: {result.team_used}")
```

### CLI Usage

```bash
python orchestrator.py "Analyze the scientific method of Ibn al-Haytham" --stakes high --json
```

## Agent Roster

### Meta-Agents (Orchestration Layer)

| ID | Agent | Domain | SNR | Role |
|----|-------|--------|-----|------|
| 26 | Polymath Integrator | Meta | 0.98-0.99 | Cross-domain synthesis |
| 27 | Adaptive Orchestrator | Meta | 0.95-0.96 | Team optimization |

### Domain Agents (27 Specialists)

| Cluster | Agents | SNR Range |
|---------|--------|-----------|
| Philosophy & Theology | Al-Kindi, Al-Farabi, Ibn Sina, Al-Ghazali, Ibn Rushd | 0.87-0.95 |
| Social Science | Ibn Khaldun | 0.92-0.93 |
| Mathematics & Astronomy | Al-Khwarizmi, Omar Khayyam, Nasir al-Din al-Tusi | 0.86-0.96 |
| Medicine | Al-Razi, Ibn Sina (Medical), Al-Zahrawi | 0.91-0.95 |
| Natural Sciences | Jabir ibn Hayyan, Ibn al-Haytham | 0.89-0.97 |
| Engineering & Architecture | Al-Jazari, Mimar Sinan | 0.88-0.91 |
| Jurisprudence (Verifiers) | Abu Hanifa, Malik, al-Shafi'i, Ahmad, Bukhari | 0.89-0.98 |
| Exploration & Creativity | Ibn Battuta, Rumi | 0.85-0.86 |
| Leadership & Strategy | Umar ibn al-Khattab, Saladin | 0.90-0.92 |

## SNR Tiers

| Tier | SNR Range | Use When | Controls |
|------|-----------|----------|----------|
| T1 | 96–98% | Authentication / maximal precision | Proof-required, 2 verifiers |
| T2 | 93–97% | Scientific method / strong evidence | 1 verifier, experiments |
| T3 | 92–95% | Medical reasoning | Checklists, second opinion |
| T4 | 93–96% | Mathematical reasoning | Formal proofs, unit tests |
| T5 | 88–92% | Philosophical synthesis | Explicit assumptions |
| T6 | 85–90% | Creative generation | Sandboxed, no fact claims |

## Reasoning Modes

### Chain-of-Thought (CoT)
- **When**: Low ambiguity, linear steps
- **Speed**: Fastest
- **Use**: Default for most tasks

### Tree-of-Thought (ToT)
- **When**: Planning, branching decisions, conflicts
- **Speed**: Medium
- **Use**: High stakes, policy design

### Graph-of-Thought (GoT)
- **When**: Interdisciplinary synthesis, complex causal analysis
- **Speed**: Slower but comprehensive
- **Use**: 3+ domains, integration tasks

## Cross-Pollination Teams

| Team | Leader | SNR | Best For |
|------|--------|-----|----------|
| Scientific Method Elite | Ibn al-Haytham | 95% | Hypothesis testing, experiments |
| Systems Architecture Dream | Ibn Khaldun | 91% | Governance, infrastructure |
| Medical Innovation Task Force | Ibn Sina | 94% | Diagnosis, treatment |
| Mathematical Computation Core | Al-Khwarizmi | 93% | Algorithms, proofs |
| Philosophical Synthesis Council | Ibn Sina | 91% | Ethics, worldviews |
| Legal Reasoning Panel | Imam al-Shafi'i | 94% | Rulings, jurisprudence |
| Innovation & Creativity Studio | Al-Jazari | 88% | Brainstorming, design |
| Strategic Leadership Command | Saladin | 92% | Coordination, strategy |

## Claim Tagging System

All claims must be tagged with epistemic status:

| Tag | Definition | SNR Weight |
|-----|------------|------------|
| `MEASURED` | Empirically measured/observed | 1.0 |
| `IMPLEMENTED` | Code/system exists and testable | 0.95 |
| `DERIVED` | Logically derived from facts | 0.90 |
| `DESIGNED` | Specification exists, not built | 0.75 |
| `TARGET` | Aspiration, not yet achieved | 0.50 |
| `HYPOTHESIS` | Proposed, requires testing | 0.40 |
| `METAPHOR` | Figurative, not literal | 0.0 |

## Execution Flow

```
1. INTAKE
   └── Parse task → Classify stakes → Detect domains → Set SNR target

2. PLAN
   └── Select team → Choose reasoning mode → Assign verifiers

3. WORK
   └── Domain agents produce solutions + evidence bundles

4. VERIFY
   └── Verifiers challenge assumptions → Check sources → Scan contradictions

5. SYNTHESIZE
   └── Polymath Integrator unifies → Resolve conflicts → Build deliverable

6. DELIVER
   └── Executive summary → What we know → What we assume → What to test
```

## Evidence Gates

### Gate 1: Claim Tagging
All non-trivial claims must have epistemic tags.

### Gate 2: Contradiction Scan
No internal contradictions. Conflicts escalate to ToT.

### Gate 3: SNR Floor
High-stakes outputs require SNR ≥ 0.93.

### Gate 4: Verifier Attestation
SNR ≥ 0.95 requires two verifiers.

### Gate 5: Source Citation
MEASURED/IMPLEMENTED claims require citations.

## Integration

### LangGraph

```python
from langgraph.graph import StateGraph
from constellation import get_loader

loader = get_loader()
prompts = loader.prompts  # System prompts for all agents

# Build your graph with agent nodes
for agent_slug, agent in loader.agents.items():
    # Create node for each agent using their system prompt
    pass
```

### Custom Orchestrator

```python
from constellation.orchestrator import (
    ConstellationOrchestrator,
    TaskAnalysis,
    AgentOutput,
)

def my_agent_executor(agent, query, reasoning_mode):
    # Your LLM call here
    response = my_llm.chat(
        system=agent.system_prompt,
        user=query,
        # ... reasoning_mode configuration
    )
    return AgentOutput(
        agent_id=agent.id,
        agent_name=agent.name,
        content=response,
        claims=extract_claims(response),
        confidence=estimate_confidence(response),
        reasoning_trace="..."
    )

orchestrator = ConstellationOrchestrator()
result = orchestrator.execute(
    query="...",
    agent_executor=my_agent_executor
)
```

## Version History

- **v1.0** (2025-12-20): Initial release with 29 agents, 8 teams, full orchestration

## License

BIZRA Elite License - For authorized BIZRA operations only.
