# Proactive Sovereign Entity

> **Version:** 2.3.0 | **Status:** Production Ready | **Tests:** 93 Passing

The Proactive Sovereign Entity transforms BIZRA from a reactive assistant into an autonomous AI partner that works 24/7 within constitutional guardrails.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROACTIVE SOVEREIGN ENTITY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 3: NODE0 ORCHESTRATION                                               │
│  ├─ Extended OODA Loop (8 phases)                                           │
│  ├─ Team Planner (goal decomposition)                                       │
│  └─ State Checkpointer (fault tolerance)                                    │
│                                                                             │
│  LAYER 2: DUAL-AGENTIC TEAM                                                 │
│  ├─ PAT Network (7 specialists)                                             │
│  ├─ SAT Network (5 validators)                                              │
│  └─ Collective Intelligence (team synergy)                                  │
│                                                                             │
│  LAYER 1: PROACTIVE ENGINE                                                  │
│  ├─ Muraqabah Engine (24/7 monitoring)                                      │
│  ├─ Opportunity Pipeline (nervous system)                                   │
│  ├─ Autonomy Matrix (6-level control)                                       │
│  └─ Background Agents (domain specialists)                                  │
│                                                                             │
│  LAYER 0: SOVEREIGN CORE                                                    │
│  ├─ Ihsān Framework (0.95+ threshold)                                       │
│  ├─ SNR Maximization (0.85+ threshold)                                      │
│  └─ Constitutional Filters (Daughter Test)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from core.sovereign import (
    ProactiveSovereignEntity,
    create_proactive_entity,
    EntityMode,
)

# Create and start the entity
entity = await create_proactive_entity(
    mode=EntityMode.PROACTIVE,
    ihsan_threshold=0.95,
    snr_threshold=0.85,
)

# Run proactive cycles
async for result in entity.run():
    print(f"Cycle {result.cycle}: {result.opportunities_detected} opportunities")
```

## Core Components

### 1. Opportunity Pipeline

The "nervous system" connecting detection to action.

```python
from core.sovereign import (
    OpportunityPipeline,
    PipelineOpportunity,
    create_opportunity_pipeline,
)

# Create pipeline
pipeline = await create_opportunity_pipeline(
    snr_threshold=0.85,
    ihsan_threshold=0.95,
)

# Submit opportunity
await pipeline.submit_from_muraqabah(
    domain="cognitive",
    description="Learning opportunity detected",
    source="muraqabah_sensor",
    snr_score=0.92,
    urgency=0.7,
)

# Check pending approvals
pending = pipeline.get_pending_approvals()
```

**Pipeline Stages:**

| Stage | Purpose | Constitutional Check |
|-------|---------|---------------------|
| DETECTION | Receive from Muraqabah | - |
| ENRICHMENT | Add knowledge context | - |
| FILTERING | Apply guardrails | SNR, Ihsān, Daughter Test |
| PLANNING | Create action plan | - |
| APPROVAL | Check autonomy level | Autonomy Matrix |
| EXECUTION | Execute action | Rate Limiting |
| REFLECTION | Record outcome | - |

### 2. Autonomy Matrix

6-level control framework with constitutional constraints.

```python
from core.sovereign import AutonomyLevel, AutonomyMatrix

matrix = AutonomyMatrix()

# Check autonomy for an action
decision = await matrix.decide(
    action_type="financial_investment",
    context={"amount": 100, "risk": "low"},
    ihsan_score=0.97,
)

print(f"Level: {decision.level.name}")
print(f"Approved: {decision.approved}")
```

**Autonomy Levels:**

| Level | Ihsān Required | Behavior |
|-------|----------------|----------|
| `OBSERVER` | 0.0 | Watch only, no action |
| `SUGGESTER` | 0.95 | Suggest, require approval |
| `AUTOLOW` | 0.97 | Auto-execute low-risk |
| `AUTOMEDIUM` | 0.98 | Auto-execute medium-risk |
| `AUTOHIGH` | 0.99 | Auto-execute high-risk |
| `SOVEREIGN` | 1.0 | Full agency (emergencies) |

### 3. Muraqabah Engine

24/7 continuous monitoring across 5 domains.

```python
from core.sovereign import MuraqabahEngine, MonitorDomain

engine = MuraqabahEngine(
    domains=[
        MonitorDomain.FINANCIAL,
        MonitorDomain.HEALTH,
        MonitorDomain.COGNITIVE,
    ],
    scan_interval=300,  # 5 minutes
)

await engine.start()

# Get detected opportunities
opportunities = engine.get_opportunities(min_snr=0.85)
```

**Monitor Domains:**

- `FINANCIAL` — Cash flow, investments, tax opportunities
- `HEALTH` — Wellness, preventive care, activity
- `SOCIAL` — Relationships, network, communication
- `COGNITIVE` — Learning, skills, creative flow
- `ENVIRONMENTAL` — Energy, maintenance, security

### 4. Background Agents

Domain-specific proactive plugins.

```python
from core.sovereign import (
    BackgroundAgentRegistry,
    CalendarOptimizer,
    EmailTriage,
    FileOrganizer,
    create_default_registry,
)

# Create registry with default agents
registry = create_default_registry()

# Or add custom agents
registry.register(CalendarOptimizer())
registry.register(EmailTriage())

# Run all agents
await registry.run_all()
```

**Built-in Agents:**

| Agent | Domain | Function |
|-------|--------|----------|
| `CalendarOptimizer` | Schedule | Meeting optimization |
| `EmailTriage` | Communication | Priority sorting |
| `FileOrganizer` | Files | Structure maintenance |

## Constitutional Filters

### SNR Filter

Ensures signal-to-noise ratio meets quality threshold.

```python
from core.sovereign import SNRFilter

filter = SNRFilter(min_snr=0.85)
result = await filter.check(opportunity)
# result.passed: bool
# result.reason: str
```

### Ihsān Filter

Enforces excellence threshold per autonomy level.

```python
from core.sovereign import IhsanFilter

filter = IhsanFilter()
# Automatically applies level-specific thresholds
result = await filter.check(opportunity)
```

### Daughter Test Filter

Safety constraint: "Would you want this action taken for your daughter?"

```python
from core.sovereign import DaughterTestFilter

filter = DaughterTestFilter()
# Blocks sensitive domains (health, financial) at high autonomy
# Blocks sensitive keywords (delete, cancel) without approval
result = await filter.check(opportunity)
```

### Rate Limit Filter

Prevents action flooding per domain.

```python
from core.sovereign import RateLimitFilter

filter = RateLimitFilter(
    max_per_hour=10,
    max_per_day=50,
)
```

## Rust Integration

For 10-100x performance improvement on cryptographic operations.

```python
from core.sovereign import (
    RustLifecycleManager,
    create_rust_lifecycle,
    create_rust_gate_filter,
)

# Create Rust lifecycle manager
rust = await create_rust_lifecycle(
    api_port=3001,
    use_pyo3=True,
)

# Use PyO3 for fast crypto
if rust.pyo3_available:
    valid = rust.pyo3_check_ihsan(0.97)  # 100x faster
    digest = rust.pyo3_domain_digest(b"message")  # 20x faster

# Add Rust filter to pipeline
pipeline._filters.append(create_rust_gate_filter(rust))
```

## Event Bus Integration

All components communicate via the event bus.

```python
from core.sovereign import EventBus, Event, get_event_bus

bus = get_event_bus()

# Subscribe to events
async def on_opportunity(event: Event):
    print(f"Opportunity: {event.payload}")

bus.subscribe("pipeline.opportunity.received", on_opportunity)

# Publish events
await bus.publish(Event(
    topic="custom.event",
    payload={"data": "value"},
))
```

## Dashboard

CLI dashboard for monitoring (requires `rich` library).

```python
from core.sovereign import ProactiveDashboard, create_dashboard

# Create dashboard
dashboard = create_dashboard(entity)

# Run interactive mode
await dashboard.run()
```

## Configuration

### Environment Variables

```bash
# Thresholds
BIZRA_IHSAN_THRESHOLD=0.95
BIZRA_SNR_THRESHOLD=0.85

# Rust integration
BIZRA_OMEGA_PATH=/path/to/bizra-omega
BIZRA_API_PORT=3001

# Monitoring
BIZRA_SCAN_INTERVAL=300
```

### Programmatic Configuration

```python
from core.sovereign import EntityConfig

config = EntityConfig(
    mode=EntityMode.PROACTIVE,
    ihsan_threshold=0.95,
    snr_threshold=0.85,
    autonomy_default=AutonomyLevel.SUGGESTER,
    monitor_domains=["financial", "cognitive"],
    rust_enabled=True,
)

entity = await create_proactive_entity(config=config)
```

## Testing

```bash
# Run all proactive system tests
pytest tests/core/sovereign/test_proactive_integration.py -v
pytest tests/core/sovereign/test_opportunity_pipeline.py -v
pytest tests/core/sovereign/test_rust_lifecycle.py -v

# Run full test suite (93 tests)
pytest tests/core/sovereign/ -v
```

## Standing on Giants

| Scholar | Contribution |
|---------|--------------|
| Al-Ghazali | Muraqabah (continuous vigilance) |
| John Boyd | OODA Loop (decision cycle) |
| Leslie Lamport | Byzantine consensus, event ordering |
| Claude Shannon | Signal-to-noise ratio |
| Anthropic | Constitutional AI (Ihsān alignment) |
| Noam Shazeer | Mixture of Experts (agent routing) |
| Thomas Malone | Collective Intelligence |

## Module Reference

| Module | Purpose |
|--------|---------|
| `proactive_integration.py` | Unified entity |
| `opportunity_pipeline.py` | Detection → Action flow |
| `muraqabah_engine.py` | 24/7 monitoring |
| `autonomy_matrix.py` | 6-level control |
| `background_agents.py` | Domain specialists |
| `rust_lifecycle.py` | Python-Rust bridge |
| `dashboard.py` | CLI interface |
| `event_bus.py` | Pub/sub messaging |

---

*BIZRA: Every human is a node. Every node is a seed.*
