# PAT Agents Guide

Your Personal Agentic Team — 7 specialized agents at your service.

## Table of Contents

1. [Overview](#overview)
2. [Agent Profiles](#agent-profiles)
3. [Working with Agents](#working-with-agents)
4. [Agent Collaboration](#agent-collaboration)
5. [Agent Selection Guide](#agent-selection-guide)
6. [Advanced Patterns](#advanced-patterns)

---

## Overview

Your PAT (Personal Agentic Team) consists of 7 specialized agents, each drawing wisdom from masters in their domain.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR THINK TANK                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ♟ STRATEGIST ─── Long-term vision, planning, competitive analysis        │
│   🔍 RESEARCHER ─── Knowledge discovery, fact verification, synthesis       │
│   ⚙ DEVELOPER ──── Code implementation, debugging, architecture            │
│   📊 ANALYST ────── Data analysis, metrics, visualization                   │
│   ✓ REVIEWER ────── Quality assurance, security audit, compliance          │
│   ▶ EXECUTOR ────── Task execution, automation, deployment                  │
│   🛡 GUARDIAN ────── Ethics, safety, FATE gate enforcement                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Profiles

### ♟ Strategist

**Role:** Strategic planning and long-term thinking

**Giants:** Sun Tzu • Clausewitz • Porter

**Specialties:**
- Strategic analysis
- Market research
- Competitive analysis
- Risk assessment
- Goal setting
- Roadmap creation

**When to Use:**
- Planning a new project
- Making strategic decisions
- Analyzing competitors
- Setting quarterly goals
- Evaluating risks

**Example Commands:**
```
/strategy plan "Launch v2.0 by Q2"
/strategy assess "Enter Asian market"
/strategy roadmap "2026 product vision"
```

**Sample Interaction:**
```
You: Help me plan the next quarter

Strategist: Based on your current goals and the milestone for Q1 2026
("Node0 production-ready"), I recommend focusing on:

1. **Core Infrastructure** (40% effort)
   - Complete CLI/TUI integration
   - Stabilize FATE gates

2. **Integration** (35% effort)
   - LM Studio connection reliability
   - Voice interface (PersonaPlex)

3. **Testing** (25% effort)
   - Federation protocol tests
   - Load testing

Risk Assessment: Medium - depends on PersonaPlex stability.
Recommendation: Prioritize CLI completion before voice.
```

---

### 🔍 Researcher

**Role:** Knowledge discovery and synthesis

**Giants:** Shannon • Turing • Dijkstra

**Specialties:**
- Web research
- Academic search
- Fact verification
- Literature review
- Trend analysis
- Knowledge synthesis

**When to Use:**
- Learning new topics
- Verifying information
- Finding best practices
- Comparing technologies
- Understanding domains

**Example Commands:**
```
/research deep "quantum computing for cryptography"
/research quick "Rust async runtime options"
/research paper "transformer attention mechanisms"
```

**Sample Interaction:**
```
You: Research distributed consensus algorithms

Researcher: ## Distributed Consensus Algorithms

### Key Algorithms:

1. **Paxos** (Lamport, 1989)
   - Original consensus protocol
   - Complex to implement
   - Used by: Google Chubby, Apache ZooKeeper

2. **Raft** (Ongaro & Ousterhout, 2014)
   - Designed for understandability
   - Leader-based
   - Used by: etcd, Consul, TiKV

3. **PBFT** (Castro & Liskov, 1999)
   - Byzantine fault tolerant
   - Higher communication overhead
   - Used by: Hyperledger Fabric

4. **Tendermint** (Kwon, 2014)
   - BFT + Proof of Stake
   - Used by: Cosmos Network

### Recommendation for BIZRA:
Given your requirements for Byzantine tolerance in a federated network,
PBFT or Tendermint would be most suitable.

Sources: [1] Lamport (1989), [2] Ongaro (2014), [3] Castro (1999)
```

---

### ⚙ Developer

**Role:** Code implementation and technical solutions

**Giants:** Knuth • Ritchie • Torvalds

**Specialties:**
- Code generation
- Code review
- Debugging
- Refactoring
- Architecture design
- Testing

**When to Use:**
- Writing new code
- Fixing bugs
- Refactoring
- Designing systems
- Code questions

**Example Commands:**
```
/code implement "rate limiting middleware"
/code debug "memory leak in worker pool"
/code explain "src/consensus.rs:42-100"
```

**Sample Interaction:**
```
You: Implement a simple rate limiter in Rust

Developer: Here's a token bucket rate limiter:

```rust
use std::time::{Duration, Instant};

pub struct RateLimiter {
    capacity: u32,
    tokens: f64,
    refill_rate: f64,  // tokens per second
    last_update: Instant,
}

impl RateLimiter {
    pub fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            refill_rate,
            last_update: Instant::now(),
        }
    }

    pub fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate)
            .min(self.capacity as f64);
        self.last_update = now;
    }
}
```

This implements the token bucket algorithm with:
- Configurable capacity and refill rate
- Thread-safe design (wrap in Mutex for concurrent use)
- Lazy refill on access
```

---

### 📊 Analyst

**Role:** Data analysis and insights

**Giants:** Tukey • Tufte • Cleveland

**Specialties:**
- Data analysis
- Statistical modeling
- Visualization
- Pattern detection
- Anomaly detection
- Forecasting

**When to Use:**
- Analyzing data
- Creating reports
- Finding patterns
- Making predictions
- Designing dashboards

**Example Commands:**
```
/analyze data "04_GOLD/metrics.parquet"
/analyze trend "user growth last 30 days"
/analyze compare "Q1 vs Q2 performance"
```

---

### ✓ Reviewer

**Role:** Quality assurance and validation

**Giants:** Fagan • Parnas • Brooks

**Specialties:**
- Code review
- Security audit
- Compliance check
- Documentation review
- Test coverage
- Performance review

**When to Use:**
- Before merging code
- Security checks
- Quality validation
- Documentation review

**Example Commands:**
```
/code review src/main.rs
/review security "authentication module"
/review docs "API documentation"
```

---

### ▶ Executor

**Role:** Task execution and automation

**Giants:** Toyota • Deming • Ohno

**Specialties:**
- Task execution
- Automation
- Deployment
- Batch processing
- Scheduling
- Monitoring

**When to Use:**
- Running commands
- Deploying code
- Automating tasks
- Batch operations

**Example Commands:**
```
/exec run "cargo test"
/exec deploy staging
/exec automate "daily-backup"
```

**Note:** Executor always requires Guardian approval for significant actions.

---

### 🛡 Guardian

**Role:** Ethics, safety, and oversight

**Giants:** Al-Ghazali • Rawls • Anthropic

**Specialties:**
- Ethical review
- Safety check
- Risk monitoring
- Compliance validation
- Anomaly alert
- Harm prevention

**When to Use:**
- Reviewing decisions
- Checking ethics
- Monitoring risks
- High-stakes actions

**Special Powers:**
- **Veto** — Can block any agent's action
- **Pause** — Can pause execution
- **Escalate** — Can request human review

**Example Commands:**
```
/guardian review "delete all user data"
/guardian status
/guardian alert critical
```

---

## Working with Agents

### Switching Agents

```bash
# Full command
/agent switch guardian

# Shortcut
/a guardian

# In TUI: use j/k to navigate, Enter to select
```

### Direct Agent Commands

```bash
# Route to specific agent
/research deep "topic"      # → Researcher
/code implement "feature"   # → Developer
/analyze data "file"        # → Analyst
/strategy plan "objective"  # → Strategist
/exec run "command"         # → Executor
/guardian review "action"   # → Guardian
```

### Agent Status

```bash
# Check all agents
/agent list

# Check specific agent
/agent status developer
```

---

## Agent Collaboration

### Automatic Collaboration

Agents collaborate automatically based on task type:

```
Task: "Review and deploy the authentication update"

Flow:
1. Developer → Prepares changes
2. Reviewer → Reviews code
3. Guardian → Approves changes
4. Executor → Deploys
```

### Manual Collaboration

```bash
# Start multi-agent session
/swarm "Design new API endpoints"

# Agents involved:
# - Strategist: Requirements
# - Developer: Implementation plan
# - Reviewer: Standards check
# - Guardian: Security review
```

### Consensus Mode

```bash
/consensus "Should we use microservices?"

# All agents provide perspective
# Votes are collected
# Guardian makes final call if no consensus
```

---

## Agent Selection Guide

### Quick Reference

| Task Type | Best Agent | Command |
|-----------|------------|---------|
| Planning | Strategist | `/strategy plan` |
| Research | Researcher | `/research` |
| Coding | Developer | `/code` |
| Data | Analyst | `/analyze` |
| Quality | Reviewer | `/review` |
| Execution | Executor | `/exec` |
| Ethics | Guardian | `/guardian` |

### Decision Tree

```
What are you doing?
│
├─ Planning something? → Strategist
│
├─ Need information? → Researcher
│
├─ Writing code? → Developer
│
├─ Working with data? → Analyst
│
├─ Checking quality? → Reviewer
│
├─ Running something? → Executor
│
└─ Unsure/Sensitive? → Guardian
```

---

## Advanced Patterns

### Agent Pipelines

```yaml
# In skills.yaml
pipelines:
  feature_development:
    - agent: researcher
      action: "gather requirements"
    - agent: developer
      action: "implement"
    - agent: reviewer
      action: "review"
    - agent: guardian
      action: "approve"
    - agent: executor
      action: "deploy"
```

### Conditional Routing

```yaml
# In a2a_protocol.yaml
routing:
  auto_route:
    - pattern: "code|implement|build"
      agent: "developer"
    - pattern: "research|find|discover"
      agent: "researcher"
```

### Agent Handoff

```bash
# Developer completes, hands to Reviewer
/handoff reviewer "Code complete, ready for review"
```

---

## Tips

1. **Start with Guardian** — Safest default
2. **Use shortcuts** — `/a dev` is faster than `/agent switch developer`
3. **Let agents collaborate** — Use `/swarm` for complex tasks
4. **Trust the routing** — System often picks the right agent
5. **Guardian is watching** — High-stakes actions always get reviewed

---

## Next Steps

- [Slash Commands](05-SLASH-COMMANDS.md) — Full command reference
- [Skills System](10-SKILLS-SYSTEM.md) — Agent workflows
- [A2A Protocol](../reference/A2A-PROTOCOL.md) — Agent communication

---

**Your team, your rules, your sovereignty.** 🛡
