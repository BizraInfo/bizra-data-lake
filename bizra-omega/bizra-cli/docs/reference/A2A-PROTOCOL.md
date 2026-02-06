# A2A Protocol Reference

Agent-to-Agent communication protocol specification.

## Table of Contents

1. [Overview](#overview)
2. [Protocol Architecture](#protocol-architecture)
3. [Agent Cards](#agent-cards)
4. [Task Cards](#task-cards)
5. [Message Types](#message-types)
6. [Communication Patterns](#communication-patterns)
7. [Routing Rules](#routing-rules)
8. [FATE Integration](#fate-integration)
9. [Protocol Examples](#protocol-examples)
10. [Implementation Guide](#implementation-guide)

---

## Overview

The A2A (Agent-to-Agent) Protocol enables structured communication between PAT agents.

### Design Principles

1. **Typed Communication** — All messages have defined schemas
2. **Capability-Based** — Agents declare what they can do
3. **Guardian-Supervised** — All significant actions reviewed
4. **FATE-Validated** — All outputs pass through FATE gates

### Protocol Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                               │
│                     Task Cards • Agent Cards • Messages                  │
├─────────────────────────────────────────────────────────────────────────┤
│                          ROUTING LAYER                                   │
│               Capability Matching • Load Balancing • Priority            │
├─────────────────────────────────────────────────────────────────────────┤
│                          VALIDATION LAYER                                │
│                   FATE Gates • Schema Validation • ACL                   │
├─────────────────────────────────────────────────────────────────────────┤
│                          TRANSPORT LAYER                                 │
│                   Message Queue • Event Bus • Direct Call                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Protocol Architecture

### Message Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Sender     │     │   Router     │     │   Receiver   │
│    Agent     │     │              │     │    Agent     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │  1. Send Message   │                    │
       │───────────────────>│                    │
       │                    │                    │
       │                    │ 2. Validate        │
       │                    │    Schema          │
       │                    │                    │
       │                    │ 3. Check FATE      │
       │                    │    Gates           │
       │                    │                    │
       │                    │ 4. Route to        │
       │                    │    Receiver        │
       │                    │───────────────────>│
       │                    │                    │
       │                    │                    │ 5. Process
       │                    │                    │
       │                    │ 6. Response        │
       │                    │<───────────────────│
       │                    │                    │
       │  7. Return Result  │                    │
       │<───────────────────│                    │
       │                    │                    │
```

### Core Components

| Component | Purpose |
|-----------|---------|
| Agent Card | Declares agent capabilities |
| Task Card | Defines task structure |
| Message | Carries communication payload |
| Router | Matches tasks to agents |
| Validator | Ensures protocol compliance |

---

## Agent Cards

Agent Cards declare what an agent can do.

### Schema

```yaml
agent_card:
  id: string              # Unique identifier
  name: string            # Display name
  role: string            # Role in system
  capabilities:           # What agent can do
    - capability_id: string
      description: string
      input_schema: object
      output_schema: object
  giants: [string]        # Inspirational figures
  personality: string     # Agent personality
  specialties: [string]   # Areas of expertise
  constraints:            # Operational constraints
    max_concurrent: number
    timeout: duration
    requires_approval: [string]
```

### PAT Agent Cards

#### Strategist

```yaml
strategist:
  id: "pat_strategist"
  name: "Strategist"
  role: "Strategic planning and long-term thinking"
  capabilities:
    - capability_id: "strategic_analysis"
      description: "Analyze strategic situations"
      input_schema:
        type: object
        properties:
          context: {type: string}
          objective: {type: string}
          constraints: {type: array}
      output_schema:
        type: object
        properties:
          analysis: {type: string}
          recommendations: {type: array}
          risks: {type: array}

    - capability_id: "competitive_analysis"
      description: "Analyze competitive landscape"
      input_schema:
        type: object
        properties:
          market: {type: string}
          competitors: {type: array}

    - capability_id: "roadmap_creation"
      description: "Create strategic roadmaps"
      input_schema:
        type: object
        properties:
          goals: {type: array}
          timeline: {type: string}
          resources: {type: object}

  giants: ["Sun Tzu", "Clausewitz", "Porter"]
  personality: "Thoughtful, analytical, long-term focused"
  specialties:
    - "Strategic planning"
    - "Market analysis"
    - "Risk assessment"
    - "Goal setting"
```

#### Researcher

```yaml
researcher:
  id: "pat_researcher"
  name: "Researcher"
  role: "Knowledge discovery and synthesis"
  capabilities:
    - capability_id: "deep_research"
      description: "Comprehensive multi-source research"
      input_schema:
        type: object
        properties:
          topic: {type: string, required: true}
          depth: {type: string, enum: [summary, moderate, thorough]}
          sources: {type: array}
      output_schema:
        type: object
        properties:
          findings: {type: array}
          synthesis: {type: string}
          sources: {type: array}
          confidence: {type: number}

    - capability_id: "fact_verification"
      description: "Verify factual claims"
      input_schema:
        type: object
        properties:
          claim: {type: string}
          context: {type: string}

    - capability_id: "knowledge_synthesis"
      description: "Synthesize multiple sources"
      input_schema:
        type: object
        properties:
          sources: {type: array}
          focus: {type: string}

  giants: ["Shannon", "Turing", "Dijkstra"]
  personality: "Curious, thorough, evidence-based"
  specialties:
    - "Web research"
    - "Academic search"
    - "Fact verification"
    - "Knowledge synthesis"
```

#### Developer

```yaml
developer:
  id: "pat_developer"
  name: "Developer"
  role: "Code implementation and technical solutions"
  capabilities:
    - capability_id: "code_generation"
      description: "Generate code implementations"
      input_schema:
        type: object
        properties:
          description: {type: string, required: true}
          language: {type: string}
          style: {type: string, enum: [minimal, standard, verbose]}
          include_tests: {type: boolean}
      output_schema:
        type: object
        properties:
          code: {type: string}
          explanation: {type: string}
          tests: {type: string}

    - capability_id: "code_review"
      description: "Review code for quality"
      input_schema:
        type: object
        properties:
          code: {type: string}
          focus: {type: array}

    - capability_id: "debugging"
      description: "Debug code issues"
      input_schema:
        type: object
        properties:
          code: {type: string}
          error: {type: string}
          context: {type: string}

    - capability_id: "refactoring"
      description: "Refactor code for improvement"
      input_schema:
        type: object
        properties:
          code: {type: string}
          goal: {type: string}

  giants: ["Knuth", "Ritchie", "Torvalds"]
  personality: "Pragmatic, precise, efficient"
  specialties:
    - "Code generation"
    - "Debugging"
    - "Architecture design"
    - "Testing"
```

#### Guardian

```yaml
guardian:
  id: "pat_guardian"
  name: "Guardian"
  role: "Ethics, safety, and oversight"
  capabilities:
    - capability_id: "fate_validation"
      description: "Validate against FATE gates"
      input_schema:
        type: object
        properties:
          content: {type: string, required: true}
          context: {type: string}
          strict: {type: boolean}
      output_schema:
        type: object
        properties:
          passed: {type: boolean}
          gates:
            ihsan: {type: number}
            adl: {type: number}
            harm: {type: number}
            confidence: {type: number}
          issues: {type: array}

    - capability_id: "ethics_review"
      description: "Review for ethical concerns"
      input_schema:
        type: object
        properties:
          action: {type: string}
          context: {type: string}

    - capability_id: "risk_assessment"
      description: "Assess risks of action"
      input_schema:
        type: object
        properties:
          action: {type: string}
          impact: {type: string}

    - capability_id: "veto"
      description: "Block harmful actions"
      input_schema:
        type: object
        properties:
          action_id: {type: string}
          reason: {type: string}

  giants: ["Al-Ghazali", "Rawls", "Anthropic"]
  personality: "Protective, ethical, watchful"
  specialties:
    - "Ethics review"
    - "Safety validation"
    - "Risk monitoring"
    - "Harm prevention"
  constraints:
    always_consulted: true
    can_veto: true
```

---

## Task Cards

Task Cards define structured work units.

### Schema

```yaml
task_card:
  id: string                    # Unique identifier
  type: string                  # Task type
  title: string                 # Human-readable title
  description: string           # Detailed description
  input_schema: object          # Expected input structure
  output_schema: object         # Expected output structure
  suitable_agents: [string]     # Agents that can handle this
  requires_guardian: boolean    # Needs Guardian review
  priority: string              # Priority level
  timeout: duration             # Max execution time
```

### Standard Task Cards

#### Research Task

```yaml
research_task:
  id: "task_research"
  type: "research"
  title: "Research Task"
  description: "Gather and synthesize information on a topic"
  input_schema:
    type: object
    required: [topic]
    properties:
      topic:
        type: string
        description: "Topic to research"
      depth:
        type: string
        enum: [summary, moderate, thorough]
        default: moderate
      sources:
        type: array
        items: {type: string}
        default: [web, academic, memory]
  output_schema:
    type: object
    properties:
      findings:
        type: array
        items:
          type: object
          properties:
            fact: {type: string}
            source: {type: string}
            confidence: {type: number}
      synthesis:
        type: string
      recommendations:
        type: array
  suitable_agents: [researcher]
  requires_guardian: false
  priority: normal
  timeout: 10m
```

#### Implementation Task

```yaml
implementation_task:
  id: "task_implement"
  type: "implementation"
  title: "Implementation Task"
  description: "Implement code solution"
  input_schema:
    type: object
    required: [description]
    properties:
      description:
        type: string
        description: "What to implement"
      language:
        type: string
        default: auto
      include_tests:
        type: boolean
        default: true
      style:
        type: string
        enum: [minimal, standard, verbose]
        default: standard
  output_schema:
    type: object
    properties:
      code:
        type: string
      tests:
        type: string
      explanation:
        type: string
      files_created:
        type: array
  suitable_agents: [developer]
  requires_guardian: true
  priority: normal
  timeout: 30m
```

#### Review Task

```yaml
review_task:
  id: "task_review"
  type: "review"
  title: "Review Task"
  description: "Review code or content for quality"
  input_schema:
    type: object
    required: [target]
    properties:
      target:
        type: string
        description: "What to review"
      focus:
        type: array
        items:
          type: string
          enum: [security, performance, style, logic, all]
        default: [all]
      depth:
        type: string
        enum: [quick, standard, thorough]
        default: standard
  output_schema:
    type: object
    properties:
      passed:
        type: boolean
      issues:
        type: array
        items:
          type: object
          properties:
            severity: {type: string}
            location: {type: string}
            description: {type: string}
            suggestion: {type: string}
      score:
        type: number
      recommendations:
        type: array
  suitable_agents: [reviewer, guardian]
  requires_guardian: false
  priority: normal
  timeout: 15m
```

#### Execution Task

```yaml
execution_task:
  id: "task_execute"
  type: "execution"
  title: "Execution Task"
  description: "Execute a command or deployment"
  input_schema:
    type: object
    required: [command]
    properties:
      command:
        type: string
        description: "Command to execute"
      environment:
        type: string
        enum: [development, staging, production]
        default: development
      dry_run:
        type: boolean
        default: false
  output_schema:
    type: object
    properties:
      success:
        type: boolean
      output:
        type: string
      error:
        type: string
      duration:
        type: number
  suitable_agents: [executor]
  requires_guardian: true
  priority: high
  timeout: 60m
```

---

## Message Types

### Request Message

```yaml
request:
  id: uuid
  type: "request"
  timestamp: datetime
  sender:
    agent_id: string
    session_id: string
  recipient:
    agent_id: string          # Specific agent
    capability_id: string     # Or capability (auto-route)
  task:
    card_id: string
    input: object
  priority: string
  timeout: duration
  context:
    conversation_id: string
    parent_message_id: string
    metadata: object
```

### Response Message

```yaml
response:
  id: uuid
  type: "response"
  timestamp: datetime
  request_id: uuid
  sender:
    agent_id: string
  status: string              # success | error | partial
  result:
    output: object            # Task output
    fate_gates:               # FATE validation results
      ihsan: number
      adl: number
      harm: number
      confidence: number
    metadata: object
  error:                      # If status == error
    code: string
    message: string
    details: object
```

### Event Message

```yaml
event:
  id: uuid
  type: "event"
  timestamp: datetime
  source:
    agent_id: string
  event_type: string          # task_started | task_completed | alert | etc.
  payload: object
  subscribers: [string]       # Who should receive
```

### Broadcast Message

```yaml
broadcast:
  id: uuid
  type: "broadcast"
  timestamp: datetime
  source:
    agent_id: string
  topic: string
  message: string
  data: object
  recipients: "all" | [string]
```

---

## Communication Patterns

### Request-Response

Standard query-response pattern.

```
┌──────────┐                    ┌──────────┐
│  Agent A │                    │  Agent B │
└────┬─────┘                    └────┬─────┘
     │                               │
     │  Request (task + input)       │
     │──────────────────────────────>│
     │                               │
     │                               │ Process
     │                               │
     │  Response (result)            │
     │<──────────────────────────────│
     │                               │
```

**Example:**
```yaml
# Request
request:
  sender: {agent_id: "user_session"}
  recipient: {capability_id: "deep_research"}
  task:
    card_id: "task_research"
    input:
      topic: "Byzantine fault tolerance"
      depth: "thorough"

# Response
response:
  status: "success"
  result:
    output:
      findings: [...]
      synthesis: "BFT allows distributed systems to..."
    fate_gates:
      ihsan: 0.96
      adl: 0.28
      harm: 0.05
      confidence: 0.91
```

### Collaboration

Multi-agent collaborative pattern.

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Strategist│    │Researcher│    │Developer │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     │ 1. Frame problem              │
     │──────────────────────────────>│
     │               │               │
     │ 2. Research   │               │
     │               │<──────────────│
     │               │               │
     │ 3. Research results           │
     │<──────────────│               │
     │               │               │
     │ 4. Implement  │               │
     │──────────────────────────────>│
     │               │               │
     │ 5. Implementation             │
     │<──────────────────────────────│
```

### Pipeline

Sequential processing pattern.

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│Researcher│ → │Developer │ → │ Reviewer │ → │ Guardian │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │
     │ Research     │ Implement    │ Review       │ Approve
     │──────────────>──────────────>──────────────>
```

**Configuration:**
```yaml
pipeline:
  name: "feature_development"
  stages:
    - agent: researcher
      task: research_task
      pass_output_to: next
    - agent: developer
      task: implementation_task
      pass_output_to: next
    - agent: reviewer
      task: review_task
      pass_output_to: next
    - agent: guardian
      task: approval_task
      final: true
```

### Consensus

Multi-agent agreement pattern.

```
┌──────────┐
│ Guardian │ ← Mediator
└────┬─────┘
     │
     ├──────────────┬──────────────┐
     │              │              │
┌────▼────┐   ┌────▼────┐   ┌────▼────┐
│Strategist│   │Developer│   │Reviewer │
└────┬────┘   └────┬────┘   └────┬────┘
     │              │              │
     │   Opinions   │              │
     └──────────────┴──────────────┘
                    │
                    ▼
              ┌──────────┐
              │ Decision │
              └──────────┘
```

**Configuration:**
```yaml
consensus:
  topic: "Architecture decision"
  participants:
    - strategist
    - developer
    - reviewer
  mediator: guardian
  threshold: 0.6        # 60% agreement needed
  timeout: 30m
  fallback: "guardian_decides"
```

---

## Routing Rules

### Capability-Based Routing

```yaml
routing:
  mode: "capability"
  rules:
    - capability: "deep_research"
      agents: [researcher]

    - capability: "code_generation"
      agents: [developer]

    - capability: "ethics_review"
      agents: [guardian]
      required: true        # Always include
```

### Pattern-Based Routing

```yaml
routing:
  mode: "pattern"
  rules:
    - pattern: "research|investigate|find|discover"
      agent: researcher
      confidence: 0.8

    - pattern: "implement|build|code|create"
      agent: developer
      confidence: 0.9

    - pattern: "review|check|audit|validate"
      agent: reviewer
      confidence: 0.85

    - pattern: "deploy|run|execute|start"
      agent: executor
      confidence: 0.9
      requires_guardian: true
```

### Priority-Based Routing

```yaml
routing:
  mode: "priority"
  rules:
    - priority: "critical"
      agents: [guardian, executor]
      timeout: "5m"

    - priority: "high"
      agents: [developer, reviewer]
      timeout: "15m"

    - priority: "normal"
      agents: "auto"
      timeout: "30m"
```

### Load Balancing

```yaml
routing:
  load_balancing:
    enabled: true
    strategy: "least_busy"    # least_busy | round_robin | random
    max_concurrent:
      developer: 3
      researcher: 2
      reviewer: 2
      guardian: unlimited
```

---

## FATE Integration

All A2A communications pass through FATE validation.

### Pre-Send Validation

```yaml
fate_integration:
  pre_send:
    enabled: true
    gates:
      - harm              # Check for harmful requests
    action_on_fail: "block"
```

### Post-Receive Validation

```yaml
fate_integration:
  post_receive:
    enabled: true
    gates:
      - ihsan            # Quality check
      - adl              # Fairness check
      - harm             # Harm check
      - confidence       # Confidence check
    action_on_fail:
      ihsan: "request_improvement"
      adl: "rebalance"
      harm: "block"
      confidence: "add_disclaimer"
```

### Guardian Override

```yaml
fate_integration:
  guardian_override:
    enabled: true
    requires:
      - guardian_approval
      - human_confirmation
      - audit_log
    max_duration: "1h"
```

---

## Protocol Examples

### Example 1: Research Request

```yaml
# Step 1: User sends request
request:
  id: "req_001"
  sender: {agent_id: "user_session"}
  recipient: {capability_id: "deep_research"}
  task:
    card_id: "task_research"
    input:
      topic: "Rust async patterns"
      depth: "thorough"

# Step 2: Router matches to researcher
routing_decision:
  request_id: "req_001"
  matched_agent: "researcher"
  confidence: 0.95
  reason: "capability_match"

# Step 3: Researcher processes
processing:
  agent: "researcher"
  task: "task_research"
  steps:
    - action: "query_analysis"
    - action: "web_search"
    - action: "memory_search"
    - action: "synthesis"

# Step 4: FATE validation
fate_validation:
  ihsan: 0.97
  adl: 0.25
  harm: 0.02
  confidence: 0.93
  passed: true

# Step 5: Response returned
response:
  id: "res_001"
  request_id: "req_001"
  status: "success"
  result:
    output:
      findings: [
        {fact: "Tokio is the most popular async runtime", source: "crates.io stats", confidence: 0.95},
        {fact: "async-std provides std-like API", source: "docs.rs", confidence: 0.92}
      ]
      synthesis: "Rust async patterns center around..."
```

### Example 2: Code Review Pipeline

```yaml
# Pipeline definition
pipeline:
  id: "pipeline_code_review"
  stages:
    - stage: 1
      agent: developer
      task: code_analysis
    - stage: 2
      agent: reviewer
      task: quality_review
    - stage: 3
      agent: guardian
      task: security_review

# Stage 1: Developer analysis
stage_1_request:
  sender: {agent_id: "pipeline_orchestrator"}
  recipient: {agent_id: "developer"}
  task:
    card_id: "code_analysis"
    input:
      code: "fn main() { ... }"

stage_1_response:
  status: "success"
  result:
    analysis:
      complexity: "low"
      patterns: ["error_handling", "async"]
      concerns: []

# Stage 2: Reviewer quality check
stage_2_request:
  sender: {agent_id: "pipeline_orchestrator"}
  recipient: {agent_id: "reviewer"}
  task:
    card_id: "quality_review"
    input:
      code: "fn main() { ... }"
      previous_analysis: {stage_1_result}

stage_2_response:
  status: "success"
  result:
    passed: true
    score: 0.92
    issues: []

# Stage 3: Guardian security review
stage_3_request:
  sender: {agent_id: "pipeline_orchestrator"}
  recipient: {agent_id: "guardian"}
  task:
    card_id: "security_review"
    input:
      code: "fn main() { ... }"
      previous_stages: [{stage_1_result}, {stage_2_result}]

stage_3_response:
  status: "success"
  result:
    approved: true
    fate_gates:
      ihsan: 0.94
      adl: 0.22
      harm: 0.08
      confidence: 0.91
```

---

## Implementation Guide

### Adding a New Agent

1. **Define Agent Card**
```yaml
my_agent:
  id: "pat_my_agent"
  name: "My Agent"
  role: "Specialized task"
  capabilities:
    - capability_id: "my_capability"
      # ...
```

2. **Register with Router**
```yaml
routing:
  rules:
    - capability: "my_capability"
      agents: [my_agent]
```

3. **Implement Handler**
```rust
impl Agent for MyAgent {
    fn handle(&self, request: Request) -> Response {
        // Process request
        // Validate output with FATE
        // Return response
    }
}
```

### Adding a New Task Type

1. **Define Task Card**
```yaml
my_task:
  id: "task_my_task"
  type: "my_type"
  input_schema: {...}
  output_schema: {...}
  suitable_agents: [...]
```

2. **Register with System**
```yaml
task_registry:
  tasks:
    - my_task
```

### Custom Routing Rule

```yaml
routing:
  custom_rules:
    - name: "my_rule"
      condition: "task.type == 'my_type' && context.priority == 'high'"
      agent: "my_agent"
      priority: 100
```

---

## Next Steps

- [Config Reference](CONFIG-REFERENCE.md) — Full configuration options
- [FATE Gates](FATE-GATES.md) — FATE gate specification
- [PAT Agents](../guides/04-PAT-AGENTS.md) — Agent capabilities

---

**Structured communication for sovereign intelligence.** 🔗
