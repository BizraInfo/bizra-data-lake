# Agent Architecture

The PAT (Personal Agentic Team) system architecture.

## Table of Contents

1. [Overview](#overview)
2. [Agent Model](#agent-model)
3. [Agent Lifecycle](#agent-lifecycle)
4. [Agent Capabilities](#agent-capabilities)
5. [Agent Communication](#agent-communication)
6. [Agent Coordination](#agent-coordination)
7. [Guardian Role](#guardian-role)
8. [Extension Model](#extension-model)

---

## Overview

The PAT system provides 7 specialized agents that collaborate to serve the user.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PAT ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ┌───────────────┐                                 │
│                           │   GUARDIAN    │                                 │
│                           │   🛡 Ethics   │                                 │
│                           │   Oversight   │                                 │
│                           └───────┬───────┘                                 │
│                                   │                                         │
│               ┌───────────────────┼───────────────────┐                    │
│               │                   │                   │                    │
│               ▼                   ▼                   ▼                    │
│       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│       │ STRATEGIST  │     │  ANALYST    │     │  EXECUTOR   │            │
│       │ ♟ Planning  │     │ 📊 Data     │     │ ▶ Actions   │            │
│       └──────┬──────┘     └──────┬──────┘     └──────┬──────┘            │
│              │                   │                   │                    │
│              └───────────────────┼───────────────────┘                    │
│                                  │                                         │
│               ┌──────────────────┼──────────────────┐                     │
│               │                  │                  │                     │
│               ▼                  ▼                  ▼                     │
│       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│       │ RESEARCHER  │     │  DEVELOPER  │     │  REVIEWER   │            │
│       │ 🔍 Knowledge │     │ ⚙ Code      │     │ ✓ Quality   │            │
│       └─────────────┘     └─────────────┘     └─────────────┘            │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────── │
│                                                                             │
│               ┌─────────────────────────────────────────┐                  │
│               │            A2A MESSAGE BUS              │                  │
│               │  Task Cards • Agent Cards • Events      │                  │
│               └─────────────────────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Specialization** — Each agent excels in specific domains
2. **Collaboration** — Agents work together on complex tasks
3. **Guardian Oversight** — All significant actions reviewed
4. **Capability-Based** — Explicit permissions per agent
5. **Personality-Driven** — Consistent behavior from "giants"

---

## Agent Model

### Agent Structure

```rust
/// Core agent definition
pub struct Agent {
    // Identity
    pub id: AgentId,
    pub role: PATRole,
    pub name: String,

    // Personality
    pub giants: Vec<Giant>,           // Inspirational figures
    pub personality: Personality,     // Behavioral traits
    pub voice_config: VoiceConfig,    // Voice settings

    // Capabilities
    pub capabilities: Vec<Capability>,
    pub constraints: AgentConstraints,

    // State
    pub state: AgentState,
    pub current_task: Option<TaskId>,
    pub task_history: TaskHistory,

    // Runtime
    pub inference_context: InferenceContext,
    pub memory_access: MemoryAccess,
}

/// Agent roles in PAT
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PATRole {
    Strategist,   // ♟ Strategic planning
    Researcher,   // 🔍 Knowledge discovery
    Developer,    // ⚙ Code implementation
    Analyst,      // 📊 Data analysis
    Reviewer,     // ✓ Quality assurance
    Executor,     // ▶ Task execution
    Guardian,     // 🛡 Ethics oversight
}

/// Agent operational state
pub enum AgentState {
    Ready,                    // Available for tasks
    Active(TaskId),           // Processing a task
    Waiting(WaitReason),      // Waiting for dependency
    Paused(PauseReason),      // Temporarily paused
    Error(AgentError),        // Error state
}
```

### Agent Personality

```rust
/// Personality traits from giants
pub struct Personality {
    pub traits: Vec<Trait>,
    pub communication_style: CommunicationStyle,
    pub decision_approach: DecisionApproach,
    pub risk_tolerance: RiskLevel,
}

/// Giants that inspire agent behavior
pub struct Giant {
    pub name: String,
    pub domain: String,
    pub key_principles: Vec<String>,
    pub influence_weight: f64,
}

// Example: Guardian's giants
impl Guardian {
    fn default_giants() -> Vec<Giant> {
        vec![
            Giant {
                name: "Al-Ghazali".to_string(),
                domain: "Islamic Ethics",
                key_principles: vec![
                    "Inner intention matters".to_string(),
                    "Knowledge requires wisdom".to_string(),
                ],
                influence_weight: 0.4,
            },
            Giant {
                name: "John Rawls".to_string(),
                domain: "Justice Theory",
                key_principles: vec![
                    "Veil of ignorance".to_string(),
                    "Fair opportunity".to_string(),
                ],
                influence_weight: 0.3,
            },
            Giant {
                name: "Anthropic".to_string(),
                domain: "AI Safety",
                key_principles: vec![
                    "Helpful, harmless, honest".to_string(),
                    "Constitutional AI".to_string(),
                ],
                influence_weight: 0.3,
            },
        ]
    }
}
```

---

## Agent Lifecycle

### State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENT LIFECYCLE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ┌──────────────┐                                 │
│                           │   CREATED    │                                 │
│                           └──────┬───────┘                                 │
│                                  │ initialize()                            │
│                                  ▼                                         │
│                           ┌──────────────┐                                 │
│                      ┌───>│    READY     │<──────────────────┐            │
│                      │    └──────┬───────┘                   │            │
│                      │           │ assign_task()             │            │
│                      │           ▼                           │            │
│                      │    ┌──────────────┐                   │            │
│                      │    │    ACTIVE    │──────────────┐    │            │
│                      │    └──────┬───────┘              │    │            │
│                      │           │                       │    │            │
│          complete()  │           │ need_dependency()    │    │ timeout()  │
│                      │           ▼                       │    │            │
│                      │    ┌──────────────┐              │    │            │
│                      │    │   WAITING    │              │    │            │
│                      │    └──────┬───────┘              │    │            │
│                      │           │ dependency_met()     │    │            │
│                      │           ▼                       ▼    │            │
│                      │    ┌──────────────┐       ┌──────────────┐         │
│                      └────│   COMPLETE   │       │    ERROR     │         │
│                           └──────────────┘       └──────┬───────┘         │
│                                                         │ recover()       │
│                                                         └─────────────────┘
│                                                                             │
│   Guardian can force any agent to:                                         │
│   • PAUSED (via veto)                                                      │
│   • READY (via release)                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Lifecycle Events

```rust
/// Agent lifecycle event
pub enum AgentEvent {
    // State transitions
    Created { agent_id: AgentId },
    Initialized { agent_id: AgentId, capabilities: Vec<Capability> },
    TaskAssigned { agent_id: AgentId, task_id: TaskId },
    TaskStarted { agent_id: AgentId, task_id: TaskId },
    WaitingForDependency { agent_id: AgentId, dependency: Dependency },
    DependencyMet { agent_id: AgentId, dependency: Dependency },
    TaskCompleted { agent_id: AgentId, task_id: TaskId, result: TaskResult },

    // Guardian events
    Paused { agent_id: AgentId, reason: PauseReason },
    Resumed { agent_id: AgentId },
    Vetoed { agent_id: AgentId, action: Action, reason: String },

    // Error events
    Error { agent_id: AgentId, error: AgentError },
    Recovered { agent_id: AgentId },
}

/// Agent event handler
impl Agent {
    async fn handle_event(&mut self, event: AgentEvent) -> Result<()> {
        match event {
            AgentEvent::TaskAssigned { task_id, .. } => {
                self.state = AgentState::Active(task_id);
                self.current_task = Some(task_id);
                self.emit_event(AgentEvent::TaskStarted {
                    agent_id: self.id.clone(),
                    task_id,
                });
            }
            AgentEvent::Paused { reason, .. } => {
                self.state = AgentState::Paused(reason);
                self.save_checkpoint()?;
            }
            // ... other handlers
        }
        Ok(())
    }
}
```

---

## Agent Capabilities

### Capability Model

```rust
/// Agent capability definition
pub struct Capability {
    pub id: CapabilityId,
    pub name: String,
    pub description: String,

    // Schema
    pub input_schema: JsonSchema,
    pub output_schema: JsonSchema,

    // Constraints
    pub requires_approval: bool,
    pub max_concurrent: u32,
    pub timeout: Duration,

    // Dependencies
    pub required_tools: Vec<ToolId>,
    pub required_permissions: Vec<Permission>,
}

/// Permission types
pub enum Permission {
    // Memory permissions
    MemoryRead,
    MemoryWrite(MemoryCategory),
    MemoryDelete,

    // Code permissions
    CodeRead,
    CodeWrite,
    CodeExecute(ExecutionScope),

    // Network permissions
    NetworkLocal,
    NetworkExternal(Vec<String>),

    // Agent permissions
    AgentInvoke(AgentId),
    AgentDelegate,

    // Special permissions
    GuardianApproval,
    HumanConfirmation,
}
```

### Capabilities by Agent

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT CAPABILITIES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STRATEGIST (♟)                                                           │
│   ├── strategic_analysis      - Analyze situations strategically           │
│   ├── competitive_analysis    - Competitive landscape analysis             │
│   ├── roadmap_creation        - Create strategic roadmaps                  │
│   ├── risk_assessment         - Evaluate risks and opportunities           │
│   └── goal_planning           - Define and track goals                     │
│       Permissions: memory:read, memory:write:decisions                     │
│                                                                             │
│   RESEARCHER (🔍)                                                           │
│   ├── deep_research           - Multi-source research synthesis            │
│   ├── fact_verification       - Verify factual claims                      │
│   ├── literature_review       - Academic paper analysis                    │
│   ├── knowledge_synthesis     - Combine multiple sources                   │
│   └── trend_analysis          - Identify patterns and trends               │
│       Permissions: memory:read, memory:write:learnings, network:external   │
│                                                                             │
│   DEVELOPER (⚙)                                                            │
│   ├── code_generation         - Generate code implementations              │
│   ├── code_review             - Review code quality                        │
│   ├── debugging               - Debug issues                               │
│   ├── refactoring             - Improve code structure                     │
│   └── test_generation         - Generate test cases                        │
│       Permissions: code:read, code:write, code:execute:sandbox             │
│                                                                             │
│   ANALYST (📊)                                                              │
│   ├── data_analysis           - Analyze datasets                           │
│   ├── statistical_modeling    - Statistical analysis                       │
│   ├── visualization           - Create visualizations                      │
│   ├── pattern_detection       - Find patterns                              │
│   └── forecasting             - Predictive analysis                        │
│       Permissions: memory:read, data:read, data:transform                  │
│                                                                             │
│   REVIEWER (✓)                                                             │
│   ├── code_audit              - Audit code for issues                      │
│   ├── security_review         - Security assessment                        │
│   ├── compliance_check        - Compliance validation                      │
│   ├── documentation_review    - Review documentation                       │
│   └── performance_review      - Performance assessment                     │
│       Permissions: code:read, memory:read, report:write                    │
│                                                                             │
│   EXECUTOR (▶)                                                             │
│   ├── command_execution       - Run system commands                        │
│   ├── deployment              - Deploy to environments                     │
│   ├── automation              - Create automations                         │
│   └── batch_processing        - Batch operations                           │
│       Permissions: code:execute (REQUIRES guardian:approval)               │
│                                                                             │
│   GUARDIAN (🛡)                                                            │
│   ├── fate_validation         - Validate FATE gates                        │
│   ├── ethics_review           - Review for ethics                          │
│   ├── risk_assessment         - Assess action risks                        │
│   ├── veto                    - Block harmful actions                      │
│   ├── approve                 - Approve sensitive actions                  │
│   └── escalate                - Escalate to human                          │
│       Permissions: ALL (can override)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Communication

### Message Types

```rust
/// A2A message types
pub enum A2AMessage {
    // Task-related
    TaskRequest {
        from: AgentId,
        to: AgentId,
        task: TaskCard,
    },
    TaskResponse {
        from: AgentId,
        to: AgentId,
        request_id: MessageId,
        result: TaskResult,
    },

    // Collaboration
    CollaborationInvite {
        initiator: AgentId,
        participants: Vec<AgentId>,
        topic: String,
    },
    PerspectiveRequest {
        from: AgentId,
        topic: String,
    },
    PerspectiveResponse {
        from: AgentId,
        perspective: Perspective,
    },

    // Guardian specific
    ApprovalRequest {
        from: AgentId,
        action: Action,
        context: Context,
    },
    ApprovalResponse {
        approved: bool,
        reason: String,
        conditions: Vec<Condition>,
    },
    Veto {
        agent_id: AgentId,
        action: Action,
        reason: String,
    },

    // Events
    Event {
        source: AgentId,
        event_type: EventType,
        payload: Value,
    },
}
```

### Communication Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMMUNICATION PATTERNS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. DIRECT REQUEST-RESPONSE                                               │
│      ─────────────────────────                                             │
│                                                                             │
│      Agent A ──── TaskRequest ────> Agent B                                │
│             <─── TaskResponse ────                                         │
│                                                                             │
│   2. PIPELINE                                                              │
│      ────────                                                              │
│                                                                             │
│      A ─→ B ─→ C ─→ D                                                     │
│      (Each passes output to next)                                          │
│                                                                             │
│   3. BROADCAST                                                             │
│      ─────────                                                             │
│                                                                             │
│              ┌─→ B                                                         │
│      A ──────┼─→ C                                                         │
│              └─→ D                                                         │
│                                                                             │
│   4. CONSENSUS                                                             │
│      ─────────                                                             │
│                                                                             │
│           ┌─── B ───┐                                                     │
│      A ───┼─── C ───┼───> Guardian (Mediator)                             │
│           └─── D ───┘                                                     │
│                                                                             │
│   5. SWARM                                                                 │
│      ─────                                                                 │
│                                                                             │
│      ┌──────────────────────────┐                                         │
│      │  Coordinator (Guardian)   │                                         │
│      └────────────┬─────────────┘                                         │
│           ┌───────┼───────┐                                               │
│           ▼       ▼       ▼                                               │
│         ┌───┐   ┌───┐   ┌───┐                                            │
│         │ B │   │ C │   │ D │  (Parallel work)                           │
│         └───┘   └───┘   └───┘                                            │
│           │       │       │                                               │
│           └───────┼───────┘                                               │
│                   ▼                                                        │
│            Synthesis                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Coordination

### Task Routing

```rust
/// Route task to appropriate agent
pub async fn route_task(task: &TaskCard, context: &Context) -> Result<AgentId> {
    // 1. Check explicit routing
    if let Some(target) = task.target_agent {
        return Ok(target);
    }

    // 2. Match by capability
    let matching_agents = find_agents_with_capability(&task.required_capability);

    // 3. Filter by availability
    let available = matching_agents
        .iter()
        .filter(|a| a.state == AgentState::Ready)
        .collect::<Vec<_>>();

    // 4. Load balance
    let selected = select_least_loaded(&available);

    // 5. Notify Guardian for significant tasks
    if task.is_significant() {
        notify_guardian(&task, &selected).await?;
    }

    Ok(selected.id)
}

/// Task routing rules
pub struct RoutingRules {
    pub capability_routes: HashMap<CapabilityId, Vec<AgentId>>,
    pub pattern_routes: Vec<PatternRoute>,
    pub fallback_agent: AgentId,  // Usually Guardian
}

pub struct PatternRoute {
    pub pattern: Regex,
    pub agent: AgentId,
    pub confidence: f64,
}
```

### Collaboration Protocol

```rust
/// Multi-agent collaboration session
pub struct CollaborationSession {
    pub id: SessionId,
    pub topic: String,
    pub participants: Vec<AgentId>,
    pub coordinator: AgentId,  // Usually Guardian
    pub mode: CollaborationMode,
    pub state: CollaborationState,
}

pub enum CollaborationMode {
    Parallel,     // All work simultaneously
    Sequential,   // Work in order
    Consensus,    // Must agree
    Competitive,  // Best answer wins
}

impl CollaborationSession {
    /// Execute consensus-based collaboration
    pub async fn run_consensus(&self) -> Result<Decision> {
        // 1. Gather perspectives
        let mut perspectives = Vec::new();
        for agent in &self.participants {
            let perspective = agent.get_perspective(&self.topic).await?;
            perspectives.push(perspective);
        }

        // 2. Analyze for consensus
        let agreement_level = calculate_agreement(&perspectives);

        // 3. If no consensus, mediate
        if agreement_level < self.consensus_threshold {
            let mediation = self.coordinator.mediate(&perspectives).await?;
            return Ok(mediation);
        }

        // 4. Synthesize decision
        let decision = synthesize_decision(&perspectives);

        // 5. Guardian review
        let approved = Guardian::review_decision(&decision).await?;
        if !approved {
            return Err(CollaborationError::GuardianRejected);
        }

        Ok(decision)
    }
}
```

---

## Guardian Role

### Guardian Special Powers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GUARDIAN SPECIAL POWERS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. VETO POWER                                                            │
│      ──────────                                                            │
│      Can block ANY agent action that fails FATE gates or poses risk.       │
│      Veto is immediate and non-negotiable.                                 │
│                                                                             │
│   2. APPROVAL AUTHORITY                                                    │
│      ──────────────────                                                    │
│      Must approve:                                                         │
│      • All Executor commands                                               │
│      • Production deployments                                              │
│      • External communications                                             │
│      • Data deletions                                                      │
│                                                                             │
│   3. ESCALATION AUTHORITY                                                  │
│      ────────────────────                                                  │
│      Can escalate to human when:                                           │
│      • Uncertainty exceeds threshold                                       │
│      • Novel situation encountered                                         │
│      • Conflicting directives                                              │
│                                                                             │
│   4. MEDIATION AUTHORITY                                                   │
│      ───────────────────                                                   │
│      Resolves conflicts between agents in consensus mode.                  │
│      Final arbiter when agents disagree.                                   │
│                                                                             │
│   5. OVERRIDE AUTHORITY                                                    │
│      ──────────────────                                                    │
│      Can override agent decisions for safety.                              │
│      All overrides logged and audited.                                     │
│                                                                             │
│   6. ALWAYS WATCHING                                                       │
│      ──────────────                                                        │
│      Guardian receives all significant events.                             │
│      Continuous background monitoring.                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Guardian Integration

```rust
impl Guardian {
    /// Review action before execution
    pub async fn review_action(&self, action: &Action) -> Result<ReviewResult> {
        // 1. Check FATE gates
        let fate_result = self.check_fate_gates(action).await?;
        if !fate_result.passed {
            return Ok(ReviewResult::Blocked {
                reason: format!("FATE gate failure: {:?}", fate_result.failed_gates),
            });
        }

        // 2. Check for patterns requiring review
        if self.matches_review_pattern(action) {
            let human_review = self.request_human_review(action).await?;
            if !human_review.approved {
                return Ok(ReviewResult::Blocked {
                    reason: "Human review declined".to_string(),
                });
            }
        }

        // 3. Check agent permissions
        if !self.verify_agent_permissions(action) {
            return Ok(ReviewResult::Blocked {
                reason: "Insufficient permissions".to_string(),
            });
        }

        // 4. Log approval
        self.audit_log.log(AuditEntry::ActionApproved {
            action: action.clone(),
            timestamp: Utc::now(),
            guardian_id: self.id.clone(),
        });

        Ok(ReviewResult::Approved {
            conditions: vec![],
        })
    }

    /// FATE gate validation
    async fn check_fate_gates(&self, action: &Action) -> Result<FATEResult> {
        let scores = FATEScores {
            ihsan: self.evaluate_ihsan(action).await?,
            adl: self.evaluate_adl(action).await?,
            harm: self.evaluate_harm(action).await?,
            confidence: self.evaluate_confidence(action).await?,
        };

        let thresholds = self.config.fate_thresholds;
        let mut failed_gates = Vec::new();

        if scores.ihsan < thresholds.ihsan {
            failed_gates.push(("ihsan", scores.ihsan, thresholds.ihsan));
        }
        if scores.adl > thresholds.adl {
            failed_gates.push(("adl", scores.adl, thresholds.adl));
        }
        if scores.harm > thresholds.harm {
            failed_gates.push(("harm", scores.harm, thresholds.harm));
        }
        if scores.confidence < thresholds.confidence {
            failed_gates.push(("confidence", scores.confidence, thresholds.confidence));
        }

        Ok(FATEResult {
            scores,
            passed: failed_gates.is_empty(),
            failed_gates,
        })
    }
}
```

---

## Extension Model

### Adding Custom Agents

```rust
/// Custom agent trait
pub trait CustomAgent: Agent {
    /// Agent-specific initialization
    fn initialize(&mut self, config: &AgentConfig) -> Result<()>;

    /// Handle agent-specific capabilities
    fn handle_capability(
        &self,
        capability: &str,
        input: Value,
    ) -> Result<Value>;

    /// Custom prompt construction
    fn build_prompt(&self, context: &Context) -> String;
}

/// Register custom agent
pub fn register_custom_agent(
    registry: &mut AgentRegistry,
    agent: Box<dyn CustomAgent>,
) -> Result<AgentId> {
    // Validate agent
    validate_agent(&agent)?;

    // Check capabilities don't conflict
    check_capability_conflicts(&registry, &agent)?;

    // Register with Guardian oversight
    let guardian_approval = Guardian::approve_new_agent(&agent)?;
    if !guardian_approval {
        return Err(AgentError::GuardianRejected);
    }

    // Add to registry
    let id = registry.add(agent);

    Ok(id)
}
```

### Plugin System

```yaml
# Agent plugin definition
plugin:
  name: "Custom Analyst"
  version: "1.0.0"
  author: "Your Name"

  agent:
    role: "custom_analyst"
    giants:
      - name: "Edward Tufte"
        domain: "Data Visualization"
    personality:
      traits: ["detail-oriented", "visual", "precise"]
      communication_style: "visual"

    capabilities:
      - id: "advanced_visualization"
        description: "Create advanced data visualizations"
        input_schema:
          type: object
          properties:
            data: {type: array}
            chart_type: {type: string}
        output_schema:
          type: object
          properties:
            visualization: {type: string}
            insights: {type: array}

    permissions:
      - "memory:read"
      - "data:read"
      - "visualization:create"

  hooks:
    on_activate: "initialize_visualization_engine"
    on_task: "prepare_data_context"
```

---

**Agents: Specialized intelligence, unified purpose.** 🤖
