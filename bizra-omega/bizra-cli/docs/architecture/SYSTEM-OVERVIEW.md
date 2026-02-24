# System Architecture Overview

Complete architectural documentation for BIZRA CLI.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Layers](#component-layers)
3. [Data Flow](#data-flow)
4. [Agent Architecture](#agent-architecture)
5. [Integration Architecture](#integration-architecture)
6. [Security Architecture](#security-architecture)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BIZRA CLI/TUI                                      │
│                         (Sovereign Command Center)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         PRESENTATION LAYER                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │   │
│  │  │   CLI Mode   │  │   TUI Mode   │  │  Voice Mode  │                   │   │
│  │  │  (Commands)  │  │  (Ratatui)   │  │ (PersonaPlex)│                   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         APPLICATION LAYER                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │   Commands   │  │    Skills    │  │    Hooks     │  │  Proactive   │ │   │
│  │  │   Handler    │  │    Engine    │  │    Engine    │  │    Engine    │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           AGENT LAYER                                    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│   │
│  │  │Strategy.│ │Research.│ │Develop. │ │Analyst  │ │Reviewer │ │Executor ││   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│   │
│  │                          ┌─────────┐                                     │   │
│  │                          │Guardian │ (Oversight)                         │   │
│  │                          └─────────┘                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  │                    A2A Protocol (Agent Communication)             │   │   │
│  │  └──────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          GATE LAYER (FATE)                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │    Ihsān     │  │     Adl      │  │     Harm     │  │  Confidence  │ │   │
│  │  │   (≥0.95)    │  │   (≤0.35)    │  │   (≤0.30)    │  │   (≥0.80)    │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        INTEGRATION LAYER                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │  LM Studio   │  │     MCP      │  │   Memory     │  │  Federation  │ │   │
│  │  │  (Inference) │  │  (Servers)   │  │  (Persist)   │  │  (Network)   │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Layers

### 1. Presentation Layer

**Purpose:** User interaction interfaces

| Component | Technology | Description |
|-----------|------------|-------------|
| CLI Mode | Clap | Command-line argument parsing |
| TUI Mode | Ratatui + Crossterm | Full-screen terminal UI |
| Voice Mode | PersonaPlex | Speech-to-text and text-to-speech |

```
User Input → Parser → Command/Message → Application Layer
                                              ↓
User Output ← Renderer ← Response ← Application Layer
```

### 2. Application Layer

**Purpose:** Business logic and workflow orchestration

| Component | Description |
|-----------|-------------|
| Commands Handler | Processes slash commands |
| Skills Engine | Executes multi-step workflows |
| Hooks Engine | Event-driven automation |
| Proactive Engine | Anticipation and suggestions |

```rust
// Command flow
async fn handle_command(cmd: Command) -> Result<Response> {
    // 1. Parse command
    let parsed = parser::parse(cmd)?;

    // 2. Route to skill
    let skill = skills::find(&parsed)?;

    // 3. Execute with hooks
    hooks::pre_execute(&parsed)?;
    let result = skill.execute(&parsed).await?;
    hooks::post_execute(&result)?;

    // 4. Pass through FATE gates
    fate::validate(&result)?;

    Ok(result)
}
```

### 3. Agent Layer

**Purpose:** Specialized AI agents for different domains

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Agent Registry                        │   │
│   │  - Agent discovery                                       │   │
│   │  - Capability matching                                   │   │
│   │  - Load balancing                                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    A2A Protocol                          │   │
│   │  - Task routing                                          │   │
│   │  - Message passing                                       │   │
│   │  - Consensus                                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│   ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐  │
│   │Strat. │ │Resear.│ │Devel. │ │Analys.│ │Review.│ │Execut.│  │
│   └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘  │
│                         │                                       │
│                    ┌─────────┐                                  │
│                    │Guardian │ ← Oversight of all agents        │
│                    └─────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Agent Lifecycle:**

```
1. IDLE      → Agent waiting for task
2. ASSIGNED  → Task received, analyzing
3. THINKING  → Processing with LLM
4. REVIEW    → Guardian reviewing (if needed)
5. COMPLETE  → Task done, result returned
```

### 4. Gate Layer (FATE)

**Purpose:** Ethical validation and quality control

```
                    Input
                      │
                      ▼
            ┌─────────────────┐
            │   Ihsān Gate    │ ── Excellence score ≥ 0.95?
            │   (Excellence)  │
            └────────┬────────┘
                     │ PASS
                     ▼
            ┌─────────────────┐
            │    Adl Gate     │ ── Gini coefficient ≤ 0.35?
            │   (Fairness)    │
            └────────┬────────┘
                     │ PASS
                     ▼
            ┌─────────────────┐
            │   Harm Gate     │ ── Harm score ≤ 0.30?
            │  (Prevention)   │
            └────────┬────────┘
                     │ PASS
                     ▼
            ┌─────────────────┐
            │ Confidence Gate │ ── Confidence ≥ 0.80?
            │   (Certainty)   │
            └────────┬────────┘
                     │ PASS
                     ▼
                  Output
```

### 5. Integration Layer

**Purpose:** External system connections

| Integration | Protocol | Purpose |
|-------------|----------|---------|
| LM Studio | OpenAI-compatible API | LLM inference |
| MCP Servers | MCP Protocol | Tool access |
| Memory | AgentDB | Persistent storage |
| Federation | gRPC + PBFT | Network consensus |

---

## Data Flow

### Request Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User    │───▶│  Parser  │───▶│  Router  │───▶│  Agent   │
│  Input   │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                     │
                                                     ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User    │◀───│ Renderer │◀───│   FATE   │◀───│   LLM    │
│  Output  │    │          │    │  Gates   │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Memory Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Session Memory                 Persistent Memory          │
│   ┌─────────────┐               ┌─────────────────────┐    │
│   │ Context     │               │ decisions/          │    │
│   │ Recent msgs │               │ learnings/          │    │
│   │ Current task│               │ preferences/        │    │
│   └──────┬──────┘               │ patterns/           │    │
│          │                      │ projects/           │    │
│          │  summarize           └──────────┬──────────┘    │
│          │  after 50K                      │               │
│          ▼                                 │               │
│   ┌─────────────┐                         │ store         │
│   │ Compressed  │────────────────────────▶│               │
│   │ Summary     │                          │               │
│   └─────────────┘                         ▼               │
│                                    ┌─────────────┐         │
│                                    │  AgentDB    │         │
│                                    │  (Vector +  │         │
│                                    │   Graph)    │         │
│                                    └─────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Agent Architecture

### Agent Structure

```rust
pub struct Agent {
    // Identity
    pub id: AgentId,
    pub role: PATRole,
    pub name: String,

    // Configuration
    pub system_prompt: String,
    pub capabilities: Vec<Capability>,
    pub giants: Vec<String>,

    // State
    pub status: AgentStatus,
    pub current_task: Option<Task>,

    // Voice (optional)
    pub voice_prompt: Option<String>,
}

impl Agent {
    pub async fn process(&mut self, task: Task) -> Result<Response> {
        self.status = AgentStatus::Thinking;

        // Build prompt with context
        let prompt = self.build_prompt(&task)?;

        // Call LLM
        let response = self.llm.complete(prompt).await?;

        // Validate through FATE
        let validated = fate::validate(&response)?;

        self.status = AgentStatus::Idle;
        Ok(validated)
    }
}
```

### A2A Communication

```
┌─────────────────────────────────────────────────────────────┐
│                    A2A MESSAGE FLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Agent A                                        Agent B    │
│   ┌──────┐                                      ┌──────┐   │
│   │      │──── TaskRequest ────────────────────▶│      │   │
│   │      │                                      │      │   │
│   │      │◀─── TaskResponse ───────────────────│      │   │
│   │      │                                      │      │   │
│   │      │──── DelegationRequest ─────────────▶│      │   │
│   │      │                                      │      │   │
│   │      │◀─── DelegationAccept ───────────────│      │   │
│   └──────┘                                      └──────┘   │
│                                                             │
│   Message Types:                                            │
│   - TaskRequest      : Assign task to agent                │
│   - TaskResponse     : Return task result                  │
│   - Delegation       : Hand off to another agent           │
│   - Collaboration    : Multi-agent session                 │
│   - GuardianReview   : Request ethics check                │
│   - GuardianVerdict  : Ethics decision                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Architecture

### LM Studio Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    LM STUDIO BRIDGE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   BIZRA CLI                          LM Studio              │
│   ┌──────────┐                      ┌──────────┐           │
│   │ Inference│   OpenAI-Compatible  │          │           │
│   │ Gateway  │──────── API ────────▶│ /v1/chat │           │
│   │          │                      │/completions           │
│   │          │◀───────────────────── │          │           │
│   └──────────┘                      └──────────┘           │
│                                                             │
│   Model Routing:                                            │
│   - Reasoning  → deepseek-r1-distill-qwen-32b              │
│   - Agentic    → qwen2.5-32b-instruct                      │
│   - Vision     → llava-v1.6-mistral-7b                     │
│   - Code       → qwen2.5-coder-32b                         │
│   - Embedding  → nomic-embed-text                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### MCP Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   MCP Client                         │   │
│   │   - Server discovery                                 │   │
│   │   - Connection management                            │   │
│   │   - Tool invocation                                  │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                 │
│           ┌───────────────┼───────────────┐                │
│           │               │               │                │
│           ▼               ▼               ▼                │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│   │ Filesystem│   │  GitHub   │   │  Memory   │           │
│   │  Server   │   │  Server   │   │  Server   │           │
│   └───────────┘   └───────────┘   └───────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                    TRUST BOUNDARIES                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              TRUSTED (Local)                         │   │
│   │                                                      │   │
│   │   - User input                                       │   │
│   │   - Configuration files                              │   │
│   │   - Local memory                                     │   │
│   │   - FATE gates                                       │   │
│   │                                                      │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                     FATE VALIDATION                         │
│                           │                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SEMI-TRUSTED (Controlled)               │   │
│   │                                                      │   │
│   │   - LM Studio responses                              │   │
│   │   - MCP server responses                             │   │
│   │   - Agent outputs                                    │   │
│   │                                                      │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                   GUARDIAN REVIEW                           │
│                           │                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              UNTRUSTED (External)                    │   │
│   │                                                      │   │
│   │   - Web content                                      │   │
│   │   - Federation messages                              │   │
│   │   - User-provided URLs                               │   │
│   │                                                      │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Security

```
Input → Sanitize → Validate → Process → FATE → Guardian → Output
           ↓          ↓          ↓        ↓        ↓
        [Log]      [Log]      [Log]    [Log]    [Log]
                                                   ↓
                                          Audit Trail
```

---

## Directory Structure

```
bizra-cli/
├── src/
│   ├── main.rs           # Entry point
│   ├── lib.rs            # Library exports
│   ├── app.rs            # Application state
│   ├── config.rs         # Configuration loading
│   ├── theme.rs          # Visual theme
│   ├── commands/         # Command handlers
│   │   └── mod.rs
│   └── widgets/          # TUI widgets
│       ├── mod.rs
│       ├── header.rs
│       ├── agent_card.rs
│       ├── fate_gauge.rs
│       └── status_bar.rs
│
├── config/               # Configuration files
│   ├── sovereign_profile.yaml
│   ├── mcp_servers.yaml
│   ├── a2a_protocol.yaml
│   ├── slash_commands.yaml
│   ├── hooks.yaml
│   ├── prompt_library.yaml
│   ├── skills.yaml
│   └── proactive.yaml
│
├── docs/                 # Documentation
│   ├── INDEX.md
│   ├── guides/
│   ├── reference/
│   ├── architecture/
│   └── tutorials/
│
└── Cargo.toml            # Rust dependencies
```

---

## Next Steps

- [Data Flow Details](DATA-FLOW.md)
- [Agent Architecture](AGENT-ARCHITECTURE.md)
- [Security Model](SECURITY-MODEL.md)
- [Integration Points](INTEGRATION-POINTS.md)

---

**Architecture is destiny.** 🏛️
