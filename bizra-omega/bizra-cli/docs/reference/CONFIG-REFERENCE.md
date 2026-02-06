# Configuration Reference

Complete reference for all BIZRA configuration options.

## Table of Contents

1. [Configuration Files](#configuration-files)
2. [Sovereign Profile](#sovereign-profile)
3. [FATE Thresholds](#fate-thresholds)
4. [PAT Agents](#pat-agents)
5. [MCP Servers](#mcp-servers)
6. [A2A Protocol](#a2a-protocol)
7. [Slash Commands](#slash-commands)
8. [Hooks](#hooks)
9. [Skills](#skills)
10. [Proactive Settings](#proactive-settings)
11. [Voice Settings](#voice-settings)
12. [Memory Settings](#memory-settings)
13. [Integration Settings](#integration-settings)

---

## Configuration Files

### Directory Structure

```
~/.bizra/
├── config/
│   ├── sovereign_profile.yaml    # User identity & preferences
│   ├── mcp_servers.yaml          # MCP server definitions
│   ├── a2a_protocol.yaml         # Agent communication
│   ├── slash_commands.yaml       # Command definitions
│   ├── hooks.yaml                # Event automation
│   ├── skills.yaml               # Skill workflows
│   ├── proactive.yaml            # Anticipation settings
│   └── prompt_library.yaml       # Prompt templates
├── memory/
│   ├── decisions/                # Decision history
│   ├── learnings/                # Extracted learnings
│   ├── patterns/                 # Behavioral patterns
│   └── index.json                # Memory index
├── cache/
│   └── ...                       # Cached data
└── logs/
    └── ...                       # Log files
```

### Loading Order

1. System defaults
2. `~/.bizra/config/*.yaml` (user config)
3. Project `.bizra/config/*.yaml` (project overrides)
4. Environment variables
5. Command-line arguments

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BIZRA_HOME` | Configuration directory | `~/.bizra` |
| `BIZRA_CONFIG` | Main config file | `$BIZRA_HOME/config` |
| `BIZRA_LOG_LEVEL` | Logging level | `info` |
| `BIZRA_LLM_BACKEND` | LLM backend URL | `http://192.168.56.1:1234` |
| `BIZRA_FATE_STRICT` | Strict FATE mode | `false` |

---

## Sovereign Profile

`config/sovereign_profile.yaml`

### Identity

```yaml
identity:
  # Basic information
  name: "Your Name"                    # Display name
  title: "Your Title"                  # Professional title
  location: "City, Country"            # Location
  timezone: "GMT+4"                    # Timezone

  # Languages
  languages:
    - code: "en"                       # ISO 639-1 code
      name: "English"
      proficiency: "native"            # native | fluent | conversational | basic
    - code: "ar"
      name: "Arabic"
      proficiency: "fluent"

  # Genesis (node identity)
  genesis:
    hash: "abc123..."                  # Unique node hash
    timestamp: "2024-01-01T00:00:00Z"  # Genesis timestamp
    node_id: "node0_xxx"               # Node identifier
```

### Values

```yaml
values:
  # FATE gate thresholds
  fate_gates:
    ihsan_threshold: 0.95              # Excellence minimum (0.0-1.0)
    adl_gini_max: 0.35                 # Justice maximum (0.0-1.0)
    harm_threshold: 0.30               # Harm maximum (0.0-1.0)
    confidence_min: 0.80               # Confidence minimum (0.0-1.0)

  # Guiding principles
  principles:
    - "Excellence (Ihsān) is not optional — it's the minimum"
    - "Every human is a node. Every node is a seed."
    - "Stand on the shoulders of giants"

  # Inspirational figures
  giants:
    philosophy: ["Al-Ghazali", "Rawls"]
    technology: ["Shannon", "Turing", "Lamport"]
    strategy: ["Sun Tzu", "Porter"]
```

### Patterns

```yaml
patterns:
  # Schedule
  schedule:
    deep_work_hours:                   # Focus time blocks
      - "09:00-12:00"
      - "15:00-18:00"
    review_time: "20:00"               # Daily review time
    planning_time: "08:00"             # Daily planning time
    offline_hours:                     # No notifications
      - "00:00-06:00"

  # Communication style
  communication:
    preferred_style: "concise"         # concise | detailed | visual
    response_format: "structured"      # structured | narrative | bullet
    language_style: "technical"        # technical | casual | formal
    include_arabic: true               # Mix Arabic phrases

  # Decision making
  decision_style:
    risk_tolerance: "calculated"       # conservative | calculated | aggressive
    speed_vs_quality: "quality"        # speed | balanced | quality
    autonomy_level: "high"             # low | medium | high
    requires_confirmation:             # Actions needing confirmation
      - "financial_transactions"
      - "public_communications"
      - "irreversible_actions"
```

### Goals

```yaml
goals:
  # Vision statement
  vision: "Your long-term vision"

  # Current quarter OKRs
  current_quarter:
    - objective: "Objective 1"
      key_results:
        - "Key result 1"
        - "Key result 2"
    - objective: "Objective 2"
      key_results:
        - "Key result 1"

  # Milestones
  milestones:
    - date: "2026-Q1"
      target: "Milestone description"
    - date: "2026-Q2"
      target: "Milestone description"
```

---

## FATE Thresholds

### Default Thresholds

```yaml
fate_gates:
  # Excellence gate (higher is better)
  ihsan:
    threshold: 0.95                    # Minimum score
    weights:
      accuracy: 0.30                   # Factual correctness
      completeness: 0.25               # Addresses all aspects
      coherence: 0.20                  # Logical flow
      relevance: 0.15                  # On-topic
      clarity: 0.10                    # Clear expression

  # Justice gate (lower is better - Gini coefficient)
  adl:
    threshold: 0.35                    # Maximum Gini
    applies_to:
      - workload_distribution
      - resource_allocation
      - information_access

  # Harm prevention (lower is better)
  harm:
    threshold: 0.30                    # Maximum harm score
    categories:
      - physical_harm
      - psychological_harm
      - financial_harm
      - privacy_violation
      - security_risk
    action_on_fail: "block"            # block | warn | log

  # Confidence gate (higher is better)
  confidence:
    threshold: 0.80                    # Minimum confidence
    weights:
      model_confidence: 0.40
      source_reliability: 0.30
      consistency: 0.20
      verification: 0.10
    action_on_fail: "disclaimer"       # disclaimer | request_research
```

### Threshold Profiles

```yaml
profiles:
  conservative:
    ihsan_threshold: 0.98
    adl_gini_max: 0.25
    harm_threshold: 0.20
    confidence_min: 0.90

  balanced:
    ihsan_threshold: 0.95
    adl_gini_max: 0.35
    harm_threshold: 0.30
    confidence_min: 0.80

  permissive:
    ihsan_threshold: 0.90
    adl_gini_max: 0.45
    harm_threshold: 0.40
    confidence_min: 0.70

# Active profile
active_profile: "balanced"
```

---

## PAT Agents

`config/sovereign_profile.yaml` (pat_team section)

```yaml
pat_team:
  # Default agent
  default_agent: "guardian"

  # Agent configurations
  agents:
    strategist:
      enabled: true
      voice: "strategist.pt"           # Voice model
      personality: "analytical, thoughtful"
      specialties:
        - "strategic planning"
        - "market analysis"
        - "risk assessment"
      auto_engage_on:                  # Automatic engagement triggers
        - "planning tasks"
        - "strategic questions"
      max_concurrent_tasks: 3
      timeout: "30m"

    researcher:
      enabled: true
      voice: "researcher.pt"
      personality: "curious, thorough"
      specialties:
        - "web research"
        - "fact verification"
        - "knowledge synthesis"
      auto_engage_on:
        - "research requests"
        - "fact checking"
      max_concurrent_tasks: 5
      timeout: "15m"

    developer:
      enabled: true
      voice: "developer.pt"
      personality: "pragmatic, precise"
      specialties:
        - "code generation"
        - "debugging"
        - "architecture"
      auto_engage_on:
        - "code tasks"
        - "technical questions"
      max_concurrent_tasks: 3
      timeout: "45m"

    analyst:
      enabled: true
      voice: "analyst.pt"
      personality: "data-driven, insightful"
      specialties:
        - "data analysis"
        - "pattern detection"
        - "visualization"
      auto_engage_on:
        - "data analysis"
        - "metrics questions"
      max_concurrent_tasks: 3
      timeout: "20m"

    reviewer:
      enabled: true
      voice: "reviewer.pt"
      personality: "meticulous, constructive"
      specialties:
        - "code review"
        - "quality assurance"
        - "security audit"
      auto_engage_on:
        - "review requests"
        - "quality checks"
      max_concurrent_tasks: 4
      timeout: "20m"

    executor:
      enabled: true
      voice: "executor.pt"
      personality: "efficient, reliable"
      specialties:
        - "task execution"
        - "deployment"
        - "automation"
      auto_engage_on:
        - "execution requests"
        - "deployment tasks"
      max_concurrent_tasks: 2
      timeout: "60m"
      requires_guardian: true          # Always needs Guardian approval

    guardian:
      enabled: true
      voice: "NATF3.pt"
      personality: "protective, ethical"
      specialties:
        - "ethics review"
        - "safety check"
        - "risk monitoring"
      auto_engage_on:
        - "ethical concerns"
        - "high-stakes decisions"
        - "unusual patterns"
      always_consulted: true           # Reviews all major decisions
      can_veto: true                   # Can block actions
      max_concurrent_tasks: unlimited
```

---

## MCP Servers

`config/mcp_servers.yaml`

```yaml
mcp_servers:
  # Server definitions
  servers:
    # Claude-Flow coordination
    claude-flow:
      command: "npx"
      args: ["-y", "@anthropic/claude-flow-mcp"]
      env:
        CLAUDE_FLOW_MODE: "coordinator"
      capabilities:
        - swarm_orchestration
        - task_coordination
        - memory_sharing
      auto_start: true
      restart_on_failure: true

    # Filesystem access
    filesystem:
      command: "npx"
      args: ["-y", "@anthropic/filesystem-mcp"]
      env:
        ALLOWED_PATHS: "/home,/tmp,~/.bizra"
      capabilities:
        - file_read
        - file_write
        - directory_list
      auto_start: true

    # GitHub integration
    github:
      command: "npx"
      args: ["-y", "@anthropic/github-mcp"]
      env:
        GITHUB_TOKEN: "${GITHUB_TOKEN}"
      capabilities:
        - repo_access
        - pr_management
        - issue_tracking
      auto_start: false

    # Memory/knowledge base
    memory:
      command: "npx"
      args: ["-y", "@anthropic/memory-mcp"]
      env:
        MEMORY_PATH: "~/.bizra/memory"
        VECTOR_MODEL: "all-MiniLM-L6-v2"
      capabilities:
        - semantic_search
        - memory_store
        - knowledge_retrieval
      auto_start: true

    # Web search
    brave-search:
      command: "npx"
      args: ["-y", "@anthropic/brave-search-mcp"]
      env:
        BRAVE_API_KEY: "${BRAVE_API_KEY}"
      capabilities:
        - web_search
      auto_start: false

    # BIZRA inference
    bizra-inference:
      command: "python"
      args: ["-m", "bizra.mcp.inference"]
      env:
        LLM_BACKEND: "${BIZRA_LLM_BACKEND}"
      capabilities:
        - local_inference
        - model_selection
        - batch_processing
      auto_start: true

    # BIZRA federation
    bizra-federation:
      command: "python"
      args: ["-m", "bizra.mcp.federation"]
      capabilities:
        - node_discovery
        - consensus
        - resource_sharing
      auto_start: false

  # Server groups
  groups:
    minimal:
      - claude-flow
      - filesystem
      - memory

    development:
      - claude-flow
      - filesystem
      - github
      - memory
      - bizra-inference

    production:
      - claude-flow
      - filesystem
      - github
      - memory
      - bizra-inference
      - bizra-federation

  # Active group
  active_group: "development"

  # Global settings
  settings:
    connection_timeout: "30s"
    max_retries: 3
    retry_delay: "5s"
    health_check_interval: "60s"
```

---

## Slash Commands

`config/slash_commands.yaml`

```yaml
commands:
  # Agent commands
  agent:
    aliases: ["/a"]
    subcommands:
      list:
        description: "List all agents"
        handler: "agent_list"
      switch:
        description: "Switch active agent"
        args: ["agent_name"]
        handler: "agent_switch"
      status:
        description: "Show agent status"
        args: ["?agent_name"]
        handler: "agent_status"

  # Task commands
  task:
    aliases: ["/t"]
    subcommands:
      add:
        description: "Create new task"
        args: ["title"]
        options:
          agent: {short: "-a", type: "string"}
          priority: {short: "-p", type: "enum", values: ["low", "normal", "high"]}
          description: {short: "-d", type: "string"}
        handler: "task_add"
      list:
        description: "List tasks"
        args: ["?filter"]
        options:
          agent: {type: "string"}
          status: {type: "enum", values: ["pending", "active", "completed"]}
        handler: "task_list"
      done:
        description: "Mark task complete"
        args: ["task_id"]
        handler: "task_done"

  # Research commands
  research:
    aliases: ["/r"]
    subcommands:
      deep:
        description: "Deep research"
        args: ["topic"]
        options:
          sources: {type: "array"}
          depth: {type: "enum", values: ["summary", "moderate", "thorough"]}
          format: {type: "enum", values: ["markdown", "json"]}
        handler: "research_deep"
        agent: "researcher"
      quick:
        description: "Quick lookup"
        args: ["query"]
        handler: "research_quick"
        agent: "researcher"

  # Code commands
  code:
    aliases: ["/c"]
    subcommands:
      implement:
        description: "Implement feature"
        args: ["description"]
        options:
          lang: {type: "string"}
          tests: {type: "boolean", default: true}
        handler: "code_implement"
        agent: "developer"
      review:
        description: "Review code"
        args: ["target"]
        options:
          depth: {type: "enum", values: ["quick", "standard", "thorough"]}
          focus: {type: "array"}
        handler: "code_review"
        agent: "reviewer"

  # Guardian commands
  guardian:
    aliases: ["/g"]
    subcommands:
      status:
        description: "FATE gates status"
        handler: "guardian_status"
      review:
        description: "Request ethics review"
        args: ["action"]
        handler: "guardian_review"
      alert:
        description: "View alerts"
        args: ["?severity"]
        handler: "guardian_alert"

# Global modifiers
modifiers:
  urgent:
    flag: "--urgent"
    effect: {priority: "high"}
  json:
    flag: "--json"
    effect: {output_format: "json"}
  brief:
    flag: "--brief"
    effect: {detail_level: "low"}
  detailed:
    flag: "--detailed"
    effect: {detail_level: "high"}
  dry_run:
    flag: "--dry-run"
    effect: {execute: false, preview: true}
  remember:
    flag: "--remember"
    effect: {store_result: true}

# Aliases
aliases:
  "/gc": "/commit"
  "/pr": "/quick pr"
  "/sum": "/quick summarize"
  "/tr": "/quick translate"
```

---

## Hooks

`config/hooks.yaml`

```yaml
hooks:
  # Session hooks
  session:
    on_start:
      actions:
        - type: "load_context"
          params:
            sources: ["recent_tasks", "pending_items"]
        - type: "greet"
          params:
            style: "morning_brief"

    on_end:
      actions:
        - type: "summarize_session"
          params:
            store: true
        - type: "save_context"

    on_idle:
      after: "5m"
      actions:
        - type: "suggest_tasks"
          params:
            max_suggestions: 3

  # Message hooks
  message:
    pre_send:
      actions:
        - type: "enrich_context"
          params:
            add_recent_files: true
            add_relevant_memory: true

    post_receive:
      actions:
        - type: "extract_tasks"
          params:
            auto_add: false
            suggest: true
        - type: "check_fate_gates"

  # Task hooks
  task:
    on_create:
      actions:
        - type: "auto_assign"
          params:
            rules:
              - pattern: "research|investigate"
                agent: "researcher"
              - pattern: "implement|build|code"
                agent: "developer"

    on_complete:
      actions:
        - type: "extract_learnings"
        - type: "suggest_next"

  # Code hooks
  code:
    on_commit:
      actions:
        - type: "review"
          params:
            depth: "quick"
        - type: "generate_message"

    on_pr:
      actions:
        - type: "full_review"
        - type: "guardian_check"

  # Schedule hooks
  schedule:
    morning_brief:
      cron: "0 8 * * *"
      actions:
        - type: "run_command"
          params:
            command: "/morning"

    weekly_review:
      cron: "0 9 * * MON"
      actions:
        - type: "run_command"
          params:
            command: "/weekly"

  # Guardian hooks
  guardian:
    always_review:
      patterns:
        - "delete.*data"
        - "production.*deploy"
        - "credentials|secrets"
      actions:
        - type: "pause"
          params:
            require: "guardian_approval"
```

---

## Skills

`config/skills.yaml`

```yaml
skills:
  # Core skills
  deep-research:
    description: "Deep multi-source research"
    agents: ["researcher"]
    tools: ["web_search", "academic_search", "memory"]
    params:
      topic: {required: true, type: "string"}
      depth: {default: "thorough", enum: ["summary", "moderate", "thorough"]}
    steps:
      - name: "Query Analysis"
        agent: "researcher"
        action: "analyze_query"
      - name: "Multi-Source Search"
        parallel: true
        steps:
          - action: "web_search"
          - action: "academic_search"
          - action: "memory_search"
      - name: "Synthesis"
        agent: "researcher"
        action: "synthesize"
      - name: "FATE Validation"
        agent: "guardian"
        action: "validate_fate_gates"

  implement:
    description: "Plan, implement, and review"
    agents: ["developer", "reviewer", "guardian"]
    params:
      description: {required: true}
      language: {default: "auto"}
      include_tests: {default: true}
    steps:
      - name: "Plan"
        agent: "developer"
        action: "create_plan"
      - name: "Implement"
        agent: "developer"
        action: "write_code"
      - name: "Test"
        condition: "{{include_tests}}"
        agent: "developer"
        action: "write_tests"
      - name: "Review"
        agent: "reviewer"
        action: "review_code"
      - name: "Guardian Check"
        agent: "guardian"
        action: "security_review"

  # Development skills
  commit:
    description: "Smart git commit"
    agents: ["developer", "guardian"]
    steps:
      - action: "git_diff"
      - action: "interactive_stage"
      - agent: "developer"
        action: "generate_commit_message"
      - agent: "guardian"
        action: "quick_review"
      - action: "confirm"
      - action: "git_commit"

# Skill compositions
compositions:
  full-feature:
    description: "Complete feature development"
    skills:
      - deep-research
      - implement
      - commit
```

---

## Proactive Settings

`config/proactive.yaml`

```yaml
proactive:
  # Mode
  mode: "balanced"                     # silent | minimal | balanced | active

  # Context awareness
  context:
    update_interval: "30s"
    sources:
      - time
      - tasks
      - goals
      - activity
      - calendar
    window:
      immediate: "5m"
      recent: "1h"
      session: "full"

  # Anticipation
  anticipation:
    enabled: true
    predictions:
      - needs
      - blockers
      - opportunities
      - transitions
    confidence_threshold: 0.6

  # Suggestions
  suggestions:
    enabled: true
    timing:
      idle_threshold: "5m"
      on_completion: true
      context_switch: true
    max_suggestions: 3
    persist_dismissed: true
    decay_time: "24h"

  # Briefings
  briefings:
    morning:
      enabled: true
      time: "08:00"
      includes:
        - greeting
        - overnight_alerts
        - today_calendar
        - priority_tasks
        - goal_progress

    evening:
      enabled: true
      time: "20:00"
      includes:
        - accomplishments
        - learnings
        - tomorrow_preview

    weekly:
      enabled: true
      day: "friday"
      time: "17:00"
      includes:
        - week_summary
        - goal_progress
        - next_week_plan

  # Pattern learning
  patterns:
    enabled: true
    categories:
      - temporal
      - behavioral
      - preferences
    min_occurrences: 3
    confidence_threshold: 0.7
    confirm_major_changes: true

  # Think tank
  think_tank:
    enabled: true
    participants:
      - strategist
      - researcher
      - developer
      - analyst
      - guardian
    require_unanimous: false
    time_limit: "30m"
```

---

## Voice Settings

```yaml
voice:
  # Enable/disable
  enabled: false

  # Model settings
  model:
    provider: "personaplex"            # personaplex | eleven_labs | local
    model_id: "nvidia/personaplex-7b-v1"

  # Agent voices
  agent_voices:
    guardian: "NATF3.pt"
    strategist: "strategist.pt"
    researcher: "researcher.pt"
    developer: "developer.pt"
    analyst: "analyst.pt"
    reviewer: "reviewer.pt"
    executor: "executor.pt"

  # Audio settings
  audio:
    sample_rate: 24000
    output_device: "default"
    input_device: "default"

  # Speech settings
  speech:
    speed: 1.0                         # 0.5 - 2.0
    pitch: 1.0                         # 0.5 - 2.0
    volume: 0.8                        # 0.0 - 1.0

  # Recognition settings
  recognition:
    language: "en-US"
    wake_word: "bizra"
    sensitivity: 0.5
```

---

## Memory Settings

```yaml
memory:
  # Session memory
  session:
    max_context_tokens: 100000
    summarize_after: 50000
    preserve_critical: true

  # Persistent memory
  persistent:
    store_path: "~/.bizra/memory"

    categories:
      decisions:
        retention: "permanent"
        index: true
      learnings:
        retention: "permanent"
        index: true
      preferences:
        retention: "permanent"
        index: true
      patterns:
        retention: "1 year"
        index: false
      contacts:
        retention: "permanent"
        index: true
      projects:
        retention: "permanent"
        index: true

    # Vector search
    vector:
      model: "all-MiniLM-L6-v2"
      dimensions: 384
      similarity: "cosine"

    # Cleanup
    cleanup:
      enabled: true
      schedule: "0 3 * * *"            # 3 AM daily
      archive_after: "1 year"
      delete_after: "3 years"
```

---

## Integration Settings

```yaml
integrations:
  # LLM backend
  llm:
    primary:
      url: "http://192.168.56.1:1234"
      type: "lmstudio"
      model: "auto"
    fallback:
      url: "http://localhost:11434"
      type: "ollama"
      model: "llama3"
    timeout: "60s"
    max_retries: 3

  # Calendar
  calendar:
    provider: "google"                 # google | outlook | caldav
    credentials_path: "~/.bizra/credentials/calendar.json"
    sync_interval: "15m"

  # GitHub
  github:
    token_env: "GITHUB_TOKEN"
    default_org: "your-org"

  # Notifications
  notifications:
    desktop:
      enabled: true
      sound: true
    email:
      enabled: false
      address: ""
    slack:
      enabled: false
      webhook: ""
```

---

## Applying Configuration

### Reload Configuration

```bash
# Reload all config
/reload config

# Reload specific file
/reload config mcp_servers

# Validate configuration
/config validate
```

### Configuration Commands

```bash
# Show current config
/config show

# Show specific section
/config show fate_gates

# Edit config
/config edit sovereign_profile

# Reset to defaults
/config reset --section hooks
```

---

**Configuration is power. Use it wisely.** ⚙️
