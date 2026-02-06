# Skills System Guide

Modular workflows that combine agents, tools, and actions into powerful capabilities.

## Table of Contents

1. [What Are Skills?](#what-are-skills)
2. [Core Skills](#core-skills)
3. [Development Skills](#development-skills)
4. [Collaboration Skills](#collaboration-skills)
5. [Analysis Skills](#analysis-skills)
6. [Workflow Skills](#workflow-skills)
7. [Creating Custom Skills](#creating-custom-skills)
8. [Skill Composition](#skill-composition)
9. [Skill Parameters](#skill-parameters)
10. [Best Practices](#best-practices)

---

## What Are Skills?

Skills are reusable, multi-step workflows that orchestrate agents, tools, and actions to accomplish complex tasks.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SKILL ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Request                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────┐                                                      │
│   │   Skill     │ ← Defines the workflow                               │
│   │  Registry   │                                                      │
│   └──────┬──────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────┐          │
│   │                    SKILL EXECUTION                       │          │
│   │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐         │          │
│   │  │Step 1 │ → │Step 2 │ → │Step 3 │ → │Step N │         │          │
│   │  │Agent A│   │Agent B│   │Tool X │   │Agent C│         │          │
│   │  └───────┘   └───────┘   └───────┘   └───────┘         │          │
│   │                     │                                    │          │
│   │                     ▼                                    │          │
│   │              FATE Gates Check                            │          │
│   └─────────────────────────────────────────────────────────┘          │
│          │                                                              │
│          ▼                                                              │
│   Final Output                                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Location

```
~/.bizra/config/skills.yaml
```

### Basic Skill Structure

```yaml
skills:
  skill_name:
    description: "What this skill does"
    agents: ["agent1", "agent2"]      # Agents involved
    tools: ["tool1", "tool2"]         # Tools required
    steps:
      - name: "Step 1"
        agent: "agent1"
        action: "action_type"
        params: {...}
      - name: "Step 2"
        agent: "agent2"
        action: "action_type"
        params: {...}
```

---

## Core Skills

Essential skills for everyday use.

### deep-research

Comprehensive multi-source research.

```yaml
deep-research:
  description: "Deep multi-source research synthesis"
  agents: ["researcher"]
  tools: ["web_search", "academic_search", "memory"]
  params:
    topic: {required: true, type: "string"}
    depth: {default: "thorough", enum: ["summary", "moderate", "thorough"]}
    sources: {default: "all", enum: ["web", "academic", "memory", "all"]}
  steps:
    - name: "Query Analysis"
      agent: "researcher"
      action: "analyze_query"
      params:
        extract: ["key_concepts", "scope", "constraints"]

    - name: "Memory Search"
      action: "search_memory"
      params:
        query: "{{topic}}"
        relevance_threshold: 0.7

    - name: "Web Research"
      condition: "{{sources}} in ['web', 'all']"
      action: "web_search"
      params:
        query: "{{topic}}"
        max_results: 10

    - name: "Academic Search"
      condition: "{{sources}} in ['academic', 'all']"
      action: "academic_search"
      params:
        query: "{{topic}}"
        max_results: 5

    - name: "Synthesis"
      agent: "researcher"
      action: "synthesize"
      params:
        sources: "{{all_results}}"
        style: "comprehensive"

    - name: "FATE Validation"
      agent: "guardian"
      action: "validate_fate_gates"

    - name: "Store Results"
      condition: "{{store_result}}"
      action: "remember"
      params:
        category: "research"
```

**Usage:**
```bash
/research deep "quantum computing applications"
/research deep "Rust async" --depth thorough --sources academic
```

### implement

Full implementation workflow with review.

```yaml
implement:
  description: "Plan, implement, and review code"
  agents: ["developer", "reviewer", "guardian"]
  tools: ["code_editor", "test_runner", "linter"]
  params:
    description: {required: true, type: "string"}
    language: {default: "auto", type: "string"}
    include_tests: {default: true, type: "boolean"}
  steps:
    - name: "Plan"
      agent: "developer"
      action: "create_plan"
      params:
        description: "{{description}}"
        output: "implementation_plan"

    - name: "Implement"
      agent: "developer"
      action: "write_code"
      params:
        plan: "{{implementation_plan}}"
        language: "{{language}}"

    - name: "Write Tests"
      condition: "{{include_tests}}"
      agent: "developer"
      action: "write_tests"
      params:
        code: "{{implementation}}"

    - name: "Lint"
      action: "lint"
      params:
        auto_fix: true

    - name: "Review"
      agent: "reviewer"
      action: "review_code"
      params:
        code: "{{implementation}}"
        focus: ["correctness", "style", "security"]

    - name: "Guardian Check"
      agent: "guardian"
      action: "security_review"
      params:
        code: "{{implementation}}"
```

**Usage:**
```bash
/code implement "rate limiting middleware"
/implement "user authentication" --language rust --include-tests
```

### guardian-review

Comprehensive quality and safety review.

```yaml
guardian-review:
  description: "Full Guardian quality review"
  agents: ["guardian", "reviewer"]
  tools: ["static_analyzer", "security_scanner"]
  params:
    target: {required: true, type: "string"}
    depth: {default: "standard", enum: ["quick", "standard", "thorough"]}
  steps:
    - name: "Static Analysis"
      action: "static_analyze"
      params:
        target: "{{target}}"

    - name: "Security Scan"
      action: "security_scan"
      params:
        target: "{{target}}"
        rules: "owasp_top_10"

    - name: "Code Review"
      agent: "reviewer"
      action: "review_code"
      params:
        focus: ["logic", "performance", "maintainability"]

    - name: "Ethics Review"
      agent: "guardian"
      action: "ethics_review"
      params:
        check:
          - "harmful_content"
          - "bias"
          - "privacy_violations"

    - name: "FATE Gates"
      agent: "guardian"
      action: "validate_fate_gates"
      params:
        strict: "{{depth == 'thorough'}}"

    - name: "Generate Report"
      agent: "guardian"
      action: "generate_report"
      params:
        include:
          - "findings"
          - "recommendations"
          - "risk_assessment"
```

**Usage:**
```bash
/guardian review src/
/guardian-review "deployment plan" --depth thorough
```

### sovereign-query

Query with full sovereignty context.

```yaml
sovereign-query:
  description: "Query with sovereign context"
  agents: ["guardian"]
  tools: ["memory", "inference"]
  params:
    query: {required: true, type: "string"}
  steps:
    - name: "Load Context"
      action: "load_sovereign_context"
      params:
        include:
          - "identity"
          - "values"
          - "goals"
          - "recent_decisions"

    - name: "Enrich Query"
      action: "enrich_query"
      params:
        query: "{{query}}"
        context: "{{sovereign_context}}"

    - name: "Execute Query"
      action: "inference"
      params:
        prompt: "{{enriched_query}}"
        model: "current"

    - name: "Validate Response"
      agent: "guardian"
      action: "validate_fate_gates"

    - name: "Learn from Interaction"
      action: "extract_learnings"
      params:
        store_if_significant: true
```

---

## Development Skills

Code-focused workflows.

### commit

Smart git commit workflow.

```yaml
commit:
  description: "Smart git commit with review"
  agents: ["developer", "guardian"]
  tools: ["git"]
  steps:
    - name: "Analyze Changes"
      action: "git_diff"
      params:
        staged_only: false

    - name: "Stage Files"
      action: "interactive_stage"
      params:
        show_preview: true

    - name: "Generate Message"
      agent: "developer"
      action: "generate_commit_message"
      params:
        style: "conventional"
        include_scope: true

    - name: "Quick Review"
      agent: "guardian"
      action: "quick_review"
      params:
        check: ["secrets", "sensitive_data"]

    - name: "Confirm"
      action: "confirm"
      params:
        show: ["message", "files"]

    - name: "Commit"
      action: "git_commit"
      params:
        message: "{{commit_message}}"
```

**Usage:**
```bash
/commit
/gc                    # Alias
```

### pr

Pull request workflow.

```yaml
pr:
  description: "Create pull request with full review"
  agents: ["developer", "reviewer", "guardian"]
  tools: ["git", "github"]
  params:
    draft: {default: false, type: "boolean"}
    reviewers: {default: [], type: "array"}
  steps:
    - name: "Analyze Branch"
      action: "git_log"
      params:
        since: "origin/main"

    - name: "Full Review"
      agent: "reviewer"
      action: "full_review"
      params:
        include: ["security", "performance", "tests"]

    - name: "Generate Description"
      agent: "developer"
      action: "generate_pr_description"
      params:
        include:
          - "summary"
          - "changes"
          - "testing"
          - "screenshots"

    - name: "Guardian Approval"
      agent: "guardian"
      action: "approve_pr"
      params:
        check: ["fate_gates", "security"]

    - name: "Create PR"
      action: "github_create_pr"
      params:
        title: "{{pr_title}}"
        body: "{{pr_description}}"
        draft: "{{draft}}"
        reviewers: "{{reviewers}}"
```

**Usage:**
```bash
/pr
/pr --draft
/pr --reviewers "alice,bob"
```

### review-code

Code review skill.

```yaml
review-code:
  description: "Thorough code review"
  agents: ["reviewer", "guardian"]
  tools: ["static_analyzer", "test_runner"]
  params:
    target: {required: true, type: "string"}
    focus: {default: "all", enum: ["security", "performance", "style", "logic", "all"]}
  steps:
    - name: "Read Code"
      action: "read_files"
      params:
        target: "{{target}}"

    - name: "Static Analysis"
      action: "lint"
      params:
        tool: "clippy"  # For Rust

    - name: "Run Tests"
      action: "run_tests"
      params:
        with_coverage: true

    - name: "Review"
      agent: "reviewer"
      action: "review"
      params:
        focus: "{{focus}}"
        style: "constructive"

    - name: "Security Check"
      condition: "{{focus}} in ['security', 'all']"
      agent: "guardian"
      action: "security_scan"

    - name: "Generate Report"
      action: "generate_review_report"
      params:
        format: "markdown"
```

**Usage:**
```bash
/code review src/main.rs
/review-code PR#123 --focus security
```

### debug

Debugging workflow.

```yaml
debug:
  description: "Systematic debugging"
  agents: ["developer"]
  tools: ["debugger", "profiler", "logger"]
  params:
    issue: {required: true, type: "string"}
    context: {default: "", type: "string"}
  steps:
    - name: "Understand Issue"
      agent: "developer"
      action: "analyze_issue"
      params:
        description: "{{issue}}"
        context: "{{context}}"

    - name: "Gather Evidence"
      action: "collect_logs"
      params:
        relevant_to: "{{issue}}"

    - name: "Hypothesize"
      agent: "developer"
      action: "generate_hypotheses"
      params:
        issue: "{{issue}}"
        evidence: "{{logs}}"

    - name: "Test Hypotheses"
      agent: "developer"
      action: "test_hypotheses"
      params:
        hypotheses: "{{hypotheses}}"

    - name: "Propose Fix"
      agent: "developer"
      action: "propose_fix"
      params:
        root_cause: "{{confirmed_cause}}"

    - name: "Review Fix"
      agent: "reviewer"
      action: "review_fix"
```

**Usage:**
```bash
/code debug "memory leak in worker pool"
/debug "authentication failing" --context "error.log"
```

---

## Collaboration Skills

Multi-agent collaborative workflows.

### swarm

Multi-agent swarm for complex tasks.

```yaml
swarm:
  description: "Multi-agent collaborative swarm"
  agents: ["strategist", "researcher", "developer", "analyst", "reviewer", "guardian"]
  params:
    task: {required: true, type: "string"}
    mode: {default: "collaborative", enum: ["collaborative", "competitive", "consensus"]}
  steps:
    - name: "Task Analysis"
      agent: "strategist"
      action: "analyze_task"
      params:
        task: "{{task}}"
        output: "task_breakdown"

    - name: "Research Phase"
      agent: "researcher"
      action: "gather_context"
      params:
        topics: "{{task_breakdown.research_needs}}"

    - name: "Parallel Work"
      parallel: true
      steps:
        - agent: "developer"
          action: "technical_analysis"
        - agent: "analyst"
          action: "data_analysis"

    - name: "Synthesis"
      agent: "strategist"
      action: "synthesize_results"
      params:
        inputs: "{{all_results}}"

    - name: "Review"
      agent: "reviewer"
      action: "review_synthesis"

    - name: "Guardian Approval"
      agent: "guardian"
      action: "final_approval"
```

**Usage:**
```bash
/swarm "Design new API architecture"
/swarm "Optimize database queries" --mode competitive
```

### consensus

Reach consensus among agents.

```yaml
consensus:
  description: "Multi-agent consensus building"
  agents: ["strategist", "developer", "reviewer", "guardian"]
  params:
    question: {required: true, type: "string"}
    require_unanimous: {default: false, type: "boolean"}
  steps:
    - name: "Present Question"
      action: "broadcast"
      params:
        message: "{{question}}"
        to: "all_agents"

    - name: "Gather Perspectives"
      parallel: true
      steps:
        - agent: "strategist"
          action: "provide_perspective"
        - agent: "developer"
          action: "provide_perspective"
        - agent: "reviewer"
          action: "provide_perspective"

    - name: "Analyze Perspectives"
      agent: "guardian"
      action: "analyze_consensus"
      params:
        perspectives: "{{all_perspectives}}"
        threshold: "{{require_unanimous ? 1.0 : 0.6}}"

    - name: "Resolve Conflicts"
      condition: "{{has_conflicts}}"
      agent: "guardian"
      action: "mediate"

    - name: "Final Decision"
      agent: "guardian"
      action: "render_decision"
      params:
        based_on: "{{analysis}}"
```

**Usage:**
```bash
/consensus "Should we use microservices?"
/consensus "Best approach for authentication" --require-unanimous
```

---

## Analysis Skills

Data and trend analysis.

### analyze-data

Comprehensive data analysis.

```yaml
analyze-data:
  description: "Comprehensive data analysis"
  agents: ["analyst"]
  tools: ["data_reader", "statistics", "visualizer"]
  params:
    source: {required: true, type: "string"}
    focus: {default: "insights", enum: ["insights", "anomalies", "trends", "all"]}
  steps:
    - name: "Load Data"
      action: "read_data"
      params:
        source: "{{source}}"

    - name: "Profile Data"
      agent: "analyst"
      action: "profile_data"
      params:
        include: ["shape", "types", "missing", "distributions"]

    - name: "Statistical Analysis"
      agent: "analyst"
      action: "statistical_analysis"
      params:
        tests: ["correlation", "distribution", "outliers"]

    - name: "Find Patterns"
      agent: "analyst"
      action: "pattern_detection"
      params:
        focus: "{{focus}}"

    - name: "Generate Insights"
      agent: "analyst"
      action: "generate_insights"
      params:
        findings: "{{all_findings}}"

    - name: "Visualize"
      action: "create_visualizations"
      params:
        type: "auto"
```

**Usage:**
```bash
/analyze data "04_GOLD/metrics.parquet"
/analyze-data "users.csv" --focus anomalies
```

---

## Workflow Skills

Daily workflow automation.

### morning-brief

Morning briefing skill.

```yaml
morning-brief:
  description: "Comprehensive morning briefing"
  agents: ["guardian", "strategist"]
  tools: ["calendar", "tasks", "memory"]
  steps:
    - name: "Gather Context"
      parallel: true
      steps:
        - action: "get_calendar"
          params:
            date: "today"
        - action: "get_pending_tasks"
        - action: "get_alerts"
        - action: "get_overnight_activity"

    - name: "Analyze Priorities"
      agent: "strategist"
      action: "prioritize_day"
      params:
        tasks: "{{pending_tasks}}"
        calendar: "{{calendar}}"
        goals: "{{current_goals}}"

    - name: "Check Health"
      agent: "guardian"
      action: "system_health_check"

    - name: "Generate Brief"
      action: "format_brief"
      params:
        style: "morning"
        include:
          - "greeting"
          - "priorities"
          - "calendar"
          - "alerts"
          - "suggestion"
```

**Usage:**
```bash
/morning
```

### weekly-review

Weekly review and planning.

```yaml
weekly-review:
  description: "Weekly review and planning"
  agents: ["strategist", "analyst", "guardian"]
  steps:
    - name: "Gather Week Data"
      parallel: true
      steps:
        - action: "get_completed_tasks"
          params:
            period: "week"
        - action: "get_metrics"
          params:
            period: "week"
        - action: "get_learnings"
          params:
            period: "week"

    - name: "Analyze Progress"
      agent: "analyst"
      action: "analyze_progress"
      params:
        against: "weekly_goals"

    - name: "Extract Insights"
      agent: "strategist"
      action: "extract_weekly_insights"

    - name: "Plan Next Week"
      agent: "strategist"
      action: "plan_week"
      params:
        goals: "{{quarterly_goals}}"
        capacity: "{{available_hours}}"

    - name: "Guardian Review"
      agent: "guardian"
      action: "review_plan"

    - name: "Generate Report"
      action: "format_report"
      params:
        style: "weekly_review"
```

**Usage:**
```bash
/weekly
```

---

## Creating Custom Skills

### Minimal Skill

```yaml
custom_skills:
  my-skill:
    description: "My custom skill"
    agents: ["developer"]
    steps:
      - name: "Do Something"
        agent: "developer"
        action: "custom_action"
```

### Skill with Parameters

```yaml
custom_skills:
  deploy-feature:
    description: "Deploy a feature to environment"
    params:
      feature: {required: true, type: "string"}
      environment: {default: "staging", enum: ["staging", "production"]}
      notify: {default: true, type: "boolean"}
    steps:
      - name: "Build"
        action: "build"
        params:
          target: "{{feature}}"

      - name: "Test"
        action: "run_tests"
        params:
          scope: "{{feature}}"

      - name: "Deploy"
        action: "deploy"
        params:
          to: "{{environment}}"
          feature: "{{feature}}"

      - name: "Notify"
        condition: "{{notify}}"
        action: "notify"
        params:
          message: "Feature {{feature}} deployed to {{environment}}"
```

### Skill with Conditions

```yaml
custom_skills:
  smart-review:
    description: "Review based on change type"
    params:
      target: {required: true}
    steps:
      - name: "Analyze Changes"
        action: "analyze_changes"
        params:
          target: "{{target}}"

      - name: "Security Review"
        condition: "{{changes_include('auth', 'security', 'crypto')}}"
        agent: "guardian"
        action: "security_review"
        params:
          depth: "thorough"

      - name: "Performance Review"
        condition: "{{changes_include('database', 'query', 'cache')}}"
        agent: "analyst"
        action: "performance_review"

      - name: "Standard Review"
        condition: "{{not special_review_needed}}"
        agent: "reviewer"
        action: "standard_review"
```

---

## Skill Composition

Combine skills into larger workflows.

### Sequential Composition

```yaml
compositions:
  full-feature:
    description: "Complete feature development"
    skills:
      - skill: "deep-research"
        params:
          topic: "{{requirements}}"
      - skill: "implement"
        params:
          description: "{{research_output}}"
      - skill: "review-code"
        params:
          target: "{{implementation}}"
      - skill: "pr"
```

### Conditional Composition

```yaml
compositions:
  smart-deploy:
    description: "Smart deployment workflow"
    steps:
      - skill: "review-code"
        params:
          target: "."

      - condition: "{{environment == 'production'}}"
        skill: "guardian-review"
        params:
          depth: "thorough"

      - skill: "deploy-feature"
        params:
          environment: "{{environment}}"
```

---

## Skill Parameters

### Parameter Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text value | `"hello"` |
| `boolean` | True/false | `true` |
| `number` | Numeric value | `42` |
| `array` | List of values | `["a", "b"]` |
| `enum` | One of options | `"option1"` |

### Parameter Attributes

```yaml
params:
  required_param:
    required: true           # Must be provided
    type: "string"
    description: "What this param does"

  optional_param:
    default: "value"         # Default if not provided
    type: "string"

  enum_param:
    default: "option1"
    enum: ["option1", "option2", "option3"]

  validated_param:
    type: "number"
    min: 0
    max: 100
```

### Using Parameters in Steps

```yaml
steps:
  - name: "Use Params"
    action: "do_something"
    params:
      direct: "{{my_param}}"                    # Direct use
      computed: "Result: {{my_param}}"          # In string
      conditional: "{{flag ? 'yes' : 'no'}}"    # Ternary
```

---

## Best Practices

### 1. Single Responsibility

```yaml
# Good: Focused skill
commit:
  description: "Smart git commit"
  # Does one thing well

# Avoid: Kitchen sink
everything:
  description: "Does everything"
  # Too broad, hard to maintain
```

### 2. Use Conditions

```yaml
# Good: Conditional steps
steps:
  - name: "Security Review"
    condition: "{{is_security_critical}}"
    agent: "guardian"
    action: "security_review"

# Avoid: Always running expensive steps
steps:
  - name: "Security Review"
    agent: "guardian"
    action: "thorough_security_review"  # Always runs
```

### 3. Include Guardian

```yaml
# Good: Guardian validates
steps:
  - name: "Generate Output"
    agent: "developer"
    action: "generate"
  - name: "Validate"
    agent: "guardian"
    action: "validate_fate_gates"

# Avoid: No validation
steps:
  - name: "Generate Output"
    agent: "developer"
    action: "generate"
  # No guardian check
```

### 4. Parallel When Possible

```yaml
# Good: Parallel independent steps
steps:
  - name: "Gather Context"
    parallel: true
    steps:
      - action: "get_calendar"
      - action: "get_tasks"
      - action: "get_alerts"

# Avoid: Sequential when not needed
steps:
  - action: "get_calendar"
  - action: "get_tasks"
  - action: "get_alerts"
```

### 5. Descriptive Names

```yaml
# Good: Clear names
steps:
  - name: "Analyze Security Implications"
  - name: "Generate Test Cases"
  - name: "Validate Against FATE Gates"

# Avoid: Vague names
steps:
  - name: "Step 1"
  - name: "Process"
  - name: "Check"
```

---

## Skill Management Commands

```bash
# List skills
/skill list

# Show skill details
/skill show deep-research

# Run skill
/skill run deep-research --topic "AI safety"

# Create custom skill
/skill create my-skill

# Test skill (dry run)
/skill test my-skill --dry-run
```

---

## Next Steps

- [Proactive System](11-PROACTIVE-SYSTEM.md) — Anticipation engine
- [A2A Protocol](../reference/A2A-PROTOCOL.md) — Agent communication
- [Config Reference](../reference/CONFIG-REFERENCE.md) — All configuration options

---

**Skills: Your workflow, modularized.** ⚡
