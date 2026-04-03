---
name: _documentation
description: This is documentation only, not an agent. Never invoke this.
tools: Read
model: haiku
---

# BIZRA Custom Subagents

Custom subagents for BIZRA development, aligned with the PAT-SAT dual-agentic architecture.

## Overview

Subagents are specialized AI assistants that can be invoked for specific tasks. They have predefined tools, models, and system prompts optimized for their role.

## Subagent Categories

### PAT-Style Execution Agents

Specialized agents for task execution, mirroring BIZRA's 7 PAT agents:

| Agent | Model | Purpose |
|-------|-------|---------|
| [master-reasoner](master-reasoner.md) | opus | Strategic thinking, multi-step planning, architecture decisions |
| [rust-expert](rust-expert.md) | sonnet | Rust development, code review, async patterns |
| [python-expert](python-expert.md) | sonnet | Python development, FastAPI, async patterns |
| [code-architect](code-architect.md) | opus | Software architecture, design patterns, integration |

### SAT-Style Validation Agents

Guardian agents for validation, mirroring BIZRA's 5 SAT guardians:

| Agent | Model | Purpose |
|-------|-------|---------|
| [ihsan-validator](ihsan-validator.md) | sonnet | Ihsān constitution validation, ethical compliance |
| [sape-analyzer](sape-analyzer.md) | sonnet | SAPE probe analysis, pattern elevation review |
| [receipt-auditor](receipt-auditor.md) | sonnet | Receipt validation, evidence integrity |
| [security-guardian](security-guardian.md) | sonnet | Security audits, vulnerability scanning |

### Utility Agents

Supporting agents for common development tasks:

| Agent | Model | Purpose |
|-------|-------|---------|
| [evidence-tracker](evidence-tracker.md) | haiku | Evidence chain tracing, audit trails |
| [debugger](debugger.md) | sonnet | Error diagnosis, log analysis, troubleshooting |

## Usage

### Invoking a Subagent

Subagents are automatically available when their description matches your task. Claude Code will proactively use them based on context.

**Examples:**

```
User: Review the Rust code in src/sape.rs for safety issues
→ Claude invokes rust-expert subagent

User: Validate the Ihsān constitution weights
→ Claude invokes ihsan-validator subagent

User: Why is SAT consensus failing?
→ Claude invokes debugger subagent
```

### Manual Invocation

You can explicitly request a subagent:

```
User: Use the security-guardian to audit the codebase
User: Have the code-architect review this integration
User: Ask the evidence-tracker to trace receipt chain
```

## Subagent Format

Each subagent is a Markdown file with YAML frontmatter:

```markdown
---
name: agent-name
description: When to use this agent proactively
tools: Read, Edit, Write, Grep, Glob, Bash
model: sonnet|opus|haiku
---

System prompt content...
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| name | Yes | Unique identifier |
| description | Yes | When Claude should use this agent |
| tools | Yes | Comma-separated list of allowed tools |
| model | No | Model to use (default: sonnet) |

### Available Tools

- `Read` - Read files
- `Edit` - Edit files
- `Write` - Write new files
- `Grep` - Search content
- `Glob` - Find files by pattern
- `Bash` - Execute commands

### Model Selection

| Model | Use For |
|-------|---------|
| opus | Complex reasoning, architecture, strategic planning |
| sonnet | Code implementation, analysis, detailed work |
| haiku | Quick lookups, simple tasks, cost-sensitive operations |

## Creating Custom Subagents

1. Create a new `.md` file in `.claude/agents/`
2. Add YAML frontmatter with required fields
3. Write a system prompt with:
   - Role description
   - BIZRA context
   - Task-specific instructions
   - Output format
   - Commands/examples

### Template

```markdown
---
name: my-agent
description: Short description of when to use this agent proactively
tools: Read, Grep, Glob, Bash
model: sonnet
---

You are a [Role Name], a [PAT/SAT/utility] agent for BIZRA.

## Your Role

You excel at:
- Capability 1
- Capability 2
- Capability 3

## BIZRA Context

[Relevant BIZRA architecture details]

## When Invoked

### For Task Type 1
1. Step 1
2. Step 2
3. Step 3

### For Task Type 2
1. Step 1
2. Step 2
3. Step 3

## Commands

```bash
# Useful command 1
command

# Useful command 2
command
```

## Output Format

Structure your output as:

### Section 1
[Content]

### Section 2
[Content]

## Key Files

- `path/to/file.rs` - Description
- `path/to/file.py` - Description
```

## BIZRA Integration

### PAT-SAT Alignment

Subagents mirror the BIZRA dual-agentic architecture:

- **PAT (execution)** → rust-expert, python-expert, code-architect, master-reasoner
- **SAT (validation)** → ihsan-validator, sape-analyzer, receipt-auditor, security-guardian

### Fail-Closed Principle

All validation agents enforce fail-closed:

```
if violation_detected:
    block_execution()
    report_violation()
    recommend_fix()
```

### Receipt-First Development

Validation agents check for receipt emission:

- `receipt-auditor` - Validates receipt integrity
- `evidence-tracker` - Traces receipt chains
- `ihsan-validator` - Ensures rejection receipts emitted

### Ihsān Gate Awareness

All agents understand:
- 8 ethical dimensions
- 0.99 threshold requirement
- Fail-closed on gate failure

## Best Practices

1. **Use proactively**: Agents auto-invoke based on description
2. **One agent per task**: Avoid overlapping responsibilities
3. **Read first**: Agents read relevant files before acting
4. **Evidence-driven**: Check for receipts and evidence
5. **Fail-closed**: Block on critical violations

## Troubleshooting

### Agent Not Invoked

Check if task matches agent description:
- Description too specific?
- Description too vague?
- Conflicting with another agent?

### Wrong Agent Invoked

Refine descriptions to be more specific about:
- Task types handled
- Keywords that trigger
- Exclusions (what NOT to handle)

### Agent Missing Tools

Update frontmatter to include needed tools:
```yaml
tools: Read, Edit, Write, Grep, Glob, Bash
```

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Main system guide
- [.claude/rules/](../rules/) - Memory rules
- [.claude/hooks/](../hooks/) - Hook system
- [.claude/commands/](../commands/) - Slash commands
