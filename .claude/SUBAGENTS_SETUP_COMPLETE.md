# Custom Subagents - Setup Complete

## Installation Summary

BIZRA's custom subagent system has been configured with specialized agents aligned to the PAT-SAT dual-agentic architecture.

## Installed Subagents

### PAT-Style Execution Agents (4)

| Agent | Model | Description |
|-------|-------|-------------|
| master-reasoner | opus | Strategic thinking, multi-step planning, architecture decisions |
| rust-expert | sonnet | Rust development, code review, async patterns, optimization |
| python-expert | sonnet | Python kernel development, FastAPI, async patterns |
| code-architect | opus | Software architecture, design patterns, cross-component integration |

### SAT-Style Validation Agents (4)

| Agent | Model | Description |
|-------|-------|-------------|
| ihsan-validator | sonnet | Ihsān constitution validation, 0.99 threshold compliance |
| sape-analyzer | sonnet | SAPE 9-probe analysis, pattern elevation review |
| receipt-auditor | sonnet | Receipt evidence validation, integrity hash verification |
| security-guardian | sonnet | Security audits, vulnerability scanning, TLS validation |

### Utility Agents (2)

| Agent | Model | Description |
|-------|-------|-------------|
| evidence-tracker | haiku | Evidence chain tracing, audit trail building |
| debugger | sonnet | Error diagnosis, log analysis, troubleshooting |

## Directory Structure

```
.claude/agents/
├── README.md              # Subagents documentation
├── master-reasoner.md     # Strategic thinking (opus)
├── rust-expert.md         # Rust development (sonnet)
├── python-expert.md       # Python development (sonnet)
├── code-architect.md      # Architecture design (opus)
├── ihsan-validator.md     # Ihsān validation (sonnet)
├── sape-analyzer.md       # SAPE analysis (sonnet)
├── receipt-auditor.md     # Receipt auditing (sonnet)
├── security-guardian.md   # Security auditing (sonnet)
├── evidence-tracker.md    # Evidence tracking (haiku)
└── debugger.md            # Error diagnosis (sonnet)
```

## Usage Examples

### Automatic Invocation

Subagents are invoked automatically based on task context:

```
"Review the Rust code in src/sape.rs"
→ Invokes rust-expert

"Validate the Ihsān constitution"
→ Invokes ihsan-validator

"Why is this request failing?"
→ Invokes debugger
```

### Explicit Invocation

Request a specific subagent:

```
"Use the security-guardian to audit for vulnerabilities"
"Have the code-architect review this design"
"Ask the evidence-tracker to trace receipts"
```

## Subagent Capabilities

### master-reasoner (opus)
- Strategic thinking and planning
- Breaking complex problems into steps
- Architecture decisions
- Risk assessment

### rust-expert (sonnet)
- Idiomatic Rust code
- Async patterns with Tokio
- Error handling with Result
- Performance optimization

### python-expert (sonnet)
- Clean, typed Python
- FastAPI async patterns
- Agent factory patterns
- Trinity Synapse integration

### code-architect (opus)
- Cross-component design
- Interface contracts
- BIZRA alignment review
- Implementation phases

### ihsan-validator (sonnet)
- Constitution validation
- Weight sum verification (1.0)
- Threshold compliance (0.99)
- Implementation consistency

### sape-analyzer (sonnet)
- 9-probe analysis
- Pattern elevation review
- Performance metrics (<100ms)
- Evidence collection

### receipt-auditor (sonnet)
- Schema validation
- Integrity hash verification
- Evidence chain completeness
- Append-only policy check

### security-guardian (sonnet)
- Secrets detection
- TLS validation
- Vulnerability scanning
- Blocklist/allowlist audit

### evidence-tracker (haiku)
- Receipt chain tracing
- Cross-system correlation
- Gap analysis
- Audit report generation

### debugger (sonnet)
- Error diagnosis
- Log analysis
- Root cause identification
- Fix proposals

## BIZRA Alignment

### PAT-SAT Architecture

Subagents mirror the dual-agentic system:
- PAT agents → rust-expert, python-expert, code-architect, master-reasoner
- SAT agents → ihsan-validator, sape-analyzer, receipt-auditor, security-guardian

### Fail-Closed Enforcement

Validation agents block on critical issues:
- Ihsān threshold violations
- SAPE probe failures
- Receipt integrity failures
- Security vulnerabilities

### Receipt-First Development

All agents understand:
- Receipt schema requirements
- Evidence chain importance
- Audit trail completeness

## Creating New Subagents

1. Create file in `.claude/agents/name.md`
2. Add YAML frontmatter:
   ```yaml
   ---
   name: agent-name
   description: When to use proactively
   tools: Read, Edit, Write, Grep, Glob, Bash
   model: sonnet
   ---
   ```
3. Write system prompt with:
   - Role description
   - BIZRA context
   - Task instructions
   - Output format

## See Also

- `.claude/agents/README.md` - Full documentation
- `CLAUDE.md` - Main system guide
- `.claude/rules/` - Memory rules
- `.claude/hooks/` - Hook system

---

**Subagent system setup complete!**

10 specialized agents ready for BIZRA development.
