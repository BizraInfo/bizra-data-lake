# BIZRA Sovereign Engine — Local Configuration

## Activated Features

### Self-Continuation (Stop Hook)
The Stop hook uses a prompt-based check to determine if tasks are complete.
If incomplete, it will continue working automatically.

### Subagents (Model Calling Itself)
Available subagents for delegation:
- `/agents sovereign-researcher` — Deep research
- `/agents sovereign-coder` — Code generation
- `/agents sovereign-reviewer` — Code review
- `/agents sovereign-planner` — Strategic planning

### Memory System
- Session memory persists via claude-flow daemon
- Failures logged to memory namespace
- Context preserved across compaction

### Hook Events (All 12 Active)
1. SessionStart — Initialize daemon, set env vars
2. UserPromptSubmit — Route prompts, log to memory
3. PreToolUse — Validate operations, security checks
4. PostToolUse — Log results, format output
5. PostToolUseFailure — Log failures for analysis
6. SubagentStart — Track spawned agents
7. SubagentStop — Log agent completion
8. Stop — Self-continuation check (prompt-based)
9. PreCompact — Save context before compaction
10. Notification — Store notifications

### Skills/Slash Commands
- `/sovereign-query` — Full Sovereign Engine query
- `/deep-research` — Multi-source research
- `/implement` — Plan + code + review
- `/guardian-review` — Multi-guardian validation
- `/snr-check` — Signal-to-noise analysis

### MCP Servers
- filesystem — File system access
- memory — Persistent memory
- github — GitHub integration
- fetch — HTTP fetching
- brave-search — Web search
- sqlite — Database queries
- sequential-thinking — Enhanced reasoning

## Quality Standards
- SNR Threshold: ≥ 0.95 (Ihsān constraint)
- All code passes Guardian review
- Every claim has provenance
