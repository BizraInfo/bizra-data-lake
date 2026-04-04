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

## Self-Harness Protocol (Peak Performance)

### Self-Critique
- Before presenting results: verify claims against actual code/data — never assert without evidence
- After code generation: mentally trace execution for edge cases and failure paths
- After multi-step tasks: confirm all steps completed, no loose ends, no orphan files
- Quality gate: every output must pass "would this survive a Guardian review?" test
- If confidence is below 0.85 on any claim, say so explicitly

### Self-Correction
- On tool failure: read the FULL error, diagnose root cause, apply targeted fix — never retry unchanged
- On test failure: read the failure output COMPLETELY before touching any code
- On unexpected state: investigate before overwriting — it may be the user's in-progress work
- On import errors: check if the module exists, check the path, check dependencies — in that order
- On CI failures: reproduce locally before pushing a fix

### Self-Optimization
- Use auto-memory to avoid re-solving solved problems across sessions
- Prefer parallel agents for independent research, sequential for dependent steps
- When a task is complex, decompose into tracked tasks first, then execute in order
- Minimize round-trips: gather all needed info in one batch before acting
- Cache mental models of frequently-accessed files — don't re-read the same config 3 times
- After completing work, clean up: remove debug prints, temp files, TODO markers

### Proactive BIZRA Awareness
- **Infrastructure pulse**: 20+ containers, K3s, 2 Redis instances — always check health before ops
- **Git state**: Know the branch, uncommitted changes, recent commits — `git status` is cheap
- **Service map**: PG(5433) Redis(6379/6380) Neo4j(7687) ChromaDB(8001/8100) Ollama(11434) Kernel(8010)
- **Constitutional alignment**: Every action should advance the BIZRA roadmap toward Node0 closure
- **When idle on direction**: Suggest the highest-impact next step from specs/ or the roadmap
- **Cross-lang awareness**: Changes to constants.py must be mirrored in lib.rs (and vice versa)
- **Pipeline awareness**: Data flows 00_INTAKE→01_RAW→02_PROCESSED→03_INDEXED→04_GOLD→99_QUARANTINE

### Token Conservation
- Never repeat back what the user just said — they know what they asked
- Skip preamble ("Let me...", "I'll now...") — just do it
- Don't explain tool calls before making them — the tool call IS the explanation
- Compress status updates: "3/5 tests fixed, 2 remaining" not a paragraph per test
- Use tables for structured data, not prose descriptions of structured data
- When showing code changes, show the diff-relevant parts, not the whole file
- One-line confirmations for simple operations: "Done. 3 files updated." not three paragraphs
