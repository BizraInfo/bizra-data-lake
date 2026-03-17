---
name: default-skill-registry
description: >
  Constitutional default skills that ship with every BIZRA node.
  These are NOT optional plugins. They are the minimum capability set
  that makes a node sovereign from minute one.
metadata:
  author: m.beshr
  version: '1.0'
  authority: Enforceable Spine v1.1 → this document → skill definitions
---

# BIZRA Default Skill Registry

Every node ships with these two capabilities pre-installed. No setup required.
`NodeCapabilityConfig::genesis()` activates both on first boot.

## Registered Skills

### 1. Smart File Management
- **Skill definition**: `docs/skills/file-management/SKILL.md`
- **Rust skill tree**: `bizra-agent/src/skills/skill_tree.rs → filesystem_skill_tree()`
- **Rust execution**: `bizra-agent/src/skills/file_management.rs`
- **MCP server**: `bizra-fs`
- **Boot state**: `fs_classify` (Novice), `fs_rename` (Novice), rest Locked
- **PAT agents**: Navigator (routing), Artisan (execution), Scholar (search), Sentinel (security)
- **SAT agents**: Guardian (manifest validation), Auditor (integrity verification)

### 2. Smart Browser Management
- **Skill definition**: `docs/skills/browser-management/SKILL.md`
- **Rust skill tree**: `bizra-agent/src/skills/skill_tree.rs → browser_skill_tree()`
- **Rust execution**: `bizra-agent/src/skills/browser_management.rs`
- **MCP server**: `bizra-browser`
- **Boot state**: `br_navigate` (Novice), `br_read` (Novice), rest Locked
- **PAT agents**: Navigator (URL routing), Scholar (page reading), Artisan (form filling)
- **SAT agents**: Guardian (URL validation), Sentinel (security scanning)

## How Skills Self-Configure

```
Node boots
  → NodeCapabilityConfig::genesis() loads both skill trees
  → Base skills start at Novice (executable immediately)
  → Agent uses skill successfully 3 times → promoted to Competent
  → Competent skill cascades → child skills unlock to Novice
  → Agent uses skill 10 times → promoted to Expert (reflex compiled)
  → Expert skill runs on System-1 fast path (50ms instead of 1800ms)
  → Agent uses skill 50 times → promoted to Master (can delegate to sub-agents)
```

This is the myelination pattern applied to capability acquisition. The organism learns
by doing. No configuration needed. No manual skill assignment. The tree self-configures.

## Constitutional Invariants

1. No skill activates without its prerequisite chain satisfied
2. No destructive skill activates without SAT approval in the tree config
3. Every skill execution produces a receipt (BLAKE3-chained)
4. Mastery is earned, not assigned — only successful governed executions count
5. Sub-agent delegation only available at Master level (50+ successes)
6. Guardian validates all skills — it is the universal SAT gate
7. Skill tree state persists across restarts (via ReflexStore)

## Adding New Skills (future)

New skills follow the same pattern:
1. Create `docs/skills/<name>/SKILL.md` with operational definition
2. Add skill tree function in `skill_tree.rs` with prerequisite chain
3. Add execution module in `bizra-agent/src/skills/<name>.rs`
4. Register in `default_capabilities.rs` (if constitutional default)
5. Wire MCP server ID for protocol integration

The skill tree is extensible. The constitutional gates are not.
