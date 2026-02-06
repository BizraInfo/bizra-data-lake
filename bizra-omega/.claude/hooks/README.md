# BIZRA Omega Hooks

Automated workflow hooks for Claude Code integration.

## Available Hooks

### pre-rust-edit.sh
Validates Rust files before modification.
- Warns when editing core security modules (identity.rs, crypto.rs, envelope.rs)
- Returns `{"continue": true/false, "decision": "allow/warn", "reason": "..."}`

```bash
./pre-rust-edit.sh "path/to/file.rs"
```

### post-rust-edit.sh
Triggers after Rust file modifications.
- Logs modifications to `/tmp/bizra-hook.log`
- Identifies affected crate
- Suggests cargo check

```bash
./post-rust-edit.sh "path/to/file.rs"
```

### snr-check.sh
Validates content quality against SNR threshold.
- Calculates word diversity
- Returns pass/fail with SNR score

```bash
./snr-check.sh "content text" 0.85
```

### cargo-watch.sh
Triggers incremental Rust compilation.
- Supports: check, build, test

```bash
./cargo-watch.sh bizra-core check
./cargo-watch.sh bizra-tests test
```

### session-end.sh
Generates session summary and persists memory.
- Creates timestamped summary in `.claude-flow/memory/`
- Clears hook log

```bash
./session-end.sh
```

## Hook Response Format

All hooks return JSON:

```json
{
  "continue": true,      // Whether to proceed
  "decision": "allow",   // allow, warn, block
  "reason": "...",       // Human-readable explanation
  "metadata": {}         // Additional context
}
```

## Configuration

Hooks are configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [...],
    "PostToolUse": [...]
  }
}
```

## Quality Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| SNR | ≥ 0.85 | Signal-to-Noise Ratio |
| Ihsān | ≥ 0.95 | Excellence constraint |

## Protected Files

The following patterns are protected:
- `*.env*` — Environment files
- `*secret*` — Secret files
- `*.key`, `*.pem` — Key files

## Logging

Hook events are logged to `/tmp/bizra-hook.log`
