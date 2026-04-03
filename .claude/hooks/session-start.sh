#!/usr/bin/env bash
# BIZRA SessionStart Hook - Compact Health Summary
set -e

INPUT=$(cat)
SOURCE=$(echo "$INPUT" | jq -r '.source // "startup"')

# Build compact status line
build_status() {
  local docker_status="off"
  local healthy=0
  local total=0
  local receipts=0
  local last_commit=""

  # Docker services
  if command -v docker &>/dev/null && docker compose ps --format json 2>/dev/null | head -1 | grep -q '{'; then
    total=$(docker compose ps --format json 2>/dev/null | jq -s 'length' 2>/dev/null || echo 0)
    healthy=$(docker compose ps --format json 2>/dev/null | jq -s '[.[] | select(.State == "running")] | length' 2>/dev/null || echo 0)
    docker_status="${healthy}/${total}"
  fi

  # Receipts (24h)
  if [ -d "$CLAUDE_PROJECT_DIR/docs/evidence/receipts" ]; then
    receipts=$(find "$CLAUDE_PROJECT_DIR/docs/evidence/receipts" \( -name "*.jsonl" -o -name "*.json" \) -mtime -1 2>/dev/null | wc -l | tr -d ' ')
  fi

  # Last commit
  if command -v git &>/dev/null && [ -d "$CLAUDE_PROJECT_DIR/.git" ]; then
    last_commit=$(cd "$CLAUDE_PROJECT_DIR" && git log -1 --format="%h" 2>/dev/null || echo "none")
  fi

  echo "Services: $docker_status | Receipts(24h): $receipts | Commit: $last_commit"
}

# Output compact context
cat <<EOF
--- BIZRA System Context ($SOURCE) ---
Build Commands:
- Rust: cargo build --release && cargo test
- Python: pip install -r requirements-kernel.txt
- Docker: docker compose up -d
- Full stack: docker compose up -d (starts all 7 services)

Docker Services:
EOF

# List services with status
if command -v docker &>/dev/null; then
  docker compose ps --format "  - {{.Service}}: {{.State}}" 2>/dev/null || echo "  - Docker not available"
fi

echo ""
echo "$(build_status)"

# ─── Auto-Configuration & PAT Memory ───
autoconfig_mode="unknown"
active_models=""
pat_memory_status="not loaded"

# Run kernel autoconfig (non-blocking, 10s timeout for service probes)
if command -v python3 &>/dev/null; then
  autoconfig_output=$(cd "$CLAUDE_PROJECT_DIR" && timeout 10 python3 -c "
import asyncio, json, sys
try:
    from core.autoconfig import auto_configure
    result = asyncio.run(auto_configure())
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({'mode': 'degraded', 'error': str(e)}))
" 2>/dev/null || echo '{"mode":"unknown"}')

  autoconfig_mode=$(echo "$autoconfig_output" | jq -r '.mode // "unknown"' 2>/dev/null || echo "unknown")

  # Extract active models
  active_models=$(echo "$autoconfig_output" | jq -r '
    [.services.ollama.info.models // [] | .[]] | join(", ")
  ' 2>/dev/null || echo "")

  # Restore PAT memory
  pat_memory_status=$(cd "$CLAUDE_PROJECT_DIR" && timeout 5 python3 -c "
import asyncio, sys
async def restore():
    from core.pat_memory import get_pat_memory
    mem = await get_pat_memory()
    await mem.load_from_disk()
    ctx = await mem.get_user_context()
    n = sum(len(v) if isinstance(v, dict) else 0 for v in ctx.values())
    return f'loaded ({n} entries)'
try:
    print(asyncio.run(restore()))
except Exception as e:
    print(f'skipped ({e})')
" 2>/dev/null || echo "skipped")
fi

echo "Local Models: ${active_models:-none detected}"
echo "Mode: $autoconfig_mode | PAT Memory: $pat_memory_status"
echo "(Context auto-injected at $(date +%H:%M:%S))"
echo "---"

# Set environment variables
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  cat >> "$CLAUDE_ENV_FILE" <<ENVEOF
export RUST_LOG=info,bizra=debug
export BIZRA_ADAPTER_MODE=${autoconfig_mode}
export IHSAN_THRESHOLD=0.95
export SYNAPSE_URL=redis://:bizra_synapse_secure@127.0.0.1:6380
export BIZRA_REDIS_URL=redis://:bizra_synapse_secure@127.0.0.1:6380
export DATABASE_URL=postgresql://bizra@127.0.0.1:5433/bizra
export OLLAMA_HOST=http://127.0.0.1:11434
export LMSTUDIO_URL=http://127.0.0.1:1234
export BIZRA_ACTIVE_MODELS="${active_models}"
ENVEOF
fi

exit 0
