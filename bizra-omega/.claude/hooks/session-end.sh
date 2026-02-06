#!/bin/bash
# BIZRA Session End Hook
# Generates session summary and persists memory

MEMORY_DIR="/mnt/c/BIZRA-DATA-LAKE/bizra-omega/.claude-flow/memory"
LOG_FILE="/tmp/bizra-hook.log"
SUMMARY_FILE="$MEMORY_DIR/session-summary-$(date '+%Y%m%d-%H%M%S').json"

# Count modifications from log
RUST_MODS=$(grep -c "\.rs$" "$LOG_FILE" 2>/dev/null || echo "0")
TOTAL_HOOKS=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")

# Generate summary
cat > "$SUMMARY_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "session_stats": {
    "rust_files_modified": $RUST_MODS,
    "total_hook_events": $TOTAL_HOOKS
  },
  "log_file": "$LOG_FILE"
}
EOF

# Clear log for next session
: > "$LOG_FILE"

echo "{\"summary_file\": \"$SUMMARY_FILE\", \"rust_modifications\": $RUST_MODS}"
