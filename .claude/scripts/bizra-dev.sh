#!/bin/bash
# BIZRA Development Session Helper
# Starts Claude Code with optimal BIZRA configuration
#
# Usage: ./bizra-dev.sh [command] [options]
#
# Commands:
#   start     Start new dev session (default)
#   continue  Continue last session
#   resume    Resume named session
#   quick     Quick non-interactive query
#   validate  Run validation
#   status    Show project status
#
# Options:
#   -m, --model   Model to use (sonnet/opus)
#   -p, --plan    Start in plan mode
#   -v, --verbose Enable verbose logging
#   -n, --name    Session name (for resume)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Defaults
COMMAND="start"
MODEL=""
PLAN_MODE=false
VERBOSE=false
SESSION_NAME=""
QUERY=""

# BIZRA system prompt addition
BIZRA_CONTEXT="BIZRA Development Session.
Core Principles:
- Receipt-First: All operations emit evidence receipts
- Fail-Closed: Critical errors block execution
- Ihsān Gate: 0.95 production threshold (8 dimensions)
- SAPE 9-Probes: Validation before output
- Evidence Chains: Complete audit trails

Quick Commands:
- /rust, /python, /docker, /ihsan, /sape, /receipts, /commit"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    start|continue|resume|quick|validate|status)
      COMMAND="$1"
      shift
      ;;
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -p|--plan)
      PLAN_MODE=true
      shift
      ;;
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    -n|--name)
      SESSION_NAME="$2"
      shift 2
      ;;
    *)
      # Remaining args are the query
      QUERY="$*"
      break
      ;;
  esac
done

# Build command
build_cmd() {
  local cmd="claude"

  # Add major in-repo workspaces explicitly for large multi-surface tasks.
  for extra_dir in ace-framework HyperGraphRAG bizra-genesis-node BIZRA-PAT; do
    if [ -d "$PROJECT_DIR/$extra_dir" ]; then
      cmd="$cmd --add-dir '$PROJECT_DIR/$extra_dir'"
    fi
  done

  # Add model
  if [ -n "$MODEL" ]; then
    cmd="$cmd --model $MODEL"
  fi

  # Add plan mode
  if [ "$PLAN_MODE" = true ]; then
    cmd="$cmd --permission-mode plan"
  fi

  # Add verbose
  if [ "$VERBOSE" = true ]; then
    cmd="$cmd --verbose"
  fi

  echo "$cmd"
}

case $COMMAND in
  start)
    echo "Starting BIZRA development session..."
    cmd="$(build_cmd) --append-system-prompt \"$BIZRA_CONTEXT\""
    if [ -n "$QUERY" ]; then
      cmd="$cmd \"$QUERY\""
    fi
    eval "$cmd"
    ;;

  continue)
    echo "Continuing last BIZRA session..."
    cmd="$(build_cmd) -c"
    if [ -n "$QUERY" ]; then
      cmd="$cmd -p \"$QUERY\""
    fi
    eval "$cmd"
    ;;

  resume)
    if [ -z "$SESSION_NAME" ]; then
      echo "Error: Session name required. Use -n or --name"
      echo "Usage: ./bizra-dev.sh resume -n session-name"
      exit 1
    fi
    echo "Resuming session: $SESSION_NAME"
    cmd="$(build_cmd) -r \"$SESSION_NAME\""
    if [ -n "$QUERY" ]; then
      cmd="$cmd \"$QUERY\""
    fi
    eval "$cmd"
    ;;

  quick)
    if [ -z "$QUERY" ]; then
      echo "Error: Query required for quick command"
      echo "Usage: ./bizra-dev.sh quick 'your query'"
      exit 1
    fi
    echo "Running quick query..."
    cmd="$(build_cmd) -p --max-turns 5 \"$QUERY\""
    eval "$cmd"
    ;;

  validate)
    echo "Running BIZRA validation..."
    "$SCRIPT_DIR/bizra-validate.sh" "$@"
    ;;

  status)
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║               BIZRA Project Status                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Git status
    echo "📦 Git Status:"
    cd "$PROJECT_DIR"
    echo "   Branch: $(git branch --show-current)"
    echo "   Changes: $(git status --short | wc -l) files"

    # Rust status
    echo ""
    echo "🦀 Rust:"
    if [ -f "$PROJECT_DIR/Cargo.toml" ]; then
      echo "   Project: $(grep '^name' Cargo.toml | head -1 | cut -d'"' -f2)"
      if [ -f "$PROJECT_DIR/target/release/meta_alpha_dual_agentic" ]; then
        echo "   Build: Release binary exists"
      else
        echo "   Build: No release binary"
      fi
    fi

    # Python status
    echo ""
    echo "🐍 Python:"
    if python3 -c "from core import main" 2>/dev/null; then
      echo "   Imports: OK"
    else
      echo "   Imports: FAIL"
    fi

    # Docker status
    echo ""
    echo "🐳 Docker:"
    if command -v docker &> /dev/null; then
      running=$(docker compose ps --format json 2>/dev/null | jq -r 'select(.State == "running") | .Name' | wc -l)
      echo "   Running: $running services"
    else
      echo "   Docker: Not available"
    fi

    # Receipt count
    echo ""
    echo "📋 Receipts:"
    receipt_count=$(find "$PROJECT_DIR/docs/evidence/receipts" \( -name "*.json" -o -name "*.jsonl" \) 2>/dev/null | wc -l)
    echo "   Count: $receipt_count receipts"

    # Ihsān status
    echo ""
    echo "🌟 Ihsān:"
    if [ -f "$PROJECT_DIR/constitution/ihsan_v1.yaml" ]; then
      threshold=$(python3 - <<'PY'
import yaml
try:
    with open("constitution/ihsan_v1.yaml", "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    value = (
        data.get("threshold_policy", {})
        .get("thresholds_by_env", {})
        .get("production")
    )
    if value is None:
        value = data.get("units", {}).get("threshold", "unknown")
    print(value)
except Exception:
    print("unknown")
PY
)
      echo "   Threshold: $threshold"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    ;;

  *)
    echo "Unknown command: $COMMAND"
    echo "Usage: ./bizra-dev.sh [start|continue|resume|quick|validate|status] [options]"
    exit 1
    ;;
esac
