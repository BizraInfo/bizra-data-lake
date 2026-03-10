#!/bin/bash
# BIZRA Smart Agent Auto-Spawning
# "Every inference carries proof. Every decision passes the gate."

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$ROOT_DIR/.swarm/smart-agents-config.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         BIZRA Smart Agent Auto-Spawning System                   ║${NC}"
echo -e "${CYAN}║              Ihsān ≥ 0.95 | SNR ≥ 0.85                           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo

# Function: Detect file types in a path
detect_file_types() {
    local path="$1"
    local types=""

    if find "$path" -name "*.py" 2>/dev/null | head -1 | grep -q .; then
        types="$types python"
    fi
    if find "$path" -name "*.ts" -o -name "*.tsx" 2>/dev/null | head -1 | grep -q .; then
        types="$types typescript"
    fi
    if find "$path" -name "*.rs" 2>/dev/null | head -1 | grep -q .; then
        types="$types rust"
    fi
    if find "$path" -name "*.json" -o -name "*.yaml" -o -name "*.yml" 2>/dev/null | head -1 | grep -q .; then
        types="$types config"
    fi
    if find "$path" -name "*.md" 2>/dev/null | head -1 | grep -q .; then
        types="$types documentation"
    fi

    echo "$types"
}

# Function: Determine task complexity
determine_complexity() {
    local task="$1"
    local lower_task=$(echo "$task" | tr '[:upper:]' '[:lower:]')

    # Critical (sovereignty-related)
    if echo "$lower_task" | grep -qE "(security|sovereignty|federation|consensus|ihsan|fate|gate)"; then
        echo "critical"
        return
    fi

    # Complex
    if echo "$lower_task" | grep -qE "(architect|design|integrate|migrate|optimize|refactor)"; then
        echo "complex"
        return
    fi

    # Moderate
    if echo "$lower_task" | grep -qE "(implement|add|create|build|develop)"; then
        echo "moderate"
        return
    fi

    # Simple
    echo "simple"
}

# Function: Get recommended agents for complexity
get_agents_for_complexity() {
    local complexity="$1"

    case "$complexity" in
        "critical")
            echo "architect security coder tester reviewer"
            ;;
        "complex")
            echo "architect coder tester researcher"
            ;;
        "moderate")
            echo "coder tester"
            ;;
        "simple")
            echo "coder"
            ;;
    esac
}

# Function: Get agents for file type
get_agents_for_filetype() {
    local filetype="$1"

    case "$filetype" in
        "python")
            echo "coder tester"
            ;;
        "typescript")
            echo "coder architect"
            ;;
        "rust")
            echo "coder security"
            ;;
        "config")
            echo "analyst"
            ;;
        "documentation")
            echo "researcher documenter"
            ;;
        *)
            echo "coder"
            ;;
    esac
}

# Function: Spawn agent (simulated - would connect to claude-flow)
spawn_agent() {
    local agent_type="$1"
    local task_desc="$2"

    echo -e "  ${GREEN}✓${NC} Spawning ${BLUE}$agent_type${NC} agent"

    # If claude-flow is available, use it
    if command -v npx &> /dev/null && [ -f "$ROOT_DIR/node_modules/.bin/claude-flow" ]; then
        npx claude-flow agent spawn --type "$agent_type" --task "$task_desc" 2>/dev/null || true
    fi
}

# Function: Analyze task and spawn agents
analyze_and_spawn() {
    local task="$1"
    local path="${2:-.}"

    echo -e "${YELLOW}Task:${NC} $task"
    echo -e "${YELLOW}Path:${NC} $path"
    echo

    # Determine complexity
    local complexity=$(determine_complexity "$task")
    echo -e "${CYAN}Complexity:${NC} $complexity"

    # Get agents for complexity
    local complexity_agents=$(get_agents_for_complexity "$complexity")
    echo -e "${CYAN}Complexity Agents:${NC} $complexity_agents"

    # Detect file types
    local file_types=$(detect_file_types "$path")
    echo -e "${CYAN}File Types:${NC} $file_types"

    # Collect all unique agents
    local all_agents="$complexity_agents"
    for ft in $file_types; do
        local ft_agents=$(get_agents_for_filetype "$ft")
        all_agents="$all_agents $ft_agents"
    done

    # Deduplicate
    local unique_agents=$(echo "$all_agents" | tr ' ' '\n' | sort -u | tr '\n' ' ')

    echo
    echo -e "${GREEN}=== Spawning Agents ===${NC}"

    for agent in $unique_agents; do
        [ -n "$agent" ] && spawn_agent "$agent" "$task"
    done

    echo
    echo -e "${GREEN}=== Agent Swarm Ready ===${NC}"

    # Log to memory
    local log_file="$ROOT_DIR/.swarm/agent-spawn-log.json"
    cat >> "$log_file" << EOF
{"timestamp": "$(date -Iseconds)", "task": "$task", "complexity": "$complexity", "agents": "$unique_agents"}
EOF
}

# Function: Show current agent status
show_status() {
    echo -e "${CYAN}=== Current Agent Configuration ===${NC}"
    echo

    if [ -f "$CONFIG_FILE" ]; then
        echo -e "${GREEN}Config:${NC} $CONFIG_FILE"

        # Show file type rules
        echo
        echo -e "${YELLOW}File Type → Agent Mapping:${NC}"
        echo "  Python      → coder, tester"
        echo "  TypeScript  → coder, architect"
        echo "  Rust        → coder, security"
        echo "  Config      → analyst"
        echo "  Markdown    → researcher, documenter"

        # Show complexity rules
        echo
        echo -e "${YELLOW}Complexity → Agent Mapping:${NC}"
        echo "  Simple   → coder (1 agent)"
        echo "  Moderate → coder, tester (3 agents)"
        echo "  Complex  → architect, coder, tester, researcher (6 agents)"
        echo "  Critical → architect, security, coder, tester, reviewer (8 agents)"

        # Show BIZRA-specific rules
        echo
        echo -e "${YELLOW}BIZRA-Specific Rules:${NC}"
        echo "  Sovereign Runtime  → architect, coder, security [CRITICAL]"
        echo "  FATE Binding       → coder, security, tester [CRITICAL]"
        echo "  Federation         → architect, coder, tester [HIGH]"
        echo "  IPC Bridge         → coder, optimizer [HIGH]"
        echo "  Capability Card    → security, coder, tester [CRITICAL]"
    else
        echo -e "${RED}Config not found: $CONFIG_FILE${NC}"
    fi
}

# Main
case "${1:-status}" in
    status)
        show_status
        ;;
    spawn)
        if [ -z "$2" ]; then
            echo "Usage: $0 spawn \"task description\" [path]"
            exit 1
        fi
        analyze_and_spawn "$2" "${3:-.}"
        ;;
    test)
        echo "=== Testing Smart Agent Spawning ==="
        echo

        echo "Test 1: Simple task"
        analyze_and_spawn "Fix typo in README" "."
        echo

        echo "Test 2: Complex task"
        analyze_and_spawn "Implement federation consensus protocol" "./core/federation"
        echo

        echo "Test 3: Critical task"
        analyze_and_spawn "Add FATE validation to sovereignty gate" "./native/fate-binding"
        ;;
    *)
        echo "Usage: $0 [status|spawn|test]"
        echo
        echo "Commands:"
        echo "  status              Show current agent configuration"
        echo "  spawn \"task\" [path] Analyze task and spawn appropriate agents"
        echo "  test                Run test spawning scenarios"
        ;;
esac
