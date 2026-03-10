#!/bin/bash
# Self-Healing Hook for BIZRA-DATA-LAKE
# Automatically detects and suggests fixes for common errors

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error patterns database
PATTERNS_FILE="/mnt/c/BIZRA-DATA-LAKE/.claude-flow/memory/error-patterns.json"

# Function to detect error type
detect_error_type() {
    local error_output="$1"

    # Check for import errors
    if echo "$error_output" | grep -q "ImportError.*AutonomyLevel.*autonomy"; then
        echo "import-autonomy-level"
        return
    fi

    # Check for Event field name errors
    if echo "$error_output" | grep -q "Event.__init__.*event_type"; then
        echo "event-field-names"
        return
    fi

    # Check for missing aiohttp
    if echo "$error_output" | grep -q "ModuleNotFoundError.*aiohttp"; then
        echo "missing-aiohttp"
        return
    fi

    # Check for async/await issues
    if echo "$error_output" | grep -q "can't be used in 'await' expression"; then
        echo "eventbus-stop-not-async"
        return
    fi

    # Check for pytest collection errors
    if echo "$error_output" | grep -q "ERROR collecting.*ImportError"; then
        echo "pytest-collection-error"
        return
    fi

    echo "unknown"
}

# Function to suggest fix
suggest_fix() {
    local error_type="$1"

    case "$error_type" in
        "import-autonomy-level")
            echo -e "${YELLOW}FIX:${NC} Change import from:"
            echo "  from .autonomy import AutonomyLevel"
            echo "To:"
            echo "  from .autonomy_matrix import AutonomyLevel"
            ;;
        "event-field-names")
            echo -e "${YELLOW}FIX:${NC} Change Event constructor:"
            echo "  Event(event_type=..., data=...)"
            echo "To:"
            echo "  Event(topic=..., payload=...)"
            ;;
        "missing-aiohttp")
            echo -e "${YELLOW}FIX:${NC} Make aiohttp optional with fallback:"
            echo "  try:"
            echo "      import aiohttp"
            echo "  except ImportError:"
            echo "      import httpx  # or use urllib"
            ;;
        "eventbus-stop-not-async")
            echo -e "${YELLOW}FIX:${NC} event_bus.stop() is sync, not async:"
            echo "  Change: await event_bus.stop()"
            echo "  To: event_bus.stop()"
            ;;
        "pytest-collection-error")
            echo -e "${YELLOW}FIX:${NC} Check these common issues:"
            echo "  1. Verify __init__.py exports the module"
            echo "  2. Check import paths are correct"
            echo "  3. Ensure all dependencies are installed"
            ;;
        *)
            echo -e "${RED}Unknown error type${NC}"
            echo "Check .claude-flow/memory/error-patterns.json for patterns"
            ;;
    esac
}

# Main self-healing logic
main() {
    local exit_code="${1:-0}"
    local command_output="${2:-}"

    if [ "$exit_code" -eq 0 ]; then
        exit 0
    fi

    echo -e "${RED}Error detected (exit code: $exit_code)${NC}"

    if [ -n "$command_output" ]; then
        error_type=$(detect_error_type "$command_output")

        if [ "$error_type" != "unknown" ]; then
            echo -e "${GREEN}Recognized error pattern: $error_type${NC}"
            suggest_fix "$error_type"

            # Log to patterns file
            echo "$(date -Iseconds) | $error_type | auto-detected" >> /mnt/c/BIZRA-DATA-LAKE/.claude-flow/memory/error-log.txt
        fi
    fi
}

# Run if called directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
