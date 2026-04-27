#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Workspace Surgery — Audit-Driven Cleanup
# ═══════════════════════════════════════════════════════════════════════════════
#
# Targets confirmed findings from multi-lens audit:
#   259 MagicMock artifact files at repo root
#   Duplicate trees: bizra_constitution/ vs bizra-constitution/
#   Stale PIDs, __pycache__ debris
#   Sprint A: CI false-pass risk fixes
#
# Usage:
#   bash workspace_surgery.sh --audit     # Report only (safe, no changes)
#   bash workspace_surgery.sh --clean     # Execute cleanup
#   bash workspace_surgery.sh --sprint-a  # Apply Sprint A code fixes
#
# Standing on Giants: Deming (PDCA quality) · Boyd (OODA observe-first)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PYTEST_PIPE_SCAN_LABEL="pytest piped into tail"
PYTEST_CMD_TOKEN='py'"test"
TAIL_CMD_TOKEN='ta'"il"
PYTEST_PIPE_TAIL_REGEX="${PYTEST_CMD_TOKEN}.*|.*${TAIL_CMD_TOKEN}"

# ── Locate workspace ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/../bizra-constitution" ] && [ -d "$SCRIPT_DIR/../core" ]; then
    ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -d "$SCRIPT_DIR/bizra-constitution" ] && [ -d "$SCRIPT_DIR/core" ]; then
    ROOT="$SCRIPT_DIR"
elif [ -d "/mnt/c/BIZRA-DATA-LAKE/bizra-constitution" ]; then
    ROOT="/mnt/c/BIZRA-DATA-LAKE"
elif [ -d "bizra-constitution" ] && [ -d "core" ]; then
    ROOT="$(pwd)"
else
    echo "ERROR: Cannot find BIZRA-DATA-LAKE. Run from repo root."
    exit 1
fi

cd "$ROOT"
MODE="${1:---audit}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  BIZRA Workspace Surgery"
echo "  Root: $(pwd)"
echo "  Mode: $MODE"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: MagicMock Artifact Files
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 1: MagicMock artifact files"
echo "─────────────────────────────────"

MOCK_COUNT=0
MOCK_FILES=()
while IFS= read -r -d '' f; do
    MOCK_FILES+=("$f")
    MOCK_COUNT=$((MOCK_COUNT + 1))
done < <(find . -maxdepth 1 -type f -name "*MagicMock*" -print0 2>/dev/null)

# Also catch angle-bracket pattern
while IFS= read -r -d '' f; do
    local_name="$(basename "$f")"
    case "$local_name" in
        *MagicMock*) ;; # Already counted
        *)
            MOCK_FILES+=("$f")
            MOCK_COUNT=$((MOCK_COUNT + 1))
            ;;
    esac
done < <(find . -maxdepth 1 -type f -name "*mock_*.tmp" -print0 2>/dev/null)

echo "  Found: $MOCK_COUNT MagicMock artifact files"
if [ "$MOCK_COUNT" -gt 0 ] && [ "$MOCK_COUNT" -le 5 ]; then
    for f in "${MOCK_FILES[@]}"; do
        echo "    $(basename "$f")"
    done
elif [ "$MOCK_COUNT" -gt 5 ]; then
    echo "    (showing first 3)"
    for f in "${MOCK_FILES[@]:0:3}"; do
        echo "    $(basename "$f")"
    done
    echo "    ... and $((MOCK_COUNT - 3)) more"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Duplicate Constitution Trees
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 2: Duplicate constitution trees"
echo "──────────────────────────────────────"

DUPE_TREE=false
DUPE_DETAIL=""

if [ -d "bizra-constitution" ] && [ -d "bizra_constitution" ]; then
    DUPE_TREE=true

    HYPHEN_PY=$(find bizra-constitution -maxdepth 1 -name "*.py" 2>/dev/null | wc -l)
    UNDER_PY=$(find bizra_constitution -maxdepth 1 -name "*.py" 2>/dev/null | wc -l)

    echo "  bizra-constitution/ (hyphen): $HYPHEN_PY .py files  <- canonical v6"
    echo "  bizra_constitution/ (underscore): $UNDER_PY .py files  <- needs review"

    UNDER_IMPORTS=$(grep -rl "from bizra_constitution" core/ 2>/dev/null | wc -l)
    UNDER_IMPORTS2=$(grep -rl "import bizra_constitution" core/ 2>/dev/null | wc -l)
    TOTAL_UNDER_IMPORTS=$((UNDER_IMPORTS + UNDER_IMPORTS2))

    if [ "$TOTAL_UNDER_IMPORTS" -gt 0 ]; then
        echo "  WARNING: core/ has $TOTAL_UNDER_IMPORTS imports from bizra_constitution (underscore)"
        grep -rl "bizra_constitution" core/ 2>/dev/null | head -5 | while read -r f; do
            echo "      $f"
        done
        DUPE_DETAIL="ACTIVE_IMPORT"
    else
        echo "  OK: core/ does NOT import from bizra_constitution (underscore)"
        DUPE_DETAIL="SAFE_TO_REMOVE"
    fi

    BRIDGE_UNDER=$(grep -c "bizra_constitution" core/bridges/constitutional_engine.py 2>/dev/null || echo 0)
    if [ "$BRIDGE_UNDER" -gt 0 ]; then
        echo "  WARNING: Bridge references bizra_constitution (underscore)"
        DUPE_DETAIL="ACTIVE_IMPORT"
    fi

    if [ -f "bizra_constitution/__init__.py" ]; then
        echo "  bizra_constitution/ has __init__.py (is a Python package)"
    fi

    echo ""
    echo "  File overlap analysis:"
    OVERLAP=0
    ONLY_UNDER=0
    for f in bizra_constitution/*.py; do
        [ -f "$f" ] || continue
        basename_f="$(basename "$f")"
        if [ -f "bizra-constitution/$basename_f" ]; then
            OVERLAP=$((OVERLAP + 1))
        else
            ONLY_UNDER=$((ONLY_UNDER + 1))
            echo "    UNIQUE to underscore: $basename_f"
        fi
    done
    echo "    Overlapping files: $OVERLAP"
    echo "    Unique to underscore: $ONLY_UNDER"
else
    echo "  Only one constitution tree found -- no duplicate issue"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: Root-level constitution file orphans
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 3: Constitution orphans at repo root"
echo "───────────────────────────────────────────"

CONSTITUTION_FILES=(
    constitution.toml bizra_constitution.py generate_from_constitution.py
    ihsan_gate.py snr.py evidence_receipt.py reflex_cache.py
    hhmm_router.py mission_pipeline.py identity_genesis.py
    ollama_provider.py production_pipeline.py genesis_engine.py
    node0_server.py node0_wire.py node0_cli.py verify_all.py
    poi.proto MIGRATION.md WIRE_GUIDE.md
)

ORPHAN_COUNT=0
for f in "${CONSTITUTION_FILES[@]}"; do
    if [ -f "$f" ]; then
        if [ -f "bizra-constitution/$f" ]; then
            echo "  X $f  (duplicate -- canonical in bizra-constitution/)"
            ORPHAN_COUNT=$((ORPHAN_COUNT + 1))
        else
            echo "  ? $f  (root only -- NO canonical copy, needs manual review)"
        fi
    fi
done
echo "  Found: $ORPHAN_COUNT root orphans with canonical copies"

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: Stale PIDs + build debris
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 4: Stale PIDs + debris"
echo "────────────────────────────"

PID_COUNT=0
while IFS= read -r pid_file; do
    PID=$(cat "$pid_file" 2>/dev/null || echo "")
    if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
        echo "  X $pid_file  (PID $PID not running)"
        PID_COUNT=$((PID_COUNT + 1))
    fi
done < <(find . -name "*.pid" -not -path "./.git/*" 2>/dev/null)

PYCACHE_COUNT=$(find . -maxdepth 6 -type d -name "__pycache__" -not -path "./.git/*" 2>/dev/null | wc -l)
PYTEST_CACHE=$(find . -maxdepth 2 -type d -name ".pytest_cache" -not -path "./.git/*" 2>/dev/null | wc -l)

echo "  Stale PIDs: $PID_COUNT"
echo "  __pycache__: $PYCACHE_COUNT directories"
echo "  .pytest_cache: $PYTEST_CACHE directories"

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: Root-level generated/ and tests/ duplicates
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 5: Root-level build directories"
echo "──────────────────────────────────────"

ROOT_BUILD_DIRS=0
if [ -d "generated" ] && [ -d "bizra-constitution/generated" ]; then
    echo "  X generated/  (duplicate of bizra-constitution/generated/)"
    ROOT_BUILD_DIRS=$((ROOT_BUILD_DIRS + 1))
fi

if [ -d "tests" ]; then
    CONST_TESTS_AT_ROOT=$(find tests -maxdepth 1 \( -name "test_ihsan*" -o -name "test_evidence*" -o -name "test_hhmm*" -o -name "test_reflex*" -o -name "test_mission_pipeline*" -o -name "test_node0*" -o -name "test_identity*" -o -name "test_ollama*" -o -name "test_production*" \) 2>/dev/null | wc -l)
    if [ "$CONST_TESTS_AT_ROOT" -gt 0 ]; then
        echo "  WARNING: tests/ contains $CONST_TESTS_AT_ROOT constitution test files (may be duplicates)"
    fi
fi

echo "  Root build dirs to remove: $ROOT_BUILD_DIRS"

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  AUDIT SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo "  MagicMock artifacts:      $MOCK_COUNT"
if [ "$DUPE_TREE" = true ]; then
    echo "  Duplicate tree:           YES ($DUPE_DETAIL)"
else
    echo "  Duplicate tree:           NO"
fi
echo "  Root orphan files:        $ORPHAN_COUNT"
echo "  Stale PIDs:               $PID_COUNT"
echo "  __pycache__ dirs:         $PYCACHE_COUNT"
echo "  .pytest_cache dirs:       $PYTEST_CACHE"
echo "  Root build dirs:          $ROOT_BUILD_DIRS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ "$MODE" = "--audit" ]; then
    echo "  Next: Run with --clean to execute removal"
    echo "  Note: bizra_constitution/ (underscore) requires manual decision"
    echo "        if core/ imports from it."
    echo ""
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# CLEAN MODE
# ═══════════════════════════════════════════════════════════════════

if [ "$MODE" != "--clean" ] && [ "$MODE" != "--sprint-a" ]; then
    echo "Usage: $0 --audit | --clean | --sprint-a"
    exit 1
fi

if [ "$MODE" = "--clean" ]; then
    echo "Executing cleanup..."
    echo ""
    REMOVED=0
    ERRORS=0

    # Phase 1: MagicMock files
    echo "  Removing MagicMock artifacts..."
    MOCK_REMOVED=0
    while IFS= read -r -d '' f; do
        rm -f "$f" 2>/dev/null && MOCK_REMOVED=$((MOCK_REMOVED + 1)) || ERRORS=$((ERRORS + 1))
    done < <(find . -maxdepth 1 -type f -name "*MagicMock*" -print0 2>/dev/null)
    echo "    Removed $MOCK_REMOVED MagicMock files"
    REMOVED=$((REMOVED + MOCK_REMOVED))

    # Phase 3: Root orphans
    echo "  Removing root-level constitution orphans..."
    for f in "${CONSTITUTION_FILES[@]}"; do
        if [ -f "$f" ] && [ -f "bizra-constitution/$f" ]; then
            rm -f "$f" && echo "    Removed $f" && REMOVED=$((REMOVED + 1)) \
                || { echo "    FAILED $f"; ERRORS=$((ERRORS + 1)); }
        fi
    done

    # Phase 4: Stale PIDs
    echo "  Removing stale PIDs..."
    while IFS= read -r pid_file; do
        PID=$(cat "$pid_file" 2>/dev/null || echo "")
        if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
            rm -f "$pid_file" && echo "    Removed $pid_file" && REMOVED=$((REMOVED + 1)) \
                || { echo "    FAILED $pid_file"; ERRORS=$((ERRORS + 1)); }
        fi
    done < <(find . -name "*.pid" -not -path "./.git/*" 2>/dev/null)

    # Phase 5: Root build dirs
    if [ -d "generated" ] && [ -d "bizra-constitution/generated" ]; then
        rm -rf "generated" && echo "    Removed generated/" && REMOVED=$((REMOVED + 1)) \
            || { echo "    FAILED generated/"; ERRORS=$((ERRORS + 1)); }
    fi

    # __pycache__ purge
    echo "  Purging __pycache__..."
    CACHE_PURGED=$(find . -maxdepth 6 -type d -name "__pycache__" -not -path "./.git/*" 2>/dev/null | wc -l)
    find . -maxdepth 6 -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
    echo "    Purged $CACHE_PURGED __pycache__ directories"
    REMOVED=$((REMOVED + CACHE_PURGED))

    # .pytest_cache purge
    echo "  Purging .pytest_cache..."
    PCACHE_PURGED=$(find . -maxdepth 2 -type d -name ".pytest_cache" -not -path "./.git/*" 2>/dev/null | wc -l)
    find . -maxdepth 2 -type d -name ".pytest_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
    echo "    Purged $PCACHE_PURGED .pytest_cache directories"
    REMOVED=$((REMOVED + PCACHE_PURGED))

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  CLEANUP COMPLETE"
    echo "  Removed: $REMOVED items"
    echo "  Errors:  $ERRORS"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    if [ "$DUPE_TREE" = true ]; then
        echo "  MANUAL DECISION REQUIRED:"
        echo "    bizra_constitution/ (underscore) still exists."
        if [ "$DUPE_DETAIL" = "ACTIVE_IMPORT" ]; then
            echo "    core/ imports from it -- DO NOT delete without updating imports."
            echo ""
            echo "    Options:"
            echo "    A) Symlink: rm -rf bizra_constitution && ln -s bizra-constitution bizra_constitution"
            echo "       (makes underscore resolve to hyphen -- cleanest)"
            echo "    B) Keep both (current state -- adds drift risk)"
            echo ""
            echo "    Recommended: Option A (symlink)"
        else
            echo "    core/ does NOT import from it -- safe to remove."
            echo "    Run: rm -rf bizra_constitution/"
        fi
    fi

    echo ""
    echo "  Verification -- canonical bizra-constitution/:"
    CANON_PY=$(find bizra-constitution -maxdepth 1 -name "*.py" 2>/dev/null | wc -l)
    CANON_TESTS=$(find bizra-constitution/tests -name "test_*.py" 2>/dev/null | wc -l)
    echo "    Modules: $CANON_PY"
    echo "    Tests:   $CANON_TESTS"
    if [ "$CANON_PY" -ge 15 ] && [ "$CANON_TESTS" -ge 10 ]; then
        echo "    Canonical v6 package intact"
    else
        echo "    WARNING: Check bizra-constitution/ -- expected 15+ modules, 10+ tests"
    fi
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# SPRINT A: CI False-Pass + Test Stability Fixes
# ═══════════════════════════════════════════════════════════════════

if [ "$MODE" = "--sprint-a" ]; then
    echo ""
    echo "Sprint A: CI & Test Stability Fixes"
    echo "════════════════════════════════════"
    echo ""

    echo "  Fix 1: CI false-pass risk (${PYTEST_PIPE_SCAN_LABEL})"
    echo "  ─────────────────────────────────────────────────"
    PIPED_TESTS=$(grep -rn "$PYTEST_PIPE_TAIL_REGEX" scripts/ deploy/ .github/ .claude/ 2>/dev/null || true)
    if [ -n "$PIPED_TESTS" ]; then
        echo "  Found piped test patterns:"
        echo "$PIPED_TESTS" | while IFS= read -r line; do
            echo "    $line"
        done
        echo ""
        echo "  Fix: Replace 'pytest ... | tail -N' with:"
        echo '    pytest ... ; RESULT=$?'
        echo '    if [ $RESULT -ne 0 ]; then echo "FAILED"; exit $RESULT; fi'
        echo ""
    else
        echo "  OK: No ${PYTEST_PIPE_SCAN_LABEL} patterns found"
    fi

    echo "  Fix 2: Wire integration state awareness"
    echo "  ────────────────────────────────────────"
    if [ -f "scripts/wire_live_integration.py" ]; then
        HARD_ASSERTS=$(grep -n "== 3\|chain_count == 3" scripts/wire_live_integration.py 2>/dev/null || true)
        if [ -n "$HARD_ASSERTS" ]; then
            echo "  Found hard-coded count assertions:"
            echo "$HARD_ASSERTS" | while IFS= read -r line; do
                echo "    $line"
            done
            echo ""
            echo "  Fix: Baseline chain count before test, assert delta instead"
        else
            echo "  OK: No problematic hard-coded assertions"
        fi
    else
        echo "  WARNING: scripts/wire_live_integration.py not found"
    fi
    echo ""

    echo "  Fix 3: node0_server TestClient timeout guard"
    echo "  ─────────────────────────────────────────────"
    if [ -f "bizra-constitution/tests/test_node0_server.py" ]; then
        HAS_TIMEOUT=$(grep -c "timeout\|TIMEOUT\|pytest.mark.timeout" bizra-constitution/tests/test_node0_server.py 2>/dev/null || echo 0)
        if [ "$HAS_TIMEOUT" -eq 0 ]; then
            echo "  No timeout guard on test_node0_server.py"
            echo "  Fix: Add 'pytestmark = pytest.mark.timeout(30)' at module level"
        else
            echo "  OK: Timeout guard exists"
        fi
    fi
    echo ""

    echo "  Sprint A analysis complete."
    echo ""
fi
