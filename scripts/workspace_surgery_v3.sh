#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Workspace Surgery v3.1 — Audit-Driven Cleanup
# ═══════════════════════════════════════════════════════════════════════════════
# Modes:
#   --audit     Report only (safe)
#   --clean     Quarantine cleanup to 99_QUARANTINE/
#   --nuke      Hard delete cleanup (irreversible)
#   --sprint-a  Patch CI/test stability issues (fail-closed)
#
# Design:
# - No global pipefail; grep no-match is data, not fatal.
# - All filesystem traversals use -print0.
# - Deletion checks live references before removing duplicate root files.
# - Sprint-A patching is per-file one-pass (no stale line-number drift).
# ═══════════════════════════════════════════════════════════════════════════════

set -eu

find_workspace_root() {
    local script_dir parent_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    parent_dir="$(dirname "$script_dir")"

    if [ -d "$script_dir/bizra-constitution" ] && [ -d "$script_dir/core" ]; then
        echo "$script_dir"; return 0
    fi
    if [ -d "$parent_dir/bizra-constitution" ] && [ -d "$parent_dir/core" ]; then
        echo "$parent_dir"; return 0
    fi
    if [ -d "bizra-constitution" ] && [ -d "core" ]; then
        pwd; return 0
    fi
    if [ -d "/mnt/c/BIZRA-DATA-LAKE/bizra-constitution" ]; then
        echo "/mnt/c/BIZRA-DATA-LAKE"; return 0
    fi
    return 1
}

is_wsl() {
    grep -qi "microsoft" /proc/version 2>/dev/null
}

is_py_module_referenced() {
    # Exact import forms only:
    #   from <module> import ...
    #   import <module>
    local module_name="$1"
    local re
    re="^[[:space:]]*(from[[:space:]]+${module_name}([[:space:]]|\\.|$)|import[[:space:]]+${module_name}([[:space:]]|\\.|$))"
    local count
    count="$(grep -rEl "$re" core/ scripts/ deploy/ tests/ 2>/dev/null | wc -l || true)"
    [ "$count" -gt 0 ]
}

is_path_referenced_literal() {
    local rel_name="$1"
    local count
    count="$(grep -rFl "$rel_name" core/ scripts/ deploy/ tests/ docs/ .github/ 2>/dev/null | wc -l || true)"
    [ "$count" -gt 0 ]
}

is_referenced() {
    local filename="$1"
    local module_name ext
    ext="${filename##*.}"

    if [ "$filename" = "$ext" ]; then
        # No extension: treat as literal path token only.
        is_path_referenced_literal "$filename"
        return $?
    fi

    if [ "$ext" = "py" ]; then
        module_name="${filename%.py}"
        if is_py_module_referenced "$module_name"; then
            return 0
        fi
    fi

    is_path_referenced_literal "$filename"
}

quarantine_item() {
    local item="$1"
    local category="$2"
    local target_dir="$QUARANTINE_DIR/$category"
    mkdir -p "$target_dir"
    mv "$item" "$target_dir/" 2>/dev/null
}

delete_item() {
    local item="$1"
    if [ -d "$item" ]; then
        rm -rf "$item" 2>/dev/null
    else
        rm -f "$item" 2>/dev/null
    fi
}

remove_item() {
    local item="$1"
    local category="$2"
    if [ "$MODE" = "--nuke" ]; then
        delete_item "$item"
    else
        quarantine_item "$item" "$category"
    fi
}

scan_pytest_tail_lines() {
    if [ "${#CI_SCAN_PATHS[@]}" -eq 0 ]; then
        return 0
    fi

    if command -v rg >/dev/null 2>&1; then
        rg -n --no-messages --max-filesize 1M \
            --glob '*.sh' --glob '*.bash' --glob '*.zsh' \
            --glob '*.yml' --glob '*.yaml' \
            --glob '!scripts/workspace_surgery*.sh' \
            --glob '!**/.venv*/**' --glob '!**/node_modules/**' \
            'pytest.*\|.*tail' "${CI_SCAN_PATHS[@]}" || true
    else
        grep -rn 'pytest.*|.*tail' "${CI_SCAN_PATHS[@]}" 2>/dev/null \
            | grep -vE 'scripts/workspace_surgery[^:]*\.sh:' || true
    fi
}

scan_pytest_tail_files() {
    if [ "${#CI_SCAN_PATHS[@]}" -eq 0 ]; then
        return 0
    fi

    if command -v rg >/dev/null 2>&1; then
        rg -l --no-messages --max-filesize 1M \
            --glob '*.sh' --glob '*.bash' --glob '*.zsh' \
            --glob '*.yml' --glob '*.yaml' \
            --glob '!scripts/workspace_surgery*.sh' \
            --glob '!**/.venv*/**' --glob '!**/node_modules/**' \
            'pytest.*\|.*tail' "${CI_SCAN_PATHS[@]}" || true
    else
        grep -rl 'pytest.*|.*tail' "${CI_SCAN_PATHS[@]}" 2>/dev/null \
            | grep -vE '^scripts/workspace_surgery[^/]*\.sh$' || true
    fi
}

scan_pid_files() {
    find . -maxdepth 1 -type f -name "*.pid" -print0 2>/dev/null || true
    for base in "${CLEAN_SCAN_PATHS[@]}"; do
        find "$base" -maxdepth 10 \
            \( -path "$base/.venv" -o -path "$base/.venv/*" \
            -o -path "$base/.venv-linux" -o -path "$base/.venv-linux/*" \
            -o -path "$base/node_modules" -o -path "$base/node_modules/*" \
            -o -path "$base/99_QUARANTINE" -o -path "$base/99_QUARANTINE/*" \) -prune \
            -o -type f -name "*.pid" -print0 2>/dev/null || true
    done
}

scan_pycache_dirs() {
    for base in "${CLEAN_SCAN_PATHS[@]}"; do
        find "$base" -maxdepth 12 \
            \( -path "$base/.venv" -o -path "$base/.venv/*" \
            -o -path "$base/.venv-linux" -o -path "$base/.venv-linux/*" \
            -o -path "$base/node_modules" -o -path "$base/node_modules/*" \
            -o -path "$base/99_QUARANTINE" -o -path "$base/99_QUARANTINE/*" \) -prune \
            -o -type d -name "__pycache__" -print0 2>/dev/null || true
    done
}

scan_pytest_cache_dirs() {
    for base in "${CLEAN_SCAN_PATHS[@]}"; do
        find "$base" -maxdepth 8 \
            \( -path "$base/.venv" -o -path "$base/.venv/*" \
            -o -path "$base/.venv-linux" -o -path "$base/.venv-linux/*" \
            -o -path "$base/node_modules" -o -path "$base/node_modules/*" \
            -o -path "$base/99_QUARANTINE" -o -path "$base/99_QUARANTINE/*" \) -prune \
            -o -type d -name ".pytest_cache" -print0 2>/dev/null || true
    done
}

count_scan_results() {
    local c=0
    while IFS= read -r -d '' _; do
        c=$((c + 1))
    done
    echo "$c"
}

ROOT="$(find_workspace_root)" || { echo "ERROR: Cannot find BIZRA-DATA-LAKE."; exit 1; }
cd "$ROOT"

MODE="${1:---audit}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
QUARANTINE_DIR="99_QUARANTINE/workspace_surgery_${TIMESTAMP}"
CI_SCAN_PATHS=()
for p in scripts deploy .github .claude; do
    [ -e "$p" ] && CI_SCAN_PATHS+=("$p")
done
CLEAN_SCAN_PATHS=()
for p in core tests scripts deploy bizra-constitution bizra_constitution .github .claude docs; do
    [ -d "$p" ] && CLEAN_SCAN_PATHS+=("$p")
done

case "$MODE" in
    --audit|--clean|--nuke|--sprint-a) ;;
    *)
        echo "Usage: $0 --audit | --clean | --nuke | --sprint-a"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  BIZRA Workspace Surgery v3.1"
echo "  Root: $(pwd)"
echo "  Mode: $MODE"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$MODE" = "--clean" ]; then
    echo "  Quarantine: $QUARANTINE_DIR"
fi
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: MagicMock artifacts
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 1: MagicMock artifact files"
echo "─────────────────────────────────"

MOCK_COUNT=0
while IFS= read -r -d '' _f; do
    MOCK_COUNT=$((MOCK_COUNT + 1))
done < <(find . -maxdepth 1 -type f \( -name "<MagicMock*" -o -name "MagicMock*" -o -name "*mock_*.tmp" \) -print0 2>/dev/null || true)

echo "  Found: $MOCK_COUNT MagicMock artifacts"

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Duplicate constitution trees
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 2: Duplicate constitution trees"
echo "──────────────────────────────────────"

DUPE_TREE=false
DUPE_DETAIL=""

if [ -d "bizra-constitution" ] && [ -d "bizra_constitution" ]; then
    DUPE_TREE=true
    HYPHEN_PY="$(find bizra-constitution -maxdepth 1 -name "*.py" 2>/dev/null | wc -l || true)"
    UNDER_PY="$(find -L bizra_constitution -maxdepth 1 -name "*.py" 2>/dev/null | wc -l || true)"
    IS_SYMLINK_BRIDGE=false
    if [ -L "bizra_constitution" ]; then
        CANON_REAL="$(readlink -f bizra-constitution 2>/dev/null || true)"
        UNDER_REAL="$(readlink -f bizra_constitution 2>/dev/null || true)"
        if [ -n "$CANON_REAL" ] && [ "$UNDER_REAL" = "$CANON_REAL" ]; then
            IS_SYMLINK_BRIDGE=true
        fi
    fi

    echo "  bizra-constitution/ (hyphen): $HYPHEN_PY .py <- canonical"
    echo "  bizra_constitution/ (underscore): $UNDER_PY .py <- review"
    if [ "$IS_SYMLINK_BRIDGE" = true ]; then
        echo "  OK underscore path is a symlink bridge to canonical"
    fi

    UNDER_IMPORTS="$(grep -rEl '^[[:space:]]*(from[[:space:]]+bizra_constitution([[:space:]]|\.|$)|import[[:space:]]+bizra_constitution([[:space:]]|\.|$))' core/ 2>/dev/null | wc -l || true)"
    if [ "$IS_SYMLINK_BRIDGE" = true ]; then
        if [ "$UNDER_IMPORTS" -gt 0 ]; then
            echo "  OK core/ imports resolve through symlink bridge ($UNDER_IMPORTS files)"
        else
            echo "  OK core/ has no underscore imports; bridge is harmless"
        fi
        DUPE_DETAIL="SYMLINK_BRIDGE"
    else
        if [ "$UNDER_IMPORTS" -gt 0 ]; then
            echo "  !! core/ has $UNDER_IMPORTS files importing bizra_constitution"
            grep -rEl '^[[:space:]]*(from[[:space:]]+bizra_constitution([[:space:]]|\.|$)|import[[:space:]]+bizra_constitution([[:space:]]|\.|$))' core/ 2>/dev/null | head -5 | while IFS= read -r f; do
                echo "      $f"
            done
            DUPE_DETAIL="ACTIVE_IMPORT"
        else
            echo "  OK core/ does NOT import from bizra_constitution"
            DUPE_DETAIL="SAFE_TO_REMOVE"
        fi
    fi

    echo ""
    echo "  File overlap:"
    OVERLAP=0
    ONLY_UNDER=0
    while IFS= read -r -d '' f; do
        bf="$(basename "$f")"
        if [ -f "bizra-constitution/$bf" ]; then
            OVERLAP=$((OVERLAP + 1))
        else
            ONLY_UNDER=$((ONLY_UNDER + 1))
            echo "    UNIQUE to underscore: $bf"
        fi
    done < <(find -L bizra_constitution -maxdepth 1 -name "*.py" -print0 2>/dev/null || true)

    echo "    Overlapping: $OVERLAP"
    echo "    Unique:     $ONLY_UNDER"

    echo ""
    echo "  Resolution strategy:"
    if [ "$IS_SYMLINK_BRIDGE" = true ]; then
        echo "    Bridge already configured correctly."
    elif is_wsl; then
        echo "    Platform: WSL"
        echo "    A) ln -s bizra-constitution bizra_constitution"
        echo "    B) Windows junction: mklink /J bizra_constitution bizra-constitution"
    else
        echo "    Platform: $(uname -s)"
        echo "    Recommended: ln -s bizra-constitution bizra_constitution"
    fi
else
    echo "  Only one constitution tree found"
fi

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: Root-level constitution duplicate files
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 3: Root-level constitution orphans"
echo "─────────────────────────────────────────"

CONSTITUTION_FILES=(
    constitution.toml bizra_constitution.py generate_from_constitution.py
    ihsan_gate.py snr.py evidence_receipt.py reflex_cache.py
    hhmm_router.py mission_pipeline.py identity_genesis.py
    ollama_provider.py production_pipeline.py genesis_engine.py
    node0_server.py node0_wire.py node0_cli.py verify_all.py
    poi.proto MIGRATION.md WIRE_GUIDE.md
)

ORPHAN_SAFE=0
ORPHAN_REFERENCED=0
for f in "${CONSTITUTION_FILES[@]}"; do
    if [ -f "$f" ] && [ -f "bizra-constitution/$f" ]; then
        if is_referenced "$f"; then
            echo "  !! $f  (duplicate but referenced)"
            ORPHAN_REFERENCED=$((ORPHAN_REFERENCED + 1))
        else
            echo "  X  $f  (duplicate, unreferenced)"
            ORPHAN_SAFE=$((ORPHAN_SAFE + 1))
        fi
    fi
done

echo "  Safe to remove: $ORPHAN_SAFE"
echo "  Referenced:     $ORPHAN_REFERENCED"

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: Stale PIDs + cache debris
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 4: Stale PIDs + cache debris"
echo "───────────────────────────────────"

PID_COUNT=0
while IFS= read -r -d '' pid_file; do
    PID="$(cat "$pid_file" 2>/dev/null || echo "")"
    if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
        echo "  X  $pid_file  (PID $PID not running)"
        PID_COUNT=$((PID_COUNT + 1))
    fi
done < <(scan_pid_files)

PYCACHE_COUNT="$(scan_pycache_dirs | count_scan_results)"
PYTEST_CACHE="$(scan_pytest_cache_dirs | count_scan_results)"

echo "  Stale PIDs: $PID_COUNT"
echo "  __pycache__: $PYCACHE_COUNT"
echo "  .pytest_cache: $PYTEST_CACHE"

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: Root build dirs
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 5: Root-level build directories"
echo "──────────────────────────────────────"

ROOT_BUILD_DIRS=0
if [ -d "generated" ] && [ -d "bizra-constitution/generated" ]; then
    echo "  X  generated/ (duplicate)"
    ROOT_BUILD_DIRS=$((ROOT_BUILD_DIRS + 1))
fi

if [ -d "tests" ]; then
    CONST_TESTS_AT_ROOT="$(find tests -maxdepth 1 \( -name "test_ihsan*" -o -name "test_evidence*" -o -name "test_hhmm*" -o -name "test_reflex*" -o -name "test_mission_pipeline*" -o -name "test_node0*" -o -name "test_identity*" -o -name "test_ollama*" -o -name "test_production*" \) 2>/dev/null | wc -l || true)"
    if [ "$CONST_TESTS_AT_ROOT" -gt 0 ]; then
        echo "  ?? tests/ has $CONST_TESTS_AT_ROOT constitution test files"
    fi
fi

echo "  Root build dirs: $ROOT_BUILD_DIRS"

# ═══════════════════════════════════════════════════════════════════
# PHASE 6: CI false-pass scan
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Phase 6: CI false-pass risk (pytest|tail)"
echo "──────────────────────────────────────────"

PIPED_COUNT=0
while IFS= read -r line; do
    echo "  !! $line"
    PIPED_COUNT=$((PIPED_COUNT + 1))
done < <(scan_pytest_tail_lines)

if [ "$PIPED_COUNT" -eq 0 ]; then
    echo "  OK No pytest|tail patterns found"
else
    echo "  Found: $PIPED_COUNT"
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  AUDIT SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo "  MagicMock artifacts:       $MOCK_COUNT"
if [ "$DUPE_TREE" = true ]; then
    echo "  Duplicate tree:            YES ($DUPE_DETAIL)"
else
    echo "  Duplicate tree:            NO"
fi
echo "  Root orphans (safe):       $ORPHAN_SAFE"
echo "  Root orphans (referenced): $ORPHAN_REFERENCED"
echo "  Stale PIDs:                $PID_COUNT"
echo "  __pycache__:               $PYCACHE_COUNT"
echo "  .pytest_cache:             $PYTEST_CACHE"
echo "  Root build dirs:           $ROOT_BUILD_DIRS"
echo "  CI false-pass risks:       $PIPED_COUNT"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ "$MODE" = "--audit" ]; then
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# CLEAN / NUKE
# ═══════════════════════════════════════════════════════════════════
if [ "$MODE" = "--clean" ] || [ "$MODE" = "--nuke" ]; then
    if [ "$MODE" = "--nuke" ]; then
        echo "Deleting..."
    else
        echo "Quarantining..."
    fi
    echo ""

    REMOVED=0
    SKIPPED=0
    ERRORS=0

    # MagicMock
    MOCK_REMOVED=0
    while IFS= read -r -d '' f; do
        if remove_item "$f" "magicmock"; then
            MOCK_REMOVED=$((MOCK_REMOVED + 1))
        else
            ERRORS=$((ERRORS + 1))
        fi
    done < <(find . -maxdepth 1 -type f \( -name "<MagicMock*" -o -name "MagicMock*" -o -name "*mock_*.tmp" \) -print0 2>/dev/null || true)
    REMOVED=$((REMOVED + MOCK_REMOVED))

    # Root duplicate files (unreferenced only)
    for f in "${CONSTITUTION_FILES[@]}"; do
        if [ -f "$f" ] && [ -f "bizra-constitution/$f" ]; then
            if is_referenced "$f"; then
                SKIPPED=$((SKIPPED + 1))
            elif remove_item "$f" "constitution_orphans"; then
                REMOVED=$((REMOVED + 1))
            else
                ERRORS=$((ERRORS + 1))
            fi
        fi
    done

    # Stale PIDs
    while IFS= read -r -d '' pid_file; do
        PID="$(cat "$pid_file" 2>/dev/null || echo "")"
        if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
            if remove_item "$pid_file" "stale_pids"; then
                REMOVED=$((REMOVED + 1))
            else
                ERRORS=$((ERRORS + 1))
            fi
        fi
    done < <(scan_pid_files)

    # Root generated
    if [ -d "generated" ] && [ -d "bizra-constitution/generated" ]; then
        if remove_item "generated" "build_dirs"; then
            REMOVED=$((REMOVED + 1))
        else
            ERRORS=$((ERRORS + 1))
        fi
    fi

    # Cache purge (safe debris)
    CACHE_PURGED=0
    while IFS= read -r -d '' d; do
        rm -rf "$d" 2>/dev/null && CACHE_PURGED=$((CACHE_PURGED + 1))
    done < <(scan_pycache_dirs)

    PCACHE_PURGED=0
    while IFS= read -r -d '' d; do
        rm -rf "$d" 2>/dev/null && PCACHE_PURGED=$((PCACHE_PURGED + 1))
    done < <(scan_pytest_cache_dirs)

    REMOVED=$((REMOVED + CACHE_PURGED + PCACHE_PURGED))

    echo "═══════════════════════════════════════════════════════════════"
    echo "  CLEANUP COMPLETE"
    echo "  Removed:     $REMOVED"
    echo "  Skipped:     $SKIPPED"
    echo "  Errors:      $ERRORS"
    if [ "$MODE" = "--clean" ]; then
        echo "  Quarantine:  $QUARANTINE_DIR"
    fi
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# SPRINT-A PATCHES
# ═══════════════════════════════════════════════════════════════════
if [ "$MODE" = "--sprint-a" ]; then
    echo ""
    echo "Sprint A: CI & Test Stability Patches"
    echo "══════════════════════════════════════"
    echo ""

    FIXES_APPLIED=0

    # Fix 1: per-file, one-pass patch for pytest|tail
    echo "  Fix 1: pytest|tail -> fail-closed"
    PATCH_FILES=()
    while IFS= read -r f; do
        [ -n "$f" ] && PATCH_FILES+=("$f")
    done < <(scan_pytest_tail_files)

    if [ "${#PATCH_FILES[@]}" -eq 0 ]; then
        echo "    No files to patch"
    else
        for file in "${PATCH_FILES[@]}"; do
            [ -f "$file" ] || continue
            [ -f "${file}.sprint-a.bak" ] || cp "$file" "${file}.sprint-a.bak"

            awk '
            {
                if ($0 ~ /pytest.*\|.*tail/ && $0 !~ /_pytest_rc=\$\?/) {
                    line = $0
                    idx = index(line, "| tail")
                    if (idx > 0) {
                        cmd = substr(line, 1, idx - 1)
                        match(line, /^[[:space:]]*/)
                        indent = substr(line, RSTART, RLENGTH)
                        print indent "# Sprint A: fail-closed (was piped to tail)"
                        print indent cmd
                        print indent "_pytest_rc=$?"
                        print indent "if [ $_pytest_rc -ne 0 ]; then echo \"PYTEST FAILED (exit $_pytest_rc)\"; exit $_pytest_rc; fi"
                        next
                    }
                }
                print
            }
            ' "$file" > "${file}.tmp"

            if ! cmp -s "$file" "${file}.tmp"; then
                mv "${file}.tmp" "$file"
                FIXES_APPLIED=$((FIXES_APPLIED + 1))
                echo "    Patched: $file"
            else
                rm -f "${file}.tmp"
            fi
        done
    fi

    echo ""

    # Fix 2: node0 server timeout marker
    echo "  Fix 2: node0_server test timeout"
    SERVER_TEST="bizra-constitution/tests/test_node0_server.py"
    if [ -f "$SERVER_TEST" ]; then
        if ! grep -q "pytestmark[[:space:]]*=[[:space:]]*pytest\.mark\.timeout" "$SERVER_TEST" 2>/dev/null; then
            [ -f "${SERVER_TEST}.sprint-a.bak" ] || cp "$SERVER_TEST" "${SERVER_TEST}.sprint-a.bak"

            if ! grep -q "^import pytest" "$SERVER_TEST" 2>/dev/null; then
                # Insert import pytest after initial import block start.
                awk '
                BEGIN { inserted=0 }
                {
                    print
                    if (!inserted && $0 ~ /^import /) {
                        print "import pytest"
                        inserted=1
                    }
                }
                ' "$SERVER_TEST" > "${SERVER_TEST}.tmp1" && mv "${SERVER_TEST}.tmp1" "$SERVER_TEST"
            fi

            LAST_IMPORT="$(grep -nE '^(from |import )' "$SERVER_TEST" | tail -1 | cut -d: -f1 || true)"
            if [ -n "$LAST_IMPORT" ]; then
                awk -v line="$LAST_IMPORT" '
                {
                    print
                    if (NR == line) {
                        print ""
                        print "# Sprint A: prevent TestClient lifespan hang"
                        print "pytestmark = pytest.mark.timeout(45)"
                    }
                }
                ' "$SERVER_TEST" > "${SERVER_TEST}.tmp2" && mv "${SERVER_TEST}.tmp2" "$SERVER_TEST"
                FIXES_APPLIED=$((FIXES_APPLIED + 1))
                echo "    Patched: $SERVER_TEST"
            else
                echo "    WARN: could not find import block in $SERVER_TEST"
            fi
        else
            echo "    Already patched: $SERVER_TEST"
        fi
    else
        echo "    SKIP: $SERVER_TEST not found"
    fi

    echo ""

    # Fix 3: wire integration delta assertions
    echo "  Fix 3: wire integration state-aware chain assertions"
    WIRE_SCRIPT="scripts/wire_live_integration.py"
    if [ -f "$WIRE_SCRIPT" ]; then
        if grep -q "_baseline_chain_count =" "$WIRE_SCRIPT" 2>/dev/null \
            && grep -q "_delta_chain == len(missions)" "$WIRE_SCRIPT" 2>/dev/null; then
            echo "    Already patched: $WIRE_SCRIPT"
        else
            [ -f "${WIRE_SCRIPT}.sprint-a.bak" ] || cp "$WIRE_SCRIPT" "${WIRE_SCRIPT}.sprint-a.bak"

            awk '
            BEGIN {
                inserted_baseline=0
            }
            {
                if (!inserted_baseline && $0 ~ /^[[:space:]]*for i, text in enumerate\(missions, 1\):/) {
                    print "    # Sprint A: baseline chain count for delta assertions"
                    print "    _baseline_health = wire.health() if wire else {}"
                    print "    _baseline_chain_count = (_baseline_health.get(\"pipeline_health\", {}) or {}).get(\"evidence_chain_count\", 0)"
                    print ""
                    inserted_baseline=1
                }

                if ($0 ~ /^[[:space:]]*assert chain_count == 3,/) {
                    print "    # Sprint A: delta-based assertion (replaces absolute count)"
                    print "    _delta_chain = chain_count - _baseline_chain_count"
                    print "    assert _delta_chain == len(missions), f\"Expected +{len(missions)}, got +{_delta_chain}\""
                    next
                }

                print
            }
            ' "$WIRE_SCRIPT" > "${WIRE_SCRIPT}.tmp"

            if ! cmp -s "$WIRE_SCRIPT" "${WIRE_SCRIPT}.tmp"; then
                mv "${WIRE_SCRIPT}.tmp" "$WIRE_SCRIPT"
                FIXES_APPLIED=$((FIXES_APPLIED + 1))
                echo "    Patched: $WIRE_SCRIPT"
            else
                rm -f "${WIRE_SCRIPT}.tmp"
                echo "    No patch needed: $WIRE_SCRIPT"
            fi
        fi
    else
        echo "    SKIP: $WIRE_SCRIPT not found"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Sprint A complete"
    echo "  Files changed: $FIXES_APPLIED"
    echo "  Backups: *.sprint-a.bak"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    exit 0
fi

echo "Usage: $0 --audit | --clean | --nuke | --sprint-a"
exit 1
