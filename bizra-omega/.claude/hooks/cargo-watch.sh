#!/bin/bash
# BIZRA Cargo Watch Hook
# Triggers incremental compilation on Rust changes

CRATE="$1"
ACTION="${2:-check}"  # check, build, test

cd /mnt/c/BIZRA-DATA-LAKE/bizra-omega || exit 1

case "$ACTION" in
    check)
        cargo check --package "$CRATE" 2>&1 | tail -5
        ;;
    build)
        cargo build --package "$CRATE" --release 2>&1 | tail -5
        ;;
    test)
        cargo test --package "$CRATE" 2>&1 | tail -10
        ;;
    *)
        echo '{"error": "Unknown action: '"$ACTION"'"}'
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "{\"crate\": \"$CRATE\", \"action\": \"$ACTION\", \"exit_code\": $EXIT_CODE}"
