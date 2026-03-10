#!/bin/bash
# BIZRA Pre-Edit Hook for Rust Files
# Validates Rust files before modification

FILE_PATH="$1"

# Check if it's a Rust file
if [[ "$FILE_PATH" != *.rs ]]; then
    echo '{"continue": true, "reason": "Not a Rust file"}'
    exit 0
fi

# Check for protected modules
PROTECTED_MODULES=(
    "identity.rs"
    "crypto.rs"
    "envelope.rs"
)

BASENAME=$(basename "$FILE_PATH")
for module in "${PROTECTED_MODULES[@]}"; do
    if [[ "$BASENAME" == "$module" ]]; then
        echo '{"continue": true, "decision": "warn", "reason": "Modifying core security module: '"$module"'"}'
        exit 0
    fi
done

echo '{"continue": true, "decision": "allow", "reason": "Standard Rust file edit"}'
