#!/bin/bash
# BIZRA Post-Edit Hook for Rust Files
# Triggers cargo check after Rust file modifications

FILE_PATH="$1"
LOG_FILE="/tmp/bizra-hook.log"

# Only process Rust files
if [[ "$FILE_PATH" != *.rs ]]; then
    echo '{"logged": false, "reason": "Not a Rust file"}'
    exit 0
fi

# Log the modification
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Modified: $FILE_PATH" >> "$LOG_FILE"

# Determine which crate was modified
CRATE=""
if [[ "$FILE_PATH" == *"bizra-core"* ]]; then
    CRATE="bizra-core"
elif [[ "$FILE_PATH" == *"bizra-inference"* ]]; then
    CRATE="bizra-inference"
elif [[ "$FILE_PATH" == *"bizra-federation"* ]]; then
    CRATE="bizra-federation"
elif [[ "$FILE_PATH" == *"bizra-api"* ]]; then
    CRATE="bizra-api"
elif [[ "$FILE_PATH" == *"bizra-installer"* ]]; then
    CRATE="bizra-installer"
elif [[ "$FILE_PATH" == *"bizra-tests"* ]]; then
    CRATE="bizra-tests"
fi

# Output result
if [[ -n "$CRATE" ]]; then
    echo "{\"logged\": true, \"crate\": \"$CRATE\", \"file\": \"$FILE_PATH\", \"action\": \"suggest_check\"}"
else
    echo "{\"logged\": true, \"file\": \"$FILE_PATH\"}"
fi
