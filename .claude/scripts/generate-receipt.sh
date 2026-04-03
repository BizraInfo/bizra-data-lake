#!/bin/bash
# BIZRA Receipt Generator
# Creates BIZRA-compliant evidence receipts
#
# Usage: ./generate-receipt.sh [type] [summary] [options]
#
# Types:
#   build       Build operation receipt
#   test        Test run receipt
#   validation  Validation gate receipt
#   commit      Git commit receipt
#   deploy      Deployment receipt
#   evidence    Meta-evidence receipt
#
# Options:
#   -s, --status   Status (success/failure, default: success)
#   -e, --error    Error/rejection code
#   -l, --level    Escalation level (None/Low/Medium/High/Critical)
#   -o, --output   Output directory (default: docs/evidence/receipts)
#   -p, --parent   Parent receipt ID (for evidence chains)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Defaults
TYPE="evidence"
SUMMARY=""
STATUS="success"
ERROR_CODE=""
ESCALATION="None"
OUTPUT_DIR="$PROJECT_DIR/docs/evidence/receipts"
PARENT_RECEIPT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    build|test|validation|commit|deploy|evidence)
      TYPE="$1"
      shift
      ;;
    -s|--status)
      STATUS="$2"
      shift 2
      ;;
    -e|--error)
      ERROR_CODE="$2"
      shift 2
      ;;
    -l|--level)
      ESCALATION="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -p|--parent)
      PARENT_RECEIPT="$2"
      shift 2
      ;;
    *)
      if [ -z "$SUMMARY" ]; then
        SUMMARY="$1"
      fi
      shift
      ;;
  esac
done

# Default summary if not provided
if [ -z "$SUMMARY" ]; then
  SUMMARY="$TYPE operation completed"
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Generate timestamp
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TIMESTAMP_SHORT=$(date -u +%Y%m%d-%H%M%S)

# Generate receipt ID
RECEIPT_ID="${TYPE}-${TIMESTAMP_SHORT}-$(openssl rand -hex 4)"

# Build rejection codes array
if [ -n "$ERROR_CODE" ]; then
  REJECTION_CODES="[\"$ERROR_CODE\"]"
else
  REJECTION_CODES="[]"
fi

# Calculate integrity hash
HASH_INPUT="${RECEIPT_ID}${TIMESTAMP}${SUMMARY}"
INTEGRITY_HASH=$(echo -n "$HASH_INPUT" | sha256sum | cut -d' ' -f1)

# Build evidence chain metadata
if [ -n "$PARENT_RECEIPT" ]; then
  CHAIN_METADATA=",
  \"evidence_chain\": {
    \"parent_receipts\": [\"$PARENT_RECEIPT\"],
    \"chain_depth\": 1
  }"
else
  CHAIN_METADATA=""
fi

# Generate receipt JSON
RECEIPT=$(cat << EOF
{
  "receipt_id": "$RECEIPT_ID",
  "timestamp": "$TIMESTAMP",
  "task_summary": "$SUMMARY",
  "rejection_codes": $REJECTION_CODES,
  "escalation_level": "$ESCALATION",
  "integrity_hash": "$INTEGRITY_HASH",
  "metadata": {
    "type": "$TYPE",
    "status": "$STATUS",
    "generator": "bizra-receipt-generator",
    "generator_version": "1.0.0"
  }$CHAIN_METADATA
}
EOF
)

# Output file path
OUTPUT_FILE="$OUTPUT_DIR/${RECEIPT_ID}.json"

# Write receipt
echo "$RECEIPT" | jq '.' > "$OUTPUT_FILE"

# Output result
echo "Receipt generated:"
echo "  ID: $RECEIPT_ID"
echo "  File: $OUTPUT_FILE"
echo "  Hash: $INTEGRITY_HASH"

# Return receipt ID for chaining
echo "$RECEIPT_ID"
