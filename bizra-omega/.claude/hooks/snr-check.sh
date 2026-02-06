#!/bin/bash
# BIZRA SNR Quality Check Hook
# Validates content quality against SNR threshold

CONTENT="$1"
SNR_THRESHOLD="${2:-0.85}"

# Simple heuristic SNR check (word diversity + length balance)
if [[ -z "$CONTENT" ]]; then
    echo '{"snr": 0.0, "pass": false, "reason": "Empty content"}'
    exit 0
fi

# Count words and unique words
WORD_COUNT=$(echo "$CONTENT" | wc -w)
UNIQUE_WORDS=$(echo "$CONTENT" | tr ' ' '\n' | sort -u | wc -l)

if [[ $WORD_COUNT -eq 0 ]]; then
    echo '{"snr": 0.0, "pass": false, "reason": "No words"}'
    exit 0
fi

# Calculate diversity ratio
DIVERSITY=$(echo "scale=3; $UNIQUE_WORDS / $WORD_COUNT" | bc)

# Estimate SNR (simplified)
SNR=$(echo "scale=3; $DIVERSITY * 0.85 + 0.15" | bc)

# Check threshold
PASS=$(echo "$SNR >= $SNR_THRESHOLD" | bc)

if [[ $PASS -eq 1 ]]; then
    echo "{\"snr\": $SNR, \"pass\": true, \"diversity\": $DIVERSITY, \"words\": $WORD_COUNT}"
else
    echo "{\"snr\": $SNR, \"pass\": false, \"diversity\": $DIVERSITY, \"words\": $WORD_COUNT, \"threshold\": $SNR_THRESHOLD}"
fi
