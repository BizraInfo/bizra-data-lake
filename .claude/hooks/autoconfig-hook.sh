#!/usr/bin/env bash
# BIZRA Auto-Configuration Hook
# Probes services and injects runtime config at session start
set -e

INPUT=$(cat)

# Run auto-configurator
echo "[BIZRA] Auto-configuring kernel..."
CONFIG_OUTPUT=$(python3 -c "
import asyncio
import json
import sys
sys.path.insert(0, '$CLAUDE_PROJECT_DIR')
try:
    from core.autoconfig import auto_configure
    config = asyncio.run(auto_configure())
    print(json.dumps(config, indent=2))
except Exception as e:
    print(json.dumps({'error': str(e), 'mode': 'simulated'}), file=sys.stderr)
    sys.exit(1)
" 2>&1)

if [ $? -ne 0 ]; then
    echo "[WARN] Auto-configuration failed: $CONFIG_OUTPUT"
    CONFIG_OUTPUT='{"mode": "simulated", "error": "autoconfig failed"}'
fi

# Parse config
MODE=$(echo "$CONFIG_OUTPUT" | jq -r '.mode // "simulated"')
CAPABILITIES=$(echo "$CONFIG_OUTPUT" | jq -r '.capabilities // [] | join(", ")')
OLLAMA_STATUS=$(echo "$CONFIG_OUTPUT" | jq -r '.services.ollama.reachable // false')
LMSTUDIO_STATUS=$(echo "$CONFIG_OUTPUT" | jq -r '.services.lmstudio.reachable // false')
REDIS_STATUS=$(echo "$CONFIG_OUTPUT" | jq -r '.services.redis.reachable // false')
NEO4J_STATUS=$(echo "$CONFIG_OUTPUT" | jq -r '.services.neo4j.reachable // false')

# Extract available models
OLLAMA_MODELS=$(echo "$CONFIG_OUTPUT" | jq -r '.services.ollama.info.models // [] | join(", ")')
LMSTUDIO_MODELS=$(echo "$CONFIG_OUTPUT" | jq -r '.services.lmstudio.info.models // [] | join(", ")')
ALL_MODELS="${OLLAMA_MODELS:+$OLLAMA_MODELS}${LMSTUDIO_MODELS:+,$LMSTUDIO_MODELS}"
ALL_MODELS=$(echo "$ALL_MODELS" | sed 's/^,//' | sed 's/,$//')

# Extract primary reasoning model
PRIMARY_REASONING=$(echo "$CONFIG_OUTPUT" | jq -r '.model_routing.cold_core.model // .model_routing.primary_reasoning.model // "none"')

# Extract online services
ONLINE_SERVICES=""
for svc in ollama lmstudio redis neo4j chromadb postgres; do
    REACHABLE=$(echo "$CONFIG_OUTPUT" | jq -r ".services.$svc.reachable // false")
    if [ "$REACHABLE" = "true" ]; then
        ONLINE_SERVICES="${ONLINE_SERVICES:+$ONLINE_SERVICES,}$svc"
    fi
done

# Output context
cat <<EOF
--- BIZRA Auto-Configuration ---
Mode: $MODE
Capabilities: ${CAPABILITIES:-none}

Backend Services:
  - Ollama: ${OLLAMA_STATUS} (${OLLAMA_MODELS:-no models})
  - LM Studio: ${LMSTUDIO_STATUS} (${LMSTUDIO_MODELS:-no models})
  - Redis/Synapse: ${REDIS_STATUS}
  - Neo4j/Wisdom: ${NEO4J_STATUS}

Primary Reasoning Model: $PRIMARY_REASONING
Online Services: ${ONLINE_SERVICES:-none}

Full config saved to: ~/.bizra/autoconfig.json
---
EOF

# Set environment variables
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    cat >> "$CLAUDE_ENV_FILE" <<ENVEOF
export BIZRA_ACTIVE_MODELS="$ALL_MODELS"
export BIZRA_PRIMARY_REASONING="$PRIMARY_REASONING"
export BIZRA_ADAPTER_MODE="$MODE"
export BIZRA_SERVICES_ONLINE="$ONLINE_SERVICES"
ENVEOF
fi

exit 0
