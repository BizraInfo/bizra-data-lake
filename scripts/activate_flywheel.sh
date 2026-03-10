#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#    BIZRA FLYWHEEL ACTIVATION SCRIPT
#    
#    Activates the complete flywheel stack:
#    1. Verifies genesis readiness
#    2. Starts embedded Ollama
#    3. Preloads models (warm start)
#    4. Starts flywheel API
#    5. Verifies everything is operational
#    
#    Usage: ./activate_flywheel.sh [--skip-genesis-check]
#    
#    Created: 2026-01-29 | BIZRA Sovereignty
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    BIZRA FLYWHEEL — Activation Sequence${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Genesis Readiness Check
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$1" != "--skip-genesis-check" ]]; then
    echo -e "${YELLOW}[Phase 1] Genesis Readiness Verification${NC}"
    
    if python3 verify_genesis_readiness.py --json > /tmp/genesis_check.json 2>&1; then
        echo -e "${GREEN}✅ Genesis checks passed${NC}"
    else
        echo -e "${RED}❌ Genesis checks failed${NC}"
        cat /tmp/genesis_check.json
        echo
        echo -e "${YELLOW}Fix issues or run with --skip-genesis-check to bypass${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[Phase 1] Genesis check skipped${NC}"
fi

echo

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Environment Setup
# ═══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Phase 2] Environment Setup${NC}"

# Check for required environment variables
if [[ -z "$BIZRA_API_TOKEN" ]]; then
    echo -e "${YELLOW}⚠️  BIZRA_API_TOKEN not set — generating secure token${NC}"
    export BIZRA_API_TOKEN=$(openssl rand -hex 32)
    echo "   Generated token (save this): ${BIZRA_API_TOKEN:0:16}..."
    echo "   export BIZRA_API_TOKEN=$BIZRA_API_TOKEN" >> ~/.bizra_env
    echo -e "${YELLOW}   Token saved to ~/.bizra_env${NC}"
fi

# Create .env file if it doesn't exist
if [[ ! -f ".env.flywheel" ]]; then
    cat > .env.flywheel << EOF
# BIZRA Flywheel Environment
BIZRA_API_TOKEN=${BIZRA_API_TOKEN}
BIZRA_AUTH_MODE=FAIL_CLOSED
LOG_LEVEL=INFO
FLYWHEEL_API_PORT=8100
FLYWHEEL_AUDIO_PORT=8101
EOF
    echo -e "${GREEN}✅ Created .env.flywheel${NC}"
fi

echo

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Docker Stack Launch
# ═══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Phase 3] Docker Stack Launch${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi

# Check for existing flywheel containers
if docker ps -q -f "name=bizra-flywheel" | grep -q .; then
    echo -e "${YELLOW}⚠️  Flywheel already running — restarting${NC}"
    docker compose -f docker-compose.flywheel.yml down
fi

# Start the flywheel stack
echo "   Starting Ollama, Model Preloader, and Flywheel..."
docker compose --env-file .env.flywheel -f docker-compose.flywheel.yml up -d

echo

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Wait for Services
# ═══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Phase 4] Waiting for Services${NC}"

# Wait for Ollama
echo -n "   Ollama: "
for i in {1..60}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for model preloader to complete
echo -n "   Model Preloader: "
for i in {1..120}; do
    STATUS=$(docker inspect -f '{{.State.Status}}' bizra-model-preloader 2>/dev/null || echo "missing")
    if [[ "$STATUS" == "exited" ]]; then
        EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' bizra-model-preloader 2>/dev/null || echo "1")
        if [[ "$EXIT_CODE" == "0" ]]; then
            echo -e "${GREEN}✅${NC}"
        else
            echo -e "${RED}❌ (exit $EXIT_CODE)${NC}"
        fi
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for Flywheel API
echo -n "   Flywheel API: "
for i in {1..60}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

echo

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Verification
# ═══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Phase 5] Verification${NC}"

# Test health endpoint
HEALTH=$(curl -s http://localhost:8100/health)
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")

if [[ "$STATUS" == "healthy" ]]; then
    echo -e "${GREEN}✅ Flywheel is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Flywheel status: $STATUS${NC}"
fi

# Test authenticated endpoint
echo -n "   Testing auth: "
AUTH_RESULT=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $BIZRA_API_TOKEN" http://localhost:8100/status)
if [[ "$AUTH_RESULT" == "200" ]]; then
    echo -e "${GREEN}✅ Authentication working${NC}"
else
    echo -e "${RED}❌ Auth failed (HTTP $AUTH_RESULT)${NC}"
fi

# Test inference
echo -n "   Testing inference: "
INFER_RESULT=$(curl -s -X POST \
    -H "Authorization: Bearer $BIZRA_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Say OK","max_tokens":5}' \
    http://localhost:8100/infer 2>/dev/null)

if echo "$INFER_RESULT" | grep -q "response"; then
    echo -e "${GREEN}✅ Inference operational${NC}"
else
    echo -e "${YELLOW}⚠️  Inference may not be ready yet${NC}"
fi

echo

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    FLYWHEEL ACTIVATED${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo
echo "   Endpoints:"
echo "   - Health:    http://localhost:8100/health"
echo "   - Status:    http://localhost:8100/status (auth required)"
echo "   - Inference: http://localhost:8100/infer (auth required)"
echo "   - Embed:     http://localhost:8100/embed (auth required)"
echo "   - Audio WS:  ws://localhost:8100/ws/audio (auth required)"
echo
echo "   API Token: ${BIZRA_API_TOKEN:0:16}..."
echo "   (Full token in ~/.bizra_env)"
echo
echo "   Commands:"
echo "   - View logs: docker logs -f bizra-flywheel"
echo "   - Stop:      docker compose -f docker-compose.flywheel.yml down"
echo "   - Status:    curl -H 'Authorization: Bearer \$BIZRA_API_TOKEN' http://localhost:8100/status"
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
