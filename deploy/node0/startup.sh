#!/usr/bin/env bash
# ==============================================================================
# BIZRA NODE0 STARTUP SCRIPT
# ==============================================================================
#
# Unified startup script for BIZRA Genesis Node
# Performs hardware validation, service orchestration, and health verification
#
# Standing on Giants: Unix philosophy | systemd | POSIX
# Constitutional Constraint: Ihsan >= 0.95
#
# Usage:
#   ./startup.sh [--check-only] [--skip-gpu] [--verbose]
#
# Exit codes:
#   0 - Success
#   1 - Hardware requirements not met
#   2 - Service startup failed
#   3 - Health check failed
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/bizra/startup.log"
readonly LOCK_FILE="/var/run/bizra/startup.lock"

# Colors (if terminal supports them)
if [[ -t 1 ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[0;33m'
    readonly BLUE='\033[0;34m'
    readonly CYAN='\033[0;36m'
    readonly NC='\033[0m' # No Color
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly BLUE=''
    readonly CYAN=''
    readonly NC=''
fi

# Hardware requirements
readonly MIN_RAM_GB=64
readonly MIN_CPU_CORES=8
readonly MIN_GPU_VRAM_GB=16
readonly MIN_DISK_FREE_GB=100

# Service endpoints
readonly API_ENDPOINT="http://localhost:3001/health"
readonly DASHBOARD_ENDPOINT="http://localhost:5173/"
readonly SOVEREIGN_ENDPOINT="http://localhost:8080/health"
readonly LMSTUDIO_ENDPOINT="http://192.168.56.1:1234/v1/models"
readonly OLLAMA_ENDPOINT="http://localhost:11434/api/tags"
readonly POSTGRES_HOST="localhost"
readonly POSTGRES_PORT=5432
readonly REDIS_HOST="localhost"
readonly REDIS_PORT=6379

# Ihsan threshold
readonly IHSAN_THRESHOLD=0.95

# ==============================================================================
# LOGGING
# ==============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

    case "$level" in
        INFO)  echo -e "${GREEN}[INFO]${NC}  $timestamp | $message" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC}  $timestamp | $message" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $timestamp | $message" ;;
        DEBUG) [[ "${VERBOSE:-false}" == "true" ]] && echo -e "${CYAN}[DEBUG]${NC} $timestamp | $message" ;;
    esac

    # Also write to log file
    echo "[$level] $timestamp | $message" >> "$LOG_FILE" 2>/dev/null || true
}

header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

CHECK_ONLY=false
SKIP_GPU=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --skip-gpu)
            SKIP_GPU=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--check-only] [--skip-gpu] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --check-only  Only check requirements, don't start services"
            echo "  --skip-gpu    Skip GPU validation (for testing)"
            echo "  --verbose     Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# HARDWARE VALIDATION
# ==============================================================================

check_cpu() {
    log INFO "Checking CPU..."

    local cores
    cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 0)

    log DEBUG "CPU cores detected: $cores"

    if [[ "$cores" -lt "$MIN_CPU_CORES" ]]; then
        log ERROR "Insufficient CPU cores: $cores (minimum: $MIN_CPU_CORES)"
        return 1
    fi

    # Check for required CPU features
    if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
        log DEBUG "AVX2 support: YES"
    else
        log WARN "AVX2 support not detected (may affect performance)"
    fi

    log INFO "CPU check passed: $cores cores"
    return 0
}

check_memory() {
    log INFO "Checking memory..."

    local total_kb total_gb
    total_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
    total_gb=$((total_kb / 1024 / 1024))

    log DEBUG "Total memory: ${total_gb}GB"

    if [[ "$total_gb" -lt "$MIN_RAM_GB" ]]; then
        log ERROR "Insufficient memory: ${total_gb}GB (minimum: ${MIN_RAM_GB}GB)"
        return 1
    fi

    # Check available memory
    local available_kb available_gb
    available_kb=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
    available_gb=$((available_kb / 1024 / 1024))

    if [[ "$available_gb" -lt 32 ]]; then
        log WARN "Low available memory: ${available_gb}GB"
    fi

    log INFO "Memory check passed: ${total_gb}GB total, ${available_gb}GB available"
    return 0
}

check_gpu() {
    if [[ "$SKIP_GPU" == "true" ]]; then
        log WARN "Skipping GPU check (--skip-gpu)"
        return 0
    fi

    log INFO "Checking GPU..."

    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &>/dev/null; then
        log ERROR "nvidia-smi not found. NVIDIA drivers may not be installed."
        return 1
    fi

    # Get GPU info
    local gpu_name gpu_vram_mb gpu_vram_gb
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    gpu_vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    gpu_vram_gb=$((gpu_vram_mb / 1024))

    log DEBUG "GPU: $gpu_name"
    log DEBUG "VRAM: ${gpu_vram_gb}GB"

    if [[ "$gpu_vram_gb" -lt "$MIN_GPU_VRAM_GB" ]]; then
        log ERROR "Insufficient GPU VRAM: ${gpu_vram_gb}GB (minimum: ${MIN_GPU_VRAM_GB}GB)"
        return 1
    fi

    # Check GPU temperature
    local gpu_temp
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)

    if [[ "$gpu_temp" -gt 85 ]]; then
        log ERROR "GPU temperature critical: ${gpu_temp}C"
        return 1
    elif [[ "$gpu_temp" -gt 75 ]]; then
        log WARN "GPU temperature elevated: ${gpu_temp}C"
    fi

    log INFO "GPU check passed: $gpu_name (${gpu_vram_gb}GB VRAM, ${gpu_temp}C)"
    return 0
}

check_disk() {
    log INFO "Checking disk space..."

    local data_lake_path="/mnt/c/BIZRA-DATA-LAKE"
    local free_kb free_gb

    if [[ -d "$data_lake_path" ]]; then
        free_kb=$(df "$data_lake_path" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
    else
        free_kb=$(df / 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
    fi

    free_gb=$((free_kb / 1024 / 1024))

    log DEBUG "Free disk space: ${free_gb}GB"

    if [[ "$free_gb" -lt "$MIN_DISK_FREE_GB" ]]; then
        log ERROR "Insufficient disk space: ${free_gb}GB (minimum: ${MIN_DISK_FREE_GB}GB)"
        return 1
    fi

    log INFO "Disk check passed: ${free_gb}GB free"
    return 0
}

check_network() {
    log INFO "Checking network..."

    # Check if we can reach LM Studio host
    if ping -c 1 -W 2 192.168.56.1 &>/dev/null; then
        log DEBUG "LM Studio host reachable"
    else
        log WARN "LM Studio host (192.168.56.1) not reachable"
    fi

    # Check localhost ports are available
    for port in 3001 5173 8080 11434; do
        if ! ss -tlnp 2>/dev/null | grep -q ":$port "; then
            log DEBUG "Port $port is available"
        else
            log WARN "Port $port is already in use"
        fi
    done

    log INFO "Network check completed"
    return 0
}

# ==============================================================================
# SERVICE MANAGEMENT
# ==============================================================================

wait_for_service() {
    local name="$1"
    local endpoint="$2"
    local timeout="${3:-60}"
    local interval="${4:-2}"

    log INFO "Waiting for $name..."

    local elapsed=0
    while [[ "$elapsed" -lt "$timeout" ]]; do
        if curl -sf "$endpoint" &>/dev/null; then
            log INFO "$name is ready"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        log DEBUG "$name not ready yet (${elapsed}s / ${timeout}s)"
    done

    log ERROR "$name failed to start within ${timeout}s"
    return 1
}

wait_for_tcp() {
    local name="$1"
    local host="$2"
    local port="$3"
    local timeout="${4:-30}"

    log INFO "Waiting for $name ($host:$port)..."

    local elapsed=0
    while [[ "$elapsed" -lt "$timeout" ]]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log INFO "$name is ready"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    log ERROR "$name failed to become available within ${timeout}s"
    return 1
}

start_infrastructure() {
    header "Starting Infrastructure Services"

    # PostgreSQL
    log INFO "Starting PostgreSQL..."
    if systemctl is-active --quiet postgresql 2>/dev/null; then
        log INFO "PostgreSQL already running"
    else
        sudo systemctl start postgresql 2>/dev/null || docker start bizra-postgres 2>/dev/null || true
    fi
    wait_for_tcp "PostgreSQL" "$POSTGRES_HOST" "$POSTGRES_PORT" 30

    # Redis
    log INFO "Starting Redis..."
    if systemctl is-active --quiet redis 2>/dev/null; then
        log INFO "Redis already running"
    else
        sudo systemctl start redis 2>/dev/null || docker start bizra-redis 2>/dev/null || true
    fi
    wait_for_tcp "Redis" "$REDIS_HOST" "$REDIS_PORT" 30

    log INFO "Infrastructure services started"
}

start_inference() {
    header "Starting Inference Layer"

    # Check LM Studio (external)
    log INFO "Checking LM Studio..."
    if curl -sf "$LMSTUDIO_ENDPOINT" &>/dev/null; then
        log INFO "LM Studio is available (PRIMARY)"
    else
        log WARN "LM Studio not available, will rely on Ollama"
    fi

    # Start Ollama
    log INFO "Starting Ollama..."
    if systemctl is-active --quiet bizra-inference 2>/dev/null; then
        log INFO "Ollama already running"
    else
        sudo systemctl start bizra-inference 2>/dev/null || (ollama serve &>/dev/null &)
    fi
    wait_for_service "Ollama" "$OLLAMA_ENDPOINT" 60

    log INFO "Inference layer started"
}

start_application() {
    header "Starting Application Services"

    # API Server
    log INFO "Starting API Server..."
    if systemctl is-active --quiet bizra-api 2>/dev/null; then
        log INFO "API Server already running"
    else
        sudo systemctl start bizra-api 2>/dev/null || true
    fi
    wait_for_service "API Server" "$API_ENDPOINT" 60

    # Dashboard
    log INFO "Starting Dashboard..."
    if systemctl is-active --quiet bizra-dashboard 2>/dev/null; then
        log INFO "Dashboard already running"
    else
        sudo systemctl start bizra-dashboard 2>/dev/null || true
    fi
    wait_for_service "Dashboard" "$DASHBOARD_ENDPOINT" 90

    log INFO "Application services started"
}

start_sovereign() {
    header "Starting Sovereign Engine"

    log INFO "Starting Sovereign Engine..."
    if systemctl is-active --quiet bizra-sovereign 2>/dev/null; then
        log INFO "Sovereign Engine already running"
    else
        sudo systemctl start bizra-sovereign 2>/dev/null || true
    fi
    wait_for_service "Sovereign Engine" "$SOVEREIGN_ENDPOINT" 120

    log INFO "Sovereign Engine started"
}

# ==============================================================================
# HEALTH VALIDATION
# ==============================================================================

validate_health() {
    header "Validating System Health"

    local all_healthy=true

    # Check each service
    log INFO "Checking service health..."

    # API Server
    if curl -sf "$API_ENDPOINT" &>/dev/null; then
        log INFO "API Server: HEALTHY"
    else
        log ERROR "API Server: UNHEALTHY"
        all_healthy=false
    fi

    # Dashboard
    if curl -sf "$DASHBOARD_ENDPOINT" &>/dev/null; then
        log INFO "Dashboard: HEALTHY"
    else
        log ERROR "Dashboard: UNHEALTHY"
        all_healthy=false
    fi

    # Inference (at least one backend)
    local inference_ok=false
    if curl -sf "$LMSTUDIO_ENDPOINT" &>/dev/null; then
        log INFO "LM Studio: AVAILABLE"
        inference_ok=true
    else
        log WARN "LM Studio: UNAVAILABLE"
    fi

    if curl -sf "$OLLAMA_ENDPOINT" &>/dev/null; then
        log INFO "Ollama: AVAILABLE"
        inference_ok=true
    else
        log WARN "Ollama: UNAVAILABLE"
    fi

    if [[ "$inference_ok" == "false" ]]; then
        log ERROR "No inference backend available!"
        all_healthy=false
    fi

    # Sovereign Engine
    if curl -sf "$SOVEREIGN_ENDPOINT" &>/dev/null; then
        local health_response
        health_response=$(curl -sf "$SOVEREIGN_ENDPOINT" 2>/dev/null || echo '{}')

        # Check Ihsan score if available
        local ihsan_score
        ihsan_score=$(echo "$health_response" | grep -oP '"ihsan_score":\s*\K[0-9.]+' 2>/dev/null || echo "0.95")

        if (( $(echo "$ihsan_score >= $IHSAN_THRESHOLD" | bc -l) )); then
            log INFO "Sovereign Engine: HEALTHY (Ihsan: $ihsan_score)"
        else
            log WARN "Sovereign Engine: DEGRADED (Ihsan: $ihsan_score < $IHSAN_THRESHOLD)"
        fi
    else
        log ERROR "Sovereign Engine: UNHEALTHY"
        all_healthy=false
    fi

    if [[ "$all_healthy" == "true" ]]; then
        log INFO "All health checks passed"
        return 0
    else
        log ERROR "Some health checks failed"
        return 1
    fi
}

# ==============================================================================
# STATUS REPORT
# ==============================================================================

print_status() {
    header "BIZRA Node0 Status"

    echo -e "${CYAN}System Information:${NC}"
    echo "  Node ID:     node0-genesis"
    echo "  Environment: production"
    echo "  Timestamp:   $(date -Iseconds)"
    echo ""

    echo -e "${CYAN}Service Status:${NC}"
    printf "  %-20s %s\n" "PostgreSQL:" "$(systemctl is-active postgresql 2>/dev/null || echo 'unknown')"
    printf "  %-20s %s\n" "Redis:" "$(systemctl is-active redis 2>/dev/null || echo 'unknown')"
    printf "  %-20s %s\n" "Inference:" "$(systemctl is-active bizra-inference 2>/dev/null || echo 'unknown')"
    printf "  %-20s %s\n" "API Server:" "$(systemctl is-active bizra-api 2>/dev/null || echo 'unknown')"
    printf "  %-20s %s\n" "Dashboard:" "$(systemctl is-active bizra-dashboard 2>/dev/null || echo 'unknown')"
    printf "  %-20s %s\n" "Sovereign:" "$(systemctl is-active bizra-sovereign 2>/dev/null || echo 'unknown')"
    echo ""

    echo -e "${CYAN}Endpoints:${NC}"
    echo "  API Server:   http://localhost:3001"
    echo "  Dashboard:    http://localhost:5173"
    echo "  Sovereign:    http://localhost:8080"
    echo "  LM Studio:    http://192.168.56.1:1234"
    echo "  Ollama:       http://localhost:11434"
    echo "  Metrics:      http://localhost:9090"
    echo ""

    echo -e "${CYAN}Quality Constraints:${NC}"
    echo "  Ihsan Threshold: >= $IHSAN_THRESHOLD"
    echo "  SNR Threshold:   >= 0.85"
    echo ""
}

# ==============================================================================
# MAIN
# ==============================================================================

main() {
    header "BIZRA Node0 Startup"

    echo "Genesis Block - Flagship Development Environment"
    echo "Constitutional Constraint: Ihsan >= $IHSAN_THRESHOLD"
    echo ""

    # Create required directories
    mkdir -p /var/log/bizra /var/run/bizra 2>/dev/null || true

    # Hardware validation
    header "Hardware Validation"

    local hw_ok=true
    check_cpu || hw_ok=false
    check_memory || hw_ok=false
    check_gpu || hw_ok=false
    check_disk || hw_ok=false
    check_network || hw_ok=false

    if [[ "$hw_ok" == "false" ]]; then
        log ERROR "Hardware requirements not met"
        exit 1
    fi

    log INFO "All hardware requirements met"

    # If check-only mode, exit here
    if [[ "$CHECK_ONLY" == "true" ]]; then
        log INFO "Check-only mode: Exiting without starting services"
        exit 0
    fi

    # Start services in order
    start_infrastructure
    start_inference
    start_application
    start_sovereign

    # Validate health
    if ! validate_health; then
        log ERROR "Health validation failed"
        exit 3
    fi

    # Print final status
    print_status

    log INFO "BIZRA Node0 startup complete"
    echo ""
    echo -e "${GREEN}BIZRA Node0 is ready.${NC}"
    echo "Dashboard: http://localhost:5173"
    echo ""
}

# Run main
main "$@"
