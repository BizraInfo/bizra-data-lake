#!/bin/bash
# BIZRA Elite Canary Rollback Script
# Usage: ./rollback.sh [immediate|gradual|verify]
#
# Standing on Giants: Shannon | Lamport | Kubernetes
set -euo pipefail

NAMESPACE="${BIZRA_NAMESPACE:-bizra}"
CANARY_DEPLOYMENT="bizra-elite-canary"
STABLE_DEPLOYMENT="bizra-elite-stable"
CANARY_INGRESS="bizra-elite-ingress-canary"
STABLE_INGRESS="bizra-elite-ingress-stable"
LOG_DIR="${BIZRA_LOG_DIR:-/tmp/bizra-rollback}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace '$NAMESPACE' does not exist."
        exit 1
    fi

    log_info "Prerequisites OK"
}

save_canary_logs() {
    log_info "Saving canary logs for analysis..."
    mkdir -p "$LOG_DIR"

    local timestamp=$(date +%Y%m%d-%H%M%S)
    local log_file="$LOG_DIR/canary-rollback-$timestamp.log"

    kubectl logs -n "$NAMESPACE" -l track=canary --tail=5000 > "$log_file" 2>&1 || true

    # Also save pod descriptions
    kubectl describe pods -n "$NAMESPACE" -l track=canary >> "$log_file" 2>&1 || true

    log_info "Logs saved to: $log_file"
}

verify_stable_health() {
    log_info "Verifying stable deployment health..."

    local stable_pods=$(kubectl get pods -n "$NAMESPACE" -l track=stable -o jsonpath='{.items[*].status.phase}')

    if [[ -z "$stable_pods" ]]; then
        log_error "No stable pods found!"
        return 1
    fi

    local running_count=$(echo "$stable_pods" | tr ' ' '\n' | grep -c "Running" || true)
    local total_count=$(echo "$stable_pods" | tr ' ' '\n' | wc -w)

    log_info "Stable pods: $running_count/$total_count running"

    if [[ "$running_count" -eq 0 ]]; then
        log_error "No stable pods are running!"
        return 1
    fi

    # Check endpoint health
    local endpoints=$(kubectl get endpoints "$STABLE_DEPLOYMENT" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || true)

    if [[ -z "$endpoints" ]]; then
        log_warn "No endpoints found for stable service"
    else
        log_info "Stable endpoints: $endpoints"
    fi

    return 0
}

rollback_immediate() {
    log_warn "EXECUTING IMMEDIATE ROLLBACK"
    log_info "This will remove all canary resources immediately"

    # Step 1: Remove canary from traffic
    log_info "Step 1: Removing canary ingress..."
    kubectl delete ingress "$CANARY_INGRESS" -n "$NAMESPACE" --ignore-not-found=true

    # Step 2: Scale down canary
    log_info "Step 2: Scaling down canary deployment..."
    kubectl scale deployment "$CANARY_DEPLOYMENT" -n "$NAMESPACE" --replicas=0 2>/dev/null || true

    # Step 3: Save logs before deletion
    save_canary_logs

    # Step 4: Delete canary deployment
    log_info "Step 3: Deleting canary deployment..."
    kubectl delete deployment "$CANARY_DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found=true

    # Step 5: Clean up canary config
    log_info "Step 4: Cleaning up canary configmap..."
    kubectl delete configmap bizra-canary-config -n "$NAMESPACE" --ignore-not-found=true

    # Step 6: Verify stable is healthy
    log_info "Step 5: Verifying stable deployment..."
    verify_stable_health

    log_info "IMMEDIATE ROLLBACK COMPLETE"
}

rollback_gradual() {
    log_info "EXECUTING GRADUAL ROLLBACK"

    # Step 1: Reduce traffic to 0%
    log_info "Step 1: Reducing canary traffic to 0%..."
    kubectl annotate ingress "$CANARY_INGRESS" -n "$NAMESPACE" \
        nginx.ingress.kubernetes.io/canary-weight="0" --overwrite 2>/dev/null || {
        log_warn "Canary ingress not found, skipping traffic reduction"
    }

    # Step 2: Wait for in-flight requests
    log_info "Step 2: Waiting 30 seconds for in-flight requests..."
    sleep 30

    # Step 3: Scale down canary
    log_info "Step 3: Scaling down canary to 0 replicas..."
    kubectl scale deployment "$CANARY_DEPLOYMENT" -n "$NAMESPACE" --replicas=0 2>/dev/null || {
        log_warn "Canary deployment not found"
    }

    # Step 4: Save logs
    save_canary_logs

    # Step 5: Verify stable health
    log_info "Step 4: Verifying stable deployment health..."
    verify_stable_health

    # Step 6: Prompt for cleanup
    log_info ""
    log_info "Canary has been scaled to 0 and removed from traffic."
    log_info "To complete cleanup, run: ./rollback.sh cleanup"

    log_info "GRADUAL ROLLBACK COMPLETE"
}

cleanup_canary() {
    log_info "CLEANING UP CANARY RESOURCES"

    kubectl delete ingress "$CANARY_INGRESS" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete deployment "$CANARY_DEPLOYMENT" -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete configmap bizra-canary-config -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete service bizra-elite-canary -n "$NAMESPACE" --ignore-not-found=true

    log_info "CLEANUP COMPLETE"
}

verify_status() {
    log_info "BIZRA Elite Deployment Status"
    echo ""

    log_info "=== Pods ==="
    kubectl get pods -n "$NAMESPACE" -l app=bizra-elite -o wide

    echo ""
    log_info "=== Deployments ==="
    kubectl get deployments -n "$NAMESPACE" -l app=bizra-elite

    echo ""
    log_info "=== Services ==="
    kubectl get services -n "$NAMESPACE" | grep bizra-elite

    echo ""
    log_info "=== Ingresses ==="
    kubectl get ingress -n "$NAMESPACE" | grep bizra-elite || echo "No ingresses found"

    echo ""
    log_info "=== Canary Weight (if present) ==="
    kubectl get ingress "$CANARY_INGRESS" -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.nginx\.ingress\.kubernetes\.io/canary-weight}' 2>/dev/null || echo "No canary ingress"
    echo ""
}

show_usage() {
    echo "BIZRA Elite Canary Rollback Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  immediate  - Emergency rollback: removes all canary resources immediately"
    echo "  gradual    - Graceful rollback: reduces traffic, then scales down"
    echo "  cleanup    - Remove canary resources after gradual rollback"
    echo "  verify     - Show current deployment status"
    echo "  help       - Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BIZRA_NAMESPACE  - Kubernetes namespace (default: bizra)"
    echo "  BIZRA_LOG_DIR    - Directory for rollback logs (default: /tmp/bizra-rollback)"
    echo ""
    echo "Examples:"
    echo "  $0 immediate           # Emergency rollback"
    echo "  $0 gradual             # Graceful rollback"
    echo "  BIZRA_NAMESPACE=bizra-staging $0 verify  # Check staging"
}

# Main
check_prerequisites

case "${1:-help}" in
    immediate)
        rollback_immediate
        ;;
    gradual)
        rollback_gradual
        ;;
    cleanup)
        cleanup_canary
        ;;
    verify|status)
        verify_status
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
