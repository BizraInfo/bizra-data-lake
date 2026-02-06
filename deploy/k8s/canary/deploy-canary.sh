#!/bin/bash
# BIZRA Elite Canary Deployment Script
# Usage: ./deploy-canary.sh [build|deploy|monitor|all]
#
# Standing on Giants: Shannon | Lamport | Kubernetes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."
NAMESPACE="${BIZRA_NAMESPACE:-bizra}"
CANARY_VERSION="${CANARY_VERSION:-v1.0.2-canary}"
CLUSTER_NAME="${K3D_CLUSTER:-bizra-prod}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

check_prerequisites() {
    log_step "Checking prerequisites..."

    local missing=()

    command -v kubectl &>/dev/null || missing+=("kubectl")
    command -v docker &>/dev/null || missing+=("docker")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    # Check cluster access
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_warn "Namespace '$NAMESPACE' does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi

    log_info "Prerequisites OK"
}

build_canary_image() {
    log_step "Building canary Docker image..."

    local dockerfile="${PROJECT_ROOT}/deploy/Dockerfile.unified"

    if [[ ! -f "$dockerfile" ]]; then
        dockerfile="${PROJECT_ROOT}/deploy/Dockerfile.elite"
    fi

    if [[ ! -f "$dockerfile" ]]; then
        log_error "No Dockerfile found in deploy/"
        exit 1
    fi

    log_info "Using Dockerfile: $dockerfile"
    log_info "Building image: bizra-elite:${CANARY_VERSION}"

    docker build \
        -f "$dockerfile" \
        -t "bizra-elite:${CANARY_VERSION}" \
        --build-arg VERSION="${CANARY_VERSION}" \
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$PROJECT_ROOT"

    # Import to K3d if using K3d
    if command -v k3d &>/dev/null; then
        log_info "Importing image to K3d cluster: $CLUSTER_NAME"
        k3d image import "bizra-elite:${CANARY_VERSION}" -c "$CLUSTER_NAME" || {
            log_warn "K3d import failed, image may already exist"
        }
    fi

    log_info "Image built successfully"
}

deploy_canary() {
    log_step "Deploying canary to Kubernetes..."

    # Update image tag in deployment
    local temp_manifest=$(mktemp)
    sed "s|bizra-elite:v1.0.2-canary|bizra-elite:${CANARY_VERSION}|g" \
        "${SCRIPT_DIR}/canary-deployment.yaml" > "$temp_manifest"

    # Apply the manifest
    log_info "Applying canary deployment..."
    kubectl apply -f "$temp_manifest"
    rm -f "$temp_manifest"

    # Wait for canary pod to be ready
    log_info "Waiting for canary pod to be ready..."
    kubectl rollout status deployment/bizra-elite-canary -n "$NAMESPACE" --timeout=120s || {
        log_error "Canary deployment failed to become ready"
        log_info "Checking pod status..."
        kubectl get pods -n "$NAMESPACE" -l track=canary
        kubectl describe pods -n "$NAMESPACE" -l track=canary | tail -50
        exit 1
    }

    log_info "Canary deployed successfully"
}

monitor_canary() {
    log_step "Monitoring canary deployment..."

    echo ""
    log_info "=== Pod Status ==="
    kubectl get pods -n "$NAMESPACE" -l app=bizra-elite -o wide

    echo ""
    log_info "=== Canary Pod Logs (last 20 lines) ==="
    kubectl logs -n "$NAMESPACE" -l track=canary --tail=20 2>/dev/null || echo "No logs yet"

    echo ""
    log_info "=== Traffic Split Status ==="
    local canary_weight=$(kubectl get ingress bizra-elite-ingress-canary -n "$NAMESPACE" \
        -o jsonpath='{.metadata.annotations.nginx\.ingress\.kubernetes\.io/canary-weight}' 2>/dev/null || echo "N/A")
    echo "Canary Weight: ${canary_weight}%"
    echo "Stable Weight: $((100 - ${canary_weight:-0}))%"

    echo ""
    log_info "=== Health Check ==="
    local canary_pod=$(kubectl get pods -n "$NAMESPACE" -l track=canary -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [[ -n "$canary_pod" ]]; then
        kubectl exec -n "$NAMESPACE" "$canary_pod" -- curl -s http://localhost:8000/v1/health 2>/dev/null || echo "Health check not available"
    fi
}

increase_traffic() {
    local new_weight="${1:-25}"
    log_step "Increasing canary traffic to ${new_weight}%..."

    kubectl annotate ingress bizra-elite-ingress-canary -n "$NAMESPACE" \
        nginx.ingress.kubernetes.io/canary-weight="${new_weight}" --overwrite

    log_info "Canary traffic increased to ${new_weight}%"
    log_info "Monitor with: ./deploy-canary.sh monitor"
}

show_usage() {
    echo "BIZRA Elite Canary Deployment Script"
    echo ""
    echo "Usage: $0 [command] [args]"
    echo ""
    echo "Commands:"
    echo "  build        - Build canary Docker image"
    echo "  deploy       - Deploy canary to Kubernetes"
    echo "  monitor      - Show canary status and logs"
    echo "  traffic N    - Set canary traffic to N% (e.g., traffic 25)"
    echo "  all          - Build and deploy canary"
    echo "  help         - Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  CANARY_VERSION    - Image tag (default: v1.0.2-canary)"
    echo "  BIZRA_NAMESPACE   - K8s namespace (default: bizra)"
    echo "  K3D_CLUSTER       - K3d cluster name (default: bizra-prod)"
    echo ""
    echo "Examples:"
    echo "  $0 all                          # Build and deploy"
    echo "  CANARY_VERSION=v1.1.0 $0 build  # Build specific version"
    echo "  $0 traffic 50                   # Increase to 50%"
    echo "  $0 monitor                      # Check status"
}

# Main
case "${1:-help}" in
    build)
        check_prerequisites
        build_canary_image
        ;;
    deploy)
        check_prerequisites
        deploy_canary
        ;;
    monitor|status)
        monitor_canary
        ;;
    traffic)
        increase_traffic "${2:-25}"
        ;;
    all)
        check_prerequisites
        build_canary_image
        deploy_canary
        monitor_canary
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
