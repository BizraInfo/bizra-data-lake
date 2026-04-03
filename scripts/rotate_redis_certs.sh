#!/bin/bash
# scripts/rotate_redis_certs.sh - Redis TLS Certificate Rotation Script
# Standing on Shoulders of Giants Protocol: TLS 1.3, Redis TLS configuration
# Extends BIZRA Ihsān security dimensions (safety: 0.22, correctness: 0.22)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REDIS_CONFIG_DIR="$PROJECT_ROOT/config/redis"
CERT_DIR="$REDIS_CONFIG_DIR"
CERT_PASSWORD="${REDIS_CERT_PASSWORD:-changeme}"
DAYS_VALID=90
OPENSSL_CNF="$CERT_DIR/openssl.cnf"

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

check_openssl() {
    if ! command -v openssl &> /dev/null; then
        log_error "openssl not found. Please install OpenSSL."
        exit 1
    fi
}

check_rredis() {
    if ! command -v redis-cli &> /dev/null; then
        log_error "redis-cli not found. Skipping Redis reload."
        return 1
    fi
    return 0
}

generate_ca() {
    local ca_key="$CERT_DIR/ca-key.pem"
    local ca_cert="$CERT_DIR/ca-cert.pem"
    local ca_serial="$CERT_DIR/ca-cert.srl"

    log_info "Generating CA key and certificate..."

    if [[ -f "$ca_key" && -f "$ca_cert" ]]; then
        log_info "CA already exists, backing up..."
        cp "$ca_key" "$ca_key.$(date +%Y%m%d_%H%M%S).bak"
        cp "$ca_cert" "$ca_cert.$(date +%Y%m%d_%H%M%S).bak"
    fi

    openssl genrsa -aes256 -passout pass:$CERT_PASSWORD -out "$ca_key" 4096 2>/dev/null
    openssl req -new -x509 -days $DAYS_VALID -passin pass:$CERT_PASSWORD \
        -key "$ca_key" -out "$ca_cert" \
        -subj "/O=BIZRA/OU=Security/CN=BIZRA CA" \
        -config "$OPENSSL_CNF" 2>/dev/null

    echo "01" > "$ca_serial"

    log_info "CA certificate generated successfully."
}

generate_server_cert() {
    local server_key="$CERT_DIR/redis-server-key.pem"
    local server_csr="$CERT_DIR/redis-server.csr"
    local server_cert="$CERT_DIR/redis-server-cert.pem"

    log_info "Generating server key and certificate..."

    if [[ -f "$server_key" && -f "$server_cert" ]]; then
        log_info "Server cert already exists, backing up..."
        cp "$server_key" "$server_key.$(date +%Y%m%d_%H%M%S).bak"
        cp "$server_cert" "$server_cert.$(date +%Y%m%d_%H%M%S).bak"
        rm -f "$server_key.EXPIRED" "$server_cert.EXPIRED" 2>/dev/null || true
    fi

    openssl genrsa -out "$server_key" 2048 2>/dev/null

    openssl req -new -key "$server_key" -out "$server_csr" \
        -subj "/O=BIZRA/OU=Redis/CN=redis.bizra.local" \
        -config "$OPENSSL_CNF" 2>/dev/null

    openssl x509 -req -days $DAYS_VALID \
        -in "$server_csr" -CA "$ca_cert" -CAkey "$ca_key" \
        -passin pass:$CERT_PASSWORD \
        -CAcreateserial -out "$server_cert" \
        -extfile "$OPENSSL_CNF" -extensions server_ext 2>/dev/null

    rm -f "$server_csr"

    chmod 600 "$server_key"
    chmod 644 "$server_cert"

    log_info "Server certificate generated successfully."
}

verify_cert() {
    local cert_file="$1"
    local ca_file="$2"

    log_info "Verifying certificate..."

    if openssl verify -CAfile "$ca_file" "$cert_file" &>/dev/null; then
        log_info "Certificate verified successfully."
        return 0
    else
        log_error "Certificate verification failed."
        return 1
    fi
}

reload_redis() {
    local redis_host="${REDIS_HOST:-localhost}"
    local redis_port="${REDIS_PORT:-6379}"

    log_info "Reloading Redis TLS configuration..."

    if check_rredis; then
        redis-cli -h "$redis_host" -p "$redis_port" CONFIG SET tls-cert-file "$CERT_DIR/redis-server-cert.pem" 2>/dev/null || true
        redis-cli -h "$redis_host" -p "$redis_port" CONFIG SET tls-key-file "$CERT_DIR/redis-server-key.pem" 2>/dev/null || true
        redis-cli -h "$redis_host" -p "$redis_port" CONFIG REWRITE 2>/dev/null || true
        log_info "Redis configuration reloaded."
    else
        log_info "Redis reload skipped (redis-cli not available)."
    fi
}

backup_certs() {
    local backup_dir="$CERT_DIR/backups/$(date +%Y%m%d)"
    mkdir -p "$backup_dir"

    log_info "Backing up certificates..."

    cp "$CERT_DIR/ca-key.pem" "$backup_dir/" 2>/dev/null || true
    cp "$CERT_DIR/ca-cert.pem" "$backup_dir/" 2>/dev/null || true
    cp "$CERT_DIR/redis-server-key.pem" "$backup_dir/" 2>/dev/null || true
    cp "$CERT_DIR/redis-server-cert.pem" "$backup_dir/" 2>/dev/null || true

    find "$CERT_DIR/backups" -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

    log_info "Backup completed."
}

check_cert_expiry() {
    local cert_file="$1"
    local label="$2"

    if [[ ! -f "$cert_file" ]]; then
        log_error "$label certificate not found: $cert_file"
        return 1
    fi

    local expiry_date
    expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" 2>/dev/null | cut -d= -f2)
    
    if [[ -z "$expiry_date" ]]; then
        log_error "Could not read expiry date for $label"
        return 1
    fi

    local expiry_ts
    expiry_ts=$(date -d "$expiry_date" +%s 2>/dev/null) || expiry_ts=0
    local now_ts
    now_ts=$(date +%s)
    local days_remaining=$(( (expiry_ts - now_ts) / 86400 ))

    log_info "$label expires in $days_remaining days ($expiry_date)"

    if [[ $days_remaining -lt 30 ]]; then
        log_error "$label expires in less than 30 days! Rotation required."
        return 1
    fi

    return 0
}

main() {
    log_info "Starting Redis TLS certificate rotation..."

    check_openssl

    mkdir -p "$CERT_DIR"

    if [[ ! -f "$OPENSSL_CNF" ]]; then
        cat > "$OPENSSL_CNF" << 'EOF'
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn

[dn]
C = US
O = BIZRA
OU = Security

[server_ext]
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = redis.bizra.local
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF
        log_info "Created OpenSSL config."
    fi

    check_cert_expiry "$CERT_DIR/ca-cert.pem" "CA" || generate_ca
    check_cert_expiry "$CERT_DIR/redis-server-cert.pem" "Server" || generate_server_cert

    verify_cert "$CERT_DIR/redis-server-cert.pem" "$CERT_DIR/ca-cert.pem" || exit 1

    backup_certs
    reload_redis

    log_info "Redis TLS certificate rotation completed successfully."
}

main "$@"