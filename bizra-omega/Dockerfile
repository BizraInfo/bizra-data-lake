# BIZRA Sovereign Node — Multi-stage Docker Build
# Version: 1.1.0 | Supports CPU and CUDA GPU acceleration
#
# Build context: bizra-omega/ directory
# CI usage:  docker build -f bizra-omega/Dockerfile -t bizra-omega bizra-omega/
#
# Build variants:
#   docker build -t bizra:latest .                    # CPU (default)
#   docker build -t bizra:cuda --build-arg CUDA=1 .   # CUDA GPU
#
# Run:
#   docker run -p 3001:3001 bizra:latest
#   docker run --gpus all -p 3001:3001 bizra:cuda
#
# Standing on Giants: Rust, Docker BuildKit, Ed25519-dalek

# =============================================================================
# Stage 1: Build Environment
# =============================================================================
ARG RUST_VERSION=1.88
FROM rust:${RUST_VERSION}-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --------------------------------------------------------------------------
# Copy workspace Cargo manifests first (layer cache: deps rebuild only when
# Cargo.toml or Cargo.lock changes)
# --------------------------------------------------------------------------
COPY Cargo.toml Cargo.lock ./

# All 13 workspace members — must match [workspace] members in Cargo.toml
COPY bizra-core/Cargo.toml bizra-core/
COPY bizra-inference/Cargo.toml bizra-inference/
COPY bizra-federation/Cargo.toml bizra-federation/
COPY bizra-autopoiesis/Cargo.toml bizra-autopoiesis/
COPY bizra-api/Cargo.toml bizra-api/
COPY bizra-installer/Cargo.toml bizra-installer/
COPY bizra-python/Cargo.toml bizra-python/
COPY bizra-tests/Cargo.toml bizra-tests/
COPY bizra-hunter/Cargo.toml bizra-hunter/
COPY bizra-telescript/Cargo.toml bizra-telescript/
COPY bizra-proofspace/Cargo.toml bizra-proofspace/
COPY bizra-resourcepool/Cargo.toml bizra-resourcepool/
COPY bizra-cli/Cargo.toml bizra-cli/

# --------------------------------------------------------------------------
# Create dummy source files so cargo can resolve the dependency graph and
# pre-compile all external crates (cached layer).
#
# Notes:
# - bizra-tests has NO src/ dir (only tests/ and benches/)
# - bizra-python is a cdylib (lib only, no main)
# - bizra-cli produces the "bizra" binary
# - bizra-resourcepool has two binaries + a lib
# --------------------------------------------------------------------------
RUN set -eux && \
    # Library crates (lib.rs only)
    for crate in bizra-core bizra-inference bizra-federation bizra-autopoiesis \
                 bizra-python bizra-hunter bizra-telescript bizra-proofspace; do \
        mkdir -p "$crate/src" && \
        echo "pub fn _dummy() {}" > "$crate/src/lib.rs"; \
    done && \
    # bizra-api: lib + binary
    mkdir -p bizra-api/src && \
    echo "pub fn _dummy() {}" > bizra-api/src/lib.rs && \
    echo "fn main() {}" > bizra-api/src/main.rs && \
    # bizra-installer: binary only (bizra-install)
    mkdir -p bizra-installer/src && \
    echo "fn main() {}" > bizra-installer/src/main.rs && \
    # bizra-cli: binary only (bizra)
    mkdir -p bizra-cli/src && \
    echo "fn main() {}" > bizra-cli/src/main.rs && \
    # bizra-resourcepool: lib + two binaries (resourcepool-node, node0-genesis)
    mkdir -p bizra-resourcepool/src/bin && \
    echo "pub fn _dummy() {}" > bizra-resourcepool/src/lib.rs && \
    echo "fn main() {}" > bizra-resourcepool/src/main.rs && \
    echo "fn main() {}" > bizra-resourcepool/src/bin/node0_genesis.rs && \
    # bizra-tests: no src/, only tests and benches
    mkdir -p bizra-tests/tests bizra-tests/benches && \
    echo "fn main() {}" > bizra-tests/benches/performance.rs && \
    echo "#[test] fn dummy() {}" > bizra-tests/tests/e2e.rs && \
    echo "#[test] fn dummy() {}" > bizra-tests/tests/integration.rs && \
    echo "#[test] fn dummy() {}" > bizra-tests/tests/got_properties.rs && \
    echo "#[test] fn dummy() {}" > bizra-tests/tests/federation_security.rs && \
    echo "#[test] fn dummy() {}" > bizra-tests/tests/inference_fallback.rs

# Pre-compile dependencies (this layer is cached until Cargo.toml changes)
RUN cargo build --release --workspace 2>&1 || true
# Note: The dummy sources may cause some workspace crates to fail to compile
# (missing actual types). That is expected — the goal is to cache *dependency*
# downloads and compilation. The real build below will succeed.

# --------------------------------------------------------------------------
# Copy actual source code (invalidates cache only when source changes)
# --------------------------------------------------------------------------
COPY bizra-core/src bizra-core/src/
COPY bizra-inference/src bizra-inference/src/
COPY bizra-federation/src bizra-federation/src/
COPY bizra-autopoiesis/src bizra-autopoiesis/src/
COPY bizra-api/src bizra-api/src/
COPY bizra-installer/src bizra-installer/src/
COPY bizra-python/src bizra-python/src/
COPY bizra-hunter/src bizra-hunter/src/
COPY bizra-cli/src bizra-cli/src/
COPY bizra-telescript/src bizra-telescript/src/
COPY bizra-proofspace/src bizra-proofspace/src/
COPY bizra-resourcepool/src bizra-resourcepool/src/
COPY bizra-tests/tests bizra-tests/tests/
COPY bizra-tests/benches bizra-tests/benches/

# Touch source files to ensure cargo detects changes from dummy sources
RUN find /app -name "*.rs" -newer /app/Cargo.lock -exec touch {} +

# Build release binaries
# - bizra-api:          REST/WebSocket API gateway
# - bizra-install:      Universal installer (from bizra-installer)
# - bizra:              CLI/TUI (from bizra-cli)
# - resourcepool-node:  Resource pool daemon (from bizra-resourcepool)
# - node0-genesis:      Genesis ceremony (from bizra-resourcepool)
RUN cargo build --release \
    -p bizra-api \
    -p bizra-installer \
    -p bizra-cli \
    -p bizra-resourcepool

# =============================================================================
# Stage 2: Runtime Environment (CPU)
# =============================================================================
FROM debian:bookworm-slim AS runtime-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash bizra

# Copy binaries (BEFORE switching to non-root user)
COPY --from=builder /app/target/release/bizra-api /usr/local/bin/
COPY --from=builder /app/target/release/bizra /usr/local/bin/
COPY --from=builder /app/target/release/bizra-install /usr/local/bin/
COPY --from=builder /app/target/release/node0-genesis /usr/local/bin/
COPY --from=builder /app/target/release/resourcepool-node /usr/local/bin/

# Switch to non-root user
USER bizra
WORKDIR /home/bizra

# Create data directories
RUN mkdir -p ~/.bizra ~/.bizra/sovereign_state ~/.bizra/pat ~/.bizra/sat

# Environment — no secrets
ENV BIZRA_ENV=production \
    RUST_LOG=info

# Expose ports: API, Gossip UDP, Gossip TCP
EXPOSE 3001 7946/udp 8946/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/api/v1/health || exit 1

# Default command
CMD ["bizra-api", "--host", "0.0.0.0", "--port", "3001"]

# =============================================================================
# Stage 3: Runtime Environment (CUDA)
# =============================================================================
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash bizra

# Copy binaries (BEFORE switching to non-root user)
COPY --from=builder /app/target/release/bizra-api /usr/local/bin/
COPY --from=builder /app/target/release/bizra /usr/local/bin/
COPY --from=builder /app/target/release/bizra-install /usr/local/bin/
COPY --from=builder /app/target/release/node0-genesis /usr/local/bin/
COPY --from=builder /app/target/release/resourcepool-node /usr/local/bin/

# Switch to non-root user
USER bizra
WORKDIR /home/bizra

# Create data directories
RUN mkdir -p ~/.bizra ~/.bizra/sovereign_state ~/.bizra/pat ~/.bizra/sat

# CUDA environment — no secrets
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    BIZRA_ENV=production \
    RUST_LOG=info

# Expose ports
EXPOSE 3001 7946/udp 8946/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/api/v1/health || exit 1

# Default command
CMD ["bizra-api", "--host", "0.0.0.0", "--port", "3001"]
