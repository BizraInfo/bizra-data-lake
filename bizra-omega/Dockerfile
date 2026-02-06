# BIZRA Sovereign Node — Multi-stage Docker Build
# Supports CPU and CUDA GPU acceleration
#
# Build:
#   docker build -t bizra:latest .
#   docker build -t bizra:cuda --build-arg CUDA=1 .
#
# Run:
#   docker run -p 3001:3001 bizra:latest
#   docker run --gpus all -p 3001:3001 bizra:cuda

ARG CUDA=0

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Build Environment
# ═══════════════════════════════════════════════════════════════════════════
FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
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

# Create dummy source files for dependency compilation
RUN mkdir -p bizra-core/src bizra-inference/src bizra-federation/src \
    bizra-autopoiesis/src bizra-api/src bizra-installer/src \
    bizra-python/src bizra-tests/src bizra-tests/tests bizra-tests/benches \
    bizra-hunter/src bizra-telescript/src bizra-proofspace/src \
    bizra-resourcepool/src bizra-resourcepool/src/bin \
    && echo "pub fn main() {}" > bizra-core/src/lib.rs \
    && echo "pub fn main() {}" > bizra-inference/src/lib.rs \
    && echo "pub fn main() {}" > bizra-federation/src/lib.rs \
    && echo "pub fn main() {}" > bizra-autopoiesis/src/lib.rs \
    && echo "pub fn main() {}" > bizra-api/src/lib.rs \
    && echo "fn main() {}" > bizra-api/src/main.rs \
    && echo "fn main() {}" > bizra-installer/src/main.rs \
    && echo "pub fn main() {}" > bizra-python/src/lib.rs \
    && echo "fn main() {}" > bizra-tests/benches/performance.rs \
    && echo "pub fn main() {}" > bizra-hunter/src/lib.rs \
    && echo "pub fn main() {}" > bizra-telescript/src/lib.rs \
    && echo "pub fn main() {}" > bizra-proofspace/src/lib.rs \
    && echo "pub fn main() {}" > bizra-resourcepool/src/lib.rs \
    && echo "fn main() {}" > bizra-resourcepool/src/main.rs \
    && echo "fn main() {}" > bizra-resourcepool/src/bin/node0_genesis.rs

# Build dependencies only (cached layer)
RUN cargo build --release || true

# Copy actual source code
COPY bizra-core/src bizra-core/src/
COPY bizra-inference/src bizra-inference/src/
COPY bizra-federation/src bizra-federation/src/
COPY bizra-autopoiesis/src bizra-autopoiesis/src/
COPY bizra-api/src bizra-api/src/
COPY bizra-installer/src bizra-installer/src/
COPY bizra-hunter/src bizra-hunter/src/
COPY bizra-telescript/src bizra-telescript/src/
COPY bizra-proofspace/src bizra-proofspace/src/
COPY bizra-resourcepool/src bizra-resourcepool/src/

# Touch to invalidate cache for source files
RUN touch bizra-core/src/lib.rs bizra-inference/src/lib.rs \
    bizra-federation/src/lib.rs bizra-autopoiesis/src/lib.rs \
    bizra-api/src/lib.rs bizra-api/src/main.rs bizra-installer/src/main.rs \
    bizra-telescript/src/lib.rs bizra-proofspace/src/lib.rs \
    bizra-resourcepool/src/lib.rs bizra-resourcepool/src/main.rs

# Build release binaries (including node0-genesis for Genesis ceremony)
RUN cargo build --release -p bizra-api -p bizra-installer -p bizra-resourcepool

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime Environment (CPU)
# ═══════════════════════════════════════════════════════════════════════════
FROM debian:bookworm-slim AS runtime-cpu

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash bizra
USER bizra
WORKDIR /home/bizra

# Copy binaries
COPY --from=builder /app/target/release/bizra-api /usr/local/bin/
COPY --from=builder /app/target/release/bizra /usr/local/bin/
COPY --from=builder /app/target/release/node0-genesis /usr/local/bin/
COPY --from=builder /app/target/release/resourcepool-node /usr/local/bin/

# Create data directories
RUN mkdir -p ~/.bizra ~/.bizra/sovereign_state ~/.bizra/pat ~/.bizra/sat

# Expose ports
EXPOSE 3001 7946/udp 8946/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/api/v1/health || exit 1

# Default command
CMD ["bizra-api", "--host", "0.0.0.0", "--port", "3001"]

# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Runtime Environment (CUDA)
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime-cuda

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash bizra
USER bizra
WORKDIR /home/bizra

# Copy binaries
COPY --from=builder /app/target/release/bizra-api /usr/local/bin/
COPY --from=builder /app/target/release/bizra /usr/local/bin/
COPY --from=builder /app/target/release/node0-genesis /usr/local/bin/
COPY --from=builder /app/target/release/resourcepool-node /usr/local/bin/

# Create data directories
RUN mkdir -p ~/.bizra ~/.bizra/sovereign_state ~/.bizra/pat ~/.bizra/sat

# Environment for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose ports
EXPOSE 3001 7946/udp 8946/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/api/v1/health || exit 1

# Default command
CMD ["bizra-api", "--host", "0.0.0.0", "--port", "3001"]

# ═══════════════════════════════════════════════════════════════════════════
# Final stage selection based on CUDA arg
# ═══════════════════════════════════════════════════════════════════════════
FROM runtime-cpu AS final-cpu
FROM runtime-cuda AS final-cuda

# Default to CPU build
FROM final-cpu
