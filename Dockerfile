# BIZRA-DATA-LAKE Docker Image
# Version: 1.1.0 | Phase 11 - Public Launch
# Multi-modal memory system with vector embeddings
#
# Build context: repository root (BIZRA-DATA-LAKE/)
# Usage:  docker build -t bizra-data-lake .
#
# Ihsan >= 0.95 | SNR >= 0.85 | Fail-Closed Enforcement

# =============================================================================
# Stage 1: Builder — install Python dependencies into isolated venv
# =============================================================================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy package definition first (layer cache optimization)
COPY pyproject.toml ./
COPY core/ core/

# Create isolated virtual environment and install dependencies in one RUN
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir ".[full]"

# =============================================================================
# Stage 2: Runtime — minimal production image
# =============================================================================
# Genesis Boot: python:3.12-slim for fast, reliable builds.
# CUDA variant available via Dockerfile.gpu when GPU inference needed.
FROM python:3.12-slim-bookworm AS runtime

# Install runtime dependencies only and cleanup in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="BIZRA Data Lake"
LABEL org.opencontainers.image.description="Multi-modal memory system with vector embeddings"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.vendor="BIZRA"
LABEL org.opencontainers.image.source="https://github.com/BizraInfo/bizra-data-lake"

# Create non-root user (nologin shell for security)
RUN useradd --create-home --shell /usr/sbin/nologin bizra

WORKDIR /app

# Copy virtual environment from builder with correct ownership
COPY --from=builder --chown=bizra:bizra /opt/venv /opt/venv

# Copy application code with correct ownership
COPY --chown=bizra:bizra core/ core/
COPY --chown=bizra:bizra pyproject.toml ./

# Copy static data assets (Sci-Reasoning patterns for RDVE)
COPY --chown=bizra:bizra data/sci_reasoning/ data/sci_reasoning/

# Create data directories and set permissions in single RUN
RUN mkdir -p 00_INTAKE 01_RAW 02_PROCESSED 03_INDEXED 04_GOLD 99_QUARANTINE && \
    chown -R bizra:bizra /app

# Switch to non-root user
USER bizra

# Environment variables — no secrets, only operational config
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BIZRA_ENV=production \
    SNR_THRESHOLD=0.85 \
    IHSAN_THRESHOLD=0.95 \
    BATCH_SIZE=128 \
    MAX_SEQ_LENGTH=512

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from core.sovereign import __main__; print('healthy')" || exit 1

# Expose port
EXPOSE 8000

# Entry point — sovereign API server (REPL via: docker exec -it <name> python -m core.sovereign)
CMD ["python", "-m", "core.sovereign", "serve", "--host", "0.0.0.0", "--port", "8000"]
