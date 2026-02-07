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

# Create isolated virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[full]"

# =============================================================================
# Stage 2: Runtime — minimal production image with CUDA support
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    curl \
    libpq5 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

LABEL org.opencontainers.image.title="BIZRA Data Lake"
LABEL org.opencontainers.image.description="Multi-modal memory system with vector embeddings"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.vendor="BIZRA"
LABEL org.opencontainers.image.source="https://github.com/BizraInfo/bizra-data-lake"

# Create non-root user
RUN useradd --create-home --shell /bin/bash bizra

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=bizra:bizra core/ core/
COPY --chown=bizra:bizra pyproject.toml ./

# Create data directories
RUN mkdir -p 00_INTAKE 01_RAW 02_PROCESSED 03_INDEXED 04_GOLD 99_QUARANTINE \
    && chown -R bizra:bizra /app

# Switch to non-root user
USER bizra

# Environment variables — no secrets, only operational config
ENV PYTHONUNBUFFERED=1 \
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

# Entry point — sovereign REPL or API server
CMD ["python", "-m", "core.sovereign"]
