# BIZRA-DATA-LAKE Docker Image
# Version: 1.0.0 | Phase 11 - Public Launch
# Multi-modal memory system with vector embeddings
#
# Ihsan >= 0.95 | SNR >= 0.99 | Fail-Closed Enforcement

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    libpq-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash bizra \
    && chown -R bizra:bizra /app

# Copy requirements first for caching
COPY --chown=bizra:bizra requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for the unified system
RUN pip3 install --no-cache-dir \
    fastapi[all]==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    chromadb==0.4.22 \
    sentence-transformers==2.2.2 \
    transformers==4.36.2 \
    torch==2.1.2 \
    torchvision==0.16.2 \
    pillow==10.2.0 \
    openai-whisper==20231117 \
    unstructured[all-docs]==0.12.0 \
    python-magic==0.4.27 \
    pyarrow==15.0.0 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    faiss-cpu==1.7.4 \
    pydantic==2.5.3 \
    httpx==0.26.0 \
    prometheus-client==0.19.0

# Copy application code
COPY --chown=bizra:bizra . .

# Create data directories
RUN mkdir -p 00_INTAKE 01_RAW 02_PROCESSED 03_INDEXED 04_GOLD 99_QUARANTINE \
    && chown -R bizra:bizra /app

# Switch to non-root user
USER bizra

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SNR_THRESHOLD=0.99 \
    IHSAN_THRESHOLD=0.95 \
    BATCH_SIZE=128 \
    MAX_SEQ_LENGTH=512

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Expose port
EXPOSE 8000

# Entry point
CMD ["python", "-m", "uvicorn", "bizra_api:app", "--host", "0.0.0.0", "--port", "8000"]
