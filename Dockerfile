# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BIZRA_KERNEL_HOST=0.0.0.0 \
    BIZRA_KERNEL_PORT=8000 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up Python alias
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy requirements and install
COPY requirements-kernel.txt /app/requirements-kernel.txt
# Install PyTorch with CUDA support explicitly first for stability
RUN python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN python -m pip install --no-cache-dir -r /app/requirements-kernel.txt

COPY tools /app/tools
COPY constellation /app/constellation
COPY core /app/core
COPY constitution /app/constitution
COPY model-family-genesis-v1-SEALED.yaml /app/model-family-genesis-v1-SEALED.yaml

# Copy Redis CA certificate for TLS validation (C2 optimization)
RUN mkdir -p /etc/redis/certs
COPY config/redis/ca-cert.pem /etc/redis/certs/ca-cert.pem
RUN chmod 644 /etc/redis/certs/ca-cert.pem

RUN mkdir -p /app/docs/evidence/receipts

EXPOSE 8000

CMD ["python", "-m", "core.main"]
