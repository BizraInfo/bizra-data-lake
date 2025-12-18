FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BIZRA_KERNEL_HOST=0.0.0.0 \
    BIZRA_KERNEL_PORT=8000

WORKDIR /app

COPY requirements-kernel.txt /app/requirements-kernel.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-kernel.txt

COPY core /app/core
COPY constitution /app/constitution
COPY model-family-genesis-v1-SEALED.yaml /app/model-family-genesis-v1-SEALED.yaml

RUN mkdir -p /app/docs/evidence/receipts

EXPOSE 8000

CMD ["python", "-m", "core.main"]
