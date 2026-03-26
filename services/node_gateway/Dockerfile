FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/services

WORKDIR /app/services/node_gateway

COPY .tmp_prod_artifacts_v2/services/_shared /app/services/_shared
COPY .tmp_prod_artifacts_v2/services/node_gateway /app/services/node_gateway
COPY core /app/core

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
