FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY proto ./proto
COPY claw_vector_svc ./claw_vector_svc

RUN pip install --upgrade pip \
    && pip install --no-cache-dir "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install '.[inference]'

ARG MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
RUN PYTHONPATH=/install/lib/python3.11/site-packages python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1001 --home-dir /root --no-create-home app \
    && mkdir -p /root/.cache/huggingface /app \
    && chmod 755 /root \
    && chown -R 1001:1001 /root/.cache/huggingface /app

COPY --from=builder /install /usr/local
COPY claw_vector_svc ./claw_vector_svc
COPY proto ./proto

ENV PYTHONUNBUFFERED=1
ENV HTTP_PORT=8080
ENV GRPC_PORT=50051

USER 1001

EXPOSE 50051 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "claw_vector_svc.main:app", "--host", "0.0.0.0", "--port", "8080"]
