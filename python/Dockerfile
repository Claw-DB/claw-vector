FROM python:3.11-slim

LABEL org.opencontainers.image.title="claw-vector-svc" \
      org.opencontainers.image.description="Embedding microservice for ClawDB" \
      org.opencontainers.image.source="https://github.com/Claw-DB/claw-vector"

# System dependencies required by sentence-transformers / onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for better Docker layer caching.
COPY pyproject.toml ./

# Install CPU-only torch then remaining dependencies.
RUN pip install --no-cache-dir "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Copy application source.
COPY python/ ./

# Pre-download the default model so the container starts immediately.
ARG MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Ports: 8000 = FastAPI REST, 50051 = gRPC, 9090 = Prometheus metrics.
EXPOSE 8000 50051 9090

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "claw_vector_svc.main"]
