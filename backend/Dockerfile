# Build context is the repo root (multi-user-doc-rag/), not backend/, because
# the app is imported as the `backend.src.*` package (see SKILL.md's "Running
# the app" note on why PYTHONPATH also needs backend/ on it directly).
#
# This image is for the FastAPI server only. It deliberately does NOT install
# docling: parser.py (the only docling caller) runs offline via
# Dockerfile.ingest, and query-time embedding goes through the OpenRouter HTTP
# API, not a local model -- so torch/torchvision (and the CUDA stack they'd
# drag in on Linux) have no reason to be in the serving image.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY data/ data/

ENV PYTHONPATH=/app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
