# Build context is the repo root (multi-user-doc-rag/), not backend/, because
# the app is imported as the `backend.src.*` package (see SKILL.md's "Running
# the app" note on why PYTHONPATH also needs backend/ on it directly).
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY data/ data/

ENV PYTHONPATH=/app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
