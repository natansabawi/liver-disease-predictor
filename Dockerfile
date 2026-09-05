# LiverAI API -- containerized FastAPI backend
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching -- only reinstalls when
# requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" "pydantic>=2"

# App code + the trained model artifacts it needs at runtime
COPY app.py .
COPY outputs_ensemble/ ./outputs_ensemble/

EXPOSE 8000

# Basic container healthcheck hitting the app's own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
