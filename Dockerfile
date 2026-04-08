# ============================================================
# AI-Powered Last Mile Delivery Automation — Production Image
# Multi-stage build: compile native deps once, ship a slim runtime.
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Build tools for native extensions (faiss-cpu, scipy, scikit-learn)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies into an isolated prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install the project package (standard install, NOT editable)
COPY setup.py README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Security: non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy only the compiled Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code and assets
COPY api.py app.py main.py ./
COPY templates/ templates/
COPY data/external/ data/external/

# Copy secrets-free config (API keys come from env vars)
COPY src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml \
     src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml

# Ensure the logs directory is writable by the non-root user
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

# ── Environment ─────────────────────────────────────────────
# Production mode: secrets from env vars, no .env loading
ENV ENV=production \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    PROJECT_ROOT=/app \
    CONFIG_PATH=/app/src/AI_Powered_Last_Mile_Delivery_Automation/config/config.yaml

# Switch to non-root user
USER appuser

EXPOSE 8080

# Health check against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
