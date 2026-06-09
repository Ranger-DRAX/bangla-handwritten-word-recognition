# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Bangla OCR Streamlit App
# Base: python:3.10-slim   |   Port: 8501
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# 1. System deps for OpenCV headless & font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    fonts-noto \
    fonts-noto-extra \
    && rm -rf /var/lib/apt/lists/*

# 2. Working directory
WORKDIR /app
ENV TF_CPP_MIN_LOG_LEVEL=3
# 3. Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app source (NOT the training dataset)
COPY app.py        .
COPY labels.json   .
COPY models/       models/

# 5. Streamlit configuration — disable browser auto-open, enable CORS
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.headless=true", \
    "--server.address=0.0.0.0", \
    "--server.port=8501"]
