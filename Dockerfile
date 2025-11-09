# Use Python 3.11 (safe for tokenizers & torch wheels)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System dependencies for LightGBM, BS4, lxml, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git wget curl \
    libgomp1 libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install CPU torch first
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1 \
 && pip install -r requirements.txt

COPY . /app

ENV PORT=8000

# ✅ Correct CMD for Railway/Docker
CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"