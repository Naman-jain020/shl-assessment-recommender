# Use Python 3.11 (safe for tokenizers & torch wheels)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System dependencies (required for lxml, LightGBM, bs4)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git wget curl \
    libgomp1 libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your dependency list
COPY requirements.txt /app/

# Install CPU PyTorch wheels first so installation is fast
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1 \
 && pip install -r requirements.txt

# Copy entire project
COPY . /app

# Railway will pass PORT automatically
ENV PORT=8000

# Start FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]