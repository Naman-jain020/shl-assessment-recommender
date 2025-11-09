# ---- Base image ----
    FROM python:3.11-slim

    # Avoid interactive tzdata etc.
    ENV DEBIAN_FRONTEND=noninteractive \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        HF_HOME=/root/.cache/huggingface \
        TRANSFORMERS_CACHE=/root/.cache/huggingface \
        SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
    
    WORKDIR /app
    
    # ---- System deps (for tokenizers / numpy) ----
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl && \
        rm -rf /var/lib/apt/lists/*
    
    # ---- Python deps ----
    # If you keep a single requirements.txt for both FE/BE, that’s fine.
    # Just ensure these are included (pin CPU-friendly wheels).
    COPY requirements.txt /app/requirements.txt
    
    # Prefer prebuilt wheels; install CPU PyTorch
    RUN pip install --upgrade pip && \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu \
            torch==2.3.1 torchvision==0.18.1 && \
        pip install -r /app/requirements.txt
    
    # ---- Copy code ----
    # Your repo should contain the backend folder with main.py and recommender/*
    COPY backend/ /app/backend/
    # Also copy your artifacts (embeddings, bm25 pack, ltr model, catalog) if you ship them
    # under backend/data or backend/models; keep the same structure you used locally.
    # Example:
    # COPY backend/data/ /app/backend/data/
    # COPY backend/models/ /app/backend/models/
    
    # Expose the port Spaces expects
    EXPOSE 7860
    
    # Start FastAPI (Spaces expects the app on 0.0.0.0:7860)
    CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]