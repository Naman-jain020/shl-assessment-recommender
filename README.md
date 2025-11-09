# SHL Assessment Recommender

- **Backend**: FastAPI (`/health`, `/recommend`)
- **Frontend**: Streamlit
- **Crawler**: `crawler/catalog_crawler.py`
- **Training**: `training/kaggle_training.ipynb`

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Backend
uvicorn backend.main:app --reload

# Frontend
BACKEND_URL=http://127.0.0.1:8000 streamlit run frontend/app.py
```
