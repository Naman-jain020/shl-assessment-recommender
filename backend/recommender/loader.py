# backend/recommender/loader.py
from pathlib import Path
import json, pickle

ROOT = Path(__file__).resolve().parents[1]              # .../backend
DATA_DIR = ROOT / "data"
CATALOG_FILE = DATA_DIR / "catalog" / "catalog.json"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
ARTIFACTS_META = ARTIFACTS_DIR / "artifacts.json"
BM25_TOKENS_PKL = ARTIFACTS_DIR / "bm25_tokens.pkl"
LGBM_TXT = ARTIFACTS_DIR / "lgbm_ltr.txt"
PRECOMP_EMB_NPY = ARTIFACTS_DIR / "item_embs.npy"       # optional

def load_catalog():
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(f"Catalog not found at {CATALOG_FILE}")
    with open(CATALOG_FILE, "r") as f:
        return json.load(f)

def _safe_load_tokens():
    if BM25_TOKENS_PKL.exists():
        with open(BM25_TOKENS_PKL, "rb") as f:
            obj = pickle.load(f)
        # Support list OR dict
        if isinstance(obj, dict):
            tokens = obj.get("tokens") or obj.get("vocab") or []
        else:
            tokens = obj
        return {"tokens": tokens}
    return {"tokens": []}

def load_artifacts():
    meta = {}
    if ARTIFACTS_META.exists():
        with open(ARTIFACTS_META, "r") as f:
            meta = json.load(f)

    bm25_pack = _safe_load_tokens()                      # {"tokens":[...]}
    ltr_path = str(LGBM_TXT) if LGBM_TXT.exists() else None

    precomp = None
    if PRECOMP_EMB_NPY.exists():
        import numpy as np
        precomp = np.load(PRECOMP_EMB_NPY)
    return meta, bm25_pack, ltr_path, precomp