# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

from backend.recommender.loader import load_catalog
from backend.recommender.features import (
    SimpleBM25,
    embed_query,
    embed_items,
    sim_scores,
    keyword_overlap,
    type_balance_weights,
)
from backend.recommender.utils import extract_requirements
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SHL Assessment Recommender API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Streamlit Render URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load catalog and models ----
CATALOG = load_catalog()                 # must be list[dict]; len = N
URL2ITEM = {x["url"]: x for x in CATALOG}
N = len(CATALOG)

EMB_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
ITEM_EMBS = embed_items(CATALOG, EMB_MODEL)        # [N, D]
BM25 = SimpleBM25(CATALOG)                         # safe BM25 built over same N

URL2IDX = {it["url"]: i for i, it in enumerate(CATALOG)}

# ---- Schemas ----
class RecommendIn(BaseModel):
    text: str
    k: int = 10

class RecItem(BaseModel):
    name: str
    url: str
    test_type: str
    score: float

class RecommendOut(BaseModel):
    results: List[RecItem]

@app.get("/health")
def health():
    return {"status": "ok", "num_items": len(CATALOG)}

def _z(x: np.ndarray) -> np.ndarray:
    # z-score then min-max to [0,1]
    if x.size == 0:
        return x.astype(np.float32)
    m = float(x.mean())
    s = float(x.std()) or 1.0
    z = (x - m) / s
    mn, mx = float(z.min()), float(z.max())
    if mx - mn < 1e-8:
        return np.zeros_like(z, dtype=np.float32)
    return ((z - mn) / (mx - mn)).astype(np.float32)

def _balanced_topk(scored: List[tuple], k: int, req: Dict) -> List[tuple]:
    # scored: [(idx, score)], select balanced by test_type
    # simple proportional selection
    types = {"K": [], "P": [], "C": [], "O": []}
    for idx, s in scored:
        t = (CATALOG[idx].get("test_type") or "O").upper()
        types.get(t if t in types else "O").append((idx, s))
    for lst in types.values():
        lst.sort(key=lambda x: x[1], reverse=True)

    tech = bool(req.get("tech_skills"))
    soft = bool(req.get("soft_skills"))

    out = []
    if tech and soft:   # 50% K, 30% P, 20% C
        quota = [("K", int(0.5 * k)), ("P", int(0.3 * k))]
        used = 0
        for t, q in quota:
            out.extend(types[t][:q])
            used += len(types[t][:q])
        rem = k - used
        pool = types["C"] + types["K"][quota[0][1]:] + types["P"][quota[1][1]:] + types["O"]
        pool.sort(key=lambda x: x[1], reverse=True)
        out.extend(pool[:rem])
    else:
        # just take best k overall
        pool = []
        for lst in types.values():
            pool.extend(lst)
        pool.sort(key=lambda x: x[1], reverse=True)
        out = pool[:k]
    return out[:k]

@app.post("/recommend", response_model=RecommendOut)
def recommend(inp: RecommendIn):
    query = inp.text.strip()
    k = max(1, min(10, inp.k))

    # 1) pick safe candidate set from BM25 (on same N)
    cand_idx = BM25.top_k(query, k=min(64, max(16, 3 * k)))  # ndarray of valid indices

    # 2) compute features aligned to cand_idx lengths
    q_emb = embed_query(query, EMB_MODEL)
    sim_vec, sim_idx = sim_scores(q_emb, ITEM_EMBS, indices=cand_idx)        # len M
    # sim_idx equals the cleaned cand_idx we will use everywhere
    bm25_vec, _ = BM25.score(query, indices=sim_idx)                          # len M
    kw_vec = np.array([keyword_overlap(query, CATALOG[i]) for i in sim_idx], dtype=np.float32)

    # 3) normalize & mix scores
    s_sim = _z(sim_vec)
    s_bm25 = _z(bm25_vec)
    s_kw = _z(kw_vec)

    # light, stable fusion
    fused = 0.55 * s_sim + 0.35 * s_bm25 + 0.10 * s_kw

    # 4) apply type weighting
    req = extract_requirements(query)
    weights = np.array([type_balance_weights(req, CATALOG[i].get("test_type", "")) for i in sim_idx], dtype=np.float32)
    final_scores = fused * weights

    # 5) rank and balance
    order = np.argsort(-final_scores)
    ranked = [(int(sim_idx[i]), float(final_scores[i])) for i in order]
    balanced = _balanced_topk(ranked, k, req)

    results = [
        RecItem(
            name=CATALOG[i]["name"],
            url=CATALOG[i]["url"],
            test_type=(CATALOG[i].get("test_type") or "").upper(),
            score=float(s),
        )
        for i, s in balanced
    ]
    return RecommendOut(results=results)