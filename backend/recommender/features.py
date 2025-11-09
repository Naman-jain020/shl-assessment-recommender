# backend/recommender/features.py
import re
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Iterable, Optional

_WORD = re.compile(r"[a-z0-9]+")

def _norm_text(s: str) -> str:
    return s.lower()

def tokenize(s: str) -> List[str]:
    return _WORD.findall(_norm_text(s))

def item_text(item: Dict) -> str:
    # robust text source for retrieval
    name = item.get("name", "")
    desc = item.get("description", "")
    kw = " ".join(item.get("keywords", []) or [])
    return f"{name}. {desc}. {kw}"

class SimpleBM25:
    """Self-contained BM25 over the loaded catalog (no external pack assumptions)."""
    def __init__(self, catalog: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(catalog)
        self.doc_tokens: List[List[str]] = []
        self.doc_tf: List[Counter] = []
        self.df: Counter = Counter()
        self.doc_len: List[int] = []
        for it in catalog:
            toks = tokenize(item_text(it))
            self.doc_tokens.append(toks)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            for t in tf.keys():
                self.df[t] += 1
        self.doc_len = np.asarray(self.doc_len, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 1.0

        # precompute idf
        self.idf: Dict[str, float] = {}
        for t, df in self.df.items():
            # BM25+ style idf, stable for small df
            self.idf[t] = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def _score_doc(self, q_tokens: List[str], i: int) -> float:
        tf = self.doc_tf[i]
        dl = self.doc_len[i]
        denom_part = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        score = 0.0
        for t in q_tokens:
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            score += self.idf[t] * (f * (self.k1 + 1.0)) / (f + denom_part)
        return score

    def score(self, query: str, indices: Optional[Iterable[int]] = None) -> np.ndarray:
        q_tokens = tokenize(query)
        if indices is None:
            indices = range(self.N)
        idx = np.array([i for i in dict.fromkeys(indices) if 0 <= i < self.N], dtype=int)
        if idx.size == 0:
            idx = np.arange(self.N, dtype=int)
        out = np.zeros(idx.size, dtype=np.float32)
        for j, i in enumerate(idx):
            out[j] = self._score_doc(q_tokens, i)
        return out, idx

    def top_k(self, query: str, k: int = 50) -> np.ndarray:
        # full scores then take top-k; safe if N is small
        scores, idx = self.score(query, None)
        # here idx is 0..N-1; scores aligned to idx
        order = np.argsort(-scores)
        return idx[order[: min(k, scores.size)]]

def embed_items(catalog: List[Dict], model) -> np.ndarray:
    texts = [item_text(it) for it in catalog]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)

def embed_query(q: str, model) -> np.ndarray:
    return np.asarray(model.encode([q], normalize_embeddings=True))[0].astype(np.float32)

def sim_scores(q_emb: np.ndarray,
               item_embs: np.ndarray,
               indices: Optional[Iterable[int]] = None) -> (np.ndarray, np.ndarray):
    """
    Returns (scores_vector_aligned_to_indices, indices_ndarray)
    indices is cleaned & clipped to valid range.
    """
    N = item_embs.shape[0]
    if indices is None:
        idx = np.arange(N, dtype=int)
    else:
        idx = np.array([i for i in dict.fromkeys(indices) if 0 <= i < N], dtype=int)
        if idx.size == 0:
            idx = np.arange(N, dtype=int)
    sub = item_embs[idx]  # [M, D]
    sims = (sub @ q_emb.reshape(-1, 1)).ravel().astype(np.float32)  # [M]
    return sims, idx

def keyword_overlap(query: str, item: Dict) -> float:
    q_set = set(tokenize(query))
    d_set = set(tokenize(item_text(item)))
    if not q_set:
        return 0.0
    return float(len(q_set & d_set)) / float(len(q_set))

def type_balance_weights(requirements: Dict, test_type: str) -> float:
    """
    Simple weighting: boost P if soft skills present, boost K if tech skills present, slight C otherwise.
    """
    tech = requirements.get("tech_skills", [])
    soft = requirements.get("soft_skills", [])
    t = (test_type or "").upper()
    w = 1.0
    if tech and t == "K":
        w *= 1.15
    if soft and t == "P":
        w *= 1.15
    if t == "C":
        w *= 1.05
    return float(w)