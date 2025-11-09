from typing import List, Tuple, Dict

def balance_kpc(scored: List[Tuple[str, float]], items_by_url: Dict[str, dict], requirements: Dict, k: int):
    if not requirements.get("needs_diversity"):
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    buckets = {"K": [], "P": [], "C": [], "Other": []}
    for url,score in scored:
        t = items_by_url[url].get("test_type","Other")
        buckets.setdefault(t, []).append((url,score))

    for b in buckets.values():
        b.sort(key=lambda x: x[1], reverse=True)

    tk = min(k//2, len(buckets["K"]))
    tp = min(max(k//3, 1), len(buckets["P"]))
    tc = max(k - tk - tp, 0)

    out = buckets["K"][:tk] + buckets["P"][:tp] + buckets["C"][:tc]
    if len(out) < k:
        rest = (buckets["K"][tk:] + buckets["P"][tp:] + buckets["C"][tc:] + buckets["Other"])
        rest.sort(key=lambda x: x[1], reverse=True)
        out += rest[:k-len(out)]
    return out[:k]