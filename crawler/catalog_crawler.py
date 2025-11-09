# crawler/catalog_crawler.py
from __future__ import annotations
import argparse, json, os, re, time, sys
from collections import deque
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

CATALOG_ROOT = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}

# Map category hints to SHL Test Types
TYPE_RULES = [
    (re.compile(r"\bknowledge|skills|programming|java|python|sql|excel|.net|aws|cloud|coding\b", re.I), "K"),
    (re.compile(r"\bpersonality|behavior|behaviour|opq|traits|collaboration|teamwork|leadership\b", re.I), "P"),
    (re.compile(r"\bability|aptitude|cognitive|numerical|verbal|inductive|deductive|reasoning\b", re.I), "A"),
    (re.compile(r"\bbiodata|situational|judgement\b", re.I), "B"),
    (re.compile(r"\bcompetenc(y|ies)\b", re.I), "C"),
    (re.compile(r"\b360\b|\bdevelopment\b|\bfeedback\b", re.I), "D"),
    (re.compile(r"\bexercise\b|\bassessment centre|assessment center\b", re.I), "E"),
    (re.compile(r"\bsimulation\b|\bsituational experience\b", re.I), "S"),
]

KEYWORDS = [
    "java","python","sql","javascript","excel",".net","c#","c++","aws","react","angular","node",
    "communication","collaboration","leadership","teamwork","analytical","problem solving",
    "numerical","verbal","reasoning","cognitive","opq","personality","behavior","sales","marketing",
    "customer service","tableau","power bi","data analysis"
]

ASS_PAGE_RE = re.compile(r"/products/product-catalog/[^/]+/$", re.I)
PREPACK_RE = re.compile(r"\bpre[-\s]?packaged\b|\bsolution\b", re.I)  # used on container pages
SAME_HOST = urlparse(CATALOG_ROOT).netloc

def guess_test_type(name: str, text: str) -> str:
    blob = f"{name} {text}".lower()
    for pat, code in TYPE_RULES:
        if pat.search(blob):
            return code
    return "K"  # safe default

def extract_keywords(text: str) -> list[str]:
    t = text.lower()
    return sorted({k for k in KEYWORDS if k in t})

def clean_text(soup: BeautifulSoup) -> str:
    # get a reasonable description from visible text blocks
    pieces = []
    for tag in soup.find_all(["p","li","div","section","article"]):
        txt = tag.get_text(" ", strip=True)
        if txt and 60 <= len(txt) <= 600:
            pieces.append(txt)
        if len(" ".join(pieces)) > 1000:
            break
    return " ".join(pieces)[:1200]

def is_same_host(url: str) -> bool:
    try:
        return urlparse(url).netloc == SAME_HOST
    except Exception:
        return False

def is_assessment_page(url: str) -> bool:
    """
    SHL individual assessments are under:
    /products/product-catalog/view/<slug>/
    We exclude pre-packaged job bundles which often contain 'solution'/'package'
    """
    url_l = url.lower()
    return (
        "/products/product-catalog/view/" in url_l
        and "solution" not in url_l         # exclude packaged job solutions
        and "job-" not in url_l             # exclude job bundles
        and not url_l.endswith("/product-catalog/view/")  # avoid directory listing
    )

def crawl(start: str, max_pages: int, delay: float, verbose: bool=False) -> list[dict]:
    seen: set[str] = set()
    q = deque([start])
    results: dict[str, dict] = {}
    pages_visited = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    while q and pages_visited < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        try:
            r = session.get(url, timeout=15)
        except requests.RequestException as e:
            if verbose:
                print(f"[WARN] {url} -> {e}", file=sys.stderr)
            continue

        pages_visited += 1
        if verbose:
            print(f"[{pages_visited}] GET {url} ({r.status_code})")

        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type",""):
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # If this is an assessment page, extract it
        if is_assessment_page(url):
            name = (soup.find("h1") or soup.find("h2") or soup.title).get_text(strip=True) if soup else url.rstrip("/").split("/")[-1].replace("-", " ").title()
            desc = clean_text(soup)
            ttype = guess_test_type(name, desc)
            kw = extract_keywords(name + " " + desc)
            results[url] = {
                "name": name,
                "url": url,
                "test_type": ttype,
                "description": desc,
                "keywords": kw,
            }

        # Discover more links (keep to same host)
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#") or href.lower().startswith("mailto:") or href.lower().startswith("tel:"):
                continue
            nxt = urljoin(url, href)
            if not is_same_host(nxt):
                continue
            if nxt in seen:
                continue
            # Skip obvious “pre-packaged job solutions” hub pages
            if "pre-packaged" in nxt.lower() or "/solutions/" in nxt.lower() and "/product-catalog/" not in nxt.lower():
                continue
            # Only queue catalog paths (reduce noise)
            if "/products/product-catalog/" in nxt and "/products/product-catalog/view/" in nxt:
                q.append(nxt)
            elif "/products/product-catalog/" in nxt:
                q.append(nxt)   # keep category pages too

        time.sleep(delay)

    # return list for stable JSON
    return sorted(results.values(), key=lambda x: x["name"].lower())

def save_json(items: list[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Crawl SHL Individual Test Solutions (exclude Pre-packaged Job Solutions)."
    )
    p.add_argument("--start", default=CATALOG_ROOT, help="Entry URL (catalog root)")
    p.add_argument("--out", required=True, help="Path to write catalog.json")
    p.add_argument("--max-pages", type=int, default=800, help="Max HTML pages to fetch")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    p.add_argument("--verbose", action="store_true", help="Print progress logs")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.verbose:
        print(f"Starting crawl from: {args.start}")
        print(f"Writing to: {args.out}")
        print(f"max_pages={args.max_pages}, delay={args.delay}")
    items = crawl(args.start, args.max_pages, args.delay, verbose=args.verbose)
    save_json(items, args.out)
    print(f"✓ Saved {len(items)} assessments to {args.out}")

if __name__ == "__main__":
    main()