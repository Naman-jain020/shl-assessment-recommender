import pandas as pd, requests, argparse, numpy as np

def recall_at_k(recommended, relevant):
    rec = set(recommended)
    rel = set(relevant)
    return len(rec & rel) / (len(rel) or 1)

def main(api_url, train_csv, k=10):
    df = pd.read_csv(train_csv)
    recalls = []
    for q, block in df.groupby("Query"):
        rel = block["Assessment_url"].dropna().tolist()
        r = requests.post(f"{api_url}/recommend", json={"text": q, "k": k})
        r.raise_for_status()
        preds = [x["url"] for x in r.json()["results"]]
        recalls.append(recall_at_k(preds, rel))
    print(f"Mean Recall@{k}: {np.mean(recalls):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--train_csv", default="data/train/Training Data.csv")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    main(args.api, args.train_csv, args.k)