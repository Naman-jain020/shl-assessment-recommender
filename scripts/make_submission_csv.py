import pandas as pd, argparse

def main(infile, outfile):
    df = pd.read_csv(infile)  # your predictions with Query + Assessment_url
    out = df[["Query","Assessment_url"]].copy()
    out.to_csv(outfile, index=False)
    print(f"Wrote {outfile} with {len(out)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/artifacts/predictions.csv")
    ap.add_argument("--outfile", default="submission.csv")
    args = ap.parse_args()
    main(args.infile, args.outfile)