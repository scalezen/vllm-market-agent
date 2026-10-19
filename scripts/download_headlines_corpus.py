"""Build a per-ticker news corpus for the retrieval experiments.

    python scripts/download_headlines_corpus.py          # AAPL
    python scripts/download_headlines_corpus.py WMT

Downloads the source once into data/raw/, converts it to parquet once, then
writes data/raw/<TICKER>_news.parquet.
"""
import argparse
import re
from pathlib import Path

import duckdb
from huggingface_hub import hf_hub_download

REPO_ID = "khaihernlow/massive-stock-news-analysis-db-for-nlpbacktests"
FILENAME = "data/raw_analyst_ratings.jsonl"
RAW_DIR = Path("data/raw")


def ticker_parquet(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}_news.parquet"


def main(ticker: str) -> None:
    all_parquet = RAW_DIR / "news_all.parquet"
    out = ticker_parquet(ticker)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Pull the source into data/raw/ rather than the HF cache; skipped if already there.
    src = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", local_dir=RAW_DIR))
    print(f"source: {src} ({src.stat().st_size / 1e6:.0f} MB)")

    con = duckdb.connect()
    con.sql("INSTALL json; LOAD json;")

    if not all_parquet.exists():
        print(f"converting to {all_parquet} ...")
        con.sql(f"COPY (SELECT * FROM read_json_auto('{src}')) TO '{all_parquet}' (FORMAT parquet)")
        print(f"  {all_parquet.stat().st_size / 1e6:.0f} MB")
        print(con.sql(f"DESCRIBE SELECT * FROM '{all_parquet}'").df())

    con.sql(f"COPY (SELECT * FROM '{all_parquet}' WHERE stock = '{ticker}') TO '{out}' (FORMAT parquet)")
    n = con.sql(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"wrote {out}: {n} rows")


def read_headlines(ticker: str) -> None:
    import pandas as pd

    df = pd.read_parquet(ticker_parquet(ticker))
    print(df.shape)
    print(df.head())


def _ticker(value: str) -> str:
    # The ticker is interpolated into SQL, so only accept ticker-shaped input.
    if not re.fullmatch(r"[A-Za-z][A-Za-z.\-]{0,9}", value):
        raise argparse.ArgumentTypeError(f"not a ticker: {value!r}")
    return value.upper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", nargs="?", default="AAPL", type=_ticker, help="default: AAPL")
    ticker = parser.parse_args().ticker
    main(ticker)
    read_headlines(ticker)
