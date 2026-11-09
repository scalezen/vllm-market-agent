"""Build per-ticker headline files for the benchmarks.

    python scripts/download_headlines_corpus.py WMT AMZN

Downloads the source and converts it to data/raw/news_all.parquet, replacing any existing
copy. Then, for each ticker, writes data/raw/<TICKER>_news.parquet (signal
headlines) and data/raw/no_<TICKER>_news.parquet (every other headline: the noise).
"""
import argparse
import re
from pathlib import Path

import duckdb
from huggingface_hub import hf_hub_download

REPO_ID = "khaihernlow/massive-stock-news-analysis-db-for-nlpbacktests"
FILENAME = "data/raw_analyst_ratings.jsonl"
RAW_DIR = Path("data/raw")
ALL_PARQUET = RAW_DIR / "news_all.parquet"


def build_corpus(con: duckdb.DuckDBPyConnection) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Into data/raw/ rather than the HF cache; downloaded again on every run.
    src = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset",
                               local_dir=RAW_DIR, force_download=True))
    print(f"source: {src} ({src.stat().st_size / 1e6:.0f} MB), converting to {ALL_PARQUET} ...")
    con.sql("INSTALL json; LOAD json;")
    # Written to a temporary file and renamed, so a failed run leaves the old file in place.
    tmp = ALL_PARQUET.with_suffix(".parquet.tmp")
    con.sql(f"COPY (SELECT * FROM read_json_auto('{src}')) TO '{tmp}' (FORMAT parquet)")
    tmp.replace(ALL_PARQUET)
    print(f"  {ALL_PARQUET.stat().st_size / 1e6:.0f} MB")


def split_ticker(con: duckdb.DuckDBPyConnection, ticker: str) -> None:
    for path, op in [(RAW_DIR / f"{ticker}_news.parquet", "="), (RAW_DIR / f"no_{ticker}_news.parquet", "<>")]:
        con.sql(f"COPY (SELECT * FROM '{ALL_PARQUET}' WHERE stock {op} '{ticker}') TO '{path}' (FORMAT parquet)")
        n = con.sql(f"SELECT count(*) FROM '{path}'").fetchall()[0][0]
        print(f"wrote {path}: {n} rows")


def _ticker(value: str) -> str:
    # The ticker is interpolated into SQL, so only accept ticker-shaped input.
    if not re.fullmatch(r"[A-Za-z][A-Za-z.\-]{0,9}", value):
        raise argparse.ArgumentTypeError(f"not a ticker: {value!r}")
    return value.upper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="+", type=_ticker)
    con = duckdb.connect()
    build_corpus(con)
    for ticker in parser.parse_args().tickers:
        split_ticker(con, ticker)
