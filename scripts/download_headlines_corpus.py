"""Build a per-ticker news corpus for the retrieval experiments.

Downloads the source once into data/raw/, converts it to parquet once, then
filters per ticker. 
"""
import argparse
from pathlib import Path

import duckdb
from huggingface_hub import hf_hub_download

REPO_ID = "khaihernlow/massive-stock-news-analysis-db-for-nlpbacktests"
FILENAME = "data/raw_analyst_ratings.jsonl"


def main() -> None:

    ticker = "AAPL"

    raw_dir = Path("data/raw")
    all_parquet = raw_dir / "news_all.parquet"
    out = raw_dir / "aapl_news.parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Pull the source into data/raw/ rather than the HF cache.
    src = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                               repo_type="dataset", local_dir=raw_dir))
    print(f"source: {src} ({src.stat().st_size / 1e6:.0f} MB)")

    con = duckdb.connect()
    con.sql("INSTALL json; LOAD json;")

    # JSONL -> parquet, once. Deduplicate key names 
    if True:
        print(f"converting to {all_parquet} ...")
        con.sql(f"COPY (SELECT * FROM read_json_auto('{src}')) "
                f"TO '{all_parquet}' (FORMAT parquet)")
        print(f"  {all_parquet.stat().st_size / 1e6:.0f} MB")

    print(con.sql(f"DESCRIBE SELECT * FROM '{all_parquet}'").df())

    # 3. Per-ticker slice.
    con.sql(f"""
        COPY (SELECT * FROM '{all_parquet}'
              WHERE stock = '{ticker}')
        TO '{out}' (FORMAT parquet)
    """)
    n = con.sql(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"wrote {out}: {n} rows")

def read_headlines(ticker:str = "AAPL"):
    import pandas as pd

    parquet_filename = f"data/raw/{ticker.lower()}_news.parquet"
    print(f"parquet_filename is {parquet_filename}")
    df = pd.read_parquet(parquet_filename)
    print(df.shape)
    print(df.head)


if __name__ == "__main__":
    #main()
    read_headlines()