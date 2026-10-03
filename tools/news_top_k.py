"""Top-k headlines by cosine similarity to a query, from the local per-ticker corpus."""
import json
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
from loguru import logger

from config import EMBED_MODEL, client

from .news import dedup_headlines
from .registry import TICKER_PARAM, register_tool

CORPUS = Path("data/raw/news_all.parquet")
EMBED_BATCH = 64


def _embed(texts: list[str]) -> np.ndarray:
    """Unit-length embeddings, so cosine similarity is a plain dot product."""
    rows = []
    for i in range(0, len(texts), EMBED_BATCH):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        rows += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
    vecs = np.array(rows, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)


@lru_cache(maxsize=16)
def _corpus(ticker: str) -> tuple[tuple[tuple[str, str], ...], np.ndarray]:
    """Headlines for one ticker (date, headline) and their embeddings; cached per process."""
    con = duckdb.connect()
    rows = con.execute(
        "SELECT CAST(date AS VARCHAR), headline FROM read_parquet(?) "
        "WHERE stock = ? AND headline IS NOT NULL ORDER BY date DESC",
        [str(CORPUS), ticker],
    ).fetchall()
    if not rows:
        return (), np.empty((0, 0), dtype=np.float32)
    return tuple(rows), _embed([headline for _, headline in rows])


@register_tool(
    "Fetch the headlines most semantically similar to a query for a stock ticker, "
    "from the historical headline corpus. Pass the user's question as the query.",
    params={**TICKER_PARAM, "query": "The question or topic to find relevant headlines for."},
)
def get_stock_news_top_k(ticker: str, query: str, k: int = 10) -> str:
    """Rank the ticker's headlines by cosine similarity to `query`, keep the top k, drop near-duplicates."""
    ticker = ticker.upper()
    logger.info("[tool] get_stock_news_top_k({}, k={})", ticker, k)
    try:
        items, vecs = _corpus(ticker)
        if not items:
            return f"No headlines found for {ticker}."
        scores = vecs @ _embed([query])[0]
    except Exception as exc:  # missing corpus file, embedding server down, ...
        return f"Error retrieving headlines for {ticker}: {exc}"

    k = max(1, min(k, len(items)))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    by_title = {items[i][1]: items[i][0] for i in top}
    unique = dedup_headlines([items[i][1] for i in top])
    return json.dumps([{"published": by_title[t], "title": t} for t in unique])
