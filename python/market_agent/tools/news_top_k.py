"""Top-k headlines by cosine similarity to a query, from the local per-ticker corpus."""
import json

from loguru import logger
from numprims import topk_cosine

from .. import corpus
from .news import dedup_headlines
from .registry import TICKER_PARAM, register_tool


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
        items, vecs = corpus.load(ticker)
        if not items:
            return f"No headlines found for {ticker}."
        k = max(1, min(k, len(items)))
        top = topk_cosine(vecs, corpus.embed([query])[0], k)
    except Exception as exc:  # missing corpus file, embedding server down, ...
        return f"Error retrieving headlines for {ticker}: {exc}"

    by_title = {items[i][1]: items[i][0] for i in top}
    unique = dedup_headlines([items[i][1] for i in top])
    return json.dumps([{"published": by_title[t], "title": t} for t in unique])
