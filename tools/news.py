"""News tool: recent dated headlines with embedding-based near-duplicate removal."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import yfinance as yf
from loguru import logger

from config import DEDUP_THRESHOLD, EMBED_MODEL, NEWS_MAX_AGE_DAYS, client
from usage import current_usage

from .registry import TICKER_PARAM, register_tool


def dedup_headlines(headlines: list[str], threshold: float = DEDUP_THRESHOLD) -> list[str]:
    """Drop near-duplicate headlines, keeping the first of each cluster.

    Uses the server's embeddings endpoint; falls back to exact (case-insensitive)
    matching if the server has no embedding model loaded.
    """
    if len(headlines) < 2:
        return headlines
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=headlines)
        if usage := current_usage.get():
            usage.add_embedding(response)
        data = response.data
        vecs = np.array([d.embedding for d in sorted(data, key=lambda d: d.index)], dtype=float)
    except Exception as exc:  # no embedding model, connection error, ...
        logger.warning("Embedding dedup unavailable ({}); using exact-match dedup", exc)
        seen: dict[str, str] = {}
        for h in headlines:
            seen.setdefault(h.strip().lower(), h)
        return list(seen.values())

    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    kept: list[int] = []
    for i in range(len(headlines)):
        if all(float(vecs[i] @ vecs[j]) < threshold for j in kept):
            kept.append(i)
    logger.info("Dedup: {} headlines -> {} unique", len(headlines), len(kept))
    return [headlines[i] for i in kept]


def _published(content: dict) -> datetime | None:
    """Publish time of a yfinance news item (new `pubDate` or older `providerPublishTime`)."""
    try:
        if raw := content.get("pubDate"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts := content.get("providerPublishTime"):
            return datetime.fromtimestamp(ts, timezone.utc)
    except (ValueError, TypeError, OSError):
        pass
    return None


@register_tool(
    "Fetch the latest dated, de-duplicated news headlines (last week, newest first) for a stock ticker.",
    params=TICKER_PARAM,
)
def get_stock_news(ticker: str, n: int = 10) -> str:
    """Newest headlines from the last NEWS_MAX_AGE_DAYS days, dated, near-duplicates removed."""
    logger.info("[tool] get_stock_news({})", ticker)
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as exc:  # network errors, Yahoo rate limits, etc.
        return f"Error fetching news for {ticker}: {exc}"

    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    dated = []
    for item in news:
        # yfinance >= 0.2.50 nests fields under "content"; older versions are flat.
        content = item.get("content", item)
        title, published = content.get("title"), _published(content)
        if title and published and published >= cutoff:  # undated items cannot be aged, so skip them
            dated.append((published, title))
    dated = sorted(dated, reverse=True)[:n]  # the feed is not strictly newest-first

    unique = dedup_headlines([title for _, title in dated])  # keeps the newest of each cluster
    published_by_title = {title: published for published, title in reversed(dated)}
    items = [{"published": published_by_title[t].strftime("%Y-%m-%dT%H:%MZ"), "title": t} for t in unique]
    return json.dumps(items) if items else f"No news found in the last {NEWS_MAX_AGE_DAYS} days."
