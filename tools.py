"""Tools the model can call: news (with embedding dedup), technicals, analyst ratings."""
import json

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from config import DEDUP_THRESHOLD, EMBED_MODEL, client
from usage import current_usage


# --------------------------------------------------------------------------
# Embeddings + dedup
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def get_stock_news(ticker: str, n: int = 10) -> str:
    """Fetch the latest news headlines for a ticker, with near-duplicates removed."""
    logger.info("[tool] get_stock_news({})", ticker)
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as exc:  # network errors, Yahoo rate limits, etc.
        return f"Error fetching news for {ticker}: {exc}"

    headlines = []
    for item in news[:n]:
        # yfinance >= 0.2.50 nests fields under "content"; older versions are flat.
        title = item.get("content", item).get("title")
        if title:
            headlines.append(title)
    headlines = dedup_headlines(headlines)
    return json.dumps(headlines) if headlines else "No news found."


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = gain.iloc[-1], loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + last_gain / last_loss))


def get_technical_indicators(ticker: str) -> str:
    """RSI(14), 50/200-day SMAs and MACD(12,26,9) computed from a year of daily closes."""
    logger.info("[tool] get_technical_indicators({})", ticker)
    try:
        close = yf.Ticker(ticker).history(period="1y")["Close"].dropna()
    except Exception as exc:
        return f"Error fetching price history for {ticker}: {exc}"
    if len(close) < 35:  # MACD needs at least the slow EMA window plus signal
        return f"Not enough price history for {ticker} to compute indicators."

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    price = float(close.iloc[-1])
    rsi = _rsi(close)
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    result = {
        "price": round(price, 2),
        "rsi_14": round(rsi, 1),
        "rsi_reading": "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral",
        "sma_50": round(sma50, 2),
        "sma_200": round(sma200, 2) if sma200 else None,
        "price_vs_sma_50": "above" if price > sma50 else "below",
        "price_vs_sma_200": None if sma200 is None else ("above" if price > sma200 else "below"),
        "macd": round(float(macd.iloc[-1]), 3),
        "macd_signal": round(float(macd_signal.iloc[-1]), 3),
        "macd_reading": "bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "bearish",
        "change_1m_pct": round(float(price / close.iloc[-22] - 1) * 100, 1) if len(close) > 22 else None,
    }
    return json.dumps(result)


def get_analyst_recommendations(ticker: str) -> str:
    """Analyst rating breakdown, price targets and the most recent upgrades/downgrades."""
    logger.info("[tool] get_analyst_recommendations({})", ticker)
    t = yf.Ticker(ticker)
    result: dict = {}
    try:
        summary = t.recommendations
        if summary is not None and not summary.empty:
            row = summary.iloc[0]  # current month
            result["ratings_current_month"] = {
                k: int(row[k]) for k in ("strongBuy", "buy", "hold", "sell", "strongSell") if k in row
            }
    except Exception as exc:
        logger.warning("recommendations failed for {}: {}", ticker, exc)
    try:
        targets = t.analyst_price_targets
        if targets:
            result["price_targets"] = {k: round(float(v), 2) for k, v in targets.items() if v is not None}
    except Exception as exc:
        logger.warning("price targets failed for {}: {}", ticker, exc)
    try:
        changes = t.upgrades_downgrades
        if changes is not None and not changes.empty:
            result["recent_changes"] = [
                {
                    "date": str(idx.date()),
                    "firm": row.get("Firm"),
                    "action": row.get("Action"),
                    "from": row.get("FromGrade"),
                    "to": row.get("ToGrade"),
                }
                for idx, row in changes.head(5).iterrows()
            ]
    except Exception as exc:
        logger.warning("upgrades/downgrades failed for {}: {}", ticker, exc)
    return json.dumps(result) if result else f"No analyst data found for {ticker}."


def _ticker_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL, GOOGL)"}
                },
                "required": ["ticker"],
            },
        },
    }


TOOLS = [
    _ticker_tool("get_stock_news", "Fetch the latest (de-duplicated) news headlines for a stock ticker."),
    _ticker_tool(
        "get_technical_indicators",
        "Get RSI, 50/200-day moving averages, MACD and 1-month price change for a stock ticker.",
    ),
    _ticker_tool(
        "get_analyst_recommendations",
        "Get analyst buy/hold/sell counts, price targets and recent upgrades/downgrades for a stock ticker.",
    ),
]
TOOL_REGISTRY = {
    "get_stock_news": get_stock_news,
    "get_technical_indicators": get_technical_indicators,
    "get_analyst_recommendations": get_analyst_recommendations,
}
