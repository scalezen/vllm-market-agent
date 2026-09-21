"""Technical-indicator tool: RSI, moving averages and MACD from daily closes."""
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from loguru import logger

from .registry import TICKER_PARAM, register_tool

NY = ZoneInfo("America/New_York")  # US market hours assumed for the intraday check


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = gain.iloc[-1], loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + last_gain / last_loss))


@register_tool(
    "Get RSI, 50/200-day moving averages, MACD and 1-month price change for a stock ticker.",
    params=TICKER_PARAM,
)
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
    # While the market is open yfinance appends a live partial bar, so the last "close" is a
    # live price; otherwise it is the final close of `as_of`.
    now_ny = datetime.now(NY)
    as_of = close.index[-1].date()
    intraday = as_of == now_ny.date() and now_ny.time() < time(16, 0)
    price = float(close.iloc[-1])
    rsi = _rsi(close)
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    result = {
        "as_of": str(as_of),
        "intraday": intraday,  # True: price/indicators use a live intraday price, not a final close
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
