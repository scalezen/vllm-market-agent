"""Analyst tool: rating counts, price targets and recent rating changes."""
import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from loguru import logger

from config import ANALYST_CHANGES_MAX_AGE_DAYS

from .registry import TICKER_PARAM, register_tool


@register_tool(
    "Get current-month analyst buy/hold/sell counts, current price targets and "
    "upgrades/downgrades from the last month for a stock ticker.",
    params=TICKER_PARAM,
)
def get_analyst_recommendations(ticker: str) -> str:
    """Analyst rating breakdown, price targets and the most recent upgrades/downgrades."""
    logger.info("[tool] get_analyst_recommendations({})", ticker)
    # Snapshot semantics, so the sections line up:
    #   ratings_current_month: the "0m" row of yfinance's monthly recommendation summary
    #   price_targets:         the current consensus at fetch time (yfinance gives no date)
    #   recent_changes:        individual rating actions, each with its own date, limited to
    #                          the last ANALYST_CHANGES_MAX_AGE_DAYS days (about 1 month)
    #                          to match the 0m snapshot; older ones are dropped
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
            idx = pd.DatetimeIndex(changes.index)
            changes.index = idx.tz_convert("UTC").tz_localize(None) if idx.tz else idx
            cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=ANALYST_CHANGES_MAX_AGE_DAYS)
            changes = changes[changes.index >= cutoff]
            result["recent_changes"] = [  # [] when there were none this month
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
    if not result:
        return f"No analyst data found for {ticker}."
    # `as_of` is when the snapshots were fetched (UTC date).
    return json.dumps({"as_of": datetime.now(timezone.utc).date().isoformat(), **result})
