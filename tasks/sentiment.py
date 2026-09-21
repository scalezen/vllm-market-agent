"""Stock sentiment from news, technicals and analyst ratings."""
from reports import SentimentReport

from .base import Task

SENTIMENT = Task(
    name="sentiment",
    description="Bullish/Bearish/Neutral sentiment for a ticker from news, technicals and analysts.",
    system_prompt=(
        "You are a quantitative financial assistant. For the requested ticker, call ALL of "
        "get_stock_news, get_technical_indicators and get_analyst_recommendations, then "
        "assess overall sentiment (Bullish, Bearish, or Neutral) from the results."
    ),
    user_prompt="Can you check the current sentiment for {ticker}?",
    report_prompt=(
        "Using only the tool results above, output the sentiment report as JSON. "
        "Give each of news_signal, technical_signal and analyst_signal (use 'Unavailable' if that "
        "tool returned no usable data), the overall sentiment, a confidence between 0 and 1, "
        "and a one-sentence summary."
    ),
    report_model=SentimentReport,
    tools=("get_stock_news", "get_technical_indicators", "get_analyst_recommendations"),
)
