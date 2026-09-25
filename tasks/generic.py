"""Free-form task: same three tools as sentiment, but the system and user prompts
are meant to be customised per run (see generic_agent.py) and the report is
just a summary + confidence, not a Bullish/Bearish call."""
from reports import GenericReport

from .base import Task

GENERIC = Task(
    name="generic",
    description="Research a ticker with the usual tools and answer whatever was asked.",
    system_prompt=(
        "You are a quantitative financial assistant. "
        #"For the requested ticker, call ALL of "
        #"get_stock_news, get_technical_indicators and get_analyst_recommendations,"
        "then answer the user's question using only what the tools returned."
    ),
    user_prompt="Tell me what's notable about {ticker} right now.",
    report_prompt=(
        "Using only the tool results above, answer as JSON: a confidence between 0 and 1, "
        "and a summary that directly answers what was asked."
    ),
    report_model=GenericReport,
    tools=("get_stock_news", "get_technical_indicators", "get_analyst_recommendations"),
)
