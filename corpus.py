"""Per-ticker headline corpus and its embedding index."""
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np

from config import EMBED_MODEL, client

CORPUS = Path("data/raw/news_all.parquet")
EMBED_BATCH = 64


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length embeddings, so cosine similarity is a plain dot product."""
    rows = []
    for i in range(0, len(texts), EMBED_BATCH):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        rows += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
    vecs = np.array(rows, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)


@lru_cache(maxsize=16)
def load(ticker: str) -> tuple[tuple[tuple[str, str], ...], np.ndarray]:
    """Headlines for one ticker (date, headline) and their embeddings; cached per process."""
    con = duckdb.connect()
    rows = con.execute(
        "SELECT CAST(date AS VARCHAR), headline FROM read_parquet(?) "
        "WHERE stock = ? AND headline IS NOT NULL ORDER BY date DESC",
        [str(CORPUS), ticker],
    ).fetchall()
    if not rows:
        return (), np.empty((0, 0), dtype=np.float32)
    return tuple(rows), embed([headline for _, headline in rows])
