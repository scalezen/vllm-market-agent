"""Unit tests for corpus.py: duckdb filtering + lru_cache, no real embedding calls."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import corpus


@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "stock": ["AAPL", "AAPL", "MSFT"],
            "date": ["2020-01-02", "2020-01-01", "2020-01-01"],
            "headline": ["Apple unveils new iPhone", "Apple beats earnings", "Microsoft cloud grows"],
        }
    )
    path = tmp_path / "news_all.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(corpus, "CORPUS", path)
    fake_embed = MagicMock(side_effect=lambda texts: np.eye(len(texts), dtype=np.float32))
    monkeypatch.setattr(corpus, "embed", fake_embed)
    corpus.load.cache_clear()
    return fake_embed


def test_load_filters_by_ticker_and_sorts_newest_first(fake_corpus):
    items, vecs = corpus.load("AAPL")
    assert [date for date, _ in items] == ["2020-01-02", "2020-01-01"]
    assert vecs.shape == (2, 2)


def test_load_is_case_sensitive_on_the_stored_ticker(fake_corpus):
    items, _ = corpus.load("aapl")
    assert items == ()  # corpus stores upper-case; callers upper-case first (see tools/news_top_k.py)


def test_load_missing_ticker_returns_empty_without_embedding(fake_corpus):
    items, vecs = corpus.load("TSLA")
    assert items == ()
    assert vecs.shape == (0, 0)
    fake_corpus.assert_not_called()


def test_load_is_cached_per_ticker(fake_corpus):
    corpus.load("AAPL")
    corpus.load("AAPL")
    assert fake_corpus.call_count == 1


def test_embed_returns_unit_length_rows(monkeypatch):
    class FakeDatum:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

    class FakeResponse:
        def __init__(self, data):
            self.data = data

    def fake_create(model, input):
        return FakeResponse([FakeDatum(i, [float(i + 1), 0.0, 0.0]) for i in range(len(input))])

    monkeypatch.setattr(corpus.client.embeddings, "create", fake_create)
    vecs = corpus.embed(["a", "b", "c"])
    assert vecs.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
