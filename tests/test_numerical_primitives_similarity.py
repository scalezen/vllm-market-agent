"""Unit tests for numprims: a pure numpy/C++ kernel, no app dependencies."""
from pathlib import Path

import numpy as np
import pytest
from numerical_primitives import available_backends, topk_cosine


def _unit_rows(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, d)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_topk_matches_brute_force_argsort():
    matrix = _unit_rows(50, 8, seed=0)
    query = _unit_rows(1, 8, seed=1)[0]
    k = 5

    got = topk_cosine(matrix, query, k, backend="numpy")
    want = np.argsort(-(matrix @ query))[:k]
    assert list(got) == list(want)


def test_topk_is_highest_score_first():
    matrix = _unit_rows(20, 4, seed=2)
    query = _unit_rows(1, 4, seed=3)[0]
    got = topk_cosine(matrix, query, 6, backend="numpy")
    scores = matrix @ query
    assert list(scores[got]) == sorted(scores[got], reverse=True)


def test_k_larger_than_rows_is_rejected():
    matrix = _unit_rows(3, 4, seed=0)
    query = _unit_rows(1, 4, seed=1)[0]
    with pytest.raises(ValueError):
        topk_cosine(matrix, query, 4, backend="numpy")


def test_k_zero_is_rejected():
    matrix = _unit_rows(3, 4, seed=0)
    query = _unit_rows(1, 4, seed=1)[0]
    with pytest.raises(ValueError):
        topk_cosine(matrix, query, 0, backend="numpy")


def test_wrong_dtype_is_rejected():
    matrix = _unit_rows(3, 4, seed=0).astype(np.float64)
    query = _unit_rows(1, 4, seed=1)[0]
    with pytest.raises(ValueError):
        topk_cosine(matrix, query, 1, backend="numpy")


def test_dimension_mismatch_is_rejected():
    matrix = _unit_rows(3, 4, seed=0)
    query = _unit_rows(1, 5, seed=1)[0]
    with pytest.raises(ValueError):
        topk_cosine(matrix, query, 1, backend="numpy")


def test_non_contiguous_input_is_rejected():
    matrix = _unit_rows(5, 4, seed=0)[:, ::-1]  # view, not C-contiguous
    query = _unit_rows(1, 4, seed=1)[0]
    with pytest.raises(ValueError):
        topk_cosine(matrix, query, 1, backend="numpy")


def test_unknown_backend_is_rejected():
    matrix = _unit_rows(3, 4, seed=0)
    query = _unit_rows(1, 4, seed=1)[0]
    with pytest.raises(ValueError):
        topk_cosine(matrix, query, 1, backend="quantum")


def test_auto_backend_matches_numpy():
    matrix = _unit_rows(5, 4, seed=0)
    query = _unit_rows(1, 4, seed=1)[0]
    # assert "native" not in available_backends()  # not built in this environment
    got = topk_cosine(matrix, query, 2, backend="auto")
    want = topk_cosine(matrix, query, 2, backend="numpy")
    assert list(got) == list(want)


# --- extraction invariant: numprims must not know this is a stock-news app ------

FORBIDDEN = ("headline", "ticker", "news", "stock", "corpus")
SOURCE_DIRS = (Path(__file__).resolve().parent.parent / "python" / "numerical_primitives",
               Path(__file__).resolve().parent.parent / "cpp" / "numerical_primitives")


def test_numprims_source_has_no_application_words():
    hits = []
    for d in SOURCE_DIRS:
        for path in d.rglob("*"):
            if path.is_file():
                text = path.read_text(errors="ignore").lower()
                hits += [(str(path), word) for word in FORBIDDEN if word in text]
    assert not hits, f"application-specific words leaked into numprims: {hits}"
