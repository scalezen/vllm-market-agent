"""Top-k cosine similarity search over a matrix of unit vectors.

Pure numpy or C++ kernel based on the backend arg of topk_cosine.
"""
import numpy as np

from ._backend import available_backends, native_topk_cosine
from .similarity import topk_cosine_numpy

__all__ = ["available_backends", "topk_cosine"]


def topk_cosine(matrix: np.ndarray, query: np.ndarray, k: int, backend: str = "auto") -> np.ndarray:
    """k indices into `matrix`, highest cosine similarity to `query` first.

    matrix: (N, d) float32, C-contiguous, L2-normalised rows.
    query:  (d,) float32, C-contiguous, L2-normalised.
    backend: "auto" (native if built, else numpy), "numpy", or "native".
    """
    _validate(matrix, query, k)
    if backend == "auto":
        backend = "native" if "native" in available_backends() else "numpy"
    if backend == "numpy":
        return topk_cosine_numpy(matrix, query, k)
    if backend == "native":
        return native_topk_cosine(matrix, query, k)
    raise ValueError(f"unknown backend {backend!r}; available: {['numpy'] + [b for b in available_backends() if b != 'numpy']}")


def _validate(matrix: np.ndarray, query: np.ndarray, k: int) -> None:
    if matrix.ndim != 2 or query.ndim != 1:
        raise ValueError(f"expected matrix.ndim == 2 and query.ndim == 1, got {matrix.ndim} and {query.ndim}")
    if matrix.dtype != np.float32 or query.dtype != np.float32:
        raise ValueError(f"expected float32 arrays, got matrix={matrix.dtype}, query={query.dtype}")
    if not matrix.flags["C_CONTIGUOUS"] or not query.flags["C_CONTIGUOUS"]:
        raise ValueError("matrix and query must be C-contiguous")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(f"dimension mismatch: matrix has {matrix.shape[1]} columns, query has {query.shape[0]}")
    if matrix.shape[0] == 0:
        raise ValueError("matrix must have at least one row")
    if not (1 <= k <= matrix.shape[0]):
        raise ValueError(f"k must be between 1 and {matrix.shape[0]}, got {k}")
