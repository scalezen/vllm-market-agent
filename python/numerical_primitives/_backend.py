"""Discovers the optional native backend (see ../../cpp/numerical_primitives/README.md).

Import failure here means the native extension hasn't been built yet, not an
error; `available_backends` and `native_topk_cosine` are how callers find out
and degrade.
"""
from typing import Any

import numpy as np


def _load_native() -> Any | None:
    try:
        from . import _core
    except ImportError:
        return None
    return _core


_native = _load_native()


def available_backends() -> list[str]:
    return ["numpy", "native"] if _native is not None else ["numpy"]


def native_topk_cosine(matrix: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    if _native is None:
        raise RuntimeError("native backend is not built; see cpp/numerical_primitives/README.md")
    return _native.topk_cosine(matrix, query, k)
