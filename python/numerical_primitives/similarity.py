"""Reference numpy implementation of top-k search over cosine-similarity scores."""
import numpy as np


def topk_cosine_numpy(matrix: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k rows of `matrix` with highest dot product with `query`, best first.

    Both must already be L2-normalised for the dot product to equal cosine
    similarity; this function does not check that. argpartition finds the k
    best in O(N), then only those k are sorted -- never the full N.
    """
    scores = matrix @ query
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])]
