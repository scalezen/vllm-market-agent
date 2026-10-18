# cpp/numerical_primitives

Native implementation of the same kernel as `python/numerical_primitives`: top-k search
over cosine-similarity scores between a query vector and a matrix of unit
vectors. 

## Interface it must satisfy

A pybind11 extension module named `_core`, importable as `numerical_primitives._core`
(i.e. built into `python/numerical_primitives/`), exposing:

```
topk_cosine(matrix: np.ndarray[float32, (N, d)],
            query:  np.ndarray[float32, (d,)],
            k: int) -> np.ndarray[int64, (k,)]
```

Same api contract as `python/numerical_primitives/similarity.py::topk_cosine_numpy`: inputs
are already L2-normalised and C-contiguous, validated by the caller;
output is k row indices into `matrix`, highest score first; the
full score vector is never sorted, only the top k.

## Build

Own `CMakeLists.txt` here, built via `pip install` (CMake + pybind11 as build
requirements) or a standalone `cmake --build`, producing `_core` next to
`__init__.py` in `python/numerical_primitives/`. `_backend.py` discovers it with a plain
`from . import _core` and falls back to the numpy implementation if that
import fails so there is no hard dependency on this ever being built.

## Scope

CPU (SIMD, multi-threaded) first; CUDA as a second backend.
