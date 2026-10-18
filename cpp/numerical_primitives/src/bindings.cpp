#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <stdexcept>

#include "numprims/topk_cosine.hpp"

namespace py = pybind11;

namespace {

py::array_t<int64_t> topk_cosine_py(py::array_t<float, py::array::c_style | py::array::forcecast> matrix,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> query,
                                     int64_t k) {
    
    // Validate args if this method is called directly.
    if (matrix.ndim() != 2 || query.ndim() != 1) {
        throw std::invalid_argument("matrix must be 2-D and query 1-D");
    }

    const int64_t n = matrix.shape(0);
    const int64_t d = matrix.shape(1);

    std::vector<int64_t> result = numprims::topk_cosine(std::span<const float>(matrix.data(), matrix.size()), std::span<const float>(query.data(), query.size()), k);

    // NOTE: copy here is O(k) much smaller than O(n*d). 
    // If this copy is to be avoided, we can return a view
    // but it will be tied to underlying vector
    py::array_t<int64_t> out(static_cast<py::ssize_t>(result.size()));
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Native top-k cosine similarity kernel (see cpp/numerical_primitives/README.md)";
    m.def("topk_cosine", &topk_cosine_py, py::arg("matrix"), py::arg("query"), py::arg("k"));
}
