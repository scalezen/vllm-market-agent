#include "numprims/topk_cosine.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace numprims {

namespace {

// Cosine similarity, dot product of two length-d float vectors 
float dot(std::span<const float> a,  std::span<const float> b) {
    
    auto t_lambda = [](auto a, auto b){ return a*b;};
    return std::transform_reduce(a.begin(), a.end(), b.begin(), 0.0f, std::plus<>(), std::multiplies<>());
}

} // namespace

std::vector<int64_t> topk_cosine(std::span<const float> matrix, std::span<const float> query, int64_t k) {

    
    int64_t m = matrix.size();
    int64_t d = query.size();

    if (m <= 0 || d <= 0) {
        throw std::invalid_argument("matrix must have at least one row and one column.");
    }
    
    int64_t n = matrix.size()/d;

    if (k < 1 || k > n) {
        throw std::invalid_argument("k must be between 1 and n");
    }

    // TODO: multithread with std::thread or OpenMP for large n
    std::vector<float> scores(n);
    for (int64_t i = 0; i < n; ++i) {
        scores[i] = dot(matrix.subspan(i * d, d), query);
    }

    // partition for the top k in O(n) and sort only k - O(n) + O(klogk) instead of O(nlogn)
    std::vector<int64_t> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), int64_t{0});

    auto score_of = [&scores](int64_t i){ return scores[i]; };
    std::ranges::nth_element(idx.begin(), idx.begin() + (k-1), idx.end(), std::ranges::greater(), score_of);

    idx.resize(k);

    std::ranges::sort(idx, std::ranges::greater(), score_of);
    return idx;
}

}  // namespace numprims
