#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace numprims {

// matrix: row-major, n rows x d cols, each row already L2-normalised.
// query:  length d, already L2-normalised.
// Returns k row indices into matrix, in descending order of cosine similarity.
// Assumes valid input.
std::vector<int64_t> topk_cosine(std::span<const float> matrix, std::span<const float> query, int64_t k);

}  // namespace numprims
