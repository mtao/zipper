/// @file transpose.hpp
/// @brief O(nnz) direct CSR <-> CSC transpose without COO intermediary.
/// @ingroup sparse_ops

#if !defined(ZIPPER_UTILS_SPARSE_TRANSPOSE_HPP)
#define ZIPPER_UTILS_SPARSE_TRANSPOSE_HPP

#include <vector>

#include <zipper/CSMatrix.hpp>
#include <zipper/CSRMatrix.hpp>

namespace zipper::utils::sparse {

/// @brief Transpose a compressed sparse matrix in O(nnz) time.
///
/// Given a CSMatrix representing A, returns a CSMatrix (same layout policy)
/// representing A^T. The dimensions are swapped: (m x n) -> (n x m).
///
/// The algorithm uses count-scatter: count entries per inner index of the
/// input (which becomes the outer index of the output), prefix-sum to build
/// indptr, then scatter values and indices in one pass.
///
/// This avoids the O(nnz log nnz) path through COO used by `as_csr()`/`as_csc()`.
///
/// @param A  The input sparse matrix (CSR or CSC).
/// @return   The transposed matrix with the same layout and swapped dimensions.
template <typename T, index_type Rows, index_type Cols, typename LayoutPolicy>
auto transpose(const CSMatrix<T, Rows, Cols, LayoutPolicy> &A)
    -> CSMatrix<T, Cols, Rows, LayoutPolicy> {
    using OutMatrix = CSMatrix<T, Cols, Rows, LayoutPolicy>;
    using out_expression_type = typename OutMatrix::expression_type;
    using out_data_type = typename out_expression_type::compressed_data_type;

    const auto &cd = A.expression().compressed_data();
    const auto n_outer = cd.outer_size();
    const auto nnz = cd.nnz();

    // Determine the inner dimension size (number of rows in output outer).
    // For CSR input: outer = rows, inner = cols -> output outer = cols = A.extent(1)
    // For CSC input: outer = cols, inner = rows -> output outer = rows = A.extent(0)
    const index_type n_inner = [&]() {
        if constexpr (std::is_same_v<LayoutPolicy, storage::layout_right>) {
            return A.extent(1);  // CSR: inner dim = cols
        } else {
            return A.extent(0);  // CSC: inner dim = rows
        }
    }();

    if (nnz == 0) {
        // Empty matrix — just swap dimensions.
        if constexpr (OutMatrix::is_static) {
            return OutMatrix{};
        } else {
            return OutMatrix(A.extent(1), A.extent(0));
        }
    }

    // Step 1: Count entries per inner index (will become outer indices in output).
    std::vector<index_type> counts(n_inner + 1, 0);
    for (size_t k = 0; k < nnz; ++k) {
        ++counts[cd.m_indices[k] + 1];
    }

    // Step 2: Prefix sum to build output indptr.
    for (index_type i = 1; i <= n_inner; ++i) {
        counts[i] += counts[i - 1];
    }

    // Step 3: Scatter — fill output indices and values.
    std::vector<index_type> out_indices(nnz);
    std::vector<T> out_values(nnz);

    // Use a working copy of counts as write cursors.
    std::vector<index_type> cursors(counts.begin(), counts.end() - 1);

    for (index_type outer = 0; outer < n_outer; ++outer) {
        const auto row_start = cd.m_indptr[outer];
        const auto row_end = cd.m_indptr[outer + 1];
        for (auto k = row_start; k < row_end; ++k) {
            const auto inner = cd.m_indices[k];
            const auto pos = cursors[inner]++;
            out_indices[pos] = outer;
            out_values[pos] = cd.m_values[k];
        }
    }

    // Step 4: Build the output compressed data.
    out_data_type out_data;
    out_data.m_indptr = std::move(counts);
    out_data.m_indices = std::move(out_indices);
    out_data.m_values = std::move(out_values);

    // Build extents: swap rows and cols.
    using out_extents_type = typename out_expression_type::extents_type;
    if constexpr (out_expression_type::IsStatic) {
        return OutMatrix(out_expression_type(std::move(out_data)));
    } else {
        return OutMatrix(out_expression_type(
            std::move(out_data),
            out_extents_type(A.extent(1), A.extent(0))));
    }
}

} // namespace zipper::utils::sparse

#endif
