/// @file multiply.hpp
/// @brief Sparse matrix-matrix multiplication (SpGEMM) via Gustavson's
///        row-by-row algorithm.
/// @ingroup sparse_ops

#if !defined(ZIPPER_UTILS_SPARSE_MULTIPLY_HPP)
#define ZIPPER_UTILS_SPARSE_MULTIPLY_HPP

#include <algorithm>
#include <expected>
#include <string>
#include <vector>

#include <zipper/CSMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/utils/sparse/error.hpp>

namespace zipper::utils::sparse {

/// @brief Compute C = A * B for CSR sparse matrices using Gustavson's
///        row-by-row algorithm.
///
/// Given A (m x k, CSR) and B (k x n, CSR), produces C (m x n, CSR).
///
/// Algorithm:
///   For each row i of A:
///     For each nonzero (j, a_ij) in row i of A:
///       For each nonzero (l, b_jl) in row j of B:
///         Accumulate C(i, l) += a_ij * b_jl
///     Sort and emit the accumulated row.
///
/// The accumulation uses a dense scatter buffer of size n to avoid hash maps,
/// with a separate marker array to track which columns have been touched.
///
/// Complexity: O(flops + m*n) where flops = sum over rows of (nnz_A_row * nnz_B_rows).
/// The m*n term is for clearing the dense accumulator; for very sparse matrices
/// where n is large, a hash-based approach would be better, but the dense
/// scatter is simpler and faster for moderate n.
///
/// @param A  Left operand (m x k, CSR).
/// @param B  Right operand (k x n, CSR).
/// @return   C = A * B (m x n, CSR), or SparseError on dimension mismatch.
template <typename T, index_type M, index_type K1, index_type K2, index_type N>
auto spgemm(const CSRMatrix<T, M, K1> &A,
            const CSRMatrix<T, K2, N> &B)
    -> std::expected<CSRMatrix<T, M, N>, SparseError> {
    using OutMatrix = CSRMatrix<T, M, N>;
    using out_expression_type = typename OutMatrix::expression_type;
    using out_data_type = typename out_expression_type::compressed_data_type;

    const auto m = A.extent(0);
    const auto k = A.extent(1);
    const auto n = B.extent(1);

    if (k != B.extent(0)) {
        return std::unexpected(SparseError{
            SparseError::Kind::dimension_mismatch,
            "sparse::spgemm: inner dimensions must agree"});
    }

    const auto &cd_a = A.expression().compressed_data();
    const auto &cd_b = B.expression().compressed_data();

    // Dense accumulator + marker for scatter/gather approach.
    std::vector<T> acc(n, T{0});
    std::vector<bool> touched(n, false);
    std::vector<index_type> touched_cols;
    touched_cols.reserve(n);

    std::vector<index_type> out_indptr(m + 1, 0);
    std::vector<index_type> out_indices;
    std::vector<T> out_values;

    for (index_type i = 0; i < m; ++i) {
        out_indptr[i] = static_cast<index_type>(out_indices.size());
        touched_cols.clear();

        // Scatter: accumulate row i of C.
        const auto a_start = cd_a.m_indptr[i];
        const auto a_end = cd_a.m_indptr[i + 1];

        for (auto ak = a_start; ak < a_end; ++ak) {
            const auto j = cd_a.m_indices[ak];
            const auto a_ij = cd_a.m_values[ak];

            const auto b_start = cd_b.m_indptr[j];
            const auto b_end = cd_b.m_indptr[j + 1];

            for (auto bk = b_start; bk < b_end; ++bk) {
                const auto l = cd_b.m_indices[bk];
                if (!touched[l]) {
                    touched[l] = true;
                    touched_cols.push_back(l);
                }
                acc[l] += a_ij * cd_b.m_values[bk];
            }
        }

        // Gather: sort touched columns and emit nonzeros.
        std::sort(touched_cols.begin(), touched_cols.end());

        for (const auto col : touched_cols) {
            if (acc[col] != T{0}) {
                out_indices.push_back(col);
                out_values.push_back(acc[col]);
            }
            // Reset for next row.
            acc[col] = T{0};
            touched[col] = false;
        }
    }
    out_indptr[m] = static_cast<index_type>(out_indices.size());

    // Build output.
    out_data_type out_data;
    out_data.m_indptr = std::move(out_indptr);
    out_data.m_indices = std::move(out_indices);
    out_data.m_values = std::move(out_values);

    using out_extents_type = typename out_expression_type::extents_type;
    if constexpr (out_expression_type::IsStatic) {
        return OutMatrix(out_expression_type(std::move(out_data)));
    } else {
        return OutMatrix(out_expression_type(
            std::move(out_data),
            out_extents_type(m, n)));
    }
}

} // namespace zipper::utils::sparse

#endif
