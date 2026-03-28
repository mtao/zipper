/// @file add.hpp
/// @brief Sorted-merge sparse matrix addition (CSR + CSR or CSC + CSC).
/// @ingroup sparse_ops

#if !defined(ZIPPER_UTILS_SPARSE_ADD_HPP)
#define ZIPPER_UTILS_SPARSE_ADD_HPP

#include <stdexcept>
#include <vector>

#include <zipper/CSMatrix.hpp>
#include <zipper/CSRMatrix.hpp>

namespace zipper::utils::sparse {

/// @brief Compute C = alpha * A + beta * B for sparse matrices with the same
///        layout, using a sorted two-pointer merge per outer index.
///
/// Both A and B must have the same dimensions and the same LayoutPolicy
/// (both CSR or both CSC). The result has the same layout.
///
/// Complexity: O(nnz_A + nnz_B).
///
/// @param A      First sparse matrix.
/// @param B      Second sparse matrix.
/// @param alpha  Scalar multiplier for A (default 1).
/// @param beta   Scalar multiplier for B (default 1).
/// @return       C = alpha * A + beta * B as a new CSMatrix.
template <typename T, index_type Rows, index_type Cols, typename LayoutPolicy>
auto add(const CSMatrix<T, Rows, Cols, LayoutPolicy> &A,
         const CSMatrix<T, Rows, Cols, LayoutPolicy> &B,
         T alpha = T{1},
         T beta = T{1})
    -> CSMatrix<T, Rows, Cols, LayoutPolicy> {
    using OutMatrix = CSMatrix<T, Rows, Cols, LayoutPolicy>;
    using out_expression_type = typename OutMatrix::expression_type;
    using out_data_type = typename out_expression_type::compressed_data_type;

    if (A.extent(0) != B.extent(0) || A.extent(1) != B.extent(1)) {
        throw std::invalid_argument(
            "sparse::add: matrices must have the same dimensions");
    }

    const auto &cd_a = A.expression().compressed_data();
    const auto &cd_b = B.expression().compressed_data();
    const auto n_outer = cd_a.outer_size();

    // Upper bound on nnz of result.
    const auto max_nnz = cd_a.nnz() + cd_b.nnz();

    std::vector<index_type> out_indptr(n_outer + 1, 0);
    std::vector<index_type> out_indices;
    std::vector<T> out_values;
    out_indices.reserve(max_nnz);
    out_values.reserve(max_nnz);

    for (index_type outer = 0; outer < n_outer; ++outer) {
        out_indptr[outer] = static_cast<index_type>(out_indices.size());

        auto a_start = cd_a.m_indptr[outer];
        auto a_end = cd_a.m_indptr[outer + 1];
        auto b_start = cd_b.m_indptr[outer];
        auto b_end = cd_b.m_indptr[outer + 1];

        auto ai = a_start;
        auto bi = b_start;

        while (ai < a_end && bi < b_end) {
            auto col_a = cd_a.m_indices[ai];
            auto col_b = cd_b.m_indices[bi];

            if (col_a < col_b) {
                T val = alpha * cd_a.m_values[ai];
                if (val != T{0}) {
                    out_indices.push_back(col_a);
                    out_values.push_back(val);
                }
                ++ai;
            } else if (col_b < col_a) {
                T val = beta * cd_b.m_values[bi];
                if (val != T{0}) {
                    out_indices.push_back(col_b);
                    out_values.push_back(val);
                }
                ++bi;
            } else {
                // Same inner index — sum.
                T val = alpha * cd_a.m_values[ai] + beta * cd_b.m_values[bi];
                if (val != T{0}) {
                    out_indices.push_back(col_a);
                    out_values.push_back(val);
                }
                ++ai;
                ++bi;
            }
        }

        // Drain remaining entries from A.
        while (ai < a_end) {
            T val = alpha * cd_a.m_values[ai];
            if (val != T{0}) {
                out_indices.push_back(cd_a.m_indices[ai]);
                out_values.push_back(val);
            }
            ++ai;
        }

        // Drain remaining entries from B.
        while (bi < b_end) {
            T val = beta * cd_b.m_values[bi];
            if (val != T{0}) {
                out_indices.push_back(cd_b.m_indices[bi]);
                out_values.push_back(val);
            }
            ++bi;
        }
    }
    out_indptr[n_outer] = static_cast<index_type>(out_indices.size());

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
            out_extents_type(A.extent(0), A.extent(1))));
    }
}

} // namespace zipper::utils::sparse

#endif
