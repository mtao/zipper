/// @file diagonal.hpp
/// @brief Sparse diagonal extraction and construction.
/// @ingroup sparse_ops

#if !defined(ZIPPER_UTILS_SPARSE_DIAGONAL_HPP)
#define ZIPPER_UTILS_SPARSE_DIAGONAL_HPP

#include <algorithm>
#include <vector>

#include <zipper/CSMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/Vector.hpp>

namespace zipper::utils::sparse {

/// @brief Extract the diagonal of a sparse matrix as a dense vector.
///
/// For an m x n matrix, the diagonal has min(m, n) entries.
/// Missing diagonal entries are treated as zero.
///
/// Complexity: O(nnz) — scans each row for the diagonal entry.
///
/// @param A  The input sparse matrix.
/// @return   Dense vector of diagonal entries.
template <typename T, index_type Rows, index_type Cols, typename LayoutPolicy>
auto extract_diagonal(const CSMatrix<T, Rows, Cols, LayoutPolicy> &A)
    -> Vector<T, dynamic_extent> {
    const auto m = A.extent(0);
    const auto n = A.extent(1);
    const auto diag_size = std::min(m, n);

    Vector<T, dynamic_extent> diag(diag_size);
    for (index_type i = 0; i < diag_size; ++i) {
        diag(i) = T{0};
    }

    const auto &cd = A.expression().compressed_data();

    if constexpr (std::is_same_v<LayoutPolicy, storage::layout_right>) {
        // CSR: outer = rows, inner = cols.
        // For each row i, binary-search for column i.
        for (index_type i = 0; i < diag_size; ++i) {
            const auto row_start = cd.m_indptr[i];
            const auto row_end = cd.m_indptr[i + 1];
            auto it = std::lower_bound(
                cd.m_indices.begin() + row_start,
                cd.m_indices.begin() + row_end,
                i);
            if (it != cd.m_indices.begin() + row_end && *it == i) {
                auto pos = static_cast<size_t>(it - cd.m_indices.begin());
                diag(i) = cd.m_values[pos];
            }
        }
    } else {
        // CSC: outer = cols, inner = rows.
        // For each column j, binary-search for row j.
        for (index_type j = 0; j < diag_size; ++j) {
            const auto col_start = cd.m_indptr[j];
            const auto col_end = cd.m_indptr[j + 1];
            auto it = std::lower_bound(
                cd.m_indices.begin() + col_start,
                cd.m_indices.begin() + col_end,
                j);
            if (it != cd.m_indices.begin() + col_end && *it == j) {
                auto pos = static_cast<size_t>(it - cd.m_indices.begin());
                diag(j) = cd.m_values[pos];
            }
        }
    }

    return diag;
}

/// @brief Construct a diagonal CSR matrix from a dense vector.
///
/// Creates an n x n CSR matrix with the given vector as the diagonal and
/// zeros everywhere else.
///
/// Complexity: O(n).
///
/// @param d  Dense vector of diagonal entries.
/// @return   n x n diagonal CSR matrix.
template <typename Derived>
    requires concepts::Vector<Derived>
auto from_diagonal(const Derived &d)
    -> CSRMatrix<typename std::decay_t<Derived>::value_type,
                 dynamic_extent, dynamic_extent> {
    using T = typename std::decay_t<Derived>::value_type;
    using OutMatrix = CSRMatrix<T, dynamic_extent, dynamic_extent>;
    using out_expression_type = typename OutMatrix::expression_type;
    using out_data_type = typename out_expression_type::compressed_data_type;

    const auto n = d.extent(0);

    // Count non-zero diagonal entries.
    index_type nnz = 0;
    for (index_type i = 0; i < n; ++i) {
        if (d(i) != T{0}) { ++nnz; }
    }

    std::vector<index_type> indptr(n + 1, 0);
    std::vector<index_type> indices;
    std::vector<T> values;
    indices.reserve(nnz);
    values.reserve(nnz);

    for (index_type i = 0; i < n; ++i) {
        indptr[i] = static_cast<index_type>(indices.size());
        if (d(i) != T{0}) {
            indices.push_back(i);
            values.push_back(d(i));
        }
    }
    indptr[n] = static_cast<index_type>(indices.size());

    out_data_type out_data;
    out_data.m_indptr = std::move(indptr);
    out_data.m_indices = std::move(indices);
    out_data.m_values = std::move(values);

    using out_extents_type = typename out_expression_type::extents_type;
    return OutMatrix(out_expression_type(
        std::move(out_data),
        out_extents_type(n, n)));
}

/// @brief Construct a sparse identity matrix of given size.
///
/// @param n  Dimension of the identity matrix.
/// @return   n x n CSR identity matrix.
template <typename T>
auto sparse_identity(index_type n)
    -> CSRMatrix<T, dynamic_extent, dynamic_extent> {
    using OutMatrix = CSRMatrix<T, dynamic_extent, dynamic_extent>;
    using out_expression_type = typename OutMatrix::expression_type;
    using out_data_type = typename out_expression_type::compressed_data_type;

    std::vector<index_type> indptr(n + 1);
    std::vector<index_type> indices(n);
    std::vector<T> values(n, T{1});

    for (index_type i = 0; i <= n; ++i) {
        indptr[i] = i;
    }
    for (index_type i = 0; i < n; ++i) {
        indices[i] = i;
    }

    out_data_type out_data;
    out_data.m_indptr = std::move(indptr);
    out_data.m_indices = std::move(indices);
    out_data.m_values = std::move(values);

    using out_extents_type = typename out_expression_type::extents_type;
    return OutMatrix(out_expression_type(
        std::move(out_data),
        out_extents_type(n, n)));
}

} // namespace zipper::utils::sparse

#endif
