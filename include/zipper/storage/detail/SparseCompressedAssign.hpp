/// @file SparseCompressedAssign.hpp
/// @brief Compile-time detection and optimized assignment specializations
///        for sparse-to-sparse operations (SpGEMM, addition, transpose).
/// @ingroup storage sparse_ops
///
/// These specializations are used by SparseCompressedAccessor::assign()
/// to bypass the generic COO-intermediary path when the source expression
/// is a sparse matrix product, sparse addition/subtraction, or transpose.
///
/// Each specialization operates directly on the compressed data arrays
/// (indptr, indices, values) for optimal performance:
///   - SpGEMM: Gustavson's row-by-row algorithm, O(flops)
///   - Addition: sorted two-pointer merge, O(nnz_A + nnz_B)
///   - Transpose: count-scatter, O(nnz)

#if !defined(ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDASSIGN_HPP)
#define ZIPPER_STORAGE_DETAIL_SPARSECOMPRESSEDASSIGN_HPP

#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

#include "zipper/expression/binary/MatrixProduct.hpp"
#include "zipper/expression/binary/ZeroAwareOperation.hpp"
#include "zipper/expression/unary/Swizzle.hpp"
#include "zipper/storage/detail/SparseCompressedData.hpp"

namespace zipper::storage {
// Forward declaration of the accessor itself (needed by traits).
template <typename, typename, typename, typename>
class SparseCompressedAccessor;
} // namespace zipper::storage

namespace zipper::storage::detail {

// ═══════════════════════════════════════════════════════════════════════
// Type detection traits
// ═══════════════════════════════════════════════════════════════════════

/// Strip const and reference qualifiers.
template <typename T>
using clean_t = std::remove_const_t<std::remove_reference_t<T>>;

/// Detect whether a type is a SparseCompressedAccessor with owned storage.
template <typename T>
struct is_owned_sparse_compressed : std::false_type {};

template <typename VT, typename E, typename LP>
struct is_owned_sparse_compressed<
    SparseCompressedAccessor<VT, E, LP, OwnedStorage>> : std::true_type {
    using layout_policy = LP;
};

template <typename VT, typename E, typename LP>
struct is_owned_sparse_compressed<
    SparseCompressedAccessor<const VT, E, LP, OwnedStorage>> : std::true_type {
    using layout_policy = LP;
};

/// Check if a (possibly const&) type is an owned sparse compressed accessor.
template <typename T>
inline constexpr bool is_owned_sparse_v =
    is_owned_sparse_compressed<clean_t<T>>::value;

// ─── MatrixProduct<SCA, SCA> detection ────────────────────────────────

template <typename V, typename TargetLayout>
struct is_sparse_matmul : std::false_type {};

template <typename A, typename B, typename TargetLayout>
struct is_sparse_matmul<expression::binary::MatrixProduct<A, B>, TargetLayout> {
    static constexpr bool children_sparse =
        is_owned_sparse_v<A> && is_owned_sparse_v<B>;
    static constexpr bool same_layout = [] {
        if constexpr (children_sparse) {
            return std::is_same_v<typename is_owned_sparse_compressed<
                                      clean_t<A>>::layout_policy,
                                  TargetLayout>
                   && std::is_same_v<typename is_owned_sparse_compressed<
                                         clean_t<B>>::layout_policy,
                                     TargetLayout>;
        } else {
            return false;
        }
    }();
    static constexpr bool value = children_sparse && same_layout;
};

template <typename V, typename TargetLayout>
inline constexpr bool is_sparse_matmul_v =
    is_sparse_matmul<clean_t<V>, TargetLayout>::value;

// ─── ZeroAwareOperation<SCA, SCA, plus/minus> detection ───────────────

template <typename V, typename TargetLayout>
struct is_sparse_add : std::false_type {};

template <typename A, typename B, typename Op, typename TargetLayout>
struct is_sparse_add<expression::binary::ZeroAwareOperation<A, B, Op>,
                     TargetLayout> {
  private:
    static constexpr bool children_sparse_ =
        is_owned_sparse_v<A> && is_owned_sparse_v<B>;
    static constexpr bool same_layout_ = [] {
        if constexpr (children_sparse_) {
            return std::is_same_v<typename is_owned_sparse_compressed<
                                      clean_t<A>>::layout_policy,
                                  TargetLayout>
                   && std::is_same_v<typename is_owned_sparse_compressed<
                                         clean_t<B>>::layout_policy,
                                     TargetLayout>;
        } else {
            return false;
        }
    }();
    using value_type_ = typename clean_t<A>::element_type;

  public:
    static constexpr bool is_plus = std::is_same_v<Op, std::plus<value_type_>>;
    static constexpr bool is_minus =
        std::is_same_v<Op, std::minus<value_type_>>;
    static constexpr bool value =
        children_sparse_ && same_layout_ && (is_plus || is_minus);
};

template <typename V, typename TargetLayout>
inline constexpr bool is_sparse_add_v =
    is_sparse_add<clean_t<V>, TargetLayout>::value;

// ─── Swizzle<SCA, 1, 0> (transpose) detection ────────────────────────

template <typename V, typename TargetLayout>
struct is_sparse_transpose : std::false_type {};

template <typename E, typename TargetLayout>
struct is_sparse_transpose<expression::unary::Swizzle<E, 1, 0>, TargetLayout> {
    static constexpr bool child_sparse = is_owned_sparse_v<E>;
    // The child's layout must be the OPPOSITE of the target layout.
    // This is the natural path: eval() applies flip_layout_t, so a CSR
    // child's transpose expression has PreferCSC and materializes into a
    // CSC target (layout_left).  The count-scatter algorithm reads CSR
    // compressed data and writes CSC compressed data (or vice versa).
    static constexpr bool flipped_layout = [] {
        if constexpr (child_sparse) {
            return !std::is_same_v<
                typename is_owned_sparse_compressed<clean_t<E>>::layout_policy,
                TargetLayout>;
        } else {
            return false;
        }
    }();
    static constexpr bool value = child_sparse && flipped_layout;
};

template <typename V, typename TargetLayout>
inline constexpr bool is_sparse_transpose_v =
    is_sparse_transpose<clean_t<V>, TargetLayout>::value;

// ═══════════════════════════════════════════════════════════════════════
// Algorithm implementations
// ═══════════════════════════════════════════════════════════════════════

struct SparseCompressedAssign {
    // ─── SpGEMM: Gustavson's row-by-row algorithm ─────────────────────
    //
    // Computes C = A * B directly in compressed format.
    // Layout-agnostic: works in terms of outer/inner dimensions.
    // For CSR: outer = rows, inner = cols.
    // For CSC: outer = cols, inner = rows.
    //
    // @param out_data   Target compressed data (written to).
    // @param expr       The MatrixProduct<A, B> expression.
    template <typename DataType, typename V>
    static void assign_spgemm(DataType &out_data, const V &expr) {
        const auto &cd_a = expr.lhs().compressed_data();
        const auto &cd_b = expr.rhs().compressed_data();

        using value_type = typename DataType::value_type;

        const auto out_outer = cd_a.outer_size();

        // Inner dimension size of B = output inner dimension.
        // For CSR: inner = cols → cd_b's inner count = B.extent(1)
        //   but we can get this from B's outer_size of the transpose,
        //   or more directly: for same-layout matmul, the output inner
        //   dim equals the inner dim of B, which is the max index in
        //   B's indices + 1. But the cleanest way is to use the
        //   expression's extents.
        const auto &rhs_acc = expr.rhs();
        const index_type out_inner = [&]() {
            using rhs_type = clean_t<decltype(rhs_acc)>;
            if constexpr (rhs_type::is_csr) {
                return rhs_acc.extent(1);
            } else {
                return rhs_acc.extent(0);
            }
        }();

        // Dense accumulator + marker for scatter/gather approach.
        std::vector<value_type> acc(out_inner, value_type{0});
        std::vector<bool> touched(out_inner, false);
        std::vector<index_type> touched_inners;
        touched_inners.reserve(out_inner);

        std::vector<index_type> result_indptr(out_outer + 1, 0);
        std::vector<index_type> result_indices;
        std::vector<value_type> result_values;

        for (index_type i = 0; i < out_outer; ++i) {
            result_indptr[i] = static_cast<index_type>(result_indices.size());
            touched_inners.clear();

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
                        touched_inners.push_back(l);
                    }
                    acc[l] += a_ij * cd_b.m_values[bk];
                }
            }

            // Gather: sort touched indices and emit nonzeros.
            std::sort(touched_inners.begin(), touched_inners.end());

            for (const auto idx : touched_inners) {
                if (acc[idx] != value_type{0}) {
                    result_indices.push_back(idx);
                    result_values.push_back(acc[idx]);
                }
                acc[idx] = value_type{0};
                touched[idx] = false;
            }
        }
        result_indptr[out_outer] =
            static_cast<index_type>(result_indices.size());

        out_data.m_indptr = std::move(result_indptr);
        out_data.m_indices = std::move(result_indices);
        out_data.m_values = std::move(result_values);
    }

    // ─── Sparse Addition: sorted two-pointer merge ────────────────────
    //
    // Computes C = A + B (or A - B) directly in compressed format.
    // Both operands must have the same layout.
    //
    // @param out_data   Target compressed data (written to).
    // @param expr       The ZeroAwareOperation<A, B, Op> expression.
    template <typename DataType, typename V>
    static void assign_add(DataType &out_data, const V &expr) {
        const auto &cd_a = expr.lhs().compressed_data();
        const auto &cd_b = expr.rhs().compressed_data();

        using value_type = typename DataType::value_type;

        // Determine sign for RHS based on operation type.
        using expr_type = clean_t<V>;
        using lhs_layout = typename is_owned_sparse_compressed<
            clean_t<decltype(expr.lhs())>>::layout_policy;
        using trait = is_sparse_add<expr_type, lhs_layout>;
        constexpr value_type beta =
            trait::is_minus ? value_type{-1} : value_type{1};

        const auto n_outer = cd_a.outer_size();
        const auto max_nnz = cd_a.nnz() + cd_b.nnz();

        std::vector<index_type> result_indptr(n_outer + 1, 0);
        std::vector<index_type> result_indices;
        std::vector<value_type> result_values;
        result_indices.reserve(max_nnz);
        result_values.reserve(max_nnz);

        for (index_type outer = 0; outer < n_outer; ++outer) {
            result_indptr[outer] =
                static_cast<index_type>(result_indices.size());

            auto a_pos = cd_a.m_indptr[outer];
            const auto a_end = cd_a.m_indptr[outer + 1];
            auto b_pos = cd_b.m_indptr[outer];
            const auto b_end = cd_b.m_indptr[outer + 1];

            while (a_pos < a_end && b_pos < b_end) {
                const auto idx_a = cd_a.m_indices[a_pos];
                const auto idx_b = cd_b.m_indices[b_pos];

                if (idx_a < idx_b) {
                    value_type val = cd_a.m_values[a_pos];
                    if (val != value_type{0}) {
                        result_indices.push_back(idx_a);
                        result_values.push_back(val);
                    }
                    ++a_pos;
                } else if (idx_b < idx_a) {
                    value_type val = beta * cd_b.m_values[b_pos];
                    if (val != value_type{0}) {
                        result_indices.push_back(idx_b);
                        result_values.push_back(val);
                    }
                    ++b_pos;
                } else {
                    // Same inner index — combine.
                    value_type val =
                        cd_a.m_values[a_pos] + beta * cd_b.m_values[b_pos];
                    if (val != value_type{0}) {
                        result_indices.push_back(idx_a);
                        result_values.push_back(val);
                    }
                    ++a_pos;
                    ++b_pos;
                }
            }

            // Drain remaining entries from A.
            while (a_pos < a_end) {
                value_type val = cd_a.m_values[a_pos];
                if (val != value_type{0}) {
                    result_indices.push_back(cd_a.m_indices[a_pos]);
                    result_values.push_back(val);
                }
                ++a_pos;
            }

            // Drain remaining entries from B.
            while (b_pos < b_end) {
                value_type val = beta * cd_b.m_values[b_pos];
                if (val != value_type{0}) {
                    result_indices.push_back(cd_b.m_indices[b_pos]);
                    result_values.push_back(val);
                }
                ++b_pos;
            }
        }
        result_indptr[n_outer] = static_cast<index_type>(result_indices.size());

        out_data.m_indptr = std::move(result_indptr);
        out_data.m_indices = std::move(result_indices);
        out_data.m_values = std::move(result_values);
    }

    // ─── Sparse Transpose: CSR↔CSC direct copy ─────────────────────────
    //
    // For cross-format transpose (CSR→CSC or CSC→CSR), the compressed
    // data of A is literally the compressed data of A^T in the other
    // format:
    //   - CSR row pointers become CSC column pointers
    //   - CSR column indices become CSC row indices
    //   - Values are unchanged
    //
    // This is because CSR(m×n) stores data sorted by (row, col), which
    // is exactly how CSC(n×m) stores data sorted by (col, row) when
    // "col" of A^T = "row" of A.
    //
    // Complexity: O(nnz) for the copy, no sorting or counting needed.
    //
    // @param out_data   Target compressed data (written to).
    // @param expr       The Swizzle<E, 1, 0> expression.
    template <typename DataType, typename V>
    static void assign_transpose(DataType &out_data, const V &expr) {
        const auto &child = expr.expression();
        const auto &cd = child.compressed_data();

        // Direct copy: the compressed representation is identical under
        // format swap (CSR↔CSC).
        out_data.m_indptr = cd.m_indptr;
        out_data.m_indices = cd.m_indices;
        out_data.m_values = cd.m_values;
    }
};

} // namespace zipper::storage::detail

#endif
