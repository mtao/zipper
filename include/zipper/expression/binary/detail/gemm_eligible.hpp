#if !defined(ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_ELIGIBLE_HPP)
#define ZIPPER_EXPRESSION_BINARY_DETAIL_GEMM_ELIGIBLE_HPP

/// @file gemm_eligible.hpp
/// @brief Compile-time gate that decides whether a MatrixProduct may be routed
/// to the optimized blocked-GEMM kernel (gemm_kernel.hpp).
///
/// The blocked kernel packs both operands through their element accessor and
/// produces a result of a floating-point type. The TARGET only needs to be a
/// writable dense rank-2 expression; aliasing is handled by AssignHelper
/// before the kernel runs. `GemmEligible<A, B, To>` composes the
/// building blocks that guarantee this; when it fails (sparse, non-writable,
/// mixed or non-float scalars, or small static extents) the MatrixProduct's
/// constrained assign_to is removed by SFINAE and AssignHelper falls back to the
/// generic coefficient path.

#include <concepts>
#include <type_traits>

#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/LayoutPreference.hpp"
#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/storage/layout_types.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::binary::detail {

/// The element scalar of an expression, stripped of cv/ref.
template <typename E>
using gemm_scalar_t = typename zipper::expression::detail::ExpressionTraits<
    std::remove_cvref_t<E>>::element_type;

/// True when E's extents are (partly) dynamic — the case the generic path
/// handles poorly and the kernel targets. Small static matrices keep their
/// already-competitive unrolled path.
template <typename E>
inline constexpr bool gemm_is_dynamic_v = zipper::detail::ExtentsTraits<
    typename zipper::expression::detail::ExpressionTraits<
        std::remove_cvref_t<E>>::extents_type>::is_dynamic;

/// Rank of an expression's extents.
template <typename E>
inline constexpr auto gemm_rank_v =
    zipper::expression::detail::ExpressionTraits<
        std::remove_cvref_t<E>>::extents_type::rank();

/// True if the expression prefers a sparse layout (CSR/CSC) — excluded, since
/// sparse products have their own path and packing assumes dense access.
template <typename E>
inline constexpr bool gemm_is_sparse_v =
    zipper::detail::is_sparse_layout_preference_v<
        typename zipper::expression::detail::ExpressionTraits<
            std::remove_cvref_t<E>>::preferred_layout>;

/// A valid GEMM *source* operand: any dense (non-sparse) rank-2 expression with
/// element access. No contiguity/layout requirement — packing reads it through
/// `e(i,j)`, so transposed views, sub-blocks, and lazy nodes all qualify.
template <typename E>
concept DenseGemmSource =
    gemm_rank_v<E> == 2 && !gemm_is_sparse_v<E> &&
    requires(const std::remove_cvref_t<E>& ce) { ce(index_type{0}, index_type{0}); };

/// A valid GEMM *target*: any dense (non-sparse) rank-2 expression that is
/// writable through its element accessor — contiguous storage of either
/// layout, or a strided view / sub-block. The kernel writes tiles through
/// zipper slices, so no layout is privileged.
/// Writability uses the expression-level trait (is_writable == is_assignable),
/// and we additionally require that `e(i, j)` yields a real assignable lvalue.
template <typename E>
concept WritableDenseRank2Target =
    gemm_rank_v<E> == 2 && !gemm_is_sparse_v<E> &&
    zipper::expression::concepts::WritableExpression<std::remove_cvref_t<E>> &&
    requires(std::remove_cvref_t<E>& e) {
        e(index_type{0}, index_type{0}) =
            std::declval<gemm_scalar_t<E>>();
    };

/// Gate for routing C = A * B to the blocked kernel.
template <typename A, typename B, typename To>
concept GemmEligible =
    DenseGemmSource<A> && DenseGemmSource<B> && WritableDenseRank2Target<To> &&
    std::floating_point<gemm_scalar_t<A>> &&
    std::is_same_v<gemm_scalar_t<A>, gemm_scalar_t<B>> &&
    std::is_same_v<gemm_scalar_t<A>, gemm_scalar_t<To>> &&
    gemm_is_dynamic_v<A> && gemm_is_dynamic_v<B> && gemm_is_dynamic_v<To>;

}  // namespace zipper::expression::binary::detail

#endif
