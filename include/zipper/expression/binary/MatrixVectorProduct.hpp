#if !defined(ZIPPER_EXPRESSION_BINARY_MATRIXVECTORPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_MATRIXVECTORPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include <cassert>
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
// #include "zipper/expression/detail/intersect_nonzeros.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
class MatrixVectorProduct;

}
template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
struct detail::ExpressionTraits<binary::MatrixVectorProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using ATraits = detail::ExpressionTraits<std::decay_t<A>>;
    using BTraits = detail::ExpressionTraits<std::decay_t<B>>;
    using extents_type = extents<ATraits::extents_type::static_extent(0)>;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
class MatrixVectorProduct
    : public BinaryExpressionBase<MatrixVectorProduct<A, B>, A, B> {
   public:
    using self_type = MatrixVectorProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;
    using Base::lhs;
    using Base::rhs;
    using extents_type = typename traits::extents_type;
    using lhs_traits = traits::ATraits;
    using rhs_traits = traits::BTraits;

    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    template <typename U, typename V>
      requires std::constructible_from<typename Base::lhs_storage_type, U&&> &&
               std::constructible_from<typename Base::rhs_storage_type, V&&>
    MatrixVectorProduct(U&& a, V&& b)
        : Base(std::forward<U>(a), std::forward<V>(b)) {
        ZIPPER_ASSERT(lhs().extent(1) == rhs().extent(0));
    }

    constexpr auto extent([[maybe_unused]] rank_type i) const -> index_type {
        ZIPPER_ASSERT(i == 0);
        return lhs().extent(0);
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }
    value_type coeff(index_type a) const {
        value_type v = 0;
        // TODO: re-enable sparse optimizations once is_sparse is available
        // in all expression traits
        // constexpr bool lhs_sparse = lhs_traits::is_sparse(1);
        // constexpr bool rhs_sparse = rhs_traits::is_sparse(0);
        // if constexpr (lhs_sparse && rhs_sparse) {
        //     const auto& lnnz = lhs().template nonZeros<1>(a);
        //     const auto& rnnz = rhs().template nonZeros<0>(a);
        //     auto nnz = expression::detail::intersect_nonzeros(lnnz, rnnz);
        //
        //     for (const auto& j : nnz) {
        //         v += lhs()(a, j) * rhs()(j);
        //     }
        // } else if constexpr (lhs_sparse) {
        //     const auto& lnnz = lhs().template nonZeros<1>(a);
        //     for (const auto& j : lnnz) {
        //         v += lhs()(a, j) * rhs()(j);
        //     }
        // } else if constexpr (rhs_sparse) {
        //     const auto& rnnz = rhs().template nonZeros<0>(a);
        //     for (const auto& j : rnnz) {
        //         v += lhs()(a, j) * rhs()(j);
        //     }
        // } else {
            for (index_type j = 0; j < lhs().extent(1); ++j) {
                v += lhs()(a, j) * rhs()(j);
            }
        // }
        return v;
    }

    /// Recursively deep-copy children so the result owns all data.
    auto make_owned() const {
        auto owned_a = lhs().make_owned();
        auto owned_b = rhs().make_owned();
        return MatrixVectorProduct<decltype(owned_a), decltype(owned_b)>(
            std::move(owned_a), std::move(owned_b));
    }

};

template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<1> B>
MatrixVectorProduct(const A& a, const B& b)
    -> MatrixVectorProduct<const A&, const B&>;

}  // namespace binary
}  // namespace zipper::expression
#endif
