#if !defined(ZIPPER_EXPRESSION_BINARY_MATRIXVECTORPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_MATRIXVECTORPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
// #include "zipper/expression/detail/intersect_nonzeros.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<1> B>
class MatrixVectorProduct;

}
template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<1> B>
struct detail::ExpressionTraits<binary::MatrixVectorProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using ATraits = detail::ExpressionTraits<A>;
    using BTraits = detail::ExpressionTraits<B>;
    using extents_type = extents<ATraits::extents_type::static_extent(0)>;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<1> B>
class MatrixVectorProduct
    : public BinaryExpressionBase<MatrixVectorProduct<A, B>, const A, const B> {
   public:
    using self_type = MatrixVectorProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, const A, const B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;
    using Base::Base;
    using Base::lhs;
    using Base::rhs;
    using extents_type = typename traits::extents_type;
    using lhs_traits = traits::ATraits;
    using rhs_traits = traits::BTraits;

    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    MatrixVectorProduct(const A& a, const B& b)
        requires(extents_traits::is_dynamic)
        : Base(a, b, extents_type{a.extent(0)}) {
        assert(a.extent(1) == b.extent(0));
    }

    MatrixVectorProduct(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {
        assert(a.extent(1) == b.extent(0));
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
};

template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<1> B>
MatrixVectorProduct(const A& a, const B& b)
    -> MatrixVectorProduct<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
