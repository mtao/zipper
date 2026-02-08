#if !defined(ZIPPER_EXPRESSION_BINARY_MATRIXPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_MATRIXPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
// #include "zipper/expression/detail/intersect_nonzeros.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<2> B>
class MatrixProduct;

}
namespace _detail_matprod {

template <typename, typename>
struct coeffwise_extents_values;
template <index_type AR, index_type AC, index_type BR, index_type BC>
struct coeffwise_extents_values<extents<AR, AC>, extents<BR, BC>> {
    using product_extents_type = zipper::extents<AR, BC>;

    constexpr static product_extents_type merge(const extents<AR, AC>& a,
                                                const extents<BR, BC>& b) {
        if constexpr (AR == std::dynamic_extent && BC == std::dynamic_extent) {
            return product_extents_type(a.extent(0), b.extent(1));
        } else if constexpr (AR == std::dynamic_extent) {
            return product_extents_type(a.extent(0));
        } else if constexpr (BC == std::dynamic_extent) {
            return product_extents_type(b.extent(1));
        } else {
            return {};
        }
    }
};
}  // namespace _detail_matprod

template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<2> B>
struct detail::ExpressionTraits<binary::MatrixProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B>
{
    using ATraits = detail::ExpressionTraits<A>;
    using BTraits = detail::ExpressionTraits<B>;
    using ConvertExtentsUtil =
        _detail_matprod::coeffwise_extents_values<typename ATraits::extents_type,
                                                  typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::product_extents_type;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<2> B>
class MatrixProduct : public BinaryExpressionBase<MatrixProduct<A, B>, const A, const B> {
   public:
    using self_type = MatrixProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, const A, const B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;
    using Base::lhs;
    using Base::rhs;
    using lhs_traits = traits::ATraits;
    using rhs_traits = traits::BTraits;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    MatrixProduct(const A& a, const B& b)
        : Base(a, b) {
        assert(a.extent(1) == b.extent(0));
    }

    constexpr auto extent(rank_type i) const -> index_type {
        if (i == 0) {
            return lhs().extent(0);
        } else {
            return rhs().extent(1);
        }
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

    value_type coeff(index_type a, index_type b) const {
        value_type v = 0;
        // TODO: re-enable sparse optimizations once is_sparse is available
        // in all expression traits
        // constexpr bool lhs_sparse = lhs_traits::is_sparse(1);
        // constexpr bool rhs_sparse = rhs_traits::is_sparse(0);
        // if constexpr (lhs_sparse && rhs_sparse) {
        //     const auto& lnnz = lhs().template nonZeros<1>(a, b);
        //     const auto& rnnz = rhs().template nonZeros<0>(a, b);
        //     auto nnz = expression::detail::intersect_nonzeros(lnnz, rnnz);
        //
        //     for (const auto& j : nnz) {
        //         v += lhs()(a, j) * rhs()(j, b);
        //     }
        // } else if constexpr (lhs_sparse) {
        //     const auto& lnnz = lhs().template nonZeros<1>(a, b);
        //     for (const auto& j : lnnz) {
        //         v += lhs()(a, j) * rhs()(j, b);
        //     }
        // } else if constexpr (rhs_sparse) {
        //     const auto& rnnz = rhs().template nonZeros<0>(a, b);
        //     for (const auto& j : rnnz) {
        //         v += lhs()(a, j) * rhs()(j, b);
        //     }
        // } else {
            for (index_type j = 0; j < lhs().extent(1); ++j) {
                v += lhs()(a, j) * rhs()(j, b);
            }
        // }
        return v;
    }

};

template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<2> B>
MatrixProduct(const A& a, const B& b) -> MatrixProduct<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
