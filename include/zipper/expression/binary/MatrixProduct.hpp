#if !defined(ZIPPER_EXPRESSION_BINARY_MATRIXPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_MATRIXPRODUCT_HPP

/// @file MatrixProduct.hpp
/// @brief Matrix-matrix product expression (rank-2 x rank-2 -> rank-2).
/// @ingroup expressions_binary sparsity
///
/// `MatrixProduct<A, B>` is a binary expression that lazily computes the
/// standard matrix product C = A * B, where A is m x k and B is k x n,
/// producing a result of shape m x n.
///
/// The `coeff(i, j)` method computes the dot product of row i of A and
/// column j of B.  When either operand satisfies `HasIndexSet` (e.g.,
/// TriangularView, Identity, or a ZeroAwareOperation result), the dot product
/// loop is restricted to the **intersection** of the non-zero index sets:
///
///   - Both have index sets: iterate over the smaller set and check
///     membership in the larger set.
///   - Only LHS has an index set: iterate LHS's `index_set<1>(row)`.
///   - Only RHS has an index set: iterate RHS's `index_set<0>(col)`.
///   - Neither: standard full-range loop over the shared dimension.
///
/// This zero-aware optimisation is especially effective for triangular and
/// identity matrices, where it reduces the inner loop from O(n) to O(row)
/// or O(1) respectively.
///
/// @code
///   auto A = Matrix<double, 3, 3>({{1,2,3},{4,5,6},{7,8,9}});
///   auto B = Matrix<double, 3, 3>({{9,8,7},{6,5,4},{3,2,1}});
///   auto C = MatrixBase(MatrixProduct(A.expression(), B.expression()));
///   // C(0,0) == 1*9 + 2*6 + 3*3 == 30
///
///   // With a triangular operand (zero-aware):
///   auto L = triangular_view<TriangularMode::Lower>(A);
///   auto D = MatrixBase(MatrixProduct(L, B.expression()));
///   // Only iterates over non-zero columns in each row of L.
/// @endcode
///
/// @see zipper::expression::detail::HasIndexSet — compile-time trait that
///      enables zero-aware dot product loops.
/// @see zipper::expression::detail::IndexSet — concept for range types
///      returned by index_set queries.
/// @see zipper::expression::binary::MatrixVectorProduct — analogous product
///      for matrix * vector (rank-2 x rank-1 -> rank-1).
/// @see zipper::expression::binary::ZeroAwareOperation — uses union semantics
///      for addition/subtraction (contrast with intersection here).
/// @see zipper::expression::unary::TriangularView — a primary source of
///      known-zero information for matrix products.

#include "BinaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<2> B>
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

template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<2> B>
struct detail::ExpressionDetail<binary::MatrixProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionDetail<A, B> {
    using _Base = binary::detail::DefaultBinaryExpressionDetail<A, B>;
    using ATraits = typename _Base::ATraits;
    using BTraits = typename _Base::BTraits;
    using ConvertExtentsUtil =
        _detail_matprod::coeffwise_extents_values<typename ATraits::extents_type,
                                                  typename BTraits::extents_type>;
};

template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<2> B>
struct detail::ExpressionTraits<binary::MatrixProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B>
{
    using _Detail = detail::ExpressionDetail<binary::MatrixProduct<A, B>>;
    using extents_type = typename _Detail::ConvertExtentsUtil::product_extents_type;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<2> B>
class MatrixProduct : public BinaryExpressionBase<MatrixProduct<A, B>, A, B> {
   public:
    using self_type = MatrixProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using value_type = traits::value_type;
    using Base::lhs;
    using Base::rhs;
    using lhs_traits = typename detail_type::ATraits;
    using rhs_traits = typename detail_type::BTraits;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    template <typename U, typename V>
      requires std::constructible_from<typename Base::lhs_storage_type, U&&> &&
               std::constructible_from<typename Base::rhs_storage_type, V&&>
    MatrixProduct(U&& a, V&& b)
        : Base(std::forward<U>(a), std::forward<V>(b)) {
        ZIPPER_ASSERT(lhs().extent(1) == rhs().extent(0));
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
        using zipper::expression::detail::HasIndexSet;
        using zipper::expression::detail::HasForEach;
        constexpr bool lhs_has_zeros = HasIndexSet<std::decay_t<A>>;
        constexpr bool rhs_has_zeros = HasIndexSet<std::decay_t<B>>;

        if constexpr (lhs_has_zeros && rhs_has_zeros) {
            auto lr = lhs().template index_set<1>(a);
            auto rr = rhs().template index_set<0>(b);
            if (lr.size() <= rr.size()) {
                auto body = [&](index_type j) {
                    if (rr.contains(j)) {
                        v += lhs()(a, j) * rhs()(j, b);
                    }
                };
                if constexpr (HasForEach<decltype(lr)>) {
                    lr.for_each(body);
                } else {
                    for (auto j : lr) { body(j); }
                }
            } else {
                auto body = [&](index_type j) {
                    if (lr.contains(j)) {
                        v += lhs()(a, j) * rhs()(j, b);
                    }
                };
                if constexpr (HasForEach<decltype(rr)>) {
                    rr.for_each(body);
                } else {
                    for (auto j : rr) { body(j); }
                }
            }
        } else if constexpr (lhs_has_zeros) {
            auto lr = lhs().template index_set<1>(a);
            auto body = [&](index_type j) {
                v += lhs()(a, j) * rhs()(j, b);
            };
            if constexpr (HasForEach<decltype(lr)>) {
                lr.for_each(body);
            } else {
                for (auto j : lr) { body(j); }
            }
        } else if constexpr (rhs_has_zeros) {
            auto rr = rhs().template index_set<0>(b);
            auto body = [&](index_type j) {
                v += lhs()(a, j) * rhs()(j, b);
            };
            if constexpr (HasForEach<decltype(rr)>) {
                rr.for_each(body);
            } else {
                for (auto j : rr) { body(j); }
            }
        } else {
            for (index_type j = 0; j < lhs().extent(1); ++j) {
                v += lhs()(a, j) * rhs()(j, b);
            }
        }
        return v;
    }

    /// Recursively deep-copy children so the result owns all data.
    auto make_owned() const {
        auto owned_a = lhs().make_owned();
        auto owned_b = rhs().make_owned();
        return MatrixProduct<decltype(owned_a), decltype(owned_b)>(
            std::move(owned_a), std::move(owned_b));
    }

};

template <zipper::concepts::RankedExpression<2> A,
          zipper::concepts::RankedExpression<2> B>
MatrixProduct(const A& a, const B& b) -> MatrixProduct<const A&, const B&>;

}  // namespace binary
}  // namespace zipper::expression
#endif
