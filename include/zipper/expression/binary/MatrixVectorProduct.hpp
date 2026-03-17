#if !defined(ZIPPER_EXPRESSION_BINARY_MATRIXVECTORPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_MATRIXVECTORPRODUCT_HPP

/// @file MatrixVectorProduct.hpp
/// @brief Matrix-vector product expression (rank-2 x rank-1 -> rank-1).
/// @ingroup expressions_binary sparsity
///
/// `MatrixVectorProduct<A, B>` lazily computes y = A * x, where A is m x n
/// and x is n-dimensional, producing an m-dimensional result.
///
/// The `coeff(i)` method computes the dot product of row i of A with x.
/// When either operand satisfies `HasIndexSet`, the dot product loop is
/// restricted to the intersection of the non-zero index sets (same strategy as
/// MatrixProduct):
///
///   - Both have index sets: iterate over the smaller set, checking
///     membership in the larger.
///   - Only LHS (matrix) has an index set: iterate LHS's `index_set<1>(row)`.
///   - Only RHS (vector) has an index set: iterate RHS's `index_set<0>()`.
///   - Neither: standard full-range loop.
///
/// This is particularly effective for triangular matrices multiplied by
/// sparse vectors (e.g., unit vectors), where it reduces per-row work from
/// O(n) to O(1).
///
/// @code
///   auto A = Matrix<double, 3, 3>({{1,0,0},{2,3,0},{4,5,6}});
///   auto L = triangular_view<TriangularMode::Lower>(A);
///   auto x = Vector<double, 3>({1, 2, 3});
///   auto y = VectorBase(MatrixVectorProduct(L, x.expression()));
///   // y(0) == 1*1 = 1, y(1) == 2*1 + 3*2 = 8, y(2) == 4*1 + 5*2 + 6*3 = 32
///
///   // With a unit vector (zero-aware):
///   auto e1 = nullary::unit_vector<double, 3, 1>();
///   auto col1 = VectorBase(MatrixVectorProduct(A.expression(), e1));
///   // Only accesses A(i, 1) for each row — O(1) per dot product.
/// @endcode
///
/// @see zipper::expression::detail::HasIndexSet — compile-time trait that
///      enables zero-aware dot product loops.
/// @see zipper::expression::detail::IndexSet — concept for range types
///      used in index_set queries.
/// @see zipper::expression::binary::MatrixProduct — analogous product for
///      matrix * matrix (rank-2 x rank-2 -> rank-2).
/// @see zipper::expression::binary::ZeroAwareOperation — uses union semantics
///      for addition/subtraction.
/// @see zipper::expression::nullary::Unit — a rank-1 expression with a single
///      non-zero, ideal for zero-aware matvec.
/// @see zipper::expression::unary::TriangularView — triangular matrix view
///      whose index_set restricts the dot product loop.

#include "BinaryExpressionBase.hpp"
#include <cassert>
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
class MatrixVectorProduct;

}

/// Implementation details for MatrixVectorProduct expressions.
///
/// Holds child traits aliases.  MatrixVectorProduct computes extents_type
/// directly (rank-1 from the matrix's row extent) without a ConvertExtentsUtil.
template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
struct detail::ExpressionDetail<binary::MatrixVectorProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionDetail<A, B> {
};

template <zipper::concepts::QualifiedRankedExpression<2> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
struct detail::ExpressionTraits<binary::MatrixVectorProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using _Detail = detail::ExpressionDetail<binary::MatrixVectorProduct<A, B>>;
    using extents_type = extents<_Detail::ATraits::extents_type::static_extent(0)>;
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
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using value_type = traits::value_type;
    using Base::lhs;
    using Base::rhs;
    using extents_type = typename traits::extents_type;
    using lhs_traits = typename detail_type::ATraits;
    using rhs_traits = typename detail_type::BTraits;

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
        using zipper::expression::detail::HasIndexSet;
        using zipper::expression::detail::HasForEach;
        constexpr bool lhs_has_zeros = HasIndexSet<std::decay_t<A>>;
        constexpr bool rhs_has_zeros = HasIndexSet<std::decay_t<B>>;

        if constexpr (lhs_has_zeros && rhs_has_zeros) {
            auto lr = lhs().template index_set<1>(a);
            auto rr = rhs().template index_set<0>();
            if (lr.size() <= rr.size()) {
                auto body = [&](index_type j) {
                    if (rr.contains(j)) {
                        v += lhs()(a, j) * rhs()(j);
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
                        v += lhs()(a, j) * rhs()(j);
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
                v += lhs()(a, j) * rhs()(j);
            };
            if constexpr (HasForEach<decltype(lr)>) {
                lr.for_each(body);
            } else {
                for (auto j : lr) { body(j); }
            }
        } else if constexpr (rhs_has_zeros) {
            auto rr = rhs().template index_set<0>();
            auto body = [&](index_type j) {
                v += lhs()(a, j) * rhs()(j);
            };
            if constexpr (HasForEach<decltype(rr)>) {
                rr.for_each(body);
            } else {
                for (auto j : rr) { body(j); }
            }
        } else {
            for (index_type j = 0; j < lhs().extent(1); ++j) {
                v += lhs()(a, j) * rhs()(j);
            }
        }
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
