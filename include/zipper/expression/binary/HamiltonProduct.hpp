/// @file HamiltonProduct.hpp
/// @brief Lazy expression for the quaternion Hamilton product.
/// @ingroup binary_expressions
///
/// `HamiltonProduct<A, B>` is a rank-1, extent-4 binary expression that
/// lazily represents the Hamilton product of two quaternion-like expressions.
///
/// Storage convention is scalar-first: index 0 = w, 1 = x, 2 = y, 3 = z.
///
/// The product q1 * q2 is defined as:
///   w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
///   x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
///   y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
///   z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
///
/// @see zipper::QuaternionBase — CRTP base that uses this expression.
/// @see zipper::expression::binary::CrossProduct — similar pattern for 3D
///      cross products.

#if !defined(ZIPPER_EXPRESSION_BINARY_HAMILTONPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_HAMILTONPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedRankedExpression<1> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
class HamiltonProduct;

} // namespace binary

// ── ExpressionDetail ────────────────────────────────────────────────────
template <zipper::concepts::QualifiedRankedExpression<1> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
  requires(binary::detail::coeffwise_extents_values<
               typename detail::ExpressionTraits<std::decay_t<A>>::extents_type,
               typename detail::ExpressionTraits<std::decay_t<B>>::extents_type
           >::merged_extents_type::static_extent(0) == std::dynamic_extent ||
           binary::detail::coeffwise_extents_values<
               typename detail::ExpressionTraits<std::decay_t<A>>::extents_type,
               typename detail::ExpressionTraits<std::decay_t<B>>::extents_type
           >::merged_extents_type::static_extent(0) == 4)
struct detail::ExpressionDetail<binary::HamiltonProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionDetail<A, B> {
    using _Base = binary::detail::DefaultBinaryExpressionDetail<A, B>;
    using ATraits = typename _Base::ATraits;
    using BTraits = typename _Base::BTraits;
    using ConvertExtentsUtil = binary::detail::coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;
};

// ── ExpressionTraits ────────────────────────────────────────────────────
template <zipper::concepts::QualifiedRankedExpression<1> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
struct detail::ExpressionTraits<binary::HamiltonProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using _Detail = detail::ExpressionDetail<binary::HamiltonProduct<A, B>>;
    using extents_type = typename _Detail::ConvertExtentsUtil::merged_extents_type;

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {

// ── HamiltonProduct expression ──────────────────────────────────────────
template <zipper::concepts::QualifiedRankedExpression<1> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
class HamiltonProduct : public BinaryExpressionBase<HamiltonProduct<A, B>, A, B> {
   public:
    using self_type = HamiltonProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using value_type = traits::value_type;
    using extents_type = typename traits::extents_type;
    using lhs_traits = typename detail_type::ATraits;
    using rhs_traits = typename detail_type::BTraits;

    using Base::lhs;
    using Base::rhs;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    HamiltonProduct(const HamiltonProduct&) = default;
    HamiltonProduct(HamiltonProduct&&) = default;

    template <typename U, typename V>
      requires std::constructible_from<typename Base::lhs_storage_type, U&&> &&
               std::constructible_from<typename Base::rhs_storage_type, V&&>
    HamiltonProduct(U&& a, V&& b)
        : Base(std::forward<U>(a), std::forward<V>(b)) {
        ZIPPER_ASSERT(lhs().extent(0) == rhs().extent(0));
        ZIPPER_ASSERT(lhs().extent(0) == 4);
    }

    constexpr auto extent([[maybe_unused]] rank_type i) const -> index_type {
        ZIPPER_ASSERT(i == 0);
        return lhs().extent(0);
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

    value_type coeff(index_type i) const {
        // Storage layout: 0=w, 1=x, 2=y, 3=z
        const auto& l = lhs();
        const auto& r = rhs();
        switch (i) {
        case 0: // w = lw*rw - lx*rx - ly*ry - lz*rz
            return l(0)*r(0) - l(1)*r(1) - l(2)*r(2) - l(3)*r(3);
        case 1: // x = lw*rx + lx*rw + ly*rz - lz*ry
            return l(0)*r(1) + l(1)*r(0) + l(2)*r(3) - l(3)*r(2);
        case 2: // y = lw*ry - lx*rz + ly*rw + lz*rx
            return l(0)*r(2) - l(1)*r(3) + l(2)*r(0) + l(3)*r(1);
        case 3: // z = lw*rz + lx*ry - ly*rx + lz*rw
            return l(0)*r(3) + l(1)*r(2) - l(2)*r(1) + l(3)*r(0);
        default:
            ZIPPER_ASSERT(false && "HamiltonProduct: index out of range");
            return value_type{};
        }
    }

    /// Recursively deep-copy children so the result owns all data.
    auto make_owned() const {
        auto owned_a = lhs().make_owned();
        auto owned_b = rhs().make_owned();
        return HamiltonProduct<decltype(owned_a), decltype(owned_b)>(
            std::move(owned_a), std::move(owned_b));
    }
};

template <zipper::concepts::RankedExpression<1> A,
          zipper::concepts::RankedExpression<1> B>
HamiltonProduct(const A& a, const B& b) -> HamiltonProduct<const A&, const B&>;

} // namespace binary
} // namespace zipper::expression
#endif
