#if !defined(ZIPPER_EXPRESSION_TERNARY_SELECT_HPP)
#define ZIPPER_EXPRESSION_TERNARY_SELECT_HPP

#include "TernaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/utils/extents/extents_formatter.hpp"

#include <compare>
#include <format>

namespace zipper::expression {
namespace ternary {

    // ═════════════════════════════════════════════════════════════════════
    // BoolSelect: condition.select(true_val, false_val)
    //
    // condition is a bool array; true_val and false_val must have the same
    // value_type.  Result value_type = value_type of true_val/false_val.
    // ═════════════════════════════════════════════════════════════════════

    template <zipper::concepts::QualifiedExpression Condition,
              zipper::concepts::QualifiedExpression TrueExpr,
              zipper::concepts::QualifiedExpression FalseExpr>
    class BoolSelect;

} // namespace ternary

// ── Traits ────────────────────────────────────────────────────────────

template <zipper::concepts::QualifiedExpression Condition,
          zipper::concepts::QualifiedExpression TrueExpr,
          zipper::concepts::QualifiedExpression FalseExpr>
struct detail::ExpressionDetail<
    ternary::BoolSelect<Condition, TrueExpr, FalseExpr>>
  : public ternary::detail::
        DefaultTernaryExpressionDetail<Condition, TrueExpr, FalseExpr> {
    using _Base = ternary::detail::
        DefaultTernaryExpressionDetail<Condition, TrueExpr, FalseExpr>;
    using CondTraits = typename _Base::ATraits;
    using TrueTraits = typename _Base::BTraits;
    using FalseTraits = typename _Base::CTraits;
    using ConvertExtentsUtil = ternary::detail::coeffwise_extents_3<
        typename CondTraits::extents_type,
        typename TrueTraits::extents_type,
        typename FalseTraits::extents_type>;
};

template <zipper::concepts::QualifiedExpression Condition,
          zipper::concepts::QualifiedExpression TrueExpr,
          zipper::concepts::QualifiedExpression FalseExpr>
struct detail::ExpressionTraits<
    ternary::BoolSelect<Condition, TrueExpr, FalseExpr>>
  : public ternary::detail::
        DefaultTernaryExpressionTraits<Condition, TrueExpr, FalseExpr> {
    using _Detail = detail::ExpressionDetail<
        ternary::BoolSelect<Condition, TrueExpr, FalseExpr>>;
    using extents_type =
        typename _Detail::ConvertExtentsUtil::merged_extents_type;
    // The result type is the true-branch value_type.
    using value_type = typename _Detail::TrueTraits::value_type;
};

namespace ternary {

    template <zipper::concepts::QualifiedExpression Condition,
              zipper::concepts::QualifiedExpression TrueExpr,
              zipper::concepts::QualifiedExpression FalseExpr>
    class BoolSelect
      : public TernaryExpressionBase<BoolSelect<Condition, TrueExpr, FalseExpr>,
                                     Condition,
                                     TrueExpr,
                                     FalseExpr> {
      public:
        using self_type = BoolSelect<Condition, TrueExpr, FalseExpr>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;
        using detail_type =
            zipper::expression::detail::ExpressionDetail<self_type>;
        using extents_type = typename traits::extents_type;
        using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
        using Base =
            TernaryExpressionBase<self_type, Condition, TrueExpr, FalseExpr>;
        using value_type = typename traits::value_type;
        constexpr static bool is_static = extents_traits::is_static;

        using cond_extents_type =
            typename detail_type::CondTraits::extents_type;
        using true_extents_type =
            typename detail_type::TrueTraits::extents_type;
        using false_extents_type =
            typename detail_type::FalseTraits::extents_type;

        using Base::first;
        using Base::second;
        using Base::third;

        template <typename U, typename V, typename W>
            requires std::constructible_from<typename Base::first_storage_type,
                                             U &&>
                     && std::constructible_from<
                         typename Base::second_storage_type,
                         V &&>
                     && std::constructible_from<
                         typename Base::third_storage_type,
                         W &&>
        BoolSelect(U &&cond, V &&true_val, W &&false_val)
          : Base(std::forward<U>(cond),
                 std::forward<V>(true_val),
                 std::forward<W>(false_val)) {}

        value_type get_value(const auto &cond,
                             const auto &true_val,
                             const auto &false_val) const {
            return cond ? true_val : false_val;
        }

        constexpr auto extent(rank_type i) const -> index_type {
            // Prefer the first child that has a known extent.
            if constexpr (cond_extents_type::rank() == 0) {
                if constexpr (true_extents_type::rank() == 0) {
                    return third().extent(i);
                } else {
                    return second().extent(i);
                }
            } else {
                return first().extent(i);
            }
        }

        constexpr auto extents() const -> extents_type {
            return extents_traits::make_extents_from(*this);
        }

        auto make_owned() const {
            auto owned_a = first().make_owned();
            auto owned_b = second().make_owned();
            auto owned_c = third().make_owned();
            return BoolSelect<decltype(owned_a),
                              decltype(owned_b),
                              decltype(owned_c)>(
                std::move(owned_a), std::move(owned_b), std::move(owned_c));
        }
    };

    template <zipper::concepts::Expression A,
              zipper::concepts::Expression B,
              zipper::concepts::Expression C>
    BoolSelect(const A &, const B &, const C &)
        -> BoolSelect<const A &, const B &, const C &>;

    // ═════════════════════════════════════════════════════════════════════
    // OrderingSelect: ordering.select(less_val, equal_val, greater_val)
    //
    // condition is an array of std::strong_ordering / std::weak_ordering /
    // std::partial_ordering.  Returns less_val when < 0, equal_val when
    // == 0, greater_val when > 0.
    // ═════════════════════════════════════════════════════════════════════

    template <zipper::concepts::QualifiedExpression Condition,
              zipper::concepts::QualifiedExpression LessExpr,
              zipper::concepts::QualifiedExpression EqualExpr,
              zipper::concepts::QualifiedExpression GreaterExpr>
    class OrderingSelect;

} // namespace ternary

// ── Traits for OrderingSelect ─────────────────────────────────────────
// We use a "quaternary" expression built on two nested ternaries
// internally, but expose it through a 4-child expression.  However,
// to keep things simple, we'll implement it as a TernaryExpressionBase
// that stores the condition separately.

// Actually, a cleaner approach: OrderingSelect is its own expression
// with 4 children.  But we can avoid a QuaternaryExpressionBase by
// storing the condition inside the expression and having a ternary
// base for the 3 value children. Let me reconsider...

// The simplest correct approach: a custom expression with 4 stored children
// and hand-written coeff().

} // namespace zipper::expression

namespace zipper::expression::ternary {

/// Four-child select expression for ordering types.
/// Condition is an ordering array; LessExpr/EqualExpr/GreaterExpr provide
/// the values to select from.
template <zipper::concepts::QualifiedExpression Condition,
          zipper::concepts::QualifiedExpression LessExpr,
          zipper::concepts::QualifiedExpression EqualExpr,
          zipper::concepts::QualifiedExpression GreaterExpr>
class OrderingSelect;

} // namespace zipper::expression::ternary

namespace zipper::expression {

template <zipper::concepts::QualifiedExpression Condition,
          zipper::concepts::QualifiedExpression LessExpr,
          zipper::concepts::QualifiedExpression EqualExpr,
          zipper::concepts::QualifiedExpression GreaterExpr>
struct detail::ExpressionDetail<
    ternary::OrderingSelect<Condition, LessExpr, EqualExpr, GreaterExpr>> {
    using CondTraits = detail::ExpressionTraits<std::decay_t<Condition>>;
    using LessTraits = detail::ExpressionTraits<std::decay_t<LessExpr>>;
    using EqualTraits = detail::ExpressionTraits<std::decay_t<EqualExpr>>;
    using GreaterTraits = detail::ExpressionTraits<std::decay_t<GreaterExpr>>;

    using first_value_type = typename CondTraits::value_type;
    using second_value_type = typename LessTraits::value_type;
    using third_value_type = typename EqualTraits::value_type;
    using fourth_value_type = typename GreaterTraits::value_type;

    // Three-way extents merge: condition + less + equal + greater
    using merged_12 = typename binary::detail::coeffwise_extents_values<
        typename CondTraits::extents_type,
        typename LessTraits::extents_type>::merged_extents_type;
    using merged_123 = typename binary::detail::coeffwise_extents_values<
        merged_12,
        typename EqualTraits::extents_type>::merged_extents_type;
    using merged_extents_type =
        typename binary::detail::coeffwise_extents_values<
            merged_123,
            typename GreaterTraits::extents_type>::merged_extents_type;
};

template <zipper::concepts::QualifiedExpression Condition,
          zipper::concepts::QualifiedExpression LessExpr,
          zipper::concepts::QualifiedExpression EqualExpr,
          zipper::concepts::QualifiedExpression GreaterExpr>
struct detail::ExpressionTraits<
    ternary::OrderingSelect<Condition, LessExpr, EqualExpr, GreaterExpr>>
  : public detail::BasicExpressionTraits<
        typename detail::ExpressionTraits<std::decay_t<LessExpr>>::value_type,
        zipper::dextents<0>,
        detail::AccessFeatures{.is_const = true, .is_reference = false},
        detail::ShapeFeatures{.is_resizable = false}> {
    using _Detail = detail::ExpressionDetail<
        ternary::OrderingSelect<Condition, LessExpr, EqualExpr, GreaterExpr>>;
    using extents_type = typename _Detail::merged_extents_type;
    using value_type = typename _Detail::LessTraits::value_type;
    constexpr static bool is_coefficient_consistent =
        _Detail::CondTraits::is_coefficient_consistent
        && _Detail::LessTraits::is_coefficient_consistent
        && _Detail::EqualTraits::is_coefficient_consistent
        && _Detail::GreaterTraits::is_coefficient_consistent;
    constexpr static bool is_value_based = true;
    constexpr static bool stores_references =
        std::is_reference_v<Condition> || std::is_reference_v<LessExpr>
        || std::is_reference_v<EqualExpr> || std::is_reference_v<GreaterExpr>
        || _Detail::CondTraits::stores_references
        || _Detail::LessTraits::stores_references
        || _Detail::EqualTraits::stores_references
        || _Detail::GreaterTraits::stores_references;
    using preferred_layout = typename _Detail::CondTraits::preferred_layout;
};

namespace ternary {

    template <zipper::concepts::QualifiedExpression Condition,
              zipper::concepts::QualifiedExpression LessExpr,
              zipper::concepts::QualifiedExpression EqualExpr,
              zipper::concepts::QualifiedExpression GreaterExpr>
    class OrderingSelect
      : public expression::ExpressionBase<
            OrderingSelect<Condition, LessExpr, EqualExpr, GreaterExpr>> {
      public:
        using self_type =
            OrderingSelect<Condition, LessExpr, EqualExpr, GreaterExpr>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;
        using detail_type =
            zipper::expression::detail::ExpressionDetail<self_type>;
        using extents_type = typename traits::extents_type;
        using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
        using value_type = typename traits::value_type;
        constexpr static bool is_static = extents_traits::is_static;
        constexpr static bool is_value_based = true;

        using Base = expression::ExpressionBase<self_type>;

        auto derived(this auto &self) -> auto & {
            if constexpr (std::is_const_v<
                              std::remove_reference_t<decltype(self)>>) {
                return static_cast<const self_type &>(self);
            } else {
                return static_cast<self_type &>(self);
            }
        }

        using cond_storage_type =
            zipper::detail::expression_storage_t<Condition>;
        using less_storage_type =
            zipper::detail::expression_storage_t<LessExpr>;
        using equal_storage_type =
            zipper::detail::expression_storage_t<EqualExpr>;
        using greater_storage_type =
            zipper::detail::expression_storage_t<GreaterExpr>;
        using cond_expression_type = std::remove_reference_t<Condition>;
        using less_expression_type = std::remove_reference_t<LessExpr>;
        using equal_expression_type = std::remove_reference_t<EqualExpr>;
        using greater_expression_type = std::remove_reference_t<GreaterExpr>;

        OrderingSelect(const OrderingSelect &) = default;
        OrderingSelect(OrderingSelect &&) = default;
        auto operator=(const OrderingSelect &) -> OrderingSelect & = delete;
        auto operator=(OrderingSelect &&) -> OrderingSelect & = delete;

        template <typename U, typename V, typename W, typename X>
            requires std::constructible_from<cond_storage_type, U &&>
                         && std::constructible_from<less_storage_type, V &&>
                         && std::constructible_from<equal_storage_type, W &&>
                         && std::constructible_from<greater_storage_type, X &&>
        OrderingSelect(U &&cond, V &&less_val, W &&equal_val, X &&greater_val)
          : m_cond(std::forward<U>(cond)), m_less(std::forward<V>(less_val)),
            m_equal(std::forward<W>(equal_val)),
            m_greater(std::forward<X>(greater_val)) {}

        auto cond() const -> const cond_expression_type & { return m_cond; }
        auto less() const -> const less_expression_type & { return m_less; }
        auto equal() const -> const equal_expression_type & { return m_equal; }
        auto greater() const -> const greater_expression_type & {
            return m_greater;
        }

        template <typename... Args>
        auto coeff(Args &&...args) const -> value_type
            requires(is_value_based)
        {
            auto ord = m_cond(std::forward<Args>(args)...);
            if (ord < 0) {
                return m_less(std::forward<Args>(args)...);
            } else if (ord == 0) {
                return m_equal(std::forward<Args>(args)...);
            } else {
                return m_greater(std::forward<Args>(args)...);
            }
        }

        constexpr auto extent(rank_type i) const -> index_type {
            // Use whichever child has a non-rank-0 extent.
            using cond_ext = typename detail_type::CondTraits::extents_type;
            using less_ext = typename detail_type::LessTraits::extents_type;
            using equal_ext = typename detail_type::EqualTraits::extents_type;
            if constexpr (cond_ext::rank() > 0) {
                return m_cond.extent(i);
            } else if constexpr (less_ext::rank() > 0) {
                return m_less.extent(i);
            } else if constexpr (equal_ext::rank() > 0) {
                return m_equal.extent(i);
            } else {
                return m_greater.extent(i);
            }
        }

        constexpr auto extents() const -> extents_type {
            return extents_traits::make_extents_from(*this);
        }

        auto make_owned() const {
            auto owned_cond = m_cond.make_owned();
            auto owned_less = m_less.make_owned();
            auto owned_equal = m_equal.make_owned();
            auto owned_greater = m_greater.make_owned();
            return OrderingSelect<decltype(owned_cond),
                                  decltype(owned_less),
                                  decltype(owned_equal),
                                  decltype(owned_greater)>(
                std::move(owned_cond),
                std::move(owned_less),
                std::move(owned_equal),
                std::move(owned_greater));
        }

      private:
        cond_storage_type m_cond;
        less_storage_type m_less;
        equal_storage_type m_equal;
        greater_storage_type m_greater;
    };

    template <zipper::concepts::Expression A,
              zipper::concepts::Expression B,
              zipper::concepts::Expression C,
              zipper::concepts::Expression D>
    OrderingSelect(const A &, const B &, const C &, const D &)
        -> OrderingSelect<const A &, const B &, const C &, const D &>;

} // namespace ternary
} // namespace zipper::expression

#endif
