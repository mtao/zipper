#if !defined(ZIPPER_EXPRESSION_TERNARY_TERNARYEXPRESSIONBASE_HPP)
#define ZIPPER_EXPRESSION_TERNARY_TERNARYEXPRESSIONBASE_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExpressionStorage.hpp"
#include "zipper/expression/ExpressionBase.hpp"

namespace zipper::expression::ternary {

namespace detail {

    /// Default implementation details for ternary expressions.
    ///
    /// Holds child traits aliases and child value types that the ternary
    /// expression class body needs for its implementation.
    template <zipper::concepts::QualifiedExpression ChildA,
              zipper::concepts::QualifiedExpression ChildB,
              zipper::concepts::QualifiedExpression ChildC>
    struct DefaultTernaryExpressionDetail {
        using ATraits =
            expression::detail::ExpressionTraits<std::decay_t<ChildA>>;
        using BTraits =
            expression::detail::ExpressionTraits<std::decay_t<ChildB>>;
        using CTraits =
            expression::detail::ExpressionTraits<std::decay_t<ChildC>>;
        using first_value_type = typename ATraits::value_type;
        using second_value_type = typename BTraits::value_type;
        using third_value_type = typename CTraits::value_type;
    };

    /// Default traits for ternary expressions.
    template <zipper::concepts::QualifiedExpression ChildA,
              zipper::concepts::QualifiedExpression ChildB,
              zipper::concepts::QualifiedExpression ChildC>
    struct DefaultTernaryExpressionTraits
      : public expression::detail::BasicExpressionTraits<
            typename expression::detail::ExpressionTraits<
                std::decay_t<ChildA>>::value_type,
            zipper::dextents<0>,
            expression::detail::AccessFeatures{.is_const = true,
                                               .is_reference = false},
            expression::detail::ShapeFeatures{.is_resizable = false}> {
        using _Detail = DefaultTernaryExpressionDetail<ChildA, ChildB, ChildC>;

        // defaulting to first parameter
        using value_type = typename _Detail::ATraits::value_type;
        constexpr static bool is_coefficient_consistent =
            _Detail::ATraits::is_coefficient_consistent
            && _Detail::BTraits::is_coefficient_consistent
            && _Detail::CTraits::is_coefficient_consistent;
        constexpr static bool is_value_based = true;

        constexpr static bool stores_references =
            std::is_reference_v<ChildA> || std::is_reference_v<ChildB>
            || std::is_reference_v<ChildC>
            || _Detail::ATraits::stores_references
            || _Detail::BTraits::stores_references
            || _Detail::CTraits::stores_references;

        using preferred_layout = typename _Detail::ATraits::preferred_layout;
    };
} // namespace detail

/// Base class for ternary expression nodes (3 children).
///
/// ChildTypeA / ChildTypeB / ChildTypeC should be const-ref-qualified
/// (e.g. `const A&`) for reference storage, or non-reference for by-value
/// storage.
template <typename Derived,
          typename ChildTypeA,
          typename ChildTypeB,
          typename ChildTypeC>
class TernaryExpressionBase : public expression::ExpressionBase<Derived> {
  public:
    using first_storage_type = zipper::detail::expression_storage_t<ChildTypeA>;
    using second_storage_type =
        zipper::detail::expression_storage_t<ChildTypeB>;
    using third_storage_type = zipper::detail::expression_storage_t<ChildTypeC>;
    using first_expression_type = std::remove_reference_t<ChildTypeA>;
    using second_expression_type = std::remove_reference_t<ChildTypeB>;
    using third_expression_type = std::remove_reference_t<ChildTypeC>;

    using self_type =
        TernaryExpressionBase<Derived, ChildTypeA, ChildTypeB, ChildTypeC>;
    using traits = zipper::expression::detail::ExpressionTraits<Derived>;
    using detail_type = zipper::expression::detail::ExpressionDetail<Derived>;
    using extents_type = typename traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using value_type = typename traits::value_type;
    constexpr static bool is_static = extents_traits::is_static;
    constexpr static bool is_value_based = traits::is_value_based;

    using Base = expression::ExpressionBase<Derived>;

    auto derived(this auto &self) -> auto & {
        if constexpr (std::is_const_v<
                          std::remove_reference_t<decltype(self)>>) {
            return static_cast<const Derived &>(self);
        } else {
            return static_cast<Derived &>(self);
        }
    }
    using first_value_type = typename detail_type::first_value_type;
    using second_value_type = typename detail_type::second_value_type;
    using third_value_type = typename detail_type::third_value_type;

    TernaryExpressionBase(const TernaryExpressionBase &) = default;
    TernaryExpressionBase(TernaryExpressionBase &&) = default;
    auto operator=(const TernaryExpressionBase &)
        -> TernaryExpressionBase & = delete;
    auto operator=(TernaryExpressionBase &&)
        -> TernaryExpressionBase & = delete;

    template <typename U, typename V, typename W>
        requires std::constructible_from<first_storage_type, U &&>
                     && std::constructible_from<second_storage_type, V &&>
                     && std::constructible_from<third_storage_type, W &&>
    TernaryExpressionBase(U &&a, V &&b, W &&c)
      : m_first(std::forward<U>(a)), m_second(std::forward<V>(b)),
        m_third(std::forward<W>(c)) {}

    constexpr auto extents() const -> extents_type {
        using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
        return extents_traits::make_extents_from(derived());
    }

    auto first() const -> const first_expression_type & { return m_first; }
    auto second() const -> const second_expression_type & { return m_second; }
    auto third() const -> const third_expression_type & { return m_third; }

    auto get_value(const first_value_type &a,
                   const second_value_type &b,
                   const third_value_type &c) const -> value_type {
        return derived().get_value(a, b, c);
    }

    template <typename... Args>
    auto coeff(Args &&...args) const -> value_type
        requires(is_value_based)
    {
        return value_type(get_value(m_first(std::forward<Args>(args)...),
                                    m_second(std::forward<Args>(args)...),
                                    m_third(std::forward<Args>(args)...)));
    }

  private:
    first_storage_type m_first;
    second_storage_type m_second;
    third_storage_type m_third;
};

} // namespace zipper::expression::ternary
#endif
