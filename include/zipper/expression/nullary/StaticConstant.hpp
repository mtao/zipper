#if !defined(ZIPPER_EXPRESSION_NULLARY_STATIC_CONSTANT_HPP)
#define ZIPPER_EXPRESSION_NULLARY_STATIC_CONSTANT_HPP

/// @file StaticConstant.hpp
/// @brief Nullary expression with a compile-time constant value.
/// @ingroup expressions_nullary
///
/// `StaticConstant<T, Value, Indices...>` is a rank-N expression whose every
/// coefficient is `Value`, a T. T must be usable as a template argument
/// (arithmetic types; for others use the run-time Constant), and a
/// floating-point value is written as a floating-point literal:
/// `StaticConstant<double, 0.5, 3>`, `StaticConstant<double, 2.0, 3>`.
///
/// This expression owns no data — its value is encoded in the type.
///
/// `Ones<T, Indices...>` = `StaticConstant<T, T{1}, Indices...>`. For zeros
/// use `Zero` (Zero.hpp): it works for any T and is structurally zero (empty
/// index sets), which a StaticConstant never is, whatever its value.
///
/// @code
///   // Static 3x3 matrix of ones
///   auto o = Ones<double, 3, 3>();
///
///   // Static 4-vector of twos
///   auto c = StaticConstant<float, 2.0f, 4>();
///
///   // Non-integral values, dynamic extents
///   auto h = StaticConstant<double, 0.5, dynamic_extent>(n);
/// @endcode

#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace nullary {

    template <typename T, T Value, index_type... Indices>
    class StaticConstant
      : public ExpressionBase<StaticConstant<T, Value, Indices...>>
      , public zipper::extents<Indices...> {
      public:
        using self_type = StaticConstant<T, Value, Indices...>;
        using traits = zipper::expression::detail::ExpressionTraits<self_type>;
        using extents_type = typename traits::extents_type;
        using extents_traits = typename traits::extents_traits;
        using value_type = typename traits::value_type;

        using extents_type::extent;
        using extents_type::rank;
        auto extents() const -> const extents_type & { return *this; }

        /// The compile-time constant value.
        constexpr static T static_value = Value;

        StaticConstant(const StaticConstant &) = default;
        StaticConstant(StaticConstant &&) = default;
        auto operator=(const StaticConstant &) -> StaticConstant & = default;
        auto operator=(StaticConstant &&) -> StaticConstant & = default;

        StaticConstant()
            requires(extents_traits::is_static)
        = default;

        StaticConstant(const extents_type &e) : extents_type(e) {}

        template <zipper::concepts::Index... Args>
        StaticConstant(Args &&...args)
          : StaticConstant(extents_type(std::forward<Args>(args)...)) {}

        /// Returns the constant value, cast to value_type.
        auto coeff(zipper::concepts::Index auto &&...) const -> value_type {
            return static_cast<value_type>(static_value);
        }

        /// StaticConstant already owns its data — make_owned() returns a copy.
        auto make_owned() const -> StaticConstant { return *this; }
    };

    /// Convenience alias: all-ones expression.
    template <typename T, index_type... Indices>
    using Ones = StaticConstant<T, T{1}, Indices...>;

} // namespace nullary

// ── ExpressionTraits specialization ─────────────────────────────

template <typename T, T Value, index_type... Indices>
struct detail::ExpressionTraits<nullary::StaticConstant<T, Value, Indices...>>
  : public BasicExpressionTraits<
        T,
        zipper::extents<Indices...>,
        expression::detail::AccessFeatures::const_value(),
        expression::detail::ShapeFeatures::resizable()> {};

} // namespace zipper::expression

#endif
