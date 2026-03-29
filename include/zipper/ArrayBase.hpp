#if !defined(ZIPPER_ARRAYBASE_HPP)
#define ZIPPER_ARRAYBASE_HPP

#include "ZipperBase.hpp"
#include "concepts/detail/IsZipperBase.hpp"
#include "concepts/stl.hpp"
#include "expression/nullary/StlMDArray.hpp"
#include "expression/reductions/All.hpp"
#include "expression/reductions/Any.hpp"
#include "expression/reductions/CoefficientProduct.hpp"
#include "expression/reductions/CoefficientSum.hpp"
#include "expression/reductions/LpNorm.hpp"
#include "expression/reductions/LpNormPowered.hpp"
#include "zipper/types.hpp"
//
#include "concepts/Array.hpp"
#include "concepts/Expression.hpp"
//
////
#include "detail/extents/static_extents_to_integral_sequence.hpp"
#include "expression/ternary/Select.hpp"
#include "expression/unary/Abs.hpp"
#include "expression/unary/ScalarPower.hpp"

#include <compare>

namespace zipper {

namespace detail {
    template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
    class Array_;
} // namespace detail

template <concepts::Expression Expr>
class ArrayBase : public ZipperBase<ArrayBase, Expr> {
  public:
    ArrayBase() = default;

    using expression_type = std::decay_t<Expr>;
    using expression_traits =
        expression::detail::ExpressionTraits<expression_type>;
    using value_type = typename expression_traits::value_type;
    using extents_type = typename expression_traits::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;
    using Base = ZipperBase<ArrayBase, Expr>;
    using Base::Base;
    using Base::expression;

    template <typename... Args>
        requires(!(concepts::Array<Args> && ...))
    ArrayBase(Args &&...args)
      : Base(std::in_place, std::forward<Args>(args)...) {}

    template <index_type... N>
    auto eval(const std::integer_sequence<index_type, N...> &) const
        requires(std::is_same_v<extents<N...>, extents_type>)
    {
        return detail::Array_<value_type, zipper::extents<N...>, true>(
            this->expression());
    }
    auto eval() const {
        return eval(detail::extents::static_extents_to_integral_sequence_t<
                    extents_type>{});
    }
    auto operator=(concepts::Array auto const &v) -> ArrayBase & {
        return Base::operator=(v.expression());
    }
    auto operator=(concepts::Array auto &&v) -> ArrayBase & {
        return Base::operator=(v.expression());
    }

    template <concepts::Array Other>
    ArrayBase(const Other &other)
        requires(expression::concepts::WritableExpression<expression_type>)
      : ArrayBase(other.expression()) {}

    auto operator*=(const value_type &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type>)
    {
        return *this = other * *this;
    }
    auto operator/=(const value_type &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type>)
    {
        return *this = *this / other;
    }
    template <concepts::Array Other>
    auto operator+=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type>)
    {
        return *this = *this + other;
    }
    template <concepts::Array Other>
    auto operator-=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type>)
    {
        return *this = *this - other;
    }
    template <concepts::Array Other>
    auto operator*=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type>)
    {
        return *this = *this * other;
    }
    template <concepts::Array Other>
    auto operator/=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type>)
    {
        return *this = *this / other;
    }

    //--------------------------------------------------------------
    //

    template <typename Self>
    auto pow(this Self &&self, value_type const &exp) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        return ArrayBase<expression::unary::ScalarPower<child_t, value_type>>(
            std::in_place, std::forward<Self>(self).expression(), exp);
    }

    template <typename Self>
    auto abs(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        return ArrayBase<expression::unary::Abs<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    auto sum() const -> value_type {
        return expression::reductions::CoefficientSum{expression()}();
    }

    auto product() const -> value_type {
        return expression::reductions::CoefficientProduct{expression()}();
    }

    template <index_type T>
    auto norm_powered() const -> value_type {
        return expression::reductions::LpNormPowered<T,
                                                     const expression_type &>(
            expression())();
    }
    auto norm_powered(value_type T) const -> value_type {
        return pow(T).abs().sum();
    }

    template <index_type T = 2>
    auto norm() const -> value_type {
        return expression::reductions::LpNorm<T, const expression_type &>(
            expression())();
    }
    auto norm(value_type T) const -> value_type {
        value_type p = value_type(1.0) / T;
        return std::pow<value_type>(norm_powered(T), p);
    }

    template <index_type T = 2>
    auto normalized() const {
        return *this / norm<T>();
    }
    auto normalized(value_type T) const -> value_type {
        return *this / norm(T);
    }
    template <index_type T = 2>
    void normalize() {
        *this /= norm<T>();
    }
    void normalize(value_type T) { *this /= norm(T); }

    [[nodiscard]] auto any() const -> bool
        requires(std::is_same_v<value_type, bool>)
    {
        return expression::reductions::Any(expression())();
    }
    [[nodiscard]] auto all() const -> bool
        requires(std::is_same_v<value_type, bool>)
    {
        return expression::reductions::All(expression())();
    }

    /// @brief Boolean select: `mask.select(true_val, false_val)`.
    ///
    /// For each coefficient, returns `true_val` where the mask is true,
    /// `false_val` where the mask is false.  Equivalent to Eigen's
    /// `.select()` or NumPy's `np.where()`.
    ///
    /// @param true_val  Value array for true entries.
    /// @param false_val Value array for false entries.
    /// @return A lazy expression with the element-wise selection.
    template <concepts::Array TrueArr, concepts::Array FalseArr, typename Self>
    auto select(this Self &&self,
                const TrueArr &true_val,
                const FalseArr &false_val)
        requires(std::is_same_v<value_type, bool>)
    {
        using cond_t = detail::member_child_storage_t<Self, expression_type>;
        using true_t = const typename TrueArr::expression_type &;
        using false_t = const typename FalseArr::expression_type &;
        using SelectType =
            expression::ternary::BoolSelect<cond_t, true_t, false_t>;
        return ArrayBase<SelectType>(std::in_place,
                                     std::forward<Self>(self).expression(),
                                     true_val.expression(),
                                     false_val.expression());
    }

    /// @brief Ordering select: `ordering.select(less, equal, greater)`.
    ///
    /// For each coefficient, returns `less` where the ordering is < 0,
    /// `equal` where it is == 0, `greater` where it is > 0.
    /// Works with std::strong_ordering, std::weak_ordering, or
    /// std::partial_ordering value types.
    ///
    /// @param less_val    Value array for less-than entries.
    /// @param equal_val   Value array for equal entries.
    /// @param greater_val Value array for greater-than entries.
    /// @return A lazy expression with the element-wise three-way selection.
    template <concepts::Array LessArr,
              concepts::Array EqualArr,
              concepts::Array GreaterArr,
              typename Self>
    auto select(this Self &&self,
                const LessArr &less_val,
                const EqualArr &equal_val,
                const GreaterArr &greater_val)
        requires(
            !std::is_same_v<value_type, bool>
            && (std::is_convertible_v<value_type, std::partial_ordering>
                || std::is_convertible_v<value_type, std::weak_ordering>
                || std::is_convertible_v<value_type, std::strong_ordering>))
    {
        using cond_t = detail::member_child_storage_t<Self, expression_type>;
        using less_t = const typename LessArr::expression_type &;
        using equal_t = const typename EqualArr::expression_type &;
        using greater_t = const typename GreaterArr::expression_type &;
        using SelectType = expression::ternary::
            OrderingSelect<cond_t, less_t, equal_t, greater_t>;
        return ArrayBase<SelectType>(std::in_place,
                                     std::forward<Self>(self).expression(),
                                     less_val.expression(),
                                     equal_val.expression(),
                                     greater_val.expression());
    }

    // Slice methods - construct wrapper in-place to avoid moving non-movable
    // expressions
    template <typename... Slices, typename Self>
    auto slice(this Self &&self, Slices &&...slices) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        using V = expression::unary::
            Slice<child_t, detail::slice_type_for_t<std::decay_t<Slices>>...>;
        return ArrayBase<V>(
            std::in_place,
            std::forward<Self>(self).expression(),
            Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
    }
    template <typename... Slices, typename Self>
    auto slice(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, expression_type>;
        using V = expression::unary::Slice<child_t, std::decay_t<Slices>...>;
        return ArrayBase<V>(
            std::in_place, std::forward<Self>(self).expression(), Slices{}...);
    }
};

template <concepts::Expression Expr>
ArrayBase(Expr &&) -> ArrayBase<Expr>;
template <concepts::Expression Expr>
ArrayBase(const Expr &) -> ArrayBase<Expr>;

// STL deduction guides: rvalue → owning StlMDArray, lvalue → borrowing
// StlMDArray
template <concepts::StlStorage S>
ArrayBase(S &&) -> ArrayBase<expression::nullary::StlMDArray<std::decay_t<S>>>;
template <concepts::StlStorage S>
ArrayBase(S &) -> ArrayBase<expression::nullary::StlMDArray<S &>>;

namespace concepts::detail {
    template <typename T>
    struct IsArray<ArrayBase<T>> : std::true_type {};
    template <typename T>
    struct IsZipperBase<ArrayBase<T>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
