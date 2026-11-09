#if !defined(ZIPPER_ARRAYBASE_HPP)
#define ZIPPER_ARRAYBASE_HPP

#include "ZipperBase.hpp"
#include "detail/as_conversion.hpp"
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
#include "expression/unary/math/all.hpp"
#include "expression/nullary/Constant.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "concepts/ArrayOrScalar.hpp"
#include "concepts/Zipper.hpp"

#include <compare>
#include <type_traits>

namespace zipper {

namespace detail {
    template <typename ValueType, concepts::Extents Extents, bool LeftMajor>
    class Array_;

    /// An argument of ArrayBase::select: an array, referenced, or a scalar,
    /// broadcast as a rank-0 Constant held by value.
    template <typename T>
    struct SelectOperand {
        using type = const typename T::expression_type &;
        static auto get(const T &value) -> type { return value.expression(); }
    };
    template <typename T>
        requires(!concepts::Zipper<T> && std::is_scalar_v<T>)
    struct SelectOperand<T> {
        using type = expression::nullary::Constant<T>;
        static auto get(const T &value) -> type { return type(value); }
    };
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
    using Base::operator=;

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
    auto operator=(concepts::Array auto const &v) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return Base::operator=(v.expression());
    }
    auto operator=(concepts::Array auto &&v) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return Base::operator=(v.expression());
    }

    template <concepts::Array Other>
    ArrayBase(const Other &other)
        requires(expression::concepts::WritableExpression<expression_type>)
      : ArrayBase(other.expression()) {}

    auto operator*=(const value_type &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return *this = other * *this;
    }
    auto operator/=(const value_type &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return *this = *this / other;
    }
    template <concepts::Array Other>
    auto operator+=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return *this = *this + other;
    }
    template <concepts::Array Other>
    auto operator-=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return *this = *this - other;
    }
    template <concepts::Array Other>
    auto operator*=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return *this = *this * other;
    }
    template <concepts::Array Other>
    auto operator/=(const Other &other) -> ArrayBase &
        requires(expression::concepts::WritableExpression<expression_type> && !Base::is_const)
    {
        return *this = *this / other;
    }

    //--------------------------------------------------------------
    //

    template <typename Self>
    auto pow(this Self &&self, value_type const &exp) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::ScalarPower<child_t, value_type>>(
            std::in_place, std::forward<Self>(self).expression(), exp);
    }

    template <typename Self>
    auto abs(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Abs<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto sqrt(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Sqrt<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto cbrt(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Cbrt<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto sin(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Sin<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto cos(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Cos<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto tan(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Tan<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto asin(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Asin<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto acos(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Acos<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto atan(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Atan<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto sinh(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Sinh<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto cosh(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Cosh<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto tanh(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Tanh<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto exp(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Exp<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto exp2(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Exp2<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto log(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Log<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto log2(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Log2<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto log10(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Log10<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto floor(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Floor<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto ceil(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Ceil<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto round(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Round<child_t>>(
            std::in_place, std::forward<Self>(self).expression());
    }

    template <typename Self>
    auto sign(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        return ArrayBase<expression::unary::Sign<child_t>>(
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
        return expression::reductions::LpNorm<
            std::integral_constant<index_type, T>, const expression_type &>(
            expression())();
    }
    auto norm(value_type T) const -> value_type {
        value_type p = value_type(1.0) / T;
        return std::pow<value_type>(norm_powered(T), p);
    }

    template <index_type T = 2, typename Self>
    auto normalized(this Self&& self) {
        return std::forward<Self>(self) / self.template norm<T>();
    }
    template <typename Self>
    auto normalized(this Self&& self, value_type T) {
        return std::forward<Self>(self) / self.norm(T);
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
    /// @param true_val  Values (an array or a scalar) for true entries.
    /// @param false_val Values (an array or a scalar) for false entries.
    /// @return A lazy expression with the element-wise selection.
    template <concepts::ArrayOrScalar TrueArr,
              concepts::ArrayOrScalar FalseArr,
              typename Self>
    auto select(this Self &&self,
                const TrueArr &true_val,
                const FalseArr &false_val)
        requires(std::is_same_v<value_type, bool>)
    {
        using cond_t = detail::member_child_storage_t<Self, Expr>;
        using true_t = typename detail::SelectOperand<TrueArr>::type;
        using false_t = typename detail::SelectOperand<FalseArr>::type;
        using SelectType =
            expression::ternary::BoolSelect<cond_t, true_t, false_t>;
        return ArrayBase<SelectType>(
            std::in_place,
            std::forward<Self>(self).expression(),
            detail::SelectOperand<TrueArr>::get(true_val),
            detail::SelectOperand<FalseArr>::get(false_val));
    }

    /// @brief Ordering select: `ordering.select(less, equal, greater)`.
    ///
    /// For each coefficient, returns `less` where the ordering is < 0,
    /// `equal` where it is == 0, `greater` where it is > 0.
    /// Works with std::strong_ordering, std::weak_ordering, or
    /// std::partial_ordering value types.
    ///
    /// @param less_val    Values (an array or a scalar) for less-than entries.
    /// @param equal_val   Values (an array or a scalar) for equal entries.
    /// @param greater_val Values (an array or a scalar) for greater-than
    ///                    entries.
    /// @return A lazy expression with the element-wise three-way selection.
    template <concepts::ArrayOrScalar LessArr,
              concepts::ArrayOrScalar EqualArr,
              concepts::ArrayOrScalar GreaterArr,
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
        using cond_t = detail::member_child_storage_t<Self, Expr>;
        using less_t = typename detail::SelectOperand<LessArr>::type;
        using equal_t = typename detail::SelectOperand<EqualArr>::type;
        using greater_t = typename detail::SelectOperand<GreaterArr>::type;
        using SelectType = expression::ternary::
            OrderingSelect<cond_t, less_t, equal_t, greater_t>;
        return ArrayBase<SelectType>(
            std::in_place,
            std::forward<Self>(self).expression(),
            detail::SelectOperand<LessArr>::get(less_val),
            detail::SelectOperand<EqualArr>::get(equal_val),
            detail::SelectOperand<GreaterArr>::get(greater_val));
    }

    // Slice methods - construct wrapper in-place to avoid moving non-movable
    // expressions
    template <typename... Slices, typename Self>
    auto slice(this Self &&self, Slices &&...slices) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        using V = expression::unary::
            Slice<child_t, detail::slice_type_for_t<std::decay_t<Slices>>...>;
        return ArrayBase<V>(
            std::in_place,
            std::forward<Self>(self).expression(),
            Base::filter_args_for_zipperbase(std::forward<Slices>(slices))...);
    }
    template <typename... Slices, typename Self>
    auto slice(this Self &&self) {
        using child_t = detail::member_child_storage_t<Self, Expr>;
        using V = expression::unary::Slice<child_t, std::decay_t<Slices>...>;
        return ArrayBase<V>(
            std::in_place, std::forward<Self>(self).expression(), Slices{}...);
    }
};

ZIPPER_DEFINE_AS(array, ArrayBase, 0)

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

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
ArrayBase(zipper::mdspan<T, Extents, Layout, Accessor>)
    -> ArrayBase<expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

namespace concepts::detail {
    template <typename T>
    struct IsArray<ArrayBase<T>> : std::true_type {};
    template <typename T>
    struct IsZipperBase<ArrayBase<T>> : std::true_type {};
} // namespace concepts::detail
} // namespace zipper

#endif
