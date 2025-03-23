#if !defined(UVL_ARRAYBASE_HPP)
#define UVL_ARRAYBASE_HPP

#include "uvl/types.hpp"
#include "views/reductions/All.hpp"
#include "views/reductions/Any.hpp"
#include "views/reductions/CoefficientSum.hpp"
#include "views/unary/SliceView.hpp"
//
#include "concepts/ArrayBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
//
#include "views/unary/CastView.hpp"
#include "views/unary/SwizzleView.hpp"
////
#include "uvl/detail/declare_operations.hpp"
#include "uvl/views/binary/ArithmeticViews.hpp"
#include "uvl/views/unary/ScalarArithmeticViews.hpp"
#include "views/unary/AbsView.hpp"
#include "views/unary/DiagonalView.hpp"
#include "views/unary/ScalarPowerView.hpp"
#include "views/unary/detail/operation_implementations.hpp"

namespace uvl {

template <concepts::ViewDerived View>
class ArrayBase {
   public:
    ArrayBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    template <index_type... N>
    auto eval(const extents<N...>&) const {
        return Array<value_type, N...>(*this);
    }
    auto eval() const { return eval(extents()); }
    template <typename... Args>
    ArrayBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    ArrayBase(View&& v) : m_view(v) {}
    ArrayBase(const View& v) : m_view(v) {}
    ArrayBase& operator=(concepts::ViewDerived auto const& v) {
        m_view = v;
        return *this;
    }
    ArrayBase& operator=(concepts::ArrayBaseDerived auto const& v) {
        m_view = v.view();
        return *this;
    }

    ArrayBase(ArrayBase&& v) = default;
    ArrayBase(const ArrayBase& v) = default;

    template <concepts::ViewDerived Other>
    ArrayBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(extents_traits::convert_from(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::ViewDerived Other>
    ArrayBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other);
        return *this;
    }
    template <concepts::ArrayBaseDerived Other>
    ArrayBase(const Other& other)
        requires(view_type::is_writable)
        : ArrayBase(other.view()) {}
    template <concepts::ArrayBaseDerived Other>
    ArrayBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    ArrayBase& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    ArrayBase& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }
    template <concepts::ArrayBaseDerived Other>
    ArrayBase& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this + other;
    }
    template <concepts::ArrayBaseDerived Other>
    ArrayBase& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }

    template <rank_type... ranks>
    auto swizzle() const {
        return ArrayBase<views::unary::SwizzleView<value_type, ranks...>>(
            view());
    }
    template <typename T>
    auto cast() const {
        return ArrayBase<views::unary::CastView<T, view_type>>(view());
    }

    //--------------------------------------------------------------
    //

    auto pow(value_type const& exp) const {
        return ArrayBase<views::unary::ScalarPowerView<view_type, value_type>>(
            view(), exp);
    }

    auto abs() const {
        return ArrayBase<views::unary::AbsView<view_type>>(view());
    }

    template <index_type T>
    value_type norm_powered() const {
        if constexpr (T == 1) {
            return views::reductions::CoefficientSum{abs().view()}();
        } else if constexpr (T == 2) {
            return views::reductions::CoefficientSum{(*this * *this).view()}();
        } else {
            return views::reductions::CoefficientSum{pow(T).abs().view()}();
        }
    }
    value_type norm_powered(value_type T) const {
        return views::reductions::CoefficientSum{pow(T).abs().view()}();
    }

    template <index_type T = 2>
    value_type norm() const {
        const value_type v = norm_powered<T>();
        if constexpr (T == 1) {
            return v;
        } else if constexpr (T == 2) {
            return std::sqrt(v);
        } else {
            const value_type p = value_type(1.0) / T;
            return std::pow<value_type>(v, p);
        }
    }
    value_type norm(value_type T) const {
        value_type p = value_type(1.0) / T;
        return std::pow<value_type>(norm_powered(T), p);
    }

    template <index_type T = 2>
    auto normalized() const {
        return *this / norm<T>();
    }
    value_type normalized(value_type T) const { return *this / norm(T); }
    template <index_type T = 2>
    void normalize() {
        *this /= norm<T>();
    }
    void normalize(value_type T) { *this /= norm(T); }

    template <typename... Args>
    value_type operator()(Args&&... idxs) const

    {
        return view()(std::forward<Args>(idxs)...);
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;

        return ArrayBase<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;
        return ArrayBase<view_type>(view_type(view(), Slices{}...));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return ArrayBase<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return ArrayBase<view_type>(view_type(view(), Slices{}...));
    }

    auto diagonal() const {
        return VectorBase<views::unary::DiagonalView<view_type, true>>(view());
    }
    auto diagonal() {
        return VectorBase<views::unary::DiagonalView<view_type, false>>(view());
    }

    const View& view() const { return m_view; }
    View& view() { return m_view; }
    const extents_type& extents() const { return view().extents(); }
    constexpr index_type extent(rank_type i) const { return m_view.extent(i); }
    bool any() const
        requires(std::is_same_v<value_type, bool>)
    {
        return views::reductions::Any(view())();
    }
    bool all() const
        requires(std::is_same_v<value_type, bool>)
    {
        return views::reductions::All(view())();
    }

   private:
    View m_view;
};

template <concepts::ViewDerived View>
ArrayBase(View&& view) -> ArrayBase<View>;
template <concepts::ViewDerived View>
ArrayBase(const View& view) -> ArrayBase<View>;

UNARY_DECLARATION(ArrayBase, LogicalNot, operator!)
UNARY_DECLARATION(ArrayBase, BitNot, operator~)
UNARY_DECLARATION(ArrayBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(ArrayBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(ArrayBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(ArrayBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(ArrayBase, Divides, operator/)
SCALAR_BINARY_DECLARATION(ArrayBase, Modulus, operator%)
SCALAR_BINARY_DECLARATION(ArrayBase, EqualsTo, operator==)
SCALAR_BINARY_DECLARATION(ArrayBase, NotEqualsTo, operator!=)
SCALAR_BINARY_DECLARATION(ArrayBase, Greater, operator>)
SCALAR_BINARY_DECLARATION(ArrayBase, Less, operator<)
SCALAR_BINARY_DECLARATION(ArrayBase, GreaterEqual, operator>=)
SCALAR_BINARY_DECLARATION(ArrayBase, LessEqual, operator<=)
SCALAR_BINARY_DECLARATION(ArrayBase, LogicalAnd, operator&&)
SCALAR_BINARY_DECLARATION(ArrayBase, LogicalOr, operator||)
SCALAR_BINARY_DECLARATION(ArrayBase, BitAnd, operator&)
SCALAR_BINARY_DECLARATION(ArrayBase, BitOr, operator|)
SCALAR_BINARY_DECLARATION(ArrayBase, BitXor, operator^)

BINARY_DECLARATION(ArrayBase, Plus, operator+)
BINARY_DECLARATION(ArrayBase, Minus, operator-)
BINARY_DECLARATION(ArrayBase, Multiplies, operator*)
BINARY_DECLARATION(ArrayBase, Divides, operator/)
BINARY_DECLARATION(ArrayBase, Modulus, operator%)
BINARY_DECLARATION(ArrayBase, EqualsTo, operator==)
BINARY_DECLARATION(ArrayBase, NotEqualsTo, operator!=)
BINARY_DECLARATION(ArrayBase, Greater, operator>)
BINARY_DECLARATION(ArrayBase, Less, operator<)
BINARY_DECLARATION(ArrayBase, GreaterEqual, operator>=)
BINARY_DECLARATION(ArrayBase, LessEqual, operator<=)
BINARY_DECLARATION(ArrayBase, LogicalAnd, operator&&)
BINARY_DECLARATION(ArrayBase, LogicalOr, operator||)
BINARY_DECLARATION(ArrayBase, BitAnd, operator&)
BINARY_DECLARATION(ArrayBase, BitOr, operator|)
BINARY_DECLARATION(ArrayBase, BitXor, operator^)

}  // namespace uvl

#endif
