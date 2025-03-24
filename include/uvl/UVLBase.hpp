#if !defined(UVL_UVLBASE_HPP)
#define UVL_UVLBASE_HPP

#include "concepts/UVLBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
#include "uvl/detail/declare_operations.hpp"
#include "uvl/types.hpp"
#include "uvl/views/binary/ArithmeticViews.hpp"
#include "uvl/views/unary/ScalarArithmeticViews.hpp"
#include "views/reductions/CoefficientSum.hpp"
#include "views/unary/AbsView.hpp"
#include "views/unary/CastView.hpp"
#include "views/unary/DiagonalView.hpp"
#include "views/unary/ScalarPowerView.hpp"
#include "views/unary/SliceView.hpp"
#include "views/unary/SwizzleView.hpp"
#include "views/unary/detail/operation_implementations.hpp"

namespace uvl {

template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class UVLBase {
   public:
    UVLBase() = default;
    using Derived = DerivedT<View>;
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    Derived& derived() { return static_cast<Derived&>(*this); }

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    template <index_type... N>
    auto eval(const extents<N...>&) const {
        return UVL<value_type, N...>(*this);
    }
    auto eval() const { return eval(extents()); }
    template <typename... Args>
    UVLBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    UVLBase(View&& v) : m_view(v) {}
    UVLBase(const View& v) : m_view(v) {}
    UVLBase& operator=(concepts::ViewDerived auto const& v) {
        m_view = v;
        return *this;
    }
    UVLBase& operator=(concepts::UVLBaseDerived auto const& v) {
        m_view = v.view();
        return *this;
    }

    UVLBase(UVLBase&& v) = default;
    UVLBase(const UVLBase& v) = default;

    template <concepts::ViewDerived Other>
    UVLBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(extents_traits::convert_from(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::ViewDerived Other>
    UVLBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other);
        return *this;
    }
    template <concepts::UVLBaseDerived Other>
    UVLBase(const Other& other)
        requires(view_type::is_writable)
        : UVLBase(other.view()) {}
    template <concepts::UVLBaseDerived Other>
    UVLBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    UVLBase& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    UVLBase& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }
    template <concepts::UVLBaseDerived Other>
    UVLBase& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this + other;
    }
    template <concepts::UVLBaseDerived Other>
    UVLBase& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }

    template <rank_type... ranks>
    auto swizzle() const {
        return DerivedT<views::unary::SwizzleView<value_type, ranks...>>(
            view());
    }
    template <typename T>
    auto cast() const {
        return DerivedT<views::unary::CastView<T, view_type>>(view());
    }

    //--------------------------------------------------------------
    //

    auto pow(value_type const& exp) const {
        return DerivedT<views::unary::ScalarPowerView<view_type, value_type>>(
            view(), exp);
    }

    auto abs() const {
        return DerivedT<views::unary::AbsView<view_type>>(view());
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

        return DerivedT<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;
        return DerivedT<view_type>(view_type(view(), Slices{}...));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return DerivedT<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return DerivedT<view_type>(view_type(view(), Slices{}...));
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

   private:
    View m_view;
};

UNARY_DECLARATION(UVLBase, LogicalNot, operator!)
UNARY_DECLARATION(UVLBase, BitNot, operator~)
UNARY_DECLARATION(UVLBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(UVLBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(UVLBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(UVLBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(UVLBase, Divides, operator/)
// SCALAR_BINARY_DECLARATION(UVLBase, Modulus, operator%)
// SCALAR_BINARY_DECLARATION(UVLBase, EqualsTo, operator==)
// SCALAR_BINARY_DECLARATION(UVLBase, NotEqualsTo, operator!=)
// SCALAR_BINARY_DECLARATION(UVLBase, Greater, operator>)
// SCALAR_BINARY_DECLARATION(UVLBase, Less, operator<)
// SCALAR_BINARY_DECLARATION(UVLBase, GreaterEqual, operator>=)
// SCALAR_BINARY_DECLARATION(UVLBase, LessEqual, operator<=)
// SCALAR_BINARY_DECLARATION(UVLBase, LogicalAnd, operator&&)
// SCALAR_BINARY_DECLARATION(UVLBase, LogicalOr, operator||)
// SCALAR_BINARY_DECLARATION(UVLBase, BitAnd, operator&)
// SCALAR_BINARY_DECLARATION(UVLBase, BitOr, operator|)
// SCALAR_BINARY_DECLARATION(UVLBase, BitXor, operator^)

BINARY_DECLARATION(UVLBase, Plus, operator+)
BINARY_DECLARATION(UVLBase, Minus, operator-)
// BINARY_DECLARATION(UVLBase, Divides, operator/)
// BINARY_DECLARATION(UVLBase, Modulus, operator%)
// BINARY_DECLARATION(UVLBase, EqualsTo, operator==)
// BINARY_DECLARATION(UVLBase, NotEqualsTo, operator!=)
// BINARY_DECLARATION(UVLBase, Greater, operator>)
// BINARY_DECLARATION(UVLBase, Less, operator<)
// BINARY_DECLARATION(UVLBase, GreaterEqual, operator>=)
// BINARY_DECLARATION(UVLBase, LessEqual, operator<=)
// BINARY_DECLARATION(UVLBase, LogicalAnd, operator&&)
// BINARY_DECLARATION(UVLBase, LogicalOr, operator||)
// BINARY_DECLARATION(UVLBase, BitAnd, operator&)
// BINARY_DECLARATION(UVLBase, BitOr, operator|)
// BINARY_DECLARATION(UVLBase, BitXor, operator^)

template <concepts::UVLBaseDerived View1, concepts::UVLBaseDerived View2>
bool operator==(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::UVLBaseDerived View1, concepts::UVLBaseDerived View2>
bool operator!=(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() != rhs.as_array()).any();
}
}  // namespace uvl

#endif
