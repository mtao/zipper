#if !defined(UVL_TENSORBASE_HPP)
#define UVL_TENSORBASE_HPP

#include "concepts/TensorBaseDerived.hpp"
#include "concepts/ViewDerived.hpp"
#include "uvl/detail/declare_operations.hpp"
#include "uvl/types.hpp"
#include "uvl/views/binary/ArithmeticViews.hpp"
#include "uvl/views/binary/TensorProductView.hpp"
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

template <concepts::ViewDerived View>
class TensorBase {
   public:
    TensorBase() = default;

    using view_type = View;
    using value_type = View::value_type;
    using extents_type = View::extents_type;
    using extents_traits = detail::ExtentsTraits<extents_type>;

    template <index_type... N>
    auto eval(const extents<N...>&) const {
        return Tensor<value_type, N...>(*this);
    }
    auto eval() const { return eval(extents()); }
    template <typename... Args>
    TensorBase(Args&&... v) : m_view(std::forward<Args>(v)...) {}

    TensorBase(View&& v) : m_view(v) {}
    TensorBase(const View& v) : m_view(v) {}
    TensorBase& operator=(concepts::ViewDerived auto const& v) {
        m_view = v;
        return *this;
    }
    TensorBase& operator=(concepts::TensorBaseDerived auto const& v) {
        m_view = v.view();
        return *this;
    }

    TensorBase(TensorBase&& v) = default;
    TensorBase(const TensorBase& v) = default;

    template <concepts::ViewDerived Other>
    TensorBase(const Other& other)
        requires(view_type::is_writable)
        : m_view(extents_traits::convert_from(other.extents())) {
        m_view.assign(other);
    }
    template <concepts::ViewDerived Other>
    TensorBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        m_view.assign(other);
        return *this;
    }
    template <concepts::TensorBaseDerived Other>
    TensorBase(const Other& other)
        requires(view_type::is_writable)
        : TensorBase(other.view()) {}
    template <concepts::TensorBaseDerived Other>
    TensorBase& operator=(const Other& other)
        requires(view_type::is_writable)
    {
        return operator=(other.view());
    }

    TensorBase& operator*=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = other * *this;
    }
    TensorBase& operator/=(const value_type& other)
        requires(view_type::is_writable)
    {
        return *this = *this / other;
    }
    template <concepts::TensorBaseDerived Other>
    TensorBase& operator+=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this + other;
    }
    template <concepts::TensorBaseDerived Other>
    TensorBase& operator-=(const Other& other)
        requires(view_type::is_writable)
    {
        return *this = *this - other;
    }

    template <rank_type... ranks>
    auto swizzle() const {
        return TensorBase<views::unary::SwizzleView<value_type, ranks...>>(
            view());
    }
    template <typename T>
    auto cast() const {
        return TensorBase<views::unary::CastView<T, view_type>>(view());
    }

    //--------------------------------------------------------------
    //

    auto pow(value_type const& exp) const {
        return TensorBase<views::unary::ScalarPowerView<view_type, value_type>>(
            view(), exp);
    }

    auto abs() const {
        return TensorBase<views::unary::AbsView<view_type>>(view());
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

        return TensorBase<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() const {
        using view_type = views::unary::SliceView<view_type, true, Slices...>;
        return TensorBase<view_type>(view_type(view(), Slices{}...));
    }

    template <typename... Slices>
    auto slice(Slices&&... slices) {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return TensorBase<view_type>(
            view_type(view(), std::forward<Slices>(slices)...));
    }
    template <typename... Slices>
    auto slice() {
        using view_type = views::unary::SliceView<view_type, false, Slices...>;
        return TensorBase<view_type>(view_type(view(), Slices{}...));
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

template <concepts::ViewDerived View>
TensorBase(View&& view) -> TensorBase<View>;
template <concepts::ViewDerived View>
TensorBase(const View& view) -> TensorBase<View>;

UNARY_DECLARATION(TensorBase, LogicalNot, operator!)
UNARY_DECLARATION(TensorBase, BitNot, operator~)
UNARY_DECLARATION(TensorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(TensorBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(TensorBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(TensorBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(TensorBase, Divides, operator/)
// SCALAR_BINARY_DECLARATION(TensorBase, Modulus, operator%)
// SCALAR_BINARY_DECLARATION(TensorBase, EqualsTo, operator==)
// SCALAR_BINARY_DECLARATION(TensorBase, NotEqualsTo, operator!=)
// SCALAR_BINARY_DECLARATION(TensorBase, Greater, operator>)
// SCALAR_BINARY_DECLARATION(TensorBase, Less, operator<)
// SCALAR_BINARY_DECLARATION(TensorBase, GreaterEqual, operator>=)
// SCALAR_BINARY_DECLARATION(TensorBase, LessEqual, operator<=)
// SCALAR_BINARY_DECLARATION(TensorBase, LogicalAnd, operator&&)
// SCALAR_BINARY_DECLARATION(TensorBase, LogicalOr, operator||)
// SCALAR_BINARY_DECLARATION(TensorBase, BitAnd, operator&)
// SCALAR_BINARY_DECLARATION(TensorBase, BitOr, operator|)
// SCALAR_BINARY_DECLARATION(TensorBase, BitXor, operator^)

BINARY_DECLARATION(TensorBase, Plus, operator+)
BINARY_DECLARATION(TensorBase, Minus, operator-)
// BINARY_DECLARATION(TensorBase, Divides, operator/)
// BINARY_DECLARATION(TensorBase, Modulus, operator%)
// BINARY_DECLARATION(TensorBase, EqualsTo, operator==)
// BINARY_DECLARATION(TensorBase, NotEqualsTo, operator!=)
// BINARY_DECLARATION(TensorBase, Greater, operator>)
// BINARY_DECLARATION(TensorBase, Less, operator<)
// BINARY_DECLARATION(TensorBase, GreaterEqual, operator>=)
// BINARY_DECLARATION(TensorBase, LessEqual, operator<=)
// BINARY_DECLARATION(TensorBase, LogicalAnd, operator&&)
// BINARY_DECLARATION(TensorBase, LogicalOr, operator||)
// BINARY_DECLARATION(TensorBase, BitAnd, operator&)
// BINARY_DECLARATION(TensorBase, BitOr, operator|)
// BINARY_DECLARATION(TensorBase, BitXor, operator^)

template <concepts::TensorBaseDerived View1, concepts::TensorBaseDerived View2>
bool operator==(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::TensorBaseDerived View1, concepts::TensorBaseDerived View2>
bool operator!=(View1 const& lhs, View2 const& rhs) {
    return (lhs.as_array() != rhs.as_array()).any();
}

template <concepts::TensorBaseDerived View1, concepts::TensorBaseDerived View2>
auto operator*(View1 const& lhs, View2 const& rhs) {
    return TensorBase<views::binary::TensorProductView<
        typename View1::view_type, typename View2::view_type>>(lhs.view(),
                                                               rhs.view());
}
}  // namespace uvl

#endif
